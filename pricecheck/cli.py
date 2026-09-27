"""The `pricecheck` command. `python -m pricecheck` runs the same thing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import __version__
from .bank import DEFAULT_BANK, load_bank
from .certification import ALPHAS, evaluate, run_main, select


def probability(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError("expected a finite number strictly between 0 and 1")
    return value


def coverage(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("expected a finite number from 0 to 1")
    return value


def _demo(args, parser):
    bank = load_bank()
    cal = [row for row in bank if row["source_id"] != args.fold]
    test = [row for row in bank if row["source_id"] == args.fold]
    selection = select(cal, alpha=args.alpha)
    print(selection)
    wrong = sum(1 - row["correct"] for row in test) / len(test)
    print(f"held out  serve every answer    coverage 100.0%  risk {wrong:.2%}")
    if selection.max_coverage is not None:
        served = evaluate(test, selection.max_coverage)
        print(f"held out  PriceCheck            coverage {served['coverage']:.1%}  "
              f"risk {served['risk']:.2%}")
    return 0


def _certify(args, parser):
    if args.output.resolve() == args.data.resolve():
        parser.error("output must differ from the input bank")
    try:
        rows = load_bank(args.data)
        result = run_main(rows, args.alphas, args.delta, args.min_coverage)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("target  selected  coverage  mean risk  violations  mean tokens")
    for item in result["summary"]:
        if item["selector"] != "max_coverage":
            continue
        risk = item["risk_selected_splits"]
        tokens = item["tokens_selected_splits"]
        risk_text = "n/a" if risk is None else f"{100 * risk:.3f}%"
        token_text = "n/a" if tokens is None else f"{tokens:.1f}"
        print(f"{100 * item['alpha']:5.2f}%  "
              f"{item['splits_selected']:2d}/{item['splits_total']:<2d}     "
              f"{100 * item['coverage_all_splits']:6.2f}%   {risk_text:>7}  "
              f"{item['risk_violations']:5d}       {token_text}")
    print(f"Saved {args.output}")
    return 0


def _price(args, parser):
    try:
        rows = load_bank(args.data)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    right = [row for row in rows if row["correct"]]
    wrong = [row for row in rows if not row["correct"]]
    if not right or not wrong:
        parser.error("pricing needs both correct and wrong answers")

    def rate(group, field, draws):
        # Agreement per draw: probe counts are out of three, votes are 0/1 lists.
        if draws is None:
            return sum(sum(row[field]) / len(row[field]) for row in group) / len(group)
        return sum(row[field] for row in group) / (draws * len(group))

    print(f"{len(rows)} answers, {len(wrong)} wrong")
    print("action                  completeness p1  leak p0")
    for label, field, draws in (("Backward probe, 8B", "m_8b", 3), ("Backward probe, 4B", "m_4b", 3),
                                ("Backward probe, 1.7B", "m_1p7b", 3),
                                ("Re-solve vote, 8B", "votes_8b", None),
                                ("Re-solve vote, 1.7B", "votes_1p7b", None)):
        print(f"{label:<24}{100 * rate(right, field, draws):14.1f}%  "
              f"{100 * rate(wrong, field, draws):6.1f}%")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pricecheck",
        description="Risk-controlled selective answering by pricing label-free checks")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the CPU quickstart on the shipped bank")
    demo.add_argument("--alpha", type=probability, default=0.015, help="selective-risk target")
    demo.add_argument("--fold", type=int, choices=range(5), default=0,
                      help="generator held out for testing (0 is F0)")
    demo.set_defaults(func=_demo)

    certify = sub.add_parser("certify", help="replay the 15-split main experiment")
    certify.add_argument("--data", type=Path, default=DEFAULT_BANK,
                         help="JSONL bank of check outcomes")
    certify.add_argument("--output", type=Path, default=Path("results/main_results.json"))
    certify.add_argument("--alphas", nargs="+", type=probability, default=ALPHAS)
    certify.add_argument("--delta", type=probability, default=0.05)
    certify.add_argument("--min-coverage", type=coverage, default=0.60)
    certify.set_defaults(func=_certify)

    price = sub.add_parser("price", help="agreement rates of each check on a bank")
    price.add_argument("--data", type=Path, default=DEFAULT_BANK,
                       help="JSONL bank of check outcomes")
    price.set_defaults(func=_price)

    args = parser.parse_args(argv)
    return args.func(args, parser)
