"""Table 8, the modal selection by split scheme, target and selector.

Each row gives the configuration selected most often on that scheme (ties go
to the one met first), how many splits select it, and its held-out coverage,
risk and tokens averaged over those splits. The script also prints, for each
target, on how many of the 15 splits every configuration passes the test.
"""

import argparse
import statistics
from collections import Counter

import pricecheck
from common import LABELS, main_result, save

SCHEMES = (("source_held_out", "Source-held-out folds"),
           ("question_disjoint", "Question-disjoint halves"))
SELECTORS = (("max_coverage", "Max. coverage"), ("min_cost", "Min. cost"))


def compute(result=None):
    result = result or main_result()
    rows = []
    for scheme, _ in SCHEMES:
        for alpha in result["protocol"]["alphas"]:
            records = [r for r in result["splits"] if r["scheme"] == scheme and r["alpha"] == alpha]
            for selector, _ in SELECTORS:
                picks = [r["selections"][selector] for r in records if r["selections"][selector]]
                if not picks:
                    continue
                name = Counter(p["policy"] for p in picks).most_common(1)[0][0]
                held = [p["test"] for p in picks if p["policy"] == name]
                rows.append({
                    "scheme": scheme, "alpha": alpha, "selector": selector, "policy": name,
                    "splits": len(held), "of": len(records),
                    "coverage": 100 * statistics.mean(t["coverage"] for t in held),
                    "risk": 100 * statistics.mean(t["risk"] for t in held),
                    "tokens": statistics.mean(t["tokens"] for t in held),
                    "weighted": statistics.mean(t["weighted_tokens"] for t in held),
                })
    counts = {}
    for alpha in result["protocol"]["alphas"]:
        records = [r for r in result["splits"] if r["alpha"] == alpha]
        counts[alpha] = {p.name: sum(p.name in r["certified"] for r in records)
                         for p in pricecheck.GRID}
    return {"rows": rows, "certified_splits": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/table8.json")
    args = parser.parse_args()
    out = compute()
    selector_names = dict(SELECTORS)
    for scheme, title in SCHEMES:
        print(title)
        print(f"{'target':>6}  {'selector':<13} {'modal configuration':<37} splits  "
              f"cov.   risk  generated  weighted")
        for row in out["rows"]:
            if row["scheme"] != scheme:
                continue
            print(f"{100 * row['alpha']:5.2f}%  {selector_names[row['selector']]:<13} "
                  f"{LABELS[row['policy']]:<37} {row['splits']:2d}/{row['of']:<2d}  "
                  f"{row['coverage']:4.1f}  {row['risk']:5.2f}  {row['tokens']:9.0f}  "
                  f"{row['weighted']:8.0f}")
        print()
    print("splits certifying, of 15, per configuration and target")
    alphas = list(out["certified_splits"])
    print(f"{'configuration':<40}" + "".join(f"{100 * a:7.2f}%" for a in alphas))
    for policy in pricecheck.GRID:
        print(f"{policy.label:<40}"
              + "".join(f"{out['certified_splits'][a][policy.name]:8d}" for a in alphas))
    save(out, args.out)


if __name__ == "__main__":
    main()
