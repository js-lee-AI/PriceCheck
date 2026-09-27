import subprocess
import sys
from pathlib import Path

import pytest

import pricecheck

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          check=True, timeout=120).stdout


def test_import_stays_light():
    # A fresh interpreter, since pytest plugins may have imported heavy modules already.
    out = run("-c", "import sys, pricecheck; print('torch' in sys.modules)")
    assert out.strip() == "False"


def test_prices_on_hand_checked_cases():
    assert pricecheck.race_price(2, 2, 0.5) == pytest.approx((0.5, 2.5))
    assert pricecheck.count_tail(3, 2, 0.4) == pytest.approx(3 * 0.4 ** 2 * 0.6 + 0.4 ** 3)
    coverage, risk = pricecheck.coverage_risk(0.9, 0.8, 0.1)
    assert (coverage, risk) == pytest.approx((0.73, 0.01 / 0.73))


def test_quickstart_prints_what_the_readme_shows():
    out = run(str(ROOT / "examples" / "quickstart.py"))
    assert "19 of 23 schedules pass p <= delta/23" in out
    assert "held out  PriceCheck            coverage 79.0%  risk 0.27%" in out


def test_pricing_example_prints_what_the_readme_shows():
    out = run(str(ROOT / "examples" / "predict_a_cascade.py"))
    assert out.splitlines() == ["predicted  coverage 73.8%  risk 0.32%",
                                "actual     coverage 71.0%  risk 0.42%"]


def test_demo_is_the_quickstart():
    assert run("-m", "pricecheck", "demo") == run(str(ROOT / "examples" / "quickstart.py"))


def test_cli_help_and_price():
    assert "certify" in run("-m", "pricecheck", "--help")
    out = run("-m", "pricecheck", "price")
    assert "Backward probe, 8B                70.1%    10.8%" in out


def test_select_matches_the_main_replay():
    bank = pricecheck.load_bank()
    result = pricecheck.run_main(bank, alphas=(0.0075, 0.015))
    splits = {split.name: split for split in pricecheck.make_splits(bank)}
    for record in result["splits"]:
        cal = [bank[i] for i in splits[record["split"]].calibration]
        choice = pricecheck.select(cal, alpha=record["alpha"])
        assert [p.name for p in choice.certified] == record["certified"]
        for selector in ("max_coverage", "min_cost"):
            picked = getattr(choice, selector)
            expected = record["selections"][selector]
            assert (picked and picked.name) == (expected and expected["policy"])


def test_decide_serves_a_new_answer_without_its_label():
    row = dict(pricecheck.load_bank()[0])
    del row["correct"]
    accept, tokens, weighted = pricecheck.decide(row, pricecheck.Policy("asc", 2, 3))
    assert accept is True and tokens == weighted == 2 * row["solve_tok_8b"]
