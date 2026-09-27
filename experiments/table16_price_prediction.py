"""Table 16, how well prices predict the 118 diagnostic schedules.

Mean absolute coverage and risk errors, the risk bias and the rank correlation
between predicted and realised coverage, over the 70 cascades and over all 118
schedules, from 20 draws of a 100-answer pricing set. Also the drop in coverage
error per extra action in a chain and the cost biases of the cascades.
"""

import argparse

import diagnostic_grid as grid
from common import save


def compute(pairs=None):
    rows = grid.per_schedule(pairs or grid.run())
    return {"summary": grid.summary(rows), "schedules": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/table16.json")
    args = parser.parse_args()
    out = compute()
    s = out["summary"]
    print(f"{'quantity':<30} {'70 cascades':>11}  {'118 schedules':>13}")
    for label, key, fmt in (("Coverage error (points)", "cov_mae", ".2f"),
                            ("Coverage rank correlation", "cov_spearman", ".2f"),
                            ("Risk error (points)", "risk_mae", ".2f"),
                            ("Risk bias (points)", "risk_bias", ".2f")):
        print(f"{label:<30} {s['cascades'][key]:>11{fmt}}  {s['all'][key]:>13{fmt}}")
    print()
    print(f"coverage error per extra action in a chain   {s['cov_mae_slope_per_action']:+.2f} points")
    print(f"cascade cost bias, generated tokens           {s['cascades']['tok_bias']:+.1f}%")
    print(f"cascade cost bias, weighted tokens            {s['cascades']['flop_bias']:+.1f}%")
    save(out, args.out)


if __name__ == "__main__":
    main()
