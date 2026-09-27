"""Table 9, the schedule selected on each of the 15 splits at the 1.5% target.

For every split, the maximum-coverage selection, how many of the 23 schedules
pass the test there, the held-out coverage, risk and weighted tokens, and the
calibration and test sizes.
"""

import argparse

from common import LABELS, main_result, save, split_name


def compute(result=None, alpha=0.015):
    result = result or main_result(alphas=(alpha,))
    rows = []
    for record in result["splits"]:
        if record["alpha"] != alpha:
            continue
        pick = record["selections"]["max_coverage"]
        rows.append({
            "split": split_name(record), "scheme": record["scheme"],
            "policy": pick["policy"] if pick else None,
            "certified": len(record["certified"]),
            "coverage": 100 * pick["test"]["coverage"] if pick else 0.0,
            "risk": 100 * pick["test"]["risk"] if pick else None,
            "weighted": pick["test"]["weighted_tokens"] if pick else None,
            "calibration": record["calibration_rows"], "test": record["test_rows"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/table9.json")
    args = parser.parse_args()
    rows = compute()
    print(f"{'split':<12} {'selected configuration':<22} certified  cov.   risk  "
          f"weighted tok.  calibration  test")
    for row in rows:
        print(f"{row['split']:<12} {LABELS[row['policy']]:<22} {row['certified']:9d}  "
              f"{row['coverage']:4.1f}  {row['risk']:5.2f}  {row['weighted']:13.0f}  "
              f"{row['calibration']:11d}  {row['test']:4d}")
    save(rows, args.out)


if __name__ == "__main__":
    main()
