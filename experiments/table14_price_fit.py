"""Table 14, MATH row. Coverage predicted from a 100-answer price against coverage realised.

Mean absolute error in points of the three single-check thresholds of the 8B
probe, the 8B vote and the 1.7B vote (three draws each), and predicted against
realised coverage of the probe-first and small-first cascades that race
(2, 2) on the band, over the 20 pricing draws of the diagnostic grid.
"""

import argparse

import numpy as np

import diagnostic_grid as grid
from common import save

CHECKS = {"8B probe": "probe 8B m>={t}", "8B vote": "vote 8B {t} of 3",
          "1.7B vote": "vote 1.7B {t} of 3"}
CASCADES = {"Cascade": "cascade 8B probe m3 abstain<=0 race (2, 2)",
            "Small-first": "cascade 1.7B votes race (2, 2)"}


def compute(pairs=None):
    pairs = pairs or grid.run()
    by_name = {s["name"]: records for s, records in zip(grid.SCHEDULES, pairs)}
    errors, out = {}, {"mae": {}, "cascades": {}}
    for label, pattern in CHECKS.items():
        errors[label] = [abs(p["cov"] - a["cov"]) for t in (1, 2, 3)
                         for p, a in by_name[pattern.format(t=t)]]
        out["mae"][label] = 100 * float(np.mean(errors[label]))
    out["mae"]["All"] = 100 * float(np.mean(sum(errors.values(), [])))
    for label, name in CASCADES.items():
        records = by_name[name]
        out["cascades"][label] = {"pred": 100 * float(np.mean([p["cov"] for p, _ in records])),
                                  "real": 100 * float(np.mean([a["cov"] for _, a in records]))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/table14.json")
    args = parser.parse_args()
    out = compute()
    print("MATH, n = 2326, mean absolute error (points)")
    print("   ".join(f"{label} {value:.2f}" for label, value in out["mae"].items()))
    for label, row in out["cascades"].items():
        print(f"{label:<12} pred. {row['pred']:.1f}   real. {row['real']:.1f}")
    save(out, args.out)


if __name__ == "__main__":
    main()
