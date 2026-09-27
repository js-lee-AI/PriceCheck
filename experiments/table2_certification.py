"""PriceCheck row of Table 2, plus the matching rows of Tables 3a and 10.

At each target, splits certifying of 15 and held-out coverage averaged over
all 15 splits (a split that certifies nothing counts as zero), wrong answers
kept at the 1.5% target, and coverage by split scheme.
"""

import argparse
import statistics

from common import main_result, save


def compute(result=None):
    result = result or main_result()
    table = []
    for item in result["summary"]:
        if item["selector"] == "max_coverage":
            table.append({"alpha": item["alpha"], "splits": item["splits_selected"],
                          "coverage": 100 * item["coverage_all_splits"],
                          "tokens": item["tokens_selected_splits"]})
    primary = [r for r in result["splits"] if r["alpha"] == 0.015]
    tests = [r["selections"]["max_coverage"]["test"] for r in primary]
    by_scheme = {}
    for scheme in ("question_disjoint", "source_held_out"):
        picked = [r["selections"]["max_coverage"]["test"]["coverage"]
                  for r in primary if r["scheme"] == scheme]
        by_scheme[scheme] = 100 * sum(picked) / len(picked)
    return {
        "table": table,
        "wrong_kept": sum(t["errors"] for t in tests),
        "coverage_sd": 100 * statistics.pstdev(t["coverage"] for t in tests),
        "by_scheme": by_scheme,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/table2.json")
    args = parser.parse_args()
    out = compute()
    print("Table 2, PriceCheck (ours)")
    print("target   Spl.   Cov.")
    for row in out["table"]:
        print(f"{100 * row['alpha']:5.2f}%  {row['splits']:4d}  {row['coverage']:6.2f}")
    print(f"wrong answers kept at 1.5%, summed over 15 splits   {out['wrong_kept']}")
    print(f"standard deviation of coverage over 15 splits      {out['coverage_sd']:.1f}")
    print()
    print("Table 3a, PriceCheck, 23 rules")
    print(f"All {out['table'][4]['coverage']:.2f}   QD {out['by_scheme']['question_disjoint']:.2f}"
          f"   SHO {out['by_scheme']['source_held_out']:.2f}")
    print()
    print("Table 10, PriceCheck (ours)")
    splits = {row["alpha"]: row["splits"] for row in out["table"]}
    print(f"tokens {out['table'][4]['tokens']:.0f}   splits at 0.5% {splits[0.005]}, 1.0% "
          f"{splits[0.01]}, 1.5% {splits[0.015]}   Cov. {out['table'][4]['coverage']:.2f}"
          f"   wrong {out['wrong_kept']}")
    save(out, args.out)


if __name__ == "__main__":
    main()
