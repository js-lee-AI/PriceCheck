"""Table 1, the price of each check on the whole bank.

Completeness p1 and leak p0 are agreement rates per draw on correct and on
wrong answers. A backward probe records how many of its three recoveries
agree, and a re-solve vote records one verdict per stored solve. The unit cost
of a re-solve is the mean stored solve length, in generated tokens and in
tokens weighted by parameter count (0.21 for 1.7B).
"""

import argparse
import statistics

import pricecheck
from common import save

CHECKS = (("Backward probe, 8B", "m_8b"), ("Backward probe, 4B", "m_4b"),
          ("Backward probe, 1.7B", "m_1p7b"), ("Re-solve vote, 8B", "votes_8b"),
          ("Re-solve vote, 1.7B", "votes_1p7b"))


def per_draw(row, field):
    value = row[field]
    return sum(value) / len(value) if isinstance(value, list) else value / 3


def compute(bank=None):
    bank = bank or pricecheck.load_bank()
    rows = []
    for label, field in CHECKS:
        rows.append({
            "check": label,
            "p1": 100 * statistics.mean(per_draw(r, field) for r in bank if r["correct"]),
            "p0": 100 * statistics.mean(per_draw(r, field) for r in bank if not r["correct"]),
        })
    costs = {}
    for label, field, weight in (("Re-solve vote, 8B", "solve_tok_8b", 1.0),
                                 ("Re-solve vote, 1.7B", "solve_tok_1p7b", pricecheck.SMALL_WEIGHT)):
        generated = statistics.mean(r[field] for r in bank)
        costs[label] = {"generated": generated, "weighted": generated * weight}
    return {"rates": rows, "costs": costs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/table1.json")
    args = parser.parse_args()
    out = compute()
    print(f"{'check':<22} completeness p1 (%)  leak p0 (%)")
    for row in out["rates"]:
        print(f"{row['check']:<22} {row['p1']:19.1f}  {row['p0']:11.1f}")
    print()
    print(f"{'cost per draw':<22} generated  weighted")
    for label, cost in out["costs"].items():
        print(f"{label:<22} {cost['generated']:9.0f}  {cost['weighted']:8.0f}")
    save(out, args.out)


if __name__ == "__main__":
    main()
