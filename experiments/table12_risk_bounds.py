"""Table 12, answer-level and problem-count upper bounds on selective risk.

For each policy at its own operating point on the whole bank, the kept answers,
wrong answers and distinct problems, the exact one-sided 95% Clopper-Pearson
bound, and the problem-count bound. The problem-count bound divides both counts
by the kept answers per kept problem, which reads no label, and evaluates the
same bound at the deflated counts.
"""

import argparse

from scipy.stats import beta

import pricecheck
from common import save
from table19_operating_points import POLICIES, operating_point, probe

RULES = {label: rule for _, label, rule in POLICIES}
RULES["Backward probe m>=2, 1.7B"] = probe("m_1p7b", 2)
ROWS = ("Backward probe m>=2, 1.7B", "Backward probe m>=2, 8B", "Unanimous 3-vote, 8B",
        "Single re-solve, 8B", "Single re-solve, 1.7B", "Adaptive vote, 8B",
        "Small-first cascade")


def upper(errors, kept, level=0.95):
    return 1.0 if errors >= kept else float(beta.ppf(level, errors + 1, kept - errors))


def compute(bank=None):
    bank = bank or pricecheck.load_bank()
    rows = []
    for name in ROWS:
        point = operating_point(bank, RULES[name])
        n, e, p = point["kept"], point["wrong"], point["problems"]
        design = n / p
        rows.append({"policy": name, "kept": n, "wrong": e, "problems": p,
                     "answer_level": 100 * upper(e, n),
                     "problem_count": 100 * upper(e / design, n / design)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/table12.json")
    args = parser.parse_args()
    rows = compute()
    print(f"{'policy':<26} answers  wrong  problems  answer level (%)  problem count (%)")
    for row in rows:
        print(f"{row['policy']:<26} {row['kept']:7d}  {row['wrong']:5d}  {row['problems']:8d}  "
              f"{row['answer_level']:16.3f}  {row['problem_count']:17.3f}")
    save(rows, args.out)


if __name__ == "__main__":
    main()
