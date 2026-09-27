"""Table 19, the operating point of single checks and schedules on the whole bank.

Coverage, wrong answers kept and selective risk on all 2326 answers, and the
cost per answer averaged over every answer, served or not. The small-first
cascade of this table races to three agreements before two refusals, a
variant outside the deployed family.
"""

import argparse

import pricecheck
from common import save

SMALL = pricecheck.SMALL_WEIGHT


def resolve(field, n, weight):
    def rule(row):
        tokens = n * row[field.replace("votes", "solve_tok")]
        return sum(row[field][:n]) >= n, tokens, tokens * weight
    return rule


def small_first(k, j):
    # Three 1.7B votes first, then the 8B race on the undecided band.
    def rule(row):
        count = sum(row["votes_1p7b"][:3])
        tokens = 3 * row["solve_tok_1p7b"]
        if count in (0, 3):
            return count == 3, tokens, tokens * SMALL
        accept, used = pricecheck.race(row["votes_8b"], k, j)
        extra = used * row["solve_tok_8b"]
        return accept, tokens + extra, tokens * SMALL + extra
    return rule


def probe(field, threshold):
    return lambda row: (row[field] >= threshold, None, None)


POLICIES = (
    ("Single checks", "Backward probe m>=1, 8B", probe("m_8b", 1)),
    ("Single checks", "Backward probe m>=2, 8B", probe("m_8b", 2)),
    ("Single checks", "Backward probe m>=2, 4B", probe("m_4b", 2)),
    ("Single checks", "Single re-solve, 1.7B", resolve("votes_1p7b", 1, SMALL)),
    ("Single checks", "Single re-solve, 8B", resolve("votes_8b", 1, 1.0)),
    ("Schedules", "Small-first cascade", small_first(3, 2)),
    ("Schedules", "Adaptive vote, 8B", lambda row: pricecheck.decide(row, pricecheck.Policy("asc", 2, 3))),
    ("Schedules", "Unanimous 3-vote, 8B", lambda row: pricecheck.decide(row, pricecheck.Policy("sc", 3))),
    ("Schedules", "Unanimous 4-vote, 8B", lambda row: pricecheck.decide(row, pricecheck.Policy("sc", 4))),
)


def operating_point(bank, rule):
    outcomes = [rule(row) for row in bank]
    kept = [row for row, (accept, _, _) in zip(bank, outcomes) if accept]
    wrong = sum(1 - row["correct"] for row in kept)
    priced = outcomes[0][1] is not None
    return {
        "coverage": 100 * len(kept) / len(bank), "wrong": wrong,
        "risk": 100 * wrong / len(kept), "kept": len(kept),
        "problems": len({row["problem_id"] for row in kept}),
        "generated": sum(t for _, t, _ in outcomes) / len(bank) if priced else None,
        "weighted": sum(w for _, _, w in outcomes) / len(bank) if priced else None,
    }


def compute(bank=None):
    bank = bank or pricecheck.load_bank()
    return [{"block": block, "policy": label, **operating_point(bank, rule)}
            for block, label, rule in POLICIES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/table19.json")
    args = parser.parse_args()
    rows = compute()
    print(f"{'policy':<26} cov. (%)  wrong  risk (%)  generated  weighted")
    block = None
    for row in rows:
        if row["block"] != block:
            block = row["block"]
            print(block)
        cost = ("        -         -" if row["generated"] is None
                else f"{row['generated']:9.0f}  {row['weighted']:8.0f}")
        print(f"{row['policy']:<26} {row['coverage']:8.1f}  {row['wrong']:5d}  "
              f"{row['risk']:8.2f}  {cost}")
    print("- the table prices a probe from its trace length, which the bank does not store")
    save(rows, args.out)


if __name__ == "__main__":
    main()
