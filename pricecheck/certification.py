"""The calibration test, the two selectors and the 15 splits of the main experiment."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from .policies import GRID, decide

ALPHAS = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02)


@dataclass(frozen=True)
class Split:
    name: str
    scheme: str
    calibration: tuple
    test: tuple


def make_splits(rows):
    splits = []
    for source in sorted({row["source_id"] for row in rows}):
        cal = tuple(i for i, row in enumerate(rows) if row["source_id"] != source)
        test = tuple(i for i, row in enumerate(rows) if row["source_id"] == source)
        splits.append(Split(f"source_{source}", "source_held_out", cal, test))
    problems = sorted({row["problem_id"] for row in rows})
    for seed in range(10):
        perm = list(problems)
        np.random.default_rng(seed).shuffle(perm)
        calibration_problems = set(perm[:len(perm) // 2])
        cal = tuple(i for i, row in enumerate(rows)
                    if row["problem_id"] in calibration_problems)
        test = tuple(i for i, row in enumerate(rows)
                     if row["problem_id"] not in calibration_problems)
        splits.append(Split(f"question_{seed}", "question_disjoint", cal, test))
    if any(not split.calibration or not split.test for split in splits):
        raise ValueError("every split must have calibration and test rows")
    return splits


def evaluate(rows, policy):
    if not rows:
        raise ValueError("cannot evaluate an empty row set")
    accepted = errors = 0
    tokens = weighted = 0.0
    for row in rows:
        accept, cost, weighted_cost = decide(row, policy)
        tokens += cost
        weighted += weighted_cost
        if accept:
            accepted += 1
            errors += 1 - row["correct"]
    return {
        "accepted": accepted,
        "errors": errors,
        "coverage": accepted / len(rows),
        "risk": errors / accepted if accepted else None,
        "tokens": tokens / len(rows),
        "weighted_tokens": weighted / len(rows),
    }


def p_value(errors, accepted, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not (type(errors) is int and type(accepted) is int
            and 0 <= errors <= accepted):
        raise ValueError("counts must satisfy 0 <= errors <= accepted")
    return 1.0 if accepted == 0 else float(binom.cdf(errors, accepted, alpha))


def certified(errors, accepted, alpha, delta=0.05, family_size=len(GRID)):
    if not 0 < delta < 1 or type(family_size) is not int or family_size < 1:
        raise ValueError("delta must be in (0, 1) and family_size must be positive")
    probability = p_value(errors, accepted, alpha)
    return accepted > 0 and probability <= delta / family_size


def choose(calibration, eligible, min_coverage=0.60):
    if not 0 <= min_coverage <= 1:
        raise ValueError("minimum coverage must be in [0, 1]")
    if not eligible:
        return {"max_coverage": None, "min_cost": None}
    most = max(eligible, key=lambda policy: calibration[policy.name]["coverage"])
    affordable = [policy for policy in eligible
                  if calibration[policy.name]["coverage"] >= min_coverage]
    cheapest = min(affordable, key=lambda policy:
                   calibration[policy.name]["weighted_tokens"]) if affordable else None
    return {"max_coverage": most, "min_cost": cheapest}


@dataclass(frozen=True)
class Selection:
    """What the calibration test returns for one risk target."""
    alpha: float
    delta: float
    answers: int
    calibration: dict
    p_values: dict
    certified: tuple
    max_coverage: object
    min_cost: object

    def __str__(self):
        size = len(self.calibration)
        lines = [f"alpha={self.alpha:g}  delta={self.delta:g}  schedules={size}  "
                 f"calibration answers={self.answers}",
                 f"{len(self.certified)} of {size} schedules pass p <= delta/{size}"]
        for tag, policy in (("widest", self.max_coverage), ("cheapest", self.min_cost)):
            if policy is None:
                lines.append(f"{tag:<9} none")
                continue
            item = self.calibration[policy.name]
            lines.append(f"{tag:<9} {policy.label:<35} coverage {item['coverage']:.1%}  "
                         f"risk {item['risk']:.2%}  weighted tokens {item['weighted_tokens']:.0f}")
        return "\n".join(lines)


def select(rows, alpha, delta=0.05, min_coverage=0.60, family=GRID):
    """Test every schedule of the family on labelled calibration rows.

    A schedule passes when it serves at least one answer and its binomial
    p-value is at most delta / len(family). The widest passing schedule is
    the default choice, the cheapest one with coverage >= min_coverage the other.
    """
    family = tuple(family)
    if not family or len({policy.name for policy in family}) != len(family):
        raise ValueError("the family must hold distinct schedules")
    if not 0 < delta < 1:
        raise ValueError("delta must be strictly between 0 and 1")
    calibration = {policy.name: evaluate(rows, policy) for policy in family}
    pvalues = {name: p_value(item["errors"], item["accepted"], alpha)
               for name, item in calibration.items()}
    passing = tuple(policy for policy in family
                    if calibration[policy.name]["accepted"] > 0
                    and pvalues[policy.name] <= delta / len(family))
    picked = choose(calibration, passing, min_coverage)
    return Selection(alpha, delta, len(rows), calibration, pvalues, passing,
                     picked["max_coverage"], picked["min_cost"])


def run_main(rows, alphas=ALPHAS, delta=0.05, min_coverage=0.60):
    if not alphas or any(not 0 < alpha < 1 for alpha in alphas):
        raise ValueError("risk targets must be in (0, 1)")
    if not 0 < delta < 1 or not 0 <= min_coverage <= 1:
        raise ValueError("invalid confidence budget or minimum coverage")
    if len(set(alphas)) != len(alphas):
        raise ValueError("risk targets must be distinct")
    splits = make_splits(rows)
    records = []
    for split in splits:
        calibration_rows = [rows[i] for i in split.calibration]
        test_rows = [rows[i] for i in split.test]
        calibration = {policy.name: evaluate(calibration_rows, policy) for policy in GRID}
        test_cache = {}
        for alpha in alphas:
            pvalues = {name: p_value(item["errors"], item["accepted"], alpha)
                       for name, item in calibration.items()}
            eligible = [policy for policy in GRID
                        if calibration[policy.name]["accepted"] > 0
                        and pvalues[policy.name] <= delta / len(GRID)]
            selections = {}
            for selector, policy in choose(calibration, eligible, min_coverage).items():
                if policy is None:
                    selections[selector] = None
                    continue
                if policy.name not in test_cache:
                    test_cache[policy.name] = evaluate(test_rows, policy)
                selections[selector] = {
                    "policy": policy.name,
                    "calibration": calibration[policy.name],
                    "test": test_cache[policy.name],
                }
            records.append({
                "split": split.name, "scheme": split.scheme, "alpha": alpha,
                "calibration_rows": len(calibration_rows), "test_rows": len(test_rows),
                "p_values": pvalues, "certified": [policy.name for policy in eligible],
                "selections": selections,
            })
    summaries = []
    for alpha in alphas:
        for selector in ("max_coverage", "min_cost"):
            selected = [record["selections"][selector] for record in records
                        if record["alpha"] == alpha]
            active = [item["test"] for item in selected if item is not None]
            risks = [item["risk"] for item in active if item["risk"] is not None]
            summaries.append({
                "alpha": alpha, "selector": selector,
                "splits_selected": len(active), "splits_total": len(splits),
                "coverage_all_splits": sum(item["coverage"] for item in active) / len(splits),
                "risk_selected_splits": sum(risks) / len(risks) if risks else None,
                "risk_violations": sum(risk > alpha for risk in risks),
                "tokens_selected_splits": (
                    sum(item["tokens"] for item in active) / len(active) if active else None),
                "weighted_tokens_selected_splits": (
                    sum(item["weighted_tokens"] for item in active) / len(active) if active else None),
            })
    return {
        "protocol": {
            "rows": len(rows), "errors": sum(1 - row["correct"] for row in rows),
            "problems": len({row["problem_id"] for row in rows}),
            "sources": len({row["source_id"] for row in rows}),
            "delta": delta, "family_size": len(GRID), "min_coverage": min_coverage,
            "alphas": list(alphas), "vote_order": "stored",
        },
        "summary": summaries, "splits": records,
    }
