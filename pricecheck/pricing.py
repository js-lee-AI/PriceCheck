"""Prices of label-free checks and how they compose into schedule predictions.

A price is the agreement rate on correct answers (completeness), the rate on
wrong answers (leak) and a unit cost. Predictions guide which schedules to put
in a family. They are not a certificate, which is what calibration is for.
"""

from functools import lru_cache

import numpy as np
from scipy.stats import betabinom, binom


def fit_count_price(counts, draws=3):
    """Fit a class-specific agreement rate and count overdispersion."""
    counts = np.asarray(counts, dtype=float)
    if (type(draws) is not int or draws < 2 or counts.ndim != 1 or counts.size == 0
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts > draws) or np.any(counts != np.floor(counts))):
        raise ValueError("counts must be integers between 0 and draws; draws must be at least 2")
    rate = float(np.clip(counts.mean() / draws, 1e-4, 1 - 1e-4))
    dispersion = 0.0
    if len(counts) >= 5:
        variance_ratio = counts.var(ddof=1) / (draws * rate * (1 - rate))
        dispersion = float(np.clip((variance_ratio - 1) / (draws - 1), 0, 0.95))
    return rate, dispersion


def count_tail(draws, threshold, rate, dispersion=0.0):
    if type(draws) is not int or draws < 1 or type(threshold) is not int:
        raise ValueError("draws must be positive and threshold must be an integer")
    if not 0 <= rate <= 1 or not 0 <= dispersion < 1:
        raise ValueError("invalid agreement rate or dispersion")
    if threshold <= 0:
        return 1.0
    if threshold > draws:
        return 0.0
    if dispersion < 0.01 or rate in (0, 1):
        return float(binom.sf(threshold - 1, draws, rate))
    a = rate * (1 - dispersion) / dispersion
    b = (1 - rate) * (1 - dispersion) / dispersion
    return float(betabinom.sf(threshold - 1, draws, a, b))


def race_price(agreements, disagreements, rate):
    """Acceptance probability and expected draws under independent verdicts."""
    if (type(agreements) is not int or type(disagreements) is not int
            or agreements < 1 or disagreements < 1 or not 0 <= rate <= 1):
        raise ValueError("invalid race boundaries or agreement rate")

    @lru_cache(None)
    def visit(k, j):
        if k == 0:
            return 1.0, 0.0
        if j == 0:
            return 0.0, 0.0
        yes, yes_draws = visit(k - 1, j)
        no, no_draws = visit(k, j - 1)
        return (rate * yes + (1 - rate) * no,
                1 + rate * yes_draws + (1 - rate) * no_draws)

    return visit(agreements, disagreements)


def cascade_price(first_rate, first_dispersion, vote_rate, agreements,
                  disagreements, first_cost, vote_cost):
    """Three first-stage agreements serve, zero abstain, and the band races.

    Apply separately to each class; stages are conditionally independent.
    The cost arguments may use generated or parameter-weighted tokens.
    """
    if not (np.isfinite(first_cost) and np.isfinite(vote_cost)
            and first_cost >= 0 and vote_cost >= 0):
        raise ValueError("costs must be finite and nonnegative")
    fast = count_tail(3, 3, first_rate, first_dispersion)
    band = count_tail(3, 1, first_rate, first_dispersion) - fast
    accept, draws = race_price(agreements, disagreements, vote_rate)
    return fast + band * accept, first_cost + band * draws * vote_cost


def coverage_risk(correct_fraction, correct_acceptance, wrong_acceptance):
    values = (correct_fraction, correct_acceptance, wrong_acceptance)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("class fractions and acceptance probabilities must be in [0, 1]")
    coverage = correct_fraction * correct_acceptance + (1 - correct_fraction) * wrong_acceptance
    risk = (1 - correct_fraction) * wrong_acceptance / coverage if coverage else None
    return coverage, risk
