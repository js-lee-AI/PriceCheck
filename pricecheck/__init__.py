"""PriceCheck: risk-controlled selective answering by pricing label-free checks.

    import pricecheck

    bank = pricecheck.load_bank()                  # stored answers and their check outcomes
    choice = pricecheck.select(bank, alpha=0.015)  # test the 23 schedules on calibration answers
    pricecheck.decide(row, choice.max_coverage)    # serve or abstain on a new answer

Each check has a price, its agreement rate on correct and on wrong answers and
its cost per run. Prices compose into predictions of a schedule's coverage and
cost, and a binomial calibration test with a Bonferroni correction picks the
schedule that keeps selective risk under the target. Everything runs on CPU
with numpy and scipy.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .bank import DEFAULT_BANK, load_bank, validate_row
from .certification import (ALPHAS, Selection, Split, certified, choose, evaluate,
                            make_splits, p_value, run_main, select)
from .policies import GRID, PROBE_TOKENS, SMALL_WEIGHT, Policy, decide, race
from .pricing import cascade_price, count_tail, coverage_risk, fit_count_price, race_price

__all__ = [
    "__version__",
    "ALPHAS", "DEFAULT_BANK", "GRID", "PROBE_TOKENS", "SMALL_WEIGHT",
    "Policy", "Selection", "Split",
    "cascade_price", "certified", "choose", "count_tail", "coverage_risk", "decide",
    "evaluate", "fit_count_price", "load_bank", "make_splits", "p_value", "race",
    "race_price", "run_main", "select", "validate_row",
]
