"""Shared helpers for the scripts in this folder."""

import json
from pathlib import Path

import pricecheck

# Source ids in the bank follow the sorted generator names, and the ten
# question-disjoint halves are seeds 0 to 9, which the paper calls A to J.
SOURCES = ("F0", "ForwardPref", "STaR", "SelfDistill", "iter2")
HALVES = "ABCDEFGHIJ"
LABELS = {policy.name: policy.label for policy in pricecheck.GRID}


def split_name(record):
    kind, index = record["split"].rsplit("_", 1)
    return SOURCES[int(index)] if kind == "source" else HALVES[int(index)]


def main_result(alphas=pricecheck.ALPHAS):
    return pricecheck.run_main(pricecheck.load_bank(), alphas=alphas)


def save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}")
