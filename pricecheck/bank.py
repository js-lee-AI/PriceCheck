"""Read and validate a bank of stored check outcomes, one JSON row per answer."""

import json
import math
from pathlib import Path

DEFAULT_BANK = Path(__file__).parent / "data" / "main_bank.jsonl"
FIELDS = {
    "problem_id", "source_id", "correct", "m_8b", "m_4b", "m_1p7b",
    "rev_score", "votes_8b", "votes_1p7b", "solve_tok_8b", "solve_tok_1p7b",
}


def validate_row(row):
    if not isinstance(row, dict) or set(row) != FIELDS:
        raise ValueError("each row must contain exactly the documented fields")
    for key in ("problem_id", "source_id"):
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if type(row["correct"]) is not int or row["correct"] not in (0, 1):
        raise ValueError("correct must be 0 or 1")
    for key in ("m_8b", "m_4b", "m_1p7b"):
        if type(row[key]) is not int or not 0 <= row[key] <= 3:
            raise ValueError(f"{key} must be an integer from 0 to 3")
    for key, length in (("votes_8b", 16), ("votes_1p7b", 8)):
        votes = row[key]
        if not isinstance(votes, list) or len(votes) != length:
            raise ValueError(f"{key} must have {length} entries")
        if any(type(v) is not int or v not in (0, 1) for v in votes):
            raise ValueError(f"{key} entries must be 0 or 1")
    for key in ("solve_tok_8b", "solve_tok_1p7b"):
        if type(row[key]) is not int or row[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    score = row["rev_score"]
    if score is not None and (
        type(score) not in (int, float) or not math.isfinite(score)
    ):
        raise ValueError("rev_score must be a finite number or null")


def load_bank(path=DEFAULT_BANK):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                validate_row(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid bank row {number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError("the bank is empty")
    return rows
