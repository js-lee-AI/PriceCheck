"""The deployed family of 23 schedules and what each one costs per answer."""

from dataclasses import dataclass

PROBE_TOKENS = 3400.0
SMALL_WEIGHT = 0.21


@dataclass(frozen=True)
class Policy:
    kind: str
    first: int
    second: int | None = None

    @property
    def name(self):
        values = (self.kind, self.first, self.second)
        return ":".join(str(value) for value in values if value is not None)

    @property
    def label(self):
        """The schedule in the paper's words, e.g. "Race (2, 3)"."""
        k, j = self.first, self.second
        if self.kind in ("plain", "plain17", "plain4b"):
            size = {"plain": "8B", "plain17": "1.7B", "plain4b": "4B"}[self.kind]
            return f"Backward probe m>={k}, {size}"
        if self.kind in ("sc", "sc17"):
            size = "8B" if self.kind == "sc" else "1.7B"
            return f"Single re-solve, {size}" if k == 1 else f"Unanimous {k}-vote, {size}"
        first = {
            "asc": "", "casc": "Probe-first cascade, ",
            "scasc": "Cascade, score fast path, ", "v17first": "Small-first cascade, ",
            "c17first": "1.7B probe-first cascade, ",
        }[self.kind]
        return f"{first}{'race' if first else 'Race'} ({k}, {j})"


# Order determines ties in calibration coverage and cost.
GRID = tuple(
    [Policy("plain", t) for t in (2, 3)]
    + [Policy("sc", n) for n in (1, 2, 3, 4)]
    + [Policy("asc", k, j) for k in (2, 3) for j in (2, 3)]
    + [Policy("casc", k, j) for k in (2, 3) for j in (2, 3)]
    + [Policy("scasc", 2, j) for j in (2, 3)]
    + [Policy("v17first", 2, j) for j in (2, 3)]
    + [Policy("c17first", 2, j) for j in (2, 3)]
    + [Policy("sc17", 1), Policy("plain17", 2), Policy("plain4b", 2)]
)


def race(votes, agreements, disagreements):
    """Stop at either boundary; abstain if the bank is exhausted."""
    if (type(agreements) is not int or type(disagreements) is not int
            or agreements < 1 or disagreements < 1):
        raise ValueError("race boundaries must be positive integers")
    matches = misses = used = 0
    for vote in votes:
        used += 1
        matches += bool(vote)
        misses += not vote
        if matches >= agreements:
            return True, used
        if misses >= disagreements:
            return False, used
    return False, used


def decide(row, policy):
    """Return acceptance, generated tokens, and parameter-weighted tokens.

    Only check observations and costs are read; correctness is never read.
    """
    kind, k, j = policy.kind, policy.first, policy.second
    solve8, solve17 = row["solve_tok_8b"], row["solve_tok_1p7b"]
    if kind in ("plain", "plain17", "plain4b"):
        field, weight = {
            "plain": ("m_8b", 1.0),
            "plain17": ("m_1p7b", SMALL_WEIGHT),
            "plain4b": ("m_4b", 0.5),
        }[kind]
        return row[field] >= k, PROBE_TOKENS, PROBE_TOKENS * weight
    if kind in ("sc", "sc17"):
        votes = row["votes_8b" if kind == "sc" else "votes_1p7b"]
        tokens = k * (solve8 if kind == "sc" else solve17)
        weight = 1.0 if kind == "sc" else SMALL_WEIGHT
        return sum(votes[:k]) >= k, tokens, tokens * weight
    if kind == "asc":
        accept, used = race(row["votes_8b"], k, j)
        return accept, used * solve8, used * solve8
    if kind in ("casc", "scasc", "c17first", "v17first"):
        tokens = PROBE_TOKENS
        weighted = PROBE_TOKENS
        if kind == "v17first":
            count = sum(row["votes_1p7b"][:3])
            tokens = 3 * solve17
            weighted = tokens * SMALL_WEIGHT
            accept = count == 3
        elif kind == "c17first":
            count = row["m_1p7b"]
            weighted *= SMALL_WEIGHT
            accept = count >= 3
        else:
            count = row["m_8b"]
            if kind == "scasc":
                score = row["rev_score"]
                accept = count >= 2 and score is not None and score >= -0.3
            else:
                accept = count >= 3
        if accept:
            return True, tokens, weighted
        if count == 0:
            return False, tokens, weighted
        accept, used = race(row["votes_8b"], k, j)
        return accept, tokens + used * solve8, weighted + used * solve8
    raise ValueError(f"unknown policy kind: {kind}")
