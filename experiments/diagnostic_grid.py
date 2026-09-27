"""The 118-schedule diagnostic grid and the price-prediction protocol.

Twenty times (seeds 0 to 19), draw a pricing set of 100 answers stratified by
generator with at least eight wrong ones, fit the price of every check on it,
predict each schedule's coverage, risk and cost by composing those prices, and
compare with what the schedule actually does on the remaining 2226 answers.
No schedule-level quantity is ever fitted. This grid is separate from the
23-schedule family that the calibration test uses.
"""

from functools import lru_cache

import numpy as np
from scipy.stats import betabinom, binom, spearmanr

import pricecheck
from pricecheck.pricing import count_tail, fit_count_price, race_price

PROBE_TOKENS = pricecheck.PROBE_TOKENS
WEIGHT = {"8b": 1.0, "4b": 0.5, "1p7b": 0.21}
SIZE = {"8b": "8B", "4b": "4B", "1p7b": "1.7B"}


def _schedules():
    grid = []
    for model in ("8b", "4b", "1p7b"):
        for t in (1, 2, 3):
            grid.append(dict(name=f"probe {SIZE[model]} m>={t}", family=f"probe_{model}",
                             kind="probe", model=model, t=t, actions=1, observations=3))
    for model, sizes in (("8b", (1, 2, 3, 4, 8, 16)), ("1p7b", (1, 2, 3, 4, 8))):
        for n in sizes:
            for t in range(max(1, n - 2), n + 1):
                grid.append(dict(name=f"vote {SIZE[model]} {t} of {n}", family=f"vote_{model}",
                                 kind="vote", model=model, n=n, t=t, actions=1, observations=n))
    for model in ("8b", "1p7b"):
        for k in (2, 3):
            for j in (2, 3, 4):
                grid.append(dict(name=f"race {SIZE[model]} ({k}, {j})", family=f"race_{model}",
                                 kind="race", model=model, k=k, j=j, actions=1,
                                 observations=k + j - 1))
    # Probe first, then the 8B race on the band. The score gate is an 8B probe signal.
    for model in ("8b", "1p7b", "4b"):
        for rule in ("m3", "m2", "m2s30", "m2s35"):
            if model != "8b" and rule.startswith("m2s"):
                continue
            for abstain in (0, 1):
                for k in (2, 3):
                    for j in (2, 3):
                        grid.append(dict(
                            name=f"cascade {SIZE[model]} probe {rule} abstain<={abstain} race ({k}, {j})",
                            family=f"cascade_probe_{model}", kind="probe_first", model=model,
                            rule=rule, abstain=abstain, k=k, j=j,
                            actions=3 if rule.startswith("m2s") else 2,
                            observations=3 + k + j - 1))
    for k in (2, 3):
        for j in (2, 3):
            grid.append(dict(name=f"cascade 1.7B votes race ({k}, {j})", family="cascade_votes",
                             kind="small_first", k=k, j=j, actions=2, observations=3 + k + j - 1))
    for j in (2, 3):
        grid.append(dict(name=f"triple 1.7B race (2, {j})", family="cascade_triple",
                         kind="triple", k=2, j=j, actions=3, observations=3 + 2 + j + 1))
    return grid


SCHEDULES = _schedules()


def _gate(row, rule, count):
    if rule == "m3":
        return count >= 3
    if rule == "m2":
        return count >= 2
    score = row["rev_score"]
    cut = -0.30 if rule == "m2s30" else -0.35
    return count >= 2 and score is not None and score >= cut


def decide(row, s):
    """Serve or abstain, with generated and parameter-weighted tokens."""
    kind = s["kind"]
    if kind == "probe":
        return row[f"m_{s['model']}"] >= s["t"], PROBE_TOKENS, PROBE_TOKENS * WEIGHT[s["model"]]
    if kind == "vote":
        cost = s["n"] * row[f"solve_tok_{s['model']}"]
        return (sum(row[f"votes_{s['model']}"][:s["n"]]) >= s["t"], cost,
                cost * WEIGHT[s["model"]])
    if kind == "race":
        accept, used = pricecheck.race(row[f"votes_{s['model']}"], s["k"], s["j"])
        cost = used * row[f"solve_tok_{s['model']}"]
        return accept, cost, cost * WEIGHT[s["model"]]
    if kind == "probe_first":
        count, weight = row[f"m_{s['model']}"], WEIGHT[s["model"]]
        if _gate(row, s["rule"], count):
            return True, PROBE_TOKENS, PROBE_TOKENS * weight
        if count <= s["abstain"]:
            return False, PROBE_TOKENS, PROBE_TOKENS * weight
        accept, used = pricecheck.race(row["votes_8b"], s["k"], s["j"])
        cost = used * row["solve_tok_8b"]
        return accept, PROBE_TOKENS + cost, PROBE_TOKENS * weight + cost
    if kind == "small_first":
        count = sum(row["votes_1p7b"][:3])
        first = 3 * row["solve_tok_1p7b"]
        if count >= 3:
            return True, first, first * 0.21
        if count <= 0:
            return False, first, first * 0.21
        accept, used = pricecheck.race(row["votes_8b"], s["k"], s["j"])
        cost = used * row["solve_tok_8b"]
        return accept, first + cost, first * 0.21 + cost
    if kind == "triple":
        # 1.7B probe, then a 1.7B race, then one confirming 8B solve.
        count = row["m_1p7b"]
        tokens, weighted = PROBE_TOKENS, PROBE_TOKENS * 0.21
        if count >= 3:
            return True, tokens, weighted
        if count == 0:
            return False, tokens, weighted
        small, used = pricecheck.race(row["votes_1p7b"], s["k"], s["j"] + 1)
        tokens += used * row["solve_tok_1p7b"]
        weighted += used * row["solve_tok_1p7b"] * 0.21
        if small:
            accept, used = pricecheck.race(row["votes_8b"], 1, 1)
            tokens += used * row["solve_tok_8b"]
            weighted += used * row["solve_tok_8b"]
            return accept, tokens, weighted
        return False, tokens, weighted
    raise ValueError(kind)


SIGNALS = {
    "probe_8b": lambda r: r["m_8b"], "probe_4b": lambda r: r["m_4b"],
    "probe_1p7b": lambda r: r["m_1p7b"],
    "vote_8b": lambda r: sum(r["votes_8b"][:3]), "vote_1p7b": lambda r: sum(r["votes_1p7b"][:3]),
}


def fit_prices(rows):
    right = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    prices = {"pi": len(right) / len(rows)}
    for name, get in SIGNALS.items():
        prices[name] = {1: fit_count_price([get(r) for r in right]),
                        0: fit_count_price([get(r) for r in wrong])}
    for cut, name in ((-0.30, "m2s30"), (-0.35, "m2s35")):
        prices[name] = {y: float(np.mean([r["rev_score"] is not None and r["rev_score"] >= cut
                                          for r in group]))
                        for y, group in ((1, right), (0, wrong))}
    prices["unit_8b"] = float(np.mean([r["solve_tok_8b"] for r in rows]))
    prices["unit_1p7b"] = float(np.mean([r["solve_tok_1p7b"] for r in rows]))
    return prices


def count_pmf(draws, m, rate, dispersion):
    if dispersion < 0.01:
        return float(binom.pmf(m, draws, rate))
    a = rate * (1 - dispersion) / dispersion
    b = (1 - rate) * (1 - dispersion) / dispersion
    return float(betabinom.pmf(m, draws, a, b))


@lru_cache(None)
def _race(k, j, rate):
    return race_price(k, j, rate)


def predict(s, prices):
    """Compose check prices into coverage, risk and cost. Stages are independent given the label."""
    pi, accept, tokens, weighted = prices["pi"], {}, {}, {}
    for y in (1, 0):
        kind = s["kind"]
        if kind == "probe":
            rate, disp = prices[f"probe_{s['model']}"][y]
            accept[y] = count_tail(3, s["t"], rate, disp)
            tokens[y], weighted[y] = PROBE_TOKENS, PROBE_TOKENS * WEIGHT[s["model"]]
        elif kind == "vote":
            rate, disp = prices[f"vote_{s['model']}"][y]
            accept[y] = count_tail(s["n"], s["t"], rate, disp)
            unit = prices[f"unit_{s['model']}"]
            tokens[y], weighted[y] = s["n"] * unit, s["n"] * unit * WEIGHT[s["model"]]
        elif kind == "race":
            rate = prices[f"vote_{s['model']}"][y][0]
            accept[y], draws = _race(s["k"], s["j"], round(rate, 6))
            unit = prices[f"unit_{s['model']}"]
            tokens[y], weighted[y] = draws * unit, draws * unit * WEIGHT[s["model"]]
        elif kind == "probe_first":
            rate, disp = prices[f"probe_{s['model']}"][y]
            if s["rule"] == "m3":
                fast = count_tail(3, 3, rate, disp)
            elif s["rule"] == "m2":
                fast = count_tail(3, 2, rate, disp)
            else:
                fast = count_tail(3, 2, rate, disp) * prices[s["rule"]][y]
            stop = 1.0 - count_tail(3, s["abstain"] + 1, rate, disp)
            band = max(0.0, 1.0 - fast - stop)
            later, draws = _race(s["k"], s["j"], round(prices["vote_8b"][y][0], 6))
            accept[y] = fast + band * later
            unit = prices["unit_8b"]
            tokens[y] = PROBE_TOKENS + band * draws * unit
            weighted[y] = PROBE_TOKENS * WEIGHT[s["model"]] + band * draws * unit
        elif kind == "small_first":
            rate, disp = prices["vote_1p7b"][y]
            fast, stop = count_pmf(3, 3, rate, disp), count_pmf(3, 0, rate, disp)
            band = max(0.0, 1.0 - fast - stop)
            later, draws = _race(s["k"], s["j"], round(prices["vote_8b"][y][0], 6))
            accept[y] = fast + band * later
            small, large = prices["unit_1p7b"], prices["unit_8b"]
            tokens[y] = 3 * small + band * draws * large
            weighted[y] = 3 * small * 0.21 + band * draws * large
        elif kind == "triple":
            rate, disp = prices["probe_1p7b"][y]
            fast, stop = count_pmf(3, 3, rate, disp), count_pmf(3, 0, rate, disp)
            band = max(0.0, 1.0 - fast - stop)
            small, draws = _race(s["k"], s["j"] + 1, round(prices["vote_1p7b"][y][0], 6))
            confirm = prices["vote_8b"][y][0]
            accept[y] = fast + band * small * confirm
            u17, u8 = prices["unit_1p7b"], prices["unit_8b"]
            tokens[y] = PROBE_TOKENS + band * (draws * u17 + small * u8)
            weighted[y] = PROBE_TOKENS * 0.21 + band * (draws * u17 * 0.21 + small * u8)
    coverage = pi * accept[1] + (1 - pi) * accept[0]
    return {"cov": coverage, "risk": (1 - pi) * accept[0] / coverage if coverage > 0 else 0.0,
            "tok": pi * tokens[1] + (1 - pi) * tokens[0],
            "flop": pi * weighted[1] + (1 - pi) * weighted[0]}


def draw_pricing_set(rows, seed, size=100, floor=8):
    """Stratified by generator, with wrong answers swapped in up to the floor."""
    rng = np.random.default_rng(seed)
    sources = sorted({r["source_id"] for r in rows})
    picked = []
    for source in sources:
        members = [i for i, r in enumerate(rows) if r["source_id"] == source]
        picked.extend(rng.choice(members, size=min(size // len(sources), len(members)),
                                 replace=False).tolist())
    wrong = [i for i in picked if not rows[i]["correct"]]
    if len(wrong) < floor:
        spare = [i for i, r in enumerate(rows) if not r["correct"] and i not in picked]
        added = rng.choice(spare, size=floor - len(wrong), replace=False).tolist()
        right = [i for i in picked if rows[i]["correct"]]
        dropped = set(rng.choice(right, size=len(added), replace=False).tolist())
        picked = [i for i in picked if i not in dropped] + added
    return picked


def run(rows=None, seeds=20):
    """Per schedule, the (prediction, realised) pair of every seed."""
    rows = rows or pricecheck.load_bank()
    outcomes = [[decide(r, s) for r in rows] for s in SCHEDULES]
    wrong = np.array([not r["correct"] for r in rows])
    pairs = [[] for _ in SCHEDULES]
    for seed in range(seeds):
        picked = draw_pricing_set(rows, seed)
        chosen = set(picked)
        prices = fit_prices([rows[i] for i in picked])
        held = [i for i in range(len(rows)) if i not in chosen]
        for index, s in enumerate(SCHEDULES):
            result = outcomes[index]
            served = np.array([result[i][0] for i in held])
            kept, errors = int(served.sum()), int((served & wrong[held]).sum())
            real = {"cov": kept / len(held), "risk": errors / kept if kept else 0.0,
                    "tok": sum(result[i][1] for i in held) / len(held),
                    "flop": sum(result[i][2] for i in held) / len(held)}
            pairs[index].append((predict(s, prices), real))
    return pairs


def per_schedule(pairs):
    rows = []
    for s, records in zip(SCHEDULES, pairs):
        pc, ac = (np.array([x[k]["cov"] for x in records]) * 100 for k in (0, 1))
        pr, ar = (np.array([x[k]["risk"] for x in records]) * 100 for k in (0, 1))
        pt, at = (np.array([x[k]["tok"] for x in records]) for k in (0, 1))
        pf, af = (np.array([x[k]["flop"] for x in records]) for k in (0, 1))
        rows.append(dict(
            name=s["name"], family=s["family"], actions=s["actions"],
            cascade=s["family"].startswith("cascade"),
            pred_cov=round(float(pc.mean()), 2), act_cov=round(float(ac.mean()), 2),
            cov_mae=round(float(np.abs(pc - ac).mean()), 3),
            pred_risk=round(float(pr.mean()), 3), act_risk=round(float(ar.mean()), 3),
            risk_mae=round(float(np.abs(pr - ar).mean()), 4),
            risk_bias=round(float((pr - ar).mean()), 4),
            tok_bias=round(float(((pt - at) / np.maximum(at, 1e-9)).mean() * 100), 2),
            flop_bias=round(float(((pf - af) / np.maximum(af, 1e-9)).mean() * 100), 2)))
    return rows


def summary(rows):
    out = {}
    for label, pick in (("cascades", lambda r: r["cascade"]), ("all", lambda r: True)):
        sub = [r for r in rows if pick(r)]
        out[label] = dict(
            n=len(sub),
            cov_mae=round(float(np.mean([r["cov_mae"] for r in sub])), 3),
            cov_spearman=round(float(spearmanr([r["pred_cov"] for r in sub],
                                               [r["act_cov"] for r in sub])[0]), 4),
            risk_mae=round(float(np.mean([r["risk_mae"] for r in sub])), 3),
            risk_bias=round(float(np.mean([r["risk_bias"] for r in sub])), 3),
            tok_bias=round(float(np.mean([r["tok_bias"] for r in sub])), 3),
            flop_bias=round(float(np.mean([r["flop_bias"] for r in sub])), 3))
    actions = np.array([r["actions"] for r in rows], float)
    out["cov_mae_slope_per_action"] = round(float(np.polyfit(
        actions, [r["cov_mae"] for r in rows], 1)[0]), 4)
    return out
