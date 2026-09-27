"""Price two checks, predict a cascade from the prices, then run it. CPU only."""

import pricecheck

bank = pricecheck.load_bank()
right = [r for r in bank if r["correct"]]
wrong = [r for r in bank if not r["correct"]]

# Price the 8B backward probe and the 8B re-solve vote on each class, three draws each.
probe = {1: pricecheck.fit_count_price([r["m_8b"] for r in right]),
         0: pricecheck.fit_count_price([r["m_8b"] for r in wrong])}
vote = {1: pricecheck.fit_count_price([sum(r["votes_8b"][:3]) for r in right])[0],
        0: pricecheck.fit_count_price([sum(r["votes_8b"][:3]) for r in wrong])[0]}

# Probe first, serve at 3 of 3, abstain at 0 of 3, race (2, 2) on the band.
accept = {y: pricecheck.cascade_price(*probe[y], vote[y], 2, 2, 3400, 5384)[0] for y in (1, 0)}
coverage, risk = pricecheck.coverage_risk(len(right) / len(bank), accept[1], accept[0])
print(f"predicted  coverage {coverage:.1%}  risk {risk:.2%}")

actual = pricecheck.evaluate(bank, pricecheck.Policy("casc", 2, 2))
print(f"actual     coverage {actual['coverage']:.1%}  risk {actual['risk']:.2%}")
