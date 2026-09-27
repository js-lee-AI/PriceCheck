"""PriceCheck on the stored check outcomes of the paper. CPU only, no downloads, about a second."""

import pricecheck

bank = pricecheck.load_bank()                     # 2326 answers with label-free check outcomes
cal = [r for r in bank if r["source_id"] != 0]    # calibrate on four generators
test = [r for r in bank if r["source_id"] == 0]   # and serve the answers of the fifth, F0

selection = pricecheck.select(cal, alpha=0.015)   # test 23 schedules at a 1.5% risk target
print(selection)

served = pricecheck.evaluate(test, selection.max_coverage)
wrong = sum(1 - r["correct"] for r in test) / len(test)
print(f"held out  serve every answer    coverage 100.0%  risk {wrong:.2%}")
print(f"held out  PriceCheck            coverage {served['coverage']:.1%}  risk {served['risk']:.2%}")
