import copy
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pricecheck.bank import load_bank, validate_row
from pricecheck.certification import certified, choose, make_splits, p_value, run_main
from pricecheck.policies import GRID, Policy, decide, race
from pricecheck.pricing import cascade_price, count_tail, coverage_risk, fit_count_price, race_price


def example():
    return {
        "problem_id": 0, "source_id": 0, "correct": 1,
        "m_8b": 2, "m_4b": 2, "m_1p7b": 2, "rev_score": -0.3,
        "votes_8b": [1, 0, 1, 0] * 4, "votes_1p7b": [1, 0, 1, 0] * 2,
        "solve_tok_8b": 100, "solve_tok_1p7b": 50,
    }


class ScheduleTests(unittest.TestCase):
    def test_race_stops_at_first_boundary(self):
        self.assertEqual(race([1, 0, 1, 0], 2, 2), (True, 3))
        self.assertEqual(race([0, 1, 0, 1], 2, 2), (False, 3))
        self.assertEqual(race([1], 2, 2), (False, 1))
        self.assertEqual(race([], 2, 2), (False, 0))
        with self.assertRaises(ValueError):
            race([1], 0, 2)
        with self.assertRaises(ValueError):
            race([1], 1.5, 2)

    def test_all_grid_members_have_label_free_decisions(self):
        self.assertEqual(len(GRID), 23)
        self.assertEqual(len({policy.name for policy in GRID}), 23)
        row = example()
        without_label = dict(row)
        del without_label["correct"]
        for policy in GRID:
            self.assertEqual(decide(row, policy), decide(without_label, policy))

    def test_threshold_and_unanimity(self):
        row = example()
        self.assertEqual(decide(row, Policy("plain", 2)), (True, 3400, 3400))
        self.assertEqual(decide(row, Policy("plain", 3)), (False, 3400, 3400))
        self.assertEqual(decide(row, Policy("plain17", 2)), (True, 3400, 714))
        self.assertEqual(decide(row, Policy("plain4b", 2)), (True, 3400, 1700))
        self.assertEqual(decide(row, Policy("sc", 2)), (False, 200, 200))
        self.assertEqual(decide(row, Policy("sc17", 1)), (True, 50, 10.5))

    def test_cascade_pays_only_reached_stages(self):
        row = example()
        self.assertEqual(decide(row, Policy("casc", 2, 2)), (True, 3700, 3700))
        self.assertEqual(decide(row, Policy("v17first", 2, 2)), (True, 450, 331.5))
        self.assertEqual(decide(row, Policy("c17first", 2, 2)), (True, 3700, 1014))
        row["m_8b"] = 0
        self.assertEqual(decide(row, Policy("casc", 2, 2)), (False, 3400, 3400))
        row["m_8b"] = 3
        row["votes_8b"] = [0] * 16
        self.assertEqual(decide(row, Policy("casc", 2, 2)), (True, 3400, 3400))
        row["votes_1p7b"] = [0] * 8
        self.assertEqual(decide(row, Policy("v17first", 2, 2)), (False, 150, 31.5))

    def test_score_gate_includes_boundary_and_zero(self):
        row = example()
        row["votes_8b"] = [0] * 16
        for score in (-0.3, 0.0):
            row["rev_score"] = score
            self.assertEqual(decide(row, Policy("scasc", 2, 2)), (True, 3400, 3400))
        for score in (-0.30000001, None):
            row["rev_score"] = score
            self.assertEqual(decide(row, Policy("scasc", 2, 2)), (False, 3600, 3600))


class CertificationTests(unittest.TestCase):
    def test_zero_error_sample_boundary(self):
        alpha, delta, size = 0.015, 0.05, 23
        required = math.ceil(math.log(delta / size) / math.log1p(-alpha))
        self.assertFalse(certified(0, required - 1, alpha))
        self.assertTrue(certified(0, required, alpha))
        self.assertAlmostEqual(p_value(0, required, alpha), (1 - alpha) ** required)
        self.assertFalse(certified(0, 0, alpha))
        self.assertFalse(certified(5, 5, alpha))
        with self.assertRaises(ValueError):
            p_value(4, 3, alpha)
        with self.assertRaises(ValueError):
            certified(0, 0, 0)

    def test_inclusive_test_and_error_monotonicity(self):
        threshold = p_value(1, 1000, 0.015)
        self.assertTrue(certified(1, 1000, 0.015, delta=threshold, family_size=1))
        self.assertFalse(certified(1, 1000, 0.015,
                                   delta=math.nextafter(threshold, 0), family_size=1))
        self.assertLess(p_value(0, 1000, 0.015), p_value(1, 1000, 0.015))

    def test_selection_ties_and_minimum_coverage(self):
        first, second, third = GRID[:3]
        calibration = {
            first.name: {"coverage": 0.7, "weighted_tokens": 300},
            second.name: {"coverage": 0.7, "weighted_tokens": 200},
            third.name: {"coverage": 0.6, "weighted_tokens": 100},
        }
        picked = choose(calibration, [first, second, third])
        self.assertEqual(picked["max_coverage"], first)
        self.assertEqual(picked["min_cost"], third)
        self.assertIsNone(choose(calibration, [third], 0.60001)["min_cost"])
        self.assertEqual(choose(calibration, []), {"max_coverage": None, "min_cost": None})

    def test_group_disjointness_and_complete_splits(self):
        rows = load_bank()
        self.assertEqual((len(rows), sum(1 - row["correct"] for row in rows)), (2326, 99))
        splits = make_splits(rows)
        self.assertEqual(len(splits), 15)
        for split in splits:
            cal, test = set(split.calibration), set(split.test)
            self.assertFalse(cal & test)
            self.assertEqual(cal | test, set(range(len(rows))))
            field = "problem_id" if split.scheme == "question_disjoint" else "source_id"
            self.assertFalse({rows[i][field] for i in cal} & {rows[i][field] for i in test})

    def test_main_operating_points(self):
        result = run_main(load_bank(), alphas=(0.005, 0.015))
        maximum = [item for item in result["summary"] if item["selector"] == "max_coverage"]
        self.assertEqual([item["splits_selected"] for item in maximum], [5, 15])
        self.assertAlmostEqual(100 * maximum[1]["coverage_all_splits"], 76.09, places=2)
        self.assertEqual(maximum[1]["risk_violations"], 0)
        self.assertTrue(all(item["selections"]["max_coverage"] is None
                            for item in result["splits"] if item["alpha"] == 0.005
                            and item["scheme"] == "question_disjoint"))


class PricingTests(unittest.TestCase):
    def test_race_formula_against_exhaustive_paths(self):
        for k, j, probability in itertools.product(range(1, 4), range(1, 4), (0.0, 0.23, 1.0)):
            expected_acceptance = expected_draws = 0.0
            length = k + j - 1
            for votes in itertools.product((0, 1), repeat=length):
                mass = probability ** sum(votes) * (1 - probability) ** (length - sum(votes))
                accept, used = race(votes, k, j)
                expected_acceptance += mass * accept
                expected_draws += mass * used
            predicted = race_price(k, j, probability)
            self.assertAlmostEqual(predicted[0], expected_acceptance)
            self.assertAlmostEqual(predicted[1], expected_draws)

    def test_count_fit_and_composition_boundaries(self):
        rate, dispersion = fit_count_price([0, 0, 0, 3, 3, 3])
        self.assertEqual((rate, dispersion), (0.5, 0.95))
        self.assertEqual(count_tail(3, 0, 0.4), 1)
        self.assertEqual(count_tail(3, 4, 0.4), 0)
        self.assertAlmostEqual(count_tail(3, 2, 0.4), 3 * 0.4 ** 2 * 0.6 + 0.4 ** 3)
        self.assertEqual(cascade_price(1, 0, 0, 2, 2, 10, 20), (1, 10))
        self.assertEqual(cascade_price(0, 0, 1, 2, 2, 10, 20), (0, 10))
        self.assertEqual(coverage_risk(0.9, 0, 0), (0, None))
        self.assertAlmostEqual(coverage_risk(0.9, 0.8, 0.1)[0], 0.73)
        with self.assertRaises(ValueError):
            fit_count_price([])
        with self.assertRaises(ValueError):
            fit_count_price([1, 2], draws=3.5)


class InputTests(unittest.TestCase):
    def test_malformed_banks(self):
        for key, value in (("correct", 2), ("votes_8b", [1]),
                           ("rev_score", float("nan")), ("source_id", "name"),
                           ("m_4b", 4), ("solve_tok_8b", -1)):
            row = copy.deepcopy(example())
            row[key] = value
            with self.assertRaises(ValueError):
                validate_row(row)
        row = example()
        row["extra"] = "text"
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_cli_rejects_invalid_inputs_without_output(self):
        package = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            for args in (("--alphas", "nan"), ("--alphas", "0"),
                         ("--delta", "1"), ("--min-coverage", "1.01"),
                         ("--alphas", "0.01", "0.01")):
                proc = subprocess.run([sys.executable, "-m", "pricecheck", "certify", *args,
                                       "--output", str(output)], cwd=package,
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertFalse(output.exists())
            bad = Path(directory) / "bad.jsonl"
            bad.write_text(json.dumps({"correct": 1}) + "\n", encoding="utf-8")
            proc = subprocess.run([sys.executable, "-m", "pricecheck", "certify", "--data", str(bad),
                                   "--output", str(output)], cwd=package,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("invalid bank row 1", proc.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
