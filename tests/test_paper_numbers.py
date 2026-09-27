"""Every number the experiment scripts rebuild, checked against the paper as printed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import common  # noqa: E402
import diagnostic_grid  # noqa: E402
import table1_prices  # noqa: E402
import table2_certification  # noqa: E402
import table8_modal_selection  # noqa: E402
import table9_split_selection  # noqa: E402
import table12_risk_bounds  # noqa: E402
import table14_price_fit  # noqa: E402
import table16_price_prediction  # noqa: E402
import table19_operating_points  # noqa: E402


@pytest.fixture(scope="module")
def main_result():
    return common.main_result()


@pytest.fixture(scope="module")
def grid_pairs():
    return diagnostic_grid.run()


def test_table2_table3a_table10(main_result):
    out = table2_certification.compute(main_result)
    printed = [(5, "23.20"), (12, "57.00"), (15, "72.11"), (15, "73.85"), (15, "76.09"), (15, "78.86")]
    assert [(r["splits"], f"{r['coverage']:.2f}") for r in out["table"]] == printed
    assert out["wrong_kept"] == 44
    assert f"{out['coverage_sd']:.1f}" == "4.4"
    assert f"{out['by_scheme']['question_disjoint']:.2f}" == "74.21"
    assert f"{out['by_scheme']['source_held_out']:.2f}" == "79.84"
    assert f"{out['table'][4]['tokens']:.0f}" == "15028"


TABLE8 = [  # scheme, target, selector, schedule, splits, cov., risk, generated, weighted
    ("source_held_out", 0.005, "max_coverage", "sc:4", "5/5", "69.6", "0.00", "21533", "21533"),
    ("source_held_out", 0.005, "min_cost", "sc:4", "5/5", "69.6", "0.00", "21533", "21533"),
    ("source_held_out", 0.0075, "max_coverage", "sc:4", "5/5", "69.6", "0.00", "21533", "21533"),
    ("source_held_out", 0.0075, "min_cost", "sc:4", "5/5", "69.6", "0.00", "21533", "21533"),
    ("source_held_out", 0.01, "max_coverage", "sc:3", "5/5", "71.6", "0.18", "16150", "16150"),
    ("source_held_out", 0.01, "min_cost", "sc:3", "5/5", "71.6", "0.18", "16150", "16150"),
    ("source_held_out", 0.0125, "max_coverage", "asc:3:3", "2/5", "78.5", "0.55", "17232", "17232"),
    ("source_held_out", 0.0125, "min_cost", "scasc:2:2", "2/5", "71.0", "0.61", "4810", "4810"),
    ("source_held_out", 0.015, "max_coverage", "asc:2:3", "5/5", "79.8", "0.54", "13426", "13426"),
    ("source_held_out", 0.015, "min_cost", "v17first:2:2", "3/5", "77.7", "0.55", "15558", "4107"),
    ("source_held_out", 0.02, "max_coverage", "asc:2:3", "5/5", "79.8", "0.54", "13426", "13426"),
    ("source_held_out", 0.02, "min_cost", "plain17:2", "5/5", "63.2", "0.61", "3400", "714"),
    ("question_disjoint", 0.0075, "max_coverage", "sc:4", "3/10", "68.1", "0.00", "21550", "21550"),
    ("question_disjoint", 0.0075, "min_cost", "sc:4", "3/10", "68.1", "0.00", "21550", "21550"),
    ("question_disjoint", 0.01, "max_coverage", "sc:3", "4/10", "72.5", "0.36", "16104", "16104"),
    ("question_disjoint", 0.01, "min_cost", "sc:3", "4/10", "72.5", "0.36", "16104", "16104"),
    ("question_disjoint", 0.0125, "max_coverage", "sc:3", "4/10", "72.5", "0.36", "16104", "16104"),
    ("question_disjoint", 0.0125, "min_cost", "sc:3", "4/10", "72.5", "0.36", "16104", "16104"),
    ("question_disjoint", 0.015, "max_coverage", "sc:3", "7/10", "71.5", "0.20", "16129", "16129"),
    ("question_disjoint", 0.015, "min_cost", "sc:3", "7/10", "71.5", "0.20", "16129", "16129"),
    ("question_disjoint", 0.02, "max_coverage", "asc:2:3", "6/10", "81.0", "0.62", "13727", "13727"),
    ("question_disjoint", 0.02, "min_cost", "sc:2", "3/10", "75.2", "0.34", "10643", "10643"),
]


def test_table8(main_result):
    out = table8_modal_selection.compute(main_result)
    got = [(r["scheme"], r["alpha"], r["selector"], r["policy"], f"{r['splits']}/{r['of']}",
            f"{r['coverage']:.1f}", f"{r['risk']:.2f}", f"{r['tokens']:.0f}", f"{r['weighted']:.0f}")
           for r in out["rows"]]
    assert got == TABLE8
    # The counts quoted under Table 8.
    counts = out["certified_splits"]
    assert {k: v for k, v in counts[0.005].items() if v} == {"sc:4": 5}
    assert counts[0.01]["sc:4"] == 15 and counts[0.01]["sc:3"] == 10
    assert max(v for k, v in counts[0.01].items() if k not in ("sc:3", "sc:4")) == 2
    others = [v for k, v in counts[0.015].items() if k not in ("sc:3", "sc:4")]
    assert counts[0.015]["sc:3"] == counts[0.015]["sc:4"] == 15 and (min(others), max(others)) == (1, 8)
    assert min(counts[0.02].values()) == 7


TABLE9 = [
    ("F0", "asc:2:3", 19, "79.0", "0.27", "13605", 1865, 461),
    ("ForwardPref", "asc:2:3", 19, "80.7", "0.53", "13145", 1860, 466),
    ("STaR", "asc:2:3", 16, "80.6", "0.54", "12927", 1863, 463),
    ("SelfDistill", "asc:2:3", 18, "79.0", "0.81", "12929", 1859, 467),
    ("iter2", "asc:2:3", 16, "80.0", "0.53", "14524", 1857, 469),
    ("A", "sc:3", 2, "71.8", "0.36", "16522", 1169, 1157),
    ("B", "sc:3", 2, "73.4", "0.35", "15481", 1150, 1176),
    ("C", "sc:3", 2, "71.5", "0.36", "17809", 1165, 1161),
    ("D", "asc:2:3", 22, "80.9", "0.75", "13896", 1167, 1159),
    ("E", "sc:3", 2, "68.5", "0.00", "16601", 1153, 1173),
    ("F", "asc:2:3", 22, "81.1", "1.06", "14184", 1159, 1167),
    ("G", "sc:3", 2, "70.3", "0.00", "16127", 1164, 1162),
    ("H", "sc:3", 2, "71.7", "0.00", "15760", 1163, 1163),
    ("I", "asc:3:3", 11, "79.6", "0.54", "17305", 1157, 1169),
    ("J", "sc:3", 2, "73.3", "0.35", "14603", 1161, 1165),
]


def test_table9(main_result):
    got = [(r["split"], r["policy"], r["certified"], f"{r['coverage']:.1f}", f"{r['risk']:.2f}",
            f"{r['weighted']:.0f}", r["calibration"], r["test"])
           for r in table9_split_selection.compute(main_result)]
    assert got == TABLE9


def test_table1():
    out = table1_prices.compute()
    assert [(f"{r['p1']:.1f}", f"{r['p0']:.1f}") for r in out["rates"]] == [
        ("70.1", "10.8"), ("73.1", "13.8"), ("63.9", "12.5"), ("80.2", "9.0"), ("79.5", "10.6")]
    assert [(f"{c['generated']:.0f}", f"{c['weighted']:.0f}") for c in out["costs"].values()] == [
        ("5384", "5384"), ("4801", "1008")]


def test_table19():
    got = [(f"{r['coverage']:.1f}", r["wrong"], f"{r['risk']:.2f}",
            None if r["generated"] is None else f"{r['generated']:.0f}",
            None if r["weighted"] is None else f"{r['weighted']:.0f}")
           for r in table19_operating_points.compute()]
    assert got == [
        ("75.8", 17, "0.96", None, None), ("68.3", 10, "0.63", None, None),
        ("71.3", 12, "0.72", None, None), ("77.3", 13, "0.72", "4801", "1008"),
        ("76.4", 12, "0.68", "5384", "5384"), ("77.3", 7, "0.39", "15744", "4365"),
        ("79.8", 10, "0.54", "13427", "13427"), ("71.6", 3, "0.18", "16152", "16152"),
        ("69.6", 0, "0.00", "21536", "21536")]


def test_table12():
    got = [(r["kept"], r["wrong"], r["problems"], f"{r['answer_level']:.3f}",
            f"{r['problem_count']:.3f}") for r in table12_risk_bounds.compute()]
    assert got == [
        (1469, 9, 350, "1.067", "1.849"), (1589, 10, 361, "1.065", "1.844"),
        (1665, 3, 341, "0.465", "1.197"), (1776, 12, 371, "1.092", "1.887"),
        (1799, 13, 378, "1.146", "1.938"), (1857, 10, 390, "0.912", "1.643"),
        (1798, 7, 375, "0.730", "1.452")]


def test_table14(grid_pairs):
    out = table14_price_fit.compute(grid_pairs)
    assert {k: f"{v:.2f}" for k, v in out["mae"].items()} == {
        "8B probe": "3.46", "8B vote": "3.95", "1.7B vote": "3.49", "All": "3.64"}
    assert {k: (f"{v['pred']:.1f}", f"{v['real']:.1f}") for k, v in out["cascades"].items()} == {
        "Cascade": ("70.8", "71.1"), "Small-first": ("76.0", "77.8")}


def test_table16(grid_pairs):
    s = table16_price_prediction.compute(grid_pairs)["summary"]
    assert (s["cascades"]["n"], s["all"]["n"]) == (70, 118)
    for key, cascades, schedules in (("cov_mae", "2.97", "3.86"), ("cov_spearman", "0.99", "0.97"),
                                     ("risk_mae", "0.97", "0.95"), ("risk_bias", "0.50", "0.52")):
        assert (f"{s['cascades'][key]:.2f}", f"{s['all'][key]:.2f}") == (cascades, schedules)
    assert f"{s['cov_mae_slope_per_action']:.2f}" == "-1.36"
    assert (f"{s['cascades']['tok_bias']:.1f}", f"{s['cascades']['flop_bias']:.1f}") == ("-3.0", "-4.9")
