from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.poi_selection_refinement import (
    Section6CConfig,
    evaluate_refined_poi_proposals,
    price_to_tick_index,
    save_section6c_tables,
    select_final_refined_pois,
)
from src.features.poi_signal_features import (
    annotate_poi_structural_validation,
    build_candidate_trade_table,
    build_retest_table,
)


def _formation_bars(
    direction: str = "bullish",
    geometry_case: int = 2,
    fvg_ticks: int = 3,
    confirmation_passes: bool = True,
) -> pd.DataFrame:
    ny = pd.date_range("2024-01-03 03:00", periods=3, freq="min", tz="America/New_York")
    if direction == "bullish":
        a = [99.5, 100.0, 99.0, 99.8]
        b = [99.9, 101.0, 99.8, 100.7]
        c_low = 100.0 + fvg_ticks * 0.1
        c_open = 101.2 if geometry_case == 1 else 100.8
        c_close = c_open + 0.2 if confirmation_passes else c_open - 0.1
        c = [c_open, max(c_open, c_close) + 0.2, min(c_low, c_open, c_close), c_close]
    else:
        a = [101.5, 102.0, 101.0, 101.2]
        b = [101.1, 102.0, 100.0, 100.3]
        c_high = 101.0 - fvg_ticks * 0.1
        c_open = 99.8 if geometry_case == 1 else 100.2
        c_close = c_open - 0.2 if confirmation_passes else c_open + 0.1
        c = [c_open, max(c_high, c_open, c_close), min(c_open, c_close) - 0.2, c_close]
    values = np.array([a, b, c], dtype="float64")
    frame = pd.DataFrame(
        {
            "bar_id": np.arange(3, dtype="int32"),
            "ts_event_utc": ny.tz_convert("UTC"),
            "ts_event_ny": ny,
            "trade_date_ny": pd.Timestamp("2024-01-03"),
            "product": "GC",
            "symbol": "GCG4",
            "open": values[:, 0],
            "high": values[:, 1],
            "low": values[:, 2],
            "close": values[:, 3],
            "volume": 100.0,
            "continuous_segment_id": 1,
            "tradable_research_flag": True,
            "roll_window_flag": False,
            "is_regular_1m_bar": True,
            "same_symbol_as_previous_bar": True,
            "is_poi_search_window": True,
            "is_execution_window": True,
            "execution_window_label": "London Execution",
            "minute_of_day_ny": [180, 181, 182],
            "rolling_atr_60m": 1.0,
            "forward_return_5m": 0.001,
            "forward_return_15m": 0.001,
            "forward_return_30m": 0.001,
            "forward_return_60m": 0.001,
        }
    )
    return frame


def _proposal(direction: str = "bullish", poi_id: str = "P1", **updates: object) -> pd.DataFrame:
    record = {
        "poi_id": poi_id,
        "product": "GC",
        "trade_date_ny": pd.Timestamp("2024-01-03"),
        "direction": direction,
        "swing_n": 3,
        "break_mode": "wick",
        "poi_middle_bar_id": 1,
        "poi_activation_bar_id": 2,
        "break_bar_id": 2,
        "displacement_start_bar_id": 0,
        "broken_swing_bar_id": 0,
        "poi_activation_minute_ny": 182,
    }
    record.update(updates)
    return pd.DataFrame([record])


def _evaluated(
    direction: str = "bullish",
    geometry_case: int = 2,
    fvg_ticks: int = 3,
    confirmation_passes: bool = True,
    **proposal_updates: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = _formation_bars(direction, geometry_case, fvg_ticks, confirmation_passes)
    proposal = _proposal(direction, **proposal_updates)
    return bars, evaluate_refined_poi_proposals(bars, proposal, Section6CConfig())


class Section6CPoiSelectionRefinementTests(unittest.TestCase):
    def test_01_bullish_case1_expands(self):
        _, out = _evaluated("bullish", 1)
        self.assertEqual(str(out.loc[0, "poi_geometry_case"]), "case_1_expanded")
        self.assertEqual(out.loc[0, "poi_high"], out.loc[0, "confirmation_open"])

    def test_02_bearish_case1_expands(self):
        _, out = _evaluated("bearish", 1)
        self.assertEqual(str(out.loc[0, "poi_geometry_case"]), "case_1_expanded")
        self.assertEqual(out.loc[0, "poi_low"], out.loc[0, "confirmation_open"])

    def test_03_bullish_case2_keeps_middle_range(self):
        _, out = _evaluated("bullish", 2)
        self.assertEqual(str(out.loc[0, "poi_geometry_case"]), "case_2_standard")
        self.assertEqual(out.loc[0, "poi_high"], out.loc[0, "middle_high"])

    def test_04_bearish_case2_keeps_middle_range(self):
        _, out = _evaluated("bearish", 2)
        self.assertEqual(str(out.loc[0, "poi_geometry_case"]), "case_2_standard")
        self.assertEqual(out.loc[0, "poi_low"], out.loc[0, "middle_low"])

    def test_05_bullish_confirmation_close_inside_gap_rejected(self):
        _, out = _evaluated("bullish", 2, confirmation_passes=False)
        self.assertFalse(bool(out.loc[0, "formation_valid_flag"]))
        self.assertEqual(
            str(out.loc[0, "rejection_reason"]), "confirmation_close_inside_opening_gap"
        )

    def test_06_bearish_confirmation_close_inside_gap_rejected(self):
        _, out = _evaluated("bearish", 2, confirmation_passes=False)
        self.assertFalse(bool(out.loc[0, "formation_valid_flag"]))
        self.assertEqual(
            str(out.loc[0, "rejection_reason"]), "confirmation_close_inside_opening_gap"
        )

    def test_07_no_classic_fvg_rejected(self):
        _, out = _evaluated("bullish", 2, fvg_ticks=0)
        self.assertFalse(bool(out.loc[0, "classic_fvg_flag"]))
        self.assertEqual(str(out.loc[0, "rejection_reason"]), "no_classic_fvg")

    def test_08_opening_gap_without_fvg_rejected(self):
        _, out = _evaluated("bearish", 2, fvg_ticks=0)
        self.assertTrue(bool(out.loc[0, "close_open_gap_flag"]))
        self.assertEqual(str(out.loc[0, "rejection_reason"]), "no_classic_fvg")

    def test_09_fvg_without_opening_gap_rejected(self):
        bars = _formation_bars("bullish", 2, 3, True)
        bars.loc[2, "open"] = bars.loc[1, "close"]
        bars.loc[2, "close"] = bars.loc[2, "open"] + 0.2
        bars.loc[2, "high"] = max(bars.loc[2, "high"], bars.loc[2, "close"])
        out = evaluate_refined_poi_proposals(bars, _proposal("bullish"))
        self.assertEqual(str(out.loc[0, "rejection_reason"]), "no_close_open_gap")

    def test_10_two_tick_fvg_fails_default_threshold(self):
        _, out = _evaluated("bullish", 2, 2)
        self.assertTrue(bool(out.loc[0, "formation_valid_flag"]))
        self.assertFalse(bool(out.loc[0, "final_poi_valid_flag"]))

    def test_11_apparent_2999999_ticks_rounds_safely(self):
        self.assertEqual(
            int(price_to_tick_index([100.29999999999997])[0] - price_to_tick_index([100.0])[0]), 3
        )

    def test_12_exactly_three_ticks_passes(self):
        _, out = _evaluated("bullish", 2, 3)
        self.assertTrue(bool(out.loc[0, "final_poi_valid_flag"]))

    def test_13_three_tick_flags(self):
        _, out = _evaluated("bullish", 2, 3)
        self.assertEqual(
            out.loc[0, ["fvg_ge_3tick_flag", "fvg_ge_4tick_flag", "fvg_ge_5tick_flag"]].to_list(),
            [True, False, False],
        )

    def test_14_four_tick_flags(self):
        _, out = _evaluated("bullish", 2, 4)
        self.assertEqual(
            out.loc[0, ["fvg_ge_3tick_flag", "fvg_ge_4tick_flag", "fvg_ge_5tick_flag"]].to_list(),
            [True, True, False],
        )

    def test_15_five_tick_flags(self):
        _, out = _evaluated("bullish", 2, 5)
        self.assertTrue(
            out.loc[0, ["fvg_ge_3tick_flag", "fvg_ge_4tick_flag", "fvg_ge_5tick_flag"]].all()
        )

    def test_16_segment_boundary_rejected(self):
        bars = _formation_bars()
        bars.loc[2, "continuous_segment_id"] = 2
        out = evaluate_refined_poi_proposals(bars, _proposal())
        self.assertEqual(str(out.loc[0, "rejection_reason"]), "segment_boundary_crossed")

    def test_17_contract_switch_rejected(self):
        bars = _formation_bars()
        bars.loc[2, "symbol"] = "GCJ4"
        out = evaluate_refined_poi_proposals(bars, _proposal())
        self.assertEqual(str(out.loc[0, "rejection_reason"]), "contract_switch_crossed")

    def test_18_missing_minute_rejected(self):
        bars = _formation_bars()
        bars.loc[2, "ts_event_utc"] += pd.Timedelta(minutes=1)
        bars.loc[2, "ts_event_ny"] += pd.Timedelta(minutes=1)
        out = evaluate_refined_poi_proposals(bars, _proposal())
        self.assertEqual(str(out.loc[0, "rejection_reason"]), "non_consecutive_bars")

    def test_19_retests_begin_strictly_after_activation(self):
        bars, poi = _evaluated("bullish", 2, 3)
        extra = pd.concat([bars.iloc[[-1]].copy(), bars.iloc[[-1]].copy()], ignore_index=True)
        extra["bar_id"] = [3, 4]
        extra["ts_event_utc"] = pd.date_range(
            bars.loc[2, "ts_event_utc"] + pd.Timedelta(minutes=1), periods=2, freq="min"
        )
        extra["ts_event_ny"] = extra["ts_event_utc"].dt.tz_convert("America/New_York")
        extra["minute_of_day_ny"] = [183, 184]
        extra["low"] = 100.7
        extra["high"] = 101.1
        full_bars = pd.concat([bars, extra], ignore_index=True)
        retests = build_retest_table(
            full_bars, select_final_refined_pois(poi), Section6CConfig(min_stop_ticks=1)
        )
        self.assertTrue(retests["retest_bar_id"].gt(2).all())

    def test_20_case1_changes_boundary_midpoint_and_distal(self):
        _, poi = _evaluated("bullish", 1, 3)
        row = poi.iloc[0]
        self.assertEqual(row["poi_high"], row["confirmation_open"])
        self.assertEqual(row["poi_mid"], (row["poi_low"] + row["poi_high"]) / 2)
        self.assertEqual(row["poi_low"], row["middle_low"])

    def test_21_stops_and_targets_use_corrected_geometry(self):
        _, poi = _evaluated("bullish", 1, 3)
        poi = select_final_refined_pois(poi)
        row = poi.iloc[0]
        retest = pd.DataFrame(
            {
                "retest_id": ["R1"],
                "poi_id": [row["poi_id"]],
                "direction": ["bullish"],
                "retest_bar_id": [3],
                "retest_ts_event_utc": [pd.Timestamp("2024-01-03 08:03", tz="UTC")],
                "retest_high": [row["poi_high"] + 0.1],
                "retest_low": [row["poi_low"] - 0.1],
                "retest_close": [row["poi_mid"]],
                "retest_forward_return_5m": [0.001],
                "retest_forward_return_15m": [0.001],
                "retest_forward_return_30m": [0.001],
                "retest_forward_return_60m": [0.001],
            }
        )
        candidates = build_candidate_trade_table(
            poi, retest, Section6CConfig(min_stop_ticks=0, max_stop_ticks=1000)
        )
        boundary = candidates.query(
            "entry_variant == 'boundary' and stop_model == 'poi_distal_edge'"
        ).iloc[0]
        self.assertEqual(boundary["entry_price"], row["poi_high"])
        self.assertEqual(boundary["stop_price"], row["poi_low"] - 0.1)
        self.assertAlmostEqual(
            boundary["target_1R_price"], boundary["entry_price"] + boundary["risk_points"]
        )

    def test_22_structural_annotations_preserve_lineage(self):
        _, poi = _evaluated("bullish", 2, 3)
        poi = select_final_refined_pois(poi)
        events = pd.DataFrame(
            {
                "continuous_segment_id": [1],
                "direction": ["bullish"],
                "structural_break_mode": ["wick"],
                "structural_swing_window_broken": [15],
                "structural_swing_bar_id": [0],
                "structural_swing_price": [100.0],
                "structural_break_bar_id": [1],
                "structural_break_distance_ticks": [2.0],
            }
        )
        annotated = annotate_poi_structural_validation(
            poi, events, Section6CConfig(structural_swing_windows=(15,))
        )
        self.assertEqual(annotated.loc[0, "poi_variant_id"], poi.loc[0, "poi_variant_id"])
        self.assertTrue(bool(annotated.loc[0, "structural_swing_break_flag"]))

    def test_23_five_tick_cohort_strict_subset_of_four(self):
        rows = pd.concat(
            [_evaluated("bullish", 2, tick)[1] for tick in (3, 4, 5)], ignore_index=True
        )
        set4 = set(rows.index[rows["fvg_ge_4tick_flag"]])
        set5 = set(rows.index[rows["fvg_ge_5tick_flag"]])
        self.assertTrue(set5 < set4)

    def test_24_four_tick_cohort_strict_subset_of_three(self):
        rows = pd.concat(
            [_evaluated("bullish", 2, tick)[1] for tick in (3, 4, 5)], ignore_index=True
        )
        set3 = set(rows.index[rows["fvg_ge_3tick_flag"]])
        set4 = set(rows.index[rows["fvg_ge_4tick_flag"]])
        self.assertTrue(set4 < set3)

    def test_25_canonical_ids_collapse_swing_break_duplicates(self):
        bars = _formation_bars()
        proposals = pd.concat(
            [_proposal("bullish", "P1"), _proposal("bullish", "P2", swing_n=5, break_mode="close")],
            ignore_index=True,
        )
        out = evaluate_refined_poi_proposals(bars, proposals)
        self.assertEqual(out["canonical_poi_id"].nunique(), 1)
        self.assertEqual(out["poi_variant_id"].nunique(), 2)

    def test_26_rejected_formations_do_not_enter_final_table(self):
        _, rejected = _evaluated("bullish", 2, 3, False)
        self.assertTrue(select_final_refined_pois(rejected).empty)

    def test_27_section6c_save_does_not_overwrite_legacy_files(self):
        empty = pd.DataFrame()
        tables = {
            key: empty
            for key in (
                "refined_poi_audit",
                "refined_poi_table",
                "refined_retest_table",
                "refined_candidate_trade_table",
                "refined_signal_frame",
                "refinement_summary",
            )
        }
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "section6_poi_table_gc.parquet"
            legacy.write_bytes(b"legacy")
            save_section6c_tables(tables, tmp)
            self.assertEqual(legacy.read_bytes(), b"legacy")

    def test_28_point_and_tick_measurements_are_consistent(self):
        _, out = _evaluated("bullish", 2, 5)
        self.assertAlmostEqual(out.loc[0, "fvg_size_points"], out.loc[0, "fvg_size_ticks"] * 0.1)

    def test_29_binary_float_does_not_change_threshold(self):
        bars = _formation_bars("bullish", 2, 3)
        bars.loc[2, "low"] = 100.29999999999997
        out = evaluate_refined_poi_proposals(bars, _proposal())
        self.assertEqual(int(out.loc[0, "fvg_size_ticks"]), 3)
        self.assertTrue(bool(out.loc[0, "fvg_threshold_pass"]))


if __name__ == "__main__":
    unittest.main()
