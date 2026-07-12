from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.features.poi_event_study import (
    SECTION7_BAR_REQUIRED_COLUMNS,
    SECTION7_SIGNAL_REQUIRED_COLUMNS,
    Section7Config,
    build_bar_horizon_path_features,
    build_section7_validation,
    build_section7_poi_selection_audit_batches,
    build_section7_visual_audit_batches,
    build_signal_horizon_metrics,
    prepare_section7_bar_frame,
    prepare_section7_signal_frame,
    summarize_section7_groups,
    EventStudyGroupSpec,
)
from src.features.poi_event_study_refined import (
    build_section7r_event_study,
    prepare_section7r_signal_populations,
)


def _synthetic_research_bars() -> pd.DataFrame:
    ts_ny = pd.date_range("2024-01-03 11:55", "2024-01-03 15:30", freq="min", tz="America/New_York")
    ts_utc = ts_ny.tz_convert("UTC")
    close = 2000.0 + np.arange(len(ts_ny), dtype="float64") * 0.10
    high = close + 0.05
    low = close - 0.05
    return pd.DataFrame(
        {
            "ts_event_utc": ts_utc,
            "ts_event_ny": ts_ny,
            "trade_date_ny": pd.Timestamp("2024-01-03"),
            "product": "GC",
            "high": high,
            "low": low,
            "close": close,
            "minute_of_day_ny": ts_ny.hour * 60 + ts_ny.minute,
            "continuous_segment_id": 1,
            "tradable_research_flag": True,
            "roll_window_flag": False,
        }
    )[SECTION7_BAR_REQUIRED_COLUMNS]


def _synthetic_signal_frame() -> pd.DataFrame:
    ts_ny = pd.Timestamp("2024-01-03 12:00", tz="America/New_York")
    record = {
        "signal_id": "GC_SIG_00000001",
        "candidate_trade_id": "GC_CAND_00000001",
        "retest_id": "GC_20240103_rt001",
        "poi_id": "GC_20240103_sw3_wick_bull_001",
        "direction": "bullish",
        "trade_side": "long",
        "entry_variant": "boundary",
        "stop_model": "poi_distal_edge",
        "entry_price": 2000.50,
        "stop_price": 1998.00,
        "risk_points": 2.50,
        "stop_ticks": 25.0,
        "valid_candidate_flag": True,
        "retest_bar_id": 5,
        "retest_ts_event_utc": ts_ny.tz_convert("UTC"),
        "retest_ts_event_ny": ts_ny,
        "retest_minute_ny": 720,
        "execution_window_label": "New York Execution",
        "trade_date_ny": pd.Timestamp("2024-01-03"),
        "swing_n": 3,
        "break_mode": "wick",
        "retest_number": 1,
        "first_touch_flag": True,
        "retest_high": 2000.55,
        "retest_low": 2000.45,
        "full_poi_cross_flag": False,
        "time_since_poi_activation_minutes": 30.0,
        "candles_since_poi_activation": 30,
        "structural_swing_break_flag": True,
        "structural_swing_window_broken": 31,
        "structural_swing_windows_broken": "15,21,31",
        "structural_break_mode": "wick",
        "local_swing_only_flag": False,
        "retest_trend_bias_20_50": "bullish",
        "retest_distance_from_vwap_ticks": 12.0,
        "retest_vwap_slope_15m_ticks": 1.0,
        "retest_relative_volume_60m": 1.2,
        "retest_volume_zscore_240m": 0.8,
        "retest_rolling_atr_60m": 2.0,
        "retest_volatility_regime": "normal",
        "retest_large_move_flag": False,
        "retest_volume_spike_flag": False,
        "retest_directional_efficiency_20m": 0.6,
        "fvg_size_ticks": 4.0,
        "abs_close_open_gap_ticks": 2.0,
        "poi_size_ticks": 8.0,
        "break_distance_ticks": 12.0,
        "displacement_range_ticks": 50.0,
        "displacement_candles": 12,
        "forward_5m_r": 0.1,
        "forward_15m_r": 0.2,
        "forward_30m_r": 0.3,
        "forward_60m_r": 0.4,
    }
    return pd.DataFrame([record])


class Section7EventStudyTests(unittest.TestCase):
    @staticmethod
    def _refined_duplicate_signal_frame() -> pd.DataFrame:
        base = _synthetic_signal_frame().iloc[0].to_dict()
        base.update(
            {
                "canonical_poi_id": "GC_POI_BULL_00000003_00000004_00000005",
                "poi_variant_id": base["poi_id"],
                "canonical_retest_id": "GC_POI_BULL_00000003_00000004_00000005_RT_00000005",
                "canonical_retest_number": 1,
                "candidate_variant_id": base["candidate_trade_id"],
                "canonical_candidate_id": "CANONICAL_CANDIDATE_1",
                "poi_definition_version": "6C_v1",
                "poi_geometry_case": "case_2_standard",
                "poi_zone_expanded_flag": False,
                "poi_zone_expansion_points": 0.0,
                "poi_zone_expansion_ticks": 0,
                "fvg_size_points": 0.5,
                "fvg_size_ticks": 5.0,
                "fvg_ge_3tick_flag": True,
                "fvg_ge_4tick_flag": True,
                "fvg_ge_5tick_flag": True,
            }
        )
        duplicate = dict(base)
        duplicate.update(
            {
                "signal_id": "GC6C_SIG_00000002",
                "candidate_trade_id": "GC6C_CAND_00000002",
                "candidate_variant_id": "GC6C_CAND_00000002",
                "poi_id": "GC_20240103_sw5_close_bull_001",
                "poi_variant_id": "GC_20240103_sw5_close_bull_001",
                "swing_n": 5,
                "break_mode": "close",
            }
        )
        return pd.DataFrame([base, duplicate])

    def test_section7r_canonical_population_collapses_correlated_variants(self):
        populations = prepare_section7r_signal_populations(
            self._refined_duplicate_signal_frame(),
            Section7Config(forward_horizons=(60,), min_ranking_sample_size=1),
        )
        self.assertEqual(len(populations["raw_variant_rows"]), 2)
        self.assertEqual(len(populations["canonical_candidate_variants"]), 1)
        self.assertTrue(populations["canonical_candidate_variants"]["canonical_candidate_id"].is_unique)

    def test_section7r_reports_geometry_threshold_and_population_counts(self):
        outputs = build_section7r_event_study(
            self._refined_duplicate_signal_frame(),
            _synthetic_research_bars(),
            Section7Config(forward_horizons=(60,), min_ranking_sample_size=1),
        )
        summary = outputs["section7r_event_study_summary"]
        self.assertTrue({"raw_variant_rows", "canonical_candidate_variants"}.issubset(summary["analysis_population"]))
        self.assertTrue({"poi_geometry_case", "fvg_min_3_ticks", "fvg_min_4_ticks", "fvg_min_5_ticks"}.issubset(summary["group_spec"]))
        baseline = summary.query("group_spec == 'baseline_all_candidates' and metric_mode == 'fixed'")
        counts = baseline.set_index("analysis_population")["candidate_rows"].to_dict()
        self.assertEqual(counts["raw_variant_rows"], 2)
        self.assertEqual(counts["canonical_candidate_variants"], 1)

    def test_validation_accepts_structural_section6b_signal_frame(self):
        validation = build_section7_validation(
            _synthetic_signal_frame(),
            _synthetic_research_bars(),
            Section7Config(),
        )

        self.assertTrue(validation["signal_required_columns_present"])
        self.assertTrue(validation["structural_partition_valid"])
        self.assertTrue(validation["no_candidate_entry_after_1200_ny"])
        self.assertTrue(validation["entry_rows_in_approved_execution_windows"])

    def test_fixed_and_capped_horizon_respect_1530_forced_exit(self):
        cfg = Section7Config(forward_horizons=(240,))
        bars = prepare_section7_bar_frame(_synthetic_research_bars())
        signals = prepare_section7_signal_frame(_synthetic_signal_frame(), cfg)
        path = build_bar_horizon_path_features(bars, 240, cfg)

        fixed = build_signal_horizon_metrics(signals, path, 240, "fixed", cfg)
        capped = build_signal_horizon_metrics(signals, path, 240, "capped", cfg)

        self.assertFalse(bool(fixed["valid"].iloc[0]))
        self.assertTrue(bool(capped["valid"].iloc[0]))
        self.assertTrue(bool(capped["forced_exit_flag"].iloc[0]))
        self.assertGreater(capped["forward_r"].iloc[0], 0)

    def test_group_summary_exposes_uncapped_and_capped_runner_metrics(self):
        cfg = Section7Config(forward_horizons=(60,))
        bars = prepare_section7_bar_frame(_synthetic_research_bars())
        signals = prepare_section7_signal_frame(_synthetic_signal_frame(), cfg)
        path = build_bar_horizon_path_features(bars, 60, cfg)
        metrics = build_signal_horizon_metrics(signals, path, 60, "fixed", cfg)

        summary = summarize_section7_groups(
            signal_base=signals,
            metrics=metrics,
            group_specs=[EventStudyGroupSpec("baseline")],
            horizon=60,
            metric_mode="fixed",
            config=cfg,
        )

        self.assertEqual(int(summary["valid_count"].iloc[0]), 1)
        self.assertIn("mean_r_cap_5", summary.columns)
        self.assertIn("hit_pos_1r_rate", summary.columns)
        self.assertIn("hit_pos_15r_rate", summary.columns)

    def test_visual_audit_batch_writes_chart_manifest(self):
        bars = _synthetic_research_bars().copy()
        bars["open"] = bars["close"].shift(1).fillna(bars["close"])
        bars = bars[
            [
                "ts_event_utc",
                "ts_event_ny",
                "trade_date_ny",
                "product",
                "open",
                "high",
                "low",
                "close",
                "minute_of_day_ny",
                "continuous_segment_id",
                "tradable_research_flag",
                "roll_window_flag",
            ]
        ]

        base = _synthetic_signal_frame().iloc[0].to_dict()
        base.update(
            {
                "poi_created_ts_event_ny": pd.Timestamp(
                    "2024-01-03 11:58",
                    tz="America/New_York",
                ),
                "poi_activation_ts_event_ny": pd.Timestamp(
                    "2024-01-03 11:59",
                    tz="America/New_York",
                ),
                "poi_low": 2000.25,
                "poi_high": 2000.75,
                "target_1R_price": 2003.00,
                "target_2R_price": 2005.50,
                "target_3R_price": 2008.00,
                "target_5R_price": 2013.00,
            }
        )
        london_short = dict(base)
        london_short.update(
            {
                "signal_id": "GC_SIG_LONDON_SHORT",
                "candidate_trade_id": "GC_CAND_LONDON_SHORT",
                "execution_window_label": "London Execution",
                "trade_side": "short",
                "entry_variant": "boundary",
                "retest_volatility_regime": "extreme",
            }
        )
        ny_short = dict(london_short)
        ny_short.update(
            {
                "signal_id": "GC_SIG_NY_SHORT",
                "candidate_trade_id": "GC_CAND_NY_SHORT",
                "execution_window_label": "New York Execution",
            }
        )
        elevated_long = dict(base)
        elevated_long.update(
            {
                "signal_id": "GC_SIG_ELEVATED_LONG",
                "candidate_trade_id": "GC_CAND_ELEVATED_LONG",
                "execution_window_label": "New York Execution",
                "trade_side": "long",
                "entry_variant": "boundary",
                "retest_volatility_regime": "elevated",
            }
        )
        signals = pd.DataFrame([london_short, ny_short, elevated_long])

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = build_section7_visual_audit_batches(
                signal_frame=signals,
                research_bars=bars,
                output_dir=tmp_dir,
                samples_per_batch=1,
                random_state=1,
                min_sample_size=1,
            )

            plotted = manifest.loc[manifest["status"].eq("plotted")]
            self.assertEqual(len(plotted), 3)
            self.assertTrue(
                all(Path(path).exists() for path in plotted["chart_path"].to_list())
            )

    def test_poi_selection_audit_writes_index_and_charts(self):
        bars = _synthetic_research_bars().copy()
        bars["open"] = bars["close"].shift(1).fillna(bars["close"])
        bars = bars[
            [
                "ts_event_utc",
                "ts_event_ny",
                "trade_date_ny",
                "product",
                "open",
                "high",
                "low",
                "close",
                "minute_of_day_ny",
                "continuous_segment_id",
                "tradable_research_flag",
                "roll_window_flag",
            ]
        ]

        base = _synthetic_signal_frame().iloc[0].to_dict()
        base.update(
            {
                "poi_id": "GC_POI_STRUCT",
                "poi_middle_bar_id": 4,
                "poi_confirm_bar_id": 5,
                "poi_activation_bar_id": 5,
                "break_bar_id": 9,
                "poi_middle_ts_event_ny": pd.Timestamp(
                    "2024-01-03 11:59",
                    tz="America/New_York",
                ),
                "poi_created_ts_event_ny": pd.Timestamp(
                    "2024-01-03 12:00",
                    tz="America/New_York",
                ),
                "poi_activation_ts_event_ny": pd.Timestamp(
                    "2024-01-03 12:00",
                    tz="America/New_York",
                ),
                "break_ts_event_ny": pd.Timestamp(
                    "2024-01-03 12:04",
                    tz="America/New_York",
                ),
                "poi_low": 2000.25,
                "poi_high": 2000.75,
                "structural_swing_break_flag": True,
                "structural_swing_window_broken": 15,
                "local_swing_only_flag": False,
            }
        )
        local = dict(base)
        local.update(
            {
                "signal_id": "GC_SIG_LOCAL_POI",
                "poi_id": "GC_POI_LOCAL",
                "direction": "bearish",
                "structural_swing_break_flag": False,
                "structural_swing_window_broken": pd.NA,
                "local_swing_only_flag": True,
            }
        )
        signals = pd.DataFrame([base, local])

        with tempfile.TemporaryDirectory() as tmp_dir:
            index = build_section7_poi_selection_audit_batches(
                signal_frame=signals,
                research_bars=bars,
                output_dir=tmp_dir,
                samples_per_group=1,
                random_state=2,
            )

            plotted = index.loc[index["status"].eq("plotted")]
            self.assertEqual(len(plotted), 2)
            self.assertTrue((Path(tmp_dir) / "poi_selection_audit_index.csv").exists())
            self.assertTrue(
                all(Path(path).exists() for path in plotted["image_path"].to_list())
            )
            self.assertIn("manual_review_status", index.columns)
            self.assertIn("manual_notes", index.columns)


if __name__ == "__main__":
    unittest.main()
