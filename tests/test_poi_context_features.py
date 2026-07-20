from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.poi_context_features import (
    CONTEXT_BAR_COLUMNS,
    add_true_poi_aliases,
    build_feature_registry,
    build_true_poi_context_frame,
    validate_entry_feature_availability,
)


def _bars() -> pd.DataFrame:
    ny = pd.date_range("2024-01-03 07:00", periods=100, freq="min", tz="America/New_York")
    close = 100 + np.sin(np.arange(100) / 8) + np.arange(100) * 0.01
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    frame = pd.DataFrame(
        {
            "ts_event_utc": ny.tz_convert("UTC"),
            "ts_event_ny": ny,
            "trade_date_ny": pd.Timestamp("2024-01-03"),
            "product": "GC",
            "symbol": "GCG4",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.arange(100) + 100,
            "minute_of_day_ny": ny.hour * 60 + ny.minute,
            "day_of_week": "Wednesday",
            "continuous_segment_id": 1,
            "bar_range": high - low,
            "candle_body": np.abs(close - open_),
            "upper_wick": high - np.maximum(open_, close),
            "lower_wick": np.minimum(open_, close) - low,
            "true_range": high - low,
            "log_return": np.r_[np.nan, np.diff(np.log(close))],
            "abs_log_return": np.r_[np.nan, np.abs(np.diff(np.log(close)))],
            "rolling_atr_20m": 0.5,
            "rolling_atr_60m": 0.7,
            "rolling_realized_vol_60m": 0.01,
            "rolling_realized_vol_240m": 0.008,
            "relative_volume_60m": 1.0,
            "volume_zscore_240m": 0.0,
            "tradable_research_flag": True,
            "roll_window_flag": False,
        }
    )
    return frame[CONTEXT_BAR_COLUMNS]


def _signals() -> pd.DataFrame:
    rows = []
    for swing, mode, variant in ((3, "wick", "V1"), (5, "close", "V2")):
        rows.append(
            {
                "canonical_poi_id": "CP1",
                "canonical_retest_id": "CR1",
                "canonical_candidate_id": f"CC{variant}",
                "canonical_retest_number": 1,
                "poi_variant_id": variant,
                "candidate_variant_id": f"C{variant}",
                "retest_bar_id": 50,
                "retest_ts_event_utc": pd.Timestamp("2024-01-03 12:50", tz="UTC"),
                "retest_ts_event_ny": pd.Timestamp("2024-01-03 07:50", tz="America/New_York"),
                "direction": "bullish",
                "trade_date_ny": pd.Timestamp("2024-01-03"),
                "poi_low": 100.0,
                "poi_high": 101.0,
                "poi_mid": 100.5,
                "poi_size_ticks": 10,
                "poi_geometry_case": "case_2_standard",
                "fvg_size_ticks": 4,
                "close_open_gap_ticks": 2,
                "prior_bar_id": 38,
                "poi_bar_id": 39,
                "confirmation_bar_id": 40,
                "activation_bar_id": 41,
                "displacement_start_bar_id": 30,
                "swing_n": swing,
                "break_mode": mode,
                "structural_swing_break_flag": swing == 3,
                "structural_swing_window_broken": 15 if swing == 3 else np.nan,
                "structural_break_distance_ticks": 4.0,
                "execution_window_label": "New York Execution",
                "time_since_previous_touch_minutes": np.nan,
                "time_since_poi_activation_minutes": 9.0,
            }
        )
    return pd.DataFrame(rows)


class PoiContextFeatureTests(unittest.TestCase):
    def test_true_aliases_are_exact(self):
        frame = add_true_poi_aliases(_signals())
        self.assertTrue(frame["true_poi_id"].eq(frame["canonical_poi_id"]).all())
        self.assertTrue(frame["true_retest_id"].eq(frame["canonical_retest_id"]).all())

    def test_true_poi_retest_deduplication_and_variant_trace(self):
        context = build_true_poi_context_frame(_signals(), _bars())
        self.assertEqual(len(context), 1)
        self.assertEqual(int(context.loc[0, "feat_research_variant_count"]), 2)
        self.assertTrue(context["true_retest_id"].is_unique)
        self.assertTrue(bool(context.loc[0, "feat_has_swing_3_variant"]))
        self.assertTrue(bool(context.loc[0, "feat_has_swing_5_variant"]))

    def test_availability_timestamps_are_ordered(self):
        context = build_true_poi_context_frame(_signals(), _bars())
        self.assertLess(context.loc[0, "diag_formation_availability_bar_id"], 50)
        self.assertEqual(context.loc[0, "diag_pretouch_availability_bar_id"], 49)
        self.assertEqual(context.loc[0, "diag_touch_close_availability_bar_id"], 50)

    def test_touch_close_features_rejected_for_same_bar_entry(self):
        registry = build_feature_registry()
        with self.assertRaises(ValueError):
            validate_entry_feature_availability(
                ["feat_touch_rejection_wick_to_body_ratio"], "boundary_touch", registry
            )

    def test_zero_denominators_become_missing(self):
        bars = _bars()
        bars.loc[39, ["open", "high", "low", "close"]] = 100.0
        context = build_true_poi_context_frame(_signals(), bars)
        self.assertTrue(pd.isna(context.loc[0, "feat_middle_body_to_range_ratio"]))

    def test_bullish_bearish_direction_neutral_symmetry(self):
        bull_bars = _bars()
        bull = build_true_poi_context_frame(_signals(), bull_bars)
        bear_bars = bull_bars.copy()
        old_open, old_high = bear_bars["open"].copy(), bear_bars["high"].copy()
        old_low, old_close = bear_bars["low"].copy(), bear_bars["close"].copy()
        bear_bars["open"] = 200 - old_open
        bear_bars["high"] = 200 - old_low
        bear_bars["low"] = 200 - old_high
        bear_bars["close"] = 200 - old_close
        signals = _signals().copy()
        signals["direction"] = "bearish"
        signals["canonical_poi_id"] = "CP2"
        signals["canonical_retest_id"] = "CR2"
        signals["poi_low"], signals["poi_high"] = 99.0, 100.0
        signals["poi_mid"] = 99.5
        bear = build_true_poi_context_frame(signals, bear_bars)
        self.assertAlmostEqual(
            bull.loc[0, "feat_touch_penetration_fraction"],
            bear.loc[0, "feat_touch_penetration_fraction"],
        )
        self.assertAlmostEqual(
            bull.loc[0, "feat_displacement_efficiency"],
            bear.loc[0, "feat_displacement_efficiency"],
        )


if __name__ == "__main__":
    unittest.main()
