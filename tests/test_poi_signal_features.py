from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd

from src.features.poi_signal_features import (
    SECTION6_REQUIRED_COLUMNS,
    Section6Config,
    build_section6b_structural_tables,
    build_section6_signal_tables,
    save_section6_tables,
)


def _synthetic_research_bars() -> pd.DataFrame:
    ts_utc = pd.date_range("2024-01-03 08:00", periods=14, freq="min", tz="UTC")
    ts_ny = ts_utc.tz_convert("America/New_York")
    open_ = np.array(
        [100.0, 100.1, 99.6, 100.2, 101.8, 102.0, 102.5, 103.6, 105.5, 105.8, 104.8, 104.4, 104.0, 103.2]
    )
    high = np.array(
        [100.5, 100.3, 100.4, 102.0, 105.0, 103.0, 104.0, 104.5, 106.0, 105.9, 104.9, 104.7, 104.2, 104.0]
    )
    low = np.array(
        [99.8, 99.0, 99.4, 100.0, 101.7, 101.8, 102.3, 103.0, 105.2, 104.6, 104.3, 103.7, 102.8, 102.9]
    )
    close = np.array(
        [100.2, 99.5, 100.2, 101.8, 102.0, 102.5, 103.5, 104.0, 105.8, 104.8, 104.4, 104.0, 103.2, 103.5]
    )
    volume = np.array([100, 90, 80, 120, 200, 150, 180, 220, 500, 300, 260, 240, 230, 210])

    df = pd.DataFrame(
        {
            "ts_event_utc": ts_utc,
            "product": "GC",
            "symbol": "GCG4",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "ts_event_ny": ts_ny,
            "trade_date_ny": pd.Timestamp("2024-01-03"),
            "day_of_week": "Wednesday",
            "hour_ny": ts_ny.hour,
            "minute_ny": ts_ny.minute,
            "minute_of_day_ny": ts_ny.hour * 60 + ts_ny.minute,
            "session_label": "London",
            "tradable_research_flag": True,
            "roll_window_flag": False,
            "roll_window_type": pd.NA,
            "is_roll_day": False,
            "is_pre_roll_day": False,
            "is_post_roll_day": False,
            "days_since_roll": 10,
            "days_to_next_roll": 10,
            "selected_contract_volume_share": 1.0,
            "ambiguous_roll_candidate": False,
            "low_liquidity_active_day_flag": False,
            "previous_symbol": "GCG4",
            "previous_close": np.r_[np.nan, close[:-1]],
            "previous_ts_event_utc": pd.Series(ts_utc).shift(1),
            "bar_gap_minutes": 1.0,
            "same_symbol_as_previous_bar": True,
            "is_regular_1m_bar": True,
            "return_1m": np.r_[np.nan, close[1:] / close[:-1] - 1.0],
            "log_return": np.r_[np.nan, np.diff(np.log(close))],
            "abs_log_return": np.r_[np.nan, np.abs(np.diff(np.log(close)))],
            "bar_range": high - low,
            "candle_body": np.abs(close - open_),
            "upper_wick": high - np.maximum(open_, close),
            "lower_wick": np.minimum(open_, close) - low,
            "true_range": high - low,
            "continuous_segment_id": 1,
            "rolling_atr_20m": high - low,
            "rolling_atr_60m": high - low,
            "rolling_realized_vol_60m": 0.001,
            "rolling_realized_vol_240m": 0.001,
            "rolling_high_low_range_60m": 6.0,
            "rolling_volume_60m": 1000.0,
            "rolling_avg_volume_60m": 100.0,
            "rolling_avg_volume_240m": 100.0,
            "rolling_std_volume_240m": 10.0,
            "relative_volume_60m": volume / 100.0,
            "volume_zscore_240m": (volume - 100.0) / 10.0,
            "low_liquidity_warning_flag": False,
            "forward_return_5m": 0.001,
            "forward_return_15m": 0.0015,
            "forward_return_30m": 0.002,
            "forward_return_60m": 0.0025,
            "forward_log_return_5m": 0.001,
            "forward_log_return_15m": 0.0015,
            "forward_log_return_30m": 0.002,
            "forward_log_return_60m": 0.0025,
        }
    )
    return df[SECTION6_REQUIRED_COLUMNS]


class Section6PoiSignalFeatureTests(unittest.TestCase):
    def test_engine_builds_bullish_poi_and_valid_candidates(self):
        config = Section6Config(
            swing_windows=(3,),
            break_modes=("wick",),
            min_stop_ticks=1.0,
            max_stop_ticks=100.0,
        )

        tables = build_section6_signal_tables(_synthetic_research_bars(), config)

        self.assertTrue(tables["section6_validation"].all())
        self.assertFalse(tables["poi_table"].empty)
        self.assertFalse(tables["retest_table"].empty)
        self.assertFalse(tables["candidate_trade_table"].empty)
        self.assertEqual(len(tables["signal_frame"]), len(tables["candidate_trade_table"]))

        bullish_pois = tables["poi_table"].query("direction == 'bullish'")
        self.assertFalse(bullish_pois.empty)
        self.assertTrue(bullish_pois["poi_high"].ge(104.5).any())
        self.assertTrue(bullish_pois["poi_low"].le(103.0).any())

        candidates = tables["candidate_trade_table"]
        self.assertTrue(candidates["valid_candidate_flag"].all())
        self.assertTrue(candidates["stop_ticks"].between(1.0, 100.0).all())

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_paths = save_section6_tables(tables, tmp_dir)
            self.assertEqual(
                set(output_paths),
                {
                    "poi_table",
                    "retest_table",
                    "candidate_trade_table",
                    "signal_frame",
                },
            )
            self.assertTrue(all(path.exists() for path in output_paths.values()))

    def test_structural_validation_marks_and_propagates_validated_pois(self):
        config = Section6Config(
            swing_windows=(3,),
            break_modes=("wick",),
            structural_swing_windows=(3,),
            min_stop_ticks=1.0,
            max_stop_ticks=100.0,
        )

        section6_tables = build_section6_signal_tables(_synthetic_research_bars(), config)
        section6b_tables = build_section6b_structural_tables(
            feature_frame=section6_tables["section6_feature_frame"],
            poi_table=section6_tables["poi_table"],
            retest_table=section6_tables["retest_table"],
            candidate_trade_table=section6_tables["candidate_trade_table"],
            signal_frame=section6_tables["signal_frame"],
            config=config,
        )

        self.assertTrue(section6b_tables["section6b_validation"].all())
        structural_pois = section6b_tables["structural_poi_table"]
        self.assertTrue(structural_pois["structural_swing_break_flag"].all())
        self.assertFalse(structural_pois["local_swing_only_flag"].any())
        self.assertEqual(int(structural_pois["structural_swing_window_broken"].iloc[0]), 3)
        self.assertEqual(structural_pois["structural_break_mode"].iloc[0], "wick")
        self.assertGreater(structural_pois["structural_break_distance_ticks"].iloc[0], 0)

        for key in (
            "structural_retest_table",
            "structural_candidate_trade_table",
            "structural_signal_frame",
        ):
            table = section6b_tables[key]
            self.assertIn("structural_swing_break_flag", table.columns)
            self.assertTrue(table["structural_swing_break_flag"].all())


if __name__ == "__main__":
    unittest.main()
