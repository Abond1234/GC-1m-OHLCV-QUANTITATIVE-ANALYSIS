from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.labels import (
    build_forward_label_table,
    validate_tick_grid,
)

NY = "America/New_York"


def _bars(
    closes,
    *,
    highs=None,
    lows=None,
    entry_open=100.0,
    entry_time="2024-01-02 09:00",
    decision_atr=2.0,
):
    closes = np.asarray(closes, dtype=float)
    path_opens = np.r_[entry_open, closes[:-1]]
    highs = np.asarray(
        highs if highs is not None else np.maximum(path_opens, closes) + 0.1, dtype=float
    )
    lows = np.asarray(
        lows if lows is not None else np.minimum(path_opens, closes) - 0.1, dtype=float
    )
    entry_ny = pd.Timestamp(entry_time, tz=NY)
    timestamps_ny = pd.date_range(
        entry_ny - pd.Timedelta(minutes=1), periods=len(closes) + 1, freq="min"
    )
    opens = np.r_[entry_open, path_opens]
    all_closes = np.r_[entry_open, closes]
    all_highs = np.r_[entry_open, highs]
    all_lows = np.r_[entry_open, lows]
    n = len(timestamps_ny)
    return pd.DataFrame(
        {
            "source_row_id": np.arange(n, dtype=np.int64),
            "ts_event_utc": timestamps_ny.tz_convert("UTC"),
            "ts_event_ny": timestamps_ny,
            "trade_date_ny": pd.DatetimeIndex(timestamps_ny.tz_localize(None).normalize()),
            "product": ["GC"] * n,
            "symbol": ["GCG4"] * n,
            "active_symbol": ["GCG4"] * n,
            "instrument_id": np.full(n, 101, dtype=np.uint32),
            "continuous_segment_id": np.ones(n, dtype=np.int32),
            "open": opens,
            "high": all_highs,
            "low": all_lows,
            "close": all_closes,
            "tradable_research_flag": np.ones(n, dtype=bool),
            "roll_window_flag": np.zeros(n, dtype=bool),
            "roll_window_type": ["none"] * n,
            "low_liquidity_active_day_flag": np.zeros(n, dtype=bool),
            "low_liquidity_warning_flag": np.zeros(n, dtype=bool),
            "rolling_atr_20m": np.r_[decision_atr, np.full(len(closes), 2.0)],
        }
    )


def _observation(bars, *, entry_position=1, partition="Development", observation_id=0):
    decision_position = entry_position - 1
    entry_ny = bars.loc[entry_position, "ts_event_ny"]
    forced_exit = entry_ny.normalize() + pd.Timedelta(hours=15, minutes=30)
    return pd.DataFrame(
        {
            "observation_id": [observation_id],
            "decision_bar_id": [bars.loc[decision_position, "source_row_id"]],
            "entry_bar_id": [bars.loc[entry_position, "source_row_id"]],
            "decision_timestamp_utc": [bars.loc[decision_position, "ts_event_utc"]],
            "decision_timestamp_ny": [bars.loc[decision_position, "ts_event_ny"]],
            "entry_timestamp_utc": [bars.loc[entry_position, "ts_event_utc"]],
            "entry_timestamp_ny": [entry_ny],
            "trade_date_ny": [bars.loc[entry_position, "trade_date_ny"]],
            "entry_session": pd.Categorical(
                ["New York"], categories=["London", "New York"], ordered=True
            ),
            "research_partition": pd.Categorical(
                [partition], categories=["Development", "Validation", "Final test"], ordered=True
            ),
            "product": [bars.loc[entry_position, "product"]],
            "symbol": [bars.loc[entry_position, "symbol"]],
            "active_symbol": [bars.loc[entry_position, "active_symbol"]],
            "instrument_id": [bars.loc[entry_position, "instrument_id"]],
            "continuous_segment_id": [bars.loc[entry_position, "continuous_segment_id"]],
            "forced_exit_timestamp_ny": [forced_exit],
        }
    )


def _build(bars, *, horizon=5):
    return build_forward_label_table(
        _observation(bars), bars, horizons=(horizon,), chunk_size=2
    ).labels


class ForwardLabelCalculationTests(unittest.TestCase):
    def test_monotonically_rising_path_and_known_return_units(self):
        closes = [100.1, 100.2, 100.3, 100.4, 100.5]
        labels = _build(_bars(closes))
        self.assertEqual(labels.loc[0, "forward_return_5_ticks"], 5)
        self.assertAlmostEqual(labels.loc[0, "forward_return_5_bps"], 50.0)
        self.assertAlmostEqual(labels.loc[0, "forward_return_5_atr"], 0.25)
        self.assertEqual(labels.loc[0, "mfe_long_5_ticks"], 6)
        self.assertEqual(labels.loc[0, "mae_long_5_ticks"], 1)
        self.assertEqual(labels.loc[0, "future_range_5_ticks"], 7)
        self.assertEqual(labels.loc[0, "time_to_mfe_long_5_minutes"], 5)
        self.assertEqual(labels.loc[0, "time_to_mae_long_5_minutes"], 1)
        self.assertEqual(labels.loc[0, "direction_label_5"], 1)

    def test_monotonically_falling_path(self):
        labels = _build(_bars([99.9, 99.8, 99.7, 99.6, 99.5]))
        self.assertEqual(labels.loc[0, "forward_return_5_ticks"], -5)
        self.assertEqual(labels.loc[0, "direction_label_5"], -1)
        self.assertEqual(labels.loc[0, "time_to_mfe_short_5_minutes"], 5)
        self.assertEqual(labels.loc[0, "time_to_mae_short_5_minutes"], 1)

    def test_flat_path(self):
        labels = _build(_bars([100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5))
        self.assertEqual(labels.loc[0, "forward_return_5_ticks"], 0)
        self.assertEqual(labels.loc[0, "future_range_5_ticks"], 0)
        self.assertEqual(labels.loc[0, "direction_label_5"], 0)
        self.assertEqual(labels.loc[0, "time_to_mfe_long_5_minutes"], 1)
        self.assertEqual(labels.loc[0, "time_to_mae_long_5_minutes"], 1)

    def test_middle_and_repeated_extrema_use_first_occurrence(self):
        labels = _build(
            _bars(
                [100.0] * 5,
                highs=[100.1, 100.5, 100.2, 100.5, 100.1],
                lows=[99.9, 99.8, 99.5, 99.8, 99.5],
            )
        )
        self.assertEqual(labels.loc[0, "time_to_mfe_long_5_minutes"], 2)
        self.assertEqual(labels.loc[0, "time_to_mfe_short_5_minutes"], 3)
        self.assertEqual(labels.loc[0, "time_to_mae_long_5_minutes"], 3)
        self.assertEqual(labels.loc[0, "time_to_mae_short_5_minutes"], 2)

    def test_known_realized_volatility(self):
        closes = np.array([100.1, 100.2, 100.1, 100.3, 100.4])
        labels = _build(_bars(closes))
        returns = np.log(np.r_[closes[0] / 100.0, closes[1:] / closes[:-1]])
        expected = np.sqrt(np.square(returns).sum()) * 10_000.0
        self.assertAlmostEqual(
            labels.loc[0, "future_realized_volatility_5_bps"], expected, places=10
        )

    def test_long_short_excursion_symmetry(self):
        labels = _build(_bars([100.1, 99.9, 100.3, 99.8, 100.0]))
        self.assertEqual(labels.loc[0, "mfe_long_5_ticks"], labels.loc[0, "mae_short_5_ticks"])
        self.assertEqual(labels.loc[0, "mae_long_5_ticks"], labels.loc[0, "mfe_short_5_ticks"])

    def test_economic_exit_timestamp_is_after_fifth_bar_close(self):
        bars = _bars([100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5)
        labels = _build(bars)
        self.assertEqual(
            labels.loc[0, "exit_timestamp_utc_5"],
            bars.loc[5, "ts_event_utc"] + pd.Timedelta(minutes=1),
        )

    def test_tick_grid_validation_rejects_material_off_grid_price(self):
        with self.assertRaises(ValueError):
            validate_tick_grid(np.array([100.0, 100.05]), field_name="test")

    def test_expansion_threshold_uses_development_only(self):
        bars = _bars([100.0] * 24, highs=[100.0] * 24, lows=[100.0] * 24)
        entry_positions = [1, 7, 13, 19]
        range_prices = [0.2, 0.4, 2.0, 4.0]
        partitions = ["Development", "Development", "Validation", "Final test"]
        observations = []
        for observation_id, (entry_position, range_price, partition) in enumerate(
            zip(entry_positions, range_prices, partitions, strict=True)
        ):
            path = np.arange(entry_position, entry_position + 5)
            bars.loc[path, "high"] = 100.0 + range_price / 2.0
            bars.loc[path, "low"] = 100.0 - range_price / 2.0
            observations.append(
                _observation(
                    bars,
                    entry_position=entry_position,
                    partition=partition,
                    observation_id=observation_id,
                )
            )
        result = build_forward_label_table(
            pd.concat(observations, ignore_index=True), bars, horizons=(5,)
        )
        threshold = result.expansion_thresholds.iloc[0]
        self.assertEqual(threshold["development_observation_count_used"], 2)
        self.assertAlmostEqual(threshold["threshold"], 0.18)
        self.assertEqual(threshold["fitting_partition"], "Development")


class ForwardLabelAvailabilityTests(unittest.TestCase):
    def _reason_after_mutation(self, column, value, row=3):
        bars = _bars([100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5)
        bars.loc[row, column] = value
        return _build(bars).loc[0, "label_unavailable_reason_5"]

    def test_missing_one_minute_bar(self):
        bars = _bars([100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5)
        bars.loc[3:, "ts_event_utc"] += pd.Timedelta(minutes=1)
        bars.loc[3:, "ts_event_ny"] += pd.Timedelta(minutes=1)
        labels = _build(bars)
        self.assertFalse(labels.loc[0, "label_available_5"])
        self.assertEqual(
            labels.loc[0, "label_unavailable_reason_5"], "missing_or_nonconsecutive_minute"
        )

    def test_first_failure_priority_is_deterministic(self):
        bars = _bars([100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5)
        bars.loc[3:, "ts_event_utc"] += pd.Timedelta(minutes=1)
        bars.loc[3:, "ts_event_ny"] += pd.Timedelta(minutes=1)
        bars.loc[3, "symbol"] = "GCJ4"
        self.assertEqual(
            _build(bars).loc[0, "label_unavailable_reason_5"], "missing_or_nonconsecutive_minute"
        )

    def test_contract_or_instrument_change(self):
        self.assertEqual(
            self._reason_after_mutation("symbol", "GCJ4"), "contract_or_instrument_change"
        )
        self.assertEqual(
            self._reason_after_mutation("instrument_id", 202), "contract_or_instrument_change"
        )

    def test_continuous_segment_change(self):
        self.assertEqual(
            self._reason_after_mutation("continuous_segment_id", 2), "continuous_segment_change"
        )

    def test_new_york_date_change(self):
        self.assertEqual(
            self._reason_after_mutation("trade_date_ny", pd.Timestamp("2024-01-03")),
            "New_York_date_change",
        )

    def test_tradability_and_roll_failure(self):
        self.assertEqual(
            self._reason_after_mutation("tradable_research_flag", False),
            "tradability_roll_or_liquidity_failure",
        )
        self.assertEqual(
            self._reason_after_mutation("roll_window_flag", True),
            "tradability_roll_or_liquidity_failure",
        )

    def test_exact_forced_exit_boundary_is_available(self):
        bars = _bars(
            [100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5, entry_time="2024-01-02 15:25"
        )
        labels = _build(bars)
        self.assertTrue(labels.loc[0, "label_available_5"])
        self.assertEqual(
            labels.loc[0, "exit_timestamp_ny_5"], pd.Timestamp("2024-01-02 15:30", tz=NY)
        )

    def test_path_closing_after_forced_exit_is_unavailable(self):
        bars = _bars(
            [100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5, entry_time="2024-01-02 15:26"
        )
        labels = _build(bars)
        self.assertFalse(labels.loc[0, "label_available_5"])
        self.assertEqual(labels.loc[0, "label_unavailable_reason_5"], "forced_exit_boundary_breach")
        self.assertTrue(pd.isna(labels.loc[0, "forward_return_5_ticks"]))

    def test_incomplete_180_minute_path_is_not_shortened(self):
        bars = _bars([100.0] * 20, highs=[100.0] * 20, lows=[100.0] * 20)
        labels = _build(bars, horizon=180)
        self.assertFalse(labels.loc[0, "label_available_180"])
        self.assertEqual(labels.loc[0, "label_unavailable_reason_180"], "insufficient_future_rows")
        self.assertTrue(pd.isna(labels.loc[0, "exit_price_180"]))

    def test_invalid_price_inside_path(self):
        bars = _bars([100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5)
        bars.loc[3, "high"] = 99.9
        labels = _build(bars)
        self.assertEqual(labels.loc[0, "label_unavailable_reason_5"], "invalid_price_data")

    def test_missing_or_nonpositive_atr_keeps_raw_path_available(self):
        for atr in (np.nan, 0.0, -1.0):
            with self.subTest(atr=atr):
                bars = _bars([100.1] * 5, decision_atr=atr)
                labels = _build(bars)
                self.assertTrue(labels.loc[0, "label_available_5"])
                self.assertFalse(labels.loc[0, "atr_normalization_available"])
                self.assertFalse(pd.isna(labels.loc[0, "forward_return_5_ticks"]))
                self.assertTrue(pd.isna(labels.loc[0, "forward_return_5_atr"]))
                self.assertTrue(pd.isna(labels.loc[0, "expansion_label_5"]))


if __name__ == "__main__":
    unittest.main()
