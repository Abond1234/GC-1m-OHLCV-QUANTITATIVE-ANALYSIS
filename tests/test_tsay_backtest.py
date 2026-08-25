"""Focused tests for the frozen Tsay sequential simulator."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.tsay_backtest import (
    frozen_no_policy,
    frozen_order_statistic,
    simulate_tsay_policy,
)


class TsayBacktestTests(unittest.TestCase):
    def test_frozen_order_statistic_uses_ceiling_and_no_interpolation(self) -> None:
        self.assertEqual(frozen_order_statistic(np.arange(1.0, 11.0), 0.9), 9.0)
        self.assertEqual(frozen_order_statistic(np.array([1.0, 1.0, 2.0]), 0.8), 2.0)

    def test_simulator_enters_next_open_and_resolves_ambiguous_bar_stop_first(self) -> None:
        timestamp = pd.date_range("2023-01-03 14:00", periods=4, freq="min", tz="UTC")
        bars = pd.DataFrame(
            {
                "timestamp_utc": timestamp,
                "timestamp_ny": timestamp.tz_convert("America/New_York"),
                "trade_date_ny": ["2023-01-03"] * 4,
                "product": ["GC"] * 4,
                "active_symbol": ["GCG3"] * 4,
                "continuous_segment_id": [1] * 4,
                "open": [100.0] * 4,
                "high": [100.0, 104.0, 100.0, 100.0],
                "low": [100.0, 98.0, 100.0, 100.0],
                "close": [100.0] * 4,
            }
        )
        signals = pd.DataFrame(
            {
                "observation_id": [7],
                "decision_timestamp_utc": [timestamp[0]],
                "direction": [1],
                "decision_atr_20m": [1.0],
                "eligible": [True],
            }
        )
        trades = simulate_tsay_policy(bars, signals)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["entry_timestamp_utc"], timestamp[1])
        self.assertEqual(trades.iloc[0]["exit_reason"], "STOP_FIRST")
        self.assertEqual(trades.iloc[0]["gross_r"], -1.0)

    def test_no_direction_freezes_without_real_bar_access(self) -> None:
        state = frozen_no_policy()
        self.assertEqual(state["policy_status"], "FROZEN_NO_POLICY")
        self.assertFalse(state["real_bar_data_loaded"])
        self.assertFalse(state["validation_economics_may_open"])


if __name__ == "__main__":
    unittest.main()
