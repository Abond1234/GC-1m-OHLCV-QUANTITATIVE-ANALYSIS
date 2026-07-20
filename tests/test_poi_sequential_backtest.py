"""Synthetic tests for the Section 8 POI sequential backtest."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.research.poi_sequential_backtest import (
    Section8BacktestConfig,
    _scenario_cost_ticks,
    run_poi_sequential_backtest,
)

FAST_CONFIG = Section8BacktestConfig(maximum_holding_minutes=10)


def _make_bars(prices, *, start_minute: int = 480) -> pd.DataFrame:
    n = len(prices)
    opens, highs, lows, closes = (np.array(x, dtype=np.float64) for x in zip(*prices, strict=True))
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "trade_date_ny": np.full(n, pd.Timestamp("2023-03-01").to_datetime64()),
            "minute_of_day_ny": np.arange(start_minute, start_minute + n, dtype=np.int64),
            "continuous_segment_id": np.zeros(n, dtype=np.int64),
        }
    )


def _make_opportunity(bar: int, entry: float, stop: float, *, partition="Development") -> dict:
    return {
        "true_retest_id": f"R{bar}",
        "trade_date_ny": pd.Timestamp("2023-03-01").to_datetime64(),
        "research_partition": partition,
        "entry_bar_id": bar,
        "entry_price": entry,
        "stop_price": stop,
    }


class ExitAndSequencingTests(unittest.TestCase):
    def test_short_target_hit_gives_three_r_gross(self) -> None:
        # Short from 100 with stop 101 (1 point risk): 3R target at 97.
        bars = _make_bars(
            [
                (100.2, 100.5, 99.8, 100.0),
                (100.0, 100.4, 96.8, 97.0),
            ]
        )
        opportunities = pd.DataFrame([_make_opportunity(0, 100.0, 101.0)])
        result = run_poi_sequential_backtest(opportunities, bars, FAST_CONFIG)
        self.assertEqual(result.trade_log["exit_reason"].tolist(), ["target"])
        np.testing.assert_allclose(result.trade_log["gross_r"], 3.0)

    def test_stop_hit_and_ambiguous_conservative(self) -> None:
        bars = _make_bars(
            [
                (100.2, 101.5, 96.5, 100.0),
            ]
        )
        opportunities = pd.DataFrame([_make_opportunity(0, 100.0, 101.0)])
        result = run_poi_sequential_backtest(opportunities, bars, FAST_CONFIG)
        self.assertEqual(result.trade_log["exit_reason"].tolist(), ["stop"])
        self.assertTrue(result.trade_log["ambiguous_bar"].all())
        np.testing.assert_allclose(result.trade_log["gross_r"], -1.0)

    def test_unfillable_entry_is_dropped(self) -> None:
        bars = _make_bars([(100.2, 100.5, 99.8, 100.0)])
        opportunities = pd.DataFrame([_make_opportunity(0, 105.0, 106.0)])
        result = run_poi_sequential_backtest(opportunities, bars, FAST_CONFIG)
        self.assertTrue(result.trade_log.empty)
        self.assertEqual(int(result.verdict["unfillable_entries_dropped"]), 1)

    def test_one_position_skips_overlap(self) -> None:
        flat = [(100.0, 100.1, 99.9, 100.0)] * 8
        bars = _make_bars(flat)
        opportunities = pd.DataFrame(
            [
                _make_opportunity(0, 100.0, 101.0),
                _make_opportunity(2, 100.0, 101.0),
            ]
        )
        config = Section8BacktestConfig(maximum_holding_minutes=5)
        result = run_poi_sequential_backtest(opportunities, bars, config)
        self.assertEqual(len(result.trade_log), 1)

    def test_cost_arithmetic_and_verdict_rule(self) -> None:
        bars = _make_bars(
            [
                (100.2, 100.5, 99.8, 100.0),
                (100.0, 100.4, 96.8, 97.0),
            ]
        )
        opportunities = pd.DataFrame(
            [
                _make_opportunity(0, 100.0, 101.0, partition="Development"),
            ]
        )
        result = run_poi_sequential_backtest(opportunities, bars, FAST_CONFIG)
        base = result.performance.loc[
            result.performance["cost_scenario"].eq("base")
            & result.performance["research_partition"].eq("Development")
        ]
        expected = 3.0 - _scenario_cost_ticks(FAST_CONFIG, "base") / 10.0
        np.testing.assert_allclose(base["mean_net_r"], expected)
        # Verdict requires positive Dev AND Val; Validation absent -> rejected.
        self.assertEqual(result.verdict["verdict_from_dev_val"], "SEQUENTIAL_REJECTED")

    def test_all_validation_checks_pass(self) -> None:
        bars = _make_bars([(100.2, 100.5, 99.8, 100.0), (100.0, 100.4, 96.8, 97.0)])
        opportunities = pd.DataFrame([_make_opportunity(0, 100.0, 101.0)])
        result = run_poi_sequential_backtest(opportunities, bars, FAST_CONFIG)
        failed = result.validation_checks.loc[~result.validation_checks["passed"]]
        self.assertTrue(failed.empty, failed.to_string())


if __name__ == "__main__":
    unittest.main()
