"""Synthetic tests for the Section 11 sequential backtest engine."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.sequential_backtest import (
    Section11Config,
    _scenario_cost_ticks,
    run_sequential_backtest,
)


def _make_bars(prices: list[tuple[float, float, float, float]], *, start_minute: int = 420) -> pd.DataFrame:
    """Deterministic one-date bar sequence starting 07:00 New York."""

    n = len(prices)
    timestamps = pd.date_range("2023-03-01 12:00", periods=n, freq="1min").to_numpy()
    opens, highs, lows, closes = (np.array(x, dtype=np.float64) for x in zip(*prices))
    return pd.DataFrame(
        {
            "ts_event_utc": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "trade_date_ny": np.full(n, pd.Timestamp("2023-03-01").to_datetime64()),
            "minute_of_day_ny": np.arange(start_minute, start_minute + n, dtype=np.int64),
            "continuous_segment_id": np.zeros(n, dtype=np.int64),
        }
    )


def _make_candidate(bars: pd.DataFrame, bar_position: int, *, stop_points: float = 1.0,
                    target_points: float = 2.0, gate: bool = True,
                    partition: str = "Development") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_timestamp_utc": [bars["ts_event_utc"].iloc[bar_position]],
            "entry_timestamp_ny": [bars["ts_event_utc"].iloc[bar_position]],
            "trade_date_ny": [bars["trade_date_ny"].iloc[bar_position]],
            "entry_session": ["New York"],
            "research_partition": [partition],
            "entry_price": [bars["open"].iloc[bar_position]],
            "decision_atr_20m": [1.0],
            "gate_prediction": [1.0],
            "expansion_gate_flag": [gate],
            "stop_points": [stop_points],
            "target_points": [target_points],
        },
        index=pd.Index([bar_position], name="observation_id"),
    )


class ExitLogicTests(unittest.TestCase):
    def test_long_target_hit_gives_two_r_gross(self) -> None:
        bars = _make_bars([
            (100.0, 100.4, 99.8, 100.2),
            (100.2, 101.0, 100.1, 100.9),
            (100.9, 102.3, 100.8, 102.2),
        ])
        candidates = _make_candidate(bars, 0)
        result = run_sequential_backtest(candidates, bars)
        long_trades = result.trade_log.loc[
            result.trade_log["direction_variant"].eq("long_benchmark")
        ]
        self.assertTrue(long_trades["exit_reason"].eq("target").all())
        np.testing.assert_allclose(long_trades["gross_r"], 2.0)

    def test_long_stop_hit_gives_minus_one_r_gross(self) -> None:
        bars = _make_bars([
            (100.0, 100.2, 99.6, 99.7),
            (99.7, 99.8, 98.9, 99.0),
        ])
        candidates = _make_candidate(bars, 0)
        result = run_sequential_backtest(candidates, bars)
        long_trades = result.trade_log.loc[
            result.trade_log["direction_variant"].eq("long_benchmark")
        ]
        self.assertTrue(long_trades["exit_reason"].eq("stop").all())
        np.testing.assert_allclose(long_trades["gross_r"], -1.0)

    def test_ambiguous_bar_is_conservative_stop_first(self) -> None:
        # The single bar spans stop and target for both directions:
        # long stop 99 / target 102, short stop 101 / target 98.
        bars = _make_bars([
            (100.0, 102.5, 97.5, 100.0),
        ])
        candidates = _make_candidate(bars, 0)
        result = run_sequential_backtest(candidates, bars)
        trades = result.trade_log
        self.assertTrue(trades["ambiguous_bar"].all())
        self.assertTrue(trades["exit_reason"].eq("stop").all())
        np.testing.assert_allclose(trades["gross_r"], -1.0)

    def test_time_exit_at_declared_holding_cap(self) -> None:
        flat = [(100.0, 100.1, 99.9, 100.0)] * 10
        bars = _make_bars(flat)
        candidates = _make_candidate(bars, 0)
        config = Section11Config(max_holding_minutes=5)
        result = run_sequential_backtest(candidates, bars, config)
        self.assertTrue(result.trade_log["exit_reason"].eq("time_exit").all())
        self.assertTrue(result.trade_log["holding_minutes"].eq(5).all())

    def test_forced_1530_exit(self) -> None:
        flat = [(100.0, 100.1, 99.9, 100.0)] * 6
        bars = _make_bars(flat, start_minute=927)
        candidates = _make_candidate(bars, 0)
        result = run_sequential_backtest(candidates, bars)
        self.assertTrue(result.trade_log["exit_reason"].eq("forced_1530").all())

    def test_short_direction_is_symmetric(self) -> None:
        bars = _make_bars([
            (100.0, 100.2, 99.6, 99.7),
            (99.7, 99.8, 97.9, 98.0),
        ])
        candidates = _make_candidate(bars, 0)
        result = run_sequential_backtest(candidates, bars)
        short_trades = result.trade_log.loc[
            result.trade_log["direction_variant"].eq("short_benchmark")
        ]
        self.assertTrue(short_trades["exit_reason"].eq("target").all())
        np.testing.assert_allclose(short_trades["gross_r"], 2.0)


class PortfolioRuleTests(unittest.TestCase):
    def test_one_position_skips_overlapping_candidates(self) -> None:
        flat = [(100.0, 100.1, 99.9, 100.0)] * 12
        bars = _make_bars(flat)
        first = _make_candidate(bars, 0)
        overlapping = _make_candidate(bars, 2)
        later = _make_candidate(bars, 8)
        candidates = pd.concat([first, overlapping, later])
        config = Section11Config(max_holding_minutes=5)
        result = run_sequential_backtest(candidates, bars, config)
        per_variant = result.trade_log.groupby(["direction_variant", "gate_variant"]).size()
        self.assertTrue(per_variant.eq(2).all())

    def test_gated_variant_excludes_gate_failures(self) -> None:
        flat = [(100.0, 100.1, 99.9, 100.0)] * 6
        bars = _make_bars(flat)
        candidates = _make_candidate(bars, 0, gate=False)
        result = run_sequential_backtest(candidates, bars)
        gated = result.trade_log.loc[result.trade_log["gate_variant"].eq("expansion_gated")]
        ungated = result.trade_log.loc[result.trade_log["gate_variant"].eq("ungated")]
        self.assertTrue(gated.empty)
        self.assertFalse(ungated.empty)


class CostAndVerdictTests(unittest.TestCase):
    def test_cost_arithmetic_in_r(self) -> None:
        bars = _make_bars([
            (100.0, 100.4, 99.8, 100.2),
            (100.2, 102.3, 100.1, 102.2),
        ])
        candidates = _make_candidate(bars, 0)
        config = Section11Config()
        result = run_sequential_backtest(candidates, bars, config)
        base = result.performance.loc[
            result.performance["cost_scenario"].eq("base")
            & result.performance["direction_variant"].eq("long_benchmark")
            & result.performance["gate_variant"].eq("ungated")
        ]
        stop_ticks = 1.0 / config.tick_size
        expected_net = 2.0 - _scenario_cost_ticks(config, "base") / stop_ticks
        np.testing.assert_allclose(base["mean_net_r"], expected_net)

    def test_verdict_requires_positive_expectancy_in_both_partitions(self) -> None:
        winning = [(100.0, 100.4, 99.9, 100.3), (100.3, 102.5, 100.2, 102.4)]
        bars_dev = _make_bars(winning)
        dev = _make_candidate(bars_dev, 0, partition="Development")
        val_bars = _make_bars(winning)
        val_bars["ts_event_utc"] = pd.date_range("2024-03-01 12:00", periods=2, freq="1min").to_numpy()
        val_bars["trade_date_ny"] = np.full(2, pd.Timestamp("2024-03-01").to_datetime64())
        val = _make_candidate(val_bars, 0, partition="Validation")
        val.index = pd.Index([100], name="observation_id")
        val["entry_timestamp_utc"] = val_bars["ts_event_utc"].iloc[0]
        val["trade_date_ny"] = val_bars["trade_date_ny"].iloc[0]
        bars = pd.concat([bars_dev, val_bars], ignore_index=True)
        candidates = pd.concat([dev, val])
        result = run_sequential_backtest(candidates, bars)
        long_ungated = result.verdicts.loc[
            result.verdicts["direction_variant"].eq("long_benchmark")
            & result.verdicts["gate_variant"].eq("ungated"), "verdict"
        ].iloc[0]
        self.assertEqual(long_ungated, "POSITIVE_EXPECTANCY")
        short_ungated = result.verdicts.loc[
            result.verdicts["direction_variant"].eq("short_benchmark")
            & result.verdicts["gate_variant"].eq("ungated"), "verdict"
        ].iloc[0]
        self.assertEqual(short_ungated, "REJECTED")

    def test_entry_price_mismatch_raises(self) -> None:
        bars = _make_bars([(100.0, 100.4, 99.8, 100.2)])
        candidates = _make_candidate(bars, 0)
        candidates["entry_price"] = 123.0
        with self.assertRaises(ValueError):
            run_sequential_backtest(candidates, bars)

    def test_final_test_rows_raise(self) -> None:
        bars = _make_bars([(100.0, 100.4, 99.8, 100.2)])
        candidates = _make_candidate(bars, 0, partition="Final test")
        with self.assertRaises(ValueError):
            run_sequential_backtest(candidates, bars)

    def test_all_validation_checks_pass(self) -> None:
        flat = [(100.0, 100.4, 99.8, 100.2), (100.2, 102.3, 100.1, 102.2)]
        bars = _make_bars(flat)
        candidates = _make_candidate(bars, 0)
        result = run_sequential_backtest(candidates, bars)
        failed = result.validation_checks.loc[~result.validation_checks["passed"]]
        self.assertTrue(failed.empty, failed.to_string())


if __name__ == "__main__":
    unittest.main()
