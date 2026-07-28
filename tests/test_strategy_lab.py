"""Tests for the multi-strategy laboratory.

Two things must hold for the lab to be trustworthy: the generalised per-trade
simulator must reproduce the verified Section 11 engine when the direction is
held constant (a regression tie-back), and every strategy must read only its
declared causal features (no forward-label leakage).  The rest confirm the
signal rules fire as specified and that governance rejects a Final-test leak.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.sequential_backtest import run_sequential_backtest
from src.statistical_research.strategy_lab import (
    SIGNAL_FEATURE_COLUMNS,
    StrategyLabConfig,
    _bar_arrays,
    precompute_directional_exits,
    run_strategy_lab,
    score_trades,
    simulate_positions,
    strategy_library,
    walk_precomputed,
)

_DATE = pd.Timestamp("2021-06-01")
_BASE_TS = pd.Timestamp("2021-06-01 08:00:00", tz="UTC")

_PATH = [
    (100.00, 100.20, 99.90, 100.10),
    (100.10, 100.30, 100.00, 100.20),
    (100.20, 100.40, 100.10, 100.30),
    (100.30, 101.00, 100.20, 100.90),
    (100.90, 101.10, 100.80, 101.00),
    (101.00, 101.10, 100.30, 100.40),
    (100.40, 100.50, 100.30, 100.40),
    (100.40, 100.50, 100.30, 100.40),
    (100.40, 100.50, 100.30, 100.40),
    (100.40, 100.50, 100.30, 100.40),
]


def _bars():
    n = len(_PATH)
    o, h, low, c = (np.array(x, dtype=float) for x in zip(*_PATH, strict=True))
    return pd.DataFrame(
        {
            "ts_event_utc": [_BASE_TS + pd.Timedelta(minutes=i) for i in range(n)],
            "product": ["GC"] * n,
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "trade_date_ny": [_DATE] * n,
            "minute_of_day_ny": np.arange(240, 240 + n, dtype=np.int64),
            "continuous_segment_id": np.ones(n, dtype=int),
        }
    )


def _candidates(positions, bars, *, partition="Development"):
    return pd.DataFrame(
        {
            "observation_id": list(positions),
            "entry_timestamp_utc": [bars["ts_event_utc"].iloc[p] for p in positions],
            "trade_date_ny": [bars["trade_date_ny"].iloc[p] for p in positions],
            "entry_session": ["London"] * len(positions),
            "research_partition": [partition] * len(positions),
            "entry_price": [bars["open"].iloc[p] for p in positions],
            "expansion_gate_flag": [True] * len(positions),
            "stop_points": [0.5] * len(positions),
            "target_points": [0.8] * len(positions),
        }
    )


class SignalRuleTests(unittest.TestCase):
    def test_momentum_deadband(self):
        specs = {s.name: s for s in strategy_library()}
        frame = pd.DataFrame({"return_60m_atr": [2.0, -2.0, 0.5, -0.5, 0.0]})
        out = specs["momentum_60m"].signal(frame, StrategyLabConfig())
        np.testing.assert_array_equal(out, np.array([1, -1, 0, 0, 0], dtype=np.int8))

    def test_session_extreme_fade_is_contrarian(self):
        specs = {s.name: s for s in strategy_library()}
        frame = pd.DataFrame({"session_range_position": [0.02, 0.98, 0.5]})
        out = specs["session_extreme_fade"].signal(frame, StrategyLabConfig())
        np.testing.assert_array_equal(out, np.array([1, -1, 0], dtype=np.int8))

    def test_signals_are_ternary(self):
        cfg = StrategyLabConfig()
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({col: rng.normal(size=200) for col in SIGNAL_FEATURE_COLUMNS})
        for spec in strategy_library():
            out = spec.signal(frame, cfg)
            self.assertEqual(len(out), 200)
            self.assertTrue(set(np.unique(out)).issubset({-1, 0, 1}), msg=spec.name)

    def test_strategies_read_only_declared_features(self):
        # A frame with only the declared feature columns must not raise; a
        # forward-label read would KeyError here.
        cfg = StrategyLabConfig()
        frame = pd.DataFrame({col: np.zeros(5) for col in SIGNAL_FEATURE_COLUMNS})
        for spec in strategy_library():
            spec.signal(frame, cfg)  # must not raise


class SimulatorTieBackTests(unittest.TestCase):
    """The generalised simulator must equal the verified engine at constant direction."""

    def test_all_long_matches_production_long_benchmark(self):
        bars = _bars()
        positions = [1, 4, 6]
        cand = _candidates(positions, bars)
        produced = run_sequential_backtest(cand, bars)
        prod_long = (
            produced.trade_log[
                (produced.trade_log["direction_variant"] == "long_benchmark")
                & (produced.trade_log["gate_variant"] == "ungated")
            ]
            .sort_values("entry_position")
            .reset_index(drop=True)
        )

        arrays = _bar_arrays(bars)
        entry_positions = np.array(positions)
        lab = (
            simulate_positions(
                entry_positions,
                np.ones(len(positions), dtype=np.int8),
                cand["stop_points"].to_numpy(float),
                cand["target_points"].to_numpy(float),
                cand[["research_partition", "entry_session", "trade_date_ny"]],
                arrays,
                StrategyLabConfig(),
            )
            .sort_values("entry_position")
            .reset_index(drop=True)
        )

        self.assertEqual(len(lab), len(prod_long))
        np.testing.assert_array_equal(
            lab["exit_position"].to_numpy(), prod_long["exit_position"].to_numpy()
        )
        np.testing.assert_array_equal(
            lab["exit_reason"].to_numpy(), prod_long["exit_reason"].to_numpy()
        )
        np.testing.assert_allclose(
            lab["gross_r"].to_numpy(), prod_long["gross_r"].to_numpy(), atol=1e-12
        )

    def test_all_short_matches_production_short_benchmark(self):
        bars = _bars()
        positions = [1, 4, 6]
        cand = _candidates(positions, bars)
        produced = run_sequential_backtest(cand, bars)
        prod_short = (
            produced.trade_log[
                (produced.trade_log["direction_variant"] == "short_benchmark")
                & (produced.trade_log["gate_variant"] == "ungated")
            ]
            .sort_values("entry_position")
            .reset_index(drop=True)
        )
        arrays = _bar_arrays(bars)
        lab = (
            simulate_positions(
                np.array(positions),
                np.full(len(positions), -1, dtype=np.int8),
                cand["stop_points"].to_numpy(float),
                cand["target_points"].to_numpy(float),
                cand[["research_partition", "entry_session", "trade_date_ny"]],
                arrays,
                StrategyLabConfig(),
            )
            .sort_values("entry_position")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(
            lab["gross_r"].to_numpy(), prod_short["gross_r"].to_numpy(), atol=1e-12
        )


class FastPathTests(unittest.TestCase):
    """The precomputed fast walk must equal the verified reference simulator."""

    def _random_bars(self, n=400, seed=11):
        rng = np.random.default_rng(seed)
        mid = 100 + np.cumsum(rng.normal(0, 0.2, n))
        o = mid + rng.normal(0, 0.05, n)
        c = mid + rng.normal(0, 0.05, n)
        h = np.maximum(o, c) + np.abs(rng.normal(0, 0.1, n))
        low = np.minimum(o, c) - np.abs(rng.normal(0, 0.1, n))
        half = n // 2
        dates = [_DATE] * half + [_DATE + pd.Timedelta(days=1)] * (n - half)
        minute = np.concatenate(
            [np.arange(850, 850 + half), np.arange(850, 850 + (n - half))]
        ).astype(np.int64)
        segment = np.array([1] * half + [2] * (n - half))
        return pd.DataFrame(
            {
                "ts_event_utc": [_BASE_TS + pd.Timedelta(minutes=i) for i in range(n)],
                "product": ["GC"] * n,
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "trade_date_ny": dates,
                "minute_of_day_ny": minute,
                "continuous_segment_id": segment,
            }
        )

    def test_precomputed_walk_equals_reference(self):
        rng = np.random.default_rng(3)
        bars = self._random_bars()
        arrays = _bar_arrays(bars)
        n = len(bars)
        entry_positions = np.arange(n)
        directions = rng.choice([-1, 1], size=n).astype(np.int8)
        stop_points = rng.uniform(0.2, 1.5, n)
        target_points = rng.uniform(0.2, 3.0, n)
        meta = pd.DataFrame(
            {
                "research_partition": ["Development"] * n,
                "entry_session": rng.choice(["London", "New_York"], size=n),
                "trade_date_ny": bars["trade_date_ny"].to_numpy(),
            }
        )
        cfg = StrategyLabConfig()

        reference = (
            simulate_positions(
                entry_positions, directions, stop_points, target_points, meta, arrays, cfg
            )
            .sort_values("entry_position")
            .reset_index(drop=True)
        )
        precomputed = precompute_directional_exits(
            entry_positions, stop_points, target_points, arrays, cfg
        )
        fast = (
            walk_precomputed(
                entry_positions, directions, precomputed, meta, stop_points / cfg.tick_size, cfg
            )
            .sort_values("entry_position")
            .reset_index(drop=True)
        )

        self.assertEqual(len(reference), len(fast))
        for col in [
            "entry_position",
            "exit_position",
            "exit_reason",
            "direction",
            "holding_minutes",
        ]:
            np.testing.assert_array_equal(
                reference[col].to_numpy(), fast[col].to_numpy(), err_msg=col
            )
        np.testing.assert_allclose(
            reference["gross_r"].to_numpy(), fast["gross_r"].to_numpy(), atol=1e-12
        )


class RunLabTests(unittest.TestCase):
    def _universe(self, positions, bars, partition="Development"):
        cand = _candidates(positions, bars, partition=partition)
        # attach zero-valued features so signal functions can run
        for col in SIGNAL_FEATURE_COLUMNS:
            cand[col] = 0.0
        return cand

    def test_lab_runs_and_passes_checks(self):
        bars = _bars()
        universe = self._universe([1, 4, 6], bars)
        result = run_strategy_lab(universe, bars, specs=strategy_library())
        self.assertTrue(result.validation_checks["passed"].all())
        self.assertIn("always_long", set(result.trade_log["strategy"]))

    def test_final_test_universe_is_rejected(self):
        bars = _bars()
        universe = self._universe([1, 4], bars, partition="Final")
        with self.assertRaises(ValueError):
            run_strategy_lab(universe, bars, specs=strategy_library())

    def test_entry_price_mismatch_is_rejected(self):
        bars = _bars()
        universe = self._universe([1, 4], bars)
        universe.loc[0, "entry_price"] = 500.0
        with self.assertRaises(ValueError):
            run_strategy_lab(universe, bars, specs=strategy_library())

    def test_score_trades_reports_expected_metrics(self):
        bars = _bars()
        universe = self._universe([1, 4, 6], bars)
        long_only = [s for s in strategy_library() if s.name == "always_long"]
        result = run_strategy_lab(universe, bars, specs=long_only)
        perf = score_trades(result.trade_log, StrategyLabConfig())
        self.assertTrue({"win_rate", "mean_net_r", "per_trade_sharpe"}.issubset(perf.columns))
        self.assertTrue((perf["cost_scenario"].isin({"frictionless", "base", "pessimistic"})).all())


if __name__ == "__main__":
    unittest.main()
