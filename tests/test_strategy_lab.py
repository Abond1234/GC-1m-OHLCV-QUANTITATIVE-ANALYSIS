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
    run_strategy_lab,
    score_trades,
    simulate_positions,
    strategy_library,
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
