"""What-if runner and the plain-language explanation engine (no Qt)."""

from __future__ import annotations

import unittest

import numpy as np

from src.app.analysis.explain import verdict
from src.app.analysis.whatif import default_variants, run_whatifs
from src.app.sim.exit_config import ExitConfig
from src.app.sim.flex_exit import single_flex_exit

_CFG = ExitConfig(
    stop_mode="atr",
    stop_value=1.5,
    target_mode="r",
    target_value=2.0,
    forced_exit_minute_ny=None,
    max_holding_minutes=500,
)
_COLORS = ("#a", "#b", "#c", "#d", "#e")


def _uptrend(n=40, slope=0.5):
    close = 100.0 + np.arange(n) * slope
    arrays = {
        "open": close - 0.01,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "segment": np.ones(n, dtype=int),
        "trade_date": np.full(n, np.datetime64("2023-06-01")),
        "minute_ny": (np.arange(n) % 1440).astype(np.int64),
    }
    return arrays, np.full(n, 1.0)


class WhatIfTests(unittest.TestCase):
    def test_default_variants_are_distinct_configs(self):
        variants = default_variants(_CFG)
        self.assertEqual(len(variants), 4)
        labels = {label for label, _ in variants}
        self.assertIn("wider stop", labels)
        self.assertTrue(any(cfg.trailing_enabled for _, cfg in variants))

    def test_run_whatifs_returns_a_run_per_variant(self):
        arrays, atr = _uptrend()
        runs = run_whatifs(1, 1, _CFG, atr, arrays, _COLORS)
        self.assertEqual(len(runs), 4)
        wider = next(r for r in runs if r.label == "wider stop")
        self.assertGreater(wider.result.initial_stop_points, 1.5)
        self.assertIsInstance(runs[0].survived, bool)


class ExplainTests(unittest.TestCase):
    def test_target_win_reads_as_a_winner(self):
        arrays, _ = _uptrend()
        result = single_flex_exit(1, 1, 1.5, 3.0, _CFG, arrays)
        lines = verdict(result, ctx=None, hook_r=1.0)
        self.assertTrue(lines)
        self.assertIn("target", lines[0].lower())

    def test_winner_on_the_hook_is_called_out(self):
        high = np.array([100.1, 101.0, 102.25, 101.0, 99.0, 97.9])
        low = np.array([99.9, 100.0, 101.0, 99.0, 98.0, 97.75])
        close = np.array([100.0, 100.5, 101.5, 100.0, 98.5, 97.8])
        n = len(high)
        arrays = {
            "open": np.full(n, 100.0),
            "high": high,
            "low": low,
            "close": close,
            "segment": np.ones(n, dtype=int),
            "trade_date": np.full(n, np.datetime64("2023-06-01")),
            "minute_ny": np.arange(n, dtype=np.int64),
        }
        result = single_flex_exit(1, 1, 2.25, 4.5, _CFG, arrays)
        text = " ".join(verdict(result, ctx=None, hook_r=1.0)).lower()
        self.assertIn("hook", text)


if __name__ == "__main__":
    unittest.main()
