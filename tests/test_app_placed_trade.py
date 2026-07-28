"""Pure logic for placed free-play trades: sizing, drag-recompute, config-recompute.

No Qt; runs in CI. Verifies that dragging a stop or changing the config re-runs the
verified engine and moves the result the expected way.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.app.analysis.placed_trade import (
    place_trade,
    recompute_config,
    recompute_levels,
    resolve_one,
)
from src.app.sim.exit_config import ExitConfig

_CFG = ExitConfig(
    stop_mode="atr",
    stop_value=1.5,
    target_mode="r",
    target_value=2.0,
    forced_exit_minute_ny=None,
    max_holding_minutes=500,
)


def _uptrend(n=60, slope=0.5, atr=1.0):
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
    return arrays, np.full(n, atr)


class ResolveTests(unittest.TestCase):
    def test_resolve_one_uses_decision_bar_atr(self):
        _arrays, atr = _uptrend()
        atr[0] = 2.0  # decision bar for entry=1 is index 0
        stop, target, trail = resolve_one(1, atr, _CFG)
        self.assertAlmostEqual(stop, 1.5 * 2.0)
        self.assertAlmostEqual(target, 2.0 * stop)
        self.assertEqual(trail, 0.0)


class PlaceAndRecomputeTests(unittest.TestCase):
    def setUp(self):
        self.arrays, self.atr = _uptrend()

    def test_place_trade_fills_at_next_open_and_hits_target(self):
        t = place_trade(1, 1, 1, _CFG, self.atr, self.arrays, color="#fff")
        self.assertEqual(t.result.entry_price, self.arrays["open"][1])
        self.assertEqual(t.result.exit_reason, "target")
        self.assertAlmostEqual(t.result.gross_r, 2.0, places=6)

    def test_widening_the_stop_lowers_realised_r(self):
        t = place_trade(1, 1, 1, _CFG, self.atr, self.arrays)
        wider = recompute_levels(t, self.arrays, stop_points=t.stop_points * 2)
        self.assertGreater(wider.stop_points, t.stop_points)
        self.assertLess(wider.result.gross_r, t.result.gross_r)

    def test_recompute_config_resizes_from_new_rule(self):
        t = place_trade(1, 1, 1, _CFG, self.atr, self.arrays)
        tighter = recompute_config(
            t, ExitConfig(**{**_CFG.__dict__, "target_value": 1.0}), self.atr, self.arrays
        )
        self.assertAlmostEqual(tighter.target_points, tighter.stop_points, places=6)
        self.assertNotEqual(tighter.result.exit_position, 0)


if __name__ == "__main__":
    unittest.main()
