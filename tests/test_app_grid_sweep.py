"""Exit-grid sweep compute (no Qt)."""

from __future__ import annotations

import unittest

import numpy as np

from src.app.analysis.grid_sweep import EXIT_REASONS, default_axes, sweep_entry
from src.app.sim.exit_config import ExitConfig

_CFG = ExitConfig(forced_exit_minute_ny=None, max_holding_minutes=500)


def _uptrend(n=60, slope=0.5):
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


class GridSweepTests(unittest.TestCase):
    def setUp(self):
        self.arrays, self.atr = _uptrend()

    def test_shapes_and_trial_count(self):
        stop_mults = np.array([0.5, 1.0, 1.5])
        target_rs = np.array([1.0, 2.0])
        grid = sweep_entry(1, 1, _CFG, self.atr, self.arrays, stop_mults, target_rs)
        self.assertEqual(grid.gross_r.shape, (3, 2))
        self.assertEqual(grid.win.shape, (3, 2))
        self.assertEqual(grid.trials, 6)
        self.assertTrue(np.all(grid.reason_code < len(EXIT_REASONS)))

    def test_surface_varies_and_best_cell_is_valid(self):
        stop_mults = np.array([0.5, 1.0, 2.0])
        target_rs = np.array([1.0, 2.0, 3.0])
        grid = sweep_entry(1, 1, _CFG, self.atr, self.arrays, stop_mults, target_rs)
        self.assertGreater(np.nanstd(grid.gross_r), 0.0)  # not a flat surface
        i, j = grid.best_cell
        self.assertTrue(0 <= i < 3 and 0 <= j < 3)
        self.assertEqual(grid.gross_r[i, j], np.nanmax(grid.gross_r))

    def test_default_axes(self):
        stops, targets = default_axes()
        self.assertGreater(len(stops), 1)
        self.assertGreater(len(targets), 1)
        self.assertLess(stops[0], stops[-1])


if __name__ == "__main__":
    unittest.main()
