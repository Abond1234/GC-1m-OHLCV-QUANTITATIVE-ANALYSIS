"""Within-trade excursion geometry and the winner-on-the-hook flag (no Qt)."""

from __future__ import annotations

import unittest

import numpy as np

from src.app.analysis.excursion import compute_excursion, is_winner_on_the_hook
from src.app.sim.exit_config import ExitConfig
from src.app.sim.flex_exit import single_flex_exit

# A long that runs to +1 R in profit, then reverses into its stop for -1 R.
_HIGH = np.array([100.1, 101.0, 102.25, 101.0, 99.0, 97.9])
_LOW = np.array([99.9, 100.0, 101.0, 99.0, 98.0, 97.75])
_CLOSE = np.array([100.0, 100.5, 101.5, 100.0, 98.5, 97.8])


def _arrays():
    n = len(_HIGH)
    return {
        "open": np.full(n, 100.0),
        "high": _HIGH,
        "low": _LOW,
        "close": _CLOSE,
        "segment": np.ones(n, dtype=int),
        "trade_date": np.full(n, np.datetime64("2023-06-01")),
        "minute_ny": np.arange(n, dtype=np.int64),
    }


class ExcursionTests(unittest.TestCase):
    def setUp(self):
        self.arrays = _arrays()
        cfg = ExitConfig(forced_exit_minute_ny=None, max_holding_minutes=100)
        self.result = single_flex_exit(1, 1, 2.25, 4.5, cfg, self.arrays)

    def test_excursion_peaks_match_the_engine(self):
        exc = compute_excursion(self.result, self.arrays["high"], self.arrays["low"])
        self.assertAlmostEqual(exc.mfe_r, self.result.mfe_r, places=9)
        self.assertAlmostEqual(exc.mae_r, self.result.mae_r, places=9)
        self.assertAlmostEqual(exc.mfe_r, 1.0, places=6)

    def test_running_envelopes_are_monotonic(self):
        exc = compute_excursion(self.result, self.arrays["high"], self.arrays["low"])
        self.assertTrue(np.all(np.diff(exc.fav_price) >= -1e-9))  # non-decreasing for a long
        self.assertTrue(np.all(np.diff(exc.adv_price) <= 1e-9))  # non-increasing

    def test_winner_on_the_hook(self):
        self.assertLessEqual(self.result.gross_r, 0)
        self.assertTrue(is_winner_on_the_hook(self.result, 1.0))
        self.assertFalse(is_winner_on_the_hook(self.result, 2.0))


if __name__ == "__main__":
    unittest.main()
