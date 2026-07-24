"""Semantics of the flexible-exit config: trailing, breakeven, forced minute,
stop-first ambiguity, and ATR level resolution. Hand-built paths with known
outcomes, contrasted against the frozen contract.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.app.sim.exit_config import ExitConfig, frozen_config, resolve_levels
from src.app.sim.flex_exit import single_flex_exit
from src.statistical_research.strategy_lab import _bar_arrays

_DATE = pd.Timestamp("2021-06-01")
_BASE_TS = pd.Timestamp("2021-06-01 08:00:00", tz="UTC")


def _bars(rows, *, minute_start=240, dates=None):
    n = len(rows)
    o, h, low, c = (np.array(x, dtype=float) for x in zip(*rows, strict=True))
    return pd.DataFrame(
        {
            "ts_event_utc": [_BASE_TS + pd.Timedelta(minutes=i) for i in range(n)],
            "product": ["GC"] * n,
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "trade_date_ny": [_DATE] * n if dates is None else dates,
            "minute_of_day_ny": np.arange(minute_start, minute_start + n, dtype=np.int64),
            "continuous_segment_id": np.ones(n, dtype=int),
        }
    )


class BreakevenTests(unittest.TestCase):
    # Long from 100, stop 1pt (=99), far target. Price makes +1.2 then falls back
    # through 100 (breakeven) but only to 99.8 first, then to 98.5.
    _PATH = [
        (100.0, 100.10, 99.90, 100.00),  # 0 filler
        (100.0, 100.20, 99.95, 100.10),  # 1 entry
        (100.1, 101.20, 100.50, 101.00),  # 2 favorable +1.2 -> breakeven arms
        (101.0, 101.10, 99.80, 99.90),  # 3 back to 99.8 (hits breakeven 100, not initial 99)
        (99.9, 100.00, 98.50, 98.60),  # 4 deep -> hits initial stop 99
        (98.6, 98.70, 98.40, 98.50),  # 5 filler
    ]

    def test_breakeven_beats_frozen(self):
        arrays = _bar_arrays(_bars(self._PATH))
        be = single_flex_exit(
            1,
            1,
            1.0,
            5.0,
            ExitConfig(breakeven_enabled=True, breakeven_trigger_r=1.0),
            arrays,
        )
        fr = single_flex_exit(1, 1, 1.0, 5.0, frozen_config(), arrays)
        self.assertEqual(be.exit_reason, "stop")
        self.assertAlmostEqual(be.gross_r, 0.0, places=9)  # stopped at breakeven 100
        self.assertAlmostEqual(fr.gross_r, -1.0, places=9)  # stopped at initial 99
        self.assertGreater(be.gross_r, fr.gross_r)


class TrailingTests(unittest.TestCase):
    # Long from 100, initial stop far (5pt), trailing 0.5. Runs to 103 then falls.
    _PATH = [
        (100.0, 100.10, 99.90, 100.00),  # 0 filler
        (100.0, 100.20, 99.90, 100.10),  # 1 entry
        (100.1, 101.50, 100.00, 101.40),  # 2 -> trail 101.0
        (101.4, 103.00, 101.20, 102.80),  # 3 peak 103 -> trail 102.5
        (102.8, 102.90, 100.40, 100.50),  # 4 low 100.4 hits trail 102.5
        (100.5, 100.60, 100.30, 100.40),  # 5 filler (frozen time-exit here)
    ]

    def test_trailing_locks_profit(self):
        arrays = _bar_arrays(_bars(self._PATH))
        tr = single_flex_exit(
            1,
            1,
            5.0,
            20.0,
            ExitConfig(trailing_enabled=True, trailing_mode="points", trailing_value=0.5),
            arrays,
            trail_pts=0.5,
        )
        fr = single_flex_exit(1, 1, 5.0, 20.0, frozen_config(), arrays)
        self.assertEqual(tr.exit_reason, "stop")
        self.assertEqual(tr.exit_position, 4)
        self.assertAlmostEqual(tr.gross_r, 0.5, places=9)  # (102.5-100)/5
        self.assertEqual(fr.exit_reason, "time_exit")
        self.assertGreater(tr.gross_r, fr.gross_r)


class ForcedMinuteTests(unittest.TestCase):
    _PATH = [
        (100.0, 100.10, 99.90, 100.00),  # 0 minute 928
        (100.0, 100.20, 99.90, 100.05),  # 1 entry, minute 929
        (100.05, 100.20, 99.95, 100.10),  # 2 minute 930 -> forced under default
        (100.10, 100.30, 100.00, 100.20),  # 3
    ]

    def test_disabling_forced_minute_keeps_trade_open(self):
        arrays = _bar_arrays(_bars(self._PATH, minute_start=928))
        forced = single_flex_exit(1, 1, 5.0, 5.0, frozen_config(), arrays)
        never = single_flex_exit(1, 1, 5.0, 5.0, ExitConfig(forced_exit_minute_ny=None), arrays)
        self.assertEqual(forced.exit_reason, "forced_1530")
        self.assertEqual(forced.exit_position, 2)
        self.assertNotEqual(never.exit_reason, "forced_1530")
        self.assertGreater(never.exit_position, forced.exit_position)


class StopFirstTests(unittest.TestCase):
    # One bar hits BOTH stop (99) and target (101).
    _PATH = [
        (100.0, 100.10, 99.90, 100.00),  # 0
        (100.0, 100.10, 99.95, 100.00),  # 1 entry
        (100.0, 101.50, 98.50, 100.00),  # 2 low 98.5<=99 and high 101.5>=101
    ]

    def test_stop_first_books_stop_and_flags_ambiguous(self):
        arrays = _bar_arrays(_bars(self._PATH))
        conservative = single_flex_exit(1, 1, 1.0, 1.0, frozen_config(), arrays)
        aggressive = single_flex_exit(1, 1, 1.0, 1.0, ExitConfig(stop_first=False), arrays)
        self.assertEqual(conservative.exit_reason, "stop")
        self.assertTrue(conservative.ambiguous_bar)
        self.assertAlmostEqual(conservative.gross_r, -1.0, places=9)
        self.assertEqual(aggressive.exit_reason, "target")
        self.assertAlmostEqual(aggressive.gross_r, 1.0, places=9)


class ResolveLevelsTests(unittest.TestCase):
    def test_atr_sizing_uses_decision_bar(self):
        atr20 = np.array([0.4, 0.5, 0.6, 0.7])
        entry_positions = np.array([2])  # decision bar = index 1 -> atr 0.5
        cfg = ExitConfig(stop_mode="atr", stop_value=1.5, target_mode="r", target_value=2.0)
        stop, target, trail = resolve_levels(entry_positions, atr20, cfg)
        self.assertAlmostEqual(stop[0], 1.5 * 0.5, places=9)
        self.assertAlmostEqual(target[0], 2.0 * 1.5 * 0.5, places=9)
        self.assertEqual(trail[0], 0.0)  # trailing disabled


if __name__ == "__main__":
    unittest.main()
