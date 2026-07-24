"""Tie-back: the app's flexible-exit engine must reproduce the verified engine.

Under the frozen contract (no trailing, no breakeven, 120-min cap, 15:30 forced
exit), ``simulate_flex`` must equal ``strategy_lab.simulate_positions`` - which is
itself reconciled to the tick against the production backtest. That transitively
verifies the app's fills. A random-bars fuzz drives it to 1e-12.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.app.sim.exit_config import frozen_config
from src.app.sim.flex_exit import simulate_flex
from src.statistical_research.strategy_lab import StrategyLabConfig, _bar_arrays, simulate_positions

_DATE = pd.Timestamp("2021-06-01")
_BASE_TS = pd.Timestamp("2021-06-01 08:00:00", tz="UTC")


def _random_bars(n=400, seed=11):
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.normal(0, 0.2, n))
    o = mid + rng.normal(0, 0.05, n)
    c = mid + rng.normal(0, 0.05, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.1, n))
    low = np.minimum(o, c) - np.abs(rng.normal(0, 0.1, n))
    half = n // 2
    dates = [_DATE] * half + [_DATE + pd.Timedelta(days=1)] * (n - half)
    minute = np.concatenate([np.arange(850, 850 + half), np.arange(850, 850 + (n - half))]).astype(
        np.int64
    )
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


class FrozenTieBackTests(unittest.TestCase):
    def test_flex_frozen_equals_reference(self):
        rng = np.random.default_rng(3)
        bars = _random_bars()
        arrays = _bar_arrays(bars)
        n = len(bars)
        entry_positions = np.arange(n)
        directions = rng.choice([-1, 1], size=n).astype(np.int8)
        stop_points = rng.uniform(0.2, 1.5, n)
        target_points = rng.uniform(0.2, 3.0, n)
        meta = pd.DataFrame(
            {
                "research_partition": ["Development"] * n,
                "entry_session": rng.choice(["London", "New York"], size=n),
                "trade_date_ny": bars["trade_date_ny"].to_numpy(),
            }
        )

        reference = (
            simulate_positions(
                entry_positions,
                directions,
                stop_points,
                target_points,
                meta,
                arrays,
                StrategyLabConfig(),
            )
            .sort_values("entry_position")
            .reset_index(drop=True)
        )
        flex = (
            simulate_flex(
                entry_positions,
                directions,
                stop_points,
                target_points,
                frozen_config(),
                meta,
                arrays,
            )
            .sort_values("entry_position")
            .reset_index(drop=True)
        )

        self.assertEqual(len(reference), len(flex))
        for col in [
            "entry_position",
            "exit_position",
            "exit_reason",
            "direction",
            "holding_minutes",
        ]:
            np.testing.assert_array_equal(
                reference[col].to_numpy(), flex[col].to_numpy(), err_msg=col
            )
        np.testing.assert_array_equal(
            reference["ambiguous_bar"].to_numpy(), flex["ambiguous_bar"].to_numpy()
        )
        np.testing.assert_allclose(
            reference["gross_r"].to_numpy(), flex["gross_r"].to_numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            reference["stop_ticks"].to_numpy(), flex["stop_ticks"].to_numpy(), atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
