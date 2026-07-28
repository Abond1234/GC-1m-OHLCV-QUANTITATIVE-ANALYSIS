"""Strategy replay and forensics services.

Replay mapping is verified on a synthetic universe (fired signals -> entry
positions -> flexible-exit trades with observation_id attached), and the
Final-test partition is rejected. Forensics is checked against the real artifacts
when present.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.app.datalayer.bar_store import BarStore
from src.app.datalayer.catalog_service import StrategyReplayService
from src.app.datalayer.forensics import ForensicsService
from src.statistical_research.strategy_lab import StrategySpec

_BASE_TS = pd.Timestamp("2021-06-01 08:00:00", tz="UTC")
_DATE = pd.Timestamp("2021-06-01")


def _bars(n=10):
    price = np.linspace(100, 101, n)
    return pd.DataFrame(
        {
            "ts_event_utc": [_BASE_TS + pd.Timedelta(minutes=i) for i in range(n)],
            "product": ["GC"] * n,
            "open": price,
            "high": price + 0.3,
            "low": price - 0.3,
            "close": price + 0.05,
            "volume": np.full(n, 100.0),
            "trade_date_ny": [_DATE] * n,
            "minute_of_day_ny": np.arange(240, 240 + n, dtype=np.int64),
            "continuous_segment_id": np.ones(n, dtype=int),
            "rolling_atr_20m": np.full(n, 0.5),
        }
    )


def _universe(bars, positions, partition="Development"):
    return pd.DataFrame(
        {
            "observation_id": list(range(len(positions))),
            "entry_timestamp_utc": [bars["ts_event_utc"].iloc[p] for p in positions],
            "trade_date_ny": [_DATE] * len(positions),
            "entry_session": ["London"] * len(positions),
            "research_partition": [partition] * len(positions),
            "stop_points": [0.5] * len(positions),
            "target_points": [1.0] * len(positions),
        }
    )


_ALL_LONG = StrategySpec(
    "all_long_test", "benchmark", "test", lambda frame, cfg: np.ones(len(frame), dtype=np.int8)
)


class ReplayServiceTests(unittest.TestCase):
    def test_replay_maps_signals_to_trades_with_observation_id(self):
        bars = _bars()
        store = BarStore.from_frame(bars)
        universe = _universe(bars, [1, 4, 6])
        service = StrategyReplayService(universe, store)
        log = service.replay(_ALL_LONG)
        self.assertGreater(len(log), 0)
        self.assertIn("observation_id", log.columns)
        self.assertIn("gross_r", log.columns)
        self.assertTrue((log["direction"] == 1).all())
        # every trade links back to a real observation
        self.assertTrue(log["observation_id"].isin(universe["observation_id"]).all())

    def test_replay_rejects_final_test_universe(self):
        bars = _bars()
        store = BarStore.from_frame(bars)
        universe = _universe(bars, [1, 4], partition="Validation")
        universe.loc[0, "research_partition"] = "Final test"
        with self.assertRaises(ValueError):
            StrategyReplayService(universe, store)

    def test_custom_exit_config_changes_the_log(self):
        from src.app.sim.exit_config import ExitConfig

        bars = _bars(20)
        store = BarStore.from_frame(bars)
        universe = _universe(bars, [1])
        service = StrategyReplayService(universe, store)
        frozen = service.replay(_ALL_LONG)
        wider = service.replay(
            _ALL_LONG,
            ExitConfig(stop_mode="atr", stop_value=3.0, target_mode="r", target_value=4.0),
        )
        # a wider stop yields a different stop_ticks than the frozen 0.5-point stop
        self.assertNotAlmostEqual(
            float(frozen["stop_ticks"].iloc[0]), float(wider["stop_ticks"].iloc[0])
        )


class ForensicsServiceTests(unittest.TestCase):
    def test_context_for_real_observation(self):
        service = ForensicsService()
        if not service.available():
            self.skipTest("research artifacts not present")
        context = service.context_for(0)
        self.assertEqual(context.observation_id, 0)
        self.assertEqual(set(context.excursions), {5, 15, 30, 60, 120, 180})
        self.assertIn("distance_from_execution_session_vwap_atr", context.features)
        self.assertIn("mfe_long_atr", context.excursions[60])


if __name__ == "__main__":
    unittest.main()
