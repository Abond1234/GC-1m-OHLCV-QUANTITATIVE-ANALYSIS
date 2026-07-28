"""VWAP reconstruction: hand-calc correctness, segment/day resets, and (when the
real artifacts are present) validation against the stored distance feature.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.app.datalayer.bar_store import BarStore
from src.app.datalayer.paths import feature_matrix_path, research_bars_path
from src.app.datalayer.vwap import research_day_vwap, rolling_vwap


def _bars_hlc(rows, *, segments=None, days=None):
    """rows = list of (hlc_price, volume); high=low=close for a clean VWAP."""

    n = len(rows)
    price = np.array([r[0] for r in rows], dtype=float)
    vol = np.array([r[1] for r in rows], dtype=float)
    base = pd.Timestamp("2021-06-01 08:00:00", tz="UTC")
    seg = np.ones(n, dtype=int) if segments is None else np.asarray(segments)
    day = [pd.Timestamp("2021-06-01")] * n if days is None else list(days)
    return pd.DataFrame(
        {
            "ts_event_utc": [base + pd.Timedelta(minutes=i) for i in range(n)],
            "product": ["GC"] * n,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": vol,
            "trade_date_ny": day,
            "minute_of_day_ny": np.arange(240, 240 + n, dtype=np.int64),
            "continuous_segment_id": seg,
            "rolling_atr_20m": np.full(n, 0.5),
        }
    )


class RollingVwapTests(unittest.TestCase):
    def test_rolling_vwap_hand_calc(self):
        # typical == price (h=l=c); pv = price*vol.
        store = BarStore.from_frame(_bars_hlc([(10.0, 100), (20.0, 100), (30.0, 200)]))
        vwap = rolling_vwap(store, window=2)
        # idx0: 1000/100=10; idx1: 3000/200=15; idx2: (2000+6000)/(100+200)=26.6667
        np.testing.assert_allclose(vwap, [10.0, 15.0, 8000 / 300], atol=1e-9)

    def test_rolling_vwap_resets_at_segment_break(self):
        store = BarStore.from_frame(
            _bars_hlc([(10.0, 100), (20.0, 100), (30.0, 200)], segments=[1, 1, 2])
        )
        vwap = rolling_vwap(store, window=2)
        # idx2 is the first bar of segment 2 -> its own VWAP (30), not blended.
        self.assertAlmostEqual(vwap[2], 30.0, places=9)

    def test_research_day_vwap_resets_per_day(self):
        store = BarStore.from_frame(
            _bars_hlc(
                [(10.0, 100), (20.0, 100), (30.0, 200)],
                days=[
                    pd.Timestamp("2021-06-01"),
                    pd.Timestamp("2021-06-01"),
                    pd.Timestamp("2021-06-02"),
                ],
            )
        )
        vwap = research_day_vwap(store)
        self.assertAlmostEqual(vwap[1], 15.0, places=9)  # (1000+2000)/200
        self.assertAlmostEqual(vwap[2], 30.0, places=9)  # new day -> only itself


class RollingVwapAgainstFeatureTests(unittest.TestCase):
    """Reconstructed rolling VWAP(20) must satisfy the stored distance feature:
    distance_from_rolling_vwap_20_atr == (close - vwap20) / rolling_atr_20m at the
    decision bar. Runs only when the real artifacts are present.
    """

    def test_matches_stored_distance_feature(self):
        import pyarrow.parquet as pq

        bars_path = research_bars_path()
        fm_path = feature_matrix_path()
        if not bars_path.exists() or not fm_path.exists():
            self.skipTest("research artifacts not present")

        day_lo = pd.Timestamp("2021-06-01")
        day_hi = pd.Timestamp("2021-06-02")
        bars = pq.read_table(
            bars_path,
            columns=[
                "ts_event_utc",
                "product",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_date_ny",
                "minute_of_day_ny",
                "continuous_segment_id",
                "rolling_atr_20m",
            ],
            filters=[("product", "==", "GC"), ("trade_date_ny", "==", day_lo.to_pydatetime())],
        ).to_pandas(ignore_metadata=True)
        if len(bars) < 200:
            self.skipTest("insufficient bars for the sampled day")
        store = BarStore.from_frame(bars)
        vwap20 = rolling_vwap(store, window=20)
        ts_index = pd.Index(store.ts)

        feats = pq.read_table(
            fm_path,
            columns=["decision_timestamp_utc", "distance_from_rolling_vwap_20_atr"],
            filters=[
                ("decision_timestamp_utc", ">=", day_lo.tz_localize("UTC").to_pydatetime()),
                ("decision_timestamp_utc", "<", day_hi.tz_localize("UTC").to_pydatetime()),
            ],
        ).to_pandas(ignore_metadata=True)

        pos = ts_index.get_indexer(feats["decision_timestamp_utc"])
        ok = pos >= 0
        pos = pos[ok]
        # Only compare bars with a full 20-window inside their segment.
        seg = store.segment[pos]
        deep = np.array(
            [
                np.sum(store.segment[max(0, p - 19) : p + 1] == seg[i]) >= 20
                for i, p in enumerate(pos)
            ]
        )
        pos = pos[deep]
        distance = feats.loc[ok, "distance_from_rolling_vwap_20_atr"].to_numpy()[deep]
        if len(pos) < 30:
            self.skipTest("too few full-window decision bars matched")
        implied = (store.close[pos] - vwap20[pos]) / store.atr20[pos]
        finite = np.isfinite(implied) & np.isfinite(distance)
        np.testing.assert_allclose(implied[finite], distance[finite], atol=1e-2)


if __name__ == "__main__":
    unittest.main()
