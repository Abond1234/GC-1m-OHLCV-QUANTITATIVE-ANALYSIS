"""Governance for the app data layer: Development+Validation only, never
Final-test, plus BarStore viewport/segment behaviour.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.app.datalayer.bar_store import BarStore
from src.app.datalayer.partitions import EVAL_CAP_DATE, assert_dev_val_only


def _bars(n=6, *, start_date="2021-06-01", segments=None, minute_start=240):
    date = pd.Timestamp(start_date)
    base = pd.Timestamp("2021-06-01 08:00:00", tz="UTC")
    seg = np.ones(n, dtype=int) if segments is None else np.asarray(segments)
    return pd.DataFrame(
        {
            "ts_event_utc": [base + pd.Timedelta(minutes=i) for i in range(n)],
            "product": ["GC"] * n,
            "open": np.linspace(100, 101, n),
            "high": np.linspace(100.2, 101.2, n),
            "low": np.linspace(99.8, 100.8, n),
            "close": np.linspace(100.1, 101.1, n),
            "volume": np.full(n, 100.0),
            "trade_date_ny": [date] * n,
            "minute_of_day_ny": np.arange(minute_start, minute_start + n, dtype=np.int64),
            "continuous_segment_id": seg,
            "rolling_atr_20m": np.full(n, 0.5),
        }
    )


class LockoutTests(unittest.TestCase):
    def test_assert_dev_val_only_raises_on_final_test(self):
        dates = pd.to_datetime(["2023-05-01", "2024-06-01", "2025-01-02"])
        with self.assertRaises(ValueError):
            assert_dev_val_only(dates)

    def test_assert_dev_val_only_passes_dev_val(self):
        dates = pd.to_datetime(["2021-01-04", "2023-12-31", "2024-12-31"])
        assert_dev_val_only(dates)  # must not raise

    def test_bar_store_rejects_final_test_rows(self):
        frame = _bars(start_date="2025-03-01")
        with self.assertRaises(ValueError):
            BarStore.from_frame(frame)

    def test_bar_store_accepts_dev_val(self):
        store = BarStore.from_frame(_bars(start_date="2024-02-01"))
        self.assertEqual(store.n_bars, 6)
        self.assertLessEqual(pd.Timestamp(store.trade_date.max()), EVAL_CAP_DATE)


class ViewportTests(unittest.TestCase):
    def test_segment_bounds_split_at_breaks(self):
        store = BarStore.from_frame(_bars(n=6, segments=[1, 1, 1, 2, 2, 2]))
        bounds = store.segment_bounds(0, 5)
        self.assertEqual(bounds, [(0, 2), (3, 5)])

    def test_segment_bounds_within_a_window(self):
        store = BarStore.from_frame(_bars(n=6, segments=[1, 1, 2, 2, 3, 3]))
        self.assertEqual(store.segment_bounds(1, 4), [(1, 1), (2, 3), (4, 4)])

    def test_bar_arrays_shape_matches_simulator(self):
        store = BarStore.from_frame(_bars())
        arrays = store.bar_arrays()
        self.assertEqual(
            set(arrays),
            {"ts", "open", "high", "low", "close", "segment", "trade_date", "minute_ny"},
        )
        self.assertEqual(len(arrays["close"]), store.n_bars)


if __name__ == "__main__":
    unittest.main()
