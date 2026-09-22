"""Display-only cache/LOD tests; simulator inputs are never transformed."""

from __future__ import annotations

import unittest

import numpy as np

from src.app.datalayer.chart_viewport import ChartViewportCache


class _Bars:
    def __init__(self, n: int = 20_000):
        self.n_bars = n
        self.minute_ny = np.arange(n, dtype=np.int64) % 1440
        day = np.arange(n, dtype=np.int64) // 1440
        self.trade_date = np.datetime64("2021-01-01") + day.astype("timedelta64[D]")
        self.segment = np.ones(n, dtype=np.int16)
        self.open = np.linspace(1800.0, 1900.0, n)
        self.high = self.open + 1.0
        self.low = self.open - 1.0
        self.close = self.open + 0.25
        self.volume = np.full(n, 50.0)


class ChartViewportCacheTests(unittest.TestCase):
    def test_one_minute_cache_reuses_bar_arrays(self):
        bars = _Bars()
        cache = ChartViewportCache(bars)
        one = cache.series(1)
        self.assertTrue(np.shares_memory(one.close, bars.close))
        self.assertIs(cache.series(1), one)

    def test_zoomed_out_uses_lod_and_zoomed_in_returns_to_one_minute(self):
        bars = _Bars()
        cache = ChartViewportCache(bars)
        overview = cache.choose_timeframe(1, 0, bars.n_bars - 1, 800)
        detail = cache.choose_timeframe(1, 10_000, 10_500, 800)
        self.assertGreater(overview, 1)
        self.assertEqual(detail, 1)

    def test_window_is_buffered_but_not_full_history(self):
        bars = _Bars()
        cache = ChartViewportCache(bars)
        view = cache.window(1, 10_000, 10_500, 800, timeframe=1)
        self.assertLess(len(view.resampled.close), bars.n_bars)
        self.assertLess(view.tile_start, view.visible_start)
        self.assertGreater(view.tile_end, view.visible_end)
        self.assertEqual(view.resampled.bucket_starts[0], view.tile_start)
        self.assertTrue(cache.tile_contains(view, *view.x_range))


if __name__ == "__main__":
    unittest.main()
