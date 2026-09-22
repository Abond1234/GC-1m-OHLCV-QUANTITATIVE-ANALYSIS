"""Cached, viewport-sized chart data for the Dev+Val history.

This module is deliberately display-only.  The simulator and research engine
continue to consume the original 1-minute ``BarStore``; the cache merely picks
an appropriate visual resolution and returns a buffered slice for painting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timeframe import TIMEFRAMES, Resampled, resample_window


@dataclass(frozen=True)
class ViewportSlice:
    """One buffered paint tile inside a cached full-history timeframe."""

    resampled: Resampled
    timeframe: int
    tile_start: int
    tile_end: int
    visible_start: int
    visible_end: int
    full_count: int

    @property
    def x_range(self) -> tuple[float, float]:
        return (float(self.visible_start) - 0.5, float(self.visible_end) + 0.5)


def _slice_resampled(rs: Resampled, start: int, end: int) -> Resampled:
    sl = slice(int(start), int(end) + 1)
    return Resampled(
        open=rs.open[sl],
        high=rs.high[sl],
        low=rs.low[sl],
        close=rs.close[sl],
        volume=rs.volume[sl],
        segment=rs.segment[sl],
        bucket_starts=rs.bucket_starts[sl],
        bucket_ends=rs.bucket_ends[sl],
    )


class ChartViewportCache:
    """Lazy timeframe cache plus LOD/tile selection for a single instrument."""

    def __init__(self, bars):
        self.bars = bars
        self._cache: dict[int, Resampled] = {}
        self._timeframes = tuple(sorted(set(TIMEFRAMES.values())))

    def series(self, timeframe: int) -> Resampled:
        """Return the cached full-history numerical series for ``timeframe``."""

        tf = max(1, int(timeframe))
        cached = self._cache.get(tf)
        if cached is not None:
            return cached
        n = int(self.bars.n_bars)
        if tf == 1:
            # OHLCV/segment remain zero-copy views.  Only the two mapping arrays
            # are allocated, avoiding the large fancy-index copies made by the
            # generic aggregation path for a million-plus 1-minute candles.
            idx = np.arange(n, dtype=np.int64)
            cached = Resampled(
                open=np.asarray(self.bars.open),
                high=np.asarray(self.bars.high),
                low=np.asarray(self.bars.low),
                close=np.asarray(self.bars.close),
                volume=np.asarray(self.bars.volume, dtype=np.float64),
                segment=np.asarray(self.bars.segment),
                bucket_starts=idx,
                bucket_ends=idx,
            )
        else:
            cached = resample_window(self.bars, 0, n - 1, tf)
        self._cache[tf] = cached
        return cached

    @staticmethod
    def _indices(rs: Resampled, global_start: int, global_end: int) -> tuple[int, int]:
        if len(rs.close) == 0:
            return 0, -1
        lo = int(np.searchsorted(rs.bucket_ends, int(global_start), side="left"))
        hi = int(np.searchsorted(rs.bucket_starts, int(global_end), side="right")) - 1
        lo = max(0, min(lo, len(rs.close) - 1))
        hi = max(lo, min(hi, len(rs.close) - 1))
        return lo, hi

    def choose_timeframe(
        self,
        requested: int,
        global_start: int,
        global_end: int,
        pixel_width: int,
    ) -> int:
        """Choose the finest timeframe that does not overdraw the viewport."""

        requested = max(1, int(requested))
        # About two source points per horizontal pixel preserves candle shape
        # while keeping QPicture construction and line painting bounded.
        budget = max(600, int(max(1, pixel_width) * 2.0))
        candidates = [tf for tf in self._timeframes if tf >= requested]
        if requested not in candidates:
            candidates.insert(0, requested)
        for tf in candidates:
            rs = self.series(tf)
            lo, hi = self._indices(rs, global_start, global_end)
            if hi - lo + 1 <= budget:
                return tf
        return candidates[-1]

    def global_range_for_x(self, timeframe: int, x0: float, x1: float) -> tuple[int, int]:
        """Convert logical full-series x coordinates into a 1-minute range."""

        rs = self.series(timeframe)
        if len(rs.close) == 0:
            return 0, -1
        a = max(0, min(int(np.floor(min(x0, x1))), len(rs.close) - 1))
        b = max(a, min(int(np.ceil(max(x0, x1))), len(rs.close) - 1))
        return int(rs.bucket_starts[a]), int(rs.bucket_ends[b])

    def window(
        self,
        requested: int,
        global_start: int,
        global_end: int,
        pixel_width: int,
        *,
        timeframe: int | None = None,
    ) -> ViewportSlice:
        """Return a visible range plus a generous panning buffer."""

        tf = timeframe or self.choose_timeframe(requested, global_start, global_end, pixel_width)
        master = self.series(tf)
        visible_start, visible_end = self._indices(master, global_start, global_end)
        visible_count = visible_end - visible_start + 1
        buffer_count = max(240, int(visible_count * 0.75))
        tile_start = max(0, visible_start - buffer_count)
        tile_end = min(len(master.close) - 1, visible_end + buffer_count)
        return ViewportSlice(
            resampled=_slice_resampled(master, tile_start, tile_end),
            timeframe=tf,
            tile_start=tile_start,
            tile_end=tile_end,
            visible_start=visible_start,
            visible_end=visible_end,
            full_count=len(master.close),
        )

    @staticmethod
    def tile_contains(view: ViewportSlice, x0: float, x1: float) -> bool:
        """True while the visible range remains inside the tile's inner buffer."""

        margin = max(32, int((view.tile_end - view.tile_start + 1) * 0.12))
        left_ok = view.tile_start == 0 or min(x0, x1) >= view.tile_start + margin
        right_ok = view.tile_end == view.full_count - 1 or max(x0, x1) <= view.tile_end - margin
        return left_ok and right_ok
