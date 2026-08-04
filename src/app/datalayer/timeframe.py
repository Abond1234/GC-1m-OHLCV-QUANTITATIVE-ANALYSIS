"""Display-timeframe resampling and the 1m <-> displayed-bar view mapping.

The simulator's truth is always the 1-minute bar store: entries, exits, level
tracks, the replay clock, and every simulation input stay per-minute. A chart
timeframe (5m/15m/.../1D) is a *display aggregation only*: this module buckets a
1m window into OHLCV bars (Open=first, High=max, Low=min, Close=last,
Volume=sum) and provides the ``ViewMap`` that converts between global 1m bar
indices and displayed x positions, so every overlay can be drawn on any
timeframe without the simulation ever seeing a bucket.

Buckets break on a New York trade-date change (which also handles the 18:00
minute-of-day wrap), on a ``minute_ny // tf`` quotient change, and on a
continuous-segment change - so no displayed candle ever spans a roll or data
discontinuity, and per-bucket ``segment`` is constant by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Display timeframes, in minutes of the NY clock.
TIMEFRAMES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1H": 60,
    "4H": 240,
    "1D": 1440,
}


@dataclass(frozen=True)
class Resampled:
    """One display bar per bucket over a global 1m window ``[lo, hi]``."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    segment: np.ndarray  # constant within each bucket by construction
    bucket_starts: np.ndarray  # global 1m index of the first minute per bucket
    bucket_ends: np.ndarray  # global 1m index of the last minute (inclusive)


def resample_window(bars, lo: int, hi: int, tf_minutes: int) -> Resampled:
    """Aggregate the global 1m window ``[lo, hi]`` into ``tf_minutes`` buckets."""

    sl = slice(lo, hi + 1)
    minute = bars.minute_ny[sl]
    date = bars.trade_date[sl]
    seg = bars.segment[sl]
    n = len(minute)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int64)
        return Resampled(empty, empty, empty, empty, empty, empty_i, empty_i, empty_i)

    new_bucket = np.empty(n, dtype=bool)
    new_bucket[0] = True
    if tf_minutes <= 1:
        new_bucket[1:] = True
    else:
        quotient = minute // tf_minutes
        new_bucket[1:] = (
            (quotient[1:] != quotient[:-1]) | (date[1:] != date[:-1]) | (seg[1:] != seg[:-1])
        )
    starts_local = np.nonzero(new_bucket)[0]
    ends_local = np.append(starts_local[1:] - 1, n - 1)

    return Resampled(
        open=bars.open[sl][starts_local],
        high=np.maximum.reduceat(bars.high[sl], starts_local),
        low=np.minimum.reduceat(bars.low[sl], starts_local),
        close=bars.close[sl][ends_local],
        volume=np.add.reduceat(np.asarray(bars.volume[sl], dtype=np.float64), starts_local),
        segment=seg[starts_local],
        bucket_starts=(lo + starts_local).astype(np.int64),
        bucket_ends=(lo + ends_local).astype(np.int64),
    )


def bucket_labels(bars, rs: Resampled, tf_minutes: int, multi_day: bool) -> np.ndarray:
    """Per-bucket time-axis labels.

    "HH:MM" within a single day, "MM-DD HH:MM" across days, "MM-DD" at 1D.
    """

    starts = rs.bucket_starts
    minutes = bars.minute_ny[starts]
    if tf_minutes >= 1440:
        return np.array(
            [pd.Timestamp(d).strftime("%m-%d") for d in bars.trade_date[starts]], dtype=object
        )
    times = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]
    if not multi_day:
        return np.array(times, dtype=object)
    days = [pd.Timestamp(d).strftime("%m-%d") for d in bars.trade_date[starts]]
    return np.array([f"{d} {t}" for d, t in zip(days, times, strict=True)], dtype=object)


def hover_labels(bars, rs: Resampled, tf_minutes: int) -> np.ndarray:
    """Exact per-bucket date/time labels for the crosshair pill.

    Always the full "YYYY-MM-DD HH:MM" (bucket start), or "YYYY-MM-DD" at 1D -
    the axis labels stay compact, the pill states the exact moment.
    """

    starts = rs.bucket_starts
    days = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in bars.trade_date[starts]]
    if tf_minutes >= 1440:
        return np.array(days, dtype=object)
    minutes = bars.minute_ny[starts]
    times = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]
    return np.array([f"{d} {t}" for d, t in zip(days, times, strict=True)], dtype=object)


def sample_last(series: np.ndarray, rs: Resampled) -> np.ndarray:
    """Display-sample a global-length line series at each bucket's last minute (VWAP)."""

    return np.asarray(series)[rs.bucket_ends]


def sample_first(series: np.ndarray, rs: Resampled) -> np.ndarray:
    """Display-sample a global-length code series at each bucket's first minute."""

    return np.asarray(series)[rs.bucket_starts]


@dataclass(frozen=True)
class ViewMap:
    """Global 1m bar index <-> displayed local x, for one rendered window.

    ``identity`` (the 1m case) maps every bar to itself, which keeps 1m
    rendering numerically identical to indexing arithmetic it replaces.
    Fractional positions place a minute inside its bucket's candle width, so
    per-minute tracks and markers stay pixel-honest at any timeframe.
    """

    bucket_starts: np.ndarray
    bucket_ends: np.ndarray
    tf_minutes: int

    @classmethod
    def identity(cls, lo: int, hi: int) -> ViewMap:
        idx = np.arange(int(lo), int(hi) + 1, dtype=np.int64)
        return cls(bucket_starts=idx, bucket_ends=idx, tf_minutes=1)

    @classmethod
    def from_resampled(cls, rs: Resampled, tf_minutes: int) -> ViewMap:
        return cls(
            bucket_starts=rs.bucket_starts, bucket_ends=rs.bucket_ends, tf_minutes=tf_minutes
        )

    @property
    def view_start(self) -> int:
        return int(self.bucket_starts[0])

    @property
    def view_end(self) -> int:
        return int(self.bucket_ends[-1])

    @property
    def n_display(self) -> int:
        return len(self.bucket_starts)

    def in_view(self, g: int) -> bool:
        return self.view_start <= int(g) <= self.view_end

    def global_to_local(self, g: int) -> int:
        """Displayed bar index containing global 1m bar ``g`` (clamped)."""

        i = int(np.searchsorted(self.bucket_starts, int(g), side="right")) - 1
        return max(0, min(i, self.n_display - 1))

    def global_to_local_f(self, g):
        """Fractional displayed x for global 1m position(s) ``g``.

        Minute k of an m-minute bucket at local index l maps to
        ``l - 0.5 + (k + 0.5) / m`` - the minutes tile the candle's width, and
        the 1m identity map returns exactly ``g - view_start``.
        """

        arr = np.asarray(g, dtype=np.float64)
        li = np.clip(
            np.searchsorted(self.bucket_starts, np.floor(arr).astype(np.int64), side="right") - 1,
            0,
            self.n_display - 1,
        )
        starts = self.bucket_starts[li]
        spans = self.bucket_ends[li] - starts + 1
        out = li - 0.5 + (arr - starts + 0.5) / spans
        if np.isscalar(g) or getattr(g, "ndim", 1) == 0:
            return float(out)
        return out

    def local_to_global_start(self, local: int) -> int:
        return int(self.bucket_starts[max(0, min(int(local), self.n_display - 1))])

    def local_to_global_end(self, local: int) -> int:
        return int(self.bucket_ends[max(0, min(int(local), self.n_display - 1))])

    def local_f_to_global(self, x: float) -> float:
        """Inverse of ``global_to_local_f`` (for drawing serialization)."""

        li = int(np.clip(np.floor(float(x) + 0.5), 0, self.n_display - 1))
        start = float(self.bucket_starts[li])
        span = float(self.bucket_ends[li] - self.bucket_starts[li] + 1)
        return start + (float(x) - (li - 0.5)) * span - 0.5

    def last_complete_local(self, t: int) -> int:
        """Local index of the last bucket fully at or before 1m time ``t`` (-1 if none)."""

        return int(np.searchsorted(self.bucket_ends, int(t), side="right")) - 1

    def minute_positions(self, start_g: int, count: int) -> np.ndarray:
        """Fractional x for ``count`` consecutive minutes from ``start_g``, clipped."""

        gs = np.arange(int(start_g), int(start_g) + int(count), dtype=np.int64)
        return np.clip(self.global_to_local_f(gs), -1.0, float(self.n_display))
