"""In-memory GC bar store for the chart and simulator.

Loads only the narrow set of columns needed to render candles and re-derive
fills, filtered to GC and capped at the Validation boundary (Final-test rows are
never read). Holds numpy arrays (one resident copy) and serves viewport slices
split at continuous-segment breaks so nothing is drawn across a discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .partitions import EVAL_CAP_DATE, assert_dev_val_only

_BAR_COLUMNS = [
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
]


@dataclass
class BarStore:
    """Resident GC bar arrays plus viewport helpers."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    atr20: np.ndarray
    segment: np.ndarray
    trade_date: np.ndarray
    minute_ny: np.ndarray

    @property
    def n_bars(self) -> int:
        return len(self.ts)

    def bar_arrays(self) -> dict:
        """The exact dict shape the verified simulator consumes."""

        return {
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "segment": self.segment,
            "trade_date": self.trade_date,
            "minute_ny": self.minute_ny,
        }

    def position_at_ts(self, ts) -> int:
        """Index of the bar at or immediately after ``ts`` (clamped)."""

        pos = int(np.searchsorted(self.ts, np.datetime64(pd.Timestamp(ts))))
        return max(0, min(pos, self.n_bars - 1))

    def segment_bounds(self, i_lo: int, i_hi: int) -> list[tuple[int, int]]:
        """Contiguous (start, end) index pairs of constant segment within a range.

        Each pair is a run the chart can draw as one candlestick item; the gaps
        between pairs are continuous-segment breaks (roll/data discontinuities).
        """

        i_lo = max(0, int(i_lo))
        i_hi = min(self.n_bars - 1, int(i_hi))
        if i_hi < i_lo:
            return []
        seg = self.segment[i_lo : i_hi + 1]
        change = np.nonzero(np.diff(seg) != 0)[0]
        starts = [i_lo, *(i_lo + change + 1)]
        ends = [*(i_lo + change), i_hi]
        return list(zip(starts, ends, strict=True))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> BarStore:
        """Build a store from an already-loaded GC bar frame (used by tests)."""

        frame = frame.sort_values("ts_event_utc").reset_index(drop=True)
        assert_dev_val_only(frame["trade_date_ny"])
        return cls(
            ts=frame["ts_event_utc"].to_numpy(),
            open=frame["open"].to_numpy(dtype=np.float64),
            high=frame["high"].to_numpy(dtype=np.float64),
            low=frame["low"].to_numpy(dtype=np.float64),
            close=frame["close"].to_numpy(dtype=np.float64),
            volume=frame["volume"].to_numpy(dtype=np.float64),
            atr20=frame["rolling_atr_20m"].to_numpy(dtype=np.float64),
            segment=frame["continuous_segment_id"].to_numpy(),
            trade_date=frame["trade_date_ny"].to_numpy(),
            minute_ny=frame["minute_of_day_ny"].to_numpy(dtype=np.int64),
        )

    @classmethod
    def load(
        cls,
        bars_path: Path,
        *,
        cap_date: pd.Timestamp = EVAL_CAP_DATE,
        product: str = "GC",
        date_floor: pd.Timestamp | None = None,
    ) -> BarStore:
        """Load GC bars through the Validation cap; Final-test rows are never read.

        ``date_floor`` optionally narrows the read to ``trade_date_ny >= date_floor``.
        It can only shrink the Dev/Val window (never cross ``cap_date``), so it is a
        pure speed knob for loading a handful of recent days - e.g. the offscreen
        render harness - and leaves the default full-window read unchanged.
        """

        import pyarrow.parquet as pq

        filters = [
            ("product", "==", product),
            ("trade_date_ny", "<=", pd.Timestamp(cap_date).to_pydatetime()),
        ]
        if date_floor is not None:
            filters.append(("trade_date_ny", ">=", pd.Timestamp(date_floor).to_pydatetime()))
        table = pq.read_table(bars_path, columns=_BAR_COLUMNS, filters=filters)
        frame = table.to_pandas(ignore_metadata=True)
        return cls.from_frame(frame)
