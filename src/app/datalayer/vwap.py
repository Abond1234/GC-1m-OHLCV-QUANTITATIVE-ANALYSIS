"""VWAP reconstruction for chart overlays.

The bars table stores no VWAP price - only ATR-normalised distance features - so
the overlay lines must be recomputed. The recipe matches the feature engineering:
``typical = (high + low + close) / 3``, ``pv = typical * volume``. Rolling VWAP is
a segment-respecting rolling ratio of the two sums (reset at each
``continuous_segment_id``); research-day VWAP is a per-NY-date cumulative ratio;
execution-session VWAP is cumulative within the Asia/NY execution windows. The
rolling reconstruction is validated against the stored
``distance_from_rolling_vwap_*_atr`` feature in the tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bar_store import BarStore

_ASIA_START, _ASIA_END = 180, 360  # minute-of-day NY: [03:00, 06:00)
_NY_START, _NY_END = 420, 720  # [07:00, 12:00)


def _typical_price_volume(store: BarStore) -> tuple[np.ndarray, np.ndarray]:
    typical = (store.high + store.low + store.close) / 3.0
    return typical * store.volume, store.volume.astype(np.float64)


def rolling_vwap(store: BarStore, window: int) -> np.ndarray:
    """Segment-respecting rolling VWAP over ``window`` bars."""

    pv, vol = _typical_price_volume(store)
    frame = pd.DataFrame({"pv": pv, "vol": vol, "seg": store.segment})
    grouped = frame.groupby("seg", sort=False)
    pv_sum = grouped["pv"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    vol_sum = grouped["vol"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = (pv_sum / vol_sum).to_numpy()
    return vwap


def research_day_vwap(store: BarStore) -> np.ndarray:
    """Cumulative VWAP reset each New York trade date."""

    pv, vol = _typical_price_volume(store)
    frame = pd.DataFrame({"pv": pv, "vol": vol, "day": store.trade_date})
    grouped = frame.groupby("day", sort=False)
    pv_cum = grouped["pv"].cumsum()
    vol_cum = grouped["vol"].cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pv_cum / vol_cum).to_numpy()


def _execution_session_code(minute_ny: np.ndarray) -> np.ndarray:
    code = np.zeros(len(minute_ny), dtype=np.int8)
    code[(minute_ny >= _ASIA_START) & (minute_ny < _ASIA_END)] = 1
    code[(minute_ny >= _NY_START) & (minute_ny < _NY_END)] = 2
    return code


def execution_session_vwap(store: BarStore) -> np.ndarray:
    """Cumulative VWAP within each execution session (Asia / NY), NaN elsewhere."""

    pv, vol = _typical_price_volume(store)
    code = _execution_session_code(store.minute_ny)
    frame = pd.DataFrame({"pv": pv, "vol": vol, "day": store.trade_date, "code": code})
    out = np.full(len(store.ts), np.nan)
    active = code != 0
    grouped = frame[active].groupby(["day", "code"], sort=False)
    pv_cum = grouped["pv"].cumsum()
    vol_cum = grouped["vol"].cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out[active.nonzero()[0]] = (pv_cum / vol_cum).to_numpy()
    return out
