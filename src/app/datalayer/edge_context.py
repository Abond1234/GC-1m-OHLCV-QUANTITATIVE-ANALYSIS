"""Validated research features for one trading day, from the frozen formulas.

Computes the four features that ADVANCED through the FES Project 1 locked
Validation batch (Section 6) for every bar of a requested day, by calling the
research module's own low-level helpers - the same functions
``build_scalar_feature_matrix`` uses - so the app can never drift from the
frozen arithmetic. The research verdict governs the display:

    PREDICTIVE_ONLY_NOT_DIRECTIONAL / FROZEN_NO_POLICY

These are descriptive, session-specific research findings about future RANGE
and participation structure, not directional signals, and no threshold is
active. Each feature's evidence is valid in exactly one entry session
(London: F07/F09, New York: F08/F10); the panel dims features outside their
validated session.

The research package import is deliberately lazy (inside functions, on the
worker thread): ``src.statistical_research.__init__`` eagerly pulls matplotlib
and pyarrow, which must never happen on the GUI thread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .partitions import EVAL_CAP_DATE
from .paths import research_bars_path

# The 15 physical source columns the frozen continuity identity reads
# (source_row_id is synthesized as a monotonic index, as the research's own
# test fixtures do).
_EDGE_COLUMNS = [
    "ts_event_utc",
    "trade_date_ny",
    "product",
    "symbol",
    "active_symbol",
    "instrument_id",
    "continuous_segment_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tradable_research_flag",
    "roll_window_flag",
    "low_liquidity_warning_flag",
]


@dataclass(frozen=True)
class EdgeFeatureInfo:
    """Citation card for one advancing feature (Section 6, locked Validation)."""

    key: str
    label: str
    session: str  # entry session carrying the validated evidence
    target: str
    dev_ic: float
    val_ic: float
    lookback: int  # contiguous legal bars required, including t


EDGE_FEATURES = (
    EdgeFeatureInfo(
        "return_acf_energy_60",
        "Return ACF energy 60",
        "London",
        "future range (60m)",
        0.0306,
        0.0345,
        61,
    ),
    EdgeFeatureInfo(
        "range_volume_spearman_30",
        "Range/volume Spearman 30",
        "London",
        "future range (60m)",
        -0.1019,
        -0.0725,
        89,
    ),
    EdgeFeatureInfo(
        "lagged_volume_return_spearman_30",
        "Lagged volume/return Spearman 30",
        "New York",
        "direction (60m)",
        0.0183,
        0.0314,
        90,
    ),
    EdgeFeatureInfo(
        "volume_profile_slope_30",
        "Volume profile slope 30",
        "New York",
        "future range (60m)",
        0.0437,
        0.0398,
        89,
    ),
)

VERDICT_CAPTION = (
    "Validated research context - PREDICTIVE_ONLY_NOT_DIRECTIONAL. "
    "Range/participation structure, not direction. FROZEN_NO_POLICY: no "
    "threshold is active and nothing here is a trade signal."
)


@dataclass(frozen=True)
class EdgeDayContext:
    """Per-bar values (float32, research fidelity) and missing causes for one day."""

    trade_date: object
    minute_ny: np.ndarray
    values: dict[str, np.ndarray]  # key -> (n,) float32, NaN where not computable
    causes: dict[str, np.ndarray]  # key -> (n,) object, "" when a value exists


def compute_edge_context(frame: pd.DataFrame, trade_date) -> EdgeDayContext:
    """Compute the four advancing features for every bar of ``frame``.

    ``frame`` must hold one GC day's rows with the 15 physical source columns
    (plus optionally ``source_row_id``); the frozen research helpers do all the
    arithmetic, row-vectorized over every bar of the day at once.
    """

    from src.statistical_research.fes_project1_features import (
        _acf_energy,
        _as_float,
        _lag,
        _legal_continuity,
        _profile_slope,
        _rolling_mean,
        _rolling_std_population,
        _safe_divide,
        _spearman_rows,
        _window,
    )

    bars = frame.reset_index(drop=True)
    if "source_row_id" not in bars.columns:
        bars = bars.assign(source_row_id=np.arange(len(bars), dtype=np.int64))

    n = len(bars)
    close = _as_float(bars["close"])
    high = _as_float(bars["high"])
    low = _as_float(bars["low"])
    volume = _as_float(bars["volume"])
    legal_group, legal_position, _latest_cause = _legal_continuity(bars)

    # Intermediates exactly as the frozen builder derives them.
    previous_close = _lag(close, 1, legal_group)
    returns_bps = np.full(n, np.nan, dtype=np.float64)
    valid_return = (
        np.isfinite(close) & np.isfinite(previous_close) & (close > 0.0) & (previous_close > 0.0)
    )
    returns_bps[valid_return] = 10_000.0 * np.log(
        close[valid_return] / previous_close[valid_return]
    )
    bar_range = high - low
    true_range = np.maximum.reduce(
        [bar_range, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    true_range[legal_position == 0] = bar_range[legal_position == 0]
    atr20 = _rolling_mean(true_range, 20, legal_group)
    log_volume = np.log1p(volume)
    log_volume_mean60 = _rolling_mean(log_volume, 60, legal_group)
    log_volume_std60 = _rolling_std_population(log_volume, 60, legal_group)
    volume_zscore60 = _safe_divide(log_volume - log_volume_mean60, log_volume_std60)

    ends = np.arange(n, dtype=np.int64)
    values: dict[str, np.ndarray] = {}
    causes: dict[str, np.ndarray] = {}

    def _finish(key, lookback, out, degenerate, window_ok):
        history_ok = legal_position >= (lookback - 1)
        cause = np.full(n, "", dtype=object)
        cause[~history_ok] = "INSUFFICIENT_LEGAL_HISTORY"
        cause[history_ok & ~window_ok] = "SOURCE_INPUT_NULL"
        if degenerate is not None:
            cause[history_ok & window_ok & degenerate] = "DEGENERATE_STATISTIC"
        final = np.where(cause == "", out, np.nan)
        values[key] = final.astype(np.float32)
        causes[key] = cause

    # F07: sqrt(mean of squared lag-1..5 return autocorrelations) over 60 bps returns.
    r60 = _window(returns_bps, ends, 60)
    r60_ok = np.isfinite(r60).all(axis=1)
    f07, f07_degenerate = _acf_energy(np.nan_to_num(r60))
    _finish("return_acf_energy_60", 61, f07, f07_degenerate, r60_ok)

    # F08: Spearman of ZV[t-30..t-1] against r[t-29..t] (note the one-bar lag).
    zv_lag30 = _window(volume_zscore60, ends - 1, 30)
    r30 = _window(returns_bps, ends, 30)
    f08_ok = np.isfinite(zv_lag30).all(axis=1) & np.isfinite(r30).all(axis=1)
    f08, f08_degenerate = _spearman_rows(np.nan_to_num(zv_lag30), np.nan_to_num(r30))
    _finish("lagged_volume_return_spearman_30", 90, f08, f08_degenerate, f08_ok)

    # F09: Spearman of TR/ATR20 against ZV over the last 30 bars; every ATR20
    # in the window must be finite and positive.
    tr30 = _window(true_range, ends, 30)
    atr30 = _window(atr20, ends, 30)
    zv30 = _window(volume_zscore60, ends, 30)
    atr_ok = np.isfinite(atr30).all(axis=1) & (np.nan_to_num(atr30) > 0.0).all(axis=1)
    f09_inputs_ok = np.isfinite(tr30).all(axis=1) & np.isfinite(zv30).all(axis=1)
    ratio = np.where(atr_ok[:, None], np.nan_to_num(tr30) / np.where(atr30 > 0, atr30, 1.0), 0.0)
    f09, f09_degenerate = _spearman_rows(ratio, np.nan_to_num(zv30))
    history_ok9 = legal_position >= 88
    cause9 = np.full(n, "", dtype=object)
    cause9[~history_ok9] = "INSUFFICIENT_LEGAL_HISTORY"
    cause9[history_ok9 & ~f09_inputs_ok] = "SOURCE_INPUT_NULL"
    cause9[history_ok9 & f09_inputs_ok & ~atr_ok] = "NONPOSITIVE_ATR"
    cause9[history_ok9 & f09_inputs_ok & atr_ok & f09_degenerate] = "DEGENERATE_STATISTIC"
    final9 = np.where(cause9 == "", f09, np.nan)
    values["range_volume_spearman_30"] = final9.astype(np.float32)
    causes["range_volume_spearman_30"] = cause9

    # F10: OLS slope of ZV over the last 30 bars on x = linspace(-1, 1, 30).
    f10_ok = np.isfinite(zv30).all(axis=1)
    f10 = _profile_slope(np.nan_to_num(zv30))
    _finish("volume_profile_slope_30", 89, f10, None, f10_ok)

    return EdgeDayContext(
        trade_date=trade_date,
        minute_ny=bars["ts_event_utc"].to_numpy(),  # placeholder; replaced by caller
        values=values,
        causes=causes,
    )


class EdgeContextService:
    """On-demand, per-day computation of the advancing features (worker thread)."""

    def __init__(self, bars_path=None):
        self._bars_path = bars_path or research_bars_path()
        self._cache: dict[str, EdgeDayContext] = {}

    def day_context(self, trade_date, minute_ny: np.ndarray) -> EdgeDayContext:
        """Fetch + compute (cached per date). ``minute_ny`` labels the day's bars."""

        key = str(pd.Timestamp(trade_date).date())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        import pyarrow.parquet as pq

        day = pd.Timestamp(trade_date).to_pydatetime()
        table = pq.read_table(
            self._bars_path,
            columns=_EDGE_COLUMNS,
            filters=[
                ("product", "==", "GC"),
                ("trade_date_ny", "==", day),
                ("trade_date_ny", "<=", pd.Timestamp(EVAL_CAP_DATE).to_pydatetime()),
            ],
        )
        frame = table.to_pandas(ignore_metadata=True).sort_values("ts_event_utc")
        context = compute_edge_context(frame, trade_date)
        context = EdgeDayContext(
            trade_date=context.trade_date,
            minute_ny=np.asarray(minute_ny),
            values=context.values,
            causes=context.causes,
        )
        self._cache[key] = context
        return context
