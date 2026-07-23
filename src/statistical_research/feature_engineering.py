"""Causal and memory-conscious Section 6 feature construction for GC.

The builder consumes the trusted full GC bar sequence and a key-only eligible
observation frame.  It never accepts a forward-label dataframe.  Rolling
features are calculated across the full bar population, reset at every invalid
continuity boundary, and only then mapped to completed decision bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd

from .feature_registry import (
    EXPERIMENTAL_FEATURE_NAMES,
    FEATURE_NAMES,
    feature_registry_frame,
    validate_registry,
)

GC_TICK_SIZE = 0.10
EXPECTED_OBSERVATION_ROWS = 586_530
REFERENCE_MINIMUM_OBSERVATIONS = 200
REFERENCE_MINIMUM_DATES = 30
WEEKDAY_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
SESSION_ORDER = ("London", "New York")
PARTITION_ORDER = ("Development", "Validation", "Final test")

FEATURE_SOURCE_COLUMNS = (
    "source_row_id",
    "ts_event_utc",
    "ts_event_ny",
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
)

METADATA_COLUMNS = (
    "observation_id",
    "decision_timestamp_utc",
    "decision_timestamp_ny",
    "entry_timestamp_utc",
    "entry_timestamp_ny",
    "trade_date_ny",
    "research_partition",
    "product",
    "symbol",
    "active_symbol",
    "instrument_id",
    "continuous_segment_id",
)

OBSERVATION_REQUIRED_COLUMNS = METADATA_COLUMNS + (
    "decision_bar_id",
    "entry_bar_id",
    "entry_session",
    "minutes_to_forced_exit",
)

FORBIDDEN_SOURCE_TOKENS = (
    "forward",
    "future",
    "mfe",
    "mae",
    "direction_label",
    "expansion_label",
    "target",
    "stop",
    "poi",
)

BOUNDARY_REASON_LABELS: Mapping[int, str] = {
    1: "source_start",
    2: "timestamp_gap",
    3: "product_change",
    4: "selected_contract_change",
    5: "instrument_change",
    6: "continuous_segment_change",
    7: "tradability_roll_or_liquidity_boundary",
}


@dataclass
class FeatureBuildResult:
    matrix: pd.DataFrame
    registry: pd.DataFrame
    reference_parameters: pd.DataFrame
    build_timings: pd.DataFrame
    construction_audit: pd.DataFrame
    observation_audit: pd.DataFrame
    estimated_unoptimized_memory_bytes: int
    optimized_memory_bytes: int
    estimated_peak_working_memory_bytes: int


def _as_float(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    return np.asarray(values, dtype=np.float64)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(
        np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float64
    )
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    np.divide(numerator, denominator, out=result, where=valid)
    result[~np.isfinite(result)] = np.nan
    return result


def _lag(values: np.ndarray, periods: int, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if periods <= 0:
        return values.astype(np.float64, copy=True)
    valid = groups[periods:] == groups[:-periods]
    target = np.arange(periods, len(values), dtype=np.int64)[valid]
    result[target] = np.asarray(values[:-periods], dtype=np.float64)[valid]
    return result


def _window_mask(
    length: int, window: int, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ends = np.arange(window - 1, length, dtype=np.int64)
    starts = ends - window + 1
    valid_group = groups[ends] == groups[starts]
    return ends, starts, valid_group


def _rolling_sum(values: np.ndarray, window: int, groups: np.ndarray) -> np.ndarray:
    """Grouped rolling sum over complete windows, on the deterministic CPU path.

    This produces a frozen research artefact, so it deliberately does not
    dispatch to a GPU array module: a device scan and a host scan over the same
    float64 array can disagree in the last bits, which would make the committed
    feature matrix depend on which machine built it.  See :mod:`src.compute` for
    the project's determinism policy.

    Index and count arrays are int64 so that a group key is never truncated:
    windows must not be joined across a continuity boundary because two
    distinct run identifiers collided in a narrower integer type.
    """

    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    sums = np.zeros(n + 1, dtype=np.float64)
    counts = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.where(finite, values, 0.0), out=sums[1:])
    np.cumsum(finite, dtype=np.int64, out=counts[1:])
    ends, starts, valid_group = _window_mask(n, window, groups)
    complete = valid_group & ((counts[ends + 1] - counts[starts]) == window)
    selected = ends[complete]
    selected_starts = starts[complete]
    result[selected] = sums[selected + 1] - sums[selected_starts]
    return result


def _rolling_mean(values: np.ndarray, window: int, groups: np.ndarray) -> np.ndarray:
    return _rolling_sum(values, window, groups) / float(window)


def _rolling_std(values: np.ndarray, window: int, groups: np.ndarray) -> np.ndarray:
    mean = _rolling_mean(values, window, groups)
    mean_square = _rolling_mean(np.asarray(values, dtype=np.float64) ** 2, window, groups)
    variance = mean_square - mean**2
    variance[np.isfinite(variance) & (variance < 0) & (variance > -1.0e-12)] = 0.0
    result = np.sqrt(np.where(variance >= 0, variance, np.nan))
    return result


def _rolling_extreme(values: np.ndarray, window: int, groups: np.ndarray, kind: str) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64), copy=False)
    roller = series.rolling(window=window, min_periods=window)
    result = (roller.max() if kind == "max" else roller.min()).to_numpy(dtype=np.float64, copy=True)
    ends, _, valid_group = _window_mask(len(values), window, groups)
    invalid_ends = ends[~valid_group]
    result[invalid_ends] = np.nan
    return result


def _rolling_correlation(
    x: np.ndarray, y: np.ndarray, window: int, groups: np.ndarray
) -> np.ndarray:
    sx = _rolling_sum(x, window, groups)
    sy = _rolling_sum(y, window, groups)
    sxx = _rolling_sum(np.asarray(x) ** 2, window, groups)
    syy = _rolling_sum(np.asarray(y) ** 2, window, groups)
    sxy = _rolling_sum(np.asarray(x) * np.asarray(y), window, groups)
    covariance_n = window * sxy - sx * sy
    variance_x_n = window * sxx - sx**2
    variance_y_n = window * syy - sy**2
    denominator = np.sqrt(np.maximum(variance_x_n, 0.0) * np.maximum(variance_y_n, 0.0))
    result = _safe_divide(covariance_n, denominator)
    finite = np.isfinite(result)
    result[finite] = np.clip(result[finite], -1.0, 1.0)
    return result


def _rolling_ols(
    values: np.ndarray, window: int, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    global_x = np.arange(len(values), dtype=np.float64)
    sum_y = _rolling_sum(values, window, groups)
    sum_global_xy = _rolling_sum(values * global_x, window, groups)
    sum_y2 = _rolling_sum(values**2, window, groups)
    starts = global_x - (window - 1)
    sum_xy = sum_global_xy - starts * sum_y
    sum_x = window * (window - 1) / 2.0
    sum_x2 = window * (window - 1) * (2 * window - 1) / 6.0
    sxx = window * sum_x2 - sum_x**2
    covariance_n = window * sum_xy - sum_x * sum_y
    slope = covariance_n / sxx
    syy = window * sum_y2 - sum_y**2
    r_squared = _safe_divide(covariance_n**2, sxx * syy)
    finite = np.isfinite(r_squared)
    r_squared[finite] = np.clip(r_squared[finite], 0.0, 1.0)
    return slope, r_squared


def _signed_streak(price_change: np.ndarray, groups: np.ndarray, cap: int = 120) -> np.ndarray:
    """Linear-time segmented run length; zero changes reset the streak."""

    signs = np.sign(np.asarray(price_change, dtype=np.float64))
    result = np.full(len(signs), np.nan, dtype=np.float64)
    previous_sign = 0.0
    length = 0
    for i, sign in enumerate(signs):
        if not np.isfinite(sign):
            previous_sign = 0.0
            length = 0
            continue
        if i == 0 or groups[i] != groups[i - 1] or sign == 0:
            length = 0 if sign == 0 else 1
        elif sign == previous_sign:
            length = min(length + 1, cap)
        else:
            length = 1
        result[i] = sign * length
        previous_sign = sign
    return result


def _compression_age(
    condition: np.ndarray, valid: np.ndarray, groups: np.ndarray, cap: int = 60
) -> np.ndarray:
    result = np.full(len(condition), np.nan, dtype=np.float64)
    age = 0
    for i in range(len(condition)):
        if not valid[i]:
            age = 0
            continue
        if i == 0 or groups[i] != groups[i - 1] or not condition[i]:
            age = 1 if condition[i] else 0
        else:
            age = min(age + 1, cap)
        result[i] = age
    return result


def _group_ids(run_id: np.ndarray, key: np.ndarray) -> np.ndarray:
    start = np.ones(len(run_id), dtype=bool)
    start[1:] = (run_id[1:] != run_id[:-1]) | (key[1:] != key[:-1])
    return np.cumsum(start, dtype=np.int64) - 1


def build_continuity_run_id(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the locked continuity run and deterministic first-boundary reason."""

    timestamp_ns = (
        pd.to_datetime(bars["ts_event_utc"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    product = bars["product"].astype(str).to_numpy()
    symbol = bars["symbol"].astype(str).to_numpy()
    active_symbol = bars["active_symbol"].astype(str).to_numpy()
    instrument = pd.to_numeric(bars["instrument_id"]).to_numpy(dtype=np.int64)
    segment = pd.to_numeric(bars["continuous_segment_id"]).to_numpy(dtype=np.int64)
    tradable = bars["tradable_research_flag"].fillna(False).to_numpy(dtype=bool)
    roll_clear = ~bars["roll_window_flag"].fillna(True).to_numpy(dtype=bool)
    liquidity_clear = ~bars["low_liquidity_warning_flag"].fillna(True).to_numpy(dtype=bool)
    valid_bar = tradable & roll_clear & liquidity_clear

    n = len(bars)
    start = np.ones(n, dtype=bool)
    reason = np.ones(n, dtype=np.uint8)
    if n > 1:
        consecutive = np.diff(timestamp_ns) == 60_000_000_000
        same_product = product[1:] == product[:-1]
        same_contract = (symbol[1:] == symbol[:-1]) & (active_symbol[1:] == active_symbol[:-1])
        same_instrument = instrument[1:] == instrument[:-1]
        same_segment = segment[1:] == segment[:-1]
        valid_boundary = valid_bar[1:] & valid_bar[:-1]
        continuation = (
            consecutive
            & same_product
            & same_contract
            & same_instrument
            & same_segment
            & valid_boundary
        )
        start[1:] = ~continuation
        reason[1:] = 0
        failures = (
            (~consecutive, 2),
            (consecutive & ~same_product, 3),
            (consecutive & same_product & ~same_contract, 4),
            (consecutive & same_product & same_contract & ~same_instrument, 5),
            (consecutive & same_product & same_contract & same_instrument & ~same_segment, 6),
            (
                consecutive
                & same_product
                & same_contract
                & same_instrument
                & same_segment
                & ~valid_boundary,
                7,
            ),
        )
        for mask, code in failures:
            target = np.flatnonzero(mask & start[1:]) + 1
            reason[target] = code
    run_id = np.cumsum(start, dtype=np.int64) - 1
    run_starts = np.maximum.accumulate(np.where(start, np.arange(n, dtype=np.int64), 0))
    run_position = np.arange(n, dtype=np.int64) - run_starts
    return run_id, run_position, reason


def _fit_time_of_day_reference(
    observations: pd.DataFrame,
    log_volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    session = observations["entry_session"].astype(str).to_numpy()
    partition = observations["research_partition"].astype(str).to_numpy()
    timestamp_ny = pd.DatetimeIndex(pd.to_datetime(observations["entry_timestamp_ny"]))
    minute = (timestamp_ny.hour * 60 + timestamp_ny.minute).astype(np.int16)
    bin_start = (minute // 15 * 15).astype(np.int16)
    dates = pd.to_datetime(observations["trade_date_ny"]).to_numpy(dtype="datetime64[D]")
    development = partition == "Development"

    dev = pd.DataFrame(
        {
            "entry_session": session[development],
            "entry_15m_bin_start_ny": bin_start[development],
            "trade_date_ny": dates[development],
            "log_volume": log_volume[development],
        }
    )
    dev["log_volume_squared"] = dev["log_volume"] ** 2
    totals = (
        dev.groupby(["entry_session", "entry_15m_bin_start_ny"], observed=True, sort=True)
        .agg(
            development_observation_count=("log_volume", "count"),
            development_date_count=("trade_date_ny", "nunique"),
            sum_log_volume=("log_volume", "sum"),
            sumsq_log_volume=("log_volume_squared", "sum"),
        )
        .reset_index()
    )
    n = totals["development_observation_count"].to_numpy(dtype=np.float64)
    sums = totals["sum_log_volume"].to_numpy(dtype=np.float64)
    sumsqs = totals["sumsq_log_volume"].to_numpy(dtype=np.float64)
    means = sums / n
    variances = np.maximum((sumsqs - sums**2 / n) / np.maximum(n - 1.0, 1.0), 0.0)
    stds = np.sqrt(variances)
    totals["mean_log_volume"] = means
    totals["std_log_volume"] = stds
    totals["geometric_volume_scale"] = np.exp(means)
    totals["minimum_observations_required"] = REFERENCE_MINIMUM_OBSERVATIONS
    totals["minimum_dates_required"] = REFERENCE_MINIMUM_DATES
    totals["fit_scope"] = "Development only"
    totals["development_row_application"] = "leave-one-New-York-trading-date-out"
    totals["validation_final_application"] = "frozen full-Development parameters"
    totals["partial_year_treatment"] = "use observed dates; no annual reweighting"
    totals["version"] = "1.0.0"

    group_index = pd.MultiIndex.from_frame(totals[["entry_session", "entry_15m_bin_start_ny"]])
    row_group_index = pd.MultiIndex.from_arrays([session, bin_start])
    positions = group_index.get_indexer(row_group_index)
    if (positions < 0).any():
        missing = sorted(set(zip(session[positions < 0], bin_start[positions < 0], strict=False)))
        raise ValueError(f"Eligible time-of-day groups lack Development support: {missing}")

    row_n = n[positions]
    row_dates = totals["development_date_count"].to_numpy(dtype=np.int64)[positions]
    row_sum = sums[positions]
    row_sumsq = sumsqs[positions]

    date_totals = (
        dev.groupby(
            ["entry_session", "entry_15m_bin_start_ny", "trade_date_ny"], observed=True, sort=True
        )
        .agg(
            date_count=("log_volume", "count"),
            date_sum=("log_volume", "sum"),
            date_sumsq=("log_volume_squared", "sum"),
        )
        .reset_index()
    )
    dev_row_index = pd.MultiIndex.from_arrays(
        [session[development], bin_start[development], dates[development]]
    )
    date_index = pd.MultiIndex.from_frame(
        date_totals[["entry_session", "entry_15m_bin_start_ny", "trade_date_ny"]]
    )
    date_positions = date_index.get_indexer(dev_row_index)
    if (date_positions < 0).any():
        raise RuntimeError("Development leave-one-date-out aggregates failed to map.")

    effective_n = row_n.copy()
    effective_sum = row_sum.copy()
    effective_sumsq = row_sumsq.copy()
    effective_dates = row_dates.copy()
    effective_n[development] -= date_totals["date_count"].to_numpy(dtype=np.float64)[date_positions]
    effective_sum[development] -= date_totals["date_sum"].to_numpy(dtype=np.float64)[date_positions]
    effective_sumsq[development] -= date_totals["date_sumsq"].to_numpy(dtype=np.float64)[
        date_positions
    ]
    effective_dates[development] -= 1

    effective_mean = _safe_divide(effective_sum, effective_n)
    centered_ss = effective_sumsq - _safe_divide(effective_sum**2, effective_n)
    effective_variance = _safe_divide(np.maximum(centered_ss, 0.0), effective_n - 1.0)
    effective_std = np.sqrt(effective_variance)
    supported = (
        (effective_n >= REFERENCE_MINIMUM_OBSERVATIONS)
        & (effective_dates >= REFERENCE_MINIMUM_DATES)
        & np.isfinite(effective_mean)
        & np.isfinite(effective_std)
        & (effective_std > 0)
    )
    zscore = _safe_divide(log_volume - effective_mean, effective_std)
    relative = np.exp(log_volume - effective_mean)
    zscore[~supported | ~np.isfinite(zscore)] = np.nan
    relative[~supported | ~np.isfinite(relative)] = np.nan
    return zscore, relative, totals


def _categorical(values: pd.Series | np.ndarray, categories: tuple[str, ...]) -> pd.Categorical:
    return pd.Categorical(pd.Series(values).astype(str), categories=list(categories), ordered=True)


def build_feature_matrix(
    full_gc_bars: pd.DataFrame,
    eligible_observations: pd.DataFrame,
    *,
    expected_rows: int | None = EXPECTED_OBSERVATION_ROWS,
) -> FeatureBuildResult:
    """Build all registered features on full GC history and map decision-bar values."""

    started = perf_counter()
    timings: list[dict[str, float | str | int]] = []

    def mark(stage: str, stage_start: float) -> float:
        now = perf_counter()
        timings.append({"stage": stage, "seconds": now - stage_start})
        return now

    registry = feature_registry_frame()
    validate_registry(registry)
    missing_source = sorted(set(FEATURE_SOURCE_COLUMNS).difference(full_gc_bars.columns))
    missing_observation = sorted(
        set(OBSERVATION_REQUIRED_COLUMNS).difference(eligible_observations.columns)
    )
    if missing_source or missing_observation:
        raise ValueError(
            f"Missing feature inputs: source={missing_source}, observations={missing_observation}"
        )
    supplied_source_columns = tuple(full_gc_bars.columns)
    forbidden_supplied = [
        name
        for name in supplied_source_columns
        if any(token in name.lower() for token in FORBIDDEN_SOURCE_TOKENS)
    ]
    if forbidden_supplied:
        raise ValueError(
            f"Forward/outcome/POI columns entered the feature source namespace: {forbidden_supplied}"
        )
    if expected_rows is not None and len(eligible_observations) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} eligible observations; received {len(eligible_observations):,}."
        )
    if set(full_gc_bars["product"].astype(str).unique()) != {"GC"}:
        raise ValueError("Section 6 feature construction is GC-only.")
    if not eligible_observations["observation_id"].is_unique:
        raise ValueError("Eligible observation identifiers must be unique.")

    stage = perf_counter()
    bars = full_gc_bars.loc[:, FEATURE_SOURCE_COLUMNS]
    if (
        not bars["source_row_id"].is_monotonic_increasing
        or not pd.to_datetime(bars["ts_event_utc"], utc=True).is_monotonic_increasing
    ):
        bars = bars.sort_values(["ts_event_utc", "source_row_id"], kind="mergesort").reset_index(
            drop=True
        )
        sorted_input = True
    else:
        bars = bars.reset_index(drop=True)
        sorted_input = False
    source_ids = pd.to_numeric(bars["source_row_id"]).to_numpy(dtype=np.int64)
    if pd.Index(source_ids).has_duplicates:
        raise ValueError("GC source_row_id must be unique.")
    requested_ids = pd.to_numeric(eligible_observations["decision_bar_id"]).to_numpy(dtype=np.int64)
    positions = np.searchsorted(source_ids, requested_ids)
    exact = (positions < len(source_ids)) & (
        source_ids[np.minimum(positions, len(source_ids) - 1)] == requested_ids
    )
    if not exact.all():
        raise ValueError(
            f"Decision bars are absent from full GC history: {requested_ids[~exact][:10].tolist()}"
        )
    decision_timestamp_ns = (
        pd.to_datetime(eligible_observations["decision_timestamp_utc"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    source_timestamp_ns = (
        pd.to_datetime(bars["ts_event_utc"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    if not np.array_equal(source_timestamp_ns[positions], decision_timestamp_ns):
        raise ValueError(
            "Decision identifiers and decision timestamps do not map to the same trusted source bars."
        )
    stage = mark("input_validation_and_mapping", stage)

    open_ = _as_float(bars["open"])
    high = _as_float(bars["high"])
    low = _as_float(bars["low"])
    close = _as_float(bars["close"])
    volume = _as_float(bars["volume"])
    if not (
        np.isfinite(open_).all()
        and np.isfinite(high).all()
        and np.isfinite(low).all()
        and np.isfinite(close).all()
        and np.isfinite(volume).all()
    ):
        raise ValueError("Approved OHLCV inputs must be finite.")

    run_id, run_position, boundary_reason_code = build_continuity_run_id(bars)
    previous_close = _lag(close, 1, run_id)
    price_change = close - previous_close
    log_return = np.full(len(close), np.nan, dtype=np.float64)
    valid_return = np.isfinite(previous_close) & (previous_close > 0) & (close > 0)
    log_return[valid_return] = np.log(close[valid_return] / previous_close[valid_return])
    absolute_change = np.abs(price_change)
    bar_range = high - low
    body = close - open_
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    true_range = np.maximum.reduce(
        [bar_range, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    first_in_run = run_position == 0
    true_range[first_in_run] = bar_range[first_in_run]
    log_volume = np.log1p(volume)
    typical_price = (high + low + close) / 3.0
    price_volume = typical_price * volume
    stage = mark("continuity_and_shared_primitives", stage)

    feature_arrays: dict[str, pd.Series | pd.Array | pd.Categorical | np.ndarray] = {}

    def put_float(name: str, full_values: np.ndarray) -> None:
        feature_arrays[name] = np.asarray(full_values, dtype=np.float64)[positions].astype(
            np.float32
        )

    def put_bool(name: str, full_values: np.ndarray, valid: np.ndarray) -> None:
        feature_arrays[name] = pd.arrays.BooleanArray(
            np.asarray(full_values, dtype=bool)[positions],
            ~np.asarray(valid, dtype=bool)[positions],
        )

    def put_int(name: str, full_values: np.ndarray, dtype: str) -> None:
        feature_arrays[name] = pd.array(
            np.asarray(full_values, dtype=np.float64)[positions], dtype=dtype
        )

    # Shared causal rolling primitives.
    atr = {window: _rolling_mean(true_range, window, run_id) for window in (5, 20, 30, 60)}
    realized_variance = {
        window: _rolling_sum(log_return**2, window, run_id) for window in (5, 15, 30, 60)
    }
    realized_volatility = {
        window: 10_000.0 * np.sqrt(np.maximum(values, 0.0))
        for window, values in realized_variance.items()
    }
    volume_mean = {window: _rolling_mean(volume, window, run_id) for window in (5, 20, 60)}
    relative_volume_20 = _safe_divide(volume, volume_mean[20])
    relative_volume_60 = _safe_divide(volume, volume_mean[60])
    rolling_log_volume_mean_60 = _rolling_mean(log_volume, 60, run_id)
    rolling_log_volume_std_60 = _rolling_std(log_volume, 60, run_id)
    high_15 = _rolling_extreme(high, 15, run_id, "max")
    low_15 = _rolling_extreme(low, 15, run_id, "min")
    high_60 = _rolling_extreme(high, 60, run_id, "max")
    low_60 = _rolling_extreme(low, 60, run_id, "min")
    stage = mark("shared_rolling_primitives", stage)

    # Price, return, momentum, and multi-horizon alignment.
    horizon_log_returns: dict[int, np.ndarray] = {}
    horizon_displacements_atr: dict[int, np.ndarray] = {}
    for horizon in (1, 5, 15, 30, 60):
        lag_close = _lag(close, horizon, run_id)
        horizon_log_returns[horizon] = np.where(
            np.isfinite(lag_close) & (lag_close > 0) & (close > 0),
            10_000.0 * np.log(close / lag_close),
            np.nan,
        )
        put_float(f"return_{horizon}m_bps", horizon_log_returns[horizon])
        if horizon in (5, 15, 30, 60):
            horizon_displacements_atr[horizon] = _safe_divide(close - lag_close, atr[20])
            put_float(f"return_{horizon}m_atr", horizon_displacements_atr[horizon])
    put_float("absolute_return_1m_atr", _safe_divide(absolute_change, atr[20]))
    put_float(
        "momentum_acceleration_5_15", horizon_log_returns[5] / 5.0 - horizon_log_returns[15] / 15.0
    )
    put_float(
        "momentum_acceleration_15_30",
        horizon_log_returns[15] / 15.0 - horizon_log_returns[30] / 30.0,
    )
    put_int("directional_streak", _signed_streak(price_change, run_id), "Int16")
    put_float("rolling_range_position_15", _safe_divide(close - low_15, high_15 - low_15))
    put_float("rolling_range_position_60", _safe_divide(close - low_60, high_60 - low_60))
    alignment_valid = np.isfinite(horizon_log_returns[5]) & np.isfinite(horizon_log_returns[30])
    alignment = (horizon_log_returns[5] * horizon_log_returns[30]) > 0
    put_bool("momentum_alignment_5_30", alignment, alignment_valid)
    stage = mark("price_return_momentum", stage)

    # Volatility, range, compression, and expansion state.
    for window in (5, 20, 60):
        put_float(f"atr_{window}", atr[window])
    put_float("atr_ratio_5_20", _safe_divide(atr[5], atr[20]))
    put_float("atr_ratio_20_60", _safe_divide(atr[20], atr[60]))
    for window in (5, 15, 30, 60):
        put_float(f"realized_volatility_{window}", realized_volatility[window])
    put_float(
        "realized_volatility_ratio_5_30",
        _safe_divide(realized_volatility[5], realized_volatility[30]),
    )
    put_float(
        "realized_volatility_ratio_15_60",
        _safe_divide(realized_volatility[15], realized_volatility[60]),
    )
    current_range_over_atr = _safe_divide(bar_range, atr[20])
    put_float("current_range_over_atr", current_range_over_atr)
    range_compression_5_30 = _safe_divide(atr[5], atr[30])
    put_float("range_compression_ratio_5_30", range_compression_5_30)
    stage = mark("volatility_range_state", stage)

    # Candle geometry and local price action.
    put_float("signed_body_atr", _safe_divide(body, atr[20]))
    body_to_range = _safe_divide(np.abs(body), bar_range)
    upper_wick_to_range = _safe_divide(upper_wick, bar_range)
    lower_wick_to_range = _safe_divide(lower_wick, bar_range)
    close_location_value = _safe_divide(2.0 * close - high - low, bar_range)
    put_float("body_to_range", body_to_range)
    put_float("upper_wick_to_range", upper_wick_to_range)
    put_float("lower_wick_to_range", lower_wick_to_range)
    put_float("close_location_value", close_location_value)
    previous_high = _lag(high, 1, run_id)
    previous_low = _lag(low, 1, run_id)
    previous_range = _lag(bar_range, 1, run_id)
    pattern_valid = np.isfinite(previous_high) & np.isfinite(previous_low)
    put_bool("inside_bar", (high <= previous_high) & (low >= previous_low), pattern_valid)
    put_bool("outside_bar", (high >= previous_high) & (low <= previous_low), pattern_valid)
    put_float("range_relative_to_previous_bar", _safe_divide(bar_range, previous_range))
    for window, name in ((2, "two_bar_directional_balance"), (3, "three_bar_directional_balance")):
        put_float(
            name,
            _safe_divide(
                _rolling_sum(price_change, window, run_id),
                _rolling_sum(absolute_change, window, run_id),
            ),
        )
    stage = mark("candle_geometry", stage)

    # Volume and activity.  Time-of-day reference features are added after observation mapping.
    put_float("log_volume", log_volume)
    put_float("relative_volume_20", relative_volume_20)
    put_float("relative_volume_60", relative_volume_60)
    put_float(
        "volume_zscore_60",
        _safe_divide(log_volume - rolling_log_volume_mean_60, rolling_log_volume_std_60),
    )
    put_float("volume_acceleration_5_20", _safe_divide(volume_mean[5], volume_mean[20]))
    put_float("volume_per_tick_range", _safe_divide(volume, bar_range / GC_TICK_SIZE))
    put_float("signed_volume_proxy", close_location_value * relative_volume_20)
    put_float("volume_price_alignment_10", _rolling_correlation(log_return, log_volume, 10, run_id))
    stage = mark("volume_activity_rolling", stage)

    # Research-day, rolling, and execution-session VWAP state.
    timestamps_ny = pd.DatetimeIndex(pd.to_datetime(bars["ts_event_ny"]))
    research_day_key = (timestamps_ny - pd.Timedelta(hours=1)).normalize().asi8
    research_group = _group_ids(run_id, research_day_key)
    cumulative_pv = (
        pd.Series(price_volume)
        .groupby(research_group, sort=False)
        .cumsum()
        .to_numpy(dtype=np.float64)
    )
    cumulative_volume = (
        pd.Series(volume).groupby(research_group, sort=False).cumsum().to_numpy(dtype=np.float64)
    )
    research_vwap = _safe_divide(cumulative_pv, cumulative_volume)
    research_distance_atr = _safe_divide(close - research_vwap, atr[20])

    rolling_vwap: dict[int, np.ndarray] = {}
    for window in (20, 60):
        rolling_vwap[window] = _safe_divide(
            _rolling_sum(price_volume, window, run_id), _rolling_sum(volume, window, run_id)
        )
        put_float(
            f"distance_from_rolling_vwap_{window}_atr",
            _safe_divide(close - rolling_vwap[window], atr[20]),
        )

    put_float("distance_from_research_day_vwap_atr", research_distance_atr)
    for window in (5, 15):
        lagged_vwap = _lag(research_vwap, window, research_group)
        put_float(f"vwap_slope_{window}_atr", _safe_divide(research_vwap - lagged_vwap, atr[20]))
    above_vwap = np.where(np.isfinite(research_vwap), (close > research_vwap).astype(float), np.nan)
    put_float("fraction_above_vwap_15", _rolling_mean(above_vwap, 15, research_group))

    minute_ny = (timestamps_ny.hour * 60 + timestamps_ny.minute).astype(np.int16)
    execution_code = np.zeros(len(bars), dtype=np.int8)
    execution_code[(minute_ny >= 180) & (minute_ny < 360)] = 1
    execution_code[(minute_ny >= 420) & (minute_ny < 720)] = 2
    execution_key = timestamps_ny.normalize().asi8 + execution_code.astype(np.int64)
    execution_group = _group_ids(run_id, execution_key)
    execution_valid = execution_code > 0
    execution_cum_pv = (
        pd.Series(price_volume)
        .groupby(execution_group, sort=False)
        .cumsum()
        .to_numpy(dtype=np.float64)
    )
    execution_cum_volume = (
        pd.Series(volume).groupby(execution_group, sort=False).cumsum().to_numpy(dtype=np.float64)
    )
    execution_vwap = _safe_divide(execution_cum_pv, execution_cum_volume)
    execution_open = (
        pd.Series(open_)
        .groupby(execution_group, sort=False)
        .transform("first")
        .to_numpy(dtype=np.float64, copy=True)
    )
    execution_high = (
        pd.Series(high)
        .groupby(execution_group, sort=False)
        .cummax()
        .to_numpy(dtype=np.float64, copy=True)
    )
    execution_low = (
        pd.Series(low)
        .groupby(execution_group, sort=False)
        .cummin()
        .to_numpy(dtype=np.float64, copy=True)
    )
    for values in (execution_vwap, execution_open, execution_high, execution_low):
        values[~execution_valid] = np.nan
    put_float(
        "distance_from_execution_session_vwap_atr", _safe_divide(close - execution_vwap, atr[20])
    )
    put_float("distance_from_session_open_atr", _safe_divide(close - execution_open, atr[20]))
    put_float(
        "session_range_position",
        _safe_divide(close - execution_low, execution_high - execution_low),
    )
    put_float("session_range_over_atr", _safe_divide(execution_high - execution_low, atr[20]))
    stage = mark("vwap_session_state", stage)

    # Trend, persistence, efficiency, autocorrelation, and choppiness.
    for window in (15, 30, 60):
        lagged_close = _lag(close, window, run_id)
        path_length = _rolling_sum(absolute_change, window, run_id)
        efficiency = _safe_divide(np.abs(close - lagged_close), path_length)
        finite = np.isfinite(efficiency)
        efficiency[finite] = np.clip(efficiency[finite], 0.0, 1.0)
        put_float(f"efficiency_ratio_{window}", efficiency)
        slope, r_squared = _rolling_ols(close, window, run_id)
        put_float(f"normalized_ols_slope_{window}", _safe_divide(slope, atr[20]))
        if window == 30:
            put_float("ols_r_squared_30", r_squared)
    direction_sign = np.sign(price_change)
    put_float("directional_persistence_15", np.abs(_rolling_sum(direction_sign, 15, run_id)) / 15.0)
    lagged_return = _lag(log_return, 1, run_id)
    put_float(
        "return_autocorrelation_15", _rolling_correlation(log_return, lagged_return, 14, run_id)
    )
    sign_change = np.where(
        np.isfinite(direction_sign) & np.isfinite(_lag(direction_sign, 1, run_id)),
        (direction_sign != _lag(direction_sign, 1, run_id)).astype(float),
        np.nan,
    )
    put_float("return_sign_change_rate_30", _rolling_mean(sign_change, 29, run_id))
    high_14 = _rolling_extreme(high, 14, run_id, "max")
    low_14 = _rolling_extreme(low, 14, run_id, "min")
    chop_ratio = _safe_divide(_rolling_sum(true_range, 14, run_id), high_14 - low_14)
    choppiness = 100.0 * np.log10(chop_ratio) / np.log10(14.0)
    finite = np.isfinite(choppiness)
    choppiness[finite] = np.clip(choppiness[finite], 0.0, 100.0)
    put_float("choppiness_14", choppiness)
    stage = mark("trend_persistence", stage)

    # Experimental batch.  Thresholds and parameter choices are fixed in the registry.
    activity_weight = np.clip(relative_volume_20, 0.5, 2.0)
    return_energy = log_return**2 * activity_weight
    directional_energy = np.sign(log_return) * return_energy
    put_float(
        "directional_energy_balance_15_exp",
        _safe_divide(
            _rolling_sum(directional_energy, 15, run_id), _rolling_sum(return_energy, 15, run_id)
        ),
    )
    wick_balance = np.zeros(len(bars), dtype=np.float64)
    positive_range = bar_range > 0
    wick_balance[positive_range] = (
        lower_wick[positive_range] - upper_wick[positive_range]
    ) / bar_range[positive_range]
    weighted_wick = wick_balance * activity_weight
    put_float(
        "wick_pressure_balance_10_exp",
        _safe_divide(
            _rolling_sum(weighted_wick, 10, run_id), _rolling_sum(activity_weight, 10, run_id)
        ),
    )
    rv_compression = _safe_divide(realized_volatility[5], realized_volatility[30])
    atr_compression = _safe_divide(atr[5], atr[30])
    compression_valid = np.isfinite(rv_compression) & np.isfinite(atr_compression)
    compression_condition = (rv_compression < 0.70) & (atr_compression < 0.75)
    put_int(
        "compression_age_exp",
        _compression_age(compression_condition, compression_valid, run_id),
        "Int8",
    )

    lagged_research_distance = _lag(research_distance_atr, 1, research_group)
    elasticity = _rolling_regression_slope(
        lagged_research_distance, log_return * 10_000.0, 30, research_group
    )
    put_float("vwap_elasticity_30_exp", elasticity)
    put_float(
        "liquidity_vacuum_score_exp",
        current_range_over_atr
        * np.abs(close_location_value)
        / np.sqrt(np.maximum(relative_volume_20, 0.25)),
    )
    short_return = horizon_displacements_atr[5]
    broad_return = horizon_displacements_atr[30]
    pullback = np.full(len(bars), np.nan, dtype=np.float64)
    pullback_valid = np.isfinite(short_return) & np.isfinite(broad_return)
    opposing = pullback_valid & ((short_return * broad_return) < 0)
    pullback[pullback_valid] = 0.0
    pullback[opposing] = np.sign(broad_return[opposing]) * np.minimum(
        np.abs(short_return[opposing]), np.abs(broad_return[opposing])
    )
    put_float("pullback_tension_5_30_exp", pullback)
    stage = mark("experimental_hypotheses", stage)

    # Known entry schedule and categorical context use t+1 only because the scheduled next minute is known at t.
    entry_timestamp_ny = pd.DatetimeIndex(
        pd.to_datetime(eligible_observations["entry_timestamp_ny"])
    )
    entry_minute = (entry_timestamp_ny.hour * 60 + entry_timestamp_ny.minute).astype(np.int16)
    sessions = eligible_observations["entry_session"].astype(str).to_numpy()
    minute_from_open = np.where(
        sessions == "London", entry_minute - 180, entry_minute - 420
    ).astype(np.int16)
    session_duration_minus_one = np.where(sessions == "London", 179.0, 299.0)
    feature_arrays["entry_session"] = pd.Categorical(
        sessions, categories=list(SESSION_ORDER), ordered=True
    )
    feature_arrays["minute_from_execution_window_open"] = pd.array(minute_from_open, dtype="Int16")
    feature_arrays["session_progress_fraction"] = (
        minute_from_open / session_duration_minus_one
    ).astype(np.float32)
    feature_arrays["minutes_to_noon_entry_cutoff"] = pd.array(720 - entry_minute, dtype="Int16")
    feature_arrays["minutes_to_1530_forced_exit"] = pd.array(
        pd.to_numeric(eligible_observations["minutes_to_forced_exit"]).to_numpy(dtype=np.int16),
        dtype="Int16",
    )
    feature_arrays["new_york_minute_of_day"] = pd.array(entry_minute, dtype="Int16")
    angle = 2.0 * np.pi * entry_minute.astype(np.float64) / 1440.0
    feature_arrays["time_of_day_sin"] = np.sin(angle).astype(np.float32)
    feature_arrays["time_of_day_cos"] = np.cos(angle).astype(np.float32)
    feature_arrays["day_of_week"] = pd.Categorical(
        entry_timestamp_ny.day_name(), categories=list(WEEKDAY_ORDER), ordered=True
    )

    tod_zscore, tod_relative, reference_parameters = _fit_time_of_day_reference(
        eligible_observations, np.asarray(feature_arrays["log_volume"], dtype=np.float64)
    )
    feature_arrays["tod_log_volume_z"] = tod_zscore.astype(np.float32)
    feature_arrays["tod_relative_volume"] = tod_relative.astype(np.float32)
    stage = mark("clock_calendar_and_development_reference", stage)

    missing_calculated = sorted(set(FEATURE_NAMES).difference(feature_arrays))
    unexpected_calculated = sorted(set(feature_arrays).difference(FEATURE_NAMES))
    if missing_calculated or unexpected_calculated:
        raise RuntimeError(
            f"Registry/build mismatch: missing={missing_calculated}, unexpected={unexpected_calculated}"
        )
    features = pd.DataFrame({name: feature_arrays[name] for name in FEATURE_NAMES})

    metadata = eligible_observations.loc[:, METADATA_COLUMNS].copy()
    metadata["observation_id"] = pd.to_numeric(metadata["observation_id"]).astype("int64")
    for column in ("decision_timestamp_utc", "entry_timestamp_utc"):
        metadata[column] = pd.to_datetime(metadata[column], utc=True)
    for column in ("decision_timestamp_ny", "entry_timestamp_ny"):
        metadata[column] = pd.to_datetime(metadata[column])
    metadata["trade_date_ny"] = pd.to_datetime(metadata["trade_date_ny"])
    metadata["research_partition"] = _categorical(metadata["research_partition"], PARTITION_ORDER)
    for column in ("product", "symbol", "active_symbol"):
        metadata[column] = metadata[column].astype(str).astype("category")
    metadata["instrument_id"] = pd.to_numeric(metadata["instrument_id"]).astype("uint32")
    metadata["continuous_segment_id"] = pd.to_numeric(metadata["continuous_segment_id"]).astype(
        "int32"
    )
    matrix = pd.concat([metadata.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    spec_dtypes = registry.set_index("feature_name")["output_dtype"].astype(str).to_dict()
    dtype_mismatches = {
        name: (spec_dtypes[name], str(matrix[name].dtype))
        for name in FEATURE_NAMES
        if not (
            (spec_dtypes[name] == "float32" and str(matrix[name].dtype) == "float32")
            or (spec_dtypes[name] == "boolean" and str(matrix[name].dtype) == "boolean")
            or (spec_dtypes[name] == "category" and str(matrix[name].dtype) == "category")
            or (spec_dtypes[name] == "Int16" and str(matrix[name].dtype) == "Int16")
            or (spec_dtypes[name] == "Int8" and str(matrix[name].dtype) == "Int8")
        )
    }
    if dtype_mismatches:
        raise TypeError(f"Feature dtype policy mismatch: {dtype_mismatches}")

    run_start_positions = np.flatnonzero(np.r_[True, run_id[1:] != run_id[:-1]])
    run_boundary_reason = boundary_reason_code[run_start_positions]
    observation_audit = pd.DataFrame(
        {
            "observation_id": matrix["observation_id"].to_numpy(dtype=np.int64),
            "source_position": positions,
            "source_row_id": source_ids[positions],
            "continuity_run_id": run_id[positions],
            "bars_since_continuity_start": run_position[positions],
            "continuity_boundary_reason": pd.Categorical(
                [
                    BOUNDARY_REASON_LABELS[int(code)]
                    for code in run_boundary_reason[run_id[positions]]
                ],
                categories=list(BOUNDARY_REASON_LABELS.values()),
            ),
        }
    )

    optimized_memory = int(matrix.memory_usage(index=True, deep=True).sum())
    numeric_specs = registry["output_dtype"].isin(["float32", "Int16", "Int8"])
    numeric_count = int(numeric_specs.sum())
    categorical_boolean_count = len(registry) - numeric_count
    estimated_unoptimized = int(
        len(matrix) * numeric_count * 8
        + len(matrix) * categorical_boolean_count * 8
        + metadata.memory_usage(index=True, deep=True).sum()
    )
    primitive_bytes = sum(
        array.nbytes
        for array in (
            open_,
            high,
            low,
            close,
            volume,
            previous_close,
            price_change,
            log_return,
            absolute_change,
            bar_range,
            body,
            upper_wick,
            lower_wick,
            true_range,
            log_volume,
            typical_price,
            price_volume,
            run_id,
            run_position,
            positions,
        )
    )
    estimated_peak = int(optimized_memory + primitive_bytes + 12 * len(bars) * 8)
    stage = mark("matrix_assembly_and_type_optimization", stage)

    construction_audit = pd.DataFrame(
        [
            {"item": "instrument_scope", "value": "GC only"},
            {"item": "source_rows", "value": str(len(bars))},
            {"item": "source_sorted_during_build", "value": str(sorted_input)},
            {"item": "source_columns", "value": "|".join(FEATURE_SOURCE_COLUMNS)},
            {"item": "forward_label_dataframe_required", "value": "False"},
            {"item": "decision_source_ids_exact", "value": "True"},
            {"item": "decision_timestamps_exact", "value": "True"},
            {"item": "continuity_run_count", "value": str(int(run_id[-1] + 1))},
            {"item": "time_of_day_fit_scope", "value": "Development only"},
            {
                "item": "development_reference_application",
                "value": "leave-one-New-York-trading-date-out",
            },
            {"item": "validation_final_reference_application", "value": "frozen full-Development"},
            {"item": "feature_definition_version", "value": "1.0.0"},
            {"item": "numeric_feature_count", "value": str(numeric_count)},
            {
                "item": "core_feature_count",
                "value": str(len(FEATURE_NAMES) - len(EXPERIMENTAL_FEATURE_NAMES)),
            },
            {"item": "experimental_feature_count", "value": str(len(EXPERIMENTAL_FEATURE_NAMES))},
        ]
    )
    timings.append({"stage": "total", "seconds": perf_counter() - started})
    return FeatureBuildResult(
        matrix=matrix,
        registry=registry,
        reference_parameters=reference_parameters,
        build_timings=pd.DataFrame(timings),
        construction_audit=construction_audit,
        observation_audit=observation_audit,
        estimated_unoptimized_memory_bytes=estimated_unoptimized,
        optimized_memory_bytes=optimized_memory,
        estimated_peak_working_memory_bytes=estimated_peak,
    )


def _rolling_regression_slope(
    predictor: np.ndarray,
    response: np.ndarray,
    window: int,
    groups: np.ndarray,
) -> np.ndarray:
    """Rolling OLS response-on-predictor slope with a zero-variance null policy."""

    sx = _rolling_sum(predictor, window, groups)
    sy = _rolling_sum(response, window, groups)
    sxx = _rolling_sum(np.asarray(predictor) ** 2, window, groups)
    sxy = _rolling_sum(np.asarray(predictor) * np.asarray(response), window, groups)
    covariance_n = window * sxy - sx * sy
    variance_x_n = window * sxx - sx**2
    return _safe_divide(covariance_n, variance_x_n)
