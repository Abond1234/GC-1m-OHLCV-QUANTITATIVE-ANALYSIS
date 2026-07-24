"""Causal feature engineering for the frozen MLAT v1 batch.

The builder accepts only a strict GC OHLCV source namespace.  It computes all
features on the complete, chronologically sorted GC history, resets every
trailing window at the established continuity boundaries, and maps results to
eligible decision bars only after construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd

from .feature_engineering import build_continuity_run_id
from .mlat_feature_registry import (
    MLAT_FEATURE_NAMES,
    build_mlat_registry,
    registry_to_frame,
    validate_mlat_registry,
)

MLAT_FEATURE_SOURCE_COLUMNS = (
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

MLAT_METADATA_COLUMNS = (
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

MLAT_OBSERVATION_REQUIRED_COLUMNS = MLAT_METADATA_COLUMNS + ("decision_bar_id",)
# Notebook-facing aliases retained as explicit contracts.
BAR_INPUT_COLUMNS = MLAT_FEATURE_SOURCE_COLUMNS
FEATURE_METADATA_COLUMNS = MLAT_METADATA_COLUMNS
MLAT_COMPLETE_HISTORY_BARS: Mapping[str, int] = {
    "bollinger_zscore_20": 20,
    "bollinger_bandwidth_20": 20,
    "cutler_rsi_14": 15,
    "chaikin_money_flow_20": 20,
    "amihud_illiquidity_60": 61,
    "parkinson_volatility_30": 30,
    "rogers_satchell_volatility_30": 30,
    "realized_semivariance_balance_60": 61,
    "bipower_jump_ratio_60": 62,
    "variance_ratio_60_5": 65,
    "return_sign_entropy_60": 61,
    "volatility_of_volatility_60": 75,
}

FORBIDDEN_CAUSAL_TOKENS = (
    "entry_price",
    "exit_price_",
    "forward",
    "future",
    "label",
    "mfe",
    "mae",
    "outcome",
    "target",
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


@dataclass(frozen=True)
class MlatFeatureBuildResult:
    """Notebook-friendly MLAT engineering outputs."""

    matrix: pd.DataFrame
    registry: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    persistence_validation: pd.DataFrame
    construction_audit: pd.DataFrame
    observation_audit: pd.DataFrame
    build_timings: pd.DataFrame
    runtime_summary: pd.DataFrame

    @property
    def diagnostics(self) -> pd.DataFrame:
        """Return the primary feature diagnostic table used by the notebook."""

        return self.feature_diagnostics


def _as_float(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    return np.asarray(values, dtype=np.float64)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    left, right = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.full(left.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right) & (right != 0)
    np.divide(left, right, out=result, where=valid)
    result[~np.isfinite(result)] = np.nan
    return result


def _lag(values: np.ndarray, periods: int, groups: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if periods <= 0:
        return np.asarray(values, dtype=np.float64).copy()
    same_group = groups[periods:] == groups[:-periods]
    target = np.arange(periods, len(values), dtype=np.int64)[same_group]
    result[target] = np.asarray(values[:-periods], dtype=np.float64)[same_group]
    return result


def _rolling_sum(values: np.ndarray, window: int, groups: np.ndarray) -> np.ndarray:
    """Complete-window segmented rolling sum without row-wise Python loops."""

    array = np.asarray(values, dtype=np.float64)
    result = np.full(len(array), np.nan, dtype=np.float64)
    if window <= 0:
        raise ValueError("rolling window must be positive")
    if len(array) < window:
        return result
    finite = np.isfinite(array)
    sums = np.zeros(len(array) + 1, dtype=np.float64)
    counts = np.zeros(len(array) + 1, dtype=np.int64)
    np.cumsum(np.where(finite, array, 0.0), out=sums[1:])
    np.cumsum(finite, dtype=np.int64, out=counts[1:])
    ends = np.arange(window - 1, len(array), dtype=np.int64)
    starts = ends - window + 1
    complete = (groups[ends] == groups[starts]) & ((counts[ends + 1] - counts[starts]) == window)
    selected_ends = ends[complete]
    selected_starts = starts[complete]
    result[selected_ends] = sums[selected_ends + 1] - sums[selected_starts]
    return result


def _rolling_mean(values: np.ndarray, window: int, groups: np.ndarray) -> np.ndarray:
    return _rolling_sum(values, window, groups) / float(window)


def _rolling_variance(values: np.ndarray, window: int, groups: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = _rolling_mean(array, window, groups)
    mean_square = _rolling_mean(array**2, window, groups)
    variance = mean_square - mean**2
    variance[np.isfinite(variance) & (variance < 0)] = 0.0
    return variance


def _validate_inputs(
    bars: pd.DataFrame, observations: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    supplied = set(bars.columns)
    allowed = set(MLAT_FEATURE_SOURCE_COLUMNS)
    missing_source = sorted(allowed.difference(supplied))
    extra_source = sorted(supplied.difference(allowed))
    missing_observations = sorted(
        set(MLAT_OBSERVATION_REQUIRED_COLUMNS).difference(observations.columns)
    )
    forbidden_observations = sorted(
        column
        for column in observations.columns
        if any(token in column.lower() for token in FORBIDDEN_CAUSAL_TOKENS)
    )
    if missing_source or extra_source or missing_observations or forbidden_observations:
        raise ValueError(
            "Invalid MLAT engineering namespace: "
            f"missing_source={missing_source}, extra_source={extra_source}, "
            f"missing_observations={missing_observations}, "
            f"forbidden_observations={forbidden_observations}"
        )
    if bars.empty:
        raise ValueError("full_gc_bars must not be empty")
    products = sorted(bars["product"].dropna().astype(str).unique().tolist())
    observation_products = sorted(observations["product"].dropna().astype(str).unique().tolist())
    if products != ["GC"] or observation_products != ["GC"]:
        raise ValueError(
            f"MLAT engineering is GC-only; bars={products}, observations={observation_products}"
        )
    source_ids = pd.to_numeric(bars["source_row_id"], errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(source_ids)) != len(source_ids) or not np.all(np.diff(source_ids) > 0):
        raise ValueError("source_row_id must be unique and strictly increasing")
    timestamps = (
        pd.to_datetime(bars["ts_event_utc"], errors="raise", utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("full_gc_bars must be strictly chronological in UTC")
    observation_ids = pd.to_numeric(observations["observation_id"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if len(np.unique(observation_ids)) != len(observation_ids) or not np.all(
        np.diff(observation_ids) > 0
    ):
        raise ValueError("observation_id must be unique and strictly increasing")
    decision_ids = pd.to_numeric(observations["decision_bar_id"], errors="raise").to_numpy(
        dtype=np.int64
    )
    positions = np.searchsorted(source_ids, decision_ids)
    in_bounds = positions < len(source_ids)
    exact = np.zeros(len(positions), dtype=bool)
    exact[in_bounds] = source_ids[positions[in_bounds]] == decision_ids[in_bounds]
    if not exact.all():
        raise ValueError(
            "decision_bar_id is absent from full GC history; "
            f"examples={decision_ids[~exact][:10].tolist()}"
        )
    if not np.all(np.diff(positions) >= 0):
        raise ValueError("eligible observations must be in decision-bar order")

    bar_utc = pd.to_datetime(bars.iloc[positions]["ts_event_utc"].reset_index(drop=True), utc=True)
    observation_utc = pd.to_datetime(
        observations["decision_timestamp_utc"].reset_index(drop=True), utc=True
    )
    if not bar_utc.equals(observation_utc):
        raise ValueError("decision_bar_id does not match decision_timestamp_utc")
    bar_ny = pd.to_datetime(bars.iloc[positions]["ts_event_ny"].reset_index(drop=True), utc=True)
    observation_ny = pd.to_datetime(
        observations["decision_timestamp_ny"].reset_index(drop=True), utc=True
    )
    if not bar_ny.equals(observation_ny):
        raise ValueError("decision_bar_id does not match decision_timestamp_ny")
    for column in (
        "trade_date_ny",
        "product",
        "symbol",
        "active_symbol",
        "instrument_id",
        "continuous_segment_id",
    ):
        left = bars.iloc[positions][column].reset_index(drop=True).astype(str)
        right = observations[column].reset_index(drop=True).astype(str)
        if not left.equals(right):
            raise ValueError(f"decision_bar_id identity mismatch for {column}")
    return source_ids, positions


def _registry_frame(registry: object | None = None) -> pd.DataFrame:
    definition = build_mlat_registry() if registry is None else registry
    frame = (
        definition.copy() if isinstance(definition, pd.DataFrame) else registry_to_frame(definition)
    )
    validate_mlat_registry(frame)
    if "feature_name" not in frame:
        raise KeyError("MLAT registry frame must contain feature_name")
    names = frame["feature_name"].astype(str).tolist()
    if names != list(MLAT_FEATURE_NAMES):
        raise ValueError(
            f"MLAT registry order mismatch: registry={names}, expected={list(MLAT_FEATURE_NAMES)}"
        )
    return frame


def _compute_features(bars: pd.DataFrame, run_id: np.ndarray) -> dict[str, np.ndarray]:
    open_ = _as_float(bars["open"])
    high = _as_float(bars["high"])
    low = _as_float(bars["low"])
    close = _as_float(bars["close"])
    volume = _as_float(bars["volume"])
    previous_close = _lag(close, 1, run_id)
    valid_return = (
        np.isfinite(close) & np.isfinite(previous_close) & (close > 0) & (previous_close > 0)
    )
    log_return = np.full(len(bars), np.nan, dtype=np.float64)
    log_return[valid_return] = np.log(close[valid_return] / previous_close[valid_return])

    close_mean_20 = _rolling_mean(close, 20, run_id)
    close_std_20 = np.sqrt(_rolling_variance(close, 20, run_id))
    bollinger_zscore = _safe_divide(close - close_mean_20, close_std_20)
    bollinger_bandwidth = _safe_divide(4.0 * close_std_20, close_mean_20)
    bollinger_bandwidth[~(close_mean_20 > 0)] = np.nan

    close_change = close - previous_close
    gains = np.where(np.isfinite(close_change), np.maximum(close_change, 0), np.nan)
    losses = np.where(np.isfinite(close_change), np.maximum(-close_change, 0), np.nan)
    gain_sum_14 = _rolling_sum(gains, 14, run_id)
    loss_sum_14 = _rolling_sum(losses, 14, run_id)
    cutler_rsi = 100.0 * _safe_divide(gain_sum_14, gain_sum_14 + loss_sum_14)

    bar_range = high - low
    valid_ohlc = np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    money_flow_multiplier = np.full(len(bars), np.nan, dtype=np.float64)
    positive_range = valid_ohlc & (bar_range != 0)
    money_flow_multiplier[positive_range] = (
        2.0 * close[positive_range] - high[positive_range] - low[positive_range]
    ) / bar_range[positive_range]
    money_flow_multiplier[valid_ohlc & (bar_range == 0)] = 0.0
    money_flow_volume = money_flow_multiplier * volume
    chaikin_money_flow = _safe_divide(
        _rolling_sum(money_flow_volume, 20, run_id),
        _rolling_sum(volume, 20, run_id),
    )

    amihud_term = np.full(len(bars), np.nan, dtype=np.float64)
    valid_notional = valid_return & np.isfinite(volume) & (volume > 0) & (close > 0)
    amihud_term[valid_notional] = np.abs(log_return[valid_notional]) / (
        close[valid_notional] * volume[valid_notional]
    )
    amihud = 1.0e9 * _rolling_mean(amihud_term, 60, run_id)

    parkinson_term = np.full(len(bars), np.nan, dtype=np.float64)
    valid_high_low = np.isfinite(high) & np.isfinite(low) & (high > 0) & (low > 0)
    parkinson_term[valid_high_low] = np.log(high[valid_high_low] / low[valid_high_low]) ** 2
    parkinson = 1.0e4 * np.sqrt(_rolling_mean(parkinson_term, 30, run_id) / (4.0 * np.log(2.0)))

    rs_term = np.full(len(bars), np.nan, dtype=np.float64)
    valid_positive_ohlc = valid_ohlc & (open_ > 0) & (high > 0) & (low > 0) & (close > 0)
    rs_term[valid_positive_ohlc] = np.log(
        high[valid_positive_ohlc] / open_[valid_positive_ohlc]
    ) * np.log(high[valid_positive_ohlc] / close[valid_positive_ohlc]) + np.log(
        low[valid_positive_ohlc] / open_[valid_positive_ohlc]
    ) * np.log(low[valid_positive_ohlc] / close[valid_positive_ohlc])
    rs_mean = _rolling_mean(rs_term, 30, run_id)
    rs_variance = np.where(
        np.isfinite(rs_mean) & (rs_mean >= -1.0e-12),
        np.maximum(rs_mean, 0.0),
        np.nan,
    )
    rogers_satchell = 1.0e4 * np.sqrt(rs_variance)

    squared_return = log_return**2
    upside = np.where(np.isfinite(log_return), squared_return * (log_return > 0), np.nan)
    downside = np.where(np.isfinite(log_return), squared_return * (log_return < 0), np.nan)
    upside_60 = _rolling_sum(upside, 60, run_id)
    downside_60 = _rolling_sum(downside, 60, run_id)
    semivariance_balance = _safe_divide(upside_60 - downside_60, upside_60 + downside_60)

    realized_variance_60 = _rolling_sum(squared_return, 60, run_id)
    previous_absolute_return = _lag(np.abs(log_return), 1, run_id)
    bipower_product = np.abs(log_return) * previous_absolute_return
    bipower_variation = (np.pi / 2.0) * (60.0 / 59.0) * _rolling_sum(bipower_product, 60, run_id)
    jump_ratio = _safe_divide(
        np.maximum(realized_variance_60 - bipower_variation, 0.0),
        realized_variance_60,
    )
    jump_ratio[np.isfinite(jump_ratio)] = np.clip(jump_ratio[np.isfinite(jump_ratio)], 0.0, 1.0)

    five_return = _rolling_sum(log_return, 5, run_id)
    one_minute_variance = _rolling_variance(log_return, 60, run_id)
    five_minute_variance = _rolling_variance(five_return, 60, run_id)
    variance_ratio = _safe_divide(five_minute_variance, 5.0 * one_minute_variance)

    negative = _rolling_sum(
        np.where(np.isfinite(log_return), (log_return < 0).astype(float), np.nan),
        60,
        run_id,
    )
    zero = _rolling_sum(
        np.where(np.isfinite(log_return), (log_return == 0).astype(float), np.nan),
        60,
        run_id,
    )
    positive = _rolling_sum(
        np.where(np.isfinite(log_return), (log_return > 0).astype(float), np.nan),
        60,
        run_id,
    )
    probabilities = np.column_stack([negative, zero, positive]) / 60.0
    entropy_terms = np.zeros_like(probabilities)
    positive_probability = probabilities > 0
    entropy_terms[positive_probability] = probabilities[positive_probability] * np.log(
        probabilities[positive_probability]
    )
    sign_entropy = -np.sum(entropy_terms, axis=1) / np.log(3.0)
    sign_entropy[~np.isfinite(probabilities).all(axis=1)] = np.nan

    realized_volatility_15 = 1.0e4 * np.sqrt(_rolling_mean(squared_return, 15, run_id))
    rv_mean_60 = _rolling_mean(realized_volatility_15, 60, run_id)
    rv_std_60 = np.sqrt(_rolling_variance(realized_volatility_15, 60, run_id))
    volatility_of_volatility = _safe_divide(rv_std_60, rv_mean_60)
    volatility_of_volatility[~(rv_mean_60 > 0)] = np.nan

    features = {
        "bollinger_zscore_20": bollinger_zscore,
        "bollinger_bandwidth_20": bollinger_bandwidth,
        "cutler_rsi_14": cutler_rsi,
        "chaikin_money_flow_20": chaikin_money_flow,
        "amihud_illiquidity_60": amihud,
        "parkinson_volatility_30": parkinson,
        "rogers_satchell_volatility_30": rogers_satchell,
        "realized_semivariance_balance_60": semivariance_balance,
        "bipower_jump_ratio_60": jump_ratio,
        "variance_ratio_60_5": variance_ratio,
        "return_sign_entropy_60": sign_entropy,
        "volatility_of_volatility_60": volatility_of_volatility,
    }
    if list(features) != list(MLAT_FEATURE_NAMES):
        raise RuntimeError(f"feature implementation order differs from registry: {list(features)}")
    for name, values in features.items():
        if values.dtype != np.float64:
            raise TypeError(f"{name} intermediate must be float64, got {values.dtype}")
        values[~np.isfinite(values)] = np.nan
    return features


def _metadata_frame(observations: pd.DataFrame) -> pd.DataFrame:
    metadata = observations.loc[:, MLAT_METADATA_COLUMNS].copy()
    metadata["observation_id"] = pd.to_numeric(metadata["observation_id"], errors="raise").astype(
        "int64"
    )
    for column in ("decision_timestamp_utc", "entry_timestamp_utc"):
        metadata[column] = pd.to_datetime(metadata[column], errors="raise", utc=True)
    for column in ("decision_timestamp_ny", "entry_timestamp_ny"):
        metadata[column] = pd.to_datetime(metadata[column], errors="raise", utc=True).dt.tz_convert(
            "America/New_York"
        )
    metadata["trade_date_ny"] = pd.to_datetime(metadata["trade_date_ny"], errors="raise")
    metadata["research_partition"] = metadata["research_partition"].astype("category")
    for column in ("product", "symbol", "active_symbol"):
        metadata[column] = metadata[column].astype(str).astype("category")
    metadata["instrument_id"] = pd.to_numeric(metadata["instrument_id"], errors="raise").astype(
        "uint32"
    )
    metadata["continuous_segment_id"] = pd.to_numeric(
        metadata["continuous_segment_id"], errors="raise"
    ).astype("int32")
    return metadata


def _diagnostic_tables(
    features: Mapping[str, np.ndarray],
    positions: np.ndarray,
    run_position: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostics = []
    persistence = []
    construction = []
    for name in MLAT_FEATURE_NAMES:
        full = features[name]
        selected = full[positions]
        persisted = selected.astype(np.float32)
        full_finite = np.isfinite(full)
        selected_finite = np.isfinite(selected)
        persisted_finite = np.isfinite(persisted)
        overflow = selected_finite & ~persisted_finite
        roundtrip = persisted.astype(np.float64)
        error = np.abs(selected[selected_finite] - roundtrip[selected_finite])
        first_valid = int(np.flatnonzero(full_finite)[0]) if full_finite.any() else pd.NA
        diagnostics.append(
            {
                "feature_name": name,
                "complete_history_bars": MLAT_COMPLETE_HISTORY_BARS[name],
                "expected_first_valid_run_position": (MLAT_COMPLETE_HISTORY_BARS[name] - 1),
                "first_valid_full_position": first_valid,
                "full_finite_count": int(full_finite.sum()),
                "mapped_finite_count": int(selected_finite.sum()),
                "mapped_null_count": int((~selected_finite).sum()),
                "mapped_minimum": (
                    float(np.min(selected[selected_finite])) if selected_finite.any() else np.nan
                ),
                "mapped_maximum": (
                    float(np.max(selected[selected_finite])) if selected_finite.any() else np.nan
                ),
                "minimum_mapped_run_position": (
                    int(np.min(run_position[positions][selected_finite]))
                    if selected_finite.any()
                    else pd.NA
                ),
            }
        )
        persistence.append(
            {
                "feature_name": name,
                "intermediate_dtype": str(full.dtype),
                "persisted_dtype": str(persisted.dtype),
                "finite_count_preserved": int(selected_finite.sum()) == int(persisted_finite.sum()),
                "float32_overflow_count": int(overflow.sum()),
                "maximum_absolute_roundtrip_error": (float(error.max()) if len(error) else np.nan),
                "passed": (
                    full.dtype == np.float64
                    and persisted.dtype == np.float32
                    and not overflow.any()
                ),
            }
        )
        construction.append(
            {
                "feature_name": name,
                "calculation_scope": "full sorted GC history",
                "mapping_scope": "exact decision_bar_id after construction",
                "availability": "close of decision bar t",
                "continuity_policy": "complete window within build_continuity_run_id",
                "complete_history_bars": MLAT_COMPLETE_HISTORY_BARS[name],
                "intermediate_dtype": "float64",
                "matrix_dtype": "float32",
            }
        )
    return (
        pd.DataFrame(diagnostics),
        pd.DataFrame(persistence),
        pd.DataFrame(construction),
    )


def build_mlat_feature_matrix(
    bars: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    registry: object | None = None,
) -> MlatFeatureBuildResult:
    """Build the frozen 12-feature MLAT matrix on complete causal GC history."""

    started = perf_counter()
    timing_rows: list[dict[str, object]] = []

    def mark(stage: str, stage_started: float) -> float:
        finished = perf_counter()
        timing_rows.append({"stage": stage, "seconds": float(finished - stage_started)})
        return finished

    stage = started
    registry_frame = _registry_frame(registry)
    source_ids, positions = _validate_inputs(bars, observations)
    stage = mark("registry_and_input_validation", stage)

    run_id, run_position, boundary_reason = build_continuity_run_id(bars)
    features = _compute_features(bars, run_id)
    stage = mark("continuity_and_float64_feature_construction", stage)

    feature_frame = pd.DataFrame(
        {name: features[name][positions].astype(np.float32) for name in MLAT_FEATURE_NAMES}
    )
    metadata = _metadata_frame(observations).reset_index(drop=True)
    matrix = pd.concat([metadata, feature_frame], axis=1)
    expected_columns = [*MLAT_METADATA_COLUMNS, *MLAT_FEATURE_NAMES]
    if matrix.columns.tolist() != expected_columns:
        raise RuntimeError(f"MLAT matrix column order mismatch: {matrix.columns.tolist()}")
    if any(str(matrix[name].dtype) != "float32" for name in MLAT_FEATURE_NAMES):
        raise TypeError("all persisted MLAT feature columns must be float32")
    stage = mark("decision_mapping_and_matrix_assembly", stage)

    diagnostics, persistence, construction = _diagnostic_tables(features, positions, run_position)
    if not persistence["passed"].all():
        failed = persistence.loc[~persistence["passed"], "feature_name"].tolist()
        raise OverflowError(f"float32 persistence validation failed: {failed}")
    run_start = np.flatnonzero(np.r_[True, run_id[1:] != run_id[:-1]])
    reason_codes_by_run = boundary_reason[run_start]
    boundary_labels = np.asarray(
        [BOUNDARY_REASON_LABELS[index] for index in range(1, 8)], dtype=object
    )
    selected_reason_codes = reason_codes_by_run[run_id[positions]]
    observation_audit = pd.DataFrame(
        {
            "observation_id": matrix["observation_id"].to_numpy(dtype=np.int64),
            "source_position": positions,
            "source_row_id": source_ids[positions],
            "continuity_run_id": run_id[positions],
            "bars_since_continuity_start": run_position[positions],
            "continuity_boundary_reason": boundary_labels[
                selected_reason_codes.astype(np.int64) - 1
            ],
        }
    )
    stage = mark("diagnostics_and_persistence_validation", stage)
    elapsed = perf_counter() - started
    runtime = pd.DataFrame(
        [
            {
                "source_rows": len(bars),
                "eligible_observations": len(observations),
                "continuity_runs": int(run_id.max(initial=-1) + 1),
                "feature_count": len(MLAT_FEATURE_NAMES),
                "matrix_rows": len(matrix),
                "matrix_columns": len(matrix.columns),
                "matrix_memory_bytes": int(matrix.memory_usage(index=True, deep=True).sum()),
                "float64_working_feature_bytes": int(
                    sum(values.nbytes for values in features.values())
                ),
                "total_seconds": float(elapsed),
                "bar_wise_python_loops": 0,
            }
        ]
    )
    return MlatFeatureBuildResult(
        matrix=matrix,
        registry=registry_frame,
        feature_diagnostics=diagnostics,
        persistence_validation=persistence,
        construction_audit=construction,
        observation_audit=observation_audit,
        build_timings=pd.DataFrame(timing_rows),
        runtime_summary=runtime,
    )
