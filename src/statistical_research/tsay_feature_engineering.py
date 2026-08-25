"""Outcome-free causal construction for the frozen Tsay v1 feature family.

The builder accepts only guarded GC OHLCV and eligible identity metadata.  It
reuses the canonical continuity and rolling primitive behavior, adds the
contracted New York-date reset to T01-T09, and maps to decision rows only after
full chronological construction.  Label frames are neither accepted nor
imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .feature_engineering import (
    BOUNDARY_REASON_LABELS,
    _group_ids,
    _lag,
    _rolling_correlation,
    _rolling_mean,
    _rolling_sum,
    _safe_divide,
    build_continuity_run_id,
)
from .tsay_feature_registry import (
    TICK_SIZE,
    TSAY_FEATURE_COLUMNS,
    TSAY_FEATURE_SPECS,
    registry_records,
)

TSAY_FEATURE_SOURCE_COLUMNS = (
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

TSAY_METADATA_COLUMNS = (
    "observation_id",
    "decision_timestamp_utc",
    "decision_timestamp_ny",
    "entry_timestamp_utc",
    "entry_timestamp_ny",
    "trade_date_ny",
    "research_partition",
    "entry_session",
    "product",
    "symbol",
    "active_symbol",
    "instrument_id",
    "continuous_segment_id",
)

TSAY_OBSERVATION_REQUIRED_COLUMNS = TSAY_METADATA_COLUMNS + (
    "decision_bar_id",
    "entry_bar_id",
    "minutes_to_forced_exit",
)

TSAY_CONTEXT_COLUMNS = (
    "tsay_roll_spread_identified_120",
    "tsay_clock_z_volume",
    "tsay_clock_z_range",
)
TSAY_MATRIX_COLUMNS = TSAY_METADATA_COLUMNS + TSAY_FEATURE_COLUMNS + TSAY_CONTEXT_COLUMNS
TSAY_PRIMITIVE_COLUMNS = ("atr_20", "current_range_over_atr", "realized_volatility_15")

FORBIDDEN_CAUSAL_TOKENS = (
    "entry_price",
    "exit_price",
    "forward",
    "future",
    "label",
    "mfe",
    "mae",
    "outcome",
    "target",
    "poi",
)

AR_RESPONSE_ROWS = 120
AR_LAGS = 5
AR_REQUIRED_CLOSES = 126
AR_CONDITION_LIMIT = 1.0e12
AR_EIGENVALUE_LIMIT = 0.999
CLOCK_BIN_MINUTES = 15
CLOCK_MINIMUM_PRIOR_DATES = 30
CLOCK_MINIMUM_PRIOR_OBSERVATIONS = 300
DEFAULT_BATCH_SIZE = 4096


@dataclass(frozen=True)
class TsayFeatureBuildResult:
    matrix: pd.DataFrame
    registry: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    coverage: pd.DataFrame
    construction_audit: pd.DataFrame
    observation_audit: pd.DataFrame
    missing_reasons: pd.DataFrame
    clock_reference: pd.DataFrame
    primitive_equality: pd.DataFrame
    build_timings: pd.DataFrame
    runtime_summary: pd.DataFrame


def _as_float(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    return np.asarray(values, dtype=np.float64)


def _run_positions(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups, dtype=np.int64)
    if len(groups) == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.r_[True, groups[1:] != groups[:-1]]
    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(len(groups), dtype=np.int64), 0)
    )
    return np.arange(len(groups), dtype=np.int64) - start_positions


def _validate_and_map_inputs(
    full_gc_bars: pd.DataFrame, eligible_observations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    supplied_source = set(full_gc_bars.columns)
    expected_source = set(TSAY_FEATURE_SOURCE_COLUMNS)
    missing_source = sorted(expected_source - supplied_source)
    extra_source = sorted(supplied_source - expected_source)
    missing_observations = sorted(
        set(TSAY_OBSERVATION_REQUIRED_COLUMNS) - set(eligible_observations.columns)
    )
    forbidden_source = sorted(
        column
        for column in full_gc_bars.columns
        if any(token in column.lower() for token in FORBIDDEN_CAUSAL_TOKENS)
    )
    forbidden_observations = sorted(
        column
        for column in eligible_observations.columns
        if any(token in column.lower() for token in FORBIDDEN_CAUSAL_TOKENS)
    )
    if (
        missing_source
        or extra_source
        or missing_observations
        or forbidden_source
        or forbidden_observations
    ):
        raise ValueError(
            "Invalid Tsay engineering namespace: "
            f"missing_source={missing_source}, extra_source={extra_source}, "
            f"missing_observations={missing_observations}, "
            f"forbidden_source={forbidden_source}, "
            f"forbidden_observations={forbidden_observations}"
        )
    if full_gc_bars.empty or eligible_observations.empty:
        raise ValueError("Tsay construction requires nonempty bars and eligible observations.")

    bars = full_gc_bars.loc[:, TSAY_FEATURE_SOURCE_COLUMNS].copy()
    bars = bars.sort_values(["ts_event_utc", "source_row_id"], kind="mergesort").reset_index(
        drop=True
    )
    observations = eligible_observations.loc[:, TSAY_OBSERVATION_REQUIRED_COLUMNS].copy()
    observations = observations.sort_values(
        ["decision_bar_id", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)

    bar_products = sorted(bars["product"].dropna().astype(str).unique().tolist())
    observation_products = sorted(observations["product"].dropna().astype(str).unique().tolist())
    if bar_products != ["GC"] or observation_products != ["GC"]:
        raise ValueError(
            f"Tsay construction is GC-only: bars={bar_products}, observations={observation_products}"
        )
    source_ids = pd.to_numeric(bars["source_row_id"], errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(source_ids)) != len(source_ids) or not np.all(np.diff(source_ids) > 0):
        raise ValueError("source_row_id must be unique and strictly increasing.")
    source_timestamp_ns = (
        pd.to_datetime(bars["ts_event_utc"], utc=True, errors="raise")
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    if not np.all(np.diff(source_timestamp_ns) > 0):
        raise ValueError("GC bars must be strictly chronological in UTC.")
    observation_ids = pd.to_numeric(observations["observation_id"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if len(np.unique(observation_ids)) != len(observation_ids):
        raise ValueError("observation_id must be unique.")
    decision_ids = pd.to_numeric(observations["decision_bar_id"], errors="raise").to_numpy(
        dtype=np.int64
    )
    positions = np.searchsorted(source_ids, decision_ids)
    in_bounds = positions < len(source_ids)
    exact = np.zeros(len(positions), dtype=bool)
    exact[in_bounds] = source_ids[positions[in_bounds]] == decision_ids[in_bounds]
    if not exact.all():
        raise ValueError(
            "decision_bar_id is absent from guarded GC history: "
            f"{decision_ids[~exact][:10].tolist()}"
        )
    decision_timestamp_ns = (
        pd.to_datetime(observations["decision_timestamp_utc"], utc=True, errors="raise")
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    if not np.array_equal(source_timestamp_ns[positions], decision_timestamp_ns):
        raise ValueError("decision_bar_id and decision_timestamp_utc do not map identically.")
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
            raise ValueError(f"Eligible identity mismatch for {column}.")
    if pd.to_datetime(bars["trade_date_ny"]).max().date() > pd.Timestamp("2024-12-31").date():
        raise ValueError("Post-2024 bars entered Tsay construction.")
    return bars, observations, source_ids, positions


def _clock_adjusted_series(
    raw: np.ndarray,
    dates: pd.DatetimeIndex,
    clock_bins: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply expanding-prior-date Development and frozen-Development Validation moments."""

    values = np.asarray(raw, dtype=np.float64)
    normalized_dates = pd.DatetimeIndex(dates).normalize()
    development = normalized_dates.year <= 2023
    validation = normalized_dates.year == 2024
    base = pd.DataFrame(
        {
            "row": np.arange(len(values), dtype=np.int64),
            "trade_date_ny": normalized_dates.tz_localize(None),
            "clock_bin": np.asarray(clock_bins, dtype=np.int16),
            "value": values,
        }
    )
    finite_dev = base.loc[development & np.isfinite(values)].copy()
    finite_dev["value_squared"] = finite_dev["value"] ** 2
    daily = (
        finite_dev.groupby(["trade_date_ny", "clock_bin"], observed=True, sort=True)
        .agg(count=("value", "size"), total=("value", "sum"), total_square=("value_squared", "sum"))
        .reset_index()
        .sort_values(["clock_bin", "trade_date_ny"], kind="mergesort")
    )
    grouped = daily.groupby("clock_bin", observed=True, sort=False)
    daily["prior_observations"] = grouped["count"].cumsum() - daily["count"]
    daily["prior_sum"] = grouped["total"].cumsum() - daily["total"]
    daily["prior_sumsq"] = grouped["total_square"].cumsum() - daily["total_square"]
    daily["prior_dates"] = grouped.cumcount()
    count = daily["prior_observations"].to_numpy(dtype=np.int64)
    total = daily["prior_sum"].to_numpy(dtype=np.float64)
    total_square = daily["prior_sumsq"].to_numpy(dtype=np.float64)
    mean = np.full(len(daily), np.nan, dtype=np.float64)
    std = np.full(len(daily), np.nan, dtype=np.float64)
    enough = count > 1
    mean[enough] = total[enough] / count[enough]
    numerator = total_square - _safe_divide(total**2, count)
    variance = _safe_divide(numerator, count - 1)
    variance[np.isfinite(variance) & (variance < 0.0) & (variance > -1.0e-12)] = 0.0
    std[enough] = np.sqrt(np.where(variance[enough] >= 0.0, variance[enough], np.nan))
    daily["reference_mean"] = mean
    daily["reference_std"] = std
    daily["reference_available"] = (
        (daily["prior_dates"] >= CLOCK_MINIMUM_PRIOR_DATES)
        & (daily["prior_observations"] >= CLOCK_MINIMUM_PRIOR_OBSERVATIONS)
        & np.isfinite(daily["reference_std"])
        & (daily["reference_std"] > 0.0)
    )

    lookup = daily.loc[
        :,
        [
            "trade_date_ny",
            "clock_bin",
            "prior_dates",
            "prior_observations",
            "reference_mean",
            "reference_std",
            "reference_available",
        ],
    ]
    mapped = base.loc[development, ["row", "trade_date_ny", "clock_bin", "value"]].merge(
        lookup,
        on=["trade_date_ny", "clock_bin"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    output = np.full(len(values), np.nan, dtype=np.float64)
    usable = mapped["reference_available"].fillna(False).to_numpy(dtype=bool) & np.isfinite(
        mapped["value"].to_numpy(dtype=np.float64)
    )
    mapped_rows = mapped["row"].to_numpy(dtype=np.int64)
    mapped_values = mapped["value"].to_numpy(dtype=np.float64)
    mapped_mean = mapped["reference_mean"].to_numpy(dtype=np.float64)
    mapped_std = mapped["reference_std"].to_numpy(dtype=np.float64)
    output[mapped_rows[usable]] = (mapped_values[usable] - mapped_mean[usable]) / mapped_std[usable]

    final = (
        finite_dev.groupby("clock_bin", observed=True, sort=True)
        .agg(
            development_observations=("value", "size"),
            development_dates=("trade_date_ny", "nunique"),
            total=("value", "sum"),
            total_square=("value_squared", "sum"),
        )
        .reset_index()
    )
    final_count = final["development_observations"].to_numpy(dtype=np.int64)
    final_total = final["total"].to_numpy(dtype=np.float64)
    final_sumsq = final["total_square"].to_numpy(dtype=np.float64)
    final["reference_mean"] = final_total / final_count
    final_variance = _safe_divide(
        final_sumsq - _safe_divide(final_total**2, final_count), final_count - 1
    )
    final_variance[
        np.isfinite(final_variance) & (final_variance < 0.0) & (final_variance > -1.0e-12)
    ] = 0.0
    final["reference_std"] = np.sqrt(np.where(final_variance >= 0.0, final_variance, np.nan))
    final["reference_available"] = (
        (final["development_dates"] >= CLOCK_MINIMUM_PRIOR_DATES)
        & (final["development_observations"] >= CLOCK_MINIMUM_PRIOR_OBSERVATIONS)
        & np.isfinite(final["reference_std"])
        & (final["reference_std"] > 0.0)
    )
    validation_rows = base.loc[validation, ["row", "clock_bin", "value"]].merge(
        final.loc[
            :,
            ["clock_bin", "reference_mean", "reference_std", "reference_available"],
        ],
        on="clock_bin",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    validation_usable = validation_rows["reference_available"].fillna(False).to_numpy(
        dtype=bool
    ) & np.isfinite(validation_rows["value"].to_numpy(dtype=np.float64))
    validation_index = validation_rows["row"].to_numpy(dtype=np.int64)
    validation_value = validation_rows["value"].to_numpy(dtype=np.float64)
    validation_mean = validation_rows["reference_mean"].to_numpy(dtype=np.float64)
    validation_std = validation_rows["reference_std"].to_numpy(dtype=np.float64)
    output[validation_index[validation_usable]] = (
        validation_value[validation_usable] - validation_mean[validation_usable]
    ) / validation_std[validation_usable]

    prior_reference = lookup.copy()
    prior_reference.insert(0, "series", name)
    prior_reference.insert(1, "reference_scope", "DEVELOPMENT_STRICTLY_PRIOR_DATE")
    final_reference = final.loc[
        :,
        [
            "clock_bin",
            "development_dates",
            "development_observations",
            "reference_mean",
            "reference_std",
            "reference_available",
        ],
    ].copy()
    final_reference.insert(0, "series", name)
    final_reference.insert(1, "reference_scope", "VALIDATION_ALL_DEVELOPMENT_FROZEN")
    final_reference.insert(2, "trade_date_ny", pd.NaT)
    final_reference = final_reference.rename(
        columns={
            "development_dates": "prior_dates",
            "development_observations": "prior_observations",
        }
    )
    reference = pd.concat([prior_reference, final_reference], ignore_index=True, sort=False)
    return output, reference


def _design_rank_and_condition(
    singular_values: np.ndarray, rows: int
) -> tuple[np.ndarray, np.ndarray]:
    tolerance = singular_values[:, :1] * np.finfo(np.float64).eps * max(rows, 6)
    rank = np.sum(singular_values > tolerance, axis=1)
    condition = _safe_divide(singular_values[:, 0], singular_values[:, -1])
    return rank, condition


def reference_ar_arch_from_closes(close_window: np.ndarray) -> dict[str, float | str]:
    """Simple scalar oracle for one exact 126-close T01/T02 window."""

    close = np.asarray(close_window, dtype=np.float64)
    result: dict[str, float | str] = {
        "t01": np.nan,
        "t02": np.nan,
        "lm": np.nan,
        "lm_pvalue": np.nan,
        "t01_reason": "INSUFFICIENT_HISTORY",
        "t02_reason": "INSUFFICIENT_HISTORY",
    }
    if close.shape != (AR_REQUIRED_CLOSES,):
        return result
    if not np.isfinite(close).all() or np.any(close <= 0.0):
        result["t01_reason"] = "NONFINITE_OR_NONPOSITIVE_INPUT"
        result["t02_reason"] = "NONFINITE_OR_NONPOSITIVE_INPUT"
        return result
    returns = 10_000.0 * np.log(close[1:] / close[:-1])
    y = returns[5:]
    design = np.column_stack(
        [np.ones(AR_RESPONSE_ROWS)]
        + [returns[5 - lag : 125 - lag] for lag in range(1, AR_LAGS + 1)]
    )
    coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    condition = singular[0] / singular[-1] if singular[-1] > 0.0 else np.inf
    if rank != 6:
        result["t01_reason"] = "AR_RANK_FAILURE"
        result["t02_reason"] = "AR_RANK_FAILURE"
        return result
    if not np.isfinite(condition) or condition > AR_CONDITION_LIMIT:
        result["t01_reason"] = "AR_CONDITION_FAILURE"
        result["t02_reason"] = "AR_CONDITION_FAILURE"
        return result

    residuals = y - design @ coefficients
    squared = residuals**2
    arch_y = squared[5:]
    arch_design = np.column_stack(
        [np.ones(115)] + [squared[5 - lag : 120 - lag] for lag in range(1, AR_LAGS + 1)]
    )
    if np.ptp(arch_y) == 0.0:
        result["t02_reason"] = "CONSTANT_RESIDUAL_SQUARE"
    else:
        arch_coef, _, arch_rank, _ = np.linalg.lstsq(arch_design, arch_y, rcond=None)
        if arch_rank != 6:
            result["t02_reason"] = "ARCH_RANK_FAILURE"
        else:
            arch_residual = arch_y - arch_design @ arch_coef
            sst = np.sum((arch_y - arch_y.mean()) ** 2)
            r_squared = 1.0 - np.sum(arch_residual**2) / sst
            if -1.0e-10 <= r_squared <= 1.0 + 1.0e-10:
                r_squared = float(np.clip(r_squared, 0.0, 1.0))
                result["t02"] = r_squared
                result["lm"] = 115.0 * r_squared
                result["lm_pvalue"] = float(chi2.sf(result["lm"], 5))
                result["t02_reason"] = "AVAILABLE"
            else:
                result["t02_reason"] = "ARCH_NUMERICAL_RANGE_FAILURE"

    companion = np.zeros((5, 5), dtype=np.float64)
    companion[0] = coefficients[1:]
    companion[1:, :-1] = np.eye(4)
    if np.max(np.abs(np.linalg.eigvals(companion))) >= AR_EIGENVALUE_LIMIT:
        result["t01_reason"] = "UNSTABLE_AR"
        return result
    state = returns[-1:-6:-1].copy()
    forecast_sum = 0.0
    for _ in range(60):
        forecast = float(coefficients[0] + coefficients[1:] @ state)
        forecast_sum += forecast
        state[1:] = state[:-1].copy()
        state[0] = forecast
    result["t01"] = forecast_sum
    result["t01_reason"] = "AVAILABLE"
    return result


def _batched_ar_arch(
    returns_bps: np.ndarray,
    candidate_position: np.ndarray,
    selected_positions: np.ndarray,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    count = len(selected_positions)
    outputs = {
        "t01": np.full(count, np.nan, dtype=np.float64),
        "t02": np.full(count, np.nan, dtype=np.float64),
        "lm": np.full(count, np.nan, dtype=np.float64),
        "lm_pvalue": np.full(count, np.nan, dtype=np.float64),
        "ar_condition": np.full(count, np.nan, dtype=np.float64),
        "ar_max_eigenvalue": np.full(count, np.nan, dtype=np.float64),
    }
    reasons = {
        "t01": np.full(count, "INSUFFICIENT_HISTORY", dtype=object),
        "t02": np.full(count, "INSUFFICIENT_HISTORY", dtype=object),
    }
    response_offsets = np.arange(119, -1, -1, dtype=np.int64)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        ends = selected_positions[start:stop]
        history = candidate_position[ends] >= 125
        if not history.any():
            continue
        local = np.flatnonzero(history)
        global_rows = start + local
        valid_ends = ends[local]
        response_index = valid_ends[:, None] - response_offsets[None, :]
        y = returns_bps[response_index]
        design = np.ones((len(valid_ends), AR_RESPONSE_ROWS, 6), dtype=np.float64)
        for lag in range(1, AR_LAGS + 1):
            design[:, :, lag] = returns_bps[response_index - lag]
        complete = np.isfinite(y).all(axis=1) & np.isfinite(design).all(axis=(1, 2))
        reasons["t01"][global_rows] = "NONFINITE_OR_NONPOSITIVE_INPUT"
        reasons["t02"][global_rows] = "NONFINITE_OR_NONPOSITIVE_INPUT"
        if not complete.any():
            continue
        complete_local = np.flatnonzero(complete)
        target_rows = global_rows[complete_local]
        x = design[complete_local]
        target = y[complete_local]
        u, singular, vh = np.linalg.svd(x, full_matrices=False)
        rank, condition = _design_rank_and_condition(singular, AR_RESPONSE_ROWS)
        outputs["ar_condition"][target_rows] = condition
        full_rank = rank == 6
        reasons["t01"][target_rows] = "AR_RANK_FAILURE"
        reasons["t02"][target_rows] = "AR_RANK_FAILURE"
        conditioned = full_rank & np.isfinite(condition) & (condition <= AR_CONDITION_LIMIT)
        reasons["t01"][target_rows[full_rank]] = "AR_CONDITION_FAILURE"
        reasons["t02"][target_rows[full_rank]] = "AR_CONDITION_FAILURE"
        if not conditioned.any():
            continue
        solve_local = np.flatnonzero(conditioned)
        solved_rows = target_rows[solve_local]
        solved_u = u[solve_local]
        solved_s = singular[solve_local]
        solved_vh = vh[solve_local]
        solved_y = target[solve_local]
        projection = np.einsum("bri,br->bi", solved_u, solved_y) / solved_s
        coefficients = np.einsum("bij,bj->bi", solved_vh.transpose(0, 2, 1), projection)
        solved_x = x[solve_local]
        residuals = solved_y - np.einsum("bri,bi->br", solved_x, coefficients)

        squared = residuals**2
        arch_y = squared[:, 5:]
        arch_x = np.ones((len(squared), 115, 6), dtype=np.float64)
        for lag in range(1, AR_LAGS + 1):
            arch_x[:, :, lag] = squared[:, 5 - lag : 120 - lag]
        reasons["t02"][solved_rows] = "CONSTANT_RESIDUAL_SQUARE"
        nonconstant = np.ptp(arch_y, axis=1) > 0.0
        if nonconstant.any():
            arch_local = np.flatnonzero(nonconstant)
            au, asing, avh = np.linalg.svd(arch_x[arch_local], full_matrices=False)
            arch_rank, _ = _design_rank_and_condition(asing, 115)
            arch_rows = solved_rows[arch_local]
            reasons["t02"][arch_rows] = "ARCH_RANK_FAILURE"
            arch_full = arch_rank == 6
            if arch_full.any():
                fit_local = np.flatnonzero(arch_full)
                fitted_rows = arch_rows[fit_local]
                arch_target = arch_y[arch_local][fit_local]
                arch_projection = (
                    np.einsum("bri,br->bi", au[fit_local], arch_target) / asing[fit_local]
                )
                arch_coef = np.einsum(
                    "bij,bj->bi",
                    avh[fit_local].transpose(0, 2, 1),
                    arch_projection,
                )
                arch_residual = arch_target - np.einsum(
                    "bri,bi->br", arch_x[arch_local][fit_local], arch_coef
                )
                sst = np.sum(
                    (arch_target - arch_target.mean(axis=1, keepdims=True)) ** 2,
                    axis=1,
                )
                r_squared = 1.0 - np.sum(arch_residual**2, axis=1) / sst
                in_range = (
                    np.isfinite(r_squared) & (r_squared >= -1.0e-10) & (r_squared <= 1.0 + 1.0e-10)
                )
                reasons["t02"][fitted_rows] = "ARCH_NUMERICAL_RANGE_FAILURE"
                accepted_rows = fitted_rows[in_range]
                accepted_r2 = np.clip(r_squared[in_range], 0.0, 1.0)
                outputs["t02"][accepted_rows] = accepted_r2
                outputs["lm"][accepted_rows] = 115.0 * accepted_r2
                outputs["lm_pvalue"][accepted_rows] = chi2.sf(outputs["lm"][accepted_rows], 5)
                reasons["t02"][accepted_rows] = "AVAILABLE"

        companion = np.zeros((len(coefficients), 5, 5), dtype=np.float64)
        companion[:, 0, :] = coefficients[:, 1:]
        companion[:, 1:, :-1] = np.eye(4, dtype=np.float64)
        eigenvalue = np.max(np.abs(np.linalg.eigvals(companion)), axis=1)
        outputs["ar_max_eigenvalue"][solved_rows] = eigenvalue
        stable = np.isfinite(eigenvalue) & (eigenvalue < AR_EIGENVALUE_LIMIT)
        reasons["t01"][solved_rows] = "UNSTABLE_AR"
        if stable.any():
            stable_local = np.flatnonzero(stable)
            stable_rows = solved_rows[stable_local]
            stable_coef = coefficients[stable_local]
            stable_ends = valid_ends[complete_local][solve_local][stable_local]
            state = np.column_stack([returns_bps[stable_ends - lag] for lag in range(AR_LAGS)])
            forecast_sum = np.zeros(len(stable_rows), dtype=np.float64)
            for _ in range(60):
                forecast = stable_coef[:, 0] + np.sum(stable_coef[:, 1:] * state, axis=1)
                forecast_sum += forecast
                state[:, 1:] = state[:, :-1].copy()
                state[:, 0] = forecast
            outputs["t01"][stable_rows] = forecast_sum
            reasons["t01"][stable_rows] = "AVAILABLE"
    return outputs, reasons


def _batched_tail_features(
    tick_change: np.ndarray,
    atr_ticks: np.ndarray,
    candidate_position: np.ndarray,
    selected_positions: np.ndarray,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    count = len(selected_positions)
    outputs = {
        "t05": np.full(count, np.nan, dtype=np.float64),
        "t06": np.full(count, np.nan, dtype=np.float64),
        "t09": np.full(count, np.nan, dtype=np.float64),
    }
    reasons = {
        key: np.full(count, "INSUFFICIENT_HISTORY", dtype=object) for key in ("t05", "t06", "t09")
    }
    offsets = np.arange(119, -1, -1, dtype=np.int64)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        ends = selected_positions[start:stop]
        history = candidate_position[ends] >= 120
        if not history.any():
            continue
        local = np.flatnonzero(history)
        rows = start + local
        valid_ends = ends[local]
        window = tick_change[valid_ends[:, None] - offsets[None, :]]
        complete = np.isfinite(window).all(axis=1)
        for key in reasons:
            reasons[key][rows] = "NONFINITE_INPUT"
        if not complete.any():
            continue
        complete_local = np.flatnonzero(complete)
        complete_rows = rows[complete_local]
        complete_ends = valid_ends[complete_local]
        values = window[complete_local]
        gains = np.partition(values, 108, axis=1)[:, 108:].mean(axis=1)
        losses = np.partition(-values, 108, axis=1)[:, 108:].mean(axis=1)
        denominator = np.abs(gains) + np.abs(losses)
        balance = np.zeros(len(values), dtype=np.float64)
        nonzero = denominator != 0.0
        balance[nonzero] = (gains[nonzero] - losses[nonzero]) / denominator[nonzero]
        outputs["t06"][complete_rows] = balance
        reasons["t06"][complete_rows] = "AVAILABLE"

        current_atr = atr_ticks[complete_ends]
        positive_atr = np.isfinite(current_atr) & (current_atr > 0.0)
        reasons["t05"][complete_rows] = "NONPOSITIVE_OR_MISSING_ATR"
        outputs["t05"][complete_rows[positive_atr]] = (
            losses[positive_atr] / current_atr[positive_atr]
        )
        reasons["t05"][complete_rows[positive_atr]] = "AVAILABLE"

        absolute = np.abs(values)
        threshold = np.partition(absolute, 107, axis=1)[:, 107]
        exceedance = absolute > threshold[:, None]
        exceedance_count = exceedance.sum(axis=1)
        clusters = exceedance[:, 0].astype(np.int64) + np.sum(
            exceedance[:, 1:] & ~exceedance[:, :-1], axis=1
        )
        available = exceedance_count > 0
        reasons["t09"][complete_rows] = "TIED_THRESHOLD_NO_EXCEEDANCE"
        outputs["t09"][complete_rows[available]] = clusters[available] / exceedance_count[available]
        reasons["t09"][complete_rows[available]] = "AVAILABLE"
    return outputs, reasons


def _canonical_primitives(
    bars: pd.DataFrame, run_id: np.ndarray, run_position: np.ndarray
) -> dict[str, np.ndarray]:
    high = _as_float(bars["high"])
    low = _as_float(bars["low"])
    close = _as_float(bars["close"])
    previous_close = _lag(close, 1, run_id)
    bar_range = high - low
    true_range = np.maximum.reduce(
        [bar_range, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    true_range[run_position == 0] = bar_range[run_position == 0]
    atr_20 = _rolling_mean(true_range, 20, run_id)
    log_return = np.full(len(close), np.nan, dtype=np.float64)
    valid = np.isfinite(previous_close) & (previous_close > 0.0) & (close > 0.0)
    log_return[valid] = np.log(close[valid] / previous_close[valid])
    realized_volatility_15 = 10_000.0 * np.sqrt(
        np.maximum(_rolling_sum(log_return**2, 15, run_id), 0.0)
    )
    return {
        "atr_20": atr_20,
        "current_range_over_atr": _safe_divide(bar_range, atr_20),
        "realized_volatility_15": realized_volatility_15,
    }


def _primitive_equality_table(
    primitives: Mapping[str, np.ndarray],
    positions: np.ndarray,
    canonical: pd.DataFrame | None,
    observation_ids: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if canonical is None:
        return pd.DataFrame(
            [
                {
                    "primitive": name,
                    "rows": len(positions),
                    "float32_exact": False,
                    "status": "NOT_CHECKED",
                }
                for name in TSAY_PRIMITIVE_COLUMNS
            ]
        )
    required = {"observation_id", *TSAY_PRIMITIVE_COLUMNS}
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise ValueError(f"Canonical primitive comparison is missing columns: {missing}")
    right = canonical.sort_values("observation_id", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(right["observation_id"].to_numpy(dtype=np.int64), observation_ids):
        raise ValueError("Canonical primitive rows do not align one-to-one by observation_id.")
    for name in TSAY_PRIMITIVE_COLUMNS:
        expected = pd.to_numeric(right[name], errors="coerce").to_numpy(dtype=np.float32)
        actual = primitives[name][positions].astype(np.float32)
        actual_null = np.isnan(actual)
        expected_null = np.isnan(expected)
        null_mask_exact = np.array_equal(actual_null, expected_null)
        exact = np.array_equal(actual, expected, equal_nan=True)
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        finite = np.isfinite(difference)
        mismatch_count = int(np.sum(~((actual == expected) | (actual_null & expected_null))))
        maximum_ulp_distance = 0
        policy = "EXACT_FLOAT32"
        passed = bool(exact and null_mask_exact)
        if name == "realized_volatility_15":
            policy = "EXACT_NULL_MASK_AND_MAXIMUM_ONE_FLOAT32_ULP"
            comparable = ~actual_null & ~expected_null
            if comparable.any():
                if not (
                    np.isfinite(actual[comparable]).all()
                    and np.isfinite(expected[comparable]).all()
                    and (actual[comparable] >= 0.0).all()
                    and (expected[comparable] >= 0.0).all()
                ):
                    raise ValueError(
                        "realized_volatility_15 compatibility inputs must be finite "
                        "and nonnegative outside the exact null mask."
                    )
                actual_bits = actual[comparable].view(np.uint32).astype(np.int64)
                expected_bits = expected[comparable].view(np.uint32).astype(np.int64)
                ulp_distance = np.abs(actual_bits - expected_bits)
                maximum_ulp_distance = int(ulp_distance.max(initial=0))
            passed = bool(null_mask_exact and maximum_ulp_distance <= 1)
        records.append(
            {
                "primitive": name,
                "rows": len(actual),
                "float32_exact": bool(exact),
                "null_mask_exact": bool(null_mask_exact),
                "mismatch_count": mismatch_count,
                "maximum_ulp_distance": maximum_ulp_distance,
                "maximum_absolute_difference": (
                    float(difference[finite].max()) if finite.any() else 0.0
                ),
                "compatibility_policy": policy,
                "status": "PASS" if passed else "FAIL",
            }
        )
        if not passed:
            raise ValueError(
                f"Recomputed canonical primitive violates its compatibility gate: {name}"
            )
    return pd.DataFrame(records)


def _reason_from_finite(values: np.ndarray, history: np.ndarray) -> np.ndarray:
    reason = np.full(len(values), "INSUFFICIENT_HISTORY", dtype=object)
    reason[history] = "NONFINITE_INPUT"
    reason[np.isfinite(values)] = "AVAILABLE"
    return reason


def build_tsay_feature_matrix(
    full_gc_bars: pd.DataFrame,
    eligible_observations: pd.DataFrame,
    *,
    canonical_primitives: pd.DataFrame | None = None,
    require_canonical_primitive_equality: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TsayFeatureBuildResult:
    """Build all T01-T09 features without accepting an outcome namespace."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    started = perf_counter()
    timing_rows: list[dict[str, object]] = []

    def mark(name: str, stage_started: float) -> float:
        now = perf_counter()
        timing_rows.append({"stage": name, "seconds": float(now - stage_started)})
        return now

    stage = started
    bars, observations, source_ids, positions = _validate_and_map_inputs(
        full_gc_bars, eligible_observations
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
        raise ValueError("Guarded OHLCV inputs must be finite.")
    if np.any(close <= 0.0) or np.any(volume < 0.0):
        raise ValueError("Close must be positive and volume must be nonnegative.")
    run_id, run_position, boundary_reason = build_continuity_run_id(bars)
    date_codes = pd.factorize(
        pd.to_datetime(bars["trade_date_ny"], errors="raise").dt.normalize(), sort=False
    )[0].astype(np.int64)
    candidate_group = _group_ids(run_id, date_codes)
    candidate_position = _run_positions(candidate_group)
    primitives = _canonical_primitives(bars, run_id, run_position)
    stage = mark("continuity_date_groups_and_canonical_primitives", stage)

    previous_close = _lag(close, 1, candidate_group)
    tick_change = (close - previous_close) / TICK_SIZE
    returns_bps = np.full(len(close), np.nan, dtype=np.float64)
    valid_return = np.isfinite(previous_close) & (previous_close > 0.0)
    returns_bps[valid_return] = 10_000.0 * np.log(
        close[valid_return] / previous_close[valid_return]
    )
    atr_ticks = primitives["atr_20"] / TICK_SIZE

    timestamps_ny = pd.DatetimeIndex(pd.to_datetime(bars["ts_event_ny"], errors="raise"))
    clock_bins = (
        CLOCK_BIN_MINUTES * ((timestamps_ny.hour * 60 + timestamps_ny.minute) // CLOCK_BIN_MINUTES)
    ).astype(np.int16)
    z_volume, volume_reference = _clock_adjusted_series(
        np.log1p(volume),
        pd.DatetimeIndex(pd.to_datetime(bars["trade_date_ny"], errors="raise")),
        clock_bins,
        name="tsay_clock_z_volume",
    )
    z_range, range_reference = _clock_adjusted_series(
        np.log1p(np.maximum(primitives["current_range_over_atr"], 0.0)),
        pd.DatetimeIndex(pd.to_datetime(bars["trade_date_ny"], errors="raise")),
        clock_bins,
        name="tsay_clock_z_range",
    )
    clock_reference = pd.concat([volume_reference, range_reference], ignore_index=True, sort=False)
    stage = mark("outcome_free_clock_profiles", stage)

    ar_outputs, ar_reasons = _batched_ar_arch(
        returns_bps,
        candidate_position,
        positions,
        batch_size=batch_size,
    )
    stage = mark("shared_batched_ar_arch", stage)

    mean_delta_120 = _rolling_mean(close - previous_close, 120, candidate_group)
    lag_delta = _lag(close - previous_close, 1, candidate_group)
    pair_product = (close - previous_close) * lag_delta
    pair_sum = _rolling_sum(pair_product, 119, candidate_group)
    current_sum = _rolling_sum(close - previous_close, 119, candidate_group)
    previous_sum = _rolling_sum(lag_delta, 119, candidate_group)
    gamma_numerator = (
        pair_sum - mean_delta_120 * (current_sum + previous_sum) + 119.0 * mean_delta_120**2
    )
    gamma = gamma_numerator / 119.0
    t03_full = np.full(len(close), np.nan, dtype=np.float64)
    t03_state_full = np.full(len(close), -1, dtype=np.int8)
    roll_history = candidate_position >= 120
    roll_valid = roll_history & np.isfinite(gamma)
    t03_state_full[roll_valid] = 0
    identified = roll_valid & (gamma < 0.0)
    t03_full[identified] = 2.0 * np.sqrt(-gamma[identified]) / TICK_SIZE
    t03_state_full[identified] = 1

    zero_indicator = np.where(np.isfinite(tick_change), tick_change == 0.0, np.nan)
    t04_full = _rolling_mean(zero_indicator, 60, candidate_group)
    tail_outputs, tail_reasons = _batched_tail_features(
        tick_change,
        atr_ticks,
        candidate_position,
        positions,
        batch_size=batch_size,
    )
    stage = mark("roll_zero_change_and_shared_tail_features", stage)

    volume_correlations = []
    range_correlations = []
    for lag in range(1, 6):
        volume_correlations.append(
            _rolling_correlation(
                returns_bps,
                _lag(z_volume, lag, candidate_group),
                120,
                candidate_group,
            )
        )
        range_correlations.append(
            _rolling_correlation(
                np.abs(returns_bps),
                _lag(z_range, lag, candidate_group),
                120,
                candidate_group,
            )
        )
    volume_stack = np.vstack(volume_correlations)
    range_stack = np.vstack(range_correlations)
    volume_complete = np.isfinite(volume_stack).all(axis=0) & np.isfinite(z_volume)
    range_complete = np.isfinite(range_stack).all(axis=0) & np.isfinite(z_range)
    t07_full = np.full(len(close), np.nan, dtype=np.float64)
    t08_full = np.full(len(close), np.nan, dtype=np.float64)
    t07_full[volume_complete] = (
        volume_stack[:, volume_complete].mean(axis=0) * z_volume[volume_complete]
    )
    t08_full[range_complete] = (
        np.sqrt(np.mean(range_stack[:, range_complete] ** 2, axis=0)) * z_range[range_complete]
    )
    stage = mark("exact_lagged_clock_adjusted_impulses", stage)

    mapped_feature_values: dict[str, np.ndarray] = {
        "tsay_ar5_cumulative_forecast_60_from_120_bps": ar_outputs["t01"],
        "tsay_arch_lm_r2_120_l5": ar_outputs["t02"],
        "tsay_roll_spread_proxy_120_ticks": t03_full[positions],
        "tsay_one_minute_close_zero_change_fraction_60": t04_full[positions],
        "tsay_loss_es_120_atr": tail_outputs["t05"],
        "tsay_es_tail_balance_120": tail_outputs["t06"],
        "tsay_volume_lead_return_impulse_120_l5": t07_full[positions],
        "tsay_range_state_lead_absreturn_impulse_120_l5": t08_full[positions],
        "tsay_extreme_cluster_ratio_120_q90": tail_outputs["t09"],
    }
    feature_frame = pd.DataFrame(
        {name: mapped_feature_values[name].astype(np.float32) for name in TSAY_FEATURE_COLUMNS}
    )
    context_frame = pd.DataFrame(
        {
            "tsay_roll_spread_identified_120": t03_state_full[positions].astype(np.int8),
            "tsay_clock_z_volume": z_volume[positions].astype(np.float32),
            "tsay_clock_z_range": z_range[positions].astype(np.float32),
        }
    )
    metadata = observations.loc[:, TSAY_METADATA_COLUMNS].reset_index(drop=True)
    matrix = pd.concat([metadata, feature_frame, context_frame], axis=1)
    if tuple(matrix.columns) != TSAY_MATRIX_COLUMNS:
        raise RuntimeError(f"Tsay matrix schema order drifted: {matrix.columns.tolist()}")

    reason_columns: dict[str, pd.Categorical] = {
        "T01__availability_reason": pd.Categorical(ar_reasons["t01"]),
        "T02__availability_reason": pd.Categorical(ar_reasons["t02"]),
    }
    t03_reason = np.full(len(positions), "UNAVAILABLE", dtype=object)
    t03_reason[t03_state_full[positions] == 0] = "VALID_NOT_IDENTIFIED"
    t03_reason[t03_state_full[positions] == 1] = "IDENTIFIED"
    reason_columns["T03__availability_reason"] = pd.Categorical(t03_reason)
    reason_columns["T04__availability_reason"] = pd.Categorical(
        _reason_from_finite(t04_full[positions], candidate_position[positions] >= 60)
    )
    reason_columns["T05__availability_reason"] = pd.Categorical(tail_reasons["t05"])
    reason_columns["T06__availability_reason"] = pd.Categorical(tail_reasons["t06"])
    reason_columns["T07__availability_reason"] = pd.Categorical(
        _reason_from_finite(t07_full[positions], candidate_position[positions] >= 124)
    )
    reason_columns["T08__availability_reason"] = pd.Categorical(
        _reason_from_finite(t08_full[positions], candidate_position[positions] >= 124)
    )
    reason_columns["T09__availability_reason"] = pd.Categorical(tail_reasons["t09"])
    missing_reasons = pd.DataFrame(
        {
            "observation_id": matrix["observation_id"].to_numpy(dtype=np.int64),
            **reason_columns,
        }
    )

    primitive_equality = _primitive_equality_table(
        primitives,
        positions,
        canonical_primitives,
        matrix["observation_id"].to_numpy(dtype=np.int64),
    )
    stage = mark("mapping_schema_reasons_and_primitive_equality", stage)

    feature_diagnostics: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    specs = {spec.feature_name: spec for spec in TSAY_FEATURE_SPECS}
    for name in TSAY_FEATURE_COLUMNS:
        values = pd.to_numeric(matrix[name], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        spec = specs[name]
        below = (
            finite & (values < spec.validation_minimum - 1.0e-6)
            if spec.validation_minimum is not None
            else np.zeros(len(values), dtype=bool)
        )
        above = (
            finite & (values > spec.validation_maximum + 1.0e-6)
            if spec.validation_maximum is not None
            else np.zeros(len(values), dtype=bool)
        )
        if below.any() or above.any():
            raise ValueError(f"Feature range validation failed for {name}.")
        feature_diagnostics.append(
            {
                "feature_name": name,
                "dtype": str(matrix[name].dtype),
                "finite_rows": int(finite.sum()),
                "missing_rows": int((~finite).sum()),
                "minimum": float(values[finite].min()) if finite.any() else None,
                "maximum": float(values[finite].max()) if finite.any() else None,
                "range_pass": True,
                "infinite_count": int(np.isinf(values).sum()),
            }
        )
        grouped = matrix.assign(_finite=finite).groupby(
            ["research_partition", "entry_session"], observed=True, sort=True
        )
        for (partition, session), group in grouped:
            coverage_rows.append(
                {
                    "feature_name": name,
                    "research_partition": str(partition),
                    "entry_session": str(session),
                    "rows": len(group),
                    "finite_rows": int(group["_finite"].sum()),
                    "coverage_rate": float(group["_finite"].mean()),
                }
            )

    run_starts = np.flatnonzero(np.r_[True, run_id[1:] != run_id[:-1]])
    reason_by_run = boundary_reason[run_starts]
    reason_labels = np.asarray(
        [BOUNDARY_REASON_LABELS[index] for index in range(1, 8)], dtype=object
    )
    candidate_starts = np.flatnonzero(np.r_[True, candidate_group[1:] != candidate_group[:-1]])
    candidate_reason = np.full(len(candidate_starts), "new_york_date_reset", dtype=object)
    candidate_reason[np.isin(candidate_starts, run_starts)] = "canonical_continuity_reset"
    observation_audit = pd.DataFrame(
        {
            "observation_id": matrix["observation_id"].to_numpy(dtype=np.int64),
            "source_position": positions,
            "source_row_id": source_ids[positions],
            "continuity_run_id": run_id[positions],
            "bars_since_continuity_start": run_position[positions],
            "continuity_boundary_reason": pd.Categorical(
                reason_labels[reason_by_run[run_id[positions]].astype(np.int64) - 1]
            ),
            "candidate_date_run_id": candidate_group[positions],
            "bars_since_candidate_date_run_start": candidate_position[positions],
            "candidate_boundary_reason": pd.Categorical(
                candidate_reason[np.searchsorted(candidate_starts, positions, side="right") - 1]
            ),
            "ar_condition_number": ar_outputs["ar_condition"],
            "ar_max_companion_eigenvalue": ar_outputs["ar_max_eigenvalue"],
            "arch_lm_statistic": ar_outputs["lm"],
            "arch_lm_chi2_pvalue": ar_outputs["lm_pvalue"],
        }
    )
    construction = pd.DataFrame(
        [
            {
                "check": "source_namespace_outcome_free",
                "passed": True,
                "detail": "Exact allowlist; no forward/future/label/outcome tokens.",
            },
            {
                "check": "gc_only_through_2024",
                "passed": True,
                "detail": "All bars and observations are GC and end no later than 2024-12-31.",
            },
            {
                "check": "canonical_primitives_amended_compatibility",
                "passed": bool(
                    not require_canonical_primitive_equality
                    or primitive_equality["status"].eq("PASS").all()
                ),
                "detail": (
                    "atr_20 and current_range_over_atr exact float32; "
                    "realized_volatility_15 exact null mask and at most one float32 ULP"
                    if canonical_primitives is not None
                    else "Not required for this construction call."
                ),
            },
            {
                "check": "new_candidate_windows_reset_at_ny_date_and_continuity",
                "passed": True,
                "detail": "candidate_date_run_id combines canonical run and New York date.",
            },
            {
                "check": "clock_profiles_strict_prior_development_and_frozen_validation",
                "passed": True,
                "detail": "30 prior dates, 300 observations, positive sample standard deviation.",
            },
            {
                "check": "t01_t02_shared_batched_svd",
                "passed": True,
                "detail": f"batch_size={batch_size}; float64; one shared AR fit per row.",
            },
            {
                "check": "persisted_feature_dtype",
                "passed": all(
                    str(matrix[name].dtype) == "float32" for name in TSAY_FEATURE_COLUMNS
                ),
                "detail": "Float64 calculation followed by validated float32 cast.",
            },
        ]
    )
    if not construction["passed"].all():
        raise RuntimeError("Tsay construction audit failed.")
    stage = mark("diagnostics_and_audits", stage)

    elapsed = perf_counter() - started
    runtime = pd.DataFrame(
        [
            {
                "source_rows": len(bars),
                "eligible_rows": len(observations),
                "continuity_runs": int(run_id.max(initial=-1) + 1),
                "candidate_date_runs": int(candidate_group.max(initial=-1) + 1),
                "feature_count": len(TSAY_FEATURE_COLUMNS),
                "context_column_count": len(TSAY_CONTEXT_COLUMNS),
                "batch_size": batch_size,
                "matrix_memory_bytes": int(matrix.memory_usage(index=True, deep=True).sum()),
                "peak_batch_bytes_estimate": int(
                    batch_size * (120 * 6 * 8 + 115 * 6 * 8 + 120 * 8 * 4)
                ),
                "total_seconds": float(elapsed),
                "compute_dtype": "float64",
                "persistence_dtype": "float32",
            }
        ]
    )
    return TsayFeatureBuildResult(
        matrix=matrix,
        registry=pd.DataFrame(registry_records()),
        feature_diagnostics=pd.DataFrame(feature_diagnostics),
        coverage=pd.DataFrame(coverage_rows),
        construction_audit=construction,
        observation_audit=observation_audit,
        missing_reasons=missing_reasons,
        clock_reference=clock_reference,
        primitive_equality=primitive_equality,
        build_timings=pd.DataFrame(timing_rows),
        runtime_summary=runtime,
    )
