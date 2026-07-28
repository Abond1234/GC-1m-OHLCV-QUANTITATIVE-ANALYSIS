"""Outcome-free scalar path features for Project 1 FES Notebook Section 2.

The builder consumes a narrow GC bar frame, key/metadata-only eligible
observations, and the existing outcome-free feature matrix. It does not accept
labels. All rolling calculations use the production continuity identity plus
the study-specific New York trading-date reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Final, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .feature_engineering import BOUNDARY_REASON_LABELS, build_continuity_run_id

CALCULATION_DTYPE: Final = np.float64
SAVED_DTYPE: Final = np.float32
MAD_SCALE: Final = 1.4826
SPEARMAN_MIN_DISTINCT_FRACTION: Final = 0.20

SOURCE_COLUMNS: Final = (
    "source_row_id",
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
)

OBSERVATION_COLUMNS: Final = (
    "observation_id",
    "decision_bar_id",
    "decision_timestamp_utc",
    "trade_date_ny",
    "entry_session",
    "research_partition",
)

BASE_FEATURE_NAMES: Final = (
    "ret_trimmed_mean_30_bps",
    "ret_mad_30_bps",
    "ret_tail_balance_60",
    "ret_outlier_fraction_60",
    "price_path_curvature_30_atr",
    "ordered_draw_balance_30",
    "return_acf_energy_60",
    "lagged_volume_return_spearman_30",
    "range_volume_spearman_30",
    "volume_profile_slope_30",
)

HELPER_FEATURE_NAMES: Final = ("relative_volume_20_clipped",)

INTERACTION_FEATURE_NAMES: Final = (
    "curvature_coherence_30",
    "tail_pressure_activity_60",
    "lagged_volume_confirmation_30",
    "vwap_trend_alignment_30",
)

SECTION2_FEATURE_NAMES: Final = (
    *BASE_FEATURE_NAMES,
    *HELPER_FEATURE_NAMES,
    *INTERACTION_FEATURE_NAMES,
)

CAUSE_COLUMN_SUFFIX: Final = "__missing_reason"
DEGENERATE_INDICATOR_SUFFIX: Final = "__degenerate"

MISSING_CAUSES: Final = (
    "SOURCE_START",
    "TIMESTAMP_GAP",
    "PRODUCT_CHANGE",
    "SELECTED_CONTRACT_CHANGE",
    "INSTRUMENT_CHANGE",
    "CONTINUOUS_SEGMENT_CHANGE",
    "TRADABILITY_ROLL_OR_LIQUIDITY_BOUNDARY",
    "NY_DATE_RESET",
    "SOURCE_INPUT_NULL",
    "NONPOSITIVE_ATR",
    "DEGENERATE_STATISTIC",
    "COMPLETE",
)

BOUNDARY_CAUSES: Final = MISSING_CAUSES[:8]
INTEGRITY_FAILURE_CAUSES: Final = ("SOURCE_INPUT_NULL", "NONPOSITIVE_ATR")
DEGENERATE_PERMITTED_FEATURES: Final = (
    "return_acf_energy_60",
    "lagged_volume_return_spearman_30",
    "range_volume_spearman_30",
)

MISSINGNESS_ACTIONS: Final = (
    {
        "cause_group": "COMPLETE",
        "scalar_action": "retain",
        "model_indicator": "none",
    },
    {
        "cause_group": "SOURCE_START_OR_BOUNDARY_WARMUP",
        "scalar_action": "exclude_from_matched_comparison",
        "model_indicator": "none",
    },
    {
        "cause_group": "SOURCE_INPUT_NULL_OR_NONPOSITIVE_ATR",
        "scalar_action": "fail_integrity_audit",
        "model_indicator": "none",
    },
    {
        "cause_group": "DEGENERATE_STATISTIC_F07_TO_F09_ONLY",
        "scalar_action": "training_fold_median_imputation",
        "model_indicator": "fixed_boolean_feature__degenerate",
    },
    {
        "cause_group": "FORMULA_DEFINED_ZERO",
        "scalar_action": "retain_numeric_zero_and_count",
        "model_indicator": "none",
    },
)

PARENT_FEATURE_NAMES: Final = (
    "efficiency_ratio_30",
    "distance_from_research_day_vwap_atr",
    "normalized_ols_slope_30",
)

COMPARATOR_MAP: Final[Mapping[str, tuple[str, ...]]] = {
    "ret_trimmed_mean_30_bps": (
        "return_30m_atr",
        "normalized_ols_slope_30",
        "efficiency_ratio_30",
    ),
    "ret_mad_30_bps": ("atr_20", "realized_volatility_30"),
    "ret_tail_balance_60": ("directional_energy_balance_15_exp",),
    "ret_outlier_fraction_60": (
        "current_range_over_atr",
        "realized_volatility_ratio_5_30",
        "liquidity_vacuum_score_exp",
    ),
    "price_path_curvature_30_atr": (
        "normalized_ols_slope_30",
        "ols_r_squared_30",
        "momentum_acceleration_15_30",
    ),
    "ordered_draw_balance_30": (
        "rolling_range_position_60",
        "efficiency_ratio_30",
        "directional_persistence_15",
    ),
    "return_acf_energy_60": (
        "return_autocorrelation_15",
        "return_sign_change_rate_30",
        "choppiness_14",
    ),
    "lagged_volume_return_spearman_30": (
        "volume_price_alignment_10",
        "signed_volume_proxy",
        "relative_volume_20",
    ),
    "range_volume_spearman_30": (
        "volume_per_tick_range",
        "current_range_over_atr",
        "relative_volume_20",
    ),
    "volume_profile_slope_30": (
        "volume_acceleration_5_20",
        "relative_volume_20",
        "tod_log_volume_z",
    ),
}

COMPARATOR_NAMES: Final = tuple(
    dict.fromkeys(name for names in COMPARATOR_MAP.values() for name in names)
)

RANGE_LIMITS: Final[Mapping[str, tuple[float | None, float | None]]] = {
    "ret_trimmed_mean_30_bps": (None, None),
    "ret_mad_30_bps": (0.0, None),
    "ret_tail_balance_60": (-1.0, 1.0),
    "ret_outlier_fraction_60": (0.0, 1.0),
    "price_path_curvature_30_atr": (None, None),
    "ordered_draw_balance_30": (-1.0, 1.0),
    "return_acf_energy_60": (0.0, 1.0),
    "lagged_volume_return_spearman_30": (-1.0, 1.0),
    "range_volume_spearman_30": (-1.0, 1.0),
    "volume_profile_slope_30": (None, None),
    "relative_volume_20_clipped": (0.5, 2.0),
    "curvature_coherence_30": (None, None),
    "tail_pressure_activity_60": (-2.0, 2.0),
    "lagged_volume_confirmation_30": (-2.0, 2.0),
    "vwap_trend_alignment_30": (None, None),
}

RAW_HISTORY_BARS: Final[Mapping[str, int]] = {
    "ret_trimmed_mean_30_bps": 31,
    "ret_mad_30_bps": 31,
    "ret_tail_balance_60": 61,
    "ret_outlier_fraction_60": 61,
    "price_path_curvature_30_atr": 31,
    "ordered_draw_balance_30": 31,
    "return_acf_energy_60": 61,
    # The earliest lagged ZV uses its own complete 60-bar causal window.
    "lagged_volume_return_spearman_30": 90,
    # The earliest concurrent ZV uses its own complete 60-bar causal window.
    "range_volume_spearman_30": 89,
    "volume_profile_slope_30": 89,
    "relative_volume_20_clipped": 20,
}


@dataclass(frozen=True)
class ScalarFeatureBuildResult:
    """Section 2 outcome-free matrices and deterministic diagnostics."""

    matrix: pd.DataFrame
    missing_reasons: pd.DataFrame
    zero_scale_flags: pd.DataFrame
    registry: pd.DataFrame
    construction_audit: pd.DataFrame


def _registry_records() -> tuple[dict[str, object], ...]:
    reset = "production continuity_run_id + trade_date_ny outer reset"
    available = "close of completed decision bar t"
    records: list[dict[str, object]] = [
        {
            "feature_id": "F01",
            "feature_name": "ret_trimmed_mean_30_bps",
            "role": "directional",
            "formula": (
                "mean of r[t-29:t] after sorting and removing exactly three "
                "observations from each tail"
            ),
            "inputs": "close",
            "history": "30 returns / 31 closes",
            "minimum_raw_bars": 31,
            "expected_range": "unbounded",
            "book_provenance": "Ch. 6.1 robust transformation; Ch. 9.4 trimming/profile standardization",
        },
        {
            "feature_id": "F02",
            "feature_name": "ret_mad_30_bps",
            "role": "opportunity_risk",
            "formula": "1.4826 * median(abs(r - median(r))); all-identical -> 0",
            "inputs": "close",
            "history": "30 returns / 31 closes",
            "minimum_raw_bars": 31,
            "expected_range": "[0,+inf)",
            "book_provenance": "Ch. 3.2.1 robust error summaries; Ch. 9.4 robust profile scale",
        },
        {
            "feature_id": "F03",
            "feature_name": "ret_tail_balance_60",
            "role": "directional",
            "formula": "(abs(q90)-abs(q10))/(abs(q90)+abs(q10)); zero denominator -> 0",
            "inputs": "close",
            "history": "60 returns / 61 closes",
            "minimum_raw_bars": 61,
            "expected_range": "[-1,1]",
            "book_provenance": "Ch. 9 profile summarization; Ch. 3 robust/rank reasoning",
        },
        {
            "feature_id": "F04",
            "feature_name": "ret_outlier_fraction_60",
            "role": "opportunity_risk",
            "formula": "mean(abs(r-median(r)) > 3*(1.4826*MAD)); zero scale -> 0",
            "inputs": "close",
            "history": "60 returns / 61 closes",
            "minimum_raw_bars": 61,
            "expected_range": "[0,1]",
            "book_provenance": "Ch. 6.1 outlier-resistant smoothing; Ch. 9.4 trimming",
        },
        {
            "feature_id": "F05",
            "feature_name": "price_path_curvature_30_atr",
            "role": "directional_path_shape",
            "formula": "quadratic beta2 from ATR20-normalized 31-close path on x in [-1,1]",
            "inputs": "close|atr_20",
            "history": "31 closes and decision ATR20",
            "minimum_raw_bars": 31,
            "expected_range": "unbounded",
            "book_provenance": "Ch. 6.2.1 polynomial/basis representations; Ch. 9.3 profile curvature",
        },
        {
            "feature_id": "F06",
            "feature_name": "ordered_draw_balance_30",
            "role": "directional_path_shape",
            "formula": "(max ordered run-up - max ordered drawdown)/(sum); zero sum -> 0",
            "inputs": "close",
            "history": "31 closes",
            "minimum_raw_bars": 31,
            "expected_range": "[-1,1]",
            "book_provenance": "Ch. 9 ordered profile reduction",
        },
        {
            "feature_id": "F07",
            "feature_name": "return_acf_energy_60",
            "role": "opportunity_regime",
            "formula": "sqrt(mean(Pearson_ACF_lag_k**2, k=1..5))",
            "inputs": "close",
            "history": "60 returns / 61 closes",
            "minimum_raw_bars": 61,
            "expected_range": "[0,1]",
            "book_provenance": "Ch. 9.2 and Ch. 9.5 profile autocorrelation",
        },
        {
            "feature_id": "F08",
            "feature_name": "lagged_volume_return_spearman_30",
            "role": "directional_participation_lead",
            "formula": "Spearman_average_rank(ZV[i-1], r[i]), i=t-29..t",
            "inputs": "close|volume_zscore_60",
            "history": "30 pairs; 90 raw bars for earliest complete lagged ZV",
            "minimum_raw_bars": 90,
            "expected_range": "[-1,1]",
            "book_provenance": "Ch. 9.5 within-profile correlation",
        },
        {
            "feature_id": "F09",
            "feature_name": "range_volume_spearman_30",
            "role": "opportunity_liquidity_state",
            "formula": "Spearman_average_rank(TR[i]/ATR20[i], ZV[i]), i=t-29..t",
            "inputs": "high|low|close|atr_20|volume_zscore_60",
            "history": "30 pairs; 89 raw bars for earliest complete concurrent ZV",
            "minimum_raw_bars": 89,
            "expected_range": "[-1,1]",
            "book_provenance": "Ch. 9.5 within-profile correlation",
        },
        {
            "feature_id": "F10",
            "feature_name": "volume_profile_slope_30",
            "role": "opportunity_participation_regime",
            "formula": "OLS beta for ZV[t-29:t] on x in [-1,1]",
            "inputs": "volume_zscore_60",
            "history": "30 ZV values; 89 raw bars for earliest complete ZV",
            "minimum_raw_bars": 89,
            "expected_range": "unbounded",
            "book_provenance": "Ch. 9 profile reduction by slope",
        },
        {
            "feature_id": "H01",
            "feature_name": "relative_volume_20_clipped",
            "role": "hierarchy_support_not_standalone_alpha",
            "formula": "clip(relative_volume_20,0.5,2.0)",
            "inputs": "volume|relative_volume_20",
            "history": "20 volume bars",
            "minimum_raw_bars": 20,
            "expected_range": "[0.5,2.0]",
            "book_provenance": "frozen deterministic interaction support transform",
        },
        {
            "feature_id": "I01",
            "feature_name": "curvature_coherence_30",
            "role": "directional_interaction",
            "formula": "price_path_curvature_30_atr * efficiency_ratio_30",
            "inputs": "price_path_curvature_30_atr|efficiency_ratio_30",
            "history": "inherits both main effects",
            "minimum_raw_bars": 31,
            "expected_range": "unbounded",
            "book_provenance": "expert interaction slate; curvature/path-efficiency coherence",
        },
        {
            "feature_id": "I02",
            "feature_name": "tail_pressure_activity_60",
            "role": "directional_interaction",
            "formula": "ret_tail_balance_60 * relative_volume_20_clipped",
            "inputs": "ret_tail_balance_60|relative_volume_20_clipped",
            "history": "inherits both main effects",
            "minimum_raw_bars": 61,
            "expected_range": "[-2,2]",
            "book_provenance": "expert interaction slate; tail pressure with current activity",
        },
        {
            "feature_id": "I03",
            "feature_name": "lagged_volume_confirmation_30",
            "role": "directional_interaction",
            "formula": "lagged_volume_return_spearman_30 * relative_volume_20_clipped",
            "inputs": "lagged_volume_return_spearman_30|relative_volume_20_clipped",
            "history": "inherits both main effects",
            "minimum_raw_bars": 90,
            "expected_range": "[-2,2]",
            "book_provenance": "expert interaction slate; lagged volume confirmation",
        },
        {
            "feature_id": "I04",
            "feature_name": "vwap_trend_alignment_30",
            "role": "directional_interaction",
            "formula": "distance_from_research_day_vwap_atr * normalized_ols_slope_30",
            "inputs": "distance_from_research_day_vwap_atr|normalized_ols_slope_30",
            "history": "inherits both existing main effects",
            "minimum_raw_bars": 30,
            "expected_range": "unbounded",
            "book_provenance": "expert interaction slate; VWAP displacement/trend alignment",
        },
    ]
    for record in records:
        record["availability"] = available
        record["reset_boundary"] = reset
        record["calculation_dtype"] = "float64"
        record["saved_dtype"] = "float32"
        record["source_function"] = "fes_project1_features.build_scalar_feature_matrix"
        record["version"] = "1.0.0"
    return tuple(records)


def scalar_feature_registry() -> pd.DataFrame:
    """Return the frozen Section 2 registry in deterministic feature order."""

    registry = pd.DataFrame.from_records(_registry_records())
    if registry["feature_name"].tolist() != list(SECTION2_FEATURE_NAMES):
        raise AssertionError("Section 2 registry order differs from the frozen slate.")
    if registry["feature_name"].duplicated().any():
        raise AssertionError("Section 2 registry feature names must be unique.")
    return registry


def _as_float(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
    return np.asarray(values, dtype=np.float64)


def _lag(values: np.ndarray, periods: int, groups: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if periods == 0:
        return np.asarray(values, dtype=np.float64).copy()
    valid = groups[periods:] == groups[:-periods]
    targets = np.arange(periods, len(values), dtype=np.int64)[valid]
    result[targets] = np.asarray(values, dtype=np.float64)[:-periods][valid]
    return result


def _rolling_sum(values: np.ndarray, window: int, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    sums = np.zeros(len(values) + 1, dtype=np.float64)
    counts = np.zeros(len(values) + 1, dtype=np.int64)
    np.cumsum(np.where(finite, values, 0.0), out=sums[1:])
    np.cumsum(finite, dtype=np.int64, out=counts[1:])
    ends = np.arange(window - 1, len(values), dtype=np.int64)
    starts = ends - window + 1
    complete = (groups[ends] == groups[starts]) & (
        (counts[ends + 1] - counts[starts]) == window
    )
    selected_ends = ends[complete]
    selected_starts = starts[complete]
    result[selected_ends] = sums[selected_ends + 1] - sums[selected_starts]
    return result


def _rolling_mean(values: np.ndarray, window: int, groups: np.ndarray) -> np.ndarray:
    return _rolling_sum(values, window, groups) / float(window)


def _rolling_std_population(
    values: np.ndarray, window: int, groups: np.ndarray
) -> np.ndarray:
    mean = _rolling_mean(values, window, groups)
    mean_square = _rolling_mean(np.asarray(values, dtype=np.float64) ** 2, window, groups)
    variance = mean_square - mean**2
    tiny_negative = np.isfinite(variance) & (variance < 0.0) & (variance > -1.0e-12)
    variance[tiny_negative] = 0.0
    return np.sqrt(np.where(variance >= 0.0, variance, np.nan))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    np.divide(numerator, denominator, out=result, where=valid)
    result[~np.isfinite(result)] = np.nan
    return result


def _legal_continuity(
    bars: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply production continuity precedence and then the NY-date outer reset."""

    production_run, _, production_reason = build_continuity_run_id(bars)
    dates = pd.to_datetime(bars["trade_date_ny"]).to_numpy(dtype="datetime64[D]")
    date_reset = np.ones(len(bars), dtype=bool)
    date_reset[1:] = dates[1:] != dates[:-1]
    production_start = production_reason != 0
    legal_start = production_start | date_reset
    legal_start[0] = True

    cause_at_start = np.empty(len(bars), dtype=object)
    cause_at_start[:] = ""
    production_map = {
        int(code): str(label).upper() for code, label in BOUNDARY_REASON_LABELS.items()
    }
    for code, label in production_map.items():
        cause_at_start[production_reason == code] = label
    ny_only = legal_start & ~production_start
    cause_at_start[ny_only] = "NY_DATE_RESET"
    cause_at_start[0] = "SOURCE_START"

    legal_group = np.cumsum(legal_start, dtype=np.int64) - 1
    start_index = np.maximum.accumulate(
        np.where(legal_start, np.arange(len(bars), dtype=np.int64), 0)
    )
    legal_position = np.arange(len(bars), dtype=np.int64) - start_index
    latest_cause = cause_at_start[start_index]
    return legal_group, legal_position, latest_cause


def _window(values: np.ndarray, ends: np.ndarray, length: int) -> np.ndarray:
    offsets = np.arange(-length + 1, 1, dtype=np.int64)
    return np.asarray(values, dtype=np.float64)[ends[:, None] + offsets[None, :]]


def _trimmed_mean(values: np.ndarray, trim_each_tail: int = 3) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64), axis=1)
    return np.mean(sorted_values[:, trim_each_tail:-trim_each_tail], axis=1)


def _mad(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=1)
    return MAD_SCALE * np.median(np.abs(values - median[:, None]), axis=1)


def _tail_balance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.quantile(
        np.asarray(values, dtype=np.float64),
        (0.10, 0.90),
        axis=1,
        method="linear",
    )
    denominator = np.abs(quantiles[1]) + np.abs(quantiles[0])
    zero = denominator == 0.0
    result = np.zeros(len(denominator), dtype=np.float64)
    np.divide(
        np.abs(quantiles[1]) - np.abs(quantiles[0]),
        denominator,
        out=result,
        where=~zero,
    )
    return result, zero


def _outlier_fraction(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=1)
    scale = MAD_SCALE * np.median(np.abs(values - median[:, None]), axis=1)
    zero = scale == 0.0
    result = np.mean(
        np.abs(values - median[:, None]) > 3.0 * scale[:, None], axis=1
    )
    result[zero] = 0.0
    return result, zero


def _curvature(close_windows: np.ndarray, atr_t: np.ndarray) -> np.ndarray:
    close_windows = np.asarray(close_windows, dtype=np.float64)
    atr_t = np.asarray(atr_t, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, close_windows.shape[1])
    design = np.column_stack([np.ones_like(x), x, x**2])
    beta2_weights = np.linalg.pinv(design)[2]
    normalized = (close_windows - close_windows[:, [0]]) / atr_t[:, None]
    return normalized @ beta2_weights


def _ordered_draw_balance(close_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(close_windows, dtype=np.float64)
    prior_min = np.minimum.accumulate(values, axis=1)
    prior_max = np.maximum.accumulate(values, axis=1)
    run_up = np.max(values - prior_min, axis=1)
    draw_down = np.max(prior_max - values, axis=1)
    denominator = run_up + draw_down
    zero = denominator == 0.0
    result = np.zeros(len(values), dtype=np.float64)
    np.divide(run_up - draw_down, denominator, out=result, where=~zero)
    return result, zero


def _pearson_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    degenerate = ~np.isfinite(denominator) | (denominator <= 0.0)
    result = np.full(len(x), np.nan, dtype=np.float64)
    np.divide(
        np.sum(x_centered * y_centered, axis=1),
        denominator,
        out=result,
        where=~degenerate,
    )
    finite = np.isfinite(result)
    result[finite] = np.clip(result[finite], -1.0, 1.0)
    return result, degenerate


def _acf_energy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    correlations: list[np.ndarray] = []
    degeneracy = np.zeros(len(values), dtype=bool)
    for lag in range(1, 6):
        correlation, degenerate = _pearson_rows(values[:, :-lag], values[:, lag:])
        correlations.append(correlation)
        degeneracy |= degenerate
    stacked = np.column_stack(correlations)
    result = np.sqrt(np.mean(stacked**2, axis=1))
    result[degeneracy] = np.nan
    return result, degeneracy


def _distinct_count_rows(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64), axis=1)
    return 1 + np.sum(sorted_values[:, 1:] != sorted_values[:, :-1], axis=1)


def _spearman_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Spearman inputs must have identical shape.")
    minimum_distinct = max(3, ceil(SPEARMAN_MIN_DISTINCT_FRACTION * x.shape[1]))
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    distinct = (_distinct_count_rows(x) >= minimum_distinct) & (
        _distinct_count_rows(y) >= minimum_distinct
    )
    valid = finite & distinct
    result = np.full(len(x), np.nan, dtype=np.float64)
    if valid.any():
        x_rank = rankdata(x[valid], method="average", axis=1)
        y_rank = rankdata(y[valid], method="average", axis=1)
        correlation, rank_degenerate = _pearson_rows(x_rank, y_rank)
        valid_indices = np.flatnonzero(valid)
        result[valid_indices] = correlation
        valid[valid_indices[rank_degenerate]] = False
    return result, ~valid


def _profile_slope(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, values.shape[1])
    centered = x - np.mean(x)
    weights = centered / np.sum(centered**2)
    return values @ weights


def _set_history_causes(
    reason: np.ndarray,
    legal_position: np.ndarray,
    decision_positions: np.ndarray,
    latest_cause: np.ndarray,
    required_bars: int,
) -> np.ndarray:
    complete_history = legal_position[decision_positions] >= required_bars - 1
    reason[~complete_history] = latest_cause[decision_positions[~complete_history]]
    return complete_history


def _assign_failure(
    reason: np.ndarray,
    candidate_indices: np.ndarray,
    failure_mask: np.ndarray,
    cause: str,
) -> np.ndarray:
    failed = candidate_indices[failure_mask]
    reason[failed] = cause
    return candidate_indices[~failure_mask]


def _validate_inputs(
    bars: pd.DataFrame,
    observations: pd.DataFrame,
    existing_features: pd.DataFrame,
) -> None:
    missing_source = sorted(set(SOURCE_COLUMNS).difference(bars.columns))
    missing_observation = sorted(set(OBSERVATION_COLUMNS).difference(observations.columns))
    required_existing = {"observation_id", *PARENT_FEATURE_NAMES, *COMPARATOR_NAMES}
    missing_existing = sorted(required_existing.difference(existing_features.columns))
    if missing_source or missing_observation or missing_existing:
        raise ValueError(
            "Missing Section 2 inputs: "
            f"source={missing_source}, observations={missing_observation}, "
            f"existing_features={missing_existing}"
        )
    if set(bars["product"].astype(str).unique()) != {"GC"}:
        raise ValueError("Section 2 scalar construction is GC-only.")
    if not bars["source_row_id"].is_unique:
        raise ValueError("source_row_id must be unique.")
    if not observations["observation_id"].is_unique:
        raise ValueError("Eligible observation_id must be unique.")
    if not existing_features["observation_id"].is_unique:
        raise ValueError("Existing feature observation_id must be unique.")
    forbidden = [
        column
        for column in (*bars.columns, *observations.columns, *existing_features.columns)
        if any(
            token in str(column).lower()
            for token in ("forward_", "future_", "target", "label", "mfe", "mae")
        )
    ]
    if forbidden:
        raise ValueError(f"Outcome columns entered the Section 2 namespace: {forbidden}")


def build_scalar_feature_matrix(
    full_gc_bars: pd.DataFrame,
    eligible_observations: pd.DataFrame,
    existing_features: pd.DataFrame,
    *,
    allowed_partitions: Iterable[str] = ("Development", "Validation"),
    chunk_size: int = 20_000,
) -> ScalarFeatureBuildResult:
    """Build F01-F10, H01, and I01-I04 without reading any outcome."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    allowed = tuple(allowed_partitions)
    observations = eligible_observations.loc[
        eligible_observations["research_partition"].astype(str).isin(allowed),
        OBSERVATION_COLUMNS,
    ].copy()
    existing_ids = set(
        existing_features.loc[
            existing_features["observation_id"].isin(observations["observation_id"]),
            "observation_id",
        ].tolist()
    )
    if existing_ids != set(observations["observation_id"].tolist()):
        raise ValueError("Existing feature rows do not exactly cover selected observations.")
    _validate_inputs(full_gc_bars, observations, existing_features)

    bars = full_gc_bars.loc[:, SOURCE_COLUMNS].copy()
    if (
        not bars["source_row_id"].is_monotonic_increasing
        or not pd.to_datetime(bars["ts_event_utc"], utc=True).is_monotonic_increasing
    ):
        bars = bars.sort_values(
            ["ts_event_utc", "source_row_id"], kind="mergesort"
        ).reset_index(drop=True)
        sorted_during_build = True
    else:
        bars = bars.reset_index(drop=True)
        sorted_during_build = False

    source_ids = pd.to_numeric(bars["source_row_id"]).to_numpy(dtype=np.int64)
    decision_ids = pd.to_numeric(observations["decision_bar_id"]).to_numpy(dtype=np.int64)
    positions = np.searchsorted(source_ids, decision_ids)
    exact = (positions < len(source_ids)) & (
        source_ids[np.minimum(positions, len(source_ids) - 1)] == decision_ids
    )
    if not exact.all():
        raise ValueError(
            f"Decision bars absent from trusted GC source: {decision_ids[~exact][:10].tolist()}"
        )
    source_ts = (
        pd.to_datetime(bars["ts_event_utc"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    decision_ts = (
        pd.to_datetime(observations["decision_timestamp_utc"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    if not np.array_equal(source_ts[positions], decision_ts):
        raise ValueError("decision_bar_id and decision_timestamp_utc do not map exactly.")

    close = _as_float(bars["close"])
    high = _as_float(bars["high"])
    low = _as_float(bars["low"])
    volume = _as_float(bars["volume"])
    legal_group, legal_position, latest_cause = _legal_continuity(bars)
    previous_close = _lag(close, 1, legal_group)
    returns_bps = np.full(len(close), np.nan, dtype=np.float64)
    valid_return = (
        np.isfinite(close)
        & np.isfinite(previous_close)
        & (close > 0.0)
        & (previous_close > 0.0)
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
    volume_mean20 = _rolling_mean(volume, 20, legal_group)
    relative_volume20 = _safe_divide(volume, volume_mean20)
    log_volume_mean60 = _rolling_mean(log_volume, 60, legal_group)
    log_volume_std60 = _rolling_std_population(log_volume, 60, legal_group)
    volume_zscore60 = _safe_divide(
        log_volume - log_volume_mean60, log_volume_std60
    )

    row_count = len(observations)
    values = {
        name: np.full(row_count, np.nan, dtype=np.float64)
        for name in (*BASE_FEATURE_NAMES, *HELPER_FEATURE_NAMES)
    }
    reasons = {
        name: np.full(row_count, "COMPLETE", dtype=object)
        for name in SECTION2_FEATURE_NAMES
    }
    zero_flags = {
        "ret_mad_30_bps__zero_scale": np.zeros(row_count, dtype=bool),
        "ret_tail_balance_60__zero_denominator": np.zeros(row_count, dtype=bool),
        "ret_outlier_fraction_60__zero_scale": np.zeros(row_count, dtype=bool),
        "ordered_draw_balance_30__zero_denominator": np.zeros(row_count, dtype=bool),
    }

    history_masks: dict[str, np.ndarray] = {}
    for name, required_bars in RAW_HISTORY_BARS.items():
        history_masks[name] = _set_history_causes(
            reasons[name],
            legal_position,
            positions,
            latest_cause,
            required_bars,
        )

    for chunk_start in range(0, row_count, chunk_size):
        chunk_stop = min(row_count, chunk_start + chunk_size)
        chunk_rows = np.arange(chunk_start, chunk_stop, dtype=np.int64)

        complete30 = chunk_rows[
            history_masks["ret_trimmed_mean_30_bps"][chunk_start:chunk_stop]
        ]
        if len(complete30):
            ends = positions[complete30]
            return30 = _window(returns_bps, ends, 30)
            close31 = _window(close, ends, 31)
            input_null = ~np.isfinite(return30).all(axis=1)
            valid_rows = _assign_failure(
                reasons["ret_trimmed_mean_30_bps"],
                complete30,
                input_null,
                "SOURCE_INPUT_NULL",
            )
            if len(valid_rows):
                valid_return30 = return30[~input_null]
                values["ret_trimmed_mean_30_bps"][valid_rows] = _trimmed_mean(
                    valid_return30
                )
                mad_values = _mad(valid_return30)
                values["ret_mad_30_bps"][valid_rows] = mad_values
                zero_flags["ret_mad_30_bps__zero_scale"][valid_rows] = (
                    np.ptp(valid_return30, axis=1) == 0.0
                )
            reasons["ret_mad_30_bps"][complete30[input_null]] = "SOURCE_INPUT_NULL"

            close_null = ~np.isfinite(close31).all(axis=1)
            curvature_rows = _assign_failure(
                reasons["price_path_curvature_30_atr"],
                complete30,
                close_null,
                "SOURCE_INPUT_NULL",
            )
            draw_rows = _assign_failure(
                reasons["ordered_draw_balance_30"],
                complete30,
                close_null,
                "SOURCE_INPUT_NULL",
            )
            if len(curvature_rows):
                retained = ~close_null
                atr_values = atr20[positions[curvature_rows]]
                bad_atr = ~np.isfinite(atr_values) | (atr_values <= 0.0)
                curvature_valid_rows = _assign_failure(
                    reasons["price_path_curvature_30_atr"],
                    curvature_rows,
                    bad_atr,
                    "NONPOSITIVE_ATR",
                )
                if len(curvature_valid_rows):
                    values["price_path_curvature_30_atr"][
                        curvature_valid_rows
                    ] = _curvature(
                        close31[retained][~bad_atr],
                        atr_values[~bad_atr],
                    )
            if len(draw_rows):
                draw_values, zero_draw = _ordered_draw_balance(close31[~close_null])
                values["ordered_draw_balance_30"][draw_rows] = draw_values
                zero_flags["ordered_draw_balance_30__zero_denominator"][
                    draw_rows
                ] = zero_draw

        complete60 = chunk_rows[
            history_masks["ret_tail_balance_60"][chunk_start:chunk_stop]
        ]
        if len(complete60):
            ends = positions[complete60]
            return60 = _window(returns_bps, ends, 60)
            input_null = ~np.isfinite(return60).all(axis=1)
            for name in (
                "ret_tail_balance_60",
                "ret_outlier_fraction_60",
                "return_acf_energy_60",
            ):
                reasons[name][complete60[input_null]] = "SOURCE_INPUT_NULL"
            valid_rows = complete60[~input_null]
            valid_return60 = return60[~input_null]
            if len(valid_rows):
                tail, zero_tail = _tail_balance(valid_return60)
                outlier, zero_outlier = _outlier_fraction(valid_return60)
                acf, acf_degenerate = _acf_energy(valid_return60)
                values["ret_tail_balance_60"][valid_rows] = tail
                values["ret_outlier_fraction_60"][valid_rows] = outlier
                values["return_acf_energy_60"][valid_rows] = acf
                zero_flags["ret_tail_balance_60__zero_denominator"][
                    valid_rows
                ] = zero_tail
                zero_flags["ret_outlier_fraction_60__zero_scale"][
                    valid_rows
                ] = zero_outlier
                reasons["return_acf_energy_60"][
                    valid_rows[acf_degenerate]
                ] = "DEGENERATE_STATISTIC"

        complete_f08 = chunk_rows[
            history_masks["lagged_volume_return_spearman_30"][chunk_start:chunk_stop]
        ]
        if len(complete_f08):
            ends = positions[complete_f08]
            lagged_zv = _window(volume_zscore60, ends - 1, 30)
            return30 = _window(returns_bps, ends, 30)
            source_null = ~np.isfinite(return30).all(axis=1)
            source_rows = _assign_failure(
                reasons["lagged_volume_return_spearman_30"],
                complete_f08,
                source_null,
                "SOURCE_INPUT_NULL",
            )
            if len(source_rows):
                spearman, degenerate = _spearman_rows(
                    lagged_zv[~source_null], return30[~source_null]
                )
                values["lagged_volume_return_spearman_30"][source_rows] = spearman
                reasons["lagged_volume_return_spearman_30"][
                    source_rows[degenerate]
                ] = "DEGENERATE_STATISTIC"

        complete_f09 = chunk_rows[
            history_masks["range_volume_spearman_30"][chunk_start:chunk_stop]
        ]
        if len(complete_f09):
            ends = positions[complete_f09]
            tr30 = _window(true_range, ends, 30)
            atr30 = _window(atr20, ends, 30)
            zv30 = _window(volume_zscore60, ends, 30)
            source_null = ~np.isfinite(tr30).all(axis=1)
            source_rows = _assign_failure(
                reasons["range_volume_spearman_30"],
                complete_f09,
                source_null,
                "SOURCE_INPUT_NULL",
            )
            if len(source_rows):
                retained = ~source_null
                bad_atr = ~np.isfinite(atr30[retained]).all(axis=1) | (
                    atr30[retained] <= 0.0
                ).any(axis=1)
                atr_rows = _assign_failure(
                    reasons["range_volume_spearman_30"],
                    source_rows,
                    bad_atr,
                    "NONPOSITIVE_ATR",
                )
                if len(atr_rows):
                    retained_again = ~bad_atr
                    normalized_range = (
                        tr30[retained][retained_again]
                        / atr30[retained][retained_again]
                    )
                    spearman, degenerate = _spearman_rows(
                        normalized_range, zv30[retained][retained_again]
                    )
                    values["range_volume_spearman_30"][atr_rows] = spearman
                    reasons["range_volume_spearman_30"][
                        atr_rows[degenerate]
                    ] = "DEGENERATE_STATISTIC"

        complete_f10 = chunk_rows[
            history_masks["volume_profile_slope_30"][chunk_start:chunk_stop]
        ]
        if len(complete_f10):
            ends = positions[complete_f10]
            zv30 = _window(volume_zscore60, ends, 30)
            source_null = ~np.isfinite(zv30).all(axis=1)
            valid_rows = _assign_failure(
                reasons["volume_profile_slope_30"],
                complete_f10,
                source_null,
                "SOURCE_INPUT_NULL",
            )
            if len(valid_rows):
                values["volume_profile_slope_30"][valid_rows] = _profile_slope(
                    zv30[~source_null]
                )

    h01_history = history_masks["relative_volume_20_clipped"]
    h01_candidate = np.flatnonzero(h01_history)
    if len(h01_candidate):
        h01_raw = relative_volume20[positions[h01_candidate]]
        h01_null = ~np.isfinite(h01_raw)
        h01_valid = _assign_failure(
            reasons["relative_volume_20_clipped"],
            h01_candidate,
            h01_null,
            "SOURCE_INPUT_NULL",
        )
        values["relative_volume_20_clipped"][h01_valid] = np.clip(
            h01_raw[~h01_null], 0.5, 2.0
        )

    join_columns = ["observation_id", *PARENT_FEATURE_NAMES, *COMPARATOR_NAMES]
    existing_subset = existing_features.loc[
        existing_features["observation_id"].isin(observations["observation_id"]),
        list(dict.fromkeys(join_columns)),
    ]
    observation_order = observations[["observation_id"]].assign(
        __eligible_order=np.arange(row_count, dtype=np.int64)
    )
    joined = observation_order.merge(
        existing_subset,
        on="observation_id",
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("__eligible_order", kind="stable")
    if not np.array_equal(
        joined["observation_id"].to_numpy(),
        observations["observation_id"].to_numpy(),
    ):
        raise AssertionError("Existing parents were not joined in eligible observation order.")

    parent = {name: _as_float(joined[name]) for name in PARENT_FEATURE_NAMES}
    i04_history_complete = _set_history_causes(
        reasons["vwap_trend_alignment_30"],
        legal_position,
        positions,
        latest_cause,
        30,
    )
    interaction_definitions = {
        "curvature_coherence_30": (
            "price_path_curvature_30_atr",
            parent["efficiency_ratio_30"],
        ),
        "tail_pressure_activity_60": (
            "ret_tail_balance_60",
            values["relative_volume_20_clipped"],
        ),
        "lagged_volume_confirmation_30": (
            "lagged_volume_return_spearman_30",
            values["relative_volume_20_clipped"],
        ),
        "vwap_trend_alignment_30": (
            None,
            parent["distance_from_research_day_vwap_atr"]
            * parent["normalized_ols_slope_30"],
        ),
    }
    interaction_values: dict[str, np.ndarray] = {}
    for name, (base_name, second) in interaction_definitions.items():
        if name == "vwap_trend_alignment_30":
            result = np.asarray(second, dtype=np.float64)
            parent_null = ~np.isfinite(
                parent["distance_from_research_day_vwap_atr"]
            ) | ~np.isfinite(parent["normalized_ols_slope_30"])
            eligible_parent_null = parent_null & i04_history_complete
            decision_atr = atr20[positions]
            bad_atr = eligible_parent_null & (
                ~np.isfinite(decision_atr) | (decision_atr <= 0.0)
            )
            reasons[name][eligible_parent_null] = "SOURCE_INPUT_NULL"
            reasons[name][bad_atr] = "NONPOSITIVE_ATR"
        else:
            assert base_name is not None
            first = values[base_name]
            result = first * np.asarray(second, dtype=np.float64)
            reasons[name][:] = reasons[base_name]
            second_null = np.isfinite(first) & ~np.isfinite(second)
            reasons[name][second_null] = "SOURCE_INPUT_NULL"
        result[np.asarray(reasons[name]) != "COMPLETE"] = np.nan
        interaction_values[name] = result

    matrix = observations.loc[
        :,
        [
            "observation_id",
            "decision_timestamp_utc",
            "trade_date_ny",
            "entry_session",
            "research_partition",
        ],
    ].reset_index(drop=True)
    all_values = {**values, **interaction_values}
    for name in SECTION2_FEATURE_NAMES:
        numeric = np.asarray(all_values[name], dtype=np.float64)
        complete = np.asarray(reasons[name]) == "COMPLETE"
        if not np.isfinite(numeric[complete]).all():
            raise AssertionError(f"{name} has non-finite values marked COMPLETE.")
        minimum, maximum = RANGE_LIMITS[name]
        if minimum is not None and np.any(numeric[complete] < minimum - 1.0e-10):
            raise AssertionError(f"{name} violates its lower bound.")
        if maximum is not None and np.any(numeric[complete] > maximum + 1.0e-10):
            raise AssertionError(f"{name} violates its upper bound.")
        matrix[name] = numeric.astype(np.float32)

    missing_frame = observations[["observation_id"]].reset_index(drop=True)
    cause_dtype = pd.CategoricalDtype(categories=list(MISSING_CAUSES), ordered=True)
    for name in SECTION2_FEATURE_NAMES:
        missing_frame[f"{name}{CAUSE_COLUMN_SUFFIX}"] = pd.Categorical(
            reasons[name], dtype=cause_dtype
        )
    for name in DEGENERATE_PERMITTED_FEATURES:
        missing_frame[f"{name}{DEGENERATE_INDICATOR_SUFFIX}"] = (
            np.asarray(reasons[name]) == "DEGENERATE_STATISTIC"
        )

    zero_scale_frame = observations[["observation_id"]].reset_index(drop=True)
    for name, flag in zero_flags.items():
        zero_scale_frame[name] = flag

    audit = pd.DataFrame.from_records(
        [
            {"item": "source_rows", "value": str(len(bars))},
            {"item": "selected_observations", "value": str(row_count)},
            {"item": "selected_partitions", "value": "|".join(allowed)},
            {"item": "historical_final_materialized", "value": "False"},
            {"item": "outcome_columns_accepted", "value": "False"},
            {"item": "source_sorted_during_build", "value": str(sorted_during_build)},
            {"item": "decision_bar_ids_exact", "value": "True"},
            {"item": "decision_timestamps_exact", "value": "True"},
            {"item": "parent_join_key", "value": "observation_id"},
            {"item": "parent_join_cardinality", "value": "one_to_one"},
            {"item": "legal_group_count", "value": str(int(legal_group[-1] + 1))},
            {"item": "ny_date_outer_reset", "value": "True"},
            {"item": "calculation_dtype", "value": "float64"},
            {"item": "saved_scalar_dtype", "value": "float32"},
            {"item": "base_feature_count", "value": str(len(BASE_FEATURE_NAMES))},
            {"item": "helper_feature_count", "value": str(len(HELPER_FEATURE_NAMES))},
            {"item": "interaction_feature_count", "value": str(len(INTERACTION_FEATURE_NAMES))},
        ]
    )
    return ScalarFeatureBuildResult(
        matrix=matrix,
        missing_reasons=missing_frame,
        zero_scale_flags=zero_scale_frame,
        registry=scalar_feature_registry(),
        construction_audit=audit,
    )


def build_coverage_audit(
    matrix: pd.DataFrame,
    missing_reasons: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize Development missing causes by feature and session."""

    metadata = matrix[
        ["observation_id", "trade_date_ny", "entry_session", "research_partition"]
    ]
    if set(metadata["research_partition"].astype(str).unique()) != {"Development"}:
        raise ValueError("Coverage diagnostics are Development-only in Section 2.")
    merged = metadata.merge(
        missing_reasons, on="observation_id", validate="one_to_one", how="left"
    )
    records: list[dict[str, object]] = []
    for feature in SECTION2_FEATURE_NAMES:
        cause_column = f"{feature}{CAUSE_COLUMN_SUFFIX}"
        for session, group in merged.groupby("entry_session", observed=True, sort=True):
            counts = group[cause_column].value_counts(dropna=False)
            complete = group[cause_column].astype(str).eq("COMPLETE")
            complete_dates = group.loc[complete, "trade_date_ny"].nunique()
            for cause in MISSING_CAUSES:
                count = int(counts.get(cause, 0))
                records.append(
                    {
                        "feature_name": feature,
                        "entry_session": str(session),
                        "cause": cause,
                        "rows": count,
                        "rate": count / len(group) if len(group) else np.nan,
                        "complete_ny_dates": int(complete_dates)
                        if cause == "COMPLETE"
                        else pd.NA,
                    }
                )
    return pd.DataFrame.from_records(records)


def build_integrity_audit(
    matrix: pd.DataFrame,
    missing_reasons: pd.DataFrame,
    zero_scale_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Check bounds, finiteness, constants, degeneracy, and exact duplicates."""

    cause_frame = missing_reasons.set_index("observation_id")
    zero_frame = zero_scale_flags.set_index("observation_id")
    duplicate_of: dict[str, str] = {}
    for index, name in enumerate(SECTION2_FEATURE_NAMES):
        left = matrix[name].to_numpy(dtype=np.float32)
        for other in SECTION2_FEATURE_NAMES[:index]:
            right = matrix[other].to_numpy(dtype=np.float32)
            if np.array_equal(left, right, equal_nan=True):
                duplicate_of[name] = other
                break

    records: list[dict[str, object]] = []
    for name in SECTION2_FEATURE_NAMES:
        numeric = pd.to_numeric(matrix[name], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        finite = np.isfinite(numeric)
        complete = cause_frame[f"{name}{CAUSE_COLUMN_SUFFIX}"].astype(str).eq(
            "COMPLETE"
        )
        minimum, maximum = RANGE_LIMITS[name]
        below = int(np.sum(finite & (numeric < minimum))) if minimum is not None else 0
        above = int(np.sum(finite & (numeric > maximum))) if maximum is not None else 0
        complete_values = numeric[np.asarray(complete)]
        unique_complete = np.unique(complete_values).size
        near_constant = (
            bool(np.nanstd(complete_values) <= 1.0e-8)
            if len(complete_values)
            else False
        )
        zero_columns = [
            column for column in zero_frame.columns if column.startswith(f"{name}__")
        ]
        zero_count = (
            int(zero_frame.loc[:, zero_columns].any(axis=1).sum())
            if zero_columns
            else 0
        )
        reason_values = cause_frame[f"{name}{CAUSE_COLUMN_SUFFIX}"].astype(str)
        integrity_failures = int(reason_values.isin(INTEGRITY_FAILURE_CAUSES).sum())
        degeneracy = int(reason_values.eq("DEGENERATE_STATISTIC").sum())
        records.append(
            {
                "feature_name": name,
                "rows": len(matrix),
                "complete_rows": int(complete.sum()),
                "finite_rows": int(finite.sum()),
                "infinite_rows": int(np.isinf(numeric).sum()),
                "below_range_rows": below,
                "above_range_rows": above,
                "unique_complete_values": int(unique_complete),
                "constant": bool(unique_complete <= 1),
                "near_constant_std_le_1e-8": near_constant,
                "exact_duplicate_of": duplicate_of.get(name, ""),
                "formula_defined_zero_rows": zero_count,
                "degenerate_rows": degeneracy,
                "integrity_failure_rows": integrity_failures,
                "status": (
                    "PASS"
                    if not any(
                        (
                            np.isinf(numeric).any(),
                            below,
                            above,
                            integrity_failures,
                            bool(duplicate_of.get(name)),
                        )
                    )
                    else "FAIL"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_distribution_diagnostics(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return concise outcome-free Development distributions by year/session."""

    if set(matrix["research_partition"].astype(str).unique()) != {"Development"}:
        raise ValueError("Distribution diagnostics are Development-only in Section 2.")
    frame = matrix.copy()
    frame["year"] = pd.to_datetime(frame["trade_date_ny"]).dt.year.astype("int16")
    records: list[dict[str, object]] = []
    for (year, session), group in frame.groupby(
        ["year", "entry_session"], observed=True, sort=True
    ):
        for name in SECTION2_FEATURE_NAMES:
            values = pd.to_numeric(group[name], errors="coerce")
            records.append(
                {
                    "year": int(year),
                    "entry_session": str(session),
                    "feature_name": name,
                    "rows": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "minimum": float(values.min()),
                    "q25": float(values.quantile(0.25, interpolation="linear")),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75, interpolation="linear")),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame.from_records(records)


def build_named_comparator_correlations(
    development_matrix: pd.DataFrame,
    existing_features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute outcome-free Spearman redundancy against frozen comparators."""

    if set(development_matrix["research_partition"].astype(str).unique()) != {
        "Development"
    }:
        raise ValueError("Redundancy diagnostics are Development-only in Section 2.")
    existing = existing_features.loc[
        existing_features["observation_id"].isin(development_matrix["observation_id"]),
        ["observation_id", *COMPARATOR_NAMES],
    ]
    joined = development_matrix.merge(
        existing, on="observation_id", how="left", validate="one_to_one", sort=False
    )
    records: list[dict[str, object]] = []
    for candidate, comparators in COMPARATOR_MAP.items():
        for slot, comparator in enumerate(comparators, start=1):
            pair = joined[[candidate, comparator]].dropna()
            correlation = (
                float(pair[candidate].corr(pair[comparator], method="spearman"))
                if len(pair) >= 3
                else np.nan
            )
            records.append(
                {
                    "feature_name": candidate,
                    "comparator_slot": slot,
                    "comparator_name": comparator,
                    "complete_pairs": len(pair),
                    "spearman_correlation": correlation,
                    "absolute_spearman_correlation": abs(correlation),
                }
            )
    return pd.DataFrame.from_records(records)
