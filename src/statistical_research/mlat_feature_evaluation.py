"""Preregistered univariate evaluation for the frozen MLAT feature batch.

This module evaluates information content, not trading performance.  All
fitted objects (notably quintile edges) are learned on Development only and
then applied unchanged to Validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .mlat_feature_registry import MLAT_METADATA_COLUMNS

MLAT_HORIZONS = (5, 15, 30, 60, 120, 180)
MLAT_SESSIONS = ("London", "New York")
MLAT_PARTITIONS = ("Development", "Validation")
MLAT_TARGET_FAMILIES = ("direction", "expansion", "volatility", "path_risk")
MLAT_SEED = 20260723
MLAT_BOOTSTRAPS = 2_000
REQUIRED_INCREMENTAL_CONTROL_SETS = (
    "atr_20",
    "frozen_15",
)
VERDICT_LABELS = (
    "ADVANCE_DIRECTIONAL",
    "ADVANCE_EXPANSION",
    "ADVANCE_RISK_STATE",
    "REDUNDANT_WITH_EXISTING",
    "ECONOMICALLY_TRIVIAL",
    "WEAK_OR_UNSTABLE",
    "FAILED_VALIDATION",
    "NO_EVIDENCE",
    "RESEARCH_ONLY",
)
MLAT_SOURCE_TRACEABILITY = {
    "bollinger_zscore_20": ("MLAT-H001", "4; Appendix", "131-133; 740-742"),
    "bollinger_bandwidth_20": ("MLAT-H002", "Appendix", "740-742"),
    "cutler_rsi_14": ("MLAT-H003", "4", "132"),
    "chaikin_money_flow_20": ("MLAT-H004", "Appendix", "752-753"),
    "amihud_illiquidity_60": ("MLAT-H005", "20; Appendix", "656; 752"),
    "parkinson_volatility_30": ("MLAT-H006", "9", "297-301"),
    "rogers_satchell_volatility_30": ("MLAT-H007", "9", "297-301"),
    "realized_semivariance_balance_60": (
        "MLAT-H008",
        "5; 9",
        "169-178; 297-301",
    ),
    "bipower_jump_ratio_60": ("MLAT-H009", "9", "297-301"),
    "variance_ratio_60_5": ("MLAT-H010", "9", "280-296"),
    "return_sign_entropy_60": ("MLAT-H011", "6", "192"),
    "volatility_of_volatility_60": ("MLAT-H012", "9", "297-301"),
}


@dataclass(frozen=True)
class MlatEvaluationConfig:
    """Frozen evaluation settings, with count overrides for unit tests."""

    horizons: tuple[int, ...] = MLAT_HORIZONS
    sessions: tuple[str, ...] = MLAT_SESSIONS
    partitions: tuple[str, ...] = MLAT_PARTITIONS
    seed: int = MLAT_SEED
    bootstrap_iterations: int = MLAT_BOOTSTRAPS
    confidence_level: float = 0.95
    minimum_observations_per_day: int = 10
    minimum_development_dates: int = 400
    minimum_validation_dates: int = 150
    minimum_development_observations: int = 10_000
    minimum_validation_observations: int = 4_000
    development_q_threshold: float = 0.10
    validation_retention: float = 0.25
    monotonicity_threshold: float = 0.80
    direction_tick_spread: float = 2.0
    overlap_threshold: float = 0.995
    atr_partial_ic_threshold: float = 0.02
    frozen_partial_ic_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")


@dataclass(frozen=True)
class MlatEvaluationFrameResult:
    """Evaluation frame and leakage/coverage evidence."""

    frame: pd.DataFrame
    checks: pd.DataFrame
    target_catalog: pd.DataFrame
    family_coverage: pd.DataFrame

    @property
    def ready(self) -> bool:
        """Whether every construction/alignment check passed."""

        return bool(
            not self.checks.empty
            and {"passed", "check"}.issubset(self.checks.columns)
            and self.checks["passed"].fillna(False).astype(bool).all()
        )


@dataclass(frozen=True)
class MlatCellEvaluationResult:
    """Cell-level evaluation tables suitable for independent persistence."""

    cell_results: pd.DataFrame
    daily_ic: pd.DataFrame
    quintile_edges: pd.DataFrame
    quintile_results: pd.DataFrame
    year_stability: pd.DataFrame
    thinning_results: pd.DataFrame


@dataclass(frozen=True)
class MlatIncrementalResult:
    """Daily partial-IC evidence and Development/Validation confirmations."""

    daily_partial_ic: pd.DataFrame
    summary: pd.DataFrame


def _as_frame(value: pd.DataFrame | MlatEvaluationFrameResult) -> pd.DataFrame:
    if isinstance(value, MlatEvaluationFrameResult):
        if not value.ready:
            failed = value.checks.loc[
                ~value.checks["passed"].fillna(False).astype(bool), "check"
            ].astype(str)
            raise ValueError(
                "evaluation frame checks failed: " + ", ".join(failed.tolist())
            )
        return value.frame
    return value


def _strict_boolean(series: pd.Series, column: str) -> pd.Series:
    """Return a non-null boolean flag or reject permissive string coercion."""

    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{column} must have a boolean dtype")
    if series.isna().any():
        raise ValueError(f"{column} must not contain missing values")
    return series.astype(bool)


def _normalized_comparison(series: pd.Series, column: str) -> pd.Series:
    if "timestamp" in column:
        return pd.to_datetime(series, errors="coerce", utc=True)
    if column == "trade_date_ny":
        return pd.to_datetime(series, errors="coerce").dt.normalize()
    return series.astype("string").fillna("<NA>")


def _features(registry: pd.DataFrame | Iterable[str]) -> list[str]:
    if isinstance(registry, pd.DataFrame):
        if "feature_name" not in registry:
            raise KeyError("registry must contain feature_name")
        names = registry["feature_name"].astype(str).tolist()
    else:
        names = list(registry)
    if len(names) != len(set(names)):
        raise ValueError("feature names must be unique")
    return names


def _target_name(family: str, horizon: int) -> str:
    names = {
        "direction": f"forward_return_{horizon}_atr",
        "expansion": f"future_range_{horizon}_atr",
        "volatility": f"future_realized_volatility_{horizon}_bps",
        "path_risk": f"path_risk_{horizon}_atr",
    }
    try:
        return names[family]
    except KeyError as error:
        raise ValueError(f"unknown target family: {family}") from error


def _target_catalog(horizons: Sequence[int]) -> pd.DataFrame:
    rows = []
    for family in MLAT_TARGET_FAMILIES:
        for horizon in horizons:
            rows.append(
                {
                    "target_family": family,
                    "horizon_minutes": horizon,
                    "target_column": _target_name(family, horizon),
                    "tick_effect_column": (
                        f"forward_return_{horizon}_ticks"
                        if family == "direction"
                        else pd.NA
                    ),
                    "atr_normalized": family in {"direction", "expansion", "path_risk"},
                    "primary_statistic": "mean New-York-date Spearman IC",
                }
            )
    return pd.DataFrame(rows)


def _check_frame_scope(
    frame: pd.DataFrame,
    *,
    config: MlatEvaluationConfig | None = None,
    require_family_masks: bool = True,
) -> None:
    settings = config or MlatEvaluationConfig()
    required = {
        "product",
        "research_partition",
        "trade_date_ny",
        "entry_session",
    }
    if require_family_masks:
        required.update(
            f"{family}_complete_sample" for family in MLAT_TARGET_FAMILIES
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"evaluation frame scope columns missing: {missing}")
    if frame.empty:
        raise ValueError("evaluation frame must not be empty")
    if frame["product"].isna().any():
        raise ValueError("evaluation frame product must not be missing")
    products = set(frame["product"].astype(str))
    if products != {"GC"}:
        raise ValueError(f"MLAT evaluation is GC-only; found {sorted(products)}")
    if frame["research_partition"].isna().any():
        raise ValueError("research_partition must not be missing")
    partitions = set(frame["research_partition"].astype(str))
    expected_partitions = set(settings.partitions)
    if partitions != expected_partitions:
        raise ValueError(
            "evaluation requires exactly the configured Development/Validation scope; "
            f"found {sorted(partitions)}, expected {sorted(expected_partitions)}"
        )
    sessions = set(frame["entry_session"].dropna().astype(str))
    disallowed_sessions = sessions.difference(settings.sessions)
    if not sessions or disallowed_sessions:
        raise ValueError(
            f"evaluation sessions must be drawn from {settings.sessions}; "
            f"found {sorted(sessions)}"
        )
    dates = pd.to_datetime(frame["trade_date_ny"], errors="coerce")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("trade_date_ny must contain non-null date-only values")
    if require_family_masks:
        for family in MLAT_TARGET_FAMILIES:
            column = f"{family}_complete_sample"
            _strict_boolean(frame[column], column)


def build_mlat_evaluation_frame(
    feature_matrix: pd.DataFrame,
    forward_labels: pd.DataFrame,
    registry: pd.DataFrame | Iterable[str],
    existing_feature_matrix: pd.DataFrame | None = None,
    *,
    frozen_features: pd.DataFrame | Iterable[str] | None = None,
    config: MlatEvaluationConfig | None = None,
) -> MlatEvaluationFrameResult:
    """Align features and labels and construct the four frozen target families.

    Historical Final rows are counted in the checks table and excluded.  MGC
    or any other product is rejected rather than silently mixed with GC.
    """

    settings = config or MlatEvaluationConfig()
    feature_names = _features(registry)
    required_feature = {*MLAT_METADATA_COLUMNS, *feature_names}
    required_label = {
        *MLAT_METADATA_COLUMNS,
        "entry_session",
        "decision_atr_20m",
        "atr_normalization_available",
    }
    for horizon in settings.horizons:
        required_label.update(
            {
                f"forward_return_{horizon}_atr",
                f"forward_return_{horizon}_ticks",
                f"future_range_{horizon}_atr",
                f"future_realized_volatility_{horizon}_bps",
                f"mae_long_{horizon}_atr",
                f"mae_short_{horizon}_atr",
                f"label_available_{horizon}",
            }
        )
    missing_features = sorted(required_feature.difference(feature_matrix.columns))
    missing_labels = sorted(required_label.difference(forward_labels.columns))
    if missing_features or missing_labels:
        raise KeyError(
            {
                "missing_feature_columns": missing_features,
                "missing_label_columns": missing_labels,
            }
        )
    if feature_matrix["observation_id"].duplicated().any():
        raise ValueError("feature_matrix observation_id must be unique")
    if forward_labels["observation_id"].duplicated().any():
        raise ValueError("forward_labels observation_id must be unique")
    _strict_boolean(
        forward_labels["atr_normalization_available"],
        "atr_normalization_available",
    )
    for horizon in settings.horizons:
        column = f"label_available_{horizon}"
        _strict_boolean(forward_labels[column], column)
    feature_products = sorted(
        feature_matrix["product"].dropna().astype(str).unique().tolist()
    )
    label_products = sorted(
        forward_labels["product"].dropna().astype(str).unique().tolist()
    )
    if feature_products != ["GC"] or label_products != ["GC"]:
        raise ValueError(
            "MLAT evaluation is GC-only; "
            f"feature products={feature_products}, label products={label_products}"
        )

    final_partition_labels = {"Final", "Final test"}
    allowed_partitions = {*settings.partitions, *final_partition_labels}
    feature_partition_values = feature_matrix["research_partition"].astype(str)
    label_partition_values = forward_labels["research_partition"].astype(str)
    feature_unknown = ~feature_partition_values.isin(allowed_partitions)
    label_unknown = ~label_partition_values.isin(allowed_partitions)
    other_rows = int(feature_unknown.sum() + label_unknown.sum())
    feature_scope = feature_matrix.loc[
        feature_partition_values.isin(settings.partitions)
    ].copy()
    label_scope = forward_labels.loc[
        label_partition_values.isin(settings.partitions)
    ].copy()
    feature_ids = set(feature_scope["observation_id"].tolist())
    label_ids = set(label_scope["observation_id"].tolist())
    feature_only_ids = feature_ids.difference(label_ids)
    label_only_ids = label_ids.difference(feature_ids)
    provenance_columns = [
        column
        for column in MLAT_METADATA_COLUMNS
        if column != "observation_id"
    ]
    aligned = feature_scope[
        ["observation_id", *provenance_columns]
    ].merge(
        label_scope[["observation_id", *provenance_columns]],
        on="observation_id",
        how="inner",
        validate="one_to_one",
        suffixes=("__feature", "__label"),
    )
    provenance_mismatches: dict[str, int] = {}
    for column in provenance_columns:
        left = _normalized_comparison(aligned[f"{column}__feature"], column)
        right = _normalized_comparison(aligned[f"{column}__label"], column)
        mismatch = ~(left.eq(right) | (left.isna() & right.isna()))
        provenance_mismatches[column] = int(mismatch.sum())

    base_columns = [*MLAT_METADATA_COLUMNS, *feature_names]
    label_columns = [
        column
        for column in label_scope.columns
        if column not in feature_names
        and column != "observation_id"
        and column not in base_columns
    ]
    frame = feature_scope[base_columns].merge(
        label_scope[["observation_id", *label_columns]],
        on="observation_id",
        how="inner",
        validate="one_to_one",
    )
    final_rows = int(
        feature_partition_values.isin(final_partition_labels).sum()
        + label_partition_values.isin(final_partition_labels).sum()
    )
    if existing_feature_matrix is not None:
        if "observation_id" not in existing_feature_matrix:
            raise KeyError("existing_feature_matrix must contain observation_id")
        if existing_feature_matrix["observation_id"].duplicated().any():
            raise ValueError("existing_feature_matrix observation_id must be unique")
        if isinstance(frozen_features, pd.DataFrame):
            frozen = frozen_features
            if "in_frozen_set" in frozen:
                frozen = frozen.loc[frozen["in_frozen_set"].fillna(False)]
            existing_names = frozen["feature_name"].astype(str).tolist()
        elif frozen_features is None:
            existing_names = [
                column
                for column in existing_feature_matrix.columns
                if column not in {"observation_id", *frame.columns}
            ]
        else:
            existing_names = list(frozen_features)
        if "atr_20" in existing_feature_matrix and "atr_20" not in existing_names:
            existing_names = ["atr_20", *existing_names]
        existing_names = list(dict.fromkeys(existing_names))
        missing_existing = sorted(set(existing_names).difference(existing_feature_matrix.columns))
        if missing_existing:
            raise KeyError(f"existing controls missing: {missing_existing}")
        frame = frame.merge(
            existing_feature_matrix[["observation_id", *existing_names]],
            on="observation_id",
            how="left",
            validate="one_to_one",
        )

    for horizon in settings.horizons:
        long_mae = pd.to_numeric(frame[f"mae_long_{horizon}_atr"], errors="coerce")
        short_mae = pd.to_numeric(frame[f"mae_short_{horizon}_atr"], errors="coerce")
        frame[f"path_risk_{horizon}_atr"] = np.maximum(long_mae, short_mae)

    atr_valid = (
        _strict_boolean(
            frame["atr_normalization_available"],
            "atr_normalization_available",
        )
        & pd.to_numeric(frame["decision_atr_20m"], errors="coerce").gt(0)
    )
    coverage_rows = []
    for family in MLAT_TARGET_FAMILIES:
        targets = [_target_name(family, horizon) for horizon in settings.horizons]
        if family == "direction":
            targets.extend(
                f"forward_return_{horizon}_ticks" for horizon in settings.horizons
            )
        complete = frame[targets].apply(pd.to_numeric, errors="coerce").apply(
            lambda series: np.isfinite(series)
        )
        family_complete = complete.all(axis=1)
        for horizon in settings.horizons:
            column = f"label_available_{horizon}"
            family_complete &= _strict_boolean(frame[column], column)
        if family in {"direction", "expansion", "path_risk"}:
            family_complete &= atr_valid
        column = f"{family}_complete_sample"
        frame[column] = family_complete
        for partition in settings.partitions:
            for session in settings.sessions:
                scope = frame["research_partition"].astype(str).eq(partition) & frame[
                    "entry_session"
                ].astype(str).eq(session)
                coverage_rows.append(
                    {
                        "target_family": family,
                        "research_partition": partition,
                        "entry_session": session,
                        "eligible_rows": int(scope.sum()),
                        "complete_rows": int((scope & family_complete).sum()),
                        "coverage_rate": (
                            float(family_complete[scope].mean()) if scope.any() else np.nan
                        ),
                    }
                )
    checks = pd.DataFrame(
        [
            {
                "check": "gc_only",
                "passed": feature_products == ["GC"] and label_products == ["GC"],
                "observed": {
                    "feature_products": feature_products,
                    "label_products": label_products,
                },
                "expected": {"feature_products": ["GC"], "label_products": ["GC"]},
            },
            {
                "check": "feature_label_id_alignment",
                "passed": not feature_only_ids and not label_only_ids,
                "observed": {
                    "feature_only": len(feature_only_ids),
                    "label_only": len(label_only_ids),
                },
                "expected": {"feature_only": 0, "label_only": 0},
            },
            {
                "check": "feature_label_provenance_alignment",
                "passed": not any(provenance_mismatches.values()),
                "observed": provenance_mismatches,
                "expected": {column: 0 for column in provenance_columns},
            },
            {
                "check": "final_excluded",
                "passed": not frame["research_partition"]
                .astype(str)
                .isin(final_partition_labels)
                .any(),
                "observed": final_rows,
                "expected": "excluded from evaluation frame",
            },
            {
                "check": "unknown_partition_excluded",
                "passed": other_rows == 0,
                "observed": other_rows,
                "expected": 0,
            },
            {
                "check": "new_york_date_evidence_present",
                "passed": frame["trade_date_ny"].notna().all(),
                "observed": int(frame["trade_date_ny"].isna().sum()),
                "expected": 0,
            },
        ]
    )
    _check_frame_scope(frame, config=settings)
    return MlatEvaluationFrameResult(
        frame=frame.reset_index(drop=True),
        checks=checks,
        target_catalog=_target_catalog(settings.horizons),
        family_coverage=pd.DataFrame(coverage_rows),
    )


def benjamini_hochberg(p_values: Sequence[float] | pd.Series) -> np.ndarray:
    """Return BH-adjusted q-values while preserving NaN positions."""

    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    finite = values[valid]
    order = np.argsort(finite, kind="mergesort")
    ranked = finite[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result[valid] = restored
    return result


def date_block_bootstrap(
    values: Sequence[float] | pd.Series,
    *,
    iterations: int = MLAT_BOOTSTRAPS,
    seed: int = MLAT_SEED,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Fixed-seed bootstrap of a mean where each value is one New York date."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"lower": np.nan, "upper": np.nan, "p_value": np.nan}
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    means = array[indices].mean(axis=1)
    alpha = 1 - confidence_level
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p_value = min(
        1.0,
        2
        * min(
            (np.count_nonzero(means <= 0) + 1) / (iterations + 1),
            (np.count_nonzero(means >= 0) + 1) / (iterations + 1),
        ),
    )
    return {"lower": float(lower), "upper": float(upper), "p_value": float(p_value)}


def _cell_seed(config: MlatEvaluationConfig, *parts: object) -> int:
    digest = sha256("|".join(map(str, parts)).encode()).digest()
    offset = int.from_bytes(digest[:4], "little")
    return (config.seed + offset) % (2**32)


def calculate_daily_spearman(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    date_column: str = "trade_date_ny",
    minimum_observations: int = 10,
) -> pd.DataFrame:
    """Calculate one Spearman IC per New York date."""

    missing = sorted({feature, target, date_column}.difference(frame.columns))
    if missing:
        raise KeyError(f"daily IC columns missing: {missing}")
    rows = []
    for date, group in frame.groupby(date_column, observed=True, sort=True):
        x = pd.to_numeric(group[feature], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        y = pd.to_numeric(group[target], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        valid = np.isfinite(x) & np.isfinite(y)
        count = int(valid.sum())
        correlation = (
            float(spearmanr(x[valid], y[valid]).statistic)
            if count >= minimum_observations
            and np.unique(x[valid]).size > 1
            and np.unique(y[valid]).size > 1
            else np.nan
        )
        rows.append(
            {
                "trade_date_ny": date,
                "observations": count,
                "daily_spearman_ic": correlation,
                "eligible_date": bool(np.isfinite(correlation)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "trade_date_ny",
            "observations",
            "daily_spearman_ic",
            "eligible_date",
        ],
    )


def fit_development_quintiles(values: Sequence[float] | pd.Series) -> np.ndarray:
    """Fit the four internal quintile edges from Development values."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.full(4, np.nan)
    return np.quantile(array, [0.2, 0.4, 0.6, 0.8])


def apply_development_quintiles(
    values: Sequence[float] | pd.Series, edges: Sequence[float]
) -> np.ndarray:
    """Apply stored Development edges unchanged; invalid values receive zero."""

    array = np.asarray(values, dtype=float)
    edge_array = np.asarray(edges, dtype=float)
    result = np.zeros(len(array), dtype=np.int8)
    valid = np.isfinite(array)
    if len(edge_array) != 4 or not np.isfinite(edge_array).all():
        return result
    result[valid] = np.searchsorted(edge_array, array[valid], side="right") + 1
    return result


def _quintile_summary(
    group: pd.DataFrame,
    feature: str,
    target: str,
    tick_target: str | None,
    edges: np.ndarray,
) -> dict[str, float | int]:
    values = pd.to_numeric(group[feature], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    target_values = pd.to_numeric(group[target], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    bins = apply_development_quintiles(values, edges)
    valid = np.isfinite(target_values) & (bins > 0)
    means = np.array(
        [
            np.nanmean(target_values[valid & (bins == quintile)])
            if np.any(valid & (bins == quintile))
            else np.nan
            for quintile in range(1, 6)
        ]
    )
    finite_means = np.isfinite(means)
    monotonicity = (
        abs(float(spearmanr(np.arange(1, 6)[finite_means], means[finite_means]).statistic))
        if finite_means.sum() >= 3
        else np.nan
    )
    low = target_values[valid & (bins == 1)]
    high = target_values[valid & (bins == 5)]
    pooled_denominator = max(len(low) + len(high) - 2, 1)
    pooled = np.sqrt(
        (
            max(len(low) - 1, 0) * np.var(low, ddof=1)
            + max(len(high) - 1, 0) * np.var(high, ddof=1)
        )
        / pooled_denominator
    ) if len(low) > 1 and len(high) > 1 else np.nan
    effect = (
        float((np.mean(high) - np.mean(low)) / pooled)
        if np.isfinite(pooled) and pooled > 0
        else np.nan
    )
    tick_spread = np.nan
    if tick_target is not None and tick_target in group:
        ticks = pd.to_numeric(group[tick_target], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        tick_valid = np.isfinite(ticks) & (bins > 0)
        if np.any(tick_valid & (bins == 1)) and np.any(tick_valid & (bins == 5)):
            tick_spread = float(
                np.mean(ticks[tick_valid & (bins == 5)])
                - np.mean(ticks[tick_valid & (bins == 1)])
            )
    return {
        "observations": int(valid.sum()),
        "q1_mean": float(means[0]) if np.isfinite(means[0]) else np.nan,
        "q2_mean": float(means[1]) if np.isfinite(means[1]) else np.nan,
        "q3_mean": float(means[2]) if np.isfinite(means[2]) else np.nan,
        "q4_mean": float(means[3]) if np.isfinite(means[3]) else np.nan,
        "q5_mean": float(means[4]) if np.isfinite(means[4]) else np.nan,
        "absolute_monotonicity": monotonicity,
        "top_bottom_effect_size": effect,
        "top_bottom_tick_spread": tick_spread,
    }


def horizon_thinning_mask(
    frame: pd.DataFrame,
    horizon_minutes: int,
    *,
    date_column: str = "trade_date_ny",
    timestamp_column: str = "entry_timestamp_ny",
) -> np.ndarray:
    """Greedily retain non-overlapping observations within date and session."""

    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    if timestamp_column not in frame:
        for fallback in ("entry_timestamp_utc", "decision_timestamp_ny"):
            if fallback in frame:
                timestamp_column = fallback
                break
        else:
            raise KeyError("a timestamp column is required for horizon thinning")
    keep = np.zeros(len(frame), dtype=bool)
    working = frame.reset_index(drop=True)
    group_columns = [date_column]
    if "entry_session" in working:
        group_columns.append("entry_session")
    for _, group in working.groupby(group_columns, observed=True, sort=False):
        timestamps = pd.to_datetime(group[timestamp_column], errors="coerce", utc=True)
        timestamp_ns = timestamps.array.as_unit("ns").asi8
        valid = timestamps.notna().to_numpy()
        positions = group.index.to_numpy()[valid]
        valid_timestamp_ns = timestamp_ns[valid]
        order = np.argsort(valid_timestamp_ns, kind="stable")
        ordered_positions = positions[order]
        ordered_timestamp_ns = valid_timestamp_ns[order]
        last_ns: int | None = None
        minimum_gap_ns = horizon_minutes * 60_000_000_000
        for position, current_ns_value in zip(
            ordered_positions, ordered_timestamp_ns, strict=True
        ):
            current_ns = int(current_ns_value)
            if last_ns is None or current_ns - last_ns >= minimum_gap_ns:
                keep[position] = True
                last_ns = current_ns
    return keep


def _evaluate_thinned_cell(
    group: pd.DataFrame,
    feature: str,
    target: str,
    horizon: int,
    config: MlatEvaluationConfig,
    thinning_mask: np.ndarray | None = None,
) -> tuple[float, int, int]:
    mask = (
        horizon_thinning_mask(group, horizon)
        if thinning_mask is None
        else thinning_mask
    )
    if len(mask) != len(group):
        raise ValueError("cached thinning mask length does not match cell group")
    daily = calculate_daily_spearman(
        group.reset_index(drop=True).loc[mask],
        feature,
        target,
        minimum_observations=config.minimum_observations_per_day,
    )
    eligible = daily.loc[daily["eligible_date"]]
    return (
        float(eligible["daily_spearman_ic"].mean()) if len(eligible) else np.nan,
        int(len(eligible)),
        int(eligible["observations"].sum()),
    )


def evaluate_mlat_feature_cells(
    evaluation_frame: pd.DataFrame | MlatEvaluationFrameResult,
    registry: pd.DataFrame | Iterable[str],
    *,
    config: MlatEvaluationConfig | None = None,
    run_horizon_thinning: bool = True,
) -> MlatCellEvaluationResult:
    """Evaluate every feature/family/horizon/session/partition cell."""

    settings = config or MlatEvaluationConfig()
    frame = _as_frame(evaluation_frame).copy()
    _check_frame_scope(frame, config=settings)
    features = _features(registry)
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise KeyError(f"evaluation features missing: {missing}")
    daily_rows: list[pd.DataFrame] = []
    cell_rows: list[dict[str, object]] = []
    edge_rows = []
    quintile_rows = []
    year_rows = []
    thinning_rows = []
    thinning_cache: dict[tuple[str, int, str, str], np.ndarray] = {}

    for session in settings.sessions:
        session_frame = frame.loc[frame["entry_session"].astype(str).eq(session)]
        for feature in features:
            development_values = session_frame.loc[
                session_frame["research_partition"].astype(str).eq("Development"),
                feature,
            ]
            edges = fit_development_quintiles(
                pd.to_numeric(development_values, errors="coerce")
            )
            edge_rows.append(
                {
                    "feature_name": feature,
                    "entry_session": session,
                    "fit_partition": "Development",
                    "q20": edges[0],
                    "q40": edges[1],
                    "q60": edges[2],
                    "q80": edges[3],
                }
            )
            for family in MLAT_TARGET_FAMILIES:
                complete_column = f"{family}_complete_sample"
                family_frame = session_frame.loc[
                    _strict_boolean(session_frame[complete_column], complete_column)
                ]
                for horizon in settings.horizons:
                    target = _target_name(family, horizon)
                    tick_target = (
                        f"forward_return_{horizon}_ticks"
                        if family == "direction"
                        else None
                    )
                    for partition in settings.partitions:
                        group = family_frame.loc[
                            family_frame["research_partition"].astype(str).eq(partition)
                        ]
                        daily = calculate_daily_spearman(
                            group,
                            feature,
                            target,
                            minimum_observations=settings.minimum_observations_per_day,
                        )
                        daily = daily.assign(
                            feature_name=feature,
                            target_family=family,
                            horizon_minutes=horizon,
                            entry_session=session,
                            research_partition=partition,
                        )
                        daily_rows.append(daily)
                        eligible = daily.loc[daily["eligible_date"]]
                        mean_ic = (
                            float(eligible["daily_spearman_ic"].mean())
                            if len(eligible)
                            else np.nan
                        )
                        bootstrap = date_block_bootstrap(
                            eligible["daily_spearman_ic"],
                            iterations=settings.bootstrap_iterations,
                            seed=_cell_seed(
                                settings, feature, family, horizon, session, partition
                            ),
                            confidence_level=settings.confidence_level,
                        )
                        quintiles = _quintile_summary(
                            group, feature, target, tick_target, edges
                        )
                        quintile_rows.append(
                            {
                                "feature_name": feature,
                                "target_family": family,
                                "horizon_minutes": horizon,
                                "entry_session": session,
                                "research_partition": partition,
                                "edges_fit_partition": "Development",
                                **quintiles,
                            }
                        )
                        thinned_ic = np.nan
                        thinning_dates = 0
                        thinning_observations = 0
                        if run_horizon_thinning and len(group):
                            cache_key = (family, horizon, session, partition)
                            if cache_key not in thinning_cache:
                                thinning_cache[cache_key] = horizon_thinning_mask(
                                    group, horizon
                                )
                            (
                                thinned_ic,
                                thinning_dates,
                                thinning_observations,
                            ) = _evaluate_thinned_cell(
                                group,
                                feature,
                                target,
                                horizon,
                                settings,
                                thinning_cache[cache_key],
                            )
                        thinning_agreement = bool(
                            np.isfinite(mean_ic)
                            and np.isfinite(thinned_ic)
                            and np.sign(mean_ic) == np.sign(thinned_ic)
                        )
                        thinning_rows.append(
                            {
                                "feature_name": feature,
                                "target_family": family,
                                "horizon_minutes": horizon,
                                "entry_session": session,
                                "research_partition": partition,
                                "full_mean_daily_ic": mean_ic,
                                "thinned_mean_daily_ic": thinned_ic,
                                "thinned_dates": thinning_dates,
                                "thinned_observations": thinning_observations,
                                "sign_agreement": thinning_agreement,
                            }
                        )
                        cell_rows.append(
                            {
                                "feature_name": feature,
                                "target_family": family,
                                "horizon_minutes": horizon,
                                "entry_session": session,
                                "research_partition": partition,
                                "finite_observations": int(
                                    eligible["observations"].sum()
                                ),
                                "eligible_dates": int(len(eligible)),
                                "mean_daily_ic": mean_ic,
                                "bootstrap_ci_lower": bootstrap["lower"],
                                "bootstrap_ci_upper": bootstrap["upper"],
                                "p_value": bootstrap["p_value"],
                                **quintiles,
                                "thinned_mean_daily_ic": thinned_ic,
                                "thinning_sign_agreement": thinning_agreement,
                            }
                        )
                        if len(eligible):
                            year_source = eligible.copy()
                            year_source["year"] = pd.to_datetime(
                                year_source["trade_date_ny"]
                            ).dt.year
                            for year, year_group in year_source.groupby(
                                "year", observed=True, sort=True
                            ):
                                year_ic = float(year_group["daily_spearman_ic"].mean())
                                year_rows.append(
                                    {
                                        "feature_name": feature,
                                        "target_family": family,
                                        "horizon_minutes": horizon,
                                        "entry_session": session,
                                        "research_partition": partition,
                                        "year": int(year),
                                        "eligible_dates": int(len(year_group)),
                                        "mean_daily_ic": year_ic,
                                        "full_period_sign_agreement": bool(
                                            np.sign(year_ic) == np.sign(mean_ic)
                                        ),
                                    }
                                )

    cells = pd.DataFrame(cell_rows)
    cells["q_value"] = np.nan
    development = cells["research_partition"].eq("Development")
    for _, indices in cells.loc[development].groupby(
        ["target_family", "entry_session"], observed=True, sort=False
    ).groups.items():
        cells.loc[indices, "q_value"] = benjamini_hochberg(
            cells.loc[indices, "p_value"]
        )
    return MlatCellEvaluationResult(
        cell_results=cells,
        daily_ic=pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame(),
        quintile_edges=pd.DataFrame(edge_rows),
        quintile_results=pd.DataFrame(quintile_rows),
        year_stability=pd.DataFrame(year_rows),
        thinning_results=pd.DataFrame(thinning_rows),
    )


def _rank_residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average")
    if controls.size == 0:
        return ranked - ranked.mean()
    ranked_controls = np.column_stack(
        [rankdata(controls[:, index], method="average") for index in range(controls.shape[1])]
    )
    design = np.column_stack([np.ones(len(values)), ranked_controls])
    coefficients = np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - design @ coefficients


def _daily_partial_ic(
    group: pd.DataFrame,
    feature: str,
    target: str,
    controls: Sequence[str],
    minimum_observations: int,
) -> pd.DataFrame:
    rows = []
    for date, date_group in group.groupby("trade_date_ny", observed=True, sort=True):
        columns = [feature, target, *controls]
        values = date_group[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(values).all(axis=1)
        required = max(minimum_observations, len(controls) + 3)
        if valid.sum() >= required:
            complete = values[valid]
            feature_residual = _rank_residual(complete[:, 0], complete[:, 2:])
            target_residual = _rank_residual(complete[:, 1], complete[:, 2:])
            if np.std(feature_residual) <= 1.0e-12 or np.std(target_residual) <= 1.0e-12:
                correlation = np.nan
            else:
                correlation = float(np.corrcoef(feature_residual, target_residual)[0, 1])
                if not np.isfinite(correlation):
                    correlation = np.nan
        else:
            correlation = np.nan
        rows.append(
            {
                "trade_date_ny": date,
                "observations": int(valid.sum()),
                "daily_partial_ic": correlation,
                "eligible_date": bool(np.isfinite(correlation)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "trade_date_ny",
            "observations",
            "daily_partial_ic",
            "eligible_date",
        ],
    )


def _rank_design(controls: np.ndarray) -> np.ndarray:
    """Build the exact ranked-control design used by ``_rank_residual``."""

    ranked_controls = np.column_stack(
        [
            rankdata(controls[:, index], method="average")
            for index in range(controls.shape[1])
        ]
    )
    return np.column_stack([np.ones(len(controls)), ranked_controls])


def _rank_residual_from_design(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residualize ranks with a previously built, row-identical design."""

    ranked = rankdata(values, method="average")
    coefficients = np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - design @ coefficients


def _batched_daily_partial_ic(
    group: pd.DataFrame,
    features: Sequence[str],
    targets: Sequence[str],
    controls: Sequence[str],
    minimum_observations: int,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Evaluate feature-target partial ICs while reusing exact date designs."""

    result_columns = [
        "trade_date_ny",
        "observations",
        "daily_partial_ic",
        "eligible_date",
    ]
    unique_features = list(dict.fromkeys(features))
    unique_targets = list(dict.fromkeys(targets))
    pairs = [
        (feature, target)
        for feature in unique_features
        for target in unique_targets
    ]
    if group.empty:
        return {
            pair: pd.DataFrame(columns=result_columns)
            for pair in pairs
        }

    numeric_columns = [*unique_features, *unique_targets, *controls]
    values = (
        group[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    date_groups = group.groupby(
        "trade_date_ny", observed=True, sort=True
    ).indices
    dates = list(date_groups)
    feature_count = len(unique_features)
    target_count = len(unique_targets)
    target_start = feature_count
    control_start = feature_count + target_count
    observations = np.zeros(
        (len(dates), feature_count, target_count), dtype=np.int64
    )
    correlations = np.full(
        (len(dates), feature_count, target_count), np.nan, dtype=float
    )
    required = max(minimum_observations, len(controls) + 3)

    for date_index, positions in enumerate(date_groups.values()):
        date_values = values[positions]
        control_values = date_values[:, control_start:]
        controls_finite = np.isfinite(control_values).all(axis=1)
        design_cache: dict[bytes, np.ndarray] = {}
        feature_residual_cache: dict[tuple[int, bytes], np.ndarray] = {}
        target_residual_cache: dict[tuple[int, bytes], np.ndarray] = {}

        for target_index in range(target_count):
            target_values = date_values[:, target_start + target_index]
            target_finite = np.isfinite(target_values)
            for feature_index in range(feature_count):
                valid = (
                    controls_finite
                    & target_finite
                    & np.isfinite(date_values[:, feature_index])
                )
                count = int(valid.sum())
                observations[date_index, feature_index, target_index] = count
                if count < required:
                    continue

                support_key = valid.tobytes()
                design = design_cache.get(support_key)
                if design is None:
                    design = _rank_design(control_values[valid])
                    design_cache[support_key] = design

                feature_key = (feature_index, support_key)
                feature_residual = feature_residual_cache.get(feature_key)
                if feature_residual is None:
                    feature_residual = _rank_residual_from_design(
                        date_values[valid, feature_index], design
                    )
                    feature_residual_cache[feature_key] = feature_residual

                target_key = (target_index, support_key)
                target_residual = target_residual_cache.get(target_key)
                if target_residual is None:
                    target_residual = _rank_residual_from_design(
                        target_values[valid], design
                    )
                    target_residual_cache[target_key] = target_residual

                if (
                    np.std(feature_residual) <= 1.0e-12
                    or np.std(target_residual) <= 1.0e-12
                ):
                    continue
                correlation = float(
                    np.corrcoef(feature_residual, target_residual)[0, 1]
                )
                if np.isfinite(correlation):
                    correlations[
                        date_index, feature_index, target_index
                    ] = correlation

    results: dict[tuple[str, str], pd.DataFrame] = {}
    for feature_index, feature in enumerate(unique_features):
        for target_index, target in enumerate(unique_targets):
            correlation = correlations[:, feature_index, target_index]
            results[(feature, target)] = pd.DataFrame(
                {
                    "trade_date_ny": dates,
                    "observations": observations[
                        :, feature_index, target_index
                    ],
                    "daily_partial_ic": correlation,
                    "eligible_date": np.isfinite(correlation),
                },
                columns=result_columns,
            )
    return results


def evaluate_incremental_information(
    evaluation_frame: pd.DataFrame | MlatEvaluationFrameResult,
    registry: pd.DataFrame | Iterable[str],
    *,
    atr_anchor: str = "atr_20",
    frozen_features: pd.DataFrame | Iterable[str] | None = None,
    config: MlatEvaluationConfig | None = None,
    horizons: Sequence[int] = (60, 180),
) -> MlatIncrementalResult:
    """Evaluate daily partial rank IC beyond ATR and the frozen 15 features."""

    settings = config or MlatEvaluationConfig()
    frame = _as_frame(evaluation_frame).copy()
    _check_frame_scope(frame, config=settings)
    features = _features(registry)
    if isinstance(frozen_features, pd.DataFrame):
        frozen = frozen_features
        if "in_frozen_set" in frozen:
            frozen = frozen.loc[
                _strict_boolean(frozen["in_frozen_set"], "in_frozen_set")
            ]
        frozen_names = frozen["feature_name"].astype(str).tolist()
    elif frozen_features is None:
        raise ValueError(
            "frozen_features is required for incremental-information evaluation"
        )
    else:
        frozen_names = list(frozen_features)
    frozen_names = list(dict.fromkeys(frozen_names))
    if not frozen_names:
        raise ValueError("frozen_features must contain at least one control")
    if atr_anchor not in frame:
        raise KeyError(f"ATR anchor {atr_anchor!r} is absent from evaluation frame")
    missing_frozen = sorted(set(frozen_names).difference(frame.columns))
    if missing_frozen:
        raise KeyError(f"frozen controls missing: {missing_frozen}")
    control_sets = {
        "atr_20": [atr_anchor],
        "frozen_15": frozen_names,
    }
    daily_cache: dict[
        tuple[str, str, int, str, str, str], pd.DataFrame
    ] = {}
    unique_horizons = list(dict.fromkeys(horizons))
    for family in MLAT_TARGET_FAMILIES:
        complete_column = f"{family}_complete_sample"
        family_frame = frame.loc[
            _strict_boolean(frame[complete_column], complete_column)
        ]
        targets = [_target_name(family, horizon) for horizon in unique_horizons]
        for session in settings.sessions:
            session_mask = family_frame["entry_session"].astype(str).eq(session)
            for partition in settings.partitions:
                group = family_frame.loc[
                    session_mask
                    & family_frame["research_partition"]
                    .astype(str)
                    .eq(partition)
                ]
                for control_name, controls in control_sets.items():
                    batched = _batched_daily_partial_ic(
                        group,
                        features,
                        targets,
                        controls,
                        settings.minimum_observations_per_day,
                    )
                    for feature in features:
                        for horizon, target in zip(
                            unique_horizons, targets, strict=True
                        ):
                            daily_cache[
                                (
                                    feature,
                                    family,
                                    horizon,
                                    session,
                                    control_name,
                                    partition,
                                )
                            ] = batched[(feature, target)]

    daily_tables = []
    summary_rows = []
    for feature in features:
        for family in MLAT_TARGET_FAMILIES:
            for horizon in horizons:
                for session in settings.sessions:
                    for control_name, controls in control_sets.items():
                        for partition in settings.partitions:
                            daily = daily_cache[
                                (
                                    feature,
                                    family,
                                    horizon,
                                    session,
                                    control_name,
                                    partition,
                                )
                            ]
                            daily = daily.assign(
                                feature_name=feature,
                                target_family=family,
                                horizon_minutes=horizon,
                                entry_session=session,
                                research_partition=partition,
                                control_set=control_name,
                                control_features="|".join(controls),
                            )
                            daily_tables.append(daily)
                            eligible = daily.loc[daily["eligible_date"]]
                            summary_rows.append(
                                {
                                    "feature_name": feature,
                                    "target_family": family,
                                    "horizon_minutes": horizon,
                                    "entry_session": session,
                                    "research_partition": partition,
                                    "control_set": control_name,
                                    "eligible_dates": int(len(eligible)),
                                    "finite_observations": int(
                                        eligible["observations"].sum()
                                    ),
                                    "mean_daily_partial_ic": (
                                        float(eligible["daily_partial_ic"].mean())
                                        if len(eligible)
                                        else np.nan
                                    ),
                                }
                            )
    summary = pd.DataFrame(summary_rows)
    key = [
        "feature_name",
        "target_family",
        "horizon_minutes",
        "entry_session",
        "control_set",
    ]
    wide = summary.pivot(index=key, columns="research_partition")
    wide.columns = [f"{metric}_{partition.lower()}" for metric, partition in wide.columns]
    wide = wide.reset_index()
    development_ic = wide["mean_daily_partial_ic_development"]
    validation_ic = wide["mean_daily_partial_ic_validation"]
    threshold = np.where(
        wide["control_set"].eq("atr_20"),
        settings.atr_partial_ic_threshold,
        settings.frozen_partial_ic_threshold,
    )
    wide["development_threshold"] = threshold
    wide["development_sample_pass"] = (
        wide["eligible_dates_development"].ge(
            settings.minimum_development_dates
        )
        & wide["finite_observations_development"].ge(
            settings.minimum_development_observations
        )
    )
    wide["validation_sample_pass"] = (
        wide["eligible_dates_validation"].ge(settings.minimum_validation_dates)
        & wide["finite_observations_validation"].ge(
            settings.minimum_validation_observations
        )
    )
    wide["development_magnitude_pass"] = development_ic.abs() >= threshold
    wide["validation_sign_agreement"] = (
        np.sign(validation_ic) == np.sign(development_ic)
    ) & development_ic.notna() & validation_ic.notna()
    wide["validation_magnitude_retention"] = validation_ic.abs() / development_ic.abs()
    wide["validation_retention_pass"] = (
        wide["validation_magnitude_retention"] >= settings.validation_retention
    )
    wide["incremental_pass"] = (
        wide["development_sample_pass"]
        & wide["validation_sample_pass"]
        & wide["development_magnitude_pass"]
        & wide["validation_sign_agreement"]
        & wide["validation_retention_pass"]
    )
    return MlatIncrementalResult(
        daily_partial_ic=(
            pd.concat(daily_tables, ignore_index=True) if daily_tables else pd.DataFrame()
        ),
        summary=wide,
    )


def _validation_failures(
    validation: object | None, features: Sequence[str]
) -> Mapping[str, bool]:
    failures = {feature: False for feature in features}
    if validation is None:
        raise ValueError("validation evidence is required for verdict assignment")
    if not hasattr(validation, "ready"):
        raise TypeError("validation evidence must expose a ready property")
    if hasattr(validation, "ready") and not bool(validation.ready):
        failures = dict.fromkeys(features, True)
    diagnostics = getattr(validation, "feature_diagnostics", None)
    if isinstance(diagnostics, pd.DataFrame) and not diagnostics.empty:
        for row in diagnostics.itertuples(index=False):
            failures[str(row.feature_name)] = failures.get(
                str(row.feature_name), False
            ) or bool(
                getattr(row, "missing_rate", 0) > 0.05
                or not getattr(row, "dtype_pass", True)
                or getattr(row, "nonfinite_count", 0) > 0
                or not getattr(row, "range_pass", True)
                or getattr(row, "constant", False)
            )
    return failures


def assign_mlat_verdicts(
    evaluation: pd.DataFrame | MlatCellEvaluationResult,
    incremental: pd.DataFrame | MlatIncrementalResult | None = None,
    overlap_audit: pd.DataFrame | None = None,
    validation: object | None = None,
    *,
    registry: pd.DataFrame | None = None,
    config: MlatEvaluationConfig | None = None,
) -> pd.DataFrame:
    """Map preregistered gates to a notebook-ready verdict table.

    The returned table includes source traceability, compact Development and
    Validation summaries, stability/redundancy/incremental summaries, an
    economic interpretation, and ``final_decision`` (the same controlled label
    as ``verdict``).  A registry may override the built-in frozen-v1 source
    mapping with ``hypothesis_id``, ``book_chapter``/``source_chapter``, and
    ``source_pdf_page`` columns.
    """

    settings = config or MlatEvaluationConfig()
    if not isinstance(evaluation, MlatCellEvaluationResult):
        raise TypeError(
            "evaluation must be MlatCellEvaluationResult so stability evidence "
            "cannot be omitted"
        )
    if not isinstance(incremental, MlatIncrementalResult):
        raise TypeError(
            "incremental must be MlatIncrementalResult with complete control evidence"
        )
    if overlap_audit is None or overlap_audit.empty:
        raise ValueError("non-empty overlap_audit evidence is required")
    overlap_required = {
        "candidate_feature",
        "reference_scope",
        "reference_feature",
        "is_exact_duplicate",
        "absolute_correlation",
        "sufficient_overlap",
    }
    missing_overlap = sorted(overlap_required.difference(overlap_audit.columns))
    if missing_overlap:
        raise KeyError(f"overlap audit columns missing: {missing_overlap}")
    if validation is None:
        raise ValueError("validation evidence is required")
    cells = evaluation.cell_results.copy()
    year_table = evaluation.year_stability.copy()
    if year_table.empty:
        raise ValueError("year-stability evidence is required")
    required = {
        "feature_name",
        "target_family",
        "horizon_minutes",
        "entry_session",
        "research_partition",
        "mean_daily_ic",
    }
    missing = sorted(required.difference(cells.columns))
    if missing:
        raise KeyError(f"cell result columns missing: {missing}")
    incremental_table = incremental.summary.copy()
    incremental_required = {
        "feature_name",
        "target_family",
        "horizon_minutes",
        "entry_session",
        "control_set",
        "incremental_pass",
    }
    missing_incremental = sorted(
        incremental_required.difference(incremental_table.columns)
    )
    if missing_incremental:
        raise KeyError(
            f"incremental evidence columns missing: {missing_incremental}"
        )
    if "thinned_mean_daily_ic" not in cells:
        raise KeyError(
            "cell result columns missing: ['thinned_mean_daily_ic']"
        )
    required_incremental_horizons = sorted(
        pd.to_numeric(
            incremental_table["horizon_minutes"], errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not required_incremental_horizons:
        raise ValueError("incremental evidence contains no evaluable horizons")
    required_thinning_cells = cells.loc[
        pd.to_numeric(cells["horizon_minutes"], errors="coerce").isin(
            required_incremental_horizons
        )
    ]
    finite_required_thinning_cells = int(
        np.isfinite(
            pd.to_numeric(
                required_thinning_cells["thinned_mean_daily_ic"],
                errors="coerce",
            )
        ).sum()
    )
    required_development_thinning_cells = required_thinning_cells.loc[
        required_thinning_cells["research_partition"]
        .astype(str)
        .eq("Development")
    ]
    finite_required_development_thinning_cells = int(
        np.isfinite(
            pd.to_numeric(
                required_development_thinning_cells[
                    "thinned_mean_daily_ic"
                ],
                errors="coerce",
            )
        ).sum()
    )
    required_thinning_cell_count = len(required_thinning_cells)
    authorization_gate_open = bool(
        required_thinning_cell_count > 0
        and finite_required_development_thinning_cells > 0
    )
    horizon_label = "/".join(
        str(horizon) for horizon in required_incremental_horizons
    )
    if authorization_gate_open:
        authorization_gate_reason = (
            "frozen v1 horizon-thinning evidence is evaluable: "
            f"{finite_required_thinning_cells}/{required_thinning_cell_count} "
            f"required {horizon_label}-minute "
            "feature-family-session-partition cells have finite thinned "
            "daily-IC evidence, including "
            f"{finite_required_development_thinning_cells} Development cells; "
            "cell-level authorization remains subject to every preregistered gate"
        )
    else:
        authorization_gate_reason = (
            "frozen v1 horizon-thinning gate is structurally non-evaluable: "
            f"{finite_required_thinning_cells}/{required_thinning_cell_count} "
            f"required {horizon_label}-minute "
            "feature-family-session-partition cells have finite thinned "
            "daily-IC evidence under the >="
            f"{settings.minimum_observations_per_day} "
            "observations-per-New-York-date rule, including "
            f"{finite_required_development_thinning_cells} Development cells; "
            "advancement is not authorized"
        )
    features = cells["feature_name"].drop_duplicates().astype(str).tolist()
    existing_overlap_features = set(
        overlap_audit.loc[
            overlap_audit["reference_scope"].eq("existing"),
            "candidate_feature",
        ].astype(str)
    )
    missing_overlap_features = sorted(set(features).difference(existing_overlap_features))
    if missing_overlap_features:
        raise ValueError(
            "existing-feature overlap evidence missing for: "
            f"{missing_overlap_features}"
        )
    validation_failed = _validation_failures(validation, features)
    metadata_by_feature: dict[str, dict[str, object]] = {}
    if registry is not None:
        if "feature_name" not in registry:
            raise KeyError("verdict registry must contain feature_name")
        for record in registry.to_dict("records"):
            metadata_by_feature[str(record["feature_name"])] = record
    rows = []
    for feature in features:
        feature_cells = cells.loc[cells["feature_name"].astype(str).eq(feature)]
        development = feature_cells.loc[
            feature_cells["research_partition"].astype(str).eq("Development")
        ].copy()
        confirmation = feature_cells.loc[
            feature_cells["research_partition"].astype(str).eq("Validation")
        ].copy()
        keys = ["target_family", "horizon_minutes", "entry_session"]
        paired = development.merge(
            confirmation,
            on=keys,
            how="left",
            suffixes=("_development", "_validation"),
        )
        dev_q = paired.get(
            "q_value_development", pd.Series(np.nan, index=paired.index)
        )
        dev_dates = paired.get(
            "eligible_dates_development", pd.Series(0, index=paired.index)
        )
        val_dates = paired.get(
            "eligible_dates_validation", pd.Series(0, index=paired.index)
        )
        dev_obs = paired.get(
            "finite_observations_development", pd.Series(0, index=paired.index)
        )
        val_obs = paired.get(
            "finite_observations_validation", pd.Series(0, index=paired.index)
        )
        dev_ic = paired["mean_daily_ic_development"]
        val_ic = paired["mean_daily_ic_validation"]
        paired["development_screen_pass"] = (
            dev_q.le(settings.development_q_threshold)
            & dev_dates.ge(settings.minimum_development_dates)
            & dev_obs.ge(settings.minimum_development_observations)
        )
        paired["validation_sign_agreement"] = (
            np.sign(dev_ic) == np.sign(val_ic)
        ) & dev_ic.notna() & val_ic.notna()
        paired["validation_magnitude_retention"] = val_ic.abs() / dev_ic.abs()
        paired["validation_confirmation_pass"] = (
            paired["validation_sign_agreement"]
            & paired["validation_magnitude_retention"].ge(
                settings.validation_retention
            )
            & val_dates.ge(settings.minimum_validation_dates)
            & val_obs.ge(settings.minimum_validation_observations)
        )
        monotonicity = paired.get(
            "absolute_monotonicity_development", pd.Series(np.nan, index=paired.index)
        )
        paired["monotonicity_pass"] = monotonicity.ge(
            settings.monotonicity_threshold
        )
        thinning = paired.get(
            "thinning_sign_agreement_development",
            pd.Series(False, index=paired.index),
        )
        paired["thinning_pass"] = thinning.fillna(False).astype(bool)
        if year_table.empty:
            paired["year_stability_pass"] = False
        else:
            feature_years = year_table.loc[
                year_table["feature_name"].astype(str).eq(feature)
            ]
            year_pass = (
                feature_years.groupby(
                    [*keys, "research_partition"], observed=True
                )["full_period_sign_agreement"]
                .all()
                .unstack("research_partition")
            )
            year_pass["year_stability_pass"] = year_pass.all(axis=1)
            paired = paired.merge(
                year_pass["year_stability_pass"].reset_index(),
                on=keys,
                how="left",
            )
            paired["year_stability_pass"] = paired[
                "year_stability_pass"
            ].fillna(False)
        session_evidence = (
            paired["mean_daily_ic_development"].notna()
            & paired["mean_daily_ic_validation"].notna()
            & paired["validation_sign_agreement"]
            & dev_dates.ge(settings.minimum_development_dates)
            & val_dates.ge(settings.minimum_validation_dates)
            & dev_obs.ge(settings.minimum_development_observations)
            & val_obs.ge(settings.minimum_validation_observations)
        )
        session_summary = (
            paired.loc[session_evidence]
            .assign(_sign=np.sign(paired.loc[session_evidence, "mean_daily_ic_development"]))
            .groupby(["target_family", "horizon_minutes"], observed=True)
            .agg(
                eligible_sessions=("entry_session", "nunique"),
                development_signs=("_sign", "nunique"),
            )
        )
        required_session_count = len(settings.sessions)
        paired["session_stability_pass"] = [
            bool(
                (family, horizon) in session_summary.index
                and session_summary.loc[
                    (family, horizon), "eligible_sessions"
                ]
                == required_session_count
                and session_summary.loc[
                    (family, horizon), "development_signs"
                ]
                == 1
            )
            for family, horizon in zip(
                paired["target_family"], paired["horizon_minutes"], strict=True
            )
        ]
        tick_dev = paired.get(
            "top_bottom_tick_spread_development",
            pd.Series(np.nan, index=paired.index),
        )
        tick_val = paired.get(
            "top_bottom_tick_spread_validation", pd.Series(np.nan, index=paired.index)
        )
        paired["economic_pass"] = np.where(
            paired["target_family"].eq("direction"),
            tick_dev.abs().ge(settings.direction_tick_spread)
            & tick_val.abs().ge(settings.direction_tick_spread),
            True,
        )

        incremental_feature = incremental_table.loc[
            incremental_table["feature_name"].astype(str).eq(feature)
        ]
        incremental_rows = []
        required_controls = set(REQUIRED_INCREMENTAL_CONTROL_SETS)
        for key_values, group in incremental_feature.groupby(
            keys, observed=True, sort=False
        ):
            key_tuple = (
                key_values if isinstance(key_values, tuple) else (key_values,)
            )
            controls = group["control_set"].astype(str)
            required_rows = group.loc[controls.isin(required_controls)]
            complete_controls = (
                set(controls) == required_controls
                and not controls.duplicated().any()
            )
            incremental_rows.append(
                {
                    **dict(zip(keys, key_tuple, strict=True)),
                    "incremental_pass": bool(
                        complete_controls
                        and required_rows["incremental_pass"]
                        .fillna(False)
                        .astype(bool)
                        .all()
                    ),
                }
            )
        incremental_grouped = pd.DataFrame(
            incremental_rows, columns=[*keys, "incremental_pass"]
        )
        paired = paired.merge(incremental_grouped, on=keys, how="left")
        incremental_pass = paired["incremental_pass"].fillna(False)
        paired["incremental_gate_pass"] = incremental_pass
        paired["all_statistical_gates"] = (
            paired["development_screen_pass"]
            & paired["validation_confirmation_pass"]
            & paired["monotonicity_pass"]
            & paired["thinning_pass"]
            & paired["year_stability_pass"]
            & paired["session_stability_pass"]
        )
        paired["all_advancement_gates"] = (
            paired["all_statistical_gates"]
            & paired["economic_pass"]
            & paired["incremental_gate_pass"]
        )

        overlap_rows = (
            overlap_audit.loc[
                overlap_audit["candidate_feature"].astype(str).eq(feature)
                & overlap_audit["reference_scope"].eq("existing")
            ]
            if overlap_audit is not None and not overlap_audit.empty
            else pd.DataFrame()
        )
        redundant = bool(
            not overlap_rows.empty
            and (
                overlap_rows["sufficient_overlap"].fillna(False)
                & (
                    overlap_rows["is_exact_duplicate"].fillna(False)
                    | overlap_rows["absolute_correlation"].ge(
                        settings.overlap_threshold
                    )
                )
            ).any()
        )
        reference = (
            str(
                overlap_rows.sort_values(
                    "absolute_correlation", ascending=False, na_position="last"
                ).iloc[0]["reference_feature"]
            )
            if redundant
            else pd.NA
        )
        advances = paired.loc[paired["all_advancement_gates"]]
        directional_trivial = bool(
            (
                paired["target_family"].eq("direction")
                & paired["all_statistical_gates"]
                & ~paired["economic_pass"]
            ).any()
        )
        if validation_failed.get(feature, False):
            verdict = "FAILED_VALIDATION"
            reason = "feature failed schema, missingness, finite-value, range, or constant gate"
        elif redundant:
            verdict = "REDUNDANT_WITH_EXISTING"
            reason = f"Development overlap veto against {reference}"
        elif advances["target_family"].eq("direction").any():
            verdict = "ADVANCE_DIRECTIONAL"
            reason = "directional cell passed Development, Validation, economic, and incremental gates"
        elif advances["target_family"].eq("expansion").any():
            verdict = "ADVANCE_EXPANSION"
            reason = "expansion cell passed Development, Validation, and incremental gates"
        elif advances["target_family"].isin(["volatility", "path_risk"]).any():
            verdict = "ADVANCE_RISK_STATE"
            reason = "volatility/path-risk cell passed Development, Validation, and incremental gates"
        elif directional_trivial:
            verdict = "ECONOMICALLY_TRIVIAL"
            reason = "directional evidence did not reach the two-GC-tick spread gate"
        elif paired["development_screen_pass"].any():
            verdict = "WEAK_OR_UNSTABLE"
            reason = "Development evidence failed confirmation, stability, or incremental gates"
        else:
            verdict = "NO_EVIDENCE"
            reason = "no Development cell passed the preregistered screen"
        pre_authorization_verdict = verdict
        if (
            not authorization_gate_open
            and verdict
            not in {"FAILED_VALIDATION", "REDUNDANT_WITH_EXISTING"}
        ):
            verdict = "RESEARCH_ONLY"
            reason = authorization_gate_reason
        best = (
            paired.assign(_strength=paired["mean_daily_ic_development"].abs())
            .sort_values("_strength", ascending=False, na_position="last")
            .head(1)
        )
        primary = best.iloc[0] if len(best) else None
        source = MLAT_SOURCE_TRACEABILITY.get(feature, (pd.NA, pd.NA, pd.NA))
        metadata = metadata_by_feature.get(feature, {})
        hypothesis_id = metadata.get("hypothesis_id", source[0])
        book_chapter = metadata.get(
            "book_chapter", metadata.get("source_chapter", source[1])
        )
        source_pdf_page = metadata.get("source_pdf_page", source[2])
        development_screened = paired["development_screen_pass"].fillna(False)
        screened_confirmed = (
            development_screened
            & paired["validation_confirmation_pass"].fillna(False)
        )
        development_result = (
            f"{int(development_screened.sum())}/{len(paired)} cells pass "
            "the Development screen; "
            f"max |Development daily IC|={development['mean_daily_ic'].abs().max():.4f}"
            if len(development) and development["mean_daily_ic"].notna().any()
            else "no eligible Development daily-IC evidence"
        )
        screened_count = int(development_screened.sum())
        confirmed_count = int(screened_confirmed.sum())
        confirmed_validation_ic = paired.loc[
            screened_confirmed, "mean_daily_ic_validation"
        ]
        if screened_count == 0:
            validation_result = (
                "0 Development cells passed the screen; "
                "Validation confirmation not applicable"
            )
        elif confirmed_validation_ic.notna().any():
            validation_result = (
                f"{confirmed_count}/{screened_count} Development-screened cells "
                "confirmed in Validation; "
                "max |confirmed Validation daily IC|="
                f"{confirmed_validation_ic.abs().max():.4f}"
            )
        else:
            validation_result = (
                f"{confirmed_count}/{screened_count} Development-screened cells "
                "confirmed in Validation"
            )
        incremental_feature = (
            incremental_table.loc[
                incremental_table["feature_name"].astype(str).eq(feature)
            ]
            if not incremental_table.empty
            else pd.DataFrame()
        )
        incremental_result = (
            f"{int(incremental_feature['incremental_pass'].fillna(False).sum())}/"
            f"{len(incremental_feature)} 60/180 horizon-control tests pass"
            if len(incremental_feature)
            else "not evaluated or no eligible partial-IC evidence"
        )
        if redundant:
            redundancy_result = f"VETO: overlaps existing feature {reference}"
        else:
            redundancy_result = "PASS: no exact or >=0.995 existing-feature overlap"
        if verdict == "ADVANCE_DIRECTIONAL":
            economic_interpretation = (
                "Directional ordering exceeds the two-GC-tick Development and "
                "Validation spread gate; this is not a PnL claim."
            )
        elif verdict == "ADVANCE_EXPANSION":
            economic_interpretation = (
                "Feature orders future ATR-normalized range; execution economics "
                "remain unevaluated."
            )
        elif verdict == "ADVANCE_RISK_STATE":
            economic_interpretation = (
                "Feature orders future volatility/path risk and is a risk-state "
                "candidate, not a directional trading rule."
            )
        elif verdict == "ECONOMICALLY_TRIVIAL":
            economic_interpretation = (
                "Statistical ordering does not reach the two-GC-tick directional gate."
            )
        else:
            economic_interpretation = (
                "No incremental, stable, nonredundant economic interpretation is supported."
            )
        any_session_stability_pass = bool(
            paired["session_stability_pass"].any()
        )
        all_session_stability_pass = bool(
            paired["session_stability_pass"].all()
        )
        if all_session_stability_pass:
            session_stability = (
                "PASS_ALL: every family/horizon meets the cross-session gate"
            )
        elif any_session_stability_pass:
            session_stability = (
                "MIXED: at least one family/horizon meets the cross-session "
                "gate and at least one does not"
            )
        else:
            session_stability = (
                "FAIL: no family/horizon has eligible, sign-consistent "
                "evidence across both sessions"
            )
        any_year_stability_pass = bool(paired["year_stability_pass"].any())
        all_year_stability_pass = bool(paired["year_stability_pass"].all())
        if all_year_stability_pass:
            year_stability = (
                "PASS_ALL: every evaluated cell has stable annual IC signs"
            )
        elif any_year_stability_pass:
            year_stability = (
                "MIXED: at least one evaluated cell has stable annual IC "
                "signs and at least one does not"
            )
        else:
            year_stability = (
                "FAIL: no evaluated cell has stable annual IC signs"
            )
        rows.append(
            {
                "feature_name": feature,
                "hypothesis_id": hypothesis_id,
                "book_chapter": book_chapter,
                "source_pdf_page": source_pdf_page,
                "validation_passed": not validation_failed.get(feature, False),
                "overlap_veto": redundant,
                "overlap_reference": reference,
                "any_development_screen_pass": bool(
                    paired["development_screen_pass"].any()
                ),
                "any_validation_confirmation_pass": bool(
                    paired["validation_confirmation_pass"].any()
                ),
                "any_screened_validation_confirmation_pass": bool(
                    screened_confirmed.any()
                ),
                "any_monotonicity_pass": bool(
                    paired["monotonicity_pass"].any()
                ),
                "any_thinning_pass": bool(paired["thinning_pass"].any()),
                "any_year_stability_pass": any_year_stability_pass,
                "any_session_stability_pass": any_session_stability_pass,
                "any_economic_gate_pass": bool(paired["economic_pass"].any()),
                "any_incremental_gate_pass": bool(
                    paired["incremental_gate_pass"].any()
                ),
                "directional_cells_passed": int(
                    (
                        paired["target_family"].eq("direction")
                        & paired["all_advancement_gates"]
                    ).sum()
                ),
                "expansion_cells_passed": int(
                    (
                        paired["target_family"].eq("expansion")
                        & paired["all_advancement_gates"]
                    ).sum()
                ),
                "risk_state_cells_passed": int(
                    (
                        paired["target_family"].isin(["volatility", "path_risk"])
                        & paired["all_advancement_gates"]
                    ).sum()
                ),
                "primary_target_family": (
                    primary["target_family"] if primary is not None else pd.NA
                ),
                "primary_horizon_minutes": (
                    primary["horizon_minutes"] if primary is not None else pd.NA
                ),
                "primary_session": (
                    primary["entry_session"] if primary is not None else pd.NA
                ),
                "development_result": development_result,
                "validation_result": validation_result,
                "session_stability": session_stability,
                "year_stability": year_stability,
                "redundancy_result": redundancy_result,
                "incremental_information_result": incremental_result,
                "economic_interpretation": economic_interpretation,
                "authorization_gate_open": authorization_gate_open,
                "authorization_gate_reason": authorization_gate_reason,
                "pre_authorization_verdict": pre_authorization_verdict,
                "verdict": verdict,
                "final_decision": verdict,
                "verdict_reason": reason,
            }
        )
    verdicts = pd.DataFrame(rows)
    if not verdicts.empty and not set(verdicts["verdict"]).issubset(VERDICT_LABELS):
        raise AssertionError("unrecognized MLAT verdict")
    return verdicts
