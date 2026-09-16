"""Governed POI-first and general feature-combination research.

The implementation is additive. It consumes the frozen Project One artifacts
through 2024, rejects any materialized Final-test row, and leaves every prior
research engine and verdict untouched.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

CONTRACT_RELATIVE_PATH = Path("project_docs/poi_feature_combination_research_contract_v1.md")
EXPECTED_CONTRACT_SHA256 = "03dd2105063e30a71fb328f8573b2a0f165c80d4ec129ab05fa67cd378e02101"
FINAL_TEST_START = pd.Timestamp("2025-01-01")
FINAL_TEST_START_UTC = pd.Timestamp("2025-01-01", tz="UTC")
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_RESTART_PROBABILITY = 0.20
BOOTSTRAP_SEED = 20260826
RIDGE_ALPHA = 10.0
BASE_COST_TICKS = 2.6
PESSIMISTIC_COST_TICKS = 4.6


POI_CONTEXT_FEATURES: tuple[str, ...] = (
    "feat_poi_width_atr",
    "feat_confirmation_directional_close_location",
    "feat_displacement_efficiency",
    "feat_displacement_speed_to_approach_speed_ratio",
    "feat_approach_15m_directional_efficiency",
    "feat_approach_15m_range_compression_ratio",
    "feat_approach_15m_candle_overlap_ratio",
    "feat_approach_15m_relative_volume",
    "feat_touch_penetration_fraction",
    "feat_touch_rejection_wick_to_body_ratio",
    "feat_distance_from_vwap_atr",
    "feat_short_to_long_volatility_ratio",
    "feat_previous_touch_count",
    "feat_poi_age_minutes",
)

STAT15_FEATURES: tuple[str, ...] = (
    "atr_20",
    "minute_from_execution_window_open",
    "minutes_to_1530_forced_exit",
    "atr_ratio_20_60",
    "volume_acceleration_5_20",
    "efficiency_ratio_60",
    "return_sign_change_rate_30",
    "ols_r_squared_30",
    "atr_ratio_5_20",
    "efficiency_ratio_30",
    "relative_volume_20",
    "vwap_elasticity_30_exp",
    "current_range_over_atr",
    "choppiness_14",
    "efficiency_ratio_15",
)

FES_FEATURES_BY_SESSION: Mapping[str, tuple[str, ...]] = {
    "London": ("return_acf_energy_60", "range_volume_spearman_30"),
    "New York": (
        "lagged_volume_return_spearman_30",
        "volume_profile_slope_30",
    ),
}

ALL_FES_SOURCE_FEATURES: tuple[str, ...] = (
    "return_acf_energy_60",
    "range_volume_spearman_30",
    "lagged_volume_return_spearman_30",
    "volume_profile_slope_30",
)

INTERACTION_FEATURES: tuple[str, ...] = (
    "int_poi_direction_x_lagged_volume",
    "int_poi_width_x_atr_ratio",
    "int_displacement_x_efficiency",
    "int_approach_overlap_x_sign_change",
    "int_touch_penetration_x_current_range",
    "int_approach_compression_x_acf",
    "int_retest_volume_x_range_volume",
    "int_poi_vwap_x_trend",
)

POI_MODEL_ORDER: tuple[str, ...] = (
    "POI1_CONTEXT",
    "POI2_CONTEXT_STAT15",
    "POI3_CONTEXT_FES4",
    "POI4_CONTEXT_INTERACTIONS",
    "POI5_ALL",
)

GENERAL_MODEL_ORDER: tuple[str, ...] = (
    "GEN1_STAT15",
    "GEN2_FES4",
    "GEN3_STAT15_FES4",
)


@dataclass(frozen=True)
class DateFold:
    """One expanding whole-date fold."""

    fold_id: int
    train_dates: tuple[pd.Timestamp, ...]
    embargo_dates: tuple[pd.Timestamp, ...]
    assessment_dates: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class Preprocessor:
    """Training-only median/IQR preprocessing state."""

    feature_names: tuple[str, ...]
    medians: np.ndarray
    scales: np.ndarray


@dataclass
class CombinationResearchResult:
    """In-memory result returned to the executed notebook."""

    population_summary: pd.DataFrame
    fold_summary: pd.DataFrame
    model_metrics: pd.DataFrame
    daily_ic: pd.DataFrame
    coefficients: pd.DataFrame
    selection_verdicts: pd.DataFrame
    policy_thresholds: pd.DataFrame
    policy_metrics: pd.DataFrame
    policy_trades: pd.DataFrame
    access_audit: pd.DataFrame
    headline: dict[str, Any]
    output_dir: Path


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    """Serialize a payload deterministically for governance hashes."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def canonical_sha256(payload: Any) -> str:
    """Hash a JSON-serializable payload deterministically."""

    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _normalized_dates(values: pd.Series | Iterable[Any]) -> pd.Series:
    series = pd.Series(values, copy=False)
    return pd.to_datetime(series, errors="raise").dt.tz_localize(None).dt.normalize()


def assert_no_final_test(
    frame: pd.DataFrame,
    *,
    date_column: str = "trade_date_ny",
    partition_columns: Sequence[str] = ("research_partition", "partition"),
    timestamp_columns: Sequence[str] = (
        "decision_timestamp_utc",
        "entry_timestamp_utc",
        "exit_timestamp_utc_60",
    ),
    context: str = "frame",
) -> None:
    """Fail closed when a materialized frame contains a 2025+ or Final-test row."""

    if date_column in frame:
        dates = _normalized_dates(frame[date_column])
        if dates.ge(FINAL_TEST_START).any():
            raise PermissionError(f"{context} materialized a row dated 2025 onward")
    for column in partition_columns:
        if column not in frame:
            continue
        values = (
            frame[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace("-", "_", regex=False)
            .str.replace(" ", "_", regex=False)
        )
        if values.str.contains("final", na=False).any():
            raise PermissionError(f"{context} materialized a Final-test partition")
        allowed = {"development", "validation", "retrospective_validation"}
        unknown = sorted(set(values.dropna()) - allowed)
        if unknown:
            raise PermissionError(f"{context} materialized unknown partition labels: {unknown}")
    for column in timestamp_columns:
        if column not in frame:
            continue
        timestamps = pd.to_datetime(frame[column], errors="raise", utc=True)
        if timestamps.ge(FINAL_TEST_START_UTC).any():
            raise PermissionError(f"{context} materialized a post-2024 timestamp in {column}")


def build_exact_date_folds(
    dates: Iterable[Any],
    *,
    minimum_train_dates: int = 126,
    assessment_dates: int = 42,
    embargo_dates: int = 1,
    minimum_assessment_dates: int = 21,
) -> list[DateFold]:
    """Build the contract's deterministic expanding whole-date folds."""

    normalized = _normalized_dates(dates)
    unique = tuple(sorted(normalized.drop_duplicates().tolist()))
    if minimum_train_dates <= 0 or assessment_dates <= 0 or embargo_dates < 0:
        raise ValueError("fold sizes must be positive and embargo nonnegative")
    if minimum_assessment_dates <= 0 or minimum_assessment_dates > assessment_dates:
        raise ValueError("minimum assessment size is invalid")

    folds: list[DateFold] = []
    train_end = minimum_train_dates
    fold_id = 1
    while train_end < len(unique):
        assessment_start = train_end + embargo_dates
        assessment_end = min(assessment_start + assessment_dates, len(unique))
        if assessment_end - assessment_start < minimum_assessment_dates:
            break
        folds.append(
            DateFold(
                fold_id=fold_id,
                train_dates=unique[:train_end],
                embargo_dates=unique[train_end:assessment_start],
                assessment_dates=unique[assessment_start:assessment_end],
            )
        )
        train_end += assessment_dates
        fold_id += 1
    return folds


def folds_to_frame(folds: Sequence[DateFold], *, branch: str, session: str) -> pd.DataFrame:
    """Return an auditable one-row-per-fold summary."""

    records = []
    for fold in folds:
        records.append(
            {
                "branch": branch,
                "session": session,
                "fold_id": fold.fold_id,
                "train_date_count": len(fold.train_dates),
                "train_start": fold.train_dates[0] if fold.train_dates else pd.NaT,
                "train_end": fold.train_dates[-1] if fold.train_dates else pd.NaT,
                "embargo_date_count": len(fold.embargo_dates),
                "embargo_dates": "|".join(
                    value.strftime("%Y-%m-%d") for value in fold.embargo_dates
                ),
                "assessment_date_count": len(fold.assessment_dates),
                "assessment_start": (fold.assessment_dates[0] if fold.assessment_dates else pd.NaT),
                "assessment_end": (fold.assessment_dates[-1] if fold.assessment_dates else pd.NaT),
            }
        )
    return pd.DataFrame.from_records(records)


def fold_manifest_payload(
    folds_by_key: Mapping[tuple[str, str], Sequence[DateFold]],
) -> dict[str, Any]:
    """Build the exact pre-outcome fold payload saved by Stage 1."""

    groups: dict[str, Any] = {}
    for (branch, session), folds in sorted(folds_by_key.items()):
        key = f"{branch}|{session}"
        groups[key] = [
            {
                "fold_id": fold.fold_id,
                "train_dates": [value.strftime("%Y-%m-%d") for value in fold.train_dates],
                "embargo_dates": [value.strftime("%Y-%m-%d") for value in fold.embargo_dates],
                "assessment_dates": [value.strftime("%Y-%m-%d") for value in fold.assessment_dates],
            }
            for fold in folds
        ]
    payload = {"groups": groups}
    payload["sha256"] = canonical_sha256(payload)
    return payload


def deduplicate_poi_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen outcome-free one-event-per-decision-bar rule."""

    required = {
        "decision_bar_id",
        "true_retest_id",
        "feat_first_touch",
        "feat_15bar_structural_validation",
        "feat_poi_age_minutes",
        "feat_poi_width_atr",
    }
    missing = required - set(frame)
    if missing:
        raise KeyError(f"POI deduplication is missing columns: {sorted(missing)}")
    work = frame.copy()
    if work["decision_bar_id"].isna().any() or work["true_retest_id"].isna().any():
        raise ValueError("POI deduplication identifiers must be nonmissing")
    work["_first_touch_sort"] = work["feat_first_touch"].fillna(False).astype(bool)
    work["_structural_sort"] = work["feat_15bar_structural_validation"].fillna(False).astype(bool)
    age = pd.to_numeric(work["feat_poi_age_minutes"], errors="coerce")
    width = pd.to_numeric(work["feat_poi_width_atr"], errors="coerce")
    work["_age_sort"] = age.where(np.isfinite(age), np.inf)
    work["_width_sort"] = width.where(np.isfinite(width), np.inf)
    tie_columns = [
        "decision_bar_id",
        "_first_touch_sort",
        "_structural_sort",
        "_age_sort",
        "_width_sort",
        "true_retest_id",
    ]
    if work.duplicated(tie_columns, keep=False).any():
        raise ValueError("POI deduplication contains ambiguous duplicate identities")
    ordered = work.sort_values(
        [
            "decision_bar_id",
            "_first_touch_sort",
            "_structural_sort",
            "_age_sort",
            "_width_sort",
            "true_retest_id",
        ],
        ascending=[True, False, False, True, True, True],
        kind="mergesort",
    )
    out = ordered.drop_duplicates("decision_bar_id", keep="first").drop(
        columns=["_first_touch_sort", "_structural_sort", "_age_sort", "_width_sort"]
    )
    if out["decision_bar_id"].duplicated().any():
        raise AssertionError("POI decision bars remain duplicated")
    return out.sort_values("decision_bar_id", kind="mergesort").reset_index(drop=True)


def add_poi_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the eight exact predeclared POI-coordinate interactions."""

    required = {
        "poi_direction_sign",
        "lagged_volume_return_spearman_30",
        "feat_poi_width_atr",
        "atr_ratio_20_60",
        "feat_displacement_efficiency",
        "efficiency_ratio_30",
        "feat_approach_15m_candle_overlap_ratio",
        "return_sign_change_rate_30",
        "feat_touch_penetration_fraction",
        "current_range_over_atr",
        "feat_approach_15m_range_compression_ratio",
        "return_acf_energy_60",
        "feat_approach_15m_relative_volume",
        "range_volume_spearman_30",
        "feat_distance_from_vwap_atr",
        "normalized_ols_slope_30",
    }
    missing = required - set(frame)
    if missing:
        raise KeyError(f"interaction construction is missing columns: {sorted(missing)}")
    out = frame.copy()
    values = {
        column: pd.to_numeric(out[column], errors="raise").astype("float64") for column in required
    }
    out["int_poi_direction_x_lagged_volume"] = (
        values["poi_direction_sign"] * values["lagged_volume_return_spearman_30"]
    )
    out["int_poi_width_x_atr_ratio"] = values["feat_poi_width_atr"] * values["atr_ratio_20_60"]
    out["int_displacement_x_efficiency"] = (
        values["feat_displacement_efficiency"] * values["efficiency_ratio_30"]
    )
    out["int_approach_overlap_x_sign_change"] = (
        values["feat_approach_15m_candle_overlap_ratio"] * values["return_sign_change_rate_30"]
    )
    out["int_touch_penetration_x_current_range"] = (
        values["feat_touch_penetration_fraction"] * values["current_range_over_atr"]
    )
    out["int_approach_compression_x_acf"] = (
        values["feat_approach_15m_range_compression_ratio"] * values["return_acf_energy_60"]
    )
    out["int_retest_volume_x_range_volume"] = (
        values["feat_approach_15m_relative_volume"] * values["range_volume_spearman_30"]
    )
    out["int_poi_vwap_x_trend"] = (
        values["poi_direction_sign"]
        * values["feat_distance_from_vwap_atr"]
        * values["normalized_ols_slope_30"]
    )
    return out


def model_features(branch: str, model_id: str, session: str) -> tuple[str, ...]:
    """Resolve the frozen model inputs for a branch/model/session cell."""

    fes = FES_FEATURES_BY_SESSION.get(session)
    if fes is None:
        raise KeyError(f"unknown session: {session}")
    if branch == "POI":
        mapping: Mapping[str, tuple[str, ...]] = {
            "POI1_CONTEXT": POI_CONTEXT_FEATURES,
            "POI2_CONTEXT_STAT15": POI_CONTEXT_FEATURES + STAT15_FEATURES,
            "POI3_CONTEXT_FES4": POI_CONTEXT_FEATURES + fes,
            "POI4_CONTEXT_INTERACTIONS": POI_CONTEXT_FEATURES + INTERACTION_FEATURES,
            "POI5_ALL": (POI_CONTEXT_FEATURES + STAT15_FEATURES + fes + INTERACTION_FEATURES),
        }
    elif branch == "GENERAL":
        mapping = {
            "GEN1_STAT15": STAT15_FEATURES,
            "GEN2_FES4": fes,
            "GEN3_STAT15_FES4": STAT15_FEATURES + fes,
        }
    else:
        raise KeyError(f"unknown branch: {branch}")
    if model_id not in mapping:
        raise KeyError(f"unknown model for {branch}: {model_id}")
    return tuple(dict.fromkeys(mapping[model_id]))


def fit_preprocessor(frame: pd.DataFrame, features: Sequence[str]) -> Preprocessor:
    """Fit the contract's finite median/IQR transform on training rows only."""

    matrix = (
        frame.loc[:, list(features)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy("float64", copy=True)
    )
    matrix[~np.isfinite(matrix)] = np.nan
    finite_by_feature = np.isfinite(matrix).any(axis=0)
    missing_features = [
        feature
        for feature, available in zip(features, finite_by_feature, strict=True)
        if not available
    ]
    if missing_features:
        raise ValueError(f"training features have no finite values: {missing_features}")
    with np.errstate(all="ignore"):
        medians = np.nanmedian(matrix, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0, method="linear")
        q75 = np.nanpercentile(matrix, 75, axis=0, method="linear")
    scales = q75 - q25
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    return Preprocessor(tuple(features), medians.astype("float64"), scales.astype("float64"))


def transform_features(frame: pd.DataFrame, state: Preprocessor) -> np.ndarray:
    """Apply a frozen preprocessing state without learning from the target frame."""

    matrix = (
        frame.loc[:, list(state.feature_names)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy("float64", copy=True)
    )
    matrix[~np.isfinite(matrix)] = np.nan
    matrix = np.where(np.isnan(matrix), state.medians, matrix)
    transformed = (matrix - state.medians) / state.scales
    return np.clip(transformed, -10.0, 10.0)


def fit_ridge(
    train: pd.DataFrame,
    assessment: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    alpha: float = RIDGE_ALPHA,
) -> tuple[np.ndarray, np.ndarray, float, Preprocessor]:
    """Fit one deterministic ridge cell and return predictions and coefficients."""

    y = pd.to_numeric(train[target], errors="coerce").to_numpy("float64")
    finite = np.isfinite(y)
    if finite.sum() < max(50, len(features) * 3):
        raise ValueError("insufficient finite training targets for ridge fit")
    fit_frame = train.loc[finite]
    state = fit_preprocessor(fit_frame, features)
    x_train = transform_features(fit_frame, state)
    x_assessment = transform_features(assessment, state)
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
    model.fit(x_train, y[finite])
    prediction = model.predict(x_assessment).astype("float64")
    return prediction, model.coef_.astype("float64"), float(model.intercept_), state


def generate_oof_predictions(
    development: pd.DataFrame,
    folds: Sequence[DateFold],
    *,
    features: Sequence[str],
    target: str,
    date_column: str = "trade_date_ny",
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Generate expanding-fold OOF predictions with audit and coefficients."""

    dates = _normalized_dates(development[date_column])
    predictions = pd.Series(np.nan, index=development.index, dtype="float64")
    audits: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = dates.isin(fold.train_dates)
        assess_mask = dates.isin(fold.assessment_dates)
        if not train_mask.any() or not assess_mask.any():
            raise ValueError(f"fold {fold.fold_id} has an empty train or assessment set")
        prediction, coefficients, intercept, _ = fit_ridge(
            development.loc[train_mask],
            development.loc[assess_mask],
            features=features,
            target=target,
        )
        predictions.loc[assess_mask] = prediction
        audits.append(
            {
                "fold_id": fold.fold_id,
                "train_rows": int(train_mask.sum()),
                "assessment_rows": int(assess_mask.sum()),
                "train_start": min(fold.train_dates),
                "train_end": max(fold.train_dates),
                "assessment_start": min(fold.assessment_dates),
                "assessment_end": max(fold.assessment_dates),
                "embargo_date_count": len(fold.embargo_dates),
            }
        )
        for feature, coefficient in zip(features, coefficients, strict=True):
            coefficient_rows.append(
                {
                    "fit_scope": "Development OOF fold",
                    "fold_id": fold.fold_id,
                    "feature_name": feature,
                    "coefficient": coefficient,
                    "intercept": intercept,
                }
            )
    return predictions, pd.DataFrame(audits), pd.DataFrame(coefficient_rows)


def fit_development_predict_validation(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Refit on all Development and predict Validation once."""

    prediction, coefficients, intercept, _ = fit_ridge(
        development,
        validation,
        features=features,
        target=target,
    )
    rows = [
        {
            "fit_scope": "All Development",
            "fold_id": 0,
            "feature_name": feature,
            "coefficient": coefficient,
            "intercept": intercept,
        }
        for feature, coefficient in zip(features, coefficients, strict=True)
    ]
    return prediction, pd.DataFrame(rows)


def stationary_bootstrap_indices(
    length: int,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    restart_probability: float = BOOTSTRAP_RESTART_PROBABILITY,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Generate deterministic stationary-bootstrap indices over ordered dates."""

    if length <= 0 or replicates <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    if not 0 < restart_probability <= 1:
        raise ValueError("restart probability must be in (0, 1]")
    rng = np.random.default_rng(seed)
    indices = np.empty((replicates, length), dtype=np.int32)
    indices[:, 0] = rng.integers(0, length, size=replicates, dtype=np.int32)
    for column in range(1, length):
        restart = rng.random(replicates) < restart_probability
        continued = (indices[:, column - 1] + 1) % length
        fresh = rng.integers(0, length, size=replicates, dtype=np.int32)
        indices[:, column] = np.where(restart, fresh, continued)
    return indices


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Return a stationary-bootstrap percentile interval for an ordered mean."""

    array = np.asarray(list(values), dtype="float64")
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return math.nan, math.nan
    indices = stationary_bootstrap_indices(len(array), seed=seed)
    means = array[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype("float64"))


def bootstrap_weighted_mean_ci(
    sums: Iterable[float],
    counts: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap a per-trade mean while preserving whole-date trade clusters."""

    numerator = np.asarray(list(sums), dtype="float64")
    denominator = np.asarray(list(counts), dtype="float64")
    finite = np.isfinite(numerator) & np.isfinite(denominator) & (denominator >= 0)
    numerator = numerator[finite]
    denominator = denominator[finite]
    if len(numerator) < 2 or denominator.sum() <= 0:
        return math.nan, math.nan
    indices = stationary_bootstrap_indices(len(numerator), seed=seed)
    sampled_num = numerator[indices].sum(axis=1)
    sampled_den = denominator[indices].sum(axis=1)
    ratios = np.divide(
        sampled_num,
        sampled_den,
        out=np.full_like(sampled_num, np.nan),
        where=sampled_den > 0,
    )
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return math.nan, math.nan
    return tuple(np.quantile(ratios, [0.025, 0.975]).astype("float64"))


def daily_spearman_table(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    target_column: str,
    date_column: str = "trade_date_ny",
    minimum_pairs: int = 10,
) -> pd.DataFrame:
    """Compute one cross-sectional Spearman IC per New York trading date."""

    work = frame[[date_column, prediction_column, target_column]].copy()
    work[date_column] = _normalized_dates(work[date_column])
    work[prediction_column] = pd.to_numeric(work[prediction_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    records = []
    for date, group in work.groupby(date_column, sort=True, observed=True):
        finite = np.isfinite(group[prediction_column]) & np.isfinite(group[target_column])
        sample = group.loc[finite]
        observations = len(sample)
        ic = math.nan
        if observations >= minimum_pairs:
            x = sample[prediction_column].rank(method="average").to_numpy("float64")
            y = sample[target_column].rank(method="average").to_numpy("float64")
            if np.std(x) > 0 and np.std(y) > 0:
                ic = float(np.corrcoef(x, y)[0, 1])
        records.append({"trade_date_ny": date, "observations": observations, "daily_ic": ic})
    return pd.DataFrame.from_records(records, columns=["trade_date_ny", "observations", "daily_ic"])


def _balanced_accuracy(actual_positive: np.ndarray, predicted_positive: np.ndarray) -> float:
    positive = actual_positive
    negative = ~actual_positive
    if positive.sum() == 0 or negative.sum() == 0:
        return math.nan
    sensitivity = (predicted_positive[positive]).mean()
    specificity = (~predicted_positive[negative]).mean()
    return float((sensitivity + specificity) / 2)


def positive_concentration(values: Iterable[float], top_n: int = 10) -> float:
    """Share of positive evidence contributed by the largest positive dates."""

    array = np.asarray(list(values), dtype="float64")
    positive = array[np.isfinite(array) & (array > 0)]
    if positive.size == 0:
        return math.nan
    return float(np.sort(positive)[-top_n:].sum() / positive.sum())


def summarize_predictions(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    target_column: str,
    anchor_prediction_column: str | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize prediction quality with date-clustered daily IC evidence."""

    prediction = pd.to_numeric(frame[prediction_column], errors="coerce").to_numpy("float64")
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy("float64")
    finite = np.isfinite(prediction) & np.isfinite(target)
    sample = frame.loc[finite].copy()
    sample[prediction_column] = prediction[finite]
    sample[target_column] = target[finite]
    daily = daily_spearman_table(
        sample,
        prediction_column=prediction_column,
        target_column=target_column,
    )
    daily["daily_ic"] = pd.to_numeric(daily["daily_ic"], errors="coerce").astype("float64")
    eligible_daily = daily.loc[np.isfinite(daily["daily_ic"].to_numpy())].copy()
    ic_low, ic_high = bootstrap_mean_ci(eligible_daily["daily_ic"], seed=seed)

    paired_mean = 0.0 if anchor_prediction_column == prediction_column else math.nan
    paired_low = 0.0 if anchor_prediction_column == prediction_column else math.nan
    paired_high = 0.0 if anchor_prediction_column == prediction_column else math.nan
    paired_concentration = 0.0 if anchor_prediction_column == prediction_column else math.nan
    if anchor_prediction_column and anchor_prediction_column != prediction_column:
        anchor_daily = daily_spearman_table(
            sample,
            prediction_column=anchor_prediction_column,
            target_column=target_column,
        ).rename(columns={"daily_ic": "anchor_daily_ic"})
        anchor_daily["anchor_daily_ic"] = pd.to_numeric(
            anchor_daily["anchor_daily_ic"], errors="coerce"
        ).astype("float64")
        daily = daily.merge(
            anchor_daily[["trade_date_ny", "anchor_daily_ic"]],
            on="trade_date_ny",
            how="left",
            validate="one_to_one",
        )
        daily["paired_daily_ic_delta"] = (daily["daily_ic"] - daily["anchor_daily_ic"]).astype(
            "float64"
        )
        paired = daily.loc[
            np.isfinite(daily["paired_daily_ic_delta"].to_numpy()),
            "paired_daily_ic_delta",
        ]
        paired_mean = float(paired.mean()) if len(paired) else math.nan
        paired_low, paired_high = bootstrap_mean_ci(paired, seed=seed + 101)
        paired_concentration = positive_concentration(paired)
    elif anchor_prediction_column == prediction_column:
        daily["anchor_daily_ic"] = daily["daily_ic"]
        daily["paired_daily_ic_delta"] = 0.0

    pearson = math.nan
    if finite.sum() >= 2 and np.std(prediction[finite]) > 0 and np.std(target[finite]) > 0:
        pearson = float(np.corrcoef(prediction[finite], target[finite])[0, 1])
    residual = prediction[finite] - target[finite]
    nonzero = target[finite] != 0
    actual_positive = target[finite][nonzero] > 0
    predicted_positive = prediction[finite][nonzero] > 0
    target_finite = target[finite]
    evaluation_date_count = int(_normalized_dates(sample["trade_date_ny"]).nunique())
    summary = {
        "observation_count": int(finite.sum()),
        "evaluation_date_count": evaluation_date_count,
        "ic_date_count": int(len(eligible_daily)),
        "trading_date_count": evaluation_date_count,
        "target_mean": float(np.mean(target_finite)) if len(target_finite) else math.nan,
        "target_std": float(np.std(target_finite, ddof=1)) if len(target_finite) > 1 else math.nan,
        "target_median": float(np.median(target_finite)) if len(target_finite) else math.nan,
        "target_p05": float(np.quantile(target_finite, 0.05)) if len(target_finite) else math.nan,
        "target_p95": float(np.quantile(target_finite, 0.95)) if len(target_finite) else math.nan,
        "daily_ic_mean": float(eligible_daily["daily_ic"].mean())
        if len(eligible_daily)
        else math.nan,
        "daily_ic_std": float(eligible_daily["daily_ic"].std(ddof=1))
        if len(eligible_daily) > 1
        else math.nan,
        "daily_ic_ci_low": ic_low,
        "daily_ic_ci_high": ic_high,
        "pearson_correlation": pearson,
        "mae": float(np.mean(np.abs(residual))) if len(residual) else math.nan,
        "rmse": float(np.sqrt(np.mean(residual**2))) if len(residual) else math.nan,
        "sign_accuracy": (
            float(np.mean(actual_positive == predicted_positive))
            if len(actual_positive)
            else math.nan
        ),
        "balanced_accuracy": _balanced_accuracy(actual_positive, predicted_positive),
        "paired_daily_ic_improvement": paired_mean,
        "paired_daily_ic_ci_low": paired_low,
        "paired_daily_ic_ci_high": paired_high,
        "best_ten_date_positive_paired_delta_share": paired_concentration,
        "best_ten_date_positive_evidence_share": positive_concentration(eligible_daily["daily_ic"]),
    }
    return summary, daily


def noninterpolated_quantile(values: Iterable[float], quantile: float) -> float:
    """One-based ceil order statistic used for frozen policy thresholds."""

    array = np.sort(np.asarray(list(values), dtype="float64"))
    array = array[np.isfinite(array)]
    if len(array) == 0 or not 0 < quantile < 1:
        raise ValueError("quantile input is empty or quantile is outside (0,1)")
    index = max(0, min(len(array) - 1, math.ceil(quantile * len(array)) - 1))
    return float(array[index])


def make_policy_candidates(
    frame: pd.DataFrame,
    *,
    branch: str,
    prediction_column: str,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    """Create cost-independent fixed-horizon tail candidates."""

    if not (
        np.isfinite(low_threshold)
        and np.isfinite(high_threshold)
        and low_threshold < 0 < high_threshold
        and low_threshold < high_threshold
    ):
        raise ValueError("policy thresholds fail the frozen polarity guard")
    work = frame.copy()
    prediction = pd.to_numeric(work[prediction_column], errors="coerce")
    low = prediction.lt(0) & prediction.le(low_threshold)
    high = prediction.gt(0) & prediction.ge(high_threshold)
    selected = work.loc[low | high].copy()
    selected["prediction"] = prediction.loc[selected.index]
    selected["prediction_tail"] = np.where(
        selected["prediction"].ge(high_threshold), "upper", "lower"
    )
    orientation = np.where(selected["prediction"].ge(high_threshold), 1.0, -1.0)
    selected["policy_orientation"] = orientation
    if branch == "POI":
        selected["gross_ticks"] = orientation * pd.to_numeric(
            selected["signed_continuation_return_60_ticks"], errors="coerce"
        )
        selected["trade_direction"] = np.where(orientation > 0, "POI continuation", "POI reversal")
    elif branch == "GENERAL":
        selected["gross_ticks"] = orientation * pd.to_numeric(
            selected["forward_return_60_ticks"], errors="coerce"
        )
        selected["trade_direction"] = np.where(orientation > 0, "long", "short")
    else:
        raise KeyError(f"unknown branch: {branch}")
    selected["low_threshold"] = low_threshold
    selected["high_threshold"] = high_threshold
    selected = selected.loc[np.isfinite(selected["gross_ticks"])].copy()
    return selected


def sequence_fixed_horizon(candidates: pd.DataFrame) -> pd.DataFrame:
    """Enforce one global position with deterministic same-timestamp ties."""

    if candidates.empty:
        return candidates.copy()
    required = {"entry_timestamp_utc", "exit_timestamp_utc_60", "observation_id"}
    missing = required - set(candidates)
    if missing:
        raise KeyError(f"policy candidates are missing columns: {sorted(missing)}")
    work = candidates.copy()
    work["entry_timestamp_utc"] = pd.to_datetime(work["entry_timestamp_utc"], utc=True)
    work["exit_timestamp_utc_60"] = pd.to_datetime(work["exit_timestamp_utc_60"], utc=True)
    if "true_retest_id" not in work:
        work["true_retest_id"] = ""
    elif work["true_retest_id"].isna().any():
        raise ValueError("policy true_retest_id tie keys must be nonmissing")
    if work["observation_id"].isna().any():
        raise ValueError("policy observation_id tie keys must be nonmissing")
    work = work.sort_values(
        ["entry_timestamp_utc", "observation_id", "true_retest_id"], kind="mergesort"
    )
    keep: list[Any] = []
    current_exit: pd.Timestamp | None = None
    for row in work.itertuples():
        entry = row.entry_timestamp_utc
        exit_timestamp = row.exit_timestamp_utc_60
        if pd.isna(entry) or pd.isna(exit_timestamp):
            continue
        if current_exit is None or entry > current_exit:
            keep.append(row.Index)
            current_exit = exit_timestamp
    out = work.loc[keep].copy()
    if len(out) > 1:
        entries = out["entry_timestamp_utc"].iloc[1:].reset_index(drop=True)
        exits = out["exit_timestamp_utc_60"].iloc[:-1].reset_index(drop=True)
        if not entries.gt(exits).all():
            raise AssertionError("sequenced policy contains overlapping positions")
    return out.reset_index(drop=True)


def summarize_policy(
    trades: pd.DataFrame,
    *,
    eligible_dates: Iterable[Any],
    round_trip_cost_ticks: float,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize a sequenced fixed-horizon policy under one cost scenario."""

    dates = tuple(sorted(pd.Timestamp(value).normalize() for value in set(eligible_dates)))
    if trades.empty:
        return {
            "eligible_date_count": int(len(dates)),
            "trade_count": 0,
            "trading_date_count": 0,
            "mean_gross_ticks": math.nan,
            "median_gross_ticks": math.nan,
            "mean_net_ticks": math.nan,
            "median_net_ticks": math.nan,
            "net_ticks_ci_low": math.nan,
            "net_ticks_ci_high": math.nan,
            "win_rate": math.nan,
            "profit_factor": math.nan,
            "total_net_ticks": 0.0,
            "daily_net_ticks_sharpe": math.nan,
            "maximum_drawdown_ticks": 0.0,
            "best_ten_date_pnl_share": math.nan,
        }
    work = trades.copy()
    work["trade_date_ny"] = _normalized_dates(work["trade_date_ny"])
    work["net_ticks"] = pd.to_numeric(work["gross_ticks"], errors="coerce") - float(
        round_trip_cost_ticks
    )
    work = work.loc[np.isfinite(work["net_ticks"])].copy()
    trade_dates = set(work["trade_date_ny"])
    unexpected_dates = sorted(trade_dates - set(dates))
    if unexpected_dates:
        raise ValueError(
            f"policy trades fall outside the frozen eligible-date universe: {unexpected_dates[:5]}"
        )
    daily_sum = work.groupby("trade_date_ny", sort=True)["net_ticks"].sum()
    daily_count = work.groupby("trade_date_ny", sort=True)["net_ticks"].size()
    full_daily = daily_sum.reindex(dates, fill_value=0.0).astype("float64")
    full_count = daily_count.reindex(dates, fill_value=0).astype("float64")
    ci_low, ci_high = bootstrap_weighted_mean_ci(
        full_daily.to_numpy(), full_count.to_numpy(), seed=seed
    )
    positives = work.loc[work["net_ticks"] > 0, "net_ticks"].sum()
    negatives = -work.loc[work["net_ticks"] < 0, "net_ticks"].sum()
    if negatives > 0:
        profit_factor = float(positives / negatives)
    elif positives > 0:
        profit_factor = math.inf
    else:
        profit_factor = math.nan
    daily_std = float(full_daily.std(ddof=1)) if len(full_daily) > 1 else math.nan
    daily_sharpe = (
        float(np.sqrt(252.0) * full_daily.mean() / daily_std)
        if np.isfinite(daily_std) and daily_std > 0
        else math.nan
    )
    equity = np.concatenate(
        [np.array([0.0], dtype="float64"), full_daily.cumsum().to_numpy("float64")]
    )
    drawdown = equity - np.maximum.accumulate(equity)
    return {
        "eligible_date_count": int(len(dates)),
        "trade_count": int(len(work)),
        "trading_date_count": int(work["trade_date_ny"].nunique()),
        "mean_gross_ticks": float(work["gross_ticks"].mean()),
        "median_gross_ticks": float(work["gross_ticks"].median()),
        "mean_net_ticks": float(work["net_ticks"].mean()),
        "median_net_ticks": float(work["net_ticks"].median()),
        "net_ticks_ci_low": ci_low,
        "net_ticks_ci_high": ci_high,
        "win_rate": float(work["net_ticks"].gt(0).mean()),
        "profit_factor": profit_factor,
        "total_net_ticks": float(work["net_ticks"].sum()),
        "daily_net_ticks_sharpe": daily_sharpe,
        "maximum_drawdown_ticks": float(drawdown.min()),
        "best_ten_date_pnl_share": positive_concentration(full_daily),
    }


def development_gate_failures(
    metric: Mapping[str, Any], *, branch: str, requires_paired: bool
) -> list[str]:
    """Apply the exact Development directional-model gates."""

    failures: list[str] = []
    min_dates = 180 if branch == "POI" else 300
    min_rows = 3_000 if branch == "POI" else 50_000
    if int(metric.get("fold_count", 0)) < 4:
        failures.append("minimum_four_folds")
    if int(metric.get("trading_date_count", 0)) < min_dates:
        failures.append("date_floor")
    if int(metric.get("observation_count", 0)) < min_rows:
        failures.append("row_floor")
    if not float(metric.get("daily_ic_mean", math.nan)) >= 0.02:
        failures.append("daily_ic")
    if not float(metric.get("daily_ic_ci_low", math.nan)) > 0:
        failures.append("daily_ic_interval")
    if not float(metric.get("sign_accuracy", math.nan)) >= 0.51:
        failures.append("sign_accuracy")
    concentration = float(metric.get("best_ten_date_positive_evidence_share", math.nan))
    if not np.isfinite(concentration) or concentration > 0.50:
        failures.append("date_concentration")
    if requires_paired:
        if not float(metric.get("paired_daily_ic_improvement", math.nan)) >= 0.01:
            failures.append("paired_improvement")
        if not float(metric.get("paired_daily_ic_ci_low", math.nan)) > 0:
            failures.append("paired_improvement_interval")
    return failures


def validation_gate_failures(
    metric: Mapping[str, Any],
    *,
    branch: str,
    requires_paired: bool,
    development_ic: float,
) -> list[str]:
    """Apply the frozen retrospective Validation confirmation gates."""

    failures: list[str] = []
    min_dates = 120 if branch == "POI" else 150
    min_rows = 2_000 if branch == "POI" else 40_000
    if int(metric.get("trading_date_count", 0)) < min_dates:
        failures.append("date_floor")
    if int(metric.get("observation_count", 0)) < min_rows:
        failures.append("row_floor")
    validation_ic = float(metric.get("daily_ic_mean", math.nan))
    if not validation_ic > 0:
        failures.append("daily_ic")
    if not float(metric.get("daily_ic_ci_low", math.nan)) > 0:
        failures.append("daily_ic_interval")
    if not (np.isfinite(development_ic) and np.sign(validation_ic) == np.sign(development_ic)):
        failures.append("sign_agreement")
    retention = (
        abs(validation_ic) / abs(development_ic)
        if np.isfinite(development_ic) and development_ic != 0
        else math.nan
    )
    if not np.isfinite(retention) or retention < 0.25:
        failures.append("ic_retention")
    if not float(metric.get("sign_accuracy", math.nan)) >= 0.51:
        failures.append("sign_accuracy")
    concentration = float(metric.get("best_ten_date_positive_evidence_share", math.nan))
    if not np.isfinite(concentration) or concentration > 0.50:
        failures.append("date_concentration")
    if requires_paired:
        if not float(metric.get("paired_daily_ic_improvement", math.nan)) >= 0.005:
            failures.append("paired_improvement")
        if not float(metric.get("paired_daily_ic_ci_low", math.nan)) >= 0:
            failures.append("paired_improvement_interval")
    return failures
