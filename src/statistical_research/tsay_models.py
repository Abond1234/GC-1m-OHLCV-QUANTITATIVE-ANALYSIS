"""Frozen Development-only nested model ladder for Tsay research Stage 4.

The module accepts an already guarded Development frame.  It has no filesystem
access and cannot open Validation, MGC, or Final-test data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests

from .tsay_feature_evaluation import stationary_bootstrap_indices
from .tsay_feature_registry import (
    D1_INPUTS_BY_SESSION,
    D2_INPUTS,
    D3_INPUTS_BY_SESSION,
    LOGISTIC_C_GRID,
    O1_INPUTS_BY_SESSION,
    O2_INPUTS,
    O3_INPUTS_BY_SESSION,
    QUANTILE_ALPHA_GRID,
    QUANTILE_INPUTS_BY_SESSION,
    QUANTILE_LEVELS,
    RIDGE_ALPHA_GRID,
    SESSIONS,
    TSAY_SEED,
)

BOOTSTRAP_REPLICATES: Final = 2_000
BOOTSTRAP_RESTART: Final = 0.20
ROLL_SPREAD: Final = "tsay_roll_spread_proxy_120_ticks"
STATE_COLUMNS: Final = (
    "tsay_roll_spread_state_valid_not_identified_120",
    "tsay_roll_spread_state_unavailable_120",
)
CONTINUOUS_ANCHOR: Final = {"directional": "D1", "opportunity": "O1"}
CONTINUOUS_CANDIDATES: Final = {
    "directional": ("D2", "D3", "D4", "D5", "D6"),
    "opportunity": ("O2", "O3", "O4", "O5"),
}


@dataclass(frozen=True)
class Stage4ModelResult:
    """Complete Stage 4 predictive evidence and immutable selection state."""

    oof_predictions: pd.DataFrame
    tuning_audit: pd.DataFrame
    fold_audit: pd.DataFrame
    coefficient_audit: pd.DataFrame
    continuous_evidence: pd.DataFrame
    classifier_evidence: pd.DataFrame
    daily_improvements: pd.DataFrame
    baseline_diagnostics: pd.DataFrame
    continuous_year_evidence: pd.DataFrame
    classifier_reliability: pd.DataFrame
    classifier_year_evidence: pd.DataFrame
    quantile_diagnostics: pd.DataFrame
    quantile_year_diagnostics: pd.DataFrame
    var_diagnostics: pd.DataFrame
    kalman_diagnostics: pd.DataFrame
    max_stat_audit: pd.DataFrame
    architecture_selection: pd.DataFrame
    deployment_objects: Mapping[str, object]


@dataclass(frozen=True)
class _Preprocessor:
    columns: tuple[str, ...]
    transformed_columns: tuple[str, ...]
    medians: np.ndarray
    scales: np.ndarray
    indicator_mask: np.ndarray


def _date_text(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.strftime("%Y-%m-%d")


def _date_weights(dates: Iterable[object]) -> np.ndarray:
    dates = pd.Series(list(dates), dtype="object").astype(str)
    counts = dates.value_counts()
    weights = dates.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=np.float64)
    return weights * (len(weights) / weights.sum())


def _fit_preprocessor(frame: pd.DataFrame, columns: Sequence[str]) -> _Preprocessor:
    values = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float64)
    medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    q25 = np.nanpercentile(values, 25.0, axis=0)
    q75 = np.nanpercentile(values, 75.0, axis=0)
    scales = q75 - q25
    standard = np.nanstd(values, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, standard)
    scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)
    indicator = np.array(
        [column != ROLL_SPREAD and column not in STATE_COLUMNS for column in columns],
        dtype=bool,
    )
    transformed = list(columns)
    transformed.extend(
        f"{column}__missing" for column, keep in zip(columns, indicator, strict=True) if keep
    )
    return _Preprocessor(
        columns=tuple(columns),
        transformed_columns=tuple(transformed),
        medians=medians,
        scales=scales,
        indicator_mask=indicator,
    )


def _transform(frame: pd.DataFrame, prep: _Preprocessor) -> np.ndarray:
    raw = frame.loc[:, prep.columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float64)
    missing = ~np.isfinite(raw)
    filled = np.where(missing, prep.medians, raw)
    scaled = np.clip((filled - prep.medians) / prep.scales, -20.0, 20.0)
    return np.column_stack((scaled, missing[:, prep.indicator_mask].astype(np.float64)))


def _weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    target_mean = float(np.average(y, weights=weights))
    target_scale = float(np.sqrt(np.average((y - target_mean) ** 2, weights=weights)))
    if not np.isfinite(target_scale) or target_scale <= 0.0:
        target_scale = 1.0
    standardized = (y - target_mean) / target_scale
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    root_w = np.sqrt(weights)
    weighted_x = design * root_w[:, None]
    weighted_y = standardized * root_w
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(weighted_x.T @ weighted_x + penalty, weighted_x.T @ weighted_y)
    return target_mean + target_scale * coefficients[0], target_scale * coefficients[1:]


def _predict_ridge(intercept: float, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    return intercept + x @ coefficients


def build_inner_folds(train_dates: Sequence[str]) -> list[dict[str, object]]:
    """Build the frozen 63/1/21/21 complete inner schedule."""

    ordered = list(dict.fromkeys(str(value) for value in train_dates))
    folds: list[dict[str, object]] = []
    train_count = 63
    while train_count + 1 + 21 <= len(ordered):
        assessment = ordered[train_count + 1 : train_count + 22]
        folds.append(
            {
                "inner_fold_id": len(folds),
                "train_dates": ordered[:train_count],
                "embargo_date": ordered[train_count],
                "assessment_dates": assessment,
            }
        )
        train_count += 21
    return folds


def _daily_spearman(frame: pd.DataFrame, observed: str, predicted: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for date, group in frame.groupby("trade_date_ny", sort=True, observed=True):
        values = group.loc[:, (observed, predicted)].to_numpy(np.float64)
        valid = np.isfinite(values).all(axis=1)
        values = values[valid]
        if len(values) < 10 or np.ptp(values[:, 0]) == 0.0 or np.ptp(values[:, 1]) == 0.0:
            continue
        coefficient = stats.spearmanr(values[:, 0], values[:, 1]).statistic
        if np.isfinite(coefficient):
            records.append(
                {
                    "trade_date_ny": str(date),
                    "observation_count": len(values),
                    "daily_ic": float(coefficient),
                }
            )
    return pd.DataFrame.from_records(records)


def _bootstrap_lower(values: np.ndarray, *, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    indices = stationary_bootstrap_indices(
        len(values),
        replicates=BOOTSTRAP_REPLICATES,
        restart_probability=BOOTSTRAP_RESTART,
        seed=seed,
    )
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025, method="linear"))


def _continuity_context(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        pd.to_datetime(frame["decision_timestamp_utc"], utc=True, errors="raise").to_numpy(),
        frame["trade_date_ny"].astype(str).to_numpy(),
        frame["continuous_segment_id"].astype(str).to_numpy(),
    )


def _continuity_pair_positions(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    lag: int,
    context: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return diagnostic pairs without crossing a date, segment, or missing minute."""

    rows = np.asarray(row_indices, dtype=np.int64)
    previous_rows = rows - lag
    previous_positions = np.searchsorted(rows, previous_rows)
    found = previous_positions < len(rows)
    found_positions = np.flatnonzero(found)
    found[found_positions] &= (
        rows[previous_positions[found_positions]] == previous_rows[found_positions]
    )
    current_positions = np.flatnonzero(found)
    lagged_positions = previous_positions[found]
    current_rows = rows[current_positions]
    lagged_rows = rows[lagged_positions]
    timestamps, dates, runs = context if context is not None else _continuity_context(frame)
    valid = (
        (dates[current_rows] == dates[lagged_rows])
        & (runs[current_rows] == runs[lagged_rows])
        & ((timestamps[current_rows] - timestamps[lagged_rows]) == np.timedelta64(lag, "m"))
    )
    return current_positions[valid], lagged_positions[valid]


def _continuity_pairs_through(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    maximum_lag: int,
    context: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        lag: _continuity_pair_positions(frame, row_indices, lag, context)
        for lag in range(1, maximum_lag + 1)
    }


def _pooled_ljung_box(
    frame: pd.DataFrame,
    values: np.ndarray,
    row_indices: np.ndarray,
    lag: int,
    context: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    pairs: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, float]:
    """Boundary-aware pooled Ljung-Box statistic for one residual sequence."""

    series = np.asarray(values, dtype=np.float64)
    centered = series - float(np.mean(series))
    denominator = float(np.dot(centered, centered))
    if len(series) <= lag + 1 or denominator <= 0.0:
        return float("nan"), float("nan")
    terms = []
    for offset in range(1, lag + 1):
        current, previous = (
            pairs[offset]
            if pairs is not None
            else _continuity_pair_positions(frame, row_indices, offset, context)
        )
        if len(current) == 0:
            return float("nan"), float("nan")
        correlation = float(np.dot(centered[current], centered[previous]) / denominator)
        terms.append(correlation * correlation / (len(series) - offset))
    statistic = float(len(series) * (len(series) + 2) * np.sum(terms))
    return statistic, float(stats.chi2.sf(statistic, lag))


def _multivariate_portmanteau(
    frame: pd.DataFrame,
    residuals: np.ndarray,
    row_indices: np.ndarray,
    lag: int,
    context: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    pairs: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, float]:
    """Boundary-aware multivariate Ljung-Box portmanteau diagnostic for VAR(2)."""

    centered = np.asarray(residuals, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    count, dimension = centered.shape
    degrees = dimension * dimension * (lag - 2)
    if count <= lag + 1 or degrees <= 0:
        return float("nan"), float("nan")
    covariance = centered.T @ centered / count
    inverse = np.linalg.pinv(covariance, hermitian=True)
    accumulator = 0.0
    for offset in range(1, lag + 1):
        current, previous = (
            pairs[offset]
            if pairs is not None
            else _continuity_pair_positions(frame, row_indices, offset, context)
        )
        if len(current) == 0:
            return float("nan"), float("nan")
        cross = centered[current].T @ centered[previous] / count
        accumulator += float(np.trace(cross.T @ inverse @ cross @ inverse)) / (count - offset)
    statistic = float(count * count * accumulator)
    return statistic, float(stats.chi2.sf(statistic, degrees))


def _pooled_arch_lm(
    frame: pd.DataFrame,
    values: np.ndarray,
    row_indices: np.ndarray,
    lag: int,
    context: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    pairs: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, float]:
    """Boundary-aware Engle ARCH-LM diagnostic on squared innovations."""

    squared = np.square(np.asarray(values, dtype=np.float64))
    lagged_by_offset = []
    common_current: np.ndarray | None = None
    for offset in range(1, lag + 1):
        current, previous = (
            pairs[offset]
            if pairs is not None
            else _continuity_pair_positions(frame, row_indices, offset, context)
        )
        if common_current is None:
            common_current = current
        else:
            common_current = np.intersect1d(common_current, current, assume_unique=True)
        lagged_by_offset.append((current, previous))
    assert common_current is not None
    if len(common_current) <= lag + 1:
        return float("nan"), float("nan")
    lagged_columns = []
    for current, previous in lagged_by_offset:
        locations = np.searchsorted(current, common_current)
        lagged_columns.append(squared[previous[locations]])
    design = np.column_stack((np.ones(len(common_current)), *lagged_columns))
    target = squared[common_current]
    fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
    centered = target - target.mean()
    total = float(np.dot(centered, centered))
    if total <= 0.0:
        return float("nan"), float("nan")
    residual = target - fitted
    r_squared = 1.0 - float(np.dot(residual, residual)) / total
    statistic = float(len(target) * max(0.0, r_squared))
    return statistic, float(stats.chi2.sf(statistic, lag))


def _holm_adjust(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Attach the frozen Holm-adjusted p-values and every rejection."""

    finite_positions = [
        index for index, record in enumerate(records) if np.isfinite(record["raw_p_value"])
    ]
    if finite_positions:
        raw = np.asarray([records[index]["raw_p_value"] for index in finite_positions])
        rejected, adjusted, _, _ = multipletests(raw, alpha=0.05, method="holm")
        for position, reject, value in zip(finite_positions, rejected, adjusted, strict=True):
            records[position]["holm_adjusted_p_value"] = float(value)
            records[position]["holm_rejected_at_0_05"] = bool(reject)
    for record in records:
        record.setdefault("holm_adjusted_p_value", float("nan"))
        record.setdefault("holm_rejected_at_0_05", False)
    return records


def _best_one_se(
    score_matrix: np.ndarray,
    grid: Sequence[float],
    *,
    maximize: bool,
    strongest_largest: bool,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    means = np.nanmean(score_matrix, axis=0)
    if not np.isfinite(means).any():
        raise RuntimeError("No finite common inner-assessment score is available.")
    best_index = int(np.nanargmax(means) if maximize else np.nanargmin(means))
    best_values = score_matrix[:, best_index]
    valid = np.isfinite(best_values)
    if valid.sum() < 2:
        best_se = 0.0
    else:
        indices = stationary_bootstrap_indices(
            int(valid.sum()),
            replicates=BOOTSTRAP_REPLICATES,
            restart_probability=BOOTSTRAP_RESTART,
            seed=seed,
        )
        bootstrap_means = best_values[valid][indices].mean(axis=1)
        best_se = float(np.std(bootstrap_means, ddof=1))
    if best_se == 0.0:
        eligible = np.isclose(means, means[best_index], rtol=0.0, atol=1.0e-12)
    elif maximize:
        eligible = means >= means[best_index] - best_se
    else:
        eligible = means <= means[best_index] + best_se
    eligible_values = np.asarray(grid, dtype=np.float64)[eligible]
    selected = float(eligible_values.max() if strongest_largest else eligible_values.min())
    audit = [
        {
            "hyperparameter": float(value),
            "mean_score": float(means[index]),
            "best_standard_error": best_se,
            "within_one_best_se": bool(eligible[index]),
            "selected": bool(float(value) == selected),
        }
        for index, value in enumerate(grid)
    ]
    return selected, audit


def _purged_masks(
    frame: pd.DataFrame,
    train_dates: Sequence[str],
    assessment_dates: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    dates = frame["trade_date_ny"].astype(str)
    train = dates.isin(train_dates).to_numpy(copy=True)
    assessment = dates.isin(assessment_dates).to_numpy(copy=True)
    assessment_timestamps = pd.to_datetime(
        frame.loc[assessment, "decision_timestamp_utc"], utc=True, errors="raise"
    )
    if assessment_timestamps.empty:
        return train & False, assessment
    first_decision = assessment_timestamps.min()
    exits = pd.to_datetime(frame["exit_timestamp_utc_60"], utc=True, errors="coerce")
    train &= exits.lt(first_decision).to_numpy()
    return train, assessment


def _ridge_inner_scores(
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str,
    folds: Sequence[Mapping[str, object]],
) -> np.ndarray:
    daily_by_alpha: list[list[float]] = [[] for _ in RIDGE_ALPHA_GRID]
    for fold in folds:
        train_mask, assessment_mask = _purged_masks(
            frame, fold["train_dates"], fold["assessment_dates"]
        )
        y = pd.to_numeric(frame[target], errors="coerce").to_numpy(np.float64)
        train_mask &= np.isfinite(y)
        assessment_mask &= np.isfinite(y)
        if train_mask.sum() == 0 or assessment_mask.sum() == 0:
            continue
        prep = _fit_preprocessor(frame.loc[train_mask], columns)
        x_train = _transform(frame.loc[train_mask], prep)
        x_assessment = _transform(frame.loc[assessment_mask], prep)
        weights = _date_weights(frame.loc[train_mask, "trade_date_ny"])
        assessment_base = frame.loc[assessment_mask, ("trade_date_ny", target)].reset_index(
            drop=True
        )
        for index, alpha in enumerate(RIDGE_ALPHA_GRID):
            intercept, coefficients = _weighted_ridge(x_train, y[train_mask], weights, alpha)
            scored = assessment_base.assign(
                prediction=_predict_ridge(intercept, coefficients, x_assessment)
            )
            daily = _daily_spearman(scored, target, "prediction")
            daily_by_alpha[index].extend(daily.get("daily_ic", pd.Series(dtype=float)).tolist())
    minimum = min((len(values) for values in daily_by_alpha), default=0)
    if minimum == 0:
        return np.empty((0, len(RIDGE_ALPHA_GRID)), dtype=np.float64)
    return np.column_stack([np.asarray(values[:minimum]) for values in daily_by_alpha])


def _fit_outer_ridge(
    frame: pd.DataFrame,
    *,
    architecture: str,
    role: str,
    target: str,
    columns: Sequence[str],
    outer_folds: Sequence[Mapping[str, object]],
) -> tuple[
    list[pd.DataFrame], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    fold_audit: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    inner_score_cache: dict[str, np.ndarray] = {}
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(np.float64)
    for outer in outer_folds:
        inner = build_inner_folds(outer["train_dates"])
        if len(inner) < 2:
            raise RuntimeError(f"{architecture} outer fold has fewer than two inner folds.")
        score_blocks = []
        for inner_fold in inner:
            key = str(inner_fold["embargo_date"])
            if key not in inner_score_cache:
                inner_score_cache[key] = _ridge_inner_scores(frame, columns, target, [inner_fold])
            score_blocks.append(inner_score_cache[key])
        scores = np.vstack(score_blocks)
        selected, audit = _best_one_se(
            scores,
            RIDGE_ALPHA_GRID,
            maximize=True,
            strongest_largest=True,
            seed=TSAY_SEED + int(outer["fold_id"]),
        )
        for record in audit:
            tuning.append(
                {
                    "architecture": architecture,
                    "role": role,
                    "entry_session": str(frame["entry_session"].iloc[0]),
                    "outer_fold_id": int(outer["fold_id"]),
                    "score": "pooled_inner_daily_ic",
                    **record,
                }
            )
        train_mask, assessment_mask = _purged_masks(
            frame, outer["train_dates"], outer["assessment_dates"]
        )
        train_mask &= np.isfinite(y)
        assessment_mask &= np.isfinite(y)
        prep = _fit_preprocessor(frame.loc[train_mask], columns)
        x_train = _transform(frame.loc[train_mask], prep)
        x_assessment = _transform(frame.loc[assessment_mask], prep)
        intercept, coef = _weighted_ridge(
            x_train,
            y[train_mask],
            _date_weights(frame.loc[train_mask, "trade_date_ny"]),
            selected,
        )
        assessment = frame.loc[
            assessment_mask,
            ("observation_id", "trade_date_ny", "entry_session", target),
        ].copy()
        assessment["prediction"] = _predict_ridge(intercept, coef, x_assessment)
        assessment["architecture"] = architecture
        assessment["role"] = role
        assessment["outer_fold_id"] = int(outer["fold_id"])
        predictions.append(assessment)
        coefficients.append(
            {
                "architecture": architecture,
                "role": role,
                "entry_session": str(frame["entry_session"].iloc[0]),
                "outer_fold_id": int(outer["fold_id"]),
                "component": "ridge_slopes",
                "coefficient_names": json.dumps(prep.transformed_columns),
                "coefficient_values": json.dumps(coef.tolist()),
                "intercept": float(intercept),
            }
        )
        fold_audit.append(
            {
                "architecture": architecture,
                "role": role,
                "entry_session": str(frame["entry_session"].iloc[0]),
                "outer_fold_id": int(outer["fold_id"]),
                "train_rows_after_purge": int(train_mask.sum()),
                "assessment_rows": int(assessment_mask.sum()),
                "assessment_dates": int(len(outer["assessment_dates"])),
                "inner_complete_folds": int(len(inner)),
                "selected_hyperparameter": selected,
                "fit_success": True,
            }
        )
    return predictions, tuning, fold_audit, coefficients


def _fit_outer_baseline(
    frame: pd.DataFrame,
    *,
    architecture: str,
    role: str,
    target: str,
    outer_folds: Sequence[Mapping[str, object]],
) -> list[pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(np.float64)
    for outer in outer_folds:
        train, assessment = _purged_masks(frame, outer["train_dates"], outer["assessment_dates"])
        train &= np.isfinite(y)
        assessment &= np.isfinite(y)
        if architecture == "D0":
            prediction = np.repeat(
                np.average(y[train], weights=_date_weights(frame.loc[train, "trade_date_ny"])),
                assessment.sum(),
            )
        elif architecture == "D0_ZERO":
            prediction = np.zeros(assessment.sum(), dtype=np.float64)
        elif architecture == "O0":
            prediction = pd.to_numeric(frame.loc[assessment, "atr_20"], errors="coerce").to_numpy(
                np.float64
            )
        else:
            raise ValueError(architecture)
        result = frame.loc[
            assessment, ("observation_id", "trade_date_ny", "entry_session", target)
        ].copy()
        result["prediction"] = prediction
        result["architecture"] = architecture
        result["role"] = role
        result["outer_fold_id"] = int(outer["fold_id"])
        predictions.append(result)
    return predictions


def _logistic_inner_scores(
    frame: pd.DataFrame,
    columns: Sequence[str],
    folds: Sequence[Mapping[str, object]],
) -> np.ndarray:
    score_by_c: list[list[float]] = [[] for _ in LOGISTIC_C_GRID]
    y = pd.to_numeric(frame["expansion_label_60"], errors="coerce").to_numpy(np.float64)
    for fold in folds:
        train, assessment = _purged_masks(frame, fold["train_dates"], fold["assessment_dates"])
        train &= np.isfinite(y)
        assessment &= np.isfinite(y)
        prep = _fit_preprocessor(frame.loc[train], columns)
        x_train = _transform(frame.loc[train], prep)
        x_assessment = _transform(frame.loc[assessment], prep)
        train_w = _date_weights(frame.loc[train, "trade_date_ny"])
        assess_dates = frame.loc[assessment, "trade_date_ny"].astype(str)
        assess_w = _date_weights(assess_dates)
        for index, c_value in enumerate(LOGISTIC_C_GRID):
            model = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                class_weight=None,
                max_iter=5_000,
                random_state=TSAY_SEED,
            ).fit(x_train, y[train].astype(np.int8), sample_weight=train_w)
            probabilities = model.predict_proba(x_assessment)[:, 1]
            daily = (
                pd.DataFrame(
                    {
                        "date": assess_dates.to_numpy(),
                        "loss": (y[assessment] - probabilities) ** 2,
                        "w": assess_w,
                    }
                )
                .groupby("date", sort=True, observed=True)
                .apply(
                    lambda group: float(np.average(group["loss"], weights=group["w"])),
                    include_groups=False,
                )
            )
            score_by_c[index].extend(daily.tolist())
    minimum = min((len(values) for values in score_by_c), default=0)
    if minimum == 0:
        return np.empty((0, len(LOGISTIC_C_GRID)), dtype=np.float64)
    return np.column_stack([np.asarray(values[:minimum]) for values in score_by_c])


def _fit_outer_logistic(
    frame: pd.DataFrame,
    *,
    architecture: str,
    columns: Sequence[str] | None,
    outer_folds: Sequence[Mapping[str, object]],
) -> tuple[
    list[pd.DataFrame], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    folds_out: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    inner_score_cache: dict[str, np.ndarray] = {}
    y = pd.to_numeric(frame["expansion_label_60"], errors="coerce").to_numpy(np.float64)
    for outer in outer_folds:
        train, assessment = _purged_masks(frame, outer["train_dates"], outer["assessment_dates"])
        train &= np.isfinite(y)
        assessment &= np.isfinite(y)
        if architecture == "E0":
            probability = float(
                np.average(y[train], weights=_date_weights(frame.loc[train, "trade_date_ny"]))
            )
            predicted = np.repeat(probability, assessment.sum())
            selected = float("nan")
            names: tuple[str, ...] = ()
            coef = np.empty(0)
            clipped = float(np.clip(probability, 1.0e-12, 1.0 - 1.0e-12))
            intercept = float(np.log(clipped / (1.0 - clipped)))
            inner_count = 0
        else:
            assert columns is not None
            inner = build_inner_folds(outer["train_dates"])
            score_blocks = []
            for inner_fold in inner:
                key = str(inner_fold["embargo_date"])
                if key not in inner_score_cache:
                    inner_score_cache[key] = _logistic_inner_scores(frame, columns, [inner_fold])
                score_blocks.append(inner_score_cache[key])
            scores = np.vstack(score_blocks)
            selected, audit = _best_one_se(
                scores,
                LOGISTIC_C_GRID,
                maximize=False,
                strongest_largest=False,
                seed=TSAY_SEED + 20_000 + int(outer["fold_id"]),
            )
            for record in audit:
                tuning.append(
                    {
                        "architecture": architecture,
                        "role": "expansion",
                        "entry_session": str(frame["entry_session"].iloc[0]),
                        "outer_fold_id": int(outer["fold_id"]),
                        "score": "pooled_inner_daily_brier",
                        **record,
                    }
                )
            prep = _fit_preprocessor(frame.loc[train], columns)
            x_train = _transform(frame.loc[train], prep)
            x_assessment = _transform(frame.loc[assessment], prep)
            model = LogisticRegression(
                C=selected,
                solver="lbfgs",
                class_weight=None,
                max_iter=5_000,
                random_state=TSAY_SEED,
            ).fit(
                x_train,
                y[train].astype(np.int8),
                sample_weight=_date_weights(frame.loc[train, "trade_date_ny"]),
            )
            predicted = model.predict_proba(x_assessment)[:, 1]
            names = prep.transformed_columns
            coef = model.coef_[0].astype(np.float64)
            intercept = float(model.intercept_[0])
            inner_count = len(inner)
        result = frame.loc[
            assessment, ("observation_id", "trade_date_ny", "entry_session", "expansion_label_60")
        ].copy()
        result["prediction"] = predicted
        result["architecture"] = architecture
        result["role"] = "expansion"
        result["outer_fold_id"] = int(outer["fold_id"])
        predictions.append(result)
        folds_out.append(
            {
                "architecture": architecture,
                "role": "expansion",
                "entry_session": str(frame["entry_session"].iloc[0]),
                "outer_fold_id": int(outer["fold_id"]),
                "train_rows_after_purge": int(train.sum()),
                "assessment_rows": int(assessment.sum()),
                "assessment_dates": len(outer["assessment_dates"]),
                "inner_complete_folds": inner_count,
                "selected_hyperparameter": selected,
                "fit_success": True,
            }
        )
        coefficients.append(
            {
                "architecture": architecture,
                "role": "expansion",
                "entry_session": str(frame["entry_session"].iloc[0]),
                "outer_fold_id": int(outer["fold_id"]),
                "component": "logistic_slopes",
                "coefficient_names": json.dumps(names),
                "coefficient_values": json.dumps(coef.tolist()),
                "intercept": intercept,
            }
        )
    return predictions, tuning, folds_out, coefficients


def _smooth_quantile_objective(
    parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    tau: float,
    alpha: float,
) -> tuple[float, np.ndarray]:
    residual = y - parameters[0] - x @ parameters[1:]
    root = np.sqrt(residual * residual + 1.0e-8)
    loss = 0.5 * (root + (2.0 * tau - 1.0) * residual)
    derivative = 0.5 * (residual / root + 2.0 * tau - 1.0)
    total_weight = weights.sum()
    objective = float(
        np.dot(weights, loss) / total_weight + 0.5 * alpha * np.dot(parameters[1:], parameters[1:])
    )
    gradient = np.empty_like(parameters)
    gradient[0] = -np.dot(weights, derivative) / total_weight
    gradient[1:] = -(x.T @ (weights * derivative)) / total_weight + alpha * parameters[1:]
    return objective, gradient


def _fit_quantile(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    tau: float,
    alpha: float,
) -> optimize.OptimizeResult:
    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    initial[0] = float(np.quantile(y, tau, method="linear"))
    return optimize.minimize(
        _smooth_quantile_objective,
        initial,
        args=(x, y, weights, tau, alpha),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "gtol": 1.0e-8},
    )


def _pinball(y: np.ndarray, prediction: np.ndarray, tau: float) -> np.ndarray:
    residual = y - prediction
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def _select_quantile_alpha(
    frame: pd.DataFrame,
    columns: Sequence[str],
    folds: Sequence[Mapping[str, object]],
    score_cache: dict[str, np.ndarray],
) -> tuple[float, list[dict[str, float]]]:
    score_by_alpha: list[list[float]] = [[] for _ in QUANTILE_ALPHA_GRID]
    y = pd.to_numeric(frame["forward_return_60_atr"], errors="coerce").to_numpy(np.float64)
    for fold in folds:
        key = str(fold["embargo_date"])
        if key in score_cache:
            block = score_cache[key]
            for index in range(len(QUANTILE_ALPHA_GRID)):
                score_by_alpha[index].extend(block[:, index].tolist())
            continue
        train, assessment = _purged_masks(frame, fold["train_dates"], fold["assessment_dates"])
        train &= np.isfinite(y)
        assessment &= np.isfinite(y)
        prep = _fit_preprocessor(frame.loc[train], columns)
        x_train = _transform(frame.loc[train], prep)
        x_assessment = _transform(frame.loc[assessment], prep)
        train_w = _date_weights(frame.loc[train, "trade_date_ny"])
        assessment_dates = frame.loc[assessment, "trade_date_ny"].astype(str).to_numpy()
        fold_scores: list[np.ndarray] = []
        for index, alpha in enumerate(QUANTILE_ALPHA_GRID):
            predictions = []
            for tau in QUANTILE_LEVELS:
                result = _fit_quantile(x_train, y[train], train_w, tau, alpha)
                if not result.success or not np.isfinite(result.fun):
                    raise RuntimeError(f"Quantile optimizer failed: {result.message}")
                predictions.append(result.x[0] + x_assessment @ result.x[1:])
            repaired = np.sort(np.column_stack(predictions), axis=1)
            row_loss = np.mean(
                np.column_stack(
                    [
                        _pinball(y[assessment], repaired[:, tau_index], tau)
                        for tau_index, tau in enumerate(QUANTILE_LEVELS)
                    ]
                ),
                axis=1,
            )
            daily = (
                pd.DataFrame({"date": assessment_dates, "loss": row_loss})
                .groupby("date", sort=True, observed=True)["loss"]
                .mean()
                .to_numpy(np.float64)
            )
            score_by_alpha[index].extend(daily.tolist())
            fold_scores.append(daily)
        minimum = min(len(values) for values in fold_scores)
        score_cache[key] = np.column_stack([values[:minimum] for values in fold_scores])
    matrix = np.column_stack(score_by_alpha)
    return _best_one_se(
        matrix,
        QUANTILE_ALPHA_GRID,
        maximize=False,
        strongest_largest=True,
        seed=TSAY_SEED + 40_000 + len(folds),
    )


def _fit_outer_quantiles(
    frame: pd.DataFrame,
    outer_folds: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> tuple[
    list[pd.DataFrame], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    folds_out: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    inner_score_cache: dict[str, np.ndarray] = {}
    y = pd.to_numeric(frame["forward_return_60_atr"], errors="coerce").to_numpy(np.float64)
    for outer in outer_folds:
        inner = build_inner_folds(outer["train_dates"])
        selected, audit = _select_quantile_alpha(frame, columns, inner, inner_score_cache)
        for record in audit:
            tuning.append(
                {
                    "architecture": "D5_O4",
                    "role": "joint_quantile",
                    "entry_session": str(frame["entry_session"].iloc[0]),
                    "outer_fold_id": int(outer["fold_id"]),
                    "score": "equal_tau_exact_pinball_loss",
                    **record,
                }
            )
        train, assessment = _purged_masks(frame, outer["train_dates"], outer["assessment_dates"])
        train &= np.isfinite(y)
        assessment &= np.isfinite(y)
        prep = _fit_preprocessor(frame.loc[train], columns)
        x_train = _transform(frame.loc[train], prep)
        x_assessment = _transform(frame.loc[assessment], prep)
        train_w = _date_weights(frame.loc[train, "trade_date_ny"])
        values: dict[str, np.ndarray] = {}
        baselines: dict[str, float] = {}
        for tau in QUANTILE_LEVELS:
            result = _fit_quantile(x_train, y[train], train_w, tau, selected)
            if not result.success or not np.isfinite(result.fun):
                raise RuntimeError(f"Outer quantile optimizer failed: {result.message}")
            values[f"q{int(tau * 100):02d}"] = result.x[0] + x_assessment @ result.x[1:]
            baselines[f"baseline_q{int(tau * 100):02d}"] = float(
                np.quantile(y[train], tau, method="linear")
            )
            coefficients.append(
                {
                    "architecture": "D5_O4",
                    "role": "joint_quantile",
                    "entry_session": str(frame["entry_session"].iloc[0]),
                    "outer_fold_id": int(outer["fold_id"]),
                    "component": f"q{int(tau * 100):02d}_slopes",
                    "coefficient_names": json.dumps(prep.transformed_columns),
                    "coefficient_values": json.dumps(result.x[1:].tolist()),
                    "intercept": float(result.x[0]),
                }
            )
        base = frame.loc[
            assessment,
            (
                "observation_id",
                "trade_date_ny",
                "entry_session",
                "forward_return_60_atr",
                "future_range_60_atr",
            ),
        ].copy()
        for name, value in {**values, **baselines}.items():
            base[name] = value
        raw = base.loc[:, ("q10", "q50", "q90")].to_numpy(np.float64)
        repaired = np.sort(raw, axis=1)
        base["pre_repair_crossing"] = (np.diff(raw, axis=1) < 0.0).any(axis=1)
        base[["q10", "q50", "q90"]] = repaired
        d5 = base.loc[
            :,
            (
                "observation_id",
                "trade_date_ny",
                "entry_session",
                "forward_return_60_atr",
                "q10",
                "q50",
                "q90",
                "baseline_q10",
                "baseline_q50",
                "baseline_q90",
                "pre_repair_crossing",
            ),
        ].copy()
        d5["prediction"] = d5["q50"]
        d5["architecture"] = "D5"
        d5["role"] = "directional"
        d5["outer_fold_id"] = int(outer["fold_id"])
        o4 = base.loc[
            :,
            (
                "observation_id",
                "trade_date_ny",
                "entry_session",
                "future_range_60_atr",
                "q10",
                "q50",
                "q90",
                "baseline_q10",
                "baseline_q50",
                "baseline_q90",
                "pre_repair_crossing",
            ),
        ].copy()
        o4["prediction"] = o4["q90"] - o4["q10"]
        o4["architecture"] = "O4"
        o4["role"] = "opportunity"
        o4["outer_fold_id"] = int(outer["fold_id"])
        predictions.extend((d5, o4))
        folds_out.append(
            {
                "architecture": "D5_O4",
                "role": "joint_quantile",
                "entry_session": str(frame["entry_session"].iloc[0]),
                "outer_fold_id": int(outer["fold_id"]),
                "train_rows_after_purge": int(train.sum()),
                "assessment_rows": int(assessment.sum()),
                "assessment_dates": len(outer["assessment_dates"]),
                "inner_complete_folds": len(inner),
                "selected_hyperparameter": selected,
                "fit_success": True,
            }
        )
    return predictions, tuning, folds_out, coefficients


def _kalman_nll(
    log_variances: np.ndarray,
    runs: np.ndarray,
    variance: float,
) -> float:
    q_value, r_value = np.exp(log_variances)
    total = 0.0
    count = 0
    finite = np.isfinite(runs)
    has_value = finite.any(axis=1)
    first_position = np.argmax(finite, axis=1)
    state = np.zeros(len(runs), dtype=np.float64)
    state[has_value] = runs[np.flatnonzero(has_value), first_position[has_value]]
    covariance = np.full(len(runs), 1.0e6 * variance, dtype=np.float64)
    for position in range(runs.shape[1]):
        observed_mask = finite[:, position]
        missing_after_start = has_value & ~observed_mask & (position >= first_position)
        covariance[missing_after_start] += q_value
        if not observed_mask.any():
            continue
        observed = runs[observed_mask, position]
        innovation_variance = covariance[observed_mask] + r_value
        innovation = observed - state[observed_mask]
        total += float(
            0.5
            * np.sum(
                np.log(2.0 * np.pi * innovation_variance)
                + innovation * innovation / innovation_variance
            )
        )
        gain = covariance[observed_mask] / innovation_variance
        state[observed_mask] += gain * innovation
        covariance[observed_mask] = (1.0 - gain) * covariance[observed_mask] + q_value
        count += int(observed_mask.sum())
    return total if count else float("inf")


def _estimate_kalman(frame: pd.DataFrame, mask: np.ndarray) -> tuple[float, float, float]:
    selected = frame.loc[mask].copy()
    selected["_kalman_y"] = np.log(
        np.maximum(
            pd.to_numeric(selected["realized_volatility_15"], errors="coerce").to_numpy(np.float64),
            1.0e-8,
        )
    )
    run_list = [
        group.sort_values("decision_timestamp_utc", kind="mergesort")["_kalman_y"].to_numpy(
            np.float64
        )
        for _, group in selected.groupby(
            ["trade_date_ny", "continuous_segment_id"], sort=True, observed=True
        )
    ]
    pooled = np.concatenate([run[np.isfinite(run)] for run in run_list])
    maximum_length = max(len(run) for run in run_list)
    runs = np.full((len(run_list), maximum_length), np.nan, dtype=np.float64)
    for index, run in enumerate(run_list):
        runs[index, : len(run)] = run
    variance = float(np.var(pooled))
    if not np.isfinite(variance) or variance <= 0.0:
        raise RuntimeError("Kalman training variance is not positive.")
    lower = np.log(variance * 1.0e-6)
    upper = np.log(variance * 1.0e2)
    best: optimize.OptimizeResult | None = None
    for q_multiplier, r_multiplier in ((0.01, 0.5), (0.1, 0.5), (0.5, 0.5)):
        result = optimize.minimize(
            _kalman_nll,
            np.log([q_multiplier * variance, r_multiplier * variance]),
            args=(runs, variance),
            method="L-BFGS-B",
            bounds=((lower, upper), (lower, upper)),
            options={"maxiter": 500, "gtol": 1.0e-6},
        )
        if result.success and np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("All three frozen Kalman likelihood starts failed.")
    q_value, r_value = np.exp(best.x)
    return float(q_value), float(r_value), float(best.fun)


def _kalman_features(
    frame: pd.DataFrame,
    mask: np.ndarray,
    q_value: float,
    r_value: float,
) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan,
        index=frame.index,
        columns=(
            "tsay_kalman_next_prior",
            "tsay_kalman_standardized_innovation",
            "tsay_kalman_gain",
            "tsay_kalman_next_state_std",
        ),
        dtype=np.float64,
    )
    selected = frame.loc[mask].copy()
    selected["_kalman_y"] = np.log(
        np.maximum(
            pd.to_numeric(selected["realized_volatility_15"], errors="coerce").to_numpy(np.float64),
            1.0e-8,
        )
    )
    variance = float(np.nanvar(selected["_kalman_y"].to_numpy(np.float64)))
    for _, run in selected.groupby(
        ["trade_date_ny", "continuous_segment_id"], sort=True, observed=True
    ):
        run = run.sort_values("decision_timestamp_utc", kind="mergesort")
        values = run["_kalman_y"].to_numpy(np.float64)
        first = np.flatnonzero(np.isfinite(values))
        if not len(first):
            continue
        state = float(values[first[0]])
        covariance = 1.0e6 * variance
        valid_count = 0
        outputs = np.full((len(run), 4), np.nan, dtype=np.float64)
        for position, observed in enumerate(values):
            if not np.isfinite(observed):
                covariance += q_value
                continue
            innovation_variance = covariance + r_value
            innovation = float(observed - state)
            gain = covariance / innovation_variance
            state += gain * innovation
            covariance = (1.0 - gain) * covariance + q_value
            valid_count += 1
            if valid_count > 15:
                outputs[position] = (
                    state,
                    innovation / np.sqrt(innovation_variance),
                    gain,
                    np.sqrt(covariance),
                )
        result.loc[run.index] = outputs
    return result


def _fit_outer_kalman_ridge(
    frame: pd.DataFrame,
    outer_folds: Sequence[Mapping[str, object]],
) -> tuple[
    list[pd.DataFrame],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    folds_out: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    inner_score_cache: dict[str, np.ndarray] = {}
    target = pd.to_numeric(frame["future_range_60_atr"], errors="coerce").to_numpy(np.float64)
    model_columns = (
        "atr_20",
        "tsay_kalman_next_prior",
        "tsay_kalman_standardized_innovation",
        "tsay_kalman_gain",
        "tsay_kalman_next_state_std",
    )
    for outer in outer_folds:
        inner_folds = build_inner_folds(outer["train_dates"])
        score_by_alpha: list[list[float]] = [[] for _ in RIDGE_ALPHA_GRID]
        for inner in inner_folds:
            key = str(inner["embargo_date"])
            if key in inner_score_cache:
                block = inner_score_cache[key]
                for index in range(len(RIDGE_ALPHA_GRID)):
                    score_by_alpha[index].extend(block[:, index].tolist())
                continue
            train, assessment = _purged_masks(
                frame, inner["train_dates"], inner["assessment_dates"]
            )
            q_value, r_value, _ = _estimate_kalman(frame, train)
            augmented = frame.copy()
            augmented.loc[:, model_columns[1:]] = _kalman_features(
                frame, train | assessment, q_value, r_value
            ).loc[:, model_columns[1:]]
            train &= np.isfinite(target)
            assessment &= np.isfinite(target)
            prep = _fit_preprocessor(augmented.loc[train], model_columns)
            x_train = _transform(augmented.loc[train], prep)
            x_assessment = _transform(augmented.loc[assessment], prep)
            weights = _date_weights(augmented.loc[train, "trade_date_ny"])
            scored_base = augmented.loc[
                assessment, ("trade_date_ny", "future_range_60_atr")
            ].reset_index(drop=True)
            fold_scores: list[list[float]] = [[] for _ in RIDGE_ALPHA_GRID]
            for index, alpha in enumerate(RIDGE_ALPHA_GRID):
                intercept, coef = _weighted_ridge(x_train, target[train], weights, alpha)
                daily = _daily_spearman(
                    scored_base.assign(prediction=_predict_ridge(intercept, coef, x_assessment)),
                    "future_range_60_atr",
                    "prediction",
                )
                values = daily.get("daily_ic", pd.Series(dtype=float)).tolist()
                score_by_alpha[index].extend(values)
                fold_scores[index].extend(values)
            minimum_fold = min(len(values) for values in fold_scores)
            inner_score_cache[key] = np.column_stack(
                [np.asarray(values[:minimum_fold]) for values in fold_scores]
            )
        minimum = min(len(values) for values in score_by_alpha)
        selected, audit = _best_one_se(
            np.column_stack([np.asarray(values[:minimum]) for values in score_by_alpha]),
            RIDGE_ALPHA_GRID,
            maximize=True,
            strongest_largest=True,
            seed=TSAY_SEED + 80_000 + int(outer["fold_id"]),
        )
        for record in audit:
            tuning.append(
                {
                    "architecture": "O5",
                    "role": "opportunity",
                    "entry_session": str(frame["entry_session"].iloc[0]),
                    "outer_fold_id": int(outer["fold_id"]),
                    "score": "pooled_inner_daily_ic",
                    **record,
                }
            )
        train, assessment = _purged_masks(frame, outer["train_dates"], outer["assessment_dates"])
        q_value, r_value, nll = _estimate_kalman(frame, train)
        augmented = frame.copy()
        augmented.loc[:, model_columns[1:]] = _kalman_features(
            frame, train | assessment, q_value, r_value
        ).loc[:, model_columns[1:]]
        train &= np.isfinite(target)
        assessment &= np.isfinite(target)
        prep = _fit_preprocessor(augmented.loc[train], model_columns)
        x_train = _transform(augmented.loc[train], prep)
        x_assessment = _transform(augmented.loc[assessment], prep)
        intercept, coef = _weighted_ridge(
            x_train,
            target[train],
            _date_weights(augmented.loc[train, "trade_date_ny"]),
            selected,
        )
        scored = augmented.loc[
            assessment,
            (
                "observation_id",
                "trade_date_ny",
                "entry_session",
                "future_range_60_atr",
            ),
        ].copy()
        scored["prediction"] = _predict_ridge(intercept, coef, x_assessment)
        scored["architecture"] = "O5"
        scored["role"] = "opportunity"
        scored["outer_fold_id"] = int(outer["fold_id"])
        predictions.append(scored)
        common = {
            "architecture": "O5",
            "role": "opportunity",
            "entry_session": str(frame["entry_session"].iloc[0]),
            "outer_fold_id": int(outer["fold_id"]),
        }
        folds_out.append(
            {
                **common,
                "train_rows_after_purge": int(train.sum()),
                "assessment_rows": int(assessment.sum()),
                "assessment_dates": len(outer["assessment_dates"]),
                "inner_complete_folds": len(inner_folds),
                "selected_hyperparameter": selected,
                "fit_success": True,
            }
        )
        coefficients.append(
            {
                **common,
                "component": "final_opportunity_ridge_slopes",
                "coefficient_names": json.dumps(prep.transformed_columns),
                "coefficient_values": json.dumps(coef.tolist()),
                "intercept": float(intercept),
            }
        )
        innovation_all = pd.to_numeric(
            augmented["tsay_kalman_standardized_innovation"], errors="coerce"
        ).to_numpy(np.float64)
        innovation_rows = np.flatnonzero(train & np.isfinite(innovation_all))
        innovations = innovation_all[innovation_rows]
        continuity = _continuity_context(augmented)
        diagnostic_pairs = _continuity_pairs_through(augmented, innovation_rows, 20, continuity)
        diagnostic_records: list[dict[str, object]] = []
        for lag in (5, 10, 20):
            statistic, p_value = _pooled_ljung_box(
                augmented,
                innovations,
                innovation_rows,
                lag,
                continuity,
                diagnostic_pairs,
            )
            diagnostic_records.append(
                {
                    "diagnostic": "innovation_ljung_box",
                    "lag": lag,
                    "statistic": statistic,
                    "raw_p_value": p_value,
                }
            )
            statistic, p_value = _pooled_arch_lm(
                augmented,
                innovations,
                innovation_rows,
                lag,
                continuity,
                diagnostic_pairs,
            )
            diagnostic_records.append(
                {
                    "diagnostic": "squared_innovation_arch_lm",
                    "lag": lag,
                    "statistic": statistic,
                    "raw_p_value": p_value,
                }
            )
        for record in _holm_adjust(diagnostic_records):
            record.update(common)
            record.update(
                {
                    "q_value": q_value,
                    "r_value": r_value,
                    "log_q": float(np.log(q_value)),
                    "log_r": float(np.log(r_value)),
                    "innovation_nll": nll,
                    "innovation_mean": float(np.mean(innovations)),
                    "innovation_std": float(np.std(innovations)),
                    "fit_success": True,
                }
            )
            diagnostics.append(record)
    return predictions, tuning, folds_out, coefficients, diagnostics


def _var_design(
    frame: pd.DataFrame,
    mask: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = (
        frame.loc[:, ("return_1m_bps", "tsay_clock_z_range", "tsay_clock_z_volume")]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(np.float64)
    )
    standardized = (components - means) / scales
    timestamp = pd.to_datetime(frame["decision_timestamp_utc"], utc=True, errors="raise")
    date = frame["trade_date_ny"].astype(str).to_numpy()
    run = frame["continuous_segment_id"].astype(str).to_numpy()
    valid = mask.copy()
    valid[:2] = False
    valid[2:] &= (
        np.isfinite(standardized[2:]).all(axis=1)
        & np.isfinite(standardized[1:-1]).all(axis=1)
        & np.isfinite(standardized[:-2]).all(axis=1)
        & (date[2:] == date[1:-1])
        & (date[2:] == date[:-2])
        & (run[2:] == run[1:-1])
        & (run[2:] == run[:-2])
        & (
            (timestamp.iloc[2:].to_numpy() - timestamp.iloc[1:-1].to_numpy())
            == np.timedelta64(1, "m")
        )
        & (
            (timestamp.iloc[1:-1].to_numpy() - timestamp.iloc[:-2].to_numpy())
            == np.timedelta64(1, "m")
        )
    )
    indices = np.flatnonzero(valid)
    design = np.column_stack((standardized[indices - 1], standardized[indices - 2]))
    return design, standardized[indices], indices


def _fit_var(
    frame: pd.DataFrame,
    train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    components = (
        frame.loc[train, ("return_1m_bps", "tsay_clock_z_range", "tsay_clock_z_volume")]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(np.float64)
    )
    means = np.nanmean(components, axis=0)
    scales = np.nanstd(components, axis=0)
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise RuntimeError("D4 component standardization failed.")
    design, target, indices = _var_design(frame, train, means, scales)
    dates = frame.loc[indices, "trade_date_ny"]
    weights = _date_weights(dates)
    full = np.column_stack((np.ones(len(design)), design))
    root = np.sqrt(weights)
    weighted = full * root[:, None]
    penalty = np.eye(full.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted.T @ weighted + penalty,
        weighted.T @ (target * root[:, None]),
    )
    lag1 = coefficients[1:4].T
    lag2 = coefficients[4:7].T
    companion = np.block([[lag1, lag2], [np.eye(3, dtype=np.float64), np.zeros((3, 3))]])
    maximum_eigenvalue = float(np.max(np.abs(np.linalg.eigvals(companion))))
    if maximum_eigenvalue >= 0.999:
        raise RuntimeError(f"D4 companion eigenvalue is unstable: {maximum_eigenvalue}")
    return coefficients[0], lag1, lag2, maximum_eigenvalue, np.column_stack((means, scales))


def _fit_var_path(
    frame: pd.DataFrame,
    train: np.ndarray,
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    components = (
        frame.loc[train, ("return_1m_bps", "tsay_clock_z_range", "tsay_clock_z_volume")]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(np.float64)
    )
    means = np.nanmean(components, axis=0)
    scales = np.nanstd(components, axis=0)
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise RuntimeError("D4 component standardization failed.")
    design, target, indices = _var_design(frame, train, means, scales)
    weights = _date_weights(frame.loc[indices, "trade_date_ny"])
    full = np.column_stack((np.ones(len(design)), design))
    root = np.sqrt(weights)
    weighted = full * root[:, None]
    cross = weighted.T @ weighted
    rhs = weighted.T @ (target * root[:, None])
    output = {}
    for alpha in RIDGE_ALPHA_GRID:
        penalty = np.eye(full.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(cross + penalty, rhs)
        lag1 = coefficients[1:4].T
        lag2 = coefficients[4:7].T
        companion = np.block([[lag1, lag2], [np.eye(3), np.zeros((3, 3), dtype=np.float64)]])
        maximum_eigenvalue = float(np.max(np.abs(np.linalg.eigvals(companion))))
        if maximum_eigenvalue < 0.999:
            output[float(alpha)] = (
                coefficients[0],
                lag1,
                lag2,
                maximum_eigenvalue,
                np.column_stack((means, scales)),
            )
    return output


def _var_raw_forecast(
    frame: pd.DataFrame,
    mask: np.ndarray,
    intercept: np.ndarray,
    lag1: np.ndarray,
    lag2: np.ndarray,
    moments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means, scales = moments[:, 0], moments[:, 1]
    components = (
        frame.loc[:, ("return_1m_bps", "tsay_clock_z_range", "tsay_clock_z_volume")]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(np.float64)
    )
    standardized = (components - means) / scales
    timestamp = pd.to_datetime(frame["decision_timestamp_utc"], utc=True, errors="raise")
    date = frame["trade_date_ny"].astype(str).to_numpy()
    run = frame["continuous_segment_id"].astype(str).to_numpy()
    valid = mask.copy()
    valid[0] = False
    valid[1:] &= (
        np.isfinite(standardized[1:]).all(axis=1)
        & np.isfinite(standardized[:-1]).all(axis=1)
        & (date[1:] == date[:-1])
        & (run[1:] == run[:-1])
        & (
            (timestamp.iloc[1:].to_numpy() - timestamp.iloc[:-1].to_numpy())
            == np.timedelta64(1, "m")
        )
    )
    indices = np.flatnonzero(valid)
    total = np.zeros(len(indices), dtype=np.float64)
    first = standardized[indices].copy()
    second = standardized[indices - 1].copy()
    for _ in range(60):
        forecast = intercept + first @ lag1.T + second @ lag2.T
        total += means[0] + scales[0] * forecast[:, 0]
        second = first
        first = forecast
    return total, indices


def _fit_var_calibration(
    frame: pd.DataFrame,
    train: np.ndarray,
    raw: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    target = pd.to_numeric(frame["forward_return_60_atr"], errors="coerce").to_numpy(np.float64)[
        indices
    ]
    valid = np.isfinite(target) & np.isfinite(raw)
    weights = _date_weights(frame.loc[indices[valid], "trade_date_ny"])
    return _weighted_calibration(target[valid], raw[valid], weights)


def _var_diagnostic_records(
    frame: pd.DataFrame,
    train: np.ndarray,
    intercept: np.ndarray,
    lag1: np.ndarray,
    lag2: np.ndarray,
    moments: np.ndarray,
    *,
    common: Mapping[str, object],
    maximum_eigenvalue: float,
    calibration_intercept: float,
    calibration_slope: float,
) -> list[dict[str, object]]:
    means, scales = moments[:, 0], moments[:, 1]
    design, target, indices = _var_design(frame, train, means, scales)
    fitted = intercept + design[:, :3] @ lag1.T + design[:, 3:] @ lag2.T
    residuals = target - fitted
    continuity = _continuity_context(frame)
    diagnostic_pairs = _continuity_pairs_through(frame, indices, 20, continuity)
    records: list[dict[str, object]] = []
    component_names = ("return_1m_bps", "tsay_clock_z_range", "tsay_clock_z_volume")
    for component_index, component in enumerate(component_names):
        for diagnostic, values in (
            ("residual_ljung_box", residuals[:, component_index]),
            ("squared_residual_ljung_box", np.square(residuals[:, component_index])),
        ):
            for lag in (5, 10, 20):
                statistic, p_value = _pooled_ljung_box(
                    frame, values, indices, lag, continuity, diagnostic_pairs
                )
                records.append(
                    {
                        "diagnostic": diagnostic,
                        "component": component,
                        "lag": lag,
                        "statistic": statistic,
                        "raw_p_value": p_value,
                    }
                )
    for lag in (5, 10, 20):
        statistic, p_value = _multivariate_portmanteau(
            frame, residuals, indices, lag, continuity, diagnostic_pairs
        )
        records.append(
            {
                "diagnostic": "multivariate_portmanteau",
                "component": "joint",
                "lag": lag,
                "statistic": statistic,
                "raw_p_value": p_value,
            }
        )
    for record in _holm_adjust(records):
        record.update(common)
        record.update(
            {
                "maximum_companion_eigenvalue": maximum_eigenvalue,
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
                "forecast_failure_count": 0,
                "fit_success": True,
            }
        )
    return records


def _fit_outer_var(
    frame: pd.DataFrame,
    outer_folds: Sequence[Mapping[str, object]],
) -> tuple[
    list[pd.DataFrame],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    folds_out: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    inner_score_cache: dict[str, np.ndarray] = {}
    for outer in outer_folds:
        inner_folds = build_inner_folds(outer["train_dates"])
        scores_by_alpha: list[list[float]] = [[] for _ in RIDGE_ALPHA_GRID]
        for inner in inner_folds:
            key = str(inner["embargo_date"])
            if key in inner_score_cache:
                block = inner_score_cache[key]
                for alpha_index in range(len(RIDGE_ALPHA_GRID)):
                    scores_by_alpha[alpha_index].extend(block[:, alpha_index].tolist())
                continue
            train, assessment = _purged_masks(
                frame, inner["train_dates"], inner["assessment_dates"]
            )
            path = _fit_var_path(frame, train)
            fold_scores: list[list[float]] = [[] for _ in RIDGE_ALPHA_GRID]
            for alpha_index, alpha in enumerate(RIDGE_ALPHA_GRID):
                try:
                    intercept, lag1, lag2, _, moments = path[float(alpha)]
                    raw_train, train_indices = _var_raw_forecast(
                        frame, train, intercept, lag1, lag2, moments
                    )
                    cal_intercept, cal_slope = _fit_var_calibration(
                        frame, train, raw_train, train_indices
                    )
                    raw_assessment, indices = _var_raw_forecast(
                        frame, assessment, intercept, lag1, lag2, moments
                    )
                    scored = frame.loc[indices, ("trade_date_ny", "forward_return_60_atr")].assign(
                        prediction=cal_intercept + cal_slope * raw_assessment
                    )
                    daily = _daily_spearman(scored, "forward_return_60_atr", "prediction")
                    values = daily.get("daily_ic", pd.Series(dtype=float)).tolist()
                    scores_by_alpha[alpha_index].extend(values)
                    fold_scores[alpha_index].extend(values)
                except (KeyError, RuntimeError):
                    values = [float("nan")] * len(inner["assessment_dates"])
                    scores_by_alpha[alpha_index].extend(values)
                    fold_scores[alpha_index].extend(values)
            minimum_fold = min(len(values) for values in fold_scores)
            inner_score_cache[key] = np.column_stack(
                [np.asarray(values[:minimum_fold]) for values in fold_scores]
            )
        minimum = min(len(values) for values in scores_by_alpha)
        selected, audit = _best_one_se(
            np.column_stack([np.asarray(values[:minimum]) for values in scores_by_alpha]),
            RIDGE_ALPHA_GRID,
            maximize=True,
            strongest_largest=True,
            seed=TSAY_SEED + 90_000 + int(outer["fold_id"]),
        )
        for record in audit:
            tuning.append(
                {
                    "architecture": "D4",
                    "role": "directional",
                    "entry_session": str(frame["entry_session"].iloc[0]),
                    "outer_fold_id": int(outer["fold_id"]),
                    "score": "pooled_inner_daily_ic",
                    **record,
                }
            )
        train, assessment = _purged_masks(frame, outer["train_dates"], outer["assessment_dates"])
        intercept, lag1, lag2, eigenvalue, moments = _fit_var(frame, train, selected)
        raw_train, train_indices = _var_raw_forecast(frame, train, intercept, lag1, lag2, moments)
        cal_intercept, cal_slope = _fit_var_calibration(frame, train, raw_train, train_indices)
        raw_assessment, indices = _var_raw_forecast(
            frame, assessment, intercept, lag1, lag2, moments
        )
        scored = frame.loc[
            indices,
            (
                "observation_id",
                "trade_date_ny",
                "entry_session",
                "forward_return_60_atr",
            ),
        ].copy()
        scored["prediction"] = cal_intercept + cal_slope * raw_assessment
        scored["architecture"] = "D4"
        scored["role"] = "directional"
        scored["outer_fold_id"] = int(outer["fold_id"])
        predictions.append(scored)
        common = {
            "architecture": "D4",
            "role": "directional",
            "entry_session": str(frame["entry_session"].iloc[0]),
            "outer_fold_id": int(outer["fold_id"]),
        }
        folds_out.append(
            {
                **common,
                "train_rows_after_purge": int(train.sum()),
                "assessment_rows": len(indices),
                "assessment_dates": len(outer["assessment_dates"]),
                "inner_complete_folds": len(inner_folds),
                "selected_hyperparameter": selected,
                "fit_success": True,
            }
        )
        flattened = np.concatenate((lag1.ravel(), lag2.ravel(), [cal_slope]))
        coefficients.append(
            {
                **common,
                "component": "lag1_lag2_calibration_slope",
                "coefficient_names": json.dumps(
                    [f"var_coefficient_{index}" for index in range(len(flattened))]
                ),
                "coefficient_values": json.dumps(flattened.tolist()),
                "intercept": cal_intercept,
            }
        )
        diagnostics.extend(
            _var_diagnostic_records(
                frame,
                train,
                intercept,
                lag1,
                lag2,
                moments,
                common=common,
                maximum_eigenvalue=eigenvalue,
                calibration_intercept=cal_intercept,
                calibration_slope=cal_slope,
            )
        )
    return predictions, tuning, folds_out, coefficients, diagnostics


def _stability(audit: pd.DataFrame, architecture: str, session: str) -> tuple[float, float, int]:
    rows = audit.loc[
        audit["architecture"]
        .astype(str)
        .isin((architecture, "D5_O4") if architecture in ("D5", "O4") else (architecture,))
        & audit["entry_session"].astype(str).eq(session)
    ]
    if architecture in ("D5", "O4"):
        component_order = {"q10_slopes": 0, "q50_slopes": 1, "q90_slopes": 2}
        vectors = []
        for _, group in rows.groupby("outer_fold_id", sort=True, observed=True):
            ordered = group.assign(
                _component_order=group["component"].map(component_order)
            ).sort_values("_component_order", kind="mergesort")
            if ordered["_component_order"].isna().any() or len(ordered) != 3:
                return float("nan"), float("nan"), 0
            vectors.append(
                np.concatenate(
                    [
                        np.asarray(json.loads(value), dtype=np.float64)
                        for value in ordered["coefficient_values"]
                    ]
                )
            )
    else:
        vectors = [
            np.asarray(json.loads(value), dtype=np.float64) for value in rows["coefficient_values"]
        ]
    if len(vectors) < 2 or len({len(vector) for vector in vectors}) != 1:
        return float("nan"), float("nan"), 0
    matrix = np.vstack(vectors)
    median = np.median(matrix, axis=0)
    qualifying = np.abs(median) > 0.05
    if not qualifying.any():
        return float("nan"), float("nan"), 0
    signs = np.sign(median[qualifying])
    agreement = float(np.mean(np.sign(matrix[:, qualifying]) == signs))
    cosines: list[float] = []
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            denom = np.linalg.norm(matrix[left]) * np.linalg.norm(matrix[right])
            if denom > 0.0:
                cosines.append(float(np.dot(matrix[left], matrix[right]) / denom))
    return (
        float(np.median(cosines)) if cosines else float("nan"),
        agreement,
        int(qualifying.sum()),
    )


def _weighted_calibration(
    y: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    design = np.column_stack((np.ones(len(prediction)), prediction))
    root = np.sqrt(weights)
    coefficients = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def _continuous_evidence(
    predictions: pd.DataFrame,
    fold_audit: pd.DataFrame,
    coefficient_audit: pd.DataFrame,
    minimum_dates_by_session: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    daily_records: list[pd.DataFrame] = []
    year_records: list[dict[str, object]] = []
    for role in ("directional", "opportunity"):
        target = "forward_return_60_atr" if role == "directional" else "future_range_60_atr"
        anchor_id = CONTINUOUS_ANCHOR[role]
        for session in SESSIONS:
            anchor = predictions.loc[
                (predictions["role"] == role)
                & (predictions["architecture"] == anchor_id)
                & (predictions["entry_session"] == session),
                ["observation_id", "prediction", "outer_fold_id"],
            ].rename(columns={"prediction": "anchor_prediction", "outer_fold_id": "anchor_fold"})
            for architecture in CONTINUOUS_CANDIDATES[role]:
                candidate = predictions.loc[
                    (predictions["role"] == role)
                    & (predictions["architecture"] == architecture)
                    & (predictions["entry_session"] == session)
                ].copy()
                if candidate.empty:
                    records.append(
                        {
                            "role": role,
                            "architecture": architecture,
                            "entry_session": session,
                            "status": "NOT_AUTHORIZED" if architecture == "D6" else "NOT_EVALUABLE",
                            "failed_gates": "model_not_fitted",
                        }
                    )
                    continue
                matched = candidate.merge(
                    anchor, on="observation_id", how="inner", validate="one_to_one"
                )
                matched = matched.loc[
                    np.isfinite(matched[target])
                    & np.isfinite(matched["prediction"])
                    & np.isfinite(matched["anchor_prediction"])
                ].copy()
                if not matched["outer_fold_id"].eq(matched["anchor_fold"]).all():
                    raise RuntimeError("Candidate and anchor outer-fold identities do not match.")
                candidate_daily = _daily_spearman(matched, target, "prediction").rename(
                    columns={"daily_ic": "candidate_ic"}
                )
                anchor_daily = _daily_spearman(matched, target, "anchor_prediction").rename(
                    columns={"daily_ic": "anchor_ic"}
                )
                daily = candidate_daily.merge(
                    anchor_daily[["trade_date_ny", "anchor_ic"]],
                    on="trade_date_ny",
                    how="inner",
                    validate="one_to_one",
                )
                daily["delta_ic"] = daily["candidate_ic"] - daily["anchor_ic"]
                date_fold = matched.groupby("trade_date_ny", sort=True, observed=True)[
                    "outer_fold_id"
                ].agg(["first", "nunique"])
                date_fold.index = date_fold.index.astype(str)
                if not date_fold["nunique"].eq(1).all():
                    raise RuntimeError("A Development date spans multiple outer folds.")
                daily = daily.merge(
                    date_fold[["first"]].rename(columns={"first": "outer_fold_id"}),
                    left_on="trade_date_ny",
                    right_index=True,
                    how="left",
                    validate="one_to_one",
                )
                daily["family"] = role
                daily["architecture"] = architecture
                daily["entry_session"] = session
                daily["delta"] = daily["delta_ic"]
                daily_records.append(daily)
                weights = _date_weights(matched["trade_date_ny"])
                observed = matched[target].to_numpy(np.float64)
                predicted = matched["prediction"].to_numpy(np.float64)
                intercept, slope = _weighted_calibration(observed, predicted, weights)
                residual = observed - predicted
                baseline = observed - np.average(observed, weights=weights)
                positive = daily.loc[daily["delta_ic"] > 0.0, "delta_ic"].sort_values(
                    ascending=False
                )
                concentration = (
                    float(positive.head(10).sum() / positive.sum())
                    if positive.sum() > 0
                    else float("inf")
                )
                cosine, sign_agreement, qualifying = _stability(
                    coefficient_audit, architecture, session
                )
                candidate_folds = fold_audit.loc[
                    (
                        fold_audit["architecture"]
                        .astype(str)
                        .isin(
                            (architecture, "D5_O4")
                            if architecture in ("D5", "O4")
                            else (architecture,)
                        )
                    )
                    & (fold_audit["entry_session"] == session)
                ]
                fold_support = []
                for fold_id, group in matched.groupby("outer_fold_id", observed=True):
                    anchor_count = len(anchor.loc[anchor["anchor_fold"] == fold_id])
                    fold_support.append(len(group) / anchor_count if anchor_count else 0.0)
                minimum_rows = 2_500 if session == "London" else 4_000
                positive_fold_fraction = float(
                    (daily.groupby("outer_fold_id", observed=True)["delta_ic"].mean() > 0.0).mean()
                )
                bootstrap_lower = _bootstrap_lower(
                    daily["delta_ic"].to_numpy(), seed=TSAY_SEED + 50_000 + len(records)
                )
                gates = {
                    "mean_ic": float(daily["candidate_ic"].mean()) > 0.0,
                    "delta": float(daily["delta_ic"].mean()) >= 0.01,
                    "delta_ci": bootstrap_lower > 0.0,
                    "coverage": bool(fold_support)
                    and min(fold_support) >= 0.90
                    and len(matched) >= 10_000
                    and len(daily) >= int(minimum_dates_by_session[session])
                    and all(
                        len(group) >= minimum_rows
                        for _, group in matched.groupby("outer_fold_id", observed=True)
                    ),
                    "concentration": concentration <= 0.50,
                    "positive_folds": positive_fold_fraction >= 0.70,
                    "calibration": 0.5 <= slope <= 1.5 and abs(intercept) <= 0.1,
                    "fits": bool(len(candidate_folds))
                    and bool(candidate_folds["fit_success"].all()),
                    "coefficient_stability": qualifying > 0
                    and cosine >= 0.50
                    and sign_agreement >= 0.70,
                }
                records.append(
                    {
                        "role": role,
                        "architecture": architecture,
                        "entry_session": session,
                        "matched_rows": len(matched),
                        "matched_dates": len(daily),
                        "matched_anchor_fraction": len(matched) / len(anchor)
                        if len(anchor)
                        else 0.0,
                        "missing_prediction_count": max(0, len(anchor) - len(matched)),
                        "mean_daily_ic": float(daily["candidate_ic"].mean()),
                        "anchor_mean_daily_ic": float(daily["anchor_ic"].mean()),
                        "mean_paired_delta_ic": float(daily["delta_ic"].mean()),
                        "paired_delta_ci_lower": bootstrap_lower,
                        "weighted_mae": float(np.average(np.abs(residual), weights=weights)),
                        "weighted_rmse": float(np.sqrt(np.average(residual**2, weights=weights))),
                        "oos_r_squared": float(
                            1.0 - np.sum(weights * residual**2) / np.sum(weights * baseline**2)
                        ),
                        "calibration_intercept": intercept,
                        "calibration_slope": slope,
                        "top10_positive_delta_share": concentration,
                        "positive_outer_fold_fraction": positive_fold_fraction,
                        "coefficient_median_pairwise_cosine": cosine,
                        "coefficient_sign_agreement": sign_agreement,
                        "qualifying_coefficients": qualifying,
                        "family_max_stat_adjusted_p": float("nan"),
                        "failed_gates": "|".join(
                            name for name, passed in gates.items() if not passed
                        ),
                        "status": "PENDING_MAX_STAT" if all(gates.values()) else "DEV_REJECTED",
                    }
                )
                matched["year"] = pd.to_datetime(matched["trade_date_ny"]).dt.year
                daily["year"] = pd.to_datetime(daily["trade_date_ny"]).dt.year
                for year, year_daily in daily.groupby("year", sort=True, observed=True):
                    year_matched = matched.loc[matched["year"].eq(year)]
                    year_weights = _date_weights(year_matched["trade_date_ny"])
                    year_observed = year_matched[target].to_numpy(np.float64)
                    year_prediction = year_matched["prediction"].to_numpy(np.float64)
                    year_intercept, year_slope = _weighted_calibration(
                        year_observed, year_prediction, year_weights
                    )
                    year_residual = year_observed - year_prediction
                    year_records.append(
                        {
                            "role": role,
                            "architecture": architecture,
                            "entry_session": session,
                            "year": int(year),
                            "matched_rows": len(year_matched),
                            "matched_dates": len(year_daily),
                            "mean_daily_ic": float(year_daily["candidate_ic"].mean()),
                            "anchor_mean_daily_ic": float(year_daily["anchor_ic"].mean()),
                            "mean_paired_delta_ic": float(year_daily["delta_ic"].mean()),
                            "weighted_mae": float(
                                np.average(np.abs(year_residual), weights=year_weights)
                            ),
                            "weighted_rmse": float(
                                np.sqrt(np.average(year_residual**2, weights=year_weights))
                            ),
                            "calibration_intercept": year_intercept,
                            "calibration_slope": year_slope,
                        }
                    )
    return (
        pd.DataFrame.from_records(records),
        pd.concat(daily_records, ignore_index=True) if daily_records else pd.DataFrame(),
        pd.DataFrame.from_records(year_records),
    )


def _append_failed_gate(value: object, gate: str) -> str:
    gates = [item for item in str(value).split("|") if item and item != "nan"]
    if gate not in gates:
        gates.append(gate)
    return "|".join(gates)


def _max_stat_adjustment(
    continuous: pd.DataFrame,
    classifier: pd.DataFrame,
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    continuous = continuous.copy()
    classifier = classifier.copy()
    audit_records: list[dict[str, object]] = []
    families = {
        "directional": ("D2", "D3", "D4", "D5"),
        "opportunity": ("O2", "O3", "O4", "O5"),
        "expansion": ("E2", "E3"),
    }
    for role, architectures in families.items():
        for session in SESSIONS:
            series = []
            for architecture in architectures:
                sub = daily.loc[
                    (daily["family"] == role)
                    & (daily["entry_session"] == session)
                    & (daily["architecture"] == architecture),
                    ["trade_date_ny", "delta"],
                ].rename(columns={"delta": architecture})
                series.append(sub)
            if not series:
                continue
            common = series[0]
            for sub in series[1:]:
                common = common.merge(sub, on="trade_date_ny", how="inner", validate="one_to_one")
            if len(common) < 2:
                continue
            matrix = common.loc[:, architectures].to_numpy(np.float64)
            means = matrix.mean(axis=0)
            standard = matrix.std(axis=0, ddof=1)
            standard_error = standard / np.sqrt(len(matrix))
            evaluable = np.isfinite(standard_error) & (standard_error > 0.0)
            observed_t = np.divide(
                means,
                standard_error,
                out=np.full_like(means, np.nan),
                where=evaluable,
            )
            centered = matrix - means
            indices = stationary_bootstrap_indices(
                len(matrix),
                replicates=BOOTSTRAP_REPLICATES,
                restart_probability=BOOTSTRAP_RESTART,
                seed=TSAY_SEED
                + 60_000
                + {"directional": 0, "opportunity": 100, "expansion": 200}[role]
                + SESSIONS.index(session),
            )
            boot_mean = centered[indices].mean(axis=1)
            boot_t = np.divide(
                boot_mean,
                standard_error[None, :],
                out=np.full_like(boot_mean, np.nan),
                where=evaluable[None, :],
            )
            null_max = np.nanmax(boot_t, axis=1) if evaluable.any() else np.full(2000, np.nan)
            for index, architecture in enumerate(architectures):
                evidence = classifier if role == "expansion" else continuous
                mask = (evidence["entry_session"] == session) & (
                    evidence["architecture"] == architecture
                )
                if role != "expansion":
                    mask &= evidence["role"].eq(role)
                adjusted = (
                    float((1 + np.sum(null_max >= observed_t[index])) / (BOOTSTRAP_REPLICATES + 1))
                    if evaluable[index]
                    else float("nan")
                )
                evidence.loc[mask, "family_max_stat_adjusted_p"] = adjusted
                pending = mask & evidence["status"].eq("PENDING_MAX_STAT")
                if not evaluable[index]:
                    evidence.loc[mask, "status"] = "NOT_EVALUABLE"
                    evidence.loc[mask, "failed_gates"] = evidence.loc[mask, "failed_gates"].map(
                        lambda value: _append_failed_gate(value, "family_zero_studentizer")
                    )
                elif adjusted <= 0.05:
                    evidence.loc[pending, "status"] = "DEV_PASS"
                else:
                    evidence.loc[pending, "status"] = "DEV_REJECTED"
                    evidence.loc[mask, "failed_gates"] = evidence.loc[mask, "failed_gates"].map(
                        lambda value: _append_failed_gate(value, "family_max_stat")
                    )
                if role == "expansion":
                    classifier = evidence
                else:
                    continuous = evidence
                audit_records.append(
                    {
                        "family": role,
                        "entry_session": session,
                        "architecture": architecture,
                        "common_dates": len(common),
                        "original_standard_error": float(standard_error[index]),
                        "observed_t": float(observed_t[index]),
                        "adjusted_p_value": adjusted,
                        "fixed_original_studentizer": True,
                    }
                )
    return continuous, classifier, pd.DataFrame.from_records(audit_records)


def _quantile_summary(sub: pd.DataFrame) -> dict[str, object]:
    y = sub["forward_return_60_atr"].to_numpy(np.float64)
    weights = _date_weights(sub["trade_date_ny"])
    output: dict[str, object] = {}
    improvements = {}
    for tau in QUANTILE_LEVELS:
        name = f"q{int(tau * 100):02d}"
        baseline = f"baseline_{name}"
        model_loss = float(np.average(_pinball(y, sub[name].to_numpy(), tau), weights=weights))
        baseline_loss = float(
            np.average(_pinball(y, sub[baseline].to_numpy(), tau), weights=weights)
        )
        improvements[name] = float(1.0 - model_loss / baseline_loss)
        output[f"{name}_pinball_loss"] = model_loss
        output[f"{name}_baseline_pinball_loss"] = baseline_loss
        output[f"{name}_pinball_improvement"] = improvements[name]
        output[f"{name}_empirical_below_probability"] = float(
            np.average(y <= sub[name].to_numpy(np.float64), weights=weights)
        )
    coverage = float(
        np.average((y >= sub["q10"].to_numpy()) & (y <= sub["q90"].to_numpy()), weights=weights)
    )
    output.update(
        {
            "matched_rows": len(sub),
            "matched_dates": sub["trade_date_ny"].nunique(),
            "central_80_coverage": coverage,
            "central_80_absolute_error": abs(coverage - 0.80),
            "mean_interval_width_atr": float(
                np.average(sub["q90"].to_numpy() - sub["q10"].to_numpy(), weights=weights)
            ),
            "pre_repair_crossing_count": int(sub["pre_repair_crossing"].sum()),
            "pre_repair_crossing_rate": float(sub["pre_repair_crossing"].mean()),
            "post_repair_crossing_count": 0,
            "gate_pass": all(value >= 0.02 for value in improvements.values())
            and abs(coverage - 0.80) <= 0.03,
        }
    )
    return output


def _quantile_diagnostics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    year_records: list[dict[str, object]] = []
    for session in SESSIONS:
        sub = predictions.loc[
            (predictions["architecture"] == "D5") & (predictions["entry_session"] == session)
        ].copy()
        if sub.empty:
            continue
        records.append({"entry_session": session, **_quantile_summary(sub)})
        sub["year"] = pd.to_datetime(sub["trade_date_ny"]).dt.year
        for year, group in sub.groupby("year", sort=True, observed=True):
            year_records.append(
                {"entry_session": session, "year": int(year), **_quantile_summary(group)}
            )
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(year_records)


def _unpenalized_classifier_calibration(
    y: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    logit = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(len(logit)), logit))
    normalized = weights / weights.sum()

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ parameters
        loss = np.logaddexp(0.0, linear) - y * linear
        fitted = 1.0 / (1.0 + np.exp(-np.clip(linear, -700.0, 700.0)))
        return float(np.dot(normalized, loss)), design.T @ (normalized * (fitted - y))

    result = optimize.minimize(
        objective,
        np.array([0.0, 1.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5_000, "gtol": 1.0e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Unpenalized classifier calibration failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def _classifier_evidence(
    predictions: pd.DataFrame,
    coefficient_audit: pd.DataFrame,
    fold_audit: pd.DataFrame,
    minimum_dates_by_session: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    daily_records: list[pd.DataFrame] = []
    reliability_records: list[dict[str, object]] = []
    year_records: list[dict[str, object]] = []
    for session in SESSIONS:
        anchor = predictions.loc[
            (predictions["role"] == "expansion")
            & (predictions["architecture"] == "E1")
            & (predictions["entry_session"] == session),
            ["observation_id", "prediction", "outer_fold_id"],
        ].rename(columns={"prediction": "anchor", "outer_fold_id": "anchor_fold"})
        for architecture in ("E2", "E3"):
            candidate = predictions.loc[
                (predictions["role"] == "expansion")
                & (predictions["architecture"] == architecture)
                & (predictions["entry_session"] == session)
            ].merge(anchor, on="observation_id", how="inner", validate="one_to_one")
            if candidate.empty:
                continue
            if not candidate["outer_fold_id"].eq(candidate["anchor_fold"]).all():
                raise RuntimeError("Classifier candidate and anchor fold identities do not match.")
            y = candidate["expansion_label_60"].to_numpy(np.int8)
            p = np.clip(candidate["prediction"].to_numpy(np.float64), 1e-12, 1 - 1e-12)
            a = np.clip(candidate["anchor"].to_numpy(np.float64), 1e-12, 1 - 1e-12)
            weights = _date_weights(candidate["trade_date_ny"])
            brier_delta = (y - a) ** 2 - (y - p) ** 2
            daily = (
                pd.DataFrame({"date": candidate["trade_date_ny"].astype(str), "delta": brier_delta})
                .groupby("date", sort=True)["delta"]
                .mean()
                .rename_axis("trade_date_ny")
                .reset_index()
            )
            daily["family"] = "expansion"
            daily["architecture"] = architecture
            daily["entry_session"] = session
            daily_records.append(daily)
            order = np.lexsort((candidate["observation_id"].astype(str).to_numpy(), p))
            bins = np.empty(len(candidate), dtype=np.int8)
            bins[order] = np.minimum(
                9, np.floor(np.arange(len(candidate)) * 10 / len(candidate)).astype(np.int8)
            )
            reliability = pd.DataFrame({"bin": bins, "p": p, "y": y, "w": weights})
            reliability_rows = []
            for bin_id, group in reliability.groupby("bin", sort=True, observed=True):
                probability_mean = float(np.average(group["p"], weights=group["w"]))
                prevalence = float(np.average(group["y"], weights=group["w"]))
                reliability_rows.append(
                    {
                        "architecture": architecture,
                        "entry_session": session,
                        "equal_count_bin": int(bin_id) + 1,
                        "observation_count": len(group),
                        "date_balanced_weight": float(group["w"].sum() / weights.sum()),
                        "weighted_mean_probability": probability_mean,
                        "weighted_prevalence": prevalence,
                        "absolute_gap": abs(probability_mean - prevalence),
                    }
                )
            if len(reliability_rows) != 10:
                raise RuntimeError("Classifier reliability table has an empty frozen bin.")
            reliability_records.extend(reliability_rows)
            ece = float(
                sum(row["date_balanced_weight"] * row["absolute_gap"] for row in reliability_rows)
            )
            calibration_intercept, calibration_slope = _unpenalized_classifier_calibration(
                y, p, weights
            )
            cosine, sign_agreement, qualifying = _stability(
                coefficient_audit, architecture, session
            )
            roc = metrics.roc_auc_score(y, p, sample_weight=weights)
            anchor_roc = metrics.roc_auc_score(y, a, sample_weight=weights)
            pr = metrics.average_precision_score(y, p, sample_weight=weights)
            anchor_pr = metrics.average_precision_score(y, a, sample_weight=weights)
            brier = float(np.average((y - p) ** 2, weights=weights))
            anchor_brier = float(np.average((y - a) ** 2, weights=weights))
            log_loss = metrics.log_loss(y, p, sample_weight=weights, labels=[0, 1])
            anchor_log_loss = metrics.log_loss(y, a, sample_weight=weights, labels=[0, 1])
            daily_delta = daily["delta"].to_numpy(np.float64)
            top = np.sort(daily_delta[daily_delta > 0])[::-1]
            concentration = float(top[:10].sum() / top.sum()) if top.sum() > 0 else float("inf")
            fold_support = []
            for fold_id, group in candidate.groupby("outer_fold_id", observed=True):
                anchor_count = len(anchor.loc[anchor["anchor_fold"] == fold_id])
                fold_support.append(len(group) / anchor_count if anchor_count else 0.0)
            minimum_rows = 2_500 if session == "London" else 4_000
            candidate_folds = fold_audit.loc[
                (fold_audit["architecture"] == architecture)
                & (fold_audit["entry_session"] == session)
            ]
            bootstrap_lower = _bootstrap_lower(daily_delta, seed=TSAY_SEED + 70_000 + len(records))
            gates = {
                "support": bool(fold_support)
                and min(fold_support) >= 0.90
                and len(candidate) >= 10_000
                and candidate["trade_date_ny"].nunique() >= int(minimum_dates_by_session[session])
                and all(
                    len(group) >= minimum_rows
                    for _, group in candidate.groupby("outer_fold_id", observed=True)
                ),
                "brier": anchor_brier - brier >= 0.002,
                "brier_ci": bootstrap_lower > 0,
                "roc": roc - anchor_roc >= 0.01,
                "pr": pr >= anchor_pr,
                "log_loss": log_loss < anchor_log_loss,
                "ece": ece <= 0.05,
                "calibration": 0.75 <= calibration_slope <= 1.25
                and abs(calibration_intercept) <= 0.1,
                "concentration": concentration <= 0.5,
                "fits": bool(len(candidate_folds)) and bool(candidate_folds["fit_success"].all()),
            }
            records.append(
                {
                    "architecture": architecture,
                    "entry_session": session,
                    "matched_rows": len(candidate),
                    "matched_dates": candidate["trade_date_ny"].nunique(),
                    "matched_anchor_fraction": len(candidate) / len(anchor) if len(anchor) else 0.0,
                    "missing_prediction_count": max(0, len(anchor) - len(candidate)),
                    "roc_auc": roc,
                    "roc_auc_improvement": roc - anchor_roc,
                    "pr_auc": pr,
                    "pr_auc_improvement": pr - anchor_pr,
                    "brier": brier,
                    "brier_reduction": anchor_brier - brier,
                    "brier_reduction_ci_lower": bootstrap_lower,
                    "log_loss": log_loss,
                    "anchor_log_loss": anchor_log_loss,
                    "balanced_accuracy_0_5": metrics.balanced_accuracy_score(
                        y, p >= 0.5, sample_weight=weights
                    ),
                    "mcc_0_5": metrics.matthews_corrcoef(y, p >= 0.5, sample_weight=weights),
                    "calibration_intercept": calibration_intercept,
                    "calibration_slope": calibration_slope,
                    "ece": ece,
                    "positive_prevalence": float(np.average(y, weights=weights)),
                    "top10_positive_delta_share": concentration,
                    "coefficient_median_pairwise_cosine": cosine,
                    "coefficient_sign_agreement": sign_agreement,
                    "qualifying_coefficients": qualifying,
                    "family_max_stat_adjusted_p": float("nan"),
                    "failed_gates": "|".join(name for name, passed in gates.items() if not passed),
                    "status": "PENDING_MAX_STAT" if all(gates.values()) else "DEV_REJECTED",
                }
            )
            candidate = candidate.copy()
            candidate["year"] = pd.to_datetime(candidate["trade_date_ny"]).dt.year
            for year, group in candidate.groupby("year", sort=True, observed=True):
                year_y = group["expansion_label_60"].to_numpy(np.int8)
                year_p = np.clip(group["prediction"].to_numpy(np.float64), 1e-12, 1 - 1e-12)
                year_a = np.clip(group["anchor"].to_numpy(np.float64), 1e-12, 1 - 1e-12)
                year_weights = _date_weights(group["trade_date_ny"])
                year_records.append(
                    {
                        "architecture": architecture,
                        "entry_session": session,
                        "year": int(year),
                        "matched_rows": len(group),
                        "matched_dates": group["trade_date_ny"].nunique(),
                        "roc_auc": metrics.roc_auc_score(
                            year_y, year_p, sample_weight=year_weights
                        ),
                        "pr_auc": metrics.average_precision_score(
                            year_y, year_p, sample_weight=year_weights
                        ),
                        "brier": float(np.average((year_y - year_p) ** 2, weights=year_weights)),
                        "brier_reduction": float(
                            np.average(
                                (year_y - year_a) ** 2 - (year_y - year_p) ** 2,
                                weights=year_weights,
                            )
                        ),
                        "log_loss": metrics.log_loss(
                            year_y, year_p, sample_weight=year_weights, labels=[0, 1]
                        ),
                    }
                )
    return (
        pd.DataFrame.from_records(records),
        pd.concat(daily_records, ignore_index=True) if daily_records else pd.DataFrame(),
        pd.DataFrame.from_records(reliability_records),
        pd.DataFrame.from_records(year_records),
    )


def _select_architectures(
    continuous: pd.DataFrame, classifier: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for role, order in (
        ("directional", ("D2", "D3", "D4", "D5", "D6")),
        ("opportunity", ("O2", "O3", "O4", "O5")),
        ("expansion", ("E2", "E3")),
    ):
        evidence = classifier if role == "expansion" else continuous.loc[continuous["role"] == role]
        passing = [
            architecture
            for architecture in order
            if len(
                evidence.loc[
                    (evidence["architecture"] == architecture) & evidence["status"].eq("DEV_PASS")
                ]
            )
            == 2
        ]
        family_daily = daily.loc[daily["family"].eq(role)]
        available = [
            architecture
            for architecture in order
            if family_daily["architecture"].eq(architecture).any()
        ]
        panels = []
        for architecture in available:
            pivot = family_daily.loc[
                family_daily["architecture"].eq(architecture),
                ("trade_date_ny", "entry_session", "delta"),
            ].pivot(index="trade_date_ny", columns="entry_session", values="delta")
            if not set(SESSIONS).issubset(pivot.columns):
                continue
            pooled = pivot.loc[:, list(SESSIONS)].dropna().mean(axis=1).rename(architecture)
            panels.append(pooled)
        panel = pd.concat(panels, axis=1, join="inner").dropna() if panels else pd.DataFrame()
        selected = None
        best_architecture = None
        best_mean = float("nan")
        best_standard_error = float("nan")
        selection_threshold = float("nan")
        selected_mean = float("nan")
        if passing:
            if panel.empty or not set(passing).issubset(panel.columns):
                raise RuntimeError(f"{role} architecture selection lacks its common date panel.")
            means = panel.loc[:, passing].mean(axis=0)
            best_architecture = str(means.idxmax())
            best_mean = float(means[best_architecture])
            indices = stationary_bootstrap_indices(
                len(panel),
                replicates=BOOTSTRAP_REPLICATES,
                restart_probability=BOOTSTRAP_RESTART,
                seed=TSAY_SEED + 75_000 + ("directional", "opportunity", "expansion").index(role),
            )
            bootstrap_means = panel[best_architecture].to_numpy(np.float64)[indices].mean(axis=1)
            best_standard_error = float(np.std(bootstrap_means, ddof=1))
            selection_threshold = best_mean - best_standard_error
            if best_standard_error == 0.0:
                eligible = {
                    architecture
                    for architecture in passing
                    if abs(float(means[architecture]) - best_mean) <= 1.0e-12
                }
            else:
                eligible = {
                    architecture
                    for architecture in passing
                    if float(means[architecture]) >= selection_threshold
                }
            selected = next(architecture for architecture in order if architecture in eligible)
            selected_mean = float(means[selected])
        records.append(
            {
                "role": role,
                "selected_architecture": selected,
                "fully_passing_architectures": json.dumps(passing),
                "selection_status": "SELECTED" if selected else "NO_MODEL_PASSED",
                "shared_architecture_across_session_fits": True,
                "family_common_dates": len(panel),
                "best_architecture": best_architecture,
                "best_mean_two_session_improvement": best_mean,
                "best_bootstrap_standard_error": best_standard_error,
                "one_standard_error_threshold": selection_threshold,
                "selected_mean_two_session_improvement": selected_mean,
            }
        )
    return pd.DataFrame.from_records(records)


def _baseline_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for session in SESSIONS:
        for architecture, role, target, forecast_kind in (
            ("D0_ZERO", "directional", "forward_return_60_atr", "zero_forecast"),
            ("D0", "directional", "forward_return_60_atr", "fold_training_mean_forecast"),
            ("D1", "directional", "forward_return_60_atr", "directional_anchor_ridge"),
            ("O0", "opportunity", "future_range_60_atr", "atr_20_forecast"),
            ("O1", "opportunity", "future_range_60_atr", "opportunity_anchor_ridge"),
        ):
            sub = predictions.loc[
                (predictions["architecture"] == architecture)
                & (predictions["entry_session"] == session)
            ].copy()
            sub = sub.loc[
                np.isfinite(pd.to_numeric(sub[target], errors="coerce"))
                & np.isfinite(pd.to_numeric(sub["prediction"], errors="coerce"))
            ]
            if sub.empty:
                continue
            observed = sub[target].to_numpy(np.float64)
            forecast = sub["prediction"].to_numpy(np.float64)
            weights = _date_weights(sub["trade_date_ny"])
            residual = observed - forecast
            centered = observed - np.average(observed, weights=weights)
            calibration_intercept, calibration_slope = _weighted_calibration(
                observed, forecast, weights
            )
            daily = _daily_spearman(sub, target, "prediction")
            records.append(
                {
                    "architecture": architecture,
                    "role": role,
                    "entry_session": session,
                    "forecast_kind": forecast_kind,
                    "matched_rows": len(sub),
                    "matched_dates": sub["trade_date_ny"].nunique(),
                    "weighted_mae": float(np.average(np.abs(residual), weights=weights)),
                    "weighted_rmse": float(np.sqrt(np.average(residual**2, weights=weights))),
                    "oos_r_squared": float(
                        1.0 - np.sum(weights * residual**2) / np.sum(weights * centered**2)
                    ),
                    "mean_daily_ic": float(daily["daily_ic"].mean())
                    if len(daily)
                    else float("nan"),
                    "calibration_intercept": calibration_intercept,
                    "calibration_slope": calibration_slope,
                }
            )
        for architecture, forecast_kind in (
            ("E0", "fold_training_prevalence"),
            ("E1", "expansion_anchor_logistic"),
        ):
            sub = predictions.loc[
                (predictions["architecture"] == architecture)
                & (predictions["entry_session"] == session)
            ].copy()
            sub = sub.loc[
                np.isfinite(pd.to_numeric(sub["expansion_label_60"], errors="coerce"))
                & np.isfinite(pd.to_numeric(sub["prediction"], errors="coerce"))
            ]
            if sub.empty:
                continue
            y = sub["expansion_label_60"].to_numpy(np.int8)
            probability = np.clip(sub["prediction"].to_numpy(np.float64), 1e-12, 1 - 1e-12)
            weights = _date_weights(sub["trade_date_ny"])
            records.append(
                {
                    "architecture": architecture,
                    "role": "expansion",
                    "entry_session": session,
                    "forecast_kind": forecast_kind,
                    "matched_rows": len(sub),
                    "matched_dates": sub["trade_date_ny"].nunique(),
                    "weighted_brier": float(np.average((y - probability) ** 2, weights=weights)),
                    "weighted_log_loss": metrics.log_loss(
                        y, probability, sample_weight=weights, labels=[0, 1]
                    ),
                    "weighted_roc_auc": metrics.roc_auc_score(
                        y, probability, sample_weight=weights
                    ),
                    "weighted_pr_auc": metrics.average_precision_score(
                        y, probability, sample_weight=weights
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def run_stage4_models(
    frame: pd.DataFrame,
    fold_memberships: Mapping[str, object],
) -> Stage4ModelResult:
    """Fit and evaluate the frozen nested ladder on Development only."""

    if set(frame["research_partition"].astype(str).unique()) != {"Development"}:
        raise PermissionError("Stage 4 model input must be Development-only.")
    if pd.to_datetime(frame["trade_date_ny"]).max() > pd.Timestamp("2023-12-31"):
        raise PermissionError("Stage 4 model input contains a post-2023 row.")
    predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    folds_out: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    var_diagnostics: list[dict[str, object]] = []
    kalman_diagnostics: list[dict[str, object]] = []
    for session in SESSIONS:
        session_frame = (
            frame.loc[frame["entry_session"].astype(str).eq(session)]
            .sort_values("decision_timestamp_utc", kind="mergesort")
            .reset_index(drop=True)
        )
        outer = fold_memberships["sessions"][session]["folds"]
        for architecture, role, target, columns in (
            ("D1", "directional", "forward_return_60_atr", D1_INPUTS_BY_SESSION[session]),
            ("D2", "directional", "forward_return_60_atr", D2_INPUTS),
            ("D3", "directional", "forward_return_60_atr", D3_INPUTS_BY_SESSION[session]),
            ("O1", "opportunity", "future_range_60_atr", O1_INPUTS_BY_SESSION[session]),
            ("O2", "opportunity", "future_range_60_atr", O2_INPUTS),
            ("O3", "opportunity", "future_range_60_atr", O3_INPUTS_BY_SESSION[session]),
        ):
            result = _fit_outer_ridge(
                session_frame,
                architecture=architecture,
                role=role,
                target=target,
                columns=columns,
                outer_folds=outer,
            )
            predictions.extend(result[0])
            tuning.extend(result[1])
            folds_out.extend(result[2])
            coefficients.extend(result[3])
        predictions.extend(
            _fit_outer_baseline(
                session_frame,
                architecture="D0",
                role="directional",
                target="forward_return_60_atr",
                outer_folds=outer,
            )
        )
        predictions.extend(
            _fit_outer_baseline(
                session_frame,
                architecture="D0_ZERO",
                role="directional",
                target="forward_return_60_atr",
                outer_folds=outer,
            )
        )
        predictions.extend(
            _fit_outer_baseline(
                session_frame,
                architecture="O0",
                role="opportunity",
                target="future_range_60_atr",
                outer_folds=outer,
            )
        )
        quantile_result = _fit_outer_quantiles(
            session_frame, outer, QUANTILE_INPUTS_BY_SESSION[session]
        )
        predictions.extend(quantile_result[0])
        tuning.extend(quantile_result[1])
        folds_out.extend(quantile_result[2])
        coefficients.extend(quantile_result[3])
        var_result = _fit_outer_var(session_frame, outer)
        predictions.extend(var_result[0])
        tuning.extend(var_result[1])
        folds_out.extend(var_result[2])
        coefficients.extend(var_result[3])
        var_diagnostics.extend(var_result[4])
        kalman_result = _fit_outer_kalman_ridge(session_frame, outer)
        predictions.extend(kalman_result[0])
        tuning.extend(kalman_result[1])
        folds_out.extend(kalman_result[2])
        coefficients.extend(kalman_result[3])
        kalman_diagnostics.extend(kalman_result[4])
        for architecture, columns in (
            ("E0", None),
            ("E1", O1_INPUTS_BY_SESSION[session]),
            ("E2", O2_INPUTS),
            ("E3", O3_INPUTS_BY_SESSION[session]),
        ):
            result = _fit_outer_logistic(
                session_frame, architecture=architecture, columns=columns, outer_folds=outer
            )
            predictions.extend(result[0])
            tuning.extend(result[1])
            folds_out.extend(result[2])
            coefficients.extend(result[3])
    oof = pd.concat(predictions, ignore_index=True, sort=False)
    tuning_frame = pd.DataFrame.from_records(tuning)
    fold_frame = pd.DataFrame.from_records(folds_out)
    coefficient_frame = pd.DataFrame.from_records(coefficients)
    minimum_dates = {
        session: int(fold_memberships["sessions"][session]["minimum_dev_oof_dates"])
        for session in SESSIONS
    }
    continuous, continuous_daily, continuous_year = _continuous_evidence(
        oof, fold_frame, coefficient_frame, minimum_dates
    )
    quantile, quantile_year = _quantile_diagnostics(oof)
    for architecture in ("D5", "O4"):
        for session in SESSIONS:
            q_pass = bool(quantile.loc[quantile["entry_session"] == session, "gate_pass"].iloc[0])
            mask = (continuous["architecture"] == architecture) & (
                continuous["entry_session"] == session
            )
            if not q_pass:
                continuous.loc[mask, "status"] = "DEV_REJECTED"
                continuous.loc[mask, "failed_gates"] = continuous.loc[mask, "failed_gates"].map(
                    lambda value: _append_failed_gate(value, "quantile_gates")
                )
    classifier, classifier_daily, reliability, classifier_year = _classifier_evidence(
        oof, coefficient_frame, fold_frame, minimum_dates
    )
    daily = pd.concat((continuous_daily, classifier_daily), ignore_index=True, sort=False)
    continuous, classifier, max_stat = _max_stat_adjustment(continuous, classifier, daily)
    selection = _select_architectures(continuous, classifier, daily)
    deployment = {
        "format": "canonical-json-v1",
        "seed": TSAY_SEED,
        "fold_membership_sha256": fold_memberships["fold_membership_sha256"],
        "oof_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(
                oof.sort_values(
                    ["role", "architecture", "entry_session", "observation_id"], kind="mergesort"
                ),
                index=False,
            ).values.tobytes()
        ).hexdigest(),
        "selected": {
            row.role: row.selected_architecture for row in selection.itertuples(index=False)
        },
        "note": "No deployment object is fit for a role whose Development ladder selects no architecture.",
    }
    return Stage4ModelResult(
        oof_predictions=oof,
        tuning_audit=tuning_frame,
        fold_audit=fold_frame,
        coefficient_audit=coefficient_frame,
        continuous_evidence=continuous,
        classifier_evidence=classifier,
        daily_improvements=daily,
        baseline_diagnostics=_baseline_diagnostics(oof),
        continuous_year_evidence=continuous_year,
        classifier_reliability=reliability,
        classifier_year_evidence=classifier_year,
        quantile_diagnostics=quantile,
        quantile_year_diagnostics=quantile_year,
        var_diagnostics=pd.DataFrame.from_records(var_diagnostics),
        kalman_diagnostics=pd.DataFrame.from_records(kalman_diagnostics),
        max_stat_audit=max_stat,
        architecture_selection=selection,
        deployment_objects=deployment,
    )
