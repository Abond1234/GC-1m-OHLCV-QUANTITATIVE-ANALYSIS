"""Development-only nested modeling and policy freeze for Project 1.

This module implements Notebook Section 5 as an additive research layer.  Its
loaders are intentionally incapable of returning retrospective Validation or
Historical Final outcomes.  Every learned transform, target threshold, model
hyperparameter, and profile decomposition is fitted inside the applicable
training fold.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
import sklearn
from scipy import stats as scipy_stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge, enet_path
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import PowerTransformer, SplineTransformer

from .fes_project1_config import (
    D1_FEATURES,
    INTERACTION_SPECS,
    O1_FEATURES,
    build_frozen_config,
    canonical_json,
    config_sha256,
    sha256_file,
)
from .fes_project1_features import BASE_FEATURE_NAMES, INTERACTION_FEATURE_NAMES
from .fes_project1_profiles import (
    P0_COLUMNS,
    PROFILE_REPRESENTATIONS,
    REPRESENTATION_COLUMNS,
    transform_profile_representation,
)

DEVELOPMENT_PARTITION: Final = "Development"
SESSIONS: Final = ("London", "New York")
CONTINUOUS_TARGETS: Final = (
    "forward_return_60_atr",
    "forward_return_30_atr",
    "future_range_60_atr",
)
PRIMARY_CONTINUOUS_TARGETS: Final = (
    "forward_return_60_atr",
    "future_range_60_atr",
)
TARGET_HORIZON: Final[Mapping[str, int]] = {
    "forward_return_60_atr": 60,
    "forward_return_30_atr": 30,
    "future_range_60_atr": 60,
}
TARGET_TICK_COLUMN: Final[Mapping[str, str]] = {
    "forward_return_60_atr": "forward_return_60_ticks",
    "forward_return_30_atr": "forward_return_30_ticks",
    "future_range_60_atr": "future_range_60_ticks",
}
TARGET_AVAILABILITY: Final[Mapping[str, str]] = {
    "forward_return_60_atr": "label_available_60",
    "forward_return_30_atr": "label_available_30",
    "future_range_60_atr": "label_available_60",
}

RIDGE_ALPHA_GRID: Final = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
ELASTIC_ALPHA_GRID: Final = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
ELASTIC_L1_RATIO_GRID: Final = (0.1, 0.5, 0.9, 1.0)
PCA_COMPONENT_GRID: Final = (3, 5, 8, 12)
PLS_COMPONENT_GRID: Final = (2, 3, 5, 8)
LOGISTIC_C_GRID: Final = (0.01, 0.1, 1.0, 10.0)

SPLINE_FEATURES: Final = (
    "ret_tail_balance_60",
    "price_path_curvature_30_atr",
    "ordered_draw_balance_30",
    "return_acf_energy_60",
    "distance_from_research_day_vwap_atr",
    "efficiency_ratio_30",
)
DEGENERATE_IMPUTABLE: Final = (
    "return_acf_energy_60",
    "lagged_volume_return_spearman_30",
    "range_volume_spearman_30",
)
DEGENERATE_INDICATORS: Final = tuple(
    f"{name}__degenerate" for name in DEGENERATE_IMPUTABLE
)
INTERACTION_PARENT_MAP: Final[Mapping[str, tuple[str, ...]]] = {
    str(item["name"]): tuple(str(parent) for parent in item.get("parents", ()))
    for item in INTERACTION_SPECS
    if str(item["id"]).startswith("I")
}

M2_FEATURES: Final = tuple(BASE_FEATURE_NAMES)
M3_DIRECTION_FEATURES: Final = tuple(
    dict.fromkeys(
        [
            *D1_FEATURES,
            *BASE_FEATURE_NAMES,
            "efficiency_ratio_30",
            "relative_volume_20_clipped",
            *INTERACTION_FEATURE_NAMES,
        ]
    )
)
M3_OPPORTUNITY_FEATURES: Final = tuple(
    dict.fromkeys([*O1_FEATURES, *BASE_FEATURE_NAMES])
)
ALL_SCALAR_INPUTS: Final = tuple(
    dict.fromkeys(
        [
            *M3_DIRECTION_FEATURES,
            *M3_OPPORTUNITY_FEATURES,
            *SPLINE_FEATURES,
            *DEGENERATE_INDICATORS,
        ]
    )
)


@dataclass(frozen=True)
class Section5Config:
    """Frozen numerical settings for Development nested modeling."""

    random_seed: int = 20260726
    bootstrap_replicates: int = 2_000
    bootstrap_confidence: float = 0.95
    min_daily_observations: int = 10
    outer_initial_train_end: int = 251
    outer_assessment_dates: int = 63
    outer_step_dates: int = 64
    inner_initial_train_end: int = 125
    inner_assessment_dates: int = 21
    inner_step_dates: int = 22
    min_valid_inner_folds: int = 2
    ridge_alpha_grid: tuple[float, ...] = RIDGE_ALPHA_GRID
    elastic_alpha_grid: tuple[float, ...] = ELASTIC_ALPHA_GRID
    elastic_l1_ratio_grid: tuple[float, ...] = ELASTIC_L1_RATIO_GRID
    pca_component_grid: tuple[int, ...] = PCA_COMPONENT_GRID
    pls_component_grid: tuple[int, ...] = PLS_COMPONENT_GRID
    logistic_c_grid: tuple[float, ...] = LOGISTIC_C_GRID
    expansion_quantile: float = 0.80
    direction_score_quantile: float = 0.90
    random_subset_draws: int = 100
    selection_frequency_min: float = 0.60
    sign_agreement_min: float = 0.75
    cluster_abs_spearman: float = 0.80
    modal_complexity_fraction_min: float = 0.60
    component_within_two_fraction_min: float = 0.75
    development_oof_dates_min: int = 400


@dataclass(frozen=True)
class DateFold:
    """One exact chronological date fold and its canonical membership hash."""

    fold_id: int
    train_dates: tuple[pd.Timestamp, ...]
    embargo_date: pd.Timestamp
    assessment_dates: tuple[pd.Timestamp, ...]
    fold_sha256: str

    def to_record(self, *, level: str, parent_outer_fold_id: int | None) -> dict[str, Any]:
        return {
            "level": level,
            "parent_outer_fold_id": parent_outer_fold_id,
            "fold_id": self.fold_id,
            "train_date_count": len(self.train_dates),
            "train_first_date": self.train_dates[0].strftime("%Y-%m-%d"),
            "train_last_date": self.train_dates[-1].strftime("%Y-%m-%d"),
            "embargo_date": self.embargo_date.strftime("%Y-%m-%d"),
            "assessment_date_count": len(self.assessment_dates),
            "assessment_first_date": self.assessment_dates[0].strftime("%Y-%m-%d"),
            "assessment_last_date": self.assessment_dates[-1].strftime("%Y-%m-%d"),
            "train_dates": "|".join(value.strftime("%Y-%m-%d") for value in self.train_dates),
            "assessment_dates": "|".join(
                value.strftime("%Y-%m-%d") for value in self.assessment_dates
            ),
            "fold_sha256": self.fold_sha256,
        }


@dataclass
class ScalarTransformState:
    """Serializable fold-local S0/S1/S2 transform state."""

    preprocessing: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    imputation_medians: dict[str, float]
    constant_flags: dict[str, bool]
    means: dict[str, float]
    scales: dict[str, float]
    power_lambdas: dict[str, float]
    boolean_columns: tuple[str, ...]
    spline_features: tuple[str, ...]
    spline_knots: dict[str, list[list[float]]]
    spline_means: dict[str, list[float]]
    spline_scales: dict[str, list[float]]
    version: str = "sklearn-1.9.0-contract"


@dataclass
class ScalarFoldTransformer(TransformerMixin, BaseEstimator):
    """Exact S0/S1/S2 fold-local scalar preprocessing.

    Only F07-F09 values explicitly marked ``DEGENERATE_STATISTIC`` may be
    median-imputed.  Their indicators are fixed Boolean inputs.  All other
    non-finite values are a support-mask error.
    """

    preprocessing: str = "S0"
    spline_features: tuple[str, ...] = SPLINE_FEATURES

    def fit(self, X: pd.DataFrame, y: object = None):
        del y
        mode = str(self.preprocessing).upper()
        if mode not in {"S0", "S1", "S2"}:
            raise ValueError(f"Unknown scalar preprocessing {mode!r}.")
        self.preprocessing_ = mode
        self.input_columns_ = tuple(str(name) for name in X.columns)
        self.boolean_columns_ = tuple(
            name for name in self.input_columns_ if name.endswith("__degenerate")
        )
        work = X.copy()
        self.imputation_medians_: dict[str, float] = {}
        for feature in DEGENERATE_IMPUTABLE:
            if feature not in work.columns:
                continue
            indicator = f"{feature}__degenerate"
            invalid = ~np.isfinite(work[feature].to_numpy(dtype=np.float64))
            permitted = (
                work[indicator].astype(bool).to_numpy()
                if indicator in work.columns
                else np.zeros(len(work), dtype=bool)
            )
            if np.any(invalid & ~permitted):
                raise ValueError(
                    f"{feature} has non-finite values without its degenerate indicator."
                )
            finite = work.loc[~invalid, feature].to_numpy(dtype=np.float64)
            median = float(np.median(finite)) if len(finite) else 0.0
            self.imputation_medians_[feature] = median
            work.loc[invalid, feature] = median
        numeric = work.astype({name: "float64" for name in work.columns})
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            missing = numeric.columns[~np.isfinite(values).all(axis=0)].tolist()
            raise ValueError(f"Scalar support contains prohibited non-finite values: {missing}")

        self.constant_flags_ = {
            name: bool(np.ptp(values[:, index]) == 0.0)
            for index, name in enumerate(self.input_columns_)
        }
        bypass = set(self.boolean_columns_) | {
            name for name, constant in self.constant_flags_.items() if constant
        }
        self.power_columns_ = tuple(
            name for name in self.input_columns_ if name not in bypass
        )
        self.power_transformer_ = None
        powered = values.copy()
        if mode == "S1" and self.power_columns_:
            positions = [self.input_columns_.index(name) for name in self.power_columns_]
            self.power_transformer_ = PowerTransformer(
                method="yeo-johnson",
                standardize=False,
            ).fit(values[:, positions])
            powered[:, positions] = self.power_transformer_.transform(
                values[:, positions]
            )

        self.means_array_ = np.mean(powered, axis=0)
        self.scales_array_ = np.std(powered, axis=0, ddof=0)
        constant_positions = np.asarray(
            [self.constant_flags_[name] for name in self.input_columns_],
            dtype=bool,
        )
        boolean_positions = np.asarray(
            [name in self.boolean_columns_ for name in self.input_columns_],
            dtype=bool,
        )
        self.means_array_[constant_positions] = powered[0, constant_positions]
        self.scales_array_[constant_positions] = 1.0
        self.means_array_[boolean_positions & ~constant_positions] = 0.0
        self.scales_array_[boolean_positions & ~constant_positions] = 1.0
        self.scales_array_[self.scales_array_ == 0.0] = 1.0

        self.spline_transformers_: dict[str, SplineTransformer] = {}
        self.spline_means_: dict[str, np.ndarray] = {}
        self.spline_scales_: dict[str, np.ndarray] = {}
        output_columns = list(self.input_columns_)
        if mode == "S2":
            for name in self.spline_features:
                if name not in self.input_columns_:
                    raise ValueError(f"S2 requires frozen spline feature {name}.")
                position = self.input_columns_.index(name)
                spline = SplineTransformer(
                    n_knots=4,
                    degree=3,
                    knots="quantile",
                    extrapolation="linear",
                    include_bias=False,
                    order="C",
                ).fit(powered[:, [position]])
                self.spline_transformers_[name] = spline
                basis = spline.transform(powered[:, [position]])
                basis_mean = basis.mean(axis=0)
                basis_scale = basis.std(axis=0, ddof=0)
                basis_scale[basis_scale == 0.0] = 1.0
                self.spline_means_[name] = basis_mean
                self.spline_scales_[name] = basis_scale
                output_columns.extend(
                    f"{name}__spline_{index:02d}"
                    for index in range(spline.n_features_out_)
                )
        self.output_columns_ = tuple(output_columns)
        self.fit_row_count_ = len(work)
        return self

    def _clean_values(self, X: pd.DataFrame) -> np.ndarray:
        if tuple(str(name) for name in X.columns) != self.input_columns_:
            raise ValueError("Scalar column order differs from the fitted contract.")
        work = X.copy()
        for feature, median in self.imputation_medians_.items():
            invalid = ~np.isfinite(work[feature].to_numpy(dtype=np.float64))
            indicator = f"{feature}__degenerate"
            permitted = (
                work[indicator].astype(bool).to_numpy()
                if indicator in work.columns
                else np.zeros(len(work), dtype=bool)
            )
            if np.any(invalid & ~permitted):
                raise ValueError(
                    f"{feature} has non-finite assessment values without a flag."
                )
            work.loc[invalid, feature] = median
        values = work.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Scalar assessment matrix contains prohibited non-finite values.")
        return values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "output_columns_"):
            raise ValueError("ScalarFoldTransformer is not fitted.")
        values = self._clean_values(X)
        powered = values.copy()
        if self.power_transformer_ is not None:
            positions = [self.input_columns_.index(name) for name in self.power_columns_]
            powered[:, positions] = self.power_transformer_.transform(
                values[:, positions]
            )
        scaled = (powered - self.means_array_) / self.scales_array_
        for index, name in enumerate(self.input_columns_):
            if self.constant_flags_[name]:
                scaled[:, index] = 0.0
        if self.preprocessing_ != "S2":
            return scaled
        bases = [scaled]
        for name in self.spline_features:
            position = self.input_columns_.index(name)
            basis = self.spline_transformers_[name].transform(powered[:, [position]])
            bases.append(
                (basis - self.spline_means_[name]) / self.spline_scales_[name]
            )
        return np.hstack(bases)

    def get_state(self) -> ScalarTransformState:
        if not hasattr(self, "output_columns_"):
            raise ValueError("ScalarFoldTransformer is not fitted.")
        lambdas = {}
        if self.power_transformer_ is not None:
            lambdas = {
                name: float(value)
                for name, value in zip(
                    self.power_columns_,
                    self.power_transformer_.lambdas_,
                    strict=True,
                )
            }
        knots = {
            name: np.asarray(transformer.bsplines_[0].t, dtype=np.float64)
            .reshape(1, -1)
            .tolist()
            for name, transformer in self.spline_transformers_.items()
        }
        return ScalarTransformState(
            preprocessing=self.preprocessing_,
            input_columns=self.input_columns_,
            output_columns=self.output_columns_,
            imputation_medians=dict(self.imputation_medians_),
            constant_flags=dict(self.constant_flags_),
            means={
                name: float(value)
                for name, value in zip(
                    self.input_columns_, self.means_array_, strict=True
                )
            },
            scales={
                name: float(value)
                for name, value in zip(
                    self.input_columns_, self.scales_array_, strict=True
                )
            },
            power_lambdas=lambdas,
            boolean_columns=self.boolean_columns_,
            spline_features=(
                tuple(self.spline_features) if self.preprocessing_ == "S2" else ()
            ),
            spline_knots=knots,
            spline_means={
                name: values.astype(float).tolist()
                for name, values in self.spline_means_.items()
            },
            spline_scales={
                name: values.astype(float).tolist()
                for name, values in self.spline_scales_.items()
            },
        )


@dataclass(frozen=True)
class DevelopmentModelInputs:
    """Projected Development-only tables needed by Section 5."""

    frame: pd.DataFrame
    raw_profiles: pd.DataFrame
    support_coverage: pd.DataFrame
    access_audit: pd.DataFrame


@dataclass
class FittedContinuousModel:
    """Serializable frozen continuous estimator and its transforms."""

    session: str
    target: str
    model_family: str
    configuration: dict[str, Any]
    scalar_transformer: ScalarFoldTransformer | None
    profile_transformer: Any | None
    ridge_model: Ridge | None
    null_mean: float | None
    scalar_columns: tuple[str, ...]
    profile_representation: str | None
    component_count: int | None
    input_schema_sha256: str

    def predict(self, scalar: pd.DataFrame, profile_p0: np.ndarray | None) -> np.ndarray:
        if self.ridge_model is None and self.null_mean is not None:
            return np.full(len(scalar), float(self.null_mean), dtype=np.float64)
        blocks: list[np.ndarray] = []
        if self.scalar_columns:
            if self.scalar_transformer is None:
                raise ValueError("Frozen model is missing its scalar transformer.")
            blocks.append(
                self.scalar_transformer.transform(
                    scalar.loc[:, list(self.scalar_columns)]
                )
            )
        if self.profile_transformer is not None:
            if profile_p0 is None:
                raise ValueError("Frozen profile model requires P0 inputs.")
            blocks.append(self.profile_transformer.transform(profile_p0))
        if not blocks or self.ridge_model is None:
            raise ValueError("Frozen continuous model is incomplete.")
        return self.ridge_model.predict(np.hstack(blocks))


@dataclass
class FittedExpansionModel:
    """Serializable frozen expansion classifier and nested label threshold."""

    session: str
    model_family: str
    configuration: dict[str, Any]
    scalar_transformer: ScalarFoldTransformer | None
    profile_transformer: Any | None
    logistic_model: LogisticRegression | None
    null_prevalence: float | None
    scalar_columns: tuple[str, ...]
    expansion_range_threshold: float
    input_schema_sha256: str

    def predict_proba(
        self, scalar: pd.DataFrame, profile_p0: np.ndarray | None
    ) -> np.ndarray:
        if self.logistic_model is None:
            return np.full(len(scalar), float(self.null_prevalence), dtype=np.float64)
        blocks: list[np.ndarray] = []
        if self.scalar_columns:
            if self.scalar_transformer is None:
                raise ValueError("Frozen classifier is missing its scalar transformer.")
            blocks.append(
                self.scalar_transformer.transform(
                    scalar.loc[:, list(self.scalar_columns)]
                )
            )
        if self.profile_transformer is not None:
            if profile_p0 is None:
                raise ValueError("Frozen profile classifier requires P0 inputs.")
            blocks.append(self.profile_transformer.transform(profile_p0))
        return self.logistic_model.predict_proba(np.hstack(blocks))[:, 1]


@dataclass(frozen=True)
class Section5Result:
    """In-memory Section 5 result tables and frozen estimators."""

    fold_table: pd.DataFrame
    threshold_audit: pd.DataFrame
    support_coverage: pd.DataFrame
    continuous_oof_predictions: pd.DataFrame
    continuous_model_comparison: pd.DataFrame
    continuous_paired_daily: pd.DataFrame
    continuous_metrics: pd.DataFrame
    continuous_deciles: pd.DataFrame
    continuous_subperiods: pd.DataFrame
    outer_choices: pd.DataFrame
    elastic_stability: pd.DataFrame
    elastic_cluster_stability: pd.DataFrame
    elastic_member_substitution: pd.DataFrame
    random_subset_diagnostics: pd.DataFrame
    profile_diagnostics: pd.DataFrame
    expansion_oof_predictions: pd.DataFrame
    expansion_metrics: pd.DataFrame
    expansion_reliability: pd.DataFrame
    frozen_thresholds: pd.DataFrame
    frozen_model_registry: pd.DataFrame
    policy_freeze: dict[str, Any]
    complete_trial_ledger: pd.DataFrame
    access_audit: pd.DataFrame
    frozen_continuous_models: Mapping[str, FittedContinuousModel]
    frozen_expansion_models: Mapping[str, FittedExpansionModel]
    config: Section5Config = field(default_factory=Section5Config)


def _canonical_fold_hash(
    train_dates: Sequence[pd.Timestamp],
    embargo_date: pd.Timestamp,
    assessment_dates: Sequence[pd.Timestamp],
) -> str:
    payload = {
        "train_dates": [value.strftime("%Y-%m-%d") for value in train_dates],
        "embargo_date": embargo_date.strftime("%Y-%m-%d"),
        "assessment_dates": [
            value.strftime("%Y-%m-%d") for value in assessment_dates
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_exact_date_folds(
    dates: Iterable[object],
    *,
    initial_train_end: int,
    assessment_dates: int,
    step_dates: int,
) -> list[DateFold]:
    """Build exact complete expanding date folds with a one-date embargo."""

    unique = (
        pd.DatetimeIndex(pd.to_datetime(list(dates)))
        .normalize()
        .unique()
        .sort_values()
    )
    folds: list[DateFold] = []
    train_end = int(initial_train_end)
    while train_end + 2 + assessment_dates <= len(unique):
        train = tuple(pd.Timestamp(value) for value in unique[: train_end + 1])
        embargo = pd.Timestamp(unique[train_end + 1])
        assessment = tuple(
            pd.Timestamp(value)
            for value in unique[train_end + 2 : train_end + 2 + assessment_dates]
        )
        folds.append(
            DateFold(
                fold_id=len(folds),
                train_dates=train,
                embargo_date=embargo,
                assessment_dates=assessment,
                fold_sha256=_canonical_fold_hash(train, embargo, assessment),
            )
        )
        train_end += int(step_dates)
    return folds


def purged_fold_masks(
    frame: pd.DataFrame,
    fold: DateFold,
    *,
    exit_timestamp_column: str,
) -> tuple[np.ndarray, np.ndarray, int, pd.Timestamp]:
    """Return exact train/assessment row masks and target-exit purge count."""

    dates = pd.to_datetime(frame["trade_date_ny"]).dt.normalize()
    assessment_base = dates.isin(fold.assessment_dates).to_numpy()
    if not assessment_base.any():
        raise ValueError(f"Fold {fold.fold_id} has no assessment rows.")
    first_assessment_decision = pd.to_datetime(
        frame.loc[assessment_base, "decision_timestamp_utc"], utc=True
    ).min()
    train_base = dates.isin(fold.train_dates).to_numpy()
    exits = pd.to_datetime(frame[exit_timestamp_column], utc=True)
    purge = train_base & (exits >= first_assessment_decision).to_numpy()
    train = train_base & ~purge
    return train, assessment_base, int(purge.sum()), pd.Timestamp(
        first_assessment_decision
    )


def nested_expansion_labels(
    future_range: np.ndarray,
    train_mask: np.ndarray,
    assessment_mask: np.ndarray,
    *,
    quantile: float = 0.80,
) -> tuple[np.ndarray, float]:
    """Construct assessment labels from a training-only linear quantile."""

    values = np.asarray(future_range, dtype=np.float64)
    train_values = values[train_mask & np.isfinite(values)]
    if len(train_values) == 0:
        raise ValueError("Expansion threshold training support is empty.")
    threshold = float(np.quantile(train_values, quantile, method="linear"))
    labels = np.full(len(values), np.nan, dtype=np.float64)
    eligible = assessment_mask & np.isfinite(values)
    labels[eligible] = (values[eligible] >= threshold).astype(np.float64)
    return labels, threshold


def hierarchy_closed_support(columns: Iterable[str]) -> tuple[str, ...]:
    """Apply exact frozen interaction parent closure and deduplicate by name."""

    ordered = list(dict.fromkeys(str(name) for name in columns))
    pending = list(ordered)
    while pending:
        name = pending.pop(0)
        for parent in INTERACTION_PARENT_MAP.get(name, ()):
            if parent not in ordered:
                ordered.append(parent)
                pending.append(parent)
    return tuple(ordered)


def assert_hierarchical_closure(columns: Iterable[str]) -> None:
    observed = tuple(dict.fromkeys(str(name) for name in columns))
    closed = hierarchy_closed_support(observed)
    missing = sorted(set(closed).difference(observed))
    if missing:
        raise AssertionError(f"Interaction hierarchy is not closed: {missing}")


def _configuration_key(value: Mapping[str, Any]) -> str:
    return canonical_json(dict(value))


def _core_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    audit_keys = {
        "inner_mean_score",
        "inner_score_se",
        "valid_inner_folds",
        "one_se_configuration_count",
        "selection_status",
        "inner_mean_log_loss",
        "inner_log_loss_se",
        "convergence_failure",
    }
    return {
        str(key): item
        for key, item in value.items()
        if str(key) not in audit_keys
    }


def select_one_standard_error(
    rows: pd.DataFrame,
    *,
    metric_column: str,
    maximize: bool,
    complexity_columns: Sequence[str],
    configuration_column: str = "configuration_json",
) -> pd.Series:
    """Apply the exact one-standard-error rule and frozen simplicity order."""

    if rows.empty:
        raise ValueError("One-standard-error selection received no candidates.")
    usable = rows.loc[np.isfinite(rows[metric_column])].copy()
    if usable.empty:
        raise ValueError("No finite candidate scores are available.")
    best_position = (
        usable[metric_column].idxmax()
        if maximize
        else usable[metric_column].idxmin()
    )
    best = usable.loc[best_position]
    best_se = float(best.get("score_se", 0.0))
    threshold = (
        float(best[metric_column]) - best_se
        if maximize
        else float(best[metric_column]) + best_se
    )
    retained = usable.loc[
        usable[metric_column].ge(threshold)
        if maximize
        else usable[metric_column].le(threshold)
    ].copy()
    sort_columns = [*complexity_columns, configuration_column]
    return retained.sort_values(sort_columns, kind="mergesort").iloc[0]


def _date_block_bootstrap_ci(
    values: pd.Series | np.ndarray,
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(replicates, len(array)))
    means = np.mean(array[draws], axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail, method="linear")),
        float(np.quantile(means, 1.0 - tail, method="linear")),
    )


def daily_spearman(
    dates: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    min_observations: int = 10,
) -> pd.DataFrame:
    """Compute equally weighted per-NY-date Spearman IC."""

    frame = pd.DataFrame(
        {
            "trade_date_ny": pd.to_datetime(dates).normalize(),
            "prediction": np.asarray(prediction, dtype=np.float64),
            "target": np.asarray(target, dtype=np.float64),
        }
    )
    records: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date_ny", sort=True):
        complete = group[["prediction", "target"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        value = np.nan
        if (
            len(complete) >= min_observations
            and complete["prediction"].nunique() > 1
            and complete["target"].nunique() > 1
        ):
            value = float(
                scipy_stats.spearmanr(
                    complete["prediction"],
                    complete["target"],
                ).statistic
            )
        records.append(
            {
                "trade_date_ny": pd.Timestamp(trade_date),
                "observations": len(complete),
                "daily_ic": value,
            }
        )
    return pd.DataFrame.from_records(records)


def _ridge_path_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_assessment: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    """Efficient centered ridge path equal to fit_intercept=True solutions."""

    target_mean = float(np.mean(y_train))
    centered_target = y_train - target_mean
    feature_mean = np.mean(x_train, axis=0)
    centered_train = x_train - feature_mean
    centered_assessment = x_assessment - feature_mean
    gram = centered_train.T @ centered_train
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    cross = eigenvectors.T @ (centered_train.T @ centered_target)
    predictions = np.empty((len(x_assessment), len(alphas)), dtype=np.float64)
    for index, alpha in enumerate(alphas):
        coefficient = eigenvectors @ (cross / (eigenvalues + float(alpha)))
        predictions[:, index] = centered_assessment @ coefficient + target_mean
    return predictions


def _mean_equal_date_log_loss(
    dates: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    frame = pd.DataFrame(
        {
            "trade_date_ny": pd.to_datetime(dates).normalize(),
            "label": labels,
            "probability": np.clip(probabilities, 1.0e-12, 1.0 - 1.0e-12),
        }
    ).dropna()
    if frame.empty:
        return np.nan
    daily = frame.groupby("trade_date_ny", sort=True).apply(
        lambda group: log_loss(
            group["label"].to_numpy(dtype=np.int64),
            group["probability"].to_numpy(dtype=np.float64),
            labels=[0, 1],
        ),
        include_groups=False,
    )
    return float(daily.mean())


def _orientation_signs(loadings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(loadings, dtype=np.float64)
    pivot = np.argmax(np.abs(matrix), axis=1)
    signs = np.sign(matrix[np.arange(len(matrix)), pivot])
    signs[signs == 0.0] = 1.0
    return signs


class Section5PCAProfileTransformer(TransformerMixin, BaseEstimator):
    """Exact P0-P3 training-standardized full-SVD PCA."""

    def __init__(self, representation: str = "P0", n_components: int = 3):
        self.representation = representation
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: object = None):
        del y
        transformed, _ = transform_profile_representation(X, self.representation)
        self.mean_ = transformed.mean(axis=0)
        self.scale_ = transformed.std(axis=0, ddof=0)
        self.constant_flags_ = self.scale_ == 0.0
        self.scale_[self.constant_flags_] = 1.0
        standardized = (transformed - self.mean_) / self.scale_
        standardized[:, self.constant_flags_] = 0.0
        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=False,
            svd_solver="full",
        ).fit(standardized)
        self.orientation_signs_ = _orientation_signs(self.pca_.components_)
        self.oriented_components_ = (
            self.pca_.components_ * self.orientation_signs_[:, None]
        )
        self.input_columns_ = tuple(P0_COLUMNS)
        self.representation_columns_ = tuple(
            REPRESENTATION_COLUMNS[str(self.representation).upper()]
        )
        self.fit_row_count_ = len(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        transformed, _ = transform_profile_representation(X, self.representation)
        standardized = (transformed - self.mean_) / self.scale_
        standardized[:, self.constant_flags_] = 0.0
        return self.pca_.transform(standardized) * self.orientation_signs_


class Section5PLSProfileTransformer(TransformerMixin, BaseEstimator):
    """Exact P2 training-standardized PLS with consistent orientation."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if y is None:
            raise ValueError("PLS fitting requires a continuous training target.")
        transformed, _ = transform_profile_representation(X, "P2")
        self.mean_ = transformed.mean(axis=0)
        self.scale_ = transformed.std(axis=0, ddof=0)
        self.constant_flags_ = self.scale_ == 0.0
        self.scale_[self.constant_flags_] = 1.0
        standardized = (transformed - self.mean_) / self.scale_
        standardized[:, self.constant_flags_] = 0.0
        self.pls_ = PLSRegression(
            n_components=self.n_components,
            scale=False,
            max_iter=500,
            tol=1.0e-6,
        ).fit(standardized, np.asarray(y, dtype=np.float64))
        weights = np.asarray(self.pls_.x_weights_, dtype=np.float64).T
        self.orientation_signs_ = _orientation_signs(weights)
        self.oriented_x_weights_ = (
            np.asarray(self.pls_.x_weights_, dtype=np.float64)
            * self.orientation_signs_[None, :]
        )
        self.oriented_x_loadings_ = (
            np.asarray(self.pls_.x_loadings_, dtype=np.float64)
            * self.orientation_signs_[None, :]
        )
        self.oriented_y_loadings_ = (
            np.asarray(self.pls_.y_loadings_, dtype=np.float64)
            * self.orientation_signs_[None, :]
        )
        self.input_columns_ = tuple(P0_COLUMNS)
        self.representation_columns_ = tuple(REPRESENTATION_COLUMNS["P2"])
        self.fit_row_count_ = len(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        transformed, _ = transform_profile_representation(X, "P2")
        standardized = (transformed - self.mean_) / self.scale_
        standardized[:, self.constant_flags_] = 0.0
        return self.pls_.transform(standardized) * self.orientation_signs_


def _assert_development_only(frame: pd.DataFrame, name: str) -> None:
    if "research_partition" not in frame.columns:
        raise ValueError(f"{name} must retain research_partition.")
    observed = set(frame["research_partition"].astype(str).unique())
    if observed != {DEVELOPMENT_PARTITION}:
        raise PermissionError(
            f"{name} contains locked outcome partitions: {sorted(observed)}"
        )


def _read_filtered_development(
    path: Path,
    columns: Sequence[str],
) -> pd.DataFrame:
    table = pq.read_table(
        path,
        columns=list(columns),
        filters=[("research_partition", "=", DEVELOPMENT_PARTITION)],
    )
    frame = table.to_pandas(ignore_metadata=True)
    _assert_development_only(frame, path.name)
    return frame


def _degenerate_indicator_frame(
    scalar_missing_reasons: pd.DataFrame,
) -> pd.DataFrame:
    indicators = scalar_missing_reasons.loc[:, ["observation_id"]].copy()
    for feature in DEGENERATE_IMPUTABLE:
        reason_column = f"{feature}__missing_reason"
        if reason_column not in scalar_missing_reasons.columns:
            raise ValueError(f"Missing frozen reason column {reason_column}.")
        indicators[f"{feature}__degenerate"] = scalar_missing_reasons[
            reason_column
        ].astype(str).eq("DEGENERATE_STATISTIC")
    return indicators


def _assert_anchor_artifact(
    project_root: Path,
    registry: pd.DataFrame,
) -> None:
    frozen = json.loads(
        (
            project_root
            / "data/processed/statistical_research/fes_project1/v1/frozen_config.json"
        ).read_text(encoding="utf-8")
    )
    if tuple(frozen["baselines"]["D1"]) != tuple(D1_FEATURES):
        raise AssertionError("D1 anchor differs from the frozen Section 1 artifact.")
    if tuple(frozen["baselines"]["O0"]) != ("atr_20",):
        raise AssertionError("O0 anchor differs from the frozen Section 1 artifact.")
    if tuple(frozen["baselines"]["O1"]) != tuple(O1_FEATURES):
        raise AssertionError("O1 anchor differs from the frozen Section 1 artifact.")
    available = set(registry["feature_name"].astype(str))
    missing = sorted(set([*D1_FEATURES, *O1_FEATURES]).difference(available))
    if missing:
        raise AssertionError(f"Frozen anchor registry is missing columns: {missing}")


def load_development_model_inputs(project_root: Path) -> DevelopmentModelInputs:
    """Load projected Section 5 inputs while physically excluding locked outcomes."""

    root = Path(project_root)
    data_dir = root / "data/processed/statistical_research/fes_project1/v1"
    scalar_path = data_dir / "scalar_features_development_gc.parquet"
    scalar_reason_path = data_dir / "scalar_missing_reasons_development_gc.parquet"
    profile_index_path = data_dir / "profile_index_development_gc.parquet"
    raw_profile_path = data_dir / "raw_profile_p0_development_gc.parquet"
    feature_path = (
        root / "data/processed/statistical_research/feature_matrix_gc.parquet"
    )
    registry_path = (
        root / "data/processed/statistical_research/feature_registry_gc.parquet"
    )
    label_path = (
        root / "data/processed/statistical_research/forward_labels_gc.parquet"
    )

    scalar = pd.read_parquet(scalar_path)
    reasons = pd.read_parquet(scalar_reason_path)
    profile_index = pd.read_parquet(profile_index_path)
    for name, frame in (
        ("scalar features", scalar),
        ("profile index", profile_index),
    ):
        _assert_development_only(frame, name)
    if not scalar["observation_id"].is_unique:
        raise ValueError("Scalar Development ids are not unique.")
    if not profile_index["observation_id"].is_unique:
        raise ValueError("Profile index Development ids are not unique.")
    registry = pd.read_parquet(registry_path)
    _assert_anchor_artifact(root, registry)

    existing_columns = list(
        dict.fromkeys(
            [
                "observation_id",
                "research_partition",
                *D1_FEATURES,
                *O1_FEATURES,
                "efficiency_ratio_30",
            ]
        )
    )
    existing = _read_filtered_development(feature_path, existing_columns)
    label_columns = [
        "observation_id",
        "decision_timestamp_utc",
        "entry_timestamp_utc",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "atr_normalization_available",
        "decision_atr_20m",
        "label_available_30",
        "label_available_60",
        "exit_timestamp_utc_30",
        "exit_timestamp_utc_60",
        *CONTINUOUS_TARGETS,
        *TARGET_TICK_COLUMN.values(),
    ]
    labels = _read_filtered_development(label_path, label_columns)
    indicators = _degenerate_indicator_frame(reasons)

    base_ids = set(scalar["observation_id"])
    for name, frame in (
        ("existing features", existing),
        ("forward labels", labels),
        ("profile index", profile_index),
        ("degenerate indicators", indicators),
    ):
        if not frame["observation_id"].is_unique:
            raise ValueError(f"{name} ids are not unique.")
        if set(frame["observation_id"]) != base_ids:
            raise ValueError(f"{name} does not cover the exact Development ids.")

    joined = (
        labels.merge(
            scalar.drop(
                columns=[
                    "decision_timestamp_utc",
                    "trade_date_ny",
                    "entry_session",
                    "research_partition",
                ]
            ),
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            existing.drop(columns=["research_partition"]),
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
        .merge(indicators, on="observation_id", how="left", validate="one_to_one")
        .merge(
            profile_index.loc[
                :,
                ["observation_id", "profile_complete_30", "profile_missing_reason"],
            ],
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
    )
    _assert_development_only(joined, "joined Section 5 frame")
    raw_profiles = pd.read_parquet(raw_profile_path)
    if not raw_profiles["observation_id"].is_unique:
        raise ValueError("Raw P0 profile ids are not unique.")
    if set(raw_profiles["observation_id"]) != set(
        joined.loc[joined["profile_complete_30"], "observation_id"]
    ):
        raise ValueError("Raw P0 ids differ from complete profile ids.")

    coverage_records: list[dict[str, Any]] = []
    dates = pd.to_datetime(joined["trade_date_ny"]).dt.normalize()
    for session in SESSIONS:
        session_mask = joined["entry_session"].astype(str).eq(session)
        for target in CONTINUOUS_TARGETS:
            availability = TARGET_AVAILABILITY[target]
            eligible = (
                session_mask
                & joined[availability].astype(bool)
                & joined["atr_normalization_available"].astype(bool)
                & np.isfinite(joined[target].to_numpy(dtype=np.float64))
            )
            scalar_complete = eligible.copy()
            for column in ALL_SCALAR_INPUTS:
                values = joined[column].to_numpy(dtype=np.float64)
                if column in DEGENERATE_IMPUTABLE:
                    indicator = joined[f"{column}__degenerate"].astype(bool).to_numpy()
                    scalar_complete &= np.isfinite(values) | indicator
                else:
                    scalar_complete &= np.isfinite(values)
            profile_complete = eligible & joined["profile_complete_30"].astype(bool)
            head_to_head = scalar_complete & profile_complete
            for support_name, mask in (
                ("eligible_target", eligible),
                ("scalar_ladder", scalar_complete),
                ("profile_ladder", profile_complete),
                ("FINAL_HEAD_TO_HEAD_SUPPORT", head_to_head),
            ):
                coverage_records.append(
                    {
                        "session": session,
                        "target": target,
                        "support_name": support_name,
                        "rows": int(mask.sum()),
                        "dates": int(dates.loc[mask].nunique()),
                        "coverage_of_eligible": (
                            float(mask.sum() / eligible.sum()) if eligible.sum() else np.nan
                        ),
                        "coverage_loss_rows": int(eligible.sum() - mask.sum()),
                    }
                )

    access_audit = pd.DataFrame.from_records(
        [
            {
                "source": scalar_path.relative_to(root).as_posix(),
                "read_scope": "presealed Development scalar features",
                "rows_loaded": len(scalar),
                "partitions_loaded": DEVELOPMENT_PARTITION,
                "outcomes_loaded": False,
            },
            {
                "source": raw_profile_path.relative_to(root).as_posix(),
                "read_scope": "presealed Development raw P0 profiles",
                "rows_loaded": len(raw_profiles),
                "partitions_loaded": DEVELOPMENT_PARTITION,
                "outcomes_loaded": False,
            },
            {
                "source": feature_path.relative_to(root).as_posix(),
                "read_scope": "physical parquet filter Development; projected anchors",
                "rows_loaded": len(existing),
                "partitions_loaded": DEVELOPMENT_PARTITION,
                "outcomes_loaded": False,
            },
            {
                "source": label_path.relative_to(root).as_posix(),
                "read_scope": "physical parquet filter Development; projected 30/60 labels",
                "rows_loaded": len(labels),
                "partitions_loaded": DEVELOPMENT_PARTITION,
                "outcomes_loaded": True,
            },
            {
                "source": "retrospective Validation outcomes",
                "read_scope": "not opened; locked until Section 6 batch",
                "rows_loaded": 0,
                "partitions_loaded": "",
                "outcomes_loaded": False,
            },
            {
                "source": "Historical Final outcomes",
                "read_scope": "not opened; locked",
                "rows_loaded": 0,
                "partitions_loaded": "",
                "outcomes_loaded": False,
            },
            {
                "source": "GC/MGC economic outcomes",
                "read_scope": "not opened; prohibited in Section 5",
                "rows_loaded": 0,
                "partitions_loaded": "",
                "outcomes_loaded": False,
            },
        ]
    )
    return DevelopmentModelInputs(
        frame=joined.sort_values(
            ["trade_date_ny", "decision_timestamp_utc", "observation_id"],
            kind="mergesort",
        ).reset_index(drop=True),
        raw_profiles=raw_profiles,
        support_coverage=pd.DataFrame.from_records(coverage_records),
        access_audit=access_audit,
    )


def _input_schema_hash(
    scalar_columns: Sequence[str],
    profile_representation: str | None,
    component_count: int | None,
) -> str:
    payload = {
        "scalar_columns": list(scalar_columns),
        "raw_profile_columns": list(P0_COLUMNS) if profile_representation else [],
        "profile_representation": profile_representation,
        "component_count": component_count,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def environment_versions() -> dict[str, str]:
    """Return the exact Section 5 runtime versions."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _columns_with_indicators(columns: Sequence[str]) -> tuple[str, ...]:
    ordered = list(dict.fromkeys(str(name) for name in columns))
    for feature in DEGENERATE_IMPUTABLE:
        if feature in ordered:
            indicator = f"{feature}__degenerate"
            if indicator not in ordered:
                ordered.append(indicator)
    return tuple(ordered)


def _model_scalar_columns(
    target: str,
    family: str,
) -> tuple[str, ...]:
    opportunity = target == "future_range_60_atr"
    if family == "M1_O0":
        if not opportunity:
            raise ValueError("M1_O0 is opportunity-only.")
        return ("atr_20",)
    if family == "M1":
        return tuple(O1_FEATURES if opportunity else D1_FEATURES)
    if family == "M2":
        return _columns_with_indicators(M2_FEATURES)
    if family == "M3":
        return _columns_with_indicators(
            M3_OPPORTUNITY_FEATURES if opportunity else M3_DIRECTION_FEATURES
        )
    if family == "M5":
        base = M3_OPPORTUNITY_FEATURES if opportunity else M3_DIRECTION_FEATURES
        return _columns_with_indicators(tuple(dict.fromkeys([*base, *SPLINE_FEATURES])))
    raise ValueError(f"No fixed scalar matrix for family {family}.")


def _head_to_head_mask(
    frame: pd.DataFrame,
    *,
    session: str,
    horizon: int,
) -> np.ndarray:
    availability = f"label_available_{horizon}"
    targets = (
        ("forward_return_60_atr", "future_range_60_atr")
        if horizon == 60
        else ("forward_return_30_atr",)
    )
    mask = (
        frame["entry_session"].astype(str).eq(session).to_numpy()
        & frame[availability].astype(bool).to_numpy()
        & frame["atr_normalization_available"].astype(bool).to_numpy()
        & frame["profile_complete_30"].astype(bool).to_numpy()
    )
    for target in targets:
        mask &= np.isfinite(frame[target].to_numpy(dtype=np.float64))
    for column in ALL_SCALAR_INPUTS:
        values = frame[column].to_numpy(dtype=np.float64)
        if column in DEGENERATE_IMPUTABLE:
            indicator = frame[f"{column}__degenerate"].astype(bool).to_numpy()
            mask &= np.isfinite(values) | indicator
        else:
            mask &= np.isfinite(values)
    return mask


def _prepare_horizon_session(
    inputs: DevelopmentModelInputs,
    *,
    session: str,
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = _head_to_head_mask(inputs.frame, session=session, horizon=horizon)
    frame = inputs.frame.loc[mask].copy().reset_index(drop=True)
    raw_indexed = inputs.raw_profiles.set_index("observation_id")
    raw = raw_indexed.loc[
        frame["observation_id"].to_numpy(dtype=np.int64),
        list(P0_COLUMNS),
    ].to_numpy(dtype=np.float64)
    if raw.shape != (len(frame), len(P0_COLUMNS)):
        raise AssertionError("Head-to-head raw-profile alignment failed.")
    if not np.isfinite(raw).all():
        raise AssertionError("Head-to-head P0 matrix is not finite.")
    return frame, raw


def _subset_scalar_transform(
    transformed: np.ndarray,
    transformer: ScalarFoldTransformer,
    columns: Sequence[str],
    *,
    include_spline_bases: bool,
) -> np.ndarray:
    requested = set(columns)
    positions = [
        index
        for index, name in enumerate(transformer.output_columns_)
        if name in requested
        or (
            include_spline_bases
            and any(name.startswith(f"{feature}__spline_") for feature in SPLINE_FEATURES)
        )
    ]
    if not positions:
        return np.empty((len(transformed), 0), dtype=np.float64)
    return transformed[:, positions]


def _score_ridge_path(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_assessment: np.ndarray,
    y_assessment: np.ndarray,
    assessment_dates: np.ndarray,
    alphas: Sequence[float],
    *,
    min_observations: int,
) -> list[float]:
    if x_train.shape[1] == 0:
        predictions = np.full(
            (len(x_assessment), len(alphas)),
            float(np.mean(y_train)),
            dtype=np.float64,
        )
    else:
        predictions = _ridge_path_predictions(
            x_train,
            y_train,
            x_assessment,
            alphas,
        )
    return _mean_daily_spearman_matrix(
        assessment_dates,
        predictions,
        y_assessment,
        min_observations=min_observations,
    ).tolist()


def _mean_daily_spearman_matrix(
    dates: np.ndarray,
    predictions: np.ndarray,
    target: np.ndarray,
    *,
    min_observations: int,
) -> np.ndarray:
    """Fast exact average-rank daily IC means for many prediction columns."""

    matrix = np.asarray(predictions, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    observed = np.asarray(target, dtype=np.float64)
    date_values = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().asi8
    scores = np.zeros(matrix.shape[1], dtype=np.float64)
    counts = np.zeros(matrix.shape[1], dtype=np.int64)
    for date_code in np.unique(date_values):
        rows = date_values == date_code
        complete = rows & np.isfinite(observed) & np.isfinite(matrix).all(axis=1)
        if int(complete.sum()) < min_observations:
            continue
        y = observed[complete]
        if np.ptp(y) == 0.0:
            continue
        y_rank = scipy_stats.rankdata(y, method="average")
        y_centered = y_rank - y_rank.mean()
        y_scale = float(np.sqrt(y_centered @ y_centered))
        prediction_rank = scipy_stats.rankdata(
            matrix[complete],
            method="average",
            axis=0,
        )
        prediction_centered = prediction_rank - prediction_rank.mean(axis=0)
        prediction_scale = np.sqrt(
            np.sum(prediction_centered * prediction_centered, axis=0)
        )
        valid = prediction_scale > 0.0
        correlations = np.full(matrix.shape[1], np.nan, dtype=np.float64)
        correlations[valid] = (
            y_centered @ prediction_centered[:, valid]
        ) / (y_scale * prediction_scale[valid])
        finite = np.isfinite(correlations)
        scores[finite] += correlations[finite]
        counts[finite] += 1
    return np.divide(
        scores,
        counts,
        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _append_ridge_tuning_records(
    records: list[dict[str, Any]],
    *,
    family: str,
    target: str,
    inner_fold_id: int,
    base_configuration: Mapping[str, Any],
    raw_predictor_count: int,
    component_count: int,
    nonlinear_rank: int,
    profile_family_rank: int,
    combined_rank: int,
    scores: Sequence[float],
    alphas: Sequence[float],
) -> None:
    for alpha, score in zip(alphas, scores, strict=True):
        configuration = {**dict(base_configuration), "ridge_alpha": float(alpha)}
        records.append(
            {
                "family": family,
                "target": target,
                "inner_fold_id": inner_fold_id,
                "configuration_json": _configuration_key(configuration),
                "configuration": configuration,
                "fold_score": score,
                "raw_predictor_count": int(raw_predictor_count),
                "component_count": int(component_count),
                "nonlinear_rank": int(nonlinear_rank),
                "profile_family_rank": int(profile_family_rank),
                "combined_rank": int(combined_rank),
            }
        )


def _summarize_tuning_records(
    records: Sequence[Mapping[str, Any]],
    *,
    family: str,
    target: str,
    min_valid_folds: int,
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame = frame.loc[
        frame["family"].eq(family) & frame["target"].eq(target)
    ].copy()
    summaries: list[dict[str, Any]] = []
    for configuration_json, group in frame.groupby(
        "configuration_json", sort=True
    ):
        scores = group["fold_score"].to_numpy(dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if len(scores) < min_valid_folds:
            continue
        first = group.iloc[0]
        summaries.append(
            {
                "family": family,
                "target": target,
                "configuration_json": configuration_json,
                "configuration": first["configuration"],
                "mean_score": float(np.mean(scores)),
                "score_se": (
                    float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
                    if len(scores) > 1
                    else 0.0
                ),
                "valid_inner_folds": len(scores),
                "raw_predictor_count": float(
                    group["raw_predictor_count"].mean()
                ),
                "component_count": int(first["component_count"]),
                "nonlinear_rank": int(first["nonlinear_rank"]),
                "profile_family_rank": int(first["profile_family_rank"]),
                "combined_rank": int(first["combined_rank"]),
            }
        )
    return pd.DataFrame.from_records(summaries)


def _select_tuned_configuration(
    records: Sequence[Mapping[str, Any]],
    *,
    family: str,
    target: str,
    min_valid_folds: int,
) -> dict[str, Any]:
    summary = _summarize_tuning_records(
        records,
        family=family,
        target=target,
        min_valid_folds=min_valid_folds,
    )
    if summary.empty:
        raw = pd.DataFrame.from_records(records)
        raw = raw.loc[
            raw["family"].eq(family) & raw["target"].eq(target)
        ].sort_values("configuration_json", kind="mergesort")
        if raw.empty:
            raise ValueError(f"{family}/{target} produced no inner trials.")
        first = raw.iloc[0]
        return {
            **dict(first["configuration"]),
            "inner_mean_score": np.nan,
            "inner_score_se": np.nan,
            "valid_inner_folds": int(raw["inner_fold_id"].nunique()),
            "one_se_configuration_count": 0,
            "selection_status": "NO_FINITE_INNER_IC_CONSTANT_MODEL",
        }
    selected = select_one_standard_error(
        summary,
        metric_column="mean_score",
        maximize=True,
        complexity_columns=(
            "raw_predictor_count",
            "component_count",
            "nonlinear_rank",
            "profile_family_rank",
            "combined_rank",
        ),
    )
    return {
        **dict(selected["configuration"]),
        "inner_mean_score": float(selected["mean_score"]),
        "inner_score_se": float(selected["score_se"]),
        "valid_inner_folds": int(selected["valid_inner_folds"]),
        "one_se_configuration_count": int(len(summary)),
    }


def _fit_elastic_support_path(
    x_train: np.ndarray,
    y_train: np.ndarray,
    columns: Sequence[str],
    *,
    alphas: Sequence[float],
    l1_ratio: float,
) -> dict[float, tuple[tuple[str, ...], dict[str, float], bool]]:
    centered_x = x_train - np.mean(x_train, axis=0)
    centered_y = y_train - np.mean(y_train)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned_alphas, coefficient_path, _, iterations = enet_path(
            centered_x,
            centered_y,
            alphas=np.asarray(sorted(alphas, reverse=True), dtype=np.float64),
            l1_ratio=float(l1_ratio),
            selection="cyclic",
            max_iter=5_000,
            tol=1.0e-6,
            return_n_iter=True,
        )
    warned = any(
        "converge" in str(item.message).lower() for item in caught
    )
    output: dict[float, tuple[tuple[str, ...], dict[str, float], bool]] = {}
    for path_index, alpha in enumerate(returned_alphas):
        coefficients = {
            str(name): float(value)
            for name, value in zip(
                columns,
                coefficient_path[:, path_index],
                strict=True,
            )
        }
        selected = [
            name for name, value in coefficients.items() if value != 0.0
        ]
        closed = hierarchy_closed_support(selected)
        failed = bool(warned or int(iterations[path_index]) >= 5_000)
        output[float(alpha)] = (closed, coefficients, failed)
    return output


def _stable_terms_from_fits(
    coefficient_records: pd.DataFrame,
    *,
    frequency_min: float,
    sign_min: float,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    if coefficient_records.empty:
        return (), pd.DataFrame()
    total_fits = int(coefficient_records["fit_id"].nunique())
    records: list[dict[str, Any]] = []
    stable: list[str] = []
    for feature, group in coefficient_records.groupby("feature_name", sort=True):
        values = group["coefficient"].to_numpy(dtype=np.float64)
        nonzero = values[values != 0.0]
        frequency = float(len(nonzero) / total_fits)
        if len(nonzero):
            positive = float(np.mean(nonzero > 0.0))
            negative = float(np.mean(nonzero < 0.0))
            sign_agreement = max(positive, negative)
            median = float(np.median(nonzero))
            q25, q75 = np.quantile(nonzero, [0.25, 0.75], method="linear")
        else:
            sign_agreement = np.nan
            median = 0.0
            q25 = 0.0
            q75 = 0.0
        is_stable = bool(
            frequency >= frequency_min
            and np.isfinite(sign_agreement)
            and sign_agreement >= sign_min
        )
        if is_stable:
            stable.append(str(feature))
        records.append(
            {
                "feature_name": str(feature),
                "fit_count": total_fits,
                "nonzero_count": len(nonzero),
                "selection_frequency": frequency,
                "sign_agreement": sign_agreement,
                "median_standardized_coefficient": median,
                "coefficient_q25": float(q25),
                "coefficient_q75": float(q75),
                "individually_stable": is_stable,
            }
        )
    return hierarchy_closed_support(stable), pd.DataFrame.from_records(records)


def _connected_correlation_clusters(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    threshold: float,
) -> list[tuple[str, ...]]:
    if not columns:
        return []
    correlation = frame.loc[:, list(columns)].corr(method="spearman").abs()
    unvisited = set(str(name) for name in columns)
    clusters: list[tuple[str, ...]] = []
    while unvisited:
        root = min(unvisited)
        stack = [root]
        members: set[str] = set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            neighbors = correlation.columns[
                correlation.loc[current].ge(threshold).to_numpy()
            ].astype(str)
            stack.extend(name for name in neighbors if name not in members)
        unvisited.difference_update(members)
        clusters.append(tuple(sorted(members)))
    return sorted(clusters)


def _fit_profile_transformer(
    *,
    profile_family: str,
    representation: str,
    component_count: int,
    raw_train: np.ndarray,
    y_train: np.ndarray,
) -> Section5PCAProfileTransformer | Section5PLSProfileTransformer:
    if profile_family == "PCA":
        return Section5PCAProfileTransformer(
            representation=representation,
            n_components=component_count,
        ).fit(raw_train)
    if profile_family == "PLS":
        if representation != "P2":
            raise ValueError("PLS profile representation must be P2.")
        return Section5PLSProfileTransformer(
            n_components=component_count
        ).fit(raw_train, y_train)
    raise ValueError(f"Unknown profile family {profile_family}.")


def _fit_predict_continuous_configuration(
    *,
    frame: pd.DataFrame,
    raw_profile: np.ndarray,
    target: str,
    family: str,
    configuration: Mapping[str, Any],
    train_mask: np.ndarray,
    assessment_mask: np.ndarray,
) -> tuple[np.ndarray, FittedContinuousModel, dict[str, float]]:
    y_train = frame.loc[train_mask, target].to_numpy(dtype=np.float64)
    if family == "M0":
        mean = float(np.mean(y_train))
        model = FittedContinuousModel(
            session=str(frame["entry_session"].iloc[0]),
            target=target,
            model_family=family,
            configuration=dict(configuration),
            scalar_transformer=None,
            profile_transformer=None,
            ridge_model=None,
            null_mean=mean,
            scalar_columns=(),
            profile_representation=None,
            component_count=None,
            input_schema_sha256=_input_schema_hash((), None, None),
        )
        return (
            np.full(int(assessment_mask.sum()), mean, dtype=np.float64),
            model,
            {},
        )

    scalar_columns = tuple(configuration.get("scalar_columns", ()))
    preprocessing = str(configuration.get("preprocessing", "S0"))
    scalar_transformer: ScalarFoldTransformer | None = None
    blocks_train: list[np.ndarray] = []
    blocks_assessment: list[np.ndarray] = []
    coefficient_names: list[str] = []
    if scalar_columns:
        scalar_transformer = ScalarFoldTransformer(
            preprocessing=preprocessing,
            spline_features=SPLINE_FEATURES,
        ).fit(frame.loc[train_mask, list(scalar_columns)])
        scalar_train = scalar_transformer.transform(
            frame.loc[train_mask, list(scalar_columns)]
        )
        scalar_assessment = scalar_transformer.transform(
            frame.loc[assessment_mask, list(scalar_columns)]
        )
        blocks_train.append(scalar_train)
        blocks_assessment.append(scalar_assessment)
        coefficient_names.extend(scalar_transformer.output_columns_)

    profile_family = configuration.get("profile_family")
    profile_representation = configuration.get("profile_representation")
    component_count = configuration.get("component_count")
    profile_transformer = None
    if profile_family is not None:
        profile_transformer = _fit_profile_transformer(
            profile_family=str(profile_family),
            representation=str(profile_representation),
            component_count=int(component_count),
            raw_train=raw_profile[train_mask],
            y_train=y_train,
        )
        blocks_train.append(profile_transformer.transform(raw_profile[train_mask]))
        blocks_assessment.append(
            profile_transformer.transform(raw_profile[assessment_mask])
        )
        coefficient_names.extend(
            f"{profile_family}_{profile_representation}_C{index + 1:02d}"
            for index in range(int(component_count))
        )
    if not blocks_train:
        mean = float(np.mean(y_train))
        model = FittedContinuousModel(
            session=str(frame["entry_session"].iloc[0]),
            target=target,
            model_family=family,
            configuration=dict(configuration),
            scalar_transformer=None,
            profile_transformer=None,
            ridge_model=None,
            null_mean=mean,
            scalar_columns=(),
            profile_representation=None,
            component_count=None,
            input_schema_sha256=_input_schema_hash((), None, None),
        )
        return (
            np.full(int(assessment_mask.sum()), mean, dtype=np.float64),
            model,
            {},
        )
    x_train = np.hstack(blocks_train)
    x_assessment = np.hstack(blocks_assessment)
    ridge = Ridge(
        alpha=float(configuration["ridge_alpha"]),
        fit_intercept=True,
        solver="svd",
    ).fit(x_train, y_train)
    predictions = ridge.predict(x_assessment)
    coefficients = {
        name: float(value)
        for name, value in zip(coefficient_names, ridge.coef_, strict=True)
    }
    model = FittedContinuousModel(
        session=str(frame["entry_session"].iloc[0]),
        target=target,
        model_family=family,
        configuration=dict(configuration),
        scalar_transformer=scalar_transformer,
        profile_transformer=profile_transformer,
        ridge_model=ridge,
        null_mean=None,
        scalar_columns=scalar_columns,
        profile_representation=(
            str(profile_representation)
            if profile_representation is not None
            else None
        ),
        component_count=(
            int(component_count) if component_count is not None else None
        ),
        input_schema_sha256=_input_schema_hash(
            scalar_columns,
            (
                str(profile_representation)
                if profile_representation is not None
                else None
            ),
            int(component_count) if component_count is not None else None,
        ),
    )
    return predictions, model, coefficients


@dataclass
class _GroupResult:
    session: str
    horizon: int
    fold_records: list[dict[str, Any]]
    threshold_records: list[dict[str, Any]]
    oof_records: list[pd.DataFrame]
    outer_choice_records: list[dict[str, Any]]
    elastic_inner_records: list[dict[str, Any]]
    elastic_outer_coefficient_records: list[dict[str, Any]]
    cluster_records: list[dict[str, Any]]
    substitution_records: list[dict[str, Any]]
    profile_records: list[dict[str, Any]]
    random_subset_records: list[dict[str, Any]]


def _apply_inner_cluster_fallback(
    *,
    frame: pd.DataFrame,
    inner_masks: Sequence[tuple[np.ndarray, np.ndarray]],
    coefficient_records: pd.DataFrame,
    base_support: Sequence[str],
    m3_columns: Sequence[str],
    config: Section5Config,
    target: str,
    session: str,
    outer_fold_id: int,
) -> tuple[tuple[str, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    if coefficient_records.empty or not inner_masks:
        return tuple(base_support), [], []
    largest_train = inner_masks[-1][0]
    continuous_columns = [
        name for name in m3_columns if not name.endswith("__degenerate")
    ]
    correlation_frame = frame.loc[largest_train, continuous_columns].copy()
    for feature in DEGENERATE_IMPUTABLE:
        if feature in correlation_frame.columns:
            values = correlation_frame[feature].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                median = float(np.nanmedian(values))
                correlation_frame[feature] = np.where(
                    np.isfinite(values), values, median
                )
    clusters = _connected_correlation_clusters(
        correlation_frame,
        continuous_columns,
        threshold=config.cluster_abs_spearman,
    )
    total_fits = int(coefficient_records["fit_id"].nunique())
    support = list(base_support)
    cluster_records: list[dict[str, Any]] = []
    substitutions: list[dict[str, Any]] = []
    for cluster_id, members in enumerate(clusters):
        cluster_coefficients = coefficient_records.loc[
            coefficient_records["feature_name"].isin(members)
        ]
        selected_fit_ids = cluster_coefficients.loc[
            cluster_coefficients["coefficient"].ne(0.0), "fit_id"
        ].unique()
        cluster_frequency = float(len(selected_fit_ids) / total_fits)
        stable_members = sorted(set(members).intersection(base_support))
        representative = ""
        fallback_used = False
        if cluster_frequency >= config.selection_frequency_min and not stable_members:
            ranking: list[tuple[float, float, str]] = []
            for member in members:
                values = coefficient_records.loc[
                    coefficient_records["feature_name"].eq(member),
                    "coefficient",
                ].to_numpy(dtype=np.float64)
                frequency = float(np.mean(values != 0.0))
                median_abs = (
                    float(np.median(np.abs(values[values != 0.0])))
                    if np.any(values != 0.0)
                    else 0.0
                )
                ranking.append((-frequency, -median_abs, member))
                substitutions.append(
                    {
                        "session": session,
                        "target": target,
                        "outer_fold_id": outer_fold_id,
                        "cluster_id": cluster_id,
                        "cluster_members": "|".join(members),
                        "member": member,
                        "member_selection_frequency": frequency,
                        "member_median_abs_coefficient": median_abs,
                    }
                )
            representative = sorted(ranking)[0][2]
            support.append(representative)
            fallback_used = True
        cluster_records.append(
            {
                "session": session,
                "target": target,
                "outer_fold_id": outer_fold_id,
                "cluster_id": cluster_id,
                "cluster_members": "|".join(members),
                "cluster_size": len(members),
                "cluster_selection_frequency": cluster_frequency,
                "individually_stable_members": "|".join(stable_members),
                "fallback_representative": representative,
                "fallback_used": fallback_used,
                "derived_from": "largest valid inner-training fold only",
            }
        )
    return (
        hierarchy_closed_support(support),
        cluster_records,
        substitutions,
    )


def _first_pass_scalar_tuning(
    *,
    frame: pd.DataFrame,
    targets: Sequence[str],
    inner_folds: Sequence[DateFold],
    exit_timestamp_column: str,
    config: Section5Config,
    session: str,
    outer_fold_id: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[np.ndarray, np.ndarray]],
    list[dict[str, Any]],
]:
    tuning_records: list[dict[str, Any]] = []
    elastic_coefficients: list[dict[str, Any]] = []
    inner_masks: list[tuple[np.ndarray, np.ndarray]] = []
    fold_records: list[dict[str, Any]] = []
    all_columns = list(ALL_SCALAR_INPUTS)
    for inner in inner_folds:
        train_mask, assessment_mask, purge_count, first_assessment = (
            purged_fold_masks(
                frame,
                inner,
                exit_timestamp_column=exit_timestamp_column,
            )
        )
        if not train_mask.any() or not assessment_mask.any():
            continue
        inner_masks.append((train_mask, assessment_mask))
        fold_record = inner.to_record(
            level="inner", parent_outer_fold_id=outer_fold_id
        )
        fold_record.update(
            {
                "session": session,
                "horizon_minutes": int(
                    exit_timestamp_column.rsplit("_", 1)[-1]
                ),
                "purged_train_rows": purge_count,
                "first_assessment_decision_utc": first_assessment.isoformat(),
                "train_rows_after_purge": int(train_mask.sum()),
                "assessment_rows": int(assessment_mask.sum()),
            }
        )
        fold_records.append(fold_record)
        transformers: dict[str, ScalarFoldTransformer] = {}
        transformed_train: dict[str, np.ndarray] = {}
        transformed_assessment: dict[str, np.ndarray] = {}
        for preprocessing in ("S0", "S1", "S2"):
            transformer = ScalarFoldTransformer(preprocessing=preprocessing).fit(
                frame.loc[train_mask, all_columns]
            )
            transformers[preprocessing] = transformer
            transformed_train[preprocessing] = transformer.transform(
                frame.loc[train_mask, all_columns]
            )
            transformed_assessment[preprocessing] = transformer.transform(
                frame.loc[assessment_mask, all_columns]
            )
        assessment_dates = frame.loc[
            assessment_mask, "trade_date_ny"
        ].to_numpy()
        for target in targets:
            y_train = frame.loc[train_mask, target].to_numpy(dtype=np.float64)
            y_assessment = frame.loc[
                assessment_mask, target
            ].to_numpy(dtype=np.float64)
            fixed_families = ["M1", "M2", "M3"]
            if target == "future_range_60_atr":
                fixed_families.insert(0, "M1_O0")
            for family in fixed_families:
                columns = _model_scalar_columns(target, family)
                for preprocessing in ("S0", "S1"):
                    transformer = transformers[preprocessing]
                    x_train = _subset_scalar_transform(
                        transformed_train[preprocessing],
                        transformer,
                        columns,
                        include_spline_bases=False,
                    )
                    x_assessment = _subset_scalar_transform(
                        transformed_assessment[preprocessing],
                        transformer,
                        columns,
                        include_spline_bases=False,
                    )
                    scores = _score_ridge_path(
                        x_train,
                        y_train,
                        x_assessment,
                        y_assessment,
                        assessment_dates,
                        config.ridge_alpha_grid,
                        min_observations=config.min_daily_observations,
                    )
                    _append_ridge_tuning_records(
                        tuning_records,
                        family=family,
                        target=target,
                        inner_fold_id=inner.fold_id,
                        base_configuration={
                            "preprocessing": preprocessing,
                            "scalar_columns": list(columns),
                        },
                        raw_predictor_count=len(columns),
                        component_count=0,
                        nonlinear_rank=0,
                        profile_family_rank=0,
                        combined_rank=0,
                        scores=scores,
                        alphas=config.ridge_alpha_grid,
                    )

            m5_columns = _model_scalar_columns(target, "M5")
            x_train_m5 = _subset_scalar_transform(
                transformed_train["S2"],
                transformers["S2"],
                m5_columns,
                include_spline_bases=True,
            )
            x_assessment_m5 = _subset_scalar_transform(
                transformed_assessment["S2"],
                transformers["S2"],
                m5_columns,
                include_spline_bases=True,
            )
            scores_m5 = _score_ridge_path(
                x_train_m5,
                y_train,
                x_assessment_m5,
                y_assessment,
                assessment_dates,
                config.ridge_alpha_grid,
                min_observations=config.min_daily_observations,
            )
            _append_ridge_tuning_records(
                tuning_records,
                family="M5",
                target=target,
                inner_fold_id=inner.fold_id,
                base_configuration={
                    "preprocessing": "S2",
                    "scalar_columns": list(m5_columns),
                },
                raw_predictor_count=len(m5_columns),
                component_count=0,
                nonlinear_rank=1,
                profile_family_rank=0,
                combined_rank=0,
                scores=scores_m5,
                alphas=config.ridge_alpha_grid,
            )

            m3_columns = _model_scalar_columns(target, "M3")
            for preprocessing in ("S0", "S1"):
                transformer = transformers[preprocessing]
                x_train_full = _subset_scalar_transform(
                    transformed_train[preprocessing],
                    transformer,
                    m3_columns,
                    include_spline_bases=False,
                )
                x_assessment_full = _subset_scalar_transform(
                    transformed_assessment[preprocessing],
                    transformer,
                    m3_columns,
                    include_spline_bases=False,
                )
                for l1_ratio in config.elastic_l1_ratio_grid:
                    path_results = _fit_elastic_support_path(
                        x_train_full,
                        y_train,
                        m3_columns,
                        alphas=config.elastic_alpha_grid,
                        l1_ratio=l1_ratio,
                    )
                    for elastic_alpha in config.elastic_alpha_grid:
                        matching_alpha = min(
                            path_results,
                            key=lambda value: abs(value - elastic_alpha),
                        )
                        support, coefficients, failed = path_results[
                            matching_alpha
                        ]
                        if not np.isclose(matching_alpha, elastic_alpha):
                            raise AssertionError(
                                "Elastic-net path changed the frozen alpha grid."
                            )
                        positions = [
                            m3_columns.index(name)
                            for name in support
                            if name in m3_columns
                        ]
                        scores = _score_ridge_path(
                            x_train_full[:, positions],
                            y_train,
                            x_assessment_full[:, positions],
                            y_assessment,
                            assessment_dates,
                            config.ridge_alpha_grid,
                            min_observations=config.min_daily_observations,
                        )
                        _append_ridge_tuning_records(
                            tuning_records,
                            family="M4_PROPOSAL",
                            target=target,
                            inner_fold_id=inner.fold_id,
                            base_configuration={
                                "preprocessing": preprocessing,
                                "elastic_alpha": float(elastic_alpha),
                                "elastic_l1_ratio": float(l1_ratio),
                            },
                            raw_predictor_count=len(support),
                            component_count=0,
                            nonlinear_rank=0,
                            profile_family_rank=0,
                            combined_rank=0,
                            scores=scores,
                            alphas=config.ridge_alpha_grid,
                        )
                        fit_id = (
                            f"{session}|{target}|{outer_fold_id}|{inner.fold_id}|"
                            f"{preprocessing}|{elastic_alpha}|{l1_ratio}"
                        )
                        for feature, coefficient in coefficients.items():
                            elastic_coefficients.append(
                                {
                                    "session": session,
                                    "target": target,
                                    "outer_fold_id": outer_fold_id,
                                    "inner_fold_id": inner.fold_id,
                                    "fit_id": fit_id,
                                    "preprocessing": preprocessing,
                                    "elastic_alpha": float(elastic_alpha),
                                    "elastic_l1_ratio": float(l1_ratio),
                                    "feature_name": feature,
                                    "coefficient": coefficient,
                                    "closed_support": "|".join(support),
                                    "convergence_failure": failed,
                                }
                            )
    return tuning_records, elastic_coefficients, inner_masks, fold_records


def _m8_scalar_columns(
    target: str,
    stable_support: Sequence[str],
) -> tuple[str, ...]:
    anchor = tuple(O1_FEATURES if target == "future_range_60_atr" else D1_FEATURES)
    selected = list(anchor)
    for name in stable_support:
        if name.endswith("__degenerate"):
            continue
        if name not in selected:
            selected.append(name)
    closed = hierarchy_closed_support(selected)
    return _columns_with_indicators(closed)


def _second_pass_profile_tuning(
    *,
    frame: pd.DataFrame,
    raw_profile: np.ndarray,
    targets: Sequence[str],
    inner_folds: Sequence[DateFold],
    inner_masks: Sequence[tuple[np.ndarray, np.ndarray]],
    stable_supports: Mapping[str, tuple[str, ...]],
    selected_elastic_configs: Mapping[str, Mapping[str, Any]],
    config: Section5Config,
    session: str,
    outer_fold_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tuning_records: list[dict[str, Any]] = []
    profile_records: list[dict[str, Any]] = []
    all_columns = list(ALL_SCALAR_INPUTS)
    for inner, (train_mask, assessment_mask) in zip(
        inner_folds, inner_masks, strict=False
    ):
        assessment_dates = frame.loc[
            assessment_mask, "trade_date_ny"
        ].to_numpy()
        required_preprocessing = sorted(
            {
                str(selected_elastic_configs[target]["preprocessing"])
                for target in targets
            }
        )
        scalar_transformers: dict[str, ScalarFoldTransformer] = {}
        scalar_train: dict[str, np.ndarray] = {}
        scalar_assessment: dict[str, np.ndarray] = {}
        for preprocessing in required_preprocessing:
            transformer = ScalarFoldTransformer(preprocessing=preprocessing).fit(
                frame.loc[train_mask, all_columns]
            )
            scalar_transformers[preprocessing] = transformer
            scalar_train[preprocessing] = transformer.transform(
                frame.loc[train_mask, all_columns]
            )
            scalar_assessment[preprocessing] = transformer.transform(
                frame.loc[assessment_mask, all_columns]
            )

        pca_scores: dict[
            tuple[str, int], tuple[np.ndarray, np.ndarray]
        ] = {}
        for representation in PROFILE_REPRESENTATIONS:
            pca = Section5PCAProfileTransformer(
                representation=representation,
                n_components=max(config.pca_component_grid),
            ).fit(raw_profile[train_mask])
            train_scores = pca.transform(raw_profile[train_mask])
            assessment_scores = pca.transform(raw_profile[assessment_mask])
            for component_count in config.pca_component_grid:
                pca_scores[(representation, component_count)] = (
                    train_scores[:, :component_count],
                    assessment_scores[:, :component_count],
                )
            transformed_assessment, _ = transform_profile_representation(
                raw_profile[assessment_mask], representation
            )
            standardized_assessment = (
                transformed_assessment - pca.mean_
            ) / pca.scale_
            standardized_assessment[:, pca.constant_flags_] = 0.0
            unflipped_scores = pca.pca_.transform(standardized_assessment)
            reconstruction = pca.pca_.inverse_transform(unflipped_scores)
            profile_records.append(
                {
                    "session": session,
                    "target": "OUTCOME_FREE",
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner.fold_id,
                    "profile_family": "PCA",
                    "profile_representation": representation,
                    "component_count": max(config.pca_component_grid),
                    "fit_rows": int(train_mask.sum()),
                    "assessment_rows": int(assessment_mask.sum()),
                    "converged": True,
                    "explained_variance_ratio": "|".join(
                        f"{value:.17g}"
                        for value in pca.pca_.explained_variance_ratio_
                    ),
                    "cumulative_explained_variance": float(
                        pca.pca_.explained_variance_ratio_.sum()
                    ),
                    "reconstruction_mse": float(
                        np.mean((standardized_assessment - reconstruction) ** 2)
                    ),
                    "oriented_loadings_sha256": hashlib.sha256(
                        np.ascontiguousarray(
                            pca.oriented_components_, dtype=np.float64
                        ).tobytes()
                    ).hexdigest(),
                    "input_column_order_sha256": hashlib.sha256(
                        canonical_json(list(pca.representation_columns_)).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                }
            )

        for target in targets:
            y_train = frame.loc[train_mask, target].to_numpy(dtype=np.float64)
            y_assessment = frame.loc[
                assessment_mask, target
            ].to_numpy(dtype=np.float64)
            stable_support = stable_supports[target]
            preprocessing = str(
                selected_elastic_configs[target]["preprocessing"]
            )
            transformer = scalar_transformers[preprocessing]
            x_full_train = scalar_train[preprocessing]
            x_full_assessment = scalar_assessment[preprocessing]
            m4_positions = [
                transformer.output_columns_.index(name)
                for name in stable_support
                if name in transformer.output_columns_
            ]
            x_m4_train = x_full_train[:, m4_positions]
            x_m4_assessment = x_full_assessment[:, m4_positions]
            m4_scores = _score_ridge_path(
                x_m4_train,
                y_train,
                x_m4_assessment,
                y_assessment,
                assessment_dates,
                config.ridge_alpha_grid,
                min_observations=config.min_daily_observations,
            )
            _append_ridge_tuning_records(
                tuning_records,
                family="M4",
                target=target,
                inner_fold_id=inner.fold_id,
                base_configuration={
                    "preprocessing": preprocessing,
                    "elastic_alpha": float(
                        selected_elastic_configs[target]["elastic_alpha"]
                    ),
                    "elastic_l1_ratio": float(
                        selected_elastic_configs[target]["elastic_l1_ratio"]
                    ),
                    "scalar_columns": list(stable_support),
                },
                raw_predictor_count=len(stable_support),
                component_count=0,
                nonlinear_rank=0,
                profile_family_rank=0,
                combined_rank=0,
                scores=m4_scores,
                alphas=config.ridge_alpha_grid,
            )

            m8_columns = _m8_scalar_columns(target, stable_support)
            m8_positions = [
                transformer.output_columns_.index(name)
                for name in m8_columns
                if name in transformer.output_columns_
            ]
            x_m8_scalar_train = x_full_train[:, m8_positions]
            x_m8_scalar_assessment = x_full_assessment[:, m8_positions]
            for (representation, component_count), (
                train_scores,
                assessment_scores,
            ) in pca_scores.items():
                profile_scores = _score_ridge_path(
                    train_scores,
                    y_train,
                    assessment_scores,
                    y_assessment,
                    assessment_dates,
                    config.ridge_alpha_grid,
                    min_observations=config.min_daily_observations,
                )
                _append_ridge_tuning_records(
                    tuning_records,
                    family="M6",
                    target=target,
                    inner_fold_id=inner.fold_id,
                    base_configuration={
                        "profile_family": "PCA",
                        "profile_representation": representation,
                        "component_count": int(component_count),
                        "scalar_columns": [],
                    },
                    raw_predictor_count=len(P0_COLUMNS),
                    component_count=component_count,
                    nonlinear_rank=0,
                    profile_family_rank=0,
                    combined_rank=0,
                    scores=profile_scores,
                    alphas=config.ridge_alpha_grid,
                )
                combined_scores = _score_ridge_path(
                    np.hstack([x_m8_scalar_train, train_scores]),
                    y_train,
                    np.hstack([x_m8_scalar_assessment, assessment_scores]),
                    y_assessment,
                    assessment_dates,
                    config.ridge_alpha_grid,
                    min_observations=config.min_daily_observations,
                )
                _append_ridge_tuning_records(
                    tuning_records,
                    family="M8",
                    target=target,
                    inner_fold_id=inner.fold_id,
                    base_configuration={
                        "preprocessing": preprocessing,
                        "scalar_columns": list(m8_columns),
                        "profile_family": "PCA",
                        "profile_representation": representation,
                        "component_count": int(component_count),
                    },
                    raw_predictor_count=len(m8_columns) + len(P0_COLUMNS),
                    component_count=component_count,
                    nonlinear_rank=0,
                    profile_family_rank=0,
                    combined_rank=1,
                    scores=combined_scores,
                    alphas=config.ridge_alpha_grid,
                )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pls = Section5PLSProfileTransformer(
                    n_components=max(config.pls_component_grid)
                ).fit(raw_profile[train_mask], y_train)
            pls_train_all = pls.transform(raw_profile[train_mask])
            pls_assessment_all = pls.transform(raw_profile[assessment_mask])
            convergence_failure = any(
                "maximum number of iterations" in str(item.message).lower()
                or "converge" in str(item.message).lower()
                for item in caught
            )
            profile_records.append(
                {
                    "session": session,
                    "target": target,
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner.fold_id,
                    "profile_family": "PLS",
                    "profile_representation": "P2",
                    "component_count": max(config.pls_component_grid),
                    "fit_rows": int(train_mask.sum()),
                    "assessment_rows": int(assessment_mask.sum()),
                    "converged": not convergence_failure,
                    "explained_variance_ratio": "",
                    "cumulative_explained_variance": np.nan,
                    "reconstruction_mse": np.nan,
                    "oriented_loadings_sha256": hashlib.sha256(
                        np.ascontiguousarray(
                            pls.oriented_x_weights_, dtype=np.float64
                        ).tobytes()
                    ).hexdigest(),
                    "input_column_order_sha256": hashlib.sha256(
                        canonical_json(list(pls.representation_columns_)).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "iterations": "|".join(
                        str(int(value)) for value in np.atleast_1d(pls.pls_.n_iter_)
                    ),
                }
            )
            for component_count in config.pls_component_grid:
                train_scores = pls_train_all[:, :component_count]
                assessment_scores = pls_assessment_all[:, :component_count]
                profile_scores = _score_ridge_path(
                    train_scores,
                    y_train,
                    assessment_scores,
                    y_assessment,
                    assessment_dates,
                    config.ridge_alpha_grid,
                    min_observations=config.min_daily_observations,
                )
                _append_ridge_tuning_records(
                    tuning_records,
                    family="M7",
                    target=target,
                    inner_fold_id=inner.fold_id,
                    base_configuration={
                        "profile_family": "PLS",
                        "profile_representation": "P2",
                        "component_count": int(component_count),
                        "scalar_columns": [],
                    },
                    raw_predictor_count=len(P0_COLUMNS),
                    component_count=component_count,
                    nonlinear_rank=0,
                    profile_family_rank=1,
                    combined_rank=0,
                    scores=profile_scores,
                    alphas=config.ridge_alpha_grid,
                )
                combined_scores = _score_ridge_path(
                    np.hstack([x_m8_scalar_train, train_scores]),
                    y_train,
                    np.hstack([x_m8_scalar_assessment, assessment_scores]),
                    y_assessment,
                    assessment_dates,
                    config.ridge_alpha_grid,
                    min_observations=config.min_daily_observations,
                )
                _append_ridge_tuning_records(
                    tuning_records,
                    family="M8",
                    target=target,
                    inner_fold_id=inner.fold_id,
                    base_configuration={
                        "preprocessing": preprocessing,
                        "scalar_columns": list(m8_columns),
                        "profile_family": "PLS",
                        "profile_representation": "P2",
                        "component_count": int(component_count),
                    },
                    raw_predictor_count=len(m8_columns) + len(P0_COLUMNS),
                    component_count=component_count,
                    nonlinear_rank=0,
                    profile_family_rank=1,
                    combined_rank=1,
                    scores=combined_scores,
                    alphas=config.ridge_alpha_grid,
                )
    return tuning_records, profile_records


def _outer_profile_diagnostic(
    *,
    session: str,
    target: str,
    outer_fold_id: int,
    family: str,
    model: FittedContinuousModel,
    raw_assessment: np.ndarray,
) -> dict[str, Any] | None:
    transformer = model.profile_transformer
    if transformer is None:
        return None
    common = {
        "session": session,
        "target": target,
        "outer_fold_id": outer_fold_id,
        "inner_fold_id": np.nan,
        "model_family": family,
        "profile_representation": model.profile_representation,
        "component_count": model.component_count,
        "fit_rows": int(transformer.fit_row_count_),
        "assessment_rows": len(raw_assessment),
        "input_column_order_sha256": hashlib.sha256(
            canonical_json(list(transformer.representation_columns_)).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    if isinstance(transformer, Section5PCAProfileTransformer):
        transformed, _ = transform_profile_representation(
            raw_assessment, str(model.profile_representation)
        )
        standardized = (transformed - transformer.mean_) / transformer.scale_
        standardized[:, transformer.constant_flags_] = 0.0
        scores = transformer.pca_.transform(standardized)
        reconstruction = transformer.pca_.inverse_transform(scores)
        return {
            **common,
            "profile_family": "PCA",
            "converged": True,
            "explained_variance_ratio": "|".join(
                f"{value:.17g}"
                for value in transformer.pca_.explained_variance_ratio_
            ),
            "cumulative_explained_variance": float(
                transformer.pca_.explained_variance_ratio_.sum()
            ),
            "reconstruction_mse": float(
                np.mean((standardized - reconstruction) ** 2)
            ),
            "oriented_loadings_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    transformer.oriented_components_, dtype=np.float64
                ).tobytes()
            ).hexdigest(),
            "iterations": "",
        }
    return {
        **common,
        "profile_family": "PLS",
        "converged": True,
        "explained_variance_ratio": "",
        "cumulative_explained_variance": np.nan,
        "reconstruction_mse": np.nan,
        "oriented_loadings_sha256": hashlib.sha256(
            np.ascontiguousarray(
                transformer.oriented_x_weights_, dtype=np.float64
            ).tobytes()
        ).hexdigest(),
        "iterations": "|".join(
            str(int(value))
            for value in np.atleast_1d(transformer.pls_.n_iter_)
        ),
    }


def _run_continuous_group(
    *,
    inputs: DevelopmentModelInputs,
    session: str,
    horizon: int,
    config: Section5Config,
    progress: bool,
) -> _GroupResult:
    frame, raw_profile = _prepare_horizon_session(
        inputs, session=session, horizon=horizon
    )
    targets = (
        ("forward_return_60_atr", "future_range_60_atr")
        if horizon == 60
        else ("forward_return_30_atr",)
    )
    dates = pd.to_datetime(frame["trade_date_ny"]).dt.normalize()
    outer_folds = build_exact_date_folds(
        dates.unique(),
        initial_train_end=config.outer_initial_train_end,
        assessment_dates=config.outer_assessment_dates,
        step_dates=config.outer_step_dates,
    )
    if not outer_folds:
        raise ValueError(f"{session}/{horizon} has no complete outer folds.")
    result = _GroupResult(
        session=session,
        horizon=horizon,
        fold_records=[],
        threshold_records=[],
        oof_records=[],
        outer_choice_records=[],
        elastic_inner_records=[],
        elastic_outer_coefficient_records=[],
        cluster_records=[],
        substitution_records=[],
        profile_records=[],
        random_subset_records=[],
    )
    exit_column = f"exit_timestamp_utc_{horizon}"
    for outer in outer_folds:
        if progress:
            print(
                f"Section 5 continuous: {session} h={horizon} "
                f"outer {outer.fold_id + 1}/{len(outer_folds)}",
                flush=True,
            )
        outer_train, outer_assessment, purge_count, first_assessment = (
            purged_fold_masks(
                frame,
                outer,
                exit_timestamp_column=exit_column,
            )
        )
        outer_record = outer.to_record(level="outer", parent_outer_fold_id=None)
        outer_record.update(
            {
                "session": session,
                "horizon_minutes": horizon,
                "purged_train_rows": purge_count,
                "first_assessment_decision_utc": first_assessment.isoformat(),
                "train_rows_after_purge": int(outer_train.sum()),
                "assessment_rows": int(outer_assessment.sum()),
            }
        )
        result.fold_records.append(outer_record)
        inner_folds = build_exact_date_folds(
            outer.train_dates,
            initial_train_end=config.inner_initial_train_end,
            assessment_dates=config.inner_assessment_dates,
            step_dates=config.inner_step_dates,
        )
        if len(inner_folds) < config.min_valid_inner_folds:
            result.outer_choice_records.append(
                {
                    "session": session,
                    "target": "|".join(targets),
                    "outer_fold_id": outer.fold_id,
                    "family": "ALL",
                    "status": "EXCLUDED_INSUFFICIENT_INNER_FOLDS",
                    "valid_inner_folds": len(inner_folds),
                }
            )
            continue
        (
            first_records,
            elastic_coefficients,
            inner_masks,
            inner_fold_records,
        ) = _first_pass_scalar_tuning(
            frame=frame,
            targets=targets,
            inner_folds=inner_folds,
            exit_timestamp_column=exit_column,
            config=config,
            session=session,
            outer_fold_id=outer.fold_id,
        )
        result.fold_records.extend(inner_fold_records)
        if len(inner_masks) < config.min_valid_inner_folds:
            continue

        selected_fixed: dict[str, dict[str, Any]] = {}
        selected_elastic: dict[str, dict[str, Any]] = {}
        stable_supports: dict[str, tuple[str, ...]] = {}
        for target in targets:
            families = ["M1", "M2", "M3", "M5"]
            if target == "future_range_60_atr":
                families.insert(0, "M1_O0")
            for family in families:
                selected_fixed[f"{target}|{family}"] = (
                    _select_tuned_configuration(
                        first_records,
                        family=family,
                        target=target,
                        min_valid_folds=config.min_valid_inner_folds,
                    )
                )
            elastic_config = _select_tuned_configuration(
                first_records,
                family="M4_PROPOSAL",
                target=target,
                min_valid_folds=config.min_valid_inner_folds,
            )
            selected_elastic[target] = elastic_config
            chosen_coefficients = pd.DataFrame.from_records(
                elastic_coefficients
            )
            chosen_coefficients = chosen_coefficients.loc[
                chosen_coefficients["target"].eq(target)
                & chosen_coefficients["preprocessing"].eq(
                    elastic_config["preprocessing"]
                )
                & np.isclose(
                    chosen_coefficients["elastic_alpha"],
                    float(elastic_config["elastic_alpha"]),
                )
                & np.isclose(
                    chosen_coefficients["elastic_l1_ratio"],
                    float(elastic_config["elastic_l1_ratio"]),
                )
            ].copy()
            stable, stability = _stable_terms_from_fits(
                chosen_coefficients,
                frequency_min=config.selection_frequency_min,
                sign_min=config.sign_agreement_min,
            )
            m3_columns = _model_scalar_columns(target, "M3")
            stable, clusters, substitutions = _apply_inner_cluster_fallback(
                frame=frame,
                inner_masks=inner_masks,
                coefficient_records=chosen_coefficients,
                base_support=stable,
                m3_columns=m3_columns,
                config=config,
                target=target,
                session=session,
                outer_fold_id=outer.fold_id,
            )
            stable_supports[target] = stable
            result.cluster_records.extend(clusters)
            result.substitution_records.extend(substitutions)
            if not stability.empty:
                stability.insert(0, "outer_fold_id", outer.fold_id)
                stability.insert(0, "target", target)
                stability.insert(0, "session", session)
                stability["preprocessing"] = elastic_config["preprocessing"]
                stability["elastic_alpha"] = elastic_config["elastic_alpha"]
                stability["elastic_l1_ratio"] = elastic_config[
                    "elastic_l1_ratio"
                ]
                stability["scope"] = "INNER_FITS_FOR_OUTER_M4_M8"
                result.elastic_inner_records.extend(
                    stability.to_dict(orient="records")
                )
            for feature, group in chosen_coefficients.groupby(
                "feature_name", sort=True
            ):
                values = group["coefficient"].to_numpy(dtype=np.float64)
                result.elastic_outer_coefficient_records.append(
                    {
                        "session": session,
                        "target": target,
                        "outer_fold_id": outer.fold_id,
                        "feature_name": feature,
                        "outer_selected": feature in stable,
                        "inner_selection_frequency": float(
                            np.mean(values != 0.0)
                        ),
                        "outer_median_standardized_coefficient": float(
                            np.median(values)
                        ),
                        "preprocessing": elastic_config["preprocessing"],
                        "elastic_alpha": elastic_config["elastic_alpha"],
                        "elastic_l1_ratio": elastic_config[
                            "elastic_l1_ratio"
                        ],
                        "convergence_failure": bool(
                            chosen_coefficients["convergence_failure"].any()
                        ),
                    }
                )

        second_records, profile_records = _second_pass_profile_tuning(
            frame=frame,
            raw_profile=raw_profile,
            targets=targets,
            inner_folds=inner_folds,
            inner_masks=inner_masks,
            stable_supports=stable_supports,
            selected_elastic_configs=selected_elastic,
            config=config,
            session=session,
            outer_fold_id=outer.fold_id,
        )
        result.profile_records.extend(profile_records)

        selected_configs: dict[str, dict[str, Any]] = {}
        for target in targets:
            selected_configs[f"{target}|M0"] = {
                "model_family": "M0",
                "scalar_columns": [],
            }
            families = ["M1", "M2", "M3", "M5"]
            if target == "future_range_60_atr":
                families.insert(0, "M1_O0")
            for family in families:
                selected_configs[f"{target}|{family}"] = selected_fixed[
                    f"{target}|{family}"
                ]
            for family in ("M4", "M6", "M7", "M8"):
                selected_configs[f"{target}|{family}"] = (
                    _select_tuned_configuration(
                        second_records,
                        family=family,
                        target=target,
                        min_valid_folds=config.min_valid_inner_folds,
                    )
                )

        if horizon == 60:
            _, expansion_threshold = nested_expansion_labels(
                frame["future_range_60_atr"].to_numpy(dtype=np.float64),
                outer_train,
                outer_assessment,
                quantile=config.expansion_quantile,
            )
            result.threshold_records.append(
                {
                    "session": session,
                    "level": "outer",
                    "outer_fold_id": outer.fold_id,
                    "inner_fold_id": np.nan,
                    "quantile": config.expansion_quantile,
                    "future_range_threshold_atr": expansion_threshold,
                    "fit_rows": int(outer_train.sum()),
                    "assessment_rows": int(outer_assessment.sum()),
                    "fit_date_count": len(outer.train_dates),
                    "assessment_date_count": len(outer.assessment_dates),
                    "fold_sha256": outer.fold_sha256,
                }
            )

        for target in targets:
            families = [f"M{index}" for index in range(9)]
            if target == "future_range_60_atr":
                families.insert(2, "M1_O0")
            assessment_frame = frame.loc[
                outer_assessment,
                [
                    "observation_id",
                    "trade_date_ny",
                    "decision_timestamp_utc",
                    "entry_session",
                    "decision_atr_20m",
                    target,
                    TARGET_TICK_COLUMN[target],
                ],
            ].reset_index(drop=True)
            for family in families:
                configuration = selected_configs[f"{target}|{family}"]
                predictions, fitted, coefficients = (
                    _fit_predict_continuous_configuration(
                        frame=frame,
                        raw_profile=raw_profile,
                        target=target,
                        family=family,
                        configuration=configuration,
                        train_mask=outer_train,
                        assessment_mask=outer_assessment,
                    )
                )
                daily = daily_spearman(
                    assessment_frame["trade_date_ny"].to_numpy(),
                    predictions,
                    assessment_frame[target].to_numpy(dtype=np.float64),
                    min_observations=config.min_daily_observations,
                )
                core_configuration = _core_configuration(configuration)
                configuration_json = _configuration_key(core_configuration)
                configuration_hash = hashlib.sha256(
                    configuration_json.encode("utf-8")
                ).hexdigest()
                choice = {
                    "session": session,
                    "target": target,
                    "outer_fold_id": outer.fold_id,
                    "family": family,
                    "status": "COMPLETE",
                    "configuration_json": configuration_json,
                    "configuration_sha256": configuration_hash,
                    "preprocessing": configuration.get("preprocessing", ""),
                    "ridge_alpha": configuration.get("ridge_alpha", np.nan),
                    "elastic_alpha": configuration.get("elastic_alpha", np.nan),
                    "elastic_l1_ratio": configuration.get(
                        "elastic_l1_ratio", np.nan
                    ),
                    "scalar_columns": "|".join(
                        configuration.get("scalar_columns", ())
                    ),
                    "scalar_predictor_count": len(
                        configuration.get("scalar_columns", ())
                    ),
                    "profile_family": configuration.get("profile_family", ""),
                    "profile_representation": configuration.get(
                        "profile_representation", ""
                    ),
                    "component_count": configuration.get(
                        "component_count", 0
                    ),
                    "inner_mean_score": configuration.get(
                        "inner_mean_score", np.nan
                    ),
                    "inner_score_se": configuration.get(
                        "inner_score_se", np.nan
                    ),
                    "valid_inner_folds": configuration.get(
                        "valid_inner_folds", len(inner_masks)
                    ),
                    "outer_fold_daily_ic": float(daily["daily_ic"].mean()),
                    "outer_valid_dates": int(daily["daily_ic"].notna().sum()),
                    "outer_train_rows": int(outer_train.sum()),
                    "outer_assessment_rows": int(outer_assessment.sum()),
                    "purged_train_rows": purge_count,
                    "coefficient_sha256": hashlib.sha256(
                        canonical_json(coefficients).encode("utf-8")
                    ).hexdigest(),
                }
                result.outer_choice_records.append(choice)
                output = assessment_frame.copy()
                output["session"] = session
                output["target"] = target
                output["model_family"] = family
                output["outer_fold_id"] = outer.fold_id
                output["prediction"] = predictions
                output["observed_atr"] = output[target].to_numpy(
                    dtype=np.float64
                )
                output["observed_ticks"] = output[
                    TARGET_TICK_COLUMN[target]
                ].to_numpy(dtype=np.float64)
                output["configuration_sha256"] = configuration_hash
                result.oof_records.append(
                    output[
                        [
                            "observation_id",
                            "decision_timestamp_utc",
                            "trade_date_ny",
                            "session",
                            "target",
                            "model_family",
                            "outer_fold_id",
                            "prediction",
                            "observed_atr",
                            "observed_ticks",
                            "decision_atr_20m",
                            "configuration_sha256",
                        ]
                    ].copy()
                )
                diagnostic = _outer_profile_diagnostic(
                    session=session,
                    target=target,
                    outer_fold_id=outer.fold_id,
                    family=family,
                    model=fitted,
                    raw_assessment=raw_profile[outer_assessment],
                )
                if diagnostic is not None:
                    result.profile_records.append(diagnostic)
    return result


def _concordance_correlation(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    complete = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[complete]
    predicted = predicted[complete]
    if len(observed) < 2:
        return np.nan
    covariance = float(
        np.mean(
            (observed - observed.mean()) * (predicted - predicted.mean())
        )
    )
    denominator = (
        float(np.var(observed, ddof=0))
        + float(np.var(predicted, ddof=0))
        + float((observed.mean() - predicted.mean()) ** 2)
    )
    return float(2.0 * covariance / denominator) if denominator > 0.0 else np.nan


def _calibration_line(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    complete = np.isfinite(observed) & np.isfinite(predicted)
    if complete.sum() < 2 or np.ptp(predicted[complete]) == 0.0:
        return np.nan, np.nan
    design = np.column_stack(
        [np.ones(int(complete.sum()), dtype=np.float64), predicted[complete]]
    )
    coefficient, *_ = np.linalg.lstsq(
        design, observed[complete], rcond=None
    )
    return float(coefficient[0]), float(coefficient[1])


def _continuous_daily_tables(
    oof: pd.DataFrame,
    *,
    min_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_records: list[pd.DataFrame] = []
    for (session, target, family), group in oof.groupby(
        ["session", "target", "model_family"], sort=True
    ):
        daily = daily_spearman(
            group["trade_date_ny"].to_numpy(),
            group["prediction"].to_numpy(dtype=np.float64),
            group["observed_atr"].to_numpy(dtype=np.float64),
            min_observations=min_observations,
        )
        fold_by_date = (
            group.groupby("trade_date_ny", sort=True)["outer_fold_id"]
            .first()
            .rename("outer_fold_id")
        )
        daily = daily.join(fold_by_date, on="trade_date_ny")
        daily.insert(0, "model_family", family)
        daily.insert(0, "target", target)
        daily.insert(0, "session", session)
        daily_records.append(daily)
    daily_all = pd.concat(daily_records, ignore_index=True)
    paired_records: list[pd.DataFrame] = []
    for (session, target), group in daily_all.groupby(
        ["session", "target"], sort=True
    ):
        anchor = group.loc[
            group["model_family"].eq("M1"),
            ["trade_date_ny", "daily_ic"],
        ].rename(columns={"daily_ic": "anchor_daily_ic"})
        for family, candidate in group.groupby("model_family", sort=True):
            paired = candidate.merge(
                anchor,
                on="trade_date_ny",
                how="inner",
                validate="one_to_one",
            )
            paired["paired_daily_ic_delta"] = (
                paired["daily_ic"] - paired["anchor_daily_ic"]
            )
            paired["session"] = session
            paired["target"] = target
            paired["model_family"] = family
            paired_records.append(
                paired[
                    [
                        "session",
                        "target",
                        "model_family",
                        "outer_fold_id",
                        "trade_date_ny",
                        "daily_ic",
                        "anchor_daily_ic",
                        "paired_daily_ic_delta",
                        "observations",
                    ]
                ]
            )
    return daily_all, pd.concat(paired_records, ignore_index=True)


def _build_continuous_metrics(
    oof: pd.DataFrame,
    daily: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    config: Section5Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_records: list[dict[str, Any]] = []
    decile_records: list[dict[str, Any]] = []
    subperiod_records: list[dict[str, Any]] = []
    keys = sorted(
        oof[["session", "target", "model_family"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for key_index, (session, target, family) in enumerate(keys):
        group = oof.loc[
            oof["session"].eq(session)
            & oof["target"].eq(target)
            & oof["model_family"].eq(family)
        ].copy()
        group = group.sort_values(
            ["trade_date_ny", "decision_timestamp_utc", "observation_id"],
            kind="mergesort",
        )
        observed = group["observed_atr"].to_numpy(dtype=np.float64)
        predicted = group["prediction"].to_numpy(dtype=np.float64)
        daily_group = daily.loc[
            daily["session"].eq(session)
            & daily["target"].eq(target)
            & daily["model_family"].eq(family)
        ].copy()
        paired_group = paired.loc[
            paired["session"].eq(session)
            & paired["target"].eq(target)
            & paired["model_family"].eq(family)
        ].copy()
        ic_low, ic_high = _date_block_bootstrap_ci(
            daily_group["daily_ic"],
            seed=config.random_seed + 10_000 + key_index,
            replicates=config.bootstrap_replicates,
            confidence=config.bootstrap_confidence,
        )
        delta_low, delta_high = _date_block_bootstrap_ci(
            paired_group["paired_daily_ic_delta"],
            seed=config.random_seed + 20_000 + key_index,
            replicates=config.bootstrap_replicates,
            confidence=config.bootstrap_confidence,
        )
        intercept, slope = _calibration_line(observed, predicted)
        atr_ticks = (
            group["decision_atr_20m"].to_numpy(dtype=np.float64) / 0.10
        )
        predicted_ticks = predicted * atr_ticks
        observed_ticks = group["observed_ticks"].to_numpy(dtype=np.float64)
        record: dict[str, Any] = {
            "session": session,
            "target": target,
            "model_family": family,
            "rows": len(group),
            "valid_oof_dates": int(daily_group["daily_ic"].notna().sum()),
            "daily_ic_mean": float(daily_group["daily_ic"].mean()),
            "daily_ic_ci_low": ic_low,
            "daily_ic_ci_high": ic_high,
            "anchor_daily_ic_mean": float(
                paired_group["anchor_daily_ic"].mean()
            ),
            "paired_daily_ic_delta_mean": float(
                paired_group["paired_daily_ic_delta"].mean()
            ),
            "paired_delta_ci_low": delta_low,
            "paired_delta_ci_high": delta_high,
            "mae_atr": float(mean_absolute_error(observed, predicted)),
            "rmse_atr": float(
                np.sqrt(mean_squared_error(observed, predicted))
            ),
            "mae_ticks": float(
                mean_absolute_error(observed_ticks, predicted_ticks)
            ),
            "concordance_correlation": _concordance_correlation(
                observed, predicted
            ),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }
        if target.startswith("forward_return"):
            nonzero = observed != 0.0
            observed_direction = observed[nonzero] > 0.0
            predicted_direction = predicted[nonzero] > 0.0
            record["sign_accuracy_nonzero"] = (
                float(np.mean(observed_direction == predicted_direction))
                if nonzero.any()
                else np.nan
            )
            if (
                nonzero.any()
                and np.unique(observed_direction).size == 2
            ):
                record["balanced_accuracy_nonzero"] = float(
                    balanced_accuracy_score(
                        observed_direction, predicted_direction
                    )
                )
                tn, fp, fn, tp = confusion_matrix(
                    observed_direction,
                    predicted_direction,
                    labels=[False, True],
                ).ravel()
                record.update(
                    {
                        "short_correct_tn": int(tn),
                        "false_long_fp": int(fp),
                        "false_short_fn": int(fn),
                        "long_correct_tp": int(tp),
                    }
                )
            else:
                record["balanced_accuracy_nonzero"] = np.nan
        positive_delta = paired_group.loc[
            paired_group["paired_daily_ic_delta"].gt(0.0),
            "paired_daily_ic_delta",
        ].sort_values(ascending=False)
        record["best_10_positive_delta_share"] = (
            float(positive_delta.head(10).sum() / positive_delta.sum())
            if positive_delta.sum() > 0.0
            else np.nan
        )
        metric_records.append(record)

        ranked = group["prediction"].rank(method="first")
        group["prediction_decile"] = pd.qcut(
            ranked,
            q=10,
            labels=False,
            duplicates="drop",
        ) + 1
        for decile, decile_group in group.groupby(
            "prediction_decile", sort=True
        ):
            decile_records.append(
                {
                    "session": session,
                    "target": target,
                    "model_family": family,
                    "prediction_decile": int(decile),
                    "rows": len(decile_group),
                    "prediction_mean_atr": float(
                        decile_group["prediction"].mean()
                    ),
                    "observed_mean_atr": float(
                        decile_group["observed_atr"].mean()
                    ),
                    "observed_mean_ticks": float(
                        decile_group["observed_ticks"].mean()
                    ),
                }
            )
        daily_group["quarter"] = (
            pd.to_datetime(daily_group["trade_date_ny"])
            .dt.to_period("Q")
            .astype(str)
        )
        paired_group["quarter"] = (
            pd.to_datetime(paired_group["trade_date_ny"])
            .dt.to_period("Q")
            .astype(str)
        )
        for quarter, quarter_daily in daily_group.groupby(
            "quarter", sort=True
        ):
            quarter_pair = paired_group.loc[
                paired_group["quarter"].eq(quarter)
            ]
            subperiod_records.append(
                {
                    "session": session,
                    "target": target,
                    "model_family": family,
                    "period_type": "quarter",
                    "period": quarter,
                    "valid_dates": int(quarter_daily["daily_ic"].notna().sum()),
                    "daily_ic_mean": float(quarter_daily["daily_ic"].mean()),
                    "paired_delta_mean": float(
                        quarter_pair["paired_daily_ic_delta"].mean()
                    ),
                    "paired_delta_sum": float(
                        quarter_pair["paired_daily_ic_delta"].sum()
                    ),
                }
            )
    return (
        pd.DataFrame.from_records(metric_records),
        pd.DataFrame.from_records(decile_records),
        pd.DataFrame.from_records(subperiod_records),
    )


def _build_model_comparison_and_selection(
    choices: pd.DataFrame,
    metrics: pd.DataFrame,
    subperiods: pd.DataFrame,
    elastic_outer: pd.DataFrame,
    *,
    config: Section5Config,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    records: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], str] = {}
    for (session, target), target_choices in choices.loc[
        choices["status"].eq("COMPLETE")
    ].groupby(["session", "target"], sort=True):
        family_records: list[dict[str, Any]] = []
        for family, group in target_choices.groupby("family", sort=True):
            if family == "M1_O0":
                eligible_for_selection = False
            else:
                eligible_for_selection = True
            fold_scores = group["outer_fold_daily_ic"].to_numpy(
                dtype=np.float64
            )
            complexity_signatures = group[
                [
                    "preprocessing",
                    "profile_family",
                    "profile_representation",
                    "component_count",
                    "scalar_columns",
                ]
            ].astype(str).agg("|".join, axis=1)
            mode_count = (
                int(complexity_signatures.value_counts().iloc[0])
                if len(complexity_signatures)
                else 0
            )
            modal_fraction = (
                float(mode_count / len(complexity_signatures))
                if len(complexity_signatures)
                else np.nan
            )
            component_values = group["component_count"].fillna(0).astype(int)
            modal_component = (
                int(component_values.mode().min())
                if len(component_values)
                else 0
            )
            within_two = (
                float(np.mean(np.abs(component_values - modal_component) <= 2))
                if modal_component > 0
                else 1.0
            )
            raw_predictors = (
                group["scalar_predictor_count"].astype(float)
                + group["profile_family"].astype(str).ne("").astype(int) * len(P0_COLUMNS)
            )
            row = {
                "session": session,
                "target": target,
                "model_family": family,
                "outer_fold_count": len(group),
                "mean_outer_fold_daily_ic": float(np.mean(fold_scores)),
                "outer_fold_score_se": (
                    float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))
                    if len(fold_scores) > 1
                    else 0.0
                ),
                "raw_predictor_count": float(raw_predictors.mean()),
                "component_count": modal_component,
                "nonlinear_rank": int(family == "M5"),
                "profile_family_rank": int(
                    family == "M7"
                    or (
                        family == "M8"
                        and group["profile_family"].astype(str).eq("PLS").mean()
                        >= 0.5
                    )
                ),
                "combined_rank": int(family == "M8"),
                "configuration_json": family,
                "modal_complexity_fraction": modal_fraction,
                "component_within_two_fraction": within_two,
                "eligible_for_primary_selection": eligible_for_selection,
            }
            family_records.append(row)
        family_frame = pd.DataFrame.from_records(family_records)
        selectable = family_frame.loc[
            family_frame["eligible_for_primary_selection"]
        ].copy()
        chosen = select_one_standard_error(
            selectable,
            metric_column="mean_outer_fold_daily_ic",
            maximize=True,
            complexity_columns=(
                "raw_predictor_count",
                "component_count",
                "nonlinear_rank",
                "profile_family_rank",
                "combined_rank",
            ),
        )
        chosen_family = str(chosen["model_family"])
        selected[(session, target)] = chosen_family
        for row in family_records:
            family = str(row["model_family"])
            metric = metrics.loc[
                metrics["session"].eq(session)
                & metrics["target"].eq(target)
                & metrics["model_family"].eq(family)
            ].iloc[0]
            quarter = subperiods.loc[
                subperiods["session"].eq(session)
                & subperiods["target"].eq(target)
                & subperiods["model_family"].eq(family)
                & subperiods["valid_dates"].ge(10)
            ]
            overall_delta = float(metric["paired_daily_ic_delta_mean"])
            if quarter.empty or overall_delta == 0.0:
                quarter_sign_fraction = np.nan
                positive_quarter_share = np.nan
            else:
                sign = np.sign(overall_delta)
                quarter_sign_fraction = float(
                    np.mean(np.sign(quarter["paired_delta_mean"]) == sign)
                )
                positive = quarter.loc[
                    quarter["paired_delta_sum"].gt(0.0), "paired_delta_sum"
                ]
                positive_quarter_share = (
                    float(positive.max() / positive.sum())
                    if positive.sum() > 0.0
                    else np.nan
                )
            convergence_failure = False
            if family in {"M4", "M8"} and not elastic_outer.empty:
                convergence_failure = bool(
                    elastic_outer.loc[
                        elastic_outer["session"].eq(session)
                        & elastic_outer["target"].eq(target),
                        "convergence_failure",
                    ].any()
                )
            complexity_stable = bool(
                row["modal_complexity_fraction"]
                >= config.modal_complexity_fraction_min
                and row["component_within_two_fraction"]
                >= config.component_within_two_fraction_min
                and not convergence_failure
            )
            development_gate = bool(
                metric["paired_daily_ic_delta_mean"] >= 0.01
                and metric["paired_delta_ci_low"] > 0.0
                and metric["daily_ic_mean"] > 0.0
                and metric["valid_oof_dates"]
                >= config.development_oof_dates_min
                and np.isfinite(quarter_sign_fraction)
                and quarter_sign_fraction >= 0.60
                and np.isfinite(positive_quarter_share)
                and positive_quarter_share <= 0.50
                and np.isfinite(metric["best_10_positive_delta_share"])
                and metric["best_10_positive_delta_share"] <= 0.50
                and complexity_stable
            )
            row.update(
                {
                    "selected_primary_family": family == chosen_family,
                    "daily_ic_mean": metric["daily_ic_mean"],
                    "daily_ic_ci_low": metric["daily_ic_ci_low"],
                    "daily_ic_ci_high": metric["daily_ic_ci_high"],
                    "paired_daily_ic_delta_mean": metric[
                        "paired_daily_ic_delta_mean"
                    ],
                    "paired_delta_ci_low": metric["paired_delta_ci_low"],
                    "paired_delta_ci_high": metric["paired_delta_ci_high"],
                    "valid_oof_dates": metric["valid_oof_dates"],
                    "quarter_same_sign_fraction": quarter_sign_fraction,
                    "largest_positive_quarter_share": positive_quarter_share,
                    "best_10_positive_delta_share": metric[
                        "best_10_positive_delta_share"
                    ],
                    "convergence_failure": convergence_failure,
                    "complexity_stable": complexity_stable,
                    "development_model_gate_pass": development_gate,
                    "validation_gate_status": "PENDING_LOCKED_SECTION_6",
                }
            )
            records.append(row)
    return pd.DataFrame.from_records(records), selected


def _deterministic_mode(values: Iterable[Any]) -> Any:
    serialized = [canonical_json({"value": value}) for value in values]
    if not serialized:
        raise ValueError("Cannot take a deterministic mode of an empty sequence.")
    counts = Counter(serialized)
    chosen = min(
        counts,
        key=lambda item: (-counts[item], item),
    )
    return json.loads(chosen)["value"]


def _derive_final_continuous_configuration(
    *,
    target: str,
    family: str,
    choices: pd.DataFrame,
    config: Section5Config,
) -> dict[str, Any]:
    group = choices.loc[choices["family"].eq(family)].copy()
    if group.empty:
        raise ValueError(f"No outer choices exist for final {family}/{target}.")
    parsed = [json.loads(value) for value in group["configuration_json"]]
    if family == "M0":
        return {"model_family": "M0", "scalar_columns": []}
    preprocessing_values = [
        item.get("preprocessing") for item in parsed if item.get("preprocessing")
    ]
    final: dict[str, Any] = {
        "ridge_alpha": float(
            _deterministic_mode(
                [float(item["ridge_alpha"]) for item in parsed]
            )
        )
    }
    if preprocessing_values:
        final["preprocessing"] = str(
            _deterministic_mode(preprocessing_values)
        )
    if family in {"M1", "M2", "M3", "M5"}:
        final["scalar_columns"] = list(
            _model_scalar_columns(target, family)
        )
    elif family == "M4":
        supports = [set(item.get("scalar_columns", ())) for item in parsed]
        universe = sorted(set().union(*supports))
        retained = [
            name
            for name in universe
            if np.mean([name in support for support in supports])
            >= config.selection_frequency_min
        ]
        final["scalar_columns"] = list(hierarchy_closed_support(retained))
        final["elastic_alpha"] = float(
            _deterministic_mode(
                [float(item["elastic_alpha"]) for item in parsed]
            )
        )
        final["elastic_l1_ratio"] = float(
            _deterministic_mode(
                [float(item["elastic_l1_ratio"]) for item in parsed]
            )
        )
    elif family in {"M6", "M7", "M8"}:
        profile_signature = _deterministic_mode(
            [
                [
                    str(item["profile_family"]),
                    str(item["profile_representation"]),
                ]
                for item in parsed
            ]
        )
        final["profile_family"] = profile_signature[0]
        final["profile_representation"] = profile_signature[1]
        matching = [
            item
            for item in parsed
            if [
                str(item["profile_family"]),
                str(item["profile_representation"]),
            ]
            == profile_signature
        ]
        final["component_count"] = int(
            _deterministic_mode(
                [int(item["component_count"]) for item in matching]
            )
        )
        if family in {"M6", "M7"}:
            final["scalar_columns"] = []
        else:
            supports = [set(item.get("scalar_columns", ())) for item in parsed]
            universe = sorted(set().union(*supports))
            anchor = set(
                O1_FEATURES
                if target == "future_range_60_atr"
                else D1_FEATURES
            )
            retained = [
                name
                for name in universe
                if name in anchor
                or np.mean([name in support for support in supports])
                >= config.selection_frequency_min
            ]
            final["scalar_columns"] = list(
                _columns_with_indicators(
                    hierarchy_closed_support(retained)
                )
            )
    else:
        raise ValueError(f"Unknown final continuous family {family}.")
    return final


def _fit_final_continuous_models(
    *,
    inputs: DevelopmentModelInputs,
    choices: pd.DataFrame,
    selected_families: Mapping[tuple[str, str], str],
    config: Section5Config,
) -> tuple[
    dict[str, FittedContinuousModel],
    pd.DataFrame,
]:
    models: dict[str, FittedContinuousModel] = {}
    registry_records: list[dict[str, Any]] = []
    prepared: dict[tuple[str, int], tuple[pd.DataFrame, np.ndarray]] = {}
    for (session, target), family in sorted(selected_families.items()):
        horizon = TARGET_HORIZON[target]
        key = (session, horizon)
        if key not in prepared:
            prepared[key] = _prepare_horizon_session(
                inputs, session=session, horizon=horizon
            )
        frame, raw = prepared[key]
        target_choices = choices.loc[
            choices["session"].eq(session)
            & choices["target"].eq(target)
            & choices["status"].eq("COMPLETE")
        ]
        final_configuration = _derive_final_continuous_configuration(
            target=target,
            family=family,
            choices=target_choices,
            config=config,
        )
        all_rows = np.ones(len(frame), dtype=bool)
        _, model, coefficients = _fit_predict_continuous_configuration(
            frame=frame,
            raw_profile=raw,
            target=target,
            family=family,
            configuration=final_configuration,
            train_mask=all_rows,
            assessment_mask=all_rows,
        )
        model_key = f"{session.lower().replace(' ', '_')}|{target}"
        models[model_key] = model
        configuration_json = _configuration_key(final_configuration)
        registry_records.append(
            {
                "model_key": model_key,
                "model_role": "continuous",
                "session": session,
                "target": target,
                "model_family": family,
                "configuration_json": configuration_json,
                "configuration_sha256": hashlib.sha256(
                    configuration_json.encode("utf-8")
                ).hexdigest(),
                "input_schema_sha256": model.input_schema_sha256,
                "fit_rows": len(frame),
                "fit_dates": int(
                    pd.to_datetime(frame["trade_date_ny"]).dt.normalize().nunique()
                ),
                "coefficient_sha256": hashlib.sha256(
                    canonical_json(coefficients).encode("utf-8")
                ).hexdigest(),
                "transform_state_sha256": (
                    hashlib.sha256(
                        canonical_json(
                            asdict(model.scalar_transformer.get_state())
                        ).encode("utf-8")
                    ).hexdigest()
                    if model.scalar_transformer is not None
                    else ""
                ),
                "validation_status": "UNREAD",
            }
        )
    return models, pd.DataFrame.from_records(registry_records)


def _mean_equal_date_log_loss_matrix(
    dates: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    matrix = np.clip(matrix, 1.0e-12, 1.0 - 1.0e-12)
    y = np.asarray(labels, dtype=np.float64)
    date_values = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().asi8
    output = np.zeros(matrix.shape[1], dtype=np.float64)
    date_count = 0
    for date_code in np.unique(date_values):
        rows = (date_values == date_code) & np.isfinite(y)
        if not rows.any():
            continue
        losses = -(
            y[rows, None] * np.log(matrix[rows])
            + (1.0 - y[rows, None]) * np.log(1.0 - matrix[rows])
        )
        output += losses.mean(axis=0)
        date_count += 1
    if date_count == 0:
        return np.full(matrix.shape[1], np.nan, dtype=np.float64)
    return output / date_count


def _logistic_probability_path(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_assessment: np.ndarray,
    c_grid: Sequence[float],
) -> tuple[np.ndarray, list[bool]]:
    if x_train.shape[1] == 0:
        prevalence = float(np.mean(y_train))
        return (
            np.full(
                (len(x_assessment), len(c_grid)),
                prevalence,
                dtype=np.float64,
            ),
            [False] * len(c_grid),
        )
    probabilities = np.empty(
        (len(x_assessment), len(c_grid)), dtype=np.float64
    )
    failures: list[bool] = []
    for index, c_value in enumerate(c_grid):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LogisticRegression(
                C=float(c_value),
                l1_ratio=0.0,
                solver="lbfgs",
                class_weight=None,
                fit_intercept=True,
                max_iter=2_000,
                tol=1.0e-6,
                random_state=None,
            ).fit(x_train, y_train.astype(np.int64))
        probabilities[:, index] = model.predict_proba(x_assessment)[:, 1]
        failures.append(
            any("converge" in str(item.message).lower() for item in caught)
        )
    return probabilities, failures


def _classifier_design_from_configuration(
    *,
    frame: pd.DataFrame,
    raw_profile: np.ndarray,
    configuration: Mapping[str, Any],
    train_mask: np.ndarray,
    assessment_mask: np.ndarray,
    outcome_for_pls: np.ndarray,
    union_transformers: Mapping[str, ScalarFoldTransformer] | None = None,
    union_train: Mapping[str, np.ndarray] | None = None,
    union_assessment: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, ScalarFoldTransformer | None, Any | None]:
    scalar_columns = tuple(configuration.get("scalar_columns", ()))
    blocks_train: list[np.ndarray] = []
    blocks_assessment: list[np.ndarray] = []
    scalar_transformer = None
    if scalar_columns:
        preprocessing = str(configuration.get("preprocessing", "S0"))
        if (
            union_transformers is not None
            and preprocessing in union_transformers
            and union_train is not None
            and union_assessment is not None
        ):
            scalar_transformer = union_transformers[preprocessing]
            include_splines = preprocessing == "S2"
            blocks_train.append(
                _subset_scalar_transform(
                    union_train[preprocessing],
                    scalar_transformer,
                    scalar_columns,
                    include_spline_bases=include_splines,
                )
            )
            blocks_assessment.append(
                _subset_scalar_transform(
                    union_assessment[preprocessing],
                    scalar_transformer,
                    scalar_columns,
                    include_spline_bases=include_splines,
                )
            )
        else:
            scalar_transformer = ScalarFoldTransformer(
                preprocessing=preprocessing
            ).fit(frame.loc[train_mask, list(scalar_columns)])
            blocks_train.append(
                scalar_transformer.transform(
                    frame.loc[train_mask, list(scalar_columns)]
                )
            )
            blocks_assessment.append(
                scalar_transformer.transform(
                    frame.loc[assessment_mask, list(scalar_columns)]
                )
            )
    profile_transformer = None
    if configuration.get("profile_family"):
        profile_transformer = _fit_profile_transformer(
            profile_family=str(configuration["profile_family"]),
            representation=str(configuration["profile_representation"]),
            component_count=int(configuration["component_count"]),
            raw_train=raw_profile[train_mask],
            y_train=outcome_for_pls,
        )
        blocks_train.append(
            profile_transformer.transform(raw_profile[train_mask])
        )
        blocks_assessment.append(
            profile_transformer.transform(raw_profile[assessment_mask])
        )
    if not blocks_train:
        return (
            np.empty((int(train_mask.sum()), 0), dtype=np.float64),
            np.empty((int(assessment_mask.sum()), 0), dtype=np.float64),
            None,
            None,
        )
    return (
        np.hstack(blocks_train),
        np.hstack(blocks_assessment),
        scalar_transformer,
        profile_transformer,
    )


def _classifier_fixed_configurations() -> dict[str, list[dict[str, Any]]]:
    return {
        "M1_O0": [
            {
                "preprocessing": preprocessing,
                "scalar_columns": ["atr_20"],
            }
            for preprocessing in ("S0", "S1")
        ],
        "M1_O1": [
            {
                "preprocessing": preprocessing,
                "scalar_columns": list(O1_FEATURES),
            }
            for preprocessing in ("S0", "S1")
        ],
        "M2": [
            {
                "preprocessing": preprocessing,
                "scalar_columns": list(
                    _columns_with_indicators(M2_FEATURES)
                ),
            }
            for preprocessing in ("S0", "S1")
        ],
        "M3": [
            {
                "preprocessing": preprocessing,
                "scalar_columns": list(
                    _columns_with_indicators(M3_OPPORTUNITY_FEATURES)
                ),
            }
            for preprocessing in ("S0", "S1")
        ],
    }


def _select_classifier_configuration(
    tuning_records: pd.DataFrame,
    *,
    family: str,
    min_valid_folds: int,
) -> dict[str, Any]:
    subset = tuning_records.loc[tuning_records["family"].eq(family)].copy()
    summaries: list[dict[str, Any]] = []
    for configuration_json, group in subset.groupby(
        "configuration_json", sort=True
    ):
        losses = group["fold_loss"].to_numpy(dtype=np.float64)
        losses = losses[np.isfinite(losses)]
        if len(losses) < min_valid_folds:
            continue
        first = group.iloc[0]
        summaries.append(
            {
                "configuration_json": configuration_json,
                "configuration": first["configuration"],
                "mean_loss": float(np.mean(losses)),
                "score_se": (
                    float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
                    if len(losses) > 1
                    else 0.0
                ),
                "anchor_rank": int(first["anchor_rank"]),
                "raw_predictor_count": int(first["raw_predictor_count"]),
                "component_count": int(first["component_count"]),
                "c_value": float(first["c_value"]),
                "valid_inner_folds": len(losses),
                "convergence_failure": bool(
                    group["convergence_failure"].any()
                ),
            }
        )
    summary = pd.DataFrame.from_records(summaries)
    if summary.empty:
        raise ValueError(f"Classifier {family} has insufficient inner folds.")
    selected = select_one_standard_error(
        summary,
        metric_column="mean_loss",
        maximize=False,
        complexity_columns=(
            "anchor_rank",
            "raw_predictor_count",
            "component_count",
            "c_value",
        ),
    )
    return {
        **dict(selected["configuration"]),
        "inner_mean_log_loss": float(selected["mean_loss"]),
        "inner_log_loss_se": float(selected["score_se"]),
        "valid_inner_folds": int(selected["valid_inner_folds"]),
        "convergence_failure": bool(selected["convergence_failure"]),
    }


def _run_expansion_classifiers(
    *,
    inputs: DevelopmentModelInputs,
    continuous_choices: pd.DataFrame,
    selected_continuous_families: Mapping[tuple[str, str], str],
    config: Section5Config,
    progress: bool,
    sessions: Sequence[str] = SESSIONS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    oof_frames: list[pd.DataFrame] = []
    choice_records: list[dict[str, Any]] = []
    threshold_records: list[dict[str, Any]] = []
    fixed_candidates = _classifier_fixed_configurations()
    family_anchor_rank = {
        "M1_O1": 0,
        "M1_O0": 1,
        "M2": 1,
        "M3": 1,
        "M_SELECTED_CONTINUOUS": 1,
    }
    for session in sessions:
        frame, raw = _prepare_horizon_session(
            inputs, session=session, horizon=60
        )
        dates = pd.to_datetime(frame["trade_date_ny"]).dt.normalize()
        outer_folds = build_exact_date_folds(
            dates.unique(),
            initial_train_end=config.outer_initial_train_end,
            assessment_dates=config.outer_assessment_dates,
            step_dates=config.outer_step_dates,
        )
        continuous_family = selected_continuous_families[
            (session, "future_range_60_atr")
        ]
        for outer in outer_folds:
            if progress:
                print(
                    f"Section 5 expansion: {session} "
                    f"outer {outer.fold_id + 1}/{len(outer_folds)}",
                    flush=True,
                )
            outer_train, outer_assessment, _, _ = purged_fold_masks(
                frame,
                outer,
                exit_timestamp_column="exit_timestamp_utc_60",
            )
            future_range = frame["future_range_60_atr"].to_numpy(
                dtype=np.float64
            )
            outer_labels, outer_threshold = nested_expansion_labels(
                future_range,
                outer_train,
                outer_assessment,
                quantile=config.expansion_quantile,
            )
            y_outer_train = (
                future_range[outer_train] >= outer_threshold
            ).astype(np.int64)
            y_outer_assessment = outer_labels[outer_assessment].astype(
                np.int64
            )
            threshold_records.append(
                {
                    "session": session,
                    "level": "outer_classifier",
                    "outer_fold_id": outer.fold_id,
                    "inner_fold_id": np.nan,
                    "quantile": config.expansion_quantile,
                    "future_range_threshold_atr": outer_threshold,
                    "fit_rows": int(outer_train.sum()),
                    "assessment_rows": int(outer_assessment.sum()),
                    "fit_date_count": len(outer.train_dates),
                    "assessment_date_count": len(outer.assessment_dates),
                    "fold_sha256": outer.fold_sha256,
                }
            )
            continuous_row = continuous_choices.loc[
                continuous_choices["session"].eq(session)
                & continuous_choices["target"].eq("future_range_60_atr")
                & continuous_choices["outer_fold_id"].eq(outer.fold_id)
                & continuous_choices["family"].eq(continuous_family)
                & continuous_choices["status"].eq("COMPLETE")
            ].iloc[0]
            selected_continuous_config = json.loads(
                continuous_row["configuration_json"]
            )
            selected_continuous_config.pop("ridge_alpha", None)
            selected_continuous_config["source_continuous_family"] = (
                continuous_family
            )
            base_candidates = {
                **fixed_candidates,
                "M_SELECTED_CONTINUOUS": [selected_continuous_config],
            }
            inner_folds = build_exact_date_folds(
                outer.train_dates,
                initial_train_end=config.inner_initial_train_end,
                assessment_dates=config.inner_assessment_dates,
                step_dates=config.inner_step_dates,
            )
            tuning_records: list[dict[str, Any]] = []
            for inner in inner_folds:
                train_mask, assessment_mask, _, _ = purged_fold_masks(
                    frame,
                    inner,
                    exit_timestamp_column="exit_timestamp_utc_60",
                )
                if not train_mask.any() or not assessment_mask.any():
                    continue
                inner_labels, inner_threshold = nested_expansion_labels(
                    future_range,
                    train_mask,
                    assessment_mask,
                    quantile=config.expansion_quantile,
                )
                y_train = (
                    future_range[train_mask] >= inner_threshold
                ).astype(np.int64)
                y_assessment = inner_labels[assessment_mask].astype(np.int64)
                threshold_records.append(
                    {
                        "session": session,
                        "level": "inner_classifier",
                        "outer_fold_id": outer.fold_id,
                        "inner_fold_id": inner.fold_id,
                        "quantile": config.expansion_quantile,
                        "future_range_threshold_atr": inner_threshold,
                        "fit_rows": int(train_mask.sum()),
                        "assessment_rows": int(assessment_mask.sum()),
                        "fit_date_count": len(inner.train_dates),
                        "assessment_date_count": len(inner.assessment_dates),
                        "fold_sha256": inner.fold_sha256,
                    }
                )
                required_preprocessing = {"S0", "S1"}
                selected_preprocessing = selected_continuous_config.get(
                    "preprocessing"
                )
                if selected_preprocessing:
                    required_preprocessing.add(
                        str(selected_preprocessing)
                    )
                union_transformers: dict[str, ScalarFoldTransformer] = {}
                union_train: dict[str, np.ndarray] = {}
                union_assessment: dict[str, np.ndarray] = {}
                for preprocessing in sorted(required_preprocessing):
                    transformer = ScalarFoldTransformer(
                        preprocessing=preprocessing
                    ).fit(frame.loc[train_mask, list(ALL_SCALAR_INPUTS)])
                    union_transformers[preprocessing] = transformer
                    union_train[preprocessing] = transformer.transform(
                        frame.loc[train_mask, list(ALL_SCALAR_INPUTS)]
                    )
                    union_assessment[preprocessing] = transformer.transform(
                        frame.loc[assessment_mask, list(ALL_SCALAR_INPUTS)]
                    )
                assessment_dates = frame.loc[
                    assessment_mask, "trade_date_ny"
                ].to_numpy()
                for family, configurations in base_candidates.items():
                    for base_configuration in configurations:
                        (
                            x_train,
                            x_assessment,
                            _,
                            _,
                        ) = _classifier_design_from_configuration(
                            frame=frame,
                            raw_profile=raw,
                            configuration=base_configuration,
                            train_mask=train_mask,
                            assessment_mask=assessment_mask,
                            outcome_for_pls=future_range[train_mask],
                            union_transformers=union_transformers,
                            union_train=union_train,
                            union_assessment=union_assessment,
                        )
                        probabilities, failures = _logistic_probability_path(
                            x_train,
                            y_train,
                            x_assessment,
                            config.logistic_c_grid,
                        )
                        losses = _mean_equal_date_log_loss_matrix(
                            assessment_dates,
                            y_assessment,
                            probabilities,
                        )
                        raw_predictor_count = len(
                            base_configuration.get("scalar_columns", ())
                        ) + (
                            len(P0_COLUMNS)
                            if base_configuration.get("profile_family")
                            else 0
                        )
                        component_count = int(
                            base_configuration.get("component_count", 0)
                        )
                        for index, c_value in enumerate(
                            config.logistic_c_grid
                        ):
                            classifier_configuration = {
                                **dict(base_configuration),
                                "C": float(c_value),
                            }
                            tuning_records.append(
                                {
                                    "family": family,
                                    "inner_fold_id": inner.fold_id,
                                    "configuration_json": _configuration_key(
                                        classifier_configuration
                                    ),
                                    "configuration": classifier_configuration,
                                    "fold_loss": float(losses[index]),
                                    "anchor_rank": family_anchor_rank[family],
                                    "raw_predictor_count": raw_predictor_count,
                                    "component_count": component_count,
                                    "c_value": float(c_value),
                                    "convergence_failure": failures[index],
                                }
                            )
            tuning_frame = pd.DataFrame.from_records(tuning_records)
            selected_classifier_configs: dict[str, dict[str, Any]] = {}
            for family in base_candidates:
                selected_classifier_configs[family] = (
                    _select_classifier_configuration(
                        tuning_frame,
                        family=family,
                        min_valid_folds=config.min_valid_inner_folds,
                    )
                )

            assessment_base = frame.loc[
                outer_assessment,
                [
                    "observation_id",
                    "decision_timestamp_utc",
                    "trade_date_ny",
                ],
            ].reset_index(drop=True)
            prevalence = float(np.mean(y_outer_train))
            family_probabilities: dict[str, np.ndarray] = {
                "M0": np.full(
                    int(outer_assessment.sum()),
                    prevalence,
                    dtype=np.float64,
                )
            }
            family_models: dict[
                str, tuple[Any, Any, LogisticRegression | None]
            ] = {"M0": (None, None, None)}
            for family, classifier_configuration in (
                selected_classifier_configs.items()
            ):
                x_train, x_assessment, scalar_transformer, profile_transformer = (
                    _classifier_design_from_configuration(
                        frame=frame,
                        raw_profile=raw,
                        configuration=classifier_configuration,
                        train_mask=outer_train,
                        assessment_mask=outer_assessment,
                        outcome_for_pls=future_range[outer_train],
                    )
                )
                if x_train.shape[1] == 0:
                    probability = np.full(
                        int(outer_assessment.sum()),
                        prevalence,
                        dtype=np.float64,
                    )
                    logistic_model = None
                else:
                    logistic_model = LogisticRegression(
                        C=float(classifier_configuration["C"]),
                        l1_ratio=0.0,
                        solver="lbfgs",
                        class_weight=None,
                        fit_intercept=True,
                        max_iter=2_000,
                        tol=1.0e-6,
                        random_state=None,
                    ).fit(x_train, y_outer_train)
                    probability = logistic_model.predict_proba(
                        x_assessment
                    )[:, 1]
                family_probabilities[family] = probability
                family_models[family] = (
                    scalar_transformer,
                    profile_transformer,
                    logistic_model,
                )
            for family, probability in family_probabilities.items():
                if family == "M0":
                    classifier_configuration = {
                        "scalar_columns": [],
                        "prevalence": prevalence,
                    }
                    convergence_failure = False
                else:
                    classifier_configuration = selected_classifier_configs[
                        family
                    ]
                    convergence_failure = bool(
                        classifier_configuration.get(
                            "convergence_failure", False
                        )
                    )
                fold_loss = float(
                    _mean_equal_date_log_loss_matrix(
                        assessment_base["trade_date_ny"].to_numpy(),
                        y_outer_assessment,
                        probability,
                    )[0]
                )
                core = _core_configuration(classifier_configuration)
                configuration_json = _configuration_key(core)
                configuration_hash = hashlib.sha256(
                    configuration_json.encode("utf-8")
                ).hexdigest()
                choice_records.append(
                    {
                        "session": session,
                        "outer_fold_id": outer.fold_id,
                        "model_family": family,
                        "configuration_json": configuration_json,
                        "configuration_sha256": configuration_hash,
                        "outer_fold_mean_daily_log_loss": fold_loss,
                        "outer_train_prevalence": prevalence,
                        "outer_train_range_threshold_atr": outer_threshold,
                        "valid_inner_folds": classifier_configuration.get(
                            "valid_inner_folds", len(inner_folds)
                        ),
                        "convergence_failure": convergence_failure,
                        "raw_predictor_count": len(
                            classifier_configuration.get(
                                "scalar_columns", ()
                            )
                        )
                        + (
                            len(P0_COLUMNS)
                            if classifier_configuration.get("profile_family")
                            else 0
                        ),
                        "component_count": classifier_configuration.get(
                            "component_count", 0
                        ),
                        "C": classifier_configuration.get("C", np.nan),
                    }
                )
                output = assessment_base.copy()
                output["session"] = session
                output["outer_fold_id"] = outer.fold_id
                output["model_family"] = family
                output["nested_expansion_label"] = y_outer_assessment
                output["future_range_60_atr"] = future_range[
                    outer_assessment
                ]
                output["outer_train_range_threshold_atr"] = outer_threshold
                output["probability"] = probability
                output["configuration_sha256"] = configuration_hash
                oof_frames.append(output)
    return (
        pd.concat(oof_frames, ignore_index=True),
        pd.DataFrame.from_records(choice_records),
        pd.DataFrame.from_records(threshold_records),
    )


def _classification_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1.0e-8,
        1.0 - 1.0e-8,
    )
    if np.unique(y).size < 2 or np.ptp(p) == 0.0:
        return np.nan, np.nan
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(
        C=1.0e12,
        l1_ratio=0.0,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2_000,
        tol=1.0e-8,
    ).fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _daily_log_loss_table(
    group: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for trade_date, date_group in group.groupby(
        "trade_date_ny", sort=True
    ):
        records.append(
            {
                "trade_date_ny": pd.Timestamp(trade_date),
                "daily_log_loss": float(
                    log_loss(
                        date_group["nested_expansion_label"].to_numpy(
                            dtype=np.int64
                        ),
                        np.clip(
                            date_group["probability"].to_numpy(
                                dtype=np.float64
                            ),
                            1.0e-12,
                            1.0 - 1.0e-12,
                        ),
                        labels=[0, 1],
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_expansion_metrics_and_selection(
    oof: pd.DataFrame,
    choices: pd.DataFrame,
    *,
    config: Section5Config,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
]:
    metric_records: list[dict[str, Any]] = []
    reliability_records: list[dict[str, Any]] = []
    selected: dict[str, str] = {}
    anchor_rank = {
        "M1_O1": 0,
        "M1_O0": 1,
        "M0": 1,
        "M2": 1,
        "M3": 1,
        "M_SELECTED_CONTINUOUS": 1,
    }
    for session in SESSIONS:
        session_choices = choices.loc[choices["session"].eq(session)]
        family_summary: list[dict[str, Any]] = []
        for family, choice_group in session_choices.groupby(
            "model_family", sort=True
        ):
            losses = choice_group[
                "outer_fold_mean_daily_log_loss"
            ].to_numpy(dtype=np.float64)
            family_summary.append(
                {
                    "model_family": family,
                    "mean_loss": float(np.mean(losses)),
                    "score_se": (
                        float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
                        if len(losses) > 1
                        else 0.0
                    ),
                    "anchor_rank": anchor_rank[family],
                    "raw_predictor_count": float(
                        choice_group["raw_predictor_count"].mean()
                    ),
                    "component_count": int(
                        choice_group["component_count"].fillna(0).mode().min()
                    ),
                    "c_value": float(
                        choice_group["C"].dropna().mode().min()
                    )
                    if choice_group["C"].notna().any()
                    else 0.0,
                    "configuration_json": family,
                }
            )
        summary = pd.DataFrame.from_records(family_summary)
        selected_row = select_one_standard_error(
            summary,
            metric_column="mean_loss",
            maximize=False,
            complexity_columns=(
                "anchor_rank",
                "raw_predictor_count",
                "component_count",
                "c_value",
            ),
        )
        selected[session] = str(selected_row["model_family"])

        session_oof = oof.loc[oof["session"].eq(session)]
        prevalence_group = session_oof.loc[
            session_oof["model_family"].eq("M0")
        ]
        anchor_group = session_oof.loc[
            session_oof["model_family"].eq("M1_O1")
        ]
        prevalence_brier = brier_score_loss(
            prevalence_group["nested_expansion_label"],
            prevalence_group["probability"],
        )
        prevalence_log_loss = float(
            _daily_log_loss_table(prevalence_group)[
                "daily_log_loss"
            ].mean()
        )
        anchor_brier = brier_score_loss(
            anchor_group["nested_expansion_label"],
            anchor_group["probability"],
        )
        anchor_daily_loss = _daily_log_loss_table(anchor_group).rename(
            columns={"daily_log_loss": "anchor_daily_log_loss"}
        )
        anchor_log_loss = float(
            anchor_daily_loss["anchor_daily_log_loss"].mean()
        )
        for family, group in session_oof.groupby(
            "model_family", sort=True
        ):
            labels = group["nested_expansion_label"].to_numpy(
                dtype=np.int64
            )
            probability = group["probability"].to_numpy(dtype=np.float64)
            calibration_intercept, calibration_slope = (
                _classification_calibration(labels, probability)
            )
            daily_loss = _daily_log_loss_table(group)
            paired_loss = daily_loss.merge(
                anchor_daily_loss,
                on="trade_date_ny",
                how="inner",
                validate="one_to_one",
            )
            paired_loss["paired_log_loss_delta_vs_o1"] = (
                paired_loss["daily_log_loss"]
                - paired_loss["anchor_daily_log_loss"]
            )
            family_index = sorted(
                session_oof["model_family"].unique()
            ).index(family)
            delta_low, delta_high = _date_block_bootstrap_ci(
                paired_loss["paired_log_loss_delta_vs_o1"],
                seed=config.random_seed + 40_000 + family_index,
                replicates=config.bootstrap_replicates,
                confidence=config.bootstrap_confidence,
            )
            p80 = float(
                np.quantile(probability, 0.80, method="linear")
            )
            flagged = probability >= p80
            true_positive = labels.astype(bool)
            precision = (
                float(np.mean(true_positive[flagged]))
                if flagged.any()
                else np.nan
            )
            recall = (
                float(
                    np.sum(true_positive & flagged)
                    / np.sum(true_positive)
                )
                if true_positive.any()
                else np.nan
            )
            prevalence = float(np.mean(labels))
            metric_records.append(
                {
                    "session": session,
                    "model_family": family,
                    "rows": len(group),
                    "valid_oof_dates": int(
                        pd.to_datetime(group["trade_date_ny"])
                        .dt.normalize()
                        .nunique()
                    ),
                    "event_prevalence": prevalence,
                    "roc_auc": (
                        float(roc_auc_score(labels, probability))
                        if np.unique(labels).size == 2
                        else np.nan
                    ),
                    "pr_auc": (
                        float(average_precision_score(labels, probability))
                        if np.unique(labels).size == 2
                        else np.nan
                    ),
                    "brier_score": float(
                        brier_score_loss(labels, probability)
                    ),
                    "prevalence_baseline_brier": float(
                        prevalence_brier
                    ),
                    "o1_baseline_brier": float(anchor_brier),
                    "equal_date_log_loss": float(
                        daily_loss["daily_log_loss"].mean()
                    ),
                    "prevalence_baseline_equal_date_log_loss": (
                        prevalence_log_loss
                    ),
                    "o1_baseline_equal_date_log_loss": anchor_log_loss,
                    "paired_log_loss_delta_vs_o1": float(
                        paired_loss[
                            "paired_log_loss_delta_vs_o1"
                        ].mean()
                    ),
                    "paired_log_loss_delta_ci_low": delta_low,
                    "paired_log_loss_delta_ci_high": delta_high,
                    "calibration_intercept": calibration_intercept,
                    "calibration_slope": calibration_slope,
                    "development_probability_p80": p80,
                    "p80_flagged_rows": int(flagged.sum()),
                    "p80_precision": precision,
                    "p80_recall": recall,
                    "p80_lift": (
                        float(precision / prevalence)
                        if np.isfinite(precision) and prevalence > 0.0
                        else np.nan
                    ),
                    "selected_expansion_family": (
                        family == selected[session]
                    ),
                    "validation_gate_status": "PENDING_LOCKED_SECTION_6",
                }
            )
            ranked = group["probability"].rank(method="first")
            group = group.copy()
            group["probability_decile"] = (
                pd.qcut(
                    ranked,
                    q=10,
                    labels=False,
                    duplicates="drop",
                )
                + 1
            )
            for decile, decile_group in group.groupby(
                "probability_decile", sort=True
            ):
                reliability_records.append(
                    {
                        "session": session,
                        "model_family": family,
                        "probability_decile": int(decile),
                        "rows": len(decile_group),
                        "mean_probability": float(
                            decile_group["probability"].mean()
                        ),
                        "observed_event_rate": float(
                            decile_group[
                                "nested_expansion_label"
                            ].mean()
                        ),
                    }
                )
    return (
        pd.DataFrame.from_records(metric_records),
        pd.DataFrame.from_records(reliability_records),
        selected,
    )


def _derive_final_classifier_configuration(
    choices: pd.DataFrame,
    *,
    session: str,
    family: str,
) -> dict[str, Any]:
    group = choices.loc[
        choices["session"].eq(session)
        & choices["model_family"].eq(family)
    ]
    if family == "M0":
        return {"scalar_columns": []}
    parsed = [json.loads(value) for value in group["configuration_json"]]
    classifier_audit_keys = {
        "inner_mean_log_loss",
        "inner_log_loss_se",
        "valid_inner_folds",
        "convergence_failure",
    }
    signatures = [
        _configuration_key(
            {
                key: value
                for key, value in item.items()
                if key != "C" and key not in classifier_audit_keys
            }
        )
        for item in parsed
    ]
    signature = _deterministic_mode(signatures)
    base = json.loads(signature)
    matching_c = [
        float(item["C"])
        for item, item_signature in zip(parsed, signatures, strict=True)
        if item_signature == signature
    ]
    base["C"] = float(_deterministic_mode(matching_c))
    return base


def _fit_final_expansion_models(
    *,
    inputs: DevelopmentModelInputs,
    choices: pd.DataFrame,
    selected_families: Mapping[str, str],
    config: Section5Config,
) -> tuple[
    dict[str, FittedExpansionModel],
    pd.DataFrame,
]:
    models: dict[str, FittedExpansionModel] = {}
    registry_records: list[dict[str, Any]] = []
    for session in SESSIONS:
        frame, raw = _prepare_horizon_session(
            inputs, session=session, horizon=60
        )
        future_range = frame["future_range_60_atr"].to_numpy(
            dtype=np.float64
        )
        threshold = float(
            np.quantile(
                future_range,
                config.expansion_quantile,
                method="linear",
            )
        )
        labels = (future_range >= threshold).astype(np.int64)
        family = selected_families[session]
        configuration = _derive_final_classifier_configuration(
            choices,
            session=session,
            family=family,
        )
        all_rows = np.ones(len(frame), dtype=bool)
        (
            x_train,
            _,
            scalar_transformer,
            profile_transformer,
        ) = _classifier_design_from_configuration(
            frame=frame,
            raw_profile=raw,
            configuration=configuration,
            train_mask=all_rows,
            assessment_mask=all_rows,
            outcome_for_pls=future_range,
        )
        if family == "M0" or x_train.shape[1] == 0:
            logistic_model = None
            null_prevalence = float(np.mean(labels))
        else:
            logistic_model = LogisticRegression(
                C=float(configuration["C"]),
                l1_ratio=0.0,
                solver="lbfgs",
                class_weight=None,
                fit_intercept=True,
                max_iter=2_000,
                tol=1.0e-6,
                random_state=None,
            ).fit(x_train, labels)
            null_prevalence = None
        scalar_columns = tuple(configuration.get("scalar_columns", ()))
        model = FittedExpansionModel(
            session=session,
            model_family=family,
            configuration=configuration,
            scalar_transformer=scalar_transformer,
            profile_transformer=profile_transformer,
            logistic_model=logistic_model,
            null_prevalence=null_prevalence,
            scalar_columns=scalar_columns,
            expansion_range_threshold=threshold,
            input_schema_sha256=_input_schema_hash(
                scalar_columns,
                configuration.get("profile_representation"),
                configuration.get("component_count"),
            ),
        )
        model_key = f"{session.lower().replace(' ', '_')}|expansion_label_60"
        models[model_key] = model
        configuration_json = _configuration_key(configuration)
        registry_records.append(
            {
                "model_key": model_key,
                "model_role": "expansion_classifier",
                "session": session,
                "target": "expansion_label_60",
                "model_family": family,
                "configuration_json": configuration_json,
                "configuration_sha256": hashlib.sha256(
                    configuration_json.encode("utf-8")
                ).hexdigest(),
                "input_schema_sha256": model.input_schema_sha256,
                "fit_rows": len(frame),
                "fit_dates": int(
                    pd.to_datetime(frame["trade_date_ny"]).dt.normalize().nunique()
                ),
                "expansion_range_threshold_atr": threshold,
                "coefficient_sha256": (
                    hashlib.sha256(
                        np.ascontiguousarray(
                            logistic_model.coef_, dtype=np.float64
                        ).tobytes()
                    ).hexdigest()
                    if logistic_model is not None
                    else ""
                ),
                "transform_state_sha256": (
                    hashlib.sha256(
                        canonical_json(
                            asdict(scalar_transformer.get_state())
                        ).encode("utf-8")
                    ).hexdigest()
                    if scalar_transformer is not None
                    else ""
                ),
                "validation_status": "UNREAD",
            }
        )
    return models, pd.DataFrame.from_records(registry_records)


def _alpha_one_se_from_fold_scores(
    scores: np.ndarray,
    alphas: Sequence[float],
) -> float:
    means = np.nanmean(scores, axis=0)
    valid_counts = np.sum(np.isfinite(scores), axis=0)
    standard_errors = np.divide(
        np.nanstd(scores, axis=0, ddof=1),
        np.sqrt(valid_counts),
        out=np.zeros(len(alphas), dtype=np.float64),
        where=valid_counts > 1,
    )
    best_index = int(np.nanargmax(means))
    threshold = means[best_index] - standard_errors[best_index]
    retained = [
        float(alpha)
        for alpha, mean in zip(alphas, means, strict=True)
        if np.isfinite(mean) and mean >= threshold
    ]
    return min(retained)


def _run_random_subset_diagnostics(
    *,
    inputs: DevelopmentModelInputs,
    continuous_choices: pd.DataFrame,
    config: Section5Config,
    progress: bool,
    sessions: Sequence[str] = SESSIONS,
    targets: Sequence[str] = PRIMARY_CONTINUOUS_TARGETS,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for session in sessions:
        frame, _ = _prepare_horizon_session(
            inputs, session=session, horizon=60
        )
        dates = pd.to_datetime(frame["trade_date_ny"]).dt.normalize()
        outer_folds = build_exact_date_folds(
            dates.unique(),
            initial_train_end=config.outer_initial_train_end,
            assessment_dates=config.outer_assessment_dates,
            step_dates=config.outer_step_dates,
        )
        for target in targets:
            universe = list(BASE_FEATURE_NAMES)
            if target == "forward_return_60_atr":
                universe.extend(INTERACTION_FEATURE_NAMES)
                anchor = tuple(D1_FEATURES)
            else:
                anchor = tuple(O1_FEATURES)
            for outer in outer_folds:
                if progress:
                    print(
                        f"Section 5 random subsets: {session} {target} "
                        f"outer {outer.fold_id + 1}/{len(outer_folds)}",
                        flush=True,
                    )
                m4_choice = continuous_choices.loc[
                    continuous_choices["session"].eq(session)
                    & continuous_choices["target"].eq(target)
                    & continuous_choices["outer_fold_id"].eq(outer.fold_id)
                    & continuous_choices["family"].eq("M4")
                    & continuous_choices["status"].eq("COMPLETE")
                ].iloc[0]
                m4_configuration = json.loads(
                    m4_choice["configuration_json"]
                )
                preprocessing = str(
                    m4_configuration.get("preprocessing", "S0")
                )
                selected_support = set(
                    m4_configuration.get("scalar_columns", ())
                )
                selected_units = [
                    unit for unit in universe if unit in selected_support
                ]
                subset_size = len(selected_units)
                draw_units: list[tuple[str, ...]] = []
                for draw_id in range(config.random_subset_draws):
                    rng = np.random.default_rng(
                        config.random_seed
                        + 1_000 * outer.fold_id
                        + draw_id
                    )
                    if subset_size:
                        sample = tuple(
                            sorted(
                                rng.choice(
                                    universe,
                                    size=subset_size,
                                    replace=False,
                                ).tolist()
                            )
                        )
                    else:
                        sample = ()
                    draw_units.append(sample)
                candidate_units = [tuple(sorted(selected_units)), *draw_units]
                inner_folds = build_exact_date_folds(
                    outer.train_dates,
                    initial_train_end=config.inner_initial_train_end,
                    assessment_dates=config.inner_assessment_dates,
                    step_dates=config.inner_step_dates,
                )
                score_cube = np.full(
                    (
                        len(candidate_units),
                        len(inner_folds),
                        len(config.ridge_alpha_grid),
                    ),
                    np.nan,
                    dtype=np.float64,
                )
                for inner_index, inner in enumerate(inner_folds):
                    train_mask, assessment_mask, _, _ = purged_fold_masks(
                        frame,
                        inner,
                        exit_timestamp_column="exit_timestamp_utc_60",
                    )
                    transformer = ScalarFoldTransformer(
                        preprocessing=preprocessing
                    ).fit(
                        frame.loc[
                            train_mask,
                            list(ALL_SCALAR_INPUTS),
                        ]
                    )
                    full_train = transformer.transform(
                        frame.loc[train_mask, list(ALL_SCALAR_INPUTS)]
                    )
                    full_assessment = transformer.transform(
                        frame.loc[
                            assessment_mask,
                            list(ALL_SCALAR_INPUTS),
                        ]
                    )
                    y_train = frame.loc[
                        train_mask, target
                    ].to_numpy(dtype=np.float64)
                    y_assessment = frame.loc[
                        assessment_mask, target
                    ].to_numpy(dtype=np.float64)
                    assessment_dates = frame.loc[
                        assessment_mask, "trade_date_ny"
                    ].to_numpy()
                    for candidate_index, units in enumerate(
                        candidate_units
                    ):
                        columns = _columns_with_indicators(
                            hierarchy_closed_support([*anchor, *units])
                        )
                        positions = [
                            transformer.output_columns_.index(name)
                            for name in columns
                            if name in transformer.output_columns_
                        ]
                        score_cube[candidate_index, inner_index, :] = (
                            _score_ridge_path(
                                full_train[:, positions],
                                y_train,
                                full_assessment[:, positions],
                                y_assessment,
                                assessment_dates,
                                config.ridge_alpha_grid,
                                min_observations=config.min_daily_observations,
                            )
                        )
                selected_alphas = [
                    _alpha_one_se_from_fold_scores(
                        score_cube[index],
                        config.ridge_alpha_grid,
                    )
                    for index in range(len(candidate_units))
                ]
                outer_train, outer_assessment, _, _ = purged_fold_masks(
                    frame,
                    outer,
                    exit_timestamp_column="exit_timestamp_utc_60",
                )
                transformer = ScalarFoldTransformer(
                    preprocessing=preprocessing
                ).fit(
                    frame.loc[
                        outer_train,
                        list(ALL_SCALAR_INPUTS),
                    ]
                )
                full_train = transformer.transform(
                    frame.loc[outer_train, list(ALL_SCALAR_INPUTS)]
                )
                full_assessment = transformer.transform(
                    frame.loc[
                        outer_assessment,
                        list(ALL_SCALAR_INPUTS),
                    ]
                )
                y_train = frame.loc[
                    outer_train, target
                ].to_numpy(dtype=np.float64)
                y_assessment = frame.loc[
                    outer_assessment, target
                ].to_numpy(dtype=np.float64)
                assessment_dates = frame.loc[
                    outer_assessment, "trade_date_ny"
                ].to_numpy()
                outer_scores: list[float] = []
                for units, alpha in zip(
                    candidate_units, selected_alphas, strict=True
                ):
                    columns = _columns_with_indicators(
                        hierarchy_closed_support([*anchor, *units])
                    )
                    positions = [
                        transformer.output_columns_.index(name)
                        for name in columns
                        if name in transformer.output_columns_
                    ]
                    model = Ridge(
                        alpha=alpha,
                        fit_intercept=True,
                        solver="svd",
                    ).fit(full_train[:, positions], y_train)
                    prediction = model.predict(
                        full_assessment[:, positions]
                    )
                    outer_scores.append(
                        float(
                            _mean_daily_spearman_matrix(
                                assessment_dates,
                                prediction,
                                y_assessment,
                                min_observations=config.min_daily_observations,
                            )[0]
                        )
                    )
                selected_score = outer_scores[0]
                random_scores = np.asarray(outer_scores[1:])
                percentile = float(
                    100.0 * np.mean(random_scores <= selected_score)
                )
                for draw_id, (
                    units,
                    alpha,
                    random_score,
                ) in enumerate(
                    zip(
                        draw_units,
                        selected_alphas[1:],
                        outer_scores[1:],
                        strict=True,
                    )
                ):
                    records.append(
                        {
                            "session": session,
                            "target": target,
                            "outer_fold_id": outer.fold_id,
                            "draw_id": draw_id,
                            "seed": (
                                config.random_seed
                                + 1_000 * outer.fold_id
                                + draw_id
                            ),
                            "selected_unit_count": subset_size,
                            "selected_units": "|".join(
                                sorted(selected_units)
                            ),
                            "random_units": "|".join(units),
                            "selected_ridge_alpha": selected_alphas[0],
                            "random_ridge_alpha": alpha,
                            "selected_outer_daily_ic": selected_score,
                            "random_outer_daily_ic": random_score,
                            "paired_daily_ic_difference": (
                                selected_score - random_score
                            ),
                            "selected_subset_percentile": percentile,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _aggregate_outer_elastic_stability(
    outer_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (session, target, feature), group in outer_coefficients.groupby(
        ["session", "target", "feature_name"], sort=True
    ):
        selected = group["outer_selected"].astype(bool).to_numpy()
        coefficient = np.where(
            selected,
            group["outer_median_standardized_coefficient"].to_numpy(
                dtype=np.float64
            ),
            0.0,
        )
        nonzero = coefficient[coefficient != 0.0]
        frequency = float(np.mean(selected))
        sign_agreement = (
            float(
                max(
                    np.mean(nonzero > 0.0),
                    np.mean(nonzero < 0.0),
                )
            )
            if len(nonzero)
            else np.nan
        )
        q25, q75 = (
            np.quantile(nonzero, [0.25, 0.75], method="linear")
            if len(nonzero)
            else (0.0, 0.0)
        )
        interaction_parent_rule = True
        if feature in INTERACTION_PARENT_MAP:
            selected_folds = set(
                group.loc[group["outer_selected"], "outer_fold_id"]
            )
            for parent in INTERACTION_PARENT_MAP[feature]:
                parent_folds = set(
                    outer_coefficients.loc[
                        outer_coefficients["session"].eq(session)
                        & outer_coefficients["target"].eq(target)
                        & outer_coefficients["feature_name"].eq(parent)
                        & outer_coefficients["outer_selected"],
                        "outer_fold_id",
                    ]
                )
                if not selected_folds.issubset(parent_folds):
                    interaction_parent_rule = False
        stable = bool(
            frequency >= 0.60
            and np.isfinite(sign_agreement)
            and sign_agreement >= 0.75
            and interaction_parent_rule
        )
        records.append(
            {
                "session": session,
                "target": target,
                "feature_name": feature,
                "outer_fold_count": int(group["outer_fold_id"].nunique()),
                "selection_frequency": frequency,
                "sign_agreement": sign_agreement,
                "median_standardized_coefficient": (
                    float(np.median(nonzero)) if len(nonzero) else 0.0
                ),
                "coefficient_q25": float(q25),
                "coefficient_q75": float(q75),
                "interaction_parent_rule": interaction_parent_rule,
                "stable_term": stable,
                "convergence_failure": bool(
                    group["convergence_failure"].any()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _components_for_variance(
    cumulative: np.ndarray,
    threshold: float,
) -> int:
    positions = np.flatnonzero(cumulative >= threshold)
    return int(positions[0] + 1) if len(positions) else int(len(cumulative))


def _component_cosine_stability(
    current: np.ndarray,
    previous: np.ndarray | None,
    *,
    max_components: int,
) -> tuple[float, float]:
    if previous is None:
        return np.nan, np.nan
    count = min(max_components, len(current), len(previous))
    similarities: list[float] = []
    for index in range(count):
        left = current[index]
        right = previous[index]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        similarities.append(
            float((left @ right) / denominator)
            if denominator > 0.0
            else np.nan
        )
    finite = np.asarray(similarities, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return (
        float(np.mean(finite)) if len(finite) else np.nan,
        float(np.min(finite)) if len(finite) else np.nan,
    )


def _build_outer_profile_stability(
    *,
    inputs: DevelopmentModelInputs,
    config: Section5Config,
    progress: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for session in SESSIONS:
        for horizon in (60, 30):
            frame, raw = _prepare_horizon_session(
                inputs, session=session, horizon=horizon
            )
            dates = pd.to_datetime(frame["trade_date_ny"]).dt.normalize()
            folds = build_exact_date_folds(
                dates.unique(),
                initial_train_end=config.outer_initial_train_end,
                assessment_dates=config.outer_assessment_dates,
                step_dates=config.outer_step_dates,
            )
            targets = (
                ("forward_return_60_atr", "future_range_60_atr")
                if horizon == 60
                else ("forward_return_30_atr",)
            )
            previous_pca: dict[str, np.ndarray] = {}
            previous_pls: dict[str, np.ndarray] = {}
            for outer in folds:
                if progress:
                    print(
                        f"Section 5 profile stability: {session} h={horizon} "
                        f"outer {outer.fold_id + 1}/{len(folds)}",
                        flush=True,
                    )
                train_mask, _, _, _ = purged_fold_masks(
                    frame,
                    outer,
                    exit_timestamp_column=f"exit_timestamp_utc_{horizon}",
                )
                for representation in PROFILE_REPRESENTATIONS:
                    transformed, _ = transform_profile_representation(
                        raw[train_mask], representation
                    )
                    mean = transformed.mean(axis=0)
                    scale = transformed.std(axis=0, ddof=0)
                    constant = scale == 0.0
                    scale[constant] = 1.0
                    standardized = (transformed - mean) / scale
                    standardized[:, constant] = 0.0
                    pca = PCA(
                        n_components=None,
                        whiten=False,
                        svd_solver="full",
                    ).fit(standardized)
                    signs = _orientation_signs(pca.components_)
                    oriented = pca.components_ * signs[:, None]
                    cumulative = np.cumsum(
                        pca.explained_variance_ratio_
                    )
                    stability_mean, stability_min = (
                        _component_cosine_stability(
                            oriented,
                            previous_pca.get(representation),
                            max_components=max(config.pca_component_grid),
                        )
                    )
                    previous_pca[representation] = oriented
                    records.append(
                        {
                            "diagnostic_scope": "OUTER_TRAIN_FULL_PCA",
                            "session": session,
                            "horizon_minutes": horizon,
                            "target": "OUTCOME_FREE",
                            "outer_fold_id": outer.fold_id,
                            "profile_family": "PCA",
                            "profile_representation": representation,
                            "component_count": len(
                                pca.explained_variance_ratio_
                            ),
                            "components_for_80pct_variance": (
                                _components_for_variance(cumulative, 0.80)
                            ),
                            "components_for_90pct_variance": (
                                _components_for_variance(cumulative, 0.90)
                            ),
                            "components_for_95pct_variance": (
                                _components_for_variance(cumulative, 0.95)
                            ),
                            "cumulative_variance_at_12": float(
                                cumulative[
                                    min(
                                        12,
                                        len(cumulative),
                                    )
                                    - 1
                                ]
                            ),
                            "effective_rank": int(
                                np.sum(
                                    pca.singular_values_
                                    > (
                                        np.finfo(np.float64).eps
                                        * max(standardized.shape)
                                        * pca.singular_values_[0]
                                    )
                                )
                            ),
                            "loading_cosine_mean_vs_previous_outer": (
                                stability_mean
                            ),
                            "loading_cosine_min_vs_previous_outer": (
                                stability_min
                            ),
                            "converged": True,
                            "fit_rows": int(train_mask.sum()),
                            "oriented_loadings_sha256": hashlib.sha256(
                                np.ascontiguousarray(
                                    oriented[: max(config.pca_component_grid)],
                                    dtype=np.float64,
                                ).tobytes()
                            ).hexdigest(),
                        }
                    )
                for target in targets:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        pls = Section5PLSProfileTransformer(
                            n_components=max(config.pls_component_grid)
                        ).fit(
                            raw[train_mask],
                            frame.loc[train_mask, target].to_numpy(
                                dtype=np.float64
                            ),
                        )
                    oriented = pls.oriented_x_weights_.T
                    stability_mean, stability_min = (
                        _component_cosine_stability(
                            oriented,
                            previous_pls.get(target),
                            max_components=max(config.pls_component_grid),
                        )
                    )
                    previous_pls[target] = oriented
                    convergence_failure = any(
                        "converge" in str(item.message).lower()
                        or "maximum number of iterations"
                        in str(item.message).lower()
                        for item in caught
                    )
                    records.append(
                        {
                            "diagnostic_scope": "OUTER_TRAIN_P2_PLS",
                            "session": session,
                            "horizon_minutes": horizon,
                            "target": target,
                            "outer_fold_id": outer.fold_id,
                            "profile_family": "PLS",
                            "profile_representation": "P2",
                            "component_count": max(
                                config.pls_component_grid
                            ),
                            "components_for_80pct_variance": np.nan,
                            "components_for_90pct_variance": np.nan,
                            "components_for_95pct_variance": np.nan,
                            "cumulative_variance_at_12": np.nan,
                            "effective_rank": np.nan,
                            "loading_cosine_mean_vs_previous_outer": (
                                stability_mean
                            ),
                            "loading_cosine_min_vs_previous_outer": (
                                stability_min
                            ),
                            "converged": not convergence_failure,
                            "fit_rows": int(train_mask.sum()),
                            "iterations": "|".join(
                                str(int(value))
                                for value in np.atleast_1d(
                                    pls.pls_.n_iter_
                                )
                            ),
                            "oriented_loadings_sha256": hashlib.sha256(
                                np.ascontiguousarray(
                                    oriented, dtype=np.float64
                                ).tobytes()
                            ).hexdigest(),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _build_frozen_thresholds_and_policy(
    *,
    continuous_oof: pd.DataFrame,
    continuous_comparison: pd.DataFrame,
    selected_continuous: Mapping[tuple[str, str], str],
    expansion_oof: pd.DataFrame,
    selected_expansion: Mapping[str, str],
    frozen_model_registry: pd.DataFrame,
    config: Section5Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    threshold_records: list[dict[str, Any]] = []
    direction_gate_by_session: dict[str, bool] = {}
    for session in SESSIONS:
        direction_family = selected_continuous[
            (session, "forward_return_60_atr")
        ]
        direction = continuous_oof.loc[
            continuous_oof["session"].eq(session)
            & continuous_oof["target"].eq("forward_return_60_atr")
            & continuous_oof["model_family"].eq(direction_family)
        ]
        p90 = float(
            np.quantile(
                np.abs(direction["prediction"].to_numpy(dtype=np.float64)),
                config.direction_score_quantile,
                method="linear",
            )
        )
        expansion_family = selected_expansion[session]
        expansion = expansion_oof.loc[
            expansion_oof["session"].eq(session)
            & expansion_oof["model_family"].eq(expansion_family)
        ]
        p80 = float(
            np.quantile(
                expansion["probability"].to_numpy(dtype=np.float64),
                0.80,
                method="linear",
            )
        )
        comparison_row = continuous_comparison.loc[
            continuous_comparison["session"].eq(session)
            & continuous_comparison["target"].eq("forward_return_60_atr")
            & continuous_comparison["model_family"].eq(direction_family)
        ].iloc[0]
        development_gate = bool(
            comparison_row["development_model_gate_pass"]
        )
        direction_gate_by_session[session] = development_gate
        threshold_records.extend(
            [
                {
                    "session": session,
                    "threshold_role": "direction_absolute_prediction_p90",
                    "quantile": config.direction_score_quantile,
                    "value": p90,
                    "fit_partition": "Development outer OOF",
                    "fit_rows": len(direction),
                    "fit_dates": int(
                        pd.to_datetime(direction["trade_date_ny"])
                        .dt.normalize()
                        .nunique()
                    ),
                    "model_family": direction_family,
                    "activated_for_policy": development_gate,
                },
                {
                    "session": session,
                    "threshold_role": "expansion_probability_p80",
                    "quantile": 0.80,
                    "value": p80,
                    "fit_partition": "Development outer OOF",
                    "fit_rows": len(expansion),
                    "fit_dates": int(
                        pd.to_datetime(expansion["trade_date_ny"])
                        .dt.normalize()
                        .nunique()
                    ),
                    "model_family": expansion_family,
                    "activated_for_policy": development_gate,
                },
            ]
        )
    thresholds = pd.DataFrame.from_records(threshold_records)
    frozen = build_frozen_config()
    policy_eligible = bool(all(direction_gate_by_session.values()))
    reason = (
        ""
        if policy_eligible
        else (
            "NO_DIRECTIONAL_MODEL_HAS_COMPLETE_DEVELOPMENT_SUPPORT: "
            "the exact six outer folds provide 378 valid OOF dates per "
            "session, below the frozen 400-date model gate."
        )
    )
    payload = {
        "schema_version": "1.0.0",
        "research_id": "fes_project1",
        "freeze_stage": "pre_validation_section5",
        "frozen_config_sha256": config_sha256(),
        "feature_configuration": frozen["features"],
        "interaction_configuration": frozen["interactions"],
        "transform_configuration": frozen["preprocessing"],
        "profile_configuration": frozen["profile"],
        "continuous_models": frozen_model_registry.loc[
            frozen_model_registry["model_role"].eq("continuous"),
            [
                "model_key",
                "model_family",
                "configuration_sha256",
                "input_schema_sha256",
            ],
        ].to_dict(orient="records"),
        "expansion_models": frozen_model_registry.loc[
            frozen_model_registry["model_role"].eq(
                "expansion_classifier"
            ),
            [
                "model_key",
                "model_family",
                "configuration_sha256",
                "input_schema_sha256",
            ],
        ].to_dict(orient="records"),
        "numerical_thresholds": thresholds.to_dict(orient="records"),
        "policy_variants": frozen["policy"]["variants"],
        "policy_status": (
            "FROZEN_CANDIDATES_PENDING_VALIDATION"
            if policy_eligible
            else "FROZEN_NO_POLICY"
        ),
        "policy_execution_allowed": policy_eligible,
        "no_policy_reason": reason,
        "cost_configuration": frozen["backtest"]["cost_ticks"],
        "costs_are_proxy_assumptions": True,
        "sequencing_configuration": {
            "global_london_new_york": True,
            "one_position_globally": True,
            "candidate_order": "decision_timestamp_utc ascending",
        },
        "risk_and_exit_configuration": frozen["backtest"],
        "evaluation_configuration": {
            "predictive_metrics": "prompt Section 19",
            "model_gate": frozen["model_gate"],
            "economic_gate": frozen["economic_gate"],
            "bootstrap": frozen["randomness"],
        },
        "validation_outcomes_status": "UNREAD",
        "historical_final_status": "UNREAD",
        "economics_status": "NOT_RUN_SECTION5",
    }
    policy_hash = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    payload["pre_validation_policy_sha256"] = policy_hash
    return thresholds, payload


def _complete_section5_trial_ledger(
    *,
    project_root: Path,
    continuous_comparison: pd.DataFrame,
    expansion_metrics: pd.DataFrame,
    policy_freeze: Mapping[str, Any],
) -> pd.DataFrame:
    path = (
        project_root
        / "data/processed/statistical_research/fes_project1/v1/"
        "section4_complete_trial_ledger.parquet"
    )
    ledger = pd.read_parquet(path).copy()
    for index, row in ledger.loc[
        ledger["notebook_section"].eq(5)
    ].iterrows():
        if row["family"] == "continuous_model":
            match = continuous_comparison.loc[
                continuous_comparison["session"].eq(row["session"])
                & continuous_comparison["target"].eq(row["target"])
                & continuous_comparison["model_family"].eq(
                    row["candidate"]
                )
            ]
            if len(match) != 1:
                raise AssertionError(
                    f"Continuous ledger row did not match once: {row['trial_id']}"
                )
            result = match.iloc[0]
            ledger.loc[index, "status"] = "COMPLETE"
            ledger.loc[index, "effect"] = result["daily_ic_mean"]
            ledger.loc[index, "ci_low"] = result["daily_ic_ci_low"]
            ledger.loc[index, "ci_high"] = result["daily_ic_ci_high"]
            ledger.loc[index, "development_label"] = (
                "DEV_MODEL_SUPPORT"
                if result["development_model_gate_pass"]
                else "DEV_NO_MODEL_SUPPORT"
            )
            failed = []
            if result["valid_oof_dates"] < 400:
                failed.append("DEVELOPMENT_OOF_DATES_LT_400")
            if result["paired_daily_ic_delta_mean"] < 0.01:
                failed.append("PAIRED_DELTA_LT_0_01")
            if result["paired_delta_ci_low"] <= 0.0:
                failed.append("PAIRED_DELTA_CI_NOT_ABOVE_ZERO")
            if not result["complexity_stable"]:
                failed.append("COMPLEXITY_NOT_STABLE")
            ledger.loc[index, "failed_gates"] = "|".join(failed)
            ledger.loc[index, "notes"] = (
                "Validation gates pending locked Section 6 batch."
            )
        elif row["family"] == "expansion_classifier":
            family = str(row["candidate"])
            match = expansion_metrics.loc[
                expansion_metrics["session"].eq(row["session"])
                & expansion_metrics["model_family"].eq(family)
            ]
            if len(match) != 1:
                raise AssertionError(
                    f"Classifier ledger row did not match once: {row['trial_id']}"
                )
            result = match.iloc[0]
            ledger.loc[index, "status"] = "COMPLETE"
            ledger.loc[index, "effect"] = result[
                "equal_date_log_loss"
            ]
            ledger.loc[index, "development_label"] = (
                "DEV_SELECTED_EXPANSION"
                if result["selected_expansion_family"]
                else "DEV_NOT_SELECTED_EXPANSION"
            )
            ledger.loc[index, "failed_gates"] = (
                "VALIDATION_CLASSIFIER_GATES_PENDING"
            )
            ledger.loc[index, "notes"] = (
                "Probability model only; does not create direction."
            )
        elif row["family"] == "policy_freeze":
            ledger.loc[index, "status"] = "COMPLETE"
            ledger.loc[index, "development_label"] = str(
                policy_freeze["policy_status"]
            )
            ledger.loc[index, "failed_gates"] = (
                ""
                if policy_freeze["policy_execution_allowed"]
                else "NO_DIRECTIONAL_MODEL_WITH_DEV_SUPPORT"
            )
            ledger.loc[index, "notes"] = (
                "No P&L inspected. Numerical thresholds frozen before Validation."
            )
    completed = ledger.loc[
        ledger["notebook_section"].eq(5), "status"
    ].eq("COMPLETE")
    if int(completed.sum()) != 68:
        raise AssertionError(
            f"Expected 68 complete Section 5 ledger rows; got {completed.sum()}."
        )
    return ledger


def _cache_contract_hash(config: Section5Config) -> str:
    return hashlib.sha256(
        canonical_json(asdict(config)).encode("utf-8")
    ).hexdigest()


def _load_or_run_continuous_groups(
    *,
    project_root: Path,
    inputs: DevelopmentModelInputs,
    config: Section5Config,
    use_cache: bool,
    progress: bool,
) -> list[_GroupResult]:
    data_dir = (
        project_root
        / "data/processed/statistical_research/fes_project1/v1"
    )
    contract_hash = _cache_contract_hash(config)
    groups: list[_GroupResult] = []
    for session in SESSIONS:
        for horizon in (60, 30):
            session_code = session.lower().replace(" ", "_")
            cache_path = (
                data_dir
                / f"section5_cache_continuous_{session_code}_{horizon}.joblib"
            )
            group = None
            if use_cache and cache_path.exists():
                cached = joblib.load(cache_path)
                if cached.get("contract_sha256") == contract_hash:
                    group = cached["group"]
                    if progress:
                        print(
                            f"Section 5 cache: loaded {session} h={horizon}",
                            flush=True,
                        )
            if group is None:
                group = _run_continuous_group(
                    inputs=inputs,
                    session=session,
                    horizon=horizon,
                    config=config,
                    progress=progress,
                )
                if isinstance(group.oof_records, list):
                    group.oof_records = pd.concat(
                        group.oof_records, ignore_index=True
                    )
                joblib.dump(
                    {
                        "contract_sha256": contract_hash,
                        "group": group,
                    },
                    cache_path,
                    compress=3,
                )
            elif isinstance(group.oof_records, list):
                group.oof_records = pd.concat(
                    group.oof_records, ignore_index=True
                )
            groups.append(group)
    return groups


def build_section5_development(
    project_root: Path,
    *,
    config: Section5Config | None = None,
    use_cache: bool = True,
    progress: bool = True,
) -> Section5Result:
    """Run all Section 5 Development-only trials and freeze model policy state."""

    root = Path(project_root)
    cfg = config or Section5Config()
    inputs = load_development_model_inputs(root)
    groups = _load_or_run_continuous_groups(
        project_root=root,
        inputs=inputs,
        config=cfg,
        use_cache=use_cache,
        progress=progress,
    )
    fold_table = pd.DataFrame.from_records(
        [
            record
            for group in groups
            for record in group.fold_records
        ]
    )
    continuous_oof = pd.concat(
        [group.oof_records for group in groups],
        ignore_index=True,
    )
    choices = pd.DataFrame.from_records(
        [
            record
            for group in groups
            for record in group.outer_choice_records
        ]
    )
    if "selection_status" not in choices.columns:
        choices["selection_status"] = np.where(
            choices["outer_fold_daily_ic"].notna(),
            "COMPLETE_FINITE_OUTER_IC",
            "COMPLETE_CONSTANT_MODEL_NO_FINITE_OUTER_IC",
        )
    elastic_outer_raw = pd.DataFrame.from_records(
        [
            record
            for group in groups
            for record in group.elastic_outer_coefficient_records
        ]
    )
    elastic_stability = _aggregate_outer_elastic_stability(
        elastic_outer_raw
    )
    cluster_stability = pd.DataFrame.from_records(
        [
            record
            for group in groups
            for record in group.cluster_records
        ]
    )
    substitutions = pd.DataFrame.from_records(
        [
            record
            for group in groups
            for record in group.substitution_records
        ]
    )
    profile_diagnostics = pd.DataFrame.from_records(
        [
            record
            for group in groups
            for record in group.profile_records
        ]
    )
    profile_stability_cache = (
        root
        / "data/processed/statistical_research/fes_project1/v1/"
        "section5_cache_profile_stability.joblib"
    )
    profile_stability_contract = hashlib.sha256(
        f"{_cache_contract_hash(cfg)}|outer_profile_stability_v1".encode(
            "utf-8"
        )
    ).hexdigest()
    outer_profile_stability = None
    if use_cache and profile_stability_cache.exists():
        cached_profile_stability = joblib.load(
            profile_stability_cache
        )
        if (
            cached_profile_stability.get("contract_sha256")
            == profile_stability_contract
        ):
            outer_profile_stability = cached_profile_stability["table"]
    if outer_profile_stability is None:
        outer_profile_stability = _build_outer_profile_stability(
            inputs=inputs,
            config=cfg,
            progress=progress,
        )
        joblib.dump(
            {
                "contract_sha256": profile_stability_contract,
                "table": outer_profile_stability,
            },
            profile_stability_cache,
            compress=3,
        )
    profile_diagnostics = pd.concat(
        [profile_diagnostics, outer_profile_stability],
        ignore_index=True,
        sort=False,
    )
    continuous_threshold_audit = pd.DataFrame.from_records(
        [
            record
            for group in groups
            for record in group.threshold_records
        ]
    )
    continuous_daily, paired_daily = _continuous_daily_tables(
        continuous_oof,
        min_observations=cfg.min_daily_observations,
    )
    (
        continuous_metrics,
        continuous_deciles,
        continuous_subperiods,
    ) = _build_continuous_metrics(
        continuous_oof,
        continuous_daily,
        paired_daily,
        config=cfg,
    )
    (
        continuous_comparison,
        selected_continuous,
    ) = _build_model_comparison_and_selection(
        choices,
        continuous_metrics,
        continuous_subperiods,
        elastic_outer_raw,
        config=cfg,
    )

    data_dir = (
        root / "data/processed/statistical_research/fes_project1/v1"
    )
    continuous_choice_hash = hashlib.sha256(
        choices.sort_values(
            ["session", "target", "outer_fold_id", "family"]
        ).to_json(orient="records", date_format="iso").encode("utf-8")
    ).hexdigest()
    random_contract_base = hashlib.sha256(
        f"{_cache_contract_hash(cfg)}|{continuous_choice_hash}".encode(
            "utf-8"
        )
    ).hexdigest()
    random_tables: list[pd.DataFrame] = []
    for random_session in SESSIONS:
        for random_target in PRIMARY_CONTINUOUS_TARGETS:
            random_code = random_session.lower().replace(" ", "_")
            random_cache = (
                data_dir
                / f"section5_cache_random_subsets_{random_code}_"
                f"{random_target}.joblib"
            )
            random_contract = hashlib.sha256(
                f"{random_contract_base}|{random_session}|{random_target}".encode(
                    "utf-8"
                )
            ).hexdigest()
            random_table = None
            if use_cache and random_cache.exists():
                cached_random = joblib.load(random_cache)
                if (
                    cached_random.get("contract_sha256")
                    == random_contract
                ):
                    random_table = cached_random["table"]
            if random_table is None:
                random_table = _run_random_subset_diagnostics(
                    inputs=inputs,
                    continuous_choices=choices,
                    config=cfg,
                    progress=progress,
                    sessions=(random_session,),
                    targets=(random_target,),
                )
                joblib.dump(
                    {
                        "contract_sha256": random_contract,
                        "table": random_table,
                    },
                    random_cache,
                    compress=3,
                )
            random_tables.append(random_table)
    random_subsets = pd.concat(random_tables, ignore_index=True)

    expansion_contract_base = hashlib.sha256(
        f"{_cache_contract_hash(cfg)}|{continuous_choice_hash}|"
        f"{canonical_json({str(k): v for k, v in selected_continuous.items()})}".encode(
            "utf-8"
        )
    ).hexdigest()
    expansion_oof_tables: list[pd.DataFrame] = []
    expansion_choice_tables: list[pd.DataFrame] = []
    expansion_threshold_tables: list[pd.DataFrame] = []
    for expansion_session in SESSIONS:
        expansion_code = expansion_session.lower().replace(" ", "_")
        expansion_cache = (
            data_dir / f"section5_cache_expansion_{expansion_code}.joblib"
        )
        expansion_contract = hashlib.sha256(
            f"{expansion_contract_base}|{expansion_session}".encode("utf-8")
        ).hexdigest()
        expansion_cached = None
        if use_cache and expansion_cache.exists():
            candidate = joblib.load(expansion_cache)
            if candidate.get("contract_sha256") == expansion_contract:
                expansion_cached = candidate
        if expansion_cached is None:
            (
                expansion_oof_session,
                expansion_choices_session,
                classifier_threshold_session,
            ) = _run_expansion_classifiers(
                inputs=inputs,
                continuous_choices=choices,
                selected_continuous_families=selected_continuous,
                config=cfg,
                progress=progress,
                sessions=(expansion_session,),
            )
            expansion_cached = {
                "contract_sha256": expansion_contract,
                "oof": expansion_oof_session,
                "choices": expansion_choices_session,
                "threshold_audit": classifier_threshold_session,
            }
            joblib.dump(expansion_cached, expansion_cache, compress=3)
        expansion_oof_tables.append(expansion_cached["oof"])
        expansion_choice_tables.append(expansion_cached["choices"])
        expansion_threshold_tables.append(
            expansion_cached["threshold_audit"]
        )
    expansion_oof = pd.concat(expansion_oof_tables, ignore_index=True)
    expansion_choices = pd.concat(
        expansion_choice_tables, ignore_index=True
    )
    classifier_threshold_audit = pd.concat(
        expansion_threshold_tables, ignore_index=True
    )
    (
        expansion_metrics,
        expansion_reliability,
        selected_expansion,
    ) = _build_expansion_metrics_and_selection(
        expansion_oof,
        expansion_choices,
        config=cfg,
    )

    continuous_models, continuous_registry = (
        _fit_final_continuous_models(
            inputs=inputs,
            choices=choices,
            selected_families=selected_continuous,
            config=cfg,
        )
    )
    expansion_models, expansion_registry = _fit_final_expansion_models(
        inputs=inputs,
        choices=expansion_choices,
        selected_families=selected_expansion,
        config=cfg,
    )
    model_registry = pd.concat(
        [continuous_registry, expansion_registry],
        ignore_index=True,
    )
    frozen_thresholds, policy_freeze = (
        _build_frozen_thresholds_and_policy(
            continuous_oof=continuous_oof,
            continuous_comparison=continuous_comparison,
            selected_continuous=selected_continuous,
            expansion_oof=expansion_oof,
            selected_expansion=selected_expansion,
            frozen_model_registry=model_registry,
            config=cfg,
        )
    )
    trial_ledger = _complete_section5_trial_ledger(
        project_root=root,
        continuous_comparison=continuous_comparison,
        expansion_metrics=expansion_metrics,
        policy_freeze=policy_freeze,
    )
    threshold_audit = pd.concat(
        [continuous_threshold_audit, classifier_threshold_audit],
        ignore_index=True,
    )
    return Section5Result(
        fold_table=fold_table,
        threshold_audit=threshold_audit,
        support_coverage=inputs.support_coverage,
        continuous_oof_predictions=continuous_oof,
        continuous_model_comparison=continuous_comparison,
        continuous_paired_daily=paired_daily,
        continuous_metrics=continuous_metrics,
        continuous_deciles=continuous_deciles,
        continuous_subperiods=continuous_subperiods,
        outer_choices=choices,
        elastic_stability=elastic_stability,
        elastic_cluster_stability=cluster_stability,
        elastic_member_substitution=substitutions,
        random_subset_diagnostics=random_subsets,
        profile_diagnostics=profile_diagnostics,
        expansion_oof_predictions=expansion_oof,
        expansion_metrics=expansion_metrics,
        expansion_reliability=expansion_reliability,
        frozen_thresholds=frozen_thresholds,
        frozen_model_registry=model_registry,
        policy_freeze=policy_freeze,
        complete_trial_ledger=trial_ledger,
        access_audit=inputs.access_audit,
        frozen_continuous_models=continuous_models,
        frozen_expansion_models=expansion_models,
        config=cfg,
    )


def _save_continuous_diagnostic_figure(
    result: Section5Result,
    path: Path,
) -> None:
    selected = result.continuous_model_comparison.loc[
        result.continuous_model_comparison["selected_primary_family"]
    ]
    figure, axes = plt.subplots(3, 2, figsize=(13, 14), constrained_layout=True)
    for axis, (_, selection) in zip(
        axes.ravel(),
        selected.sort_values(["target", "session"]).iterrows(),
        strict=True,
    ):
        group = result.continuous_oof_predictions.loc[
            result.continuous_oof_predictions["session"].eq(
                selection["session"]
            )
            & result.continuous_oof_predictions["target"].eq(
                selection["target"]
            )
            & result.continuous_oof_predictions["model_family"].eq(
                selection["model_family"]
            )
        ]
        if len(group) > 50_000:
            group = group.sample(
                50_000,
                random_state=result.config.random_seed,
            )
        predicted = group["prediction"].to_numpy(dtype=np.float64)
        observed = group["observed_atr"].to_numpy(dtype=np.float64)
        axis.hexbin(
            predicted,
            observed,
            gridsize=45,
            mincnt=1,
            cmap="viridis",
        )
        intercept, slope = _calibration_line(observed, predicted)
        x_line = np.linspace(np.min(predicted), np.max(predicted), 100)
        if np.isfinite(intercept) and np.isfinite(slope):
            axis.plot(
                x_line,
                intercept + slope * x_line,
                color="crimson",
                linewidth=1.5,
                label=f"observed={intercept:.3f}+{slope:.3f}·pred",
            )
            axis.legend(loc="best", fontsize=8)
        axis.set_title(
            f"{selection['session']} · {selection['target']} · "
            f"{selection['model_family']}"
        )
        axis.set_xlabel("Outer-OOF prediction (ATR)")
        axis.set_ylabel("Observed label (ATR)")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Section 5 selected continuous models: Development outer-OOF calibration",
        fontsize=14,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _save_expansion_reliability_figure(
    result: Section5Result,
    path: Path,
) -> None:
    selected = result.expansion_metrics.loc[
        result.expansion_metrics["selected_expansion_family"]
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, (_, selection) in zip(
        axes,
        selected.sort_values("session").iterrows(),
        strict=True,
    ):
        table = result.expansion_reliability.loc[
            result.expansion_reliability["session"].eq(
                selection["session"]
            )
            & result.expansion_reliability["model_family"].eq(
                selection["model_family"]
            )
        ]
        axis.plot(
            table["mean_probability"],
            table["observed_event_rate"],
            marker="o",
            linewidth=1.5,
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set_title(
            f"{selection['session']} · {selection['model_family']}"
        )
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed nested expansion rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Section 5 expansion reliability: Development outer OOF",
        fontsize=14,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_section5_outputs(
    result: Section5Result,
    project_root: Path,
) -> dict[str, Any]:
    """Persist, serialize, hash, and checkpoint every Section 5 output."""

    root = Path(project_root)
    data_dir = (
        root / "data/processed/statistical_research/fes_project1/v1"
    )
    report_dir = (
        root / "reports/statistical_research/fes_project1/v1"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    parquet_tables: dict[str, pd.DataFrame] = {
        "section5_fold_table.parquet": result.fold_table,
        "section5_nested_threshold_audit.parquet": result.threshold_audit,
        "section5_continuous_oof_predictions.parquet": (
            result.continuous_oof_predictions
        ),
        "section5_continuous_paired_daily.parquet": (
            result.continuous_paired_daily
        ),
        "section5_outer_choices.parquet": result.outer_choices,
        "section5_random_subset_diagnostics.parquet": (
            result.random_subset_diagnostics
        ),
        "section5_profile_diagnostics.parquet": result.profile_diagnostics,
        "section5_expansion_oof_predictions.parquet": (
            result.expansion_oof_predictions
        ),
        "section5_complete_trial_ledger.parquet": (
            result.complete_trial_ledger
        ),
    }
    artifact_paths: list[Path] = []
    for filename, table in parquet_tables.items():
        path = data_dir / filename
        table.to_parquet(path, index=False)
        artifact_paths.append(path)

    csv_tables: dict[str, pd.DataFrame] = {
        "section5_fold_table.csv": result.fold_table,
        "section5_nested_threshold_audit.csv": result.threshold_audit,
        "section5_support_coverage.csv": result.support_coverage,
        "section5_continuous_model_comparison.csv": (
            result.continuous_model_comparison
        ),
        "section5_continuous_metrics.csv": result.continuous_metrics,
        "section5_continuous_deciles.csv": result.continuous_deciles,
        "section5_continuous_subperiods.csv": (
            result.continuous_subperiods
        ),
        "section5_outer_choices.csv": result.outer_choices,
        "section5_elastic_stability.csv": result.elastic_stability,
        "section5_elastic_cluster_stability.csv": (
            result.elastic_cluster_stability
        ),
        "section5_elastic_member_substitution.csv": (
            result.elastic_member_substitution
        ),
        "section5_random_subset_diagnostics.csv": (
            result.random_subset_diagnostics
        ),
        "section5_profile_diagnostics.csv": result.profile_diagnostics,
        "section5_expansion_metrics.csv": result.expansion_metrics,
        "section5_expansion_reliability.csv": (
            result.expansion_reliability
        ),
        "section5_frozen_thresholds.csv": result.frozen_thresholds,
        "section5_frozen_model_registry.csv": (
            result.frozen_model_registry
        ),
        "section5_access_audit.csv": result.access_audit,
        "section5_complete_trial_ledger.csv": (
            result.complete_trial_ledger
        ),
    }
    for filename, table in csv_tables.items():
        path = report_dir / filename
        table.to_csv(path, index=False, float_format="%.17g")
        artifact_paths.append(path)

    models_path = data_dir / "section5_frozen_models.joblib"
    joblib.dump(
        {
            "continuous": dict(result.frozen_continuous_models),
            "expansion": dict(result.frozen_expansion_models),
            "config": asdict(result.config),
            "environment": environment_versions(),
            "policy_sha256": result.policy_freeze[
                "pre_validation_policy_sha256"
            ],
        },
        models_path,
        compress=3,
    )
    artifact_paths.append(models_path)

    policy_path = data_dir / "section5_pre_validation_policy.json"
    policy_path.write_text(
        json.dumps(
            result.policy_freeze,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths.append(policy_path)

    config_path = data_dir / "section5_runtime_config.json"
    config_path.write_text(
        json.dumps(
            {
                "config": asdict(result.config),
                "environment": environment_versions(),
                "frozen_config_sha256": config_sha256(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths.append(config_path)

    continuous_figure = (
        report_dir / "section5_continuous_oof_calibration.png"
    )
    _save_continuous_diagnostic_figure(result, continuous_figure)
    artifact_paths.append(continuous_figure)
    expansion_figure = (
        report_dir / "section5_expansion_reliability.png"
    )
    _save_expansion_reliability_figure(result, expansion_figure)
    artifact_paths.append(expansion_figure)

    manifest_records: list[dict[str, Any]] = []
    for path in sorted(artifact_paths):
        relative = path.relative_to(root).as_posix()
        manifest_records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "artifact_class": (
                    "data"
                    if path.is_relative_to(data_dir)
                    else "report"
                ),
            }
        )
    manifest_payload = {
        "schema_version": "1.0.0",
        "section": 5,
        "artifacts": manifest_records,
    }
    manifest_path = report_dir / "section5_hash_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = sha256_file(manifest_path)
    locked_command = (
        ".\\.venv\\Scripts\\python.exe "
        "scripts\\run_fes_project1_research.py --section 6 "
        "--non-interactive "
        f"--section5-manifest-sha256 {manifest_sha256} "
        "--policy-sha256 "
        f"{result.policy_freeze['pre_validation_policy_sha256']}"
    )
    checkpoint = {
        "schema_version": "1.0.0",
        "section": 5,
        "status": "COMPLETED_STOP_BEFORE_SECTION_6",
        "development_only": True,
        "retrospective_validation_outcomes_read": False,
        "historical_final_outcomes_read": False,
        "gc_mgc_economics_read": False,
        "continuous_models_attempted": 54,
        "expansion_models_attempted": 12,
        "policy_freezes_attempted": 2,
        "outer_fold_count_per_session_target": 6,
        "outer_assessment_dates_per_fold": 63,
        "maximum_oof_dates_per_session_target": 378,
        "development_model_date_floor": (
            result.config.development_oof_dates_min
        ),
        "selected_continuous_models": (
            result.continuous_model_comparison.loc[
                result.continuous_model_comparison[
                    "selected_primary_family"
                ]
            ].to_dict(orient="records")
        ),
        "selected_expansion_models": (
            result.expansion_metrics.loc[
                result.expansion_metrics[
                    "selected_expansion_family"
                ]
            ].to_dict(orient="records")
        ),
        "frozen_thresholds": result.frozen_thresholds.to_dict(
            orient="records"
        ),
        "policy_status": result.policy_freeze["policy_status"],
        "no_policy_reason": result.policy_freeze["no_policy_reason"],
        "frozen_hashes": {
            "base_frozen_config_sha256": config_sha256(),
            "pre_validation_policy_sha256": result.policy_freeze[
                "pre_validation_policy_sha256"
            ],
            "frozen_models_file_sha256": sha256_file(models_path),
            "section5_manifest_sha256": manifest_sha256,
        },
        "exact_locked_validation_batch_command": locked_command,
        "tests_status": "PENDING_NOTEBOOK_AND_FULL_SUITE",
        "stop_boundary": (
            "Section 5 complete. Do not execute the locked Validation batch "
            "or inspect P&L before explicit Section 6 instruction."
        ),
    }
    checkpoint_path = report_dir / "section5_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            checkpoint,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": checkpoint_path,
        "policy_path": policy_path,
        "policy_sha256": result.policy_freeze[
            "pre_validation_policy_sha256"
        ],
        "models_path": models_path,
        "models_sha256": sha256_file(models_path),
        "locked_validation_command": locked_command,
        "artifact_count": len(manifest_records),
    }
