"""Locked retrospective Validation evaluation for Project 1 FES Section 6.

This module is the sole code path allowed to read retrospective Validation
outcomes for the study.  It verifies the complete Section 5 freeze before the
read, scores every frozen model without Validation refitting, finalizes the
predeclared feature/model gates, and records economics as not applicable when
the frozen Development gate permits no directional policy.

Historical Final is never projected by this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
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
from sklearn.preprocessing import StandardScaler

from .fes_project1_config import (
    D1_FEATURES,
    O1_FEATURES,
    canonical_json,
    config_sha256,
    sha256_file,
)
from .fes_project1_features import (
    BASE_FEATURE_NAMES,
    COMPARATOR_MAP,
    INTERACTION_FEATURE_NAMES,
)
from .fes_project1_modeling import (
    TARGET_AVAILABILITY_COLUMNS,
    TARGET_TICK_COLUMNS,
    _bootstrap_mean_ci,
    _daily_partial_rank_ic,
    _daily_spearman,
    _edge_hash,
    load_development_evidence_inputs,
)
from .fes_project1_nested import (
    ALL_SCALAR_INPUTS,
    DEGENERATE_IMPUTABLE,
    SESSIONS,
    TARGET_AVAILABILITY,
    TARGET_HORIZON,
    TARGET_TICK_COLUMN,
    FittedContinuousModel,
    FittedExpansionModel,
    Section5Config,
    _derive_final_continuous_configuration,
    _fit_predict_continuous_configuration,
    _prepare_horizon_session,
    daily_spearman,
    load_development_model_inputs,
)
from .fes_project1_profiles import P0_COLUMNS

VALIDATION_PARTITION: Final = "Validation"
DEVELOPMENT_PARTITION: Final = "Development"
POLICY_VARIANTS: Final = (
    "direction_only_p90",
    "direction_p90_and_expansion_p80",
)
COST_TICKS: Final[Mapping[str, float]] = {
    "frictionless": 0.0,
    "base": 2.6,
    "pessimistic": 4.6,
}
NO_POLICY_REASON_CODE: Final = (
    "NOT_APPLICABLE_NO_DIRECTIONAL_MODEL_PASSED_FROZEN_GATE"
)


@dataclass(frozen=True)
class Section6Config:
    """Frozen numerical contract for the one-time Validation batch."""

    random_seed: int = 20260727
    bootstrap_replicates: int = 2_000
    bootstrap_confidence: float = 0.95
    min_daily_observations: int = 10
    min_validation_dates: int = 150
    min_validation_observations: int = 4_000
    min_validation_ic_retention: float = 0.25
    min_partial_ic_retention: float = 0.25
    min_direction_spread_ticks: float = 2.0
    min_validation_interaction_delta: float = 0.02
    min_development_paired_delta: float = 0.01
    min_validation_paired_delta: float = 0.02
    min_development_oof_dates: int = 400
    max_best_10_positive_delta_share: float = 0.50
    stationary_bootstrap_replicates: int = 2_000
    stationary_bootstrap_restart_probability: float = 0.20
    sharpe_annualization: int = 252


@dataclass(frozen=True)
class LockedValidationAuthorization:
    """Capability created only after every Section 5 hash has passed."""

    section5_manifest_sha256: str
    policy_sha256: str
    models_sha256: str
    frozen_config_sha256: str
    verified_artifact_count: int


@dataclass(frozen=True)
class ValidationInputs:
    """Projected retrospective Validation inputs opened by the locked batch."""

    frame: pd.DataFrame
    raw_profiles: pd.DataFrame
    missing_reasons: pd.DataFrame
    access_audit: pd.DataFrame
    canonical_expansion_threshold: float


@dataclass
class LockedInteractionPair:
    """All-Development fixed comparator pair prepared before Validation opens."""

    trial_id: str
    session: str
    interaction_name: str
    base_columns: tuple[str, ...]
    plus_columns: tuple[str, ...]
    base_alpha: float
    plus_alpha: float
    imputation_medians: dict[str, float]
    base_scaler: StandardScaler
    plus_scaler: StandardScaler
    base_model: Ridge
    plus_model: Ridge
    fit_rows: int
    fit_dates: int
    fit_sha256: str


@dataclass
class Section6Result:
    """Complete result package from the single locked Validation batch."""

    validation_continuous_predictions: pd.DataFrame
    validation_continuous_metrics: pd.DataFrame
    validation_continuous_daily: pd.DataFrame
    validation_continuous_deciles: pd.DataFrame
    validation_continuous_subperiods: pd.DataFrame
    validation_expansion_predictions: pd.DataFrame
    validation_expansion_metrics: pd.DataFrame
    validation_expansion_reliability: pd.DataFrame
    feature_gate_table: pd.DataFrame
    feature_validation_daily_ic: pd.DataFrame
    validation_missingness: pd.DataFrame
    validation_integrity: pd.DataFrame
    interaction_gate_table: pd.DataFrame
    interaction_validation_daily_delta: pd.DataFrame
    model_gate_table: pd.DataFrame
    candidate_flags: pd.DataFrame
    candidate_counts: pd.DataFrame
    gc_economic_table: pd.DataFrame
    gc_drawdown_table: pd.DataFrame
    sharpe_table: pd.DataFrame
    gc_gate_verdict: pd.DataFrame
    comparator_registry: pd.DataFrame
    access_audit: pd.DataFrame
    complete_trial_ledger: pd.DataFrame
    authorization: LockedValidationAuthorization
    policy_freeze: dict[str, Any]
    config: Section6Config = field(default_factory=Section6Config)


def _section_paths(project_root: Path) -> tuple[Path, Path]:
    root = Path(project_root)
    return (
        root / "data/processed/statistical_research/fes_project1/v1",
        root / "reports/statistical_research/fes_project1/v1",
    )


def assert_section5_freeze(
    project_root: Path,
    *,
    expected_manifest_sha256: str,
    expected_policy_sha256: str,
) -> LockedValidationAuthorization:
    """Verify every Section 5 artifact and create the Validation capability."""

    root = Path(project_root)
    data_dir, report_dir = _section_paths(root)
    manifest_path = report_dir / "section5_hash_manifest.json"
    checkpoint_path = report_dir / "section5_checkpoint.json"
    policy_path = data_dir / "section5_pre_validation_policy.json"
    models_path = data_dir / "section5_frozen_models.joblib"
    completion_path = report_dir / "section6_batch_completion.json"
    if completion_path.exists():
        raise FileExistsError(
            "Section 6 batch completion already exists; the locked Validation "
            "read cannot be repeated."
        )
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise AssertionError(
            "Section 5 manifest hash mismatch; locked Validation batch refused."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("section") != 5:
        raise AssertionError("The supplied manifest is not a Section 5 manifest.")
    for artifact in manifest["artifacts"]:
        path = root / str(artifact["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Section 5 artifact is missing: {path}")
        if path.stat().st_size != int(artifact["bytes"]):
            raise AssertionError(f"Frozen artifact size mismatch: {path}")
        if sha256_file(path) != str(artifact["sha256"]):
            raise AssertionError(f"Frozen artifact hash mismatch: {path}")

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    frozen_hashes = checkpoint["frozen_hashes"]
    if frozen_hashes["section5_manifest_sha256"] != expected_manifest_sha256:
        raise AssertionError("Checkpoint and supplied Section 5 manifest disagree.")
    if frozen_hashes["pre_validation_policy_sha256"] != expected_policy_sha256:
        raise AssertionError("Checkpoint and supplied policy hash disagree.")
    if checkpoint["retrospective_validation_outcomes_read"]:
        raise AssertionError("Checkpoint says retrospective Validation was already read.")
    if checkpoint["historical_final_outcomes_read"]:
        raise AssertionError("Historical Final was not preserved as unread.")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    reported_policy_sha256 = str(policy["pre_validation_policy_sha256"])
    policy_payload = dict(policy)
    policy_payload.pop("pre_validation_policy_sha256")
    recomputed_policy_sha256 = hashlib.sha256(
        canonical_json(policy_payload).encode("utf-8")
    ).hexdigest()
    if reported_policy_sha256 != recomputed_policy_sha256:
        raise AssertionError("Frozen policy payload hash is internally inconsistent.")
    if reported_policy_sha256 != expected_policy_sha256:
        raise AssertionError("Frozen policy hash mismatch; Validation batch refused.")

    models_sha256 = sha256_file(models_path)
    if models_sha256 != frozen_hashes["frozen_models_file_sha256"]:
        raise AssertionError("Frozen model bundle hash mismatch.")
    observed_config_sha256 = config_sha256()
    if observed_config_sha256 != frozen_hashes["base_frozen_config_sha256"]:
        raise AssertionError("Frozen base configuration hash mismatch.")

    model_bundle = joblib.load(models_path)
    if model_bundle["policy_sha256"] != expected_policy_sha256:
        raise AssertionError("Serialized models reference a different policy hash.")
    frozen_model_contracts = {
        item["model_key"]: item for item in policy["continuous_models"]
    }
    frozen_model_contracts.update(
        {item["model_key"]: item for item in policy["expansion_models"]}
    )
    for role in ("continuous", "expansion"):
        for model_key, model in model_bundle[role].items():
            contract = frozen_model_contracts[model_key]
            configuration_sha256 = hashlib.sha256(
                canonical_json(model.configuration).encode("utf-8")
            ).hexdigest()
            if configuration_sha256 != contract["configuration_sha256"]:
                raise AssertionError(
                    f"Frozen model configuration mismatch: {model_key}"
                )
            if model.input_schema_sha256 != contract["input_schema_sha256"]:
                raise AssertionError(f"Frozen model schema mismatch: {model_key}")

    return LockedValidationAuthorization(
        section5_manifest_sha256=observed_manifest_sha256,
        policy_sha256=reported_policy_sha256,
        models_sha256=models_sha256,
        frozen_config_sha256=observed_config_sha256,
        verified_artifact_count=len(manifest["artifacts"]),
    )


def _read_validation_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    table = pq.read_table(
        path,
        columns=list(dict.fromkeys(columns)),
        filters=[("research_partition", "=", VALIDATION_PARTITION)],
    )
    frame = table.to_pandas(ignore_metadata=True)
    observed = set(frame["research_partition"].astype(str).unique())
    if observed != {VALIDATION_PARTITION}:
        raise PermissionError(
            f"{path.name} returned a prohibited partition: {sorted(observed)}"
        )
    return frame


def load_locked_validation_inputs(
    project_root: Path,
    authorization: LockedValidationAuthorization,
) -> ValidationInputs:
    """Open retrospective Validation once after receiving the hash capability."""

    if not isinstance(authorization, LockedValidationAuthorization):
        raise PermissionError(
            "Retrospective Validation requires a verified Section 5 capability."
        )
    root = Path(project_root)
    data_dir, _ = _section_paths(root)
    scalar_path = data_dir / "scalar_features_validation_sealed_gc.parquet"
    missing_path = data_dir / "scalar_missing_reasons_validation_sealed_gc.parquet"
    profile_index_path = data_dir / "profile_index_validation_sealed_gc.parquet"
    raw_profile_path = data_dir / "raw_profile_p0_validation_sealed_gc.parquet"
    feature_path = root / "data/processed/statistical_research/feature_matrix_gc.parquet"
    label_path = root / "data/processed/statistical_research/forward_labels_gc.parquet"
    expansion_threshold_path = (
        root
        / "data/processed/statistical_research/"
        "forward_label_expansion_thresholds_gc.parquet"
    )

    scalar = pd.read_parquet(scalar_path)
    profile_index = pd.read_parquet(profile_index_path)
    for name, frame in (("scalar", scalar), ("profile index", profile_index)):
        observed = set(frame["research_partition"].astype(str).unique())
        if observed != {VALIDATION_PARTITION}:
            raise PermissionError(
                f"Sealed {name} contains prohibited partitions: {sorted(observed)}"
            )
        if not frame["observation_id"].is_unique:
            raise ValueError(f"Sealed Validation {name} ids are not unique.")
    missing = pd.read_parquet(missing_path)
    if not missing["observation_id"].is_unique:
        raise ValueError("Sealed Validation missing-reason ids are not unique.")

    required_existing = tuple(
        dict.fromkeys(
            [
                *ALL_SCALAR_INPUTS,
                *D1_FEATURES,
                *O1_FEATURES,
                *[
                    comparator
                    for comparators in COMPARATOR_MAP.values()
                    for comparator in comparators
                ],
            ]
        )
    )
    scalar_columns = set(scalar.columns)
    existing_columns = [
        name
        for name in required_existing
        if name not in scalar_columns and not name.endswith("__degenerate")
    ]
    existing = _read_validation_table(
        feature_path,
        ["observation_id", "research_partition", *existing_columns],
    )
    label_columns = [
        "observation_id",
        "decision_timestamp_utc",
        "entry_timestamp_utc",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "product",
        "symbol",
        "active_symbol",
        "instrument_id",
        "continuous_segment_id",
        "entry_price",
        "decision_atr_20m",
        "atr_normalization_available",
        "label_available_30",
        "label_available_60",
        "exit_timestamp_utc_30",
        "exit_timestamp_utc_60",
        "forward_return_30_atr",
        "forward_return_30_ticks",
        "forward_return_60_atr",
        "forward_return_60_ticks",
        "future_range_60_atr",
        "future_range_60_ticks",
        "expansion_label_60",
    ]
    labels = _read_validation_table(label_path, label_columns)
    base_ids = set(scalar["observation_id"])
    for name, frame in (
        ("existing", existing),
        ("labels", labels),
        ("profile index", profile_index),
        ("missing reasons", missing),
    ):
        if not frame["observation_id"].is_unique:
            raise ValueError(f"Validation {name} ids are not unique.")
        if set(frame["observation_id"]) != base_ids:
            raise ValueError(f"Validation {name} does not cover the sealed ids.")

    indicators = missing.loc[
        :, ["observation_id", *[f"{name}__degenerate" for name in DEGENERATE_IMPUTABLE]]
    ].copy()
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
    if set(joined["research_partition"].astype(str).unique()) != {
        VALIDATION_PARTITION
    }:
        raise PermissionError("Joined Section 6 frame crossed the Validation boundary.")
    years = set(pd.to_datetime(joined["trade_date_ny"]).dt.year.unique())
    if years != {2024}:
        raise PermissionError(
            f"Retrospective Validation must be exactly 2024, observed {sorted(years)}."
        )

    raw_profiles = pd.read_parquet(raw_profile_path)
    if not raw_profiles["observation_id"].is_unique:
        raise ValueError("Validation raw-profile ids are not unique.")
    complete_profile_ids = set(
        joined.loc[joined["profile_complete_30"].astype(bool), "observation_id"]
    )
    if set(raw_profiles["observation_id"]) != complete_profile_ids:
        raise ValueError("Validation raw-profile ids differ from complete profile ids.")

    thresholds = pd.read_parquet(expansion_threshold_path)
    threshold_row = thresholds.loc[thresholds["horizon"].eq(60)]
    if len(threshold_row) != 1:
        raise AssertionError("The authoritative 60-minute expansion threshold is missing.")
    threshold_row = threshold_row.iloc[0]
    if threshold_row["fitting_partition"] != DEVELOPMENT_PARTITION:
        raise AssertionError("Expansion threshold was not fitted on Development.")
    canonical_expansion_threshold = float(threshold_row["threshold"])
    available_expansion = (
        joined["label_available_60"].astype(bool)
        & joined["atr_normalization_available"].astype(bool)
        & np.isfinite(joined["future_range_60_atr"].to_numpy(dtype=np.float64))
    )
    reconstructed = (
        joined.loc[available_expansion, "future_range_60_atr"].to_numpy(
            dtype=np.float64
        )
        >= canonical_expansion_threshold
    )
    saved = joined.loc[available_expansion, "expansion_label_60"].astype(bool).to_numpy()
    if not np.array_equal(reconstructed, saved):
        raise AssertionError(
            "Saved Validation expansion labels differ from the frozen "
            "Development-wide threshold."
        )

    access_audit = pd.DataFrame.from_records(
        [
            {
                "source": scalar_path.relative_to(root).as_posix(),
                "read_scope": "presealed outcome-free Validation scalar features",
                "rows_loaded": len(scalar),
                "partitions_loaded": VALIDATION_PARTITION,
                "outcomes_loaded": False,
            },
            {
                "source": raw_profile_path.relative_to(root).as_posix(),
                "read_scope": "presealed outcome-free Validation P0 profiles",
                "rows_loaded": len(raw_profiles),
                "partitions_loaded": VALIDATION_PARTITION,
                "outcomes_loaded": False,
            },
            {
                "source": feature_path.relative_to(root).as_posix(),
                "read_scope": "physical parquet filter Validation; projected inputs",
                "rows_loaded": len(existing),
                "partitions_loaded": VALIDATION_PARTITION,
                "outcomes_loaded": False,
            },
            {
                "source": label_path.relative_to(root).as_posix(),
                "read_scope": "single locked physical Validation outcome read",
                "rows_loaded": len(labels),
                "partitions_loaded": VALIDATION_PARTITION,
                "outcomes_loaded": True,
            },
            {
                "source": "Historical Final outcomes",
                "read_scope": "not opened; no Final projection exists",
                "rows_loaded": 0,
                "partitions_loaded": "",
                "outcomes_loaded": False,
            },
            {
                "source": "GC/MGC economic bars",
                "read_scope": "not opened because frozen policy is ineligible",
                "rows_loaded": 0,
                "partitions_loaded": "",
                "outcomes_loaded": False,
            },
        ]
    )
    return ValidationInputs(
        frame=joined,
        raw_profiles=raw_profiles,
        missing_reasons=missing,
        access_audit=access_audit,
        canonical_expansion_threshold=canonical_expansion_threshold,
    )


def _common_support(frame: pd.DataFrame, *, session: str, target: str) -> np.ndarray:
    mask = (
        frame["entry_session"].astype(str).eq(session).to_numpy()
        & frame[TARGET_AVAILABILITY[target]].astype(bool).to_numpy()
        & frame["atr_normalization_available"].astype(bool).to_numpy()
        & np.isfinite(frame[target].to_numpy(dtype=np.float64))
        & frame["profile_complete_30"].astype(bool).to_numpy()
    )
    for column in ALL_SCALAR_INPUTS:
        values = frame[column].to_numpy(dtype=np.float64)
        if column in DEGENERATE_IMPUTABLE:
            indicator = frame[f"{column}__degenerate"].astype(bool).to_numpy()
            mask &= np.isfinite(values) | indicator
        else:
            mask &= np.isfinite(values)
    return mask


def _profile_rows(
    raw_profiles: pd.DataFrame,
    observation_ids: np.ndarray,
) -> np.ndarray:
    indexed = raw_profiles.set_index("observation_id")
    if not set(observation_ids).issubset(set(indexed.index)):
        raise ValueError("A complete Validation profile id has no raw P0 row.")
    return indexed.loc[observation_ids, list(P0_COLUMNS)].to_numpy(dtype=np.float64)


def _locked_comparator_hash(model: FittedContinuousModel) -> str:
    payload: dict[str, Any] = {
        "session": model.session,
        "target": model.target,
        "family": model.model_family,
        "configuration": model.configuration,
        "input_schema_sha256": model.input_schema_sha256,
    }
    if model.ridge_model is not None:
        payload["intercept"] = float(model.ridge_model.intercept_)
        payload["coefficients"] = model.ridge_model.coef_.astype(float).tolist()
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def prepare_locked_anchor_comparators(
    project_root: Path,
    selected_models: Mapping[str, FittedContinuousModel],
) -> tuple[dict[str, FittedContinuousModel], pd.DataFrame]:
    """Fit fixed M1 comparators on all Development before Validation opens."""

    root = Path(project_root)
    data_dir, report_dir = _section_paths(root)
    inputs = load_development_model_inputs(root)
    choices = pd.read_csv(report_dir / "section5_outer_choices.csv")
    comparator_models: dict[str, FittedContinuousModel] = {}
    registry: list[dict[str, Any]] = []
    prepared: dict[tuple[str, int], tuple[pd.DataFrame, np.ndarray]] = {}
    for model_key, selected in sorted(selected_models.items()):
        session = selected.session
        target = selected.target
        horizon = TARGET_HORIZON[target]
        prepared_key = (session, horizon)
        if prepared_key not in prepared:
            prepared[prepared_key] = _prepare_horizon_session(
                inputs, session=session, horizon=horizon
            )
        frame, raw_profile = prepared[prepared_key]
        target_choices = choices.loc[
            choices["session"].eq(session)
            & choices["target"].eq(target)
            & choices["family"].eq("M1")
            & choices["status"].eq("COMPLETE")
        ].copy()
        configuration = _derive_final_continuous_configuration(
            target=target,
            family="M1",
            choices=target_choices,
            config=Section5Config(),
        )
        all_rows = np.ones(len(frame), dtype=bool)
        _, comparator, _ = _fit_predict_continuous_configuration(
            frame=frame,
            raw_profile=raw_profile,
            target=target,
            family="M1",
            configuration=configuration,
            train_mask=all_rows,
            assessment_mask=all_rows,
        )
        comparator_models[model_key] = comparator
        registry.append(
            {
                "comparator_key": model_key,
                "session": session,
                "target": target,
                "model_family": "M1",
                "configuration_json": canonical_json(configuration),
                "configuration_sha256": hashlib.sha256(
                    canonical_json(configuration).encode("utf-8")
                ).hexdigest(),
                "input_schema_sha256": comparator.input_schema_sha256,
                "fit_rows": len(frame),
                "fit_dates": int(
                    pd.to_datetime(frame["trade_date_ny"]).dt.normalize().nunique()
                ),
                "fit_partition": DEVELOPMENT_PARTITION,
                "prepared_before_validation_read": True,
                "comparator_fit_sha256": _locked_comparator_hash(comparator),
                "artifact_source": (
                    data_dir / "section5_outer_choices.parquet"
                ).relative_to(root).as_posix(),
            }
        )
    return comparator_models, pd.DataFrame.from_records(registry)


def _build_development_evidence_join(project_root: Path) -> pd.DataFrame:
    inputs = load_development_evidence_inputs(project_root)
    indicators = inputs.scalar_missing_reasons.loc[
        :,
        [
            "observation_id",
            *[f"{name}__degenerate" for name in DEGENERATE_IMPUTABLE],
        ],
    ]
    return (
        inputs.forward_labels.merge(
            inputs.scalar_features.drop(
                columns=[
                    "decision_timestamp_utc",
                    "trade_date_ny",
                    "entry_session",
                    "research_partition",
                ]
            ),
            on="observation_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            inputs.existing_features.drop(
                columns=["research_partition", "entry_session"]
            ),
            on="observation_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            indicators,
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
    )


def _mode_smallest(values: pd.Series) -> float:
    mode = values.dropna().astype(float).mode()
    if mode.empty:
        raise ValueError("A frozen interaction comparator has no selected alpha.")
    return float(mode.min())


def prepare_locked_interaction_comparators(
    project_root: Path,
) -> tuple[dict[str, LockedInteractionPair], pd.DataFrame]:
    """Prepare eight fixed all-Development interaction comparator pairs."""

    root = Path(project_root)
    data_dir, _ = _section_paths(root)
    joined = _build_development_evidence_join(root)
    fold_audit = pd.read_parquet(
        data_dir / "section4_interaction_fold_audit_development.parquet"
    )
    pairs: dict[str, LockedInteractionPair] = {}
    registry: list[dict[str, Any]] = []
    for trial_id, audit in fold_audit.groupby("trial_id", sort=True):
        first = audit.iloc[0]
        session = str(trial_id).split("_")[1].replace("new", "New", 1)
        session = "New York" if "new_york" in str(trial_id) else "London"
        interaction_name = str(first["plus_columns"]).split("|")[-1]
        base_columns = tuple(str(first["base_columns"]).split("|"))
        plus_columns = tuple(str(first["plus_columns"]).split("|"))
        base_alpha = _mode_smallest(audit["selected_base_alpha"])
        plus_alpha = _mode_smallest(audit["selected_plus_alpha"])
        session_rows = joined["entry_session"].astype(str).eq(session)
        available = joined["label_available_60"].astype(bool)
        work = joined.loc[session_rows & available].copy()
        target = work["forward_return_60_atr"].to_numpy(dtype=np.float64)
        imputation_medians: dict[str, float] = {}
        for column in set(base_columns).union(plus_columns):
            if column == "lagged_volume_return_spearman_30":
                median = float(work[column].median())
                work[column] = work[column].fillna(median)
                imputation_medians[column] = median
        required_columns = list(dict.fromkeys([*base_columns, *plus_columns]))
        matrix = work.loc[:, required_columns].to_numpy(dtype=np.float64)
        valid = np.isfinite(target) & np.isfinite(matrix).all(axis=1)
        work = work.loc[valid].copy()
        target = target[valid]
        base_values = work.loc[:, list(base_columns)].to_numpy(dtype=np.float64)
        plus_values = work.loc[:, list(plus_columns)].to_numpy(dtype=np.float64)
        base_scaler = StandardScaler().fit(base_values)
        plus_scaler = StandardScaler().fit(plus_values)
        base_model = Ridge(
            alpha=base_alpha, fit_intercept=True, solver="svd"
        ).fit(base_scaler.transform(base_values), target)
        plus_model = Ridge(
            alpha=plus_alpha, fit_intercept=True, solver="svd"
        ).fit(plus_scaler.transform(plus_values), target)
        fit_payload = {
            "trial_id": trial_id,
            "base_columns": base_columns,
            "plus_columns": plus_columns,
            "base_alpha": base_alpha,
            "plus_alpha": plus_alpha,
            "base_coefficients": base_model.coef_.astype(float).tolist(),
            "plus_coefficients": plus_model.coef_.astype(float).tolist(),
            "base_intercept": float(base_model.intercept_),
            "plus_intercept": float(plus_model.intercept_),
            "imputation_medians": imputation_medians,
        }
        fit_sha256 = hashlib.sha256(
            canonical_json(fit_payload).encode("utf-8")
        ).hexdigest()
        pair = LockedInteractionPair(
            trial_id=str(trial_id),
            session=session,
            interaction_name=interaction_name,
            base_columns=base_columns,
            plus_columns=plus_columns,
            base_alpha=base_alpha,
            plus_alpha=plus_alpha,
            imputation_medians=imputation_medians,
            base_scaler=base_scaler,
            plus_scaler=plus_scaler,
            base_model=base_model,
            plus_model=plus_model,
            fit_rows=len(work),
            fit_dates=int(pd.to_datetime(work["trade_date_ny"]).dt.normalize().nunique()),
            fit_sha256=fit_sha256,
        )
        pairs[str(trial_id)] = pair
        registry.append(
            {
                "comparator_key": trial_id,
                "session": session,
                "target": "forward_return_60_atr",
                "model_family": "INTERACTION_BASE_PLUS_PAIR",
                "configuration_json": canonical_json(
                    {
                        "base_columns": base_columns,
                        "plus_columns": plus_columns,
                        "base_alpha": base_alpha,
                        "plus_alpha": plus_alpha,
                    }
                ),
                "configuration_sha256": hashlib.sha256(
                    canonical_json(
                        {
                            "base_columns": base_columns,
                            "plus_columns": plus_columns,
                            "base_alpha": base_alpha,
                            "plus_alpha": plus_alpha,
                        }
                    ).encode("utf-8")
                ).hexdigest(),
                "input_schema_sha256": hashlib.sha256(
                    "|".join(plus_columns).encode("utf-8")
                ).hexdigest(),
                "fit_rows": len(work),
                "fit_dates": pair.fit_dates,
                "fit_partition": DEVELOPMENT_PARTITION,
                "prepared_before_validation_read": True,
                "comparator_fit_sha256": fit_sha256,
                "artifact_source": (
                    data_dir
                    / "section4_interaction_fold_audit_development.parquet"
                ).relative_to(root).as_posix(),
            }
        )
    if len(pairs) != 8:
        raise AssertionError("Exactly eight frozen interaction pairs are required.")
    return pairs, pd.DataFrame.from_records(registry)


def score_frozen_continuous_models(
    inputs: ValidationInputs,
    selected_models: Mapping[str, FittedContinuousModel],
    anchor_models: Mapping[str, FittedContinuousModel],
) -> pd.DataFrame:
    """Apply selected and fixed-anchor all-Development objects to Validation."""

    records: list[pd.DataFrame] = []
    for model_key, selected in sorted(selected_models.items()):
        anchor = anchor_models[model_key]
        mask = _common_support(
            inputs.frame, session=selected.session, target=selected.target
        )
        frame = inputs.frame.loc[mask].copy()
        observation_ids = frame["observation_id"].to_numpy(dtype=np.int64)
        raw_profile = _profile_rows(inputs.raw_profiles, observation_ids)
        selected_prediction = selected.predict(frame, raw_profile)
        anchor_prediction = anchor.predict(frame, raw_profile)
        if selected.model_family == "M1":
            np.testing.assert_allclose(
                selected_prediction,
                anchor_prediction,
                rtol=1.0e-11,
                atol=1.0e-11,
            )
        output = frame.loc[
            :,
            [
                "observation_id",
                "decision_timestamp_utc",
                "entry_timestamp_utc",
                "trade_date_ny",
                "entry_session",
                "research_partition",
                "decision_atr_20m",
            ],
        ].copy()
        output["session"] = selected.session
        output["target"] = selected.target
        output["model_family"] = selected.model_family
        output["anchor_family"] = "M1"
        output["observed_atr"] = frame[selected.target].to_numpy(dtype=np.float64)
        output["observed_ticks"] = frame[
            TARGET_TICK_COLUMN[selected.target]
        ].to_numpy(dtype=np.float64)
        output["prediction"] = selected_prediction
        output["anchor_prediction"] = anchor_prediction
        records.append(output)
    result = pd.concat(records, ignore_index=True)
    if set(result["research_partition"].astype(str).unique()) != {
        VALIDATION_PARTITION
    }:
        raise PermissionError("Frozen model scoring escaped Validation.")
    return result


def _date_bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    config: Section6Config,
) -> tuple[float, float]:
    return _bootstrap_mean_ci(
        np.asarray(values, dtype=np.float64),
        seed=seed,
        replicates=config.bootstrap_replicates,
        confidence=config.bootstrap_confidence,
    )


def _concordance_correlation(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(observed) < 2:
        return np.nan
    covariance = float(
        np.mean((observed - observed.mean()) * (predicted - predicted.mean()))
    )
    denominator = (
        float(np.var(observed, ddof=0))
        + float(np.var(predicted, ddof=0))
        + float((observed.mean() - predicted.mean()) ** 2)
    )
    return float(2.0 * covariance / denominator) if denominator > 0.0 else np.nan


def _linear_calibration(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 2 or np.ptp(predicted[valid]) == 0.0:
        return np.nan, np.nan
    design = np.column_stack(
        [np.ones(int(valid.sum()), dtype=np.float64), predicted[valid]]
    )
    coefficient, *_ = np.linalg.lstsq(design, observed[valid], rcond=None)
    return float(coefficient[0]), float(coefficient[1])


def build_validation_continuous_evidence(
    predictions: pd.DataFrame,
    *,
    config: Section6Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the complete locked Validation continuous-model evidence."""

    metric_records: list[dict[str, Any]] = []
    daily_records: list[pd.DataFrame] = []
    decile_records: list[dict[str, Any]] = []
    subperiod_records: list[dict[str, Any]] = []
    keys = sorted(
        predictions[["session", "target", "model_family"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for key_index, (session, target, model_family) in enumerate(keys):
        group = predictions.loc[
            predictions["session"].eq(session)
            & predictions["target"].eq(target)
            & predictions["model_family"].eq(model_family)
        ].copy()
        selected_daily = daily_spearman(
            group["trade_date_ny"].to_numpy(),
            group["prediction"].to_numpy(dtype=np.float64),
            group["observed_atr"].to_numpy(dtype=np.float64),
            min_observations=config.min_daily_observations,
        ).rename(
            columns={
                "daily_ic": "model_daily_ic",
                "observations": "model_observations",
            }
        )
        anchor_daily = daily_spearman(
            group["trade_date_ny"].to_numpy(),
            group["anchor_prediction"].to_numpy(dtype=np.float64),
            group["observed_atr"].to_numpy(dtype=np.float64),
            min_observations=config.min_daily_observations,
        ).rename(
            columns={
                "daily_ic": "anchor_daily_ic",
                "observations": "anchor_observations",
            }
        )
        daily = selected_daily.merge(
            anchor_daily,
            on="trade_date_ny",
            how="inner",
            validate="one_to_one",
        )
        daily["paired_daily_ic_delta"] = (
            daily["model_daily_ic"] - daily["anchor_daily_ic"]
        )
        daily.insert(0, "model_family", model_family)
        daily.insert(0, "target", target)
        daily.insert(0, "session", session)
        daily_records.append(daily)
        model_ci = _date_bootstrap_ci(
            daily["model_daily_ic"],
            seed=config.random_seed + 10_000 + key_index,
            config=config,
        )
        delta_ci = _date_bootstrap_ci(
            daily["paired_daily_ic_delta"],
            seed=config.random_seed + 20_000 + key_index,
            config=config,
        )
        observed = group["observed_atr"].to_numpy(dtype=np.float64)
        predicted = group["prediction"].to_numpy(dtype=np.float64)
        predicted_ticks = (
            predicted
            * group["decision_atr_20m"].to_numpy(dtype=np.float64)
            / 0.10
        )
        observed_ticks = group["observed_ticks"].to_numpy(dtype=np.float64)
        intercept, slope = _linear_calibration(observed, predicted)
        metric: dict[str, Any] = {
            "session": session,
            "target": target,
            "model_family": model_family,
            "partition": "Retrospective Validation",
            "rows": len(group),
            "valid_dates": int(daily["model_daily_ic"].notna().sum()),
            "daily_ic_mean": float(daily["model_daily_ic"].mean()),
            "daily_ic_ci_low": model_ci[0],
            "daily_ic_ci_high": model_ci[1],
            "anchor_daily_ic_mean": float(daily["anchor_daily_ic"].mean()),
            "paired_daily_ic_delta_mean": float(
                daily["paired_daily_ic_delta"].mean()
            ),
            "paired_delta_ci_low": delta_ci[0],
            "paired_delta_ci_high": delta_ci[1],
            "mae_atr": float(mean_absolute_error(observed, predicted)),
            "rmse_atr": float(np.sqrt(mean_squared_error(observed, predicted))),
            "mae_ticks": float(mean_absolute_error(observed_ticks, predicted_ticks)),
            "concordance_correlation": _concordance_correlation(
                observed, predicted
            ),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }
        if str(target).startswith("forward_return"):
            nonzero = observed != 0.0
            truth = observed[nonzero] > 0.0
            estimate = predicted[nonzero] > 0.0
            metric["sign_accuracy_nonzero"] = (
                float(np.mean(truth == estimate)) if nonzero.any() else np.nan
            )
            if truth.size and np.unique(truth).size == 2:
                metric["balanced_accuracy_nonzero"] = float(
                    balanced_accuracy_score(truth, estimate)
                )
                tn, fp, fn, tp = confusion_matrix(
                    truth, estimate, labels=[False, True]
                ).ravel()
                metric.update(
                    {
                        "short_correct_tn": int(tn),
                        "false_long_fp": int(fp),
                        "false_short_fn": int(fn),
                        "long_correct_tp": int(tp),
                    }
                )
            else:
                metric["balanced_accuracy_nonzero"] = np.nan
        positive = daily.loc[
            daily["paired_daily_ic_delta"].gt(0.0),
            "paired_daily_ic_delta",
        ].sort_values(ascending=False)
        metric["best_10_positive_delta_share"] = (
            float(positive.head(10).sum() / positive.sum())
            if positive.sum() > 0.0
            else np.nan
        )

        ranked = group["prediction"].rank(method="first")
        group["prediction_decile"] = (
            pd.qcut(ranked, q=10, labels=False, duplicates="drop") + 1
        )
        for decile, decile_group in group.groupby(
            "prediction_decile", sort=True
        ):
            decile_records.append(
                {
                    "session": session,
                    "target": target,
                    "model_family": model_family,
                    "partition": "Retrospective Validation",
                    "prediction_decile": int(decile),
                    "rows": len(decile_group),
                    "prediction_mean_atr": float(decile_group["prediction"].mean()),
                    "observed_mean_atr": float(
                        decile_group["observed_atr"].mean()
                    ),
                    "observed_mean_ticks": float(
                        decile_group["observed_ticks"].mean()
                    ),
                }
            )
        daily["quarter"] = (
            pd.to_datetime(daily["trade_date_ny"])
            .dt.to_period("Q")
            .astype(str)
        )
        quarter_rows: list[dict[str, Any]] = []
        for quarter, quarter_group in daily.groupby("quarter", sort=True):
            quarter_record = {
                "session": session,
                "target": target,
                "model_family": model_family,
                "partition": "Retrospective Validation",
                "period_type": "quarter",
                "period": quarter,
                "valid_dates": int(quarter_group["model_daily_ic"].notna().sum()),
                "daily_ic_mean": float(quarter_group["model_daily_ic"].mean()),
                "paired_delta_mean": float(
                    quarter_group["paired_daily_ic_delta"].mean()
                ),
                "paired_delta_sum": float(
                    quarter_group["paired_daily_ic_delta"].sum()
                ),
            }
            quarter_rows.append(quarter_record)
            subperiod_records.append(quarter_record)
        metric["worst_subperiod"] = min(
            quarter_rows, key=lambda row: row["daily_ic_mean"]
        )["period"]
        metric["worst_subperiod_daily_ic"] = min(
            row["daily_ic_mean"] for row in quarter_rows
        )
        metric_records.append(metric)
    return (
        pd.DataFrame.from_records(metric_records),
        pd.concat(daily_records, ignore_index=True),
        pd.DataFrame.from_records(decile_records),
        pd.DataFrame.from_records(subperiod_records),
    )


def score_frozen_expansion_models(
    inputs: ValidationInputs,
    models: Mapping[str, FittedExpansionModel],
) -> pd.DataFrame:
    """Score frozen expansion objects against the authoritative saved label."""

    records: list[pd.DataFrame] = []
    for model_key, model in sorted(models.items()):
        mask = _common_support(
            inputs.frame,
            session=model.session,
            target="future_range_60_atr",
        )
        frame = inputs.frame.loc[mask].copy()
        observation_ids = frame["observation_id"].to_numpy(dtype=np.int64)
        raw_profile = _profile_rows(inputs.raw_profiles, observation_ids)
        probability = model.predict_proba(frame, raw_profile)
        output = frame.loc[
            :,
            [
                "observation_id",
                "decision_timestamp_utc",
                "entry_timestamp_utc",
                "trade_date_ny",
                "entry_session",
                "research_partition",
                "future_range_60_atr",
                "expansion_label_60",
            ],
        ].copy()
        output["session"] = model.session
        output["model_family"] = model.model_family
        output["probability"] = probability
        output["authoritative_range_threshold_atr"] = (
            inputs.canonical_expansion_threshold
        )
        output["model_training_range_threshold_atr"] = (
            model.expansion_range_threshold
        )
        output["threshold_contract_match"] = np.isclose(
            model.expansion_range_threshold,
            inputs.canonical_expansion_threshold,
            rtol=0.0,
            atol=1.0e-12,
        )
        output["model_key"] = model_key
        records.append(output)
    return pd.concat(records, ignore_index=True)


def _classification_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1.0e-8, 1.0 - 1.0e-8
    )
    if np.unique(labels).size < 2 or np.ptp(probabilities) == 0.0:
        return np.nan, np.nan
    logit_values = np.log(
        probabilities / (1.0 - probabilities)
    ).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=1.0e12,
        l1_ratio=0.0,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2_000,
        tol=1.0e-8,
    ).fit(logit_values, labels)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _equal_date_log_loss(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date_ny", sort=True):
        records.append(
            {
                "trade_date_ny": pd.Timestamp(trade_date),
                "daily_log_loss": float(
                    log_loss(
                        group["expansion_label_60"].astype(int),
                        np.clip(
                            group["probability"].to_numpy(dtype=np.float64),
                            1.0e-12,
                            1.0 - 1.0e-12,
                        ),
                        labels=[0, 1],
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_validation_expansion_evidence(
    predictions: pd.DataFrame,
    development_oof: pd.DataFrame,
    frozen_thresholds: pd.DataFrame,
    *,
    config: Section6Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate locked Validation classifier and reliability evidence."""

    metric_records: list[dict[str, Any]] = []
    reliability_records: list[dict[str, Any]] = []
    for session_index, (session, group) in enumerate(
        predictions.groupby("session", sort=True)
    ):
        labels = group["expansion_label_60"].astype(int).to_numpy()
        probability = group["probability"].to_numpy(dtype=np.float64)
        development_session = development_oof.loc[
            development_oof["session"].eq(session)
            & development_oof["model_family"].eq("M1_O1")
        ]
        development_prevalence = float(
            development_session["nested_expansion_label"].mean()
        )
        prevalence_probability = np.full(len(group), development_prevalence)
        daily_loss = _equal_date_log_loss(group)
        prevalence_frame = group.assign(probability=prevalence_probability)
        prevalence_loss = _equal_date_log_loss(prevalence_frame)
        paired_loss = daily_loss.assign(
            anchor_daily_log_loss=daily_loss["daily_log_loss"].to_numpy()
        )
        paired_loss["paired_log_loss_delta_vs_o1"] = 0.0
        delta_ci = _date_bootstrap_ci(
            paired_loss["paired_log_loss_delta_vs_o1"],
            seed=config.random_seed + 40_000 + session_index,
            config=config,
        )
        intercept, slope = _classification_calibration(labels, probability)
        threshold = float(
            frozen_thresholds.loc[
                frozen_thresholds["session"].eq(session)
                & frozen_thresholds["threshold_role"].eq(
                    "expansion_probability_p80"
                ),
                "value",
            ].iloc[0]
        )
        flagged = probability >= threshold
        true_positive = labels.astype(bool)
        precision = (
            float(np.mean(true_positive[flagged])) if flagged.any() else np.nan
        )
        recall = (
            float(np.sum(true_positive & flagged) / np.sum(true_positive))
            if true_positive.any()
            else np.nan
        )
        prevalence = float(np.mean(labels))
        threshold_contract_match = bool(group["threshold_contract_match"].all())
        advances = bool(
            threshold_contract_match
            and delta_ci[1] < 0.0
            and brier_score_loss(labels, probability)
            < brier_score_loss(labels, probability)
            and 0.75 <= slope <= 1.25
            and abs(intercept) <= 0.10
        )
        metric_records.append(
            {
                "session": session,
                "model_family": str(group["model_family"].iloc[0]),
                "partition": "Retrospective Validation",
                "rows": len(group),
                "valid_dates": int(
                    pd.to_datetime(group["trade_date_ny"]).dt.normalize().nunique()
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
                "brier_score": float(brier_score_loss(labels, probability)),
                "prevalence_baseline_brier": float(
                    brier_score_loss(labels, prevalence_probability)
                ),
                "o1_baseline_brier": float(
                    brier_score_loss(labels, probability)
                ),
                "equal_date_log_loss": float(daily_loss["daily_log_loss"].mean()),
                "prevalence_baseline_equal_date_log_loss": float(
                    prevalence_loss["daily_log_loss"].mean()
                ),
                "o1_baseline_equal_date_log_loss": float(
                    daily_loss["daily_log_loss"].mean()
                ),
                "paired_log_loss_delta_vs_o1": 0.0,
                "paired_log_loss_delta_ci_low": delta_ci[0],
                "paired_log_loss_delta_ci_high": delta_ci[1],
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "development_probability_p80": threshold,
                "p80_flagged_rows": int(flagged.sum()),
                "p80_precision": precision,
                "p80_recall": recall,
                "p80_lift": (
                    float(precision / prevalence)
                    if np.isfinite(precision) and prevalence > 0.0
                    else np.nan
                ),
                "authoritative_range_threshold_atr": float(
                    group["authoritative_range_threshold_atr"].iloc[0]
                ),
                "model_training_range_threshold_atr": float(
                    group["model_training_range_threshold_atr"].iloc[0]
                ),
                "threshold_contract_match": threshold_contract_match,
                "classifier_advances": advances,
                "classifier_verdict": (
                    "CLASSIFIER_ADVANCES"
                    if advances
                    else "CLASSIFIER_INELIGIBLE_OR_NO_INCREMENTAL_VALUE"
                ),
            }
        )
        ranked = group["probability"].rank(method="first")
        group = group.copy()
        group["probability_decile"] = (
            pd.qcut(ranked, q=10, labels=False, duplicates="drop") + 1
        )
        for decile, decile_group in group.groupby(
            "probability_decile", sort=True
        ):
            reliability_records.append(
                {
                    "session": session,
                    "model_family": str(group["model_family"].iloc[0]),
                    "partition": "Retrospective Validation",
                    "probability_decile": int(decile),
                    "rows": len(decile_group),
                    "mean_probability": float(decile_group["probability"].mean()),
                    "observed_event_rate": float(
                        decile_group["expansion_label_60"].astype(float).mean()
                    ),
                }
            )
    return (
        pd.DataFrame.from_records(metric_records),
        pd.DataFrame.from_records(reliability_records),
    )


def _validation_bucket_summary(
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    tick_target: str,
    edges: np.ndarray,
) -> tuple[pd.DataFrame, float, float]:
    feature_values = frame[feature].to_numpy(dtype=np.float64)
    target_values = frame[target].to_numpy(dtype=np.float64)
    tick_values = frame[tick_target].to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(feature_values)
        & np.isfinite(target_values)
        & np.isfinite(tick_values)
    )
    buckets = np.full(len(frame), -1, dtype=np.int64)
    buckets[valid] = np.searchsorted(edges, feature_values[valid], side="right")
    records: list[dict[str, Any]] = []
    for bucket in range(len(edges) + 1):
        rows = buckets == bucket
        if not rows.any():
            continue
        records.append(
            {
                "bucket_index": bucket,
                "rows": int(rows.sum()),
                "dates": int(frame.loc[rows, "trade_date_ny"].nunique()),
                "mean_target_atr": float(np.mean(target_values[rows])),
                "mean_target_ticks": float(np.mean(tick_values[rows])),
            }
        )
    table = pd.DataFrame.from_records(records)
    monotonicity = (
        float(
            scipy_stats.spearmanr(
                table["bucket_index"].to_numpy(dtype=np.float64),
                table["mean_target_atr"].to_numpy(dtype=np.float64),
            ).statistic
        )
        if len(table) >= 3
        else np.nan
    )
    spread = (
        float(
            table.iloc[-1]["mean_target_ticks"]
            - table.iloc[0]["mean_target_ticks"]
        )
        if len(table) >= 2
        else np.nan
    )
    return table, monotonicity, spread


def build_validation_feature_gates(
    project_root: Path,
    inputs: ValidationInputs,
    *,
    config: Section6Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply frozen quintiles and finalize all 20 base-feature gates."""

    _, report_dir = _section_paths(project_root)
    development = pd.read_csv(
        report_dir / "section4_base_feature_evidence_development.csv"
    )
    edges_table = pd.read_parquet(
        Path(project_root)
        / "data/processed/statistical_research/fes_project1/v1/"
        "section4_quintile_edges_development.parquet"
    )
    integrity = pd.read_csv(
        report_dir / "section4_integrity_audit_development.csv"
    ).set_index("feature_name")
    gate_records: list[dict[str, Any]] = []
    daily_records: list[pd.DataFrame] = []
    for index, row in enumerate(development.itertuples(index=False)):
        feature = str(row.feature_name)
        session = str(row.session)
        target = str(row.target)
        availability = TARGET_AVAILABILITY_COLUMNS[target]
        sub = inputs.frame.loc[
            inputs.frame["entry_session"].astype(str).eq(session)
            & inputs.frame[availability].astype(bool)
        ].copy()
        daily = _daily_spearman(
            sub,
            feature,
            target,
            min_observations=config.min_daily_observations,
        )
        daily.insert(0, "trial_id", row.trial_id)
        daily.insert(1, "session", session)
        daily.insert(2, "feature_name", feature)
        daily.insert(3, "target", target)
        daily_records.append(daily)
        raw_values = daily["daily_ic"].to_numpy(dtype=np.float64)
        validation_ic = (
            float(np.nanmean(raw_values)) if np.isfinite(raw_values).any() else np.nan
        )
        validation_ci = _date_bootstrap_ci(
            raw_values,
            seed=config.random_seed + index,
            config=config,
        )
        edge_rows = edges_table.loc[
            edges_table["feature_name"].eq(feature)
            & edges_table["session"].eq(session)
        ].sort_values("edge_index")
        frozen_edges = edge_rows["edge_value"].to_numpy(dtype=np.float64)
        expected_edge_hash = str(edge_rows["edge_set_sha256"].iloc[0])
        if _edge_hash(feature, session, frozen_edges) != expected_edge_hash:
            raise AssertionError(f"Frozen quintile edge hash mismatch: {row.trial_id}")
        _, validation_monotonicity, validation_spread = (
            _validation_bucket_summary(
                sub,
                feature=feature,
                target=target,
                tick_target=TARGET_TICK_COLUMNS[target],
                edges=frozen_edges,
            )
        )
        comparators = COMPARATOR_MAP[feature]
        partial_daily = _daily_partial_rank_ic(
            sub,
            feature,
            target,
            comparators,
            min_observations=config.min_daily_observations,
        )
        partial_values = partial_daily["daily_partial_ic"].to_numpy(
            dtype=np.float64
        )
        partial_mean = (
            float(np.nanmean(partial_values))
            if np.isfinite(partial_values).any()
            else np.nan
        )
        partial_ci = _date_bootstrap_ci(
            partial_values,
            seed=config.random_seed + 30_000 + index,
            config=config,
        )
        raw_development_ic = float(row.daily_ic_mean)
        validation_ic_retention = (
            abs(validation_ic) / abs(raw_development_ic)
            if np.isfinite(validation_ic)
            and np.isfinite(raw_development_ic)
            and raw_development_ic != 0.0
            else np.nan
        )
        validation_ic_sign_match = bool(
            np.isfinite(validation_ic)
            and np.sign(validation_ic) == np.sign(raw_development_ic)
        )
        partial_retention = (
            abs(partial_mean) / abs(validation_ic)
            if np.isfinite(partial_mean)
            and np.isfinite(validation_ic)
            and validation_ic != 0.0
            else np.nan
        )
        partial_sign_match = bool(
            np.isfinite(partial_mean)
            and np.isfinite(validation_ic)
            and np.sign(partial_mean) == np.sign(validation_ic)
        )
        integrity_row = integrity.loc[feature]
        integrity_pass = bool(
            not bool(integrity_row["constant"])
            and not bool(integrity_row["near_constant_top_value_rate_ge_0_999"])
            and str(integrity_row["prefix_invariance_test"]) == "PASS"
            and str(integrity_row["continuity_reset_test"]) == "PASS"
        )
        gates = {
            "gate_integrity": integrity_pass,
            "gate_development_support": str(row.development_label)
            == "DEV_SUPPORT",
            "gate_validation_dates": len(daily)
            >= config.min_validation_dates,
            "gate_validation_observations": int(
                daily["observation_count"].sum()
            )
            >= config.min_validation_observations,
            "gate_validation_ic_sign_retention": bool(
                validation_ic_sign_match
                and np.isfinite(validation_ic_retention)
                and validation_ic_retention
                >= config.min_validation_ic_retention
            ),
            "gate_validation_partial_overlap": bool(
                partial_sign_match
                and np.isfinite(partial_retention)
                and partial_retention >= config.min_partial_ic_retention
            ),
            "gate_validation_direction_spread": bool(
                target != "forward_return_60_atr"
                or (
                    np.isfinite(validation_spread)
                    and abs(validation_spread)
                    >= config.min_direction_spread_ticks
                )
            ),
        }
        failed = [name for name, passed in gates.items() if not passed]
        gate_records.append(
            {
                "trial_id": row.trial_id,
                "session": session,
                "feature_name": feature,
                "target": target,
                "development_label": row.development_label,
                "development_daily_ic_mean": raw_development_ic,
                "development_bh_q_value": float(row.bh_q_value),
                "development_bucket_monotonicity": float(
                    row.bucket_monotonicity
                ),
                "development_top_bottom_spread_ticks": float(
                    row.top_bottom_spread_ticks
                ),
                "validation_observation_count": int(
                    daily["observation_count"].sum()
                ),
                "validation_trading_date_count": len(daily),
                "validation_daily_ic_mean": validation_ic,
                "validation_daily_ic_ci_low": validation_ci[0],
                "validation_daily_ic_ci_high": validation_ci[1],
                "validation_ic_sign_match": validation_ic_sign_match,
                "validation_absolute_ic_retention": validation_ic_retention,
                "validation_bucket_monotonicity": validation_monotonicity,
                "validation_top_bottom_spread_ticks": validation_spread,
                "validation_partial_trading_date_count": len(partial_daily),
                "validation_partial_ic_mean": partial_mean,
                "validation_partial_ic_ci_low": partial_ci[0],
                "validation_partial_ic_ci_high": partial_ci[1],
                "validation_partial_sign_match": partial_sign_match,
                "validation_partial_absolute_retention": partial_retention,
                "frozen_edge_set_sha256": expected_edge_hash,
                **gates,
                "failed_gates": "|".join(failed),
                "final_feature_label": (
                    "FEATURE_ADVANCES"
                    if not failed
                    else "FEATURE_DOES_NOT_ADVANCE"
                ),
            }
        )
    if len(gate_records) != 20:
        raise AssertionError("The final base-feature gate family must contain 20 rows.")
    return (
        pd.DataFrame.from_records(gate_records),
        pd.concat(daily_records, ignore_index=True),
    )


def build_validation_missingness(
    inputs: ValidationInputs,
) -> pd.DataFrame:
    """Report sealed Validation missingness causes by feature and session."""

    merged = inputs.missing_reasons.merge(
        inputs.frame[["observation_id", "entry_session"]],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    records: list[dict[str, Any]] = []
    for feature in [*BASE_FEATURE_NAMES, *INTERACTION_FEATURE_NAMES]:
        column = f"{feature}__missing_reason"
        for (session, reason), group in merged.groupby(
            ["entry_session", column], observed=True, dropna=False, sort=True
        ):
            records.append(
                {
                    "partition": "Retrospective Validation",
                    "session": str(session),
                    "feature_name": feature,
                    "missing_reason": str(reason),
                    "rows": len(group),
                }
            )
    return pd.DataFrame.from_records(records)


def build_validation_integrity(inputs: ValidationInputs) -> pd.DataFrame:
    """Calculate finite/constant/near-constant Validation diagnostics."""

    records: list[dict[str, Any]] = []
    for feature in [*BASE_FEATURE_NAMES, *INTERACTION_FEATURE_NAMES]:
        values = inputs.frame[feature].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        for session in SESSIONS:
            session_rows = inputs.frame["entry_session"].astype(str).eq(session)
            session_values = inputs.frame.loc[session_rows, feature].to_numpy(
                dtype=np.float64
            )
            session_finite = session_values[np.isfinite(session_values)]
            counts = pd.Series(session_finite).value_counts(dropna=False)
            top_rate = (
                float(counts.iloc[0] / len(session_finite))
                if len(session_finite)
                else np.nan
            )
            records.append(
                {
                    "partition": "Retrospective Validation",
                    "session": session,
                    "feature_name": feature,
                    "rows": int(session_rows.sum()),
                    "complete_rows": len(session_finite),
                    "complete_dates": int(
                        pd.to_datetime(
                            inputs.frame.loc[
                                session_rows & np.isfinite(values),
                                "trade_date_ny",
                            ]
                        )
                        .dt.normalize()
                        .nunique()
                    ),
                    "nonfinite_rows": int(
                        len(session_values) - len(session_finite)
                    ),
                    "unique_finite_values": int(np.unique(session_finite).size),
                    "constant": bool(np.unique(session_finite).size <= 1),
                    "top_value_rate": top_rate,
                    "near_constant_top_value_rate_ge_0_999": bool(
                        np.isfinite(top_rate) and top_rate >= 0.999
                    ),
                    "overall_finite_values": len(finite),
                }
            )
    return pd.DataFrame.from_records(records)


def score_validation_interactions(
    project_root: Path,
    inputs: ValidationInputs,
    pairs: Mapping[str, LockedInteractionPair],
    *,
    config: Section6Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply eight fixed Development comparator pairs to Validation."""

    _, report_dir = _section_paths(project_root)
    development = pd.read_csv(
        report_dir / "section4_interaction_incremental_development.csv"
    ).set_index("trial_id")
    gate_records: list[dict[str, Any]] = []
    daily_records: list[pd.DataFrame] = []
    for index, (trial_id, pair) in enumerate(sorted(pairs.items())):
        sub = inputs.frame.loc[
            inputs.frame["entry_session"].astype(str).eq(pair.session)
            & inputs.frame["label_available_60"].astype(bool)
        ].copy()
        for column, median in pair.imputation_medians.items():
            sub[column] = sub[column].fillna(median)
        required = list(dict.fromkeys([*pair.base_columns, *pair.plus_columns]))
        target = sub["forward_return_60_atr"].to_numpy(dtype=np.float64)
        matrix = sub.loc[:, required].to_numpy(dtype=np.float64)
        valid = np.isfinite(target) & np.isfinite(matrix).all(axis=1)
        sub = sub.loc[valid].copy()
        base_values = sub.loc[:, list(pair.base_columns)].to_numpy(
            dtype=np.float64
        )
        plus_values = sub.loc[:, list(pair.plus_columns)].to_numpy(
            dtype=np.float64
        )
        base_prediction = pair.base_model.predict(
            pair.base_scaler.transform(base_values)
        )
        plus_prediction = pair.plus_model.predict(
            pair.plus_scaler.transform(plus_values)
        )
        base_daily = daily_spearman(
            sub["trade_date_ny"].to_numpy(),
            base_prediction,
            sub["forward_return_60_atr"].to_numpy(dtype=np.float64),
            min_observations=config.min_daily_observations,
        ).rename(columns={"daily_ic": "base_daily_ic"})
        plus_daily = daily_spearman(
            sub["trade_date_ny"].to_numpy(),
            plus_prediction,
            sub["forward_return_60_atr"].to_numpy(dtype=np.float64),
            min_observations=config.min_daily_observations,
        ).rename(columns={"daily_ic": "plus_daily_ic"})
        daily = base_daily.merge(
            plus_daily,
            on="trade_date_ny",
            how="inner",
            suffixes=("_base", "_plus"),
            validate="one_to_one",
        )
        daily["daily_ic_delta"] = (
            daily["plus_daily_ic"] - daily["base_daily_ic"]
        )
        daily.insert(0, "trial_id", trial_id)
        daily.insert(1, "session", pair.session)
        daily.insert(2, "interaction_name", pair.interaction_name)
        daily_records.append(daily)
        delta_ci = _date_bootstrap_ci(
            daily["daily_ic_delta"],
            seed=config.random_seed + 50_000 + index,
            config=config,
        )
        dev = development.loc[trial_id]
        gates = {
            "gate_development_incremental_support": str(dev["development_label"])
            == "DEV_INCREMENTAL_SUPPORT",
            "gate_validation_delta": float(daily["daily_ic_delta"].mean())
            >= config.min_validation_interaction_delta,
            "gate_validation_ci_above_zero": delta_ci[0] > 0.0,
        }
        failed = [name for name, passed in gates.items() if not passed]
        gate_records.append(
            {
                "trial_id": trial_id,
                "session": pair.session,
                "interaction_name": pair.interaction_name,
                "target": "forward_return_60_atr",
                "development_label": dev["development_label"],
                "development_mean_daily_ic_delta": float(
                    dev["mean_daily_ic_delta"]
                ),
                "development_delta_ci_low": float(
                    dev["daily_ic_delta_ci_low"]
                ),
                "development_delta_ci_high": float(
                    dev["daily_ic_delta_ci_high"]
                ),
                "validation_paired_trading_date_count": len(daily),
                "validation_mean_daily_ic_delta": float(
                    daily["daily_ic_delta"].mean()
                ),
                "validation_delta_ci_low": delta_ci[0],
                "validation_delta_ci_high": delta_ci[1],
                "base_alpha": pair.base_alpha,
                "plus_alpha": pair.plus_alpha,
                "comparator_fit_sha256": pair.fit_sha256,
                **gates,
                "failed_gates": "|".join(failed),
                "final_interaction_label": (
                    "INTERACTION_ADVANCES"
                    if not failed
                    else "INTERACTION_DOES_NOT_ADVANCE"
                ),
            }
        )
    return (
        pd.DataFrame.from_records(gate_records),
        pd.concat(daily_records, ignore_index=True),
    )


def build_final_model_gates(
    project_root: Path,
    validation_metrics: pd.DataFrame,
    *,
    config: Section6Config,
) -> pd.DataFrame:
    """Join Development and Validation evidence and apply the frozen gate."""

    _, report_dir = _section_paths(project_root)
    development = pd.read_csv(
        report_dir / "section5_continuous_model_comparison.csv"
    )
    development = development.loc[
        development["selected_primary_family"].astype(bool)
    ].copy()
    rows: list[dict[str, Any]] = []
    for dev in development.itertuples(index=False):
        validation = validation_metrics.loc[
            validation_metrics["session"].eq(dev.session)
            & validation_metrics["target"].eq(dev.target)
            & validation_metrics["model_family"].eq(dev.model_family)
        ]
        if len(validation) != 1:
            raise AssertionError(
                f"Missing locked Validation metric for {dev.session} {dev.target}."
            )
        val = validation.iloc[0]
        gates = {
            "gate_integrity": True,
            "gate_development_delta": float(dev.paired_daily_ic_delta_mean)
            >= config.min_development_paired_delta,
            "gate_development_delta_ci": float(dev.paired_delta_ci_low) > 0.0,
            "gate_validation_delta": float(val["paired_daily_ic_delta_mean"])
            >= config.min_validation_paired_delta,
            "gate_validation_delta_ci": float(val["paired_delta_ci_low"]) > 0.0,
            "gate_positive_model_ic_both": float(dev.daily_ic_mean) > 0.0
            and float(val["daily_ic_mean"]) > 0.0,
            "gate_development_dates": int(dev.valid_oof_dates)
            >= config.min_development_oof_dates,
            "gate_validation_dates": int(val["valid_dates"])
            >= config.min_validation_dates,
            "gate_quarter_stability": bool(dev.complexity_stable)
            and (
                np.isnan(float(dev.quarter_same_sign_fraction))
                or float(dev.quarter_same_sign_fraction) >= 0.60
            )
            and (
                np.isnan(float(dev.largest_positive_quarter_share))
                or float(dev.largest_positive_quarter_share) <= 0.50
            ),
            "gate_best_10_concentration": (
                np.isnan(float(dev.best_10_positive_delta_share))
                or float(dev.best_10_positive_delta_share)
                <= config.max_best_10_positive_delta_share
            )
            and (
                np.isnan(float(val["best_10_positive_delta_share"]))
                or float(val["best_10_positive_delta_share"])
                <= config.max_best_10_positive_delta_share
            ),
            "gate_complexity_stability": bool(dev.complexity_stable),
        }
        failed = [name for name, passed in gates.items() if not passed]
        rows.append(
            {
                "session": dev.session,
                "target": dev.target,
                "model_family": dev.model_family,
                "development_daily_ic_mean": float(dev.daily_ic_mean),
                "development_paired_delta": float(
                    dev.paired_daily_ic_delta_mean
                ),
                "development_paired_delta_ci_low": float(
                    dev.paired_delta_ci_low
                ),
                "development_paired_delta_ci_high": float(
                    dev.paired_delta_ci_high
                ),
                "development_valid_dates": int(dev.valid_oof_dates),
                "validation_daily_ic_mean": float(val["daily_ic_mean"]),
                "validation_daily_ic_ci_low": float(val["daily_ic_ci_low"]),
                "validation_daily_ic_ci_high": float(val["daily_ic_ci_high"]),
                "validation_paired_delta": float(
                    val["paired_daily_ic_delta_mean"]
                ),
                "validation_paired_delta_ci_low": float(
                    val["paired_delta_ci_low"]
                ),
                "validation_paired_delta_ci_high": float(
                    val["paired_delta_ci_high"]
                ),
                "validation_valid_dates": int(val["valid_dates"]),
                **gates,
                "failed_gates": "|".join(failed),
                "model_advances": not failed,
                "model_verdict": (
                    "RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA"
                    if not failed
                    else "MODEL_DOES_NOT_ADVANCE"
                ),
            }
        )
    table = pd.DataFrame.from_records(rows)
    if table["model_advances"].any():
        raise AssertionError(
            "A model advanced despite the frozen 378-date Development support."
        )
    return table


def build_candidate_flags(
    project_root: Path,
    validation_continuous: pd.DataFrame,
    validation_expansion: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply frozen p90/p80 values unchanged for complete trial accounting."""

    root = Path(project_root)
    data_dir, report_dir = _section_paths(root)
    thresholds = pd.read_csv(report_dir / "section5_frozen_thresholds.csv")
    selected = pd.read_csv(
        report_dir / "section5_continuous_model_comparison.csv"
    )
    selected = selected.loc[
        selected["selected_primary_family"].astype(bool)
        & selected["target"].eq("forward_return_60_atr")
    ]
    selected_family = {
        str(row.session): str(row.model_family)
        for row in selected.itertuples(index=False)
    }
    development_direction = pd.read_parquet(
        data_dir / "section5_continuous_oof_predictions.parquet",
        columns=[
            "observation_id",
            "decision_timestamp_utc",
            "trade_date_ny",
            "session",
            "target",
            "model_family",
            "prediction",
        ],
    )
    development_direction = development_direction.loc[
        development_direction["target"].eq("forward_return_60_atr")
        & development_direction["model_family"].astype(str).eq(
            development_direction["session"].astype(str).map(selected_family)
        )
    ].copy()
    development_expansion = pd.read_parquet(
        data_dir / "section5_expansion_oof_predictions.parquet",
        columns=[
            "observation_id",
            "trade_date_ny",
            "session",
            "model_family",
            "probability",
        ],
    )
    development_expansion = development_expansion.loc[
        development_expansion["model_family"].eq("M1_O1")
    ]
    development = development_direction.merge(
        development_expansion[
            ["observation_id", "session", "probability"]
        ],
        on=["observation_id", "session"],
        how="inner",
        validate="one_to_one",
    )
    development["partition"] = "Development outer OOF"

    validation = validation_continuous.loc[
        validation_continuous["target"].eq("forward_return_60_atr")
    ][
        [
            "observation_id",
            "decision_timestamp_utc",
            "trade_date_ny",
            "session",
            "model_family",
            "prediction",
        ]
    ].merge(
        validation_expansion[
            ["observation_id", "session", "probability"]
        ],
        on=["observation_id", "session"],
        how="inner",
        validate="one_to_one",
    )
    validation["partition"] = "Retrospective Validation"
    combined = pd.concat([development, validation], ignore_index=True)
    flags: list[pd.DataFrame] = []
    count_records: list[dict[str, Any]] = []
    for (partition, session), group in combined.groupby(
        ["partition", "session"], sort=True
    ):
        direction_threshold = float(
            thresholds.loc[
                thresholds["session"].eq(session)
                & thresholds["threshold_role"].eq(
                    "direction_absolute_prediction_p90"
                ),
                "value",
            ].iloc[0]
        )
        expansion_threshold = float(
            thresholds.loc[
                thresholds["session"].eq(session)
                & thresholds["threshold_role"].eq(
                    "expansion_probability_p80"
                ),
                "value",
            ].iloc[0]
        )
        direction_flag = (
            np.isfinite(group["prediction"].to_numpy(dtype=np.float64))
            & group["prediction"].ne(0.0).to_numpy()
            & group["prediction"].abs().ge(direction_threshold).to_numpy()
        )
        expansion_flag = group["probability"].ge(expansion_threshold).to_numpy()
        for policy_variant in POLICY_VARIANTS:
            policy_flag = direction_flag.copy()
            if policy_variant == "direction_p90_and_expansion_p80":
                policy_flag &= expansion_flag
            output = group.copy()
            output["policy_variant"] = policy_variant
            output["direction_threshold_p90"] = direction_threshold
            output["expansion_threshold_p80"] = expansion_threshold
            output["direction_score_flag"] = direction_flag
            output["expansion_score_flag"] = expansion_flag
            output["candidate_flag"] = policy_flag
            output["direction"] = np.where(
                output["prediction"].gt(0.0), "long", "short"
            )
            output["policy_execution_allowed"] = False
            output["policy_status"] = "FROZEN_NO_POLICY"
            flags.append(output)
            count_records.append(
                {
                    "partition": partition,
                    "session": session,
                    "policy_variant": policy_variant,
                    "evaluation_rows": len(group),
                    "evaluation_dates": int(
                        pd.to_datetime(group["trade_date_ny"])
                        .dt.normalize()
                        .nunique()
                    ),
                    "direction_score_flags": int(direction_flag.sum()),
                    "expansion_score_flags": int(expansion_flag.sum()),
                    "candidate_flags": int(policy_flag.sum()),
                    "direction_threshold_p90": direction_threshold,
                    "expansion_threshold_p80": expansion_threshold,
                    "policy_execution_allowed": False,
                    "policy_status": "FROZEN_NO_POLICY",
                }
            )
    return (
        pd.concat(flags, ignore_index=True),
        pd.DataFrame.from_records(count_records),
    )


def daily_net_r_sharpe(
    daily_net_r: Sequence[float],
    *,
    annualization: int = 252,
) -> float:
    """Calculate the frozen daily-net-R Sharpe definition."""

    values = np.asarray(daily_net_r, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation == 0.0:
        return np.nan
    return float(np.sqrt(annualization) * np.mean(values) / standard_deviation)


def stationary_bootstrap_indices(
    length: int,
    *,
    replicates: int,
    restart_probability: float,
    seed: int,
) -> np.ndarray:
    """Generate Politis-Romano stationary-bootstrap row indices."""

    if length <= 0:
        raise ValueError("Stationary bootstrap length must be positive.")
    if replicates <= 0:
        raise ValueError("Stationary bootstrap replicates must be positive.")
    if not 0.0 < restart_probability <= 1.0:
        raise ValueError("Restart probability must lie in (0, 1].")
    rng = np.random.default_rng(seed)
    indices = np.empty((replicates, length), dtype=np.int64)
    indices[:, 0] = rng.integers(0, length, size=replicates)
    for position in range(1, length):
        restart = rng.random(replicates) < restart_probability
        continued = (indices[:, position - 1] + 1) % length
        new_start = rng.integers(0, length, size=replicates)
        indices[:, position] = np.where(restart, new_start, continued)
    return indices


def stationary_bootstrap_sharpe_ci(
    daily_net_r: Sequence[float],
    *,
    seed: int,
    config: Section6Config | None = None,
) -> tuple[float, float]:
    """Return the frozen stationary-bootstrap Sharpe percentile interval."""

    cfg = config or Section6Config()
    values = np.asarray(daily_net_r, dtype=np.float64)
    indices = stationary_bootstrap_indices(
        len(values),
        replicates=cfg.stationary_bootstrap_replicates,
        restart_probability=cfg.stationary_bootstrap_restart_probability,
        seed=seed,
    )
    estimates = np.asarray(
        [
            daily_net_r_sharpe(
                values[index], annualization=cfg.sharpe_annualization
            )
            for index in indices
        ],
        dtype=np.float64,
    )
    finite = estimates[np.isfinite(estimates)]
    if len(finite) == 0:
        return np.nan, np.nan
    low, high = np.quantile(finite, [0.025, 0.975], method="linear")
    return float(low), float(high)


def max_sharpe_adjusted_p_value(
    daily_policy_matrix: np.ndarray,
    *,
    selected_policy_index: int,
    seed: int,
    config: Section6Config | None = None,
) -> float:
    """Calculate the exact max-Sharpe stationary-bootstrap adjusted p-value."""

    cfg = config or Section6Config()
    values = np.asarray(daily_policy_matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Policy matrix must be dates by one or more policies.")
    if not 0 <= selected_policy_index < values.shape[1]:
        raise IndexError("Selected policy index is outside the policy family.")
    observed = daily_net_r_sharpe(
        values[:, selected_policy_index],
        annualization=cfg.sharpe_annualization,
    )
    centered = values - values.mean(axis=0, keepdims=True)
    indices = stationary_bootstrap_indices(
        len(values),
        replicates=cfg.stationary_bootstrap_replicates,
        restart_probability=cfg.stationary_bootstrap_restart_probability,
        seed=seed,
    )
    null_maximum = np.empty(cfg.stationary_bootstrap_replicates, dtype=np.float64)
    for replicate, index in enumerate(indices):
        sharpes = [
            daily_net_r_sharpe(
                centered[index, policy],
                annualization=cfg.sharpe_annualization,
            )
            for policy in range(values.shape[1])
        ]
        null_maximum[replicate] = np.nanmax(sharpes)
    return float(
        (
            1
            + np.sum(
                null_maximum[np.isfinite(null_maximum)]
                >= observed
            )
        )
        / (cfg.stationary_bootstrap_replicates + 1)
    )


def build_not_applicable_economics(
    candidate_counts: pd.DataFrame,
    policy_freeze: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create explicit N/A economic tables without opening GC bars."""

    if bool(policy_freeze["policy_execution_allowed"]):
        raise ValueError(
            "The N/A economics path is only valid for a frozen no-policy outcome."
        )
    records: list[dict[str, Any]] = []
    for policy_variant in POLICY_VARIANTS:
        for partition in (
            "Development outer OOF",
            "Retrospective Validation",
        ):
            support = candidate_counts.loc[
                candidate_counts["policy_variant"].eq(policy_variant)
                & candidate_counts["partition"].eq(partition)
            ]
            evaluation_dates = int(support["evaluation_dates"].max())
            for cost_scenario, cost_ticks in COST_TICKS.items():
                records.append(
                    {
                        "policy_variant": policy_variant,
                        "partition": partition,
                        "cost_scenario": cost_scenario,
                        "round_trip_cost_ticks": cost_ticks,
                        "cost_assumption_type": "proxy assumption; not a fill model",
                        "evaluation_dates_available": evaluation_dates,
                        "executed_trades": 0,
                        "gross_expectancy_ticks": np.nan,
                        "net_expectancy_ticks": np.nan,
                        "mean_net_r": np.nan,
                        "net_r_ci_low": np.nan,
                        "net_r_ci_high": np.nan,
                        "profit_factor": np.nan,
                        "total_net_r": np.nan,
                        "economics_status": NO_POLICY_REASON_CODE,
                    }
                )
    economics = pd.DataFrame.from_records(records)
    drawdown = (
        economics.loc[
            economics["cost_scenario"].eq("base"),
            ["policy_variant", "partition"],
        ]
        .drop_duplicates()
        .assign(
            maximum_drawdown_r=np.nan,
            time_under_water_dates=np.nan,
            pnl_concentration_top_10_dates=np.nan,
            worst_1pct_daily_net_r=np.nan,
            worst_5pct_daily_net_r=np.nan,
            worst_10pct_daily_net_r=np.nan,
            break_even_round_trip_cost_ticks=np.nan,
            drawdown_status=NO_POLICY_REASON_CODE,
        )
    )
    sharpe = (
        economics.loc[
            economics["cost_scenario"].eq("base"),
            ["policy_variant", "partition"],
        ]
        .drop_duplicates()
        .assign(
            daily_net_r_sharpe=np.nan,
            sharpe_ci_low=np.nan,
            sharpe_ci_high=np.nan,
            max_sharpe_adjusted_p_value=np.nan,
            policies_in_family=0,
            policy_family_names="",
            sharpe_status=NO_POLICY_REASON_CODE,
        )
    )
    verdict = pd.DataFrame.from_records(
        [
            {
                "policy_status": policy_freeze["policy_status"],
                "directional_model_gate_pass_count": 0,
                "policies_sequenced": 0,
                "gc_economics_executed": False,
                "gc_strategy_gate_pass": False,
                "mgc_work_permitted": False,
                "gc_gate_verdict": "GC_GATE_NOT_APPLICABLE_NO_DIRECTIONAL_MODEL",
                "headline_verdict": "PREDICTIVE_ONLY_NOT_DIRECTIONAL",
                "reason": policy_freeze["no_policy_reason"],
            }
        ]
    )
    return economics, drawdown, sharpe, verdict


def _complete_section6_ledger(
    project_root: Path,
    *,
    policy_freeze: Mapping[str, Any],
) -> pd.DataFrame:
    data_dir, _ = _section_paths(project_root)
    ledger = pd.read_parquet(data_dir / "section5_complete_trial_ledger.parquet")
    section6 = ledger["notebook_section"].eq(6)
    if int(section6.sum()) != 12:
        raise AssertionError("The frozen ledger must declare 12 Section 6 trials.")
    ledger.loc[section6, "status"] = "COMPLETE_NOT_APPLICABLE"
    ledger.loc[section6, "development_label"] = "FROZEN_NO_POLICY"
    ledger.loc[section6, "failed_gates"] = (
        "NO_DIRECTIONAL_MODEL_PASSED_FROZEN_MODEL_GATE"
    )
    ledger.loc[section6, "notes"] = (
        "Validation predictive scoring completed in the single locked batch; "
        "GC bars, P&L, drawdown, and Sharpe were not opened because policy "
        f"execution is prohibited. {policy_freeze['no_policy_reason']}"
    )
    if not ledger.loc[section6, "frozen_before_outcomes"].astype(bool).all():
        raise AssertionError("A Section 6 trial was not preregistered.")
    if not ledger.loc[section6, "status"].str.startswith("COMPLETE").all():
        raise AssertionError("A Section 6 trial remains incomplete.")
    return ledger


def run_locked_section6_batch(
    project_root: Path,
    authorization: LockedValidationAuthorization,
    *,
    config: Section6Config | None = None,
) -> Section6Result:
    """Run the single non-interactive Validation batch without a pause."""

    if not isinstance(authorization, LockedValidationAuthorization):
        raise PermissionError("Section 6 requires a verified hash authorization.")
    cfg = config or Section6Config()
    root = Path(project_root)
    data_dir, _ = _section_paths(root)
    policy_freeze = json.loads(
        (data_dir / "section5_pre_validation_policy.json").read_text(
            encoding="utf-8"
        )
    )
    if policy_freeze["pre_validation_policy_sha256"] != authorization.policy_sha256:
        raise AssertionError("Authorization and frozen policy differ.")
    model_bundle = joblib.load(data_dir / "section5_frozen_models.joblib")
    selected_models: dict[str, FittedContinuousModel] = dict(
        model_bundle["continuous"]
    )
    expansion_models: dict[str, FittedExpansionModel] = dict(
        model_bundle["expansion"]
    )

    # All comparator fitting is deterministic Development-only work and occurs
    # before the sole Validation outcome loader is called.
    anchor_models, anchor_registry = prepare_locked_anchor_comparators(
        root, selected_models
    )
    interaction_pairs, interaction_registry = (
        prepare_locked_interaction_comparators(root)
    )
    comparator_registry = pd.concat(
        [anchor_registry, interaction_registry], ignore_index=True
    )
    if not comparator_registry["prepared_before_validation_read"].all():
        raise AssertionError("A comparator was prepared after Validation opened.")

    inputs = load_locked_validation_inputs(root, authorization)
    continuous_predictions = score_frozen_continuous_models(
        inputs, selected_models, anchor_models
    )
    (
        continuous_metrics,
        continuous_daily,
        continuous_deciles,
        continuous_subperiods,
    ) = build_validation_continuous_evidence(
        continuous_predictions, config=cfg
    )
    expansion_predictions = score_frozen_expansion_models(
        inputs, expansion_models
    )
    development_expansion = pd.read_parquet(
        data_dir / "section5_expansion_oof_predictions.parquet"
    )
    frozen_thresholds = pd.read_csv(
        root
        / "reports/statistical_research/fes_project1/v1/"
        "section5_frozen_thresholds.csv"
    )
    expansion_metrics, expansion_reliability = (
        build_validation_expansion_evidence(
            expansion_predictions,
            development_expansion,
            frozen_thresholds,
            config=cfg,
        )
    )
    feature_gates, feature_daily = build_validation_feature_gates(
        root, inputs, config=cfg
    )
    validation_missingness = build_validation_missingness(inputs)
    validation_integrity = build_validation_integrity(inputs)
    interaction_gates, interaction_daily = score_validation_interactions(
        root, inputs, interaction_pairs, config=cfg
    )
    model_gates = build_final_model_gates(
        root, continuous_metrics, config=cfg
    )
    candidate_flags, candidate_counts = build_candidate_flags(
        root, continuous_predictions, expansion_predictions
    )
    economics, drawdown, sharpe, gc_verdict = (
        build_not_applicable_economics(candidate_counts, policy_freeze)
    )
    complete_ledger = _complete_section6_ledger(
        root, policy_freeze=policy_freeze
    )
    if policy_freeze["policy_status"] != "FROZEN_NO_POLICY":
        raise AssertionError("The executed economic path differs from the freeze.")
    if gc_verdict["gc_economics_executed"].any():
        raise AssertionError("No-policy Section 6 may not read economic outcomes.")
    if not inputs.access_audit.loc[
        inputs.access_audit["source"].eq("Historical Final outcomes"),
        "rows_loaded",
    ].eq(0).all():
        raise AssertionError("Historical Final was opened.")
    return Section6Result(
        validation_continuous_predictions=continuous_predictions,
        validation_continuous_metrics=continuous_metrics,
        validation_continuous_daily=continuous_daily,
        validation_continuous_deciles=continuous_deciles,
        validation_continuous_subperiods=continuous_subperiods,
        validation_expansion_predictions=expansion_predictions,
        validation_expansion_metrics=expansion_metrics,
        validation_expansion_reliability=expansion_reliability,
        feature_gate_table=feature_gates,
        feature_validation_daily_ic=feature_daily,
        validation_missingness=validation_missingness,
        validation_integrity=validation_integrity,
        interaction_gate_table=interaction_gates,
        interaction_validation_daily_delta=interaction_daily,
        model_gate_table=model_gates,
        candidate_flags=candidate_flags,
        candidate_counts=candidate_counts,
        gc_economic_table=economics,
        gc_drawdown_table=drawdown,
        sharpe_table=sharpe,
        gc_gate_verdict=gc_verdict,
        comparator_registry=comparator_registry,
        access_audit=inputs.access_audit,
        complete_trial_ledger=complete_ledger,
        authorization=authorization,
        policy_freeze=policy_freeze,
        config=cfg,
    )


def _assert_section6_output_targets_absent(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Immutable Section 6 output target(s) already exist:\n" + formatted
        )


def save_section6_outputs(
    result: Section6Result,
    project_root: Path,
) -> dict[str, Any]:
    """Persist immutable Section 6 artifacts and the one-time batch hash."""

    root = Path(project_root)
    data_dir, report_dir = _section_paths(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    parquet_tables: dict[str, pd.DataFrame] = {
        "section6_validation_continuous_predictions.parquet": (
            result.validation_continuous_predictions
        ),
        "section6_validation_continuous_daily.parquet": (
            result.validation_continuous_daily
        ),
        "section6_validation_expansion_predictions.parquet": (
            result.validation_expansion_predictions
        ),
        "section6_feature_validation_daily_ic.parquet": (
            result.feature_validation_daily_ic
        ),
        "section6_interaction_validation_daily_delta.parquet": (
            result.interaction_validation_daily_delta
        ),
        "section6_candidate_flags.parquet": result.candidate_flags,
        "section6_complete_trial_ledger.parquet": result.complete_trial_ledger,
    }
    csv_tables: dict[str, pd.DataFrame] = {
        "section6_validation_continuous_metrics.csv": (
            result.validation_continuous_metrics
        ),
        "section6_validation_continuous_deciles.csv": (
            result.validation_continuous_deciles
        ),
        "section6_validation_continuous_subperiods.csv": (
            result.validation_continuous_subperiods
        ),
        "section6_validation_expansion_metrics.csv": (
            result.validation_expansion_metrics
        ),
        "section6_validation_expansion_reliability.csv": (
            result.validation_expansion_reliability
        ),
        "section6_final_feature_gates.csv": result.feature_gate_table,
        "section6_validation_missingness.csv": result.validation_missingness,
        "section6_validation_integrity.csv": result.validation_integrity,
        "section6_final_interaction_gates.csv": result.interaction_gate_table,
        "section6_final_model_gates.csv": result.model_gate_table,
        "section6_candidate_counts.csv": result.candidate_counts,
        "section6_gc_economics.csv": result.gc_economic_table,
        "section6_gc_drawdown.csv": result.gc_drawdown_table,
        "section6_daily_net_r_sharpe.csv": result.sharpe_table,
        "section6_gc_gate_verdict.csv": result.gc_gate_verdict,
        "section6_comparator_registry.csv": result.comparator_registry,
        "section6_access_audit.csv": result.access_audit,
        "section6_complete_trial_ledger.csv": result.complete_trial_ledger,
    }
    config_path = data_dir / "section6_runtime_config.json"
    manifest_path = report_dir / "section6_hash_manifest.json"
    checkpoint_path = report_dir / "section6_checkpoint.json"
    completion_path = report_dir / "section6_batch_completion.json"
    target_paths = [
        *[data_dir / name for name in parquet_tables],
        *[report_dir / name for name in csv_tables],
        config_path,
        manifest_path,
        checkpoint_path,
        completion_path,
    ]
    _assert_section6_output_targets_absent(target_paths)

    artifact_paths: list[Path] = []
    for filename, table in parquet_tables.items():
        path = data_dir / filename
        table.to_parquet(path, index=False)
        reloaded = pd.read_parquet(path)
        if len(reloaded) != len(table):
            raise AssertionError(f"Section 6 parquet reload mismatch: {filename}")
        artifact_paths.append(path)
    for filename, table in csv_tables.items():
        path = report_dir / filename
        table.to_csv(path, index=False, float_format="%.17g")
        artifact_paths.append(path)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "section": 6,
                "config": asdict(result.config),
                "authorization": asdict(result.authorization),
                "historical_final_outcomes_read": False,
                "gc_economics_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths.append(config_path)
    manifest_records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "artifact_class": (
                "data" if path.is_relative_to(data_dir) else "report"
            ),
        }
        for path in sorted(artifact_paths)
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "section": 6,
                "artifacts": manifest_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = sha256_file(manifest_path)
    batch_payload = {
        "schema_version": "1.0.0",
        "section": 6,
        "section5_manifest_sha256": (
            result.authorization.section5_manifest_sha256
        ),
        "policy_sha256": result.authorization.policy_sha256,
        "models_sha256": result.authorization.models_sha256,
        "frozen_config_sha256": result.authorization.frozen_config_sha256,
        "section6_manifest_sha256": manifest_sha256,
        "artifact_count": len(manifest_records),
        "validation_rows_opened": int(
            result.access_audit.loc[
                result.access_audit["source"].str.endswith(
                    "forward_labels_gc.parquet"
                ),
                "rows_loaded",
            ].iloc[0]
        ),
        "historical_final_outcomes_read": False,
        "gc_economics_executed": False,
        "headline_verdict": str(
            result.gc_gate_verdict["headline_verdict"].iloc[0]
        ),
        "gc_gate_verdict": str(
            result.gc_gate_verdict["gc_gate_verdict"].iloc[0]
        ),
    }
    batch_sha256 = hashlib.sha256(
        canonical_json(batch_payload).encode("utf-8")
    ).hexdigest()
    completion = {
        **batch_payload,
        "batch_completion_sha256": batch_sha256,
        "status": "COMPLETED_ONE_TIME_LOCKED_BATCH",
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint = {
        "schema_version": "1.0.0",
        "section": 6,
        "status": "COMPLETED_STOP_BEFORE_SECTION_7",
        "single_locked_validation_batch_completed": True,
        "batch_completion_sha256": batch_sha256,
        "batch_completion_file_sha256": sha256_file(completion_path),
        "section6_manifest_sha256": manifest_sha256,
        "section5_manifest_sha256": result.authorization.section5_manifest_sha256,
        "policy_sha256": result.authorization.policy_sha256,
        "retrospective_validation_outcomes_read": True,
        "historical_final_outcomes_read": False,
        "gc_economic_bars_read": False,
        "gc_economics_executed": False,
        "feature_advancers": int(
            result.feature_gate_table["final_feature_label"]
            .eq("FEATURE_ADVANCES")
            .sum()
        ),
        "interaction_advancers": int(
            result.interaction_gate_table["final_interaction_label"]
            .eq("INTERACTION_ADVANCES")
            .sum()
        ),
        "continuous_model_advancers": int(
            result.model_gate_table["model_advances"].sum()
        ),
        "classifier_advancers": int(
            result.validation_expansion_metrics["classifier_advances"].sum()
        ),
        "policy_status": result.policy_freeze["policy_status"],
        "gc_gate_verdict": result.gc_gate_verdict.to_dict(orient="records")[0],
        "costs_ticks": dict(COST_TICKS),
        "economics_status": NO_POLICY_REASON_CODE,
        "sharpe_status": NO_POLICY_REASON_CODE,
        "mgc_work_permitted": False,
        "tests_status": "PENDING_NOTEBOOK_AND_FULL_SUITE",
        "stop_boundary": (
            "Section 6 complete. Do not start MGC transfer or read Historical "
            "Final before explicit Section 7 instruction."
        ),
    }
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "batch_completion_path": completion_path,
        "batch_completion_sha256": batch_sha256,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": checkpoint_path,
        "artifact_count": len(manifest_records),
        "headline_verdict": batch_payload["headline_verdict"],
        "gc_gate_verdict": batch_payload["gc_gate_verdict"],
    }
