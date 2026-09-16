"""Governed runner for the POI-first feature-combination study."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.statistical_research.feature_combination import (
    ALL_FES_SOURCE_FEATURES,
    BASE_COST_TICKS,
    BOOTSTRAP_SEED,
    CONTRACT_RELATIVE_PATH,
    EXPECTED_CONTRACT_SHA256,
    FES_FEATURES_BY_SESSION,
    GENERAL_MODEL_ORDER,
    PESSIMISTIC_COST_TICKS,
    POI_CONTEXT_FEATURES,
    POI_MODEL_ORDER,
    STAT15_FEATURES,
    CombinationResearchResult,
    DateFold,
    add_poi_interactions,
    assert_no_final_test,
    build_exact_date_folds,
    canonical_sha256,
    deduplicate_poi_events,
    development_gate_failures,
    fit_development_predict_validation,
    fold_manifest_payload,
    folds_to_frame,
    generate_oof_predictions,
    make_policy_candidates,
    model_features,
    noninterpolated_quantile,
    sequence_fixed_horizon,
    sha256_file,
    summarize_policy,
    summarize_predictions,
    validation_gate_failures,
)

RESEARCH_ID = "poi_feature_combination_v1"
SESSIONS = ("London", "New York")
OUTPUT_RELATIVE_DIR = Path("data/processed/statistical_research/poi_feature_combination/v1")
REPORT_RELATIVE_DIR = Path("reports/statistical_research/poi_feature_combination/v1")

POI_PATH = Path("data/processed/section7_true_poi_context_frame_gc.parquet")
ELIGIBLE_PATH = Path("data/processed/statistical_research/eligible_observations_gc.parquet")
LABEL_PATH = Path("data/processed/statistical_research/forward_labels_gc.parquet")
FEATURE_PATH = Path("data/processed/statistical_research/feature_matrix_gc.parquet")
FROZEN_EXPANSION_PATH = Path(
    "data/processed/statistical_research/frozen_expansion_feature_set_gc.parquet"
)
BRANCH_B_REGISTRY_PATH = Path("data/processed/statistical_research/feature_registry_gc.parquet")
POI_REGISTRY_PATH = Path("reports/section7_true_poi_feature_registry.csv")
FES_SELECTION_PATH = Path("reports/statistical_research/fes_project1/v1/section6_checkpoint.json")
FES_HASH_MANIFEST_PATH = Path(
    "reports/statistical_research/fes_project1/v1/section2_hash_manifest.json"
)
FES_INPUT_MANIFEST_PATH = Path("reports/statistical_research/fes_project1/v1/input_manifest.json")
FES_FINAL_GATES_PATH = Path(
    "reports/statistical_research/fes_project1/v1/section6_final_feature_gates.csv"
)
FES_REGISTRY_PATH = Path(
    "data/processed/statistical_research/fes_project1/v1/scalar_feature_registry_gc.parquet"
)
FES_DEVELOPMENT_PATH = Path(
    "data/processed/statistical_research/fes_project1/v1/scalar_features_development_gc.parquet"
)
FES_VALIDATION_PATH = Path(
    "data/processed/statistical_research/fes_project1/v1/"
    "scalar_features_validation_sealed_gc.parquet"
)

POI_COLUMNS = (
    "true_retest_id",
    "retest_bar_id",
    "retest_ts_event_utc",
    "trade_date_ny",
    "direction",
    "research_partition",
    "feat_execution_session",
    "feat_first_touch",
    "feat_15bar_structural_validation",
) + POI_CONTEXT_FEATURES

ELIGIBLE_COLUMNS = (
    "observation_id",
    "decision_bar_id",
    "decision_timestamp_utc",
    "entry_timestamp_utc",
    "trade_date_ny",
    "entry_session",
    "research_partition",
)

FEATURE_COLUMNS = (
    (
        "observation_id",
        "trade_date_ny",
        "entry_session",
        "research_partition",
    )
    + STAT15_FEATURES
    + ("normalized_ols_slope_30",)
)

FES_COLUMNS = (
    "observation_id",
    "decision_timestamp_utc",
    "trade_date_ny",
    "entry_session",
    "research_partition",
) + ALL_FES_SOURCE_FEATURES

LABEL_COLUMNS = (
    "observation_id",
    "decision_bar_id",
    "decision_timestamp_utc",
    "entry_timestamp_utc",
    "trade_date_ny",
    "entry_session",
    "research_partition",
    "exit_timestamp_utc_60",
    "forward_return_60_ticks",
    "forward_return_60_atr",
    "future_range_60_atr",
    "label_available_60",
)

POLICY_TRADE_COLUMNS = (
    "branch",
    "session",
    "source_model_id",
    "policy_model_id",
    "partition",
    "observation_id",
    "true_retest_id",
    "trade_date_ny",
    "entry_timestamp_utc",
    "exit_timestamp_utc_60",
    "prediction",
    "prediction_tail",
    "policy_orientation",
    "trade_direction",
    "low_threshold",
    "high_threshold",
    "gross_ticks",
    "net_ticks_base",
    "net_ticks_pessimistic",
)

MODEL_ANCHOR = {"POI": "POI1_CONTEXT", "GENERAL": "GEN1_STAT15"}
MODEL_ORDER = {"POI": POI_MODEL_ORDER, "GENERAL": GENERAL_MODEL_ORDER}
PAIRED_REQUIRED = {
    "POI": set(POI_MODEL_ORDER[1:]),
    "GENERAL": {"GEN3_STAT15_FES4"},
}


class BoundaryAccessError(PermissionError):
    """Carry non-descriptive forensic context for a rejected protected read."""

    def __init__(
        self,
        message: str,
        *,
        access_record: Mapping[str, Any],
        final_test_opened: bool,
    ) -> None:
        super().__init__(message)
        self.access_record = dict(access_record)
        self.final_test_opened = bool(final_test_opened)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, date)):
        return pd.Timestamp(value).isoformat()
    if value is pd.NaT:
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _partition_filters(partition: str, *, lowercase: bool = False) -> list[tuple]:
    if partition == "Development":
        label = "development" if lowercase else "Development"
        return [
            ("research_partition", "==", label),
            ("trade_date_ny", "<", date(2024, 1, 1)),
        ]
    if partition == "Validation":
        label = "validation" if lowercase else "Validation"
        return [
            ("research_partition", "==", label),
            ("trade_date_ny", ">=", date(2024, 1, 1)),
            ("trade_date_ny", "<", date(2025, 1, 1)),
        ]
    raise KeyError(f"unauthorized partition: {partition}")


def _materialized_final_test(frame: pd.DataFrame, *, timestamp_columns: Sequence[str]) -> bool:
    if "trade_date_ny" in frame:
        dates = pd.to_datetime(frame["trade_date_ny"], errors="coerce")
        if dates.dt.tz_localize(None).ge(pd.Timestamp("2025-01-01")).any():
            return True
    for column in ("research_partition", "partition"):
        if column in frame:
            values = (
                frame[column]
                .astype("string")
                .str.strip()
                .str.lower()
                .str.replace("-", "_", regex=False)
                .str.replace(" ", "_", regex=False)
            )
            if values.str.contains("final", na=False).any():
                return True
    for column in timestamp_columns:
        if column in frame:
            timestamps = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if timestamps.ge(pd.Timestamp("2025-01-01", tz="UTC")).any():
                return True
    return False


def _read_projected(
    root: Path,
    relative_path: Path,
    *,
    columns: Sequence[str],
    filters: list[tuple],
    audit: list[dict[str, Any]],
    stage: str,
    timestamp_columns: Sequence[str] = (),
) -> pd.DataFrame:
    path = root / relative_path
    table = pq.read_table(
        path,
        columns=list(columns),
        filters=filters,
        use_pandas_metadata=False,
    )
    frame = table.to_pandas(ignore_metadata=True)
    access_record = {
        "stage": stage,
        "path": relative_path.as_posix(),
        "projected_columns": "|".join(columns),
        "filters": json.dumps(filters, default=str, sort_keys=True),
        "materialized_rows": int(len(frame)),
        "materialized_columns": int(len(frame.columns)),
        "minimum_trade_date": (
            pd.to_datetime(frame["trade_date_ny"]).min()
            if len(frame) and "trade_date_ny" in frame
            else pd.NaT
        ),
        "maximum_trade_date": (
            pd.to_datetime(frame["trade_date_ny"]).max()
            if len(frame) and "trade_date_ny" in frame
            else pd.NaT
        ),
        "boundary_check": "PASS",
    }
    try:
        assert_no_final_test(
            frame,
            context=f"{stage}:{relative_path.as_posix()}",
            timestamp_columns=timestamp_columns,
        )
    except PermissionError as error:
        access_record["boundary_check"] = "FAILED"
        audit.append(access_record)
        raise BoundaryAccessError(
            str(error),
            access_record=access_record,
            final_test_opened=_materialized_final_test(frame, timestamp_columns=timestamp_columns),
        ) from error
    audit.append(access_record)
    return frame


def _input_provenance(root: Path, relative_paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_path in relative_paths:
        path = root / relative_path
        row: dict[str, Any] = {
            "path": relative_path.as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "format": path.suffix.lower().lstrip("."),
        }
        if path.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(path)
            row.update(
                {
                    "footer_total_rows_not_analytically_reported": int(parquet.metadata.num_rows),
                    "row_groups": int(parquet.metadata.num_row_groups),
                    "schema": str(parquet.schema_arrow),
                }
            )
        elif path.suffix.lower() == ".csv":
            header = pd.read_csv(path, nrows=0)
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                row["rows"] = max(sum(1 for _ in stream) - 1, 0)
            row["schema_columns"] = list(header.columns)
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            row["top_level_keys"] = sorted(payload) if isinstance(payload, dict) else []
        rows.append(row)
    return rows


def _validate_frozen_feature_sets(root: Path) -> None:
    fes_hash_manifest = json.loads((root / FES_HASH_MANIFEST_PATH).read_text(encoding="utf-8"))
    frozen_fes_hashes = fes_hash_manifest.get("artifacts", {})
    for relative_path in (
        FES_REGISTRY_PATH,
        FES_DEVELOPMENT_PATH,
        FES_VALIDATION_PATH,
    ):
        expected_hash = frozen_fes_hashes.get(relative_path.as_posix())
        observed_hash = sha256_file(root / relative_path)
        if expected_hash != observed_hash:
            raise PermissionError(f"frozen FES digest mismatch for {relative_path.as_posix()}")

    input_manifest = json.loads((root / FES_INPUT_MANIFEST_PATH).read_text(encoding="utf-8"))
    frozen_input_hashes = {
        row["path"]: row["sha256"]
        for row in input_manifest.get("inputs", [])
        if row.get("status") == "VERIFIED_INPUT"
    }
    for relative_path in (
        ELIGIBLE_PATH,
        LABEL_PATH,
        FEATURE_PATH,
        BRANCH_B_REGISTRY_PATH,
        FROZEN_EXPANSION_PATH,
    ):
        expected_hash = frozen_input_hashes.get(relative_path.as_posix())
        observed_hash = sha256_file(root / relative_path)
        if expected_hash != observed_hash:
            raise PermissionError(
                f"frozen shared-input digest mismatch for {relative_path.as_posix()}"
            )

    expansion = pq.read_table(
        root / FROZEN_EXPANSION_PATH,
        columns=["feature_name", "in_frozen_set"],
        use_pandas_metadata=False,
    ).to_pandas(ignore_metadata=True)
    frozen_stat15 = tuple(
        expansion.loc[expansion["in_frozen_set"].fillna(False).astype(bool), "feature_name"].astype(
            str
        )
    )
    if frozen_stat15 != STAT15_FEATURES:
        raise ValueError("STAT15 constants differ from the frozen expansion feature set")

    fes_registry = pq.read_table(
        root / FES_REGISTRY_PATH,
        columns=["feature_name"],
        use_pandas_metadata=False,
    ).to_pandas(ignore_metadata=True)
    registered_fes = set(fes_registry["feature_name"].astype(str))
    missing_fes = sorted(set(ALL_FES_SOURCE_FEATURES) - registered_fes)
    if missing_fes:
        raise ValueError(f"FES4 source fields are absent from the frozen registry: {missing_fes}")

    feature_gates = pd.read_csv(
        root / FES_FINAL_GATES_PATH,
        usecols=["session", "feature_name", "target", "final_feature_label"],
    )
    observed_advancers = {
        (row.session, row.feature_name, row.target)
        for row in feature_gates.loc[
            feature_gates["final_feature_label"].eq("FEATURE_ADVANCES")
        ].itertuples()
    }
    expected_advancers = {
        ("London", FES_FEATURES_BY_SESSION["London"][0], "future_range_60_atr"),
        ("London", FES_FEATURES_BY_SESSION["London"][1], "future_range_60_atr"),
        (
            "New York",
            FES_FEATURES_BY_SESSION["New York"][0],
            "forward_return_60_atr",
        ),
        (
            "New York",
            FES_FEATURES_BY_SESSION["New York"][1],
            "future_range_60_atr",
        ),
    }
    if observed_advancers != expected_advancers:
        raise PermissionError("frozen FES advancing session/feature/target cells changed")

    checkpoint = json.loads((root / FES_SELECTION_PATH).read_text(encoding="utf-8"))
    checkpoint_requirements = (
        checkpoint.get("feature_advancers") == 4,
        checkpoint.get("historical_final_outcomes_read") is False,
        checkpoint.get("mgc_work_permitted") is False,
        checkpoint.get("policy_status") == "FROZEN_NO_POLICY",
        checkpoint.get("single_locked_validation_batch_completed") is True,
        checkpoint.get("status") == "COMPLETED_STOP_BEFORE_SECTION_7",
    )
    if not all(checkpoint_requirements):
        raise PermissionError("frozen FES authorization checkpoint changed")


def _canonical_partition(values: pd.Series) -> pd.Series:
    normalized = (
        values.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    mapping = {
        "development": "Development",
        "validation": "Validation",
        "retrospective_validation": "Validation",
    }
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        bad = sorted(set(normalized.loc[mapped.isna()].dropna()))
        raise PermissionError(f"unknown partition labels after filtering: {bad}")
    return mapped


def _assert_equal_columns(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    kind: str = "text",
) -> None:
    if kind == "date":
        left_values = pd.to_datetime(frame[left]).dt.tz_localize(None).dt.normalize()
        right_values = pd.to_datetime(frame[right]).dt.tz_localize(None).dt.normalize()
    elif kind == "timestamp":
        left_values = pd.to_datetime(frame[left], utc=True)
        right_values = pd.to_datetime(frame[right], utc=True)
    elif kind == "partition":
        left_values = _canonical_partition(frame[left])
        right_values = _canonical_partition(frame[right])
    else:
        left_values = frame[left].astype("string")
        right_values = frame[right].astype("string")
    equal = left_values.eq(right_values) | (left_values.isna() & right_values.isna())
    if not bool(equal.all()):
        mismatch = int((~equal).sum())
        raise ValueError(f"join invariant failed for {left}/{right}: {mismatch} rows")


def _load_predictor_base(
    root: Path,
    *,
    partition: str,
    audit: list[dict[str, Any]],
    sessions: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filters = _partition_filters(partition)
    if sessions is not None:
        if not sessions:
            raise ValueError("predictor loader received an empty session capability")
        filters.append(("entry_session", "in", list(sessions)))
    eligible = _read_projected(
        root,
        ELIGIBLE_PATH,
        columns=ELIGIBLE_COLUMNS,
        filters=filters,
        audit=audit,
        stage=f"{partition}:eligible",
        timestamp_columns=("decision_timestamp_utc", "entry_timestamp_utc"),
    )
    features = _read_projected(
        root,
        FEATURE_PATH,
        columns=FEATURE_COLUMNS,
        filters=filters,
        audit=audit,
        stage=f"{partition}:statistical_features",
        timestamp_columns=(),
    )
    fes_path = FES_DEVELOPMENT_PATH if partition == "Development" else FES_VALIDATION_PATH
    fes = _read_projected(
        root,
        fes_path,
        columns=FES_COLUMNS,
        filters=filters,
        audit=audit,
        stage=f"{partition}:fes_features",
        timestamp_columns=("decision_timestamp_utc",),
    )
    base = eligible.merge(
        features,
        on="observation_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    if len(base) != len(eligible) or len(features) != len(eligible):
        raise ValueError("eligible/statistical-feature observation coverage is not exact")
    for column, kind in (
        ("trade_date_ny", "date"),
        ("entry_session", "text"),
        ("research_partition", "partition"),
    ):
        _assert_equal_columns(base, column, f"{column}_feature", kind=kind)
        base = base.drop(columns=f"{column}_feature")

    base = base.merge(
        fes,
        on="observation_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_fes"),
    )
    if len(base) != len(eligible) or len(fes) != len(eligible):
        raise ValueError("eligible/FES observation coverage is not exact")
    for column, kind in (
        ("decision_timestamp_utc", "timestamp"),
        ("trade_date_ny", "date"),
        ("entry_session", "text"),
        ("research_partition", "partition"),
    ):
        _assert_equal_columns(base, column, f"{column}_fes", kind=kind)
        base = base.drop(columns=f"{column}_fes")
    base["research_partition"] = _canonical_partition(base["research_partition"])
    base["trade_date_ny"] = pd.to_datetime(base["trade_date_ny"]).dt.normalize()
    base = base.loc[base["entry_session"].isin(SESSIONS)].copy()
    return base.reset_index(drop=True), eligible


def _load_poi_candidates(
    root: Path,
    *,
    partition: str,
    eligible: pd.DataFrame,
    audit: list[dict[str, Any]],
    sessions: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    filters = _partition_filters(partition, lowercase=True)
    if sessions is not None:
        if not sessions:
            raise ValueError("POI loader received an empty session capability")
        filters.append(("feat_execution_session", "in", list(sessions)))
    poi = _read_projected(
        root,
        POI_PATH,
        columns=POI_COLUMNS,
        filters=filters,
        audit=audit,
        stage=f"{partition}:poi_candidates_pre_outcome",
        timestamp_columns=("retest_ts_event_utc",),
    )
    before = len(poi)
    joined = poi.merge(
        eligible,
        left_on="retest_bar_id",
        right_on="decision_bar_id",
        how="inner",
        validate="many_to_one",
        suffixes=("_poi", ""),
    )
    coverage = len(joined) / before if before else math.nan
    if not np.isfinite(coverage) or coverage < 0.99:
        raise ValueError(f"POI-to-eligible bridge coverage below 99%: {coverage}")
    _assert_equal_columns(joined, "retest_ts_event_utc", "decision_timestamp_utc", kind="timestamp")
    _assert_equal_columns(joined, "trade_date_ny_poi", "trade_date_ny", kind="date")
    _assert_equal_columns(joined, "feat_execution_session", "entry_session", kind="text")
    _assert_equal_columns(joined, "research_partition_poi", "research_partition", kind="partition")
    joined = joined.drop(
        columns=[
            "trade_date_ny_poi",
            "research_partition_poi",
            "retest_ts_event_utc",
        ]
    )
    joined["research_partition"] = _canonical_partition(joined["research_partition"])
    deduplicated = deduplicate_poi_events(joined)
    audit_row = {
        "partition": partition,
        "poi_candidate_rows": int(before),
        "bridge_matched_rows": int(len(joined)),
        "bridge_unmatched_rows": int(before - len(joined)),
        "bridge_coverage": float(coverage),
        "deduplicated_rows": int(len(deduplicated)),
        "dedup_removed_rows": int(len(joined) - len(deduplicated)),
    }
    return deduplicated, audit_row


def _load_labels(
    root: Path,
    *,
    partition: str,
    audit: list[dict[str, Any]],
    stage: str,
    observation_ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    filters = _partition_filters(partition)
    filters.extend([("entry_timestamp_utc", "<", pd.Timestamp("2025-01-01", tz="UTC"))])
    if partition == "Validation":
        filters.append(("exit_timestamp_utc_60", "<", pd.Timestamp("2025-01-01", tz="UTC")))
    if observation_ids is not None:
        if not observation_ids:
            raise ValueError("outcome loader received an empty observation-ID capability")
        filters.append(("observation_id", "in", list(observation_ids)))
    return _read_projected(
        root,
        LABEL_PATH,
        columns=LABEL_COLUMNS,
        filters=filters,
        audit=audit,
        stage=stage,
        timestamp_columns=(
            "decision_timestamp_utc",
            "entry_timestamp_utc",
            "exit_timestamp_utc_60",
        ),
    )


def _join_labels(base: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(
        labels,
        on="observation_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_label"),
    )
    coverage = len(merged) / len(base) if len(base) else math.nan
    if not np.isfinite(coverage) or coverage < 0.99:
        raise ValueError(f"60-minute label observation coverage is below 99%: {coverage}")
    for column, kind in (
        ("decision_bar_id", "text"),
        ("decision_timestamp_utc", "timestamp"),
        ("entry_timestamp_utc", "timestamp"),
        ("trade_date_ny", "date"),
        ("entry_session", "text"),
        ("research_partition", "partition"),
    ):
        _assert_equal_columns(merged, column, f"{column}_label", kind=kind)
        merged = merged.drop(columns=f"{column}_label")
    merged = merged.loc[merged["label_available_60"].fillna(False).astype(bool)].copy()
    assert_no_final_test(
        merged,
        context="joined 60-minute label frame",
        timestamp_columns=(
            "decision_timestamp_utc",
            "entry_timestamp_utc",
            "exit_timestamp_utc_60",
        ),
    )
    return merged.reset_index(drop=True)


def _build_poi_model_frame(
    deduplicated: pd.DataFrame,
    predictor_base: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    selected = deduplicated.merge(
        predictor_base,
        on=(
            "observation_id",
            "decision_bar_id",
            "decision_timestamp_utc",
            "entry_timestamp_utc",
            "trade_date_ny",
            "entry_session",
            "research_partition",
        ),
        how="inner",
        validate="one_to_one",
        suffixes=("", "_predictor"),
    )
    selected = _join_labels(selected, labels)
    direction = selected["direction"].astype("string").str.strip().str.lower()
    sign = direction.map({"bullish": 1.0, "bearish": -1.0})
    if sign.isna().any():
        bad = sorted(set(direction.loc[sign.isna()].dropna()))
        raise ValueError(f"invalid POI directions: {bad}")
    selected["poi_direction_sign"] = sign.astype("float64")
    selected["signed_continuation_return_60_atr"] = (
        selected["poi_direction_sign"] * selected["forward_return_60_atr"]
    )
    selected["signed_continuation_return_60_ticks"] = (
        selected["poi_direction_sign"] * selected["forward_return_60_ticks"]
    )
    return add_poi_interactions(selected)


def _build_general_model_frame(predictor_base: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return _join_labels(predictor_base, labels)


def _target_for(branch: str, target_role: str) -> str:
    if target_role == "OPPORTUNITY_DIAGNOSTIC":
        return "future_range_60_atr"
    if target_role != "DIRECTIONAL":
        raise KeyError(f"unknown target role: {target_role}")
    return "signed_continuation_return_60_atr" if branch == "POI" else "forward_return_60_atr"


def _prediction_column(target_role: str, model_id: str) -> str:
    prefix = "pred_direction" if target_role == "DIRECTIONAL" else "pred_opportunity"
    return f"{prefix}__{model_id}"


def _analyze_development(
    *,
    branch: str,
    frame: pd.DataFrame,
    folds_by_session: Mapping[str, Sequence[DateFold]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    metrics: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    models = MODEL_ORDER[branch]
    anchor = MODEL_ANCHOR[branch]

    for session in SESSIONS:
        session_frame = frame.loc[frame["entry_session"].eq(session)].copy().reset_index(drop=True)
        folds = list(folds_by_session.get(session, ()))
        for target_role in ("DIRECTIONAL", "OPPORTUNITY_DIAGNOSTIC"):
            target = _target_for(branch, target_role)
            for model_id in models:
                prediction_column = _prediction_column(target_role, model_id)
                if folds:
                    prediction, _, coefficients = generate_oof_predictions(
                        session_frame,
                        folds,
                        features=model_features(branch, model_id, session),
                        target=target,
                    )
                    session_frame[prediction_column] = prediction
                    if not coefficients.empty:
                        coefficients = coefficients.assign(
                            branch=branch,
                            session=session,
                            model_id=model_id,
                            target_role=target_role,
                            partition="Development OOF",
                        )
                        coefficient_frames.append(coefficients)
                else:
                    session_frame[prediction_column] = np.nan

        for target_role in ("DIRECTIONAL", "OPPORTUNITY_DIAGNOSTIC"):
            target = _target_for(branch, target_role)
            anchor_column = _prediction_column(target_role, anchor)
            for model_id in models:
                prediction_column = _prediction_column(target_role, model_id)
                summary, daily = summarize_predictions(
                    session_frame,
                    prediction_column=prediction_column,
                    target_column=target,
                    anchor_prediction_column=anchor_column,
                    seed=BOOTSTRAP_SEED,
                )
                summary.update(
                    {
                        "branch": branch,
                        "session": session,
                        "model_id": model_id,
                        "target_role": target_role,
                        "target_column": target,
                        "partition": "Development OOF",
                        "fold_count": len(folds),
                        "feature_count": len(model_features(branch, model_id, session)),
                    }
                )
                metrics.append(summary)
                if not daily.empty:
                    daily_frames.append(
                        daily.assign(
                            branch=branch,
                            session=session,
                            model_id=model_id,
                            target_role=target_role,
                            partition="Development OOF",
                        )
                    )
                if target_role == "DIRECTIONAL":
                    selectable = not (branch == "POI" and session == "London")
                    failures = development_gate_failures(
                        summary,
                        branch=branch,
                        requires_paired=model_id in PAIRED_REQUIRED[branch],
                    )
                    if not selectable:
                        failures = ["diagnostic_only_london_poi", *failures]
                    selection_rows.append(
                        {
                            "branch": branch,
                            "session": session,
                            "model_id": model_id,
                            "selectable": selectable,
                            "development_eligible": selectable and not failures,
                            "development_gate_failures": "|".join(failures),
                            "development_daily_ic": summary["daily_ic_mean"],
                            "development_ic_ci_low": summary["daily_ic_ci_low"],
                            "development_observation_count": summary["observation_count"],
                            "development_evaluation_date_count": summary["evaluation_date_count"],
                            "development_ic_date_count": summary["ic_date_count"],
                            "development_selected": False,
                            "validation_opened": False,
                            "validation_confirmed": False,
                            "validation_gate_failures": "not_opened",
                        }
                    )
        prediction_frames[session] = session_frame

    selection = pd.DataFrame.from_records(selection_rows)
    for _session, group in selection.groupby("session", sort=False):
        eligible = group.loc[group["development_eligible"]].copy()
        if eligible.empty:
            continue
        maximum_ic = float(eligible["development_daily_ic"].max())
        contenders = eligible.loc[eligible["development_daily_ic"].ge(maximum_ic - 0.005)].copy()
        priority = {model_id: index for index, model_id in enumerate(models)}
        contenders["_priority"] = contenders["model_id"].map(priority)
        selected_index = contenders.sort_values(["_priority", "model_id"], kind="mergesort").index[
            0
        ]
        selection.loc[selected_index, "development_selected"] = True

    metric_frame = pd.DataFrame.from_records(metrics)
    daily_frame = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    coefficient_frame = (
        pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    )
    return metric_frame, daily_frame, coefficient_frame, selection, prediction_frames


def _score_validation(
    *,
    branch: str,
    development_frames: Mapping[str, pd.DataFrame],
    validation_frame: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    metric_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    scored_validation: dict[str, pd.DataFrame] = {}
    updated = selection.copy()
    if updated["validation_opened"].fillna(False).astype(bool).any():
        raise PermissionError(f"{branch} Validation capability was already consumed")
    anchor = MODEL_ANCHOR[branch]

    selected_rows = updated.loc[updated["development_selected"]]
    for row in selected_rows.itertuples():
        session = row.session
        selected_model = row.model_id
        development = development_frames[session]
        validation = (
            validation_frame.loc[validation_frame["entry_session"].eq(session)]
            .copy()
            .reset_index(drop=True)
        )
        if validation.empty:
            raise ValueError(f"selected {branch}/{session} has no Validation rows")
        requires_anchor = selected_model in PAIRED_REQUIRED[branch]
        scored_models = (
            tuple(dict.fromkeys((anchor, selected_model))) if requires_anchor else (selected_model,)
        )
        for model_id in scored_models:
            prediction_column = _prediction_column("DIRECTIONAL", model_id)
            prediction, coefficients = fit_development_predict_validation(
                development,
                validation,
                features=model_features(branch, model_id, session),
                target=_target_for(branch, "DIRECTIONAL"),
            )
            validation[prediction_column] = prediction
            coefficients = coefficients.assign(
                branch=branch,
                session=session,
                model_id=model_id,
                target_role="DIRECTIONAL",
                partition="Validation",
            )
            coefficient_frames.append(coefficients)

        anchor_column = _prediction_column("DIRECTIONAL", anchor) if requires_anchor else None
        for model_id in scored_models:
            prediction_column = _prediction_column("DIRECTIONAL", model_id)
            comparator_column = prediction_column if model_id == anchor else anchor_column
            summary, daily = summarize_predictions(
                validation,
                prediction_column=prediction_column,
                target_column=_target_for(branch, "DIRECTIONAL"),
                anchor_prediction_column=comparator_column,
                seed=BOOTSTRAP_SEED + 10_000,
            )
            development_ic = float(
                updated.loc[
                    updated["session"].eq(session) & updated["model_id"].eq(model_id),
                    "development_daily_ic",
                ].iloc[0]
            )
            summary.update(
                {
                    "branch": branch,
                    "session": session,
                    "model_id": model_id,
                    "target_role": "DIRECTIONAL",
                    "target_column": _target_for(branch, "DIRECTIONAL"),
                    "partition": "Validation",
                    "fold_count": 0,
                    "feature_count": len(model_features(branch, model_id, session)),
                    "development_daily_ic": development_ic,
                    "ic_retention": (
                        abs(summary["daily_ic_mean"]) / abs(development_ic)
                        if np.isfinite(development_ic) and development_ic != 0
                        else math.nan
                    ),
                    "validation_role": (
                        "frozen_selection" if model_id == selected_model else "required_anchor"
                    ),
                }
            )
            metric_rows.append(summary)
            daily_frames.append(
                daily.assign(
                    branch=branch,
                    session=session,
                    model_id=model_id,
                    target_role="DIRECTIONAL",
                    partition="Validation",
                )
            )

        selected_summary = next(
            item
            for item in metric_rows
            if item["session"] == session and item["model_id"] == selected_model
        )
        failures = validation_gate_failures(
            selected_summary,
            branch=branch,
            requires_paired=selected_model in PAIRED_REQUIRED[branch],
            development_ic=float(row.development_daily_ic),
        )
        mask = updated["session"].eq(session) & updated["model_id"].eq(selected_model)
        updated.loc[mask, "validation_opened"] = True
        updated.loc[mask, "validation_confirmed"] = not failures
        updated.loc[mask, "validation_gate_failures"] = "|".join(failures)
        scored_validation[session] = validation

    return (
        pd.DataFrame.from_records(metric_rows),
        pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        (
            pd.concat(coefficient_frames, ignore_index=True)
            if coefficient_frames
            else pd.DataFrame()
        ),
        updated,
        scored_validation,
    )


def _assessment_dates(folds: Sequence[DateFold]) -> tuple[pd.Timestamp, ...]:
    return tuple(
        sorted(
            {pd.Timestamp(value).normalize() for fold in folds for value in fold.assessment_dates}
        )
    )


def _fit_policy_thresholds(
    *,
    branch: str,
    prediction_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, frame in prediction_frames.items():
        for model_id in MODEL_ORDER[branch]:
            column = _prediction_column("DIRECTIONAL", model_id)
            values = pd.to_numeric(frame[column], errors="coerce")
            finite = values.loc[np.isfinite(values)]
            low = math.nan
            high = math.nan
            enabled = False
            reason = "no_finite_development_oof_predictions"
            if len(finite):
                low = noninterpolated_quantile(finite, 0.10)
                high = noninterpolated_quantile(finite, 0.90)
                enabled = bool(low < 0 < high and low < high)
                reason = "enabled" if enabled else "polarity_guard_failed"
            rows.append(
                {
                    "branch": branch,
                    "session": session,
                    "model_id": model_id,
                    "development_oof_prediction_count": int(len(finite)),
                    "p10": low,
                    "p90": high,
                    "enabled": enabled,
                    "reason": reason,
                }
            )
    return pd.DataFrame.from_records(rows)


def _policy_candidates_for_model(
    *,
    branch: str,
    model_id: str,
    frames: Mapping[str, pd.DataFrame],
    thresholds: pd.DataFrame,
    selected_sessions: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    for session, frame in frames.items():
        source_model = selected_sessions.get(session) if selected_sessions is not None else model_id
        if source_model is None:
            continue
        threshold = thresholds.loc[
            thresholds["session"].eq(session) & thresholds["model_id"].eq(source_model)
        ]
        if len(threshold) != 1 or not bool(threshold.iloc[0]["enabled"]):
            continue
        prediction_column = _prediction_column("DIRECTIONAL", source_model)
        if prediction_column not in frame:
            continue
        candidate = make_policy_candidates(
            frame,
            branch=branch,
            prediction_column=prediction_column,
            low_threshold=float(threshold.iloc[0]["p10"]),
            high_threshold=float(threshold.iloc[0]["p90"]),
        )
        if candidate.empty:
            continue
        candidate["branch"] = branch
        candidate["session"] = session
        candidate["source_model_id"] = source_model
        candidate["policy_model_id"] = model_id
        candidates.append(candidate)
    return pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()


def _summarize_policy_scenarios(
    trades: pd.DataFrame,
    *,
    branch: str,
    policy_model_id: str,
    partition: str,
    eligible_dates: Iterable[Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, cost in (
        ("frictionless", 0.0),
        ("base", BASE_COST_TICKS),
        ("pessimistic", PESSIMISTIC_COST_TICKS),
    ):
        summary = summarize_policy(
            trades,
            eligible_dates=eligible_dates,
            round_trip_cost_ticks=cost,
            seed=BOOTSTRAP_SEED + int(round(cost * 100)),
        )
        summary.update(
            {
                "branch": branch,
                "model_id": policy_model_id,
                "partition": partition,
                "cost_scenario": name,
                "round_trip_cost_ticks": cost,
            }
        )
        rows.append(summary)
    return pd.DataFrame.from_records(rows)


def _development_policy_diagnostics(
    *,
    branch: str,
    prediction_frames: Mapping[str, pd.DataFrame],
    folds_by_session: Mapping[str, Sequence[DateFold]],
    thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    for model_id in MODEL_ORDER[branch]:
        candidate = _policy_candidates_for_model(
            branch=branch,
            model_id=model_id,
            frames=prediction_frames,
            thresholds=thresholds,
        )
        sequenced = sequence_fixed_horizon(candidate) if not candidate.empty else candidate
        eligible_dates = sorted(
            {
                value
                for session in SESSIONS
                for value in _assessment_dates(folds_by_session.get(session, ()))
            }
        )
        metrics.append(
            _summarize_policy_scenarios(
                sequenced,
                branch=branch,
                policy_model_id=model_id,
                partition="Development OOF",
                eligible_dates=eligible_dates,
            )
        )
        if not sequenced.empty:
            sequenced = sequenced.assign(partition="Development OOF")
            sequenced["net_ticks_base"] = sequenced["gross_ticks"] - BASE_COST_TICKS
            sequenced["net_ticks_pessimistic"] = sequenced["gross_ticks"] - PESSIMISTIC_COST_TICKS
            trades.append(sequenced)
    return (
        pd.concat(metrics, ignore_index=True),
        pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
    )


def _selected_session_models(selection: pd.DataFrame) -> dict[str, str]:
    return {
        row.session: row.model_id
        for row in selection.loc[selection["development_selected"]].itertuples()
    }


def _build_validation_capability(
    *,
    branch: str,
    contract_sha: str,
    selection: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> dict[str, Any]:
    """Freeze the realized Development selection and thresholds before outcomes."""

    selection_columns = (
        "session",
        "model_id",
        "development_eligible",
        "development_selected",
        "development_daily_ic",
        "development_gate_failures",
    )
    threshold_columns = (
        "session",
        "model_id",
        "development_oof_prediction_count",
        "p10",
        "p90",
        "enabled",
        "reason",
    )
    selection_records = json.loads(
        selection.loc[:, selection_columns]
        .sort_values(["session", "model_id"], kind="mergesort")
        .to_json(orient="records", date_format="iso")
    )
    threshold_records = json.loads(
        thresholds.loc[:, threshold_columns]
        .sort_values(["session", "model_id"], kind="mergesort")
        .to_json(orient="records", date_format="iso")
    )
    selected = _selected_session_models(selection)
    authorized_cells: list[dict[str, Any]] = []
    for session, model_id in sorted(selected.items()):
        threshold = thresholds.loc[
            thresholds["session"].eq(session) & thresholds["model_id"].eq(model_id)
        ]
        if len(threshold) != 1:
            raise ValueError(f"missing unique frozen threshold for {branch}/{session}/{model_id}")
        row = threshold.iloc[0]
        authorized_cells.append(
            {
                "session": session,
                "model_id": model_id,
                "p10": _json_safe(row["p10"]),
                "p90": _json_safe(row["p90"]),
                "enabled": bool(row["enabled"]),
                "reason": str(row["reason"]),
            }
        )
    payload: dict[str, Any] = {
        "research_id": RESEARCH_ID,
        "branch": branch,
        "contract_sha256": contract_sha,
        "selection_sha256": canonical_sha256(selection_records),
        "thresholds_sha256": canonical_sha256(threshold_records),
        "selection_records": selection_records,
        "threshold_records": threshold_records,
        "authorized_validation_cells": authorized_cells,
        "single_locked_batch": True,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _confirmed_session_models(selection: pd.DataFrame) -> dict[str, str]:
    return {
        row.session: row.model_id
        for row in selection.loc[
            selection["development_selected"] & selection["validation_confirmed"]
        ].itertuples()
    }


def _eligible_dates_for_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    branch: str,
    selected_sessions: Mapping[str, str],
) -> tuple[pd.Timestamp, ...]:
    """Return exact finite prediction/target dates for the frozen session union."""

    target_column = _target_for(branch, "DIRECTIONAL")
    dates: set[pd.Timestamp] = set()
    for session, model_id in selected_sessions.items():
        frame = frames[session]
        prediction_column = _prediction_column("DIRECTIONAL", model_id)
        if prediction_column not in frame or target_column not in frame:
            raise ValueError(
                f"missing selected evaluation columns for {branch}/{session}/{model_id}"
            )
        prediction = pd.to_numeric(frame[prediction_column], errors="coerce")
        target = pd.to_numeric(frame[target_column], errors="coerce")
        finite = np.isfinite(prediction) & np.isfinite(target)
        dates.update(
            pd.to_datetime(frame.loc[finite, "trade_date_ny"]).dt.tz_localize(None).dt.normalize()
        )
    return tuple(sorted(dates))


def _selected_policy(
    *,
    branch: str,
    partition: str,
    frames: Mapping[str, pd.DataFrame],
    selection: pd.DataFrame,
    thresholds: pd.DataFrame,
    eligible_dates: Iterable[Any],
    confirmed_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = (
        _confirmed_session_models(selection)
        if confirmed_only
        else _selected_session_models(selection)
    )
    if not selected:
        empty_metrics = _summarize_policy_scenarios(
            pd.DataFrame(),
            branch=branch,
            policy_model_id="FROZEN_SELECTION_UNION",
            partition=partition,
            eligible_dates=eligible_dates,
        )
        return empty_metrics, pd.DataFrame()
    candidate = _policy_candidates_for_model(
        branch=branch,
        model_id="FROZEN_SELECTION_UNION",
        frames=frames,
        thresholds=thresholds,
        selected_sessions=selected,
    )
    sequenced = sequence_fixed_horizon(candidate) if not candidate.empty else candidate
    metrics = _summarize_policy_scenarios(
        sequenced,
        branch=branch,
        policy_model_id="FROZEN_SELECTION_UNION",
        partition=partition,
        eligible_dates=eligible_dates,
    )
    if not sequenced.empty:
        sequenced = sequenced.assign(partition=partition)
        sequenced["net_ticks_base"] = sequenced["gross_ticks"] - BASE_COST_TICKS
        sequenced["net_ticks_pessimistic"] = sequenced["gross_ticks"] - PESSIMISTIC_COST_TICKS
    return metrics, sequenced


def _economic_gate_failures(
    development: Mapping[str, Any], validation: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    for prefix, metric, minimum_trades, minimum_dates in (
        ("development", development, 300, 100),
        ("validation", validation, 200, 60),
    ):
        if int(metric.get("trade_count", 0)) < minimum_trades:
            failures.append(f"{prefix}_trade_floor")
        if int(metric.get("trading_date_count", 0)) < minimum_dates:
            failures.append(f"{prefix}_date_floor")
        if not float(metric.get("mean_net_ticks", math.nan)) > 0:
            failures.append(f"{prefix}_mean_net_ticks")
        if not float(metric.get("net_ticks_ci_low", math.nan)) > 0:
            failures.append(f"{prefix}_mean_net_interval")
        if not float(metric.get("daily_net_ticks_sharpe", math.nan)) > 0:
            failures.append(f"{prefix}_daily_sharpe")
        concentration = float(metric.get("best_ten_date_pnl_share", math.nan))
        if not np.isfinite(concentration) or concentration > 0.50:
            failures.append(f"{prefix}_pnl_concentration")
    return failures


def _branch_state(
    *,
    selection: pd.DataFrame,
    policy_metrics: pd.DataFrame,
) -> tuple[str, list[str]]:
    selected = selection.loc[selection["development_selected"]]
    if selected.empty:
        return "FROZEN_NO_DIRECTIONAL_MODEL", ["no_development_eligible_model"]
    confirmed = selected.loc[selected["validation_confirmed"]]
    if confirmed.empty:
        failures = sorted(
            {
                failure
                for value in selected["validation_gate_failures"]
                for failure in str(value).split("|")
                if failure
            }
        )
        return "NO_TRADING_SYSTEM_AUTHORIZED", failures
    base = policy_metrics.loc[
        policy_metrics["model_id"].eq("FROZEN_SELECTION_UNION")
        & policy_metrics["cost_scenario"].eq("base")
    ]
    development = base.loc[base["partition"].eq("Development OOF")]
    validation = base.loc[base["partition"].eq("Validation")]
    if len(development) != 1 or len(validation) != 1:
        return "PREDICTIVE_ONLY_NOT_ECONOMIC", ["selected_policy_metrics_missing"]
    failures = _economic_gate_failures(development.iloc[0].to_dict(), validation.iloc[0].to_dict())
    if failures:
        return "PREDICTIVE_ONLY_NOT_ECONOMIC", failures
    return "RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA", []


def _folds_for_frame(frame: pd.DataFrame) -> dict[str, list[DateFold]]:
    return {
        session: build_exact_date_folds(
            frame.loc[frame["entry_session"].eq(session), "trade_date_ny"]
        )
        for session in SESSIONS
    }


def _fold_summary(folds_by_branch: Mapping[str, Mapping[str, Sequence[DateFold]]]) -> pd.DataFrame:
    frames = [
        folds_to_frame(folds, branch=branch, session=session)
        for branch, by_session in folds_by_branch.items()
        for session, folds in by_session.items()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _population_rows(
    *,
    branch: str,
    partition: str,
    frame: pd.DataFrame,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in SESSIONS:
        sample = frame.loc[frame["entry_session"].eq(session)]
        rows.append(
            {
                "branch": branch,
                "partition": partition,
                "session": session,
                "stage": stage,
                "rows": int(len(sample)),
                "trading_dates": int(
                    pd.to_datetime(sample["trade_date_ny"]).dt.normalize().nunique()
                ),
                "minimum_trade_date": (
                    pd.to_datetime(sample["trade_date_ny"]).min() if len(sample) else pd.NaT
                ),
                "maximum_trade_date": (
                    pd.to_datetime(sample["trade_date_ny"]).max() if len(sample) else pd.NaT
                ),
            }
        )
    return rows


def _persist_frames(output_dir: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        if len(frame.columns) == 0:
            raise ValueError(f"artifact {name} has no frozen schema")
        frame.to_parquet(output_dir / f"{name}.parquet", index=False, engine="pyarrow")


def _publish_directories(replacements: Sequence[tuple[Path, Path]]) -> None:
    published: list[tuple[Path, Path | None]] = []
    try:
        for staging, target in replacements:
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if target.exists():
                backup = target.with_name(f".{target.name}.backup-{uuid4().hex}")
                target.replace(backup)
            try:
                staging.replace(target)
            except Exception:
                if backup is not None and backup.exists():
                    backup.replace(target)
                raise
            published.append((target, backup))
    except Exception:
        for target, backup in reversed(published):
            if target.exists():
                shutil.rmtree(target)
            if backup is not None and backup.exists():
                backup.replace(target)
        raise
    for _, backup in published:
        if backup is not None and backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError:
                # The coherent targets are already committed; a Windows lock on an
                # obsolete backup must not turn a successful publication into a
                # misleading failed run.
                continue


def _write_blocked_artifacts(
    output_dir: Path,
    report_dir: Path,
    *,
    contract_sha: str,
    error: Exception,
    final_test_opened: bool = False,
    access_record: Mapping[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    headline = {
        "research_id": RESEARCH_ID,
        "status": "BLOCKED",
        "contract_sha256": contract_sha,
        "branch_states": {},
        "branch_failures": {"INTEGRITY": [type(error).__name__]},
        "overall_state": "BLOCKED_INTEGRITY_FAILURE",
        "final_test_opened": bool(final_test_opened),
        "final_test_access_state": (
            "OPENED_BY_INTEGRITY_VIOLATION" if final_test_opened else "NOT_OPENED"
        ),
        "mgc_test_run": False,
    }
    _write_json(output_dir / "headline.json", headline)
    _write_json(
        output_dir / "integrity_failure.json",
        {
            "research_id": RESEARCH_ID,
            "state": "BLOCKED_INTEGRITY_FAILURE",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "final_test_opened": bool(final_test_opened),
        },
    )
    if access_record is not None:
        _write_json(
            output_dir / "blocked_access_audit.json",
            {
                "research_id": RESEARCH_ID,
                "records": [dict(access_record)],
                "outcome_values_reported": False,
            },
        )
    boundary_line = (
        "A protected 2025+ row was materialized and immediately rejected; no outcome "
        "values or distributions are reported."
        if final_test_opened
        else "The 2025+ Final-test partition was not opened."
    )
    (report_dir / "status.md").write_text(
        "# POI Feature Combination v1 Status\n\n"
        "STATUS: BLOCKED\n\n"
        "Overall state: `BLOCKED_INTEGRITY_FAILURE`\n\n"
        f"{boundary_line}\n",
        encoding="utf-8",
    )


def _overall_state(branch_states: Mapping[str, str]) -> str:
    states = set(branch_states.values())
    if "RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA" in states:
        return "RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA"
    if "PREDICTIVE_ONLY_NOT_ECONOMIC" in states:
        return "PREDICTIVE_ONLY_NOT_ECONOMIC"
    if states == {"FROZEN_NO_DIRECTIONAL_MODEL"}:
        return "FROZEN_NO_DIRECTIONAL_MODEL"
    return "NO_TRADING_SYSTEM_AUTHORIZED"


def _run_feature_combination_research_into(
    project_root: Path,
    *,
    output_dir: Path,
    report_dir: Path,
) -> CombinationResearchResult:
    """Run the frozen POI-first study without accessing 2025+ outcomes."""

    project_root = project_root.resolve()
    contract_path = project_root / CONTRACT_RELATIVE_PATH
    contract_sha = sha256_file(contract_path)
    if contract_sha != EXPECTED_CONTRACT_SHA256:
        raise PermissionError(
            "feature-combination contract hash differs from the frozen checkpoint"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    access_rows: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    join_audits: list[dict[str, Any]] = []

    provenance_paths = (
        POI_PATH,
        POI_REGISTRY_PATH,
        ELIGIBLE_PATH,
        LABEL_PATH,
        FEATURE_PATH,
        FROZEN_EXPANSION_PATH,
        BRANCH_B_REGISTRY_PATH,
        FES_DEVELOPMENT_PATH,
        FES_VALIDATION_PATH,
        FES_REGISTRY_PATH,
        FES_HASH_MANIFEST_PATH,
        FES_SELECTION_PATH,
        FES_INPUT_MANIFEST_PATH,
        FES_FINAL_GATES_PATH,
    )
    provenance = _input_provenance(project_root, provenance_paths)
    _validate_frozen_feature_sets(project_root)

    development_predictors, development_eligible = _load_predictor_base(
        project_root,
        partition="Development",
        audit=access_rows,
    )
    poi_development_candidates, poi_development_join = _load_poi_candidates(
        project_root,
        partition="Development",
        eligible=development_eligible,
        audit=access_rows,
    )
    join_audits.append(poi_development_join)
    population_rows.extend(
        _population_rows(
            branch="GENERAL",
            partition="Development",
            frame=development_predictors,
            stage="outcome_free_predictor_population",
        )
    )
    population_rows.extend(
        _population_rows(
            branch="POI",
            partition="Development",
            frame=poi_development_candidates,
            stage="outcome_free_deduplicated_population",
        )
    )

    folds_by_branch = {
        "POI": _folds_for_frame(poi_development_candidates),
        "GENERAL": _folds_for_frame(development_predictors),
    }
    fold_payload = fold_manifest_payload(
        {
            (branch, session): folds
            for branch, by_session in folds_by_branch.items()
            for session, folds in by_session.items()
        }
    )
    _write_json(output_dir / "pre_outcome_fold_manifest.json", fold_payload)
    fold_summary = _fold_summary(folds_by_branch)
    fold_summary.to_parquet(output_dir / "fold_summary.parquet", index=False, engine="pyarrow")

    poi_development_labels = _load_labels(
        project_root,
        partition="Development",
        audit=access_rows,
        stage="POI:Development:authorized_outcomes",
        observation_ids=poi_development_candidates["observation_id"].tolist(),
    )
    poi_development = _build_poi_model_frame(
        poi_development_candidates,
        development_predictors,
        poi_development_labels,
    )
    population_rows.extend(
        _population_rows(
            branch="POI",
            partition="Development",
            frame=poi_development,
            stage="label_available_model_population",
        )
    )

    (
        poi_metrics,
        poi_daily,
        poi_coefficients,
        poi_selection,
        poi_prediction_frames,
    ) = _analyze_development(
        branch="POI",
        frame=poi_development,
        folds_by_session=folds_by_branch["POI"],
    )
    poi_thresholds = _fit_policy_thresholds(branch="POI", prediction_frames=poi_prediction_frames)
    poi_policy_metrics, poi_policy_trades = _development_policy_diagnostics(
        branch="POI",
        prediction_frames=poi_prediction_frames,
        folds_by_session=folds_by_branch["POI"],
        thresholds=poi_thresholds,
    )
    poi_validation_capability = _build_validation_capability(
        branch="POI",
        contract_sha=contract_sha,
        selection=poi_selection,
        thresholds=poi_thresholds,
    )
    _write_json(
        output_dir / "poi_pre_validation_capability.json",
        poi_validation_capability,
    )
    poi_validation_opened = bool(poi_selection["development_selected"].any())
    if poi_validation_opened:
        poi_selected_sessions = tuple(sorted(_selected_session_models(poi_selection)))
        validation_predictors, validation_eligible = _load_predictor_base(
            project_root,
            partition="Validation",
            audit=access_rows,
            sessions=poi_selected_sessions,
        )
        poi_validation_candidates, poi_validation_join = _load_poi_candidates(
            project_root,
            partition="Validation",
            eligible=validation_eligible,
            audit=access_rows,
            sessions=poi_selected_sessions,
        )
        join_audits.append(poi_validation_join)
        poi_validation_labels = _load_labels(
            project_root,
            partition="Validation",
            audit=access_rows,
            stage="POI:Validation:frozen_selection_outcomes",
            observation_ids=poi_validation_candidates["observation_id"].tolist(),
        )
        poi_validation = _build_poi_model_frame(
            poi_validation_candidates,
            validation_predictors,
            poi_validation_labels,
        )
        population_rows.extend(
            _population_rows(
                branch="POI",
                partition="Validation",
                frame=poi_validation,
                stage="label_available_frozen_selection_population",
            )
        )
        (
            poi_validation_metrics,
            poi_validation_daily,
            poi_validation_coefficients,
            poi_selection,
            poi_validation_prediction_frames,
        ) = _score_validation(
            branch="POI",
            development_frames=poi_prediction_frames,
            validation_frame=poi_validation,
            selection=poi_selection,
        )
        poi_metrics = pd.concat([poi_metrics, poi_validation_metrics], ignore_index=True)
        poi_daily = pd.concat([poi_daily, poi_validation_daily], ignore_index=True)
        poi_coefficients = pd.concat(
            [poi_coefficients, poi_validation_coefficients], ignore_index=True
        )
        poi_confirmed_models = _confirmed_session_models(poi_selection)
        validation_dates = _eligible_dates_for_frames(
            poi_validation_prediction_frames,
            branch="POI",
            selected_sessions=poi_confirmed_models,
        )
        poi_selected_validation_metrics, poi_selected_validation_trades = _selected_policy(
            branch="POI",
            partition="Validation",
            frames=poi_validation_prediction_frames,
            selection=poi_selection,
            thresholds=poi_thresholds,
            eligible_dates=validation_dates,
            confirmed_only=True,
        )
        poi_policy_metrics = pd.concat(
            [poi_policy_metrics, poi_selected_validation_metrics], ignore_index=True
        )
        if not poi_selected_validation_trades.empty:
            poi_policy_trades = pd.concat(
                [poi_policy_trades, poi_selected_validation_trades], ignore_index=True
            )

    poi_confirmed_models = _confirmed_session_models(poi_selection)
    poi_development_dates = _eligible_dates_for_frames(
        poi_prediction_frames,
        branch="POI",
        selected_sessions=poi_confirmed_models,
    )
    poi_selected_dev_metrics, poi_selected_dev_trades = _selected_policy(
        branch="POI",
        partition="Development OOF",
        frames=poi_prediction_frames,
        selection=poi_selection,
        thresholds=poi_thresholds,
        eligible_dates=poi_development_dates,
        confirmed_only=True,
    )
    poi_policy_metrics = pd.concat(
        [poi_policy_metrics, poi_selected_dev_metrics], ignore_index=True
    )
    if not poi_selected_dev_trades.empty:
        poi_policy_trades = pd.concat(
            [poi_policy_trades, poi_selected_dev_trades], ignore_index=True
        )

    poi_state, poi_state_failures = _branch_state(
        selection=poi_selection,
        policy_metrics=poi_policy_metrics,
    )
    poi_checkpoint = {
        "research_id": RESEARCH_ID,
        "branch": "POI",
        "contract_sha256": contract_sha,
        "state": poi_state,
        "failures": poi_state_failures,
        "development_selected": poi_selection.loc[
            poi_selection["development_selected"], ["session", "model_id"]
        ].to_dict("records"),
        "pre_validation_capability_sha256": poi_validation_capability["sha256"],
        "validation_opened": poi_validation_opened,
        "general_branch_may_open": (poi_state != "RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA"),
    }
    _write_json(output_dir / "poi_branch_checkpoint.json", poi_checkpoint)

    metric_frames = [poi_metrics]
    daily_frames = [poi_daily]
    coefficient_frames = [poi_coefficients]
    selection_frames = [poi_selection]
    threshold_frames = [poi_thresholds]
    policy_metric_frames = [poi_policy_metrics]
    policy_trade_frames = [poi_policy_trades]
    branch_states = {"POI": poi_state}
    branch_failures = {"POI": poi_state_failures}
    general_opened = poi_checkpoint["general_branch_may_open"]

    if general_opened:
        general_development_labels = _load_labels(
            project_root,
            partition="Development",
            audit=access_rows,
            stage="GENERAL:Development:authorized_outcomes_after_poi_checkpoint",
        )
        general_development = _build_general_model_frame(
            development_predictors, general_development_labels
        )
        population_rows.extend(
            _population_rows(
                branch="GENERAL",
                partition="Development",
                frame=general_development,
                stage="label_available_model_population",
            )
        )
        (
            general_metrics,
            general_daily,
            general_coefficients,
            general_selection,
            general_prediction_frames,
        ) = _analyze_development(
            branch="GENERAL",
            frame=general_development,
            folds_by_session=folds_by_branch["GENERAL"],
        )
        general_thresholds = _fit_policy_thresholds(
            branch="GENERAL", prediction_frames=general_prediction_frames
        )
        general_policy_metrics, general_policy_trades = _development_policy_diagnostics(
            branch="GENERAL",
            prediction_frames=general_prediction_frames,
            folds_by_session=folds_by_branch["GENERAL"],
            thresholds=general_thresholds,
        )
        general_validation_capability = _build_validation_capability(
            branch="GENERAL",
            contract_sha=contract_sha,
            selection=general_selection,
            thresholds=general_thresholds,
        )
        _write_json(
            output_dir / "general_pre_validation_capability.json",
            general_validation_capability,
        )
        general_validation_opened = bool(general_selection["development_selected"].any())
        if general_validation_opened:
            general_selected_sessions = tuple(sorted(_selected_session_models(general_selection)))
            general_validation_predictors, _ = _load_predictor_base(
                project_root,
                partition="Validation",
                audit=access_rows,
                sessions=general_selected_sessions,
            )
            general_validation_labels = _load_labels(
                project_root,
                partition="Validation",
                audit=access_rows,
                stage="GENERAL:Validation:frozen_selection_outcomes",
                observation_ids=general_validation_predictors["observation_id"].tolist(),
            )
            general_validation = _build_general_model_frame(
                general_validation_predictors, general_validation_labels
            )
            population_rows.extend(
                _population_rows(
                    branch="GENERAL",
                    partition="Validation",
                    frame=general_validation,
                    stage="label_available_frozen_selection_population",
                )
            )
            (
                general_validation_metrics,
                general_validation_daily,
                general_validation_coefficients,
                general_selection,
                general_validation_prediction_frames,
            ) = _score_validation(
                branch="GENERAL",
                development_frames=general_prediction_frames,
                validation_frame=general_validation,
                selection=general_selection,
            )
            general_metrics = pd.concat(
                [general_metrics, general_validation_metrics], ignore_index=True
            )
            general_daily = pd.concat([general_daily, general_validation_daily], ignore_index=True)
            general_coefficients = pd.concat(
                [general_coefficients, general_validation_coefficients],
                ignore_index=True,
            )
            general_confirmed_models = _confirmed_session_models(general_selection)
            validation_dates = _eligible_dates_for_frames(
                general_validation_prediction_frames,
                branch="GENERAL",
                selected_sessions=general_confirmed_models,
            )
            general_selected_validation_metrics, general_selected_validation_trades = (
                _selected_policy(
                    branch="GENERAL",
                    partition="Validation",
                    frames=general_validation_prediction_frames,
                    selection=general_selection,
                    thresholds=general_thresholds,
                    eligible_dates=validation_dates,
                    confirmed_only=True,
                )
            )
            general_policy_metrics = pd.concat(
                [general_policy_metrics, general_selected_validation_metrics],
                ignore_index=True,
            )
            if not general_selected_validation_trades.empty:
                general_policy_trades = pd.concat(
                    [general_policy_trades, general_selected_validation_trades],
                    ignore_index=True,
                )

        general_confirmed_models = _confirmed_session_models(general_selection)
        general_development_dates = _eligible_dates_for_frames(
            general_prediction_frames,
            branch="GENERAL",
            selected_sessions=general_confirmed_models,
        )
        general_selected_dev_metrics, general_selected_dev_trades = _selected_policy(
            branch="GENERAL",
            partition="Development OOF",
            frames=general_prediction_frames,
            selection=general_selection,
            thresholds=general_thresholds,
            eligible_dates=general_development_dates,
            confirmed_only=True,
        )
        general_policy_metrics = pd.concat(
            [general_policy_metrics, general_selected_dev_metrics], ignore_index=True
        )
        if not general_selected_dev_trades.empty:
            general_policy_trades = pd.concat(
                [general_policy_trades, general_selected_dev_trades], ignore_index=True
            )

        general_state, general_state_failures = _branch_state(
            selection=general_selection,
            policy_metrics=general_policy_metrics,
        )
        general_checkpoint = {
            "research_id": RESEARCH_ID,
            "branch": "GENERAL",
            "contract_sha256": contract_sha,
            "poi_checkpoint_sha256": canonical_sha256(poi_checkpoint),
            "state": general_state,
            "failures": general_state_failures,
            "development_selected": general_selection.loc[
                general_selection["development_selected"], ["session", "model_id"]
            ].to_dict("records"),
            "pre_validation_capability_sha256": general_validation_capability["sha256"],
            "validation_opened": general_validation_opened,
        }
        _write_json(output_dir / "general_branch_checkpoint.json", general_checkpoint)
        metric_frames.append(general_metrics)
        daily_frames.append(general_daily)
        coefficient_frames.append(general_coefficients)
        selection_frames.append(general_selection)
        threshold_frames.append(general_thresholds)
        policy_metric_frames.append(general_policy_metrics)
        policy_trade_frames.append(general_policy_trades)
        branch_states["GENERAL"] = general_state
        branch_failures["GENERAL"] = general_state_failures

    population_summary = pd.DataFrame.from_records(population_rows)
    model_metrics = pd.concat(metric_frames, ignore_index=True)
    daily_ic = pd.concat(daily_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    selection_verdicts = pd.concat(selection_frames, ignore_index=True)
    policy_thresholds = pd.concat(threshold_frames, ignore_index=True)
    policy_metrics = pd.concat(policy_metric_frames, ignore_index=True)
    nonempty_trades = [frame for frame in policy_trade_frames if not frame.empty]
    policy_trades = (
        pd.concat(nonempty_trades, ignore_index=True) if nonempty_trades else pd.DataFrame()
    )
    policy_trades = policy_trades.reindex(columns=POLICY_TRADE_COLUMNS)
    access_audit = pd.DataFrame.from_records(access_rows)
    overall_state = _overall_state(branch_states)
    headline = {
        "research_id": RESEARCH_ID,
        "status": "READY",
        "contract_sha256": contract_sha,
        "fold_manifest_sha256": fold_payload["sha256"],
        "branch_states": branch_states,
        "branch_failures": branch_failures,
        "overall_state": overall_state,
        "poi_first_completed_before_general": True,
        "general_branch_opened": bool(general_opened),
        "final_test_opened": False,
        "mgc_test_run": False,
        "mgc_execution_recommendation": (
            "No MGC execution test is authorized. If future GC evidence passes, use "
            "synchronized MGC next-bar-open entry with telemetry-measured costs."
        ),
        "claim_boundary": (
            "Retrospective research only; 2024 reuses evidence and cannot authorize "
            "live GC or MGC trading."
        ),
    }

    frames = {
        "population_summary": population_summary,
        "fold_summary": fold_summary,
        "model_metrics": model_metrics,
        "daily_ic": daily_ic,
        "coefficients": coefficients,
        "selection_verdicts": selection_verdicts,
        "policy_thresholds": policy_thresholds,
        "policy_metrics": policy_metrics,
        "policy_trades": policy_trades,
        "access_audit": access_audit,
        "poi_join_audit": pd.DataFrame.from_records(join_audits),
    }
    _persist_frames(output_dir, frames)
    _write_json(output_dir / "headline.json", headline)
    _write_json(
        output_dir / "input_provenance.json",
        {
            "research_id": RESEARCH_ID,
            "contract_sha256": contract_sha,
            "inputs": provenance,
            "authorized_access_rows": access_rows,
        },
    )
    summary_lines = [
        "# POI Feature Combination v1 Status",
        "",
        "STATUS: READY",
        "",
        f"Overall state: `{overall_state}`",
        "",
        f"POI branch: `{branch_states['POI']}`",
    ]
    if "GENERAL" in branch_states:
        summary_lines.extend(["", f"General branch: `{branch_states['GENERAL']}`"])
    summary_lines.extend(
        [
            "",
            "The 2025+ Final-test partition was not opened.",
            "No live GC or MGC trading system is authorized by this study.",
        ]
    )
    (report_dir / "status.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return CombinationResearchResult(
        population_summary=population_summary,
        fold_summary=fold_summary,
        model_metrics=model_metrics,
        daily_ic=daily_ic,
        coefficients=coefficients,
        selection_verdicts=selection_verdicts,
        policy_thresholds=policy_thresholds,
        policy_metrics=policy_metrics,
        policy_trades=policy_trades,
        access_audit=access_audit,
        headline=headline,
        output_dir=output_dir,
    )


def run_feature_combination_research(
    root: Path | None = None,
) -> CombinationResearchResult:
    """Run into fresh staging directories and publish one coherent artifact set."""

    project_root = (root or Path(__file__).resolve().parents[2]).resolve()
    output_target = project_root / OUTPUT_RELATIVE_DIR
    report_target = project_root / REPORT_RELATIVE_DIR
    output_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    staging_output = Path(tempfile.mkdtemp(prefix=".v1-staging-", dir=output_target.parent))
    staging_report = Path(tempfile.mkdtemp(prefix=".v1-staging-", dir=report_target.parent))
    contract_path = project_root / CONTRACT_RELATIVE_PATH
    contract_sha = sha256_file(contract_path) if contract_path.exists() else "MISSING"
    try:
        result = _run_feature_combination_research_into(
            project_root,
            output_dir=staging_output,
            report_dir=staging_report,
        )
    except Exception as error:
        shutil.rmtree(staging_output, ignore_errors=True)
        shutil.rmtree(staging_report, ignore_errors=True)
        staging_output.mkdir(parents=True, exist_ok=False)
        staging_report.mkdir(parents=True, exist_ok=False)
        _write_blocked_artifacts(
            staging_output,
            staging_report,
            contract_sha=contract_sha,
            error=error,
            final_test_opened=(
                error.final_test_opened if isinstance(error, BoundaryAccessError) else False
            ),
            access_record=(error.access_record if isinstance(error, BoundaryAccessError) else None),
        )
        _publish_directories(((staging_output, output_target), (staging_report, report_target)))
        raise
    _publish_directories(((staging_output, output_target), (staging_report, report_target)))
    result.output_dir = output_target
    return result
