"""Outcome-free Stage 1 validation and fold-freezing helpers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .tsay_feature_engineering import (
    TSAY_CONTEXT_COLUMNS,
    TSAY_MATRIX_COLUMNS,
    TSAY_METADATA_COLUMNS,
)
from .tsay_feature_registry import (
    D1_INPUTS_BY_SESSION,
    O1_INPUTS_BY_SESSION,
    SESSIONS,
    TSAY_FEATURE_COLUMNS,
    TSAY_FEATURE_SPECS,
    TSAY_LOGICAL_TO_PHYSICAL,
    TSAY_PRIMARY_TEST_COUNT,
)


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TsayDateFold:
    fold_id: int
    train_dates: tuple[str, ...]
    embargo_date: str
    assessment_dates: tuple[str, ...]
    fold_sha256: str


def build_exact_date_folds(
    dates: Iterable[object],
    *,
    initial_training_dates: int,
    assessment_dates: int,
    step_dates: int,
) -> tuple[TsayDateFold, ...]:
    """Build complete expanding folds with the frozen one-date embargo."""

    normalized = (
        pd.DatetimeIndex(pd.to_datetime(list(dates), errors="raise"))
        .normalize()
        .unique()
        .sort_values()
    )
    if initial_training_dates <= 0 or assessment_dates <= 0 or step_dates <= 0:
        raise ValueError("Fold sizes must be positive.")
    folds: list[TsayDateFold] = []
    train_count = int(initial_training_dates)
    while train_count + 1 + assessment_dates <= len(normalized):
        train = tuple(pd.Timestamp(value).date().isoformat() for value in normalized[:train_count])
        embargo = pd.Timestamp(normalized[train_count]).date().isoformat()
        assessment = tuple(
            pd.Timestamp(value).date().isoformat()
            for value in normalized[train_count + 1 : train_count + 1 + assessment_dates]
        )
        payload = {
            "train_dates": train,
            "embargo_date": embargo,
            "assessment_dates": assessment,
        }
        folds.append(
            TsayDateFold(
                fold_id=len(folds),
                train_dates=train,
                embargo_date=embargo,
                assessment_dates=assessment,
                fold_sha256=canonical_sha256(payload),
            )
        )
        train_count += int(step_dates)
    return tuple(folds)


def freeze_outer_folds(eligible_metadata: pd.DataFrame) -> dict[str, object]:
    """Freeze the Stage 1 outer schedule using Development date metadata only."""

    required = {"observation_id", "trade_date_ny", "entry_session", "research_partition"}
    missing = sorted(required - set(eligible_metadata.columns))
    if missing:
        raise ValueError(f"Eligible metadata is missing fold columns: {missing}")
    development = eligible_metadata.loc[
        eligible_metadata["research_partition"].astype(str).eq("Development")
    ].copy()
    if development.empty:
        raise ValueError("Development metadata is empty.")

    sessions: dict[str, object] = {}
    for session in SESSIONS:
        dates = development.loc[
            development["entry_session"].astype(str).eq(session), "trade_date_ny"
        ]
        folds = build_exact_date_folds(
            dates,
            initial_training_dates=126,
            assessment_dates=21,
            step_dates=21,
        )
        assessment_union = tuple(
            dict.fromkeys(value for fold in folds for value in fold.assessment_dates)
        )
        attainable = len(assessment_union)
        minimum = math.ceil(0.90 * attainable)
        if minimum > attainable:
            raise ValueError("The minimum OOF coverage gate is arithmetically impossible.")
        sessions[session] = {
            "development_date_count": int(pd.to_datetime(dates).dt.normalize().nunique()),
            "fold_count": len(folds),
            "folds": [asdict(fold) for fold in folds],
            "attainable_dev_oof_dates": attainable,
            "minimum_dev_oof_dates": minimum,
            "structural_evaluability_minimum": 180,
            "structurally_evaluable": attainable >= 180,
            "assessment_date_union": assessment_union,
        }
    if not all(bool(payload["structurally_evaluable"]) for payload in sessions.values()):
        raise ValueError("At least one session cannot attain the frozen 180-date minimum.")
    frozen = {
        "initial_training_dates": 126,
        "embargo_dates": 1,
        "assessment_dates": 21,
        "step_dates": 21,
        "complete_blocks_only": True,
        "sessions": sessions,
    }
    frozen["fold_membership_sha256"] = canonical_sha256(frozen)
    return frozen


def summarize_metadata(frame: pd.DataFrame, *, source: str) -> dict[str, object]:
    """Summarize only authorized identity/date/session metadata."""

    required = {
        "observation_id",
        "decision_timestamp_utc",
        "trade_date_ny",
        "entry_session",
        "research_partition",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing metadata columns: {missing}")
    working = frame.copy()
    working["trade_date_ny"] = pd.to_datetime(working["trade_date_ny"], errors="raise")
    if working["observation_id"].isna().any() or working["observation_id"].duplicated().any():
        raise ValueError(f"{source} observation_id is not a complete unique key.")
    if working["decision_timestamp_utc"].isna().any():
        raise ValueError(f"{source} contains a missing decision timestamp.")
    if working.duplicated(["observation_id", "decision_timestamp_utc"]).any():
        raise ValueError(f"{source} composite identity is not unique.")
    allowed_partitions = {"Development", "Validation"}
    partitions = set(working["research_partition"].astype(str).unique())
    if not partitions <= allowed_partitions:
        raise ValueError(f"{source} returned unauthorized partitions: {sorted(partitions)}")
    rows: list[dict[str, object]] = []
    grouped = working.groupby(["research_partition", "entry_session"], observed=True, sort=True)
    for (partition, session), group in grouped:
        rows.append(
            {
                "research_partition": str(partition),
                "entry_session": str(session),
                "rows": len(group),
                "dates": int(group["trade_date_ny"].dt.normalize().nunique()),
                "minimum_date": group["trade_date_ny"].min().date().isoformat(),
                "maximum_date": group["trade_date_ny"].max().date().isoformat(),
            }
        )
    return {
        "source": source,
        "rows": len(working),
        "unique_observation_ids": int(working["observation_id"].nunique()),
        "unique_decision_timestamps": int(working["decision_timestamp_utc"].nunique()),
        "identity_unique": True,
        "partition_session_counts": rows,
        "metadata_sample_sha256": stable_metadata_sample_hash(working),
    }


def stable_metadata_sample_hash(frame: pd.DataFrame, *, sample_size: int = 32) -> str:
    """Hash deterministic identity/date/session samples without outcome values."""

    columns = [
        column
        for column in (
            "observation_id",
            "decision_timestamp_utc",
            "trade_date_ny",
            "entry_session",
            "research_partition",
            "product",
        )
        if column in frame.columns
    ]
    ordered = frame.sort_values(["observation_id"], kind="mergesort").reset_index(drop=True)
    if len(ordered) <= sample_size:
        positions = list(range(len(ordered)))
    else:
        positions = sorted(
            set(
                round(index * (len(ordered) - 1) / (sample_size - 1))
                for index in range(sample_size)
            )
        )
    records: list[dict[str, object]] = []
    for position in positions:
        record: dict[str, object] = {"position": position}
        for column in columns:
            value = ordered.at[position, column]
            if isinstance(value, pd.Timestamp):
                record[column] = value.isoformat()
            elif pd.isna(value):
                record[column] = None
            else:
                record[column] = str(value)
        records.append(record)
    payload = {
        "row_count": len(ordered),
        "columns": columns,
        "sample_positions": positions,
        "records": records,
    }
    return canonical_sha256(payload)


def verify_metadata_alignment(frames: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    """Prove one-to-one identity/timestamp/session/partition alignment."""

    if not frames:
        raise ValueError("No metadata frames supplied for alignment.")
    names = list(frames)
    reference_name = names[0]
    columns = [
        "observation_id",
        "decision_timestamp_utc",
        "trade_date_ny",
        "entry_session",
        "research_partition",
    ]
    reference = (
        frames[reference_name].loc[:, columns].sort_values("observation_id").reset_index(drop=True)
    )
    checks: list[dict[str, object]] = []
    for name in names[1:]:
        candidate = (
            frames[name].loc[:, columns].sort_values("observation_id").reset_index(drop=True)
        )
        matched = len(reference) == len(candidate) and reference.equals(candidate)
        checks.append(
            {
                "reference": reference_name,
                "candidate": name,
                "reference_rows": len(reference),
                "candidate_rows": len(candidate),
                "exact_metadata_match": matched,
            }
        )
        if not matched:
            raise ValueError(f"Metadata alignment failed: {reference_name} versus {name}")
    return {"checks": checks, "all_exact": True}


def build_overlap_audit(
    *,
    base_feature_names: Sequence[str],
    frozen_opportunity_names: Sequence[str],
    mlat_feature_names: Sequence[str],
    fes_feature_names: Sequence[str],
) -> list[dict[str, object]]:
    """Build the complete outcome-free T01-T09 overlap disposition."""

    universes = {
        "base_85": set(base_feature_names),
        "frozen_15": set(frozen_opportunity_names),
        "mlat_12": set(mlat_feature_names),
        "fes_10": set(fes_feature_names),
    }
    if len(universes["base_85"]) != 85:
        raise ValueError("The canonical base registry must contain exactly 85 unique features.")
    if len(universes["mlat_12"]) != 12:
        raise ValueError("The MLAT comparison registry must contain exactly 12 unique features.")
    if len(universes["fes_10"]) != 10:
        raise ValueError(
            "The FES scalar comparison registry must contain exactly 10 unique features."
        )

    records: list[dict[str, object]] = []
    for spec in TSAY_FEATURE_SPECS:
        related_by_universe = {
            name: tuple(value for value in spec.related_existing_features if value in universe)
            for name, universe in universes.items()
        }
        exact_name_matches = tuple(
            name for name, universe in universes.items() if spec.feature_name in universe
        )
        if exact_name_matches:
            raise ValueError(
                f"Unexpected exact physical-name match for {spec.logical_id}: {exact_name_matches}"
            )
        records.append(
            {
                "logical_id": spec.logical_id,
                "physical_column": spec.feature_name,
                "disposition": spec.overlap_disposition,
                "logical_to_physical": TSAY_LOGICAL_TO_PHYSICAL[spec.logical_id],
                "related_existing_features": spec.related_existing_features,
                "related_by_universe": related_by_universe,
                "exact_formula_match": False,
                "prior_locked_verdict_reused": False,
                "requires_stage2_implementation": True,
                "confirmatory_hypothesis_retained": True,
                "partial_information_controls": spec.partial_information_controls,
                "resolution_reason": (
                    "The frozen formula, window, timing, or transform differs materially from "
                    "every related canonical implementation; no outcome was consulted."
                ),
            }
        )
    if len(records) * len(SESSIONS) != TSAY_PRIMARY_TEST_COUNT:
        raise ValueError("Overlap resolution changed the frozen primary family count.")
    return records


def validate_model_inputs(available_names: Sequence[str]) -> dict[str, object]:
    """Confirm all fixed anchor/control names exist before outcomes are opened."""

    available = set(available_names)
    required = set()
    for session in SESSIONS:
        required.update(D1_INPUTS_BY_SESSION[session])
        required.update(O1_INPUTS_BY_SESSION[session])
    for spec in TSAY_FEATURE_SPECS:
        required.update(spec.partial_information_controls)
    external = required - set(TSAY_LOGICAL_TO_PHYSICAL.values())
    missing = sorted(external - available)
    if missing:
        raise ValueError(f"Frozen anchor/control columns are absent: {missing}")
    return {"required_existing_columns": sorted(external), "all_present": True}


def deterministic_feature_sample_hash(
    matrix: pd.DataFrame, *, sample_size: int = 64
) -> dict[str, object]:
    """Hash deterministic full-schema samples from the constructed feature matrix."""

    if matrix.empty:
        raise ValueError("Cannot hash an empty feature matrix.")
    ordered = matrix.sort_values("observation_id", kind="mergesort").reset_index(drop=True)
    if len(ordered) <= sample_size:
        positions = list(range(len(ordered)))
    else:
        positions = sorted(
            set(
                round(index * (len(ordered) - 1) / (sample_size - 1))
                for index in range(sample_size)
            )
        )
    records: list[dict[str, object]] = []
    for position in positions:
        record: dict[str, object] = {"position": position}
        for column in ordered.columns:
            value = ordered.at[position, column]
            if pd.isna(value):
                record[column] = None
            elif isinstance(value, pd.Timestamp):
                record[column] = value.isoformat()
            elif isinstance(value, (np.bool_, bool)):
                record[column] = bool(value)
            elif isinstance(value, (np.integer, int)):
                record[column] = int(value)
            elif isinstance(value, (np.floating, float)):
                record[column] = float(value).hex()
            else:
                record[column] = str(value)
        records.append(record)
    payload = {
        "algorithm": "sha256-canonical-json-float-hex",
        "row_count": len(ordered),
        "column_count": len(ordered.columns),
        "columns": ordered.columns.tolist(),
        "dtypes": {column: str(ordered[column].dtype) for column in ordered.columns},
        "sample_positions": positions,
        "records": records,
    }
    return {**payload, "digest": canonical_sha256(payload)}


def validate_tsay_feature_matrix(
    matrix: pd.DataFrame,
    eligible_observations: pd.DataFrame,
) -> dict[str, object]:
    """Validate schema, alignment, dtypes, finite values, and frozen ranges."""

    if tuple(matrix.columns) != TSAY_MATRIX_COLUMNS:
        raise ValueError("Tsay feature-matrix column order does not match the frozen schema.")
    if len(matrix) != len(eligible_observations):
        raise ValueError("Tsay matrix row count does not match eligible observations.")
    if matrix["observation_id"].duplicated().any():
        raise ValueError("Tsay matrix observation_id must be unique.")
    if sorted(matrix["product"].astype(str).unique().tolist()) != ["GC"]:
        raise ValueError("Tsay matrix must be GC-only.")
    maximum_date = pd.to_datetime(matrix["trade_date_ny"], errors="raise").max().date()
    if maximum_date > pd.Timestamp("2024-12-31").date():
        raise ValueError("Tsay matrix contains a post-2024 row.")
    forbidden_tokens = ("forward", "future", "label", "outcome", "target", "mfe", "mae")
    forbidden = [
        column
        for column in matrix.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise ValueError(f"Outcome/future columns entered the Tsay matrix: {forbidden}")

    left = matrix.loc[:, TSAY_METADATA_COLUMNS].reset_index(drop=True)
    right = (
        eligible_observations.sort_values(["decision_bar_id", "observation_id"], kind="mergesort")
        .loc[:, TSAY_METADATA_COLUMNS]
        .reset_index(drop=True)
    )
    for column in TSAY_METADATA_COLUMNS:
        if column.endswith("timestamp_utc") or column.endswith("timestamp_ny"):
            equal = pd.to_datetime(left[column], utc=True).equals(
                pd.to_datetime(right[column], utc=True)
            )
        else:
            equal = left[column].astype(str).equals(right[column].astype(str))
        if not equal:
            raise ValueError(f"Eligible metadata alignment failed for {column}.")

    diagnostics: list[dict[str, object]] = []
    specs = {spec.feature_name: spec for spec in TSAY_FEATURE_SPECS}
    for name in TSAY_FEATURE_COLUMNS:
        if str(matrix[name].dtype) != "float32":
            raise TypeError(f"Persisted feature must be float32: {name}={matrix[name].dtype}")
        values = pd.to_numeric(matrix[name], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if np.isinf(values).any():
            raise ValueError(f"Infinite values are forbidden: {name}")
        spec = specs[name]
        if spec.validation_minimum is not None and np.any(
            values[finite] < spec.validation_minimum - 1.0e-6
        ):
            raise ValueError(f"Feature below frozen minimum: {name}")
        if spec.validation_maximum is not None and np.any(
            values[finite] > spec.validation_maximum + 1.0e-6
        ):
            raise ValueError(f"Feature above frozen maximum: {name}")
        diagnostics.append(
            {
                "feature_name": name,
                "dtype": str(matrix[name].dtype),
                "finite_rows": int(finite.sum()),
                "missing_rows": int((~finite).sum()),
                "range_pass": True,
            }
        )
    if str(matrix["tsay_roll_spread_identified_120"].dtype) != "int8":
        raise TypeError("T03 companion state must persist as int8.")
    states = set(matrix["tsay_roll_spread_identified_120"].astype(int).unique().tolist())
    if not states <= {-1, 0, 1}:
        raise ValueError(f"Unexpected T03 companion states: {sorted(states)}")
    for name in TSAY_CONTEXT_COLUMNS[1:]:
        if str(matrix[name].dtype) != "float32":
            raise TypeError(f"Clock intermediate must persist as float32: {name}")
        if np.isinf(pd.to_numeric(matrix[name], errors="coerce")).any():
            raise ValueError(f"Infinite clock intermediate values: {name}")
    return {
        "status": "PASS",
        "rows": len(matrix),
        "columns": len(matrix.columns),
        "features": len(TSAY_FEATURE_COLUMNS),
        "context_columns": len(TSAY_CONTEXT_COLUMNS),
        "maximum_date": maximum_date.isoformat(),
        "diagnostics": diagnostics,
        "sample_hash": deterministic_feature_sample_hash(matrix),
    }


def verify_reloaded_tsay_matrix(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Require exact schema, dtype, category, null, and value equality after reload."""

    try:
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_exact=True,
            check_dtype=True,
            check_categorical=True,
        )
    except AssertionError as exc:
        raise ValueError(f"Tsay matrix save/reload equality failed: {exc}") from exc
