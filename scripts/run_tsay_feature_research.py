"""Run an explicitly authorized outcome-free Tsay research stage."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow
import pyarrow.parquet as pq
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from src.compute import compute_plan  # noqa: E402
from statistical_research.feature_registry import FEATURE_SPECS  # noqa: E402
from statistical_research.fes_project1_config import (  # noqa: E402
    FEATURE_SPECS as FES_FEATURE_SPECS,
)
from statistical_research.mlat_feature_registry import (  # noqa: E402
    MLAT_FEATURE_SPECS,
)
from statistical_research.tsay_access import (  # noqa: E402
    AccessMode,
    TsayGuardedLoader,
    assert_access_manifest_cutoff,
)
from statistical_research.tsay_artifacts import (  # noqa: E402
    load_parquet,
    read_json,
    save_parquet_atomic,
    sha256_file,
    tracked_file_identity,
    verify_json_round_trip,
    verify_parquet_round_trip,
    write_json_atomic,
)
from statistical_research.tsay_backtest import frozen_no_policy  # noqa: E402
from statistical_research.tsay_feature_engineering import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    TSAY_FEATURE_SOURCE_COLUMNS,
    TSAY_OBSERVATION_REQUIRED_COLUMNS,
    TSAY_PRIMITIVE_COLUMNS,
    build_tsay_feature_matrix,
)
from statistical_research.tsay_feature_evaluation import (  # noqa: E402
    CLOCK_BINS_BY_SESSION,
    CLOCK_REFERENCE_BIN_BY_SESSION,
    ROLE_TARGETS,
    evaluate_development_feature_evidence,
)
from statistical_research.tsay_feature_registry import (  # noqa: E402
    D1_INPUTS_BY_SESSION,
    MODEL_LADDER,
    O1_INPUTS_BY_SESSION,
    SECONDARY_TICK_COLUMNS,
    SESSIONS,
    TARGET_COLUMNS,
    TSAY_FEATURE_SPECS,
    TSAY_PRIMARY_TEST_COUNT,
    TSAY_SEED,
    registry_records,
    validate_tsay_registry,
)
from statistical_research.tsay_feature_validation import (  # noqa: E402
    build_overlap_audit,
    canonical_sha256,
    freeze_outer_folds,
    stable_metadata_sample_hash,
    summarize_metadata,
    validate_model_inputs,
    validate_tsay_feature_matrix,
    verify_metadata_alignment,
    verify_reloaded_tsay_matrix,
)
from statistical_research.tsay_models import run_stage4_models  # noqa: E402

EXPECTED_PDF_SHA256 = "B5630A4774C8C23F6BB8F624D05B536E49F09E828F4DB3B01D00B5B38A119E33"
EXPECTED_BRANCH = "feature-research"
STARTING_COMMIT = "362f4061616ff2bfc4a3e40803419c5afd22f967"
ARTIFACT_ROOT = (
    ROOT / "data" / "processed" / "statistical_research" / "tsay_feature_research" / "v1"
)
CONTRACT_PATH = (
    ROOT / "project_docs" / "tsay_feature_research" / "tsay_feature_research_contract_v1.md"
)
PROMPT_PATH = ROOT / "project_docs" / "tsay_project1_implementation_agent_prompt.md"
AMENDMENT_PATH = (
    ROOT
    / "project_docs"
    / "tsay_feature_research"
    / "tsay_stage2_numerical_compatibility_amendment_v1.md"
)
DETERMINISTIC_ROLLING_COMMIT = "1871dc3eba64e11cb95a2d018d53da6953d764f9"

TRUSTED_INPUTS = {
    "bars": "data/processed/research_bars_gc_mgc_1m.parquet",
    "eligible": "data/processed/statistical_research/eligible_observations_gc.parquet",
    "labels": "data/processed/statistical_research/forward_labels_gc.parquet",
    "features": "data/processed/statistical_research/feature_matrix_gc.parquet",
    "frozen": "data/processed/statistical_research/frozen_expansion_feature_set_gc.parquet",
    "fes_development": (
        "data/processed/statistical_research/fes_project1/v1/scalar_features_development_gc.parquet"
    ),
    "fes_validation_sealed": (
        "data/processed/statistical_research/fes_project1/v1/"
        "scalar_features_validation_sealed_gc.parquet"
    ),
    "mlat_features": (
        "data/processed/statistical_research/mlat_feature_research/v1/"
        "feature_matrix_mlat_gc.parquet"
    ),
}

IDENTITY_COLUMNS = (
    "observation_id",
    "decision_timestamp_utc",
    "trade_date_ny",
    "entry_session",
    "research_partition",
    "product",
)
FES_IDENTITY_COLUMNS = IDENTITY_COLUMNS[:-1]
MLAT_IDENTITY_COLUMNS = tuple(column for column in IDENTITY_COLUMNS if column != "entry_session")

STAGE3_LABEL_COLUMNS = (
    *IDENTITY_COLUMNS,
    "decision_atr_20m",
    "forward_return_60_atr",
    "forward_return_60_ticks",
    "future_range_60_atr",
    "future_range_60_ticks",
    "forward_return_30_atr",
    "future_range_30_atr",
    "label_available_60",
    "label_available_30",
)

STAGE4_LABEL_COLUMNS = (
    "observation_id",
    "decision_timestamp_utc",
    "trade_date_ny",
    "entry_session",
    "research_partition",
    "product",
    "forward_return_60_atr",
    "future_range_60_atr",
    "expansion_label_60",
    "label_available_60",
    "exit_timestamp_utc_60",
)

FES_STAGE3_CONTROLS = (
    "ret_tail_balance_60",
    "ret_outlier_fraction_60",
    "ordered_draw_balance_30",
    "return_acf_energy_60",
    "lagged_volume_return_spearman_30",
    "range_volume_spearman_30",
    "volume_profile_slope_30",
)

MLAT_STAGE3_CONTROLS = (
    "amihud_illiquidity_60",
    "realized_semivariance_balance_60",
    "bipower_jump_ratio_60",
    "volatility_of_volatility_60",
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _locate_verified_pdf() -> dict[str, object]:
    candidates = sorted((Path.home() / "Downloads").glob("*.pdf"))
    for candidate in candidates:
        digest = sha256_file(candidate).upper()
        if digest == EXPECTED_PDF_SHA256:
            return {
                "file_name": candidate.name,
                "bytes": candidate.stat().st_size,
                "sha256": digest,
                "hash_verified": True,
                "content_researched": False,
            }
    raise RuntimeError(
        "The supplied Tsay PDF was not found by its required SHA-256 in the Downloads folder."
    )


def _schema_record(loader: TsayGuardedLoader, key: str, mode: AccessMode) -> dict[str, object]:
    relative = TRUSTED_INPUTS[key]
    schema = loader.footer_schema(relative, mode=mode)
    fields = [{"name": field.name, "type": str(field.type)} for field in schema]
    return {
        "source": relative,
        "column_count": len(schema),
        "fields": fields,
        "schema_sha256": canonical_sha256(fields),
        "footer_only": True,
    }


def _partitioned_bar_metadata(loader: TsayGuardedLoader) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = loader.read_parquet(
        TRUSTED_INPUTS["bars"],
        columns=("ts_event_utc", "trade_date_ny", "session_label", "product"),
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    dates = pd.to_datetime(frame["trade_date_ny"], errors="raise")
    frame = frame.assign(
        research_partition=np.where(
            dates.dt.year <= 2023,
            "Development",
            "Validation",
        )
    )
    counts = (
        frame.groupby(["research_partition", "session_label"], observed=True, sort=True)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict(orient="records")
    )
    audit = {
        "source": TRUSTED_INPUTS["bars"],
        "returned_gc_rows_through_2024": len(frame),
        "unique_timestamps": int(frame["ts_event_utc"].nunique()),
        "timestamp_unique": not frame["ts_event_utc"].duplicated().any(),
        "partition_session_counts": counts,
        "metadata_sample_sha256": stable_metadata_sample_hash(
            frame.rename(columns={"ts_event_utc": "decision_timestamp_utc"}).assign(
                observation_id=np.arange(len(frame), dtype=np.int64),
                entry_session=frame["session_label"].astype(str),
            )
        ),
    }
    if not audit["timestamp_unique"]:
        raise RuntimeError("Guarded GC bar timestamps are not unique.")
    return frame, audit


def _existing_hash_proofs() -> dict[str, object]:
    fes_manifest = read_json(
        ROOT / "reports/statistical_research/fes_project1/v1/section2_hash_manifest.json"
    )
    fes_artifacts = fes_manifest["artifacts"]
    fes_dev_relative = TRUSTED_INPUTS["fes_development"]
    fes_dev_actual = sha256_file(ROOT / fes_dev_relative)
    fes_dev_expected = fes_artifacts[fes_dev_relative]
    if fes_dev_actual != fes_dev_expected:
        raise RuntimeError("FES Development scalar hash does not match its frozen manifest.")

    mlat_manifest = read_json(
        ROOT / "reports/statistical_research/mlat_feature_research/v1/table_manifest.json"
    )
    mlat_key = next(
        key for key in mlat_manifest["artifacts"] if key.endswith("feature_matrix_mlat_gc.parquet")
    )
    mlat_expected = mlat_manifest["artifacts"][mlat_key]["sha256"]
    mlat_actual = sha256_file(ROOT / TRUSTED_INPUTS["mlat_features"])
    if mlat_actual != mlat_expected:
        raise RuntimeError("MLAT feature-matrix hash does not match its frozen manifest.")

    sealed_relative = TRUSTED_INPUTS["fes_validation_sealed"]
    return {
        "fes_development": {
            "path": fes_dev_relative,
            "expected_sha256": fes_dev_expected,
            "actual_sha256": fes_dev_actual,
            "verified": True,
        },
        "fes_validation_sealed": {
            "path": sealed_relative,
            "declared_sha256": fes_artifacts[sealed_relative],
            "stage1_action": "PATH_AND_FOOTER_SCHEMA_ONLY_NO_ROW_OR_CONTENT_HASH",
        },
        "mlat_features": {
            "path": TRUSTED_INPUTS["mlat_features"],
            "expected_sha256": mlat_expected,
            "actual_sha256": mlat_actual,
            "verified": True,
        },
    }


def _file_identity_without_mixed_content_hash(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"Required trusted input is absent: {relative}")
    parquet = pq.ParquetFile(path)
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "row_groups": parquet.metadata.num_row_groups,
        "whole_file_hash_recomputed": False,
        "reason": "Mixed-date artifact; Stage 1 hashes only guarded metadata samples.",
    }


def run_stage1() -> dict[str, object]:
    """Execute Stage 1 only and return its integrity checkpoint."""

    os.environ["PROJECT_COMPUTE"] = "cpu"
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"Stage 1 must run on milestone branch {EXPECTED_BRANCH!r}.")
    if _git("merge-base", "HEAD", STARTING_COMMIT) != STARTING_COMMIT:
        raise RuntimeError("The milestone branch does not descend from the user-provided start.")
    if not CONTRACT_PATH.is_file() or not PROMPT_PATH.is_file():
        raise RuntimeError("The frozen contract or controlling prompt is absent.")
    contract_hash_before = sha256_file(CONTRACT_PATH)
    validate_tsay_registry()
    pdf_identity = _locate_verified_pdf()
    hash_proofs = _existing_hash_proofs()

    loader = TsayGuardedLoader(ROOT, stage=1)
    schema_modes = {
        "bars": AccessMode.TIMESERIES_WITH_PRODUCT,
        "eligible": AccessMode.TIMESERIES_WITH_PRODUCT,
        "labels": AccessMode.TIMESERIES_WITH_PRODUCT,
        "features": AccessMode.TIMESERIES_WITH_PRODUCT,
        "frozen": AccessMode.NON_TIMESERIES_METADATA,
        "fes_development": AccessMode.GC_MANIFEST_TIMESERIES,
        "fes_validation_sealed": AccessMode.GC_MANIFEST_TIMESERIES,
        "mlat_features": AccessMode.TIMESERIES_WITH_PRODUCT,
    }
    schemas = {key: _schema_record(loader, key, mode) for key, mode in schema_modes.items()}
    label_schema_names = {field["name"] for field in schemas["labels"]["fields"]}
    required_label_columns = set(TARGET_COLUMNS + SECONDARY_TICK_COLUMNS)
    missing_label_columns = sorted(required_label_columns - label_schema_names)
    if missing_label_columns:
        raise RuntimeError(f"Required label schema columns are absent: {missing_label_columns}")
    schemas["labels"]["required_target_columns"] = sorted(required_label_columns)
    schemas["labels"]["required_target_columns_present"] = True
    schemas["labels"]["outcome_values_materialized"] = False

    _, bar_audit = _partitioned_bar_metadata(loader)
    eligible = loader.read_parquet(
        TRUSTED_INPUTS["eligible"],
        columns=IDENTITY_COLUMNS,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    labels = loader.read_parquet(
        TRUSTED_INPUTS["labels"],
        columns=IDENTITY_COLUMNS,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    features = loader.read_parquet(
        TRUSTED_INPUTS["features"],
        columns=IDENTITY_COLUMNS,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    mlat = loader.read_parquet(
        TRUSTED_INPUTS["mlat_features"],
        columns=MLAT_IDENTITY_COLUMNS,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    fes_development = loader.read_parquet(
        TRUSTED_INPUTS["fes_development"],
        columns=FES_IDENTITY_COLUMNS,
        mode=AccessMode.GC_MANIFEST_TIMESERIES,
        gc_manifest_verified=True,
    )
    frozen = loader.read_parquet(
        TRUSTED_INPUTS["frozen"],
        columns=("feature_name", "in_frozen_set", "role"),
        mode=AccessMode.NON_TIMESERIES_METADATA,
    )

    if mlat["observation_id"].duplicated().any():
        raise RuntimeError("MLAT observation_id is not unique.")
    mlat_with_session = mlat.merge(
        eligible.loc[:, ["observation_id", "entry_session"]],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    if mlat_with_session["entry_session"].isna().any():
        raise RuntimeError("MLAT identity rows do not map completely to eligible sessions.")

    metadata_audits = {
        "bars": bar_audit,
        "eligible": summarize_metadata(eligible, source=TRUSTED_INPUTS["eligible"]),
        "labels_identity_only": summarize_metadata(labels, source=TRUSTED_INPUTS["labels"]),
        "features_identity_only": summarize_metadata(features, source=TRUSTED_INPUTS["features"]),
        "mlat_identity_only": summarize_metadata(
            mlat_with_session, source=TRUSTED_INPUTS["mlat_features"]
        ),
        "fes_development_identity_only": summarize_metadata(
            fes_development, source=TRUSTED_INPUTS["fes_development"]
        ),
    }
    alignment = verify_metadata_alignment(
        {
            "eligible": eligible,
            "labels": labels,
            "features": features,
            "mlat": mlat_with_session,
        }
    )
    development_eligible = eligible.loc[
        eligible["research_partition"].astype(str).eq("Development"),
        list(FES_IDENTITY_COLUMNS),
    ]
    fes_alignment = verify_metadata_alignment(
        {"eligible_development": development_eligible, "fes_development": fes_development}
    )

    frozen_names = tuple(
        frozen.loc[frozen["in_frozen_set"].astype(bool), "feature_name"].astype(str).unique()
    )
    if len(frozen_names) != 15:
        raise RuntimeError(
            f"Frozen opportunity set must resolve to 15 names, got {len(frozen_names)}"
        )
    base_names = tuple(spec.feature_name for spec in FEATURE_SPECS)
    mlat_names = tuple(spec.feature_name for spec in MLAT_FEATURE_SPECS)
    fes_names = tuple(spec["name"] for spec in FES_FEATURE_SPECS)
    overlap = build_overlap_audit(
        base_feature_names=base_names,
        frozen_opportunity_names=frozen_names,
        mlat_feature_names=mlat_names,
        fes_feature_names=fes_names,
    )
    model_inputs = validate_model_inputs(base_names + mlat_names + fes_names)
    folds = freeze_outer_folds(eligible)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_payloads: dict[str, Any] = {
        "feature_registry.json": {
            "family_count": len(TSAY_FEATURE_SPECS),
            "primary_test_count": TSAY_PRIMARY_TEST_COUNT,
            "records": registry_records(),
        },
        "overlap_audit.json": {
            "outcome_values_consulted": False,
            "resolved_family_count": len(overlap),
            "resolved_primary_tests": len(overlap) * 2,
            "records": overlap,
        },
        "schema_audit.json": schemas,
        "metadata_audit.json": {
            "audits": metadata_audits,
            "canonical_alignment": alignment,
            "fes_development_alignment": fes_alignment,
            "label_outcome_values_materialized": False,
            "validation_outcomes_materialized": False,
            "mgc_rows_materialized": False,
            "post_2024_rows_materialized": False,
        },
        "fold_memberships.json": folds,
        "model_input_manifest.json": {
            "fixed_model_ladder": MODEL_LADDER,
            "existing_inputs": model_inputs,
            "outcome_selected_inputs": False,
        },
        "input_provenance.json": {
            "pdf": pdf_identity,
            "existing_manifest_hash_proofs": hash_proofs,
            "mixed_date_file_identities": {
                key: _file_identity_without_mixed_content_hash(TRUSTED_INPUTS[key])
                for key in ("bars", "eligible", "labels", "features")
            },
            "prompt": tracked_file_identity(
                ROOT, "project_docs/tsay_project1_implementation_agent_prompt.md"
            ),
            "contract": tracked_file_identity(
                ROOT,
                "project_docs/tsay_feature_research/tsay_feature_research_contract_v1.md",
            ),
            "git": {
                "branch": EXPECTED_BRANCH,
                "head": _git("rev-parse", "HEAD"),
                "starting_commit": STARTING_COMMIT,
                "descends_from_start": True,
                "working_tree_dirty": bool(_git("status", "--porcelain")),
            },
        },
        "environment_manifest.json": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pyarrow.__version__,
            "project_compute": os.environ["PROJECT_COMPUTE"],
            "compute_plan": compute_plan().describe(),
            "seed": TSAY_SEED,
            "float_research_dtype": "float64",
            "deterministic_cpu_required": True,
        },
    }
    expected_versions = {
        "numpy": "2.5.1",
        "pandas": "3.0.3",
        "scipy": "1.18.0",
        "pyarrow": "25.0.0",
    }
    observed_versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
    }
    if observed_versions != expected_versions:
        raise RuntimeError(
            f"Pinned numerical environment mismatch: expected {expected_versions}, got {observed_versions}"
        )

    for name, payload in artifact_payloads.items():
        path = ARTIFACT_ROOT / name
        write_json_atomic(path, payload)
        verify_json_round_trip(path, payload)

    access_path = ARTIFACT_ROOT / "access_manifest.json"
    loader.write_manifest(access_path)
    access_payload = read_json(access_path)
    assert_access_manifest_cutoff(access_payload)

    contract_hash_after = sha256_file(CONTRACT_PATH)
    if contract_hash_after != contract_hash_before:
        raise RuntimeError("The frozen contract changed during Stage 1 execution.")

    hashed_files = {name: sha256_file(ARTIFACT_ROOT / name) for name in sorted(artifact_payloads)}
    hashed_files["access_manifest.json"] = sha256_file(access_path)
    source_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in (
            "project_docs/tsay_project1_implementation_agent_prompt.md",
            "project_docs/tsay_feature_research/tsay_feature_research_contract_v1.md",
            "project_docs/tsay_feature_research/tsay_book_map.md",
            "src/statistical_research/tsay_access.py",
            "src/statistical_research/tsay_artifacts.py",
            "src/statistical_research/tsay_feature_registry.py",
            "src/statistical_research/tsay_feature_validation.py",
            "scripts/build_tsay_feature_research_notebook.py",
            "scripts/run_tsay_feature_research.py",
            "reports/statistical_research/summaries/tsay_feature_research_status.md",
        )
    }
    hash_manifest = {
        "contract_sha256": contract_hash_after,
        "artifact_sha256": hashed_files,
        "source_sha256": source_hashes,
        "fold_membership_sha256": folds["fold_membership_sha256"],
        "access_manifest_cutoff_verified": True,
    }
    write_json_atomic(ARTIFACT_ROOT / "stage1_hash_manifest.json", hash_manifest)
    verify_json_round_trip(ARTIFACT_ROOT / "stage1_hash_manifest.json", hash_manifest)

    checkpoint = {
        "stage": 1,
        "stage_name": "mandate_provenance_schema_frozen_contract",
        "status": "READY",
        "integrity_reasons": [],
        "research_verdict": "NOT_AUTHORIZED",
        "contract_sha256": contract_hash_after,
        "fold_membership_sha256": folds["fold_membership_sha256"],
        "access_manifest_sha256": sha256_file(access_path),
        "hash_manifest_sha256": sha256_file(ARTIFACT_ROOT / "stage1_hash_manifest.json"),
        "outcome_values_read": False,
        "validation_outcomes_read": False,
        "mgc_research_rows_read": False,
        "post_2024_rows_read": False,
        "stage2_authorized": True,
        "stage2_authorization_scope": (
            "Outcome-free causal feature construction only, after the user says continue."
        ),
    }
    write_json_atomic(ARTIFACT_ROOT / "stage1_checkpoint.json", checkpoint)
    verify_json_round_trip(ARTIFACT_ROOT / "stage1_checkpoint.json", checkpoint)
    return checkpoint


def verify_stage1_for_stage2() -> dict[str, object]:
    """Verify the Stage 1 checkpoint, ignored artifacts, and immutable sources."""

    checkpoint_path = ARTIFACT_ROOT / "stage1_checkpoint.json"
    manifest_path = ARTIFACT_ROOT / "stage1_hash_manifest.json"
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Stage 1 checkpoint or hash manifest is absent.")
    checkpoint = read_json(checkpoint_path)
    manifest = read_json(manifest_path)
    if checkpoint.get("status") != "READY" or not checkpoint.get("stage2_authorized"):
        raise RuntimeError("Stage 1 did not authorize outcome-free Stage 2 construction.")
    if checkpoint.get("research_verdict") != "NOT_AUTHORIZED":
        raise RuntimeError("Unexpected Stage 1 research verdict.")
    if sha256_file(manifest_path) != checkpoint.get("hash_manifest_sha256"):
        raise RuntimeError("Stage 1 hash-manifest identity changed.")
    if sha256_file(CONTRACT_PATH) != manifest.get("contract_sha256"):
        raise RuntimeError("The Stage 1-frozen contract hash changed.")
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = ARTIFACT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Stage 1 artifact hash changed: {name}")
    immutable_sources = (
        "project_docs/tsay_project1_implementation_agent_prompt.md",
        "project_docs/tsay_feature_research/tsay_feature_research_contract_v1.md",
        "project_docs/tsay_feature_research/tsay_book_map.md",
        "src/statistical_research/tsay_feature_registry.py",
    )
    for relative in immutable_sources:
        expected = manifest["source_sha256"][relative]
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Stage 1 immutable source hash changed: {relative}")
    stage1_access = read_json(ARTIFACT_ROOT / "access_manifest.json")
    assert_access_manifest_cutoff(stage1_access)
    if checkpoint.get("fold_membership_sha256") != manifest.get("fold_membership_sha256"):
        raise RuntimeError("Stage 1 fold hash does not match its checkpoint.")
    return {
        "status": "VERIFIED",
        "stage1_status": checkpoint["status"],
        "stage1_research_verdict": checkpoint["research_verdict"],
        "stage1_hash_manifest_sha256": checkpoint["hash_manifest_sha256"],
        "contract_sha256": manifest["contract_sha256"],
        "fold_membership_sha256": manifest["fold_membership_sha256"],
        "artifact_hashes_verified": len(manifest["artifact_sha256"]),
        "immutable_source_hashes_verified": list(immutable_sources),
        "full_stage1_hash_verification_before_stage2_edits": True,
    }


class _PeakMemoryMonitor:
    """Sample process RSS while the resource-aware Stage 2 build runs."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._process = psutil.Process()
        self.peak_rss_bytes = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.05):
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._process.memory_info().rss)

    def __enter__(self) -> _PeakMemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._process.memory_info().rss)
        self._stop.set()
        self._thread.join()


def _assert_stage2_access(payload: dict[str, object]) -> None:
    assert_access_manifest_cutoff(payload)
    if payload.get("stage") != 2:
        raise RuntimeError("Stage 2 access manifest has the wrong stage.")
    records = payload.get("records", [])
    if any(
        str(record.get("source", "")).endswith("forward_labels_gc.parquet") for record in records
    ):
        raise RuntimeError("The Stage 2 access manifest contains label access.")
    for record in records:
        if record.get("access_kind") != "PARQUET_READ":
            continue
        if record.get("mode") == AccessMode.TIMESERIES_WITH_PRODUCT.value and record.get(
            "returned_product_set"
        ) not in ([], ["GC"], (), ("GC",)):
            raise RuntimeError(f"Non-GC Stage 2 materialization: {record}")


def _guarded_primitive_sha256(frame: pd.DataFrame) -> str:
    """Hash only the guarded through-2024 canonical primitive rows."""

    ordered = frame.sort_values("observation_id", kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(b"tsay-guarded-canonical-primitives-v1\0")
    for column in ("observation_id", *TSAY_PRIMITIVE_COLUMNS):
        digest.update(column.encode("ascii") + b"\0")
        dtype = np.int64 if column == "observation_id" else np.float32
        values = pd.to_numeric(ordered[column], errors="raise").to_numpy(dtype=dtype)
        digest.update(str(values.dtype).encode("ascii") + b"\0")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def run_stage2() -> dict[str, object]:
    """Construct and persist T01-T09 without reading any label value or row."""

    os.environ["PROJECT_COMPUTE"] = "cpu"
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"Stage 2 must run on milestone branch {EXPECTED_BRANCH!r}.")
    prior_verification = verify_stage1_for_stage2()
    if not AMENDMENT_PATH.is_file():
        raise RuntimeError("The authorized Stage 2 numerical amendment is absent.")
    if _git("rev-parse", DETERMINISTIC_ROLLING_COMMIT) != DETERMINISTIC_ROLLING_COMMIT:
        raise RuntimeError("The deterministic rolling-window provenance commit is absent.")
    amendment_sha256 = sha256_file(AMENDMENT_PATH)
    legacy_feature_path = ROOT / TRUSTED_INPUTS["features"]
    legacy_identity_before = {
        "path": TRUSTED_INPUTS["features"],
        "bytes": legacy_feature_path.stat().st_size,
        "modified_time_ns": legacy_feature_path.stat().st_mtime_ns,
        "whole_file_hash_recomputed": False,
    }
    previous_checkpoint_path = ARTIFACT_ROOT / "stage2_checkpoint.json"
    if previous_checkpoint_path.is_file():
        previous_checkpoint = read_json(previous_checkpoint_path)
        if previous_checkpoint.get("status") == "BLOCKED":
            historical_path = ARTIFACT_ROOT / "stage2_checkpoint_pre_amendment_blocked.json"
            if historical_path.is_file():
                verify_json_round_trip(historical_path, previous_checkpoint)
            else:
                write_json_atomic(historical_path, previous_checkpoint)
                verify_json_round_trip(historical_path, previous_checkpoint)
    loader = TsayGuardedLoader(ROOT, stage=2)

    bar_columns = tuple(
        column for column in TSAY_FEATURE_SOURCE_COLUMNS if column != "source_row_id"
    )
    bars = loader.read_parquet(
        TRUSTED_INPUTS["bars"],
        columns=bar_columns,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    bars = bars.sort_values("ts_event_utc", kind="mergesort").reset_index(drop=True)
    bars.insert(0, "source_row_id", np.arange(len(bars), dtype=np.int64))
    bars = bars.loc[:, TSAY_FEATURE_SOURCE_COLUMNS]

    eligible = loader.read_parquet(
        TRUSTED_INPUTS["eligible"],
        columns=TSAY_OBSERVATION_REQUIRED_COLUMNS,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    canonical = loader.read_parquet(
        TRUSTED_INPUTS["features"],
        columns=("observation_id", "trade_date_ny", "product", *TSAY_PRIMITIVE_COLUMNS),
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    canonical = canonical.loc[:, ("observation_id", *TSAY_PRIMITIVE_COLUMNS)]
    guarded_legacy_primitive_sha256 = _guarded_primitive_sha256(canonical)

    build_started = perf_counter()
    with _PeakMemoryMonitor() as memory:
        result = build_tsay_feature_matrix(
            bars,
            eligible,
            canonical_primitives=canonical,
            require_canonical_primitive_equality=True,
            batch_size=DEFAULT_BATCH_SIZE,
        )
    elapsed = perf_counter() - build_started
    validation = validate_tsay_feature_matrix(result.matrix, eligible)
    if not result.construction_audit["passed"].all():
        raise RuntimeError("Stage 2 construction audit did not pass.")
    if not result.primitive_equality["status"].eq("PASS").all():
        raise RuntimeError("Canonical primitive compatibility did not pass.")
    primitive_by_name = result.primitive_equality.set_index("primitive")
    for exact_name in ("atr_20", "current_range_over_atr"):
        row = primitive_by_name.loc[exact_name]
        if not bool(row["float32_exact"]) or not bool(row["null_mask_exact"]):
            raise RuntimeError(f"Exact canonical primitive gate failed: {exact_name}")
    rv_row = primitive_by_name.loc["realized_volatility_15"]
    if not bool(rv_row["null_mask_exact"]) or int(rv_row["maximum_ulp_distance"]) > 1:
        raise RuntimeError("The amended realized_volatility_15 compatibility gate failed.")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    frames = {
        "feature_matrix_tsay_gc.parquet": result.matrix,
        "feature_observation_audit_tsay_gc.parquet": result.observation_audit,
        "feature_missing_reasons_tsay_gc.parquet": result.missing_reasons,
        "clock_profile_reference_tsay_gc.parquet": result.clock_reference,
        "feature_diagnostics_tsay_gc.parquet": result.feature_diagnostics,
        "feature_coverage_tsay_gc.parquet": result.coverage,
        "primitive_equality_tsay_gc.parquet": result.primitive_equality,
        "construction_audit_tsay_gc.parquet": result.construction_audit,
    }
    artifact_hashes: dict[str, str] = {}
    for name, frame in frames.items():
        path = ARTIFACT_ROOT / name
        save_parquet_atomic(path, frame)
        artifact_hashes[name] = verify_parquet_round_trip(path, frame)
    reloaded_matrix = load_parquet(ARTIFACT_ROOT / "feature_matrix_tsay_gc.parquet")
    verify_reloaded_tsay_matrix(result.matrix, reloaded_matrix)
    reloaded_validation = validate_tsay_feature_matrix(reloaded_matrix, eligible)
    if reloaded_validation["sample_hash"]["digest"] != validation["sample_hash"]["digest"]:
        raise RuntimeError("The reloaded feature-matrix sample hash changed.")

    registry_payload = {
        "research_id": "tsay_feature_research",
        "version": "v1",
        "records": registry_records(),
        "outcomes_consulted": False,
    }
    overlap_payload = read_json(ARTIFACT_ROOT / "overlap_audit.json")
    numerical_environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "project_compute": os.environ["PROJECT_COMPUTE"],
    }
    expected_environment = {
        "numpy": "2.5.1",
        "pandas": "3.0.3",
        "scipy": "1.18.0",
        "pyarrow": "25.0.0",
    }
    if {key: numerical_environment[key] for key in expected_environment} != expected_environment:
        raise RuntimeError("Stage 2 numerical environment differs from the pinned versions.")
    compatibility_payload = {
        "amendment_sha256": amendment_sha256,
        "parent_contract_sha256": sha256_file(CONTRACT_PATH),
        "legacy_feature_matrix_identity": legacy_identity_before,
        "guarded_legacy_primitive_sha256": guarded_legacy_primitive_sha256,
        "deterministic_rolling_commit": DETERMINISTIC_ROLLING_COMMIT,
        "provenance_explanation": (
            "The legacy feature_matrix_gc.parquet predates commit 1871dc3, which "
            "established the current deterministic CPU rolling-window implementation "
            "and pinned numerical stack. The legacy artifact remains unchanged."
        ),
        "environment": numerical_environment,
        "records": result.primitive_equality.to_dict(orient="records"),
        "general_numerical_tolerance_authorized": False,
        "feature_formula_modified": False,
    }
    json_payloads: dict[str, Any] = {
        "feature_registry_stage2.json": registry_payload,
        "overlap_audit_stage2.json": overlap_payload,
        "stage1_continuation_verification.json": prior_verification,
        "feature_validation_stage2.json": validation,
        "canonical_primitive_compatibility_stage2.json": compatibility_payload,
        "stage2_run_manifest.json": {
            "stage": 2,
            "compute": "deterministic CPU float64",
            "batch_size": DEFAULT_BATCH_SIZE,
            "source_rows": len(bars),
            "eligible_rows": len(eligible),
            "elapsed_seconds": elapsed,
            "peak_process_rss_bytes": memory.peak_rss_bytes,
            "sampled_rows": False,
            "labels_read": False,
            "validation_outcomes_read": False,
            "mgc_rows_read": False,
            "post_2024_rows_read": False,
            "build_runtime": result.runtime_summary.to_dict(orient="records"),
            "build_timings": result.build_timings.to_dict(orient="records"),
        },
    }
    for name, payload in json_payloads.items():
        path = ARTIFACT_ROOT / name
        write_json_atomic(path, payload)
        artifact_hashes[name] = verify_json_round_trip(path, payload)

    access_path = ARTIFACT_ROOT / "access_manifest_stage2.json"
    loader.write_manifest(access_path)
    access_payload = read_json(access_path)
    _assert_stage2_access(access_payload)
    artifact_hashes[access_path.name] = sha256_file(access_path)
    legacy_identity_after = {
        "path": TRUSTED_INPUTS["features"],
        "bytes": legacy_feature_path.stat().st_size,
        "modified_time_ns": legacy_feature_path.stat().st_mtime_ns,
        "whole_file_hash_recomputed": False,
    }
    if legacy_identity_after != legacy_identity_before:
        raise RuntimeError("The legacy canonical feature artifact changed during Stage 2.")

    source_paths = (
        "src/statistical_research/tsay_access.py",
        "src/statistical_research/tsay_artifacts.py",
        "src/statistical_research/tsay_feature_engineering.py",
        "src/statistical_research/tsay_feature_registry.py",
        "src/statistical_research/tsay_feature_validation.py",
        "scripts/run_tsay_feature_research.py",
        "scripts/build_tsay_feature_research_notebook.py",
        "project_docs/tsay_feature_research/tsay_stage2_numerical_compatibility_amendment_v1.md",
        "reports/statistical_research/summaries/tsay_feature_research_status.md",
    )
    stage2_hash_manifest = {
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "amendment_sha256": amendment_sha256,
        "legacy_feature_matrix_identity": legacy_identity_after,
        "guarded_legacy_primitive_sha256": guarded_legacy_primitive_sha256,
        "deterministic_rolling_commit": DETERMINISTIC_ROLLING_COMMIT,
        "stage1_hash_manifest_sha256": prior_verification["stage1_hash_manifest_sha256"],
        "feature_matrix_sample_sha256": validation["sample_hash"]["digest"],
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "source_sha256": {relative: sha256_file(ROOT / relative) for relative in source_paths},
        "access_manifest_cutoff_verified": True,
        "label_access_absent": True,
        "mgc_access_absent": True,
        "post_2024_access_absent": True,
    }
    hash_path = ARTIFACT_ROOT / "stage2_hash_manifest.json"
    write_json_atomic(hash_path, stage2_hash_manifest)
    verify_json_round_trip(hash_path, stage2_hash_manifest)

    checkpoint = {
        "stage": 2,
        "stage_name": "outcome_free_causal_feature_construction",
        "status": "READY",
        "integrity_reasons": [],
        "research_verdict": "NOT_AUTHORIZED",
        "contract_sha256": stage2_hash_manifest["contract_sha256"],
        "amendment_sha256": amendment_sha256,
        "legacy_feature_matrix_identity": legacy_identity_after,
        "guarded_legacy_primitive_sha256": guarded_legacy_primitive_sha256,
        "deterministic_rolling_commit": DETERMINISTIC_ROLLING_COMMIT,
        "canonical_primitive_compatibility": compatibility_payload["records"],
        "stage1_hash_manifest_sha256": prior_verification["stage1_hash_manifest_sha256"],
        "stage2_hash_manifest_sha256": sha256_file(hash_path),
        "feature_matrix_sha256": artifact_hashes["feature_matrix_tsay_gc.parquet"],
        "feature_matrix_sample_sha256": validation["sample_hash"]["digest"],
        "access_manifest_sha256": artifact_hashes["access_manifest_stage2.json"],
        "outcome_values_read": False,
        "validation_outcomes_read": False,
        "mgc_research_rows_read": False,
        "post_2024_rows_read": False,
        "stage3_authorized": True,
        "stage3_authorization_scope": (
            "Development-only feature evidence, after the user says continue."
        ),
    }
    checkpoint_path = ARTIFACT_ROOT / "stage2_checkpoint.json"
    write_json_atomic(checkpoint_path, checkpoint)
    verify_json_round_trip(checkpoint_path, checkpoint)
    return checkpoint


def verify_stage2_for_stage3() -> dict[str, object]:
    """Verify the completed Stage 2 checkpoint before Development labels open."""

    checkpoint_path = ARTIFACT_ROOT / "stage2_checkpoint.json"
    manifest_path = ARTIFACT_ROOT / "stage2_hash_manifest.json"
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Stage 2 checkpoint or hash manifest is absent.")
    checkpoint = read_json(checkpoint_path)
    manifest = read_json(manifest_path)
    if checkpoint.get("status") != "READY" or not checkpoint.get("stage3_authorized"):
        raise RuntimeError("Stage 2 did not authorize Development-only Stage 3 evidence.")
    if checkpoint.get("research_verdict") != "NOT_AUTHORIZED":
        raise RuntimeError("Unexpected Stage 2 research verdict.")
    if sha256_file(manifest_path) != checkpoint.get("stage2_hash_manifest_sha256"):
        raise RuntimeError("Stage 2 hash-manifest identity changed.")
    if sha256_file(CONTRACT_PATH) != manifest.get("contract_sha256"):
        raise RuntimeError("The Stage 1-frozen contract hash changed.")
    if sha256_file(AMENDMENT_PATH) != manifest.get("amendment_sha256"):
        raise RuntimeError("The Stage 2 amendment hash changed.")
    verified_artifacts = 0
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = ARTIFACT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Stage 2 artifact hash changed: {name}")
        verified_artifacts += 1
    immutable_stage2_sources = (
        "src/statistical_research/tsay_artifacts.py",
        "src/statistical_research/tsay_feature_engineering.py",
        "src/statistical_research/tsay_feature_registry.py",
        "src/statistical_research/tsay_feature_validation.py",
        "project_docs/tsay_feature_research/tsay_stage2_numerical_compatibility_amendment_v1.md",
    )
    for relative in immutable_stage2_sources:
        if sha256_file(ROOT / relative) != manifest["source_sha256"][relative]:
            raise RuntimeError(f"Stage 2 immutable source hash changed: {relative}")
    stage2_access = read_json(ARTIFACT_ROOT / "access_manifest_stage2.json")
    _assert_stage2_access(stage2_access)
    if sha256_file(ARTIFACT_ROOT / "feature_matrix_tsay_gc.parquet") != checkpoint.get(
        "feature_matrix_sha256"
    ):
        raise RuntimeError("The frozen Stage 2 feature matrix hash changed.")
    return {
        "status": "VERIFIED",
        "stage2_status": checkpoint["status"],
        "stage2_research_verdict": checkpoint["research_verdict"],
        "stage2_hash_manifest_sha256": checkpoint["stage2_hash_manifest_sha256"],
        "contract_sha256": checkpoint["contract_sha256"],
        "amendment_sha256": checkpoint["amendment_sha256"],
        "feature_matrix_sha256": checkpoint["feature_matrix_sha256"],
        "feature_matrix_sample_sha256": checkpoint["feature_matrix_sample_sha256"],
        "access_manifest_sha256": checkpoint["access_manifest_sha256"],
        "artifact_hashes_verified": verified_artifacts,
        "immutable_source_hashes_verified": len(immutable_stage2_sources),
        "full_stage2_hash_verification_before_stage3_edits": True,
        "stage3_continuation_source_changes": [
            "src/statistical_research/tsay_access.py",
            "scripts/run_tsay_feature_research.py",
            "scripts/build_tsay_feature_research_notebook.py",
        ],
    }


def _assert_stage3_access(payload: dict[str, object]) -> None:
    assert_access_manifest_cutoff(payload)
    if payload.get("stage") != 3:
        raise RuntimeError("Stage 3 access manifest has the wrong stage.")
    if payload.get("materialization_cutoff_date_ny") != "2023-12-31":
        raise RuntimeError("Stage 3 access did not use the Development cutoff.")
    records = payload.get("records", [])
    label_records = [
        record
        for record in records
        if str(record.get("source", "")).endswith("forward_labels_gc.parquet")
        and record.get("access_kind") == "PARQUET_READ"
    ]
    if len(label_records) != 1:
        raise RuntimeError("Stage 3 requires exactly one guarded Development label read.")
    label_record = label_records[0]
    if tuple(label_record.get("requested_columns", ())) != STAGE3_LABEL_COLUMNS:
        raise RuntimeError("Stage 3 label columns differ from the exact frozen allowlist.")
    if label_record.get("returned_research_partition_set") not in (
        ["Development"],
        ("Development",),
    ):
        raise RuntimeError("Stage 3 label access was not Development-only.")
    if "research_partition == Development" not in str(label_record.get("pushed_predicate")):
        raise RuntimeError("Stage 3 label access lacks the pushed Development predicate.")
    forbidden_sources = (
        "scalar_features_validation_sealed_gc.parquet",
        "research_bars_gc_mgc_1m.parquet",
    )
    for record in records:
        source = str(record.get("source", ""))
        if any(source.endswith(forbidden) for forbidden in forbidden_sources):
            raise RuntimeError(f"Forbidden Stage 3 source access: {source}")
        maximum = record.get("returned_maximum_date")
        if maximum is not None and str(maximum) > "2023-12-31":
            raise RuntimeError(f"Post-Development Stage 3 row materialized: {record}")
        partitions = record.get("returned_research_partition_set")
        if partitions not in (None, [], (), ["Development"], ("Development",)):
            raise RuntimeError(f"Non-Development Stage 3 partition materialized: {record}")
        products = record.get("returned_product_set")
        if products not in (None, [], (), ["GC"], ("GC",)):
            raise RuntimeError(f"Non-GC Stage 3 product materialized: {record}")


def _label_atr_tick_consistency(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    atr = pd.to_numeric(frame["decision_atr_20m"], errors="coerce").to_numpy(dtype=np.float64)
    for role, targets in ROLE_TARGETS.items():
        atr_values = pd.to_numeric(frame[targets["primary"]], errors="coerce").to_numpy(
            dtype=np.float64
        )
        tick_values = pd.to_numeric(frame[targets["tick"]], errors="coerce").to_numpy(
            dtype=np.float64
        )
        eligible = (
            frame["label_available_60"].fillna(False).astype(bool).to_numpy()
            & np.isfinite(atr)
            & (atr > 0.0)
            & np.isfinite(atr_values)
            & np.isfinite(tick_values)
        )
        recomputed = tick_values[eligible] * 0.10 / atr[eligible]
        difference = np.abs(recomputed - atr_values[eligible])
        passed = bool(
            len(difference)
            and np.allclose(
                recomputed,
                atr_values[eligible],
                rtol=1.0e-12,
                atol=1.0e-12,
                equal_nan=False,
            )
        )
        records.append(
            {
                "role": role,
                "atr_target": targets["primary"],
                "tick_target": targets["tick"],
                "eligible_rows": int(eligible.sum()),
                "maximum_absolute_difference": (
                    float(difference.max()) if len(difference) else float("nan")
                ),
                "rowwise_consistent": passed,
                "aggregate_average_atr_conversion_used": False,
            }
        )
    result = pd.DataFrame.from_records(records)
    if not result["rowwise_consistent"].all():
        raise RuntimeError("Stage 3 ATR/tick label consistency QA failed.")
    return result


def _verify_stage3_identity(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    source_name: str,
    identity_columns: tuple[str, ...],
) -> None:
    if other["observation_id"].duplicated().any():
        raise RuntimeError(f"Duplicate observation_id in {source_name}.")
    if len(base) != len(other) or set(base["observation_id"]) != set(other["observation_id"]):
        raise RuntimeError(f"Development observation identity mismatch: {source_name}")
    aligned = base.loc[:, identity_columns].merge(
        other.loc[:, identity_columns],
        on="observation_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_base", "_other"),
    )
    for column in identity_columns:
        if column == "observation_id":
            continue
        left = aligned[f"{column}_base"]
        right = aligned[f"{column}_other"]
        if column.endswith("timestamp_utc") or column == "trade_date_ny":
            equal = pd.to_datetime(left, errors="raise").equals(
                pd.to_datetime(right, errors="raise")
            )
        else:
            equal = left.astype(str).equals(right.astype(str))
        if not equal:
            raise RuntimeError(f"Development identity mismatch for {column}: {source_name}")


def run_stage3() -> dict[str, object]:
    """Evaluate only the frozen Development feature-evidence family."""

    os.environ["PROJECT_COMPUTE"] = "cpu"
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"Stage 3 must run on milestone branch {EXPECTED_BRANCH!r}.")
    prior_verification = verify_stage2_for_stage3()
    if {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
    } != {
        "numpy": "2.5.1",
        "pandas": "3.0.3",
        "scipy": "1.18.0",
        "pyarrow": "25.0.0",
    }:
        raise RuntimeError("Stage 3 numerical environment differs from pinned versions.")

    all_controls = set()
    for session in SESSIONS:
        all_controls.update(D1_INPUTS_BY_SESSION[session])
        all_controls.update(O1_INPUTS_BY_SESSION[session])
    for spec in TSAY_FEATURE_SPECS:
        all_controls.update(spec.partial_information_controls)
    external_controls = set(FES_STAGE3_CONTROLS) | set(MLAT_STAGE3_CONTROLS)
    canonical_controls = tuple(sorted(all_controls - external_controls))
    if all_controls != set(canonical_controls) | external_controls:
        raise RuntimeError("Stage 3 control-source resolution is incomplete.")

    loader = TsayGuardedLoader(ROOT, stage=3)
    tsay_columns = (
        "observation_id",
        "decision_timestamp_utc",
        "decision_timestamp_ny",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "product",
        *(spec.feature_name for spec in TSAY_FEATURE_SPECS),
    )
    tsay = loader.read_parquet(
        ARTIFACT_ROOT.relative_to(ROOT) / "feature_matrix_tsay_gc.parquet",
        columns=tsay_columns,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    labels = loader.read_parquet(
        TRUSTED_INPUTS["labels"],
        columns=STAGE3_LABEL_COLUMNS,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    canonical_columns = (
        "observation_id",
        "decision_timestamp_utc",
        "trade_date_ny",
        "research_partition",
        "product",
        *canonical_controls,
    )
    canonical = loader.read_parquet(
        TRUSTED_INPUTS["features"],
        columns=canonical_columns,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    fes_columns = (*FES_IDENTITY_COLUMNS, *FES_STAGE3_CONTROLS)
    fes = loader.read_parquet(
        TRUSTED_INPUTS["fes_development"],
        columns=fes_columns,
        mode=AccessMode.GC_MANIFEST_TIMESERIES,
        gc_manifest_verified=True,
    )
    mlat_columns = (*MLAT_IDENTITY_COLUMNS, *MLAT_STAGE3_CONTROLS)
    mlat = loader.read_parquet(
        TRUSTED_INPUTS["mlat_features"],
        columns=mlat_columns,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )

    _verify_stage3_identity(
        tsay,
        labels,
        source_name="Development labels",
        identity_columns=IDENTITY_COLUMNS,
    )
    _verify_stage3_identity(
        tsay,
        canonical,
        source_name="canonical feature matrix",
        identity_columns=(
            "observation_id",
            "decision_timestamp_utc",
            "trade_date_ny",
            "research_partition",
            "product",
        ),
    )
    _verify_stage3_identity(
        tsay,
        fes,
        source_name="FES Development scalar features",
        identity_columns=FES_IDENTITY_COLUMNS,
    )
    _verify_stage3_identity(
        tsay,
        mlat,
        source_name="MLAT feature matrix",
        identity_columns=MLAT_IDENTITY_COLUMNS,
    )

    label_value_columns = tuple(
        column for column in STAGE3_LABEL_COLUMNS if column not in IDENTITY_COLUMNS
    )
    evidence_frame = tsay.merge(
        labels.loc[:, ("observation_id", *label_value_columns)],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    evidence_frame = evidence_frame.merge(
        canonical.loc[:, ("observation_id", *canonical_controls)],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    evidence_frame = evidence_frame.merge(
        fes.loc[:, ("observation_id", *FES_STAGE3_CONTROLS)],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    evidence_frame = evidence_frame.merge(
        mlat.loc[:, ("observation_id", *MLAT_STAGE3_CONTROLS)],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    if len(evidence_frame) != len(tsay):
        raise RuntimeError("Stage 3 one-to-one evidence join changed row count.")
    label_qa = _label_atr_tick_consistency(evidence_frame)

    started = perf_counter()
    with _PeakMemoryMonitor() as memory:
        result = evaluate_development_feature_evidence(evidence_frame)
    elapsed = perf_counter() - started

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    frames = {
        "development_feature_evidence_ledger.parquet": result.evidence_ledger,
        "development_primary_daily_ic.parquet": result.primary_daily_ic,
        "development_diagnostic_30m_daily_ic.parquet": result.diagnostic_daily_ic,
        "development_diagnostic_30m_summary.parquet": result.diagnostic_summary,
        "development_year_stability.parquet": result.year_stability,
        "development_quintile_edges.parquet": result.quintile_edges,
        "development_quintile_summary.parquet": result.quintile_summary,
        "development_partial_daily_ic.parquet": result.partial_daily_ic,
        "development_partial_summary.parquet": result.partial_summary,
        "development_related_feature_redundancy.parquet": (result.related_feature_redundancy),
        "development_candidate_correlation.parquet": result.candidate_correlation,
        "development_label_atr_tick_qa.parquet": label_qa,
    }
    artifact_hashes: dict[str, str] = {}
    for name, frame in frames.items():
        path = ARTIFACT_ROOT / name
        save_parquet_atomic(path, frame)
        artifact_hashes[name] = verify_parquet_round_trip(path, frame)

    continuation_path = ARTIFACT_ROOT / "stage2_continuation_verification_stage3.json"
    write_json_atomic(continuation_path, prior_verification)
    artifact_hashes[continuation_path.name] = verify_json_round_trip(
        continuation_path, prior_verification
    )
    input_manifest = {
        "stage": 3,
        "authorized_partition": "Development",
        "maximum_trade_date_ny": "2023-12-31",
        "product": "GC",
        "joined_rows": len(evidence_frame),
        "sources": {
            "tsay_features": {
                "path": (
                    ARTIFACT_ROOT.relative_to(ROOT) / "feature_matrix_tsay_gc.parquet"
                ).as_posix(),
                "columns": list(tsay_columns),
                "sha256": prior_verification["feature_matrix_sha256"],
            },
            "labels": {
                "path": TRUSTED_INPUTS["labels"],
                "columns": list(STAGE3_LABEL_COLUMNS),
                "row_values_accessed": "Development only",
                "validation_label_values_accessed": False,
            },
            "canonical_controls": {
                "path": TRUSTED_INPUTS["features"],
                "columns": list(canonical_columns),
            },
            "fes_controls": {
                "path": TRUSTED_INPUTS["fes_development"],
                "columns": list(fes_columns),
            },
            "mlat_controls": {
                "path": TRUSTED_INPUTS["mlat_features"],
                "columns": list(mlat_columns),
            },
        },
        "identity_alignment": "PASS_ONE_TO_ONE",
        "outcome_dependent_feature_changes": False,
    }
    input_manifest_path = ARTIFACT_ROOT / "stage3_input_manifest.json"
    write_json_atomic(input_manifest_path, input_manifest)
    artifact_hashes[input_manifest_path.name] = verify_json_round_trip(
        input_manifest_path, input_manifest
    )

    access_path = ARTIFACT_ROOT / "access_manifest_stage3.json"
    loader.write_manifest(access_path)
    access_payload = read_json(access_path)
    _assert_stage3_access(access_payload)
    artifact_hashes[access_path.name] = sha256_file(access_path)

    run_manifest = {
        "stage": 3,
        "compute": "deterministic CPU float64",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pyarrow.__version__,
            "project_compute": os.environ["PROJECT_COMPUTE"],
        },
        "elapsed_seconds": elapsed,
        "peak_process_rss_bytes": memory.peak_rss_bytes,
        "confirmatory_test_count": len(result.evidence_ledger),
        "bootstrap_replicates_per_inference": 2_000,
        "bootstrap_restart_probability": 0.20,
        "seed": TSAY_SEED,
        "partial_information_clock_bins": CLOCK_BINS_BY_SESSION,
        "partial_information_reference_clock_bins": CLOCK_REFERENCE_BIN_BY_SESSION,
        "development_label_rows_read": len(labels),
        "development_label_columns_read": list(STAGE3_LABEL_COLUMNS),
        "validation_label_values_read": False,
        "validation_performance_evaluated": False,
        "post_2023_rows_materialized": False,
        "post_2024_rows_materialized": False,
        "mgc_rows_materialized": False,
        "models_fitted": False,
        "strategy_pnl_calculated": False,
        "diagnostic_30m_selection_authority": False,
        "feature_replacements_or_tuning": False,
    }
    run_manifest_path = ARTIFACT_ROOT / "stage3_run_manifest.json"
    write_json_atomic(run_manifest_path, run_manifest)
    artifact_hashes[run_manifest_path.name] = verify_json_round_trip(
        run_manifest_path, run_manifest
    )

    source_paths = (
        "src/statistical_research/tsay_access.py",
        "src/statistical_research/tsay_artifacts.py",
        "src/statistical_research/tsay_feature_evaluation.py",
        "src/statistical_research/tsay_feature_registry.py",
        "scripts/run_tsay_feature_research.py",
        "scripts/build_tsay_feature_research_notebook.py",
        "project_docs/tsay_feature_research/tsay_feature_research_contract_v1.md",
        "project_docs/tsay_feature_research/tsay_stage2_numerical_compatibility_amendment_v1.md",
        "reports/statistical_research/summaries/tsay_feature_research_status.md",
        "tests/test_tsay_access.py",
        "tests/test_tsay_feature_evaluation.py",
        "tests/test_tsay_notebook_builder.py",
        "tests/test_tsay_stage3_runner.py",
    )
    stage3_hash_manifest = {
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "stage2_hash_manifest_sha256": prior_verification["stage2_hash_manifest_sha256"],
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "source_sha256": {relative: sha256_file(ROOT / relative) for relative in source_paths},
        "access_manifest_development_cutoff_verified": True,
        "development_label_access_only": True,
        "validation_label_access_absent": True,
        "mgc_access_absent": True,
        "post_2023_materialization_absent": True,
    }
    hash_path = ARTIFACT_ROOT / "stage3_hash_manifest.json"
    write_json_atomic(hash_path, stage3_hash_manifest)
    verify_json_round_trip(hash_path, stage3_hash_manifest)

    status_counts = {
        str(key): int(value)
        for key, value in result.evidence_ledger["status"].value_counts().items()
    }
    checkpoint = {
        "stage": 3,
        "stage_name": "development_only_confirmatory_feature_evidence",
        "status": "READY",
        "integrity_reasons": [],
        "research_verdict": "NOT_AUTHORIZED",
        "contract_sha256": stage3_hash_manifest["contract_sha256"],
        "amendment_sha256": stage3_hash_manifest["amendment_sha256"],
        "stage2_hash_manifest_sha256": prior_verification["stage2_hash_manifest_sha256"],
        "stage3_hash_manifest_sha256": sha256_file(hash_path),
        "evidence_ledger_sha256": artifact_hashes["development_feature_evidence_ledger.parquet"],
        "access_manifest_sha256": artifact_hashes["access_manifest_stage3.json"],
        "input_manifest_sha256": artifact_hashes["stage3_input_manifest.json"],
        "label_qa_sha256": artifact_hashes["development_label_atr_tick_qa.parquet"],
        "confirmatory_test_count": len(result.evidence_ledger),
        "feature_status_counts": status_counts,
        "development_label_values_read": True,
        "validation_label_values_read": False,
        "validation_performance_evaluated": False,
        "mgc_research_rows_read": False,
        "post_2023_rows_materialized": False,
        "models_fitted": False,
        "stage4_authorized": True,
        "stage4_authorization_scope": (
            "Fixed Development nested model ladder only, after the user says continue."
        ),
    }
    checkpoint_path = ARTIFACT_ROOT / "stage3_checkpoint.json"
    write_json_atomic(checkpoint_path, checkpoint)
    verify_json_round_trip(checkpoint_path, checkpoint)
    return checkpoint


def verify_stage3_for_stage4() -> dict[str, object]:
    """Verify the completed Stage 3 checkpoint before model fitting."""

    checkpoint_path = ARTIFACT_ROOT / "stage3_checkpoint.json"
    manifest_path = ARTIFACT_ROOT / "stage3_hash_manifest.json"
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Stage 3 checkpoint or hash manifest is absent.")
    checkpoint = read_json(checkpoint_path)
    manifest = read_json(manifest_path)
    if checkpoint.get("status") != "READY" or not checkpoint.get("stage4_authorized"):
        raise RuntimeError("Stage 3 did not authorize the frozen Stage 4 ladder.")
    if sha256_file(manifest_path) != checkpoint.get("stage3_hash_manifest_sha256"):
        raise RuntimeError("Stage 3 hash-manifest identity changed.")
    if sha256_file(CONTRACT_PATH) != manifest.get("contract_sha256"):
        raise RuntimeError("The Stage 1-frozen contract hash changed.")
    if sha256_file(AMENDMENT_PATH) != manifest.get("amendment_sha256"):
        raise RuntimeError("The Stage 2 amendment hash changed.")
    verified_artifacts = 0
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = ARTIFACT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Stage 3 artifact hash changed: {name}")
        verified_artifacts += 1
    immutable_sources = (
        "src/statistical_research/tsay_artifacts.py",
        "src/statistical_research/tsay_feature_evaluation.py",
        "src/statistical_research/tsay_feature_registry.py",
        "project_docs/tsay_feature_research/tsay_feature_research_contract_v1.md",
        "project_docs/tsay_feature_research/tsay_stage2_numerical_compatibility_amendment_v1.md",
    )
    for relative in immutable_sources:
        if sha256_file(ROOT / relative) != manifest["source_sha256"][relative]:
            raise RuntimeError(f"Stage 3 immutable source hash changed: {relative}")
    access = read_json(ARTIFACT_ROOT / "access_manifest_stage3.json")
    _assert_stage3_access(access)
    if sha256_file(ARTIFACT_ROOT / "access_manifest_stage3.json") != checkpoint.get(
        "access_manifest_sha256"
    ):
        raise RuntimeError("Stage 3 access-manifest hash changed.")
    return {
        "status": "VERIFIED",
        "stage3_status": checkpoint["status"],
        "stage3_hash_manifest_sha256": checkpoint["stage3_hash_manifest_sha256"],
        "contract_sha256": checkpoint["contract_sha256"],
        "amendment_sha256": checkpoint["amendment_sha256"],
        "access_manifest_sha256": checkpoint["access_manifest_sha256"],
        "artifact_hashes_verified": verified_artifacts,
        "immutable_source_hashes_verified": len(immutable_sources),
    }


def _assert_stage4_access(payload: dict[str, object]) -> None:
    assert_access_manifest_cutoff(payload)
    if payload.get("stage") != 4:
        raise RuntimeError("Stage 4 access manifest has the wrong stage.")
    if payload.get("materialization_cutoff_date_ny") != "2023-12-31":
        raise RuntimeError("Stage 4 access did not use the Development cutoff.")
    records = payload.get("records", [])
    label_records = [
        record
        for record in records
        if str(record.get("source", "")).endswith("forward_labels_gc.parquet")
        and record.get("access_kind") == "PARQUET_READ"
    ]
    if len(label_records) != 1:
        raise RuntimeError("Stage 4 requires exactly one guarded Development label read.")
    if tuple(label_records[0].get("requested_columns", ())) != STAGE4_LABEL_COLUMNS:
        raise RuntimeError("Stage 4 label columns differ from the exact frozen allowlist.")
    if label_records[0].get("returned_research_partition_set") not in (
        ["Development"],
        ("Development",),
    ):
        raise RuntimeError("Stage 4 label access was not Development-only.")
    forbidden = (
        "scalar_features_validation_sealed_gc.parquet",
        "research_bars_gc_mgc_1m.parquet",
    )
    for record in records:
        source = str(record.get("source", ""))
        if any(source.endswith(name) for name in forbidden):
            raise RuntimeError(f"Forbidden Stage 4 source access: {source}")
        if (
            record.get("returned_maximum_date") is not None
            and str(record["returned_maximum_date"]) > "2023-12-31"
        ):
            raise RuntimeError(f"Post-Development Stage 4 row materialized: {record}")
        partitions = record.get("returned_research_partition_set")
        if partitions not in (None, [], (), ["Development"], ("Development",)):
            raise RuntimeError(f"Non-Development Stage 4 partition materialized: {record}")
        products = record.get("returned_product_set")
        if products not in (None, [], (), ["GC"], ("GC",)):
            raise RuntimeError(f"Non-GC Stage 4 product materialized: {record}")


def run_stage4() -> dict[str, object]:
    """Fit, gate, and freeze only the prescribed Development model ladder."""

    os.environ["PROJECT_COMPUTE"] = "cpu"
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"Stage 4 must run on milestone branch {EXPECTED_BRANCH!r}.")
    prior = verify_stage3_for_stage4()
    actual_environment = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "sklearn": sklearn.__version__,
    }
    expected_environment = {
        "numpy": "2.5.1",
        "pandas": "3.0.3",
        "scipy": "1.18.0",
        "pyarrow": "25.0.0",
        "sklearn": "1.9.0",
    }
    if actual_environment != expected_environment:
        raise RuntimeError(
            f"Stage 4 numerical environment differs from pinned versions: {actual_environment}"
        )

    feature_inputs: set[str] = {
        "atr_20",
        "realized_volatility_15",
        "return_1m_bps",
    }
    for session in SESSIONS:
        feature_inputs.update(D1_INPUTS_BY_SESSION[session])
        feature_inputs.update(O1_INPUTS_BY_SESSION[session])
    tsay_feature_names = {spec.feature_name for spec in TSAY_FEATURE_SPECS}
    fes_inputs = tuple(sorted(feature_inputs & set(FES_STAGE3_CONTROLS)))
    canonical_inputs = tuple(sorted(feature_inputs - tsay_feature_names - set(fes_inputs)))
    tsay_columns = (
        "observation_id",
        "decision_timestamp_utc",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "product",
        "continuous_segment_id",
        *(spec.feature_name for spec in TSAY_FEATURE_SPECS),
        "tsay_roll_spread_identified_120",
        "tsay_clock_z_volume",
        "tsay_clock_z_range",
    )
    canonical_columns = (
        "observation_id",
        "decision_timestamp_utc",
        "trade_date_ny",
        "research_partition",
        "product",
        *canonical_inputs,
    )
    loader = TsayGuardedLoader(ROOT, stage=4)
    tsay = loader.read_parquet(
        ARTIFACT_ROOT.relative_to(ROOT) / "feature_matrix_tsay_gc.parquet",
        columns=tsay_columns,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    canonical = loader.read_parquet(
        TRUSTED_INPUTS["features"],
        columns=canonical_columns,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    fes_columns = (*FES_IDENTITY_COLUMNS, *fes_inputs)
    fes = loader.read_parquet(
        TRUSTED_INPUTS["fes_development"],
        columns=fes_columns,
        mode=AccessMode.GC_MANIFEST_TIMESERIES,
        gc_manifest_verified=True,
    )
    labels = loader.read_parquet(
        TRUSTED_INPUTS["labels"],
        columns=STAGE4_LABEL_COLUMNS,
        mode=AccessMode.TIMESERIES_WITH_PRODUCT,
    )
    _verify_stage3_identity(
        tsay,
        canonical,
        source_name="canonical feature matrix",
        identity_columns=(
            "observation_id",
            "decision_timestamp_utc",
            "trade_date_ny",
            "research_partition",
            "product",
        ),
    )
    _verify_stage3_identity(
        tsay,
        labels,
        source_name="Development labels",
        identity_columns=(
            "observation_id",
            "decision_timestamp_utc",
            "trade_date_ny",
            "entry_session",
            "research_partition",
            "product",
        ),
    )
    _verify_stage3_identity(
        tsay,
        fes,
        source_name="FES Development scalar features",
        identity_columns=FES_IDENTITY_COLUMNS,
    )
    frame = (
        tsay.merge(
            canonical.loc[:, ("observation_id", *canonical_inputs)],
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            fes.loc[:, ("observation_id", *fes_inputs)],
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            labels.loc[
                :,
                (
                    "observation_id",
                    "forward_return_60_atr",
                    "future_range_60_atr",
                    "expansion_label_60",
                    "label_available_60",
                    "exit_timestamp_utc_60",
                ),
            ],
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
    )
    state = pd.to_numeric(frame["tsay_roll_spread_identified_120"], errors="raise").to_numpy(
        np.int8
    )
    frame["tsay_roll_spread_state_valid_not_identified_120"] = (state == 0).astype(np.float64)
    frame["tsay_roll_spread_state_unavailable_120"] = (state == -1).astype(np.float64)
    frame = frame.loc[frame["label_available_60"].fillna(False).astype(bool)].copy()
    if len(frame) == 0:
        raise RuntimeError("No eligible Stage 4 Development labels remain.")

    folds = read_json(ARTIFACT_ROOT / "fold_memberships.json")
    started = perf_counter()
    with _PeakMemoryMonitor() as memory:
        result = run_stage4_models(frame, folds)
    elapsed = perf_counter() - started

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    frames = {
        "development_model_oof_predictions.parquet": result.oof_predictions,
        "development_model_tuning_audit.parquet": result.tuning_audit,
        "development_model_fold_audit.parquet": result.fold_audit,
        "development_model_coefficient_audit.parquet": result.coefficient_audit,
        "development_continuous_model_evidence.parquet": result.continuous_evidence,
        "development_classifier_model_evidence.parquet": result.classifier_evidence,
        "development_model_daily_improvements.parquet": result.daily_improvements,
        "development_model_baseline_diagnostics.parquet": result.baseline_diagnostics,
        "development_continuous_model_year_evidence.parquet": result.continuous_year_evidence,
        "development_classifier_reliability.parquet": result.classifier_reliability,
        "development_classifier_year_evidence.parquet": result.classifier_year_evidence,
        "development_quantile_diagnostics.parquet": result.quantile_diagnostics,
        "development_quantile_year_diagnostics.parquet": result.quantile_year_diagnostics,
        "development_var_diagnostics.parquet": result.var_diagnostics,
        "development_kalman_diagnostics.parquet": result.kalman_diagnostics,
        "development_model_max_stat_audit.parquet": result.max_stat_audit,
        "development_architecture_selection.parquet": result.architecture_selection,
    }
    artifact_hashes: dict[str, str] = {}
    for name, artifact in frames.items():
        path = ARTIFACT_ROOT / name
        save_parquet_atomic(path, artifact)
        artifact_hashes[name] = verify_parquet_round_trip(path, artifact)

    selected = {
        str(row.role): row.selected_architecture
        for row in result.architecture_selection.itertuples(index=False)
    }
    directional_authorized = selected.get("directional") is not None
    opportunity_authorized = selected.get("opportunity") is not None
    expansion_authorized = selected.get("expansion") is not None
    if directional_authorized:
        raise RuntimeError(
            "A directional architecture passed, so the frozen real-bar policy batch is required "
            "before Stage 4 can close."
        )
    policy_state = frozen_no_policy()
    policy_path = ARTIFACT_ROOT / "frozen_policy_state.json"
    write_json_atomic(policy_path, policy_state)
    artifact_hashes[policy_path.name] = verify_json_round_trip(policy_path, policy_state)
    deployment_path = ARTIFACT_ROOT / "stage4_deployment_state.json"
    write_json_atomic(deployment_path, dict(result.deployment_objects))
    artifact_hashes[deployment_path.name] = verify_json_round_trip(
        deployment_path, dict(result.deployment_objects)
    )
    continuation_path = ARTIFACT_ROOT / "stage3_continuation_verification_stage4.json"
    write_json_atomic(continuation_path, prior)
    artifact_hashes[continuation_path.name] = verify_json_round_trip(continuation_path, prior)

    access_path = ARTIFACT_ROOT / "access_manifest_stage4.json"
    loader.write_manifest(access_path)
    access_payload = read_json(access_path)
    _assert_stage4_access(access_payload)
    artifact_hashes[access_path.name] = sha256_file(access_path)

    run_manifest = {
        "stage": 4,
        "compute": "deterministic CPU float64",
        "environment": {
            "python": platform.python_version(),
            **actual_environment,
            "project_compute": os.environ["PROJECT_COMPUTE"],
        },
        "elapsed_seconds": elapsed,
        "peak_process_rss_bytes": memory.peak_rss_bytes,
        "authorized_partition": "Development",
        "maximum_trade_date_ny": "2023-12-31",
        "label_columns_read": list(STAGE4_LABEL_COLUMNS),
        "validation_label_values_read": False,
        "validation_performance_evaluated": False,
        "post_2023_rows_materialized": False,
        "mgc_rows_materialized": False,
        "real_bar_data_loaded": False,
        "strategy_pnl_calculated": False,
        "architecture_shared_across_session_specific_fits": True,
        "role_selection_score": "datewise mean of the two session improvements on the common panel",
    }
    run_path = ARTIFACT_ROOT / "stage4_run_manifest.json"
    write_json_atomic(run_path, run_manifest)
    artifact_hashes[run_path.name] = verify_json_round_trip(run_path, run_manifest)

    source_paths = (
        "src/statistical_research/tsay_access.py",
        "src/statistical_research/tsay_artifacts.py",
        "src/statistical_research/tsay_backtest.py",
        "src/statistical_research/tsay_feature_registry.py",
        "src/statistical_research/tsay_models.py",
        "scripts/run_tsay_feature_research.py",
        "scripts/build_tsay_feature_research_notebook.py",
        "project_docs/tsay_feature_research/tsay_feature_research_contract_v1.md",
        "project_docs/tsay_feature_research/tsay_stage2_numerical_compatibility_amendment_v1.md",
        "reports/statistical_research/summaries/tsay_feature_research_status.md",
        "tests/test_tsay_access.py",
        "tests/test_tsay_backtest.py",
        "tests/test_tsay_models.py",
        "tests/test_tsay_notebook_builder.py",
        "tests/test_tsay_stage4_runner.py",
    )
    hash_manifest = {
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "stage3_hash_manifest_sha256": prior["stage3_hash_manifest_sha256"],
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "source_sha256": {relative: sha256_file(ROOT / relative) for relative in source_paths},
        "access_manifest_development_cutoff_verified": True,
        "development_label_access_only": True,
        "validation_label_access_absent": True,
        "mgc_access_absent": True,
        "post_2023_materialization_absent": True,
    }
    hash_path = ARTIFACT_ROOT / "stage4_hash_manifest.json"
    write_json_atomic(hash_path, hash_manifest)
    verify_json_round_trip(hash_path, hash_manifest)
    checkpoint = {
        "stage": 4,
        "stage_name": "development_nested_models_oof_economics_and_freeze",
        "status": "READY",
        "integrity_reasons": [],
        "research_verdict": (
            "OPPORTUNITY_ONLY_AUTHORIZED"
            if opportunity_authorized and not directional_authorized
            else "NO_PREDICTIVE_MODEL_AUTHORIZED"
        ),
        "contract_sha256": hash_manifest["contract_sha256"],
        "amendment_sha256": hash_manifest["amendment_sha256"],
        "stage3_hash_manifest_sha256": prior["stage3_hash_manifest_sha256"],
        "stage4_hash_manifest_sha256": sha256_file(hash_path),
        "access_manifest_sha256": artifact_hashes["access_manifest_stage4.json"],
        "deployment_state_sha256": artifact_hashes["stage4_deployment_state.json"],
        "policy_state_sha256": artifact_hashes["frozen_policy_state.json"],
        "predictive_model_authorization": any(
            (directional_authorized, opportunity_authorized, expansion_authorized)
        ),
        "directional_model_authorized": directional_authorized,
        "opportunity_only_authorized": (opportunity_authorized and not directional_authorized),
        "opportunity_model_authorized": opportunity_authorized,
        "expansion_model_authorized": expansion_authorized,
        "policy_authorized": False,
        "policy_status": policy_state["policy_status"],
        "validation_economics_may_open": False,
        "validation_outcomes_read": False,
        "mgc_research_rows_read": False,
        "post_2023_rows_materialized": False,
        "stage5_authorized": any(
            (directional_authorized, opportunity_authorized, expansion_authorized)
        ),
    }
    checkpoint_path = ARTIFACT_ROOT / "stage4_checkpoint.json"
    write_json_atomic(checkpoint_path, checkpoint)
    verify_json_round_trip(checkpoint_path, checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    if args.stage == 1:
        checkpoint = run_stage1()
    elif args.stage == 2:
        checkpoint = run_stage2()
    elif args.stage == 3:
        checkpoint = run_stage3()
    elif args.stage == 4:
        checkpoint = run_stage4()
    else:
        raise SystemExit("Only Stages 1 through 4 are implemented.")
    print(f"STATUS: {checkpoint['status']}")
    print(f"RESEARCH_VERDICT: {checkpoint['research_verdict']}")
    if args.stage == 2:
        print(f"STAGE_3_AUTHORIZED: {checkpoint['stage3_authorized']}")
    if args.stage == 3:
        print(f"STAGE_4_AUTHORIZED: {checkpoint['stage4_authorized']}")
    if args.stage == 4:
        print(f"STAGE_5_AUTHORIZED: {checkpoint['stage5_authorized']}")


if __name__ == "__main__":
    main()
