"""Validation and overlap audits for the frozen MLAT feature batch.

The functions in this module intentionally return ordinary pandas tables so a
research notebook can persist every check independently.  They do not write
artifacts or rely on notebook state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_METADATA_COLUMNS = (
    "observation_id",
    "decision_timestamp_utc",
    "decision_timestamp_ny",
    "entry_timestamp_utc",
    "entry_timestamp_ny",
    "trade_date_ny",
    "research_partition",
    "product",
    "symbol",
    "active_symbol",
    "instrument_id",
    "continuous_segment_id",
)
DEFAULT_SAMPLE_SIZE = 257
NEAR_DUPLICATE_THRESHOLD = 0.995
DEFAULT_MINIMUM_OVERLAP_OBSERVATIONS = 30
OVERLAP_AUDIT_COLUMNS = (
    "candidate_feature",
    "reference_feature",
    "reference_scope",
    "audit_type",
    "development_observations",
    "minimum_observations",
    "sufficient_overlap",
    "spearman_correlation",
    "absolute_correlation",
    "is_exact_duplicate",
    "is_near_duplicate",
    "is_constant",
    "threshold",
)


@dataclass(frozen=True)
class MlatValidationResult:
    """Independently saveable outputs from the MLAT validation gate."""

    checks: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    coverage: pd.DataFrame
    overlap_audit: pd.DataFrame
    sample_hash: pd.DataFrame

    @property
    def ready(self) -> bool:
        """Whether every error-level validation check passed."""

        errors = self.checks["severity"].eq("ERROR")
        return bool(self.checks.loc[errors, "passed"].all())


def _check(
    rows: list[dict[str, object]],
    name: str,
    passed: bool,
    *,
    observed: object,
    expected: object,
    details: str = "",
    severity: str = "ERROR",
) -> None:
    rows.append(
        {
            "check": name,
            "passed": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "details": details,
        }
    )


def _registry_features(registry: pd.DataFrame) -> list[str]:
    if "feature_name" not in registry:
        raise KeyError("registry must contain feature_name")
    names = registry["feature_name"].astype(str).tolist()
    if len(names) != len(set(names)):
        raise ValueError("registry feature_name values must be unique")
    return names


def _stable_scalar(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.17g}"
    return str(value)


def deterministic_sample_hash(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> str:
    """Hash deterministic, evenly spaced rows plus schema and row count."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    selected = list(columns) if columns is not None else frame.columns.tolist()
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise KeyError(f"hash columns missing from frame: {missing}")
    count = min(sample_size, len(frame))
    positions = (
        np.linspace(0, len(frame) - 1, count, dtype=np.int64)
        if count
        else np.empty(0, dtype=np.int64)
    )
    digest = sha256()
    digest.update(f"rows={len(frame)}|sample={count}".encode())
    for column in selected:
        digest.update(f"|{column}:{frame[column].dtype}".encode())
    for position in positions:
        digest.update(f"\n@{int(position)}".encode())
        row = frame.iloc[int(position)]
        for column in selected:
            digest.update(b"\x1f")
            digest.update(_stable_scalar(row[column]).encode("utf-8"))
    return digest.hexdigest()


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.array_equal(left_values, right_values, equal_nan=True))


def _development_mask(frame: pd.DataFrame, partition_column: str) -> np.ndarray:
    if partition_column not in frame:
        return np.ones(len(frame), dtype=bool)
    return frame[partition_column].astype(str).eq("Development").to_numpy()


def audit_feature_overlap(
    candidate_matrix: pd.DataFrame,
    existing_feature_matrix: pd.DataFrame | None = None,
    *,
    candidate_features: Iterable[str] | None = None,
    existing_features: Iterable[str] | None = None,
    partition_column: str = "research_partition",
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    minimum_observations: int = DEFAULT_MINIMUM_OVERLAP_OBSERVATIONS,
) -> pd.DataFrame:
    """Audit constants and exact/near duplicates on Development observations.

    Existing features are aligned by ``observation_id`` when both matrices
    expose it.  Candidate-to-candidate and candidate-to-existing comparisons
    are reported in one long-form table.
    """

    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2")
    candidate_names = (
        list(candidate_features)
        if candidate_features is not None
        else [
            column
            for column in candidate_matrix.select_dtypes(include=[np.number]).columns
            if column != "observation_id"
        ]
    )
    missing_candidates = sorted(set(candidate_names).difference(candidate_matrix.columns))
    if missing_candidates:
        raise KeyError(f"candidate features missing: {missing_candidates}")

    candidate = candidate_matrix.copy()
    existing = existing_feature_matrix
    if existing is not None and "observation_id" in candidate and "observation_id" in existing:
        existing_names = (
            list(existing_features)
            if existing_features is not None
            else [
                column
                for column in existing.select_dtypes(include=[np.number]).columns
                if column != "observation_id"
            ]
        )
        candidate = candidate.merge(
            existing[["observation_id", *existing_names]],
            on="observation_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "__existing"),
        )
        mapped_existing = [
            name if name not in candidate_names else f"{name}__existing" for name in existing_names
        ]
    elif existing is not None:
        if len(candidate) != len(existing):
            raise ValueError("matrices without observation_id must have equal row counts")
        existing_names = (
            list(existing_features)
            if existing_features is not None
            else list(existing.select_dtypes(include=[np.number]).columns)
        )
        mapped_existing = []
        for name in existing_names:
            mapped = name if name not in candidate else f"{name}__existing"
            candidate[mapped] = existing[name].to_numpy()
            mapped_existing.append(mapped)
    else:
        existing_names = []
        mapped_existing = []

    mask = _development_mask(candidate, partition_column)
    rows: list[dict[str, object]] = []
    numeric: dict[str, np.ndarray] = {
        name: pd.to_numeric(candidate.loc[mask, name], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        for name in candidate_names
    }
    for name, values in numeric.items():
        finite = values[np.isfinite(values)]
        unique = int(np.unique(finite).size)
        rows.append(
            {
                "candidate_feature": name,
                "reference_feature": pd.NA,
                "reference_scope": "candidate",
                "audit_type": "constant",
                "development_observations": int(finite.size),
                "minimum_observations": int(minimum_observations),
                "sufficient_overlap": bool(finite.size >= minimum_observations),
                "spearman_correlation": np.nan,
                "absolute_correlation": np.nan,
                "is_exact_duplicate": False,
                "is_near_duplicate": False,
                "is_constant": unique <= 1,
                "threshold": float(near_duplicate_threshold),
            }
        )

    comparison_values: dict[str, np.ndarray] = dict(numeric)
    mapped_display: dict[str, str] = {}
    for display_name, mapped_name in zip(existing_names, mapped_existing, strict=True):
        key = f"existing::{mapped_name}"
        comparison_values[key] = pd.to_numeric(
            candidate.loc[mask, mapped_name], errors="coerce"
        ).to_numpy(dtype=float, na_value=np.nan)
        mapped_display[display_name] = key

    comparisons: list[tuple[str, str, str, str, str]] = []
    for index, left_name in enumerate(candidate_names):
        for right_name in candidate_names[index + 1 :]:
            comparisons.append(
                (left_name, right_name, "candidate", left_name, right_name)
            )
        for display_name in existing_names:
            comparisons.append(
                (
                    left_name,
                    display_name,
                    "existing",
                    left_name,
                    mapped_display[display_name],
                )
            )

    pair_records: list[dict[str, object]] = []
    mask_groups: dict[bytes, dict[str, object]] = {}
    for left_name, right_name, scope, left_key, right_key in comparisons:
        left = comparison_values[left_key]
        right = comparison_values[right_key]
        valid = np.isfinite(left) & np.isfinite(right)
        count = int(valid.sum())
        sufficient = count >= minimum_observations
        exact = bool(
            sufficient
            and np.array_equal(left[valid], right[valid], equal_nan=True)
        )
        record = {
            "candidate_feature": left_name,
            "reference_feature": right_name,
            "reference_scope": scope,
            "audit_type": "overlap",
            "development_observations": count,
            "minimum_observations": int(minimum_observations),
            "sufficient_overlap": bool(sufficient),
            "spearman_correlation": np.nan,
            "absolute_correlation": np.nan,
            "is_exact_duplicate": exact,
            "is_near_duplicate": exact,
            "is_constant": False,
            "threshold": float(near_duplicate_threshold),
        }
        pair_index = len(pair_records)
        pair_records.append(record)
        if sufficient:
            packed = np.packbits(valid, bitorder="little")
            group_key = sha256(packed.tobytes()).digest()
            group = mask_groups.setdefault(
                group_key,
                {"mask": valid, "columns": set(), "pairs": []},
            )
            if not np.array_equal(group["mask"], valid):
                group_key = sha256(packed.tobytes() + pair_index.to_bytes(8, "little")).digest()
                group = mask_groups.setdefault(
                    group_key,
                    {"mask": valid, "columns": set(), "pairs": []},
                )
            group["columns"].update((left_key, right_key))
            group["pairs"].append((pair_index, left_key, right_key))

    for group in mask_groups.values():
        valid = group["mask"]
        columns = sorted(group["columns"])
        column_positions = {name: index for index, name in enumerate(columns)}
        values = np.column_stack([comparison_values[name][valid] for name in columns])
        ranked = rankdata(values, axis=0, method="average")
        correlations = np.corrcoef(ranked, rowvar=False)
        correlations = np.atleast_2d(correlations)
        for pair_index, left_key, right_key in group["pairs"]:
            correlation = correlations[
                column_positions[left_key], column_positions[right_key]
            ]
            if not np.isfinite(correlation):
                continue
            absolute = abs(float(correlation))
            pair_records[pair_index]["spearman_correlation"] = float(correlation)
            pair_records[pair_index]["absolute_correlation"] = absolute
            pair_records[pair_index]["is_near_duplicate"] = bool(
                pair_records[pair_index]["is_exact_duplicate"]
                or absolute >= near_duplicate_threshold
            )

    rows.extend(pair_records)
    return pd.DataFrame(rows, columns=OVERLAP_AUDIT_COLUMNS)


def _dtype_matches(series: pd.Series, expected: object) -> bool:
    text = str(expected).lower()
    if "float" in text:
        return pd.api.types.is_float_dtype(series.dtype)
    if "int" in text:
        return pd.api.types.is_integer_dtype(series.dtype)
    if "bool" in text:
        return pd.api.types.is_bool_dtype(series.dtype)
    if "timestamp" in text or "datetime" in text:
        return pd.api.types.is_datetime64_any_dtype(series.dtype)
    if "string" in text or text in {"str", "object"}:
        return pd.api.types.is_string_dtype(series.dtype) or series.dtype == object
    return str(series.dtype).lower() == text


def _feature_diagnostics(
    matrix: pd.DataFrame, registry: pd.DataFrame, features: list[str], max_missing_rate: float
) -> pd.DataFrame:
    indexed = registry.set_index("feature_name", drop=False)
    rows: list[dict[str, object]] = []
    for name in features:
        numeric = pd.to_numeric(matrix[name], errors="coerce")
        raw_null = matrix[name].isna()
        finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        missing_rate = float(raw_null.mean())
        expected_dtype = indexed.at[name, "output_dtype"] if "output_dtype" in indexed else pd.NA
        lower = (
            pd.to_numeric(pd.Series([indexed.at[name, "validation_minimum"]]), errors="coerce").iloc[0]
            if "validation_minimum" in indexed
            else np.nan
        )
        upper = (
            pd.to_numeric(pd.Series([indexed.at[name, "validation_maximum"]]), errors="coerce").iloc[0]
            if "validation_maximum" in indexed
            else np.nan
        )
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        below = int(np.sum(finite & (values < lower))) if np.isfinite(lower) else 0
        above = int(np.sum(finite & (values > upper))) if np.isfinite(upper) else 0
        rows.append(
            {
                "feature_name": name,
                "dtype": str(matrix[name].dtype),
                "expected_dtype": expected_dtype,
                "dtype_pass": (
                    _dtype_matches(matrix[name], expected_dtype)
                    if not pd.isna(expected_dtype)
                    else pd.api.types.is_numeric_dtype(matrix[name])
                ),
                "null_count": int(raw_null.sum()),
                "missing_rate": missing_rate,
                "missing_rate_pass": missing_rate <= max_missing_rate,
                "nonfinite_count": int((~finite & ~raw_null.to_numpy()).sum()),
                "minimum": float(np.nanmin(values)) if finite.any() else np.nan,
                "maximum": float(np.nanmax(values)) if finite.any() else np.nan,
                "validation_minimum": lower,
                "validation_maximum": upper,
                "below_minimum_count": below,
                "above_maximum_count": above,
                "range_pass": below == 0 and above == 0,
                "finite_unique_count": int(pd.Series(values[finite]).nunique()),
                "constant": int(pd.Series(values[finite]).nunique()) <= 1,
            }
        )
    return pd.DataFrame(rows)


def _coverage_table(matrix: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    group_columns = [
        column
        for column in ("research_partition", "entry_session")
        if column in matrix.columns
    ]
    if not group_columns:
        groups = [((), matrix)]
    else:
        groups = list(matrix.groupby(group_columns, observed=True, dropna=False, sort=True))
    rows: list[dict[str, object]] = []
    for key, group in groups:
        keys = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_columns, keys, strict=True))
        for feature in features:
            numeric = pd.to_numeric(group[feature], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            rows.append(
                {
                    **base,
                    "feature_name": feature,
                    "observations": int(len(group)),
                    "finite_observations": int(np.isfinite(numeric).sum()),
                    "coverage_rate": float(np.isfinite(numeric).mean()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def validate_mlat_feature_matrix(
    matrix: pd.DataFrame,
    registry: pd.DataFrame,
    eligible_observations: pd.DataFrame | None = None,
    existing_feature_matrix: pd.DataFrame | None = None,
    *,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    max_missing_rate: float = 0.05,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    minimum_overlap_observations: int = DEFAULT_MINIMUM_OVERLAP_OBSERVATIONS,
    include_existing_overlap: bool = False,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> MlatValidationResult:
    """Run the frozen MLAT registry, schema, value, and overlap gates."""

    if not 0 <= max_missing_rate <= 1:
        raise ValueError("max_missing_rate must be in [0, 1]")
    features = _registry_features(registry)
    metadata = list(metadata_columns)
    missing_metadata = sorted(set(metadata).difference(matrix.columns))
    missing_features = sorted(set(features).difference(matrix.columns))
    unexpected = [
        column for column in matrix.columns if column not in set(metadata).union(features)
    ]
    expected_columns = [*metadata, *features]
    checks: list[dict[str, object]] = []
    _check(
        checks,
        "registry_matrix_column_set",
        not missing_metadata and not missing_features and not unexpected,
        observed={
            "missing_metadata": missing_metadata,
            "missing_features": missing_features,
            "unexpected": unexpected,
        },
        expected="exact metadata plus frozen registry feature set",
    )
    _check(
        checks,
        "registry_feature_order",
        matrix.columns.tolist() == expected_columns,
        observed=matrix.columns.tolist(),
        expected=expected_columns,
    )
    if "observation_id" in matrix:
        ids = matrix["observation_id"]
        _check(
            checks,
            "observation_id_unique",
            not ids.duplicated().any(),
            observed=int(ids.duplicated().sum()),
            expected=0,
        )
        _check(
            checks,
            "observation_id_order",
            ids.is_monotonic_increasing,
            observed=bool(ids.is_monotonic_increasing),
            expected=True,
        )
    else:
        _check(
            checks,
            "observation_id_present",
            False,
            observed=False,
            expected=True,
        )

    if eligible_observations is not None and "observation_id" in matrix:
        if "observation_id" not in eligible_observations:
            raise KeyError("eligible_observations must contain observation_id")
        expected_ids = eligible_observations["observation_id"].reset_index(drop=True)
        actual_ids = matrix["observation_id"].reset_index(drop=True)
        _check(
            checks,
            "eligible_observation_id_alignment",
            actual_ids.equals(expected_ids),
            observed={"matrix_rows": len(actual_ids), "eligible_rows": len(expected_ids)},
            expected="identical IDs in identical order",
        )
        timestamp_columns = [
            column
            for column in (
                "decision_timestamp_utc",
                "decision_timestamp_ny",
                "entry_timestamp_utc",
                "entry_timestamp_ny",
                "trade_date_ny",
                "research_partition",
                "product",
            )
            if column in matrix and column in eligible_observations
        ]
        for column in timestamp_columns:
            left = matrix[column].reset_index(drop=True)
            right = eligible_observations[column].reset_index(drop=True)
            equal = left.astype(str).equals(right.astype(str))
            _check(
                checks,
                f"eligible_{column}_alignment",
                equal,
                observed=int((left.astype(str) != right.astype(str)).sum()),
                expected=0,
            )

    if "product" in matrix:
        products = sorted(matrix["product"].dropna().astype(str).unique().tolist())
        _check(
            checks,
            "gc_only",
            products == ["GC"],
            observed=products,
            expected=["GC"],
        )
    for column in ("decision_timestamp_utc", "entry_timestamp_utc"):
        if column in matrix:
            parsed = pd.to_datetime(matrix[column], errors="coerce", utc=True)
            _check(
                checks,
                f"{column}_valid_utc",
                parsed.notna().all(),
                observed=int(parsed.isna().sum()),
                expected=0,
            )
    if {"decision_timestamp_utc", "entry_timestamp_utc"}.issubset(matrix.columns):
        decision = pd.to_datetime(matrix["decision_timestamp_utc"], errors="coerce", utc=True)
        entry = pd.to_datetime(matrix["entry_timestamp_utc"], errors="coerce", utc=True)
        _check(
            checks,
            "decision_precedes_entry",
            bool((decision < entry).all()),
            observed=int((decision >= entry).sum()),
            expected=0,
        )

    available = [name for name in features if name in matrix]
    diagnostics = _feature_diagnostics(matrix, registry, available, max_missing_rate)
    for field, check_name in (
        ("dtype_pass", "feature_dtypes"),
        ("missing_rate_pass", "feature_missing_rate"),
        ("nonfinite_count", "feature_nonfinite_values"),
        ("range_pass", "feature_validation_ranges"),
        ("constant", "feature_constants"),
    ):
        if diagnostics.empty:
            passed = False
            failures = features
        elif field == "nonfinite_count":
            failures = diagnostics.loc[diagnostics[field].gt(0), "feature_name"].tolist()
            passed = not failures
        elif field == "constant":
            failures = diagnostics.loc[diagnostics[field], "feature_name"].tolist()
            passed = not failures
        else:
            failures = diagnostics.loc[~diagnostics[field].astype(bool), "feature_name"].tolist()
            passed = not failures
        _check(
            checks,
            check_name,
            passed,
            observed=failures,
            expected=[],
        )

    coverage = _coverage_table(matrix, available)
    overlap = audit_feature_overlap(
        matrix,
        existing_feature_matrix if include_existing_overlap else None,
        candidate_features=available,
        partition_column="research_partition",
        near_duplicate_threshold=near_duplicate_threshold,
        minimum_observations=minimum_overlap_observations,
    )
    exact_pairs = overlap.loc[
        overlap["audit_type"].eq("overlap")
        & overlap["reference_scope"].eq("candidate")
        & overlap["is_exact_duplicate"],
        ["candidate_feature", "reference_feature"],
    ]
    _check(
        checks,
        "exact_duplicate_audit",
        exact_pairs.empty,
        observed=exact_pairs.to_dict("records"),
        expected=[],
    )
    candidate_near = overlap.loc[
        overlap["audit_type"].eq("overlap")
        & overlap["reference_scope"].eq("candidate")
        & overlap["is_near_duplicate"],
        ["candidate_feature", "reference_feature", "absolute_correlation"],
    ]
    _check(
        checks,
        "candidate_near_duplicate_audit",
        candidate_near.empty,
        observed=candidate_near.to_dict("records"),
        expected=[],
    )
    hash_columns = [column for column in expected_columns if column in matrix]
    digest = deterministic_sample_hash(
        matrix, columns=hash_columns, sample_size=sample_size
    )
    sample_hash = pd.DataFrame(
        [
            {
                "algorithm": "sha256",
                "sample_method": "evenly_spaced_rows",
                "sample_size": min(sample_size, len(matrix)),
                "row_count": len(matrix),
                "column_count": len(matrix.columns),
                "hashed_column_count": len(hash_columns),
                "schema_complete": not missing_metadata and not missing_features,
                "digest": digest,
            }
        ]
    )
    return MlatValidationResult(
        checks=pd.DataFrame(checks),
        feature_diagnostics=diagnostics,
        coverage=coverage,
        overlap_audit=overlap,
        sample_hash=sample_hash,
    )


def verify_reloaded_feature_matrix(
    original: pd.DataFrame,
    reloaded: pd.DataFrame,
    registry: pd.DataFrame | None = None,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Compare an in-memory matrix with its reloaded persisted artifact."""

    rows: list[dict[str, object]] = []
    _check(
        rows,
        "reload_shape",
        original.shape == reloaded.shape,
        observed=reloaded.shape,
        expected=original.shape,
    )
    _check(
        rows,
        "reload_column_order",
        original.columns.tolist() == reloaded.columns.tolist(),
        observed=reloaded.columns.tolist(),
        expected=original.columns.tolist(),
    )
    shared = [column for column in original.columns if column in reloaded]
    mismatched_dtypes = [
        column for column in shared if str(original[column].dtype) != str(reloaded[column].dtype)
    ]
    _check(
        rows,
        "reload_dtypes",
        not mismatched_dtypes,
        observed=mismatched_dtypes,
        expected=[],
    )
    for column in ("observation_id", "decision_timestamp_utc", "entry_timestamp_utc"):
        if column in shared:
            left = original[column].reset_index(drop=True).astype(str)
            right = reloaded[column].reset_index(drop=True).astype(str)
            _check(
                rows,
                f"reload_{column}",
                left.equals(right),
                observed=int((left != right).sum()) if len(left) == len(right) else "row mismatch",
                expected=0,
            )
    null_mismatches = [
        column
        for column in shared
        if int(original[column].isna().sum()) != int(reloaded[column].isna().sum())
    ]
    _check(
        rows,
        "reload_null_counts",
        not null_mismatches,
        observed=null_mismatches,
        expected=[],
    )
    metadata_columns = [
        column for column in DEFAULT_METADATA_COLUMNS if column in shared
    ]
    unequal_metadata = []
    for column in metadata_columns:
        left = original[column].reset_index(drop=True)
        right = reloaded[column].reset_index(drop=True)
        if len(left) != len(right) or not left.equals(right):
            unequal_metadata.append(column)
    _check(
        rows,
        "reload_metadata_values",
        not unequal_metadata,
        observed=unequal_metadata,
        expected=[],
    )
    nonnumeric_columns = [
        column
        for column in shared
        if not pd.api.types.is_numeric_dtype(original[column])
        or not pd.api.types.is_numeric_dtype(reloaded[column])
    ]
    unequal_nonnumeric = []
    for column in nonnumeric_columns:
        left = original[column].reset_index(drop=True)
        right = reloaded[column].reset_index(drop=True)
        if len(left) != len(right) or not left.equals(right):
            unequal_nonnumeric.append(column)
    _check(
        rows,
        "reload_nonnumeric_values",
        not unequal_nonnumeric,
        observed=unequal_nonnumeric,
        expected=[],
    )
    columns = (
        [*DEFAULT_METADATA_COLUMNS, *_registry_features(registry)]
        if registry is not None
        else original.columns.tolist()
    )
    columns = [column for column in columns if column in original and column in reloaded]
    original_hash = deterministic_sample_hash(
        original, columns=columns, sample_size=sample_size
    )
    reloaded_hash = deterministic_sample_hash(
        reloaded, columns=columns, sample_size=sample_size
    )
    _check(
        rows,
        "reload_deterministic_sample_hash",
        original_hash == reloaded_hash,
        observed=reloaded_hash,
        expected=original_hash,
    )
    numeric = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(original[column])
        and pd.api.types.is_numeric_dtype(reloaded[column])
    ]
    unequal_numeric = []
    for column in numeric:
        left = pd.to_numeric(original[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        right = pd.to_numeric(reloaded[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        if len(left) != len(right) or not np.allclose(left, right, equal_nan=True, rtol=0, atol=0):
            unequal_numeric.append(column)
    _check(
        rows,
        "reload_numeric_values",
        not unequal_numeric,
        observed=unequal_numeric,
        expected=[],
    )
    return pd.DataFrame(rows)
