"""Versioned, verifiable artifact persistence for MLAT research notebooks.

The helpers in this module are intentionally independent of the historical
statistical-feature artifact layout.  Tables are written through a temporary
file in the destination directory, described by a deterministic manifest
entry, and can be checked against their in-memory source after a reload.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_extension_array_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)

ARTIFACT_LAYOUT_VERSION = "mlat-artifacts-v1"
DEFAULT_SAMPLE_SIZE = 16
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CSV_NULL_TOKEN = "__MLAT_CSV_NULL_V1__"

__all__ = [
    "ARTIFACT_LAYOUT_VERSION",
    "ArtifactManifestEntry",
    "ArtifactVerificationError",
    "MLATArtifactPaths",
    "load_saved_table",
    "prepare_csv_table",
    "save_csv",
    "save_json",
    "save_parquet",
    "verify_saved_table",
]


class ArtifactVerificationError(RuntimeError):
    """Raised when a saved table does not reload exactly as expected."""

    def __init__(self, path: Path, diagnostics: pd.DataFrame) -> None:
        self.path = Path(path)
        self.diagnostics = diagnostics.copy()
        failed = diagnostics.loc[~diagnostics["passed"], "check_name"].astype(str).tolist()
        super().__init__(f"Artifact verification failed for {self.path}: failed_checks={failed}")


def _safe_component(value: str, *, label: str) -> str:
    value = str(value)
    if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"{label} must contain only letters, numbers, '.', '_', or '-' and "
            "must not contain path traversal."
        )
    return value


@dataclass(frozen=True)
class MLATArtifactPaths:
    """Canonical, versioned output paths for one MLAT notebook run."""

    project_root: Path | str
    version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        object.__setattr__(self, "version", _safe_component(self.version, label="version"))

    @property
    def data_dir(self) -> Path:
        """Return the versioned processed-data directory."""

        return (
            Path(self.project_root)
            / "data"
            / "processed"
            / "statistical_research"
            / "mlat_feature_research"
            / self.version
        )

    @property
    def report_dir(self) -> Path:
        """Return the versioned report directory."""

        return (
            Path(self.project_root)
            / "reports"
            / "statistical_research"
            / "mlat_feature_research"
            / self.version
        )

    @property
    def figure_dir(self) -> Path:
        """Return the versioned figure directory."""

        return (
            Path(self.project_root) / "reports" / "figures" / "mlat_feature_research" / self.version
        )

    @property
    def table_dir(self) -> Path:
        """Return the versioned report-table directory."""

        return self.report_dir / "tables"

    @property
    def manifest_path(self) -> Path:
        """Return the aggregate table-manifest path."""

        return self.report_dir / "table_manifest.json"

    def ensure(self) -> MLATArtifactPaths:
        """Create all output directories idempotently and return ``self``."""

        for directory in (self.data_dir, self.table_dir, self.figure_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def data_path(self, name: str, *, suffix: str = ".parquet") -> Path:
        """Return a safe path for a processed data artifact."""

        return self.data_dir / f"{_safe_component(name, label='name')}{_safe_suffix(suffix)}"

    def table_path(self, name: str, *, suffix: str = ".csv") -> Path:
        """Return a safe path for a report table."""

        return self.table_dir / f"{_safe_component(name, label='name')}{_safe_suffix(suffix)}"

    def json_path(self, name: str) -> Path:
        """Return a safe path for a JSON report artifact."""

        return self.report_dir / f"{_safe_component(name, label='name')}.json"

    def figure_path(self, name: str, *, suffix: str = ".png") -> Path:
        """Return a safe path for a figure artifact."""

        return self.figure_dir / f"{_safe_component(name, label='name')}{_safe_suffix(suffix)}"


def _safe_suffix(suffix: str) -> str:
    value = str(suffix)
    if not value.startswith("."):
        value = f".{value}"
    if not re.fullmatch(r"\.[A-Za-z0-9]+", value):
        raise ValueError("suffix must be a simple extension such as '.parquet'.")
    return value.lower()


@dataclass(frozen=True)
class ArtifactManifestEntry:
    """Serializable provenance and integrity metadata for one artifact."""

    artifact_path: str
    artifact_format: str
    sha256: str
    byte_size: int
    layout_version: str = ARTIFACT_LAYOUT_VERSION
    row_count: int | None = None
    column_count: int | None = None
    columns: tuple[str, ...] = ()
    pandas_dtypes: Mapping[str, str] = field(default_factory=dict)
    arrow_types: Mapping[str, str] = field(default_factory=dict)
    schema_fingerprint: str | None = None
    null_counts: Mapping[str, int] = field(default_factory=dict)
    timestamp_metadata: Mapping[str, Mapping[str, str | None]] = field(default_factory=dict)
    id_columns: tuple[str, ...] = ()
    sample_positions: tuple[int, ...] = ()
    sample_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with deterministic collection types."""

        result = asdict(self)
        result["columns"] = list(self.columns)
        result["id_columns"] = list(self.id_columns)
        result["sample_positions"] = list(self.sample_positions)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactManifestEntry:
        """Reconstruct an entry from an aggregate manifest record."""

        values = dict(payload)
        for key in ("columns", "id_columns", "sample_positions"):
            values[key] = tuple(values.get(key, ()))
        return cls(**values)


def _require_table(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Artifact tables require unique columns.")
    if not all(isinstance(column, str) for column in frame.columns):
        raise ValueError("Artifact table column labels must all be strings.")
    return frame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            default=_json_default,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, float) and not np.isfinite(scalar):
            return None
        return scalar
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sample_positions(length: int, sample_size: int = DEFAULT_SAMPLE_SIZE) -> tuple[int, ...]:
    if length <= 0 or sample_size <= 0:
        return ()
    count = min(length, sample_size)
    return tuple(
        int(position)
        for position in np.unique(np.linspace(0, length - 1, num=count, dtype=np.int64))
    )


def _is_missing(value: Any) -> bool:
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _scalar_token(value: Any) -> Mapping[str, Any]:
    if _is_missing(value):
        return {"kind": "null"}
    if isinstance(value, pd.Timestamp):
        timestamp = value.as_unit("ns")
        return {
            "kind": "timestamp",
            "nanoseconds": int(timestamp.value),
            "timezone": None if timestamp.tz is None else str(timestamp.tz),
        }
    if isinstance(value, pd.Timedelta):
        return {"kind": "timedelta", "nanoseconds": int(value.as_unit("ns").value)}
    if isinstance(value, (bool, np.bool_)):
        return {"kind": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"kind": "integer", "value": str(int(value))}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return {"kind": "float", "value": number.hex()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "bytes", "value": bytes(value).hex()}
    return {"kind": type(value).__name__, "value": str(value)}


def _schema_payload(frame: pd.DataFrame) -> dict[str, Any]:
    try:
        import pyarrow as pa

        arrow_schema = pa.Schema.from_pandas(frame, preserve_index=False)
        arrow_types = {str(field.name): str(field.type) for field in arrow_schema}
        arrow_nullable = {str(field.name): bool(field.nullable) for field in arrow_schema}
    except (ImportError, TypeError, ValueError):
        arrow_types = {column: "unavailable" for column in frame.columns}
        arrow_nullable = {column: True for column in frame.columns}
    return {
        "columns": [
            {
                "name": column,
                "pandas_dtype": str(frame[column].dtype),
                "arrow_type": arrow_types[column],
                "arrow_nullable": arrow_nullable[column],
            }
            for column in frame.columns
        ]
    }


def _schema_fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_canonical_json_bytes(_schema_payload(frame))).hexdigest()


def _sample_hash(
    frame: pd.DataFrame,
    *,
    positions: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], str]:
    selected = (
        tuple(int(position) for position in positions)
        if positions is not None
        else _sample_positions(len(frame))
    )
    payload = {
        "schema_fingerprint": _schema_fingerprint(frame),
        "positions": list(selected),
        "rows": [
            {column: _scalar_token(frame.iloc[position][column]) for column in frame.columns}
            for position in selected
        ],
    }
    return selected, hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _timestamp_metadata(frame: pd.DataFrame) -> dict[str, dict[str, str | None]]:
    metadata: dict[str, dict[str, str | None]] = {}
    for column in frame.columns:
        dtype = frame[column].dtype
        if is_datetime64_any_dtype(dtype):
            timezone = getattr(dtype, "tz", None)
            unit = getattr(dtype, "unit", None)
            metadata[column] = {
                "dtype": str(dtype),
                "timezone": None if timezone is None else str(timezone),
                "unit": None if unit is None else str(unit),
            }
    return metadata


def _portable_path(path: Path, manifest_path: Path | None) -> str:
    if manifest_path is None:
        return path.name
    return Path(os.path.relpath(path, start=manifest_path.parent)).as_posix()


def _table_entry(
    frame: pd.DataFrame,
    path: Path,
    *,
    artifact_format: str,
    id_columns: Sequence[str],
    manifest_path: Path | None,
) -> ArtifactManifestEntry:
    missing_ids = sorted(set(id_columns).difference(frame.columns))
    if missing_ids:
        raise ValueError(f"id_columns are absent from table: {missing_ids}")
    positions, sample_sha256 = _sample_hash(frame)
    schema = _schema_payload(frame)
    return ArtifactManifestEntry(
        artifact_path=_portable_path(path, manifest_path),
        artifact_format=artifact_format,
        sha256=_sha256_file(path),
        byte_size=path.stat().st_size,
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=tuple(frame.columns),
        pandas_dtypes={column: str(frame[column].dtype) for column in frame.columns},
        arrow_types={str(item["name"]): str(item["arrow_type"]) for item in schema["columns"]},
        schema_fingerprint=_schema_fingerprint(frame),
        null_counts={column: int(frame[column].isna().sum()) for column in frame.columns},
        timestamp_metadata=_timestamp_metadata(frame),
        id_columns=tuple(id_columns),
        sample_positions=positions,
        sample_sha256=sample_sha256,
    )


def _update_manifest(
    manifest_path: Path | None,
    entry: ArtifactManifestEntry,
) -> None:
    if manifest_path is None:
        return
    manifest_path = Path(manifest_path)
    entries: dict[str, Any] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("layout_version") != ARTIFACT_LAYOUT_VERSION:
            raise ValueError(
                f"Unsupported artifact manifest version at {manifest_path}: "
                f"{payload.get('layout_version')!r}"
            )
        entries = dict(payload.get("artifacts", {}))
    entries[entry.artifact_path] = entry.to_dict()
    output = {
        "layout_version": ARTIFACT_LAYOUT_VERSION,
        "artifacts": {key: entries[key] for key in sorted(entries)},
    }
    encoded = _canonical_json_bytes(output, indent=2)
    _atomic_write(manifest_path, lambda temporary: temporary.write_bytes(encoded))


def save_parquet(
    table: pd.DataFrame,
    path: Path | str,
    *,
    id_columns: Sequence[str] = (),
    manifest_path: Path | str | None = None,
    compression: str = "zstd",
) -> ArtifactManifestEntry:
    """Atomically save a table as Parquet and update an optional manifest."""

    frame = _require_table(table)
    destination = Path(path)
    manifest = None if manifest_path is None else Path(manifest_path)
    _atomic_write(
        destination,
        lambda temporary: frame.to_parquet(
            temporary,
            engine="pyarrow",
            compression=compression,
            index=False,
        ),
    )
    entry = _table_entry(
        frame,
        destination,
        artifact_format="parquet",
        id_columns=id_columns,
        manifest_path=manifest,
    )
    _update_manifest(manifest, entry)
    return entry


def save_csv(
    table: pd.DataFrame,
    path: Path | str,
    *,
    id_columns: Sequence[str] = (),
    manifest_path: Path | str | None = None,
) -> ArtifactManifestEntry:
    """Atomically save a deterministic UTF-8 CSV and update a manifest."""

    frame = _require_table(table)
    for column in frame.columns:
        if (is_object_dtype(frame[column].dtype) or is_string_dtype(frame[column].dtype)) and frame[
            column
        ].map(lambda value: isinstance(value, str) and value == _CSV_NULL_TOKEN).any():
            raise ValueError(
                f"CSV column {column!r} contains the reserved null token {_CSV_NULL_TOKEN!r}"
            )
    destination = Path(path)
    manifest = None if manifest_path is None else Path(manifest_path)
    _atomic_write(
        destination,
        lambda temporary: frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            float_format="%.17g",
            na_rep=_CSV_NULL_TOKEN,
        ),
    )
    entry = _table_entry(
        frame,
        destination,
        artifact_format="csv",
        id_columns=id_columns,
        manifest_path=manifest,
    )
    _update_manifest(manifest, entry)
    return entry


def _canonical_csv_object(value: Any) -> str | pd.NA:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    if value is None or value is pd.NA:
        return pd.NA
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return pd.NA
    return str(value)


def prepare_csv_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a CSV-safe copy with mixed object cells encoded as strings.

    CSV has no representation for Python container types or mixed scalar
    object columns. Structured values therefore use canonical JSON and every
    other non-null value in an object column uses its stable string form.
    Native numeric, boolean, categorical, and timestamp columns are preserved.
    """

    frame = _require_table(table).copy()
    for column in frame.columns:
        if is_object_dtype(frame[column].dtype):
            frame[column] = frame[column].map(_canonical_csv_object).astype("string")
    return frame


def save_json(
    payload: Any,
    path: Path | str,
    *,
    manifest_path: Path | str | None = None,
) -> ArtifactManifestEntry:
    """Atomically save stable JSON and optionally record its content hash."""

    destination = Path(path)
    manifest = None if manifest_path is None else Path(manifest_path)
    encoded = _canonical_json_bytes(payload, indent=2)
    _atomic_write(destination, lambda temporary: temporary.write_bytes(encoded))
    entry = ArtifactManifestEntry(
        artifact_path=_portable_path(destination, manifest),
        artifact_format="json",
        sha256=_sha256_file(destination),
        byte_size=destination.stat().st_size,
    )
    _update_manifest(manifest, entry)
    return entry


def _restore_csv_dtypes(loaded: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    result = loaded.copy()
    for column in expected.columns:
        dtype = expected[column].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            values = pd.to_datetime(result[column], utc=True, errors="coerce")
            result[column] = values.dt.tz_convert(dtype.tz).astype(dtype)
        elif is_datetime64_any_dtype(dtype):
            result[column] = pd.to_datetime(result[column], errors="coerce").astype(dtype)
        elif is_timedelta64_dtype(dtype):
            result[column] = pd.to_timedelta(result[column], errors="coerce").astype(dtype)
        elif isinstance(dtype, pd.CategoricalDtype):
            result[column] = pd.Categorical(
                result[column],
                categories=expected[column].cat.categories,
                ordered=expected[column].cat.ordered,
            )
        elif is_bool_dtype(dtype):
            if is_extension_array_dtype(dtype):
                result[column] = (
                    result[column].astype(str).replace({"nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
                )
                result[column] = (
                    result[column]
                    .map({"True": True, "False": False, True: True, False: False})
                    .astype(dtype)
                )
            else:
                result[column] = result[column].astype(dtype)
        elif is_integer_dtype(dtype) or is_extension_array_dtype(dtype):
            result[column] = result[column].astype(dtype)
        elif is_string_dtype(dtype) and str(dtype).startswith("string"):
            result[column] = result[column].astype(dtype)
        else:
            result[column] = result[column].astype(dtype)
    return result.loc[:, expected.columns]


def load_saved_table(
    path: Path | str,
    *,
    expected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load a Parquet or CSV artifact, optionally restoring exact CSV dtypes."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source, engine="pyarrow")
    if suffix == ".csv":
        loaded = pd.read_csv(
            source,
            encoding="utf-8",
            float_precision="round_trip",
            keep_default_na=False,
            na_values=[_CSV_NULL_TOKEN],
        )
        if expected is not None:
            return _restore_csv_dtypes(loaded, _require_table(expected))
        return loaded
    raise ValueError(f"Unsupported saved table format: {suffix!r}")


def _exact_timestamp_match(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, str]:
    timestamp_columns = [
        column for column in expected.columns if is_datetime64_any_dtype(expected[column].dtype)
    ]
    for column in timestamp_columns:
        expected_series = expected[column]
        actual_series = actual[column]
        if str(expected_series.dtype) != str(actual_series.dtype):
            return (
                False,
                f"{column}: dtype {actual_series.dtype!s} != {expected_series.dtype!s}",
            )
        expected_values = expected_series.astype("int64").to_numpy()
        actual_values = actual_series.astype("int64").to_numpy()
        if not np.array_equal(expected_values, actual_values):
            return False, f"{column}: timestamp values or order differ"
    return True, f"checked={timestamp_columns}"


def _exact_id_match(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    id_columns: Sequence[str],
) -> tuple[bool, str]:
    for column in id_columns:
        try:
            pd.testing.assert_series_equal(
                expected[column].reset_index(drop=True),
                actual[column].reset_index(drop=True),
                check_dtype=True,
                check_exact=True,
                check_names=False,
            )
        except AssertionError as error:
            return False, f"{column}: {str(error).splitlines()[0]}"
    return True, f"checked={list(id_columns)}"


def verify_saved_table(
    expected: pd.DataFrame,
    path: Path | str,
    *,
    id_columns: Sequence[str] = (),
    manifest_entry: ArtifactManifestEntry | Mapping[str, Any] | None = None,
    raise_on_failure: bool = True,
) -> pd.DataFrame:
    """Reload and verify table identity, ordering, schema, and fingerprints.

    The returned DataFrame is suitable for direct notebook display or
    persistence.  By default any failed check also raises
    :class:`ArtifactVerificationError`, whose ``diagnostics`` attribute
    contains the same table.
    """

    frame = _require_table(expected)
    source = Path(path)
    manifest = (
        ArtifactManifestEntry.from_dict(manifest_entry)
        if isinstance(manifest_entry, Mapping)
        else manifest_entry
    )
    records: list[dict[str, object]] = []

    def add(check_name: str, passed: bool, expected_value: Any, actual_value: Any) -> None:
        records.append(
            {
                "check_name": check_name,
                "passed": bool(passed),
                "expected": str(expected_value),
                "actual": str(actual_value),
            }
        )

    exists = source.is_file()
    add("file_exists", exists, True, exists)
    if not exists:
        diagnostics = pd.DataFrame.from_records(records)
        if raise_on_failure:
            raise ArtifactVerificationError(source, diagnostics)
        return diagnostics

    checksum = _sha256_file(source)
    if manifest is not None:
        add("sha256", checksum == manifest.sha256, manifest.sha256, checksum)
        add(
            "byte_size",
            source.stat().st_size == manifest.byte_size,
            manifest.byte_size,
            source.stat().st_size,
        )
    try:
        actual = load_saved_table(source, expected=frame)
    except (OSError, ValueError, TypeError) as error:
        add("reload", False, "successful", f"{type(error).__name__}: {error}")
        diagnostics = pd.DataFrame.from_records(records)
        if raise_on_failure:
            raise ArtifactVerificationError(source, diagnostics) from error
        return diagnostics

    add("row_count", len(actual) == len(frame), len(frame), len(actual))
    add(
        "column_count",
        len(actual.columns) == len(frame.columns),
        len(frame.columns),
        len(actual.columns),
    )
    expected_columns = list(frame.columns)
    actual_columns = list(actual.columns)
    add(
        "column_order",
        actual_columns == expected_columns,
        expected_columns,
        actual_columns,
    )
    expected_dtypes = {column: str(frame[column].dtype) for column in frame.columns}
    actual_dtypes = {column: str(actual[column].dtype) for column in actual.columns}
    add("pandas_dtypes", actual_dtypes == expected_dtypes, expected_dtypes, actual_dtypes)
    expected_nulls = {column: int(frame[column].isna().sum()) for column in frame.columns}
    actual_nulls = {column: int(actual[column].isna().sum()) for column in actual.columns}
    add("null_counts", actual_nulls == expected_nulls, expected_nulls, actual_nulls)

    if set(id_columns).issubset(frame.columns) and set(id_columns).issubset(actual.columns):
        ids_match, id_detail = _exact_id_match(frame, actual, id_columns)
    else:
        ids_match = False
        id_detail = "one or more ID columns are missing"
    add("id_values_and_order", ids_match, "exact", id_detail)

    if expected_columns == actual_columns:
        timestamps_match, timestamp_detail = _exact_timestamp_match(frame, actual)
        actual_schema = _schema_fingerprint(actual)
        expected_schema = _schema_fingerprint(frame)
        positions = (
            manifest.sample_positions
            if manifest is not None and manifest.sample_positions
            else _sample_positions(len(frame))
        )
        _, expected_sample = _sample_hash(frame, positions=positions)
        _, actual_sample = _sample_hash(actual, positions=positions)
    else:
        timestamps_match = False
        timestamp_detail = "column order mismatch"
        actual_schema = "unavailable"
        expected_schema = _schema_fingerprint(frame)
        expected_sample = "unavailable"
        actual_sample = "unavailable"
    add("timestamps_exact", timestamps_match, "exact", timestamp_detail)
    add("schema_fingerprint", actual_schema == expected_schema, expected_schema, actual_schema)
    add(
        "deterministic_sample_hash",
        actual_sample == expected_sample,
        expected_sample,
        actual_sample,
    )

    try:
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
            check_like=False,
            check_categorical=True,
            check_freq=False,
        )
        exact_match = True
        exact_detail = "exact values, nulls, dtypes, and row/column order"
    except AssertionError as error:
        exact_match = False
        exact_detail = str(error).splitlines()[0]
    add("full_table_exact", exact_match, "exact", exact_detail)

    if manifest is not None:
        add(
            "manifest_schema_fingerprint",
            expected_schema == manifest.schema_fingerprint,
            manifest.schema_fingerprint,
            expected_schema,
        )
        add(
            "manifest_sample_hash",
            expected_sample == manifest.sample_sha256,
            manifest.sample_sha256,
            expected_sample,
        )

    diagnostics = pd.DataFrame.from_records(records)
    if raise_on_failure and not bool(diagnostics["passed"].all()):
        raise ArtifactVerificationError(source, diagnostics)
    return diagnostics
