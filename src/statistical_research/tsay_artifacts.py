"""Deterministic hashing and ignored Stage 1 artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class TsayArtifactError(RuntimeError):
    """Raised when a persisted Tsay artifact fails an integrity check."""


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def canonical_payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_json_round_trip(path: Path, expected: Any) -> str:
    actual = read_json(path)
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise TsayArtifactError(f"JSON save/reload mismatch: {path}")
    return sha256_file(path)


def tracked_file_identity(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    if not path.is_file():
        raise TsayArtifactError(f"Required file is absent: {relative_path}")
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def save_parquet_atomic(path: Path, frame: pd.DataFrame) -> str:
    """Persist a dataframe atomically with deterministic column order."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Parquet artifacts must be pandas DataFrames.")
    if frame.columns.has_duplicates:
        raise TsayArtifactError(f"Duplicate Parquet columns are forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet.tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def load_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def verify_parquet_round_trip(path: Path, expected: pd.DataFrame) -> str:
    actual = load_parquet(path)
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected.reset_index(drop=True),
            check_exact=True,
            check_dtype=True,
            check_categorical=True,
        )
    except AssertionError as exc:
        raise TsayArtifactError(f"Parquet save/reload mismatch: {path}: {exc}") from exc
    return sha256_file(path)
