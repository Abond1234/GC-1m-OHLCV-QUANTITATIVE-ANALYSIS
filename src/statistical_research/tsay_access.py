"""Fail-closed, cutoff-enforcing data access for Tsay research.

All time-series reads materialize only rows dated through 2024-12-31.  Stage 1
also prohibits outcome columns and the sealed FES Validation table.  Footer
inspection is schema/planning metadata and never reports post-cutoff counts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

TSAY_CUTOFF_DATE = date(2024, 12, 31)
TSAY_CUTOFF_TIMESTAMP = pd.Timestamp(TSAY_CUTOFF_DATE)
TSAY_DEVELOPMENT_CUTOFF_DATE = date(2023, 12, 31)
TSAY_DEVELOPMENT_CUTOFF_TIMESTAMP = pd.Timestamp(TSAY_DEVELOPMENT_CUTOFF_DATE)


class AccessMode(StrEnum):
    """The three access modes frozen by the implementation contract."""

    TIMESERIES_WITH_PRODUCT = "TIMESERIES_WITH_PRODUCT"
    GC_MANIFEST_TIMESERIES = "GC_MANIFEST_TIMESERIES"
    NON_TIMESERIES_METADATA = "NON_TIMESERIES_METADATA"


DEFAULT_ALLOWLIST: Mapping[str, AccessMode] = {
    "data/processed/research_bars_gc_mgc_1m.parquet": AccessMode.TIMESERIES_WITH_PRODUCT,
    "data/processed/statistical_research/eligible_observations_gc.parquet": (
        AccessMode.TIMESERIES_WITH_PRODUCT
    ),
    "data/processed/statistical_research/forward_labels_gc.parquet": (
        AccessMode.TIMESERIES_WITH_PRODUCT
    ),
    "data/processed/statistical_research/feature_matrix_gc.parquet": (
        AccessMode.TIMESERIES_WITH_PRODUCT
    ),
    "data/processed/statistical_research/tsay_feature_research/v1/feature_matrix_tsay_gc.parquet": (
        AccessMode.TIMESERIES_WITH_PRODUCT
    ),
    "data/processed/statistical_research/mlat_feature_research/v1/feature_matrix_mlat_gc.parquet": (
        AccessMode.TIMESERIES_WITH_PRODUCT
    ),
    "data/processed/statistical_research/fes_project1/v1/scalar_features_development_gc.parquet": (
        AccessMode.GC_MANIFEST_TIMESERIES
    ),
    "data/processed/statistical_research/fes_project1/v1/scalar_features_validation_sealed_gc.parquet": (
        AccessMode.GC_MANIFEST_TIMESERIES
    ),
    "data/processed/statistical_research/frozen_expansion_feature_set_gc.parquet": (
        AccessMode.NON_TIMESERIES_METADATA
    ),
    "data/processed/statistical_research/feature_registry_gc.parquet": (
        AccessMode.NON_TIMESERIES_METADATA
    ),
}

STAGE1_LABEL_METADATA_COLUMNS = frozenset(
    {
        "observation_id",
        "decision_bar_id",
        "entry_bar_id",
        "decision_timestamp_utc",
        "decision_timestamp_ny",
        "entry_timestamp_utc",
        "entry_timestamp_ny",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "product",
        "symbol",
        "active_symbol",
        "instrument_id",
        "continuous_segment_id",
        "forced_exit_timestamp_ny",
    }
)

STAGE3_DEVELOPMENT_LABEL_COLUMNS = frozenset(
    {
        "observation_id",
        "decision_timestamp_utc",
        "decision_timestamp_ny",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "product",
        "decision_atr_20m",
        "forward_return_60_atr",
        "forward_return_60_ticks",
        "future_range_60_atr",
        "future_range_60_ticks",
        "forward_return_30_atr",
        "future_range_30_atr",
        "label_available_60",
        "label_available_30",
    }
)

STAGE4_DEVELOPMENT_LABEL_COLUMNS = frozenset(
    {
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
    }
)


class TsayAccessError(RuntimeError):
    """Raised when a request does not match the frozen access contract."""


@dataclass(frozen=True)
class AccessRecord:
    source: str
    mode: str
    requested_columns: tuple[str, ...]
    pushed_predicate: str
    physically_scanned_row_groups: tuple[int, ...]
    returned_row_count: int | None
    returned_product_set: tuple[str, ...] | None
    returned_research_partition_set: tuple[str, ...] | None
    returned_minimum_date: str | None
    returned_maximum_date: str | None
    access_kind: str


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise TsayAccessError(f"Access outside the repository is forbidden: {path}") from exc


def _date_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


class TsayGuardedLoader:
    """Repository-scoped Parquet loader with immutable cutoff and audit log."""

    def __init__(
        self,
        root: Path,
        *,
        stage: int,
        allowlist: Mapping[str, AccessMode] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.stage = int(stage)
        self.allowlist = dict(DEFAULT_ALLOWLIST if allowlist is None else allowlist)
        self.records: list[AccessRecord] = []

    def _resolve(self, relative_path: str | Path, mode: AccessMode) -> tuple[Path, str]:
        path = (self.root / relative_path).resolve()
        relative = _relative(self.root, path)
        expected_mode = self.allowlist.get(relative)
        if expected_mode is None:
            raise TsayAccessError(f"Source is not allowlisted for Tsay research: {relative}")
        if expected_mode != mode:
            raise TsayAccessError(
                f"Mode mismatch for {relative}: requested {mode.value}, expected {expected_mode.value}"
            )
        if not path.is_file():
            raise TsayAccessError(f"Required source does not exist: {relative}")
        return path, relative

    def footer_schema(self, relative_path: str | Path, *, mode: AccessMode) -> pa.Schema:
        """Return only footer schema and log no row count or post-cutoff statistic."""

        path, relative = self._resolve(relative_path, mode)
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        self.records.append(
            AccessRecord(
                source=relative,
                mode=mode.value,
                requested_columns=tuple(schema.names),
                pushed_predicate="FOOTER_ONLY_NO_ROW_MATERIALIZATION",
                physically_scanned_row_groups=(),
                returned_row_count=None,
                returned_product_set=None,
                returned_research_partition_set=None,
                returned_minimum_date=None,
                returned_maximum_date=None,
                access_kind="FOOTER_SCHEMA",
            )
        )
        return schema

    def _materialization_cutoff(self) -> pd.Timestamp:
        if self.stage in (3, 4):
            return TSAY_DEVELOPMENT_CUTOFF_TIMESTAMP
        return TSAY_CUTOFF_TIMESTAMP

    def _expression(self, mode: AccessMode, relative: str) -> ds.Expression | None:
        if mode == AccessMode.NON_TIMESERIES_METADATA:
            return None
        expression = ds.field("trade_date_ny") <= pa.scalar(self._materialization_cutoff())
        if mode == AccessMode.TIMESERIES_WITH_PRODUCT:
            expression = expression & (ds.field("product") == "GC")
        if self.stage in (3, 4) and relative.endswith("forward_labels_gc.parquet"):
            expression = expression & (ds.field("research_partition") == "Development")
        return expression

    def _filters(self, mode: AccessMode, relative: str) -> list[tuple[str, str, object]] | None:
        if mode == AccessMode.NON_TIMESERIES_METADATA:
            return None
        filters: list[tuple[str, str, object]] = [
            ("trade_date_ny", "<=", self._materialization_cutoff())
        ]
        if mode == AccessMode.TIMESERIES_WITH_PRODUCT:
            filters.append(("product", "==", "GC"))
        if self.stage in (3, 4) and relative.endswith("forward_labels_gc.parquet"):
            filters.append(("research_partition", "==", "Development"))
        return filters

    @staticmethod
    def _row_groups(path: Path, expression: ds.Expression | None) -> tuple[int, ...]:
        parquet = pq.ParquetFile(path)
        if expression is None:
            return tuple(range(parquet.metadata.num_row_groups))
        dataset = ds.dataset(path, format="parquet")
        row_groups: list[int] = []
        for fragment in dataset.get_fragments(filter=expression):
            for split in fragment.split_by_row_group(filter=expression):
                row_groups.extend(group.id for group in split.row_groups)
        return tuple(sorted(set(row_groups)))

    def read_parquet(
        self,
        relative_path: str | Path,
        *,
        columns: Iterable[str],
        mode: AccessMode,
        gc_manifest_verified: bool = False,
    ) -> pd.DataFrame:
        """Read an allowlisted table with immutable predicates and Stage 1 guards."""

        path, relative = self._resolve(relative_path, mode)
        requested = tuple(columns)
        if not requested:
            raise TsayAccessError("Column-wise access is mandatory; an empty request is forbidden.")
        schema = pq.ParquetFile(path).schema_arrow
        absent = sorted(set(requested) - set(schema.names))
        if absent:
            raise TsayAccessError(f"Missing requested columns in {relative}: {absent}")

        if mode == AccessMode.GC_MANIFEST_TIMESERIES and not gc_manifest_verified:
            raise TsayAccessError(f"GC-only manifest proof is required for {relative}")
        if self.stage < 5 and relative.endswith("scalar_features_validation_sealed_gc.parquet"):
            raise TsayAccessError(
                "The sealed FES Validation table cannot materialize before Stage 5."
            )
        if self.stage == 2 and relative.endswith("forward_labels_gc.parquet"):
            raise TsayAccessError("The label artifact cannot be read in Stage 2.")
        if self.stage == 1 and relative.endswith("forward_labels_gc.parquet"):
            forbidden = sorted(set(requested) - STAGE1_LABEL_METADATA_COLUMNS)
            if forbidden:
                raise TsayAccessError(
                    f"Stage 1 label access is schema/metadata-only; forbidden columns: {forbidden}"
                )
        if self.stage in (3, 4) and relative.endswith("forward_labels_gc.parquet"):
            allowed = (
                STAGE3_DEVELOPMENT_LABEL_COLUMNS
                if self.stage == 3
                else STAGE4_DEVELOPMENT_LABEL_COLUMNS
            )
            forbidden = sorted(set(requested) - allowed)
            if forbidden:
                raise TsayAccessError(
                    f"Stage {self.stage} label access exceeds the frozen Development columns: "
                    f"{forbidden}"
                )
            if "research_partition" not in requested:
                raise TsayAccessError(
                    f"Stage {self.stage} label reads must include research_partition for verification."
                )

        expression = self._expression(mode, relative)
        filters = self._filters(mode, relative)
        row_groups = self._row_groups(path, expression)
        table = pq.read_table(path, columns=list(requested), filters=filters)
        frame = table.to_pandas(ignore_metadata=True)

        minimum_date = None
        maximum_date = None
        product_set: tuple[str, ...] | None = None
        partition_set: tuple[str, ...] | None = None
        if mode != AccessMode.NON_TIMESERIES_METADATA:
            if "trade_date_ny" not in frame:
                raise TsayAccessError("Every time-series request must include trade_date_ny.")
            dates = pd.to_datetime(frame["trade_date_ny"], errors="raise")
            minimum_date = _date_text(dates.min()) if len(dates) else None
            maximum_date = _date_text(dates.max()) if len(dates) else None
            materialization_cutoff = self._materialization_cutoff().date()
            if len(dates) and dates.max().date() > materialization_cutoff:
                raise TsayAccessError(f"Post-cutoff row materialized from {relative}")
            if "research_partition" in frame:
                partition_set = tuple(
                    sorted(frame["research_partition"].astype(str).unique().tolist())
                )
        if mode == AccessMode.TIMESERIES_WITH_PRODUCT:
            if "product" not in frame:
                raise TsayAccessError(
                    "Product-filtered requests must include product for verification."
                )
            product_set = tuple(sorted(frame["product"].astype(str).unique().tolist()))
            if product_set not in ((), ("GC",)):
                raise TsayAccessError(f"Non-GC products returned from {relative}: {product_set}")
        if self.stage in (3, 4) and relative.endswith("forward_labels_gc.parquet"):
            if partition_set not in ((), ("Development",)):
                raise TsayAccessError(
                    f"Non-Development label rows returned from {relative}: {partition_set}"
                )

        predicate = "NONE_NON_TIMESERIES_METADATA"
        cutoff_text = self._materialization_cutoff().date().isoformat()
        if mode == AccessMode.TIMESERIES_WITH_PRODUCT:
            predicate = f"trade_date_ny <= {cutoff_text} AND product == GC"
        elif mode == AccessMode.GC_MANIFEST_TIMESERIES:
            predicate = f"trade_date_ny <= {cutoff_text} AND existing_manifest_proves_GC_only"
        if self.stage in (3, 4) and relative.endswith("forward_labels_gc.parquet"):
            predicate += " AND research_partition == Development"
        self.records.append(
            AccessRecord(
                source=relative,
                mode=mode.value,
                requested_columns=requested,
                pushed_predicate=predicate,
                physically_scanned_row_groups=row_groups,
                returned_row_count=len(frame),
                returned_product_set=product_set,
                returned_research_partition_set=partition_set,
                returned_minimum_date=minimum_date,
                returned_maximum_date=maximum_date,
                access_kind="PARQUET_READ",
            )
        )
        return frame

    def manifest_payload(self) -> dict[str, object]:
        return {
            "cutoff_date_ny": TSAY_CUTOFF_DATE.isoformat(),
            "materialization_cutoff_date_ny": self._materialization_cutoff().date().isoformat(),
            "stage": self.stage,
            "records": [asdict(record) for record in self.records],
        }

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.manifest_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def assert_access_manifest_cutoff(payload: Mapping[str, object]) -> None:
    """Verify that an access manifest proves no 2025+ row materialized."""

    if payload.get("cutoff_date_ny") != TSAY_CUTOFF_DATE.isoformat():
        raise TsayAccessError("Access manifest cutoff does not match the frozen contract.")
    materialization_cutoff = date.fromisoformat(
        str(payload.get("materialization_cutoff_date_ny", TSAY_CUTOFF_DATE.isoformat()))
    )
    if payload.get("stage") in (3, 4) and materialization_cutoff != TSAY_DEVELOPMENT_CUTOFF_DATE:
        raise TsayAccessError("Stages 3 and 4 must use the Development materialization cutoff.")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise TsayAccessError("Access manifest contains no records.")
    for record in records:
        if not isinstance(record, dict):
            raise TsayAccessError("Malformed access record.")
        maximum = record.get("returned_maximum_date")
        if maximum is not None and date.fromisoformat(str(maximum)) > materialization_cutoff:
            raise TsayAccessError(f"Post-cutoff access recorded: {record}")
        predicate = str(record.get("pushed_predicate", ""))
        mode = record.get("mode")
        if (
            record.get("access_kind") == "PARQUET_READ"
            and mode != AccessMode.NON_TIMESERIES_METADATA
        ):
            if materialization_cutoff.isoformat() not in predicate:
                raise TsayAccessError(f"Missing pushed cutoff predicate: {record}")
