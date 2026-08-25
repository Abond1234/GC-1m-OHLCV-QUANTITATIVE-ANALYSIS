"""Fail-closed tests for the Tsay guarded loader."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.statistical_research.tsay_access import (
    AccessMode,
    TsayAccessError,
    TsayGuardedLoader,
    assert_access_manifest_cutoff,
)


class TsayAccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.relative = "data/source.parquet"
        path = self.root / self.relative
        path.parent.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "observation_id": [1, 2, 3, 4],
                "trade_date_ny": pd.to_datetime(
                    ["2024-12-30", "2024-12-31", "2025-01-01", "2024-12-31"]
                ),
                "product": ["GC", "GC", "GC", "MGC"],
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _loader(self, *, stage: int = 1) -> TsayGuardedLoader:
        return TsayGuardedLoader(
            self.root,
            stage=stage,
            allowlist={self.relative: AccessMode.TIMESERIES_WITH_PRODUCT},
        )

    def test_cutoff_and_product_predicates_are_pushed_and_verified(self) -> None:
        loader = self._loader()
        actual = loader.read_parquet(
            self.relative,
            columns=("observation_id", "trade_date_ny", "product"),
            mode=AccessMode.TIMESERIES_WITH_PRODUCT,
        )
        self.assertEqual(actual["observation_id"].tolist(), [1, 2])
        self.assertEqual(set(actual["product"]), {"GC"})
        record = loader.records[-1]
        self.assertEqual(record.returned_maximum_date, "2024-12-31")
        self.assertIn("product == GC", record.pushed_predicate)
        self.assertTrue(record.physically_scanned_row_groups)

    def test_time_series_request_must_include_date_and_product_verifiers(self) -> None:
        loader = self._loader()
        with self.assertRaisesRegex(TsayAccessError, "trade_date_ny"):
            loader.read_parquet(
                self.relative,
                columns=("observation_id", "product"),
                mode=AccessMode.TIMESERIES_WITH_PRODUCT,
            )
        with self.assertRaisesRegex(TsayAccessError, "Product-filtered"):
            loader.read_parquet(
                self.relative,
                columns=("observation_id", "trade_date_ny"),
                mode=AccessMode.TIMESERIES_WITH_PRODUCT,
            )

    def test_mode_mismatch_and_missing_gc_manifest_proof_fail_closed(self) -> None:
        loader = self._loader()
        with self.assertRaisesRegex(TsayAccessError, "Mode mismatch"):
            loader.read_parquet(
                self.relative,
                columns=("observation_id", "trade_date_ny", "product"),
                mode=AccessMode.NON_TIMESERIES_METADATA,
            )

        productless = "data/productless.parquet"
        frame = pd.DataFrame(
            {"trade_date_ny": pd.to_datetime(["2024-01-01"]), "observation_id": [1]}
        )
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), self.root / productless)
        productless_loader = TsayGuardedLoader(
            self.root,
            stage=1,
            allowlist={productless: AccessMode.GC_MANIFEST_TIMESERIES},
        )
        with self.assertRaisesRegex(TsayAccessError, "manifest proof"):
            productless_loader.read_parquet(
                productless,
                columns=("observation_id", "trade_date_ny"),
                mode=AccessMode.GC_MANIFEST_TIMESERIES,
            )

    def test_stage1_label_outcomes_and_sealed_validation_rows_are_rejected(self) -> None:
        labels = "data/processed/statistical_research/forward_labels_gc.parquet"
        label_path = self.root / labels
        label_path.parent.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "observation_id": [1],
                "trade_date_ny": pd.to_datetime(["2023-01-01"]),
                "product": ["GC"],
                "forward_return_60_atr": [1.0],
            }
        )
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), label_path)
        loader = TsayGuardedLoader(
            self.root,
            stage=1,
            allowlist={labels: AccessMode.TIMESERIES_WITH_PRODUCT},
        )
        with self.assertRaisesRegex(TsayAccessError, "forbidden columns"):
            loader.read_parquet(
                labels,
                columns=(
                    "observation_id",
                    "trade_date_ny",
                    "product",
                    "forward_return_60_atr",
                ),
                mode=AccessMode.TIMESERIES_WITH_PRODUCT,
            )

        sealed = (
            "data/processed/statistical_research/fes_project1/v1/"
            "scalar_features_validation_sealed_gc.parquet"
        )
        sealed_path = self.root / sealed
        sealed_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(
                pd.DataFrame(
                    {
                        "observation_id": [1],
                        "trade_date_ny": pd.to_datetime(["2024-01-01"]),
                    }
                ),
                preserve_index=False,
            ),
            sealed_path,
        )
        sealed_loader = TsayGuardedLoader(
            self.root,
            stage=1,
            allowlist={sealed: AccessMode.GC_MANIFEST_TIMESERIES},
        )
        sealed_loader.footer_schema(sealed, mode=AccessMode.GC_MANIFEST_TIMESERIES)
        with self.assertRaisesRegex(TsayAccessError, "cannot materialize"):
            sealed_loader.read_parquet(
                sealed,
                columns=("observation_id", "trade_date_ny"),
                mode=AccessMode.GC_MANIFEST_TIMESERIES,
                gc_manifest_verified=True,
            )

    def test_stage2_rejects_every_label_row_request(self) -> None:
        labels = "data/processed/statistical_research/forward_labels_gc.parquet"
        label_path = self.root / labels
        label_path.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pandas(
                pd.DataFrame(
                    {
                        "observation_id": [1],
                        "trade_date_ny": pd.to_datetime(["2023-01-01"]),
                        "product": ["GC"],
                    }
                ),
                preserve_index=False,
            ),
            label_path,
        )
        loader = TsayGuardedLoader(
            self.root,
            stage=2,
            allowlist={labels: AccessMode.TIMESERIES_WITH_PRODUCT},
        )
        with self.assertRaisesRegex(TsayAccessError, "cannot be read in Stage 2"):
            loader.read_parquet(
                labels,
                columns=("observation_id", "trade_date_ny", "product"),
                mode=AccessMode.TIMESERIES_WITH_PRODUCT,
            )

    def test_stage3_pushes_development_label_partition_and_column_guards(self) -> None:
        labels = "data/processed/statistical_research/forward_labels_gc.parquet"
        label_path = self.root / labels
        label_path.parent.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "observation_id": [1, 2, 3],
                "trade_date_ny": pd.to_datetime(["2023-12-31", "2024-01-02", "2023-12-31"]),
                "research_partition": ["Development", "Validation", "Validation"],
                "product": ["GC", "GC", "GC"],
                "forward_return_60_atr": [1.0, 2.0, 3.0],
                "expansion_label_60": [True, False, True],
            }
        )
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), label_path)
        loader = TsayGuardedLoader(
            self.root,
            stage=3,
            allowlist={labels: AccessMode.TIMESERIES_WITH_PRODUCT},
        )
        actual = loader.read_parquet(
            labels,
            columns=(
                "observation_id",
                "trade_date_ny",
                "research_partition",
                "product",
                "forward_return_60_atr",
            ),
            mode=AccessMode.TIMESERIES_WITH_PRODUCT,
        )
        self.assertEqual(actual["observation_id"].tolist(), [1])
        record = loader.records[-1]
        self.assertEqual(record.returned_research_partition_set, ("Development",))
        self.assertEqual(record.returned_maximum_date, "2023-12-31")
        self.assertIn("research_partition == Development", record.pushed_predicate)
        self.assertEqual(loader.manifest_payload()["materialization_cutoff_date_ny"], "2023-12-31")
        assert_access_manifest_cutoff(loader.manifest_payload())

        with self.assertRaisesRegex(TsayAccessError, "exceeds the frozen"):
            loader.read_parquet(
                labels,
                columns=(
                    "observation_id",
                    "trade_date_ny",
                    "research_partition",
                    "product",
                    "expansion_label_60",
                ),
                mode=AccessMode.TIMESERIES_WITH_PRODUCT,
            )

    def test_stage3_manifest_assertion_rejects_loosened_development_cutoff(self) -> None:
        payload = {
            "cutoff_date_ny": "2024-12-31",
            "materialization_cutoff_date_ny": "2024-12-31",
            "stage": 3,
            "records": [],
        }
        with self.assertRaisesRegex(TsayAccessError, "Development materialization"):
            assert_access_manifest_cutoff(payload)

    def test_stage4_keeps_development_cutoff_and_allows_only_frozen_model_labels(self) -> None:
        labels = "data/processed/statistical_research/forward_labels_gc.parquet"
        label_path = self.root / labels
        label_path.parent.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "observation_id": [1, 2],
                "decision_timestamp_utc": pd.to_datetime(
                    ["2023-12-31 12:00Z", "2024-01-02 12:00Z"]
                ),
                "trade_date_ny": pd.to_datetime(["2023-12-31", "2024-01-02"]),
                "entry_session": ["London", "London"],
                "research_partition": ["Development", "Validation"],
                "product": ["GC", "GC"],
                "forward_return_60_atr": [0.1, 0.2],
                "future_range_60_atr": [1.1, 1.2],
                "expansion_label_60": [True, False],
                "label_available_60": [True, True],
                "exit_timestamp_utc_60": pd.to_datetime(["2023-12-31 13:00Z", "2024-01-02 13:00Z"]),
                "forward_return_30_atr": [0.0, 0.0],
            }
        )
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), label_path)
        loader = TsayGuardedLoader(
            self.root,
            stage=4,
            allowlist={labels: AccessMode.TIMESERIES_WITH_PRODUCT},
        )
        columns = tuple(column for column in frame.columns if column != "forward_return_30_atr")
        actual = loader.read_parquet(
            labels, columns=columns, mode=AccessMode.TIMESERIES_WITH_PRODUCT
        )
        self.assertEqual(actual["observation_id"].tolist(), [1])
        self.assertEqual(loader.manifest_payload()["materialization_cutoff_date_ny"], "2023-12-31")
        with self.assertRaisesRegex(TsayAccessError, "exceeds the frozen"):
            loader.read_parquet(
                labels,
                columns=(*columns, "forward_return_30_atr"),
                mode=AccessMode.TIMESERIES_WITH_PRODUCT,
            )

    def test_access_manifest_assertion_rejects_post_cutoff_record(self) -> None:
        loader = self._loader()
        loader.footer_schema(self.relative, mode=AccessMode.TIMESERIES_WITH_PRODUCT)
        loader.read_parquet(
            self.relative,
            columns=("observation_id", "trade_date_ny", "product"),
            mode=AccessMode.TIMESERIES_WITH_PRODUCT,
        )
        payload = loader.manifest_payload()
        assert_access_manifest_cutoff(payload)
        corrupted = json.loads(json.dumps(payload))
        corrupted["records"][-1]["returned_maximum_date"] = "2025-01-01"
        with self.assertRaisesRegex(TsayAccessError, "Post-cutoff"):
            assert_access_manifest_cutoff(corrupted)


if __name__ == "__main__":
    unittest.main()
