"""Integrity tests for the Development-only Stage 4 runner."""

from __future__ import annotations

import unittest

from scripts.run_tsay_feature_research import STAGE4_LABEL_COLUMNS, _assert_stage4_access


class TsayStage4RunnerTests(unittest.TestCase):
    def test_access_gate_requires_exact_development_label_projection(self) -> None:
        payload = {
            "cutoff_date_ny": "2024-12-31",
            "materialization_cutoff_date_ny": "2023-12-31",
            "stage": 4,
            "records": [
                {
                    "source": "data/processed/statistical_research/forward_labels_gc.parquet",
                    "mode": "TIMESERIES_WITH_PRODUCT",
                    "requested_columns": list(STAGE4_LABEL_COLUMNS),
                    "pushed_predicate": (
                        "trade_date_ny <= 2023-12-31 AND product == GC "
                        "AND research_partition == Development"
                    ),
                    "returned_row_count": 10,
                    "returned_product_set": ["GC"],
                    "returned_research_partition_set": ["Development"],
                    "returned_minimum_date": "2023-01-01",
                    "returned_maximum_date": "2023-12-31",
                    "access_kind": "PARQUET_READ",
                }
            ],
        }
        _assert_stage4_access(payload)
        payload["records"][0]["requested_columns"] = list(STAGE4_LABEL_COLUMNS[:-1])
        with self.assertRaisesRegex(RuntimeError, "exact frozen allowlist"):
            _assert_stage4_access(payload)

    def test_access_gate_rejects_real_bar_or_validation_source(self) -> None:
        payload = {
            "cutoff_date_ny": "2024-12-31",
            "materialization_cutoff_date_ny": "2023-12-31",
            "stage": 4,
            "records": [
                {
                    "source": "data/processed/statistical_research/forward_labels_gc.parquet",
                    "mode": "TIMESERIES_WITH_PRODUCT",
                    "requested_columns": list(STAGE4_LABEL_COLUMNS),
                    "pushed_predicate": (
                        "trade_date_ny <= 2023-12-31 AND product == GC "
                        "AND research_partition == Development"
                    ),
                    "returned_row_count": 10,
                    "returned_product_set": ["GC"],
                    "returned_research_partition_set": ["Development"],
                    "returned_minimum_date": "2023-01-01",
                    "returned_maximum_date": "2023-12-31",
                    "access_kind": "PARQUET_READ",
                },
                {
                    "source": "data/processed/research_bars_gc_mgc_1m.parquet",
                    "mode": "TIMESERIES_WITH_PRODUCT",
                    "requested_columns": ["trade_date_ny", "product"],
                    "pushed_predicate": "trade_date_ny <= 2023-12-31 AND product == GC",
                    "returned_row_count": 1,
                    "returned_product_set": ["GC"],
                    "returned_research_partition_set": None,
                    "returned_minimum_date": "2023-01-01",
                    "returned_maximum_date": "2023-01-01",
                    "access_kind": "PARQUET_READ",
                },
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "Forbidden Stage 4 source"):
            _assert_stage4_access(payload)


if __name__ == "__main__":
    unittest.main()
