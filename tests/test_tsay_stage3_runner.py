"""Integrity tests for the Development-only Stage 3 runner."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_tsay_feature_research import (
    STAGE3_LABEL_COLUMNS,
    _assert_stage3_access,
    _label_atr_tick_consistency,
)


class TsayStage3RunnerTests(unittest.TestCase):
    def test_access_gate_requires_exact_development_label_read(self) -> None:
        payload = {
            "cutoff_date_ny": "2024-12-31",
            "materialization_cutoff_date_ny": "2023-12-31",
            "stage": 3,
            "records": [
                {
                    "source": "data/processed/statistical_research/forward_labels_gc.parquet",
                    "mode": "TIMESERIES_WITH_PRODUCT",
                    "requested_columns": list(STAGE3_LABEL_COLUMNS),
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
        _assert_stage3_access(payload)
        payload["records"][0]["returned_research_partition_set"] = ["Validation"]
        with self.assertRaisesRegex(RuntimeError, "Development-only"):
            _assert_stage3_access(payload)

    def test_label_tick_qa_is_rowwise_and_rejects_inconsistent_field(self) -> None:
        frame = pd.DataFrame(
            {
                "decision_atr_20m": [2.0, 4.0],
                "label_available_60": [True, True],
                "forward_return_60_ticks": [10.0, -4.0],
                "forward_return_60_atr": [0.5, -0.1],
                "future_range_60_ticks": [20.0, 8.0],
                "future_range_60_atr": [1.0, 0.2],
            }
        )
        actual = _label_atr_tick_consistency(frame)
        self.assertTrue(actual["rowwise_consistent"].all())
        self.assertFalse(actual["aggregate_average_atr_conversion_used"].any())

        changed = frame.copy()
        changed.loc[1, "future_range_60_atr"] = 0.2001
        with self.assertRaisesRegex(RuntimeError, "consistency QA failed"):
            _label_atr_tick_consistency(changed)


if __name__ == "__main__":
    unittest.main()
