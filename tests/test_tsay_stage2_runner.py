"""Integrity tests for the amended outcome-free Stage 2 runner."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_tsay_feature_research import (
    AMENDMENT_PATH,
    DETERMINISTIC_ROLLING_COMMIT,
    _assert_stage2_access,
    _guarded_primitive_sha256,
)


class TsayStage2RunnerTests(unittest.TestCase):
    def test_amendment_is_separate_and_names_exact_provenance_commit(self) -> None:
        self.assertTrue(AMENDMENT_PATH.is_file())
        source = AMENDMENT_PATH.read_text(encoding="utf-8")
        self.assertIn(DETERMINISTIC_ROLLING_COMMIT, source)
        self.assertIn("not a general numerical tolerance", source)
        self.assertIn("overwritten or regenerated", source)

    def test_guarded_primitive_hash_is_order_stable_and_value_sensitive(self) -> None:
        frame = pd.DataFrame(
            {
                "observation_id": np.asarray([2, 1], dtype=np.int64),
                "atr_20": np.asarray([2.0, 1.0], dtype=np.float32),
                "current_range_over_atr": np.asarray([0.2, 0.1], dtype=np.float32),
                "realized_volatility_15": np.asarray([4.0, 3.0], dtype=np.float32),
            }
        )
        baseline = _guarded_primitive_sha256(frame)
        self.assertEqual(baseline, _guarded_primitive_sha256(frame.iloc[::-1]))
        changed = frame.copy()
        changed.loc[0, "realized_volatility_15"] = np.nextafter(
            np.float32(4.0), np.float32(np.inf), dtype=np.float32
        )
        self.assertNotEqual(baseline, _guarded_primitive_sha256(changed))

    def test_stage2_access_gate_rejects_label_source(self) -> None:
        payload = {
            "cutoff_date_ny": "2024-12-31",
            "stage": 2,
            "records": [
                {
                    "source": "data/processed/statistical_research/forward_labels_gc.parquet",
                    "mode": "TIMESERIES_WITH_PRODUCT",
                    "access_kind": "PARQUET_READ",
                    "pushed_predicate": ("trade_date_ny <= 2024-12-31 AND product == GC"),
                    "returned_maximum_date": "2023-12-31",
                    "returned_product_set": ["GC"],
                }
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "label access"):
            _assert_stage2_access(payload)


if __name__ == "__main__":
    unittest.main()
