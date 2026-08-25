"""Outcome-free Stage 1 validation tests."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.statistical_research.feature_registry import FEATURE_SPECS
from src.statistical_research.fes_project1_config import FEATURE_SPECS as FES_FEATURE_SPECS
from src.statistical_research.mlat_feature_registry import MLAT_FEATURE_SPECS
from src.statistical_research.tsay_artifacts import (
    load_parquet,
    save_parquet_atomic,
    verify_parquet_round_trip,
)
from src.statistical_research.tsay_feature_engineering import (
    TSAY_CONTEXT_COLUMNS,
    TSAY_MATRIX_COLUMNS,
    TSAY_METADATA_COLUMNS,
)
from src.statistical_research.tsay_feature_registry import TSAY_FEATURE_COLUMNS
from src.statistical_research.tsay_feature_validation import (
    build_exact_date_folds,
    build_overlap_audit,
    deterministic_feature_sample_hash,
    freeze_outer_folds,
    summarize_metadata,
    validate_tsay_feature_matrix,
    verify_reloaded_tsay_matrix,
)


class TsayFeatureValidationTests(unittest.TestCase):
    def test_exact_outer_fold_boundaries_step_and_hash(self) -> None:
        dates = pd.date_range("2021-01-01", periods=400, freq="D")
        folds = build_exact_date_folds(
            dates,
            initial_training_dates=126,
            assessment_dates=21,
            step_dates=21,
        )
        self.assertEqual(folds[0].train_dates[0], "2021-01-01")
        self.assertEqual(folds[0].train_dates[-1], dates[125].date().isoformat())
        self.assertEqual(folds[0].embargo_date, dates[126].date().isoformat())
        self.assertEqual(folds[0].assessment_dates[0], dates[127].date().isoformat())
        self.assertEqual(folds[0].assessment_dates[-1], dates[147].date().isoformat())
        self.assertEqual(folds[1].assessment_dates[0], dates[148].date().isoformat())
        self.assertEqual(len(folds[0].fold_sha256), 64)

    def test_attainable_date_gate_is_arithmetically_and_structurally_evaluable(self) -> None:
        dates = pd.date_range("2021-01-01", periods=400, freq="D")
        rows = []
        observation_id = 0
        for session in ("London", "New York"):
            for value in dates:
                rows.append(
                    {
                        "observation_id": observation_id,
                        "trade_date_ny": value,
                        "entry_session": session,
                        "research_partition": "Development",
                    }
                )
                observation_id += 1
        frozen = freeze_outer_folds(pd.DataFrame(rows))
        for payload in frozen["sessions"].values():
            self.assertGreaterEqual(payload["attainable_dev_oof_dates"], 180)
            self.assertEqual(
                payload["minimum_dev_oof_dates"],
                math.ceil(0.90 * payload["attainable_dev_oof_dates"]),
            )
            self.assertLessEqual(
                payload["minimum_dev_oof_dates"], payload["attainable_dev_oof_dates"]
            )
        self.assertEqual(len(frozen["fold_membership_sha256"]), 64)

    def test_metadata_summary_uses_identity_columns_only_and_requires_unique_key(self) -> None:
        frame = pd.DataFrame(
            {
                "observation_id": [1, 2],
                "decision_timestamp_utc": pd.to_datetime(
                    ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"]
                ),
                "trade_date_ny": pd.to_datetime(["2023-01-01", "2024-01-01"]),
                "entry_session": ["London", "New York"],
                "research_partition": ["Development", "Validation"],
            }
        )
        audit = summarize_metadata(frame, source="synthetic")
        self.assertEqual(audit["rows"], 2)
        self.assertTrue(audit["identity_unique"])
        self.assertEqual(len(audit["metadata_sample_sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "unique key"):
            summarize_metadata(pd.concat([frame, frame.iloc[[0]]]), source="duplicate")

    def test_overlap_resolution_retains_all_nine_without_outcome_selection(self) -> None:
        base = tuple(spec.feature_name for spec in FEATURE_SPECS)
        mlat = tuple(spec.feature_name for spec in MLAT_FEATURE_SPECS)
        fes = tuple(spec["name"] for spec in FES_FEATURE_SPECS)
        audit = build_overlap_audit(
            base_feature_names=base,
            frozen_opportunity_names=base[:15],
            mlat_feature_names=mlat,
            fes_feature_names=fes,
        )
        self.assertEqual(len(audit), 9)
        self.assertTrue(all(row["confirmatory_hypothesis_retained"] for row in audit))
        self.assertTrue(all(row["requires_stage2_implementation"] for row in audit))
        self.assertTrue(all(not row["exact_formula_match"] for row in audit))

    def test_matrix_schema_ranges_hash_and_exact_parquet_round_trip(self) -> None:
        metadata = {
            "observation_id": np.asarray([1], dtype=np.int64),
            "decision_timestamp_utc": pd.to_datetime(["2023-01-01T12:00:00Z"]),
            "decision_timestamp_ny": pd.to_datetime(["2023-01-01T07:00:00-05:00"]),
            "entry_timestamp_utc": pd.to_datetime(["2023-01-01T12:01:00Z"]),
            "entry_timestamp_ny": pd.to_datetime(["2023-01-01T07:01:00-05:00"]),
            "trade_date_ny": pd.to_datetime(["2023-01-01"]),
            "research_partition": ["Development"],
            "entry_session": ["London"],
            "product": ["GC"],
            "symbol": ["GCZ3"],
            "active_symbol": ["GCZ3"],
            "instrument_id": np.asarray([1], dtype=np.int64),
            "continuous_segment_id": np.asarray([1], dtype=np.int64),
        }
        matrix = pd.DataFrame(metadata)
        for name in TSAY_FEATURE_COLUMNS:
            matrix[name] = np.asarray([np.nan], dtype=np.float32)
        matrix["tsay_roll_spread_identified_120"] = np.asarray([-1], dtype=np.int8)
        matrix["tsay_clock_z_volume"] = np.asarray([np.nan], dtype=np.float32)
        matrix["tsay_clock_z_range"] = np.asarray([np.nan], dtype=np.float32)
        matrix = matrix.loc[:, TSAY_MATRIX_COLUMNS]
        eligible = matrix.loc[:, TSAY_METADATA_COLUMNS].copy()
        eligible["decision_bar_id"] = np.asarray([10], dtype=np.int64)
        report = validate_tsay_feature_matrix(matrix, eligible)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["sample_hash"]["digest"]), 64)
        self.assertEqual(
            deterministic_feature_sample_hash(matrix)["digest"],
            report["sample_hash"]["digest"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.parquet"
            digest = save_parquet_atomic(path, matrix)
            self.assertEqual(len(digest), 64)
            verify_parquet_round_trip(path, matrix)
            reloaded = load_parquet(path)
            verify_reloaded_tsay_matrix(matrix, reloaded)
            self.assertEqual(tuple(reloaded.columns), TSAY_MATRIX_COLUMNS)
            for name in TSAY_FEATURE_COLUMNS + TSAY_CONTEXT_COLUMNS[1:]:
                self.assertEqual(str(reloaded[name].dtype), "float32")


if __name__ == "__main__":
    unittest.main()
