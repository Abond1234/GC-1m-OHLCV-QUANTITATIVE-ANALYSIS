from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.mlat_feature_validation import (
    DEFAULT_METADATA_COLUMNS,
    audit_feature_overlap,
    deterministic_sample_hash,
    validate_mlat_feature_matrix,
    verify_reloaded_feature_matrix,
)


def _matrix() -> pd.DataFrame:
    size = 20
    decision = pd.date_range("2024-01-02 13:00", periods=size, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "observation_id": np.arange(size, dtype=np.int64),
            "decision_timestamp_utc": decision,
            "decision_timestamp_ny": decision.tz_convert("America/New_York"),
            "entry_timestamp_utc": decision + pd.Timedelta(minutes=1),
            "entry_timestamp_ny": decision.tz_convert("America/New_York") + pd.Timedelta(minutes=1),
            "trade_date_ny": pd.to_datetime("2024-01-02").date(),
            "research_partition": ["Development"] * 15 + ["Validation"] * 5,
            "product": "GC",
            "symbol": "GCG4",
            "active_symbol": "GCG4",
            "instrument_id": 1,
            "continuous_segment_id": 2,
            "candidate_a": np.linspace(-1, 1, size),
            "candidate_b": np.sin(np.arange(size, dtype=float)),
        }
    )
    return frame[list(DEFAULT_METADATA_COLUMNS) + ["candidate_a", "candidate_b"]]


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_name": ["candidate_a", "candidate_b"],
            "output_dtype": ["float64", "float64"],
            "validation_minimum": [-2.0, -2.0],
            "validation_maximum": [2.0, 2.0],
        }
    )


class MlatFeatureValidationTests(unittest.TestCase):
    def test_validation_checks_registry_alignment_and_hash_is_deterministic(self) -> None:
        matrix = _matrix()
        result = validate_mlat_feature_matrix(matrix, _registry(), matrix.iloc[:, :12])

        assert result.ready
        assert result.checks["passed"].all()
        assert deterministic_sample_hash(matrix) == deterministic_sample_hash(matrix.copy())
        assert len(result.coverage) == 4
        assert result.sample_hash.loc[0, "algorithm"] == "sha256"

    def test_validation_reports_order_nonfinite_range_and_constant_failures(self) -> None:
        matrix = _matrix()
        matrix["candidate_a"] = 3.0
        matrix.loc[0, "candidate_b"] = np.inf
        matrix = matrix[list(DEFAULT_METADATA_COLUMNS) + ["candidate_b", "candidate_a"]]

        result = validate_mlat_feature_matrix(matrix, _registry())
        failed = set(result.checks.loc[~result.checks["passed"], "check"])

        assert "registry_feature_order" in failed
        assert "feature_nonfinite_values" in failed
        assert "feature_validation_ranges" in failed
        assert "feature_constants" in failed

    def test_overlap_audit_uses_development_and_finds_exact_and_near_duplicates(self) -> None:
        matrix = _matrix()
        existing = pd.DataFrame(
            {
                "observation_id": matrix["observation_id"],
                "old_exact": matrix["candidate_a"],
                "old_near": matrix["candidate_b"] * 2 + 1,
            }
        )
        audit = audit_feature_overlap(
            matrix,
            existing,
            candidate_features=["candidate_a", "candidate_b"],
            existing_features=["old_exact", "old_near"],
            minimum_observations=10,
        )

        exact = audit.loc[
            audit["candidate_feature"].eq("candidate_a")
            & audit["reference_feature"].eq("old_exact")
        ].iloc[0]
        near = audit.loc[
            audit["candidate_feature"].eq("candidate_b") & audit["reference_feature"].eq("old_near")
        ].iloc[0]
        assert exact["development_observations"] == 15
        assert exact["is_exact_duplicate"]
        assert near["is_near_duplicate"]

    def test_overlap_matches_pairwise_spearman_and_requires_minimum_common_rows(self) -> None:
        matrix = _matrix()
        matrix.loc[[0, 2, 4], "candidate_a"] = np.nan
        existing = pd.DataFrame(
            {
                "observation_id": matrix["observation_id"],
                "old": matrix["candidate_b"] * 3.0 + 2.0,
            }
        )
        existing.loc[[1, 3, 5], "old"] = np.nan
        audit = audit_feature_overlap(
            matrix,
            existing,
            candidate_features=["candidate_a"],
            existing_features=["old"],
            minimum_observations=5,
        )
        row = audit.loc[audit["audit_type"].eq("overlap")].iloc[0]
        development = matrix["research_partition"].eq("Development")
        joined = matrix.loc[development, ["observation_id", "candidate_a"]].merge(
            existing, on="observation_id", validate="one_to_one"
        )
        expected = joined["candidate_a"].corr(joined["old"], method="spearman")
        assert row["spearman_correlation"] == expected

        insufficient = (
            audit_feature_overlap(
                matrix,
                existing,
                candidate_features=["candidate_a"],
                existing_features=["old"],
                minimum_observations=15,
            )
            .loc[lambda value: value["audit_type"].eq("overlap")]
            .iloc[0]
        )
        assert not insufficient["sufficient_overlap"]
        assert not insufficient["is_exact_duplicate"]
        assert not insufficient["is_near_duplicate"]
        assert np.isnan(insufficient["spearman_correlation"])

    def test_missing_schema_returns_failed_result_instead_of_raising(self) -> None:
        matrix = _matrix().drop(columns="candidate_b")
        result = validate_mlat_feature_matrix(matrix, _registry())

        assert not result.ready
        failed = set(result.checks.loc[~result.checks["passed"], "check"])
        assert "registry_matrix_column_set" in failed
        assert not result.sample_hash.loc[0, "schema_complete"]

    def test_reload_comparison_detects_sample_value_change(self) -> None:
        original = _matrix()
        reloaded = original.copy()
        assert verify_reloaded_feature_matrix(original, reloaded)["passed"].all()

        reloaded.loc[0, "candidate_a"] += 0.01
        report = verify_reloaded_feature_matrix(original, reloaded)
        failed = set(report.loc[~report["passed"], "check"])
        assert "reload_deterministic_sample_hash" in failed
        assert "reload_numeric_values" in failed

        metadata_change = original.copy()
        metadata_change.loc[6, "symbol"] = "DIFFERENT"
        metadata_report = verify_reloaded_feature_matrix(original, metadata_change)
        metadata_failed = set(metadata_report.loc[~metadata_report["passed"], "check"])
        assert "reload_metadata_values" in metadata_failed
        assert "reload_nonnumeric_values" in metadata_failed


if __name__ == "__main__":
    unittest.main()
