from __future__ import annotations

import dataclasses
import unittest

import numpy as np
import pandas as pd

from src.statistical_research.mlat_feature_registry import (
    MLAT_FEATURE_NAMES,
    MLAT_FEATURE_SPECS,
    MLAT_METADATA_COLUMNS,
    MLAT_MINIMUM_HISTORY,
    MLAT_REGISTRY_COLUMNS,
    build_mlat_registry,
    mlat_feature_columns,
    registry_to_frame,
    validate_mlat_feature_frame,
    validate_mlat_registry,
)


class MLATFeatureRegistryTests(unittest.TestCase):
    def test_registry_has_exact_frozen_membership_order_and_histories(self):
        expected_names = (
            "bollinger_zscore_20",
            "bollinger_bandwidth_20",
            "cutler_rsi_14",
            "chaikin_money_flow_20",
            "amihud_illiquidity_60",
            "parkinson_volatility_30",
            "rogers_satchell_volatility_30",
            "realized_semivariance_balance_60",
            "bipower_jump_ratio_60",
            "variance_ratio_60_5",
            "return_sign_entropy_60",
            "volatility_of_volatility_60",
        )
        expected_histories = (20, 20, 15, 20, 61, 30, 30, 61, 62, 65, 61, 75)
        self.assertEqual(MLAT_FEATURE_NAMES, expected_names)
        self.assertEqual(MLAT_MINIMUM_HISTORY, expected_histories)
        self.assertEqual(mlat_feature_columns(), expected_names)
        self.assertEqual(tuple(spec.feature_name for spec in build_mlat_registry()), expected_names)

    def test_specs_are_typed_and_immutable(self):
        spec = MLAT_FEATURE_SPECS[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.feature_name = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            spec.input_columns[0] = "future_return"  # type: ignore[index]

    def test_registry_frame_is_complete_and_machine_readable(self):
        frame = registry_to_frame()
        validate_mlat_registry(frame)
        self.assertEqual(tuple(frame.columns), MLAT_REGISTRY_COLUMNS)
        self.assertEqual(frame["feature_name"].tolist(), list(MLAT_FEATURE_NAMES))
        self.assertEqual(frame["hypothesis_id"].tolist(), [f"MLAT-H{i:03d}" for i in range(1, 13)])
        self.assertEqual(frame["formula_id"].tolist(), [f"MLAT-F{i:03d}" for i in range(2, 14)])
        self.assertTrue(frame["output_dtype"].eq("float32").all())
        self.assertTrue(frame["is_experimental"].all())
        self.assertFalse(frame.astype(str).apply(lambda column: column.str.strip().eq("").any()).any())
        self.assertEqual(frame.loc[3, "input_columns"], "high|low|close|volume")

    def test_registry_validation_rejects_membership_order_and_schema_drift(self):
        reordered = registry_to_frame().iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "membership/order"):
            validate_mlat_registry(reordered)

        missing = registry_to_frame().drop(columns=["fit_scope"])
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            validate_mlat_registry(missing)

        changed_history = registry_to_frame()
        changed_history.loc[4, "minimum_history"] = 60
        with self.assertRaisesRegex(ValueError, "minimum-history"):
            validate_mlat_registry(changed_history)

    def test_feature_frame_helper_enforces_exact_alignment_dtype_and_bounds(self):
        metadata = {
            "observation_id": pd.Series([1, 2], dtype="int64"),
            "decision_timestamp_utc": pd.to_datetime(
                ["2024-01-02 12:00:00Z", "2024-01-02 12:01:00Z"], utc=True
            ),
            "decision_timestamp_ny": pd.to_datetime(
                ["2024-01-02 07:00:00-05:00", "2024-01-02 07:01:00-05:00"]
            ),
            "entry_timestamp_utc": pd.to_datetime(
                ["2024-01-02 12:01:00Z", "2024-01-02 12:02:00Z"], utc=True
            ),
            "entry_timestamp_ny": pd.to_datetime(
                ["2024-01-02 07:01:00-05:00", "2024-01-02 07:02:00-05:00"]
            ),
            "trade_date_ny": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "research_partition": ["Development", "Validation"],
            "product": ["GC", "GC"],
            "symbol": ["GCZ4", "GCZ4"],
            "active_symbol": ["GCZ4", "GCZ4"],
            "instrument_id": pd.Series([101, 101], dtype="uint32"),
            "continuous_segment_id": pd.Series([1, 1], dtype="int32"),
        }
        features = {
            name: np.array([np.nan, 0.0], dtype=np.float32) for name in MLAT_FEATURE_NAMES
        }
        frame = pd.DataFrame({**metadata, **features})
        frame = frame.loc[:, MLAT_METADATA_COLUMNS + MLAT_FEATURE_NAMES]
        validate_mlat_feature_frame(frame)

        wrong_dtype = frame.copy()
        wrong_dtype["bollinger_zscore_20"] = wrong_dtype["bollinger_zscore_20"].astype("float64")
        with self.assertRaisesRegex(TypeError, "float32"):
            validate_mlat_feature_frame(wrong_dtype)

        out_of_range = frame.copy()
        out_of_range.loc[1, "cutler_rsi_14"] = np.float32(101.0)
        with self.assertRaisesRegex(ValueError, "upper"):
            validate_mlat_feature_frame(out_of_range)


if __name__ == "__main__":
    unittest.main()
