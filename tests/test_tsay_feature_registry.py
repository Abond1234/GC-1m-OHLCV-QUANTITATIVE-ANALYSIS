"""Frozen declaration tests for Tsay Stage 1."""

from __future__ import annotations

import unittest

from src.statistical_research.tsay_feature_registry import (
    D2_INPUTS,
    MODEL_LADDER,
    O2_INPUTS,
    SECONDARY_TICK_COLUMNS,
    TARGET_COLUMNS,
    TSAY_FEATURE_SPECS,
    TSAY_LOGICAL_TO_PHYSICAL,
    TSAY_PRIMARY_TEST_COUNT,
    validate_tsay_registry,
)


class TsayFeatureRegistryTests(unittest.TestCase):
    def test_order_names_family_and_dispositions_are_frozen(self) -> None:
        validate_tsay_registry()
        self.assertEqual(
            tuple(spec.logical_id for spec in TSAY_FEATURE_SPECS),
            tuple(f"T{index:02d}" for index in range(1, 10)),
        )
        self.assertEqual(len({spec.feature_name for spec in TSAY_FEATURE_SPECS}), 9)
        self.assertEqual(TSAY_PRIMARY_TEST_COUNT, 18)
        self.assertTrue(
            all(
                spec.overlap_disposition == "RELATED_BUT_MATERIALLY_DISTINCT"
                for spec in TSAY_FEATURE_SPECS
            )
        )

    def test_formula_parameters_dtype_roles_and_orientation_are_exact(self) -> None:
        expected = {
            "T01": ("directional", 1, "W=120;m=5;forecast_horizon=60"),
            "T02": ("opportunity", 1, "effective_rows=115"),
            "T03": ("opportunity", 1, "sample_lag1_autocovariance_denominator=119"),
            "T04": ("opportunity", -1, "W=60;exact_tick_change_zero"),
            "T05": ("opportunity", 1, "tail_count=12"),
            "T06": ("directional", 1, "tail_count=12"),
            "T07": ("directional", 1, "W=120;m=5;clock_bin=15m"),
            "T08": ("opportunity", 1, "W=120;m=5;clock_bin=15m"),
            "T09": ("opportunity", -1, "order_statistic=108"),
        }
        for spec in TSAY_FEATURE_SPECS:
            role, sign, parameter_fragment = expected[spec.logical_id]
            with self.subTest(spec.logical_id):
                self.assertEqual(spec.role, role)
                self.assertEqual(spec.expected_sign, sign)
                self.assertIn(parameter_fragment, spec.parameters)
                self.assertTrue(spec.formula)
                self.assertEqual(spec.computation_dtype, "float64")
                self.assertEqual(spec.output_dtype, "float32")

    def test_logical_mapping_and_fixed_model_inputs_are_outcome_free(self) -> None:
        self.assertEqual(
            D2_INPUTS,
            tuple(TSAY_LOGICAL_TO_PHYSICAL[key] for key in ("T01", "T06", "T07")),
        )
        self.assertEqual(len(O2_INPUTS), 8)
        self.assertEqual(
            tuple(MODEL_LADDER),
            (
                "D0",
                "D1",
                "D2",
                "D3",
                "D4",
                "D5",
                "D6",
                "O0",
                "O1",
                "O2",
                "O3",
                "O4",
                "O5",
                "E0",
                "E1",
                "E2",
                "E3",
            ),
        )
        forbidden = set(TARGET_COLUMNS + SECONDARY_TICK_COLUMNS)
        model_text = repr(MODEL_LADDER)
        self.assertFalse(any(column in model_text for column in forbidden))


if __name__ == "__main__":
    unittest.main()
