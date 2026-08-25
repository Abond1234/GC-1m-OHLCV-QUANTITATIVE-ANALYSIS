"""Tests for the frozen Tsay Development feature-evidence procedures."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.tsay_feature_evaluation import (
    _partial_daily_ic,
    assign_frozen_quintiles,
    benjamini_hochberg,
    daily_spearman,
    evaluate_development_feature_evidence,
    frozen_quintile_edges,
    stationary_bootstrap_indices,
    stationary_bootstrap_mean_inference,
)
from src.statistical_research.tsay_feature_registry import (
    D1_INPUTS_BY_SESSION,
    O1_INPUTS_BY_SESSION,
    SESSIONS,
    TSAY_FEATURE_SPECS,
)


class TsayFeatureEvaluationTests(unittest.TestCase):
    def test_stationary_bootstrap_is_deterministic_and_null_p_is_exact_fraction(self) -> None:
        first = stationary_bootstrap_indices(12, replicates=25, seed=20260824)
        second = stationary_bootstrap_indices(12, replicates=25, seed=20260824)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(((first >= 0) & (first < 12)).all())

        inference = stationary_bootstrap_mean_inference(
            np.linspace(0.01, 0.12, 12), replicates=25, seed=20260824
        )
        self.assertAlmostEqual(inference["observed_mean"], 0.065)
        numerator = inference["null_p_value"] * 26
        self.assertAlmostEqual(numerator, round(numerator))
        self.assertLess(inference["bootstrap_lower"], inference["bootstrap_upper"])

    def test_bh_uses_complete_finite_family_and_monotone_adjustment(self) -> None:
        actual = benjamini_hochberg(np.asarray([0.01, 0.04, 0.03, 0.002]))
        np.testing.assert_allclose(actual, [0.02, 0.04, 0.04, 0.008], rtol=0.0, atol=0.0)

    def test_noninterpolated_quintiles_keep_edge_ties_in_lower_bin(self) -> None:
        values = np.arange(1.0, 11.0)
        edges = frozen_quintile_edges(values)
        np.testing.assert_array_equal(edges, [2.0, 4.0, 6.0, 8.0])
        assigned = assign_frozen_quintiles(np.asarray([2.0, 2.0001, 4.0, 8.0, 9.0, np.nan]), edges)
        np.testing.assert_array_equal(assigned, [1, 2, 2, 4, 5, -1])

    def test_daily_spearman_requires_ten_common_finite_observations(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date_ny": pd.to_datetime(["2023-01-03"] * 10 + ["2023-01-04"] * 9),
                "feature": np.arange(19, dtype=np.float64),
                "target": np.arange(19, dtype=np.float64),
            }
        )
        actual = daily_spearman(frame, feature="feature", target="target")
        self.assertEqual(actual.loc[0, "status"], "AVAILABLE")
        self.assertEqual(actual.loc[0, "ic"], 1.0)
        self.assertEqual(actual.loc[1, "status"], "INSUFFICIENT_OBSERVATIONS")
        self.assertTrue(np.isnan(actual.loc[1, "ic"]))

    def test_partial_rank_ic_uses_common_panel_and_reports_matched_unadjusted(self) -> None:
        rng = np.random.default_rng(4)
        rows: list[dict[str, object]] = []
        for date_index in range(3):
            control = rng.normal(size=50)
            signal = rng.normal(size=50)
            for row_index in range(50):
                rows.append(
                    {
                        "trade_date_ny": pd.Timestamp("2023-01-03") + pd.Timedelta(days=date_index),
                        "decision_timestamp_ny": pd.Timestamp("2023-01-03 09:00")
                        + pd.Timedelta(days=date_index, minutes=row_index),
                        "feature": control[row_index] + signal[row_index],
                        "target": control[row_index] + 0.8 * signal[row_index],
                        "control": control[row_index],
                    }
                )
        actual = _partial_daily_ic(
            pd.DataFrame.from_records(rows),
            feature="feature",
            target="target",
            controls=("control",),
            clock_bins=tuple(range(36, 40)),
            reference_clock_bin=36,
        )
        self.assertTrue(actual["status"].eq("AVAILABLE").all())
        self.assertTrue((actual["partial_ic"] > 0.5).all())
        self.assertTrue(np.isfinite(actual["matched_unadjusted_ic"]).all())

    def test_complete_evaluator_keeps_all_18_frozen_tests_and_30m_is_diagnostic(self) -> None:
        frame = self._synthetic_evidence_frame()
        result = evaluate_development_feature_evidence(frame)
        self.assertEqual(len(result.evidence_ledger), 18)
        self.assertEqual(
            set(result.evidence_ledger["logical_id"]),
            {f"T{index:02d}" for index in range(1, 10)},
        )
        self.assertEqual(set(result.evidence_ledger["entry_session"]), set(SESSIONS))
        self.assertTrue(result.diagnostic_summary["selection_authority"].eq(False).all())
        self.assertTrue(result.evidence_ledger["status"].eq("NOT_EVALUABLE").all())
        self.assertEqual(len(result.candidate_correlation), 2 * 9 * 9)

    @staticmethod
    def _synthetic_evidence_frame() -> pd.DataFrame:
        rng = np.random.default_rng(20260824)
        rows: list[dict[str, object]] = []
        all_controls: set[str] = set()
        for session in SESSIONS:
            all_controls.update(D1_INPUTS_BY_SESSION[session])
            all_controls.update(O1_INPUTS_BY_SESSION[session])
        for spec in TSAY_FEATURE_SPECS:
            all_controls.update(spec.partial_information_controls)

        observation_id = 0
        for session_index, session in enumerate(SESSIONS):
            for date_index in range(3):
                date = pd.Timestamp("2023-01-03") + pd.Timedelta(days=date_index)
                for minute in range(40):
                    observation_id += 1
                    directional = rng.normal()
                    opportunity = 1.0 + abs(rng.normal())
                    row: dict[str, object] = {
                        "observation_id": observation_id,
                        "decision_timestamp_ny": date
                        + pd.Timedelta(hours=3 + session_index * 4, minutes=minute),
                        "trade_date_ny": date,
                        "entry_session": session,
                        "research_partition": "Development",
                        "product": "GC",
                        "decision_atr_20m": 2.0,
                        "forward_return_60_atr": directional,
                        "forward_return_60_ticks": directional * 20.0,
                        "future_range_60_atr": opportunity,
                        "future_range_60_ticks": opportunity * 20.0,
                        "forward_return_30_atr": 0.5 * directional,
                        "future_range_30_atr": 0.5 * opportunity,
                        "label_available_60": True,
                        "label_available_30": True,
                    }
                    for spec in TSAY_FEATURE_SPECS:
                        row[spec.feature_name] = rng.normal()
                    for control in all_controls:
                        row[control] = rng.normal()
                    rows.append(row)
        return pd.DataFrame.from_records(rows)


if __name__ == "__main__":
    unittest.main()
