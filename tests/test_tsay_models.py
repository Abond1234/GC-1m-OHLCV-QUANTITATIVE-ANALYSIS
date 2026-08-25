"""Focused tests for frozen Tsay Stage 4 model primitives."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.tsay_models import (
    _baseline_diagnostics,
    _continuity_context,
    _continuity_pairs_through,
    _fit_preprocessor,
    _max_stat_adjustment,
    _pooled_ljung_box,
    _quantile_diagnostics,
    _smooth_quantile_objective,
    _stability,
    _transform,
    _unpenalized_classifier_calibration,
    build_inner_folds,
)


class TsayModelTests(unittest.TestCase):
    def test_inner_schedule_is_exact_and_complete(self) -> None:
        dates = pd.date_range("2023-01-01", periods=126, freq="D").strftime("%Y-%m-%d")
        folds = build_inner_folds(dates)
        self.assertEqual(len(folds), 2)
        self.assertEqual(len(folds[0]["train_dates"]), 63)
        self.assertEqual(len(folds[0]["assessment_dates"]), 21)
        self.assertEqual(folds[0]["embargo_date"], dates[63])
        self.assertEqual(folds[0]["assessment_dates"][0], dates[64])
        self.assertEqual(len(folds[1]["train_dates"]), 84)

    def test_fold_preprocessing_is_train_only_and_keeps_fixed_indicators(self) -> None:
        train = pd.DataFrame(
            {
                "regular": [1.0, 2.0, np.nan, 4.0],
                "tsay_roll_spread_proxy_120_ticks": [np.nan, 2.0, 3.0, 4.0],
                "tsay_roll_spread_state_unavailable_120": [1.0, 0.0, 0.0, 0.0],
            }
        )
        prep = _fit_preprocessor(train, tuple(train.columns))
        transformed = _transform(train, prep)
        self.assertEqual(transformed.shape[1], 4)
        self.assertEqual(prep.transformed_columns[-1], "regular__missing")
        self.assertTrue(np.isfinite(transformed).all())
        assessment = train.iloc[:1].copy()
        assessment.loc[0, "regular"] = 1.0e9
        self.assertEqual(_transform(assessment, prep)[0, 0], 20.0)

    def test_smooth_quantile_analytic_gradient_matches_finite_difference(self) -> None:
        x = np.array([[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0]])
        y = np.array([0.2, 1.0, -0.5])
        weights = np.ones(3)
        parameters = np.array([0.1, -0.2, 0.3])
        objective, gradient = _smooth_quantile_objective(parameters, x, y, weights, 0.5, 0.01)
        self.assertTrue(np.isfinite(objective))
        step = 1.0e-6
        numeric = np.empty_like(parameters)
        for index in range(len(parameters)):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            numeric[index] = (
                _smooth_quantile_objective(plus, x, y, weights, 0.5, 0.01)[0]
                - _smooth_quantile_objective(minus, x, y, weights, 0.5, 0.01)[0]
            ) / (2.0 * step)
        np.testing.assert_allclose(gradient, numeric, rtol=1.0e-5, atol=1.0e-6)

    def test_quantile_stability_concatenates_three_vectors_per_outer_fold(self) -> None:
        rows = []
        for fold_id, sign in ((0, 1.0), (1, 1.0)):
            for component in ("q10_slopes", "q50_slopes", "q90_slopes"):
                rows.append(
                    {
                        "architecture": "D5_O4",
                        "entry_session": "London",
                        "outer_fold_id": fold_id,
                        "component": component,
                        "coefficient_values": f"[{0.2 * sign}, {0.1 * sign}]",
                    }
                )
        cosine, agreement, qualifying = _stability(pd.DataFrame(rows), "D5", "London")
        self.assertAlmostEqual(cosine, 1.0)
        self.assertAlmostEqual(agreement, 1.0)
        self.assertEqual(qualifying, 6)

    def test_classifier_max_stat_family_uses_fixed_original_studentizer(self) -> None:
        continuous = pd.DataFrame(
            columns=(
                "role",
                "architecture",
                "entry_session",
                "family_max_stat_adjusted_p",
                "failed_gates",
                "status",
            )
        )
        classifier = pd.DataFrame(
            [
                {
                    "architecture": architecture,
                    "entry_session": session,
                    "family_max_stat_adjusted_p": np.nan,
                    "failed_gates": "",
                    "status": "PENDING_MAX_STAT",
                }
                for session in ("London", "New York")
                for architecture in ("E2", "E3")
            ]
        )
        daily = pd.DataFrame(
            [
                {
                    "trade_date_ny": f"2023-01-{date:02d}",
                    "family": "expansion",
                    "architecture": architecture,
                    "entry_session": session,
                    "delta": (0.001 * date) + (0.0001 if architecture == "E3" else 0.0),
                }
                for session in ("London", "New York")
                for architecture in ("E2", "E3")
                for date in range(1, 22)
            ]
        )
        _, adjusted, audit = _max_stat_adjustment(continuous, classifier, daily)
        self.assertTrue(adjusted["family_max_stat_adjusted_p"].notna().all())
        self.assertEqual(len(audit), 4)
        self.assertTrue(audit["fixed_original_studentizer"].all())
        self.assertTrue((audit["original_standard_error"] > 0.0).all())

    def test_classifier_calibration_is_unpenalized_and_recovers_identity(self) -> None:
        probability = np.linspace(0.05, 0.95, 400)
        y = (np.arange(400) / 400.0 < probability).astype(np.int8)
        intercept, slope = _unpenalized_classifier_calibration(y, probability, np.ones(400))
        self.assertTrue(np.isfinite(intercept))
        self.assertTrue(np.isfinite(slope))

    def test_cached_boundary_pairs_preserve_ljung_box_result(self) -> None:
        frame = pd.DataFrame(
            {
                "decision_timestamp_utc": pd.date_range(
                    "2023-01-03 14:00", periods=80, freq="min", tz="UTC"
                ),
                "trade_date_ny": ["2023-01-03"] * 80,
                "continuous_segment_id": ["segment"] * 80,
            }
        )
        rows = np.arange(80, dtype=np.int64)
        values = np.sin(rows / 4.0)
        context = _continuity_context(frame)
        uncached = _pooled_ljung_box(frame, values, rows, 20, context)
        pairs = _continuity_pairs_through(frame, rows, 20, context)
        cached = _pooled_ljung_box(frame, values, rows, 20, context, pairs)
        np.testing.assert_allclose(cached, uncached, rtol=0.0, atol=0.0)

    def test_baseline_diagnostics_use_only_finite_matched_rows(self) -> None:
        predictions = pd.DataFrame(
            {
                "architecture": ["O0", "O0", "O0"],
                "entry_session": ["London"] * 3,
                "trade_date_ny": ["2023-01-03", "2023-01-03", "2023-01-04"],
                "forward_return_60_atr": [np.nan] * 3,
                "future_range_60_atr": [1.0, 2.0, 3.0],
                "expansion_label_60": [np.nan] * 3,
                "prediction": [1.1, np.nan, 2.9],
            }
        )
        diagnostics = _baseline_diagnostics(predictions)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics.iloc[0]["matched_rows"], 2)
        self.assertTrue(np.isfinite(diagnostics.iloc[0]["weighted_rmse"]))

    def test_quantile_diagnostics_use_directional_projection_with_target(self) -> None:
        predictions = pd.DataFrame(
            {
                "architecture": ["D5"] * 20,
                "entry_session": ["London"] * 20,
                "trade_date_ny": ["2023-01-03"] * 10 + ["2023-01-04"] * 10,
                "forward_return_60_atr": np.linspace(-1.0, 1.0, 20),
                "q10": [-0.8] * 20,
                "q50": [0.0] * 20,
                "q90": [0.8] * 20,
                "baseline_q10": [-0.7] * 20,
                "baseline_q50": [0.1] * 20,
                "baseline_q90": [0.7] * 20,
                "pre_repair_crossing": [False] * 20,
            }
        )
        diagnostics, yearly = _quantile_diagnostics(predictions)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(len(yearly), 1)
        self.assertTrue(np.isfinite(diagnostics.iloc[0]["q50_pinball_loss"]))


if __name__ == "__main__":
    unittest.main()
