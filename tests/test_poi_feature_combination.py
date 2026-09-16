"""Synthetic P0 tests for governed POI feature-combination research."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.statistical_research.feature_combination import (
    FES_FEATURES_BY_SESSION,
    INTERACTION_FEATURES,
    POI_CONTEXT_FEATURES,
    STAT15_FEATURES,
    DateFold,
    add_poi_interactions,
    assert_no_final_test,
    bootstrap_mean_ci,
    bootstrap_weighted_mean_ci,
    build_exact_date_folds,
    deduplicate_poi_events,
    fit_preprocessor,
    generate_oof_predictions,
    make_policy_candidates,
    model_features,
    noninterpolated_quantile,
    sequence_fixed_horizon,
    stationary_bootstrap_indices,
    summarize_policy,
    transform_features,
)


class FinalTestGuardTests(unittest.TestCase):
    def test_development_and_validation_rows_are_allowed(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date_ny": pd.to_datetime(["2023-12-29", "2024-12-31"]),
                "research_partition": ["Development", "retrospective-validation"],
                "entry_timestamp_utc": pd.to_datetime(
                    ["2023-12-29T15:00:00Z", "2024-12-31T15:00:00Z"], utc=True
                ),
            }
        )
        assert_no_final_test(frame)

    def test_utc_aware_final_test_trade_date_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date_ny": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
                "partition": ["validation"],
            }
        )
        with self.assertRaisesRegex(PermissionError, "2025 onward"):
            assert_no_final_test(frame, context="UTC-date test")

    def test_post_2024_utc_timestamp_is_rejected_even_for_2024_trade_date(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date_ny": pd.to_datetime(["2024-12-31"]),
                "research_partition": ["validation"],
                "entry_timestamp_utc": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
            }
        )
        with self.assertRaisesRegex(PermissionError, "timestamp|2025 onward"):
            assert_no_final_test(frame, context="UTC-timestamp test")

    def test_final_partition_label_is_rejected_case_insensitively(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date_ny": pd.to_datetime(["2024-06-03"]),
                "partition": ["FINAL-TEST"],
            }
        )
        with self.assertRaisesRegex(PermissionError, "Final-test partition"):
            assert_no_final_test(frame)

    def test_unknown_partition_fails_closed(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date_ny": pd.to_datetime(["2024-06-03"]),
                "research_partition": ["unlabelled_holdout"],
            }
        )
        with self.assertRaisesRegex(PermissionError, "unknown|permitted"):
            assert_no_final_test(frame, context="unknown-partition test")


class ExactDateFoldTests(unittest.TestCase):
    def test_exact_boundaries_and_eligible_tail_are_preserved(self) -> None:
        dates = pd.date_range("2022-01-03", periods=210, freq="B")
        folds = build_exact_date_folds(dates)

        self.assertEqual(len(folds), 2)
        first, tail = folds
        self.assertEqual(first.train_dates, tuple(dates[:126]))
        self.assertEqual(first.embargo_dates, (dates[126],))
        self.assertEqual(first.assessment_dates, tuple(dates[127:169]))
        self.assertEqual(tail.train_dates, tuple(dates[:168]))
        self.assertEqual(tail.embargo_dates, (dates[168],))
        self.assertEqual(tail.assessment_dates, tuple(dates[169:210]))

    def test_tail_shorter_than_minimum_is_not_emitted(self) -> None:
        dates = pd.date_range("2022-01-03", periods=190, freq="B")
        folds = build_exact_date_folds(dates)
        self.assertEqual(len(folds), 2)
        self.assertEqual(len(folds[0].assessment_dates), 42)
        self.assertEqual(len(folds[1].assessment_dates), 21)

    def test_whole_dates_are_deduplicated_after_timestamp_normalization(self) -> None:
        dates = pd.date_range("2022-01-03", periods=169, freq="B", tz="UTC")
        duplicated_intraday = [
            timestamp + offset
            for timestamp in dates
            for offset in (pd.Timedelta(hours=1), pd.Timedelta(hours=20))
        ]
        folds = build_exact_date_folds(reversed(duplicated_intraday))
        self.assertEqual(len(folds), 1)
        self.assertEqual(len(folds[0].train_dates), 126)
        self.assertEqual(len(folds[0].embargo_dates), 1)
        self.assertEqual(len(folds[0].assessment_dates), 42)


def _poi_event(
    decision_bar_id: str,
    true_retest_id: str,
    *,
    first_touch: object = False,
    structural: object = False,
    age: object = 10.0,
    width: object = 1.0,
    marker: str = "",
) -> dict[str, object]:
    return {
        "decision_bar_id": decision_bar_id,
        "true_retest_id": true_retest_id,
        "feat_first_touch": first_touch,
        "feat_15bar_structural_validation": structural,
        "feat_poi_age_minutes": age,
        "feat_poi_width_atr": width,
        "marker": marker or true_retest_id,
    }


class PoiDeduplicationTests(unittest.TestCase):
    def test_priority_is_lexicographic_in_the_frozen_order(self) -> None:
        rows = [
            _poi_event("A", "a_first", first_touch=True, age=99, width=99),
            _poi_event("A", "a_struct", structural=True, age=1, width=0.1),
            _poi_event("B", "b_plain", age=1, width=0.1),
            _poi_event("B", "b_struct", structural=True, age=99, width=99),
            _poi_event("C", "c_old", structural=True, age=20, width=0.1),
            _poi_event("C", "c_young", structural=True, age=10, width=99),
            _poi_event("D", "d_wide", age=10, width=2.0),
            _poi_event("D", "d_narrow", age=10, width=1.0),
            _poi_event("E", "e_zulu", age=10, width=1.0),
            _poi_event("E", "e_alpha", age=10, width=1.0),
        ]
        selected = deduplicate_poi_events(pd.DataFrame(rows))
        self.assertEqual(selected["decision_bar_id"].tolist(), ["A", "B", "C", "D", "E"])
        self.assertEqual(
            selected["true_retest_id"].tolist(),
            ["a_first", "b_struct", "c_young", "d_narrow", "e_alpha"],
        )

    def test_every_nonfinite_age_and_width_sort_after_finite_values(self) -> None:
        rows = [_poi_event("finite", "chosen", age=7.0, width=0.7)]
        for index, value in enumerate((np.nan, np.inf, -np.inf, "not-a-number")):
            rows.append(_poi_event("finite", f"bad_age_{index}", age=value, width=0.1))
            rows.append(_poi_event("finite", f"bad_width_{index}", age=7.0, width=value))

        selected = deduplicate_poi_events(pd.DataFrame(rows))
        self.assertEqual(selected.loc[0, "true_retest_id"], "chosen")

    def test_selection_is_invariant_to_input_order(self) -> None:
        rows = [
            _poi_event("A", "z", first_touch=False, structural=True, age=10, width=2),
            _poi_event("A", "a", first_touch=True, structural=False, age=20, width=3),
            _poi_event("B", "c", age=10, width=2),
            _poi_event("B", "b", age=10, width=1),
        ]
        expected = deduplicate_poi_events(pd.DataFrame(rows))[["decision_bar_id", "true_retest_id"]]
        for seed in range(8):
            shuffled = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)
            actual = deduplicate_poi_events(shuffled)[["decision_bar_id", "true_retest_id"]]
            pd.testing.assert_frame_equal(actual, expected)

    def test_conflicting_rows_tied_through_true_retest_id_are_rejected(self) -> None:
        rows = [
            _poi_event("A", "same", marker="payload_one"),
            _poi_event("A", "same", marker="payload_two"),
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous|conflicting"):
            deduplicate_poi_events(pd.DataFrame(rows))


class PreprocessingTests(unittest.TestCase):
    def test_finite_only_median_iqr_imputation_scaling_and_clip(self) -> None:
        training = pd.DataFrame(
            {
                "varying": [1.0, 3.0, 5.0, 7.0, np.nan, np.inf, -np.inf],
                "constant": [2.0, 2.0, 2.0, 2.0, np.nan, np.inf, -np.inf],
            }
        )
        state = fit_preprocessor(training, ("varying", "constant"))
        np.testing.assert_allclose(state.medians, [4.0, 2.0])
        np.testing.assert_allclose(state.scales, [3.0, 1.0])

        assessment = pd.DataFrame(
            {
                "varying": [4.0, np.nan, 100.0, -100.0],
                "constant": [2.0, np.inf, 3.0, 1.0],
            }
        )
        transformed = transform_features(assessment, state)
        np.testing.assert_allclose(
            transformed,
            np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 1.0], [-10.0, -1.0]]),
        )
        self.assertTrue(np.isfinite(transformed).all())

    def test_training_feature_with_no_finite_value_is_integrity_failure(self) -> None:
        frame = pd.DataFrame(
            {
                "valid": [1.0, 2.0, 3.0],
                "invalid": [np.nan, np.inf, -np.inf],
            }
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_preprocessor(frame, ("valid", "invalid"))


class FeatureDefinitionTests(unittest.TestCase):
    def test_all_predeclared_interactions_have_exact_products(self) -> None:
        source = pd.DataFrame(
            {
                "poi_direction_sign": [-1.0],
                "lagged_volume_return_spearman_30": [2.0],
                "feat_poi_width_atr": [3.0],
                "atr_ratio_20_60": [4.0],
                "feat_displacement_efficiency": [5.0],
                "efficiency_ratio_30": [6.0],
                "feat_approach_15m_candle_overlap_ratio": [7.0],
                "return_sign_change_rate_30": [8.0],
                "feat_touch_penetration_fraction": [9.0],
                "current_range_over_atr": [10.0],
                "feat_approach_15m_range_compression_ratio": [11.0],
                "return_acf_energy_60": [12.0],
                "feat_approach_15m_relative_volume": [13.0],
                "range_volume_spearman_30": [14.0],
                "feat_distance_from_vwap_atr": [15.0],
                "normalized_ols_slope_30": [16.0],
            }
        )
        result = add_poi_interactions(source)
        expected = {
            "int_poi_direction_x_lagged_volume": -2.0,
            "int_poi_width_x_atr_ratio": 12.0,
            "int_displacement_x_efficiency": 30.0,
            "int_approach_overlap_x_sign_change": 56.0,
            "int_touch_penetration_x_current_range": 90.0,
            "int_approach_compression_x_acf": 132.0,
            "int_retest_volume_x_range_volume": 182.0,
            "int_poi_vwap_x_trend": -240.0,
        }
        self.assertEqual(set(INTERACTION_FEATURES), set(expected))
        for name, value in expected.items():
            self.assertEqual(result.loc[0, name], value)
        self.assertFalse(set(INTERACTION_FEATURES) & set(source.columns))

    def test_model_membership_is_exact_and_session_specific(self) -> None:
        london_fes = FES_FEATURES_BY_SESSION["London"]
        new_york_fes = FES_FEATURES_BY_SESSION["New York"]
        self.assertEqual(model_features("GENERAL", "GEN2_FES4", "London"), london_fes)
        self.assertEqual(model_features("GENERAL", "GEN2_FES4", "New York"), new_york_fes)

        poi_all = model_features("POI", "POI5_ALL", "New York")
        expected = POI_CONTEXT_FEATURES + STAT15_FEATURES + new_york_fes + INTERACTION_FEATURES
        self.assertEqual(poi_all, tuple(dict.fromkeys(expected)))
        self.assertFalse(set(london_fes) & set(poi_all))
        self.assertEqual(len(poi_all), len(set(poi_all)))

        with self.assertRaises(KeyError):
            model_features("UNKNOWN", "GEN1_STAT15", "London")
        with self.assertRaises(KeyError):
            model_features("GENERAL", "GEN1_STAT15", "Asia")


class OofIsolationTests(unittest.TestCase):
    @staticmethod
    def _frame() -> tuple[pd.DataFrame, DateFold]:
        dates = pd.date_range("2021-01-04", periods=65, freq="B")
        feature = np.linspace(-2.0, 2.0, len(dates))
        frame = pd.DataFrame(
            {
                "trade_date_ny": dates,
                "feature": feature,
                "target": 1.25 * feature + 0.4,
            }
        )
        fold = DateFold(
            fold_id=1,
            train_dates=tuple(dates[:50]),
            embargo_dates=(dates[50],),
            assessment_dates=tuple(dates[51:61]),
        )
        return frame, fold

    def test_assessment_targets_cannot_influence_oof_predictions(self) -> None:
        frame, fold = self._frame()
        baseline, audit, coefficients = generate_oof_predictions(
            frame,
            [fold],
            features=("feature",),
            target="target",
        )
        mutated = frame.copy()
        assessment = mutated["trade_date_ny"].isin(fold.assessment_dates)
        mutated.loc[assessment, "target"] = np.linspace(1e9, -1e9, assessment.sum())
        after_mutation, _, _ = generate_oof_predictions(
            mutated,
            [fold],
            features=("feature",),
            target="target",
        )

        np.testing.assert_allclose(baseline.loc[assessment], after_mutation.loc[assessment])
        self.assertTrue(baseline.loc[assessment].notna().all())
        self.assertTrue(baseline.loc[~assessment].isna().all())
        self.assertEqual(audit.loc[0, "train_rows"], 50)
        self.assertEqual(audit.loc[0, "assessment_rows"], 10)
        self.assertEqual(audit.loc[0, "embargo_date_count"], 1)
        self.assertEqual(coefficients["fold_id"].unique().tolist(), [1])


class PolicyAndBootstrapTests(unittest.TestCase):
    def test_noninterpolated_quantile_uses_one_based_ceil_and_filters_nonfinite(self) -> None:
        values = [10.0, 1.0, 5.0, np.nan, np.inf, 7.0, 3.0, -np.inf, 9.0, 2.0]
        self.assertEqual(noninterpolated_quantile(values, 0.10), 1.0)
        self.assertEqual(noninterpolated_quantile(values, 0.50), 5.0)
        self.assertEqual(noninterpolated_quantile(values, 0.90), 10.0)
        with self.assertRaises(ValueError):
            noninterpolated_quantile([], 0.50)
        with self.assertRaises(ValueError):
            noninterpolated_quantile([1.0], 1.0)

    def test_policy_tails_are_inclusive_and_oriented_by_branch(self) -> None:
        general = pd.DataFrame(
            {
                "score": [-1.0, 0.0, 1.0, np.nan],
                "forward_return_60_ticks": [-4.0, 99.0, 3.0, 99.0],
            }
        )
        general_candidates = make_policy_candidates(
            general,
            branch="GENERAL",
            prediction_column="score",
            low_threshold=-1.0,
            high_threshold=1.0,
        )
        self.assertEqual(general_candidates["prediction_tail"].tolist(), ["lower", "upper"])
        self.assertEqual(general_candidates["trade_direction"].tolist(), ["short", "long"])
        np.testing.assert_allclose(general_candidates["gross_ticks"], [4.0, 3.0])

        poi = pd.DataFrame(
            {
                "score": [-1.0, 1.0],
                "signed_continuation_return_60_ticks": [-2.0, 5.0],
            }
        )
        poi_candidates = make_policy_candidates(
            poi,
            branch="POI",
            prediction_column="score",
            low_threshold=-1.0,
            high_threshold=1.0,
        )
        self.assertEqual(
            poi_candidates["trade_direction"].tolist(), ["POI reversal", "POI continuation"]
        )
        np.testing.assert_allclose(poi_candidates["gross_ticks"], [2.0, 5.0])

    def test_sequencing_uses_ties_and_treats_entry_at_exit_as_overlap(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "marker": "same-z",
                    "entry_timestamp_utc": "2024-01-02T10:00:00Z",
                    "exit_timestamp_utc_60": "2024-01-02T10:30:00Z",
                    "observation_id": 2,
                    "true_retest_id": "z",
                },
                {
                    "marker": "same-b",
                    "entry_timestamp_utc": "2024-01-02T10:00:00Z",
                    "exit_timestamp_utc_60": "2024-01-02T10:20:00Z",
                    "observation_id": 1,
                    "true_retest_id": "b",
                },
                {
                    "marker": "same-a",
                    "entry_timestamp_utc": "2024-01-02T10:00:00Z",
                    "exit_timestamp_utc_60": "2024-01-02T10:10:00Z",
                    "observation_id": 1,
                    "true_retest_id": "a",
                },
                {
                    "marker": "at-exit",
                    "entry_timestamp_utc": "2024-01-02T10:10:00Z",
                    "exit_timestamp_utc_60": "2024-01-02T10:20:00Z",
                    "observation_id": 3,
                    "true_retest_id": "",
                },
                {
                    "marker": "after",
                    "entry_timestamp_utc": "2024-01-02T10:10:01Z",
                    "exit_timestamp_utc_60": "2024-01-02T10:30:00Z",
                    "observation_id": 4,
                    "true_retest_id": "",
                },
                {
                    "marker": "late",
                    "entry_timestamp_utc": "2024-01-02T10:31:00Z",
                    "exit_timestamp_utc_60": "2024-01-02T11:31:00Z",
                    "observation_id": 5,
                    "true_retest_id": "",
                },
            ]
        )
        expected = ["same-a", "after", "late"]
        for seed in range(5):
            sequenced = sequence_fixed_horizon(frame.sample(frac=1.0, random_state=seed))
            self.assertEqual(sequenced["marker"].tolist(), expected)

    def test_cost_is_charged_once_per_trade_and_zero_trade_dates_enter_sharpe(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        trades = pd.DataFrame(
            {
                "trade_date_ny": [dates[0], dates[0], dates[2]],
                "gross_ticks": [5.0, -1.0, 3.0],
            }
        )
        frictionless = summarize_policy(
            trades,
            eligible_dates=dates,
            round_trip_cost_ticks=0.0,
            seed=17,
        )
        costed = summarize_policy(
            trades,
            eligible_dates=dates,
            round_trip_cost_ticks=2.0,
            seed=17,
        )
        self.assertAlmostEqual(costed["mean_net_ticks"], frictionless["mean_net_ticks"] - 2.0)
        self.assertAlmostEqual(costed["total_net_ticks"], 1.0)
        self.assertAlmostEqual(costed["profit_factor"], 4.0 / 3.0)
        expected_daily = np.array([0.0, 0.0, 1.0])
        expected_sharpe = np.sqrt(252.0) * expected_daily.mean() / expected_daily.std(ddof=1)
        self.assertAlmostEqual(costed["daily_net_ticks_sharpe"], expected_sharpe)
        self.assertEqual(costed["trading_date_count"], 2)

    def test_profit_factor_is_undefined_when_there_is_neither_profit_nor_loss(self) -> None:
        trades = pd.DataFrame(
            {
                "trade_date_ny": pd.to_datetime(["2024-01-02"]),
                "gross_ticks": [2.6],
            }
        )
        summary = summarize_policy(
            trades,
            eligible_dates=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            round_trip_cost_ticks=2.6,
            seed=19,
        )
        self.assertTrue(math.isnan(summary["profit_factor"]))

    def test_stationary_bootstrap_is_deterministic_bounded_and_weighted(self) -> None:
        first = stationary_bootstrap_indices(7, replicates=12, restart_probability=0.2, seed=1234)
        second = stationary_bootstrap_indices(7, replicates=12, restart_probability=0.2, seed=1234)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (12, 7))
        self.assertTrue(((first >= 0) & (first < 7)).all())

        mean_ci_one = bootstrap_mean_ci([1.0, np.nan, 2.0, np.inf, 3.0], seed=7)
        mean_ci_two = bootstrap_mean_ci([1.0, np.nan, 2.0, np.inf, 3.0], seed=7)
        self.assertEqual(mean_ci_one, mean_ci_two)
        self.assertGreaterEqual(mean_ci_one[0], 1.0)
        self.assertLessEqual(mean_ci_one[1], 3.0)

        weighted_ci = bootstrap_weighted_mean_ci(
            sums=[10.0, 0.0, 20.0],
            counts=[2.0, 0.0, 4.0],
            seed=11,
        )
        np.testing.assert_allclose(weighted_ci, [5.0, 5.0])
        with self.assertRaises(ValueError):
            stationary_bootstrap_indices(0)
        with self.assertRaises(ValueError):
            stationary_bootstrap_indices(4, restart_probability=0.0)


if __name__ == "__main__":
    unittest.main()
