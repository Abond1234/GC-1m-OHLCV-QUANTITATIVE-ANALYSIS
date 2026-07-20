"""Synthetic tests for the Section 7 univariate feature evaluation engine."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.feature_evaluation import (
    EVALUATION_PARTITIONS,
    OUTCOME_FAMILY_ATR_TEMPLATES,
    OUTCOME_FAMILY_TICK_TEMPLATES,
    Section7Config,
    _assign_buckets,
    _bootstrap_matrix_ci,
    _fit_development_bucket_edges,
    benjamini_hochberg_q_values,
    build_evaluation_frame,
    build_univariate_evaluation,
    evaluated_feature_names,
)
from src.statistical_research.labels import FORWARD_HORIZONS_MINUTES


FAST_CONFIG = Section7Config(
    bootstrap_replicates=100,
    min_development_trading_dates=50,
    min_validation_trading_dates=20,
    min_development_observations=1_000,
    min_validation_observations=400,
)


def _make_registry(feature_names: list[str], experimental: set[str] | None = None) -> pd.DataFrame:
    experimental = experimental or set()
    return pd.DataFrame(
        {
            "feature_name": feature_names,
            "output_dtype": ["float32"] * len(feature_names),
            "is_experimental": [name in experimental for name in feature_names],
        }
    )


def _make_evaluation_frame(
    *,
    seed: int = 7,
    dev_dates: int = 120,
    val_dates: int = 40,
    obs_per_session_date: int = 30,
    signal_strength: float = 0.6,
    tick_scale: float = 25.0,
    london_only_signal: bool = False,
) -> pd.DataFrame:
    """Synthetic frame with one planted signal feature and one noise feature."""

    rng = np.random.default_rng(seed)
    rows = []
    for partition, n_dates, start in (
        ("Development", dev_dates, "2022-01-03"),
        ("Validation", val_dates, "2024-01-02"),
    ):
        dates = pd.bdate_range(start, periods=n_dates)
        for date in dates:
            for session in ("London", "New York"):
                n = obs_per_session_date
                signal = rng.normal(0.0, 1.0, n)
                noise_feature = rng.normal(0.0, 1.0, n)
                strength = signal_strength
                if london_only_signal and session == "New York":
                    strength = 0.0
                base_outcome = strength * signal + rng.normal(0.0, 1.0, n)
                record = {
                    "trade_date_ny": np.full(n, date.to_datetime64()),
                    "entry_session": np.full(n, session, dtype=object),
                    "research_partition": np.full(n, partition, dtype=object),
                    "entry_year": np.full(n, date.year, dtype=np.int16),
                    "signal_feature": signal,
                    "noise_feature": noise_feature,
                }
                for horizon in FORWARD_HORIZONS_MINUTES:
                    outcome = base_outcome + rng.normal(0.0, 0.1, n)
                    expansion = np.abs(outcome) + rng.normal(0.0, 0.1, n)
                    record[OUTCOME_FAMILY_ATR_TEMPLATES["direction"].format(h=horizon)] = outcome
                    record[OUTCOME_FAMILY_TICK_TEMPLATES["direction"].format(h=horizon)] = outcome * tick_scale
                    record[OUTCOME_FAMILY_ATR_TEMPLATES["expansion"].format(h=horizon)] = expansion
                    record[OUTCOME_FAMILY_TICK_TEMPLATES["expansion"].format(h=horizon)] = expansion * tick_scale
                rows.append(pd.DataFrame(record))
    return pd.concat(rows, ignore_index=True)


class BenjaminiHochbergTests(unittest.TestCase):
    def test_hand_computed_example(self) -> None:
        p = np.array([0.005, 0.011, 0.02, 0.04, 0.5])
        expected = np.array([0.025, 0.0275, 0.0333333333, 0.05, 0.5])
        np.testing.assert_allclose(benjamini_hochberg_q_values(p), expected, rtol=1e-6)

    def test_nan_passthrough_and_monotone_clip(self) -> None:
        p = np.array([0.04, np.nan, 0.01])
        q = benjamini_hochberg_q_values(p)
        self.assertTrue(np.isnan(q[1]))
        self.assertTrue(np.all(q[np.isfinite(q)] <= 1.0))
        self.assertAlmostEqual(q[2], 0.02, places=10)


class BootstrapTests(unittest.TestCase):
    def test_deterministic_given_seed(self) -> None:
        rng = np.random.default_rng(3)
        daily = rng.normal(0.02, 0.05, size=(180, 4))
        first = _bootstrap_matrix_ci(daily, seed=99, replicates=200, confidence=0.95)
        second = _bootstrap_matrix_ci(daily, seed=99, replicates=200, confidence=0.95)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_interval_brackets_true_mean(self) -> None:
        rng = np.random.default_rng(4)
        daily = rng.normal(0.1, 0.02, size=(250, 1))
        low, high = _bootstrap_matrix_ci(daily, seed=1, replicates=500, confidence=0.95)
        self.assertLess(low[0], 0.1)
        self.assertGreater(high[0], 0.1)


class BucketFittingTests(unittest.TestCase):
    def test_edges_fitted_on_development_only(self) -> None:
        development = np.concatenate([np.zeros(500), np.ones(500)]) + np.linspace(0, 0.001, 1000)
        shifted_validation = development + 100.0
        edges = _fit_development_bucket_edges(development, 5)
        buckets = _assign_buckets(shifted_validation, edges)
        self.assertTrue((buckets == len(edges)).all())

    def test_low_cardinality_feature_collapses_buckets(self) -> None:
        binary = np.array([0.0, 1.0] * 200)
        edges = _fit_development_bucket_edges(binary, 5)
        self.assertLessEqual(len(edges), 2)
        buckets = _assign_buckets(binary, edges)
        self.assertLessEqual(len(np.unique(buckets)), len(edges) + 1)


class EvaluationFrameTests(unittest.TestCase):
    def _make_artifacts(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = 30
        ids = np.arange(n, dtype=np.int64)
        partitions = np.array(
            ["Development"] * 12 + ["Validation"] * 8 + ["Final test"] * 10, dtype=object
        )
        labels = pd.DataFrame(
            {
                "observation_id": ids,
                "trade_date_ny": pd.to_datetime("2023-03-01"),
                "entry_session": "London",
                "research_partition": partitions,
                "atr_normalization_available": True,
            }
        )
        for horizon in FORWARD_HORIZONS_MINUTES:
            labels[f"label_available_{horizon}"] = True
            for family, template in OUTCOME_FAMILY_ATR_TEMPLATES.items():
                labels[template.format(h=horizon)] = 0.5
            for family, template in OUTCOME_FAMILY_TICK_TEMPLATES.items():
                labels[template.format(h=horizon)] = 12.0
        labels.loc[3, "label_available_60"] = False
        matrix = pd.DataFrame({"observation_id": ids, "alpha_feature": np.linspace(-1, 1, n)})
        registry = _make_registry(["alpha_feature"])
        return matrix, labels, registry

    def test_final_test_and_incomplete_rows_excluded(self) -> None:
        matrix, labels, registry = self._make_artifacts()
        frame, summary = build_evaluation_frame(matrix, labels, registry)
        self.assertEqual(set(frame["research_partition"]), {"Development", "Validation"})
        self.assertEqual(summary["final_test_observations_excluded"], 10)
        self.assertEqual(summary["incomplete_label_rows_dropped"], 1)
        self.assertEqual(summary["evaluation_observations"], 19)

    def test_categorical_features_are_not_evaluated(self) -> None:
        registry = pd.DataFrame(
            {
                "feature_name": ["numeric_a", "entry_session", "day_of_week"],
                "output_dtype": ["float32", "category", "category"],
                "is_experimental": [False, False, False],
            }
        )
        self.assertEqual(evaluated_feature_names(registry), ["numeric_a"])


class UnivariateEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _make_registry(["signal_feature", "noise_feature"])
        cls.frame = _make_evaluation_frame()
        cls.result = build_univariate_evaluation(cls.frame, cls.registry, FAST_CONFIG)

    def test_planted_signal_advances_and_noise_is_rejected(self) -> None:
        verdicts = self.result.feature_verdicts.set_index("feature_name")["verdict"]
        self.assertEqual(verdicts["signal_feature"], "ADVANCE_DIRECTIONAL")
        self.assertEqual(verdicts["noise_feature"], "NO_EVIDENCE")

    def test_noise_never_passes_bh_screen(self) -> None:
        cells = self.result.cell_results
        noise_q = cells.loc[
            cells["feature_name"].eq("noise_feature")
            & cells["research_partition"].eq("Development"),
            "development_q_value",
        ]
        self.assertTrue((noise_q.dropna() > FAST_CONFIG.max_development_q_value).all())

    def test_q_values_absent_outside_development(self) -> None:
        cells = self.result.cell_results
        validation_q = cells.loc[cells["research_partition"].eq("Validation"), "development_q_value"]
        self.assertTrue(validation_q.isna().all())

    def test_monotonicity_detected_for_signal(self) -> None:
        cells = self.result.cell_results
        row = cells.loc[
            cells["feature_name"].eq("signal_feature")
            & cells["outcome_family"].eq("direction")
            & cells["horizon_minutes"].eq(60)
            & cells["session"].eq("London")
            & cells["research_partition"].eq("Development")
        ].iloc[0]
        self.assertGreaterEqual(abs(row["bucket_monotonicity"]), 0.9)

    def test_all_validation_checks_pass(self) -> None:
        failed = self.result.validation_checks.loc[~self.result.validation_checks["passed"]]
        self.assertTrue(failed.empty, failed.to_string())

    def test_deterministic_shortlist(self) -> None:
        repeat = build_univariate_evaluation(self.frame, self.registry, FAST_CONFIG)
        pd.testing.assert_frame_equal(repeat.shortlist, self.result.shortlist)

    def test_final_test_rows_raise(self) -> None:
        polluted = self.frame.copy()
        polluted.loc[polluted.index[:5], "research_partition"] = "Final test"
        with self.assertRaises(ValueError):
            build_univariate_evaluation(polluted, self.registry, FAST_CONFIG)


class SessionSeparationTests(unittest.TestCase):
    def test_london_only_signal_stays_in_london(self) -> None:
        registry = _make_registry(["signal_feature", "noise_feature"])
        frame = _make_evaluation_frame(seed=11, london_only_signal=True)
        result = build_univariate_evaluation(frame, registry, FAST_CONFIG)
        cells = result.cell_results
        signal_direction = cells.loc[
            cells["feature_name"].eq("signal_feature")
            & cells["outcome_family"].eq("direction")
            & cells["horizon_minutes"].eq(60)
            & cells["research_partition"].eq("Development")
        ].set_index("session")["daily_ic_mean"]
        self.assertGreater(abs(signal_direction["London"]), 0.3)
        self.assertLess(abs(signal_direction["New York"]), 0.1)


class ShortlistGateTests(unittest.TestCase):
    def test_direction_requires_economic_spread(self) -> None:
        registry = _make_registry(["signal_feature", "noise_feature"])
        # Strong IC but tick outcomes scaled so the Q5-Q1 spread stays below the
        # economic hurdle: the direction family must not shortlist the feature.
        frame = _make_evaluation_frame(seed=13, tick_scale=0.5)
        result = build_univariate_evaluation(frame, registry, FAST_CONFIG)
        direction_rows = result.shortlist.loc[result.shortlist["outcome_family"].eq("direction")]
        self.assertTrue(direction_rows.empty)
        verdicts = result.feature_verdicts.set_index("feature_name")["verdict"]
        self.assertIn(verdicts["signal_feature"], {"ADVANCE_EXPANSION", "WEAK_UNSTABLE"})

    def test_partition_constant_is_exported(self) -> None:
        self.assertEqual(EVALUATION_PARTITIONS, ("Development", "Validation"))


if __name__ == "__main__":
    unittest.main()
