"""Synthetic tests for the Section 9 multivariate benchmark engine."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.multivariate import (
    Section9Config,
    _auc,
    _fit_logistic_irls,
    _walk_forward_folds,
    build_multivariate_benchmarks,
)


FAST_CONFIG = Section9Config(
    bootstrap_replicates=200,
    horizons=(60,),
    ridge_lambda_grid=(1e-3, 1e-1),
)


def _make_frame(*, seed: int = 3, second_driver_weight: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic frame with anchor + second driver + noise feature."""

    rng = np.random.default_rng(seed)
    rows = []
    label_rows = []
    observation = 0
    for partition, years in (("Development", (2021, 2022, 2023)), ("Validation", (2024,))):
        for year in years:
            for date in pd.bdate_range(f"{year}-01-02", periods=60):
                for session in ("London", "New York"):
                    n = 25
                    anchor = rng.normal(0.0, 1.0, n)
                    second = rng.normal(0.0, 1.0, n)
                    noise = rng.normal(0.0, 1.0, n)
                    outcome = anchor + second_driver_weight * second + rng.normal(0.0, 0.8, n)
                    ids = np.arange(observation, observation + n)
                    observation += n
                    rows.append(
                        pd.DataFrame(
                            {
                                "trade_date_ny": np.full(n, date.to_datetime64()),
                                "entry_session": np.full(n, session, dtype=object),
                                "research_partition": np.full(n, partition, dtype=object),
                                "anchor_feat": anchor,
                                "second_feat": second,
                                "noise_feat": noise,
                                "future_range_60_atr": outcome,
                            },
                            index=pd.Index(ids, name="observation_id"),
                        )
                    )
                    label_rows.append(
                        pd.DataFrame(
                            {"expansion_label_60": outcome > np.quantile(outcome, 0.8)},
                            index=pd.Index(ids, name="observation_id"),
                        )
                    )
    return pd.concat(rows), pd.concat(label_rows)


class HelperTests(unittest.TestCase):
    def test_auc_on_separable_scores(self) -> None:
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = np.array([True, True, True, False, False])
        self.assertEqual(_auc(scores, labels), 1.0)
        self.assertTrue(np.isnan(_auc(scores, np.zeros(5, dtype=bool))))

    def test_logistic_irls_recovers_separation(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 1.0, size=(2_000, 2))
        y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.5, 2_000)) > 0
        beta = _fit_logistic_irls(x, y, 1e-4, 50)
        p = 1.0 / (1.0 + np.exp(-(np.hstack([np.ones((len(x), 1)), x]) @ beta)))
        self.assertGreater(_auc(p, y), 0.85)
        self.assertGreater(beta[1], 0.0)

    def test_walk_forward_folds_are_chronological_with_embargo(self) -> None:
        frame, _ = _make_frame()
        sub = frame.loc[frame["entry_session"].eq("London")]
        date_codes, date_index = pd.factorize(sub["trade_date_ny"], sort=True)
        years = pd.DatetimeIndex(sub["trade_date_ny"]).year.to_numpy()
        partitions = sub["research_partition"].astype(str).to_numpy()
        folds = _walk_forward_folds(years, date_codes, date_index, partitions, 1)
        self.assertEqual([f["test_year"] for f in folds], [2022, 2023])
        for fold in folds:
            train_years = {int(y) for y in fold["train_years"].split(",")}
            self.assertTrue(all(y < fold["test_year"] for y in train_years))
            self.assertFalse((fold["train_rows"] & fold["test_rows"]).any())
            self.assertGreaterEqual(fold["embargoed_dates"], 1)
            self.assertFalse((partitions[fold["test_rows"]] == "Validation").any())


class MultivariateBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame, cls.labels = _make_frame()
        cls.features = ["anchor_feat", "second_feat", "noise_feat"]
        cls.result = build_multivariate_benchmarks(
            cls.frame, cls.features, "anchor_feat", cls.labels, FAST_CONFIG
        )

    def test_model_beats_anchor_when_second_driver_exists(self) -> None:
        verdicts = self.result.verdicts
        self.assertTrue(verdicts["verdict"].eq("MODEL_ADVANCES").all(), verdicts.to_string())

    def test_no_advancement_when_anchor_is_the_only_driver(self) -> None:
        frame, labels = _make_frame(seed=11, second_driver_weight=0.0)
        result = build_multivariate_benchmarks(
            frame, self.features, "anchor_feat", labels, FAST_CONFIG
        )
        self.assertTrue(result.verdicts["verdict"].eq("ANCHOR_SUFFICIENT").all())

    def test_coefficients_favor_true_drivers(self) -> None:
        coef = self.result.coefficient_table
        pivot = coef.groupby("feature_name")["coefficient"].mean()
        self.assertGreater(pivot["anchor_feat"], abs(pivot["noise_feat"]))
        self.assertGreater(pivot["second_feat"], abs(pivot["noise_feat"]))

    def test_classification_auc_beats_anchor_and_is_calibrated(self) -> None:
        cls_rows = self.result.classification_results
        val = cls_rows.loc[cls_rows["research_partition"].eq("Validation")]
        self.assertTrue((val["model_auc"] > 0.6).all())
        self.assertTrue((val["model_auc"] >= val["anchor_auc"] - 0.02).all())
        self.assertTrue(bool(self.result.summary["calibration_monotone_on_validation"]))

    def test_all_validation_checks_pass(self) -> None:
        failed = self.result.validation_checks.loc[~self.result.validation_checks["passed"]]
        self.assertTrue(failed.empty, failed.to_string())

    def test_final_test_rows_raise(self) -> None:
        polluted = self.frame.copy()
        polluted.iloc[:50, polluted.columns.get_loc("research_partition")] = "Final test"
        with self.assertRaises(ValueError):
            build_multivariate_benchmarks(
                polluted, self.features, "anchor_feat", self.labels, FAST_CONFIG
            )

    def test_anchor_must_be_frozen_feature(self) -> None:
        with self.assertRaises(ValueError):
            build_multivariate_benchmarks(
                self.frame, self.features, "not_a_feature", self.labels, FAST_CONFIG
            )

    def test_deterministic(self) -> None:
        repeat = build_multivariate_benchmarks(
            self.frame, self.features, "anchor_feat", self.labels, FAST_CONFIG
        )
        pd.testing.assert_frame_equal(repeat.verdicts, self.result.verdicts)

    def test_tree_models_recorded_as_deferred(self) -> None:
        self.assertIn("deferred", str(self.result.summary["tree_models"]))


if __name__ == "__main__":
    unittest.main()
