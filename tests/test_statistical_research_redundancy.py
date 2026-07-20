"""Synthetic tests for the Section 8 redundancy and incremental-information engine."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.feature_redundancy import (
    Section8Config,
    build_redundancy_analysis,
)

FAST_CONFIG = Section8Config(
    incremental_horizons=(60,),
    min_development_trading_dates=50,
    min_validation_trading_dates=20,
)


def _make_registry(names: list[str], experimental: set[str] | None = None) -> pd.DataFrame:
    experimental = experimental or set()
    return pd.DataFrame(
        {
            "feature_name": names,
            "output_dtype": ["float32"] * len(names),
            "is_experimental": [n in experimental for n in names],
        }
    )


def _make_frame(
    *, seed: int = 5, decorrelate_duplicate_in_validation: bool = False
) -> pd.DataFrame:
    """Two true outcome drivers, one near-duplicate, one pure-noise feature."""

    rng = np.random.default_rng(seed)
    rows = []
    for partition, n_dates, start in (
        ("Development", 120, "2022-01-03"),
        ("Validation", 40, "2024-01-02"),
    ):
        for date in pd.bdate_range(start, periods=n_dates):
            for session in ("London", "New York"):
                n = 30
                base = rng.normal(0.0, 1.0, n)
                indep = rng.normal(0.0, 1.0, n)
                if decorrelate_duplicate_in_validation and partition == "Validation":
                    dup = rng.normal(0.0, 1.0, n)
                else:
                    dup = base + rng.normal(0.0, 0.05, n)
                noise = rng.normal(0.0, 1.0, n)
                outcome = 0.6 * base + 0.4 * indep + rng.normal(0.0, 0.6, n)
                rows.append(
                    pd.DataFrame(
                        {
                            "trade_date_ny": np.full(n, date.to_datetime64()),
                            "entry_session": np.full(n, session, dtype=object),
                            "research_partition": np.full(n, partition, dtype=object),
                            "base_signal": base,
                            "dup_signal": dup,
                            "indep_signal": indep,
                            "noise_feat": noise,
                            "future_range_60_atr": outcome,
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def _make_section7_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    names = ["base_signal", "dup_signal", "indep_signal", "noise_feat"]
    verdicts = pd.DataFrame({"feature_name": names, "verdict": ["ADVANCE_EXPANSION"] * 4})
    shortlist = pd.DataFrame(
        {
            "feature_name": names,
            "outcome_family": ["expansion"] * 4,
            "horizon_minutes": [60] * 4,
            "session": ["New York"] * 4,
            "daily_ic_mean_val": [0.55, 0.50, 0.35, 0.02],
        }
    )
    return shortlist, verdicts


class RedundancyAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = _make_frame()
        cls.shortlist, cls.verdicts = _make_section7_inputs()
        cls.registry = _make_registry(["base_signal", "dup_signal", "indep_signal", "noise_feat"])
        cls.result = build_redundancy_analysis(
            cls.frame, cls.shortlist, cls.verdicts, cls.registry, FAST_CONFIG
        )

    def test_duplicate_clusters_with_base(self) -> None:
        clusters = self.result.cluster_members.set_index("feature_name")["cluster_id"]
        self.assertEqual(clusters["base_signal"], clusters["dup_signal"])
        self.assertNotEqual(clusters["base_signal"], clusters["indep_signal"])
        self.assertNotEqual(clusters["base_signal"], clusters["noise_feat"])

    def test_one_representative_per_cluster_and_anchor(self) -> None:
        members = self.result.cluster_members
        per_cluster = members.groupby("cluster_id")["is_representative"].sum()
        self.assertTrue(per_cluster.eq(1).all())
        self.assertEqual(self.result.summary["anchor_feature"], "base_signal")
        reps = set(self.result.representatives["feature_name"])
        self.assertIn("base_signal", reps)
        self.assertNotIn("dup_signal", reps)

    def test_independent_driver_confirms_incremental_value(self) -> None:
        frozen = self.result.frozen_feature_set.set_index("feature_name")
        self.assertTrue(frozen.loc["indep_signal", "in_frozen_set"])
        self.assertEqual(
            frozen.loc["indep_signal", "decision_reason"],
            "confirmed_incremental_information",
        )

    def test_noise_representative_is_excluded(self) -> None:
        frozen = self.result.frozen_feature_set.set_index("feature_name")
        self.assertFalse(frozen.loc["noise_feat", "in_frozen_set"])
        self.assertEqual(
            frozen.loc["noise_feat", "decision_reason"],
            "no_confirmed_incremental_information_beyond_anchor",
        )

    def test_anchor_role_and_frozen_membership(self) -> None:
        frozen = self.result.frozen_feature_set.set_index("feature_name")
        self.assertEqual(frozen.loc["base_signal", "role"], "anchor")
        self.assertTrue(frozen.loc["base_signal", "in_frozen_set"])

    def test_directional_frozen_set_is_explicitly_empty(self) -> None:
        self.assertEqual(int(self.result.summary["frozen_directional_features"]), 0)
        checks = self.result.validation_checks.set_index("check")["passed"]
        self.assertTrue(checks["directional_frozen_set_empty"])

    def test_all_validation_checks_pass(self) -> None:
        failed = self.result.validation_checks.loc[~self.result.validation_checks["passed"]]
        self.assertTrue(failed.empty, failed.to_string())

    def test_deterministic(self) -> None:
        repeat = build_redundancy_analysis(
            self.frame, self.shortlist, self.verdicts, self.registry, FAST_CONFIG
        )
        pd.testing.assert_frame_equal(repeat.frozen_feature_set, self.result.frozen_feature_set)

    def test_final_test_rows_raise(self) -> None:
        polluted = self.frame.copy()
        polluted.loc[polluted.index[:10], "research_partition"] = "Final test"
        with self.assertRaises(ValueError):
            build_redundancy_analysis(
                polluted, self.shortlist, self.verdicts, self.registry, FAST_CONFIG
            )

    def test_clustering_is_fitted_on_development_only(self) -> None:
        frame = _make_frame(seed=9, decorrelate_duplicate_in_validation=True)
        result = build_redundancy_analysis(
            frame, self.shortlist, self.verdicts, self.registry, FAST_CONFIG
        )
        clusters = result.cluster_members.set_index("feature_name")["cluster_id"]
        self.assertEqual(clusters["base_signal"], clusters["dup_signal"])


if __name__ == "__main__":
    unittest.main()
