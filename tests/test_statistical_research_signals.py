"""Synthetic tests for the Section 10 signal-construction engine."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.signal_construction import (
    Section10Config,
    build_signal_candidates,
)


def _make_inputs(*, seed: int = 5, shift_validation_prediction: float = 0.0):
    rng = np.random.default_rng(seed)
    rows = []
    entries = []
    observation = 0
    for partition, n_dates, start in (("Development", 60, "2022-01-03"), ("Validation", 30, "2024-01-02")):
        for date in pd.bdate_range(start, periods=n_dates):
            for session in ("London", "New York"):
                n = 20
                feature = rng.normal(0.0, 1.0, n)
                if partition == "Validation":
                    feature = feature + shift_validation_prediction
                ids = np.arange(observation, observation + n)
                observation += n
                rows.append(
                    pd.DataFrame(
                        {
                            "trade_date_ny": np.full(n, date.to_datetime64()),
                            "entry_session": np.full(n, session, dtype=object),
                            "research_partition": np.full(n, partition, dtype=object),
                            "only_feature": feature,
                        },
                        index=pd.Index(ids, name="observation_id"),
                    )
                )
                entries.append(
                    pd.DataFrame(
                        {
                            "entry_timestamp_utc": np.full(n, date.to_datetime64()),
                            "entry_timestamp_ny": np.full(n, date.to_datetime64()),
                            "trade_date_ny": np.full(n, date.to_datetime64()),
                            "entry_session": np.full(n, session, dtype=object),
                            "research_partition": np.full(n, partition, dtype=object),
                            "entry_price": np.full(n, 2000.0),
                            "decision_atr_20m": rng.uniform(0.5, 5.0, n),
                        },
                        index=pd.Index(ids, name="observation_id"),
                    )
                )
    frame = pd.concat(rows)
    entry_fields = pd.concat(entries)
    coefficients = pd.DataFrame(
        {
            "session": ["London", "New York"],
            "horizon_minutes": [60, 60],
            "feature_name": ["only_feature", "only_feature"],
            "coefficient": [1.0, 1.0],
        }
    )
    return frame, entry_fields, coefficients


class SignalConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame, cls.entries, cls.coefficients = _make_inputs()
        cls.result = build_signal_candidates(cls.frame, cls.entries, cls.coefficients)

    def test_gate_rate_matches_declared_percentile_on_development(self) -> None:
        thresholds = self.result.gate_thresholds
        self.assertTrue((thresholds["development_gate_rate"].sub(0.20).abs() < 0.01).all())

    def test_gate_threshold_is_fitted_on_development_only(self) -> None:
        # Mutating every Validation feature value must leave the frozen gate
        # threshold untouched, because the quantile is fitted on Development.
        mutated = self.frame.copy()
        validation_rows = mutated["research_partition"].eq("Validation")
        rng = np.random.default_rng(99)
        mutated.loc[validation_rows, "only_feature"] = rng.normal(
            50.0, 5.0, int(validation_rows.sum())
        )
        result = build_signal_candidates(mutated, self.entries, self.coefficients)
        pd.testing.assert_series_equal(
            result.gate_thresholds["gate_threshold"],
            self.result.gate_thresholds["gate_threshold"],
        )

    def test_stops_are_clamped_and_targets_scaled(self) -> None:
        cfg = Section10Config()
        candidates = self.result.candidates
        stop_ticks = candidates["stop_points"] / cfg.tick_size
        self.assertGreaterEqual(float(stop_ticks.min()), cfg.min_stop_ticks - 1e-9)
        self.assertLessEqual(float(stop_ticks.max()), cfg.max_stop_ticks + 1e-9)
        np.testing.assert_allclose(
            candidates["target_points"], cfg.target_r_multiple * candidates["stop_points"]
        )

    def test_all_validation_checks_pass(self) -> None:
        failed = self.result.validation_checks.loc[~self.result.validation_checks["passed"]]
        self.assertTrue(failed.empty, failed.to_string())

    def test_final_test_rows_raise(self) -> None:
        polluted = self.frame.copy()
        polluted.iloc[:20, polluted.columns.get_loc("research_partition")] = "Final test"
        with self.assertRaises(ValueError):
            build_signal_candidates(polluted, self.entries, self.coefficients)

    def test_missing_entry_coverage_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_signal_candidates(self.frame, self.entries.iloc[:100], self.coefficients)

    def test_deterministic(self) -> None:
        repeat = build_signal_candidates(self.frame, self.entries, self.coefficients)
        pd.testing.assert_frame_equal(repeat.candidates, self.result.candidates)

    def test_direction_variants_are_labelled_benchmarks(self) -> None:
        self.assertIn("benchmark", self.result.rule_specification["direction_variants"])


if __name__ == "__main__":
    unittest.main()
