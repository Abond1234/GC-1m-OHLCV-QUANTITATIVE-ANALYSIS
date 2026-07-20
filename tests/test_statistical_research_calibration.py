"""Tests for isotonic probability recalibration."""

from __future__ import annotations

import unittest

import numpy as np

from src.statistical_research.calibration import (
    brier_score,
    fit_isotonic_calibration,
)


class IsotonicCalibrationTests(unittest.TestCase):
    def test_output_is_monotone_and_bounded(self) -> None:
        rng = np.random.default_rng(3)
        predictions = rng.uniform(0, 1, 5_000)
        outcomes = rng.uniform(0, 1, 5_000) < predictions**2
        calibration = fit_isotonic_calibration(predictions, outcomes)
        grid = np.linspace(0, 1, 101)
        mapped = calibration.apply(grid)
        self.assertTrue(np.all(np.diff(mapped) >= -1e-12))
        self.assertTrue(np.all((mapped >= 0.0) & (mapped <= 1.0)))

    def test_recalibration_fixes_overconfident_probabilities(self) -> None:
        rng = np.random.default_rng(7)
        true_probability = rng.uniform(0.05, 0.6, 20_000)
        outcomes = rng.uniform(0, 1, 20_000) < true_probability
        overconfident = np.clip(true_probability * 1.6, 0, 1)
        calibration = fit_isotonic_calibration(overconfident, outcomes)
        recalibrated = calibration.apply(overconfident)
        self.assertLess(
            brier_score(recalibrated, outcomes), brier_score(overconfident, outcomes)
        )

    def test_development_only_fit_is_frozen_for_new_data(self) -> None:
        rng = np.random.default_rng(11)
        development_predictions = rng.uniform(0, 1, 5_000)
        development_outcomes = rng.uniform(0, 1, 5_000) < development_predictions
        calibration = fit_isotonic_calibration(development_predictions, development_outcomes)
        before = calibration.apply(np.array([0.2, 0.5, 0.8])).copy()
        rng.uniform(0, 1, 1_000)
        after = calibration.apply(np.array([0.2, 0.5, 0.8]))
        np.testing.assert_array_equal(before, after)

    def test_perfectly_calibrated_input_is_preserved(self) -> None:
        rng = np.random.default_rng(19)
        predictions = rng.uniform(0.1, 0.9, 50_000)
        outcomes = rng.uniform(0, 1, 50_000) < predictions
        calibration = fit_isotonic_calibration(predictions, outcomes)
        grid = np.linspace(0.15, 0.85, 15)
        np.testing.assert_allclose(calibration.apply(grid), grid, atol=0.05)

    def test_too_few_observations_raise(self) -> None:
        with self.assertRaises(ValueError):
            fit_isotonic_calibration(np.array([0.5]), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
