"""Regression tests for grouped rolling-window numerics.

The feature matrix is a frozen research artefact, so these tests pin the two
properties that make it reproducible between machines: a window that contains
no non-zero contribution sums to exactly zero (so the derived ratio stays
undefined instead of becoming a large spurious number that varies with the
NumPy build), and group identity is compared without truncating the group key.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.statistical_research.feature_engineering import (
    _rolling_sum,
    _safe_divide,
)


def _reference_rolling_sum(values, window, groups):
    """Direct per-window summation used as the correctness oracle."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    for end in range(window - 1, n):
        start = end - window + 1
        if groups[end] != groups[start]:
            continue
        block = values[start : end + 1]
        if not np.isfinite(block).all():
            continue
        out[end] = block.sum()
    return out


class RollingSumCorrectnessTests(unittest.TestCase):
    def test_matches_direct_windowed_summation(self):
        rng = np.random.default_rng(20260723)
        values = rng.normal(size=2000)
        groups = np.repeat(np.arange(20, dtype=np.int64), 100)
        for window in (2, 3, 5, 20):
            with self.subTest(window=window):
                np.testing.assert_allclose(
                    _rolling_sum(values, window, groups),
                    _reference_rolling_sum(values, window, groups),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_incomplete_and_cross_group_windows_are_null(self):
        values = np.arange(6.0)
        groups = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        result = _rolling_sum(values, 2, groups)
        self.assertTrue(np.isnan(result[0]))  # warm-up
        self.assertTrue(np.isnan(result[3]))  # spans a group boundary
        self.assertEqual(result[1], 1.0)

    def test_non_finite_values_block_the_window(self):
        values = np.array([1.0, np.nan, 3.0, 4.0])
        groups = np.zeros(4, dtype=np.int64)
        result = _rolling_sum(values, 2, groups)
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))
        self.assertEqual(result[3], 7.0)


class ZeroWindowDeterminismTests(unittest.TestCase):
    """A flat window must leave the derived ratio undefined, not merely small.

    ``_safe_divide`` treats only an exact zero denominator as undefined, so the
    null pattern of the directional-balance features depends on the rolling sum
    of absolute changes being exactly zero over a flat window.  Accumulating
    zeros cannot perturb a running total, so this holds even after a long
    stretch of large values - these tests pin that property rather than leave
    it to be rediscovered.
    """

    def test_all_zero_window_sums_to_exact_zero(self):
        # A large running total first, so the property is exercised where
        # cancellation would be lossy if the values were not exactly zero.
        values = np.concatenate([np.full(50_000, 0.1), np.zeros(200)])
        groups = np.zeros(values.size, dtype=np.int64)
        result = _rolling_sum(values, 2, groups)
        self.assertTrue(np.all(result[50_001:] == 0.0))

    def test_flat_window_leaves_directional_ratio_undefined(self):
        price_change = np.concatenate([np.full(50_000, 0.1), np.zeros(200)])
        groups = np.zeros(price_change.size, dtype=np.int64)
        ratio = _safe_divide(
            _rolling_sum(price_change, 2, groups),
            _rolling_sum(np.abs(price_change), 2, groups),
        )
        self.assertTrue(np.all(np.isnan(ratio[50_001:])))
        # No denominator may land in the band that would divide into a large
        # spurious value instead of a null.
        denominator = _rolling_sum(np.abs(price_change), 2, groups)
        finite = denominator[np.isfinite(denominator)]
        suspicious = (finite != 0.0) & (np.abs(finite) < 1e-6)
        self.assertEqual(int(suspicious.sum()), 0)

    def test_zero_windows_do_not_disturb_non_zero_windows(self):
        values = np.array([0.0, 0.0, 2.0, 3.0, 0.0, 0.0])
        groups = np.zeros(6, dtype=np.int64)
        result = _rolling_sum(values, 2, groups)
        np.testing.assert_array_equal(result[1:], np.array([0.0, 2.0, 5.0, 3.0, 0.0]))


class GroupKeyIntegrityTests(unittest.TestCase):
    def test_group_ids_beyond_int32_are_not_conflated(self):
        """Truncating the group key to int32 would merge these distinct runs."""
        groups = np.array([0, 0, 0, 2**32, 2**32, 2**32], dtype=np.int64)
        self.assertEqual(
            np.asarray(groups, dtype=np.int32)[0],
            np.asarray(groups, dtype=np.int32)[3],
            "premise: these keys collide once truncated to int32",
        )
        result = _rolling_sum(np.arange(1.0, 7.0), 2, groups)
        self.assertTrue(np.isnan(result[3]), "window spanning two runs must stay undefined")
        self.assertEqual(result[4], 9.0)


class DeviceIndependenceTests(unittest.TestCase):
    def test_rolling_sum_ignores_the_active_compute_device(self):
        """Research numerics must not change when a GPU is present."""
        from unittest import mock

        rng = np.random.default_rng(11)
        values = rng.normal(size=500)
        groups = np.repeat(np.arange(5, dtype=np.int64), 100)
        baseline = _rolling_sum(values, 4, groups)
        with mock.patch("src.compute.gpu_available", return_value=True):
            with_gpu = _rolling_sum(values, 4, groups)
        np.testing.assert_array_equal(baseline, with_gpu)


if __name__ == "__main__":
    unittest.main()
