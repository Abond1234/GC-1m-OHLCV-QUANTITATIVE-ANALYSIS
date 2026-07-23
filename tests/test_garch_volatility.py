"""Tests for the GARCH(1,1) conditional volatility feature.

The contract in project_docs/garch_volatility_research_contract.md makes two
promises that matter more than the fitted numbers: no future information may
enter the feature, and the regime threshold must come from development bars
alone.  Both are asserted here directly.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.garch_volatility import (
    GarchConfig,
    GarchParameters,
    attach_garch_features,
    build_garch_features,
    fit_development_garch,
    forward_conditional_volatility,
    log_returns_by_product,
)

CONFIG = GarchConfig(products=("GC",))


def _bars(n_dev: int = 1500, n_val: int = 400, n_final: int = 300, seed: int = 20260723):
    """Synthetic bars with volatility clustering and the three partitions."""
    rng = np.random.default_rng(seed)
    total = n_dev + n_val + n_final
    shocks = rng.normal(0.0, 1.0, total)
    sigma = np.empty(total)
    sigma[0] = 1.0
    for t in range(1, total):
        sigma[t] = np.sqrt(
            0.05 + 0.10 * (sigma[t - 1] * shocks[t - 1]) ** 2 + 0.85 * sigma[t - 1] ** 2
        )
    returns = sigma * shocks / 10_000.0
    close = 2000.0 * np.exp(np.cumsum(returns))
    partitions = ["development"] * n_dev + ["validation"] * n_val + ["final_test"] * n_final
    return pd.DataFrame(
        {
            "product": ["GC"] * total,
            "close": close,
            "research_partition": partitions,
        }
    )


class LogReturnTests(unittest.TestCase):
    def test_first_bar_of_each_product_is_null(self):
        frame = pd.DataFrame(
            {"product": ["GC", "GC", "MGC", "MGC"], "close": [100.0, 101.0, 50.0, 50.5]}
        )
        returns = log_returns_by_product(frame, GarchConfig())
        self.assertTrue(np.isnan(returns[0]))
        self.assertTrue(np.isnan(returns[2]), "product change must not produce a jump return")
        self.assertAlmostEqual(returns[1], np.log(101.0 / 100.0))
        self.assertAlmostEqual(returns[3], np.log(50.5 / 50.0))


class ForwardRecursionTests(unittest.TestCase):
    def setUp(self):
        self.parameters = GarchParameters(
            product="GC",
            omega=0.04,
            alpha=0.10,
            beta=0.85,
            mu=0.0,
            seed_variance=1.0,
            development_bars=1000,
            converged=True,
            optimiser_message="ok",
        )

    def test_matches_the_declared_recursion(self):
        returns = np.array([0.0001, -0.0002, 0.00015, 0.0])
        config = GarchConfig()
        produced = forward_conditional_volatility(returns, self.parameters, config=config)
        scaled = returns * config.return_scale
        expected = np.empty(4)
        expected[0] = 1.0
        for t in range(1, 4):
            expected[t] = 0.04 + 0.10 * scaled[t - 1] ** 2 + 0.85 * expected[t - 1]
        np.testing.assert_allclose(produced, np.sqrt(expected) / config.return_scale, rtol=1e-12)

    def test_persistence_and_stationarity_reporting(self):
        self.assertAlmostEqual(self.parameters.persistence, 0.95)
        self.assertTrue(self.parameters.is_stationary(0.999))
        boundary = GarchParameters(
            product="GC",
            omega=0.04,
            alpha=0.0961,
            beta=0.9039,
            mu=0.0,
            seed_variance=1.0,
            development_bars=10,
            converged=False,
            optimiser_message="code 8",
        )
        self.assertFalse(boundary.is_stationary(0.999))

    def test_missing_returns_do_not_poison_the_series(self):
        returns = np.array([0.0001, np.nan, 0.0002, 0.0003])
        produced = forward_conditional_volatility(returns, self.parameters)
        self.assertTrue(np.isnan(produced[1]), "the gap itself has no feature value")
        self.assertTrue(np.isfinite(produced[3]), "the recursion must recover after a gap")


class LeakageTests(unittest.TestCase):
    def test_future_bars_cannot_change_earlier_feature_values(self):
        """Perturbing final-test prices must not move any development value."""
        frame = _bars()
        baseline, _ = attach_garch_features(frame, CONFIG)

        perturbed = frame.copy()
        final_rows = perturbed["research_partition"].eq("final_test").to_numpy()
        perturbed.loc[final_rows, "close"] = perturbed.loc[final_rows, "close"] * 1.25
        moved, _ = attach_garch_features(perturbed, CONFIG)

        earlier = ~final_rows
        np.testing.assert_array_equal(
            baseline.loc[earlier, "feat_garch_cond_vol"].to_numpy(),
            moved.loc[earlier, "feat_garch_cond_vol"].to_numpy(),
        )
        np.testing.assert_array_equal(
            baseline.loc[earlier, "feat_garch_high_vol_regime"].to_numpy(),
            moved.loc[earlier, "feat_garch_high_vol_regime"].to_numpy(),
        )

    def test_threshold_is_fitted_on_development_bars_only(self):
        frame = _bars()
        baseline = build_garch_features(frame, CONFIG)
        perturbed = frame.copy()
        later = ~perturbed["research_partition"].eq("development").to_numpy()
        perturbed.loc[later, "close"] = perturbed.loc[later, "close"] * 1.4
        moved = build_garch_features(perturbed, CONFIG)
        self.assertEqual(baseline.threshold, moved.threshold)

    def test_development_regime_rate_matches_the_declared_quantile(self):
        result = build_garch_features(_bars(), CONFIG)
        self.assertAlmostEqual(
            result.diagnostics["regime_rate_development"],
            1.0 - CONFIG.high_vol_quantile,
            places=2,
        )


class FitAndDiagnosticsTests(unittest.TestCase):
    def test_fit_reports_convergence_and_seed_variance(self):
        frame = _bars()
        returns = log_returns_by_product(frame, CONFIG)
        development = frame["research_partition"].eq("development").to_numpy()
        fitted = fit_development_garch(returns, development, product="GC", config=CONFIG)
        self.assertEqual(fitted.product, "GC")
        self.assertGreater(fitted.seed_variance, 0.0)
        self.assertGreater(fitted.development_bars, 1000)
        self.assertIsInstance(fitted.converged, bool)
        self.assertGreater(fitted.alpha, 0.0)
        self.assertGreater(fitted.beta, 0.0)

    def test_rejects_insufficient_development_data(self):
        returns = np.linspace(0.0001, 0.0002, 50)
        mask = np.ones(50, dtype=bool)
        with self.assertRaises(ValueError):
            fit_development_garch(returns, mask, product="GC", config=CONFIG)

    def test_missing_columns_raise_clearly(self):
        frame = _bars().drop(columns=["research_partition"])
        with self.assertRaises(KeyError):
            build_garch_features(frame, CONFIG)

    def test_attach_does_not_mutate_the_input_frame(self):
        frame = _bars()
        original_columns = list(frame.columns)
        enriched, _ = attach_garch_features(frame, CONFIG)
        self.assertEqual(list(frame.columns), original_columns)
        self.assertIn("feat_garch_cond_vol", enriched.columns)
        self.assertIn("feat_garch_high_vol_regime", enriched.columns)
        self.assertEqual(enriched["feat_garch_high_vol_regime"].dtype, bool)

    def test_result_is_reproducible(self):
        frame = _bars()
        first = build_garch_features(frame, CONFIG)
        second = build_garch_features(frame, CONFIG)
        np.testing.assert_array_equal(first.conditional_volatility, second.conditional_volatility)
        self.assertEqual(first.threshold, second.threshold)


if __name__ == "__main__":
    unittest.main()
