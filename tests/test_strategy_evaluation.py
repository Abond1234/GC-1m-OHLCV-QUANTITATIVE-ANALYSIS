"""Tests for the anti-overfitting evaluation.

The corrections must themselves be correct, so these tests pin the statistics to
known behaviour: the deflated Sharpe rises for a genuine signal and falls as the
trial count grows; the CSCV overfitting probability sits near one half on pure
noise and collapses when one strategy carries persistent edge; the bootstrap is
deterministic by seed; and the one-sided test points the right way.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.strategy_evaluation import (
    EvaluationConfig,
    bootstrap_mean_ci,
    daily_sharpe,
    deflated_sharpe_ratio,
    evaluate_strategies,
    one_sided_t_pvalue,
    probability_of_backtest_overfitting,
)


class DeflatedSharpeTests(unittest.TestCase):
    def test_positive_signal_beats_null(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.25, 1.0, size=500)
        out = deflated_sharpe_ratio(returns, n_trials=2, trial_sharpe_variance=0.02)
        self.assertGreater(out["deflated_sharpe"], 0.9)

    def test_negative_signal_fails_null(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(-0.15, 1.0, size=500)
        out = deflated_sharpe_ratio(returns, n_trials=12, trial_sharpe_variance=0.02)
        self.assertLess(out["deflated_sharpe"], 0.05)

    def test_more_trials_deflate_the_sharpe(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.08, 1.0, size=400)
        few = deflated_sharpe_ratio(returns, n_trials=2, trial_sharpe_variance=0.03)
        many = deflated_sharpe_ratio(returns, n_trials=100, trial_sharpe_variance=0.03)
        self.assertGreaterEqual(few["deflated_sharpe"], many["deflated_sharpe"])


class PBOTests(unittest.TestCase):
    def test_pure_noise_averages_near_one_half(self):
        # A single fixed noise draw has a chance full-sample winner that
        # persists across splits, so PBO must be averaged over draws to expose
        # its E[PBO] = 0.5 behaviour under the no-skill null.
        pbos = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            matrix = pd.DataFrame(rng.normal(0.0, 1.0, size=(300, 12)))
            out = probability_of_backtest_overfitting(matrix, n_blocks=10)
            self.assertEqual(out["combinations"], 252)
            pbos.append(out["pbo"])
        self.assertGreater(float(np.mean(pbos)), 0.4)
        self.assertLess(float(np.mean(pbos)), 0.6)

    def test_persistent_winner_lowers_pbo(self):
        rng = np.random.default_rng(5)
        data = rng.normal(0.0, 1.0, size=(400, 12))
        data[:, 0] += 0.5  # one strategy carries persistent edge
        noisy = probability_of_backtest_overfitting(pd.DataFrame(data), n_blocks=10)
        self.assertLess(noisy["pbo"], 0.2)
        self.assertGreater(noisy["prob_oos_best_positive"], 0.8)


class BootstrapAndTestTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_by_seed(self):
        cfg = EvaluationConfig()
        rng = np.random.default_rng(6)
        daily = rng.normal(0.0, 1.0, size=300)
        first = bootstrap_mean_ci(daily, seed=42, cfg=cfg)
        second = bootstrap_mean_ci(daily, seed=42, cfg=cfg)
        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])

    def test_one_sided_pvalue_direction(self):
        rng = np.random.default_rng(7)
        positive = rng.normal(0.5, 1.0, size=400)
        negative = rng.normal(-0.5, 1.0, size=400)
        self.assertLess(one_sided_t_pvalue(positive), 0.01)
        self.assertGreater(one_sided_t_pvalue(negative), 0.99)

    def test_daily_sharpe_sign(self):
        self.assertGreater(daily_sharpe(np.array([0.5, 0.4, 0.6, 0.55, 0.45])), 0)


class EvaluateStrategiesTests(unittest.TestCase):
    def _synthetic_log(self, partitions=("Development", "Validation")):
        rng = np.random.default_rng(8)
        rows = []
        dates = pd.date_range("2021-01-04", periods=60, freq="B")
        for strat, family, edge in [
            ("winner", "trend", 0.2),
            ("loser", "reversion", -0.3),
            ("always_long", "benchmark", -0.1),
        ]:
            for partition in partitions:
                for d in dates:
                    for _ in range(3):
                        rows.append(
                            {
                                "strategy": strat,
                                "family": family,
                                "research_partition": partition,
                                "trade_date_ny": d,
                                "direction": 1,
                                "stop_ticks": 10.0,
                                "gross_r": float(rng.normal(edge, 1.0)),
                            }
                        )
        return pd.DataFrame(rows)

    def test_evaluation_runs_and_flags_no_final(self):
        result = evaluate_strategies(self._synthetic_log())
        self.assertEqual(set(result.strategy_stats["strategy"]), {"winner", "loser", "always_long"})
        self.assertIn("verdict", result.verdicts.columns)
        # benchmark is labelled, not gated
        bench = result.verdicts.set_index("strategy").loc["always_long", "verdict"]
        self.assertEqual(bench, "BENCHMARK")

    def test_final_test_partition_is_rejected(self):
        log = self._synthetic_log(partitions=("Development", "Final"))
        with self.assertRaises(ValueError):
            evaluate_strategies(log)


if __name__ == "__main__":
    unittest.main()
