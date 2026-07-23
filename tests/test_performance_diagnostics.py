from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from src.statistical_research.performance_diagnostics import (
    annualized_sharpe_ratio,
    build_backtest_daily_sharpe,
    build_feature_spread_sharpe,
    build_sharpe_applicability_table,
)


class AnnualizedSharpeTests(unittest.TestCase):
    def test_uses_sample_volatility_and_square_root_annualization(self) -> None:
        returns = np.linspace(-0.5, 1.5, 30)
        expected = np.sqrt(252.0) * returns.mean() / returns.std(ddof=1)
        self.assertAlmostEqual(annualized_sharpe_ratio(returns), expected)

    def test_rejects_short_or_zero_variance_series(self) -> None:
        self.assertTrue(np.isnan(annualized_sharpe_ratio(np.ones(30))))
        self.assertTrue(np.isnan(annualized_sharpe_ratio(np.arange(10.0))))


class FeatureSpreadSharpeTests(unittest.TestCase):
    def test_development_orientation_is_reused_in_validation(self) -> None:
        dates = pd.date_range("2022-01-03", periods=25, freq="B")
        validation_dates = pd.date_range("2024-01-02", periods=25, freq="B")
        records = []
        for partition, partition_dates in [
            ("Development", dates),
            ("Validation", validation_dates),
        ]:
            for date in partition_dates:
                for value in range(20):
                    records.append(
                        {
                            "trade_date_ny": date,
                            "entry_session": "London",
                            "research_partition": partition,
                            "feature_a": float(value),
                            "forward_return_5_ticks": float(value - 9.5),
                        }
                    )
        frame = pd.DataFrame.from_records(records)
        cells = pd.DataFrame(
            [
                {
                    "feature_name": "feature_a",
                    "outcome_family": "direction",
                    "horizon_minutes": 5,
                    "session": "London",
                    "research_partition": "Development",
                    "daily_ic_mean": 0.8,
                }
            ]
        )

        result = build_feature_spread_sharpe(
            frame,
            cells,
            ["feature_a"],
            horizons=[5],
            sessions=["London"],
            min_daily_bucket_observations=2,
        )

        self.assertEqual(set(result["development_orientation"]), {"high_minus_low"})
        self.assertTrue(result["mean_daily_spread_ticks"].gt(0).all())
        self.assertEqual(result.set_index("research_partition").loc["Validation", "trading_dates"], 25)

    def test_collapsed_discrete_buckets_return_nan_without_failing(self) -> None:
        dates = pd.date_range("2022-01-03", periods=25, freq="B")
        frame = pd.DataFrame(
            {
                "trade_date_ny": np.repeat(dates, 20),
                "entry_session": "London",
                "research_partition": "Development",
                "feature_flag": 0.0,
                "forward_return_5_ticks": np.tile(np.arange(20.0), len(dates)),
            }
        )
        cells = pd.DataFrame(
            [
                {
                    "feature_name": "feature_flag",
                    "outcome_family": "direction",
                    "horizon_minutes": 5,
                    "session": "London",
                    "research_partition": "Development",
                    "daily_ic_mean": 0.0,
                }
            ]
        )

        result = build_feature_spread_sharpe(
            frame,
            cells,
            ["feature_flag"],
            horizons=[5],
            sessions=["London"],
            partitions=["Development"],
            min_daily_bucket_observations=2,
        )

        self.assertEqual(result.loc[0, "bucket_count_observed"], 1)
        self.assertEqual(result.loc[0, "trading_dates"], 0)
        self.assertTrue(np.isnan(result.loc[0, "annualized_sharpe"]))


class BacktestSharpeTests(unittest.TestCase):
    def test_zero_trade_eligible_dates_are_retained(self) -> None:
        dates = pd.date_range("2022-01-03", periods=25, freq="B")
        candidates = pd.DataFrame(
            {
                "trade_date_ny": dates,
                "research_partition": "Development",
            }
        )
        trade_log = pd.DataFrame(
            {
                "direction_variant": "long_benchmark",
                "gate_variant": "expansion_gated",
                "research_partition": "Development",
                "trade_date_ny": dates[:20],
                "gross_r": np.tile([1.0, -0.5], 10),
                "stop_ticks": 10.0,
            }
        )
        config = SimpleNamespace(
            cost_scenarios=("frictionless", "base", "pessimistic"),
            commission_ticks_round_trip=0.6,
            base_slippage_ticks_per_side=1.0,
            pessimistic_slippage_ticks_per_side=2.0,
        )

        summary, daily = build_backtest_daily_sharpe(trade_log, candidates, config)
        frictionless = summary.loc[summary["cost_scenario"].eq("frictionless")].iloc[0]
        self.assertEqual(frictionless["calendar_days"], 25)
        self.assertEqual(frictionless["active_days"], 20)
        self.assertEqual(len(daily.loc[daily["cost_scenario"].eq("frictionless")]), 25)
        self.assertEqual(
            int(
                daily.loc[daily["cost_scenario"].eq("frictionless"), "daily_net_r"]
                .eq(0.0)
                .sum()
            ),
            5,
        )


class ApplicabilityTests(unittest.TestCase):
    def test_only_feature_screen_and_sequential_backtest_are_marked_applicable(self) -> None:
        table = build_sharpe_applicability_table()
        applicable = set(table.loc[table["sharpe_applicable"], "section"])
        self.assertEqual(applicable, {"7", "11"})


if __name__ == "__main__":
    unittest.main()
