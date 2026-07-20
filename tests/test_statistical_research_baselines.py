from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.baselines import (
    COST_THRESHOLDS_TICKS,
    OutcomeSpec,
    _bootstrap_mean_ci,
    build_cost_thresholds,
    summarize_outcomes,
)


def _frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
    sessions = pd.Categorical(
        ["London", "New York", "London", "New York"], categories=["London", "New York"]
    )
    frame = pd.DataFrame(
        {
            "trade_date_ny": dates,
            "entry_session": sessions,
            "label_available_5": [True, True, False, True],
            "forward_return_5_ticks": pd.array([1, -1, pd.NA, 3], dtype="Int32"),
            "mfe_long_5_ticks": pd.array([2, 1, pd.NA, 4], dtype="Int32"),
            "mae_long_5_ticks": pd.array([1, 2, pd.NA, 1], dtype="Int32"),
            "mfe_short_5_ticks": pd.array([1, 2, pd.NA, 1], dtype="Int32"),
            "mae_short_5_ticks": pd.array([2, 1, pd.NA, 4], dtype="Int32"),
            "future_range_5_ticks": pd.array([3, 3, pd.NA, 5], dtype="Int32"),
        }
    )
    return frame


class BaselineAggregationTests(unittest.TestCase):
    def test_unconditional_metrics_respect_availability(self):
        result = summarize_outcomes(
            _frame(),
            specs=(
                OutcomeSpec("forward_return_{h}_ticks", "forward_return", "forward_return_ticks"),
            ),
            analysis_type="test",
            horizons=(5,),
        ).iloc[0]
        self.assertEqual(result["observation_count"], 4)
        self.assertEqual(result["available_observation_count"], 3)
        self.assertAlmostEqual(result["availability_rate"], 0.75)
        self.assertAlmostEqual(result["mean"], 1.0)
        self.assertAlmostEqual(result["median"], 1.0)
        self.assertAlmostEqual(result["positive_rate"], 2 / 3)
        self.assertAlmostEqual(result["negative_rate"], 1 / 3)

    def test_group_counts_and_date_counts_are_explicit(self):
        result = summarize_outcomes(
            _frame(),
            specs=(
                OutcomeSpec("forward_return_{h}_ticks", "forward_return", "forward_return_ticks"),
            ),
            group_columns=("entry_session",),
            analysis_type="test",
            horizons=(5,),
        ).set_index("entry_session")
        self.assertEqual(result.loc["London", "observation_count"], 2)
        self.assertEqual(result.loc["London", "available_observation_count"], 1)
        self.assertEqual(result.loc["London", "trading_date_count"], 2)

    def test_cost_thresholds_separate_return_and_path_hurdles(self):
        result = build_cost_thresholds(_frame(), horizons=(5,))
        five = result.loc[
            result["horizon_minutes"].eq(5)
            & result["session"].eq("Combined")
            & result["cost_threshold_ticks"].eq(1)
        ].iloc[0]
        self.assertAlmostEqual(five["probability_return_above_positive_threshold"], 1 / 3)
        self.assertAlmostEqual(five["probability_return_below_negative_threshold"], 0.0)
        self.assertAlmostEqual(five["future_range_exceedance_rate"], 1.0)
        self.assertEqual(
            tuple(sorted(result["cost_threshold_ticks"].unique())), COST_THRESHOLDS_TICKS
        )

    def test_date_bootstrap_is_reproducible(self):
        values = np.array([1.0, 2.0, 4.0, 8.0])
        first = _bootstrap_mean_ci(values, seed=123, replicates=500)
        second = _bootstrap_mean_ci(values, seed=123, replicates=500)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
