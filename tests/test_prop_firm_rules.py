"""Synthetic tests for the versioned prop-firm rules engine (FR-10)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.execution.prop_firm_rules import (
    OUTCOME_FAILED_DAILY_LOSS,
    OUTCOME_FAILED_MAX_DRAWDOWN,
    OUTCOME_INCOMPLETE_MAX_DAYS,
    OUTCOME_PASS_BLOCKED_CONSISTENCY,
    OUTCOME_PASSED,
    PropFirmPolicy,
    estimate_evaluation_statistics,
    example_policies,
    simulate_evaluation,
)


def _policy(**overrides) -> PropFirmPolicy:
    base = dict(
        policy_id="test",
        version="1.0",
        effective_date="2026-07-20",
        account_size_usd=50_000.0,
        profit_target_usd=3_000.0,
        daily_loss_limit_usd=1_000.0,
        max_drawdown_usd=2_000.0,
        drawdown_mode="static",
        trailing_caps_at_initial_balance=False,
        max_contracts=5,
        minimum_trading_days=3,
    )
    base.update(overrides)
    policy = PropFirmPolicy(**base)
    policy.validate()
    return policy


def _trades(day_pnls: list[list[float]]) -> pd.DataFrame:
    rows = [
        {"trade_date": day, "pnl_usd": pnl} for day, pnls in enumerate(day_pnls) for pnl in pnls
    ]
    return pd.DataFrame.from_records(rows)


class SimulatorTests(unittest.TestCase):
    def test_daily_loss_breach_at_exact_limit(self) -> None:
        result = simulate_evaluation(_trades([[500.0], [-600.0, -400.0]]), _policy())
        self.assertEqual(result.outcome, OUTCOME_FAILED_DAILY_LOSS)
        self.assertEqual(result.trading_days, 2)

    def test_static_drawdown_survives_where_trailing_breaches(self) -> None:
        # +1,900 then -3,700: equity 48,200 stays above the static floor of
        # 48,000 but falls below the intraday trailing floor of 49,900.
        path = _trades([[1_900.0], [-950.0, -950.0, -900.0, -900.0]])
        lenient_daily = dict(daily_loss_limit_usd=5_000.0)
        static = simulate_evaluation(path, _policy(**lenient_daily))
        trailing = simulate_evaluation(
            path, _policy(drawdown_mode="trailing_intraday", **lenient_daily)
        )
        self.assertNotEqual(static.outcome, OUTCOME_FAILED_MAX_DRAWDOWN)
        self.assertEqual(trailing.outcome, OUTCOME_FAILED_MAX_DRAWDOWN)

    def test_trailing_eod_ignores_intraday_peak(self) -> None:
        # Day 1 peaks +2,500 intraday but closes at +1,900: the intraday
        # variant anchors its floor at the peak (50,500) and breaches on day
        # 2's slide to 50,000; the EOD variant anchors at the close (floor
        # 49,900) and survives (lenient daily limit).
        path = _trades([[2_500.0, -600.0], [-950.0, -950.0]])
        lenient_daily = dict(daily_loss_limit_usd=5_000.0)
        eod = simulate_evaluation(path, _policy(drawdown_mode="trailing_eod", **lenient_daily))
        intraday = simulate_evaluation(
            path, _policy(drawdown_mode="trailing_intraday", **lenient_daily)
        )
        self.assertNotEqual(eod.outcome, OUTCOME_FAILED_MAX_DRAWDOWN)
        self.assertEqual(intraday.outcome, OUTCOME_FAILED_MAX_DRAWDOWN)

    def test_trailing_floor_caps_at_initial_balance(self) -> None:
        # After +10,000 the uncapped trailing floor would be 58,000; the capped
        # variant freezes at the 50,000 initial balance, so a -9,500 slide
        # breaches only the uncapped policy.
        path = _trades([[10_000.0], [-4_750.0], [-4_750.0]])
        kwargs = dict(drawdown_mode="trailing_eod", daily_loss_limit_usd=50_000.0)
        capped = simulate_evaluation(path, _policy(trailing_caps_at_initial_balance=True, **kwargs))
        uncapped = simulate_evaluation(
            path, _policy(trailing_caps_at_initial_balance=False, **kwargs)
        )
        self.assertEqual(uncapped.outcome, OUTCOME_FAILED_MAX_DRAWDOWN)
        self.assertNotEqual(capped.outcome, OUTCOME_FAILED_MAX_DRAWDOWN)

    def test_target_needs_minimum_days(self) -> None:
        result = simulate_evaluation(
            _trades([[3_100.0], [50.0], [50.0]]), _policy(minimum_trading_days=3)
        )
        self.assertEqual(result.outcome, OUTCOME_PASSED)
        self.assertEqual(result.trading_days, 3)

    def test_consistency_rule_blocks_pass(self) -> None:
        result = simulate_evaluation(
            _trades([[2_800.0], [150.0], [150.0]]),
            _policy(consistency_max_daily_share=0.5),
        )
        self.assertEqual(result.outcome, OUTCOME_PASS_BLOCKED_CONSISTENCY)

    def test_max_days_limit(self) -> None:
        result = simulate_evaluation(_trades([[10.0]] * 5), _policy(maximum_evaluation_days=5))
        self.assertEqual(result.outcome, OUTCOME_INCOMPLETE_MAX_DAYS)

    def test_internal_buffers_count_without_breach(self) -> None:
        # -850 hits the 80 percent internal daily buffer (-800) without
        # touching the -1,000 firm limit; the account then recovers.
        result = simulate_evaluation(
            _trades([[-850.0], [900.0], [3_100.0]]), _policy(minimum_trading_days=1)
        )
        self.assertEqual(result.outcome, OUTCOME_PASSED)
        self.assertEqual(result.internal_daily_buffer_hits, 1)

    def test_ledger_reconciles_equity(self) -> None:
        result = simulate_evaluation(
            _trades([[250.0, -100.0], [500.0]]), _policy(profit_target_usd=10_000.0)
        )
        ledger = result.daily_ledger
        np.testing.assert_allclose(
            ledger["equity_usd"].to_numpy(),
            50_000.0 + ledger["day_pnl_usd"].cumsum().to_numpy(),
        )


class PolicyAndEstimatorTests(unittest.TestCase):
    def test_policy_round_trip_and_validation(self) -> None:
        for policy in example_policies():
            self.assertEqual(PropFirmPolicy.from_dict(policy.to_dict()), policy)
        with self.assertRaises(ValueError):
            _policy(drawdown_mode="unknown")
        with self.assertRaises(ValueError):
            _policy(daily_loss_limit_usd=-5.0)

    def test_estimator_bounds_and_determinism(self) -> None:
        policy = _policy(minimum_trading_days=1)
        winners = [1_000.0, 1_100.0, 900.0]
        losers = [-1_200.0, -1_100.0]
        sure_pass = estimate_evaluation_statistics(
            winners, policy, replicates=50, horizon_days=30, seed=11
        )
        sure_fail = estimate_evaluation_statistics(
            losers, policy, replicates=50, horizon_days=30, seed=11
        )
        self.assertEqual(sure_pass["pass_probability"], 1.0)
        self.assertEqual(sure_fail["pass_probability"], 0.0)
        self.assertGreater(sure_fail["daily_loss_breach_probability"], 0.9)
        mixed = [800.0, -700.0, 400.0, -300.0, 1_200.0, -900.0]
        first = estimate_evaluation_statistics(
            mixed, policy, replicates=80, horizon_days=40, seed=7
        )
        second = estimate_evaluation_statistics(
            mixed, policy, replicates=80, horizon_days=40, seed=7
        )
        pd.testing.assert_series_equal(first, second)


if __name__ == "__main__":
    unittest.main()
