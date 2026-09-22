"""Deterministic rules for serial prop-firm challenge simulations."""

from __future__ import annotations

import unittest

import pandas as pd

from src.app.analysis.challenge_simulator import (
    ChallengeRules,
    PhaseRules,
    RiskRules,
    simulate_challenges,
)


def _log(rs, dates=None, entries=None, exits=None) -> pd.DataFrame:
    count = len(rs)
    dates = dates or ["2024-01-02"] * count
    entries = entries or [index * 10 for index in range(count)]
    exits = exits or [position + 1 for position in entries]
    return pd.DataFrame(
        {
            "trade_date_ny": pd.to_datetime(dates),
            "gross_r": rs,
            "entry_position": entries,
            "exit_position": exits,
            "direction": [1] * count,
            "exit_reason": ["target"] * count,
            "holding_minutes": [1] * count,
            "strategy": ["Test strategy"] * count,
        }
    )


class ChallengeSimulatorTests(unittest.TestCase):
    def test_daily_target_rotates_serially_between_accounts(self):
        result = simulate_challenges(
            _log([1.0, 1.0]),
            ChallengeRules(
                account_count=2,
                phases=(PhaseRules(20.0, 20.0, 20.0),),
                payout_target_pct=3.0,
            ),
            RiskRules(
                risk_per_trade_pct=1.0,
                daily_profit_target_pct=1.0,
                daily_loss_limit_pct=0.0,
            ),
        )

        self.assertEqual(result.trades["account_id"].tolist(), [1, 2])
        self.assertEqual(
            result.trades["account_event"].tolist(),
            ["Daily profit target", "Daily profit target"],
        )

    def test_three_consecutive_firm_breaches_stop_the_day(self):
        result = simulate_challenges(
            _log(
                [-1.0] * 5,
                dates=["2024-01-02"] * 4 + ["2024-01-03"],
            ),
            ChallengeRules(
                account_count=5,
                phases=(PhaseRules(20.0, 0.5, 20.0),),
                payout_target_pct=3.0,
            ),
            RiskRules(
                risk_per_trade_pct=1.0,
                daily_profit_target_pct=0.0,
                daily_loss_limit_pct=0.0,
            ),
        )

        self.assertEqual(result.trades["account_id"].tolist(), [1, 2, 3, 4])
        self.assertEqual(result.accounts_breached, 4)
        self.assertEqual(result.three_breach_stop_days, 1)
        self.assertEqual(result.skipped_trades, 1)

    def test_two_step_account_advances_to_funded_then_payout(self):
        result = simulate_challenges(
            _log(
                [1.0, 1.0, 3.0],
                dates=["2024-01-02", "2024-01-03", "2024-01-04"],
            ),
            ChallengeRules(
                account_count=1,
                starting_balance=100_000.0,
                phases=(PhaseRules(1.0, 5.0, 10.0), PhaseRules(1.0, 5.0, 10.0)),
                payout_target_pct=3.0,
            ),
            RiskRules(
                risk_per_trade_pct=1.0,
                daily_profit_target_pct=0.0,
                daily_loss_limit_pct=0.0,
            ),
        )

        self.assertEqual(result.phase1_passes, 1)
        self.assertEqual(result.phase2_passes, 1)
        self.assertEqual(result.challenges_passed, 1)
        self.assertEqual(result.payouts_received, 1)
        self.assertEqual(result.accounts[0].status, "PAYOUT")
        self.assertEqual(
            result.trades["account_event"].tolist(),
            ["Advanced to Phase 2", "Challenge passed - funded", "Payout received"],
        )
        self.assertAlmostEqual(result.total_payout, 3_000.0)
        self.assertAlmostEqual(result.trades.iloc[0]["simulated_equity"], 101_000.0)

    def test_overall_loss_limit_breaches_account(self):
        result = simulate_challenges(
            _log([-1.1]),
            ChallengeRules(
                account_count=1,
                phases=(PhaseRules(20.0, 10.0, 1.0),),
                payout_target_pct=3.0,
            ),
            RiskRules(
                risk_per_trade_pct=1.0,
                daily_profit_target_pct=0.0,
                daily_loss_limit_pct=0.0,
            ),
        )

        self.assertEqual(result.accounts_breached, 1)
        self.assertEqual(result.accounts[0].breach_reason, "Firm overall loss limit")

    def test_overlapping_setup_is_skipped_while_a_trade_is_open(self):
        result = simulate_challenges(
            _log([0.2, 0.3], entries=[10, 15], exits=[20, 16]),
            ChallengeRules(
                account_count=2,
                phases=(PhaseRules(20.0, 10.0, 20.0),),
                payout_target_pct=3.0,
            ),
            RiskRules(
                risk_per_trade_pct=1.0,
                daily_profit_target_pct=0.0,
                daily_loss_limit_pct=0.0,
            ),
        )

        self.assertEqual(result.trades_taken, 1)
        self.assertEqual(result.skipped_trades, 1)

    def test_phase_consistency_delays_pass_until_profit_is_distributed(self):
        result = simulate_challenges(
            _log(
                [1.0, 1.0, 1.0],
                dates=["2024-01-02", "2024-01-03", "2024-01-04"],
            ),
            ChallengeRules(
                account_count=1,
                phases=(PhaseRules(1.0, 10.0, 20.0, consistency_pct=40.0),),
                payout_target_pct=3.0,
            ),
            RiskRules(
                risk_per_trade_pct=1.0,
                daily_profit_target_pct=0.0,
                daily_loss_limit_pct=0.0,
            ),
        )

        self.assertEqual(result.challenges_passed, 1)
        self.assertEqual(
            result.trades["account_event"].tolist(),
            [
                "Consistency 100.0% > 40.0%",
                "Consistency 50.2% > 40.0%",
                "Challenge passed - funded",
            ],
        )

    def test_funded_consistency_is_independent_and_off_by_default(self):
        challenge = ChallengeRules(
            account_count=1,
            phases=(PhaseRules(1.0, 10.0, 20.0),),
            payout_target_pct=3.0,
            funded_consistency_pct=40.0,
        )
        result = simulate_challenges(
            _log(
                [1.0, 3.0, 3.0, 3.0],
                dates=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            ),
            challenge,
            RiskRules(
                risk_per_trade_pct=1.0,
                daily_profit_target_pct=0.0,
                daily_loss_limit_pct=0.0,
            ),
        )

        self.assertIsNone(challenge.phases[0].consistency_pct)
        self.assertEqual(result.payouts_received, 1)
        self.assertEqual(
            result.trades["account_event"].tolist(),
            [
                "Challenge passed - funded",
                "Consistency 100.0% > 40.0%",
                "Consistency 50.7% > 40.0%",
                "Payout received",
            ],
        )


if __name__ == "__main__":
    unittest.main()
