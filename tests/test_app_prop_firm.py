"""Prop-firm policy evaluation: each rule breach and the pass/payout path (no Qt)."""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

from src.app.analysis.account import EquityCurve
from src.app.analysis.evaluation import (
    RULE_DAILY_LOSS,
    RULE_MAX_CONTRACTS,
    RULE_MAX_DAYS,
    RULE_MAX_DRAWDOWN,
    PropFirmPolicy,
    evaluate_policy,
)

_SB = 100_000.0


def _curve(pnls, days, contracts=None):
    pnl = np.asarray(pnls, dtype=float)
    equity = _SB + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[_SB], equity]))[1:]
    n = len(pnl)
    return EquityCurve(
        entry_positions=np.arange(n),
        trade_ids=np.arange(n),
        trade_dates=np.array(days, dtype="datetime64[ns]"),
        contracts=np.asarray(contracts if contracts is not None else [1] * n),
        pnl=pnl,
        equity=equity,
        drawdown=equity - peak,
    )


def _days(*specs):
    out = []
    for date, count in specs:
        out += [f"2024-01-{date:02d}"] * count
    return out


class PropFirmTests(unittest.TestCase):
    def test_daily_loss_breach(self):
        curve = _curve([-6000], _days((1, 1)))
        rep = evaluate_policy(curve, PropFirmPolicy(daily_loss_limit_pct=5.0))
        self.assertEqual(rep.state, "FAILED")
        self.assertEqual(rep.failed_rule, RULE_DAILY_LOSS)
        self.assertEqual(rep.breach_date, "2024-01-01")

    def test_static_max_drawdown_breach(self):
        # spread losses across days so the daily-loss rule never trips first
        curve = _curve([-4000, -4000, -4000], _days((1, 1), (2, 1), (3, 1)))
        rep = evaluate_policy(
            curve, PropFirmPolicy(max_drawdown_pct=10.0, daily_loss_limit_pct=5.0)
        )
        self.assertEqual(rep.state, "FAILED")
        self.assertEqual(rep.failed_rule, RULE_MAX_DRAWDOWN)

    def test_trailing_drawdown_breach(self):
        # rise to a peak, then give back more than the 4% trailing cap but less
        # than the daily-loss limit, so the drawdown rule is the one that trips.
        curve = _curve([3000, 3000, 2000, -4500], _days((1, 1), (2, 1), (3, 1), (4, 1)))
        rep = evaluate_policy(curve, PropFirmPolicy(trailing_drawdown=True, max_drawdown_pct=4.0))
        self.assertEqual(rep.state, "FAILED")
        self.assertEqual(rep.failed_rule, RULE_MAX_DRAWDOWN)

    def test_max_contracts_breach(self):
        curve = _curve([100, 100], _days((1, 1), (2, 1)), contracts=[1, 9])
        rep = evaluate_policy(curve, PropFirmPolicy(max_contracts=5))
        self.assertEqual(rep.state, "FAILED")
        self.assertEqual(rep.failed_rule, RULE_MAX_CONTRACTS)

    def test_max_evaluation_days_breach(self):
        curve = _curve([100] * 4, _days((1, 1), (2, 1), (3, 1), (4, 1)))
        rep = evaluate_policy(curve, PropFirmPolicy(max_evaluation_days=3, profit_target_pct=99.0))
        self.assertEqual(rep.state, "FAILED")
        self.assertEqual(rep.failed_rule, RULE_MAX_DAYS)

    def test_pass_and_payout(self):
        curve = _curve([2000] * 5, _days((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)))
        rep = evaluate_policy(
            curve, PropFirmPolicy(profit_target_pct=8.0, min_trading_days=5, payout_split_pct=80.0)
        )
        self.assertEqual(rep.state, "PASSED")
        self.assertEqual(rep.days_to_pass, 5)
        self.assertEqual(rep.trades_to_pass, 5)
        self.assertTrue(rep.payout_eligible)
        self.assertAlmostEqual(rep.simulated_payout, 10_000.0 * 0.8)
        self.assertAlmostEqual(rep.post_payout_balance, 110_000.0 - 8_000.0)

    def test_consistency_blocks_the_pass(self):
        # target reached over 5 days, but one day is 80% of the profit
        curve = _curve([8000, 500, 500, 500, 500], _days((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)))
        rep = evaluate_policy(
            curve,
            PropFirmPolicy(
                profit_target_pct=8.0, min_trading_days=5, consistency_max_daily_share=0.4
            ),
        )
        self.assertEqual(rep.state, "IN_PROGRESS")  # not failed, but not passed
        self.assertFalse(rep.payout_eligible)

    def test_curves_have_one_point_per_trade(self):
        curve = _curve([2000] * 5, _days((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)))
        rep = evaluate_policy(curve, PropFirmPolicy())
        self.assertEqual(len(rep.equity), 5)
        self.assertEqual(len(rep.drawdown), 5)
        self.assertEqual(len(rep.utilization), 5)


_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class PropFirmPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.ui import theme

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        theme.apply(cls.app, "dark")

    def test_policy_inputs_round_trip(self):
        from src.app.ui.session_panel import SessionPanel

        panel = SessionPanel()
        custom = {
            "starting_balance": 50_000.0,
            "profit_target_pct": 6.0,
            "daily_loss_limit_pct": 3.0,
            "max_drawdown_pct": 4.0,
            "trailing_drawdown": True,
            "min_trading_days": 7,
            "max_evaluation_days": 30,
            "consistency_max_daily_share": 0.4,
            "max_contracts": 5,
            "payout_split_pct": 90.0,
        }
        panel.load_policy_dict(custom)
        got = panel.policy()
        self.assertEqual(got.starting_balance, 50_000.0)
        self.assertTrue(got.trailing_drawdown)
        self.assertEqual(got.max_contracts, 5)
        self.assertAlmostEqual(got.consistency_max_daily_share, 0.4)

    def test_update_session_shows_the_verdict(self):
        from src.app.analysis.account import SessionStats
        from src.app.ui.session_panel import SessionPanel

        panel = SessionPanel()
        curve = _curve([-6000], _days((1, 1)))
        report = evaluate_policy(curve, PropFirmPolicy())
        stats = SessionStats(1, 0, 0.0, -1.0, -6000.0, 0.0, -6000.0, 94_000.0, 6000.0, 6.0, 1)
        panel.update_session(curve, stats, report)
        self.assertIn("FAILED", panel.verdict.text())
        self.assertEqual(panel._outputs["payout_eligible"].text(), "No")


if __name__ == "__main__":
    unittest.main()
