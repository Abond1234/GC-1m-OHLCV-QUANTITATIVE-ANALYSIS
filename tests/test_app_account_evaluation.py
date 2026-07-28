"""Dollar account model and prop-style evaluation rules (Qt-free)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.app.analysis.account import (
    GC_SPEC,
    MGC_SPEC,
    AccountSettings,
    build_equity_curve,
    contracts_for,
    daily_pnl,
    session_stats,
    trade_dollars,
)
from src.app.analysis.evaluation import (
    PRESETS,
    EvaluationRules,
    evaluate,
)


def _trade(tid, entry, stop_points, gross_r):
    return SimpleNamespace(
        id=tid,
        entry_position=entry,
        stop_points=stop_points,
        result=SimpleNamespace(gross_r=gross_r),
    )


def _dates(n_days=5, bars_per_day=100):
    days = np.arange(np.datetime64("2024-06-03"), np.datetime64("2024-06-03") + n_days)
    return np.repeat(days, bars_per_day)


class SizingTests(unittest.TestCase):
    def test_risk_percent_sizing_floors_at_one_contract(self):
        s = AccountSettings(starting_balance=100_000, risk_percent=1.0)
        # risk $1000; GC stop of 4.0 points = $400/contract -> 2 contracts
        self.assertEqual(contracts_for(s, 4.0, 100_000), 2)
        # a huge stop still trades one contract
        self.assertEqual(contracts_for(s, 50.0, 100_000), 1)

    def test_fixed_contracts_mode(self):
        s = AccountSettings(sizing_mode="fixed_contracts", fixed_contracts=3)
        self.assertEqual(contracts_for(s, 2.0, 100_000), 3)

    def test_trade_dollars_gc_vs_mgc(self):
        gc = AccountSettings(instrument=GC_SPEC)
        mgc = AccountSettings(instrument=MGC_SPEC)
        # +2R on a 3-point stop, 1 contract: GC $600, MGC $60
        self.assertAlmostEqual(trade_dollars(gc, 2.0, 3.0, 1), 600.0)
        self.assertAlmostEqual(trade_dollars(mgc, 2.0, 3.0, 1), 60.0)


class EquityCurveTests(unittest.TestCase):
    def test_curve_orders_compounds_and_draws_down(self):
        dates = _dates()
        s = AccountSettings(sizing_mode="fixed_contracts", fixed_contracts=1)
        trades = [
            _trade(2, 150, 2.0, -1.0),  # day 1: -$200
            _trade(1, 10, 2.0, 2.0),  # day 0: +$400 (entered first)
            _trade(3, 320, 2.0, 1.0),  # day 3: +$200
        ]
        curve = build_equity_curve(trades, s, dates)
        self.assertEqual(list(curve.trade_ids), [1, 2, 3])
        self.assertEqual(list(curve.pnl), [400.0, -200.0, 200.0])
        self.assertEqual(list(curve.equity), [100_400.0, 100_200.0, 100_400.0])
        self.assertEqual(list(curve.drawdown), [0.0, -200.0, 0.0])

    def test_daily_pnl_groups_by_trade_date(self):
        dates = _dates()
        s = AccountSettings(sizing_mode="fixed_contracts")
        trades = [_trade(1, 5, 2.0, 1.0), _trade(2, 50, 2.0, -0.5), _trade(3, 150, 2.0, 1.0)]
        days, sums = daily_pnl(build_equity_curve(trades, s, dates))
        self.assertEqual(len(days), 2)
        self.assertAlmostEqual(sums[0], 100.0)  # +200 - 100
        self.assertAlmostEqual(sums[1], 200.0)

    def test_stats_summary(self):
        dates = _dates()
        s = AccountSettings(sizing_mode="fixed_contracts")
        trades = [_trade(1, 5, 2.0, 2.0), _trade(2, 150, 2.0, -1.0)]
        curve = build_equity_curve(trades, s, dates)
        stats = session_stats(curve, [2.0, -1.0], s)
        self.assertEqual(stats.n_trades, 2)
        self.assertEqual(stats.wins, 1)
        self.assertAlmostEqual(stats.win_rate, 0.5)
        self.assertAlmostEqual(stats.avg_r, 0.5)
        self.assertAlmostEqual(stats.total_pnl, 200.0)
        self.assertAlmostEqual(stats.profit_factor, 2.0)
        self.assertEqual(stats.days_traded, 2)

    def test_empty_session(self):
        s = AccountSettings()
        curve = build_equity_curve([], s, _dates())
        stats = session_stats(curve, [], s)
        self.assertEqual(stats.n_trades, 0)
        self.assertEqual(stats.end_equity, s.starting_balance)


class EvaluationTests(unittest.TestCase):
    def _curve(self, r_list, stop=10.0, per_day=1):
        dates = _dates(n_days=30, bars_per_day=100)
        s = AccountSettings(sizing_mode="fixed_contracts", fixed_contracts=1)
        trades = [_trade(i + 1, i * (100 // per_day), stop, r) for i, r in enumerate(r_list)]
        return build_equity_curve(trades, s, dates), s

    def test_daily_loss_breach_fails(self):
        # Two -3R trades on one day at $1,000 risk = -$6,000 > 5% of 100k
        curve, s = self._curve([-3.0, -3.0], per_day=2)
        status = evaluate(curve, EvaluationRules(), s.starting_balance)
        self.assertEqual(status.state, "FAILED")
        self.assertEqual(status.failed_rule, "daily_loss")

    def test_drawdown_floor_breach_fails(self):
        # Slow bleed across days: 11 days x -$1,000 stays under the daily limit
        # but crosses the 10% static floor.
        curve, s = self._curve([-1.0] * 11)
        status = evaluate(curve, EvaluationRules(), s.starting_balance)
        self.assertEqual(status.state, "FAILED")
        self.assertEqual(status.failed_rule, "max_drawdown")

    def test_pass_requires_target_and_days(self):
        # +2R/day at $1,000 risk: 4 days = $8,000 = target, but min days is 5.
        curve, s = self._curve([2.0] * 4)
        status = evaluate(curve, EvaluationRules(), s.starting_balance)
        self.assertEqual(status.state, "IN_PROGRESS")
        curve5, _ = self._curve([2.0] * 5)
        status5 = evaluate(curve5, EvaluationRules(), s.starting_balance)
        self.assertEqual(status5.state, "PASSED")

    def test_breach_beats_later_recovery(self):
        curve, s = self._curve([-3.0, -3.0, 6.0, 6.0], per_day=2)
        status = evaluate(curve, EvaluationRules(), s.starting_balance)
        self.assertEqual(status.state, "FAILED")

    def test_empty_session_is_in_progress(self):
        curve, s = self._curve([])
        status = evaluate(curve, EvaluationRules(), s.starting_balance)
        self.assertEqual(status.state, "IN_PROGRESS")
        self.assertEqual(len(status.rules), 4)

    def test_presets_registry(self):
        self.assertIn("Practice - no rules", PRESETS)
        self.assertIsNone(PRESETS["Practice - no rules"])
        self.assertEqual(PRESETS["Prop 100k"].daily_loss_limit_pct, 5.0)


if __name__ == "__main__":
    unittest.main()
