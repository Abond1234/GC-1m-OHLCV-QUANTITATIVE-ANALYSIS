"""Tests for the independent backtest verifier.

A verifier is only worth trusting if it fails on bad input, so these tests do
two things: confirm the clean-room re-simulation reproduces hand-computed trade
outcomes, and confirm each audit actually rejects a corrupted artifact (a
tampered trade log, a gap-through fill, a mismatched entry price, a Final-test
leak, and a drifted cost contract).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.backtest_verification import (
    VerificationConfig,
    _bar_arrays,
    audit_entry_fidelity,
    audit_fills,
    cross_check_forward_labels,
    independent_resimulate,
    map_entry_positions,
    reconcile_trades,
    run_verification,
)
from src.statistical_research.sequential_backtest import run_sequential_backtest

_DATE = pd.Timestamp("2021-06-01")
_BASE_TS = pd.Timestamp("2021-06-01 08:00:00", tz="UTC")


def _bars_from_ohlc(rows: list[tuple], *, minute_start: int = 240, segment=None, dates=None):
    """Build a GC bar frame from (open, high, low, close) rows."""

    n = len(rows)
    opens, highs, lows, closes = (np.array(x, dtype=float) for x in zip(*rows, strict=True))
    segment = np.ones(n, dtype=int) if segment is None else np.asarray(segment)
    dates = np.array([_DATE] * n) if dates is None else np.asarray(dates)
    return pd.DataFrame(
        {
            "ts_event_utc": [_BASE_TS + pd.Timedelta(minutes=i) for i in range(n)],
            "product": ["GC"] * n,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "trade_date_ny": dates,
            "minute_of_day_ny": np.arange(minute_start, minute_start + n, dtype=np.int64),
            "continuous_segment_id": segment,
            "rolling_atr_20m": np.full(n, 0.5),
        }
    )


def _one_candidate(
    position, bars, *, direction_long=True, stop=0.5, target=0.8, partition="Development"
):
    """A single-row candidate frame aligned to a bar position."""

    return pd.DataFrame(
        {
            "entry_timestamp_utc": [bars["ts_event_utc"].iloc[position]],
            "trade_date_ny": [bars["trade_date_ny"].iloc[position]],
            "entry_session": ["London"],
            "research_partition": [partition],
            "entry_price": [bars["open"].iloc[position]],
            "decision_atr_20m": [bars["rolling_atr_20m"].iloc[position - 1] if position else 0.5],
            "gate_prediction": [0.0],
            "expansion_gate_flag": [True],
            "stop_points": [stop],
            "target_points": [target],
        },
        index=pd.Index([position], name="observation_id"),
    )


# A deterministic price path with hand-computable outcomes.
_PATH = [
    (100.00, 100.20, 99.90, 100.10),  # 0
    (100.10, 100.30, 100.00, 100.20),  # 1  long entry -> target at bar 3
    (100.20, 100.40, 100.10, 100.30),  # 2
    (100.30, 101.00, 100.20, 100.90),  # 3  high 101.00 reaches target 100.90
    (100.90, 101.10, 100.80, 101.00),  # 4  long entry -> stop at bar 5
    (101.00, 101.10, 100.30, 100.40),  # 5  low 100.30 reaches stop 100.40
    (100.40, 100.50, 100.30, 100.40),  # 6  long entry -> time exit (flat)
    (100.40, 100.50, 100.30, 100.40),  # 7
    (100.40, 100.50, 100.30, 100.40),  # 8
    (100.40, 100.50, 100.30, 100.40),  # 9
]


class ResimulationOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.bars = _bars_from_ohlc(_PATH)
        self.arrays = _bar_arrays(self.bars)
        self.cfg = VerificationConfig()

    def _run_single(self, position, **kw):
        cand = _one_candidate(position, self.bars, **kw)
        return independent_resimulate(
            cand, np.array([position]), self.arrays, "long_benchmark", "ungated", self.cfg
        ).iloc[0]

    def test_long_target_exit_matches_hand_calc(self):
        trade = self._run_single(1, stop=0.5, target=0.8)
        self.assertEqual(trade["exit_position"], 3)
        self.assertEqual(trade["exit_reason"], "target")
        # gross R = (100.90 - 100.10) / 0.5 = 1.6
        self.assertAlmostEqual(trade["gross_r"], 1.6, places=9)

    def test_long_stop_exit_matches_hand_calc(self):
        trade = self._run_single(4, stop=0.5, target=2.0)
        self.assertEqual(trade["exit_position"], 5)
        self.assertEqual(trade["exit_reason"], "stop")
        self.assertAlmostEqual(trade["gross_r"], -1.0, places=9)

    def test_time_exit_when_neither_level_touched(self):
        trade = self._run_single(6, stop=5.0, target=10.0)
        self.assertEqual(trade["exit_reason"], "time_exit")
        self.assertEqual(trade["exit_position"], len(_PATH) - 1)
        self.assertAlmostEqual(trade["gross_r"], 0.0, places=9)

    def test_short_target_is_mirror_of_long(self):
        # short entry at bar 4 (open 100.90); price falls to 100.40 by bar 5, a
        # 0.5-point gain. With stop==target==0.5, one R of profit -> gross_r 1.0.
        cand = _one_candidate(4, self.bars, stop=0.5, target=0.5)
        trade = independent_resimulate(
            cand, np.array([4]), self.arrays, "short_benchmark", "ungated", self.cfg
        ).iloc[0]
        self.assertEqual(trade["exit_reason"], "target")
        # gross R = (100.90 - 100.40) / 0.5 = 1.0
        self.assertAlmostEqual(trade["gross_r"], 1.0, places=9)


class ForcedExitAndBoundaryTests(unittest.TestCase):
    def test_forced_exit_fires_at_1530_boundary(self):
        bars = _bars_from_ohlc(
            [(100.0, 100.1, 99.9, 100.0), (100.0, 100.1, 99.9, 100.05)],
            minute_start=929,  # bar 0 = 929, bar 1 = 930 (15:30)
        )
        cand = _one_candidate(0, bars, stop=5.0, target=5.0)
        trade = independent_resimulate(
            cand,
            np.array([0]),
            _bar_arrays(bars),
            "long_benchmark",
            "ungated",
            VerificationConfig(),
        ).iloc[0]
        self.assertEqual(trade["exit_reason"], "forced_1530")
        self.assertEqual(trade["exit_position"], 1)

    def test_boundary_exit_when_date_changes(self):
        bars = _bars_from_ohlc(
            [(100.0, 100.1, 99.9, 100.0), (100.0, 100.1, 99.9, 100.05)],
            dates=[_DATE, _DATE + pd.Timedelta(days=1)],
        )
        cand = _one_candidate(0, bars, stop=5.0, target=5.0)
        trade = independent_resimulate(
            cand,
            np.array([0]),
            _bar_arrays(bars),
            "long_benchmark",
            "ungated",
            VerificationConfig(),
        ).iloc[0]
        self.assertEqual(trade["exit_reason"], "boundary_exit")
        self.assertEqual(trade["exit_position"], 0)


class FillAuditTests(unittest.TestCase):
    def test_gap_through_stop_is_flagged(self):
        # bar 1 gaps far below the stop (99.5): the whole bar trades under it,
        # so filling at 99.5 is optimistic and must be counted as gap-through.
        bars = _bars_from_ohlc(
            [(100.0, 100.1, 99.9, 100.0), (99.0, 99.1, 98.8, 99.0)],
        )
        cand = _one_candidate(0, bars, stop=0.5, target=5.0)
        trades = independent_resimulate(
            cand,
            np.array([0]),
            _bar_arrays(bars),
            "long_benchmark",
            "ungated",
            VerificationConfig(),
        )
        audit = audit_fills(trades, _bar_arrays(bars), VerificationConfig())
        stop_row = audit[audit["exit_reason"] == "stop"].iloc[0]
        self.assertEqual(stop_row["gap_through_trades"], 1)
        self.assertLess(stop_row["within_bar_rate"], 1.0)
        self.assertTrue(stop_row["all_reachable"])  # still reachable, just gapped

    def test_within_bar_fill_is_not_flagged(self):
        bars = _bars_from_ohlc(_PATH)
        cand = _one_candidate(4, bars, stop=0.5, target=2.0)
        trades = independent_resimulate(
            cand,
            np.array([4]),
            _bar_arrays(bars),
            "long_benchmark",
            "ungated",
            VerificationConfig(),
        )
        audit = audit_fills(trades, _bar_arrays(bars), VerificationConfig())
        self.assertEqual(int(audit["gap_through_trades"].sum()), 0)


class EntryFidelityTests(unittest.TestCase):
    def test_entry_price_mismatch_is_detected(self):
        bars = _bars_from_ohlc(_PATH)
        cand = _one_candidate(1, bars)
        cand.loc[1, "entry_price"] = 999.0  # corrupt the recorded fill
        positions = map_entry_positions(cand, bars)
        audit = audit_entry_fidelity(cand, positions, _bar_arrays(bars), VerificationConfig())
        self.assertFalse(bool(audit["entry_open_match"]))

    def test_clean_entry_matches_bar_open(self):
        bars = _bars_from_ohlc(_PATH)
        cand = _one_candidate(1, bars)
        positions = map_entry_positions(cand, bars)
        audit = audit_entry_fidelity(cand, positions, _bar_arrays(bars), VerificationConfig())
        self.assertTrue(bool(audit["entry_open_match"]))
        self.assertTrue(bool(audit["entry_is_next_bar_after_decision"]))


class ReconciliationTests(unittest.TestCase):
    def _independent_all_variants(self, cand, bars):
        positions = map_entry_positions(cand, bars)
        arrays = _bar_arrays(bars)
        frames = [
            independent_resimulate(cand, positions, arrays, d, g, VerificationConfig())
            for d in ("long_benchmark", "short_benchmark")
            for g in ("ungated", "expansion_gated")
        ]
        return pd.concat(frames, ignore_index=True)

    def test_identical_logs_reconcile(self):
        bars = _bars_from_ohlc(_PATH)
        cand = _one_candidate(1, bars)
        independent = self._independent_all_variants(cand, bars)
        report, mism = reconcile_trades(independent, independent, VerificationConfig())
        self.assertTrue(bool(report["reconciled"].all()))
        self.assertTrue(mism.empty)

    def test_corrupted_log_fails_reconciliation(self):
        bars = _bars_from_ohlc(_PATH)
        cand = _one_candidate(1, bars)
        independent = self._independent_all_variants(cand, bars)
        corrupted = independent.copy()
        corrupted.loc[0, "exit_position"] = corrupted.loc[0, "exit_position"] + 1
        report, mism = reconcile_trades(corrupted, independent, VerificationConfig())
        self.assertFalse(bool(report["reconciled"].all()))
        self.assertGreater(int(report["field_mismatches"].sum()), 0)


class CrossArtifactTests(unittest.TestCase):
    def test_label_disagreement_is_detected(self):
        bars = _bars_from_ohlc(_PATH)
        cand = _one_candidate(1, bars)
        labels = pd.DataFrame(
            {
                "observation_id": [1],
                "entry_price": [cand.loc[1, "entry_price"] + 5.0],  # disagree
                "decision_atr_20m": [cand.loc[1, "decision_atr_20m"]],
            }
        )
        audit = cross_check_forward_labels(cand, labels, VerificationConfig())
        self.assertFalse(bool(audit["entry_price_agrees"]))


class FullRunTests(unittest.TestCase):
    """End-to-end: the verifier must agree with the production engine and reject leaks."""

    def _consistent_case(self):
        bars = _bars_from_ohlc(_PATH)
        cand = pd.concat(
            [
                _one_candidate(1, bars, stop=0.5, target=0.8),
                _one_candidate(4, bars, stop=0.5, target=2.0),
            ]
        )
        return bars, cand

    def test_verifier_agrees_with_production_engine(self):
        bars, cand = self._consistent_case()
        produced = run_sequential_backtest(cand, bars)
        result = run_verification(cand, bars, produced.trade_log, produced.performance)
        self.assertTrue(result.all_passed, msg=result.checks.to_string())
        self.assertEqual(int(result.summary["trade_field_mismatches"]), 0)
        self.assertEqual(int(result.summary["independent_trades"]), len(produced.trade_log))

    def test_final_test_leak_is_rejected(self):
        bars, cand = self._consistent_case()
        cand.iloc[0, cand.columns.get_loc("research_partition")] = "Final"
        produced_cand = cand.copy()
        # The production engine itself refuses Final rows, so build the recorded
        # log on the clean partitions and verify the leak trips the audit.
        clean = cand.copy()
        clean["research_partition"] = "Development"
        produced = run_sequential_backtest(clean, bars)
        result = run_verification(produced_cand, bars, produced.trade_log, produced.performance)
        self.assertFalse(result.all_passed)
        governance = result.checks.set_index("check").loc[
            "partitions_limited_to_development_validation", "passed"
        ]
        self.assertFalse(bool(governance))

    def test_cost_contract_drift_raises(self):
        bars, cand = self._consistent_case()
        produced = run_sequential_backtest(cand, bars)
        drifted = VerificationConfig(commission_ticks_round_trip=99.0)
        with self.assertRaises(ValueError):
            run_verification(cand, bars, produced.trade_log, produced.performance, config=drifted)


if __name__ == "__main__":
    unittest.main()
