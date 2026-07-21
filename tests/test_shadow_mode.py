"""Synthetic tests for the shadow-mode forward-test scaffolding."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.execution.shadow_mode import (
    ShadowDecision,
    ShadowModeLogger,
    config_fingerprint,
    reconcile_theoretical_fills,
    stale_data_violation,
    verify_decision_log,
)


def _decision(index: int, action: str = "enter_short") -> ShadowDecision:
    return ShadowDecision(
        decision_ts_utc=f"2026-07-20T14:{index:02d}:00Z",
        signal_id=f"S{index}",
        instrument="MGC",
        action=action,
        entry_price=2000.0 if action.startswith("enter") else None,
        stop_price=2001.0 if action.startswith("enter") else None,
        bar_ts_utc=f"2026-07-20T14:{index:02d}:00Z",
        bar_age_seconds=30.0,
        config_hash="abc123",
        code_version="test",
    )


class HashChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmp.name) / "decisions.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_chain_verifies_and_survives_logger_restart(self) -> None:
        logger = ShadowModeLogger(self.log_path)
        for i in range(3):
            logger.append(_decision(i))
        resumed = ShadowModeLogger(self.log_path)
        resumed.append(_decision(3, action="no_trade"))
        report = verify_decision_log(self.log_path)
        self.assertEqual(int(report["records"]), 4)
        self.assertTrue(bool(report["chain_intact"]))

    def test_tampered_field_breaks_chain(self) -> None:
        logger = ShadowModeLogger(self.log_path)
        for i in range(3):
            logger.append(_decision(i))
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[1])
        record["entry_price"] = 1999.0
        lines[1] = json.dumps(record, sort_keys=True)
        self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = verify_decision_log(self.log_path)
        self.assertFalse(bool(report["chain_intact"]))
        self.assertEqual(int(report["first_broken_index"]), 1)

    def test_deleted_record_breaks_chain(self) -> None:
        logger = ShadowModeLogger(self.log_path)
        for i in range(3):
            logger.append(_decision(i))
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        self.log_path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
        report = verify_decision_log(self.log_path)
        self.assertFalse(bool(report["chain_intact"]))

    def test_entry_decision_requires_prices(self) -> None:
        logger = ShadowModeLogger(self.log_path)
        bad = ShadowDecision(
            decision_ts_utc="2026-07-20T14:00:00Z",
            signal_id="S0",
            instrument="MGC",
            action="enter_short",
            entry_price=None,
            stop_price=None,
            bar_ts_utc="2026-07-20T14:00:00Z",
            bar_age_seconds=10.0,
            config_hash="abc",
            code_version="test",
        )
        with self.assertRaises(ValueError):
            logger.append(bad)


class GuardAndReconciliationTests(unittest.TestCase):
    def test_stale_data_boundary(self) -> None:
        self.assertFalse(stale_data_violation(119.0, max_age_seconds=120.0))
        self.assertTrue(stale_data_violation(121.0, max_age_seconds=120.0))
        self.assertTrue(stale_data_violation(float("nan")))

    def test_config_fingerprint_is_order_insensitive(self) -> None:
        first = config_fingerprint({"a": 1, "b": [1, 2]})
        second = config_fingerprint({"b": [1, 2], "a": 1})
        self.assertEqual(first, second)
        self.assertNotEqual(first, config_fingerprint({"a": 2, "b": [1, 2]}))

    def test_theoretical_fill_uses_next_bar_range(self) -> None:
        bars = pd.DataFrame(
            {
                "ts_event_utc": pd.date_range("2026-07-20 14:00", periods=3, freq="min"),
                "high": [2000.5, 2000.4, 2000.3],
                "low": [1999.5, 1999.8, 1999.9],
            }
        )
        decisions = pd.DataFrame(
            {
                "signal_id": ["fillable", "unfillable", "no_next_bar"],
                "action": ["enter_short", "enter_short", "enter_short"],
                "decision_ts_utc": [
                    "2026-07-20 14:00:00",
                    "2026-07-20 14:00:00",
                    "2026-07-20 14:02:00",
                ],
                "entry_price": [2000.2, 2001.5, 2000.0],
            }
        )
        report = reconcile_theoretical_fills(decisions, bars)
        by_id = report.set_index("signal_id")
        self.assertTrue(bool(by_id.loc["fillable", "theoretical_fill"]))
        self.assertFalse(bool(by_id.loc["unfillable", "theoretical_fill"]))
        self.assertFalse(bool(by_id.loc["no_next_bar", "next_bar_available"]))

    def test_non_entry_decisions_yield_empty_report(self) -> None:
        decisions = pd.DataFrame(
            {
                "signal_id": ["S0"],
                "action": ["no_trade"],
                "decision_ts_utc": ["2026-07-20 14:00:00"],
                "entry_price": [None],
            }
        )
        bars = pd.DataFrame(
            {
                "ts_event_utc": pd.date_range("2026-07-20 14:00", periods=2, freq="min"),
                "high": [2000.5, 2000.4],
                "low": [1999.5, 1999.8],
            }
        )
        self.assertTrue(reconcile_theoretical_fills(decisions, bars).empty)


if __name__ == "__main__":
    unittest.main()
