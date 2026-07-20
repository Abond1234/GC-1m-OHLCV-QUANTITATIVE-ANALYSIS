"""Synthetic tests for the GC-to-MGC transfer validation engine (FR-09)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.execution.mgc_transfer import (
    MgcTransferConfig,
    build_minute_alignment,
    build_transfer_result,
)

CONFIG = MgcTransferConfig()


def _minutes(n: int, *, start: str = "2023-03-01 14:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="min")


def _gc_frame(ts: pd.DatetimeIndex, close: float = 2000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_event_utc": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 100,
            "session_label": "New York",
            "trade_date_ny": ts.normalize(),
        }
    )


def _mgc_frame(
    ts: pd.DatetimeIndex,
    close: float = 2000.0,
    *,
    volume: int = 40,
    half_range: float = 0.6,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_event_utc": ts,
            "open": close,
            "high": close + half_range,
            "low": close - half_range,
            "close": close,
            "volume": volume,
            "session_label": "New York",
            "trade_date_ny": ts.normalize(),
        }
    )


def _decisions(ts: pd.DatetimeIndex, entry_price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_retest_id": [f"R{i}" for i in range(len(ts))],
            "retest_ts_event_utc": ts,
            "research_partition": "development",
            "feat_first_contact_edge_price": entry_price,
        }
    )


class AlignmentTests(unittest.TestCase):
    def test_missing_mgc_minutes_reduce_coverage(self) -> None:
        ts = _minutes(100)
        aligned = build_minute_alignment(_gc_frame(ts), _mgc_frame(ts[:80]), CONFIG)
        self.assertEqual(len(aligned), 100)
        np.testing.assert_allclose(aligned["mgc_available"].mean(), 0.80)
        self.assertTrue(aligned.loc[~aligned["mgc_available"], "close_basis_ticks"].isna().all())

    def test_basis_measured_in_ticks(self) -> None:
        ts = _minutes(50)
        aligned = build_minute_alignment(_gc_frame(ts, 2000.0), _mgc_frame(ts, 2000.2), CONFIG)
        np.testing.assert_allclose(aligned["close_basis_ticks"], 2.0)


class TransferResultTests(unittest.TestCase):
    def test_clean_synthetic_market_passes_provisionally(self) -> None:
        ts = _minutes(400)
        aligned = build_minute_alignment(_gc_frame(ts), _mgc_frame(ts), CONFIG)
        result = build_transfer_result(aligned, _decisions(ts[:50], 2000.0), CONFIG)
        self.assertEqual(result.verdict, "G5_PROVISIONAL_PASS")
        self.assertTrue(result.threshold_checks["passed"].all())
        self.assertTrue(result.validation_checks["passed"].all())
        self.assertEqual(int(result.decision_summary["decision_bars"]), 50)
        np.testing.assert_allclose(result.decision_summary["entry_price_containment"], 1.0)

    def test_wide_basis_fails_the_basis_thresholds(self) -> None:
        ts = _minutes(400)
        aligned = build_minute_alignment(_gc_frame(ts, 2000.0), _mgc_frame(ts, 2000.5), CONFIG)
        result = build_transfer_result(aligned, _decisions(ts[:50], 2000.0), CONFIG)
        checks = result.threshold_checks.set_index("check")["passed"]
        self.assertFalse(checks["median_abs_decision_basis_ticks"])
        self.assertFalse(checks["p95_abs_decision_basis_ticks"])
        self.assertEqual(result.verdict, "G5_PROVISIONAL_FAIL")

    def test_uncontained_entry_prices_fail_containment(self) -> None:
        ts = _minutes(400)
        aligned = build_minute_alignment(_gc_frame(ts), _mgc_frame(ts), CONFIG)
        result = build_transfer_result(aligned, _decisions(ts[:50], 2005.0), CONFIG)
        checks = result.threshold_checks.set_index("check")["passed"]
        np.testing.assert_allclose(result.decision_summary["entry_price_containment"], 0.0)
        self.assertFalse(checks["entry_price_containment"])

    def test_zero_volume_decisions_fail_liquidity(self) -> None:
        ts = _minutes(400)
        aligned = build_minute_alignment(_gc_frame(ts), _mgc_frame(ts, volume=0), CONFIG)
        result = build_transfer_result(aligned, _decisions(ts[:50], 2000.0), CONFIG)
        checks = result.threshold_checks.set_index("check")["passed"]
        self.assertFalse(checks["decision_zero_volume_rate"])
        self.assertFalse(checks["median_decision_mgc_volume"])

    def test_unsynchronized_decisions_fail_decision_coverage(self) -> None:
        ts = _minutes(400)
        aligned = build_minute_alignment(_gc_frame(ts), _mgc_frame(ts[:200]), CONFIG)
        late = ts[190:240]
        result = build_transfer_result(aligned, _decisions(late, 2000.0), CONFIG)
        checks = result.threshold_checks.set_index("check")["passed"]
        np.testing.assert_allclose(result.decision_summary["decision_coverage"], 0.2)
        self.assertFalse(checks["decision_coverage"])


if __name__ == "__main__":
    unittest.main()
