"""Shadow-mode forward-test scaffolding (PRD Phase 5, stage 1).

The PRD's first forward-test stage runs signals and risk checks against live
data without transmitting orders, with exit criteria of no stale-data
violations, deterministic decisions, complete telemetry, and reconciled
theoretical fills.  This module is that harness: a tamper-evident append-only
decision log, a stale-data guard, and a theoretical-fill reconciler.  It has
no network dependencies - a future Rithmic (FR-11) feed plugs into it, and
until credentials and platform approval exist, historical bars can drive dry
runs of the identical code path.

Telemetry integrity: every record carries the SHA-256 of its predecessor, so
any edit, deletion, or reordering of the log breaks the chain and is detected
by :func:`verify_decision_log`.  Records are JSON lines with sorted keys;
hashing is over the canonical serialization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class ShadowDecision:
    """One logged decision at one bar close - the unit of telemetry."""

    decision_ts_utc: str
    signal_id: str
    instrument: str
    action: str
    entry_price: float | None
    stop_price: float | None
    bar_ts_utc: str
    bar_age_seconds: float
    config_hash: str
    code_version: str
    note: str = ""

    def validate(self) -> None:
        if self.action not in ("enter_short", "enter_long", "no_trade", "risk_block"):
            raise ValueError(f"unknown action: {self.action}")
        if self.action.startswith("enter") and (
            self.entry_price is None or self.stop_price is None
        ):
            raise ValueError("entry decisions need entry and stop prices")


def config_fingerprint(payload: dict) -> str:
    """Deterministic hash of a configuration mapping for telemetry stamping."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def stale_data_violation(bar_age_seconds: float, *, max_age_seconds: float = 120.0) -> bool:
    """True when the decision bar is older than the declared freshness bound."""

    return not np.isfinite(bar_age_seconds) or bar_age_seconds > max_age_seconds


class ShadowModeLogger:
    """Append-only JSONL decision log with a SHA-256 hash chain."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self._previous_hash = self._recover_tail_hash()

    def _recover_tail_hash(self) -> str:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return GENESIS_HASH
        last_line = self.log_path.read_text(encoding="utf-8").rstrip("\n").rsplit("\n", 1)[-1]
        return json.loads(last_line)["record_hash"]

    def append(self, decision: ShadowDecision) -> str:
        decision.validate()
        body = {"previous_hash": self._previous_hash, **asdict(decision)}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        record_hash = hashlib.sha256(canonical.encode()).hexdigest()
        body["record_hash"] = record_hash
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(body, sort_keys=True) + "\n")
        self._previous_hash = record_hash
        return record_hash


def verify_decision_log(log_path: Path) -> pd.Series:
    """Re-derive the whole hash chain; any tampering breaks verification."""

    lines = [
        line for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    previous = GENESIS_HASH
    intact = True
    broken_at = -1
    for index, line in enumerate(lines):
        record = json.loads(line)
        claimed = record.pop("record_hash", None)
        if record.get("previous_hash") != previous:
            intact, broken_at = False, index
            break
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        derived = hashlib.sha256(canonical.encode()).hexdigest()
        if derived != claimed:
            intact, broken_at = False, index
            break
        previous = claimed
    return pd.Series(
        {"records": len(lines), "chain_intact": intact, "first_broken_index": broken_at},
        name="value",
    )


def load_decision_log(log_path: Path) -> pd.DataFrame:
    """Decision log as a DataFrame (verification is the caller's step)."""

    rows = [
        json.loads(line)
        for line in Path(log_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame.from_records(rows)


def reconcile_theoretical_fills(
    decisions: pd.DataFrame, bars: pd.DataFrame, *, tick_size: float = 0.10
) -> pd.DataFrame:
    """Would each logged entry have filled on its decision bar's successor?

    Mirrors the research entry-realism rule: a boundary-touch limit is
    theoretically filled when the entry price trades within the next bar's
    range.  ``bars`` needs ``ts_event_utc``, ``high``, ``low`` sorted by time.
    Output pairs every entry decision with its theoretical fill flag so the
    PRD's reconciled-theoretical-fills exit criterion is checkable.
    """

    entry_rows = decisions.loc[decisions["action"].str.startswith("enter")].copy()
    if entry_rows.empty:
        return pd.DataFrame(
            columns=["signal_id", "decision_ts_utc", "entry_price", "theoretical_fill"]
        )
    ordered = bars.sort_values("ts_event_utc").reset_index(drop=True)
    bar_ts = pd.to_datetime(ordered["ts_event_utc"]).to_numpy()
    decision_ts = pd.to_datetime(entry_rows["decision_ts_utc"]).to_numpy()
    next_index = np.searchsorted(bar_ts, decision_ts, side="right")
    valid = next_index < len(ordered)
    high = np.full(len(entry_rows), np.nan)
    low = np.full(len(entry_rows), np.nan)
    high[valid] = ordered["high"].to_numpy(dtype=np.float64)[next_index[valid]]
    low[valid] = ordered["low"].to_numpy(dtype=np.float64)[next_index[valid]]
    entry = pd.to_numeric(entry_rows["entry_price"], errors="coerce").to_numpy(dtype=np.float64)
    filled = valid & (entry >= low - tick_size / 2) & (entry <= high + tick_size / 2)
    return pd.DataFrame(
        {
            "signal_id": entry_rows["signal_id"].to_numpy(),
            "decision_ts_utc": entry_rows["decision_ts_utc"].to_numpy(),
            "entry_price": entry,
            "next_bar_available": valid,
            "theoretical_fill": filled,
        }
    )
