"""Frozen sequential GC simulator for the Tsay Stage 4 policy gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

TICK_SIZE: Final = 0.10
COST_TICKS: Final = (0.0, 2.6, 4.6)


@dataclass(frozen=True)
class TsayTrade:
    """One immutable sequential-simulator trade record."""

    observation_id: int
    entry_timestamp_utc: pd.Timestamp
    exit_timestamp_utc: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    stop_distance_points: float
    exit_reason: str
    gross_r: float


def frozen_order_statistic(values: np.ndarray, quantile: float) -> float:
    """Return the one-based ceil(q*n) order statistic without interpolation."""

    finite = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if len(finite) == 0:
        raise ValueError("A frozen cutoff requires at least one finite score.")
    position = int(np.ceil(float(quantile) * len(finite)))
    return float(finite[max(1, position) - 1])


def simulate_tsay_policy(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Run next-open entries, stop-first exits, and one global position."""

    required_bars = {
        "timestamp_utc",
        "timestamp_ny",
        "trade_date_ny",
        "product",
        "active_symbol",
        "continuous_segment_id",
        "open",
        "high",
        "low",
        "close",
    }
    required_signals = {
        "observation_id",
        "decision_timestamp_utc",
        "direction",
        "decision_atr_20m",
        "eligible",
    }
    if missing := required_bars - set(bars):
        raise ValueError(f"Tsay simulator bar columns missing: {sorted(missing)}")
    if missing := required_signals - set(signals):
        raise ValueError(f"Tsay simulator signal columns missing: {sorted(missing)}")
    ordered = bars.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True).copy()
    ordered["timestamp_utc"] = pd.to_datetime(ordered["timestamp_utc"], utc=True, errors="raise")
    ordered["timestamp_ny"] = pd.to_datetime(ordered["timestamp_ny"], errors="raise")
    if set(ordered["product"].astype(str).unique()) != {"GC"}:
        raise ValueError("The Tsay simulator accepts GC bars only.")
    timestamp_to_position = {
        timestamp: position for position, timestamp in enumerate(ordered["timestamp_utc"])
    }
    signal_rows = signals.loc[signals["eligible"].astype(bool)].copy()
    signal_rows["decision_timestamp_utc"] = pd.to_datetime(
        signal_rows["decision_timestamp_utc"], utc=True, errors="raise"
    )
    signal_rows = signal_rows.sort_values(
        ["decision_timestamp_utc", "observation_id"], kind="mergesort"
    )
    trades: list[dict[str, object]] = []
    occupied_through = -1
    for signal in signal_rows.itertuples(index=False):
        decision_position = timestamp_to_position.get(signal.decision_timestamp_utc)
        if decision_position is None or decision_position < occupied_through:
            continue
        entry_position = decision_position + 1
        if entry_position >= len(ordered):
            continue
        decision_bar = ordered.iloc[decision_position]
        entry_bar = ordered.iloc[entry_position]
        if (
            str(entry_bar["trade_date_ny"]) != str(decision_bar["trade_date_ny"])
            or str(entry_bar["active_symbol"]) != str(decision_bar["active_symbol"])
            or str(entry_bar["continuous_segment_id"]) != str(decision_bar["continuous_segment_id"])
            or entry_bar["timestamp_utc"] - decision_bar["timestamp_utc"] != pd.Timedelta(minutes=1)
        ):
            continue
        direction = int(np.sign(float(signal.direction)))
        atr = float(signal.decision_atr_20m)
        if direction == 0 or not np.isfinite(atr) or atr <= 0.0:
            continue
        stop_distance = float(np.clip(1.5 * atr, 1.0, 10.0))
        entry_price = float(entry_bar["open"])
        stop_price = entry_price - direction * stop_distance
        target_price = entry_price + direction * 2.0 * stop_distance
        maximum_exit = min(entry_position + 119, len(ordered) - 1)
        exit_position = maximum_exit
        exit_price = float(ordered.iloc[maximum_exit]["close"])
        exit_reason = "MAX_HOLD_120"
        for position in range(entry_position, maximum_exit + 1):
            bar = ordered.iloc[position]
            boundary = (
                str(bar["trade_date_ny"]) != str(entry_bar["trade_date_ny"])
                or str(bar["active_symbol"]) != str(entry_bar["active_symbol"])
                or str(bar["continuous_segment_id"]) != str(entry_bar["continuous_segment_id"])
            )
            if boundary:
                previous = ordered.iloc[position - 1]
                exit_position = position - 1
                exit_price = float(previous["close"])
                exit_reason = "BOUNDARY"
                break
            high = float(bar["high"])
            low = float(bar["low"])
            stop_hit = low <= stop_price if direction > 0 else high >= stop_price
            target_hit = high >= target_price if direction > 0 else low <= target_price
            if stop_hit:
                exit_position = position
                exit_price = stop_price
                exit_reason = "STOP_FIRST" if target_hit else "STOP"
                break
            if target_hit:
                exit_position = position
                exit_price = target_price
                exit_reason = "TARGET"
                break
            if bar["timestamp_ny"].strftime("%H:%M") >= "15:30":
                exit_position = position
                exit_price = float(bar["close"])
                exit_reason = "FORCED_1530"
                break
        occupied_through = exit_position
        gross_r = direction * (exit_price - entry_price) / stop_distance
        trades.append(
            {
                "observation_id": int(signal.observation_id),
                "entry_timestamp_utc": entry_bar["timestamp_utc"],
                "exit_timestamp_utc": ordered.iloc[exit_position]["timestamp_utc"],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_distance_points": stop_distance,
                "exit_reason": exit_reason,
                "gross_r": float(gross_r),
                **{
                    f"net_r_{str(cost).replace('.', '_')}_ticks": float(
                        gross_r - cost * TICK_SIZE / stop_distance
                    )
                    for cost in COST_TICKS
                },
            }
        )
    return pd.DataFrame.from_records(trades)


def frozen_no_policy() -> dict[str, object]:
    """Return the mandatory state when no directional architecture passes."""

    return {
        "policy_status": "FROZEN_NO_POLICY",
        "real_bar_data_loaded": False,
        "development_economics_calculated": False,
        "validation_economics_may_open": False,
        "reason": "No fully passing Development directional architecture.",
    }
