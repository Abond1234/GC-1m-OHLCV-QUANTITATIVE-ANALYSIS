"""Flexible-exit trade simulator.

The per-bar exit ordering is identical to the verified research engine
(`strategy_lab._single_exit`): boundary -> stop (conservative stop-first) ->
target -> forced-1530 -> time exit; entry fills at the next-bar open. The only
addition is a path-dependent update applied *after* the exit checks and taking
effect on the *next* bar (a trailing stop that ratchets on completed-bar extremes,
and a breakeven move). When trailing and breakeven are both disabled that update
is a strict no-op, so the run reproduces `simulate_positions` to the tick - the
tie-back that keeps the app honest.

`gross_r` is always normalised by the *initial* stop distance (risk), matching the
research convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exit_config import ExitConfig


@dataclass
class FlexTradeResult:
    """A single simulated trade, with per-bar levels for drawing."""

    entry_position: int
    entry_price: float
    exit_position: int
    exit_price: float
    exit_reason: str
    direction: int
    ambiguous_bar: bool
    holding_minutes: int
    initial_stop_points: float
    stop_ticks: float
    gross_r: float
    mfe_r: float
    mae_r: float
    stop_track: np.ndarray  # per-held-bar stop level (for the chart)
    target_track: np.ndarray


def _walk_one(
    start: int,
    is_long: bool,
    stop_pts: float,
    target_pts: float,
    trail_pts: float,
    exit_cfg: ExitConfig,
    arrays: dict,
    record: bool,
):
    """Core single-trade walk. Returns the exit tuple (and optional tracks)."""

    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    open_price = arrays["open"]
    segment = arrays["segment"]
    trade_date = arrays["trade_date"]
    minute_ny = arrays["minute_ny"]
    n_bars = len(high)

    price_in = open_price[start]
    stop_cur = price_in - stop_pts if is_long else price_in + stop_pts
    target = price_in + target_pts if is_long else price_in - target_pts
    start_segment = segment[start]
    start_date = trade_date[start]
    max_position = min(start + exit_cfg.max_holding_minutes - 1, n_bars - 1)

    exit_position = start
    exit_reason = "end_of_data"
    exit_price = close[start]
    ambiguous = False
    breakeven_armed = False
    best_favorable = price_in  # highest high (long) / lowest low (short) so far
    stop_track: list[float] = []
    target_track: list[float] = []

    for position in range(start, max_position + 1):
        if exit_cfg.respect_boundary and (
            segment[position] != start_segment or trade_date[position] != start_date
        ):
            exit_position = position - 1
            exit_reason = "boundary_exit"
            exit_price = close[position - 1]
            break

        hit_stop = low[position] <= stop_cur if is_long else high[position] >= stop_cur
        hit_target = high[position] >= target if is_long else low[position] <= target

        if exit_cfg.stop_first:
            if hit_stop:
                exit_position, exit_reason, exit_price = position, "stop", stop_cur
                ambiguous = bool(hit_target)
                break
            if hit_target:
                exit_position, exit_reason, exit_price = position, "target", target
                break
        else:
            if hit_target:
                exit_position, exit_reason, exit_price = position, "target", target
                break
            if hit_stop:
                exit_position, exit_reason, exit_price = position, "stop", stop_cur
                ambiguous = bool(hit_target)
                break

        if (
            exit_cfg.forced_exit_minute_ny is not None
            and minute_ny[position] >= exit_cfg.forced_exit_minute_ny
        ):
            exit_position, exit_reason, exit_price = position, "forced_1530", close[position]
            break
        if position == max_position:
            exit_position, exit_reason, exit_price = position, "time_exit", close[position]
            break

        # ---- path-dependent update for the NEXT bar (no-op when static) ----
        if exit_cfg.is_dynamic:
            extreme = high[position] if is_long else low[position]
            best_favorable = (
                max(best_favorable, extreme) if is_long else min(best_favorable, extreme)
            )
            favorable_pts = best_favorable - price_in if is_long else price_in - best_favorable
            if (
                exit_cfg.breakeven_enabled
                and not breakeven_armed
                and favorable_pts >= exit_cfg.breakeven_trigger_r * stop_pts
            ):
                offset = exit_cfg.breakeven_offset_ticks * exit_cfg.tick_size
                be_level = price_in + offset if is_long else price_in - offset
                stop_cur = max(stop_cur, be_level) if is_long else min(stop_cur, be_level)
                breakeven_armed = True
            if exit_cfg.trailing_enabled and trail_pts > 0:
                trail_level = best_favorable - trail_pts if is_long else best_favorable + trail_pts
                stop_cur = max(stop_cur, trail_level) if is_long else min(stop_cur, trail_level)
        if record:
            stop_track.append(stop_cur)
            target_track.append(target)

    points = exit_price - price_in if is_long else price_in - exit_price
    gross_r = float(points / stop_pts)

    # Within-trade excursions over the held bars (for forensics).
    lo = start
    hi = exit_position
    if is_long:
        mfe_r = float((np.max(high[lo : hi + 1]) - price_in) / stop_pts)
        mae_r = float((price_in - np.min(low[lo : hi + 1])) / stop_pts)
    else:
        mfe_r = float((price_in - np.min(low[lo : hi + 1])) / stop_pts)
        mae_r = float((np.max(high[lo : hi + 1]) - price_in) / stop_pts)

    return (
        exit_position,
        exit_reason,
        float(exit_price),
        ambiguous,
        gross_r,
        mfe_r,
        mae_r,
        float(price_in),
        np.asarray(stop_track, dtype=np.float64),
        np.asarray(target_track, dtype=np.float64),
    )


def single_flex_exit(
    start: int,
    direction: int,
    stop_pts: float,
    target_pts: float,
    exit_cfg: ExitConfig,
    arrays: dict,
    trail_pts: float = 0.0,
) -> FlexTradeResult:
    """Simulate one trade and return its full result including per-bar levels."""

    is_long = direction > 0
    (
        exit_position,
        exit_reason,
        exit_price,
        ambiguous,
        gross_r,
        mfe_r,
        mae_r,
        price_in,
        stop_track,
        target_track,
    ) = _walk_one(
        int(start), is_long, stop_pts, target_pts, trail_pts, exit_cfg, arrays, record=True
    )
    return FlexTradeResult(
        entry_position=int(start),
        entry_price=price_in,
        exit_position=int(exit_position),
        exit_price=exit_price,
        exit_reason=exit_reason,
        direction=1 if is_long else -1,
        ambiguous_bar=ambiguous,
        holding_minutes=int(exit_position - start + 1),
        initial_stop_points=float(stop_pts),
        stop_ticks=float(stop_pts / exit_cfg.tick_size),
        gross_r=gross_r,
        mfe_r=mfe_r,
        mae_r=mae_r,
        stop_track=stop_track,
        target_track=target_track,
    )


def simulate_flex(
    entry_positions: np.ndarray,
    directions: np.ndarray,
    stop_points: np.ndarray,
    target_points: np.ndarray,
    exit_cfg: ExitConfig,
    metadata: pd.DataFrame,
    arrays: dict,
    trail_points: np.ndarray | None = None,
) -> pd.DataFrame:
    """One-position chronological walk over fired entries with flexible exits.

    The overlap-skip walk is identical to ``strategy_lab.simulate_positions``. The
    columns match it (plus mfe_r/mae_r), so a frozen-config run reconciles to the
    tick.
    """

    partitions = metadata["research_partition"].to_numpy()
    sessions = metadata["entry_session"].to_numpy()
    dates = metadata["trade_date_ny"].to_numpy()
    if trail_points is None:
        trail_points = np.zeros(len(entry_positions), dtype=np.float64)

    order = np.argsort(entry_positions, kind="mergesort")
    flat_from = -1
    records: list[dict] = []
    for row in order:
        start = int(entry_positions[row])
        if start <= flat_from:
            continue
        is_long = directions[row] > 0
        (
            exit_position,
            exit_reason,
            _exit_price,
            ambiguous,
            gross_r,
            mfe_r,
            mae_r,
            _price_in,
            _st,
            _tt,
        ) = _walk_one(
            start,
            is_long,
            float(stop_points[row]),
            float(target_points[row]),
            float(trail_points[row]),
            exit_cfg,
            arrays,
            record=False,
        )
        records.append(
            {
                "research_partition": partitions[row],
                "entry_session": sessions[row],
                "trade_date_ny": dates[row],
                "direction": 1 if is_long else -1,
                "entry_position": start,
                "exit_position": exit_position,
                "holding_minutes": int(exit_position - start + 1),
                "exit_reason": exit_reason,
                "ambiguous_bar": ambiguous,
                "stop_ticks": float(stop_points[row]) / exit_cfg.tick_size,
                "gross_r": gross_r,
                "mfe_r": mfe_r,
                "mae_r": mae_r,
            }
        )
        flat_from = int(exit_position)
    return pd.DataFrame.from_records(records)
