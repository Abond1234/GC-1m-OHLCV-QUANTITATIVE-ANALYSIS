"""Ordered first-passage research for Section 7 True POI opportunities.

The engine evaluates many event-specific levels against one sorted GC bar array.
It processes events in chunks, so complexity is O(E * H * T) for E events,
maximum H one-minute bars, and T targets, with O(chunk_size * H) peak memory.
It never scans the full bar table separately for each policy row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FirstPassageConfig:
    """Execution and path constraints for intraday GC research."""

    tick_size: float = 0.10
    last_entry_minute_ny: int = 12 * 60
    forced_exit_minute_ny: int = 15 * 60 + 30
    chunk_size: int = 4_000
    max_path_minutes: int = 750


FIRST_PASSAGE_REQUIRED_BAR_COLUMNS = (
    "bar_id",
    "ts_event_utc",
    "trade_date_ny",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "minute_of_day_ny",
    "continuous_segment_id",
)


def prepare_first_passage_bars(research_bars: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted GC bar frame with the Section 6 positional ``bar_id``."""

    required_without_id = [c for c in FIRST_PASSAGE_REQUIRED_BAR_COLUMNS if c != "bar_id"]
    _require_columns(research_bars, required_without_id, "research_bars")
    columns = list(dict.fromkeys(required_without_id + ["bar_id"]))
    columns = [c for c in columns if c in research_bars.columns]
    bars = (
        research_bars.loc[research_bars["product"].eq("GC"), columns]
        .sort_values(["trade_date_ny", "ts_event_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    bars["bar_id"] = np.arange(len(bars), dtype=np.int32)
    return bars[list(FIRST_PASSAGE_REQUIRED_BAR_COLUMNS)]


def evaluate_first_passage(
    opportunities: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    target_rs: Iterable[float],
    ambiguity_treatment: str = "conservative",
    max_holding_minutes: int | None = None,
    config: FirstPassageConfig | None = None,
) -> pd.DataFrame:
    """Evaluate event-specific entry, stop, and target levels in path order.

    Required opportunity columns are ``true_trade_opportunity_id``,
    ``retest_bar_id``, ``entry_bar_id``, ``entry_price``, ``stop_price``, and
    ``trade_side``. ``entry_model`` may be ``boundary_touch`` (a resting limit
    that must trade in the entry bar) or ``next_bar_confirmation`` (a market
    entry at the supplied next-bar price).
    """

    cfg = config or FirstPassageConfig()
    if ambiguity_treatment not in {"conservative", "exclude", "optimistic"}:
        raise ValueError("ambiguity_treatment must be conservative, exclude, or optimistic")
    required = {
        "true_trade_opportunity_id",
        "retest_bar_id",
        "entry_bar_id",
        "entry_price",
        "stop_price",
        "trade_side",
    }
    _require_columns(opportunities, required, "opportunities")
    _require_columns(bars, FIRST_PASSAGE_REQUIRED_BAR_COLUMNS, "bars")

    target_values = np.asarray(tuple(target_rs), dtype="float64")
    if target_values.size == 0 or (~np.isfinite(target_values)).any() or (target_values <= 0).any():
        raise ValueError("target_rs must contain positive finite values")

    base = opportunities.reset_index(drop=True).copy()
    n = len(base)
    if n == 0:
        return _empty_result()

    bar_id = bars["bar_id"].to_numpy("int64", copy=False)
    if not np.array_equal(bar_id, np.arange(len(bars))):
        raise ValueError("bars.bar_id must be contiguous and positional")
    high = bars["high"].to_numpy("float64", copy=False)
    low = bars["low"].to_numpy("float64", copy=False)
    close = bars["close"].to_numpy("float64", copy=False)
    minute = bars["minute_of_day_ny"].to_numpy("int32", copy=False)
    segment = bars["continuous_segment_id"].to_numpy("int64", copy=False)
    symbol = bars["symbol"].astype("string").to_numpy()
    date_code, _ = pd.factorize(bars["trade_date_ny"], sort=False)
    forced_exit = _forced_exit_index(minute, segment, date_code, cfg.forced_exit_minute_ny)
    state_run_end = _state_run_end(segment, date_code, symbol)

    entry_idx = pd.to_numeric(base["entry_bar_id"], errors="coerce").fillna(-1).to_numpy("int64")
    retest_idx = pd.to_numeric(base["retest_bar_id"], errors="coerce").fillna(-1).to_numpy("int64")
    entry = pd.to_numeric(base["entry_price"], errors="coerce").to_numpy("float64")
    stop = pd.to_numeric(base["stop_price"], errors="coerce").to_numpy("float64")
    long_side = base["trade_side"].eq("long").to_numpy(bool)
    entry_model = base.get("entry_model", pd.Series("boundary_touch", index=base.index)).astype("string")

    in_bounds = (entry_idx >= 0) & (entry_idx < len(bars)) & (retest_idx >= 0) & (retest_idx < len(bars))
    risk = np.where(long_side, entry - stop, stop - entry)
    valid_level = np.isfinite(entry) & np.isfinite(stop) & np.isfinite(risk) & (risk > 0)
    valid_path = np.zeros(n, dtype=bool)
    entry_filled = np.zeros(n, dtype=bool)
    no_fill = np.zeros(n, dtype=bool)
    invalid_path = ~(in_bounds & valid_level)
    end_idx = np.full(n, -1, dtype="int64")

    eligible = np.flatnonzero(in_bounds & valid_level)
    if eligible.size:
        ei = entry_idx[eligible]
        ri = retest_idx[eligible]
        same_state = (
            (segment[ei] == segment[ri])
            & (date_code[ei] == date_code[ri])
            & (symbol[ei] == symbol[ri])
            & (minute[ei] <= cfg.last_entry_minute_ny)
        )
        forced = forced_exit[ei]
        # A valid path needs at least one observable bar after entry. A segment
        # ending on the entry bar cannot borrow the next segment's prices.
        has_exit = forced > ei
        requested_end = forced.copy()
        if max_holding_minutes is not None:
            requested_end = np.minimum(requested_end, ei + int(max_holding_minutes))
        same_end_state = np.zeros_like(has_exit)
        ok_end = requested_end < len(bars)
        same_end_state[ok_end] = (
            (segment[requested_end[ok_end]] == segment[ei[ok_end]])
            & (date_code[requested_end[ok_end]] == date_code[ei[ok_end]])
            & (symbol[requested_end[ok_end]] == symbol[ei[ok_end]])
            & (requested_end[ok_end] <= state_run_end[ei[ok_end]])
        )
        local_valid = same_state & has_exit & same_end_state
        valid_rows = eligible[local_valid]
        end_idx[valid_rows] = requested_end[local_valid]
        valid_path[valid_rows] = True

    limit_model = entry_model.eq("boundary_touch").to_numpy(bool)
    valid_rows = np.flatnonzero(valid_path)
    if valid_rows.size:
        ei = entry_idx[valid_rows]
        limit_fill = (low[ei] <= entry[valid_rows]) & (high[ei] >= entry[valid_rows])
        entry_filled[valid_rows] = np.where(limit_model[valid_rows], limit_fill, True)
        no_fill[valid_rows] = ~entry_filled[valid_rows]
    invalid_path |= in_bounds & valid_level & ~valid_path

    records: list[pd.DataFrame] = []
    max_h = cfg.max_path_minutes if max_holding_minutes is None else min(
        cfg.max_path_minutes, int(max_holding_minutes)
    )
    offsets = np.arange(max_h + 1, dtype="int64")

    for start in range(0, n, cfg.chunk_size):
        stop_row = min(start + cfg.chunk_size, n)
        rows = np.arange(start, stop_row, dtype="int64")
        m = len(rows)
        idx = entry_idx[rows, None] + offsets[None, :]
        safe_idx = np.clip(idx, 0, len(bars) - 1)
        path_mask = (
            valid_path[rows, None]
            & entry_filled[rows, None]
            & (idx <= end_idx[rows, None])
            & (idx >= entry_idx[rows, None])
        )
        path_high = high[safe_idx]
        path_low = low[safe_idx]
        path_close = close[safe_idx]
        is_long = long_side[rows, None]
        stop_hit = path_mask & np.where(
            is_long,
            path_low <= stop[rows, None],
            path_high >= stop[rows, None],
        )
        first_stop = _first_true(stop_hit)
        any_stop = stop_hit.any(axis=1)

        favorable = np.where(
            is_long,
            path_high - entry[rows, None],
            entry[rows, None] - path_low,
        )
        adverse = np.where(
            is_long,
            entry[rows, None] - path_low,
            path_high - entry[rows, None],
        )
        favorable = np.where(path_mask, favorable, -np.inf)
        adverse = np.where(path_mask, adverse, -np.inf)
        max_uncapped_r = _safe_divide(np.max(favorable, axis=1), risk[rows])
        max_adverse_r = _safe_divide(np.max(adverse, axis=1), risk[rows])
        max_uncapped_r[~(valid_path[rows] & entry_filled[rows])] = np.nan
        max_adverse_r[~(valid_path[rows] & entry_filled[rows])] = np.nan

        exit_offset = np.maximum(end_idx[rows] - entry_idx[rows], 0)
        exit_offset = np.minimum(exit_offset, max_h)
        exit_close = path_close[np.arange(m), exit_offset]
        forced_points = np.where(
            long_side[rows],
            exit_close - entry[rows],
            entry[rows] - exit_close,
        )
        forced_r = _safe_divide(forced_points, risk[rows])

        for target_r in target_values:
            target_price = np.where(
                long_side[rows],
                entry[rows] + target_r * risk[rows],
                entry[rows] - target_r * risk[rows],
            )
            target_hit = path_mask & np.where(
                is_long,
                path_high >= target_price[:, None],
                path_low <= target_price[:, None],
            )
            first_target = _first_true(target_hit)
            any_target = target_hit.any(axis=1)
            ambiguity = any_stop & any_target & (first_stop == first_target)
            entry_bar_ambiguity = ambiguity & first_stop.eq(0) if isinstance(first_stop, pd.Series) else ambiguity & (first_stop == 0)

            target_first = any_target & (~any_stop | (first_target < first_stop))
            stop_first = any_stop & (~any_target | (first_stop < first_target))
            if ambiguity_treatment == "conservative":
                stop_first |= ambiguity
            elif ambiguity_treatment == "optimistic":
                target_first |= ambiguity

            unresolved = valid_path[rows] & entry_filled[rows] & ~(target_first | stop_first)
            if ambiguity_treatment == "exclude":
                unresolved &= ~ambiguity
            realized_r = np.where(target_first, target_r, np.where(stop_first, -1.0, forced_r))
            realized_r[~(valid_path[rows] & entry_filled[rows])] = np.nan
            if ambiguity_treatment == "exclude":
                realized_r[ambiguity] = np.nan

            frame = pd.DataFrame(
                {
                    "row_number": rows,
                    "target_r": target_r,
                    "entry_filled": entry_filled[rows],
                    "target_hit_before_stop": target_first,
                    "stop_hit_before_target": stop_first,
                    "forced_exit_before_either": unresolved,
                    "no_fill": no_fill[rows],
                    "same_bar_ambiguity": ambiguity,
                    "entry_bar_ambiguity": entry_bar_ambiguity,
                    "invalid_path": invalid_path[rows],
                    "time_to_entry_minutes": np.where(entry_filled[rows], entry_idx[rows] - retest_idx[rows], np.nan),
                    "time_from_entry_to_stop_minutes": np.where(any_stop, first_stop, np.nan),
                    "time_from_entry_to_target_minutes": np.where(any_target, first_target, np.nan),
                    "realized_r": realized_r,
                    "realized_r_at_forced_exit": forced_r,
                    "maximum_uncapped_r": max_uncapped_r,
                    "maximum_adverse_r": max_adverse_r,
                    "ambiguity_treatment": ambiguity_treatment,
                }
            )
            records.append(frame)

    result = pd.concat(records, ignore_index=True) if records else _empty_result()
    result = result.merge(
        base.reset_index(names="row_number")[
            [
                "row_number",
                "true_trade_opportunity_id",
                *[
                    c
                    for c in (
                        "true_poi_id",
                        "true_retest_id",
                        "trade_date_ny",
                        "hypothesis",
                        "trade_side",
                        "entry_model",
                        "stop_model",
                        "risk_ticks",
                        "risk_atr",
                        "risk_in_poi_widths",
                        "risk_25_100_tick_flag",
                    )
                    if c in base.columns
                ],
            ]
        ],
        on="row_number",
        how="left",
        validate="many_to_one",
    )
    return result.drop(columns="row_number")


def _forced_exit_index(
    minute: np.ndarray,
    segment: np.ndarray,
    date_code: np.ndarray,
    forced_exit_minute: int,
) -> np.ndarray:
    out = np.full(len(minute), -1, dtype="int64")
    keys = pd.DataFrame({"segment": segment, "date": date_code, "pos": np.arange(len(minute))})
    for _, positions in keys.groupby(["date", "segment"], sort=False, observed=True)["pos"]:
        pos = positions.to_numpy("int64", copy=False)
        eligible = pos[minute[pos] <= forced_exit_minute]
        if eligible.size:
            out[pos] = eligible[-1]
    return out


def _first_true(values: np.ndarray) -> np.ndarray:
    any_true = values.any(axis=1)
    first = np.argmax(values, axis=1).astype("float64")
    first[~any_true] = np.nan
    return first


def _state_run_end(
    segment: np.ndarray, date_code: np.ndarray, symbol: np.ndarray
) -> np.ndarray:
    """Return the last contiguous index sharing date, segment, and contract."""

    n = len(segment)
    if n == 0:
        return np.empty(0, dtype="int64")
    changes = np.flatnonzero(
        (segment[1:] != segment[:-1])
        | (date_code[1:] != date_code[:-1])
        | (symbol[1:] != symbol[:-1])
    ) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, n - 1]
    return np.repeat(ends, ends - starts + 1).astype("int64", copy=False)


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "true_trade_opportunity_id",
            "target_r",
            "entry_filled",
            "target_hit_before_stop",
            "stop_hit_before_target",
            "forced_exit_before_either",
            "no_fill",
            "same_bar_ambiguity",
            "entry_bar_ambiguity",
            "invalid_path",
            "time_to_entry_minutes",
            "time_from_entry_to_stop_minutes",
            "time_from_entry_to_target_minutes",
            "realized_r",
            "realized_r_at_forced_exit",
            "maximum_uncapped_r",
            "maximum_adverse_r",
            "ambiguity_treatment",
        ]
    )


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan)
    np.divide(numerator, denominator, out=out, where=np.isfinite(denominator) & (denominator > 0))
    return out
