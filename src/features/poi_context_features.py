"""Decision-time feature engineering for Section 7 True POI research.

The headline unit is one physical refined POI by one physical retest. Legacy
``canonical_*`` identifiers are accepted only as compatibility inputs and are
exposed as exact ``true_*`` aliases in all new objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PoiContextConfig:
    tick_size: float = 0.10
    approach_windows: tuple[int, ...] = (3, 5, 10, 15, 30)
    trend_windows: tuple[int, ...] = (5, 15, 30, 60)
    london_start_minute: int = 180
    london_end_minute: int = 360
    ny_start_minute: int = 420
    last_entry_minute_ny: int = 720
    forced_exit_minute_ny: int = 930
    development_end: str = "2023-12-31"


CONTEXT_BAR_COLUMNS = [
    "ts_event_utc",
    "ts_event_ny",
    "trade_date_ny",
    "product",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "minute_of_day_ny",
    "day_of_week",
    "continuous_segment_id",
    "bar_range",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "true_range",
    "log_return",
    "abs_log_return",
    "rolling_atr_20m",
    "rolling_atr_60m",
    "rolling_realized_vol_60m",
    "rolling_realized_vol_240m",
    "relative_volume_60m",
    "volume_zscore_240m",
    "tradable_research_flag",
    "roll_window_flag",
]


def add_true_poi_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exact True POI identity aliases without deleting legacy fields."""

    out = frame.copy()
    mapping = {
        "canonical_poi_id": "true_poi_id",
        "canonical_retest_id": "true_retest_id",
        "canonical_candidate_id": "true_trade_opportunity_id",
        "canonical_retest_number": "true_retest_number",
    }
    for legacy, current in mapping.items():
        if current not in out and legacy in out:
            out[current] = out[legacy]
        if legacy in out and current in out:
            mismatch = ~(out[legacy].astype("string").fillna("<NA>").eq(
                out[current].astype("string").fillna("<NA>")
            ))
            if mismatch.any():
                raise ValueError(f"{current} is not an exact alias of legacy field {legacy}")
    return out


def prepare_context_bars(research_bars: pd.DataFrame) -> pd.DataFrame:
    """Create the narrow, sorted GC feature base used throughout Section 7."""

    _require_columns(research_bars, CONTEXT_BAR_COLUMNS, "research_bars")
    bars = (
        research_bars.loc[research_bars["product"].eq("GC"), CONTEXT_BAR_COLUMNS]
        .sort_values(["trade_date_ny", "ts_event_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    bars["bar_id"] = np.arange(len(bars), dtype=np.int32)
    return bars


def build_true_poi_context_frame(
    signal_frame: pd.DataFrame,
    research_bars: pd.DataFrame,
    config: PoiContextConfig | None = None,
) -> pd.DataFrame:
    """Build one decision-time row per True POI/retest opportunity."""

    cfg = config or PoiContextConfig()
    signals = add_true_poi_aliases(signal_frame)
    required = {
        "true_poi_id",
        "true_retest_id",
        "retest_bar_id",
        "direction",
        "trade_date_ny",
        "poi_low",
        "poi_high",
        "poi_mid",
        "poi_size_ticks",
        "poi_geometry_case",
        "fvg_size_ticks",
        "close_open_gap_ticks",
        "prior_bar_id",
        "poi_bar_id",
        "confirmation_bar_id",
        "activation_bar_id",
        "displacement_start_bar_id",
        "swing_n",
        "break_mode",
    }
    _require_columns(signals, required, "signal_frame")
    bars = prepare_context_bars(research_bars)

    selected_columns = list(dict.fromkeys([
        *required,
        "canonical_poi_id", "canonical_retest_id", "canonical_candidate_id",
        "true_retest_number", "canonical_retest_number", "poi_variant_id",
        "candidate_variant_id", "retest_ts_event_utc", "retest_ts_event_ny",
        "execution_window_label", "time_since_previous_touch_minutes",
        "time_since_poi_activation_minutes", "structural_swing_break_flag",
        "structural_swing_window_broken", "structural_break_distance_ticks",
        "break_distance_ticks",
    ]))
    selected_columns = [c for c in selected_columns if c in signals.columns]

    # Only variants that were already active at this retest are present in the
    # signal frame. Aggregating within true_retest_id therefore cannot import a
    # later-activating research variant into an earlier decision.
    work = (
        signals.loc[:, selected_columns]
        .drop_duplicates(["true_retest_id", "poi_variant_id"], keep="first")
        .copy()
    )
    work["_structural_sort"] = work.get(
        "structural_swing_break_flag", pd.Series(False, index=work.index)
    ).fillna(False).astype(bool)
    work["_window_sort"] = pd.to_numeric(
        work.get("structural_swing_window_broken", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    ).fillna(-1)
    work = work.sort_values(
        ["true_retest_id", "_structural_sort", "_window_sort", "swing_n", "break_mode"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    base = work.drop_duplicates("true_retest_id", keep="first").copy()
    grouped = work.groupby("true_retest_id", observed=True, sort=False)
    aggregates = grouped.agg(
        feat_research_variant_count=("poi_variant_id", "nunique"),
        feat_smallest_swing_window=("swing_n", "min"),
        feat_largest_swing_window=("swing_n", "max"),
        feat_displacement_start_bar_id=("displacement_start_bar_id", "min"),
        feat_formation_available_bar_id=("activation_bar_id", "max"),
    )
    base = base.drop(columns=["_structural_sort", "_window_sort"], errors="ignore")
    base = base.merge(aggregates, on="true_retest_id", how="left", validate="one_to_one")
    for window in (3, 5, 7):
        values = grouped["swing_n"].apply(lambda s, value=window: bool(s.eq(value).any()))
        base[f"feat_has_swing_{window}_variant"] = base["true_retest_id"].map(values).fillna(False)
    for mode in ("wick", "close"):
        values = grouped["break_mode"].apply(lambda s, value=mode: bool(s.eq(value).any()))
        base[f"feat_has_{mode}_break_variant"] = base["true_retest_id"].map(values).fillna(False)

    structural_any = grouped["structural_swing_break_flag"].any()
    structural_max = grouped["structural_swing_window_broken"].max()
    base["feat_structural_swing_validation"] = base["true_retest_id"].map(structural_any).fillna(False)
    base["feat_largest_structural_window_broken"] = pd.to_numeric(
        base["true_retest_id"].map(structural_max), errors="coerce"
    ).astype("Int64")
    base["feat_15bar_structural_validation"] = (
        base["feat_largest_structural_window_broken"].eq(15).fillna(False)
    )
    base["feat_local_swing_only"] = ~base["feat_structural_swing_validation"]

    base["true_trade_opportunity_id"] = base["true_retest_id"].astype("string")
    base["true_retest_number"] = pd.to_numeric(
        base.get("true_retest_number", base.get("canonical_retest_number")), errors="coerce"
    ).astype("Int32")
    base["feat_previous_touch_count"] = (base["true_retest_number"] - 1).clip(lower=0)
    base["feat_first_touch"] = base["true_retest_number"].eq(1)
    base["feat_time_since_previous_touch_minutes"] = pd.to_numeric(
        base.get("time_since_previous_touch_minutes"), errors="coerce"
    )
    base["feat_poi_age_minutes"] = pd.to_numeric(
        base.get("time_since_poi_activation_minutes"), errors="coerce"
    )
    base["feat_execution_session"] = base["execution_window_label"].map(
        {"London Execution": "London", "New York Execution": "New York"}
    ).astype("category")

    # Feature families are built separately for timing auditability. Pandas may
    # emit fragmentation warnings while the wide research object is assembled;
    # the final column projection below creates the compact delivered frame.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        _add_formation_features(base, bars, cfg)
        _add_displacement_features(base, bars, cfg)
        _add_approach_features(base, bars, cfg)
        _add_touch_features(base, bars, cfg)
        _add_market_context_features(base, bars, cfg)
        _add_partition(base, cfg)
    _validate_feature_timing(base)

    identity = [
        "true_poi_id",
        "true_retest_id",
        "true_trade_opportunity_id",
        "true_retest_number",
        "trade_date_ny",
        "direction",
        "retest_bar_id",
        "retest_ts_event_utc",
        "retest_ts_event_ny",
    ]
    trace = [
        c
        for c in (
            "canonical_poi_id",
            "canonical_retest_id",
            "poi_variant_id",
            "candidate_variant_id",
        )
        if c in base
    ]
    diagnostics = [c for c in base if c.startswith("diag_")]
    features = [c for c in base if c.startswith("feat_")]
    passthrough = [
        c
        for c in (
            "poi_low",
            "poi_high",
            "poi_mid",
            "poi_size_ticks",
            "poi_geometry_case",
            "prior_bar_id",
            "poi_bar_id",
            "confirmation_bar_id",
            "activation_bar_id",
            "feat_displacement_start_bar_id",
            "feat_formation_available_bar_id",
            "execution_window_label",
            "research_partition",
        )
        if c in base
    ]
    ordered = list(dict.fromkeys(identity + trace + passthrough + features + diagnostics))
    return base[ordered].sort_values(["retest_bar_id", "true_poi_id"], kind="mergesort").reset_index(drop=True)


def _add_formation_features(out: pd.DataFrame, bars: pd.DataFrame, cfg: PoiContextConfig) -> None:
    arrays = _bar_arrays(bars)
    a = out["prior_bar_id"].to_numpy("int64")
    b = out["poi_bar_id"].to_numpy("int64")
    c = out["confirmation_bar_id"].to_numpy("int64")
    activation = out["feat_formation_available_bar_id"].to_numpy("int64")
    direction_sign = np.where(out["direction"].eq("bullish"), 1.0, -1.0)
    atr = arrays["atr60"][activation]
    atr_ticks = atr / cfg.tick_size
    out["feat_poi_case"] = out["poi_geometry_case"].astype("category")
    out["feat_poi_width_ticks"] = pd.to_numeric(out["poi_size_ticks"], errors="coerce")
    out["feat_poi_width_atr"] = _safe_divide(out["feat_poi_width_ticks"], atr_ticks)
    out["feat_fvg_size_ticks"] = pd.to_numeric(out["fvg_size_ticks"], errors="coerce")
    out["feat_fvg_size_atr"] = _safe_divide(out["feat_fvg_size_ticks"], atr_ticks)
    out["feat_opening_gap_ticks"] = pd.to_numeric(out["close_open_gap_ticks"], errors="coerce").abs()
    out["feat_opening_gap_atr"] = _safe_divide(out["feat_opening_gap_ticks"], atr_ticks)
    out["feat_fvg_to_poi_width_ratio"] = _safe_divide(out["feat_fvg_size_ticks"], out["feat_poi_width_ticks"])
    out["feat_opening_gap_to_poi_width_ratio"] = _safe_divide(out["feat_opening_gap_ticks"], out["feat_poi_width_ticks"])
    middle_body = np.abs(arrays["close"][b] - arrays["open"][b])
    middle_range = arrays["high"][b] - arrays["low"][b]
    confirm_body = np.abs(arrays["close"][c] - arrays["open"][c])
    confirm_range = arrays["high"][c] - arrays["low"][c]
    out["feat_middle_body_ticks"] = middle_body / cfg.tick_size
    out["feat_middle_body_to_range_ratio"] = _safe_divide(middle_body, middle_range)
    out["feat_confirmation_body_ticks"] = confirm_body / cfg.tick_size
    out["feat_confirmation_body_to_range_ratio"] = _safe_divide(confirm_body, confirm_range)
    out["feat_confirmation_close_location"] = _safe_divide(
        arrays["close"][c] - arrays["low"][c], confirm_range
    )
    out["feat_confirmation_directional_close_location"] = np.where(
        direction_sign > 0,
        out["feat_confirmation_close_location"],
        1.0 - out["feat_confirmation_close_location"],
    )
    out["feat_distance_beyond_broken_swing_ticks"] = pd.to_numeric(
        out.get("structural_break_distance_ticks", out.get("break_distance_ticks")), errors="coerce"
    )
    out["diag_formation_availability_bar_id"] = activation.astype("int32")
    out["diag_formation_availability_ts"] = arrays["ts"][activation]


def _add_displacement_features(out: pd.DataFrame, bars: pd.DataFrame, cfg: PoiContextConfig) -> None:
    arrays = _bar_arrays(bars)
    starts = out["feat_displacement_start_bar_id"].to_numpy("int64")
    ends = out["feat_formation_available_bar_id"].to_numpy("int64")
    direction = np.where(out["direction"].eq("bullish"), 1.0, -1.0)
    keys = pd.DataFrame({"start": starts, "end": ends, "direction": direction}).drop_duplicates()
    records: dict[tuple[int, int, float], dict[str, float]] = {}
    for row in keys.itertuples(index=False):
        start, end, sign = int(row.start), int(row.end), float(row.direction)
        key = (start, end, sign)
        if start < 0 or end < start or end >= len(bars):
            records[key] = {}
            continue
        idx = np.arange(start, end + 1)
        if arrays["segment"][start] != arrays["segment"][end] or arrays["date"][start] != arrays["date"][end]:
            records[key] = {}
            continue
        close = arrays["close"][idx]
        open_ = arrays["open"][idx]
        high = arrays["high"][idx]
        low = arrays["low"][idx]
        volume = arrays["volume"][idx]
        ranges = high - low
        bodies = np.abs(close - open_)
        changes = np.diff(close)
        net = sign * (close[-1] - open_[0])
        gross = np.abs(changes).sum()
        progress = sign * (close - open_[0])
        peak = np.maximum.accumulate(progress)
        pullback = np.max(peak - progress) if len(progress) else np.nan
        half = max(1, len(idx) // 2)
        first_speed = sign * (close[half - 1] - open_[0]) / half
        second_speed = sign * (close[-1] - close[half - 1]) / max(len(idx) - half, 1)
        volume_total = volume.sum()
        records[key] = {
            "duration_bars": len(idx),
            "net_ticks": net / cfg.tick_size,
            "gross_ticks": gross / cfg.tick_size,
            "efficiency": float(_safe_scalar(abs(net), gross)),
            "cumulative_range_ticks": ranges.sum() / cfg.tick_size,
            "average_range_ticks": ranges.mean() / cfg.tick_size,
            "maximum_range_ticks": ranges.max() / cfg.tick_size,
            "average_body_ticks": bodies.mean() / cfg.tick_size,
            "directional_body_share": float(np.mean(sign * (close - open_) > 0)),
            "directional_candle_count": int(np.sum(sign * (close - open_) > 0)),
            "opposing_candle_count": int(np.sum(sign * (close - open_) < 0)),
            "maximum_internal_pullback_ticks": pullback / cfg.tick_size,
            "maximum_internal_pullback_fraction": float(_safe_scalar(pullback, abs(net))),
            "volume": volume_total,
            "relative_volume": float(_safe_scalar(volume.mean(), np.nanmean(arrays["volume"][max(0, start - 60):start]) if start else np.nan)),
            "peak_volume": volume.max(),
            "volume_concentration": float(_safe_scalar(volume.max(), volume_total)),
            "range_expansion": float(_safe_scalar(ranges[-min(3, len(ranges)):].mean(), ranges[:min(3, len(ranges))].mean())),
            "body_expansion": float(_safe_scalar(bodies[-min(3, len(bodies)):].mean(), bodies[:min(3, len(bodies))].mean())),
            "acceleration_ticks_per_bar2": (second_speed - first_speed) / cfg.tick_size,
            "imbalance_density": float(np.mean((low[2:] > high[:-2]) | (high[2:] < low[:-2]))) if len(idx) >= 3 else 0.0,
        }
    mapped = [records.get((int(s), int(e), float(d)), {}) for s, e, d in zip(starts, ends, direction)]
    frame = pd.DataFrame(mapped, index=out.index)
    for col in frame:
        out[f"feat_displacement_{col}"] = frame[col]
    atr_ticks = bars["rolling_atr_60m"].to_numpy("float64")[np.clip(ends, 0, len(bars) - 1)] / cfg.tick_size
    out["feat_displacement_net_atr"] = _safe_divide(out["feat_displacement_net_ticks"], atr_ticks)
    out["feat_displacement_speed_ticks_per_minute"] = _safe_divide(
        out["feat_displacement_net_ticks"], out["feat_displacement_duration_bars"]
    )


def _add_approach_features(out: pd.DataFrame, bars: pd.DataFrame, cfg: PoiContextConfig) -> None:
    arrays = _bar_arrays(bars)
    retest = out["retest_bar_id"].to_numpy("int64")
    prev = retest - 1
    prev_close = arrays["close"][prev]
    poi_low = out["poi_low"].to_numpy("float64")
    poi_high = out["poi_high"].to_numpy("float64")
    above = prev_close > poi_high
    below = prev_close < poi_low
    nearest_high = np.abs(prev_close - poi_high) <= np.abs(prev_close - poi_low)
    edge = np.where(above | (~below & nearest_high), poi_high, poi_low)
    attack_sign = np.where(above, -1.0, np.where(below, 1.0, np.sign(arrays["close"][retest] - prev_close)))
    attack_sign[attack_sign == 0] = np.where(out.loc[attack_sign == 0, "direction"].eq("bullish"), -1.0, 1.0)
    out["feat_first_contact_edge_price"] = edge
    out["feat_first_contact_edge"] = np.where(edge == poi_high, "high_edge", "low_edge")
    out["feat_approach_direction_sign"] = attack_sign.astype("int8")

    for window in cfg.approach_windows:
        offsets = np.arange(window, 0, -1)
        idx = retest[:, None] - offsets[None, :]
        valid = (
            (idx[:, 0] >= 0)
            & (arrays["segment"][np.clip(idx[:, 0], 0, len(bars) - 1)] == arrays["segment"][retest])
            & (arrays["date"][np.clip(idx[:, 0], 0, len(bars) - 1)] == arrays["date"][retest])
        )
        safe = np.clip(idx, 0, len(bars) - 1)
        close = arrays["close"][safe]
        open_ = arrays["open"][safe]
        high = arrays["high"][safe]
        low = arrays["low"][safe]
        volume = arrays["volume"][safe]
        ranges = high - low
        bodies = np.abs(close - open_)
        changes = np.diff(close, axis=1, prepend=open_[:, :1])
        aligned_changes = changes * attack_sign[:, None]
        net = (close[:, -1] - close[:, 0]) * attack_sign
        gross = np.abs(changes).sum(axis=1)
        x = np.arange(window, dtype="float64")
        x_centered = x - x.mean()
        slope = ((close - close.mean(axis=1, keepdims=True)) * x_centered).sum(axis=1) / np.square(x_centered).sum() if window > 1 else np.zeros(len(out))
        half = max(1, window // 2)
        speed1 = (close[:, half - 1] - close[:, 0]) * attack_sign / half
        speed2 = (close[:, -1] - close[:, half - 1]) * attack_sign / max(window - half, 1)
        progress = (close - close[:, :1]) * attack_sign[:, None]
        counter = np.maximum.accumulate(progress, axis=1) - progress
        toward = aligned_changes > 0
        consecutive = np.cumprod(toward[:, ::-1], axis=1).sum(axis=1)
        prior_ranges = ranges[:, :-3].mean(axis=1) if window > 3 else ranges.mean(axis=1)
        recent_ranges = ranges[:, -min(3, window):].mean(axis=1)
        overlap = np.maximum(0.0, np.minimum(high[:, 1:], high[:, :-1]) - np.maximum(low[:, 1:], low[:, :-1]))
        overlap_denom = np.minimum(ranges[:, 1:], ranges[:, :-1])
        vol_slope = ((volume - volume.mean(axis=1, keepdims=True)) * x_centered).sum(axis=1) / np.square(x_centered).sum() if window > 1 else np.zeros(len(out))
        prefix = f"feat_approach_{window}m_"
        values = {
            "origin_distance_to_poi_ticks": np.abs(edge - close[:, 0]) / cfg.tick_size,
            "duration_minutes": np.full(len(out), window, dtype="float64"),
            "speed_ticks_per_minute": net / cfg.tick_size / window,
            "speed_atr": _safe_divide(net, arrays["atr60"][prev]),
            "net_return": _safe_divide(close[:, -1] - close[:, 0], close[:, 0]),
            "net_ticks_toward_poi": net / cfg.tick_size,
            "gross_path_ticks": gross / cfg.tick_size,
            "efficiency_ratio": _safe_divide(np.abs(net), gross),
            "slope_ticks_per_minute": slope * attack_sign / cfg.tick_size,
            "acceleration_ticks_per_minute2": (speed2 - speed1) / cfg.tick_size,
            "consecutive_candles_toward_poi": consecutive,
            "opposing_candle_count": (aligned_changes < 0).sum(axis=1),
            "maximum_countermove_ticks": counter.max(axis=1) / cfg.tick_size,
            "average_body_ticks": bodies.mean(axis=1) / cfg.tick_size,
            "average_range_ticks": ranges.mean(axis=1) / cfg.tick_size,
            "body_to_range_ratio": _safe_divide(bodies.sum(axis=1), ranges.sum(axis=1)),
            "candle_overlap_ratio": np.nanmean(_safe_divide(overlap, overlap_denom), axis=1) if window > 1 else np.zeros(len(out)),
            "range_compression_ratio": _safe_divide(recent_ranges, prior_ranges),
            "range_expansion_ratio": _safe_divide(ranges[:, -1], ranges[:, 0]),
            "directional_efficiency": _safe_divide(np.abs(net), gross),
            "volume": volume.sum(axis=1),
            "relative_volume": _safe_divide(volume.mean(axis=1), arrays["volume_tod_mean"][prev]),
            "volume_slope": vol_slope,
            "volume_acceleration": volume[:, -min(3, window):].mean(axis=1) - volume[:, :min(3, window)].mean(axis=1),
            "volume_spike_flag": (arrays["volume_z"][safe] >= 2.0).any(axis=1),
        }
        for name, value in values.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                arr = np.repeat(arr, len(out))
            if arr.dtype.kind in "fc":
                arr = arr.astype("float64")
                arr[~valid] = np.nan
            elif arr.dtype.kind in "iu":
                arr = arr.astype("float64")
                arr[~valid] = np.nan
            else:
                arr = arr.copy()
                arr[~valid] = False
            out[prefix + name] = arr

    out["feat_displacement_volume_to_approach_volume_ratio"] = _safe_divide(
        out["feat_displacement_volume"], out["feat_approach_15m_volume"]
    )
    out["feat_displacement_speed_to_approach_speed_ratio"] = _safe_divide(
        out["feat_displacement_speed_ticks_per_minute"].abs(),
        out["feat_approach_15m_speed_ticks_per_minute"].abs(),
    )
    large = arrays["abs_return"] >= np.nan_to_num(arrays["atr60"] / arrays["close"], nan=np.inf) * 0.75
    last = np.maximum.accumulate(np.where(large, np.arange(len(bars)), -1))
    out["feat_minutes_since_latest_large_move"] = np.where(last[prev] >= 0, prev - last[prev], np.nan)
    gross_prefix = np.cumsum(arrays["abs_change"])
    activation = out["feat_formation_available_bar_id"].to_numpy("int64")
    out["feat_distance_travelled_since_activation_ticks"] = (
        gross_prefix[prev] - np.where(activation > 0, gross_prefix[activation - 1], 0.0)
    ) / cfg.tick_size
    out["diag_pretouch_availability_bar_id"] = prev.astype("int32")
    out["diag_pretouch_availability_ts"] = arrays["ts"][prev]


def _add_touch_features(out: pd.DataFrame, bars: pd.DataFrame, cfg: PoiContextConfig) -> None:
    arrays = _bar_arrays(bars)
    rid = out["retest_bar_id"].to_numpy("int64")
    attack = out["feat_approach_direction_sign"].to_numpy("float64")
    edge = out["feat_first_contact_edge_price"].to_numpy("float64")
    low_zone = out["poi_low"].to_numpy("float64")
    high_zone = out["poi_high"].to_numpy("float64")
    width = high_zone - low_zone
    distal = np.where(attack < 0, low_zone, high_zone)
    penetration = np.where(attack < 0, edge - arrays["low"][rid], arrays["high"][rid] - edge)
    penetration = np.maximum(penetration, 0.0)
    touch_range = arrays["high"][rid] - arrays["low"][rid]
    body = np.abs(arrays["close"][rid] - arrays["open"][rid])
    upper = arrays["high"][rid] - np.maximum(arrays["open"][rid], arrays["close"][rid])
    lower = np.minimum(arrays["open"][rid], arrays["close"][rid]) - arrays["low"][rid]
    rejection = np.where(attack < 0, lower, upper)
    out["feat_touch_penetration_ticks"] = penetration / cfg.tick_size
    out["feat_touch_penetration_fraction"] = _safe_divide(penetration, width)
    out["feat_touch_penetration_atr"] = _safe_divide(penetration, arrays["atr60"][rid])
    out["feat_touch_boundary_reached"] = penetration >= 0
    out["feat_touch_midpoint_reached"] = out["feat_touch_penetration_fraction"] >= 0.5
    out["feat_touch_distal_reached"] = out["feat_touch_penetration_fraction"] >= 1.0
    out["feat_touch_full_zone_traversal"] = (arrays["high"][rid] >= high_zone) & (arrays["low"][rid] <= low_zone)
    out["feat_touch_wick_through"] = np.where(attack < 0, arrays["low"][rid] < distal, arrays["high"][rid] > distal)
    out["feat_touch_close_through"] = np.where(attack < 0, arrays["close"][rid] < distal, arrays["close"][rid] > distal)
    out["feat_touch_close_inside_poi"] = (arrays["close"][rid] >= low_zone) & (arrays["close"][rid] <= high_zone)
    out["feat_touch_candle_direction"] = np.sign(arrays["close"][rid] - arrays["open"][rid]).astype("int8")
    out["feat_touch_body_ticks"] = body / cfg.tick_size
    out["feat_touch_body_to_range_ratio"] = _safe_divide(body, touch_range)
    out["feat_touch_upper_wick_ticks"] = upper / cfg.tick_size
    out["feat_touch_lower_wick_ticks"] = lower / cfg.tick_size
    out["feat_touch_directional_rejection_wick_ticks"] = rejection / cfg.tick_size
    out["feat_touch_rejection_wick_to_body_ratio"] = _safe_divide(rejection, body)
    out["feat_touch_close_location"] = _safe_divide(arrays["close"][rid] - arrays["low"][rid], touch_range)
    out["feat_touch_relative_volume"] = arrays["rel_volume"][rid]
    out["feat_touch_volume_zscore"] = arrays["volume_z"][rid]
    history = rid[:, None] - np.arange(29, -1, -1)[None, :]
    safe = np.clip(history, 0, len(bars) - 1)
    overlap = (arrays["high"][safe] >= low_zone[:, None]) & (arrays["low"][safe] <= high_zone[:, None])
    valid = (history >= 0) & (arrays["segment"][safe] == arrays["segment"][rid, None])
    overlap &= valid
    out["feat_touch_consecutive_bars_inside"] = np.cumprod(overlap[:, ::-1], axis=1).sum(axis=1)
    out["feat_touch_recent_bars_inside_30m"] = overlap.sum(axis=1)
    out["diag_touch_close_availability_bar_id"] = rid.astype("int32")
    out["diag_touch_close_availability_ts"] = arrays["ts"][rid]


def _add_market_context_features(out: pd.DataFrame, bars: pd.DataFrame, cfg: PoiContextConfig) -> None:
    arrays = _bar_arrays(bars)
    rid = out["retest_bar_id"].to_numpy("int64")
    decision = rid - 1
    close = arrays["close"]
    vwap = arrays["vwap"]
    atr = arrays["atr60"]
    out["feat_price_minus_vwap_ticks"] = (close[decision] - vwap[decision]) / cfg.tick_size
    out["feat_distance_from_vwap_ticks"] = np.abs(out["feat_price_minus_vwap_ticks"])
    out["feat_distance_from_vwap_atr"] = _safe_divide(close[decision] - vwap[decision], atr[decision])
    out["feat_price_above_vwap"] = close[decision] > vwap[decision]
    activation = out["feat_formation_available_bar_id"].to_numpy("int64")
    out["feat_poi_distance_from_vwap_at_activation_atr"] = _safe_divide(
        out["poi_mid"].to_numpy("float64") - vwap[activation], atr[activation]
    )
    out["feat_change_in_vwap_distance_during_approach_ticks"] = (
        (close[decision] - vwap[decision]) - (close[np.maximum(decision - 15, 0)] - vwap[np.maximum(decision - 15, 0)])
    ) / cfg.tick_size
    cross = np.sign(close - vwap)
    cross_change = np.r_[False, cross[1:] != cross[:-1]].astype("int32")
    cross_prefix = np.cumsum(cross_change)
    out["feat_recent_vwap_crosses_30m"] = cross_prefix[decision] - np.where(decision >= 30, cross_prefix[decision - 30], 0)
    for window in (5, 15, 30):
        prior = decision - window
        valid = prior >= 0
        values = np.full(len(out), np.nan)
        values[valid] = (vwap[decision[valid]] - vwap[prior[valid]]) / cfg.tick_size / window
        out[f"feat_vwap_slope_{window}m_ticks_per_minute"] = values
    for window in cfg.trend_windows:
        prior = decision - window
        valid = (prior >= 0) & (arrays["segment"][np.clip(prior, 0, len(bars) - 1)] == arrays["segment"][decision])
        ret = np.full(len(out), np.nan)
        slope = np.full(len(out), np.nan)
        ret[valid] = close[decision[valid]] / close[prior[valid]] - 1.0
        slope[valid] = (close[decision[valid]] - close[prior[valid]]) / cfg.tick_size / window
        out[f"feat_return_{window}m"] = ret
        out[f"feat_slope_{window}m_ticks_per_minute"] = slope
    out["feat_trend_efficiency_30m"] = out["feat_approach_30m_directional_efficiency"]
    h15 = _rolling_extreme(arrays["high"], 15, "max")
    l15 = _rolling_extreme(arrays["low"], 15, "min")
    h30 = _rolling_extreme(arrays["high"], 30, "max")
    l30 = _rolling_extreme(arrays["low"], 30, "min")
    out["feat_rolling_high_low_structure_ticks"] = (h30[decision] - l30[decision]) / cfg.tick_size
    out["feat_higher_high_higher_low_state"] = (h15[decision] > h30[np.maximum(decision - 15, 0)]) & (l15[decision] > l30[np.maximum(decision - 15, 0)])
    out["feat_lower_high_lower_low_state"] = (h15[decision] < h30[np.maximum(decision - 15, 0)]) & (l15[decision] < l30[np.maximum(decision - 15, 0)])
    direction_sign = np.where(out["direction"].eq("bullish"), 1.0, -1.0)
    out["feat_trend_alignment_continuation"] = np.sign(out["feat_return_30m"].fillna(0)) == direction_sign
    out["feat_trend_alignment_reversal"] = np.sign(out["feat_return_30m"].fillna(0)) == -direction_sign

    short_atr = arrays["atr20"][decision]
    long_atr = atr[decision]
    out["feat_short_term_atr_ticks"] = short_atr / cfg.tick_size
    out["feat_long_term_atr_ticks"] = long_atr / cfg.tick_size
    out["feat_short_to_long_volatility_ratio"] = _safe_divide(short_atr, long_atr)
    out["feat_realized_volatility_60m"] = arrays["rv60"][decision]
    out["feat_realized_volatility_240m"] = arrays["rv240"][decision]
    out["feat_volatility_acceleration"] = _safe_divide(arrays["rv60"][decision], arrays["rv240"][decision])
    dev_mask = pd.to_datetime(out["trade_date_ny"]) <= pd.Timestamp(cfg.development_end)
    fit_values = pd.to_numeric(
        out.loc[dev_mask, "feat_realized_volatility_60m"], errors="coerce"
    ).dropna()
    if fit_values.empty:
        # Synthetic/sliced runs may contain no development observations. This
        # fallback is diagnostic only; the authoritative full run fits on the
        # development partition as required.
        fit_values = pd.to_numeric(out["feat_realized_volatility_60m"], errors="coerce").dropna()
    thresholds = (
        np.asarray(fit_values.quantile([0.25, 0.75, 0.90]).to_numpy(), dtype="float64")
        if not fit_values.empty
        else np.asarray([0.0, 0.0, 0.0], dtype="float64")
    )
    vol_values = out["feat_realized_volatility_60m"].to_numpy("float64")
    out["feat_volatility_percentile_devfit_bucket"] = pd.Categorical(
        np.select(
            [
                vol_values <= thresholds[0],
                vol_values <= thresholds[1],
                vol_values <= thresholds[2],
            ],
            ["low", "normal", "high"],
            default="extreme",
        ),
        categories=["low", "normal", "high", "extreme"],
        ordered=True,
    )
    out["diag_volatility_devfit_q25"] = thresholds[0]
    out["diag_volatility_devfit_q75"] = thresholds[1]
    out["diag_volatility_devfit_q90"] = thresholds[2]

    out["feat_raw_volume"] = arrays["volume"][decision]
    out["feat_relative_volume"] = arrays["rel_volume"][decision]
    out["feat_volume_zscore"] = arrays["volume_z"][decision]
    out["feat_time_of_day_adjusted_relative_volume"] = _safe_divide(arrays["volume"][decision], arrays["volume_tod_mean"][decision])
    out["feat_volume_spike_flag"] = arrays["volume_z"][decision] >= 2.0
    out["feat_touch_volume_to_approach_volume_ratio"] = _safe_divide(
        arrays["volume"][rid] * 15.0, out["feat_approach_15m_volume"]
    )

    minute = arrays["minute"][rid]
    session_open = np.where(minute < cfg.ny_start_minute, cfg.london_start_minute, cfg.ny_start_minute)
    out["feat_minutes_since_execution_window_open"] = minute - session_open
    out["feat_minutes_until_no_new_entry_cutoff"] = cfg.last_entry_minute_ny - minute
    out["feat_minutes_until_forced_exit"] = cfg.forced_exit_minute_ny - minute
    out["feat_minute_of_day"] = minute
    out["feat_day_of_week"] = bars["day_of_week"].astype("string").to_numpy()[rid]
    session_key = arrays["date"] * 2 + (
        arrays["minute"] >= cfg.ny_start_minute
    ).astype("int64")
    session_open_price, session_high, session_low = _session_state(arrays, session_key)
    out["feat_session_return_before_retest"] = close[decision] / session_open_price[decision] - 1.0
    out["feat_session_range_before_retest_ticks"] = (session_high[decision] - session_low[decision]) / cfg.tick_size
    out["feat_position_within_session_range"] = _safe_divide(close[decision] - session_low[decision], session_high[decision] - session_low[decision])
    out["feat_distance_from_session_open_atr"] = _safe_divide(close[decision] - session_open_price[decision], atr[decision])
    out["feat_fraction_recent_expected_range_travelled"] = _safe_divide(session_high[decision] - session_low[decision], atr[decision])
    out["feat_successive_body_decay_3m"] = _safe_divide(arrays["body"][decision], np.nanmean(np.column_stack([arrays["body"][np.maximum(decision - i, 0)] for i in (1, 2, 3)]), axis=1))
    out["feat_successive_range_decay_3m"] = _safe_divide(arrays["range"][decision], np.nanmean(np.column_stack([arrays["range"][np.maximum(decision - i, 0)] for i in (1, 2, 3)]), axis=1))
    out["feat_volume_price_divergence_proxy"] = np.sign(out["feat_return_15m"].fillna(0)) * -out["feat_approach_15m_volume_slope"]
    out["feat_acceleration_then_deceleration"] = (out["feat_approach_30m_acceleration_ticks_per_minute2"] < 0) & (out["feat_approach_30m_speed_ticks_per_minute"] > 0)
    out["feat_compression_after_displacement"] = out["feat_approach_15m_range_compression_ratio"] < 0.8


def build_feature_registry(config: PoiContextConfig | None = None) -> pd.DataFrame:
    """Return the formal Section 7 predictor registry."""

    cfg = config or PoiContextConfig()
    rows: list[dict[str, Any]] = []

    def add(name: str, family: str, definition: str, source: str, lookback: str,
            normalization: str, availability: str, entries: str, missing: str,
            interpretation: str, risk: str = "PASS") -> None:
        rows.append({
            "feature_name": name,
            "feature_family": family,
            "precise_mathematical_definition": definition,
            "source_columns": source,
            "lookback_window": lookback,
            "normalization": normalization,
            "availability_timestamp": availability,
            "entry_models_allowed_to_use_it": entries,
            "missing_value_treatment": missing,
            "expected_interpretation": interpretation,
            "lookahead_risk_status": risk,
        })

    formation = "Formation-time feature"
    pretouch = "Pre-touch feature"
    touch = "Touch-close confirmation feature"
    for name, definition in {
        "feat_poi_width_ticks": "(POI_high-POI_low)/0.10",
        "feat_poi_width_atr": "POI width / ATR60 at availability",
        "feat_fvg_size_ticks": "classic wick gap in integer GC ticks",
        "feat_fvg_size_atr": "FVG points / ATR60 at availability",
        "feat_opening_gap_ticks": "abs(open[C]-close[B])/0.10",
        "feat_opening_gap_atr": "opening gap points / ATR60",
        "feat_fvg_to_poi_width_ratio": "FVG ticks / POI width ticks",
        "feat_opening_gap_to_poi_width_ratio": "opening-gap ticks / POI width ticks",
        "feat_middle_body_to_range_ratio": "abs(close[B]-open[B])/(high[B]-low[B])",
        "feat_confirmation_body_to_range_ratio": "abs(close[C]-open[C])/(high[C]-low[C])",
        "feat_confirmation_directional_close_location": "CLV oriented to POI direction",
        "feat_distance_beyond_broken_swing_ticks": "available structural break distance in ticks",
        "feat_structural_swing_validation": "any active research variant broke 15/21/31-bar structure",
        "feat_largest_structural_window_broken": "maximum active confirmed structural window",
        "feat_15bar_structural_validation": "largest active structural window equals 15",
        "feat_local_swing_only": "no active 15/21/31-bar structural break",
    }.items():
        add(name, "formation_geometry", definition, "refined signal + A/B/C bars", "A/B/C through active break", "ticks/ATR/ratio", formation, "all", "NaN retained; zero denominators -> NaN", "formation/location quality")

    displacement_names = [
        "duration_bars", "net_ticks", "net_atr", "gross_ticks", "efficiency",
        "cumulative_range_ticks", "average_range_ticks", "maximum_range_ticks",
        "average_body_ticks", "directional_body_share", "directional_candle_count",
        "opposing_candle_count", "maximum_internal_pullback_ticks",
        "maximum_internal_pullback_fraction", "volume", "relative_volume",
        "peak_volume", "volume_concentration", "range_expansion", "body_expansion",
        "acceleration_ticks_per_bar2", "imbalance_density", "speed_ticks_per_minute",
    ]
    for suffix in displacement_names:
        add(f"feat_displacement_{suffix}", "displacement", "direction-neutral statistic over maximal active displacement interval", "OHLCV; displacement_start; activation", "displacement start through availability", "ticks/ATR/ratio", formation, "all", "NaN when interval crosses segment/date or denominator is zero", "displacement quality")

    for window in cfg.approach_windows:
        for suffix in (
            "origin_distance_to_poi_ticks", "duration_minutes", "speed_ticks_per_minute",
            "speed_atr", "net_return", "net_ticks_toward_poi", "gross_path_ticks",
            "efficiency_ratio", "slope_ticks_per_minute", "acceleration_ticks_per_minute2",
            "consecutive_candles_toward_poi", "opposing_candle_count",
            "maximum_countermove_ticks", "average_body_ticks", "average_range_ticks",
            "body_to_range_ratio", "candle_overlap_ratio", "range_compression_ratio",
            "range_expansion_ratio", "directional_efficiency", "volume",
            "relative_volume", "volume_slope", "volume_acceleration", "volume_spike_flag",
        ):
            add(f"feat_approach_{window}m_{suffix}", "approach", f"{suffix} over bars [touch-{window}, touch-1], oriented toward first-contact edge", "pre-touch OHLCV", f"{window} minutes", "ticks/ATR/ratio", pretouch, "boundary_touch; next_bar_confirmation", "NaN if history crosses date/segment", "attack, drift, chop, compression, or exhaustion")

    for name in (
        "feat_touch_penetration_ticks", "feat_touch_penetration_fraction",
        "feat_touch_penetration_atr", "feat_touch_midpoint_reached",
        "feat_touch_distal_reached", "feat_touch_full_zone_traversal",
        "feat_touch_wick_through", "feat_touch_close_through",
        "feat_touch_close_inside_poi", "feat_touch_candle_direction",
        "feat_touch_body_ticks", "feat_touch_body_to_range_ratio",
        "feat_touch_upper_wick_ticks", "feat_touch_lower_wick_ticks",
        "feat_touch_directional_rejection_wick_ticks",
        "feat_touch_rejection_wick_to_body_ratio", "feat_touch_close_location",
        "feat_touch_relative_volume", "feat_touch_volume_zscore",
        "feat_touch_consecutive_bars_inside", "feat_touch_recent_bars_inside_30m",
    ):
        add(name, "interaction", "direction-neutral touch-candle statistic measured after its close", "touch OHLCV + POI geometry", "touch bar; trailing overlap history where named", "ticks/ATR/ratio", touch, "next_bar_confirmation only", "NaN on zero range/body", "penetration, rejection, acceptance, or traversal", "PASS only for next-bar entry")

    market_names = [
        "feat_price_minus_vwap_ticks", "feat_distance_from_vwap_atr",
        "feat_vwap_slope_5m_ticks_per_minute", "feat_vwap_slope_15m_ticks_per_minute",
        "feat_vwap_slope_30m_ticks_per_minute", "feat_recent_vwap_crosses_30m",
        "feat_return_5m", "feat_return_15m", "feat_return_30m", "feat_return_60m",
        "feat_trend_efficiency_30m", "feat_rolling_high_low_structure_ticks",
        "feat_higher_high_higher_low_state", "feat_lower_high_lower_low_state",
        "feat_short_term_atr_ticks", "feat_long_term_atr_ticks",
        "feat_short_to_long_volatility_ratio", "feat_realized_volatility_60m",
        "feat_realized_volatility_240m", "feat_volatility_acceleration",
        "feat_relative_volume", "feat_volume_zscore",
        "feat_time_of_day_adjusted_relative_volume", "feat_volume_spike_flag",
        "feat_minutes_since_execution_window_open", "feat_minutes_until_no_new_entry_cutoff",
        "feat_minutes_until_forced_exit", "feat_session_return_before_retest",
        "feat_session_range_before_retest_ticks", "feat_position_within_session_range",
        "feat_distance_from_session_open_atr", "feat_fraction_recent_expected_range_travelled",
        "feat_successive_body_decay_3m", "feat_successive_range_decay_3m",
        "feat_volume_price_divergence_proxy", "feat_acceleration_then_deceleration",
        "feat_compression_after_displacement",
    ]
    for name in market_names:
        add(name, "market_context", "past-only rolling/session statistic through touch-1", "GC OHLCV and historical time-of-day baseline", "named suffix or session-to-date", "ticks/ATR/return/ratio", pretouch, "all", "NaN when required history unavailable", "trend, VWAP, volatility, volume, time, extension or exhaustion")

    additional = {
        "feat_research_variant_count": ("formation_geometry", "number of distinct active poi_variant_id values for this True POI/retest", formation),
        "feat_smallest_swing_window": ("formation_geometry", "minimum active research swing window", formation),
        "feat_largest_swing_window": ("formation_geometry", "maximum active research swing window", formation),
        "feat_displacement_start_bar_id": ("displacement", "earliest displacement start among variants active before this retest", formation),
        "feat_formation_available_bar_id": ("formation_geometry", "latest activation bar among variants represented at this retest", formation),
        "feat_has_swing_3_variant": ("formation_geometry", "one when an active 3-bar research variant exists", formation),
        "feat_has_swing_5_variant": ("formation_geometry", "one when an active 5-bar research variant exists", formation),
        "feat_has_swing_7_variant": ("formation_geometry", "one when an active 7-bar research variant exists", formation),
        "feat_has_wick_break_variant": ("formation_geometry", "one when an active wick-break research variant exists", formation),
        "feat_has_close_break_variant": ("formation_geometry", "one when an active close-break research variant exists", formation),
        "feat_poi_case": ("formation_geometry", "refined Case 1 expanded or Case 2 standard label", formation),
        "feat_middle_body_ticks": ("formation_geometry", "abs(close[B]-open[B])/0.10", formation),
        "feat_confirmation_body_ticks": ("formation_geometry", "abs(close[C]-open[C])/0.10", formation),
        "feat_confirmation_close_location": ("formation_geometry", "(close[C]-low[C])/(high[C]-low[C])", formation),
        "feat_first_contact_edge_price": ("approach", "POI high/low edge first encountered from touch-1 close", pretouch),
        "feat_first_contact_edge": ("approach", "high_edge or low_edge label for first contact", pretouch),
        "feat_approach_direction_sign": ("approach", "+1 for upward approach and -1 for downward approach", pretouch),
        "feat_displacement_volume_to_approach_volume_ratio": ("approach", "displacement volume / 15-minute approach volume", pretouch),
        "feat_displacement_speed_to_approach_speed_ratio": ("approach", "absolute displacement speed / absolute 15-minute approach speed", pretouch),
        "feat_minutes_since_latest_large_move": ("approach", "touch-1 bar_id minus latest past-only large-move bar_id", pretouch),
        "feat_distance_travelled_since_activation_ticks": ("approach", "sum absolute close changes from active formation through touch-1 / 0.10", pretouch),
        "feat_first_touch": ("retest_state", "true_retest_number equals one", pretouch),
        "feat_previous_touch_count": ("retest_state", "max(true_retest_number-1,0)", pretouch),
        "feat_time_since_previous_touch_minutes": ("retest_state", "minutes from previous physical retest to current retest", pretouch),
        "feat_poi_age_minutes": ("retest_state", "minutes from active formation availability to retest", pretouch),
        "feat_execution_session": ("session_context", "London or New York execution-window label", pretouch),
        "feat_price_above_vwap": ("market_context", "close[touch-1] > session VWAP[touch-1]", pretouch),
        "feat_distance_from_vwap_ticks": ("market_context", "abs(close[touch-1]-VWAP[touch-1])/0.10", pretouch),
        "feat_poi_distance_from_vwap_at_activation_atr": ("market_context", "(POI midpoint-VWAP at availability)/ATR60", formation),
        "feat_change_in_vwap_distance_during_approach_ticks": ("market_context", "change in signed VWAP distance from touch-16 to touch-1 / 0.10", pretouch),
        "feat_slope_5m_ticks_per_minute": ("market_context", "endpoint close slope through touch-1 over 5 minutes", pretouch),
        "feat_slope_15m_ticks_per_minute": ("market_context", "endpoint close slope through touch-1 over 15 minutes", pretouch),
        "feat_slope_30m_ticks_per_minute": ("market_context", "endpoint close slope through touch-1 over 30 minutes", pretouch),
        "feat_slope_60m_ticks_per_minute": ("market_context", "endpoint close slope through touch-1 over 60 minutes", pretouch),
        "feat_trend_alignment_continuation": ("market_context", "sign of past 30-minute return equals True POI direction", pretouch),
        "feat_trend_alignment_reversal": ("market_context", "sign of past 30-minute return opposes True POI direction", pretouch),
        "feat_volatility_percentile_devfit_bucket": ("market_context", "low/normal/high/extreme from development-fitted RV60 quartile/90% edges", pretouch),
        "feat_raw_volume": ("market_context", "volume at touch-1", pretouch),
        "feat_minute_of_day": ("session_context", "New York hour*60+minute at retest", pretouch),
        "feat_day_of_week": ("session_context", "New York trading weekday", pretouch),
        "feat_touch_boundary_reached": ("interaction", "penetration from first-contact edge is nonnegative", touch),
        "feat_touch_volume_to_approach_volume_ratio": ("interaction", "touch volume divided by 15-minute approach average volume", touch),
    }
    for name, (family, definition, availability) in additional.items():
        entries = "next_bar_confirmation only" if availability == touch else "all"
        add(name, family, definition, "refined identity and past-only GC OHLCV", "definition-specific", "ticks/ATR/ratio/label", availability, entries, "NaN retained when required history or denominator is unavailable", "identity, timing, approach, interaction, or context diagnostic")
    return pd.DataFrame(rows).drop_duplicates("feature_name").sort_values(["feature_family", "feature_name"]).reset_index(drop=True)


def save_feature_registry(path: str | Path, config: PoiContextConfig | None = None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_feature_registry(config).to_csv(output, index=False)
    return output


def validate_entry_feature_availability(
    feature_names: Iterable[str], entry_model: str, registry: pd.DataFrame
) -> None:
    """Reject touch-close predictors from a same-bar boundary entry model."""

    selected = registry.loc[registry["feature_name"].isin(feature_names)]
    missing = sorted(set(feature_names).difference(selected["feature_name"]))
    if missing:
        raise KeyError(f"features absent from registry: {missing}")
    if entry_model == "boundary_touch":
        leaking = selected.loc[
            selected["availability_timestamp"].eq("Touch-close confirmation feature"),
            "feature_name",
        ].tolist()
        if leaking:
            raise ValueError(f"touch-close features cannot justify boundary_touch entry: {leaking}")


def _add_partition(out: pd.DataFrame, cfg: PoiContextConfig) -> None:
    date = pd.to_datetime(out["trade_date_ny"])
    out["research_partition"] = pd.Categorical(
        np.select(
            [date <= pd.Timestamp("2023-12-31"), date <= pd.Timestamp("2024-12-31")],
            ["development", "validation"],
            default="final_test",
        ),
        categories=["development", "validation", "final_test"],
        ordered=True,
    )


def _validate_feature_timing(out: pd.DataFrame) -> None:
    formation = out["diag_formation_availability_bar_id"].to_numpy("int64")
    pretouch = out["diag_pretouch_availability_bar_id"].to_numpy("int64")
    touch = out["diag_touch_close_availability_bar_id"].to_numpy("int64")
    retest = out["retest_bar_id"].to_numpy("int64")
    if not ((formation < retest).all() and (pretouch < retest).all() and (touch == retest).all()):
        raise ValueError("feature availability timestamps violate the no-lookahead contract")


def _bar_arrays(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    # These derived arrays are past-only. VWAP resets by New York date; the
    # time-of-day volume denominator uses only earlier observations.
    volume = bars["volume"].to_numpy("float64", copy=False)
    typical = bars[["high", "low", "close"]].mean(axis=1).to_numpy("float64")
    date_codes, _ = pd.factorize(bars["trade_date_ny"], sort=False)
    temp = pd.DataFrame({"date": date_codes, "pv": typical * volume, "volume": volume})
    cum_pv = temp.groupby("date", sort=False, observed=True)["pv"].cumsum().to_numpy()
    cum_vol = temp.groupby("date", sort=False, observed=True)["volume"].cumsum().to_numpy()
    vwap = _safe_divide(cum_pv, cum_vol)
    minute = bars["minute_of_day_ny"].to_numpy("int32", copy=False)
    tod = pd.DataFrame({"minute": minute, "volume": volume})
    count = tod.groupby("minute", sort=False, observed=True).cumcount().to_numpy("int64")
    csum = tod.groupby("minute", sort=False, observed=True)["volume"].cumsum().to_numpy()
    tod_mean = _safe_divide(csum - volume, count)
    global_prior = np.r_[np.nan, np.cumsum(volume)[:-1] / np.arange(1, len(volume))]
    tod_mean = np.where(np.isfinite(tod_mean), tod_mean, global_prior)
    close = bars["close"].to_numpy("float64", copy=False)
    segment = bars["continuous_segment_id"].to_numpy("int64", copy=False)
    change = np.r_[0.0, np.diff(close)]
    change[1:][segment[1:] != segment[:-1]] = 0.0
    return {
        "ts": bars["ts_event_utc"].to_numpy(),
        "date": date_codes,
        "segment": segment,
        "minute": minute,
        "open": bars["open"].to_numpy("float64", copy=False),
        "high": bars["high"].to_numpy("float64", copy=False),
        "low": bars["low"].to_numpy("float64", copy=False),
        "close": close,
        "volume": volume,
        "range": bars["bar_range"].to_numpy("float64", copy=False),
        "body": bars["candle_body"].to_numpy("float64", copy=False),
        "upper": bars["upper_wick"].to_numpy("float64", copy=False),
        "lower": bars["lower_wick"].to_numpy("float64", copy=False),
        "atr20": bars["rolling_atr_20m"].to_numpy("float64", copy=False),
        "atr60": bars["rolling_atr_60m"].to_numpy("float64", copy=False),
        "rv60": bars["rolling_realized_vol_60m"].to_numpy("float64", copy=False),
        "rv240": bars["rolling_realized_vol_240m"].to_numpy("float64", copy=False),
        "rel_volume": bars["relative_volume_60m"].to_numpy("float64", copy=False),
        "volume_z": bars["volume_zscore_240m"].to_numpy("float64", copy=False),
        "abs_return": bars["abs_log_return"].to_numpy("float64", copy=False),
        "abs_change": np.abs(change),
        "vwap": vwap,
        "volume_tod_mean": tod_mean,
    }


def _session_state(arrays: dict[str, np.ndarray], key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"key": key, "open": arrays["open"], "high": arrays["high"], "low": arrays["low"]})
    session_open = frame.groupby("key", sort=False, observed=True)["open"].transform("first").to_numpy()
    session_high = frame.groupby("key", sort=False, observed=True)["high"].cummax().to_numpy()
    session_low = frame.groupby("key", sort=False, observed=True)["low"].cummin().to_numpy()
    return session_open, session_high, session_low


def _rolling_extreme(values: np.ndarray, window: int, op: str) -> np.ndarray:
    series = pd.Series(values)
    if op == "max":
        return series.rolling(window, min_periods=1).max().to_numpy()
    return series.rolling(window, min_periods=1).min().to_numpy()


def _safe_divide(numerator: Any, denominator: Any) -> np.ndarray:
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    shape = np.broadcast_shapes(num.shape, den.shape)
    out = np.full(shape, np.nan, dtype="float64")
    np.divide(num, den, out=out, where=np.isfinite(den) & (den != 0))
    return out


def _safe_scalar(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and denominator != 0 else np.nan


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def refit_development_volatility_bucket(
    context: pd.DataFrame, config: PoiContextConfig | None = None
) -> pd.DataFrame:
    """Refit the development-fitted volatility bucket on a full context frame.

    Chunked pipeline runs build the context frame in chronological pieces, so
    the development-quantile thresholds inside each piece see only that
    piece's events.  Applying this refit to the concatenated frame reproduces
    exactly the thresholds and bucket labels a single-pass build produces.
    """

    cfg = config or PoiContextConfig()
    out = context
    dev_mask = pd.to_datetime(out["trade_date_ny"]) <= pd.Timestamp(cfg.development_end)
    fit_values = pd.to_numeric(
        out.loc[dev_mask, "feat_realized_volatility_60m"], errors="coerce"
    ).dropna()
    if fit_values.empty:
        fit_values = pd.to_numeric(out["feat_realized_volatility_60m"], errors="coerce").dropna()
    thresholds = (
        np.asarray(fit_values.quantile([0.25, 0.75, 0.90]).to_numpy(), dtype="float64")
        if not fit_values.empty
        else np.asarray([0.0, 0.0, 0.0], dtype="float64")
    )
    vol_values = out["feat_realized_volatility_60m"].to_numpy("float64")
    out["feat_volatility_percentile_devfit_bucket"] = pd.Categorical(
        np.select(
            [
                vol_values <= thresholds[0],
                vol_values <= thresholds[1],
                vol_values <= thresholds[2],
            ],
            ["low", "normal", "high"],
            default="extreme",
        ),
        categories=["low", "normal", "high", "extreme"],
        ordered=True,
    )
    out["diag_volatility_devfit_q25"] = thresholds[0]
    out["diag_volatility_devfit_q75"] = thresholds[1]
    out["diag_volatility_devfit_q90"] = thresholds[2]
    return out
