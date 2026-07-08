"""POI feature engineering for the GC/MGC research workflow.

This module implements Section 6 of the exploration notebook.  It converts
the Section 5 discretionary POI definitions into machine-readable research
tables without performing a full trade backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class Section6Config:
    """Configuration for the first-pass POI signal table builder."""

    tick_size: float = 0.10
    poi_search_start_minute: int = 60
    poi_search_end_minute: int = 720
    london_start_minute: int = 180
    london_end_minute: int = 360
    ny_start_minute: int = 420
    ny_end_minute: int = 720
    swing_windows: tuple[int, ...] = (3, 5, 7)
    break_modes: tuple[str, ...] = ("wick", "close")
    min_fvg_ticks: float = 1.0
    min_close_open_gap_ticks: float = 1.0
    stop_buffer_ticks: float = 1.0
    min_stop_ticks: float = 25.0
    max_stop_ticks: float = 100.0
    target_r_multiples: tuple[int, ...] = (1, 2, 3, 4, 5)
    forward_horizons: tuple[int, ...] = (5, 15, 30, 60)
    structural_swing_windows: tuple[int, ...] = (15, 21, 31)


SECTION6_REQUIRED_COLUMNS = [
    "ts_event_utc",
    "product",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ts_event_ny",
    "trade_date_ny",
    "day_of_week",
    "hour_ny",
    "minute_ny",
    "minute_of_day_ny",
    "session_label",
    "tradable_research_flag",
    "roll_window_flag",
    "roll_window_type",
    "is_roll_day",
    "is_pre_roll_day",
    "is_post_roll_day",
    "days_since_roll",
    "days_to_next_roll",
    "selected_contract_volume_share",
    "ambiguous_roll_candidate",
    "low_liquidity_active_day_flag",
    "previous_symbol",
    "previous_close",
    "previous_ts_event_utc",
    "bar_gap_minutes",
    "same_symbol_as_previous_bar",
    "is_regular_1m_bar",
    "return_1m",
    "log_return",
    "abs_log_return",
    "bar_range",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "true_range",
    "continuous_segment_id",
    "rolling_atr_20m",
    "rolling_atr_60m",
    "rolling_realized_vol_60m",
    "rolling_realized_vol_240m",
    "rolling_high_low_range_60m",
    "rolling_volume_60m",
    "rolling_avg_volume_60m",
    "rolling_avg_volume_240m",
    "rolling_std_volume_240m",
    "relative_volume_60m",
    "volume_zscore_240m",
    "low_liquidity_warning_flag",
    "forward_return_5m",
    "forward_return_15m",
    "forward_return_30m",
    "forward_return_60m",
    "forward_log_return_5m",
    "forward_log_return_15m",
    "forward_log_return_30m",
    "forward_log_return_60m",
]


def build_section6_signal_tables(
    research_bars: pd.DataFrame,
    config: Section6Config | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Build Section 6 POI, retest, candidate, and signal tables.

    The first implementation validates GC-only signal logic.  MGC execution
    mapping is intentionally deferred until Section 7+ evidence identifies
    which candidate variants deserve trade simulation.
    """

    cfg = config or Section6Config()
    _validate_input_columns(research_bars, SECTION6_REQUIRED_COLUMNS)

    feature_frame = prepare_section6_feature_frame(research_bars, cfg)
    poi_table = build_poi_table(feature_frame, cfg)
    retest_table = build_retest_table(feature_frame, poi_table, cfg)
    candidate_trade_table = build_candidate_trade_table(poi_table, retest_table, cfg)
    signal_frame = build_signal_frame(
        poi_table=poi_table,
        retest_table=retest_table,
        candidate_trade_table=candidate_trade_table,
    )

    validation = build_section6_validation(
        poi_table=poi_table,
        retest_table=retest_table,
        candidate_trade_table=candidate_trade_table,
        signal_frame=signal_frame,
        config=cfg,
    )
    summary = build_section6_summary(
        feature_frame=feature_frame,
        poi_table=poi_table,
        retest_table=retest_table,
        candidate_trade_table=candidate_trade_table,
        signal_frame=signal_frame,
    )

    return {
        "section6_feature_frame": feature_frame,
        "poi_table": poi_table,
        "retest_table": retest_table,
        "candidate_trade_table": candidate_trade_table,
        "signal_frame": signal_frame,
        "section6_validation": validation,
        "section6_summary": summary,
    }


def prepare_section6_feature_frame(
    research_bars: pd.DataFrame,
    config: Section6Config | None = None,
) -> pd.DataFrame:
    """Create the GC feature base used by the POI engine."""

    cfg = config or Section6Config()
    cols = [c for c in SECTION6_REQUIRED_COLUMNS if c in research_bars.columns]
    gc = (
        research_bars.loc[research_bars["product"].eq("GC"), cols]
        .sort_values(["trade_date_ny", "ts_event_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    gc["bar_id"] = np.arange(len(gc), dtype=np.int32)

    minute = gc["minute_of_day_ny"]
    gc["is_poi_search_window"] = minute.between(
        cfg.poi_search_start_minute,
        cfg.poi_search_end_minute,
        inclusive="both",
    )
    gc["is_execution_window"] = (
        minute.between(cfg.london_start_minute, cfg.london_end_minute - 1, inclusive="both")
        | minute.between(cfg.ny_start_minute, cfg.ny_end_minute, inclusive="both")
    )
    gc["execution_window_label"] = np.select(
        [
            minute.between(cfg.london_start_minute, cfg.london_end_minute - 1, inclusive="both"),
            minute.between(cfg.ny_start_minute, cfg.ny_end_minute, inclusive="both"),
        ],
        ["London Execution", "New York Execution"],
        default="Outside Execution",
    )

    volume_float = gc["volume"].astype("float64")
    typical_price = (gc["high"] + gc["low"] + gc["close"]) / 3.0
    price_volume = typical_price * volume_float
    day_key = gc["trade_date_ny"]
    day_group = gc.groupby(day_key, observed=True, sort=False)

    cumulative_volume = volume_float.groupby(day_key, observed=True, sort=False).cumsum()
    gc["day_vwap"] = (
        price_volume.groupby(day_key, observed=True, sort=False).cumsum()
        / cumulative_volume.replace(0, np.nan)
    )
    gc["distance_from_vwap_ticks"] = (gc["close"] - gc["day_vwap"]) / cfg.tick_size
    gc["vwap_slope_15m_ticks"] = day_group["day_vwap"].diff(15) / cfg.tick_size
    gc["day_open"] = day_group["open"].transform("first")
    gc["day_high_so_far"] = day_group["high"].cummax()
    gc["day_low_so_far"] = day_group["low"].cummin()
    gc["session_high_so_far"] = gc.groupby(
        ["trade_date_ny", "session_label"],
        observed=True,
        sort=False,
    )["high"].cummax()
    gc["session_low_so_far"] = gc.groupby(
        ["trade_date_ny", "session_label"],
        observed=True,
        sort=False,
    )["low"].cummin()
    gc["distance_from_day_open_ticks"] = (gc["close"] - gc["day_open"]) / cfg.tick_size
    gc["distance_to_day_high_ticks"] = (gc["day_high_so_far"] - gc["close"]) / cfg.tick_size
    gc["distance_to_day_low_ticks"] = (gc["close"] - gc["day_low_so_far"]) / cfg.tick_size

    daily_levels = (
        gc.groupby("trade_date_ny", observed=True, sort=False)
        .agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"))
        .sort_index()
    )
    daily_levels[["previous_day_high", "previous_day_low", "previous_day_close"]] = (
        daily_levels[["day_high", "day_low", "day_close"]].shift(1)
    )
    gc = gc.merge(
        daily_levels[["previous_day_high", "previous_day_low", "previous_day_close"]],
        left_on="trade_date_ny",
        right_index=True,
        how="left",
        sort=False,
    )
    gc["distance_to_previous_day_high_ticks"] = (
        gc["previous_day_high"] - gc["close"]
    ) / cfg.tick_size
    gc["distance_to_previous_day_low_ticks"] = (
        gc["close"] - gc["previous_day_low"]
    ) / cfg.tick_size

    segment_group = gc.groupby("continuous_segment_id", observed=True, sort=False)
    gc["sma_20"] = segment_group["close"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    gc["sma_50"] = segment_group["close"].transform(
        lambda s: s.rolling(50, min_periods=50).mean()
    )
    gc["trend_bias_20_50"] = np.select(
        [gc["sma_20"].gt(gc["sma_50"]), gc["sma_20"].lt(gc["sma_50"])],
        ["bullish", "bearish"],
        default="neutral",
    )
    gc["trend_distance_ticks"] = (gc["sma_20"] - gc["sma_50"]) / cfg.tick_size

    close_diff = segment_group["close"].diff()
    rolling_abs_move = close_diff.abs().groupby(
        gc["continuous_segment_id"],
        observed=True,
        sort=False,
    ).transform(lambda s: s.rolling(20, min_periods=20).sum())
    rolling_net_move = segment_group["close"].diff(20).abs()
    gc["directional_efficiency_20m"] = rolling_net_move / rolling_abs_move.replace(0, np.nan)
    gc["large_move_flag"] = gc["bar_range"].ge(gc["rolling_atr_60m"] * 2.0)
    gc["volume_spike_flag"] = (
        gc["volume_zscore_240m"].ge(2.0) | gc["relative_volume_60m"].ge(2.0)
    )

    vol_source = gc["rolling_realized_vol_60m"].replace([np.inf, -np.inf], np.nan)
    q = vol_source.quantile([0.25, 0.50, 0.75]).to_numpy()
    if np.isfinite(q).all() and q[0] < q[1] < q[2]:
        gc["volatility_regime"] = np.select(
            [
                vol_source.le(q[0]),
                vol_source.le(q[1]),
                vol_source.le(q[2]),
                vol_source.gt(q[2]),
            ],
            ["low", "normal", "elevated", "extreme"],
            default="unknown",
        )
    else:
        gc["volatility_regime"] = "unknown"

    return gc


def build_poi_table(gc: pd.DataFrame, config: Section6Config | None = None) -> pd.DataFrame:
    """Detect strict FVG plus close-open gap POIs inside swing-break legs."""

    cfg = config or Section6Config()
    records: list[dict[str, object]] = []

    for trade_date, day in gc.groupby("trade_date_ny", observed=True, sort=False):
        n = len(day)
        if n < max(cfg.swing_windows, default=3) + 4:
            continue

        high = day["high"].to_numpy("float64")
        low = day["low"].to_numpy("float64")
        open_ = day["open"].to_numpy("float64")
        close = day["close"].to_numpy("float64")
        volume = day["volume"].to_numpy("float64")
        minute = day["minute_of_day_ny"].to_numpy("int32")
        bar_ids = day["bar_id"].to_numpy("int32")
        ts_utc = day["ts_event_utc"].to_numpy()
        ts_ny = day["ts_event_ny"].to_numpy()
        segment = day["continuous_segment_id"].to_numpy("int32")
        tradable = day["tradable_research_flag"].to_numpy(bool)
        roll_flag = day["roll_window_flag"].to_numpy(bool)
        regular = day["is_regular_1m_bar"].to_numpy(bool)
        search_window = day["is_poi_search_window"].to_numpy(bool)
        feature_arrays = _feature_arrays(day)

        bull_poi_base, bear_poi_base, diagnostics = _precompute_poi_masks(
            high=high,
            low=low,
            open_=open_,
            close=close,
            segment=segment,
            regular=regular,
            cfg=cfg,
        )

        for swing_n in cfg.swing_windows:
            swing_high, swing_low = _confirmed_swings(high, low, swing_n)
            swing_state = _build_prior_swing_state(high, low, swing_high, swing_low, swing_n)

            for break_mode in cfg.break_modes:
                if break_mode not in {"wick", "close"}:
                    raise ValueError(f"Unsupported break_mode: {break_mode!r}")
                bull_break_price = high if break_mode == "wick" else close
                bear_break_price = low if break_mode == "wick" else close

                bull_break_idxs = np.flatnonzero(
                    (swing_state["prior_high_swing_idx"] >= 0)
                    & (swing_state["prior_low_swing_idx"] >= 0)
                    & (bull_break_price > swing_state["prior_high_level"])
                )
                bear_break_idxs = np.flatnonzero(
                    (swing_state["prior_low_swing_idx"] >= 0)
                    & (swing_state["prior_high_swing_idx"] >= 0)
                    & (bear_break_price < swing_state["prior_low_level"])
                )

                _append_directional_pois(
                    records=records,
                    direction="bullish",
                    trade_date=trade_date,
                    day=day,
                    break_mode=break_mode,
                    swing_n=swing_n,
                    break_idxs=bull_break_idxs,
                    poi_base=bull_poi_base,
                    fvg_ticks=diagnostics["bull_fvg_ticks"],
                    gap_ticks=diagnostics["bull_gap_ticks"],
                    break_price=bull_break_price,
                    broken_swing_levels=swing_state["prior_high_level"],
                    opposite_swing_levels=swing_state["prior_low_level"],
                    broken_swing_idxs=swing_state["prior_high_swing_idx"],
                    opposite_swing_idxs=swing_state["prior_low_swing_idx"],
                    high=high,
                    low=low,
                    volume=volume,
                    minute=minute,
                    bar_ids=bar_ids,
                    ts_utc=ts_utc,
                    ts_ny=ts_ny,
                    tradable=tradable,
                    roll_flag=roll_flag,
                    search_window=search_window,
                    feature_arrays=feature_arrays,
                    cfg=cfg,
                )
                _append_directional_pois(
                    records=records,
                    direction="bearish",
                    trade_date=trade_date,
                    day=day,
                    break_mode=break_mode,
                    swing_n=swing_n,
                    break_idxs=bear_break_idxs,
                    poi_base=bear_poi_base,
                    fvg_ticks=diagnostics["bear_fvg_ticks"],
                    gap_ticks=diagnostics["bear_gap_ticks"],
                    break_price=bear_break_price,
                    broken_swing_levels=swing_state["prior_low_level"],
                    opposite_swing_levels=swing_state["prior_high_level"],
                    broken_swing_idxs=swing_state["prior_low_swing_idx"],
                    opposite_swing_idxs=swing_state["prior_high_swing_idx"],
                    high=high,
                    low=low,
                    volume=volume,
                    minute=minute,
                    bar_ids=bar_ids,
                    ts_utc=ts_utc,
                    ts_ny=ts_ny,
                    tradable=tradable,
                    roll_flag=roll_flag,
                    search_window=search_window,
                    feature_arrays=feature_arrays,
                    cfg=cfg,
                )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out

    out = out.sort_values(
        [
            "trade_date_ny",
            "swing_n",
            "break_mode",
            "direction",
            "poi_activation_bar_id",
            "break_bar_id",
        ],
        kind="mergesort",
    )
    out = out.drop_duplicates(
        ["trade_date_ny", "swing_n", "break_mode", "direction", "poi_middle_bar_id"],
        keep="first",
    ).reset_index(drop=True)
    out["poi_sequence"] = (
        out.groupby(["trade_date_ny", "swing_n", "break_mode", "direction"], observed=True)
        .cumcount()
        .add(1)
    )
    date_str = pd.to_datetime(out["trade_date_ny"]).dt.strftime("%Y%m%d")
    direction_code = np.where(out["direction"].eq("bullish"), "bull", "bear")
    out["poi_id"] = (
        "GC_"
        + date_str
        + "_sw"
        + out["swing_n"].astype(str)
        + "_"
        + out["break_mode"].astype(str)
        + "_"
        + direction_code
        + "_"
        + out["poi_sequence"].astype(str).str.zfill(3)
    )
    first_cols = ["poi_id", "poi_sequence"]
    return out[first_cols + [c for c in out.columns if c not in set(first_cols)]]


def build_retest_table(
    gc: pd.DataFrame,
    poi_table: pd.DataFrame,
    config: Section6Config | None = None,
) -> pd.DataFrame:
    """Record same-day POI retests inside the permitted execution windows."""

    cfg = config or Section6Config()
    if poi_table.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    day_lookup = {
        trade_date: day.reset_index(drop=True)
        for trade_date, day in gc.groupby("trade_date_ny", observed=True, sort=False)
    }

    for trade_date, day_pois in poi_table.groupby("trade_date_ny", observed=True, sort=False):
        day = day_lookup.get(trade_date)
        if day is None or day.empty:
            continue

        high = day["high"].to_numpy("float64")
        low = day["low"].to_numpy("float64")
        close = day["close"].to_numpy("float64")
        minute = day["minute_of_day_ny"].to_numpy("int32")
        bar_ids = day["bar_id"].to_numpy("int32")
        ts_utc = day["ts_event_utc"].to_numpy()
        ts_ny = day["ts_event_ny"].to_numpy()
        tradable = day["tradable_research_flag"].to_numpy(bool)
        roll_flag = day["roll_window_flag"].to_numpy(bool)
        execution = day["is_execution_window"].to_numpy(bool)
        open_ = day["open"].to_numpy("float64")
        volume = day["volume"].to_numpy("float64")
        execution_labels = day["execution_window_label"].to_numpy()
        feature_arrays = _feature_arrays(day)

        exec_pos = np.flatnonzero(
            execution & tradable & ~roll_flag & (minute <= cfg.poi_search_end_minute)
        )
        path_pos = np.flatnonzero(tradable & ~roll_flag & (minute <= cfg.poi_search_end_minute))
        if exec_pos.size == 0 or path_pos.size == 0:
            continue

        day_pois = day_pois.reset_index(drop=True)
        activation_pos_all = np.searchsorted(
            bar_ids,
            day_pois["poi_activation_bar_id"].to_numpy("int32"),
        )
        max_cells = 2_500_000
        chunk_size = max(1, max_cells // max(int(max(exec_pos.size, path_pos.size)), 1))

        for start in range(0, len(day_pois), chunk_size):
            stop = min(start + chunk_size, len(day_pois))
            poi_chunk = day_pois.iloc[start:stop].reset_index(drop=True)
            activation_pos = activation_pos_all[start:stop]
            poi_low = poi_chunk["poi_low"].to_numpy("float64")
            poi_high = poi_chunk["poi_high"].to_numpy("float64")

            overlap = (
                (exec_pos[None, :] > activation_pos[:, None])
                & (high[exec_pos][None, :] >= poi_low[:, None])
                & (low[exec_pos][None, :] <= poi_high[:, None])
            )
            poi_idx, exec_idx = np.where(overlap)
            if poi_idx.size == 0:
                continue

            retest_pos = exec_pos[exec_idx]
            selected_pois = poi_chunk.iloc[poi_idx].reset_index(drop=True)
            selected_activation_pos = activation_pos[poi_idx]

            path_after_activation = path_pos[None, :] > activation_pos[:, None]
            low_path = np.where(path_after_activation, low[path_pos][None, :], np.inf)
            high_path = np.where(path_after_activation, high[path_pos][None, :], -np.inf)
            close_min_path = np.where(path_after_activation, close[path_pos][None, :], np.inf)
            close_max_path = np.where(path_after_activation, close[path_pos][None, :], -np.inf)
            cum_low = np.minimum.accumulate(low_path, axis=1)
            cum_high = np.maximum.accumulate(high_path, axis=1)
            cum_close_min = np.minimum.accumulate(close_min_path, axis=1)
            cum_close_max = np.maximum.accumulate(close_max_path, axis=1)

            prior_col = np.searchsorted(path_pos, retest_pos, side="left") - 1
            has_prior = prior_col >= 0
            prior_low_min = np.full(retest_pos.size, np.inf)
            prior_high_max = np.full(retest_pos.size, -np.inf)
            prior_close_min = np.full(retest_pos.size, np.inf)
            prior_close_max = np.full(retest_pos.size, -np.inf)
            prior_low_min[has_prior] = cum_low[poi_idx[has_prior], prior_col[has_prior]]
            prior_high_max[has_prior] = cum_high[poi_idx[has_prior], prior_col[has_prior]]
            prior_close_min[has_prior] = cum_close_min[poi_idx[has_prior], prior_col[has_prior]]
            prior_close_max[has_prior] = cum_close_max[poi_idx[has_prior], prior_col[has_prior]]

            bullish = selected_pois["direction"].eq("bullish").to_numpy()
            selected_low = selected_pois["poi_low"].to_numpy("float64")
            selected_high = selected_pois["poi_high"].to_numpy("float64")
            distal_wick_violation = np.where(
                bullish,
                prior_low_min < selected_low,
                prior_high_max > selected_high,
            )
            distal_close_violation = np.where(
                bullish,
                prior_close_min < selected_low,
                prior_close_max > selected_high,
            )

            ts_utc_ns = day["ts_event_utc"].astype("int64").to_numpy()
            activation_ns = ts_utc_ns[selected_activation_pos]
            retest_ns = ts_utc_ns[retest_pos]

            frame = pd.DataFrame(
                {
                    "poi_id": selected_pois["poi_id"].to_numpy(),
                    "product": "GC",
                    "trade_date_ny": trade_date,
                    "swing_n": selected_pois["swing_n"].to_numpy(),
                    "break_mode": selected_pois["break_mode"].to_numpy(),
                    "direction": selected_pois["direction"].to_numpy(),
                    "retest_bar_id": bar_ids[retest_pos],
                    "retest_ts_event_utc": ts_utc[retest_pos],
                    "retest_ts_event_ny": ts_ny[retest_pos],
                    "retest_minute_ny": minute[retest_pos],
                    "execution_window_label": execution_labels[retest_pos],
                    "time_since_poi_activation_minutes": (retest_ns - activation_ns)
                    / 60_000_000_000.0,
                    "candles_since_poi_activation": retest_pos - selected_activation_pos,
                    "retest_open": open_[retest_pos],
                    "retest_high": high[retest_pos],
                    "retest_low": low[retest_pos],
                    "retest_close": close[retest_pos],
                    "retest_volume": volume[retest_pos],
                    "full_poi_cross_flag": (
                        (high[retest_pos] >= selected_high) & (low[retest_pos] <= selected_low)
                    ),
                    "distal_wick_violation_before_retest": distal_wick_violation,
                    "distal_close_violation_before_retest": distal_close_violation,
                }
            )
            for suffix, values in feature_arrays.items():
                frame[f"retest_{suffix}"] = values[retest_pos]
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["trade_date_ny", "swing_n", "break_mode", "direction", "poi_id", "retest_bar_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    out["retest_number"] = out.groupby("poi_id", observed=True).cumcount() + 1
    out["first_touch_flag"] = out["retest_number"].eq(1)
    out["time_since_previous_touch_minutes"] = (
        out.groupby("poi_id", observed=True)["retest_ts_event_utc"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )
    out["retest_id"] = (
        out["poi_id"] + "_rt" + out["retest_number"].astype(str).str.zfill(3)
    )
    first_cols = [
        "retest_id",
        "poi_id",
        "retest_number",
        "first_touch_flag",
    ]
    return out[first_cols + [c for c in out.columns if c not in set(first_cols)]]


def build_candidate_trade_table(
    poi_table: pd.DataFrame,
    retest_table: pd.DataFrame,
    config: Section6Config | None = None,
) -> pd.DataFrame:
    """Build continuation entry/stop/target variants from retest events."""

    cfg = config or Section6Config()
    if poi_table.empty or retest_table.empty:
        return pd.DataFrame()

    poi_cols = [
        "poi_id",
        "poi_high",
        "poi_low",
        "poi_mid",
        "prev_candle_high",
        "prev_candle_low",
        "next_candle_high",
        "next_candle_low",
    ]
    retest_cols = [
        "retest_id",
        "poi_id",
        "direction",
        "retest_bar_id",
        "retest_ts_event_utc",
        "retest_high",
        "retest_low",
        "retest_close",
        *[f"retest_forward_return_{horizon}m" for horizon in cfg.forward_horizons],
    ]
    base = retest_table[retest_cols].merge(
        poi_table[poi_cols],
        on="poi_id",
        how="left",
        validate="many_to_one",
    )

    valid_frames = []
    buffer = cfg.stop_buffer_ticks * cfg.tick_size
    for variant in ("boundary", "midpoint", "distal"):
        frame = base.copy()
        bullish = frame["direction"].eq("bullish")
        frame["entry_variant"] = variant
        if variant == "boundary":
            frame["entry_price"] = np.where(bullish, frame["poi_high"], frame["poi_low"])
        elif variant == "midpoint":
            frame["entry_price"] = frame["poi_mid"]
        else:
            frame["entry_price"] = np.where(bullish, frame["poi_low"], frame["poi_high"])
        frame["entry_touched_flag"] = (
            frame["retest_low"].le(frame["entry_price"])
            & frame["retest_high"].ge(frame["entry_price"])
        )
        entries = frame.loc[frame["entry_touched_flag"]].copy()
        if entries.empty:
            continue

        entries["trade_bias"] = "continuation"
        entries["trade_side"] = np.where(entries["direction"].eq("bullish"), "long", "short")
        entries["is_red_folder_news_window"] = False
        entries["news_event_name"] = pd.NA
        entries["minutes_to_news"] = np.nan
        entries["minutes_since_news"] = np.nan

        for stop_model in ("poi_distal_edge", "conservative_adjacent"):
            candidate = entries.copy()
            candidate["stop_model"] = stop_model
            long_side = candidate["trade_side"].eq("long")
            if stop_model == "poi_distal_edge":
                candidate["stop_price"] = np.where(
                    long_side,
                    candidate["poi_low"] - buffer,
                    candidate["poi_high"] + buffer,
                )
            else:
                long_stop = (
                    np.minimum.reduce(
                        [
                            candidate["prev_candle_low"].to_numpy(),
                            candidate["poi_low"].to_numpy(),
                            candidate["next_candle_low"].to_numpy(),
                        ]
                    )
                    - buffer
                )
                short_stop = (
                    np.maximum.reduce(
                        [
                            candidate["prev_candle_high"].to_numpy(),
                            candidate["poi_high"].to_numpy(),
                            candidate["next_candle_high"].to_numpy(),
                        ]
                    )
                    + buffer
                )
                candidate["stop_price"] = np.where(long_side, long_stop, short_stop)

            candidate["risk_points"] = np.where(
                long_side,
                candidate["entry_price"] - candidate["stop_price"],
                candidate["stop_price"] - candidate["entry_price"],
            )
            candidate["stop_ticks"] = candidate["risk_points"] / cfg.tick_size
            valid_stop = candidate["stop_ticks"].between(
                cfg.min_stop_ticks,
                cfg.max_stop_ticks,
                inclusive="both",
            )
            candidate = candidate.loc[valid_stop].copy()
            if candidate.empty:
                continue

            long_side = candidate["trade_side"].eq("long")
            candidate["valid_candidate_flag"] = True
            candidate["invalid_reason"] = "valid"

            for r_multiple in cfg.target_r_multiples:
                candidate[f"target_{r_multiple}R_price"] = np.where(
                    long_side,
                    candidate["entry_price"] + r_multiple * candidate["risk_points"],
                    candidate["entry_price"] - r_multiple * candidate["risk_points"],
                )

            risk_points = candidate["risk_points"].replace(0, np.nan)
            for horizon in cfg.forward_horizons:
                fwd_col = f"retest_forward_return_{horizon}m"
                if fwd_col not in candidate.columns:
                    fwd_col = f"forward_return_{horizon}m"
                future_close = candidate["retest_close"] * (1.0 + candidate[fwd_col])
                candidate[f"forward_{horizon}m_points_from_entry"] = np.where(
                    long_side,
                    future_close - candidate["entry_price"],
                    candidate["entry_price"] - future_close,
                )
                candidate[f"forward_{horizon}m_r"] = (
                    candidate[f"forward_{horizon}m_points_from_entry"] / risk_points
                )
                candidate[f"continuation_{horizon}m_positive_flag"] = candidate[
                    f"forward_{horizon}m_r"
                ].gt(0)
                candidate[f"reversal_{horizon}m_positive_flag"] = candidate[
                    f"forward_{horizon}m_r"
                ].lt(0)

            valid_frames.append(candidate)

    if not valid_frames:
        return pd.DataFrame()

    candidates = pd.concat(valid_frames, ignore_index=True)

    candidates = candidates.sort_values(
        ["retest_ts_event_utc", "poi_id", "entry_variant", "stop_model"],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates["candidate_trade_id"] = (
        "GC_CAND_" + (candidates.index + 1).astype(str).str.zfill(8)
    )
    first_cols = [
        "candidate_trade_id",
        "retest_id",
        "poi_id",
        "direction",
        "retest_bar_id",
        "retest_ts_event_utc",
        "trade_bias",
        "trade_side",
        "entry_variant",
        "stop_model",
        "entry_price",
        "stop_price",
        "risk_points",
        "stop_ticks",
        "valid_candidate_flag",
        "invalid_reason",
    ]
    audit_cols = [
        "entry_touched_flag",
        "is_red_folder_news_window",
        "news_event_name",
        "minutes_to_news",
        "minutes_since_news",
    ]
    target_cols = [f"target_{r_multiple}R_price" for r_multiple in cfg.target_r_multiples]
    forward_cols = []
    for horizon in cfg.forward_horizons:
        forward_cols.extend(
            [
                f"forward_{horizon}m_points_from_entry",
                f"forward_{horizon}m_r",
                f"continuation_{horizon}m_positive_flag",
                f"reversal_{horizon}m_positive_flag",
            ]
        )
    final_cols = first_cols + audit_cols + target_cols + forward_cols
    return candidates[[c for c in final_cols if c in candidates.columns]]


def build_signal_frame(
    poi_table: pd.DataFrame,
    retest_table: pd.DataFrame,
    candidate_trade_table: pd.DataFrame,
) -> pd.DataFrame:
    """Attach POI and retest context to valid candidate variants."""

    if candidate_trade_table.empty:
        return pd.DataFrame()

    signal = (
        candidate_trade_table.loc[candidate_trade_table["valid_candidate_flag"]]
        .copy()
        .reset_index(drop=True)
    )
    if signal.empty:
        return signal

    retest_context_cols = [
        "retest_id",
        "retest_number",
        "first_touch_flag",
        "trade_date_ny",
        "swing_n",
        "break_mode",
        "retest_ts_event_ny",
        "retest_minute_ny",
        "execution_window_label",
        "time_since_poi_activation_minutes",
        "candles_since_poi_activation",
        "time_since_previous_touch_minutes",
        "retest_open",
        "retest_high",
        "retest_low",
        "retest_close",
        "retest_volume",
        "full_poi_cross_flag",
        "distal_wick_violation_before_retest",
        "distal_close_violation_before_retest",
        "retest_trend_bias_20_50",
        "retest_distance_from_vwap_ticks",
        "retest_vwap_slope_15m_ticks",
        "retest_relative_volume_60m",
        "retest_volume_zscore_240m",
        "retest_rolling_atr_60m",
        "retest_volatility_regime",
        "retest_large_move_flag",
        "retest_volume_spike_flag",
        "retest_directional_efficiency_20m",
        "retest_distance_to_day_high_ticks",
        "retest_distance_to_day_low_ticks",
        "retest_distance_to_previous_day_high_ticks",
        "retest_distance_to_previous_day_low_ticks",
    ]
    poi_context_cols = [
        "poi_id",
        "poi_sequence",
        "poi_middle_bar_id",
        "poi_confirm_bar_id",
        "poi_activation_bar_id",
        "break_bar_id",
        "displacement_start_bar_id",
        "broken_swing_bar_id",
        "poi_middle_ts_event_utc",
        "poi_created_ts_event_utc",
        "poi_activation_ts_event_utc",
        "poi_middle_ts_event_ny",
        "poi_created_ts_event_ny",
        "poi_activation_ts_event_ny",
        "break_ts_event_utc",
        "break_ts_event_ny",
        "poi_activation_minute_ny",
        "poi_high",
        "poi_low",
        "poi_mid",
        "fvg_size_ticks",
        "close_open_gap_ticks",
        "abs_close_open_gap_ticks",
        "poi_size_ticks",
        "broken_swing_price",
        "opposite_swing_price",
        "break_distance_ticks",
        "displacement_range_ticks",
        "displacement_candles",
        "displacement_volume",
        "candles_from_break_to_poi_activation",
        "poi_trend_bias_20_50",
        "poi_distance_from_vwap_ticks",
        "poi_vwap_slope_15m_ticks",
        "poi_relative_volume_60m",
        "poi_volume_zscore_240m",
        "poi_rolling_atr_60m",
        "poi_volatility_regime",
        "poi_large_move_flag",
        "poi_volume_spike_flag",
        "poi_directional_efficiency_20m",
        "poi_distance_to_day_high_ticks",
        "poi_distance_to_day_low_ticks",
        "poi_distance_to_previous_day_high_ticks",
        "poi_distance_to_previous_day_low_ticks",
    ]
    retest_context_cols = [
        c
        for c in retest_context_cols
        if c in retest_table.columns and (c == "retest_id" or c not in signal.columns)
    ]
    poi_context_cols = [
        c
        for c in poi_context_cols
        if c in poi_table.columns and (c == "poi_id" or c not in signal.columns)
    ]

    signal = signal.merge(
        retest_table[retest_context_cols],
        on="retest_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_retest_context"),
    )
    signal = signal.merge(
        poi_table[poi_context_cols],
        on="poi_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_poi_context"),
    )
    signal["signal_id"] = "GC_SIG_" + (signal.index + 1).astype(str).str.zfill(8)
    first_cols = ["signal_id"] + [c for c in signal.columns if c != "signal_id"]
    return signal[first_cols]


def build_section6_validation(
    poi_table: pd.DataFrame,
    retest_table: pd.DataFrame,
    candidate_trade_table: pd.DataFrame,
    signal_frame: pd.DataFrame,
    config: Section6Config | None = None,
) -> pd.Series:
    """Return lightweight validation checks for Section 6 outputs."""

    cfg = config or Section6Config()
    return pd.Series(
        {
            "poi_ids_unique": poi_table["poi_id"].is_unique if not poi_table.empty else True,
            "retest_ids_unique": retest_table["retest_id"].is_unique
            if not retest_table.empty
            else True,
            "candidate_trade_ids_unique": candidate_trade_table["candidate_trade_id"].is_unique
            if not candidate_trade_table.empty
            else True,
            "poi_activation_in_search_window": poi_table["poi_activation_minute_ny"].between(
                cfg.poi_search_start_minute,
                cfg.poi_search_end_minute,
                inclusive="both",
            ).all()
            if not poi_table.empty
            else True,
            "retests_after_poi_activation": retest_table["candles_since_poi_activation"].gt(0).all()
            if not retest_table.empty
            else True,
            "retests_in_execution_window": retest_table["execution_window_label"]
            .isin(["London Execution", "New York Execution"])
            .all()
            if not retest_table.empty
            else True,
            "valid_signals_have_valid_stops": signal_frame["stop_ticks"].between(
                cfg.min_stop_ticks,
                cfg.max_stop_ticks,
                inclusive="both",
            ).all()
            if not signal_frame.empty
            else True,
        }
    )


def build_section6_summary(
    feature_frame: pd.DataFrame,
    poi_table: pd.DataFrame,
    retest_table: pd.DataFrame,
    candidate_trade_table: pd.DataFrame,
    signal_frame: pd.DataFrame,
) -> pd.Series:
    """Return headline Section 6 counts."""

    return pd.Series(
        {
            "gc_feature_rows": len(feature_frame),
            "gc_tradable_rows": int(feature_frame["tradable_research_flag"].sum()),
            "poi_count": len(poi_table),
            "poi_with_retest_count": retest_table["poi_id"].nunique()
            if not retest_table.empty
            else 0,
            "retest_count": len(retest_table),
            "candidate_trade_variant_count": len(candidate_trade_table),
            "valid_signal_count": len(signal_frame),
            "first_touch_retest_count": int(retest_table["first_touch_flag"].sum())
            if not retest_table.empty
            else 0,
        }
    )


def save_section6_tables(
    tables: dict[str, pd.DataFrame | pd.Series],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save the Section 6 research tables to parquet files."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "poi_table": out_dir / "section6_poi_table_gc.parquet",
        "retest_table": out_dir / "section6_retest_table_gc.parquet",
        "candidate_trade_table": out_dir / "section6_candidate_trade_table_gc.parquet",
        "signal_frame": out_dir / "section6_signal_frame_gc.parquet",
    }
    for key, path in output_paths.items():
        table = tables[key]
        if isinstance(table, pd.Series):
            table = table.to_frame("value")
        table.to_parquet(path, index=False)
    return output_paths


STRUCTURAL_COLUMNS = [
    "structural_swing_break_flag",
    "structural_swing_window_broken",
    "structural_swing_windows_broken",
    "structural_break_mode",
    "structural_swing_bar_id",
    "structural_swing_price",
    "structural_break_bar_id",
    "structural_break_distance_ticks",
    "local_swing_only_flag",
]


def build_section6b_structural_tables(
    feature_frame: pd.DataFrame,
    poi_table: pd.DataFrame,
    retest_table: pd.DataFrame,
    candidate_trade_table: pd.DataFrame,
    signal_frame: pd.DataFrame,
    config: Section6Config | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Build non-destructive Section 6B structural swing validation tables."""

    cfg = config or Section6Config()
    structural_break_events = build_structural_swing_break_events(feature_frame, cfg)
    structural_poi_table = annotate_poi_structural_validation(
        poi_table=poi_table,
        structural_break_events=structural_break_events,
        config=cfg,
    )
    structural_retest_table = _merge_structural_columns(retest_table, structural_poi_table)
    structural_candidate_trade_table = _merge_structural_columns(
        candidate_trade_table,
        structural_poi_table,
    )
    structural_signal_frame = _merge_structural_columns(signal_frame, structural_poi_table)

    validation = build_section6b_validation(
        poi_table=poi_table,
        retest_table=retest_table,
        candidate_trade_table=candidate_trade_table,
        signal_frame=signal_frame,
        structural_poi_table=structural_poi_table,
        structural_retest_table=structural_retest_table,
        structural_candidate_trade_table=structural_candidate_trade_table,
        structural_signal_frame=structural_signal_frame,
    )
    summary = build_section6b_summary(
        structural_poi_table=structural_poi_table,
        structural_retest_table=structural_retest_table,
        structural_candidate_trade_table=structural_candidate_trade_table,
        structural_signal_frame=structural_signal_frame,
    )

    return {
        "structural_break_events": structural_break_events,
        "structural_poi_table": structural_poi_table,
        "structural_retest_table": structural_retest_table,
        "structural_candidate_trade_table": structural_candidate_trade_table,
        "structural_signal_frame": structural_signal_frame,
        "section6b_validation": validation,
        "section6b_summary": summary,
    }


def build_structural_swing_break_events(
    feature_frame: pd.DataFrame,
    config: Section6Config | None = None,
) -> pd.DataFrame:
    """Detect first breaks of confirmed higher-order structural swings."""

    cfg = config or Section6Config()
    required = {
        "bar_id",
        "high",
        "low",
        "close",
        "continuous_segment_id",
        "trade_date_ny",
        "ts_event_utc",
        "ts_event_ny",
    }
    _validate_input_columns(feature_frame, required)

    records: list[pd.DataFrame] = []
    for segment_id, segment in feature_frame.groupby(
        "continuous_segment_id",
        observed=True,
        sort=False,
    ):
        segment = segment.sort_values("bar_id", kind="mergesort").reset_index(drop=True)
        n = len(segment)
        if n < min(cfg.structural_swing_windows, default=15):
            continue

        high = segment["high"].to_numpy("float64")
        low = segment["low"].to_numpy("float64")
        close = segment["close"].to_numpy("float64")
        bar_ids = segment["bar_id"].to_numpy("int32")

        for swing_window in cfg.structural_swing_windows:
            if n < swing_window:
                continue
            swing_high, swing_low = _confirmed_swings(high, low, swing_window)
            swing_state = _build_prior_swing_state(high, low, swing_high, swing_low, swing_window)

            for break_mode in cfg.break_modes:
                if break_mode not in {"wick", "close"}:
                    raise ValueError(f"Unsupported break_mode: {break_mode!r}")
                bullish_break_price = high if break_mode == "wick" else close
                bearish_break_price = low if break_mode == "wick" else close

                bullish_break_idx = np.flatnonzero(
                    (swing_state["prior_high_swing_idx"] >= 0)
                    & (bullish_break_price > swing_state["prior_high_level"])
                )
                bearish_break_idx = np.flatnonzero(
                    (swing_state["prior_low_swing_idx"] >= 0)
                    & (bearish_break_price < swing_state["prior_low_level"])
                )

                records.extend(
                    [
                        _structural_break_frame(
                            segment=segment,
                            segment_id=segment_id,
                            direction="bullish",
                            break_mode=break_mode,
                            swing_window=swing_window,
                            break_idx=bullish_break_idx,
                            break_price=bullish_break_price,
                            structural_level=swing_state["prior_high_level"],
                            structural_swing_idx=swing_state["prior_high_swing_idx"],
                            bar_ids=bar_ids,
                            cfg=cfg,
                        ),
                        _structural_break_frame(
                            segment=segment,
                            segment_id=segment_id,
                            direction="bearish",
                            break_mode=break_mode,
                            swing_window=swing_window,
                            break_idx=bearish_break_idx,
                            break_price=bearish_break_price,
                            structural_level=swing_state["prior_low_level"],
                            structural_swing_idx=swing_state["prior_low_swing_idx"],
                            bar_ids=bar_ids,
                            cfg=cfg,
                        ),
                    ]
                )

    frames = [frame for frame in records if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "continuous_segment_id",
                "direction",
                "structural_break_mode",
                "structural_swing_window_broken",
                "structural_swing_bar_id",
                "structural_swing_price",
                "structural_break_bar_id",
                "structural_break_distance_ticks",
                "structural_break_ts_event_utc",
                "structural_break_ts_event_ny",
                "structural_break_trade_date_ny",
            ]
        )

    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values(
        [
            "continuous_segment_id",
            "direction",
            "structural_break_mode",
            "structural_swing_window_broken",
            "structural_swing_bar_id",
            "structural_break_bar_id",
        ],
        kind="mergesort",
    )
    # A structural swing level is considered broken when it is first exceeded
    # for a given window/mode/direction. Later repeats are not new structure.
    events = events.drop_duplicates(
        [
            "continuous_segment_id",
            "direction",
            "structural_break_mode",
            "structural_swing_window_broken",
            "structural_swing_bar_id",
        ],
        keep="first",
    ).reset_index(drop=True)
    return events


def annotate_poi_structural_validation(
    poi_table: pd.DataFrame,
    structural_break_events: pd.DataFrame,
    config: Section6Config | None = None,
) -> pd.DataFrame:
    """Annotate Version A POIs with higher-order structural break metadata."""

    cfg = config or Section6Config()
    _validate_input_columns(
        poi_table,
        {
            "poi_id",
            "direction",
            "break_mode",
            "displacement_start_bar_id",
            "poi_activation_bar_id",
        },
    )
    out = poi_table.copy().reset_index(drop=True)
    n = len(out)

    matches_by_window: dict[int, np.ndarray] = {}
    primary_window = np.full(n, -1, dtype=np.int32)
    primary_break_bar = np.full(n, -1, dtype=np.int32)
    primary_swing_bar = np.full(n, -1, dtype=np.int32)
    primary_swing_price = np.full(n, np.nan)
    primary_break_distance = np.full(n, np.nan)
    primary_break_mode = np.full(n, None, dtype=object)

    events = structural_break_events
    for window in sorted(cfg.structural_swing_windows):
        window_match = np.zeros(n, dtype=bool)
        window_events = events.loc[
            events["structural_swing_window_broken"].eq(window)
        ]
        if window_events.empty:
            matches_by_window[window] = window_match
            continue

        for (direction, break_mode), poi_group in out.groupby(
            ["direction", "break_mode"],
            observed=True,
            sort=False,
        ):
            event_group = window_events.loc[
                window_events["direction"].eq(direction)
                & window_events["structural_break_mode"].eq(break_mode)
            ]
            if event_group.empty or poi_group.empty:
                continue

            event_group = event_group.sort_values("structural_break_bar_id", kind="mergesort")
            event_break_ids = event_group["structural_break_bar_id"].to_numpy("int32")
            poi_pos = poi_group.index.to_numpy()
            starts = out.loc[poi_pos, "displacement_start_bar_id"].to_numpy("int32")
            ends = out.loc[poi_pos, "poi_activation_bar_id"].to_numpy("int32")
            event_pos = np.searchsorted(event_break_ids, starts, side="left")
            has_event = event_pos < len(event_break_ids)
            if not has_event.any():
                continue

            candidate_pos = poi_pos[has_event]
            candidate_event_pos = event_pos[has_event]
            candidate_ends = ends[has_event]
            is_match = event_break_ids[candidate_event_pos] <= candidate_ends
            if not is_match.any():
                continue

            matched_poi_pos = candidate_pos[is_match]
            matched_event_pos = candidate_event_pos[is_match]
            matched_events = event_group.iloc[matched_event_pos]
            window_match[matched_poi_pos] = True

            replace_primary = window > primary_window[matched_poi_pos]
            if replace_primary.any():
                replace_pos = matched_poi_pos[replace_primary]
                replace_events = matched_events.iloc[np.flatnonzero(replace_primary)]
                primary_window[replace_pos] = window
                primary_break_bar[replace_pos] = replace_events[
                    "structural_break_bar_id"
                ].to_numpy("int32")
                primary_swing_bar[replace_pos] = replace_events[
                    "structural_swing_bar_id"
                ].to_numpy("int32")
                primary_swing_price[replace_pos] = replace_events[
                    "structural_swing_price"
                ].to_numpy("float64")
                primary_break_distance[replace_pos] = replace_events[
                    "structural_break_distance_ticks"
                ].to_numpy("float64")
                primary_break_mode[replace_pos] = replace_events[
                    "structural_break_mode"
                ].to_numpy(object)

        matches_by_window[window] = window_match

    structural_flag = primary_window >= 0
    out["structural_swing_break_flag"] = structural_flag
    out["structural_swing_window_broken"] = pd.Series(
        np.where(structural_flag, primary_window, pd.NA),
        index=out.index,
        dtype="Int64",
    )
    out["structural_swing_windows_broken"] = _format_structural_window_list(
        matches_by_window,
        n,
    )
    out["structural_break_mode"] = pd.Series(primary_break_mode, index=out.index).where(
        structural_flag,
        pd.NA,
    )
    out["structural_swing_bar_id"] = pd.Series(
        np.where(structural_flag, primary_swing_bar, pd.NA),
        index=out.index,
        dtype="Int64",
    )
    out["structural_swing_price"] = np.where(structural_flag, primary_swing_price, np.nan)
    out["structural_break_bar_id"] = pd.Series(
        np.where(structural_flag, primary_break_bar, pd.NA),
        index=out.index,
        dtype="Int64",
    )
    out["structural_break_distance_ticks"] = np.where(
        structural_flag,
        primary_break_distance,
        np.nan,
    )
    out["local_swing_only_flag"] = ~structural_flag
    _optimize_structural_dtypes(out)
    return out


def build_section6b_validation(
    *,
    poi_table: pd.DataFrame,
    retest_table: pd.DataFrame,
    candidate_trade_table: pd.DataFrame,
    signal_frame: pd.DataFrame,
    structural_poi_table: pd.DataFrame,
    structural_retest_table: pd.DataFrame,
    structural_candidate_trade_table: pd.DataFrame,
    structural_signal_frame: pd.DataFrame,
) -> pd.Series:
    """Return structural Version B validation checks."""

    required = set(STRUCTURAL_COLUMNS)
    validated = structural_poi_table["structural_swing_break_flag"]
    local_only = structural_poi_table["local_swing_only_flag"]

    return pd.Series(
        {
            "poi_row_count_preserved": len(structural_poi_table) == len(poi_table),
            "retest_row_count_preserved": len(structural_retest_table) == len(retest_table),
            "candidate_row_count_preserved": len(structural_candidate_trade_table)
            == len(candidate_trade_table),
            "signal_row_count_preserved": len(structural_signal_frame) == len(signal_frame),
            "structural_columns_present_poi": required.issubset(structural_poi_table.columns),
            "structural_columns_present_retest": required.issubset(
                structural_retest_table.columns
            ),
            "structural_columns_present_candidate": required.issubset(
                structural_candidate_trade_table.columns
            ),
            "structural_columns_present_signal": required.issubset(
                structural_signal_frame.columns
            ),
            "structural_partition_valid": bool((validated ^ local_only).all()),
            "validated_rows_have_structural_window": structural_poi_table.loc[
                validated,
                "structural_swing_window_broken",
            ]
            .notna()
            .all(),
            "local_only_rows_have_no_structural_window": structural_poi_table.loc[
                local_only,
                "structural_swing_window_broken",
            ]
            .isna()
            .all(),
        }
    )


def build_section6b_summary(
    *,
    structural_poi_table: pd.DataFrame,
    structural_retest_table: pd.DataFrame,
    structural_candidate_trade_table: pd.DataFrame,
    structural_signal_frame: pd.DataFrame,
) -> pd.Series:
    """Return headline Version A vs Version B structural counts."""

    return pd.Series(
        {
            "baseline_poi_count": len(structural_poi_table),
            "structurally_validated_poi_count": int(
                structural_poi_table["structural_swing_break_flag"].sum()
            ),
            "local_swing_only_poi_count": int(
                structural_poi_table["local_swing_only_flag"].sum()
            ),
            "baseline_retest_count": len(structural_retest_table),
            "structurally_validated_retest_count": int(
                structural_retest_table["structural_swing_break_flag"].sum()
            ),
            "local_swing_only_retest_count": int(
                structural_retest_table["local_swing_only_flag"].sum()
            ),
            "baseline_candidate_count": len(structural_candidate_trade_table),
            "structurally_validated_candidate_count": int(
                structural_candidate_trade_table["structural_swing_break_flag"].sum()
            ),
            "local_swing_only_candidate_count": int(
                structural_candidate_trade_table["local_swing_only_flag"].sum()
            ),
            "baseline_signal_count": len(structural_signal_frame),
            "structurally_validated_signal_count": int(
                structural_signal_frame["structural_swing_break_flag"].sum()
            ),
            "local_swing_only_signal_count": int(
                structural_signal_frame["local_swing_only_flag"].sum()
            ),
        }
    )


def save_section6b_structural_tables(
    tables: dict[str, pd.DataFrame | pd.Series],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save Version B structural enhanced tables without touching Version A."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "structural_poi_table": out_dir / "section6b_structural_poi_table_gc.parquet",
        "structural_retest_table": out_dir / "section6b_structural_retest_table_gc.parquet",
        "structural_candidate_trade_table": out_dir
        / "section6b_structural_candidate_trade_table_gc.parquet",
        "structural_signal_frame": out_dir / "section6b_structural_signal_frame_gc.parquet",
    }
    for key, path in output_paths.items():
        tables[key].to_parquet(path, index=False)
    return output_paths


def _append_directional_pois(
    *,
    records: list[dict[str, object]],
    direction: str,
    trade_date: object,
    day: pd.DataFrame,
    break_mode: str,
    swing_n: int,
    break_idxs: np.ndarray,
    poi_base: np.ndarray,
    fvg_ticks: np.ndarray,
    gap_ticks: np.ndarray,
    break_price: np.ndarray,
    broken_swing_levels: np.ndarray,
    opposite_swing_levels: np.ndarray,
    broken_swing_idxs: np.ndarray,
    opposite_swing_idxs: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    minute: np.ndarray,
    bar_ids: np.ndarray,
    ts_utc: np.ndarray,
    ts_ny: np.ndarray,
    tradable: np.ndarray,
    roll_flag: np.ndarray,
    search_window: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
    cfg: Section6Config,
) -> None:
    seen_broken_swings: set[int] = set()
    poi_candidates = np.flatnonzero(poi_base)

    for break_idx in break_idxs:
        broken_swing_idx = int(broken_swing_idxs[break_idx])
        if broken_swing_idx in seen_broken_swings:
            continue
        seen_broken_swings.add(broken_swing_idx)

        start_idx = int(opposite_swing_idxs[break_idx])
        if start_idx < 0 or start_idx >= break_idx:
            continue

        lower = np.searchsorted(poi_candidates, start_idx + 1, side="left")
        upper = np.searchsorted(poi_candidates, break_idx, side="left")
        candidates = poi_candidates[lower:upper]
        if candidates.size == 0:
            continue

        leg_high = float(np.nanmax(high[start_idx : break_idx + 1]))
        leg_low = float(np.nanmin(low[start_idx : break_idx + 1]))
        leg_volume = float(np.nansum(volume[start_idx : break_idx + 1]))
        for middle_idx in candidates:
            confirm_idx = int(middle_idx + 1)
            activation_idx = int(max(confirm_idx, break_idx))
            if not (search_window[confirm_idx] and search_window[activation_idx]):
                continue
            if not (tradable[middle_idx] and tradable[confirm_idx] and tradable[activation_idx]):
                continue
            if roll_flag[middle_idx] or roll_flag[confirm_idx] or roll_flag[activation_idx]:
                continue

            broken_level = float(broken_swing_levels[break_idx])
            if direction == "bullish":
                break_distance_ticks = (break_price[break_idx] - broken_level) / cfg.tick_size
            else:
                break_distance_ticks = (broken_level - break_price[break_idx]) / cfg.tick_size

            record = {
                "product": "GC",
                "trade_date_ny": trade_date,
                "swing_n": swing_n,
                "break_mode": break_mode,
                "direction": direction,
                "poi_middle_bar_id": int(bar_ids[middle_idx]),
                "poi_confirm_bar_id": int(bar_ids[confirm_idx]),
                "poi_activation_bar_id": int(bar_ids[activation_idx]),
                "break_bar_id": int(bar_ids[break_idx]),
                "displacement_start_bar_id": int(bar_ids[start_idx]),
                "broken_swing_bar_id": int(bar_ids[broken_swing_idx]),
                "poi_middle_ts_event_utc": ts_utc[middle_idx],
                "poi_created_ts_event_utc": ts_utc[confirm_idx],
                "poi_activation_ts_event_utc": ts_utc[activation_idx],
                "poi_middle_ts_event_ny": ts_ny[middle_idx],
                "poi_created_ts_event_ny": ts_ny[confirm_idx],
                "poi_activation_ts_event_ny": ts_ny[activation_idx],
                "break_ts_event_utc": ts_utc[break_idx],
                "break_ts_event_ny": ts_ny[break_idx],
                "poi_activation_minute_ny": int(minute[activation_idx]),
                "poi_high": float(high[middle_idx]),
                "poi_low": float(low[middle_idx]),
                "poi_mid": float((high[middle_idx] + low[middle_idx]) / 2.0),
                "fvg_size_ticks": float(fvg_ticks[middle_idx]),
                "close_open_gap_ticks": float(gap_ticks[middle_idx]),
                "abs_close_open_gap_ticks": float(abs(gap_ticks[middle_idx])),
                "poi_size_ticks": float((high[middle_idx] - low[middle_idx]) / cfg.tick_size),
                "prev_candle_high": float(high[middle_idx - 1]),
                "prev_candle_low": float(low[middle_idx - 1]),
                "next_candle_high": float(high[middle_idx + 1]),
                "next_candle_low": float(low[middle_idx + 1]),
                "broken_swing_price": broken_level,
                "opposite_swing_price": float(opposite_swing_levels[break_idx]),
                "break_distance_ticks": float(break_distance_ticks),
                "displacement_range_ticks": float((leg_high - leg_low) / cfg.tick_size),
                "displacement_candles": int(break_idx - start_idx + 1),
                "displacement_volume": leg_volume,
                "candles_from_break_to_poi_activation": int(activation_idx - break_idx),
                **_feature_snapshot_from_arrays(feature_arrays, activation_idx, prefix="poi"),
            }
            records.append(record)


def _structural_break_frame(
    *,
    segment: pd.DataFrame,
    segment_id: object,
    direction: str,
    break_mode: str,
    swing_window: int,
    break_idx: np.ndarray,
    break_price: np.ndarray,
    structural_level: np.ndarray,
    structural_swing_idx: np.ndarray,
    bar_ids: np.ndarray,
    cfg: Section6Config,
) -> pd.DataFrame:
    if break_idx.size == 0:
        return pd.DataFrame()

    swing_idx = structural_swing_idx[break_idx].astype("int32", copy=False)
    valid = swing_idx >= 0
    if not valid.any():
        return pd.DataFrame()

    break_idx = break_idx[valid]
    swing_idx = swing_idx[valid]
    swing_price = structural_level[break_idx]
    if direction == "bullish":
        distance = (break_price[break_idx] - swing_price) / cfg.tick_size
    else:
        distance = (swing_price - break_price[break_idx]) / cfg.tick_size

    return pd.DataFrame(
        {
            "continuous_segment_id": segment_id,
            "direction": direction,
            "structural_break_mode": break_mode,
            "structural_swing_window_broken": swing_window,
            "structural_swing_bar_id": bar_ids[swing_idx],
            "structural_swing_price": swing_price,
            "structural_break_bar_id": bar_ids[break_idx],
            "structural_break_distance_ticks": distance,
            "structural_break_ts_event_utc": segment["ts_event_utc"].to_numpy()[break_idx],
            "structural_break_ts_event_ny": segment["ts_event_ny"].to_numpy()[break_idx],
            "structural_break_trade_date_ny": segment["trade_date_ny"].to_numpy()[break_idx],
        }
    )


def _merge_structural_columns(
    table: pd.DataFrame,
    structural_poi_table: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    structural_cols = ["poi_id"] + [c for c in STRUCTURAL_COLUMNS if c in structural_poi_table]
    columns_to_drop = [c for c in STRUCTURAL_COLUMNS if c in table.columns]
    base = table.drop(columns=columns_to_drop)
    out = base.merge(
        structural_poi_table[structural_cols],
        on="poi_id",
        how="left",
        validate="many_to_one",
    )
    _optimize_structural_dtypes(out)
    return out


def _format_structural_window_list(
    matches_by_window: dict[int, np.ndarray],
    row_count: int,
) -> pd.Series:
    windows = sorted(matches_by_window)
    labels = []
    for row_idx in range(row_count):
        broken = [str(window) for window in windows if matches_by_window[window][row_idx]]
        labels.append(",".join(broken) if broken else pd.NA)
    return pd.Series(labels, dtype="category")


def _optimize_structural_dtypes(df: pd.DataFrame) -> None:
    for col in ("structural_break_mode", "structural_swing_windows_broken"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in (
        "structural_swing_window_broken",
        "structural_swing_bar_id",
        "structural_break_bar_id",
    ):
        if col in df.columns and str(df[col].dtype) != "Int64":
            df[col] = df[col].astype("Int64")


def _precompute_poi_masks(
    *,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    segment: np.ndarray,
    regular: np.ndarray,
    cfg: Section6Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n = len(high)
    bull_fvg_ticks = np.full(n, np.nan)
    bear_fvg_ticks = np.full(n, np.nan)
    bull_gap_ticks = np.full(n, np.nan)
    bear_gap_ticks = np.full(n, np.nan)

    if n >= 3:
        mid = np.arange(1, n - 1)
        bull_fvg_ticks[mid] = (low[mid + 1] - high[mid - 1]) / cfg.tick_size
        bear_fvg_ticks[mid] = (low[mid - 1] - high[mid + 1]) / cfg.tick_size
        bull_gap_ticks[mid] = (open_[mid + 1] - close[mid]) / cfg.tick_size
        bear_gap_ticks[mid] = (open_[mid + 1] - close[mid]) / cfg.tick_size

    prev_same_segment = np.r_[False, segment[1:] == segment[:-1]]
    next_same_segment = np.r_[segment[:-1] == segment[1:], False]
    three_bar_clean = prev_same_segment & next_same_segment & regular & np.r_[regular[1:], False]
    bull_poi_base = (
        (bull_fvg_ticks >= cfg.min_fvg_ticks - 1e-9)
        & (bull_gap_ticks >= cfg.min_close_open_gap_ticks - 1e-9)
        & three_bar_clean
    )
    bear_poi_base = (
        (bear_fvg_ticks >= cfg.min_fvg_ticks - 1e-9)
        & (bear_gap_ticks <= -cfg.min_close_open_gap_ticks + 1e-9)
        & three_bar_clean
    )
    return (
        bull_poi_base,
        bear_poi_base,
        {
            "bull_fvg_ticks": bull_fvg_ticks,
            "bear_fvg_ticks": bear_fvg_ticks,
            "bull_gap_ticks": bull_gap_ticks,
            "bear_gap_ticks": bear_gap_ticks,
        },
    )


def _confirmed_swings(
    high: np.ndarray,
    low: np.ndarray,
    swing_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    if swing_n % 2 == 0 or swing_n < 3:
        raise ValueError("swing_n must be an odd integer >= 3")

    k = swing_n // 2
    n = len(high)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    if n < swing_n:
        return swing_high, swing_low

    high_window = sliding_window_view(high, swing_n)
    low_window = sliding_window_view(low, swing_n)
    center_high = high_window[:, k]
    center_low = low_window[:, k]
    neighbor_high = np.maximum(high_window[:, :k].max(axis=1), high_window[:, k + 1 :].max(axis=1))
    neighbor_low = np.minimum(low_window[:, :k].min(axis=1), low_window[:, k + 1 :].min(axis=1))
    swing_high[k : n - k] = center_high > neighbor_high
    swing_low[k : n - k] = center_low < neighbor_low
    return swing_high, swing_low


def _build_prior_swing_state(
    high: np.ndarray,
    low: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    swing_n: int,
) -> dict[str, np.ndarray]:
    k = swing_n // 2
    n = len(high)
    high_confirm_level = np.full(n, np.nan)
    low_confirm_level = np.full(n, np.nan)
    high_confirm_swing_idx = np.full(n, -1, dtype=np.int32)
    low_confirm_swing_idx = np.full(n, -1, dtype=np.int32)

    sh_idx = np.flatnonzero(swing_high)
    sl_idx = np.flatnonzero(swing_low)
    sh_confirm = sh_idx + k
    sl_confirm = sl_idx + k
    sh_valid = sh_confirm < n
    sl_valid = sl_confirm < n
    high_confirm_level[sh_confirm[sh_valid]] = high[sh_idx[sh_valid]]
    high_confirm_swing_idx[sh_confirm[sh_valid]] = sh_idx[sh_valid]
    low_confirm_level[sl_confirm[sl_valid]] = low[sl_idx[sl_valid]]
    low_confirm_swing_idx[sl_confirm[sl_valid]] = sl_idx[sl_valid]

    return {
        "prior_high_level": _shift_one(_ffill_float(high_confirm_level), np.nan),
        "prior_low_level": _shift_one(_ffill_float(low_confirm_level), np.nan),
        "prior_high_swing_idx": _shift_one(_ffill_int(high_confirm_swing_idx), np.int32(-1)),
        "prior_low_swing_idx": _shift_one(_ffill_int(low_confirm_swing_idx), np.int32(-1)),
    }


def _ffill_float(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    mask = np.isnan(out)
    idx = np.where(~mask, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out[mask] = out[idx[mask]]
    return out


def _ffill_int(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    valid = out >= 0
    idx = np.where(valid, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out[~valid] = out[idx[~valid]]
    return out


def _shift_one(arr: np.ndarray, fill_value: object) -> np.ndarray:
    out = np.empty_like(arr)
    out[0] = fill_value
    out[1:] = arr[:-1]
    return out


def _feature_arrays(day: pd.DataFrame) -> dict[str, np.ndarray]:
    cols = {
        "trend_bias_20_50": "trend_bias_20_50",
        "distance_from_vwap_ticks": "distance_from_vwap_ticks",
        "vwap_slope_15m_ticks": "vwap_slope_15m_ticks",
        "relative_volume_60m": "relative_volume_60m",
        "volume_zscore_240m": "volume_zscore_240m",
        "rolling_atr_60m": "rolling_atr_60m",
        "volatility_regime": "volatility_regime",
        "large_move_flag": "large_move_flag",
        "volume_spike_flag": "volume_spike_flag",
        "directional_efficiency_20m": "directional_efficiency_20m",
        "distance_to_day_high_ticks": "distance_to_day_high_ticks",
        "distance_to_day_low_ticks": "distance_to_day_low_ticks",
        "distance_to_previous_day_high_ticks": "distance_to_previous_day_high_ticks",
        "distance_to_previous_day_low_ticks": "distance_to_previous_day_low_ticks",
        "forward_return_5m": "forward_return_5m",
        "forward_return_15m": "forward_return_15m",
        "forward_return_30m": "forward_return_30m",
        "forward_return_60m": "forward_return_60m",
    }
    return {
        suffix: day[source_col].to_numpy()
        for suffix, source_col in cols.items()
        if source_col in day.columns
    }


def _feature_snapshot_from_arrays(
    feature_arrays: dict[str, np.ndarray],
    pos: int,
    prefix: str,
) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for suffix, values in feature_arrays.items():
        value = values[pos]
        snapshot[f"{prefix}_{suffix}"] = value
    return snapshot


def _validate_input_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = sorted(set(required_columns).difference(df.columns))
    if missing:
        raise KeyError(f"research_bars is missing required Section 6 columns: {missing}")
