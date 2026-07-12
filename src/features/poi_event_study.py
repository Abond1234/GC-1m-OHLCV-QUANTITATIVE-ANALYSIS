"""Section 7 POI event-study research utilities.

The functions in this module evaluate Section 6/6B POI candidate rows as
forward event studies.  They deliberately do not simulate position state,
equity curves, trade sequencing, commissions, or prop-firm rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Section7Config:
    """Configuration for Section 7 prototype signal research."""

    tick_size: float = 0.10
    forward_horizons: tuple[int, ...] = (5, 15, 30, 60, 120, 180, 240, 360)
    runner_thresholds: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0)
    r_caps: tuple[float, ...] = (5.0, 10.0)
    london_start_minute: int = 180
    london_end_minute: int = 360
    ny_start_minute: int = 420
    last_entry_minute_ny: int = 720
    forced_exit_minute_ny: int = 930
    min_ranking_sample_size: int = 1_000


@dataclass(frozen=True)
class EventStudyGroupSpec:
    """Definition of a Section 7 grouped event-study view."""

    name: str
    columns: tuple[str, ...] = ()
    filter_column: str | None = None
    filter_value: Any | None = None


SECTION7_SIGNAL_REQUIRED_COLUMNS = [
    "signal_id",
    "candidate_trade_id",
    "retest_id",
    "poi_id",
    "direction",
    "trade_side",
    "entry_variant",
    "stop_model",
    "entry_price",
    "stop_price",
    "risk_points",
    "stop_ticks",
    "valid_candidate_flag",
    "retest_bar_id",
    "retest_ts_event_utc",
    "retest_ts_event_ny",
    "retest_minute_ny",
    "execution_window_label",
    "trade_date_ny",
    "swing_n",
    "break_mode",
    "retest_number",
    "first_touch_flag",
    "retest_high",
    "retest_low",
    "full_poi_cross_flag",
    "time_since_poi_activation_minutes",
    "candles_since_poi_activation",
    "structural_swing_break_flag",
    "structural_swing_window_broken",
    "structural_swing_windows_broken",
    "structural_break_mode",
    "local_swing_only_flag",
]

SECTION7_CONTEXT_COLUMNS = [
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
    "fvg_size_ticks",
    "abs_close_open_gap_ticks",
    "poi_size_ticks",
    "break_distance_ticks",
    "displacement_range_ticks",
    "displacement_candles",
]

# Optional Section 6C lineage and geometry fields.  Legacy Section 7 remains
# valid when they are absent; Section 7R retains them when present.
SECTION7_REFINED_CONTEXT_COLUMNS = [
    "canonical_poi_id",
    "poi_variant_id",
    "canonical_retest_id",
    "canonical_retest_number",
    "candidate_variant_id",
    "canonical_candidate_id",
    "poi_definition_version",
    "poi_geometry_case",
    "poi_zone_expanded_flag",
    "poi_zone_expansion_points",
    "poi_zone_expansion_ticks",
    "fvg_size_points",
    "fvg_ge_3tick_flag",
    "fvg_ge_4tick_flag",
    "fvg_ge_5tick_flag",
]

SECTION7_EXISTING_FORWARD_COLUMNS = [
    "forward_5m_r",
    "forward_15m_r",
    "forward_30m_r",
    "forward_60m_r",
]

SECTION7_BAR_REQUIRED_COLUMNS = [
    "ts_event_utc",
    "ts_event_ny",
    "trade_date_ny",
    "product",
    "high",
    "low",
    "close",
    "minute_of_day_ny",
    "continuous_segment_id",
    "tradable_research_flag",
    "roll_window_flag",
]

SECTION7_VISUAL_AUDIT_BAR_COLUMNS = list(
    dict.fromkeys(SECTION7_BAR_REQUIRED_COLUMNS + ["open"])
)

SECTION7_VISUAL_AUDIT_SIGNAL_COLUMNS = list(
    dict.fromkeys(
        SECTION7_SIGNAL_REQUIRED_COLUMNS
        + [
            "poi_created_ts_event_ny",
            "poi_activation_ts_event_ny",
            "poi_low",
            "poi_high",
            "target_1R_price",
            "target_2R_price",
            "target_3R_price",
            "target_5R_price",
            "forward_60m_r",
            "retest_volatility_regime",
        ]
    )
)

SECTION7_POI_SELECTION_AUDIT_SIGNAL_COLUMNS = list(
    dict.fromkeys(
        [
            "signal_id",
            "poi_id",
            "direction",
            "trade_date_ny",
            "swing_n",
            "break_mode",
            "execution_window_label",
            "poi_middle_bar_id",
            "poi_confirm_bar_id",
            "poi_activation_bar_id",
            "break_bar_id",
            "poi_middle_ts_event_ny",
            "poi_created_ts_event_ny",
            "poi_activation_ts_event_ny",
            "break_ts_event_ny",
            "poi_low",
            "poi_high",
            "structural_swing_break_flag",
            "structural_swing_window_broken",
            "local_swing_only_flag",
        ]
    )
)


def default_section7_group_specs() -> list[EventStudyGroupSpec]:
    """Return the standard Section 7 diagnostic grouping contract."""

    candidate_group = (
        "structural_validation_bucket",
        "structural_window_bucket",
        "swing_n",
        "break_mode",
        "trade_side",
        "entry_variant",
        "stop_model",
        "execution_window_label",
        "retest_volatility_regime",
        "relative_volume_bucket",
    )
    return [
        EventStudyGroupSpec("baseline_all_candidates"),
        EventStudyGroupSpec(
            "primary_first_touch_candidates",
            filter_column="first_touch_flag",
            filter_value=True,
        ),
        EventStudyGroupSpec("poi_touch_direction", ("direction", "trade_side")),
        EventStudyGroupSpec("poi_touch_order", ("touch_order_bucket", "trade_side")),
        EventStudyGroupSpec(
            "poi_touch_number",
            ("retest_number_bucket", "direction", "trade_side"),
        ),
        EventStudyGroupSpec("poi_age_minutes", ("time_since_poi_bucket", "direction")),
        EventStudyGroupSpec("poi_age_candles", ("candles_since_poi_bucket", "direction")),
        EventStudyGroupSpec("structural_validation", ("structural_validation_bucket",)),
        EventStudyGroupSpec("structural_window", ("structural_window_bucket",)),
        EventStudyGroupSpec(
            "structural_partition",
            ("structural_swing_break_flag", "local_swing_only_flag"),
        ),
        EventStudyGroupSpec(
            "swing_break_mode",
            ("swing_n", "break_mode", "direction"),
        ),
        EventStudyGroupSpec(
            "swing_break_structural",
            ("swing_n", "break_mode", "direction", "structural_validation_bucket"),
        ),
        EventStudyGroupSpec(
            "entry_stop_model",
            ("entry_variant", "stop_model", "trade_side"),
        ),
        EventStudyGroupSpec(
            "stop_ticks_bucket",
            ("stop_ticks_bucket", "entry_variant", "stop_model"),
        ),
        EventStudyGroupSpec("execution_session", ("execution_window_label", "trade_side")),
        EventStudyGroupSpec("entry_hour", ("entry_hour_bucket", "trade_side")),
        EventStudyGroupSpec("volatility_regime", ("retest_volatility_regime", "trade_side")),
        EventStudyGroupSpec("relative_volume", ("relative_volume_bucket", "trade_side")),
        EventStudyGroupSpec("volume_zscore", ("volume_zscore_bucket", "trade_side")),
        EventStudyGroupSpec("volume_spike", ("retest_volume_spike_flag", "trade_side")),
        EventStudyGroupSpec("trend_alignment", ("trend_alignment", "trade_side")),
        EventStudyGroupSpec("vwap_alignment", ("vwap_alignment", "trade_side")),
        EventStudyGroupSpec("vwap_extension", ("vwap_extension_bucket", "trade_side")),
        EventStudyGroupSpec(
            "directional_efficiency",
            ("directional_efficiency_bucket", "trade_side"),
        ),
        EventStudyGroupSpec(
            "poi_quality",
            ("poi_size_bucket", "fvg_size_bucket", "break_distance_bucket"),
        ),
        EventStudyGroupSpec(
            "candidate_signal_group",
            candidate_group,
        ),
        EventStudyGroupSpec(
            "candidate_signal_group_first_touch",
            candidate_group,
            filter_column="first_touch_flag",
            filter_value=True,
        ),
    ]


def build_section7_event_study(
    signal_frame: pd.DataFrame,
    research_bars: pd.DataFrame,
    config: Section7Config | None = None,
    group_specs: list[EventStudyGroupSpec] | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Build Section 7 validation, grouped event studies, and ranking tables."""

    cfg = config or Section7Config()
    specs = group_specs or default_section7_group_specs()
    validation = build_section7_validation(signal_frame, research_bars, cfg)
    _raise_if_missing_inputs(signal_frame, research_bars)

    bars = prepare_section7_bar_frame(research_bars)
    signals = prepare_section7_signal_frame(signal_frame, cfg)

    summary_frames: list[pd.DataFrame] = []
    horizon_quality_records: list[dict[str, Any]] = []

    for horizon in cfg.forward_horizons:
        path_features = build_bar_horizon_path_features(bars, horizon, cfg)
        for mode in ("fixed", "capped"):
            metrics = build_signal_horizon_metrics(signals, path_features, horizon, mode, cfg)
            horizon_quality_records.append(
                _horizon_quality_record(metrics, horizon=horizon, metric_mode=mode)
            )
            summary_frames.append(
                summarize_section7_groups(
                    signal_base=signals,
                    metrics=metrics,
                    group_specs=specs,
                    horizon=horizon,
                    metric_mode=mode,
                    config=cfg,
                )
            )

    event_study_summary = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame()
    )
    horizon_quality = pd.DataFrame.from_records(horizon_quality_records)
    candidate_signal_ranking = rank_section7_candidate_groups(event_study_summary, cfg)

    return {
        "section7_validation": validation,
        "section7_horizon_quality": horizon_quality,
        "section7_event_study_summary": event_study_summary,
        "section7_candidate_signal_ranking": candidate_signal_ranking,
    }


def build_section7_validation(
    signal_frame: pd.DataFrame,
    research_bars: pd.DataFrame,
    config: Section7Config | None = None,
) -> pd.Series:
    """Return lightweight validation checks for Section 7 inputs."""

    cfg = config or Section7Config()
    missing_signal = sorted(set(SECTION7_SIGNAL_REQUIRED_COLUMNS).difference(signal_frame.columns))
    missing_bars = sorted(set(SECTION7_BAR_REQUIRED_COLUMNS).difference(research_bars.columns))
    structural_cols = {
        "structural_swing_break_flag",
        "structural_swing_window_broken",
        "structural_swing_windows_broken",
        "structural_break_mode",
        "local_swing_only_flag",
    }
    session_ok = (
        signal_frame["execution_window_label"]
        .isin(["London Execution", "New York Execution"])
        .all()
        if "execution_window_label" in signal_frame
        else False
    )
    entry_minute_ok = (
        signal_frame["retest_minute_ny"].le(cfg.last_entry_minute_ny).all()
        if "retest_minute_ny" in signal_frame
        else False
    )
    if structural_cols.issubset(signal_frame.columns):
        structural_flag = signal_frame["structural_swing_break_flag"].astype(bool)
        local_only = signal_frame["local_swing_only_flag"].astype(bool)
        structural_partition_valid = bool((structural_flag ^ local_only).all())
    else:
        structural_partition_valid = False

    checks = {
        "signal_row_count": len(signal_frame),
        "research_bar_row_count": len(research_bars),
        "signal_required_columns_present": len(missing_signal) == 0,
        "bar_required_columns_present": len(missing_bars) == 0,
        "missing_signal_columns": ", ".join(missing_signal),
        "missing_bar_columns": ", ".join(missing_bars),
        "signal_ids_unique": signal_frame["signal_id"].is_unique
        if "signal_id" in signal_frame
        else False,
        "existing_5_15_30_60_forward_r_present": set(SECTION7_EXISTING_FORWARD_COLUMNS).issubset(
            signal_frame.columns
        ),
        "structural_columns_present": structural_cols.issubset(signal_frame.columns),
        "structural_partition_valid": structural_partition_valid,
        "session_columns_present": {
            "execution_window_label",
            "retest_minute_ny",
            "retest_ts_event_ny",
        }.issubset(signal_frame.columns),
        "entry_rows_in_approved_execution_windows": session_ok,
        "no_candidate_entry_after_1200_ny": entry_minute_ok,
        "risk_points_positive": signal_frame["risk_points"].gt(0).all()
        if "risk_points" in signal_frame
        else False,
        "valid_candidate_flag_all_true": signal_frame["valid_candidate_flag"].astype(bool).all()
        if "valid_candidate_flag" in signal_frame
        else False,
        "volatility_context_present": "retest_volatility_regime" in signal_frame.columns,
        "volume_context_present": {
            "retest_relative_volume_60m",
            "retest_volume_zscore_240m",
        }.issubset(signal_frame.columns),
        "gc_research_bars_available": bool(
            research_bars["product"].eq("GC").any() if "product" in research_bars else False
        ),
    }
    return pd.Series(checks)


def prepare_section7_bar_frame(research_bars: pd.DataFrame) -> pd.DataFrame:
    """Build the GC bar frame whose positional bar_id matches Section 6."""

    _validate_columns(research_bars, SECTION7_BAR_REQUIRED_COLUMNS, "research_bars")
    bar_cols = list(dict.fromkeys(SECTION7_BAR_REQUIRED_COLUMNS + ["bar_id"]))
    existing_cols = [c for c in bar_cols if c in research_bars.columns]
    bars = (
        research_bars.loc[research_bars["product"].eq("GC"), existing_cols]
        .sort_values(["trade_date_ny", "ts_event_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    bars["bar_id"] = np.arange(len(bars), dtype=np.int32)
    return bars


def prepare_section7_signal_frame(
    signal_frame: pd.DataFrame,
    config: Section7Config | None = None,
) -> pd.DataFrame:
    """Create the narrow candidate/context frame used by Section 7 summaries."""

    cfg = config or Section7Config()
    _validate_columns(signal_frame, SECTION7_SIGNAL_REQUIRED_COLUMNS, "signal_frame")

    selected_cols = list(
        dict.fromkeys(
            SECTION7_SIGNAL_REQUIRED_COLUMNS
            + SECTION7_CONTEXT_COLUMNS
            + SECTION7_REFINED_CONTEXT_COLUMNS
            + SECTION7_EXISTING_FORWARD_COLUMNS
        )
    )
    selected_cols = [c for c in selected_cols if c in signal_frame.columns]
    out = signal_frame.loc[:, selected_cols].copy()

    out["entry_hour_bucket"] = (
        out["retest_minute_ny"].floordiv(60).astype("int16").astype(str).str.zfill(2)
        + ":00"
    )
    out["touch_order_bucket"] = np.where(
        out["first_touch_flag"].astype(bool),
        "first_touch",
        "later_touch",
    )
    out["retest_number_bucket"] = _cut_numeric(
        out["retest_number"],
        bins=[0, 1, 2, 3, 5, 10, np.inf],
        labels=["1", "2", "3", "4-5", "6-10", "11+"],
    )
    out["time_since_poi_bucket"] = _cut_numeric(
        out["time_since_poi_activation_minutes"],
        bins=[-np.inf, 15, 30, 60, 120, 240, np.inf],
        labels=["<=15m", "15-30m", "30-60m", "1-2h", "2-4h", ">4h"],
    )
    out["candles_since_poi_bucket"] = _cut_numeric(
        out["candles_since_poi_activation"],
        bins=[-np.inf, 15, 30, 60, 120, 240, np.inf],
        labels=["<=15", "15-30", "30-60", "60-120", "120-240", ">240"],
    )
    out["structural_validation_bucket"] = np.where(
        out["structural_swing_break_flag"].astype(bool),
        "structural_validated",
        "local_swing_only",
    )
    structural_window = out["structural_swing_window_broken"].astype("Float64")
    out["structural_window_bucket"] = np.select(
        [
            structural_window.eq(15).fillna(False).to_numpy(bool),
            structural_window.eq(21).fillna(False).to_numpy(bool),
            structural_window.eq(31).fillna(False).to_numpy(bool),
        ],
        ["15", "21", "31"],
        default="none",
    )
    out["stop_ticks_bucket"] = _cut_numeric(
        out["stop_ticks"],
        bins=[-np.inf, 25, 40, 60, 80, 100, np.inf],
        labels=["<=25", "25-40", "40-60", "60-80", "80-100", ">100"],
    )
    out["relative_volume_bucket"] = _cut_numeric(
        out.get("retest_relative_volume_60m", pd.Series(np.nan, index=out.index)),
        bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
        labels=["very_low", "low", "normal", "high", "spike"],
    )
    out["volume_zscore_bucket"] = _cut_numeric(
        out.get("retest_volume_zscore_240m", pd.Series(np.nan, index=out.index)),
        bins=[-np.inf, -1.0, 0.0, 1.0, 2.0, np.inf],
        labels=["<-1", "-1-0", "0-1", "1-2", ">2"],
    )
    out["vwap_extension_bucket"] = _cut_numeric(
        out.get("retest_distance_from_vwap_ticks", pd.Series(np.nan, index=out.index)).abs(),
        bins=[-np.inf, 10, 30, 60, 120, np.inf],
        labels=["near", "moderate", "extended", "very_extended", "extreme"],
    )
    out["directional_efficiency_bucket"] = _cut_numeric(
        out.get("retest_directional_efficiency_20m", pd.Series(np.nan, index=out.index)),
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["choppy", "mixed", "directional", "efficient"],
    )
    out["poi_size_bucket"] = _cut_numeric(
        out.get("poi_size_ticks", pd.Series(np.nan, index=out.index)),
        bins=[-np.inf, 5, 10, 20, 40, np.inf],
        labels=["tiny", "small", "medium", "large", "very_large"],
    )
    if "fvg_ge_3tick_flag" in out.columns:
        fvg_ticks = pd.to_numeric(out["fvg_size_ticks"], errors="coerce")
        out["fvg_size_bucket"] = pd.Categorical(
            np.select(
                [
                    fvg_ticks.eq(3),
                    fvg_ticks.eq(4),
                    fvg_ticks.eq(5),
                    fvg_ticks.between(6, 9),
                    fvg_ticks.between(10, 19),
                    fvg_ticks.ge(20),
                ],
                ["3", "4", "5", "6-9", "10-19", "20+"],
                default="below_3",
            ),
            categories=["below_3", "3", "4", "5", "6-9", "10-19", "20+"],
            ordered=True,
        )
        expansion_ticks = pd.to_numeric(
            out.get("poi_zone_expansion_ticks", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        out["zone_expansion_tick_bucket"] = pd.Categorical(
            np.select(
                [
                    expansion_ticks.eq(0),
                    expansion_ticks.eq(1),
                    expansion_ticks.eq(2),
                    expansion_ticks.between(3, 5),
                    expansion_ticks.between(6, 9),
                    expansion_ticks.ge(10),
                ],
                ["0", "1", "2", "3-5", "6-9", "10+"],
                default="unknown",
            ),
            categories=["0", "1", "2", "3-5", "6-9", "10+", "unknown"],
            ordered=True,
        )
    else:
        out["fvg_size_bucket"] = _cut_numeric(
            out.get("fvg_size_ticks", pd.Series(np.nan, index=out.index)),
            bins=[-np.inf, 2, 5, 10, 20, np.inf],
            labels=["1-2", "2-5", "5-10", "10-20", ">20"],
        )
    out["break_distance_bucket"] = _cut_numeric(
        out.get("break_distance_ticks", pd.Series(np.nan, index=out.index)),
        bins=[-np.inf, 5, 10, 20, 40, np.inf],
        labels=["<=5", "5-10", "10-20", "20-40", ">40"],
    )
    out["trend_alignment"] = _trade_alignment(
        trade_side=out["trade_side"],
        bullish_condition=out.get("retest_trend_bias_20_50", pd.Series("neutral", index=out.index)).eq(
            "bullish"
        ),
        bearish_condition=out.get("retest_trend_bias_20_50", pd.Series("neutral", index=out.index)).eq(
            "bearish"
        ),
        neutral_condition=out.get("retest_trend_bias_20_50", pd.Series("neutral", index=out.index)).eq(
            "neutral"
        ),
    )
    vwap_distance = out.get("retest_distance_from_vwap_ticks", pd.Series(np.nan, index=out.index))
    out["vwap_alignment"] = _trade_alignment(
        trade_side=out["trade_side"],
        bullish_condition=vwap_distance.gt(5),
        bearish_condition=vwap_distance.lt(-5),
        neutral_condition=vwap_distance.abs().le(5),
    )

    for col in [col for col in out.columns if out[col].dtype == "object"]:
        out[col] = out[col].astype("string")
    for col in (
        "entry_hour_bucket",
        "touch_order_bucket",
        "retest_number_bucket",
        "time_since_poi_bucket",
        "candles_since_poi_bucket",
        "structural_validation_bucket",
        "structural_window_bucket",
        "stop_ticks_bucket",
        "relative_volume_bucket",
        "volume_zscore_bucket",
        "vwap_extension_bucket",
        "directional_efficiency_bucket",
        "poi_size_bucket",
        "fvg_size_bucket",
        "zone_expansion_tick_bucket",
        "break_distance_bucket",
        "trend_alignment",
        "vwap_alignment",
    ):
        if col in out.columns:
            out[col] = out[col].astype("category")

    out = out.loc[out["retest_minute_ny"].le(cfg.last_entry_minute_ny)].reset_index(drop=True)
    return out


def build_bar_horizon_path_features(
    bars: pd.DataFrame,
    horizon_minutes: int,
    config: Section7Config | None = None,
) -> pd.DataFrame:
    """Precompute fixed and forced-exit-capped path features by GC bar."""

    cfg = config or Section7Config()
    required = [
        "bar_id",
        "ts_event_utc",
        "trade_date_ny",
        "high",
        "low",
        "close",
        "minute_of_day_ny",
        "continuous_segment_id",
    ]
    _validate_columns(bars, required, "bars")

    n = len(bars)
    fixed_valid = np.zeros(n, dtype=bool)
    capped_valid = np.zeros(n, dtype=bool)
    capped_forced_exit = np.zeros(n, dtype=bool)
    fixed_exit_close = np.full(n, np.nan, dtype="float64")
    capped_exit_close = np.full(n, np.nan, dtype="float64")
    fixed_max_high = np.full(n, np.nan, dtype="float64")
    capped_max_high = np.full(n, np.nan, dtype="float64")
    fixed_min_low = np.full(n, np.nan, dtype="float64")
    capped_min_low = np.full(n, np.nan, dtype="float64")
    fixed_exit_bar_id = np.full(n, -1, dtype="int32")
    capped_exit_bar_id = np.full(n, -1, dtype="int32")

    delta_ns = np.int64(horizon_minutes) * np.int64(60_000_000_000)

    grouped = bars.groupby(["trade_date_ny", "continuous_segment_id"], observed=True, sort=False)
    for _, segment in grouped:
        positions = segment.index.to_numpy("int64", copy=False)
        if positions.size == 0:
            continue

        minute = segment["minute_of_day_ny"].to_numpy("int32", copy=False)
        ts_ns = _datetime64_ns(segment["ts_event_utc"])
        high = segment["high"].to_numpy("float64", copy=False)
        low = segment["low"].to_numpy("float64", copy=False)
        close = segment["close"].to_numpy("float64", copy=False)
        bar_ids = segment["bar_id"].to_numpy("int32", copy=False)

        forced_candidates = np.flatnonzero(minute <= cfg.forced_exit_minute_ny)
        if forced_candidates.size == 0:
            continue
        forced_loc = int(forced_candidates[-1])
        start_locs = np.flatnonzero(minute <= cfg.last_entry_minute_ny)
        if start_locs.size == 0:
            continue

        target_ns = ts_ns[start_locs] + delta_ns
        target_locs = np.searchsorted(ts_ns, target_ns, side="left")
        target_inside = target_locs < len(segment)
        exact_target = np.zeros(start_locs.size, dtype=bool)
        exact_target[target_inside] = ts_ns[target_locs[target_inside]] == target_ns[
            target_inside
        ]

        fixed_within_forced_exit = minute[start_locs] + horizon_minutes <= cfg.forced_exit_minute_ny
        fixed_local_valid = (
            fixed_within_forced_exit
            & exact_target
            & (target_locs > start_locs)
        )

        capped_uses_forced_exit = minute[start_locs] + horizon_minutes > cfg.forced_exit_minute_ny
        capped_local_valid = np.where(
            capped_uses_forced_exit,
            forced_loc > start_locs,
            fixed_local_valid,
        )
        capped_target_locs = np.where(capped_uses_forced_exit, forced_loc, target_locs)

        high_table = _build_sparse_table(high, op="max")
        low_table = _build_sparse_table(low, op="min")

        if fixed_local_valid.any():
            local_left = start_locs[fixed_local_valid]
            local_right = target_locs[fixed_local_valid]
            global_left = positions[local_left]
            fixed_valid[global_left] = True
            fixed_exit_close[global_left] = close[local_right]
            fixed_max_high[global_left] = _query_sparse_table(
                high_table,
                local_left,
                local_right,
                op="max",
            )
            fixed_min_low[global_left] = _query_sparse_table(
                low_table,
                local_left,
                local_right,
                op="min",
            )
            fixed_exit_bar_id[global_left] = bar_ids[local_right]

        if capped_local_valid.any():
            local_left = start_locs[capped_local_valid]
            local_right = capped_target_locs[capped_local_valid]
            global_left = positions[local_left]
            capped_valid[global_left] = True
            capped_forced_exit[global_left] = capped_uses_forced_exit[capped_local_valid]
            capped_exit_close[global_left] = close[local_right]
            capped_max_high[global_left] = _query_sparse_table(
                high_table,
                local_left,
                local_right,
                op="max",
            )
            capped_min_low[global_left] = _query_sparse_table(
                low_table,
                local_left,
                local_right,
                op="min",
            )
            capped_exit_bar_id[global_left] = bar_ids[local_right]

    return pd.DataFrame(
        {
            "bar_id": bars["bar_id"].to_numpy("int32", copy=False),
            "horizon_minutes": np.int16(horizon_minutes),
            "fixed_valid": fixed_valid,
            "fixed_exit_close": fixed_exit_close,
            "fixed_max_high": fixed_max_high,
            "fixed_min_low": fixed_min_low,
            "fixed_exit_bar_id": fixed_exit_bar_id,
            "capped_valid": capped_valid,
            "capped_forced_exit_flag": capped_forced_exit,
            "capped_exit_close": capped_exit_close,
            "capped_max_high": capped_max_high,
            "capped_min_low": capped_min_low,
            "capped_exit_bar_id": capped_exit_bar_id,
        }
    )


def build_signal_horizon_metrics(
    signal_base: pd.DataFrame,
    path_features: pd.DataFrame,
    horizon_minutes: int,
    metric_mode: str,
    config: Section7Config | None = None,
) -> pd.DataFrame:
    """Map bar-level path features to candidate rows and express outcomes in R."""

    cfg = config or Section7Config()
    if metric_mode not in {"fixed", "capped"}:
        raise ValueError("metric_mode must be 'fixed' or 'capped'")

    path = path_features.sort_values("bar_id", kind="mergesort").reset_index(drop=True)
    bar_id = path["bar_id"].to_numpy("int64", copy=False)
    if not np.array_equal(bar_id, np.arange(len(path))):
        path = path.set_index("bar_id", drop=False)
        path_idx = signal_base["retest_bar_id"].to_numpy("int64", copy=False)
        aligned = path.reindex(path_idx)
    else:
        path_idx = signal_base["retest_bar_id"].to_numpy("int64", copy=False)
        if path_idx.max(initial=-1) >= len(path) or path_idx.min(initial=0) < 0:
            raise IndexError("Signal retest_bar_id is outside the Section 7 bar frame.")
        aligned = path.iloc[path_idx]

    valid = aligned[f"{metric_mode}_valid"].to_numpy(bool, copy=False)
    exit_close = aligned[f"{metric_mode}_exit_close"].to_numpy("float64", copy=False)
    max_high = aligned[f"{metric_mode}_max_high"].to_numpy("float64", copy=False)
    min_low = aligned[f"{metric_mode}_min_low"].to_numpy("float64", copy=False)
    forced_exit_flag = (
        aligned["capped_forced_exit_flag"].to_numpy(bool, copy=False)
        if metric_mode == "capped"
        else np.zeros(len(signal_base), dtype=bool)
    )

    entry = signal_base["entry_price"].to_numpy("float64", copy=False)
    risk = signal_base["risk_points"].to_numpy("float64", copy=False)
    long_side = signal_base["trade_side"].eq("long").to_numpy(bool, copy=False)
    valid = valid & np.isfinite(entry) & np.isfinite(risk) & (risk > 0)

    forward_points = np.full(len(signal_base), np.nan, dtype="float64")
    forward_r = np.full(len(signal_base), np.nan, dtype="float64")
    mfe_r = np.full(len(signal_base), np.nan, dtype="float64")
    mae_r = np.full(len(signal_base), np.nan, dtype="float64")

    if valid.any():
        idx = np.flatnonzero(valid)
        forward_points[idx] = np.where(
            long_side[idx],
            exit_close[idx] - entry[idx],
            entry[idx] - exit_close[idx],
        )
        forward_r[idx] = forward_points[idx] / risk[idx]
        mfe_r[idx] = np.where(
            long_side[idx],
            max_high[idx] - entry[idx],
            entry[idx] - min_low[idx],
        ) / risk[idx]
        mae_r[idx] = np.where(
            long_side[idx],
            entry[idx] - min_low[idx],
            max_high[idx] - entry[idx],
        ) / risk[idx]

    data: dict[str, Any] = {
        "valid": valid,
        "forced_exit_flag": forced_exit_flag & valid,
        "forward_points": forward_points,
        "forward_r": forward_r,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
    }
    for cap in cfg.r_caps:
        cap_label = _threshold_label(cap)
        data[f"forward_r_cap_{cap_label}"] = np.clip(forward_r, -cap, cap)
        data[f"mfe_r_cap_{cap_label}"] = np.minimum(mfe_r, cap)

    hit_neg_1r = np.where(valid, mae_r >= 1.0, np.nan)
    data["hit_neg_1r"] = hit_neg_1r

    retest_high = signal_base["retest_high"].to_numpy("float64", copy=False)
    retest_low = signal_base["retest_low"].to_numpy("float64", copy=False)
    entry_bar_adverse_1r = np.where(
        long_side,
        retest_low <= entry - risk,
        retest_high >= entry + risk,
    )

    for threshold in cfg.runner_thresholds:
        label = _threshold_label(threshold)
        hit_pos = np.where(valid, mfe_r >= threshold, np.nan)
        data[f"hit_pos_{label}r"] = hit_pos

        entry_bar_favorable = np.where(
            long_side,
            retest_high >= entry + threshold * risk,
            retest_low <= entry - threshold * risk,
        )
        data[f"entry_bar_ambiguous_{label}r"] = np.where(
            valid,
            entry_bar_favorable & entry_bar_adverse_1r,
            np.nan,
        )
        data[f"path_order_ambiguous_{label}r"] = np.where(
            valid,
            (mfe_r >= threshold) & (mae_r >= 1.0),
            np.nan,
        )

    return pd.DataFrame(data)


def summarize_section7_groups(
    signal_base: pd.DataFrame,
    metrics: pd.DataFrame,
    group_specs: list[EventStudyGroupSpec],
    horizon: int,
    metric_mode: str,
    config: Section7Config | None = None,
) -> pd.DataFrame:
    """Summarize Section 7 metrics for each diagnostic group specification."""

    cfg = config or Section7Config()
    frames: list[pd.DataFrame] = []
    analysis = pd.concat([signal_base.reset_index(drop=True), metrics.reset_index(drop=True)], axis=1)
    analysis["forward_r_abs"] = analysis["forward_r"].abs()

    for spec in group_specs:
        columns = [c for c in spec.columns if c in analysis.columns]
        frame = analysis
        if spec.filter_column is not None:
            if spec.filter_column not in analysis.columns:
                continue
            frame = frame.loc[frame[spec.filter_column].eq(spec.filter_value)]
        if frame.empty:
            continue
        summary = _summarize_group_frame(frame, columns, cfg)
        summary.insert(0, "group_spec", spec.name)
        summary.insert(1, "horizon_minutes", horizon)
        summary.insert(2, "metric_mode", metric_mode)
        summary.insert(3, "filter_column", spec.filter_column or "")
        summary.insert(4, "filter_value", str(spec.filter_value) if spec.filter_column else "")
        frames.append(summary)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rank_section7_candidate_groups(
    event_study_summary: pd.DataFrame,
    config: Section7Config | None = None,
) -> pd.DataFrame:
    """Rank candidate signal groups using robust capped and path metrics."""

    cfg = config or Section7Config()
    if event_study_summary.empty:
        return pd.DataFrame()

    candidates = event_study_summary.loc[
        (event_study_summary["group_spec"].eq("candidate_signal_group"))
        & (event_study_summary["metric_mode"].eq("capped"))
        & (event_study_summary["valid_count"].ge(cfg.min_ranking_sample_size))
    ].copy()
    if candidates.empty:
        return candidates

    cap5_col = "mean_r_cap_5"
    mfe_cap5_col = "mean_mfe_r_cap_5"
    score = (
        0.25 * candidates[cap5_col].clip(-2.0, 2.0).fillna(0.0)
        + 0.20 * candidates["median_r"].clip(-2.0, 2.0).fillna(0.0)
        + 0.15 * candidates["q25_r"].clip(-2.0, 2.0).fillna(0.0)
        + 0.10 * candidates["q10_r"].clip(-3.0, 1.0).fillna(0.0)
        + 0.10 * ((candidates["positive_r_rate"].fillna(0.5) - 0.5) * 2.0)
        + 0.08 * candidates.get("hit_pos_3r_rate", 0.0).fillna(0.0)
        + 0.08 * candidates.get("hit_pos_5r_rate", 0.0).fillna(0.0)
        + 0.06 * (candidates[mfe_cap5_col].fillna(0.0).clip(0.0, 5.0) / 5.0)
        - 0.17 * candidates["hit_neg_1r_rate"].fillna(0.0)
        - 0.05 * (candidates["mean_mae_r"].fillna(0.0).clip(0.0, 5.0) / 5.0)
    )
    candidates["horizon_score"] = score

    group_value_cols = [
        "structural_validation_bucket",
        "structural_window_bucket",
        "swing_n",
        "break_mode",
        "trade_side",
        "entry_variant",
        "stop_model",
        "execution_window_label",
        "retest_volatility_regime",
        "relative_volume_bucket",
    ]
    existing_group_cols = [c for c in group_value_cols if c in candidates.columns]

    grouped = candidates.groupby("group_key", observed=True, sort=False, dropna=False)
    ranking = grouped.agg(
        valid_horizon_count=("horizon_minutes", "nunique"),
        min_valid_count=("valid_count", "min"),
        median_valid_count=("valid_count", "median"),
        mean_horizon_score=("horizon_score", "mean"),
        worst_horizon_score=("horizon_score", "min"),
        best_horizon_score=("horizon_score", "max"),
        mean_r_cap_5_avg=("mean_r_cap_5", "mean"),
        median_r_avg=("median_r", "mean"),
        q10_r_avg=("q10_r", "mean"),
        positive_r_rate_avg=("positive_r_rate", "mean"),
        hit_pos_2r_rate_avg=("hit_pos_2r_rate", "mean"),
        hit_pos_3r_rate_avg=("hit_pos_3r_rate", "mean"),
        hit_pos_5r_rate_avg=("hit_pos_5r_rate", "mean"),
        hit_neg_1r_rate_avg=("hit_neg_1r_rate", "mean"),
        mean_mfe_r_cap_5_avg=("mean_mfe_r_cap_5", "mean"),
        mean_mae_r_avg=("mean_mae_r", "mean"),
        positive_mean_horizons=("mean_r", lambda s: int((s > 0).sum())),
        positive_median_horizons=("median_r", lambda s: int((s > 0).sum())),
    ).reset_index()

    for col in existing_group_cols:
        ranking[col] = grouped[col].first().to_numpy()

    max_sample = ranking["median_valid_count"].max()
    if pd.isna(max_sample) or max_sample <= 0:
        sample_weight = 1.0
    else:
        sample_weight = np.log1p(ranking["median_valid_count"]) / np.log1p(max_sample)
    ranking["horizon_consistency_rate"] = (
        ranking["positive_mean_horizons"] / ranking["valid_horizon_count"].replace(0, np.nan)
    )
    ranking["median_consistency_rate"] = (
        ranking["positive_median_horizons"] / ranking["valid_horizon_count"].replace(0, np.nan)
    )
    ranking["rank_score"] = (
        ranking["mean_horizon_score"]
        * sample_weight
        * (0.75 + 0.25 * ranking["horizon_consistency_rate"].fillna(0.0))
    )
    ranking = ranking.sort_values(
        ["rank_score", "valid_horizon_count", "min_valid_count"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=np.int32))
    return ranking


def save_section7_outputs(
    outputs: dict[str, pd.DataFrame | pd.Series],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save reproducible Section 7 output tables."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "event_study_summary": out_dir / "section7_event_study_summary_gc.parquet",
        "candidate_signal_ranking": out_dir / "section7_candidate_signal_ranking_gc.parquet",
    }
    outputs["section7_event_study_summary"].to_parquet(
        paths["event_study_summary"],
        index=False,
    )
    outputs["section7_candidate_signal_ranking"].to_parquet(
        paths["candidate_signal_ranking"],
        index=False,
    )
    return paths


def build_section7_visual_audit_batches(
    signal_frame: pd.DataFrame,
    research_bars: pd.DataFrame,
    output_dir: str | Path,
    samples_per_batch: int = 12,
    random_state: int = 714,
    context_minutes_before: int = 90,
    context_minutes_after: int = 240,
    forward_r_column: str = "forward_60m_r",
    min_sample_size: int | None = None,
) -> pd.DataFrame:
    """Create Section 7.14 visual audit chart batches.

    The charts are for algorithm validation only. They randomly sample candidate
    rows from pre-defined audit cohorts and overlay the Section 6/7 event levels
    on the underlying 1-minute GC bars.
    """

    _validate_columns(
        signal_frame,
        [c for c in SECTION7_VISUAL_AUDIT_SIGNAL_COLUMNS if c != forward_r_column]
        + ([forward_r_column] if forward_r_column in signal_frame.columns else []),
        "signal_frame",
    )
    _validate_columns(research_bars, SECTION7_VISUAL_AUDIT_BAR_COLUMNS, "research_bars")
    if samples_per_batch <= 0:
        raise ValueError("samples_per_batch must be positive.")

    minimum = samples_per_batch if min_sample_size is None else min_sample_size
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = prepare_section7_visual_audit_bar_frame(research_bars)
    rng = np.random.default_rng(random_state)
    batch_specs = [
        {
            "batch_name": "london_short_boundary",
            "description": "London short boundary candidates",
            "mask": signal_frame["execution_window_label"].eq("London Execution")
            & signal_frame["trade_side"].eq("short")
            & signal_frame["entry_variant"].eq("boundary"),
        },
        {
            "batch_name": "new_york_short_boundary",
            "description": "New York short boundary candidates",
            "mask": signal_frame["execution_window_label"].eq("New York Execution")
            & signal_frame["trade_side"].eq("short")
            & signal_frame["entry_variant"].eq("boundary"),
        },
        {
            "batch_name": "elevated_volatility_long",
            "description": "Elevated-volatility long candidates",
            "mask": signal_frame["trade_side"].eq("long")
            & signal_frame["retest_volatility_regime"].eq("elevated"),
        },
    ]

    records: list[dict[str, Any]] = []
    for batch in batch_specs:
        candidates = signal_frame.loc[batch["mask"]].reset_index(drop=True)
        available = len(candidates)
        if available < minimum:
            records.append(
                {
                    "batch_name": batch["batch_name"],
                    "description": batch["description"],
                    "status": "skipped_insufficient_sample",
                    "available_count": available,
                    "chart_path": pd.NA,
                    "signal_id": pd.NA,
                    "candidate_trade_id": pd.NA,
                }
            )
            continue

        sample_count = min(samples_per_batch, available)
        sample_idx = rng.choice(available, size=sample_count, replace=False)
        sampled = candidates.iloc[np.sort(sample_idx)].reset_index(drop=True)
        batch_dir = out_dir / str(batch["batch_name"])
        batch_dir.mkdir(parents=True, exist_ok=True)

        for position, (_, event) in enumerate(sampled.iterrows(), start=1):
            chart_path = batch_dir / (
                f"{batch['batch_name']}_{position:02d}_{_safe_filename(event['signal_id'])}.png"
            )
            title = plot_section7_visual_audit_event(
                event=event,
                bars=bars,
                output_path=chart_path,
                context_minutes_before=context_minutes_before,
                context_minutes_after=context_minutes_after,
                forward_r_column=forward_r_column,
            )
            records.append(
                {
                    "batch_name": batch["batch_name"],
                    "description": batch["description"],
                    "status": "plotted",
                    "available_count": available,
                    "sample_position": position,
                    "signal_id": event["signal_id"],
                    "candidate_trade_id": event["candidate_trade_id"],
                    "retest_id": event["retest_id"],
                    "poi_id": event["poi_id"],
                    "chart_path": str(chart_path),
                    "title": title,
                }
            )

    return pd.DataFrame.from_records(records)


def prepare_section7_visual_audit_bar_frame(research_bars: pd.DataFrame) -> pd.DataFrame:
    """Build the GC OHLC bar frame used by the Section 7.14 visual audit."""

    _validate_columns(research_bars, SECTION7_VISUAL_AUDIT_BAR_COLUMNS, "research_bars")
    bars = (
        research_bars.loc[research_bars["product"].eq("GC"), SECTION7_VISUAL_AUDIT_BAR_COLUMNS]
        .sort_values(["trade_date_ny", "ts_event_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    bars["bar_id"] = np.arange(len(bars), dtype=np.int32)
    return bars


def plot_section7_visual_audit_event(
    event: pd.Series,
    bars: pd.DataFrame,
    output_path: str | Path,
    context_minutes_before: int = 90,
    context_minutes_after: int = 240,
    forward_r_column: str = "forward_60m_r",
) -> str:
    """Plot one Section 7 candidate event with POI, entry, stop, and targets."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    required_event_cols = [
        "signal_id",
        "retest_bar_id",
        "retest_ts_event_ny",
        "entry_price",
        "stop_price",
        "risk_points",
        "trade_side",
        "swing_n",
        "break_mode",
        "entry_variant",
        "stop_model",
        "execution_window_label",
        "poi_low",
        "poi_high",
    ]
    missing = sorted(set(required_event_cols).difference(event.index))
    if missing:
        raise KeyError(f"event is missing required columns: {missing}")

    _validate_columns(
        bars,
        [
            "bar_id",
            "ts_event_ny",
            "trade_date_ny",
            "open",
            "high",
            "low",
            "close",
            "continuous_segment_id",
        ],
        "bars",
    )

    retest_bar_id = int(event["retest_bar_id"])
    if retest_bar_id < 0 or retest_bar_id >= len(bars):
        raise IndexError(f"retest_bar_id {retest_bar_id} is outside the bar frame.")

    event_bar = bars.iloc[retest_bar_id]
    same_context = (
        bars["trade_date_ny"].eq(event_bar["trade_date_ny"])
        & bars["continuous_segment_id"].eq(event_bar["continuous_segment_id"])
    )
    window = bars.loc[
        same_context
        & bars["bar_id"].between(
            max(0, retest_bar_id - context_minutes_before),
            retest_bar_id + context_minutes_after,
            inclusive="both",
        )
    ].copy()
    if window.empty:
        raise ValueError(f"No bars available for visual audit event {event['signal_id']}.")

    x_dt = pd.to_datetime(window["ts_event_ny"]).dt.tz_localize(None)
    x = mdates.date2num(x_dt)
    candle_width = 0.70 / (24 * 60)
    up_color = "#12715b"
    down_color = "#a23a3a"

    fig, ax = plt.subplots(figsize=(15, 7))
    for x_val, open_, high, low, close in zip(
        x,
        window["open"].to_numpy("float64"),
        window["high"].to_numpy("float64"),
        window["low"].to_numpy("float64"),
        window["close"].to_numpy("float64"),
    ):
        color = up_color if close >= open_ else down_color
        ax.vlines(x_val, low, high, color=color, linewidth=0.9, alpha=0.85)
        body_low = min(open_, close)
        body_height = max(abs(close - open_), 0.01)
        ax.add_patch(
            Rectangle(
                (x_val - candle_width / 2, body_low),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.6,
                alpha=0.82,
            )
        )

    poi_low = float(event["poi_low"])
    poi_high = float(event["poi_high"])
    ax.axhspan(
        poi_low,
        poi_high,
        color="#f2a900",
        alpha=0.16,
        label="POI zone",
        zorder=0,
    )

    entry_ts = _local_naive_timestamp(event["retest_ts_event_ny"])
    entry_x = mdates.date2num(entry_ts)
    entry_price = float(event["entry_price"])
    stop_price = float(event["stop_price"])
    risk_points = float(event["risk_points"])
    long_side = str(event["trade_side"]) == "long"

    ax.scatter(
        [entry_x],
        [entry_price],
        marker="^" if long_side else "v",
        s=90,
        color="#111111",
        label="Entry",
        zorder=5,
    )
    ax.axvline(
        entry_x,
        color="#111111",
        linewidth=0.9,
        linestyle=":",
        alpha=0.70,
        label="Entry time",
    )
    ax.axhline(entry_price, color="#111111", linewidth=1.1, linestyle="-", label="Entry price")
    ax.axhline(stop_price, color="#d62728", linewidth=1.1, linestyle="--", label="Stop")

    target_colors = {1: "#2ca02c", 2: "#1f77b4", 3: "#9467bd", 5: "#ff7f0e"}
    for multiple in (1, 2, 3, 5):
        target_col = f"target_{multiple}R_price"
        if target_col in event.index and pd.notna(event[target_col]):
            target_price = float(event[target_col])
        else:
            target_price = (
                entry_price + multiple * risk_points
                if long_side
                else entry_price - multiple * risk_points
            )
        ax.axhline(
            target_price,
            color=target_colors[multiple],
            linewidth=0.9,
            linestyle=":",
            label=f"{multiple}R target",
        )

    if "poi_created_ts_event_ny" in event.index and pd.notna(event["poi_created_ts_event_ny"]):
        created_ts = _local_naive_timestamp(event["poi_created_ts_event_ny"])
        if x_dt.min() <= created_ts <= x_dt.max():
            ax.axvline(
                mdates.date2num(created_ts),
                color="#6c757d",
                linewidth=1.0,
                linestyle="-.",
                label="POI created",
            )

    forced_exit_ts = entry_ts.normalize() + pd.Timedelta(hours=15, minutes=30)
    if x_dt.min() <= forced_exit_ts <= x_dt.max():
        ax.axvline(
            mdates.date2num(forced_exit_ts),
            color="#8c564b",
            linewidth=1.1,
            linestyle="--",
            label="15:30 forced exit",
        )

    forward_r = event[forward_r_column] if forward_r_column in event.index else np.nan
    structural_status = (
        "structural"
        if bool(event.get("structural_swing_break_flag", False))
        else "local-only"
    )
    title = (
        f"{event['execution_window_label']} | {event['trade_side']} | "
        f"sw{event['swing_n']} {event['break_mode']} | "
        f"{event['entry_variant']} / {event['stop_model']} | "
        f"{structural_status} | {forward_r_column}={_fmt_float(forward_r)}"
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("GC price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, axis="y", alpha=0.22)
    ax.grid(True, axis="x", alpha=0.10)

    y_values = [
        window["low"].min(),
        window["high"].max(),
        poi_low,
        poi_high,
        entry_price,
        stop_price,
    ]
    y_values.extend(
        [
            float(event[f"target_{multiple}R_price"])
            for multiple in (1, 2, 3, 5)
            if f"target_{multiple}R_price" in event.index
            and pd.notna(event[f"target_{multiple}R_price"])
        ]
    )
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    margin = max((y_max - y_min) * 0.08, 0.5)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlim(x.min() - candle_width, x.max() + candle_width)

    handles, labels = ax.get_legend_handles_labels()
    deduped = dict(zip(labels, handles))
    ax.legend(
        deduped.values(),
        deduped.keys(),
        loc="upper left",
        fontsize=8,
        ncol=2,
        frameon=True,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return title


def build_section7_poi_selection_audit_batches(
    signal_frame: pd.DataFrame,
    research_bars: pd.DataFrame,
    output_dir: str | Path,
    samples_per_group: int = 15,
    random_state: int = 7142,
    context_bars_before: int = 40,
    context_bars_after: int = 60,
) -> pd.DataFrame:
    """Create Section 7.14B POI-selection visual audit batches.

    This audit validates whether the POI candle and zone are selected correctly.
    It intentionally does not draw entries, stops, targets, or forward outcomes.
    """

    _validate_columns(
        signal_frame,
        SECTION7_POI_SELECTION_AUDIT_SIGNAL_COLUMNS,
        "signal_frame",
    )
    _validate_columns(research_bars, SECTION7_VISUAL_AUDIT_BAR_COLUMNS, "research_bars")
    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive.")

    out_dir = Path(output_dir)
    structural_dir = out_dir / "structural"
    local_dir = out_dir / "local_only"
    structural_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    bars = prepare_section7_visual_audit_bar_frame(research_bars)
    unique_pois = (
        signal_frame.loc[:, SECTION7_POI_SELECTION_AUDIT_SIGNAL_COLUMNS]
        .sort_values(["poi_id", "signal_id"], kind="mergesort")
        .drop_duplicates("poi_id", keep="first")
        .reset_index(drop=True)
    )

    batch_specs = [
        {
            "batch_name": "structural",
            "mask": unique_pois["structural_swing_break_flag"].astype(bool),
            "output_dir": structural_dir,
        },
        {
            "batch_name": "local_only",
            "mask": unique_pois["local_swing_only_flag"].astype(bool),
            "output_dir": local_dir,
        },
    ]

    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    sample_counter = 1
    for batch in batch_specs:
        candidates = unique_pois.loc[batch["mask"]].reset_index(drop=True)
        available = len(candidates)
        if candidates.empty:
            records.append(
                _poi_selection_index_record(
                    sample_id=pd.NA,
                    event=None,
                    image_path=pd.NA,
                    batch_name=batch["batch_name"],
                    status="skipped_no_candidates",
                    available_count=available,
                )
            )
            continue

        sampled = _sample_poi_selection_candidates(
            candidates,
            sample_count=min(samples_per_group, available),
            rng=rng,
        )
        for _, event in sampled.iterrows():
            sample_id = f"POI_SEL_{sample_counter:03d}"
            file_name = (
                f"{sample_id}_{batch['batch_name']}_{_safe_filename(event['signal_id'])}.png"
            )
            image_path = Path(batch["output_dir"]) / file_name
            plot_section7_poi_selection_audit_event(
                event=event,
                bars=bars,
                output_path=image_path,
                context_bars_before=context_bars_before,
                context_bars_after=context_bars_after,
            )
            records.append(
                _poi_selection_index_record(
                    sample_id=sample_id,
                    event=event,
                    image_path=image_path,
                    batch_name=batch["batch_name"],
                    status="plotted",
                    available_count=available,
                )
            )
            sample_counter += 1

    out = pd.DataFrame.from_records(records)
    index_path = out_dir / "poi_selection_audit_index.csv"
    out.to_csv(index_path, index=False)
    return out


def plot_section7_poi_selection_audit_event(
    event: pd.Series,
    bars: pd.DataFrame,
    output_path: str | Path,
    context_bars_before: int = 40,
    context_bars_after: int = 60,
) -> str:
    """Plot one POI-selection audit event without trade overlays."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    required_event_cols = [
        "signal_id",
        "direction",
        "swing_n",
        "break_mode",
        "poi_middle_bar_id",
        "poi_confirm_bar_id",
        "break_bar_id",
        "poi_created_ts_event_ny",
        "poi_low",
        "poi_high",
        "structural_swing_break_flag",
    ]
    missing = sorted(set(required_event_cols).difference(event.index))
    if missing:
        raise KeyError(f"event is missing required columns: {missing}")
    _validate_columns(
        bars,
        [
            "bar_id",
            "ts_event_ny",
            "trade_date_ny",
            "open",
            "high",
            "low",
            "close",
            "continuous_segment_id",
        ],
        "bars",
    )

    poi_bar_id = int(event["poi_middle_bar_id"])
    confirm_bar_id = int(event["poi_confirm_bar_id"])
    if poi_bar_id < 0 or poi_bar_id >= len(bars):
        raise IndexError(f"poi_middle_bar_id {poi_bar_id} is outside the bar frame.")
    if confirm_bar_id < 0 or confirm_bar_id >= len(bars):
        confirm_bar_id = poi_bar_id

    poi_bar = bars.iloc[poi_bar_id]
    same_context = (
        bars["trade_date_ny"].eq(poi_bar["trade_date_ny"])
        & bars["continuous_segment_id"].eq(poi_bar["continuous_segment_id"])
    )
    left_bar = max(0, confirm_bar_id - context_bars_before)
    right_bar = confirm_bar_id + context_bars_after
    left_bar = min(left_bar, poi_bar_id)
    right_bar = max(right_bar, poi_bar_id)
    if "break_bar_id" in event.index and pd.notna(event["break_bar_id"]):
        break_bar_id = int(event["break_bar_id"])
        left_bar = min(left_bar, break_bar_id)
        right_bar = max(right_bar, break_bar_id)

    window = bars.loc[
        same_context & bars["bar_id"].between(left_bar, right_bar, inclusive="both")
    ].copy()
    if window.empty:
        raise ValueError(f"No bars available for POI-selection audit {event['signal_id']}.")

    x_dt = pd.to_datetime(window["ts_event_ny"]).dt.tz_localize(None)
    x = mdates.date2num(x_dt)
    candle_width = 0.70 / (24 * 60)
    up_color = "#12715b"
    down_color = "#a23a3a"

    fig, ax = plt.subplots(figsize=(13, 6))
    for x_val, open_, high, low, close in zip(
        x,
        window["open"].to_numpy("float64"),
        window["high"].to_numpy("float64"),
        window["low"].to_numpy("float64"),
        window["close"].to_numpy("float64"),
    ):
        color = up_color if close >= open_ else down_color
        ax.vlines(x_val, low, high, color=color, linewidth=0.9, alpha=0.88)
        body_low = min(open_, close)
        body_height = max(abs(close - open_), 0.01)
        ax.add_patch(
            Rectangle(
                (x_val - candle_width / 2, body_low),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.6,
                alpha=0.85,
            )
        )

    poi_low = float(event["poi_low"])
    poi_high = float(event["poi_high"])
    ax.axhspan(
        poi_low,
        poi_high,
        color="#f2a900",
        alpha=0.18,
        label="POI zone",
        zorder=0,
    )

    poi_x = mdates.date2num(_local_naive_timestamp(poi_bar["ts_event_ny"]))
    direction = str(event["direction"])
    bullish = direction == "bullish"
    arrow_price = poi_low if bullish else poi_high
    y_min = float(min(window["low"].min(), poi_low))
    y_max = float(max(window["high"].max(), poi_high))
    y_range = max(y_max - y_min, 0.5)
    marker = "^" if bullish else "v"
    ax.scatter(
        [poi_x],
        [arrow_price],
        marker=marker,
        s=110,
        color="#111111",
        label="POI selected",
        zorder=6,
    )
    ax.annotate(
        "POI selected",
        xy=(poi_x, arrow_price),
        xytext=(26, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#111111", "linewidth": 1.0},
        color="#111111",
        zorder=7,
    )

    structural_flag = bool(event.get("structural_swing_break_flag", False))
    structural_status = "structural" if structural_flag else "local-only"
    structural_window = event.get("structural_swing_window_broken", pd.NA)
    structural_window_text = (
        f"sw_window={int(structural_window)}"
        if pd.notna(structural_window)
        else "sw_window=NA"
    )
    poi_created_text = (
        _local_naive_timestamp(event["poi_created_ts_event_ny"]).strftime("%Y-%m-%d %H:%M")
        if pd.notna(event["poi_created_ts_event_ny"])
        else "NA"
    )
    session = (
        str(event["execution_window_label"])
        if "execution_window_label" in event.index
        and pd.notna(event["execution_window_label"])
        else "session=NA"
    )
    title = (
        f"{event['signal_id']} | {direction} | sw{event['swing_n']} {event['break_mode']} | "
        f"{structural_status} {structural_window_text} | POI created {poi_created_text} | "
        f"{session}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("GC price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, axis="y", alpha=0.22)
    ax.grid(True, axis="x", alpha=0.10)
    ax.set_xlim(x.min() - candle_width, x.max() + candle_width)
    margin = y_range * 0.10
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return title


def _summarize_group_frame(
    frame: pd.DataFrame,
    group_cols: list[str],
    config: Section7Config,
) -> pd.DataFrame:
    metrics = _summary_metric_columns(config)
    if not group_cols:
        summary = _single_group_summary(frame, metrics)
        summary.insert(0, "group_key", "all")
        return summary

    grouped = frame.groupby(group_cols, observed=True, sort=False, dropna=False)
    agg_dict: dict[str, tuple[str, str]] = {
        "count": ("valid", "size"),
        "candidate_rows": ("valid", "size"),
        "valid_count": ("valid", "sum"),
        "mean_r": ("forward_r", "mean"),
        "median_r": ("forward_r", "median"),
        "std_r": ("forward_r", "std"),
        "mean_abs_r": ("forward_r_abs", "mean"),
        "mean_mfe_r": ("mfe_r", "mean"),
        "median_mfe_r": ("mfe_r", "median"),
        "mean_mae_r": ("mae_r", "mean"),
        "median_mae_r": ("mae_r", "median"),
        "forced_exit_rate": ("forced_exit_flag", "mean"),
    }
    optional_cardinalities = {
        "unique_candidate_variants": "candidate_variant_id",
        "unique_canonical_retests": "canonical_retest_id",
        "unique_canonical_pois": "canonical_poi_id",
        "unique_trading_dates": "trade_date_ny",
    }
    for output_col, source_col in optional_cardinalities.items():
        if source_col in frame.columns:
            agg_dict[output_col] = (source_col, "nunique")
    for col in metrics["capped_forward"]:
        agg_dict[f"mean_{col.replace('forward_', '')}"] = (col, "mean")
    for col in metrics["capped_mfe"]:
        agg_dict[f"mean_{col}"] = (col, "mean")
    for col in metrics["rates"]:
        agg_dict[f"{col}_rate"] = (col, "mean")

    summary = grouped.agg(**agg_dict).reset_index()

    quantiles = grouped["forward_r"].quantile([0.10, 0.25, 0.75, 0.90]).unstack()
    quantiles.columns = ["q10_r", "q25_r", "q75_r", "q90_r"]
    quantiles = quantiles.reset_index()
    summary = summary.merge(quantiles, on=group_cols, how="left", validate="one_to_one")

    summary["positive_r_rate"] = grouped["forward_r"].apply(
        lambda s: s.dropna().gt(0).mean()
    ).to_numpy()
    summary["negative_r_rate"] = grouped["forward_r"].apply(
        lambda s: s.dropna().lt(0).mean()
    ).to_numpy()
    summary["t_stat"] = np.where(
        (summary["valid_count"] > 1) & summary["std_r"].gt(0),
        summary["mean_r"] / (summary["std_r"] / np.sqrt(summary["valid_count"])),
        np.nan,
    )
    summary.insert(0, "group_key", _format_group_key(summary, group_cols))
    return summary


def _single_group_summary(frame: pd.DataFrame, metrics: dict[str, list[str]]) -> pd.DataFrame:
    valid_count = int(frame["valid"].sum())
    forward_r = frame["forward_r"]
    row: dict[str, Any] = {
        "count": len(frame),
        "candidate_rows": len(frame),
        "valid_count": valid_count,
        "mean_r": forward_r.mean(),
        "median_r": forward_r.median(),
        "std_r": forward_r.std(),
        "mean_abs_r": forward_r.abs().mean(),
        "mean_mfe_r": frame["mfe_r"].mean(),
        "median_mfe_r": frame["mfe_r"].median(),
        "mean_mae_r": frame["mae_r"].mean(),
        "median_mae_r": frame["mae_r"].median(),
        "forced_exit_rate": frame["forced_exit_flag"].mean(),
        "q10_r": forward_r.quantile(0.10),
        "q25_r": forward_r.quantile(0.25),
        "q75_r": forward_r.quantile(0.75),
        "q90_r": forward_r.quantile(0.90),
        "positive_r_rate": forward_r.dropna().gt(0).mean(),
        "negative_r_rate": forward_r.dropna().lt(0).mean(),
    }
    optional_cardinalities = {
        "unique_candidate_variants": "candidate_variant_id",
        "unique_canonical_retests": "canonical_retest_id",
        "unique_canonical_pois": "canonical_poi_id",
        "unique_trading_dates": "trade_date_ny",
    }
    for output_col, source_col in optional_cardinalities.items():
        if source_col in frame.columns:
            row[output_col] = frame[source_col].nunique()
    for col in metrics["capped_forward"]:
        row[f"mean_{col.replace('forward_', '')}"] = frame[col].mean()
    for col in metrics["capped_mfe"]:
        row[f"mean_{col}"] = frame[col].mean()
    for col in metrics["rates"]:
        row[f"{col}_rate"] = frame[col].mean()
    std = row["std_r"]
    row["t_stat"] = (
        row["mean_r"] / (std / np.sqrt(valid_count))
        if valid_count > 1 and pd.notna(std) and std > 0
        else np.nan
    )
    return pd.DataFrame([row])


def _summary_metric_columns(config: Section7Config) -> dict[str, list[str]]:
    capped_forward = [f"forward_r_cap_{_threshold_label(cap)}" for cap in config.r_caps]
    capped_mfe = [f"mfe_r_cap_{_threshold_label(cap)}" for cap in config.r_caps]
    rates = ["hit_neg_1r"]
    for threshold in config.runner_thresholds:
        label = _threshold_label(threshold)
        rates.extend(
            [
                f"hit_pos_{label}r",
                f"entry_bar_ambiguous_{label}r",
                f"path_order_ambiguous_{label}r",
            ]
        )
    return {
        "capped_forward": capped_forward,
        "capped_mfe": capped_mfe,
        "rates": rates,
    }


def _horizon_quality_record(
    metrics: pd.DataFrame,
    horizon: int,
    metric_mode: str,
) -> dict[str, Any]:
    return {
        "horizon_minutes": horizon,
        "metric_mode": metric_mode,
        "count": len(metrics),
        "valid_count": int(metrics["valid"].sum()),
        "valid_rate": float(metrics["valid"].mean()),
        "forced_exit_rate": float(metrics["forced_exit_flag"].mean()),
    }


def _sample_poi_selection_candidates(
    candidates: pd.DataFrame,
    sample_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if sample_count >= len(candidates):
        return candidates.sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1)))

    direction_counts = candidates["direction"].value_counts()
    if {"bullish", "bearish"}.issubset(set(direction_counts.index)) and sample_count >= 2:
        base_each = sample_count // 2
        allocations = {
            "bullish": min(base_each, int(direction_counts["bullish"])),
            "bearish": min(sample_count - base_each, int(direction_counts["bearish"])),
        }
        shortfall = sample_count - sum(allocations.values())
        if shortfall > 0:
            for direction in ("bullish", "bearish"):
                available_extra = int(direction_counts[direction]) - allocations[direction]
                take = min(shortfall, available_extra)
                allocations[direction] += take
                shortfall -= take
                if shortfall == 0:
                    break

        sampled_parts = []
        for direction, count in allocations.items():
            if count <= 0:
                continue
            group = candidates.loc[candidates["direction"].eq(direction)]
            sampled_parts.append(
                group.sample(
                    n=count,
                    replace=False,
                    random_state=int(rng.integers(0, 2**32 - 1)),
                )
            )
        sampled = pd.concat(sampled_parts, ignore_index=True)
        if len(sampled) < sample_count:
            remainder = candidates.loc[~candidates["poi_id"].isin(sampled["poi_id"])]
            sampled = pd.concat(
                [
                    sampled,
                    remainder.sample(
                        n=sample_count - len(sampled),
                        replace=False,
                        random_state=int(rng.integers(0, 2**32 - 1)),
                    ),
                ],
                ignore_index=True,
            )
        return sampled.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 2**32 - 1)),
        ).reset_index(drop=True)

    return candidates.sample(
        n=sample_count,
        replace=False,
        random_state=int(rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True)


def _poi_selection_index_record(
    *,
    sample_id: Any,
    event: pd.Series | None,
    image_path: Any,
    batch_name: str,
    status: str,
    available_count: int,
) -> dict[str, Any]:
    if event is None:
        return {
            "sample_id": sample_id,
            "signal_id": pd.NA,
            "image_path": image_path,
            "poi_time": pd.NA,
            "poi_low": np.nan,
            "poi_high": np.nan,
            "direction": pd.NA,
            "swing_n": pd.NA,
            "break_mode": pd.NA,
            "structural_validation_flag": pd.NA,
            "structural_window": pd.NA,
            "session_label": pd.NA,
            "manual_review_status": "",
            "manual_notes": "",
            "batch_name": batch_name,
            "status": status,
            "available_count": available_count,
        }

    structural_window = event.get("structural_swing_window_broken", pd.NA)
    return {
        "sample_id": sample_id,
        "signal_id": event["signal_id"],
        "image_path": str(image_path),
        "poi_time": event.get("poi_middle_ts_event_ny", event.get("poi_created_ts_event_ny", pd.NA)),
        "poi_low": event["poi_low"],
        "poi_high": event["poi_high"],
        "direction": event["direction"],
        "swing_n": event["swing_n"],
        "break_mode": event["break_mode"],
        "structural_validation_flag": bool(event.get("structural_swing_break_flag", False)),
        "structural_window": structural_window if pd.notna(structural_window) else pd.NA,
        "session_label": event.get("execution_window_label", pd.NA),
        "manual_review_status": "",
        "manual_notes": "",
        "batch_name": batch_name,
        "status": status,
        "available_count": available_count,
    }


def _format_group_key(summary: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(["all"] * len(summary), index=summary.index, dtype="string")
    parts = []
    for col in group_cols:
        parts.append(col + "=" + summary[col].astype("string").fillna("missing"))
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key.astype("string")


def _build_sparse_table(values: np.ndarray, op: str) -> list[np.ndarray]:
    if op not in {"max", "min"}:
        raise ValueError("op must be 'max' or 'min'")
    table = [np.asarray(values)]
    step = 1
    while step * 2 <= len(values):
        prev = table[-1]
        reducer = np.maximum if op == "max" else np.minimum
        table.append(reducer(prev[:-step], prev[step:]))
        step *= 2
    return table


def _query_sparse_table(
    table: list[np.ndarray],
    left: np.ndarray,
    right: np.ndarray,
    op: str,
) -> np.ndarray:
    if left.size == 0:
        return np.array([], dtype="float64")
    length = right - left + 1
    levels = np.floor(np.log2(length)).astype("int64")
    out = np.empty(left.size, dtype="float64")
    reducer = np.maximum if op == "max" else np.minimum
    for level in np.unique(levels):
        mask = levels == level
        width = 1 << int(level)
        segment = table[int(level)]
        out[mask] = reducer(
            segment[left[mask]],
            segment[right[mask] - width + 1],
        )
    return out


def _cut_numeric(
    values: pd.Series,
    bins: list[float],
    labels: list[str],
) -> pd.Series:
    return (
        pd.cut(values.astype("float64"), bins=bins, labels=labels, include_lowest=True)
        .astype("object")
        .fillna("unknown")
    )


def _trade_alignment(
    trade_side: pd.Series,
    bullish_condition: pd.Series,
    bearish_condition: pd.Series,
    neutral_condition: pd.Series,
) -> np.ndarray:
    long_side = trade_side.eq("long").fillna(False)
    short_side = trade_side.eq("short").fillna(False)
    bullish = bullish_condition.fillna(False)
    bearish = bearish_condition.fillna(False)
    neutral = neutral_condition.fillna(False)
    aligned = (long_side & bullish) | (short_side & bearish)
    against = (long_side & bearish) | (short_side & bullish)
    return np.select(
        [
            aligned.to_numpy(bool),
            against.to_numpy(bool),
            neutral.to_numpy(bool),
        ],
        ["aligned", "against", "neutral"],
        default="unknown",
    )


def _threshold_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def _datetime64_ns(values: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(values, utc=True)
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )


def _local_naive_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def _fmt_float(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def _safe_filename(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:120]


def _validate_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    missing = sorted(set(required_columns).difference(df.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _raise_if_missing_inputs(signal_frame: pd.DataFrame, research_bars: pd.DataFrame) -> None:
    _validate_columns(signal_frame, SECTION7_SIGNAL_REQUIRED_COLUMNS, "signal_frame")
    _validate_columns(research_bars, SECTION7_BAR_REQUIRED_COLUMNS, "research_bars")
