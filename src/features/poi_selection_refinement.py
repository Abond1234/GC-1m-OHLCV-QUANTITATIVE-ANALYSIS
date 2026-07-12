"""Section 6C POI-selection refinement for the GC research workflow.

This module is deliberately non-destructive.  It re-evaluates Section 6
three-candle POI proposals against the refined formation contract, rebuilds
all geometry-dependent downstream objects, and writes only ``section6c_*``
artifacts.  Section 6 Version A and Section 6B Version B remain historical
research outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.features.poi_signal_features import (
    STRUCTURAL_COLUMNS,
    Section6Config,
    annotate_poi_structural_validation,
    build_candidate_trade_table,
    build_poi_table,
    build_retest_table,
    build_signal_frame,
    build_structural_swing_break_events,
    prepare_section6_feature_frame,
)


@dataclass(frozen=True)
class Section6CConfig(Section6Config):
    """Configuration for the refined Version C definition."""

    min_fvg_ticks: float = 3.0
    min_close_open_gap_ticks: float = 1.0
    poi_definition_version: str = "6C_v1"


SECTION6C_OUTPUT_FILENAMES = {
    "refined_poi_audit": "section6c_refined_poi_audit_gc.parquet",
    "refined_poi_table": "section6c_refined_poi_table_gc.parquet",
    "refined_retest_table": "section6c_refined_retest_table_gc.parquet",
    "refined_candidate_trade_table": "section6c_refined_candidate_trade_table_gc.parquet",
    "refined_signal_frame": "section6c_refined_signal_frame_gc.parquet",
    "refinement_summary": "section6c_refinement_summary_gc.parquet",
}


HAND_LABELLED_REGRESSION_CASES: tuple[dict[str, Any], ...] = (
    {
        "signal_id": "GC_SIG_00254316",
        "expected_classification": "case_1_expanded",
        "expected_formation_validity": True,
        "expected_rejection_reason": "valid",
    },
    {
        "signal_id": "GC_SIG_00330917",
        "expected_classification": "case_1_expanded",
        "expected_formation_validity": True,
        "expected_rejection_reason": "valid",
    },
    {
        "signal_id": "GC_SIG_01516818",
        "expected_classification": "case_1_expanded",
        "expected_formation_validity": True,
        "expected_rejection_reason": "valid",
    },
    {
        "signal_id": "GC_SIG_00136776",
        "expected_classification": "case_2_standard",
        "expected_formation_validity": True,
        "expected_rejection_reason": "valid",
    },
    {
        "signal_id": "GC_SIG_00066187",
        "expected_classification": "case_2_standard",
        "expected_formation_validity": True,
        "expected_rejection_reason": "valid",
    },
    {
        "signal_id": "GC_SIG_01011397",
        "expected_classification": "case_2_standard",
        "expected_formation_validity": True,
        "expected_rejection_reason": "valid",
    },
    {
        "signal_id": "GC_SIG_00173275",
        "expected_classification": "case_2_standard",
        "expected_formation_validity": True,
        "expected_rejection_reason": "valid",
    },
    {
        "signal_id": "GC_SIG_00537020",
        "expected_classification": "case_2_standard",
        "expected_formation_validity": True,
        "expected_rejection_reason": "valid",
    },
    {
        "signal_id": "GC_SIG_00246472",
        "expected_classification": "case_2_standard",
        "expected_formation_validity": True,
        "expected_rejection_reason": "valid",
    },
    {
        "signal_id": "GC_SIG_00258361",
        "expected_classification": "case_3_invalid",
        "expected_formation_validity": False,
        "expected_rejection_reason": "confirmation_close_inside_opening_gap",
    },
    {
        "signal_id": "GC_SIG_00886808",
        "expected_classification": "case_3_invalid",
        "expected_formation_validity": False,
        "expected_rejection_reason": "confirmation_close_inside_opening_gap",
    },
    {
        "signal_id": "GC_SIG_00775734",
        "expected_classification": "case_3_invalid",
        "expected_formation_validity": False,
        "expected_rejection_reason": "confirmation_close_inside_opening_gap",
    },
    {
        "signal_id": "GC_SIG_00433067",
        "expected_classification": "case_3_invalid",
        "expected_formation_validity": False,
        "expected_rejection_reason": "confirmation_close_inside_opening_gap",
    },
    {
        "signal_id": "GC_SIG_00656876",
        "expected_classification": "case_3_invalid",
        "expected_formation_validity": False,
        "expected_rejection_reason": "confirmation_close_inside_opening_gap",
    },
    {
        "signal_id": "GC_SIG_00088931",
        "expected_classification": "case_3_invalid",
        "expected_formation_validity": False,
        "expected_rejection_reason": "confirmation_close_inside_opening_gap",
    },
    {
        "signal_id": "GC_SIG_01285931",
        "expected_classification": "case_3_invalid",
        "expected_formation_validity": False,
        "expected_rejection_reason": "confirmation_close_inside_opening_gap",
    },
    {
        "signal_id": "GC_SIG_00098666",
        "expected_classification": "case_3_invalid",
        "expected_formation_validity": False,
        "expected_rejection_reason": "confirmation_close_inside_opening_gap",
    },
    {
        "signal_id": "GC_SIG_00384676",
        "expected_classification": "invalid_no_fvg",
        "expected_formation_validity": False,
        "expected_rejection_reason": "no_classic_fvg",
    },
    {
        "signal_id": "GC_SIG_01388071",
        "expected_classification": "invalid_no_fvg",
        "expected_formation_validity": False,
        "expected_rejection_reason": "no_classic_fvg",
    },
)


REFINEMENT_REQUIRED_BAR_COLUMNS = {
    "bar_id",
    "ts_event_utc",
    "ts_event_ny",
    "trade_date_ny",
    "product",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "continuous_segment_id",
    "tradable_research_flag",
    "roll_window_flag",
    "is_regular_1m_bar",
    "same_symbol_as_previous_bar",
    "is_poi_search_window",
    "minute_of_day_ny",
}


def price_to_tick_index(values: Any, tick_size: float = 0.10) -> np.ndarray:
    """Convert GC prices to nearest integer ticks without equality checks."""

    arr = np.asarray(values, dtype="float64")
    return np.rint(arr / tick_size).astype("int64")


def fvg_tick_size(
    *,
    prior_price: Any,
    confirmation_price: Any,
    direction: str,
    tick_size: float = 0.10,
) -> np.ndarray:
    """Return an active-direction FVG size using integer tick-space prices."""

    prior_ticks = price_to_tick_index(prior_price, tick_size)
    confirmation_ticks = price_to_tick_index(confirmation_price, tick_size)
    if direction == "bullish":
        return confirmation_ticks - prior_ticks
    if direction == "bearish":
        return prior_ticks - confirmation_ticks
    raise ValueError("direction must be 'bullish' or 'bearish'")


def evaluate_refined_poi_proposals(
    feature_frame: pd.DataFrame,
    proposals: pd.DataFrame,
    config: Section6CConfig | None = None,
    *,
    audit_record_type: str = "variant_proposal",
) -> pd.DataFrame:
    """Vectorize the exact Section 6C A/B/C formation contract.

    ``proposals`` must identify the middle candle with ``poi_middle_bar_id``
    and the active ``direction``.  Variant proposals may also carry the local
    swing/break lineage produced by Section 6.  The input frame must use the
    global positional ``bar_id`` created by ``prepare_section6_feature_frame``.
    """

    cfg = config or Section6CConfig()
    _validate_columns(feature_frame, REFINEMENT_REQUIRED_BAR_COLUMNS, "feature_frame")
    _validate_columns(proposals, {"poi_middle_bar_id", "direction"}, "proposals")
    if proposals.empty:
        return proposals.copy()

    bars = feature_frame.sort_values("bar_id", kind="mergesort").reset_index(drop=True)
    bar_ids = bars["bar_id"].to_numpy("int64", copy=False)
    if not np.array_equal(bar_ids, np.arange(len(bars), dtype="int64")):
        raise ValueError("feature_frame.bar_id must be a zero-based contiguous positional index")

    out = proposals.copy().reset_index(drop=True)
    middle = out["poi_middle_bar_id"].to_numpy("int64", copy=False)
    if (middle <= 0).any() or (middle >= len(bars) - 1).any():
        raise IndexError("Every POI proposal must have both a prior and confirmation bar")
    prior = middle - 1
    confirm = middle + 1
    activation = out.get("poi_activation_bar_id", pd.Series(confirm, index=out.index))
    activation = pd.to_numeric(activation, errors="coerce").fillna(pd.Series(confirm)).to_numpy(
        "int64"
    )
    activation = np.maximum(activation, confirm)
    if (activation < 0).any() or (activation >= len(bars)).any():
        raise IndexError("poi_activation_bar_id is outside the feature frame")

    open_ = bars["open"].to_numpy("float64", copy=False)
    high = bars["high"].to_numpy("float64", copy=False)
    low = bars["low"].to_numpy("float64", copy=False)
    close = bars["close"].to_numpy("float64", copy=False)
    price_ticks = price_to_tick_index(np.column_stack([open_, high, low, close]), cfg.tick_size)
    open_ticks, high_ticks, low_ticks, close_ticks = price_ticks.T
    bullish = out["direction"].eq("bullish").to_numpy(bool, copy=False)
    bearish = out["direction"].eq("bearish").to_numpy(bool, copy=False)
    if not (bullish | bearish).all():
        raise ValueError("Proposal directions must be 'bullish' or 'bearish'")

    bull_fvg_points = low[confirm] - high[prior]
    bear_fvg_points = low[prior] - high[confirm]
    bull_fvg_ticks = low_ticks[confirm] - high_ticks[prior]
    bear_fvg_ticks = low_ticks[prior] - high_ticks[confirm]
    fvg_points = np.where(bullish, bull_fvg_points, bear_fvg_points)
    fvg_ticks = np.where(bullish, bull_fvg_ticks, bear_fvg_ticks).astype("int32")
    classic_fvg = np.where(bullish, bull_fvg_points > 0, bear_fvg_points > 0)
    fvg_low = np.where(bullish, high[prior], high[confirm])
    fvg_high = np.where(bullish, low[confirm], low[prior])

    bull_gap_points = open_[confirm] - close[middle]
    bear_gap_points = close[middle] - open_[confirm]
    bull_gap_ticks = open_ticks[confirm] - close_ticks[middle]
    bear_gap_ticks = close_ticks[middle] - open_ticks[confirm]
    gap_points = np.where(bullish, bull_gap_points, bear_gap_points)
    gap_ticks = np.where(bullish, bull_gap_ticks, bear_gap_ticks).astype("int32")
    close_open_gap = gap_ticks >= int(round(cfg.min_close_open_gap_ticks))
    gap_low = np.minimum(close[middle], open_[confirm])
    gap_high = np.maximum(close[middle], open_[confirm])

    body_direction = np.select(
        [close[confirm] > open_[confirm], close[confirm] < open_[confirm]],
        ["bullish", "bearish"],
        default="doji",
    )
    close_inside_gap = np.where(
        bullish,
        close[confirm] < open_[confirm],
        close[confirm] > open_[confirm],
    )
    gap_fully_closed = np.where(
        bullish,
        close[confirm] <= close[middle],
        close[confirm] >= close[middle],
    )
    gap_fill_points = np.where(
        bullish,
        open_[confirm] - close[confirm],
        close[confirm] - open_[confirm],
    )
    gap_fill_fraction = np.divide(
        np.maximum(gap_fill_points, 0.0),
        gap_points,
        out=np.full(len(out), np.nan),
        where=gap_points > 0,
    )
    gap_fill_fraction = np.clip(gap_fill_fraction, 0.0, 1.0)
    opening_gap_survives = close_open_gap & ~close_inside_gap

    symbols = bars["symbol"].astype("string").to_numpy()
    products = bars["product"].astype("string").to_numpy()
    segments = bars["continuous_segment_id"].to_numpy()
    dates = bars["trade_date_ny"].to_numpy()
    ts_ns = (
        pd.to_datetime(bars["ts_event_utc"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )
    same_product = (products[prior] == products[middle]) & (products[middle] == products[confirm])
    same_contract = (symbols[prior] == symbols[middle]) & (symbols[middle] == symbols[confirm])
    same_segment = (segments[prior] == segments[middle]) & (segments[middle] == segments[confirm])
    same_date = (dates[prior] == dates[middle]) & (dates[middle] == dates[confirm])
    timestamp_continuity = (
        (ts_ns[middle] - ts_ns[prior] == 60_000_000_000)
        & (ts_ns[confirm] - ts_ns[middle] == 60_000_000_000)
    )
    regular = bars["is_regular_1m_bar"].to_numpy(bool, copy=False)
    prior_symbol_match = bars["same_symbol_as_previous_bar"].to_numpy(bool, copy=False)
    exact_three_bar_continuity = (
        same_product
        & same_contract
        & same_segment
        & same_date
        & timestamp_continuity
        & regular[middle]
        & regular[confirm]
        & prior_symbol_match[middle]
        & prior_symbol_match[confirm]
    )

    tradable = bars["tradable_research_flag"].to_numpy(bool, copy=False)
    roll = bars["roll_window_flag"].to_numpy(bool, copy=False)
    rollover_integrity = (
        tradable[prior]
        & tradable[middle]
        & tradable[confirm]
        & tradable[activation]
        & ~roll[prior]
        & ~roll[middle]
        & ~roll[confirm]
        & ~roll[activation]
    )
    search = bars["is_poi_search_window"].to_numpy(bool, copy=False)
    valid_search_window = search[confirm] & search[activation]
    direction_match = np.ones(len(out), dtype=bool)

    formation_valid = (
        exact_three_bar_continuity
        & classic_fvg
        & close_open_gap
        & direction_match
        & opening_gap_survives
        & valid_search_window
        & rollover_integrity
    )
    threshold_pass = fvg_ticks >= int(round(cfg.min_fvg_ticks))
    final_valid = formation_valid & threshold_pass

    expanded = np.where(
        bullish,
        open_[confirm] > high[middle],
        open_[confirm] < low[middle],
    )
    base_geometry_case = np.where(expanded, "case_1_expanded", "case_2_standard")
    geometry_case = np.where(close_inside_gap & close_open_gap, "case_3_invalid", base_geometry_case)
    geometry_case = np.where(classic_fvg & close_open_gap, geometry_case, "invalid")
    poi_low = np.where(bullish, low[middle], np.minimum(low[middle], open_[confirm]))
    poi_high = np.where(bullish, np.maximum(high[middle], open_[confirm]), high[middle])
    expansion_points = np.where(
        bullish,
        np.maximum(open_[confirm] - high[middle], 0.0),
        np.maximum(low[middle] - open_[confirm], 0.0),
    )
    expansion_ticks = np.where(
        bullish,
        np.maximum(open_ticks[confirm] - high_ticks[middle], 0),
        np.maximum(low_ticks[middle] - open_ticks[confirm], 0),
    ).astype("int32")

    reasons = np.full(len(out), "valid", dtype=object)
    details = np.full(len(out), "formation and default threshold passed", dtype=object)
    reason_masks = [
        (~same_contract, "contract_switch_crossed", "A/B/C symbols are not identical"),
        (~same_segment, "segment_boundary_crossed", "A/B/C continuous_segment_id values differ"),
        (
            ~(same_date & timestamp_continuity & regular[middle] & regular[confirm]),
            "non_consecutive_bars",
            "A/B/C are not consecutive one-minute observations on one NY date",
        ),
        (~rollover_integrity, "rollover_integrity_failure", "A/B/C or activation failed tradable/roll integrity"),
        (~valid_search_window, "invalid_search_window", "confirmation or activation is outside the POI search window"),
        (~classic_fvg, "no_classic_fvg", "active-direction A/C wick interval is not a positive classic FVG"),
        (~close_open_gap, "no_close_open_gap", "B-close/C-open gap is absent, reversed, or below one tick"),
        (~direction_match, "direction_mismatch", "formation direction does not match the displacement break"),
        (
            close_inside_gap & close_open_gap,
            "confirmation_close_inside_opening_gap",
            "confirmation candle C closed against its opening-gap direction",
        ),
        (
            formation_valid & ~threshold_pass,
            f"fvg_below_{int(round(cfg.min_fvg_ticks))}_ticks",
            f"formation is correct but FVG is below {int(round(cfg.min_fvg_ticks))} ticks",
        ),
    ]
    # Apply in reverse so the first item above has the highest precedence.
    for mask, reason, detail in reversed(reason_masks):
        reasons[mask] = reason
        details[mask] = detail

    atr = bars.get("rolling_atr_60m", pd.Series(np.nan, index=bars.index)).to_numpy(
        "float64", copy=False
    )
    atr_ratio = np.divide(
        fvg_points,
        atr[middle],
        out=np.full(len(out), np.nan),
        where=np.isfinite(atr[middle]) & (atr[middle] > 0),
    )

    out["audit_record_type"] = audit_record_type
    out["prior_bar_id"] = prior.astype("int32")
    out["poi_bar_id"] = middle.astype("int32")
    out["confirmation_bar_id"] = confirm.astype("int32")
    out["poi_confirm_bar_id"] = confirm.astype("int32")
    out["activation_bar_id"] = activation.astype("int32")
    out["poi_activation_bar_id"] = activation.astype("int32")
    if "break_bar_id" in out:
        out["swing_break_bar_id"] = out["break_bar_id"].astype("Int64")
    elif "swing_break_bar_id" not in out:
        out["swing_break_bar_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    for prefix, pos in (("prior", prior), ("poi", middle), ("confirmation", confirm)):
        out[f"{prefix}_ts_event_utc"] = bars["ts_event_utc"].to_numpy()[pos]
        out[f"{prefix}_ts_event_ny"] = bars["ts_event_ny"].to_numpy()[pos]
        out[f"{prefix}_symbol"] = symbols[pos]
        out[f"{prefix}_continuous_segment_id"] = segments[pos]
        out[f"{prefix}_open"] = open_[pos]
        out[f"{prefix}_high"] = high[pos]
        out[f"{prefix}_low"] = low[pos]
        out[f"{prefix}_close"] = close[pos]

    out["middle_bar_id"] = middle.astype("int32")
    out["middle_ts_event_utc"] = bars["ts_event_utc"].to_numpy()[middle]
    out["middle_ts_event_ny"] = bars["ts_event_ny"].to_numpy()[middle]
    out["middle_symbol"] = symbols[middle]
    out["middle_continuous_segment_id"] = segments[middle]
    out["middle_open"] = open_[middle]
    out["middle_high"] = high[middle]
    out["middle_low"] = low[middle]
    out["middle_close"] = close[middle]
    out["prev_candle_high"] = high[prior]
    out["prev_candle_low"] = low[prior]
    out["next_candle_high"] = high[confirm]
    out["next_candle_low"] = low[confirm]

    out["activation_ts_event_utc"] = bars["ts_event_utc"].to_numpy()[activation]
    out["activation_ts_event_ny"] = bars["ts_event_ny"].to_numpy()[activation]
    out["confirmation_body_direction"] = pd.Categorical(
        body_direction, categories=["bearish", "doji", "bullish"]
    )
    out["confirmation_close_inside_gap_flag"] = close_inside_gap
    out["confirmation_gap_fully_closed_flag"] = gap_fully_closed & close_open_gap
    out["confirmation_gap_fill_fraction"] = gap_fill_fraction
    out["opening_gap_survives_close_flag"] = opening_gap_survives
    out["classic_fvg_flag"] = classic_fvg
    out["fvg_low"] = fvg_low
    out["fvg_high"] = fvg_high
    out["fvg_size_points"] = fvg_points
    out["fvg_size_ticks"] = fvg_ticks
    out["fvg_size_atr_ratio"] = atr_ratio
    out["fvg_ge_3tick_flag"] = fvg_ticks >= 3
    out["fvg_ge_4tick_flag"] = fvg_ticks >= 4
    out["fvg_ge_5tick_flag"] = fvg_ticks >= 5
    out["close_open_gap_flag"] = close_open_gap
    out["close_open_gap_low"] = gap_low
    out["close_open_gap_high"] = gap_high
    out["close_open_gap_points"] = gap_points
    out["close_open_gap_ticks"] = gap_ticks
    out["abs_close_open_gap_ticks"] = np.abs(gap_ticks).astype("int32")
    out["same_product_flag"] = same_product
    out["same_contract_flag"] = same_contract
    out["same_segment_flag"] = same_segment
    out["one_minute_continuity_flag"] = timestamp_continuity
    out["exact_three_bar_continuity_flag"] = exact_three_bar_continuity
    out["rollover_integrity_flag"] = rollover_integrity
    out["valid_search_window_flag"] = valid_search_window
    out["matching_displacement_direction_flag"] = direction_match
    out["base_poi_geometry_case"] = pd.Categorical(
        base_geometry_case, categories=["case_1_expanded", "case_2_standard"]
    )
    out["poi_geometry_case"] = pd.Categorical(
        geometry_case,
        categories=["case_1_expanded", "case_2_standard", "case_3_invalid", "invalid"],
    )
    out["poi_zone_expanded_flag"] = expanded & classic_fvg & close_open_gap
    out["poi_zone_expansion_points"] = expansion_points
    out["poi_zone_expansion_ticks"] = expansion_ticks
    out["poi_low"] = poi_low
    out["poi_high"] = poi_high
    out["poi_mid"] = (poi_low + poi_high) / 2.0
    out["poi_size_ticks"] = (
        price_to_tick_index(poi_high, cfg.tick_size) - price_to_tick_index(poi_low, cfg.tick_size)
    ).astype("int32")
    out["formation_valid_flag"] = formation_valid
    out["fvg_threshold_pass"] = threshold_pass
    out["final_poi_valid_flag"] = final_valid
    out["rejection_reason"] = pd.Categorical(reasons)
    out["rejection_reason_detail"] = details
    out["poi_definition_version"] = cfg.poi_definition_version
    out["canonical_poi_id"] = _canonical_poi_ids(out)
    if "poi_id" in out:
        out["poi_variant_id"] = out["poi_id"].astype("string")
    elif "poi_variant_id" not in out:
        out["poi_variant_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["audit_id"] = np.where(
        out["poi_variant_id"].notna(),
        out["poi_variant_id"].astype("string") + "_audit",
        "GC_AUD_" + out["direction"].astype("string") + "_" + out["poi_middle_bar_id"].astype(str).str.zfill(8),
    )
    return out


def build_supplemental_refinement_audit(
    feature_frame: pd.DataFrame,
    config: Section6CConfig | None = None,
) -> pd.DataFrame:
    """Retain gap-only and FVG-only formations for false-positive diagnosis.

    These rows are diagnostic physical formations rather than local-swing
    variants, so swing/break lineage is intentionally nullable.  They ensure
    that ``no_classic_fvg`` and ``no_close_open_gap`` rejections remain
    inspectable without multiplying them across correlated 3/5/7 and
    wick/close variants.
    """

    cfg = config or Section6CConfig()
    _validate_columns(feature_frame, REFINEMENT_REQUIRED_BAR_COLUMNS, "feature_frame")
    bars = feature_frame.sort_values("bar_id", kind="mergesort").reset_index(drop=True)
    n = len(bars)
    if n < 3:
        return pd.DataFrame()

    prices = price_to_tick_index(bars[["open", "high", "low", "close"]].to_numpy(), cfg.tick_size)
    open_ticks, high_ticks, low_ticks, close_ticks = prices.T
    middle = np.arange(1, n - 1, dtype="int64")
    prior = middle - 1
    confirm = middle + 1
    bull_fvg = low_ticks[confirm] - high_ticks[prior]
    bear_fvg = low_ticks[prior] - high_ticks[confirm]
    bull_gap = open_ticks[confirm] - close_ticks[middle]
    bear_gap = close_ticks[middle] - open_ticks[confirm]

    search = bars["is_poi_search_window"].to_numpy(bool, copy=False)
    in_scope = search[confirm]
    selections = [
        ("bullish", middle[in_scope & (bull_gap >= 1) & (bull_fvg <= 0)]),
        ("bearish", middle[in_scope & (bear_gap >= 1) & (bear_fvg <= 0)]),
        ("bullish", middle[in_scope & (bull_fvg >= 1) & (bull_gap <= 0)]),
        ("bearish", middle[in_scope & (bear_fvg >= 1) & (bear_gap <= 0)]),
    ]
    proposal_frames: list[pd.DataFrame] = []
    for direction, selected in selections:
        if selected.size == 0:
            continue
        frame = pd.DataFrame(
            {
                "product": "GC",
                "trade_date_ny": bars["trade_date_ny"].to_numpy()[selected],
                "direction": direction,
                "poi_middle_bar_id": selected.astype("int32"),
                "poi_activation_bar_id": (selected + 1).astype("int32"),
                "swing_n": pd.Series(pd.NA, index=np.arange(selected.size), dtype="Int64"),
                "break_mode": pd.Series(pd.NA, index=np.arange(selected.size), dtype="string"),
                "break_bar_id": pd.Series(pd.NA, index=np.arange(selected.size), dtype="Int64"),
                "displacement_start_bar_id": pd.Series(
                    pd.NA, index=np.arange(selected.size), dtype="Int64"
                ),
                "broken_swing_bar_id": pd.Series(pd.NA, index=np.arange(selected.size), dtype="Int64"),
            }
        )
        proposal_frames.append(frame)
    if not proposal_frames:
        return pd.DataFrame()
    proposals = pd.concat(proposal_frames, ignore_index=True)
    proposals = proposals.drop_duplicates(["direction", "poi_middle_bar_id"]).reset_index(drop=True)
    return evaluate_refined_poi_proposals(
        bars,
        proposals,
        cfg,
        audit_record_type="supplemental_diagnostic_rejection",
    )


def build_refined_poi_audit(
    feature_frame: pd.DataFrame,
    config: Section6CConfig | None = None,
) -> pd.DataFrame:
    """Rebuild local-swing proposals and retain all Section 6C audit rows."""

    cfg = config or Section6CConfig()
    # The audit begins from the original one-tick proposal population.  The
    # refined three-tick rule is applied only after formation correctness.
    proposal_cfg = replace(cfg, min_fvg_ticks=1.0, min_close_open_gap_ticks=1.0)
    variant_proposals = build_poi_table(feature_frame, proposal_cfg)
    variant_audit = evaluate_refined_poi_proposals(feature_frame, variant_proposals, cfg)
    supplemental = build_supplemental_refinement_audit(feature_frame, cfg)
    frames = [frame for frame in (variant_audit, supplemental) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    audit = pd.concat(frames, ignore_index=True, sort=False)
    audit = audit.sort_values(
        ["poi_middle_bar_id", "direction", "audit_record_type", "swing_n", "break_mode"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return audit


def select_final_refined_pois(audit: pd.DataFrame) -> pd.DataFrame:
    """Return only valid variant-level POIs under the default 3-tick rule."""

    required = {
        "audit_record_type",
        "final_poi_valid_flag",
        "poi_variant_id",
        "canonical_poi_id",
    }
    _validate_columns(audit, required, "audit")
    out = audit.loc[
        audit["audit_record_type"].eq("variant_proposal")
        & audit["final_poi_valid_flag"].astype(bool)
        & audit["poi_variant_id"].notna()
    ].copy()
    if out.empty:
        return out
    out["poi_id"] = out["poi_variant_id"].astype("string")
    out = out.sort_values(
        ["trade_date_ny", "swing_n", "break_mode", "direction", "poi_activation_bar_id", "poi_middle_bar_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def build_section6c_refined_tables(
    research_bars: pd.DataFrame,
    config: Section6CConfig | None = None,
    *,
    legacy_counts: dict[str, int] | None = None,
) -> dict[str, pd.DataFrame | pd.Series | dict[str, float]]:
    """Build every Section 6C object from bars, never from old downstream rows."""

    cfg = config or Section6CConfig()
    timings: dict[str, float] = {}
    full_start = time.perf_counter()

    stage_start = time.perf_counter()
    feature_frame = prepare_section6_feature_frame(research_bars, cfg)
    timings["feature_preparation_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    audit = build_refined_poi_audit(feature_frame, cfg)
    timings["poi_formation_audit_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    refined_pois = select_final_refined_pois(audit)
    timings["final_poi_filtering_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    structural_events = build_structural_swing_break_events(feature_frame, cfg)
    refined_pois = annotate_poi_structural_validation(refined_pois, structural_events, cfg)
    timings["structural_annotation_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    retests = build_retest_table(feature_frame, refined_pois, cfg)
    retests = _enrich_refined_retests(retests, refined_pois, cfg)
    timings["retest_construction_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    candidates = build_candidate_trade_table(refined_pois, retests, cfg)
    candidates = _enrich_refined_candidates(candidates, retests, refined_pois, cfg)
    signals = build_signal_frame(refined_pois, retests, candidates)
    signals = _finalize_refined_signal_frame(signals, cfg)
    timings["candidate_construction_seconds"] = time.perf_counter() - stage_start

    audit = _attach_refined_poi_structural_context(audit, refined_pois)
    audit = _attach_first_downstream_lineage(audit, retests, candidates)
    validation = build_section6c_validation(
        audit=audit,
        poi_table=refined_pois,
        retest_table=retests,
        candidate_trade_table=candidates,
        signal_frame=signals,
        config=cfg,
    )
    timings["section6c_full_runtime_seconds"] = time.perf_counter() - full_start
    summary = build_section6c_refinement_summary(
        feature_frame=feature_frame,
        audit=audit,
        poi_table=refined_pois,
        retest_table=retests,
        candidate_trade_table=candidates,
        signal_frame=signals,
        timings=timings,
        legacy_counts=legacy_counts,
        config=cfg,
    )

    return {
        "section6_feature_frame": feature_frame,
        "structural_break_events": structural_events,
        "refined_poi_audit": audit,
        "refined_poi_table": refined_pois,
        "refined_retest_table": retests,
        "refined_candidate_trade_table": candidates,
        "refined_signal_frame": signals,
        "section6c_validation": validation,
        "refinement_summary": summary,
        "section6c_timings": timings,
    }


def _poi_lineage_columns(poi_table: pd.DataFrame) -> list[str]:
    wanted = [
        "poi_id",
        "poi_variant_id",
        "canonical_poi_id",
        "poi_definition_version",
        "poi_geometry_case",
        "base_poi_geometry_case",
        "poi_zone_expanded_flag",
        "poi_zone_expansion_points",
        "poi_zone_expansion_ticks",
        "formation_valid_flag",
        "fvg_threshold_pass",
        "final_poi_valid_flag",
        "classic_fvg_flag",
        "fvg_low",
        "fvg_high",
        "fvg_size_points",
        "fvg_size_ticks",
        "fvg_size_atr_ratio",
        "fvg_ge_3tick_flag",
        "fvg_ge_4tick_flag",
        "fvg_ge_5tick_flag",
        "close_open_gap_flag",
        "close_open_gap_low",
        "close_open_gap_high",
        "close_open_gap_points",
        "close_open_gap_ticks",
        "prior_bar_id",
        "poi_bar_id",
        "confirmation_bar_id",
        "activation_bar_id",
        "swing_break_bar_id",
        "prior_ts_event_utc",
        "prior_ts_event_ny",
        "confirmation_ts_event_utc",
        "confirmation_ts_event_ny",
        "activation_ts_event_utc",
        "activation_ts_event_ny",
        "confirmation_body_direction",
        "confirmation_close_inside_gap_flag",
        "confirmation_gap_fully_closed_flag",
        "confirmation_gap_fill_fraction",
        "opening_gap_survives_close_flag",
        "poi_low",
        "poi_high",
        "poi_mid",
        "poi_size_ticks",
        "swing_n",
        "break_mode",
        "direction",
        "trade_date_ny",
        *STRUCTURAL_COLUMNS,
    ]
    return [col for col in dict.fromkeys(wanted) if col in poi_table.columns]


def _enrich_refined_retests(
    retest_table: pd.DataFrame,
    poi_table: pd.DataFrame,
    config: Section6CConfig,
) -> pd.DataFrame:
    if retest_table.empty:
        return retest_table.copy()
    context_cols = _poi_lineage_columns(poi_table)
    context_cols = [
        col for col in context_cols if col == "poi_id" or col not in retest_table.columns
    ]
    out = retest_table.merge(
        poi_table[context_cols],
        on="poi_id",
        how="left",
        validate="many_to_one",
    )
    out["poi_variant_id"] = out["poi_id"].astype("string")
    canonical_touch = (
        out[["canonical_poi_id", "retest_bar_id"]]
        .drop_duplicates()
        .sort_values(["canonical_poi_id", "retest_bar_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    canonical_touch["canonical_retest_number"] = (
        canonical_touch.groupby("canonical_poi_id", observed=True).cumcount() + 1
    ).astype("int32")
    canonical_touch["canonical_retest_id"] = (
        canonical_touch["canonical_poi_id"].astype("string")
        + "_RT_"
        + canonical_touch["retest_bar_id"].astype(str).str.zfill(8)
    )
    out = out.merge(
        canonical_touch,
        on=["canonical_poi_id", "retest_bar_id"],
        how="left",
        validate="many_to_one",
    )
    out["poi_definition_version"] = config.poi_definition_version
    return out


def _enrich_refined_candidates(
    candidate_table: pd.DataFrame,
    retest_table: pd.DataFrame,
    poi_table: pd.DataFrame,
    config: Section6CConfig,
) -> pd.DataFrame:
    if candidate_table.empty:
        return candidate_table.copy()
    retest_wanted = [
        "retest_id",
        "canonical_retest_id",
        "canonical_retest_number",
        "canonical_poi_id",
        "poi_variant_id",
        "retest_number",
        "first_touch_flag",
        "trade_date_ny",
        "swing_n",
        "break_mode",
        "execution_window_label",
        "retest_ts_event_ny",
        "retest_minute_ny",
        "time_since_poi_activation_minutes",
        "candles_since_poi_activation",
    ]
    retest_cols = [
        col
        for col in retest_wanted
        if col in retest_table.columns
        and (col == "retest_id" or col not in candidate_table.columns)
    ]
    out = candidate_table.merge(
        retest_table[retest_cols],
        on="retest_id",
        how="left",
        validate="many_to_one",
    )
    poi_cols = [
        col
        for col in _poi_lineage_columns(poi_table)
        if col == "poi_id" or col not in out.columns
    ]
    out = out.merge(
        poi_table[poi_cols],
        on="poi_id",
        how="left",
        validate="many_to_one",
    )
    out = out.sort_values(
        ["retest_ts_event_utc", "poi_id", "entry_variant", "stop_model"],
        kind="mergesort",
    ).reset_index(drop=True)
    out["candidate_trade_id"] = "GC6C_CAND_" + (out.index + 1).astype(str).str.zfill(8)
    out["candidate_variant_id"] = out["candidate_trade_id"].astype("string")
    out["canonical_candidate_id"] = (
        out["canonical_retest_id"].astype("string")
        + "_"
        + out["entry_variant"].astype("string")
        + "_"
        + out["stop_model"].astype("string")
    )
    out["poi_definition_version"] = config.poi_definition_version
    return out


def _finalize_refined_signal_frame(
    signal_frame: pd.DataFrame,
    config: Section6CConfig,
) -> pd.DataFrame:
    if signal_frame.empty:
        return signal_frame.copy()
    out = signal_frame.copy().reset_index(drop=True)
    out["signal_id"] = "GC6C_SIG_" + (out.index + 1).astype(str).str.zfill(8)
    out["poi_definition_version"] = config.poi_definition_version
    first = ["signal_id", "candidate_variant_id", "canonical_candidate_id"]
    return out[first + [col for col in out.columns if col not in set(first)]]


def _attach_first_downstream_lineage(
    audit: pd.DataFrame,
    retests: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    out = audit.copy()
    if not retests.empty and "poi_variant_id" in out:
        first_retest = (
            retests.sort_values(["poi_variant_id", "retest_bar_id"], kind="mergesort")
            .drop_duplicates("poi_variant_id")
            .loc[
                :,
                [
                    "poi_variant_id",
                    "retest_id",
                    "canonical_retest_id",
                    "retest_bar_id",
                    "retest_ts_event_utc",
                    "retest_ts_event_ny",
                ],
            ]
            .rename(
                columns={
                    "retest_id": "first_retest_id",
                    "canonical_retest_id": "first_canonical_retest_id",
                    "retest_bar_id": "first_retest_bar_id",
                    "retest_ts_event_utc": "first_retest_ts_event_utc",
                    "retest_ts_event_ny": "first_retest_ts_event_ny",
                }
            )
        )
        out = out.merge(first_retest, on="poi_variant_id", how="left", validate="many_to_one")
        out["retest_bar_id"] = out["first_retest_bar_id"].astype("Int64")
        out["retest_ts_event_utc"] = out["first_retest_ts_event_utc"]
        out["retest_ts_event_ny"] = out["first_retest_ts_event_ny"]
    else:
        out["retest_bar_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["retest_ts_event_utc"] = pd.NaT
        out["retest_ts_event_ny"] = pd.NaT
    if not candidates.empty:
        first_candidate = (
            candidates.sort_values(
                ["poi_variant_id", "retest_bar_id", "entry_variant", "stop_model"],
                kind="mergesort",
            )
            .drop_duplicates("poi_variant_id")
            .loc[:, ["poi_variant_id", "candidate_variant_id", "entry_price"]]
            .rename(
                columns={
                    "candidate_variant_id": "first_candidate_variant_id",
                    "entry_price": "first_candidate_entry_price",
                }
            )
        )
        out = out.merge(first_candidate, on="poi_variant_id", how="left", validate="many_to_one")
    return out


def _attach_refined_poi_structural_context(
    audit: pd.DataFrame,
    poi_table: pd.DataFrame,
) -> pd.DataFrame:
    out = audit.drop(columns=[col for col in STRUCTURAL_COLUMNS if col in audit], errors="ignore")
    if poi_table.empty:
        for col in STRUCTURAL_COLUMNS:
            out[col] = pd.NA
        return out
    context = poi_table[["poi_variant_id", *STRUCTURAL_COLUMNS]].drop_duplicates(
        "poi_variant_id"
    )
    return out.merge(context, on="poi_variant_id", how="left", validate="many_to_one")


def build_section6c_validation(
    *,
    audit: pd.DataFrame,
    poi_table: pd.DataFrame,
    retest_table: pd.DataFrame,
    candidate_trade_table: pd.DataFrame,
    signal_frame: pd.DataFrame,
    config: Section6CConfig | None = None,
) -> pd.Series:
    """Return acceptance-oriented validation checks for the refined layer."""

    cfg = config or Section6CConfig()
    variant_audit = audit.loc[audit["audit_record_type"].eq("variant_proposal")]
    rejected_variant_ids = set(
        variant_audit.loc[~variant_audit["final_poi_valid_flag"], "poi_variant_id"]
        .dropna()
        .astype(str)
    )
    downstream_ids: set[str] = set()
    for table in (retest_table, candidate_trade_table, signal_frame):
        if "poi_variant_id" in table:
            downstream_ids.update(table["poi_variant_id"].dropna().astype(str))
        elif "poi_id" in table:
            downstream_ids.update(table["poi_id"].dropna().astype(str))
    point_consistency = np.isclose(
        poi_table["fvg_size_points"].to_numpy("float64"),
        poi_table["fvg_size_ticks"].to_numpy("float64") * cfg.tick_size,
        atol=1e-8,
        rtol=0.0,
    ).all() if not poi_table.empty else True
    cohort3 = set(poi_table.loc[poi_table["fvg_ge_3tick_flag"], "poi_variant_id"].astype(str))
    cohort4 = set(poi_table.loc[poi_table["fvg_ge_4tick_flag"], "poi_variant_id"].astype(str))
    cohort5 = set(poi_table.loc[poi_table["fvg_ge_5tick_flag"], "poi_variant_id"].astype(str))
    structural_partition = (
        (
            poi_table["structural_swing_break_flag"].astype(bool)
            ^ poi_table["local_swing_only_flag"].astype(bool)
        ).all()
        if not poi_table.empty
        else True
    )
    return pd.Series(
        {
            "audit_ids_unique": audit["audit_id"].is_unique,
            "poi_variant_ids_unique": poi_table["poi_variant_id"].is_unique
            if not poi_table.empty
            else True,
            "final_poi_table_all_valid": poi_table["final_poi_valid_flag"].astype(bool).all()
            if not poi_table.empty
            else True,
            "default_three_tick_threshold_enforced": poi_table["fvg_size_ticks"]
            .ge(cfg.min_fvg_ticks)
            .all()
            if not poi_table.empty
            else True,
            "five_tick_subset_of_four_tick": cohort5.issubset(cohort4),
            "four_tick_subset_of_three_tick": cohort4.issubset(cohort3),
            "point_tick_measurements_consistent": bool(point_consistency),
            "structural_partition_valid": bool(structural_partition),
            "retests_strictly_after_activation": retest_table["retest_bar_id"]
            .gt(retest_table["activation_bar_id"])
            .all()
            if not retest_table.empty
            else True,
            "canonical_retest_ids_present": retest_table["canonical_retest_id"].notna().all()
            if not retest_table.empty
            else True,
            "candidate_variant_ids_unique": candidate_trade_table["candidate_variant_id"].is_unique
            if not candidate_trade_table.empty
            else True,
            "signal_ids_unique": signal_frame["signal_id"].is_unique
            if not signal_frame.empty
            else True,
            "rejected_formations_do_not_leak_downstream": rejected_variant_ids.isdisjoint(
                downstream_ids
            ),
            "definition_version_consistent": all(
                table["poi_definition_version"].eq(cfg.poi_definition_version).all()
                for table in (poi_table, retest_table, candidate_trade_table, signal_frame)
                if not table.empty
            ),
        }
    )


def build_section6c_refinement_summary(
    *,
    feature_frame: pd.DataFrame,
    audit: pd.DataFrame,
    poi_table: pd.DataFrame,
    retest_table: pd.DataFrame,
    candidate_trade_table: pd.DataFrame,
    signal_frame: pd.DataFrame,
    timings: dict[str, float],
    legacy_counts: dict[str, int] | None = None,
    config: Section6CConfig | None = None,
) -> pd.DataFrame:
    """Build a long-form funnel, breakdown, downstream, and runtime summary."""

    cfg = config or Section6CConfig()
    records: list[dict[str, Any]] = []

    def add(scope: str, metric: str, value: Any, dimension: str = "all", label: str = "all") -> None:
        records.append(
            {
                "summary_scope": scope,
                "dimension": dimension,
                "dimension_value": label,
                "metric": metric,
                "value": value,
                "poi_definition_version": cfg.poi_definition_version,
            }
        )

    variants = audit.loc[audit["audit_record_type"].eq("variant_proposal")]
    add("formation_funnel", "proposed_three_bar_formations", len(variants))
    add("formation_funnel", "genuine_classic_fvgs", int(variants["classic_fvg_flag"].sum()))
    add("formation_funnel", "genuine_close_open_gaps", int(variants["close_open_gap_flag"].sum()))
    add(
        "formation_funnel",
        "rejected_confirmation_body_closures",
        int(variants["rejection_reason"].eq("confirmation_close_inside_opening_gap").sum()),
    )
    add(
        "formation_funnel",
        "rejected_no_fvg_cases",
        int(audit["rejection_reason"].eq("no_classic_fvg").sum()),
    )
    add(
        "formation_funnel",
        "rejected_continuity_segment_cases",
        int(
            audit["rejection_reason"]
            .isin(["non_consecutive_bars", "segment_boundary_crossed", "contract_switch_crossed"])
            .sum()
        ),
    )
    add("formation_funnel", "formation_valid_pois", int(variants["formation_valid_flag"].sum()))
    add(
        "formation_funnel",
        "rejected_below_3_ticks",
        int(variants["rejection_reason"].eq("fvg_below_3_ticks").sum()),
    )
    add("formation_funnel", "final_ge_3_tick_pois", len(poi_table))
    add("formation_funnel", "final_ge_4_tick_pois", int(poi_table["fvg_ge_4tick_flag"].sum()))
    add("formation_funnel", "final_ge_5_tick_pois", int(poi_table["fvg_ge_5tick_flag"].sum()))
    add(
        "formation_funnel",
        "case_1_count",
        int(poi_table["poi_geometry_case"].eq("case_1_expanded").sum()),
    )
    add(
        "formation_funnel",
        "case_2_count",
        int(poi_table["poi_geometry_case"].eq("case_2_standard").sum()),
    )

    rejection_counts = audit["rejection_reason"].astype("string").value_counts(dropna=False)
    for label, value in rejection_counts.items():
        add("rejection_counts", "row_count", int(value), "rejection_reason", str(label))

    breakdown_frame = poi_table.copy()
    if not breakdown_frame.empty:
        breakdown_frame["year"] = pd.to_datetime(breakdown_frame["trade_date_ny"]).dt.year
        minute = breakdown_frame["poi_activation_minute_ny"]
        breakdown_frame["poi_session"] = np.select(
            [minute.between(180, 359), minute.between(420, 720)],
            ["London Execution", "New York Execution"],
            default="Outside Execution",
        )
        breakdown_frame["structural_partition"] = np.where(
            breakdown_frame["structural_swing_break_flag"],
            "structural",
            "local_only",
        )
        breakdown_frame["fvg_tick_bucket"] = _fvg_tick_bucket(
            breakdown_frame["fvg_size_ticks"]
        )
        breakdown_frame["zone_expansion_tick_bucket"] = _expansion_tick_bucket(
            breakdown_frame["poi_zone_expansion_ticks"]
        )
        dimensions = [
            "direction",
            "poi_session",
            "year",
            "swing_n",
            "break_mode",
            "structural_partition",
            "structural_swing_window_broken",
            "fvg_size_ticks",
            "fvg_tick_bucket",
            "poi_geometry_case",
            "zone_expansion_tick_bucket",
        ]
        for dimension in dimensions:
            counts = breakdown_frame[dimension].astype("string").value_counts(dropna=False)
            for label, value in counts.items():
                add("poi_breakdown", "poi_variant_count", int(value), dimension, str(label))

    downstream = {
        "gc_feature_rows": len(feature_frame),
        "valid_variant_pois": len(poi_table),
        "unique_canonical_pois": poi_table["canonical_poi_id"].nunique(),
        "pois_with_retests": retest_table["poi_variant_id"].nunique()
        if not retest_table.empty
        else 0,
        "retests": len(retest_table),
        "unique_canonical_retests": retest_table["canonical_retest_id"].nunique()
        if not retest_table.empty
        else 0,
        "first_touches": int(retest_table["first_touch_flag"].sum())
        if not retest_table.empty
        else 0,
        "later_touches": int((~retest_table["first_touch_flag"]).sum())
        if not retest_table.empty
        else 0,
        "valid_candidate_variants": len(candidate_trade_table),
        "unique_canonical_candidate_variants": candidate_trade_table["canonical_candidate_id"].nunique()
        if not candidate_trade_table.empty
        else 0,
        "signal_rows": len(signal_frame),
        "unique_trading_dates": signal_frame["trade_date_ny"].nunique()
        if not signal_frame.empty
        else 0,
    }
    for metric, value in downstream.items():
        add("downstream", metric, int(value))
    for metric, value in (legacy_counts or {}).items():
        add("version_comparison", metric, int(value))
    for metric, value in timings.items():
        add("runtime", metric, float(value))
    return pd.DataFrame.from_records(records)


def save_section6c_tables(
    tables: dict[str, pd.DataFrame | pd.Series | dict[str, float]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save only the six non-destructive Section 6C parquet outputs."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: out_dir / name for key, name in SECTION6C_OUTPUT_FILENAMES.items()}
    for key, path in paths.items():
        table = tables[key]
        if isinstance(table, pd.Series):
            table = table.rename("value").reset_index(names="metric")
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"{key} must be a DataFrame or Series")
        table.to_parquet(path, index=False)
    return paths


def _fvg_tick_bucket(values: pd.Series) -> pd.Series:
    ticks = pd.to_numeric(values, errors="coerce")
    labels = np.select(
        [ticks.eq(3), ticks.eq(4), ticks.eq(5), ticks.between(6, 9), ticks.between(10, 19), ticks.ge(20)],
        ["3", "4", "5", "6-9", "10-19", "20+"],
        default="below_3",
    )
    return pd.Series(labels, index=values.index, dtype="category")


def _expansion_tick_bucket(values: pd.Series) -> pd.Series:
    ticks = pd.to_numeric(values, errors="coerce")
    labels = np.select(
        [ticks.eq(0), ticks.eq(1), ticks.eq(2), ticks.between(3, 5), ticks.between(6, 9), ticks.ge(10)],
        ["0", "1", "2", "3-5", "6-9", "10+"],
        default="unknown",
    )
    return pd.Series(labels, index=values.index, dtype="category")


def _canonical_poi_ids(frame: pd.DataFrame) -> pd.Series:
    direction_code = np.where(frame["direction"].eq("bullish"), "BULL", "BEAR")
    prior = frame.get("prior_bar_id", frame["poi_middle_bar_id"] - 1).astype("int64")
    middle = frame["poi_middle_bar_id"].astype("int64")
    confirm = frame.get("confirmation_bar_id", middle + 1).astype("int64")
    return (
        "GC_POI_"
        + pd.Series(direction_code, index=frame.index).astype("string")
        + "_"
        + prior.astype(str).str.zfill(8)
        + "_"
        + middle.astype(str).str.zfill(8)
        + "_"
        + confirm.astype(str).str.zfill(8)
    )


def _validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")
