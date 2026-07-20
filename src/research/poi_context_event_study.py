"""Section 7 True POI context, outcome, and candidate-policy research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.features.poi_context_features import (
    PoiContextConfig,
    build_true_poi_context_frame,
    prepare_context_bars,
)
from src.features.poi_event_study import (
    Section7Config,
    build_bar_horizon_path_features,
)
from src.features.poi_first_passage import (
    FIRST_PASSAGE_REQUIRED_BAR_COLUMNS,
    FirstPassageConfig,
    evaluate_first_passage,
    prepare_first_passage_bars,
)


@dataclass(frozen=True)
class PoiContextResearchConfig:
    horizons: tuple[int, ...] = (5, 15, 30, 60, 120, 180, 240)
    target_rs: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0)
    time_exits: tuple[int, ...] = (15, 30, 60, 120, 180, 240)
    bootstrap_samples: int = 200
    min_candidate_true_pois: int = 100
    min_candidate_dates: int = 40
    random_state: int = 20260712


FEATURE_SCREEN = (
    "feat_poi_width_atr",
    "feat_fvg_to_poi_width_ratio",
    "feat_opening_gap_to_poi_width_ratio",
    "feat_confirmation_directional_close_location",
    "feat_displacement_efficiency",
    "feat_displacement_speed_ticks_per_minute",
    "feat_displacement_relative_volume",
    "feat_displacement_maximum_internal_pullback_fraction",
    "feat_approach_5m_speed_ticks_per_minute",
    "feat_approach_15m_speed_ticks_per_minute",
    "feat_approach_15m_directional_efficiency",
    "feat_approach_15m_range_compression_ratio",
    "feat_approach_15m_candle_overlap_ratio",
    "feat_approach_15m_relative_volume",
    "feat_displacement_volume_to_approach_volume_ratio",
    "feat_displacement_speed_to_approach_speed_ratio",
    "feat_poi_age_minutes",
    "feat_previous_touch_count",
    "feat_distance_from_vwap_atr",
    "feat_vwap_slope_15m_ticks_per_minute",
    "feat_return_30m",
    "feat_short_to_long_volatility_ratio",
    "feat_realized_volatility_60m",
    "feat_time_of_day_adjusted_relative_volume",
    "feat_session_range_before_retest_ticks",
    "feat_position_within_session_range",
    "feat_distance_from_session_open_atr",
    "feat_fraction_recent_expected_range_travelled",
    "feat_successive_body_decay_3m",
    "feat_successive_range_decay_3m",
)


INTERACTION_HYPOTHESES = (
    ("displacement_quality_x_approach_quality", "High-efficiency displacement followed by low-efficiency approach should favour continuation."),
    ("displacement_speed_x_approach_speed", "Fast displacement followed by slow approach should favour continuation."),
    ("displacement_volume_x_approach_volume", "High displacement/approach volume ratio should favour continuation."),
    ("approach_aggression_x_penetration", "Aggressive approach plus deep penetration should increase zone failure/reversal risk."),
    ("poi_geometry_x_fvg_size", "Standard geometry with a large normalized FVG should be more stable than expanded geometry."),
    ("vwap_distance_x_slope_x_side", "VWAP extension and slope alignment should condition trade side."),
    ("trend_state_x_hypothesis", "Trend alignment should favour continuation and misalignment should favour reversal."),
    ("volatility_x_stop_atr", "Higher volatility requires wider normalized stops."),
    ("session_x_time_x_side", "London and New York exhibit different side and holding behaviour."),
    ("structural15_x_displacement", "15-bar validation should help only when displacement quality is also high."),
    ("extension_exhaustion_x_reversal", "Extension plus body/range decay should favour reversal."),
    ("compression_x_continuation", "Compression into the True POI should condition breakout/continuation."),
)


def build_directional_outcome_labels(
    context: pd.DataFrame,
    research_bars: pd.DataFrame,
    config: PoiContextResearchConfig | None = None,
) -> pd.DataFrame:
    """Create paired continuation/reversal counterfactual path labels."""

    cfg = config or PoiContextResearchConfig()
    bars = prepare_context_bars(research_bars)
    section7_bars = bars[[
        "bar_id", "ts_event_utc", "ts_event_ny", "trade_date_ny", "product",
        "high", "low", "close", "minute_of_day_ny", "continuous_segment_id",
        "tradable_research_flag", "roll_window_flag",
    ]]
    base = context[[
        "true_poi_id", "true_retest_id", "true_trade_opportunity_id",
        "trade_date_ny", "direction", "retest_bar_id", "poi_low", "poi_high",
        "poi_mid", "feat_first_contact_edge_price", "feat_long_term_atr_ticks",
        "feat_approach_direction_sign", "feat_touch_penetration_fraction",
        "feat_touch_full_zone_traversal",
        "research_partition",
    ]].copy()
    frames = []
    for hypothesis in ("continuation", "reversal"):
        frame = base.copy()
        bullish = frame["direction"].eq("bullish")
        if hypothesis == "continuation":
            frame["trade_side"] = np.where(bullish, "long", "short")
        else:
            frame["trade_side"] = np.where(bullish, "short", "long")
        frame["hypothesis"] = hypothesis
        frame["label_entry_price"] = frame["feat_first_contact_edge_price"]
        long_side = frame["trade_side"].eq("long")
        frame["label_normalized_stop_price"] = np.where(
            long_side, frame["poi_low"] - 0.10, frame["poi_high"] + 0.10
        )
        frame["label_risk_points"] = np.where(
            long_side,
            frame["label_entry_price"] - frame["label_normalized_stop_price"],
            frame["label_normalized_stop_price"] - frame["label_entry_price"],
        )
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    rid = out["retest_bar_id"].to_numpy("int64")
    entry = out["label_entry_price"].to_numpy("float64")
    risk = out["label_risk_points"].to_numpy("float64")
    long_side = out["trade_side"].eq("long").to_numpy(bool)

    # Path labels are accumulated in a plain dict and joined in one concat so
    # the wide output frame stays block-consolidated (no fragmented inserts).
    path_columns: dict[str, np.ndarray] = {}
    for horizon in cfg.horizons:
        path = build_bar_horizon_path_features(
            section7_bars, horizon, Section7Config(forward_horizons=(horizon,))
        )
        aligned = path.iloc[rid]
        for mode in ("fixed", "capped"):
            valid = aligned[f"{mode}_valid"].to_numpy(bool)
            exit_close = aligned[f"{mode}_exit_close"].to_numpy("float64")
            max_high = aligned[f"{mode}_max_high"].to_numpy("float64")
            min_low = aligned[f"{mode}_min_low"].to_numpy("float64")
            forward_points = np.where(long_side, exit_close - entry, entry - exit_close)
            mfe = np.where(long_side, max_high - entry, entry - min_low)
            mae = np.where(long_side, entry - min_low, max_high - entry)
            prefix = f"label_{mode}_{horizon}m_"
            path_columns[prefix + "valid"] = valid
            path_columns[prefix + "signed_return"] = np.where(valid, forward_points / entry, np.nan)
            path_columns[prefix + "r"] = np.where(valid, forward_points / risk, np.nan)
            path_columns[prefix + "mfe_r"] = np.where(valid, mfe / risk, np.nan)
            path_columns[prefix + "mae_r"] = np.where(valid, mae / risk, np.nan)
            path_columns[prefix + "positive_final_r"] = path_columns[prefix + "r"] > 0
            if mode == "capped":
                path_columns[prefix + "forced_exit_flag"] = (
                    aligned["capped_forced_exit_flag"].to_numpy(bool) & valid
                )
    out = pd.concat([out, pd.DataFrame(path_columns, index=out.index)], axis=1)

    paired = out.pivot(index="true_retest_id", columns="hypothesis", values="label_capped_60m_r")
    dominance = np.sign(paired.get("continuation", np.nan) - paired.get("reversal", np.nan))
    derived: dict[str, np.ndarray | pd.Series] = {}
    derived["label_continuation_dominance"] = out["true_retest_id"].map(dominance).eq(1)
    derived["label_reversal_dominance"] = out["true_retest_id"].map(dominance).eq(-1)
    zone_hold = np.where(
        out["direction"].eq("bullish"),
        out["label_capped_60m_r"].where(out["trade_side"].eq("long"), -out["label_capped_60m_r"]),
        out["label_capped_60m_r"].where(out["trade_side"].eq("short"), -out["label_capped_60m_r"]),
    ) > 0
    derived["label_zone_hold"] = zone_hold
    derived["label_zone_failure"] = ~zone_hold
    derived.update(_build_interaction_path_labels(out, section7_bars))
    location_mfe_points = (
        out["label_capped_60m_mfe_r"] * out["label_risk_points"]
    ).groupby(out["true_retest_id"]).max()
    location_r = out["label_capped_60m_mfe_r"].groupby(out["true_retest_id"]).max()
    location_points = out["true_retest_id"].map(location_mfe_points)
    location_atr = location_points / (out["feat_long_term_atr_ticks"] * 0.10)
    derived["label_location_max_move_away_points_60m"] = location_points
    derived["label_location_max_move_away_atr_60m"] = location_atr
    derived["label_location_reaction_0_5atr"] = location_atr >= 0.5
    derived["label_location_reaction_1_0atr"] = location_atr >= 1.0
    mapped_location_r = out["true_retest_id"].map(location_r)
    for threshold in (1, 2, 3):
        derived[f"label_location_reaction_{threshold}r"] = mapped_location_r >= threshold
    return pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)


def _build_interaction_path_labels(
    out: pd.DataFrame, bars: pd.DataFrame
) -> dict[str, np.ndarray | pd.Series]:
    """Touch aftermath and time-to-extreme diagnostic columns, returned unattached."""

    high = bars["high"].to_numpy("float64", copy=False)
    low = bars["low"].to_numpy("float64", copy=False)
    close = bars["close"].to_numpy("float64", copy=False)
    segment = bars["continuous_segment_id"].to_numpy("int64", copy=False)
    rid = out["retest_bar_id"].to_numpy("int64")
    entry = out["label_entry_price"].to_numpy("float64")
    long_side = out["trade_side"].eq("long").to_numpy(bool)
    time_mfe = np.full(len(out), np.nan)
    time_mae = np.full(len(out), np.nan)
    for start in range(0, len(out), 10_000):
        rows = np.arange(start, min(start + 10_000, len(out)))
        idx = rid[rows, None] + np.arange(61)[None, :]
        safe = np.clip(idx, 0, len(bars) - 1)
        valid = (idx < len(bars)) & (segment[safe] == segment[rid[rows], None])
        favorable = np.where(
            long_side[rows, None], high[safe] - entry[rows, None], entry[rows, None] - low[safe]
        )
        adverse = np.where(
            long_side[rows, None], entry[rows, None] - low[safe], high[safe] - entry[rows, None]
        )
        favorable[~valid] = -np.inf
        adverse[~valid] = -np.inf
        time_mfe[rows] = np.argmax(favorable, axis=1)
        time_mae[rows] = np.argmax(adverse, axis=1)
    columns: dict[str, np.ndarray | pd.Series] = {
        "label_time_to_mfe_60m": time_mfe,
        "label_time_to_mae_60m": time_mae,
    }

    unique = out.drop_duplicates("true_retest_id").copy()
    urid = unique["retest_bar_id"].to_numpy("int64")
    next_idx = np.minimum(urid + 1, len(bars) - 1)
    same_segment = segment[next_idx] == segment[urid]
    zone_overlap_next = (high[next_idx] >= unique["poi_low"].to_numpy()) & (
        low[next_idx] <= unique["poi_high"].to_numpy()
    )
    immediate_exit = same_segment & ~zone_overlap_next
    attack = unique["feat_approach_direction_sign"].to_numpy("float64")
    edge = unique["feat_first_contact_edge_price"].to_numpy("float64")
    rejection = np.where(attack < 0, high[next_idx] - edge, edge - low[next_idx])
    reentry = np.zeros(len(unique), dtype=bool)
    time_inside = np.zeros(len(unique), dtype="int16")
    for i, bar in enumerate(urid):
        end = min(bar + 16, len(bars))
        valid = segment[bar:end] == segment[bar]
        overlap = (high[bar:end] >= unique.iloc[i]["poi_low"]) & (low[bar:end] <= unique.iloc[i]["poi_high"]) & valid
        time_inside[i] = int(overlap.sum())
        if immediate_exit[i] and len(overlap) > 2:
            reentry[i] = bool(overlap[2:].any())
    mapping = unique.assign(
        _immediate_exit=immediate_exit,
        _rejection_ticks=np.maximum(rejection, 0) / 0.10,
        _reentry=reentry,
        _inside=time_inside,
    ).set_index("true_retest_id")
    columns["label_interaction_immediate_exit_next_bar"] = out["true_retest_id"].map(mapping["_immediate_exit"])
    columns["label_interaction_immediate_rejection_ticks"] = out["true_retest_id"].map(mapping["_rejection_ticks"])
    columns["label_interaction_reentry_within_15m"] = out["true_retest_id"].map(mapping["_reentry"])
    columns["label_interaction_bars_inside_poi_next_15m"] = out["true_retest_id"].map(mapping["_inside"])
    return columns


def build_stop_policy_opportunities(
    context: pd.DataFrame,
    research_bars: pd.DataFrame,
    *,
    hypothesis: str,
    stop_model: str,
    entry_model: str | None = None,
) -> pd.DataFrame:
    """Construct one compact opportunity row per True POI/retest and stop rule."""

    if hypothesis not in {"continuation", "reversal"}:
        raise ValueError("hypothesis must be continuation or reversal")
    valid_stops = {
        "poi_invalidation_1tick",
        "recent_micro_swing",
        "volatility_hybrid",
        "touch_rejection_candle",
        "legacy_conservative_adjacent",
    }
    if stop_model not in valid_stops:
        raise ValueError(f"unknown stop model: {stop_model}")
    if entry_model is None:
        entry_model = "next_bar_confirmation" if stop_model == "touch_rejection_candle" else "boundary_touch"
    bars = (
        research_bars
        if set(FIRST_PASSAGE_REQUIRED_BAR_COLUMNS).issubset(research_bars.columns)
        else prepare_first_passage_bars(research_bars)
    )
    base = context[[
        "true_poi_id", "true_retest_id", "true_trade_opportunity_id",
        "trade_date_ny", "direction", "retest_bar_id", "poi_low", "poi_high",
        "poi_size_ticks", "prior_bar_id", "poi_bar_id", "confirmation_bar_id",
        "feat_first_contact_edge_price", "feat_long_term_atr_ticks", "research_partition",
    ]].copy()
    bullish = base["direction"].eq("bullish")
    base["hypothesis"] = hypothesis
    base["trade_side"] = np.where(
        bullish,
        "long" if hypothesis == "continuation" else "short",
        "short" if hypothesis == "continuation" else "long",
    )
    base["entry_model"] = entry_model
    base["stop_model"] = stop_model
    rid = base["retest_bar_id"].to_numpy("int64")
    if entry_model == "next_bar_confirmation":
        entry_bar = rid + 1
        in_bounds = entry_bar < len(bars)
        entry_bar = np.minimum(entry_bar, len(bars) - 1)
        base["entry_price"] = bars["open"].to_numpy("float64")[entry_bar]
        base["entry_bar_id"] = entry_bar
        base.loc[~in_bounds, "entry_bar_id"] = -1
    else:
        base["entry_price"] = base["feat_first_contact_edge_price"]
        base["entry_bar_id"] = rid
    entry = base["entry_price"].to_numpy("float64")
    long_side = base["trade_side"].eq("long").to_numpy(bool)
    high = bars["high"].to_numpy("float64")
    low = bars["low"].to_numpy("float64")
    if stop_model == "poi_invalidation_1tick":
        stop = np.where(long_side, base["poi_low"] - 0.10, base["poi_high"] + 0.10)
    elif stop_model == "volatility_hybrid":
        minimum = base["feat_long_term_atr_ticks"].to_numpy("float64") * 0.10 * 0.50
        poi_stop = np.where(long_side, base["poi_low"] - 0.10, base["poi_high"] + 0.10)
        stop = np.where(long_side, np.minimum(poi_stop, entry - minimum), np.maximum(poi_stop, entry + minimum))
    elif stop_model == "touch_rejection_candle":
        stop = np.where(long_side, low[rid] - 0.10, high[rid] + 0.10)
    elif stop_model == "legacy_conservative_adjacent":
        a = base["prior_bar_id"].to_numpy("int64")
        b = base["poi_bar_id"].to_numpy("int64")
        c = base["confirmation_bar_id"].to_numpy("int64")
        stop = np.where(
            long_side,
            np.minimum.reduce([low[a], low[b], low[c]]) - 0.10,
            np.maximum.reduce([high[a], high[b], high[c]]) + 0.10,
        )
    else:
        entry_idx = base["entry_bar_id"].to_numpy("int64")
        matrix = np.clip(entry_idx[:, None] - np.arange(7, 0, -1)[None, :], 0, len(bars) - 1)
        stop = np.where(long_side, low[matrix].min(axis=1) - 0.10, high[matrix].max(axis=1) + 0.10)
    base["stop_price"] = stop
    base["risk_points"] = np.where(long_side, entry - stop, stop - entry)
    base["risk_ticks"] = base["risk_points"] / 0.10
    base["risk_atr"] = base["risk_ticks"] / base["feat_long_term_atr_ticks"]
    base["risk_in_poi_widths"] = base["risk_ticks"] / base["poi_size_ticks"]
    base["risk_25_100_tick_flag"] = base["risk_ticks"].between(25, 100, inclusive="both")
    base["true_trade_opportunity_id"] = (
        base["true_retest_id"].astype("string") + "_" + hypothesis + "_" + entry_model + "_" + stop_model
    )
    return base


def build_feature_study_summary(
    context: pd.DataFrame,
    labels: pd.DataFrame,
    config: PoiContextResearchConfig | None = None,
    features: Iterable[str] = FEATURE_SCREEN,
) -> pd.DataFrame:
    """Fit development bins and report locked chronological feature summaries."""

    cfg = config or PoiContextResearchConfig()
    analysis = labels[[
        "true_retest_id", "hypothesis", "trade_side", "research_partition",
        "label_capped_60m_r", "trade_date_ny",
    ]].merge(context, on=["true_retest_id", "trade_date_ny", "research_partition"], how="left", validate="many_to_one")
    analysis["screen_r"] = analysis["label_capped_60m_r"].clip(-5.0, 5.0)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(cfg.random_state)
    for feature in features:
        if feature not in analysis:
            continue
        dev_values = pd.to_numeric(
            analysis.loc[analysis["research_partition"].eq("development"), feature], errors="coerce"
        ).dropna()
        if dev_values.nunique() < 4:
            continue
        edges = np.unique(np.nanquantile(dev_values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        bins = pd.cut(pd.to_numeric(analysis[feature], errors="coerce"), edges, include_lowest=True, duplicates="drop")
        temp = analysis.assign(feature_bin=bins.astype("string"))
        for (partition, hypothesis, feature_bin), group in temp.dropna(subset=["feature_bin", "screen_r"]).groupby(
            ["research_partition", "hypothesis", "feature_bin"], observed=True, sort=False
        ):
            values = group["screen_r"].to_numpy("float64")
            uncapped = group["label_capped_60m_r"].to_numpy("float64")
            ci_low, ci_high = _date_block_bootstrap_ci(group, "screen_r", rng, cfg.bootstrap_samples)
            test = stats.ttest_1samp(values, 0.0, nan_policy="omit") if len(values) > 1 else None
            rows.append({
                "feature_name": feature,
                "development_bin_edges": "|".join(f"{x:.8g}" for x in edges),
                "research_partition": str(partition),
                "hypothesis": hypothesis,
                "feature_bin": feature_bin,
                "true_retest_count": group["true_retest_id"].nunique(),
                "true_poi_count": group["true_poi_id"].nunique(),
                "trading_date_count": group["trade_date_ny"].nunique(),
                "mean_r": np.mean(values),
                "mean_uncapped_r": np.mean(uncapped),
                "median_r": np.median(values),
                "q10_r": np.quantile(values, 0.10),
                "q25_r": np.quantile(values, 0.25),
                "date_cluster_bootstrap_ci_low": ci_low,
                "date_cluster_bootstrap_ci_high": ci_high,
                "raw_p_value": float(test.pvalue) if test is not None else np.nan,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    dev = result["research_partition"].eq("development")
    result.loc[dev, "bh_q_value"] = _benjamini_hochberg(result.loc[dev, "raw_p_value"])
    result["monotonic_direction"] = "not_assessed"
    for (feature, partition, hypothesis), group in result.groupby(
        ["feature_name", "research_partition", "hypothesis"], observed=True
    ):
        ordered = group.sort_values("feature_bin")
        rho = stats.spearmanr(np.arange(len(ordered)), ordered["mean_r"]).statistic if len(ordered) > 2 else np.nan
        label = "increasing" if rho >= 0.7 else "decreasing" if rho <= -0.7 else "non_monotonic"
        result.loc[group.index, "monotonic_direction"] = label
    return result


def build_interaction_summary(
    context: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the twelve pre-specified, economically motivated interactions."""

    analysis = labels[[
        "true_retest_id", "hypothesis", "trade_side", "research_partition",
        "label_capped_60m_r", "trade_date_ny",
    ]].merge(context, on=["true_retest_id", "trade_date_ny", "research_partition"], how="left")
    analysis["screen_r"] = analysis["label_capped_60m_r"].clip(-5.0, 5.0)
    dev = analysis["research_partition"].eq("development")
    q = lambda col, value: float(pd.to_numeric(analysis.loc[dev, col], errors="coerce").quantile(value))
    conditions = {
        "displacement_quality_x_approach_quality": (analysis["feat_displacement_efficiency"] >= q("feat_displacement_efficiency", .5)) & (analysis["feat_approach_15m_directional_efficiency"] <= q("feat_approach_15m_directional_efficiency", .5)),
        "displacement_speed_x_approach_speed": analysis["feat_displacement_speed_to_approach_speed_ratio"] >= q("feat_displacement_speed_to_approach_speed_ratio", .6),
        "displacement_volume_x_approach_volume": analysis["feat_displacement_volume_to_approach_volume_ratio"] >= q("feat_displacement_volume_to_approach_volume_ratio", .6),
        "approach_aggression_x_penetration": (analysis["feat_approach_15m_speed_ticks_per_minute"] >= q("feat_approach_15m_speed_ticks_per_minute", .6)) & (analysis["feat_touch_penetration_fraction"] >= .5),
        "poi_geometry_x_fvg_size": analysis["feat_poi_case"].astype("string").eq("case_2_standard") & (analysis["feat_fvg_size_atr"] >= q("feat_fvg_size_atr", .5)),
        "vwap_distance_x_slope_x_side": (analysis["feat_distance_from_vwap_atr"].abs() >= q("feat_distance_from_vwap_atr", .6)) & (np.sign(analysis["feat_vwap_slope_15m_ticks_per_minute"]) == np.where(analysis["trade_side"].eq("long"), 1, -1)),
        "trend_state_x_hypothesis": np.where(analysis["hypothesis"].eq("continuation"), analysis["feat_trend_alignment_continuation"], analysis["feat_trend_alignment_reversal"]),
        "volatility_x_stop_atr": (analysis["feat_short_to_long_volatility_ratio"] >= q("feat_short_to_long_volatility_ratio", .6)) & (analysis["feat_poi_width_atr"] >= q("feat_poi_width_atr", .5)),
        "session_x_time_x_side": analysis["feat_execution_session"].astype("string").eq("London") & analysis["trade_side"].eq("short") & (analysis["feat_minutes_since_execution_window_open"] <= 120),
        "structural15_x_displacement": analysis["feat_15bar_structural_validation"] & (analysis["feat_displacement_efficiency"] >= q("feat_displacement_efficiency", .5)),
        "extension_exhaustion_x_reversal": analysis["hypothesis"].eq("reversal") & (analysis["feat_distance_from_vwap_atr"].abs() >= q("feat_distance_from_vwap_atr", .75)) & (analysis["feat_successive_body_decay_3m"] < 1),
        "compression_x_continuation": analysis["hypothesis"].eq("continuation") & analysis["feat_compression_after_displacement"],
    }
    hypotheses = dict(INTERACTION_HYPOTHESES)
    rows = []
    for name, condition in conditions.items():
        analysis["_condition"] = np.asarray(condition, dtype=bool)
        for (partition, hypothesis, flag), group in analysis.groupby(
            ["research_partition", "hypothesis", "_condition"], observed=True
        ):
            values = group["screen_r"].dropna()
            if values.empty:
                continue
            rows.append({
                "interaction_name": name,
                "pre_registered_hypothesis": hypotheses[name],
                "condition_met": bool(flag),
                "research_partition": str(partition),
                "hypothesis": hypothesis,
                "true_poi_count": group["true_poi_id"].nunique(),
                "true_retest_count": group["true_retest_id"].nunique(),
                "trading_date_count": group["trade_date_ny"].nunique(),
                "mean_r": values.mean(),
                "mean_uncapped_r": group["label_capped_60m_r"].mean(),
                "median_r": values.median(),
                "q25_r": values.quantile(.25),
                "positive_r_rate": values.gt(0).mean(),
            })
    return pd.DataFrame(rows)


def build_matched_control_location_summary(
    context: pd.DataFrame,
    labels: pd.DataFrame,
    research_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Compare True POI expansion with state-matched non-POI control bars."""

    bars = prepare_context_bars(research_bars)
    section7_bars = bars[[
        "bar_id", "ts_event_utc", "ts_event_ny", "trade_date_ny", "product",
        "high", "low", "close", "minute_of_day_ny", "continuous_segment_id",
        "tradable_research_flag", "roll_window_flag",
    ]]
    path = build_bar_horizon_path_features(
        section7_bars, 60, Section7Config(forward_horizons=(60,))
    )
    dates = pd.to_datetime(bars["trade_date_ny"])
    partition = np.select(
        [dates <= pd.Timestamp("2023-12-31"), dates <= pd.Timestamp("2024-12-31")],
        ["development", "validation"], default="final_test"
    )
    minute = bars["minute_of_day_ny"].to_numpy("int32")
    session = np.where(minute < 360, "London", np.where(minute >= 420, "New York", "outside"))
    time_bin = minute // 30
    rv = bars["rolling_realized_vol_60m"].to_numpy("float64")
    dev_rv = rv[partition == "development"]
    vol_edges = np.nanquantile(dev_rv[np.isfinite(dev_rv)], [.25, .75, .90])
    vol_bucket = np.digitize(rv, vol_edges)
    close = bars["close"].to_numpy("float64")
    prior = np.maximum(np.arange(len(bars)) - 15, 0)
    recent_move = np.abs(close / close[prior] - 1.0)
    dev_move = recent_move[partition == "development"]
    move_edges = np.nanquantile(dev_move[np.isfinite(dev_move)], [.33, .67])
    move_bucket = np.digitize(recent_move, move_edges)
    eligible = (
        path["capped_valid"].to_numpy(bool)
        & bars["tradable_research_flag"].to_numpy(bool)
        & ~bars["roll_window_flag"].to_numpy(bool)
        & (((minute >= 180) & (minute < 360)) | ((minute >= 420) & (minute <= 720)))
    )
    retest_bars = set(context["retest_bar_id"].astype(int).tolist())
    candidate = pd.DataFrame({
        "bar_id": np.arange(len(bars)), "partition": partition, "session": session,
        "time_bin": time_bin, "vol_bucket": vol_bucket, "move_bucket": move_bucket,
    })
    candidate = candidate.loc[eligible & ~candidate["bar_id"].isin(retest_bars)].copy()
    candidate["key"] = list(zip(candidate["partition"], candidate["session"], candidate["time_bin"], candidate["vol_bucket"], candidate["move_bucket"]))
    pools = candidate.groupby("key", sort=False)["bar_id"].apply(lambda x: x.to_numpy("int64")).to_dict()

    rid = context["retest_bar_id"].to_numpy("int64")
    decision = np.maximum(rid - 1, 0)
    keys = list(zip(
        context["research_partition"].astype("string"),
        context["feat_execution_session"].astype("string"),
        time_bin[rid], vol_bucket[decision], move_bucket[decision],
    ))
    controls = np.full(len(context), -1, dtype="int64")
    counters: dict[tuple[Any, ...], int] = {}
    for idx, key in enumerate(keys):
        pool = pools.get(key)
        if pool is None or len(pool) == 0:
            continue
        count = counters.get(key, 0)
        controls[idx] = pool[count % len(pool)]
        counters[key] = count + 1
    valid = controls >= 0
    aligned = path.iloc[np.clip(controls, 0, len(path) - 1)]
    control_move = np.maximum(
        aligned["capped_max_high"].to_numpy() - close[np.clip(controls, 0, len(bars) - 1)],
        close[np.clip(controls, 0, len(bars) - 1)] - aligned["capped_min_low"].to_numpy(),
    ) / bars["rolling_atr_60m"].to_numpy("float64")[np.clip(controls, 0, len(bars) - 1)]
    control_move[~valid] = np.nan
    poi_move = labels.drop_duplicates("true_retest_id").set_index("true_retest_id")[
        "label_location_max_move_away_atr_60m"
    ].reindex(context["true_retest_id"]).to_numpy()
    observations = pd.concat([
        pd.DataFrame({"research_partition": context["research_partition"].astype("string"), "sample_type": "True POI retest", "reaction_atr_60m": poi_move}),
        pd.DataFrame({"research_partition": context["research_partition"].astype("string"), "sample_type": "matched non-POI control", "reaction_atr_60m": control_move}),
    ], ignore_index=True)
    rows = []
    for (research_partition, sample_type), group in observations.dropna().groupby(
        ["research_partition", "sample_type"], observed=True
    ):
        values = group["reaction_atr_60m"]
        rows.append({
            "research_partition": research_partition,
            "sample_type": sample_type,
            "observation_count": len(values),
            "mean_reaction_atr_60m": values.mean(),
            "median_reaction_atr_60m": values.median(),
            "q25_reaction_atr_60m": values.quantile(.25),
            "reaction_0_5atr_rate": values.ge(.5).mean(),
            "reaction_1_0atr_rate": values.ge(1.0).mean(),
            "matching_dimensions": "partition|session|30m time bin|volatility bucket|recent-move bucket",
        })
    return pd.DataFrame(rows)


def build_stop_target_summary(
    context: pd.DataFrame,
    labels: pd.DataFrame,
    research_bars: pd.DataFrame,
    config: PoiContextResearchConfig | None = None,
) -> pd.DataFrame:
    """Study five stop families, nine targets, and fixed holding exits."""

    cfg = config or PoiContextResearchConfig()
    bars = prepare_first_passage_bars(research_bars)
    rows: list[dict[str, Any]] = []
    stop_models = (
        "poi_invalidation_1tick", "recent_micro_swing", "volatility_hybrid",
        "touch_rejection_candle", "legacy_conservative_adjacent",
    )
    for hypothesis in ("continuation", "reversal"):
        for stop_model in stop_models:
            opp = build_stop_policy_opportunities(
                context, bars, hypothesis=hypothesis, stop_model=stop_model
            )
            path = evaluate_first_passage(
                opp, bars, target_rs=cfg.target_rs, max_holding_minutes=240,
                ambiguity_treatment="conservative",
            )
            enriched = path.merge(
                opp[["true_trade_opportunity_id", "research_partition", "trade_date_ny", "true_poi_id"]],
                on=["true_trade_opportunity_id", "true_poi_id", "trade_date_ny"], how="left",
            )
            for (partition, target_r), group in enriched.groupby(["research_partition", "target_r"], observed=True):
                rows.extend(_ambiguity_sensitivity_summaries(group, hypothesis, stop_model, target_r, "240m"))

            if stop_model == "poi_invalidation_1tick":
                mandatory = evaluate_first_passage(
                    opp, bars, target_rs=cfg.target_rs, max_holding_minutes=None,
                    ambiguity_treatment="conservative",
                ).merge(
                    opp[["true_trade_opportunity_id", "research_partition", "trade_date_ny", "true_poi_id"]],
                    on=["true_trade_opportunity_id", "true_poi_id", "trade_date_ny"], how="left",
                )
                for (partition, target_r), group in mandatory.groupby(["research_partition", "target_r"], observed=True):
                    rows.extend(_ambiguity_sensitivity_summaries(
                        group, hypothesis, stop_model, target_r, "15:30_mandatory"
                    ))
                del mandatory

            # Stop timing is identical across target multiples. Reuse the first
            # target's ordered stop result, then mark -1R whenever that stop
            # occurs before the requested time exit.
            first_target_path = path.loc[path["target_r"].eq(cfg.target_rs[0])].set_index(
                "true_trade_opportunity_id"
            )
            for horizon in cfg.time_exits:
                time_r = _time_exit_realized_r(opp, first_target_path, bars, horizon)
                for partition, values in time_r.groupby(opp["research_partition"], observed=True):
                    valid = values.dropna()
                    rows.append({
                        "analysis_type": "time_exit",
                        "hypothesis": hypothesis,
                        "stop_model": stop_model,
                        "entry_model": opp["entry_model"].iloc[0],
                        "target_r": np.nan,
                        "holding_policy": f"{horizon}m",
                        "research_partition": str(partition),
                        "opportunity_count": len(valid),
                        "true_poi_count": opp.loc[values.index, "true_poi_id"].nunique(),
                        "trading_date_count": opp.loc[values.index, "trade_date_ny"].nunique(),
                        "mean_realized_r": valid.mean(),
                        "median_realized_r": valid.median(),
                        "q10_realized_r": valid.quantile(.10),
                        "q25_realized_r": valid.quantile(.25),
                        "positive_r_rate": valid.gt(0).mean(),
                    })
            del path, enriched, opp
    return pd.DataFrame(rows)


def _time_exit_realized_r(
    opportunities: pd.DataFrame,
    first_target_path: pd.DataFrame,
    bars: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    entry_idx = opportunities["entry_bar_id"].to_numpy("int64")
    minute = bars["minute_of_day_ny"].to_numpy("int32")
    close = bars["close"].to_numpy("float64")
    segment = bars["continuous_segment_id"].to_numpy("int64")
    date_code, _ = pd.factorize(bars["trade_date_ny"], sort=False)
    actual_holding = np.minimum(horizon, np.maximum(930 - minute[np.clip(entry_idx, 0, len(bars)-1)], 0))
    exit_idx = entry_idx + actual_holding
    safe = np.clip(exit_idx, 0, len(bars) - 1)
    valid = (
        (entry_idx >= 0) & (entry_idx < len(bars)) & (exit_idx < len(bars))
        & (segment[safe] == segment[np.clip(entry_idx, 0, len(bars)-1)])
        & (date_code[safe] == date_code[np.clip(entry_idx, 0, len(bars)-1)])
    )
    long_side = opportunities["trade_side"].eq("long").to_numpy(bool)
    points = np.where(
        long_side,
        close[safe] - opportunities["entry_price"].to_numpy("float64"),
        opportunities["entry_price"].to_numpy("float64") - close[safe],
    )
    risk = opportunities["risk_points"].to_numpy("float64")
    realized = np.full(len(opportunities), np.nan)
    np.divide(points, risk, out=realized, where=valid & np.isfinite(risk) & (risk > 0))
    aligned = first_target_path.reindex(opportunities["true_trade_opportunity_id"])
    filled = aligned["entry_filled"].fillna(False).to_numpy(bool)
    stop_time = aligned["time_from_entry_to_stop_minutes"].to_numpy("float64")
    stopped = np.isfinite(stop_time) & (stop_time <= actual_holding)
    realized[stopped & filled & valid] = -1.0
    realized[~filled] = np.nan
    return pd.Series(realized, index=opportunities.index)


def build_candidate_policy_results(
    context: pd.DataFrame,
    research_bars: pd.DataFrame,
    config: PoiContextResearchConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze a small pre-specified universe on development/validation, then test."""

    cfg = config or PoiContextResearchConfig()
    bars = prepare_first_passage_bars(research_bars)
    specs = _candidate_specs(context)
    all_rows = []
    policy_outcomes: dict[str, pd.DataFrame] = {}
    for spec in specs:
        opp = build_stop_policy_opportunities(
            context, bars, hypothesis=spec["hypothesis"],
            stop_model=spec["stop_model"], entry_model=spec["entry_model"],
        )
        eligible = _candidate_mask(opp, context, spec)
        opp = opp.loc[eligible].reset_index(drop=True)
        if opp.empty:
            continue
        path = evaluate_first_passage(
            opp, bars, target_rs=(spec["target_r"],),
            max_holding_minutes=spec["maximum_holding_minutes"],
            ambiguity_treatment="conservative",
        )
        path = path.merge(
            opp[["true_trade_opportunity_id", "research_partition", "true_poi_id", "true_retest_id", "trade_date_ny"]],
            on=["true_trade_opportunity_id", "true_poi_id", "true_retest_id", "trade_date_ny"], how="left",
        )
        policy_outcomes[spec["policy_id"]] = path
        for partition, group in path.groupby("research_partition", observed=True):
            record = _candidate_partition_record(group, spec, str(partition))
            all_rows.append(record)
    ranking = pd.DataFrame(all_rows)
    if ranking.empty:
        return ranking, ranking

    pivot = ranking.pivot(index="policy_id", columns="research_partition", values="mean_realized_r")
    counts = ranking.pivot(index="policy_id", columns="research_partition", values="true_poi_count")
    dates = ranking.pivot(index="policy_id", columns="research_partition", values="trading_date_count")
    missing_float = pd.Series(-np.inf, index=pivot.index, dtype="float64")
    missing_zero = pd.Series(0.0, index=pivot.index, dtype="float64")
    freeze_score = (
        pivot.get("development", missing_float).fillna(-np.inf)
        + pivot.get("validation", missing_float).fillna(-np.inf)
    )
    adequate = (
        counts.get("development", missing_zero).fillna(0).ge(cfg.min_candidate_true_pois)
        & counts.get("validation", missing_zero).fillna(0).ge(max(50, cfg.min_candidate_true_pois // 2))
        & dates.get("development", missing_zero).fillna(0).ge(cfg.min_candidate_dates)
        & dates.get("validation", missing_zero).fillna(0).ge(max(20, cfg.min_candidate_dates // 2))
    )
    frozen_ids = freeze_score.loc[adequate].sort_values(ascending=False).head(5).index.tolist()
    ranking["frozen_before_final_test"] = ranking["policy_id"].isin(frozen_ids)
    ranking["selection_score_dev_validation"] = ranking["policy_id"].map(freeze_score)

    registry_rows = []
    for policy_id in frozen_ids:
        spec = next(s for s in specs if s["policy_id"] == policy_id)
        metrics = ranking.loc[ranking["policy_id"].eq(policy_id)].set_index("research_partition")
        dev = metrics.loc["development"] if "development" in metrics.index else None
        val = metrics.loc["validation"] if "validation" in metrics.index else None
        test = metrics.loc["final_test"] if "final_test" in metrics.index else None
        stable = all(x is not None and x["mean_realized_r"] > 0 for x in (dev, val, test))
        ambiguity_ok = all(x is not None and x["same_bar_ambiguity_rate"] <= 0.10 for x in (val, test))
        downside_ok = all(x is not None and x["q25_realized_r"] > -1.0 for x in (val, test))
        if stable and ambiguity_ok and downside_ok:
            decision = "ADVANCE_TO_SECTION_8"
            reason = "Positive development, validation, and final-test first-passage expectancy with acceptable ambiguity and downside."
        elif val is not None and test is not None and val["mean_realized_r"] > 0 and test["mean_realized_r"] > 0:
            decision = "RESEARCH_ONLY"
            reason = "Out-of-sample expectancy is positive, but downside or ambiguity fails the advancement threshold."
        else:
            decision = "REJECT"
            reason = "Validation/final-test first-passage expectancy is not consistently positive."
        registry_rows.append({**spec, "decision": decision, "decision_reason": reason})
    return ranking.sort_values(["frozen_before_final_test", "selection_score_dev_validation"], ascending=False), pd.DataFrame(registry_rows)


def save_section7_outputs(outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "context_frame": "section7_true_poi_context_frame_gc.parquet",
        "outcome_labels": "section7_true_poi_outcome_labels_gc.parquet",
        "feature_study_summary": "section7_true_poi_feature_study_summary_gc.parquet",
        "interaction_summary": "section7_true_poi_interaction_summary_gc.parquet",
        "stop_target_summary": "section7_true_poi_stop_target_summary_gc.parquet",
        "candidate_ranking": "section7_true_poi_candidate_ranking_gc.parquet",
        "candidate_registry": "section7_true_poi_backtest_candidate_registry_gc.parquet",
        "location_quality_summary": "section7_true_poi_location_quality_summary_gc.parquet",
    }
    paths = {}
    for key, filename in names.items():
        path = out_dir / filename
        outputs[key].to_parquet(path, index=False)
        paths[key] = path
    return paths


def _candidate_specs(context: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"policy_id": "S7P01_LDN_BEAR_CONT", "hypothesis": "continuation", "poi_direction": "bearish", "trade_side": "short", "eligible_session": "London", "poi_geometry": "case_2_standard", "required_feature_conditions": "slow_or_mixed_15m_approach", "entry_model": "boundary_touch", "entry_price_rule": "first-contact edge", "stop_model": "poi_invalidation_1tick", "target_model": "fixed_2R", "target_r": 2.0, "maximum_holding_minutes": 60, "forced_exit_rule": "same-day 15:30 NY", "re_entry_rule": "each unique retest evaluated; Section 8 must sequence", "ambiguity_treatment": "conservative_stop_first"},
        {"policy_id": "S7P02_NY_BEAR_CONT", "hypothesis": "continuation", "poi_direction": "bearish", "trade_side": "short", "eligible_session": "New York", "poi_geometry": "case_2_standard", "required_feature_conditions": "compressed_15m_approach", "entry_model": "boundary_touch", "entry_price_rule": "first-contact edge", "stop_model": "volatility_hybrid", "target_model": "fixed_3R", "target_r": 3.0, "maximum_holding_minutes": 240, "forced_exit_rule": "same-day 15:30 NY", "re_entry_rule": "each unique retest evaluated; Section 8 must sequence", "ambiguity_treatment": "conservative_stop_first"},
        {"policy_id": "S7P03_LDN_BULL_REV", "hypothesis": "reversal", "poi_direction": "bullish", "trade_side": "short", "eligible_session": "London", "poi_geometry": "case_2_standard", "required_feature_conditions": "extended_from_vwap", "entry_model": "boundary_touch", "entry_price_rule": "first-contact edge", "stop_model": "recent_micro_swing", "target_model": "fixed_2R", "target_r": 2.0, "maximum_holding_minutes": 60, "forced_exit_rule": "same-day 15:30 NY", "re_entry_rule": "each unique retest evaluated; Section 8 must sequence", "ambiguity_treatment": "conservative_stop_first"},
        {"policy_id": "S7P04_NY_BULL_REV_CONFIRM", "hypothesis": "reversal", "poi_direction": "bullish", "trade_side": "short", "eligible_session": "New York", "poi_geometry": "case_2_standard", "required_feature_conditions": "touch_rejection_and_extension", "entry_model": "next_bar_confirmation", "entry_price_rule": "next eligible bar open", "stop_model": "touch_rejection_candle", "target_model": "fixed_2R", "target_r": 2.0, "maximum_holding_minutes": 120, "forced_exit_rule": "same-day 15:30 NY", "re_entry_rule": "each unique retest evaluated; Section 8 must sequence", "ambiguity_treatment": "conservative_stop_first"},
        {"policy_id": "S7P05_NY_BULL_CONT", "hypothesis": "continuation", "poi_direction": "bullish", "trade_side": "long", "eligible_session": "New York", "poi_geometry": "case_2_standard", "required_feature_conditions": "high_volatility_slow_approach", "entry_model": "boundary_touch", "entry_price_rule": "first-contact edge", "stop_model": "volatility_hybrid", "target_model": "fixed_2R", "target_r": 2.0, "maximum_holding_minutes": 120, "forced_exit_rule": "same-day 15:30 NY", "re_entry_rule": "each unique retest evaluated; Section 8 must sequence", "ambiguity_treatment": "conservative_stop_first"},
        {"policy_id": "S7P06_NY_BEAR_REV", "hypothesis": "reversal", "poi_direction": "bearish", "trade_side": "long", "eligible_session": "New York", "poi_geometry": "case_2_standard", "required_feature_conditions": "exhaustion_body_decay", "entry_model": "next_bar_confirmation", "entry_price_rule": "next eligible bar open", "stop_model": "touch_rejection_candle", "target_model": "fixed_1.5R", "target_r": 1.5, "maximum_holding_minutes": 60, "forced_exit_rule": "same-day 15:30 NY", "re_entry_rule": "each unique retest evaluated; Section 8 must sequence", "ambiguity_treatment": "conservative_stop_first"},
    ]


def _candidate_mask(opp: pd.DataFrame, context: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    features = context.set_index("true_retest_id")
    aligned = features.reindex(opp["true_retest_id"])
    mask = (
        opp["direction"].eq(spec["poi_direction"])
        & aligned["feat_execution_session"].astype("string").to_numpy().astype(str).__eq__(spec["eligible_session"])
        & aligned["feat_poi_case"].astype("string").eq(spec["poi_geometry"]).to_numpy()
    )
    condition = spec["required_feature_conditions"]
    dev = aligned["research_partition"].eq("development")
    if condition == "slow_or_mixed_15m_approach":
        threshold = aligned.loc[dev, "feat_approach_15m_speed_ticks_per_minute"].quantile(.60)
        mask &= aligned["feat_approach_15m_speed_ticks_per_minute"].le(threshold).to_numpy()
    elif condition == "compressed_15m_approach":
        threshold = aligned.loc[dev, "feat_approach_15m_range_compression_ratio"].quantile(.40)
        mask &= aligned["feat_approach_15m_range_compression_ratio"].le(threshold).to_numpy()
    elif condition == "extended_from_vwap":
        threshold = aligned.loc[dev, "feat_distance_from_vwap_atr"].abs().quantile(.70)
        mask &= aligned["feat_distance_from_vwap_atr"].abs().ge(threshold).to_numpy()
    elif condition == "touch_rejection_and_extension":
        threshold = aligned.loc[dev, "feat_distance_from_vwap_atr"].abs().quantile(.60)
        mask &= (aligned["feat_touch_rejection_wick_to_body_ratio"].ge(1) & aligned["feat_distance_from_vwap_atr"].abs().ge(threshold)).to_numpy()
    elif condition == "high_volatility_slow_approach":
        vol = aligned.loc[dev, "feat_realized_volatility_60m"].quantile(.70)
        speed = aligned.loc[dev, "feat_approach_15m_speed_ticks_per_minute"].quantile(.50)
        mask &= (aligned["feat_realized_volatility_60m"].ge(vol) & aligned["feat_approach_15m_speed_ticks_per_minute"].le(speed)).to_numpy()
    elif condition == "exhaustion_body_decay":
        mask &= aligned["feat_successive_body_decay_3m"].lt(1).to_numpy()
    return np.asarray(mask, dtype=bool)


def _summarize_first_passage_group(group: pd.DataFrame, hypothesis: str, stop_model: str, target_r: float, hold: str) -> dict[str, Any]:
    valid = group["realized_r"].dropna()
    return {
        "analysis_type": "first_passage_target",
        "hypothesis": hypothesis,
        "stop_model": stop_model,
        "entry_model": group["entry_model"].iloc[0] if "entry_model" in group else "",
        "target_r": target_r,
        "holding_policy": hold,
        "ambiguity_treatment": group["ambiguity_treatment"].iloc[0] if "ambiguity_treatment" in group else "conservative",
        "research_partition": str(group["research_partition"].iloc[0]),
        "opportunity_count": len(group),
        "true_poi_count": group["true_poi_id"].nunique(),
        "trading_date_count": group["trade_date_ny"].nunique(),
        "entry_fill_rate": group["entry_filled"].mean(),
        "target_before_stop_rate": group["target_hit_before_stop"].mean(),
        "stop_before_target_rate": group["stop_hit_before_target"].mean(),
        "forced_exit_rate": group["forced_exit_before_either"].mean(),
        "same_bar_ambiguity_rate": group["same_bar_ambiguity"].mean(),
        "median_time_to_target": group["time_from_entry_to_target_minutes"].median(),
        "median_time_to_stop": group["time_from_entry_to_stop_minutes"].median(),
        "mean_realized_r": valid.mean(),
        "median_realized_r": valid.median(),
        "q10_realized_r": valid.quantile(.10),
        "q25_realized_r": valid.quantile(.25),
        "mean_uncapped_r": group["maximum_uncapped_r"].mean(),
        "runner_tail_rate_5r": group["maximum_uncapped_r"].ge(5).mean(),
        "risk_25_100_tick_rate": group.get("risk_25_100_tick_flag", pd.Series(False, index=group.index)).mean(),
        "median_risk_ticks": group.get("risk_ticks", pd.Series(np.nan, index=group.index)).median(),
        "risk_below_25_tick_rate": group.get("risk_ticks", pd.Series(np.nan, index=group.index)).lt(25).mean(),
        "risk_above_100_tick_rate": group.get("risk_ticks", pd.Series(np.nan, index=group.index)).gt(100).mean(),
    }


def _ambiguity_sensitivity_summaries(
    group: pd.DataFrame,
    hypothesis: str,
    stop_model: str,
    target_r: float,
    hold: str,
) -> list[dict[str, Any]]:
    summaries = [_summarize_first_passage_group(group, hypothesis, stop_model, target_r, hold)]
    excluded = group.loc[~group["same_bar_ambiguity"]].copy()
    if not excluded.empty:
        excluded["ambiguity_treatment"] = "exclude"
        summaries.append(_summarize_first_passage_group(excluded, hypothesis, stop_model, target_r, hold))
    optimistic = group.copy()
    ambiguous = optimistic["same_bar_ambiguity"].astype(bool)
    optimistic.loc[ambiguous, "target_hit_before_stop"] = True
    optimistic.loc[ambiguous, "stop_hit_before_target"] = False
    optimistic.loc[ambiguous, "realized_r"] = target_r
    optimistic["ambiguity_treatment"] = "optimistic"
    summaries.append(_summarize_first_passage_group(optimistic, hypothesis, stop_model, target_r, hold))
    return summaries


def _candidate_partition_record(group: pd.DataFrame, spec: dict[str, Any], partition: str) -> dict[str, Any]:
    valid = group["realized_r"].dropna()
    by_year = group.assign(year=pd.to_datetime(group["trade_date_ny"]).dt.year).groupby("year")["realized_r"].mean()
    return {
        **spec,
        "research_partition": partition,
        "true_poi_count": group["true_poi_id"].nunique(),
        "true_retest_count": group["true_retest_id"].nunique(),
        "trading_date_count": group["trade_date_ny"].nunique(),
        "mean_realized_r": valid.mean(),
        "median_realized_r": valid.median(),
        "robust_capped_mean_r": valid.clip(-5, 5).mean(),
        "q10_realized_r": valid.quantile(.10),
        "q25_realized_r": valid.quantile(.25),
        "target_before_stop_rate": group["target_hit_before_stop"].mean(),
        "stop_before_target_rate": group["stop_hit_before_target"].mean(),
        "mean_uncapped_r": group["maximum_uncapped_r"].mean(),
        "mean_mfe_r": group["maximum_uncapped_r"].mean(),
        "mean_mae_r": group["maximum_adverse_r"].mean(),
        "same_bar_ambiguity_rate": group["same_bar_ambiguity"].mean(),
        "forced_exit_rate": group["forced_exit_before_either"].mean(),
        "positive_year_rate": by_year.gt(0).mean() if len(by_year) else np.nan,
        "worst_year_mean_r": by_year.min() if len(by_year) else np.nan,
    }


def _date_block_bootstrap_ci(group: pd.DataFrame, value_col: str, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    daily = group.groupby("trade_date_ny", observed=True)[value_col].mean().dropna().to_numpy()
    if len(daily) < 2:
        return np.nan, np.nan
    draws = rng.integers(0, len(daily), size=(samples, len(daily)))
    means = daily[draws].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy("float64")
    valid = np.isfinite(p)
    result = np.full(len(p), np.nan)
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    positions = np.flatnonzero(valid)[order]
    result[positions] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=pvalues.index)
