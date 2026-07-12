"""Section 7R event studies using only refined Section 6C signals."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.features.poi_event_study import (
    EventStudyGroupSpec,
    Section7Config,
    build_bar_horizon_path_features,
    build_section7_validation,
    build_signal_horizon_metrics,
    default_section7_group_specs,
    prepare_section7_bar_frame,
    prepare_section7_signal_frame,
    rank_section7_candidate_groups,
    summarize_section7_groups,
)


SECTION7R_REQUIRED_COLUMNS = {
    "canonical_poi_id",
    "poi_variant_id",
    "canonical_retest_id",
    "candidate_variant_id",
    "canonical_candidate_id",
    "poi_definition_version",
    "poi_geometry_case",
    "poi_zone_expanded_flag",
    "poi_zone_expansion_ticks",
    "fvg_size_points",
    "fvg_size_ticks",
    "fvg_ge_3tick_flag",
    "fvg_ge_4tick_flag",
    "fvg_ge_5tick_flag",
}


def default_section7r_group_specs() -> list[EventStudyGroupSpec]:
    """Return legacy Section 7 diagnostics plus refined-only comparisons."""

    return [
        *default_section7_group_specs(),
        EventStudyGroupSpec("poi_geometry_case", ("poi_geometry_case", "trade_side")),
        EventStudyGroupSpec("fvg_tick_bucket", ("fvg_size_bucket", "trade_side")),
        EventStudyGroupSpec(
            "zone_expansion_tick_bucket",
            ("zone_expansion_tick_bucket", "trade_side"),
        ),
        EventStudyGroupSpec(
            "fvg_min_3_ticks",
            ("trade_side",),
            filter_column="fvg_ge_3tick_flag",
            filter_value=True,
        ),
        EventStudyGroupSpec(
            "fvg_min_4_ticks",
            ("trade_side",),
            filter_column="fvg_ge_4tick_flag",
            filter_value=True,
        ),
        EventStudyGroupSpec(
            "fvg_min_5_ticks",
            ("trade_side",),
            filter_column="fvg_ge_5tick_flag",
            filter_value=True,
        ),
        EventStudyGroupSpec(
            "canonical_variant_multiplicity",
            ("canonical_variant_multiplicity_bucket", "trade_side"),
        ),
    ]


def prepare_section7r_signal_populations(
    signal_frame: pd.DataFrame,
    config: Section7Config | None = None,
) -> dict[str, pd.DataFrame]:
    """Build raw correlated variants and canonical economic-candidate rows."""

    cfg = config or Section7Config()
    missing = sorted(SECTION7R_REQUIRED_COLUMNS.difference(signal_frame.columns))
    if missing:
        raise KeyError(f"refined signal_frame is missing Section 7R columns: {missing}")
    if not signal_frame["poi_definition_version"].eq("6C_v1").all():
        raise ValueError("Section 7R accepts only poi_definition_version='6C_v1'")

    raw = prepare_section7_signal_frame(signal_frame, cfg)
    multiplicity = raw.groupby("canonical_candidate_id", observed=True)[
        "candidate_variant_id"
    ].transform("nunique")
    raw["canonical_variant_multiplicity"] = multiplicity.astype("int16")
    raw["canonical_variant_multiplicity_bucket"] = pd.Categorical(
        np.select(
            [multiplicity.eq(1), multiplicity.eq(2), multiplicity.between(3, 4), multiplicity.ge(5)],
            ["1", "2", "3-4", "5+"],
            default="unknown",
        ),
        categories=["1", "2", "3-4", "5+", "unknown"],
        ordered=True,
    )

    sort_cols = [
        "canonical_candidate_id",
        "structural_swing_break_flag",
        "structural_swing_window_broken",
        "swing_n",
        "break_mode",
        "candidate_variant_id",
    ]
    ascending = [True, False, False, True, True, True]
    canonical = (
        raw.sort_values(sort_cols, ascending=ascending, kind="mergesort", na_position="last")
        .drop_duplicates("canonical_candidate_id", keep="first")
        .reset_index(drop=True)
    )
    structural_any = raw.groupby("canonical_candidate_id", observed=True)[
        "structural_swing_break_flag"
    ].any()
    structural_window_max = raw.groupby("canonical_candidate_id", observed=True)[
        "structural_swing_window_broken"
    ].max()
    canonical["structural_swing_break_flag"] = (
        canonical["canonical_candidate_id"].map(structural_any).fillna(False).astype(bool)
    )
    canonical["local_swing_only_flag"] = ~canonical["structural_swing_break_flag"]
    canonical["structural_swing_window_broken"] = pd.to_numeric(
        canonical["canonical_candidate_id"].map(structural_window_max),
        errors="coerce",
    ).astype("Int64")
    canonical["structural_validation_bucket"] = np.where(
        canonical["structural_swing_break_flag"],
        "structural_validated",
        "local_swing_only",
    )
    canonical["structural_window_bucket"] = (
        canonical["structural_swing_window_broken"].astype("string").fillna("none")
    )
    if "canonical_retest_number" in canonical.columns:
        canonical["retest_number"] = canonical["canonical_retest_number"].astype("int32")
        canonical["first_touch_flag"] = canonical["retest_number"].eq(1)
        canonical["touch_order_bucket"] = np.where(
            canonical["first_touch_flag"], "first_touch", "later_touch"
        )
    return {
        "raw_variant_rows": raw,
        "canonical_candidate_variants": canonical,
    }


def build_section7r_event_study(
    signal_frame: pd.DataFrame,
    research_bars: pd.DataFrame,
    config: Section7Config | None = None,
    group_specs: list[EventStudyGroupSpec] | None = None,
) -> dict[str, pd.DataFrame | pd.Series | dict[str, float]]:
    """Rerun Section 7 on refined signals with raw/canonical populations."""

    cfg = config or Section7Config()
    specs = group_specs or default_section7r_group_specs()
    full_start = time.perf_counter()
    base_validation = build_section7_validation(signal_frame, research_bars, cfg)
    missing_refined = sorted(SECTION7R_REQUIRED_COLUMNS.difference(signal_frame.columns))
    if missing_refined:
        raise KeyError(f"refined signal_frame is missing Section 7R columns: {missing_refined}")
    bars = prepare_section7_bar_frame(research_bars)
    populations = prepare_section7r_signal_populations(signal_frame, cfg)

    summary_frames: list[pd.DataFrame] = []
    horizon_quality_records: list[dict[str, Any]] = []
    path_seconds = 0.0
    for horizon in cfg.forward_horizons:
        path_start = time.perf_counter()
        path_features = build_bar_horizon_path_features(bars, horizon, cfg)
        path_seconds += time.perf_counter() - path_start
        for population_name, signals in populations.items():
            for mode in ("fixed", "capped"):
                metrics = build_signal_horizon_metrics(signals, path_features, horizon, mode, cfg)
                valid = metrics["valid"].astype(bool)
                horizon_quality_records.append(
                    {
                        "analysis_population": population_name,
                        "horizon_minutes": horizon,
                        "metric_mode": mode,
                        "candidate_rows": len(signals),
                        "valid_count": int(valid.sum()),
                        "invalid_count": int((~valid).sum()),
                        "forced_exit_count": int(metrics["forced_exit_flag"].sum()),
                    }
                )
                summary = summarize_section7_groups(
                    signal_base=signals,
                    metrics=metrics,
                    group_specs=specs,
                    horizon=horizon,
                    metric_mode=mode,
                    config=cfg,
                )
                summary.insert(0, "analysis_population", population_name)
                summary_frames.append(summary)

    event_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    rankings: list[pd.DataFrame] = []
    for population_name in populations:
        population_summary = event_summary.loc[
            event_summary["analysis_population"].eq(population_name)
        ]
        ranking = rank_section7_candidate_groups(population_summary, cfg)
        if not ranking.empty:
            ranking.insert(0, "analysis_population", population_name)
            rankings.append(ranking)
    candidate_ranking = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    threshold_comparison = event_summary.loc[
        event_summary["group_spec"].isin(
            ["fvg_min_3_ticks", "fvg_min_4_ticks", "fvg_min_5_ticks", "fvg_tick_bucket"]
        )
    ].copy()

    validation = base_validation.copy()
    validation["refined_columns_present"] = len(missing_refined) == 0
    validation["only_6c_v1_signals"] = signal_frame["poi_definition_version"].eq("6C_v1").all()
    validation["three_tick_default_all_true"] = signal_frame["fvg_ge_3tick_flag"].astype(bool).all()
    validation["canonical_population_no_duplicate_candidates"] = populations[
        "canonical_candidate_variants"
    ]["canonical_candidate_id"].is_unique
    validation["raw_variant_rows"] = len(populations["raw_variant_rows"])
    validation["canonical_candidate_variant_rows"] = len(
        populations["canonical_candidate_variants"]
    )
    timings = {
        "section7r_path_label_construction_seconds": path_seconds,
        "section7r_full_runtime_seconds": time.perf_counter() - full_start,
    }
    return {
        "section7r_validation": validation,
        "section7r_horizon_quality": pd.DataFrame.from_records(horizon_quality_records),
        "section7r_event_study_summary": event_summary,
        "section7r_candidate_signal_ranking": candidate_ranking,
        "section7r_fvg_threshold_comparison": threshold_comparison,
        "section7r_timings": timings,
    }


def save_section7r_outputs(
    outputs: dict[str, pd.DataFrame | pd.Series | dict[str, float]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save revised outputs separately from legacy Section 7 files."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "event_study_summary": out_dir / "section7r_event_study_summary_gc.parquet",
        "candidate_signal_ranking": out_dir / "section7r_candidate_signal_ranking_gc.parquet",
        "fvg_threshold_comparison": out_dir / "section7r_fvg_threshold_comparison_gc.parquet",
    }
    outputs["section7r_event_study_summary"].to_parquet(paths["event_study_summary"], index=False)
    outputs["section7r_candidate_signal_ranking"].to_parquet(
        paths["candidate_signal_ranking"], index=False
    )
    outputs["section7r_fvg_threshold_comparison"].to_parquet(
        paths["fvg_threshold_comparison"], index=False
    )
    return paths
