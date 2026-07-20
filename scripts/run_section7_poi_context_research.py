from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Single-pass memory reference measured on the full population; the adaptive
# plan scales this estimate by the loaded signal count.
FULL_RUN_SIGNAL_ROWS = 1_080_394
FULL_RUN_PEAK_GB = 4.2
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.poi_context_features import (
    CONTEXT_BAR_COLUMNS,
    PoiContextConfig,
    build_true_poi_context_frame,
    refit_development_volatility_bucket,
    save_feature_registry,
)
from src.research.poi_context_event_study import (
    PoiContextResearchConfig,
    build_candidate_policy_results,
    build_directional_outcome_labels,
    build_feature_study_summary,
    build_interaction_summary,
    build_matched_control_location_summary,
    build_stop_target_summary,
    save_section7_outputs,
)
from src.resources import chronological_chunks, plan_processing


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 7 True POI context research")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--skip-stop-target", action="store_true")
    parser.add_argument(
        "--force-chunks",
        type=int,
        default=None,
        help="override the adaptive memory plan with an explicit chunk count",
    )
    args = parser.parse_args()

    processed = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    figures = reports / "figures" / "section7_poi_context"
    figures.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    timings: list[dict[str, float | str]] = []

    def stage(name: str, fn):
        start = time.perf_counter()
        before = process.memory_info().rss / (1024**2)
        value = fn()
        seconds = time.perf_counter() - start
        after = process.memory_info().rss / (1024**2)
        timings.append(
            {
                "stage": name,
                "seconds": seconds,
                "rss_before_mb": before,
                "rss_after_mb": after,
                "rss_delta_mb": after - before,
            }
        )
        print(f"[section7] {name}: {seconds:.2f}s, RSS {after:.1f} MB", flush=True)
        return value

    print("[section7] loading authoritative refined signal frame", flush=True)
    signals = pd.read_parquet(processed / "section6c_refined_signal_frame_gc.parquet")
    if args.start_date:
        signals = signals.loc[
            pd.to_datetime(signals["trade_date_ny"]) >= pd.Timestamp(args.start_date)
        ]
    if args.end_date:
        signals = signals.loc[
            pd.to_datetime(signals["trade_date_ny"]) <= pd.Timestamp(args.end_date)
        ]
    print(f"[section7] signal rows={len(signals):,}", flush=True)
    print("[section7] loading narrow GC bar base", flush=True)
    bars = pd.read_parquet(
        processed / "research_bars_gc_mgc_1m.parquet",
        columns=CONTEXT_BAR_COLUMNS,
        filters=[("product", "==", "GC")],
    )
    context_cfg = PoiContextConfig()
    research_cfg = PoiContextResearchConfig()
    registry_path = save_feature_registry(
        reports / "section7_true_poi_feature_registry.csv", context_cfg
    )
    # Adaptive memory plan: the context/label build is the single-pass memory
    # peak (about 4.2 GB for the full 1,080,394-row population).  Machines
    # without that headroom process chronological whole-day chunks instead;
    # context features and same-day labels are per-event against the full bar
    # history, so chunked output is identical to a single pass.
    estimated_peak_gb = max(FULL_RUN_PEAK_GB * len(signals) / FULL_RUN_SIGNAL_ROWS, 0.5)
    plan = plan_processing(estimated_peak_gb)
    chunk_count = args.force_chunks if args.force_chunks else plan.chunk_count
    suffix = f"; forced chunks={args.force_chunks}" if args.force_chunks else ""
    print(f"[section7] {plan.describe()}{suffix}", flush=True)

    if chunk_count > 1:
        date_chunks = chronological_chunks(
            pd.to_datetime(signals["trade_date_ny"]).to_numpy(), chunk_count
        )
        context_parts: list[pd.DataFrame] = []
        label_parts: list[pd.DataFrame] = []
        for index, chunk_dates in enumerate(date_chunks, start=1):
            chunk_signals = signals.loc[pd.to_datetime(signals["trade_date_ny"]).isin(chunk_dates)]
            chunk_context = stage(
                f"true_poi_context_frame_chunk_{index}_of_{len(date_chunks)}",
                lambda s=chunk_signals: build_true_poi_context_frame(s, bars, context_cfg),
            )
            label_parts.append(
                stage(
                    f"paired_directional_outcomes_chunk_{index}_of_{len(date_chunks)}",
                    lambda c=chunk_context: build_directional_outcome_labels(c, bars, research_cfg),
                )
            )
            context_parts.append(chunk_context)
        context = pd.concat(context_parts, ignore_index=True)
        labels = pd.concat(label_parts, ignore_index=True)
        del context_parts, label_parts
        # Development-fitted thresholds must see the full event population, so
        # they are refitted once on the concatenated frame; this reproduces the
        # single-pass output exactly.
        context = refit_development_volatility_bucket(context, context_cfg)
    else:
        context = stage(
            "true_poi_context_frame",
            lambda: build_true_poi_context_frame(signals, bars, context_cfg),
        )
        labels = stage(
            "paired_directional_outcomes",
            lambda: build_directional_outcome_labels(context, bars, research_cfg),
        )
    feature_summary = stage(
        "conditional_feature_studies",
        lambda: build_feature_study_summary(context, labels, research_cfg),
    )
    interaction_summary = stage(
        "pre_specified_interactions",
        lambda: build_interaction_summary(context, labels),
    )
    location_quality_summary = stage(
        "matched_control_location_quality",
        lambda: build_matched_control_location_summary(context, labels, bars),
    )
    if args.skip_stop_target:
        stop_target_summary = pd.DataFrame()
    else:
        stop_target_summary = stage(
            "stop_target_first_passage",
            lambda: build_stop_target_summary(context, labels, bars, research_cfg),
        )
    candidate_ranking, candidate_registry = stage(
        "candidate_policy_freeze_and_test",
        lambda: build_candidate_policy_results(context, bars, research_cfg),
    )
    outputs = {
        "context_frame": context,
        "outcome_labels": labels,
        "feature_study_summary": feature_summary,
        "interaction_summary": interaction_summary,
        "stop_target_summary": stop_target_summary,
        "candidate_ranking": candidate_ranking,
        "candidate_registry": candidate_registry,
        "location_quality_summary": location_quality_summary,
    }
    paths = stage("save_outputs", lambda: save_section7_outputs(outputs, processed))
    candidate_registry.to_csv(
        reports / "section7_true_poi_backtest_candidate_registry_gc.csv", index=False
    )
    candidate_ranking.to_csv(reports / "section7_true_poi_candidate_ranking_gc.csv", index=False)
    timings_frame = pd.DataFrame(timings)
    timings_frame.to_csv(reports / "section7_true_poi_runtime_diagnostics.csv", index=False)
    _save_figures(
        feature_summary, interaction_summary, stop_target_summary, candidate_ranking, figures
    )

    validation = {
        "true_retests_unique": context["true_retest_id"].is_unique,
        "true_trade_opportunities_unique_before_direction_expansion": context[
            "true_trade_opportunity_id"
        ].is_unique,
        "true_alias_matches_legacy_poi": context["true_poi_id"]
        .astype("string")
        .eq(context["canonical_poi_id"].astype("string"))
        .all(),
        "true_alias_matches_legacy_retest": context["true_retest_id"]
        .astype("string")
        .eq(context["canonical_retest_id"].astype("string"))
        .all(),
        "formation_before_touch": context["diag_formation_availability_bar_id"]
        .lt(context["retest_bar_id"])
        .all(),
        "pretouch_before_touch": context["diag_pretouch_availability_bar_id"]
        .lt(context["retest_bar_id"])
        .all(),
        "touch_close_at_touch": context["diag_touch_close_availability_bar_id"]
        .eq(context["retest_bar_id"])
        .all(),
        "paired_hypotheses_per_retest": labels.groupby("true_retest_id")["hypothesis"]
        .nunique()
        .eq(2)
        .all(),
        "no_entry_after_noon_in_context": context["feat_minute_of_day"].le(720).all(),
        "registry_written": registry_path.exists(),
    }
    validation_frame = pd.Series(validation, name="passed")
    validation_frame.to_csv(reports / "section7_true_poi_validation.csv", header=True)
    print("[section7] validation", flush=True)
    print(validation_frame.to_string(), flush=True)
    print("[section7] outputs", {k: str(v) for k, v in paths.items()}, flush=True)
    print(
        "[section7] counts",
        {
            "true_pois": context["true_poi_id"].nunique(),
            "true_retests": context["true_retest_id"].nunique(),
            "trading_dates": context["trade_date_ny"].nunique(),
            "directional_label_rows": len(labels),
            "candidate_policies": len(candidate_registry),
        },
        flush=True,
    )
    if not validation_frame.all():
        raise RuntimeError(
            f"Section 7 validation failed:\n{validation_frame[~validation_frame].to_string()}"
        )


def _save_figures(
    feature_summary: pd.DataFrame,
    interaction_summary: pd.DataFrame,
    stop_target_summary: pd.DataFrame,
    candidate_ranking: pd.DataFrame,
    output_dir: Path,
) -> None:
    if not feature_summary.empty:
        validation = feature_summary.loc[
            feature_summary["research_partition"].eq("validation")
        ].copy()
        effects = (
            validation.groupby("feature_name")["mean_r"]
            .agg(lambda s: s.max() - s.min())
            .nlargest(15)
        )
        if not effects.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            effects.sort_values().plot.barh(ax=ax, color="#3A6EA5")
            ax.set_title("Validation spread across locked development bins")
            ax.set_xlabel("Max minus min mean 60m R")
            fig.tight_layout()
            fig.savefig(output_dir / "feature_validation_effect_spread.png", dpi=150)
            plt.close(fig)
    if not interaction_summary.empty:
        data = interaction_summary.query(
            "research_partition == 'validation' and condition_met == True"
        )
        if not data.empty:
            pivot = data.pivot_table(
                index="interaction_name", columns="hypothesis", values="mean_r"
            )
            fig, ax = plt.subplots(figsize=(11, 7))
            pivot.plot.barh(ax=ax)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Pre-specified interaction validation results")
            ax.set_xlabel("Mean 60m R")
            fig.tight_layout()
            fig.savefig(output_dir / "interaction_validation_mean_r.png", dpi=150)
            plt.close(fig)
    if not stop_target_summary.empty:
        data = stop_target_summary.query(
            "analysis_type == 'first_passage_target' and research_partition == 'validation'"
        )
        if not data.empty:
            pivot = data.pivot_table(
                index="target_r", columns="stop_model", values="mean_realized_r"
            )
            fig, ax = plt.subplots(figsize=(11, 6))
            pivot.plot(ax=ax, marker="o")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title("Validation first-passage expectancy by stop and target")
            ax.set_ylabel("Mean realized R")
            fig.tight_layout()
            fig.savefig(output_dir / "stop_target_validation_expectancy.png", dpi=150)
            plt.close(fig)
    if not candidate_ranking.empty:
        data = candidate_ranking.loc[candidate_ranking["frozen_before_final_test"]]
        if not data.empty:
            pivot = data.pivot(
                index="policy_id", columns="research_partition", values="mean_realized_r"
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot.plot.bar(ax=ax)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title("Frozen candidate expectancy by chronological partition")
            ax.set_ylabel("Mean realized R")
            fig.tight_layout()
            fig.savefig(output_dir / "candidate_partition_expectancy.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
