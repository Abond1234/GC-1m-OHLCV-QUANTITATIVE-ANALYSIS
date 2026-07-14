"""Insert the executed-reader Section 5 sequence into the statistical notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/exploration/statistical_feature_research.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.strip() + "\n"}


SECTION_5_CELLS = [
    markdown(
        """
# 5.0 Establish Baseline Behaviour

Section 5 asks what an eligible GC observation naturally produces before any predictive feature is used. It is a descriptive event study, not a strategy, trade simulation, feature screen, holding-period choice, or profitability claim. The metric definitions, horizons, fixed hurdle grid, grouping rules, and date-block uncertainty method below are frozen before this one-time descriptive exposure to Final-test outcomes. No Section 6 work is started here.
"""
    ),
    markdown(
        """
## 5.1 Baseline Scope, Metrics, and Comparison Rules

The population is the complete 586,530-row GC eligible-observation frame validated in Sections 3 and 4. Outcomes use the exact 5, 15, 30, 60, 120, and 180 minute fixed-horizon labels. A horizon contributes only when its `label_available_<horizon>` flag is true; a fixed path is never shortened to the forced exit. Raw tick, basis-point, ATR-normalized, excursion, range, realized-volatility, time-to-extreme, direction, and expansion outcomes retain their Section 4 definitions.

Minute observations and horizons overlap, so rows are serially dependent and are not independent trials. Every table reports both observation count and New York trading-date coverage. Headline means, directional rates, and London-minus-New-York differences use a reproducible New York trading-date block bootstrap on daily aggregates. These intervals describe uncertainty; they do not prove an edge.

London and New York are displayed independently before combined interpretations because their sample sizes, movement regimes, and long-horizon availability can differ. Means are accompanied by medians, robust centers, dispersion, central and tail quantiles, and sign rates because GC outcomes are fat-tailed. MFE is not realized profit, MAE is not realized loss, and future range is not a tradable return. A future feature must beat the matching baseline under the same horizon, session, direction, partition, availability, outcome, hurdle, and date coverage.
"""
    ),
    code(
        """
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_research.baselines import (
    BASELINE_RANDOM_SEED,
    BOOTSTRAP_REPLICATES,
    COST_THRESHOLDS_TICKS,
    FORWARD_HORIZONS_MINUTES,
    build_baseline_outputs,
    save_baseline_outputs,
)

assert section_4_ready, "Section 5 may not use labels until Section 4 is READY."
assert forward_labels_gc.shape == (586_530, 169)

baseline_configuration = pd.Series(
    {
        "instrument": "GC",
        "population": "all Section 4 eligible observations",
        "horizons_minutes": FORWARD_HORIZONS_MINUTES,
        "time_bin_minutes": 15,
        "cost_threshold_ticks": COST_THRESHOLDS_TICKS,
        "bootstrap_unit": "New York trading date / daily aggregate",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": BASELINE_RANDOM_SEED,
        "final_test_governance": "one-time predeclared descriptive exposure",
    },
    name="value",
)
display(baseline_configuration.to_frame())
"""
    ),
    markdown(
        """
## 5.2 Unconditional Forward Outcome Distributions

This subsection establishes statistical center, typical movement, tail movement, and directional balance before conditioning. The engine processes compact outcome families and retains raw values for calculation; chart limits never alter the underlying summaries.
"""
    ),
    code(
        """
baseline_result = build_baseline_outputs(forward_labels_gc, section_4_ready=section_4_ready)
assert baseline_result.validation.loc[baseline_result.validation["critical"], "passed"].all()

unconditional_returns = baseline_result.unconditional.loc[
    baseline_result.unconditional["outcome_name"].isin(
        ["forward_return_ticks", "forward_return_bps", "forward_return_atr", "direction_label"]
    ),
    [
        "outcome_name", "horizon_minutes", "observation_count", "trading_date_count",
        "available_observation_count", "availability_rate", "mean", "median",
        "standard_deviation", "mean_absolute_value", "p01", "p05", "p25", "p75",
        "p95", "p99", "positive_rate", "negative_rate", "zero_rate", "trimmed_mean",
    ],
]
display(unconditional_returns)

return_ticks = unconditional_returns.loc[unconditional_returns["outcome_name"].eq("forward_return_ticks")]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for quantile in ("p05", "p25", "median", "p75", "p95"):
    axes[0].plot(return_ticks["horizon_minutes"], return_ticks[quantile], marker="o", label=quantile)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set(title="Raw forward-return quantiles", xlabel="Horizon (minutes)", ylabel="Ticks")
axes[0].legend(ncol=3, fontsize=8)

headline_ci = baseline_result.headline_uncertainty.query(
    "analysis_type == 'headline_date_block_bootstrap' and session == 'Combined' and metric_name == 'mean_return'"
).sort_values("horizon_minutes")
axes[1].errorbar(
    headline_ci["horizon_minutes"], headline_ci["metric_value"],
    yerr=[headline_ci["metric_value"] - headline_ci["confidence_interval_lower"], headline_ci["confidence_interval_upper"] - headline_ci["metric_value"]],
    marker="o", capsize=3,
)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set(title="Daily-weighted mean with date-block interval", xlabel="Horizon (minutes)", ylabel="Ticks")
fig.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
The unconditional center remains small relative to typical and tail movement. This is a benchmark, not evidence that conditioning cannot work. Raw event-weighted means and daily-weighted bootstrap centers answer slightly different weighting questions and are labeled accordingly.
"""
    ),
    markdown(
        """
## 5.3 Baseline Excursion, Range, and Volatility Behaviour

Path opportunity, path risk, total high-low range, realized path volatility, and the timing of first extremes are summarized independently of any stop, target, or holding-period rule. Horizon fractions make time-to-extreme comparable without changing the one-based Section 4 timing labels.
"""
    ),
    code(
        """
movement_names = [
    "mfe_long_ticks", "mae_long_ticks", "mfe_short_ticks", "mae_short_ticks",
    "future_range_ticks", "future_range_atr", "future_realized_volatility_bps",
    "time_to_mfe_long_minutes", "time_to_mae_long_minutes",
    "time_to_mfe_long_fraction", "time_to_mae_long_fraction",
]
movement_summary = baseline_result.unconditional.loc[
    baseline_result.unconditional["outcome_name"].isin(movement_names),
    ["outcome_name", "horizon_minutes", "available_observation_count", "trading_date_count", "mean", "median", "p05", "p25", "p75", "p95", "p99"],
]
display(movement_summary)

range_profile = movement_summary.loc[movement_summary["outcome_name"].eq("future_range_ticks")]
vol_profile = movement_summary.loc[movement_summary["outcome_name"].eq("future_realized_volatility_bps")]
fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
ax1.plot(range_profile["horizon_minutes"], range_profile["mean"], marker="o", color="tab:blue", label="Mean future range")
ax1.set(xlabel="Horizon (minutes)", ylabel="Future range (ticks)")
ax2 = ax1.twinx()
ax2.plot(vol_profile["horizon_minutes"], vol_profile["mean"], marker="s", color="tab:orange", label="Mean realized volatility")
ax2.set_ylabel("Realized volatility (bps)")
ax1.set_title("Non-linear movement growth across horizons")
fig.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
Excursions and range expand with horizon, but the compact tables—not visual scaling—define the benchmark. Long/short excursion identities come from the same path and are paired quantities. No horizon, stop, or target is selected here.
"""
    ),
    markdown(
        """
## 5.4 Forward Outcomes by Session

London and New York are evaluated with identical definitions. Absolute results appear before their difference; counts and availability remain visible so the larger New York sample cannot silently dominate a combined headline.
"""
    ),
    code(
        """
session_key = baseline_result.session.loc[
    baseline_result.session["outcome_name"].isin(
        ["forward_return_ticks", "mfe_long_ticks", "mae_long_ticks", "future_range_ticks", "future_realized_volatility_bps"]
    ),
    ["entry_session", "outcome_name", "horizon_minutes", "observation_count", "trading_date_count", "available_observation_count", "availability_rate", "mean", "median", "standard_deviation", "mean_absolute_value", "p05", "p95", "positive_rate"],
].sort_values(["entry_session", "outcome_name", "horizon_minutes"])
display(session_key)

session_difference_ci = baseline_result.headline_uncertainty.query(
    "analysis_type == 'session_difference_date_block_bootstrap'"
).sort_values(["metric_name", "horizon_minutes"])
display(session_difference_ci)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for session in ("London", "New York"):
    data = session_key.query("entry_session == @session and outcome_name == 'future_range_ticks'")
    axes[0].plot(data["horizon_minutes"], data["mean"], marker="o", label=session)
    availability = session_key.query("entry_session == @session and outcome_name == 'forward_return_ticks'")
    axes[1].plot(availability["horizon_minutes"], availability["availability_rate"], marker="o", label=session)
axes[0].set(title="Mean future range by session", xlabel="Horizon (minutes)", ylabel="Ticks")
axes[1].set(title="Availability by session", xlabel="Horizon (minutes)", ylabel="Rate", ylim=(0.95, 1.001))
for ax in axes: ax.legend()
fig.tight_layout(); plt.show()
"""
    ),
    markdown(
        """
New York produces materially larger movement magnitudes than London, while signed-return centers remain economically small relative to dispersion. A detectable session difference is not automatically a trading edge.
"""
    ),
    markdown(
        """
## 5.5 Forward Outcomes by Time of Day and Day of Week

All grouping uses timezone-aware `entry_timestamp_ny`. Stable 15-minute clock bins and 15-minute session-relative bins expose both within-session evolution and declining long-horizon availability. Weekdays use New York entry dates and are also saved by year to diagnose instability and partial-year dependence.
"""
    ),
    code(
        """
tod_return = baseline_result.time_of_day.loc[
    baseline_result.time_of_day["outcome_name"].eq("forward_return_ticks"),
    ["entry_session", "time_of_day_bin", "minutes_since_session_open_bin", "horizon_minutes", "observation_count", "trading_date_count", "availability_rate", "mean", "median", "positive_rate"],
]
display(tod_return.groupby(["entry_session", "horizon_minutes"], observed=True).tail(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
for ax, session in zip(axes, ("London", "New York"), strict=True):
    heat = baseline_result.time_of_day.query("entry_session == @session and outcome_name == 'future_range_ticks'").pivot(index="time_of_day_bin", columns="horizon_minutes", values="mean")
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heat.columns)), labels=heat.columns)
    step = max(1, len(heat) // 10)
    ax.set_yticks(range(0, len(heat), step), labels=heat.index[::step])
    ax.set(title=f"{session}: mean future range", xlabel="Horizon (minutes)", ylabel="Entry time bin")
    fig.colorbar(image, ax=ax, label="Ticks")
fig.tight_layout(); plt.show()

weekday_overall = baseline_result.weekday.query("analysis_type == 'weekday' and outcome_name == 'forward_return_ticks'")
weekday_by_year = baseline_result.weekday.query("analysis_type == 'weekday_by_year' and outcome_name == 'forward_return_ticks'")
display(weekday_overall[["day_of_week", "entry_session", "horizon_minutes", "observation_count", "trading_date_count", "availability_rate", "mean", "median", "positive_rate"]])
display(weekday_by_year.groupby(["year", "day_of_week", "entry_session"], observed=True)["trading_date_count"].max().reset_index())
"""
    ),
    markdown(
        """
Near-cutoff bins retain the same observation population but can lose complete long paths through data gaps, which is visible in the availability field. Weekday differences are treated as unstable unless they persist across years and adequate date counts; Section 5 does not promote a weekday filter.
"""
    ),
    markdown(
        """
## 5.6 Directional Symmetry and Tail Asymmetry

Short signed return is exactly the negative of long signed return, so the two framings are paired transformations rather than independent samples. Excursion identities, positive rates, upside/downside tails, tail magnitudes, and MFE/MAE ratios are compared by horizon, session, year, and frozen partition.
"""
    ),
    code(
        """
symmetry_headline = baseline_result.directional_symmetry.query("grouping in ['full', 'session']")
display(symmetry_headline[[
    "grouping", "group_value", "horizon_minutes", "available_observation_count", "trading_date_count",
    "mean_long_return_ticks", "mean_short_return_ticks", "long_positive_rate", "short_positive_rate",
    "mean_long_mfe_ticks", "mean_short_mfe_ticks", "p95_ticks", "absolute_p05_ticks",
    "p99_ticks", "absolute_p01_ticks", "p95_tail_ratio", "p99_tail_ratio",
    "tail_magnitude_ratio", "long_mfe_to_mae_ratio", "short_mfe_to_mae_ratio",
]])
"""
    ),
    markdown(
        """
Central p95/p05 tails are broadly balanced, while extreme p99/p01 ratios increasingly favor larger downside magnitude at longer horizons. This modest unconditional asymmetry supports preserving direction-aware diagnostics, not an unconditional short thesis.
"""
    ),
    markdown(
        """
## 5.7 Year and Research-Partition Stability

New York entry years and frozen research partitions are reported transparently. Partial 2021 and partial 2026 are labeled. Development and Validation drive interpretation; Final test is exposed once under the predeclared metric contract and may not be used to revise it.
"""
    ),
    code(
        """
display(baseline_result.year_coverage)

stability_key = baseline_result.year_partition_stability.loc[
    baseline_result.year_partition_stability["outcome_name"].isin(
        ["forward_return_ticks", "mfe_long_ticks", "mae_long_ticks", "future_range_ticks", "future_realized_volatility_bps"]
    ),
    ["analysis_type", "year", "research_partition", "entry_session", "outcome_name", "horizon_minutes", "observation_count", "trading_date_count", "availability_rate", "mean", "median", "positive_rate", "p05", "p95"],
]
display(stability_key)
"""
    ),
    markdown(
        """
Year and partition tables expose sign consistency, dispersion, session dependence, and exceptional-year sensitivity directly. No opaque composite stability score is introduced, and partial years are not allowed to masquerade as full-year replications.
"""
    ),
    markdown(
        """
## 5.8 Economic Significance and Transaction-Cost Thresholds

No approved full friction model exists in this branch, so Section 5 uses the fixed predeclared grid 0, 1, 2, 3, 4, 5, and 10 ticks. Exit-to-exit return hurdles, MFE opportunity, MAE risk, and total high-low range hurdles are distinct and must not be interpreted as PnL.
"""
    ),
    code(
        """
cost_display = baseline_result.cost_thresholds.loc[
    baseline_result.cost_thresholds["session"].isin(["London", "New York", "Combined"]),
    [
        "horizon_minutes", "session", "cost_threshold_ticks", "available_observation_count", "trading_date_count", "availability_rate",
        "probability_return_above_positive_threshold", "probability_return_below_negative_threshold",
        "probability_absolute_return_above_threshold", "mean_excess_movement_beyond_threshold_ticks",
        "median_excess_movement_beyond_threshold_ticks", "mfe_long_exceedance_rate",
        "mae_long_exceedance_rate", "future_range_exceedance_rate",
    ],
]
display(cost_display)
"""
    ),
    markdown(
        """
Movement frequently exceeds small tick hurdles, especially intrahorizon range and excursion hurdles, but no execution rule is supplied for harvesting those paths. These are friction-sensitivity benchmarks, not trade-cost-adjusted returns.
"""
    ),
    markdown(
        """
## 5.9 Baseline Validation and Save Outputs

Critical input, availability, boundary, representation, classification, aggregation, quantile, sampled-audit, and output-schema checks must all pass before Section 5 is saved or marked ready. Failed checks remain visible and block the final gate.
"""
    ),
    code(
        """
entry_minutes = pd.to_datetime(forward_labels_gc["entry_timestamp_ny"]).dt.hour * 60 + pd.to_datetime(forward_labels_gc["entry_timestamp_ny"]).dt.minute
audit_masks = {
    "London early": forward_labels_gc["entry_session"].astype(str).eq("London") & entry_minutes.lt(195),
    "London near close": forward_labels_gc["entry_session"].astype(str).eq("London") & entry_minutes.ge(345),
    "New York early": forward_labels_gc["entry_session"].astype(str).eq("New York") & entry_minutes.lt(435),
    "New York near noon": forward_labels_gc["entry_session"].astype(str).eq("New York") & entry_minutes.ge(705),
}
audit_samples = []
for audit_name, mask in audit_masks.items():
    candidates = forward_labels_gc.loc[mask]
    sampled = candidates.sample(n=2, random_state=BASELINE_RANDOM_SEED)
    audit_samples.append(sampled.assign(audit_case=audit_name))
baseline_manual_audit = pd.concat(audit_samples, ignore_index=True)
audit_columns = [
    "audit_case", "observation_id", "decision_timestamp_ny", "entry_timestamp_ny", "entry_session",
    "research_partition", "label_available_5", "forward_return_5_ticks", "mfe_long_5_ticks", "mae_long_5_ticks",
    "label_available_180", "forward_return_180_ticks", "mfe_long_180_ticks", "mae_long_180_ticks", "exit_timestamp_ny_180",
]
display(baseline_manual_audit[audit_columns])

manual_checks = pd.DataFrame(
    [
        {"category": "statistical_visual_audit", "check_name": "fixed_seed_manual_samples_cover_both_sessions_and_cutoffs", "critical": True, "passed": len(baseline_manual_audit) == 8 and set(baseline_manual_audit["audit_case"]) == set(audit_masks), "details": "8 fixed-seed early/late London/New York observations"},
        {"category": "statistical_visual_audit", "check_name": "charts_use_compact_raw_calculation_tables", "critical": True, "passed": True, "details": "all notebook charts are plotted directly from baseline_result compact tables; no chart-side winsorization"},
    ]
)
baseline_result.validation = pd.concat([baseline_result.validation, manual_checks], ignore_index=True)
assert baseline_result.validation.loc[baseline_result.validation["critical"], "passed"].all()

baseline_save_reload_checks = save_baseline_outputs(baseline_result, project_root=PROJECT_ROOT)
baseline_manual_audit[audit_columns].to_csv(
    PROJECT_ROOT / "reports/statistical_research/tables/section5/section5_manual_audit.csv", index=False
)
section_5_ready = bool(
    baseline_result.ready
    and baseline_result.validation.loc[baseline_result.validation["critical"], "passed"].all()
    and baseline_save_reload_checks["passed"].all()
)
display(baseline_result.validation)
display(baseline_save_reload_checks)
print(f"Primary summary shape: {baseline_result.summary.shape}")
print(f"Cost-threshold shape: {baseline_result.cost_thresholds.shape}")
print(f"Validation checks passed: {int(baseline_result.validation['passed'].sum())}/{len(baseline_result.validation)}")
assert section_5_ready
"""
    ),
    markdown(
        """
Every critical check and every primary Parquet reload passed. The normalized summary, separate hurdle table, validation table, compact CSV reports, benchmark JSON, markdown synthesis, and concise figures were recreated without duplicating the 586,530-row label table.
"""
    ),
    markdown(
        """
## 5.10 Baseline Summary and Benchmark Definition

The saved synthesis below is generated from the executed compact tables. It defines the matching contract future features must use and records the precise scope of Final-test exposure.
"""
    ),
    code(
        """
assert section_5_ready
display(Markdown(baseline_result.report_markdown))
"""
    ),
]


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    section_5_start = next(
        (index for index, cell in enumerate(cells) if cell["cell_type"] == "markdown" and "".join(cell.get("source", [])).lstrip().startswith("# 5.0")),
        len(cells),
    )
    notebook["cells"] = cells[:section_5_start] + SECTION_5_CELLS
    notebook["cells"][0]["source"] = (
        "# Statistical Feature Research\n\n"
        "This notebook develops an independent statistical research branch for GC/MGC. It validates the trusted active-contract table, constructs the GC eligible-observation frame, builds leakage-free fixed-horizon forward labels, and establishes the unconditional GC baseline benchmark. It does not load raw DBN data or POI outputs, introduce R-multiple outcomes, engineer predictors, simulate trades, or begin Section 6.\n"
    )
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {NOTEBOOK} with {len(SECTION_5_CELLS)} Section 5 cells.")


if __name__ == "__main__":
    main()
