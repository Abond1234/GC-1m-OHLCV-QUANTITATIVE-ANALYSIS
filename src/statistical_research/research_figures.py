"""External figure pack for the statistical-feature research notebook.

Every chart is written below ``reports/figures/statistical_feature_research``.
The functions intentionally close figures after saving so notebook execution
does not embed large visual outputs in the ``.ipynb`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DPI = 180
PRIMARY = "#1f77b4"
SECONDARY = "#ff7f0e"
POSITIVE = "#2a9d8f"
NEGATIVE = "#d1495b"
NEUTRAL = "#6c757d"


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _save(
    fig: plt.Figure,
    *,
    root: Path,
    section: str,
    slug: str,
    title: str,
    purpose: str,
    records: list[dict[str, Any]],
) -> None:
    directory = root / section
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    records.append(
        {
            "figure_number": len(records) + 1,
            "section": section,
            "title": title,
            "purpose": purpose,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
    )


def _variant_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["direction_variant"].str.replace("_benchmark", "", regex=False).str.title()
        + " / "
        + frame["gate_variant"].str.replace("_", " ", regex=False).str.title()
    )


def generate_statistical_research_figures(
    *,
    project_root: Path,
    coverage_by_product: pd.DataFrame,
    eligible_observations: pd.DataFrame,
    horizon_availability: pd.DataFrame,
    baseline_result: Any,
    feature_registry: pd.DataFrame,
    feature_diagnostics: pd.DataFrame,
    section_7_result: Any,
    feature_spread_sharpe: pd.DataFrame,
    section_8_result: Any,
    section_9_result: Any,
    section_10_result: Any,
    section_11_result: Any,
    backtest_sharpe: pd.DataFrame,
    backtest_daily_returns: pd.DataFrame,
    section_12_result: Any,
    section_12b_result: Any,
    garch_frame: pd.DataFrame,
    garch_threshold: float,
) -> pd.DataFrame:
    """Create the notebook's complete static diagnostic figure pack."""

    root = project_root / "reports" / "figures" / "statistical_feature_research"
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    # 1. Trusted source coverage -------------------------------------------------
    coverage = coverage_by_product.reset_index().rename(
        columns={coverage_by_product.index.name or "index": "product"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(
        coverage["product"].astype(str),
        coverage["rows"].astype(float) / 1e6,
        color=[PRIMARY, SECONDARY],
    )
    axes[0].set(title="Trusted source rows", ylabel="Rows (millions)", xlabel="Product")
    axes[1].bar(
        coverage["product"].astype(str),
        coverage["trading_dates"].astype(float),
        color=[PRIMARY, SECONDARY],
    )
    axes[1].set(title="Trading-date coverage", ylabel="New York trading dates", xlabel="Product")
    for ax in axes:
        _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section02_data",
        slug="01_trusted_source_coverage",
        title="Trusted source coverage",
        purpose="Confirms GC/MGC scale and date coverage before research filtering.",
        records=records,
    )

    # 2. Eligible population composition ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    session_counts = (
        eligible_observations["entry_session"]
        .astype(str)
        .value_counts()
        .reindex(["London", "New York"])
    )
    partition_counts = (
        eligible_observations["research_partition"]
        .astype(str)
        .value_counts()
        .reindex(["Development", "Validation", "Final test"])
    )
    axes[0].bar(session_counts.index, session_counts.values / 1e3, color=[PRIMARY, SECONDARY])
    axes[0].set(title="Eligible observations by session", ylabel="Observations (thousands)")
    axes[1].bar(
        partition_counts.index, partition_counts.values / 1e3, color=[PRIMARY, SECONDARY, NEUTRAL]
    )
    axes[1].set(title="Frozen research partitions", ylabel="Observations (thousands)")
    axes[1].tick_params(axis="x", rotation=18)
    for ax in axes:
        _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section03_population",
        slug="02_eligible_population",
        title="Eligible population composition",
        purpose="Shows sample balance across sessions and frozen partitions.",
        records=records,
    )

    # 3. Label availability ------------------------------------------------------
    availability = horizon_availability.sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(
        availability["horizon"],
        availability["availability_rate"],
        marker="o",
        color=PRIMARY,
        linewidth=2,
    )
    ax.set(
        title="Forward-label availability by horizon",
        xlabel="Horizon (minutes)",
        ylabel="Availability rate",
        ylim=(max(0.9, float(availability["availability_rate"].min()) - 0.01), 1.002),
    )
    for x, y in zip(availability["horizon"], availability["availability_rate"], strict=True):
        ax.annotate(
            f"{y:.2%}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8
        )
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section04_labels",
        slug="03_label_availability",
        title="Forward-label availability",
        purpose="Makes the sample loss from longer complete paths explicit.",
        records=records,
    )

    # 4. Baseline return quantiles ----------------------------------------------
    returns = baseline_result.unconditional.query(
        "outcome_name == 'forward_return_ticks'"
    ).sort_values("horizon_minutes")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for quantile, color in zip(
        ["p05", "p25", "median", "p75", "p95"],
        [NEGATIVE, "#e76f51", NEUTRAL, "#4c78a8", POSITIVE],
        strict=True,
    ):
        ax.plot(
            returns["horizon_minutes"], returns[quantile], marker="o", label=quantile, color=color
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="Unconditional forward-return distribution",
        xlabel="Horizon (minutes)",
        ylabel="Ticks",
    )
    ax.legend(ncol=5, frameon=False)
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section05_baseline",
        slug="04_return_quantiles",
        title="Unconditional return quantiles",
        purpose="Benchmarks center and tails before conditioning on features.",
        records=records,
    )

    # 5. Baseline session movement ----------------------------------------------
    session_range = baseline_result.session.query(
        "outcome_name == 'future_range_ticks'"
    ).sort_values("horizon_minutes")
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    for session, color in [("London", PRIMARY), ("New York", SECONDARY)]:
        sub = session_range.loc[session_range["entry_session"].astype(str).eq(session)]
        ax.plot(
            sub["horizon_minutes"], sub["mean"], marker="o", label=session, color=color, linewidth=2
        )
    ax.set(title="Mean future range by session", xlabel="Horizon (minutes)", ylabel="Ticks")
    ax.legend(frameon=False)
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section05_baseline",
        slug="05_session_movement",
        title="Session movement comparison",
        purpose="Separates London and New York opportunity scales.",
        records=records,
    )

    # 6. Feature family inventory ------------------------------------------------
    family_counts = (
        feature_registry.groupby("family", observed=True)
        .agg(features=("feature_name", "size"), experimental=("is_experimental", "sum"))
        .sort_values("features")
    )
    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.35 * len(family_counts))))
    ax.barh(
        family_counts.index.astype(str),
        family_counts["features"],
        color=PRIMARY,
        label="Core + experimental",
    )
    ax.barh(
        family_counts.index.astype(str),
        family_counts["experimental"],
        color=SECONDARY,
        label="Experimental subset",
    )
    ax.set(title="Engineered-feature inventory by family", xlabel="Registered features")
    ax.legend(frameon=False)
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section06_features",
        slug="06_feature_family_inventory",
        title="Feature family inventory",
        purpose="Shows breadth and experimental concentration of the engineered feature set.",
        records=records,
    )

    # 7. Feature missingness -----------------------------------------------------
    diagnostics = (
        feature_diagnostics.loc[
            feature_diagnostics["analysis_type"].eq("Development_feature_summary")
        ]
        .nlargest(20, "missing_rate")
        .sort_values("missing_rate")
    )
    fig, ax = plt.subplots(figsize=(9.8, 6.8))
    ax.barh(
        diagnostics["feature_name"],
        diagnostics["missing_rate"],
        color=np.where(diagnostics["missing_rate"].gt(0.05), SECONDARY, PRIMARY),
    )
    ax.set(title="Highest Development feature missingness", xlabel="Missing rate", ylabel="")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section06_features",
        slug="07_feature_missingness",
        title="Feature missingness diagnostics",
        purpose="Highlights coverage constraints before predictive evaluation.",
        records=records,
    )

    # 8. Validation IC leaderboard ----------------------------------------------
    validation_cells = section_7_result.cell_results.loc[
        section_7_result.cell_results["research_partition"].eq("Validation")
    ].copy()
    leaderboard = (
        validation_cells.assign(abs_ic=validation_cells["daily_ic_mean"].abs())
        .nlargest(20, "abs_ic")
        .sort_values("abs_ic")
    )
    labels = (
        leaderboard["feature_name"]
        + " | "
        + leaderboard["outcome_family"]
        + " | "
        + leaderboard["session"]
        + " | "
        + leaderboard["horizon_minutes"].astype(str)
        + "m"
    )
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    colors = np.where(leaderboard["daily_ic_mean"].ge(0), POSITIVE, NEGATIVE)
    ax.barh(labels, leaderboard["daily_ic_mean"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Strongest out-of-sample feature relationships", xlabel="Validation daily rank IC")
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section07_univariate",
        slug="08_validation_ic_leaderboard",
        title="Validation IC leaderboard",
        purpose="Ranks the strongest confirmed feature/outcome relationships without mixing sessions or horizons.",
        records=records,
    )

    # 9. Feature-spread Sharpe leaderboard --------------------------------------
    validation_sharpe = feature_spread_sharpe.loc[
        feature_spread_sharpe["research_partition"].eq("Validation")
    ].copy()
    sharpe_top = (
        validation_sharpe.dropna(subset=["annualized_sharpe"])
        .nlargest(20, "annualized_sharpe")
        .sort_values("annualized_sharpe")
    )
    labels = (
        sharpe_top["feature_name"]
        + " | "
        + sharpe_top["session"]
        + " | "
        + sharpe_top["horizon_minutes"].astype(str)
        + "m"
    )
    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    ax.barh(
        labels,
        sharpe_top["annualized_sharpe"],
        color=np.where(sharpe_top["annualized_sharpe"].ge(0), POSITIVE, NEGATIVE),
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title="Validation feature-spread Sharpe diagnostics",
        xlabel="Annualized daily spread Sharpe (screening only)",
    )
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section07_univariate",
        slug="09_feature_spread_sharpe",
        title="Feature-spread Sharpe leaderboard",
        purpose="Compares engineered features on equal-date top/bottom directional spreads; not executable PnL.",
        records=records,
    )

    # 10. IC versus feature-spread Sharpe ---------------------------------------
    directional_ic = validation_cells.loc[
        validation_cells["outcome_family"].eq("direction"),
        ["feature_name", "horizon_minutes", "session", "daily_ic_mean"],
    ]
    comparison = validation_sharpe.merge(
        directional_ic,
        on=["feature_name", "horizon_minutes", "session"],
        how="inner",
        suffixes=("", "_validation"),
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for session, color, marker in [("London", PRIMARY, "o"), ("New York", SECONDARY, "s")]:
        sub = comparison.loc[comparison["session"].eq(session)]
        ax.scatter(
            sub["daily_ic_mean"].abs(),
            sub["annualized_sharpe"],
            s=24,
            alpha=0.55,
            color=color,
            marker=marker,
            label=session,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="Predictive rank relationship versus economic spread",
        xlabel="Absolute Validation daily rank IC",
        ylabel="Validation annualized spread Sharpe",
    )
    ax.legend(frameon=False)
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section07_univariate",
        slug="10_ic_vs_spread_sharpe",
        title="IC versus spread Sharpe",
        purpose="Shows why statistical association and risk-adjusted directional spread are distinct tests.",
        records=records,
    )

    # 11. Yearly IC stability ----------------------------------------------------
    top_features = (
        sharpe_top.sort_values("annualized_sharpe", ascending=False)["feature_name"]
        .drop_duplicates()
        .head(5)
        .tolist()
    )
    yearly = section_7_result.yearly_stability.loc[
        section_7_result.yearly_stability["feature_name"].isin(top_features)
        & section_7_result.yearly_stability["outcome_family"].eq("direction")
        & section_7_result.yearly_stability["horizon_minutes"].eq(60)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, session in zip(axes, ["London", "New York"], strict=True):
        sub = yearly.loc[yearly["session"].eq(session)]
        for feature_name, group in sub.groupby("feature_name", sort=False):
            ax.plot(group["entry_year"], group["daily_ic_mean"], marker="o", label=feature_name)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=session, xlabel="Entry year", ylabel="Daily rank IC")
        _style_axis(ax)
    if top_features:
        axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Year-by-year stability of leading directional spread features")
    _save(
        fig,
        root=root,
        section="section07_univariate",
        slug="11_yearly_ic_stability",
        title="Yearly feature stability",
        purpose="Checks whether leading spread diagnostics depend on a single year.",
        records=records,
    )

    # 12. Redundancy clusters ----------------------------------------------------
    clusters = (
        section_8_result.cluster_members.groupby("cluster_id", observed=True)
        .agg(cluster_size=("feature_name", "size"), representative=("feature_name", "first"))
        .nlargest(15, "cluster_size")
        .sort_values("cluster_size")
    )
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.barh(
        [f"C{idx}: {row.representative}" for idx, row in clusters.iterrows()],
        clusters["cluster_size"],
        color=PRIMARY,
    )
    ax.set(title="Largest feature redundancy clusters", xlabel="Features in cluster")
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section08_redundancy",
        slug="12_redundancy_clusters",
        title="Feature redundancy clusters",
        purpose="Makes volatility and clock-family redundancy visible before multivariate modelling.",
        records=records,
    )

    # 13. Incremental IC ---------------------------------------------------------
    incremental = section_8_result.incremental_results.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)
    for ax, suffix, title in [(axes[0], "dev", "Development"), (axes[1], "val", "Validation")]:
        ax.scatter(
            incremental[f"raw_daily_ic_mean_{suffix}"],
            incremental[f"partial_daily_ic_mean_{suffix}"],
            c=np.where(incremental["passes_incremental_gates"], POSITIVE, NEUTRAL),
            alpha=0.65,
            s=32,
        )
        lim = max(
            0.05,
            float(
                np.nanmax(
                    np.abs(
                        incremental[
                            [f"raw_daily_ic_mean_{suffix}", f"partial_daily_ic_mean_{suffix}"]
                        ].to_numpy()
                    )
                )
            ),
        )
        ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set(title=title, xlabel="Raw daily IC", ylabel="Partial daily IC beyond anchor")
        _style_axis(ax)
    fig.suptitle("Incremental information after controlling for the anchor")
    _save(
        fig,
        root=root,
        section="section08_redundancy",
        slug="13_incremental_information",
        title="Incremental information",
        purpose="Separates genuinely incremental features from strong but redundant ones.",
        records=records,
    )

    # 14. Multivariate regression improvement ----------------------------------
    regression = section_9_result.regression_results.copy()
    validation_reg = regression.loc[regression["research_partition"].eq("Validation")].copy()
    validation_reg["cell"] = (
        validation_reg["session"] + " | " + validation_reg["horizon_minutes"].astype(str) + "m"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    x = np.arange(len(validation_reg))
    axes[0].bar(
        x - 0.18, validation_reg["anchor_daily_ic_mean"], width=0.36, label="Anchor", color=NEUTRAL
    )
    axes[0].bar(
        x + 0.18, validation_reg["model_daily_ic_mean"], width=0.36, label="Model", color=PRIMARY
    )
    axes[0].set_xticks(x, validation_reg["cell"], rotation=20)
    axes[0].set(title="Validation daily IC", ylabel="Daily rank IC")
    axes[0].legend(frameon=False)
    yerr = np.vstack(
        [
            validation_reg["ic_improvement_mean"] - validation_reg["ic_improvement_ci_low"],
            validation_reg["ic_improvement_ci_high"] - validation_reg["ic_improvement_mean"],
        ]
    )
    axes[1].errorbar(
        x, validation_reg["ic_improvement_mean"], yerr=yerr, fmt="o", capsize=4, color=PRIMARY
    )
    axes[1].axhline(0.02, color=SECONDARY, linestyle="--", label="Declared margin")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xticks(x, validation_reg["cell"], rotation=20)
    axes[1].set(title="Paired IC improvement", ylabel="Model minus anchor IC")
    axes[1].legend(frameon=False)
    for ax in axes:
        _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section09_models",
        slug="14_multivariate_regression",
        title="Multivariate regression performance",
        purpose="Compares the ridge model with the anchor and its declared Validation margin.",
        records=records,
    )

    # 15. Classification performance -------------------------------------------
    classification = section_9_result.classification_results.loc[
        section_9_result.classification_results["research_partition"].eq("Validation")
    ].copy()
    classification["cell"] = (
        classification["session"] + " | " + classification["horizon_minutes"].astype(str) + "m"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    x = np.arange(len(classification))
    axes[0].bar(x - 0.18, classification["anchor_auc"], width=0.36, label="Anchor", color=NEUTRAL)
    axes[0].bar(x + 0.18, classification["model_auc"], width=0.36, label="Model", color=PRIMARY)
    axes[0].set_xticks(x, classification["cell"], rotation=20)
    axes[0].set(title="Validation discrimination", ylabel="AUC", ylim=(0.5, 1.0))
    axes[1].bar(x - 0.18, classification["anchor_brier"], width=0.36, label="Anchor", color=NEUTRAL)
    axes[1].bar(x + 0.18, classification["model_brier"], width=0.36, label="Model", color=SECONDARY)
    axes[1].set_xticks(x, classification["cell"], rotation=20)
    axes[1].set(title="Validation probability error", ylabel="Brier score (lower is better)")
    for ax in axes:
        ax.legend(frameon=False)
        _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section09_models",
        slug="15_classification_performance",
        title="Classification performance",
        purpose="Compares discrimination and calibration loss without misusing Sharpe for an unsigned target.",
        records=records,
    )

    # 16. Signal construction diagnostics --------------------------------------
    gates = section_10_result.gate_thresholds.copy()
    candidates = section_10_result.candidates.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    x = np.arange(len(gates))
    axes[0].bar(
        x - 0.18, gates["development_gate_rate"], width=0.36, label="Development", color=PRIMARY
    )
    axes[0].bar(
        x + 0.18, gates["validation_gate_rate"], width=0.36, label="Validation", color=SECONDARY
    )
    axes[0].set_xticks(x, gates["session"])
    axes[0].set(
        title="Frozen opportunity-gate rates",
        ylabel="Share of candidates",
        ylim=(
            0,
            max(
                0.3,
                float(gates[["development_gate_rate", "validation_gate_rate"]].max().max()) + 0.05,
            ),
        ),
    )
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    stop_ticks = candidates["stop_points"].to_numpy(dtype=float) / float(
        section_10_result.config.tick_size
    )
    axes[1].hist(stop_ticks, bins=35, color=PRIMARY, alpha=0.85)
    axes[1].axvline(
        np.median(stop_ticks),
        color=SECONDARY,
        linestyle="--",
        label=f"Median {np.median(stop_ticks):.1f} ticks",
    )
    axes[1].set(
        title="Volatility-scaled stop distribution",
        xlabel="Stop distance (ticks)",
        ylabel="Candidates",
    )
    for ax in axes:
        ax.legend(frameon=False)
        _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section10_signals",
        slug="16_signal_diagnostics",
        title="Signal construction diagnostics",
        purpose="Shows out-of-sample gate retention and risk-rule scale before backtesting.",
        records=records,
    )

    # 17. Strategy Sharpe by costs ----------------------------------------------
    strategy = backtest_sharpe.copy()
    strategy["variant"] = _variant_label(strategy)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, partition in zip(axes, ["Development", "Validation"], strict=True):
        sub = strategy.loc[strategy["research_partition"].eq(partition)]
        pivot = sub.pivot(
            index="variant", columns="cost_scenario", values="annualized_sharpe"
        ).reindex(columns=["frictionless", "base", "pessimistic"])
        pivot.plot(kind="bar", ax=ax, color=[NEUTRAL, PRIMARY, NEGATIVE], width=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=partition, xlabel="", ylabel="Annualized daily net-R Sharpe")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Cost scenario", frameon=False, fontsize=8)
        _style_axis(ax)
    fig.suptitle("Sequential strategy Sharpe under transaction-cost scenarios")
    _save(
        fig,
        root=root,
        section="section11_backtest",
        slug="17_strategy_sharpe_by_cost",
        title="Strategy Sharpe by cost scenario",
        purpose="Primary Sharpe view for the only sequential, cost-adjusted return stream in the notebook.",
        records=records,
    )

    # 18. Expectancy versus Sharpe ----------------------------------------------
    base_perf = section_11_result.performance.loc[
        section_11_result.performance["cost_scenario"].eq("base")
    ].copy()
    base_perf = base_perf.merge(
        strategy.loc[
            strategy["cost_scenario"].eq("base"),
            ["direction_variant", "gate_variant", "research_partition", "annualized_sharpe"],
        ],
        on=["direction_variant", "gate_variant", "research_partition"],
        how="left",
    )
    base_perf["variant"] = _variant_label(base_perf)
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for partition, color, marker in [("Development", PRIMARY, "o"), ("Validation", SECONDARY, "s")]:
        sub = base_perf.loc[base_perf["research_partition"].eq(partition)]
        ax.scatter(
            sub["mean_net_r"],
            sub["annualized_sharpe"],
            s=80,
            color=color,
            marker=marker,
            label=partition,
        )
        for row in sub.itertuples():
            ax.annotate(
                row.variant,
                (row.mean_net_r, row.annualized_sharpe),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set(
        title="Base-cost expectancy versus daily Sharpe",
        xlabel="Mean net R per trade",
        ylabel="Annualized daily net-R Sharpe",
    )
    ax.legend(frameon=False)
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section11_backtest",
        slug="18_expectancy_vs_sharpe",
        title="Expectancy versus Sharpe",
        purpose="Tests whether average trade edge and time-series risk adjustment tell the same story.",
        records=records,
    )

    # 19. Daily equity curves ----------------------------------------------------
    base_daily = backtest_daily_returns.loc[
        backtest_daily_returns["cost_scenario"].eq("base")
    ].copy()
    base_daily["variant"] = _variant_label(base_daily)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=False)
    for ax, partition in zip(axes, ["Development", "Validation"], strict=True):
        sub = base_daily.loc[base_daily["research_partition"].eq(partition)]
        for variant, group in sub.groupby("variant", sort=False):
            ax.plot(group["trade_date_ny"], group["cumulative_net_r"], label=variant, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(title=partition, xlabel="Trading date", ylabel="Cumulative net R")
        ax.legend(frameon=False, fontsize=8)
        _style_axis(ax)
    fig.suptitle("Base-cost daily strategy equity curves")
    _save(
        fig,
        root=root,
        section="section11_backtest",
        slug="19_daily_equity_curves",
        title="Daily equity curves",
        purpose="Displays path dependence and persistent losses at the same daily frequency used for Sharpe.",
        records=records,
    )

    # 20. Profit factor and drawdown --------------------------------------------
    base_perf["variant_partition"] = base_perf["variant"] + " | " + base_perf["research_partition"]
    ordered = base_perf.sort_values("profit_factor")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    axes[0].barh(
        ordered["variant_partition"],
        ordered["profit_factor"],
        color=np.where(ordered["profit_factor"].ge(1), POSITIVE, NEGATIVE),
    )
    axes[0].axvline(1, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(title="Base-cost profit factor", xlabel="Gross gains / gross losses")
    axes[1].barh(ordered["variant_partition"], ordered["max_drawdown_r"], color=NEGATIVE)
    axes[1].set(title="Trade-sequence maximum drawdown", xlabel="R")
    for ax in axes:
        _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section11_backtest",
        slug="20_profit_factor_drawdown",
        title="Profit factor and drawdown",
        purpose="Complements Sharpe with payoff efficiency and path risk.",
        records=records,
    )

    # 21. Hybrid improvements ----------------------------------------------------
    hybrid = section_12_result.family_results.copy()
    hybrid["family"] = (
        hybrid["hypothesis"].str.title()
        + " / "
        + hybrid["trade_side"].str.title()
        + " | "
        + hybrid["research_partition"]
    )
    hybrid = hybrid.sort_values("improvement_mean_robust_r")
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    yerr = np.vstack(
        [
            hybrid["improvement_mean_robust_r"] - hybrid["improvement_daily_ci_low"],
            hybrid["improvement_daily_ci_high"] - hybrid["improvement_mean_robust_r"],
        ]
    )
    ax.errorbar(
        hybrid["improvement_mean_robust_r"],
        np.arange(len(hybrid)),
        xerr=yerr,
        fmt="o",
        capsize=3,
        color=PRIMARY,
    )
    ax.set_yticks(np.arange(len(hybrid)), hybrid["family"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title="Hybrid gate improvement with date-block uncertainty",
        xlabel="Gated minus baseline mean robust R",
    )
    _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section12_hybrid",
        slug="21_hybrid_improvements",
        title="Hybrid integration improvements",
        purpose="Shows effect size, uncertainty, and instability without treating overlapping events as strategy PnL.",
        records=records,
    )

    # 22. Opportunity-conditioned sizing ---------------------------------------
    sizing = section_12b_result.h1_effects.copy()
    sizing["scheme"] = (
        sizing["sizing_scheme"].str.title() + " | " + sizing["cost_scenario"].str.title()
    )
    sizing = sizing.sort_values("development_ratio_effect")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))
    y = np.arange(len(sizing))
    yerr = np.vstack(
        [
            sizing["development_ratio_effect"] - sizing["development_ci_low"],
            sizing["development_ci_high"] - sizing["development_ratio_effect"],
        ]
    )
    axes[0].errorbar(
        sizing["development_ratio_effect"], y, xerr=yerr, fmt="o", capsize=3, color=PRIMARY
    )
    axes[0].set_yticks(y, sizing["scheme"])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set(title="Development mean/MAD effect", xlabel="Effect with date-block interval")
    axes[1].scatter(
        sizing["development_ratio_effect"],
        sizing["validation_ratio_effect"],
        c=np.where(sizing["criteria_pass"], POSITIVE, NEGATIVE),
        s=70,
    )
    lim = max(
        0.05,
        float(
            np.nanmax(
                np.abs(sizing[["development_ratio_effect", "validation_ratio_effect"]].to_numpy())
            )
        ),
    )
    axes[1].plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].set(
        title="Development-to-Validation retention",
        xlabel="Development effect",
        ylabel="Validation effect",
    )
    for ax in axes:
        _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section12b_conditioning",
        slug="22_sizing_effects",
        title="Opportunity-conditioned sizing effects",
        purpose="Displays uncertainty, retention, and failed materiality criteria for H1.",
        records=records,
    )

    # 23. Exit and suppression effects ------------------------------------------
    h2 = section_12b_result.h2_results.copy()
    h3 = section_12b_result.h3_results.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].bar(h2["research_partition"], h2["conditioning_effect_r"], color=[PRIMARY, SECONDARY])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(
        title="H2 exit-horizon conditioning", ylabel="Conditional minus unconditional mean R"
    )
    x = np.arange(len(h3))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        h3["median_improvement_r"],
        width=width,
        label="Median R improvement",
        color=PRIMARY,
    )
    axes[1].bar(
        x + width / 2,
        h3["stop_rate_reduction"],
        width=width,
        label="Stop-rate reduction",
        color=SECONDARY,
    )
    axes[1].set_xticks(x, h3["research_partition"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="H3 bottom-quintile suppression", ylabel="Effect")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        _style_axis(ax)
    _save(
        fig,
        root=root,
        section="section12b_conditioning",
        slug="23_exit_and_suppression",
        title="Exit and suppression effects",
        purpose="Shows the Development-to-Validation sign failures for H2 and H3.",
        records=records,
    )

    # 24. GARCH regime history ---------------------------------------------------
    gc = garch_frame.loc[
        garch_frame["product"].eq("GC") & garch_frame["tradable_research_flag"]
    ].copy()
    daily_vol = gc.set_index("ts_event_utc")["feat_garch_cond_vol"].resample("1D").median().dropna()
    high_rate = (
        gc.set_index("ts_event_utc")["feat_garch_high_vol_regime"]
        .astype(float)
        .resample("1D")
        .mean()
        .reindex(daily_vol.index)
    )
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 7.0), sharex=True)
    axes[0].plot(daily_vol.index, daily_vol.values, color=SECONDARY, linewidth=1.0)
    axes[0].axhline(
        float(garch_threshold), color=NEGATIVE, linestyle="--", label="Development p75 threshold"
    )
    axes[0].set(title="Daily median GARCH conditional volatility", ylabel="Log-return volatility")
    axes[0].legend(frameon=False)
    axes[1].fill_between(high_rate.index, 0, high_rate.values, color=NEGATIVE, alpha=0.45)
    axes[1].set(
        title="Share of tradable GC bars in high-volatility regime",
        ylabel="Daily high-vol share",
        xlabel="UTC date",
        ylim=(0, 1),
    )
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for ax in axes:
        _style_axis(ax)
    _save(
        fig,
        root=root,
        section="garch_volatility",
        slug="24_garch_regimes",
        title="GARCH volatility regimes",
        purpose="Visualizes the risk-state feature without treating a volatility forecast as return performance.",
        records=records,
    )

    manifest = pd.DataFrame.from_records(records)
    manifest["path"] = manifest["path"].map(
        lambda value: str(Path(value).relative_to(project_root))
    )
    manifest_path = root / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest
