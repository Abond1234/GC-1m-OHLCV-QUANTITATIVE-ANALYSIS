"""Append the completed Section 7 reader flow to the statistical notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "exploration" / "statistical_feature_research.ipynb"


def md(source: str):
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip() + "\n")


notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
section_7_start = next(
    (i for i, cell in enumerate(notebook.cells) if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# 7.0 ")),
    len(notebook.cells),
)
notebook.cells = notebook.cells[:section_7_start]

section_7_cells = [
    md(
        """
# 7.0 Univariate Feature Evaluation

Section 7 screens every registered Section 6 predictor against the frozen Section 4 forward outcomes. The evaluation uses the **Development and Validation partitions only**: Final-test rows are excluded when the evaluation frame is constructed, and no feature-to-outcome relationship on the Final test is computed anywhere in this section. All bins and screen statistics are fitted on Development and applied unchanged to Validation.

The evidence unit is the per-New-York-trading-date cross-sectional Spearman rank information coefficient (IC), summarized with date-block bootstrap intervals, because overlapping minute observations are correlated events rather than independent trades. Broad Development screens are corrected with Benjamini-Hochberg false-discovery control within declared outcome-family/session screen families. Shortlist criteria were frozen in `Section7Config` before any result below was computed. Section 7 ranks candidate features only; it does not construct signals, select stops or targets, train models, or estimate PnL.
"""
    ),
    md(
        """
## 7.1 Evaluation Contract, Frozen Criteria, and Feature Scope

The screen covers the 80 numeric and 3 boolean registered predictors. The two categorical context fields are excluded with recorded reasons: `entry_session` is one of the declared stratification dimensions of the screen itself, and Section 5 already established descriptively that no weekday filter is approved. Feature distribution, missingness, and validity checks were completed by the Section 6 diagnostics and are not repeated here.
"""
    ),
    code(
        """
from time import perf_counter

from src.statistical_research.feature_evaluation import (
    EVALUATION_PARTITIONS,
    EVALUATION_SESSIONS,
    EXCLUDED_CATEGORICAL_FEATURES,
    Section7Config,
    build_evaluation_frame,
    build_univariate_evaluation,
    evaluated_feature_names,
    load_forward_labels_for_evaluation,
    save_evaluation_outputs,
)

assert section_6_ready, "Section 7 requires the validated Section 6 feature matrix."

SECTION_7_STARTED = perf_counter()
section_7_config = Section7Config()
section_7_criteria = pd.Series(
    {
        "evaluation_partitions": " + ".join(EVALUATION_PARTITIONS),
        "final_test_feature_outcome_relationships": "never computed in Section 7",
        "evidence_unit": "per-NY-date cross-sectional Spearman IC",
        "bootstrap": f"date-block, {section_7_config.bootstrap_replicates} replicates, seed {section_7_config.random_seed}",
        "quantile_buckets": section_7_config.quantile_bucket_count,
        "max_development_q_value": section_7_config.max_development_q_value,
        "min_dev_dates/obs": f"{section_7_config.min_development_trading_dates} / {section_7_config.min_development_observations:,}",
        "min_val_dates/obs": f"{section_7_config.min_validation_trading_dates} / {section_7_config.min_validation_observations:,}",
        "min_validation_ic_retention": section_7_config.min_validation_ic_retention,
        "min_dev_bucket_monotonicity": section_7_config.min_development_bucket_monotonicity,
        "direction_economic_gate": f"|Q-top - Q-bottom| >= {section_7_config.min_direction_spread_ticks} ticks in Dev AND Val",
    },
    name="frozen_value",
)
display(section_7_criteria.to_frame())
display(pd.Series(EXCLUDED_CATEGORICAL_FEATURES, name="exclusion_reason").to_frame())

section_7_features = evaluated_feature_names(feature_registry_gc)
assert len(section_7_features) == 83
"""
    ),
    md(
        """
## 7.2 Locked Evaluation Frame

The frame joins the in-memory Section 6 matrix with a column-scoped reload of the Section 4 forward labels. Rows must be Development or Validation and must have every horizon available with valid ATR normalization, so exactly the same observations support every horizon comparison. Large Section 6 construction intermediates are released first to respect the memory envelope.
"""
    ),
    code(
        """
for section_7_large_name in (
    "gc_feature_source", "research_bars", "gc_path_source",
    "manual_feature_audit", "reloaded_feature_matrix",
):
    globals().pop(section_7_large_name, None)

section_7_labels = load_forward_labels_for_evaluation(
    PROJECT_ROOT / "data" / "processed" / "statistical_research" / "forward_labels_gc.parquet"
)
section_7_frame, section_7_frame_summary = build_evaluation_frame(
    feature_matrix_gc, section_7_labels, feature_registry_gc
)
del section_7_labels
assert not section_7_frame["research_partition"].eq("Final test").any()
assert set(section_7_frame["research_partition"].unique()) <= set(EVALUATION_PARTITIONS)
display(section_7_frame_summary.to_frame())
"""
    ),
    md(
        """
## 7.3 Predictive Relationship — Daily Rank IC Screen

One pass produces, for every feature x outcome family x horizon x session x partition cell: daily IC mean, dispersion, t statistic, p-value, and a date-block bootstrap interval. The two outcome families are `direction` (signed `forward_return_{h}_atr`) and `expansion` (unsigned `future_range_{h}_atr`).
"""
    ),
    code(
        """
section_7_timer = perf_counter()
section_7_result = build_univariate_evaluation(
    section_7_frame, feature_registry_gc, section_7_config
)
print(f"univariate evaluation runtime: {perf_counter() - section_7_timer:.1f}s")
display(section_7_result.frame_summary.to_frame())
section_7_cells_frame = section_7_result.cell_results
display(
    section_7_cells_frame.loc[
        section_7_cells_frame["research_partition"].eq("Development")
    ]
    .reindex(columns=[
        "feature_name", "outcome_family", "horizon_minutes", "session",
        "daily_ic_mean", "daily_ic_ci_low", "daily_ic_ci_high", "development_q_value",
    ])
    .assign(abs_ic=lambda t: t["daily_ic_mean"].abs())
    .sort_values("abs_ic", ascending=False)
    .drop(columns="abs_ic")
    .head(15)
)
"""
    ),
    md(
        """
## 7.4 Validation Checks

Structural gates for the screen itself: partition and session containment, expected cell counts, q-values restricted to the Development screen, shortlist consistency, and the absence of any forward-looking column in the evaluated feature set. All checks must pass before any interpretation.
"""
    ),
    code(
        """
display(section_7_result.validation_checks)
section_7_checks_passed = bool(section_7_result.validation_checks["passed"].all())
assert section_7_checks_passed, section_7_result.validation_checks.loc[
    ~section_7_result.validation_checks["passed"]
].to_string()
"""
    ),
    md(
        """
## 7.5 Development-Fitted Quantile Analysis and Long/Short Framing

Bucket boundaries are quintiles fitted on Development per session and applied unchanged to Validation. For the `direction` family the signed-return IC and bucket spread describe both trade framings at once: Section 5 established that long and short signed returns are exact algebraic pairs, so a negative directional relationship is short-side information, not a missing test.
"""
    ),
    code(
        """
section_7_buckets = section_7_result.bucket_results
section_7_example_feature = (
    section_7_cells_frame.loc[
        section_7_cells_frame["outcome_family"].eq("expansion")
        & section_7_cells_frame["research_partition"].eq("Validation")
        & section_7_cells_frame["horizon_minutes"].eq(section_7_config.primary_horizon_minutes)
    ]
    .assign(abs_ic=lambda t: t["daily_ic_mean"].abs())
    .sort_values("abs_ic", ascending=False)["feature_name"]
    .iloc[0]
)
print(f"strongest expansion feature at the primary horizon: {section_7_example_feature}")
display(
    section_7_buckets.loc[
        section_7_buckets["feature_name"].eq(section_7_example_feature)
        & section_7_buckets["horizon_minutes"].eq(section_7_config.primary_horizon_minutes)
        & section_7_buckets["outcome_family"].eq("expansion")
    ].sort_values(["session", "research_partition", "bucket_index"])
)
"""
    ),
    md(
        """
## 7.6 Stability by Year

Per-year mean daily IC for the leading Validation-confirmed features. A candidate that depends on one exceptional year fails the declared criteria regardless of its pooled statistics.
"""
    ),
    code(
        """
section_7_top_features = (
    section_7_result.shortlist["feature_name"].drop_duplicates().head(8).tolist()
)
section_7_yearly = section_7_result.yearly_stability
display(
    section_7_yearly.loc[
        section_7_yearly["feature_name"].isin(section_7_top_features)
        & section_7_yearly["outcome_family"].eq("expansion")
        & section_7_yearly["horizon_minutes"].eq(section_7_config.primary_horizon_minutes)
    ]
    .pivot_table(
        index=["feature_name", "session"],
        columns="entry_year",
        values="daily_ic_mean",
        observed=True,
    )
    .round(3)
)
"""
    ),
    md(
        """
## 7.7 Stability by Session

London and New York are evaluated separately throughout; a shortlist row is session-specific by construction. The comparison below shows the primary-horizon Validation IC side by side for the leading features.
"""
    ),
    code(
        """
display(
    section_7_cells_frame.loc[
        section_7_cells_frame["feature_name"].isin(section_7_top_features)
        & section_7_cells_frame["research_partition"].eq("Validation")
        & section_7_cells_frame["horizon_minutes"].eq(section_7_config.primary_horizon_minutes)
    ]
    .pivot_table(
        index=["feature_name", "outcome_family"],
        columns="session",
        values="daily_ic_mean",
        observed=True,
    )
    .round(3)
)
"""
    ),
    md(
        """
## 7.8 Multiple-Testing Landscape

Benjamini-Hochberg q-values are computed on the Development screen only, within each outcome-family/session family of 498 related tests. With ~420,000 correlated observations, statistical detectability far exceeds economic usefulness — which is exactly why the shortlist requires Validation confirmation, monotonic structure, and an explicit tick-denominated economic gate rather than a q-value alone.
"""
    ),
    code(
        """
section_7_dev_cells = section_7_cells_frame.loc[
    section_7_cells_frame["research_partition"].eq("Development")
]
display(
    section_7_dev_cells.assign(
        q_significant=section_7_dev_cells["development_q_value"]
        <= section_7_config.max_development_q_value
    )
    .groupby(["outcome_family", "session"], observed=True)
    .agg(
        screen_cells=("feature_name", "size"),
        q_significant_cells=("q_significant", "sum"),
        median_abs_ic=("daily_ic_mean", lambda s: float(s.abs().median())),
    )
)
"""
    ),
    md(
        """
## 7.9 Candidate Feature Ranking, Verdicts, and Save

Every confirmation row is judged mechanically against the frozen criteria. Feature-level verdicts: `ADVANCE_DIRECTIONAL` (a direction cell passed everything including the economic gate), `ADVANCE_EXPANSION` (expansion evidence only), `WEAK_UNSTABLE` (Development screen passed somewhere but confirmation failed), `NO_EVIDENCE`. Experimental features carry their flag through the outputs and receive no special treatment.
"""
    ),
    code(
        """
display(section_7_result.feature_verdicts["verdict"].value_counts().to_frame("features"))
display(
    section_7_result.feature_verdicts.groupby(["verdict", "is_experimental"], observed=True)
    .size()
    .rename("features")
    .reset_index()
)
display(section_7_result.shortlist.head(20))

section_7_save_report = save_evaluation_outputs(
    section_7_result, project_root=PROJECT_ROOT
)
display(section_7_save_report)
"""
    ),
    md(
        """
## 7.10 Section 7 Summary and Completion Gate

Section 7 is complete when the structural checks pass, the saved outputs reload, and the shortlist/verdict artifacts exist. The Final test remains locked: advancing any shortlisted feature toward signal rules first requires the Section 8 redundancy and incremental-information review, and only a frozen shortlist may ever be examined against Final-test outcomes.
"""
    ),
    code(
        """
section_7_ready = bool(
    section_7_checks_passed
    and section_7_save_report["reload_row_match"].all()
    and len(section_7_result.feature_verdicts) == len(section_7_features)
)
section_7_summary = pd.Series(
    {
        "evaluation_observations": int(section_7_frame_summary["evaluation_observations"]),
        "evaluated_features": len(section_7_features),
        "screen_cells": int(section_7_result.frame_summary["screen_cells"]),
        "shortlist_cells": int(section_7_result.frame_summary["shortlist_cells"]),
        "advance_directional": int(section_7_result.frame_summary["advance_directional_features"]),
        "advance_expansion": int(section_7_result.frame_summary["advance_expansion_features"]),
        "weak_unstable": int(section_7_result.frame_summary["weak_unstable_features"]),
        "no_evidence": int(section_7_result.frame_summary["no_evidence_features"]),
        "section_runtime_seconds": round(perf_counter() - SECTION_7_STARTED, 1),
    },
    name="value",
)
display(section_7_summary.to_frame())
assert section_7_ready
print("SECTION 7 STATUS: READY" if section_7_ready else "SECTION 7 STATUS: NOT READY")
"""
    ),
]

notebook.cells.extend(section_7_cells)
nbformat.validate(notebook)
nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Section 7 cells appended: {len(section_7_cells)} (total cells: {len(notebook.cells)})")
