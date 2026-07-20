"""Append the completed Section 8 reader flow to the statistical notebook."""

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
section_8_start = next(
    (
        i
        for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# 8.0 ")
    ),
    len(notebook.cells),
)
notebook.cells = notebook.cells[:section_8_start]

section_8_cells = [
    md(
        """
# 8.0 Redundancy and Incremental Information

Section 8 reduces the Section 7 expansion advancers to a small frozen set of interpretable representatives. Development-only Spearman correlations are clustered hierarchically, one representative is selected per cluster by a mechanical rule declared before computation, and every non-anchor representative must demonstrate incremental information beyond the anchor — measured as a per-NY-date **partial rank IC** controlling for the anchor — on Development with Validation confirmation.

Section 7's directional result stands unchanged: no feature advanced for direction, so the frozen directional set is explicitly empty. The Final test remains locked; the evaluation frame is the same Development+Validation population as Section 7, and a frame containing Final-test rows is rejected with an error.
"""
    ),
    md(
        """
## 8.1 Frozen Contract and Inputs
"""
    ),
    code(
        """
from src.statistical_research.feature_redundancy import (
    Section8Config,
    build_redundancy_analysis,
    save_redundancy_outputs,
)

assert section_7_ready, "Section 8 requires the completed Section 7 evaluation."

SECTION_8_STARTED = perf_counter()
section_8_config = Section8Config()
section_8_contract = pd.Series(
    {
        "input_population": "Section 7 evaluation frame (Development + Validation only)",
        "candidate_features": "Section 7 ADVANCE_EXPANSION verdicts",
        "correlation_fit_scope": "Development partition only",
        "clustering": "average linkage on 1 - |Spearman rho|",
        "cluster_cut": f"|rho| >= {section_8_config.cluster_absolute_correlation_threshold}",
        "representative_rule": "best passing-cell Validation |IC| at 60m; ties -> core over experimental, then name",
        "incremental_metric": "per-NY-date partial rank IC beyond the anchor",
        "incremental_horizons": str(section_8_config.incremental_horizons),
        "incremental_gates": (
            f"dev |partial IC| >= {section_8_config.min_development_abs_partial_ic}, "
            f"val sign agreement, val retention >= {section_8_config.min_validation_partial_ic_retention}, "
            f"date floors {section_8_config.min_development_trading_dates}/{section_8_config.min_validation_trading_dates}"
        ),
        "directional_frozen_set": "explicitly empty (no Section 7 directional advancer)",
    },
    name="frozen_value",
)
display(section_8_contract.to_frame())
"""
    ),
    md(
        """
## 8.2 Run the Redundancy and Incremental Analysis
"""
    ),
    code(
        """
section_8_timer = perf_counter()
section_8_result = build_redundancy_analysis(
    section_7_frame,
    section_7_result.shortlist,
    section_7_result.feature_verdicts,
    feature_registry_gc,
    section_8_config,
)
print(f"section 8 analysis runtime: {perf_counter() - section_8_timer:.1f}s")
display(section_8_result.summary.to_frame())
"""
    ),
    md(
        """
## 8.3 Validation Checks
"""
    ),
    code(
        """
display(section_8_result.validation_checks)
section_8_checks_passed = bool(section_8_result.validation_checks["passed"].all())
assert section_8_checks_passed, section_8_result.validation_checks.loc[
    ~section_8_result.validation_checks["passed"]
].to_string()
"""
    ),
    md(
        """
## 8.4 Feature Correlations and Clusters

The largest clusters show the expected redundancy: the ATR/realized-volatility ladder collapses into one volatility-state cluster, and the session-clock features form small clusters of their own. Cluster membership below is fitted on Development only.
"""
    ),
    code(
        """
display(
    section_8_result.cluster_members.groupby("cluster_id")
    .agg(
        cluster_size=("feature_name", "size"),
        representative=("feature_name", lambda s: s.iloc[0]),
        members=("feature_name", lambda s: ", ".join(sorted(s))),
    )
    .sort_values("cluster_size", ascending=False)
    .head(12)
)
"""
    ),
    md(
        """
## 8.5 Incremental Information Beyond the Anchor

Raw IC measures a feature alone; the partial IC asks what remains after the anchor's information is removed date by date. Features with strong raw ICs but no confirmed partial IC are redundant with the volatility anchor, however impressive they look univariately.
"""
    ),
    code(
        """
section_8_incremental_view = section_8_result.incremental_results.reindex(columns=[
    "feature_name", "horizon_minutes", "session",
    "raw_daily_ic_mean_dev", "partial_daily_ic_mean_dev",
    "raw_daily_ic_mean_val", "partial_daily_ic_mean_val",
    "passes_incremental_gates",
])
display(
    section_8_incremental_view.assign(
        abs_partial_dev=section_8_incremental_view["partial_daily_ic_mean_dev"].abs()
    )
    .sort_values("abs_partial_dev", ascending=False)
    .drop(columns="abs_partial_dev")
    .head(15)
)
"""
    ),
    md(
        """
## 8.6 Frozen Candidate Feature Set, Save, and Completion Gate

The frozen set is the anchor plus every representative with confirmed incremental information. This set — not the 55 raw advancers — is the authorized input for Section 9 multivariate research. The directional frozen set is empty by evidence, not by omission.
"""
    ),
    code(
        """
display(section_8_result.frozen_feature_set)
section_8_save_report = save_redundancy_outputs(section_8_result, project_root=PROJECT_ROOT)
display(section_8_save_report)

section_8_ready = bool(
    section_8_checks_passed and section_8_save_report["reload_row_match"].all()
)
section_8_summary = pd.Series(
    {
        "expansion_advancers": int(section_8_result.summary["expansion_advancers"]),
        "clusters": int(section_8_result.summary["clusters"]),
        "representatives": int(section_8_result.summary["representatives"]),
        "anchor_feature": section_8_result.summary["anchor_feature"],
        "incremental_confirmed": int(section_8_result.summary["incremental_confirmed"]),
        "frozen_expansion_features": int(section_8_result.summary["frozen_expansion_features"]),
        "frozen_directional_features": int(section_8_result.summary["frozen_directional_features"]),
        "section_runtime_seconds": round(perf_counter() - SECTION_8_STARTED, 1),
    },
    name="value",
)
display(section_8_summary.to_frame())
assert section_8_ready
print("SECTION 8 STATUS: READY" if section_8_ready else "SECTION 8 STATUS: NOT READY")
"""
    ),
]

notebook.cells.extend(section_8_cells)
nbformat.validate(notebook)
nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Section 8 cells appended: {len(section_8_cells)} (total cells: {len(notebook.cells)})")
