"""Append the completed Section 9 reader flow to the statistical notebook."""

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
section_start = next(
    (i for i, cell in enumerate(notebook.cells) if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# 9.0 ")),
    len(notebook.cells),
)
notebook.cells = notebook.cells[:section_start]

section_cells = [
    md(
        """
# 9.0 Multivariate Research

One question: does combining the Section 8 frozen expansion features beat the anchor `atr_20` alone? Features and outcomes enter as per-date cross-sectional rank z-scores, so the regression objective matches the daily-IC evidence unit used since Section 7. Ridge strength is selected by expanding-window walk-forward inside Development (year folds, one-trading-date embargo, which exceeds the longest 180-minute label); Validation is touched once per model after that choice is frozen. Models are fitted per session. A ridge-stabilized logistic benchmark predicts the frozen expansion classification label, reported with AUC, Brier score, and decile calibration. Tree models are deferred by contract until these linear benchmarks earn them.

Advancement requires the Validation daily IC of the model to beat the anchor-only benchmark by at least 0.02 with a date-block bootstrap interval on the paired daily difference excluding zero.
"""
    ),
    code(
        """
from src.statistical_research.multivariate import (
    Section9Config,
    build_multivariate_benchmarks,
    load_expansion_labels,
    save_multivariate_outputs,
)

assert section_8_ready, "Section 9 requires the frozen Section 8 feature set."

SECTION_9_STARTED = perf_counter()
section_9_config = Section9Config()
section_9_frozen_features = section_8_result.frozen_feature_set.loc[
    section_8_result.frozen_feature_set["in_frozen_set"], "feature_name"
].tolist()
section_9_anchor = section_8_result.summary["anchor_feature"]
section_9_expansion_labels = load_expansion_labels(
    PROJECT_ROOT / "data" / "processed" / "statistical_research" / "forward_labels_gc.parquet",
    section_9_config.horizons,
)
section_9_result = build_multivariate_benchmarks(
    section_7_frame,
    section_9_frozen_features,
    section_9_anchor,
    section_9_expansion_labels,
    section_9_config,
)
display(section_9_result.summary.to_frame())
"""
    ),
    md(
        """
## 9.1 Validation Checks
"""
    ),
    code(
        """
display(section_9_result.validation_checks)
section_9_checks_passed = bool(section_9_result.validation_checks["passed"].all())
assert section_9_checks_passed, section_9_result.validation_checks.loc[
    ~section_9_result.validation_checks["passed"]
].to_string()
"""
    ),
    md(
        """
## 9.2 Regression Benchmark and Verdicts

The model beats the anchor when the Validation improvement clears the declared margin with its bootstrap interval above zero. London at 180 minutes fails that margin and is recorded as anchor-sufficient.
"""
    ),
    code(
        """
section_9_regression = section_9_result.regression_results
display(section_9_regression.loc[section_9_regression["research_partition"].eq("Validation")])
display(section_9_result.verdicts)
"""
    ),
    md(
        """
## 9.3 Walk-Forward Selection Detail
"""
    ),
    code(
        """
display(
    section_9_result.walk_forward_results.pivot_table(
        index=["session", "horizon_minutes", "test_year"],
        columns="ridge_lambda",
        values="test_daily_ic_mean",
        observed=True,
    ).round(4)
)
"""
    ),
    md(
        """
## 9.4 Classification Benchmark and Calibration

The logistic model exceeds the anchor AUC in every Validation cell. Calibration is rank-correct for London and New York 180 minutes; New York 60 minutes overstates its upper probability deciles (predicted 0.65 to 0.82 against realized 0.41 to 0.55), so these probabilities require recalibration before any sizing use. That caveat is recorded in the summary.
"""
    ),
    code(
        """
section_9_classification = section_9_result.classification_results
display(section_9_classification.loc[section_9_classification["research_partition"].eq("Validation")])
display(
    section_9_result.calibration_table.pivot_table(
        index=["session", "horizon_minutes", "probability_decile"],
        values=["observation_count", "mean_predicted_probability", "realized_positive_rate"],
        observed=True,
    ).round(3)
)
"""
    ),
    md(
        """
## 9.5 Coefficients, Save, and Completion Gate
"""
    ),
    code(
        """
display(
    section_9_result.coefficient_table.pivot_table(
        index="feature_name", columns=["session", "horizon_minutes"],
        values="coefficient", observed=True,
    ).round(4)
)
section_9_save_report = save_multivariate_outputs(section_9_result, project_root=PROJECT_ROOT)
display(section_9_save_report)

section_9_ready = bool(
    section_9_checks_passed and section_9_save_report["reload_row_match"].all()
)
section_9_summary = pd.Series(
    {
        "model_advances": int(section_9_result.summary["model_advances"]),
        "anchor_sufficient": int(section_9_result.summary["anchor_sufficient"]),
        "best_validation_ic_improvement": float(
            section_9_result.summary["best_validation_ic_improvement"]
        ),
        "ny_60m_calibration": "requires recalibration before sizing use",
        "section_runtime_seconds": round(perf_counter() - SECTION_9_STARTED, 1),
    },
    name="value",
)
display(section_9_summary.to_frame())
assert section_9_ready
print("SECTION 9 STATUS: READY" if section_9_ready else "SECTION 9 STATUS: NOT READY")
"""
    ),
]

notebook.cells.extend(section_cells)
nbformat.validate(notebook)
nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Section 9 cells appended: {len(section_cells)} (total cells: {len(notebook.cells)})")
