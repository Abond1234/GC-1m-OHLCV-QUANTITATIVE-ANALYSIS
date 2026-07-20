"""Append the completed Section 12 reader flow to the statistical notebook."""

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
    (
        i
        for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# 12.0 ")
    ),
    len(notebook.cells),
)
notebook.cells = notebook.cells[:section_start]

section_cells = [
    md(
        """
# 12.0 Hybrid Integration Research

Both branches are frozen, so the integration boundary is crossed for the first time, in one direction: Branch A True POI retests supply direction candidates and the frozen Branch B opportunity model supplies the filter and ranking. The declared question: does the frozen statistical gate add stable out-of-sample value beyond the POI baseline alone?

Rules frozen in `Section12Config` before computation: the outcome is Branch A's own headline metric (capped 60-minute R, robust-capped at 5R); the gate is the saved Section 10 model applied unchanged, with the full-coverage table verified to reproduce the saved Development/Validation predictions exactly before use; verdicts use Development and Validation only, with a date-block bootstrap on the pooled gated-minus-baseline improvement and pre-declared event/date floors; the Final test is read once, separately, after verdicts are fixed - mirroring the treatment Branch A's candidate policies received.
"""
    ),
    code(
        """
from src.research.hybrid_integration import (
    Section12Config,
    build_full_gate_table,
    build_hybrid_analysis,
    build_hybrid_event_frame,
    load_integration_inputs,
    save_hybrid_outputs,
)

assert section_11_ready, "Section 12 requires the completed Section 11 backtest."

SECTION_12_STARTED = perf_counter()
section_12_config = Section12Config()
section_12_labels, section_12_context, _ = load_integration_inputs(PROJECT_ROOT)
section_12_gate = build_full_gate_table(PROJECT_ROOT, section_12_config)
section_12_events, section_12_frame_summary = build_hybrid_event_frame(
    section_12_labels, section_12_context, section_12_gate, section_12_config
)
display(section_12_frame_summary.to_frame())
section_12_result = build_hybrid_analysis(section_12_events, section_12_config)
display(section_12_result.validation_checks)
section_12_checks_passed = bool(section_12_result.validation_checks["passed"].all())
assert section_12_checks_passed
"""
    ),
    md(
        """
## 12.1 Verdicts and Family Results

Continuation short is the instructive case: the gated Development and Validation means look dramatic (+1.17R and +0.87R improvements), but only 410 and 241 gated events across 33 and 30 trading dates sit behind them - below the pre-declared floors - so the verdict is no incremental value. The gate passes only about 5 percent of POI events because POIs already form in extended conditions where the model forecasts below-median relative expansion.
"""
    ),
    code(
        """
display(section_12_result.verdicts.round(4))
display(
    section_12_result.family_results.reindex(columns=[
        "hypothesis", "trade_side", "research_partition",
        "baseline_events", "baseline_mean_robust_r",
        "gated_events", "gated_trading_dates", "gated_mean_robust_r",
        "improvement_mean_robust_r", "improvement_daily_ci_low", "improvement_daily_ci_high",
    ]).round(4)
)
"""
    ),
    md(
        """
## 12.2 One-Time Frozen Final-Test Read

Read after the verdicts were fixed and feeding no criterion. On 301,610 final-test events the continuation-short improvement collapses to -0.02R: the small-sample Development/Validation effect does not generalize, which is precisely the outcome the sample floors guarded against. The frozen gate adds no reliable incremental value to any POI direction family in filter form.
"""
    ),
    code(
        """
display(section_12_result.final_test_report.round(4))
section_12_save_report = save_hybrid_outputs(section_12_result, project_root=PROJECT_ROOT)
display(section_12_save_report)
section_12_ready = bool(
    section_12_checks_passed and section_12_save_report["reload_row_match"].all()
)
section_12_summary = pd.Series(
    {
        "evaluation_events": int(section_12_result.summary["evaluation_events"]),
        "final_test_events": int(section_12_result.summary["final_test_events"]),
        "hybrid_advances": int(section_12_result.summary["hybrid_advances"]),
        "integration_decision": "gate-filter form adds no confirmed incremental value; "
        "sizing/no-trade integration forms remain open research",
        "section_runtime_seconds": round(perf_counter() - SECTION_12_STARTED, 1),
    },
    name="value",
)
display(section_12_summary.to_frame())
assert section_12_ready
print("SECTION 12 STATUS: READY" if section_12_ready else "SECTION 12 STATUS: NOT READY")
"""
    ),
]

notebook.cells.extend(section_cells)
nbformat.validate(notebook)
nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Section 12 cells appended: {len(section_cells)} (total cells: {len(notebook.cells)})")
