"""Append the completed Section 10 reader flow to the statistical notebook."""

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
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# 10.0 ")
    ),
    len(notebook.cells),
)
notebook.cells = notebook.cells[:section_start]

section_cells = [
    md(
        """
# 10.0 Statistical Signal Construction

Section 7 approved no directional feature, so this section does not invent a directional signal. It formalizes the candidate family the evidence supports for a sequential test: two declared benchmark directions (long, short), with and without the only evidence-based component available - an opportunity gate built from the Section 9 frozen per-session ridge model of 60-minute ATR-relative future range, thresholded at its Development 80th percentile. Stops are 1.5 times the decision-bar ATR clamped to 10 to 100 ticks, targets are 2R, holding is capped at 120 minutes, risk is one R per trade, and all locked session, noon, and 15:30 rules are inherited from the eligible-observation construction. Contract-count sizing is an execution-phase concern and is recorded as such.
"""
    ),
    code(
        """
from src.statistical_research.signal_construction import (
    Section10Config,
    build_signal_candidates,
    load_entry_fields,
    save_signal_outputs,
)

assert section_9_ready, "Section 10 requires the completed Section 9 benchmarks."

SECTION_10_STARTED = perf_counter()
section_10_config = Section10Config()
section_10_entry_fields = load_entry_fields(
    PROJECT_ROOT / "data" / "processed" / "statistical_research" / "forward_labels_gc.parquet"
)
section_10_result = build_signal_candidates(
    section_7_frame,
    section_10_entry_fields,
    section_9_result.coefficient_table,
    section_10_config,
)
display(section_10_result.rule_specification.to_frame())
display(section_10_result.summary.to_frame())
display(section_10_result.gate_thresholds)
"""
    ),
    md(
        """
## 10.1 Validation Checks, Save, and Completion Gate

The Development gate rate must sit at the declared 20 percent by construction; the Validation rate is an out-of-sample observation, not a fit.
"""
    ),
    code(
        """
display(section_10_result.validation_checks)
section_10_checks_passed = bool(section_10_result.validation_checks["passed"].all())
assert section_10_checks_passed, section_10_result.validation_checks.loc[
    ~section_10_result.validation_checks["passed"]
].to_string()

section_10_save_report = save_signal_outputs(section_10_result, project_root=PROJECT_ROOT)
display(section_10_save_report)
section_10_ready = bool(
    section_10_checks_passed and section_10_save_report["reload_row_match"].all()
)
print(f"section 10 runtime: {perf_counter() - SECTION_10_STARTED:.1f}s")
assert section_10_ready
print("SECTION 10 STATUS: READY" if section_10_ready else "SECTION 10 STATUS: NOT READY")
"""
    ),
]

notebook.cells.extend(section_cells)
nbformat.validate(notebook)
nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Section 10 cells appended: {len(section_cells)} (total cells: {len(notebook.cells)})")
