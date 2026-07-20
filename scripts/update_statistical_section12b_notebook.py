"""Append the completed Section 12B reader flow to the statistical notebook."""

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
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# 12B.0 ")
    ),
    len(notebook.cells),
)
notebook.cells = notebook.cells[:section_start]

section_cells = [
    md(
        """
# 12B.0 Opportunity-Conditioned Sizing, Exits, and Suppression

Section 12 rejected the frozen opportunity model as an entry filter. The pre-declared contract in `project_docs/section12b_research_contract.md` tests the roles magnitude information can legitimately play for the True POI continuation-short family: H1 position sizing (constant, proportional-to-quintile, inverse-to-quintile weight ladders), H2 exit-horizon conditioning restricted to the capped 60/120/240-minute labels of the frozen Section 7 grid, and H3 suppression of the bottom Development-fitted prediction quintile.

Everything binding was frozen before computation: the population is the Section 12 decision-bar join with the reproduction check enforced; quintiles are fitted on Development only; each Development effect needs a date-block bootstrap interval excluding zero, Validation sign agreement with at least 25 percent retention, and floors of 1,000/700 events across 200/120 dates on the affected subsets; H1 must additionally survive both `Section11Config` cost scenarios and a 0.05R mean-shift materiality bound, because a sizing scheme that materially moves mean R is a return bet, not a risk reallocation.
"""
    ),
    code(
        """
from src.research.opportunity_conditioning import (
    Section12BConfig,
    run_opportunity_conditioning,
    save_conditioning_outputs,
)

assert section_12_ready, "Section 12B requires the completed Section 12 integration."

SECTION_12B_STARTED = perf_counter()
section_12b_config = Section12BConfig()
section_12b_result = run_opportunity_conditioning(PROJECT_ROOT, section_12b_config)
display(section_12b_result.population_summary.to_frame())
display(section_12b_result.validation_checks)
section_12b_checks_passed = bool(section_12b_result.validation_checks["passed"].all())
assert section_12b_checks_passed
"""
    ),
    md(
        """
## 12B.1 Hypothesis Results

H1 is the instructive near-miss. Proportional sizing improves the mean/MAD ratio in every scenario with strong Validation retention (0.81-1.07), the most persistent positive signal the opportunity model has produced - but the Development interval includes zero in the base scenario, and the scheme lifts mean R by 0.04-0.08R in both partitions, failing the materiality bound. The lift comes from concentrating exposure in high-prediction quintiles, which is exactly the return effect Section 12 already rejected on floors and the final test; the bound prevents relabeling it as a sizing win. H2 fits 120m as the unconditional best horizon and its conditional mapping flips negative in Validation (retention -2.0). H3 improves both co-primary effects in Development and flips both in Validation.
"""
    ),
    code(
        """
display(section_12b_result.h1_effects.round(4))
display(section_12b_result.h2_mapping.round(4))
display(section_12b_result.h2_results.round(4))
display(section_12b_result.h3_results.round(4))
display(section_12b_result.verdicts)
"""
    ),
    md(
        """
## 12B.2 One-Time Final-Test Read and Family Closure

Read once after every verdict was fixed, feeding no criterion: every effect is within 0.03R of zero. No hypothesis advances. Per the linkage in `project_docs/section8_authorization_memo.md`, with the Option 1 sequential backtest already SEQUENTIAL_REJECTED at base costs and this contract negative, Option 3 applies: the S7P02 family is archived with the complete evidence chain - event-level edge confirmed frictionless, no surviving execution, sizing, exit, or suppression form.
"""
    ),
    code(
        """
display(section_12b_result.final_test_report.round(4))
section_12b_save_report = save_conditioning_outputs(section_12b_result, project_root=PROJECT_ROOT)
display(section_12b_save_report)
section_12b_ready = bool(
    section_12b_checks_passed and section_12b_save_report["reload_row_match"].all()
)
section_12b_summary = pd.Series(
    {
        "events_kept": int(section_12b_result.summary["events_kept"]),
        "h1_verdict": section_12b_result.summary["h1_verdict"],
        "h2_verdict": section_12b_result.summary["h2_verdict"],
        "h3_verdict": section_12b_result.summary["h3_verdict"],
        "family_decision": "S7P02 family archived per memo linkage (Option 3)",
        "section_runtime_seconds": round(perf_counter() - SECTION_12B_STARTED, 1),
    },
    name="value",
)
display(section_12b_summary.to_frame())
assert section_12b_ready
print("SECTION 12B STATUS: READY" if section_12b_ready else "SECTION 12B STATUS: NOT READY")
"""
    ),
]

notebook.cells.extend(section_cells)
nbformat.validate(notebook)
nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Section 12B cells appended: {len(section_cells)} (total cells: {len(notebook.cells)})")
