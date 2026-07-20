"""Append the completed Section 11 reader flow to the statistical notebook."""

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
    (i for i, cell in enumerate(notebook.cells) if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# 11.0 ")),
    len(notebook.cells),
)
notebook.cells = notebook.cells[:section_start]

section_cells = [
    md(
        """
# 11.0 Independent Sequential Backtest

A chronological, single-position, cost-aware simulation of the Section 10 candidate family. Entries fill at the recorded next-bar open and the mapping is verified against the bar open price for every trade. Exits are stop, target, 120-minute holding cap, or the forced 15:30 close; a bar touching both stop and target is scored stop-first and flagged ambiguous. Costs are declared assumptions in ticks (0.6 commission round trip plus 1 tick slippage per side in the base scenario, 2 ticks pessimistic), reported alongside a frictionless case. Development and Validation only.

The declared decision rule: a variant advances only with positive net base-scenario expectancy in both partitions. Anything else is a rejection of the standalone statistical system, which the project treats as a valid research outcome.
"""
    ),
    code(
        """
from src.statistical_research.sequential_backtest import (
    Section11Config,
    load_backtest_bars,
    run_sequential_backtest,
    save_backtest_outputs,
)

assert section_10_ready, "Section 11 requires the Section 10 candidate table."

SECTION_11_STARTED = perf_counter()
section_11_config = Section11Config()
section_11_bars = load_backtest_bars(
    PROJECT_ROOT / "data" / "processed" / "research_bars_gc_mgc_1m.parquet"
)
section_11_result = run_sequential_backtest(
    section_10_result.candidates, section_11_bars, section_11_config
)
del section_11_bars
display(section_11_result.summary.to_frame())
"""
    ),
    md(
        """
## 11.1 Validation Checks
"""
    ),
    code(
        """
display(section_11_result.validation_checks)
section_11_checks_passed = bool(section_11_result.validation_checks["passed"].all())
assert section_11_checks_passed, section_11_result.validation_checks.loc[
    ~section_11_result.validation_checks["passed"]
].to_string()
"""
    ),
    md(
        """
## 11.2 Performance, Verdicts, Save, and Completion Gate

Win rates near one third at a 2R target are what a directionless market produces; costs then set the sign. The expansion gate concentrates trades in high-opportunity periods but cannot supply direction, so gated variants are not better. Every variant is rejected: the standalone statistical system has no edge, and its validated value - opportunity forecasting - transfers to the hybrid integration phase.
"""
    ),
    code(
        """
section_11_performance = section_11_result.performance
display(
    section_11_performance.loc[section_11_performance["cost_scenario"].eq("base")]
)
display(section_11_result.verdicts)
display(
    section_11_result.yearly_performance.pivot_table(
        index=["direction_variant", "gate_variant", "entry_year"],
        columns="research_partition",
        values="mean_net_r",
        observed=True,
    ).round(3)
)

section_11_save_report = save_backtest_outputs(section_11_result, project_root=PROJECT_ROOT)
display(section_11_save_report)
section_11_ready = bool(
    section_11_checks_passed and section_11_save_report["reload_row_match"].all()
)
section_11_summary = pd.Series(
    {
        "total_trades": int(section_11_result.summary["total_trades"]),
        "positive_expectancy_variants": int(
            section_11_result.summary["positive_expectancy_variants"]
        ),
        "system_decision": section_11_result.summary[
            "standalone_statistical_system_decision"
        ],
        "section_runtime_seconds": round(perf_counter() - SECTION_11_STARTED, 1),
    },
    name="value",
)
display(section_11_summary.to_frame())
assert section_11_ready
print("SECTION 11 STATUS: READY" if section_11_ready else "SECTION 11 STATUS: NOT READY")
"""
    ),
]

notebook.cells.extend(section_cells)
nbformat.validate(notebook)
nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Section 11 cells appended: {len(section_cells)} (total cells: {len(notebook.cells)})")
