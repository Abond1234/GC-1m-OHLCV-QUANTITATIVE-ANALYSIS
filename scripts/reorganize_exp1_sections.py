"""Reorganize exp1 Sections 6/7 without executing or changing code cells."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import nbformat


SECTION6_START_ID = "9a1e32cb"
LEGACY_SECTION7_START_ID = "7e076934"
REFINED_SECTION6_START_ID = "6c10a001"
REFINED_SECTION7_START_ID = "7r00a001"


SECTION6_HEADING_MAP = {
    "9a1e32cb": "# 6.0 POI Definition & Signal Feature Engineering",
    "4b4e4fa6": "## 6.1 Implementation Contract and Parameters",
    "6624a21a": "## 6.2 Baseline POI Research Tables",
    "ca97c6eb": "## 6.3 Baseline Validation and Diagnostics",
    "c6a48012": "### 6.3.1 POI, Retest, and Candidate Diagnostics",
    "40cfee35": "### 6.3.2 Save Baseline Research Tables",
    "27bb58ed": "### 6.3.3 Baseline Research Summary",
    "3df81ce7": "## 6.4 Structural Swing Validation",
    "8d986be4": "### 6.4.1 Baseline vs Structural-Layer Summary",
    "f93a1c6a": "### 6.4.2 Save Historical Structural Tables",
    "e7d20877": "### 6.4.3 Historical Structural Validation Summary",
    "6c10a001": "## 6.5 POI Definition Refinement and Final Research Contract",
    "6c10a003": "## 6.6 Final POI Formation Rules",
    "6c10a004": "## 6.7 POI Geometry: Standard and Expanded Cases",
    "6c10a005": "### 6.7.1 Standard Geometry",
    "6c10a006": "## 6.8 Confirmation-Candle Rejection Rule",
    "6c10a007": "## 6.9 FVG Threshold Sensitivity",
    "6c10a008": "## 6.10 False-FVG and Index-Alignment Audit",
    "6c10a010": "## 6.11 Build Final Refined POI Tables",
    "6c10a012": "## 6.12 Apply Structural Validation to Final POIs",
    "6c10a014": "## 6.13 Rebuild Retests, Candidate Trades and Signal Frame",
    "6c10a016": "## 6.14 Hand-Labelled Regression Validation",
    "6c10a017": "## 6.15 Full-Run Diagnostics and Save Outputs",
    "6c10a019": "## 6.16 Final Section 6 Summary",
}


APPENDIX_HEADING_MAP = {
    "e8f46819": "## A.1 Legacy Research Contract",
    "031043b5": "## A.2 Legacy Inputs and Validation",
    "1120e0e3": "## A.3 Legacy Baseline Forward-R Study",
    "c0d15bc5": "## A.4 Legacy POI Touch Event Study",
    "6403baf2": "## A.5 Legacy Structural Validation Study",
    "5ce639cf": "## A.6 Legacy Swing Window and Break-Mode Study",
    "b516e5c2": "## A.7 Legacy Entry Variant and Stop-Model Study",
    "3b735c4b": "## A.8 Legacy Session-Specific Results",
    "fc4c2062": "## A.9 Legacy Volatility-Regime Results",
    "9517bbb8": "## A.10 Legacy Volume and Relative-Volume Study",
    "8f9236bf": "## A.11 Legacy Trend, VWAP, Extension, and Exhaustion Study",
    "8e16134d": "## A.12 Legacy Candidate Signal Ranking",
    "2008fb21": "## A.13 Legacy Summary and Historical Backtest Candidates",
    "307aa5d6": "## A.14 Legacy Visual Audit Utility",
}


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_snapshot(nb: nbformat.NotebookNode) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        records.append(
            {
                "cell_id": cell.id,
                "original_index": index,
                "source_sha256": hashlib.sha256(cell.source.encode("utf-8")).hexdigest(),
                "outputs_sha256": _digest(cell.get("outputs", [])),
                "metadata_sha256": _digest(cell.get("metadata", {})),
                "execution_count": cell.get("execution_count"),
            }
        )
    return records


def _replace_heading(cell: nbformat.NotebookNode, heading: str) -> None:
    lines = cell.source.splitlines()
    if not lines:
        cell.source = heading
        return
    lines[0] = heading
    cell.source = "\n".join(lines)


def _clean_reader_facing_labels(cell: nbformat.NotebookNode) -> None:
    if cell.cell_type != "markdown":
        return
    replacements = {
        "Section 6B": "historical structural-validation layer",
        "Section 6C": "final refined Section 6",
        "Section 7R": "authoritative refined Section 7",
        "Section 7S": "current Section 7",
    }
    for old, new in replacements.items():
        cell.source = cell.source.replace(old, new)


def _index_by_id(cells: list[nbformat.NotebookNode], cell_id: str) -> int:
    matches = [index for index, cell in enumerate(cells) if cell.id == cell_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cell id {cell_id!r}; found {len(matches)}")
    return matches[0]


def reorganize_notebook(
    notebook_path: Path,
    backup_dir: Path,
) -> dict[str, Any]:
    nb = nbformat.read(notebook_path, as_version=4)
    original_cell_count = len(nb.cells)
    original_ids = [cell.id for cell in nb.cells]
    original_code = _code_snapshot(nb)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{notebook_path.stem}_pre_cleanup_{timestamp}.ipynb"
    manifest_path = backup_dir / f"{notebook_path.stem}_pre_cleanup_{timestamp}_manifest.json"
    shutil.copy2(notebook_path, backup_path)
    manifest_path.write_text(
        json.dumps(
            {
                "notebook": str(notebook_path),
                "backup": str(backup_path),
                "original_cell_count": original_cell_count,
                "original_code_cell_count": len(original_code),
                "original_cell_ids_sha256": _digest(original_ids),
                "code_cells": original_code,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    section6_start = _index_by_id(nb.cells, SECTION6_START_ID)
    legacy_start = _index_by_id(nb.cells, LEGACY_SECTION7_START_ID)
    refined6_start = _index_by_id(nb.cells, REFINED_SECTION6_START_ID)
    refined7_start = _index_by_id(nb.cells, REFINED_SECTION7_START_ID)
    if not (section6_start < legacy_start < refined6_start < refined7_start):
        raise ValueError("Unexpected pre-cleanup Section 6/7 block order")

    prefix = nb.cells[:section6_start]
    baseline6 = nb.cells[section6_start:legacy_start]
    legacy7 = nb.cells[legacy_start:refined6_start]
    refined6 = nb.cells[refined6_start:refined7_start]
    refined7 = nb.cells[refined7_start:]

    for cell in [*baseline6, *refined6]:
        if cell.id in SECTION6_HEADING_MAP:
            _replace_heading(cell, SECTION6_HEADING_MAP[cell.id])
        _clean_reader_facing_labels(cell)

    final_summary = next(cell for cell in refined6 if cell.id == "6c10a019")
    final_summary.source += (
        "\n\n**Authoritative project contract:** The refined POI definition is the "
        "authoritative project POI baseline. Earlier baseline and structural versions are "
        "retained only for historical comparison and regression testing. Future Section 7 "
        "research must use the refined Section 6 signal frame."
    )

    refined7_intro, refined7_code, refined7_findings = refined7
    refined7_intro.source = (
        "# 7.0 POI Context & Signal Research\n\n"
        "## 7.1 Research Contract and Authoritative Inputs\n\n"
        "The refined POI definition is frozen as the authoritative baseline. Section 7 now "
        "engineers and tests market context around each POI to distinguish continuation, "
        "reversal, and no-trade conditions. The authoritative input is the refined Section 6 "
        "signal frame (`section6c_refined_signal_frame_gc.parquet`). A full sequential "
        "backtest has not started."
    )
    section7_baseline_heading = nbformat.v4.new_markdown_cell(
        "## 7.2 Refined Baseline Event Study\n\n"
        "The existing refined event-study outputs establish the current baseline. Internal "
        "reproducibility names such as `section7r_*` remain unchanged, but they are the "
        "authoritative Section 7 results."
    )
    section7_baseline_heading.id = "7main002"
    _clean_reader_facing_labels(refined7_findings)
    refined7_findings.source = "## 7.3 Current Findings and Limitations\n\n" + refined7_findings.source
    section7_roadmap = nbformat.v4.new_markdown_cell(
        "## 7.4 Feature-Engineering and Conditional-Signal Roadmap\n\n"
        "The next research pass will evaluate these feature groups without changing the frozen "
        "POI definition:\n\n"
        "1. Approach-to-POI behaviour\n"
        "2. POI interaction and penetration\n"
        "3. Displacement quality\n"
        "4. Extension and exhaustion\n"
        "5. VWAP and trend context\n"
        "6. Volatility and relative-volume context\n"
        "7. Session context\n"
        "8. Continuation versus reversal labels\n"
        "9. Stop-before-target first-passage outcomes\n"
        "10. Stop-loss and target-policy research\n\n"
        "These features are a roadmap only; none are implemented in this notebook cleanup."
    )
    section7_roadmap.id = "7main004"
    section7_status = nbformat.v4.new_markdown_cell(
        "## 7.5 Section 7 Current Status\n\n"
        "The POI definition and refined baseline event study are complete. POI-context and "
        "conditional-signal research is the current phase. The purpose is to separate "
        "continuation, reversal, and no-trade conditions before any sequential backtest. "
        "Section 8 has not started."
    )
    section7_status.id = "7main005"
    authoritative7 = [
        refined7_intro,
        section7_baseline_heading,
        refined7_code,
        refined7_findings,
        section7_roadmap,
        section7_status,
    ]

    appendix_intro = legacy7[0]
    appendix_intro.source = (
        "# Appendix A — Legacy Prototype Event Study\n\n"
        "**Status:** Superseded by the refined Section 7 baseline.\n\n"
        "**Purpose:** Historical comparison, regression verification, and research "
        "traceability.\n\n"
        "**Authority:** Not authoritative for future strategy decisions. The code, saved "
        "outputs, figures, and original conclusions below are preserved as legacy evidence "
        "and must not be counted as independent current evidence."
    )
    for cell in legacy7[1:]:
        if cell.id in APPENDIX_HEADING_MAP:
            _replace_heading(cell, APPENDIX_HEADING_MAP[cell.id])
        _clean_reader_facing_labels(cell)
    legacy_summary = next(cell for cell in legacy7 if cell.id == "2008fb21")
    legacy_lines = legacy_summary.source.splitlines()
    legacy_lines.insert(
        1,
        "\n> Legacy conclusion retained for traceability; superseded by the authoritative Section 7 baseline.",
    )
    legacy_summary.source = "\n".join(legacy_lines)

    nb.cells = [*prefix, *baseline6, *refined6, *authoritative7, *legacy7]
    for cell in nb.cells:
        _clean_reader_facing_labels(cell)
    nbformat.validate(nb)
    nbformat.write(nb, notebook_path)

    written = nbformat.read(notebook_path, as_version=4)
    final_code = _code_snapshot(written)
    original_by_id = {record["cell_id"]: record for record in original_code}
    final_by_id = {record["cell_id"]: record for record in final_code}
    original_code_ids = set(original_by_id)
    final_code_ids = set(final_by_id)
    source_mismatches = sorted(
        cell_id
        for cell_id in original_code_ids & final_code_ids
        if original_by_id[cell_id]["source_sha256"] != final_by_id[cell_id]["source_sha256"]
    )
    output_mismatches = sorted(
        cell_id
        for cell_id in original_code_ids & final_code_ids
        if original_by_id[cell_id]["outputs_sha256"] != final_by_id[cell_id]["outputs_sha256"]
    )
    execution_mismatches = sorted(
        cell_id
        for cell_id in original_code_ids & final_code_ids
        if original_by_id[cell_id]["execution_count"] != final_by_id[cell_id]["execution_count"]
    )
    all_text = "\n".join(cell.source for cell in written.cells if cell.cell_type == "markdown")
    prohibited_heading_lines = [
        line
        for line in all_text.splitlines()
        if line.lstrip().startswith("#")
        and any(label in line for label in ("6B", "6C", "7R", "7S"))
    ]
    validation = {
        "backup_path": str(backup_path),
        "manifest_path": str(manifest_path),
        "original_cell_count": original_cell_count,
        "final_cell_count": len(written.cells),
        "cell_count_delta": len(written.cells) - original_cell_count,
        "original_code_cell_count": len(original_code),
        "final_code_cell_count": len(final_code),
        "missing_original_cell_ids": sorted(set(original_ids) - {cell.id for cell in written.cells}),
        "duplicate_final_cell_ids": sorted(
            cell_id
            for cell_id in {cell.id for cell in written.cells}
            if sum(cell.id == cell_id for cell in written.cells) > 1
        ),
        "missing_code_cell_ids": sorted(original_code_ids - final_code_ids),
        "new_code_cell_ids": sorted(final_code_ids - original_code_ids),
        "code_source_hash_mismatches": source_mismatches,
        "code_output_hash_mismatches": output_mismatches,
        "code_execution_count_mismatches": execution_mismatches,
        "prohibited_reader_facing_headings": prohibited_heading_lines,
        "appendix_is_last_block": written.cells[-len(legacy7)].id == LEGACY_SECTION7_START_ID,
        "section6_before_section7": _index_by_id(written.cells, SECTION6_START_ID)
        < _index_by_id(written.cells, REFINED_SECTION7_START_ID),
        "section7_before_appendix": _index_by_id(written.cells, REFINED_SECTION7_START_ID)
        < _index_by_id(written.cells, LEGACY_SECTION7_START_ID),
        "notebook_validated": True,
    }
    failed = {
        key: value
        for key, value in validation.items()
        if key
        in {
            "missing_original_cell_ids",
            "duplicate_final_cell_ids",
            "missing_code_cell_ids",
            "new_code_cell_ids",
            "code_source_hash_mismatches",
            "code_output_hash_mismatches",
            "code_execution_count_mismatches",
            "prohibited_reader_facing_headings",
        }
        and value
    }
    if not validation["appendix_is_last_block"]:
        failed["appendix_is_last_block"] = False
    if not validation["section6_before_section7"]:
        failed["section6_before_section7"] = False
    if not validation["section7_before_appendix"]:
        failed["section7_before_appendix"] = False
    validation_path = backup_dir / f"{notebook_path.stem}_cleanup_validation_{timestamp}.json"
    validation["validation_path"] = str(validation_path)
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(f"Notebook cleanup validation failed: {failed}")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/exploration/exp1.ipynb"),
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("logs/notebook_backups"),
    )
    args = parser.parse_args()
    result = reorganize_notebook(args.notebook, args.backup_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
