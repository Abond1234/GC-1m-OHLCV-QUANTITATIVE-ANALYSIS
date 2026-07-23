"""Build the data-backed final and context reports for MLAT Feature Research v1.

This script is intentionally downstream of the executed notebook.  It refuses
to write either report unless the Stage-1 coverage record, executed notebook,
versioned evidence tables, execution manifest, and artifact-integrity controls
are all present and internally consistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "project_docs" / "mlat_feature_research"
NOTEBOOK = ROOT / "notebooks" / "exploration" / "mlat_feature_research.ipynb"
REPORT_ROOT = ROOT / "reports" / "statistical_research" / "mlat_feature_research" / "v1"
TABLE_ROOT = REPORT_ROOT / "tables"
DATA_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "statistical_research"
    / "mlat_feature_research"
    / "v1"
)
FIGURE_ROOT = ROOT / "reports" / "figures" / "mlat_feature_research" / "v1"
EXECUTION_MANIFEST = REPORT_ROOT / "execution_manifest.json"
ARTIFACT_MANIFEST = REPORT_ROOT / "table_manifest.json"
FINAL_REPORT = DOCS / "mlat_final_research_report.md"
CONTEXT_REPORT = DOCS / "mlat_feature_research_context_report.md"

OLD_NOTEBOOK = ROOT / "notebooks" / "exploration" / "statistical_feature_research.ipynb"
OLD_CONTEXT_REPORT = DOCS.parent / "statistical_feature_research_context_report.md"

STAGE1_FILES = (
    "README.md",
    "mlat_book_map.md",
    "mlat_book_coverage_log.md",
    "mlat_book_ingestion_manifest.csv",
    "mlat_current_state_audit.md",
    "mlat_upstream_artifact_manifest.csv",
    "mlat_concept_registry.md",
    "mlat_concept_registry.csv",
    "mlat_formulas_and_definitions.md",
    "mlat_methodological_warnings.md",
    "mlat_implementation_patterns.md",
    "mlat_source_traceability.csv",
    "mlat_project_applicability_matrix.md",
    "mlat_feature_hypothesis_catalog.md",
    "mlat_existing_feature_overlap_audit.csv",
    "mlat_feature_batch_v1.md",
    "mlat_feature_research_contract_v1.md",
    "mlat_final_test_governance_note.md",
    "mlat_implementation_plan_v1.md",
    "mlat_stage1_completion_report.md",
)

REQUIRED_TABLE_NAMES = (
    "construction_performance",
    "feature_build_timings",
    "feature_build_runtime",
    "feature_construction_audit",
    "feature_engineering_diagnostics",
    "feature_persistence_validation",
    "feature_summary",
    "validation_checks",
    "validation_feature_diagnostics",
    "validation_coverage",
    "evaluation_checks",
    "evaluation_target_catalog",
    "evaluation_family_coverage",
    "univariate_cells",
    "quintile_edges",
    "quintile_results",
    "year_stability",
    "thinning_results",
    "overlap_audit",
    "candidate_correlation",
    "incremental_summary",
    "feature_verdicts",
    "garch_parameters",
    "garch_fit_diagnostics",
    "garch_residual_diagnostics",
    "garch_calibration",
    "garch_benchmarks",
    "garch_audit_checks",
    "garch_warnings",
    "figure_manifest",
    "artifact_verification",
)

REQUIRED_DATA_NAMES = (
    "feature_registry_mlat_gc",
    "feature_matrix_mlat_gc",
    "feature_observation_audit",
    "daily_ic_evidence",
    "incremental_daily_partial_ic",
)

REPORT_SECTIONS = (
    "Executive conclusion",
    "Book-ingestion completion status",
    "Page coverage",
    "Chapter coverage",
    "Most relevant book concepts",
    "Methods rejected as inapplicable",
    "Existing-feature overlap findings",
    "Frozen MLAT feature batch",
    "Exact formulas and sources",
    "Features implemented",
    "Features rejected before implementation",
    "Leakage and boundary test results",
    "Development results",
    "Validation results",
    "Directional findings",
    "Expansion findings",
    "Risk-state findings",
    "Session stability",
    "Year stability",
    "Incremental information",
    "GARCH audit result",
    "Feature verdicts",
    "Limitations",
    "Final-test governance",
    "Recommended next research stage",
    "Files created and modified",
)

CONTEXT_SECTIONS = (
    "Purpose",
    "Upstream dependencies",
    "Book coverage",
    "Frozen hypothesis batch",
    "Architecture",
    "Implemented features",
    "Evaluation contract",
    "Key results",
    "GARCH decision",
    "Feature verdicts",
    "Artifact paths",
    "Unresolved issues",
    "Next step",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_files(paths: Iterable[Path], *, label: str) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} are missing: {missing}")


def _read_csv(
    path: Path,
    *,
    required_columns: Sequence[str] = (),
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence table is absent: {path}")
    frame = pd.read_csv(
        path,
        low_memory=False,
        na_values=["__MLAT_CSV_NULL_V1__"],
    )
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if frame.empty and not allow_empty:
        raise ValueError(f"Required evidence table is empty: {path}")
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    unknown = normalized.notna() & ~normalized.isin(
        {"true", "false", "1", "0", "yes", "no"}
    )
    if unknown.any():
        values = sorted(normalized.loc[unknown].dropna().unique().tolist())
        raise ValueError(f"Cannot interpret boolean values: {values}")
    return normalized.isin({"true", "1", "yes"})


def _format_value(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        if not pd.notna(value):
            return ""
        if value == 0:
            return "0"
        absolute = abs(value)
        if absolute >= 10_000 or absolute < 0.0001:
            return f"{value:.4g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return re.sub(r"\s+", " ", text).strip()


def _markdown_table(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    boolean_labels: Mapping[str, tuple[str, str]] | None = None,
    max_rows: int | None = None,
) -> str:
    selected = frame.copy()
    if columns is not None:
        missing = sorted(set(columns).difference(selected.columns))
        if missing:
            raise ValueError(f"Cannot render table; columns are absent: {missing}")
        selected = selected.loc[:, list(columns)]
    if max_rows is not None:
        selected = selected.head(max_rows)
    if boolean_labels:
        missing_boolean_columns = sorted(set(boolean_labels).difference(selected.columns))
        if missing_boolean_columns:
            raise ValueError(
                "Cannot render boolean labels; columns are absent: "
                f"{missing_boolean_columns}"
            )
        for column, (true_label, false_label) in boolean_labels.items():
            selected[column] = selected[column].map(
                lambda value, true_label=true_label, false_label=false_label: (
                    ""
                    if value is None or value is pd.NA or bool(pd.isna(value))
                    else true_label
                    if bool(value)
                    else false_label
                )
            )
    if selected.empty:
        return "_The corresponding executed evidence table contains zero rows._"
    display_names = [labels.get(column, column) if labels else column for column in selected]
    header = "| " + " | ".join(_format_value(name) for name in display_names) + " |"
    divider = "| " + " | ".join("---" for _ in display_names) + " |"
    rows = [
        "| "
        + " | ".join(_format_value(value) for value in row)
        + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _bullet_list(values: Iterable[str]) -> str:
    items = [f"- {value}" for value in values]
    return "\n".join(items) if items else "- None recorded by the executed evidence."


def _parse_first_markdown_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    for index in range(len(lines) - 1):
        header = lines[index].strip()
        divider = lines[index + 1].strip()
        if not (header.startswith("|") and header.endswith("|")):
            continue
        if not re.fullmatch(r"\|?[\s:|-]+\|?", divider):
            continue
        columns = [part.strip() for part in header.strip("|").split("|")]
        records: list[list[str]] = []
        for line in lines[index + 2 :]:
            stripped = line.strip()
            if not (stripped.startswith("|") and stripped.endswith("|")):
                break
            values = [part.strip().replace("\\|", "|") for part in stripped.strip("|").split("|")]
            if len(values) != len(columns):
                raise ValueError(f"Malformed Markdown table row in {path}: {line}")
            records.append(values)
        return pd.DataFrame(records, columns=columns)
    raise ValueError(f"No Markdown table found in {path}")


def _hypothesis_catalog() -> pd.DataFrame:
    table = _parse_first_markdown_table(DOCS / "mlat_feature_hypothesis_catalog.md")
    required = {"ID", "Feature", "Source", "Type", "Target", "Decision"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Hypothesis catalog table is missing columns: {missing}")
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for record in table.to_dict("records"):
        hypothesis_id = record["ID"].strip()
        if hypothesis_id:
            current = {key: str(value).strip() for key, value in record.items()}
            current["Reason"] = ""
            records.append(current)
        elif current is not None and record["Feature"].strip() == "Rejection/defer reason":
            current["Reason"] = record["Decision"].strip()
    result = pd.DataFrame(records)
    if result.empty or result["ID"].duplicated().any():
        raise ValueError("Hypothesis catalog did not yield unique hypothesis records")
    return result


def _formula_registry() -> pd.DataFrame:
    text = (DOCS / "mlat_formulas_and_definitions.md").read_text(encoding="utf-8")
    sections = re.split(r"(?=^## MLAT-F\d+\s+-\s+)", text, flags=re.MULTILINE)
    rows: list[dict[str, str]] = []
    for section in sections:
        heading = re.match(r"^## (MLAT-F\d+)\s+-\s+(.+)$", section, flags=re.MULTILINE)
        if heading is None:
            continue
        fields = {
            key.strip(): value.strip()
            for key, value in re.findall(r"^- ([^:]+):\s*(.+)$", section, flags=re.MULTILINE)
        }
        rows.append(
            {
                "formula_id": heading.group(1),
                "name": heading.group(2).strip(),
                "definition": fields.get("Definition", ""),
                "required_history": fields.get("Required history", ""),
                "causal_availability": fields.get("Causal availability", ""),
                "boundary_requirements": fields.get("Boundary requirements", ""),
                "book_trace": fields.get("Book trace", ""),
                "status": fields.get("Status", ""),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 14 or not result["definition"].astype(bool).all():
        raise ValueError("Formula registry must contain 14 fully defined entries")
    return result


def _verify_book_coverage(coverage: pd.DataFrame) -> dict[str, int]:
    starts = pd.to_numeric(coverage["pdf_page_start"], errors="raise").astype(int)
    ends = pd.to_numeric(coverage["pdf_page_end"], errors="raise").astype(int)
    counts = pd.to_numeric(coverage["page_count"], errors="raise").astype(int)
    if not ((ends - starts + 1) == counts).all():
        raise ValueError("Book-ingestion manifest contains inconsistent page counts")
    pages: list[int] = []
    for start, end in zip(starts, ends, strict=True):
        pages.extend(range(start, end + 1))
    if sorted(pages) != list(range(1, 859)):
        raise ValueError("Book-ingestion manifest must account for physical PDF pages 1-858 exactly")
    if not coverage["status"].astype(str).eq("COMPLETE").all():
        raise ValueError("Book-ingestion manifest contains incomplete rows")
    if not coverage["text_extraction_status"].astype(str).eq("COMPLETE").all():
        raise ValueError("Book-ingestion manifest contains incomplete text extraction")
    substantive = coverage.loc[
        coverage["section_type"].isin(["preface", "chapter", "appendix"])
    ]
    chapters = coverage.loc[coverage["section_type"].eq("chapter")]
    appendix = coverage.loc[coverage["section_type"].eq("appendix")]
    if len(chapters) != 23 or len(appendix) != 1:
        raise ValueError("Expected 23 chapter rows and one appendix row")
    chapter_paths = [
        DOCS / Path(str(path).replace("\\", "/"))
        for path in pd.concat([chapters, appendix])["summary_path"].dropna().astype(str)
    ]
    if len(chapter_paths) != 24 or len(set(chapter_paths)) != 24:
        raise ValueError("Each of the 23 chapters and the Appendix needs a unique summary path")
    _require_files(chapter_paths, label="Chapter/Appendix summaries")
    visually_inspected = set()
    for value in coverage["visual_inspection_pages"].dropna().astype(str):
        visually_inspected.update(int(token) for token in re.findall(r"\d+", value))
    return {
        "physical_pages": len(pages),
        "substantive_pages": int(substantive["page_count"].sum()),
        "chapters": len(chapters),
        "appendices": len(appendix),
        "coverage_rows": len(coverage),
        "visually_inspected_pages": len(visually_inspected),
    }


def _verify_notebook() -> dict[str, int]:
    payload = _read_json(NOTEBOOK)
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("MLAT notebook does not contain cells")
    title = "".join(cells[0].get("source", []))
    if not title.startswith("# MLAT Feature Research"):
        raise ValueError("MLAT notebook does not start with the required title")
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    unexecuted = [
        index
        for index, cell in enumerate(code_cells, start=1)
        if cell.get("execution_count") is None
    ]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if unexecuted or errors:
        raise ValueError(
            "Notebook execution is incomplete: "
            f"unexecuted_code_cells={unexecuted}, error_outputs={len(errors)}"
        )
    return {
        "cells": len(cells),
        "code_cells": len(code_cells),
        "executed_code_cells": len(code_cells),
        "error_outputs": 0,
    }


def _resolve_manifest_artifact(path_text: str) -> Path:
    candidate = (ARTIFACT_MANIFEST.parent / path_text).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Artifact manifest path escapes the repository: {path_text}") from error
    return candidate


def _verify_artifact_manifest() -> tuple[dict[str, Any], list[Path]]:
    manifest = _read_json(ARTIFACT_MANIFEST)
    if manifest.get("layout_version") != "mlat-artifacts-v1":
        raise ValueError(
            "Unsupported MLAT artifact-manifest layout: "
            f"{manifest.get('layout_version')!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Artifact manifest has no artifact records")
    verified: list[Path] = []
    for key, entry in artifacts.items():
        if not isinstance(entry, dict):
            raise TypeError(f"Artifact manifest entry {key!r} is not an object")
        path = _resolve_manifest_artifact(str(entry.get("artifact_path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-listed artifact is absent: {path}")
        expected_size = int(entry.get("byte_size", -1))
        if path.stat().st_size != expected_size:
            raise ValueError(f"Artifact size does not match manifest: {path}")
        expected_hash = str(entry.get("sha256", ""))
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Artifact SHA-256 does not match manifest: {path}")
        verified.append(path)
    return manifest, verified


def _verify_figure_manifest(figure_manifest: pd.DataFrame) -> None:
    for row in figure_manifest.itertuples(index=False):
        path = (ROOT / str(row.path)).resolve()
        try:
            path.relative_to(FIGURE_ROOT.resolve())
        except ValueError as error:
            raise ValueError(f"Figure path is outside the v1 figure root: {row.path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Figure-manifest file is absent: {path}")
        if path.stat().st_size != int(row.bytes):
            raise ValueError(f"Figure byte size does not match its manifest row: {path}")
        if _sha256(path) != str(row.sha256):
            raise ValueError(f"Figure SHA-256 does not match its manifest row: {path}")


def _verify_execution_manifest(execution: Mapping[str, Any]) -> None:
    required = {
        "run_seconds",
        "engineering_gate",
        "evaluation_gate",
        "advancement_gate_feasible",
        "advancement_gate_reason",
        "artifact_gate",
        "upstream_gate",
        "development_validation_only",
        "final_outcomes_loaded",
        "mgc_loaded",
        "multivariate_authorized",
        "garch_verdict",
        "batch_sha256",
        "contract_sha256",
        "saved_data",
        "saved_tables",
        "saved_figures",
        "next_stage",
    }
    missing = sorted(required.difference(execution))
    if missing:
        raise ValueError(f"Execution manifest is missing required keys: {missing}")
    gates = {
        "engineering_gate": True,
        "evaluation_gate": True,
        "advancement_gate_feasible": False,
        "artifact_gate": True,
        "upstream_gate": True,
        "development_validation_only": True,
        "final_outcomes_loaded": False,
        "mgc_loaded": False,
    }
    failures = [
        f"{key}={execution[key]!r} (expected {expected!r})"
        for key, expected in gates.items()
        if execution[key] is not expected
    ]
    if failures:
        raise ValueError("Execution governance gates failed: " + "; ".join(failures))
    if not str(execution["advancement_gate_reason"]).strip():
        raise ValueError("Execution manifest must explain the closed advancement gate")
    saved_tables = execution["saved_tables"]
    saved_data = execution["saved_data"]
    saved_figures = execution["saved_figures"]
    if not isinstance(saved_tables, dict) or not isinstance(saved_data, dict):
        raise TypeError("saved_tables and saved_data must be JSON objects")
    if not isinstance(saved_figures, list) or len(saved_figures) < 4:
        raise TypeError("saved_figures must be a JSON list containing at least four figures")
    missing_tables = sorted(set(REQUIRED_TABLE_NAMES).difference(saved_tables))
    if missing_tables:
        raise ValueError(f"Execution manifest omits required saved tables: {missing_tables}")
    missing_data = sorted(
        {"daily_ic_evidence", "incremental_daily_partial_ic"}.difference(saved_data)
    )
    if missing_data:
        raise ValueError(f"Execution manifest omits required evidence data: {missing_data}")
    for relative in [*saved_tables.values(), *saved_data.values(), *saved_figures]:
        path = (ROOT / str(relative)).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as error:
            raise ValueError(f"Execution-manifest path escapes the repository: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Execution-manifest artifact is absent: {path}")


def _load_tables() -> dict[str, pd.DataFrame]:
    tables = {
        name: _read_csv(
            TABLE_ROOT / f"{name}.csv",
            allow_empty=name in {"garch_warnings"},
        )
        for name in REQUIRED_TABLE_NAMES
    }
    column_contracts = {
        "validation_checks": ("check", "passed"),
        "evaluation_checks": ("check", "passed"),
        "univariate_cells": (
            "feature_name",
            "target_family",
            "horizon_minutes",
            "entry_session",
            "research_partition",
            "mean_daily_ic",
            "q_value",
        ),
        "year_stability": (
            "feature_name",
            "target_family",
            "research_partition",
            "year",
            "mean_daily_ic",
            "full_period_sign_agreement",
        ),
        "thinning_results": (
            "feature_name",
            "target_family",
            "research_partition",
            "sign_agreement",
        ),
        "overlap_audit": (
            "candidate_feature",
            "reference_feature",
            "reference_scope",
            "absolute_correlation",
            "is_exact_duplicate",
        ),
        "incremental_summary": (
            "feature_name",
            "target_family",
            "control_set",
            "mean_daily_partial_ic_development",
            "mean_daily_partial_ic_validation",
            "incremental_pass",
        ),
        "feature_verdicts": (
            "feature_name",
            "hypothesis_id",
            "book_chapter",
            "source_pdf_page",
            "development_result",
            "validation_result",
            "session_stability",
            "year_stability",
            "redundancy_result",
            "incremental_information_result",
            "economic_interpretation",
            "authorization_gate_open",
            "authorization_gate_reason",
            "pre_authorization_verdict",
            "verdict",
            "final_decision",
        ),
        "garch_audit_checks": ("check_name", "critical", "passed", "details"),
        "figure_manifest": ("figure_name", "path", "sha256", "bytes"),
        "artifact_verification": ("artifact", "format", "checks", "passed"),
    }
    for name, columns in column_contracts.items():
        missing = sorted(set(columns).difference(tables[name].columns))
        if missing:
            raise ValueError(f"{name}.csv is missing required columns: {missing}")
    for name in ("validation_checks", "evaluation_checks", "artifact_verification"):
        if not _bool_series(tables[name]["passed"]).all():
            failures = tables[name].loc[~_bool_series(tables[name]["passed"])]
            raise ValueError(
                f"{name}.csv contains failed completion checks: "
                f"{failures.to_dict('records')}"
            )
    return tables


def _load_registry() -> pd.DataFrame:
    path = DATA_ROOT / "feature_registry_mlat_gc.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Saved MLAT registry is absent: {path}")
    registry = pd.read_parquet(path)
    required = {
        "feature_name",
        "hypothesis_id",
        "source_chapter",
        "source_pdf_page",
        "source_type",
        "formula_or_definition",
        "target_family",
        "minimum_history",
        "availability_timestamp",
        "reset_boundary",
        "status",
    }
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise ValueError(f"Saved MLAT registry is missing required columns: {missing}")
    if len(registry) != 12 or registry["feature_name"].duplicated().any():
        raise ValueError("Saved MLAT registry must contain 12 unique frozen features")
    return registry


def _top_cells(cells: pd.DataFrame, partition: str, *, max_rows: int = 12) -> pd.DataFrame:
    selected = cells.loc[cells["research_partition"].astype(str).eq(partition)].copy()
    selected["absolute_mean_daily_ic"] = pd.to_numeric(
        selected["mean_daily_ic"], errors="coerce"
    ).abs()
    return selected.sort_values(
        "absolute_mean_daily_ic", ascending=False, na_position="last"
    ).head(max_rows)


def _family_cells(
    cells: pd.DataFrame,
    families: Sequence[str],
    *,
    max_rows: int = 10,
) -> pd.DataFrame:
    selected = cells.loc[cells["target_family"].astype(str).isin(families)].copy()
    selected["absolute_mean_daily_ic"] = pd.to_numeric(
        selected["mean_daily_ic"], errors="coerce"
    ).abs()
    return selected.sort_values(
        ["research_partition", "absolute_mean_daily_ic"],
        ascending=[True, False],
        na_position="last",
    ).groupby("research_partition", observed=True, sort=False).head(max_rows // 2)


def _verdict_summary(verdicts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        verdicts.groupby("final_decision", observed=True, dropna=False)
        .size()
        .rename("feature_count")
        .reset_index()
        .sort_values(["feature_count", "final_decision"], ascending=[False, True])
    )
    return summary


def _numeric_column_sum(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _inventory() -> list[tuple[str, list[Path]]]:
    documentation = {
        path for path in DOCS.rglob("*") if path.is_file()
    } | {FINAL_REPORT, CONTEXT_REPORT}
    groups = [
        ("Documentation", sorted(documentation)),
        ("Notebook", [NOTEBOOK] if NOTEBOOK.is_file() else []),
        (
            "Reusable source",
            sorted((ROOT / "src" / "statistical_research").glob("mlat_*.py")),
        ),
        ("Tests", sorted((ROOT / "tests").glob("test_mlat_*.py"))),
        ("Build scripts", sorted((ROOT / "scripts").glob("build_mlat_*.py"))),
        ("Processed data", sorted(DATA_ROOT.glob("*")) if DATA_ROOT.is_dir() else []),
        ("Report artifacts", sorted(REPORT_ROOT.rglob("*")) if REPORT_ROOT.is_dir() else []),
        ("Figures", sorted(FIGURE_ROOT.glob("*")) if FIGURE_ROOT.is_dir() else []),
    ]
    result = []
    for name, paths in groups:
        files = [
            path
            for path in dict.fromkeys(paths)
            if path.is_file() or path in {FINAL_REPORT, CONTEXT_REPORT}
        ]
        if files:
            result.append((name, files))
    return result


def _inventory_markdown(groups: Sequence[tuple[str, Sequence[Path]]]) -> str:
    sections = []
    for label, paths in groups:
        entries = [
            f"- `{path.relative_to(ROOT).as_posix()}`"
            for path in paths
        ]
        sections.append(f"### {label}\n\n" + "\n".join(entries))
    return "\n\n".join(sections)


def _report_inputs() -> dict[str, Any]:
    _require_files((DOCS / name for name in STAGE1_FILES), label="Stage-1 artifacts")
    _require_files(
        (
            NOTEBOOK,
            EXECUTION_MANIFEST,
            ARTIFACT_MANIFEST,
            OLD_NOTEBOOK,
            OLD_CONTEXT_REPORT,
        ),
        label="Completion inputs",
    )
    _require_files(
        (DATA_ROOT / f"{name}.parquet" for name in REQUIRED_DATA_NAMES),
        label="Versioned MLAT data artifacts",
    )
    old_hashes = {
        "statistical_feature_research.ipynb": _sha256(OLD_NOTEBOOK),
        "statistical_feature_research_context_report.md": _sha256(OLD_CONTEXT_REPORT),
    }

    coverage = _read_csv(
        DOCS / "mlat_book_ingestion_manifest.csv",
        required_columns=(
            "section_type",
            "title",
            "pdf_page_start",
            "pdf_page_end",
            "page_count",
            "text_extraction_status",
            "visual_inspection_pages",
            "summary_path",
            "status",
        ),
    )
    concepts = _read_csv(
        DOCS / "mlat_concept_registry.csv",
        required_columns=(
            "concept_id",
            "concept_name",
            "chapter",
            "pdf_page",
            "category",
            "definition",
            "applicability",
            "recommendation",
        ),
    )
    upstream = _read_csv(
        DOCS / "mlat_upstream_artifact_manifest.csv",
        required_columns=(
            "artifact_name",
            "exact_path",
            "row_count",
            "column_count",
            "research_partition_coverage",
            "file_sha256",
            "validation_status",
            "mlat_use",
        ),
    )
    source_traceability = _read_csv(
        DOCS / "mlat_source_traceability.csv",
        required_columns=(
            "hypothesis_id",
            "feature_or_method",
            "source_chapter",
            "source_pdf_page",
            "status",
        ),
    )
    execution = _read_json(EXECUTION_MANIFEST)
    _verify_execution_manifest(execution)
    artifact_manifest, verified_artifacts = _verify_artifact_manifest()
    notebook_state = _verify_notebook()
    tables = _load_tables()
    _verify_figure_manifest(tables["figure_manifest"])
    required_manifested = {
        *(DATA_ROOT / f"{name}.parquet" for name in REQUIRED_DATA_NAMES),
        *(TABLE_ROOT / f"{name}.csv" for name in REQUIRED_TABLE_NAMES),
        EXECUTION_MANIFEST,
    }
    missing_manifested = sorted(
        path.relative_to(ROOT).as_posix()
        for path in required_manifested.difference(set(verified_artifacts))
    )
    if missing_manifested:
        raise ValueError(
            "Required outputs are absent from the aggregate artifact manifest: "
            f"{missing_manifested}"
        )
    registry = _load_registry()
    hypotheses = _hypothesis_catalog()
    formulas = _formula_registry()
    coverage_state = _verify_book_coverage(coverage)

    frozen_hypotheses = hypotheses.loc[hypotheses["Decision"].eq("SELECT_V1")]
    if len(frozen_hypotheses) != 12:
        raise ValueError("Frozen hypothesis catalog must contain exactly 12 SELECT_V1 rows")
    if set(frozen_hypotheses["Feature"]) != set(registry["feature_name"].astype(str)):
        raise ValueError("Saved registry membership differs from the frozen Stage-1 batch")
    verdicts = tables["feature_verdicts"]
    if set(verdicts["feature_name"].astype(str)) != set(registry["feature_name"].astype(str)):
        raise ValueError("Feature verdict membership differs from the saved registry")
    if not verdicts["final_decision"].astype(str).eq(verdicts["verdict"].astype(str)).all():
        raise ValueError("Feature verdict and final_decision columns disagree")
    authorization_open = _bool_series(verdicts["authorization_gate_open"])
    if authorization_open.any():
        raise ValueError("Feature verdicts unexpectedly open the frozen v1 advancement gate")
    if verdicts["final_decision"].astype(str).str.startswith("ADVANCE_").any():
        raise ValueError("Closed advancement gate cannot contain an advancing verdict")
    authorization_reasons = (
        verdicts["authorization_gate_reason"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if authorization_reasons != [str(execution["advancement_gate_reason"])]:
        raise ValueError(
            "Feature-verdict authorization reason differs from the execution manifest"
        )
    if execution["batch_sha256"] != (
        "8c8b74267635566f07011c2789ad4d323647f19eba1f1f3426a154c1a850acc0"
    ):
        raise ValueError("Execution used an unrecognized frozen feature-batch hash")
    if execution["contract_sha256"] != (
        "31ea8ff8255285d580dc9c2cb71bcbff85d05011e771b49f2f09d24cad810320"
    ):
        raise ValueError("Execution used an unrecognized evaluation-contract hash")

    return {
        "coverage": coverage,
        "coverage_state": coverage_state,
        "concepts": concepts,
        "upstream": upstream,
        "source_traceability": source_traceability,
        "execution": execution,
        "artifact_manifest": artifact_manifest,
        "verified_artifacts": verified_artifacts,
        "notebook_state": notebook_state,
        "tables": tables,
        "registry": registry,
        "hypotheses": hypotheses,
        "formulas": formulas,
        "old_hashes": old_hashes,
    }


def _build_final_report(inputs: Mapping[str, Any]) -> str:
    coverage: pd.DataFrame = inputs["coverage"]
    coverage_state: Mapping[str, int] = inputs["coverage_state"]
    concepts: pd.DataFrame = inputs["concepts"]
    execution: Mapping[str, Any] = inputs["execution"]
    tables: Mapping[str, pd.DataFrame] = inputs["tables"]
    registry: pd.DataFrame = inputs["registry"]
    hypotheses: pd.DataFrame = inputs["hypotheses"]
    formulas: pd.DataFrame = inputs["formulas"]
    notebook_state: Mapping[str, int] = inputs["notebook_state"]

    verdicts = tables["feature_verdicts"].copy()
    cells = tables["univariate_cells"].copy()
    incremental = tables["incremental_summary"].copy()
    overlap = tables["overlap_audit"].copy()
    garch_checks = tables["garch_audit_checks"].copy()
    verdict_counts = _verdict_summary(verdicts)
    advanced = verdicts["final_decision"].astype(str).str.startswith("ADVANCE_")
    critical_garch = _bool_series(garch_checks["critical"])
    passed_garch = _bool_series(garch_checks["passed"])
    failed_critical_garch = garch_checks.loc[critical_garch & ~passed_garch]

    complete_chapters = coverage.loc[
        coverage["section_type"].isin(["chapter", "appendix"]),
        ["section_type", "title", "pdf_page_start", "pdf_page_end", "status"],
    ].copy()
    complete_chapters["pdf_pages"] = (
        complete_chapters["pdf_page_start"].astype(str)
        + "-"
        + complete_chapters["pdf_page_end"].astype(str)
    )

    relevant_concepts = concepts.loc[
        concepts["recommendation"].astype(str).isin(["ADOPT", "IMPLEMENT", "AUDIT"])
        | concepts["applicability"].astype(str).isin(["DIRECT", "ADAPTED"])
    ].head(12)

    applicability = _parse_first_markdown_table(
        DOCS / "mlat_project_applicability_matrix.md"
    )
    rejected_applicability = applicability.loc[
        applicability["Classification"].str.contains(
            "REJECTED|REQUIRES DIFFERENT DATA|IMPRACTICAL|REDUNDANT",
            regex=True,
            na=False,
        )
    ]
    rejected_hypotheses = hypotheses.loc[~hypotheses["Decision"].eq("SELECT_V1")].copy()

    existing_overlap = overlap.loc[overlap["reference_scope"].astype(str).eq("existing")].copy()
    existing_overlap["absolute_correlation"] = pd.to_numeric(
        existing_overlap["absolute_correlation"], errors="coerce"
    )
    existing_overlap["is_exact_duplicate"] = _bool_series(
        existing_overlap["is_exact_duplicate"]
    )
    nearest_overlap = (
        existing_overlap.sort_values(
            "absolute_correlation", ascending=False, na_position="last"
        )
        .groupby("candidate_feature", observed=True, sort=False)
        .head(1)
        .sort_values("absolute_correlation", ascending=False, na_position="last")
    )
    vetoes = existing_overlap.loc[
        existing_overlap["is_exact_duplicate"]
        | existing_overlap["absolute_correlation"].ge(0.995)
    ]

    frozen_registry = registry[
        [
            "hypothesis_id",
            "feature_name",
            "source_type",
            "target_family",
            "minimum_history",
            "availability_timestamp",
            "reset_boundary",
        ]
    ].copy()
    exact_formulas = registry[
        [
            "hypothesis_id",
            "feature_name",
            "formula_or_definition",
            "source_chapter",
            "source_pdf_page",
        ]
    ].copy()
    formula_controls = formulas.loc[formulas["status"].eq("FROZEN_V1")]

    validation_checks = tables["validation_checks"].copy()
    evaluation_checks = tables["evaluation_checks"].copy()
    artifact_checks = tables["artifact_verification"].copy()
    boundary_evidence = pd.concat(
        [
            validation_checks.assign(evidence_table="validation_checks"),
            evaluation_checks.assign(evidence_table="evaluation_checks"),
        ],
        ignore_index=True,
        sort=False,
    )
    display_check_columns = [
        column
        for column in ("evidence_table", "check", "passed", "details")
        if column in boundary_evidence.columns
    ]

    top_development = _top_cells(cells, "Development")
    top_validation = _top_cells(cells, "Validation")
    cell_columns = [
        "feature_name",
        "target_family",
        "horizon_minutes",
        "entry_session",
        "research_partition",
        "eligible_dates",
        "finite_observations",
        "mean_daily_ic",
        "q_value",
        "absolute_monotonicity",
        "top_bottom_tick_spread",
        "thinned_mean_daily_ic",
    ]
    cell_columns = [column for column in cell_columns if column in cells]

    directional = _family_cells(cells, ["direction"])
    expansion = _family_cells(cells, ["expansion"])
    risk_state = _family_cells(cells, ["volatility", "path_risk"])

    session_summary = (
        verdicts.groupby(["session_stability", "final_decision"], observed=True)
        .size()
        .rename("feature_count")
        .reset_index()
    )
    year_verdict_summary = (
        verdicts.groupby(["year_stability", "final_decision"], observed=True)
        .size()
        .rename("feature_count")
        .reset_index()
    )
    year_rows = tables["year_stability"].copy()
    year_rows["absolute_mean_daily_ic"] = pd.to_numeric(
        year_rows["mean_daily_ic"], errors="coerce"
    ).abs()
    year_rows = year_rows.sort_values(
        "absolute_mean_daily_ic", ascending=False, na_position="last"
    ).head(16)

    incremental["incremental_pass"] = _bool_series(incremental["incremental_pass"])
    incremental_summary = (
        incremental.groupby("control_set", observed=True)
        .agg(
            tests=("feature_name", "size"),
            passing_tests=("incremental_pass", "sum"),
            features=("feature_name", "nunique"),
        )
        .reset_index()
    )
    top_incremental = incremental.assign(
        absolute_development_partial_ic=pd.to_numeric(
            incremental["mean_daily_partial_ic_development"], errors="coerce"
        ).abs()
    ).sort_values(
        "absolute_development_partial_ic", ascending=False, na_position="last"
    ).head(16)

    garch_benchmarks = tables["garch_benchmarks"]
    garch_parameters = tables["garch_parameters"]
    garch_fit = tables["garch_fit_diagnostics"]
    garch_calibration = tables["garch_calibration"]
    garch_residuals = tables["garch_residual_diagnostics"]
    garch_warnings = tables["garch_warnings"]

    inventory = _inventory()
    artifact_count = len(inputs["verified_artifacts"])
    warning_count = len(garch_warnings)
    authorization_reason = str(execution["advancement_gate_reason"])
    executive = (
        "The executed v1 notebook passed its engineering, evidence-integrity, "
        "Development/Validation-only, artifact-persistence, and governance checks. "
        "However, the frozen advancement design is structurally non-evaluable, so v1 "
        f"fails closed: {int(advanced.sum())} of {len(registry)} features were "
        "authorized to advance. This is not a clean empirical rejection of the raw "
        f"relationships. The authorization blocker is: {authorization_reason} "
        f"The multivariate authorization gate was "
        f"{'OPEN' if execution['multivariate_authorized'] else 'CLOSED'}. "
        f"The separately governed GARCH audit verdict was "
        f"`{execution['garch_verdict']}`. These are feature-research findings, not a "
        "signal, PnL, or Sharpe claim."
    )

    sections: dict[str, str] = {
        "Executive conclusion": executive
        + "\n\n"
        + _markdown_table(verdict_counts),
        "Book-ingestion completion status": (
            "Stage 1 is `COMPLETE`. The persisted completion report, ingestion manifest, "
            "24 chapter/Appendix summaries, concept registry, formula registry, hypothesis "
            "catalog, frozen batch, and frozen contract all passed the pre-outcome build "
            "controls.\n\n"
            f"- Frozen batch SHA-256: `{execution['batch_sha256']}`\n"
            f"- Frozen contract SHA-256: `{execution['contract_sha256']}`\n"
            f"- Concepts recorded: {len(concepts)}\n"
            f"- Formula/definition entries: {len(formulas)}\n"
            f"- Hypotheses recorded: {len(hypotheses)}"
        ),
        "Page coverage": (
            f"The coverage manifest accounts exactly once for all "
            f"{coverage_state['physical_pages']} physical PDF pages. Substantive "
            f"preface/chapter/Appendix coverage comprises "
            f"{coverage_state['substantive_pages']} pages, and "
            f"{coverage_state['visually_inspected_pages']} distinct pages containing "
            "important equations, tables, figures, or diagrams were separately rendered "
            "and inspected.\n\n"
            + _markdown_table(
                coverage[
                    [
                        "section_type",
                        "pdf_page_start",
                        "pdf_page_end",
                        "page_count",
                        "status",
                    ]
                ].groupby("section_type", observed=True, as_index=False)
                .agg(
                    pdf_page_start=("pdf_page_start", "min"),
                    pdf_page_end=("pdf_page_end", "max"),
                    page_count=("page_count", "sum"),
                    records=("status", "size"),
                    complete_records=("status", lambda values: int(values.eq("COMPLETE").sum())),
                )
            )
        ),
        "Chapter coverage": (
            f"All {coverage_state['chapters']} chapters and the Appendix have complete "
            "durable summaries with page maps and source citations.\n\n"
            + _markdown_table(
                complete_chapters,
                columns=("section_type", "title", "pdf_pages", "status"),
            )
        ),
        "Most relevant book concepts": _markdown_table(
            relevant_concepts,
            columns=(
                "concept_id",
                "concept_name",
                "chapter",
                "pdf_page",
                "category",
                "applicability",
                "recommendation",
            ),
        ),
        "Methods rejected as inapplicable": (
            "The applicability screen rejected, deferred, or moved methods to a "
            "different-data phase when they violated causality/governance, duplicated "
            "existing features, or required unavailable cross-sectional, text, quote, "
            "trade, order-book, macro, or image data.\n\n"
            + _markdown_table(
                rejected_applicability,
                columns=("Idea", "Source (chapter:PDF)", "Classification", "Reason"),
            )
        ),
        "Existing-feature overlap findings": (
            f"The empirical Development overlap audit evaluated "
            f"{existing_overlap['candidate_feature'].nunique()} candidates against "
            f"{existing_overlap['reference_feature'].nunique()} existing references. "
            f"It found {len(vetoes)} exact or absolute-correlation-at-least-0.995 veto "
            "relationships. Final redundancy decisions are carried in the verdict table.\n\n"
            + _markdown_table(
                nearest_overlap,
                columns=(
                    "candidate_feature",
                    "reference_feature",
                    "absolute_correlation",
                    "is_exact_duplicate",
                ),
                boolean_labels={"is_exact_duplicate": ("YES", "NO")},
            )
        ),
        "Frozen MLAT feature batch": (
            "Membership and parameters were frozen before any MLAT feature/outcome "
            "relationship was calculated.\n\n"
            + _markdown_table(frozen_registry)
        ),
        "Exact formulas and sources": (
            _markdown_table(exact_formulas)
            + "\n\nThe formula registry independently records causal history and "
            "boundary controls:\n\n"
            + _markdown_table(
                formula_controls,
                columns=(
                    "formula_id",
                    "name",
                    "required_history",
                    "causal_availability",
                    "boundary_requirements",
                    "book_trace",
                ),
            )
        ),
        "Features implemented": (
            f"The saved, registry-controlled feature matrix implements exactly "
            f"{len(registry)} features. Every feature is available at the completed "
            "decision-bar close and is mapped to stable upstream observation IDs after "
            "chronological GC-only construction.\n\n"
            + _markdown_table(
                tables["feature_summary"],
                columns=[
                    column
                    for column in (
                        "feature_name",
                        "count",
                        "mean",
                        "std",
                        "min",
                        "max",
                        "missing_fraction",
                    )
                    if column in tables["feature_summary"].columns
                ],
            )
        ),
        "Features rejected before implementation": _markdown_table(
            rejected_hypotheses,
            columns=("ID", "Feature", "Source", "Type", "Target", "Decision", "Reason"),
        ),
        "Leakage and boundary test results": (
            f"All {len(validation_checks)} feature-validation checks, "
            f"{len(evaluation_checks)} evaluation-scope checks, and "
            f"{len(artifact_checks)} saved-artifact verification rows passed. "
            "The executed manifest additionally proves that Final-test outcomes and MGC "
            "were not loaded. These integrity checks do not open the separate advancement "
            f"gate, which fails closed because {authorization_reason} Synthetic and "
            "invariance coverage is recorded in the MLAT test suite; this report does not "
            "infer test outcomes from source presence.\n\n"
            + _markdown_table(boundary_evidence, columns=display_check_columns)
        ),
        "Development results": (
            "The table below reports the largest absolute Development daily Spearman IC "
            "cells from the complete preregistered screen. Ranking here is descriptive; "
            "advancement still requires every registered Validation, stability, economic, "
            "redundancy, and incremental gate.\n\n"
            + _markdown_table(top_development, columns=cell_columns)
        ),
        "Validation results": (
            "Validation used frozen Development transformations, quantile edges, and "
            "thresholds. The largest absolute Validation cells are shown without using "
            "the previously exposed Final-test outcomes.\n\n"
            + _markdown_table(top_validation, columns=cell_columns)
        ),
        "Directional findings": (
            f"Directional advancement cells across final feature verdicts: "
            f"{_numeric_column_sum(verdicts, 'directional_cells_passed')}. "
            "The tick-spread field is retained because a statistically detectable rank "
            "relationship was not allowed to substitute for the preregistered two-GC-tick "
            "economic screen. The closed structural advancement gate also prevents any "
            "v1 authorization.\n\n"
            + _markdown_table(directional, columns=cell_columns)
        ),
        "Expansion findings": (
            f"Expansion advancement cells across final feature verdicts: "
            f"{_numeric_column_sum(verdicts, 'expansion_cells_passed')}. "
            "Expansion evidence is interpreted separately from signed direction and "
            "remains descriptive because the v1 advancement gate is structurally "
            "non-evaluable.\n\n"
            + _markdown_table(expansion, columns=cell_columns)
        ),
        "Risk-state findings": (
            f"Volatility/path-risk advancement cells across final feature verdicts: "
            f"{_numeric_column_sum(verdicts, 'risk_state_cells_passed')}. "
            "A risk-state feature need not predict signed returns, and no risk-state "
            "finding is presented as a trading policy. The raw relationships remain "
            "research-only under the closed v1 advancement gate.\n\n"
            + _markdown_table(risk_state, columns=cell_columns)
        ),
        "Session stability": (
            "London and New York were evaluated separately. The verdict-level session "
            "gate requires eligible evidence and Development-sign consistency across "
            "both sessions for the relevant family/horizon.\n\n"
            + _markdown_table(session_summary)
        ),
        "Year stability": (
            "Annual sign agreement is evaluated against each corresponding full-period "
            "cell. Verdict-level results and the largest annual absolute-IC rows follow.\n\n"
            + _markdown_table(year_verdict_summary)
            + "\n\n"
            + _markdown_table(
                year_rows,
                columns=(
                    "feature_name",
                    "target_family",
                    "horizon_minutes",
                    "entry_session",
                    "research_partition",
                    "year",
                    "eligible_dates",
                    "mean_daily_ic",
                    "full_period_sign_agreement",
                ),
            )
        ),
        "Incremental information": (
            "Partial rank IC was evaluated at 60 and 180 minutes beyond both the ATR(20) "
            "anchor and the frozen existing-feature control set. Development magnitude, "
            "Validation sign/retention, and the registered thresholds jointly determine "
            "the incremental pass flag. The frozen set already contains ATR(20), so only "
            "the two distinct controls `atr_20` and `frozen_15` are reported. Incremental "
            "passes are descriptive: at the same 60/180-minute horizons, the frozen "
            "daily-thinning rule has no eligible dates under its at-least-10-observations "
            "requirement, so it cannot authorize advancement.\n\n"
            + _markdown_table(incremental_summary)
            + "\n\n"
            + _markdown_table(
                top_incremental,
                columns=(
                    "feature_name",
                    "target_family",
                    "horizon_minutes",
                    "entry_session",
                    "control_set",
                    "mean_daily_partial_ic_development",
                    "mean_daily_partial_ic_validation",
                    "validation_magnitude_retention",
                    "incremental_pass",
                ),
            )
        ),
        "GARCH audit result": (
            f"The governed audit verdict is `{execution['garch_verdict']}`. "
            f"{len(failed_critical_garch)} critical audit checks failed, and "
            f"{warning_count} captured warnings were persisted. GARCH remains audit-only "
            "and outside the frozen feature matrix irrespective of diagnostic success.\n\n"
            "### Audit checks\n\n"
            + _markdown_table(
                garch_checks,
                boolean_labels={
                    "critical": ("YES", "NO"),
                    "advancement_authorized": ("YES", "NO"),
                },
            )
            + "\n\n### Parameters\n\n"
            + _markdown_table(garch_parameters)
            + "\n\n### Fit diagnostics\n\n"
            + _markdown_table(
                garch_fit,
                boolean_labels={"critical": ("YES", "NO")},
            )
            + "\n\n### Residual diagnostics\n\n"
            + _markdown_table(garch_residuals)
            + "\n\n### Development calibration\n\n"
            + _markdown_table(garch_calibration)
            + "\n\n### Simple-anchor comparison\n\n"
            + _markdown_table(garch_benchmarks)
            + "\n\n### Captured warnings\n\n"
            + _markdown_table(garch_warnings)
        ),
        "Feature verdicts": (
            "Every final decision below fails closed as `RESEARCH_ONLY` because the "
            "frozen v1 thinning gate is structurally non-evaluable. The preserved "
            "pre-authorization label summarizes other descriptive gates only and must "
            "not be read as an advancement decision.\n\n"
            + _markdown_table(
                verdicts,
                columns=(
                    "hypothesis_id",
                    "feature_name",
                    "book_chapter",
                    "source_pdf_page",
                    "development_result",
                    "validation_result",
                    "session_stability",
                    "year_stability",
                    "redundancy_result",
                    "incremental_information_result",
                    "economic_interpretation",
                    "pre_authorization_verdict",
                    "final_decision",
                    "verdict_reason",
                ),
                labels={
                    "pre_authorization_verdict": (
                        "descriptive_pre_authorization_verdict"
                    )
                },
            )
        ),
        "Limitations": (
            "- Minute observations and overlapping forward paths are dependent; dates, "
            "sessions, block uncertainty, and horizon thinning mitigate but do not erase "
            "that dependence.\n"
            f"- Frozen v1 advancement is structurally non-evaluable: "
            f"{authorization_reason}\n"
            "- Outcomes are now viewed, so changing the v1 thinning statistic would be "
            "post hoc; a feasible replacement belongs in a preregistered v2 contract.\n"
            "- The historical Final-test interval was exposed in the prior research cycle "
            "and is not a pristine holdout.\n"
            "- MGC transfer validation was locked and not inspected.\n"
            "- The experiment uses one-minute OHLCV and cannot answer quote-, trade-, "
            "order-book-, macro-, text-, or cross-sectional hypotheses.\n"
            "- No sequential entry/exit policy, transaction-cost model, PnL, or Sharpe "
            "analysis was authorized.\n"
            f"- Notebook execution recorded {execution['run_seconds']:.3f} seconds and "
            f"{int(execution.get('rss_bytes_final', 0)):,} final-process RSS bytes; these "
            "are machine/run observations, not universal performance benchmarks."
        ),
        "Final-test governance": (
            "The execution manifest records "
            f"`development_validation_only={execution['development_validation_only']}`, "
            f"`final_outcomes_loaded={execution['final_outcomes_loaded']}`, and "
            f"`mgc_loaded={execution['mgc_loaded']}`. The historical Final-test period "
            "was therefore excluded from feature/outcome evaluation, selection, parameter "
            "tuning, and displays. A new future holdout or live paper period is required "
            "for genuine final confirmation."
        ),
        "Recommended next research stage": (
            f"{execution['next_stage']}\n\nThe executed multivariate authorization flag is "
            f"`{execution['multivariate_authorized']}`. No nonlinear-model or backtest "
            "stage is implied when that gate is closed."
        ),
        "Files created and modified": (
            f"The inventory below is generated from the known MLAT documentation, "
            f"notebook, source, test, data, report, and figure paths. "
            f"{artifact_count} manifest-listed artifacts were hash-verified before this "
            "report was built. The historical statistical notebook and context report "
            "were required as present, read only, and are not write targets of this "
            "builder. Their hashes at report time were "
            f"`{inputs['old_hashes']['statistical_feature_research.ipynb']}` and "
            f"`{inputs['old_hashes']['statistical_feature_research_context_report.md']}`, "
            "respectively.\n\n"
            + _inventory_markdown(inventory)
        ),
    }

    missing_sections = [name for name in REPORT_SECTIONS if name not in sections]
    if missing_sections:
        raise AssertionError(f"Final report sections were not built: {missing_sections}")
    body = [
        "# MLAT Feature Research — Final Research Report",
        "",
        (
            f"Generated strictly from the frozen Stage-1 record and the executed v1 "
            f"evidence artifacts. Notebook evidence: {notebook_state['executed_code_cells']} "
            f"executed code cells, {notebook_state['error_outputs']} error outputs."
        ),
    ]
    for index, name in enumerate(REPORT_SECTIONS, start=1):
        body.extend(["", f"## {index}. {name}", "", sections[name]])
    report = "\n".join(body).rstrip() + "\n"
    for index, name in enumerate(REPORT_SECTIONS, start=1):
        if f"## {index}. {name}" not in report:
            raise AssertionError(f"Required final-report section is absent: {name}")
    return report


def _build_context_report(inputs: Mapping[str, Any]) -> str:
    coverage_state: Mapping[str, int] = inputs["coverage_state"]
    upstream: pd.DataFrame = inputs["upstream"]
    execution: Mapping[str, Any] = inputs["execution"]
    tables: Mapping[str, pd.DataFrame] = inputs["tables"]
    registry: pd.DataFrame = inputs["registry"]
    hypotheses: pd.DataFrame = inputs["hypotheses"]
    notebook_state: Mapping[str, int] = inputs["notebook_state"]
    verdicts = tables["feature_verdicts"]
    verdict_counts = _verdict_summary(verdicts)
    authorization_reason = str(execution["advancement_gate_reason"])
    advanced = verdicts.loc[
        verdicts["final_decision"].astype(str).str.startswith("ADVANCE_"),
        ["feature_name", "final_decision"],
    ]
    garch_checks = tables["garch_audit_checks"]
    critical = _bool_series(garch_checks["critical"])
    passed = _bool_series(garch_checks["passed"])
    failed_garch = garch_checks.loc[critical & ~passed, ["check_name", "details"]]
    rejected = hypotheses.loc[~hypotheses["Decision"].eq("SELECT_V1")]

    source_modules = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src" / "statistical_research").glob("mlat_*.py")
    )
    artifact_paths = [
        DATA_ROOT.relative_to(ROOT).as_posix(),
        REPORT_ROOT.relative_to(ROOT).as_posix(),
        FIGURE_ROOT.relative_to(ROOT).as_posix(),
        NOTEBOOK.relative_to(ROOT).as_posix(),
    ]
    issues = [
        (
            "The frozen v1 advancement gate is structurally non-evaluable and "
            f"remains fail-closed: {authorization_reason}"
        ),
        (
            "Outcomes have been viewed, so a feasible thinning replacement must "
            "be preregistered as v2 and evaluated on new data or a new holdout."
        ),
        "The prior Final-test period is exposed and remains unavailable as a pristine holdout.",
        "MGC transfer validation was not authorized or inspected.",
        "No trading policy, transaction-cost model, PnL, or Sharpe result exists.",
    ]
    issues.extend(
        f"GARCH critical check `{row.check_name}` failed: {row.details}"
        for row in failed_garch.itertuples(index=False)
    )

    sections = {
        "Purpose": (
            "A separate, from-scratch experiment testing whether a bounded batch of "
            "book-derived/adapted features adds stable, economically meaningful "
            "Development-to-Validation information beyond the trusted one-minute GC "
            "feature matrix. It does not rewrite the historical statistical notebook."
        ),
        "Upstream dependencies": _markdown_table(
            upstream,
            columns=(
                "artifact_name",
                "exact_path",
                "row_count",
                "column_count",
                "research_partition_coverage",
                "validation_status",
                "mlat_use",
            ),
        ),
        "Book coverage": (
            f"Stage 1 is complete for all {coverage_state['physical_pages']} physical "
            f"pages, including {coverage_state['substantive_pages']} substantive "
            f"preface/chapter/Appendix pages, {coverage_state['chapters']} chapters, and "
            "the Appendix. Every chapter/Appendix has a page-traceable summary."
        ),
        "Frozen hypothesis batch": (
            f"12 of {len(hypotheses)} preregistered hypotheses were selected before "
            "outcome evaluation. Batch SHA-256: "
            f"`{execution['batch_sha256']}`. Rejected/deferred hypotheses: "
            f"{len(rejected)}."
        ),
        "Architecture": (
            "The old notebook remains an immutable upstream producer. The new notebook "
            "loads hash-documented GC artifacts, builds registry-controlled causal "
            "features in reusable typed modules, evaluates only Development and "
            "Validation, persists versioned evidence, reload-verifies it, and records "
            "completion in an execution manifest.\n\n"
            + _bullet_list(f"`{module}`" for module in source_modules)
        ),
        "Implemented features": _markdown_table(
            registry,
            columns=(
                "hypothesis_id",
                "feature_name",
                "source_type",
                "target_family",
                "minimum_history",
                "status",
            ),
        ),
        "Evaluation contract": (
            "Chronological Development and Validation; London/New York separation; "
            "New York dates as evidence units; daily Spearman IC; 2,000 date-block "
            "bootstrap iterations; Development-fitted quintiles applied unchanged to "
            "Validation; Benjamini-Hochberg control; year/session stability; horizon "
            "thinning; two-GC-tick directional economics; Development overlap veto; and "
            "60/180-minute partial-IC tests beyond ATR(20) and the frozen existing set. "
            f"Contract SHA-256: `{execution['contract_sha256']}`. The frozen v1 "
            "contract was not relaxed after observing outcomes."
        ),
        "Key results": (
            f"The notebook completed {notebook_state['executed_code_cells']} code cells "
            f"with no error outputs and passed its engineering gate. "
            f"{len(advanced)} of {len(verdicts)} features received an advancement verdict. "
            "All otherwise valid, nonredundant candidates are retained as "
            "`RESEARCH_ONLY`: this is a structural authorization veto, not an "
            "empirical rejection of their descriptive evidence. The frozen v1 "
            f"advancement gate is `{execution['advancement_gate_feasible']}` because "
            f"{authorization_reason} The multivariate authorization gate is "
            f"`{execution['multivariate_authorized']}`.\n\n"
            + _markdown_table(verdict_counts)
        ),
        "GARCH decision": (
            f"`{execution['garch_verdict']}`. GARCH is audit-only and is not a v1 matrix "
            f"feature. Critical failed checks: {len(failed_garch)}."
        ),
        "Feature verdicts": _markdown_table(
            verdicts,
            columns=(
                "feature_name",
                "pre_authorization_verdict",
                "final_decision",
                "verdict_reason",
            ),
            labels={
                "pre_authorization_verdict": (
                    "descriptive_pre_authorization_verdict"
                )
            },
        ),
        "Artifact paths": _bullet_list(f"`{path}`" for path in artifact_paths),
        "Unresolved issues": _bullet_list(issues),
        "Next step": str(execution["next_stage"]),
    }
    missing_sections = [name for name in CONTEXT_SECTIONS if name not in sections]
    if missing_sections:
        raise AssertionError(f"Context report sections were not built: {missing_sections}")
    body = [
        "# MLAT Feature Research — Context Report",
        "",
        "Concise handoff for reproducing and extending the completed v1 experiment.",
    ]
    for name in CONTEXT_SECTIONS:
        body.extend(["", f"## {name}", "", sections[name]])
    report = "\n".join(body).rstrip() + "\n"
    for name in CONTEXT_SECTIONS:
        if f"## {name}" not in report:
            raise AssertionError(f"Required context-report section is absent: {name}")
    return report


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_mlat_final_docs(*, write: bool = True) -> tuple[str, str]:
    """Validate executed MLAT v1 evidence and build both required Markdown reports."""

    inputs = _report_inputs()
    final_report = _build_final_report(inputs)
    context_report = _build_context_report(inputs)
    if write:
        _atomic_write(FINAL_REPORT, final_report)
        _atomic_write(CONTEXT_REPORT, context_report)
    return final_report, context_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate all inputs and render reports in memory without writing them",
    )
    arguments = parser.parse_args()
    final_report, context_report = build_mlat_final_docs(write=not arguments.verify_only)
    action = "validated" if arguments.verify_only else "wrote"
    print(
        f"{action} final report ({len(final_report):,} characters) and "
        f"context report ({len(context_report):,} characters)"
    )


if __name__ == "__main__":
    main()
