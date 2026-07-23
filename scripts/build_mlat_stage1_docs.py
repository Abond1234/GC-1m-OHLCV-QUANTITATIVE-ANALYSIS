"""Build the frozen Stage 1 MLAT research documentation.

The script is deterministic and outcome-free.  It may inspect the presence of
chapter summaries, but it never loads forward labels or feature/outcome results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "project_docs" / "mlat_feature_research"
CHAPTERS = DOCS / "chapter_summaries"

CHAPTER_RANGES = (
    (1, 42, 58),
    (2, 59, 94),
    (3, 95, 114),
    (4, 115, 152),
    (5, 153, 178),
    (6, 179, 202),
    (7, 203, 248),
    (8, 249, 279),
    (9, 280, 319),
    (10, 320, 349),
    (11, 350, 386),
    (12, 387, 428),
    (13, 429, 460),
    (14, 461, 485),
    (15, 486, 505),
    (16, 506, 532),
    (17, 533, 568),
    (18, 569, 606),
    (19, 607, 638),
    (20, 639, 662),
    (21, 663, 690),
    (22, 691, 723),
    (23, 724, 734),
)
APPENDIX_RANGE = (735, 764)
REQUIRED_SUMMARY_SECTIONS = (
    "## Main argument",
    "## Concepts and definitions",
    "## Formulas and notation",
    "## Assumptions",
    "## Procedures described",
    "## Feature and model examples",
    "## Statistical, validation, and backtesting warnings",
    "## Implementation patterns worth preserving",
    "## Dated APIs and examples",
    "## Project 1 relevance",
    "## One-minute GC adaptation",
    "## Unsuitable or deferred ideas",
    "## Exact PDF page map",
    "## Unresolved ambiguities and extraction confidence",
)
SOURCE_PDF_SHA256 = "4439a7be25210efb1173f30bcb75cb67743acff78de880578dc8a6de351aa39a"
FROZEN_BATCH_SHA256 = "8c8b74267635566f07011c2789ad4d323647f19eba1f1f3426a154c1a850acc0"
FROZEN_CONTRACT_SHA256 = "31ea8ff8255285d580dc9c2cb71bcbff85d05011e771b49f2f09d24cad810320"
STAGE1_REQUIRED_ARTIFACTS = (
    "README.md",
    "mlat_book_map.md",
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
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _summary_validation(path: Path, start: int, end: int) -> tuple[bool, list[str]]:
    """Validate the persisted coverage contract, not merely file presence."""

    issues: list[str] = []
    if not path.exists():
        return False, ["missing file"]
    text = path.read_text(encoding="utf-8")
    if len(text) <= 500:
        issues.append("summary is too short")
    for heading in REQUIRED_SUMMARY_SECTIONS:
        if heading not in text:
            issues.append(f"missing section: {heading}")
    if f"{start}-{end}" not in text:
        issues.append(f"missing assigned physical range {start}-{end}")
    citations = []
    for match in re.finditer(r"PDF pp?\.?\s*(\d+)(?:-(\d+))?", text):
        citation_start = int(match.group(1))
        citation_end = int(match.group(2) or citation_start)
        citations.extend(range(citation_start, citation_end + 1))
    if not citations:
        issues.append("missing exact PDF page citations")
    outside_citations = sorted({page for page in citations if page < start or page > end})
    if outside_citations:
        issues.append(f"PDF citations outside assigned range: {outside_citations}")
    page_map_marker = "## Exact PDF page map"
    if page_map_marker in text:
        page_map = text.split(page_map_marker, 1)[1].split("\n## ", 1)[0]
        covered: set[int] = set()
        for line in page_map.splitlines():
            match = re.search(r"\|\s*(\d+)(?:-(\d+))?\s*\|", line)
            if match:
                map_start = int(match.group(1))
                map_end = int(match.group(2) or map_start)
                covered.update(range(map_start, map_end + 1))
        expected = set(range(start, end + 1))
        missing_pages = sorted(expected - covered)
        outside_pages = sorted(covered - expected)
        if missing_pages:
            issues.append(f"exact page map misses pages: {missing_pages}")
        if outside_pages:
            issues.append(f"exact page map includes outside pages: {outside_pages}")
    return not issues, issues


def _read_manifest_rows() -> list[dict[str, str]]:
    path = DOCS / "mlat_book_ingestion_manifest.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _expanded_numbers(text: str) -> set[int]:
    values: set[int] = set()
    for match in re.finditer(r"\b(\d+)(?:-(\d+))?\b", text):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end >= start:
            values.update(range(start, end + 1))
    return values


def _documented_visual_pages(path: Path) -> set[int]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    canonical = [
        line for line in lines if "**visual inspection pages:**" in line.lower()
    ]
    candidates = canonical or [
        line for line in lines if "visually inspected" in line.lower()
    ]
    pages: set[int] = set()
    for line in candidates:
        expressions = re.findall(
            r"PDF pp?\.?\s*("
            r"\d+(?:-\d+)?"
            r"(?:\s*,\s*\d+(?:-\d+)?)*"
            r"(?:\s*,?\s*and\s*\d+(?:-\d+)?)?"
            r")",
            line,
        )
        for expression in expressions:
            pages.update(_expanded_numbers(expression))
    return pages


def _manifest_validation() -> tuple[bool, list[str]]:
    """Verify substantive units are complete and visual pages stay in range."""

    path = DOCS / "mlat_book_ingestion_manifest.csv"
    if not path.exists():
        return False, ["missing ingestion manifest"]
    rows = _read_manifest_rows()
    expected_start = 1
    issues: list[str] = []
    for row in rows:
        coverage_id = row["coverage_id"]
        start = int(row["pdf_page_start"])
        end = int(row["pdf_page_end"])
        count = int(row["page_count"])
        if start != expected_start:
            issues.append(
                f"{coverage_id} begins at {start}; expected contiguous page {expected_start}"
            )
        if count != end - start + 1:
            issues.append(
                f"{coverage_id} page_count {count} disagrees with range {start}-{end}"
            )
        if row["text_extraction_status"] != "COMPLETE":
            issues.append(
                f"{coverage_id} extraction is {row['text_extraction_status']}"
            )
        expected_start = end + 1
    if expected_start != 859:
        issues.append(f"manifest ends at page {expected_start - 1}, not 858")
    substantive = [
        row for row in rows if row["section_type"] in {"chapter", "appendix"}
    ]
    if len(substantive) != 24:
        issues.append(f"expected 24 chapter/appendix rows; found {len(substantive)}")
    expected_units = {
        f"MLAT-COV-{number + 2:03d}": (
            "chapter",
            f"Chapter {number} -",
            start,
            end,
            f"chapter_summaries/chapter_{number:02d}.md",
        )
        for number, start, end in CHAPTER_RANGES
    }
    expected_units["MLAT-COV-026"] = (
        "appendix",
        "Appendix -",
        *APPENDIX_RANGE,
        "chapter_summaries/appendix_alpha_factor_library.md",
    )
    for row in substantive:
        coverage_id = row["coverage_id"]
        start = int(row["pdf_page_start"])
        end = int(row["pdf_page_end"])
        expected = expected_units.get(coverage_id)
        if expected is None:
            issues.append(f"unexpected substantive coverage id: {coverage_id}")
        else:
            expected_type, title_prefix, expected_start, expected_end, summary_path = expected
            if row["section_type"] != expected_type:
                issues.append(f"{coverage_id} has wrong section_type")
            if not row["title"].startswith(title_prefix):
                issues.append(f"{coverage_id} has wrong title: {row['title']}")
            if (start, end) != (expected_start, expected_end):
                issues.append(
                    f"{coverage_id} range {start}-{end}; expected {expected_start}-{expected_end}"
                )
            if row["summary_path"] != summary_path:
                issues.append(
                    f"{coverage_id} summary_path {row['summary_path']}; expected {summary_path}"
                )
        if row["status"] != "COMPLETE":
            issues.append(f"{coverage_id} status is {row['status']}")
        visual = [
            int(value)
            for value in row["visual_inspection_pages"].split(",")
            if value.strip()
        ]
        if not visual:
            issues.append(f"{coverage_id} has no visual inspection page")
        outside = [page for page in visual if page < start or page > end]
        if outside:
            issues.append(f"{coverage_id} visual pages outside range: {outside}")
        summary = DOCS / row["summary_path"]
        documented_visual = _documented_visual_pages(summary)
        if set(visual) != documented_visual:
            issues.append(
                f"{coverage_id} manifest visual pages {sorted(visual)} disagree with "
                f"summary {sorted(documented_visual)}"
            )
        summary_text = summary.read_text(encoding="utf-8") if summary.exists() else ""
        if "confidence: medium" in summary_text.lower() and row["extraction_uncertainty"] == "LOW":
            issues.append(
                f"{coverage_id} is LOW uncertainty despite a medium-confidence note"
            )
    return not issues, issues


def _stage1_artifact_validation() -> tuple[bool, list[str]]:
    """Require the durable Stage 1/audit trail and portable source identity."""

    issues: list[str] = []
    for relative in STAGE1_REQUIRED_ARTIFACTS:
        path = DOCS / relative
        if not path.exists() or path.stat().st_size == 0:
            issues.append(f"missing or empty artifact: {relative}")
    map_path = DOCS / "mlat_book_map.md"
    if map_path.exists():
        map_text = map_path.read_text(encoding="utf-8")
        for part in ("Part 1", "Part 2", "Part 3", "Part 4"):
            if f"| {part} |" not in map_text:
                issues.append(f"book map is missing {part}")
        if SOURCE_PDF_SHA256 not in map_text:
            issues.append("book map is missing the supplied PDF SHA-256")
    csv_contracts = (
        (
            "mlat_concept_registry.csv",
            30,
            ("concept_id", "chapter", "pdf_page", "definition", "recommendation"),
        ),
        (
            "mlat_source_traceability.csv",
            13,
            (
                "hypothesis_id",
                "source_chapter",
                "source_pdf_page",
                "project_adaptation",
                "status",
            ),
        ),
        (
            "mlat_existing_feature_overlap_audit.csv",
            12,
            (
                "hypothesis_id",
                "feature_name",
                "formula_overlap_assessment",
                "preimplementation_decision",
            ),
        ),
        (
            "mlat_upstream_artifact_manifest.csv",
            10,
            (
                "artifact_name",
                "exact_path",
                "schema_fingerprint",
                "file_sha256",
                "validation_status",
            ),
        ),
    )
    for relative, expected_rows, required_fields in csv_contracts:
        path = DOCS / relative
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_rows:
            issues.append(
                f"{relative} has {len(rows)} rows; expected {expected_rows}"
            )
        for row_number, row in enumerate(rows, start=2):
            missing = [field for field in required_fields if not row.get(field, "").strip()]
            if missing:
                issues.append(
                    f"{relative}:{row_number} has empty required fields {missing}"
                )
    formulas = DOCS / "mlat_formulas_and_definitions.md"
    if formulas.exists():
        text = formulas.read_text(encoding="utf-8")
        if len(re.findall(r"^## MLAT-F\d+", text, flags=re.MULTILINE)) != 14:
            issues.append("formula registry does not contain exactly 14 formula entries")
        if text.count("- Book trace:") != 14:
            issues.append("formula registry does not page-trace all 14 entries")
    hypotheses = DOCS / "mlat_feature_hypothesis_catalog.md"
    if hypotheses.exists():
        text = hypotheses.read_text(encoding="utf-8")
        if len(re.findall(r"^### MLAT-H\d+", text, flags=re.MULTILINE)) != 29:
            issues.append("hypothesis catalog does not contain exactly 29 full entries")
    frozen_files = (
        ("mlat_feature_batch_v1.md", FROZEN_BATCH_SHA256),
        ("mlat_feature_research_contract_v1.md", FROZEN_CONTRACT_SHA256),
    )
    for relative, expected_hash in frozen_files:
        path = DOCS / relative
        if path.exists():
            observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed_hash != expected_hash:
                issues.append(
                    f"{relative} hash {observed_hash} differs from frozen {expected_hash}"
                )
    return not issues, issues


def _outcome_artifacts_exist() -> bool:
    roots = (
        ROOT
        / "data"
        / "processed"
        / "statistical_research"
        / "mlat_feature_research",
        ROOT / "reports" / "statistical_research" / "mlat_feature_research",
        ROOT / "reports" / "figures" / "mlat_feature_research",
    )
    return any(path.exists() and any(path.rglob("*")) for path in roots)


CONCEPTS = [
    {
        "concept_id": "MLAT-C001",
        "concept_name": "Research-to-execution workflow",
        "chapter": "1",
        "pdf_page": "42-58",
        "category": "model validation",
        "definition": "A staged path from economic hypothesis and data through validation, portfolio/execution design, and monitoring.",
        "assumptions": "Each stage has explicit information and decision boundaries.",
        "intended_target": "governance",
        "applicability": "DIRECT",
        "required_data": "research metadata",
        "implementation_difficulty": "LOW",
        "leakage_risk": "MEDIUM",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "statistical notebook section gates",
        "recommendation": "ADOPT",
    },
    {
        "concept_id": "MLAT-C002",
        "concept_name": "Point-in-time market data",
        "chapter": "2",
        "pdf_page": "59-94",
        "category": "backtesting",
        "definition": "Data values and identifiers must reflect only what was knowable at the simulated decision time.",
        "assumptions": "Timestamps, contract selection, and revisions are governed.",
        "intended_target": "execution",
        "applicability": "DIRECT",
        "required_data": "timestamped OHLCV and contract metadata",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "HIGH",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "trusted active-contract table and t to t+1 convention",
        "recommendation": "ADOPT",
    },
    {
        "concept_id": "MLAT-C003",
        "concept_name": "Alternative-data quality screen",
        "chapter": "3",
        "pdf_page": "95-114",
        "category": "alternative data",
        "definition": "Evaluate provenance, coverage, history, survivorship, legal rights, latency, and signal content before research use.",
        "assumptions": "The data source can be audited.",
        "intended_target": "later research",
        "applicability": "LATER",
        "required_data": "alternative dataset",
        "implementation_difficulty": "HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "UNKNOWN",
        "existing_project_equivalent": "none",
        "recommendation": "DEFER_DIFFERENT_DATA",
    },
    {
        "concept_id": "MLAT-C004",
        "concept_name": "Lagged-return features",
        "chapter": "4",
        "pdf_page": "130-132",
        "category": "direction",
        "definition": "Associate returns ending before or at the feature timestamp with the current observation.",
        "assumptions": "The series is sorted and lagging is segmented.",
        "intended_target": "direction",
        "applicability": "DIRECT",
        "required_data": "close prices",
        "implementation_difficulty": "LOW",
        "leakage_risk": "MEDIUM",
        "redundancy_risk": "HIGH",
        "existing_project_equivalent": "return_1m_bps through return_60m_bps",
        "recommendation": "REJECT_REDUNDANT",
    },
    {
        "concept_id": "MLAT-C005",
        "concept_name": "Bollinger standardized location and bandwidth",
        "chapter": "4 / Appendix",
        "pdf_page": "131-133; 740-742",
        "category": "direction / volatility",
        "definition": "Express price location and band width relative to a trailing mean and standard deviation.",
        "assumptions": "Trailing windows are complete and causal; standard deviation is nonzero.",
        "intended_target": "direction and expansion",
        "applicability": "ADAPTED",
        "required_data": "close",
        "implementation_difficulty": "LOW",
        "leakage_risk": "LOW",
        "redundancy_risk": "MEDIUM",
        "existing_project_equivalent": "range position and compression families",
        "recommendation": "IMPLEMENT_AND_TEST_OVERLAP",
    },
    {
        "concept_id": "MLAT-C006",
        "concept_name": "Relative strength index",
        "chapter": "4",
        "pdf_page": "132",
        "category": "direction",
        "definition": "A bounded transformation of recent positive versus negative price changes.",
        "assumptions": "A smoothing convention and warm-up are declared.",
        "intended_target": "direction / state",
        "applicability": "ADAPTED",
        "required_data": "close",
        "implementation_difficulty": "LOW",
        "leakage_risk": "LOW",
        "redundancy_risk": "MEDIUM",
        "existing_project_equivalent": "momentum and directional-persistence features",
        "recommendation": "IMPLEMENT_CUTLER_VARIANT",
    },
    {
        "concept_id": "MLAT-C007",
        "concept_name": "Kalman filtering",
        "chapter": "4",
        "pdf_page": "133-136",
        "category": "risk state",
        "definition": "A sequential state-space update combining a predicted latent state with a noisy observation.",
        "assumptions": "Linear state dynamics and declared noise distributions/covariances.",
        "intended_target": "filtered state",
        "applicability": "ADAPTED",
        "required_data": "price or return series",
        "implementation_difficulty": "HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "HIGH",
        "existing_project_equivalent": "rolling trend and EMA-like state",
        "recommendation": "DEFER",
    },
    {
        "concept_id": "MLAT-C008",
        "concept_name": "Wavelet denoising",
        "chapter": "4",
        "pdf_page": "137-140",
        "category": "risk state",
        "definition": "Decompose a signal across scales, threshold coefficients, and reconstruct a denoised series.",
        "assumptions": "Transform support, threshold, and boundary treatment are valid.",
        "intended_target": "filtered state",
        "applicability": "REJECTED",
        "required_data": "return series",
        "implementation_difficulty": "HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "MEDIUM",
        "existing_project_equivalent": "none",
        "recommendation": "REJECT_LEAKAGE_UNLESS_CAUSAL_REDESIGN",
    },
    {
        "concept_id": "MLAT-C009",
        "concept_name": "Information coefficient and factor quantiles",
        "chapter": "4",
        "pdf_page": "141-150",
        "category": "model validation",
        "definition": "Use rank association and ordered buckets to assess factor/outcome relationships and stability.",
        "assumptions": "Evidence units account for dependence and bins are fitted out of sample.",
        "intended_target": "all feature targets",
        "applicability": "DIRECT",
        "required_data": "features and forward outcomes",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "MEDIUM",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "daily Spearman IC and Development-fitted quintiles",
        "recommendation": "ADOPT",
    },
    {
        "concept_id": "MLAT-C010",
        "concept_name": "Risk-adjusted performance metrics",
        "chapter": "5",
        "pdf_page": "169-178",
        "category": "portfolio",
        "definition": "Evaluate returns jointly with volatility, downside, drawdown, and benchmark exposure.",
        "assumptions": "A valid sequential return series exists.",
        "intended_target": "strategy evaluation",
        "applicability": "LATER",
        "required_data": "sequential net returns",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "MEDIUM",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "performance diagnostics",
        "recommendation": "DEFER_UNTIL_POLICY",
    },
    {
        "concept_id": "MLAT-C011",
        "concept_name": "Mutual information",
        "chapter": "6",
        "pdf_page": "192",
        "category": "model validation",
        "definition": "A dependence measure that can detect nonlinear information beyond linear correlation.",
        "assumptions": "Estimator bias, sample size, and repeated testing are controlled.",
        "intended_target": "feature screening",
        "applicability": "LATER",
        "required_data": "features and outcomes",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "MEDIUM",
        "redundancy_risk": "MEDIUM",
        "existing_project_equivalent": "rank IC and incremental partial IC",
        "recommendation": "RESEARCH_ONLY",
    },
    {
        "concept_id": "MLAT-C012",
        "concept_name": "Bias-variance trade-off",
        "chapter": "6",
        "pdf_page": "192-195",
        "category": "model validation",
        "definition": "Model error reflects both systematic underfit and sample-sensitive overfit.",
        "assumptions": "Generalization is estimated on unseen chronological data.",
        "intended_target": "model selection",
        "applicability": "DIRECT",
        "required_data": "chronological partitions",
        "implementation_difficulty": "LOW",
        "leakage_risk": "HIGH",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "bounded batch and model authorization gate",
        "recommendation": "ADOPT",
    },
    {
        "concept_id": "MLAT-C013",
        "concept_name": "Purging and embargo",
        "chapter": "6",
        "pdf_page": "199-200",
        "category": "model validation",
        "definition": "Remove or separate samples whose label information overlaps training and validation intervals.",
        "assumptions": "Label spans are known.",
        "intended_target": "model validation",
        "applicability": "DIRECT",
        "required_data": "timestamps and horizon metadata",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "HIGH",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "date blocks and chronological embargo",
        "recommendation": "ADOPT_WHEN_MODELLING",
    },
    {
        "concept_id": "MLAT-C014",
        "concept_name": "Regularized linear benchmark",
        "chapter": "7",
        "pdf_page": "214-237",
        "category": "direction / volatility",
        "definition": "Use shrinkage to benchmark incremental multivariate information with controlled complexity.",
        "assumptions": "Features and penalties are fitted inside chronological training data.",
        "intended_target": "multivariate benchmark",
        "applicability": "LATER",
        "required_data": "frozen shortlist",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "HIGH",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "Section 9 ridge benchmark",
        "recommendation": "GATED",
    },
    {
        "concept_id": "MLAT-C015",
        "concept_name": "Event-driven backtesting",
        "chapter": "8",
        "pdf_page": "255-273",
        "category": "backtesting",
        "definition": "Simulate time, calendars, information arrival, orders, positions, and costs explicitly.",
        "assumptions": "A frozen sequential policy exists.",
        "intended_target": "execution",
        "applicability": "LATER",
        "required_data": "bars, signals, order rules, costs",
        "implementation_difficulty": "HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "independent sequential backtest",
        "recommendation": "DEFER_UNTIL_SIGNAL",
    },
    {
        "concept_id": "MLAT-C016",
        "concept_name": "Stationarity and differencing",
        "chapter": "9",
        "pdf_page": "280-290",
        "category": "model validation",
        "definition": "Time-series model assumptions require stable distributional structure or an explicit transformation.",
        "assumptions": "Structural breaks and intraday seasonality are diagnosed.",
        "intended_target": "time-series modelling",
        "applicability": "DIRECT",
        "required_data": "chronological returns",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "MEDIUM",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "return transforms and year/session stability",
        "recommendation": "ADOPT",
    },
    {
        "concept_id": "MLAT-C017",
        "concept_name": "AR and variance-ratio state",
        "chapter": "9",
        "pdf_page": "290-296",
        "category": "direction",
        "definition": "Lag dependence distinguishes persistence from mean reversion relative to a random-walk benchmark.",
        "assumptions": "Rolling estimation is causal and locally stable.",
        "intended_target": "direction / regime",
        "applicability": "ADAPTED",
        "required_data": "log returns",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "LOW",
        "redundancy_risk": "MEDIUM",
        "existing_project_equivalent": "return_autocorrelation_15",
        "recommendation": "IMPLEMENT_VARIANCE_RATIO_AND_TEST_OVERLAP",
    },
    {
        "concept_id": "MLAT-C018",
        "concept_name": "ARCH/GARCH conditional variance",
        "chapter": "9",
        "pdf_page": "297-301",
        "category": "volatility",
        "definition": "Forecast conditional variance from lagged shocks and lagged conditional variance.",
        "assumptions": "Returns and residuals are correctly specified; persistence is stationary; diagnostics pass.",
        "intended_target": "volatility / risk state",
        "applicability": "ADAPTED",
        "required_data": "segmented GC returns",
        "implementation_difficulty": "HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "HIGH",
        "existing_project_equivalent": "exploratory notebook GARCH and ATR/RV anchors",
        "recommendation": "AUDIT_SEPARATELY",
    },
    {
        "concept_id": "MLAT-C019",
        "concept_name": "Bayesian state uncertainty",
        "chapter": "10",
        "pdf_page": "320-349",
        "category": "risk state",
        "definition": "Represent parameter/state estimates as distributions updated with evidence.",
        "assumptions": "Priors, likelihood, convergence, and calibration are defensible.",
        "intended_target": "risk state",
        "applicability": "LATER",
        "required_data": "time series and prior model",
        "implementation_difficulty": "HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "MEDIUM",
        "existing_project_equivalent": "none",
        "recommendation": "DEFER",
    },
    {
        "concept_id": "MLAT-C020",
        "concept_name": "Tree feature importance",
        "chapter": "11",
        "pdf_page": "366-376",
        "category": "model validation",
        "definition": "Summarize split-based or permutation-based model reliance on predictors.",
        "assumptions": "The model is valid and importance bias/correlation are addressed.",
        "intended_target": "interpretation",
        "applicability": "LATER",
        "required_data": "fitted nonlinear model",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "MEDIUM",
        "redundancy_risk": "HIGH",
        "existing_project_equivalent": "none",
        "recommendation": "DO_NOT_USE_AS_EDGE_PROOF",
    },
    {
        "concept_id": "MLAT-C021",
        "concept_name": "Gradient boosting",
        "chapter": "12",
        "pdf_page": "387-428",
        "category": "model validation",
        "definition": "Sequentially add weak learners that correct prior residuals under a regularized objective.",
        "assumptions": "Complexity and tuning are nested in chronological validation.",
        "intended_target": "nonlinear benchmark",
        "applicability": "LATER",
        "required_data": "frozen shortlist",
        "implementation_difficulty": "HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "none",
        "recommendation": "DEFER",
    },
    {
        "concept_id": "MLAT-C022",
        "concept_name": "PCA and feature clustering",
        "chapter": "13",
        "pdf_page": "429-460",
        "category": "risk state",
        "definition": "Compress or group correlated variables to identify shared variation and redundancy.",
        "assumptions": "Scaling and fit scope are Development-only and components are stable.",
        "intended_target": "redundancy / regime",
        "applicability": "ADAPTED",
        "required_data": "feature matrix",
        "implementation_difficulty": "MEDIUM",
        "leakage_risk": "MEDIUM",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "Development-only hierarchical feature clustering",
        "recommendation": "USE_CLUSTERING_NOT_NEW_PCA_FEATURES",
    },
    {
        "concept_id": "MLAT-C023",
        "concept_name": "Sentiment and topic features",
        "chapter": "14-16",
        "pdf_page": "461-532",
        "category": "alternative data",
        "definition": "Transform text into sentiment, topics, or dense semantic representations.",
        "assumptions": "Time-stamped text is available without revision or publication leakage.",
        "intended_target": "direction / risk state",
        "applicability": "DIFFERENT_DATA",
        "required_data": "news, filings, or communications",
        "implementation_difficulty": "HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "UNKNOWN",
        "existing_project_equivalent": "none",
        "recommendation": "DEFER_DIFFERENT_DATA",
    },
    {
        "concept_id": "MLAT-C024",
        "concept_name": "Deep sequence and representation models",
        "chapter": "17-21",
        "pdf_page": "533-690",
        "category": "model validation",
        "definition": "Learn nonlinear representations or sequence dynamics with neural networks.",
        "assumptions": "Large representative data, nested validation, calibration, and strong baselines.",
        "intended_target": "later modelling",
        "applicability": "LATER",
        "required_data": "large frozen training corpus",
        "implementation_difficulty": "VERY HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "UNKNOWN",
        "existing_project_equivalent": "none",
        "recommendation": "DEFER",
    },
    {
        "concept_id": "MLAT-C025",
        "concept_name": "Amihud illiquidity",
        "chapter": "20 / Appendix",
        "pdf_page": "656; 752",
        "category": "risk state",
        "definition": "Average absolute return per unit of dollar-volume proxy.",
        "assumptions": "Volume is comparable within the instrument and the futures adaptation is explicit.",
        "intended_target": "expansion / risk state",
        "applicability": "ADAPTED",
        "required_data": "close and volume",
        "implementation_difficulty": "LOW",
        "leakage_risk": "LOW",
        "redundancy_risk": "MEDIUM",
        "existing_project_equivalent": "volume_per_tick_range and liquidity_vacuum_score_exp",
        "recommendation": "IMPLEMENT_AND_TEST_OVERLAP",
    },
    {
        "concept_id": "MLAT-C026",
        "concept_name": "Synthetic time-series fidelity",
        "chapter": "21",
        "pdf_page": "663-690",
        "category": "model validation",
        "definition": "Synthetic data require distributional, predictive, and discriminative fidelity tests.",
        "assumptions": "Synthetic generation does not erase rare risk events.",
        "intended_target": "data augmentation",
        "applicability": "LATER",
        "required_data": "validated generator and holdout",
        "implementation_difficulty": "VERY HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "UNKNOWN",
        "existing_project_equivalent": "none",
        "recommendation": "DEFER",
    },
    {
        "concept_id": "MLAT-C027",
        "concept_name": "Reinforcement-learning trading environment",
        "chapter": "22",
        "pdf_page": "691-723",
        "category": "execution",
        "definition": "An agent learns actions from rewards in a stateful market environment.",
        "assumptions": "The simulator and reward encode realistic market dynamics and costs.",
        "intended_target": "execution policy",
        "applicability": "REJECTED_GOVERNANCE",
        "required_data": "approved sequential environment",
        "implementation_difficulty": "VERY HIGH",
        "leakage_risk": "HIGH",
        "redundancy_risk": "UNKNOWN",
        "existing_project_equivalent": "none",
        "recommendation": "REJECT_FOR_V1",
    },
    {
        "concept_id": "MLAT-C028",
        "concept_name": "Chaikin money flow adaptation",
        "chapter": "Appendix",
        "pdf_page": "752-753",
        "category": "direction / risk state",
        "definition": "Weight volume by close location within the bar range and aggregate causally.",
        "assumptions": "OHLC ranges are valid; zero-range bars receive an explicit neutral policy.",
        "intended_target": "direction / activity",
        "applicability": "ADAPTED",
        "required_data": "OHLCV",
        "implementation_difficulty": "LOW",
        "leakage_risk": "LOW",
        "redundancy_risk": "MEDIUM",
        "existing_project_equivalent": "close_location_value and volume_price_alignment_10",
        "recommendation": "IMPLEMENT_AND_TEST_OVERLAP",
    },
    {
        "concept_id": "MLAT-C029",
        "concept_name": "Average true range",
        "chapter": "Appendix",
        "pdf_page": "754-755",
        "category": "volatility",
        "definition": "A trailing average of true range including the prior-close gap.",
        "assumptions": "Prior close belongs to the same valid continuity segment.",
        "intended_target": "expansion / risk state",
        "applicability": "DIRECT",
        "required_data": "high, low, close",
        "implementation_difficulty": "LOW",
        "leakage_risk": "LOW",
        "redundancy_risk": "VERY HIGH",
        "existing_project_equivalent": "atr_5, atr_20, atr_60",
        "recommendation": "USE_EXISTING_ANCHOR",
    },
    {
        "concept_id": "MLAT-C030",
        "concept_name": "Backtest-overfitting warning",
        "chapter": "23",
        "pdf_page": "724-734",
        "category": "backtesting",
        "definition": "Repeated design choices against the same history can convert noise into an apparently successful strategy.",
        "assumptions": "The research trail records every tested family and holdout exposure.",
        "intended_target": "governance",
        "applicability": "DIRECT",
        "required_data": "research registry and dated holdout",
        "implementation_difficulty": "LOW",
        "leakage_risk": "HIGH",
        "redundancy_risk": "LOW",
        "existing_project_equivalent": "bounded batch and non-pristine Final-test note",
        "recommendation": "ADOPT",
    },
]


FORMULAS = [
    (
        "MLAT-F001",
        "Log return",
        "r_t = ln(C_t / C_{t-1})",
        "C is close; prior close must be in the same continuity run.",
        "1 prior bar",
        "close of t",
        "none",
        "dimensionless; output may be expressed in bps",
        "Local return distribution may change by regime.",
        "No gap, contract, segment, roll, or invalid boundary.",
        "Non-positive prices and first-in-run are null.",
        "Chapters 4 and 9, PDF 131 and 280-301",
        "Use one-minute GC closes only.",
        "IMPLEMENTED_PRIMITIVE",
    ),
    (
        "MLAT-F002",
        "Bollinger z-score",
        "z_t = (C_t - mean_20(C)_t) / std_20(C)_t",
        "Trailing population standard deviation; project uses 20 complete one-minute bars.",
        "20 bars",
        "close of t",
        "window=20",
        "unitless",
        "Local mean/std are descriptive, not stationary guarantees.",
        "Complete single continuity run.",
        "Zero standard deviation is null.",
        "Chapter 4 PDF 131-133; Appendix 740-742",
        "Use standardized location rather than equity trading-rule thresholds.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F003",
        "Bollinger bandwidth",
        "bandwidth_t = 4 * std_20(C)_t / mean_20(C)_t",
        "Two-standard-deviation upper and lower bands imply total width 4 sigma.",
        "20 bars",
        "close of t",
        "window=20; band multiplier=2",
        "fraction of price",
        "Only locally scaled.",
        "Complete single continuity run.",
        "Zero/non-positive mean is null.",
        "Appendix PDF 740-742",
        "Expansion-state hypothesis; compare with existing compression features.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F004",
        "Cutler RSI",
        "RSI_t = 100 * G_t / (G_t + L_t), where G_t=mean_14(max(dC,0)), L_t=mean_14(max(-dC,0))",
        "This is the explicitly named simple-rolling Cutler variant, not TA-Lib Wilder recursion.",
        "15 bars",
        "close of t",
        "window=14 changes",
        "0 to 100",
        "Bounded transform does not create stationarity.",
        "Complete single continuity run.",
        "If gains+losses are zero, value is null.",
        "Chapter 4 PDF 132 (RSI example)",
        "Avoid fixed 30/70 trade rules; test continuous rank relationship.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F005",
        "Chaikin money flow adaptation",
        "CMF_t = sum_20(V_i * ((2C_i-H_i-L_i)/(H_i-L_i))) / sum_20(V_i)",
        "Rolling normalized adaptation of the Appendix A/D money-flow volume.",
        "20 bars",
        "close of t",
        "window=20",
        "-1 to 1",
        "Volume comparability is local to GC.",
        "Complete single continuity run.",
        "Zero-range bars contribute neutral multiplier 0; zero total volume is null.",
        "Appendix PDF 752-753",
        "Treat as volume-price state; do not infer aggressor side.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F006",
        "Amihud illiquidity adaptation",
        "ILLIQ_t = 1e9 * mean_60(|r_i| / (C_i * V_i))",
        "C*V is a within-GC notional-activity proxy; the fixed futures multiplier would only rescale ranks.",
        "60 returns (61 bars)",
        "close of t",
        "window=60",
        "scaled inverse notional",
        "Comparable within GC, not across contracts/products without further scaling.",
        "Complete single continuity run.",
        "Zero volume or non-positive close is null.",
        "Chapter 20 PDF 656; Appendix 752",
        "One-hour adaptation of the book's rolling daily equity measure.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F007",
        "Parkinson range volatility",
        "sigma_P,t = 1e4 * sqrt(mean_30(ln(H_i/L_i)^2) / (4 ln 2))",
        "Project-original OHLC estimator under the book's volatility-feature program; formula is not attributed to the book.",
        "30 bars",
        "close of t",
        "window=30",
        "bps per one-minute interval",
        "Assumes a diffusion-like high-low process; microstructure and jumps violate it.",
        "Complete single continuity run.",
        "Non-positive prices are null.",
        "Chapter 9 context PDF 297-301",
        "Use as expansion/risk state and compare with ATR/RV.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F008",
        "Rogers-Satchell range volatility",
        "sigma_RS,t = 1e4 * sqrt(mean_30[ln(H/O)ln(H/C)+ln(L/O)ln(L/C)])",
        "Project-original drift-robust OHLC estimator; formula is not attributed to the book.",
        "30 bars",
        "close of t",
        "window=30",
        "bps per one-minute interval",
        "Requires valid OHLC and may be noisy at one-minute frequency.",
        "Complete single continuity run.",
        "Tiny negative round-off is clipped to zero; material negatives are null.",
        "Chapter 9 context PDF 297-301",
        "Use as expansion/risk state and compare with simpler anchors.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F009",
        "Realized semivariance balance",
        "SVB_t = (sum_60(r_i^2 1[r_i>0]) - sum_60(r_i^2 1[r_i<0])) / sum_60(r_i^2)",
        "Project extension separating upside and downside realized variation.",
        "60 returns (61 bars)",
        "close of t",
        "window=60",
        "-1 to 1",
        "Local sign asymmetry can change by session/regime.",
        "Complete single continuity run.",
        "Zero total variation is null.",
        "Chapters 5 and 9 context PDF 169-178 and 297-301",
        "Potential risk-state/directional-shape feature, not a signed-return claim.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F010",
        "Bipower jump ratio",
        "JR_t = max(RV_t-BV_t,0)/RV_t; RV=sum_60(r_i^2); BV=(pi/2)*(60/59)*sum_60(|r_i||r_{i-1}|)",
        "Project extension distinguishing discontinuous variation from local continuous variation.",
        "61 returns (62 bars)",
        "close of t",
        "window=60",
        "0 to 1",
        "Bipower approximation can be distorted by microstructure and sparse bars.",
        "Complete single continuity run.",
        "Zero RV is null; negative numerator is floored at zero.",
        "Chapter 9 context PDF 297-301",
        "Risk/expansion-state candidate; exact incremental evidence required.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F011",
        "Variance ratio",
        "VR_t = Var_60(sum_5(r)_i) / (5 * Var_60(r_i))",
        "Overlapping five-minute sums provide a local persistence/mean-reversion state.",
        "64 returns (65 bars)",
        "close of t",
        "outer=60; aggregation=5",
        "unitless; random-walk reference near 1",
        "Overlapping observations bias naive inference; feature use is descriptive.",
        "Complete single continuity run.",
        "Zero one-minute variance is null.",
        "Chapter 9 PDF 280-296",
        "Compare with existing return autocorrelation and sign-change features.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F012",
        "Return-sign entropy",
        "H_t = -sum_{s in negative,zero,positive} p_s ln(p_s) / ln(3)",
        "Rolling empirical probabilities use 60 causal return signs.",
        "60 returns (61 bars)",
        "close of t",
        "window=60",
        "0 to 1",
        "Discrete sign bins discard magnitude and depend on zero-tick frequency.",
        "Complete single continuity run.",
        "Terms with p=0 contribute zero.",
        "Chapter 6 entropy context PDF 192",
        "Project adaptation for choppiness/risk state; compare with sign-change rate.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F013",
        "Volatility of volatility",
        "VoV_t = std_60(RV15_i) / mean_60(RV15_i), RV15_i=1e4*sqrt(mean_15(r^2)_i)",
        "Project extension measuring instability of a short realized-volatility state.",
        "74 returns (75 bars)",
        "close of t",
        "inner=15; outer=60",
        "non-negative coefficient of variation",
        "Only a local state; denominator can approach zero.",
        "Complete single continuity run.",
        "Zero/non-finite mean is null.",
        "Chapter 9 context PDF 297-301",
        "Risk-state candidate distinct from volatility level.",
        "FROZEN_V1",
    ),
    (
        "MLAT-F014",
        "GARCH(1,1) conditional variance",
        "sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2",
        "Gaussian diagnostic model with omega>0, alpha>=0, beta>=0, alpha+beta<1.",
        "expanding history",
        "before forecasted bar",
        "Development fit schedule declared separately",
        "return variance",
        "Conditional-variance specification and innovation distribution must be diagnosed.",
        "Recursion resets at every invalid continuity boundary.",
        "Convergence, persistence, initialization, and scaling are explicit.",
        "Chapter 9 PDF 297-301",
        "GC-only audit; not part of the frozen v1 feature matrix.",
        "AUDIT_ONLY",
    ),
]


HYPOTHESES = [
    ("MLAT-H001", "bollinger_zscore_20", "4 / Appendix", "131-133; 740-742", "MLAT-ADAPTED", "Standardized local price displacement may identify reversal/continuation states.", "Bollinger z-score in formula registry", "20 bars", "close", "close of t", "continuity run", "direction", "nonlinear or monotone", "range position, OLS slope", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H002", "bollinger_bandwidth_20", "Appendix", "740-742", "MLAT-DIRECT", "Narrow bands may precede expansion while wide bands may mean-revert.", "Normalized four-sigma band width", "20 bars", "close", "close of t", "continuity run", "expansion", "possibly negative at short horizons and nonlinear", "compression ratio, ATR ratios", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H003", "cutler_rsi_14", "4", "132", "MLAT-ADAPTED", "Bounded gain/loss balance may distinguish exhausted from persistent moves.", "Simple-rolling Cutler RSI", "15 bars", "close", "close of t", "continuity run", "direction / risk state", "no fixed 30/70 sign assumed", "momentum, persistence", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H004", "chaikin_money_flow_20", "Appendix", "752-753", "MLAT-ADAPTED", "Close location weighted by volume may reveal pressure not present in either input alone.", "Rolling normalized money-flow volume", "20 bars", "OHLCV", "close of t", "continuity run", "direction / risk state", "positive may indicate buying pressure", "CLV, signed volume, alignment", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H005", "amihud_illiquidity_60", "20 / Appendix", "656; 752", "MLAT-ADAPTED", "Movement per unit activity may identify fragile/liquidity-vacuum states.", "Mean absolute return divided by close-volume proxy", "60 returns (61 bars)", "close, volume", "close of t", "continuity run", "expansion / risk state", "higher may imply greater future risk", "volume_per_tick_range, liquidity_vacuum", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H006", "parkinson_volatility_30", "9", "297-301", "PROJECT-ORIGINAL-EXTENSION", "High-low information may estimate latent volatility more efficiently than close-only RV.", "Parkinson estimator", "30 bars", "high, low", "close of t", "continuity run", "volatility / expansion", "positive monotone", "ATR, RV, range", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H007", "rogers_satchell_volatility_30", "9", "297-301", "PROJECT-ORIGINAL-EXTENSION", "Drift-robust OHLC variation may retain information beyond ATR and close-only RV.", "Rogers-Satchell estimator", "30 bars", "OHLC", "close of t", "continuity run", "volatility / expansion", "positive monotone", "ATR, RV, range", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H008", "realized_semivariance_balance_60", "5 / 9", "169-178; 297-301", "PROJECT-ORIGINAL-EXTENSION", "Asymmetry in recent signed variation may distinguish downside/upside risk state.", "Signed semivariance difference over total RV", "60 returns (61 bars)", "close", "close of t", "continuity run", "risk state / direction", "shape unknown", "directional energy balance", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H009", "bipower_jump_ratio_60", "9", "297-301", "PROJECT-ORIGINAL-EXTENSION", "The share of local variation attributable to jumps may alter subsequent expansion and risk.", "Positive RV-minus-bipower share", "61 returns (62 bars)", "close", "close of t", "continuity run", "risk state / expansion", "higher may mark stress then mean reversion or persistence", "current range/ATR, liquidity vacuum", "LOW", "MEDIUM", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H010", "variance_ratio_60_5", "9", "280-296", "MLAT-ADAPTED", "Deviation from local random-walk variance scaling may identify persistence or mean reversion.", "Five-minute to one-minute variance ratio", "64 returns (65 bars)", "close", "close of t", "continuity run", "direction / risk state", "below 1 mean-reverting; above 1 persistent", "return autocorrelation, sign change", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H011", "return_sign_entropy_60", "6", "192", "PROJECT-ORIGINAL-EXTENSION", "Low sign entropy may reflect directional organization; high entropy may reflect chop.", "Normalized three-state Shannon entropy", "60 returns (61 bars)", "close", "close of t", "continuity run", "risk state / expansion", "shape unknown", "choppiness, sign change", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H012", "volatility_of_volatility_60", "9", "297-301", "PROJECT-ORIGINAL-EXTENSION", "Instability of volatility, rather than its level, may identify transition risk.", "CV of trailing RV15 over 60 values", "74 returns (75 bars)", "close", "close of t", "continuity run", "risk state / expansion", "higher may imply unstable expansion", "ATR/RV ratios", "LOW", "LOW", "LOW", "HIGH", "SELECT_V1", ""),
    ("MLAT-H013", "kalman_innovation_state", "4", "133-136", "MLAT-ADAPTED", "Sequential innovations may capture filtered state surprises.", "State-space innovation", "expanding", "close", "close of t", "continuity run", "direction / risk state", "unknown", "EMA/trend states", "HIGH", "HIGH", "MEDIUM", "MEDIUM", "DEFER", "Noise parameters and Development cross-fitting are not justified in v1."),
    ("MLAT-H014", "wavelet_denoised_return", "4", "137-140", "MLAT-ADAPTED", "Multiscale denoising may isolate trend.", "Wavelet threshold and inverse transform", "transform dependent", "close", "uncertain", "continuity run", "direction", "unknown", "trend features", "VERY HIGH", "HIGH", "HIGH", "LOW", "REJECT", "Book example uses a two-sided transform; causal reconstruction is not established."),
    ("MLAT-H015", "garch_variance_forecast", "9", "297-301", "MLAT-DIRECT", "Conditional variance may forecast future expansion beyond ATR.", "GARCH(1,1)", "expanding", "close", "before forecast bar", "continuity run", "volatility / risk state", "positive monotone", "ATR/RV anchors", "HIGH", "HIGH", "HIGH", "MEDIUM", "AUDIT_ONLY", "Prior implementation is invalid; keep outside frozen matrix pending segmented audit."),
    ("MLAT-H016", "ppo_12_26", "11 / Appendix", "355; 748-749", "MLAT-DIRECT", "Normalized EMA spread captures trend.", "PPO/MACD family", "26+ bars", "close", "close of t", "continuity run", "direction", "trend-following", "multi-horizon momentum/OLS", "LOW", "MEDIUM", "LOW", "HIGH", "REJECT", "High formula and monotonic overlap with existing momentum/trend features."),
    ("MLAT-H017", "normalized_atr_14", "11 / Appendix", "355; 754-755", "MLAT-DIRECT", "Price-normalized ATR measures volatility.", "ATR divided by price", "15 bars", "OHLC", "close of t", "continuity run", "expansion", "positive", "atr_20 and existing ATR family", "LOW", "LOW", "LOW", "HIGH", "REJECT", "Existing ATR anchors already dominate this hypothesis."),
    ("MLAT-H018", "kama_gap", "Appendix", "738-740", "MLAT-DIRECT", "Adaptive smoothing may distinguish efficient trends from noise.", "KAMA gap", "adaptive", "close", "close of t", "continuity run", "direction / risk state", "unknown", "efficiency ratios and slopes", "MEDIUM", "HIGH", "MEDIUM", "MEDIUM", "DEFER", "Recursive cost and near-direct overlap with existing efficiency ratios."),
    ("MLAT-H019", "on_balance_volume", "Appendix", "753", "MLAT-DIRECT", "Cumulative signed volume may lead price.", "OBV cumulative sum", "expanding", "close, volume", "close of t", "contract/session reset unresolved", "direction", "trend-following", "signed volume proxy", "MEDIUM", "MEDIUM", "LOW", "MEDIUM", "REJECT", "Cumulative scale/reset ambiguity and overlap with existing signed-volume features."),
    ("MLAT-H020", "rolling_ar1_coefficient", "9", "290-296", "MLAT-DIRECT", "Local AR coefficient estimates serial persistence.", "Rolling OLS r_t on r_t-1", "60 returns", "close", "close of t", "continuity run", "direction / regime", "sign indicates persistence/reversion", "return_autocorrelation_15", "LOW", "MEDIUM", "LOW", "HIGH", "REJECT", "Formula-equivalent family already exists; variance ratio is the distinct selected adaptation."),
    ("MLAT-H021", "williams_r_14", "Appendix", "751", "MLAT-DIRECT", "Price location in trailing high-low range may identify overbought/oversold state.", "Williams percent range", "14 bars", "HLC", "close of t", "continuity run", "direction", "unknown", "rolling_range_position_15", "LOW", "LOW", "LOW", "HIGH", "REJECT", "Near formula-equivalent to existing rolling range position."),
    ("MLAT-H022", "intraday_seasonal_state", "4", "130-131", "MLAT-ADAPTED", "Clock time may forecast volatility/activity.", "clock indicators", "none", "timestamp", "known at t", "NY date/session", "expansion", "session dependent", "existing clock calendar family", "LOW", "LOW", "LOW", "HIGH", "REJECT", "Existing clock features are already strong frozen expansion representatives."),
    ("MLAT-H023", "cross_sectional_risk_factor", "4 / 13", "115-130; 429-460", "MLAT-DIRECT", "Cross-sectional factor exposure predicts relative returns.", "cross-sectional factor score", "multi-asset", "universe", "point in time", "universe membership", "portfolio", "relative", "none", "HIGH", "HIGH", "HIGH", "LOW", "DEFER_DIFFERENT_DATA", "Single-instrument GC discovery has no cross-section."),
    ("MLAT-H024", "news_sentiment", "14-16", "461-532", "MLAT-DIRECT", "Text sentiment or semantic content may predict futures response.", "NLP feature", "publication history", "text", "publication timestamp", "source dependent", "direction / risk", "unknown", "none", "HIGH", "HIGH", "HIGH", "LOW", "DEFER_DIFFERENT_DATA", "No point-in-time news/text source is in the current OHLCV dataset."),
    ("MLAT-H025", "pca_regime_state", "13", "429-460", "MLAT-ADAPTED", "Low-dimensional components may summarize correlated states.", "Development-fitted PCA", "feature history", "feature matrix", "close of t", "continuity inherited", "risk state", "unknown", "existing clustering/frozen set", "MEDIUM", "HIGH", "LOW", "MEDIUM", "LATER_PHASE", "Requires frozen engineered inputs and stability analysis first."),
    ("MLAT-H026", "tree_or_boosted_feature_importance", "11-12", "350-428", "MLAT-DIRECT", "Nonlinear models may rank interactions.", "model importance", "training sample", "feature matrix", "after training", "n/a", "model validation", "n/a", "none", "HIGH", "HIGH", "HIGH", "LOW", "LATER_PHASE", "Importance is not a feature and cannot prove economic value."),
    ("MLAT-H027", "deep_sequence_model", "17-21", "533-690", "MLAT-DIRECT", "Neural models may learn nonlinear temporal representations.", "NN/CNN/RNN/AE/GAN", "large sample", "various", "model dependent", "model dependent", "later modelling", "unknown", "none", "VERY HIGH", "VERY HIGH", "HIGH", "LOW", "LATER_PHASE", "Simple features and linear benchmarks have not established directional information."),
    ("MLAT-H028", "reinforcement_learning_policy", "22", "691-723", "MLAT-DIRECT", "An agent may optimize sequential actions.", "MDP policy", "environment", "state/actions/rewards", "sequential", "environment", "execution", "n/a", "none", "VERY HIGH", "VERY HIGH", "VERY HIGH", "LOW", "REJECT_GOVERNANCE", "No approved signal, reward, or validated simulator for this feature task."),
    ("MLAT-H029", "mgc_transfer_check", "1-2", "42-94", "PROJECT-ORIGINAL-EXTENSION", "A frozen GC feature may transfer to MGC.", "same frozen feature on MGC", "same as GC", "MGC OHLCV", "close of t", "MGC continuity", "transfer validation", "same expected shape", "n/a", "MEDIUM", "HIGH", "LOW", "HIGH", "LATER_PHASE", "MGC is locked until GC definitions and verdicts are frozen."),
]


SELECTED_FEATURES = [row[1] for row in HYPOTHESES if row[18] == "SELECT_V1"]


OVERLAP = [
    ("MLAT-H001", "bollinger_zscore_20", "rolling_range_position_15; normalized_ols_slope_30", "related but not formula-equivalent", "MEDIUM", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H002", "bollinger_bandwidth_20", "range_compression_ratio_5_30; atr_ratio_20_60", "same broad compression target; different close-standard-deviation formula", "HIGH", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H003", "cutler_rsi_14", "return_15m_bps; directional_persistence_15", "bounded gain/loss transform is distinct", "MEDIUM", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H004", "chaikin_money_flow_20", "close_location_value; signed_volume_proxy; volume_price_alignment_10", "interaction/aggregation is distinct but likely correlated", "HIGH", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H005", "amihud_illiquidity_60", "volume_per_tick_range; liquidity_vacuum_score_exp", "inverse activity-impact form is distinct", "HIGH", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H006", "parkinson_volatility_30", "atr_20; realized_volatility_30; current_range_over_atr", "high-low quadratic estimator is distinct", "HIGH", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H007", "rogers_satchell_volatility_30", "atr_20; realized_volatility_30", "OHLC drift-robust estimator is distinct", "HIGH", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H008", "realized_semivariance_balance_60", "directional_energy_balance_15_exp", "related signed-energy idea at a different definition/horizon", "HIGH", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H009", "bipower_jump_ratio_60", "current_range_over_atr; liquidity_vacuum_score_exp", "jump share is distinct", "MEDIUM", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H010", "variance_ratio_60_5", "return_autocorrelation_15; return_sign_change_rate_30", "same persistence family but non-equivalent variance scaling", "HIGH", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H011", "return_sign_entropy_60", "return_sign_change_rate_30; choppiness_14", "distributional sign disorder versus transition rate/path choppiness", "HIGH", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
    ("MLAT-H012", "volatility_of_volatility_60", "realized_volatility_ratio_15_60; atr_ratio_20_60", "volatility instability versus level ratio", "MEDIUM", "ELIGIBLE_PENDING_EMPIRICAL", "", "", "", ""),
]


SOURCE_TRACE = [
    ("MLAT-H001", "bollinger_zscore_20", "4; Appendix", "131-133; 740-742", "Bollinger mean/bands and examples", "Adapted standardized z-score", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H002", "bollinger_bandwidth_20", "Appendix", "740-742", "Normalized band width/squeeze", "Direct formula adaptation", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H003", "cutler_rsi_14", "4", "132", "RSI with 14-period example", "Cutler rolling convention declared by project", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H004", "chaikin_money_flow_20", "Appendix", "752-753", "Money-flow multiplier and volume", "Rolling normalized adaptation", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H005", "amihud_illiquidity_60", "20; Appendix", "656; 752", "Absolute return / dollar volume rolling measure", "Within-GC close-volume proxy", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H006", "parkinson_volatility_30", "9", "297-301", "Volatility modelling motivation and anchor comparison", "Project-original estimator; formula not claimed as book formula", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H007", "rogers_satchell_volatility_30", "9", "297-301", "Volatility modelling motivation and anchor comparison", "Project-original estimator; formula not claimed as book formula", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H008", "realized_semivariance_balance_60", "5; 9", "169-178; 297-301", "Downside risk and changing variance", "Project-original signed variation state", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H009", "bipower_jump_ratio_60", "9", "297-301", "Changing variance and residual diagnostics", "Project-original jump-state estimator", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H010", "variance_ratio_60_5", "9", "280-296", "Stationarity, AR dependence, forecast diagnostics", "Adapted local random-walk diagnostic", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H011", "return_sign_entropy_60", "6", "192", "Entropy and mutual information", "Project-original rolling sign entropy", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H012", "volatility_of_volatility_60", "9", "297-301", "Volatility clustering and forecasting", "Project-original second-order volatility state", "mlat_feature_engineering.py", "FROZEN"),
    ("MLAT-H015", "garch_variance_forecast", "9", "297-301", "ARCH/GARCH equations and diagnostics", "Independent segmented audit only", "mlat_volatility_models.py", "AUDIT_ONLY"),
]


def build_concept_registry() -> None:
    fields = list(CONCEPTS[0])
    _write_csv(DOCS / "mlat_concept_registry.csv", fields, CONCEPTS)
    lines = [
        "# MLAT Concept Registry",
        "",
        "The CSV beside this file is authoritative. Recommendations are project",
        "decisions, not claims that the book endorses the exact GC adaptation.",
        "",
        "| ID | Concept | Source | Target | Applicability | Leakage | Recommendation |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in CONCEPTS:
        lines.append(
            f"| {row['concept_id']} | {row['concept_name']} | Ch. {row['chapter']}, PDF "
            f"{row['pdf_page']} | {row['intended_target']} | {row['applicability']} | "
            f"{row['leakage_risk']} | {row['recommendation']} |"
        )
    _write(DOCS / "mlat_concept_registry.md", "\n".join(lines))


def build_formula_registry() -> None:
    headings = [
        "formula_id",
        "name",
        "mathematical_definition",
        "variables",
        "required_history",
        "causal_availability",
        "parameters",
        "scaling",
        "stationarity_assumptions",
        "boundary_requirements",
        "numerical_edge_cases",
        "book_page",
        "gc_adaptation",
        "implementation_status",
    ]
    lines = [
        "# MLAT Formulas and Definitions",
        "",
        "Book pages identify the motivating discussion. Entries explicitly marked",
        "project-original do not imply that the displayed estimator appears in the book.",
    ]
    for values in FORMULAS:
        row = dict(zip(headings, values, strict=True))
        lines.extend(
            [
                "",
                f"## {row['formula_id']} - {row['name']}",
                "",
                f"- Definition: `{row['mathematical_definition']}`",
                f"- Variables: {row['variables']}",
                f"- Required history: {row['required_history']}",
                f"- Causal availability: {row['causal_availability']}",
                f"- Parameters: {row['parameters']}",
                f"- Scaling: {row['scaling']}",
                f"- Stationarity assumptions: {row['stationarity_assumptions']}",
                f"- Boundary requirements: {row['boundary_requirements']}",
                f"- Numerical edge cases: {row['numerical_edge_cases']}",
                f"- Book trace: {row['book_page']}",
                f"- GC adaptation: {row['gc_adaptation']}",
                f"- Status: {row['implementation_status']}",
            ]
        )
    _write(DOCS / "mlat_formulas_and_definitions.md", "\n".join(lines))


def build_warnings() -> None:
    text = """# MLAT Methodological Warnings

## Book-derived warnings

| Warning | Source | MLAT consequence |
|---|---|---|
| Data quality and point-in-time integrity dominate model sophistication. | Chapters 1-3 and 23, PDF 42-114 and 724-734 | Verify physical artifacts and exclude legacy forward columns before engineering. |
| Lag and target alignment can create look-ahead bias. | Chapter 4, PDF 130-132 | Features end at completed `t`; labels and entry begin at `t+1`. |
| Technical indicators are hypotheses, not validated trading rules. | Chapter 4 and Appendix, PDF 131-152 and 735-764 | No 30/70 RSI or band-cross signal is assumed. |
| Bias-variance and repeated model selection can overfit noise. | Chapter 6, PDF 192-201 | Use a bounded frozen batch, BH correction, chronological Validation, and no broad tuning. |
| Ordinary shuffled CV is invalid for dependent financial samples. | Chapter 6, PDF 195-200 | Use chronological partitions, date evidence units, thinning, and purging/embargo for models. |
| Backtests require explicit timestamp, calendar, order, position, and cost mechanics. | Chapter 8, PDF 249-279 | Feature evidence does not authorize a strategy or Sharpe calculation. |
| ARIMA-type mean models assume stable structure; changing variance needs separate modelling. | Chapter 9, PDF 280-301 | Diagnose stationarity, volatility clustering, residuals, and forecast calibration. |
| A fitted GARCH model must leave standardized residuals close to white noise. | Chapter 9, PDF 297-301 | Record convergence, persistence, Ljung-Box/ARCH diagnostics, scaling, and anchor comparison. |
| Feature importance reflects model behavior, not causal or economic value. | Chapters 11-12, PDF 350-428 | No nonlinear model or importance plot is used to prove a feature edge. |
| Complex deep/RL systems require data, diagnostics, and valid environments. | Chapters 17-22, PDF 533-723 | These methods remain outside v1. |
| Backtest overfitting remains a central failure mode. | Chapter 23, PDF 724-734 | The exposed historical Final test is not reused for selection. |

## Project-specific additions

- GC alone defines every v1 feature, parameter, threshold, and verdict. MGC is
  inaccessible to discovery code.
- Rolling state resets on missing minutes, contract/product/instrument/segment
  changes, roll windows, invalid tradability, and low-liquidity warnings.
- Development-fitted buckets and transformations are applied unchanged to
  Validation.
- The historical Final-test partition may be counted to prove exclusion but
  cannot enter an outcome screen.
- Overlapping minute labels make rows invalid as independent evidence units.
  Daily rank IC, date-block intervals, and horizon thinning are mandatory.
- A statistically detectable direction effect below the frozen two-tick
  quintile-spread floor is economically trivial.
- Expansion and risk-state features need not predict signed returns, but must
  add information beyond `atr_20` and the frozen expansion set.
- Global clipping, scaling, quantiles, Kalman noise fits, wavelet thresholds,
  and GARCH parameters are leakage channels unless fitted under a declared
  chronological protocol.
- Missingness is information about warm-up/boundaries and is not silently
  backfilled.
- Final reports must retain negative and redundant results; feature count is
  never a completion metric.
"""
    _write(DOCS / "mlat_methodological_warnings.md", text)


def build_patterns() -> None:
    text = """# MLAT Implementation Patterns

## Adopted patterns

1. Discover the repository root from the notebook path/current parents; use
   only repository-relative artifact paths.
2. Load Parquet columns with
   `pyarrow.parquet.read_table(path, columns=...).to_pandas(ignore_metadata=True)`.
3. Add the raw table's physical row number as `source_row_id` before filtering
   to GC; map `decision_bar_id` by exact ID and timestamp equality.
4. Use a strict causal-source whitelist. The engineering function has no label
   argument and rejects names containing forward/outcome tokens.
5. Reuse the established continuity-run contract. Compute rolling arrays on
   the full GC sequence, then select eligible decision rows.
6. Use complete trailing windows and preserve null warm-up/boundary values.
7. Instantiate and validate a frozen registry before construction. Predictor
   columns in the saved matrix must equal registry order exactly.
8. Keep sensitive arithmetic in float64; validate, then save feature outputs as
   float32 where the registry permits.
9. Save to `data/processed/statistical_research/mlat_feature_research/v1`,
   reload through PyArrow, and verify IDs, order, schema, nulls, and
   deterministic sample hashes.
10. Join labels only after the feature artifact has passed reload validation.
11. Build one Development+Validation frame and raise if any other partition is
   supplied to evaluation.
12. Fit buckets on Development per session, apply unchanged to Validation, and
   use daily rank IC/date-block intervals rather than minute-row t tests.
13. Persist each result table and concise manifest under the versioned report
   root; keep the notebook as narrative/orchestration.

## Patterns rejected for v1

- Copying notebook cells or appending to the old feature registry.
- Loading all raw columns and dropping forbidden fields later.
- Row-wise Python loops across the minute table.
- Centered windows, whole-sample smoothers, backward fill, or full-series
  wavelet reconstruction.
- Joint GC/MGC thresholds or parameter fits.
- Random CV, unrestricted hyperparameter search, and model importance as proof.
- Hard-coded absolute paths, environment-specific Python assertions, or hidden
  reliance on an already-running notebook kernel.
- Calling old save helpers whose paths would overwrite historical artifacts.
"""
    _write(DOCS / "mlat_implementation_patterns.md", text)


def build_applicability() -> None:
    rows = [
        ("Lagged returns", "4:130-132", "OHLCV", "existing return ladder", "None", "direction", "REDUNDANT WITH EXISTING FEATURES", "Already represented at frozen horizons."),
        ("Bollinger location", "4:131-133; App:740-742", "close", "range/trend state", "Use causal 20-bar z-score; no trading thresholds", "direction", "APPLICABLE AFTER MODIFICATION", "Distinct continuous hypothesis; empirical overlap required."),
        ("Bollinger bandwidth", "App:740-742", "close", "compression features", "Normalize by trailing mean", "expansion", "APPLICABLE AFTER MODIFICATION", "Direct squeeze concept but likely redundant."),
        ("RSI", "4:132", "close", "momentum/persistence", "Declare Cutler rolling convention", "direction / state", "APPLICABLE AFTER MODIFICATION", "Avoid arbitrary 30/70 rule."),
        ("Kalman filter", "4:133-136", "close/returns", "trend filters", "Segmented online filter and chronological parameter fit", "state", "LATER RESEARCH PHASE", "Noise parameters and overlap are unresolved."),
        ("Wavelet denoising", "4:137-140", "returns", "none", "Would require one-sided online transform", "state", "REJECTED DUE TO LEAKAGE", "Book demonstration uses full-window decomposition/reconstruction."),
        ("Mutual information", "6:192", "features/labels", "rank IC", "Chronological resampling and multiple-test control", "screening", "LATER RESEARCH PHASE", "Estimator cost/bias adds little to the first batch."),
        ("Purging and embargo", "6:199-200", "label spans", "date blocks", "Use for any trained multivariate model", "validation", "DIRECTLY APPLICABLE", "Mandatory if modelling is authorized."),
        ("Linear/regularized model", "7:203-248", "frozen features", "ridge benchmark", "Chronological nested fit", "direction/expansion", "LATER RESEARCH PHASE", "Requires a surviving shortlist."),
        ("Event-driven backtest", "8:249-279", "signals/orders/costs", "sequential backtester", "Use only after feature and signal gates", "execution", "LATER RESEARCH PHASE", "No policy is authorized in v1."),
        ("AR/variance dependence", "9:280-296", "returns", "autocorrelation/sign change", "Use variance-ratio adaptation", "direction/state", "APPLICABLE AFTER MODIFICATION", "Distinct formula but high overlap risk."),
        ("GARCH", "9:297-301", "segmented returns", "ATR/RV and old exploratory cells", "GC-only segmented fit and diagnostics", "volatility/risk", "LATER RESEARCH PHASE", "Audit separately; not a frozen v1 predictor."),
        ("Cointegration/pairs", "9-10:301-349", "multi-asset prices", "none", "Would require a defensible instrument basket", "portfolio", "REQUIRES DIFFERENT DATA", "Single GC series cannot supply a pair."),
        ("Tree/boosting models", "11-12:350-428", "frozen features/labels", "none", "Chronological tuning after linear gate", "model", "LATER RESEARCH PHASE", "Premature complexity."),
        ("PCA/clustering", "13:429-460", "feature matrix/universe", "hierarchical feature clustering", "Use Development-only clustering for redundancy", "risk/redundancy", "APPLICABLE AFTER MODIFICATION", "No new PCA factor in v1."),
        ("Sentiment/topics/embeddings", "14-16:461-532", "point-in-time text", "none", "Obtain governed publication data", "alternative data", "REQUIRES DIFFERENT DATA", "Unavailable in OHLCV."),
        ("Deep/CNN/RNN/autoencoder", "17-20:533-662", "large model-ready corpus", "none", "Establish simpler evidence first", "model", "LATER RESEARCH PHASE", "Not justified by current results."),
        ("Amihud illiquidity", "20:656; App:752", "close/volume", "liquidity proxies", "Within-GC 60-minute notional proxy", "risk/expansion", "APPLICABLE AFTER MODIFICATION", "Distinct formula with high overlap risk."),
        ("GAN synthetic data", "21:663-690", "training sequences", "none", "Fidelity/tail validation", "data augmentation", "REJECTED DUE TO GOVERNANCE", "Could manufacture or erase rare risk behavior."),
        ("Reinforcement learning", "22:691-723", "validated environment/reward", "none", "Requires approved sequential policy", "execution", "REJECTED DUE TO GOVERNANCE", "No signal or simulator authorization."),
        ("Chaikin A/D", "App:752-753", "OHLCV", "CLV/signed-volume features", "Use rolling normalized money flow, reset boundaries", "direction/state", "APPLICABLE AFTER MODIFICATION", "Cumulative raw A/D has reset/scale problems."),
        ("ATR/NATR", "App:754-755", "OHLC", "atr_20 anchor", "None", "volatility", "REDUNDANT WITH EXISTING FEATURES", "Use frozen anchor rather than add another ATR."),
        ("Quotes/trades/order book/MBO", "2:59-94", "quotes/trades/depth/MBO", "none", "Acquire different Databento schemas", "execution/microstructure", "REQUIRES DIFFERENT DATA", "Not present in one-minute OHLCV."),
        ("Macro/fundamental factors", "2-4:59-152", "point-in-time macro/fundamentals", "none", "Build release/vintage-aware sources", "direction/risk", "REQUIRES DIFFERENT DATA", "Unavailable and often equity-specific."),
        ("Satellite/imagery", "3/18:95-114;569-606", "images", "none", "Define economically linked imagery", "alternative data", "IMPRACTICAL WITH CURRENT OHLCV", "No relevant image source."),
    ]
    lines = [
        "# MLAT Project Applicability Matrix",
        "",
        "| Idea | Source (chapter:PDF) | Required data | Existing equivalent | Adaptation | Target | Classification | Reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    _write(DOCS / "mlat_project_applicability_matrix.md", "\n".join(lines))


def build_hypotheses() -> None:
    headers = [
        "hypothesis_id",
        "proposed_feature_name",
        "source_chapter",
        "source_pdf_page",
        "source_type",
        "economic_rationale",
        "mathematical_definition",
        "required_lookback",
        "required_input_data",
        "decision_time_availability",
        "reset_boundaries",
        "expected_target",
        "expected_sign_or_shape",
        "existing_feature_overlap",
        "leakage_risk",
        "computational_cost",
        "numerical_risk",
        "interpretability",
        "recommendation",
        "rejection_reason",
    ]
    lines = [
        "# MLAT Feature Hypothesis Catalog",
        "",
        "This catalog was frozen before any MLAT feature/outcome relationship was",
        "calculated. Project-original extensions cite the motivating book discussion",
        "without claiming their exact estimator appears in the book.",
        "",
        "| ID | Feature | Source | Type | Target | Overlap | Leakage | Cost | Decision |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for values in HYPOTHESES:
        row = dict(zip(headers, values, strict=True))
        lines.append(
            f"| {row['hypothesis_id']} | {row['proposed_feature_name']} | Ch. "
            f"{row['source_chapter']}, PDF {row['source_pdf_page']} | {row['source_type']} | "
            f"{row['expected_target']} | {row['existing_feature_overlap']} | "
            f"{row['leakage_risk']} | {row['computational_cost']} | {row['recommendation']} |"
        )
        if row["rejection_reason"]:
            lines.append(f"|  | Rejection/defer reason |  |  |  |  |  |  | {row['rejection_reason']} |")
    lines.extend(["", "## Full definitions", ""])
    for values in HYPOTHESES:
        row = dict(zip(headers, values, strict=True))
        lines.extend(
            [
                f"### {row['hypothesis_id']} - `{row['proposed_feature_name']}`",
                "",
                f"- Rationale: {row['economic_rationale']}",
                f"- Definition: {row['mathematical_definition']}",
                f"- History/input: {row['required_lookback']}; {row['required_input_data']}",
                f"- Availability/reset: {row['decision_time_availability']}; {row['reset_boundaries']}",
                f"- Target and expected shape: {row['expected_target']}; {row['expected_sign_or_shape']}",
                f"- Existing overlap: {row['existing_feature_overlap']}",
                f"- Risks: leakage {row['leakage_risk']}; computation {row['computational_cost']}; numerical {row['numerical_risk']}",
                f"- Decision: {row['recommendation']}"
                + (f" - {row['rejection_reason']}" if row["rejection_reason"] else ""),
                "",
            ]
        )
    _write(DOCS / "mlat_feature_hypothesis_catalog.md", "\n".join(lines))


def build_overlap_and_traceability() -> None:
    _write_csv(
        DOCS / "mlat_existing_feature_overlap_audit.csv",
        [
            "hypothesis_id",
            "feature_name",
            "nearest_existing_features",
            "formula_overlap_assessment",
            "preimplementation_redundancy_risk",
            "preimplementation_decision",
            "observed_nearest_feature",
            "observed_max_abs_spearman_development",
            "observed_exact_duplicate",
            "final_eligibility",
        ],
        [
            dict(
                zip(
                    [
                        "hypothesis_id",
                        "feature_name",
                        "nearest_existing_features",
                        "formula_overlap_assessment",
                        "preimplementation_redundancy_risk",
                        "preimplementation_decision",
                        "observed_nearest_feature",
                        "observed_max_abs_spearman_development",
                        "observed_exact_duplicate",
                        "final_eligibility",
                    ],
                    row,
                    strict=True,
                )
            )
            for row in OVERLAP
        ],
    )
    _write_csv(
        DOCS / "mlat_source_traceability.csv",
        [
            "hypothesis_id",
            "feature_or_method",
            "source_chapter",
            "source_pdf_page",
            "book_concept",
            "project_adaptation",
            "implementation_target",
            "status",
        ],
        [
            dict(
                zip(
                    [
                        "hypothesis_id",
                        "feature_or_method",
                        "source_chapter",
                        "source_pdf_page",
                        "book_concept",
                        "project_adaptation",
                        "implementation_target",
                        "status",
                    ],
                    row,
                    strict=True,
                )
            )
            for row in SOURCE_TRACE
        ],
    )


def build_batch() -> None:
    scores = {
        "bollinger_zscore_20": (5, 4, 3, 5, 5, 5, 4, 3, 5, 5, 5, 4),
        "bollinger_bandwidth_20": (5, 4, 3, 5, 5, 5, 5, 2, 5, 5, 5, 4),
        "cutler_rsi_14": (4, 3, 3, 5, 5, 5, 3, 3, 5, 5, 5, 4),
        "chaikin_money_flow_20": (5, 4, 4, 5, 5, 5, 4, 2, 5, 4, 5, 4),
        "amihud_illiquidity_60": (4, 4, 4, 5, 5, 4, 5, 2, 5, 4, 5, 4),
        "parkinson_volatility_30": (3, 5, 5, 5, 5, 5, 5, 2, 5, 4, 5, 4),
        "rogers_satchell_volatility_30": (3, 5, 5, 5, 5, 4, 5, 2, 5, 4, 5, 4),
        "realized_semivariance_balance_60": (3, 5, 5, 5, 5, 5, 4, 3, 5, 5, 5, 4),
        "bipower_jump_ratio_60": (3, 5, 5, 5, 5, 4, 5, 3, 5, 4, 4, 4),
        "variance_ratio_60_5": (4, 5, 4, 5, 5, 5, 4, 2, 5, 4, 5, 4),
        "return_sign_entropy_60": (3, 4, 5, 5, 5, 5, 4, 2, 5, 5, 5, 4),
        "volatility_of_volatility_60": (3, 5, 5, 5, 5, 5, 5, 3, 5, 4, 5, 4),
    }
    source_type = {row[1]: row[4] for row in HYPOTHESES}
    lines = [
        "# Frozen MLAT Feature Batch v1",
        "",
        "Frozen before loading MLAT outcomes. Hard vetoes are future/entry-bar use,",
        "unavailable data, invalid boundary behavior, MGC dependence, Final-test",
        "dependence, or an exact existing formula.",
        "",
        "Each score is 1 (weak) to 5 (strong). For redundancy, leakage, compute, and",
        "overfitting, 5 means lower risk. The selection floor is 44/60 with no",
        "hard veto. Scores prioritize a coherent batch rather than optimizing a",
        "retrospective result.",
        "",
        "| Feature | Type | Book | Rationale | Novelty | Causal | Data | Interpret | Target | Redundancy | Leakage | Numeric | Compute | Overfit | Total |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in SELECTED_FEATURES:
        values = scores[feature]
        lines.append(
            f"| `{feature}` | {source_type[feature]} | "
            + " | ".join(str(value) for value in values)
            + f" | {sum(values)} |"
        )
    lines.extend(
        [
            "",
            "## Frozen membership",
            "",
            *[f"{index}. `{feature}`" for index, feature in enumerate(SELECTED_FEATURES, 1)],
            "",
            "## Explicit exclusions",
            "",
            "- GARCH is an audit-only method, not a v1 matrix feature. The prior",
            "  implementation fails GC isolation and boundary governance.",
            "- Kalman/KAMA are deferred because their fitted/recursive state and overlap",
            "  are not justified for the first batch.",
            "- Wavelet denoising is rejected until a genuinely one-sided implementation",
            "  is defined.",
            "- PPO/MACD, NATR, Williams %R, raw AR(1), OBV, and clock features are",
            "  rejected as redundant or reset-ambiguous.",
            "- Cross-sectional, text, quote/trade/order-book, macro, image, deep, GAN,",
            "  and reinforcement-learning methods require a different phase or data.",
            "",
            "No membership or parameter may change after Validation results are viewed.",
        ]
    )
    _write(DOCS / "mlat_feature_batch_v1.md", "\n".join(lines))


def build_contract() -> None:
    text = """# MLAT Feature Research Contract v1

Status: FROZEN BEFORE MLAT OUTCOME EVALUATION

## Scope

- GC only; MGC is inaccessible to feature construction and evaluation.
- Features use information available at or before the close of decision bar
  `t`; theoretical entry is the open of `t+1`.
- London and New York are reported separately.
- Six fixed horizons are 5, 15, 30, 60, 120, and 180 minutes.
- Development and Validation only. Historical Final-test outcomes are excluded.
- No signal construction, PnL backtest, or Sharpe is authorized.

## Engineering contract

- Predictor names and order equal the frozen registry exactly.
- Rolling calculations use full chronological GC history and reset at every
  missing-minute, product, contract, instrument, segment, roll, tradability, or
  liquidity boundary.
- No forward, MFE, MAE, future range/volatility, direction, expansion, target,
  stop, POI, or entry-bar OHLCV may enter engineering.
- Complete windows are required. No backward fill or cross-boundary fill.
- Float64 calculations precede validated float32 persistence.
- Feature artifact save/reload must preserve IDs, order, dtypes, timestamps,
  registry alignment, null counts, and deterministic sample hash.

## Evaluation frame

- Evaluation seed: `20260723`.
- Date-block bootstrap: 2,000 replicates, 95% percentile interval.
- Same complete sample is used across the six horizons within each target
  family; ATR-normalized targets require valid decision ATR.
- Targets:
  - direction: `forward_return_{h}_atr` with tick spread reported;
  - expansion: `future_range_{h}_atr`;
  - volatility: `future_realized_volatility_{h}_bps`;
  - path risk: `max(mae_long_{h}_atr, mae_short_{h}_atr)`.
- Primary evidence is the mean per-New-York-date Spearman rank IC.
- Each daily IC requires at least 10 observations.
- Development-fitted quintiles are session-specific and applied unchanged to
  Validation.
- Benjamini-Hochberg q-values are computed within Development
  target-family x session screens across the full batch and all horizons.
- Horizon-thinned sensitivity keeps chronologically spaced observations within
  date/session; sign disagreement blocks advancement.

## Minimum evidence

- Development q-value <= 0.10.
- At least 400 Development and 150 Validation trading dates.
- At least 10,000 Development and 4,000 Validation finite observations.
- Validation IC sign agreement and retention >= 25% of Development magnitude.
- Absolute Development quintile monotonicity >= 0.8.
- Direction also requires absolute top-minus-bottom spread >= 2 GC ticks in
  both Development and Validation.
- An exact duplicate or Development absolute Spearman >= 0.995 with an
  existing feature is `REDUNDANT_WITH_EXISTING`.
- Incremental screen at 60 and 180 minutes:
  - beyond `atr_20`: Development absolute partial daily IC >= 0.02;
  - beyond the frozen 15-feature set: Development absolute partial daily IC
    >= 0.01;
  - both require Validation sign agreement and >=25% retention.
- Coverage exceeding 5% missingness is `FAILED_VALIDATION` unless a stricter
  feature-specific registry policy is declared before outcomes.

## Verdict mapping

- `ADVANCE_DIRECTIONAL`: every direction and incremental/economic gate passes.
- `ADVANCE_EXPANSION`: an expansion or volatility target passes and adds
  incremental information beyond existing expansion anchors.
- `ADVANCE_RISK_STATE`: a path-risk/volatility state passes and is not merely a
  level replica.
- `REDUNDANT_WITH_EXISTING`: valid feature but overlap veto fires.
- `ECONOMICALLY_TRIVIAL`: statistically confirmed direction below the tick floor.
- `WEAK_OR_UNSTABLE`: Development evidence does not retain in Validation,
  sessions, years, or thinning.
- `FAILED_VALIDATION`: engineering, coverage, numerical, boundary, or
  persistence gate fails.
- `NO_EVIDENCE`: no predeclared screen passes.
- Rejected/deferred catalog entries retain their explicit
  `REJECTED_LEAKAGE`, `REJECTED_IMPLEMENTATION`, or `DEFER_DIFFERENT_DATA`
  reason.

## GARCH audit protocol

- Audit GC only; never pool an MGC value or threshold.
- Returns reset at the established continuity boundary.
- Parameters fit only on GC history ending 2022-12-31.
- 2023 is a Development audit period; 2024 is Validation.
- Forecast initialization is chronological and segment-specific.
- Record optimizer result, persistence, standardized-residual and squared-
  residual diagnostics, ARCH effects, calibration, and simple-anchor metrics.
- Reject advancement on optimizer failure, persistence >= 0.999, invalid
  parameters, material residual structure, or no Validation improvement over
  the simple frozen volatility anchor.
- The model is audit-only in v1, even if diagnostics pass; adding it to a
  registry requires a new frozen experiment.

## Multivariate authorization gate

At least three non-redundant MLAT features must survive univariate, stability,
thinning, and full-anchor incremental gates. If authorized, the next experiment
starts with a chronological regularized linear benchmark. No nonlinear model or
hyperparameter sweep is authorized here.
"""
    _write(DOCS / "mlat_feature_research_contract_v1.md", text)


def build_coverage_log() -> None:
    validations = {
        f"chapter_{number:02d}.md": _summary_validation(
            CHAPTERS / f"chapter_{number:02d}.md", start, end
        )
        for number, start, end in CHAPTER_RANGES
    }
    appendix = "appendix_alpha_factor_library.md"
    validations[appendix] = _summary_validation(
        CHAPTERS / appendix, *APPENDIX_RANGE
    )
    lines = [
        "# MLAT Book Coverage Log",
        "",
        "- PDF pages accounted for in ingestion manifest: 1-858 with no gaps.",
        "- Substantive preface/chapter/appendix range: 30-764.",
        "- References and index range: 765-858.",
        "- Text extraction: complete for all 858 physical pages.",
        "- Layout-sensitive inspection includes equations, diagrams, workflows,",
        "  diagnostics, and factor tables recorded in the ingestion manifest.",
        "",
        "| Unit | PDF range | Summary | Status |",
        "|---|---:|---|---|",
    ]
    for number, start, end in CHAPTER_RANGES:
        name = f"chapter_{number:02d}.md"
        passed, _ = validations[name]
        status = "COMPLETE" if passed else "IN_PROGRESS"
        lines.append(
            f"| Chapter {number} | {start}-{end} | `chapter_summaries/{name}` | {status} |"
        )
    appendix_passed, _ = validations[appendix]
    lines.append(
        f"| Appendix | 735-764 | `chapter_summaries/{appendix}` | "
        f"{'COMPLETE' if appendix_passed else 'IN_PROGRESS'} |"
    )
    failed_issues = {
        name: issues for name, (passed, issues) in validations.items() if not passed
    }
    if failed_issues:
        lines.extend(["", "## Outstanding summary validation issues", ""])
        for name, issues in sorted(failed_issues.items()):
            lines.append(f"- `{name}`: {'; '.join(issues)}")
    lines.extend(
        [
            "",
            "## Extraction uncertainty",
            "",
            "Text extraction is high confidence for prose and code. Equations whose",
            "symbols were ambiguous in extracted text were resolved from rendered",
            "pages or are explicitly marked unresolved in the chapter/formula entry.",
            "No equation was reconstructed from memory and presented as a book quote.",
        ]
    )
    _write(DOCS / "mlat_book_coverage_log.md", "\n".join(lines))


def build_stage1_report() -> None:
    validations = [
        _summary_validation(CHAPTERS / f"chapter_{number:02d}.md", start, end)
        for number, start, end in CHAPTER_RANGES
    ]
    validations.append(
        _summary_validation(
            CHAPTERS / "appendix_alpha_factor_library.md", *APPENDIX_RANGE
        )
    )
    manifest_passed, manifest_issues = _manifest_validation()
    artifacts_passed, artifact_issues = _stage1_artifact_validation()
    complete = (
        all(passed for passed, _ in validations)
        and manifest_passed
        and artifacts_passed
    )
    batch = DOCS / "mlat_feature_batch_v1.md"
    contract = DOCS / "mlat_feature_research_contract_v1.md"
    batch_hash = hashlib.sha256(batch.read_bytes()).hexdigest() if batch.exists() else "MISSING"
    contract_hash = hashlib.sha256(contract.read_bytes()).hexdigest() if contract.exists() else "MISSING"
    status = "COMPLETE" if complete else "IN PROGRESS - CHAPTER SUMMARIES PENDING"
    text = f"""# MLAT Stage 1 Completion Report

Status: {status}

## Coverage

- Physical PDF pages: 858.
- Page-accounting manifest: pages 1-858, no intended gaps or overlap.
- Full substantive preface/chapter/appendix coverage: pages 30-764.
- Chapter summaries passing structural/page-reference QA: {sum(passed for passed, _ in validations)} of 24
  (23 chapters plus Appendix).
- Front matter, references, and index are recorded in the ingestion manifest.
- Ingestion-manifest validation: {'PASS' if manifest_passed else 'FAIL'}.
- Required durable-artifact validation: {'PASS' if artifacts_passed else 'FAIL'}.

## Frozen research artifacts

- Concept registry: {len(CONCEPTS)} concepts.
- Formula registry: {len(FORMULAS)} formulas/definitions.
- Hypothesis catalog: {len(HYPOTHESES)} hypotheses.
- Frozen implementation batch: {len(SELECTED_FEATURES)} predictors.
- Batch SHA-256: `{batch_hash}`.
- Evaluation-contract SHA-256: `{contract_hash}`.

The frozen batch and contract were generated by this outcome-free Stage 1
script. The script does not load forward labels or MLAT feature/outcome results.

## Completion controls

- Every summary must contain all required research, warning, adaptation,
  implementation, page-map, and ambiguity sections.
- Every summary must state its assigned physical page range and contain exact
  PDF page citations.
- Every chapter/Appendix manifest row must be `COMPLETE`, name at least one
  visually inspected page, and keep those pages inside its assigned range.
- Manifest issues: {('; '.join(manifest_issues)) if manifest_issues else 'none'}.
- Durable-artifact issues: {('; '.join(artifact_issues)) if artifact_issues else 'none'}.

## Completion condition

Stage 2 source/notebook implementation may begin only when this report says
`COMPLETE` and the coverage log marks every chapter and Appendix complete.
"""
    _write(DOCS / "mlat_stage1_completion_report.md", text)


def main(*, verify_only: bool = False, rebuild_frozen: bool = False) -> bool:
    """Build pre-outcome docs, then become verification-only after outcomes exist."""

    outcome_locked = _outcome_artifacts_exist()
    if rebuild_frozen and outcome_locked:
        raise RuntimeError(
            "Refusing to rebuild the frozen batch/contract after outcome artifacts exist. "
            "Create a new versioned experiment instead."
        )
    if verify_only:
        summary_results = [
            _summary_validation(
                CHAPTERS / f"chapter_{number:02d}.md", start, end
            )
            for number, start, end in CHAPTER_RANGES
        ]
        summary_results.append(
            _summary_validation(
                CHAPTERS / "appendix_alpha_factor_library.md", *APPENDIX_RANGE
            )
        )
        manifest_passed, _ = _manifest_validation()
        artifacts_passed, _ = _stage1_artifact_validation()
        return (
            all(passed for passed, _ in summary_results)
            and manifest_passed
            and artifacts_passed
        )
    should_rebuild = not verify_only and (rebuild_frozen or not outcome_locked)
    if should_rebuild:
        build_concept_registry()
        build_formula_registry()
        build_warnings()
        build_patterns()
        build_applicability()
        build_hypotheses()
        build_overlap_and_traceability()
        build_batch()
        build_contract()
    build_coverage_log()
    build_stage1_report()
    summary_results = [
        _summary_validation(CHAPTERS / f"chapter_{number:02d}.md", start, end)
        for number, start, end in CHAPTER_RANGES
    ]
    summary_results.append(
        _summary_validation(
            CHAPTERS / "appendix_alpha_factor_library.md", *APPENDIX_RANGE
        )
    )
    manifest_passed, _ = _manifest_validation()
    artifacts_passed, _ = _stage1_artifact_validation()
    return (
        all(passed for passed, _ in summary_results)
        and manifest_passed
        and artifacts_passed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate persisted frozen artifacts without rewriting them.",
    )
    parser.add_argument(
        "--rebuild-frozen",
        action="store_true",
        help="Explicit pre-outcome rebuild; refused once outcome artifacts exist.",
    )
    args = parser.parse_args()
    passed = main(verify_only=args.verify_only, rebuild_frozen=args.rebuild_frozen)
    print(f"Stage 1 verification: {'PASS' if passed else 'FAIL'}")
    if args.verify_only and not passed:
        raise SystemExit(1)
