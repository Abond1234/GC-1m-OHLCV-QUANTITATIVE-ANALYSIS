"""Run the GC-to-MGC transfer validation (FR-09) on the trusted research table.

Usage: python scripts/run_mgc_transfer_validation.py
Writes parquet outputs under data/processed/execution/ (excluded from Git) and
tracked CSVs under reports/execution/tables/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.execution.mgc_transfer import (  # noqa: E402
    MgcTransferConfig,
    build_minute_alignment,
    build_transfer_result,
    load_poi_decision_bars,
    load_product_minutes,
    save_transfer_outputs,
)


def main() -> int:
    pd.set_option("display.width", 200)
    started = perf_counter()
    config = MgcTransferConfig()
    gc, mgc = load_product_minutes(
        PROJECT_ROOT / "data" / "processed" / "research_bars_gc_mgc_1m.parquet"
    )
    decisions = load_poi_decision_bars(PROJECT_ROOT)
    aligned = build_minute_alignment(gc, mgc, config)
    result = build_transfer_result(aligned, decisions, config)

    print(f"GC minutes: {len(gc):,} | MGC minutes: {len(mgc):,} | decisions: {len(decisions):,}")
    print("\n=== coverage by year ===")
    print(result.coverage_by_year.round(4).to_string(index=False))
    print("\n=== basis by year ===")
    print(result.basis_by_year.round(4).to_string(index=False))
    print("\n=== basis by session ===")
    print(result.basis_by_session.round(4).to_string(index=False))
    print("\n=== decision bars by year ===")
    print(result.decision_by_year.round(4).to_string(index=False))
    print("\n=== decision-bar summary ===")
    print(result.decision_summary.round(4).to_string())
    print("\n=== provisional G5 threshold checks ===")
    print(result.threshold_checks.round(4).to_string(index=False))
    print("\n=== validation checks ===")
    print(result.validation_checks.to_string(index=False))
    print(f"\nVERDICT: {result.verdict}")

    save_report = save_transfer_outputs(result, project_root=PROJECT_ROOT)
    print("\n=== save report ===")
    print(save_report.to_string(index=False))
    print(f"\nruntime: {perf_counter() - started:.1f}s")
    if not bool(result.validation_checks["passed"].all()):
        print("VALIDATION CHECKS FAILED")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
