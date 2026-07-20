from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.poi_refinement_visuals import (
    HAND_SIGNAL_COLUMNS,
    build_hand_labelled_regression_visuals,
    build_random_stratified_refinement_audit,
)
from src.features.poi_selection_refinement import (
    HAND_LABELLED_REGRESSION_CASES,
    Section6CConfig,
    build_section6c_refined_tables,
    save_section6c_tables,
)
from src.features.poi_signal_features import SECTION6_REQUIRED_COLUMNS

LEGACY_OUTPUTS = (
    "section6_poi_table_gc.parquet",
    "section6_retest_table_gc.parquet",
    "section6_candidate_trade_table_gc.parquet",
    "section6_signal_frame_gc.parquet",
    "section6b_structural_poi_table_gc.parquet",
    "section6b_structural_retest_table_gc.parquet",
    "section6b_structural_candidate_trade_table_gc.parquet",
    "section6b_structural_signal_frame_gc.parquet",
    "section7_event_study_summary_gc.parquet",
    "section7_candidate_signal_ranking_gc.parquet",
)


def _signatures(processed: Path) -> dict[str, tuple[int, int]]:
    return {
        name: (path.stat().st_size, path.stat().st_mtime_ns)
        for name in LEGACY_OUTPUTS
        if (path := processed / name).exists()
    }


def _legacy_counts(processed: Path) -> dict[str, int]:
    mapping = {
        "version_a_poi_count": "section6_poi_table_gc.parquet",
        "version_a_retest_count": "section6_retest_table_gc.parquet",
        "version_a_candidate_count": "section6_candidate_trade_table_gc.parquet",
        "version_a_signal_count": "section6_signal_frame_gc.parquet",
        "version_b_poi_count": "section6b_structural_poi_table_gc.parquet",
        "version_b_retest_count": "section6b_structural_retest_table_gc.parquet",
        "version_b_candidate_count": "section6b_structural_candidate_trade_table_gc.parquet",
        "version_b_signal_count": "section6b_structural_signal_frame_gc.parquet",
    }
    return {
        metric: pq.ParquetFile(processed / filename).metadata.num_rows
        for metric, filename in mapping.items()
    }


def main() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    figure_dir = PROJECT_ROOT / "reports" / "figures" / "section6c_poi_refinement"
    before = _signatures(processed)
    print("[section6c] loading GC research bars", flush=True)
    research_bars = pd.read_parquet(
        processed / "research_bars_gc_mgc_1m.parquet",
        columns=SECTION6_REQUIRED_COLUMNS,
        filters=[("product", "==", "GC")],
    )
    print(f"[section6c] research_bars shape={research_bars.shape}", flush=True)

    cfg = Section6CConfig()
    print("[section6c] building refined tables", flush=True)
    tables = build_section6c_refined_tables(
        research_bars,
        cfg,
        legacy_counts=_legacy_counts(processed),
    )
    print("[section6c] saving non-destructive outputs", flush=True)
    paths = save_section6c_tables(tables, processed)

    signal_ids = [case["signal_id"] for case in HAND_LABELLED_REGRESSION_CASES]
    legacy_signals = pd.read_parquet(
        processed / "section6b_structural_signal_frame_gc.parquet",
        columns=HAND_SIGNAL_COLUMNS,
        filters=[("signal_id", "in", signal_ids)],
    )
    print("[section6c] generating 19-case and stratified visual regression", flush=True)
    manifest, abc = build_hand_labelled_regression_visuals(
        audit=tables["refined_poi_audit"],
        legacy_signal_frame=legacy_signals,
        research_bars=tables["section6_feature_frame"],
        output_dir=figure_dir,
    )
    random_manifest = build_random_stratified_refinement_audit(
        audit=tables["refined_poi_audit"],
        research_bars=tables["section6_feature_frame"],
        output_dir=figure_dir,
    )

    after = _signatures(processed)
    legacy_untouched = before == after
    validation = tables["section6c_validation"].copy()
    validation["legacy_version_a_b_and_section7_files_untouched"] = legacy_untouched
    print("[section6c] validation", flush=True)
    print(validation.to_string(), flush=True)
    print("[section6c] timings", tables["section6c_timings"], flush=True)
    print("[section6c] output_paths", {k: str(v) for k, v in paths.items()}, flush=True)
    print(
        "[section6c] hand_labelled_result",
        manifest["pass_fail"].value_counts(dropna=False).to_dict(),
        flush=True,
    )
    print(f"[section6c] abc_rows={len(abc)} stratified_rows={len(random_manifest)}", flush=True)
    if not validation.astype(bool).all():
        failed = validation.loc[~validation.astype(bool)]
        raise RuntimeError(f"Section 6C validation failed:\n{failed.to_string()}")


if __name__ == "__main__":
    main()
