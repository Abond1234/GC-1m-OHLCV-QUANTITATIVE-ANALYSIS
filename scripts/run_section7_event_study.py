"""LEGACY: original structural event study, superseded by run_section7_poi_context_research.py.

Kept for historical reproducibility only. The full population needs roughly 4 GB
of free memory and this legacy path has no chunked fallback.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.poi_event_study import (
    SECTION7_BAR_REQUIRED_COLUMNS,
    Section7Config,
    build_section7_event_study,
    save_section7_outputs,
)


def main() -> None:
    processed = Path("data") / "processed"
    start = time.perf_counter()

    print("[section7] loading signal frame", flush=True)
    signals = pd.read_parquet(processed / "section6b_structural_signal_frame_gc.parquet")
    print(f"[section7] signals shape={signals.shape}", flush=True)

    print("[section7] loading research bars", flush=True)
    bars = pd.read_parquet(
        processed / "research_bars_gc_mgc_1m.parquet",
        columns=SECTION7_BAR_REQUIRED_COLUMNS,
    )
    print(f"[section7] bars shape={bars.shape}", flush=True)

    print("[section7] building event study", flush=True)
    outputs = build_section7_event_study(signals, bars, Section7Config())

    print("[section7] saving outputs", flush=True)
    paths = save_section7_outputs(outputs, processed)
    runtime = time.perf_counter() - start

    print(f"[section7] runtime_seconds={runtime:.2f}", flush=True)
    print("[section7] validation", flush=True)
    print(outputs["section7_validation"].to_string(), flush=True)
    print("[section7] horizon_quality", flush=True)
    print(outputs["section7_horizon_quality"].to_string(index=False), flush=True)
    print(
        "[section7] summary_shape",
        outputs["section7_event_study_summary"].shape,
        flush=True,
    )
    print(
        "[section7] ranking_shape",
        outputs["section7_candidate_signal_ranking"].shape,
        flush=True,
    )
    print("[section7] output_paths", {k: str(v) for k, v in paths.items()}, flush=True)

    ranking = outputs["section7_candidate_signal_ranking"]
    cols = [
        "rank",
        "rank_score",
        "group_key",
        "valid_horizon_count",
        "min_valid_count",
        "mean_r_cap_5_avg",
        "median_r_avg",
        "q10_r_avg",
        "positive_r_rate_avg",
        "hit_pos_3r_rate_avg",
        "hit_pos_5r_rate_avg",
        "hit_neg_1r_rate_avg",
    ]
    cols = [col for col in cols if col in ranking.columns]
    print("[section7] top_ranked_groups", flush=True)
    print(ranking.loc[:, cols].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
