"""LEGACY BRIDGE: refined event-study baseline, superseded by run_section7_poi_context_research.py.

Kept for historical reproducibility only. The full population needs roughly 4 GB
of free memory and this legacy path has no chunked fallback.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.poi_event_study import SECTION7_BAR_REQUIRED_COLUMNS, Section7Config
from src.features.poi_event_study_refined import (
    build_section7r_event_study,
    save_section7r_outputs,
)


def main() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    print("[section7r] loading refined Section 6C signals", flush=True)
    signals = pd.read_parquet(processed / "section6c_refined_signal_frame_gc.parquet")
    print(f"[section7r] signals shape={signals.shape}", flush=True)
    print("[section7r] loading GC research bars", flush=True)
    bars = pd.read_parquet(
        processed / "research_bars_gc_mgc_1m.parquet",
        columns=SECTION7_BAR_REQUIRED_COLUMNS,
        filters=[("product", "==", "GC")],
    )
    print(f"[section7r] bars shape={bars.shape}", flush=True)

    outputs = build_section7r_event_study(signals, bars, Section7Config())
    paths = save_section7r_outputs(outputs, processed)
    validation = outputs["section7r_validation"]
    boolean_checks = validation.loc[
        [isinstance(value, (bool, np.bool_)) for value in validation.to_list()]
    ]
    print("[section7r] validation", flush=True)
    print(validation.to_string(), flush=True)
    print("[section7r] timings", outputs["section7r_timings"], flush=True)
    print("[section7r] output_paths", {k: str(v) for k, v in paths.items()}, flush=True)
    print(
        "[section7r] shapes",
        {
            "summary": outputs["section7r_event_study_summary"].shape,
            "ranking": outputs["section7r_candidate_signal_ranking"].shape,
            "threshold": outputs["section7r_fvg_threshold_comparison"].shape,
        },
        flush=True,
    )
    if not boolean_checks.astype(bool).all():
        failed = boolean_checks.loc[~boolean_checks.astype(bool)]
        raise RuntimeError(f"Section 7R validation failed:\n{failed.to_string()}")


if __name__ == "__main__":
    main()
