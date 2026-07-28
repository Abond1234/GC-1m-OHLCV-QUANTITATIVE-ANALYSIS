"""Run the independent, clean-room verification of the Section 11 backtest.

Loads the recorded backtest artifacts and the raw bars, re-derives every trade
through the independent verifier, and prints the reconciliation, fill-fidelity,
and cross-artifact audit tables.  Generated tables are written under
``data/processed`` and ``reports/**/tables`` and stay outside Git.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_research.backtest_verification import (
    load_gc_bars,
    run_verification,
    save_verification_outputs,
)

SR = PROJECT_ROOT / "data" / "processed" / "statistical_research"


def main() -> None:
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.width", 200)

    candidates = pd.read_parquet(SR / "signal_candidates_gc.parquet").set_index("observation_id")
    bars = load_gc_bars(PROJECT_ROOT / "data" / "processed" / "research_bars_gc_mgc_1m.parquet")
    trade_log = pd.read_parquet(SR / "backtest_trade_log_gc.parquet")
    performance = pd.read_parquet(SR / "backtest_performance_gc.parquet")
    gate_thresholds = pd.read_parquet(SR / "signal_gate_thresholds_gc.parquet")
    import pyarrow.parquet as pq

    # forward_labels_gc carries pandas metadata (a dictionary-encoded column)
    # that pandas 3.0 cannot map on read; take the pyarrow path and drop it.
    forward_labels = pq.read_table(
        SR / "forward_labels_gc.parquet",
        columns=["observation_id", "entry_price", "decision_atr_20m"],
    ).to_pandas(ignore_metadata=True)

    result = run_verification(
        candidates,
        bars,
        trade_log,
        performance,
        gate_thresholds=gate_thresholds,
        forward_labels=forward_labels,
    )

    print("\n================ VERIFICATION CHECKS ================")
    print(result.checks.to_string(index=False))
    print("\n================ TRADE RECONCILIATION ==============")
    print(result.reconciliation.to_string(index=False))
    print("\n================ ENTRY-FILL FIDELITY ===============")
    print(result.entry_fidelity.to_string())
    print("\n================ EXIT-FILL FIDELITY ================")
    print(result.fill_fidelity.to_string(index=False))
    print("\n================ PERFORMANCE RECONCILIATION ========")
    print(result.performance_reconciliation.to_string(index=False))
    print("\n================ GATE CONSISTENCY ==================")
    print(result.gate_consistency.to_string())
    print("\n================ CROSS-ARTIFACT TRIANGULATION ======")
    print(result.cross_artifact.to_string())
    print("\n================ SUMMARY ===========================")
    print(result.summary.to_string())

    saved = save_verification_outputs(result, project_root=PROJECT_ROOT)
    print("\n================ SAVED OUTPUTS =====================")
    print(saved.to_string(index=False))

    if not result.all_passed:
        raise SystemExit("VERIFICATION FAILED - see the checks table above")
    print("\nALL VERIFICATION CHECKS PASSED")


if __name__ == "__main__":
    main()
