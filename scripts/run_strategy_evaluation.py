"""Run the anti-overfitting evaluation over the strategy-lab trade log.

Loads the strategy-lab trade log and applies the bootstrap, Benjamini-Hochberg,
deflated-Sharpe, and PBO/CSCV corrections, then prints the per-strategy verdicts
and the overfitting probability. Generated tables stay outside Git.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_research.strategy_evaluation import (
    evaluate_strategies,
    save_strategy_evaluation_outputs,
)

SR = PROJECT_ROOT / "data" / "processed" / "statistical_research"


def main() -> None:
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 240)

    trade_log = pd.read_parquet(SR / "strategy_lab_trade_log_gc.parquet")
    result = evaluate_strategies(trade_log)

    cols = [
        "strategy",
        "family",
        "dev_mean_net_r",
        "dev_bh_q",
        "val_mean_net_r",
        "val_ci_low",
        "val_ci_high",
        "combined_daily_sharpe",
        "deflated_sharpe",
        "verdict",
    ]
    stats = result.strategy_stats[cols].sort_values("combined_daily_sharpe", ascending=False)
    print("\n================ PER-STRATEGY EVALUATION ================")
    print(stats.to_string(index=False))
    print("\n================ PROBABILITY OF BACKTEST OVERFITTING (CSCV) ================")
    print(result.pbo.to_string())
    print("\n================ SUMMARY ================")
    print(result.summary.to_string())

    saved = save_strategy_evaluation_outputs(result, project_root=PROJECT_ROOT)
    print("\n================ SAVED ================")
    print(saved.to_string(index=False))


if __name__ == "__main__":
    main()
