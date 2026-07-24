"""Run the multi-strategy laboratory over the eligible GC universe.

Loads the signal candidates joined to their causal features and the raw GC bars,
simulates every pre-registered strategy through the verified one-position engine,
and prints Development/Validation performance under the base cost scenario.
Generated tables are written under ``data/processed`` and stay outside Git.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_research.sequential_backtest import load_backtest_bars
from src.statistical_research.strategy_lab import (
    load_strategy_universe,
    run_strategy_lab,
    save_strategy_lab_outputs,
)


def main() -> None:
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 220)

    universe = load_strategy_universe(PROJECT_ROOT)
    bars = load_backtest_bars(
        PROJECT_ROOT / "data" / "processed" / "research_bars_gc_mgc_1m.parquet"
    )
    result = run_strategy_lab(universe, bars)

    base = result.performance[result.performance["cost_scenario"] == "base"].copy()
    cols = [
        "strategy",
        "family",
        "research_partition",
        "trades",
        "win_rate",
        "mean_net_r",
        "per_trade_sharpe",
        "profit_factor",
        "max_drawdown_r",
        "long_fraction",
    ]
    dev = base[base["research_partition"] == "Development"][cols].sort_values(
        "mean_net_r", ascending=False
    )
    val = base[base["research_partition"] == "Validation"][cols]

    print("\n================ VALIDATION CHECKS ================")
    print(result.validation_checks.to_string(index=False))
    print("\n================ DEVELOPMENT (base costs, ranked) ================")
    print(dev.to_string(index=False))
    print("\n================ VALIDATION (base costs) =========================")
    print(val.set_index("strategy").reindex(dev["strategy"]).reset_index().to_string(index=False))
    print("\n================ SUMMARY ================")
    print(result.summary.to_string())

    saved = save_strategy_lab_outputs(result, project_root=PROJECT_ROOT)
    print("\n================ SAVED ================")
    print(saved.to_string(index=False))


if __name__ == "__main__":
    main()
