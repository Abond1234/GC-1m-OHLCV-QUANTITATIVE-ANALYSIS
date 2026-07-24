"""Run the ~200-strategy catalog through the verified pipeline and evaluation.

Builds the pre-registered catalog, simulates every strategy on the verified
one-position engine, applies the anti-overfitting evaluation (bootstrap,
Benjamini-Hochberg, deflated Sharpe, PBO/CSCV), and saves the outputs.  Generated
tables stay outside Git; the tracked record is the generated catalog document.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_research.sequential_backtest import load_backtest_bars
from src.statistical_research.strategy_catalog import catalog_library, load_catalog_universe
from src.statistical_research.strategy_evaluation import (
    evaluate_strategies,
    save_strategy_evaluation_outputs,
)
from src.statistical_research.strategy_lab import run_strategy_lab, save_strategy_lab_outputs

SR = PROJECT_ROOT / "data" / "processed" / "statistical_research"


def main() -> None:
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 240)

    universe = load_catalog_universe(PROJECT_ROOT)
    bars = load_backtest_bars(
        PROJECT_ROOT / "data" / "processed" / "research_bars_gc_mgc_1m.parquet"
    )
    specs, dropped = catalog_library(universe)
    print(f"catalog: {len(specs)} strategies (dropped {len(dropped)} near-dead: {dropped})")

    result = run_strategy_lab(universe, bars, specs=specs)
    save_strategy_lab_outputs(result, project_root=PROJECT_ROOT)
    print("\n=== lab validation checks ===")
    print(result.validation_checks.to_string(index=False))
    print("\n=== lab summary ===")
    print(result.summary.to_string())

    evaluation = evaluate_strategies(result.trade_log)
    save_strategy_evaluation_outputs(evaluation, project_root=PROJECT_ROOT)

    stats = evaluation.strategy_stats
    trials = stats[stats["is_trial"]]
    top = trials.sort_values("combined_daily_sharpe", ascending=False).head(15)
    cols = [
        "strategy",
        "family",
        "dev_mean_net_r",
        "val_mean_net_r",
        "dev_bh_q",
        "val_ci_low",
        "deflated_sharpe",
        "verdict",
    ]
    print("\n=== top 15 trials by combined daily Sharpe ===")
    print(top[cols].to_string(index=False))
    print("\n=== evaluation summary ===")
    print(evaluation.summary.to_string())
    print("\n=== PBO / CSCV ===")
    print(evaluation.pbo.to_string())

    # frictionless (signal) vs base (cost) decomposition across trials
    perf = result.performance
    dev = perf[
        (perf["cost_scenario"].isin(["frictionless", "base"]))
        & (perf["research_partition"] == "Development")
    ]
    piv = dev.pivot_table(index="strategy", columns="cost_scenario", values="mean_net_r")
    piv = piv.reindex(columns=["frictionless", "base"]).dropna()
    piv = piv[piv.index.isin(trials["strategy"])]
    print("\n=== frictionless (gross) per-trade edge across trials ===")
    print(
        f"  strategies with positive gross edge: {int((piv['frictionless'] > 0).sum())} / {len(piv)}"
    )
    print(
        f"  gross edge range: [{piv['frictionless'].min():.4f}, {piv['frictionless'].max():.4f}] R"
    )
    print(
        f"  best gross-edge strategy: {piv['frictionless'].idxmax()} ({piv['frictionless'].max():.4f} R)"
    )


if __name__ == "__main__":
    main()
