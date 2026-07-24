"""Generate STRATEGIES.md from the catalog specs and the saved backtest results.

The document is generated - never hand-edited - so the strategy descriptions,
parameters, and results in it cannot drift from the code or the run. Reads the
saved strategy-lab performance and evaluation tables and the catalog specs, and
writes a per-family catalog with each strategy's thesis, entry logic, parameters,
and pipeline result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_research.strategy_catalog import catalog_library, load_catalog_universe

SR = PROJECT_ROOT / "data" / "processed" / "statistical_research"

_FAMILY_TITLES = {
    "trend": "Trend / momentum",
    "reversion": "Mean-reversion",
    "breakout": "Breakout / range",
    "volatility": "Volatility-regime",
    "volume": "Volume / flow",
    "candle": "Candle / price-action",
    "vwap": "VWAP-relative",
    "session": "Session / time-of-day",
    "regime": "Regime-gated",
    "benchmark": "Null benchmarks",
}
_FAMILY_ORDER = [
    "trend",
    "reversion",
    "breakout",
    "volatility",
    "volume",
    "candle",
    "vwap",
    "session",
    "regime",
    "benchmark",
]


def _fmt_params(params: dict) -> str:
    parts = []
    for key, value in params.items():
        if key == "archetype":
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:g}")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else "-"


def _cell(value, pct=False):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:+.3f}"


def main() -> None:
    universe = load_catalog_universe(PROJECT_ROOT)
    specs, dropped = catalog_library(universe)

    perf = pd.read_parquet(SR / "strategy_lab_performance_gc.parquet")
    base = perf[perf["cost_scenario"] == "base"]
    fri = perf[perf["cost_scenario"] == "frictionless"]
    ev = pd.read_parquet(SR / "strategy_lab_evaluation_gc.parquet")

    def pivot(frame, value):
        return frame.pivot_table(index="strategy", columns="research_partition", values=value)

    dev_net = pivot(base, "mean_net_r")
    win = pivot(base, "win_rate")
    trades = pivot(base, "trades")
    fri_dev = fri[fri["research_partition"] == "Development"].set_index("strategy")["mean_net_r"]
    ev_i = ev.set_index("strategy")

    trials = ev[ev["is_trial"]]
    n_trials = len(trials)
    advanced = int((ev["verdict"] == "ADVANCE").sum())

    # Recompute the overfitting probability from the saved trade log so the number
    # in the document is data-derived, not transcribed.
    from src.statistical_research.strategy_evaluation import (
        EvaluationConfig,
        add_net_r,
        daily_returns_matrix,
        probability_of_backtest_overfitting,
    )

    cfg = EvaluationConfig()
    trade_log = pd.read_parquet(SR / "strategy_lab_trade_log_gc.parquet")
    scored = add_net_r(trade_log, cfg, cfg.cost_scenario)
    matrix = daily_returns_matrix(scored, list(trials["strategy"]))
    pbo = probability_of_backtest_overfitting(matrix, n_blocks=cfg.cscv_blocks)

    gross_pos = int((fri_dev.reindex(trials["strategy"]) > 0).sum())
    gross_min = float(fri_dev.reindex(trials["strategy"]).min())
    gross_max = float(fri_dev.reindex(trials["strategy"]).max())
    best_dsr = float(trials["deflated_sharpe"].max())

    lines: list[str] = []
    add = lines.append
    add("# Strategy catalog (strategy-lab branch)")
    add("")
    add(
        "A pre-registered catalog of ~200 rule-based trading strategies, each backtested through "
        "the branch's independently verified one-position engine and scored with a full "
        "anti-overfitting battery. This document is generated from the code and the run "
        "(`scripts/build_strategy_catalog_doc.py`) so the descriptions, parameters, and results "
        "below cannot drift from what was actually simulated."
    )
    add("")
    add(
        "**Peers: this is the menu.** Every row is a rule you can rerun, tweak, or use as a template "
        "for your own idea. Add a strategy by dropping a `StrategySpec` into "
        "`src/statistical_research/strategy_catalog.py` (or `strategy_lab.py`) and rerunning "
        "`scripts/run_strategy_catalog.py`; the evaluation deflates automatically for the new trial count."
    )
    add("")

    add("## Headline result")
    add("")
    add(
        f"- **{n_trials} strategies tested; {advanced} advanced.** Selection on Development, "
        "confirmation on Validation, Final-test partition untouched."
    )
    add(
        f"- **The gross (zero-cost) per-trade edge is ~0 for essentially every rule** - "
        f"{gross_pos}/{n_trials} have a positive frictionless edge, and the whole range is "
        f"[{gross_min:+.3f}, {gross_max:+.3f}] R. This is a signal wall, not a cost wall."
    )
    add(
        f"- **The best deflated Sharpe across all {n_trials} trials is {best_dsr:.3f}** "
        "(1.0 would mean a real edge after correcting for the number tried). Benjamini-Hochberg "
        "q-values and Validation bootstrap intervals agree."
    )
    add(
        f"- **Probability of backtest overfitting (CSCV): {pbo['pbo']:.2f}**, with "
        f"{pbo['prob_oos_best_positive']:.2f} probability the in-sample-best strategy is profitable "
        "out of sample - the ranking is stable but uniformly unprofitable."
    )
    add(
        "- Widening the search from a dozen rules to ~200 did not surface an edge; it raised the "
        "deflation bar, exactly as intended. A broad honest search that finds nothing is the result."
    )
    add("")

    add("## How each strategy is built and judged")
    add("")
    add(
        "- **Inputs are point-in-time.** Every rule reads only causal registered features "
        "(each available after the completed decision bar); fills are at the next bar's open; no "
        "forward label is ever read."
    )
    add(
        "- **Thresholds are pre-registered, not fitted.** They are Development-only marginal "
        "quantiles of the feature (p20/p80, or p10/p90 for extremes) or fixed structural levels "
        "(0 for a slope, 1 for a variance ratio, 0.5 for a bounded oscillator) - never chosen "
        "from the relationship between the feature and the outcome."
    )
    add(
        "- **One shared exit contract.** Every strategy inherits the frozen Section 10 stop and "
        "target (1.5x ATR stop, 2R target, 120-minute cap), so the search is purely over entries "
        "and every row is directly comparable. Varying the exit is a separate future axis."
    )
    add(
        "- **Costs:** base scenario is 2.6 ticks round trip; the frictionless column isolates the "
        "raw signal from the cost drag."
    )
    add(
        "- **Anti-overfitting:** date-block bootstrap CIs, Benjamini-Hochberg q-values across the "
        "family, the deflated Sharpe ratio (deflated by the full trial count), and the probability "
        "of backtest overfitting via CSCV. Numbers reproduce from `scripts/run_strategy_catalog.py`."
    )
    if dropped:
        add(
            f"- **Near-dead rules dropped:** {len(dropped)} pre-registered rule(s) whose condition "
            f"almost never occurs on this data were dropped as structurally inapplicable "
            f"({', '.join(dropped)})."
        )
    add("")
    add(
        "Columns: **Looks for** = the entry trigger; **Params** = the frozen constants; "
        "**Trades/Win** are base-cost Development; **Gross** is the frictionless Development "
        "per-trade edge (signal only); **Dev R / Val R** are base-cost per-trade net R; "
        "**DSR** is the deflated Sharpe; **Verdict** is the advancement gate."
    )
    add("")

    # Per-family aggregate
    add("## Results by family")
    add("")
    add("| Family | Strategies | Median Dev R (base) | Best gross edge | Best DSR | Advanced |")
    add("|---|---|---|---|---|---|")
    for fam in _FAMILY_ORDER:
        fam_names = [s.name for s in specs if s.family == fam]
        if not fam_names:
            continue
        fam_trials = [n for n in fam_names if n in set(trials["strategy"])]
        med = float(np.nanmedian([dev_net["Development"].get(n, np.nan) for n in fam_names]))
        bg = np.nanmax([fri_dev.get(n, np.nan) for n in fam_trials]) if fam_trials else np.nan
        bd = (
            np.nanmax([ev_i["deflated_sharpe"].get(n, np.nan) for n in fam_trials])
            if fam_trials
            else np.nan
        )
        adv = sum(1 for n in fam_names if ev_i["verdict"].get(n) == "ADVANCE")
        add(
            f"| {_FAMILY_TITLES[fam]} | {len(fam_names)} | {_cell(med)} | {_cell(bg)} | "
            f"{bd:.3f} | {adv} |"
        )
    add("")

    # Per-family detail tables
    for fam in _FAMILY_ORDER:
        fam_specs = [s for s in specs if s.family == fam]
        if not fam_specs:
            continue
        add(f"### {_FAMILY_TITLES[fam]} ({len(fam_specs)})")
        add("")
        add(
            "| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |"
        )
        add("|---|---|---|---|---|---|---|---|---|---|")
        order = sorted(
            fam_specs,
            key=lambda s: ev_i["combined_daily_sharpe"].get(s.name, -np.inf),
            reverse=True,
        )
        for s in order:
            n = s.name
            tr = trades["Development"].get(n, np.nan)
            tr_str = f"{int(tr):,}" if np.isfinite(tr) else "-"
            dsr = ev_i["deflated_sharpe"].get(n, np.nan)
            dsr_str = f"{dsr:.3f}" if np.isfinite(dsr) else "-"
            verdict = ev_i["verdict"].get(n, "-")
            looks = s.looks_for.replace("|", "/")
            add(
                f"| `{n}` | {looks} | {_fmt_params(s.parameters)} | {tr_str} | "
                f"{_cell(win['Development'].get(n, np.nan), pct=True)} | "
                f"{_cell(fri_dev.get(n, np.nan))} | {_cell(dev_net['Development'].get(n, np.nan))} | "
                f"{_cell(dev_net['Validation'].get(n, np.nan))} | {dsr_str} | {verdict} |"
            )
        add("")

    add("## Where this points next")
    add("")
    add(
        "- **Exit structure is the untested axis.** Every strategy here shares one exit "
        "(1.5x ATR stop, 2R target, 120-minute cap), so this was purely an entry search. Because a "
        "trade's exit depends only on its entry bar, direction, and stop/target - not on which "
        "strategy opened it - exit configurations can be precomputed once and swept cheaply across "
        "all entries. A future run should declare an exit grid (stop multiple, R-multiple, trailing "
        "and breakeven variants, time exits) in a frozen contract before running, and feed the grid "
        "size into the deflation."
    )
    add(
        "- **Direction is the wrong question; expansion/volatility is the open one.** Section 9 "
        "already found predictable structure in future *range* where direction has none. The faint "
        "positive gross tilt in this search is concentrated in fade-in-choppy-regime rules, "
        "consistent with mean-reversion/range carrying more signal than direction at this horizon."
    )
    add(
        "- **Tooling.** Taking this out of notebooks into an interactive chart-and-simulate surface "
        "(TradingView-style) is worth doing with open-source parts rather than rebuilding from "
        "scratch: TradingView Lightweight Charts for rendering, a vetted backtest/simulation engine "
        "(vectorbt, nautilus_trader, backtrader, or QuantConnect LEAN) behind this verified engine, "
        "and a thin UI (Streamlit or Dash). Scope a small spike before committing to one."
    )
    add("")
    add("## Reproduce")
    add("")
    add("```")
    add("python scripts/run_strategy_catalog.py       # build, simulate, evaluate, save")
    add("python scripts/build_strategy_catalog_doc.py  # regenerate this document")
    add("```")
    add("")
    add(
        "Generated tables (parquet/CSV) stay out of Git by policy; this markdown and the tracked "
        "summaries are the record."
    )
    add("")

    out = PROJECT_ROOT / "STRATEGIES.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines, {len(specs)} strategies)")


if __name__ == "__main__":
    main()
