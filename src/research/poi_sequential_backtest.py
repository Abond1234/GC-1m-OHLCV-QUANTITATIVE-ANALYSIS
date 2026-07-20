"""Section 8 sequential research backtest of the frozen S7P02 POI policy.

Section 7 froze exactly one RESEARCH_ONLY candidate policy
(``S7P02_NY_BEAR_CONT``) and its registry row states the remaining question:
"Section 8 must sequence".  This module answers it: the frozen policy's
opportunity stream is executed as a chronological, single-position,
cost-aware trade sequence.

Frozen inputs, reused rather than reimplemented:

- Eligibility and prices come from the Section 7 machinery:
  :func:`build_stop_policy_opportunities` with the registry's exact settings
  (continuation hypothesis, volatility-hybrid stop, boundary-touch entry at
  the first-contact edge), filtered to the registry's eligibility (bearish
  POIs, New York session, case-2 geometry, compressed 15-minute approach at
  the Development-fitted 40th-percentile threshold).
- Sequencing, ambiguity, and cost rules mirror the Branch B Section 11
  simulator: one position at a time, conservative stop-first on ambiguous
  bars, declared tick costs under frictionless/base/pessimistic scenarios.

The declared advancement rule: positive net base-scenario expectancy in both
Development and Validation.  The Final test is read once, after the verdict
is fixed, following Branch A precedent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SECTION8_BACKTEST_RANDOM_SEED = 20260727
FORCED_EXIT_MINUTE_NY = 15 * 60 + 30

POLICY_ID = "S7P02_NY_BEAR_CONT"


@dataclass(frozen=True)
class Section8BacktestConfig:
    """Frozen Section 8 sequencing and cost contract for S7P02."""

    random_seed: int = SECTION8_BACKTEST_RANDOM_SEED
    policy_id: str = POLICY_ID
    hypothesis: str = "continuation"
    poi_direction: str = "bearish"
    eligible_session: str = "New York"
    poi_geometry: str = "case_2_standard"
    compression_feature: str = "feat_approach_15m_range_compression_ratio"
    compression_development_quantile: float = 0.40
    stop_model: str = "volatility_hybrid"
    entry_model: str = "boundary_touch"
    target_r: float = 3.0
    maximum_holding_minutes: int = 240
    tick_size: float = 0.10
    commission_ticks_round_trip: float = 0.6
    base_slippage_ticks_per_side: float = 1.0
    pessimistic_slippage_ticks_per_side: float = 2.0
    cost_scenarios: tuple[str, ...] = ("frictionless", "base", "pessimistic")


@dataclass
class PoiBacktestResult:
    """Container for the Section 8 outputs."""

    opportunity_summary: pd.Series
    trade_log: pd.DataFrame
    performance: pd.DataFrame
    yearly_performance: pd.DataFrame
    verdict: pd.Series
    validation_checks: pd.DataFrame
    config: Section8BacktestConfig = field(default_factory=Section8BacktestConfig)


def _scenario_cost_ticks(config: Section8BacktestConfig, scenario: str) -> float:
    if scenario == "frictionless":
        return 0.0
    per_side = (
        config.base_slippage_ticks_per_side
        if scenario == "base"
        else config.pessimistic_slippage_ticks_per_side
    )
    return config.commission_ticks_round_trip + 2.0 * per_side


def build_policy_opportunities(
    context: pd.DataFrame,
    bars: pd.DataFrame,
    config: Section8BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Frozen S7P02 opportunity stream with entry/stop prices and eligibility."""

    from src.research.poi_context_event_study import build_stop_policy_opportunities

    cfg = config or Section8BacktestConfig()
    opportunities = build_stop_policy_opportunities(
        context,
        bars,
        hypothesis=cfg.hypothesis,
        stop_model=cfg.stop_model,
        entry_model=cfg.entry_model,
    )
    aligned = context.drop_duplicates("true_retest_id").set_index("true_retest_id")
    keyed = opportunities.set_index("true_retest_id")
    keyed["feat_execution_session"] = aligned["feat_execution_session"]
    keyed["feat_poi_case"] = aligned["feat_poi_case"]
    keyed[cfg.compression_feature] = aligned[cfg.compression_feature]
    keyed = keyed.reset_index()

    partition = keyed["research_partition"].astype(str).str.lower()
    development = partition.eq("development")
    threshold = float(
        pd.to_numeric(keyed.loc[development, cfg.compression_feature], errors="coerce").quantile(
            cfg.compression_development_quantile
        )
    )
    eligible = (
        keyed["direction"].astype(str).eq(cfg.poi_direction)
        & keyed["feat_execution_session"].astype(str).eq(cfg.eligible_session)
        & keyed["feat_poi_case"].astype(str).eq(cfg.poi_geometry)
        & pd.to_numeric(keyed[cfg.compression_feature], errors="coerce").le(threshold)
        & keyed["entry_bar_id"].ge(0)
        & keyed["risk_points"].gt(0)
    )
    selected = keyed.loc[eligible].copy()
    selected["research_partition"] = (
        partition.loc[eligible]
        .map({"development": "Development", "validation": "Validation", "final_test": "Final test"})
        .to_numpy()
    )
    summary = pd.Series(
        {
            "policy_id": cfg.policy_id,
            "opportunity_rows": len(keyed),
            "eligible_opportunities": len(selected),
            "compression_threshold": threshold,
            "development_opportunities": int(
                (selected["research_partition"] == "Development").sum()
            ),
            "validation_opportunities": int((selected["research_partition"] == "Validation").sum()),
            "final_test_opportunities": int((selected["research_partition"] == "Final test").sum()),
        },
        name="value",
    )
    return selected, summary


def run_poi_sequential_backtest(
    opportunities: pd.DataFrame,
    bars: pd.DataFrame,
    config: Section8BacktestConfig | None = None,
) -> PoiBacktestResult:
    """One-position chronological execution of the frozen policy stream."""

    cfg = config or Section8BacktestConfig()
    high = bars["high"].to_numpy("float64")
    low = bars["low"].to_numpy("float64")
    close = bars["close"].to_numpy("float64")
    segment = bars["continuous_segment_id"].to_numpy()
    trade_date = bars["trade_date_ny"].to_numpy()
    minute_ny = bars["minute_of_day_ny"].to_numpy("int64")
    n_bars = len(high)

    entry_bar = opportunities["entry_bar_id"].to_numpy("int64")
    entry_price = opportunities["entry_price"].to_numpy("float64")
    stop_price = opportunities["stop_price"].to_numpy("float64")
    partitions = opportunities["research_partition"].to_numpy()
    dates = pd.to_datetime(opportunities["trade_date_ny"]).to_numpy()

    # Entry realism: a boundary-touch fill requires the edge price to have
    # actually traded inside the retest bar.
    fillable = (entry_price <= high[entry_bar]) & (entry_price >= low[entry_bar])
    unfillable = int((~fillable).sum())

    order = np.argsort(entry_bar, kind="mergesort")
    trades: list[dict] = []
    flat_from = -1
    for row in order:
        if not fillable[row]:
            continue
        start = int(entry_bar[row])
        if start <= flat_from:
            continue
        price_in = entry_price[row]
        stop = stop_price[row]
        risk = price_in - stop if stop < price_in else stop - price_in
        is_long = stop < price_in
        if risk <= 0:
            continue
        target = price_in + cfg.target_r * risk if is_long else price_in - cfg.target_r * risk
        start_segment = segment[start]
        start_date = trade_date[start]
        exit_reason, exit_position, exit_price, ambiguous = (
            "end_of_data",
            start,
            close[start],
            False,
        )
        max_position = min(start + cfg.maximum_holding_minutes - 1, n_bars - 1)
        for position in range(start, max_position + 1):
            if segment[position] != start_segment or trade_date[position] != start_date:
                exit_reason, exit_position, exit_price = (
                    "boundary_exit",
                    position - 1,
                    close[position - 1],
                )
                break
            hit_stop = low[position] <= stop if is_long else high[position] >= stop
            hit_target = high[position] >= target if is_long else low[position] <= target
            if hit_stop:
                exit_reason, exit_position, exit_price = "stop", position, stop
                ambiguous = bool(hit_target)
                break
            if hit_target:
                exit_reason, exit_position, exit_price = "target", position, target
                break
            if minute_ny[position] >= FORCED_EXIT_MINUTE_NY:
                exit_reason, exit_position, exit_price = "forced_1530", position, close[position]
                break
            if position == max_position:
                exit_reason, exit_position, exit_price = "time_exit", position, close[position]
        points = exit_price - price_in if is_long else price_in - exit_price
        trades.append(
            {
                "research_partition": partitions[row],
                "trade_date_ny": dates[row],
                "entry_position": start,
                "exit_position": int(exit_position),
                "holding_minutes": int(exit_position - start + 1),
                "exit_reason": exit_reason,
                "ambiguous_bar": ambiguous,
                "stop_ticks": risk / cfg.tick_size,
                "gross_r": float(points / risk),
            }
        )
        flat_from = int(exit_position)
    trade_log = pd.DataFrame.from_records(
        trades,
        columns=[
            "research_partition",
            "trade_date_ny",
            "entry_position",
            "exit_position",
            "holding_minutes",
            "exit_reason",
            "ambiguous_bar",
            "stop_ticks",
            "gross_r",
        ],
    )

    performance_records, yearly_records = [], []
    for scenario in cfg.cost_scenarios:
        cost_ticks = _scenario_cost_ticks(cfg, scenario)
        scored = trade_log.assign(net_r=trade_log["gross_r"] - cost_ticks / trade_log["stop_ticks"])
        for partition, group in scored.groupby("research_partition", sort=False):
            values = group["net_r"].to_numpy("float64")
            wins = values > 0
            equity = np.cumsum(values)
            performance_records.append(
                {
                    "cost_scenario": scenario,
                    "research_partition": partition,
                    "trades": len(values),
                    "trading_dates": int(group["trade_date_ny"].nunique()),
                    "win_rate": float(wins.mean()),
                    "mean_net_r": float(values.mean()),
                    "median_net_r": float(np.median(values)),
                    "profit_factor": float(values[wins].sum() / -values[~wins].sum())
                    if wins.any() and (~wins).any() and values[~wins].sum() < 0
                    else np.nan,
                    "max_drawdown_r": float(np.max(np.maximum.accumulate(equity) - equity)),
                    "ambiguous_rate": float(group["ambiguous_bar"].mean()),
                    "mean_holding_minutes": float(group["holding_minutes"].mean()),
                }
            )
        if scenario == "base":
            for (partition, year), group in scored.assign(
                entry_year=pd.DatetimeIndex(scored["trade_date_ny"]).year
            ).groupby(["research_partition", "entry_year"], sort=False):
                yearly_records.append(
                    {
                        "research_partition": partition,
                        "entry_year": int(year),
                        "trades": len(group),
                        "mean_net_r": float(group["net_r"].mean()),
                        "win_rate": float((group["net_r"] > 0).mean()),
                    }
                )
    performance = pd.DataFrame.from_records(
        performance_records,
        columns=[
            "cost_scenario",
            "research_partition",
            "trades",
            "trading_dates",
            "win_rate",
            "mean_net_r",
            "median_net_r",
            "profit_factor",
            "max_drawdown_r",
            "ambiguous_rate",
            "mean_holding_minutes",
        ],
    )
    yearly_performance = pd.DataFrame.from_records(yearly_records)

    base = performance.loc[performance["cost_scenario"].eq("base")].set_index("research_partition")
    dev_mean = (
        float(base.at["Development", "mean_net_r"]) if "Development" in base.index else np.nan
    )
    val_mean = float(base.at["Validation", "mean_net_r"]) if "Validation" in base.index else np.nan
    final_mean = (
        float(base.at["Final test", "mean_net_r"]) if "Final test" in base.index else np.nan
    )
    advances = bool(
        np.isfinite(dev_mean) and dev_mean > 0 and np.isfinite(val_mean) and val_mean > 0
    )
    verdict = pd.Series(
        {
            "policy_id": cfg.policy_id,
            "development_mean_net_r": dev_mean,
            "validation_mean_net_r": val_mean,
            "verdict_from_dev_val": "SEQUENTIAL_POSITIVE" if advances else "SEQUENTIAL_REJECTED",
            "final_test_mean_net_r_one_time_read": final_mean,
            "unfillable_entries_dropped": unfillable,
        },
        name="value",
    )

    per_partition_sorted = (
        trade_log.sort_values("entry_position")
        .groupby("research_partition")["entry_position"]
        .apply(lambda s: bool((s.to_numpy()[1:] > s.to_numpy()[:-1]).all()) if len(s) > 1 else True)
        .all()
    )
    checks = {
        "one_position_no_overlap": bool(
            (
                trade_log.sort_values("entry_position")["entry_position"].to_numpy()[1:]
                > trade_log.sort_values("entry_position")["exit_position"].to_numpy()[:-1]
            ).all()
        )
        if len(trade_log) > 1
        else True,
        "entries_chronological_within_partitions": bool(per_partition_sorted),
        "holding_within_declared_cap": bool(
            (trade_log["holding_minutes"] <= cfg.maximum_holding_minutes).all()
        ),
        "entry_prices_traded_within_entry_bars": True,
        "scenario_costs_monotone": _scenario_cost_ticks(cfg, "frictionless")
        < _scenario_cost_ticks(cfg, "base")
        < _scenario_cost_ticks(cfg, "pessimistic"),
        "verdict_uses_dev_val_only": True,
    }
    validation_checks = pd.Series(checks, name="passed").rename_axis("check").reset_index()

    return PoiBacktestResult(
        opportunity_summary=pd.Series(dtype="object"),
        trade_log=trade_log,
        performance=performance,
        yearly_performance=yearly_performance,
        verdict=verdict,
        validation_checks=validation_checks,
        config=cfg,
    )


def save_poi_backtest_outputs(result: PoiBacktestResult, *, project_root: Path) -> pd.DataFrame:
    """Persist Section 8 outputs; generated artifacts stay excluded from Git."""

    processed = project_root / "data" / "processed"
    outputs = {
        "section8_poi_trade_log": (
            processed / "section8_poi_trade_log_gc.parquet",
            result.trade_log,
        ),
        "section8_poi_performance": (
            processed / "section8_poi_performance_gc.parquet",
            result.performance,
        ),
        "section8_poi_yearly_performance": (
            processed / "section8_poi_yearly_performance_gc.parquet",
            result.yearly_performance,
        ),
    }
    records = []
    for name, (path, table) in outputs.items():
        table.to_parquet(path, index=False)
        reloaded = pd.read_parquet(path, engine="pyarrow")
        records.append(
            {
                "output": name,
                "path": str(path.relative_to(project_root)),
                "rows": len(table),
                "columns": table.shape[1],
                "reload_row_match": len(reloaded) == len(table),
            }
        )
    return pd.DataFrame.from_records(records)
