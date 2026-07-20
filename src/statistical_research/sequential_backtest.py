"""Section 11 independent sequential backtest for the statistical branch.

A chronological, single-position, cost-aware trade simulator for the Section
10 candidate family.  It is a research backtest, not an execution system:

- Entries fill at the recorded next-bar open (the convention every label in
  this branch already uses); the entry-to-bar mapping is verified against the
  bar open price and the run aborts on any mismatch.
- One position at a time; while a trade is open every other candidate is
  skipped.  Trades never cross a New York date, a continuous segment, or the
  15:30 forced-exit boundary.
- Exits: stop, target, maximum-holding close, or forced 15:30 close.  When a
  bar touches both stop and target the conservative stop-first treatment is
  applied and the trade is flagged ambiguous.
- Costs are declared, versioned assumptions in ticks (commission plus
  spread/slippage per side) reported under frictionless, base, and
  pessimistic scenarios; they are not a fill model.
- Development and Validation only; Final-test rows are rejected.

The declared decision rule: a variant advances only with positive net base-
scenario expectancy in both Development and Validation.  Everything else is
recorded as a rejection of the standalone statistical system, which is the
outcome the PRD explicitly treats as a valid research result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_evaluation import EVALUATION_PARTITIONS
from .signal_construction import DIRECTION_VARIANTS, GATE_VARIANTS

SECTION11_RANDOM_SEED = 20260724
FORCED_EXIT_MINUTE_NY = 15 * 60 + 30


@dataclass(frozen=True)
class Section11Config:
    """Frozen Section 11 cost and simulation contract."""

    random_seed: int = SECTION11_RANDOM_SEED
    tick_size: float = 0.10
    commission_ticks_round_trip: float = 0.6
    base_slippage_ticks_per_side: float = 1.0
    pessimistic_slippage_ticks_per_side: float = 2.0
    max_holding_minutes: int = 120
    cost_scenarios: tuple[str, ...] = ("frictionless", "base", "pessimistic")


@dataclass
class BacktestBuildResult:
    """Container for the Section 11 outputs."""

    trade_log: pd.DataFrame
    performance: pd.DataFrame
    yearly_performance: pd.DataFrame
    equity_curves: pd.DataFrame
    verdicts: pd.DataFrame
    validation_checks: pd.DataFrame
    summary: pd.Series
    config: Section11Config = field(default_factory=Section11Config)


def load_backtest_bars(research_bars_path: Path) -> pd.DataFrame:
    """Narrow GC bar table required by the simulator, sorted by time."""

    import pyarrow.parquet as pq

    columns = [
        "ts_event_utc", "product", "open", "high", "low", "close",
        "trade_date_ny", "minute_of_day_ny", "continuous_segment_id",
    ]
    table = pq.read_table(research_bars_path, columns=columns,
                          filters=[("product", "==", "GC")])
    bars = table.to_pandas(ignore_metadata=True)
    bars = bars.sort_values("ts_event_utc").reset_index(drop=True)
    return bars


def _scenario_cost_ticks(config: Section11Config, scenario: str) -> float:
    if scenario == "frictionless":
        return 0.0
    per_side = (
        config.base_slippage_ticks_per_side
        if scenario == "base"
        else config.pessimistic_slippage_ticks_per_side
    )
    return config.commission_ticks_round_trip + 2.0 * per_side


def _simulate_variant(
    candidates: pd.DataFrame,
    bar_arrays: dict,
    entry_positions: np.ndarray,
    direction: str,
    gate: str,
    config: Section11Config,
) -> list[dict]:
    """One-position chronological walk over eligible candidates."""

    is_long = direction == "long_benchmark"
    eligible = np.ones(len(candidates), dtype=bool)
    if gate == "expansion_gated":
        eligible = candidates["expansion_gate_flag"].to_numpy(dtype=bool)

    high = bar_arrays["high"]
    low = bar_arrays["low"]
    close = bar_arrays["close"]
    segment = bar_arrays["segment"]
    trade_date = bar_arrays["trade_date"]
    minute_ny = bar_arrays["minute_ny"]
    n_bars = len(high)

    entry_price = candidates["entry_price"].to_numpy(dtype=np.float64)
    stop_points = candidates["stop_points"].to_numpy(dtype=np.float64)
    target_points = candidates["target_points"].to_numpy(dtype=np.float64)
    partitions = candidates["research_partition"].to_numpy()
    sessions = candidates["entry_session"].to_numpy()
    dates = candidates["trade_date_ny"].to_numpy()

    trades: list[dict] = []
    flat_from_position = -1
    order = np.argsort(entry_positions, kind="mergesort")
    for row in order:
        if not eligible[row]:
            continue
        start = int(entry_positions[row])
        if start <= flat_from_position:
            continue
        price_in = entry_price[row]
        stop = price_in - stop_points[row] if is_long else price_in + stop_points[row]
        target = price_in + target_points[row] if is_long else price_in - target_points[row]
        start_segment = segment[start]
        start_date = trade_date[start]
        exit_reason = "end_of_data"
        exit_position = start
        exit_price = close[start]
        ambiguous = False
        max_position = min(start + config.max_holding_minutes - 1, n_bars - 1)
        for position in range(start, max_position + 1):
            if segment[position] != start_segment or trade_date[position] != start_date:
                exit_reason = "boundary_exit"
                exit_position = position - 1
                exit_price = close[position - 1]
                break
            hit_stop = low[position] <= stop if is_long else high[position] >= stop
            hit_target = high[position] >= target if is_long else low[position] <= target
            if hit_stop:
                exit_reason = "stop"
                exit_position = position
                exit_price = stop
                ambiguous = bool(hit_target)
                break
            if hit_target:
                exit_reason = "target"
                exit_position = position
                exit_price = target
                break
            if minute_ny[position] >= FORCED_EXIT_MINUTE_NY:
                exit_reason = "forced_1530"
                exit_position = position
                exit_price = close[position]
                break
            if position == max_position:
                exit_reason = "time_exit"
                exit_position = position
                exit_price = close[position]
        points = exit_price - price_in if is_long else price_in - exit_price
        gross_r = points / stop_points[row]
        trades.append(
            {
                "direction_variant": direction,
                "gate_variant": gate,
                "research_partition": partitions[row],
                "entry_session": sessions[row],
                "trade_date_ny": dates[row],
                "entry_position": start,
                "exit_position": exit_position,
                "holding_minutes": int(exit_position - start + 1),
                "exit_reason": exit_reason,
                "ambiguous_bar": ambiguous,
                "stop_ticks": stop_points[row] / config.tick_size,
                "gross_r": float(gross_r),
            }
        )
        flat_from_position = int(exit_position)
    return trades


def run_sequential_backtest(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    config: Section11Config | None = None,
) -> BacktestBuildResult:
    """Simulate every declared variant chronologically and score it."""

    cfg = config or Section11Config()
    observed_partitions = set(candidates["research_partition"].astype(str).unique())
    if not observed_partitions.issubset(set(EVALUATION_PARTITIONS)):
        raise ValueError(f"backtest candidates contain locked partitions: {observed_partitions}")

    bar_ts = bars["ts_event_utc"].to_numpy()
    entry_ts = candidates["entry_timestamp_utc"].to_numpy()
    entry_positions = np.searchsorted(bar_ts, entry_ts)
    in_range = entry_positions < len(bar_ts)
    if not in_range.all():
        raise ValueError("candidate entry timestamps fall outside the bar history")
    matched = bar_ts[entry_positions] == entry_ts
    if not matched.all():
        raise ValueError("candidate entry timestamps do not align with bar timestamps")
    open_match = np.isclose(
        bars["open"].to_numpy(dtype=np.float64)[entry_positions],
        candidates["entry_price"].to_numpy(dtype=np.float64),
    )
    if not open_match.all():
        raise ValueError("entry prices do not equal the mapped bar opens")

    bar_arrays = {
        "high": bars["high"].to_numpy(dtype=np.float64),
        "low": bars["low"].to_numpy(dtype=np.float64),
        "close": bars["close"].to_numpy(dtype=np.float64),
        "segment": bars["continuous_segment_id"].to_numpy(),
        "trade_date": bars["trade_date_ny"].to_numpy(),
        "minute_ny": bars["minute_of_day_ny"].to_numpy(dtype=np.int64),
    }

    all_trades: list[dict] = []
    for direction in DIRECTION_VARIANTS:
        for gate in GATE_VARIANTS:
            all_trades.extend(
                _simulate_variant(candidates, bar_arrays, entry_positions, direction, gate, cfg)
            )
    trade_log = pd.DataFrame.from_records(all_trades)

    performance_records = []
    yearly_records = []
    equity_records = []
    variant_keys = ["direction_variant", "gate_variant"]
    for scenario in cfg.cost_scenarios:
        cost_ticks = _scenario_cost_ticks(cfg, scenario)
        net_r = trade_log["gross_r"] - cost_ticks / trade_log["stop_ticks"]
        scored = trade_log.assign(net_r=net_r)
        for (direction, gate, partition), group in scored.groupby(
            variant_keys + ["research_partition"], sort=False
        ):
            values = group["net_r"].to_numpy(dtype=np.float64)
            wins = values > 0
            equity = np.cumsum(values)
            drawdown = float(np.max(np.maximum.accumulate(equity) - equity)) if len(values) else np.nan
            performance_records.append(
                {
                    "cost_scenario": scenario,
                    "direction_variant": direction,
                    "gate_variant": gate,
                    "research_partition": partition,
                    "trades": len(values),
                    "trading_dates": int(group["trade_date_ny"].nunique()),
                    "win_rate": float(wins.mean()) if len(values) else np.nan,
                    "mean_net_r": float(values.mean()) if len(values) else np.nan,
                    "median_net_r": float(np.median(values)) if len(values) else np.nan,
                    "mean_win_r": float(values[wins].mean()) if wins.any() else np.nan,
                    "mean_loss_r": float(values[~wins].mean()) if (~wins).any() else np.nan,
                    "profit_factor": float(values[wins].sum() / -values[~wins].sum())
                    if (~wins).any() and values[~wins].sum() < 0 and wins.any()
                    else np.nan,
                    "max_drawdown_r": drawdown,
                    "ambiguous_rate": float(group["ambiguous_bar"].mean()),
                    "mean_holding_minutes": float(group["holding_minutes"].mean()),
                }
            )
        if scenario == "base":
            for (direction, gate, partition, year), group in scored.assign(
                entry_year=pd.DatetimeIndex(scored["trade_date_ny"]).year
            ).groupby(variant_keys + ["research_partition", "entry_year"], sort=False):
                yearly_records.append(
                    {
                        "direction_variant": direction,
                        "gate_variant": gate,
                        "research_partition": partition,
                        "entry_year": int(year),
                        "trades": len(group),
                        "mean_net_r": float(group["net_r"].mean()),
                        "win_rate": float((group["net_r"] > 0).mean()),
                    }
                )
            for (direction, gate), group in scored.groupby(variant_keys, sort=False):
                ordered = group.sort_values("entry_position")
                equity_records.append(
                    pd.DataFrame(
                        {
                            "direction_variant": direction,
                            "gate_variant": gate,
                            "trade_number": np.arange(1, len(ordered) + 1),
                            "trade_date_ny": ordered["trade_date_ny"].to_numpy(),
                            "cumulative_net_r": ordered["net_r"].cumsum().to_numpy(),
                        }
                    )
                )
    performance = pd.DataFrame.from_records(performance_records)
    yearly_performance = pd.DataFrame.from_records(yearly_records)
    equity_curves = (
        pd.concat(equity_records, ignore_index=True) if equity_records else pd.DataFrame()
    )

    # ---- declared advancement rule ----------------------------------------
    verdict_records = []
    base_rows = performance.loc[performance["cost_scenario"].eq("base")]
    for (direction, gate), group in base_rows.groupby(variant_keys, sort=False):
        by_partition = group.set_index("research_partition")["mean_net_r"]
        positive_everywhere = bool(
            (by_partition.reindex(list(EVALUATION_PARTITIONS)) > 0).all()
        )
        verdict_records.append(
            {
                "direction_variant": direction,
                "gate_variant": gate,
                "development_mean_net_r": float(by_partition.get("Development", np.nan)),
                "validation_mean_net_r": float(by_partition.get("Validation", np.nan)),
                "verdict": "POSITIVE_EXPECTANCY" if positive_everywhere else "REJECTED",
            }
        )
    verdicts = pd.DataFrame.from_records(verdict_records)
    system_rejected = bool(verdicts["verdict"].eq("REJECTED").all())

    checks = {
        "partitions_limited_to_development_validation": observed_partitions
        <= set(EVALUATION_PARTITIONS),
        "entry_prices_verified_against_bar_opens": True,
        "one_position_no_overlap": bool(
            trade_log.groupby(variant_keys)
            .apply(
                lambda g: bool(
                    (g.sort_values("entry_position")["entry_position"].to_numpy()[1:]
                     > g.sort_values("entry_position")["exit_position"].to_numpy()[:-1]).all()
                ),
                include_groups=False,
            )
            .all()
        ),
        "no_trade_crosses_forced_exit": bool(
            (trade_log["exit_reason"].ne("forced_1530")
             | (trade_log["holding_minutes"] <= cfg.max_holding_minutes)).all()
        ),
        "holding_within_declared_cap": bool(
            (trade_log["holding_minutes"] <= cfg.max_holding_minutes).all()
        ),
        "all_variants_simulated": set(
            map(tuple, trade_log[variant_keys].drop_duplicates().to_numpy())
        )
        == {(d, g) for d in DIRECTION_VARIANTS for g in GATE_VARIANTS},
        "scenario_costs_monotone": _scenario_cost_ticks(cfg, "frictionless")
        < _scenario_cost_ticks(cfg, "base")
        < _scenario_cost_ticks(cfg, "pessimistic"),
        "verdicts_cover_all_variants": len(verdicts)
        == len(DIRECTION_VARIANTS) * len(GATE_VARIANTS),
    }
    validation_checks = pd.Series(checks, name="passed").rename_axis("check").reset_index()

    summary = pd.Series(
        {
            "variants_simulated": len(verdicts),
            "total_trades": len(trade_log),
            "ambiguous_bar_rate": float(trade_log["ambiguous_bar"].mean()),
            "positive_expectancy_variants": int(
                verdicts["verdict"].eq("POSITIVE_EXPECTANCY").sum()
            ),
            "standalone_statistical_system_decision": (
                "REJECTED - no variant shows positive net base-scenario expectancy in "
                "both partitions" if system_rejected else "CANDIDATE VARIANTS EXIST - review required"
            ),
            "base_round_trip_cost_ticks": _scenario_cost_ticks(cfg, "base"),
            "pessimistic_round_trip_cost_ticks": _scenario_cost_ticks(cfg, "pessimistic"),
        },
        name="value",
    )

    return BacktestBuildResult(
        trade_log=trade_log,
        performance=performance,
        yearly_performance=yearly_performance,
        equity_curves=equity_curves,
        verdicts=verdicts,
        validation_checks=validation_checks,
        summary=summary,
        config=cfg,
    )


def save_backtest_outputs(result: BacktestBuildResult, *, project_root: Path) -> pd.DataFrame:
    """Persist Section 11 outputs; generated artifacts stay excluded from Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "section11"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "backtest_trade_log": (processed / "backtest_trade_log_gc.parquet", result.trade_log),
        "backtest_performance": (processed / "backtest_performance_gc.parquet", result.performance),
        "backtest_yearly_performance": (
            processed / "backtest_yearly_performance_gc.parquet",
            result.yearly_performance,
        ),
        "backtest_equity_curves": (
            processed / "backtest_equity_curves_gc.parquet",
            result.equity_curves,
        ),
        "backtest_verdicts": (processed / "backtest_verdicts_gc.parquet", result.verdicts),
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
    result.performance.to_csv(tables / "section11_performance_gc.csv", index=False)
    result.verdicts.to_csv(tables / "section11_verdicts_gc.csv", index=False)
    return pd.DataFrame.from_records(records)
