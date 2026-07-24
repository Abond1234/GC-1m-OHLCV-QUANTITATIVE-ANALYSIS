"""A multi-strategy laboratory on the verified sequential-backtest engine.

Section 11 backtested only two *benchmarks* - always-long and always-short of
every eligible observation - because Section 7 approved no directional signal.
This module asks the natural next question honestly: does any economically
motivated directional rule, drawn from the causal registered features, survive
the same cost-aware, one-position simulation that the benchmarks did not?

Design principles that keep the search trustworthy:

- **Reuse the verified contract.** The one-position chronological walk, the
  stop/target/time/forced/boundary exits, the conservative stop-first treatment,
  and the three cost scenarios are exactly those the independent verifier
  reconciled to the tick (`backtest_verification`). The only generalisation is a
  per-trade direction, so a real rule can go long on some bars and short on
  others.  Run with a constant direction it reproduces the Section 11 benchmark
  exactly - a regression tie-back the tests assert.
- **Point-in-time inputs only.** Every signal reads registered features, each
  documented as causal ("available after completed decision bar t"); fills are at
  the next-bar open.  No forward label is ever read.
- **Pre-registered rules.** Each strategy is a fixed rule with round thresholds
  chosen from feature *marginals*, never from their relationship to outcomes.
  The stop/target contract is shared (Section 10's 1.5x ATR clamp, 2R target) so
  strategies are comparable and the search space is not inflated by exit tuning.
- **Development and Validation only.**  Final-test rows are rejected.

Scoring here is deliberately descriptive (win rate, mean net R, profit factor,
drawdown, per-trade Sharpe).  The multiple-testing defence - bootstrap
confidence intervals, Benjamini-Hochberg, the deflated Sharpe ratio, and the
probability of backtest overfitting - lives in the evaluation layer, because a
zoo of pre-registered rules still demands correction before anything is believed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_evaluation import EVALUATION_PARTITIONS
from .sequential_backtest import FORCED_EXIT_MINUTE_NY

STRATEGY_LAB_SEED = 20260724

# Feature columns the strategy library reads (all causal registered predictors).
SIGNAL_FEATURE_COLUMNS = (
    "return_60m_atr",
    "return_15m_atr",
    "efficiency_ratio_60",
    "normalized_ols_slope_60",
    "distance_from_rolling_vwap_60_atr",
    "distance_from_rolling_vwap_20_atr",
    "vwap_slope_15_atr",
    "session_range_position",
    "lower_wick_to_range",
    "upper_wick_to_range",
    "close_location_value",
    "current_range_over_atr",
    "signed_body_atr",
    "atr_ratio_5_20",
    "minute_from_execution_window_open",
    "relative_volume_60",
    "signed_volume_proxy",
    "return_sign_change_rate_30",
    "directional_streak",
)


@dataclass(frozen=True)
class StrategyLabConfig:
    """Frozen cost contract and pre-registered signal thresholds."""

    random_seed: int = STRATEGY_LAB_SEED
    tick_size: float = 0.10
    commission_ticks_round_trip: float = 0.6
    base_slippage_ticks_per_side: float = 1.0
    pessimistic_slippage_ticks_per_side: float = 2.0
    max_holding_minutes: int = 120
    forced_exit_minute_ny: int = FORCED_EXIT_MINUTE_NY
    cost_scenarios: tuple[str, ...] = ("frictionless", "base", "pessimistic")

    # Pre-registered thresholds (chosen from feature marginals, not outcomes).
    momentum_deadband_atr: float = 1.0
    short_momentum_deadband_atr: float = 0.5
    efficiency_trend_min: float = 0.25
    vwap_reversion_stretch_atr: float = 1.5
    session_extreme_low: float = 0.10
    session_extreme_high: float = 0.90
    wick_min: float = 0.50
    close_location_confirm: float = 0.30
    range_breakout_over_atr: float = 1.5
    atr_expansion_ratio: float = 1.2
    opening_window_minutes: float = 15.0
    relative_volume_min: float = 1.5
    sign_change_choppy_min: float = 0.65
    directional_streak_min: float = 2.0

    def scenario_cost_ticks(self, scenario: str) -> float:
        if scenario == "frictionless":
            return 0.0
        per_side = (
            self.base_slippage_ticks_per_side
            if scenario == "base"
            else self.pessimistic_slippage_ticks_per_side
        )
        return self.commission_ticks_round_trip + 2.0 * per_side


SignalFn = Callable[[pd.DataFrame, StrategyLabConfig], np.ndarray]


@dataclass(frozen=True)
class StrategySpec:
    """A named, pre-registered directional rule."""

    name: str
    family: str
    thesis: str
    signal: SignalFn


# ---------------------------------------------------------------------------
# Signal library.  Each returns an int array in {-1, 0, +1}: short, flat, long.
# ---------------------------------------------------------------------------
def _sign_deadband(values: np.ndarray, deadband: float) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.int8)
    out[values > deadband] = 1
    out[values < -deadband] = -1
    return out


def _signal_momentum_60m(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    return _sign_deadband(
        frame["return_60m_atr"].to_numpy(dtype=np.float64), cfg.momentum_deadband_atr
    )


def _signal_trend_efficiency(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    er = frame["efficiency_ratio_60"].to_numpy(dtype=np.float64)
    slope = frame["normalized_ols_slope_60"].to_numpy(dtype=np.float64)
    out = np.sign(slope).astype(np.int8)
    out[er < cfg.efficiency_trend_min] = 0
    return out


def _signal_vwap_trend(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    dist = frame["distance_from_rolling_vwap_60_atr"].to_numpy(dtype=np.float64)
    slope = frame["vwap_slope_15_atr"].to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.int8)
    out[(dist > 0) & (slope > 0)] = 1
    out[(dist < 0) & (slope < 0)] = -1
    return out


def _signal_streak_follow(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    streak = frame["directional_streak"].to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.int8)
    out[streak >= cfg.directional_streak_min] = 1
    out[streak <= -cfg.directional_streak_min] = -1
    return out


def _signal_session_extreme_fade(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    pos = frame["session_range_position"].to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.int8)
    out[pos < cfg.session_extreme_low] = 1  # near session low -> fade upward
    out[pos > cfg.session_extreme_high] = -1  # near session high -> fade downward
    return out


def _signal_vwap_reversion(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    dist = frame["distance_from_rolling_vwap_20_atr"].to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.int8)
    out[dist < -cfg.vwap_reversion_stretch_atr] = 1  # stretched below -> revert up
    out[dist > cfg.vwap_reversion_stretch_atr] = -1
    return out


def _signal_wick_reversal(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    lower = frame["lower_wick_to_range"].to_numpy(dtype=np.float64)
    upper = frame["upper_wick_to_range"].to_numpy(dtype=np.float64)
    clv = frame["close_location_value"].to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.int8)
    out[(lower > cfg.wick_min) & (clv > cfg.close_location_confirm)] = 1
    out[(upper > cfg.wick_min) & (clv < -cfg.close_location_confirm)] = -1
    return out


def _signal_range_breakout(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    rng = frame["current_range_over_atr"].to_numpy(dtype=np.float64)
    pos = frame["session_range_position"].to_numpy(dtype=np.float64)
    body = frame["signed_body_atr"].to_numpy(dtype=np.float64)
    expand = rng > cfg.range_breakout_over_atr
    out = np.zeros(len(frame), dtype=np.int8)
    out[expand & (pos > cfg.session_extreme_high) & (body > 0)] = 1
    out[expand & (pos < cfg.session_extreme_low) & (body < 0)] = -1
    return out


def _signal_vol_expansion_momentum(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    ratio = frame["atr_ratio_5_20"].to_numpy(dtype=np.float64)
    body = frame["signed_body_atr"].to_numpy(dtype=np.float64)
    out = np.sign(body).astype(np.int8)
    out[ratio < cfg.atr_expansion_ratio] = 0
    return out


def _signal_opening_drive(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    minute = frame["minute_from_execution_window_open"].to_numpy(dtype=np.float64)
    ret = frame["return_15m_atr"].to_numpy(dtype=np.float64)
    out = _sign_deadband(ret, cfg.short_momentum_deadband_atr)
    out[minute >= cfg.opening_window_minutes] = 0
    return out


def _signal_volume_momentum(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    rvol = frame["relative_volume_60"].to_numpy(dtype=np.float64)
    signed = frame["signed_volume_proxy"].to_numpy(dtype=np.float64)
    out = np.sign(signed).astype(np.int8)
    out[rvol < cfg.relative_volume_min] = 0
    return out


def _signal_choppiness_reversion(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    chop = frame["return_sign_change_rate_30"].to_numpy(dtype=np.float64)
    ret = frame["return_15m_atr"].to_numpy(dtype=np.float64)
    out = -_sign_deadband(ret, cfg.short_momentum_deadband_atr)  # fade the move
    out[chop < cfg.sign_change_choppy_min] = 0
    return out


def _signal_always_long(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    return np.ones(len(frame), dtype=np.int8)


def _signal_always_short(frame: pd.DataFrame, cfg: StrategyLabConfig) -> np.ndarray:
    return np.full(len(frame), -1, dtype=np.int8)


def strategy_library() -> list[StrategySpec]:
    """The pre-registered strategy family, plus the two null benchmarks."""

    return [
        StrategySpec(
            "momentum_60m",
            "trend",
            "Follow 60-minute ATR-normalised displacement beyond 1 ATR.",
            _signal_momentum_60m,
        ),
        StrategySpec(
            "trend_efficiency_60",
            "trend",
            "Trade the OLS-slope direction only when 60m efficiency ratio is high.",
            _signal_trend_efficiency,
        ),
        StrategySpec(
            "vwap_trend_60",
            "trend",
            "Long above a rising rolling VWAP, short below a falling one.",
            _signal_vwap_trend,
        ),
        StrategySpec(
            "streak_follow",
            "trend",
            "Follow a directional streak of two or more bars.",
            _signal_streak_follow,
        ),
        StrategySpec(
            "session_extreme_fade",
            "reversion",
            "Fade the session extremes of the intraday range.",
            _signal_session_extreme_fade,
        ),
        StrategySpec(
            "vwap_reversion_20",
            "reversion",
            "Fade a stretch beyond 1.5 ATR from the 20m rolling VWAP.",
            _signal_vwap_reversion,
        ),
        StrategySpec(
            "wick_reversal",
            "reversion",
            "Fade rejection wicks confirmed by close location.",
            _signal_wick_reversal,
        ),
        StrategySpec(
            "range_breakout",
            "breakout",
            "Trade expansion bars breaking to a session extreme.",
            _signal_range_breakout,
        ),
        StrategySpec(
            "vol_expansion_momentum",
            "volatility",
            "Follow the candle body only when short-horizon volatility expands.",
            _signal_vol_expansion_momentum,
        ),
        StrategySpec(
            "opening_drive",
            "session",
            "Follow the opening 15-minute drift of each execution window.",
            _signal_opening_drive,
        ),
        StrategySpec(
            "volume_momentum",
            "volume",
            "Follow signed volume flow when relative volume is elevated.",
            _signal_volume_momentum,
        ),
        StrategySpec(
            "choppiness_reversion",
            "reversion",
            "Fade short-horizon moves when the sign-change rate is high.",
            _signal_choppiness_reversion,
        ),
        StrategySpec(
            "always_long",
            "benchmark",
            "Null benchmark: long every eligible observation.",
            _signal_always_long,
        ),
        StrategySpec(
            "always_short",
            "benchmark",
            "Null benchmark: short every eligible observation.",
            _signal_always_short,
        ),
    ]


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------
def load_strategy_universe(project_root: Path) -> pd.DataFrame:
    """Join the signal candidates (entry fields + stops) to their causal features."""

    import pyarrow.parquet as pq

    sr = project_root / "data" / "processed" / "statistical_research"
    candidates = pd.read_parquet(sr / "signal_candidates_gc.parquet")
    features = pq.read_table(
        sr / "feature_matrix_gc.parquet",
        columns=["observation_id", *SIGNAL_FEATURE_COLUMNS],
    ).to_pandas(ignore_metadata=True)
    universe = candidates.merge(features, on="observation_id", how="inner")
    missing = universe[list(SIGNAL_FEATURE_COLUMNS)].isna().any(axis=1)
    universe = universe.loc[~missing].reset_index(drop=True)
    return universe


# ---------------------------------------------------------------------------
# Generalised one-position simulation (per-trade direction)
# ---------------------------------------------------------------------------
def _bar_arrays(bars: pd.DataFrame) -> dict:
    return {
        "ts": bars["ts_event_utc"].to_numpy(),
        "open": bars["open"].to_numpy(dtype=np.float64),
        "high": bars["high"].to_numpy(dtype=np.float64),
        "low": bars["low"].to_numpy(dtype=np.float64),
        "close": bars["close"].to_numpy(dtype=np.float64),
        "segment": bars["continuous_segment_id"].to_numpy(),
        "trade_date": bars["trade_date_ny"].to_numpy(),
        "minute_ny": bars["minute_of_day_ny"].to_numpy(dtype=np.int64),
    }


def simulate_positions(
    entry_positions: np.ndarray,
    directions: np.ndarray,
    stop_points: np.ndarray,
    target_points: np.ndarray,
    metadata: pd.DataFrame,
    arrays: dict,
    cfg: StrategyLabConfig,
) -> pd.DataFrame:
    """One-position chronological walk with a per-trade direction.

    ``directions`` is +1 (long) or -1 (short) for each fired entry.  The exit
    logic is identical to the verified Section 11 contract.
    """

    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    open_price = arrays["open"]
    segment = arrays["segment"]
    trade_date = arrays["trade_date"]
    minute_ny = arrays["minute_ny"]
    n_bars = len(high)

    partitions = metadata["research_partition"].to_numpy()
    sessions = metadata["entry_session"].to_numpy()
    dates = metadata["trade_date_ny"].to_numpy()

    order = np.argsort(entry_positions, kind="mergesort")
    flat_from = -1
    records: list[dict] = []
    for row in order:
        start = int(entry_positions[row])
        if start <= flat_from:
            continue
        is_long = directions[row] > 0
        price_in = open_price[start]
        stop_pts = stop_points[row]
        target_pts = target_points[row]
        stop = price_in - stop_pts if is_long else price_in + stop_pts
        target = price_in + target_pts if is_long else price_in - target_pts
        start_segment = segment[start]
        start_date = trade_date[start]
        max_position = min(start + cfg.max_holding_minutes - 1, n_bars - 1)
        exit_position = start
        exit_reason = "end_of_data"
        exit_price = close[start]
        ambiguous = False
        for position in range(start, max_position + 1):
            if segment[position] != start_segment or trade_date[position] != start_date:
                exit_position = position - 1
                exit_reason = "boundary_exit"
                exit_price = close[position - 1]
                break
            hit_stop = low[position] <= stop if is_long else high[position] >= stop
            hit_target = high[position] >= target if is_long else low[position] <= target
            if hit_stop:
                exit_position = position
                exit_reason = "stop"
                exit_price = stop
                ambiguous = bool(hit_target)
                break
            if hit_target:
                exit_position = position
                exit_reason = "target"
                exit_price = target
                break
            if minute_ny[position] >= cfg.forced_exit_minute_ny:
                exit_position = position
                exit_reason = "forced_1530"
                exit_price = close[position]
                break
            if position == max_position:
                exit_position = position
                exit_reason = "time_exit"
                exit_price = close[position]
        points = exit_price - price_in if is_long else price_in - exit_price
        records.append(
            {
                "research_partition": partitions[row],
                "entry_session": sessions[row],
                "trade_date_ny": dates[row],
                "direction": 1 if is_long else -1,
                "entry_position": start,
                "exit_position": exit_position,
                "holding_minutes": int(exit_position - start + 1),
                "exit_reason": exit_reason,
                "ambiguous_bar": ambiguous,
                "stop_ticks": stop_pts / cfg.tick_size,
                "gross_r": float(points / stop_pts),
            }
        )
        flat_from = int(exit_position)
    return pd.DataFrame.from_records(records)


def simulate_strategy(
    universe: pd.DataFrame,
    entry_positions: np.ndarray,
    arrays: dict,
    spec: StrategySpec,
    cfg: StrategyLabConfig,
) -> pd.DataFrame:
    """Generate a strategy's signals and simulate its fired entries."""

    directions = spec.signal(universe, cfg)
    fired = directions != 0
    if not fired.any():
        return pd.DataFrame()
    trades = simulate_positions(
        entry_positions[fired],
        directions[fired],
        universe["stop_points"].to_numpy(dtype=np.float64)[fired],
        universe["target_points"].to_numpy(dtype=np.float64)[fired],
        universe.loc[fired, ["research_partition", "entry_session", "trade_date_ny"]].reset_index(
            drop=True
        ),
        arrays,
        cfg,
    )
    trades.insert(0, "strategy", spec.name)
    trades.insert(1, "family", spec.family)
    return trades


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_trades(trade_log: pd.DataFrame, cfg: StrategyLabConfig) -> pd.DataFrame:
    """Per strategy x partition x cost scenario performance metrics."""

    records: list[dict] = []
    for scenario in cfg.cost_scenarios:
        cost_ticks = cfg.scenario_cost_ticks(scenario)
        scored = trade_log.assign(net_r=trade_log["gross_r"] - cost_ticks / trade_log["stop_ticks"])
        keys = ["strategy", "family", "research_partition"]
        for (strategy, family, partition), group in scored.groupby(keys, sort=False):
            values = group["net_r"].to_numpy(dtype=np.float64)
            wins = values > 0
            equity = np.cumsum(values)
            drawdown = (
                float(np.max(np.maximum.accumulate(equity) - equity)) if len(values) else np.nan
            )
            std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            records.append(
                {
                    "cost_scenario": scenario,
                    "strategy": strategy,
                    "family": family,
                    "research_partition": partition,
                    "trades": len(values),
                    "trading_dates": int(group["trade_date_ny"].nunique()),
                    "long_fraction": float((group["direction"] > 0).mean()),
                    "win_rate": float(wins.mean()) if len(values) else np.nan,
                    "mean_net_r": float(values.mean()) if len(values) else np.nan,
                    "median_net_r": float(np.median(values)) if len(values) else np.nan,
                    "std_net_r": std,
                    "per_trade_sharpe": float(values.mean() / std) if std and std > 0 else np.nan,
                    "profit_factor": float(values[wins].sum() / -values[~wins].sum())
                    if (~wins).any() and values[~wins].sum() < 0 and wins.any()
                    else np.nan,
                    "max_drawdown_r": drawdown,
                    "total_net_r": float(values.sum()),
                    "mean_holding_minutes": float(group["holding_minutes"].mean()),
                    "ambiguous_rate": float(group["ambiguous_bar"].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


@dataclass
class StrategyLabResult:
    """Container for the strategy-lab outputs."""

    trade_log: pd.DataFrame
    performance: pd.DataFrame
    validation_checks: pd.DataFrame
    summary: pd.Series
    config: StrategyLabConfig = field(default_factory=StrategyLabConfig)


def run_strategy_lab(
    universe: pd.DataFrame,
    bars: pd.DataFrame,
    specs: list[StrategySpec] | None = None,
    config: StrategyLabConfig | None = None,
) -> StrategyLabResult:
    """Simulate every strategy over the eligible universe and score it."""

    cfg = config or StrategyLabConfig()
    specs = specs or strategy_library()

    observed = set(universe["research_partition"].astype(str).unique())
    if not observed.issubset(set(EVALUATION_PARTITIONS)):
        raise ValueError(f"strategy universe contains locked partitions: {observed}")

    bar_index = pd.Index(bars["ts_event_utc"])
    entry_positions = bar_index.get_indexer(universe["entry_timestamp_utc"])
    if (entry_positions < 0).any():
        raise ValueError("universe entry timestamps not found in the GC bars")
    open_match = np.isclose(
        bars["open"].to_numpy(dtype=np.float64)[entry_positions],
        universe["entry_price"].to_numpy(dtype=np.float64),
    )
    if not open_match.all():
        raise ValueError("universe entry prices do not equal the mapped bar opens")
    arrays = _bar_arrays(bars)

    logs = [simulate_strategy(universe, entry_positions, arrays, spec, cfg) for spec in specs]
    trade_log = pd.concat([log for log in logs if not log.empty], ignore_index=True)
    performance = score_trades(trade_log, cfg)

    # ---- integrity checks --------------------------------------------------
    per_strategy_no_overlap = (
        trade_log.groupby("strategy")
        .apply(
            lambda g: bool(
                (
                    g.sort_values("entry_position")["entry_position"].to_numpy()[1:]
                    > g.sort_values("entry_position")["exit_position"].to_numpy()[:-1]
                ).all()
            ),
            include_groups=False,
        )
        .all()
    )
    checks = {
        "partitions_limited_to_development_validation": observed <= set(EVALUATION_PARTITIONS),
        "entry_prices_verified_against_bar_opens": bool(open_match.all()),
        "directions_are_long_or_short": bool(trade_log["direction"].isin((-1, 1)).all()),
        "gross_r_finite": bool(np.isfinite(trade_log["gross_r"]).all()),
        "one_position_no_overlap_per_strategy": bool(per_strategy_no_overlap),
        "holding_within_cap": bool((trade_log["holding_minutes"] <= cfg.max_holding_minutes).all()),
        "all_strategies_scored": set(performance["strategy"].unique())
        == {s.name for s in specs if not simulate_strategy_is_empty(s, universe, cfg)},
    }
    validation_checks = pd.Series(checks, name="passed").rename_axis("check").reset_index()

    base = performance.loc[performance["cost_scenario"].eq("base")]
    summary = pd.Series(
        {
            "strategies_simulated": trade_log["strategy"].nunique(),
            "total_trades": len(trade_log),
            "base_positive_dev_and_val": int(_positive_both_partitions(base)),
            "best_base_dev_mean_net_r": float(
                base.loc[base["research_partition"].eq("Development"), "mean_net_r"].max()
            ),
            "best_base_val_mean_net_r": float(
                base.loc[base["research_partition"].eq("Validation"), "mean_net_r"].max()
            ),
            "all_checks_passed": bool(validation_checks["passed"].all()),
        },
        name="value",
    )
    return StrategyLabResult(
        trade_log=trade_log,
        performance=performance,
        validation_checks=validation_checks,
        summary=summary,
        config=cfg,
    )


def simulate_strategy_is_empty(
    spec: StrategySpec, universe: pd.DataFrame, cfg: StrategyLabConfig
) -> bool:
    """True when a strategy fires on no eligible observation."""

    return not bool((spec.signal(universe, cfg) != 0).any())


def _positive_both_partitions(base_performance: pd.DataFrame) -> int:
    """Count strategies with positive base-scenario mean net R in both partitions."""

    pivot = base_performance.pivot_table(
        index="strategy", columns="research_partition", values="mean_net_r"
    )
    if "Development" not in pivot or "Validation" not in pivot:
        return 0
    return int(((pivot["Development"] > 0) & (pivot["Validation"] > 0)).sum())


def save_strategy_lab_outputs(result: StrategyLabResult, *, project_root: Path) -> pd.DataFrame:
    """Persist strategy-lab outputs; generated artifacts stay outside Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "strategy_lab"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "strategy_lab_trade_log": (
            processed / "strategy_lab_trade_log_gc.parquet",
            result.trade_log,
        ),
        "strategy_lab_performance": (
            processed / "strategy_lab_performance_gc.parquet",
            result.performance,
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
                "reload_row_match": len(reloaded) == len(table),
            }
        )
    result.performance.to_csv(tables / "strategy_lab_performance_gc.csv", index=False)
    return pd.DataFrame.from_records(records)
