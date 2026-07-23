"""Sharpe-ratio diagnostics for the statistical-feature research notebook.

The notebook contains several different research objects, and Sharpe is only
meaningful when an object can be reduced to a time-indexed return stream.  This
module therefore keeps two deliberately separate diagnostics:

* a *feature-spread Sharpe* for Section 7.  Development-fitted feature buckets
  and Development-fitted orientation create an equal-date-weighted top-minus-
  bottom forward-return spread.  It is a screening diagnostic, not executable
  PnL, because the underlying event returns overlap and omit trading costs;
* a *strategy Sharpe* for Section 11.  Sequential, cost-adjusted trade returns
  are summed by New York trading date, zero-trade eligible dates are retained,
  and the resulting daily R stream is annualized.

No Sharpe value in this module changes the notebook's frozen advancement gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
MIN_SHARPE_OBSERVATIONS = 20


def annualized_sharpe_ratio(
    returns: Sequence[float] | np.ndarray | pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_per_period: float = 0.0,
    min_observations: int = MIN_SHARPE_OBSERVATIONS,
) -> float:
    """Return the sample-volatility annualized Sharpe ratio.

    The function uses ``ddof=1`` and returns ``NaN`` for an undersized or
    zero-variance series.  A zero risk-free rate is appropriate for the
    notebook's dimensionless R and tick spread streams; callers working with
    capital returns can provide a per-period risk-free return explicitly.
    """

    values = np.asarray(returns, dtype=np.float64)
    values = values[np.isfinite(values)] - float(risk_free_per_period)
    if len(values) < int(min_observations):
        return float("nan")
    volatility = float(np.std(values, ddof=1))
    if not np.isfinite(volatility) or volatility <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.sqrt(float(periods_per_year)) * np.mean(values) / volatility)


def build_sharpe_applicability_table() -> pd.DataFrame:
    """Document where Sharpe is and is not valid across the notebook."""

    records = [
        ("1-4", "Environment, population, and labels", False, "No return-generating policy; use validation, coverage, and label-integrity metrics."),
        ("5", "Unconditional baseline event study", False, "Overlapping forward outcomes are descriptive labels, not a position or tradable PnL stream."),
        ("6", "Feature engineering", False, "A feature value has no payoff until paired with a direction, sizing rule, holding period, and costs."),
        ("7", "Univariate feature evaluation", True, "Use only as a secondary feature-spread diagnostic: daily top-minus-bottom forward-return spreads, Development-fitted orientation, Validation confirmation."),
        ("8", "Redundancy and incremental information", False, "Partial rank IC measures incremental information, not returns; retain IC and correlation diagnostics."),
        ("9", "Multivariate expansion models", False, "The target is unsigned future range; IC, AUC, Brier score, and calibration are appropriate, while Sharpe would mislabel volatility forecasts as PnL."),
        ("10", "Signal construction", False, "Rules exist but realized sequential returns do not; defer Sharpe until the cost-aware backtest."),
        ("11", "Independent sequential backtest", True, "Primary valid use: annualized Sharpe of daily net R, including zero-trade eligible dates, reported by variant, partition, and cost scenario."),
        ("12", "Hybrid integration event study", False, "Events overlap and the robust-R comparison is not a sequential portfolio return stream; use date-block improvement intervals and sample floors."),
        ("12B", "Sizing, exits, and suppression event study", False, "The tests are non-sequential conditional event studies; mean/MAD, effect intervals, retention, and drawdown diagnostics remain the governed measures."),
        ("GARCH", "Volatility modelling", False, "Conditional volatility is a risk forecast, not a return stream; use forecast persistence and regime diagnostics until tied to a frozen strategy."),
    ]
    return pd.DataFrame(
        records,
        columns=["section", "research_object", "sharpe_applicable", "quant_rationale"],
    )


def _fit_bucket_edges(values: np.ndarray, bucket_count: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.array([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, int(bucket_count) + 1)[1:-1]
    return np.unique(np.quantile(finite, quantiles)).astype(np.float64)


def build_feature_spread_sharpe(
    frame: pd.DataFrame,
    cell_results: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    horizons: Sequence[int] = (5, 15, 30, 60, 120, 180),
    sessions: Sequence[str] = ("London", "New York"),
    partitions: Sequence[str] = ("Development", "Validation"),
    quantile_bucket_count: int = 5,
    min_daily_bucket_observations: int = 10,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Build daily feature-spread Sharpe diagnostics for every numeric feature.

    Bucket edges are fitted once on Development for each feature/session.  The
    sign of the Development directional IC fixes the economic orientation for
    both partitions: positive IC means high-minus-low, negative IC means
    low-minus-high.  Validation therefore remains genuinely out of sample.
    """

    observed_partitions = set(frame["research_partition"].astype(str).unique())
    if not observed_partitions.issubset(set(partitions)):
        raise ValueError(f"feature-spread frame contains locked partitions: {observed_partitions}")

    features = [str(name) for name in feature_names]
    outcome_columns = [f"forward_return_{int(h)}_ticks" for h in horizons]
    required = {
        "trade_date_ny",
        "entry_session",
        "research_partition",
        *features,
        *outcome_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"feature-spread frame is missing columns: {missing[:10]}")

    development_ic = (
        cell_results.loc[
            cell_results["outcome_family"].eq("direction")
            & cell_results["research_partition"].eq("Development"),
            ["feature_name", "horizon_minutes", "session", "daily_ic_mean"],
        ]
        .drop_duplicates(["feature_name", "horizon_minutes", "session"])
        .set_index(["feature_name", "horizon_minutes", "session"])["daily_ic_mean"]
    )

    session_values = frame["entry_session"].astype(str).to_numpy()
    partition_values = frame["research_partition"].astype(str).to_numpy()
    development_mask = partition_values == "Development"
    records: list[dict] = []

    for session in sessions:
        session_mask = session_values == str(session)
        for feature_name in features:
            feature_values = frame[feature_name].to_numpy(dtype=np.float64)
            edges = _fit_bucket_edges(
                feature_values[session_mask & development_mask], quantile_bucket_count
            )
            assigned = np.full(len(frame), -1, dtype=np.int16)
            finite = np.isfinite(feature_values) & session_mask
            assigned[finite] = np.searchsorted(edges, feature_values[finite], side="right")
            observed_buckets = np.unique(assigned[finite & development_mask])
            observed_buckets = observed_buckets[observed_buckets >= 0]
            observed_bucket_count = len(observed_buckets)
            bottom_bucket = int(observed_buckets.min()) if observed_bucket_count >= 2 else None
            top_bucket = int(observed_buckets.max()) if observed_bucket_count >= 2 else None

            for partition in partitions:
                mask = session_mask & (partition_values == str(partition)) & (assigned >= 0)
                sub = frame.loc[mask, ["trade_date_ny", *outcome_columns]].copy()
                sub["bucket_index"] = assigned[mask]
                if sub.empty or observed_bucket_count < 2:
                    grouped = None
                else:
                    grouped = sub.groupby(
                        ["trade_date_ny", "bucket_index"], observed=True, sort=True
                    )[outcome_columns].agg(["mean", "count"])

                for horizon, outcome_column in zip(horizons, outcome_columns, strict=True):
                    dev_ic = float(
                        development_ic.get((feature_name, int(horizon), str(session)), np.nan)
                    )
                    orientation = 1.0 if dev_ic >= 0.0 else -1.0
                    orientation_label = "high_minus_low" if orientation > 0 else "low_minus_high"
                    daily_spread = pd.Series(dtype=np.float64)

                    partition_buckets = (
                        set(grouped.index.get_level_values("bucket_index"))
                        if grouped is not None
                        else set()
                    )
                    if (
                        grouped is not None
                        and bottom_bucket in partition_buckets
                        and top_bucket in partition_buckets
                    ):
                        bottom = grouped.xs(bottom_bucket, level="bucket_index")
                        top = grouped.xs(top_bucket, level="bucket_index")
                        joined = pd.concat(
                            {
                                "bottom_mean": bottom[(outcome_column, "mean")],
                                "bottom_count": bottom[(outcome_column, "count")],
                                "top_mean": top[(outcome_column, "mean")],
                                "top_count": top[(outcome_column, "count")],
                            },
                            axis=1,
                            join="inner",
                        )
                        enough = (
                            joined["bottom_count"].ge(min_daily_bucket_observations)
                            & joined["top_count"].ge(min_daily_bucket_observations)
                        )
                        daily_spread = orientation * (
                            joined.loc[enough, "top_mean"] - joined.loc[enough, "bottom_mean"]
                        )
                        daily_spread = daily_spread.replace([np.inf, -np.inf], np.nan).dropna()

                    records.append(
                        {
                            "feature_name": feature_name,
                            "horizon_minutes": int(horizon),
                            "session": str(session),
                            "research_partition": str(partition),
                            "development_orientation": orientation_label,
                            "development_daily_ic_mean": dev_ic,
                            "bucket_count_observed": int(observed_bucket_count),
                            "trading_dates": int(len(daily_spread)),
                            "mean_daily_spread_ticks": float(daily_spread.mean()) if len(daily_spread) else np.nan,
                            "daily_spread_std_ticks": float(daily_spread.std(ddof=1)) if len(daily_spread) > 1 else np.nan,
                            "annualized_sharpe": annualized_sharpe_ratio(
                                daily_spread,
                                periods_per_year=periods_per_year,
                            ),
                            "metric_status": "screening_diagnostic_not_executable_pnl",
                        }
                    )

    return pd.DataFrame.from_records(records)


def _scenario_cost_ticks(config: object, scenario: str) -> float:
    if scenario == "frictionless":
        return 0.0
    per_side = (
        float(getattr(config, "base_slippage_ticks_per_side"))
        if scenario == "base"
        else float(getattr(config, "pessimistic_slippage_ticks_per_side"))
    )
    return float(getattr(config, "commission_ticks_round_trip")) + 2.0 * per_side


def build_backtest_daily_sharpe(
    trade_log: pd.DataFrame,
    candidates: pd.DataFrame,
    config: object,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the Section 11 strategy on daily net-R streams.

    Daily returns are the sum of sequential trade net R.  Every eligible
    candidate date is present in each variant's calendar, with explicit zeros
    when a gated strategy takes no trade.  This avoids an upward-biased Sharpe
    based only on active days.
    """

    calendar = candidates[["trade_date_ny", "research_partition"]].copy()
    calendar["trade_date_ny"] = pd.to_datetime(calendar["trade_date_ny"]).dt.normalize()
    calendar["research_partition"] = calendar["research_partition"].astype(str)
    calendar = calendar.drop_duplicates().sort_values("trade_date_ny")

    trades = trade_log.copy()
    trades["trade_date_ny"] = pd.to_datetime(trades["trade_date_ny"]).dt.normalize()
    trades["research_partition"] = trades["research_partition"].astype(str)
    variant_columns = ["direction_variant", "gate_variant", "research_partition"]
    summary_records: list[dict] = []
    daily_frames: list[pd.DataFrame] = []

    scenarios = tuple(getattr(config, "cost_scenarios"))
    for scenario in scenarios:
        cost_ticks = _scenario_cost_ticks(config, str(scenario))
        scored = trades.assign(net_r=trades["gross_r"] - cost_ticks / trades["stop_ticks"])
        for keys, group in scored.groupby(variant_columns, sort=False):
            direction, gate, partition = keys
            dates = calendar.loc[
                calendar["research_partition"].eq(str(partition)), "trade_date_ny"
            ]
            daily = group.groupby("trade_date_ny", sort=True)["net_r"].sum().reindex(
                pd.DatetimeIndex(dates), fill_value=0.0
            )
            cumulative = daily.cumsum()
            drawdown = cumulative.cummax() - cumulative
            detail = pd.DataFrame(
                {
                    "cost_scenario": str(scenario),
                    "direction_variant": str(direction),
                    "gate_variant": str(gate),
                    "research_partition": str(partition),
                    "trade_date_ny": daily.index,
                    "daily_net_r": daily.to_numpy(dtype=np.float64),
                    "cumulative_net_r": cumulative.to_numpy(dtype=np.float64),
                    "drawdown_r": drawdown.to_numpy(dtype=np.float64),
                }
            )
            daily_frames.append(detail)
            summary_records.append(
                {
                    "cost_scenario": str(scenario),
                    "direction_variant": str(direction),
                    "gate_variant": str(gate),
                    "research_partition": str(partition),
                    "calendar_days": int(len(daily)),
                    "active_days": int(daily.ne(0.0).sum()),
                    "trades": int(len(group)),
                    "mean_daily_net_r": float(daily.mean()),
                    "daily_net_r_std": float(daily.std(ddof=1)),
                    "annualized_sharpe": annualized_sharpe_ratio(
                        daily,
                        periods_per_year=periods_per_year,
                    ),
                    "total_net_r": float(daily.sum()),
                    "daily_positive_rate": float(daily.gt(0.0).mean()),
                    "best_day_r": float(daily.max()),
                    "worst_day_r": float(daily.min()),
                    "max_daily_drawdown_r": float(drawdown.max()),
                    "metric_status": "sequential_cost_adjusted_strategy_pnl",
                }
            )

    return (
        pd.DataFrame.from_records(summary_records),
        pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
    )


def save_performance_diagnostics(
    *,
    project_root: Path,
    applicability: pd.DataFrame,
    feature_spread_sharpe: pd.DataFrame,
    backtest_sharpe: pd.DataFrame,
    backtest_daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Persist Sharpe diagnostics and verify every artifact reloads."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "performance_diagnostics"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_spread_sharpe": (
            processed / "feature_spread_sharpe_gc.parquet",
            tables / "feature_spread_sharpe_gc.csv",
            feature_spread_sharpe,
        ),
        "backtest_daily_sharpe": (
            processed / "backtest_daily_sharpe_gc.parquet",
            tables / "backtest_daily_sharpe_gc.csv",
            backtest_sharpe,
        ),
        "backtest_daily_returns": (
            processed / "backtest_daily_returns_gc.parquet",
            tables / "backtest_daily_returns_gc.csv",
            backtest_daily_returns,
        ),
    }
    records = []
    for name, (parquet_path, csv_path, table) in outputs.items():
        table.to_parquet(parquet_path, index=False)
        table.to_csv(csv_path, index=False)
        reloaded = pd.read_parquet(parquet_path)
        records.append(
            {
                "output": name,
                "parquet_path": str(parquet_path.relative_to(project_root)),
                "csv_path": str(csv_path.relative_to(project_root)),
                "rows": len(table),
                "columns": table.shape[1],
                "reload_row_match": len(reloaded) == len(table),
                "reload_column_match": list(reloaded.columns) == list(table.columns),
            }
        )

    applicability_path = tables / "sharpe_applicability.csv"
    applicability.to_csv(applicability_path, index=False)
    records.append(
        {
            "output": "sharpe_applicability",
            "parquet_path": "not_applicable",
            "csv_path": str(applicability_path.relative_to(project_root)),
            "rows": len(applicability),
            "columns": applicability.shape[1],
            "reload_row_match": len(pd.read_csv(applicability_path)) == len(applicability),
            "reload_column_match": list(pd.read_csv(applicability_path, nrows=0).columns)
            == list(applicability.columns),
        }
    )
    return pd.DataFrame.from_records(records)
