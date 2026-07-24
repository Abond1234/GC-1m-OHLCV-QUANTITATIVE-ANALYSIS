"""Anti-overfitting evaluation for the strategy laboratory.

A backtested strategy zoo is a multiple-comparisons trap: search enough rules and
the best one looks profitable by luck alone.  This module applies the corrections
that separate a real edge from a fluke, so a rejection (or, someday, a survivor)
can be believed.

Four instruments, each answering a different failure mode:

- **Date-block bootstrap confidence intervals** on mean net R, resampling whole
  New York trading dates (seed and replicate count mirror `baselines`), so
  intra-day dependence does not masquerade as significance.
- **Benjamini-Hochberg q-values** across the pre-registered family, controlling
  the false-discovery rate over the number of rules tried.
- **The deflated Sharpe ratio** (Bailey & Lopez de Prado): the probability that a
  strategy's Sharpe exceeds the Sharpe that the *best of N trials* would reach
  under the null, correcting for trial count, sample length, skew, and kurtosis.
- **The probability of backtest overfitting** via combinatorially symmetric
  cross-validation (CSCV): across every in-sample/out-of-sample block split, how
  often does the in-sample-best strategy land below the out-of-sample median.

Governance: selection statistics use Development, confirmation uses Validation,
and the deflation/CSCV run on the combined research sample.  The Final-test
partition is never read here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import e
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .feature_evaluation import EVALUATION_PARTITIONS, benjamini_hochberg_q_values

STRATEGY_EVALUATION_SEED = 20260714
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
EULER_MASCHERONI = 0.5772156649015329
CSCV_BLOCKS = 10
BENCHMARK_FAMILY = "benchmark"


@dataclass(frozen=True)
class EvaluationConfig:
    """Frozen evaluation contract."""

    random_seed: int = STRATEGY_EVALUATION_SEED
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL
    cscv_blocks: int = CSCV_BLOCKS
    cost_scenario: str = "base"
    tick_size: float = 0.10
    commission_ticks_round_trip: float = 0.6
    base_slippage_ticks_per_side: float = 1.0
    pessimistic_slippage_ticks_per_side: float = 2.0
    # Advancement gate (all must hold to advance a strategy).
    max_bh_q: float = 0.10
    min_deflated_sharpe: float = 0.95

    def scenario_cost_ticks(self, scenario: str) -> float:
        if scenario == "frictionless":
            return 0.0
        per_side = (
            self.base_slippage_ticks_per_side
            if scenario == "base"
            else self.pessimistic_slippage_ticks_per_side
        )
        return self.commission_ticks_round_trip + 2.0 * per_side


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------
def add_net_r(trade_log: pd.DataFrame, cfg: EvaluationConfig, scenario: str) -> pd.DataFrame:
    cost_ticks = cfg.scenario_cost_ticks(scenario)
    return trade_log.assign(net_r=trade_log["gross_r"] - cost_ticks / trade_log["stop_ticks"])


def daily_net_r(scored: pd.DataFrame, strategy: str, partition: str | None) -> np.ndarray:
    """Summed net R per New York date for one strategy (a returns series)."""

    mask = scored["strategy"].eq(strategy)
    if partition is not None:
        mask &= scored["research_partition"].eq(partition)
    group = scored.loc[mask]
    if group.empty:
        return np.array([], dtype=np.float64)
    daily = group.groupby("trade_date_ny")["net_r"].sum()
    return daily.to_numpy(dtype=np.float64)


def daily_returns_matrix(scored: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Dates x strategies matrix of summed net R over the research sample.

    A date on which a strategy holds no position contributes zero return, so the
    matrix is dense and the columns share one calendar for CSCV.
    """

    frame = (
        scored[scored["strategy"].isin(strategies)]
        .groupby(["trade_date_ny", "strategy"])["net_r"]
        .sum()
        .unstack("strategy")
        .reindex(columns=strategies)
    )
    return frame.fillna(0.0).sort_index()


# ---------------------------------------------------------------------------
# Bootstrap, Sharpe, deflated Sharpe
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(
    daily: np.ndarray, *, seed: int, cfg: EvaluationConfig
) -> tuple[float, float]:
    values = np.asarray(daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    estimates = np.empty(cfg.bootstrap_replicates, dtype=np.float64)
    batch = 200
    for start in range(0, cfg.bootstrap_replicates, batch):
        stop = min(start + batch, cfg.bootstrap_replicates)
        draw = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[draw].mean(axis=1)
    alpha = (1.0 - cfg.confidence_level) / 2.0
    lo, hi = np.quantile(estimates, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def one_sided_t_pvalue(daily: np.ndarray) -> float:
    """P(mean > 0) test: H0 mean <= 0.  Returns the one-sided p-value."""

    values = np.asarray(daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    mean = values.mean()
    std = values.std(ddof=1)
    if std == 0:
        return 0.0 if mean > 0 else 1.0
    t_stat = mean / (std / np.sqrt(len(values)))
    return float(stats.t.sf(t_stat, df=len(values) - 1))


def daily_sharpe(daily: np.ndarray) -> float:
    values = np.asarray(daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    return float(values.mean() / std) if std > 0 else np.nan


def deflated_sharpe_ratio(
    daily: np.ndarray, *, n_trials: int, trial_sharpe_variance: float
) -> dict:
    """Probability the true Sharpe exceeds the best-of-N-trials null Sharpe."""

    values = np.asarray(daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3 or n_trials < 2 or not np.isfinite(trial_sharpe_variance):
        return {
            "sharpe": np.nan,
            "sharpe0": np.nan,
            "deflated_sharpe": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "observations": n,
        }
    std = values.std(ddof=1)
    if std == 0:
        return {
            "sharpe": np.nan,
            "sharpe0": np.nan,
            "deflated_sharpe": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "observations": n,
        }
    sr = values.mean() / std
    skew = float(stats.skew(values, bias=True))
    kurt = float(stats.kurtosis(values, fisher=False, bias=True))
    sharpe0 = np.sqrt(trial_sharpe_variance) * (
        (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    )
    denominator = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2)
    if not np.isfinite(denominator) or denominator <= 0:
        return {
            "sharpe": sr,
            "sharpe0": float(sharpe0),
            "deflated_sharpe": np.nan,
            "skew": skew,
            "kurtosis": kurt,
            "observations": n,
        }
    dsr = float(stats.norm.cdf((sr - sharpe0) * np.sqrt(n - 1) / denominator))
    return {
        "sharpe": float(sr),
        "sharpe0": float(sharpe0),
        "deflated_sharpe": dsr,
        "skew": skew,
        "kurtosis": kurt,
        "observations": n,
    }


# ---------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------
def _block_sharpe(matrix: np.ndarray) -> np.ndarray:
    """Per-column Sharpe over the rows of a block, zero when a column is flat."""

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])
    sharpe = np.zeros_like(mean)
    nonzero = std > 0
    sharpe[nonzero] = mean[nonzero] / std[nonzero]
    return sharpe


def probability_of_backtest_overfitting(returns_matrix: pd.DataFrame, *, n_blocks: int) -> dict:
    """CSCV estimate of P(in-sample-best strategy is below the OOS median)."""

    matrix = returns_matrix.to_numpy(dtype=np.float64)
    n_periods, n_strategies = matrix.shape
    if n_strategies < 2 or n_periods < n_blocks:
        return {
            "pbo": np.nan,
            "combinations": 0,
            "n_blocks": n_blocks,
            "median_logit": np.nan,
            "prob_oos_best_positive": np.nan,
        }
    blocks = np.array_split(np.arange(n_periods), n_blocks)
    logits: list[float] = []
    oos_best_returns: list[float] = []
    for is_choice in combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[b] for b in is_choice])
        oos_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in is_choice])
        is_perf = _block_sharpe(matrix[is_rows])
        oos_perf = _block_sharpe(matrix[oos_rows])
        best = int(np.argmax(is_perf))
        ranks = stats.rankdata(oos_perf)  # 1 = worst ... N = best
        relative = ranks[best] / (n_strategies + 1)
        relative = min(max(relative, 1e-9), 1 - 1e-9)
        logits.append(float(np.log(relative / (1.0 - relative))))
        oos_best_returns.append(float(matrix[oos_rows, best].mean()))
    logits_arr = np.array(logits)
    return {
        "pbo": float(np.mean(logits_arr < 0.0)),
        "combinations": len(logits_arr),
        "n_blocks": n_blocks,
        "median_logit": float(np.median(logits_arr)),
        "prob_oos_best_positive": float(np.mean(np.array(oos_best_returns) > 0.0)),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    strategy_stats: pd.DataFrame
    pbo: pd.Series
    verdicts: pd.DataFrame
    summary: pd.Series
    config: EvaluationConfig = field(default_factory=EvaluationConfig)


def evaluate_strategies(
    trade_log: pd.DataFrame, config: EvaluationConfig | None = None
) -> EvaluationResult:
    """Full anti-overfitting evaluation of the strategy-lab trade log."""

    cfg = config or EvaluationConfig()
    observed = set(trade_log["research_partition"].astype(str).unique())
    if not observed.issubset(set(EVALUATION_PARTITIONS)):
        raise ValueError(f"trade log contains locked partitions: {observed}")

    scored = add_net_r(trade_log, cfg, cfg.cost_scenario)
    families = scored[["strategy", "family"]].drop_duplicates().set_index("strategy")["family"]
    strategies = list(families.index)
    trial_strategies = [s for s in strategies if families[s] != BENCHMARK_FAMILY]

    # Trial-Sharpe variance uses only the searched (non-benchmark) strategies.
    combined_sharpes = {s: daily_sharpe(daily_net_r(scored, s, None)) for s in trial_strategies}
    trial_sr_values = np.array([v for v in combined_sharpes.values() if np.isfinite(v)])
    trial_sr_variance = float(trial_sr_values.var(ddof=1)) if len(trial_sr_values) > 1 else np.nan
    n_trials = len(trial_strategies)

    records = []
    for index, strategy in enumerate(strategies):
        dev = daily_net_r(scored, strategy, "Development")
        val = daily_net_r(scored, strategy, "Validation")
        combined = daily_net_r(scored, strategy, None)
        dev_lo, dev_hi = bootstrap_mean_ci(dev, seed=cfg.random_seed + index, cfg=cfg)
        val_lo, val_hi = bootstrap_mean_ci(val, seed=cfg.random_seed + 500 + index, cfg=cfg)
        dsr = deflated_sharpe_ratio(
            combined, n_trials=n_trials, trial_sharpe_variance=trial_sr_variance
        )
        records.append(
            {
                "strategy": strategy,
                "family": families[strategy],
                "is_trial": strategy in trial_strategies,
                "dev_trading_dates": len(dev),
                "dev_mean_net_r": float(dev.mean()) if len(dev) else np.nan,
                "dev_daily_sharpe": daily_sharpe(dev),
                "dev_ci_low": dev_lo,
                "dev_ci_high": dev_hi,
                "dev_t_pvalue": one_sided_t_pvalue(dev),
                "val_trading_dates": len(val),
                "val_mean_net_r": float(val.mean()) if len(val) else np.nan,
                "val_daily_sharpe": daily_sharpe(val),
                "val_ci_low": val_lo,
                "val_ci_high": val_hi,
                "combined_daily_sharpe": dsr["sharpe"],
                "deflated_sharpe": dsr["deflated_sharpe"],
                "return_skew": dsr["skew"],
                "return_kurtosis": dsr["kurtosis"],
            }
        )
    stats_frame = pd.DataFrame.from_records(records)

    # Benjamini-Hochberg across the searched family only (Development p-values).
    trial_mask = stats_frame["is_trial"].to_numpy()
    q_values = np.full(len(stats_frame), np.nan)
    q_values[trial_mask] = benjamini_hochberg_q_values(
        stats_frame.loc[trial_mask, "dev_t_pvalue"].to_numpy()
    )
    stats_frame["dev_bh_q"] = q_values

    # Advancement gate.
    def _verdict(row: pd.Series) -> str:
        if not row["is_trial"]:
            return "BENCHMARK"
        advances = (
            np.isfinite(row["dev_bh_q"])
            and row["dev_bh_q"] <= cfg.max_bh_q
            and row["dev_mean_net_r"] > 0
            and row["val_mean_net_r"] > 0
            and np.isfinite(row["val_ci_low"])
            and row["val_ci_low"] > 0
            and np.isfinite(row["deflated_sharpe"])
            and row["deflated_sharpe"] >= cfg.min_deflated_sharpe
        )
        return "ADVANCE" if advances else "REJECTED_NO_EDGE"

    stats_frame["verdict"] = stats_frame.apply(_verdict, axis=1)

    matrix = daily_returns_matrix(scored, trial_strategies)
    pbo = pd.Series(
        probability_of_backtest_overfitting(matrix, n_blocks=cfg.cscv_blocks), name="value"
    )

    verdicts = stats_frame[
        [
            "strategy",
            "family",
            "dev_mean_net_r",
            "val_mean_net_r",
            "dev_bh_q",
            "deflated_sharpe",
            "verdict",
        ]
    ].copy()

    best_trial = stats_frame.loc[trial_mask].sort_values("combined_daily_sharpe", ascending=False)
    summary = pd.Series(
        {
            "strategies_evaluated": len(stats_frame),
            "trials": n_trials,
            "advanced": int((stats_frame["verdict"] == "ADVANCE").sum()),
            "min_dev_bh_q": float(np.nanmin(stats_frame.loc[trial_mask, "dev_bh_q"])),
            "max_deflated_sharpe": float(np.nanmax(stats_frame.loc[trial_mask, "deflated_sharpe"])),
            "best_trial_by_sharpe": best_trial["strategy"].iloc[0] if len(best_trial) else "",
            "probability_of_backtest_overfitting": float(pbo["pbo"]),
            "prob_oos_best_positive": float(pbo["prob_oos_best_positive"]),
        },
        name="value",
    )
    return EvaluationResult(
        strategy_stats=stats_frame, pbo=pbo, verdicts=verdicts, summary=summary, config=cfg
    )


def save_strategy_evaluation_outputs(
    result: EvaluationResult, *, project_root: Path
) -> pd.DataFrame:
    """Persist evaluation outputs; generated artifacts stay outside Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "strategy_lab"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    path = processed / "strategy_lab_evaluation_gc.parquet"
    result.strategy_stats.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path, engine="pyarrow")
    result.strategy_stats.to_csv(tables / "strategy_lab_evaluation_gc.csv", index=False)
    result.verdicts.to_csv(tables / "strategy_lab_verdicts_gc.csv", index=False)
    return pd.DataFrame.from_records(
        [
            {
                "output": "strategy_lab_evaluation",
                "path": str(path.relative_to(project_root)),
                "rows": len(result.strategy_stats),
                "reload_row_match": len(reloaded) == len(result.strategy_stats),
            }
        ]
    )
