"""Dependence-aware descriptive baselines for independent GC research.

Section 5 summarizes overlapping event-level outcomes; it does not simulate
trades.  All calculations respect the per-horizon availability flags built in
Section 4.  Confidence intervals resample New York trading dates (or daily
aggregates), never individual minute observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .labels import FORWARD_HORIZONS_MINUTES, GC_TICK_SIZE

BASELINE_RANDOM_SEED = 20260714
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
COST_THRESHOLDS_TICKS = (0, 1, 2, 3, 4, 5, 10)
SESSION_ORDER = ("London", "New York")
PARTITION_ORDER = ("Development", "Validation", "Final test")
WEEKDAY_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

SUMMARY_DIMENSIONS = (
    "analysis_type",
    "outcome_family",
    "outcome_name",
    "horizon_minutes",
    "session",
    "time_of_day_bin",
    "minutes_since_session_open_bin",
    "day_of_week",
    "year",
    "research_partition",
    "direction",
    "cost_threshold_ticks",
    "metric_name",
    "metric_value",
    "observation_count",
    "trading_date_count",
    "availability_rate",
    "confidence_interval_lower",
    "confidence_interval_upper",
)

WIDE_METRICS = (
    "available_observation_count",
    "availability_rate",
    "mean",
    "median",
    "standard_deviation",
    "mean_absolute_value",
    "minimum",
    "maximum",
    "p01",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "positive_rate",
    "negative_rate",
    "zero_rate",
    "interquartile_range",
    "median_absolute_deviation",
    "skewness",
    "excess_kurtosis",
    "trimmed_mean",
)


@dataclass(frozen=True)
class OutcomeSpec:
    template: str
    family: str
    name: str
    direction: str | None = None

    def column(self, horizon: int) -> str:
        return self.template.format(h=horizon)


ALL_OUTCOME_SPECS = (
    OutcomeSpec("forward_return_{h}_ticks", "forward_return", "forward_return_ticks", "long"),
    OutcomeSpec("forward_return_{h}_bps", "forward_return", "forward_return_bps", "long"),
    OutcomeSpec("forward_return_{h}_atr", "forward_return", "forward_return_atr", "long"),
    OutcomeSpec("direction_label_{h}", "classification", "direction_label", "long"),
    OutcomeSpec("mfe_long_{h}_ticks", "excursion", "mfe_long_ticks", "long"),
    OutcomeSpec("mae_long_{h}_ticks", "excursion", "mae_long_ticks", "long"),
    OutcomeSpec("mfe_short_{h}_ticks", "excursion", "mfe_short_ticks", "short"),
    OutcomeSpec("mae_short_{h}_ticks", "excursion", "mae_short_ticks", "short"),
    OutcomeSpec("mfe_long_{h}_atr", "excursion", "mfe_long_atr", "long"),
    OutcomeSpec("mae_long_{h}_atr", "excursion", "mae_long_atr", "long"),
    OutcomeSpec("mfe_short_{h}_atr", "excursion", "mfe_short_atr", "short"),
    OutcomeSpec("mae_short_{h}_atr", "excursion", "mae_short_atr", "short"),
    OutcomeSpec(
        "time_to_mfe_long_{h}_minutes", "time_to_extreme", "time_to_mfe_long_minutes", "long"
    ),
    OutcomeSpec(
        "time_to_mae_long_{h}_minutes", "time_to_extreme", "time_to_mae_long_minutes", "long"
    ),
    OutcomeSpec(
        "time_to_mfe_short_{h}_minutes", "time_to_extreme", "time_to_mfe_short_minutes", "short"
    ),
    OutcomeSpec(
        "time_to_mae_short_{h}_minutes", "time_to_extreme", "time_to_mae_short_minutes", "short"
    ),
    OutcomeSpec(
        "time_to_mfe_long_{h}_fraction", "time_to_extreme", "time_to_mfe_long_fraction", "long"
    ),
    OutcomeSpec(
        "time_to_mae_long_{h}_fraction", "time_to_extreme", "time_to_mae_long_fraction", "long"
    ),
    OutcomeSpec(
        "time_to_mfe_short_{h}_fraction", "time_to_extreme", "time_to_mfe_short_fraction", "short"
    ),
    OutcomeSpec(
        "time_to_mae_short_{h}_fraction", "time_to_extreme", "time_to_mae_short_fraction", "short"
    ),
    OutcomeSpec("future_range_{h}_ticks", "range", "future_range_ticks"),
    OutcomeSpec("future_range_{h}_atr", "range", "future_range_atr"),
    OutcomeSpec(
        "future_realized_volatility_{h}_bps", "volatility", "future_realized_volatility_bps"
    ),
    OutcomeSpec("expansion_label_{h}", "classification", "expansion_label"),
)

KEY_GROUP_SPECS = tuple(
    spec
    for spec in ALL_OUTCOME_SPECS
    if spec.name
    in {
        "forward_return_ticks",
        "forward_return_bps",
        "forward_return_atr",
        "mfe_long_ticks",
        "mae_long_ticks",
        "mfe_short_ticks",
        "mae_short_ticks",
        "future_range_ticks",
        "future_range_atr",
        "future_realized_volatility_bps",
    }
)


@dataclass
class BaselineBuildResult:
    summary: pd.DataFrame
    cost_thresholds: pd.DataFrame
    validation: pd.DataFrame
    unconditional: pd.DataFrame
    session: pd.DataFrame
    time_of_day: pd.DataFrame
    weekday: pd.DataFrame
    directional_symmetry: pd.DataFrame
    year_partition_stability: pd.DataFrame
    headline_uncertainty: pd.DataFrame
    year_coverage: pd.DataFrame
    benchmark_definition: dict
    report_markdown: str
    ready: bool


def _to_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def _distribution_metrics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return {
            name: (0 if name == "available_observation_count" else np.nan) for name in WIDE_METRICS
        }
    quantiles = np.quantile(values, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    mean = float(values.mean())
    centered = values - mean
    variance = float(np.mean(centered**2))
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    skew = float(np.mean(centered**3) / variance**1.5) if n > 2 and variance > 0 else 0.0
    kurt = float(np.mean(centered**4) / variance**2 - 3.0) if n > 3 and variance > 0 else 0.0
    median = float(quantiles[4])
    trim = int(np.floor(0.01 * n))
    if trim:
        partitioned = np.partition(values, (trim, n - trim - 1))[trim : n - trim]
        trimmed_mean = float(partitioned.mean())
    else:
        trimmed_mean = mean
    return {
        "available_observation_count": int(n),
        "availability_rate": np.nan,
        "mean": mean,
        "median": median,
        "standard_deviation": std,
        "mean_absolute_value": float(np.mean(np.abs(values))),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p10": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "p50": median,
        "p75": float(quantiles[5]),
        "p90": float(quantiles[6]),
        "p95": float(quantiles[7]),
        "p99": float(quantiles[8]),
        "positive_rate": float(np.mean(values > 0)),
        "negative_rate": float(np.mean(values < 0)),
        "zero_rate": float(np.mean(values == 0)),
        "interquartile_range": float(quantiles[5] - quantiles[3]),
        "median_absolute_deviation": float(np.median(np.abs(values - median))),
        "skewness": skew,
        "excess_kurtosis": kurt,
        "trimmed_mean": trimmed_mean,
    }


def _group_positions(frame: pd.DataFrame, columns: Sequence[str]) -> list[tuple[tuple, np.ndarray]]:
    if not columns:
        return [((), np.arange(len(frame), dtype=np.int64))]
    grouped = frame.groupby(list(columns), observed=True, sort=True, dropna=False).indices
    result = []
    for key, positions in grouped.items():
        result.append(
            (key if isinstance(key, tuple) else (key,), np.asarray(positions, dtype=np.int64))
        )
    return result


def summarize_outcomes(
    frame: pd.DataFrame,
    *,
    specs: Sequence[OutcomeSpec],
    group_columns: Sequence[str] = (),
    analysis_type: str,
    horizons: Sequence[int] = FORWARD_HORIZONS_MINUTES,
) -> pd.DataFrame:
    """Apply one stable aggregation contract to one or more outcome families."""

    groups = _group_positions(frame, group_columns)
    dates = pd.to_datetime(frame["trade_date_ny"]).to_numpy(dtype="datetime64[D]")
    records: list[dict] = []
    for horizon in horizons:
        flag = frame[f"label_available_{horizon}"].fillna(False).to_numpy(dtype=bool)
        for spec in specs:
            column = spec.column(horizon)
            values = _to_float(frame[column])
            finite = np.isfinite(values) & flag
            for keys, positions in groups:
                group_record = dict(zip(group_columns, keys, strict=True))
                valid_positions = positions[finite[positions]]
                metrics = _distribution_metrics(values[valid_positions])
                observation_count = int(len(positions))
                metrics["availability_rate"] = (
                    metrics["available_observation_count"] / observation_count
                    if observation_count
                    else np.nan
                )
                records.append(
                    {
                        "analysis_type": analysis_type,
                        "outcome_family": spec.family,
                        "outcome_name": spec.name,
                        "direction": spec.direction,
                        "horizon_minutes": int(horizon),
                        **group_record,
                        "observation_count": observation_count,
                        "trading_date_count": int(np.unique(dates[positions]).size),
                        **metrics,
                    }
                )
    return pd.DataFrame.from_records(records)


def _bootstrap_mean_ci(
    daily_values: np.ndarray,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    values = np.asarray(daily_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    batch = 200
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        draw = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[draw].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return tuple(float(x) for x in np.quantile(estimates, [alpha, 1.0 - alpha]))


def build_headline_uncertainty(frame: pd.DataFrame) -> pd.DataFrame:
    """Date-block intervals for headline centers, rates, and session differences."""

    records = []
    dates = pd.to_datetime(frame["trade_date_ny"]).dt.normalize()
    sessions = frame["entry_session"].astype(str)
    for h_index, horizon in enumerate(FORWARD_HORIZONS_MINUTES):
        available = frame[f"label_available_{horizon}"].fillna(False).to_numpy(dtype=bool)
        returns = _to_float(frame[f"forward_return_{horizon}_ticks"])
        valid = available & np.isfinite(returns)
        compact = pd.DataFrame(
            {"date": dates[valid], "session": sessions[valid], "return": returns[valid]}
        )
        for session_index, session in enumerate((*SESSION_ORDER, "Combined")):
            selected = (
                compact if session == "Combined" else compact.loc[compact["session"].eq(session)]
            )
            daily = selected.groupby("date", observed=True)["return"].agg(
                mean_return="mean", positive_rate=lambda x: float(np.mean(x > 0))
            )
            for metric_index, metric in enumerate(("mean_return", "positive_rate")):
                estimate = float(daily[metric].mean())
                low, high = _bootstrap_mean_ci(
                    daily[metric].to_numpy(),
                    seed=BASELINE_RANDOM_SEED + 100 * h_index + 10 * session_index + metric_index,
                )
                records.append(
                    {
                        "analysis_type": "headline_date_block_bootstrap",
                        "horizon_minutes": horizon,
                        "session": session,
                        "metric_name": metric,
                        "metric_value": estimate,
                        "confidence_interval_lower": low,
                        "confidence_interval_upper": high,
                        "trading_date_count": len(daily),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "random_seed": BASELINE_RANDOM_SEED,
                    }
                )
        daily_session = (
            compact.groupby(["date", "session"], observed=True)["return"]
            .agg(mean_return="mean", positive_rate=lambda x: float(np.mean(x > 0)))
            .unstack("session")
        )
        for metric_index, metric in enumerate(("mean_return", "positive_rate")):
            paired = daily_session[metric].dropna(subset=list(SESSION_ORDER))
            difference = paired["London"].to_numpy() - paired["New York"].to_numpy()
            low, high = _bootstrap_mean_ci(
                difference,
                seed=BASELINE_RANDOM_SEED + 1_000 + 10 * h_index + metric_index,
            )
            records.append(
                {
                    "analysis_type": "session_difference_date_block_bootstrap",
                    "horizon_minutes": horizon,
                    "session": "London minus New York",
                    "metric_name": metric,
                    "metric_value": float(difference.mean()),
                    "confidence_interval_lower": low,
                    "confidence_interval_upper": high,
                    "trading_date_count": len(difference),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "random_seed": BASELINE_RANDOM_SEED,
                }
            )
    return pd.DataFrame.from_records(records)


def _add_time_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["entry_timestamp_ny", "entry_session"]].copy()
    timestamp = pd.to_datetime(result["entry_timestamp_ny"])
    minute_of_day = timestamp.dt.hour.to_numpy() * 60 + timestamp.dt.minute.to_numpy()
    bin_start = (minute_of_day // 15) * 15
    result["time_of_day_bin"] = [f"{m // 60:02d}:{m % 60:02d}" for m in bin_start]
    session_start = np.where(result["entry_session"].astype(str).eq("London"), 180, 420)
    relative = minute_of_day - session_start
    relative_start = (relative // 15) * 15
    result["minutes_since_session_open_bin"] = [f"{m:03d}-{m + 14:03d}" for m in relative_start]
    return result[["time_of_day_bin", "minutes_since_session_open_bin"]]


def build_directional_symmetry(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    group_definitions: list[tuple[str, str, np.ndarray]] = [
        ("full", "Combined", np.ones(len(frame), dtype=bool))
    ]
    session_values = frame["entry_session"].astype(str).to_numpy()
    for session in SESSION_ORDER:
        group_definitions.append(("session", session, session_values == session))
    years = pd.to_datetime(frame["entry_timestamp_ny"]).dt.year.to_numpy()
    for year in np.unique(years):
        group_definitions.append(("year", str(int(year)), years == year))
    partitions = frame["research_partition"].astype(str).to_numpy()
    for partition in PARTITION_ORDER:
        group_definitions.append(("partition", partition, partitions == partition))

    dates = pd.to_datetime(frame["trade_date_ny"]).to_numpy(dtype="datetime64[D]")
    for horizon in FORWARD_HORIZONS_MINUTES:
        available = frame[f"label_available_{horizon}"].fillna(False).to_numpy(dtype=bool)
        returns = _to_float(frame[f"forward_return_{horizon}_ticks"])
        mfe_long = _to_float(frame[f"mfe_long_{horizon}_ticks"])
        mae_long = _to_float(frame[f"mae_long_{horizon}_ticks"])
        mfe_short = _to_float(frame[f"mfe_short_{horizon}_ticks"])
        mae_short = _to_float(frame[f"mae_short_{horizon}_ticks"])
        for grouping, value, selected in group_definitions:
            valid = selected & available & np.isfinite(returns)
            x = returns[valid]
            if not len(x):
                continue
            p01, p05, p95, p99 = np.quantile(x, [0.01, 0.05, 0.95, 0.99])
            upper_tail = x[x >= p95]
            lower_tail = x[x <= p05]
            record = {
                "analysis_type": "directional_symmetry",
                "grouping": grouping,
                "group_value": value,
                "session": value
                if grouping == "session"
                else ("Combined" if grouping == "full" else None),
                "year": int(value) if grouping == "year" else pd.NA,
                "research_partition": value if grouping == "partition" else None,
                "horizon_minutes": horizon,
                "observation_count": int(selected.sum()),
                "available_observation_count": int(valid.sum()),
                "trading_date_count": int(np.unique(dates[valid]).size),
                "availability_rate": float(valid.sum() / selected.sum()),
                "mean_long_return_ticks": float(x.mean()),
                "mean_short_return_ticks": float((-x).mean()),
                "long_positive_rate": float(np.mean(x > 0)),
                "short_positive_rate": float(np.mean(x < 0)),
                "zero_rate": float(np.mean(x == 0)),
                "mean_long_mfe_ticks": float(np.nanmean(mfe_long[valid])),
                "mean_long_mae_ticks": float(np.nanmean(mae_long[valid])),
                "mean_short_mfe_ticks": float(np.nanmean(mfe_short[valid])),
                "mean_short_mae_ticks": float(np.nanmean(mae_short[valid])),
                "p95_ticks": float(p95),
                "absolute_p05_ticks": float(abs(p05)),
                "p99_ticks": float(p99),
                "absolute_p01_ticks": float(abs(p01)),
                "p95_tail_ratio": float(p95 / abs(p05)) if p05 else np.nan,
                "p99_tail_ratio": float(p99 / abs(p01)) if p01 else np.nan,
                "mean_upper_tail_ticks": float(upper_tail.mean()),
                "absolute_mean_lower_tail_ticks": float(abs(lower_tail.mean())),
                "tail_magnitude_ratio": float(upper_tail.mean() / abs(lower_tail.mean()))
                if lower_tail.mean()
                else np.nan,
                "mfe_asymmetry_ticks": float(np.nanmean(mfe_long[valid] - mfe_short[valid])),
                "mae_asymmetry_ticks": float(np.nanmean(mae_long[valid] - mae_short[valid])),
                "long_mfe_to_mae_ratio": float(
                    np.nanmean(mfe_long[valid]) / np.nanmean(mae_long[valid])
                )
                if np.nanmean(mae_long[valid])
                else np.nan,
                "short_mfe_to_mae_ratio": float(
                    np.nanmean(mfe_short[valid]) / np.nanmean(mae_short[valid])
                )
                if np.nanmean(mae_short[valid])
                else np.nan,
            }
            records.append(record)
    return pd.DataFrame.from_records(records)


def build_cost_thresholds(
    frame: pd.DataFrame,
    *,
    horizons: Sequence[int] = FORWARD_HORIZONS_MINUTES,
) -> pd.DataFrame:
    records = []
    dates = pd.to_datetime(frame["trade_date_ny"]).to_numpy(dtype="datetime64[D]")
    sessions = frame["entry_session"].astype(str).to_numpy()
    for horizon in horizons:
        available = frame[f"label_available_{horizon}"].fillna(False).to_numpy(dtype=bool)
        returns = _to_float(frame[f"forward_return_{horizon}_ticks"])
        mfe_long = _to_float(frame[f"mfe_long_{horizon}_ticks"])
        mae_long = _to_float(frame[f"mae_long_{horizon}_ticks"])
        mfe_short = _to_float(frame[f"mfe_short_{horizon}_ticks"])
        mae_short = _to_float(frame[f"mae_short_{horizon}_ticks"])
        ranges = _to_float(frame[f"future_range_{horizon}_ticks"])
        for session in (*SESSION_ORDER, "Combined"):
            selected = (
                np.ones(len(frame), dtype=bool) if session == "Combined" else sessions == session
            )
            valid = selected & available & np.isfinite(returns)
            x = returns[valid]
            for threshold in COST_THRESHOLDS_TICKS:
                abs_excess = np.abs(x) - threshold
                beyond = abs_excess > 0
                upper = x > threshold
                lower = x < -threshold
                records.append(
                    {
                        "analysis_type": "cost_threshold_sensitivity",
                        "outcome_family": "movement_hurdle",
                        "horizon_minutes": horizon,
                        "session": session,
                        "cost_threshold_ticks": threshold,
                        "observation_count": int(selected.sum()),
                        "available_observation_count": int(valid.sum()),
                        "trading_date_count": int(np.unique(dates[valid]).size),
                        "availability_rate": float(valid.sum() / selected.sum()),
                        "probability_return_above_positive_threshold": float(np.mean(upper)),
                        "probability_return_below_negative_threshold": float(np.mean(lower)),
                        "probability_absolute_return_above_threshold": float(np.mean(beyond)),
                        "mean_return_above_positive_threshold_ticks": float(x[upper].mean())
                        if upper.any()
                        else np.nan,
                        "mean_absolute_return_below_negative_threshold_ticks": float(
                            abs(x[lower].mean())
                        )
                        if lower.any()
                        else np.nan,
                        "mean_excess_movement_beyond_threshold_ticks": float(
                            abs_excess[beyond].mean()
                        )
                        if beyond.any()
                        else np.nan,
                        "median_excess_movement_beyond_threshold_ticks": float(
                            np.median(abs_excess[beyond])
                        )
                        if beyond.any()
                        else np.nan,
                        "long_cost_hurdle_exceedance_rate": float(np.mean(upper)),
                        "short_cost_hurdle_exceedance_rate": float(np.mean(lower)),
                        "mfe_long_exceedance_rate": float(np.mean(mfe_long[valid] > threshold)),
                        "mae_long_exceedance_rate": float(np.mean(mae_long[valid] > threshold)),
                        "mfe_short_exceedance_rate": float(np.mean(mfe_short[valid] > threshold)),
                        "mae_short_exceedance_rate": float(np.mean(mae_short[valid] > threshold)),
                        "future_range_exceedance_rate": float(np.mean(ranges[valid] > threshold)),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_year_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(frame["trade_date_ny"]).dt.normalize()
    coverage = (
        pd.DataFrame({"year": dates.dt.year, "date": dates})
        .groupby("year", observed=True)["date"]
        .agg(first_entry_date="min", last_entry_date="max", trading_date_count="nunique")
        .reset_index()
    )
    max_date = dates.max()
    min_date = dates.min()
    coverage["coverage_status"] = "complete observed year"
    coverage.loc[coverage["year"].eq(min_date.year), "coverage_status"] = (
        "partial from dataset start"
    )
    coverage.loc[coverage["year"].eq(max_date.year), "coverage_status"] = (
        f"partial through {max_date.date().isoformat()}"
    )
    return coverage


def _wide_to_normalized(table: pd.DataFrame) -> pd.DataFrame:
    identifiers = [c for c in table.columns if c not in WIDE_METRICS]
    metric_columns = [c for c in WIDE_METRICS if c in table.columns and c != "availability_rate"]
    long = table.melt(
        id_vars=identifiers,
        value_vars=metric_columns,
        var_name="metric_name",
        value_name="metric_value",
    )
    for column in SUMMARY_DIMENSIONS:
        if column not in long.columns:
            long[column] = pd.NA
    return long.loc[:, SUMMARY_DIMENSIONS]


def _uncertainty_to_normalized(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["outcome_family"] = "forward_return"
    result["outcome_name"] = np.where(
        result["metric_name"].eq("mean_return"), "forward_return_ticks", "direction_label"
    )
    result["observation_count"] = pd.NA
    result["availability_rate"] = pd.NA
    for column in SUMMARY_DIMENSIONS:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, SUMMARY_DIMENSIONS]


def _validate_inputs(frame: pd.DataFrame, section_4_ready: bool) -> list[dict]:
    checks: list[dict] = []

    def add(category: str, name: str, passed: bool, details: str, critical: bool = True) -> None:
        checks.append(
            {
                "category": category,
                "check_name": name,
                "critical": critical,
                "passed": bool(passed),
                "details": details,
            }
        )

    add(
        "input_identity",
        "section_4_status_ready",
        section_4_ready,
        f"section_4_ready={section_4_ready}",
    )
    add(
        "input_identity",
        "expected_gc_only_population",
        set(frame["product"].astype(str).unique()) == {"GC"},
        f"products={sorted(frame['product'].astype(str).unique())}",
    )
    add(
        "input_identity",
        "unique_observation_id",
        frame["observation_id"].is_unique,
        f"rows={len(frame):,}; unique={frame['observation_id'].nunique():,}",
    )
    add(
        "input_identity",
        "no_duplicate_observations",
        not frame.duplicated("observation_id").any(),
        f"duplicates={int(frame.duplicated('observation_id').sum()):,}",
    )
    add(
        "input_identity",
        "decision_timestamps_sorted",
        frame["decision_timestamp_utc"].is_monotonic_increasing,
        "UTC decision timestamps are monotonic",
    )
    add(
        "input_identity",
        "entry_timestamps_sorted",
        frame["entry_timestamp_utc"].is_monotonic_increasing,
        "UTC entry timestamps are monotonic",
    )
    poi_columns = [
        c
        for c in frame.columns
        if "poi" in c.lower() or c.lower().startswith(("section6_", "section7_"))
    ]
    add(
        "input_identity",
        "no_poi_derived_inputs",
        not poi_columns,
        f"forbidden_columns={poi_columns}",
    )
    observed_horizons = sorted(
        int(c.rsplit("_", 1)[1]) for c in frame.columns if c.startswith("label_available_")
    )
    add(
        "input_identity",
        "expected_six_horizons",
        tuple(observed_horizons) == FORWARD_HORIZONS_MINUTES,
        f"horizons={observed_horizons}",
    )

    counts = []
    for horizon in FORWARD_HORIZONS_MINUTES:
        available = frame[f"label_available_{horizon}"].fillna(False).to_numpy(dtype=bool)
        counts.append(int(available.sum()))
        return_finite = np.isfinite(_to_float(frame[f"forward_return_{horizon}_ticks"]))
        add(
            "availability",
            f"h{horizon}_available_count_matches_flag",
            int(return_finite.sum()) == int(available.sum()),
            f"flag={available.sum():,}; finite_return={return_finite.sum():,}",
        )
        numeric_columns = [
            spec.column(horizon) for spec in ALL_OUTCOME_SPECS if spec.column(horizon) in frame
        ]
        raw_columns = [
            c
            for c in numeric_columns
            if not c.endswith("_atr") and c != f"expansion_label_{horizon}"
        ]
        raw_finite = np.column_stack([np.isfinite(_to_float(frame[c])) for c in raw_columns])
        add(
            "availability",
            f"h{horizon}_available_labels_finite",
            bool(raw_finite[available].all()),
            f"available={available.sum():,}; raw_columns={len(raw_columns)}",
        )
        add(
            "availability",
            f"h{horizon}_unavailable_labels_null",
            bool((~raw_finite[~available]).all()),
            f"unavailable={(~available).sum():,}",
        )
        nonnegative_columns = [
            f"mfe_long_{horizon}_ticks",
            f"mae_long_{horizon}_ticks",
            f"mfe_short_{horizon}_ticks",
            f"mae_short_{horizon}_ticks",
            f"future_range_{horizon}_ticks",
            f"future_realized_volatility_{horizon}_bps",
        ]
        nonnegative = all(np.nanmin(_to_float(frame[c])) >= 0 for c in nonnegative_columns)
        add(
            "outcome_invariants",
            f"h{horizon}_magnitudes_nonnegative",
            nonnegative,
            f"columns={nonnegative_columns}",
        )
        time_columns = [
            f"time_to_mfe_long_{horizon}_minutes",
            f"time_to_mfe_short_{horizon}_minutes",
            f"time_to_mae_long_{horizon}_minutes",
            f"time_to_mae_short_{horizon}_minutes",
        ]
        times_ok = all(
            np.nanmin(_to_float(frame[c])) >= 1 and np.nanmax(_to_float(frame[c])) <= horizon
            for c in time_columns
        )
        add(
            "outcome_invariants",
            f"h{horizon}_times_within_horizon",
            times_ok,
            f"bounds=1..{horizon}",
        )
        r = _to_float(frame[f"forward_return_{horizon}_ticks"])
        long_mfe = _to_float(frame[f"mfe_long_{horizon}_ticks"])
        long_mae = _to_float(frame[f"mae_long_{horizon}_ticks"])
        short_mfe = _to_float(frame[f"mfe_short_{horizon}_ticks"])
        short_mae = _to_float(frame[f"mae_short_{horizon}_ticks"])
        add(
            "outcome_invariants",
            f"h{horizon}_long_short_excursions_reconcile",
            bool(
                np.allclose(long_mfe[available], short_mae[available])
                and np.allclose(long_mae[available], short_mfe[available])
            ),
            "mfe_long=mae_short and mae_long=mfe_short",
        )
        ranges = _to_float(frame[f"future_range_{horizon}_ticks"])
        add(
            "outcome_invariants",
            f"h{horizon}_range_reconciles",
            bool(np.allclose(ranges[available], (long_mfe + long_mae)[available])),
            "range_ticks=mfe_long_ticks+mae_long_ticks",
        )
        bps = _to_float(frame[f"forward_return_{horizon}_bps"])
        expected_bps = r * GC_TICK_SIZE / _to_float(frame["entry_price"]) * 10_000.0
        add(
            "outcome_invariants",
            f"h{horizon}_tick_bps_reconcile",
            bool(np.allclose(bps[available], expected_bps[available], rtol=1e-10, atol=1e-10)),
            "bps=ticks*tick_size/entry_price*10000",
        )
        atr = _to_float(frame["decision_atr_20m"])
        atr_valid = available & np.isfinite(atr) & (atr > 0)
        return_atr = _to_float(frame[f"forward_return_{horizon}_atr"])
        range_atr = _to_float(frame[f"future_range_{horizon}_atr"])
        add(
            "outcome_invariants",
            f"h{horizon}_tick_atr_reconcile",
            bool(
                np.allclose(
                    return_atr[atr_valid],
                    (r * GC_TICK_SIZE / atr)[atr_valid],
                    rtol=1e-10,
                    atol=1e-10,
                )
            ),
            "return_atr=ticks*tick_size/decision_atr_20m",
        )
        add(
            "outcome_invariants",
            f"h{horizon}_range_atr_reconcile",
            bool(
                np.allclose(
                    range_atr[atr_valid],
                    (ranges * GC_TICK_SIZE / atr)[atr_valid],
                    rtol=1e-10,
                    atol=1e-10,
                )
            ),
            "range_atr=range_ticks*tick_size/decision_atr_20m",
        )
        direction = _to_float(frame[f"direction_label_{horizon}"])
        add(
            "outcome_invariants",
            f"h{horizon}_direction_class_reconciles",
            bool(np.array_equal(direction[available], np.sign(r[available]))),
            "direction_label=sign(forward_return_ticks)",
        )
        expansion = _to_float(frame[f"expansion_label_{horizon}"])
        expansion_values = set(np.unique(expansion[np.isfinite(expansion)]).tolist())
        add(
            "outcome_invariants",
            f"h{horizon}_expansion_class_valid",
            expansion_values.issubset({0.0, 1.0}) and np.isfinite(expansion[atr_valid]).all(),
            f"observed_values={sorted(expansion_values)}; Development-fitted threshold validated in Section 4",
        )
        exits = pd.to_datetime(frame[f"exit_timestamp_ny_{horizon}"])
        forced = pd.to_datetime(frame["forced_exit_timestamp_ny"])
        add(
            "path_boundaries",
            f"h{horizon}_paths_end_by_1530",
            bool((exits[available] <= forced[available]).all()),
            "economic exits do not exceed forced exit",
        )
        entry_dates = pd.to_datetime(frame["entry_timestamp_ny"]).dt.date.to_numpy()
        exit_dates = exits.dt.date.to_numpy()
        add(
            "path_boundaries",
            f"h{horizon}_paths_same_ny_date",
            bool(np.equal(entry_dates[available], exit_dates[available]).all()),
            "entry and economic exit share New York date",
        )
    add(
        "availability",
        "availability_non_increasing",
        all(a >= b for a, b in zip(counts, counts[1:], strict=False)),
        f"counts={counts}",
    )
    return checks


def _validate_aggregations(
    frame: pd.DataFrame,
    unconditional: pd.DataFrame,
    session: pd.DataFrame,
    time_of_day: pd.DataFrame,
    weekday: pd.DataFrame,
    year_partition: pd.DataFrame,
    costs: pd.DataFrame,
) -> list[dict]:
    checks = []

    def add(name: str, passed: bool, details: str, critical: bool = True) -> None:
        checks.append(
            {
                "category": "aggregation_reconciliation",
                "check_name": name,
                "critical": critical,
                "passed": bool(passed),
                "details": details,
            }
        )

    key = "forward_return_ticks"
    uncond = unconditional.loc[unconditional["outcome_name"].eq(key)].set_index("horizon_minutes")
    sess = session.loc[session["outcome_name"].eq(key)]
    session_counts = sess.groupby("horizon_minutes", observed=True)["observation_count"].sum()
    add(
        "unconditional_count_equals_session_sum",
        bool(session_counts.eq(len(frame)).all()),
        f"per_horizon_session_sum={session_counts.to_dict()}",
    )
    weighted = (
        sess.assign(weighted=sess["mean"] * sess["available_observation_count"])
        .groupby("horizon_minutes", observed=True)
        .agg(weighted_sum=("weighted", "sum"), available=("available_observation_count", "sum"))
    )
    weighted_mean = weighted["weighted_sum"] / weighted["available"]
    add(
        "weighted_session_means_reconcile",
        bool(np.allclose(weighted_mean.sort_index(), uncond["mean"].sort_index(), atol=1e-12)),
        "weighted by available observations",
    )
    tod = time_of_day.loc[time_of_day["outcome_name"].eq(key)]
    tod_counts = tod.groupby("horizon_minutes", observed=True)["observation_count"].sum()
    add(
        "time_bin_counts_reconcile",
        bool(tod_counts.eq(len(frame)).all()),
        f"per_horizon_time_bin_sum={tod_counts.to_dict()}",
    )
    dow = weekday.loc[weekday["analysis_type"].eq("weekday") & weekday["outcome_name"].eq(key)]
    dow_counts = dow.groupby("horizon_minutes", observed=True)["observation_count"].sum()
    add(
        "weekday_counts_reconcile",
        bool(dow_counts.eq(len(frame)).all()),
        f"per_horizon_weekday_sum={dow_counts.to_dict()}",
    )
    partitions = year_partition.loc[
        (year_partition["analysis_type"].eq("partition_stability"))
        & year_partition["outcome_name"].eq(key)
    ]
    partition_counts = partitions.groupby("horizon_minutes", observed=True)[
        "observation_count"
    ].sum()
    add(
        "partition_counts_reconcile",
        bool(partition_counts.eq(len(frame)).all()),
        f"per_horizon_partition_sum={partition_counts.to_dict()}",
    )
    years = year_partition.loc[
        (year_partition["analysis_type"].eq("year_stability"))
        & year_partition["outcome_name"].eq(key)
    ]
    year_counts = years.groupby("horizon_minutes", observed=True)["observation_count"].sum()
    add(
        "year_counts_reconcile",
        bool(year_counts.eq(len(frame)).all()),
        f"per_horizon_year_sum={year_counts.to_dict()}",
    )
    rate_columns = [c for c in costs if c.endswith("_rate") or c.startswith("probability_")]
    rates = costs[rate_columns].to_numpy(dtype=np.float64)
    add(
        "cost_threshold_rates_in_unit_interval",
        bool(((rates >= 0) & (rates <= 1)).all()),
        f"rate_columns={len(rate_columns)}",
    )
    quantile_columns = [
        "minimum",
        "p01",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "maximum",
    ]
    quantile_ok = all(
        (table[quantile_columns].diff(axis=1).iloc[:, 1:] >= -1e-12).all().all()
        for table in (unconditional, session, time_of_day, weekday, year_partition)
    )
    add(
        "quantile_ordering_possible",
        quantile_ok,
        "min<=p01<=...<=p99<=max across all summary tables",
    )
    return checks


def _write_figures(result: BaselineBuildResult, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    horizons = np.asarray(FORWARD_HORIZONS_MINUTES)
    returns = result.unconditional.loc[
        result.unconditional["outcome_name"].eq("forward_return_ticks")
    ].sort_values("horizon_minutes")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for quantile, label in (
        ("p05", "p05"),
        ("p25", "p25"),
        ("p50", "median"),
        ("p75", "p75"),
        ("p95", "p95"),
    ):
        ax.plot(horizons, returns[quantile], marker="o", label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        title="GC unconditional forward-return quantiles",
        xlabel="Horizon (minutes)",
        ylabel="Ticks",
    )
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "return_quantile_curves.png", dpi=150)
    plt.close(fig)

    ci = result.headline_uncertainty
    ci = ci.loc[
        (ci["analysis_type"].eq("headline_date_block_bootstrap"))
        & ci["session"].eq("Combined")
        & ci["metric_name"].eq("mean_return")
    ].sort_values("horizon_minutes")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(
        ci["horizon_minutes"],
        ci["metric_value"],
        yerr=[
            ci["metric_value"] - ci["confidence_interval_lower"],
            ci["confidence_interval_upper"] - ci["metric_value"],
        ],
        marker="o",
        capsize=3,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        title="Mean forward return with trading-date block intervals",
        xlabel="Horizon (minutes)",
        ylabel="Daily-weighted mean ticks",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "mean_return_block_ci.png", dpi=150)
    plt.close(fig)

    session_range = result.session.loc[result.session["outcome_name"].eq("future_range_ticks")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for session in SESSION_ORDER:
        data = session_range.loc[
            session_range["entry_session"].astype(str).eq(session)
        ].sort_values("horizon_minutes")
        ax.plot(data["horizon_minutes"], data["mean"], marker="o", label=session)
    ax.set(title="Mean future range by session", xlabel="Horizon (minutes)", ylabel="Ticks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "session_future_range.png", dpi=150)
    plt.close(fig)

    availability = result.session.loc[result.session["outcome_name"].eq("forward_return_ticks")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for session in SESSION_ORDER:
        data = availability.loc[availability["entry_session"].astype(str).eq(session)].sort_values(
            "horizon_minutes"
        )
        ax.plot(data["horizon_minutes"], data["availability_rate"], marker="o", label=session)
    ax.set(
        title="Fixed-horizon availability by session",
        xlabel="Horizon (minutes)",
        ylabel="Availability rate",
        ylim=(0.95, 1.001),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "session_availability.png", dpi=150)
    plt.close(fig)

    tod = result.time_of_day.loc[result.time_of_day["outcome_name"].eq("future_range_ticks")]
    for session in SESSION_ORDER:
        pivot = tod.loc[tod["entry_session"].astype(str).eq(session)].pivot(
            index="time_of_day_bin", columns="horizon_minutes", values="mean"
        )
        fig, ax = plt.subplots(figsize=(9, max(4.5, len(pivot) * 0.22)))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        step = max(1, len(pivot) // 12)
        ax.set_yticks(range(0, len(pivot), step), labels=pivot.index[::step])
        ax.set(
            title=f"{session} mean future range by entry time",
            xlabel="Horizon (minutes)",
            ylabel="15-minute entry bin",
        )
        fig.colorbar(image, ax=ax, label="Ticks")
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"time_of_day_range_{session.lower().replace(' ', '_')}.png", dpi=150
        )
        plt.close(fig)


def _build_report(result: BaselineBuildResult) -> str:
    unconditional_return = result.unconditional.loc[
        result.unconditional["outcome_name"].eq("forward_return_ticks")
    ].sort_values("horizon_minutes")
    session_return = result.session.loc[result.session["outcome_name"].eq("forward_return_ticks")]
    session_range = result.session.loc[result.session["outcome_name"].eq("future_range_ticks")]
    symmetry = result.directional_symmetry.loc[
        result.directional_symmetry["grouping"].eq("full")
    ].sort_values("horizon_minutes")
    costs = result.cost_thresholds.loc[
        (result.cost_thresholds["session"].eq("Combined"))
        & result.cost_thresholds["cost_threshold_ticks"].eq(5)
    ].sort_values("horizon_minutes")

    def compact(values: Iterable[float], digits: int = 3) -> str:
        return ", ".join(f"{value:.{digits}f}" for value in values)

    london = session_return.loc[
        session_return["entry_session"].astype(str).eq("London")
    ].sort_values("horizon_minutes")
    new_york = session_return.loc[
        session_return["entry_session"].astype(str).eq("New York")
    ].sort_values("horizon_minutes")
    london_range = session_range.loc[
        session_range["entry_session"].astype(str).eq("London")
    ].sort_values("horizon_minutes")
    new_york_range = session_range.loc[
        session_range["entry_session"].astype(str).eq("New York")
    ].sort_values("horizon_minutes")
    unconditional_range = result.unconditional.loc[
        result.unconditional["outcome_name"].eq("future_range_ticks")
    ].sort_values("horizon_minutes")
    unconditional_volatility = result.unconditional.loc[
        result.unconditional["outcome_name"].eq("future_realized_volatility_bps")
    ].sort_values("horizon_minutes")
    range_exponent = float(
        np.polyfit(
            np.log(unconditional_range["horizon_minutes"]), np.log(unconditional_range["mean"]), 1
        )[0]
    )
    volatility_exponent = float(
        np.polyfit(
            np.log(unconditional_volatility["horizon_minutes"]),
            np.log(unconditional_volatility["mean"]),
            1,
        )[0]
    )
    weekday_dates = result.weekday.loc[
        result.weekday["analysis_type"].eq("weekday")
        & result.weekday["outcome_name"].eq("forward_return_ticks"),
        "trading_date_count",
    ]
    coverage_lines = "\n".join(
        f"- {int(r.year)}: {r.first_entry_date.date()} to {r.last_entry_date.date()}, {int(r.trading_date_count):,} dates ({r.coverage_status})."
        for r in result.year_coverage.itertuples()
    )
    failed = result.validation.loc[result.validation["critical"] & ~result.validation["passed"]]
    status = "READY" if failed.empty else "NOT READY"
    blockers = "None." if failed.empty else "; ".join(failed["check_name"].tolist())
    return f"""# Section 5 Baseline Summary and Benchmark Definition

## Scope and statistical interpretation

This report describes overlapping GC observation outcomes at 5, 15, 30, 60, 120, and 180 minutes. These are event-level labels, not independent trials or trades. MFE is not realized profit, MAE is not realized loss, and future range is not a tradable return. Raw calculations are never clipped; figure clipping, where used for display, does not alter the tables. Dependence-aware intervals resample New York trading-date aggregates with seed {BASELINE_RANDOM_SEED} and {BOOTSTRAP_REPLICATES:,} replicates.

## Unconditional benchmark

- Available counts by horizon: {", ".join(f"{int(r.horizon_minutes)}m={int(r.available_observation_count):,}" for r in unconditional_return.itertuples())}.
- Mean signed returns in ticks (5m through 180m): {compact(unconditional_return["mean"])}.
- Median signed returns in ticks: {compact(unconditional_return["median"])}.
- Positive-return rates: {compact(unconditional_return["positive_rate"], 4)}.
- Mean absolute returns in ticks: {compact(unconditional_return["mean_absolute_value"])}.
- The p05/p95 tick pairs are {", ".join(f"{r.p05:.1f}/{r.p95:.1f}" for r in unconditional_return.itertuples())}; tails and robust centers must accompany means in future comparisons.
- Mean future ranges in ticks are {compact(unconditional_range["mean"])}; the fitted log-log horizon exponent is {range_exponent:.3f} (a transparent growth diagnostic, not a holding-period choice).
- Mean future realized volatility in basis points is {compact(unconditional_volatility["mean"])}; its fitted log-log horizon exponent is {volatility_exponent:.3f}. Movement growth is therefore measured directly rather than assumed proportional to clock time.

## Sessions, time of day, and weekdays

- London mean returns in ticks: {compact(london["mean"])}; New York: {compact(new_york["mean"])}.
- London positive rates: {compact(london["positive_rate"], 4)}; New York: {compact(new_york["positive_rate"], 4)}.
- London mean future ranges in ticks: {compact(london_range["mean"])}; New York: {compact(new_york_range["mean"])}.
- Availability, 15-minute entry bins, and minutes-since-session-open bins are saved explicitly. Long-horizon near-cutoff subsets must not be compared with early-session subsets without matching their availability and date coverage.
- Weekday estimates are descriptive and are saved both overall and by year with per-session date coverage (overall cells span {int(weekday_dates.min())} to {int(weekday_dates.max())} dates). No weekday is declared an effect from a single year or a small subset.

## Directional and tail symmetry

- Short signed returns are exactly the algebraic negative of long returns and are not independent samples.
- Full-population p95/|p05| tail ratios by horizon: {compact(symmetry["p95_tail_ratio"])}.
- Full-population p99/|p01| tail ratios by horizon: {compact(symmetry["p99_tail_ratio"])}.
- Long/short excursion identities reconcile exactly. Any modest unconditional tail imbalance remains descriptive and does not establish a directional edge.

## Year and research-partition stability

{coverage_lines}

Development and Validation are the primary interpretation samples. The baseline code path and metric contract were frozen before the one-time descriptive Final-test exposure. Section 5 exposes Final-test means, medians, positive rates, magnitude, excursion, range, volatility, tails, availability, session, time, weekday, symmetry, and fixed hurdle results. Future work must not tune definitions, horizons, cost hurdles, or features to those exposed values.

## Movement hurdles, not PnL

- At a fixed five-tick hurdle, combined absolute exit-to-exit exceedance rates (5m through 180m) are {compact(costs["probability_absolute_return_above_threshold"], 4)}.
- Five-tick long directional exceedance rates are {compact(costs["long_cost_hurdle_exceedance_rate"], 4)}; short directional exceedance rates are {compact(costs["short_cost_hurdle_exceedance_rate"], 4)}.
- Exit-to-exit return, MFE opportunity, MAE risk, and total high-low range hurdles are saved separately. Frequent intrahorizon exceedance is not evidence of realizable profit.

## Benchmark contract for future features

A future feature must be compared with the matching baseline using the same horizon, session, direction, research partition, availability rules, outcome definition, cost hurdle, and trading-date coverage. A positive bucket mean is insufficient. Evidence must include adequate observations and dates, economically meaningful effect size, Development-to-Validation consistency, reasonable year stability, session stability or a justified session thesis, friction-aware magnitude, tail robustness, date-clustered uncertainty, interpretability, and no leakage or Final-test tuning. No arbitrary pass/fail thresholds are introduced here.

Critical validation blockers: {blockers}

SECTION 5 STATUS: {status}
"""


def build_baseline_outputs(
    frame: pd.DataFrame, *, section_4_ready: bool = True
) -> BaselineBuildResult:
    """Build all compact Section 5 tables in memory without saving artifacts."""

    work = frame.copy(deep=False)
    time_dimensions = _add_time_dimensions(work)
    work = work.assign(
        time_of_day_bin=time_dimensions["time_of_day_bin"].to_numpy(),
        minutes_since_session_open_bin=time_dimensions["minutes_since_session_open_bin"].to_numpy(),
        day_of_week=pd.to_datetime(work["entry_timestamp_ny"]).dt.day_name().to_numpy(),
        year=pd.to_datetime(work["entry_timestamp_ny"]).dt.year.to_numpy(),
    )
    for horizon in FORWARD_HORIZONS_MINUTES:
        for side in ("long", "short"):
            for extreme in ("mfe", "mae"):
                source = f"time_to_{extreme}_{side}_{horizon}_minutes"
                work[f"time_to_{extreme}_{side}_{horizon}_fraction"] = (
                    _to_float(work[source]) / horizon
                )

    input_checks = _validate_inputs(work, section_4_ready)
    unconditional = summarize_outcomes(work, specs=ALL_OUTCOME_SPECS, analysis_type="unconditional")
    session = summarize_outcomes(
        work, specs=KEY_GROUP_SPECS, group_columns=("entry_session",), analysis_type="session"
    )
    time_of_day = summarize_outcomes(
        work,
        specs=KEY_GROUP_SPECS,
        group_columns=("entry_session", "time_of_day_bin", "minutes_since_session_open_bin"),
        analysis_type="time_of_day",
    )
    weekday = pd.concat(
        [
            summarize_outcomes(
                work,
                specs=KEY_GROUP_SPECS,
                group_columns=("day_of_week", "entry_session"),
                analysis_type="weekday",
            ),
            summarize_outcomes(
                work,
                specs=KEY_GROUP_SPECS,
                group_columns=("year", "day_of_week", "entry_session"),
                analysis_type="weekday_by_year",
            ),
        ],
        ignore_index=True,
    )
    yearly = summarize_outcomes(
        work,
        specs=KEY_GROUP_SPECS,
        group_columns=("year", "entry_session"),
        analysis_type="year_stability",
    )
    partitions = summarize_outcomes(
        work,
        specs=KEY_GROUP_SPECS,
        group_columns=("research_partition", "entry_session"),
        analysis_type="partition_stability",
    )
    year_partition = pd.concat([yearly, partitions], ignore_index=True)
    directional_symmetry = build_directional_symmetry(work)
    costs = build_cost_thresholds(work)
    uncertainty = build_headline_uncertainty(work)
    year_coverage = build_year_coverage(work)
    aggregation_checks = _validate_aggregations(
        work, unconditional, session, time_of_day, weekday, year_partition, costs
    )
    validation = pd.DataFrame.from_records(input_checks + aggregation_checks)

    summary = pd.concat(
        [
            _wide_to_normalized(unconditional),
            _wide_to_normalized(session.rename(columns={"entry_session": "session"})),
            _wide_to_normalized(time_of_day.rename(columns={"entry_session": "session"})),
            _wide_to_normalized(weekday.rename(columns={"entry_session": "session"})),
            _wide_to_normalized(year_partition.rename(columns={"entry_session": "session"})),
            _uncertainty_to_normalized(uncertainty),
        ],
        ignore_index=True,
    )
    summary["horizon_minutes"] = pd.to_numeric(summary["horizon_minutes"], errors="coerce").astype(
        "Int16"
    )
    summary["year"] = pd.to_numeric(summary["year"], errors="coerce").astype("Int16")
    summary["cost_threshold_ticks"] = pd.to_numeric(
        summary["cost_threshold_ticks"], errors="coerce"
    ).astype("Int16")
    summary["metric_value"] = pd.to_numeric(summary["metric_value"], errors="coerce").astype(float)

    benchmark = {
        "instrument": "GC",
        "event_level_not_trades": True,
        "horizons_minutes": list(FORWARD_HORIZONS_MINUTES),
        "sessions": list(SESSION_ORDER),
        "timezone": "America/New_York",
        "decision_information": "completed bar t",
        "theoretical_entry": "open of bar t+1",
        "forward_path_begins": "bar t+1",
        "availability_rule": "use an outcome only when label_available_<horizon> is true; never shorten a fixed horizon",
        "research_partitions": {
            "Development": "through 2023-12-31",
            "Validation": "2024",
            "Final test": "2025-01-01 through 2026-05-22",
        },
        "cost_threshold_ticks": list(COST_THRESHOLDS_TICKS),
        "bootstrap": {
            "unit": "New York trading date",
            "seed": BASELINE_RANDOM_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        },
        "matching_dimensions": [
            "horizon",
            "session",
            "direction",
            "research_partition",
            "availability",
            "outcome_definition",
            "cost_hurdle",
            "date_coverage",
        ],
        "final_test_exposure": "One-time descriptive Section 5 exposure; do not tune subsequent feature definitions or thresholds to these results.",
    }
    ready = bool(validation.loc[validation["critical"], "passed"].all())
    provisional = BaselineBuildResult(
        summary,
        costs,
        validation,
        unconditional,
        session,
        time_of_day,
        weekday,
        directional_symmetry,
        year_partition,
        uncertainty,
        year_coverage,
        benchmark,
        "",
        ready,
    )
    provisional.report_markdown = _build_report(provisional)
    return provisional


def save_baseline_outputs(result: BaselineBuildResult, *, project_root: Path) -> pd.DataFrame:
    """Save Section 5 tables/reports and perform independent Parquet reload checks."""

    project_root = Path(project_root)
    data_dir = project_root / "data/processed/statistical_research"
    table_dir = project_root / "reports/statistical_research/tables/section5"
    figure_dir = project_root / "reports/statistical_research/figures/section5"
    summary_dir = project_root / "reports/statistical_research/summaries"
    for path in (data_dir, table_dir, figure_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "baseline_summary_gc.parquet"
    cost_path = data_dir / "baseline_cost_thresholds_gc.parquet"
    validation_path = data_dir / "baseline_validation_gc.parquet"
    result.summary.to_parquet(summary_path, engine="pyarrow", compression="zstd", index=False)
    result.cost_thresholds.to_parquet(cost_path, engine="pyarrow", compression="zstd", index=False)
    result.validation.to_parquet(validation_path, engine="pyarrow", compression="zstd", index=False)

    reload_records = [
        {
            "artifact": "baseline_summary",
            "check_name": "stable_key_unique",
            "passed": bool(
                ~result.summary.duplicated(list(SUMMARY_DIMENSIONS[:13]), keep=False).any()
            ),
            "details": "normalized dimension and metric key",
        },
        {
            "artifact": "baseline_cost_thresholds",
            "check_name": "stable_key_unique",
            "passed": bool(
                ~result.cost_thresholds.duplicated(
                    ["analysis_type", "horizon_minutes", "session", "cost_threshold_ticks"],
                    keep=False,
                ).any()
            ),
            "details": "analysis/horizon/session/threshold key",
        },
    ]
    for name, path, expected in (
        ("baseline_summary", summary_path, result.summary),
        ("baseline_cost_thresholds", cost_path, result.cost_thresholds),
        ("baseline_validation", validation_path, result.validation),
    ):
        reloaded = pd.read_parquet(path, engine="pyarrow")
        expected_arrow = pa.Table.from_pandas(expected, preserve_index=False)
        reloaded_arrow = pq.read_table(path)
        checks = {
            "row_count": len(reloaded) == len(expected),
            "column_count": reloaded.shape[1] == expected.shape[1],
            "column_order": list(reloaded.columns) == list(expected.columns),
            "arrow_schema": reloaded_arrow.schema.remove_metadata()
            == expected_arrow.schema.remove_metadata(),
            "null_behavior": reloaded.isna().sum().equals(expected.isna().sum()),
        }
        for check, passed in checks.items():
            reload_records.append(
                {
                    "artifact": name,
                    "check_name": check,
                    "passed": bool(passed),
                    "details": str(path.relative_to(project_root)),
                }
            )
        if not all(checks.values()):
            raise AssertionError({name: checks})

    reload_validation_rows = pd.DataFrame(
        {
            "category": "save_reload",
            "check_name": [
                f"{row.artifact}_{row.check_name}"
                for row in pd.DataFrame(reload_records).itertuples()
            ],
            "critical": True,
            "passed": [bool(row.passed) for row in pd.DataFrame(reload_records).itertuples()],
            "details": [row.details for row in pd.DataFrame(reload_records).itertuples()],
        }
    )
    result.validation = pd.concat([result.validation, reload_validation_rows], ignore_index=True)
    result.validation.to_parquet(validation_path, engine="pyarrow", compression="zstd", index=False)
    final_validation_reload = pd.read_parquet(validation_path, engine="pyarrow")
    if not (
        len(final_validation_reload) == len(result.validation)
        and list(final_validation_reload.columns) == list(result.validation.columns)
        and final_validation_reload.isna().sum().equals(result.validation.isna().sum())
        and bool(final_validation_reload.loc[final_validation_reload["critical"], "passed"].all())
    ):
        raise AssertionError("Final baseline validation Parquet reload failed.")
    result.ready = bool(result.validation.loc[result.validation["critical"], "passed"].all())
    result.report_markdown = _build_report(result)

    result.unconditional.to_csv(table_dir / "section5_unconditional_outcomes.csv", index=False)
    result.session.to_csv(table_dir / "section5_session_outcomes.csv", index=False)
    result.time_of_day.to_csv(table_dir / "section5_time_of_day_outcomes.csv", index=False)
    result.weekday.to_csv(table_dir / "section5_weekday_outcomes.csv", index=False)
    result.directional_symmetry.to_csv(table_dir / "section5_directional_symmetry.csv", index=False)
    result.year_partition_stability.to_csv(
        table_dir / "section5_year_partition_stability.csv", index=False
    )
    result.cost_thresholds.to_csv(table_dir / "section5_cost_thresholds.csv", index=False)
    result.validation.to_csv(table_dir / "section5_validation.csv", index=False)
    result.year_coverage.to_csv(table_dir / "section5_year_coverage.csv", index=False)
    (summary_dir / "section5_baseline_summary.md").write_text(
        result.report_markdown, encoding="utf-8"
    )
    (summary_dir / "section5_benchmark_definition.json").write_text(
        json.dumps(result.benchmark_definition, indent=2), encoding="utf-8"
    )
    _write_figures(result, figure_dir)

    reload_checks = pd.DataFrame.from_records(reload_records)
    reload_checks.to_csv(table_dir / "section5_save_reload_validation.csv", index=False)
    return reload_checks
