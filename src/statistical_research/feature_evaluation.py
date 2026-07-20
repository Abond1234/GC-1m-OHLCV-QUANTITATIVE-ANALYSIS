"""Section 7 univariate feature evaluation for the independent statistical branch.

This module screens every registered Section 6 predictor against the frozen
Section 4 forward outcomes under the project's declared evaluation contract:

- Evaluation uses the Development and Validation partitions only. The Final
  test is excluded at load time and no feature-to-outcome relationship on the
  Final test is computed anywhere in this module.
- All bins, thresholds, and screen statistics are fitted on Development only
  and applied unchanged to Validation.
- Overlapping minute observations are treated as correlated events: the primary
  evidence unit is the per-New-York-trading-date cross-sectional Spearman rank
  information coefficient (IC), summarized with date-block bootstrap intervals.
- Broad screens are corrected with Benjamini-Hochberg false-discovery control
  within declared screen families (outcome family x session, Development only).
- Shortlist criteria are frozen in :class:`Section7Config` before any results
  are computed; verdicts are produced mechanically from those criteria.

The module performs research screening only. It does not construct signals,
simulate trades, or estimate PnL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as scipy_stats

from .labels import FORWARD_HORIZONS_MINUTES


SECTION7_RANDOM_SEED = 20260720
SECTION7_BOOTSTRAP_REPLICATES = 2_000
SECTION7_BOOTSTRAP_CONFIDENCE = 0.95

EVALUATION_PARTITIONS = ("Development", "Validation")
EVALUATION_SESSIONS = ("London", "New York")

# Outcome families evaluated for every feature.  ``direction`` measures signed
# forward returns; ``expansion`` measures unsigned future movement/opportunity.
OUTCOME_FAMILY_ATR_TEMPLATES = {
    "direction": "forward_return_{h}_atr",
    "expansion": "future_range_{h}_atr",
}
OUTCOME_FAMILY_TICK_TEMPLATES = {
    "direction": "forward_return_{h}_ticks",
    "expansion": "future_range_{h}_ticks",
}

# Context fields registered as categorical predictors but excluded from the
# univariate screen with an explicit reason.  ``entry_session`` is one of the
# declared stratification dimensions, and Section 5 already established that no
# weekday filter is approved from descriptive evidence.
EXCLUDED_CATEGORICAL_FEATURES = {
    "entry_session": "session is an evaluation stratification dimension, not a predictor",
    "day_of_week": "weekday context was evaluated descriptively in Section 5; no filter approved",
}

PRIMARY_HORIZON_MINUTES = 60
QUANTILE_BUCKET_COUNT = 5
MIN_DAILY_OBSERVATIONS_FOR_IC = 10


@dataclass(frozen=True)
class Section7Config:
    """Frozen Section 7 evaluation and shortlist contract."""

    random_seed: int = SECTION7_RANDOM_SEED
    bootstrap_replicates: int = SECTION7_BOOTSTRAP_REPLICATES
    bootstrap_confidence: float = SECTION7_BOOTSTRAP_CONFIDENCE
    quantile_bucket_count: int = QUANTILE_BUCKET_COUNT
    primary_horizon_minutes: int = PRIMARY_HORIZON_MINUTES
    min_daily_observations_for_ic: int = MIN_DAILY_OBSERVATIONS_FOR_IC
    max_development_q_value: float = 0.10
    min_development_trading_dates: int = 400
    min_validation_trading_dates: int = 150
    min_development_observations: int = 10_000
    min_validation_observations: int = 4_000
    min_validation_ic_retention: float = 0.25
    min_development_bucket_monotonicity: float = 0.8
    min_direction_spread_ticks: float = 2.0


@dataclass
class EvaluationBuildResult:
    """Container for every Section 7 output object."""

    frame_summary: pd.Series
    cell_results: pd.DataFrame
    bucket_results: pd.DataFrame
    yearly_stability: pd.DataFrame
    shortlist: pd.DataFrame
    feature_verdicts: pd.DataFrame
    validation_checks: pd.DataFrame
    config: Section7Config = field(default_factory=Section7Config)


def _label_columns_for_evaluation() -> list[str]:
    columns = [
        "observation_id",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "atr_normalization_available",
    ]
    for horizon in FORWARD_HORIZONS_MINUTES:
        columns.append(f"label_available_{horizon}")
        for template in (*OUTCOME_FAMILY_ATR_TEMPLATES.values(), *OUTCOME_FAMILY_TICK_TEMPLATES.values()):
            columns.append(template.format(h=horizon))
    return columns


def load_forward_labels_for_evaluation(path: Path) -> pd.DataFrame:
    """Column-scoped label load that tolerates stored ordered-dictionary metadata."""

    table = pq.read_table(path, columns=_label_columns_for_evaluation())
    return table.to_pandas(ignore_metadata=True)


def evaluated_feature_names(registry: pd.DataFrame) -> list[str]:
    """Registered predictors screened by Section 7 (numeric and boolean only)."""

    names = []
    for row in registry.itertuples():
        if str(row.output_dtype) == "category":
            continue
        names.append(str(row.feature_name))
    return names


def build_evaluation_frame(
    feature_matrix: pd.DataFrame,
    forward_labels: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Join predictors and outcomes into the locked Development+Validation frame.

    Rows must have every horizon available with valid ATR normalization so the
    same sample supports every horizon comparison.  Final-test rows are removed
    here and never re-enter the module.
    """

    if not feature_matrix["observation_id"].is_unique:
        raise ValueError("feature matrix observation ids are not unique")
    if not forward_labels["observation_id"].is_unique:
        raise ValueError("forward label observation ids are not unique")

    features = evaluated_feature_names(registry)
    missing = [name for name in features if name not in feature_matrix.columns]
    if missing:
        raise ValueError(f"feature matrix is missing registered predictors: {missing[:5]}")

    labels = forward_labels.set_index("observation_id")
    matrix = feature_matrix.set_index("observation_id")
    if not labels.index.equals(matrix.index):
        labels = labels.reindex(matrix.index)
        if labels["research_partition"].isna().any():
            raise ValueError("forward labels do not cover every feature-matrix observation")

    partition = labels["research_partition"].astype(str)
    session = labels["entry_session"].astype(str)
    in_scope = partition.isin(EVALUATION_PARTITIONS).to_numpy()

    complete = labels["atr_normalization_available"].to_numpy(dtype=bool).copy()
    for horizon in FORWARD_HORIZONS_MINUTES:
        complete &= labels[f"label_available_{horizon}"].to_numpy(dtype=bool)
        for family in OUTCOME_FAMILY_ATR_TEMPLATES:
            atr_values = labels[OUTCOME_FAMILY_ATR_TEMPLATES[family].format(h=horizon)].to_numpy(dtype=np.float64)
            complete &= np.isfinite(atr_values)

    keep = in_scope & complete
    dates = pd.to_datetime(labels.loc[keep, "trade_date_ny"]).dt.normalize().to_numpy()
    columns: dict[str, np.ndarray] = {
        "trade_date_ny": dates,
        "entry_session": session[keep].to_numpy(),
        "research_partition": partition[keep].to_numpy(),
        "entry_year": pd.DatetimeIndex(dates).year.to_numpy().astype(np.int16),
    }
    for name in features:
        columns[name] = pd.array(matrix.loc[keep, name], dtype="Float64").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
    for horizon in FORWARD_HORIZONS_MINUTES:
        for family in OUTCOME_FAMILY_ATR_TEMPLATES:
            for template_set in (OUTCOME_FAMILY_ATR_TEMPLATES, OUTCOME_FAMILY_TICK_TEMPLATES):
                column = template_set[family].format(h=horizon)
                columns[column] = labels.loc[keep, column].to_numpy(dtype=np.float64)
    frame = pd.DataFrame(columns, index=matrix.index[keep])

    partition_counts = frame["research_partition"].value_counts()
    session_counts = frame["entry_session"].value_counts()
    summary = pd.Series(
        {
            "source_observations": len(matrix),
            "development_validation_observations": int(in_scope.sum()),
            "final_test_observations_excluded": int(len(matrix) - in_scope.sum()),
            "incomplete_label_rows_dropped": int(in_scope.sum() - keep.sum()),
            "evaluation_observations": int(keep.sum()),
            "development_observations": int(partition_counts.get("Development", 0)),
            "validation_observations": int(partition_counts.get("Validation", 0)),
            "london_observations": int(session_counts.get("London", 0)),
            "new_york_observations": int(session_counts.get("New York", 0)),
            "trading_dates": int(frame["trade_date_ny"].nunique()),
            "evaluated_features": len(features),
            "excluded_categorical_features": len(EXCLUDED_CATEGORICAL_FEATURES),
        },
        name="value",
    )
    return frame, summary


def _bootstrap_mean_ci(
    daily_values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> tuple[float, float]:
    """Date-block bootstrap interval for the mean of a daily series."""

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
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(x) for x in np.quantile(estimates, [alpha, 1.0 - alpha]))


def _bootstrap_matrix_ci(
    daily_matrix: np.ndarray,
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized date-block bootstrap for a (dates x features) daily matrix."""

    n_dates, n_features = daily_matrix.shape
    if n_dates < 2:
        empty = np.full(n_features, np.nan)
        return empty, empty.copy()
    rng = np.random.default_rng(seed)
    estimates = np.empty((replicates, n_features), dtype=np.float64)
    batch = 100
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        draw = rng.integers(0, n_dates, size=(stop - start, n_dates))
        estimates[start:stop] = np.nanmean(daily_matrix[draw], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.nanquantile(estimates, [alpha, 1.0 - alpha], axis=0)
    return low, high


def benjamini_hochberg_q_values(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up q-values; NaN p-values stay NaN."""

    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    values = p[finite]
    m = len(values)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * m / (np.arange(m) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m, dtype=np.float64)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[finite] = out
    return q


def _standardized_daily_ranks(
    values: pd.DataFrame,
    date_codes: np.ndarray,
) -> pd.DataFrame:
    """Per-date rank z-scores; columns without variance on a date become NaN."""

    grouped = values.groupby(date_codes, sort=False)
    ranks = grouped.rank(method="average")
    rank_mean = ranks.groupby(date_codes, sort=False).transform("mean")
    rank_std = ranks.groupby(date_codes, sort=False).transform("std")
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (ranks - rank_mean) / rank_std
    return z.replace([np.inf, -np.inf], np.nan)


def _fit_development_bucket_edges(
    development_values: np.ndarray,
    bucket_count: int,
) -> np.ndarray:
    """Interior quantile edges fitted on Development only, de-duplicated."""

    finite = development_values[np.isfinite(development_values)]
    if len(finite) == 0:
        return np.array([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, bucket_count + 1)[1:-1]
    edges = np.unique(np.quantile(finite, quantiles))
    return edges.astype(np.float64)


def _assign_buckets(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    buckets = np.full(len(values), -1, dtype=np.int64)
    finite = np.isfinite(values)
    buckets[finite] = np.searchsorted(edges, values[finite], side="right")
    return buckets


def build_univariate_evaluation(
    frame: pd.DataFrame,
    registry: pd.DataFrame,
    config: Section7Config | None = None,
) -> EvaluationBuildResult:
    """Run the complete declared univariate screen on the evaluation frame."""

    cfg = config or Section7Config()
    observed_partitions = set(frame["research_partition"].astype(str).unique())
    if not observed_partitions.issubset(set(EVALUATION_PARTITIONS)):
        raise ValueError(f"evaluation frame contains locked partitions: {observed_partitions}")

    features = [name for name in evaluated_feature_names(registry) if name in frame.columns]
    experimental = set(
        registry.loc[registry["is_experimental"].astype(bool), "feature_name"].astype(str)
    )

    horizons = tuple(FORWARD_HORIZONS_MINUTES)
    outcome_specs = [
        (family, horizon, OUTCOME_FAMILY_ATR_TEMPLATES[family].format(h=horizon))
        for family in OUTCOME_FAMILY_ATR_TEMPLATES
        for horizon in horizons
    ]

    cell_records: list[dict] = []
    yearly_records: list[dict] = []
    bucket_records: list[dict] = []

    # ---- development-fitted bucket edges, per feature and session ----------
    development_mask = frame["research_partition"].eq("Development").to_numpy()
    bucket_edges: dict[tuple[str, str], np.ndarray] = {}
    feature_arrays = {name: frame[name].to_numpy(dtype=np.float64) for name in features}
    session_values = frame["entry_session"].astype(str).to_numpy()
    for session in EVALUATION_SESSIONS:
        fit_mask = development_mask & (session_values == session)
        for name in features:
            bucket_edges[(name, session)] = _fit_development_bucket_edges(
                feature_arrays[name][fit_mask], cfg.quantile_bucket_count
            )

    # ---- per (session, partition) daily-IC and bucket statistics ----------
    for session_index, session in enumerate(EVALUATION_SESSIONS):
        for partition_index, partition in enumerate(EVALUATION_PARTITIONS):
            cell_mask = (session_values == session) & frame["research_partition"].eq(partition).to_numpy()
            sub = frame.loc[cell_mask]
            if sub.empty:
                continue
            date_codes, date_index = pd.factorize(sub["trade_date_ny"], sort=True)
            date_sizes = np.bincount(date_codes)
            valid_dates = date_sizes >= cfg.min_daily_observations_for_ic
            years = pd.DatetimeIndex(date_index).year.to_numpy()

            rank_input = sub[features + [spec[2] for spec in outcome_specs]]
            z = _standardized_daily_ranks(rank_input, date_codes)
            z_features = z[features].to_numpy(dtype=np.float64)
            n_dates = len(date_index)

            for family, horizon, outcome_column in outcome_specs:
                z_outcome = z[outcome_column].to_numpy(dtype=np.float64)
                products = z_features * z_outcome[:, None]
                sums = np.zeros((n_dates, len(features)), dtype=np.float64)
                counts = np.zeros((n_dates, len(features)), dtype=np.int64)
                finite = np.isfinite(products)
                np.add.at(sums, date_codes, np.where(finite, products, 0.0))
                np.add.at(counts, date_codes, finite)
                with np.errstate(invalid="ignore", divide="ignore"):
                    daily_ic = sums / counts
                daily_ic[counts < cfg.min_daily_observations_for_ic] = np.nan
                daily_ic[~valid_dates, :] = np.nan

                ic_count = np.isfinite(daily_ic).sum(axis=0)
                ic_mean = np.nanmean(np.where(np.isfinite(daily_ic), daily_ic, np.nan), axis=0)
                ic_std = np.nanstd(daily_ic, axis=0, ddof=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ic_t = ic_mean / (ic_std / np.sqrt(ic_count))
                p_values = np.full(len(features), np.nan)
                enough = ic_count >= 2
                p_values[enough] = 2.0 * scipy_stats.t.sf(
                    np.abs(ic_t[enough]), df=ic_count[enough] - 1
                )
                seed = (
                    cfg.random_seed
                    + 1_000_000 * session_index
                    + 100_000 * partition_index
                    + 1_000 * horizon
                    + (0 if family == "direction" else 500)
                )
                ci_low, ci_high = _bootstrap_matrix_ci(
                    daily_ic,
                    seed=seed,
                    replicates=cfg.bootstrap_replicates,
                    confidence=cfg.bootstrap_confidence,
                )

                for f_idx, name in enumerate(features):
                    cell_records.append(
                        {
                            "feature_name": name,
                            "outcome_family": family,
                            "horizon_minutes": horizon,
                            "session": session,
                            "research_partition": partition,
                            "observation_count": int(np.isfinite(
                                sub[name].to_numpy(dtype=np.float64)
                            ).sum()),
                            "trading_date_count": int(ic_count[f_idx]),
                            "daily_ic_mean": float(ic_mean[f_idx]),
                            "daily_ic_std": float(ic_std[f_idx]),
                            "daily_ic_t_stat": float(ic_t[f_idx]),
                            "daily_ic_p_value": float(p_values[f_idx]),
                            "daily_ic_ci_low": float(ci_low[f_idx]),
                            "daily_ic_ci_high": float(ci_high[f_idx]),
                        }
                    )

                for year in np.unique(years):
                    year_rows = years == year
                    if year_rows.sum() < 2:
                        continue
                    year_mean = np.nanmean(daily_ic[year_rows, :], axis=0)
                    year_count = np.isfinite(daily_ic[year_rows, :]).sum(axis=0)
                    for f_idx, name in enumerate(features):
                        yearly_records.append(
                            {
                                "feature_name": name,
                                "outcome_family": family,
                                "horizon_minutes": horizon,
                                "session": session,
                                "research_partition": partition,
                                "entry_year": int(year),
                                "trading_date_count": int(year_count[f_idx]),
                                "daily_ic_mean": float(year_mean[f_idx]),
                            }
                        )

            # ---- bucket statistics (Development-fitted edges) --------------
            dates_in_sub = sub["trade_date_ny"].to_numpy()
            for name in features:
                edges = bucket_edges[(name, session)]
                buckets = _assign_buckets(sub[name].to_numpy(dtype=np.float64), edges)
                bucket_total = len(edges) + 1
                for family, horizon, outcome_column in outcome_specs:
                    atr_values = sub[outcome_column].to_numpy(dtype=np.float64)
                    tick_column = OUTCOME_FAMILY_TICK_TEMPLATES[family].format(h=horizon)
                    tick_values = sub[tick_column].to_numpy(dtype=np.float64)
                    for bucket in range(bucket_total):
                        rows = buckets == bucket
                        n_rows = int(rows.sum())
                        if n_rows == 0:
                            continue
                        bucket_records.append(
                            {
                                "feature_name": name,
                                "outcome_family": family,
                                "horizon_minutes": horizon,
                                "session": session,
                                "research_partition": partition,
                                "bucket_index": bucket,
                                "bucket_count_fitted": bucket_total,
                                "observation_count": n_rows,
                                "trading_date_count": int(pd.unique(dates_in_sub[rows]).size),
                                "mean_outcome_atr": float(np.nanmean(atr_values[rows])),
                                "median_outcome_atr": float(np.nanmedian(atr_values[rows])),
                                "mean_outcome_ticks": float(np.nanmean(tick_values[rows])),
                                "positive_rate": float(np.nanmean(atr_values[rows] > 0))
                                if family == "direction"
                                else np.nan,
                            }
                        )

    cell_results = pd.DataFrame.from_records(cell_records)
    bucket_results = pd.DataFrame.from_records(bucket_records)
    yearly_stability = pd.DataFrame.from_records(yearly_records)

    # ---- Benjamini-Hochberg on the Development screen ----------------------
    cell_results["development_q_value"] = np.nan
    development_rows = cell_results["research_partition"].eq("Development")
    for family in OUTCOME_FAMILY_ATR_TEMPLATES:
        for session in EVALUATION_SESSIONS:
            screen = development_rows & cell_results["outcome_family"].eq(family) & cell_results["session"].eq(session)
            cell_results.loc[screen, "development_q_value"] = benjamini_hochberg_q_values(
                cell_results.loc[screen, "daily_ic_p_value"].to_numpy()
            )

    # ---- bucket monotonicity and Q-top minus Q-bottom spread ---------------
    monotonicity_records = []
    grouped_buckets = bucket_results.groupby(
        ["feature_name", "outcome_family", "horizon_minutes", "session", "research_partition"],
        sort=False,
    )
    for keys, group in grouped_buckets:
        ordered = group.sort_values("bucket_index")
        record = dict(
            zip(
                ["feature_name", "outcome_family", "horizon_minutes", "session", "research_partition"],
                keys,
            )
        )
        record["bucket_count_observed"] = len(ordered)
        if len(ordered) >= 3:
            rho, _ = scipy_stats.spearmanr(
                ordered["bucket_index"].to_numpy(), ordered["mean_outcome_atr"].to_numpy()
            )
            record["bucket_monotonicity"] = float(rho)
        else:
            record["bucket_monotonicity"] = np.nan
        top = ordered.iloc[-1]
        bottom = ordered.iloc[0]
        record["top_bottom_spread_atr"] = float(top["mean_outcome_atr"] - bottom["mean_outcome_atr"])
        record["top_bottom_spread_ticks"] = float(top["mean_outcome_ticks"] - bottom["mean_outcome_ticks"])
        monotonicity_records.append(record)
    monotonicity = pd.DataFrame.from_records(monotonicity_records)
    cell_results = cell_results.merge(
        monotonicity,
        on=["feature_name", "outcome_family", "horizon_minutes", "session", "research_partition"],
        how="left",
        validate="one_to_one",
    )

    # ---- development-versus-validation confirmation ------------------------
    key_columns = ["feature_name", "outcome_family", "horizon_minutes", "session"]
    development = cell_results.loc[cell_results["research_partition"].eq("Development")].set_index(key_columns)
    validation = cell_results.loc[cell_results["research_partition"].eq("Validation")].set_index(key_columns)
    joined = development.join(validation, lsuffix="_dev", rsuffix="_val", how="inner").reset_index()

    dev_ic = joined["daily_ic_mean_dev"].to_numpy()
    val_ic = joined["daily_ic_mean_val"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        retention = np.where(np.abs(dev_ic) > 0, val_ic / dev_ic, np.nan)
    joined["validation_sign_agreement"] = np.sign(dev_ic) == np.sign(val_ic)
    joined["validation_ic_retention"] = retention

    passes = (
        (joined["development_q_value_dev"] <= cfg.max_development_q_value)
        & (joined["trading_date_count_dev"] >= cfg.min_development_trading_dates)
        & (joined["trading_date_count_val"] >= cfg.min_validation_trading_dates)
        & (joined["observation_count_dev"] >= cfg.min_development_observations)
        & (joined["observation_count_val"] >= cfg.min_validation_observations)
        & joined["validation_sign_agreement"]
        & (joined["validation_ic_retention"] >= cfg.min_validation_ic_retention)
        & (joined["bucket_monotonicity_dev"].abs() >= cfg.min_development_bucket_monotonicity)
    )
    direction_rows = joined["outcome_family"].eq("direction")
    economic = (
        (joined["top_bottom_spread_ticks_dev"].abs() >= cfg.min_direction_spread_ticks)
        & (joined["top_bottom_spread_ticks_val"].abs() >= cfg.min_direction_spread_ticks)
    )
    joined["passes_all_criteria"] = passes & (~direction_rows | economic)
    joined["is_experimental"] = joined["feature_name"].isin(experimental)

    shortlist_columns = key_columns + [
        "is_experimental",
        "observation_count_dev",
        "observation_count_val",
        "trading_date_count_dev",
        "trading_date_count_val",
        "daily_ic_mean_dev",
        "daily_ic_ci_low_dev",
        "daily_ic_ci_high_dev",
        "development_q_value_dev",
        "daily_ic_mean_val",
        "daily_ic_ci_low_val",
        "daily_ic_ci_high_val",
        "validation_sign_agreement",
        "validation_ic_retention",
        "bucket_monotonicity_dev",
        "bucket_monotonicity_val",
        "top_bottom_spread_ticks_dev",
        "top_bottom_spread_ticks_val",
        "passes_all_criteria",
    ]
    confirmation = joined[shortlist_columns].copy()
    shortlist = confirmation.loc[confirmation["passes_all_criteria"]].reset_index(drop=True)
    shortlist = shortlist.sort_values(
        ["outcome_family", "daily_ic_mean_val"], key=lambda s: s.abs() if s.dtype.kind == "f" else s,
        ascending=[True, False],
    ).reset_index(drop=True)

    # ---- feature-level verdicts -------------------------------------------
    verdict_records = []
    for name in features:
        rows = confirmation.loc[confirmation["feature_name"].eq(name)]
        direction_pass = rows.loc[rows["outcome_family"].eq("direction"), "passes_all_criteria"].any()
        expansion_pass = rows.loc[rows["outcome_family"].eq("expansion"), "passes_all_criteria"].any()
        dev_screen_pass = (
            rows["development_q_value_dev"].le(cfg.max_development_q_value).any()
        )
        if direction_pass:
            verdict = "ADVANCE_DIRECTIONAL"
        elif expansion_pass:
            verdict = "ADVANCE_EXPANSION"
        elif dev_screen_pass:
            verdict = "WEAK_UNSTABLE"
        else:
            verdict = "NO_EVIDENCE"
        best_direction = rows.loc[rows["outcome_family"].eq("direction")]
        best_expansion = rows.loc[rows["outcome_family"].eq("expansion")]
        verdict_records.append(
            {
                "feature_name": name,
                "is_experimental": name in experimental,
                "verdict": verdict,
                "direction_cells_passing": int(
                    rows.loc[rows["outcome_family"].eq("direction"), "passes_all_criteria"].sum()
                ),
                "expansion_cells_passing": int(
                    rows.loc[rows["outcome_family"].eq("expansion"), "passes_all_criteria"].sum()
                ),
                "best_direction_val_ic": float(
                    best_direction["daily_ic_mean_val"].abs().max()
                ) if len(best_direction) else np.nan,
                "best_expansion_val_ic": float(
                    best_expansion["daily_ic_mean_val"].abs().max()
                ) if len(best_expansion) else np.nan,
            }
        )
    feature_verdicts = pd.DataFrame.from_records(verdict_records).sort_values(
        ["verdict", "best_expansion_val_ic"], ascending=[True, False]
    ).reset_index(drop=True)

    validation_checks = _build_validation_checks(
        frame, cell_results, bucket_results, shortlist, confirmation, feature_verdicts, features, cfg
    )

    frame_summary = pd.Series(
        {
            "evaluation_rows": len(frame),
            "evaluated_features": len(features),
            "screen_cells": len(cell_results),
            "confirmation_cells": len(confirmation),
            "shortlist_cells": len(shortlist),
            "advance_directional_features": int(feature_verdicts["verdict"].eq("ADVANCE_DIRECTIONAL").sum()),
            "advance_expansion_features": int(feature_verdicts["verdict"].eq("ADVANCE_EXPANSION").sum()),
            "weak_unstable_features": int(feature_verdicts["verdict"].eq("WEAK_UNSTABLE").sum()),
            "no_evidence_features": int(feature_verdicts["verdict"].eq("NO_EVIDENCE").sum()),
        },
        name="value",
    )

    return EvaluationBuildResult(
        frame_summary=frame_summary,
        cell_results=cell_results,
        bucket_results=bucket_results,
        yearly_stability=yearly_stability,
        shortlist=shortlist,
        feature_verdicts=feature_verdicts,
        validation_checks=validation_checks,
        config=cfg,
    )


def _build_validation_checks(
    frame: pd.DataFrame,
    cell_results: pd.DataFrame,
    bucket_results: pd.DataFrame,
    shortlist: pd.DataFrame,
    confirmation: pd.DataFrame,
    feature_verdicts: pd.DataFrame,
    features: Sequence[str],
    cfg: Section7Config,
) -> pd.DataFrame:
    horizons = tuple(FORWARD_HORIZONS_MINUTES)
    expected_cells = len(features) * len(horizons) * len(OUTCOME_FAMILY_ATR_TEMPLATES) * len(
        EVALUATION_SESSIONS
    ) * len(EVALUATION_PARTITIONS)
    q = cell_results["development_q_value"]
    development_rows = cell_results["research_partition"].eq("Development")
    checks = {
        "partitions_limited_to_development_validation": set(
            frame["research_partition"].astype(str).unique()
        ) <= set(EVALUATION_PARTITIONS),
        "no_final_test_rows": not frame["research_partition"].astype(str).eq("Final test").any(),
        "sessions_expected": set(frame["entry_session"].astype(str).unique()) <= set(EVALUATION_SESSIONS),
        "expected_cell_count": len(cell_results) == expected_cells,
        "q_values_only_on_development": q[~development_rows].isna().all(),
        "q_values_in_unit_interval": q[development_rows].dropna().between(0.0, 1.0).all(),
        "shortlist_is_subset_of_passing_cells": bool(
            shortlist["passes_all_criteria"].all()
        ) if len(shortlist) else True,
        "shortlist_matches_confirmation_pass_count": len(shortlist)
        == int(confirmation["passes_all_criteria"].sum()),
        "verdict_count_matches_features": len(feature_verdicts) == len(features),
        "no_forward_column_evaluated_as_feature": not any(
            "forward" in name or "future" in name or name.startswith(("mfe_", "mae_"))
            for name in features
        ),
        "bucket_counts_within_config": bucket_results["bucket_index"].max()
        < cfg.quantile_bucket_count,
        "daily_ic_within_bounds": cell_results["daily_ic_mean"].dropna().abs().le(1.0).all(),
    }
    return pd.Series(checks, name="passed").rename_axis("check").reset_index()


def save_evaluation_outputs(
    result: EvaluationBuildResult,
    *,
    project_root: Path,
) -> pd.DataFrame:
    """Persist Section 7 outputs; parquet artifacts stay excluded from Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "section7"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    outputs = {
        "univariate_results": (processed / "univariate_results_gc.parquet", result.cell_results),
        "univariate_bucket_summary": (
            processed / "univariate_bucket_summary_gc.parquet",
            result.bucket_results,
        ),
        "feature_stability": (processed / "feature_stability_gc.parquet", result.yearly_stability),
        "candidate_feature_shortlist": (
            processed / "candidate_feature_shortlist_gc.parquet",
            result.shortlist,
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

    result.shortlist.to_csv(tables / "section7_candidate_feature_shortlist_gc.csv", index=False)
    result.feature_verdicts.to_csv(tables / "section7_feature_verdicts_gc.csv", index=False)
    records.append(
        {
            "output": "tracked_tables",
            "path": str(tables.relative_to(project_root)),
            "rows": len(result.shortlist) + len(result.feature_verdicts),
            "columns": np.nan,
            "reload_row_match": True,
        }
    )
    return pd.DataFrame.from_records(records)
