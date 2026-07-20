"""Section 8 redundancy and incremental-information research for Branch B.

This module reduces the Section 7 expansion advancers to a small frozen set of
interpretable representatives:

1. Development-only Spearman correlation among the advancing features.
2. Hierarchical clustering on ``1 - |rho|`` with a frozen cut threshold.
3. Mechanical representative selection per cluster from Section 7 evidence.
4. Incremental-information testing of every non-anchor representative beyond
   the anchor representative, using per-NY-date partial rank ICs on
   Development with Validation confirmation.
5. A frozen candidate feature set for the multivariate phase.

Governance mirrors Section 7: Development and Validation only (a frame
containing Final-test rows is rejected), clustering and thresholds are fitted
on Development alone, and all criteria are frozen in :class:`Section8Config`
before any computation. The Section 7 directional result stands: no
directional feature advanced, so the frozen directional set is explicitly
empty rather than silently absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from .feature_evaluation import EVALUATION_PARTITIONS, EVALUATION_SESSIONS

SECTION8_RANDOM_SEED = 20260721


@dataclass(frozen=True)
class Section8Config:
    """Frozen Section 8 redundancy and incremental-information contract."""

    random_seed: int = SECTION8_RANDOM_SEED
    cluster_absolute_correlation_threshold: float = 0.7
    incremental_horizons: tuple[int, ...] = (60, 180)
    outcome_family: str = "expansion"
    outcome_template: str = "future_range_{h}_atr"
    primary_horizon_minutes: int = 60
    min_daily_observations: int = 10
    min_development_trading_dates: int = 400
    min_validation_trading_dates: int = 150
    min_development_abs_partial_ic: float = 0.05
    min_validation_partial_ic_retention: float = 0.25


@dataclass
class RedundancyBuildResult:
    """Container for every Section 8 output object."""

    correlation_matrix: pd.DataFrame
    cluster_members: pd.DataFrame
    representatives: pd.DataFrame
    incremental_results: pd.DataFrame
    frozen_feature_set: pd.DataFrame
    validation_checks: pd.DataFrame
    summary: pd.Series
    config: Section8Config = field(default_factory=Section8Config)


def _development_spearman(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Spearman correlation among features, Development partition only."""

    development = frame.loc[frame["research_partition"].eq("Development"), features]
    ranks = development.rank(method="average")
    return ranks.corr(method="pearson")


def _cluster_features(
    correlation: pd.DataFrame, threshold_abs_rho: float
) -> pd.Series:
    """Average-linkage clusters cut where within-cluster |rho| >= threshold."""

    features = list(correlation.columns)
    if len(features) == 1:
        return pd.Series([1], index=features, name="cluster_id")
    distance = 1.0 - correlation.abs().to_numpy()
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, 1.0)
    linkage = hierarchy.linkage(squareform(distance, checks=False), method="average")
    labels = hierarchy.fcluster(linkage, t=1.0 - threshold_abs_rho, criterion="distance")
    return pd.Series(labels, index=features, name="cluster_id")


def _select_representatives(
    cluster_ids: pd.Series,
    shortlist: pd.DataFrame,
    registry: pd.DataFrame,
    config: Section8Config,
) -> pd.DataFrame:
    """Mechanical per-cluster representative selection from Section 7 evidence.

    Score is the best passing-cell Validation |IC| at the primary horizon
    (falling back to the best passing cell at any horizon).  Ties break to
    core over experimental features, then alphabetically.
    """

    experimental = set(
        registry.loc[registry["is_experimental"].astype(bool), "feature_name"].astype(str)
    )
    relevant = shortlist.loc[shortlist["outcome_family"].eq(config.outcome_family)]
    primary = relevant.loc[relevant["horizon_minutes"].eq(config.primary_horizon_minutes)]

    def best_abs_val_ic(feature: str, table: pd.DataFrame) -> float:
        rows = table.loc[table["feature_name"].eq(feature), "daily_ic_mean_val"]
        return float(rows.abs().max()) if len(rows) else np.nan

    records = []
    for cluster_id, members in cluster_ids.groupby(cluster_ids):
        scored = []
        for feature in members.index:
            score = best_abs_val_ic(feature, primary)
            fallback = best_abs_val_ic(feature, relevant)
            scored.append(
                {
                    "cluster_id": int(cluster_id),
                    "feature_name": feature,
                    "is_experimental": feature in experimental,
                    "primary_horizon_abs_val_ic": score,
                    "best_abs_val_ic": fallback,
                    "selection_score": score if np.isfinite(score) else fallback,
                }
            )
        table = pd.DataFrame.from_records(scored).sort_values(
            ["selection_score", "is_experimental", "feature_name"],
            ascending=[False, True, True],
        )
        table["is_representative"] = False
        table.iloc[0, table.columns.get_loc("is_representative")] = True
        records.append(table)
    return pd.concat(records, ignore_index=True)


def _daily_pair_statistics(
    frame: pd.DataFrame,
    anchor: str,
    others: list[str],
    outcome_column: str,
    config: Section8Config,
) -> pd.DataFrame:
    """Per-date raw and anchor-partial rank ICs for each candidate feature."""

    records = []
    columns = [anchor, *others, outcome_column]
    for session in EVALUATION_SESSIONS:
        for partition in EVALUATION_PARTITIONS:
            mask = frame["entry_session"].eq(session) & frame["research_partition"].eq(partition)
            sub = frame.loc[mask, ["trade_date_ny", *columns]]
            date_codes, date_index = pd.factorize(sub["trade_date_ny"], sort=True)
            grouped = sub[columns].groupby(date_codes, sort=False)
            ranks = grouped.rank(method="average")
            mean = ranks.groupby(date_codes, sort=False).transform("mean")
            std = ranks.groupby(date_codes, sort=False).transform("std")
            with np.errstate(invalid="ignore", divide="ignore"):
                z = ((ranks - mean) / std).replace([np.inf, -np.inf], np.nan)
            z_anchor = z[anchor].to_numpy()
            z_outcome = z[outcome_column].to_numpy()
            n_dates = len(date_index)
            sizes = np.bincount(date_codes, minlength=n_dates)

            def daily_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                product = a * b
                finite = np.isfinite(product)
                sums = np.bincount(date_codes, weights=np.where(finite, product, 0.0), minlength=n_dates)
                counts = np.bincount(date_codes, weights=finite.astype(np.float64), minlength=n_dates)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out = sums / counts
                out[counts < config.min_daily_observations] = np.nan
                return out

            r_ao = daily_corr(z_anchor, z_outcome)
            for feature in others:
                z_feature = z[feature].to_numpy()
                r_bo = daily_corr(z_feature, z_outcome)
                r_ba = daily_corr(z_feature, z_anchor)
                with np.errstate(invalid="ignore", divide="ignore"):
                    partial = (r_bo - r_ba * r_ao) / np.sqrt(
                        (1.0 - r_ba**2) * (1.0 - r_ao**2)
                    )
                partial[(np.abs(r_ba) >= 1.0) | (np.abs(r_ao) >= 1.0)] = np.nan
                valid = np.isfinite(partial) & (sizes >= config.min_daily_observations)
                records.append(
                    {
                        "feature_name": feature,
                        "session": session,
                        "research_partition": partition,
                        "trading_date_count": int(valid.sum()),
                        "raw_daily_ic_mean": float(np.nanmean(np.where(valid, r_bo, np.nan))),
                        "partial_daily_ic_mean": float(np.nanmean(np.where(valid, partial, np.nan))),
                        "anchor_daily_ic_mean": float(np.nanmean(np.where(valid, r_ao, np.nan))),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_redundancy_analysis(
    frame: pd.DataFrame,
    shortlist: pd.DataFrame,
    feature_verdicts: pd.DataFrame,
    registry: pd.DataFrame,
    config: Section8Config | None = None,
) -> RedundancyBuildResult:
    """Run the complete declared Section 8 analysis."""

    cfg = config or Section8Config()
    observed_partitions = set(frame["research_partition"].astype(str).unique())
    if not observed_partitions.issubset(set(EVALUATION_PARTITIONS)):
        raise ValueError(f"redundancy frame contains locked partitions: {observed_partitions}")

    advancers = (
        feature_verdicts.loc[
            feature_verdicts["verdict"].eq("ADVANCE_EXPANSION"), "feature_name"
        ]
        .astype(str)
        .tolist()
    )
    directional_advancers = feature_verdicts.loc[
        feature_verdicts["verdict"].eq("ADVANCE_DIRECTIONAL"), "feature_name"
    ].astype(str).tolist()
    missing = [name for name in advancers if name not in frame.columns]
    if missing:
        raise ValueError(f"advancing features missing from frame: {missing[:5]}")
    if not advancers:
        raise ValueError("no expansion advancers were provided to Section 8")

    correlation = _development_spearman(frame, advancers)
    cluster_ids = _cluster_features(correlation, cfg.cluster_absolute_correlation_threshold)
    membership = _select_representatives(cluster_ids, shortlist, registry, cfg)

    representatives = membership.loc[membership["is_representative"]].reset_index(drop=True)
    anchor = representatives.sort_values(
        ["selection_score", "is_experimental", "feature_name"],
        ascending=[False, True, True],
    )["feature_name"].iloc[0]
    non_anchor = [name for name in representatives["feature_name"] if name != anchor]

    incremental_frames = []
    for horizon in cfg.incremental_horizons:
        outcome_column = cfg.outcome_template.format(h=horizon)
        stats = _daily_pair_statistics(frame, anchor, non_anchor, outcome_column, cfg)
        stats.insert(1, "horizon_minutes", horizon)
        incremental_frames.append(stats)
    incremental = pd.concat(incremental_frames, ignore_index=True)
    incremental["anchor_feature"] = anchor

    # ---- declared incremental gates (Development screen, Validation confirm) --
    key = ["feature_name", "horizon_minutes", "session"]
    development = incremental.loc[incremental["research_partition"].eq("Development")].set_index(key)
    validation = incremental.loc[incremental["research_partition"].eq("Validation")].set_index(key)
    joined = development.join(validation, lsuffix="_dev", rsuffix="_val", how="inner").reset_index()
    dev_partial = joined["partial_daily_ic_mean_dev"].to_numpy()
    val_partial = joined["partial_daily_ic_mean_val"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        retention = np.where(np.abs(dev_partial) > 0, val_partial / dev_partial, np.nan)
    joined["passes_incremental_gates"] = (
        (np.abs(dev_partial) >= cfg.min_development_abs_partial_ic)
        & (np.sign(dev_partial) == np.sign(val_partial))
        & (retention >= cfg.min_validation_partial_ic_retention)
        & (joined["trading_date_count_dev"] >= cfg.min_development_trading_dates)
        & (joined["trading_date_count_val"] >= cfg.min_validation_trading_dates)
    )
    incremental_by_feature = joined.groupby("feature_name")["passes_incremental_gates"].any()

    frozen_records = []
    for row in representatives.itertuples():
        if row.feature_name == anchor:
            reason = "anchor_representative"
            keep = True
        else:
            keep = bool(incremental_by_feature.get(row.feature_name, False))
            reason = (
                "confirmed_incremental_information"
                if keep
                else "no_confirmed_incremental_information_beyond_anchor"
            )
        frozen_records.append(
            {
                "feature_name": row.feature_name,
                "cluster_id": row.cluster_id,
                "is_experimental": row.is_experimental,
                "role": "anchor" if row.feature_name == anchor else "representative",
                "in_frozen_set": keep,
                "decision_reason": reason,
                "selection_score": row.selection_score,
            }
        )
    frozen_feature_set = pd.DataFrame.from_records(frozen_records).sort_values(
        ["in_frozen_set", "selection_score"], ascending=[False, False]
    ).reset_index(drop=True)

    cluster_sizes = cluster_ids.value_counts()
    summary = pd.Series(
        {
            "expansion_advancers": len(advancers),
            "directional_advancers": len(directional_advancers),
            "clusters": int(cluster_ids.nunique()),
            "largest_cluster_size": int(cluster_sizes.max()),
            "representatives": len(representatives),
            "anchor_feature": anchor,
            "incremental_candidates": len(non_anchor),
            "incremental_confirmed": int(
                frozen_feature_set.loc[
                    frozen_feature_set["role"].eq("representative"), "in_frozen_set"
                ].sum()
            ),
            "frozen_expansion_features": int(frozen_feature_set["in_frozen_set"].sum()),
            "frozen_directional_features": 0,
        },
        name="value",
    )

    checks = {
        "partitions_limited_to_development_validation": observed_partitions
        <= set(EVALUATION_PARTITIONS),
        "correlation_matrix_square_on_advancers": correlation.shape
        == (len(advancers), len(advancers)),
        "every_advancer_assigned_one_cluster": sorted(cluster_ids.index) == sorted(advancers),
        "one_representative_per_cluster": bool(
            membership.groupby("cluster_id")["is_representative"].sum().eq(1).all()
        ),
        "anchor_in_frozen_set": bool(
            frozen_feature_set.loc[
                frozen_feature_set["feature_name"].eq(anchor), "in_frozen_set"
            ].all()
        ),
        "frozen_set_subset_of_representatives": set(
            frozen_feature_set.loc[frozen_feature_set["in_frozen_set"], "feature_name"]
        )
        <= set(representatives["feature_name"]),
        "directional_frozen_set_empty": len(directional_advancers) == 0,
        "incremental_rows_cover_both_partitions": set(
            incremental["research_partition"].unique()
        )
        == set(EVALUATION_PARTITIONS),
    }
    validation_checks = pd.Series(checks, name="passed").rename_axis("check").reset_index()

    return RedundancyBuildResult(
        correlation_matrix=correlation,
        cluster_members=membership,
        representatives=representatives,
        incremental_results=joined,
        frozen_feature_set=frozen_feature_set,
        validation_checks=validation_checks,
        summary=summary,
        config=cfg,
    )


def save_redundancy_outputs(
    result: RedundancyBuildResult,
    *,
    project_root: Path,
) -> pd.DataFrame:
    """Persist Section 8 outputs; generated artifacts stay excluded from Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "section8"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    correlation = result.correlation_matrix.reset_index(names="feature_name")
    outputs = {
        "feature_correlation": (processed / "feature_correlation_gc.parquet", correlation),
        "feature_clusters": (
            processed / "feature_clusters_gc.parquet",
            result.cluster_members,
        ),
        "incremental_information": (
            processed / "feature_incremental_information_gc.parquet",
            result.incremental_results,
        ),
        "frozen_expansion_feature_set": (
            processed / "frozen_expansion_feature_set_gc.parquet",
            result.frozen_feature_set,
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
    result.frozen_feature_set.to_csv(tables / "section8_frozen_expansion_feature_set_gc.csv", index=False)
    result.cluster_members.to_csv(tables / "section8_feature_clusters_gc.csv", index=False)
    return pd.DataFrame.from_records(records)
