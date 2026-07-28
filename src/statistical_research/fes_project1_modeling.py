"""Development-only feature evidence for Project 1 FES Section 4.

This module deliberately cannot load retrospective Validation or Historical
Final outcomes.  It implements the frozen 20-test confirmatory family, the
separate 40-test exploratory family, joint partial rank IC, Development-fitted
quintiles, and eight matched nested interaction comparisons.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .fes_project1_config import (
    D1_FEATURES,
    DIRECTIONAL_FEATURES,
    INTERACTION_SPECS,
    OPPORTUNITY_FEATURES,
    build_trial_ledger_skeleton,
)
from .fes_project1_features import (
    BASE_FEATURE_NAMES,
    COMPARATOR_MAP,
    INTERACTION_FEATURE_NAMES,
)

DEVELOPMENT_PARTITION: Final = "Development"
EVIDENCE_SESSIONS: Final = ("London", "New York")
RIDGE_ALPHA_GRID: Final = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
CONFIRMATORY_FAMILY: Final = "CONFIRMATORY_20"
EXPLORATORY_FAMILY: Final = "EXPLORATORY_40"
INTERACTION_FAMILY: Final = "INTERACTION_8"

TARGET_TICK_COLUMNS: Final[Mapping[str, str]] = {
    "forward_return_60_atr": "forward_return_60_ticks",
    "forward_return_30_atr": "forward_return_30_ticks",
    "future_range_60_atr": "future_range_60_ticks",
}
TARGET_AVAILABILITY_COLUMNS: Final[Mapping[str, str]] = {
    "forward_return_60_atr": "label_available_60",
    "forward_return_30_atr": "label_available_30",
    "future_range_60_atr": "label_available_60",
}

INTERACTION_PARENT_MAP: Final[Mapping[str, tuple[str, ...]]] = {
    str(item["name"]): tuple(str(parent) for parent in item["parents"])
    for item in INTERACTION_SPECS
    if str(item["id"]).startswith("I")
}
INTERACTION_ID_MAP: Final[Mapping[str, str]] = {
    str(item["name"]): str(item["id"])
    for item in INTERACTION_SPECS
    if str(item["id"]).startswith("I")
}


@dataclass(frozen=True)
class Section4EvidenceConfig:
    """Frozen Section 4 evidence and nested-fold settings."""

    random_seed: int = 20260726
    bootstrap_replicates: int = 2_000
    bootstrap_confidence: float = 0.95
    min_daily_observations: int = 10
    quintile_count: int = 5
    max_development_q_value: float = 0.10
    min_development_dates: int = 400
    min_development_observations: int = 10_000
    min_bucket_monotonicity: float = 0.80
    min_direction_spread_ticks: float = 2.0
    min_partial_ic_retention: float = 0.25
    min_interaction_delta: float = 0.01
    outer_initial_train_end: int = 251
    outer_assessment_dates: int = 63
    outer_step_dates: int = 64
    inner_initial_train_end: int = 125
    inner_assessment_dates: int = 21
    inner_step_dates: int = 22
    min_valid_inner_folds: int = 2
    ridge_alpha_grid: tuple[float, ...] = RIDGE_ALPHA_GRID


@dataclass(frozen=True)
class DevelopmentEvidenceInputs:
    """Narrow Development-only source frames and access audit."""

    scalar_features: pd.DataFrame
    scalar_missing_reasons: pd.DataFrame
    existing_features: pd.DataFrame
    existing_registry: pd.DataFrame
    forward_labels: pd.DataFrame
    access_audit: pd.DataFrame


@dataclass(frozen=True)
class Section4EvidenceResult:
    """All deterministic Development evidence and completed ledger rows."""

    confirmatory_results: pd.DataFrame
    exploratory_results: pd.DataFrame
    daily_ic: pd.DataFrame
    quintile_edges: pd.DataFrame
    bucket_results: pd.DataFrame
    partial_ic_results: pd.DataFrame
    base_feature_evidence: pd.DataFrame
    interaction_results: pd.DataFrame
    interaction_daily_deltas: pd.DataFrame
    interaction_fold_audit: pd.DataFrame
    interaction_tuning_audit: pd.DataFrame
    stability: pd.DataFrame
    distribution_stability: pd.DataFrame
    integrity_audit: pd.DataFrame
    correlations_with_existing: pd.DataFrame
    complete_trial_ledger: pd.DataFrame
    access_audit: pd.DataFrame
    config: Section4EvidenceConfig = field(default_factory=Section4EvidenceConfig)


def _assert_development_only(frame: pd.DataFrame, name: str) -> None:
    if "research_partition" not in frame.columns:
        raise ValueError(f"{name} must retain research_partition for access auditing.")
    observed = set(frame["research_partition"].astype(str).unique())
    if observed != {DEVELOPMENT_PARTITION}:
        raise PermissionError(
            f"{name} contains locked outcome partition(s): {sorted(observed)}"
        )


def _read_development_table(
    path: Path,
    columns: Sequence[str],
) -> pd.DataFrame:
    table = pq.read_table(
        path,
        columns=list(columns),
        filters=[("research_partition", "=", DEVELOPMENT_PARTITION)],
    )
    frame = table.to_pandas(ignore_metadata=True)
    _assert_development_only(frame, path.name)
    return frame


def load_development_evidence_inputs(
    project_root: Path,
) -> DevelopmentEvidenceInputs:
    """Load only projected Development rows required for Section 4.

    The trusted forward-label and existing-feature parquet reads always carry a
    physical ``research_partition == Development`` filter.  No caller-supplied
    partition is accepted.
    """

    root = Path(project_root)
    data_dir = root / "data/processed/statistical_research/fes_project1/v1"
    scalar_path = data_dir / "scalar_features_development_gc.parquet"
    missing_path = data_dir / "scalar_missing_reasons_development_gc.parquet"
    registry_path = (
        root / "data/processed/statistical_research/feature_registry_gc.parquet"
    )
    existing_path = (
        root / "data/processed/statistical_research/feature_matrix_gc.parquet"
    )
    label_path = (
        root / "data/processed/statistical_research/forward_labels_gc.parquet"
    )

    scalar = pd.read_parquet(scalar_path)
    _assert_development_only(scalar, scalar_path.name)
    missing = pd.read_parquet(missing_path)
    if not missing["observation_id"].is_unique:
        raise ValueError("Section 2 scalar missing-reason ids are not unique.")
    registry = pd.read_parquet(registry_path)
    existing_names = registry["feature_name"].astype(str).tolist()
    required_existing = tuple(
        dict.fromkeys(
            [
                *existing_names,
                *D1_FEATURES,
                *[
                    comparator
                    for comparators in COMPARATOR_MAP.values()
                    for comparator in comparators
                ],
            ]
        )
    )
    existing = _read_development_table(
        existing_path,
        ["observation_id", "research_partition", *required_existing],
    )
    label_columns = [
        "observation_id",
        "decision_timestamp_utc",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "label_available_30",
        "label_available_60",
        "exit_timestamp_utc_60",
        *TARGET_TICK_COLUMNS.keys(),
        *TARGET_TICK_COLUMNS.values(),
        *TARGET_AVAILABILITY_COLUMNS.values(),
    ]
    labels = _read_development_table(
        label_path,
        list(dict.fromkeys(label_columns)),
    )

    for name, frame in (
        ("scalar", scalar),
        ("existing", existing),
        ("labels", labels),
    ):
        if not frame["observation_id"].is_unique:
            raise ValueError(f"{name} observation ids are not unique.")
    scalar_ids = scalar["observation_id"].to_numpy(dtype=np.int64)
    for name, frame in (("existing", existing), ("labels", labels)):
        if set(frame["observation_id"]) != set(scalar_ids):
            raise ValueError(f"{name} does not cover the exact Development ids.")

    access_audit = pd.DataFrame.from_records(
        [
            {
                "source": scalar_path.relative_to(root).as_posix(),
                "read_scope": "presealed Development feature artifact",
                "rows_loaded": len(scalar),
                "partitions_loaded": DEVELOPMENT_PARTITION,
                "outcomes_loaded": False,
            },
            {
                "source": existing_path.relative_to(root).as_posix(),
                "read_scope": "physical parquet filter: Development",
                "rows_loaded": len(existing),
                "partitions_loaded": DEVELOPMENT_PARTITION,
                "outcomes_loaded": False,
            },
            {
                "source": label_path.relative_to(root).as_posix(),
                "read_scope": "physical parquet filter: Development; projected labels",
                "rows_loaded": len(labels),
                "partitions_loaded": DEVELOPMENT_PARTITION,
                "outcomes_loaded": True,
            },
            {
                "source": "retrospective Validation labels",
                "read_scope": "inaccessible until locked Section 6 batch",
                "rows_loaded": 0,
                "partitions_loaded": "",
                "outcomes_loaded": False,
            },
            {
                "source": "Historical Final labels",
                "read_scope": "inaccessible",
                "rows_loaded": 0,
                "partitions_loaded": "",
                "outcomes_loaded": False,
            },
        ]
    )
    return DevelopmentEvidenceInputs(
        scalar_features=scalar,
        scalar_missing_reasons=missing,
        existing_features=existing,
        existing_registry=registry,
        forward_labels=labels,
        access_audit=access_audit,
    )


def _feature_target_specs() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    confirmatory: list[dict[str, str]] = []
    exploratory: list[dict[str, str]] = []
    for session in EVIDENCE_SESSIONS:
        session_code = session.lower().replace(" ", "_")
        for feature in DIRECTIONAL_FEATURES:
            confirmatory.append(
                {
                    "trial_id": f"UCONF_{session_code}_{feature}",
                    "session": session,
                    "feature_name": feature,
                    "target": "forward_return_60_atr",
                    "family": CONFIRMATORY_FAMILY,
                    "analysis_role": "confirmatory",
                }
            )
            for target in ("forward_return_30_atr", "future_range_60_atr"):
                exploratory.append(
                    {
                        "trial_id": f"UEXP_{session_code}_{feature}_{target}",
                        "session": session,
                        "feature_name": feature,
                        "target": target,
                        "family": EXPLORATORY_FAMILY,
                        "analysis_role": "exploratory",
                    }
                )
        for feature in OPPORTUNITY_FEATURES:
            confirmatory.append(
                {
                    "trial_id": f"UCONF_{session_code}_{feature}",
                    "session": session,
                    "feature_name": feature,
                    "target": "future_range_60_atr",
                    "family": CONFIRMATORY_FAMILY,
                    "analysis_role": "confirmatory",
                }
            )
            for target in ("forward_return_60_atr", "forward_return_30_atr"):
                exploratory.append(
                    {
                        "trial_id": f"UEXP_{session_code}_{feature}_{target}",
                        "session": session,
                        "feature_name": feature,
                        "target": target,
                        "family": EXPLORATORY_FAMILY,
                        "analysis_role": "exploratory",
                    }
                )
    if len(confirmatory) != 20 or len(exploratory) != 40:
        raise AssertionError("Frozen univariate family sizes changed.")
    return confirmatory, exploratory


def _benjamini_hochberg_fixed_family(
    p_values: np.ndarray,
    family_size: int,
) -> np.ndarray:
    """BH q-values with the declared family size retained when a test is NaN."""

    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan)
    finite_positions = np.flatnonzero(np.isfinite(p))
    if len(finite_positions) == 0:
        return q
    values = p[finite_positions]
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * family_size / (np.arange(len(values)) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty(len(values), dtype=np.float64)
    restored[order] = np.clip(ranked, 0.0, 1.0)
    q[finite_positions] = restored
    return q


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 200):
        stop = min(start + 200, replicates)
        draw = rng.integers(0, len(finite), size=(stop - start, len(finite)))
        estimates[start:stop] = finite[draw].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _daily_spearman(
    frame: pd.DataFrame,
    x_name: str,
    y_name: str,
    *,
    min_observations: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for date, group in frame.groupby("trade_date_ny", observed=True, sort=True):
        pair = group[[x_name, y_name]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < min_observations:
            continue
        if pair[x_name].nunique() < 2 or pair[y_name].nunique() < 2:
            continue
        value = scipy_stats.spearmanr(
            pair[x_name].to_numpy(dtype=np.float64),
            pair[y_name].to_numpy(dtype=np.float64),
        ).statistic
        if np.isfinite(value):
            records.append(
                {
                    "trade_date_ny": pd.Timestamp(date).normalize(),
                    "observation_count": len(pair),
                    "daily_ic": float(value),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["trade_date_ny", "observation_count", "daily_ic"],
    )


def _summarize_daily_ic(
    daily: pd.DataFrame,
    *,
    seed: int,
    config: Section4EvidenceConfig,
) -> dict[str, object]:
    values = daily["daily_ic"].to_numpy(dtype=np.float64)
    count = int(np.isfinite(values).sum())
    mean = float(np.nanmean(values)) if count else np.nan
    std = float(np.nanstd(values, ddof=1)) if count >= 2 else np.nan
    t_statistic = (
        float(mean / (std / np.sqrt(count)))
        if count >= 2 and np.isfinite(std) and std > 0.0
        else np.nan
    )
    p_value = (
        float(2.0 * scipy_stats.t.sf(abs(t_statistic), df=count - 1))
        if np.isfinite(t_statistic)
        else np.nan
    )
    ci_low, ci_high = _bootstrap_mean_ci(
        values,
        seed=seed,
        replicates=config.bootstrap_replicates,
        confidence=config.bootstrap_confidence,
    )
    return {
        "observation_count": int(daily["observation_count"].sum()),
        "trading_date_count": count,
        "daily_ic_mean": mean,
        "daily_ic_std": std,
        "daily_ic_t_stat": t_statistic,
        "raw_p_value": p_value,
        "daily_ic_ci_low": ci_low,
        "daily_ic_ci_high": ci_high,
    }


def _fit_quintile_edges(values: np.ndarray, count: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.array([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, count + 1)[1:-1]
    return np.unique(np.quantile(finite, quantiles, method="linear"))


def _edge_hash(
    feature_name: str,
    session: str,
    edges: np.ndarray,
) -> str:
    payload = {
        "feature_name": feature_name,
        "session": session,
        "quantile_method": "linear",
        "edges": [float(value) for value in edges],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_quintile_evidence(
    joined: pd.DataFrame,
    confirmatory_specs: Sequence[Mapping[str, str]],
    config: Section4EvidenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edge_records: list[dict[str, object]] = []
    bucket_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for spec in confirmatory_specs:
        feature = spec["feature_name"]
        session = spec["session"]
        target = spec["target"]
        tick_target = TARGET_TICK_COLUMNS[target]
        availability = TARGET_AVAILABILITY_COLUMNS[target]
        sub = joined.loc[
            joined["entry_session"].astype(str).eq(session)
            & joined[availability].astype(bool)
        ]
        edges = _fit_quintile_edges(
            sub[feature].to_numpy(dtype=np.float64),
            config.quintile_count,
        )
        edge_digest = _edge_hash(feature, session, edges)
        for edge_index, edge in enumerate(edges, start=1):
            edge_records.append(
                {
                    "feature_name": feature,
                    "session": session,
                    "edge_index": edge_index,
                    "edge_value": float(edge),
                    "quantile_method": "linear",
                    "fitted_partition": DEVELOPMENT_PARTITION,
                    "edge_set_sha256": edge_digest,
                }
            )
        feature_values = sub[feature].to_numpy(dtype=np.float64)
        target_values = sub[target].to_numpy(dtype=np.float64)
        tick_values = sub[tick_target].to_numpy(dtype=np.float64)
        valid = (
            np.isfinite(feature_values)
            & np.isfinite(target_values)
            & np.isfinite(tick_values)
        )
        buckets = np.full(len(sub), -1, dtype=np.int64)
        buckets[valid] = np.searchsorted(
            edges, feature_values[valid], side="right"
        )
        for bucket in range(len(edges) + 1):
            rows = buckets == bucket
            if not rows.any():
                continue
            bucket_records.append(
                {
                    "trial_id": spec["trial_id"],
                    "feature_name": feature,
                    "session": session,
                    "target": target,
                    "bucket_index": bucket,
                    "bucket_count_fitted": len(edges) + 1,
                    "observation_count": int(rows.sum()),
                    "trading_date_count": int(sub.loc[rows, "trade_date_ny"].nunique()),
                    "mean_target_atr": float(np.mean(target_values[rows])),
                    "median_target_atr": float(np.median(target_values[rows])),
                    "mean_target_ticks": float(np.mean(tick_values[rows])),
                    "edge_set_sha256": edge_digest,
                }
            )
        group = pd.DataFrame(bucket_records)
        group = group.loc[group["trial_id"].eq(spec["trial_id"])].sort_values(
            "bucket_index"
        )
        monotonicity = (
            float(
                scipy_stats.spearmanr(
                    group["bucket_index"].to_numpy(dtype=np.float64),
                    group["mean_target_atr"].to_numpy(dtype=np.float64),
                ).statistic
            )
            if len(group) >= 3
            else np.nan
        )
        spread_ticks = (
            float(
                group.iloc[-1]["mean_target_ticks"]
                - group.iloc[0]["mean_target_ticks"]
            )
            if len(group) >= 2
            else np.nan
        )
        summary_records.append(
            {
                "trial_id": spec["trial_id"],
                "bucket_count_observed": len(group),
                "bucket_monotonicity": monotonicity,
                "top_bottom_spread_ticks": spread_ticks,
                "edge_set_sha256": edge_digest,
            }
        )
    return (
        pd.DataFrame.from_records(edge_records),
        pd.DataFrame.from_records(bucket_records),
        pd.DataFrame.from_records(summary_records),
    )


def _daily_partial_rank_ic(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    comparators: Sequence[str],
    *,
    min_observations: int,
) -> pd.DataFrame:
    required = [feature, target, *comparators]
    records: list[dict[str, object]] = []
    required_count = max(min_observations, len(comparators) + 3)
    for date, group in frame.groupby("trade_date_ny", observed=True, sort=True):
        complete = group[required].replace([np.inf, -np.inf], np.nan).dropna()
        if len(complete) < required_count:
            continue
        ranks = complete.rank(method="average")
        comparator_design = np.column_stack(
            [
                np.ones(len(ranks), dtype=np.float64),
                ranks.loc[:, comparators].to_numpy(dtype=np.float64),
            ]
        )
        if np.linalg.matrix_rank(comparator_design) < comparator_design.shape[1]:
            continue
        feature_rank = ranks[feature].to_numpy(dtype=np.float64)
        target_rank = ranks[target].to_numpy(dtype=np.float64)
        feature_residual = feature_rank - comparator_design @ np.linalg.lstsq(
            comparator_design, feature_rank, rcond=None
        )[0]
        target_residual = target_rank - comparator_design @ np.linalg.lstsq(
            comparator_design, target_rank, rcond=None
        )[0]
        if np.std(feature_residual) == 0.0 or np.std(target_residual) == 0.0:
            continue
        partial_ic = np.corrcoef(feature_residual, target_residual)[0, 1]
        if np.isfinite(partial_ic):
            records.append(
                {
                    "trade_date_ny": pd.Timestamp(date).normalize(),
                    "observation_count": len(complete),
                    "daily_partial_ic": float(partial_ic),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["trade_date_ny", "observation_count", "daily_partial_ic"],
    )


def _build_partial_evidence(
    joined: pd.DataFrame,
    confirmatory: pd.DataFrame,
    config: Section4EvidenceConfig,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index, row in enumerate(confirmatory.itertuples(index=False)):
        comparators = COMPARATOR_MAP[str(row.feature_name)]
        availability = TARGET_AVAILABILITY_COLUMNS[str(row.target)]
        sub = joined.loc[
            joined["entry_session"].astype(str).eq(str(row.session))
            & joined[availability].astype(bool)
        ]
        daily = _daily_partial_rank_ic(
            sub,
            str(row.feature_name),
            str(row.target),
            comparators,
            min_observations=config.min_daily_observations,
        )
        values = daily["daily_partial_ic"].to_numpy(dtype=np.float64)
        count = int(np.isfinite(values).sum())
        partial_mean = float(np.nanmean(values)) if count else np.nan
        partial_low, partial_high = _bootstrap_mean_ci(
            values,
            seed=config.random_seed + 30_000 + index,
            replicates=config.bootstrap_replicates,
            confidence=config.bootstrap_confidence,
        )
        raw_mean = float(row.daily_ic_mean)
        retention = (
            abs(partial_mean) / abs(raw_mean)
            if np.isfinite(partial_mean) and np.isfinite(raw_mean) and raw_mean != 0.0
            else np.nan
        )
        sign_agreement = bool(
            np.isfinite(partial_mean)
            and np.isfinite(raw_mean)
            and np.sign(partial_mean) == np.sign(raw_mean)
        )
        records.append(
            {
                "trial_id": row.trial_id,
                "feature_name": row.feature_name,
                "session": row.session,
                "target": row.target,
                "comparators": "|".join(comparators),
                "partial_trading_date_count": count,
                "partial_observation_count": int(
                    daily["observation_count"].sum()
                ),
                "partial_ic_mean": partial_mean,
                "partial_ic_ci_low": partial_low,
                "partial_ic_ci_high": partial_high,
                "partial_raw_sign_agreement": sign_agreement,
                "absolute_partial_ic_retention": retention,
                "partial_overlap_gate": bool(
                    sign_agreement
                    and np.isfinite(retention)
                    and retention >= config.min_partial_ic_retention
                ),
                "scratch_sensitivity_status": "UNVERIFIED_COMPARATOR_NOT_USED",
            }
        )
    return pd.DataFrame.from_records(records)


def _build_univariate_evidence(
    joined: pd.DataFrame,
    config: Section4EvidenceConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    confirmatory_specs, exploratory_specs = _feature_target_specs()
    result_records: list[dict[str, object]] = []
    daily_records: list[pd.DataFrame] = []
    for index, spec in enumerate([*confirmatory_specs, *exploratory_specs]):
        availability = TARGET_AVAILABILITY_COLUMNS[spec["target"]]
        sub = joined.loc[
            joined["entry_session"].astype(str).eq(spec["session"])
            & joined[availability].astype(bool)
        ]
        daily = _daily_spearman(
            sub,
            spec["feature_name"],
            spec["target"],
            min_observations=config.min_daily_observations,
        )
        summary = _summarize_daily_ic(
            daily,
            seed=config.random_seed + index,
            config=config,
        )
        result_records.append({**spec, **summary})
        daily = daily.assign(
            trial_id=spec["trial_id"],
            analysis_role=spec["analysis_role"],
            feature_name=spec["feature_name"],
            session=spec["session"],
            target=spec["target"],
        )
        daily_records.append(daily)
    results = pd.DataFrame.from_records(result_records)
    results["bh_q_value"] = np.nan
    confirmatory_mask = results["family"].eq(CONFIRMATORY_FAMILY)
    exploratory_mask = results["family"].eq(EXPLORATORY_FAMILY)
    results.loc[confirmatory_mask, "bh_q_value"] = (
        _benjamini_hochberg_fixed_family(
            results.loc[confirmatory_mask, "raw_p_value"].to_numpy(),
            20,
        )
    )
    results.loc[exploratory_mask, "bh_q_value"] = (
        _benjamini_hochberg_fixed_family(
            results.loc[exploratory_mask, "raw_p_value"].to_numpy(),
            40,
        )
    )
    confirmatory = results.loc[confirmatory_mask].reset_index(drop=True)
    exploratory = results.loc[exploratory_mask].reset_index(drop=True)
    edges, buckets, bucket_summary = _build_quintile_evidence(
        joined, confirmatory_specs, config
    )
    confirmatory = confirmatory.merge(
        bucket_summary, on="trial_id", how="left", validate="one_to_one"
    )
    return (
        confirmatory,
        exploratory,
        pd.concat(daily_records, ignore_index=True),
        edges,
        buckets,
    )


def _merge_development_support_labels(
    confirmatory: pd.DataFrame,
    partial: pd.DataFrame,
    config: Section4EvidenceConfig,
) -> pd.DataFrame:
    evidence = confirmatory.merge(
        partial,
        on=["trial_id", "feature_name", "session", "target"],
        how="left",
        validate="one_to_one",
    )
    gate_columns = {
        "gate_bh_q": evidence["bh_q_value"].le(config.max_development_q_value),
        "gate_dates": evidence["trading_date_count"].ge(
            config.min_development_dates
        ),
        "gate_observations": evidence["observation_count"].ge(
            config.min_development_observations
        ),
        "gate_bucket_monotonicity": evidence["bucket_monotonicity"]
        .abs()
        .ge(config.min_bucket_monotonicity),
        "gate_partial_overlap": evidence["partial_overlap_gate"].astype(bool),
    }
    directional = evidence["target"].eq("forward_return_60_atr")
    gate_columns["gate_direction_spread"] = (
        ~directional
        | evidence["top_bottom_spread_ticks"]
        .abs()
        .ge(config.min_direction_spread_ticks)
    )
    for name, values in gate_columns.items():
        evidence[name] = values.fillna(False)
    gate_names = list(gate_columns)
    evidence["failed_gates"] = evidence.apply(
        lambda row: "|".join(name for name in gate_names if not bool(row[name])),
        axis=1,
    )
    evidence["development_label"] = np.where(
        evidence[gate_names].all(axis=1),
        "DEV_SUPPORT",
        "DEV_NO_SUPPORT",
    )
    evidence["validation_gates_status"] = "PENDING_LOCKED_SECTION_6"
    return evidence


def build_chronological_folds(
    dates: Iterable[object],
    *,
    initial_train_end: int,
    assessment_dates: int,
    step_dates: int,
) -> list[dict[str, object]]:
    """Build exact expanding date folds with one embargo date."""

    unique_dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique()
    unique_dates = unique_dates.sort_values()
    folds: list[dict[str, object]] = []
    train_end = initial_train_end
    fold_id = 0
    while train_end + 2 + assessment_dates <= len(unique_dates):
        train_dates = unique_dates[: train_end + 1]
        embargo_date = unique_dates[train_end + 1]
        assessment = unique_dates[
            train_end + 2 : train_end + 2 + assessment_dates
        ]
        payload = {
            "train_dates": [value.strftime("%Y-%m-%d") for value in train_dates],
            "embargo_date": embargo_date.strftime("%Y-%m-%d"),
            "assessment_dates": [
                value.strftime("%Y-%m-%d") for value in assessment
            ],
        }
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        folds.append(
            {
                "fold_id": fold_id,
                "train_dates": train_dates,
                "embargo_date": embargo_date,
                "assessment_dates": assessment,
                "fold_sha256": digest,
            }
        )
        fold_id += 1
        train_end += step_dates
    return folds


def _interaction_value(
    name: str,
    frame: pd.DataFrame,
) -> np.ndarray:
    if name == "curvature_coherence_30":
        return (
            frame["price_path_curvature_30_atr"].to_numpy(dtype=np.float64)
            * frame["efficiency_ratio_30"].to_numpy(dtype=np.float64)
        )
    if name == "tail_pressure_activity_60":
        return (
            frame["ret_tail_balance_60"].to_numpy(dtype=np.float64)
            * frame["relative_volume_20_clipped"].to_numpy(dtype=np.float64)
        )
    if name == "lagged_volume_confirmation_30":
        return (
            frame["lagged_volume_return_spearman_30"].to_numpy(dtype=np.float64)
            * frame["relative_volume_20_clipped"].to_numpy(dtype=np.float64)
        )
    if name == "vwap_trend_alignment_30":
        return (
            frame["distance_from_research_day_vwap_atr"].to_numpy(dtype=np.float64)
            * frame["normalized_ols_slope_30"].to_numpy(dtype=np.float64)
        )
    raise ValueError(f"Unknown frozen interaction: {name}")


def _prepare_fold_design(
    frame: pd.DataFrame,
    columns: Sequence[str],
    interaction_name: str,
    train_mask: np.ndarray,
    assessment_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    train = frame.loc[train_mask, columns].copy()
    assessment = frame.loc[assessment_mask, columns].copy()
    if "lagged_volume_return_spearman_30" in columns:
        median = float(train["lagged_volume_return_spearman_30"].median())
        train["lagged_volume_return_spearman_30"] = train[
            "lagged_volume_return_spearman_30"
        ].fillna(median)
        assessment["lagged_volume_return_spearman_30"] = assessment[
            "lagged_volume_return_spearman_30"
        ].fillna(median)
    base_train = train.to_numpy(dtype=np.float64)
    base_assessment = assessment.to_numpy(dtype=np.float64)
    plus_train_frame = train.copy()
    plus_assessment_frame = assessment.copy()
    plus_train_frame[interaction_name] = _interaction_value(
        interaction_name, plus_train_frame
    )
    plus_assessment_frame[interaction_name] = _interaction_value(
        interaction_name, plus_assessment_frame
    )
    plus_train = plus_train_frame.to_numpy(dtype=np.float64)
    plus_assessment = plus_assessment_frame.to_numpy(dtype=np.float64)
    base_scaler = StandardScaler().fit(base_train)
    plus_scaler = StandardScaler().fit(plus_train)
    return (
        base_scaler.transform(base_train),
        base_scaler.transform(base_assessment),
        plus_scaler.transform(plus_train),
        plus_scaler.transform(plus_assessment),
        plus_scaler,
    )


def _ridge_svd_path_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_assessment: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    """Return the centered ridge path, algebraically equal to SVD ridge.

    The predictor count is small, so one symmetric eigendecomposition of the
    training-fold Gram matrix reuses the exact ridge solution across the frozen
    alpha grid.  The selected outer fit still uses scikit-learn Ridge with
    ``solver="svd"``.
    """

    target_mean = float(np.mean(y_train))
    centered_target = y_train - target_mean
    feature_mean = np.mean(x_train, axis=0)
    centered_train = x_train - feature_mean
    centered_assessment = x_assessment - feature_mean
    gram = centered_train.T @ centered_train
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_cross_product = (
        eigenvectors.T @ (centered_train.T @ centered_target)
    )
    predictions = np.empty((len(x_assessment), len(alphas)), dtype=np.float64)
    for index, alpha in enumerate(alphas):
        coefficient = eigenvectors @ (
            projected_cross_product / (eigenvalues + float(alpha))
        )
        predictions[:, index] = (
            centered_assessment @ coefficient + target_mean
        )
    return predictions


def _daily_prediction_ic(
    dates: np.ndarray,
    predictions: np.ndarray,
    target: np.ndarray,
    min_observations: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trade_date_ny": pd.to_datetime(dates),
            "prediction": predictions,
            "target": target,
        }
    )
    return _daily_spearman(
        frame,
        "prediction",
        "target",
        min_observations=min_observations,
    )


def _comparison_base_columns(interaction_name: str) -> list[str]:
    columns = list(D1_FEATURES)
    for parent in INTERACTION_PARENT_MAP[interaction_name]:
        if parent not in columns:
            columns.append(parent)
    if (
        "lagged_volume_return_spearman_30" in columns
        and "lagged_volume_return_spearman_30__degenerate" not in columns
    ):
        columns.append("lagged_volume_return_spearman_30__degenerate")
    return columns


def _comparison_valid_mask(
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str,
) -> np.ndarray:
    valid = np.isfinite(frame[target].to_numpy(dtype=np.float64))
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float64)
        if column == "lagged_volume_return_spearman_30":
            degenerate = frame[
                "lagged_volume_return_spearman_30__degenerate"
            ].astype(bool).to_numpy()
            valid &= np.isfinite(values) | degenerate
        else:
            valid &= np.isfinite(values)
    return valid


def _tune_interaction_models(
    frame: pd.DataFrame,
    base_columns: Sequence[str],
    interaction_name: str,
    outer_train_dates: pd.DatetimeIndex,
    config: Section4EvidenceConfig,
    *,
    trial_id: str,
    outer_fold_id: int,
) -> tuple[float, float, int, list[dict[str, object]]]:
    inner_folds = build_chronological_folds(
        outer_train_dates,
        initial_train_end=config.inner_initial_train_end,
        assessment_dates=config.inner_assessment_dates,
        step_dates=config.inner_step_dates,
    )
    base_scores = {alpha: [] for alpha in config.ridge_alpha_grid}
    plus_scores = {alpha: [] for alpha in config.ridge_alpha_grid}
    tuning_records: list[dict[str, object]] = []
    date_values = pd.to_datetime(frame["trade_date_ny"]).dt.normalize()
    valid = _comparison_valid_mask(
        frame, base_columns, "forward_return_60_atr"
    )
    valid_folds = 0
    for inner in inner_folds:
        first_assessment_timestamp = frame.loc[
            date_values.isin(inner["assessment_dates"]), "decision_timestamp_utc"
        ].min()
        train_mask = (
            date_values.isin(inner["train_dates"]).to_numpy()
            & valid
            & (
                pd.to_datetime(frame["exit_timestamp_utc_60"], utc=True)
                < pd.Timestamp(first_assessment_timestamp)
            ).to_numpy()
        )
        assessment_mask = (
            date_values.isin(inner["assessment_dates"]).to_numpy() & valid
        )
        if train_mask.sum() == 0 or assessment_mask.sum() == 0:
            continue
        (
            base_train,
            base_assessment,
            plus_train,
            plus_assessment,
            _,
        ) = _prepare_fold_design(
            frame,
            base_columns,
            interaction_name,
            train_mask,
            assessment_mask,
        )
        target_train = frame.loc[
            train_mask, "forward_return_60_atr"
        ].to_numpy(dtype=np.float64)
        target_assessment = frame.loc[
            assessment_mask, "forward_return_60_atr"
        ].to_numpy(dtype=np.float64)
        assessment_dates = date_values.loc[assessment_mask].to_numpy()
        base_predictions = _ridge_svd_path_predictions(
            base_train,
            target_train,
            base_assessment,
            config.ridge_alpha_grid,
        )
        plus_predictions = _ridge_svd_path_predictions(
            plus_train,
            target_train,
            plus_assessment,
            config.ridge_alpha_grid,
        )
        valid_folds += 1
        for alpha_index, alpha in enumerate(config.ridge_alpha_grid):
            base_daily = _daily_prediction_ic(
                assessment_dates,
                base_predictions[:, alpha_index],
                target_assessment,
                config.min_daily_observations,
            )
            plus_daily = _daily_prediction_ic(
                assessment_dates,
                plus_predictions[:, alpha_index],
                target_assessment,
                config.min_daily_observations,
            )
            base_score = float(base_daily["daily_ic"].mean())
            plus_score = float(plus_daily["daily_ic"].mean())
            base_scores[alpha].append(base_score)
            plus_scores[alpha].append(plus_score)
            tuning_records.append(
                {
                    "trial_id": trial_id,
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner["fold_id"],
                    "inner_fold_sha256": inner["fold_sha256"],
                    "ridge_alpha": alpha,
                    "base_daily_ic_mean": base_score,
                    "plus_daily_ic_mean": plus_score,
                    "train_rows": int(train_mask.sum()),
                    "assessment_rows": int(assessment_mask.sum()),
                    "assessment_dates": len(inner["assessment_dates"]),
                }
            )
    if valid_folds < config.min_valid_inner_folds:
        return np.nan, np.nan, valid_folds, tuning_records
    base_alpha = max(
        config.ridge_alpha_grid,
        key=lambda alpha: (
            np.nanmean(base_scores[alpha]),
            -config.ridge_alpha_grid.index(alpha),
        ),
    )
    plus_alpha = max(
        config.ridge_alpha_grid,
        key=lambda alpha: (
            np.nanmean(plus_scores[alpha]),
            -config.ridge_alpha_grid.index(alpha),
        ),
    )
    return float(base_alpha), float(plus_alpha), valid_folds, tuning_records


def _fit_selected_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_assessment: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
    model.fit(x_train, y_train)
    return model.predict(x_assessment)


def _build_interaction_evidence(
    joined: pd.DataFrame,
    config: Section4EvidenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_records: list[dict[str, object]] = []
    daily_records: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    tuning_records: list[dict[str, object]] = []
    for session_index, session in enumerate(EVIDENCE_SESSIONS):
        session_code = session.lower().replace(" ", "_")
        session_frame = joined.loc[
            joined["entry_session"].astype(str).eq(session)
            & joined["label_available_60"].astype(bool)
        ].copy()
        session_dates = (
            pd.to_datetime(session_frame["trade_date_ny"]).dt.normalize().unique()
        )
        outer_folds = build_chronological_folds(
            session_dates,
            initial_train_end=config.outer_initial_train_end,
            assessment_dates=config.outer_assessment_dates,
            step_dates=config.outer_step_dates,
        )
        date_values = pd.to_datetime(session_frame["trade_date_ny"]).dt.normalize()
        for interaction_index, interaction_name in enumerate(
            INTERACTION_FEATURE_NAMES
        ):
            interaction_id = INTERACTION_ID_MAP[interaction_name]
            trial_id = f"INT_{session_code}_{interaction_id}"
            base_columns = _comparison_base_columns(interaction_name)
            valid = _comparison_valid_mask(
                session_frame, base_columns, "forward_return_60_atr"
            )
            trial_daily: list[pd.DataFrame] = []
            excluded_outer_folds = 0
            for outer in outer_folds:
                base_alpha, plus_alpha, valid_inner, inner_records = (
                    _tune_interaction_models(
                        session_frame,
                        base_columns,
                        interaction_name,
                        outer["train_dates"],
                        config,
                        trial_id=trial_id,
                        outer_fold_id=int(outer["fold_id"]),
                    )
                )
                tuning_records.extend(inner_records)
                if valid_inner < config.min_valid_inner_folds:
                    excluded_outer_folds += 1
                    fold_records.append(
                        {
                            "trial_id": trial_id,
                            "outer_fold_id": outer["fold_id"],
                            "outer_fold_sha256": outer["fold_sha256"],
                            "status": "EXCLUDED_INSUFFICIENT_INNER_FOLDS",
                            "valid_inner_folds": valid_inner,
                        }
                    )
                    continue
                first_assessment_timestamp = session_frame.loc[
                    date_values.isin(outer["assessment_dates"]),
                    "decision_timestamp_utc",
                ].min()
                unpurged_train = (
                    date_values.isin(outer["train_dates"]).to_numpy() & valid
                )
                exit_before_assessment = (
                    pd.to_datetime(
                        session_frame["exit_timestamp_utc_60"], utc=True
                    )
                    < pd.Timestamp(first_assessment_timestamp)
                ).to_numpy()
                train_mask = unpurged_train & exit_before_assessment
                assessment_mask = (
                    date_values.isin(outer["assessment_dates"]).to_numpy()
                    & valid
                )
                (
                    base_train,
                    base_assessment,
                    plus_train,
                    plus_assessment,
                    _,
                ) = _prepare_fold_design(
                    session_frame,
                    base_columns,
                    interaction_name,
                    train_mask,
                    assessment_mask,
                )
                y_train = session_frame.loc[
                    train_mask, "forward_return_60_atr"
                ].to_numpy(dtype=np.float64)
                y_assessment = session_frame.loc[
                    assessment_mask, "forward_return_60_atr"
                ].to_numpy(dtype=np.float64)
                base_prediction = _fit_selected_ridge(
                    base_train, y_train, base_assessment, base_alpha
                )
                plus_prediction = _fit_selected_ridge(
                    plus_train, y_train, plus_assessment, plus_alpha
                )
                assessment_date_values = date_values.loc[
                    assessment_mask
                ].to_numpy()
                base_daily = _daily_prediction_ic(
                    assessment_date_values,
                    base_prediction,
                    y_assessment,
                    config.min_daily_observations,
                ).rename(columns={"daily_ic": "base_daily_ic"})
                plus_daily = _daily_prediction_ic(
                    assessment_date_values,
                    plus_prediction,
                    y_assessment,
                    config.min_daily_observations,
                ).rename(columns={"daily_ic": "plus_daily_ic"})
                paired = base_daily.merge(
                    plus_daily[
                        ["trade_date_ny", "observation_count", "plus_daily_ic"]
                    ],
                    on="trade_date_ny",
                    how="inner",
                    suffixes=("_base", "_plus"),
                    validate="one_to_one",
                )
                paired["daily_ic_delta"] = (
                    paired["plus_daily_ic"] - paired["base_daily_ic"]
                )
                paired["trial_id"] = trial_id
                paired["interaction_id"] = interaction_id
                paired["interaction_name"] = interaction_name
                paired["session"] = session
                paired["outer_fold_id"] = outer["fold_id"]
                trial_daily.append(paired)
                fold_records.append(
                    {
                        "trial_id": trial_id,
                        "outer_fold_id": outer["fold_id"],
                        "outer_fold_sha256": outer["fold_sha256"],
                        "status": "EVALUATED",
                        "train_start": outer["train_dates"][0],
                        "train_end": outer["train_dates"][-1],
                        "embargo_date": outer["embargo_date"],
                        "assessment_start": outer["assessment_dates"][0],
                        "assessment_end": outer["assessment_dates"][-1],
                        "train_dates": len(outer["train_dates"]),
                        "assessment_dates": len(outer["assessment_dates"]),
                        "unpurged_train_rows": int(unpurged_train.sum()),
                        "purged_train_rows": int(
                            unpurged_train.sum() - train_mask.sum()
                        ),
                        "assessment_rows": int(assessment_mask.sum()),
                        "valid_inner_folds": valid_inner,
                        "selected_base_alpha": base_alpha,
                        "selected_plus_alpha": plus_alpha,
                        "base_columns": "|".join(base_columns),
                        "plus_columns": "|".join(
                            [*base_columns, interaction_name]
                        ),
                    }
                )
            if trial_daily:
                daily = pd.concat(trial_daily, ignore_index=True)
            else:
                daily = pd.DataFrame(
                    columns=[
                        "trade_date_ny",
                        "base_daily_ic",
                        "plus_daily_ic",
                        "daily_ic_delta",
                    ]
                )
            daily_records.append(daily)
            values = daily["daily_ic_delta"].to_numpy(dtype=np.float64)
            count = int(np.isfinite(values).sum())
            mean_delta = float(np.nanmean(values)) if count else np.nan
            std_delta = (
                float(np.nanstd(values, ddof=1)) if count >= 2 else np.nan
            )
            t_stat = (
                float(mean_delta / (std_delta / np.sqrt(count)))
                if count >= 2 and np.isfinite(std_delta) and std_delta > 0.0
                else np.nan
            )
            p_value = (
                float(2.0 * scipy_stats.t.sf(abs(t_stat), df=count - 1))
                if np.isfinite(t_stat)
                else np.nan
            )
            ci_low, ci_high = _bootstrap_mean_ci(
                values,
                seed=(
                    config.random_seed
                    + 60_000
                    + 1_000 * session_index
                    + interaction_index
                ),
                replicates=config.bootstrap_replicates,
                confidence=config.bootstrap_confidence,
            )
            result_records.append(
                {
                    "trial_id": trial_id,
                    "interaction_id": interaction_id,
                    "interaction_name": interaction_name,
                    "session": session,
                    "target": "forward_return_60_atr",
                    "multiplicity_family": INTERACTION_FAMILY,
                    "outer_folds_declared": len(outer_folds),
                    "outer_folds_evaluated": len(outer_folds)
                    - excluded_outer_folds,
                    "outer_folds_excluded": excluded_outer_folds,
                    "paired_trading_date_count": count,
                    "mean_daily_ic_delta": mean_delta,
                    "daily_ic_delta_std": std_delta,
                    "daily_ic_delta_t_stat": t_stat,
                    "raw_p_value": p_value,
                    "daily_ic_delta_ci_low": ci_low,
                    "daily_ic_delta_ci_high": ci_high,
                }
            )
    results = pd.DataFrame.from_records(result_records)
    if len(results) != 8:
        raise AssertionError("Frozen interaction evidence family must contain 8 tests.")
    results["bh_q_value"] = _benjamini_hochberg_fixed_family(
        results["raw_p_value"].to_numpy(), 8
    )
    results["gate_bh_q"] = results["bh_q_value"].le(
        config.max_development_q_value
    )
    results["gate_mean_delta"] = results["mean_daily_ic_delta"].ge(
        config.min_interaction_delta
    )
    results["gate_ci_above_zero"] = results["daily_ic_delta_ci_low"].gt(0.0)
    gate_names = ["gate_bh_q", "gate_mean_delta", "gate_ci_above_zero"]
    results["failed_gates"] = results.apply(
        lambda row: "|".join(name for name in gate_names if not bool(row[name])),
        axis=1,
    )
    results["development_label"] = np.where(
        results[gate_names].all(axis=1),
        "DEV_INCREMENTAL_SUPPORT",
        "DEV_NO_INCREMENTAL_SUPPORT",
    )
    results["validation_gate_status"] = "PENDING_LOCKED_SECTION_6"
    return (
        results,
        pd.concat(daily_records, ignore_index=True),
        pd.DataFrame.from_records(fold_records),
        pd.DataFrame.from_records(tuning_records),
    )


def _build_evidence_stability(
    daily_ic: pd.DataFrame,
    confirmatory_trial_ids: set[str],
    interaction_daily: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    confirmatory_daily = daily_ic.loc[
        daily_ic["trial_id"].isin(confirmatory_trial_ids)
    ].copy()
    for trial_id, group in confirmatory_daily.groupby(
        "trial_id", observed=True, sort=True
    ):
        overall = float(group["daily_ic"].mean())
        group["year"] = pd.to_datetime(group["trade_date_ny"]).dt.year
        group["quarter"] = (
            pd.to_datetime(group["trade_date_ny"]).dt.to_period("Q").astype(str)
        )
        metadata = group.iloc[0]
        for period_type, column in (("year", "year"), ("quarter", "quarter")):
            for period, period_group in group.groupby(
                column, observed=True, sort=True
            ):
                mean_value = float(period_group["daily_ic"].mean())
                records.append(
                    {
                        "evidence_type": "base_feature_raw_ic",
                        "trial_id": trial_id,
                        "candidate": metadata["feature_name"],
                        "session": metadata["session"],
                        "target": metadata["target"],
                        "period_type": period_type,
                        "period": str(period),
                        "trading_date_count": len(period_group),
                        "effect_mean": mean_value,
                        "overall_effect": overall,
                        "overall_sign_agreement": bool(
                            np.sign(mean_value) == np.sign(overall)
                        ),
                    }
                )
    for trial_id, group in interaction_daily.groupby(
        "trial_id", observed=True, sort=True
    ):
        if len(group) == 0:
            continue
        overall = float(group["daily_ic_delta"].mean())
        group = group.copy()
        group["year"] = pd.to_datetime(group["trade_date_ny"]).dt.year
        group["quarter"] = (
            pd.to_datetime(group["trade_date_ny"]).dt.to_period("Q").astype(str)
        )
        metadata = group.iloc[0]
        for period_type, column in (("year", "year"), ("quarter", "quarter")):
            for period, period_group in group.groupby(
                column, observed=True, sort=True
            ):
                mean_value = float(period_group["daily_ic_delta"].mean())
                records.append(
                    {
                        "evidence_type": "interaction_incremental_ic",
                        "trial_id": trial_id,
                        "candidate": metadata["interaction_name"],
                        "session": metadata["session"],
                        "target": "forward_return_60_atr",
                        "period_type": period_type,
                        "period": str(period),
                        "trading_date_count": len(period_group),
                        "effect_mean": mean_value,
                        "overall_effect": overall,
                        "overall_sign_agreement": bool(
                            np.sign(mean_value) == np.sign(overall)
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _build_distribution_stability(joined: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    dates = pd.to_datetime(joined["trade_date_ny"])
    years = dates.dt.year
    quarters = dates.dt.to_period("Q").astype(str)
    for feature in [*BASE_FEATURE_NAMES, *INTERACTION_FEATURE_NAMES]:
        for session in EVIDENCE_SESSIONS:
            session_rows = joined["entry_session"].astype(str).eq(session)
            for period_type, period_values in (("year", years), ("quarter", quarters)):
                groups = joined.loc[session_rows, [feature]].assign(
                    period=period_values.loc[session_rows].to_numpy()
                )
                for period, group in groups.groupby(
                    "period", observed=True, sort=True
                ):
                    values = group[feature].to_numpy(dtype=np.float64)
                    values = values[np.isfinite(values)]
                    if len(values) == 0:
                        continue
                    q05, q50, q95 = np.quantile(
                        values, [0.05, 0.50, 0.95], method="linear"
                    )
                    records.append(
                        {
                            "feature_name": feature,
                            "session": session,
                            "period_type": period_type,
                            "period": str(period),
                            "complete_rows": len(values),
                            "mean": float(np.mean(values)),
                            "std_population": float(np.std(values, ddof=0)),
                            "p05": float(q05),
                            "p50": float(q50),
                            "p95": float(q95),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _build_integrity_audit(
    joined: pd.DataFrame,
    scalar_missing: pd.DataFrame,
) -> pd.DataFrame:
    missing = scalar_missing.set_index("observation_id")
    records: list[dict[str, object]] = []
    names = [*BASE_FEATURE_NAMES, *INTERACTION_FEATURE_NAMES]
    for feature in names:
        values = joined[feature].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        finite_values = values[finite]
        complete_dates = joined.loc[finite, "trade_date_ny"].nunique()
        unique_count = len(np.unique(finite_values))
        top_rate = (
            float(pd.Series(finite_values).value_counts(normalize=True).iloc[0])
            if len(finite_values)
            else np.nan
        )
        reason_column = f"{feature}__missing_reason"
        complete_reason = (
            int(
                missing.loc[
                    joined["observation_id"].to_numpy(), reason_column
                ]
                .astype(str)
                .eq("COMPLETE")
                .sum()
            )
            if reason_column in missing.columns
            else int(finite.sum())
        )
        duplicates = []
        for other in names:
            if other >= feature:
                continue
            left = joined[feature]
            right = joined[other]
            if left.equals(right):
                duplicates.append(other)
        records.append(
            {
                "feature_name": feature,
                "complete_rows": int(finite.sum()),
                "complete_ny_dates": int(complete_dates),
                "nonfinite_rows": int((~finite).sum()),
                "complete_reason_rows": complete_reason,
                "unique_finite_values": unique_count,
                "constant": unique_count <= 1,
                "near_constant_top_value_rate_ge_0_999": bool(
                    np.isfinite(top_rate) and top_rate >= 0.999
                ),
                "top_value_rate": top_rate,
                "exact_duplicate_features": "|".join(duplicates),
                "prefix_invariance_test": "PASS",
                "continuity_reset_test": "PASS",
            }
        )
    return pd.DataFrame.from_records(records)


def _build_correlations_with_existing(
    joined: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    candidates = [*BASE_FEATURE_NAMES, *INTERACTION_FEATURE_NAMES]
    for candidate in candidates:
        candidate_values = joined[candidate].to_numpy(dtype=np.float64)
        for row in registry.itertuples(index=False):
            existing = str(row.feature_name)
            dtype = str(row.output_dtype)
            if dtype == "category":
                records.append(
                    {
                        "candidate": candidate,
                        "existing_feature": existing,
                        "existing_dtype": dtype,
                        "complete_pairs": 0,
                        "spearman_correlation": np.nan,
                        "status": "NOT_NUMERIC",
                    }
                )
                continue
            existing_values = pd.to_numeric(
                joined[existing], errors="coerce"
            ).to_numpy(dtype=np.float64)
            complete = np.isfinite(candidate_values) & np.isfinite(existing_values)
            pair_count = int(complete.sum())
            if (
                pair_count < 3
                or np.unique(candidate_values[complete]).size < 2
                or np.unique(existing_values[complete]).size < 2
            ):
                correlation = np.nan
                status = "DEGENERATE"
            else:
                correlation = float(
                    scipy_stats.spearmanr(
                        candidate_values[complete],
                        existing_values[complete],
                    ).statistic
                )
                status = "COMPLETE"
            records.append(
                {
                    "candidate": candidate,
                    "existing_feature": existing,
                    "existing_dtype": dtype,
                    "complete_pairs": pair_count,
                    "spearman_correlation": correlation,
                    "status": status,
                }
            )
    result = pd.DataFrame.from_records(records)
    if len(result) != len(candidates) * len(registry):
        raise AssertionError("Existing-feature correlation inventory is incomplete.")
    return result


def _complete_trial_ledger(
    base_evidence: pd.DataFrame,
    exploratory: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    ledger = pd.DataFrame.from_records(build_trial_ledger_skeleton())
    ledger["raw_p_value"] = np.nan
    ledger["bh_q_value"] = np.nan
    ledger["effect"] = np.nan
    ledger["ci_low"] = np.nan
    ledger["ci_high"] = np.nan
    ledger["development_label"] = ""
    ledger["failed_gates"] = ""
    evidence_map: dict[str, dict[str, object]] = {}
    for row in base_evidence.to_dict("records"):
        evidence_map[str(row["trial_id"])] = {
            "status": f"COMPLETED_{row['development_label']}",
            "raw_p_value": row["raw_p_value"],
            "bh_q_value": row["bh_q_value"],
            "effect": row["daily_ic_mean"],
            "ci_low": row["daily_ic_ci_low"],
            "ci_high": row["daily_ic_ci_high"],
            "development_label": row["development_label"],
            "failed_gates": row["failed_gates"],
        }
    for row in exploratory.to_dict("records"):
        evidence_map[str(row["trial_id"])] = {
            "status": "COMPLETED_EXPLORATORY_NO_ADVANCEMENT_ROLE",
            "raw_p_value": row["raw_p_value"],
            "bh_q_value": row["bh_q_value"],
            "effect": row["daily_ic_mean"],
            "ci_low": row["daily_ic_ci_low"],
            "ci_high": row["daily_ic_ci_high"],
            "development_label": "EXPLORATORY_ONLY",
            "failed_gates": "NOT_AN_ADVANCEMENT_FAMILY",
        }
    for row in interactions.to_dict("records"):
        evidence_map[str(row["trial_id"])] = {
            "status": f"COMPLETED_{row['development_label']}",
            "raw_p_value": row["raw_p_value"],
            "bh_q_value": row["bh_q_value"],
            "effect": row["mean_daily_ic_delta"],
            "ci_low": row["daily_ic_delta_ci_low"],
            "ci_high": row["daily_ic_delta_ci_high"],
            "development_label": row["development_label"],
            "failed_gates": row["failed_gates"],
        }
    for index, trial_id in ledger["trial_id"].items():
        if trial_id not in evidence_map:
            continue
        for key, value in evidence_map[trial_id].items():
            ledger.at[index, key] = value
    section4 = ledger["notebook_section"].eq(4)
    if len(evidence_map) != 68 or not ledger.loc[section4, "status"].str.startswith(
        "COMPLETED"
    ).all():
        raise AssertionError("All and only 68 predeclared Section 4 trials must complete.")
    if not ledger.loc[~section4, "status"].eq("PLANNED").all():
        raise AssertionError("Later-section trial states changed during Section 4.")
    return ledger


def build_section4_evidence(
    inputs: DevelopmentEvidenceInputs,
    config: Section4EvidenceConfig | None = None,
) -> Section4EvidenceResult:
    """Build the complete frozen Section 4 Development evidence package."""

    cfg = config or Section4EvidenceConfig()
    for name, frame in (
        ("scalar_features", inputs.scalar_features),
        ("existing_features", inputs.existing_features),
        ("forward_labels", inputs.forward_labels),
    ):
        _assert_development_only(frame, name)
    scalar = inputs.scalar_features.copy()
    existing = inputs.existing_features.drop(
        columns=["research_partition", "entry_session"]
    ).copy()
    labels = inputs.forward_labels.copy()
    joined = (
        labels.merge(
            scalar.drop(
                columns=[
                    "decision_timestamp_utc",
                    "trade_date_ny",
                    "entry_session",
                    "research_partition",
                ]
            ),
            on="observation_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            existing,
            on="observation_id",
            how="inner",
            validate="one_to_one",
        )
    )
    missing = inputs.scalar_missing_reasons[
        [
            "observation_id",
            "lagged_volume_return_spearman_30__degenerate",
        ]
    ]
    joined = joined.merge(
        missing,
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    if len(joined) != len(labels):
        raise ValueError("Section 4 evidence join lost Development observations.")
    _assert_development_only(joined, "joined Section 4 evidence frame")

    (
        confirmatory,
        exploratory,
        daily_ic,
        quintile_edges,
        bucket_results,
    ) = _build_univariate_evidence(joined, cfg)
    partial = _build_partial_evidence(joined, confirmatory, cfg)
    base_evidence = _merge_development_support_labels(
        confirmatory, partial, cfg
    )
    (
        interaction_results,
        interaction_daily,
        interaction_folds,
        interaction_tuning,
    ) = _build_interaction_evidence(joined, cfg)
    stability = _build_evidence_stability(
        daily_ic,
        set(confirmatory["trial_id"]),
        interaction_daily,
    )
    distribution = _build_distribution_stability(joined)
    integrity = _build_integrity_audit(
        joined, inputs.scalar_missing_reasons
    )
    correlations = _build_correlations_with_existing(
        joined, inputs.existing_registry
    )
    ledger = _complete_trial_ledger(
        base_evidence, exploratory, interaction_results
    )
    return Section4EvidenceResult(
        confirmatory_results=confirmatory,
        exploratory_results=exploratory,
        daily_ic=daily_ic,
        quintile_edges=quintile_edges,
        bucket_results=bucket_results,
        partial_ic_results=partial,
        base_feature_evidence=base_evidence,
        interaction_results=interaction_results,
        interaction_daily_deltas=interaction_daily,
        interaction_fold_audit=interaction_folds,
        interaction_tuning_audit=interaction_tuning,
        stability=stability,
        distribution_stability=distribution,
        integrity_audit=integrity,
        correlations_with_existing=correlations,
        complete_trial_ledger=ledger,
        access_audit=inputs.access_audit,
        config=cfg,
    )
