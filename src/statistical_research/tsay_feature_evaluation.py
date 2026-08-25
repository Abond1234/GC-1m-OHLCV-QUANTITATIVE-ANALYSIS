"""Frozen Development-only evidence procedures for Tsay features.

This module implements the Stage 1 contract mechanically.  It does not select
features, alter orientations, or fit predictive/trading models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .tsay_feature_registry import (
    D1_INPUTS_BY_SESSION,
    O1_INPUTS_BY_SESSION,
    SESSIONS,
    STATIONARY_BOOTSTRAP_REPLICATES,
    STATIONARY_BOOTSTRAP_RESTART_PROBABILITY,
    TSAY_FEATURE_SPECS,
    TSAY_SEED,
)

MIN_DAILY_OBSERVATIONS: Final = 10
MIN_VALID_DATES: Final = 200
MIN_VALID_OBSERVATIONS: Final = 10_000
BH_Q_GATE: Final = 0.10
MONOTONICITY_GATE: Final = 0.80
PARTIAL_RETENTION_GATE: Final = 0.25
MAX_CONDITION_NUMBER: Final = 1.0e12
CONCENTRATION_GATE: Final = 0.50
CLOCK_BINS_BY_SESSION: Final = {
    "London": tuple(range(11, 24)),
    "New York": tuple(range(27, 48)),
}
CLOCK_REFERENCE_BIN_BY_SESSION: Final = {"London": 11, "New York": 27}

ROLE_TARGETS: Final = {
    "directional": {
        "primary": "forward_return_60_atr",
        "tick": "forward_return_60_ticks",
        "diagnostic": "forward_return_30_atr",
    },
    "opportunity": {
        "primary": "future_range_60_atr",
        "tick": "future_range_60_ticks",
        "diagnostic": "future_range_30_atr",
    },
}


@dataclass(frozen=True)
class TsayEvidenceResult:
    """All compact Stage 3 evidence artifacts."""

    evidence_ledger: pd.DataFrame
    primary_daily_ic: pd.DataFrame
    diagnostic_daily_ic: pd.DataFrame
    diagnostic_summary: pd.DataFrame
    year_stability: pd.DataFrame
    quintile_edges: pd.DataFrame
    quintile_summary: pd.DataFrame
    partial_daily_ic: pd.DataFrame
    partial_summary: pd.DataFrame
    related_feature_redundancy: pd.DataFrame
    candidate_correlation: pd.DataFrame


def _finite(values: pd.Series) -> np.ndarray:
    return np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    return _pearson(
        rankdata(left, method="average").astype(np.float64),
        rankdata(right, method="average").astype(np.float64),
    )


def stationary_bootstrap_indices(
    n_dates: int,
    *,
    replicates: int = STATIONARY_BOOTSTRAP_REPLICATES,
    restart_probability: float = STATIONARY_BOOTSTRAP_RESTART_PROBABILITY,
    seed: int = TSAY_SEED,
) -> np.ndarray:
    """Return deterministic stationary-bootstrap indices over ordered dates."""

    if n_dates < 1:
        raise ValueError("Stationary bootstrap requires at least one date.")
    if replicates < 1:
        raise ValueError("Stationary bootstrap requires at least one replicate.")
    if not 0.0 < restart_probability <= 1.0:
        raise ValueError("Restart probability must lie in (0, 1].")
    rng = np.random.default_rng(seed)
    indices = np.empty((replicates, n_dates), dtype=np.int32)
    indices[:, 0] = rng.integers(0, n_dates, size=replicates, dtype=np.int32)
    for column in range(1, n_dates):
        restarts = rng.random(replicates) < restart_probability
        continuing = (indices[:, column - 1] + 1) % n_dates
        starts = rng.integers(0, n_dates, size=replicates, dtype=np.int32)
        indices[:, column] = np.where(restarts, starts, continuing)
    return indices


def stationary_bootstrap_mean_inference(
    values: np.ndarray,
    *,
    replicates: int = STATIONARY_BOOTSTRAP_REPLICATES,
    restart_probability: float = STATIONARY_BOOTSTRAP_RESTART_PROBABILITY,
    seed: int = TSAY_SEED,
) -> dict[str, float | int]:
    """Compute the frozen percentile interval and orientation-aware null p-value."""

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or len(series) == 0 or not np.isfinite(series).all():
        return {
            "observed_mean": float("nan"),
            "bootstrap_lower": float("nan"),
            "bootstrap_upper": float("nan"),
            "null_p_value": float("nan"),
            "replicates": replicates,
            "seed": seed,
        }
    indices = stationary_bootstrap_indices(
        len(series),
        replicates=replicates,
        restart_probability=restart_probability,
        seed=seed,
    )
    observed = float(series.mean())
    uncentered_means = series[indices].mean(axis=1)
    centered = series - observed
    null_means = centered[indices].mean(axis=1)
    return {
        "observed_mean": observed,
        "bootstrap_lower": float(np.percentile(uncentered_means, 2.5)),
        "bootstrap_upper": float(np.percentile(uncentered_means, 97.5)),
        "null_p_value": float((1 + np.count_nonzero(null_means >= observed)) / (replicates + 1)),
        "replicates": replicates,
        "seed": seed,
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return monotone BH q-values without changing the 18-test family."""

    values = np.asarray(p_values, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return result
    ordered_positions = finite_positions[np.argsort(values[finite_positions], kind="mergesort")]
    ordered = values[ordered_positions]
    family_count = len(values)
    adjusted = ordered * family_count / np.arange(1, len(ordered) + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[ordered_positions] = np.minimum(adjusted, 1.0)
    return result


def daily_spearman(
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    minimum_observations: int = MIN_DAILY_OBSERVATIONS,
) -> pd.DataFrame:
    """Calculate daily Spearman IC on exact common finite rows."""

    records: list[dict[str, object]] = []
    for trade_date, group in frame.groupby("trade_date_ny", sort=True, observed=True):
        finite = _finite(group[feature]) & _finite(group[target])
        count = int(finite.sum())
        correlation = float("nan")
        status = "INSUFFICIENT_OBSERVATIONS"
        if count >= minimum_observations:
            correlation = _spearman(
                group.loc[finite, feature].to_numpy(dtype=np.float64),
                group.loc[finite, target].to_numpy(dtype=np.float64),
            )
            status = "AVAILABLE" if np.isfinite(correlation) else "ZERO_VARIANCE"
        records.append(
            {
                "trade_date_ny": pd.Timestamp(trade_date),
                "observations": count,
                "ic": correlation,
                "status": status,
            }
        )
    return pd.DataFrame.from_records(records)


def frozen_quintile_edges(values: np.ndarray) -> np.ndarray:
    """Fit the four noninterpolated one-based Development order statistics."""

    finite = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if len(finite) == 0:
        return np.full(4, np.nan, dtype=np.float64)
    positions = np.ceil(np.arange(1, 5, dtype=np.float64) * len(finite) / 5.0).astype(int)
    return finite[positions - 1]


def assign_frozen_quintiles(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign 1 + count(value > edge), retaining ties in the lower bin."""

    array = np.asarray(values, dtype=np.float64)
    fitted = np.asarray(edges, dtype=np.float64)
    result = np.full(len(array), -1, dtype=np.int8)
    finite = np.isfinite(array)
    if np.isfinite(fitted).all():
        result[finite] = 1 + (array[finite, None] > fitted[None, :]).sum(axis=1)
    return result


def _primary_concentration(daily: pd.DataFrame) -> tuple[float, float]:
    available = daily.loc[np.isfinite(daily["oriented_ic"])].copy()
    available["positive"] = np.maximum(available["oriented_ic"], 0.0)
    total = float(available["positive"].sum())
    if total <= 0.0:
        return float("nan"), float("nan")
    year_share = float(
        available.groupby(available["trade_date_ny"].dt.year)["positive"].sum().max() / total
    )
    ten_share = float(available["positive"].nlargest(10).sum() / total)
    return year_share, ten_share


def _partial_daily_ic(
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    controls: tuple[str, ...],
    clock_bins: tuple[int, ...],
    reference_clock_bin: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    calculated_clock_bins = (
        pd.to_datetime(frame["decision_timestamp_ny"], errors="raise").dt.hour * 60
        + pd.to_datetime(frame["decision_timestamp_ny"], errors="raise").dt.minute
    ) // 15
    if not set(calculated_clock_bins.unique()).issubset(set(clock_bins)):
        raise ValueError("A decision timestamp lies outside the frozen session clock bins.")
    if reference_clock_bin not in clock_bins:
        raise ValueError("The frozen reference clock bin is absent from the session bins.")
    dummy_bins = tuple(value for value in clock_bins if value != reference_clock_bin)
    working = frame.assign(_clock_bin=calculated_clock_bins.to_numpy(dtype=np.int16))

    for trade_date, group in working.groupby("trade_date_ny", sort=True, observed=True):
        common = _finite(group[feature]) & _finite(group[target])
        for control in controls:
            common &= _finite(group[control])
        selected = group.loc[common]
        count = len(selected)
        continuous_columns: list[np.ndarray] = []
        retained_controls: list[str] = []
        for control in controls:
            values = selected[control].to_numpy(dtype=np.float64)
            ranked = (rankdata(values, method="average") - 0.5) / max(count, 1)
            if count and float(np.ptp(ranked)) > 0.0:
                continuous_columns.append(ranked)
                retained_controls.append(control)
        for clock_bin in dummy_bins:
            values = (selected["_clock_bin"].to_numpy() == clock_bin).astype(np.float64)
            if count and float(np.ptp(values)) > 0.0:
                continuous_columns.append(values)
                retained_controls.append(f"clock_bin_{clock_bin}")

        retained_count = len(retained_controls)
        required_count = max(30, retained_count + 10)
        record: dict[str, object] = {
            "trade_date_ny": pd.Timestamp(trade_date),
            "observations": count,
            "retained_control_columns": retained_count,
            "minimum_required_observations": required_count,
            "design_rank": 0,
            "design_columns": retained_count + 1,
            "condition_number": float("nan"),
            "reference_clock_bin": reference_clock_bin,
            "partial_ic": float("nan"),
            "matched_unadjusted_ic": float("nan"),
            "status": "INSUFFICIENT_OBSERVATIONS",
        }
        if count < required_count:
            records.append(record)
            continue

        controls_matrix = (
            np.column_stack(continuous_columns)
            if continuous_columns
            else np.empty((count, 0), dtype=np.float64)
        )
        design = np.column_stack((np.ones(count, dtype=np.float64), controls_matrix))
        singular_values = np.linalg.svd(design, compute_uv=False)
        rank = int(np.linalg.matrix_rank(design))
        condition = float(
            singular_values[0] / singular_values[-1] if singular_values[-1] > 0.0 else np.inf
        )
        record["design_rank"] = rank
        record["condition_number"] = condition
        if rank != design.shape[1]:
            record["status"] = "RANK_DEFICIENT"
            records.append(record)
            continue
        if not np.isfinite(condition) or condition > MAX_CONDITION_NUMBER:
            record["status"] = "ILL_CONDITIONED"
            records.append(record)
            continue

        candidate_rank = (
            rankdata(selected[feature].to_numpy(dtype=np.float64), method="average") - 0.5
        ) / count
        target_rank = (
            rankdata(selected[target].to_numpy(dtype=np.float64), method="average") - 0.5
        ) / count
        candidate_beta = np.linalg.lstsq(design, candidate_rank, rcond=None)[0]
        target_beta = np.linalg.lstsq(design, target_rank, rcond=None)[0]
        candidate_residual = candidate_rank - design @ candidate_beta
        target_residual = target_rank - design @ target_beta
        partial = _pearson(candidate_residual, target_residual)
        matched = _pearson(candidate_rank, target_rank)
        record["partial_ic"] = partial
        record["matched_unadjusted_ic"] = matched
        record["status"] = "AVAILABLE" if np.isfinite(partial) else "ZERO_RESIDUAL_VARIANCE"
        records.append(record)
    return pd.DataFrame.from_records(records)


def _summarize_diagnostic(daily: pd.DataFrame) -> dict[str, object]:
    available = daily.loc[np.isfinite(daily["ic"])].copy()
    if available.empty:
        return {
            "valid_dates": 0,
            "valid_observations": 0,
            "mean_ic": float("nan"),
            "median_ic": float("nan"),
            "standard_deviation_ic": float("nan"),
            "oriented_sign_fraction": float("nan"),
        }
    return {
        "valid_dates": len(available),
        "valid_observations": int(available["observations"].sum()),
        "mean_ic": float(available["ic"].mean()),
        "median_ic": float(available["ic"].median()),
        "standard_deviation_ic": float(available["ic"].std(ddof=1)),
        "oriented_sign_fraction": float((available["oriented_ic"] > 0.0).mean()),
    }


def _related_redundancy(
    frame: pd.DataFrame,
    *,
    feature: str,
    controls: tuple[str, ...],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for control in controls:
        common = _finite(frame[feature]) & _finite(frame[control])
        pooled = _spearman(
            frame.loc[common, feature].to_numpy(dtype=np.float64),
            frame.loc[common, control].to_numpy(dtype=np.float64),
        )
        daily = daily_spearman(frame, feature=feature, target=control)
        available = daily.loc[np.isfinite(daily["ic"])]
        records.append(
            {
                "related_feature": control,
                "pooled_spearman": pooled,
                "common_observations": int(common.sum()),
                "valid_dates": len(available),
                "mean_daily_spearman": (
                    float(available["ic"].mean()) if len(available) else float("nan")
                ),
            }
        )
    return records


def _validate_development_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "observation_id",
        "decision_timestamp_ny",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "product",
        "decision_atr_20m",
        "label_available_60",
        "label_available_30",
    }
    required.update(spec.feature_name for spec in TSAY_FEATURE_SPECS)
    for targets in ROLE_TARGETS.values():
        required.update(targets.values())
    absent = sorted(required - set(frame.columns))
    if absent:
        raise ValueError(f"Stage 3 evidence frame is missing columns: {absent}")
    if frame["observation_id"].duplicated().any():
        raise ValueError("Stage 3 evidence frame must be one row per observation_id.")
    partitions = set(frame["research_partition"].astype(str).unique())
    if partitions != {"Development"}:
        raise ValueError(f"Only Development rows are authorized in Stage 3: {partitions}")
    products = set(frame["product"].astype(str).unique())
    if products != {"GC"}:
        raise ValueError(f"Only GC rows are authorized in Stage 3: {products}")
    dates = pd.to_datetime(frame["trade_date_ny"], errors="raise")
    if len(dates) and dates.max() > pd.Timestamp("2023-12-31"):
        raise ValueError("Post-Development rows are forbidden in Stage 3.")
    return frame.assign(trade_date_ny=dates).sort_values(
        ["trade_date_ny", "decision_timestamp_ny", "observation_id"], kind="mergesort"
    )


def evaluate_development_feature_evidence(frame: pd.DataFrame) -> TsayEvidenceResult:
    """Run the exact frozen 18-test Development feature-evidence family."""

    working = _validate_development_frame(frame)
    primary_daily_frames: list[pd.DataFrame] = []
    diagnostic_daily_frames: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, object]] = []
    year_records: list[dict[str, object]] = []
    edge_records: list[dict[str, object]] = []
    quintile_records: list[dict[str, object]] = []
    partial_daily_frames: list[pd.DataFrame] = []
    partial_records: list[dict[str, object]] = []
    redundancy_records: list[dict[str, object]] = []
    ledger_records: list[dict[str, object]] = []

    for spec in TSAY_FEATURE_SPECS:
        targets = ROLE_TARGETS[spec.role]
        for session in SESSIONS:
            session_frame = working.loc[working["entry_session"] == session].copy()
            primary_eligible = session_frame["label_available_60"].fillna(False).astype(bool)
            diagnostic_eligible = session_frame["label_available_30"].fillna(False).astype(bool)
            primary_frame = session_frame.loc[primary_eligible]
            diagnostic_frame = session_frame.loc[diagnostic_eligible]

            primary_daily = daily_spearman(
                primary_frame,
                feature=spec.feature_name,
                target=targets["primary"],
            )
            primary_daily.insert(0, "logical_id", spec.logical_id)
            primary_daily.insert(1, "feature_name", spec.feature_name)
            primary_daily.insert(2, "entry_session", session)
            primary_daily.insert(3, "target", targets["primary"])
            primary_daily["oriented_ic"] = spec.expected_sign * primary_daily["ic"]
            primary_daily_frames.append(primary_daily)
            primary_available = primary_daily.loc[np.isfinite(primary_daily["ic"])].copy()
            oriented_values = primary_available["oriented_ic"].to_numpy(dtype=np.float64)
            inference = stationary_bootstrap_mean_inference(oriented_values)
            year_share, ten_share = _primary_concentration(primary_available)

            for year, year_group in primary_available.groupby(
                primary_available["trade_date_ny"].dt.year, sort=True
            ):
                year_records.append(
                    {
                        "logical_id": spec.logical_id,
                        "feature_name": spec.feature_name,
                        "entry_session": session,
                        "year": int(year),
                        "valid_dates": len(year_group),
                        "valid_observations": int(year_group["observations"].sum()),
                        "mean_ic": float(year_group["ic"].mean()),
                        "mean_oriented_ic": float(year_group["oriented_ic"].mean()),
                        "oriented_positive_fraction": float(
                            (year_group["oriented_ic"] > 0.0).mean()
                        ),
                        "positive_evidence": float(
                            np.maximum(year_group["oriented_ic"], 0.0).sum()
                        ),
                    }
                )

            diagnostic_daily = daily_spearman(
                diagnostic_frame,
                feature=spec.feature_name,
                target=targets["diagnostic"],
            )
            diagnostic_daily.insert(0, "logical_id", spec.logical_id)
            diagnostic_daily.insert(1, "feature_name", spec.feature_name)
            diagnostic_daily.insert(2, "entry_session", session)
            diagnostic_daily.insert(3, "target", targets["diagnostic"])
            diagnostic_daily["oriented_ic"] = spec.expected_sign * diagnostic_daily["ic"]
            diagnostic_daily_frames.append(diagnostic_daily)
            diagnostic_records.append(
                {
                    "logical_id": spec.logical_id,
                    "feature_name": spec.feature_name,
                    "entry_session": session,
                    "target": targets["diagnostic"],
                    **_summarize_diagnostic(diagnostic_daily),
                    "selection_authority": False,
                }
            )

            edges = frozen_quintile_edges(
                session_frame[spec.feature_name].to_numpy(dtype=np.float64)
            )
            for index, edge in enumerate(edges, start=1):
                edge_records.append(
                    {
                        "logical_id": spec.logical_id,
                        "feature_name": spec.feature_name,
                        "entry_session": session,
                        "edge_number": index,
                        "edge_value": float(edge),
                        "fitted_finite_observations": int(
                            _finite(session_frame[spec.feature_name]).sum()
                        ),
                        "partition": "Development",
                    }
                )
            quintile_frame = primary_frame.copy()
            quintile_frame["_quintile"] = assign_frozen_quintiles(
                quintile_frame[spec.feature_name].to_numpy(dtype=np.float64), edges
            )
            target_finite = _finite(quintile_frame[targets["primary"]])
            tick_finite = _finite(quintile_frame[targets["tick"]])
            target_means: list[float] = []
            tick_means: list[float] = []
            for quintile in range(1, 6):
                in_bin = quintile_frame["_quintile"].to_numpy() == quintile
                target_values = quintile_frame.loc[
                    in_bin & target_finite, targets["primary"]
                ].to_numpy(dtype=np.float64)
                tick_values = quintile_frame.loc[in_bin & tick_finite, targets["tick"]].to_numpy(
                    dtype=np.float64
                )
                target_mean = float(target_values.mean()) if len(target_values) else float("nan")
                tick_mean = float(tick_values.mean()) if len(tick_values) else float("nan")
                target_means.append(target_mean)
                tick_means.append(tick_mean)
                quintile_records.append(
                    {
                        "logical_id": spec.logical_id,
                        "feature_name": spec.feature_name,
                        "entry_session": session,
                        "quintile": quintile,
                        "target": targets["primary"],
                        "target_observations": len(target_values),
                        "target_mean_atr": target_mean,
                        "tick_observations": len(tick_values),
                        "target_mean_ticks": tick_mean,
                    }
                )
            quintile_available = bool(
                np.isfinite(target_means).all() and np.isfinite(tick_means).all()
            )
            monotonicity = (
                _spearman(
                    np.arange(1, 6, dtype=np.float64),
                    spec.expected_sign * np.asarray(target_means),
                )
                if quintile_available
                else float("nan")
            )
            target_spread = (
                float(spec.expected_sign * (target_means[4] - target_means[0]))
                if quintile_available
                else float("nan")
            )
            tick_spread = (
                float(spec.expected_sign * (tick_means[4] - tick_means[0]))
                if quintile_available
                else float("nan")
            )

            anchors = (
                D1_INPUTS_BY_SESSION[session]
                if spec.role == "directional"
                else O1_INPUTS_BY_SESSION[session]
            )
            controls = tuple(dict.fromkeys((*anchors, *spec.partial_information_controls)))
            partial_daily = _partial_daily_ic(
                primary_frame,
                feature=spec.feature_name,
                target=targets["primary"],
                controls=controls,
                clock_bins=CLOCK_BINS_BY_SESSION[session],
                reference_clock_bin=CLOCK_REFERENCE_BIN_BY_SESSION[session],
            )
            partial_daily.insert(0, "logical_id", spec.logical_id)
            partial_daily.insert(1, "feature_name", spec.feature_name)
            partial_daily.insert(2, "entry_session", session)
            partial_daily.insert(3, "target", targets["primary"])
            partial_daily["oriented_partial_ic"] = spec.expected_sign * partial_daily["partial_ic"]
            partial_daily["oriented_matched_unadjusted_ic"] = (
                spec.expected_sign * (partial_daily["matched_unadjusted_ic"])
            )
            partial_daily_frames.append(partial_daily)
            partial_available = partial_daily.loc[
                np.isfinite(partial_daily["partial_ic"])
                & np.isfinite(partial_daily["matched_unadjusted_ic"])
            ].copy()
            partial_inference = stationary_bootstrap_mean_inference(
                partial_available["oriented_partial_ic"].to_numpy(dtype=np.float64)
            )
            partial_mean = (
                float(partial_available["partial_ic"].mean())
                if len(partial_available)
                else float("nan")
            )
            matched_mean = (
                float(partial_available["matched_unadjusted_ic"].mean())
                if len(partial_available)
                else float("nan")
            )
            retention = (
                abs(partial_mean) / abs(matched_mean)
                if np.isfinite(matched_mean) and matched_mean != 0.0
                else float("nan")
            )
            partial_records.append(
                {
                    "logical_id": spec.logical_id,
                    "feature_name": spec.feature_name,
                    "entry_session": session,
                    "target": targets["primary"],
                    "controls": "|".join(controls),
                    "valid_dates": len(partial_available),
                    "valid_observations": int(partial_available["observations"].sum()),
                    "mean_partial_ic": partial_mean,
                    "mean_oriented_partial_ic": (
                        float(partial_available["oriented_partial_ic"].mean())
                        if len(partial_available)
                        else float("nan")
                    ),
                    "mean_matched_unadjusted_ic": matched_mean,
                    "absolute_retention": retention,
                    "bootstrap_lower_oriented": partial_inference["bootstrap_lower"],
                    "bootstrap_upper_oriented": partial_inference["bootstrap_upper"],
                    "null_p_value": partial_inference["null_p_value"],
                    "unavailable_date_count": int(
                        (~np.isfinite(partial_daily["partial_ic"])).sum()
                    ),
                }
            )

            for record in _related_redundancy(
                session_frame,
                feature=spec.feature_name,
                controls=spec.partial_information_controls,
            ):
                redundancy_records.append(
                    {
                        "logical_id": spec.logical_id,
                        "feature_name": spec.feature_name,
                        "entry_session": session,
                        **record,
                    }
                )

            finite_feature = int(_finite(primary_frame[spec.feature_name]).sum())
            primary_target_finite = int(_finite(primary_frame[targets["primary"]]).sum())
            common_primary = int(
                (
                    _finite(primary_frame[spec.feature_name])
                    & _finite(primary_frame[targets["primary"]])
                ).sum()
            )
            ledger_records.append(
                {
                    "logical_id": spec.logical_id,
                    "feature_name": spec.feature_name,
                    "entry_session": session,
                    "role": spec.role,
                    "expected_orientation": spec.expected_orientation,
                    "expected_sign": spec.expected_sign,
                    "primary_target": targets["primary"],
                    "tick_target": targets["tick"],
                    "eligible_label_rows": len(primary_frame),
                    "finite_feature_rows": finite_feature,
                    "finite_primary_target_rows": primary_target_finite,
                    "common_primary_rows": common_primary,
                    "feature_missing_rows": len(primary_frame) - finite_feature,
                    "feature_missing_rate": (
                        (len(primary_frame) - finite_feature) / len(primary_frame)
                        if len(primary_frame)
                        else float("nan")
                    ),
                    "valid_dates": len(primary_available),
                    "valid_observations": int(primary_available["observations"].sum()),
                    "mean_ic": (
                        float(primary_available["ic"].mean())
                        if len(primary_available)
                        else float("nan")
                    ),
                    "mean_oriented_ic": (
                        float(primary_available["oriented_ic"].mean())
                        if len(primary_available)
                        else float("nan")
                    ),
                    "median_ic": (
                        float(primary_available["ic"].median())
                        if len(primary_available)
                        else float("nan")
                    ),
                    "standard_deviation_ic": (
                        float(primary_available["ic"].std(ddof=1))
                        if len(primary_available) > 1
                        else float("nan")
                    ),
                    "oriented_sign_fraction": (
                        float((primary_available["oriented_ic"] > 0.0).mean())
                        if len(primary_available)
                        else float("nan")
                    ),
                    "bootstrap_lower_oriented": inference["bootstrap_lower"],
                    "bootstrap_upper_oriented": inference["bootstrap_upper"],
                    "null_p_value": inference["null_p_value"],
                    "bh_q_value": float("nan"),
                    "quintile_monotonicity_oriented": monotonicity,
                    "absolute_quintile_monotonicity": abs(monotonicity),
                    "oriented_top_bottom_atr": target_spread,
                    "oriented_top_bottom_ticks": tick_spread,
                    "partial_valid_dates": len(partial_available),
                    "partial_mean_ic": partial_mean,
                    "partial_mean_matched_unadjusted_ic": matched_mean,
                    "partial_absolute_retention": retention,
                    "partial_bootstrap_lower_oriented": partial_inference["bootstrap_lower"],
                    "partial_bootstrap_upper_oriented": partial_inference["bootstrap_upper"],
                    "maximum_year_positive_evidence_share": year_share,
                    "ten_best_dates_positive_evidence_share": ten_share,
                }
            )

    ledger = pd.DataFrame.from_records(ledger_records)
    if len(ledger) != 18:
        raise RuntimeError(f"Frozen confirmatory family drifted: expected 18, got {len(ledger)}")
    ledger["bh_q_value"] = benjamini_hochberg(ledger["null_p_value"].to_numpy())
    tick_threshold = np.where(ledger["role"].eq("directional"), 2.0, 1.0)
    ledger["gate_support"] = (ledger["valid_dates"] >= MIN_VALID_DATES) & (
        ledger["valid_observations"] >= MIN_VALID_OBSERVATIONS
    )
    ledger["gate_expected_orientation"] = ledger["mean_oriented_ic"] > 0.0
    ledger["gate_bh_q"] = ledger["bh_q_value"] <= BH_Q_GATE
    ledger["gate_primary_interval"] = ledger["bootstrap_lower_oriented"] > 0.0
    ledger["gate_quintile_monotonicity"] = (
        ledger["absolute_quintile_monotonicity"] >= MONOTONICITY_GATE
    )
    ledger["gate_tick_spread"] = ledger["oriented_top_bottom_ticks"] >= tick_threshold
    ledger["gate_partial_retention"] = (
        ledger["partial_absolute_retention"] >= PARTIAL_RETENTION_GATE
    )
    ledger["gate_partial_interval"] = ledger["partial_bootstrap_lower_oriented"] > 0.0
    ledger["gate_year_concentration"] = (
        ledger["maximum_year_positive_evidence_share"] <= CONCENTRATION_GATE
    )
    ledger["gate_ten_date_concentration"] = (
        ledger["ten_best_dates_positive_evidence_share"] <= CONCENTRATION_GATE
    )
    gate_columns = [column for column in ledger.columns if column.startswith("gate_")]
    reasons: list[str] = []
    statuses: list[str] = []
    for _, row in ledger.iterrows():
        failed = [column.removeprefix("gate_") for column in gate_columns if not bool(row[column])]
        reasons.append("|".join(failed))
        numeric_required = (
            "mean_oriented_ic",
            "bootstrap_lower_oriented",
            "bh_q_value",
            "absolute_quintile_monotonicity",
            "oriented_top_bottom_ticks",
            "partial_absolute_retention",
            "partial_bootstrap_lower_oriented",
            "maximum_year_positive_evidence_share",
            "ten_best_dates_positive_evidence_share",
        )
        mechanically_evaluable = all(np.isfinite(float(row[column])) for column in numeric_required)
        if not bool(row["gate_support"]) or not mechanically_evaluable:
            statuses.append("NOT_EVALUABLE")
        elif failed:
            statuses.append("DEV_REJECTED")
        else:
            statuses.append("DEV_SHORTLISTED")
    ledger["failed_gate_reasons"] = reasons
    ledger["status"] = statuses

    candidate_records: list[dict[str, object]] = []
    for session in SESSIONS:
        session_frame = working.loc[working["entry_session"] == session]
        for left_spec in TSAY_FEATURE_SPECS:
            for right_spec in TSAY_FEATURE_SPECS:
                common = _finite(session_frame[left_spec.feature_name]) & _finite(
                    session_frame[right_spec.feature_name]
                )
                candidate_records.append(
                    {
                        "entry_session": session,
                        "left_logical_id": left_spec.logical_id,
                        "right_logical_id": right_spec.logical_id,
                        "common_observations": int(common.sum()),
                        "pooled_spearman": _spearman(
                            session_frame.loc[common, left_spec.feature_name].to_numpy(
                                dtype=np.float64
                            ),
                            session_frame.loc[common, right_spec.feature_name].to_numpy(
                                dtype=np.float64
                            ),
                        ),
                    }
                )

    return TsayEvidenceResult(
        evidence_ledger=ledger,
        primary_daily_ic=pd.concat(primary_daily_frames, ignore_index=True),
        diagnostic_daily_ic=pd.concat(diagnostic_daily_frames, ignore_index=True),
        diagnostic_summary=pd.DataFrame.from_records(diagnostic_records),
        year_stability=pd.DataFrame.from_records(year_records),
        quintile_edges=pd.DataFrame.from_records(edge_records),
        quintile_summary=pd.DataFrame.from_records(quintile_records),
        partial_daily_ic=pd.concat(partial_daily_frames, ignore_index=True),
        partial_summary=pd.DataFrame.from_records(partial_records),
        related_feature_redundancy=pd.DataFrame.from_records(redundancy_records),
        candidate_correlation=pd.DataFrame.from_records(candidate_records),
    )
