"""Section 12 hybrid integration research: POI events x frozen statistical gate.

Both branches are complete and frozen, so the integration boundary may now be
crossed in one direction only: Branch A's True POI retests supply direction
candidates, and Branch B's frozen Section 9/10 opportunity model supplies the
filter and ranking. The declared question follows the research contract:

    Does the frozen statistical opportunity gate add stable out-of-sample
    value beyond the POI baseline alone?

Design rules, frozen in :class:`Section12Config` before computation:

- The statistical component is used exactly as frozen: the saved Section 10
  gate predictions and thresholds are loaded, never refitted.  Joining is by
  decision-bar identity (the POI retest bar equals the Branch B decision bar,
  whose saved entry timestamp is one minute later).
- The outcome metric matches Branch A's own headline convention: the capped
  60-minute R label, robust-capped at +/-5R.
- Advancement verdicts use Development and Validation only.  The Final test
  is reported once, separately, for the frozen comparison - mirroring the
  treatment Branch A's candidate policies already received - and never feeds
  a criterion.
- Ranking value is measured with Development-fitted prediction quintiles
  applied unchanged to later partitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SECTION12_RANDOM_SEED = 20260726

EVALUATION_PARTITIONS = ("Development", "Validation")
ALL_PARTITIONS = ("Development", "Validation", "Final test")
PARTITION_NORMALIZATION = {
    "development": "Development",
    "validation": "Validation",
    "final_test": "Final test",
    "final test": "Final test",
}
FAMILY_KEYS = ("hypothesis", "trade_side")


@dataclass(frozen=True)
class Section12Config:
    """Frozen Section 12 integration contract."""

    random_seed: int = SECTION12_RANDOM_SEED
    outcome_label: str = "label_capped_60m_r"
    outcome_valid_flag: str = "label_capped_60m_valid"
    robust_cap_r: float = 5.0
    quintile_bucket_count: int = 5
    bootstrap_replicates: int = 2_000
    bootstrap_confidence: float = 0.95
    min_daily_events_for_diff: int = 3
    min_development_gated_events: int = 500
    min_validation_gated_events: int = 300
    min_development_trading_dates: int = 150
    min_validation_trading_dates: int = 100
    min_development_improvement_r: float = 0.05
    min_validation_retention: float = 0.25


@dataclass
class HybridIntegrationResult:
    """Container for every Section 12 output object."""

    event_frame_summary: pd.Series
    family_results: pd.DataFrame
    quintile_results: pd.DataFrame
    verdicts: pd.DataFrame
    final_test_report: pd.DataFrame
    validation_checks: pd.DataFrame
    summary: pd.Series
    config: Section12Config = field(default_factory=Section12Config)


def load_integration_inputs(project_root: Path, config: Section12Config | None = None):
    """Column-scoped loads of the three frozen artifacts."""

    cfg = config or Section12Config()
    processed = project_root / "data" / "processed"
    labels = pq.read_table(
        processed / "section7_true_poi_outcome_labels_gc.parquet",
        columns=[
            "true_retest_id",
            "trade_date_ny",
            "research_partition",
            "hypothesis",
            "trade_side",
            "retest_bar_id",
            cfg.outcome_label,
            cfg.outcome_valid_flag,
        ],
    ).to_pandas(ignore_metadata=True)
    context = pq.read_table(
        processed / "section7_true_poi_context_frame_gc.parquet",
        columns=["true_retest_id", "retest_ts_event_utc", "feat_execution_session"],
    ).to_pandas(ignore_metadata=True)
    candidates = pq.read_table(
        processed / "statistical_research" / "signal_candidates_gc.parquet",
        columns=[
            "observation_id",
            "entry_timestamp_utc",
            "entry_session",
            "gate_prediction",
            "expansion_gate_flag",
        ],
    ).to_pandas(ignore_metadata=True)
    return labels, context, candidates


def build_hybrid_event_frame(
    labels: pd.DataFrame,
    context: pd.DataFrame,
    candidates: pd.DataFrame,
    config: Section12Config | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Join POI outcome rows to the frozen statistical gate at decision bars."""

    cfg = config or Section12Config()
    frame = labels.merge(context, on="true_retest_id", how="left", validate="many_to_one")
    if frame["retest_ts_event_utc"].isna().any():
        raise ValueError("every outcome row must have a context retest timestamp")

    partition = frame["research_partition"].astype(str).str.lower().map(PARTITION_NORMALIZATION)
    if partition.isna().any():
        unknown = sorted(frame.loc[partition.isna(), "research_partition"].astype(str).unique())
        raise ValueError(f"unknown research partitions: {unknown[:4]}")
    frame["research_partition"] = partition

    gate = candidates.copy()
    if "decision_timestamp_utc" not in gate.columns:
        gate["decision_timestamp_utc"] = pd.to_datetime(gate["entry_timestamp_utc"]) - pd.Timedelta(
            minutes=1
        )
    gate = gate.drop_duplicates("decision_timestamp_utc")
    frame["retest_ts_event_utc"] = pd.to_datetime(frame["retest_ts_event_utc"])
    frame = frame.merge(
        gate[["decision_timestamp_utc", "gate_prediction", "expansion_gate_flag"]],
        left_on="retest_ts_event_utc",
        right_on="decision_timestamp_utc",
        how="left",
        validate="many_to_one",
    )
    matched = frame["gate_prediction"].notna()

    valid = frame[cfg.outcome_valid_flag].fillna(False).astype(bool)
    outcome = pd.to_numeric(frame[cfg.outcome_label], errors="coerce")
    keep = matched & valid & outcome.notna()

    events = pd.DataFrame(
        {
            "true_retest_id": frame.loc[keep, "true_retest_id"].to_numpy(),
            "trade_date_ny": pd.to_datetime(frame.loc[keep, "trade_date_ny"])
            .dt.normalize()
            .to_numpy(),
            "research_partition": frame.loc[keep, "research_partition"].to_numpy(),
            "hypothesis": frame.loc[keep, "hypothesis"].astype(str).to_numpy(),
            "trade_side": frame.loc[keep, "trade_side"].astype(str).to_numpy(),
            "session": frame.loc[keep, "feat_execution_session"].astype(str).to_numpy(),
            "robust_r": outcome.loc[keep].clip(-cfg.robust_cap_r, cfg.robust_cap_r).to_numpy(),
            "gate_prediction": frame.loc[keep, "gate_prediction"].to_numpy(dtype=np.float64),
            "expansion_gate_flag": frame.loc[keep, "expansion_gate_flag"].astype(bool).to_numpy(),
        }
    )
    summary = pd.Series(
        {
            "outcome_rows": len(frame),
            "gate_matched_rows": int(matched.sum()),
            "gate_match_rate": float(matched.mean()),
            "valid_outcome_rows_kept": len(events),
            "development_events": int((events["research_partition"] == "Development").sum()),
            "validation_events": int((events["research_partition"] == "Validation").sum()),
            "final_test_events": int((events["research_partition"] == "Final test").sum()),
            "gate_pass_rate_development": float(
                events.loc[
                    events["research_partition"] == "Development", "expansion_gate_flag"
                ].mean()
            ),
        },
        name="value",
    )
    return events, summary


def _bootstrap_mean_ci(daily_values, *, seed, replicates, confidence):
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
    return tuple(float(v) for v in np.quantile(estimates, [alpha, 1.0 - alpha]))


def _date_block_improvement_ci(part, *, seed, replicates, confidence):
    """Date-block bootstrap interval for the pooled gated-minus-baseline mean.

    Trading dates are resampled with replacement; each replicate pools the
    events of the drawn dates and recomputes the improvement.  Unlike a
    within-day paired mean, this uses every event even when a day has few
    gated observations, which matters at low gate-pass rates.
    """

    date_codes, date_index = pd.factorize(part["trade_date_ny"], sort=True)
    n_dates = len(date_index)
    if n_dates < 2:
        return (np.nan, np.nan)
    values = part["robust_r"].to_numpy(dtype=np.float64)
    gated = part["expansion_gate_flag"].to_numpy(dtype=bool)
    sum_all = np.bincount(date_codes, weights=values, minlength=n_dates)
    count_all = np.bincount(date_codes, minlength=n_dates).astype(np.float64)
    sum_gated = np.bincount(date_codes, weights=np.where(gated, values, 0.0), minlength=n_dates)
    count_gated = np.bincount(date_codes, weights=gated.astype(np.float64), minlength=n_dates)
    rng = np.random.default_rng(seed)
    estimates = np.full(replicates, np.nan)
    batch = 200
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        draw = rng.integers(0, n_dates, size=(stop - start, n_dates))
        total_gated = count_gated[draw].sum(axis=1)
        total_all = count_all[draw].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            estimates[start:stop] = (
                sum_gated[draw].sum(axis=1) / total_gated - sum_all[draw].sum(axis=1) / total_all
            )
    alpha = (1.0 - confidence) / 2.0
    finite = estimates[np.isfinite(estimates)]
    if len(finite) < replicates // 2:
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.quantile(finite, [alpha, 1.0 - alpha]))


def build_full_gate_table(
    project_root: Path, config: Section12Config | None = None
) -> pd.DataFrame:
    """Apply the frozen Section 9/10 gate to every partition, Final test included.

    The saved Section 10 candidates cover Development and Validation only, by
    construction.  For the one-time Final-test read the frozen per-session
    ridge coefficients and gate thresholds are applied unchanged to the saved
    feature matrix; per-date cross-sectional rank z-scores involve no fitting.
    Development/Validation rows are verified to reproduce the saved candidate
    predictions exactly before the table is returned.
    """

    processed = project_root / "data" / "processed" / "statistical_research"
    coefficients = pd.read_parquet(
        processed / "multivariate_coefficients_gc.parquet", engine="pyarrow"
    )
    coefficients = coefficients.loc[coefficients["horizon_minutes"].eq(60)]
    thresholds = pd.read_parquet(
        processed / "signal_gate_thresholds_gc.parquet", engine="pyarrow"
    ).set_index("session")["gate_threshold"]
    needed = sorted(coefficients["feature_name"].unique())
    matrix = pq.read_table(
        processed / "feature_matrix_gc.parquet",
        columns=[
            "observation_id",
            "decision_timestamp_utc",
            "trade_date_ny",
            "entry_session",
            *needed,
        ],
    ).to_pandas(ignore_metadata=True)

    # Rank z-scores depend on each day's cross-section, so the population must
    # match the Section 7/10 complete-label rule exactly - now applied to every
    # partition: all six horizons available with valid, finite ATR outcomes.
    from src.statistical_research.feature_evaluation import (
        OUTCOME_FAMILY_ATR_TEMPLATES,
        load_forward_labels_for_evaluation,
    )
    from src.statistical_research.labels import FORWARD_HORIZONS_MINUTES

    availability = load_forward_labels_for_evaluation(
        processed / "forward_labels_gc.parquet"
    ).set_index("observation_id")
    complete = availability["atr_normalization_available"].to_numpy(dtype=bool).copy()
    for horizon in FORWARD_HORIZONS_MINUTES:
        complete &= availability[f"label_available_{horizon}"].to_numpy(dtype=bool)
        for template in OUTCOME_FAMILY_ATR_TEMPLATES.values():
            complete &= np.isfinite(
                availability[template.format(h=horizon)].to_numpy(dtype=np.float64)
            )
    complete_ids = set(availability.index[complete])
    matrix = matrix.loc[matrix["observation_id"].isin(complete_ids)].reset_index(drop=True)

    prediction = np.full(len(matrix), np.nan)
    session_values = matrix["entry_session"].astype(str).to_numpy()
    for session, session_beta in coefficients.groupby("session", observed=True):
        features = session_beta["feature_name"].tolist()
        beta = session_beta["coefficient"].to_numpy(dtype=np.float64)
        mask = session_values == str(session)
        sub = matrix.loc[mask, [*features, "trade_date_ny"]]
        date_codes, _ = pd.factorize(sub["trade_date_ny"], sort=True)
        grouped = sub[features].astype("float64").groupby(date_codes, sort=False)
        ranks = grouped.rank(method="average")
        mean = ranks.groupby(date_codes, sort=False).transform("mean")
        std = ranks.groupby(date_codes, sort=False).transform("std")
        with np.errstate(invalid="ignore", divide="ignore"):
            z = ((ranks - mean) / std).replace([np.inf, -np.inf], np.nan)
        prediction[mask] = z.to_numpy(dtype=np.float64) @ beta

    gate = pd.DataFrame(
        {
            "decision_timestamp_utc": pd.to_datetime(matrix["decision_timestamp_utc"]),
            "gate_prediction": prediction,
            "expansion_gate_flag": prediction
            >= matrix["entry_session"].astype(str).map(thresholds).to_numpy(dtype=np.float64),
        }
    ).dropna(subset=["gate_prediction"])

    saved = pd.read_parquet(
        processed / "signal_candidates_gc.parquet",
        engine="pyarrow",
        columns=["entry_timestamp_utc", "gate_prediction", "expansion_gate_flag"],
    )
    saved["decision_timestamp_utc"] = pd.to_datetime(saved["entry_timestamp_utc"]) - pd.Timedelta(
        minutes=1
    )
    check = saved.merge(gate, on="decision_timestamp_utc", suffixes=("_saved", "_full"))
    if len(check) < 0.99 * len(saved):
        raise ValueError("full gate table does not cover the saved candidates")
    if (
        not np.allclose(check["gate_prediction_saved"], check["gate_prediction_full"], atol=1e-9)
        or not (check["expansion_gate_flag_saved"] == check["expansion_gate_flag_full"]).all()
    ):
        raise ValueError("full gate table does not reproduce the frozen saved predictions")
    return gate


def _group_stats(group: pd.DataFrame) -> dict:
    values = group["robust_r"].to_numpy(dtype=np.float64)
    return {
        "events": len(values),
        "trading_dates": int(group["trade_date_ny"].nunique()),
        "mean_robust_r": float(values.mean()) if len(values) else np.nan,
        "median_robust_r": float(np.median(values)) if len(values) else np.nan,
        "positive_rate": float((values > 0).mean()) if len(values) else np.nan,
    }


def build_hybrid_analysis(
    events: pd.DataFrame,
    config: Section12Config | None = None,
) -> HybridIntegrationResult:
    """Baseline-versus-gated comparison, ranking quintiles, and verdicts."""

    cfg = config or Section12Config()
    observed = set(events["research_partition"].unique())
    if not observed.issubset(set(ALL_PARTITIONS)):
        raise ValueError(f"unexpected partitions: {observed}")

    evaluation = events.loc[events["research_partition"].isin(EVALUATION_PARTITIONS)]
    final_test = events.loc[events["research_partition"] == "Final test"]

    family_records: list[dict] = []
    quintile_records: list[dict] = []
    verdict_records: list[dict] = []

    for family_index, ((hypothesis, side), family_all) in enumerate(
        sorted(evaluation.groupby(list(FAMILY_KEYS), sort=False), key=lambda kv: kv[0])
    ):
        development = family_all.loc[family_all["research_partition"] == "Development"]
        edges = np.unique(
            np.nanquantile(
                development["gate_prediction"].to_numpy(dtype=np.float64),
                np.linspace(0.0, 1.0, cfg.quintile_bucket_count + 1)[1:-1],
            )
        )
        improvements: dict[str, float] = {}
        ci_lows: dict[str, float] = {}
        floors_ok: dict[str, bool] = {}
        for partition_index, partition in enumerate(EVALUATION_PARTITIONS):
            part = family_all.loc[family_all["research_partition"] == partition]
            baseline = _group_stats(part)
            gated = _group_stats(part.loc[part["expansion_gate_flag"]])
            improvement = float(gated["mean_robust_r"] - baseline["mean_robust_r"])
            seed = cfg.random_seed + 10_000 * family_index + 100 * partition_index
            ci_low, ci_high = _date_block_improvement_ci(
                part,
                seed=seed,
                replicates=cfg.bootstrap_replicates,
                confidence=cfg.bootstrap_confidence,
            )
            improvements[partition] = improvement
            ci_lows[partition] = ci_low
            floors_ok[partition] = (
                gated["events"]
                >= (
                    cfg.min_development_gated_events
                    if partition == "Development"
                    else cfg.min_validation_gated_events
                )
            ) and (
                gated["trading_dates"]
                >= (
                    cfg.min_development_trading_dates
                    if partition == "Development"
                    else cfg.min_validation_trading_dates
                )
            )
            family_records.append(
                {
                    "hypothesis": hypothesis,
                    "trade_side": side,
                    "research_partition": partition,
                    "baseline_events": baseline["events"],
                    "baseline_mean_robust_r": baseline["mean_robust_r"],
                    "baseline_median_robust_r": baseline["median_robust_r"],
                    "gated_events": gated["events"],
                    "gated_trading_dates": gated["trading_dates"],
                    "gated_mean_robust_r": gated["mean_robust_r"],
                    "gated_median_robust_r": gated["median_robust_r"],
                    "gated_positive_rate": gated["positive_rate"],
                    "improvement_mean_robust_r": improvement,
                    "improvement_daily_ci_low": ci_low,
                    "improvement_daily_ci_high": ci_high,
                }
            )
            buckets = np.searchsorted(edges, part["gate_prediction"].to_numpy(dtype=np.float64))
            for bucket in range(len(edges) + 1):
                rows = part.loc[buckets == bucket]
                if rows.empty:
                    continue
                stats = _group_stats(rows)
                quintile_records.append(
                    {
                        "hypothesis": hypothesis,
                        "trade_side": side,
                        "research_partition": partition,
                        "prediction_quintile": bucket,
                        **stats,
                    }
                )

        dev_improvement = improvements["Development"]
        val_improvement = improvements["Validation"]
        with np.errstate(invalid="ignore", divide="ignore"):
            retention = val_improvement / dev_improvement if dev_improvement else np.nan
        advances = bool(
            np.isfinite(dev_improvement)
            and dev_improvement >= cfg.min_development_improvement_r
            and np.isfinite(ci_lows["Development"])
            and ci_lows["Development"] > 0.0
            and np.isfinite(val_improvement)
            and np.sign(val_improvement) == np.sign(dev_improvement)
            and np.isfinite(retention)
            and retention >= cfg.min_validation_retention
            and floors_ok["Development"]
            and floors_ok["Validation"]
        )
        verdict_records.append(
            {
                "hypothesis": hypothesis,
                "trade_side": side,
                "development_improvement_r": dev_improvement,
                "development_ci_low": ci_lows["Development"],
                "validation_improvement_r": val_improvement,
                "validation_retention": float(retention) if np.isfinite(retention) else np.nan,
                "verdict": "HYBRID_ADVANCES" if advances else "NO_INCREMENTAL_VALUE",
            }
        )

    family_results = pd.DataFrame.from_records(family_records)
    quintile_results = pd.DataFrame.from_records(quintile_records)
    verdicts = pd.DataFrame.from_records(verdict_records)

    # One-time frozen Final-test read, reported for every family after the
    # verdicts above were fixed; it feeds no criterion.
    final_records = []
    for (hypothesis, side), part in final_test.groupby(list(FAMILY_KEYS), sort=False):
        baseline = _group_stats(part)
        gated = _group_stats(part.loc[part["expansion_gate_flag"]])
        final_records.append(
            {
                "hypothesis": hypothesis,
                "trade_side": side,
                "baseline_events": baseline["events"],
                "baseline_mean_robust_r": baseline["mean_robust_r"],
                "gated_events": gated["events"],
                "gated_mean_robust_r": gated["mean_robust_r"],
                "improvement_mean_robust_r": float(
                    gated["mean_robust_r"] - baseline["mean_robust_r"]
                ),
            }
        )
    final_test_report = pd.DataFrame.from_records(final_records)

    expected_families = evaluation.groupby(list(FAMILY_KEYS), sort=False).ngroups
    checks = {
        "verdicts_use_development_and_validation_only": True,
        "verdict_per_family": len(verdicts) == expected_families,
        "family_rows_cover_both_evaluation_partitions": set(
            family_results["research_partition"].unique()
        )
        == set(EVALUATION_PARTITIONS),
        "final_test_report_separate_from_verdicts": "verdict" not in final_test_report.columns,
        "robust_r_within_cap": bool(events["robust_r"].abs().le(cfg.robust_cap_r + 1e-9).all()),
        "quintile_buckets_within_config": quintile_results["prediction_quintile"].max()
        < cfg.quintile_bucket_count,
        "gate_flags_boolean": events["expansion_gate_flag"].dtype == bool,
    }
    validation_checks = pd.Series(checks, name="passed").rename_axis("check").reset_index()

    event_frame_summary = pd.Series(
        {
            "evaluation_events": len(evaluation),
            "final_test_events": len(final_test),
            "families": expected_families,
            "hybrid_advances": int(verdicts["verdict"].eq("HYBRID_ADVANCES").sum()),
            "no_incremental_value": int(verdicts["verdict"].eq("NO_INCREMENTAL_VALUE").sum()),
        },
        name="value",
    )
    summary = pd.Series(
        {
            **event_frame_summary.to_dict(),
            "best_family": (
                verdicts.sort_values("validation_improvement_r", ascending=False)
                .iloc[0][["hypothesis", "trade_side"]]
                .str.cat(sep=" ")
                if len(verdicts)
                else ""
            ),
        },
        name="value",
    )

    return HybridIntegrationResult(
        event_frame_summary=event_frame_summary,
        family_results=family_results,
        quintile_results=quintile_results,
        verdicts=verdicts,
        final_test_report=final_test_report,
        validation_checks=validation_checks,
        summary=summary,
        config=cfg,
    )


def save_hybrid_outputs(result: HybridIntegrationResult, *, project_root: Path) -> pd.DataFrame:
    """Persist Section 12 outputs; generated artifacts stay excluded from Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "section12"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "hybrid_family_results": (
            processed / "hybrid_family_results_gc.parquet",
            result.family_results,
        ),
        "hybrid_quintile_results": (
            processed / "hybrid_quintile_results_gc.parquet",
            result.quintile_results,
        ),
        "hybrid_verdicts": (processed / "hybrid_verdicts_gc.parquet", result.verdicts),
        "hybrid_final_test_report": (
            processed / "hybrid_final_test_report_gc.parquet",
            result.final_test_report,
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
    result.verdicts.to_csv(tables / "section12_hybrid_verdicts_gc.csv", index=False)
    result.family_results.to_csv(tables / "section12_hybrid_family_results_gc.csv", index=False)
    return pd.DataFrame.from_records(records)
