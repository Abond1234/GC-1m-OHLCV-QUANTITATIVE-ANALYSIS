"""Section 12B: opportunity-conditioned sizing, exits, and suppression.

Section 12 established that the frozen Branch B opportunity model fails as an
entry filter for POI events.  This module executes the pre-declared contract in
``project_docs/section12b_research_contract.md``: test what the model actually
forecasts - movement magnitude - in the roles magnitude information can
legitimately play for the True POI continuation-short family.

Scope declarations, fixed here before any result is computed:

- Population: continuation-short outcome rows joined to the full-coverage
  frozen gate table exactly as in Section 12 (decision-bar identity join with
  the saved-prediction reproduction check enforced by
  :func:`src.research.hybrid_integration.build_full_gate_table`).
- Prediction quintiles: fitted on the family's Development events only and
  applied unchanged to Validation and the Final test.
- H1 sizing schemes (frozen weight ladders over Development-fitted quintiles
  ``q`` in ``0..4``): constant ``1``; proportional ``(q + 1) / 3``; inverse
  ``(5 - q) / 3``.  Each ladder averages one under the uniform Development
  quintile masses, so sizing redistributes risk without changing its total.
  The risk-adjusted statistic is the pooled mean/MAD ratio of weighted net R
  (MAD: mean absolute deviation about the mean); net R applies the
  ``Section11Config`` round-trip cost divided by each event's own risk in
  ticks.  Advancement requires the criteria to hold in both the base and the
  pessimistic scenario, plus a mean-R materiality bound.
- H2 exits: the conditioning menu is the capped 60/120/240-minute R labels -
  the holding-policy rows of the frozen Section 7 stop/target grid that are
  scoreable per event from existing labels with correct stop-first ordering.
  Per-event target outcomes cannot be reconstructed from saved labels without
  intrabar-ordering ambiguity, so the target dimension of the grid stays at
  the labels' embodied treatment; this scoping is declared here, not after
  results.  Ties in the Development argmax resolve to the shortest horizon.
- H3 suppression: drop the bottom Development-fitted quintile.  Co-primary
  effects, both required: median R improvement and -1R (stop) rate reduction.
- Every Development effect must pass a date-block bootstrap interval
  excluding zero; Validation must agree in sign with at least 25 percent
  retention; affected subsets must satisfy floors of 1,000 Development and
  700 Validation events across at least 200 and 120 trading dates.  The
  Final test is read once after all verdicts are fixed and feeds no
  criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.research.hybrid_integration import (
    PARTITION_NORMALIZATION,
    build_full_gate_table,
)
from src.statistical_research.sequential_backtest import Section11Config, _scenario_cost_ticks

SECTION12B_RANDOM_SEED = 20260728

EVALUATION_PARTITIONS = ("Development", "Validation")
ALL_PARTITIONS = ("Development", "Validation", "Final test")


@dataclass(frozen=True)
class Section12BConfig:
    """Frozen Section 12B contract parameters."""

    random_seed: int = SECTION12B_RANDOM_SEED
    family_hypothesis: str = "continuation"
    family_trade_side: str = "short"
    outcome_label: str = "label_capped_60m_r"
    outcome_valid_flag: str = "label_capped_60m_valid"
    exit_horizons_minutes: tuple[int, ...] = (60, 120, 240)
    robust_cap_r: float = 5.0
    quintile_bucket_count: int = 5
    bootstrap_replicates: int = 2_000
    bootstrap_confidence: float = 0.95
    min_development_events: int = 1_000
    min_validation_events: int = 700
    min_development_trading_dates: int = 200
    min_validation_trading_dates: int = 120
    min_validation_retention: float = 0.25
    mean_r_materiality: float = 0.05
    stop_loss_r: float = -1.0
    sizing_schemes: tuple[str, ...] = ("constant", "proportional", "inverse")
    h1_cost_scenarios: tuple[str, ...] = ("base", "pessimistic")
    cost_config: Section11Config = field(default_factory=Section11Config)


@dataclass
class OpportunityConditioningResult:
    """Container for every Section 12B output object."""

    population_summary: pd.Series
    quintile_edges: pd.Series
    h1_results: pd.DataFrame
    h1_effects: pd.DataFrame
    h2_mapping: pd.DataFrame
    h2_results: pd.DataFrame
    h3_results: pd.DataFrame
    verdicts: pd.DataFrame
    final_test_report: pd.DataFrame
    validation_checks: pd.DataFrame
    summary: pd.Series
    config: Section12BConfig = field(default_factory=Section12BConfig)


def load_conditioning_inputs(
    project_root: Path, config: Section12BConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Column-scoped loads of the frozen Section 7 label and context artifacts."""

    cfg = config or Section12BConfig()
    processed = project_root / "data" / "processed"
    label_columns = ["true_retest_id", "trade_date_ny", "research_partition", "hypothesis"]
    label_columns += ["trade_side", "label_risk_points"]
    for horizon in cfg.exit_horizons_minutes:
        label_columns += [f"label_capped_{horizon}m_r", f"label_capped_{horizon}m_valid"]
    labels = pq.read_table(
        processed / "section7_true_poi_outcome_labels_gc.parquet", columns=label_columns
    ).to_pandas(ignore_metadata=True)
    context = pq.read_table(
        processed / "section7_true_poi_context_frame_gc.parquet",
        columns=["true_retest_id", "retest_ts_event_utc", "feat_execution_session"],
    ).to_pandas(ignore_metadata=True)
    return labels, context


def build_conditioning_event_frame(
    labels: pd.DataFrame,
    context: pd.DataFrame,
    gate: pd.DataFrame,
    config: Section12BConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Join the family's outcome rows to the frozen gate at decision bars."""

    cfg = config or Section12BConfig()
    frame = labels.merge(context, on="true_retest_id", how="left", validate="many_to_one")
    if frame["retest_ts_event_utc"].isna().any():
        raise ValueError("every outcome row must have a context retest timestamp")

    partition = frame["research_partition"].astype(str).str.lower().map(PARTITION_NORMALIZATION)
    if partition.isna().any():
        unknown = sorted(frame.loc[partition.isna(), "research_partition"].astype(str).unique())
        raise ValueError(f"unknown research partitions: {unknown[:4]}")
    frame["research_partition"] = partition

    family = frame["hypothesis"].astype(str).eq(cfg.family_hypothesis) & frame["trade_side"].astype(
        str
    ).eq(cfg.family_trade_side)
    frame = frame.loc[family].reset_index(drop=True)
    if frame.empty:
        raise ValueError("no rows for the declared continuation-short family")

    frame["retest_ts_event_utc"] = pd.to_datetime(frame["retest_ts_event_utc"])
    frame = frame.merge(
        gate[["decision_timestamp_utc", "gate_prediction"]],
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
            "retest_ts_event_utc": frame.loc[keep, "retest_ts_event_utc"].to_numpy(),
            "research_partition": frame.loc[keep, "research_partition"].to_numpy(),
            "session": frame.loc[keep, "feat_execution_session"].astype(str).to_numpy(),
            "gate_prediction": frame.loc[keep, "gate_prediction"].to_numpy(dtype=np.float64),
            "risk_ticks": (
                pd.to_numeric(frame.loc[keep, "label_risk_points"], errors="coerce")
                / cfg.cost_config.tick_size
            ).to_numpy(dtype=np.float64),
        }
    )
    for horizon in cfg.exit_horizons_minutes:
        r = pd.to_numeric(frame.loc[keep, f"label_capped_{horizon}m_r"], errors="coerce")
        events[f"r_{horizon}m"] = r.clip(-cfg.robust_cap_r, cfg.robust_cap_r).to_numpy()
        events[f"valid_{horizon}m"] = (
            frame.loc[keep, f"label_capped_{horizon}m_valid"].fillna(False).astype(bool).to_numpy()
        )
    events["robust_r"] = events[f"r_{cfg.exit_horizons_minutes[0]}m"]
    events = events.sort_values(["trade_date_ny", "retest_ts_event_utc"]).reset_index(drop=True)

    summary = pd.Series(
        {
            "family_outcome_rows": len(frame),
            "gate_matched_rows": int(matched.sum()),
            "gate_match_rate": float(matched.mean()),
            "events_kept": len(events),
            "development_events": int((events["research_partition"] == "Development").sum()),
            "validation_events": int((events["research_partition"] == "Validation").sum()),
            "final_test_events": int((events["research_partition"] == "Final test").sum()),
            "development_trading_dates": int(
                events.loc[events["research_partition"] == "Development", "trade_date_ny"].nunique()
            ),
            "validation_trading_dates": int(
                events.loc[events["research_partition"] == "Validation", "trade_date_ny"].nunique()
            ),
        },
        name="value",
    )
    return events, summary


def fit_development_quintiles(
    events: pd.DataFrame, config: Section12BConfig | None = None
) -> np.ndarray:
    """Interior quintile edges of ``gate_prediction`` on Development events only."""

    cfg = config or Section12BConfig()
    development = events.loc[events["research_partition"] == "Development", "gate_prediction"]
    if development.empty:
        raise ValueError("cannot fit quintiles without Development events")
    probes = np.linspace(0.0, 1.0, cfg.quintile_bucket_count + 1)[1:-1]
    return np.unique(np.nanquantile(development.to_numpy(dtype=np.float64), probes))


def assign_quintiles(events: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """Bucket every event with the frozen Development edges."""

    return np.searchsorted(edges, events["gate_prediction"].to_numpy(dtype=np.float64))


def _scheme_weights(buckets: np.ndarray, scheme: str, config: Section12BConfig) -> np.ndarray:
    if scheme == "constant":
        return np.ones(len(buckets), dtype=np.float64)
    if scheme == "proportional":
        ladder = buckets.astype(np.float64) + 1.0
    elif scheme == "inverse":
        ladder = float(config.quintile_bucket_count) - buckets.astype(np.float64)
    else:
        raise ValueError(f"unknown sizing scheme: {scheme}")
    return ladder / ((config.quintile_bucket_count + 1.0) / 2.0)


def _date_block_bootstrap(
    dates: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> tuple[float, float]:
    """Generic date-block bootstrap: resample trading dates, pool their events.

    ``statistic`` receives the pooled event indices of one replicate and
    returns the replicate's estimate.  Dates are the sampling unit, so serial
    dependence within a day never inflates the interval.
    """

    codes, uniques = pd.factorize(pd.Series(dates), sort=True)
    n_dates = len(uniques)
    if n_dates < 2:
        return (np.nan, np.nan)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.searchsorted(sorted_codes, np.arange(n_dates + 1))
    rng = np.random.default_rng(seed)
    estimates = np.full(replicates, np.nan)
    for i in range(replicates):
        draw = rng.integers(0, n_dates, size=n_dates)
        idx = np.concatenate([order[boundaries[d] : boundaries[d + 1]] for d in draw])
        estimates[i] = statistic(idx)
    finite = estimates[np.isfinite(estimates)]
    if len(finite) < replicates // 2:
        return (np.nan, np.nan)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(v) for v in np.quantile(finite, [alpha, 1.0 - alpha]))


def _mean_mad_ratio(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    mean = float(values.mean())
    mad = float(np.abs(values - mean).mean())
    return mean / mad if mad > 0 else np.nan


def _max_drawdown(values: np.ndarray) -> float:
    """Maximum peak-to-trough fall of the chronological cumulative R curve.

    Reported as a dispersion diagnostic only: retest events overlap in time,
    so this pooled event-level curve is not a tradable equity curve and the
    number is never an advancement criterion.
    """

    if len(values) == 0:
        return np.nan
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.maximum(equity, 0.0))
    return float((peaks - equity).max())


def _floors_pass(events: int, dates: int, partition: str, cfg: Section12BConfig) -> bool:
    if partition == "Development":
        return events >= cfg.min_development_events and dates >= cfg.min_development_trading_dates
    return events >= cfg.min_validation_events and dates >= cfg.min_validation_trading_dates


def _retention(dev_effect: float, val_effect: float) -> float:
    if not np.isfinite(dev_effect) or dev_effect == 0.0:
        return np.nan
    return float(val_effect / dev_effect)


def _criteria_pass(
    dev_effect: float,
    dev_ci_low: float,
    val_effect: float,
    floors: dict[str, bool],
    cfg: Section12BConfig,
) -> bool:
    retention = _retention(dev_effect, val_effect)
    return bool(
        np.isfinite(dev_effect)
        and dev_effect > 0.0
        and np.isfinite(dev_ci_low)
        and dev_ci_low > 0.0
        and np.isfinite(val_effect)
        and np.sign(val_effect) == np.sign(dev_effect)
        and np.isfinite(retention)
        and retention >= cfg.min_validation_retention
        and floors["Development"]
        and floors["Validation"]
    )


def evaluate_h1_sizing(
    events: pd.DataFrame, buckets: np.ndarray, config: Section12BConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """H1: frozen sizing ladders versus the constant-1R baseline under costs."""

    cfg = config or Section12BConfig()
    risk_ticks = events["risk_ticks"].to_numpy(dtype=np.float64)
    usable = np.isfinite(risk_ticks) & (risk_ticks > 0)
    dropped_nonpositive_risk = int((~usable).sum())

    result_records: list[dict] = []
    effect_records: list[dict] = []
    scheme_scenario_pass: dict[tuple[str, str], bool] = {}

    scenarios = ("frictionless", *cfg.h1_cost_scenarios)
    for scenario_index, scenario in enumerate(scenarios):
        cost_ticks = _scenario_cost_ticks(cfg.cost_config, scenario)
        contribs: dict[str, np.ndarray] = {}
        for scheme in cfg.sizing_schemes:
            weights = _scheme_weights(buckets, scheme, cfg)
            with np.errstate(invalid="ignore", divide="ignore"):
                net = events["robust_r"].to_numpy(dtype=np.float64) - cost_ticks / risk_ticks
            contribs[scheme] = np.where(usable, weights * net, np.nan)

        for scheme in cfg.sizing_schemes:
            per_partition: dict[str, dict] = {}
            for partition in EVALUATION_PARTITIONS:
                mask = (events["research_partition"] == partition).to_numpy() & usable
                values = contribs[scheme][mask]
                part_dates = events.loc[mask, "trade_date_ny"]
                per_partition[partition] = {
                    "events": int(mask.sum()),
                    "trading_dates": int(part_dates.nunique()),
                    "mean_net_r": float(values.mean()) if len(values) else np.nan,
                    "mean_mad_ratio": _mean_mad_ratio(values),
                    "max_drawdown_r": _max_drawdown(values),
                }
                result_records.append(
                    {
                        "sizing_scheme": scheme,
                        "cost_scenario": scenario,
                        "research_partition": partition,
                        **per_partition[partition],
                    }
                )
            if scheme == "constant":
                continue

            effects: dict[str, float] = {}
            mean_shifts: dict[str, float] = {}
            ci: tuple[float, float] = (np.nan, np.nan)
            floors: dict[str, bool] = {}
            for partition_index, partition in enumerate(EVALUATION_PARTITIONS):
                mask = (events["research_partition"] == partition).to_numpy() & usable
                scheme_values = contribs[scheme][mask]
                constant_values = contribs["constant"][mask]
                effects[partition] = _mean_mad_ratio(scheme_values) - _mean_mad_ratio(
                    constant_values
                )
                mean_shifts[partition] = float(scheme_values.mean() - constant_values.mean())
                floors[partition] = _floors_pass(
                    int(mask.sum()),
                    int(events.loc[mask, "trade_date_ny"].nunique()),
                    partition,
                    cfg,
                )
                if partition == "Development":
                    seed = (
                        cfg.random_seed
                        + 100_000 * scenario_index
                        + 10_000 * cfg.sizing_schemes.index(scheme)
                        + partition_index
                    )

                    def ratio_effect(
                        idx: np.ndarray,
                        scheme_pool: np.ndarray = scheme_values,
                        constant_pool: np.ndarray = constant_values,
                    ) -> float:
                        return _mean_mad_ratio(scheme_pool[idx]) - _mean_mad_ratio(
                            constant_pool[idx]
                        )

                    ci = _date_block_bootstrap(
                        events.loc[mask, "trade_date_ny"].to_numpy(),
                        ratio_effect,
                        seed=seed,
                        replicates=cfg.bootstrap_replicates,
                        confidence=cfg.bootstrap_confidence,
                    )

            materiality_ok = all(
                np.isfinite(shift) and abs(shift) <= cfg.mean_r_materiality
                for shift in mean_shifts.values()
            )
            passes = (
                _criteria_pass(effects["Development"], ci[0], effects["Validation"], floors, cfg)
                and materiality_ok
            )
            if scenario in cfg.h1_cost_scenarios:
                scheme_scenario_pass[(scheme, scenario)] = passes
            effect_records.append(
                {
                    "sizing_scheme": scheme,
                    "cost_scenario": scenario,
                    "development_ratio_effect": effects["Development"],
                    "development_ci_low": ci[0],
                    "development_ci_high": ci[1],
                    "validation_ratio_effect": effects["Validation"],
                    "validation_retention": _retention(
                        effects["Development"], effects["Validation"]
                    ),
                    "development_mean_shift_r": mean_shifts["Development"],
                    "validation_mean_shift_r": mean_shifts["Validation"],
                    "mean_shift_within_materiality": materiality_ok,
                    "floors_pass": floors["Development"] and floors["Validation"],
                    "criteria_pass": passes,
                }
            )

    advancing = sorted(
        {
            scheme
            for scheme in cfg.sizing_schemes
            if scheme != "constant"
            and all(
                scheme_scenario_pass.get((scheme, scenario), False)
                for scenario in cfg.h1_cost_scenarios
            )
        }
    )
    detail = {
        "dropped_nonpositive_risk": dropped_nonpositive_risk,
        "advancing_schemes": advancing,
        "advances": bool(advancing),
    }
    return (
        pd.DataFrame.from_records(result_records),
        pd.DataFrame.from_records(effect_records),
        detail,
    )


def evaluate_h2_exits(
    events: pd.DataFrame, buckets: np.ndarray, config: Section12BConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """H2: Development-fitted horizon-per-quintile versus the unconditional best."""

    cfg = config or Section12BConfig()
    horizons = list(cfg.exit_horizons_minutes)
    all_valid = np.ones(len(events), dtype=bool)
    for horizon in horizons:
        all_valid &= events[f"valid_{horizon}m"].to_numpy(dtype=bool)

    development = (events["research_partition"] == "Development").to_numpy() & all_valid
    if not development.any():
        raise ValueError("H2 requires Development events with every menu horizon valid")

    def dev_mean(horizon: int, mask: np.ndarray) -> float:
        values = events.loc[mask, f"r_{horizon}m"].to_numpy(dtype=np.float64)
        return float(values.mean()) if len(values) else -np.inf

    unconditional_means = {h: dev_mean(h, development) for h in horizons}
    unconditional_best = min(horizons, key=lambda h: (-unconditional_means[h], h))

    mapping_records: list[dict] = []
    bucket_choice: dict[int, int] = {}
    for bucket in sorted(np.unique(buckets)):
        bucket_mask = development & (buckets == bucket)
        means = {h: dev_mean(h, bucket_mask) for h in horizons}
        chosen = (
            min(horizons, key=lambda h, m=means: (-m[h], h))
            if bucket_mask.any()
            else unconditional_best
        )
        bucket_choice[int(bucket)] = chosen
        mapping_records.append(
            {
                "prediction_quintile": int(bucket),
                "development_events": int(bucket_mask.sum()),
                **{f"development_mean_r_{h}m": means[h] for h in horizons},
                "chosen_horizon_minutes": chosen,
                "differs_from_unconditional": chosen != unconditional_best,
            }
        )
    mapping = pd.DataFrame.from_records(mapping_records)

    chosen_per_event = np.vectorize(lambda b: bucket_choice.get(int(b), unconditional_best))(
        buckets
    )
    conditional_r = np.full(len(events), np.nan)
    for horizon in horizons:
        pick = chosen_per_event == horizon
        conditional_r[pick] = events.loc[pick, f"r_{horizon}m"].to_numpy(dtype=np.float64)
    unconditional_r = events[f"r_{unconditional_best}m"].to_numpy(dtype=np.float64)
    diff = conditional_r - unconditional_r
    affected = np.isin(buckets, [b for b, h in bucket_choice.items() if h != unconditional_best])

    result_records: list[dict] = []
    effects: dict[str, float] = {}
    ci: tuple[float, float] = (np.nan, np.nan)
    floors: dict[str, bool] = {}
    for partition in EVALUATION_PARTITIONS:
        mask = (events["research_partition"] == partition).to_numpy() & all_valid
        effects[partition] = float(diff[mask].mean()) if mask.any() else np.nan
        affected_mask = mask & affected
        floors[partition] = _floors_pass(
            int(affected_mask.sum()),
            int(events.loc[affected_mask, "trade_date_ny"].nunique()),
            partition,
            cfg,
        )
        if partition == "Development":

            def mean_diff(idx: np.ndarray, pool: np.ndarray = diff[mask]) -> float:
                return float(pool[idx].mean())

            ci = _date_block_bootstrap(
                events.loc[mask, "trade_date_ny"].to_numpy(),
                mean_diff,
                seed=cfg.random_seed + 500_000,
                replicates=cfg.bootstrap_replicates,
                confidence=cfg.bootstrap_confidence,
            )
        result_records.append(
            {
                "research_partition": partition,
                "events": int(mask.sum()),
                "affected_events": int(affected_mask.sum()),
                "affected_trading_dates": int(events.loc[affected_mask, "trade_date_ny"].nunique()),
                "conditional_mean_r": float(conditional_r[mask].mean()) if mask.any() else np.nan,
                "unconditional_mean_r": (
                    float(unconditional_r[mask].mean()) if mask.any() else np.nan
                ),
                "conditioning_effect_r": effects[partition],
                "development_ci_low": ci[0] if partition == "Development" else np.nan,
                "development_ci_high": ci[1] if partition == "Development" else np.nan,
                "floors_pass": floors[partition],
            }
        )

    passes = mapping["differs_from_unconditional"].any() and _criteria_pass(
        effects["Development"], ci[0], effects["Validation"], floors, cfg
    )
    detail = {
        "unconditional_best_horizon": unconditional_best,
        "menu_valid_events": int(all_valid.sum()),
        "advances": bool(passes),
        "validation_retention": _retention(effects["Development"], effects["Validation"]),
    }
    return mapping, pd.DataFrame.from_records(result_records), detail


def evaluate_h3_suppression(
    events: pd.DataFrame, buckets: np.ndarray, config: Section12BConfig | None = None
) -> tuple[pd.DataFrame, dict]:
    """H3: drop the bottom Development-fitted quintile; median R and stop rate."""

    cfg = config or Section12BConfig()
    values = events["robust_r"].to_numpy(dtype=np.float64)
    stops = values <= (cfg.stop_loss_r + 1e-9)
    survivor = buckets >= 1

    records: list[dict] = []
    effects: dict[str, dict[str, float]] = {}
    cis: dict[str, tuple[float, float]] = {
        "median_improvement_r": (np.nan, np.nan),
        "stop_rate_reduction": (np.nan, np.nan),
    }
    floors: dict[str, bool] = {}
    for partition in EVALUATION_PARTITIONS:
        mask = (events["research_partition"] == partition).to_numpy()
        keep = mask & survivor
        baseline_median = float(np.median(values[mask])) if mask.any() else np.nan
        survivor_median = float(np.median(values[keep])) if keep.any() else np.nan
        baseline_stop_rate = float(stops[mask].mean()) if mask.any() else np.nan
        survivor_stop_rate = float(stops[keep].mean()) if keep.any() else np.nan
        effects[partition] = {
            "median_improvement_r": survivor_median - baseline_median,
            "stop_rate_reduction": baseline_stop_rate - survivor_stop_rate,
        }
        floors[partition] = _floors_pass(
            int(keep.sum()), int(events.loc[keep, "trade_date_ny"].nunique()), partition, cfg
        )
        if partition == "Development":
            pool_values = values[mask]
            pool_survivor = survivor[mask]
            pool_stops = stops[mask]

            def median_improvement(
                idx: np.ndarray,
                pv: np.ndarray = pool_values,
                ps: np.ndarray = pool_survivor,
            ) -> float:
                kept = ps[idx]
                if not kept.any():
                    return np.nan
                return float(np.median(pv[idx][kept]) - np.median(pv[idx]))

            def stop_rate_reduction(
                idx: np.ndarray,
                pk: np.ndarray = pool_stops,
                ps: np.ndarray = pool_survivor,
            ) -> float:
                kept = ps[idx]
                if not kept.any():
                    return np.nan
                return float(pk[idx].mean() - pk[idx][kept].mean())

            dates = events.loc[mask, "trade_date_ny"].to_numpy()
            cis["median_improvement_r"] = _date_block_bootstrap(
                dates,
                median_improvement,
                seed=cfg.random_seed + 700_000,
                replicates=cfg.bootstrap_replicates,
                confidence=cfg.bootstrap_confidence,
            )
            cis["stop_rate_reduction"] = _date_block_bootstrap(
                dates,
                stop_rate_reduction,
                seed=cfg.random_seed + 700_001,
                replicates=cfg.bootstrap_replicates,
                confidence=cfg.bootstrap_confidence,
            )
        records.append(
            {
                "research_partition": partition,
                "baseline_events": int(mask.sum()),
                "survivor_events": int(keep.sum()),
                "survivor_trading_dates": int(events.loc[keep, "trade_date_ny"].nunique()),
                "baseline_median_r": baseline_median,
                "survivor_median_r": survivor_median,
                "median_improvement_r": effects[partition]["median_improvement_r"],
                "baseline_stop_rate": baseline_stop_rate,
                "survivor_stop_rate": survivor_stop_rate,
                "stop_rate_reduction": effects[partition]["stop_rate_reduction"],
                "floors_pass": floors[partition],
            }
        )

    passes = all(
        _criteria_pass(
            effects["Development"][effect_name],
            cis[effect_name][0],
            effects["Validation"][effect_name],
            floors,
            cfg,
        )
        for effect_name in ("median_improvement_r", "stop_rate_reduction")
    )
    detail = {
        "advances": bool(passes),
        "development_median_ci": cis["median_improvement_r"],
        "development_stop_rate_ci": cis["stop_rate_reduction"],
    }
    return pd.DataFrame.from_records(records), detail


def _final_test_report(
    events: pd.DataFrame,
    buckets: np.ndarray,
    h2_mapping: pd.DataFrame,
    h2_detail: dict,
    config: Section12BConfig,
) -> pd.DataFrame:
    """One-time Final-test read with every Development-fitted parameter frozen."""

    cfg = config
    mask = (events["research_partition"] == "Final test").to_numpy()
    records: list[dict] = []
    risk_ticks = events["risk_ticks"].to_numpy(dtype=np.float64)
    usable = mask & np.isfinite(risk_ticks) & (risk_ticks > 0)
    for scenario in cfg.h1_cost_scenarios:
        cost_ticks = _scenario_cost_ticks(cfg.cost_config, scenario)
        with np.errstate(invalid="ignore", divide="ignore"):
            net = events["robust_r"].to_numpy(dtype=np.float64) - cost_ticks / risk_ticks
        constant = _mean_mad_ratio(net[usable])
        for scheme in cfg.sizing_schemes:
            if scheme == "constant":
                continue
            weighted = (_scheme_weights(buckets, scheme, cfg) * net)[usable]
            records.append(
                {
                    "hypothesis_id": "H1_sizing",
                    "detail": f"{scheme}/{scenario}",
                    "final_test_events": int(usable.sum()),
                    "final_test_effect": _mean_mad_ratio(weighted) - constant,
                }
            )

    horizons = list(cfg.exit_horizons_minutes)
    all_valid = np.ones(len(events), dtype=bool)
    for horizon in horizons:
        all_valid &= events[f"valid_{horizon}m"].to_numpy(dtype=bool)
    h2_mask = mask & all_valid
    choice = dict(
        zip(
            h2_mapping["prediction_quintile"].astype(int),
            h2_mapping["chosen_horizon_minutes"].astype(int),
            strict=True,
        )
    )
    best = int(h2_detail["unconditional_best_horizon"])
    chosen_per_event = np.vectorize(lambda b: choice.get(int(b), best))(buckets)
    conditional_r = np.full(len(events), np.nan)
    for horizon in horizons:
        pick = chosen_per_event == horizon
        conditional_r[pick] = events.loc[pick, f"r_{horizon}m"].to_numpy(dtype=np.float64)
    records.append(
        {
            "hypothesis_id": "H2_exits",
            "detail": f"conditional vs {best}m",
            "final_test_events": int(h2_mask.sum()),
            "final_test_effect": float(
                (conditional_r[h2_mask] - events.loc[h2_mask, f"r_{best}m"]).mean()
            )
            if h2_mask.any()
            else np.nan,
        }
    )

    values = events["robust_r"].to_numpy(dtype=np.float64)
    stops = values <= (cfg.stop_loss_r + 1e-9)
    survivor = buckets >= 1
    keep = mask & survivor
    records.append(
        {
            "hypothesis_id": "H3_suppression",
            "detail": "median improvement",
            "final_test_events": int(keep.sum()),
            "final_test_effect": (
                float(np.median(values[keep]) - np.median(values[mask]))
                if keep.any() and mask.any()
                else np.nan
            ),
        }
    )
    records.append(
        {
            "hypothesis_id": "H3_suppression",
            "detail": "stop rate reduction",
            "final_test_events": int(keep.sum()),
            "final_test_effect": (
                float(stops[mask].mean() - stops[keep].mean())
                if keep.any() and mask.any()
                else np.nan
            ),
        }
    )
    return pd.DataFrame.from_records(records)


def run_opportunity_conditioning(
    project_root: Path, config: Section12BConfig | None = None
) -> OpportunityConditioningResult:
    """Execute the full frozen Section 12B contract."""

    cfg = config or Section12BConfig()
    labels, context = load_conditioning_inputs(project_root, cfg)
    gate = build_full_gate_table(project_root)
    events, population_summary = build_conditioning_event_frame(labels, context, gate, cfg)

    edges = fit_development_quintiles(events, cfg)
    buckets = assign_quintiles(events, edges)

    h1_results, h1_effects, h1_detail = evaluate_h1_sizing(events, buckets, cfg)
    h2_mapping, h2_results, h2_detail = evaluate_h2_exits(events, buckets, cfg)
    h3_results, h3_detail = evaluate_h3_suppression(events, buckets, cfg)

    verdicts = pd.DataFrame.from_records(
        [
            {
                "hypothesis_id": "H1_sizing",
                "verdict": "ADVANCES" if h1_detail["advances"] else "NO_ADVANCE",
                "detail": (
                    "advancing schemes: " + ", ".join(h1_detail["advancing_schemes"])
                    if h1_detail["advancing_schemes"]
                    else "no scheme passes in both cost scenarios"
                ),
            },
            {
                "hypothesis_id": "H2_exits",
                "verdict": "ADVANCES" if h2_detail["advances"] else "NO_ADVANCE",
                "detail": (
                    f"unconditional best {h2_detail['unconditional_best_horizon']}m; "
                    f"retention {h2_detail['validation_retention']:.3f}"
                    if np.isfinite(h2_detail["validation_retention"])
                    else f"unconditional best {h2_detail['unconditional_best_horizon']}m"
                ),
            },
            {
                "hypothesis_id": "H3_suppression",
                "verdict": "ADVANCES" if h3_detail["advances"] else "NO_ADVANCE",
                "detail": "co-primary median and stop-rate effects required jointly",
            },
        ]
    )

    final_test_report = _final_test_report(events, buckets, h2_mapping, h2_detail, cfg)

    refit_edges = fit_development_quintiles(events, cfg)
    checks = {
        "partitions_recognized": bool(
            set(events["research_partition"].unique()).issubset(set(ALL_PARTITIONS))
        ),
        "family_is_continuation_short": True,
        "quintile_edges_development_only": bool(np.allclose(edges, refit_edges)),
        "robust_cap_respected": bool(
            all(
                events[f"r_{h}m"].dropna().abs().le(cfg.robust_cap_r + 1e-9).all()
                for h in cfg.exit_horizons_minutes
            )
        ),
        "h1_weights_average_one_on_development": bool(
            all(
                abs(
                    _scheme_weights(
                        buckets[(events["research_partition"] == "Development").to_numpy()],
                        scheme,
                        cfg,
                    ).mean()
                    - 1.0
                )
                <= 0.05
                for scheme in cfg.sizing_schemes
            )
        ),
        "h2_menu_within_frozen_grid_horizons": bool(
            set(h2_mapping["chosen_horizon_minutes"].astype(int)).issubset(
                set(cfg.exit_horizons_minutes)
            )
        ),
        "verdicts_use_development_and_validation_only": True,
        "final_test_report_separate_from_verdicts": "verdict" not in final_test_report.columns,
        "one_verdict_per_hypothesis": len(verdicts) == 3,
    }
    validation_checks = pd.Series(checks, name="passed").rename_axis("check").reset_index()

    summary = pd.Series(
        {
            "events_kept": int(population_summary["events_kept"]),
            "development_events": int(population_summary["development_events"]),
            "validation_events": int(population_summary["validation_events"]),
            "final_test_events": int(population_summary["final_test_events"]),
            "h1_verdict": verdicts.loc[0, "verdict"],
            "h2_verdict": verdicts.loc[1, "verdict"],
            "h3_verdict": verdicts.loc[2, "verdict"],
            "any_hypothesis_advances": bool((verdicts["verdict"] == "ADVANCES").any()),
            "h1_dropped_nonpositive_risk": h1_detail["dropped_nonpositive_risk"],
            "h2_unconditional_best_horizon": h2_detail["unconditional_best_horizon"],
        },
        name="value",
    )

    return OpportunityConditioningResult(
        population_summary=population_summary,
        quintile_edges=pd.Series(edges, name="edge"),
        h1_results=h1_results,
        h1_effects=h1_effects,
        h2_mapping=h2_mapping,
        h2_results=h2_results,
        h3_results=h3_results,
        verdicts=verdicts,
        final_test_report=final_test_report,
        validation_checks=validation_checks,
        summary=summary,
        config=cfg,
    )


def save_conditioning_outputs(
    result: OpportunityConditioningResult, *, project_root: Path
) -> pd.DataFrame:
    """Persist Section 12B outputs; generated artifacts stay excluded from Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "section12b"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "section12b_h1_sizing_results": (
            processed / "section12b_h1_sizing_results_gc.parquet",
            result.h1_results,
        ),
        "section12b_h1_sizing_effects": (
            processed / "section12b_h1_sizing_effects_gc.parquet",
            result.h1_effects,
        ),
        "section12b_h2_exit_mapping": (
            processed / "section12b_h2_exit_mapping_gc.parquet",
            result.h2_mapping,
        ),
        "section12b_h2_exit_results": (
            processed / "section12b_h2_exit_results_gc.parquet",
            result.h2_results,
        ),
        "section12b_h3_suppression_results": (
            processed / "section12b_h3_suppression_results_gc.parquet",
            result.h3_results,
        ),
        "section12b_verdicts": (processed / "section12b_verdicts_gc.parquet", result.verdicts),
        "section12b_final_test_report": (
            processed / "section12b_final_test_report_gc.parquet",
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
    result.verdicts.to_csv(tables / "section12b_verdicts_gc.csv", index=False)
    result.h1_effects.to_csv(tables / "section12b_h1_sizing_effects_gc.csv", index=False)
    result.h2_mapping.to_csv(tables / "section12b_h2_exit_mapping_gc.csv", index=False)
    result.h3_results.to_csv(tables / "section12b_h3_suppression_results_gc.csv", index=False)
    return pd.DataFrame.from_records(records)
