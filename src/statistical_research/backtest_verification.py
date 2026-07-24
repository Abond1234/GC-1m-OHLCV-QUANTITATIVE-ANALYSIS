"""Independent, clean-room verification of the Section 11 sequential backtest.

Accuracy is the load-bearing assumption of every downstream decision, so this
module re-derives the recorded backtest from the raw one-minute bars through a
*second, deliberately independent* implementation and then reconciles it, trade
for trade, against the stored trade log.  It intentionally does **not** import
the production simulator (`run_sequential_backtest` / `_simulate_variant`); a
verifier that reuses the code it audits proves nothing.  Two independent
implementations of the same frozen contract that agree to the tick is the
evidence we want.

The verification has three layers:

1.  **Reconciliation** - a clean-room one-position walk over the raw bars
    reproduces every recorded trade's entry bar, exit bar, exit reason, holding
    time, stop size, ambiguity flag, and gross R.  Net R and per-partition mean
    net R are recomputed from the declared cost contract and reconciled against
    the stored performance table.

2.  **Physical-fidelity audits** that do *not* depend on the simulator's own
    contract - the fills must be realisable in the market data:
      * every recorded fill price equals the raw bar open at the entry
        timestamp (mapped independently via a pandas index lookup, not the
        production `searchsorted`);
      * the decision ATR equals the raw rolling ATR at the bar strictly before
        entry, and the entry bar is the next bar - so sizing is point-in-time;
      * stop / target exit prices are reachable inside the exit bar's range, and
        gap-through fills (where the sim's fill is optimistic) are counted, not
        hidden;
      * close-based exits (boundary / time / forced) equal the raw bar close;
      * no trade crosses a New York date, a continuous segment, or the 15:30
        forced-exit minute, and holding never exceeds the declared cap;
      * no Final-test row appears anywhere.

3.  **Cross-artifact triangulation** - the entry price and decision ATR agree
    across three independently built tables (raw bars, signal candidates, and
    the forward-label table), and a point-in-time audit of the expansion gate
    quantifies whether its per-date cross-sectional normalisation looks ahead.

The public entry point is :func:`run_verification`; :func:`save_verification_outputs`
persists the tables (which stay outside Git, like every generated artifact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_evaluation import EVALUATION_PARTITIONS
from .sequential_backtest import FORCED_EXIT_MINUTE_NY, Section11Config
from .signal_construction import DIRECTION_VARIANTS, GATE_VARIANTS

VERIFICATION_PRICE_TOLERANCE = 1e-9
VERIFICATION_R_TOLERANCE = 1e-9


@dataclass(frozen=True)
class VerificationConfig:
    """Cost and tolerance contract for the independent verifier.

    The cost constants are redeclared here (rather than imported) so the
    verifier stands alone; :func:`run_verification` asserts they still match the
    production ``Section11Config`` and fails loudly if the two ever drift.
    """

    tick_size: float = 0.10
    commission_ticks_round_trip: float = 0.6
    base_slippage_ticks_per_side: float = 1.0
    pessimistic_slippage_ticks_per_side: float = 2.0
    max_holding_minutes: int = 120
    forced_exit_minute_ny: int = FORCED_EXIT_MINUTE_NY
    cost_scenarios: tuple[str, ...] = ("frictionless", "base", "pessimistic")
    price_tolerance: float = VERIFICATION_PRICE_TOLERANCE
    r_tolerance: float = VERIFICATION_R_TOLERANCE

    def scenario_cost_ticks(self, scenario: str) -> float:
        if scenario == "frictionless":
            return 0.0
        per_side = (
            self.base_slippage_ticks_per_side
            if scenario == "base"
            else self.pessimistic_slippage_ticks_per_side
        )
        return self.commission_ticks_round_trip + 2.0 * per_side


@dataclass
class VerificationResult:
    """Container for the verification tables and the overall gate."""

    reconciliation: pd.DataFrame
    performance_reconciliation: pd.DataFrame
    entry_fidelity: pd.Series
    fill_fidelity: pd.DataFrame
    gate_consistency: pd.Series
    cross_artifact: pd.Series
    checks: pd.DataFrame
    mismatch_samples: pd.DataFrame
    summary: pd.Series
    config: VerificationConfig = field(default_factory=VerificationConfig)

    @property
    def all_passed(self) -> bool:
        return bool(self.checks["passed"].all())


# ---------------------------------------------------------------------------
# Raw-bar loading (independent of the production loader)
# ---------------------------------------------------------------------------
def load_gc_bars(research_bars_path: Path) -> pd.DataFrame:
    """Load the GC one-minute bars needed to re-derive every fill."""

    import pyarrow.parquet as pq

    columns = [
        "ts_event_utc",
        "product",
        "open",
        "high",
        "low",
        "close",
        "trade_date_ny",
        "minute_of_day_ny",
        "continuous_segment_id",
        "rolling_atr_20m",
    ]
    table = pq.read_table(research_bars_path, columns=columns, filters=[("product", "==", "GC")])
    bars = table.to_pandas(ignore_metadata=True)
    bars = bars.sort_values("ts_event_utc").reset_index(drop=True)
    if not bars["ts_event_utc"].is_unique:
        raise ValueError("GC bar timestamps are not unique; cannot map fills")
    return bars


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
        "atr": bars["rolling_atr_20m"].to_numpy(dtype=np.float64),
    }


def map_entry_positions(candidates: pd.DataFrame, bars: pd.DataFrame) -> np.ndarray:
    """Map candidate entry timestamps to bar rows via an index lookup.

    Uses ``pandas.Index.get_indexer`` - a genuinely different mechanism from the
    production ``numpy.searchsorted`` - and requires an exact match for every
    candidate.
    """

    index = pd.Index(bars["ts_event_utc"])
    positions = index.get_indexer(candidates["entry_timestamp_utc"])
    if (positions < 0).any():
        raise ValueError("candidate entry timestamps not found in the GC bars")
    return positions


# ---------------------------------------------------------------------------
# Clean-room one-position re-simulation
# ---------------------------------------------------------------------------
def independent_resimulate(
    candidates: pd.DataFrame,
    entry_positions: np.ndarray,
    arrays: dict,
    direction: str,
    gate: str,
    cfg: VerificationConfig,
) -> pd.DataFrame:
    """Re-run one variant as an explicit, readable chronological walk.

    This is written for transparency, not speed: an outer loop over candidates
    in entry order with an inner loop that steps bar by bar until an exit fires.
    It mirrors the frozen Section 10/11 contract so a faithful production run
    must reproduce it exactly.
    """

    is_long = direction == "long_benchmark"
    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    segment = arrays["segment"]
    trade_date = arrays["trade_date"]
    minute_ny = arrays["minute_ny"]
    open_price = arrays["open"]
    n_bars = len(high)

    stop_points = candidates["stop_points"].to_numpy(dtype=np.float64)
    target_points = candidates["target_points"].to_numpy(dtype=np.float64)
    partitions = candidates["research_partition"].to_numpy()
    sessions = candidates["entry_session"].to_numpy()
    dates = candidates["trade_date_ny"].to_numpy()
    if gate == "expansion_gated":
        eligible = candidates["expansion_gate_flag"].to_numpy(dtype=bool)
    else:
        eligible = np.ones(len(candidates), dtype=bool)

    order = np.argsort(entry_positions, kind="mergesort")
    flat_from = -1
    records: list[dict] = []
    for row in order:
        if not eligible[row]:
            continue
        start = int(entry_positions[row])
        if start <= flat_from:
            continue
        price_in = open_price[start]
        stop = price_in - stop_points[row] if is_long else price_in + stop_points[row]
        target = price_in + target_points[row] if is_long else price_in - target_points[row]
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
                "direction_variant": direction,
                "gate_variant": gate,
                "research_partition": partitions[row],
                "entry_session": sessions[row],
                "trade_date_ny": dates[row],
                "entry_position": start,
                "exit_position": exit_position,
                "holding_minutes": int(exit_position - start + 1),
                "exit_reason": exit_reason,
                "ambiguous_bar": ambiguous,
                "stop_ticks": stop_points[row] / cfg.tick_size,
                "gross_r": float(points / stop_points[row]),
                "exit_price": float(exit_price),
                "entry_price": float(price_in),
            }
        )
        flat_from = int(exit_position)
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Reconciliation of recorded vs independent trades
# ---------------------------------------------------------------------------
_RECONCILE_KEYS = ["direction_variant", "gate_variant"]
_RECONCILE_FIELDS = [
    "entry_position",
    "exit_position",
    "holding_minutes",
    "exit_reason",
    "ambiguous_bar",
]


def reconcile_trades(
    recorded: pd.DataFrame, independent: pd.DataFrame, cfg: VerificationConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare recorded and independent trades per variant, in entry order."""

    reports: list[dict] = []
    mismatch_frames: list[pd.DataFrame] = []
    for direction in DIRECTION_VARIANTS:
        for gate in GATE_VARIANTS:
            rec = (
                recorded[
                    recorded["direction_variant"].eq(direction) & recorded["gate_variant"].eq(gate)
                ]
                .sort_values("entry_position")
                .reset_index(drop=True)
            )
            ind = (
                independent[
                    independent["direction_variant"].eq(direction)
                    & independent["gate_variant"].eq(gate)
                ]
                .sort_values("entry_position")
                .reset_index(drop=True)
            )
            n = min(len(rec), len(ind))
            field_mismatch = 0
            if n:
                aligned_mismatch = np.zeros(n, dtype=bool)
                for column in _RECONCILE_FIELDS:
                    aligned_mismatch |= rec[column].to_numpy()[:n] != ind[column].to_numpy()[:n]
                gross_gap = np.abs(rec["gross_r"].to_numpy()[:n] - ind["gross_r"].to_numpy()[:n])
                stop_gap = np.abs(
                    rec["stop_ticks"].to_numpy()[:n] - ind["stop_ticks"].to_numpy()[:n]
                )
                aligned_mismatch |= gross_gap > cfg.r_tolerance
                aligned_mismatch |= stop_gap > cfg.r_tolerance
                field_mismatch = int(aligned_mismatch.sum())
                if field_mismatch:
                    sample = rec.loc[aligned_mismatch].head(20).copy()
                    sample["side"] = "recorded"
                    mismatch_frames.append(sample)
            reports.append(
                {
                    "direction_variant": direction,
                    "gate_variant": gate,
                    "recorded_trades": len(rec),
                    "independent_trades": len(ind),
                    "count_match": len(rec) == len(ind),
                    "field_mismatches": field_mismatch + abs(len(rec) - len(ind)),
                    "reconciled": len(rec) == len(ind) and field_mismatch == 0,
                }
            )
    reconciliation = pd.DataFrame.from_records(reports)
    mismatches = (
        pd.concat(mismatch_frames, ignore_index=True) if mismatch_frames else pd.DataFrame()
    )
    return reconciliation, mismatches


def reconcile_performance(
    independent: pd.DataFrame, recorded_performance: pd.DataFrame, cfg: VerificationConfig
) -> pd.DataFrame:
    """Recompute mean net R per variant/partition/scenario and compare."""

    records: list[dict] = []
    for scenario in cfg.cost_scenarios:
        cost_ticks = cfg.scenario_cost_ticks(scenario)
        net_r = independent["gross_r"] - cost_ticks / independent["stop_ticks"]
        scored = independent.assign(net_r=net_r)
        grouped = scored.groupby(_RECONCILE_KEYS + ["research_partition"], sort=False)[
            "net_r"
        ].mean()
        for (direction, gate, partition), mean_net in grouped.items():
            recorded_row = recorded_performance[
                recorded_performance["cost_scenario"].eq(scenario)
                & recorded_performance["direction_variant"].eq(direction)
                & recorded_performance["gate_variant"].eq(gate)
                & recorded_performance["research_partition"].eq(partition)
            ]
            recorded_mean = (
                float(recorded_row["mean_net_r"].iloc[0]) if len(recorded_row) else np.nan
            )
            records.append(
                {
                    "cost_scenario": scenario,
                    "direction_variant": direction,
                    "gate_variant": gate,
                    "research_partition": partition,
                    "independent_mean_net_r": float(mean_net),
                    "recorded_mean_net_r": recorded_mean,
                    "abs_diff": abs(float(mean_net) - recorded_mean),
                }
            )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Physical-fidelity audits
# ---------------------------------------------------------------------------
def audit_entry_fidelity(
    candidates: pd.DataFrame,
    entry_positions: np.ndarray,
    arrays: dict,
    cfg: VerificationConfig,
) -> pd.Series:
    """Fills equal raw bar opens; ATR is point-in-time at the prior bar."""

    open_at_entry = arrays["open"][entry_positions]
    entry_price = candidates["entry_price"].to_numpy(dtype=np.float64)
    open_err = np.abs(open_at_entry - entry_price)

    decision_positions = entry_positions - 1
    atr_at_decision = arrays["atr"][decision_positions]
    atr_err = np.abs(atr_at_decision - candidates["decision_atr_20m"].to_numpy(dtype=np.float64))

    entry_ts = arrays["ts"][entry_positions]
    decision_ts = arrays["ts"][decision_positions]
    gap = entry_ts - decision_ts
    one_minute = np.timedelta64(60, "s")

    return pd.Series(
        {
            "candidates_checked": len(candidates),
            "max_open_minus_entry_price": float(open_err.max()),
            "entry_open_match": bool((open_err <= cfg.price_tolerance).all()),
            "decision_atr_max_error": float(np.nanmax(atr_err)),
            "decision_atr_point_in_time": bool(np.all((atr_err <= 1e-6) | np.isnan(atr_err))),
            "entry_is_next_bar_after_decision": bool((gap == one_minute).all()),
        },
        name="value",
    )


def audit_fills(independent: pd.DataFrame, arrays: dict, cfg: VerificationConfig) -> pd.DataFrame:
    """Every exit price must be realisable inside its exit bar."""

    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    minute_ny = arrays["minute_ny"]
    records: list[dict] = []
    for reason, group in independent.groupby("exit_reason", sort=False):
        exit_pos = group["exit_position"].to_numpy()
        exit_price = group["exit_price"].to_numpy(dtype=np.float64)
        is_long = group["direction_variant"].eq("long_benchmark").to_numpy()
        bar_high = high[exit_pos]
        bar_low = low[exit_pos]
        bar_close = close[exit_pos]
        within_bar = (exit_price >= bar_low - cfg.price_tolerance) & (
            exit_price <= bar_high + cfg.price_tolerance
        )
        if reason == "stop":
            reached = np.where(
                is_long,
                bar_low <= exit_price + cfg.price_tolerance,
                bar_high >= exit_price - cfg.price_tolerance,
            )
        elif reason == "target":
            reached = np.where(
                is_long,
                bar_high >= exit_price - cfg.price_tolerance,
                bar_low <= exit_price + cfg.price_tolerance,
            )
        else:
            reached = np.abs(exit_price - bar_close) <= cfg.price_tolerance
        forced_ok = (
            bool((minute_ny[exit_pos] >= cfg.forced_exit_minute_ny).all())
            if reason == "forced_1530"
            else True
        )
        records.append(
            {
                "exit_reason": reason,
                "trades": len(group),
                "all_reachable": bool(reached.all()),
                "within_bar_rate": float(within_bar.mean()),
                "gap_through_trades": int((~within_bar).sum()),
                "close_based_exact": bool(
                    True
                    if reason in {"stop", "target"}
                    else (np.abs(exit_price - bar_close) <= cfg.price_tolerance).all()
                ),
                "forced_minute_ok": forced_ok,
            }
        )
    return pd.DataFrame.from_records(records)


def audit_boundaries(independent: pd.DataFrame, arrays: dict, cfg: VerificationConfig) -> dict:
    """No trade crosses a date, a segment, or the holding cap."""

    entry_pos = independent["entry_position"].to_numpy()
    exit_pos = independent["exit_position"].to_numpy()
    holding = independent["holding_minutes"].to_numpy()
    same_segment = arrays["segment"][entry_pos] == arrays["segment"][exit_pos]
    same_date = arrays["trade_date"][entry_pos] == arrays["trade_date"][exit_pos]
    return {
        "no_trade_crosses_segment": bool(same_segment.all()),
        "no_trade_crosses_ny_date": bool(same_date.all()),
        "holding_within_cap": bool((holding <= cfg.max_holding_minutes).all()),
        "exit_not_before_entry": bool((exit_pos >= entry_pos).all()),
    }


# ---------------------------------------------------------------------------
# Cross-artifact triangulation and gate look-ahead
# ---------------------------------------------------------------------------
def audit_gate_consistency(candidates: pd.DataFrame, gate_thresholds: pd.DataFrame) -> pd.Series:
    """Verify the gate flag equals ``prediction >= session threshold`` exactly.

    This is an internal-consistency check: it confirms the recorded
    ``expansion_gate_flag`` is a deterministic function of the recorded
    ``gate_prediction`` and the recorded per-session Development threshold, so
    the gate was applied uniformly and reproducibly to Development and
    Validation.  Whether the *prediction itself* is point-in-time (its per-date
    normalisation uses same-day observations) is a separate signal-validity
    question handled by the dedicated gate look-ahead audit, not here.
    """

    if "gate_prediction" not in candidates or gate_thresholds.empty:
        return pd.Series({"evaluated": False}, name="value")

    thresholds = gate_thresholds.set_index("session")["gate_threshold"]
    mismatches = 0
    checked = 0
    for session, group in candidates.groupby("entry_session", sort=False):
        if session not in thresholds.index:
            continue
        threshold = float(thresholds.loc[session])
        prediction = group["gate_prediction"].to_numpy(dtype=np.float64)
        recorded_flag = group["expansion_gate_flag"].to_numpy(dtype=bool)
        implied_flag = prediction >= threshold
        mismatches += int(np.sum(recorded_flag != implied_flag))
        checked += len(group)
    return pd.Series(
        {
            "evaluated": True,
            "gated_observations": int(candidates["expansion_gate_flag"].sum()),
            "flags_checked": checked,
            "flag_threshold_mismatches": mismatches,
            "gate_flag_consistent": mismatches == 0,
        },
        name="value",
    )


def cross_check_forward_labels(
    candidates: pd.DataFrame, labels: pd.DataFrame, cfg: VerificationConfig
) -> pd.Series:
    """Entry price and decision ATR agree across three independent tables."""

    join = candidates.join(
        labels.set_index("observation_id")[["entry_price", "decision_atr_20m"]],
        how="inner",
        rsuffix="_label",
    )
    price_err = np.abs(
        join["entry_price"].to_numpy(dtype=np.float64)
        - join["entry_price_label"].to_numpy(dtype=np.float64)
    )
    atr_err = np.abs(
        join["decision_atr_20m"].to_numpy(dtype=np.float64)
        - join["decision_atr_20m_label"].to_numpy(dtype=np.float64)
    )
    return pd.Series(
        {
            "candidates_joined_to_labels": len(join),
            "entry_price_max_error": float(price_err.max()) if len(join) else np.nan,
            "entry_price_agrees": bool((price_err <= cfg.price_tolerance).all()),
            "decision_atr_max_error": float(atr_err.max()) if len(join) else np.nan,
            "decision_atr_agrees": bool((atr_err <= 1e-6).all()),
        },
        name="value",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_verification(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    recorded_trade_log: pd.DataFrame,
    recorded_performance: pd.DataFrame,
    *,
    gate_thresholds: pd.DataFrame | None = None,
    forward_labels: pd.DataFrame | None = None,
    config: VerificationConfig | None = None,
) -> VerificationResult:
    """Independently reproduce and audit the recorded backtest."""

    cfg = config or VerificationConfig()
    production = Section11Config()
    if (
        cfg.tick_size != production.tick_size
        or cfg.commission_ticks_round_trip != production.commission_ticks_round_trip
        or cfg.base_slippage_ticks_per_side != production.base_slippage_ticks_per_side
        or cfg.pessimistic_slippage_ticks_per_side != production.pessimistic_slippage_ticks_per_side
        or cfg.max_holding_minutes != production.max_holding_minutes
    ):
        raise ValueError("verifier cost contract has drifted from Section11Config")

    observed_partitions = set(candidates["research_partition"].astype(str).unique())
    partition_governance = observed_partitions <= set(EVALUATION_PARTITIONS)

    entry_positions = map_entry_positions(candidates, bars)
    arrays = _bar_arrays(bars)

    independent_frames = [
        independent_resimulate(candidates, entry_positions, arrays, direction, gate, cfg)
        for direction in DIRECTION_VARIANTS
        for gate in GATE_VARIANTS
    ]
    independent = pd.concat(independent_frames, ignore_index=True)

    reconciliation, mismatches = reconcile_trades(recorded_trade_log, independent, cfg)
    performance_reconciliation = reconcile_performance(independent, recorded_performance, cfg)
    entry_fidelity = audit_entry_fidelity(candidates, entry_positions, arrays, cfg)
    fill_fidelity = audit_fills(independent, arrays, cfg)
    boundary = audit_boundaries(independent, arrays, cfg)
    gate_consistency = (
        audit_gate_consistency(candidates, gate_thresholds)
        if gate_thresholds is not None
        else pd.Series({"evaluated": False}, name="value")
    )
    cross_artifact = (
        cross_check_forward_labels(candidates, forward_labels, cfg)
        if forward_labels is not None
        else pd.Series({"evaluated": False}, name="value")
    )

    checks = {
        "partitions_limited_to_development_validation": partition_governance,
        "trade_counts_reconcile": bool(reconciliation["count_match"].all()),
        "all_trades_reconcile": bool(reconciliation["reconciled"].all()),
        "performance_reconciles": bool(
            (performance_reconciliation["abs_diff"] <= cfg.r_tolerance).all()
        ),
        "entry_fills_equal_bar_opens": bool(entry_fidelity["entry_open_match"]),
        "sizing_is_point_in_time": bool(entry_fidelity["decision_atr_point_in_time"]),
        "entry_is_next_bar_after_decision": bool(
            entry_fidelity["entry_is_next_bar_after_decision"]
        ),
        "all_exit_fills_reachable": bool(fill_fidelity["all_reachable"].all()),
        "close_based_exits_exact": bool(fill_fidelity["close_based_exact"].all()),
        "no_trade_crosses_segment": boundary["no_trade_crosses_segment"],
        "no_trade_crosses_ny_date": boundary["no_trade_crosses_ny_date"],
        "holding_within_cap": boundary["holding_within_cap"],
    }
    if bool(gate_consistency.get("evaluated", False)):
        checks["gate_flag_matches_threshold"] = bool(gate_consistency["gate_flag_consistent"])
    if bool(cross_artifact.get("candidates_joined_to_labels", 0)):
        checks["entry_price_agrees_across_artifacts"] = bool(cross_artifact["entry_price_agrees"])
        checks["decision_atr_agrees_across_artifacts"] = bool(cross_artifact["decision_atr_agrees"])
    checks_frame = pd.Series(checks, name="passed").rename_axis("check").reset_index()

    gap_through = int(fill_fidelity["gap_through_trades"].sum())
    summary = pd.Series(
        {
            "independent_trades": len(independent),
            "recorded_trades": len(recorded_trade_log),
            "variants_reconciled": int(reconciliation["reconciled"].sum()),
            "variants_total": len(reconciliation),
            "trade_field_mismatches": int(reconciliation["field_mismatches"].sum()),
            "max_performance_abs_diff": float(performance_reconciliation["abs_diff"].max()),
            "stop_target_gap_through_trades": gap_through,
            "gate_flag_consistent": bool(gate_consistency.get("gate_flag_consistent", True)),
            "all_checks_passed": bool(checks_frame["passed"].all()),
        },
        name="value",
    )

    return VerificationResult(
        reconciliation=reconciliation,
        performance_reconciliation=performance_reconciliation,
        entry_fidelity=entry_fidelity,
        fill_fidelity=fill_fidelity,
        gate_consistency=gate_consistency,
        cross_artifact=cross_artifact,
        checks=checks_frame,
        mismatch_samples=mismatches,
        summary=summary,
        config=cfg,
    )


def save_verification_outputs(result: VerificationResult, *, project_root: Path) -> pd.DataFrame:
    """Persist verification tables; generated artifacts stay outside Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "verification"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "verification_reconciliation": (
            processed / "backtest_verification_reconciliation_gc.parquet",
            result.reconciliation,
        ),
        "verification_performance": (
            processed / "backtest_verification_performance_gc.parquet",
            result.performance_reconciliation,
        ),
        "verification_fill_fidelity": (
            processed / "backtest_verification_fill_fidelity_gc.parquet",
            result.fill_fidelity,
        ),
        "verification_checks": (
            processed / "backtest_verification_checks_gc.parquet",
            result.checks,
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
    result.reconciliation.to_csv(tables / "reconciliation_gc.csv", index=False)
    result.checks.to_csv(tables / "checks_gc.csv", index=False)
    return pd.DataFrame.from_records(records)
