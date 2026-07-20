"""Section 10 statistical signal construction for the independent branch.

Section 7 produced no directional advancer, so this section does not invent a
directional signal.  It formalizes the honest candidate family the evidence
supports for a sequential test:

- Direction variants are declared **benchmarks** (`long_benchmark`,
  `short_benchmark`), not evidence-based signals.
- The only evidence-based component is the opportunity gate: the Section 9
  frozen per-session ridge model of 60-minute ATR-relative future range,
  thresholded at its Development 80th percentile (fitted on Development,
  applied unchanged to Validation).
- Stops are volatility-based (a declared multiple of the decision-bar ATR
  with tick clamps), targets are a declared R multiple, holding time is
  capped, and every trade obeys the locked session/noon/15:30 rules already
  enforced upstream by the eligible-observation construction.
- Risk per trade is one R; contract-count sizing is deferred to the
  execution phase and recorded as such.

The output is a candidate table for the Section 11 sequential backtest, plus
frozen rule metadata.  Governance mirrors earlier sections: Development and
Validation only, with Final-test rows rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .feature_evaluation import EVALUATION_PARTITIONS, EVALUATION_SESSIONS

SECTION10_RANDOM_SEED = 20260723

DIRECTION_VARIANTS = ("long_benchmark", "short_benchmark")
GATE_VARIANTS = ("ungated", "expansion_gated")


@dataclass(frozen=True)
class Section10Config:
    """Frozen Section 10 candidate-rule contract."""

    random_seed: int = SECTION10_RANDOM_SEED
    gate_model_horizon_minutes: int = 60
    gate_development_percentile: float = 0.80
    stop_atr_multiple: float = 1.5
    min_stop_ticks: float = 10.0
    max_stop_ticks: float = 100.0
    target_r_multiple: float = 2.0
    max_holding_minutes: int = 120
    tick_size: float = 0.10
    risk_per_trade_r: float = 1.0
    contract_sizing_note: str = (
        "one R of risk per trade; contract-count and dollar sizing are an "
        "execution-phase concern and are intentionally not modelled here"
    )


@dataclass
class SignalBuildResult:
    """Container for the Section 10 outputs."""

    candidates: pd.DataFrame
    gate_thresholds: pd.DataFrame
    rule_specification: pd.Series
    validation_checks: pd.DataFrame
    summary: pd.Series
    config: Section10Config = field(default_factory=Section10Config)


def load_entry_fields(path: Path) -> pd.DataFrame:
    """Entry timestamp, price, and decision ATR for every observation."""

    columns = [
        "observation_id",
        "entry_timestamp_utc",
        "entry_timestamp_ny",
        "trade_date_ny",
        "entry_session",
        "research_partition",
        "entry_price",
        "decision_atr_20m",
    ]
    table = pq.read_table(path, columns=columns)
    return table.to_pandas(ignore_metadata=True).set_index("observation_id")


def build_signal_candidates(
    frame: pd.DataFrame,
    entry_fields: pd.DataFrame,
    ridge_coefficients: pd.DataFrame,
    config: Section10Config | None = None,
) -> SignalBuildResult:
    """Construct the declared candidate family from frozen components."""

    cfg = config or Section10Config()
    observed_partitions = set(frame["research_partition"].astype(str).unique())
    if not observed_partitions.issubset(set(EVALUATION_PARTITIONS)):
        raise ValueError(f"signal frame contains locked partitions: {observed_partitions}")

    coefficients = ridge_coefficients.loc[
        ridge_coefficients["horizon_minutes"].eq(cfg.gate_model_horizon_minutes)
    ]
    if coefficients.empty:
        raise ValueError("ridge coefficients for the gate horizon are missing")

    entries = entry_fields.reindex(frame.index)
    if entries["entry_price"].isna().any():
        raise ValueError("entry fields do not cover every signal observation")

    # ---- Section 9 frozen gate: per-session ridge prediction on rank-z -----
    session_values = frame["entry_session"].astype(str).to_numpy()
    prediction = np.full(len(frame), np.nan)
    for session in EVALUATION_SESSIONS:
        session_beta = coefficients.loc[coefficients["session"].eq(session)]
        features = session_beta["feature_name"].tolist()
        beta = session_beta["coefficient"].to_numpy(dtype=np.float64)
        mask = session_values == session
        sub = frame.loc[mask, features + ["trade_date_ny"]]
        date_codes, _ = pd.factorize(sub["trade_date_ny"], sort=True)
        grouped = sub[features].groupby(date_codes, sort=False)
        ranks = grouped.rank(method="average")
        mean = ranks.groupby(date_codes, sort=False).transform("mean")
        std = ranks.groupby(date_codes, sort=False).transform("std")
        with np.errstate(invalid="ignore", divide="ignore"):
            z = ((ranks - mean) / std).replace([np.inf, -np.inf], np.nan)
        prediction[mask] = z.to_numpy(dtype=np.float64) @ beta

    development = frame["research_partition"].eq("Development").to_numpy()
    gate_records = []
    gate_flag = np.zeros(len(frame), dtype=bool)
    for session in EVALUATION_SESSIONS:
        mask = session_values == session
        fit = prediction[mask & development]
        threshold = float(np.nanquantile(fit, cfg.gate_development_percentile))
        gate_flag[mask] = prediction[mask] >= threshold
        gate_records.append(
            {
                "session": session,
                "gate_development_percentile": cfg.gate_development_percentile,
                "gate_threshold": threshold,
                "development_gate_rate": float(np.nanmean(prediction[mask & development] >= threshold)),
                "validation_gate_rate": float(
                    np.nanmean(prediction[mask & ~development] >= threshold)
                ),
            }
        )
    gate_thresholds = pd.DataFrame.from_records(gate_records)

    # ---- volatility stops with tick clamps ---------------------------------
    atr = entries["decision_atr_20m"].to_numpy(dtype=np.float64)
    raw_stop_points = cfg.stop_atr_multiple * atr
    stop_points = np.clip(
        raw_stop_points, cfg.min_stop_ticks * cfg.tick_size, cfg.max_stop_ticks * cfg.tick_size
    )
    clamped_fraction = float(np.mean(raw_stop_points != stop_points))

    candidates = pd.DataFrame(
        {
            "entry_timestamp_utc": entries["entry_timestamp_utc"].to_numpy(),
            "entry_timestamp_ny": entries["entry_timestamp_ny"].to_numpy(),
            "trade_date_ny": pd.to_datetime(frame["trade_date_ny"]).to_numpy(),
            "entry_session": session_values,
            "research_partition": frame["research_partition"].astype(str).to_numpy(),
            "entry_price": entries["entry_price"].to_numpy(dtype=np.float64),
            "decision_atr_20m": atr,
            "gate_prediction": prediction,
            "expansion_gate_flag": gate_flag,
            "stop_points": stop_points,
            "target_points": cfg.target_r_multiple * stop_points,
        },
        index=frame.index,
    )
    valid = np.isfinite(prediction) & np.isfinite(atr) & (stop_points > 0)
    candidates = candidates.loc[valid]

    rule_specification = pd.Series(
        {
            "direction_variants": " / ".join(DIRECTION_VARIANTS)
            + " (declared benchmarks; Section 7 approved no directional signal)",
            "gate_variants": " / ".join(GATE_VARIANTS),
            "gate_definition": (
                f"Section 9 frozen per-session ridge model of {cfg.gate_model_horizon_minutes}m "
                f"ATR-relative future range, >= Development p{int(cfg.gate_development_percentile * 100)}"
            ),
            "stop_rule": (
                f"{cfg.stop_atr_multiple} x decision ATR(20m), clamped to "
                f"[{cfg.min_stop_ticks:.0f}, {cfg.max_stop_ticks:.0f}] ticks"
            ),
            "target_rule": f"{cfg.target_r_multiple}R",
            "max_holding_minutes": cfg.max_holding_minutes,
            "position_rule": "one position at a time; conservative stop-first on ambiguous bars",
            "risk_per_trade": f"{cfg.risk_per_trade_r}R ({cfg.contract_sizing_note})",
            "session_rules": "inherited: London [03:00,06:00), New York [07:00,12:00), flat by 15:30",
        },
        name="frozen_rule",
    )

    checks = {
        "partitions_limited_to_development_validation": observed_partitions
        <= set(EVALUATION_PARTITIONS),
        "gate_fitted_on_development_only": True,
        "development_gate_rate_near_declared": all(
            abs(r["development_gate_rate"] - (1.0 - cfg.gate_development_percentile)) < 0.01
            for r in gate_records
        ),
        "stops_within_tick_clamps": bool(
            (
                (candidates["stop_points"] / cfg.tick_size >= cfg.min_stop_ticks - 1e-9)
                & (candidates["stop_points"] / cfg.tick_size <= cfg.max_stop_ticks + 1e-9)
            ).all()
        ),
        "targets_are_declared_r_multiple": bool(
            np.allclose(
                candidates["target_points"],
                cfg.target_r_multiple * candidates["stop_points"],
            )
        ),
        "entry_prices_positive": bool((candidates["entry_price"] > 0).all()),
        "candidate_sessions_expected": set(candidates["entry_session"].unique())
        <= set(EVALUATION_SESSIONS),
    }
    validation_checks = pd.Series(checks, name="passed").rename_axis("check").reset_index()

    partition_counts = candidates["research_partition"].value_counts()
    summary = pd.Series(
        {
            "candidate_observations": len(candidates),
            "development_candidates": int(partition_counts.get("Development", 0)),
            "validation_candidates": int(partition_counts.get("Validation", 0)),
            "gate_pass_observations": int(candidates["expansion_gate_flag"].sum()),
            "stop_clamped_fraction": clamped_fraction,
            "median_stop_ticks": float(
                (candidates["stop_points"] / cfg.tick_size).median()
            ),
            "direction_variants": len(DIRECTION_VARIANTS),
            "gate_variants": len(GATE_VARIANTS),
        },
        name="value",
    )

    return SignalBuildResult(
        candidates=candidates,
        gate_thresholds=gate_thresholds,
        rule_specification=rule_specification,
        validation_checks=validation_checks,
        summary=summary,
        config=cfg,
    )


def save_signal_outputs(result: SignalBuildResult, *, project_root: Path) -> pd.DataFrame:
    """Persist Section 10 outputs; generated artifacts stay excluded from Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    processed.mkdir(parents=True, exist_ok=True)
    outputs = {
        "signal_candidates": (
            processed / "signal_candidates_gc.parquet",
            result.candidates.reset_index(),
        ),
        "signal_gate_thresholds": (
            processed / "signal_gate_thresholds_gc.parquet",
            result.gate_thresholds,
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
    return pd.DataFrame.from_records(records)
