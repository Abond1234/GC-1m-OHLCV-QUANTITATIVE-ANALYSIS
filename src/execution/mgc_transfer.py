"""GC-to-MGC transfer validation (PRD FR-09, gate G5).

Quantifies whether decisions researched on GC remain executable on MGC:
synchronized-minute coverage, close-price basis, bar-range containment of GC
decision prices, and MGC liquidity at the exact decision minutes Branch A
research would trade (True POI retest bars).

Provisional G5 acceptance thresholds are declared in
:class:`MgcTransferConfig` before any statistic is computed.  The PRD requires
numeric gate thresholds to be declared before observing decisive results while
noting the exact numbers await lead approval, so the verdict this module emits
is explicitly ``G5_PROVISIONAL_PASS`` / ``G5_PROVISIONAL_FAIL`` - an
engineering acceptance readout, not a final gate decision.

Honest measurement boundary: the repository carries one-minute OHLCV only.
Bid/ask spread and queue-depth cannot be observed from bars; this module
reports bar-range and zero-volume proxies and states the limitation instead of
synthesizing spread numbers it cannot support.  A definitive spread/slippage
study needs order-book (MBP) data and belongs to the Rithmic forward-test
phase, where decision-to-fill telemetry is recorded directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass(frozen=True)
class MgcTransferConfig:
    """Frozen measurement definitions and provisional G5 thresholds."""

    tick_size: float = 0.10
    gc_tick_value_usd: float = 10.0
    mgc_tick_value_usd: float = 1.0
    min_minute_coverage: float = 0.98
    min_decision_coverage: float = 0.99
    max_median_abs_basis_ticks: float = 1.0
    max_p95_abs_basis_ticks: float = 3.0
    min_entry_containment: float = 0.95
    max_decision_zero_volume_rate: float = 0.02
    min_median_decision_volume: float = 10.0


@dataclass
class MgcTransferResult:
    """Container for every FR-09 output object."""

    coverage_by_year: pd.DataFrame
    basis_by_year: pd.DataFrame
    basis_by_session: pd.DataFrame
    decision_by_year: pd.DataFrame
    decision_summary: pd.Series
    threshold_checks: pd.DataFrame
    verdict: str
    validation_checks: pd.DataFrame
    config: MgcTransferConfig = field(default_factory=MgcTransferConfig)


def load_product_minutes(research_bars_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Narrow per-product minute frames from the trusted research table."""

    columns = [
        "ts_event_utc",
        "product",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "session_label",
        "trade_date_ny",
    ]
    table = pq.read_table(research_bars_path, columns=columns).to_pandas(ignore_metadata=True)
    table["ts_event_utc"] = pd.to_datetime(table["ts_event_utc"])
    gc = table.loc[table["product"] == "GC"].drop(columns="product").reset_index(drop=True)
    mgc = table.loc[table["product"] == "MGC"].drop(columns="product").reset_index(drop=True)
    if gc.empty or mgc.empty:
        raise ValueError("research bars must contain both GC and MGC products")
    return gc, mgc


def load_poi_decision_bars(project_root: Path) -> pd.DataFrame:
    """Unique True POI retest decision bars with the GC first-contact price."""

    processed = project_root / "data" / "processed"
    context = pq.read_table(
        processed / "section7_true_poi_context_frame_gc.parquet",
        columns=["true_retest_id", "retest_ts_event_utc", "research_partition"],
    ).to_pandas(ignore_metadata=True)
    labels = pq.read_table(
        processed / "section7_true_poi_outcome_labels_gc.parquet",
        columns=["true_retest_id", "feat_first_contact_edge_price"],
    ).to_pandas(ignore_metadata=True)
    labels = labels.drop_duplicates("true_retest_id")
    decisions = context.drop_duplicates("true_retest_id").merge(
        labels, on="true_retest_id", how="left", validate="one_to_one"
    )
    decisions["retest_ts_event_utc"] = pd.to_datetime(decisions["retest_ts_event_utc"])
    return decisions


def build_minute_alignment(
    gc: pd.DataFrame, mgc: pd.DataFrame, config: MgcTransferConfig | None = None
) -> pd.DataFrame:
    """Left-align every GC minute with the synchronized MGC minute."""

    cfg = config or MgcTransferConfig()
    merged = gc.merge(
        mgc[["ts_event_utc", "open", "high", "low", "close", "volume"]],
        on="ts_event_utc",
        how="left",
        suffixes=("_gc", "_mgc"),
        validate="one_to_one",
    )
    merged["mgc_available"] = merged["close_mgc"].notna()
    # Both products quote in 0.10 increments, so tick differences are exact
    # integers; rounding removes float-division noise only.
    merged["close_basis_ticks"] = np.round(
        (merged["close_mgc"] - merged["close_gc"]) / cfg.tick_size, 6
    )
    merged["gc_close_inside_mgc_range"] = (
        merged["mgc_available"]
        & (merged["close_gc"] >= merged["low_mgc"] - cfg.tick_size / 2)
        & (merged["close_gc"] <= merged["high_mgc"] + cfg.tick_size / 2)
    )
    merged["year"] = merged["ts_event_utc"].dt.year
    return merged


def _basis_stats(group: pd.DataFrame) -> dict:
    basis = group.loc[group["mgc_available"], "close_basis_ticks"].to_numpy(dtype=np.float64)
    if len(basis) == 0:
        return {
            "matched_minutes": 0,
            "median_basis_ticks": np.nan,
            "median_abs_basis_ticks": np.nan,
            "p95_abs_basis_ticks": np.nan,
            "max_abs_basis_ticks": np.nan,
            "gc_close_containment": np.nan,
        }
    return {
        "matched_minutes": len(basis),
        "median_basis_ticks": float(np.median(basis)),
        "median_abs_basis_ticks": float(np.median(np.abs(basis))),
        "p95_abs_basis_ticks": float(np.quantile(np.abs(basis), 0.95)),
        "max_abs_basis_ticks": float(np.abs(basis).max()),
        "gc_close_containment": float(
            group.loc[group["mgc_available"], "gc_close_inside_mgc_range"].mean()
        ),
    }


def build_transfer_result(
    aligned: pd.DataFrame,
    decisions: pd.DataFrame,
    config: MgcTransferConfig | None = None,
) -> MgcTransferResult:
    """Coverage, basis, liquidity, and decision-bar executability with checks."""

    cfg = config or MgcTransferConfig()

    coverage_records = []
    for year, group in aligned.groupby("year", sort=True):
        coverage_records.append(
            {
                "year": int(year),
                "gc_minutes": len(group),
                "mgc_matched_minutes": int(group["mgc_available"].sum()),
                "coverage": float(group["mgc_available"].mean()),
                "mgc_zero_volume_rate": float(
                    (group.loc[group["mgc_available"], "volume_mgc"] == 0).mean()
                ),
            }
        )
    coverage_by_year = pd.DataFrame.from_records(coverage_records)

    basis_by_year = pd.DataFrame.from_records(
        [{"year": int(year), **_basis_stats(g)} for year, g in aligned.groupby("year", sort=True)]
    )
    basis_by_session = pd.DataFrame.from_records(
        [
            {"session": str(session), **_basis_stats(g)}
            for session, g in aligned.groupby("session_label", sort=True)
        ]
    )

    keyed = aligned.set_index("ts_event_utc")
    decision_rows = decisions.dropna(subset=["retest_ts_event_utc"]).copy()
    joined = decision_rows.join(
        keyed[
            [
                "mgc_available",
                "close_basis_ticks",
                "high_mgc",
                "low_mgc",
                "volume_mgc",
            ]
        ],
        on="retest_ts_event_utc",
        how="left",
    )
    matched = joined["mgc_available"].fillna(False).astype(bool)
    entry = pd.to_numeric(joined["feat_first_contact_edge_price"], errors="coerce")
    containable = matched & entry.notna()
    contained = (
        containable
        & (entry >= joined["low_mgc"] - cfg.tick_size / 2)
        & (entry <= joined["high_mgc"] + cfg.tick_size / 2)
    )

    decision_year = joined["retest_ts_event_utc"].dt.year
    decision_year_records = []
    for year in sorted(decision_year.unique()):
        in_year = decision_year == year
        year_containable = containable & in_year
        year_basis = joined.loc[matched & in_year, "close_basis_ticks"].to_numpy(dtype=np.float64)
        decision_year_records.append(
            {
                "year": int(year),
                "decision_bars": int(in_year.sum()),
                "decision_coverage": float(matched[in_year].mean()),
                "entry_price_containment": (
                    float((contained & in_year).sum() / year_containable.sum())
                    if year_containable.any()
                    else np.nan
                ),
                "median_abs_basis_ticks": (
                    float(np.median(np.abs(year_basis))) if len(year_basis) else np.nan
                ),
                "p95_abs_basis_ticks": (
                    float(np.quantile(np.abs(year_basis), 0.95)) if len(year_basis) else np.nan
                ),
            }
        )
    decision_by_year = pd.DataFrame.from_records(decision_year_records)
    decision_volume = joined.loc[matched, "volume_mgc"].to_numpy(dtype=np.float64)
    decision_basis = joined.loc[matched, "close_basis_ticks"].to_numpy(dtype=np.float64)
    decision_summary = pd.Series(
        {
            "decision_bars": len(joined),
            "mgc_synchronized": int(matched.sum()),
            "decision_coverage": float(matched.mean()),
            "entry_price_containment": (
                float(contained.sum() / containable.sum()) if containable.any() else np.nan
            ),
            "median_abs_decision_basis_ticks": (
                float(np.median(np.abs(decision_basis))) if len(decision_basis) else np.nan
            ),
            "p95_abs_decision_basis_ticks": (
                float(np.quantile(np.abs(decision_basis), 0.95)) if len(decision_basis) else np.nan
            ),
            "median_decision_mgc_volume": (
                float(np.median(decision_volume)) if len(decision_volume) else np.nan
            ),
            "decision_zero_volume_rate": (
                float((decision_volume == 0).mean()) if len(decision_volume) else np.nan
            ),
        },
        name="value",
    )

    overall_coverage = float(aligned["mgc_available"].mean())
    checks = [
        ("minute_coverage", overall_coverage, ">=", cfg.min_minute_coverage),
        (
            "decision_coverage",
            float(decision_summary["decision_coverage"]),
            ">=",
            cfg.min_decision_coverage,
        ),
        (
            "median_abs_decision_basis_ticks",
            float(decision_summary["median_abs_decision_basis_ticks"]),
            "<=",
            cfg.max_median_abs_basis_ticks,
        ),
        (
            "p95_abs_decision_basis_ticks",
            float(decision_summary["p95_abs_decision_basis_ticks"]),
            "<=",
            cfg.max_p95_abs_basis_ticks,
        ),
        (
            "entry_price_containment",
            float(decision_summary["entry_price_containment"]),
            ">=",
            cfg.min_entry_containment,
        ),
        (
            "decision_zero_volume_rate",
            float(decision_summary["decision_zero_volume_rate"]),
            "<=",
            cfg.max_decision_zero_volume_rate,
        ),
        (
            "median_decision_mgc_volume",
            float(decision_summary["median_decision_mgc_volume"]),
            ">=",
            cfg.min_median_decision_volume,
        ),
    ]
    threshold_records = []
    for name, observed, direction, bound in checks:
        passed = bool(
            np.isfinite(observed)
            and (observed >= bound if direction == ">=" else observed <= bound)
        )
        threshold_records.append(
            {
                "check": name,
                "observed": observed,
                "direction": direction,
                "provisional_threshold": bound,
                "passed": passed,
            }
        )
    threshold_checks = pd.DataFrame.from_records(threshold_records)
    verdict = (
        "G5_PROVISIONAL_PASS" if bool(threshold_checks["passed"].all()) else "G5_PROVISIONAL_FAIL"
    )

    validation = {
        "aligned_timestamps_unique": bool(aligned["ts_event_utc"].is_unique),
        "decision_join_unique": bool(joined["true_retest_id"].is_unique),
        "basis_defined_only_when_matched": bool(
            aligned.loc[~aligned["mgc_available"], "close_basis_ticks"].isna().all()
        ),
        "coverage_within_unit_interval": bool(coverage_by_year["coverage"].between(0.0, 1.0).all()),
        "threshold_checks_complete": len(threshold_checks) == 7,
    }
    validation_checks = pd.Series(validation, name="passed").rename_axis("check").reset_index()

    return MgcTransferResult(
        coverage_by_year=coverage_by_year,
        basis_by_year=basis_by_year,
        basis_by_session=basis_by_session,
        decision_by_year=decision_by_year,
        decision_summary=decision_summary,
        threshold_checks=threshold_checks,
        verdict=verdict,
        validation_checks=validation_checks,
        config=cfg,
    )


def save_transfer_outputs(result: MgcTransferResult, *, project_root: Path) -> pd.DataFrame:
    """Persist FR-09 outputs; parquet artifacts stay excluded from Git."""

    processed = project_root / "data" / "processed" / "execution"
    tables = project_root / "reports" / "execution" / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mgc_transfer_coverage_by_year": (
            processed / "mgc_transfer_coverage_by_year.parquet",
            result.coverage_by_year,
        ),
        "mgc_transfer_basis_by_year": (
            processed / "mgc_transfer_basis_by_year.parquet",
            result.basis_by_year,
        ),
        "mgc_transfer_basis_by_session": (
            processed / "mgc_transfer_basis_by_session.parquet",
            result.basis_by_session,
        ),
        "mgc_transfer_decision_by_year": (
            processed / "mgc_transfer_decision_by_year.parquet",
            result.decision_by_year,
        ),
        "mgc_transfer_threshold_checks": (
            processed / "mgc_transfer_threshold_checks.parquet",
            result.threshold_checks,
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
    result.threshold_checks.to_csv(tables / "mgc_transfer_threshold_checks.csv", index=False)
    result.coverage_by_year.to_csv(tables / "mgc_transfer_coverage_by_year.csv", index=False)
    result.basis_by_session.to_csv(tables / "mgc_transfer_basis_by_session.csv", index=False)
    result.decision_by_year.to_csv(tables / "mgc_transfer_decision_by_year.csv", index=False)
    return pd.DataFrame.from_records(records)
