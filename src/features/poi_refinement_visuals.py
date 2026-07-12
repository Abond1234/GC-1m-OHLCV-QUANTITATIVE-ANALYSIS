"""Visual and manifest validation for Section 6C POI refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.poi_selection_refinement import HAND_LABELLED_REGRESSION_CASES


HAND_SIGNAL_COLUMNS = [
    "signal_id",
    "poi_id",
    "direction",
    "swing_n",
    "break_mode",
    "poi_middle_bar_id",
    "poi_confirm_bar_id",
    "poi_activation_bar_id",
    "break_bar_id",
    "retest_bar_id",
    "retest_ts_event_ny",
    "entry_price",
]

VISUAL_BAR_COLUMNS = [
    "ts_event_utc",
    "ts_event_ny",
    "trade_date_ny",
    "product",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "continuous_segment_id",
]


def build_hand_labelled_regression_visuals(
    *,
    audit: pd.DataFrame,
    legacy_signal_frame: pd.DataFrame,
    research_bars: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot all 19 supplied cases and write expected-vs-actual manifests."""

    out_dir = Path(output_dir)
    hand_dir = out_dir / "hand_labelled"
    hand_dir.mkdir(parents=True, exist_ok=True)
    bars = _prepare_bars(research_bars)
    manifest_records: list[dict[str, Any]] = []
    ohlc_records: list[dict[str, Any]] = []

    for fixture in HAND_LABELLED_REGRESSION_CASES:
        signal_id = fixture["signal_id"]
        matching_signal = legacy_signal_frame.loc[legacy_signal_frame["signal_id"].eq(signal_id)]
        if len(matching_signal) != 1:
            manifest_records.append(
                {
                    **fixture,
                    "actual_classification": "missing_signal",
                    "actual_formation_validity": False,
                    "actual_rejection_reason": "missing_signal",
                    "fvg_size_ticks": np.nan,
                    "fvg_ge_3tick_flag": False,
                    "fvg_ge_4tick_flag": False,
                    "fvg_ge_5tick_flag": False,
                    "pass_fail": "FAIL",
                    "reviewer_note": f"Expected exactly one legacy signal row; found {len(matching_signal)}.",
                    "chart_path": pd.NA,
                }
            )
            continue
        signal = matching_signal.iloc[0]
        matching_audit = audit.loc[
            audit["poi_variant_id"].astype("string").eq(str(signal["poi_id"]))
        ]
        if matching_audit.empty:
            # The formation itself is physical, so fall back to direction+B if
            # an older POI id cannot be mapped after a schema migration.
            matching_audit = audit.loc[
                audit["direction"].eq(signal["direction"])
                & audit["poi_middle_bar_id"].eq(signal["poi_middle_bar_id"])
            ]
        if matching_audit.empty:
            manifest_records.append(
                {
                    **fixture,
                    "actual_classification": "missing_audit_row",
                    "actual_formation_validity": False,
                    "actual_rejection_reason": "missing_audit_row",
                    "fvg_size_ticks": np.nan,
                    "fvg_ge_3tick_flag": False,
                    "fvg_ge_4tick_flag": False,
                    "fvg_ge_5tick_flag": False,
                    "pass_fail": "FAIL",
                    "reviewer_note": "No Section 6C audit row matched the legacy POI lineage.",
                    "chart_path": pd.NA,
                }
            )
            continue
        event = matching_audit.iloc[0]
        actual_classification = _actual_classification(event)
        actual_validity = bool(event["formation_valid_flag"])
        actual_reason = str(event["rejection_reason"])
        matches = (
            actual_classification == fixture["expected_classification"]
            and actual_validity == fixture["expected_formation_validity"]
            and actual_reason == fixture["expected_rejection_reason"]
        )
        note = _reviewer_note(event, fixture)
        chart_path = hand_dir / f"{signal_id}_{actual_classification}.png"
        plot_refined_poi_event(
            event=event,
            bars=bars,
            output_path=chart_path,
            display_signal_id=signal_id,
            legacy_signal=signal,
        )
        manifest_records.append(
            {
                "signal_id": signal_id,
                "expected_classification": fixture["expected_classification"],
                "actual_classification": actual_classification,
                "expected_formation_validity": fixture["expected_formation_validity"],
                "actual_formation_validity": actual_validity,
                "expected_rejection_reason": fixture["expected_rejection_reason"],
                "actual_rejection_reason": actual_reason,
                "fvg_size_ticks": int(event["fvg_size_ticks"]),
                "fvg_ge_3tick_flag": bool(event["fvg_ge_3tick_flag"]),
                "fvg_ge_4tick_flag": bool(event["fvg_ge_4tick_flag"]),
                "fvg_ge_5tick_flag": bool(event["fvg_ge_5tick_flag"]),
                "pass_fail": "PASS" if matches else "REVIEW",
                "reviewer_note": note,
                "chart_path": str(chart_path),
            }
        )
        ohlc_records.extend(_abc_ohlc_records(signal_id, event))

    manifest = pd.DataFrame.from_records(manifest_records)
    abc_ohlc = pd.DataFrame.from_records(ohlc_records)
    manifest.to_csv(out_dir / "hand_labelled_regression_manifest.csv", index=False)
    abc_ohlc.to_csv(out_dir / "hand_labelled_abc_ohlc.csv", index=False)
    return manifest, abc_ohlc


def build_random_stratified_refinement_audit(
    *,
    audit: pd.DataFrame,
    research_bars: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 6_703,
) -> pd.DataFrame:
    """Plot one deterministic random example from each required audit stratum."""

    out_dir = Path(output_dir)
    sample_dir = out_dir / "random_stratified"
    sample_dir.mkdir(parents=True, exist_ok=True)
    bars = _prepare_bars(research_bars)
    structural = audit.get("structural_swing_break_flag", pd.Series(False, index=audit.index)).fillna(False)
    local_only = audit.get("local_swing_only_flag", pd.Series(False, index=audit.index)).fillna(False)
    strata = [
        ("valid_case1_bullish", audit["final_poi_valid_flag"] & audit["poi_geometry_case"].eq("case_1_expanded") & audit["direction"].eq("bullish")),
        ("valid_case1_bearish", audit["final_poi_valid_flag"] & audit["poi_geometry_case"].eq("case_1_expanded") & audit["direction"].eq("bearish")),
        ("valid_case2_bullish", audit["final_poi_valid_flag"] & audit["poi_geometry_case"].eq("case_2_standard") & audit["direction"].eq("bullish")),
        ("valid_case2_bearish", audit["final_poi_valid_flag"] & audit["poi_geometry_case"].eq("case_2_standard") & audit["direction"].eq("bearish")),
        ("case3_reject", audit["rejection_reason"].eq("confirmation_close_inside_opening_gap")),
        ("no_fvg_reject", audit["rejection_reason"].eq("no_classic_fvg")),
        ("below_3_tick_reject", audit["rejection_reason"].eq("fvg_below_3_ticks")),
        ("exactly_3_ticks", audit["final_poi_valid_flag"] & audit["fvg_size_ticks"].eq(3)),
        ("exactly_4_ticks", audit["final_poi_valid_flag"] & audit["fvg_size_ticks"].eq(4)),
        ("exactly_5_ticks", audit["final_poi_valid_flag"] & audit["fvg_size_ticks"].eq(5)),
        ("above_5_ticks", audit["final_poi_valid_flag"] & audit["fvg_size_ticks"].gt(5)),
        ("structural", audit["final_poi_valid_flag"] & structural.astype(bool)),
        ("local_only", audit["final_poi_valid_flag"] & local_only.astype(bool)),
    ]
    rng = np.random.default_rng(random_state)
    records: list[dict[str, Any]] = []
    for name, mask in strata:
        candidates = audit.loc[mask].reset_index(drop=True)
        if candidates.empty:
            records.append(
                {
                    "stratum": name,
                    "status": "skipped_no_candidates",
                    "available_count": 0,
                    "audit_id": pd.NA,
                    "chart_path": pd.NA,
                }
            )
            continue
        pos = int(rng.integers(0, len(candidates)))
        event = candidates.iloc[pos]
        path = sample_dir / f"{name}_{event['audit_id']}.png"
        plot_refined_poi_event(event=event, bars=bars, output_path=path)
        records.append(
            {
                "stratum": name,
                "status": "plotted",
                "available_count": len(candidates),
                "audit_id": event["audit_id"],
                "canonical_poi_id": event["canonical_poi_id"],
                "direction": event["direction"],
                "classification": _actual_classification(event),
                "rejection_reason": str(event["rejection_reason"]),
                "fvg_size_ticks": event["fvg_size_ticks"],
                "chart_path": str(path),
            }
        )
    manifest = pd.DataFrame.from_records(records)
    manifest.to_csv(out_dir / "random_stratified_audit_manifest.csv", index=False)
    return manifest


def plot_refined_poi_event(
    *,
    event: pd.Series,
    bars: pd.DataFrame,
    output_path: str | Path,
    display_signal_id: str | None = None,
    legacy_signal: pd.Series | None = None,
) -> str:
    """Plot A/B/C, FVG, opening gap, corrected zone, and event lineage."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    middle_id = int(event["poi_middle_bar_id"])
    prior_id = int(event["prior_bar_id"])
    confirm_id = int(event["confirmation_bar_id"])
    activation_id = int(event["activation_bar_id"])
    retest_value = event.get("first_retest_bar_id", event.get("retest_bar_id", pd.NA))
    retest_id = int(retest_value) if pd.notna(retest_value) else None
    formation_left = max(0, prior_id - 2)
    formation_right = min(len(bars) - 1, confirm_id + 2)
    formation = bars.iloc[formation_left : formation_right + 1].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.2, 1.0]})
    ax = axes[0]
    _draw_candles(ax, formation)
    role_colors = {prior_id: ("A", "#375a9e"), middle_id: ("B / POI", "#f2a900"), confirm_id: ("C", "#7a3e9d")}
    for bar_id, (role, color) in role_colors.items():
        row = bars.iloc[bar_id]
        x = mdates.date2num(_naive(row["ts_event_ny"]))
        ax.axvline(x, color=color, linewidth=1.5, alpha=0.8)
        ax.text(x, float(row["high"]), role, color=color, fontsize=9, ha="center", va="bottom")

    x_left = mdates.date2num(_naive(bars.iloc[prior_id]["ts_event_ny"])) - 0.35 / (24 * 60)
    x_right = mdates.date2num(_naive(bars.iloc[confirm_id]["ts_event_ny"])) + 0.35 / (24 * 60)
    width = x_right - x_left
    overlays = [
        (float(event["poi_low"]), float(event["poi_high"]), "#f2a900", 0.11, "refined POI zone"),
        (float(event["middle_low"]), float(event["middle_high"]), "#4c78a8", 0.12, "middle candle B range"),
        (float(event["fvg_low"]), float(event["fvg_high"]), "#2a9d8f", 0.24, "classic FVG"),
        (float(event["close_open_gap_low"]), float(event["close_open_gap_high"]), "#d95f02", 0.24, "B-close/C-open gap"),
    ]
    for y0, y1, color, alpha, label in overlays:
        if np.isfinite(y0) and np.isfinite(y1) and y1 >= y0:
            ax.add_patch(Rectangle((x_left, y0), width, max(y1 - y0, 0.005), facecolor=color, edgecolor=color, alpha=alpha, label=label))
    if bool(event.get("poi_zone_expanded_flag", False)) and float(event["poi_zone_expansion_points"]) > 0:
        if event["direction"] == "bullish":
            y0, y1 = float(event["poi_high"] - event["poi_zone_expansion_points"]), float(event["poi_high"])
        else:
            y0, y1 = float(event["poi_low"]), float(event["poi_low"] + event["poi_zone_expansion_points"])
        ax.add_patch(Rectangle((x_left, y0), width, y1 - y0, facecolor="none", edgecolor="#c0392b", hatch="////", linewidth=1.1, label="Case 1 extension"))
    ax.legend(loc="best", fontsize=8)
    ax.set_ylabel("GC price")
    ax.grid(True, alpha=0.16)

    timeline_ids = [prior_id, middle_id, confirm_id, activation_id]
    break_value = event.get("swing_break_bar_id", pd.NA)
    if pd.notna(break_value):
        timeline_ids.append(int(break_value))
    if retest_id is not None:
        timeline_ids.append(retest_id)
    left = max(0, min(timeline_ids) - 5)
    right = min(len(bars) - 1, max(timeline_ids) + 5)
    timeline = bars.iloc[left : right + 1].copy()
    ax2 = axes[1]
    _draw_candles(ax2, timeline)
    markers = [
        (middle_id, "POI middle B", "#f2a900"),
        (confirm_id, "confirmation C", "#7a3e9d"),
        (activation_id, "activation", "#1f77b4"),
    ]
    if pd.notna(break_value):
        markers.append((int(break_value), "swing break", "#555555"))
    if retest_id is not None and left <= retest_id <= right:
        markers.append((retest_id, "retest / entry", "#c0392b"))
    for bar_id, label, color in markers:
        if left <= bar_id <= right:
            x = mdates.date2num(_naive(bars.iloc[bar_id]["ts_event_ny"]))
            ax2.axvline(x, color=color, linewidth=1.3, alpha=0.85, label=label)
    ax2.axhspan(float(event["poi_low"]), float(event["poi_high"]), color="#f2a900", alpha=0.11)
    ax2.set_ylabel("GC price")
    ax2.grid(True, alpha=0.16)
    ax2.legend(loc="best", fontsize=8)

    signal_label = display_signal_id or str(event.get("audit_id", event.get("canonical_poi_id", "POI")))
    middle_time = _fmt_time(event.get("middle_ts_event_ny", event.get("poi_middle_ts_event_ny")))
    confirm_time = _fmt_time(event.get("confirmation_ts_event_ny"))
    activation_time = _fmt_time(event.get("activation_ts_event_ny"))
    retest_time = _fmt_time(event.get("first_retest_ts_event_ny", event.get("retest_ts_event_ny")))
    if not bool(event.get("final_poi_valid_flag", False)):
        structural = "rejected before structural eligibility"
    elif bool(event.get("structural_swing_break_flag", False)):
        structural = "structural"
    else:
        structural = "local-only"
    title = (
        f"{signal_label} | {event['direction']} | {_actual_classification(event)} | "
        f"FVG={int(event['fvg_size_ticks'])} ticks | >=3={bool(event['fvg_ge_3tick_flag'])} | "
        f">=4={bool(event['fvg_ge_4tick_flag'])} | >=5={bool(event['fvg_ge_5tick_flag'])} | "
        f"formation={bool(event['formation_valid_flag'])} | final={bool(event['final_poi_valid_flag'])} | "
        f"{event['rejection_reason']}\n"
        f"POI middle={middle_time} | confirmation={confirm_time} | activation={activation_time} | "
        f"retest/entry={retest_time} | {structural}"
    )
    fig.suptitle(title, fontsize=10)
    for axis in axes:
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return title


def _draw_candles(ax: Any, frame: pd.DataFrame) -> None:
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    x = mdates.date2num(pd.to_datetime(frame["ts_event_ny"]).dt.tz_localize(None))
    width = 0.68 / (24 * 60)
    for x_value, open_, high, low, close in zip(
        x,
        frame["open"].to_numpy("float64"),
        frame["high"].to_numpy("float64"),
        frame["low"].to_numpy("float64"),
        frame["close"].to_numpy("float64"),
    ):
        color = "#12715b" if close >= open_ else "#a23a3a"
        ax.vlines(x_value, low, high, color=color, linewidth=0.9)
        body_low = min(open_, close)
        ax.add_patch(Rectangle((x_value - width / 2, body_low), width, max(abs(close - open_), 0.005), facecolor=color, edgecolor=color, alpha=0.86))


def _prepare_bars(research_bars: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(VISUAL_BAR_COLUMNS).difference(research_bars.columns))
    if missing:
        raise KeyError(f"research_bars is missing visual columns: {missing}")
    bars = (
        research_bars.loc[research_bars["product"].eq("GC"), VISUAL_BAR_COLUMNS]
        .sort_values(["trade_date_ny", "ts_event_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    bars["bar_id"] = np.arange(len(bars), dtype="int32")
    return bars


def _actual_classification(event: pd.Series) -> str:
    reason = str(event["rejection_reason"])
    if reason == "no_classic_fvg":
        return "invalid_no_fvg"
    if reason == "no_close_open_gap":
        return "invalid_no_close_open_gap"
    return str(event["poi_geometry_case"])


def _reviewer_note(event: pd.Series, fixture: dict[str, Any]) -> str:
    actual = _actual_classification(event)
    if actual == fixture["expected_classification"] and bool(event["formation_valid_flag"]) == fixture["expected_formation_validity"] and str(event["rejection_reason"]) == fixture["expected_rejection_reason"]:
        return "Matches supplied human label."
    direction = str(event["direction"])
    if direction == "bullish":
        inequality = f"low[C]={event['confirmation_low']:.1f} > high[A]={event['prior_high']:.1f}"
    else:
        inequality = f"high[C]={event['confirmation_high']:.1f} < low[A]={event['prior_low']:.1f}"
    return (
        "Underlying OHLC overrides the supplied label: "
        f"{inequality}; C open={event['confirmation_open']:.1f}, "
        f"C close={event['confirmation_close']:.1f}."
    )


def _abc_ohlc_records(signal_id: str, event: pd.Series) -> list[dict[str, Any]]:
    rows = []
    for role, prefix in (("A", "prior"), ("B", "middle"), ("C", "confirmation")):
        rows.append(
            {
                "signal_id": signal_id,
                "role": role,
                "bar_id": event[f"{prefix}_bar_id"] if f"{prefix}_bar_id" in event else event["poi_middle_bar_id"],
                "timestamp_ny": event[f"{prefix}_ts_event_ny"],
                "symbol": event[f"{prefix}_symbol"],
                "open": event[f"{prefix}_open"],
                "high": event[f"{prefix}_high"],
                "low": event[f"{prefix}_low"],
                "close": event[f"{prefix}_close"],
                "direction": event["direction"],
                "classic_fvg_flag": event["classic_fvg_flag"],
                "fvg_size_ticks": event["fvg_size_ticks"],
                "classification": _actual_classification(event),
                "formation_valid_flag": event["formation_valid_flag"],
                "final_poi_valid_flag": event["final_poi_valid_flag"],
                "rejection_reason": str(event["rejection_reason"]),
            }
        )
    return rows


def _naive(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _fmt_time(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return _naive(value).strftime("%Y-%m-%d %H:%M")
