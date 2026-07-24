"""A ~200-strategy pre-registered catalog on the verified simulator.

This module scales the strategy laboratory from a dozen hand-written rules to a
broad, documented family of ~200 archetypes drawn from the technical-analysis
canon, the academic factor/anomaly literature, market microstructure, and the
alpha-factor material in Jansen's *Machine Learning for Algorithmic Trading*.
Each archetype is a fixed, deterministic rule on the causal registered features;
classic indicators map onto those features (RSI/stochastic -> range position,
MACD -> slope/acceleration, Bollinger/Keltner -> VWAP-distance + ATR ratio,
ADX -> efficiency ratio / R-squared, OBV -> signed volume).

Governance that keeps a search this wide honest:

- **Pre-registration.** Every rule and its thresholds are declared here before any
  result is seen.  Thresholds are Development-only marginal quantiles of the
  feature (or fixed structural levels such as 0 for a slope, 1 for a variance
  ratio, 0.5 for a bounded oscillator) - never fitted to outcomes.
- **Point-in-time inputs.** Only causal registered features; fills at the next
  bar open; no forward label is read.
- **Shared exit contract.** All strategies inherit the frozen Section 10
  stop/target, so the search is purely over entries and is directly comparable.
- **Trial-count honesty.** ~200 trials is a severe multiple-comparisons problem;
  the deflated Sharpe ratio and CSCV in :mod:`strategy_evaluation` deflate by
  exactly this trial count, so the bar to "advance" rises with the search size.

Every :class:`StrategySpec` carries ``parameters`` and ``looks_for`` metadata so
the generated strategy catalog documentation cannot drift from the code.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy_lab import StrategyLabConfig, StrategySpec

# Features the catalog reads (all causal registered predictors).  NaN values make
# a comparison False, so a strategy simply does not fire where its input is
# missing; no row is dropped.
CATALOG_FEATURES = (
    "return_5m_atr",
    "return_15m_atr",
    "return_30m_atr",
    "return_60m_atr",
    "absolute_return_1m_atr",
    "momentum_acceleration_5_15",
    "momentum_acceleration_15_30",
    "momentum_alignment_5_30",
    "directional_streak",
    "directional_persistence_15",
    "return_autocorrelation_15",
    "return_sign_change_rate_30",
    "rolling_range_position_15",
    "rolling_range_position_60",
    "atr_ratio_5_20",
    "atr_ratio_20_60",
    "realized_volatility_ratio_5_30",
    "realized_volatility_ratio_15_60",
    "current_range_over_atr",
    "range_compression_ratio_5_30",
    "signed_body_atr",
    "body_to_range",
    "upper_wick_to_range",
    "lower_wick_to_range",
    "close_location_value",
    "range_relative_to_previous_bar",
    "two_bar_directional_balance",
    "three_bar_directional_balance",
    "inside_bar",
    "outside_bar",
    "relative_volume_20",
    "relative_volume_60",
    "volume_zscore_60",
    "volume_acceleration_5_20",
    "volume_per_tick_range",
    "signed_volume_proxy",
    "volume_price_alignment_10",
    "tod_log_volume_z",
    "tod_relative_volume",
    "distance_from_research_day_vwap_atr",
    "distance_from_execution_session_vwap_atr",
    "distance_from_rolling_vwap_20_atr",
    "distance_from_rolling_vwap_60_atr",
    "vwap_slope_5_atr",
    "vwap_slope_15_atr",
    "fraction_above_vwap_15",
    "distance_from_session_open_atr",
    "session_range_position",
    "session_range_over_atr",
    "efficiency_ratio_15",
    "efficiency_ratio_30",
    "efficiency_ratio_60",
    "normalized_ols_slope_15",
    "normalized_ols_slope_30",
    "normalized_ols_slope_60",
    "ols_r_squared_30",
    "choppiness_14",
    "minute_from_execution_window_open",
    "session_progress_fraction",
)

_QUANTILES = (0.10, 0.20, 0.50, 0.80, 0.90)


# ---------------------------------------------------------------------------
# Universe + thresholds
# ---------------------------------------------------------------------------
def load_catalog_universe(project_root: Path) -> pd.DataFrame:
    """Signal candidates (entry fields + stops) joined to the catalog features."""

    import pyarrow.parquet as pq

    sr = project_root / "data" / "processed" / "statistical_research"
    candidates = pd.read_parquet(sr / "signal_candidates_gc.parquet")
    features = pq.read_table(
        sr / "feature_matrix_gc.parquet",
        columns=["observation_id", *CATALOG_FEATURES],
    ).to_pandas(ignore_metadata=True)
    return candidates.merge(features, on="observation_id", how="inner")


def feature_thresholds(universe: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Development-only marginal quantiles for every catalog feature."""

    dev = universe.loc[universe["research_partition"].eq("Development"), list(CATALOG_FEATURES)]
    q = dev.quantile(_QUANTILES)
    out: dict[str, dict[str, float]] = {}
    for feature in CATALOG_FEATURES:
        col = q[feature]
        out[feature] = {
            "p10": float(col.loc[0.10]),
            "p20": float(col.loc[0.20]),
            "p50": float(col.loc[0.50]),
            "p80": float(col.loc[0.80]),
            "p90": float(col.loc[0.90]),
        }
    return out


# ---------------------------------------------------------------------------
# Archetype signal builders (each returns a SignalFn closure over numeric params)
# ---------------------------------------------------------------------------
SignalFn = Callable[[pd.DataFrame, StrategyLabConfig], np.ndarray]


def _mk_follow(feature: str, lo: float, hi: float) -> SignalFn:
    def sig(frame, cfg):
        v = frame[feature].to_numpy(dtype=np.float64)
        out = np.zeros(len(v), dtype=np.int8)
        out[v > hi] = 1
        out[v < lo] = -1
        return out

    return sig


def _mk_fade(feature: str, lo: float, hi: float) -> SignalFn:
    def sig(frame, cfg):
        v = frame[feature].to_numpy(dtype=np.float64)
        out = np.zeros(len(v), dtype=np.int8)
        out[v < lo] = 1
        out[v > hi] = -1
        return out

    return sig


def _mk_sign(feature: str, level: float = 0.0, fade: bool = False) -> SignalFn:
    pos = -1 if fade else 1

    def sig(frame, cfg):
        v = frame[feature].to_numpy(dtype=np.float64)
        out = np.zeros(len(v), dtype=np.int8)
        out[v > level] = pos
        out[v < level] = -pos
        return out

    return sig


def _mk_cross(fast: str, slow: str, fade: bool = False) -> SignalFn:
    pos = -1 if fade else 1

    def sig(frame, cfg):
        a = frame[fast].to_numpy(dtype=np.float64)
        b = frame[slow].to_numpy(dtype=np.float64)
        out = np.zeros(len(a), dtype=np.int8)
        out[a > b] = pos
        out[a < b] = -pos
        return out

    return sig


def _apply(frame: pd.DataFrame, feature: str, op: str, thr: float) -> np.ndarray:
    v = frame[feature].to_numpy(dtype=np.float64)
    if op == "gt":
        return v > thr
    return v < thr


def _mk_gate(base: SignalFn, gate_feature: str, op: str, thr: float) -> SignalFn:
    def sig(frame, cfg):
        s = base(frame, cfg).copy()
        ok = _apply(frame, gate_feature, op, thr)
        s[~ok] = 0
        return s

    return sig


def _mk_session_gate(base: SignalFn, session: str) -> SignalFn:
    def sig(frame, cfg):
        s = base(frame, cfg).copy()
        ok = frame["entry_session"].to_numpy() == session
        s[~ok] = 0
        return s

    return sig


def _mk_both_sign(fa: str, fb: str, fade: bool = False) -> SignalFn:
    pos = -1 if fade else 1

    def sig(frame, cfg):
        a = frame[fa].to_numpy(dtype=np.float64)
        b = frame[fb].to_numpy(dtype=np.float64)
        out = np.zeros(len(a), dtype=np.int8)
        out[(a > 0) & (b > 0)] = pos
        out[(a < 0) & (b < 0)] = -pos
        return out

    return sig


def _mk_multi(long_conds: list[tuple], short_conds: list[tuple]) -> SignalFn:
    def sig(frame, cfg):
        n = len(frame)
        long_ok = np.ones(n, dtype=bool)
        for feature, op, thr in long_conds:
            long_ok &= _apply(frame, feature, op, thr)
        short_ok = np.ones(n, dtype=bool)
        for feature, op, thr in short_conds:
            short_ok &= _apply(frame, feature, op, thr)
        out = np.zeros(n, dtype=np.int8)
        out[long_ok] = 1
        out[short_ok & ~long_ok] = -1
        return out

    return sig


def _g(value: float) -> float:
    return round(float(value), 4)


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------
# (feature, short_name, family, concept-phrase) for symmetric follow / fade rules.
_FOLLOW = [
    ("return_60m_atr", "tsmom_60m", "trend", "60-minute time-series momentum"),
    ("return_30m_atr", "tsmom_30m", "trend", "30-minute momentum"),
    ("return_15m_atr", "tsmom_15m", "trend", "15-minute momentum"),
    ("return_5m_atr", "tsmom_5m", "trend", "5-minute momentum"),
    ("signed_body_atr", "body_momentum", "trend", "signed candle-body thrust"),
    ("normalized_ols_slope_60", "slope_trend_60", "trend", "60-minute regression slope"),
    ("normalized_ols_slope_30", "slope_trend_30", "trend", "30-minute regression slope"),
    ("normalized_ols_slope_15", "slope_trend_15", "trend", "15-minute regression slope"),
    ("momentum_acceleration_5_15", "accel_5_15", "trend", "short-horizon momentum acceleration"),
    ("momentum_acceleration_15_30", "accel_15_30", "trend", "medium-horizon acceleration"),
    ("three_bar_directional_balance", "three_bar_drive", "trend", "three-bar directional thrust"),
    (
        "directional_persistence_15",
        "persistence_trend",
        "trend",
        "15-minute directional persistence",
    ),
    ("vwap_slope_15_atr", "vwap_slope_trend", "trend", "rising/falling VWAP slope"),
    ("vwap_slope_5_atr", "vwap_slope_fast", "trend", "fast VWAP slope"),
    ("volume_price_alignment_10", "vpa_follow", "volume", "volume-price alignment"),
    ("signed_volume_proxy", "obv_follow", "volume", "signed cumulative volume flow"),
    (
        "distance_from_session_open_atr",
        "open_drive_follow",
        "trend",
        "extension from the session open",
    ),
    ("return_autocorrelation_15", "autocorr_trend", "trend", "positive return autocorrelation"),
]

_FADE = [
    (
        "distance_from_execution_session_vwap_atr",
        "vwap_revert_session",
        "reversion",
        "session VWAP",
    ),
    ("distance_from_rolling_vwap_20_atr", "vwap_revert_20", "reversion", "20-bar rolling VWAP"),
    ("distance_from_rolling_vwap_60_atr", "vwap_revert_60", "reversion", "60-bar rolling VWAP"),
    ("distance_from_research_day_vwap_atr", "vwap_revert_day", "reversion", "research-day VWAP"),
    ("distance_from_session_open_atr", "open_fade", "reversion", "the session open"),
    ("return_15m_atr", "reversal_15m", "reversion", "the 15-minute move"),
    ("return_5m_atr", "reversal_5m", "reversion", "the 5-minute move"),
    ("signed_body_atr", "body_fade", "reversion", "the candle body"),
]

# Bounded [0,1]-style oscillators: fade extremes (p20/p80), and a breakout-follow.
_BOUNDED_FADE = [
    ("rolling_range_position_15", "rsi15_fade", "reversion", "15-minute range position (RSI-like)"),
    ("rolling_range_position_60", "rsi60_fade", "reversion", "60-minute range position"),
    ("session_range_position", "session_pos_fade", "reversion", "session range position"),
    ("close_location_value", "clv_fade", "reversion", "close-location value"),
]
_BOUNDED_FOLLOW = [
    ("rolling_range_position_60", "rsi60_follow", "breakout", "60-minute range position"),
    ("session_range_position", "session_pos_follow", "breakout", "session range position"),
]

# Price-vs-anchor and sign-of-feature rules (which side of a level the value sits).
_SIGN = [
    (
        "distance_from_execution_session_vwap_atr",
        "above_session_vwap",
        "vwap",
        0.0,
        "the session VWAP",
    ),
    ("distance_from_rolling_vwap_60_atr", "above_rolling_vwap_60", "vwap", 0.0, "the 60-bar VWAP"),
    (
        "fraction_above_vwap_15",
        "vwap_majority_side",
        "vwap",
        0.5,
        "the VWAP (majority of last 15m)",
    ),
    ("normalized_ols_slope_30", "slope_sign_30", "trend", 0.0, "a zero regression slope"),
    ("signed_volume_proxy", "flow_sign", "volume", 0.0, "zero signed flow"),
    ("two_bar_directional_balance", "two_bar_sign", "trend", 0.0, "a two-bar balance turn"),
    ("three_bar_directional_balance", "three_bar_sign", "trend", 0.0, "a three-bar balance turn"),
    ("momentum_acceleration_5_15", "accel_sign", "trend", 0.0, "acceleration turning positive"),
    ("distance_from_session_open_atr", "open_side", "trend", 0.0, "the session open"),
    ("directional_streak", "streak_sign", "trend", 0.0, "the directional streak"),
    ("close_location_value", "clv_sign", "candle", 0.0, "the bar midpoint (close location)"),
]

# Directional crossovers of two features.
_CROSS = [
    (
        "normalized_ols_slope_15",
        "normalized_ols_slope_60",
        "ma_cross_15_60",
        "trend",
        "fast slope crossing the slow slope (MA/MACD cross)",
    ),
    (
        "normalized_ols_slope_15",
        "normalized_ols_slope_30",
        "ma_cross_15_30",
        "trend",
        "15m slope crossing the 30m slope",
    ),
    (
        "normalized_ols_slope_30",
        "normalized_ols_slope_60",
        "ma_cross_30_60",
        "trend",
        "30m slope crossing the 60m slope",
    ),
    (
        "return_5m_atr",
        "return_30m_atr",
        "return_cross_5_30",
        "trend",
        "5m momentum crossing 30m momentum",
    ),
    (
        "rolling_range_position_15",
        "rolling_range_position_60",
        "stoch_cross",
        "trend",
        "fast range position crossing slow (stochastic %K/%D)",
    ),
    (
        "distance_from_rolling_vwap_20_atr",
        "distance_from_rolling_vwap_60_atr",
        "vwap_band_cross",
        "vwap",
        "20-bar VWAP distance crossing the 60-bar distance",
    ),
    (
        "vwap_slope_5_atr",
        "vwap_slope_15_atr",
        "vwap_slope_cross",
        "vwap",
        "fast VWAP slope crossing the slow VWAP slope",
    ),
    (
        "efficiency_ratio_15",
        "efficiency_ratio_60",
        "efficiency_cross",
        "trend",
        "short efficiency crossing long efficiency",
    ),
    (
        "return_15m_atr",
        "return_60m_atr",
        "return_cross_15_60",
        "trend",
        "15m momentum crossing 60m momentum",
    ),
    (
        "efficiency_ratio_15",
        "efficiency_ratio_30",
        "efficiency_cross_15_30",
        "trend",
        "short efficiency crossing medium efficiency",
    ),
    (
        "atr_ratio_5_20",
        "atr_ratio_20_60",
        "atr_ratio_cross",
        "volatility",
        "short-term ATR ratio crossing the long-term ratio",
    ),
    (
        "distance_from_session_open_atr",
        "distance_from_execution_session_vwap_atr",
        "open_vs_vwap_cross",
        "vwap",
        "distance from open crossing distance from VWAP",
    ),
    (
        "realized_volatility_ratio_5_30",
        "realized_volatility_ratio_15_60",
        "rvol_ratio_cross",
        "volatility",
        "short realized-vol ratio crossing the long ratio",
    ),
]

# Regime-gated directional bases: base = sign(direction feature), gated by a regime.
_GATE_BASES = [
    ("return_30m_atr", "mom30"),
    ("return_60m_atr", "mom60"),
    ("normalized_ols_slope_30", "slope30"),
    ("normalized_ols_slope_60", "slope60"),
    ("signed_body_atr", "body"),
    ("return_15m_atr", "mom15"),
    ("momentum_acceleration_5_15", "accel"),
]
_REGIMES = [
    ("atr_ratio_5_20", "gt", "p80", "volx", "expanding short-term volatility"),
    ("atr_ratio_20_60", "gt", "p80", "volx2", "expanding medium-term volatility"),
    ("realized_volatility_ratio_5_30", "gt", "p80", "rvx", "expanding realized volatility"),
    ("efficiency_ratio_30", "gt", "p80", "trendy", "a strong (efficient) trend"),
    ("ols_r_squared_30", "gt", "p80", "linear", "a linear price path"),
    ("choppiness_14", "lt", "p20", "lowchop", "a low-choppiness regime"),
    ("relative_volume_60", "gt", "p80", "highvol", "elevated relative volume"),
    ("volume_zscore_60", "gt", "p80", "volsurge", "a volume surge"),
    ("tod_relative_volume", "gt", "p80", "activetod", "an active time-of-day window"),
    ("session_range_over_atr", "gt", "p80", "widerange", "a wide developing session range"),
    ("fraction_above_vwap_15", "gt", "p50", "abovevwap", "price mostly above VWAP"),
]

# Fade bases gated by mean-reverting regimes.
_FADE_GATE_BASES = [
    ("distance_from_rolling_vwap_20_atr", "vwap20"),
    ("distance_from_rolling_vwap_60_atr", "vwap60"),
    ("distance_from_execution_session_vwap_atr", "vwapS"),
    ("return_15m_atr", "ret15"),
]
_FADE_REGIMES = [
    ("choppiness_14", "gt", "p80", "highchop", "a high-choppiness regime"),
    ("ols_r_squared_30", "lt", "p20", "nontrend", "a non-trending (low R-squared) regime"),
    ("realized_volatility_ratio_5_30", "lt", "p20", "lowvol", "a contracting-volatility regime"),
]

# Session/time-gated momentum.
_TIME_GATES = [
    (
        "opening_drive",
        "return_5m_atr",
        "minute_from_execution_window_open",
        "lt",
        "p20",
        "session",
        "the opening drift, in the first minutes of the window",
    ),
    (
        "late_session_trend",
        "normalized_ols_slope_30",
        "session_progress_fraction",
        "gt",
        "p80",
        "session",
        "the trend late in the session",
    ),
    (
        "early_session_trend",
        "return_30m_atr",
        "session_progress_fraction",
        "lt",
        "p20",
        "session",
        "momentum early in the session",
    ),
]


# Multi-condition breakout and candle patterns (long_conds, short_conds).
def _breakout_specs(th):
    p = th
    specs = []
    specs.append(
        (
            "session_break",
            "breakout",
            "Break to a session extreme with a directional body.",
            _mk_multi(
                [
                    ("session_range_position", "gt", p["session_range_position"]["p90"]),
                    ("signed_body_atr", "gt", 0.0),
                ],
                [
                    ("session_range_position", "lt", p["session_range_position"]["p10"]),
                    ("signed_body_atr", "lt", 0.0),
                ],
            ),
            {"long": "session_range_position>p90 & body>0", "short": "mirror"},
            "Long on a new session-range high with a positive body; short on a new low with a negative body.",
        )
    )
    specs.append(
        (
            "donchian60_break",
            "breakout",
            "Donchian/Turtle 60-minute channel breakout.",
            _mk_multi(
                [
                    ("rolling_range_position_60", "gt", p["rolling_range_position_60"]["p90"]),
                    ("signed_body_atr", "gt", 0.0),
                ],
                [
                    ("rolling_range_position_60", "lt", p["rolling_range_position_60"]["p10"]),
                    ("signed_body_atr", "lt", 0.0),
                ],
            ),
            {"long": "rolling_range_position_60>p90 & body>0", "short": "mirror"},
            "Long on a fresh 60-minute high with a positive body; short on a fresh low.",
        )
    )
    specs.append(
        (
            "range_expansion",
            "breakout",
            "ATR range-expansion ignition bar.",
            _mk_multi(
                [
                    ("current_range_over_atr", "gt", p["current_range_over_atr"]["p80"]),
                    ("signed_body_atr", "gt", 0.0),
                ],
                [
                    ("current_range_over_atr", "gt", p["current_range_over_atr"]["p80"]),
                    ("signed_body_atr", "lt", 0.0),
                ],
            ),
            {"long": "current_range_over_atr>p80 & body>0", "short": "range>p80 & body<0"},
            "Long on an expansion bar closing up; short on an expansion bar closing down.",
        )
    )
    specs.append(
        (
            "wide_range_bar",
            "breakout",
            "Wide-range-bar directional thrust.",
            _mk_multi(
                [("range_relative_to_previous_bar", "gt", 2.0), ("signed_body_atr", "gt", 0.0)],
                [("range_relative_to_previous_bar", "gt", 2.0), ("signed_body_atr", "lt", 0.0)],
            ),
            {"long": "range>2x prior & body>0", "short": "range>2x prior & body<0"},
            "Long on a bar more than twice the prior range closing up; short if closing down.",
        )
    )
    specs.append(
        (
            "squeeze_release",
            "breakout",
            "TTM-squeeze release: coil then expansion.",
            _mk_multi(
                [
                    (
                        "range_compression_ratio_5_30",
                        "lt",
                        p["range_compression_ratio_5_30"]["p20"],
                    ),
                    ("current_range_over_atr", "gt", p["current_range_over_atr"]["p50"]),
                    ("normalized_ols_slope_15", "gt", 0.0),
                ],
                [
                    (
                        "range_compression_ratio_5_30",
                        "lt",
                        p["range_compression_ratio_5_30"]["p20"],
                    ),
                    ("current_range_over_atr", "gt", p["current_range_over_atr"]["p50"]),
                    ("normalized_ols_slope_15", "lt", 0.0),
                ],
            ),
            {"long": "compression<p20 & range>p50 & slope>0", "short": "mirror"},
            "After a volatility squeeze, take the expansion in the direction of the slope.",
        )
    )
    specs.append(
        (
            "inside_bar_break",
            "breakout",
            "Inside-bar (Harami) breakout by body direction.",
            _mk_multi(
                [("inside_bar", "gt", 0.5), ("signed_body_atr", "gt", 0.0)],
                [("inside_bar", "gt", 0.5), ("signed_body_atr", "lt", 0.0)],
            ),
            {"long": "inside_bar & body>0", "short": "inside_bar & body<0"},
            "On an inside bar, go with the sign of the body.",
        )
    )
    specs.append(
        (
            "outside_bar_break",
            "breakout",
            "Outside/engulfing bar directional break.",
            _mk_multi(
                [("outside_bar", "gt", 0.5), ("signed_body_atr", "gt", 0.0)],
                [("outside_bar", "gt", 0.5), ("signed_body_atr", "lt", 0.0)],
            ),
            {"long": "outside_bar & body>0", "short": "outside_bar & body<0"},
            "On an outside (engulfing) bar, go with the sign of the body.",
        )
    )
    specs.append(
        (
            "volume_breakout",
            "breakout",
            "Volume-confirmed session breakout.",
            _mk_multi(
                [
                    ("session_range_position", "gt", p["session_range_position"]["p90"]),
                    ("volume_zscore_60", "gt", p["volume_zscore_60"]["p80"]),
                    ("signed_body_atr", "gt", 0.0),
                ],
                [
                    ("session_range_position", "lt", p["session_range_position"]["p10"]),
                    ("volume_zscore_60", "gt", p["volume_zscore_60"]["p80"]),
                    ("signed_body_atr", "lt", 0.0),
                ],
            ),
            {"long": "new high & volume z>p80 & body>0", "short": "mirror"},
            "Long on a session high confirmed by a volume z-spike; short on a confirmed low.",
        )
    )
    return specs


def _candle_specs(th):
    p = th
    specs = []
    specs.append(
        (
            "pin_bar",
            "candle",
            "Pin-bar / hammer wick rejection.",
            _mk_multi(
                [
                    ("lower_wick_to_range", "gt", p["lower_wick_to_range"]["p80"]),
                    ("close_location_value", "gt", p["close_location_value"]["p80"]),
                ],
                [
                    ("upper_wick_to_range", "gt", p["upper_wick_to_range"]["p80"]),
                    ("close_location_value", "lt", p["close_location_value"]["p20"]),
                ],
            ),
            {"long": "long lower wick & close high", "short": "long upper wick & close low"},
            "Long a hammer (long lower wick, close near high); short a shooting star.",
        )
    )
    specs.append(
        (
            "marubozu",
            "candle",
            "Marubozu full-body conviction bar.",
            _mk_multi(
                [
                    ("body_to_range", "gt", p["body_to_range"]["p80"]),
                    ("signed_body_atr", "gt", 0.0),
                ],
                [
                    ("body_to_range", "gt", p["body_to_range"]["p80"]),
                    ("signed_body_atr", "lt", 0.0),
                ],
            ),
            {"long": "body/range>p80 & body>0", "short": "body/range>p80 & body<0"},
            "Long a strong full-bodied up bar; short a full-bodied down bar.",
        )
    )
    specs.append(
        (
            "doji_fade",
            "candle",
            "Doji indecision fade at a range extreme.",
            _mk_multi(
                [
                    ("body_to_range", "lt", p["body_to_range"]["p20"]),
                    ("rolling_range_position_60", "lt", p["rolling_range_position_60"]["p20"]),
                ],
                [
                    ("body_to_range", "lt", p["body_to_range"]["p20"]),
                    ("rolling_range_position_60", "gt", p["rolling_range_position_60"]["p80"]),
                ],
            ),
            {"long": "tiny body at range low", "short": "tiny body at range high"},
            "Fade a small-bodied indecision bar back toward the mean from a range extreme.",
        )
    )
    specs.append(
        (
            "wick_rejection_level",
            "candle",
            "Wick rejection at a session extreme.",
            _mk_multi(
                [
                    ("lower_wick_to_range", "gt", p["lower_wick_to_range"]["p80"]),
                    ("session_range_position", "lt", p["session_range_position"]["p20"]),
                ],
                [
                    ("upper_wick_to_range", "gt", p["upper_wick_to_range"]["p80"]),
                    ("session_range_position", "gt", p["session_range_position"]["p80"]),
                ],
            ),
            {"long": "lower wick at session low", "short": "upper wick at session high"},
            "Buy a lower-wick rejection at the session low; sell an upper-wick rejection at the high.",
        )
    )
    specs.append(
        (
            "key_reversal",
            "candle",
            "Key-reversal bar from a range extreme.",
            _mk_multi(
                [
                    ("rolling_range_position_15", "lt", p["rolling_range_position_15"]["p20"]),
                    ("close_location_value", "gt", p["close_location_value"]["p80"]),
                ],
                [
                    ("rolling_range_position_15", "gt", p["rolling_range_position_15"]["p80"]),
                    ("close_location_value", "lt", p["close_location_value"]["p20"]),
                ],
            ),
            {"long": "new short-term low but close high", "short": "new high but close low"},
            "Long when price makes a new low but closes near its high; short the mirror.",
        )
    )
    return specs


def _volume_specs(th):
    p = th
    specs = []
    specs.append(
        (
            "climax_fade",
            "volume",
            "Volume-climax exhaustion fade.",
            _mk_multi(
                [
                    ("volume_zscore_60", "gt", p["volume_zscore_60"]["p90"]),
                    ("signed_body_atr", "lt", 0.0),
                ],
                [
                    ("volume_zscore_60", "gt", p["volume_zscore_60"]["p90"]),
                    ("signed_body_atr", "gt", 0.0),
                ],
            ),
            {"long": "volume z>p90 on a down bar (fade)", "short": "volume z>p90 on an up bar"},
            "Fade a blow-off: buy a high-volume down bar, sell a high-volume up bar.",
        )
    )
    specs.append(
        (
            "volume_dryup_reversion",
            "volume",
            "Volume dry-up reversion at an extreme.",
            _mk_gate(
                _mk_fade(
                    "session_range_position",
                    p["session_range_position"]["p20"],
                    p["session_range_position"]["p80"],
                ),
                "relative_volume_20",
                "lt",
                p["relative_volume_20"]["p20"],
            ),
            {"gate": "relative_volume_20<p20", "base": "fade session position"},
            "When participation dries up, fade the session-range extreme.",
        )
    )
    specs.append(
        (
            "rvol_momentum",
            "volume",
            "Relative-volume-gated momentum.",
            _mk_gate(
                _mk_sign("return_15m_atr"),
                "relative_volume_20",
                "gt",
                p["relative_volume_20"]["p80"],
            ),
            {"gate": "relative_volume_20>p80", "base": "sign(return_15m)"},
            "Follow the 15-minute move only when relative volume is elevated.",
        )
    )
    specs.append(
        (
            "tod_volume_surprise",
            "volume",
            "Time-of-day volume-surprise momentum.",
            _mk_gate(
                _mk_sign("signed_body_atr"), "tod_log_volume_z", "gt", p["tod_log_volume_z"]["p80"]
            ),
            {"gate": "tod_log_volume_z>p80", "base": "sign(body)"},
            "Follow the body direction when volume exceeds its time-of-day norm.",
        )
    )
    specs.append(
        (
            "volume_accel_momentum",
            "volume",
            "Volume-acceleration ignition.",
            _mk_gate(_mk_sign("signed_body_atr"), "volume_acceleration_5_20", "gt", 0.0),
            {"gate": "volume_acceleration_5_20>0", "base": "sign(body)"},
            "Follow the body direction when volume is accelerating.",
        )
    )
    return specs


def _session_specs(th):
    p = th
    specs = []
    for name, base_feat, gate_feat, op, q, fam, phrase in _TIME_GATES:
        specs.append(
            (
                name,
                fam,
                f"Trade {phrase}.",
                _mk_gate(_mk_sign(base_feat), gate_feat, op, p[gate_feat][q]),
                {"gate": f"{gate_feat} {op} {q}", "base": f"sign({base_feat})"},
                f"Follow {base_feat} only when {gate_feat} indicates {phrase}.",
            )
        )
    # session-restricted momentum and breakouts
    for session in ("London", "New York"):
        tag = session.lower().replace(" ", "").replace("_", "")
        specs.append(
            (
                f"{tag}_momentum",
                "session",
                f"{session}-session momentum.",
                _mk_session_gate(_mk_sign("return_30m_atr"), session),
                {"session": session, "base": "sign(return_30m)"},
                f"Follow the 30-minute move only during the {session} window.",
            )
        )
        specs.append(
            (
                f"{tag}_breakout",
                "session",
                f"{session}-session range breakout.",
                _mk_session_gate(
                    _mk_multi(
                        [
                            ("session_range_position", "gt", p["session_range_position"]["p90"]),
                            ("signed_body_atr", "gt", 0.0),
                        ],
                        [
                            ("session_range_position", "lt", p["session_range_position"]["p10"]),
                            ("signed_body_atr", "lt", 0.0),
                        ],
                    ),
                    session,
                ),
                {"session": session, "base": "session breakout"},
                f"Take session-range breakouts only during the {session} window.",
            )
        )
    return specs


def build_catalog(thresholds: dict[str, dict[str, float]]) -> list[StrategySpec]:
    """Expand every archetype into a named, documented StrategySpec (~200)."""

    th = thresholds
    specs: list[StrategySpec] = []

    def add(name, family, thesis, signal, parameters, looks_for):
        specs.append(
            StrategySpec(name, family, thesis, signal, parameters=parameters, looks_for=looks_for)
        )

    # -- symmetric follow (p20/p80) and extreme follow (p10/p90) ---------------
    for feature, name, family, concept in _FOLLOW:
        lo, hi = th[feature]["p20"], th[feature]["p80"]
        add(
            name,
            family,
            f"Follow {concept}.",
            _mk_follow(feature, lo, hi),
            {
                "archetype": "follow",
                "feature": feature,
                "long_above": _g(hi),
                "short_below": _g(lo),
                "thresholds": "Dev p20 / p80",
            },
            f"Long when {feature} > {hi:.3g}; short when < {lo:.3g}.",
        )
    for feature, name, family, concept in _FOLLOW:
        lo, hi = th[feature]["p10"], th[feature]["p90"]
        add(
            f"{name}_x",
            family,
            f"Follow {concept} (extreme).",
            _mk_follow(feature, lo, hi),
            {
                "archetype": "follow_extreme",
                "feature": feature,
                "long_above": _g(hi),
                "short_below": _g(lo),
                "thresholds": "Dev p10 / p90",
            },
            f"Long when {feature} > {hi:.3g}; short when < {lo:.3g} (extreme tails).",
        )

    # -- symmetric fade (p20/p80) and extreme fade (p10/p90) -------------------
    for feature, name, family, anchor in _FADE:
        lo, hi = th[feature]["p20"], th[feature]["p80"]
        add(
            name,
            family,
            f"Fade stretch from {anchor}.",
            _mk_fade(feature, lo, hi),
            {
                "archetype": "fade",
                "feature": feature,
                "long_below": _g(lo),
                "short_above": _g(hi),
                "thresholds": "Dev p20 / p80",
            },
            f"Long when {feature} < {lo:.3g} (stretched down); short when > {hi:.3g}.",
        )
    for feature, name, family, anchor in _FADE:
        lo, hi = th[feature]["p10"], th[feature]["p90"]
        add(
            f"{name}_x",
            family,
            f"Fade extreme stretch from {anchor}.",
            _mk_fade(feature, lo, hi),
            {
                "archetype": "fade_extreme",
                "feature": feature,
                "long_below": _g(lo),
                "short_above": _g(hi),
                "thresholds": "Dev p10 / p90",
            },
            f"Long when {feature} < {lo:.3g}; short when > {hi:.3g} (extreme tails).",
        )

    # -- bounded oscillators ---------------------------------------------------
    for feature, name, family, concept in _BOUNDED_FADE:
        lo, hi = th[feature]["p20"], th[feature]["p80"]
        add(
            name,
            family,
            f"Fade {concept} extremes.",
            _mk_fade(feature, lo, hi),
            {
                "archetype": "bounded_fade",
                "feature": feature,
                "long_below": _g(lo),
                "short_above": _g(hi),
                "thresholds": "Dev p20 / p80",
            },
            f"Long when {feature} < {lo:.3g} (oversold); short when > {hi:.3g} (overbought).",
        )
    for feature, name, family, concept in _BOUNDED_FOLLOW:
        lo, hi = th[feature]["p10"], th[feature]["p90"]
        add(
            name,
            family,
            f"Follow {concept} breakout.",
            _mk_follow(feature, lo, hi),
            {
                "archetype": "bounded_follow",
                "feature": feature,
                "long_above": _g(hi),
                "short_below": _g(lo),
                "thresholds": "Dev p10 / p90",
            },
            f"Long when {feature} > {hi:.3g} (upper extreme); short when < {lo:.3g}.",
        )

    # -- price-vs-anchor sign --------------------------------------------------
    for feature, name, family, level, anchor in _SIGN:
        add(
            name,
            family,
            f"Trade the side of {anchor}.",
            _mk_sign(feature, level=level),
            {"archetype": "sign", "feature": feature, "level": _g(level)},
            f"Long when {feature} > {level:g}; short when < {level:g}.",
        )

    # -- crossovers ------------------------------------------------------------
    for fast, slow, name, family, concept in _CROSS:
        add(
            name,
            family,
            f"Trade {concept}.",
            _mk_cross(fast, slow),
            {"archetype": "cross", "fast": fast, "slow": slow},
            f"Long when {fast} > {slow}; short when {fast} < {slow}.",
        )
    # VWAP dual-slope agreement (both above 0)
    add(
        "vwap_trend_confluence",
        "vwap",
        "Long above a rising VWAP, short below a falling one.",
        _mk_both_sign("distance_from_rolling_vwap_60_atr", "vwap_slope_15_atr"),
        {"archetype": "both_sign", "features": "vwap distance & slope"},
        "Long when 60-bar VWAP distance>0 and VWAP slope>0; short when both<0.",
    )

    # -- regime-gated momentum -------------------------------------------------
    for base_feat, base_tag in _GATE_BASES:
        for gate_feat, op, q, reg_tag, phrase in _REGIMES:
            add(
                f"{base_tag}_{reg_tag}",
                "regime",
                f"Momentum, active only in {phrase}.",
                _mk_gate(_mk_sign(base_feat), gate_feat, op, th[gate_feat][q]),
                {
                    "archetype": "gated_momentum",
                    "base": f"sign({base_feat})",
                    "gate": f"{gate_feat} {op} {q}",
                    "gate_value": _g(th[gate_feat][q]),
                },
                f"Follow {base_feat} only when {gate_feat} indicates {phrase}.",
            )

    # -- regime-gated fade -----------------------------------------------------
    for base_feat, base_tag in _FADE_GATE_BASES:
        lo, hi = th[base_feat]["p20"], th[base_feat]["p80"]
        for gate_feat, op, q, reg_tag, phrase in _FADE_REGIMES:
            add(
                f"fade_{base_tag}_{reg_tag}",
                "regime",
                f"Mean-reversion, active only in {phrase}.",
                _mk_gate(_mk_fade(base_feat, lo, hi), gate_feat, op, th[gate_feat][q]),
                {
                    "archetype": "gated_fade",
                    "base": f"fade({base_feat})",
                    "gate": f"{gate_feat} {op} {q}",
                    "gate_value": _g(th[gate_feat][q]),
                },
                f"Fade {base_feat} only when {gate_feat} indicates {phrase}.",
            )

    # -- multi-condition breakout / candle / volume / session ------------------
    for name, family, thesis, signal, params, looks in _breakout_specs(th):
        add(name, family, thesis, signal, {"archetype": "multi", **params}, looks)
    for name, family, thesis, signal, params, looks in _candle_specs(th):
        add(name, family, thesis, signal, {"archetype": "multi", **params}, looks)
    for name, family, thesis, signal, params, looks in _volume_specs(th):
        add(name, family, thesis, signal, {"archetype": "gated/multi", **params}, looks)
    for name, family, thesis, signal, params, looks in _session_specs(th):
        add(name, family, thesis, signal, {"archetype": "gated", **params}, looks)

    # -- null benchmarks -------------------------------------------------------
    add(
        "always_long",
        "benchmark",
        "Null benchmark: long every eligible observation.",
        lambda frame, cfg: np.ones(len(frame), dtype=np.int8),
        {"archetype": "benchmark"},
        "Always long.",
    )
    add(
        "always_short",
        "benchmark",
        "Null benchmark: short every eligible observation.",
        lambda frame, cfg: np.full(len(frame), -1, dtype=np.int8),
        {"archetype": "benchmark"},
        "Always short.",
    )

    _assert_unique_names(specs)
    return specs


def _assert_unique_names(specs: list[StrategySpec]) -> None:
    names = [s.name for s in specs]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate strategy names in catalog: {sorted(dupes)}")


CATALOG_MIN_FIRE_RATE = 0.005


def catalog_library(
    universe: pd.DataFrame, min_fire_rate: float = CATALOG_MIN_FIRE_RATE
) -> tuple[list[StrategySpec], list[str]]:
    """Build the catalog and drop rules that never meaningfully fire.

    A pre-registered rule whose condition almost never occurs on this data is
    structurally inapplicable, not a strategy.  Dropping it is a marginal
    validity filter (does the condition ever occur), not outcome-based selection.
    Returns the kept specs and the names of the dropped (near-dead) rules.
    """

    cfg = StrategyLabConfig()
    specs = build_catalog(feature_thresholds(universe))
    kept: list[StrategySpec] = []
    dropped: list[str] = []
    for spec in specs:
        if spec.family == "benchmark":
            kept.append(spec)
            continue
        fire_rate = float((spec.signal(universe, cfg) != 0).mean())
        if fire_rate >= min_fire_rate:
            kept.append(spec)
        else:
            dropped.append(spec.name)
    return kept, dropped
