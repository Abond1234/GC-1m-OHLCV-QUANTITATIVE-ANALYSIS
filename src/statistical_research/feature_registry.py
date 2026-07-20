"""Authoritative Section 6 feature registry.

The registry is deliberately defined before matrix construction.  Later research
must obtain predictor columns from this registry rather than infer them by
excluding identifiers or outcome columns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

FEATURE_VERSION = "1.0.0"
EXPERIMENTAL_SUFFIX = "_exp"


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    display_name: str
    family: str
    feature_tier: str
    is_experimental: bool
    description: str
    economic_or_market_hypothesis: str
    formula_or_definition: str
    lookback: str
    input_columns: tuple[str, ...]
    availability_timestamp: str
    reset_boundary: str
    minimum_history: int
    missing_value_policy: str
    normalization_method: str
    fit_scope: str
    expected_range: str
    signed_or_unsigned: str
    output_dtype: str
    source_function: str
    version: str
    creative_rationale: str
    validation_minimum: float | None = None
    validation_maximum: float | None = None


FAMILY_HYPOTHESES = {
    "price_return_momentum": (
        "Recent displacement, alignment, and path position may characterize continuation, pullback, or exhaustion state."
    ),
    "volatility_range_state": (
        "Completed-bar range and realized-variation measures characterize the current opportunity and risk regime."
    ),
    "candle_geometry": (
        "Continuous candle geometry summarizes local directional travel and rejection without brittle pattern taxonomies."
    ),
    "volume_activity": (
        "Causal volume normalization measures participation and activity context, not true aggressor-side order flow."
    ),
    "vwap_session_state": (
        "Location relative to causal price-volume anchors and completed session state may distinguish extension from reversion."
    ),
    "trend_persistence": (
        "Path efficiency, normalized trend, persistence, and choppiness describe recent market state without asserting an edge."
    ),
    "session_clock_calendar": (
        "Known schedule context may condition activity and market state across the London and New York execution windows."
    ),
    "experimental_hypothesis": (
        "A predefined causal OHLCV hypothesis may capture a market-state dimension not represented directly by the core set."
    ),
}


def _spec(
    feature_name: str,
    display_name: str,
    family: str,
    formula: str,
    lookback: str,
    inputs: Iterable[str],
    minimum_history: int,
    normalization: str,
    expected_range: str,
    signed_or_unsigned: str,
    output_dtype: str = "float32",
    *,
    reset_boundary: str = "continuity run",
    missing_policy: str = "Null until complete history exists or when a required denominator is zero.",
    fit_scope: str = "No fitted parameters",
    experimental: bool = False,
    creative_rationale: str = "",
    validation_minimum: float | None = None,
    validation_maximum: float | None = None,
) -> FeatureSpec:
    tier = "experimental" if experimental else "core"
    display = f"* {display_name}" if experimental else display_name
    description = f"Causal {display_name.lower()} available after completed decision bar t."
    return FeatureSpec(
        feature_name=feature_name,
        display_name=display,
        family=family,
        feature_tier=tier,
        is_experimental=experimental,
        description=description,
        economic_or_market_hypothesis=FAMILY_HYPOTHESES[family],
        formula_or_definition=formula,
        lookback=lookback,
        input_columns=tuple(inputs),
        availability_timestamp="Close of completed decision bar t",
        reset_boundary=reset_boundary,
        minimum_history=minimum_history,
        missing_value_policy=missing_policy,
        normalization_method=normalization,
        fit_scope=fit_scope,
        expected_range=expected_range,
        signed_or_unsigned=signed_or_unsigned,
        output_dtype=output_dtype,
        source_function=f"feature_engineering.build_feature_matrix::{family}",
        version=FEATURE_VERSION,
        creative_rationale=creative_rationale,
        validation_minimum=validation_minimum,
        validation_maximum=validation_maximum,
    )


def _build_feature_specs() -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []
    add = specs.append

    # Price, return, momentum, and multi-horizon alignment (16).
    for horizon in (1, 5, 15, 30, 60):
        add(
            _spec(
                f"return_{horizon}m_bps",
                f"Return — {horizon}m",
                "price_return_momentum",
                f"10,000 × log(close_t / close_(t-{horizon}))",
                f"{horizon} bars",
                ("close",),
                horizon + 1,
                "Log return in basis points",
                "Unbounded",
                "signed",
            )
        )
    for horizon in (5, 15, 30, 60):
        add(
            _spec(
                f"return_{horizon}m_atr",
                f"ATR-normalized displacement — {horizon}m",
                "price_return_momentum",
                f"(close_t - close_(t-{horizon})) / ATR_20(t)",
                f"{horizon} bars",
                ("close", "high", "low"),
                max(horizon + 1, 20),
                "Current causal 20-bar ATR",
                "Unbounded",
                "signed",
            )
        )
    add(
        _spec(
            "absolute_return_1m_atr",
            "Absolute return — 1m in ATR",
            "price_return_momentum",
            "abs(close_t - close_(t-1)) / ATR_20(t)",
            "1 bar",
            ("close", "high", "low"),
            20,
            "Current causal 20-bar ATR",
            "[0, +inf)",
            "unsigned",
            validation_minimum=0.0,
        )
    )
    for short, long in ((5, 15), (15, 30)):
        add(
            _spec(
                f"momentum_acceleration_{short}_{long}",
                f"Momentum acceleration — {short}/{long}m",
                "price_return_momentum",
                f"return_{short}m_bps/{short} - return_{long}m_bps/{long}",
                f"{long} bars",
                ("close",),
                long + 1,
                "Difference in log-bps-per-minute rates",
                "Unbounded",
                "signed",
            )
        )
    add(
        _spec(
            "directional_streak",
            "Directional streak",
            "price_return_momentum",
            "Signed consecutive count of non-zero close-to-close directions, capped at ±120",
            "Run length",
            ("close",),
            2,
            "Signed run length",
            "[-120, 120]",
            "signed",
            "Int16",
            validation_minimum=-120.0,
            validation_maximum=120.0,
        )
    )
    for horizon in (15, 60):
        add(
            _spec(
                f"rolling_range_position_{horizon}",
                f"Rolling range position — {horizon}m",
                "price_return_momentum",
                f"(close_t - rolling_low_{horizon}) / (rolling_high_{horizon} - rolling_low_{horizon})",
                f"{horizon} bars",
                ("high", "low", "close"),
                horizon,
                "Rolling high-low range",
                "[0, 1]",
                "unsigned",
                validation_minimum=0.0,
                validation_maximum=1.0,
            )
        )
    add(
        _spec(
            "momentum_alignment_5_30",
            "Momentum alignment — 5/30m",
            "price_return_momentum",
            "True when non-zero 5- and 30-minute log returns have the same sign",
            "30 bars",
            ("close",),
            31,
            "None",
            "{False, True}",
            "boolean",
            "boolean",
        )
    )

    # Volatility, range, compression, and expansion state (13).
    for horizon in (5, 20, 60):
        add(
            _spec(
                f"atr_{horizon}",
                f"Average true range — {horizon}m",
                "volatility_range_state",
                f"Mean true range over {horizon} completed bars",
                f"{horizon} bars",
                ("high", "low", "close"),
                horizon,
                "GC price points",
                "[0, +inf)",
                "unsigned",
                validation_minimum=0.0,
            )
        )
    for numerator, denominator in ((5, 20), (20, 60)):
        add(
            _spec(
                f"atr_ratio_{numerator}_{denominator}",
                f"ATR ratio — {numerator}/{denominator}m",
                "volatility_range_state",
                f"ATR_{numerator}(t) / ATR_{denominator}(t)",
                f"{denominator} bars",
                ("high", "low", "close"),
                denominator,
                "Ratio of causal ATR estimates",
                "[0, +inf)",
                "unsigned",
                validation_minimum=0.0,
            )
        )
    for horizon in (5, 15, 30, 60):
        add(
            _spec(
                f"realized_volatility_{horizon}",
                f"Realized volatility — {horizon}m",
                "volatility_range_state",
                f"10,000 × sqrt(sum(log_return_i², {horizon}))",
                f"{horizon} one-minute returns",
                ("close",),
                horizon + 1,
                "Unannualized total log variation in basis points",
                "[0, +inf)",
                "unsigned",
                validation_minimum=0.0,
            )
        )
    for numerator, denominator in ((5, 30), (15, 60)):
        add(
            _spec(
                f"realized_volatility_ratio_{numerator}_{denominator}",
                f"Realized-volatility ratio — {numerator}/{denominator}m",
                "volatility_range_state",
                f"realized_volatility_{numerator} / realized_volatility_{denominator}",
                f"{denominator} one-minute returns",
                ("close",),
                denominator + 1,
                "Ratio of causal realized-volatility estimates",
                "[0, +inf)",
                "unsigned",
                validation_minimum=0.0,
            )
        )
    add(
        _spec(
            "current_range_over_atr",
            "Current range over ATR",
            "volatility_range_state",
            "(high_t - low_t) / ATR_20(t)",
            "20 bars",
            ("high", "low", "close"),
            20,
            "Current causal 20-bar ATR",
            "[0, +inf)",
            "unsigned",
            validation_minimum=0.0,
        )
    )
    add(
        _spec(
            "range_compression_ratio_5_30",
            "Range compression ratio — 5/30m",
            "volatility_range_state",
            "ATR_5(t) / ATR_30(t)",
            "30 bars",
            ("high", "low", "close"),
            30,
            "Ratio of causal ATR estimates",
            "[0, +inf)",
            "unsigned",
            validation_minimum=0.0,
        )
    )

    # Candle geometry and local price action (10).
    candle_specs = (
        (
            "signed_body_atr",
            "Signed body in ATR",
            "(close_t-open_t)/ATR_20(t)",
            ("open", "close", "high", "low"),
            20,
            "Current causal 20-bar ATR",
            "Unbounded",
            "signed",
            None,
            None,
        ),
        (
            "body_to_range",
            "Body-to-range ratio",
            "abs(close_t-open_t)/(high_t-low_t)",
            ("open", "high", "low", "close"),
            1,
            "Current bar range",
            "[0, 1]",
            "unsigned",
            0.0,
            1.0,
        ),
        (
            "upper_wick_to_range",
            "Upper-wick-to-range ratio",
            "(high_t-max(open_t,close_t))/(high_t-low_t)",
            ("open", "high", "low", "close"),
            1,
            "Current bar range",
            "[0, 1]",
            "unsigned",
            0.0,
            1.0,
        ),
        (
            "lower_wick_to_range",
            "Lower-wick-to-range ratio",
            "(min(open_t,close_t)-low_t)/(high_t-low_t)",
            ("open", "high", "low", "close"),
            1,
            "Current bar range",
            "[0, 1]",
            "unsigned",
            0.0,
            1.0,
        ),
        (
            "close_location_value",
            "Close-location value",
            "(2×close_t-high_t-low_t)/(high_t-low_t)",
            ("high", "low", "close"),
            1,
            "Current bar range",
            "[-1, 1]",
            "signed",
            -1.0,
            1.0,
        ),
        (
            "range_relative_to_previous_bar",
            "Range relative to previous bar",
            "range_t/range_(t-1)",
            ("high", "low"),
            2,
            "Previous completed-bar range",
            "[0, +inf)",
            "unsigned",
            0.0,
            None,
        ),
        (
            "two_bar_directional_balance",
            "Directional balance — 2 bars",
            "sum(price_change,2)/sum(abs(price_change),2)",
            ("close",),
            3,
            "Absolute completed price movement",
            "[-1, 1]",
            "signed",
            -1.0,
            1.0,
        ),
        (
            "three_bar_directional_balance",
            "Directional balance — 3 bars",
            "sum(price_change,3)/sum(abs(price_change),3)",
            ("close",),
            4,
            "Absolute completed price movement",
            "[-1, 1]",
            "signed",
            -1.0,
            1.0,
        ),
    )
    for name, display, formula, inputs, history, norm, expected, signed, vmin, vmax in candle_specs:
        add(
            _spec(
                name,
                display,
                "candle_geometry",
                formula,
                f"{history} bars",
                inputs,
                history,
                norm,
                expected,
                signed,
                validation_minimum=vmin,
                validation_maximum=vmax,
            )
        )
    add(
        _spec(
            "inside_bar",
            "Inside bar",
            "candle_geometry",
            "high_t <= high_(t-1) and low_t >= low_(t-1)",
            "2 bars",
            ("high", "low"),
            2,
            "None",
            "{False, True}",
            "boolean",
            "boolean",
        )
    )
    add(
        _spec(
            "outside_bar",
            "Outside bar",
            "candle_geometry",
            "high_t >= high_(t-1) and low_t <= low_(t-1)",
            "2 bars",
            ("high", "low"),
            2,
            "None",
            "{False, True}",
            "boolean",
            "boolean",
        )
    )

    # Volume, activity, and time-of-day normalization (10).
    volume_specs = (
        (
            "log_volume",
            "Log volume",
            "log1p(volume_t)",
            "1 bar",
            ("volume",),
            1,
            "log1p",
            "[0, +inf)",
            "unsigned",
            0.0,
            None,
            "No fitted parameters",
        ),
        (
            "relative_volume_20",
            "Relative volume — 20m",
            "volume_t/mean(volume,20)",
            "20 bars",
            ("volume",),
            20,
            "20-bar causal mean",
            "[0, +inf)",
            "unsigned",
            0.0,
            None,
            "No fitted parameters",
        ),
        (
            "relative_volume_60",
            "Relative volume — 60m",
            "volume_t/mean(volume,60)",
            "60 bars",
            ("volume",),
            60,
            "60-bar causal mean",
            "[0, +inf)",
            "unsigned",
            0.0,
            None,
            "No fitted parameters",
        ),
        (
            "volume_zscore_60",
            "Log-volume z-score — 60m",
            "(log1p(volume_t)-mean(log1p(volume),60))/std(log1p(volume),60)",
            "60 bars",
            ("volume",),
            60,
            "60-bar causal population z-score",
            "Unbounded",
            "signed",
            None,
            None,
            "No fitted parameters",
        ),
        (
            "volume_acceleration_5_20",
            "Volume acceleration — 5/20m",
            "mean(volume,5)/mean(volume,20)",
            "20 bars",
            ("volume",),
            20,
            "Ratio of causal rolling means",
            "[0, +inf)",
            "unsigned",
            0.0,
            None,
            "No fitted parameters",
        ),
        (
            "volume_per_tick_range",
            "Volume per tick of range",
            "volume_t/max((high_t-low_t)/0.10, non-zero)",
            "1 bar",
            ("volume", "high", "low"),
            1,
            "GC ticks",
            "[0, +inf)",
            "unsigned",
            0.0,
            None,
            "No fitted parameters",
        ),
        (
            "signed_volume_proxy",
            "Signed volume proxy",
            "close_location_value × relative_volume_20",
            "20 bars",
            ("high", "low", "close", "volume"),
            20,
            "OHLCV proxy; not aggressor-side volume",
            "Unbounded",
            "signed",
            None,
            None,
            "No fitted parameters",
        ),
        (
            "volume_price_alignment_10",
            "Volume-price alignment — 10m",
            "corr(log_return,log1p(volume),10)",
            "10 one-minute returns",
            ("close", "volume"),
            11,
            "Rolling Pearson correlation",
            "[-1, 1]",
            "signed",
            -1.0,
            1.0,
            "No fitted parameters",
        ),
        (
            "tod_log_volume_z",
            "Time-of-day log-volume z-score",
            "Development-only clock-bin z-score; Development uses leave-one-date-out moments",
            "Development reference",
            ("volume", "entry_session", "entry_timestamp_ny", "trade_date_ny"),
            0,
            "Session plus 15-minute clock bin",
            "Unbounded",
            "signed",
            None,
            None,
            "Development only; leave-one-date-out for Development rows",
        ),
        (
            "tod_relative_volume",
            "Time-of-day relative volume",
            "exp(log1p(volume_t)-Development clock-bin mean); Development is leave-one-date-out",
            "Development reference",
            ("volume", "entry_session", "entry_timestamp_ny", "trade_date_ny"),
            0,
            "Development geometric clock-bin baseline",
            "[0, +inf)",
            "unsigned",
            0.0,
            None,
            "Development only; leave-one-date-out for Development rows",
        ),
    )
    for (
        name,
        display,
        formula,
        lookback,
        inputs,
        history,
        norm,
        expected,
        signed,
        vmin,
        vmax,
        fit,
    ) in volume_specs:
        add(
            _spec(
                name,
                display,
                "volume_activity",
                formula,
                lookback,
                inputs,
                history,
                norm,
                expected,
                signed,
                fit_scope=fit,
                reset_boundary="continuity run and Development clock-bin contract"
                if name.startswith("tod_")
                else "continuity run",
                validation_minimum=vmin,
                validation_maximum=vmax,
            )
        )

    # VWAP, anchored location, and session state (10).
    vwap_specs = (
        (
            "distance_from_research_day_vwap_atr",
            "Distance from research-day VWAP",
            "(close_t-research_day_vwap_t)/ATR_20(t)",
            "Research day to t",
            "research day at 01:00 New York",
        ),
        (
            "distance_from_execution_session_vwap_atr",
            "Distance from execution-session VWAP",
            "(close_t-execution_session_vwap_t)/ATR_20(t)",
            "Execution session to t",
            "London 03:00 or New York 07:00 anchor",
        ),
        (
            "distance_from_rolling_vwap_20_atr",
            "Distance from rolling VWAP — 20m",
            "(close_t-rolling_vwap_20(t))/ATR_20(t)",
            "20 bars",
            "continuity run",
        ),
        (
            "distance_from_rolling_vwap_60_atr",
            "Distance from rolling VWAP — 60m",
            "(close_t-rolling_vwap_60(t))/ATR_20(t)",
            "60 bars",
            "continuity run",
        ),
        (
            "vwap_slope_5_atr",
            "Research-day VWAP slope — 5m",
            "(research_day_vwap_t-research_day_vwap_(t-5))/ATR_20(t)",
            "5 bars",
            "research day at 01:00 New York",
        ),
        (
            "vwap_slope_15_atr",
            "Research-day VWAP slope — 15m",
            "(research_day_vwap_t-research_day_vwap_(t-15))/ATR_20(t)",
            "15 bars",
            "research day at 01:00 New York",
        ),
        (
            "fraction_above_vwap_15",
            "Fraction above research-day VWAP — 15m",
            "mean(close_i>research_day_vwap_i,15)",
            "15 bars",
            "research day at 01:00 New York",
        ),
        (
            "distance_from_session_open_atr",
            "Distance from execution-session open",
            "(close_t-first_open_in_execution_session)/ATR_20(t)",
            "Execution session to t",
            "London 03:00 or New York 07:00 anchor",
        ),
        (
            "session_range_position",
            "Execution-session range position",
            "(close_t-session_low_to_t)/(session_high_to_t-session_low_to_t)",
            "Execution session to t",
            "London 03:00 or New York 07:00 anchor",
        ),
        (
            "session_range_over_atr",
            "Execution-session range over ATR",
            "(session_high_to_t-session_low_to_t)/ATR_20(t)",
            "Execution session to t",
            "London 03:00 or New York 07:00 anchor",
        ),
    )
    for name, display, formula, lookback, reset in vwap_specs:
        bounded = name in {"fraction_above_vwap_15", "session_range_position"}
        unsigned = name == "session_range_over_atr" or bounded
        add(
            _spec(
                name,
                display,
                "vwap_session_state",
                formula,
                lookback,
                ("open", "high", "low", "close", "volume", "ts_event_ny"),
                1
                if "session" in lookback.lower() or "research" in lookback.lower()
                else int(lookback.split()[0]) + (1 if "slope" in name else 0),
                "Current causal 20-bar ATR" if not bounded else "Causal anchored range/count",
                "[0, 1]"
                if bounded
                else ("[0, +inf)" if name == "session_range_over_atr" else "Unbounded"),
                "unsigned" if unsigned else "signed",
                reset_boundary=reset,
                validation_minimum=0.0 if unsigned else None,
                validation_maximum=1.0 if bounded else None,
            )
        )

    # Trend, persistence, efficiency, and mean-reversion state (11).
    for horizon in (15, 30, 60):
        add(
            _spec(
                f"efficiency_ratio_{horizon}",
                f"Efficiency ratio — {horizon}m",
                "trend_persistence",
                f"abs(close_t-close_(t-{horizon}))/sum(abs(close_i-close_(i-1)),{horizon})",
                f"{horizon} one-minute movements",
                ("close",),
                horizon + 1,
                "Absolute path length",
                "[0, 1]",
                "unsigned",
                validation_minimum=0.0,
                validation_maximum=1.0,
            )
        )
        add(
            _spec(
                f"normalized_ols_slope_{horizon}",
                f"Normalized OLS slope — {horizon}m",
                "trend_persistence",
                f"OLS slope of close on x=0..{horizon - 1}, divided by ATR_20(t)",
                f"{horizon} bars",
                ("close", "high", "low"),
                max(horizon, 20),
                "Current causal 20-bar ATR",
                "Unbounded",
                "signed",
            )
        )
    add(
        _spec(
            "ols_r_squared_30",
            "OLS R-squared — 30m",
            "trend_persistence",
            "R² from OLS of close on x=0..29",
            "30 bars",
            ("close",),
            30,
            "Rolling linear regression",
            "[0, 1]",
            "unsigned",
            validation_minimum=0.0,
            validation_maximum=1.0,
        )
    )
    add(
        _spec(
            "directional_persistence_15",
            "Directional persistence — 15m",
            "trend_persistence",
            "abs(sum(sign(one-minute price change),15))/15",
            "15 one-minute movements",
            ("close",),
            16,
            "Signed-movement count",
            "[0, 1]",
            "unsigned",
            validation_minimum=0.0,
            validation_maximum=1.0,
        )
    )
    add(
        _spec(
            "return_autocorrelation_15",
            "Return autocorrelation — 15m",
            "trend_persistence",
            "Pearson correlation of adjacent one-minute log returns within 15 returns",
            "15 one-minute returns",
            ("close",),
            16,
            "Rolling Pearson correlation",
            "[-1, 1]",
            "signed",
            validation_minimum=-1.0,
            validation_maximum=1.0,
        )
    )
    add(
        _spec(
            "return_sign_change_rate_30",
            "Return sign-change rate — 30m",
            "trend_persistence",
            "Mean(sign(return_i) != sign(return_(i-1))) over 29 adjacent pairs",
            "30 one-minute returns",
            ("close",),
            31,
            "Adjacent sign pairs",
            "[0, 1]",
            "unsigned",
            validation_minimum=0.0,
            validation_maximum=1.0,
        )
    )
    add(
        _spec(
            "choppiness_14",
            "Choppiness index — 14m",
            "trend_persistence",
            "100×log10(sum(TR,14)/(rolling_high_14-rolling_low_14))/log10(14)",
            "14 bars",
            ("high", "low", "close"),
            14,
            "Classical bounded choppiness scaling",
            "[0, 100]",
            "unsigned",
            validation_minimum=0.0,
            validation_maximum=100.0,
        )
    )

    # Session, clock, and calendar context (9; seven numeric plus two categorical).
    context_specs = (
        (
            "entry_session",
            "Entry session",
            "Known London/New York session of entry bar t+1",
            "category",
            "{London, New York}",
            "categorical",
        ),
        (
            "minute_from_execution_window_open",
            "Minute from execution-window open",
            "Entry minute minus 03:00 (London) or 07:00 (New York)",
            "Int16",
            "[0, 299]",
            "unsigned",
        ),
        (
            "session_progress_fraction",
            "Execution-window progress fraction",
            "minute_from_open/(window_minutes-1)",
            "float32",
            "[0, 1]",
            "unsigned",
        ),
        (
            "minutes_to_noon_entry_cutoff",
            "Minutes to noon entry cutoff",
            "12:00 New York minus entry timestamp t+1, in scheduled minutes",
            "Int16",
            "[1, 540]",
            "unsigned",
        ),
        (
            "minutes_to_1530_forced_exit",
            "Minutes to 15:30 forced exit",
            "Saved schedule minutes from entry timestamp t+1 to 15:30 New York",
            "Int16",
            "[1, 750]",
            "unsigned",
        ),
        (
            "new_york_minute_of_day",
            "New York minute of day",
            "60×entry_hour_ny+entry_minute_ny",
            "Int16",
            "[0, 1439]",
            "unsigned",
        ),
        (
            "time_of_day_sin",
            "Time-of-day sine",
            "sin(2π×New_York_minute_of_day/1440)",
            "float32",
            "[-1, 1]",
            "signed",
        ),
        (
            "time_of_day_cos",
            "Time-of-day cosine",
            "cos(2π×New_York_minute_of_day/1440)",
            "float32",
            "[-1, 1]",
            "signed",
        ),
        (
            "day_of_week",
            "Day of week",
            "Known New York weekday of entry timestamp t+1",
            "category",
            "{Monday, Tuesday, Wednesday, Thursday, Friday}",
            "categorical",
        ),
    )
    for name, display, formula, dtype, expected, signed in context_specs:
        vmin, vmax = (None, None)
        if expected.startswith("["):
            bounds = expected.strip("[]").split(",")
            vmin = float(bounds[0])
            try:
                vmax = float(bounds[1])
            except ValueError:
                vmax = None
        add(
            _spec(
                name,
                display,
                "session_clock_calendar",
                formula,
                "Known schedule at t",
                ("entry_timestamp_ny", "entry_session"),
                0,
                "Schedule/cyclical encoding",
                expected,
                signed,
                dtype,
                reset_boundary="Execution-window schedule",
                missing_policy="No missing values are expected for eligible observations.",
                validation_minimum=vmin,
                validation_maximum=vmax,
            )
        )

    # Predeclared experimental batch (6).
    experimental = (
        (
            "directional_energy_balance_15_exp",
            "Directional Energy Balance — 15m",
            "sum(sign(r_i)×r_i²×clip(relative_volume_20,0.5,2.0))/sum(r_i²×weight_i), 15 bars",
            "15 weighted returns",
            ("close", "volume"),
            35,
            "Energy-weighted bounded balance",
            "[-1, 1]",
            "signed",
            "Squared returns emphasize the bars contributing most to realized movement and test whether that energy was directionally coherent.",
            -1.0,
            1.0,
        ),
        (
            "wick_pressure_balance_10_exp",
            "Wick Pressure Balance — 10m",
            "weighted mean((lower_wick-upper_wick)/range,10) using clipped causal relative-volume weights; zero-range bars are neutral",
            "10 weighted bars",
            ("open", "high", "low", "close", "volume"),
            30,
            "Weighted bounded wick balance",
            "[-1, 1]",
            "signed",
            "Repeated tail rejection may contain information beyond one-bar wick geometry; this is not measured order flow or liquidity.",
            -1.0,
            1.0,
        ),
        (
            "compression_age_exp",
            "Compression Age",
            "Consecutive bars, capped at 60, where RV_5/RV_30<0.70 and ATR_5/ATR_30<0.75",
            "Run length after 30-return warm-up",
            ("high", "low", "close"),
            31,
            "Predeclared dual compression thresholds",
            "[0, 60]",
            "unsigned",
            "Duration distinguishes fresh compression from a prolonged stagnant regime without selecting thresholds from outcomes.",
            0.0,
            60.0,
        ),
        (
            "vwap_elasticity_30_exp",
            "VWAP Elasticity — 30m",
            "Rolling OLS slope: response return_i in bps, predictor research-day VWAP distance_(i-1) in ATR units, 30 valid pairs",
            "30 predictor-response pairs",
            ("close", "high", "low", "volume", "ts_event_ny"),
            51,
            "Response bps per lagged ATR-distance unit",
            "Unbounded",
            "signed",
            "The local slope estimates whether completed price responses have recently reverted toward or continued away from research-day VWAP.",
            None,
            None,
        ),
        (
            "liquidity_vacuum_score_exp",
            "Liquidity Vacuum Proxy",
            "current_range_over_atr×abs(close_location_value)/sqrt(max(relative_volume_20,0.25))",
            "20 bars",
            ("high", "low", "close", "volume"),
            20,
            "Range, edge-close strength, and floored activity",
            "[0, +inf)",
            "unsigned",
            "A large edge-closing bar on relatively light activity may proxy for low resistance; it is not measured depth or actual liquidity.",
            0.0,
            None,
        ),
        (
            "pullback_tension_5_30_exp",
            "Pullback Tension Across Time Scales — 5/30m",
            "sign(return_30m_atr)×min(abs(return_5m_atr),abs(return_30m_atr)) when the signs oppose; otherwise 0",
            "30 bars",
            ("close", "high", "low"),
            31,
            "Current causal 20-bar ATR",
            "Unbounded",
            "signed",
            "The feature isolates a short counter-move inside a broader completed directional displacement without exhaustive interactions.",
            None,
            None,
        ),
    )
    for (
        name,
        display,
        formula,
        lookback,
        inputs,
        history,
        norm,
        expected,
        signed,
        rationale,
        vmin,
        vmax,
    ) in experimental:
        add(
            _spec(
                name,
                display,
                "experimental_hypothesis",
                formula,
                lookback,
                inputs,
                history,
                norm,
                expected,
                signed,
                "Int8" if name == "compression_age_exp" else "float32",
                experimental=True,
                creative_rationale=rationale,
                validation_minimum=vmin,
                validation_maximum=vmax,
            )
        )

    return tuple(specs)


FEATURE_SPECS = _build_feature_specs()
FEATURE_NAMES = tuple(spec.feature_name for spec in FEATURE_SPECS)
CORE_FEATURE_NAMES = tuple(spec.feature_name for spec in FEATURE_SPECS if not spec.is_experimental)
EXPERIMENTAL_FEATURE_NAMES = tuple(
    spec.feature_name for spec in FEATURE_SPECS if spec.is_experimental
)


def feature_registry_frame(specs: Iterable[FeatureSpec] = FEATURE_SPECS) -> pd.DataFrame:
    """Return a stable, machine-readable registry table."""

    records = []
    for spec in specs:
        record = asdict(spec)
        record["input_columns"] = "|".join(spec.input_columns)
        records.append(record)
    registry = pd.DataFrame.from_records(records)
    registry["feature_tier"] = pd.Categorical(
        registry["feature_tier"], categories=["core", "experimental"], ordered=True
    )
    registry["family"] = registry["family"].astype("category")
    registry["is_experimental"] = registry["is_experimental"].astype(bool)
    return registry


def validate_registry(registry: pd.DataFrame | None = None) -> None:
    """Raise on structural registry defects before any feature is calculated."""

    frame = feature_registry_frame() if registry is None else registry
    if frame["feature_name"].duplicated().any():
        raise ValueError("Feature registry contains duplicate feature names.")
    if "*" in "".join(frame["feature_name"].astype(str)):
        raise ValueError("Saved feature names must not contain literal asterisks.")
    experimental = frame["is_experimental"].astype(bool)
    if not frame.loc[experimental, "display_name"].astype(str).str.startswith("* ").all():
        raise ValueError("Every experimental display name must begin with an asterisk.")
    if frame.loc[~experimental, "display_name"].astype(str).str.startswith("* ").any():
        raise ValueError("Core display names must not be marked experimental.")
    if (
        not frame.loc[experimental, "feature_name"]
        .astype(str)
        .str.endswith(EXPERIMENTAL_SUFFIX)
        .all()
    ):
        raise ValueError("Experimental saved names must use the _exp suffix.")
    if not frame.loc[experimental, "creative_rationale"].astype(str).str.len().gt(0).all():
        raise ValueError("Experimental features require a written creative rationale.")
    if len(frame) != 85 or int(experimental.sum()) != 6:
        raise ValueError(
            f"Frozen Phase 1 registry must contain 79 core and 6 experimental features; got {len(frame)} rows."
        )


validate_registry()
