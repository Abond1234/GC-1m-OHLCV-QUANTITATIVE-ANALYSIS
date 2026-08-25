"""Frozen Stage 1 registry for the Project One Tsay research line.

This module contains declarations only.  It must not import or inspect market
data, labels, or prior research outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

TSAY_RESEARCH_ID: Final = "tsay_feature_research"
TSAY_VERSION: Final = "v1"
TSAY_SEED: Final = 20260824
STATIONARY_BOOTSTRAP_REPLICATES: Final = 2_000
STATIONARY_BOOTSTRAP_RESTART_PROBABILITY: Final = 0.20
TICK_SIZE: Final = 0.10
PRIMARY_HORIZON_MINUTES: Final = 60
DIAGNOSTIC_HORIZON_MINUTES: Final = 30
REGISTERED_HORIZONS: Final = (5, 15, 30, 60, 120, 180)
SESSIONS: Final = ("London", "New York")

OVERLAP_DISPOSITIONS: Final = (
    "NOVEL_IMPLEMENTATION",
    "RELATED_BUT_MATERIALLY_DISTINCT",
    "EXACT_EXISTING_IMPLEMENTATION_REUSED_AND_REEVALUATED",
    "EXACT_EXISTING_LOCKED_VERDICT_REUSED",
    "CONTEXT_ONLY_NOT_A_HYPOTHESIS",
)


@dataclass(frozen=True)
class TsayFeatureSpec:
    """One immutable logical candidate and its physical implementation contract."""

    logical_id: str
    feature_name: str
    display_name: str
    role: str
    expected_orientation: str
    expected_sign: int
    source_chapter: int
    source_pdf_pages: str
    formula: str
    parameters: str
    lookback: str
    minimum_history: int
    output_dtype: str
    computation_dtype: str
    validation_minimum: float | None
    validation_maximum: float | None
    overlap_disposition: str
    related_existing_features: tuple[str, ...]
    partial_information_controls: tuple[str, ...]
    companion_columns: tuple[str, ...] = ()
    implementation_note: str = ""


def _spec(
    logical_id: str,
    feature_name: str,
    display_name: str,
    role: str,
    expected_orientation: str,
    expected_sign: int,
    source_chapter: int,
    source_pdf_pages: str,
    formula: str,
    parameters: str,
    lookback: str,
    minimum_history: int,
    validation_minimum: float | None,
    validation_maximum: float | None,
    related_existing_features: tuple[str, ...],
    partial_information_controls: tuple[str, ...],
    companion_columns: tuple[str, ...] = (),
    implementation_note: str = "",
) -> TsayFeatureSpec:
    return TsayFeatureSpec(
        logical_id=logical_id,
        feature_name=feature_name,
        display_name=display_name,
        role=role,
        expected_orientation=expected_orientation,
        expected_sign=expected_sign,
        source_chapter=source_chapter,
        source_pdf_pages=source_pdf_pages,
        formula=formula,
        parameters=parameters,
        lookback=lookback,
        minimum_history=minimum_history,
        output_dtype="float32",
        computation_dtype="float64",
        validation_minimum=validation_minimum,
        validation_maximum=validation_maximum,
        overlap_disposition="RELATED_BUT_MATERIALLY_DISTINCT",
        related_existing_features=related_existing_features,
        partial_information_controls=partial_information_controls,
        companion_columns=companion_columns,
        implementation_note=implementation_note,
    )


TSAY_FEATURE_SPECS: Final = (
    _spec(
        "T01",
        "tsay_ar5_cumulative_forecast_60_from_120_bps",
        "Rolling AR(5) cumulative 60-minute return forecast",
        "directional",
        "positive",
        1,
        2,
        "64-84",
        "Fit r_i=alpha+sum(phi_j*r_(i-j),j=1..5) on 120 response rows; "
        "recursively forecast h=1..60 from [r_t,...,r_(t-4)] and sum the forecasts.",
        "W=120;m=5;forecast_horizon=60;rank=full;condition<=1e12;"
        "max_companion_eigenvalue<0.999;no_regularization",
        "126 consecutive closes; 120 complete response rows",
        126,
        None,
        None,
        (
            "return_1m_bps",
            "normalized_ols_slope_30",
            "return_autocorrelation_15",
            "efficiency_ratio_30",
            "return_acf_energy_60",
        ),
        ("efficiency_ratio_30", "return_acf_energy_60"),
    ),
    _spec(
        "T02",
        "tsay_arch_lm_r2_120_l5",
        "Rolling ARCH-LM dependence state",
        "opportunity",
        "positive",
        1,
        3,
        "141-143",
        "From T01's 120 in-window AR residuals regress e_i^2 on an intercept "
        "and five lags over 115 effective rows; return ordinary R^2.",
        "W=120;m=5;effective_rows=115;LM=115*R2 diagnostic only;chi2_df=5",
        "Shared 120-row AR fit plus five residual-square lags",
        126,
        0.0,
        1.0,
        (
            "atr_ratio_20_60",
            "realized_volatility_ratio_15_60",
            "volatility_of_volatility_60",
            "return_acf_energy_60",
        ),
        (
            "realized_volatility_ratio_15_60",
            "volatility_of_volatility_60",
            "return_acf_energy_60",
        ),
        implementation_note=(
            "Project One adaptation: a dependence state, not a conditional-variance forecast."
        ),
    ),
    _spec(
        "T03",
        "tsay_roll_spread_proxy_120_ticks",
        "Roll implied-spread proxy",
        "opportunity",
        "positive",
        1,
        5,
        "262-264",
        "gamma1=(1/119)*sum((dC_i-mean_dC)*(dC_(i-1)-mean_dC),i=2..120); "
        "if gamma1<0 return 2*sqrt(-gamma1)/0.10, otherwise missing.",
        "W=120;sample_lag1_autocovariance_denominator=119;tick_size=0.10",
        "120 consecutive close changes",
        121,
        0.0,
        None,
        (
            "amihud_illiquidity_60",
            "volume_per_tick_range",
            "relative_volume_60",
        ),
        ("amihud_illiquidity_60", "volume_per_tick_range", "relative_volume_60"),
        companion_columns=("tsay_roll_spread_identified_120",),
        implementation_note=(
            "Companion states: 1 IDENTIFIED, 0 VALID_NOT_IDENTIFIED, -1 UNAVAILABLE. "
            "This is an OHLCV proxy, not an observed spread or executable cost."
        ),
    ),
    _spec(
        "T04",
        "tsay_one_minute_close_zero_change_fraction_60",
        "Aggregated one-minute close zero-change fraction",
        "opportunity",
        "negative",
        -1,
        5,
        "264-271",
        "mean(d_i == 0) over the last 60 completed GC tick changes.",
        "W=60;exact_tick_change_zero",
        "60 consecutive close changes",
        61,
        0.0,
        1.0,
        (
            "relative_volume_20",
            "return_sign_change_rate_30",
            "amihud_illiquidity_60",
        ),
        ("relative_volume_20", "return_sign_change_rate_30", "amihud_illiquidity_60"),
        implementation_note="Aggregated-bar analogue, not a transaction no-change probability.",
    ),
    _spec(
        "T05",
        "tsay_loss_es_120_atr",
        "Empirical downside expected shortfall",
        "opportunity",
        "positive",
        1,
        7,
        "366-369",
        "For L_i=-d_i, deterministically sort 120 values, average the largest 12, "
        "and divide by current ATRticks=atr_20/0.10.",
        "W=120;tail_fraction=0.10;tail_count=12;tick_size=0.10;no_interpolation",
        "120 consecutive close changes and current positive ATR20",
        121,
        None,
        None,
        (
            "realized_semivariance_balance_60",
            "ret_outlier_fraction_60",
            "bipower_jump_ratio_60",
        ),
        (
            "realized_semivariance_balance_60",
            "ret_outlier_fraction_60",
            "bipower_jump_ratio_60",
        ),
    ),
    _spec(
        "T06",
        "tsay_es_tail_balance_120",
        "Empirical gain/loss expected-shortfall balance",
        "directional",
        "positive",
        1,
        7,
        "366-369",
        "Let ES_gain be the mean of the largest 12 d_i and ES_loss the mean of the "
        "largest 12 -d_i; return (ES_gain-ES_loss)/(|ES_gain|+|ES_loss|), with zero "
        "for an exactly zero denominator.",
        "W=120;tail_fraction=0.10;tail_count=12;no_interpolation",
        "120 consecutive close changes",
        121,
        -1.0,
        1.0,
        (
            "ret_tail_balance_60",
            "realized_semivariance_balance_60",
            "ordered_draw_balance_30",
        ),
        ("ret_tail_balance_60", "realized_semivariance_balance_60", "ordered_draw_balance_30"),
    ),
    _spec(
        "T07",
        "tsay_volume_lead_return_impulse_120_l5",
        "Volume-lead return impulse",
        "directional",
        "positive",
        1,
        8,
        "417-423",
        "For k=1..5 compute Pearson corr(r_i,zV_(i-k)) across the exact last 120 "
        "response timestamps; return mean(rho_k)*zV_t.",
        "W=120;m=5;clock_bin=15m;prior_dates>=30;prior_observations>=300;"
        "sample_std;all_lag_variances_positive",
        "Exact 125-timestamp same-date and same-continuity-run span",
        125,
        None,
        None,
        (
            "lagged_volume_return_spearman_30",
            "volume_price_alignment_10",
            "signed_volume_proxy",
        ),
        ("lagged_volume_return_spearman_30", "volume_price_alignment_10"),
        companion_columns=("tsay_clock_z_volume",),
    ),
    _spec(
        "T08",
        "tsay_range_state_lead_absreturn_impulse_120_l5",
        "Clock-adjusted range-state lead impulse",
        "opportunity",
        "positive",
        1,
        8,
        "417-423",
        "For k=1..5 compute Pearson corr(abs(r_i),zG_(i-k)) across the exact last 120 "
        "response timestamps; return sqrt(mean(rho_k^2))*zG_t.",
        "W=120;m=5;clock_bin=15m;prior_dates>=30;prior_observations>=300;"
        "sample_std;all_lag_variances_positive",
        "Exact 125-timestamp same-date and same-continuity-run span",
        125,
        None,
        None,
        (
            "range_volume_spearman_30",
            "realized_volatility_15",
            "volatility_of_volatility_60",
            "current_range_over_atr",
        ),
        ("range_volume_spearman_30", "realized_volatility_15", "volatility_of_volatility_60"),
        companion_columns=("tsay_clock_z_range",),
    ),
    _spec(
        "T09",
        "tsay_extreme_cluster_ratio_120_q90",
        "Extreme-return cluster ratio",
        "opportunity",
        "negative",
        -1,
        7,
        "401-408",
        "Use the 108th one-based order statistic of 120 abs(d_i) values as the fixed "
        "threshold; mark strict exceedances and return clusters/exceedances.",
        "W=120;q=0.90;order_statistic=108;strict_exceedance",
        "120 consecutive close changes",
        121,
        0.0,
        1.0,
        (
            "ret_outlier_fraction_60",
            "return_acf_energy_60",
            "bipower_jump_ratio_60",
        ),
        ("ret_outlier_fraction_60", "return_acf_energy_60", "bipower_jump_ratio_60"),
        implementation_note="Runs-based cluster-ratio proxy, not a fully estimated extremal index.",
    ),
)

TSAY_LOGICAL_TO_PHYSICAL: Final = {
    spec.logical_id: spec.feature_name for spec in TSAY_FEATURE_SPECS
}
TSAY_FEATURE_COLUMNS: Final = tuple(spec.feature_name for spec in TSAY_FEATURE_SPECS)
TSAY_CONFIRMATORY_FAMILY_COUNT: Final = len(TSAY_FEATURE_SPECS)
TSAY_PRIMARY_TEST_COUNT: Final = TSAY_CONFIRMATORY_FAMILY_COUNT * len(SESSIONS)

DIRECTIONAL_ANCHOR_BASE: Final = (
    "return_5m_atr",
    "return_15m_atr",
    "return_30m_atr",
    "normalized_ols_slope_30",
    "close_location_value",
    "signed_volume_proxy",
    "distance_from_research_day_vwap_atr",
    "return_autocorrelation_15",
)
OPPORTUNITY_ANCHOR_BASE: Final = (
    "atr_20",
    "atr_ratio_20_60",
    "atr_ratio_5_20",
    "current_range_over_atr",
    "minute_from_execution_window_open",
    "minutes_to_1530_forced_exit",
    "efficiency_ratio_15",
    "efficiency_ratio_30",
    "efficiency_ratio_60",
    "ols_r_squared_30",
    "choppiness_14",
    "return_sign_change_rate_30",
    "relative_volume_20",
    "volume_acceleration_5_20",
    "vwap_elasticity_30_exp",
)

D1_INPUTS_BY_SESSION: Final = {
    "London": DIRECTIONAL_ANCHOR_BASE,
    "New York": DIRECTIONAL_ANCHOR_BASE + ("lagged_volume_return_spearman_30",),
}
O1_INPUTS_BY_SESSION: Final = {
    "London": OPPORTUNITY_ANCHOR_BASE + ("return_acf_energy_60", "range_volume_spearman_30"),
    "New York": OPPORTUNITY_ANCHOR_BASE + ("volume_profile_slope_30",),
}
D2_INPUTS: Final = tuple(TSAY_LOGICAL_TO_PHYSICAL[key] for key in ("T01", "T06", "T07"))
ROLL_MODEL_STATE_COLUMNS: Final = (
    "tsay_roll_spread_state_valid_not_identified_120",
    "tsay_roll_spread_state_unavailable_120",
)
O2_INPUTS: Final = (
    tuple(TSAY_LOGICAL_TO_PHYSICAL[key] for key in ("T02", "T03", "T04", "T05", "T08", "T09"))
    + ROLL_MODEL_STATE_COLUMNS
)


def _ordered_union(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for group in groups for value in group))


D3_INPUTS_BY_SESSION: Final = {
    session: _ordered_union(D1_INPUTS_BY_SESSION[session], D2_INPUTS) for session in SESSIONS
}
O3_INPUTS_BY_SESSION: Final = {
    session: _ordered_union(O1_INPUTS_BY_SESSION[session], O2_INPUTS) for session in SESSIONS
}
QUANTILE_INPUTS_BY_SESSION: Final = {
    session: _ordered_union(D3_INPUTS_BY_SESSION[session], O3_INPUTS_BY_SESSION[session])
    for session in SESSIONS
}
D4_INPUTS: Final = ("return_1m_bps", "tsay_clock_z_range", "tsay_clock_z_volume")
O5_INPUTS: Final = (
    "atr_20",
    "tsay_kalman_next_prior",
    "tsay_kalman_standardized_innovation",
    "tsay_kalman_gain",
    "tsay_kalman_next_state_std",
)

MODEL_LADDER: Final = {
    "D0": ("zero_forecast", "development_training_mean_forecast"),
    "D1": D1_INPUTS_BY_SESSION,
    "D2": D2_INPUTS,
    "D3": D3_INPUTS_BY_SESSION,
    "D4": D4_INPUTS,
    "D5": QUANTILE_INPUTS_BY_SESSION,
    "D6": D3_INPUTS_BY_SESSION,
    "O0": ("atr_20",),
    "O1": O1_INPUTS_BY_SESSION,
    "O2": O2_INPUTS,
    "O3": O3_INPUTS_BY_SESSION,
    "O4": QUANTILE_INPUTS_BY_SESSION,
    "O5": O5_INPUTS,
    "E0": ("development_training_prevalence",),
    "E1": O1_INPUTS_BY_SESSION,
    "E2": O2_INPUTS,
    "E3": O3_INPUTS_BY_SESSION,
}

RIDGE_ALPHA_GRID: Final = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
LOGISTIC_C_GRID: Final = (0.01, 0.1, 1.0, 10.0)
QUANTILE_ALPHA_GRID: Final = (1e-4, 1e-3, 1e-2)
QUANTILE_LEVELS: Final = (0.10, 0.50, 0.90)

TARGET_COLUMNS: Final = (
    "forward_return_60_atr",
    "forward_return_30_atr",
    "future_range_60_atr",
    "expansion_label_60",
)
SECONDARY_TICK_COLUMNS: Final = ("forward_return_60_ticks", "future_range_60_ticks")


def registry_records() -> list[dict[str, object]]:
    """Return a JSON-serializable copy of the immutable registry."""

    return [asdict(spec) for spec in TSAY_FEATURE_SPECS]


def validate_tsay_registry() -> None:
    """Fail closed if the frozen registry has drifted."""

    logical_ids = tuple(spec.logical_id for spec in TSAY_FEATURE_SPECS)
    names = tuple(spec.feature_name for spec in TSAY_FEATURE_SPECS)
    if logical_ids != tuple(f"T{index:02d}" for index in range(1, 10)):
        raise ValueError(f"Unexpected logical registry order: {logical_ids}")
    if len(names) != len(set(names)):
        raise ValueError("Tsay physical feature names must be unique.")
    if any(spec.expected_sign not in (-1, 1) for spec in TSAY_FEATURE_SPECS):
        raise ValueError("Every Tsay feature requires a fixed orientation sign.")
    if any(spec.overlap_disposition not in OVERLAP_DISPOSITIONS for spec in TSAY_FEATURE_SPECS):
        raise ValueError("Invalid overlap disposition.")
    if TSAY_CONFIRMATORY_FAMILY_COUNT != 9 or TSAY_PRIMARY_TEST_COUNT != 18:
        raise ValueError("The frozen confirmatory family must contain 9 features and 18 tests.")
    if set(TARGET_COLUMNS) & set(TSAY_FEATURE_COLUMNS):
        raise ValueError("Outcome columns cannot appear in the feature registry.")


validate_tsay_registry()
