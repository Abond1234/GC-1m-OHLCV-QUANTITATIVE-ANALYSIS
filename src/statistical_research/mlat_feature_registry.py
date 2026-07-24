"""Frozen registry for the MLAT Feature Research v1 predictor batch.

The registry is outcome-free and immutable.  Feature construction and saved
artifacts must obtain predictor names and ordering from this module rather than
infer them by excluding metadata or label columns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Iterable

import numpy as np
import pandas as pd

MLAT_IMPLEMENTATION_VERSION = "1.0.0"
MLAT_FEATURE_TIER = "frozen_v1"
MLAT_STATUS = "FROZEN_V1"

MLAT_FEATURE_NAMES = (
    "bollinger_zscore_20",
    "bollinger_bandwidth_20",
    "cutler_rsi_14",
    "chaikin_money_flow_20",
    "amihud_illiquidity_60",
    "parkinson_volatility_30",
    "rogers_satchell_volatility_30",
    "realized_semivariance_balance_60",
    "bipower_jump_ratio_60",
    "variance_ratio_60_5",
    "return_sign_entropy_60",
    "volatility_of_volatility_60",
)

# These are the twelve metadata columns in the existing statistical feature
# matrix.  MLAT matrices append MLAT_FEATURE_NAMES in the exact order above.
MLAT_METADATA_COLUMNS = (
    "observation_id",
    "decision_timestamp_utc",
    "decision_timestamp_ny",
    "entry_timestamp_utc",
    "entry_timestamp_ny",
    "trade_date_ny",
    "research_partition",
    "product",
    "symbol",
    "active_symbol",
    "instrument_id",
    "continuous_segment_id",
)

MLAT_CAUSAL_INPUT_COLUMNS = ("open", "high", "low", "close", "volume")

_CONTINUITY_REQUIREMENTS = (
    "Consecutive one-minute GC bars with unchanged product, selected contract, "
    "instrument, and continuous segment; both sides of every boundary must be "
    "tradable, outside roll windows, and outside low-liquidity warnings."
)
_RESET_BOUNDARY = (
    "Reset at missing-minute, product, selected-contract, instrument, "
    "continuous-segment, roll, tradability, or liquidity boundary."
)
_AVAILABILITY = "Close of completed decision bar t"


@dataclass(frozen=True, slots=True)
class MLATFeatureSpec:
    """Immutable machine-readable specification for one frozen MLAT feature."""

    feature_name: str
    display_name: str
    feature_family: str
    feature_tier: str
    hypothesis_id: str
    formula_id: str
    source_chapter: str
    source_pdf_page: str
    source_type: str
    formula_or_definition: str
    economic_rationale: str
    target_family: str
    lookback: str
    minimum_history: int
    input_columns: tuple[str, ...]
    availability_timestamp: str
    reset_boundary: str
    continuity_requirements: str
    fitted_parameters: str
    fit_scope: str
    expected_range: str
    missing_value_policy: str
    normalization_method: str
    output_dtype: str
    existing_feature_overlap: str
    leakage_risk: str
    implementation_version: str
    is_experimental: bool
    status: str
    validation_minimum: float | None = None
    validation_maximum: float | None = None


def _spec(
    *,
    feature_name: str,
    display_name: str,
    feature_family: str,
    hypothesis_number: int,
    formula_number: int,
    source_chapter: str,
    source_pdf_page: str,
    source_type: str,
    formula: str,
    rationale: str,
    target_family: str,
    lookback: str,
    minimum_history: int,
    input_columns: tuple[str, ...],
    expected_range: str,
    missing_value_policy: str,
    normalization_method: str,
    existing_feature_overlap: str,
    validation_minimum: float | None = None,
    validation_maximum: float | None = None,
) -> MLATFeatureSpec:
    return MLATFeatureSpec(
        feature_name=feature_name,
        display_name=display_name,
        feature_family=feature_family,
        feature_tier=MLAT_FEATURE_TIER,
        hypothesis_id=f"MLAT-H{hypothesis_number:03d}",
        formula_id=f"MLAT-F{formula_number:03d}",
        source_chapter=source_chapter,
        source_pdf_page=source_pdf_page,
        source_type=source_type,
        formula_or_definition=formula,
        economic_rationale=rationale,
        target_family=target_family,
        lookback=lookback,
        minimum_history=minimum_history,
        input_columns=input_columns,
        availability_timestamp=_AVAILABILITY,
        reset_boundary=_RESET_BOUNDARY,
        continuity_requirements=_CONTINUITY_REQUIREMENTS,
        fitted_parameters="None",
        fit_scope="No fitted parameters",
        expected_range=expected_range,
        missing_value_policy=missing_value_policy,
        normalization_method=normalization_method,
        output_dtype="float32",
        existing_feature_overlap=existing_feature_overlap,
        leakage_risk="LOW",
        implementation_version=MLAT_IMPLEMENTATION_VERSION,
        is_experimental=True,
        status=MLAT_STATUS,
        validation_minimum=validation_minimum,
        validation_maximum=validation_maximum,
    )


def _build_specs() -> tuple[MLATFeatureSpec, ...]:
    return (
        _spec(
            feature_name="bollinger_zscore_20",
            display_name="Bollinger z-score (20)",
            feature_family="price_location",
            hypothesis_number=1,
            formula_number=2,
            source_chapter="4 / Appendix",
            source_pdf_page="131-133; 740-742",
            source_type="MLAT-ADAPTED",
            formula="(close_t - population_mean_20(close)) / population_std_20(close)",
            rationale=(
                "Standardized local price displacement may identify reversal or "
                "continuation states."
            ),
            target_family="direction",
            lookback="20 complete bars",
            minimum_history=20,
            input_columns=("close",),
            expected_range="Unbounded",
            missing_value_policy="Null until 20 complete bars; zero standard deviation is null.",
            normalization_method="Trailing 20-bar population mean and population standard deviation",
            existing_feature_overlap="rolling_range_position_15; normalized_ols_slope_30",
        ),
        _spec(
            feature_name="bollinger_bandwidth_20",
            display_name="Bollinger bandwidth (20)",
            feature_family="volatility_expansion",
            hypothesis_number=2,
            formula_number=3,
            source_chapter="Appendix",
            source_pdf_page="740-742",
            source_type="MLAT-DIRECT",
            formula="4 * population_std_20(close) / population_mean_20(close)",
            rationale="Narrow bands may precede expansion while wide bands may mean-revert.",
            target_family="expansion",
            lookback="20 complete bars",
            minimum_history=20,
            input_columns=("close",),
            expected_range="[0, +inf)",
            missing_value_policy="Null until 20 complete bars; zero or non-positive mean is null.",
            normalization_method="Four-sigma width divided by trailing 20-bar mean close",
            existing_feature_overlap="range_compression_ratio_5_30; atr_ratio_5_20; atr_ratio_20_60",
            validation_minimum=0.0,
        ),
        _spec(
            feature_name="cutler_rsi_14",
            display_name="Cutler RSI (14 changes)",
            feature_family="momentum_state",
            hypothesis_number=3,
            formula_number=4,
            source_chapter="4",
            source_pdf_page="132",
            source_type="MLAT-ADAPTED",
            formula=(
                "100 * sum_14(max(delta_close, 0)) / "
                "(sum_14(max(delta_close, 0)) + sum_14(max(-delta_close, 0)))"
            ),
            rationale=(
                "A bounded gain/loss balance may distinguish exhausted from persistent moves."
            ),
            target_family="direction / risk state",
            lookback="14 close changes from 15 complete bars",
            minimum_history=15,
            input_columns=("close",),
            expected_range="[0, 100]",
            missing_value_policy="Null until 14 complete changes; zero gains-plus-losses is null.",
            normalization_method="Simple rolling Cutler ratio; no Wilder recursion",
            existing_feature_overlap="return momentum; directional_persistence_15",
            validation_minimum=0.0,
            validation_maximum=100.0,
        ),
        _spec(
            feature_name="chaikin_money_flow_20",
            display_name="Chaikin money flow (20)",
            feature_family="volume_price_state",
            hypothesis_number=4,
            formula_number=5,
            source_chapter="Appendix",
            source_pdf_page="752-753",
            source_type="MLAT-ADAPTED",
            formula=("sum_20(volume * ((2*close-high-low)/(high-low))) / sum_20(volume)"),
            rationale=(
                "Close location weighted by volume may reveal pressure not present in either "
                "input alone."
            ),
            target_family="direction / risk state",
            lookback="20 complete bars",
            minimum_history=20,
            input_columns=("high", "low", "close", "volume"),
            expected_range="[-1, 1]",
            missing_value_policy=(
                "Null until 20 complete bars; zero-range bars contribute zero; "
                "zero total volume is null."
            ),
            normalization_method="Rolling money-flow volume divided by rolling volume",
            existing_feature_overlap="close_location_value; signed_volume_proxy; volume_price_alignment_10",
            validation_minimum=-1.0,
            validation_maximum=1.0,
        ),
        _spec(
            feature_name="amihud_illiquidity_60",
            display_name="Amihud illiquidity adaptation (60 returns)",
            feature_family="liquidity_risk",
            hypothesis_number=5,
            formula_number=6,
            source_chapter="20 / Appendix",
            source_pdf_page="656; 752",
            source_type="MLAT-ADAPTED",
            formula="1e9 * mean_60(abs(log_return) / (close * volume))",
            rationale="Movement per unit activity may identify fragile liquidity-vacuum states.",
            target_family="expansion / risk state",
            lookback="60 close-to-close returns from 61 complete bars",
            minimum_history=61,
            input_columns=("close", "volume"),
            expected_range="[0, +inf)",
            missing_value_policy=(
                "Null until 60 complete returns; zero volume or non-positive close is null."
            ),
            normalization_method="One-billion-scaled mean inverse notional-activity proxy",
            existing_feature_overlap="volume_per_tick_range; liquidity_vacuum_score_exp",
            validation_minimum=0.0,
        ),
        _spec(
            feature_name="parkinson_volatility_30",
            display_name="Parkinson range volatility (30)",
            feature_family="range_volatility",
            hypothesis_number=6,
            formula_number=7,
            source_chapter="9",
            source_pdf_page="297-301",
            source_type="PROJECT-ORIGINAL-EXTENSION",
            formula="1e4 * sqrt(mean_30(log(high/low)^2) / (4*log(2)))",
            rationale=(
                "High-low information may estimate latent volatility more efficiently than "
                "close-only realized volatility."
            ),
            target_family="volatility / expansion",
            lookback="30 complete bars",
            minimum_history=30,
            input_columns=("high", "low"),
            expected_range="[0, +inf)",
            missing_value_policy="Null until 30 complete bars; non-positive high or low is null.",
            normalization_method="Unannualized one-minute estimator in basis points",
            existing_feature_overlap="ATR; realized volatility; bar range",
            validation_minimum=0.0,
        ),
        _spec(
            feature_name="rogers_satchell_volatility_30",
            display_name="Rogers-Satchell volatility (30)",
            feature_family="range_volatility",
            hypothesis_number=7,
            formula_number=8,
            source_chapter="9",
            source_pdf_page="297-301",
            source_type="PROJECT-ORIGINAL-EXTENSION",
            formula=(
                "1e4 * sqrt(mean_30(log(high/open)*log(high/close) + log(low/open)*log(low/close)))"
            ),
            rationale=(
                "A drift-robust OHLC estimator may retain information beyond ATR and "
                "close-only realized volatility."
            ),
            target_family="volatility / expansion",
            lookback="30 complete bars",
            minimum_history=30,
            input_columns=("open", "high", "low", "close"),
            expected_range="[0, +inf)",
            missing_value_policy=(
                "Null until 30 complete bars; non-positive OHLC is null; tiny negative "
                "round-off is zero and material negative estimates are null."
            ),
            normalization_method="Unannualized one-minute estimator in basis points",
            existing_feature_overlap="ATR; realized volatility; bar range",
            validation_minimum=0.0,
        ),
        _spec(
            feature_name="realized_semivariance_balance_60",
            display_name="Realized semivariance balance (60 returns)",
            feature_family="return_asymmetry",
            hypothesis_number=8,
            formula_number=9,
            source_chapter="5 / 9",
            source_pdf_page="169-178; 297-301",
            source_type="PROJECT-ORIGINAL-EXTENSION",
            formula=("(sum_60(r^2 * 1[r>0]) - sum_60(r^2 * 1[r<0])) / sum_60(r^2)"),
            rationale=(
                "Asymmetry in recent signed variation may distinguish downside and upside "
                "risk states."
            ),
            target_family="risk state / direction",
            lookback="60 close-to-close returns from 61 complete bars",
            minimum_history=61,
            input_columns=("close",),
            expected_range="[-1, 1]",
            missing_value_policy="Null until 60 complete returns; zero total variation is null.",
            normalization_method="Signed semivariance difference divided by total realized variance",
            existing_feature_overlap="directional_energy_balance_15_exp",
            validation_minimum=-1.0,
            validation_maximum=1.0,
        ),
        _spec(
            feature_name="bipower_jump_ratio_60",
            display_name="Bipower jump ratio (60 products)",
            feature_family="jump_risk",
            hypothesis_number=9,
            formula_number=10,
            source_chapter="9",
            source_pdf_page="297-301",
            source_type="PROJECT-ORIGINAL-EXTENSION",
            formula=(
                "max(RV-BV, 0)/RV; RV=sum_60(r^2); BV=(pi/2)*(60/59)*sum_60(abs(r_i)*abs(r_(i-1)))"
            ),
            rationale=(
                "The share of local variation attributable to jumps may alter subsequent "
                "expansion and risk."
            ),
            target_family="risk state / expansion",
            lookback="61 close-to-close returns from 62 complete bars",
            minimum_history=62,
            input_columns=("close",),
            expected_range="[0, 1]",
            missing_value_policy=(
                "Null until 61 complete returns; zero RV is null; negative RV-minus-BV "
                "is floored at zero."
            ),
            normalization_method="Positive RV-minus-bipower share of realized variance",
            existing_feature_overlap="current_range_over_atr; liquidity_vacuum_score_exp",
            validation_minimum=0.0,
            validation_maximum=1.0,
        ),
        _spec(
            feature_name="variance_ratio_60_5",
            display_name="Variance ratio (60 overlapping 5-return sums)",
            feature_family="serial_dependence",
            hypothesis_number=10,
            formula_number=11,
            source_chapter="9",
            source_pdf_page="280-296",
            source_type="MLAT-ADAPTED",
            formula="population_var_60(sum_5(log_return)) / (5*population_var_60(log_return))",
            rationale=(
                "Deviation from local random-walk variance scaling may identify persistence "
                "or mean reversion."
            ),
            target_family="direction / risk state",
            lookback="64 close-to-close returns from 65 complete bars",
            minimum_history=65,
            input_columns=("close",),
            expected_range="[0, +inf)",
            missing_value_policy=(
                "Null until 60 complete overlapping 5-return sums; zero one-minute "
                "return variance is null."
            ),
            normalization_method="Five-return variance divided by five times one-return variance",
            existing_feature_overlap="return_autocorrelation_15; return_sign_change_rate_30",
            validation_minimum=0.0,
        ),
        _spec(
            feature_name="return_sign_entropy_60",
            display_name="Return-sign entropy (60 returns)",
            feature_family="information_state",
            hypothesis_number=11,
            formula_number=12,
            source_chapter="6",
            source_pdf_page="192",
            source_type="PROJECT-ORIGINAL-EXTENSION",
            formula=("-sum(p_s*log(p_s), s in {negative, zero, positive}) / log(3)"),
            rationale=(
                "Low sign entropy may reflect directional organization while high entropy "
                "may reflect choppy state."
            ),
            target_family="risk state / expansion",
            lookback="60 close-to-close returns from 61 complete bars",
            minimum_history=61,
            input_columns=("close",),
            expected_range="[0, 1]",
            missing_value_policy=(
                "Null until 60 complete returns; sign states with zero probability contribute zero."
            ),
            normalization_method="Three-state Shannon entropy divided by log(3)",
            existing_feature_overlap="choppiness_14; return_sign_change_rate_30",
            validation_minimum=0.0,
            validation_maximum=1.0,
        ),
        _spec(
            feature_name="volatility_of_volatility_60",
            display_name="Volatility of volatility (60 RV15 states)",
            feature_family="volatility_instability",
            hypothesis_number=12,
            formula_number=13,
            source_chapter="9",
            source_pdf_page="297-301",
            source_type="PROJECT-ORIGINAL-EXTENSION",
            formula=(
                "population_std_60(RV15) / mean_60(RV15), RV15=1e4*sqrt(mean_15(log_return^2))"
            ),
            rationale=(
                "Instability of volatility, rather than its level, may identify transition risk."
            ),
            target_family="risk state / expansion",
            lookback="74 close-to-close returns from 75 complete bars",
            minimum_history=75,
            input_columns=("close",),
            expected_range="[0, +inf)",
            missing_value_policy=(
                "Null until 60 complete RV15 states; zero or non-finite mean RV15 is null."
            ),
            normalization_method="Coefficient of variation of trailing RV15 states",
            existing_feature_overlap="atr_ratio_20_60; realized_volatility_ratio_15_60",
            validation_minimum=0.0,
        ),
    )


MLAT_FEATURE_SPECS = _build_specs()
MLAT_REGISTRY_COLUMNS = tuple(field.name for field in fields(MLATFeatureSpec))
MLAT_MINIMUM_HISTORY = tuple(spec.minimum_history for spec in MLAT_FEATURE_SPECS)


def build_mlat_registry() -> tuple[MLATFeatureSpec, ...]:
    """Return the immutable, ordered v1 registry."""

    return MLAT_FEATURE_SPECS


def registry_to_frame(
    registry: Iterable[MLATFeatureSpec] = MLAT_FEATURE_SPECS,
) -> pd.DataFrame:
    """Convert registry specs to a stable machine-readable DataFrame."""

    records: list[dict[str, object]] = []
    for spec in registry:
        if not isinstance(spec, MLATFeatureSpec):
            raise TypeError("Registry iterables must contain MLATFeatureSpec instances.")
        record = asdict(spec)
        record["input_columns"] = "|".join(spec.input_columns)
        records.append(record)
    return pd.DataFrame.from_records(records, columns=MLAT_REGISTRY_COLUMNS)


def mlat_feature_columns(
    registry: Iterable[MLATFeatureSpec] | pd.DataFrame = MLAT_FEATURE_SPECS,
) -> tuple[str, ...]:
    """Return predictor names in registry order."""

    if isinstance(registry, pd.DataFrame):
        if "feature_name" not in registry:
            raise ValueError("Registry frame is missing feature_name.")
        return tuple(registry["feature_name"].astype(str))
    return tuple(spec.feature_name for spec in registry)


def validate_mlat_registry(
    registry: Iterable[MLATFeatureSpec] | pd.DataFrame | None = None,
) -> None:
    """Raise when the frozen registry membership, order, or schema drifts."""

    if registry is None:
        frame = registry_to_frame()
    elif isinstance(registry, pd.DataFrame):
        frame = registry.copy()
    else:
        frame = registry_to_frame(registry)

    missing_columns = sorted(set(MLAT_REGISTRY_COLUMNS).difference(frame.columns))
    extra_columns = sorted(set(frame.columns).difference(MLAT_REGISTRY_COLUMNS))
    if missing_columns or extra_columns:
        raise ValueError(
            f"MLAT registry schema mismatch: missing={missing_columns}, extra={extra_columns}"
        )
    if len(frame) != len(MLAT_FEATURE_NAMES):
        raise ValueError(f"Frozen MLAT registry requires 12 rows; received {len(frame)}.")
    names = tuple(frame["feature_name"].astype(str))
    if names != MLAT_FEATURE_NAMES:
        raise ValueError("MLAT feature membership/order drifted from the frozen v1 contract.")
    if frame["feature_name"].duplicated().any():
        raise ValueError("MLAT registry contains duplicate feature names.")

    expected_hypotheses = tuple(f"MLAT-H{number:03d}" for number in range(1, 13))
    expected_formulas = tuple(f"MLAT-F{number:03d}" for number in range(2, 14))
    if tuple(frame["hypothesis_id"].astype(str)) != expected_hypotheses:
        raise ValueError("MLAT hypothesis IDs must be MLAT-H001 through MLAT-H012.")
    if tuple(frame["formula_id"].astype(str)) != expected_formulas:
        raise ValueError("MLAT formula IDs must be MLAT-F002 through MLAT-F013.")
    if tuple(pd.to_numeric(frame["minimum_history"]).astype(int)) != MLAT_MINIMUM_HISTORY:
        raise ValueError("MLAT minimum-history contract drifted.")

    required_text = (
        "display_name",
        "feature_family",
        "feature_tier",
        "source_chapter",
        "source_pdf_page",
        "source_type",
        "formula_or_definition",
        "economic_rationale",
        "target_family",
        "lookback",
        "input_columns",
        "availability_timestamp",
        "reset_boundary",
        "continuity_requirements",
        "fitted_parameters",
        "fit_scope",
        "expected_range",
        "missing_value_policy",
        "normalization_method",
        "output_dtype",
        "existing_feature_overlap",
        "leakage_risk",
        "implementation_version",
        "status",
    )
    empty = {
        column: frame.index[frame[column].astype(str).str.strip().eq("")].tolist()
        for column in required_text
        if frame[column].astype(str).str.strip().eq("").any()
    }
    if empty:
        raise ValueError(f"MLAT registry contains empty required text fields: {empty}")

    if not frame["feature_tier"].astype(str).eq(MLAT_FEATURE_TIER).all():
        raise ValueError("Every MLAT feature must use the frozen_v1 tier.")
    if not frame["status"].astype(str).eq(MLAT_STATUS).all():
        raise ValueError("Every MLAT feature must remain FROZEN_V1.")
    if not frame["output_dtype"].astype(str).eq("float32").all():
        raise ValueError("Every frozen MLAT predictor must persist as float32.")
    if not frame["implementation_version"].astype(str).eq(MLAT_IMPLEMENTATION_VERSION).all():
        raise ValueError("MLAT implementation versions must match.")
    if not frame["is_experimental"].astype(bool).all():
        raise ValueError("The MLAT v1 additions must remain explicitly experimental.")
    if not frame["leakage_risk"].astype(str).eq("LOW").all():
        raise ValueError("Frozen v1 registry leakage classifications must remain LOW.")

    allowed_source_types = {
        "MLAT-DIRECT",
        "MLAT-ADAPTED",
        "PROJECT-ORIGINAL-EXTENSION",
    }
    invalid_source_types = sorted(
        set(frame["source_type"].astype(str)).difference(allowed_source_types)
    )
    if invalid_source_types:
        raise ValueError(f"Invalid MLAT source types: {invalid_source_types}")

    for row in frame.itertuples(index=False):
        inputs = tuple(str(row.input_columns).split("|"))
        if not inputs or not set(inputs).issubset(MLAT_CAUSAL_INPUT_COLUMNS):
            raise ValueError(f"{row.feature_name} has non-causal or unsupported inputs: {inputs}")
        if int(row.minimum_history) <= 0:
            raise ValueError(f"{row.feature_name} minimum_history must be positive.")
        lower = row.validation_minimum
        upper = row.validation_maximum
        if pd.notna(lower) and pd.notna(upper) and float(lower) > float(upper):
            raise ValueError(f"{row.feature_name} has inverted validation bounds.")


def validate_mlat_feature_frame(frame: pd.DataFrame) -> None:
    """Validate exact MLAT matrix alignment and basic persisted-value policies."""

    expected_columns = MLAT_METADATA_COLUMNS + MLAT_FEATURE_NAMES
    if tuple(frame.columns) != expected_columns:
        raise ValueError(
            "MLAT feature frame columns must be the 12 metadata columns followed "
            "by the frozen 12 predictors."
        )
    if not frame["observation_id"].is_unique:
        raise ValueError("MLAT feature frame observation_id values must be unique.")
    if set(frame["product"].astype(str).unique()) != {"GC"}:
        raise ValueError("MLAT feature frames are GC-only.")

    registry = registry_to_frame().set_index("feature_name")
    for name in MLAT_FEATURE_NAMES:
        if str(frame[name].dtype) != "float32":
            raise TypeError(f"{name} must have float32 dtype; got {frame[name].dtype}.")
        values = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"{name} contains infinite values.")
        finite = values[np.isfinite(values)]
        lower = registry.at[name, "validation_minimum"]
        upper = registry.at[name, "validation_maximum"]
        tolerance = 1.0e-6
        if pd.notna(lower) and finite.size and (finite < float(lower) - tolerance).any():
            raise ValueError(f"{name} violates its lower validation bound.")
        if pd.notna(upper) and finite.size and (finite > float(upper) + tolerance).any():
            raise ValueError(f"{name} violates its upper validation bound.")


validate_mlat_registry()
