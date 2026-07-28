"""Causal multichannel profile representations for Project 1 FES Section 3.

Only the raw P0 profile is materialized. P1-P3 are deterministic row transforms;
their column standardization and PCA/PLS reductions are fitted through explicit
fold-local transformer interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from .fes_project1_features import (
    MISSING_CAUSES,
    OBSERVATION_COLUMNS,
    SOURCE_COLUMNS,
    _as_float,
    _lag,
    _legal_continuity,
    _rolling_mean,
    _rolling_std_population,
    _safe_divide,
    _window,
)

PROFILE_LENGTH: Final = 30
TRIM_EACH_TAIL: Final = 3
TRIMMED_LENGTH: Final = PROFILE_LENGTH - 2 * TRIM_EACH_TAIL
PROFILE_RAW_HISTORY_BARS: Final = 89
PROFILE_CHANNELS: Final = ("R", "G", "Q")
PROFILE_REPRESENTATIONS: Final = ("P0", "P1", "P2", "P3")

P0_COLUMNS: Final = tuple(
    f"{channel}_{position:02d}"
    for channel in PROFILE_CHANNELS
    for position in range(PROFILE_LENGTH)
)
P1_COLUMNS: Final = tuple(
    f"P1_{channel}_{position:02d}"
    for channel in PROFILE_CHANNELS
    for position in range(PROFILE_LENGTH)
)
P2_COLUMNS: Final = tuple(
    f"P2_{channel}_{position:02d}"
    for channel in PROFILE_CHANNELS
    for position in range(1, PROFILE_LENGTH)
)
P3_COLUMNS: Final = tuple(
    f"P3_{channel}_{position:02d}"
    for channel in PROFILE_CHANNELS
    for position in range(2, PROFILE_LENGTH)
)

REPRESENTATION_COLUMNS: Final = {
    "P0": P0_COLUMNS,
    "P1": P1_COLUMNS,
    "P2": P2_COLUMNS,
    "P3": P3_COLUMNS,
}

PROFILE_INDEX_COLUMNS: Final = (
    "observation_id",
    "decision_timestamp_utc",
    "trade_date_ny",
    "entry_session",
    "research_partition",
    "profile_complete_30",
    "profile_missing_reason",
)


@dataclass(frozen=True)
class ProfileBuildResult:
    """Outcome-free profile index, complete raw rows, schema, and audit."""

    profile_index: pd.DataFrame
    raw_profiles: pd.DataFrame
    channel_lag_schema: pd.DataFrame
    construction_audit: pd.DataFrame


def profile_channel_lag_schema() -> pd.DataFrame:
    """Return the exact P0 channel/lag contract in stored-column order."""

    definitions = {
        "R": {
            "channel_name": "return_bps",
            "formula": "10000*log(C_i/C_(i-1))",
            "units": "basis_points",
            "required_input": "close and predecessor close",
        },
        "G": {
            "channel_name": "true_range_over_atr20",
            "formula": "TR_i/A_i",
            "units": "ratio",
            "required_input": "high|low|close|atr_20",
        },
        "Q": {
            "channel_name": "volume_zscore_60",
            "formula": "(log1p(V_i)-mean60)/population_std60",
            "units": "z_score",
            "required_input": "volume",
        },
    }
    records: list[dict[str, object]] = []
    stored_index = 0
    for channel in PROFILE_CHANNELS:
        for position in range(PROFILE_LENGTH):
            records.append(
                {
                    "stored_index": stored_index,
                    "column_name": f"{channel}_{position:02d}",
                    "channel_code": channel,
                    "channel_name": definitions[channel]["channel_name"],
                    "position": position,
                    "bar_offset_from_t": position - (PROFILE_LENGTH - 1),
                    "formula": definitions[channel]["formula"],
                    "units": definitions[channel]["units"],
                    "required_input": definitions[channel]["required_input"],
                    "availability": "close of completed decision bar t",
                    "reset_boundary": (
                        "production continuity_run_id + trade_date_ny outer reset"
                    ),
                    "calculation_dtype": "float64",
                    "saved_dtype": "float32",
                    "version": "1.0.0",
                }
            )
            stored_index += 1
    schema = pd.DataFrame.from_records(records)
    if schema["column_name"].tolist() != list(P0_COLUMNS):
        raise AssertionError("P0 schema order differs from the frozen channel order.")
    return schema


def _validate_profile_inputs(
    bars: pd.DataFrame,
    observations: pd.DataFrame,
) -> None:
    missing_source = sorted(set(SOURCE_COLUMNS).difference(bars.columns))
    missing_observation = sorted(set(OBSERVATION_COLUMNS).difference(observations.columns))
    if missing_source or missing_observation:
        raise ValueError(
            f"Missing profile inputs: source={missing_source}, "
            f"observations={missing_observation}"
        )
    forbidden = [
        str(column)
        for column in (*bars.columns, *observations.columns)
        if any(
            token in str(column).lower()
            for token in ("forward_", "future_", "target", "label", "mfe", "mae")
        )
    ]
    if forbidden:
        raise ValueError(f"Outcome columns entered the profile namespace: {forbidden}")
    if set(bars["product"].astype(str).unique()) != {"GC"}:
        raise ValueError("Section 3 profile construction is GC-only.")
    if not bars["source_row_id"].is_unique:
        raise ValueError("source_row_id must be unique.")
    if not observations["observation_id"].is_unique:
        raise ValueError("Eligible observation_id must be unique.")


def build_profile_matrix(
    full_gc_bars: pd.DataFrame,
    eligible_observations: pd.DataFrame,
    *,
    allowed_partitions: Iterable[str] = ("Development", "Validation"),
    chunk_size: int = 20_000,
) -> ProfileBuildResult:
    """Build the exact 90-column P0 block for complete profiles only."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    _validate_profile_inputs(full_gc_bars, eligible_observations)
    allowed = tuple(allowed_partitions)
    observations = eligible_observations.loc[
        eligible_observations["research_partition"].astype(str).isin(allowed),
        OBSERVATION_COLUMNS,
    ].copy()
    if len(observations) == 0:
        raise ValueError("No eligible observations remain in the allowed partitions.")

    bars = full_gc_bars.loc[:, SOURCE_COLUMNS].copy()
    if (
        not bars["source_row_id"].is_monotonic_increasing
        or not pd.to_datetime(bars["ts_event_utc"], utc=True).is_monotonic_increasing
    ):
        bars = bars.sort_values(
            ["ts_event_utc", "source_row_id"], kind="mergesort"
        ).reset_index(drop=True)
        sorted_during_build = True
    else:
        bars = bars.reset_index(drop=True)
        sorted_during_build = False

    source_ids = pd.to_numeric(bars["source_row_id"]).to_numpy(dtype=np.int64)
    decision_ids = pd.to_numeric(observations["decision_bar_id"]).to_numpy(dtype=np.int64)
    positions = np.searchsorted(source_ids, decision_ids)
    exact = (positions < len(source_ids)) & (
        source_ids[np.minimum(positions, len(source_ids) - 1)] == decision_ids
    )
    if not exact.all():
        raise ValueError(
            f"Decision bars absent from trusted GC source: {decision_ids[~exact][:10].tolist()}"
        )
    source_timestamp_ns = (
        pd.to_datetime(bars["ts_event_utc"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    decision_timestamp_ns = (
        pd.to_datetime(observations["decision_timestamp_utc"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    if not np.array_equal(source_timestamp_ns[positions], decision_timestamp_ns):
        raise ValueError("decision_bar_id and decision_timestamp_utc do not map exactly.")

    close = _as_float(bars["close"])
    high = _as_float(bars["high"])
    low = _as_float(bars["low"])
    volume = _as_float(bars["volume"])
    legal_group, legal_position, latest_cause = _legal_continuity(bars)
    previous_close = _lag(close, 1, legal_group)
    return_bps = np.full(len(close), np.nan, dtype=np.float64)
    valid_return = (
        np.isfinite(close)
        & np.isfinite(previous_close)
        & (close > 0.0)
        & (previous_close > 0.0)
    )
    return_bps[valid_return] = 10_000.0 * np.log(
        close[valid_return] / previous_close[valid_return]
    )
    bar_range = high - low
    true_range = np.maximum.reduce(
        [bar_range, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    true_range[legal_position == 0] = bar_range[legal_position == 0]
    atr20 = _rolling_mean(true_range, 20, legal_group)
    log_volume = np.log1p(volume)
    log_volume_mean60 = _rolling_mean(log_volume, 60, legal_group)
    log_volume_std60 = _rolling_std_population(log_volume, 60, legal_group)
    volume_zscore60 = _safe_divide(
        log_volume - log_volume_mean60, log_volume_std60
    )

    row_count = len(observations)
    profile_values = np.full((row_count, len(P0_COLUMNS)), np.nan, dtype=np.float32)
    reasons = np.full(row_count, "COMPLETE", dtype=object)
    complete_history = (
        legal_position[positions] >= PROFILE_RAW_HISTORY_BARS - 1
    )
    reasons[~complete_history] = latest_cause[positions[~complete_history]]

    for chunk_start in range(0, row_count, chunk_size):
        chunk_stop = min(row_count, chunk_start + chunk_size)
        chunk_rows = np.arange(chunk_start, chunk_stop, dtype=np.int64)
        candidates = chunk_rows[complete_history[chunk_start:chunk_stop]]
        if len(candidates) == 0:
            continue
        ends = positions[candidates]

        # Inspect the exact source spans before derived inputs so an actual
        # source null retains precedence over a derived denominator:
        # R needs close[t-30:t], G needs high/low[t-48:t] plus
        # close[t-49:t], and Q needs volume[t-88:t].
        close50 = _window(close, ends, 50)
        high49 = _window(high, ends, 49)
        low49 = _window(low, ends, 49)
        volume89 = _window(volume, ends, PROFILE_RAW_HISTORY_BARS)
        source_null = ~(
            np.isfinite(close50).all(axis=1)
            & np.isfinite(high49).all(axis=1)
            & np.isfinite(low49).all(axis=1)
            & np.isfinite(volume89).all(axis=1)
        )
        reasons[candidates[source_null]] = "SOURCE_INPUT_NULL"
        source_valid_rows = candidates[~source_null]
        if len(source_valid_rows) == 0:
            continue
        source_valid_volume89 = volume89[~source_null]
        volume_windows60 = np.lib.stride_tricks.sliding_window_view(
            source_valid_volume89,
            window_shape=60,
            axis=1,
        )
        constant_zscore_window = (
            np.ptp(volume_windows60, axis=2) == 0.0
        ).any(axis=1)

        source_valid_ends = positions[source_valid_rows]
        returns30 = _window(return_bps, source_valid_ends, PROFILE_LENGTH)
        true_range30 = _window(true_range, source_valid_ends, PROFILE_LENGTH)
        atr30 = _window(atr20, source_valid_ends, PROFILE_LENGTH)
        zscore30 = _window(volume_zscore60, source_valid_ends, PROFILE_LENGTH)
        bad_atr = ~np.isfinite(atr30).all(axis=1) | (atr30 <= 0.0).any(axis=1)
        reasons[source_valid_rows[bad_atr]] = "NONPOSITIVE_ATR"
        atr_valid_rows = source_valid_rows[~bad_atr]
        if len(atr_valid_rows) == 0:
            continue

        retained = ~bad_atr
        channels_finite = (
            np.isfinite(returns30[retained]).all(axis=1)
            & np.isfinite(true_range30[retained]).all(axis=1)
            & np.isfinite(zscore30[retained]).all(axis=1)
            & ~constant_zscore_window[retained]
        )
        reasons[atr_valid_rows[~channels_finite]] = "SOURCE_INPUT_NULL"
        complete_rows = atr_valid_rows[channels_finite]
        if len(complete_rows) == 0:
            continue

        channel_selection = retained.copy()
        channel_selection[retained] = channels_finite
        return_values = returns30[channel_selection]
        range_values = (
            true_range30[channel_selection] / atr30[channel_selection]
        )
        volume_values = zscore30[channel_selection]
        concatenated = np.concatenate(
            [return_values, range_values, volume_values], axis=1
        )
        if not np.isfinite(concatenated).all():
            raise AssertionError("A profile marked complete contains a non-finite value.")
        profile_values[complete_rows] = concatenated.astype(np.float32)

    complete = reasons == "COMPLETE"
    if not np.isfinite(profile_values[complete]).all():
        raise AssertionError("Complete profiles contain non-finite stored values.")
    if not np.isnan(profile_values[~complete]).all():
        raise AssertionError("Incomplete profiles must not retain numeric profile values.")

    cause_dtype = pd.CategoricalDtype(categories=list(MISSING_CAUSES), ordered=True)
    profile_index = observations.loc[
        :,
        [
            "observation_id",
            "decision_timestamp_utc",
            "trade_date_ny",
            "entry_session",
            "research_partition",
        ],
    ].reset_index(drop=True)
    profile_index["profile_complete_30"] = complete
    profile_index["profile_missing_reason"] = pd.Categorical(
        reasons, dtype=cause_dtype
    )
    profile_index = profile_index.loc[:, PROFILE_INDEX_COLUMNS]

    raw_profiles = pd.DataFrame(
        profile_values[complete],
        columns=P0_COLUMNS,
    )
    raw_profiles.insert(
        0,
        "observation_id",
        observations.loc[complete, "observation_id"].to_numpy(),
    )
    for name in P0_COLUMNS:
        if raw_profiles[name].dtype != np.dtype("float32"):
            raise AssertionError(f"{name} is not stored as float32.")

    construction_audit = pd.DataFrame.from_records(
        [
            {"item": "source_rows", "value": str(len(bars))},
            {"item": "selected_observations", "value": str(row_count)},
            {"item": "selected_partitions", "value": "|".join(allowed)},
            {"item": "historical_final_materialized", "value": "False"},
            {"item": "outcome_columns_accepted", "value": "False"},
            {"item": "source_sorted_during_build", "value": str(sorted_during_build)},
            {"item": "decision_bar_ids_exact", "value": "True"},
            {"item": "decision_timestamps_exact", "value": "True"},
            {"item": "profile_positions", "value": str(PROFILE_LENGTH)},
            {"item": "profile_channels", "value": "|".join(PROFILE_CHANNELS)},
            {"item": "raw_profile_columns", "value": str(len(P0_COLUMNS))},
            {"item": "stored_channel_order", "value": "R_0..R_29|G_0..G_29|Q_0..Q_29"},
            {"item": "t_minus_30_predecessor_required", "value": "True"},
            {"item": "minimum_raw_history_bars", "value": str(PROFILE_RAW_HISTORY_BARS)},
            {"item": "ny_date_outer_reset", "value": "True"},
            {"item": "raw_profile_persisted_once", "value": "True"},
            {"item": "full_data_pca_pls_persisted", "value": "False"},
            {"item": "calculation_dtype", "value": "float64"},
            {"item": "saved_dtype", "value": "float32"},
        ]
    )
    return ProfileBuildResult(
        profile_index=profile_index,
        raw_profiles=raw_profiles,
        channel_lag_schema=profile_channel_lag_schema(),
        construction_audit=construction_audit,
    )


def _validate_raw_profile_array(values: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        missing = sorted(set(P0_COLUMNS).difference(values.columns))
        if missing:
            raise ValueError(f"Missing raw profile columns: {missing}")
        array = values.loc[:, P0_COLUMNS].to_numpy(dtype=np.float64)
    else:
        array = check_array(
            values,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite=True,
        )
    if array.shape[1] != len(P0_COLUMNS):
        raise ValueError(
            f"Expected {len(P0_COLUMNS)} raw columns; received {array.shape[1]}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Profile transforms require complete finite raw rows.")
    return array


def p1_robust_standardize(
    raw_values: np.ndarray | pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact per-row/per-channel trimmed P1 transform."""

    raw = _validate_raw_profile_array(raw_values)
    channels = raw.reshape(-1, len(PROFILE_CHANNELS), PROFILE_LENGTH)
    ordered = np.sort(channels, axis=2)
    trimmed = ordered[:, :, TRIM_EACH_TAIL:-TRIM_EACH_TAIL]
    trimmed_mean = np.mean(trimmed, axis=2)
    trimmed_std = np.std(trimmed, axis=2, ddof=0)
    zero_scale = trimmed_std == 0.0
    transformed = np.zeros_like(channels, dtype=np.float64)
    valid_scale = ~zero_scale
    np.divide(
        channels - trimmed_mean[:, :, None],
        trimmed_std[:, :, None],
        out=transformed,
        where=valid_scale[:, :, None],
    )
    return transformed.reshape(-1, len(P1_COLUMNS)), zero_scale


def p2_first_difference(
    p1_values: np.ndarray,
) -> np.ndarray:
    """Create three ordered 29-position first-difference channels."""

    values = check_array(
        p1_values,
        ensure_2d=True,
        dtype=np.float64,
        ensure_all_finite=True,
    )
    if values.shape[1] != len(P1_COLUMNS):
        raise ValueError(f"P2 requires {len(P1_COLUMNS)} P1 columns.")
    channels = values.reshape(-1, len(PROFILE_CHANNELS), PROFILE_LENGTH)
    return np.diff(channels, axis=2).reshape(-1, len(P2_COLUMNS))


def p3_trailing_median(
    p1_values: np.ndarray,
) -> np.ndarray:
    """Create three causal 28-position trailing-median channels."""

    values = check_array(
        p1_values,
        ensure_2d=True,
        dtype=np.float64,
        ensure_all_finite=True,
    )
    if values.shape[1] != len(P1_COLUMNS):
        raise ValueError(f"P3 requires {len(P1_COLUMNS)} P1 columns.")
    channels = values.reshape(-1, len(PROFILE_CHANNELS), PROFILE_LENGTH)
    stacked = np.stack(
        [channels[:, :, :-2], channels[:, :, 1:-1], channels[:, :, 2:]],
        axis=-1,
    )
    return np.median(stacked, axis=-1).reshape(-1, len(P3_COLUMNS))


def transform_profile_representation(
    raw_values: np.ndarray | pd.DataFrame,
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return P0-P3 deterministic row values and P1 zero-scale flags."""

    representation = str(representation).upper()
    if representation not in PROFILE_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {representation!r}; "
            f"expected one of {PROFILE_REPRESENTATIONS}."
        )
    raw = _validate_raw_profile_array(raw_values)
    zero_scale = np.zeros((len(raw), len(PROFILE_CHANNELS)), dtype=bool)
    if representation == "P0":
        return raw.copy(), zero_scale
    p1, zero_scale = p1_robust_standardize(raw)
    if representation == "P1":
        return p1, zero_scale
    if representation == "P2":
        return p2_first_difference(p1), zero_scale
    return p3_trailing_median(p1), zero_scale


def _orientation_signs(loadings: np.ndarray) -> np.ndarray:
    loadings = np.asarray(loadings, dtype=np.float64)
    pivot = np.argmax(np.abs(loadings), axis=1)
    signs = np.sign(loadings[np.arange(len(loadings)), pivot])
    signs[signs == 0.0] = 1.0
    return signs


class FoldLocalProfileTransformer(TransformerMixin, BaseEstimator):
    """P0-P3 row transform plus training-fold-only column standardization."""

    def __init__(self, representation: str = "P0") -> None:
        self.representation = representation

    def fit(self, X: np.ndarray | pd.DataFrame, y: object = None):
        del y
        raw = _validate_raw_profile_array(X)
        transformed, _ = transform_profile_representation(raw, self.representation)
        self.scaler_ = StandardScaler().fit(transformed)
        self.n_features_in_ = raw.shape[1]
        self.fit_row_count_ = raw.shape[0]
        self.output_columns_ = REPRESENTATION_COLUMNS[str(self.representation).upper()]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ("scaler_", "output_columns_"))
        raw = _validate_raw_profile_array(X)
        transformed, _ = transform_profile_representation(raw, self.representation)
        return self.scaler_.transform(transformed)


class FoldLocalPCAProfileTransformer(TransformerMixin, BaseEstimator):
    """Fold-local standardized P0-P3 PCA with deterministic score orientation."""

    def __init__(
        self,
        representation: str = "P0",
        n_components: int = 3,
    ) -> None:
        self.representation = representation
        self.n_components = n_components

    def fit(self, X: np.ndarray | pd.DataFrame, y: object = None):
        del y
        self.profile_transformer_ = FoldLocalProfileTransformer(
            representation=self.representation
        )
        standardized = self.profile_transformer_.fit_transform(X)
        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=False,
            svd_solver="full",
        ).fit(standardized)
        self.orientation_signs_ = _orientation_signs(self.pca_.components_)
        self.oriented_components_ = (
            self.pca_.components_ * self.orientation_signs_[:, None]
        )
        self.fit_row_count_ = standardized.shape[0]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self,
            ("profile_transformer_", "pca_", "orientation_signs_"),
        )
        standardized = self.profile_transformer_.transform(X)
        return self.pca_.transform(standardized) * self.orientation_signs_


class FoldLocalPLSProfileTransformer(TransformerMixin, BaseEstimator):
    """Fold-local P2 PLS interface; fitting requires training-fold targets."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | None = None):
        if y is None:
            raise ValueError("FoldLocalPLSProfileTransformer.fit requires training targets.")
        self.profile_transformer_ = FoldLocalProfileTransformer(representation="P2")
        standardized = self.profile_transformer_.fit_transform(X)
        target = check_array(
            y,
            ensure_2d=False,
            dtype=np.float64,
            ensure_all_finite=True,
        )
        self.pls_ = PLSRegression(
            n_components=self.n_components,
            scale=False,
            max_iter=500,
            tol=1.0e-6,
        ).fit(standardized, target)
        rotations = np.asarray(self.pls_.x_rotations_, dtype=np.float64).T
        self.orientation_signs_ = _orientation_signs(rotations)
        self.oriented_x_rotations_ = (
            np.asarray(self.pls_.x_rotations_, dtype=np.float64)
            * self.orientation_signs_[None, :]
        )
        self.fit_row_count_ = standardized.shape[0]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self,
            ("profile_transformer_", "pls_", "orientation_signs_"),
        )
        standardized = self.profile_transformer_.transform(X)
        return self.pls_.transform(standardized) * self.orientation_signs_


def build_profile_coverage_audit(profile_index: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact Development profile missing causes by session."""

    if set(profile_index["research_partition"].astype(str).unique()) != {
        "Development"
    }:
        raise ValueError("Profile coverage diagnostics are Development-only in Section 3.")
    records: list[dict[str, object]] = []
    for session, group in profile_index.groupby(
        "entry_session", observed=True, sort=True
    ):
        counts = group["profile_missing_reason"].value_counts(dropna=False)
        complete = group["profile_complete_30"].astype(bool)
        complete_dates = group.loc[complete, "trade_date_ny"].nunique()
        for cause in MISSING_CAUSES:
            count = int(counts.get(cause, 0))
            records.append(
                {
                    "entry_session": str(session),
                    "cause": cause,
                    "rows": count,
                    "rate": count / len(group) if len(group) else np.nan,
                    "complete_ny_dates": (
                        int(complete_dates) if cause == "COMPLETE" else pd.NA
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def build_p1_zero_scale_audit(
    raw_profiles: pd.DataFrame,
    *,
    chunk_size: int = 20_000,
) -> pd.DataFrame:
    """Count formula-defined P1 zero-scale channels without persisting P1."""

    counts = np.zeros(len(PROFILE_CHANNELS), dtype=np.int64)
    for start in range(0, len(raw_profiles), chunk_size):
        stop = min(len(raw_profiles), start + chunk_size)
        _, zero_scale = p1_robust_standardize(
            raw_profiles.iloc[start:stop].loc[:, P0_COLUMNS]
        )
        counts += zero_scale.sum(axis=0)
    return pd.DataFrame.from_records(
        [
            {
                "channel_code": channel,
                "complete_profiles": len(raw_profiles),
                "zero_scale_rows": int(count),
                "zero_scale_rate": count / len(raw_profiles)
                if len(raw_profiles)
                else np.nan,
            }
            for channel, count in zip(PROFILE_CHANNELS, counts, strict=True)
        ]
    )


def _row_autocorrelation(
    values: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    left = values[:, :-lag]
    right = values[:, lag:]
    left_centered = left - np.mean(left, axis=1, keepdims=True)
    right_centered = right - np.mean(right, axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    valid = np.isfinite(denominator) & (denominator > 0.0)
    correlation = np.full(len(values), np.nan, dtype=np.float64)
    np.divide(
        np.sum(left_centered * right_centered, axis=1),
        denominator,
        out=correlation,
        where=valid,
    )
    finite = np.isfinite(correlation)
    correlation[finite] = np.clip(correlation[finite], -1.0, 1.0)
    return correlation, valid


def build_profile_autocorrelation_diagnostics(
    raw_profiles: pd.DataFrame,
    *,
    maximum_lag: int = 10,
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """Aggregate outcome-free row autocorrelation for P0-P3."""

    aggregates = {
        (representation, channel, lag): [0.0, 0]
        for representation in PROFILE_REPRESENTATIONS
        for channel in PROFILE_CHANNELS
        for lag in range(1, maximum_lag + 1)
    }
    for start in range(0, len(raw_profiles), chunk_size):
        stop = min(len(raw_profiles), start + chunk_size)
        raw = raw_profiles.iloc[start:stop].loc[:, P0_COLUMNS]
        for representation in PROFILE_REPRESENTATIONS:
            transformed, _ = transform_profile_representation(raw, representation)
            channel_length = len(REPRESENTATION_COLUMNS[representation]) // len(
                PROFILE_CHANNELS
            )
            channels = transformed.reshape(
                -1, len(PROFILE_CHANNELS), channel_length
            )
            for channel_index, channel in enumerate(PROFILE_CHANNELS):
                values = channels[:, channel_index, :]
                for lag in range(1, maximum_lag + 1):
                    correlation, valid = _row_autocorrelation(values, lag)
                    aggregate = aggregates[(representation, channel, lag)]
                    aggregate[0] += float(np.nansum(np.abs(correlation[valid])))
                    aggregate[1] += int(valid.sum())
    records: list[dict[str, object]] = []
    for (representation, channel, lag), (total, count) in aggregates.items():
        records.append(
            {
                "representation": representation,
                "channel_code": channel,
                "lag": lag,
                "valid_profiles": count,
                "mean_absolute_autocorrelation": (
                    total / count if count else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_profile_rank_condition_diagnostics(
    raw_profiles: pd.DataFrame,
    *,
    chunk_size: int = 20_000,
) -> pd.DataFrame:
    """Compute full-Development outcome-free rank/condition/PCA summaries."""

    records: list[dict[str, object]] = []
    row_count = len(raw_profiles)
    for representation in PROFILE_REPRESENTATIONS:
        column_count = len(REPRESENTATION_COLUMNS[representation])
        sums = np.zeros(column_count, dtype=np.float64)
        cross_products = np.zeros((column_count, column_count), dtype=np.float64)
        for start in range(0, row_count, chunk_size):
            stop = min(row_count, start + chunk_size)
            transformed, _ = transform_profile_representation(
                raw_profiles.iloc[start:stop].loc[:, P0_COLUMNS],
                representation,
            )
            sums += transformed.sum(axis=0)
            cross_products += transformed.T @ transformed
        means = sums / row_count
        covariance = cross_products / row_count - np.outer(means, means)
        variance = np.diag(covariance).copy()
        variance[(variance < 0.0) & (variance > -1.0e-12)] = 0.0
        scale = np.sqrt(np.maximum(variance, 0.0))
        nonconstant = scale > 0.0
        denominator = np.outer(
            np.where(nonconstant, scale, 1.0),
            np.where(nonconstant, scale, 1.0),
        )
        correlation = covariance / denominator
        correlation[~nonconstant, :] = 0.0
        correlation[:, ~nonconstant] = 0.0
        correlation = (correlation + correlation.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
        eigenvalues[(eigenvalues < 0.0) & (eigenvalues > -1.0e-10)] = 0.0
        positive = eigenvalues[eigenvalues > 1.0e-12]
        numerical_rank = len(positive)
        if len(positive) and numerical_rank == column_count:
            condition_number = float(np.sqrt(positive[0] / positive[-1]))
        else:
            condition_number = np.inf
        eigen_total = float(np.sum(positive))
        probability = positive / eigen_total if eigen_total > 0.0 else positive
        effective_rank = (
            float(np.exp(-np.sum(probability * np.log(probability))))
            if len(probability)
            else 0.0
        )
        stable_rank = (
            float(eigen_total / positive[0]) if len(positive) else 0.0
        )
        cumulative = (
            np.cumsum(positive) / eigen_total if eigen_total > 0.0 else positive
        )

        records.append(
            {
                "representation": representation,
                "rows": row_count,
                "columns": column_count,
                "zero_variance_columns": int((~nonconstant).sum()),
                "numerical_rank": numerical_rank,
                "effective_rank_entropy": effective_rank,
                "stable_rank": stable_rank,
                "condition_number_standardized": condition_number,
                "pca_components_80_percent": _components_for_cumulative(
                    cumulative, 0.80
                ),
                "pca_components_90_percent": _components_for_cumulative(
                    cumulative, 0.90
                ),
                "pca_components_95_percent": _components_for_cumulative(
                    cumulative, 0.95
                ),
                "pca_cumulative_variance_3": (
                    float(cumulative[min(2, len(cumulative) - 1)])
                    if len(cumulative)
                    else np.nan
                ),
                "pca_cumulative_variance_5": (
                    float(cumulative[min(4, len(cumulative) - 1)])
                    if len(cumulative)
                    else np.nan
                ),
                "pca_cumulative_variance_8": (
                    float(cumulative[min(7, len(cumulative) - 1)])
                    if len(cumulative)
                    else np.nan
                ),
                "pca_cumulative_variance_12": (
                    float(cumulative[min(11, len(cumulative) - 1)])
                    if len(cumulative)
                    else np.nan
                ),
                "diagnostic_fit_scope": (
                    "full Development; outcome-free diagnostic only; "
                    "not persisted or used for model fitting"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _components_for_cumulative(cumulative: np.ndarray, threshold: float) -> int:
    if len(cumulative) == 0:
        return 0
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)
