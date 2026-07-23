"""Leakage-controlled, GC-only volatility-model audit utilities.

The module deliberately keeps GARCH outside the frozen feature registry.  It
fits one Gaussian GARCH(1,1) parameter set on returns ending 2022-12-31,
restarts the recursion at every established continuity boundary, calibrates
forecast scale on 2023 only, and evaluates the frozen result on 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from scipy.stats import chi2

from .feature_engineering import build_continuity_run_id

MODEL_NAME = "Gaussian GARCH(1,1)"
MODEL_VERSION = "mlat-garch-audit-v1"
DEFAULT_RETURN_SCALE = 10_000.0
DEFAULT_MAX_PERSISTENCE = 0.999
DEFAULT_FIT_END = pd.Timestamp("2022-12-31 23:59:59.999999999", tz="UTC")
DEFAULT_AUDIT_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
DEFAULT_AUDIT_END = pd.Timestamp("2023-12-31 23:59:59.999999999", tz="UTC")
DEFAULT_VALIDATION_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
DEFAULT_VALIDATION_END = pd.Timestamp("2024-12-31 23:59:59.999999999", tz="UTC")
MIN_VARIANCE = 1.0e-18

_ESTABLISHED_CONTINUITY_INPUTS = {
    "ts_event_utc",
    "product",
    "symbol",
    "active_symbol",
    "instrument_id",
    "continuous_segment_id",
    "tradable_research_flag",
    "roll_window_flag",
    "low_liquidity_warning_flag",
}

__all__ = [
    "GARCHAuditResult",
    "GARCHFitResult",
    "GARCHInputError",
    "GARCHParameters",
    "apply_volatility_calibration",
    "audit_segmented_garch",
    "fit_segmented_garch",
    "fit_volatility_calibration",
    "forecast_segmented_garch",
    "prepare_segmented_log_returns",
]


class GARCHInputError(ValueError):
    """Raised when the volatility audit cannot honor its input contract."""


@dataclass(frozen=True)
class GARCHParameters:
    """Typed raw-return-scale Gaussian GARCH(1,1) parameters."""

    mean: float
    omega: float
    alpha: float
    beta: float
    initial_variance: float
    return_scale: float
    fit_end_utc: pd.Timestamp
    n_fit_observations: int
    n_fit_segments: int
    optimizer_method: str
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    objective_value: float
    iterations: int
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION

    @property
    def persistence(self) -> float:
        """Return alpha + beta."""

        return float(self.alpha + self.beta)

    def mathematical_failures(self) -> tuple[str, ...]:
        """Return violations of the declared mathematical parameter domain."""

        failures: list[str] = []
        if not np.isfinite(self.mean):
            failures.append("mean_not_finite")
        if not np.isfinite(self.omega) or self.omega <= 0.0:
            failures.append("omega_not_positive")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            failures.append("alpha_negative_or_not_finite")
        if not np.isfinite(self.beta) or self.beta < 0.0:
            failures.append("beta_negative_or_not_finite")
        if not np.isfinite(self.persistence) or self.persistence >= 1.0:
            failures.append("persistence_not_below_one")
        if not np.isfinite(self.initial_variance) or self.initial_variance <= 0.0:
            failures.append("initial_variance_not_positive")
        if not np.isfinite(self.return_scale) or self.return_scale <= 0.0:
            failures.append("return_scale_not_positive")
        return tuple(failures)

    def to_frame(self) -> pd.DataFrame:
        """Serialize parameters to a deterministic one-row table."""

        return pd.DataFrame(
            {
                "model_name": [self.model_name],
                "model_version": [self.model_version],
                "mean": np.array([self.mean], dtype=np.float64),
                "omega": np.array([self.omega], dtype=np.float64),
                "alpha": np.array([self.alpha], dtype=np.float64),
                "beta": np.array([self.beta], dtype=np.float64),
                "persistence": np.array([self.persistence], dtype=np.float64),
                "initial_variance": np.array([self.initial_variance], dtype=np.float64),
                "return_scale": np.array([self.return_scale], dtype=np.float64),
                "fit_end_utc": pd.DatetimeIndex([_as_utc_timestamp(self.fit_end_utc)]),
                "n_fit_observations": np.array(
                    [self.n_fit_observations], dtype=np.int64
                ),
                "n_fit_segments": np.array([self.n_fit_segments], dtype=np.int64),
                "optimizer_method": [self.optimizer_method],
                "optimizer_success": np.array([self.optimizer_success], dtype=bool),
                "optimizer_status": np.array([self.optimizer_status], dtype=np.int64),
                "optimizer_message": [self.optimizer_message],
                "objective_value": np.array([self.objective_value], dtype=np.float64),
                "iterations": np.array([self.iterations], dtype=np.int64),
            }
        )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> GARCHParameters:
        """Reconstruct a parameter object from :meth:`to_frame` output."""

        if len(frame) != 1:
            raise GARCHInputError("GARCH parameter table must contain exactly one row.")
        required = {
            "model_name",
            "model_version",
            "mean",
            "omega",
            "alpha",
            "beta",
            "initial_variance",
            "return_scale",
            "fit_end_utc",
            "n_fit_observations",
            "n_fit_segments",
            "optimizer_method",
            "optimizer_success",
            "optimizer_status",
            "optimizer_message",
            "objective_value",
            "iterations",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise GARCHInputError(f"GARCH parameter table is missing columns: {missing}")
        row = frame.iloc[0]
        return cls(
            mean=float(row["mean"]),
            omega=float(row["omega"]),
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
            initial_variance=float(row["initial_variance"]),
            return_scale=float(row["return_scale"]),
            fit_end_utc=_as_utc_timestamp(row["fit_end_utc"]),
            n_fit_observations=int(row["n_fit_observations"]),
            n_fit_segments=int(row["n_fit_segments"]),
            optimizer_method=str(row["optimizer_method"]),
            optimizer_success=bool(row["optimizer_success"]),
            optimizer_status=int(row["optimizer_status"]),
            optimizer_message=str(row["optimizer_message"]),
            objective_value=float(row["objective_value"]),
            iterations=int(row["iterations"]),
            model_name=str(row["model_name"]),
            model_version=str(row["model_version"]),
        )


@dataclass(frozen=True)
class GARCHFitResult:
    """Parameter object plus explicit optimizer and constraint diagnostics."""

    parameters: GARCHParameters
    diagnostics: pd.DataFrame

    @property
    def usable(self) -> bool:
        """Whether all critical fit checks passed."""

        critical = self.diagnostics["critical"].astype(bool)
        return bool(self.diagnostics.loc[critical, "passed"].astype(bool).all())


@dataclass(frozen=True)
class GARCHAuditResult:
    """Serializable tables produced by the complete isolated audit."""

    fit: GARCHFitResult
    forecasts: pd.DataFrame
    residual_diagnostics: pd.DataFrame
    calibration: pd.DataFrame
    benchmark_metrics: pd.DataFrame
    audit_checks: pd.DataFrame

    @property
    def parameters(self) -> pd.DataFrame:
        """Return the one-row parameter table used by persistence helpers."""

        return self.fit.parameters.to_frame()


def _as_utc_timestamp(value: object, *, end_of_day: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if end_of_day and timestamp == timestamp.normalize():
        timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def _strict_runs(
    timestamps: pd.Series,
    source_runs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    timestamp_ns = (
        pd.to_datetime(timestamps, utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .view(np.int64)
    )
    source_runs = np.asarray(source_runs, dtype=np.int64)
    starts = np.ones(len(source_runs), dtype=bool)
    if len(source_runs) > 1:
        starts[1:] = (source_runs[1:] != source_runs[:-1]) | (
            np.diff(timestamp_ns) != 60_000_000_000
        )
    strict = np.cumsum(starts, dtype=np.int64) - 1
    first_positions = np.maximum.accumulate(
        np.where(starts, np.arange(len(strict), dtype=np.int64), 0)
    )
    positions = np.arange(len(strict), dtype=np.int64) - first_positions
    return strict, positions


def _ordered_gc(frame: pd.DataFrame, *, timestamp_column: str) -> pd.DataFrame:
    required = {"product", timestamp_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise GARCHInputError(f"Volatility input is missing columns: {missing}")
    gc = frame.loc[frame["product"].astype(str).eq("GC")].copy()
    if gc.empty:
        raise GARCHInputError("Volatility audit requires at least one GC row.")
    gc[timestamp_column] = pd.to_datetime(gc[timestamp_column], utc=True)
    sort_columns = [timestamp_column]
    if "source_row_id" in gc.columns:
        sort_columns.append("source_row_id")
    if not gc[timestamp_column].is_monotonic_increasing:
        gc = gc.sort_values(sort_columns, kind="mergesort")
    gc = gc.reset_index(drop=True)
    if gc[timestamp_column].duplicated().any():
        duplicated = gc.loc[gc[timestamp_column].duplicated(), timestamp_column].head(3)
        raise GARCHInputError(
            "GC timestamps must be unique after product filtering; examples="
            f"{duplicated.astype(str).tolist()}"
        )
    return gc


def prepare_segmented_log_returns(
    bars: pd.DataFrame,
    *,
    timestamp_column: str = "ts_event_utc",
    price_column: str = "close",
    continuity_column: str = "continuity_run_id",
) -> pd.DataFrame:
    """Build GC-only log returns that are null at every continuity boundary.

    If an established ``continuity_run_id`` is supplied, it is consumed
    directly and additionally guarded against timestamp gaps.  Otherwise the
    repository's locked :func:`build_continuity_run_id` implementation is used,
    which requires the full continuity-source column set.
    """

    gc = _ordered_gc(bars, timestamp_column=timestamp_column)
    if price_column not in gc.columns:
        raise GARCHInputError(f"Volatility input is missing price column {price_column!r}.")
    close = pd.to_numeric(gc[price_column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(close).all() or np.any(close <= 0.0):
        raise GARCHInputError("GC close prices must be finite and strictly positive.")

    if continuity_column in gc.columns:
        source_runs_series = pd.to_numeric(gc[continuity_column], errors="coerce")
        if source_runs_series.isna().any():
            raise GARCHInputError(f"{continuity_column} must be complete and numeric.")
        source_runs = source_runs_series.to_numpy(dtype=np.int64)
    elif _ESTABLISHED_CONTINUITY_INPUTS.issubset(gc.columns):
        source_runs, _, _ = build_continuity_run_id(gc)
    else:
        missing = sorted(_ESTABLISHED_CONTINUITY_INPUTS.difference(gc.columns))
        raise GARCHInputError(
            "Provide established continuity_run_id or all locked continuity inputs; "
            f"missing={missing}"
        )

    strict_runs, run_positions = _strict_runs(gc[timestamp_column], source_runs)
    previous_close = pd.Series(close).groupby(strict_runs, sort=False).shift(1)
    previous = previous_close.to_numpy(dtype=np.float64, na_value=np.nan)
    log_return = np.full(len(gc), np.nan, dtype=np.float64)
    available = np.isfinite(previous) & (previous > 0.0)
    log_return[available] = np.log(close[available] / previous[available])

    output_columns: dict[str, object] = {}
    for column in (
        "source_row_id",
        timestamp_column,
        "product",
        "open",
        "high",
        "low",
        price_column,
        "atr_20",
        "rolling_atr_20m",
        "decision_atr_20m",
    ):
        if column in gc.columns and column not in output_columns:
            output_columns[column] = gc[column].reset_index(drop=True)
    output_columns["source_continuity_run_id"] = source_runs
    output_columns["continuity_run_id"] = strict_runs
    output_columns["continuity_run_position"] = run_positions
    output_columns["log_return"] = log_return
    output_columns["return_available"] = available
    result = pd.DataFrame(output_columns)
    if timestamp_column != "ts_event_utc":
        result = result.rename(columns={timestamp_column: "ts_event_utc"})
    if price_column != "close":
        result = result.rename(columns={price_column: "close"})
    result.attrs["excluded_non_gc_rows"] = int(len(bars) - len(gc))
    return result


def _coerce_returns_frame(
    data: pd.DataFrame,
    *,
    timestamp_column: str,
    return_column: str,
    segment_column: str,
) -> pd.DataFrame:
    if return_column not in data.columns:
        return prepare_segmented_log_returns(
            data,
            timestamp_column=timestamp_column,
            continuity_column=segment_column,
        )

    gc = _ordered_gc(data, timestamp_column=timestamp_column)
    if segment_column not in gc.columns:
        raise GARCHInputError(
            f"Prepared returns require segment column {segment_column!r}."
        )
    source_runs_series = pd.to_numeric(gc[segment_column], errors="coerce")
    if source_runs_series.isna().any():
        raise GARCHInputError(f"{segment_column} must be complete and numeric.")
    source_runs = source_runs_series.to_numpy(dtype=np.int64)
    strict_runs, run_positions = _strict_runs(gc[timestamp_column], source_runs)
    result = gc.copy()
    result["source_continuity_run_id"] = source_runs
    result["continuity_run_id"] = strict_runs
    result["continuity_run_position"] = run_positions
    log_return = pd.to_numeric(result[return_column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    log_return = log_return.copy()
    boundary = run_positions == 0
    sanitized_boundaries = int(np.isfinite(log_return[boundary]).sum())
    log_return[boundary] = np.nan
    result["log_return"] = log_return
    result["return_available"] = np.isfinite(log_return)
    if timestamp_column != "ts_event_utc":
        result = result.rename(columns={timestamp_column: "ts_event_utc"})
    result.attrs["excluded_non_gc_rows"] = int(len(data) - len(gc))
    result.attrs["sanitized_boundary_returns"] = sanitized_boundaries
    return result


def _segment_arrays(
    frame: pd.DataFrame,
    *,
    value_column: str,
    mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    selected = frame if mask is None else frame.loc[np.asarray(mask, dtype=bool)]
    arrays: list[np.ndarray] = []
    for _, group in selected.groupby("continuity_run_id", sort=False):
        values = pd.to_numeric(group[value_column], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        values = values[np.isfinite(values)]
        if values.size:
            arrays.append(values)
    return arrays


def _variance_path(
    residuals: np.ndarray,
    *,
    omega: float,
    alpha: float,
    beta: float,
    initial_variance: float,
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.size == 0:
        return np.empty(0, dtype=np.float64)
    variances = np.empty(residuals.size, dtype=np.float64)
    variances[0] = initial_variance
    if residuals.size > 1:
        innovations = omega + alpha * np.square(residuals[:-1])
        variances[1:] = lfilter(
            [1.0],
            [1.0, -beta],
            innovations,
            zi=np.array([beta * initial_variance], dtype=np.float64),
        )[0]
    return variances


def _logit(probability: float) -> float:
    probability = float(np.clip(probability, 1.0e-12, 1.0 - 1.0e-12))
    return float(np.log(probability / (1.0 - probability)))


def _decode_parameters(
    theta: np.ndarray,
    *,
    max_persistence: float,
) -> tuple[float, float, float, float]:
    mean = float(theta[0])
    omega = float(np.exp(np.clip(theta[1], -50.0, 50.0)))
    persistence = float(max_persistence * expit(theta[2]))
    alpha_share = float(expit(theta[3]))
    alpha = persistence * alpha_share
    beta = persistence * (1.0 - alpha_share)
    return mean, omega, alpha, beta


def _diagnostic_record(
    check_name: str,
    passed: bool,
    *,
    value: float | None,
    threshold: str,
    details: str,
    critical: bool = True,
) -> dict[str, object]:
    return {
        "category": "garch_fit",
        "check_name": check_name,
        "critical": bool(critical),
        "passed": bool(passed),
        "numeric_value": np.nan if value is None else float(value),
        "threshold": threshold,
        "details": details,
    }


def fit_segmented_garch(
    data: pd.DataFrame,
    *,
    fit_end: object = DEFAULT_FIT_END,
    timestamp_column: str = "ts_event_utc",
    return_column: str = "log_return",
    segment_column: str = "continuity_run_id",
    return_scale: float = DEFAULT_RETURN_SCALE,
    max_persistence: float = DEFAULT_MAX_PERSISTENCE,
    min_observations: int = 250,
    max_iterations: int = 500,
    tolerance: float = 1.0e-11,
) -> GARCHFitResult:
    """Fit one deterministic Gaussian GARCH(1,1) parameter set on GC only.

    ``omega``, ``alpha``, ``beta``, and persistence constraints are enforced by
    a smooth transformed parameterization.  Returns after ``fit_end`` never
    enter initialization, likelihood evaluation, or optimizer diagnostics.
    """

    if not np.isfinite(return_scale) or return_scale <= 0.0:
        raise GARCHInputError("return_scale must be finite and positive.")
    if not 0.0 < max_persistence < 1.0:
        raise GARCHInputError("max_persistence must be strictly between zero and one.")
    if min_observations < 20:
        raise GARCHInputError("min_observations must be at least 20.")
    if max_iterations < 0:
        raise GARCHInputError("max_iterations cannot be negative.")

    returns = _coerce_returns_frame(
        data,
        timestamp_column=timestamp_column,
        return_column=return_column,
        segment_column=segment_column,
    )
    fit_end_utc = _as_utc_timestamp(fit_end, end_of_day=True)
    timestamp = pd.to_datetime(returns["ts_event_utc"], utc=True)
    finite = np.isfinite(returns["log_return"].to_numpy(dtype=np.float64))
    fit_mask = (timestamp <= fit_end_utc).to_numpy() & finite
    fit_segments_raw = _segment_arrays(
        returns,
        value_column="log_return",
        mask=fit_mask,
    )
    n_fit = int(sum(segment.size for segment in fit_segments_raw))
    if n_fit < min_observations:
        raise GARCHInputError(
            "Insufficient pre-cutoff GC returns for GARCH fit: "
            f"n={n_fit}, required={min_observations}, fit_end={fit_end_utc.isoformat()}"
        )
    fit_segments = [segment * return_scale for segment in fit_segments_raw]
    pooled = np.concatenate(fit_segments)
    mean0 = float(np.mean(pooled))
    variance0 = float(np.var(pooled, ddof=0))
    if not np.isfinite(variance0) or variance0 <= 1.0e-14:
        raise GARCHInputError("Pre-cutoff GC returns must have positive finite variance.")

    persistence0 = min(0.90, max_persistence * 0.95)
    alpha0 = min(0.08, persistence0 * 0.25)
    omega0 = max(variance0 * (1.0 - persistence0), 1.0e-12)
    theta0 = np.array(
        [
            mean0,
            np.log(omega0),
            _logit(persistence0 / max_persistence),
            _logit(alpha0 / persistence0),
        ],
        dtype=np.float64,
    )
    mean_radius = max(10.0 * np.sqrt(variance0), abs(mean0) + 1.0)
    omega_lower = np.log(max(variance0 * 1.0e-10, 1.0e-14))
    omega_upper = np.log(max(variance0 * 10.0, 1.0e-12))
    bounds = [
        (mean0 - mean_radius, mean0 + mean_radius),
        (omega_lower, omega_upper),
        (-14.0, 14.0),
        (-14.0, 14.0),
    ]

    def objective(theta: np.ndarray) -> float:
        mean, omega, alpha, beta = _decode_parameters(
            theta, max_persistence=max_persistence
        )
        persistence = alpha + beta
        initial = omega / max(1.0 - persistence, 1.0e-12)
        if not np.isfinite(initial) or initial <= 0.0:
            return 1.0e100
        total = 0.0
        count = 0
        for segment in fit_segments:
            residuals = segment - mean
            variance = _variance_path(
                residuals,
                omega=omega,
                alpha=alpha,
                beta=beta,
                initial_variance=initial,
            )
            if (
                not np.isfinite(variance).all()
                or np.any(variance <= 0.0)
                or not np.isfinite(residuals).all()
            ):
                return 1.0e100
            total += 0.5 * float(
                np.sum(np.log(2.0 * np.pi) + np.log(variance) + residuals**2 / variance)
            )
            count += len(segment)
        return total / max(count, 1)

    optimizer = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(max_iterations),
            "ftol": float(tolerance),
            "gtol": 1.0e-8,
            "maxls": 50,
        },
    )
    optimized_theta = (
        np.asarray(optimizer.x, dtype=np.float64)
        if np.isfinite(np.asarray(optimizer.x, dtype=np.float64)).all()
        else theta0
    )
    mean_scaled, omega_scaled, alpha, beta = _decode_parameters(
        optimized_theta,
        max_persistence=max_persistence,
    )
    persistence = alpha + beta
    mean = mean_scaled / return_scale
    omega = omega_scaled / (return_scale**2)
    initial_variance = omega / max(1.0 - persistence, 1.0e-12)
    objective_value = float(objective(optimized_theta))
    parameters = GARCHParameters(
        mean=mean,
        omega=omega,
        alpha=alpha,
        beta=beta,
        initial_variance=initial_variance,
        return_scale=float(return_scale),
        fit_end_utc=fit_end_utc,
        n_fit_observations=n_fit,
        n_fit_segments=len(fit_segments),
        optimizer_method="L-BFGS-B",
        optimizer_success=bool(optimizer.success),
        optimizer_status=int(optimizer.status),
        optimizer_message=str(optimizer.message),
        objective_value=objective_value,
        iterations=int(getattr(optimizer, "nit", 0)),
    )
    max_fit_timestamp = timestamp.loc[fit_mask].max()
    records = [
        _diagnostic_record(
            "gc_only_input",
            True,
            value=1.0,
            threshold="products == {'GC'} after explicit filtering",
            details=f"excluded_non_gc_rows={returns.attrs.get('excluded_non_gc_rows', 0)}",
        ),
        _diagnostic_record(
            "fit_scope_ends_2022",
            bool(max_fit_timestamp <= fit_end_utc),
            value=float(max_fit_timestamp.value),
            threshold=f"timestamp <= {fit_end_utc.isoformat()}",
            details=f"max_fit_timestamp={max_fit_timestamp.isoformat()}",
        ),
        _diagnostic_record(
            "optimizer_success",
            bool(optimizer.success),
            value=float(optimizer.status),
            threshold="scipy success == True",
            details=f"{optimizer.message}; iterations={getattr(optimizer, 'nit', 0)}",
        ),
        _diagnostic_record(
            "objective_finite",
            np.isfinite(objective_value),
            value=objective_value,
            threshold="finite mean negative log likelihood",
            details="likelihood evaluated on scaled fit returns only",
        ),
        _diagnostic_record(
            "omega_positive",
            bool(np.isfinite(omega) and omega > 0.0),
            value=omega,
            threshold="omega > 0",
            details="raw log-return variance scale",
        ),
        _diagnostic_record(
            "alpha_nonnegative",
            bool(np.isfinite(alpha) and alpha >= 0.0),
            value=alpha,
            threshold="alpha >= 0",
            details="ARCH coefficient",
        ),
        _diagnostic_record(
            "beta_nonnegative",
            bool(np.isfinite(beta) and beta >= 0.0),
            value=beta,
            threshold="beta >= 0",
            details="GARCH coefficient",
        ),
        _diagnostic_record(
            "persistence_below_one",
            bool(np.isfinite(persistence) and persistence < 1.0),
            value=persistence,
            threshold="alpha + beta < 1",
            details="mathematical stationarity constraint",
        ),
        _diagnostic_record(
            "persistence_below_audit_limit",
            bool(np.isfinite(persistence) and persistence < DEFAULT_MAX_PERSISTENCE),
            value=persistence,
            threshold=f"alpha + beta < {DEFAULT_MAX_PERSISTENCE}",
            details="frozen audit rejection threshold",
        ),
    ]
    return GARCHFitResult(
        parameters=parameters,
        diagnostics=pd.DataFrame.from_records(records),
    )


def _parameter_object(
    parameters: GARCHParameters | GARCHFitResult | pd.DataFrame,
) -> GARCHParameters:
    if isinstance(parameters, GARCHFitResult):
        return parameters.parameters
    if isinstance(parameters, GARCHParameters):
        return parameters
    if isinstance(parameters, pd.DataFrame):
        return GARCHParameters.from_frame(parameters)
    raise TypeError("parameters must be GARCHParameters, GARCHFitResult, or a one-row DataFrame")


def _cumulative_garch_variance(
    one_step_variance: np.ndarray,
    parameters: GARCHParameters,
    *,
    horizon: int,
) -> np.ndarray:
    """Return the expected sum of the next ``horizon`` GARCH variances."""

    persistence = parameters.persistence
    unconditional = parameters.omega / (1.0 - persistence)
    decay_sum = (1.0 - persistence**horizon) / (1.0 - persistence)
    cumulative = (
        horizon * unconditional
        + (np.asarray(one_step_variance, dtype=np.float64) - unconditional) * decay_sum
    )
    return np.maximum(cumulative, MIN_VARIANCE)


def _finite_blocks(finite: np.ndarray) -> list[np.ndarray]:
    finite = np.asarray(finite, dtype=bool)
    positions = np.flatnonzero(finite)
    if positions.size == 0:
        return []
    split = np.flatnonzero(np.diff(positions) != 1) + 1
    return [block for block in np.split(positions, split) if block.size]


def _contiguous_run_bounds(runs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return half-open slices for adjacent continuity runs in O(N) time."""

    runs = np.asarray(runs, dtype=np.int64)
    if runs.ndim != 1:
        raise GARCHInputError("continuity run IDs must be one-dimensional.")
    if runs.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty
    starts = np.r_[0, np.flatnonzero(runs[1:] != runs[:-1]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], runs.size].astype(np.int64)
    return starts, ends


def _period_labels(
    timestamp: pd.Series,
    *,
    fit_end: pd.Timestamp,
    audit_start: pd.Timestamp,
    audit_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> np.ndarray:
    values = pd.to_datetime(timestamp, utc=True)
    labels = np.full(len(values), "out_of_scope", dtype=object)
    labels[(values <= fit_end).to_numpy()] = "fit"
    labels[((values >= audit_start) & (values <= audit_end)).to_numpy()] = (
        "development_audit"
    )
    labels[
        ((values >= validation_start) & (values <= validation_end)).to_numpy()
    ] = "validation"
    return labels


def _complete_rolling_sum(
    values: np.ndarray,
    runs: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    runs = np.asarray(runs, dtype=np.int64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    starts, ends = _contiguous_run_bounds(runs)
    for start, end in zip(starts, ends, strict=True):
        part = values[start:end]
        valid = np.isfinite(part)
        cumulative = np.r_[0.0, np.cumsum(np.where(valid, part, 0.0))]
        counts = np.r_[0, np.cumsum(valid.astype(np.int64))]
        endpoints = np.arange(window, len(part) + 1)
        window_starts = endpoints - window
        complete = (counts[endpoints] - counts[window_starts]) == window
        sums = cumulative[endpoints] - cumulative[window_starts]
        local = np.full(len(part), np.nan, dtype=np.float64)
        local[endpoints[complete] - 1] = sums[complete]
        output[start:end] = local
    return output


def _forward_realized_volatility(
    returns: np.ndarray,
    runs: np.ndarray,
    timestamps: pd.Series,
    *,
    horizon: int,
) -> tuple[np.ndarray, pd.Series]:
    returns = np.asarray(returns, dtype=np.float64)
    runs = np.asarray(runs, dtype=np.int64)
    realized = np.full(len(returns), np.nan, dtype=np.float64)
    end_ns = np.full(len(returns), np.datetime64("NaT"), dtype="datetime64[ns]")
    timestamp_ns = (
        pd.to_datetime(timestamps, utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype("datetime64[ns]")
    )
    starts, ends = _contiguous_run_bounds(runs)
    for start, end in zip(starts, ends, strict=True):
        part = returns[start:end]
        squared = np.square(part)
        valid = np.isfinite(squared)
        cumulative = np.r_[0.0, np.cumsum(np.where(valid, squared, 0.0))]
        counts = np.r_[0, np.cumsum(valid.astype(np.int64))]
        local_start = np.arange(0, max(len(part) - horizon, 0), dtype=np.int64)
        target_start = local_start + 1
        target_end = target_start + horizon
        complete = (counts[target_end] - counts[target_start]) == horizon
        selected = local_start[complete]
        sums = cumulative[target_end[complete]] - cumulative[target_start[complete]]
        realized[start + selected] = np.sqrt(np.maximum(sums, 0.0))
        end_ns[start + selected] = timestamp_ns[
            start + target_end[complete] - 1
        ]
    end_timestamp = pd.Series(pd.to_datetime(end_ns, utc=True), name="target_end_timestamp_utc")
    return realized, end_timestamp


def _atr_anchor(
    frame: pd.DataFrame,
    *,
    horizon: int,
) -> np.ndarray:
    atr = np.full(len(frame), np.nan, dtype=np.float64)
    for column in ("atr_20", "rolling_atr_20m", "decision_atr_20m"):
        if column in frame.columns:
            atr = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            break
    else:
        if not {"high", "low", "close"}.issubset(frame.columns):
            return atr
        high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
        low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
        close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
        previous = pd.Series(close).groupby(
            frame["continuity_run_id"].to_numpy(), sort=False
        ).shift(1)
        previous_close = previous.to_numpy(dtype=np.float64, na_value=np.nan)
        true_range = np.maximum.reduce(
            [high - low, np.abs(high - previous_close), np.abs(low - previous_close)]
        )
        first = frame["continuity_run_position"].to_numpy(dtype=np.int64) == 0
        true_range[first] = (high - low)[first]
        atr_sum = _complete_rolling_sum(
            true_range,
            frame["continuity_run_id"].to_numpy(dtype=np.int64),
            window=20,
        )
        atr = atr_sum / 20.0
    if "close" not in frame.columns:
        return np.full(len(frame), np.nan, dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    anchor = np.full(len(frame), np.nan, dtype=np.float64)
    valid = np.isfinite(atr) & np.isfinite(close) & (atr >= 0.0) & (close > 0.0)
    anchor[valid] = (atr[valid] / close[valid]) * np.sqrt(horizon / 20.0)
    return anchor


def forecast_segmented_garch(
    data: pd.DataFrame,
    parameters: GARCHParameters | GARCHFitResult | pd.DataFrame,
    *,
    timestamp_column: str = "ts_event_utc",
    return_column: str = "log_return",
    segment_column: str = "continuity_run_id",
    horizon_minutes: int = 60,
    audit_start: object = DEFAULT_AUDIT_START,
    audit_end: object = DEFAULT_AUDIT_END,
    validation_start: object = DEFAULT_VALIDATION_START,
    validation_end: object = DEFAULT_VALIDATION_END,
) -> pd.DataFrame:
    """Apply fixed parameters chronologically with a reset for every segment."""

    if horizon_minutes < 1:
        raise GARCHInputError("horizon_minutes must be positive.")
    parameter = _parameter_object(parameters)
    failures = parameter.mathematical_failures()
    if failures:
        raise GARCHInputError(f"Cannot forecast with invalid GARCH parameters: {failures}")
    frame = _coerce_returns_frame(
        data,
        timestamp_column=timestamp_column,
        return_column=return_column,
        segment_column=segment_column,
    )
    returns = frame["log_return"].to_numpy(dtype=np.float64)
    runs = frame["continuity_run_id"].to_numpy(dtype=np.int64)
    conditional = np.full(len(frame), np.nan, dtype=np.float64)
    next_variance = np.full(len(frame), np.nan, dtype=np.float64)
    standardized = np.full(len(frame), np.nan, dtype=np.float64)

    starts, ends = _contiguous_run_bounds(runs)
    for start, end in zip(starts, ends, strict=True):
        part = returns[start:end]
        finite = np.isfinite(part)
        local_positions = np.arange(start, end, dtype=np.int64)
        missing_positions = local_positions[~finite]
        conditional[missing_positions] = parameter.initial_variance
        next_variance[missing_positions] = parameter.initial_variance
        for block in _finite_blocks(finite):
            residuals = part[block] - parameter.mean
            variance = _variance_path(
                residuals,
                omega=parameter.omega,
                alpha=parameter.alpha,
                beta=parameter.beta,
                initial_variance=parameter.initial_variance,
            )
            absolute = start + block
            conditional[absolute] = variance
            next_variance[absolute] = (
                parameter.omega
                + parameter.alpha * np.square(residuals)
                + parameter.beta * variance
            )
            standardized[absolute] = residuals / np.sqrt(variance)

    if (
        not np.isfinite(conditional).all()
        or not np.isfinite(next_variance).all()
        or np.any(conditional <= 0.0)
        or np.any(next_variance <= 0.0)
    ):
        raise ArithmeticError("GARCH recursion produced non-positive or non-finite variance.")

    output = frame.copy()
    output["garch_conditional_variance_1m"] = conditional
    output["garch_variance_forecast_1m"] = next_variance
    output[f"garch_sigma_forecast_{horizon_minutes}m"] = np.sqrt(
        _cumulative_garch_variance(
            next_variance,
            parameter,
            horizon=horizon_minutes,
        )
    )
    output["standardized_residual"] = standardized
    realized, target_end = _forward_realized_volatility(
        returns,
        runs,
        output["ts_event_utc"],
        horizon=horizon_minutes,
    )
    output[f"realized_volatility_{horizon_minutes}m"] = realized
    output[f"target_{horizon_minutes}m_end_timestamp_utc"] = target_end
    trailing_variance = _complete_rolling_sum(
        np.square(returns),
        runs,
        window=horizon_minutes,
    )
    output[f"rv_anchor_sigma_{horizon_minutes}m"] = np.sqrt(
        np.maximum(trailing_variance, 0.0)
    )
    output[f"atr_anchor_sigma_{horizon_minutes}m"] = _atr_anchor(
        output,
        horizon=horizon_minutes,
    )
    output["audit_period"] = _period_labels(
        output["ts_event_utc"],
        fit_end=_as_utc_timestamp(parameter.fit_end_utc),
        audit_start=_as_utc_timestamp(audit_start),
        audit_end=_as_utc_timestamp(audit_end, end_of_day=True),
        validation_start=_as_utc_timestamp(validation_start),
        validation_end=_as_utc_timestamp(validation_end, end_of_day=True),
    )
    return output


def _default_method_columns(horizon_minutes: int) -> dict[str, str]:
    return {
        "garch": f"garch_sigma_forecast_{horizon_minutes}m",
        "realized_volatility_anchor": f"rv_anchor_sigma_{horizon_minutes}m",
        "atr_anchor": f"atr_anchor_sigma_{horizon_minutes}m",
    }


def fit_volatility_calibration(
    forecasts: pd.DataFrame,
    *,
    method_columns: Mapping[str, str] | None = None,
    target_column: str = "realized_volatility_60m",
    target_end_column: str = "target_60m_end_timestamp_utc",
    calibration_start: object = DEFAULT_AUDIT_START,
    calibration_end: object = DEFAULT_AUDIT_END,
    min_observations: int = 30,
) -> pd.DataFrame:
    """Fit zero-intercept forecast-scale multipliers on 2023 only."""

    if method_columns is None:
        method_columns = _default_method_columns(60)
    required = {
        "ts_event_utc",
        target_column,
        target_end_column,
        *method_columns.values(),
    }
    missing = sorted(required.difference(forecasts.columns))
    if missing:
        raise GARCHInputError(f"Calibration input is missing columns: {missing}")
    start = _as_utc_timestamp(calibration_start)
    end = _as_utc_timestamp(calibration_end, end_of_day=True)
    timestamps = pd.to_datetime(forecasts["ts_event_utc"], utc=True)
    target_end = pd.to_datetime(forecasts[target_end_column], utc=True)
    target = pd.to_numeric(forecasts[target_column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    base_scope = (
        (timestamps >= start)
        & (timestamps <= end)
        & (target_end <= end)
        & target_end.notna()
    ).to_numpy()
    records: list[dict[str, object]] = []
    for method, column in method_columns.items():
        raw = pd.to_numeric(forecasts[column], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        valid = (
            base_scope
            & np.isfinite(raw)
            & np.isfinite(target)
            & (raw > 0.0)
            & (target >= 0.0)
        )
        n_fit = int(valid.sum())
        denominator = float(np.dot(raw[valid], raw[valid])) if n_fit else np.nan
        if n_fit >= min_observations and np.isfinite(denominator) and denominator > 0.0:
            multiplier = max(float(np.dot(raw[valid], target[valid]) / denominator), 0.0)
            success = bool(np.isfinite(multiplier) and multiplier > 0.0)
            message = "frozen 2023 zero-intercept least-squares scale"
        else:
            multiplier = np.nan
            success = False
            message = (
                f"insufficient calibration support: n={n_fit}, required={min_observations}"
            )
        records.append(
            {
                "method": method,
                "raw_forecast_column": column,
                "calibrated_forecast_column": column.replace(
                    "_forecast_", "_calibrated_"
                ).replace("_anchor_sigma_", "_anchor_sigma_calibrated_"),
                "calibration_method": "nonnegative_zero_intercept_least_squares",
                "calibration_start_utc": start,
                "calibration_end_utc": end,
                "n_fit_observations": n_fit,
                "multiplier": multiplier,
                "intercept": 0.0,
                "success": success,
                "message": message,
            }
        )
    return pd.DataFrame.from_records(records)


def apply_volatility_calibration(
    forecasts: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a frozen calibration table without refitting."""

    required = {
        "raw_forecast_column",
        "calibrated_forecast_column",
        "multiplier",
        "success",
    }
    missing = sorted(required.difference(calibration.columns))
    if missing:
        raise GARCHInputError(f"Calibration table is missing columns: {missing}")
    result = forecasts.copy()
    for row in calibration.itertuples(index=False):
        raw_column = str(row.raw_forecast_column)
        output_column = str(row.calibrated_forecast_column)
        if raw_column not in result.columns:
            raise GARCHInputError(f"Forecast table is missing {raw_column!r}.")
        multiplier = float(row.multiplier)
        if bool(row.success) and np.isfinite(multiplier):
            raw = pd.to_numeric(result[raw_column], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            result[output_column] = raw * multiplier
        else:
            result[output_column] = np.nan
    return result


def _ljung_box_segmented(
    arrays: Sequence[np.ndarray],
    *,
    lags: int,
    squared: bool,
) -> tuple[float, float, int]:
    transformed: list[np.ndarray] = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        if squared:
            values = np.square(values)
        transformed.append(values)
    if not transformed:
        return np.nan, np.nan, 0
    pooled_mean = float(np.mean(np.concatenate(transformed)))
    centered = [values - pooled_mean for values in transformed]
    n_obs = int(sum(len(values) for values in centered))
    if n_obs <= lags + 2:
        return np.nan, np.nan, n_obs
    autocorrelations: list[float] = []
    pair_counts: list[int] = []
    for lag in range(1, lags + 1):
        numerator = 0.0
        left_ss = 0.0
        right_ss = 0.0
        pairs = 0
        for values in centered:
            if len(values) <= lag:
                continue
            left = values[lag:]
            right = values[:-lag]
            numerator += float(np.dot(left, right))
            left_ss += float(np.dot(left, left))
            right_ss += float(np.dot(right, right))
            pairs += len(left)
        denominator = np.sqrt(left_ss * right_ss)
        autocorrelations.append(numerator / denominator if denominator > 0.0 else 0.0)
        pair_counts.append(pairs)
    terms = [
        rho**2 / max(pairs, 1)
        for rho, pairs in zip(autocorrelations, pair_counts, strict=True)
    ]
    statistic = float(n_obs * (n_obs + 2.0) * np.sum(terms))
    return statistic, float(chi2.sf(statistic, lags)), n_obs


def _arch_lm_segmented(
    arrays: Sequence[np.ndarray],
    *,
    lags: int,
) -> tuple[float, float, int]:
    dimension = lags + 1
    xtx = np.zeros((dimension, dimension), dtype=np.float64)
    xty = np.zeros(dimension, dtype=np.float64)
    yty = 0.0
    ysum = 0.0
    n_obs = 0
    for array in arrays:
        squared = np.square(np.asarray(array, dtype=np.float64))
        if len(squared) <= lags:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(squared, lags + 1)
        y = windows[:, -1]
        lagged = windows[:, :-1][:, ::-1]
        design = np.column_stack([np.ones(len(y), dtype=np.float64), lagged])
        xtx += design.T @ design
        xty += design.T @ y
        yty += float(np.dot(y, y))
        ysum += float(np.sum(y))
        n_obs += len(y)
    if n_obs <= dimension:
        return np.nan, np.nan, n_obs
    coefficients = np.linalg.pinv(xtx, rcond=1.0e-12) @ xty
    residual_ss = float(yty - 2.0 * coefficients @ xty + coefficients @ xtx @ coefficients)
    total_ss = float(yty - ysum**2 / n_obs)
    r_squared = max(0.0, min(1.0, 1.0 - residual_ss / total_ss)) if total_ss > 0 else 0.0
    statistic = float(n_obs * r_squared)
    return statistic, float(chi2.sf(statistic, lags)), n_obs


def _residual_diagnostics(
    forecasts: pd.DataFrame,
    *,
    lags: Sequence[int],
    significance: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for period in ("fit", "development_audit", "validation"):
        period_frame = forecasts.loc[forecasts["audit_period"].eq(period)]
        arrays = _segment_arrays(period_frame, value_column="standardized_residual")
        for lag in lags:
            for squared, name in (
                (False, "ljung_box_standardized_residual"),
                (True, "ljung_box_squared_standardized_residual"),
            ):
                statistic, p_value, n_obs = _ljung_box_segmented(
                    arrays,
                    lags=int(lag),
                    squared=squared,
                )
                records.append(
                    {
                        "period": period,
                        "diagnostic": name,
                        "lags": int(lag),
                        "n_observations": n_obs,
                        "statistic": statistic,
                        "p_value": p_value,
                        "significance": significance,
                        "passed": bool(np.isfinite(p_value) and p_value > significance),
                    }
                )
        arch_lag = int(max(lags))
        statistic, p_value, n_obs = _arch_lm_segmented(arrays, lags=arch_lag)
        records.append(
            {
                "period": period,
                "diagnostic": "arch_lm_standardized_residual",
                "lags": arch_lag,
                "n_observations": n_obs,
                "statistic": statistic,
                "p_value": p_value,
                "significance": significance,
                "passed": bool(np.isfinite(p_value) and p_value > significance),
            }
        )
    return pd.DataFrame.from_records(records)


def _benchmark_metrics(
    forecasts: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    target_column: str,
    target_end_column: str,
    audit_start: pd.Timestamp,
    audit_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(forecasts["ts_event_utc"], utc=True)
    target_end = pd.to_datetime(forecasts[target_end_column], utc=True)
    target = pd.to_numeric(forecasts[target_column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    periods = {
        "development_audit": (audit_start, audit_end),
        "validation": (validation_start, validation_end),
    }
    records: list[dict[str, object]] = []
    for row in calibration.itertuples(index=False):
        method = str(row.method)
        column = str(row.calibrated_forecast_column)
        if column not in forecasts.columns:
            continue
        predicted = pd.to_numeric(forecasts[column], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        for period, (start, end) in periods.items():
            scope = (
                (timestamps >= start)
                & (timestamps <= end)
                & (target_end <= end)
                & target_end.notna()
            ).to_numpy()
            valid = (
                scope
                & np.isfinite(predicted)
                & np.isfinite(target)
                & (predicted > 0.0)
                & (target >= 0.0)
            )
            n_obs = int(valid.sum())
            if n_obs:
                error = predicted[valid] - target[valid]
                forecast_variance = np.square(predicted[valid])
                realized_variance = np.square(target[valid])
                ratio = np.maximum(
                    realized_variance / np.maximum(forecast_variance, MIN_VARIANCE),
                    MIN_VARIANCE,
                )
                qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
                correlation = (
                    float(np.corrcoef(predicted[valid], target[valid])[0, 1])
                    if n_obs > 2
                    and np.std(predicted[valid]) > 0.0
                    and np.std(target[valid]) > 0.0
                    else np.nan
                )
                record = {
                    "method": method,
                    "period": period,
                    "n_observations": n_obs,
                    "rmse_sigma": float(np.sqrt(np.mean(np.square(error)))),
                    "mae_sigma": float(np.mean(np.abs(error))),
                    "mean_forecast_sigma": float(np.mean(predicted[valid])),
                    "mean_realized_sigma": float(np.mean(target[valid])),
                    "mean_forecast_to_realized_ratio": float(
                        np.mean(predicted[valid]) / max(np.mean(target[valid]), MIN_VARIANCE)
                    ),
                    "qlike": qlike,
                    "pearson_correlation": correlation,
                }
            else:
                record = {
                    "method": method,
                    "period": period,
                    "n_observations": 0,
                    "rmse_sigma": np.nan,
                    "mae_sigma": np.nan,
                    "mean_forecast_sigma": np.nan,
                    "mean_realized_sigma": np.nan,
                    "mean_forecast_to_realized_ratio": np.nan,
                    "qlike": np.nan,
                    "pearson_correlation": np.nan,
                }
            records.append(record)
    return pd.DataFrame.from_records(records)


def _audit_checks(
    fit: GARCHFitResult,
    residual_diagnostics: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def add(check_name: str, passed: bool, details: str, *, critical: bool = True) -> None:
        records.append(
            {
                "category": "garch_audit",
                "check_name": check_name,
                "critical": critical,
                "passed": bool(passed),
                "details": details,
                "advancement_authorized": False,
            }
        )

    add("fit_critical_checks", fit.usable, "all optimizer and parameter gates")
    for period in ("fit", "development_audit", "validation"):
        period_residual = residual_diagnostics.loc[
            residual_diagnostics["period"].eq(period)
        ]
        add(
            f"{period}_residual_diagnostics",
            bool(
                not period_residual.empty
                and period_residual["passed"].astype(bool).all()
            ),
            "segment-aware Ljung-Box on residuals/squares and ARCH LM",
        )
    add(
        "development_calibration_success",
        bool(not calibration.empty and calibration["success"].astype(bool).all()),
        "multipliers fitted on 2023 only and frozen",
    )
    validation = benchmark_metrics.loc[
        benchmark_metrics["period"].eq("validation")
    ].set_index("method")
    garch_rmse = (
        float(validation.loc["garch", "rmse_sigma"])
        if "garch" in validation.index
        else np.nan
    )
    anchor_rmse = pd.to_numeric(
        validation.reindex(["realized_volatility_anchor", "atr_anchor"])["rmse_sigma"],
        errors="coerce",
    ).dropna()
    improves = bool(
        np.isfinite(garch_rmse)
        and not anchor_rmse.empty
        and garch_rmse < float(anchor_rmse.min())
    )
    add(
        "validation_improves_best_simple_anchor",
        improves,
        f"garch_rmse={garch_rmse}; anchor_rmse={anchor_rmse.to_dict()}",
    )
    add(
        "audit_only_no_feature_authorization",
        True,
        "Passing diagnostics do not add GARCH to the frozen v1 matrix.",
        critical=False,
    )
    return pd.DataFrame.from_records(records)


def audit_segmented_garch(
    data: pd.DataFrame,
    *,
    fit_end: object = DEFAULT_FIT_END,
    audit_start: object = DEFAULT_AUDIT_START,
    audit_end: object = DEFAULT_AUDIT_END,
    validation_start: object = DEFAULT_VALIDATION_START,
    validation_end: object = DEFAULT_VALIDATION_END,
    horizon_minutes: int = 60,
    diagnostic_lags: Sequence[int] = (10, 20),
    diagnostic_significance: float = 0.01,
    min_fit_observations: int = 250,
    min_calibration_observations: int = 30,
    max_iterations: int = 500,
) -> GARCHAuditResult:
    """Run the complete GC-only fit, forecast, diagnostic, and anchor audit."""

    if not diagnostic_lags or min(diagnostic_lags) < 1:
        raise GARCHInputError("diagnostic_lags must contain positive integers.")
    if not 0.0 < diagnostic_significance < 1.0:
        raise GARCHInputError("diagnostic_significance must be between zero and one.")
    fit = fit_segmented_garch(
        data,
        fit_end=fit_end,
        min_observations=min_fit_observations,
        max_iterations=max_iterations,
    )
    forecasts = forecast_segmented_garch(
        data,
        fit,
        horizon_minutes=horizon_minutes,
        audit_start=audit_start,
        audit_end=audit_end,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    methods = _default_method_columns(horizon_minutes)
    target_column = f"realized_volatility_{horizon_minutes}m"
    target_end_column = f"target_{horizon_minutes}m_end_timestamp_utc"
    calibration = fit_volatility_calibration(
        forecasts,
        method_columns=methods,
        target_column=target_column,
        target_end_column=target_end_column,
        calibration_start=audit_start,
        calibration_end=audit_end,
        min_observations=min_calibration_observations,
    )
    forecasts = apply_volatility_calibration(forecasts, calibration)
    residual_diagnostics = _residual_diagnostics(
        forecasts,
        lags=tuple(sorted({int(lag) for lag in diagnostic_lags})),
        significance=diagnostic_significance,
    )
    audit_start_utc = _as_utc_timestamp(audit_start)
    audit_end_utc = _as_utc_timestamp(audit_end, end_of_day=True)
    validation_start_utc = _as_utc_timestamp(validation_start)
    validation_end_utc = _as_utc_timestamp(validation_end, end_of_day=True)
    benchmark_metrics = _benchmark_metrics(
        forecasts,
        calibration,
        target_column=target_column,
        target_end_column=target_end_column,
        audit_start=audit_start_utc,
        audit_end=audit_end_utc,
        validation_start=validation_start_utc,
        validation_end=validation_end_utc,
    )
    checks = _audit_checks(
        fit,
        residual_diagnostics,
        calibration,
        benchmark_metrics,
    )
    return GARCHAuditResult(
        fit=fit,
        forecasts=forecasts,
        residual_diagnostics=residual_diagnostics,
        calibration=calibration,
        benchmark_metrics=benchmark_metrics,
        audit_checks=checks,
    )
