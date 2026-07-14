"""Leakage-free fixed-horizon labels for independent statistical research.

The theoretical entry is the open of the saved entry bar.  An ``h`` minute
path contains exactly ``h`` consecutive one-minute bars beginning at that
entry bar.  The last source bar starts at ``entry + (h - 1) minutes`` and its
close is economically available at ``entry + h minutes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa


FORWARD_HORIZONS_MINUTES = (5, 15, 30, 60, 120, 180)
GC_TICK_SIZE = 0.10
EXPANSION_QUANTILE = 0.80
TICK_GRID_TOLERANCE_TICKS = 1.0e-8

# Deterministic first-failure priority.  ``available`` is included so the
# saved categorical dictionary has one stable, documented set of meanings.
LABEL_REASON_CATEGORIES = (
    "available",
    "insufficient_future_rows",
    "missing_or_nonconsecutive_minute",
    "contract_or_instrument_change",
    "continuous_segment_change",
    "New_York_date_change",
    "tradability_roll_or_liquidity_failure",
    "forced_exit_boundary_breach",
    "invalid_price_data",
)

PATH_SOURCE_COLUMNS = (
    "source_row_id",
    "ts_event_utc",
    "ts_event_ny",
    "trade_date_ny",
    "product",
    "symbol",
    "active_symbol",
    "instrument_id",
    "continuous_segment_id",
    "open",
    "high",
    "low",
    "close",
    "tradable_research_flag",
    "roll_window_flag",
    "roll_window_type",
    "low_liquidity_active_day_flag",
    "low_liquidity_warning_flag",
    "rolling_atr_20m",
)

BASE_LABEL_COLUMNS = (
    "observation_id",
    "decision_bar_id",
    "entry_bar_id",
    "decision_timestamp_utc",
    "decision_timestamp_ny",
    "entry_timestamp_utc",
    "entry_timestamp_ny",
    "trade_date_ny",
    "entry_session",
    "research_partition",
    "product",
    "symbol",
    "active_symbol",
    "instrument_id",
    "continuous_segment_id",
    "forced_exit_timestamp_ny",
)


@dataclass(frozen=True)
class ForwardLabelBuildResult:
    """Outputs and compact build diagnostics."""

    labels: pd.DataFrame
    expansion_thresholds: pd.DataFrame
    tick_grid_validation: pd.DataFrame
    column_order: tuple[str, ...]


def _arrow_array(values: np.ndarray, valid: np.ndarray, arrow_type: pa.DataType) -> pd.arrays.ArrowExtensionArray:
    return pd.arrays.ArrowExtensionArray(pa.array(values, mask=~valid, type=arrow_type))


def _float_array(values: np.ndarray, valid: np.ndarray) -> pd.arrays.ArrowExtensionArray:
    return _arrow_array(values.astype(np.float64, copy=False), valid, pa.float64())


def _int_array(values: np.ndarray, valid: np.ndarray, arrow_type: pa.DataType) -> pd.arrays.ArrowExtensionArray:
    return _arrow_array(values, valid, arrow_type)


def _bool_array(values: np.ndarray, valid: np.ndarray) -> pd.arrays.ArrowExtensionArray:
    return _arrow_array(values.astype(bool, copy=False), valid, pa.bool_())


def _prefix_counts(flags: np.ndarray) -> np.ndarray:
    prefix = np.zeros(len(flags) + 1, dtype=np.int64)
    np.cumsum(flags, dtype=np.int64, out=prefix[1:])
    return prefix


def _timestamp_ns(values: pd.Series) -> np.ndarray:
    """Return UTC epoch nanoseconds independent of pandas' saved time unit."""

    return pd.to_datetime(values, utc=True).to_numpy(dtype="datetime64[ns]").view(np.int64)


def _map_source_ids(source_ids: np.ndarray, requested_ids: np.ndarray, label: str) -> np.ndarray:
    positions = np.searchsorted(source_ids, requested_ids)
    in_bounds = positions < len(source_ids)
    exact = np.zeros(len(positions), dtype=bool)
    exact[in_bounds] = source_ids[positions[in_bounds]] == requested_ids[in_bounds]
    if not exact.all():
        missing = requested_ids[~exact][:10].tolist()
        raise ValueError(f"{label} values are absent from the GC path source; examples={missing}")
    return positions.astype(np.int64, copy=False)


def validate_tick_grid(
    values: np.ndarray | pd.Series,
    *,
    tick_size: float = GC_TICK_SIZE,
    tolerance_ticks: float = TICK_GRID_TOLERANCE_TICKS,
    field_name: str = "price",
) -> dict[str, float | int | str]:
    """Validate finite prices against the GC tick grid without rounding defects away."""

    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    scaled = array[finite] / tick_size
    deviations = np.abs(scaled - np.rint(scaled))
    max_deviation = float(deviations.max(initial=0.0))
    off_grid_count = int(np.count_nonzero(deviations > tolerance_ticks))
    if off_grid_count:
        examples = array[finite][deviations > tolerance_ticks][:10].tolist()
        raise ValueError(
            f"{field_name} contains {off_grid_count:,} materially off-grid prices "
            f"(max deviation={max_deviation:.3e} ticks, examples={examples})."
        )
    return {
        "field": field_name,
        "finite_prices_checked": int(finite.sum()),
        "tick_size": float(tick_size),
        "tolerance_ticks": float(tolerance_ticks),
        "maximum_grid_deviation_ticks": max_deviation,
        "off_grid_prices": off_grid_count,
    }


def _validate_inputs(observations: pd.DataFrame, path_source: pd.DataFrame, horizons: tuple[int, ...]) -> None:
    missing_observation = sorted(set(BASE_LABEL_COLUMNS).difference(observations.columns))
    missing_source = sorted(set(PATH_SOURCE_COLUMNS).difference(path_source.columns))
    if missing_observation or missing_source:
        raise KeyError({"missing_observation_columns": missing_observation, "missing_source_columns": missing_source})
    if not horizons or tuple(sorted(set(horizons))) != horizons or any(h <= 0 for h in horizons):
        raise ValueError("Horizons must be unique positive integers in increasing order.")
    if observations["observation_id"].duplicated().any():
        raise ValueError("observation_id must be unique.")
    if not observations["observation_id"].is_monotonic_increasing:
        raise ValueError("observations must be ordered by observation_id.")
    if path_source["source_row_id"].duplicated().any() or not path_source["source_row_id"].is_monotonic_increasing:
        raise ValueError("path_source source_row_id must be unique and increasing.")


def _horizon_column_order(horizon: int) -> list[str]:
    h = horizon
    return [
        f"exit_timestamp_utc_{h}",
        f"exit_timestamp_ny_{h}",
        f"exit_price_{h}",
        f"forward_return_{h}_ticks",
        f"forward_return_{h}_bps",
        f"forward_return_{h}_atr",
        f"mfe_long_{h}_ticks",
        f"mfe_long_{h}_atr",
        f"mfe_short_{h}_ticks",
        f"mfe_short_{h}_atr",
        f"time_to_mfe_long_{h}_minutes",
        f"time_to_mfe_short_{h}_minutes",
        f"mae_long_{h}_ticks",
        f"mae_long_{h}_atr",
        f"mae_short_{h}_ticks",
        f"mae_short_{h}_atr",
        f"time_to_mae_long_{h}_minutes",
        f"time_to_mae_short_{h}_minutes",
        f"future_range_{h}_ticks",
        f"future_range_{h}_atr",
        f"future_realized_volatility_{h}_bps",
        f"direction_label_{h}",
        f"expansion_label_{h}",
        f"label_available_{h}",
        f"label_unavailable_reason_{h}",
    ]


def build_forward_label_table(
    observations: pd.DataFrame,
    path_source: pd.DataFrame,
    *,
    horizons: Iterable[int] = FORWARD_HORIZONS_MINUTES,
    tick_size: float = GC_TICK_SIZE,
    expansion_quantile: float = EXPANSION_QUANTILE,
    chunk_size: int = 20_000,
) -> ForwardLabelBuildResult:
    """Build fixed-horizon return, excursion, range, volatility, and class labels.

    Explicit path matrices are created only for available observations in
    bounded chunks and only for high/low extrema and first-extreme timing.
    Validity and realized-volatility components use reusable prefix arrays.
    """

    horizons = tuple(int(h) for h in horizons)
    _validate_inputs(observations, path_source, horizons)
    if not (0.0 < expansion_quantile < 1.0):
        raise ValueError("expansion_quantile must lie strictly between zero and one.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    n_observations = len(observations)
    n_bars = len(path_source)
    source_ids = path_source["source_row_id"].to_numpy(dtype=np.int64)
    entry_ids = observations["entry_bar_id"].to_numpy(dtype=np.int64)
    decision_ids = observations["decision_bar_id"].to_numpy(dtype=np.int64)
    entry_positions = _map_source_ids(source_ids, entry_ids, "entry_bar_id")
    decision_positions = _map_source_ids(source_ids, decision_ids, "decision_bar_id")

    timestamps_utc_ns = _timestamp_ns(path_source["ts_event_utc"])
    timestamps_ny_ns = _timestamp_ns(path_source["ts_event_ny"])
    source_dates = pd.to_datetime(path_source["trade_date_ny"]).to_numpy(dtype="datetime64[D]")
    products = path_source["product"].astype(str).to_numpy()
    symbols = path_source["symbol"].astype(str).to_numpy()
    active_symbols = path_source["active_symbol"].astype(str).to_numpy()
    instrument_ids = path_source["instrument_id"].to_numpy(dtype=np.int64)
    segment_ids = path_source["continuous_segment_id"].to_numpy(dtype=np.int64)
    opens = path_source["open"].to_numpy(dtype=np.float64, na_value=np.nan)
    highs = path_source["high"].to_numpy(dtype=np.float64, na_value=np.nan)
    lows = path_source["low"].to_numpy(dtype=np.float64, na_value=np.nan)
    closes = path_source["close"].to_numpy(dtype=np.float64, na_value=np.nan)
    atr = path_source["rolling_atr_20m"].to_numpy(dtype=np.float64, na_value=np.nan)

    grid_records = [
        validate_tick_grid(values, tick_size=tick_size, field_name=field)
        for field, values in (("open", opens), ("high", highs), ("low", lows), ("close", closes))
    ]
    tick_grid_validation = pd.DataFrame(grid_records)

    valid_ohlc = (
        np.isfinite(opens)
        & np.isfinite(highs)
        & np.isfinite(lows)
        & np.isfinite(closes)
        & (opens > 0.0)
        & (highs > 0.0)
        & (lows > 0.0)
        & (closes > 0.0)
        & (highs >= np.maximum(opens, closes))
        & (lows <= np.minimum(opens, closes))
        & (highs >= lows)
    )
    path_control_valid = (
        path_source["tradable_research_flag"].fillna(False).to_numpy(dtype=bool)
        & ~path_source["roll_window_flag"].fillna(True).to_numpy(dtype=bool)
        & path_source["roll_window_type"].eq("none").fillna(False).to_numpy(dtype=bool)
        & ~path_source["low_liquidity_active_day_flag"].fillna(True).to_numpy(dtype=bool)
        & ~path_source["low_liquidity_warning_flag"].fillna(True).to_numpy(dtype=bool)
    )

    transition_time_bad = np.zeros(n_bars, dtype=bool)
    transition_contract_bad = np.zeros(n_bars, dtype=bool)
    transition_segment_bad = np.zeros(n_bars, dtype=bool)
    transition_date_bad = np.zeros(n_bars, dtype=bool)
    transition_time_bad[1:] = np.diff(timestamps_utc_ns) != 60_000_000_000
    transition_contract_bad[1:] = (
        (products[1:] != products[:-1])
        | (symbols[1:] != symbols[:-1])
        | (active_symbols[1:] != active_symbols[:-1])
        | (instrument_ids[1:] != instrument_ids[:-1])
    )
    transition_segment_bad[1:] = segment_ids[1:] != segment_ids[:-1]
    transition_date_bad[1:] = source_dates[1:] != source_dates[:-1]

    prefix_time_bad = _prefix_counts(transition_time_bad)
    prefix_contract_bad = _prefix_counts(transition_contract_bad)
    prefix_segment_bad = _prefix_counts(transition_segment_bad)
    prefix_date_bad = _prefix_counts(transition_date_bad)
    prefix_control_bad = _prefix_counts(~path_control_valid)
    prefix_price_bad = _prefix_counts(~valid_ohlc)

    close_to_close_sq = np.zeros(n_bars, dtype=np.float64)
    pair_close_valid = valid_ohlc[1:] & valid_ohlc[:-1]
    close_to_close_sq[1:][pair_close_valid] = np.square(
        np.log(closes[1:][pair_close_valid] / closes[:-1][pair_close_valid])
    )
    prefix_close_to_close_sq = np.zeros(n_bars + 1, dtype=np.float64)
    np.cumsum(close_to_close_sq, out=prefix_close_to_close_sq[1:])
    open_to_close_sq = np.full(n_bars, np.nan, dtype=np.float64)
    open_to_close_sq[valid_ohlc] = np.square(np.log(closes[valid_ohlc] / opens[valid_ohlc]))

    observed_entry_utc_ns = _timestamp_ns(observations["entry_timestamp_utc"])
    observed_entry_ny_ns = _timestamp_ns(observations["entry_timestamp_ny"])
    forced_exit_ny_ns = _timestamp_ns(observations["forced_exit_timestamp_ny"])
    if not np.array_equal(timestamps_utc_ns[entry_positions], observed_entry_utc_ns):
        raise ValueError("Saved entry timestamps do not match entry_bar_id in the path source.")
    if not np.array_equal(timestamps_ny_ns[entry_positions], observed_entry_ny_ns):
        raise ValueError("Saved New York entry timestamps do not match entry_bar_id in the path source.")
    if not np.array_equal(timestamps_utc_ns[decision_positions], _timestamp_ns(observations["decision_timestamp_utc"])):
        raise ValueError("Saved decision timestamps do not match decision_bar_id in the path source.")

    entry_prices = opens[entry_positions]
    decision_atr = atr[decision_positions]
    atr_available = np.isfinite(decision_atr) & (decision_atr > 0.0)
    if not valid_ohlc[entry_positions].all():
        raise ValueError("At least one saved entry bar has invalid OHLC data.")

    labels = pd.concat(
        [
            observations.loc[:, BASE_LABEL_COLUMNS].copy(),
            pd.DataFrame(
                {
                    "entry_price": pd.arrays.ArrowExtensionArray(pa.array(entry_prices, type=pa.float64())),
                    "decision_atr_20m": pd.arrays.ArrowExtensionArray(
                        pa.array(decision_atr, mask=~np.isfinite(decision_atr), type=pa.float64())
                    ),
                    "atr_normalization_available": pd.arrays.ArrowExtensionArray(
                        pa.array(atr_available, type=pa.bool_())
                    ),
                },
                index=observations.index,
            ),
        ],
        axis=1,
    )
    horizon_columns: dict[str, object] = {}
    base_order = list(BASE_LABEL_COLUMNS) + ["entry_price", "decision_atr_20m", "atr_normalization_available"]

    availability_by_horizon: dict[int, np.ndarray] = {}
    for h in horizons:
        within = entry_positions + h <= n_bars
        time_bad = np.zeros(n_observations, dtype=bool)
        contract_bad = np.zeros(n_observations, dtype=bool)
        segment_bad = np.zeros(n_observations, dtype=bool)
        date_bad = np.zeros(n_observations, dtype=bool)
        control_bad = np.zeros(n_observations, dtype=bool)
        forced_exit_bad = observed_entry_ny_ns + h * 60_000_000_000 > forced_exit_ny_ns
        price_bad = np.zeros(n_observations, dtype=bool)
        within_rows = np.flatnonzero(within)
        starts = entry_positions[within_rows]
        ends_exclusive = starts + h
        time_bad[within_rows] = prefix_time_bad[ends_exclusive] - prefix_time_bad[starts + 1] > 0
        contract_bad[within_rows] = prefix_contract_bad[ends_exclusive] - prefix_contract_bad[starts + 1] > 0
        segment_bad[within_rows] = prefix_segment_bad[ends_exclusive] - prefix_segment_bad[starts + 1] > 0
        date_bad[within_rows] = prefix_date_bad[ends_exclusive] - prefix_date_bad[starts + 1] > 0
        control_bad[within_rows] = prefix_control_bad[ends_exclusive] - prefix_control_bad[starts] > 0
        price_bad[within_rows] = prefix_price_bad[ends_exclusive] - prefix_price_bad[starts] > 0

        reasons = np.full(n_observations, "available", dtype=object)
        remaining = np.ones(n_observations, dtype=bool)
        failure_masks = (
            ("insufficient_future_rows", ~within),
            ("missing_or_nonconsecutive_minute", time_bad),
            ("contract_or_instrument_change", contract_bad),
            ("continuous_segment_change", segment_bad),
            ("New_York_date_change", date_bad),
            ("tradability_roll_or_liquidity_failure", control_bad),
            ("forced_exit_boundary_breach", forced_exit_bad),
            ("invalid_price_data", price_bad),
        )
        for reason, failed in failure_masks:
            first_failure = remaining & failed
            reasons[first_failure] = reason
            remaining[first_failure] = False
        available = remaining
        availability_by_horizon[h] = available

        exit_prices = np.full(n_observations, np.nan, dtype=np.float64)
        forward_price = np.full(n_observations, np.nan, dtype=np.float64)
        max_high = np.full(n_observations, np.nan, dtype=np.float64)
        min_low = np.full(n_observations, np.nan, dtype=np.float64)
        time_max = np.zeros(n_observations, dtype=np.int16)
        time_min = np.zeros(n_observations, dtype=np.int16)
        realized_vol_bps = np.full(n_observations, np.nan, dtype=np.float64)
        available_rows = np.flatnonzero(available)
        offsets = np.arange(h, dtype=np.int64)
        for chunk_start in range(0, len(available_rows), chunk_size):
            rows = available_rows[chunk_start : chunk_start + chunk_size]
            starts_chunk = entry_positions[rows]
            window_indices = starts_chunk[:, None] + offsets
            high_windows = highs[window_indices]
            max_high[rows] = high_windows.max(axis=1)
            time_max[rows] = (high_windows.argmax(axis=1) + 1).astype(np.int16)
            del high_windows
            low_windows = lows[window_indices]
            min_low[rows] = low_windows.min(axis=1)
            time_min[rows] = (low_windows.argmin(axis=1) + 1).astype(np.int16)
            del low_windows, window_indices

        ends = entry_positions[available_rows] + h - 1
        exit_prices[available_rows] = closes[ends]
        forward_price[available_rows] = exit_prices[available_rows] - entry_prices[available_rows]
        rv_sum_sq = (
            open_to_close_sq[entry_positions[available_rows]]
            + prefix_close_to_close_sq[ends + 1]
            - prefix_close_to_close_sq[entry_positions[available_rows] + 1]
        )
        realized_vol_bps[available_rows] = np.sqrt(np.maximum(rv_sum_sq, 0.0)) * 10_000.0

        mfe_long_price = max_high - entry_prices
        mfe_short_price = entry_prices - min_low
        mae_long_price = mfe_short_price.copy()
        mae_short_price = mfe_long_price.copy()
        future_range_price = max_high - min_low
        tick_price_arrays = {
            "forward": forward_price,
            "mfe_long": mfe_long_price,
            "mfe_short": mfe_short_price,
            "mae_long": mae_long_price,
            "mae_short": mae_short_price,
            "range": future_range_price,
        }
        raw_ticks: dict[str, np.ndarray] = {}
        for name, price_values in tick_price_arrays.items():
            tick_values = np.zeros(n_observations, dtype=np.int32)
            tick_values[available] = np.rint(price_values[available] / tick_size).astype(np.int32)
            raw_ticks[name] = tick_values
        forward_bps = np.full(n_observations, np.nan, dtype=np.float64)
        forward_bps[available] = (exit_prices[available] / entry_prices[available] - 1.0) * 10_000.0
        atr_valid_for_horizon = available & atr_available

        economic_exit_ns = np.full(n_observations, np.iinfo(np.int64).min, dtype=np.int64)
        economic_exit_ns[available] = observed_entry_utc_ns[available] + h * 60_000_000_000
        exit_utc = pd.Series(pd.to_datetime(economic_exit_ns, utc=True), index=labels.index)
        exit_ny = exit_utc.dt.tz_convert("America/New_York")
        horizon_columns[f"exit_timestamp_utc_{h}"] = exit_utc
        horizon_columns[f"exit_timestamp_ny_{h}"] = exit_ny
        horizon_columns[f"exit_price_{h}"] = _float_array(exit_prices, available)
        horizon_columns[f"forward_return_{h}_ticks"] = _int_array(raw_ticks["forward"], available, pa.int32())
        horizon_columns[f"forward_return_{h}_bps"] = _float_array(forward_bps, available)
        normalized_forward = np.full(n_observations, np.nan, dtype=np.float64)
        normalized_mfe_long = np.full(n_observations, np.nan, dtype=np.float64)
        normalized_mfe_short = np.full(n_observations, np.nan, dtype=np.float64)
        normalized_range = np.full(n_observations, np.nan, dtype=np.float64)
        normalized_forward[atr_valid_for_horizon] = (
            forward_price[atr_valid_for_horizon] / decision_atr[atr_valid_for_horizon]
        )
        normalized_mfe_long[atr_valid_for_horizon] = (
            mfe_long_price[atr_valid_for_horizon] / decision_atr[atr_valid_for_horizon]
        )
        normalized_mfe_short[atr_valid_for_horizon] = (
            mfe_short_price[atr_valid_for_horizon] / decision_atr[atr_valid_for_horizon]
        )
        normalized_range[atr_valid_for_horizon] = (
            future_range_price[atr_valid_for_horizon] / decision_atr[atr_valid_for_horizon]
        )
        horizon_columns[f"forward_return_{h}_atr"] = _float_array(normalized_forward, atr_valid_for_horizon)
        horizon_columns[f"mfe_long_{h}_ticks"] = _int_array(raw_ticks["mfe_long"], available, pa.int32())
        horizon_columns[f"mfe_long_{h}_atr"] = _float_array(normalized_mfe_long, atr_valid_for_horizon)
        horizon_columns[f"mfe_short_{h}_ticks"] = _int_array(raw_ticks["mfe_short"], available, pa.int32())
        horizon_columns[f"mfe_short_{h}_atr"] = _float_array(normalized_mfe_short, atr_valid_for_horizon)
        horizon_columns[f"time_to_mfe_long_{h}_minutes"] = _int_array(time_max, available, pa.int16())
        horizon_columns[f"time_to_mfe_short_{h}_minutes"] = _int_array(time_min, available, pa.int16())
        horizon_columns[f"mae_long_{h}_ticks"] = _int_array(raw_ticks["mae_long"], available, pa.int32())
        horizon_columns[f"mae_long_{h}_atr"] = _float_array(normalized_mfe_short, atr_valid_for_horizon)
        horizon_columns[f"mae_short_{h}_ticks"] = _int_array(raw_ticks["mae_short"], available, pa.int32())
        horizon_columns[f"mae_short_{h}_atr"] = _float_array(normalized_mfe_long, atr_valid_for_horizon)
        horizon_columns[f"time_to_mae_long_{h}_minutes"] = _int_array(time_min, available, pa.int16())
        horizon_columns[f"time_to_mae_short_{h}_minutes"] = _int_array(time_max, available, pa.int16())
        horizon_columns[f"future_range_{h}_ticks"] = _int_array(raw_ticks["range"], available, pa.int32())
        horizon_columns[f"future_range_{h}_atr"] = _float_array(normalized_range, atr_valid_for_horizon)
        horizon_columns[f"future_realized_volatility_{h}_bps"] = _float_array(realized_vol_bps, available)
        direction = np.sign(raw_ticks["forward"]).astype(np.int8)
        horizon_columns[f"direction_label_{h}"] = _int_array(direction, available, pa.int8())
        horizon_columns[f"label_available_{h}"] = pd.arrays.ArrowExtensionArray(pa.array(available, type=pa.bool_()))
        horizon_columns[f"label_unavailable_reason_{h}"] = pd.Categorical(
            reasons, categories=LABEL_REASON_CATEGORIES, ordered=True
        )

    threshold_records: list[dict[str, object]] = []
    development = labels["research_partition"].eq("Development").to_numpy(dtype=bool)
    for h in horizons:
        valid = development & availability_by_horizon[h] & atr_available
        values = pd.Series(horizon_columns[f"future_range_{h}_atr"]).to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        fit_values = values[valid]
        threshold = float(np.quantile(fit_values, expansion_quantile, method="linear")) if len(fit_values) else np.nan
        expansion_valid = availability_by_horizon[h] & atr_available & np.isfinite(threshold)
        expansion = np.zeros(n_observations, dtype=bool)
        expansion[expansion_valid] = values[expansion_valid] >= threshold
        horizon_columns[f"expansion_label_{h}"] = _bool_array(expansion, expansion_valid)
        threshold_records.append(
            {
                "horizon": h,
                "quantile": expansion_quantile,
                "threshold": threshold,
                "development_observation_count_used": int(valid.sum()),
                "fitting_partition": "Development",
                "normalization_denominator": "decision_atr_20m",
            }
        )

    expansion_thresholds = pd.DataFrame(threshold_records)
    expansion_thresholds["horizon"] = expansion_thresholds["horizon"].astype("int16[pyarrow]")
    expansion_thresholds["development_observation_count_used"] = expansion_thresholds[
        "development_observation_count_used"
    ].astype("int64[pyarrow]")
    expansion_thresholds["fitting_partition"] = pd.Categorical(
        expansion_thresholds["fitting_partition"], categories=["Development"], ordered=True
    )
    expansion_thresholds["normalization_denominator"] = pd.Categorical(
        expansion_thresholds["normalization_denominator"], categories=["decision_atr_20m"], ordered=True
    )

    labels = pd.concat([labels, pd.DataFrame(horizon_columns, index=labels.index)], axis=1)
    final_order = base_order + [column for h in horizons for column in _horizon_column_order(h)]
    labels = labels.loc[:, final_order]
    return ForwardLabelBuildResult(
        labels=labels,
        expansion_thresholds=expansion_thresholds,
        tick_grid_validation=tick_grid_validation,
        column_order=tuple(final_order),
    )


def build_label_availability_report(
    labels: pd.DataFrame,
    *,
    horizons: Iterable[int] = FORWARD_HORIZONS_MINUTES,
) -> pd.DataFrame:
    """Return reason-level mechanical coverage by session and partition."""

    records: list[pd.DataFrame] = []
    for horizon in horizons:
        h = int(horizon)
        available_column = f"label_available_{h}"
        reason_column = f"label_unavailable_reason_{h}"
        base = labels[["entry_session", "research_partition", available_column, reason_column]].copy()
        totals = (
            base.groupby(["entry_session", "research_partition"], observed=True)[available_column]
            .agg(["sum", "count"])
            .rename(columns={"sum": "available_observations", "count": "total_observations"})
        )
        totals["unavailable_observations"] = totals["total_observations"] - totals["available_observations"]
        totals["availability_rate"] = totals["available_observations"] / totals["total_observations"]
        reasons = (
            base.groupby(["entry_session", "research_partition", reason_column], observed=True)
            .size()
            .rename("reason_observations")
            .reset_index()
            .rename(columns={reason_column: "unavailable_reason", "entry_session": "session"})
        )
        joined = reasons.merge(
            totals.reset_index().rename(columns={"entry_session": "session"}),
            on=["session", "research_partition"],
            how="left",
            validate="many_to_one",
        )
        joined.insert(0, "horizon", h)
        records.append(joined)
    report = pd.concat(records, ignore_index=True)
    return report[
        [
            "horizon",
            "available_observations",
            "unavailable_observations",
            "availability_rate",
            "unavailable_reason",
            "reason_observations",
            "session",
            "research_partition",
        ]
    ]
