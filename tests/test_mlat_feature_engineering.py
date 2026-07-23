from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statistical_research.mlat_feature_engineering import (
    BAR_INPUT_COLUMNS,
    FEATURE_METADATA_COLUMNS,
    MLAT_COMPLETE_HISTORY_BARS,
    MLAT_FEATURE_SOURCE_COLUMNS,
    MLAT_METADATA_COLUMNS,
    build_mlat_feature_matrix,
)
from statistical_research.mlat_feature_registry import (
    MLAT_FEATURE_NAMES,
    build_mlat_registry,
)


def _bars(size: int = 100) -> pd.DataFrame:
    index = np.arange(size, dtype=float)
    utc = pd.date_range("2024-03-10 06:30", periods=size, freq="min", tz="UTC")
    ny = utc.tz_convert("America/New_York")
    close = 2000.0 + 0.07 * index + 0.13 * np.sin(index / 3.0)
    open_ = close - 0.04 * np.cos(index / 4.0)
    high = np.maximum(open_, close) + 0.20 + 0.01 * np.sin(index)
    low = np.minimum(open_, close) - 0.18 - 0.01 * np.cos(index)
    frame = pd.DataFrame(
        {
            "source_row_id": 10_000 + np.arange(size, dtype=np.int64),
            "ts_event_utc": utc,
            "ts_event_ny": ny,
            "trade_date_ny": ny.date,
            "product": "GC",
            "symbol": "GCJ4",
            "active_symbol": "GCJ4",
            "instrument_id": 101,
            "continuous_segment_id": 7,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0 + (index % 17),
            "tradable_research_flag": True,
            "roll_window_flag": False,
            "low_liquidity_warning_flag": False,
        }
    )
    return frame[list(MLAT_FEATURE_SOURCE_COLUMNS)]


def _observations(bars: pd.DataFrame, positions: np.ndarray | None = None) -> pd.DataFrame:
    if positions is None:
        positions = np.arange(len(bars), dtype=np.int64)
    selected = bars.iloc[positions].reset_index(drop=True)
    entry_utc = pd.to_datetime(selected["ts_event_utc"], utc=True) + pd.Timedelta(
        minutes=1
    )
    entry_ny = entry_utc.dt.tz_convert("America/New_York")
    return pd.DataFrame(
        {
            "observation_id": np.arange(len(selected), dtype=np.int64),
            "decision_timestamp_utc": selected["ts_event_utc"],
            "decision_timestamp_ny": selected["ts_event_ny"],
            "entry_timestamp_utc": entry_utc,
            "entry_timestamp_ny": entry_ny,
            "trade_date_ny": selected["trade_date_ny"],
            "research_partition": "Development",
            "product": selected["product"],
            "symbol": selected["symbol"],
            "active_symbol": selected["active_symbol"],
            "instrument_id": selected["instrument_id"],
            "continuous_segment_id": selected["continuous_segment_id"],
            "decision_bar_id": selected["source_row_id"],
        }
    )


def _manual_last_row(bars: pd.DataFrame) -> dict[str, float]:
    open_ = bars["open"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    volume = bars["volume"].to_numpy(float)
    returns = np.log(close[1:] / close[:-1])

    close_20 = close[-20:]
    delta_14 = np.diff(close[-15:])
    gains = np.maximum(delta_14, 0)
    losses = np.maximum(-delta_14, 0)
    multiplier = (2 * close[-20:] - high[-20:] - low[-20:]) / (
        high[-20:] - low[-20:]
    )
    parkinson_terms = np.log(high[-30:] / low[-30:]) ** 2
    rs_terms = np.log(high[-30:] / open_[-30:]) * np.log(
        high[-30:] / close[-30:]
    ) + np.log(low[-30:] / open_[-30:]) * np.log(low[-30:] / close[-30:])
    returns_60 = returns[-60:]
    upside = np.sum(returns_60[returns_60 > 0] ** 2)
    downside = np.sum(returns_60[returns_60 < 0] ** 2)
    rv = np.sum(returns_60**2)
    bipower = (
        (np.pi / 2)
        * (60 / 59)
        * np.sum(np.abs(returns[-60:]) * np.abs(returns[-61:-1]))
    )
    five_minute_returns = np.convolve(returns, np.ones(5), mode="valid")
    counts = np.array(
        [
            np.count_nonzero(returns_60 < 0),
            np.count_nonzero(returns_60 == 0),
            np.count_nonzero(returns_60 > 0),
        ],
        dtype=float,
    )
    probabilities = counts / 60
    nonzero = probabilities > 0
    rv15 = 1.0e4 * np.sqrt(
        np.convolve(returns**2, np.ones(15) / 15, mode="valid")
    )
    return {
        "bollinger_zscore_20": (
            close[-1] - np.mean(close_20)
        ) / np.std(close_20, ddof=0),
        "bollinger_bandwidth_20": 4 * np.std(close_20, ddof=0) / np.mean(close_20),
        "cutler_rsi_14": 100 * np.sum(gains) / (np.sum(gains) + np.sum(losses)),
        "chaikin_money_flow_20": np.sum(volume[-20:] * multiplier)
        / np.sum(volume[-20:]),
        "amihud_illiquidity_60": 1.0e9
        * np.mean(np.abs(returns_60) / (close[-60:] * volume[-60:])),
        "parkinson_volatility_30": 1.0e4
        * np.sqrt(np.mean(parkinson_terms) / (4 * np.log(2))),
        "rogers_satchell_volatility_30": 1.0e4 * np.sqrt(np.mean(rs_terms)),
        "realized_semivariance_balance_60": (upside - downside)
        / (upside + downside),
        "bipower_jump_ratio_60": max(rv - bipower, 0) / rv,
        "variance_ratio_60_5": np.var(five_minute_returns[-60:], ddof=0)
        / (5 * np.var(returns_60, ddof=0)),
        "return_sign_entropy_60": -np.sum(
            probabilities[nonzero] * np.log(probabilities[nonzero])
        )
        / np.log(3),
        "volatility_of_volatility_60": np.std(rv15[-60:], ddof=0)
        / np.mean(rv15[-60:]),
    }


def test_manual_formula_reconstruction_and_first_valid_positions() -> None:
    bars = _bars()
    result = build_mlat_feature_matrix(bars, _observations(bars))
    expected = _manual_last_row(bars)

    for feature, value in expected.items():
        assert result.matrix.iloc[-1][feature] == pytest.approx(
            value, rel=2.0e-5, abs=1.0e-7
        )
    for feature, history in MLAT_COMPLETE_HISTORY_BARS.items():
        assert result.matrix[feature].first_valid_index() == history - 1
    assert result.persistence_validation["passed"].all()


def test_future_and_entry_bar_mutation_cannot_change_decision_features() -> None:
    bars = _bars()
    observations = _observations(bars, np.array([75], dtype=np.int64))
    baseline = build_mlat_feature_matrix(bars, observations).matrix

    mutated = bars.copy()
    mutated.loc[76:, ["open", "high", "low", "close", "volume"]] *= 10
    changed = build_mlat_feature_matrix(mutated, observations).matrix
    pd.testing.assert_frame_equal(
        baseline[list(MLAT_FEATURE_NAMES)],
        changed[list(MLAT_FEATURE_NAMES)],
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "GCM4"),
        ("active_symbol", "GCM4"),
        ("instrument_id", 202),
        ("continuous_segment_id", 8),
        ("roll_window_flag", True),
        ("tradable_research_flag", False),
        ("low_liquidity_warning_flag", True),
    ],
)
def test_locked_boundaries_reset_complete_windows(field: str, value: object) -> None:
    bars = _bars()
    bars.loc[40, field] = value
    observations = _observations(bars)
    result = build_mlat_feature_matrix(bars, observations)

    audit = result.observation_audit.set_index("source_position")
    assert audit.loc[40, "bars_since_continuity_start"] == 0
    assert result.matrix.loc[40, list(MLAT_FEATURE_NAMES)].isna().all()


def test_missing_minute_resets_without_arbitrary_session_or_dst_reset() -> None:
    continuous = _bars()
    continuous_result = build_mlat_feature_matrix(
        continuous, _observations(continuous)
    )
    assert continuous_result.runtime_summary.loc[0, "continuity_runs"] == 1
    assert str(continuous_result.matrix["decision_timestamp_ny"].dt.tz) == (
        "America/New_York"
    )
    local_hours = continuous_result.matrix["decision_timestamp_ny"].dt.hour
    assert 1 in set(local_hours) and 3 in set(local_hours)

    gapped = continuous.drop(index=40).reset_index(drop=True)
    gapped_result = build_mlat_feature_matrix(gapped, _observations(gapped))
    source_position_after_gap = 40
    audit = gapped_result.observation_audit.set_index("source_position")
    assert audit.loc[source_position_after_gap, "bars_since_continuity_start"] == 0
    assert gapped_result.matrix.loc[
        source_position_after_gap, list(MLAT_FEATURE_NAMES)
    ].isna().all()


def test_zero_denominators_ranges_volume_and_missing_values_are_honest() -> None:
    bars = _bars()
    bars[["open", "high", "low", "close"]] = 2000.0
    bars["volume"] = 100.0
    result = build_mlat_feature_matrix(bars, _observations(bars)).matrix.iloc[-1]
    assert np.isnan(result["bollinger_zscore_20"])
    assert result["bollinger_bandwidth_20"] == 0
    assert np.isnan(result["cutler_rsi_14"])
    assert result["chaikin_money_flow_20"] == 0
    assert result["parkinson_volatility_30"] == 0
    assert result["rogers_satchell_volatility_30"] == 0
    assert np.isnan(result["realized_semivariance_balance_60"])
    assert np.isnan(result["bipower_jump_ratio_60"])
    assert np.isnan(result["variance_ratio_60_5"])
    assert result["return_sign_entropy_60"] == 0
    assert np.isnan(result["volatility_of_volatility_60"])

    zero_volume = bars.copy()
    zero_volume["volume"] = 0.0
    zero_result = build_mlat_feature_matrix(
        zero_volume, _observations(zero_volume)
    ).matrix.iloc[-1]
    assert np.isnan(zero_result["chaikin_money_flow_20"])
    assert np.isnan(zero_result["amihud_illiquidity_60"])

    missing = _bars()
    missing.loc[90, "close"] = np.nan
    missing_result = build_mlat_feature_matrix(
        missing, _observations(missing)
    ).matrix.iloc[-1]
    assert np.isnan(missing_result["bollinger_zscore_20"])
    assert np.isfinite(missing_result["parkinson_volatility_30"])


def test_strict_namespace_gc_scope_order_and_dtypes() -> None:
    bars = _bars()
    observations = _observations(bars)
    first = build_mlat_feature_matrix(bars, observations)
    second = build_mlat_feature_matrix(
        bars=bars.copy(),
        observations=observations.copy(),
        registry=build_mlat_registry(),
    )

    assert BAR_INPUT_COLUMNS == MLAT_FEATURE_SOURCE_COLUMNS
    assert FEATURE_METADATA_COLUMNS == MLAT_METADATA_COLUMNS
    assert second.diagnostics.equals(second.feature_diagnostics)
    assert first.matrix.columns.tolist() == [
        *MLAT_METADATA_COLUMNS,
        *MLAT_FEATURE_NAMES,
    ]
    assert all(
        str(first.matrix[feature].dtype) == "float32"
        for feature in MLAT_FEATURE_NAMES
    )
    pd.testing.assert_frame_equal(first.matrix, second.matrix)

    scheduled = observations.assign(
        entry_bar_id=observations["decision_bar_id"] + 1,
        forced_exit_timestamp_ny=observations["entry_timestamp_ny"]
        + pd.Timedelta(hours=1),
        minutes_to_forced_exit=60,
    )
    assert len(build_mlat_feature_matrix(bars, scheduled).matrix) == len(scheduled)

    extra = bars.assign(future_return_5=0.0)
    with pytest.raises(ValueError, match="extra_source"):
        build_mlat_feature_matrix(extra, observations)
    mgc = bars.copy()
    mgc["product"] = "MGC"
    mgc_observations = _observations(mgc)
    with pytest.raises(ValueError, match="GC-only"):
        build_mlat_feature_matrix(mgc, mgc_observations)
    labelled_observations = observations.assign(forward_return_5=0.0)
    with pytest.raises(ValueError, match="forbidden_observations"):
        build_mlat_feature_matrix(bars, labelled_observations)
    unsorted = bars.copy()
    unsorted.loc[[40, 41], "ts_event_utc"] = unsorted.loc[
        [41, 40], "ts_event_utc"
    ].to_numpy()
    with pytest.raises(ValueError, match="strictly chronological"):
        build_mlat_feature_matrix(unsorted, observations)
