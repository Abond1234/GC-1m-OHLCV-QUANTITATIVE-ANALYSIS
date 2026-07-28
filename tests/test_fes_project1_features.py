from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistical_research.fes_project1_features import (
    BASE_FEATURE_NAMES,
    BOUNDARY_CAUSES,
    COMPARATOR_NAMES,
    DEGENERATE_PERMITTED_FEATURES,
    INTERACTION_FEATURE_NAMES,
    MISSING_CAUSES,
    MISSINGNESS_ACTIONS,
    PARENT_FEATURE_NAMES,
    SECTION2_FEATURE_NAMES,
    SOURCE_COLUMNS,
    _acf_energy,
    _curvature,
    _legal_continuity,
    _mad,
    _ordered_draw_balance,
    _outlier_fraction,
    _profile_slope,
    _rolling_mean,
    _rolling_std_population,
    _spearman_rows,
    _tail_balance,
    _trimmed_mean,
    build_scalar_feature_matrix,
    scalar_feature_registry,
)


def _bars(
    n: int = 220,
    *,
    start: str = "2024-01-02 14:00:00+00:00",
    flat_price: bool = False,
) -> pd.DataFrame:
    utc = pd.date_range(start, periods=n, freq="min", tz="UTC")
    ny = utc.tz_convert("America/New_York")
    if flat_price:
        close = np.full(n, 2_000.0)
        open_ = close.copy()
        high = close.copy()
        low = close.copy()
    else:
        changes = np.resize(
            np.array([0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.4, -0.3]),
            n,
        )
        close = 2_000.0 + np.cumsum(changes)
        open_ = np.r_[2_000.0, close[:-1]]
        high = np.maximum(open_, close) + 0.2
        low = np.minimum(open_, close) - 0.2
    volume = 100.0 + (np.arange(n) % 17) * 7.0 + (np.arange(n) % 5)
    frame = pd.DataFrame(
        {
            "source_row_id": np.arange(n, dtype=np.int64),
            "ts_event_utc": utc,
            "trade_date_ny": ny.tz_localize(None).normalize(),
            "product": "GC",
            "symbol": "GCZ4",
            "active_symbol": "GCZ4",
            "instrument_id": np.uint32(101),
            "continuous_segment_id": np.int32(1),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "tradable_research_flag": True,
            "roll_window_flag": False,
            "low_liquidity_warning_flag": False,
        }
    )
    return frame.loc[:, SOURCE_COLUMNS]


def _observations(
    bars: pd.DataFrame,
    positions: list[int],
    *,
    partition: str = "Development",
) -> pd.DataFrame:
    records = []
    for observation_id, position in enumerate(positions):
        timestamp = pd.Timestamp(bars.loc[position, "ts_event_utc"])
        timestamp_ny = timestamp.tz_convert("America/New_York")
        records.append(
            {
                "observation_id": 10_000 + observation_id,
                "decision_bar_id": int(bars.loc[position, "source_row_id"]),
                "decision_timestamp_utc": timestamp,
                "trade_date_ny": pd.Timestamp(bars.loc[position, "trade_date_ny"]),
                "entry_session": (
                    "London" if timestamp_ny.hour < 7 else "New York"
                ),
                "research_partition": partition,
            }
        )
    return pd.DataFrame.from_records(records)


def _existing(observations: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"observation_id": observations["observation_id"].iloc[::-1]})
    row_count = len(frame)
    for name in dict.fromkeys((*PARENT_FEATURE_NAMES, *COMPARATOR_NAMES)):
        frame[name] = np.linspace(0.4, 1.2, row_count)
    frame["efficiency_ratio_30"] = np.linspace(0.2, 0.8, row_count)
    frame["distance_from_research_day_vwap_atr"] = np.linspace(-1.0, 1.0, row_count)
    frame["normalized_ols_slope_30"] = np.linspace(0.5, -0.5, row_count)
    return frame


def _build(
    bars: pd.DataFrame,
    positions: list[int],
    *,
    partition: str = "Development",
):
    observations = _observations(bars, positions, partition=partition)
    return build_scalar_feature_matrix(
        bars,
        observations,
        _existing(observations),
        allowed_partitions=(partition,),
        chunk_size=2,
    )


def test_trimmed_mean_removes_exactly_three_from_each_tail() -> None:
    values = np.arange(30.0)[None, :]
    assert _trimmed_mean(values)[0] == pytest.approx(np.mean(np.arange(3.0, 27.0)))


def test_mad_scaling_and_all_equal_behavior() -> None:
    values = np.array([[0.0, 1.0, 2.0, 3.0], [7.0, 7.0, 7.0, 7.0]])
    actual = _mad(values)
    assert actual[0] == pytest.approx(1.4826)
    assert actual[1] == 0.0


def test_tail_balance_sign_bounds_and_zero_tail() -> None:
    positive = np.r_[np.full(30, -1.0), np.full(30, 5.0)]
    negative = -positive
    zero = np.zeros(60)
    actual, zero_flag = _tail_balance(np.vstack([positive, negative, zero]))
    assert 0.0 < actual[0] <= 1.0
    assert -1.0 <= actual[1] < 0.0
    assert actual[2] == 0.0
    assert zero_flag.tolist() == [False, False, True]


def test_outlier_fraction_uses_strict_three_scaled_mad_threshold() -> None:
    values = np.arange(60.0)
    expected_scale = 1.4826 * np.median(np.abs(values - np.median(values)))
    expected = np.mean(np.abs(values - np.median(values)) > 3.0 * expected_scale)
    actual, zero = _outlier_fraction(values[None, :])
    assert actual[0] == pytest.approx(expected)
    assert not zero[0]
    flat, flat_zero = _outlier_fraction(np.ones((1, 60)))
    assert flat[0] == 0.0
    assert flat_zero[0]


@pytest.mark.parametrize("beta2", [2.5, -1.75, 0.0])
def test_known_quadratic_curvature(beta2: float) -> None:
    x = np.linspace(-1.0, 1.0, 31)
    path = 2_000.0 + 0.4 * x + beta2 * x**2
    actual = _curvature(path[None, :], np.array([1.0]))
    assert actual[0] == pytest.approx(beta2, abs=1.0e-10)


def test_ordered_runup_drawdown_balance() -> None:
    windows = np.vstack(
        [np.arange(31.0), np.arange(31.0)[::-1], np.ones(31)]
    )
    actual, zero = _ordered_draw_balance(windows)
    assert actual.tolist() == pytest.approx([1.0, -1.0, 0.0])
    assert zero.tolist() == [False, False, True]


def test_known_acf_energy_and_degeneracy() -> None:
    alternating = np.resize(np.array([-1.0, 1.0]), 60)
    actual, degenerate = _acf_energy(
        np.vstack([alternating, np.ones(60)])
    )
    assert actual[0] == pytest.approx(1.0)
    assert not degenerate[0]
    assert np.isnan(actual[1])
    assert degenerate[1]


def test_known_spearman_correlations_with_ties() -> None:
    tied = np.repeat(np.arange(10.0), 3)[None, :]
    actual, degenerate = _spearman_rows(
        np.vstack([tied, tied]), np.vstack([tied, tied[:, ::-1]])
    )
    assert actual.tolist() == pytest.approx([1.0, -1.0])
    assert not degenerate.any()
    _, insufficient = _spearman_rows(np.ones((1, 30)), np.arange(30.0)[None, :])
    assert insufficient[0]


def test_known_volume_profile_slope() -> None:
    x = np.linspace(-1.0, 1.0, 30)
    values = 4.0 + 3.25 * x
    assert _profile_slope(values[None, :])[0] == pytest.approx(3.25)


def test_registry_is_exact_and_complete() -> None:
    registry = scalar_feature_registry()
    assert registry["feature_name"].tolist() == list(SECTION2_FEATURE_NAMES)
    assert len(registry) == 15
    assert {
        "formula",
        "role",
        "inputs",
        "history",
        "reset_boundary",
        "expected_range",
        "book_provenance",
    }.issubset(registry.columns)


def test_interaction_formulas_clipping_and_exact_observation_join() -> None:
    bars = _bars()
    observations = _observations(bars, [180, 181])
    existing = _existing(observations)
    # Deliberately reverse parent rows and force distinct parent values.
    existing.loc[existing["observation_id"].eq(10_000), "efficiency_ratio_30"] = 0.25
    existing.loc[existing["observation_id"].eq(10_001), "efficiency_ratio_30"] = 0.75
    result = build_scalar_feature_matrix(
        bars,
        observations,
        existing,
        allowed_partitions=("Development",),
        chunk_size=1,
    )
    matrix = result.matrix
    assert np.all(
        (matrix["relative_volume_20_clipped"] >= 0.5)
        & (matrix["relative_volume_20_clipped"] <= 2.0)
    )
    expected_i01 = (
        matrix["price_path_curvature_30_atr"].to_numpy()
        * np.array([0.25, 0.75])
    )
    assert matrix["curvature_coherence_30"].to_numpy() == pytest.approx(
        expected_i01
    )
    assert matrix["tail_pressure_activity_60"].to_numpy() == pytest.approx(
        matrix["ret_tail_balance_60"].to_numpy()
        * matrix["relative_volume_20_clipped"].to_numpy()
    )
    assert matrix["lagged_volume_confirmation_30"].to_numpy() == pytest.approx(
        matrix["lagged_volume_return_spearman_30"].to_numpy()
        * matrix["relative_volume_20_clipped"].to_numpy()
    )
    expected_i04 = np.array([-1.0, 1.0]) * np.array([0.5, -0.5])
    assert matrix["vwap_trend_alignment_30"].to_numpy() == pytest.approx(
        expected_i04
    )


def test_future_mutation_and_source_truncation_are_causal() -> None:
    bars = _bars(240)
    baseline = _build(bars, [180]).matrix
    mutated = bars.copy()
    future = np.arange(len(mutated)) > 180
    mutated.loc[future, ["open", "high", "low", "close"]] += 500.0
    mutated.loc[future, "volume"] *= 100.0
    recomputed = _build(mutated, [180]).matrix
    truncated = _build(bars.iloc[:181].copy(), [180]).matrix
    for name in SECTION2_FEATURE_NAMES:
        assert baseline[name].equals(recomputed[name]), name
        assert baseline[name].equals(truncated[name]), name


@pytest.mark.parametrize(
    ("mutation", "expected_cause"),
    [
        ("timestamp", "TIMESTAMP_GAP"),
        ("date", "NY_DATE_RESET"),
        ("contract", "SELECTED_CONTRACT_CHANGE"),
        ("instrument", "INSTRUMENT_CHANGE"),
        ("segment", "CONTINUOUS_SEGMENT_CHANGE"),
        ("roll", "TRADABILITY_ROLL_OR_LIQUIDITY_BOUNDARY"),
    ],
)
def test_boundaries_reset_full_windows(mutation: str, expected_cause: str) -> None:
    bars = _bars()
    boundary = 165
    decision = 180
    if mutation == "timestamp":
        bars.loc[boundary, "ts_event_utc"] += pd.Timedelta(minutes=1)
    elif mutation == "date":
        bars.loc[boundary:, "trade_date_ny"] += pd.Timedelta(days=1)
    elif mutation == "contract":
        bars.loc[boundary:, ["symbol", "active_symbol"]] = "GCG5"
    elif mutation == "instrument":
        bars.loc[boundary:, "instrument_id"] = 202
    elif mutation == "segment":
        bars.loc[boundary:, "continuous_segment_id"] = 2
    elif mutation == "roll":
        bars.loc[boundary, "roll_window_flag"] = True
    result = _build(bars, [decision])
    cause = result.missing_reasons.loc[
        0, "ret_trimmed_mean_30_bps__missing_reason"
    ]
    assert str(cause) == expected_cause
    assert np.isnan(result.matrix.loc[0, "ret_trimmed_mean_30_bps"])


def test_boundary_precedence_is_production_reason_before_date_reset() -> None:
    bars = _bars()
    boundary = 170
    bars.loc[boundary, "ts_event_utc"] += pd.Timedelta(minutes=1)
    bars.loc[boundary:, "trade_date_ny"] += pd.Timedelta(days=1)
    _, _, latest_cause = _legal_continuity(bars)
    assert latest_cause[boundary] == "TIMESTAMP_GAP"


def test_product_change_has_its_locked_boundary_cause() -> None:
    bars = _bars()
    bars.loc[170:, "product"] = "MGC"
    _, _, latest_cause = _legal_continuity(bars)
    assert latest_cause[170] == "PRODUCT_CHANGE"


def test_source_null_nonpositive_atr_and_degeneracy_causes() -> None:
    null_bars = _bars()
    null_bars.loc[175, "close"] = np.nan
    null_result = _build(null_bars, [180])
    assert (
        str(
            null_result.missing_reasons.loc[
                0, "ret_trimmed_mean_30_bps__missing_reason"
            ]
        )
        == "SOURCE_INPUT_NULL"
    )

    flat_bars = _bars(flat_price=True)
    flat_result = _build(flat_bars, [180])
    assert (
        str(
            flat_result.missing_reasons.loc[
                0, "price_path_curvature_30_atr__missing_reason"
            ]
        )
        == "NONPOSITIVE_ATR"
    )
    assert (
        str(
            flat_result.missing_reasons.loc[
                0, "return_acf_energy_60__missing_reason"
            ]
        )
        == "DEGENERATE_STATISTIC"
    )
    assert bool(
        flat_result.missing_reasons.loc[
            0, "return_acf_energy_60__degenerate"
        ]
    )


def test_lagged_volume_profile_order_is_exact() -> None:
    bars = _bars()
    result = _build(bars, [180])
    groups, _, _ = _legal_continuity(bars)
    volume = bars["volume"].to_numpy(dtype=np.float64)
    log_volume = np.log1p(volume)
    mean60 = _rolling_mean(log_volume, 60, groups)
    std60 = _rolling_std_population(log_volume, 60, groups)
    zv = (log_volume - mean60) / std60
    close = bars["close"].to_numpy(dtype=np.float64)
    returns = np.full(len(close), np.nan)
    returns[1:] = 10_000.0 * np.log(close[1:] / close[:-1])
    expected, _ = _spearman_rows(zv[150:180][None, :], returns[151:181][None, :])
    concurrent, _ = _spearman_rows(
        zv[151:181][None, :], returns[151:181][None, :]
    )
    actual = result.matrix.loc[0, "lagged_volume_return_spearman_30"]
    assert actual == pytest.approx(expected[0], abs=1.0e-6)
    assert actual != pytest.approx(concurrent[0], abs=1.0e-4)


def test_no_eligible_feature_window_crosses_1530_or_ny_date() -> None:
    # The frozen eligible entry window ends at noon. Even the longest 90-bar
    # scalar history therefore starts after 10:29 on the same NY date and cannot
    # cross the prior 15:30 mandatory-flat boundary.
    bars = _bars(181, start="2024-01-02 14:00:00+00:00")  # 09:00 New York
    decision = 179  # 11:59 New York
    timestamp_ny = pd.Timestamp(bars.loc[decision, "ts_event_utc"]).tz_convert(
        "America/New_York"
    )
    start_ny = pd.Timestamp(
        bars.loc[decision - 89, "ts_event_utc"]
    ).tz_convert("America/New_York")
    assert timestamp_ny.strftime("%H:%M") == "11:59"
    assert start_ny.strftime("%H:%M") == "10:30"
    assert timestamp_ny.date() == start_ny.date()
    result = _build(bars, [decision])
    assert result.matrix.loc[0, list(BASE_FEATURE_NAMES)].notna().all()


def test_missingness_cause_and_action_contract_is_exact() -> None:
    assert MISSING_CAUSES == (
        "SOURCE_START",
        "TIMESTAMP_GAP",
        "PRODUCT_CHANGE",
        "SELECTED_CONTRACT_CHANGE",
        "INSTRUMENT_CHANGE",
        "CONTINUOUS_SEGMENT_CHANGE",
        "TRADABILITY_ROLL_OR_LIQUIDITY_BOUNDARY",
        "NY_DATE_RESET",
        "SOURCE_INPUT_NULL",
        "NONPOSITIVE_ATR",
        "DEGENERATE_STATISTIC",
        "COMPLETE",
    )
    assert BOUNDARY_CAUSES == MISSING_CAUSES[:8]
    assert DEGENERATE_PERMITTED_FEATURES == BASE_FEATURE_NAMES[6:9]
    assert [record["cause_group"] for record in MISSINGNESS_ACTIONS] == [
        "COMPLETE",
        "SOURCE_START_OR_BOUNDARY_WARMUP",
        "SOURCE_INPUT_NULL_OR_NONPOSITIVE_ATR",
        "DEGENERATE_STATISTIC_F07_TO_F09_ONLY",
        "FORMULA_DEFINED_ZERO",
    ]
    assert len(INTERACTION_FEATURE_NAMES) == 4
