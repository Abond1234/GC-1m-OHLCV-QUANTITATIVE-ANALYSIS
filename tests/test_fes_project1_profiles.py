from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistical_research.fes_project1_features import (
    MISSING_CAUSES,
    OBSERVATION_COLUMNS,
    SOURCE_COLUMNS,
    _lag,
    _legal_continuity,
    _rolling_mean,
    _rolling_std_population,
)
from src.statistical_research.fes_project1_profiles import (
    P0_COLUMNS,
    P1_COLUMNS,
    P2_COLUMNS,
    P3_COLUMNS,
    PROFILE_CHANNELS,
    PROFILE_INDEX_COLUMNS,
    PROFILE_REPRESENTATIONS,
    FoldLocalPCAProfileTransformer,
    FoldLocalPLSProfileTransformer,
    FoldLocalProfileTransformer,
    ProfileBuildResult,
    build_p1_zero_scale_audit,
    build_profile_autocorrelation_diagnostics,
    build_profile_coverage_audit,
    build_profile_matrix,
    build_profile_rank_condition_diagnostics,
    p1_robust_standardize,
    p2_first_difference,
    p3_trailing_median,
    profile_channel_lag_schema,
    transform_profile_representation,
)


def _bars(
    n: int = 240,
    *,
    start: str = "2024-01-02 14:00:00+00:00",
    flat_price: bool = False,
    constant_volume: bool = False,
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
    volume = (
        np.full(n, 100.0)
        if constant_volume
        else 100.0 + (np.arange(n) % 17) * 7.0 + (np.arange(n) % 5)
    )
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
    for number, position in enumerate(positions):
        timestamp = pd.Timestamp(bars.loc[position, "ts_event_utc"])
        timestamp_ny = timestamp.tz_convert("America/New_York")
        records.append(
            {
                "observation_id": 20_000 + number,
                "decision_bar_id": int(bars.loc[position, "source_row_id"]),
                "decision_timestamp_utc": timestamp,
                "trade_date_ny": pd.Timestamp(bars.loc[position, "trade_date_ny"]),
                "entry_session": (
                    "London" if timestamp_ny.hour < 7 else "New York"
                ),
                "research_partition": partition,
            }
        )
    return pd.DataFrame.from_records(records).loc[:, OBSERVATION_COLUMNS]


def _build(
    bars: pd.DataFrame,
    positions: list[int],
    *,
    partition: str = "Development",
) -> ProfileBuildResult:
    return build_profile_matrix(
        bars,
        _observations(bars, positions, partition=partition),
        allowed_partitions=(partition,),
        chunk_size=2,
    )


def _synthetic_raw(rows: int = 40, seed: int = 7) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(rows, len(P0_COLUMNS)))
    # Add smooth channel structure without creating exact collinearity.
    for channel in range(3):
        block = slice(channel * 30, (channel + 1) * 30)
        values[:, block] += np.linspace(-1.0, 1.0, 30)
    return values


def test_channel_lag_schema_and_dimensions_are_exact() -> None:
    schema = profile_channel_lag_schema()
    assert len(P0_COLUMNS) == 90
    assert len(P1_COLUMNS) == 90
    assert len(P2_COLUMNS) == 87
    assert len(P3_COLUMNS) == 84
    assert schema["column_name"].tolist() == list(P0_COLUMNS)
    assert schema["channel_code"].tolist()[:30] == ["R"] * 30
    assert schema["channel_code"].tolist()[30:60] == ["G"] * 30
    assert schema["channel_code"].tolist()[60:] == ["Q"] * 30
    assert schema.groupby("channel_code", sort=False)["bar_offset_from_t"].apply(
        list
    ).tolist() == [list(range(-29, 1))] * 3


def test_raw_profile_matches_hand_computed_channels_and_order() -> None:
    bars = _bars()
    decision = 180
    result = _build(bars, [decision])
    assert result.profile_index.loc[0, "profile_complete_30"]
    raw = result.raw_profiles.loc[0, P0_COLUMNS].to_numpy(dtype=np.float64)

    groups, positions, _ = _legal_continuity(bars)
    close = bars["close"].to_numpy(dtype=np.float64)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    volume = bars["volume"].to_numpy(dtype=np.float64)
    previous_close = _lag(close, 1, groups)
    returns = np.full(len(close), np.nan)
    valid = np.isfinite(previous_close)
    returns[valid] = 10_000.0 * np.log(close[valid] / previous_close[valid])
    true_range = np.maximum.reduce(
        [high - low, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    true_range[positions == 0] = (high - low)[positions == 0]
    atr20 = _rolling_mean(true_range, 20, groups)
    log_volume = np.log1p(volume)
    mean60 = _rolling_mean(log_volume, 60, groups)
    std60 = _rolling_std_population(log_volume, 60, groups)
    zscore = (log_volume - mean60) / std60
    expected = np.concatenate(
        [
            returns[decision - 29 : decision + 1],
            true_range[decision - 29 : decision + 1]
            / atr20[decision - 29 : decision + 1],
            zscore[decision - 29 : decision + 1],
        ]
    )
    assert raw == pytest.approx(expected, abs=1.0e-5)


def test_t_minus_30_predecessor_is_required() -> None:
    bars = _bars()
    decision = 180
    bars.loc[decision - 30, "close"] = np.nan
    result = _build(bars, [decision])
    assert not result.profile_index.loc[0, "profile_complete_30"]
    assert (
        str(result.profile_index.loc[0, "profile_missing_reason"])
        == "SOURCE_INPUT_NULL"
    )
    assert result.raw_profiles.empty


def test_p1_trimmed_mean_population_scale_and_zero_channel() -> None:
    raw = np.concatenate(
        [
            np.arange(30.0),
            np.full(30, 5.0),
            np.arange(30.0)[::-1],
        ]
    )[None, :]
    p1, zero = p1_robust_standardize(raw)
    trimmed = np.arange(3.0, 27.0)
    expected_r = (np.arange(30.0) - np.mean(trimmed)) / np.std(
        trimmed, ddof=0
    )
    assert p1[0, :30] == pytest.approx(expected_r)
    assert p1[0, 30:60] == pytest.approx(np.zeros(30))
    assert p1[0, 60:] == pytest.approx(expected_r[::-1])
    assert zero.tolist() == [[False, True, False]]


def test_p2_is_first_difference_within_each_channel_only() -> None:
    p1 = np.concatenate(
        [
            np.arange(30.0),
            2.0 * np.arange(30.0),
            -3.0 * np.arange(30.0),
        ]
    )[None, :]
    p2 = p2_first_difference(p1)
    assert p2.shape == (1, 87)
    assert p2[0, :29] == pytest.approx(np.ones(29))
    assert p2[0, 29:58] == pytest.approx(np.full(29, 2.0))
    assert p2[0, 58:] == pytest.approx(np.full(29, -3.0))


def test_p3_uses_causal_trailing_median_not_centered_smoothing() -> None:
    channel = np.array([0.0, 100.0, 2.0, 3.0, 4.0, *range(5, 30)])
    p1 = np.tile(channel, 3)[None, :]
    p3 = p3_trailing_median(p1).reshape(1, 3, 28)
    assert p3[0, 0, 0] == 2.0  # median(0, 100, 2)
    assert p3[0, 0, 1] == 3.0  # median(100, 2, 3)
    assert p3.shape == (1, 3, 28)


def test_representation_dispatch_shapes_and_finiteness() -> None:
    raw = _synthetic_raw(5)
    expected_columns = {"P0": 90, "P1": 90, "P2": 87, "P3": 84}
    for representation in PROFILE_REPRESENTATIONS:
        values, zero = transform_profile_representation(raw, representation)
        assert values.shape == (5, expected_columns[representation])
        assert zero.shape == (5, len(PROFILE_CHANNELS))
        assert np.isfinite(values).all()


def test_future_mutation_and_source_truncation_are_causal() -> None:
    bars = _bars()
    baseline = _build(bars, [180])
    mutated = bars.copy()
    future = np.arange(len(mutated)) > 180
    mutated.loc[future, ["open", "high", "low", "close"]] += 500.0
    mutated.loc[future, "volume"] *= 100.0
    recomputed = _build(mutated, [180])
    truncated = _build(bars.iloc[:181].copy(), [180])
    pd.testing.assert_frame_equal(baseline.profile_index, recomputed.profile_index)
    pd.testing.assert_frame_equal(baseline.raw_profiles, recomputed.raw_profiles)
    pd.testing.assert_frame_equal(baseline.profile_index, truncated.profile_index)
    pd.testing.assert_frame_equal(baseline.raw_profiles, truncated.raw_profiles)


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
def test_locked_boundaries_invalidate_full_profile(
    mutation: str,
    expected_cause: str,
) -> None:
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
    assert not result.profile_index.loc[0, "profile_complete_30"]
    assert (
        str(result.profile_index.loc[0, "profile_missing_reason"])
        == expected_cause
    )
    assert result.raw_profiles.empty


def test_boundary_precedence_and_product_cause_are_exact() -> None:
    bars = _bars()
    boundary = 170
    bars.loc[boundary, "ts_event_utc"] += pd.Timedelta(minutes=1)
    bars.loc[boundary:, "trade_date_ny"] += pd.Timedelta(days=1)
    _, _, latest = _legal_continuity(bars)
    assert latest[boundary] == "TIMESTAMP_GAP"

    product_bars = _bars()
    product_bars.loc[boundary:, "product"] = "MGC"
    _, _, product_latest = _legal_continuity(product_bars)
    assert product_latest[boundary] == "PRODUCT_CHANGE"


def test_nonpositive_atr_and_nonfinite_zscore_causes() -> None:
    flat = _build(_bars(flat_price=True), [180])
    assert (
        str(flat.profile_index.loc[0, "profile_missing_reason"])
        == "NONPOSITIVE_ATR"
    )
    constant_volume = _build(_bars(constant_volume=True), [180])
    assert (
        str(constant_volume.profile_index.loc[0, "profile_missing_reason"])
        == "SOURCE_INPUT_NULL"
    )


def test_no_eligible_profile_crosses_1530_or_ny_date() -> None:
    bars = _bars(181, start="2024-01-02 14:00:00+00:00")
    decision = 179  # 11:59 New York
    start = pd.Timestamp(bars.loc[decision - 88, "ts_event_utc"]).tz_convert(
        "America/New_York"
    )
    end = pd.Timestamp(bars.loc[decision, "ts_event_utc"]).tz_convert(
        "America/New_York"
    )
    assert start.strftime("%H:%M") == "10:31"
    assert end.strftime("%H:%M") == "11:59"
    assert start.date() == end.date()
    assert _build(bars, [decision]).profile_index.loc[0, "profile_complete_30"]


def test_profile_index_missingness_contract_and_raw_only_result() -> None:
    bars = _bars()
    result = _build(bars, [20, 180])
    assert tuple(result.profile_index.columns) == PROFILE_INDEX_COLUMNS
    assert list(result.profile_index["profile_missing_reason"].cat.categories) == list(
        MISSING_CAUSES
    )
    assert not result.profile_index.loc[0, "profile_complete_30"]
    assert str(result.profile_index.loc[0, "profile_missing_reason"]) == "SOURCE_START"
    assert result.profile_index.loc[1, "profile_complete_30"]
    assert len(result.raw_profiles) == 1
    assert set(result.__dataclass_fields__) == {
        "profile_index",
        "raw_profiles",
        "channel_lag_schema",
        "construction_audit",
    }


def test_outcome_columns_are_rejected() -> None:
    bars = _bars()
    observations = _observations(bars, [180])
    observations["future_target"] = 1.0
    with pytest.raises(ValueError, match="Outcome columns"):
        build_profile_matrix(
            bars,
            observations,
            allowed_partitions=("Development",),
        )


def test_fold_local_standardization_fits_training_rows_only() -> None:
    train = _synthetic_raw(50, seed=1)
    assessment = _synthetic_raw(10, seed=2) + 100.0
    transformer = FoldLocalProfileTransformer("P1").fit(train)
    train_p1, _ = transform_profile_representation(train, "P1")
    assert transformer.scaler_.mean_ == pytest.approx(train_p1.mean(axis=0))
    before_mean = transformer.scaler_.mean_.copy()
    transformed = transformer.transform(assessment)
    assert np.array_equal(before_mean, transformer.scaler_.mean_)
    assert transformed.shape == (10, 90)
    assert transformer.fit_row_count_ == 50


def test_assessment_mutation_cannot_change_training_fitted_pca() -> None:
    train = _synthetic_raw(80, seed=3)
    assessment = _synthetic_raw(12, seed=4)
    transformer = FoldLocalPCAProfileTransformer("P2", n_components=5).fit(train)
    components = transformer.oriented_components_.copy()
    scaler_mean = transformer.profile_transformer_.scaler_.mean_.copy()
    baseline = transformer.transform(assessment)
    altered_assessment = assessment.copy()
    altered_assessment[:, 0] += 1_000.0
    mutated = transformer.transform(altered_assessment)
    assert np.array_equal(components, transformer.oriented_components_)
    assert np.array_equal(scaler_mean, transformer.profile_transformer_.scaler_.mean_)
    assert not np.allclose(baseline, mutated)


def test_pca_components_are_deterministic_with_fixed_orientation() -> None:
    train = _synthetic_raw(100, seed=5)
    first = FoldLocalPCAProfileTransformer("P3", n_components=8).fit(train)
    second = FoldLocalPCAProfileTransformer("P3", n_components=8).fit(train)
    assert first.oriented_components_ == pytest.approx(second.oriented_components_)
    assert first.transform(train) == pytest.approx(second.transform(train))


def test_pls_interface_is_p2_only_target_required_and_deterministic() -> None:
    train = _synthetic_raw(100, seed=6)
    target = 0.5 * train[:, 0] - 0.2 * train[:, 60]
    with pytest.raises(ValueError, match="requires training targets"):
        FoldLocalPLSProfileTransformer(2).fit(train)
    first = FoldLocalPLSProfileTransformer(2).fit(train, target)
    second = FoldLocalPLSProfileTransformer(2).fit(train, target)
    assert first.profile_transformer_.representation == "P2"
    assert first.oriented_x_rotations_ == pytest.approx(
        second.oriented_x_rotations_
    )
    assert first.transform(train).shape == (100, 2)
    assert np.max(np.atleast_1d(first.pls_.n_iter_)) <= 500


def test_diagnostics_are_outcome_free_and_dimensionally_complete() -> None:
    raw_values = _synthetic_raw(150, seed=8).astype(np.float32)
    raw = pd.DataFrame(raw_values, columns=P0_COLUMNS)
    raw.insert(0, "observation_id", np.arange(len(raw)))
    index = pd.DataFrame(
        {
            "observation_id": np.arange(len(raw)),
            "decision_timestamp_utc": pd.date_range(
                "2023-01-02", periods=len(raw), freq="min", tz="UTC"
            ),
            "trade_date_ny": pd.Timestamp("2023-01-02"),
            "entry_session": np.where(np.arange(len(raw)) % 2, "London", "New York"),
            "research_partition": "Development",
            "profile_complete_30": True,
            "profile_missing_reason": pd.Categorical(
                ["COMPLETE"] * len(raw),
                categories=list(MISSING_CAUSES),
                ordered=True,
            ),
        }
    )
    coverage = build_profile_coverage_audit(index)
    zero = build_p1_zero_scale_audit(raw, chunk_size=37)
    autocorrelation = build_profile_autocorrelation_diagnostics(
        raw, maximum_lag=10, chunk_size=40
    )
    rank = build_profile_rank_condition_diagnostics(raw, chunk_size=50)
    assert len(coverage) == 2 * len(MISSING_CAUSES)
    assert len(zero) == 3
    assert len(autocorrelation) == 4 * 3 * 10
    assert rank["representation"].tolist() == list(PROFILE_REPRESENTATIONS)
    assert rank["columns"].tolist() == [90, 90, 87, 84]
    joined_columns = "|".join(
        [
            *coverage.columns,
            *zero.columns,
            *autocorrelation.columns,
            *rank.columns,
        ]
    ).lower()
    assert not any(
        token in joined_columns
        for token in ("target", "forward", "future", "label", "performance")
    )
