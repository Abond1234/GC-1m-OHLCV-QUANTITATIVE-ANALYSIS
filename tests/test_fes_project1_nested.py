from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - unittest discovery without pytest
    import unittest

    raise unittest.SkipTest("pytest is not installed; run this module with pytest") from None
from sklearn.linear_model import ElasticNet

from src.statistical_research.fes_project1_config import (
    D1_FEATURES,
    O1_FEATURES,
    canonical_json,
)
from src.statistical_research.fes_project1_nested import (
    ALL_SCALAR_INPUTS,
    DEGENERATE_IMPUTABLE,
    SPLINE_FEATURES,
    ScalarFoldTransformer,
    Section5PCAProfileTransformer,
    Section5PLSProfileTransformer,
    _fit_elastic_support_path,
    _mean_daily_spearman_matrix,
    assert_hierarchical_closure,
    build_exact_date_folds,
    hierarchy_closed_support,
    load_development_model_inputs,
    nested_expansion_labels,
    purged_fold_masks,
    select_one_standard_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scalar_frame(rows: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            name: rng.normal(size=rows)
            for name in ALL_SCALAR_INPUTS
            if not name.endswith("__degenerate")
        }
    )
    for feature in DEGENERATE_IMPUTABLE:
        frame[f"{feature}__degenerate"] = False
    return frame.loc[:, list(ALL_SCALAR_INPUTS)]


def test_exact_outer_fold_boundaries_and_hash_are_canonical() -> None:
    dates = pd.date_range("2020-01-01", periods=638, freq="D")
    folds = build_exact_date_folds(
        dates,
        initial_train_end=251,
        assessment_dates=63,
        step_dates=64,
    )
    assert len(folds) == 6
    first = folds[0]
    assert first.train_dates[0] == pd.Timestamp("2020-01-01")
    assert first.train_dates[-1] == dates[251]
    assert first.embargo_date == dates[252]
    assert first.assessment_dates[0] == dates[253]
    assert first.assessment_dates[-1] == dates[315]
    assert sum(len(fold.assessment_dates) for fold in folds) == 378
    payload = {
        "train_dates": [value.strftime("%Y-%m-%d") for value in first.train_dates],
        "embargo_date": first.embargo_date.strftime("%Y-%m-%d"),
        "assessment_dates": [value.strftime("%Y-%m-%d") for value in first.assessment_dates],
    }
    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    assert first.fold_sha256 == expected


def test_exact_inner_folds_require_complete_21_date_blocks() -> None:
    dates = pd.date_range("2021-01-01", periods=252, freq="D")
    folds = build_exact_date_folds(
        dates,
        initial_train_end=125,
        assessment_dates=21,
        step_dates=22,
    )
    assert len(folds) == 5
    assert folds[0].embargo_date == dates[126]
    assert folds[0].assessment_dates == tuple(dates[127:148])
    assert folds[-1].assessment_dates[-1] == dates[235]


def test_target_exit_purge_is_greater_than_or_equal_to_cutoff() -> None:
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    fold = build_exact_date_folds(
        dates,
        initial_train_end=2,
        assessment_dates=3,
        step_dates=4,
    )[0]
    frame = pd.DataFrame(
        {
            "trade_date_ny": np.repeat(dates, 2),
            "decision_timestamp_utc": [
                value + pd.Timedelta(hours=12) for value in np.repeat(dates, 2)
            ],
            "exit_timestamp_utc_60": [
                value + pd.Timedelta(hours=13) for value in np.repeat(dates, 2)
            ],
        }
    )
    assessment_first = dates[4] + pd.Timedelta(hours=12)
    frame.loc[0, "exit_timestamp_utc_60"] = assessment_first
    train, assessment, purged, cutoff = purged_fold_masks(
        frame,
        fold,
        exit_timestamp_column="exit_timestamp_utc_60",
    )
    assert cutoff == assessment_first.tz_localize("UTC")
    assert purged == 1
    assert not train[0]
    assert assessment.sum() == 6


def test_nested_expansion_threshold_uses_training_only_linear_quantile() -> None:
    values = np.arange(10, dtype=np.float64)
    train = np.arange(10) < 5
    assessment = np.arange(10) >= 5
    labels, threshold = nested_expansion_labels(
        values,
        train,
        assessment,
        quantile=0.80,
    )
    assert threshold == pytest.approx(np.quantile(values[:5], 0.8, method="linear"))
    mutated = values.copy()
    mutated[assessment] += 10_000.0
    _, mutated_threshold = nested_expansion_labels(
        mutated,
        train,
        assessment,
        quantile=0.80,
    )
    assert mutated_threshold == threshold
    assert np.isnan(labels[train]).all()


def test_s0_only_imputes_flagged_degenerate_values() -> None:
    frame = _scalar_frame()
    feature = "return_acf_energy_60"
    indicator = f"{feature}__degenerate"
    frame.loc[0, feature] = np.nan
    frame.loc[0, indicator] = True
    transformer = ScalarFoldTransformer("S0").fit(frame)
    transformed = transformer.transform(frame)
    assert np.isfinite(transformed).all()
    assert transformer.imputation_medians_[feature] == pytest.approx(
        frame.loc[1:, feature].median()
    )
    unflagged = frame.copy()
    unflagged.loc[1, feature] = np.nan
    with pytest.raises(ValueError, match="without its degenerate indicator"):
        ScalarFoldTransformer("S0").fit(unflagged)


def test_boolean_indicators_bypass_scaling_and_constants_zero() -> None:
    frame = _scalar_frame()
    indicator = "return_acf_energy_60__degenerate"
    frame.loc[::2, indicator] = True
    frame["relative_volume_20_clipped"] = 1.25
    transformed = ScalarFoldTransformer("S1").fit_transform(frame)
    columns = list(ALL_SCALAR_INPUTS)
    indicator_values = transformed[:, columns.index(indicator)]
    assert np.array_equal(indicator_values, frame[indicator].to_numpy(dtype=np.float64))
    constant_values = transformed[:, columns.index("relative_volume_20_clipped")]
    assert np.array_equal(constant_values, np.zeros(len(frame)))


def test_s1_is_fold_local_and_persists_lambdas_order_and_scales() -> None:
    train = _scalar_frame(seed=10)
    assessment = _scalar_frame(rows=40, seed=11)
    transformer = ScalarFoldTransformer("S1").fit(train)
    state = transformer.get_state()
    before = dict(state.power_lambdas)
    transformer.transform(assessment)
    altered = assessment.copy()
    altered.iloc[:, 0] += 10_000.0
    transformer.transform(altered)
    after = transformer.get_state()
    assert before == after.power_lambdas
    assert after.input_columns == tuple(ALL_SCALAR_INPUTS)
    assert set(after.means) == set(ALL_SCALAR_INPUTS)
    assert set(after.scales) == set(ALL_SCALAR_INPUTS)


def test_s2_exact_spline_contract_retains_linear_effects() -> None:
    frame = _scalar_frame(seed=12)
    transformer = ScalarFoldTransformer("S2").fit(frame)
    transformed = transformer.transform(frame)
    assert transformed.shape[1] > frame.shape[1]
    assert tuple(transformer.output_columns_[: frame.shape[1]]) == tuple(frame.columns)
    assert set(transformer.spline_transformers_) == set(SPLINE_FEATURES)
    for spline in transformer.spline_transformers_.values():
        assert spline.n_knots == 4
        assert spline.degree == 3
        assert spline.knots == "quantile"
        assert spline.extrapolation == "linear"
        assert not spline.include_bias
        assert spline.order == "C"


def test_hierarchy_closure_adds_every_interaction_parent() -> None:
    closed = hierarchy_closed_support(["curvature_coherence_30", "tail_pressure_activity_60"])
    assert "price_path_curvature_30_atr" in closed
    assert "efficiency_ratio_30" in closed
    assert "ret_tail_balance_60" in closed
    assert "relative_volume_20_clipped" in closed
    assert_hierarchical_closure(closed)
    with pytest.raises(AssertionError):
        assert_hierarchical_closure(["curvature_coherence_30"])


def test_one_standard_error_rule_prefers_simplest_retained_candidate() -> None:
    rows = pd.DataFrame(
        {
            "name": ["complex_best", "simple_retained", "too_weak"],
            "mean": [0.10, 0.091, 0.07],
            "score_se": [0.01, 0.02, 0.01],
            "raw": [20, 5, 1],
            "components": [8, 0, 0],
            "configuration_json": ["a", "b", "c"],
        }
    )
    selected = select_one_standard_error(
        rows,
        metric_column="mean",
        maximize=True,
        complexity_columns=("raw", "components"),
    )
    assert selected["name"] == "simple_retained"


def test_fast_daily_spearman_matches_pandas_spearman() -> None:
    rng = np.random.default_rng(14)
    dates = np.repeat(pd.date_range("2020-01-01", periods=5), 20)
    target = rng.normal(size=len(dates))
    predictions = np.column_stack(
        [target + rng.normal(scale=0.3, size=len(dates)), rng.normal(size=len(dates))]
    )
    fast = _mean_daily_spearman_matrix(
        dates,
        predictions,
        target,
        min_observations=10,
    )
    expected = []
    frame = pd.DataFrame({"date": dates, "target": target})
    for index in range(predictions.shape[1]):
        frame["prediction"] = predictions[:, index]
        expected.append(
            frame.groupby("date")
            .apply(
                lambda group: group["prediction"].corr(group["target"], method="spearman"),
                include_groups=False,
            )
            .mean()
        )
    assert fast == pytest.approx(expected)


def test_elastic_path_matches_individual_fit_intercept_estimators() -> None:
    rng = np.random.default_rng(15)
    x = rng.normal(size=(500, 6))
    y = 0.7 * x[:, 0] - 0.3 * x[:, 2] + rng.normal(scale=0.2, size=500)
    alphas = (0.001, 0.01, 0.1)
    path = _fit_elastic_support_path(
        x,
        y,
        [f"x{index}" for index in range(x.shape[1])],
        alphas=alphas,
        l1_ratio=0.5,
    )
    for alpha in alphas:
        individual = ElasticNet(
            alpha=alpha,
            l1_ratio=0.5,
            fit_intercept=True,
            selection="cyclic",
            max_iter=5_000,
            tol=1.0e-6,
        ).fit(x, y)
        _, coefficients, failed = path[min(path, key=lambda value: abs(value - alpha))]
        assert not failed
        assert np.asarray(list(coefficients.values())) == pytest.approx(
            individual.coef_, abs=1.0e-6
        )


def test_profile_pca_and_pls_are_deterministically_oriented() -> None:
    rng = np.random.default_rng(16)
    raw = rng.normal(size=(300, 90))
    target = 0.5 * raw[:, 0] - 0.2 * raw[:, 60]
    first_pca = Section5PCAProfileTransformer("P3", 8).fit(raw)
    second_pca = Section5PCAProfileTransformer("P3", 8).fit(raw)
    assert first_pca.oriented_components_ == pytest.approx(second_pca.oriented_components_)
    pca_pivots = np.argmax(np.abs(first_pca.oriented_components_), axis=1)
    assert np.all(first_pca.oriented_components_[np.arange(8), pca_pivots] >= 0.0)
    first_pls = Section5PLSProfileTransformer(5).fit(raw, target)
    second_pls = Section5PLSProfileTransformer(5).fit(raw, target)
    assert first_pls.oriented_x_weights_ == pytest.approx(second_pls.oriented_x_weights_)


def test_development_loader_asserts_frozen_anchors_and_never_loads_validation() -> None:
    scalar_artifact = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "statistical_research"
        / "fes_project1"
        / "v1"
        / "scalar_features_development_gc.parquet"
    )
    if not scalar_artifact.exists():
        # Generated research artifacts live under the git-ignored data/ tree
        # and exist only where the pipeline has run; every other checkout
        # skips rather than fails.
        pytest.skip("fes_project1 v1 scalar-feature artifact not present in this checkout")
    inputs = load_development_model_inputs(PROJECT_ROOT)
    assert set(inputs.frame["research_partition"].astype(str)) == {"Development"}
    assert set(D1_FEATURES).issubset(inputs.frame.columns)
    assert set(O1_FEATURES).issubset(inputs.frame.columns)
    locked = inputs.access_audit.loc[
        inputs.access_audit["source"].str.contains(
            "Validation|Historical Final|economic", case=False, regex=True
        )
    ]
    assert (locked["rows_loaded"] == 0).all()
    support = inputs.support_coverage.loc[
        inputs.support_coverage["support_name"].eq("FINAL_HEAD_TO_HEAD_SUPPORT")
    ]
    assert set(support["dates"]) == {638}
