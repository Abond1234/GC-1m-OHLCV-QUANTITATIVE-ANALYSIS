from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - unittest discovery without pytest
    import unittest

    raise unittest.SkipTest("pytest is not installed; run this module with pytest") from None
from sklearn.linear_model import Ridge

from src.statistical_research.fes_project1_config import (
    D1_FEATURES,
    build_trial_ledger_skeleton,
)
from src.statistical_research.fes_project1_modeling import (
    Section4EvidenceConfig,
    _assert_development_only,
    _benjamini_hochberg_fixed_family,
    _bootstrap_mean_ci,
    _build_interaction_evidence,
    _comparison_base_columns,
    _comparison_valid_mask,
    _complete_trial_ledger,
    _daily_partial_rank_ic,
    _daily_spearman,
    _edge_hash,
    _feature_target_specs,
    _fit_quintile_edges,
    _merge_development_support_labels,
    _read_development_table,
    _ridge_svd_path_predictions,
    build_chronological_folds,
)


def test_frozen_univariate_family_sizes_and_ids() -> None:
    confirmatory, exploratory = _feature_target_specs()
    assert len(confirmatory) == 20
    assert len(exploratory) == 40
    assert len({row["trial_id"] for row in [*confirmatory, *exploratory]}) == 60
    assert {row["family"] for row in confirmatory} == {"CONFIRMATORY_20"}
    assert {row["family"] for row in exploratory} == {"EXPLORATORY_40"}


def test_bh_uses_declared_family_size_and_preserves_nan() -> None:
    result = _benjamini_hochberg_fixed_family(
        np.array([0.01, 0.04, np.nan, 0.03]),
        family_size=4,
    )
    np.testing.assert_allclose(result[[0, 1, 3]], [0.04, 0.053333333333, 0.053333333333])
    assert np.isnan(result[2])


def test_daily_spearman_uses_dates_and_average_ranks() -> None:
    frame = pd.DataFrame(
        {
            "trade_date_ny": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4),
            "x": [1, 2, 2, 4, 1, 2, 3, 4],
            "y": [10, 20, 20, 40, 4, 3, 2, 1],
        }
    )
    daily = _daily_spearman(frame, "x", "y", min_observations=4)
    np.testing.assert_allclose(daily["daily_ic"], [1.0, -1.0])
    assert daily["observation_count"].tolist() == [4, 4]


def test_bootstrap_is_date_deterministic() -> None:
    values = np.linspace(-0.2, 0.3, 20)
    first = _bootstrap_mean_ci(values, seed=20260726, replicates=200, confidence=0.95)
    second = _bootstrap_mean_ci(values, seed=20260726, replicates=200, confidence=0.95)
    assert first == second


def test_quintiles_use_linear_interpolation_and_stable_hash() -> None:
    edges = _fit_quintile_edges(np.arange(10, dtype=float), 5)
    np.testing.assert_allclose(edges, [1.8, 3.6, 5.4, 7.2])
    assert _edge_hash("F", "London", edges) == _edge_hash("F", "London", edges.copy())
    assert _edge_hash("F", "London", edges) != _edge_hash("F", "New York", edges)


def test_joint_partial_ic_removes_comparator_overlap() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for date in pd.date_range("2023-01-02", periods=8, freq="D"):
        comparator = rng.normal(size=30)
        independent = rng.normal(size=30)
        rows.extend(
            {
                "trade_date_ny": date,
                "candidate": comparator + independent,
                "target": comparator + independent + rng.normal(scale=0.05),
                "comparator": comparator,
            }
            for comparator, independent in zip(comparator, independent, strict=True)
        )
    daily = _daily_partial_rank_ic(
        pd.DataFrame(rows),
        "candidate",
        "target",
        ["comparator"],
        min_observations=10,
    )
    assert len(daily) == 8
    assert daily["daily_partial_ic"].mean() > 0.9


def test_partial_ic_skips_rank_deficient_comparator_design() -> None:
    frame = pd.DataFrame(
        {
            "trade_date_ny": pd.to_datetime(["2023-01-02"] * 12),
            "candidate": np.arange(12),
            "target": np.arange(12),
            "c1": np.arange(12),
            "c2": np.arange(12),
        }
    )
    daily = _daily_partial_rank_ic(frame, "candidate", "target", ["c1", "c2"], min_observations=10)
    assert daily.empty


def test_development_guard_rejects_locked_partitions() -> None:
    _assert_development_only(pd.DataFrame({"research_partition": ["Development"]}), "safe")
    with pytest.raises(PermissionError, match="locked outcome"):
        _assert_development_only(
            pd.DataFrame({"research_partition": ["Development", "Validation"]}),
            "unsafe",
        )


def test_physical_parquet_filter_loads_development_only(tmp_path) -> None:
    path = tmp_path / "partitioned.parquet"
    pd.DataFrame(
        {
            "observation_id": [1, 2, 3],
            "research_partition": ["Development", "Validation", "Final test"],
            "value": [10.0, 20.0, 30.0],
        }
    ).to_parquet(path, index=False)
    loaded = _read_development_table(path, ["observation_id", "research_partition", "value"])
    assert loaded["observation_id"].tolist() == [1]
    assert loaded["research_partition"].tolist() == ["Development"]


def test_exact_outer_fold_boundaries_embargo_and_hash() -> None:
    dates = pd.date_range("2020-01-01", periods=380, freq="D")
    folds = build_chronological_folds(
        dates,
        initial_train_end=251,
        assessment_dates=63,
        step_dates=64,
    )
    assert len(folds) == 2
    first = folds[0]
    assert first["train_dates"][0] == dates[0]
    assert first["train_dates"][-1] == dates[251]
    assert first["embargo_date"] == dates[252]
    assert first["assessment_dates"][0] == dates[253]
    assert first["assessment_dates"][-1] == dates[315]
    assert not set(first["train_dates"]).intersection(first["assessment_dates"])
    repeat = build_chronological_folds(
        dates,
        initial_train_end=251,
        assessment_dates=63,
        step_dates=64,
    )
    assert first["fold_sha256"] == repeat[0]["fold_sha256"]


def test_cached_svd_ridge_path_matches_sklearn_solver() -> None:
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(100, 6))
    x_assessment = rng.normal(size=(25, 6))
    y = rng.normal(size=100)
    alphas = (0.001, 0.1, 10.0)
    actual = _ridge_svd_path_predictions(x_train, y, x_assessment, alphas)
    expected = np.column_stack(
        [
            Ridge(alpha=alpha, fit_intercept=True, solver="svd")
            .fit(x_train, y)
            .predict(x_assessment)
            for alpha in alphas
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_hierarchical_base_columns_and_degenerate_indicator() -> None:
    assert _comparison_base_columns("curvature_coherence_30") == [
        *D1_FEATURES,
        "price_path_curvature_30_atr",
        "efficiency_ratio_30",
    ]
    i03 = _comparison_base_columns("lagged_volume_confirmation_30")
    assert "lagged_volume_return_spearman_30" in i03
    assert "relative_volume_20_clipped" in i03
    assert "lagged_volume_return_spearman_30__degenerate" in i03


def test_degenerate_f08_is_imputable_but_unflagged_nan_is_excluded() -> None:
    columns = _comparison_base_columns("lagged_volume_confirmation_30")
    frame = pd.DataFrame({name: [1.0, 1.0] for name in columns})
    frame["forward_return_60_atr"] = [0.1, 0.2]
    frame["lagged_volume_return_spearman_30"] = [np.nan, np.nan]
    frame["lagged_volume_return_spearman_30__degenerate"] = [True, False]
    valid = _comparison_valid_mask(frame, columns, "forward_return_60_atr")
    assert valid.tolist() == [True, False]


def test_development_support_labels_are_mechanical() -> None:
    confirmatory = pd.DataFrame(
        {
            "trial_id": ["a", "b"],
            "feature_name": ["f1", "f2"],
            "session": ["London", "London"],
            "target": ["forward_return_60_atr", "future_range_60_atr"],
            "bh_q_value": [0.01, 0.20],
            "trading_date_count": [500, 500],
            "observation_count": [20_000, 20_000],
            "bucket_monotonicity": [1.0, 1.0],
            "top_bottom_spread_ticks": [3.0, 0.0],
        }
    )
    partial = pd.DataFrame(
        {
            "trial_id": ["a", "b"],
            "feature_name": ["f1", "f2"],
            "session": ["London", "London"],
            "target": ["forward_return_60_atr", "future_range_60_atr"],
            "partial_overlap_gate": [True, True],
        }
    )
    result = _merge_development_support_labels(confirmatory, partial, Section4EvidenceConfig())
    assert result["development_label"].tolist() == [
        "DEV_SUPPORT",
        "DEV_NO_SUPPORT",
    ]
    assert result.iloc[1]["failed_gates"] == "gate_bh_q"


def _minimal_completed_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    skeleton = pd.DataFrame(build_trial_ledger_skeleton())
    section4 = skeleton.loc[skeleton["notebook_section"].eq(4)]
    base = section4.loc[section4["analysis_role"].eq("confirmatory")][["trial_id"]].copy()
    base["development_label"] = "DEV_NO_SUPPORT"
    base["raw_p_value"] = 0.5
    base["bh_q_value"] = 0.5
    base["daily_ic_mean"] = 0.0
    base["daily_ic_ci_low"] = -0.1
    base["daily_ic_ci_high"] = 0.1
    base["failed_gates"] = "gate_bh_q"
    exploratory = section4.loc[section4["analysis_role"].eq("exploratory")][["trial_id"]].copy()
    exploratory["raw_p_value"] = 0.5
    exploratory["bh_q_value"] = 0.5
    exploratory["daily_ic_mean"] = 0.0
    exploratory["daily_ic_ci_low"] = -0.1
    exploratory["daily_ic_ci_high"] = 0.1
    interactions = section4.loc[section4["analysis_role"].eq("confirmatory_incremental")][
        ["trial_id"]
    ].copy()
    interactions["development_label"] = "DEV_NO_INCREMENTAL_SUPPORT"
    interactions["raw_p_value"] = 0.5
    interactions["bh_q_value"] = 0.5
    interactions["mean_daily_ic_delta"] = 0.0
    interactions["daily_ic_delta_ci_low"] = -0.1
    interactions["daily_ic_delta_ci_high"] = 0.1
    interactions["failed_gates"] = "gate_bh_q"
    return base, exploratory, interactions


def test_complete_trial_ledger_updates_exactly_68_rows() -> None:
    ledger = _complete_trial_ledger(*_minimal_completed_tables())
    section4 = ledger["notebook_section"].eq(4)
    assert len(ledger) == 150
    assert section4.sum() == 68
    assert ledger.loc[section4, "status"].str.startswith("COMPLETED").all()
    assert ledger.loc[~section4, "status"].eq("PLANNED").all()


def _synthetic_interaction_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260726)
    records = []
    observation_id = 0
    for session in ("London", "New York"):
        for date in pd.date_range("2022-01-03", periods=52, freq="D"):
            for minute in range(12):
                values = {name: rng.normal() for name in D1_FEATURES}
                values["efficiency_ratio_30"] = rng.normal()
                curvature = rng.normal()
                tail = rng.normal()
                lagged = rng.normal()
                h01 = rng.uniform(0.5, 2.0)
                planted = curvature * values["efficiency_ratio_30"]
                target = 2.5 * planted + rng.normal(scale=0.20)
                records.append(
                    {
                        "observation_id": observation_id,
                        "decision_timestamp_utc": pd.Timestamp(date, tz="UTC")
                        + pd.Timedelta(hours=8, minutes=minute),
                        "exit_timestamp_utc_60": pd.Timestamp(date, tz="UTC")
                        + pd.Timedelta(hours=9, minutes=minute),
                        "trade_date_ny": date,
                        "entry_session": session,
                        "research_partition": "Development",
                        "label_available_60": True,
                        "forward_return_60_atr": target,
                        "price_path_curvature_30_atr": curvature,
                        "ret_tail_balance_60": tail,
                        "lagged_volume_return_spearman_30": lagged,
                        "relative_volume_20_clipped": h01,
                        "lagged_volume_return_spearman_30__degenerate": False,
                        **values,
                    }
                )
                observation_id += 1
    return pd.DataFrame.from_records(records)


def test_nested_interaction_screen_detects_planted_signal_and_rejects_noise() -> None:
    config = Section4EvidenceConfig(
        bootstrap_replicates=200,
        min_development_dates=10,
        min_development_observations=100,
        outer_initial_train_end=15,
        outer_assessment_dates=5,
        outer_step_dates=6,
        inner_initial_train_end=5,
        inner_assessment_dates=3,
        inner_step_dates=4,
        min_valid_inner_folds=2,
    )
    results, daily, folds, tuning = _build_interaction_evidence(
        _synthetic_interaction_frame(), config
    )
    planted = results.loc[results["interaction_id"].eq("I01")]
    noise = results.loc[results["interaction_id"].isin(["I02", "I03"])]
    assert planted["development_label"].eq("DEV_INCREMENTAL_SUPPORT").all()
    assert noise["development_label"].eq("DEV_NO_INCREMENTAL_SUPPORT").all()
    assert len(daily) > 0
    assert folds["status"].eq("EVALUATED").all()
    assert tuning["ridge_alpha"].isin(config.ridge_alpha_grid).all()
