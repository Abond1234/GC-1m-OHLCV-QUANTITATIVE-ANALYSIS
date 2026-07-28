"""Synthetic tests for the locked Project 1 Section 6 evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistical_research.fes_project1_evaluation import (
    NO_POLICY_REASON_CODE,
    Section6Config,
    _validation_bucket_summary,
    build_not_applicable_economics,
    build_validation_continuous_evidence,
    build_validation_expansion_evidence,
    daily_net_r_sharpe,
    load_locked_validation_inputs,
    max_sharpe_adjusted_p_value,
    stationary_bootstrap_indices,
    stationary_bootstrap_sharpe_ci,
)


def test_validation_loader_requires_hash_capability(tmp_path) -> None:
    with pytest.raises(PermissionError, match="verified Section 5 capability"):
        load_locked_validation_inputs(tmp_path, object())  # type: ignore[arg-type]


def test_frozen_bucket_edges_are_applied_without_refitting() -> None:
    frame = pd.DataFrame(
        {
            "trade_date_ny": pd.to_datetime(["2024-01-02"] * 8),
            "feature": [-10.0, -1.0, 0.2, 0.8, 1.2, 1.8, 3.0, 100.0],
            "target": [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
            "ticks": [-20.0, -10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0],
        }
    )
    buckets, monotonicity, spread = _validation_bucket_summary(
        frame,
        feature="feature",
        target="target",
        tick_target="ticks",
        edges=np.array([0.0, 1.0, 2.0, 4.0]),
    )
    assert buckets["bucket_index"].tolist() == [0, 1, 2, 3, 4]
    assert monotonicity == pytest.approx(1.0)
    assert spread == pytest.approx(45.0)


def test_daily_net_r_sharpe_includes_zero_trade_dates() -> None:
    values = np.array([1.0, 0.0, -0.5, 0.0, 0.25])
    expected = np.sqrt(252) * values.mean() / values.std(ddof=1)
    assert daily_net_r_sharpe(values) == pytest.approx(expected)


def test_stationary_bootstrap_indices_are_deterministic_and_chronological() -> None:
    first = stationary_bootstrap_indices(
        20, replicates=30, restart_probability=0.2, seed=20260726
    )
    second = stationary_bootstrap_indices(
        20, replicates=30, restart_probability=0.2, seed=20260726
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (30, 20)
    assert ((first >= 0) & (first < 20)).all()


def test_stationary_bootstrap_sharpe_interval_is_reproducible() -> None:
    config = Section6Config(stationary_bootstrap_replicates=100)
    values = np.array([0.4, -0.2, 0.0, 0.1, 0.3, -0.1] * 8)
    first = stationary_bootstrap_sharpe_ci(
        values, seed=20260726, config=config
    )
    second = stationary_bootstrap_sharpe_ci(
        values, seed=20260726, config=config
    )
    assert first == pytest.approx(second)
    assert first[0] <= first[1]


def test_max_sharpe_adjustment_preserves_policy_dependence() -> None:
    config = Section6Config(stationary_bootstrap_replicates=100)
    base = np.array([0.3, -0.1, 0.0, 0.2, -0.2, 0.1] * 10)
    matrix = np.column_stack([base, 0.8 * base + 0.01])
    value = max_sharpe_adjusted_p_value(
        matrix,
        selected_policy_index=1,
        seed=20260726,
        config=config,
    )
    assert 1.0 / 101.0 <= value <= 1.0


def test_continuous_validation_evidence_detects_planted_signal() -> None:
    rng = np.random.default_rng(20260726)
    rows = []
    for date_index, date in enumerate(pd.date_range("2024-01-02", periods=20, freq="B")):
        observed = rng.normal(size=30)
        prediction = observed + rng.normal(scale=0.1, size=30)
        anchor = rng.normal(size=30)
        for row_index in range(30):
            rows.append(
                {
                    "session": "London",
                    "target": "forward_return_60_atr",
                    "model_family": "M8",
                    "trade_date_ny": date,
                    "decision_timestamp_utc": date
                    + pd.Timedelta(minutes=row_index),
                    "observation_id": date_index * 100 + row_index,
                    "observed_atr": observed[row_index],
                    "observed_ticks": observed[row_index] * 10.0,
                    "prediction": prediction[row_index],
                    "anchor_prediction": anchor[row_index],
                    "decision_atr_20m": 1.0,
                }
            )
    metrics, daily, deciles, subperiods = (
        build_validation_continuous_evidence(
            pd.DataFrame.from_records(rows),
            config=Section6Config(bootstrap_replicates=100),
        )
    )
    assert metrics.loc[0, "daily_ic_mean"] > 0.9
    assert metrics.loc[0, "paired_daily_ic_delta_mean"] > 0.8
    assert len(daily) == 20
    assert len(deciles) == 10
    assert not subperiods.empty


def test_classifier_contract_mismatch_is_ineligible() -> None:
    dates = np.repeat(pd.date_range("2024-01-02", periods=20, freq="B"), 10)
    labels = np.tile([0, 0, 0, 0, 0, 1, 0, 1, 0, 1], 20)
    probabilities = np.tile(np.linspace(0.05, 0.95, 10), 20)
    predictions = pd.DataFrame(
        {
            "session": "London",
            "model_family": "M1_O1",
            "trade_date_ny": dates,
            "expansion_label_60": labels,
            "probability": probabilities,
            "threshold_contract_match": False,
            "authoritative_range_threshold_atr": 11.8,
            "model_training_range_threshold_atr": 10.7,
        }
    )
    development = pd.DataFrame(
        {
            "session": ["London"] * 20,
            "model_family": ["M1_O1"] * 20,
            "nested_expansion_label": [0, 0, 0, 1, 0] * 4,
        }
    )
    thresholds = pd.DataFrame(
        {
            "session": ["London"],
            "threshold_role": ["expansion_probability_p80"],
            "value": [0.8],
        }
    )
    metrics, reliability = build_validation_expansion_evidence(
        predictions,
        development,
        thresholds,
        config=Section6Config(bootstrap_replicates=100),
    )
    assert not bool(metrics.loc[0, "threshold_contract_match"])
    assert not bool(metrics.loc[0, "classifier_advances"])
    assert len(reliability) == 10


def test_no_policy_economics_are_explicitly_not_applicable() -> None:
    counts = pd.DataFrame(
        [
            {
                "policy_variant": policy,
                "partition": partition,
                "evaluation_dates": 100,
            }
            for policy in (
                "direction_only_p90",
                "direction_p90_and_expansion_p80",
            )
            for partition in (
                "Development outer OOF",
                "Retrospective Validation",
            )
        ]
    )
    policy = {
        "policy_execution_allowed": False,
        "policy_status": "FROZEN_NO_POLICY",
        "no_policy_reason": "synthetic hard gate failure",
    }
    economics, drawdown, sharpe, verdict = build_not_applicable_economics(
        counts, policy
    )
    assert len(economics) == 12
    assert economics["executed_trades"].eq(0).all()
    assert economics["economics_status"].eq(NO_POLICY_REASON_CODE).all()
    assert drawdown["maximum_drawdown_r"].isna().all()
    assert sharpe["daily_net_r_sharpe"].isna().all()
    assert not bool(verdict.loc[0, "mgc_work_permitted"])


def test_no_policy_economics_rejects_an_active_policy() -> None:
    counts = pd.DataFrame()
    policy = {
        "policy_execution_allowed": True,
        "policy_status": "FROZEN_POLICY",
        "no_policy_reason": "",
    }
    with pytest.raises(ValueError, match="only valid for a frozen no-policy"):
        build_not_applicable_economics(counts, policy)
