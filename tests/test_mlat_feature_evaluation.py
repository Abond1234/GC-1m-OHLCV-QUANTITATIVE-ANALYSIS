from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statistical_research.mlat_feature_evaluation as mlat_feature_evaluation
from statistical_research.mlat_feature_evaluation import (
    MLAT_HORIZONS,
    MlatCellEvaluationResult,
    MlatEvaluationConfig,
    MlatIncrementalResult,
    apply_development_quintiles,
    assign_mlat_verdicts,
    benjamini_hochberg,
    build_mlat_evaluation_frame,
    calculate_daily_spearman,
    date_block_bootstrap,
    evaluate_incremental_information,
    evaluate_mlat_feature_cells,
    fit_development_quintiles,
    horizon_thinning_mask,
)
from statistical_research.mlat_feature_validation import (
    MlatValidationResult,
    audit_feature_overlap,
)


def _inputs(rows_per_day: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2022-01-03", periods=4, freq="D")
    partitions = ["Development", "Development", "Validation", "Final test"]
    records = []
    identifier = 0
    for date, partition in zip(dates, partitions, strict=True):
        for minute in range(rows_per_day):
            timestamp = (
                date.tz_localize("America/New_York")
                + pd.Timedelta(hours=7, minutes=minute)
            )
            records.append((identifier, date.date(), timestamp, partition, minute))
            identifier += 1
    feature = pd.DataFrame(
        {
            "observation_id": [row[0] for row in records],
            "decision_timestamp_utc": [
                (row[2] - pd.Timedelta(minutes=1)).tz_convert("UTC") for row in records
            ],
            "decision_timestamp_ny": [
                row[2] - pd.Timedelta(minutes=1) for row in records
            ],
            "entry_timestamp_utc": [row[2].tz_convert("UTC") for row in records],
            "entry_timestamp_ny": [row[2] for row in records],
            "trade_date_ny": [row[1] for row in records],
            "research_partition": [row[3] for row in records],
            "product": "GC",
            "symbol": "GCG2",
            "active_symbol": "GCG2",
            "instrument_id": 1,
            "continuous_segment_id": 1,
            "candidate": np.tile(np.arange(rows_per_day), len(dates)).astype(float),
        }
    )
    labels = pd.DataFrame(
        {
            "observation_id": [row[0] for row in records],
            "decision_timestamp_utc": feature["decision_timestamp_utc"],
            "decision_timestamp_ny": feature["decision_timestamp_ny"],
            "entry_timestamp_utc": feature["entry_timestamp_utc"],
            "trade_date_ny": [row[1] for row in records],
            "entry_timestamp_ny": [row[2] for row in records],
            "entry_session": "New York",
            "research_partition": [row[3] for row in records],
            "product": "GC",
            "symbol": "GCG2",
            "active_symbol": "GCG2",
            "instrument_id": 1,
            "continuous_segment_id": 1,
            "decision_atr_20m": 10.0,
            "atr_normalization_available": True,
        }
    )
    signal = np.tile(np.arange(rows_per_day), len(dates)).astype(float)
    for horizon in MLAT_HORIZONS:
        labels[f"forward_return_{horizon}_atr"] = signal / 10
        labels[f"forward_return_{horizon}_ticks"] = signal
        labels[f"future_range_{horizon}_atr"] = signal / 5
        labels[f"future_realized_volatility_{horizon}_bps"] = signal + 1
        labels[f"mae_long_{horizon}_atr"] = signal / 20
        labels[f"mae_short_{horizon}_atr"] = signal / 25
        labels[f"label_available_{horizon}"] = True
    registry = pd.DataFrame({"feature_name": ["candidate"]})
    return feature, labels, registry


def _small_config() -> MlatEvaluationConfig:
    return MlatEvaluationConfig(
        bootstrap_iterations=50,
        minimum_observations_per_day=5,
        minimum_development_dates=1,
        minimum_validation_dates=1,
        minimum_development_observations=5,
        minimum_validation_observations=5,
    )


def _legacy_horizon_thinning_mask(
    frame: pd.DataFrame,
    horizon_minutes: int,
    *,
    date_column: str = "trade_date_ny",
    timestamp_column: str = "entry_timestamp_ny",
) -> np.ndarray:
    keep = np.zeros(len(frame), dtype=bool)
    working = frame.reset_index(drop=True)
    group_columns = [date_column]
    if "entry_session" in working:
        group_columns.append("entry_session")
    for _, group in working.groupby(group_columns, observed=True, sort=False):
        timestamps = pd.to_datetime(
            group[timestamp_column], errors="coerce", utc=True
        )
        timestamp_ns = timestamps.array.as_unit("ns").asi8
        valid = timestamps.notna().to_numpy()
        positions = group.index.to_numpy()[valid]
        ordered = positions[np.argsort(timestamp_ns[valid], kind="stable")]
        last_ns: int | None = None
        minimum_gap_ns = horizon_minutes * 60_000_000_000
        for position in ordered:
            current_ns = int(
                pd.Timestamp(working.at[position, timestamp_column]).value
            )
            if last_ns is None or current_ns - last_ns >= minimum_gap_ns:
                keep[position] = True
                last_ns = current_ns
    return keep


def _legacy_incremental_information(
    frame: pd.DataFrame,
    feature_names: list[str],
    frozen_names: list[str],
    config: MlatEvaluationConfig,
    horizons: tuple[int, ...],
) -> MlatIncrementalResult:
    control_sets = {
        "atr_20": ["atr_20"],
        "frozen_15": frozen_names,
    }
    daily_tables = []
    summary_rows = []
    for feature in feature_names:
        for family in mlat_feature_evaluation.MLAT_TARGET_FAMILIES:
            complete_column = f"{family}_complete_sample"
            family_frame = frame.loc[
                mlat_feature_evaluation._strict_boolean(
                    frame[complete_column], complete_column
                )
            ]
            for horizon in horizons:
                target = mlat_feature_evaluation._target_name(family, horizon)
                for session in config.sessions:
                    for control_name, controls in control_sets.items():
                        for partition in config.partitions:
                            group = family_frame.loc[
                                family_frame["entry_session"]
                                .astype(str)
                                .eq(session)
                                & family_frame["research_partition"]
                                .astype(str)
                                .eq(partition)
                            ]
                            daily = mlat_feature_evaluation._daily_partial_ic(
                                group,
                                feature,
                                target,
                                controls,
                                config.minimum_observations_per_day,
                            )
                            daily = daily.assign(
                                feature_name=feature,
                                target_family=family,
                                horizon_minutes=horizon,
                                entry_session=session,
                                research_partition=partition,
                                control_set=control_name,
                                control_features="|".join(controls),
                            )
                            daily_tables.append(daily)
                            eligible = daily.loc[daily["eligible_date"]]
                            summary_rows.append(
                                {
                                    "feature_name": feature,
                                    "target_family": family,
                                    "horizon_minutes": horizon,
                                    "entry_session": session,
                                    "research_partition": partition,
                                    "control_set": control_name,
                                    "eligible_dates": int(len(eligible)),
                                    "finite_observations": int(
                                        eligible["observations"].sum()
                                    ),
                                    "mean_daily_partial_ic": (
                                        float(
                                            eligible[
                                                "daily_partial_ic"
                                            ].mean()
                                        )
                                        if len(eligible)
                                        else np.nan
                                    ),
                                }
                            )

    summary = pd.DataFrame(summary_rows)
    key = [
        "feature_name",
        "target_family",
        "horizon_minutes",
        "entry_session",
        "control_set",
    ]
    wide = summary.pivot(index=key, columns="research_partition")
    wide.columns = [
        f"{metric}_{partition.lower()}"
        for metric, partition in wide.columns
    ]
    wide = wide.reset_index()
    development_ic = wide["mean_daily_partial_ic_development"]
    validation_ic = wide["mean_daily_partial_ic_validation"]
    wide["development_threshold"] = np.where(
        wide["control_set"].eq("atr_20"),
        config.atr_partial_ic_threshold,
        config.frozen_partial_ic_threshold,
    )
    wide["development_sample_pass"] = (
        wide["eligible_dates_development"].ge(
            config.minimum_development_dates
        )
        & wide["finite_observations_development"].ge(
            config.minimum_development_observations
        )
    )
    wide["validation_sample_pass"] = (
        wide["eligible_dates_validation"].ge(
            config.minimum_validation_dates
        )
        & wide["finite_observations_validation"].ge(
            config.minimum_validation_observations
        )
    )
    wide["development_magnitude_pass"] = (
        development_ic.abs() >= wide["development_threshold"]
    )
    wide["validation_sign_agreement"] = (
        np.sign(validation_ic) == np.sign(development_ic)
    ) & development_ic.notna() & validation_ic.notna()
    wide["validation_magnitude_retention"] = (
        validation_ic.abs() / development_ic.abs()
    )
    wide["validation_retention_pass"] = (
        wide["validation_magnitude_retention"]
        >= config.validation_retention
    )
    wide["incremental_pass"] = (
        wide["development_sample_pass"]
        & wide["validation_sample_pass"]
        & wide["development_magnitude_pass"]
        & wide["validation_sign_agreement"]
        & wide["validation_retention_pass"]
    )
    return MlatIncrementalResult(
        daily_partial_ic=pd.concat(daily_tables, ignore_index=True),
        summary=wide,
    )


def test_build_frame_excludes_final_and_rejects_mgc() -> None:
    feature, labels, registry = _inputs()
    result = build_mlat_evaluation_frame(feature, labels, registry, config=_small_config())

    assert set(result.frame["research_partition"]) == {"Development", "Validation"}
    assert result.checks.loc[result.checks["check"].eq("final_excluded"), "passed"].item()
    labels.loc[0, "product"] = "MGC"
    with pytest.raises(ValueError, match="GC-only"):
        build_mlat_evaluation_frame(feature, labels, registry, config=_small_config())


def test_frame_alignment_flags_and_complete_samples_fail_closed() -> None:
    feature, labels, registry = _inputs()
    mismatched = labels.copy()
    mismatched.loc[0, "symbol"] = "BAD"
    result = build_mlat_evaluation_frame(
        feature, mismatched, registry, config=_small_config()
    )
    assert not result.ready
    with pytest.raises(ValueError, match="provenance"):
        evaluate_mlat_feature_cells(
            result,
            registry,
            config=_small_config(),
            run_horizon_thinning=False,
        )

    string_flags = labels.copy()
    string_flags["label_available_5"] = "False"
    with pytest.raises(TypeError, match="boolean dtype"):
        build_mlat_evaluation_frame(
            feature, string_flags, registry, config=_small_config()
        )

    missing_ticks = labels.copy()
    missing_ticks.loc[0, "forward_return_5_ticks"] = np.nan
    complete = build_mlat_evaluation_frame(
        feature, missing_ticks, registry, config=_small_config()
    )
    row = complete.frame.loc[complete.frame["observation_id"].eq(0)].iloc[0]
    assert not row["direction_complete_sample"]
    assert row["expansion_complete_sample"]

    with pytest.raises(KeyError, match="scope columns"):
        evaluate_mlat_feature_cells(
            complete.frame.drop(columns="product"),
            registry,
            config=_small_config(),
            run_horizon_thinning=False,
        )


def test_development_edges_apply_unchanged_to_validation() -> None:
    development = np.arange(10, dtype=float)
    validation = np.array([-100.0, 2.0, 100.0])
    edges = fit_development_quintiles(development)
    assigned = apply_development_quintiles(validation, edges)

    assert np.allclose(edges, [1.8, 3.6, 5.4, 7.2])
    assert assigned.tolist() == [1, 2, 5]


def test_bh_and_bootstrap_are_correct_and_deterministic() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])

    first = date_block_bootstrap([0.1, 0.2, -0.1], iterations=100, seed=7)
    second = date_block_bootstrap([0.1, 0.2, -0.1], iterations=100, seed=7)
    assert first == second


def test_daily_ic_minimum_observations_and_horizon_thinning() -> None:
    frame = pd.DataFrame(
        {
            "trade_date_ny": [pd.Timestamp("2022-01-03").date()] * 12,
            "entry_session": "New York",
            "entry_timestamp_ny": pd.date_range(
                "2022-01-03 07:00", periods=12, freq="min", tz="America/New_York"
            ),
            "feature": np.arange(12),
            "target": np.arange(12),
        }
    )
    daily = calculate_daily_spearman(
        frame, "feature", "target", minimum_observations=10
    )
    assert daily.loc[0, "daily_spearman_ic"] == pytest.approx(1)

    mask = horizon_thinning_mask(frame, 5)
    assert np.flatnonzero(mask).tolist() == [0, 5, 10]


def test_horizon_thinning_matches_legacy_timestamp_lookup_exactly() -> None:
    timestamps = pd.to_datetime(
        [
            "2022-01-03 07:07:00-05:00",
            "2022-01-03 07:00:00-05:00",
            None,
            "2022-01-03 07:05:00-05:00",
            "2022-01-03 07:05:00-05:00",
            "2022-01-03 08:00:00-05:00",
            "2022-01-04 07:02:00-05:00",
            "2022-01-04 07:00:00-05:00",
            "2022-01-04 07:15:00-05:00",
        ],
        utc=True,
    ).tz_convert("America/New_York")
    frame = pd.DataFrame(
        {
            "trade_date_ny": [
                pd.Timestamp("2022-01-03").date()
            ] * 6
            + [pd.Timestamp("2022-01-04").date()] * 3,
            "entry_session": [
                "New York",
                "New York",
                "New York",
                "New York",
                "London",
                "London",
                "New York",
                "New York",
                "New York",
            ],
            "entry_timestamp_ny": timestamps,
        },
        index=[41, 3, 99, 17, 5, 88, 12, 1, 77],
    )

    for horizon in (1, 5, 30):
        expected = _legacy_horizon_thinning_mask(frame, horizon)
        observed = horizon_thinning_mask(frame, horizon)
        assert np.array_equal(observed, expected)


def test_cell_evaluator_keeps_bh_on_development_and_emits_report_tables() -> None:
    feature, labels, registry = _inputs(rows_per_day=12)
    built = build_mlat_evaluation_frame(
        feature, labels, registry, config=_small_config()
    )
    evaluated = evaluate_mlat_feature_cells(
        built, registry, config=_small_config(), run_horizon_thinning=False
    )

    development = evaluated.cell_results["research_partition"].eq("Development")
    validation = evaluated.cell_results["research_partition"].eq("Validation")
    assert evaluated.cell_results.loc[development, "q_value"].notna().any()
    assert evaluated.cell_results.loc[validation, "q_value"].isna().all()
    assert set(evaluated.quintile_edges["fit_partition"]) == {"Development"}
    assert not evaluated.daily_ic.empty


def test_overlap_and_partial_ic_beyond_controls() -> None:
    feature, labels, registry = _inputs(rows_per_day=20)
    existing = pd.DataFrame(
        {
            "observation_id": feature["observation_id"],
            "atr_20": np.tile(np.linspace(1, 2, 20), 4),
            "frozen_a": np.tile(np.sin(np.arange(20)), 4),
            "old_duplicate": feature["candidate"],
        }
    )
    built = build_mlat_evaluation_frame(
        feature,
        labels,
        registry,
        existing,
        frozen_features=["frozen_a"],
        config=_small_config(),
    )
    overlap = audit_feature_overlap(
        built.frame,
        existing,
        candidate_features=["candidate"],
        existing_features=["old_duplicate"],
    )
    assert overlap.loc[overlap["audit_type"].eq("overlap"), "is_exact_duplicate"].item()

    incremental = evaluate_incremental_information(
        built,
        registry,
        frozen_features=["frozen_a"],
        config=_small_config(),
    )
    assert set(incremental.summary["control_set"]) == {
        "atr_20",
        "frozen_15",
    }
    assert set(incremental.summary["horizon_minutes"]) == {60, 180}
    assert incremental.daily_partial_ic["eligible_date"].any()

    strict_counts = MlatEvaluationConfig(
        bootstrap_iterations=10,
        minimum_observations_per_day=5,
        minimum_development_dates=2,
        minimum_validation_dates=2,
        minimum_development_observations=5,
        minimum_validation_observations=5,
    )
    insufficient = evaluate_incremental_information(
        built,
        registry,
        frozen_features=["frozen_a"],
        config=strict_counts,
    )
    assert not insufficient.summary["incremental_pass"].any()

    with pytest.raises(ValueError, match="frozen_features"):
        evaluate_incremental_information(
            built,
            registry,
            frozen_features=None,
            config=_small_config(),
        )


def test_incremental_batch_matches_legacy_reference_bit_for_bit() -> None:
    feature, labels, _ = _inputs(rows_per_day=24)
    minute = np.tile(np.arange(24, dtype=float), 4)
    feature["candidate_tied"] = minute % 5
    feature["candidate_missing"] = np.sin(minute / 3)
    feature.loc[
        feature["observation_id"].isin([3, 27, 51, 75]),
        "candidate_missing",
    ] = np.nan
    registry = pd.DataFrame(
        {
            "feature_name": [
                "candidate",
                "candidate_tied",
                "candidate_missing",
            ]
        }
    )
    existing = pd.DataFrame(
        {
            "observation_id": feature["observation_id"],
            "atr_20": 1 + minute / 100,
            "frozen_a": np.cos(minute / 4),
            "frozen_b": minute % 7,
        }
    )
    existing.loc[
        existing["observation_id"].isin([9, 57]), "frozen_b"
    ] = np.nan
    config = _small_config()
    built = build_mlat_evaluation_frame(
        feature,
        labels,
        registry,
        existing,
        frozen_features=["frozen_a", "frozen_b"],
        config=config,
    )

    expected = _legacy_incremental_information(
        built.frame,
        registry["feature_name"].tolist(),
        ["frozen_a", "frozen_b"],
        config,
        (60, 180),
    )
    observed = evaluate_incremental_information(
        built,
        registry,
        frozen_features=["frozen_a", "frozen_b"],
        config=config,
        horizons=(60, 180),
    )

    pd.testing.assert_frame_equal(
        observed.daily_partial_ic,
        expected.daily_partial_ic,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        observed.summary,
        expected.summary,
        check_exact=True,
    )


def _verdict_artifacts(
    rows: list[dict[str, object]],
) -> tuple[
    MlatCellEvaluationResult,
    MlatIncrementalResult,
    pd.DataFrame,
    MlatValidationResult,
]:
    cells = pd.DataFrame(rows)
    if "thinned_mean_daily_ic" not in cells:
        cells["thinned_mean_daily_ic"] = 0.10
    year_stability = cells[
        [
            "feature_name",
            "target_family",
            "horizon_minutes",
            "entry_session",
            "research_partition",
        ]
    ].copy()
    year_stability["year"] = 2022
    year_stability["eligible_dates"] = 1
    year_stability["mean_daily_ic"] = 0.10
    year_stability["full_period_sign_agreement"] = True
    evaluation = MlatCellEvaluationResult(
        cell_results=cells,
        daily_ic=pd.DataFrame(),
        quintile_edges=pd.DataFrame(),
        quintile_results=pd.DataFrame(),
        year_stability=year_stability,
        thinning_results=pd.DataFrame(),
    )
    sessions = cells["entry_session"].drop_duplicates().astype(str).tolist()
    incremental_rows = []
    for session in sessions:
        for control_set in (
            "atr_20",
            "frozen_15",
        ):
            incremental_rows.append(
                {
                    "feature_name": "candidate",
                    "target_family": "direction",
                    "horizon_minutes": 60,
                    "entry_session": session,
                    "control_set": control_set,
                    "incremental_pass": True,
                }
            )
    incremental = MlatIncrementalResult(
        daily_partial_ic=pd.DataFrame(),
        summary=pd.DataFrame(incremental_rows),
    )
    overlap = pd.DataFrame(
        {
            "candidate_feature": ["candidate"],
            "reference_scope": ["existing"],
            "reference_feature": ["old"],
            "is_exact_duplicate": [False],
            "absolute_correlation": [0.10],
            "sufficient_overlap": [True],
        }
    )
    validation = MlatValidationResult(
        checks=pd.DataFrame(
            [{"check": "schema", "severity": "ERROR", "passed": True}]
        ),
        feature_diagnostics=pd.DataFrame(
            [
                {
                    "feature_name": "candidate",
                    "missing_rate": 0.0,
                    "dtype_pass": True,
                    "nonfinite_count": 0,
                    "range_pass": True,
                    "constant": False,
                }
            ]
        ),
        coverage=pd.DataFrame(),
        overlap_audit=pd.DataFrame(),
        sample_hash=pd.DataFrame(),
    )
    return evaluation, incremental, overlap, validation


def test_verdict_mapping_priorities_and_fails_closed() -> None:
    rows = []
    for session in ("London", "New York"):
        for partition in ("Development", "Validation"):
            rows.append(
                {
                    "feature_name": "candidate",
                    "target_family": "direction",
                    "horizon_minutes": 60,
                    "entry_session": session,
                    "research_partition": partition,
                    "mean_daily_ic": 0.10,
                    "q_value": 0.01 if partition == "Development" else np.nan,
                    "eligible_dates": 2,
                    "finite_observations": 20,
                    "absolute_monotonicity": 1.0,
                    "top_bottom_tick_spread": 3.0,
                    "thinning_sign_agreement": True,
                }
            )
    evaluation, incremental, overlap, validation = _verdict_artifacts(rows)
    verdict = assign_mlat_verdicts(
        evaluation,
        incremental,
        overlap,
        validation,
        config=_small_config(),
    )
    assert verdict.loc[0, "verdict"] == "ADVANCE_DIRECTIONAL"
    required_report_fields = {
        "hypothesis_id",
        "book_chapter",
        "source_pdf_page",
        "development_result",
        "validation_result",
        "session_stability",
        "year_stability",
        "redundancy_result",
        "incremental_information_result",
        "economic_interpretation",
        "authorization_gate_open",
        "authorization_gate_reason",
        "pre_authorization_verdict",
        "final_decision",
    }
    assert required_report_fields.issubset(verdict.columns)
    assert verdict.loc[0, "final_decision"] == verdict.loc[0, "verdict"]
    assert verdict.loc[0, "session_stability"].startswith("PASS_ALL:")
    assert verdict.loc[0, "validation_result"].startswith(
        "2/2 Development-screened cells confirmed in Validation"
    )
    assert "development_screen_pass" not in verdict.columns
    assert bool(verdict.loc[0, "any_development_screen_pass"])
    assert bool(verdict.loc[0, "any_screened_validation_confirmation_pass"])
    assert bool(verdict.loc[0, "authorization_gate_open"])
    assert verdict.loc[0, "pre_authorization_verdict"] == "ADVANCE_DIRECTIONAL"

    one_session_rows = [
        row for row in rows if row["entry_session"] == "New York"
    ]
    (
        one_session_evaluation,
        one_session_incremental,
        one_session_overlap,
        one_session_validation,
    ) = _verdict_artifacts(one_session_rows)
    one_session = assign_mlat_verdicts(
        one_session_evaluation,
        one_session_incremental,
        one_session_overlap,
        one_session_validation,
        config=_small_config(),
    )
    assert one_session.loc[0, "verdict"] != "ADVANCE_DIRECTIONAL"
    assert not bool(one_session.loc[0, "any_session_stability_pass"])
    assert one_session.loc[0, "session_stability"].startswith("FAIL:")

    redundant_overlap = overlap.copy()
    redundant_overlap["is_exact_duplicate"] = True
    redundant_overlap["absolute_correlation"] = 1.0
    redundant = assign_mlat_verdicts(
        evaluation,
        incremental,
        redundant_overlap,
        validation,
        config=_small_config(),
    )
    assert redundant.loc[0, "verdict"] == "REDUNDANT_WITH_EXISTING"

    with pytest.raises(TypeError, match="MlatCellEvaluationResult"):
        assign_mlat_verdicts(
            pd.DataFrame(rows),
            incremental,
            overlap,
            validation,
            config=_small_config(),
        )

    incomplete_incremental = MlatIncrementalResult(
        daily_partial_ic=pd.DataFrame(),
        summary=incremental.summary.loc[
            incremental.summary["control_set"].ne("frozen_15")
        ],
    )
    incomplete = assign_mlat_verdicts(
        evaluation,
        incomplete_incremental,
        overlap,
        validation,
        config=_small_config(),
    )
    assert not bool(incomplete.loc[0, "any_incremental_gate_pass"])
    assert incomplete.loc[0, "verdict"] != "ADVANCE_DIRECTIONAL"

    one_unscreened = evaluation.cell_results.copy()
    one_unscreened.loc[
        one_unscreened["research_partition"].eq("Development")
        & one_unscreened["entry_session"].eq("London"),
        "q_value",
    ] = 0.20
    partially_screened = assign_mlat_verdicts(
        MlatCellEvaluationResult(
            cell_results=one_unscreened,
            daily_ic=evaluation.daily_ic,
            quintile_edges=evaluation.quintile_edges,
            quintile_results=evaluation.quintile_results,
            year_stability=evaluation.year_stability,
            thinning_results=evaluation.thinning_results,
        ),
        incremental,
        overlap,
        validation,
        config=_small_config(),
    )
    assert partially_screened.loc[0, "validation_result"].startswith(
        "1/1 Development-screened cells confirmed in Validation"
    )
    assert bool(
        partially_screened.loc[
            0, "any_screened_validation_confirmation_pass"
        ]
    )


def test_structurally_nonevaluable_thinning_fails_closed() -> None:
    rows = []
    for session in ("London", "New York"):
        for partition in ("Development", "Validation"):
            rows.append(
                {
                    "feature_name": "candidate",
                    "target_family": "direction",
                    "horizon_minutes": 60,
                    "entry_session": session,
                    "research_partition": partition,
                    "mean_daily_ic": 0.10,
                    "q_value": (
                        0.01 if partition == "Development" else np.nan
                    ),
                    "eligible_dates": 2,
                    "finite_observations": 20,
                    "absolute_monotonicity": 1.0,
                    "top_bottom_tick_spread": 3.0,
                    "thinning_sign_agreement": True,
                    "thinned_mean_daily_ic": np.nan,
                }
            )
    evaluation, incremental, overlap, validation = _verdict_artifacts(rows)

    verdict = assign_mlat_verdicts(
        evaluation,
        incremental,
        overlap,
        validation,
        config=_small_config(),
    )

    assert verdict.loc[0, "verdict"] == "RESEARCH_ONLY"
    assert verdict.loc[0, "final_decision"] == "RESEARCH_ONLY"
    assert verdict.loc[0, "pre_authorization_verdict"] == "ADVANCE_DIRECTIONAL"
    assert not bool(verdict.loc[0, "authorization_gate_open"])
    assert "0/4 required 60-minute" in verdict.loc[
        0, "authorization_gate_reason"
    ]
    assert "structurally non-evaluable" in verdict.loc[0, "verdict_reason"]

    validation_only_cells = evaluation.cell_results.copy()
    validation_only_cells.loc[
        validation_only_cells["research_partition"].eq("Validation"),
        "thinned_mean_daily_ic",
    ] = 0.10
    validation_only_evaluation = MlatCellEvaluationResult(
        cell_results=validation_only_cells,
        daily_ic=evaluation.daily_ic,
        quintile_edges=evaluation.quintile_edges,
        quintile_results=evaluation.quintile_results,
        year_stability=evaluation.year_stability,
        thinning_results=evaluation.thinning_results,
    )
    validation_only_verdict = assign_mlat_verdicts(
        validation_only_evaluation,
        incremental,
        overlap,
        validation,
        config=_small_config(),
    )
    assert validation_only_verdict.loc[0, "verdict"] == "RESEARCH_ONLY"
    assert not bool(
        validation_only_verdict.loc[0, "authorization_gate_open"]
    )
    assert "including 0 Development cells" in validation_only_verdict.loc[
        0, "authorization_gate_reason"
    ]
