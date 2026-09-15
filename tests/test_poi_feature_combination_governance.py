"""Regression tests for feature-combination data and publication governance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.statistical_research.feature_combination import (
    ALL_FES_SOURCE_FEATURES,
    INTERACTION_FEATURES,
    POI_CONTEXT_FEATURES,
    STAT15_FEATURES,
    add_poi_interactions,
    model_features,
    summarize_policy,
)
from src.statistical_research.feature_combination_runner import (
    ELIGIBLE_COLUMNS,
    FEATURE_COLUMNS,
    FES_COLUMNS,
    LABEL_COLUMNS,
    POI_COLUMNS,
    BoundaryAccessError,
    _build_validation_capability,
    _eligible_dates_for_frames,
    _load_labels,
    _load_poi_candidates,
    _load_predictor_base,
    _persist_frames,
    _publish_directories,
    _read_projected,
    _score_validation,
    _selected_policy,
    _write_blocked_artifacts,
)


class PolicySummaryRegressionTests(unittest.TestCase):
    def test_initial_loss_is_drawdown_from_zero_and_gross_median_is_reported(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        trades = pd.DataFrame(
            {
                "trade_date_ny": dates,
                "gross_ticks": [-5.0, 3.0],
            }
        )

        summary = summarize_policy(
            trades,
            eligible_dates=dates,
            round_trip_cost_ticks=0.0,
            seed=31,
        )

        self.assertEqual(summary["maximum_drawdown_ticks"], -5.0)
        self.assertEqual(summary["median_gross_ticks"], -1.0)
        self.assertEqual(summary["eligible_date_count"], 2)

    def test_trade_dates_must_belong_to_the_frozen_eligible_universe(self) -> None:
        trades = pd.DataFrame(
            {
                "trade_date_ny": pd.to_datetime(["2024-01-03"]),
                "gross_ticks": [1.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "eligible-date universe"):
            summarize_policy(
                trades,
                eligible_dates=pd.to_datetime(["2024-01-02"]),
                round_trip_cost_ticks=0.0,
            )


class InteractionDtypeRegressionTests(unittest.TestCase):
    def test_all_predeclared_products_are_formed_in_float64(self) -> None:
        source_columns = {
            "poi_direction_sign",
            "lagged_volume_return_spearman_30",
            "feat_poi_width_atr",
            "atr_ratio_20_60",
            "feat_displacement_efficiency",
            "efficiency_ratio_30",
            "feat_approach_15m_candle_overlap_ratio",
            "return_sign_change_rate_30",
            "feat_touch_penetration_fraction",
            "current_range_over_atr",
            "feat_approach_15m_range_compression_ratio",
            "return_acf_energy_60",
            "feat_approach_15m_relative_volume",
            "range_volume_spearman_30",
            "feat_distance_from_vwap_atr",
            "normalized_ols_slope_30",
        }
        frame = pd.DataFrame(
            {
                column: pd.Series([1, 2], dtype="int32")
                if index % 2 == 0
                else pd.Series([1.5, 2.5], dtype="float32")
                for index, column in enumerate(sorted(source_columns))
            }
        )

        result = add_poi_interactions(frame)

        self.assertTrue(
            all(
                result[column].dtype == np.dtype("float64")
                for column in INTERACTION_FEATURES
            )
        )


class FrozenProjectionRegressionTests(unittest.TestCase):
    def test_outcome_bearing_projections_match_the_frozen_contract_exactly(self) -> None:
        expected_eligible = (
            "observation_id",
            "decision_bar_id",
            "decision_timestamp_utc",
            "entry_timestamp_utc",
            "trade_date_ny",
            "entry_session",
            "research_partition",
        )
        self.assertEqual(ELIGIBLE_COLUMNS, expected_eligible)
        self.assertEqual(
            LABEL_COLUMNS,
            expected_eligible
            + (
                "exit_timestamp_utc_60",
                "forward_return_60_ticks",
                "forward_return_60_atr",
                "future_range_60_atr",
                "label_available_60",
            ),
        )
        self.assertEqual(
            FEATURE_COLUMNS,
            (
                "observation_id",
                "trade_date_ny",
                "entry_session",
                "research_partition",
            )
            + STAT15_FEATURES
            + ("normalized_ols_slope_30",),
        )
        self.assertEqual(
            FES_COLUMNS,
            (
                "observation_id",
                "decision_timestamp_utc",
                "trade_date_ny",
                "entry_session",
                "research_partition",
            )
            + ALL_FES_SOURCE_FEATURES,
        )
        self.assertEqual(
            POI_COLUMNS,
            (
                "true_retest_id",
                "retest_bar_id",
                "retest_ts_event_utc",
                "trade_date_ny",
                "direction",
                "research_partition",
                "feat_execution_session",
                "feat_first_touch",
                "feat_15bar_structural_validation",
            )
            + POI_CONTEXT_FEATURES,
        )

        projected = set(ELIGIBLE_COLUMNS + LABEL_COLUMNS + FEATURE_COLUMNS + FES_COLUMNS)
        prohibited_outcomes = {
            "entry_bar_id",
            "entry_price",
            "atr_20_decision",
            "exit_price_60",
            "label_unavailable_reason_60",
            "mfe_60_ticks",
            "mae_60_ticks",
            "stop_hit_60",
            "target_hit_60",
            "forward_return_120_ticks",
        }
        self.assertFalse(projected & prohibited_outcomes)


class CapabilityFilterRegressionTests(unittest.TestCase):
    @staticmethod
    def _eligible_row() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "observation_id": ["obs-1"],
                "decision_bar_id": ["bar-1"],
                "decision_timestamp_utc": pd.to_datetime(
                    ["2024-01-02T15:00:00Z"], utc=True
                ),
                "entry_timestamp_utc": pd.to_datetime(
                    ["2024-01-02T15:01:00Z"], utc=True
                ),
                "trade_date_ny": pd.to_datetime(["2024-01-02"]),
                "entry_session": ["New York"],
                "research_partition": ["Validation"],
            }
        )

    def test_validation_predictors_are_physically_filtered_to_selected_sessions(
        self,
    ) -> None:
        eligible = self._eligible_row()
        features = eligible[
            ["observation_id", "trade_date_ny", "entry_session", "research_partition"]
        ].copy()
        for feature in STAT15_FEATURES + ("normalized_ols_slope_30",):
            features[feature] = 0.0
        fes = eligible[
            [
                "observation_id",
                "decision_timestamp_utc",
                "trade_date_ny",
                "entry_session",
                "research_partition",
            ]
        ].copy()
        for feature in ALL_FES_SOURCE_FEATURES:
            fes[feature] = 0.0
        returned = iter((eligible, features, fes))

        with patch(
            "src.statistical_research.feature_combination_runner._read_projected",
            side_effect=lambda *args, **kwargs: next(returned),
        ) as read_mock:
            base, loaded_eligible = _load_predictor_base(
                Path("."),
                partition="Validation",
                sessions=("New York",),
                audit=[],
            )

        self.assertEqual(len(base), 1)
        self.assertEqual(len(loaded_eligible), 1)
        for call in read_mock.call_args_list:
            self.assertIn(
                ("entry_session", "in", ["New York"]),
                call.kwargs["filters"],
            )

    def test_validation_labels_are_physically_scoped_to_authorized_ids(self) -> None:
        with patch(
            "src.statistical_research.feature_combination_runner._read_projected",
            return_value=pd.DataFrame(columns=LABEL_COLUMNS),
        ) as read_mock:
            _load_labels(
                Path("."),
                partition="Validation",
                stage="test",
                observation_ids=("obs-1", "obs-2"),
                audit=[],
            )

        filters = read_mock.call_args.kwargs["filters"]
        self.assertIn(("research_partition", "==", "Validation"), filters)
        self.assertIn(("observation_id", "in", ["obs-1", "obs-2"]), filters)
        self.assertTrue(any(item[0] == "entry_timestamp_utc" for item in filters))
        self.assertTrue(any(item[0] == "exit_timestamp_utc_60" for item in filters))
        self.assertEqual(tuple(read_mock.call_args.kwargs["columns"]), LABEL_COLUMNS)

    def test_validation_poi_candidates_are_physically_session_scoped(self) -> None:
        eligible = self._eligible_row()
        poi = pd.DataFrame(
            {
                "true_retest_id": ["poi-1"],
                "retest_bar_id": ["bar-1"],
                "retest_ts_event_utc": pd.to_datetime(
                    ["2024-01-02T15:00:00Z"], utc=True
                ),
                "trade_date_ny": pd.to_datetime(["2024-01-02"]),
                "direction": ["bullish"],
                "research_partition": ["validation"],
                "feat_execution_session": ["New York"],
                "feat_first_touch": [True],
                "feat_15bar_structural_validation": [True],
                **{feature: [0.0] for feature in POI_CONTEXT_FEATURES},
            }
        )

        with patch(
            "src.statistical_research.feature_combination_runner._read_projected",
            return_value=poi,
        ) as read_mock:
            candidates, audit = _load_poi_candidates(
                Path("."),
                partition="Validation",
                eligible=eligible,
                sessions=("New York",),
                audit=[],
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(audit["bridge_coverage"], 1.0)
        self.assertIn(
            ("feat_execution_session", "in", ["New York"]),
            read_mock.call_args.kwargs["filters"],
        )

    def test_rejected_final_row_preserves_forensic_access_state(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date_ny": pd.to_datetime(["2025-01-02"]),
                "research_partition": ["Final test"],
            }
        )

        class FakeArrowTable:
            def to_pandas(self, *, ignore_metadata: bool) -> pd.DataFrame:
                self.ignore_metadata = ignore_metadata
                return frame

        audit: list[dict[str, object]] = []
        with patch(
            "src.statistical_research.feature_combination_runner.pq.read_table",
            return_value=FakeArrowTable(),
        ):
            with self.assertRaises(BoundaryAccessError) as raised:
                _read_projected(
                    Path("."),
                    Path("synthetic.parquet"),
                    columns=("trade_date_ny", "research_partition"),
                    filters=[("research_partition", "==", "Validation")],
                    audit=audit,
                    stage="synthetic-boundary-test",
                )

        self.assertTrue(raised.exception.final_test_opened)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["boundary_check"], "FAILED")


def _selection_for_models(
    selected_model: str, models: tuple[str, ...], *, session: str = "New York"
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "session": session,
                "model_id": model_id,
                "development_selected": model_id == selected_model,
                "development_eligible": model_id == selected_model,
                "development_daily_ic": 0.03,
                "development_gate_failures": "",
                "validation_opened": False,
                "validation_confirmed": False,
                "validation_gate_failures": "",
            }
            for model_id in models
        ]
    )


def _fake_validation_summary(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    target_column: str,
    anchor_prediction_column: str | None = None,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    del prediction_column, target_column, anchor_prediction_column, seed
    return (
        {
            "daily_ic_mean": 0.03,
            "daily_ic_ci_low": 0.01,
            "daily_ic_ci_high": 0.05,
            "paired_daily_ic_improvement": 0.01,
            "paired_daily_ic_ci_low": 0.001,
            "paired_daily_ic_ci_high": 0.02,
            "trading_date_count": 200,
            "observation_count": 50_000,
            "sign_accuracy": 0.55,
            "best_ten_date_positive_evidence_share": 0.25,
        },
        pd.DataFrame({"trade_date_ny": pd.to_datetime(frame["trade_date_ny"])}),
    )


class ValidationOpeningRegressionTests(unittest.TestCase):
    @staticmethod
    def _validation_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "entry_session": ["New York", "New York"],
                "trade_date_ny": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "forward_return_60_atr": [0.1, -0.1],
            }
        )

    @staticmethod
    def _fake_fit(
        development: pd.DataFrame,
        validation: pd.DataFrame,
        *,
        features: tuple[str, ...],
        target: str,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        del development, target
        coefficients = pd.DataFrame(
            {"feature_name": features, "coefficient": np.zeros(len(features))}
        )
        return np.zeros(len(validation), dtype="float64"), coefficients

    @patch(
        "src.statistical_research.feature_combination_runner.validation_gate_failures",
        return_value=[],
    )
    @patch(
        "src.statistical_research.feature_combination_runner.summarize_predictions",
        side_effect=_fake_validation_summary,
    )
    @patch(
        "src.statistical_research.feature_combination_runner."
        "fit_development_predict_validation",
        side_effect=_fake_fit,
    )
    def test_gen2_validation_does_not_open_gen1(
        self,
        fit_mock,
        _summary_mock,
        _gate_mock,
    ) -> None:
        selection = _selection_for_models("GEN2_FES4", ("GEN1_STAT15", "GEN2_FES4"))
        metrics, _, _, updated, scored = _score_validation(
            branch="GENERAL",
            development_frames={"New York": pd.DataFrame()},
            validation_frame=self._validation_frame(),
            selection=selection,
        )

        self.assertEqual(metrics["model_id"].tolist(), ["GEN2_FES4"])
        self.assertEqual(fit_mock.call_count, 1)
        self.assertEqual(
            fit_mock.call_args.kwargs["features"],
            model_features("GENERAL", "GEN2_FES4", "New York"),
        )
        self.assertNotIn("pred_direction__GEN1_STAT15", scored["New York"].columns)
        selected = updated.loc[updated["model_id"].eq("GEN2_FES4")].iloc[0]
        self.assertTrue(selected["validation_opened"])

    @patch(
        "src.statistical_research.feature_combination_runner.validation_gate_failures",
        return_value=[],
    )
    @patch(
        "src.statistical_research.feature_combination_runner.summarize_predictions",
        side_effect=_fake_validation_summary,
    )
    @patch(
        "src.statistical_research.feature_combination_runner."
        "fit_development_predict_validation",
        side_effect=_fake_fit,
    )
    def test_nested_gen3_validation_opens_the_required_gen1_anchor(
        self,
        fit_mock,
        _summary_mock,
        _gate_mock,
    ) -> None:
        selection = _selection_for_models(
            "GEN3_STAT15_FES4", ("GEN1_STAT15", "GEN3_STAT15_FES4")
        )
        metrics, _, _, _, scored = _score_validation(
            branch="GENERAL",
            development_frames={"New York": pd.DataFrame()},
            validation_frame=self._validation_frame(),
            selection=selection,
        )

        self.assertEqual(
            metrics["model_id"].tolist(), ["GEN1_STAT15", "GEN3_STAT15_FES4"]
        )
        self.assertEqual(fit_mock.call_count, 2)
        fitted_features = [call.kwargs["features"] for call in fit_mock.call_args_list]
        self.assertEqual(
            fitted_features,
            [
                model_features("GENERAL", "GEN1_STAT15", "New York"),
                model_features("GENERAL", "GEN3_STAT15_FES4", "New York"),
            ],
        )
        self.assertIn("pred_direction__GEN1_STAT15", scored["New York"].columns)

    def test_validation_capability_cannot_be_consumed_twice(self) -> None:
        selection = _selection_for_models("GEN2_FES4", ("GEN2_FES4",))
        selection.loc[:, "validation_opened"] = True

        with self.assertRaisesRegex(PermissionError, "already consumed"):
            _score_validation(
                branch="GENERAL",
                development_frames={"New York": pd.DataFrame()},
                validation_frame=self._validation_frame(),
                selection=selection,
            )


class ValidationCapabilityCheckpointTests(unittest.TestCase):
    def test_realized_selection_and_thresholds_are_hashed_before_access(self) -> None:
        selection = _selection_for_models(
            "GEN2_FES4", ("GEN1_STAT15", "GEN2_FES4")
        )
        thresholds = pd.DataFrame(
            [
                {
                    "session": "New York",
                    "model_id": model_id,
                    "development_oof_prediction_count": 100,
                    "p10": -0.2,
                    "p90": 0.3,
                    "enabled": True,
                    "reason": "enabled",
                }
                for model_id in ("GEN1_STAT15", "GEN2_FES4")
            ]
        )

        capability = _build_validation_capability(
            branch="GENERAL",
            contract_sha="contract-sha",
            selection=selection,
            thresholds=thresholds,
        )

        self.assertTrue(capability["single_locked_batch"])
        self.assertEqual(
            capability["authorized_validation_cells"],
            [
                {
                    "session": "New York",
                    "model_id": "GEN2_FES4",
                    "p10": -0.2,
                    "p90": 0.3,
                    "enabled": True,
                    "reason": "enabled",
                }
            ],
        )
        self.assertEqual(len(capability["sha256"]), 64)


class PolicyScopeRegressionTests(unittest.TestCase):
    def test_eligible_dates_are_limited_to_requested_sessions(self) -> None:
        frames = {
            "London": pd.DataFrame(
                {
                    "trade_date_ny": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "pred_direction__GEN1_STAT15": [0.1, 0.2],
                    "forward_return_60_atr": [0.1, 0.2],
                }
            ),
            "New York": pd.DataFrame(
                {
                    "trade_date_ny": pd.to_datetime(
                        [
                            "2024-01-01",
                            "2024-02-01 14:00",
                            "2024-02-01 15:00",
                            "2024-02-02",
                        ],
                        format="mixed",
                    ),
                    "pred_direction__GEN2_FES4": [np.nan, 0.1, 0.2, 0.3],
                    "forward_return_60_atr": [0.1, 0.1, 0.2, np.nan],
                }
            ),
        }

        dates = _eligible_dates_for_frames(
            frames,
            branch="GENERAL",
            selected_sessions={"New York": "GEN2_FES4"},
        )

        self.assertEqual(
            dates,
            (pd.Timestamp("2024-02-01"),),
        )

    @patch(
        "src.statistical_research.feature_combination_runner._summarize_policy_scenarios",
        return_value=pd.DataFrame({"trade_count": [0]}),
    )
    @patch(
        "src.statistical_research.feature_combination_runner._policy_candidates_for_model",
        return_value=pd.DataFrame(),
    )
    def test_confirmed_policy_union_excludes_unconfirmed_sessions(
        self,
        candidate_mock,
        _summary_mock,
    ) -> None:
        selection = pd.DataFrame(
            [
                {
                    "session": "London",
                    "model_id": "GEN1_STAT15",
                    "development_selected": True,
                    "validation_confirmed": False,
                },
                {
                    "session": "New York",
                    "model_id": "GEN2_FES4",
                    "development_selected": True,
                    "validation_confirmed": True,
                },
            ]
        )

        _selected_policy(
            branch="GENERAL",
            partition="Validation",
            frames={"London": pd.DataFrame(), "New York": pd.DataFrame()},
            selection=selection,
            thresholds=pd.DataFrame(),
            eligible_dates=pd.to_datetime(["2024-02-01"]),
            confirmed_only=True,
        )

        self.assertEqual(
            candidate_mock.call_args.kwargs["selected_sessions"],
            {"New York": "GEN2_FES4"},
        )


class ArtifactPublicationRegressionTests(unittest.TestCase):
    def test_schema_correct_empty_frame_persists_and_zero_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "artifacts"
            with self.assertRaisesRegex(ValueError, "frozen schema"):
                _persist_frames(output_dir, {"bad": pd.DataFrame()})

            expected = pd.DataFrame(
                {
                    "observation_id": pd.Series(dtype="int64"),
                    "trade_date_ny": pd.Series(dtype="datetime64[ns]"),
                }
            )
            _persist_frames(output_dir, {"good": expected})

            materialized = pd.read_parquet(output_dir / "good.parquet")
            self.assertTrue(materialized.empty)
            self.assertEqual(materialized.columns.tolist(), expected.columns.tolist())
            self.assertEqual(materialized.dtypes.tolist(), expected.dtypes.tolist())

    def test_blocked_publication_replaces_stale_success_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target_output = root / "published-output"
            target_report = root / "published-report"
            staging_output = root / "staging-output"
            staging_report = root / "staging-report"
            target_output.mkdir()
            target_report.mkdir()
            (target_output / "stale-success.parquet").write_text("stale", encoding="utf-8")
            (target_report / "stale-success.md").write_text("stale", encoding="utf-8")

            _write_blocked_artifacts(
                staging_output,
                staging_report,
                contract_sha="contract-sha",
                error=ValueError("synthetic integrity failure"),
            )
            _publish_directories(
                (
                    (staging_output, target_output),
                    (staging_report, target_report),
                )
            )

            self.assertFalse((target_output / "stale-success.parquet").exists())
            self.assertFalse((target_report / "stale-success.md").exists())
            self.assertFalse(staging_output.exists())
            self.assertFalse(staging_report.exists())
            headline = json.loads((target_output / "headline.json").read_text("utf-8"))
            failure = json.loads(
                (target_output / "integrity_failure.json").read_text("utf-8")
            )
            self.assertEqual(headline["overall_state"], "BLOCKED_INTEGRITY_FAILURE")
            self.assertFalse(headline["final_test_opened"])
            self.assertEqual(failure["error_type"], "ValueError")
            self.assertIn("STATUS: BLOCKED", (target_report / "status.md").read_text("utf-8"))
            self.assertEqual(list(root.glob(".*.backup-*")), [])

    def test_boundary_blocked_state_never_claims_final_remained_unopened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            report_dir = root / "report"
            access_record = {
                "stage": "synthetic-boundary-test",
                "path": "synthetic.parquet",
                "projected_columns": "trade_date_ny|research_partition",
                "filters": "[]",
                "materialized_rows": 1,
                "materialized_columns": 2,
                "minimum_trade_date": pd.Timestamp("2025-01-02"),
                "maximum_trade_date": pd.Timestamp("2025-01-02"),
                "boundary_check": "FAILED",
            }

            _write_blocked_artifacts(
                output_dir,
                report_dir,
                contract_sha="contract-sha",
                error=PermissionError("synthetic final boundary violation"),
                final_test_opened=True,
                access_record=access_record,
            )

            headline = json.loads((output_dir / "headline.json").read_text("utf-8"))
            audit = json.loads(
                (output_dir / "blocked_access_audit.json").read_text("utf-8")
            )
            status = (report_dir / "status.md").read_text("utf-8")
            self.assertTrue(headline["final_test_opened"])
            self.assertEqual(
                headline["final_test_access_state"],
                "OPENED_BY_INTEGRITY_VIOLATION",
            )
            self.assertFalse(audit["outcome_values_reported"])
            self.assertIn("was materialized and immediately rejected", status)


if __name__ == "__main__":
    unittest.main()
