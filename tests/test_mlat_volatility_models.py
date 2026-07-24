from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.statistical_research.mlat_artifacts import (
    ArtifactVerificationError,
    MLATArtifactPaths,
    load_saved_table,
    save_csv,
    save_json,
    save_parquet,
    verify_saved_table,
)
from src.statistical_research.mlat_volatility_models import (
    GARCHParameters,
    _coerce_returns_frame,
    _complete_rolling_sum,
    _forward_realized_volatility,
    audit_segmented_garch,
    fit_segmented_garch,
    fit_volatility_calibration,
    forecast_segmented_garch,
    prepare_segmented_log_returns,
)

HORIZON_MINUTES = 10


def _garch_segment(start: str, n_rows: int, segment: int, seed: int) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    innovations = generator.standard_normal(n_rows)
    returns = np.zeros(n_rows, dtype=np.float64)
    variance = 2.0e-8
    for position in range(1, n_rows):
        variance = 2.0e-9 + 0.10 * returns[position - 1] ** 2 + 0.84 * variance
        returns[position] = np.sqrt(variance) * innovations[position]

    close = 2_000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    timestamp = pd.date_range(start, periods=n_rows, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "source_row_id": np.arange(n_rows, dtype=np.int64) + segment * 10_000,
            "ts_event_utc": timestamp,
            "product": "GC",
            "symbol": f"GC_TEST_{segment}",
            "active_symbol": f"GC_TEST_{segment}",
            "instrument_id": np.uint32(segment),
            "continuous_segment_id": np.int32(segment),
            "continuity_run_id": np.int64(segment),
            "open": open_,
            "high": np.maximum(open_, close) + 0.25,
            "low": np.minimum(open_, close) - 0.25,
            "close": close,
            "tradable_research_flag": True,
            "roll_window_flag": False,
            "low_liquidity_warning_flag": False,
        }
    )


def _bars(*, include_micro: bool = True) -> pd.DataFrame:
    gc = pd.concat(
        [
            _garch_segment("2022-01-03 06:00:00", 400, 1, 101),
            _garch_segment("2023-01-03 06:00:00", 180, 2, 202),
            _garch_segment("2024-01-03 06:00:00", 180, 3, 303),
        ],
        ignore_index=True,
    )
    if not include_micro:
        return gc
    micro = gc.iloc[::50].copy()
    micro["source_row_id"] = micro["source_row_id"].to_numpy() + 1_000_000
    micro["product"] = "MGC"
    micro["symbol"] = "MGC_TEST"
    micro["active_symbol"] = "MGC_TEST"
    micro["instrument_id"] = np.uint32(999)
    micro["continuity_run_id"] = np.int64(999)
    return pd.concat([gc, micro], ignore_index=True)


def _method_columns() -> dict[str, str]:
    return {
        "garch": f"garch_sigma_forecast_{HORIZON_MINUTES}m",
        "realized_volatility_anchor": f"rv_anchor_sigma_{HORIZON_MINUTES}m",
        "atr_anchor": f"atr_anchor_sigma_{HORIZON_MINUTES}m",
    }


class TestSegmentedGARCHAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bars = _bars()
        cls.audit = audit_segmented_garch(
            cls.bars,
            horizon_minutes=HORIZON_MINUTES,
            diagnostic_lags=(5, 10),
            min_fit_observations=250,
            min_calibration_observations=30,
            max_iterations=250,
        )

    def test_returns_are_gc_only_and_null_at_every_segment_boundary(self) -> None:
        prepared = prepare_segmented_log_returns(self.bars)

        self.assertEqual(set(prepared["product"]), {"GC"})
        self.assertEqual(
            prepared.attrs["excluded_non_gc_rows"],
            int(self.bars["product"].eq("MGC").sum()),
        )
        first_rows = prepared.groupby("continuity_run_id", sort=False).head(1)
        self.assertTrue(first_rows["log_return"].isna().all())
        self.assertEqual(prepared["continuity_run_id"].nunique(), 3)

        for _, run in prepared.groupby("continuity_run_id", sort=False):
            expected = np.log(
                run["close"].to_numpy(dtype=np.float64)[1:]
                / run["close"].to_numpy(dtype=np.float64)[:-1]
            )
            np.testing.assert_array_equal(
                run["log_return"].to_numpy(dtype=np.float64)[1:],
                expected,
            )

    def test_fragmented_runs_use_contiguous_slices_not_full_array_rescans(self) -> None:
        runs = np.arange(4_000, dtype=np.int64)
        values = np.ones(len(runs), dtype=np.float64)
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=len(runs), freq="min", tz="UTC"))
        original_flatnonzero = np.flatnonzero

        def reject_full_length_scan(array: np.ndarray) -> np.ndarray:
            candidate = np.asarray(array)
            if candidate.size == len(runs):
                self.fail("helper rescanned the full array inside a per-run loop")
            return original_flatnonzero(candidate)

        with mock.patch(
            "src.statistical_research.mlat_volatility_models.np.flatnonzero",
            side_effect=reject_full_length_scan,
        ):
            trailing = _complete_rolling_sum(values, runs, window=2)
            realized, target_end = _forward_realized_volatility(
                values,
                runs,
                timestamps,
                horizon=2,
            )

        self.assertTrue(np.isnan(trailing).all())
        self.assertTrue(np.isnan(realized).all())
        self.assertTrue(target_end.isna().all())

    def test_prepared_returns_are_sanitized_at_new_strict_gap_boundaries(self) -> None:
        prepared = pd.DataFrame(
            {
                "source_row_id": [0, 1, 2],
                "ts_event_utc": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00+00:00",
                        "2024-01-01 00:01:00+00:00",
                        "2024-01-01 00:05:00+00:00",
                    ],
                    utc=True,
                ),
                "product": "GC",
                "continuity_run_id": [7, 7, 7],
                "log_return": [np.nan, 0.01, 0.50],
            }
        )
        coerced = _coerce_returns_frame(
            prepared,
            timestamp_column="ts_event_utc",
            return_column="log_return",
            segment_column="continuity_run_id",
        )

        self.assertEqual(coerced["continuity_run_id"].tolist(), [0, 0, 1])
        self.assertTrue(np.isnan(coerced.loc[2, "log_return"]))
        self.assertFalse(bool(coerced.loc[2, "return_available"]))
        self.assertEqual(coerced.attrs["sanitized_boundary_returns"], 1)

    def test_fit_scope_is_unchanged_by_development_and_validation_data(self) -> None:
        modified = self.bars.copy()
        timestamp = pd.to_datetime(modified["ts_event_utc"], utc=True)
        future = modified["product"].eq("GC") & timestamp.ge("2023-01-01")
        phase = np.arange(int(future.sum()), dtype=np.float64)
        modified.loc[future, "close"] *= np.exp(0.03 * np.sin(phase / 9.0))

        refit = fit_segmented_garch(
            modified,
            min_observations=250,
            max_iterations=250,
        )
        pd.testing.assert_frame_equal(
            self.audit.parameters,
            refit.parameters.to_frame(),
            check_exact=True,
        )

    def test_fit_is_deterministic_and_parameters_satisfy_constraints(self) -> None:
        repeated = fit_segmented_garch(
            self.bars,
            min_observations=250,
            max_iterations=250,
        )
        pd.testing.assert_frame_equal(
            self.audit.parameters,
            repeated.parameters.to_frame(),
            check_exact=True,
        )
        parameters = repeated.parameters
        self.assertGreater(parameters.omega, 0.0)
        self.assertGreaterEqual(parameters.alpha, 0.0)
        self.assertGreaterEqual(parameters.beta, 0.0)
        self.assertLess(parameters.persistence, 1.0)
        self.assertTrue(repeated.usable)

    def test_forecast_recursion_resets_and_mean_reverts_across_horizon(self) -> None:
        forecasts = self.audit.forecasts
        parameters = self.audit.fit.parameters

        first_rows = forecasts.groupby("continuity_run_id", sort=False).head(1)
        np.testing.assert_array_equal(
            first_rows["garch_variance_forecast_1m"].to_numpy(dtype=np.float64),
            np.full(len(first_rows), parameters.initial_variance),
        )
        finite = forecasts["log_return"].notna()
        row = forecasts.loc[finite].iloc[20]
        expected_next = (
            parameters.omega
            + parameters.alpha * (float(row["log_return"]) - parameters.mean) ** 2
            + parameters.beta * float(row["garch_conditional_variance_1m"])
        )
        self.assertEqual(float(row["garch_variance_forecast_1m"]), expected_next)
        persistence = parameters.persistence
        unconditional = parameters.omega / (1.0 - persistence)
        decay_sum = (1.0 - persistence**HORIZON_MINUTES) / (1.0 - persistence)
        expected_cumulative = (
            HORIZON_MINUTES * unconditional + (expected_next - unconditional) * decay_sum
        )
        self.assertEqual(
            float(row[f"garch_sigma_forecast_{HORIZON_MINUTES}m"]),
            np.sqrt(expected_cumulative),
        )

        run = forecasts.loc[forecasts["continuity_run_id"].eq(0)].reset_index(drop=True)
        first_finite = int(np.flatnonzero(run["log_return"].notna().to_numpy())[0])
        self.assertEqual(
            float(run.loc[first_finite + 1, "garch_conditional_variance_1m"]),
            float(run.loc[first_finite, "garch_variance_forecast_1m"]),
        )

        default_forecasts = forecast_segmented_garch(self.bars, self.audit.fit)
        default_decay_sum = (1.0 - persistence**60) / (1.0 - persistence)
        default_next = default_forecasts["garch_variance_forecast_1m"].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(
            default_forecasts["garch_sigma_forecast_60m"].to_numpy(dtype=np.float64),
            np.sqrt(60.0 * unconditional + (default_next - unconditional) * default_decay_sum),
        )

    def test_calibration_is_fit_on_2023_and_unchanged_by_2024(self) -> None:
        modified = self.bars.copy()
        timestamp = pd.to_datetime(modified["ts_event_utc"], utc=True)
        validation = modified["product"].eq("GC") & timestamp.ge("2024-01-01")
        phase = np.arange(int(validation.sum()), dtype=np.float64)
        modified.loc[validation, "close"] *= np.exp(0.04 * np.cos(phase / 7.0))
        modified_forecasts = forecast_segmented_garch(
            modified,
            self.audit.fit,
            horizon_minutes=HORIZON_MINUTES,
        )
        modified_calibration = fit_volatility_calibration(
            modified_forecasts,
            method_columns=_method_columns(),
            target_column=f"realized_volatility_{HORIZON_MINUTES}m",
            target_end_column=f"target_{HORIZON_MINUTES}m_end_timestamp_utc",
            min_observations=30,
        )

        pd.testing.assert_frame_equal(
            self.audit.calibration,
            modified_calibration,
            check_exact=True,
        )
        self.assertTrue(self.audit.calibration["success"].all())
        self.assertTrue(
            self.audit.calibration["calibration_end_utc"]
            .eq(pd.Timestamp("2023-12-31 23:59:59.999999999", tz="UTC"))
            .all()
        )

    def test_audit_separates_periods_diagnostics_and_simple_anchors(self) -> None:
        self.assertEqual(
            set(self.audit.forecasts["audit_period"].dropna()),
            {"fit", "development_audit", "validation"},
        )
        self.assertEqual(
            set(self.audit.residual_diagnostics["diagnostic"]),
            {
                "ljung_box_standardized_residual",
                "ljung_box_squared_standardized_residual",
                "arch_lm_standardized_residual",
            },
        )
        self.assertEqual(
            set(self.audit.benchmark_metrics["method"]),
            {"garch", "realized_volatility_anchor", "atr_anchor"},
        )
        self.assertEqual(
            set(self.audit.benchmark_metrics["period"]),
            {"development_audit", "validation"},
        )
        self.assertTrue(
            {
                f"rv_anchor_sigma_{HORIZON_MINUTES}m",
                f"atr_anchor_sigma_{HORIZON_MINUTES}m",
            }.issubset(self.audit.forecasts.columns)
        )
        self.assertFalse(self.audit.audit_checks["advancement_authorized"].any())
        self.assertIn(
            "audit_only_no_feature_authorization",
            set(self.audit.audit_checks["check_name"]),
        )

    def test_optimizer_failure_is_reported_in_diagnostics(self) -> None:
        failed = fit_segmented_garch(
            self.bars,
            min_observations=250,
            max_iterations=0,
        )
        optimizer = failed.diagnostics.set_index("check_name").loc["optimizer_success"]
        self.assertFalse(bool(optimizer["passed"]))
        self.assertFalse(failed.parameters.optimizer_success)
        self.assertFalse(failed.usable)
        self.assertIn("iterations=", str(optimizer["details"]))


class TestMLATArtifactPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = GARCHParameters(
            mean=1.25e-6,
            omega=2.5e-9,
            alpha=0.08,
            beta=0.87,
            initial_variance=5.0e-8,
            return_scale=10_000.0,
            fit_end_utc=pd.Timestamp(
                "2022-12-31 23:59:59.999999999",
                tz="UTC",
            ),
            n_fit_observations=12_345,
            n_fit_segments=7,
            optimizer_method="L-BFGS-B",
            optimizer_success=True,
            optimizer_status=0,
            optimizer_message="deterministic convergence",
            objective_value=1.2345678901234567,
            iterations=18,
        )
        self.table = self.parameters.to_frame()

    def test_versioned_paths_match_the_locked_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            # Resolve so the expected paths match MLATArtifactPaths, which resolves
            # its root; on Windows the raw TemporaryDirectory can be an 8.3 short
            # path (e.g. RUNNER~1) that resolve() expands to the long form.
            root = Path(temporary).resolve()
            paths = MLATArtifactPaths(root, version="v1").ensure()

            self.assertEqual(
                paths.data_dir,
                root
                / "data"
                / "processed"
                / "statistical_research"
                / "mlat_feature_research"
                / "v1",
            )
            self.assertEqual(
                paths.report_dir,
                root / "reports" / "statistical_research" / "mlat_feature_research" / "v1",
            )
            self.assertEqual(
                paths.figure_dir,
                root / "reports" / "figures" / "mlat_feature_research" / "v1",
            )
            self.assertTrue(paths.data_dir.is_dir())
            self.assertTrue(paths.table_dir.is_dir())
            self.assertTrue(paths.figure_dir.is_dir())
            self.assertEqual(paths.figure_path("volatility_audit").suffix, ".png")
            with self.assertRaises(ValueError):
                paths.data_path("../escape")

    def test_parameter_tables_save_reload_and_manifest_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = MLATArtifactPaths(temporary, version="garch_v1").ensure()
            parquet_path = paths.data_path("garch_parameters")
            csv_path = paths.table_path("garch_parameters")
            json_path = paths.json_path("garch_parameters")
            parquet_entry = save_parquet(
                self.table,
                parquet_path,
                id_columns=("model_version",),
                manifest_path=paths.manifest_path,
            )
            csv_entry = save_csv(
                self.table,
                csv_path,
                id_columns=("model_version",),
                manifest_path=paths.manifest_path,
            )
            save_json(
                {"parameters": self.table.to_dict(orient="records")},
                json_path,
                manifest_path=paths.manifest_path,
            )

            parquet_checks = verify_saved_table(
                self.table,
                parquet_path,
                id_columns=("model_version",),
                manifest_entry=parquet_entry,
            )
            csv_checks = verify_saved_table(
                self.table,
                csv_path,
                id_columns=("model_version",),
                manifest_entry=csv_entry,
            )
            self.assertTrue(parquet_checks["passed"].all())
            self.assertTrue(csv_checks["passed"].all())

            reloaded = load_saved_table(parquet_path, expected=self.table)
            reconstructed = GARCHParameters.from_frame(reloaded)
            pd.testing.assert_frame_equal(
                self.table,
                reconstructed.to_frame(),
                check_exact=True,
            )
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["layout_version"], "mlat-artifacts-v1")
            self.assertEqual(len(manifest["artifacts"]), 3)
            self.assertTrue(
                all(
                    not Path(artifact_path).is_absolute() for artifact_path in manifest["artifacts"]
                )
            )

    def test_verification_exposes_actionable_failure_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "parameters.parquet"
            entry = save_parquet(
                self.table,
                path,
                id_columns=("model_version",),
            )
            changed = self.table.copy()
            changed.loc[0, "omega"] *= 2.0
            diagnostics = verify_saved_table(
                changed,
                path,
                id_columns=("model_version",),
                manifest_entry=entry,
                raise_on_failure=False,
            ).set_index("check_name")

            self.assertFalse(bool(diagnostics.loc["deterministic_sample_hash", "passed"]))
            self.assertFalse(bool(diagnostics.loc["full_table_exact", "passed"]))
            self.assertFalse(bool(diagnostics.loc["manifest_sample_hash", "passed"]))
            with self.assertRaises(ArtifactVerificationError) as raised:
                verify_saved_table(
                    changed,
                    path,
                    id_columns=("model_version",),
                    manifest_entry=entry,
                )
            self.assertIn("full_table_exact", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
