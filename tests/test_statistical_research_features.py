from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.feature_engineering import (
    BOUNDARY_REASON_LABELS,
    FEATURE_SOURCE_COLUMNS,
    METADATA_COLUMNS,
    _fit_time_of_day_reference,
    _rolling_regression_slope,
    build_continuity_run_id,
    build_feature_matrix,
)
from src.statistical_research.feature_registry import (
    EXPERIMENTAL_FEATURE_NAMES,
    FEATURE_NAMES,
    feature_registry_frame,
    validate_registry,
)
from src.statistical_research.feature_validation import compute_feature_diagnostics


def _bars(n: int = 500, *, start: str = "2024-01-02 06:00:00+00:00", flat: bool = False) -> pd.DataFrame:
    utc = pd.date_range(start, periods=n, freq="min", tz="UTC")
    ny = utc.tz_convert("America/New_York")
    if flat:
        close = np.full(n, 2_000.0)
        open_ = close.copy()
        high = close.copy()
        low = close.copy()
    else:
        changes = np.resize(np.array([0.1, 0.2, -0.1, 0.0, 0.3, -0.2], dtype=np.float64), n)
        close = 2_000.0 + np.cumsum(changes)
        open_ = np.r_[2_000.0, close[:-1]]
        high = np.maximum(open_, close) + 0.2
        low = np.minimum(open_, close) - 0.2
    frame = pd.DataFrame(
        {
            "source_row_id": np.arange(n, dtype=np.int64),
            "ts_event_utc": utc,
            "ts_event_ny": ny,
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
            "volume": np.resize(np.array([100, 120, 90, 150, 110, 130], dtype=np.uint64), n),
            "tradable_research_flag": True,
            "roll_window_flag": False,
            "low_liquidity_warning_flag": False,
        }
    )
    return frame.loc[:, FEATURE_SOURCE_COLUMNS]


def _observations(bars: pd.DataFrame, decision_positions: list[int], partition: str = "Development") -> pd.DataFrame:
    records = []
    for observation_id, position in enumerate(decision_positions):
        decision = bars.iloc[position]
        entry = bars.iloc[position + 1]
        entry_ny = pd.Timestamp(entry["ts_event_ny"])
        minute = entry_ny.hour * 60 + entry_ny.minute
        session = "New York" if 420 <= minute < 720 else "London"
        records.append(
            {
                "observation_id": observation_id,
                "decision_bar_id": int(decision["source_row_id"]),
                "entry_bar_id": int(entry["source_row_id"]),
                "decision_timestamp_utc": decision["ts_event_utc"],
                "decision_timestamp_ny": decision["ts_event_ny"],
                "entry_timestamp_utc": entry["ts_event_utc"],
                "entry_timestamp_ny": entry["ts_event_ny"],
                "trade_date_ny": pd.Timestamp(entry_ny.date()),
                "entry_session": session,
                "research_partition": partition,
                "product": "GC",
                "symbol": entry["symbol"],
                "active_symbol": entry["active_symbol"],
                "instrument_id": entry["instrument_id"],
                "continuous_segment_id": entry["continuous_segment_id"],
                "minutes_to_forced_exit": max(1, 930 - minute),
            }
        )
    return pd.DataFrame.from_records(records)


class FeatureRegistryTests(unittest.TestCase):
    def test_registry_is_frozen_and_one_row_per_feature(self):
        registry = feature_registry_frame()
        validate_registry(registry)
        self.assertEqual(len(registry), 85)
        self.assertEqual(registry["feature_name"].tolist(), list(FEATURE_NAMES))
        self.assertEqual(int(registry["is_experimental"].sum()), 6)
        self.assertEqual(int(registry["output_dtype"].isin(["float32", "Int16", "Int8"]).sum()), 80)

    def test_experimental_names_and_reader_marks_are_consistent(self):
        registry = feature_registry_frame().set_index("feature_name")
        self.assertEqual(len(EXPERIMENTAL_FEATURE_NAMES), 6)
        for name in EXPERIMENTAL_FEATURE_NAMES:
            self.assertTrue(name.endswith("_exp"))
            self.assertNotIn("*", name)
            self.assertTrue(str(registry.loc[name, "display_name"]).startswith("* "))
            self.assertTrue(str(registry.loc[name, "creative_rationale"]))


class FeatureCausalityAndBoundaryTests(unittest.TestCase):
    def test_future_and_entry_bar_mutation_cannot_change_decision_features(self):
        bars = _bars()
        observations = _observations(bars, [150])
        baseline = build_feature_matrix(bars, observations, expected_rows=None).matrix
        mutated = bars.copy()
        future = np.arange(len(mutated)) > 150
        mutated.loc[future, ["open", "high", "low", "close"]] = (
            mutated.loc[future, ["open", "high", "low", "close"]].to_numpy() + 500.0
        )
        mutated.loc[future, "volume"] = mutated.loc[future, "volume"] * 100
        recomputed = build_feature_matrix(mutated, observations, expected_rows=None).matrix
        self.assertTrue(baseline.loc[:, FEATURE_NAMES].equals(recomputed.loc[:, FEATURE_NAMES]))

    def test_continuity_reason_priority_covers_locked_boundaries(self):
        bars = _bars(40)
        bars.loc[5, "ts_event_utc"] = bars.loc[5, "ts_event_utc"] + pd.Timedelta(minutes=1)
        bars.loc[10:, ["symbol", "active_symbol"]] = "GCG5"
        bars.loc[15:, "instrument_id"] = 202
        bars.loc[20:, "continuous_segment_id"] = 2
        bars.loc[25, "tradable_research_flag"] = False
        _, _, reasons = build_continuity_run_id(bars)
        labels = {i: BOUNDARY_REASON_LABELS[int(code)] for i, code in enumerate(reasons) if code}
        self.assertEqual(labels[5], "timestamp_gap")
        self.assertEqual(labels[10], "selected_contract_change")
        self.assertEqual(labels[15], "instrument_change")
        self.assertEqual(labels[20], "continuous_segment_change")
        self.assertEqual(labels[25], "tradability_roll_or_liquidity_boundary")

    def test_gap_and_contract_change_null_complete_windows(self):
        bars = _bars()
        bars.loc[180, "ts_event_utc"] = bars.loc[180, "ts_event_utc"] + pd.Timedelta(minutes=1)
        bars.loc[300:, ["symbol", "active_symbol"]] = "GCG5"
        observations = _observations(bars, [185, 250, 305, 370])
        matrix = build_feature_matrix(bars, observations, expected_rows=None).matrix
        self.assertTrue(pd.isna(matrix.loc[0, "return_15m_bps"]))
        self.assertFalse(pd.isna(matrix.loc[1, "return_15m_bps"]))
        self.assertTrue(pd.isna(matrix.loc[2, "return_15m_bps"]))
        self.assertFalse(pd.isna(matrix.loc[3, "return_15m_bps"]))

    def test_execution_session_vwap_is_honestly_null_at_session_open(self):
        bars = _bars(520)
        # Source begins at 01:00 New York: positions 119/120 map entry 03:00/03:01;
        # positions 359/360 map entry 07:00/07:01.
        observations = _observations(bars, [119, 120, 359, 360])
        matrix = build_feature_matrix(bars, observations, expected_rows=None).matrix
        self.assertTrue(pd.isna(matrix.loc[0, "distance_from_execution_session_vwap_atr"]))
        self.assertFalse(pd.isna(matrix.loc[1, "distance_from_execution_session_vwap_atr"]))
        self.assertTrue(pd.isna(matrix.loc[2, "distance_from_execution_session_vwap_atr"]))
        self.assertFalse(pd.isna(matrix.loc[3, "distance_from_execution_session_vwap_atr"]))

    def test_research_day_vwap_resets_at_0100_New_York(self):
        bars = _bars(120, start="2024-01-02 05:30:00+00:00")  # 00:30 New York
        observations = _observations(bars, [30])  # decision is exactly 01:00
        matrix = build_feature_matrix(bars, observations, expected_rows=None).matrix
        position = 30
        typical = (bars.loc[position, "high"] + bars.loc[position, "low"] + bars.loc[position, "close"]) / 3.0
        true_range = []
        for i in range(position - 19, position + 1):
            previous_close = bars.loc[i - 1, "close"]
            true_range.append(max(
                bars.loc[i, "high"] - bars.loc[i, "low"],
                abs(bars.loc[i, "high"] - previous_close),
                abs(bars.loc[i, "low"] - previous_close),
            ))
        expected = (bars.loc[position, "close"] - typical) / np.mean(true_range)
        self.assertAlmostEqual(float(matrix.loc[0, "distance_from_research_day_vwap_atr"]), expected, places=5)

    def test_zero_denominators_never_create_infinity(self):
        bars = _bars(flat=True)
        observations = _observations(bars, [150])
        matrix = build_feature_matrix(bars, observations, expected_rows=None).matrix
        for name in FEATURE_NAMES:
            if pd.api.types.is_numeric_dtype(matrix[name].dtype):
                values = pd.to_numeric(matrix[name], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                self.assertFalse(np.isinf(values).any(), name)
        self.assertTrue(pd.isna(matrix.loc[0, "body_to_range"]))
        self.assertTrue(pd.isna(matrix.loc[0, "vwap_elasticity_30_exp"]))

    def test_rolling_regression_is_null_for_zero_predictor_variance(self):
        predictor = np.ones(50)
        response = np.linspace(-1, 1, 50)
        groups = np.zeros(50, dtype=np.int64)
        slope = _rolling_regression_slope(predictor, response, 30, groups)
        self.assertTrue(np.isnan(slope[29:]).all())


class FeatureReferenceAndAssemblyTests(unittest.TestCase):
    def _reference_observations(self) -> tuple[pd.DataFrame, np.ndarray]:
        rows = []
        values = []
        for date_number, date in enumerate(pd.bdate_range("2023-01-02", periods=40)):
            for minute in range(15):
                timestamp = (date + pd.Timedelta(hours=3, minutes=minute)).tz_localize("America/New_York")
                rows.append(
                    {
                        "entry_session": "London",
                        "research_partition": "Development",
                        "entry_timestamp_ny": timestamp,
                        "trade_date_ny": date,
                    }
                )
                values.append(np.log1p(100 + date_number + minute))
        for partition, offset in (("Validation", 1_000), ("Final test", 2_000)):
            timestamp = pd.Timestamp("2024-01-02 03:00", tz="America/New_York")
            rows.append(
                {
                    "entry_session": "London",
                    "research_partition": partition,
                    "entry_timestamp_ny": timestamp,
                    "trade_date_ny": pd.Timestamp("2024-01-02"),
                }
            )
            values.append(np.log1p(offset))
        return pd.DataFrame(rows), np.asarray(values, dtype=np.float64)

    def test_validation_and_final_values_cannot_change_reference_parameters(self):
        observations, log_volume = self._reference_observations()
        _, _, reference_a = _fit_time_of_day_reference(observations, log_volume)
        mutated = log_volume.copy()
        non_development = ~observations["research_partition"].eq("Development").to_numpy()
        mutated[non_development] += 50.0
        _, _, reference_b = _fit_time_of_day_reference(observations, mutated)
        pd.testing.assert_frame_equal(reference_a, reference_b)

    def test_Development_reference_excludes_current_trading_date(self):
        observations, log_volume = self._reference_observations()
        zscore, _, _ = _fit_time_of_day_reference(observations, log_volume)
        current_date = observations.loc[0, "trade_date_ny"]
        mask = (
            observations["research_partition"].eq("Development")
            & ~observations["trade_date_ny"].eq(current_date)
        )
        baseline = log_volume[mask.to_numpy()]
        expected = (log_volume[0] - baseline.mean()) / baseline.std(ddof=1)
        self.assertAlmostEqual(zscore[0], expected, places=12)

    def test_matrix_order_and_saved_dtype_contract_follow_registry(self):
        bars = _bars()
        observations = _observations(bars, [120, 150, 200])
        result = build_feature_matrix(bars, observations, expected_rows=None)
        self.assertEqual(list(result.matrix.columns), list(METADATA_COLUMNS) + list(FEATURE_NAMES))
        registry = result.registry.set_index("feature_name")
        for name in FEATURE_NAMES:
            expected = str(registry.loc[name, "output_dtype"])
            self.assertEqual(str(result.matrix[name].dtype), expected)

    def test_experimental_bounded_features_respect_definitions(self):
        bars = _bars()
        observations = _observations(bars, [120, 150, 200, 250, 300, 350])
        matrix = build_feature_matrix(bars, observations, expected_rows=None).matrix
        for name in ("directional_energy_balance_15_exp", "wick_pressure_balance_10_exp"):
            values = pd.to_numeric(matrix[name], errors="coerce").dropna()
            self.assertTrue(values.between(-1, 1).all(), name)
        age = pd.to_numeric(matrix["compression_age_exp"], errors="coerce").dropna()
        self.assertTrue(age.between(0, 60).all())

    def test_diagnostics_are_feature_only_and_label_free(self):
        bars = _bars()
        observations = _observations(bars, [120, 150, 200, 250])
        result = build_feature_matrix(bars, observations, expected_rows=None)
        diagnostics = compute_feature_diagnostics(result)
        names = set(diagnostics.loc[diagnostics["feature_name"].ne("__family_aggregate__"), "feature_name"])
        self.assertEqual(names, set(FEATURE_NAMES))
        self.assertFalse(any("forward" in column.lower() for column in diagnostics.columns))


if __name__ == "__main__":
    unittest.main()
