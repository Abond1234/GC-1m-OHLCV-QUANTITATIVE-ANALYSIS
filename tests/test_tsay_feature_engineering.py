"""Outcome-free construction tests for the frozen Tsay feature family."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.feature_engineering import build_continuity_run_id
from src.statistical_research.tsay_feature_engineering import (
    TSAY_FEATURE_SOURCE_COLUMNS,
    _batched_ar_arch,
    _batched_tail_features,
    _canonical_primitives,
    _clock_adjusted_series,
    _primitive_equality_table,
    build_tsay_feature_matrix,
    reference_ar_arch_from_closes,
)
from src.statistical_research.tsay_feature_registry import TSAY_FEATURE_COLUMNS


def _fixture(*, dates: int = 32, bars_per_date: int = 150) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    source_row_id = 0
    start = pd.Timestamp("2023-01-02", tz="America/New_York")
    for date_index in range(dates):
        date_start = start + pd.Timedelta(days=date_index)
        for minute in range(bars_per_date):
            timestamp_ny = date_start + pd.Timedelta(hours=8, minutes=minute)
            phase = source_row_id
            close = 1_900.0 + 0.002 * phase + 0.14 * np.sin(phase / 7.0)
            rows.append(
                {
                    "source_row_id": source_row_id,
                    "ts_event_utc": timestamp_ny.tz_convert("UTC"),
                    "ts_event_ny": timestamp_ny,
                    "trade_date_ny": timestamp_ny.tz_localize(None).normalize(),
                    "product": "GC",
                    "symbol": "GCZ3",
                    "active_symbol": "GCZ3",
                    "instrument_id": 1,
                    "continuous_segment_id": 1,
                    "open": close - 0.03,
                    "high": close + 0.12 + 0.01 * (phase % 3),
                    "low": close - 0.11,
                    "close": close,
                    "volume": 100 + (phase % 37),
                    "tradable_research_flag": True,
                    "roll_window_flag": False,
                    "low_liquidity_warning_flag": False,
                }
            )
            source_row_id += 1
    bars = pd.DataFrame(rows).loc[:, TSAY_FEATURE_SOURCE_COLUMNS]
    selected = [index * bars_per_date + bars_per_date - 1 for index in range(dates)]
    observations: list[dict[str, object]] = []
    for observation_id, position in enumerate(selected):
        bar = bars.iloc[position]
        decision_utc = pd.Timestamp(bar["ts_event_utc"])
        decision_ny = pd.Timestamp(bar["ts_event_ny"])
        observations.append(
            {
                "observation_id": observation_id,
                "decision_timestamp_utc": decision_utc,
                "decision_timestamp_ny": decision_ny,
                "entry_timestamp_utc": decision_utc + pd.Timedelta(minutes=1),
                "entry_timestamp_ny": decision_ny + pd.Timedelta(minutes=1),
                "trade_date_ny": bar["trade_date_ny"],
                "research_partition": "Development",
                "entry_session": "London",
                "product": "GC",
                "symbol": bar["symbol"],
                "active_symbol": bar["active_symbol"],
                "instrument_id": bar["instrument_id"],
                "continuous_segment_id": bar["continuous_segment_id"],
                "decision_bar_id": int(bar["source_row_id"]),
                "entry_bar_id": int(bar["source_row_id"]) + 1,
                "minutes_to_forced_exit": 60,
            }
        )
    return bars, pd.DataFrame(observations)


class TsayFeatureEngineeringTests(unittest.TestCase):
    def test_narrow_realized_volatility_one_ulp_amendment(self) -> None:
        actual_rv = np.asarray([3.0, 4.0, np.nan], dtype=np.float32)
        one_ulp = np.nextafter(actual_rv[:2], np.float32(np.inf), dtype=np.float32)
        primitives = {
            "atr_20": np.asarray([1.0, 2.0, np.nan], dtype=np.float64),
            "current_range_over_atr": np.asarray([0.5, 0.25, np.nan], dtype=np.float64),
            "realized_volatility_15": actual_rv.astype(np.float64),
        }
        canonical = pd.DataFrame(
            {
                "observation_id": [10, 11, 12],
                "atr_20": np.asarray([1.0, 2.0, np.nan], dtype=np.float32),
                "current_range_over_atr": np.asarray([0.5, 0.25, np.nan], dtype=np.float32),
                "realized_volatility_15": np.r_[one_ulp, np.float32(np.nan)],
            }
        )
        table = _primitive_equality_table(
            primitives,
            np.arange(3, dtype=np.int64),
            canonical,
            np.asarray([10, 11, 12], dtype=np.int64),
        ).set_index("primitive")
        self.assertTrue(bool(table.loc["atr_20", "float32_exact"]))
        self.assertTrue(bool(table.loc["current_range_over_atr", "float32_exact"]))
        self.assertEqual(int(table.loc["realized_volatility_15", "mismatch_count"]), 2)
        self.assertEqual(int(table.loc["realized_volatility_15", "maximum_ulp_distance"]), 1)
        self.assertEqual(table.loc["realized_volatility_15", "status"], "PASS")

        two_ulp = canonical.copy()
        two_ulp.loc[0, "realized_volatility_15"] = np.nextafter(
            one_ulp[0], np.float32(np.inf), dtype=np.float32
        )
        with self.assertRaisesRegex(ValueError, "compatibility gate"):
            _primitive_equality_table(
                primitives,
                np.arange(3, dtype=np.int64),
                two_ulp,
                np.asarray([10, 11, 12], dtype=np.int64),
            )

        wrong_null = canonical.copy()
        wrong_null.loc[2, "realized_volatility_15"] = np.float32(0.0)
        with self.assertRaisesRegex(ValueError, "compatibility gate"):
            _primitive_equality_table(
                primitives,
                np.arange(3, dtype=np.int64),
                wrong_null,
                np.asarray([10, 11, 12], dtype=np.int64),
            )

    def test_shared_ar_arch_kernel_matches_scalar_reference_to_1e10(self) -> None:
        rng = np.random.default_rng(20260824)
        returns = rng.normal(0.0, 0.35, 125)
        closes = 1_900.0 * np.exp(np.r_[0.0, np.cumsum(returns / 10_000.0)])
        reference = reference_ar_arch_from_closes(closes)
        production, reasons = _batched_ar_arch(
            returns,
            np.arange(125, dtype=np.int64) + 1,
            np.asarray([124], dtype=np.int64),
            batch_size=1,
        )
        self.assertEqual(reasons["t01"][0], reference["t01_reason"])
        self.assertEqual(reasons["t02"][0], reference["t02_reason"])
        self.assertAlmostEqual(production["t01"][0], float(reference["t01"]), places=10)
        self.assertAlmostEqual(production["t02"][0], float(reference["t02"]), places=10)

        constant = reference_ar_arch_from_closes(np.full(126, 1_900.0))
        self.assertEqual(constant["t01_reason"], "AR_RANK_FAILURE")
        self.assertEqual(constant["t02_reason"], "AR_RANK_FAILURE")

    def test_tail_features_hand_calculation_zero_atr_and_ties(self) -> None:
        changes = np.arange(-60.0, 60.0)
        outputs, reasons = _batched_tail_features(
            changes,
            np.r_[np.ones(119), 20.0],
            np.arange(120, dtype=np.int64) + 1,
            np.asarray([119], dtype=np.int64),
            batch_size=1,
        )
        expected_loss = np.sort(-changes)[-12:].mean()
        expected_gain = np.sort(changes)[-12:].mean()
        self.assertAlmostEqual(outputs["t05"][0], expected_loss / 20.0)
        self.assertAlmostEqual(
            outputs["t06"][0],
            (expected_gain - expected_loss) / (abs(expected_gain) + abs(expected_loss)),
        )
        threshold = np.sort(np.abs(changes))[107]
        exceed = np.abs(changes) > threshold
        clusters = int(exceed[0]) + int(np.sum(exceed[1:] & ~exceed[:-1]))
        self.assertAlmostEqual(outputs["t09"][0], clusters / exceed.sum())
        self.assertTrue(all(reasons[key][0] == "AVAILABLE" for key in reasons))

        zero_atr, zero_reasons = _batched_tail_features(
            changes,
            np.zeros(120),
            np.arange(120, dtype=np.int64) + 1,
            np.asarray([119], dtype=np.int64),
            batch_size=1,
        )
        self.assertTrue(np.isnan(zero_atr["t05"][0]))
        self.assertEqual(zero_reasons["t05"][0], "NONPOSITIVE_OR_MISSING_ATR")
        tied, tied_reasons = _batched_tail_features(
            np.ones(120),
            np.ones(120),
            np.arange(120, dtype=np.int64) + 1,
            np.asarray([119], dtype=np.int64),
            batch_size=1,
        )
        self.assertTrue(np.isnan(tied["t09"][0]))
        self.assertEqual(tied_reasons["t09"][0], "TIED_THRESHOLD_NO_EXCEEDANCE")

    def test_clock_profile_is_strictly_prior_and_validation_is_frozen(self) -> None:
        development_dates = pd.date_range("2023-01-01", periods=31, freq="D")
        dates = np.repeat(development_dates, 10)
        raw = np.tile(np.arange(10, dtype=np.float64), 31)
        validation_raw = np.asarray([100.0, 200.0])
        all_raw = np.r_[raw, validation_raw]
        all_dates = pd.DatetimeIndex(
            np.r_[dates.to_numpy(), np.asarray(["2024-01-02", "2024-01-03"], dtype="datetime64[D]")]
        )
        adjusted, reference = _clock_adjusted_series(
            all_raw,
            all_dates,
            np.zeros(len(all_raw), dtype=np.int16),
            name="synthetic",
        )
        self.assertTrue(np.isnan(adjusted[:300]).all())
        expected = (raw[300] - raw[:300].mean()) / raw[:300].std(ddof=1)
        self.assertAlmostEqual(adjusted[300], expected)
        frozen_mean = raw.mean()
        frozen_std = raw.std(ddof=1)
        self.assertAlmostEqual(adjusted[-2], (100.0 - frozen_mean) / frozen_std)
        self.assertAlmostEqual(adjusted[-1], (200.0 - frozen_mean) / frozen_std)
        frozen = reference[reference["reference_scope"].eq("VALIDATION_ALL_DEVELOPMENT_FROZEN")]
        self.assertEqual(int(frozen.iloc[0]["prior_dates"]), 31)

    def test_full_builder_hand_features_alignment_and_primitive_equality(self) -> None:
        bars, observations = _fixture()
        run_id, run_position, _ = build_continuity_run_id(bars)
        primitive = _canonical_primitives(bars, run_id, run_position)
        positions = observations["decision_bar_id"].to_numpy(dtype=np.int64)
        canonical = observations.loc[:, ["observation_id"]].copy()
        for name, values in primitive.items():
            canonical[name] = values[positions].astype(np.float32)
        result = build_tsay_feature_matrix(
            bars,
            observations,
            canonical_primitives=canonical,
            require_canonical_primitive_equality=True,
            batch_size=7,
        )
        self.assertEqual(len(result.matrix), len(observations))
        self.assertTrue(result.primitive_equality["status"].eq("PASS").all())
        self.assertTrue(result.construction_audit["passed"].all())
        self.assertEqual(
            result.matrix["observation_id"].tolist(), observations["observation_id"].tolist()
        )
        for name in TSAY_FEATURE_COLUMNS:
            self.assertEqual(str(result.matrix[name].dtype), "float32")
        last = result.matrix.iloc[-1]
        self.assertTrue(np.isfinite(last["tsay_one_minute_close_zero_change_fraction_60"]))
        self.assertTrue(np.isfinite(last["tsay_loss_es_120_atr"]))
        self.assertTrue(np.isfinite(last["tsay_es_tail_balance_120"]))
        self.assertTrue(np.isfinite(last["tsay_volume_lead_return_impulse_120_l5"]))
        self.assertTrue(np.isfinite(last["tsay_range_state_lead_absreturn_impulse_120_l5"]))

    def test_mutation_row_order_batching_and_observation_partition_are_deterministic(self) -> None:
        bars, observations = _fixture(dates=3)
        selected = observations.iloc[[1]].copy()
        baseline = build_tsay_feature_matrix(bars, selected, batch_size=1).matrix
        mutated = bars.copy()
        after = mutated["source_row_id"] > int(selected.iloc[0]["decision_bar_id"])
        mutated.loc[after, "close"] += 100.0
        causal = build_tsay_feature_matrix(mutated, selected, batch_size=19).matrix
        pd.testing.assert_frame_equal(baseline, causal, check_exact=True)

        shuffled = build_tsay_feature_matrix(
            bars.sample(frac=1.0, random_state=1),
            observations.sample(frac=1.0, random_state=2),
            batch_size=2,
        ).matrix
        full = build_tsay_feature_matrix(bars, observations, batch_size=11).matrix
        pd.testing.assert_frame_equal(full, shuffled, check_exact=True)
        left = build_tsay_feature_matrix(bars, observations.iloc[:1], batch_size=1).matrix
        right = build_tsay_feature_matrix(bars, observations.iloc[1:], batch_size=9).matrix
        pd.testing.assert_frame_equal(
            full,
            pd.concat([left, right], ignore_index=True),
            check_exact=True,
        )

    def test_date_gap_contract_and_tradability_boundaries_reset_candidate_history(self) -> None:
        bars, observations = _fixture(dates=2)
        result = build_tsay_feature_matrix(bars, observations, batch_size=2)
        self.assertTrue(
            np.isnan(result.matrix["tsay_one_minute_close_zero_change_fraction_60"]).eq(False).all()
        )
        self.assertTrue(
            (result.observation_audit["bars_since_candidate_date_run_start"] == 149).all()
        )
        boundary = bars.copy()
        boundary.loc[100, "tradable_research_flag"] = False
        selected = observations.iloc[[0]].copy()
        reset = build_tsay_feature_matrix(boundary, selected, batch_size=1)
        self.assertLess(int(reset.observation_audit.iloc[0]["bars_since_continuity_start"]), 60)
        self.assertTrue(
            np.isnan(reset.matrix.iloc[0]["tsay_one_minute_close_zero_change_fraction_60"])
        )

    def test_missing_pair_and_forbidden_outcome_namespace_fail_closed(self) -> None:
        bars, observations = _fixture()
        missing_pair = bars.drop(index=bars.index[-20]).reset_index(drop=True)
        unavailable = build_tsay_feature_matrix(missing_pair, observations.iloc[[-1]])
        self.assertTrue(
            np.isnan(unavailable.matrix.iloc[0]["tsay_volume_lead_return_impulse_120_l5"])
        )
        self.assertTrue(
            np.isnan(unavailable.matrix.iloc[0]["tsay_range_state_lead_absreturn_impulse_120_l5"])
        )
        forbidden = bars.assign(future_return=0.0)
        with self.assertRaisesRegex(ValueError, "forbidden_source"):
            build_tsay_feature_matrix(forbidden, observations)
        mismapped = observations.copy()
        mismapped.loc[0, "decision_timestamp_utc"] += pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "map identically"):
            build_tsay_feature_matrix(bars, mismapped)


if __name__ == "__main__":
    unittest.main()
