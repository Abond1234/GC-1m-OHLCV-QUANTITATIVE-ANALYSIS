"""Synthetic tests for the Section 12 hybrid integration engine."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.research.hybrid_integration import (
    Section12Config,
    build_hybrid_analysis,
    build_hybrid_event_frame,
)

FAST_CONFIG = Section12Config(
    bootstrap_replicates=200,
    min_development_gated_events=100,
    min_validation_gated_events=60,
    min_development_trading_dates=30,
    min_validation_trading_dates=20,
)


def _make_inputs(*, seed: int = 5, gate_effect_r: float = 0.4):
    """POI outcome rows plus a Branch-B candidate table sharing decision bars."""

    rng = np.random.default_rng(seed)
    label_rows, context_rows, candidate_rows = [], [], []
    retest_id = 0
    for partition, n_dates, start in (
        ("development", 60, "2022-01-03"),
        ("validation", 30, "2024-01-02"),
        ("final_test", 20, "2025-01-02"),
    ):
        for date in pd.bdate_range(start, periods=n_dates):
            for event in range(12):
                retest_id += 1
                ts = (date + pd.Timedelta(hours=8, minutes=5 * event)).to_datetime64()
                prediction = rng.normal(0.0, 1.0)
                gate = prediction >= 0.8416  # population 80th percentile
                base = rng.normal(0.0, 1.0)
                for hypothesis, side in (("continuation", "short"), ("reversal", "long")):
                    effect = gate_effect_r if (gate and hypothesis == "continuation") else 0.0
                    label_rows.append(
                        {
                            "true_retest_id": f"R{retest_id}",
                            "trade_date_ny": date.to_datetime64(),
                            "research_partition": partition,
                            "hypothesis": hypothesis,
                            "trade_side": side,
                            "retest_bar_id": retest_id,
                            "label_capped_60m_r": base + effect + rng.normal(0.0, 0.3),
                            "label_capped_60m_valid": True,
                        }
                    )
                context_rows.append(
                    {
                        "true_retest_id": f"R{retest_id}",
                        "retest_ts_event_utc": ts,
                        "feat_execution_session": "New York",
                    }
                )
                candidate_rows.append(
                    {
                        "observation_id": retest_id,
                        "entry_timestamp_utc": ts + np.timedelta64(1, "m"),
                        "entry_session": "New York",
                        "gate_prediction": prediction,
                        "expansion_gate_flag": gate,
                    }
                )
    return (
        pd.DataFrame(label_rows).drop_duplicates(
            subset=["true_retest_id", "hypothesis"], keep="first"
        ),
        pd.DataFrame(context_rows).drop_duplicates("true_retest_id"),
        pd.DataFrame(candidate_rows),
    )


class EventFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.labels, cls.context, cls.candidates = _make_inputs()
        cls.events, cls.summary = build_hybrid_event_frame(
            cls.labels, cls.context, cls.candidates, FAST_CONFIG
        )

    def test_join_matches_decision_bars_exactly(self) -> None:
        self.assertEqual(self.summary["gate_match_rate"], 1.0)
        self.assertEqual(self.summary["valid_outcome_rows_kept"], len(self.labels))

    def test_partitions_are_normalized(self) -> None:
        self.assertEqual(
            set(self.events["research_partition"].unique()),
            {"Development", "Validation", "Final test"},
        )

    def test_offset_timestamps_do_not_match(self) -> None:
        shifted = self.candidates.copy()
        shifted["entry_timestamp_utc"] = shifted["entry_timestamp_utc"] + np.timedelta64(2, "m")
        _, summary = build_hybrid_event_frame(self.labels, self.context, shifted, FAST_CONFIG)
        self.assertEqual(summary["gate_matched_rows"], 0)

    def test_robust_cap_applied(self) -> None:
        labels = self.labels.copy()
        labels.loc[labels.index[0], "label_capped_60m_r"] = 40.0
        events, _ = build_hybrid_event_frame(labels, self.context, self.candidates, FAST_CONFIG)
        self.assertLessEqual(events["robust_r"].abs().max(), FAST_CONFIG.robust_cap_r)


class HybridAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        labels, context, candidates = _make_inputs()
        cls.events, _ = build_hybrid_event_frame(labels, context, candidates, FAST_CONFIG)
        cls.result = build_hybrid_analysis(cls.events, FAST_CONFIG)

    def test_planted_gate_effect_advances_only_that_family(self) -> None:
        verdicts = self.result.verdicts.set_index(["hypothesis", "trade_side"])["verdict"]
        self.assertEqual(verdicts[("continuation", "short")], "HYBRID_ADVANCES")
        self.assertEqual(verdicts[("reversal", "long")], "NO_INCREMENTAL_VALUE")

    def test_null_gate_never_advances(self) -> None:
        labels, context, candidates = _make_inputs(seed=11, gate_effect_r=0.0)
        events, _ = build_hybrid_event_frame(labels, context, candidates, FAST_CONFIG)
        result = build_hybrid_analysis(events, FAST_CONFIG)
        self.assertTrue(result.verdicts["verdict"].eq("NO_INCREMENTAL_VALUE").all())

    def test_final_test_does_not_influence_verdicts(self) -> None:
        mutated = self.events.copy()
        final_rows = mutated["research_partition"] == "Final test"
        mutated.loc[final_rows, "robust_r"] = -5.0
        result = build_hybrid_analysis(mutated, FAST_CONFIG)
        pd.testing.assert_frame_equal(result.verdicts, self.result.verdicts)
        self.assertLess(result.final_test_report["baseline_mean_robust_r"].max(), -4.0)

    def test_quintiles_are_monotone_for_planted_effect(self) -> None:
        q = self.result.quintile_results
        mask = q["hypothesis"].eq("continuation") & q["research_partition"].eq("Validation")
        cell = q.loc[mask].sort_values("prediction_quintile")
        top = cell.iloc[-1]["mean_robust_r"]
        bottom = cell.iloc[0]["mean_robust_r"]
        self.assertGreater(top, bottom)

    def test_all_validation_checks_pass(self) -> None:
        failed = self.result.validation_checks.loc[~self.result.validation_checks["passed"]]
        self.assertTrue(failed.empty, failed.to_string())

    def test_deterministic(self) -> None:
        repeat = build_hybrid_analysis(self.events, FAST_CONFIG)
        pd.testing.assert_frame_equal(repeat.verdicts, self.result.verdicts)

    def test_unknown_partition_raises(self) -> None:
        broken = self.events.copy()
        broken.loc[broken.index[0], "research_partition"] = "Weird"
        with self.assertRaises(ValueError):
            build_hybrid_analysis(broken, FAST_CONFIG)


if __name__ == "__main__":
    unittest.main()
