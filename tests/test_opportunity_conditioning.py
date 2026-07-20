"""Synthetic tests for the Section 12B opportunity-conditioning engine."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.research.opportunity_conditioning import (
    Section12BConfig,
    _date_block_bootstrap,
    _final_test_report,
    assign_quintiles,
    evaluate_h1_sizing,
    evaluate_h2_exits,
    evaluate_h3_suppression,
    fit_development_quintiles,
)

FAST_CONFIG = Section12BConfig(bootstrap_replicates=200)
BUCKET_PREDICTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _family_events(outcome_fn, *, per_bucket: int, risk_ticks: float = 1e9) -> pd.DataFrame:
    """Deterministic family frame: identical bucket structure on every date.

    ``outcome_fn(bucket, slot) -> (r60, r120, r240)`` fixes each event's
    outcomes, so every trading date is an exact copy and the date-block
    bootstrap distribution is degenerate at the planted effect.
    """

    partitions = {
        "Development": pd.date_range("2022-01-03", periods=210, freq="B"),
        "Validation": pd.date_range("2024-01-02", periods=130, freq="B"),
        "Final test": pd.date_range("2025-01-02", periods=40, freq="B"),
    }
    rows = []
    for partition, dates in partitions.items():
        for date in dates:
            minute = 0
            for bucket in range(5):
                for slot in range(per_bucket):
                    r60, r120, r240 = outcome_fn(bucket, slot)
                    rows.append(
                        {
                            "true_retest_id": f"{partition[:3]}-{date.date()}-{bucket}-{slot}",
                            "trade_date_ny": date.to_datetime64(),
                            "retest_ts_event_utc": (
                                date + pd.Timedelta(hours=14, minutes=minute)
                            ).to_datetime64(),
                            "research_partition": partition,
                            "session": "New York",
                            "gate_prediction": BUCKET_PREDICTIONS[bucket],
                            "risk_ticks": risk_ticks,
                            "r_60m": r60,
                            "valid_60m": True,
                            "r_120m": r120,
                            "valid_120m": True,
                            "r_240m": r240,
                            "valid_240m": True,
                            "robust_r": r60,
                        }
                    )
                    minute += 1
    return pd.DataFrame.from_records(rows)


def _buckets(events: pd.DataFrame) -> np.ndarray:
    edges = fit_development_quintiles(events, FAST_CONFIG)
    return assign_quintiles(events, edges)


class QuintileFittingTests(unittest.TestCase):
    def test_edges_come_from_development_only(self) -> None:
        events = pd.DataFrame(
            {
                "research_partition": ["Development"] * 100 + ["Validation"] * 50,
                "gate_prediction": np.concatenate(
                    [np.linspace(0.0, 1.0, 100), np.linspace(100.0, 101.0, 50)]
                ),
            }
        )
        edges = fit_development_quintiles(events, FAST_CONFIG)
        self.assertTrue((edges <= 1.0).all(), edges)
        buckets = assign_quintiles(events, edges)
        self.assertTrue((buckets[100:] == 4).all())


class H1SizingTests(unittest.TestCase):
    """Bucket 0 carries all dispersion at zero extra mean; buckets share mass.

    Constant per-date values: two of (+5.5, -4.5) in bucket 0 and 0.5
    elsewhere give mean 0.5, MAD 1, ratio 0.5.  The proportional ladder
    downweights bucket 0 to a 1.071 ratio (+0.571 effect, zero mean shift);
    the inverse ladder upweights it and degrades the ratio.
    """

    @staticmethod
    def _outcomes(bucket: int, slot: int):
        r60 = (5.5 if slot == 0 else -4.5) if bucket == 0 else 0.5
        return r60, r60, r60

    def test_proportional_advances_and_inverse_fails(self) -> None:
        events = _family_events(self._outcomes, per_bucket=2)
        _, effects, detail = evaluate_h1_sizing(events, _buckets(events), FAST_CONFIG)
        self.assertEqual(detail["advancing_schemes"], ["proportional"])
        self.assertTrue(detail["advances"])
        row = effects.loc[
            effects["sizing_scheme"].eq("proportional") & effects["cost_scenario"].eq("base")
        ].iloc[0]
        np.testing.assert_allclose(row["development_ratio_effect"], 0.5714, atol=1e-3)
        self.assertGreater(row["development_ci_low"], 0.0)
        np.testing.assert_allclose(row["development_mean_shift_r"], 0.0, atol=1e-9)
        inverse = effects.loc[
            effects["sizing_scheme"].eq("inverse") & effects["cost_scenario"].eq("base")
        ].iloc[0]
        self.assertLess(inverse["development_ratio_effect"], 0.0)
        self.assertFalse(inverse["criteria_pass"])

    def test_cost_arithmetic_subtracts_ticks_over_risk(self) -> None:
        events = _family_events(self._outcomes, per_bucket=2, risk_ticks=26.0)
        results, _, _ = evaluate_h1_sizing(events, _buckets(events), FAST_CONFIG)
        constant = results.loc[
            results["sizing_scheme"].eq("constant")
            & results["research_partition"].eq("Development")
        ].set_index("cost_scenario")["mean_net_r"]
        np.testing.assert_allclose(constant["frictionless"] - constant["base"], 2.6 / 26.0)
        np.testing.assert_allclose(constant["frictionless"] - constant["pessimistic"], 4.6 / 26.0)

    def test_material_mean_shift_blocks_advancement(self) -> None:
        def shifted(bucket: int, slot: int):
            r60 = (6.5 if slot == 0 else -4.5) if bucket == 0 else 0.5
            return r60, r60, r60

        events = _family_events(shifted, per_bucket=2)
        _, effects, detail = evaluate_h1_sizing(events, _buckets(events), FAST_CONFIG)
        proportional = effects.loc[
            effects["sizing_scheme"].eq("proportional") & effects["cost_scenario"].eq("base")
        ].iloc[0]
        self.assertFalse(proportional["mean_shift_within_materiality"])
        self.assertNotIn("proportional", detail["advancing_schemes"])


class H2ExitTests(unittest.TestCase):
    """Horizon 60m is best unconditionally, but bucket 4 strongly prefers 240m."""

    @staticmethod
    def _outcomes(bucket: int, slot: int):
        del slot
        return 0.2, 0.0, (1.0 if bucket == 4 else -0.5)

    def test_conditional_mapping_advances(self) -> None:
        events = _family_events(self._outcomes, per_bucket=10)
        mapping, results, detail = evaluate_h2_exits(events, _buckets(events), FAST_CONFIG)
        self.assertEqual(detail["unconditional_best_horizon"], 60)
        chosen = mapping.set_index("prediction_quintile")["chosen_horizon_minutes"]
        self.assertEqual(int(chosen.loc[4]), 240)
        self.assertTrue((chosen.drop(4) == 60).all())
        development = results.loc[results["research_partition"].eq("Development")].iloc[0]
        np.testing.assert_allclose(development["conditioning_effect_r"], 0.16, atol=1e-9)
        self.assertGreater(development["development_ci_low"], 0.0)
        self.assertTrue(detail["advances"])

    def test_tie_resolves_to_shortest_horizon(self) -> None:
        events = _family_events(lambda b, s: (0.3, 0.3, 0.3), per_bucket=2)
        mapping, _, detail = evaluate_h2_exits(events, _buckets(events), FAST_CONFIG)
        self.assertEqual(detail["unconditional_best_horizon"], 60)
        self.assertTrue((mapping["chosen_horizon_minutes"] == 60).all())
        self.assertFalse(detail["advances"])


class H3SuppressionTests(unittest.TestCase):
    """Bucket 0 is pure stop-outs; suppression lifts both co-primary effects."""

    @staticmethod
    def _outcomes(bucket: int, slot: int):
        r60 = -1.0 if (bucket == 0 or slot >= 6) else 0.5
        return r60, r60, r60

    def test_suppression_advances_on_both_effects(self) -> None:
        events = _family_events(self._outcomes, per_bucket=10)
        results, detail = evaluate_h3_suppression(events, _buckets(events), FAST_CONFIG)
        development = results.loc[results["research_partition"].eq("Development")].iloc[0]
        np.testing.assert_allclose(development["median_improvement_r"], 1.5)
        np.testing.assert_allclose(development["stop_rate_reduction"], 0.52 - 0.4, atol=1e-9)
        self.assertGreater(detail["development_median_ci"][0], 0.0)
        self.assertGreater(detail["development_stop_rate_ci"][0], 0.0)
        self.assertTrue(detail["advances"])

    def test_stop_rate_regression_vetoes_advancement(self) -> None:
        def no_stop_bottom(bucket: int, slot: int):
            r60 = 0.1 if bucket == 0 else (-1.0 if slot >= 6 else 0.5)
            return r60, r60, r60

        events = _family_events(no_stop_bottom, per_bucket=10)
        results, detail = evaluate_h3_suppression(events, _buckets(events), FAST_CONFIG)
        development = results.loc[results["research_partition"].eq("Development")].iloc[0]
        self.assertGreater(development["median_improvement_r"], 0.0)
        self.assertLess(development["stop_rate_reduction"], 0.0)
        self.assertFalse(detail["advances"])


class ReportingAndBootstrapTests(unittest.TestCase):
    def test_final_test_report_has_no_verdict_column(self) -> None:
        events = _family_events(H2ExitTests._outcomes, per_bucket=10)
        buckets = _buckets(events)
        mapping, _, detail = evaluate_h2_exits(events, buckets, FAST_CONFIG)
        report = _final_test_report(events, buckets, mapping, detail, FAST_CONFIG)
        self.assertNotIn("verdict", report.columns)
        self.assertEqual(set(report["hypothesis_id"]), {"H1_sizing", "H2_exits", "H3_suppression"})
        h2_row = report.loc[report["hypothesis_id"].eq("H2_exits")].iloc[0]
        np.testing.assert_allclose(h2_row["final_test_effect"], 0.16, atol=1e-9)

    def test_bootstrap_is_deterministic_by_seed(self) -> None:
        rng = np.random.default_rng(1)
        dates = np.repeat(pd.date_range("2022-01-03", periods=50, freq="B").to_numpy(), 4)
        values = rng.normal(0.1, 1.0, size=len(dates))

        def statistic(idx: np.ndarray) -> float:
            return float(values[idx].mean())

        first = _date_block_bootstrap(dates, statistic, seed=7, replicates=200, confidence=0.95)
        second = _date_block_bootstrap(dates, statistic, seed=7, replicates=200, confidence=0.95)
        self.assertEqual(first, second)
        self.assertTrue(np.isfinite(first[0]) and first[0] < first[1])


if __name__ == "__main__":
    unittest.main()
