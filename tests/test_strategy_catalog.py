"""Tests for the ~200-strategy catalog.

Confirm the archetype builders behave as specified, the catalog is well-formed
(unique names, ternary signals, documented metadata), thresholds are computed on
Development only, and the near-dead filter and benchmarks behave correctly.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.statistical_research.strategy_catalog import (
    CATALOG_FEATURES,
    _mk_cross,
    _mk_fade,
    _mk_follow,
    _mk_gate,
    _mk_sign,
    build_catalog,
    catalog_library,
    feature_thresholds,
)
from src.statistical_research.strategy_lab import StrategyLabConfig


def _synthetic_universe(n=800, seed=0):
    rng = np.random.default_rng(seed)
    data = {feature: rng.normal(0, 1, n) for feature in CATALOG_FEATURES}
    # bounded features into [0, 1] so quantiles are well-defined
    for feature in (
        "session_range_position",
        "rolling_range_position_15",
        "rolling_range_position_60",
        "fraction_above_vwap_15",
        "efficiency_ratio_15",
        "efficiency_ratio_30",
        "efficiency_ratio_60",
        "ols_r_squared_30",
        "body_to_range",
        "upper_wick_to_range",
        "lower_wick_to_range",
    ):
        data[feature] = rng.uniform(0, 1, n)
    frame = pd.DataFrame(data)
    frame["research_partition"] = rng.choice(["Development", "Validation"], size=n, p=[0.7, 0.3])
    frame["entry_session"] = rng.choice(["London", "New York"], size=n)
    return frame


class ArchetypeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = StrategyLabConfig()

    def test_follow_fires_above_and_below(self):
        frame = pd.DataFrame({"x": [3.0, -3.0, 0.0]})
        out = _mk_follow("x", -1.0, 1.0)(frame, self.cfg)
        np.testing.assert_array_equal(out, np.array([1, -1, 0], dtype=np.int8))

    def test_fade_is_contrarian(self):
        frame = pd.DataFrame({"x": [-3.0, 3.0, 0.0]})
        out = _mk_fade("x", -1.0, 1.0)(frame, self.cfg)
        np.testing.assert_array_equal(out, np.array([1, -1, 0], dtype=np.int8))

    def test_sign_uses_level(self):
        frame = pd.DataFrame({"x": [0.6, 0.4, 0.5]})
        out = _mk_sign("x", level=0.5)(frame, self.cfg)
        np.testing.assert_array_equal(out, np.array([1, -1, 0], dtype=np.int8))

    def test_cross_direction(self):
        frame = pd.DataFrame({"a": [2.0, 0.0], "b": [1.0, 1.0]})
        out = _mk_cross("a", "b")(frame, self.cfg)
        np.testing.assert_array_equal(out, np.array([1, -1], dtype=np.int8))

    def test_gate_zeroes_when_regime_fails(self):
        frame = pd.DataFrame({"d": [1.0, 1.0], "g": [5.0, 0.0]})
        base = _mk_sign("d")
        gated = _mk_gate(base, "g", "gt", 1.0)(frame, self.cfg)
        np.testing.assert_array_equal(gated, np.array([1, 0], dtype=np.int8))

    def test_nan_does_not_fire(self):
        frame = pd.DataFrame({"x": [np.nan, 3.0]})
        out = _mk_follow("x", -1.0, 1.0)(frame, self.cfg)
        np.testing.assert_array_equal(out, np.array([0, 1], dtype=np.int8))


class ThresholdTests(unittest.TestCase):
    def test_thresholds_use_development_only(self):
        n = 600
        dev = pd.DataFrame({f: np.linspace(0, 1, n) for f in CATALOG_FEATURES})
        dev["research_partition"] = "Development"
        val = pd.DataFrame({f: np.linspace(100, 101, n) for f in CATALOG_FEATURES})
        val["research_partition"] = "Validation"
        universe = pd.concat([dev, val], ignore_index=True)
        th = feature_thresholds(universe)
        # Development spans [0, 1]; Validation's [100, 101] must not leak in.
        self.assertLess(th["return_30m_atr"]["p90"], 2.0)
        self.assertGreater(th["return_30m_atr"]["p10"], -0.01)


class CatalogTests(unittest.TestCase):
    def test_catalog_is_well_formed(self):
        universe = _synthetic_universe()
        specs = build_catalog(feature_thresholds(universe))
        self.assertGreater(len(specs), 150)
        names = [s.name for s in specs]
        self.assertEqual(len(names), len(set(names)))  # unique
        cfg = StrategyLabConfig()
        for spec in specs:
            out = spec.signal(universe, cfg)
            self.assertEqual(len(out), len(universe))
            self.assertTrue(set(np.unique(out)).issubset({-1, 0, 1}), msg=spec.name)

    def test_trials_carry_documentation_metadata(self):
        universe = _synthetic_universe()
        specs = build_catalog(feature_thresholds(universe))
        trials = [s for s in specs if s.family != "benchmark"]
        for spec in trials:
            self.assertTrue(spec.looks_for, msg=f"{spec.name} missing looks_for")
            self.assertTrue(spec.parameters, msg=f"{spec.name} missing parameters")

    def test_catalog_library_returns_kept_and_dropped(self):
        universe = _synthetic_universe()
        kept, dropped = catalog_library(universe)
        names = {s.name for s in kept}
        self.assertIn("always_long", names)
        self.assertIn("always_short", names)
        self.assertIsInstance(dropped, list)
        # benchmarks are never dropped
        self.assertNotIn("always_long", dropped)


if __name__ == "__main__":
    unittest.main()
