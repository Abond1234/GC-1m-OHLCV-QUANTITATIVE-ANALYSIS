"""Edge-context adapter fidelity against the frozen research builder.

The app computes the four ADVANCING features with the research module's own
low-level helpers; this test proves the adapter reproduces
``build_scalar_feature_matrix`` exactly (float32-for-float32) on the research
suite's own synthetic fixture, so the app can never drift from the frozen
arithmetic. Also covers the missing-cause ladder and session metadata.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.app.datalayer.edge_context import (
    EDGE_FEATURES,
    VERDICT_CAPTION,
    compute_edge_context,
)

try:
    from tests.test_fes_project1_features import _bars, _build
except (ImportError, unittest.SkipTest):  # pragma: no cover - fixtures need pytest
    _bars = None

_FEATURE_KEYS = [f.key for f in EDGE_FEATURES]


@unittest.skipUnless(_bars is not None, "research fixture helpers unavailable")
class GoldenFidelityTests(unittest.TestCase):
    def test_adapter_matches_frozen_builder_exactly(self):
        bars = _bars(220)
        positions = [95, 130, 180, 219]
        matrix = _build(bars, positions).matrix
        context = compute_edge_context(bars, trade_date=bars["trade_date_ny"].iloc[0])
        for key in _FEATURE_KEYS:
            for obs_row, position in enumerate(positions):
                expected = matrix[key].to_numpy()[obs_row]
                got = context.values[key][position]
                if np.isnan(expected):
                    self.assertTrue(np.isnan(got), f"{key}@{position}: expected NaN, got {got}")
                else:
                    self.assertEqual(
                        np.float32(expected),
                        np.float32(got),
                        f"{key}@{position}: {expected} != {got}",
                    )

    def test_insufficient_history_reports_cause_not_number(self):
        bars = _bars(220)
        context = compute_edge_context(bars, trade_date=bars["trade_date_ny"].iloc[0])
        for key, info in zip(_FEATURE_KEYS, EDGE_FEATURES, strict=True):
            early = info.lookback - 2  # one bar short of the requirement
            self.assertTrue(np.isnan(context.values[key][early]))
            self.assertEqual(context.causes[key][early], "INSUFFICIENT_LEGAL_HISTORY")
            late = 219
            self.assertEqual(context.causes[key][late], "")

    def test_flat_price_hits_degenerate_ladder(self):
        bars = _bars(220, flat_price=True)
        context = compute_edge_context(bars, trade_date=bars["trade_date_ny"].iloc[0])
        # Flat closes give zero-variance returns: F07's Pearson denominators die.
        self.assertEqual(context.causes["return_acf_energy_60"][219], "DEGENERATE_STATISTIC")
        self.assertTrue(np.isnan(context.values["return_acf_energy_60"][219]))


class MetadataTests(unittest.TestCase):
    def test_session_split_matches_the_findings_report(self):
        by_session = {}
        for info in EDGE_FEATURES:
            by_session.setdefault(info.session, []).append(info.key)
        self.assertEqual(
            sorted(by_session["London"]),
            ["range_volume_spearman_30", "return_acf_energy_60"],
        )
        self.assertEqual(
            sorted(by_session["New York"]),
            ["lagged_volume_return_spearman_30", "volume_profile_slope_30"],
        )

    def test_caption_carries_the_frozen_verdicts(self):
        self.assertIn("PREDICTIVE_ONLY_NOT_DIRECTIONAL", VERDICT_CAPTION)
        self.assertIn("FROZEN_NO_POLICY", VERDICT_CAPTION)
        self.assertIn("nothing here is a trade signal", VERDICT_CAPTION)


if __name__ == "__main__":
    unittest.main()
