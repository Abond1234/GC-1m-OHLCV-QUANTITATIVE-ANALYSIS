from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.research.poi_context_event_study import (
    PoiContextResearchConfig,
    build_feature_study_summary,
    build_stop_policy_opportunities,
)


class PoiContextEventStudyTests(unittest.TestCase):
    def test_continuation_reversal_side_mapping_in_stop_opportunities(self):
        # Mapping is tested through a minimal context and a minimal full bar contract.
        from tests.test_poi_context_features import _bars, _signals
        from src.features.poi_context_features import build_true_poi_context_frame
        context = build_true_poi_context_frame(_signals(), _bars())
        continuation = build_stop_policy_opportunities(context, _bars(), hypothesis="continuation", stop_model="poi_invalidation_1tick")
        reversal = build_stop_policy_opportunities(context, _bars(), hypothesis="reversal", stop_model="poi_invalidation_1tick")
        self.assertEqual(continuation.loc[0, "trade_side"], "long")
        self.assertEqual(reversal.loc[0, "trade_side"], "short")

    def test_development_bins_are_reused_out_of_sample(self):
        n = 100
        context = pd.DataFrame({
            "true_retest_id": [f"R{i}" for i in range(n)],
            "true_poi_id": [f"P{i}" for i in range(n)],
            "trade_date_ny": pd.date_range("2023-01-01", periods=n),
            "research_partition": ["development"]*50+["validation"]*25+["final_test"]*25,
            "feat_poi_width_atr": np.r_[np.arange(50), np.arange(100,150)],
        })
        labels = pd.concat([
            pd.DataFrame({
                "true_retest_id": context.true_retest_id,
                "hypothesis": hypothesis,
                "trade_side": "long",
                "research_partition": context.research_partition,
                "label_capped_60m_r": np.linspace(-1,1,n),
                "trade_date_ny": context.trade_date_ny,
            }) for hypothesis in ("continuation", "reversal")
        ], ignore_index=True)
        result = build_feature_study_summary(
            context, labels,
            PoiContextResearchConfig(bootstrap_samples=10),
            features=("feat_poi_width_atr",),
        )
        edges = result.groupby("research_partition")["development_bin_edges"].nunique()
        self.assertTrue(edges.eq(1).all())
        self.assertEqual(result["development_bin_edges"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
