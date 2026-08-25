"""Tests that the generated notebook stops exactly at Stage 4."""

from __future__ import annotations

import unittest

from scripts.build_tsay_feature_research_notebook import build_tsay_notebook


class TsayNotebookBuilderTests(unittest.TestCase):
    def test_builder_declares_stage4_only_and_narrow_outcome_authorization(self) -> None:
        notebook = build_tsay_notebook()
        self.assertEqual(notebook.metadata["tsay"]["completed_stage"], 4)
        self.assertTrue(notebook.metadata["tsay"]["stage2_present"])
        self.assertTrue(notebook.metadata["tsay"]["stage3_present"])
        self.assertTrue(notebook.metadata["tsay"]["development_outcomes_authorized"])
        self.assertFalse(notebook.metadata["tsay"]["validation_outcomes_authorized"])
        self.assertTrue(notebook.metadata["tsay"]["stage4_present"])
        source = "\n".join(cell.source for cell in notebook.cells)
        self.assertIn("Stage 4 stops here", source)
        self.assertIn("run_stage4", source)
        self.assertIn("development_continuous_model_evidence.parquet", source)
        self.assertIn("Validation labels/performance", source)
        self.assertNotIn("run_stage5", source)

    def test_checkpoint_cell_prints_status_and_separate_verdict(self) -> None:
        notebook = build_tsay_notebook()
        source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        self.assertIn("STATUS:", source)
        self.assertIn("RESEARCH_VERDICT:", source)
        self.assertIn("STAGE_5_AUTHORIZED:", source)
        self.assertIn("POLICY_AUTHORIZED:", source)


if __name__ == "__main__":
    unittest.main()
