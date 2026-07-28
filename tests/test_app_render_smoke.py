"""End-to-end offscreen render smoke test.

Builds the real MainWindow against a narrow bar load and writes screenshots,
so a regression that breaks the whole app (not just one widget) is caught.
Skips without the GUI deps or the local parquet artifacts.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.app.datalayer.paths import research_bars_path

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None
_HAS_BARS = research_bars_path().exists()


@unittest.skipUnless(_HAS_QT and _HAS_PG and _HAS_BARS, "GUI deps or bars parquet missing")
class RenderSmokeTests(unittest.TestCase):
    def test_render_all_produces_screenshots(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from scripts.render_screens import render_all

        with tempfile.TemporaryDirectory() as tmp:
            # A tight window (last days of Validation) keeps the load fast.
            saved = render_all(out_dir=Path(tmp), date_floor=pd.Timestamp("2024-12-24"))
            self.assertTrue(saved, "no screenshots were produced")
            for path in saved:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)  # a real PNG, not empty


if __name__ == "__main__":
    unittest.main()
