"""Configurable cross-asset comparison: split modes, normalized overlay, sync."""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

from src.app.datalayer.paths import research_bars_path

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None
_HAS_BARS = research_bars_path().exists()


@unittest.skipUnless(_HAS_QT and _HAS_PG and _HAS_BARS, "GUI deps or bars parquet missing")
class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.datalayer.instruments import load_instrument_bars
        from src.app.ui import theme
        from src.app.ui.main_window import MainWindow, load_app_data

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        theme.apply(cls.app, "dark")
        cls.win = MainWindow(data=load_app_data(date_floor=pd.Timestamp("2024-12-24")))
        cls.win.show()
        # Inject MGC synchronously (avoid the async worker load in the test).
        cls.win._compare_bars["MGC"] = load_instrument_bars(
            "MGC", date_floor=pd.Timestamp("2024-12-24")
        )
        cls.win._compare_symbol = "MGC"

    def setUp(self):
        self.win._compare_symbol = "MGC"  # each test starts with MGC selected

    def _apply(self, mode):
        self.win._compare_mode = mode
        self.win._apply_compare_mode()
        for _ in range(3):
            self.app.processEvents()

    def test_horizontal_split_stacks_and_time_syncs(self):
        from PySide6 import QtCore

        self._apply("Horizontal")
        self.assertTrue(self.win.compare_chart.isVisible())
        self.assertEqual(self.win._chart_col.orientation(), QtCore.Qt.Vertical)
        linked = self.win.compare_chart._price.vb.state["linkedViews"][0]
        self.assertIsNotNone(linked)  # X-linked to the primary (time sync)

    def test_vertical_split_is_side_by_side(self):
        from PySide6 import QtCore

        self._apply("Vertical")
        self.assertTrue(self.win.compare_chart.isVisible())
        self.assertEqual(self.win._chart_col.orientation(), QtCore.Qt.Horizontal)

    def test_normalized_overlay_states_method_and_keeps_primary_scale(self):
        self._apply("Normalized")
        self.assertFalse(self.win.compare_chart.isVisible())
        item = self.win.chart._compare_item
        self.assertIsNotNone(item)
        self.assertIn("rebased", item.name())
        # Rebasing keeps the overlay inside the primary's price band (no distortion).
        prim = self.win._bars
        lo, hi = self.win._view_start, self.win._view_end
        gc_lo, gc_hi = float(prim.low[lo : hi + 1].min()), float(prim.high[lo : hi + 1].max())
        y = np.asarray(item.yData, dtype=float)
        self.assertGreater(y.min(), gc_lo * 0.5)
        self.assertLess(y.max(), gc_hi * 1.5)

    def test_off_clears_everything(self):
        self._apply("Horizontal")
        self.win._compare_symbol = None
        self.win._apply_compare_mode()
        self.assertFalse(self.win.compare_chart.isVisible())
        self.assertIsNone(self.win.chart._compare_item)

    def test_overlay_popover_lists_off_instruments_and_modes(self):
        self.win._populate_compare_combo()
        labels = [a.text() for a in self.win._overlay_menu.actions() if a.text()]
        for expected in ("Off", "Horizontal split", "Vertical split", "Normalized overlay"):
            self.assertIn(expected, labels)
        self.assertTrue(any("MGC" in t or "Micro" in t for t in labels))
        self.assertFalse(any("GC -" in t and "Micro" not in t for t in labels))  # primary excluded
        self.assertIn("MGC", self.win.overlay_btn.text())

    def test_toggling_overlay_never_shifts_primary_chart_state(self):
        x_before = self.win.chart._price.vb.viewRange()[0]
        self._apply("Horizontal")
        self._apply("Normalized")
        self.win._compare_symbol = None
        self.win._apply_compare_mode()
        x_after = self.win.chart._price.vb.viewRange()[0]
        self.assertAlmostEqual(x_after[0], x_before[0], places=9)
        self.assertAlmostEqual(x_after[1], x_before[1], places=9)


if __name__ == "__main__":
    unittest.main()
