"""Strategy inspector: single-click overview, double-click detail, Escape returns."""

from __future__ import annotations

import importlib.util
import os
import re
import unittest

import pandas as pd

from src.app.datalayer.paths import research_bars_path

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None
_HAS_BARS = research_bars_path().exists()


@unittest.skipUnless(_HAS_QT and _HAS_PG and _HAS_BARS, "GUI deps or bars parquet missing")
class InspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.ui import theme
        from src.app.ui.main_window import MainWindow, load_app_data

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        theme.apply(cls.app, "dark")
        cls.win = MainWindow(data=load_app_data(date_floor=pd.Timestamp("2024-12-24")))
        cls.win.show()
        spec = next(
            s for s in cls.win._data["replay"].list_strategies() if "long" in s.name.lower()
        )
        cls.log = cls.win._data["replay"].replay(spec, None)
        cls.win._on_replayed(cls.log)
        cls._drain()

    @classmethod
    def _drain(cls):
        for _ in range(4):
            cls.app.processEvents()

    def test_single_click_shows_overview_without_focusing(self):
        self.win._focused_result = None
        self.win.trade_table.clearSelection()
        self.win.trade_table.selectRow(1)  # a real selection change fires the slot
        self._drain()
        text = re.sub("<[^>]+>", "", self.win.overview.text())
        self.assertIn("R", text)  # overview populated
        self.assertRegex(text, r"held \d+ min")
        self.assertIsNone(self.win._focused_result)  # chart not consumed

    def test_double_click_opens_full_detail(self):
        self.win.trade_table.selectRow(0)
        self.win._activate_selected_row()
        self._drain()
        self.assertIsNotNone(self.win._focused_result)

    def test_escape_returns_to_full_chart_keeping_state(self):
        from PySide6 import QtCore, QtGui

        self.win.trade_table.selectRow(0)
        self.win._activate_selected_row()
        self._drain()
        self.win.keyPressEvent(
            QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier)
        )
        self.assertIsNone(self.win._focused_result)
        self.assertEqual(self.win.overview.text(), self.win._inspector_hint)
        self.assertIsNotNone(self.win._log)  # replay state preserved

    def test_escape_cancels_an_armed_draw_tool(self):
        self.win._arm_draw_tool("hline")
        self.assertEqual(self.win.chart._draw_mode, "hline")
        from PySide6 import QtCore, QtGui

        self.win.keyPressEvent(
            QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier)
        )
        self.assertIsNone(self.win.chart._draw_mode)


if __name__ == "__main__":
    unittest.main()
