"""The five-workspace shell: config controls live in their workspace, not the toolbar."""

from __future__ import annotations

import importlib.util
import os
import unittest

import pandas as pd

from src.app.datalayer.paths import research_bars_path

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None
_HAS_BARS = research_bars_path().exists()


def _in_workspace(win, widget, index) -> bool:
    page = win._workspace_stack.widget(index)
    node = widget
    while node is not None:
        if node is page:
            return True
        node = node.parentWidget()
    return False


@unittest.skipUnless(_HAS_QT and _HAS_PG and _HAS_BARS, "GUI deps or bars parquet missing")
class WorkspaceShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.ui import theme
        from src.app.ui.main_window import MainWindow, load_app_data

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        theme.apply(cls.app, "dark")
        cls.win = MainWindow(data=load_app_data(date_floor=pd.Timestamp("2024-12-27")))
        cls.win.show()
        for _ in range(4):
            cls.app.processEvents()

    def test_five_workspaces(self):
        self.assertEqual(self.win._workspace_stack.count(), 5)
        self.assertEqual(len(self.win._ws_actions), 5)

    def test_controls_live_in_their_workspace(self):
        w = self.win
        self.assertTrue(_in_workspace(w, w.strategy_browser, 0))  # Strategy
        self.assertTrue(_in_workspace(w, w._tabs, 0))
        self.assertTrue(_in_workspace(w, w.indicators_panel, 1))  # Indicators
        self.assertTrue(_in_workspace(w, w.edge_panel, 1))
        self.assertTrue(_in_workspace(w, w._draw_buttons["hline"], 2))  # Drawing
        self.assertTrue(_in_workspace(w, w.exit_panel, 3))  # Risk
        self.assertTrue(_in_workspace(w, w.session_panel, 4))  # Prop firm

    def test_switching_and_collapse(self):
        w = self.win
        w._select_workspace(2)
        self.assertEqual(w._workspace_stack.currentIndex(), 2)
        self.assertTrue(w._workspace_panel.isVisible())
        w._select_workspace(2)  # re-click collapses, returning chart width
        self.assertFalse(w._workspace_panel.isVisible())
        w._select_workspace(0)
        self.assertTrue(w._workspace_panel.isVisible())

    def test_toolbar_is_slim(self):
        # The date/timeframe controls remain reachable at the top; the moved
        # controls are no longer direct siblings in the toolbar layout.
        w = self.win
        top = {w._controls.itemAt(i).widget() for i in range(w._controls.count())}
        self.assertIn(w.instrument_combo, top)
        self.assertIn(w.tf_combo, top)
        self.assertNotIn(w.strategy_browser, top)
        self.assertNotIn(w.session_check, top)


if __name__ == "__main__":
    unittest.main()
