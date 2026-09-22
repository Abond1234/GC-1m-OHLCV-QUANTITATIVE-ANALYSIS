"""Chart tools on the left, simulation setup and results in a resizable right panel."""

from __future__ import annotations

import importlib.util
import os
import time
import unittest
from unittest.mock import patch

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
        self.assertEqual(self.win._workspace_stack.count(), 2)
        self.assertEqual([a.text() for a in self.win._ws_actions], ["Indicators", "Drawing"])
        tabs = self.win.simulator_panel.tabs
        self.assertEqual(
            [tabs.tabText(i) for i in range(tabs.count())],
            ["Prop firm", "Risk", "Strategy", "Performance"],
        )

    def test_controls_live_in_their_workspace(self):
        w = self.win
        self.assertTrue(_in_workspace(w, w.indicators_panel, 0))
        self.assertFalse(_in_workspace(w, w.edge_panel, 0))
        self.assertFalse(w.indicators_panel.volume_check.isChecked())
        self.assertTrue(_in_workspace(w, w._draw_buttons["hline"], 1))
        tabs = w.simulator_panel.tabs
        self.assertTrue(tabs.widget(0).isAncestorOf(w.prop_firm_panel.account_count))
        self.assertTrue(w.simulator_panel.isAncestorOf(w.start_edit))
        self.assertTrue(w.simulator_panel.isAncestorOf(w.end_edit))
        for index in range(tabs.count()):
            self.assertFalse(tabs.widget(index).isAncestorOf(w.start_edit))
            self.assertFalse(tabs.widget(index).isAncestorOf(w.end_edit))
        layout = w.simulator_panel.layout()
        self.assertLess(layout.indexOf(w.simulation_period), layout.indexOf(w.replay_btn))
        self.assertTrue(tabs.widget(1).isAncestorOf(w.risk_panel.risk_per_trade))
        self.assertTrue(tabs.widget(2).isAncestorOf(w.strategy_browser))
        self.assertFalse(w.simulator_panel.isAncestorOf(w.exit_panel))
        self.assertFalse(w.simulator_panel.isAncestorOf(w.session_panel))
        self.assertTrue(tabs.widget(3).isAncestorOf(w.trade_table))
        self.assertTrue(tabs.widget(3).isAncestorOf(w.replay_summary.account_table))
        self.assertFalse(tabs.widget(3).isAncestorOf(w._tabs))

    def test_switching_and_collapse(self):
        w = self.win
        w._workspace_panel.hide()
        w._select_workspace(1)
        self.assertEqual(w._workspace_stack.currentIndex(), 1)
        self.assertTrue(w._workspace_panel.isVisible())
        w._select_workspace(1)  # re-click collapses, returning chart width
        self.assertFalse(w._workspace_panel.isVisible())
        w._select_workspace(0)
        self.assertTrue(w._workspace_panel.isVisible())

    def test_toolbar_is_slim(self):
        # Simulation dates live only inside Simulator, never in the chart toolbar.
        w = self.win
        top = {w._controls.itemAt(i).widget() for i in range(w._controls.count())}
        self.assertIn(w.instrument_combo, top)
        self.assertIn(w.tf_combo, top)
        self.assertIn(w.simulator_button, top)
        self.assertNotIn(w.start_edit, top)
        self.assertNotIn(w.end_edit, top)
        self.assertNotIn(w.strategy_browser, top)
        self.assertNotIn(w.session_check, top)

    def test_sidebar_button_close_and_resize_preserve_settings(self):
        from PySide6 import QtCore, QtTest

        w = self.win
        w.resize(1400, 900)
        w._workspace_panel.hide()
        w._set_simulator_visible(False)
        QtTest.QTest.mouseClick(w.simulator_button, QtCore.Qt.LeftButton)
        self.app.processEvents()
        self.assertTrue(w.simulator_panel.isVisible())
        w.simulator_panel.select("Risk")
        w.risk_panel.risk_per_trade.setValue(1.7)
        old_width = w.simulator_panel.width()
        handle = w._split.handle(2)
        center = handle.rect().center()
        QtTest.QTest.mousePress(handle, QtCore.Qt.LeftButton, pos=center)
        QtTest.QTest.mouseMove(handle, center - QtCore.QPoint(140, 0), delay=30)
        QtTest.QTest.mouseRelease(handle, QtCore.Qt.LeftButton, pos=center - QtCore.QPoint(140, 0))
        self.app.processEvents()
        self.assertGreater(w.simulator_panel.width(), old_width + 50)
        expanded_width = w.simulator_panel.width()
        w.simulator_panel.close_button.click()
        self.assertFalse(w.simulator_panel.isVisible())
        self.assertFalse(w.simulator_button.isChecked())
        w.simulator_button.click()
        self.app.processEvents()
        self.assertAlmostEqual(w.simulator_panel.width(), expanded_width, delta=10)
        self.assertEqual(
            w.simulator_panel.tabs.tabText(w.simulator_panel.tabs.currentIndex()), "Risk"
        )
        self.assertAlmostEqual(w.risk_panel.risk_per_trade.value(), 1.7)
        w.risk_panel.risk_per_trade.setValue(1.0)

    def test_simulate_button_runs_selected_strategy_and_shows_performance(self):
        from PySide6 import QtCore

        w = self.win
        dates = [pd.Timestamp(d) for d in w._data["dates"]]
        chosen = dates[-1]
        qd = QtCore.QDate(chosen.year, chosen.month, chosen.day)
        w.start_edit.setDate(qd)
        w.end_edit.setDate(qd)
        spec = next(s for s in w._data["replay"].list_strategies() if "long" in s.name.lower())
        w.strategy_browser.select(spec.name)
        w.custom_check.setChecked(False)
        w.replay_btn.click()
        self.assertTrue(w._replay_running)
        self.assertFalse(w.replay_btn.isEnabled())
        deadline = time.monotonic() + 15
        while w._replay_running and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertFalse(w._replay_running)
        self.assertTrue(w.replay_btn.isEnabled())
        self.assertGreater(len(w._log), 0)
        self.assertTrue((pd.to_datetime(w._log["trade_date_ny"]) == chosen).all())
        self.assertIn(spec.name, w.replay_summary.description.text())
        self.assertEqual(w.replay_summary.values["trades"].text(), f"{len(w._log):,}")
        self.assertEqual(w.replay_summary.values["accounts"].text(), "10")
        self.assertEqual(w.simulator_panel.tabs.currentIndex(), 3)
        self.assertTrue(w.simulator_panel.tabs.widget(3).isAncestorOf(w.trade_table))
        self.assertIn(chosen.date().isoformat(), w.replay_summary.scope.text())

    def test_failed_run_can_retry_and_duplicate_runs_are_blocked(self):
        w = self.win
        spec = w._data["replay"].list_strategies()[0]
        w.strategy_browser.select(spec.name)
        with patch.object(w, "_start") as start, patch.object(w, "_on_error"):
            w._on_replay()
            w._on_replay()
            self.assertEqual(start.call_count, 1)
            w._on_replay_error(spec.name, "test error", w._replay_token)
        self.assertFalse(w._replay_running)
        self.assertTrue(w.replay_btn.isEnabled())
        self.assertEqual(w.replay_btn.text(), "Simulate")
        self.assertIn("failed", w.simulator_panel.status.text())

    def test_empty_run_clears_previous_results_and_stale_run_is_ignored(self):
        w = self.win
        w._on_replayed(pd.DataFrame(), strategy="No entries")
        self.assertEqual(w.trade_table.rowCount(), 0)
        self.assertEqual(w.replay_summary.values["trades"].text(), "0")
        self.assertIn("No trades", w.simulator_panel.status.text())
        w._on_replayed(pd.DataFrame(), token=w._replay_token - 1, strategy="Stale")
        self.assertIn("No entries", w.replay_summary.description.text())

    @classmethod
    def tearDownClass(cls):
        cls.win.close()
        cls.app.processEvents()

    def test_short_window_scrolls_performance_without_squeezing_metrics(self):
        w = self.win
        w.resize(1092, 614)
        w._workspace_panel.hide()
        w._on_replayed(pd.DataFrame(), strategy="No entries")
        for _ in range(4):
            self.app.processEvents()
        for metric in w.replay_summary.values.values():
            self.assertGreaterEqual(metric.height(), metric.minimumSizeHint().height())
        self.assertGreater(w._performance_scroll.verticalScrollBar().maximum(), 0)
        w.resize(1400, 900)


if __name__ == "__main__":
    unittest.main()
