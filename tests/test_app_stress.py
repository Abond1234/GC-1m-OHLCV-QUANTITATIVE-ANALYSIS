"""Cross-surface integration stress: replay, studies, drawings, sessions.

The corrections brief's definition of done demands that sidebar collapse,
indicator toggles, timeframe switches and session round-trips never lose
chart, strategy, risk or replay state - and the replay reveal must keep the
future hidden through all of it. This module drives the real MainWindow
through those combinations.
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
class AppStressTests(unittest.TestCase):
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

    def _pump(self, n=3):
        for _ in range(n):
            self.app.processEvents()

    def _wait_for(self, condition, timeout_s=15.0):
        import time

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return
            time.sleep(0.02)
        self.fail("timed out waiting for an async worker")

    def setUp(self):
        w = self.win
        w._animator.stop()
        w.chart.stop_replay()
        w.chart.clear_drawings()
        w.chart._drawing_store.clear()
        w.chart._redo_specs.clear()
        for inst_id in [i.id for i in w.indicators_panel.instances]:
            w.indicators_panel.remove_instance(inst_id)
        self._pump()

    def test_workspace_cycling_preserves_replay_reveal(self):
        w = self.win
        start = w._view_start + 30
        w.chart.start_replay(None, start_global=start)
        w.chart.replay_frame(None, start + 10, start_global=start)
        region_before = w.chart._curtains[0].getRegion()
        for i in range(5):
            w._select_workspace(i)
            self._pump(1)
        self.assertTrue(w.chart._curtains[0].isVisible())
        self.assertEqual(w.chart._curtains[0].getRegion(), region_before)
        w._select_workspace(0)

    def test_adding_studies_mid_replay_keeps_the_future_hidden(self):
        w = self.win
        start = w._view_start + 30
        w.chart.start_replay(None, start_global=start)
        w.chart.replay_frame(None, start + 5, start_global=start)
        edge = w.chart._curtains[0].getRegion()[0]
        w.indicators_panel.add_study("rsi")
        w.indicators_panel.add_study("ema")
        self._pump()
        # The oscillator pane exists now and its curtain matches the reveal.
        self.assertTrue(w.chart._osc.isVisible())
        for curtain in w.chart._curtains:
            self.assertAlmostEqual(curtain.getRegion()[0], edge)
        w.chart.stop_replay()

    def test_volume_and_overlay_toggles_leave_replay_and_view_alone(self):
        w = self.win
        start = w._view_start + 30
        w.chart.start_replay(None, start_global=start)
        w.chart.replay_frame(None, start + 5, start_global=start)
        edge = w.chart._curtains[0].getRegion()[0]
        w.indicators_panel.volume_check.setChecked(False)
        self._pump(1)
        self.assertFalse(w.chart._volume.isVisible())
        w.indicators_panel.volume_check.setChecked(True)
        self._pump(1)
        self.assertAlmostEqual(w.chart._curtains[0].getRegion()[0], edge)
        w.chart.stop_replay()

    def test_timeframe_switch_preserves_drawings_and_studies(self):
        w = self.win
        w.chart.place_drawing("hline", 5.0, float(w._bars.close[w._view_start] + 1.0))
        w.chart.place_drawing("fibret", 5.0, float(w._bars.close[w._view_start]))
        w.chart.place_drawing("fibret", 25.0, float(w._bars.close[w._view_start] + 3.0))
        w.indicators_panel.add_study("sma")
        self._pump()
        w.tf_combo.setCurrentText("15m")
        self._pump()
        kinds = sorted(kind for kind, _items in w.chart._drawing_items)
        self.assertEqual(kinds, ["fibret", "hline"])
        self.assertEqual(len(w.chart._indicator_items), 1)  # the study re-rendered
        w.tf_combo.setCurrentText("1m")
        self._pump()
        kinds = sorted(kind for kind, _items in w.chart._drawing_items)
        self.assertEqual(kinds, ["fibret", "hline"])

    def test_full_session_round_trip_recomputes_and_restores(self):
        w = self.win
        # A free-play trade through the real click path.
        w.freeplay_check.setChecked(True)
        w.long_radio.setChecked(True)
        w.chart.bar_clicked.emit(w._view_start + 60)
        self._pump()
        self.assertEqual(len(w._placed), 1)
        placed_result = next(iter(w._placed.values())).result
        self.assertIsNotNone(placed_result)
        # Studies, drawings, a measurement.
        w.indicators_panel.add_study("ema")
        w.chart.place_drawing("measure", 5.0, float(w._bars.close[w._view_start]))
        w.chart.place_drawing("measure", 15.0, float(w._bars.close[w._view_start] + 2.0))
        self._pump()
        payload = w._collect_session_payload()
        from src.app.datalayer.session_store import read_session, write_session

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stress_session.json"
            write_session(path, payload)
            loaded = read_session(path)
        # Wipe, then restore.
        w._placed.clear()
        w._active_id = None
        w.blotter.set_trades([])
        w.chart.clear_drawings()
        w.chart._drawing_store.clear()
        for inst_id in [i.id for i in w.indicators_panel.instances]:
            w.indicators_panel.remove_instance(inst_id)
        self._pump()
        w._apply_session_payload(loaded)
        self._pump()
        self.assertEqual(len(w._placed), 1)
        restored = next(iter(w._placed.values())).result
        self.assertIsNotNone(restored)  # re-simulated through the engine
        self.assertEqual(restored.exit_reason, placed_result.exit_reason)
        self.assertAlmostEqual(restored.gross_r, placed_result.gross_r)
        self.assertEqual([i.key for i in w.indicators_panel.instances], ["ema"])
        self.assertIn("measure", [kind for kind, _ in w.chart._drawing_items])
        w.freeplay_check.setChecked(False)

    def test_escape_chain_cancels_in_priority_order(self):
        from PySide6 import QtCore, QtGui

        w = self.win
        esc = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier)
        # 1. An in-progress measurement wins.
        w.chart._measure_anchor = (5.0, 1890.0)
        w.keyPressEvent(esc)
        self.assertFalse(w.chart.measuring())
        # 2. Then an armed drawing tool.
        w._arm_draw_tool("fibwedge")
        w.chart.place_drawing("fibwedge", 5.0, 1890.0)
        w.keyPressEvent(esc)
        self.assertIsNone(w.chart._draw_mode)
        self.assertEqual(w.chart._multi_anchors, [])

    def test_instrument_switch_and_back_survives_everything(self):
        w = self.win
        available = {w.instrument_combo.itemData(i) for i in range(w.instrument_combo.count())}
        if "MGC" not in available:
            self.skipTest("MGC data not available")
        w.indicators_panel.add_study("vwap_day")
        self._pump()
        w.instrument_combo.setCurrentIndex(w.instrument_combo.findData("MGC"))
        self._wait_for(lambda: w._instrument == "MGC" and not w._instrument_loading)
        self.assertEqual(w._instrument, "MGC")
        self.assertFalse(w.strategy_browser.tree.isEnabled())  # disabled with reason
        self.assertTrue(w.strategy_browser.reason.isVisibleTo(w.strategy_browser))
        w.instrument_combo.setCurrentIndex(w.instrument_combo.findData("GC"))
        self._wait_for(lambda: w._instrument == "GC" and not w._instrument_loading)
        self.assertEqual(w._instrument, "GC")
        self.assertTrue(w.strategy_browser.tree.isEnabled())
        # The studies survived the round trip.
        self.assertEqual([i.key for i in w.indicators_panel.instances], ["vwap_day"])


if __name__ == "__main__":
    unittest.main()
