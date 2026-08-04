"""Chart header and crosshair contract: Undo/Redo/Clear, pills, last-price marker."""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class ChartHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.ui import theme

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        theme.apply(cls.app, "dark")

    def _chart(self):
        from src.app.datalayer.timeframe import ViewMap
        from src.app.ui.chart_widget import ChartWidget

        c = ChartWidget()
        n = 30
        close = np.linspace(1890.0, 1893.0, n)
        ohlc = {
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(n, 100.0),
            "segment": np.ones(n, dtype=int),
        }
        hover = np.array([f"2024-06-03 09:{i:02d}" for i in range(n)], dtype=object)
        c.set_view(
            ohlc,
            np.array([f"09:{i:02d}" for i in range(n)]),
            ViewMap.identity(0, n - 1),
            hover_labels=hover,
        )
        return c

    def test_header_has_undo_redo_clear(self):
        c = self._chart()
        self.assertEqual(set(c._header_buttons), {"Undo", "Redo", "Clear"})
        for btn in c._header_buttons.values():
            self.assertTrue(btn.isEnabled())

    def test_undo_redo_restores_the_drawing(self):
        c = self._chart()
        c.place_drawing("hline", 5.0, 1891.0)
        self.assertEqual(len(c._drawing_items), 1)
        c.undo_drawing()
        self.assertEqual(len(c._drawing_items), 0)
        self.assertEqual(len(c._redo_specs), 1)
        c.redo_drawing()
        self.assertEqual(len(c._drawing_items), 1)
        self.assertEqual(len(c._redo_specs), 0)
        spec = c._serialize_drawings()[0]
        self.assertEqual(spec["kind"], "hline")
        self.assertAlmostEqual(spec["y"], 1891.0)

    def test_clear_is_redoable_and_new_drawing_invalidates_redo(self):
        c = self._chart()
        c.place_drawing("hline", 5.0, 1891.0)
        c.place_drawing("vline", 7.0, 0.0)
        c.clear_drawings()
        self.assertEqual(len(c._drawing_items), 0)
        self.assertEqual(len(c._redo_specs), 2)
        c.redo_drawing()
        c.redo_drawing()
        self.assertEqual(len(c._drawing_items), 2)
        c.undo_drawing()
        self.assertEqual(len(c._redo_specs), 1)
        c.place_drawing("hline", 9.0, 1892.0)  # a fresh drawing clears redo history
        self.assertEqual(len(c._redo_specs), 0)

    def test_crosshair_pill_shows_exact_date_time(self):
        c = self._chart()
        c._update_badges(5.0, 1891.0, 5)
        self.assertIn("2024-06-03 09:05", c._x_badge.toPlainText())
        self.assertTrue(c._x_badge.isVisible())
        self.assertTrue(c._y_badge.isVisible())

    def test_price_pill_snaps_to_tick(self):
        c = self._chart()
        c.set_tick_size(0.25)
        c._update_badges(5.0, 1891.13, 5)
        self.assertEqual(c._y_badge.toPlainText().strip(), "1,891.25")

    def test_last_price_marker_tracks_view_and_replay(self):
        c = self._chart()
        self.assertTrue(c._last_badge.isVisible())
        self.assertIn("1,893.0", c._last_badge.toPlainText())
        # The pill follows the replay tape and returns to the window close after.
        c.start_replay(None, start_global=5)
        c.replay_frame(None, 10, start_global=5)
        self.assertIn("1,89", c._last_badge.toPlainText())
        self.assertAlmostEqual(c._last_price, float(c._minute_close[10]))
        c.stop_replay()
        self.assertAlmostEqual(c._last_price, 1893.0)

    def test_readout_and_header_never_overlap(self):
        # The readout and the buttons share one layout row: the readout can
        # never be covered, whatever the chart width.
        c = self._chart()
        c.resize(420, 300)
        c.show()
        self.app.processEvents()
        readout = c._readout_label.geometry()
        for btn in c._header_buttons.values():
            self.assertFalse(readout.intersects(btn.geometry()))
        c.hide()


if __name__ == "__main__":
    unittest.main()
