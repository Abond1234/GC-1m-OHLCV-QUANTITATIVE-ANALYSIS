"""Line/position drawing tools: placement, serialization, restore, undo/redo."""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class DrawingToolsTests(unittest.TestCase):
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
        n = 60
        close = np.linspace(1890.0, 1896.0, n)
        ohlc = {
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(n, 100.0),
            "segment": np.ones(n, dtype=int),
        }
        c.set_view(ohlc, np.array([f"{i:02d}" for i in range(n)]), ViewMap.identity(0, n - 1))
        return c

    def _two_point(self, c, kind, p1=(5.0, 1890.5), p2=(15.0, 1893.5)):
        c.place_drawing(kind, *p1)
        c.place_drawing(kind, *p2)

    def test_every_line_tool_places_and_serializes(self):
        for kind in ("trend", "ray", "info", "extline", "angle"):
            c = self._chart()
            self._two_point(c, kind)
            self.assertEqual(len(c._drawing_items), 1, kind)
            spec = c._serialize_drawings()[0]
            self.assertEqual(spec["kind"], kind)
            self.assertAlmostEqual(spec["p1"][0], 5.0, places=6)
            self.assertAlmostEqual(spec["p2"][1], 1893.5, places=6)

    def test_ray_extends_beyond_the_second_point(self):
        c = self._chart()
        self._two_point(c, "ray")
        line = c._drawing_items[0][1][0]
        self.assertGreater(float(np.max(line.xData)), 1000.0)  # reaches far right
        self.assertAlmostEqual(float(np.min(line.xData)), 5.0)  # anchored at p1

    def test_extended_line_reaches_both_directions(self):
        c = self._chart()
        self._two_point(c, "extline")
        line = c._drawing_items[0][1][0]
        self.assertGreater(float(np.max(line.xData)), 1000.0)
        self.assertLess(float(np.min(line.xData)), -1000.0)

    def test_info_line_and_trend_angle_carry_labels(self):
        c = self._chart()
        self._two_point(c, "info")
        info_label = c._drawing_items[0][1][1]
        self.assertIn("pts", info_label.toPlainText())
        self.assertIn("bars", info_label.toPlainText())
        self._two_point(c, "angle")
        angle_label = c._drawing_items[1][1][1]
        self.assertIn("deg", angle_label.toPlainText())
        self.assertIn("pts/bar", angle_label.toPlainText())

    def test_horizontal_ray_and_crossline_place_and_restore(self):
        from src.app.datalayer.timeframe import ViewMap

        c = self._chart()
        c.place_drawing("hray", 10.0, 1891.0)
        c.place_drawing("cross", 20.0, 1892.0)
        specs = c._serialize_drawings()
        self.assertEqual({s["kind"] for s in specs}, {"hray", "cross"})
        hray = next(s for s in specs if s["kind"] == "hray")
        self.assertAlmostEqual(hray["gx"], 10.0, places=6)
        self.assertAlmostEqual(hray["y"], 1891.0)
        # Round-trip through a view rebuild.
        n = 60
        close = np.linspace(1890.0, 1896.0, n)
        ohlc = {
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(n, 100.0),
            "segment": np.ones(n, dtype=int),
        }
        c.set_view(ohlc, np.array([f"{i:02d}" for i in range(n)]), ViewMap.identity(0, n - 1))
        self.assertEqual(len(c._drawing_items), 2)

    def test_long_position_tool_geometry_and_serialization(self):
        c = self._chart()
        self._two_point(c, "longpos", p1=(10.0, 1891.0), p2=(20.0, 1893.0))
        spec = c._serialize_drawings()[0]
        self.assertEqual(spec["kind"], "longpos")
        self.assertAlmostEqual(spec["p1"][1], 1891.0)  # entry
        self.assertAlmostEqual(spec["p2"][1], 1893.0)  # target above entry
        self.assertAlmostEqual(spec["stop"], 1890.0)  # default: half the reward below
        label = c._drawing_items[0][1][3]
        self.assertIn("Long", label.toPlainText())
        self.assertIn("RR 2.00", label.toPlainText())

    def test_short_position_mirrors(self):
        c = self._chart()
        self._two_point(c, "shortpos", p1=(10.0, 1893.0), p2=(20.0, 1891.0))
        spec = c._serialize_drawings()[0]
        self.assertAlmostEqual(spec["p2"][1], 1891.0)  # target below entry
        self.assertAlmostEqual(spec["stop"], 1894.0)  # risk above
        label = c._drawing_items[0][1][3]
        self.assertIn("Short", label.toPlainText())

    def test_position_stop_survives_undo_redo(self):
        c = self._chart()
        self._two_point(c, "longpos", p1=(10.0, 1891.0), p2=(20.0, 1893.0))
        h_stop = c._drawing_items[0][1][-1]
        h_stop.setPos(20.0, 1890.5)  # user drags the stop
        c.undo_drawing()
        c.redo_drawing()
        spec = c._serialize_drawings()[0]
        self.assertAlmostEqual(spec["stop"], 1890.5)  # the dragged stop came back

    def test_dragging_a_handle_updates_the_line(self):
        c = self._chart()
        self._two_point(c, "trend")
        line, h1, h2 = c._drawing_items[0][1]
        h2.setPos(30.0, 1895.0)
        self.assertAlmostEqual(float(line.xData[1]), 30.0)
        self.assertAlmostEqual(float(line.yData[1]), 1895.0)


if __name__ == "__main__":
    unittest.main()
