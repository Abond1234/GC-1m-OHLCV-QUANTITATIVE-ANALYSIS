"""Fibonacci tools: geometry correctness and on-chart lifecycle."""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

from src.app.analysis.fib import (
    FIB_POINTS,
    RETRACEMENT_LEVELS,
    TIME_SEQUENCE,
    geometry,
)

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


class FibGeometryTests(unittest.TestCase):
    def test_retracement_levels_span_the_swing(self):
        lines = geometry("fibret", [(10, 100.0), (20, 110.0)])
        self.assertEqual(len(lines), len(RETRACEMENT_LEVELS))
        ys = [line.ys[0] for line in lines]
        self.assertAlmostEqual(max(ys), 110.0)  # level 0 at the swing end
        self.assertAlmostEqual(min(ys), 100.0)  # level 1 back at the start
        golden = next(line for line in lines if line.label.startswith("0.618"))
        self.assertAlmostEqual(golden.ys[0], 110.0 - 0.618 * 10.0)

    def test_extension_projects_from_the_third_point(self):
        lines = geometry("fibext", [(0, 100.0), (10, 110.0), (15, 105.0)])
        one = next(line for line in lines if line.label.startswith("1 "))
        self.assertAlmostEqual(one.ys[0], 115.0)  # 105 + 1.0 * (110-100)

    def test_time_zones_at_fibonacci_multiples(self):
        lines = geometry("fibtime", [(100, 50.0), (110, 60.0)])
        xs = [line.xs[0] for line in lines]
        np.testing.assert_allclose(xs, [100 + n * 10 for n in TIME_SEQUENCE])
        for line in lines:  # vertical: both x equal
            self.assertEqual(line.xs[0], line.xs[1])

    def test_channel_lines_are_parallel(self):
        lines = geometry("fibchan", [(0, 100.0), (10, 110.0), (0, 95.0)], extend=1.0)
        slopes = {round((line.ys[1] - line.ys[0]) / (line.xs[1] - line.xs[0]), 9) for line in lines}
        self.assertEqual(len(slopes), 1)  # every channel line shares the base slope

    def test_fan_rays_start_at_the_anchor(self):
        lines = geometry("fibfan", [(5, 100.0), (15, 110.0)])
        for line in lines:
            self.assertAlmostEqual(line.xs[0], 5.0)
            self.assertAlmostEqual(line.ys[0], 100.0)
            self.assertGreater(line.xs[1], 1000.0)  # extends beyond the anchor

    def test_circles_arcs_spiral_wedge_pitchfan_produce_polylines(self):
        cases = {
            "fibcircles": [(10, 100.0), (20, 110.0)],
            "fibarcs": [(10, 100.0), (20, 110.0)],
            "fibspiral": [(10, 100.0), (20, 110.0)],
            "fibwedge": [(0, 100.0), (10, 110.0), (10, 95.0)],
            "pitchfan": [(0, 100.0), (10, 110.0), (10, 95.0)],
        }
        for kind, pts in cases.items():
            lines = geometry(kind, pts)
            self.assertGreater(len(lines), 0, kind)
            for line in lines:
                self.assertEqual(len(line.xs), len(line.ys), kind)
                self.assertTrue(all(np.isfinite(line.xs)), kind)
                self.assertTrue(all(np.isfinite(line.ys)), kind)

    def test_every_registered_kind_has_geometry(self):
        pts3 = [(0, 100.0), (10, 110.0), (5, 105.0)]
        for kind, n in FIB_POINTS.items():
            lines = geometry(kind, pts3[:n])
            self.assertGreater(len(lines), 0, kind)

    def test_too_few_points_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            geometry("fibext", [(0, 1.0), (1, 2.0)])


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class FibLifecycleTests(unittest.TestCase):
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

    def test_two_point_fib_places_by_two_clicks(self):
        c = self._chart()
        c.place_drawing("fibret", 5.0, 1890.5)
        c.place_drawing("fibret", 15.0, 1893.5)
        self.assertEqual(len(c._drawing_items), 1)
        spec = c._serialize_drawings()[0]
        self.assertEqual(spec["kind"], "fibret")
        self.assertEqual(len(spec["points"]), 2)

    def test_three_point_fib_places_by_three_clicks(self):
        c = self._chart()
        c.place_drawing("fibext", 5.0, 1890.5)
        c.place_drawing("fibext", 15.0, 1893.5)
        self.assertEqual(len(c._drawing_items), 0)  # still pending
        self.assertEqual(len(c._multi_anchors), 2)
        c.place_drawing("fibext", 25.0, 1892.0)
        self.assertEqual(len(c._drawing_items), 1)
        self.assertEqual(len(c._multi_anchors), 0)
        spec = c._serialize_drawings()[0]
        self.assertEqual(len(spec["points"]), 3)

    def test_pending_three_point_gesture_undoes_cleanly(self):
        c = self._chart()
        c.place_drawing("fibwedge", 5.0, 1890.5)
        c.undo_drawing()  # cancels the pending anchors, not a placed drawing
        self.assertEqual(len(c._multi_anchors), 0)
        self.assertEqual(len(c._drawing_items), 0)

    def test_fib_survives_view_rebuild_and_undo_redo(self):
        from src.app.datalayer.timeframe import ViewMap

        c = self._chart()
        c.place_drawing("fibret", 5.0, 1890.5)
        c.place_drawing("fibret", 15.0, 1893.5)
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
        self.assertEqual(len(c._drawing_items), 1)
        c.undo_drawing()
        c.redo_drawing()
        spec = c._serialize_drawings()[0]
        self.assertAlmostEqual(spec["points"][0][0], 5.0, places=6)
        self.assertAlmostEqual(spec["points"][1][1], 1893.5, places=6)

    def test_dragging_a_fib_handle_moves_the_levels(self):
        c = self._chart()
        c.place_drawing("fibret", 5.0, 1890.0)
        c.place_drawing("fibret", 15.0, 1894.0)
        handles = c._drawing_items[0][1][-2:]
        curves = c._drawing_items[0][1][: len(RETRACEMENT_LEVELS)]
        top_before = max(float(np.max(curve.yData)) for curve in curves)
        self.assertAlmostEqual(top_before, 1894.0)
        handles[1].setPos(15.0, 1895.0)
        top_after = max(float(np.max(curve.yData)) for curve in curves)
        self.assertAlmostEqual(top_after, 1895.0)


if __name__ == "__main__":
    unittest.main()
