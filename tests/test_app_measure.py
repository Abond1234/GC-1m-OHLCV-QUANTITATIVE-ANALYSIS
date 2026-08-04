"""Measurement mode: math, lifecycle, undo, and zoom-stable values."""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

from src.app.analysis.measure import Measurement, format_duration, format_measurement, measure

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


class MeasureMathTests(unittest.TestCase):
    def test_signed_change_ticks_and_percent(self):
        m = measure(1890.0, 1893.5, 0.1, bars=10, minutes=10)
        self.assertAlmostEqual(m.price_change, 3.5)
        self.assertAlmostEqual(m.ticks, 35.0)
        self.assertAlmostEqual(m.percent, 3.5 / 1890.0 * 100.0)
        down = measure(1890.0, 1885.0, 0.1, bars=3, minutes=3)
        self.assertAlmostEqual(down.price_change, -5.0)
        self.assertAlmostEqual(down.ticks, -50.0)

    def test_tick_size_comes_from_the_instrument(self):
        es = measure(5000.0, 5001.0, 0.25, bars=4, minutes=4)  # ES tick 0.25
        self.assertAlmostEqual(es.ticks, 4.0)
        btc = measure(60000.0, 60010.0, 1.0, bars=1, minutes=1)
        self.assertAlmostEqual(btc.ticks, 10.0)

    def test_bars_and_minutes_are_absolute(self):
        m = measure(10.0, 11.0, 0.1, bars=-7, minutes=-420)
        self.assertEqual(m.bars, 7)
        self.assertEqual(m.minutes, 420)

    def test_duration_formatting(self):
        self.assertEqual(format_duration(45), "45m")
        self.assertEqual(format_duration(200), "3h 20m")
        self.assertEqual(format_duration(180), "3h")
        self.assertEqual(format_duration(60 * 28), "1d 4h")

    def test_readout_contains_every_fact(self):
        text = format_measurement(
            Measurement(price_change=3.5, ticks=35, percent=0.19, bars=10, minutes=10), 1
        )
        self.assertIn("+3.5 pts", text)
        self.assertIn("(+0.19%)", text)
        self.assertIn("+35 ticks", text)
        self.assertIn("10 bars", text)
        self.assertIn("10m", text)


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class MeasureLifecycleTests(unittest.TestCase):
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

    def test_measurement_commits_as_a_drawing_and_undoes(self):
        c = self._chart()
        c.place_drawing("measure", 5.0, 1890.5)  # anchor
        c.place_drawing("measure", 15.0, 1893.5)  # destination completes it
        self.assertEqual(len(c._drawing_items), 1)
        self.assertEqual(c._drawing_items[0][0], "measure")
        c.undo_drawing()
        self.assertEqual(len(c._drawing_items), 0)
        c.redo_drawing()
        self.assertEqual(c._drawing_items[0][0], "measure")

    def test_measurement_serializes_globally_and_survives_reviews(self):
        from src.app.datalayer.timeframe import ViewMap

        c = self._chart()
        c.place_drawing("measure", 5.0, 1890.5)
        c.place_drawing("measure", 15.0, 1893.5)
        spec = c._serialize_drawings()[0]
        self.assertEqual(spec["kind"], "measure")
        self.assertAlmostEqual(spec["p1"][0], 5.0, places=6)
        self.assertAlmostEqual(spec["p2"][0], 15.0, places=6)
        # Re-render the same window: the measurement comes back identically.
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
        restored = c._serialize_drawings()[0]
        self.assertAlmostEqual(restored["p1"][0], 5.0, places=6)
        self.assertAlmostEqual(restored["p2"][1], 1893.5, places=6)

    def test_values_identical_before_and_after_zoom(self):
        c = self._chart()
        c.place_drawing("measure", 5.0, 1890.5)
        c.place_drawing("measure", 15.0, 1893.5)
        text_item = c._drawing_items[0][1][2]
        before = text_item.toPlainText()
        c._price.vb.setXRange(0, 20, padding=0)  # zoom in hard
        c._price.vb.setYRange(1890, 1892, padding=0)
        after = c._drawing_items[0][1][2].toPlainText()
        self.assertEqual(before, after)
        self.assertIn("+3.0 pts", before)
        self.assertIn("+30 ticks", before)
        self.assertIn("10 bars", before)

    def test_esc_cancels_an_anchored_gesture(self):
        c = self._chart()
        c._measure_anchor = (5.0, 1890.5)
        c._update_measure_preview((5.0, 1890.5), (8.0, 1891.0))
        self.assertTrue(c.measuring())
        c.cancel_measurement()
        self.assertFalse(c.measuring())
        self.assertIsNone(c._measure_preview)
        self.assertEqual(len(c._drawing_items), 0)


if __name__ == "__main__":
    unittest.main()
