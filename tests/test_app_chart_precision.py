"""Chart-correctness: tick-size precision, single price scale, candle geometry."""

from __future__ import annotations

import importlib.util
import os
import re
import unittest

import numpy as np

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class ChartPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.ui import theme

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        theme.apply(cls.app, "dark")

    def _chart(self, tick=0.1):
        from src.app.datalayer.timeframe import ViewMap
        from src.app.ui.chart_widget import ChartWidget

        c = ChartWidget()
        n = 12
        close = np.linspace(1890.0, 1891.1, n)
        ohlc = {
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(n, 100.0),
            "segment": np.ones(n, dtype=int),
        }
        c.set_tick_size(tick)
        c.set_view(ohlc, np.array([f"{i:02d}:00" for i in range(n)]), ViewMap.identity(0, n - 1))
        return c

    def test_decimals_for_tick_sizes(self):
        from src.app.ui.chart_widget import _decimals_for

        self.assertEqual(_decimals_for(0.1), 1)
        self.assertEqual(_decimals_for(0.25), 2)
        self.assertEqual(_decimals_for(1.0), 0)
        self.assertEqual(_decimals_for(0.01), 2)

    def test_axis_formats_to_tick_precision(self):
        c = self._chart(tick=0.1)
        self.assertEqual(c._price_axis.tickStrings([1890.4], 1, 1), ["1,890.4"])  # never "1890"
        c.set_tick_size(0.25)
        self.assertEqual(c._price_axis.tickStrings([1890.25], 1, 1), ["1,890.25"])

    def test_single_price_scale_on_the_right(self):
        c = self._chart()
        self.assertTrue(c._price.getAxis("right").isVisible())
        self.assertFalse(c._price.getAxis("left").isVisible())
        self.assertFalse(c._volume.isVisible())  # volume is opt-in at startup
        c.set_volume_visible(True)
        self.assertTrue(c._volume.getAxis("right").isVisible())
        self.assertFalse(c._volume.getAxis("left").isVisible())

    def test_readout_drops_the_time_and_adds_change(self):
        c = self._chart()
        text = re.sub("<[^>]+>", "", c._readout(5))
        self.assertNotIn(":", text)  # no time/date label
        self.assertIn("%", text)  # shows the change percent
        self.assertRegex(text, r"[+-]\d")  # signed absolute change

    def test_doji_body_is_visible(self):
        from src.app.ui.chart_widget import candle_min_body

        highs = np.array([1891.0, 1892.0, 1890.5])
        lows = np.array([1890.0, 1890.0, 1890.5])  # last is a flat doji
        floor = candle_min_body(highs, lows)
        self.assertGreater(floor, 0.0)
        self.assertAlmostEqual(floor, 0.08 * np.median([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
