"""Headless smoke test for the UI widgets.

Skips when PySide6 is not installed (CI and the research env do not have it). When
it is installed, the chart is constructed under the Qt "offscreen" platform and
fed a synthetic view so a broken import or a candlestick-drawing regression is
caught without a display.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class ChartWidgetSmokeTests(unittest.TestCase):
    def test_chart_constructs_and_renders_synthetic_view(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.ui.chart_widget import ChartWidget

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.assertIsNotNone(app)
        chart = ChartWidget()
        n = 30
        ohlc = {
            "open": np.linspace(100, 101, n),
            "high": np.linspace(100.2, 101.2, n),
            "low": np.linspace(99.8, 100.8, n),
            "close": np.linspace(100.1, 101.1, n),
            "volume": np.full(n, 100.0),
        }
        labels = np.array([f"{i:02d}:00" for i in range(n)])
        chart.set_view(ohlc, labels, start_index=0)
        chart.add_vwap(np.linspace(100, 101, n))
        chart.shade_sessions(np.where(np.arange(n) % 10 < 5, 2, 0).astype(np.int8))
        chart.mark_trades([2, 8], [100.1, 100.6], [1, -1], [0.5, -1.0])
        chart.draw_trade(3, 9, 100.2, np.full(6, 99.7), np.full(6, 101.2), 101.2)
        chart.clear_trades()


if __name__ == "__main__":
    unittest.main()
