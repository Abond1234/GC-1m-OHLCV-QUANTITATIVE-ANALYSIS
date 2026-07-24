"""Smoke tests for the exit-grid heatmap and the replay animator. Skips without Qt."""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

from src.app.analysis.grid_sweep import sweep_entry
from src.app.sim.exit_config import ExitConfig
from src.app.sim.flex_exit import single_flex_exit

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None

_CFG = ExitConfig(forced_exit_minute_ny=None, max_holding_minutes=500)


def _uptrend(n=30):
    close = 100.0 + np.arange(n) * 0.5
    return {
        "open": close - 0.01,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": np.full(n, 100.0),
        "segment": np.ones(n, dtype=int),
        "trade_date": np.full(n, np.datetime64("2023-06-01")),
        "minute_ny": (np.arange(n) % 1440).astype(np.int64),
    }


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class M4WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_heatmap_renders_a_grid(self):
        from src.app.ui.heatmap_widget import HeatmapWidget

        arrays = _uptrend()
        grid = sweep_entry(
            1, 1, _CFG, np.full(30, 1.0), arrays, np.array([0.5, 1.0, 1.5]), np.array([1.0, 2.0])
        )
        w = HeatmapWidget()
        w.set_grid(grid)
        w.metric_combo.setCurrentText("Exit reason")  # exercises the other colour map

    def test_animator_loads_and_steps(self):
        from src.app.ui.chart_widget import ChartWidget
        from src.app.ui.replay_animator import ReplayAnimator

        arrays = _uptrend()
        chart = ChartWidget()
        labels = np.array([f"{i:02d}:00" for i in range(30)])
        chart.set_view(arrays, labels, start_index=0)
        result = single_flex_exit(1, 1, 1.5, 3.0, _CFG, arrays)
        animator = ReplayAnimator(chart)
        animator.load(result)
        animator.step(1)
        animator.step(-1)
        animator.stop()


if __name__ == "__main__":
    unittest.main()
