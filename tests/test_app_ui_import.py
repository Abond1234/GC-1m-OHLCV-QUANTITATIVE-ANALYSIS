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

from src.app.datalayer.timeframe import ViewMap

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


def _synthetic_ohlc(n: int, segment=None) -> dict:
    return {
        "open": np.linspace(100, 101, n),
        "high": np.linspace(100.2, 101.2, n),
        "low": np.linspace(99.8, 100.8, n),
        "close": np.linspace(100.1, 101.1, n),
        "volume": np.full(n, 100.0),
        "segment": np.ones(n, dtype=int) if segment is None else np.asarray(segment),
    }


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class ChartWidgetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_chart_constructs_and_renders_synthetic_view(self):
        from src.app.ui.chart_widget import ChartWidget

        chart = ChartWidget()
        n = 30
        ohlc = _synthetic_ohlc(n)
        labels = np.array([f"{i:02d}:00" for i in range(n)])
        chart.set_view(ohlc, labels, ViewMap.identity(0, n - 1))
        for kind in ("rolling", "day", "session"):
            chart.add_vwap(np.linspace(100, 101, n), kind)
        chart.shade_sessions(np.where(np.arange(n) % 10 < 5, 2, 0).astype(np.int8))
        chart.mark_trades([2, 8], [100.1, 100.6], [1, -1], [0.5, -1.0])
        chart.draw_trade(3, 9, 100.2, np.full(6, 99.7), np.full(6, 101.2), 101.2)
        chart.clear_trades()

    def test_candles_split_at_segment_break(self):
        from src.app.ui.chart_widget import CandlestickItem, ChartWidget

        chart = ChartWidget()
        n = 8
        ohlc = _synthetic_ohlc(n, segment=[1, 1, 1, 1, 2, 2, 2, 2])
        labels = np.array([f"{i:02d}:00" for i in range(n)])
        chart.set_view(ohlc, labels, ViewMap.identity(0, n - 1))
        candles = [it for it in chart._price.items if isinstance(it, CandlestickItem)]
        self.assertEqual(len(candles), 2)  # one item per continuous segment

    def test_crosshair_readout_reports_ohlc(self):
        from src.app.ui.chart_widget import ChartWidget

        chart = ChartWidget()
        n = 12
        chart.set_view(
            _synthetic_ohlc(n),
            np.array([f"{i:02d}:00" for i in range(n)]),
            ViewMap.identity(0, n - 1),
        )
        text = chart._readout(5)
        for tag in ("O", "H", "L", "C"):
            self.assertIn(tag, text)

    def test_reveal_curtain_hides_and_lifts(self):
        from src.app.ui.chart_widget import ChartWidget

        chart = ChartWidget()
        n = 20
        chart.set_view(
            _synthetic_ohlc(n),
            np.array([f"{i:02d}:00" for i in range(n)]),
            ViewMap.identity(0, n - 1),
        )
        self.assertEqual(len(chart._curtains), 3)  # price, volume, oscillator
        self.assertFalse(chart._curtains[0].isVisible())
        chart.set_reveal(7)
        for i, curtain in enumerate(chart._curtains):
            if i < 2:  # the oscillator pane is itself hidden until a study needs it
                self.assertTrue(curtain.isVisible())
            lo, hi = curtain.getRegion()
            self.assertAlmostEqual(lo, 7.5)
            self.assertGreaterEqual(hi, n)
        chart.set_reveal(None)
        self.assertFalse(chart._curtains[0].isVisible())

    def test_day_replay_without_a_trade(self):
        from src.app.ui.chart_widget import ChartWidget

        chart = ChartWidget()
        n = 20
        chart.set_view(
            _synthetic_ohlc(n),
            np.array([f"{i:02d}:00" for i in range(n)]),
            ViewMap.identity(0, n - 1),
        )
        chart.start_replay(None, start_global=0)
        self.assertTrue(chart._curtains[0].isVisible())
        chart.replay_frame(None, 9, start_global=0)
        self.assertAlmostEqual(chart._curtains[0].getRegion()[0], 9.5)
        self.assertIn("+9m", chart._replay_label.toPlainText())
        chart.stop_replay()
        self.assertFalse(chart._curtains[0].isVisible())

    def test_drawings_place_undo_clear_and_survive_view_switch(self):
        from src.app.ui.chart_widget import ChartWidget

        chart = ChartWidget()
        n = 20
        labels = np.array([f"{i:02d}:00" for i in range(n)])
        chart.set_view(_synthetic_ohlc(n), labels, ViewMap.identity(0, n - 1))
        chart.place_drawing("hline", 0, 100.5)
        chart.place_drawing("vline", 4, 0)
        chart.place_drawing("trend", 2, 100.2)  # first click: anchor only
        self.assertEqual(len(chart._drawing_items), 2)
        chart.place_drawing("trend", 9, 100.9)  # second click completes it
        self.assertEqual(len(chart._drawing_items), 3)
        # Switch to another window and back: this day's drawings return.
        chart.set_view(_synthetic_ohlc(n), labels, ViewMap.identity(500, 500 + n - 1))
        self.assertEqual(len(chart._drawing_items), 0)
        chart.set_view(_synthetic_ohlc(n), labels, ViewMap.identity(0, n - 1))
        self.assertEqual(len(chart._drawing_items), 3)
        chart.undo_drawing()
        self.assertEqual(len(chart._drawing_items), 2)
        chart.clear_drawings()
        self.assertEqual(len(chart._drawing_items), 0)

    def test_draw_mode_click_routing(self):
        from src.app.ui.chart_widget import ChartWidget

        chart = ChartWidget()
        n = 10
        chart.set_view(
            _synthetic_ohlc(n),
            np.array([f"{i:02d}:00" for i in range(n)]),
            ViewMap.identity(0, n - 1),
        )
        placed = []
        chart.drawing_placed.connect(lambda: placed.append(True))
        chart.set_draw_mode("hline")
        chart.place_drawing("hline", 3, 100.4)
        self.assertEqual(placed, [True])
        chart.set_draw_mode(None)

    def test_mirror_trade_draws_ghost_items(self):
        from src.app.ui.chart_widget import ChartWidget

        chart = ChartWidget()
        n = 20
        chart.set_view(
            _synthetic_ohlc(n),
            np.array([f"{i:02d}:00" for i in range(n)]),
            ViewMap.identity(0, n - 1),
        )
        before = len(chart._trade_items)
        chart.mirror_trade(4, 12, 100.4, 100.1, 100.9, 100.8)
        self.assertEqual(len(chart._trade_items), before + 4)  # 2 levels + entry + exit
        chart.clear_trades()
        self.assertEqual(len(chart._trade_items), 0)

    def test_rect_zone_places_and_serializes(self):
        from src.app.ui.chart_widget import ChartWidget

        chart = ChartWidget()
        n = 12
        chart.set_view(
            _synthetic_ohlc(n),
            np.array([f"{i:02d}:00" for i in range(n)]),
            ViewMap.identity(0, n - 1),
        )
        chart.place_drawing("rect", 2, 100.2)  # anchor corner
        chart.place_drawing("rect", 8, 100.9)  # opposite corner
        self.assertEqual([k for k, _ in chart._drawing_items], ["rect"])
        spec = chart._serialize_drawings()[0]
        self.assertEqual(spec["kind"], "rect")
        self.assertAlmostEqual(spec["p1"][0], 2.0)
        self.assertAlmostEqual(spec["p2"][1], 100.9)

    def test_drawings_are_zoom_stable(self):
        from src.app.ui.chart_widget import ChartWidget

        chart = ChartWidget()
        n = 30
        chart.set_view(
            _synthetic_ohlc(n),
            np.array([f"{i:02d}:00" for i in range(n)]),
            ViewMap.identity(0, n - 1),
        )
        chart.place_drawing("hline", 0, 100.5)
        chart.place_drawing("trend", 3, 100.2)
        chart.place_drawing("trend", 20, 100.9)
        chart.place_drawing("rect", 5, 100.3)
        chart.place_drawing("rect", 12, 100.7)
        before = chart._serialize_drawings()
        # Zoom hard in and out: data-coordinate drawings must not move.
        chart._price.setXRange(8, 12, padding=0)
        chart._price.setYRange(100.4, 100.6, padding=0)
        self.app.processEvents()
        chart._price.setXRange(-5, 60, padding=0)
        chart._price.setYRange(95, 106, padding=0)
        self.app.processEvents()
        after = chart._serialize_drawings()
        self.assertEqual(len(before), len(after))
        for b, a in zip(before, after, strict=True):
            self.assertEqual(b["kind"], a["kind"])
            for key in ("y", "gx", "p1", "p2"):
                if key in b:
                    if isinstance(b[key], tuple):
                        self.assertAlmostEqual(b[key][0], a[key][0], places=9)
                        self.assertAlmostEqual(b[key][1], a[key][1], places=9)
                    else:
                        self.assertAlmostEqual(b[key], a[key], places=9)


if __name__ == "__main__":
    unittest.main()
