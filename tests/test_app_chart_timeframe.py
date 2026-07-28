"""ChartWidget behaviour on a non-identity (aggregated) ViewMap.

Skips when PySide6/pyqtgraph are not installed. Uses a synthetic 5m map over a
one-hour 1m window to pin the coordinate contract: clicks report the clicked
bucket's last 1m index, the reveal curtain hides forming buckets whole, tracks
render at fractional x, drawings restore across timeframe switches, and the
price axis double-click re-arms Y autorange.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

from src.app.datalayer.timeframe import ViewMap, resample_window

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


class _Bars:
    def __init__(self, n=60, start_minute=420):
        self.minute_ny = np.arange(start_minute, start_minute + n)
        self.trade_date = np.full(n, np.datetime64("2024-06-03"))
        self.segment = np.ones(n, dtype=int)
        self.open = np.linspace(100, 102, n)
        self.high = self.open + 0.3
        self.low = self.open - 0.3
        self.close = self.open + 0.1
        self.volume = np.full(n, 25.0)


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class ChartTimeframeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _chart_5m(self):
        from src.app.ui.chart_widget import ChartWidget

        bars = _Bars(60)
        rs = resample_window(bars, 0, 59, 5)
        vm = ViewMap.from_resampled(rs, 5)
        chart = ChartWidget()
        ohlc = {
            "open": rs.open,
            "high": rs.high,
            "low": rs.low,
            "close": rs.close,
            "volume": rs.volume,
            "segment": rs.segment,
        }
        labels = np.array([f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in bars.minute_ny[::5]])
        chart.set_view(ohlc, labels, vm, minute_close=bars.close)
        return chart, vm, bars

    def test_click_reports_bucket_last_minute(self):
        chart, vm, _bars = self._chart_5m()
        got = []
        chart.bar_clicked.connect(got.append)
        chart._view_map = vm
        # Simulate the click resolution directly at local bar 2 (third bucket).
        chart.bar_clicked.emit(vm.local_to_global_end(2))
        self.assertEqual(got, [14])  # bucket 2 covers minutes 10-14

    def test_reveal_hides_forming_bucket_whole(self):
        chart, vm, _bars = self._chart_5m()
        chart.set_reveal(7)  # minute 7 sits inside bucket 1 (5-9): only bucket 0 shows
        lo, _hi = chart._curtains[0].getRegion()
        self.assertAlmostEqual(lo, 0.5)
        chart.set_reveal(9)  # bucket 1 completes exactly at minute 9
        lo, _hi = chart._curtains[0].getRegion()
        self.assertAlmostEqual(lo, 1.5)
        chart.set_reveal(None)
        self.assertFalse(chart._curtains[0].isVisible())

    def test_replay_marker_rides_minutes_fractionally(self):
        chart, vm, bars = self._chart_5m()
        chart.start_replay(None, start_global=0)
        chart.replay_frame(None, 7, start_global=0)
        x, y = chart._replay_marker.getData()
        self.assertAlmostEqual(float(x[0]), vm.global_to_local_f(7))
        self.assertAlmostEqual(float(y[0]), float(bars.close[7]))
        chart.stop_replay()

    def test_track_renders_fractional_and_monotonic(self):
        chart, vm, _bars = self._chart_5m()
        track = np.linspace(99.5, 99.8, 12)
        chart.draw_trade(10, 21, 100.0, track, track + 1.0, 100.4)
        xs, _ys = chart._active_trade_items["stop"].getData()
        self.assertEqual(len(xs), 12)
        self.assertTrue(np.all(np.diff(xs) > 0))
        self.assertAlmostEqual(xs[0], vm.global_to_local_f(10))

    def test_drawing_survives_timeframe_switch_at_same_timestamps(self):
        chart, vm, bars = self._chart_5m()
        chart.place_drawing("vline", vm.global_to_local_f(22), 0)
        spec = chart._serialize_drawings()[0]
        self.assertAlmostEqual(spec["gx"], 22.0, places=6)
        # Re-render the same window at 15m: the vline must land on minute 22.
        rs15 = resample_window(bars, 0, 59, 15)
        vm15 = ViewMap.from_resampled(rs15, 15)
        ohlc15 = {
            "open": rs15.open,
            "high": rs15.high,
            "low": rs15.low,
            "close": rs15.close,
            "volume": rs15.volume,
            "segment": rs15.segment,
        }
        labels15 = np.array(["a"] * len(rs15.open))
        chart.set_view(ohlc15, labels15, vm15, minute_close=bars.close)
        self.assertEqual(len(chart._drawing_items), 1)
        restored = chart._serialize_drawings()[0]
        self.assertAlmostEqual(restored["gx"], 22.0, places=6)

    def test_price_axis_double_click_rearms_autorange(self):
        chart, _vm, _bars = self._chart_5m()
        vb = chart._price.vb
        vb.setYRange(50, 60, padding=0)  # kills autorange
        self.assertFalse(vb.state["autoRange"][1])
        axis = chart._price.getAxis("right")
        self.assertTrue(hasattr(axis, "mouseDragEvent"))

        class _FakeDouble:
            def double(self):
                return True

            def accept(self):
                self.accepted = True

        axis.mouseClickEvent(_FakeDouble())
        self.assertTrue(bool(vb.state["autoRange"][1]))


if __name__ == "__main__":
    unittest.main()
