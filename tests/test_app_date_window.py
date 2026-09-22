"""The chart uses all loaded data while Simulator owns the date period.

Skips without the GUI deps or the local parquet artifacts (it builds the real
MainWindow under a narrow bar load).
"""

from __future__ import annotations

import importlib.util
import os
import unittest

import pandas as pd

from src.app.datalayer.paths import research_bars_path

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None
_HAS_BARS = research_bars_path().exists()


@unittest.skipUnless(_HAS_QT and _HAS_PG and _HAS_BARS, "GUI deps or bars parquet missing")
class DateWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.ui import theme
        from src.app.ui.main_window import MainWindow, load_app_data

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        theme.apply(cls.app, "dark")
        data = load_app_data(date_floor=pd.Timestamp("2024-12-01"))
        cls.win = MainWindow(data=data)
        cls.dates = [pd.Timestamp(d) for d in cls.win._active_data()["dates"]]

    def setUp(self):
        self.win._set_date_bounds(self.win._data["dates"])

    def _set_period(self, start_ts, end_ts):
        from PySide6 import QtCore

        self.win._date_guard = True
        for edit, t in ((self.win.start_edit, start_ts), (self.win.end_edit, end_ts)):
            edit.setDate(QtCore.QDate(t.year, t.month, t.day))
        self.win._date_guard = False

    def test_chart_always_spans_all_loaded_dev_val_bars(self):
        start, end = self.dates[1], self.dates[4]
        self._set_period(start, end)
        bars = self.win._bars
        self.win._render_view()
        self.assertEqual(self.win._view_start, 0)
        self.assertEqual(self.win._view_end, bars.n_bars - 1)
        self.assertEqual(pd.Timestamp(bars.trade_date[self.win._view_start]), self.dates[0])
        self.assertEqual(pd.Timestamp(bars.trade_date[self.win._view_end]), self.dates[-1])

    def test_simulation_period_defaults_to_all_loaded_dates(self):
        self.assertEqual(self.win.start_edit.date().toPython(), self.dates[0].date())
        self.assertEqual(self.win.end_edit.date().toPython(), self.dates[-1].date())

    def test_simulation_date_pair_stays_ordered_without_changing_chart(self):
        from PySide6 import QtCore

        before = (self.win._view_start, self.win._view_end)
        later = self.dates[5]
        earlier = self.dates[2]
        self.win.start_edit.setDate(QtCore.QDate(later.year, later.month, later.day))
        self.win.end_edit.setDate(QtCore.QDate(earlier.year, earlier.month, earlier.day))
        self.assertEqual(self.win.end_edit.date(), self.win.start_edit.date())
        self.assertEqual((self.win._view_start, self.win._view_end), before)


if __name__ == "__main__":
    unittest.main()
