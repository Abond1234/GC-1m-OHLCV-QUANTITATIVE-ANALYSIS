"""The literal Start/End date pipeline loads exactly the requested interval.

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

    def _set_window(self, start_ts, end_ts):
        from PySide6 import QtCore

        self.win._date_guard = True
        for edit, t in ((self.win.start_edit, start_ts), (self.win.end_edit, end_ts)):
            edit.setDate(QtCore.QDate(t.year, t.month, t.day))
        self.win._date_guard = False
        self.win._render_view()

    def test_first_and_last_bar_match_the_chosen_dates(self):
        start, end = self.dates[1], self.dates[4]
        self._set_window(start, end)
        bars = self.win._bars
        self.assertEqual(pd.Timestamp(bars.trade_date[self.win._view_start]), start)
        self.assertEqual(pd.Timestamp(bars.trade_date[self.win._view_end]), end)

    def test_single_day_window_is_exactly_one_day(self):
        day = self.dates[3]
        self._set_window(day, day)
        bars = self.win._bars
        self.assertEqual(pd.Timestamp(bars.trade_date[self.win._view_start]), day)
        self.assertEqual(pd.Timestamp(bars.trade_date[self.win._view_end]), day)
        # every bar in the window falls on that single date
        window = bars.trade_date[self.win._view_start : self.win._view_end + 1]
        self.assertTrue((pd.to_datetime(window) == day).all())

    def test_ensure_date_for_widens_the_window(self):
        self._set_window(self.dates[5], self.dates[5])  # a single later day
        earlier = self.win._window_bounds(self.win._bars, self.dates[0], self.dates[0])[0]
        moved = self.win._ensure_date_for(earlier)  # a bar before the window
        self.assertTrue(moved)
        self.assertLessEqual(
            pd.Timestamp(self.win._bars.trade_date[self.win._view_start]), self.dates[0]
        )


if __name__ == "__main__":
    unittest.main()
