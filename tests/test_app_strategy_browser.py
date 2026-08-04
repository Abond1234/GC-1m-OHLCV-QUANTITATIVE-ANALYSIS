"""Strategy browser: search, family groups, details, and the four row states."""

from __future__ import annotations

import importlib.util
import os
import unittest
from dataclasses import dataclass, field

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


@dataclass
class _Spec:
    name: str
    family: str
    thesis: str
    parameters: dict = field(default_factory=dict)
    looks_for: str = ""


_SPECS = [
    _Spec("NY breakout long", "Breakout", "Buy the NY range break.", {"n": 30}, "range break up"),
    _Spec("NY breakout short", "Breakout", "Sell the NY range break."),
    _Spec("London fade", "Mean reversion", "Fade the London extreme."),
]


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class StrategyBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.ui import theme

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        theme.apply(cls.app, "dark")

    def _browser(self):
        from src.app.ui.strategy_browser import StrategyBrowser

        b = StrategyBrowser()
        b.populate(_SPECS)
        return b

    def test_grouped_by_family_with_counts(self):
        b = self._browser()
        groups = {
            b.tree.topLevelItem(i).text(0): b.tree.topLevelItem(i).childCount()
            for i in range(b.tree.topLevelItemCount())
        }
        self.assertEqual(groups, {"Breakout (2)": 2, "Mean reversion (1)": 1})

    def test_search_filters_and_expands(self):
        b = self._browser()
        b.search.setText("fade")
        visible = [name for name, row in b._rows.items() if not row.isHidden()]
        self.assertEqual(visible, ["London fade"])
        self.assertTrue(b.tree.topLevelItem(1).isExpanded())
        self.assertTrue(b.tree.topLevelItem(0).isHidden())  # empty group hides
        b.search.setText("")
        self.assertFalse(any(row.isHidden() for row in b._rows.values()))

    def test_family_filter(self):
        b = self._browser()
        b.family_combo.setCurrentText("Breakout")
        self.assertFalse(b.tree.topLevelItem(0).isHidden())
        self.assertTrue(b.tree.topLevelItem(1).isHidden())

    def test_details_only_for_selection(self):
        b = self._browser()
        self.assertIn("Select a strategy", b.details.text())
        b.select("NY breakout long")
        self.assertEqual(b.current_name(), "NY breakout long")
        self.assertIn("Buy the NY range break.", b.details.text())
        self.assertIn("range break up", b.details.text())
        self.assertIn("n=30", b.details.text())

    def test_loading_state_round_trip(self):
        b = self._browser()
        b.set_loading("London fade")
        self.assertIn("computing", b._rows["London fade"].text(0))
        self.assertTrue(b._rows["London fade"].font(0).italic())
        b.select("London fade")
        self.assertEqual(b.current_name(), "London fade")  # suffix never leaks out
        b.set_loading(None)
        self.assertEqual(b._rows["London fade"].text(0), "London fade")
        self.assertFalse(b._rows["London fade"].font(0).italic())

    def test_error_state_and_clear(self):
        from src.app.ui import theme

        b = self._browser()
        b.set_error("London fade", "boom")
        row = b._rows["London fade"]
        self.assertEqual(row.toolTip(0), "boom")
        self.assertEqual(row.foreground(0).color().name(), theme.active().down.lower())
        b.clear_error("London fade")
        self.assertEqual(row.toolTip(0), "Fade the London extreme.")

    def test_disabled_state_shows_reason(self):
        b = self._browser()
        b.set_enabled_with_reason(False, "GC-only (research validity).")
        self.assertFalse(b.tree.isEnabled())
        self.assertIn("GC-only", b.reason.text())
        b.set_enabled_with_reason(True)
        self.assertTrue(b.tree.isEnabled())
        self.assertFalse(b.reason.isVisibleTo(b))


if __name__ == "__main__":
    unittest.main()
