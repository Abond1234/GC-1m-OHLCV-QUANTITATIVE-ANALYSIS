"""Indicator lifecycle: activate, hide, remove, panes, chips, persistence."""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class IndicatorLifecycleTests(unittest.TestCase):
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
        n = 40
        close = 1890 + np.cumsum(np.sin(np.arange(n)))
        ohlc = {
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(n, 100.0),
            "segment": np.ones(n, dtype=int),
        }
        c.set_view(ohlc, np.array([f"{i:02d}" for i in range(n)]), ViewMap.identity(0, n - 1))
        return c, ohlc

    def test_price_study_renders_and_removes(self):
        from src.app.analysis.indicators import compute

        c, ohlc = self._chart()
        series = compute("sma", {"n": 5}, ohlc)
        c.set_indicator(1, "price", series, "#c9a227")
        self.assertIn(1, c._indicator_items)
        c.remove_indicator(1)
        self.assertNotIn(1, c._indicator_items)

    def test_oscillator_pane_appears_only_while_needed(self):
        from src.app.analysis.indicators import compute

        c, ohlc = self._chart()
        self.assertFalse(c._osc.isVisible())
        c.set_indicator(2, "osc", compute("rsi", {"n": 14}, ohlc), "#c9a227", guides=(30, 70))
        self.assertTrue(c._osc.isVisible())
        c.set_indicator_visible(2, False)
        self.assertFalse(c._osc.isVisible())  # hidden study frees the pane
        c.set_indicator_visible(2, True)
        self.assertTrue(c._osc.isVisible())
        c.remove_indicator(2)
        self.assertFalse(c._osc.isVisible())

    def test_volume_toggle_never_touches_price_state(self):
        c, _ohlc = self._chart()
        x_before = c._price.vb.viewRange()[0]
        self.assertTrue(c.volume_visible())
        c.set_volume_visible(False)
        self.assertFalse(c._volume.isVisible())
        self.assertEqual(c._price.vb.viewRange()[0], x_before)
        c.set_volume_visible(True)
        self.assertTrue(c._volume.isVisible())

    def test_header_chips_follow_active_set(self):
        c, _ohlc = self._chart()
        toggles, removes = [], []
        chips = [{"id": 1, "label": "EMA 20", "color": "#c9a227", "visible": True}]
        c.set_header_chips(chips, toggles.append, removes.append)
        self.assertEqual(set(c._chip_widgets), {1})
        c.set_header_chips([], lambda _i: None, lambda _i: None)
        self.assertEqual(c._chip_widgets, {})

    def test_panel_add_hide_remove_and_persistence(self):
        from src.app.ui.indicators_panel import IndicatorsPanel

        panel = IndicatorsPanel()
        events = []
        panel.changed.connect(lambda: events.append(True))
        inst = panel.add_study("ema")
        panel.set_param(inst.id, "n", 34)
        panel.toggle_instance(inst.id)
        self.assertFalse(panel.instances[0].visible)
        specs = panel.to_dicts()
        self.assertEqual(specs[0]["key"], "ema")
        self.assertEqual(specs[0]["params"]["n"], 34.0)
        self.assertFalse(specs[0]["visible"])
        panel2 = IndicatorsPanel()
        panel2.load_dicts(specs)
        self.assertEqual(len(panel2.instances), 1)
        self.assertEqual(panel2.instances[0].display_label(), "EMA 34")
        self.assertFalse(panel2.instances[0].visible)
        panel.remove_instance(inst.id)
        self.assertEqual(panel.instances, [])
        self.assertGreaterEqual(len(events), 4)

    def test_search_filters_available_list(self):
        from src.app.ui.indicators_panel import IndicatorsPanel

        panel = IndicatorsPanel()
        panel.search.setText("boll")
        visible = [
            panel.available.item(i).text()
            for i in range(panel.available.count())
            if not panel.available.item(i).isHidden()
        ]
        self.assertEqual(visible, ["Bollinger Bands"])
        panel.search.setText("")
        hidden = [
            panel.available.item(i)
            for i in range(panel.available.count())
            if panel.available.item(i).isHidden()
        ]
        self.assertEqual(hidden, [])


if __name__ == "__main__":
    unittest.main()
