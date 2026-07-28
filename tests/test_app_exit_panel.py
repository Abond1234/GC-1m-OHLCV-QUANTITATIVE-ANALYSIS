"""ExitPanel round-trips every ExitConfig field. Skips without PySide6."""

from __future__ import annotations

import importlib.util
import os
import unittest

from src.app.sim.exit_config import ExitConfig, frozen_config

_HAS_QT = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(_HAS_QT, "PySide6 not installed")
class ExitPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _round_trip(self, cfg: ExitConfig):
        from src.app.ui.exit_panel import ExitPanel

        panel = ExitPanel()
        panel.from_config(cfg)
        got = panel.to_config()
        for field in (
            "stop_mode",
            "stop_value",
            "target_mode",
            "target_value",
            "trailing_enabled",
            "trailing_mode",
            "trailing_value",
            "breakeven_enabled",
            "breakeven_trigger_r",
            "breakeven_offset_ticks",
            "max_holding_minutes",
            "forced_exit_minute_ny",
            "stop_first",
        ):
            self.assertEqual(getattr(got, field), getattr(cfg, field), field)

    def test_round_trip_frozen(self):
        self._round_trip(frozen_config())

    def test_round_trip_creative(self):
        self._round_trip(
            ExitConfig(
                stop_mode="ticks",
                stop_value=40.0,
                target_mode="atr",
                target_value=3.0,
                trailing_enabled=True,
                trailing_mode="atr",
                trailing_value=2.0,
                breakeven_enabled=True,
                breakeven_trigger_r=1.5,
                breakeven_offset_ticks=2.0,
                max_holding_minutes=300,
                forced_exit_minute_ny=None,
                stop_first=False,
            )
        )

    def test_config_changed_emits(self):
        from src.app.ui.exit_panel import ExitPanel

        panel = ExitPanel()
        received = []
        panel.configChanged.connect(lambda c: received.append(c))
        panel.from_config(frozen_config())  # emits once
        self.assertTrue(received)
        self.assertIsInstance(received[-1], ExitConfig)


if __name__ == "__main__":
    unittest.main()
