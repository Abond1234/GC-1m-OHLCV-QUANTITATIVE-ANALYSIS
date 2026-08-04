"""Risk workspace contract: validation feedback, disabled states, persistence."""

from __future__ import annotations

import importlib.util
import os
import unittest

_HAS_QT = importlib.util.find_spec("PySide6") is not None
_HAS_PG = importlib.util.find_spec("pyqtgraph") is not None


@unittest.skipUnless(_HAS_QT and _HAS_PG, "PySide6/pyqtgraph not installed")
class RiskPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from src.app.ui import theme

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        theme.apply(cls.app, "dark")

    def _panel(self):
        from src.app.ui.exit_panel import ExitPanel

        return ExitPanel()

    def test_typed_out_of_range_flags_the_field(self):
        panel = self._panel()
        spin = panel.stop_value
        spin.lineEdit().setText("999")
        spin.lineEdit().textEdited.emit("999")
        self.assertTrue(bool(spin.property("invalid")))
        self.assertIn("Allowed range", spin.toolTip())
        spin.lineEdit().textEdited.emit("2.0")
        self.assertFalse(bool(spin.property("invalid")))
        self.assertNotIn("Allowed range", spin.toolTip())  # original tooltip returns

    def test_commit_clamps_and_clears_the_flag(self):
        panel = self._panel()
        spin = panel.stop_value
        spin.lineEdit().setText("999")
        spin.lineEdit().textEdited.emit("999")
        self.assertTrue(bool(spin.property("invalid")))
        spin.interpretText()
        spin.editingFinished.emit()
        self.assertFalse(bool(spin.property("invalid")))
        self.assertEqual(spin.value(), spin.maximum())  # clamped, never ignored

    def test_disabled_rows_dim_labels_until_enabled(self):
        panel = self._panel()
        label = panel._trail_form.labelForField(panel.trail_value)
        self.assertFalse(panel.trail_value.isEnabled())
        self.assertFalse(label.isEnabled())
        panel.trail_check.setChecked(True)
        self.assertTrue(panel.trail_value.isEnabled())
        self.assertTrue(label.isEnabled())

    def test_breakeven_after_target_shows_advisory(self):
        panel = self._panel()
        panel.be_check.setChecked(True)
        panel.be_trigger.setValue(3.0)  # target default is 2 R
        warn = panel._warn_labels["be"]
        self.assertTrue(warn.isVisibleTo(panel))
        self.assertIn("target", warn.text())
        panel.be_trigger.setValue(1.0)
        self.assertFalse(warn.isVisibleTo(panel))

    def test_trailing_wider_than_stop_shows_advisory(self):
        panel = self._panel()
        panel.trail_check.setChecked(True)
        panel.trail_value.setValue(5.0)  # stop default is 1.5 atr, same mode
        warn = panel._warn_labels["trail"]
        self.assertTrue(warn.isVisibleTo(panel))
        panel.trail_value.setValue(1.0)
        self.assertFalse(warn.isVisibleTo(panel))

    def test_settings_survive_hide_and_show(self):
        # The workspace collapse hides the widget; values must persist exactly.
        panel = self._panel()
        panel.stop_value.setValue(3.7)
        panel.hold_spin.setValue(240)
        panel.trail_check.setChecked(True)
        panel.hide()
        panel.show()
        cfg = panel.to_config()
        self.assertEqual(cfg.stop_value, 3.7)
        self.assertEqual(cfg.max_holding_minutes, 240)
        self.assertTrue(cfg.trailing_enabled)


if __name__ == "__main__":
    unittest.main()
