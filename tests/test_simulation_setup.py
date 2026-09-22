"""Prop-firm setup exposes independent optional consistency rules."""

from __future__ import annotations

import importlib.util
import os
import unittest

_HAS_QT = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(_HAS_QT, "PySide6 not installed")
class SimulationSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_consistency_is_off_by_default_and_independent_per_stage(self):
        from src.app.ui.simulation_setup import PropFirmPanel

        panel = PropFirmPanel()
        rules = panel.rules()
        self.assertIsNone(rules.phases[0].consistency_pct)
        self.assertIsNone(rules.funded_consistency_pct)
        self.assertFalse(panel.phase1_consistency.isEnabled())

        panel.steps.setCurrentIndex(1)
        panel.phase1_consistency_enabled.setChecked(True)
        panel.phase1_consistency.setValue(35.0)
        panel.phase2_consistency_enabled.setChecked(True)
        panel.phase2_consistency.setValue(40.0)
        panel.funded_consistency_enabled.setChecked(True)
        panel.funded_consistency.setValue(45.0)
        rules = panel.rules()

        self.assertEqual([phase.consistency_pct for phase in rules.phases], [35.0, 40.0])
        self.assertEqual(rules.funded_consistency_pct, 45.0)
        panel.close()
