"""Mouse wheels scroll panels without mutating tabs or field values."""

from __future__ import annotations

import importlib.util
import os
import unittest

_HAS_QT = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(_HAS_QT, "PySide6 not installed")
class WheelInputGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    @staticmethod
    def _wheel(widget, delta=-120):
        from PySide6 import QtCore, QtGui, QtWidgets

        event = QtGui.QWheelEvent(
            QtCore.QPointF(5, 5),
            QtCore.QPointF(5, 5),
            QtCore.QPoint(),
            QtCore.QPoint(0, delta),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QtWidgets.QApplication.sendEvent(widget, event)

    def setUp(self):
        from PySide6 import QtWidgets

        from src.app.ui.input_guard import WheelInputGuard

        self.guard = WheelInputGuard(self.app)
        self.app.installEventFilter(self.guard)
        self.window = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.window)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(False)
        content = QtWidgets.QWidget()
        content.setMinimumSize(260, 700)
        content_layout = QtWidgets.QVBoxLayout(content)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(0, 100)
        self.spin.setValue(10)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["One", "Two", "Three"])
        content_layout.addWidget(self.spin)
        content_layout.addWidget(self.combo)
        content_layout.addStretch(1)
        self.scroll.setWidget(content)
        layout.addWidget(self.scroll)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(QtWidgets.QWidget(), "First")
        self.tabs.addTab(QtWidgets.QWidget(), "Second")
        layout.addWidget(self.tabs)
        self.window.resize(300, 260)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.app.removeEventFilter(self.guard)
        self.window.close()

    def test_wheel_scrolls_parent_without_changing_spin_or_combo(self):
        start_scroll = self.scroll.verticalScrollBar().value()
        self._wheel(self.spin)
        self._wheel(self.combo)
        self.assertEqual(self.spin.value(), 10)
        self.assertEqual(self.combo.currentIndex(), 0)
        self.assertGreater(self.scroll.verticalScrollBar().value(), start_scroll)

    def test_wheel_does_not_switch_tabs(self):
        self.tabs.setCurrentIndex(0)
        self._wheel(self.tabs.tabBar())
        self.assertEqual(self.tabs.currentIndex(), 0)


if __name__ == "__main__":
    unittest.main()
