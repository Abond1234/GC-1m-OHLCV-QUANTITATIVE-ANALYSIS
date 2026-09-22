"""Application-wide protection against accidental mouse-wheel value changes."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class WheelInputGuard(QtCore.QObject):
    """Keep wheel gestures for scrolling, never for changing controls."""

    _BLOCKED = (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QTabBar)

    @classmethod
    def _blocked_owner(cls, watched):
        node = watched if isinstance(watched, QtWidgets.QWidget) else None
        while node is not None:
            if isinstance(node, cls._BLOCKED):
                return node
            if isinstance(node, QtWidgets.QAbstractScrollArea):
                return None
            node = node.parentWidget()
        return None

    @staticmethod
    def _scroll_area(widget):
        node = widget.parentWidget()
        while node is not None:
            if isinstance(node, QtWidgets.QAbstractScrollArea):
                return node
            node = node.parentWidget()
        return None

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        if event.type() != QtCore.QEvent.Type.Wheel:
            return super().eventFilter(watched, event)
        owner = self._blocked_owner(watched)
        if owner is None:
            return super().eventFilter(watched, event)

        scroll_area = self._scroll_area(owner)
        if scroll_area is not None:
            bar = scroll_area.verticalScrollBar()
            if bar.maximum() > bar.minimum():
                pixel_delta = event.pixelDelta().y()
                angle_delta = event.angleDelta().y()
                delta = pixel_delta or angle_delta
                if delta:
                    if pixel_delta:
                        distance = max(abs(pixel_delta), bar.singleStep())
                    else:
                        notches = max(1, round(abs(angle_delta) / 120))
                        distance = max(48, bar.singleStep() * 3) * notches
                    bar.setValue(bar.value() - distance if delta > 0 else bar.value() + distance)
        event.accept()
        return True
