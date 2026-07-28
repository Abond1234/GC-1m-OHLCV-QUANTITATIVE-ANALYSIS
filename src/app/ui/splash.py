"""Animated Gold Quant loading splash.

Shown while the Development+Validation dataset loads on the worker thread. The
static composition (background, coin monogram, wordmark) is painted once into a
DPI-aware pixmap; a 60 fps timer then overlays the living elements - a soft pulse
on the coin ring, a gold shimmer sweeping an indeterminate progress track, and a
status line with cycling ellipsis - so the app feels alive from the first frame
without repainting the whole scene each tick.

The splash closes with a short fade once the main window signals ``loadFinished``.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from . import brand, theme

_W, _H = 560, 380
_TRACK = QtCore.QRectF(150, 330, 260, 3)  # indeterminate progress line
_MONO = QtCore.QRectF(_W / 2 - 84, 62, 168, 168)  # coin monogram box


class GoldSplash(QtWidgets.QSplashScreen):
    """Frameless branded splash with subtle motion; ``set_status`` updates the line."""

    def __init__(self):
        self._palette = theme.DARK  # splash always matches the app's dark boot theme
        dpr = QtWidgets.QApplication.primaryScreen().devicePixelRatio() or 1.0
        super().__init__(self._base_pixmap(dpr))
        self._status = "Loading Development + Validation data"
        self._clock = QtCore.QElapsedTimer()
        self._clock.start()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)
        self._fade: QtCore.QPropertyAnimation | None = None

    # -- public API --------------------------------------------------------
    def set_status(self, text: str) -> None:
        self._status = text
        self.update()

    def finish_into(self, window: QtWidgets.QWidget) -> None:
        """Fade out over the opening window, then release the splash."""

        self._timer.stop()
        self._fade = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(260)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._fade.finished.connect(lambda: self.finish(window))
        self._fade.start()

    # -- painting ----------------------------------------------------------
    def _base_pixmap(self, dpr: float) -> QtGui.QPixmap:
        p = self._palette
        px = QtGui.QPixmap(round(_W * dpr), round(_H * dpr))
        px.setDevicePixelRatio(dpr)
        px.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(px)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

        # Card: rounded, near-black, with a faint radial warmth behind the coin.
        card = QtCore.QRectF(0, 0, _W, _H)
        path = QtGui.QPainterPath()
        path.addRoundedRect(card, 14, 14)
        painter.setClipPath(path)
        painter.fillRect(card, QtGui.QColor(p.bg))
        warm = QtGui.QRadialGradient(QtCore.QPointF(_W / 2, 150), 260)
        warm.setColorAt(0.0, QtGui.QColor(p.panel_alt))
        warm.setColorAt(1.0, QtGui.QColor(p.bg))
        painter.fillRect(card, warm)
        painter.setClipping(False)
        border = QtGui.QColor(p.gold)
        border.setAlpha(60)
        painter.setPen(QtGui.QPen(border, 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)

        theme_active = theme.active()
        try:
            theme._active = p  # paint the mark with splash palette regardless of app theme
            brand.paint_monogram(painter, _MONO)
        finally:
            theme._active = theme_active

        # Wordmark, generously letterspaced.
        word = QtGui.QFont("Segoe UI")
        word.setPixelSize(21)
        word.setWeight(QtGui.QFont.DemiBold)
        word.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 7.0)
        painter.setFont(word)
        painter.setPen(QtGui.QColor(p.text))
        painter.drawText(QtCore.QRectF(0, 252, _W + 7, 30), QtCore.Qt.AlignHCenter, "GOLD QUANT")
        cap = QtGui.QFont("Segoe UI")
        cap.setPixelSize(11)
        cap.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 1.5)
        painter.setFont(cap)
        painter.setPen(QtGui.QColor(p.text_dim))
        painter.drawText(
            QtCore.QRectF(0, 284, _W + 1.5, 18),
            QtCore.Qt.AlignHCenter,
            "GC TRADE SIMULATOR",
        )

        # Static progress track (the moving shimmer is painted per-frame).
        track = QtGui.QColor(p.border)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(_TRACK, 1.5, 1.5)
        painter.end()
        return px

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        p = self._palette
        t = self._clock.elapsed() / 1000.0
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Breathing glow on the coin ring.
        pulse = 0.5 + 0.5 * math.sin(t * 1.6)
        glow = QtGui.QColor(p.hook)
        glow.setAlpha(int(26 + 40 * pulse))
        r = _MONO.width() * 0.470
        painter.setPen(QtGui.QPen(glow, 5.5))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(_MONO.center(), r, r)

        # Gold shimmer sweeping the track (indeterminate).
        span = 84.0
        travel = _TRACK.width() + span
        head = _TRACK.x() - span + ((t * 170.0) % travel)
        seg = QtCore.QRectF(head, _TRACK.y() - 0.5, span, _TRACK.height() + 1)
        grad = QtGui.QLinearGradient(seg.topLeft(), seg.topRight())
        gold = QtGui.QColor(p.gold)
        gold.setAlpha(0)
        bright = QtGui.QColor(p.hook)
        grad.setColorAt(0.0, gold)
        grad.setColorAt(0.5, bright)
        grad.setColorAt(1.0, gold)
        painter.setClipRect(_TRACK.adjusted(0, -1, 0, 1))
        painter.fillRect(seg, grad)
        painter.setClipping(False)

        # Status line with cycling ellipsis.
        dots = "." * (1 + int(t * 2.2) % 3)
        painter.setPen(QtGui.QColor(p.text_faint))
        font = QtGui.QFont("Segoe UI")
        font.setPixelSize(11)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRectF(0, 344, _W, 20), QtCore.Qt.AlignHCenter, f"{self._status}{dots}"
        )
        painter.end()
