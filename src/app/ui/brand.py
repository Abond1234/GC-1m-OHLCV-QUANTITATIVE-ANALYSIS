"""The Gold Quant brand mark, drawn in code.

A "GQ" monogram set inside a coin-like double ring, rendered with QPainter from
the theme palette so the mark stays in lockstep with the app's design system and
no binary image asset needs to live in the repository. One painter is the single
source of truth for every surface that shows the mark: the animated splash
screen, the runtime window/taskbar icon, and the packaged executable's icon
(``scripts/build_app.py`` renders it to a ``.ico`` at build time).

Everything scales off the target rect so the same drawing reads cleanly from a
16 px taskbar icon to a 200 px splash centrepiece; fine detail (the bezel ticks)
is added only at sizes where it can resolve.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui

from . import theme

# Metallic gold gradient stops (light catch at the top, deep bullion at the base).
_GOLD_STOPS = (
    (0.00, "#f6dd96"),
    (0.42, "#d9b45c"),
    (0.55, "#c39a3f"),
    (1.00, "#8a6a24"),
)

# Sizes baked into the runtime QIcon and the built executable's .ico.
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _gold_gradient(rect: QtCore.QRectF) -> QtGui.QLinearGradient:
    grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
    for pos, colour in _GOLD_STOPS:
        grad.setColorAt(pos, QtGui.QColor(colour))
    return grad


def paint_monogram(painter: QtGui.QPainter, rect: QtCore.QRectF, *, filled: bool = False) -> None:
    """Draw the GQ coin mark filling ``rect`` (assumed square-ish; centred).

    ``filled`` paints an opaque dark coin face first, so the mark stays legible
    on any backdrop - used for the window/taskbar/executable icon. The splash
    leaves it off and lets the mark sit directly on its own dark card.
    """

    p = theme.active()
    painter.save()
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

    side = min(rect.width(), rect.height())
    centre = rect.center()
    box = QtCore.QRectF(centre.x() - side / 2, centre.y() - side / 2, side, side)

    if filled:
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(theme.DARK.panel))
        r_face = side * 0.485
        painter.drawEllipse(centre, r_face, r_face)

    # Coin face: a barely-there radial lift off the background so the ring
    # reads as an object, not a line floating in space.
    face = QtGui.QRadialGradient(centre, side * 0.5)
    lift = QtGui.QColor(p.gold)
    lift.setAlpha(20)
    lift_edge = QtGui.QColor(p.gold)
    lift_edge.setAlpha(8)
    face.setColorAt(0.0, lift)
    face.setColorAt(0.75, lift_edge)
    face.setColorAt(1.0, QtCore.Qt.transparent)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(face)
    painter.drawEllipse(box)

    # Outer ring, drawn with the metal gradient so light appears to catch it.
    ring_pen = QtGui.QPen(QtGui.QBrush(_gold_gradient(box)), side * 0.030)
    painter.setPen(ring_pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    r_outer = side * 0.470
    painter.drawEllipse(centre, r_outer, r_outer)

    # Inner hairline ring.
    hair = QtGui.QColor(p.gold)
    hair.setAlpha(110)
    painter.setPen(QtGui.QPen(hair, max(1.0, side * 0.008)))
    r_inner = side * 0.408
    painter.drawEllipse(centre, r_inner, r_inner)

    # Bezel ticks between the rings, only at sizes where they resolve.
    if side >= 96:
        tick = QtGui.QColor(p.gold)
        tick.setAlpha(70)
        painter.setPen(QtGui.QPen(tick, max(1.0, side * 0.006)))
        for k in range(60):
            a = k * (2 * math.pi / 60)
            x0 = centre.x() + math.cos(a) * side * 0.425
            y0 = centre.y() + math.sin(a) * side * 0.425
            x1 = centre.x() + math.cos(a) * side * 0.452
            y1 = centre.y() + math.sin(a) * side * 0.452
            painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

    # The monogram itself, in metallic gradient serif.
    font = QtGui.QFont("Georgia")
    font.setBold(True)
    font.setPixelSize(round(side * 0.40))
    font.setLetterSpacing(QtGui.QFont.PercentageSpacing, 96)
    painter.setFont(font)
    text_box = QtCore.QRectF(box.x(), box.y() + side * 0.015, side, side)
    path = QtGui.QPainterPath()
    metrics = QtGui.QFontMetricsF(font)
    advance = metrics.horizontalAdvance("GQ")
    baseline_x = text_box.center().x() - advance / 2
    baseline_y = text_box.center().y() + metrics.capHeight() / 2
    path.addText(QtCore.QPointF(baseline_x, baseline_y), font, "GQ")
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QBrush(_gold_gradient(box)))
    painter.drawPath(path)

    painter.restore()


def logo_pixmap(
    size: int, device_pixel_ratio: float = 1.0, *, filled: bool = False
) -> QtGui.QPixmap:
    """The monogram on a transparent square, DPI-aware."""

    px = QtGui.QPixmap(round(size * device_pixel_ratio), round(size * device_pixel_ratio))
    px.setDevicePixelRatio(device_pixel_ratio)
    px.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(px)
    paint_monogram(painter, QtCore.QRectF(0, 0, size, size), filled=filled)
    painter.end()
    return px


def app_icon() -> QtGui.QIcon:
    """Multi-size window/taskbar icon built from the painted mark."""

    icon = QtGui.QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(logo_pixmap(size, filled=True))
    return icon


def write_ico(path) -> None:
    """Render the mark to a multi-resolution ``.ico`` for the packaged build.

    Prefers Pillow (present as a matplotlib dependency) for a true multi-size
    icon; falls back to Qt's single-image ICO writer if Pillow is unavailable.
    """

    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        logo_pixmap(256, filled=True).save(str(path), "ICO")
        return
    frames = []
    for size in ICON_SIZES:
        buf = QtCore.QBuffer()
        buf.open(QtCore.QIODevice.WriteOnly)
        logo_pixmap(size, filled=True).save(buf, "PNG")
        frames.append(Image.open(BytesIO(bytes(buf.data()))))
    largest = frames[-1]
    largest.save(path, format="ICO", sizes=[(s, s) for s in ICON_SIZES], append_images=frames[:-1])
