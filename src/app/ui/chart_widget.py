"""pyqtgraph candlestick chart with VWAP, session shading, and trade drawing.

The chart renders one day (or a bar window) at a time using an integer x-axis of
bar indices, so non-trading gaps never stretch the candles; a custom bottom axis
maps indices back to New York time labels. It exposes a small, backend-agnostic
interface (set_view / add_vwap / mark_trades / draw_trade / clear_trades) so the
rest of the UI does not depend on pyqtgraph specifics. A left-click emits the
bar index under the cursor for free-play entry placement.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

# Warm, quiet palette (dark) consistent with the project's research aesthetic.
_BG = "#0d0c0a"
_UP = "#63b491"
_DOWN = "#d0705d"
_GOLD = "#cfa94e"
_INK_FAINT = "#726a58"
_VWAP = "#5aa0d6"


class CandlestickItem(pg.GraphicsObject):
    """Candles drawn once into a QPicture for fast repaint."""

    def __init__(self, x, o, h, low, c):
        super().__init__()
        self._x, self._o, self._h, self._l, self._c = x, o, h, low, c
        self.picture = QtGui.QPicture()
        self._draw()

    def _draw(self):
        painter = QtGui.QPainter(self.picture)
        half = 0.36
        up_pen = pg.mkPen(_UP)
        down_pen = pg.mkPen(_DOWN)
        up_brush = pg.mkBrush(_UP)
        down_brush = pg.mkBrush(_DOWN)
        for i in range(len(self._x)):
            x = float(self._x[i])
            is_up = self._c[i] >= self._o[i]
            painter.setPen(up_pen if is_up else down_pen)
            painter.setBrush(up_brush if is_up else down_brush)
            painter.drawLine(
                QtCore.QPointF(x, float(self._l[i])), QtCore.QPointF(x, float(self._h[i]))
            )
            top = float(max(self._o[i], self._c[i]))
            bot = float(min(self._o[i], self._c[i]))
            height = top - bot or 1e-6
            painter.drawRect(QtCore.QRectF(x - half, bot, 2 * half, height))
        painter.end()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class _TimeAxis(pg.AxisItem):
    """Bottom axis that maps bar index -> New York time label."""

    def __init__(self, labels: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels = labels

    def set_labels(self, labels: np.ndarray) -> None:
        self._labels = labels

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            i = int(round(v))
            if 0 <= i < len(self._labels):
                out.append(str(self._labels[i]))
            else:
                out.append("")
        return out


class ChartWidget(QtWidgets.QWidget):
    """Backend-agnostic chart surface used by the main window."""

    bar_clicked = QtCore.Signal(int)  # emits the global bar index under the cursor

    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background=_BG, foreground=_INK_FAINT)
        self._time_axis = _TimeAxis(np.array([]), orientation="bottom")
        self._layout = pg.GraphicsLayoutWidget()
        self._price = self._layout.addPlot(row=0, col=0, axisItems={"bottom": self._time_axis})
        self._volume = self._layout.addPlot(row=1, col=0)
        self._layout.ci.layout.setRowStretchFactor(0, 4)
        self._layout.ci.layout.setRowStretchFactor(1, 1)
        self._price.showGrid(x=False, y=True, alpha=0.15)
        self._volume.setXLink(self._price)
        self._volume.showAxis("bottom", False)

        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self._layout)

        self._view_start = 0  # global index of the first rendered bar
        self._trade_items: list = []
        self._price.scene().sigMouseClicked.connect(self._on_click)

    # -- rendering ---------------------------------------------------------
    def set_view(self, ohlc: dict, labels: np.ndarray, start_index: int) -> None:
        """Render a contiguous window. ``ohlc`` has open/high/low/close/volume arrays."""

        self._price.clear()
        self._volume.clear()
        self._trade_items.clear()
        self._view_start = int(start_index)
        n = len(ohlc["close"])
        x = np.arange(n)
        self._time_axis.set_labels(labels)
        self._price.addItem(
            CandlestickItem(x, ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        )
        vol = np.asarray(ohlc["volume"], dtype=float)
        self._volume.addItem(pg.BarGraphItem(x=x, height=vol, width=0.7, brush=_INK_FAINT))
        self._price.setLimits(xMin=-1, xMax=n)
        self._price.enableAutoRange()

    def add_vwap(self, vwap: np.ndarray) -> None:
        x = np.arange(len(vwap))
        self._price.addItem(pg.PlotDataItem(x, vwap, pen=pg.mkPen(_VWAP, width=1.4)))

    def shade_sessions(self, session_code: np.ndarray) -> None:
        """Light vertical shading for the New York execution window (code == 2)."""

        in_ny = session_code == 2
        if not in_ny.any():
            return
        edges = np.diff(in_ny.astype(int))
        starts = list(np.nonzero(edges == 1)[0] + 1)
        ends = list(np.nonzero(edges == -1)[0] + 1)
        if in_ny[0]:
            starts = [0, *starts]
        if in_ny[-1]:
            ends = [*ends, len(in_ny)]
        for a, b in zip(starts, ends, strict=False):
            region = pg.LinearRegionItem([a, b], movable=False, brush=pg.mkBrush(207, 169, 78, 18))
            region.setZValue(-10)
            self._price.addItem(region)

    def mark_trades(self, local_entries, y, directions, gross_r) -> None:
        """Scatter entry markers at entry price, coloured by outcome (local indices)."""

        brushes = [pg.mkBrush(_UP) if r > 0 else pg.mkBrush(_DOWN) for r in gross_r]
        symbols = ["t1" if d > 0 else "t" for d in directions]
        item = pg.ScatterPlotItem(
            x=list(local_entries),
            y=list(y),
            symbol=symbols,
            brush=brushes,
            size=11,
            pen=pg.mkPen("#1a1710"),
        )
        self._price.addItem(item)
        self._trade_items.append(item)

    def draw_trade(
        self, entry_local, exit_local, entry_price, stop_track, target_track, exit_price
    ):
        """Draw one trade: entry/exit markers, stop and target tracks."""

        held = np.arange(entry_local, entry_local + len(stop_track))
        if len(stop_track):
            self._price.addItem(
                pg.PlotDataItem(held, stop_track, pen=pg.mkPen(_DOWN, style=QtCore.Qt.DashLine))
            )
            self._price.addItem(
                pg.PlotDataItem(held, target_track, pen=pg.mkPen(_UP, style=QtCore.Qt.DashLine))
            )
        entry_marker = pg.ScatterPlotItem(
            x=[entry_local], y=[entry_price], symbol="o", size=12, brush=pg.mkBrush(_GOLD)
        )
        exit_marker = pg.ScatterPlotItem(
            x=[exit_local], y=[exit_price], symbol="x", size=13, pen=pg.mkPen(_GOLD, width=2)
        )
        self._price.addItem(entry_marker)
        self._price.addItem(exit_marker)
        self._trade_items += [entry_marker, exit_marker]

    def clear_trades(self) -> None:
        for item in self._trade_items:
            self._price.removeItem(item)
        self._trade_items.clear()

    def center_on(self, local_index: int) -> None:
        self._price.setXRange(local_index - 60, local_index + 60, padding=0)

    # -- interaction -------------------------------------------------------
    def _on_click(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        pos = event.scenePos()
        if not self._price.sceneBoundingRect().contains(pos):
            return
        mouse_point = self._price.vb.mapSceneToView(pos)
        local = int(round(mouse_point.x()))
        self.bar_clicked.emit(self._view_start + local)
