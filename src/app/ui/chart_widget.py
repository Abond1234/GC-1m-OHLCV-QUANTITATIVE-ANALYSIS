"""pyqtgraph candlestick chart with VWAP, session shading, and trade drawing.

The chart renders one day (or a bar window) at a time on an integer x-axis of bar
indices, so non-trading gaps never stretch the candles; a custom bottom axis maps
indices back to New York time labels. Candles are split at continuous-segment
breaks so nothing is drawn across a roll/data discontinuity. A crosshair with an
O/H/L/C readout follows the cursor, price is labelled on both edges, and a left
click emits the bar index under the cursor for free-play entry placement.

It exposes a small, backend-agnostic interface (set_view / add_vwap /
shade_sessions / mark_trades / draw_trade / clear_trades / center_on) so the rest
of the UI does not depend on pyqtgraph specifics. All colours come from ``theme``.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis.fib import FIB_POINTS
from ..analysis.fib import geometry as fib_geometry
from ..datalayer.timeframe import ViewMap
from . import theme


def _decimals_for(tick_size: float) -> int:
    """Number of decimals implied by a tick size (0.1 -> 1, 0.25 -> 2, 1 -> 0)."""

    text = f"{float(tick_size):.10f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text and text.split(".")[1] else 0


def candle_min_body(highs, lows) -> float:
    """Floor for a candle body so a doji renders visibly, not sub-pixel.

    A fraction of the window's typical (positive) high-low range, so dojis scale
    with the chart's price geometry at any zoom.
    """

    rng = np.asarray(highs, dtype=float) - np.asarray(lows, dtype=float)
    positive = rng[rng > 0]
    typical = float(np.median(positive)) if positive.size else theme.CANDLE_MIN_HEIGHT
    return max(theme.CANDLE_MIN_HEIGHT, 0.08 * typical)


class _PriceAxis(pg.AxisItem):
    """Price axis with TradingView interactions and tick-size precision.

    Stock pyqtgraph left-drags an axis into a PAN of the linked ViewBox; here a
    left-drag on the price axis expands or compresses the visible price range,
    anchored near where the drag started, and a double-click re-arms Y
    autorange (auto-fit). Wheel, right-drag, and the context menu keep their
    inherited behaviour. Tick labels format to the instrument's tick size so a
    price of 1890.4 is never shown as 1890.
    """

    def __init__(self, *args, tick_size: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._tick_size = tick_size
        self._decimals = _decimals_for(tick_size)

    def set_tick_size(self, tick_size: float) -> None:
        self._tick_size = tick_size
        self._decimals = _decimals_for(tick_size)
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):  # noqa: N802 - pyqtgraph override
        return [f"{v:,.{self._decimals}f}" for v in values]

    def mouseDragEvent(self, event):  # noqa: N802 - pyqtgraph override
        lv = self.linkedView()
        if lv is None:
            return super().mouseDragEvent(event)
        if lv.sceneBoundingRect().contains(event.buttonDownScenePos()):
            event.ignore()  # the press began inside the plot; not an axis drag
            return
        if event.button() != QtCore.Qt.LeftButton:
            return super().mouseDragEvent(event)
        if event.isStart():
            self._anchor_y = lv.mapSceneToView(event.buttonDownScenePos()).y()
        dy = event.pos().y() - event.lastPos().y()
        scale = float(np.exp(dy * 0.01))  # drag down = expand range (TradingView feel)
        anchor_y = getattr(self, "_anchor_y", lv.viewRect().center().y())
        lv.scaleBy(y=scale, center=pg.Point(lv.viewRect().center().x(), anchor_y))
        event.accept()

    def mouseClickEvent(self, event):  # noqa: N802 - pyqtgraph override
        lv = self.linkedView()
        if lv is not None and event.double():
            lv.enableAutoRange(axis=pg.ViewBox.YAxis)
            event.accept()
            return
        super().mouseClickEvent(event)


def _seg_runs(segment: np.ndarray) -> list[tuple[int, int]]:
    """Local (start, end) index runs of constant segment id within a window."""

    n = len(segment)
    if n == 0:
        return []
    change = np.nonzero(np.diff(segment) != 0)[0]
    starts = [0, *(change + 1)]
    ends = [*change, n - 1]
    return list(zip(starts, ends, strict=True))


class CandlestickItem(pg.GraphicsObject):
    """Candles drawn once into a QPicture for fast repaint."""

    def __init__(self, x, o, h, low, c):
        super().__init__()
        self._x, self._o, self._h, self._l, self._c = x, o, h, low, c
        self.picture = QtGui.QPicture()
        self._draw()
        # Rasterize once per zoom level: without this every overlapping repaint
        # (crosshair moves, drag frames, replay ticks) replays ~2 antialiased
        # primitives per bar from the QPicture command log.
        self.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)

    def _draw(self):
        p = theme.active()
        painter = QtGui.QPainter(self.picture)
        half = theme.CANDLE_HALF_WIDTH
        # Cosmetic pens keep wicks and body edges a crisp 1px at every zoom level
        # instead of scaling into fat, fuzzy, off-centre lines.
        up_pen = pg.mkPen(p.up, width=1, cosmetic=True)
        down_pen = pg.mkPen(p.down, width=1, cosmetic=True)
        up_brush = pg.mkBrush(p.up)
        down_brush = pg.mkBrush(p.down)
        # A doji's body is floored so it renders as a visible bar, not a sliver.
        min_body = candle_min_body(self._h, self._l)
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
            height = top - bot
            if height < min_body:  # centre a minimum body on the open/close level
                mid = 0.5 * (top + bot)
                bot, height = mid - 0.5 * min_body, min_body
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
    bar_hovered = QtCore.Signal(int)  # hovered displayed bar's last 1m index (crosshair)
    drawing_placed = QtCore.Signal()  # a one-shot draw mode finished placing

    def __init__(self, parent=None):
        super().__init__(parent)
        p = theme.active()
        pg.setConfigOptions(antialias=True, background=p.bg, foreground=p.text_faint)
        self._time_axis = _TimeAxis(np.array([]), orientation="bottom")
        self._tick_size = 0.1
        self._decimals = _decimals_for(self._tick_size)
        self._layout = pg.GraphicsLayoutWidget()
        # One primary price axis on the RIGHT (TradingView convention), aligned
        # with the volume pane's right axis below it; no duplicate left scale.
        self._price = self._layout.addPlot(
            row=0,
            col=0,
            axisItems={
                "bottom": self._time_axis,
                "right": _PriceAxis(orientation="right", tick_size=self._tick_size),
            },
        )
        self._price.showAxis("right", True)
        self._price.showAxis("left", False)
        self._price_axis = self._price.getAxis("right")
        self._volume = self._layout.addPlot(row=1, col=0)
        self._volume.showAxis("right", True)
        self._volume.showAxis("left", False)
        # Oscillator pane for bounded/derived studies (RSI, ATR, OBV, ...);
        # zero-height until an active study needs it.
        self._osc = self._layout.addPlot(row=2, col=0)
        self._osc.showAxis("right", True)
        self._osc.showAxis("left", False)
        self._osc.showAxis("bottom", False)
        self._osc.showGrid(x=False, y=True, alpha=0.12)
        self._osc.setXLink(self._price)
        self._layout.ci.layout.setRowStretchFactor(0, 5)
        self._layout.ci.layout.setRowStretchFactor(1, 1)
        self._layout.ci.layout.setRowStretchFactor(2, 0)
        self._layout.ci.layout.setRowMaximumHeight(2, 0)
        self._osc.setVisible(False)
        self._price.showGrid(x=False, y=True, alpha=0.12)
        self._legend = self._price.addLegend(offset=(10, 8), labelTextColor=p.text_dim)
        self._volume.setXLink(self._price)
        self._volume.showAxis("bottom", False)
        self._volume.setLabel("right", "vol")
        self._volume_visible = True

        # Chart header: the O/H/L/C readout (a plain QLabel - updating a plot
        # title forces a graphics-layout pass per mouse move, a label repaint
        # does not), an active-indicator legend strip, and compact Undo / Redo /
        # Clear icons. A fixed layout row, so the controls are reachable at any
        # chart width and can never cover the readout.
        self._readout_label = QtWidgets.QLabel(" ")
        self._readout_label.setTextFormat(QtCore.Qt.RichText)
        self._readout_label.setFixedHeight(22)
        self._readout_label.setContentsMargins(8, 0, 8, 0)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 4, 0)
        header.setSpacing(4)
        header.addWidget(self._readout_label)
        self._legend_bar = QtWidgets.QHBoxLayout()  # active-indicator chips live here
        self._legend_bar.setSpacing(4)
        header.addLayout(self._legend_bar)
        header.addStretch(1)
        self._header_buttons: dict[str, QtWidgets.QToolButton] = {}
        for name, tip, slot in (
            ("Undo", "Remove the last drawing or measurement (Ctrl+Z).", self.undo_drawing),
            ("Redo", "Restore the last undone drawing (Ctrl+Y).", self.redo_drawing),
            ("Clear", "Remove every drawing on this view.", self.clear_drawings),
        ):
            btn = QtWidgets.QToolButton()
            btn.setText(name)
            btn.setToolTip(tip)
            btn.setAutoRaise(True)
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            btn.clicked.connect(slot)
            header.addWidget(btn)
            self._header_buttons[name] = btn

        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addLayout(header)
        box.addWidget(self._layout)

        self._view_start = 0  # global index of the first rendered 1m bar
        self._view_map: ViewMap = ViewMap.identity(0, 0)  # replaced by set_view
        self._minute_close: np.ndarray = np.array([])  # 1m closes for the window
        self._trade_items: list = []
        self._whatif_items: list = []  # alternative-exit overlays, cleared separately
        self._level_lines: list = []  # draggable stop/target lines (kept across path redraws)
        self._level_proxies: list = []
        self._replay_items: list = []  # bar-by-bar animation items
        self._on_level_changed = None
        self._ohlc: dict | None = None  # current window arrays, for the crosshair readout
        self._labels: np.ndarray = np.array([])
        self._hover_labels: np.ndarray = np.array([])  # exact date/time per bar
        self._last_price: float | None = None  # newest revealed close (scale pill)
        self._last_readout_i = -1
        self._vwap_items: dict[str, pg.PlotDataItem] = {}  # kind -> overlay line
        self._indicator_items: dict[int, tuple[str, list]] = {}  # id -> (pane, items)
        self._indicator_visible: dict[int, bool] = {}  # declared visibility per id
        self._chip_widgets: dict[int, QtWidgets.QWidget] = {}  # id -> header chip
        self._compare_item = None  # normalized comparison overlay line
        self._session_regions: list = []  # NY-session shading regions
        # Persistent slots for the one detailed trade and its excursion ribbon,
        # so a 60 Hz stop/target drag updates data in place instead of tearing
        # down and rebuilding the whole overlay layer each event.
        self._active_trade_items: dict | None = None
        self._excursion_items: dict | None = None
        self._curtains: list = []  # bar-replay covers, rebuilt per view
        # User drawings (levels/trendlines/time markers). Live items for the
        # current day plus a per-day store so annotations survive date switches.
        self._draw_mode: str | None = None
        self._measure_anchor: tuple[float, float] | None = None  # Shift+drag origin
        self._measure_preview: list | None = None  # live readout items while dragging
        self._drawing_items: list[tuple[str, object]] = []  # (kind, items), creation order
        self._drawing_store: list[dict] = []  # global-coordinate specs not currently live
        self._redo_specs: list[dict] = []  # undone drawings (global specs), newest last
        self._trend_anchor: tuple[float, float] | None = None
        self._trend_anchor_dot = None
        self._multi_anchors: list[tuple[float, float]] = []  # 3-point tool clicks
        self._multi_dots: list = []
        self._trend_preview = None  # rubber-band line while placing a trendline
        self._draw_press_view: tuple[float, float] | None = None
        self._draw_press_pixel = None
        # Armed draw tools must own the whole press-drag-release gesture:
        # otherwise the tiniest drag becomes a chart pan and nothing is placed.
        self._layout.viewport().installEventFilter(self)

        # Crosshair (hidden until the cursor is over the plot).
        pen = pg.mkPen(p.text_faint, width=1, style=QtCore.Qt.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        for line in (self._vline, self._hline):
            line.setVisible(False)
            line.setZValue(20)
            self._price.addItem(line, ignoreBounds=True)
        # Axis badges: exact time (bottom edge) and tick-formatted price (right
        # edge) that follow the crosshair, TradingView-style. The last price is
        # a dotted line with a pill on the price scale, rebuilt per view; its
        # label repositions itself on view changes (never via a raw
        # sigRangeChanged lambda, which would fire into destroyed graphics
        # during widget teardown and corrupt the heap).
        self._x_badge = self._make_badge(anchor=(0.5, 1.0))
        self._y_badge = self._make_badge(anchor=(1.0, 0.5))
        self._last_line = None  # built per set_view

        self._price.scene().sigMouseClicked.connect(self._on_click)
        self._mouse_proxy = pg.SignalProxy(
            self._price.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

    def _make_badge(self, anchor) -> pg.TextItem:
        p = theme.active()
        badge = pg.TextItem(
            "", color=p.bg, anchor=anchor, fill=pg.mkBrush(p.gold), border=pg.mkPen(None)
        )
        badge.setVisible(False)
        badge.setZValue(25)
        self._price.addItem(badge, ignoreBounds=True)
        return badge

    def _restyle_badges(self) -> None:
        p = theme.active()
        for badge in (self._x_badge, self._y_badge):
            badge.setColor(p.bg)
            badge.fill = pg.mkBrush(p.gold)

    def apply_theme(self) -> None:
        """Repaint the chart surface with the active palette (light/dark switch).

        Items rebuilt by ``set_view`` pick the palette up on the next render; this
        restyles the pieces created once in the constructor (crosshair, legend,
        axes) so nothing keeps the previous theme's colours.
        """

        p = theme.active()
        self._layout.setBackground(p.bg)
        cross = pg.mkPen(p.text_faint, width=1, style=QtCore.Qt.DashLine)
        self._vline.setPen(cross)
        self._hline.setPen(cross)
        self._restyle_badges()
        self._legend.setLabelTextColor(p.text_dim)
        for plot in (self._price, self._volume, self._osc):
            for name in ("right", "bottom"):
                axis = plot.getAxis(name)
                if axis is not None:
                    axis.setPen(pg.mkPen(p.text_faint))
                    axis.setTextPen(pg.mkPen(p.text_dim))

    def set_tick_size(self, tick_size: float) -> None:
        """Format the axis, crosshair badge, and readout to this instrument's tick."""

        self._tick_size = float(tick_size)
        self._decimals = _decimals_for(tick_size)
        self._price_axis.set_tick_size(tick_size)
        if self._last_line is not None:
            self._last_line.label.setFormat(f"{{value:,.{self._decimals}f}}")

    # -- rendering ---------------------------------------------------------
    def set_view(
        self,
        ohlc: dict,
        labels: np.ndarray,
        view_map: ViewMap,
        *,
        minute_close: np.ndarray | None = None,
        hover_labels: np.ndarray | None = None,
    ) -> None:
        """Render a window of displayed bars described by ``view_map``.

        ``ohlc`` holds one entry per displayed bar (1m bars or aggregated
        buckets) plus optionally ``segment``; candles are split at segment
        breaks so none is drawn across a roll/data discontinuity. ``view_map``
        converts between global 1-minute indices - the simulation truth every
        caller keeps speaking - and displayed x positions. ``minute_close`` is
        the raw 1m close slice for ``[view_start, view_end]``, used so the
        replay marker can ride actual minute prices at any timeframe; at 1m it
        simply equals ``ohlc["close"]``. ``hover_labels`` are exact per-bar
        date/time strings for the crosshair pill (falls back to the axis
        labels when omitted).
        """

        p = theme.active()
        self._stash_live_drawings()  # keep annotations for when this window returns
        self._price.clear()
        self._volume.clear()
        self._osc.clear()
        self._trade_items.clear()
        self._whatif_items.clear()
        self._level_lines.clear()  # removed by _price.clear(); drop stale references
        self._level_proxies.clear()
        self._replay_items.clear()
        self._vwap_items.clear()
        self._indicator_items.clear()  # graphics died with the plot clears
        self._indicator_visible.clear()
        self._compare_item = None  # removed by _price.clear(); redrawn by the caller
        self._session_regions.clear()
        self._active_trade_items = None
        self._excursion_items = None
        self._drawing_items.clear()
        self._trend_anchor = None
        self._trend_anchor_dot = None
        self._multi_anchors.clear()
        self._multi_dots.clear()  # their graphics died with the plot clear
        self._trend_preview = None
        self._draw_press_view = None
        self._draw_press_pixel = None
        self._measure_anchor = None
        self._measure_preview = None  # its graphics died with the plot clear
        self._last_readout_i = -1
        for line in (self._vline, self._hline):
            line.setVisible(False)
            self._price.addItem(line, ignoreBounds=True)
        for badge in (self._x_badge, self._y_badge):
            badge.setVisible(False)
            self._price.addItem(badge, ignoreBounds=True)
        self._view_map = view_map
        self._view_start = view_map.view_start
        self._ohlc = ohlc
        self._minute_close = (
            np.asarray(minute_close) if minute_close is not None else np.asarray(ohlc["close"])
        )
        self._labels = np.asarray(labels)
        self._hover_labels = np.asarray(hover_labels) if hover_labels is not None else self._labels
        n = len(ohlc["close"])
        x = np.arange(n)
        self._time_axis.set_labels(labels)

        segment = ohlc.get("segment")
        runs = _seg_runs(np.asarray(segment)) if segment is not None else [(0, n - 1)]
        for a, b in runs:
            sl = slice(a, b + 1)
            self._price.addItem(
                CandlestickItem(
                    x[sl], ohlc["open"][sl], ohlc["high"][sl], ohlc["low"][sl], ohlc["close"][sl]
                )
            )
        vol = np.asarray(ohlc["volume"], dtype=float)
        self._volume.addItem(pg.BarGraphItem(x=x, height=vol, width=0.7, brush=p.text_faint))
        self._price.setLimits(xMin=-1, xMax=n)
        self._price.enableAutoRange()
        self._last_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(p.gold, width=1, style=QtCore.Qt.DotLine),
            label=f"{{value:,.{self._decimals}f}}",
            labelOpts={
                "position": 0.99,
                "anchors": [(1.0, 0.5), (1.0, 0.5)],
                "color": p.bg,
                "fill": pg.mkBrush(p.gold),
                "movable": False,
            },
        )
        self._last_line.setZValue(9)  # above the reveal curtain, under the crosshair
        self._price.addItem(self._last_line, ignoreBounds=True)
        self._set_last_price(float(ohlc["close"][-1]) if n else None)

        # Reveal curtains: opaque covers that hide every bar after the replay
        # clock, so an animated trade unfolds candle by candle with the future
        # genuinely invisible (TradingView-style bar replay). Hidden until a
        # replay starts; z sits above data/overlays, below the crosshair, the
        # draggable levels, and the replay marker.
        self._curtains = []
        for plot in (self._price, self._volume, self._osc):
            curtain = pg.LinearRegionItem(
                values=(n + 1, n + 2),
                movable=False,
                brush=pg.mkBrush(p.bg),
                pen=pg.mkPen(None),
            )
            curtain.setZValue(8)
            curtain.setVisible(False)
            plot.addItem(curtain, ignoreBounds=True)
            self._curtains.append(curtain)

        # Bring back annotations whose global coordinates intersect this window.
        self._restore_drawings()

    # -- user drawings (levels / trendlines / time markers) -----------------
    def set_draw_mode(self, mode: str | None) -> None:
        """Arm a one-shot drawing tool: "hline", "vline", "trend", or None."""

        self._draw_mode = mode
        self._clear_trend_anchor()
        self._remove_drag_preview()
        self._draw_press_view = None
        self._draw_press_pixel = None

    def eventFilter(self, obj, event):  # noqa: N802 - Qt override
        """Own the mouse while a draw tool is armed.

        Without this the tiniest press-move becomes a ViewBox pan and nothing is
        placed. Levels/markers place on release; a trendline supports both the
        TradingView press-drag-release gesture (with a live rubber band) and the
        two-click flow. Right-click cancels the tool.
        """

        if obj is not self._layout.viewport():
            return super().eventFilter(obj, event)
        if self._handle_measure_gesture(event):
            return True
        if self._draw_mode is None:
            return super().eventFilter(obj, event)
        etype = event.type()
        if etype == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.RightButton:
                self.set_draw_mode(None)
                self.drawing_placed.emit()  # hand the toolbar back to the cursor
                return True
            if event.button() == QtCore.Qt.LeftButton:
                pressed_view = self._map_pixel_to_view(event.position())
                if pressed_view is None:
                    return False  # axis strip or volume pane: let scaling/panning work
                self._draw_press_view = pressed_view
                self._draw_press_pixel = event.position()
                return True
        elif etype == QtCore.QEvent.MouseMove:
            current = self._map_pixel_to_view(event.position())
            if current is not None and self._draw_mode in self._TWO_POINT:
                if self._draw_press_view is not None:
                    self._update_drag_preview(self._draw_mode, self._draw_press_view, current)
                    return True
                if self._trend_anchor is not None:
                    self._update_drag_preview(self._draw_mode, self._trend_anchor, current)
            return False  # plain hovers keep feeding the crosshair
        elif etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            start = self._draw_press_view
            start_pixel = self._draw_press_pixel
            self._draw_press_view = None
            self._draw_press_pixel = None
            end = self._map_pixel_to_view(event.position())
            if end is None:
                self._remove_drag_preview()
                return True
            if self._draw_mode in self._TWO_POINT and start is not None and start_pixel is not None:
                moved = (event.position() - start_pixel).manhattanLength()
                if moved > 6:  # a real drag: place the two-point shape start -> end
                    self._remove_drag_preview()
                    self._clear_trend_anchor()
                    self._create_drawing({"kind": self._draw_mode, "p1": start, "p2": end})
                    self._redo_specs.clear()  # a fresh drawing invalidates redo history
                    self.drawing_placed.emit()
                    return True
            self.place_drawing(self._draw_mode, end[0], end[1])
            return True
        elif etype == QtCore.QEvent.ContextMenu:
            return True  # no pan/zoom menu while a tool is armed
        return super().eventFilter(obj, event)

    # -- Shift+Left measurement mode -----------------------------------------
    def _handle_measure_gesture(self, event) -> bool:
        """Own the mouse for Shift+Left drag measuring; True when consumed.

        Press anchors, drag shows the live readout, release commits the
        measurement as a drawing (so Undo removes it and it survives view
        switches), Esc or right-click cancels. All in data coordinates, so the
        reported values are identical at any zoom.
        """

        etype = event.type()
        if etype == QtCore.QEvent.MouseButtonPress:
            if (
                event.button() == QtCore.Qt.LeftButton
                and event.modifiers() & QtCore.Qt.ShiftModifier
                and self._ohlc is not None
            ):
                anchor = self._map_pixel_to_view(event.position())
                if anchor is not None:
                    self._measure_anchor = anchor
                    return True
            if self._measure_anchor is not None and event.button() == QtCore.Qt.RightButton:
                self.cancel_measurement()
                return True
            return False
        if self._measure_anchor is None:
            return False
        if etype == QtCore.QEvent.MouseMove:
            current = self._map_pixel_to_view(event.position())
            if current is not None:
                self._update_measure_preview(self._measure_anchor, current)
            return True
        if etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            anchor = self._measure_anchor
            end = self._map_pixel_to_view(event.position())
            self.cancel_measurement()
            if end is not None and abs(end[0] - anchor[0]) + abs(end[1] - anchor[1]) > 0:
                self._create_drawing({"kind": "measure", "p1": anchor, "p2": end})
                self._redo_specs.clear()
            return True
        return False

    def cancel_measurement(self) -> None:
        """Drop any in-progress measurement drag (Esc / right-click)."""

        self._measure_anchor = None
        if self._measure_preview is not None:
            for item in self._measure_preview:
                self._price.removeItem(item)
            self._measure_preview = None

    def measuring(self) -> bool:
        return self._measure_anchor is not None

    def _update_measure_preview(self, p1, p2) -> None:
        if self._measure_preview is not None:
            for item in self._measure_preview:
                self._price.removeItem(item)
        self._measure_preview = self._build_measure_items(p1, p2)
        for item in self._measure_preview:
            item.setZValue(14)
            self._price.addItem(item, ignoreBounds=True)

    def _build_measure_items(self, p1, p2) -> list:
        """Zone + diagonal + readout for a measurement (data coordinates)."""

        from ..analysis.measure import format_measurement, measure

        p = theme.active()
        up = p2[1] >= p1[1]
        colour = QtGui.QColor(p.target if up else p.stop)
        rect = QtWidgets.QGraphicsRectItem(
            QtCore.QRectF(QtCore.QPointF(p1[0], p1[1]), QtCore.QPointF(p2[0], p2[1])).normalized()
        )
        pen = pg.mkPen(colour, width=1.2)
        pen.setCosmetic(True)
        rect.setPen(pen)
        fill = QtGui.QColor(colour)
        fill.setAlpha(26)
        rect.setBrush(fill)
        line = pg.PlotDataItem(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            pen=pg.mkPen(colour, width=1.4, style=QtCore.Qt.DashLine),
        )
        vm = self._view_map
        minutes = abs(vm.local_f_to_global(p2[0]) - vm.local_f_to_global(p1[0]))
        bars = round(abs(p2[0] - p1[0]))
        m = measure(p1[1], p2[1], self._tick_size, bars, minutes)
        text = pg.TextItem(
            format_measurement(m, self._decimals),
            color=p.bg,
            anchor=(0.5, 1.1 if up else -0.1),
            fill=pg.mkBrush(colour),
            border=pg.mkPen(None),
        )
        text.setPos(0.5 * (p1[0] + p2[0]), max(p1[1], p2[1]) if up else min(p1[1], p2[1]))
        return [rect, line, text]

    def _map_pixel_to_view(self, pos) -> tuple[float, float] | None:
        """Viewport pixel -> (bar index, price), or None outside the price box."""

        scene_pos = self._layout.mapToScene(QtCore.QPoint(round(pos.x()), round(pos.y())))
        if not self._price.vb.sceneBoundingRect().contains(scene_pos):
            return None
        view_pos = self._price.vb.mapSceneToView(scene_pos)
        return float(view_pos.x()), float(view_pos.y())

    def _update_drag_preview(self, kind: str, p1, p2) -> None:
        if kind == "measure":  # the measure preview IS the live readout
            self._update_measure_preview(p1, p2)
            return
        p = theme.active()
        if self._trend_preview is None:
            if kind == "rect":
                preview = QtWidgets.QGraphicsRectItem()
                pen = pg.mkPen(p.gold, width=1.2, style=QtCore.Qt.DashLine)
                pen.setCosmetic(True)
                preview.setPen(pen)
                fill = QtGui.QColor(p.gold)
                fill.setAlpha(24)
                preview.setBrush(fill)
            else:
                preview = pg.PlotDataItem(pen=pg.mkPen(p.gold, width=1.2, style=QtCore.Qt.DashLine))
            preview.setZValue(12)
            self._price.addItem(preview, ignoreBounds=True)
            self._trend_preview = preview
        if isinstance(self._trend_preview, QtWidgets.QGraphicsRectItem):
            self._trend_preview.setRect(
                QtCore.QRectF(
                    QtCore.QPointF(p1[0], p1[1]), QtCore.QPointF(p2[0], p2[1])
                ).normalized()
            )
        else:
            self._trend_preview.setData([p1[0], p2[0]], [p1[1], p2[1]])

    def _remove_drag_preview(self) -> None:
        if self._trend_preview is not None:
            self._price.removeItem(self._trend_preview)
            self._trend_preview = None
        if self._measure_preview is not None:
            for item in self._measure_preview:
                self._price.removeItem(item)
            self._measure_preview = None

    def place_drawing(self, kind: str, x: float, y: float) -> None:
        """Create a drawing at view coordinates (bar index, price)."""

        if self._ohlc is None:
            return
        n = len(self._ohlc["close"])
        x = max(0.0, min(float(x), float(n - 1)))
        if kind in self._FIB3:
            # Three-point tools place by three clicks; dots mark the anchors.
            self._multi_anchors.append((x, float(y)))
            p = theme.active()
            dot = pg.ScatterPlotItem(
                x=[x], y=[float(y)], symbol="+", size=12, pen=pg.mkPen(p.gold, width=1.5)
            )
            self._price.addItem(dot, ignoreBounds=True)
            self._multi_dots.append(dot)
            if len(self._multi_anchors) < 3:
                return
            spec = {"kind": kind, "points": list(self._multi_anchors)}
            self._clear_trend_anchor()
        elif kind in self._TWO_POINT:
            if self._trend_anchor is None:
                self._trend_anchor = (x, float(y))
                p = theme.active()
                self._trend_anchor_dot = pg.ScatterPlotItem(
                    x=[x], y=[float(y)], symbol="+", size=12, pen=pg.mkPen(p.gold, width=1.5)
                )
                self._price.addItem(self._trend_anchor_dot, ignoreBounds=True)
                return  # second click (or drag release) completes the shape
            spec = {"kind": kind, "p1": self._trend_anchor, "p2": (x, float(y))}
            self._clear_trend_anchor()
            self._remove_drag_preview()
        elif kind == "hline":
            spec = {"kind": "hline", "y": float(y)}
        elif kind == "vline":
            spec = {"kind": "vline", "x": x}
        elif kind in ("hray", "cross"):
            spec = {"kind": kind, "x": x, "y": float(y)}
        else:
            return
        self._create_drawing(spec)
        self._redo_specs.clear()  # a fresh drawing invalidates redo history
        self.drawing_placed.emit()

    # Drawing kind families. Two-point kinds share the anchor/drag gesture;
    # the line family shares endpoint handles and differs only in extension
    # and labelling; the position kinds are the long/short R:R tools.
    _LINE_FAMILY = ("trend", "ray", "extline", "info", "angle")
    _POSITION_KINDS = ("longpos", "shortpos")
    _FIB_KINDS = tuple(FIB_POINTS)
    _FIB2 = tuple(k for k, n in FIB_POINTS.items() if n == 2)
    _FIB3 = tuple(k for k, n in FIB_POINTS.items() if n == 3)
    _TWO_POINT = (*_LINE_FAMILY, *_POSITION_KINDS, "rect", "measure", *_FIB2)
    _EXTEND = 100_000.0  # parametric reach for rays/extended lines (clipped by Qt)

    def _handle(self, pos, color=None) -> pg.TargetItem:
        p = theme.active()
        return pg.TargetItem(
            pos=pos,
            size=9,
            movable=True,
            pen=pg.mkPen(color or p.gold, width=1.2),
            hoverPen=pg.mkPen(p.hook, width=2.0),
            brush=pg.mkBrush(0, 0, 0, 0),
        )

    @staticmethod
    def _line_label_text(kind: str, a, b, tick_size: float, decimals: int, vm) -> str:
        """Live annotation for info lines and trend angles (data-coordinate).

        Static on purpose: it is called from signal closures attached to
        graphics items, and those closures must never capture ``self`` - a
        chart -> item -> closure -> chart cycle makes Python's GC destroy the
        Qt objects in arbitrary order and corrupts the heap at teardown.
        """

        import math

        from ..analysis.measure import format_measurement, measure

        if kind == "info":
            minutes = abs(vm.local_f_to_global(b[0]) - vm.local_f_to_global(a[0]))
            m = measure(a[1], b[1], tick_size, round(abs(b[0] - a[0])), minutes)
            return format_measurement(m, decimals)
        dx, dy = b[0] - a[0], b[1] - a[1]
        angle = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0.0
        slope = dy / dx if dx else float("inf")
        slope_text = f"{slope:+.2f} pts/bar" if math.isfinite(slope) else "vertical"
        return f"{angle:+.1f} deg ({slope_text})"

    def _create_drawing(self, spec: dict) -> None:
        p = theme.active()
        kind = spec["kind"]
        if kind == "measure":
            items = self._build_measure_items(tuple(spec["p1"]), tuple(spec["p2"]))
            # The zone rect remembers its local anchors for serialization
            # (measurements have no drag handles to read back).
            items[0].setData(0, (tuple(spec["p1"]), tuple(spec["p2"])))
        elif kind == "hline":
            items = [
                pg.InfiniteLine(
                    pos=float(spec["y"]),
                    angle=0,
                    movable=True,
                    pen=pg.mkPen(p.vwap_rolling, width=1.2),
                    hoverPen=pg.mkPen(p.vwap_rolling, width=2.2),
                    label=f"{{value:.{self._decimals}f}}",
                    labelOpts={"position": 0.97, "color": p.vwap_rolling, "movable": True},
                )
            ]
        elif kind == "vline":
            items = [
                pg.InfiniteLine(
                    pos=float(spec["x"]),
                    angle=90,
                    movable=True,
                    pen=pg.mkPen(p.text_faint, width=1.2, style=QtCore.Qt.DashLine),
                    hoverPen=pg.mkPen(p.text_dim, width=2.2),
                )
            ]
        elif kind in self._LINE_FAMILY:
            # A data-coordinate line with two TargetItem endpoint handles.
            # Everything lives in view coordinates with cosmetic pens, so the
            # drawing is exact under any zoom (LineSegmentROI is not). Rays and
            # extended lines reach a huge parametric distance instead of
            # re-projecting on every range change; info lines and trend angles
            # carry a live label that follows the second handle.
            line = pg.PlotDataItem(pen=pg.mkPen(p.gold, width=1.6))
            label = None
            if kind in ("info", "angle"):
                label = pg.TextItem("", color=p.bg, anchor=(0.5, 1.2), fill=pg.mkBrush(p.gold))
            handles = [self._handle(spec[key]) for key in ("p1", "p2")]

            # NOTE: everything the closure needs is bound as a default - it
            # must not capture ``self`` (see _line_label_text).
            def _sync(
                *_a,
                _kind=kind,
                _line=line,
                _label=label,
                _handles=handles,
                _ext=self._EXTEND,
                _tick=self._tick_size,
                _dec=self._decimals,
                _vm=self._view_map,
                _label_fn=self._line_label_text,
            ):
                a = (_handles[0].pos().x(), _handles[0].pos().y())
                b = (_handles[1].pos().x(), _handles[1].pos().y())
                dx, dy = b[0] - a[0], b[1] - a[1]
                if _kind == "ray" and (dx or dy):
                    ext = (a[0] + dx * _ext, a[1] + dy * _ext)
                    _line.setData([a[0], ext[0]], [a[1], ext[1]])
                elif _kind == "extline" and (dx or dy):
                    lo = (a[0] - dx * _ext, a[1] - dy * _ext)
                    ext = (a[0] + dx * _ext, a[1] + dy * _ext)
                    _line.setData([lo[0], ext[0]], [lo[1], ext[1]])
                else:
                    _line.setData([a[0], b[0]], [a[1], b[1]])
                if _label is not None:
                    _label.setText(_label_fn(_kind, a, b, _tick, _dec, _vm))
                    _label.setPos(b[0], b[1])

            for handle in handles:
                handle.sigPositionChanged.connect(_sync)
            _sync()
            items = [line, *([label] if label is not None else []), *handles]
        elif kind in self._POSITION_KINDS:
            items = self._build_position_items(kind, spec)
        elif kind in self._FIB_KINDS:
            pts = spec.get("points") or [spec["p1"], spec["p2"]]
            items = self._build_fib_items(kind, [tuple(pt) for pt in pts])
        elif kind == "hray":
            # A horizontal ray: from its origin to the right, TradingView-style.
            line = pg.PlotDataItem(pen=pg.mkPen(p.vwap_rolling, width=1.2))
            origin = self._handle((float(spec["x"]), float(spec["y"])), color=p.vwap_rolling)

            def _sync_hray(*_a, _line=line, _origin=origin, _ext=self._EXTEND):
                x, y = _origin.pos().x(), _origin.pos().y()
                _line.setData([x, x + _ext], [y, y])

            origin.sigPositionChanged.connect(_sync_hray)
            _sync_hray()
            items = [line, origin]
        elif kind == "cross":
            # A crossline: full-height and full-width lines through one point.
            h = pg.InfiniteLine(
                angle=0,
                movable=False,
                pen=pg.mkPen(p.text_dim, width=1.1, style=QtCore.Qt.DashLine),
            )
            v = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen(p.text_dim, width=1.1, style=QtCore.Qt.DashLine),
            )
            origin = self._handle((float(spec["x"]), float(spec["y"])), color=p.text_dim)

            def _sync_cross(*_a, _h=h, _v=v, _origin=origin):
                _h.setPos(_origin.pos().y())
                _v.setPos(_origin.pos().x())

            origin.sigPositionChanged.connect(_sync_cross)
            _sync_cross()
            items = [h, v, origin]
        elif kind == "rect":
            # A zone box (supply/demand style): data-coordinate rect with a
            # cosmetic border and translucent fill, resized by dragging its
            # two corner handles. Exact under any zoom.
            rect = QtWidgets.QGraphicsRectItem()
            pen = pg.mkPen(p.gold, width=1.2)
            pen.setCosmetic(True)
            rect.setPen(pen)
            fill = QtGui.QColor(p.gold)
            fill.setAlpha(28)
            rect.setBrush(fill)
            corners = [
                pg.TargetItem(
                    pos=spec[key],
                    size=9,
                    movable=True,
                    pen=pg.mkPen(p.gold, width=1.2),
                    hoverPen=pg.mkPen(p.hook, width=2.0),
                    brush=pg.mkBrush(0, 0, 0, 0),
                )
                for key in ("p1", "p2")
            ]

            def _sync_rect(*_a, _rect=rect, _corners=corners):
                _rect.setRect(
                    QtCore.QRectF(
                        QtCore.QPointF(_corners[0].pos().x(), _corners[0].pos().y()),
                        QtCore.QPointF(_corners[1].pos().x(), _corners[1].pos().y()),
                    ).normalized()
                )

            for corner in corners:
                corner.sigPositionChanged.connect(_sync_rect)
            _sync_rect()
            items = [rect, *corners]
        else:
            return
        for item in items:
            item.setZValue(12)  # above the curtain so annotations stay usable in replay
            self._price.addItem(item, ignoreBounds=True)
        self._drawing_items.append((kind, items))

    def _build_position_items(self, kind: str, spec: dict) -> list:
        """Long/Short position tool: entry, target and stop with R:R zones.

        Three independent handles - entry (left), target and stop (right) -
        drive the shaded reward/risk zones and a live R:R label. Placing by
        drag sets the entry and the reward; the stop defaults to half the
        reward (R:R 2) on the opposite side, then drags freely.
        """

        p = theme.active()
        long_side = kind == "longpos"
        x1, entry_y = float(spec["p1"][0]), float(spec["p1"][1])
        x2 = float(spec["p2"][0])
        if x2 == x1:
            x2 = x1 + 1.0
        reward = abs(float(spec["p2"][1]) - entry_y)
        if reward == 0.0:
            reward = self._tick_size * 20
        target_y = entry_y + reward if long_side else entry_y - reward
        default_stop = entry_y - reward / 2 if long_side else entry_y + reward / 2
        stop_y = float(spec.get("stop", default_stop))

        reward_rect = QtWidgets.QGraphicsRectItem()
        risk_rect = QtWidgets.QGraphicsRectItem()
        for rect, colour in ((reward_rect, p.target), (risk_rect, p.stop)):
            pen = pg.mkPen(colour, width=1.1)
            pen.setCosmetic(True)
            rect.setPen(pen)
            fill = QtGui.QColor(colour)
            fill.setAlpha(30)
            rect.setBrush(fill)
        entry_line = pg.PlotDataItem(pen=pg.mkPen(p.text_dim, width=1.2))
        label = pg.TextItem("", color=p.bg, anchor=(0.5, 1.15), fill=pg.mkBrush(p.gold))
        h_entry = self._handle((x1, entry_y))
        h_target = self._handle((x2, target_y), color=p.target)
        h_stop = self._handle((x2, stop_y), color=p.stop)

        # No ``self`` capture in the closure (see _line_label_text): bind all.
        def _sync(
            *_a,
            _reward=reward_rect,
            _risk=risk_rect,
            _entry_line=entry_line,
            _label=label,
            _he=h_entry,
            _ht=h_target,
            _hs=h_stop,
            _long=long_side,
            _dec=self._decimals,
        ):
            ex, ey = _he.pos().x(), _he.pos().y()
            tx, ty = _ht.pos().x(), _ht.pos().y()
            sy = _hs.pos().y()
            left, right = min(ex, tx), max(ex, tx)
            if right == left:
                right = left + 1.0
            _reward.setRect(
                QtCore.QRectF(QtCore.QPointF(left, ey), QtCore.QPointF(right, ty)).normalized()
            )
            _risk.setRect(
                QtCore.QRectF(QtCore.QPointF(left, ey), QtCore.QPointF(right, sy)).normalized()
            )
            _entry_line.setData([left, right], [ey, ey])
            win = abs(ty - ey)
            loss = abs(ey - sy)
            rr = win / loss if loss else float("inf")
            rr_text = f"RR {rr:.2f}" if rr != float("inf") else "RR -"
            side = "Long" if _long else "Short"
            _label.setText(
                f"{side}  {rr_text}\n"
                f"target {ty:,.{_dec}f} (+{win:,.{_dec}f})\n"
                f"stop {sy:,.{_dec}f} (-{loss:,.{_dec}f})"
            )
            _label.setPos(0.5 * (left + right), ty if _long else sy)

        for handle in (h_entry, h_target, h_stop):
            handle.sigPositionChanged.connect(_sync)
        _sync()
        return [reward_rect, risk_rect, entry_line, label, h_entry, h_target, h_stop]

    _FIB_DASHED = ("fibret", "fibext", "fibchan", "fibtime", "fibttime")

    def _build_fib_items(self, kind: str, pts: list) -> list:
        """A Fibonacci tool: anchor handles + geometry polylines + level labels.

        The polyline count is fixed per tool, so a handle drag recomputes the
        geometry and updates data in place. The sync closure binds everything
        it needs (never ``self``) - see _line_label_text for why.
        """

        p = theme.active()
        style = QtCore.Qt.DashLine if kind in self._FIB_DASHED else QtCore.Qt.SolidLine
        pen = pg.mkPen(p.gold, width=1.1, style=style)
        lines = fib_geometry(kind, pts, extend=self._EXTEND)
        curves = [pg.PlotDataItem(pen=pen, connect="all") for _ in lines]
        labels = [
            pg.TextItem(f" {fl.label} ", color=p.gold, anchor=(0.0, 0.5)) if fl.label else None
            for fl in lines
        ]
        handles = [self._handle(pt) for pt in pts]

        def _sync(
            *_a, _kind=kind, _handles=handles, _curves=curves, _labels=labels, _ext=self._EXTEND
        ):
            pts_now = [(h.pos().x(), h.pos().y()) for h in _handles]
            geo = fib_geometry(_kind, pts_now, extend=_ext)
            for curve, lab, fl in zip(_curves, _labels, geo, strict=True):
                curve.setData(np.asarray(fl.xs), np.asarray(fl.ys))
                if lab is not None:
                    lab.setText(f" {fl.label} ")
                    lab.setPos(fl.xs[0], fl.ys[0])

        for handle in handles:
            handle.sigPositionChanged.connect(_sync)
        _sync()
        return [*curves, *[lab for lab in labels if lab is not None], *handles]

    def undo_drawing(self) -> None:
        """Remove the most recent drawing on this view (or a pending anchor).

        The removed drawing's global-coordinate spec is kept so ``redo_drawing``
        can restore it exactly, whatever zoom or timeframe is showing by then.
        """

        if self._trend_anchor is not None or self._multi_anchors:
            self._clear_trend_anchor()
            self._remove_drag_preview()
            return
        if self._drawing_items:
            kind, items = self._drawing_items.pop()
            spec = self._spec_for(kind, items)
            if spec is not None:
                self._redo_specs.append(spec)
            for item in items:
                self._price.removeItem(item)

    def redo_drawing(self) -> None:
        """Restore the most recently undone drawing."""

        if not self._redo_specs:
            return
        spec = self._redo_specs.pop()
        if self._spec_intersects_view(spec):
            self._create_drawing(self._spec_to_local(spec))
        else:  # restored on a window that no longer shows it: keep it stored
            self._drawing_store.append(spec)

    def clear_drawings(self) -> None:
        """Remove every drawing on this view (other windows' drawings survive)."""

        self._clear_trend_anchor()
        self._remove_drag_preview()
        for kind, items in self._drawing_items:
            spec = self._spec_for(kind, items)
            if spec is not None:
                self._redo_specs.append(spec)
            for item in items:
                self._price.removeItem(item)
        self._drawing_items.clear()

    def _clear_trend_anchor(self) -> None:
        self._trend_anchor = None
        if self._trend_anchor_dot is not None:
            self._price.removeItem(self._trend_anchor_dot)
            self._trend_anchor_dot = None
        self._multi_anchors.clear()
        for dot in self._multi_dots:
            self._price.removeItem(dot)
        self._multi_dots.clear()

    def _serialize_drawings(self) -> list[dict]:
        """Live drawings as JSON-safe specs in GLOBAL coordinates.

        X positions are fractional global 1m indices via the view map, so a
        drawing lands on the same timestamps whatever timeframe or range is
        showing when it is restored.
        """

        specs = (self._spec_for(kind, items) for kind, items in self._drawing_items)
        return [spec for spec in specs if spec is not None]

    def _spec_for(self, kind: str, items) -> dict | None:
        """One live drawing's JSON-safe spec in global coordinates."""

        vm = self._view_map
        if kind == "hline":
            return {
                "kind": "hline",
                "y": float(items[0].value()),
                "gspan": [vm.view_start, vm.view_end],
            }
        if kind == "vline":
            return {"kind": "vline", "gx": vm.local_f_to_global(items[0].value())}
        if kind == "measure":
            p1, p2 = items[0].data(0)
            return {
                "kind": "measure",
                "p1": (vm.local_f_to_global(p1[0]), float(p1[1])),
                "p2": (vm.local_f_to_global(p2[0]), float(p2[1])),
            }
        if kind in ("hray", "cross"):
            origin = items[-1]
            return {
                "kind": kind,
                "gx": vm.local_f_to_global(origin.pos().x()),
                "y": float(origin.pos().y()),
            }
        if kind in self._FIB_KINDS:
            n = FIB_POINTS[kind]
            handles = items[-n:]
            return {
                "kind": kind,
                "points": [
                    (vm.local_f_to_global(h.pos().x()), float(h.pos().y())) for h in handles
                ],
            }
        if kind in self._POSITION_KINDS:
            h_entry, h_target, h_stop = items[-3], items[-2], items[-1]
            return {
                "kind": kind,
                "p1": (vm.local_f_to_global(h_entry.pos().x()), float(h_entry.pos().y())),
                "p2": (vm.local_f_to_global(h_target.pos().x()), float(h_target.pos().y())),
                "stop": float(h_stop.pos().y()),
            }
        if kind in (*self._LINE_FAMILY, "rect"):
            h1, h2 = items[-2], items[-1]
            return {
                "kind": kind,
                "p1": (vm.local_f_to_global(h1.pos().x()), float(h1.pos().y())),
                "p2": (vm.local_f_to_global(h2.pos().x()), float(h2.pos().y())),
            }
        return None

    def _spec_to_local(self, spec: dict) -> dict:
        """A stored global spec converted to this view's local coordinates."""

        vm = self._view_map
        kind = spec["kind"]
        if kind == "hline":
            return dict(spec)
        if kind in ("vline", "hray", "cross"):
            local = {"kind": kind, "x": vm.global_to_local_f(spec["gx"])}
            if "y" in spec:
                local["y"] = spec["y"]
            return local
        if "points" in spec:
            return {
                "kind": kind,
                "points": [(vm.global_to_local_f(gx), y) for gx, y in spec["points"]],
            }
        local = {
            "kind": kind,
            "p1": (vm.global_to_local_f(spec["p1"][0]), spec["p1"][1]),
            "p2": (vm.global_to_local_f(spec["p2"][0]), spec["p2"][1]),
        }
        if "stop" in spec:
            local["stop"] = spec["stop"]
        return local

    def _stash_live_drawings(self) -> None:
        """Move live drawings into the store (called before the view rebuilds)."""

        if self._ohlc is None:
            return
        self._drawing_store.extend(self._serialize_drawings())
        self._drawing_items.clear()  # the graphics items die with _price.clear()

    def _spec_intersects_view(self, spec: dict) -> bool:
        vm = self._view_map
        lo, hi = vm.view_start, vm.view_end
        kind = spec["kind"]
        if kind == "hline":
            a, b = spec.get("gspan", [lo, hi])
            return a <= hi and b >= lo
        if kind in ("vline", "cross"):
            return lo <= spec["gx"] <= hi
        if kind == "hray":  # extends right from its origin
            return spec["gx"] <= hi
        if kind in ("ray", "extline"):  # reach far beyond their anchors
            return True
        if "points" in spec:
            return any(lo <= gx <= hi for gx, _y in spec["points"])
        return any(lo <= spec[key][0] <= hi for key in ("p1", "p2"))

    def _restore_drawings(self) -> None:
        """Materialize stored specs that intersect the current view."""

        keep: list[dict] = []
        for spec in self._drawing_store:
            if not self._spec_intersects_view(spec):
                keep.append(spec)
                continue
            self._create_drawing(self._spec_to_local(spec))
        self._drawing_store = keep

    def export_drawings(self) -> list[dict]:
        """Every drawing (live and stored) as global-coordinate specs."""

        return [*self._drawing_store, *self._serialize_drawings()]

    def import_drawings(self, specs: list[dict]) -> None:
        """Replace all drawings from serialized global specs."""

        self.clear_drawings()
        self._redo_specs.clear()  # a loaded session starts with clean history
        self._drawing_store = [dict(spec) for spec in specs]
        self._restore_drawings()

    def set_reveal(self, up_to_global: int | None) -> None:
        """Show only fully-elapsed displayed bars at 1m time ``up_to_global``.

        Whole buckets only: revealing a forming bucket's candle would leak its
        aggregated high/low/close - minutes the replay clock has not reached.
        ``None`` reveals the whole window.
        """

        if self._ohlc is None or not self._curtains:
            return
        if up_to_global is None:
            for curtain in self._curtains:
                curtain.setVisible(False)
            return
        n = len(self._ohlc["close"])
        last_complete = self._view_map.last_complete_local(int(up_to_global))
        edge = min(float(last_complete) + 0.5, n + 1.0)
        for curtain in self._curtains:
            curtain.setRegion((edge, n + 1.0))
            curtain.setVisible(True)

    def add_vwap(self, vwap: np.ndarray, kind: str = "rolling", *, visible: bool = True) -> None:
        """Overlay a VWAP line. ``kind`` selects colour/legend label.

        Lines persist for the rendered window; checkbox toggles flip visibility
        via ``set_vwap_visible`` instead of rebuilding the whole chart.
        """

        p = theme.active()
        colour = {
            "rolling": p.vwap_rolling,
            "day": p.vwap_day,
            "session": p.vwap_session,
        }.get(kind, p.vwap_rolling)
        label = {"rolling": "VWAP 20", "day": "VWAP day", "session": "VWAP session"}.get(
            kind, "VWAP"
        )
        x = np.arange(len(vwap))
        item = pg.PlotDataItem(x, vwap, pen=pg.mkPen(colour, width=1.4), name=label)
        item.setVisible(visible)
        self._price.addItem(item)
        self._vwap_items[kind] = item

    def set_compare_line(self, values: np.ndarray, label: str) -> None:
        """Overlay a normalized comparison series; the label states the method."""

        self.clear_compare_line()
        p = theme.active()
        x = np.arange(len(values))
        self._compare_item = pg.PlotDataItem(
            x, values, pen=pg.mkPen(p.gold, width=1.6, style=QtCore.Qt.DashLine), name=label
        )
        self._price.addItem(self._compare_item)

    def clear_compare_line(self) -> None:
        if self._compare_item is not None:
            self._price.removeItem(self._compare_item)
            self._compare_item = None

    def set_vwap_visible(self, kind: str, on: bool) -> None:
        item = self._vwap_items.get(kind)
        if item is not None:
            item.setVisible(on)

    # -- indicator studies ---------------------------------------------------
    _PANES = ("price", "osc", "volume")

    def _pane(self, pane: str):
        return {"price": self._price, "osc": self._osc, "volume": self._volume}[pane]

    def set_indicator(
        self,
        inst_id: int,
        pane: str,
        series: dict[str, np.ndarray],
        color: str,
        *,
        visible: bool = True,
        band: tuple[str, str] | None = None,
        guides: tuple[float, ...] = (),
    ) -> None:
        """Draw (or redraw) one study instance's output lines on its pane.

        ``band`` names two outputs shaded as a translucent fill (Bollinger);
        ``guides`` are fixed reference levels drawn dashed (RSI 30/70). Only
        active studies are ever passed here, so nothing inactive is computed
        or rendered.
        """

        self.remove_indicator(inst_id)
        plot = self._pane(pane)
        items: list = []
        x = None
        for name, values in series.items():
            values = np.asarray(values, dtype=float)
            if x is None or len(x) != len(values):
                x = np.arange(len(values))
            style = QtCore.Qt.DashLine if band is not None and name in band else QtCore.Qt.SolidLine
            item = pg.PlotDataItem(
                x, values, pen=pg.mkPen(color, width=1.3, style=style), connect="finite"
            )
            plot.addItem(item, ignoreBounds=pane == "price")
            items.append(item)
            series[name] = values
        if band is not None and band[0] in series and band[1] in series:
            fill_color = QtGui.QColor(color)
            fill_color.setAlpha(22)
            upper = pg.PlotDataItem(x, series[band[0]], pen=None, connect="finite")
            lower = pg.PlotDataItem(x, series[band[1]], pen=None, connect="finite")
            fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(fill_color))
            fill.setZValue(-7)
            for extra in (upper, lower, fill):
                plot.addItem(extra, ignoreBounds=True)
                items.append(extra)
        p = theme.active()
        for level in guides:
            guide = pg.InfiniteLine(
                pos=float(level),
                angle=0,
                movable=False,
                pen=pg.mkPen(p.text_faint, width=1, style=QtCore.Qt.DashLine),
            )
            plot.addItem(guide, ignoreBounds=True)
            items.append(guide)
        for item in items:
            item.setVisible(visible)
        self._indicator_items[inst_id] = (pane, items)
        self._indicator_visible[inst_id] = bool(visible)
        self._sync_osc_pane()

    def set_indicator_visible(self, inst_id: int, on: bool) -> None:
        pane_items = self._indicator_items.get(inst_id)
        if pane_items is None:
            return
        for item in pane_items[1]:
            item.setVisible(on)
        self._indicator_visible[inst_id] = bool(on)
        self._sync_osc_pane()

    def remove_indicator(self, inst_id: int) -> None:
        pane_items = self._indicator_items.pop(inst_id, None)
        self._indicator_visible.pop(inst_id, None)
        if pane_items is None:
            return
        plot = self._pane(pane_items[0])
        for item in pane_items[1]:
            plot.removeItem(item)
        self._sync_osc_pane()

    def _sync_osc_pane(self) -> None:
        """The oscillator pane exists only while a visible study needs it.

        Declared visibility is tracked separately because a child item of a
        hidden pane reports ``isVisible() == False`` even when it was just
        shown - reading it back would keep the pane hidden forever.
        """

        active = any(
            pane == "osc" and self._indicator_visible.get(inst_id, False)
            for inst_id, (pane, _items) in self._indicator_items.items()
        )
        self._osc.setVisible(active)
        self._layout.ci.layout.setRowStretchFactor(2, 1 if active else 0)
        self._layout.ci.layout.setRowMaximumHeight(2, 16777215 if active else 0)

    def set_volume_visible(self, on: bool) -> None:
        """Explicit show/hide for the volume pane (chart state is untouched)."""

        self._volume_visible = bool(on)
        self._volume.setVisible(self._volume_visible)
        self._layout.ci.layout.setRowStretchFactor(1, 1 if on else 0)
        self._layout.ci.layout.setRowMaximumHeight(1, 16777215 if on else 0)

    def volume_visible(self) -> bool:
        return self._volume_visible

    # -- header legend chips -------------------------------------------------
    def set_header_chips(self, chips: list[dict], on_toggle, on_remove) -> None:
        """Rebuild the active-study strip in the chart header.

        ``chips`` rows are {"id", "label", "color", "visible"}. Only active
        studies appear; each chip carries visible hide and remove controls, so
        nothing inactive is ever listed permanently.
        """

        for widget in self._chip_widgets.values():
            self._legend_bar.removeWidget(widget)
            widget.hide()
            widget.setParent(None)  # a deferred delete must never paint over the chart
            widget.deleteLater()
        self._chip_widgets.clear()
        p = theme.active()
        for chip in chips:
            box = QtWidgets.QFrame()
            box.setObjectName("legendChip")
            lay = QtWidgets.QHBoxLayout(box)
            lay.setContentsMargins(6, 0, 2, 0)
            lay.setSpacing(2)
            name = QtWidgets.QLabel(chip["label"])
            colour = chip["color"] if chip["visible"] else p.text_faint
            name.setStyleSheet(f"color:{colour}; font-size:11px")
            lay.addWidget(name)
            hide_btn = QtWidgets.QToolButton()
            hide_btn.setText("-" if chip["visible"] else "+")
            hide_btn.setToolTip("Hide this study" if chip["visible"] else "Show this study")
            remove_btn = QtWidgets.QToolButton()
            remove_btn.setText("x")
            remove_btn.setToolTip("Remove this study from the chart.")
            for btn in (hide_btn, remove_btn):
                btn.setAutoRaise(True)
                btn.setFocusPolicy(QtCore.Qt.NoFocus)
                btn.setFixedSize(16, 16)
                lay.addWidget(btn)
            hide_btn.clicked.connect(lambda _c=False, i=chip["id"]: on_toggle(i))
            remove_btn.clicked.connect(lambda _c=False, i=chip["id"]: on_remove(i))
            self._legend_bar.addWidget(box)
            self._chip_widgets[chip["id"]] = box

    def shade_sessions(self, session_code: np.ndarray, *, visible: bool = True) -> None:
        """Light vertical shading for the New York execution window (code == 2)."""

        p = theme.active()
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
            region = pg.LinearRegionItem([a, b], movable=False, brush=pg.mkBrush(*p.session_shade))
            region.setZValue(-10)
            region.setVisible(visible)
            self._price.addItem(region)
            self._session_regions.append(region)

    def set_sessions_visible(self, on: bool) -> None:
        for region in self._session_regions:
            region.setVisible(on)

    def mark_trades(self, entries_global, y, directions, gross_r) -> None:
        """Scatter entry markers at entry price, coloured by outcome (global 1m indices)."""

        p = theme.active()
        brushes = [pg.mkBrush(p.up) if r > 0 else pg.mkBrush(p.down) for r in gross_r]
        symbols = ["t1" if d > 0 else "t" for d in directions]
        xs = self._view_map.global_to_local_f(np.asarray(entries_global))
        item = pg.ScatterPlotItem(
            x=list(np.atleast_1d(xs)),
            y=list(y),
            symbol=symbols,
            brush=brushes,
            size=11,
            pen=pg.mkPen(p.bg),
        )
        # Annotations sit inside the candle range; they must never drive
        # autorange (which would make the view jump on every redraw).
        self._add_trade_item(item, ignore_bounds=True)

    def draw_trade(
        self,
        entry_global,
        exit_global,
        entry_price,
        stop_track,
        target_track,
        exit_price,
        *,
        color: str | None = None,
    ):
        """Draw (or update) the one detailed trade: level tracks and entry/exit.

        Positions are global 1m indices; per-minute tracks render at each
        minute's fractional x inside its displayed bucket, so a trailing stop's
        ratchet stays pixel-honest at any timeframe. The items persist between
        calls, so a stop/target drag at 60 Hz is a handful of ``setData`` calls,
        not a teardown of the overlay layer.
        """

        p = theme.active()
        marker = color or p.entry
        entry_local = self._view_map.global_to_local_f(entry_global)
        exit_local = self._view_map.global_to_local_f(exit_global)
        items = self._active_trade_items
        if items is None:
            entry_line = pg.PlotDataItem(pen=None)  # invisible anchor for the zones
            stop_curve = pg.PlotDataItem()
            target_curve = pg.PlotDataItem()
            # Risk/reward zones between entry and the level tracks, like a
            # long/short position tool: red where the trade can be cut, green
            # where it pays. Following the tracks means a trailing stop's
            # ratchet is visible as the red zone tightening.
            risk_fill = pg.FillBetweenItem(entry_line, stop_curve)
            reward_fill = pg.FillBetweenItem(entry_line, target_curve)
            for fill in (risk_fill, reward_fill):
                fill.setZValue(-6)  # under the excursion ribbons
            items = {
                "entry_line": entry_line,
                "stop": stop_curve,
                "target": target_curve,
                "risk_fill": risk_fill,
                "reward_fill": reward_fill,
                "entry": pg.ScatterPlotItem(symbol="o", size=12),
                "exit": pg.ScatterPlotItem(symbol="x", size=13),
            }
            for item in items.values():
                self._add_trade_item(item, ignore_bounds=True)
            self._active_trade_items = items
        items["stop"].setPen(pg.mkPen(p.stop, style=QtCore.Qt.DashLine))
        items["target"].setPen(pg.mkPen(p.target, style=QtCore.Qt.DashLine))
        risk = QtGui.QColor(p.stop)
        risk.setAlpha(26)
        reward = QtGui.QColor(p.target)
        reward.setAlpha(26)
        items["risk_fill"].setBrush(pg.mkBrush(risk))
        items["reward_fill"].setBrush(pg.mkBrush(reward))
        if len(stop_track):
            held = self._view_map.minute_positions(int(entry_global), len(stop_track))
            items["entry_line"].setData(held, np.full(len(held), float(entry_price)))
            items["stop"].setData(held, stop_track)
            items["target"].setData(held, target_track)
        else:
            for key in ("entry_line", "stop", "target"):
                items[key].clear()
        items["entry"].setData(x=[entry_local], y=[entry_price], brush=pg.mkBrush(marker))
        items["exit"].setData(x=[exit_local], y=[exit_price], pen=pg.mkPen(marker, width=2))

    def mirror_trade(
        self, entry_global, exit_global, entry_price, stop_level, target_level, exit_price
    ) -> None:
        """Ghost a trade from the other instrument onto this tape.

        Entry/exit markers at the times mapped from the source trade (global 1m
        indices on THIS chart's clock) plus its initial stop/target levels, so
        "did the pattern replicate" is a visual check rather than a spreadsheet
        exercise.
        """

        p = theme.active()
        entry_local = self._view_map.global_to_local_f(entry_global)
        exit_local = self._view_map.global_to_local_f(exit_global)
        span = [entry_local, max(exit_local, entry_local + 1)]
        if stop_level is not None:
            self._add_trade_item(
                pg.PlotDataItem(
                    span,
                    [stop_level] * 2,
                    pen=pg.mkPen(p.stop, width=1.2, style=QtCore.Qt.DashLine),
                ),
                ignore_bounds=True,
            )
        if target_level is not None:
            self._add_trade_item(
                pg.PlotDataItem(
                    span,
                    [target_level] * 2,
                    pen=pg.mkPen(p.target, width=1.2, style=QtCore.Qt.DashLine),
                ),
                ignore_bounds=True,
            )
        self._add_trade_item(
            pg.ScatterPlotItem(
                x=[entry_local], y=[entry_price], symbol="o", size=11, brush=pg.mkBrush(p.entry)
            ),
            ignore_bounds=True,
        )
        self._add_trade_item(
            pg.ScatterPlotItem(
                x=[exit_local], y=[exit_price], symbol="x", size=12, pen=pg.mkPen(p.entry, width=2)
            ),
            ignore_bounds=True,
        )

    def light_marker(self, entry_global, entry_price, exit_global, exit_price, gross_r, color):
        """A faint entry-to-exit connector for a non-active placed trade (global 1m)."""

        p = theme.active()
        outcome = p.up if gross_r > 0 else p.down
        entry_local = self._view_map.global_to_local_f(entry_global)
        exit_local = self._view_map.global_to_local_f(exit_global)
        self._add_trade_item(
            pg.PlotDataItem(
                [entry_local, exit_local],
                [entry_price, exit_price],
                pen=pg.mkPen(color, width=1, style=QtCore.Qt.DotLine),
            ),
            ignore_bounds=True,
        )
        self._add_trade_item(
            pg.ScatterPlotItem(
                x=[entry_local], y=[entry_price], symbol="o", size=8, brush=pg.mkBrush(color)
            ),
            ignore_bounds=True,
        )
        self._add_trade_item(
            pg.ScatterPlotItem(
                x=[exit_local], y=[exit_price], symbol="x", size=9, pen=pg.mkPen(outcome, width=1)
            ),
            ignore_bounds=True,
        )

    def draw_excursion(self, exc, hook: bool = False) -> None:
        """Shade how far the trade ran for/against, with peak/trough markers.

        ``exc`` is an analysis.excursion.Excursion (global x). When ``hook`` the
        favourable ribbon is brightened to gold to flag a 'winner on the hook'.
        """

        p = theme.active()
        x = self._view_map.global_to_local_f(np.asarray(exc.x))
        items = self._excursion_items
        if items is None:
            baseline = pg.PlotDataItem(pen=None)
            fav_curve = pg.PlotDataItem()
            adv_curve = pg.PlotDataItem()
            fav_fill = pg.FillBetweenItem(baseline, fav_curve)
            adv_fill = pg.FillBetweenItem(baseline, adv_curve)
            for item in (fav_fill, adv_fill):
                item.setZValue(-5)
            items = {
                "baseline": baseline,
                "fav_curve": fav_curve,
                "adv_curve": adv_curve,
                "fav_fill": fav_fill,
                "adv_fill": adv_fill,
                "peak": pg.ScatterPlotItem(symbol="d", size=12),
                "trough": pg.ScatterPlotItem(symbol="d", size=11),
                "label": pg.TextItem("", anchor=(0, 1)),
            }
            # Overlays are within the candles' range, so they must not drive
            # autorange (FillBetweenItem reports a [0,0] bound that would
            # otherwise collapse Y). Items persist for cheap drag updates.
            for item in items.values():
                self._add_trade_item(item, ignore_bounds=True)
            self._excursion_items = items
        items["fav_curve"].setPen(pg.mkPen(p.target, width=1))
        items["adv_curve"].setPen(pg.mkPen(p.stop, width=1))
        items["baseline"].setData(x, np.full(len(x), exc.entry_price))
        items["fav_curve"].setData(x, exc.fav_price)
        items["adv_curve"].setData(x, exc.adv_price)
        items["fav_fill"].setBrush(pg.mkBrush(*(p.hook_fill if hook else p.mfe_fill)))
        items["adv_fill"].setBrush(pg.mkBrush(*p.mae_fill))
        peak_x = self._view_map.global_to_local_f(exc.peak_x)
        trough_x = self._view_map.global_to_local_f(exc.trough_x)
        items["peak"].setData(
            x=[peak_x],
            y=[exc.peak_price],
            brush=pg.mkBrush(p.hook if hook else p.target),
            pen=pg.mkPen(p.bg),
        )
        items["trough"].setData(
            x=[trough_x],
            y=[exc.trough_price],
            brush=pg.mkBrush(p.stop),
            pen=pg.mkPen(p.bg),
        )
        items["label"].setColor(p.hook if hook else p.target)
        items["label"].setText(f"+{exc.mfe_r:.1f} R @ +{exc.time_to_peak_min}m")
        items["label"].setPos(peak_x, exc.peak_price)

    def draw_whatif(self, runs) -> None:
        """Overlay each alternative-exit run's stop track and exit in its colour."""

        self.clear_whatif()
        for run in runs:
            r = run.result
            if len(r.stop_track):
                held = self._view_map.minute_positions(int(r.entry_position), len(r.stop_track))
                item = pg.PlotDataItem(
                    held,
                    r.stop_track,
                    pen=pg.mkPen(run.color, width=1.2, style=QtCore.Qt.DashDotLine),
                )
                self._price.addItem(item, ignoreBounds=True)
                self._whatif_items.append(item)
            marker = pg.ScatterPlotItem(
                x=[self._view_map.global_to_local_f(r.exit_position)],
                y=[r.exit_price],
                symbol="x",
                size=11,
                pen=pg.mkPen(run.color, width=2),
            )
            self._price.addItem(marker, ignoreBounds=True)
            self._whatif_items.append(marker)

    def clear_whatif(self) -> None:
        """Drop the alternative-exit overlays (e.g. once their config is stale)."""

        for item in self._whatif_items:
            self._price.removeItem(item)
        self._whatif_items.clear()

    def _add_trade_item(self, item, ignore_bounds: bool = False) -> None:
        self._price.addItem(item, ignoreBounds=ignore_bounds)
        self._trade_items.append(item)

    def clear_trades(self) -> None:
        for item in self._trade_items:
            self._price.removeItem(item)
        self._trade_items.clear()
        self.clear_whatif()
        self._active_trade_items = None  # items were just removed with the layer
        self._excursion_items = None

    # -- draggable stop/target levels (active free-play trade) --------------
    def set_draggable_levels(self, stop_price, target_price, on_changed) -> None:
        """Two draggable horizontal lines; ``on_changed(kind, price)`` on drag."""

        self.clear_draggable_levels()
        p = theme.active()
        self._on_level_changed = on_changed
        specs = (("stop", stop_price, p.stop), ("target", target_price, p.target))
        for kind, price, colour in specs:
            line = pg.InfiniteLine(
                pos=float(price),
                angle=0,
                movable=True,
                pen=pg.mkPen(colour, width=1.5, style=QtCore.Qt.DashLine),
                hoverPen=pg.mkPen(colour, width=2.5),
                label=kind,
                labelOpts={"position": 0.04, "color": colour, "movable": True},
            )
            line.setZValue(15)
            self._price.addItem(line)
            self._level_lines.append(line)
            self._level_proxies.append(
                pg.SignalProxy(
                    line.sigDragged, rateLimit=60, slot=lambda _e, k=kind: self._emit_level(k)
                )
            )

    def _emit_level(self, kind: str) -> None:
        # A rate-limited SignalProxy event can arrive after clear_draggable_levels
        # (e.g. the redraw that follows a config change), so re-check both.
        if self._on_level_changed is None or len(self._level_lines) < 2:
            return
        line = self._level_lines[0] if kind == "stop" else self._level_lines[1]
        self._on_level_changed(kind, float(line.value()))

    def clear_draggable_levels(self) -> None:
        for line in self._level_lines:
            self._price.removeItem(line)
        self._level_lines.clear()
        self._level_proxies.clear()
        self._on_level_changed = None

    # -- bar-by-bar replay animation ---------------------------------------
    def start_replay(self, result=None, *, start_global: int | None = None) -> None:
        """Set up the animated replay and drop the reveal curtain at its start.

        With a trade ``result``, the moving marker is joined by its stop/target
        lines. With ``result=None`` (a bare day replay) only the marker and the
        elapsed-time label ride the tape. Either way, history before the start
        stays visible for context and everything after "now" is hidden.
        """

        self.stop_replay()
        p = theme.active()
        start = int(result.entry_position) if result is not None else int(start_global)
        self._replay_marker = pg.ScatterPlotItem(
            symbol="o", size=13, brush=pg.mkBrush(p.gold), pen=pg.mkPen(p.bg)
        )
        self._replay_label = pg.TextItem("", color=p.text, anchor=(0, 1))
        self._replay_items = [self._replay_marker, self._replay_label]
        self._replay_stop = self._replay_target = None
        if result is not None:
            self._replay_stop = pg.InfiniteLine(
                angle=0, movable=False, pen=pg.mkPen(p.stop, width=1.5, style=QtCore.Qt.DashLine)
            )
            self._replay_target = pg.InfiniteLine(
                angle=0,
                movable=False,
                pen=pg.mkPen(p.target, width=1.5, style=QtCore.Qt.DashLine),
            )
            self._replay_items = [
                self._replay_stop,
                self._replay_target,
                self._replay_marker,
                self._replay_label,
            ]
        for item in self._replay_items:
            item.setZValue(10)  # ride above the reveal curtain (z=8)
            self._price.addItem(item, ignoreBounds=True)
        self.set_reveal(start)

    def replay_frame(self, result, t: int, *, start_global: int | None = None) -> None:
        """Advance the animation to global 1m bar ``t`` (``result`` may be None).

        The gold marker rides actual minute closes at fractional x inside the
        displayed bucket, so playback stays continuous between candle
        completions at any timeframe.
        """

        if not self._replay_items or self._ohlc is None:
            return
        start = int(result.entry_position) if result is not None else int(start_global)
        vm = self._view_map
        if not vm.in_view(int(t)):
            return
        local = vm.global_to_local_f(int(t))
        price = float(self._minute_close[int(t) - vm.view_start])
        self.set_reveal(int(t))  # whole elapsed buckets appear; the future stays hidden
        # Auto-scroll when the marker nears the right of the view.
        (x0, x1), _y = self._price.viewRange()
        if local > x1 - 8:
            width = x1 - x0
            self._price.setXRange(local - width * 0.7, local + width * 0.3, padding=0)
        self._replay_marker.setData([local], [price])
        self._set_last_price(price)  # the scale pill rides the replay tape
        label = f"+{int(t) - start}m"
        if result is not None:
            idx = max(0, min(int(t) - start, len(result.stop_track) - 1))
            if len(result.stop_track):
                self._replay_stop.setPos(float(result.stop_track[idx]))
                self._replay_target.setPos(float(result.target_track[idx]))
            stop_points = float(result.initial_stop_points)
            if stop_points > 0 and np.isfinite(stop_points):
                running_r = (price - result.entry_price) / stop_points
                if result.direction < 0:
                    running_r = -running_r
                label = f"+{int(t) - start}m   {running_r:+.2f} R"
        self._replay_label.setText(label)
        self._replay_label.setPos(local, price)

    def stop_replay(self) -> None:
        for item in self._replay_items:
            self._price.removeItem(item)
        self._replay_items = []
        self.set_reveal(None)  # lift the curtain; the whole day returns
        if self._ohlc is not None and len(self._ohlc["close"]):
            self._set_last_price(float(self._ohlc["close"][-1]))

    def center_on(self, global_index: int, pad: int | None = None) -> None:
        """Centre the view on a global 1m bar; ``pad`` is in displayed bars."""

        local_index = self._view_map.global_to_local(int(global_index))
        if pad is None:
            if self._view_map.tf_minutes <= 1:
                pad = theme.CHART_PAD_BARS
            else:  # coarser buckets: keep a sensible share of the window visible
                pad = max(10, min(theme.CHART_PAD_BARS, self._view_map.n_display // 4))
        else:
            pad = int(pad)
        x0, x1 = local_index - pad, local_index + pad
        self._price.setXRange(x0, x1, padding=0)
        # Fit Y to the candles actually visible so a single trade isn't squashed
        # (annotation items must not drag autorange toward zero).
        if self._ohlc is not None:
            n = len(self._ohlc["close"])
            a, b = max(0, x0), min(n - 1, x1)
            if b > a:
                window_low = self._ohlc["low"][a : b + 1]
                window_high = self._ohlc["high"][a : b + 1]
                if np.isnan(window_low).all() or np.isnan(window_high).all():
                    return  # nothing finite to frame; keep the current range
                lo = float(np.nanmin(window_low))
                hi = float(np.nanmax(window_high))
                margin = (hi - lo) * 0.08 or 1.0
                self._price.setYRange(lo - margin, hi + margin, padding=0)

    # -- interaction -------------------------------------------------------
    def _on_click(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        pos = event.scenePos()
        # Test against the ViewBox, not the whole PlotItem: the plot rect
        # includes the axis strips, where a click maps to an out-of-range bar
        # (numpy would accept the negative index and place a trade on the
        # wrong day entirely).
        if self._ohlc is None or not self._price.vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = self._price.vb.mapSceneToView(pos)
        if self._draw_mode is not None:  # an armed drawing tool takes the click
            self.place_drawing(self._draw_mode, mouse_point.x(), mouse_point.y())
            return
        local = int(round(mouse_point.x()))
        if not (0 <= local < len(self._ohlc["close"])):
            return
        # Emit the clicked displayed bar's LAST 1m index: the free-play caller
        # adds one, entering at the first minute of the NEXT displayed bar -
        # the honest next-bar-open at any timeframe (identical at 1m).
        self.bar_clicked.emit(self._view_map.local_to_global_end(local))

    def _on_mouse_moved(self, evt) -> None:
        pos = evt[0]
        if self._ohlc is None or not self._price.sceneBoundingRect().contains(pos):
            self._vline.setVisible(False)
            self._hline.setVisible(False)
            self._x_badge.setVisible(False)
            self._y_badge.setVisible(False)
            return
        mp = self._price.vb.mapSceneToView(pos)
        i = int(round(mp.x()))
        self._vline.setPos(mp.x())
        self._hline.setPos(mp.y())
        self._vline.setVisible(True)
        self._hline.setVisible(True)
        self._update_badges(mp.x(), mp.y(), i)
        n = len(self._ohlc["close"])
        if 0 <= i < n and i != self._last_readout_i:
            self._last_readout_i = i  # the readout only changes per bar, not per pixel
            self._readout_label.setText(self._readout(i))
            self.bar_hovered.emit(self._view_map.local_to_global_end(i))

    def _update_badges(self, x: float, y: float, i: int) -> None:
        """Pin the exact time and tick-formatted price to the view edges."""

        (x0, x1), (y0, y1) = self._price.vb.viewRange()
        xr, yr = (x1 - x0) or 1.0, (y1 - y0) or 1.0
        if 0 <= i < len(self._hover_labels):
            # The exact date/time pill sits directly beneath the vertical
            # crosshair, clamped so it stays readable at the view extremes.
            self._x_badge.setText(f" {self._hover_labels[i]} ")
            self._x_badge.setPos(min(max(x, x0 + xr * 0.03), x1 - xr * 0.03), y0)
            self._x_badge.setVisible(True)
        else:
            self._x_badge.setVisible(False)
        tick_price = round(y / self._tick_size) * self._tick_size
        self._y_badge.setText(f" {tick_price:,.{self._decimals}f} ")
        self._y_badge.setPos(x1, min(max(y, y0 + yr * 0.02), y1 - yr * 0.02))
        self._y_badge.setVisible(True)

    def _set_last_price(self, price: float | None) -> None:
        """Move the last-price line/pill to the newest revealed close."""

        self._last_price = price
        if self._last_line is None:
            return
        if price is None:
            self._last_line.setVisible(False)
            return
        self._last_line.setVisible(True)
        self._last_line.setPos(float(price))

    def _readout(self, i: int) -> str:
        p = theme.active()
        dec = self._decimals
        o = self._ohlc["open"][i]
        h = self._ohlc["high"][i]
        low = self._ohlc["low"][i]
        c = self._ohlc["close"][i]
        prev = self._ohlc["close"][i - 1] if i > 0 else o
        change = c - prev
        pct = (change / prev * 100.0) if prev else 0.0
        candle = p.up if c >= o else p.down
        move = p.up if change >= 0 else p.down
        return (
            f"<span style='color:{p.text_dim}'>O</span> {o:,.{dec}f}  "
            f"<span style='color:{p.text_dim}'>H</span> {h:,.{dec}f}  "
            f"<span style='color:{p.text_dim}'>L</span> {low:,.{dec}f}  "
            f"<span style='color:{candle}'>C {c:,.{dec}f}</span>  "
            f"<span style='color:{move}'>{change:+,.{dec}f} ({pct:+.2f}%)</span>"
        )
