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

from . import theme


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
        up_pen = pg.mkPen(p.up)
        down_pen = pg.mkPen(p.down)
        up_brush = pg.mkBrush(p.up)
        down_brush = pg.mkBrush(p.down)
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
            height = top - bot or theme.CANDLE_MIN_HEIGHT
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
        p = theme.active()
        pg.setConfigOptions(antialias=True, background=p.bg, foreground=p.text_faint)
        self._time_axis = _TimeAxis(np.array([]), orientation="bottom")
        self._layout = pg.GraphicsLayoutWidget()
        self._price = self._layout.addPlot(row=0, col=0, axisItems={"bottom": self._time_axis})
        self._volume = self._layout.addPlot(row=1, col=0)
        self._layout.ci.layout.setRowStretchFactor(0, 4)
        self._layout.ci.layout.setRowStretchFactor(1, 1)
        self._price.showGrid(x=False, y=True, alpha=0.12)
        self._price.showAxis("right")
        self._price.getAxis("right").setStyle(showValues=True)
        self._price.setLabel("right", "price")
        self._legend = self._price.addLegend(offset=(10, 8), labelTextColor=p.text_dim)
        self._volume.setXLink(self._price)
        self._volume.showAxis("bottom", False)
        self._volume.setLabel("left", "vol")

        # O/H/L/C readout as a plain QLabel: updating a plot title forces a
        # graphics-layout pass on every mouse move; a label repaint does not.
        self._readout_label = QtWidgets.QLabel(" ")
        self._readout_label.setTextFormat(QtCore.Qt.RichText)
        self._readout_label.setFixedHeight(20)
        self._readout_label.setContentsMargins(8, 0, 8, 0)

        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addWidget(self._readout_label)
        box.addWidget(self._layout)

        self._view_start = 0  # global index of the first rendered bar
        self._trade_items: list = []
        self._whatif_items: list = []  # alternative-exit overlays, cleared separately
        self._level_lines: list = []  # draggable stop/target lines (kept across path redraws)
        self._level_proxies: list = []
        self._replay_items: list = []  # bar-by-bar animation items
        self._on_level_changed = None
        self._ohlc: dict | None = None  # current window arrays, for the crosshair readout
        self._labels: np.ndarray = np.array([])
        self._last_readout_i = -1
        self._vwap_items: dict[str, pg.PlotDataItem] = {}  # kind -> overlay line
        self._session_regions: list = []  # NY-session shading regions
        # Persistent slots for the one detailed trade and its excursion ribbon,
        # so a 60 Hz stop/target drag updates data in place instead of tearing
        # down and rebuilding the whole overlay layer each event.
        self._active_trade_items: dict | None = None
        self._excursion_items: dict | None = None

        # Crosshair (hidden until the cursor is over the plot).
        pen = pg.mkPen(p.text_faint, width=1, style=QtCore.Qt.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        for line in (self._vline, self._hline):
            line.setVisible(False)
            line.setZValue(20)
            self._price.addItem(line, ignoreBounds=True)

        self._price.scene().sigMouseClicked.connect(self._on_click)
        self._mouse_proxy = pg.SignalProxy(
            self._price.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

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
        self._legend.setLabelTextColor(p.text_dim)
        for plot in (self._price, self._volume):
            for name in ("left", "right", "bottom"):
                axis = plot.getAxis(name)
                if axis is not None:
                    axis.setPen(pg.mkPen(p.text_faint))
                    axis.setTextPen(pg.mkPen(p.text_dim))

    # -- rendering ---------------------------------------------------------
    def set_view(self, ohlc: dict, labels: np.ndarray, start_index: int) -> None:
        """Render a contiguous window.

        ``ohlc`` has open/high/low/close/volume arrays and, optionally, a
        ``segment`` array; candles are split at segment breaks so none is drawn
        across a roll/data discontinuity.
        """

        p = theme.active()
        self._price.clear()
        self._volume.clear()
        self._trade_items.clear()
        self._whatif_items.clear()
        self._level_lines.clear()  # removed by _price.clear(); drop stale references
        self._level_proxies.clear()
        self._replay_items.clear()
        self._vwap_items.clear()
        self._session_regions.clear()
        self._active_trade_items = None
        self._excursion_items = None
        self._last_readout_i = -1
        for line in (self._vline, self._hline):
            line.setVisible(False)
            self._price.addItem(line, ignoreBounds=True)
        self._view_start = int(start_index)
        self._ohlc = ohlc
        self._labels = np.asarray(labels)
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

    def set_vwap_visible(self, kind: str, on: bool) -> None:
        item = self._vwap_items.get(kind)
        if item is not None:
            item.setVisible(on)

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

    def mark_trades(self, local_entries, y, directions, gross_r) -> None:
        """Scatter entry markers at entry price, coloured by outcome (local indices)."""

        p = theme.active()
        brushes = [pg.mkBrush(p.up) if r > 0 else pg.mkBrush(p.down) for r in gross_r]
        symbols = ["t1" if d > 0 else "t" for d in directions]
        item = pg.ScatterPlotItem(
            x=list(local_entries),
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
        entry_local,
        exit_local,
        entry_price,
        stop_track,
        target_track,
        exit_price,
        *,
        color: str | None = None,
    ):
        """Draw (or update) the one detailed trade: level tracks and entry/exit.

        The four items persist between calls, so a stop/target drag at 60 Hz is
        four ``setData`` calls, not a teardown of the overlay layer.
        """

        p = theme.active()
        marker = color or p.entry
        items = self._active_trade_items
        if items is None:
            items = {
                "stop": pg.PlotDataItem(),
                "target": pg.PlotDataItem(),
                "entry": pg.ScatterPlotItem(symbol="o", size=12),
                "exit": pg.ScatterPlotItem(symbol="x", size=13),
            }
            for item in items.values():
                self._add_trade_item(item, ignore_bounds=True)
            self._active_trade_items = items
        items["stop"].setPen(pg.mkPen(p.stop, style=QtCore.Qt.DashLine))
        items["target"].setPen(pg.mkPen(p.target, style=QtCore.Qt.DashLine))
        if len(stop_track):
            held = np.arange(entry_local, entry_local + len(stop_track))
            items["stop"].setData(held, stop_track)
            items["target"].setData(held, target_track)
        else:
            items["stop"].clear()
            items["target"].clear()
        items["entry"].setData(x=[entry_local], y=[entry_price], brush=pg.mkBrush(marker))
        items["exit"].setData(x=[exit_local], y=[exit_price], pen=pg.mkPen(marker, width=2))

    def light_marker(self, entry_local, entry_price, exit_local, exit_price, gross_r, color):
        """A faint entry-to-exit connector for a non-active placed trade."""

        p = theme.active()
        outcome = p.up if gross_r > 0 else p.down
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
        x = np.asarray(exc.x) - self._view_start
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
        items["peak"].setData(
            x=[exc.peak_x - self._view_start],
            y=[exc.peak_price],
            brush=pg.mkBrush(p.hook if hook else p.target),
            pen=pg.mkPen(p.bg),
        )
        items["trough"].setData(
            x=[exc.trough_x - self._view_start],
            y=[exc.trough_price],
            brush=pg.mkBrush(p.stop),
            pen=pg.mkPen(p.bg),
        )
        items["label"].setColor(p.hook if hook else p.target)
        items["label"].setText(f"+{exc.mfe_r:.1f} R @ +{exc.time_to_peak_min}m")
        items["label"].setPos(exc.peak_x - self._view_start, exc.peak_price)

    def draw_whatif(self, runs) -> None:
        """Overlay each alternative-exit run's stop track and exit in its colour."""

        self.clear_whatif()
        for run in runs:
            r = run.result
            start = r.entry_position - self._view_start
            if len(r.stop_track):
                held = np.arange(start, start + len(r.stop_track))
                item = pg.PlotDataItem(
                    held,
                    r.stop_track,
                    pen=pg.mkPen(run.color, width=1.2, style=QtCore.Qt.DashDotLine),
                )
                self._price.addItem(item, ignoreBounds=True)
                self._whatif_items.append(item)
            marker = pg.ScatterPlotItem(
                x=[r.exit_position - self._view_start],
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
    def start_replay(self, result) -> None:
        """Set up the moving marker and stop/target lines for an animated replay."""

        self.stop_replay()
        p = theme.active()
        self._replay_marker = pg.ScatterPlotItem(
            symbol="o", size=13, brush=pg.mkBrush(p.gold), pen=pg.mkPen(p.bg)
        )
        self._replay_stop = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(p.stop, width=1.5, style=QtCore.Qt.DashLine)
        )
        self._replay_target = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(p.target, width=1.5, style=QtCore.Qt.DashLine)
        )
        self._replay_label = pg.TextItem("", color=p.text, anchor=(0, 1))
        self._replay_items = [
            self._replay_stop,
            self._replay_target,
            self._replay_marker,
            self._replay_label,
        ]
        for item in self._replay_items:
            self._price.addItem(item, ignoreBounds=True)

    def replay_frame(self, result, t: int) -> None:
        """Advance the animation to global bar ``t``."""

        if not self._replay_items or self._ohlc is None:
            return
        start = int(result.entry_position)
        idx = max(0, min(int(t) - start, len(result.stop_track) - 1))
        local = int(t) - self._view_start
        n = len(self._ohlc["close"])
        if not (0 <= local < n):
            return
        price = float(self._ohlc["close"][local])
        self._replay_marker.setData([local], [price])
        if len(result.stop_track):
            self._replay_stop.setPos(float(result.stop_track[idx]))
            self._replay_target.setPos(float(result.target_track[idx]))
        stop_points = float(result.initial_stop_points)
        if stop_points > 0 and np.isfinite(stop_points):
            running_r = (price - result.entry_price) / stop_points
            if result.direction < 0:
                running_r = -running_r
            label = f"+{int(t) - start}m   {running_r:+.2f} R"
        else:
            label = f"+{int(t) - start}m"
        self._replay_label.setText(label)
        self._replay_label.setPos(local, price)

    def stop_replay(self) -> None:
        for item in self._replay_items:
            self._price.removeItem(item)
        self._replay_items = []

    def center_on(self, local_index: int, pad: int | None = None) -> None:
        pad = theme.CHART_PAD_BARS if pad is None else int(pad)
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
        local = int(round(mouse_point.x()))
        if not (0 <= local < len(self._ohlc["close"])):
            return
        self.bar_clicked.emit(self._view_start + local)

    def _on_mouse_moved(self, evt) -> None:
        pos = evt[0]
        if self._ohlc is None or not self._price.sceneBoundingRect().contains(pos):
            self._vline.setVisible(False)
            self._hline.setVisible(False)
            return
        mp = self._price.vb.mapSceneToView(pos)
        i = int(round(mp.x()))
        self._vline.setPos(mp.x())
        self._hline.setPos(mp.y())
        self._vline.setVisible(True)
        self._hline.setVisible(True)
        n = len(self._ohlc["close"])
        if 0 <= i < n and i != self._last_readout_i:
            self._last_readout_i = i  # the readout only changes per bar, not per pixel
            self._readout_label.setText(self._readout(i))

    def _readout(self, i: int) -> str:
        p = theme.active()
        o = self._ohlc["open"][i]
        h = self._ohlc["high"][i]
        low = self._ohlc["low"][i]
        c = self._ohlc["close"][i]
        t = str(self._labels[i]) if i < len(self._labels) else ""
        up = c >= o
        colour = p.up if up else p.down
        return (
            f"<span style='color:{p.text_dim}'>{t}</span>  "
            f"<span style='color:{p.text_dim}'>O</span> {o:.1f}  "
            f"<span style='color:{p.text_dim}'>H</span> {h:.1f}  "
            f"<span style='color:{p.text_dim}'>L</span> {low:.1f}  "
            f"<span style='color:{colour}'>C {c:.1f}</span>"
        )
