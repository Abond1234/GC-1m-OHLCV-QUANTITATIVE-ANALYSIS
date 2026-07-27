"""Main window: chart + strategy replay + free-play + forensics.

Loads the Dev/Val universe and bars on a worker thread, renders one day at a time,
overlays VWAP and the New York session, and supports two modes:

* Replay - pick any catalog/peer strategy, plot its trades (coloured by outcome),
  click a trade to redraw its exact path and read why it worked or failed.
* Free-play - enable free-play, click the chart to place an entry (filled next
  bar), then size the exit freely in the Exit-rule panel and drag the stop/target
  on the chart to watch the outcome recompute live. Several trades can sit on the
  chart at once (the Free-play blotter); select one to make it active.

All fills come from the verified flexible-exit engine, so what you see on screen is
what the research engine would record.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis.excursion import compute_excursion, is_winner_on_the_hook
from ..analysis.grid_sweep import default_axes, sweep_entry
from ..analysis.placed_trade import place_trade, recompute_config, recompute_levels
from ..analysis.whatif import run_whatifs
from ..datalayer.catalog_service import StrategyReplayService
from ..datalayer.forensics import ForensicsService
from ..datalayer.paths import project_root
from ..datalayer.vwap import execution_session_vwap, research_day_vwap, rolling_vwap
from ..sim.exit_config import frozen_config
from ..sim.flex_exit import single_flex_exit
from ..workers.tasks import Task
from . import theme
from .blotter import TradeBlotter, ny_time_label
from .chart_widget import ChartWidget
from .exit_panel import ExitPanel
from .forensics_panel import ForensicsPanel
from .heatmap_widget import HeatmapWidget
from .replay_animator import ReplayAnimator

_WHATIF_COLUMNS = ["Exit rule", "Survived?", "R", "Exit", "Held"]

_TRADE_COLUMNS = ["Strategy", "Entry (NY)", "Dir", "Exit", "R", "Held"]


def _session_code(minute_ny: np.ndarray) -> np.ndarray:
    code = np.zeros(len(minute_ny), dtype=np.int8)
    a0, a1 = theme.ASIA_WINDOW
    n0, n1 = theme.NY_WINDOW
    code[(minute_ny >= a0) & (minute_ny < a1)] = 1
    code[(minute_ny >= n0) & (minute_ny < n1)] = 2
    return code


def load_app_data(root=None, date_floor=None) -> dict:
    """Worker payload: services, VWAP variants, session codes, and available dates.

    ``date_floor`` narrows the bar load to recent Dev days for the offscreen render
    harness; production passes ``None`` for the full Dev/Val window.
    """

    replay = StrategyReplayService.load(root or project_root(), date_floor=date_floor)
    bars = replay.bars
    dates = pd.to_datetime(pd.unique(bars.trade_date))
    return {
        "replay": replay,
        "forensics": ForensicsService(),
        "vwap20": rolling_vwap(bars, 20),
        "vwap_day": research_day_vwap(bars),
        "vwap_session": execution_session_vwap(bars),
        "session_code": _session_code(bars.minute_ny),
        "dates": np.sort(dates),
    }


class MainWindow(QtWidgets.QMainWindow):
    loadFinished = QtCore.Signal(bool)  # initial data load done (True) or failed (False)

    def __init__(self, data: dict | None = None):
        super().__init__()
        self.setWindowTitle("GC Trade Simulator - Development + Validation")
        self.setMinimumSize(960, 600)
        self._fit_to_screen(1600, 940)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._tasks: set = set()  # keep runnables alive until they finish (PySide6 GC)
        self._data: dict | None = None
        self._log: pd.DataFrame | None = None  # replay trade log
        self._view_start = 0
        self._view_end = 0
        self._replay_cfg = frozen_config()
        self._cfg = frozen_config()  # current free-play exit rule
        self._placed: dict[int, object] = {}  # id -> PlacedTrade
        self._active_id: int | None = None
        self._focused_result = None  # a clicked replay trade's detailed path
        self._focused_obs_id = None  # its observation id, for forensics re-fetches
        self._whatif_runs = []  # alternative-exit overlays for the current trade
        self._next_id = 1
        self._forensics_token = 0
        self._whatif_token = 0
        self._ny_label = None

        self._build_ui()
        if data is not None:
            # Synchronous path for the offscreen render harness and tests.
            self._on_loaded(data)
            return
        self._set_status("Loading Development + Validation data ...")
        task = Task(load_app_data)
        task.signals.finished.connect(self._on_loaded)
        task.signals.error.connect(self._on_error)
        self._start(task)

    def _fit_to_screen(self, preferred_w: int, preferred_h: int) -> None:
        """Open at the preferred size but never larger than the available screen.

        The window stays freely resizable and can be maximised; this only keeps the
        initial geometry from spilling off a display narrower or shorter than the
        layout was designed for. Centres the window within the available area.
        """

        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(preferred_w, preferred_h)
            return
        avail = screen.availableGeometry()
        width = min(preferred_w, avail.width())
        height = min(preferred_h, avail.height())
        self.resize(width, height)
        self.move(
            avail.x() + (avail.width() - width) // 2,
            avail.y() + (avail.height() - height) // 2,
        )

    # -- construction ------------------------------------------------------
    def _build_ui(self) -> None:
        self.chart = ChartWidget()
        self.chart.bar_clicked.connect(self._on_bar_clicked)
        self._animator = ReplayAnimator(self.chart)
        self._animator.stateChanged.connect(self._on_replay_state)
        self._animator.positionChanged.connect(self._on_replay_position)

        self._build_menu()
        self._build_topbar()
        self._build_side()
        self._build_exit_dock()
        self._build_transport()
        self._build_draw_toolbar()

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.chart)
        split.addWidget(self._right)
        split.setSizes([1080, 460])
        split.setCollapsible(0, False)  # the chart itself can never collapse
        split.setCollapsible(1, True)  # the sidebar can be dragged fully shut
        self._split = split

        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addLayout(self._controls)
        outer.addWidget(split, 1)
        self.setCentralWidget(central)
        self._set_status = self.statusBar().showMessage
        self._populate_view_menu()

    def _build_topbar(self) -> None:
        self.date_combo = QtWidgets.QComboBox()
        self.date_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.date_combo.currentTextChanged.connect(lambda _t: self._render_current_date())
        self.vwap_check = QtWidgets.QCheckBox("VWAP 20")
        self.vwap_check.setChecked(True)
        self.vwap_check.setToolTip("Rolling 20-bar volume-weighted average price.")
        self.vwap_day_check = QtWidgets.QCheckBox("VWAP day")
        self.vwap_day_check.setToolTip("VWAP anchored to the start of each New York trade date.")
        self.vwap_session_check = QtWidgets.QCheckBox("VWAP session")
        self.vwap_session_check.setToolTip("VWAP within each Asia/NY execution session.")
        self.session_check = QtWidgets.QCheckBox("NY session")
        self.session_check.setChecked(True)
        self.session_check.setToolTip("Shade the New York execution window (07:00-12:00 NY).")
        # Overlay toggles flip item visibility directly; rebuilding the whole
        # chart for a checkbox is what made these toggles feel heavy.
        self.vwap_check.toggled.connect(lambda on: self.chart.set_vwap_visible("rolling", on))
        self.vwap_day_check.toggled.connect(lambda on: self.chart.set_vwap_visible("day", on))
        self.vwap_session_check.toggled.connect(
            lambda on: self.chart.set_vwap_visible("session", on)
        )
        self.session_check.toggled.connect(self.chart.set_sessions_visible)

        self.strategy_combo = QtWidgets.QComboBox()
        # Long catalog names: keep the closed combo compact, let the popup widen.
        self.strategy_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.strategy_combo.setMinimumContentsLength(16)
        self.custom_check = QtWidgets.QCheckBox("Custom exits")
        self.custom_check.setToolTip(
            "Replay using the Exit-rule panel instead of the frozen contract."
        )
        self.replay_btn = QtWidgets.QPushButton("Replay")
        self.replay_btn.clicked.connect(self._on_replay)
        self.freeplay_check = QtWidgets.QCheckBox("Free-play (click to enter)")
        self.freeplay_check.setToolTip("Click the chart to place a trade at the next bar's open.")
        self.long_radio = QtWidgets.QRadioButton("Long")
        self.long_radio.setChecked(True)
        self.short_radio = QtWidgets.QRadioButton("Short")

        self.excursion_check = QtWidgets.QCheckBox("Excursion")
        self.excursion_check.setChecked(True)
        self.excursion_check.setToolTip(
            "Shade how far the selected trade ran in your favour (green) and against\n"
            "you (red), with the peak marked - so you can see a winner on the hook."
        )
        self.excursion_check.stateChanged.connect(lambda _s: self._redraw_overlays())
        self.whatif_btn = QtWidgets.QPushButton("What-if exits")
        self.whatif_btn.setToolTip(
            "Re-run the selected trade under a wider stop, a trailing stop, a longer\n"
            "cap, and a tighter target - to see which exit would have survived."
        )
        self.whatif_btn.clicked.connect(self._on_whatif)
        self.grid_btn = QtWidgets.QPushButton("Exit grid")
        self.grid_btn.setToolTip(
            "Sweep a grid of stop x target exits for the selected trade and heatmap\n"
            "the outcome. Exploratory - the best cell is partly luck, not an edge."
        )
        self.grid_btn.clicked.connect(self._on_exit_grid)

        # Two rows so nothing clips off the right edge on a narrow display:
        # row 1 = what is on the chart, row 2 = acting on it.
        row1 = QtWidgets.QHBoxLayout()
        for w in (
            QtWidgets.QLabel("Date"),
            self.date_combo,
            self.vwap_check,
            self.vwap_day_check,
            self.vwap_session_check,
            self.session_check,
            _sep(),
            QtWidgets.QLabel("Strategy"),
            self.strategy_combo,
            self.custom_check,
            self.replay_btn,
        ):
            row1.addWidget(w)
        row1.addStretch(1)
        row2 = QtWidgets.QHBoxLayout()
        for w in (
            self.freeplay_check,
            self.long_radio,
            self.short_radio,
            _sep(),
            self.excursion_check,
            self.whatif_btn,
            self.grid_btn,
        ):
            row2.addWidget(w)
        row2.addStretch(1)
        self._controls = QtWidgets.QVBoxLayout()
        self._controls.setSpacing(4)
        self._controls.addLayout(row1)
        self._controls.addLayout(row2)

    def _build_side(self) -> None:
        self.trade_table = QtWidgets.QTableWidget(0, len(_TRADE_COLUMNS))
        self.trade_table.setHorizontalHeaderLabels(_TRADE_COLUMNS)
        trade_header = self.trade_table.horizontalHeader()
        trade_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        trade_header.setStretchLastSection(True)
        self.trade_table.verticalHeader().setVisible(False)
        self.trade_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.trade_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.trade_table.itemSelectionChanged.connect(self._on_row_selected)

        self.blotter = TradeBlotter()
        self.blotter.tradeSelected.connect(self._on_placed_selected)
        self.blotter.tradeRemoved.connect(self._on_placed_removed)
        self.blotter.cleared.connect(self._on_placed_cleared)

        self.whatif_table = QtWidgets.QTableWidget(0, len(_WHATIF_COLUMNS))
        self.whatif_table.setHorizontalHeaderLabels(_WHATIF_COLUMNS)
        whatif_header = self.whatif_table.horizontalHeader()
        whatif_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        whatif_header.setStretchLastSection(True)
        self.whatif_table.verticalHeader().setVisible(False)
        self.whatif_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.heatmap = HeatmapWidget()
        self.heatmap.cellChosen.connect(self._on_grid_cell)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self.trade_table, "Replay")
        self._tabs.addTab(self.blotter, "Free-play")
        self._tabs.addTab(self.whatif_table, "What-if")
        self._tabs.addTab(self.heatmap, "Exit grid")

        self.forensics = ForensicsPanel()

        self._right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._right.addWidget(self._tabs)
        self._right.addWidget(self.forensics)
        self._right.setSizes([460, 440])

    def _build_exit_dock(self) -> None:
        self.exit_panel = ExitPanel()
        self.exit_panel.configChanged.connect(self._on_config_changed)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.exit_panel)
        dock = QtWidgets.QDockWidget("Exit rule (unfrozen)", self)
        dock.setToolTip(
            "Every exit field here is editable - deliberately not the frozen research contract."
        )
        dock.setWidget(scroll)
        dock.setMinimumWidth(260)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self._exit_dock = dock

    def _build_menu(self) -> None:
        self._view_menu = self.menuBar().addMenu("View")
        for label, mode in (("Dark theme", "dark"), ("Light theme", "light")):
            act = self._view_menu.addAction(label)
            act.triggered.connect(lambda _c=False, m=mode: self._set_theme(m))

    def _populate_view_menu(self) -> None:
        """Panel toggles, added once the docks/toolbars they control exist."""

        menu = self._view_menu
        menu.addSeparator()
        exit_act = self._exit_dock.toggleViewAction()
        exit_act.setText("Exit-rule panel")
        exit_act.setShortcut("Ctrl+1")
        menu.addAction(exit_act)
        side_act = menu.addAction("Analysis sidebar")
        side_act.setCheckable(True)
        side_act.setChecked(True)
        side_act.setShortcut("Ctrl+2")
        side_act.toggled.connect(self._right.setVisible)
        self._sidebar_action = side_act
        transport_act = self._transport.toggleViewAction()
        transport_act.setText("Replay transport")
        transport_act.setShortcut("Ctrl+3")
        menu.addAction(transport_act)
        draw_act = self._draw_toolbar.toggleViewAction()
        draw_act.setText("Draw toolbar")
        draw_act.setShortcut("Ctrl+4")
        menu.addAction(draw_act)

    def _build_transport(self) -> None:
        bar = QtWidgets.QToolBar("Replay")
        bar.setMovable(False)
        self.replay_day_btn = QtWidgets.QPushButton("Replay day")
        self.replay_day_btn.setToolTip(
            "Hide this day's bars and play the tape from its first bar - candles\n"
            "appear one by one with the future hidden. Free-play entries can be\n"
            "placed on revealed bars while it runs."
        )
        self.replay_day_btn.clicked.connect(self._replay_day)
        self.animate_btn = QtWidgets.QPushButton("Animate trade")
        self.animate_btn.setToolTip(
            "Replay the selected trade bar by bar from entry to exit, with the\n"
            "future hidden - watch price walk into your stop or target."
        )
        self.animate_btn.clicked.connect(self._animate_current)
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._animator.toggle)
        step_back = QtWidgets.QPushButton("<")
        step_back.clicked.connect(lambda: self._animator.step(-1))
        step_fwd = QtWidgets.QPushButton(">")
        step_fwd.clicked.connect(lambda: self._animator.step(1))
        stop_btn = QtWidgets.QPushButton("Stop")
        stop_btn.clicked.connect(self._animator.stop)
        self.speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed.setRange(20, 400)  # ms per bar (right = slower)
        self.speed.setValue(120)
        self.speed.setFixedWidth(120)
        self.speed.setToolTip("Playback speed (left = faster).")
        self.speed.valueChanged.connect(self._animator.set_interval)
        self.replay_pos = QtWidgets.QLabel("")
        # Reserve the widest plausible text so per-tick updates never trigger a
        # toolbar relayout mid-animation.
        self.replay_pos.setMinimumWidth(
            self.replay_pos.fontMetrics().horizontalAdvance("  +8888 / 8888 min")
        )
        for w in (
            self.replay_day_btn,
            self.animate_btn,
            step_back,
            self.play_btn,
            step_fwd,
            stop_btn,
            QtWidgets.QLabel(" speed"),
            self.speed,
            self.replay_pos,
        ):
            bar.addWidget(w)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, bar)
        self._transport = bar

    def _build_draw_toolbar(self) -> None:
        """Vertical drawing strip on the left edge, TradingView-style."""

        bar = QtWidgets.QToolBar("Draw")
        bar.setMovable(False)
        bar.setOrientation(QtCore.Qt.Vertical)
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        self._draw_actions = {}
        for label, mode, tip in (
            ("Cursor", None, "Normal interaction: crosshair, free-play clicks, drags."),
            ("Level", "hline", "Click a price to drop a draggable horizontal level."),
            ("Trend", "trend", "Two clicks place a trendline; drag its endpoints to adjust."),
            ("Time", "vline", "Click a bar to mark a moment with a draggable vertical line."),
        ):
            act = bar.addAction(label)
            act.setCheckable(True)
            act.setToolTip(tip)
            act.setActionGroup(group)
            act.triggered.connect(lambda _c=False, m=mode: self.chart.set_draw_mode(m))
            self._draw_actions[mode] = act
        self._draw_actions[None].setChecked(True)
        bar.addSeparator()
        undo = bar.addAction("Undo")
        undo.setToolTip("Remove the most recent drawing on this day.")
        undo.triggered.connect(self.chart.undo_drawing)
        clear = bar.addAction("Clear")
        clear.setToolTip("Remove every drawing on this day.")
        clear.triggered.connect(self.chart.clear_drawings)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, bar)
        self._draw_toolbar = bar
        # One-shot tools: fall back to the cursor once a drawing lands.
        self.chart.drawing_placed.connect(self._on_drawing_placed)

    def _on_drawing_placed(self) -> None:
        self.chart.set_draw_mode(None)
        self._draw_actions[None].setChecked(True)

    def _start(self, task) -> None:
        """Start a worker, retaining a reference so PySide6 does not GC it early."""

        self._tasks.add(task)
        task.signals.finished.connect(lambda *_a: self._tasks.discard(task))
        task.signals.error.connect(lambda *_a: self._tasks.discard(task))
        self._pool.start(task)

    # -- loading -----------------------------------------------------------
    def _on_error(self, message: str) -> None:
        if self._data is None:
            self.loadFinished.emit(False)  # let the splash release before the modal
        self._set_status("Error - see dialog")
        QtWidgets.QMessageBox.critical(self, "Load error", message)

    def _on_loaded(self, data: dict) -> None:
        self._data = data
        self._ny_label = ny_time_label(data["replay"].bars)
        self.blotter.set_time_label(self._ny_label)
        self.date_combo.blockSignals(True)
        self.date_combo.addItems([pd.Timestamp(d).strftime("%Y-%m-%d") for d in data["dates"]])
        self.date_combo.blockSignals(False)
        for spec in data["replay"].list_strategies():
            self.strategy_combo.addItem(spec.name)
        # The closed combo stays compact; widen the popup to the longest name.
        metrics = self.strategy_combo.fontMetrics()
        widest = max(
            (
                metrics.horizontalAdvance(self.strategy_combo.itemText(i))
                for i in range(self.strategy_combo.count())
            ),
            default=0,
        )
        self.strategy_combo.view().setMinimumWidth(widest + 48)
        self._set_status(f"Loaded {data['replay'].bars.n_bars:,} GC bars (Dev+Val).")
        self._render_current_date()
        self.loadFinished.emit(True)

    # -- rendering ---------------------------------------------------------
    @property
    def _bars(self):
        return self._data["replay"].bars

    def _render_current_date(self) -> None:
        if self._data is None or not self.date_combo.count():
            return
        self._animator.stop()  # leaving a day ends any running animation
        bars = self._bars
        date = np.datetime64(pd.Timestamp(self.date_combo.currentText()))
        # Bars are chronological, so the day window is two binary searches, not
        # a full-array comparison over the whole Dev+Val history.
        lo = int(np.searchsorted(bars.trade_date, date, side="left"))
        hi = int(np.searchsorted(bars.trade_date, date, side="right")) - 1
        if hi < lo:
            return
        self._view_start, self._view_end = lo, hi
        sl = slice(lo, hi + 1)
        minute = bars.minute_ny[sl]
        labels = np.array([f"{m // 60:02d}:{m % 60:02d}" for m in minute])
        ohlc = {
            "open": bars.open[sl],
            "high": bars.high[sl],
            "low": bars.low[sl],
            "close": bars.close[sl],
            "volume": bars.volume[sl],
            "segment": bars.segment[sl],
        }
        self.chart.set_view(ohlc, labels, lo)
        # Always build the overlays; the checkboxes only flip visibility.
        self.chart.add_vwap(
            self._data["vwap20"][sl], "rolling", visible=self.vwap_check.isChecked()
        )
        self.chart.add_vwap(
            self._data["vwap_day"][sl], "day", visible=self.vwap_day_check.isChecked()
        )
        self.chart.add_vwap(
            self._data["vwap_session"][sl],
            "session",
            visible=self.vwap_session_check.isChecked(),
        )
        self.chart.shade_sessions(
            self._data["session_code"][sl], visible=self.session_check.isChecked()
        )
        self._redraw_overlays()

    def _in_view(self, position: int) -> bool:
        return self._view_start <= position <= self._view_end

    def _redraw_overlays(self, reset_levels: bool = True) -> None:
        """Redraw replay markers, a focused replay path, and all placed trades."""

        self.chart.clear_trades()
        self._draw_markers_for_view()
        if self._focused_result is not None and self._in_view(self._focused_result.entry_position):
            self._draw_result_path(self._focused_result)
        active = self._placed.get(self._active_id)
        for t in self._placed.values():
            if not self._in_view(t.entry_position):
                continue
            if t is active:
                self._draw_result_path(t.result, color=t.color)
            else:
                self._draw_light(t)
        self._draw_analysis_overlays()
        if reset_levels:
            self.chart.clear_draggable_levels()
            if active is not None and self._in_view(active.entry_position):
                stop_price, target_price = self._level_prices(active)
                self.chart.set_draggable_levels(stop_price, target_price, self._on_level_dragged)

    def _draw_analysis_overlays(self) -> None:
        """Excursion ribbon and any what-if overlays for the current trade."""

        result, _cfg = self._current()
        if result is None or not self._in_view(result.entry_position):
            return
        if self.excursion_check.isChecked():
            exc = compute_excursion(result, self._bars.high, self._bars.low)
            self.chart.draw_excursion(exc, hook=is_winner_on_the_hook(result, theme.HOOK_R))
        if self._whatif_runs:
            self.chart.draw_whatif(self._whatif_runs)

    def _current(self):
        """The trade currently being explained: (result, cfg) or (None, None)."""

        if self._focused_result is not None:
            return self._focused_result, self._replay_cfg
        active = self._placed.get(self._active_id)
        if active is not None:
            return active.result, active.cfg
        return None, None

    def _draw_result_path(self, result, color=None) -> None:
        self.chart.draw_trade(
            result.entry_position - self._view_start,
            result.exit_position - self._view_start,
            result.entry_price,
            result.stop_track,
            result.target_track,
            result.exit_price,
            color=color,
        )

    def _draw_light(self, t) -> None:
        r = t.result
        self.chart.light_marker(
            r.entry_position - self._view_start,
            r.entry_price,
            r.exit_position - self._view_start,
            r.exit_price,
            r.gross_r,
            t.color,
        )

    def _draw_markers_for_view(self) -> None:
        if self._log is None or self._log.empty:
            return
        bars = self._bars
        inside = self._log[
            (self._log["entry_position"] >= self._view_start)
            & (self._log["entry_position"] <= self._view_end)
        ]
        if inside.empty:
            return
        entries = inside["entry_position"].to_numpy()
        self.chart.mark_trades(
            entries - self._view_start,
            bars.open[entries],
            inside["direction"].to_numpy(),
            inside["gross_r"].to_numpy(),
        )

    def _level_prices(self, t) -> tuple[float, float]:
        entry = t.result.entry_price
        if t.direction > 0:
            return entry - t.stop_points, entry + t.target_points
        return entry + t.stop_points, entry - t.target_points

    def _ensure_date_for(self, position: int) -> bool:
        """Switch the view to the bar's date; True if that re-rendered the chart."""

        date = pd.Timestamp(self._bars.trade_date[position]).strftime("%Y-%m-%d")
        if self.date_combo.currentText() != date:
            self.date_combo.setCurrentText(date)  # triggers _render_current_date
            return True
        return False

    # -- replay ------------------------------------------------------------
    def _on_replay(self) -> None:
        if self._data is None:
            return
        name = self.strategy_combo.currentText()
        specs = {s.name: s for s in self._data["replay"].list_strategies()}
        if name not in specs:
            return
        custom = self.custom_check.isChecked()
        # The config is captured here but only installed when its log arrives,
        # so clicking rows of the still-visible previous log keeps using the
        # config that log was actually produced with.
        cfg_used = self.exit_panel.to_config() if custom else frozen_config()
        cfg = cfg_used if custom else None
        self._set_status(f"Replaying {name} ...")
        task = Task(self._data["replay"].replay, specs[name], cfg)
        task.signals.finished.connect(lambda log, c=cfg_used: self._on_replayed(log, c))
        task.signals.error.connect(self._on_error)
        self._start(task)

    def _on_replayed(self, log: pd.DataFrame, cfg=None) -> None:
        if cfg is not None:
            self._replay_cfg = cfg
        self._log = log
        self.trade_table.setRowCount(0)
        wins = int((log["gross_r"] > 0).sum()) if not log.empty else 0
        self._set_status(
            f"{len(log):,} trades, {wins:,} winners, mean R {log['gross_r'].mean():.3f}"
            if not log.empty
            else "No trades fired."
        )
        for _, row in log.head(theme.TABLE_ROW_CAP).iterrows():
            self._append_trade_row(row)
        self._tabs.setCurrentWidget(self.trade_table)
        self._render_current_date()

    def _append_trade_row(self, row) -> None:
        r = self.trade_table.rowCount()
        self.trade_table.insertRow(r)
        values = [
            str(row.get("strategy", "")),
            self._ny_label(int(row["entry_position"])),
            "L" if row["direction"] > 0 else "S",
            str(row["exit_reason"]),
            f"{row['gross_r']:+.2f}",
            str(int(row["holding_minutes"])),
        ]
        for c, v in enumerate(values):
            item = QtWidgets.QTableWidgetItem(v)
            item.setData(QtCore.Qt.UserRole, int(row["entry_position"]))
            item.setToolTip(v)  # full text survives any column elision
            self.trade_table.setItem(r, c, item)

    def _on_row_selected(self) -> None:
        rows = self.trade_table.selectionModel().selectedRows()
        if not rows or self._log is None:
            return
        item = self.trade_table.item(rows[0].row(), 0)
        if item is None:  # selection event racing a table refresh
            return
        entry_position = item.data(QtCore.Qt.UserRole)
        match = self._log[self._log["entry_position"] == entry_position]
        if not match.empty:
            self._focus_trade(match.iloc[0])

    def _focus_trade(self, trade) -> None:
        result = single_flex_exit(
            int(trade["entry_position"]),
            int(trade["direction"]),
            float(trade["initial_stop_points"]),
            float(trade["initial_target_points"]),
            self._replay_cfg,
            self._bars.bar_arrays(),
            float(trade.get("trail_points", 0.0)),
        )
        self._focused_result = result
        self._focused_obs_id = trade.get("observation_id")
        self._active_id = None  # a replay focus is not a draggable placed trade
        self._whatif_runs = []
        # A date switch re-renders (and redraws overlays) on its own; only
        # redraw explicitly when the view stayed put.
        if not self._ensure_date_for(int(trade["entry_position"])):
            self._redraw_overlays()
        self.chart.center_on(result.entry_position - self._view_start)
        self._explain(result, observation_id=self._focused_obs_id)

    # -- free-play ---------------------------------------------------------
    def _on_bar_clicked(self, global_index: int) -> None:
        if self._data is None or not self.freeplay_check.isChecked():
            return
        bars = self._bars
        entry = global_index + 1  # fill on the next bar's open
        if entry >= bars.n_bars:
            return
        direction = 1 if self.long_radio.isChecked() else -1
        cfg = self.exit_panel.to_config()
        self._cfg = cfg
        cycle = theme.active().whatif_cycle
        color = cycle[self._next_id % len(cycle)]
        trade = place_trade(
            self._next_id, entry, direction, cfg, bars.atr20, bars.bar_arrays(), color=color
        )
        self._placed[trade.id] = trade
        self._active_id = trade.id
        self._focused_result = None
        self._focused_obs_id = None
        self._whatif_runs = []
        self._next_id += 1
        self._show_active_trade(center=True)

    def _show_active_trade(self, center: bool = False) -> None:
        active = self._placed.get(self._active_id)
        if active is None:
            self._redraw_overlays()
            return
        if not self._ensure_date_for(active.entry_position):  # a switch redraws itself
            self._redraw_overlays()
        if center:
            self.chart.center_on(active.entry_position - self._view_start)
        self.blotter.set_trades(list(self._placed.values()))
        self.blotter.select_trade(active.id)
        self._tabs.setCurrentWidget(self.blotter)
        self._explain(active.result, observation_id=None)

    def _on_level_dragged(self, kind: str, price: float) -> None:
        active = self._placed.get(self._active_id)
        if active is None:
            return
        entry = active.result.entry_price
        if kind == "stop":
            pts = (entry - price) if active.direction > 0 else (price - entry)
            updated = recompute_levels(active, self._bars.bar_arrays(), stop_points=pts)
        else:
            pts = (price - entry) if active.direction > 0 else (entry - price)
            updated = recompute_levels(active, self._bars.bar_arrays(), target_points=pts)
        self._placed[active.id] = updated
        # Fast path: update the persistent chart items in place instead of a
        # full overlay teardown - this runs at up to 60 Hz during a drag.
        if self._whatif_runs:
            self._whatif_runs = []  # config changed under them; they are stale
            self.whatif_table.setRowCount(0)
            self.chart.clear_whatif()
        result = updated.result
        self.chart.draw_trade(
            result.entry_position - self._view_start,
            result.exit_position - self._view_start,
            result.entry_price,
            result.stop_track,
            result.target_track,
            result.exit_price,
            color=updated.color,
        )
        if self.excursion_check.isChecked():
            exc = compute_excursion(result, self._bars.high, self._bars.low)
            self.chart.draw_excursion(exc, hook=is_winner_on_the_hook(result, theme.HOOK_R))
        self.blotter.update_trade(updated)
        self._explain(updated.result, observation_id=None)

    def _on_config_changed(self, cfg) -> None:
        self._cfg = cfg
        active = self._placed.get(self._active_id)
        if active is None:
            return
        if self._whatif_runs:
            self._whatif_runs = []  # computed against the previous config
            self.whatif_table.setRowCount(0)
        updated = recompute_config(active, cfg, self._bars.atr20, self._bars.bar_arrays())
        self._placed[active.id] = updated
        self._redraw_overlays(reset_levels=True)
        self.blotter.set_trades(list(self._placed.values()))
        self.blotter.select_trade(active.id)
        self._explain(updated.result, observation_id=None)

    def _on_placed_selected(self, trade_id: int) -> None:
        if trade_id in self._placed:
            self._active_id = trade_id
            self._focused_result = None
            self._focused_obs_id = None
            self._whatif_runs = []
            self._show_active_trade(center=True)

    def _drop_stale_analysis(self) -> None:
        """Clear what-if state and re-point forensics after the subject changed."""

        self._whatif_runs = []
        self.whatif_table.setRowCount(0)
        active = self._placed.get(self._active_id)
        if active is not None:
            self._explain(active.result, observation_id=None)
        elif self._focused_result is not None:
            self._explain(self._focused_result, observation_id=self._focused_obs_id)
        else:
            self.forensics.clear()

    def _on_placed_removed(self, trade_id: int) -> None:
        self._placed.pop(trade_id, None)
        if self._active_id == trade_id:
            self._active_id = next(iter(self._placed), None)
        self.blotter.set_trades(list(self._placed.values()))
        if self._active_id is not None:
            self.blotter.select_trade(self._active_id)
        self._drop_stale_analysis()
        self._redraw_overlays()

    def _on_placed_cleared(self) -> None:
        self._placed.clear()
        self._active_id = None
        self.blotter.set_trades([])
        self._drop_stale_analysis()
        self._redraw_overlays()

    # -- forensics ---------------------------------------------------------
    def _explain(self, result, observation_id) -> None:
        self.forensics.show_realized(result)
        if observation_id is None or not self._data["forensics"].available():
            return
        self._forensics_token += 1
        token = self._forensics_token
        task = Task(self._data["forensics"].context_for, int(observation_id))
        task.signals.finished.connect(
            lambda ctx, r=result, t=token: self._on_forensics_ctx(r, ctx, t)
        )
        task.signals.error.connect(self._on_error)
        self._start(task)

    def _on_forensics_ctx(self, result, ctx, token) -> None:
        if token != self._forensics_token:
            return  # a newer selection superseded this fetch
        self.forensics.show_context(result, ctx)

    # -- what-if -----------------------------------------------------------
    def _on_whatif(self) -> None:
        result, cfg = self._current()
        if result is None:
            return
        self._whatif_token += 1
        token = self._whatif_token
        entry = int(result.entry_position)
        self._set_status("Running what-if exits ...")
        task = Task(
            run_whatifs,
            result.entry_position,
            result.direction,
            cfg,
            self._bars.atr20,
            self._bars.bar_arrays(),
            theme.active().whatif_cycle,
        )
        task.signals.finished.connect(
            lambda runs, t=token, e=entry: self._on_whatifs_ready(runs, t, e)
        )
        task.signals.error.connect(self._on_error)
        self._start(task)

    def _on_whatifs_ready(self, runs, token: int, entry: int) -> None:
        if token != self._whatif_token:
            return  # a newer request superseded this sweep
        result, _cfg = self._current()
        if result is None or int(result.entry_position) != entry:
            return  # the user moved on to a different trade meanwhile
        self._whatif_runs = runs
        self._fill_whatif_table(runs)
        self._tabs.setCurrentWidget(self.whatif_table)
        self._redraw_overlays(reset_levels=False)
        self._set_status(f"{len(runs)} alternative exits over the same entry.")

    def _fill_whatif_table(self, runs) -> None:
        p = theme.active()
        self.whatif_table.setRowCount(0)
        for run in runs:
            r = run.result
            row = self.whatif_table.rowCount()
            self.whatif_table.insertRow(row)
            values = [
                run.label,
                "yes" if run.survived else "no",
                f"{r.gross_r:+.2f}",
                str(r.exit_reason),
                str(r.holding_minutes),
            ]
            for c, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(v)
                item.setToolTip(v)
                if c == 0:
                    item.setForeground(QtGui.QColor(run.color))
                elif c == 2:
                    item.setForeground(QtGui.QColor(p.up if r.gross_r > 0 else p.down))
                self.whatif_table.setItem(row, c, item)

    # -- exit-grid explorer ------------------------------------------------
    def _on_exit_grid(self) -> None:
        result, cfg = self._current()
        if result is None:
            return
        stop_mults, target_rs = default_axes()
        self._set_status("Sweeping exit grid ...")
        task = Task(
            sweep_entry,
            result.entry_position,
            result.direction,
            cfg,
            self._bars.atr20,
            self._bars.bar_arrays(),
            stop_mults,
            target_rs,
        )
        task.signals.finished.connect(self._on_grid_ready)
        task.signals.error.connect(self._on_error)
        self._start(task)

    def _on_grid_ready(self, grid) -> None:
        self.heatmap.set_grid(grid)
        self._tabs.setCurrentWidget(self.heatmap)
        self._set_status(f"Exit grid: {grid.trials} exits swept on one entry (exploratory).")

    def _on_grid_cell(self, stop_mult: float, target_r: float) -> None:
        cfg = replace(
            self.exit_panel.to_config(),
            stop_mode="atr",
            stop_value=stop_mult,
            target_mode="r",
            target_value=target_r,
        )
        if self._active_id is not None:
            self.exit_panel.from_config(cfg)  # configChanged -> recompute the active trade
            return
        if self._focused_result is None:
            self._set_status("Select or place a trade, sweep the grid, then click a cell.")
            return
        # A focused replay trade has no live recompute path through the panel,
        # so honour "click to apply" directly: re-simulate this entry under the
        # chosen cell (the exact sim the heatmap cell reports) and show it.
        self.exit_panel.from_config(cfg)  # reflect the choice; no active trade to touch
        base = self._focused_result
        trade = place_trade(
            0,
            int(base.entry_position),
            int(base.direction),
            cfg,
            self._bars.atr20,
            self._bars.bar_arrays(),
        )
        self._focused_result = trade.result
        self._whatif_runs = []
        self._redraw_overlays()
        self._explain(trade.result, observation_id=self._focused_obs_id)
        self._set_status(
            f"Applied stop {stop_mult:.2f}x ATR / target {target_r:.1f}R to this entry "
            f"(exploratory)."
        )

    # -- animated replay ---------------------------------------------------
    def _replay_day(self) -> None:
        """Play the whole visible day tape-style, no trade required."""

        if self._data is None or self._view_end <= self._view_start:
            return
        self._animator.load_day(self._view_start, self._view_end)
        self._animator.play()

    def _animate_current(self) -> None:
        result, _cfg = self._current()
        if result is None:
            return
        self._ensure_date_for(result.entry_position)
        self.chart.center_on(result.entry_position - self._view_start)
        self._animator.load(result)
        self._animator.play()

    def _on_replay_state(self, playing: bool) -> None:
        self.play_btn.setText("Pause" if playing else "Play")
        if not playing and not self._animator.active:
            self.replay_pos.setText("")  # the animation ended; do not show a stale position

    def _on_replay_position(self, offset: int, total: int) -> None:
        text = f"  +{offset} / {total} min"
        if text != self.replay_pos.text():  # skip no-op relayouts at 50 fps
            self.replay_pos.setText(text)

    # -- theme -------------------------------------------------------------
    def _set_theme(self, mode: str) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            theme.apply(app, mode)
        self.chart.apply_theme()
        self._render_current_date()
        # Re-render everything that baked the previous palette into itself.
        self.heatmap.refresh_theme()
        self.forensics.retheme()
        if self._placed:
            self.blotter.set_trades(list(self._placed.values()))
            if self._active_id is not None:
                self.blotter.select_trade(self._active_id)
        if self._whatif_runs:
            self._fill_whatif_table(self._whatif_runs)
        result, _cfg = self._current()
        if result is not None:
            # Keep the forensics context alive across the switch: re-fetch with
            # the focused trade's observation id instead of dropping to the
            # realized-only view.
            obs = self._focused_obs_id if self._focused_result is not None else None
            self._explain(result, observation_id=obs)
            if self._in_view(result.entry_position):
                self.chart.center_on(result.entry_position - self._view_start)

    # -- lifecycle ---------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._animator.stop()
        self._forensics_token += 1  # orphan any in-flight worker completions
        self._whatif_token += 1
        self._pool.waitForDone(1500)  # let workers drain before Qt teardown
        super().closeEvent(event)


def _sep() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.VLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line
