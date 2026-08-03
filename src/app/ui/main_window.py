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

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis.account import build_equity_curve, session_stats
from ..analysis.evaluation import evaluate
from ..analysis.excursion import compute_excursion, is_winner_on_the_hook
from ..analysis.grid_sweep import default_axes, sweep_entry
from ..analysis.placed_trade import place_trade, recompute_config, recompute_levels
from ..analysis.whatif import run_whatifs
from ..datalayer.bar_store import BarStore
from ..datalayer.catalog_service import StrategyReplayService
from ..datalayer.edge_context import EdgeContextService
from ..datalayer.forensics import ForensicsService
from ..datalayer.instruments import REGISTRY, available_instruments, load_instrument_bars
from ..datalayer.paths import project_root, research_bars_path
from ..datalayer.session_store import (
    build_payload,
    default_sessions_dir,
    exit_config_from_dict,
    exit_config_to_dict,
    read_session,
    write_session,
)
from ..datalayer.timeframe import (
    TIMEFRAMES,
    ViewMap,
    bucket_labels,
    resample_window,
    sample_first,
    sample_last,
)
from ..datalayer.vwap import execution_session_vwap, research_day_vwap, rolling_vwap
from ..sim.exit_config import frozen_config
from ..sim.flex_exit import single_flex_exit
from ..workers.tasks import Task
from . import theme
from .blotter import TradeBlotter, ny_time_label
from .chart_widget import ChartWidget
from .edge_panel import EdgeContextPanel
from .exit_panel import ExitPanel
from .forensics_panel import ForensicsPanel
from .heatmap_widget import HeatmapWidget
from .replay_animator import ReplayAnimator
from .session_panel import SessionPanel

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
        "date_floor": date_floor,
    }


def load_instrument_bundle(symbol: str, date_floor=None) -> dict:
    """Worker payload for a non-GC primary instrument: bars + derived overlays."""

    bars = load_instrument_bars(symbol, date_floor=date_floor)
    dates = pd.to_datetime(pd.unique(bars.trade_date))
    return {
        "symbol": symbol,
        "bars": bars,
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
        self._mgc_bars = None  # MGC BarStore, loaded on first toggle
        self._mgc_loading = False
        self._mgc_range: tuple[int, int, int] | None = None  # rendered MGC window + tf
        self._instrument = "GC"  # active primary instrument symbol
        self._instrument_data: dict | None = None  # non-GC bundle (None = GC research)
        self._instrument_loading = False
        self._pending_session: dict | None = None  # session applied after a switch
        self._date_guard = False  # suppress re-render while setting both date edits
        self._edge_service = EdgeContextService()
        self._edge_token = 0
        self._edge_day: tuple[int, int] | None = None  # anchor day's 1m [lo, hi]

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
        self.chart.bar_hovered.connect(self._on_bar_hovered)
        self._animator = ReplayAnimator(self.chart)
        self._animator.stateChanged.connect(self._on_replay_state)
        self._animator.positionChanged.connect(self._on_replay_position)

        # MGC (micro gold) mirror pane: same day, same clock, the execution
        # instrument's own tape. Hidden until toggled; bars load on demand.
        self.mgc_chart = ChartWidget()
        self.mgc_chart._price.setLabel("left", "MGC (micro)")
        self.mgc_chart.setVisible(False)

        self._build_menu()
        self._build_topbar()
        self._build_side()
        self._build_exit_panel()
        self._build_draw_actions()
        self._build_workspaces()
        self._build_transport()

        chart_col = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        chart_col.addWidget(self.chart)
        chart_col.addWidget(self.mgc_chart)
        chart_col.setSizes([650, 320])
        chart_col.setCollapsible(0, False)
        self._chart_col = chart_col

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._workspace_panel)  # collapsible left workspace
        split.addWidget(chart_col)  # the chart stays central
        split.addWidget(self._right)  # trade inspector
        split.setSizes([320, 980, 380])
        split.setStretchFactor(1, 1)
        split.setCollapsible(0, True)
        split.setCollapsible(1, False)  # the chart itself never collapses
        split.setCollapsible(2, True)
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
        self.instrument_combo = QtWidgets.QComboBox()
        self.instrument_combo.setToolTip(
            "Primary chart instrument. Only assets with local data are listed;\n"
            "see the app README's 'Adding an instrument' for NQ/ES/BTCUSD.\n"
            "Strategy replay and Edge context stay GC-only (research validity)."
        )
        self.instrument_combo.currentIndexChanged.connect(self._on_instrument_changed)
        # Literal date window: Start and End load exactly the requested interval,
        # with no "ending at the selected date" lookback behaviour.
        self.start_edit = QtWidgets.QDateEdit()
        self.end_edit = QtWidgets.QDateEdit()
        for edit, tip in (
            (self.start_edit, "First trade date to load (inclusive)."),
            (self.end_edit, "Last trade date to load (inclusive)."),
        ):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
            edit.setToolTip(tip + " The chart loads exactly this interval.")
            edit.dateChanged.connect(self._on_dates_changed)
        self.tf_combo = QtWidgets.QComboBox()
        self.tf_combo.addItems(list(TIMEFRAMES))
        self.tf_combo.setToolTip(
            "Chart display timeframe. Aggregation is display-only: every\n"
            "simulation, entry, exit, and replay stays on true 1-minute bars."
        )
        self.tf_combo.currentTextChanged.connect(lambda _t: self._render_view())
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
        # Searchable: type to filter ~200 catalog names by substring, Enter selects.
        self.strategy_combo.setEditable(True)
        self.strategy_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        completer = self.strategy_combo.completer()
        completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
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

        # The global toolbar stays lean: instrument, the literal date window, and
        # the display timeframe. Every other control lives in a left workspace.
        self.mgc_check = QtWidgets.QCheckBox("MGC mirror")
        self.mgc_check.setToolTip(
            "Show the MGC (micro gold) tape for the same day underneath, with the\n"
            "current GC trade's entry/exit and initial levels mirrored onto it."
        )
        self.mgc_check.toggled.connect(self._on_mgc_toggled)
        self._controls = QtWidgets.QHBoxLayout()
        self._controls.setSpacing(6)
        for w in (
            self.instrument_combo,
            QtWidgets.QLabel("Start"),
            self.start_edit,
            QtWidgets.QLabel("End"),
            self.end_edit,
            QtWidgets.QLabel("TF"),
            self.tf_combo,
        ):
            self._controls.addWidget(w)
        self._controls.addStretch(1)

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

        # Trade list + analysis live in the Strategy workspace (left); the right
        # sidebar is the trade inspector.
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self.trade_table, "Replay")
        self._tabs.addTab(self.blotter, "Free-play")
        self._tabs.addTab(self.whatif_table, "What-if")
        self._tabs.addTab(self.heatmap, "Exit grid")
        self.session_panel = SessionPanel()
        self.session_panel.settingsChanged.connect(self._recompute_session)
        self.edge_panel = EdgeContextPanel()

        # Trade inspector: a compact overview on single click; the full detail
        # (below) opens on double-click and Escape returns to the full chart.
        self._inspector_hint = "Click a trade for its overview; double-click for full detail."
        self.overview = QtWidgets.QLabel(self._inspector_hint)
        self.overview.setWordWrap(True)
        self.overview.setTextFormat(QtCore.Qt.RichText)
        self.overview.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.overview.setContentsMargins(10, 8, 10, 8)
        overview_box = QtWidgets.QGroupBox("Trade overview")
        QtWidgets.QVBoxLayout(overview_box).addWidget(self.overview)

        self.forensics = ForensicsPanel()
        self._right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._right.addWidget(overview_box)
        self._right.addWidget(self.forensics)
        self._right.setSizes([150, 520])
        self.trade_table.doubleClicked.connect(lambda _i: self._activate_selected_row())

    def _build_exit_panel(self) -> None:
        self.exit_panel = ExitPanel()
        self.exit_panel.configChanged.connect(self._on_config_changed)

    # -- workspaces --------------------------------------------------------
    _WORKSPACES = ("Strategy", "Indicators", "Drawing", "Risk", "Prop firm")

    def _build_workspaces(self) -> None:
        """Five collapsible left workspaces behind a vertical nav rail."""

        self._workspace_stack = QtWidgets.QStackedWidget()
        for maker in (
            self._make_strategy_ws,
            self._make_indicators_ws,
            self._make_drawing_ws,
            self._make_risk_ws,
            self._make_propfirm_ws,
        ):
            self._workspace_stack.addWidget(maker())

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(300)
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self._workspace_stack)
        self._workspace_panel = panel

        rail = QtWidgets.QToolBar("Workspaces")
        rail.setMovable(False)
        rail.setOrientation(QtCore.Qt.Vertical)
        rail.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._ws_actions = []
        for i, label in enumerate(self._WORKSPACES):
            act = rail.addAction(label)
            act.setCheckable(True)
            act.setShortcut(f"Ctrl+{i + 1}")
            act.triggered.connect(lambda _c=False, idx=i: self._select_workspace(idx))
            self._ws_actions.append(act)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, rail)
        self._nav_rail = rail
        self._select_workspace(0)  # Strategy open by default

    def _select_workspace(self, index: int) -> None:
        """Show a workspace; clicking the open one again collapses the panel."""

        collapse = (
            self._workspace_panel.isVisible() and self._workspace_stack.currentIndex() == index
        )
        for i, act in enumerate(self._ws_actions):
            act.setChecked(i == index and not collapse)
        if collapse:
            self._workspace_panel.setVisible(False)
        else:
            self._workspace_stack.setCurrentIndex(index)
            self._workspace_panel.setVisible(True)

    @staticmethod
    def _ws_widget() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        return w, lay

    def _make_strategy_ws(self) -> QtWidgets.QWidget:
        w, lay = self._ws_widget()
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Strategy"))
        top.addWidget(self.strategy_combo, 1)
        top.addWidget(self.replay_btn)
        lay.addLayout(top)
        lay.addWidget(self.custom_check)
        fp = QtWidgets.QHBoxLayout()
        for x in (self.freeplay_check, self.long_radio, self.short_radio):
            fp.addWidget(x)
        fp.addStretch(1)
        lay.addLayout(fp)
        acts = QtWidgets.QHBoxLayout()
        for x in (self.excursion_check, self.whatif_btn, self.grid_btn):
            acts.addWidget(x)
        acts.addStretch(1)
        lay.addLayout(acts)
        lay.addWidget(self._tabs, 1)
        return w

    def _make_indicators_ws(self) -> QtWidgets.QWidget:
        w, lay = self._ws_widget()
        box = QtWidgets.QGroupBox("Overlays")
        bl = QtWidgets.QVBoxLayout(box)
        for x in (
            self.vwap_check,
            self.vwap_day_check,
            self.vwap_session_check,
            self.session_check,
        ):
            bl.addWidget(x)
        lay.addWidget(box)
        lay.addWidget(self.edge_panel, 1)
        return w

    def _make_drawing_ws(self) -> QtWidgets.QWidget:
        w, lay = self._ws_widget()
        box = QtWidgets.QGroupBox("Drawing tools")
        bl = QtWidgets.QVBoxLayout(box)
        for _label, mode, _tip, _hint in self._DRAW_TOOLS:
            bl.addWidget(self._draw_buttons[mode])
        lay.addWidget(box)
        hint = QtWidgets.QLabel("Undo and Clear are on the chart's top-left. Right-click cancels.")
        hint.setWordWrap(True)
        hint.setProperty("role", "caption")
        lay.addWidget(hint)
        lay.addStretch(1)
        return w

    def _make_risk_ws(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.exit_panel)
        lay.addWidget(scroll)
        return w

    def _make_propfirm_ws(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.session_panel)
        return w

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        save_act = file_menu.addAction("Save session...")
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_session)
        load_act = file_menu.addAction("Load session...")
        load_act.setShortcut("Ctrl+O")
        load_act.triggered.connect(self._load_session)
        self._view_menu = self.menuBar().addMenu("View")
        for label, mode in (("Dark theme", "dark"), ("Light theme", "light")):
            act = self._view_menu.addAction(label)
            act.triggered.connect(lambda _c=False, m=mode: self._set_theme(m))

    def _populate_view_menu(self) -> None:
        """Toggles for the panels the workspace nav does not own."""

        menu = self._view_menu
        menu.addSeparator()
        inspector = menu.addAction("Trade inspector")
        inspector.setCheckable(True)
        inspector.setChecked(True)
        inspector.setShortcut("Ctrl+I")
        inspector.toggled.connect(self._right.setVisible)
        self._inspector_action = inspector
        transport_act = self._transport.toggleViewAction()
        transport_act.setText("Replay transport")
        transport_act.setShortcut("Ctrl+R")
        menu.addAction(transport_act)
        mgc_act = menu.addAction("MGC mirror pane")
        mgc_act.setCheckable(True)
        mgc_act.setShortcut("Ctrl+M")
        mgc_act.toggled.connect(self.mgc_check.setChecked)
        self.mgc_check.toggled.connect(mgc_act.setChecked)

    def _build_transport(self) -> None:
        bar = QtWidgets.QToolBar("Replay")
        bar.setMovable(False)
        self.replay_day_btn = QtWidgets.QPushButton("Replay view")
        self.replay_day_btn.setToolTip(
            "Hide the visible range and play its tape from the first bar - candles\n"
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

    _DRAW_TOOLS = (
        ("Cursor", None, "Normal interaction: crosshair, free-play clicks, drags.", ""),
        (
            "Level",
            "hline",
            "A horizontal price level (support/resistance). Drag it later to move it.",
            "Level armed: click a price on the chart. Right-click cancels.",
        ),
        (
            "Trend",
            "trend",
            "A trendline. Drag across the chart (or click two points); drag the\n"
            "endpoint handles later to adjust.",
            "Trendline armed: drag across the chart, or click two points. Right-click cancels.",
        ),
        (
            "Zone",
            "rect",
            "A shaded box for supply/demand or consolidation zones. Drag a box;\n"
            "drag its corner handles later to resize.",
            "Zone armed: drag a box on the chart. Right-click cancels.",
        ),
        (
            "Time",
            "vline",
            "A vertical time marker. Drag it later to move it.",
            "Time marker armed: click a bar on the chart. Right-click cancels.",
        ),
    )

    def _build_draw_actions(self) -> None:
        """Drawing-tool buttons for the Drawing Tools workspace.

        Arming a tool checks its button and puts a plain-language hint in the
        status bar; every tool is one-shot and hands back to the cursor. Undo and
        Clear are compact icons directly on the chart.
        """

        self._draw_buttons: dict = {}
        self._draw_hints: dict = {}
        self._draw_group = QtWidgets.QButtonGroup(self)
        self._draw_group.setExclusive(True)
        for label, mode, tip, hint in self._DRAW_TOOLS:
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _c=False, m=mode: self._arm_draw_tool(m))
            self._draw_group.addButton(btn)
            self._draw_buttons[mode] = btn
            self._draw_hints[mode] = hint
        self._draw_buttons[None].setChecked(True)
        self.chart.drawing_placed.connect(self._on_drawing_placed)
        self._add_chart_draw_icons()

    def _add_chart_draw_icons(self) -> None:
        """Compact Undo/Clear buttons floating over the chart's top-left."""

        bar = QtWidgets.QWidget(self.chart)
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        for text, tip, slot in (
            ("Undo", "Remove the last drawing on this day (Ctrl+Z).", self.chart.undo_drawing),
            ("Clear", "Remove every drawing on this day.", self.chart.clear_drawings),
        ):
            btn = QtWidgets.QToolButton()
            btn.setText(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            lay.addWidget(btn)
        bar.move(58, 6)  # clear of the left price axis
        bar.raise_()
        self._chart_draw_icons = bar
        undo_sc = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        undo_sc.activated.connect(self.chart.undo_drawing)

    def _arm_draw_tool(self, mode: str | None) -> None:
        self.chart.set_draw_mode(mode)
        self._set_status(self._draw_hints.get(mode, ""))

    def _on_drawing_placed(self) -> None:
        self.chart.set_draw_mode(None)
        self._draw_buttons[None].setChecked(True)
        self._set_status("Drawing placed. Drag it to adjust; Undo/Clear are on the chart.")

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
        self._set_date_bounds(data["dates"])
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
        self.strategy_combo.completer().popup().setMinimumWidth(widest + 48)
        self.strategy_combo.setCurrentIndex(0)  # editable combo starts blank otherwise
        self.instrument_combo.blockSignals(True)
        self.instrument_combo.clear()
        for instrument in available_instruments():
            self.instrument_combo.addItem(instrument.display, instrument.symbol)
        gc_index = self.instrument_combo.findData("GC")
        self.instrument_combo.setCurrentIndex(max(0, gc_index))
        self.instrument_combo.blockSignals(False)
        self.session_panel.sync_instruments(available_instruments(), self._instrument)
        self._set_status(f"Loaded {data['replay'].bars.n_bars:,} GC bars (Dev+Val).")
        self._render_view()
        self.loadFinished.emit(True)

    # -- rendering ---------------------------------------------------------
    @property
    def _bars(self):
        if self._instrument_data is not None:
            return self._instrument_data["bars"]
        return self._data["replay"].bars

    def _active_data(self) -> dict:
        """Overlay/date arrays for the active instrument (GC = research bundle)."""

        return self._instrument_data if self._instrument_data is not None else self._data

    @staticmethod
    def _qdate(ts) -> QtCore.QDate:
        t = pd.Timestamp(ts)
        return QtCore.QDate(t.year, t.month, t.day)

    def _set_date_bounds(self, dates) -> None:
        """Constrain both date edits to the loaded window and default to the last day."""

        first, last = self._qdate(dates[0]), self._qdate(dates[-1])
        self._date_guard = True
        for edit in (self.start_edit, self.end_edit):
            edit.setDateRange(first, last)
        self.start_edit.setDate(last)  # default: the most recent single day
        self.end_edit.setDate(last)
        self._date_guard = False

    def _window_bounds(self, bars, start_d, end_d) -> tuple[int, int]:
        """First and last 1m bar index covering the inclusive [start, end] dates."""

        start = np.datetime64(pd.Timestamp(start_d))
        end = np.datetime64(pd.Timestamp(end_d))
        lo = int(np.searchsorted(bars.trade_date, start, side="left"))
        hi = int(np.searchsorted(bars.trade_date, end, side="right")) - 1
        return lo, hi

    def _on_dates_changed(self, _d=None) -> None:
        if self._date_guard or self._data is None:
            return
        if self.start_edit.date() > self.end_edit.date():
            # Keep the pair ordered by snapping the edit the user did not touch.
            self._date_guard = True
            if self.sender() is self.start_edit:
                self.end_edit.setDate(self.start_edit.date())
            else:
                self.start_edit.setDate(self.end_edit.date())
            self._date_guard = False
        self._render_view()

    def _render_view(self) -> None:
        """Render the literal [Start, End] date window at the display timeframe."""

        if self._data is None:
            return
        self._animator.stop()  # leaving the window ends any running animation
        bars = self._bars
        lo, hi = self._window_bounds(
            bars, self.start_edit.date().toPython(), self.end_edit.date().toPython()
        )
        if hi < lo:
            self._set_status("No bars in the selected date range.")
            return
        self._view_start, self._view_end = lo, hi
        multi_day = bars.trade_date[lo] != bars.trade_date[hi]
        tf = TIMEFRAMES.get(self.tf_combo.currentText(), 1)
        rs = resample_window(bars, lo, hi, tf)
        view_map = ViewMap.from_resampled(rs, tf)
        labels = bucket_labels(bars, rs, tf, multi_day=multi_day)
        ohlc = {
            "open": rs.open,
            "high": rs.high,
            "low": rs.low,
            "close": rs.close,
            "volume": rs.volume,
            "segment": rs.segment,
        }
        self.chart.set_tick_size(REGISTRY[self._instrument].spec.tick_size)
        self.chart.set_view(ohlc, labels, view_map, minute_close=bars.close[lo : hi + 1])
        # Always build the overlays; the checkboxes only flip visibility. Lines
        # are display-sampled at each bucket's close, shading at its open.
        self.chart.add_vwap(
            sample_last(self._active_data()["vwap20"], rs),
            "rolling",
            visible=self.vwap_check.isChecked(),
        )
        self.chart.add_vwap(
            sample_last(self._active_data()["vwap_day"], rs),
            "day",
            visible=self.vwap_day_check.isChecked(),
        )
        self.chart.add_vwap(
            sample_last(self._active_data()["vwap_session"], rs),
            "session",
            visible=self.vwap_session_check.isChecked(),
        )
        self.chart.shade_sessions(
            sample_first(self._active_data()["session_code"], rs),
            visible=self.session_check.isChecked(),
        )
        self._redraw_overlays()
        self._refresh_edge_context()

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
        self._update_mgc_pane()

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
            result.entry_position,
            result.exit_position,
            result.entry_price,
            result.stop_track,
            result.target_track,
            result.exit_price,
            color=color,
        )

    def _draw_light(self, t) -> None:
        r = t.result
        self.chart.light_marker(
            r.entry_position,
            r.entry_price,
            r.exit_position,
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
            entries,
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
        """Widen the date window to include the bar; True if that re-rendered."""

        if self._in_view(position):
            return False
        qd = self._qdate(self._bars.trade_date[position])
        self._date_guard = True
        if qd < self.start_edit.date():
            self.start_edit.setDate(qd)
        elif qd > self.end_edit.date():
            self.end_edit.setDate(qd)
        self._date_guard = False
        self._render_view()
        return True

    # -- replay ------------------------------------------------------------
    def _on_replay(self) -> None:
        if self._data is None or self._instrument != "GC":
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
        self._render_view()

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

    def _selected_row(self):
        """The trade-log Series for the selected replay row, or None."""

        rows = self.trade_table.selectionModel().selectedRows()
        if not rows or self._log is None:
            return None
        item = self.trade_table.item(rows[0].row(), 0)
        if item is None:  # selection event racing a table refresh
            return None
        match = self._log[self._log["entry_position"] == item.data(QtCore.Qt.UserRole)]
        return None if match.empty else match.iloc[0]

    def _on_row_selected(self) -> None:
        """Single click: a compact overview only - the chart is not consumed."""

        trade = self._selected_row()
        if trade is not None:
            self.overview.setText(self._overview_html(trade))

    def _activate_selected_row(self) -> None:
        """Double click: the full inspector (chart focus + forensics detail)."""

        trade = self._selected_row()
        if trade is not None:
            self._focus_trade(trade)

    def _overview_html(self, trade) -> str:
        p = theme.active()
        side = "Long" if trade["direction"] > 0 else "Short"
        r = float(trade["gross_r"])
        colour = p.up if r > 0 else p.down
        return (
            f"<b style='color:{colour}'>{side} &middot; {trade['exit_reason']} "
            f"&middot; {r:+.2f} R</b><br>"
            f"Entry {self._ny_label(int(trade['entry_position']))} &middot; "
            f"held {int(trade['holding_minutes'])} min<br>"
            f"In favour +{float(trade['mfe_r']):.2f} R &middot; "
            f"against -{float(trade['mae_r']):.2f} R<br>"
            f"<span style='color:{p.text_faint}'>Double-click for the full inspector.</span>"
        )

    def _focus_trade(self, trade) -> None:
        self.overview.setText(self._overview_html(trade))
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
        self.chart.center_on(result.entry_position)
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
            self.chart.center_on(active.entry_position)
        self.blotter.set_trades(list(self._placed.values()))
        self.blotter.select_trade(active.id)
        self._recompute_session()
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
            result.entry_position,
            result.exit_position,
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
        self._recompute_session()
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
        self._recompute_session()
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
        self._recompute_session()
        self._redraw_overlays()

    def _on_placed_cleared(self) -> None:
        self._placed.clear()
        self._active_id = None
        self.blotter.set_trades([])
        self._drop_stale_analysis()
        self._recompute_session()
        self._redraw_overlays()

    # -- session persistence -----------------------------------------------
    def _collect_session_payload(self) -> dict:
        sp = self.session_panel
        placed = [
            {
                "id": int(trade.id),
                "entry_position": int(trade.entry_position),
                "direction": int(trade.direction),
                "color": trade.color,
                "cfg": exit_config_to_dict(trade.cfg),
            }
            for trade in self._placed.values()
        ]
        return build_payload(
            theme_mode=theme.active().name,
            view={
                "instrument": self._instrument,
                "start_date": self.start_edit.date().toString("yyyy-MM-dd"),
                "end_date": self.end_edit.date().toString("yyyy-MM-dd"),
                "timeframe": self.tf_combo.currentText(),
            },
            account={
                "starting_balance": float(sp.balance_spin.value()),
                "sizing_index": sp.sizing_combo.currentIndex(),
                "risk_percent": float(sp.risk_spin.value()),
                "fixed_contracts": int(sp.contracts_spin.value()),
                "instrument_index": sp.instrument_combo.currentIndex(),
            },
            evaluation_preset=sp.preset_combo.currentText(),
            placed=placed,
            active_id=self._active_id,
            next_id=self._next_id,
            drawings=self.chart.export_drawings(),
        )

    def _save_session(self) -> None:
        if self._data is None:
            return
        sessions_dir = default_sessions_dir(project_root())
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session", str(sessions_dir / "session.json"), "Session (*.json)"
        )
        if not path:
            return
        write_session(path, self._collect_session_payload())
        self._set_status(f"Session saved to {path}")

    def _load_session(self) -> None:
        if self._data is None:
            return
        sessions_dir = default_sessions_dir(project_root())
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load session", str(sessions_dir), "Session (*.json)"
        )
        if not path:
            return
        try:
            payload = read_session(path)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            QtWidgets.QMessageBox.warning(self, "Load session", str(error))
            return
        self._apply_session_payload(payload)
        self._set_status(f"Session loaded from {path} (trades re-simulated).")

    def _apply_session_payload(self, payload: dict) -> None:
        wanted = payload.get("view", {}).get("instrument", "GC")
        if wanted != self._instrument:
            index = self.instrument_combo.findData(wanted)
            if index < 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Load session",
                    f"This session was saved on {wanted}, which has no local data.",
                )
                return
            self._pending_session = payload  # applied after the switch completes
            self.instrument_combo.setCurrentIndex(index)
            return
        sp = self.session_panel
        account = payload.get("account", {})
        for widget in (
            sp.balance_spin,
            sp.sizing_combo,
            sp.risk_spin,
            sp.contracts_spin,
            sp.instrument_combo,
            sp.preset_combo,
        ):
            widget.blockSignals(True)
        sp.balance_spin.setValue(float(account.get("starting_balance", 100_000.0)))
        sp.sizing_combo.setCurrentIndex(int(account.get("sizing_index", 0)))
        sp.risk_spin.setValue(float(account.get("risk_percent", 1.0)))
        sp.contracts_spin.setValue(int(account.get("fixed_contracts", 1)))
        sp.instrument_combo.setCurrentIndex(int(account.get("instrument_index", 0)))
        preset = payload.get("evaluation_preset", "Practice - no rules")
        if sp.preset_combo.findText(preset) >= 0:
            sp.preset_combo.setCurrentText(preset)
        for widget in (
            sp.balance_spin,
            sp.sizing_combo,
            sp.risk_spin,
            sp.contracts_spin,
            sp.instrument_combo,
            sp.preset_combo,
        ):
            widget.blockSignals(False)

        # Re-simulate every trade through the verified engine; results are
        # recomputed, never trusted from disk.
        bars = self._bars
        self._placed.clear()
        skipped = 0
        for record in payload.get("placed", []):
            entry = int(record.get("entry_position", -1))
            if not (0 <= entry < bars.n_bars):
                skipped += 1
                continue
            cfg = exit_config_from_dict(record.get("cfg", {}))
            trade = place_trade(
                int(record["id"]),
                entry,
                int(record.get("direction", 1)),
                cfg,
                bars.atr20,
                bars.bar_arrays(),
                color=record.get("color", ""),
            )
            self._placed[trade.id] = trade
        self._active_id = payload.get("active_id")
        if self._active_id is not None and self._active_id not in self._placed:
            self._active_id = next(iter(self._placed), None)
        self._next_id = max(
            int(payload.get("next_id", 1)),
            max(self._placed, default=0) + 1,
        )
        self._focused_result = None
        self._focused_obs_id = None
        self._whatif_runs = []

        view = payload.get("view", {})
        self.tf_combo.blockSignals(True)
        tf = view.get("timeframe", "1m")
        if self.tf_combo.findText(tf) >= 0:
            self.tf_combo.setCurrentText(tf)
        self.tf_combo.blockSignals(False)
        self._set_theme(payload.get("theme", theme.active().name))
        # Restore the literal date window (clamped to the instrument's data range).
        self._date_guard = True
        for edit, key in ((self.start_edit, "start_date"), (self.end_edit, "end_date")):
            saved = QtCore.QDate.fromString(view.get(key, ""), "yyyy-MM-dd")
            if saved.isValid():
                edit.setDate(saved)
        self._date_guard = False
        self._render_view()
        self.chart.import_drawings(payload.get("drawings", []))
        self.blotter.set_trades(list(self._placed.values()))
        if self._active_id is not None:
            self.blotter.select_trade(self._active_id)
        self._recompute_session()
        self._redraw_overlays()
        if skipped:
            self._set_status(f"Session loaded; {skipped} trade(s) outside the loaded data.")

    # -- instrument switching ----------------------------------------------
    def _on_instrument_changed(self, index: int) -> None:
        symbol = self.instrument_combo.itemData(index)
        if symbol is None or symbol == self._instrument or self._data is None:
            return
        if self._instrument_loading:
            return
        self._instrument_loading = True
        self._set_status(f"Loading {symbol} bars ...")
        if symbol == "GC":
            self._install_instrument("GC", None)
            self._instrument_loading = False
            return
        task = Task(load_instrument_bundle, symbol, self._data.get("date_floor"))
        task.signals.finished.connect(self._on_instrument_loaded)
        task.signals.error.connect(self._on_instrument_error)
        self._start(task)

    def _on_instrument_loaded(self, bundle: dict) -> None:
        self._instrument_loading = False
        self._install_instrument(bundle["symbol"], bundle)

    def _on_instrument_error(self, message: str) -> None:
        self._instrument_loading = False
        # Revert the combo to the active instrument and surface the reason.
        idx = self.instrument_combo.findData(self._instrument)
        self.instrument_combo.blockSignals(True)
        self.instrument_combo.setCurrentIndex(idx)
        self.instrument_combo.blockSignals(False)
        self._on_error(message)

    def _install_instrument(self, symbol: str, bundle: dict | None) -> None:
        """Make ``symbol`` the primary instrument and reset per-instrument state.

        Placed trades index into the previous instrument's bar store, so the
        blotter clears (the status line says so); the strategy catalog, replay
        markers, MGC mirror, and Edge context stay GC-only with the reason on
        screen rather than silently computing on unvalidated data.
        """

        self._animator.stop()
        self._instrument = symbol
        self._instrument_data = bundle
        self._placed.clear()
        self._active_id = None
        self._focused_result = None
        self._focused_obs_id = None
        self._whatif_runs = []
        self._log = None
        self.trade_table.setRowCount(0)
        self.whatif_table.setRowCount(0)
        self.blotter.set_trades([])
        self.forensics.clear()
        self._edge_day = None
        self._ny_label = ny_time_label(self._bars)
        self.blotter.set_time_label(self._ny_label)
        instrument = REGISTRY[symbol]
        gc_active = symbol == "GC"
        for widget in (self.strategy_combo, self.replay_btn, self.custom_check):
            widget.setEnabled(gc_active)
        if not gc_active:
            self.replay_btn.setToolTip(
                "Strategy catalog and replay are GC-only (research validity)."
            )
        self.mgc_check.setVisible(gc_active)
        if not gc_active:
            self.mgc_check.setChecked(False)
            self.mgc_chart.setVisible(False)
        self.session_panel.sync_instruments(available_instruments(), symbol)
        self._set_date_bounds(self._active_data()["dates"])
        self.setWindowTitle(f"GQ Trade Simulator - {instrument.display} - Development + Validation")
        self._render_view()
        self._recompute_session()
        note = "" if gc_active else " Blotter cleared; replay and edge context are GC-only."
        self._set_status(f"{instrument.display}: {self._bars.n_bars:,} bars loaded.{note}")
        if self._pending_session is not None:
            payload, self._pending_session = self._pending_session, None
            self._apply_session_payload(payload)

    # -- edge context ------------------------------------------------------
    def _refresh_edge_context(self) -> None:
        """Compute the last visible day's validated features on the worker pool."""

        if self._data is None:
            return
        if self._instrument != "GC":
            self.edge_panel.clear_context(
                "Edge context is GC-only research evidence; the validated slate "
                "was established on GC and does not transfer."
            )
            self._edge_day = None
            return
        bars = self._bars
        day_str = self.end_edit.date().toString("yyyy-MM-dd")
        anchor = np.datetime64(pd.Timestamp(day_str))
        day_lo = int(np.searchsorted(bars.trade_date, anchor, side="left"))
        day_hi = int(np.searchsorted(bars.trade_date, anchor, side="right")) - 1
        if day_hi < day_lo:
            return
        if self._edge_day == (day_lo, day_hi):
            return  # this day is already loaded or loading
        self._edge_day = (day_lo, day_hi)
        self._edge_token += 1
        token = self._edge_token
        self.edge_panel.clear_context("Computing the day's context ...")
        minute_slice = bars.minute_ny[day_lo : day_hi + 1].copy()
        task = Task(self._edge_service.day_context, day_str, minute_slice)
        task.signals.finished.connect(
            lambda context, tok=token: self._on_edge_context(context, tok)
        )
        task.signals.error.connect(
            lambda message, tok=token: self._on_edge_context_error(message, tok)
        )
        self._start(task)

    def _on_edge_context(self, context, token: int) -> None:
        if token != self._edge_token:
            return
        self.edge_panel.set_context(context)

    def _on_edge_context_error(self, message: str, token: int) -> None:
        if token != self._edge_token:
            return
        first_line = message.strip().splitlines()[-1] if message.strip() else "unknown error"
        self.edge_panel.clear_context(f"Edge context unavailable: {first_line}")

    def _on_bar_hovered(self, global_index: int) -> None:
        if self._edge_day is None:
            return
        day_lo, day_hi = self._edge_day
        if day_lo <= global_index <= day_hi:
            self.edge_panel.show_bar(global_index - day_lo)

    # -- session accounting ------------------------------------------------
    def _recompute_session(self) -> None:
        """Rebuild the dollar equity curve, stats, and evaluation verdict."""

        if self._data is None:
            return
        settings = self.session_panel.account_settings()
        curve = build_equity_curve(list(self._placed.values()), settings, self._bars.trade_date)
        order = {int(tid): i for i, tid in enumerate(curve.trade_ids)}
        r_list = [0.0] * curve.n_trades
        dollars: dict[int, float] = {}
        for trade in self._placed.values():
            i = order.get(int(trade.id))
            if i is not None:
                r_list[i] = float(trade.result.gross_r)
                dollars[int(trade.id)] = float(curve.pnl[i])
        stats = session_stats(curve, r_list, settings)
        rules = self.session_panel.evaluation_rules()
        status = evaluate(curve, rules, settings.starting_balance) if rules else None
        self.session_panel.update_session(curve, stats, status)
        self.blotter.set_dollars(dollars)

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

    # -- MGC mirror pane ----------------------------------------------------
    def _on_mgc_toggled(self, on: bool) -> None:
        if not on:
            self.mgc_chart.setVisible(False)
            return
        if self._mgc_bars is None:
            if self._mgc_loading:
                return
            self._mgc_loading = True
            self._set_status("Loading MGC bars ...")
            task = Task(BarStore.load, research_bars_path(), product="MGC")
            task.signals.finished.connect(self._on_mgc_loaded)
            task.signals.error.connect(self._on_mgc_error)
            self._start(task)
            return
        self.mgc_chart.setVisible(True)
        self._update_mgc_pane(force_render=True)

    def _on_mgc_loaded(self, store) -> None:
        self._mgc_loading = False
        self._mgc_bars = store
        self._set_status(f"MGC: {store.n_bars:,} bars loaded (Dev+Val).")
        if self.mgc_check.isChecked():
            self.mgc_chart.setVisible(True)
            self._update_mgc_pane(force_render=True)

    def _on_mgc_error(self, message: str) -> None:
        self._mgc_loading = False
        self.mgc_check.setChecked(False)
        self._on_error(message)

    def _update_mgc_pane(self, force_render: bool = False) -> None:
        """Render the MGC day window and ghost the current GC trade onto it."""

        if (
            self._instrument != "GC"
            or self._mgc_bars is None
            or not self.mgc_chart.isVisible()
            or self._data is None
        ):
            return
        mgc = self._mgc_bars
        lo, hi = self._window_bounds(
            mgc, self.start_edit.date().toPython(), self.end_edit.date().toPython()
        )
        if hi < lo:
            self.mgc_chart.clear_trades()
            self._set_status("No MGC bars in this range.")
            return
        multi_day = mgc.trade_date[lo] != mgc.trade_date[hi]
        tf = TIMEFRAMES.get(self.tf_combo.currentText(), 1)
        if force_render or self._mgc_range != (lo, hi, tf):
            self._mgc_range = (lo, hi, tf)
            rs = resample_window(mgc, lo, hi, tf)
            labels = bucket_labels(mgc, rs, tf, multi_day=multi_day)
            self.mgc_chart.set_tick_size(REGISTRY["MGC"].spec.tick_size)
            self.mgc_chart.set_view(
                {
                    "open": rs.open,
                    "high": rs.high,
                    "low": rs.low,
                    "close": rs.close,
                    "volume": rs.volume,
                    "segment": rs.segment,
                },
                labels,
                ViewMap.from_resampled(rs, tf),
                minute_close=mgc.close[lo : hi + 1],
            )
            self.mgc_chart.shade_sessions(
                sample_first(_session_code(mgc.minute_ny), rs),
                visible=self.session_check.isChecked(),
            )
        self.mgc_chart.clear_trades()
        result, _cfg = self._current()
        if result is None or not self._in_view(result.entry_position):
            return
        # Map the GC trade onto the MGC clock by timestamp (minute-of-day is
        # not monotonic within an NY trade date; timestamps are). Search the
        # raw ts arrays directly - both stores read the same parquet column,
        # so the values compare without any tz-dropping conversion.
        gc = self._bars
        entry_pos = int(np.searchsorted(mgc.ts, gc.ts[int(result.entry_position)]))
        exit_pos = int(
            np.searchsorted(mgc.ts, gc.ts[min(int(result.exit_position), gc.n_bars - 1)])
        )
        entry_pos = max(0, min(entry_pos, mgc.n_bars - 1))
        exit_pos = max(0, min(exit_pos, mgc.n_bars - 1))
        if not (lo <= entry_pos <= hi):
            return
        exit_pos = max(lo, min(exit_pos, hi))
        stop0 = float(result.stop_track[0]) if len(result.stop_track) else None
        target0 = float(result.target_track[0]) if len(result.target_track) else None
        self.mgc_chart.mirror_trade(
            entry_pos,
            exit_pos,
            float(result.entry_price),
            stop0,
            target0,
            float(result.exit_price),
        )
        self.mgc_chart.center_on(entry_pos)

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
        self.chart.center_on(result.entry_position)
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
        self._render_view()
        # Re-render everything that baked the previous palette into itself.
        self.heatmap.refresh_theme()
        self.forensics.retheme()
        self.session_panel.retheme()
        self.edge_panel.retheme()
        self._recompute_session()
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
                self.chart.center_on(result.entry_position)

    # -- lifecycle ---------------------------------------------------------
    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() == QtCore.Qt.Key_Escape:
            if self.chart._draw_mode is not None:  # cancel an armed drawing tool
                self._arm_draw_tool(None)
                self._draw_buttons[None].setChecked(True)
                return
            if self._focused_result is not None:  # return to the full chart
                self._focused_result = None
                self._focused_obs_id = None
                self.forensics.clear()
                self.overview.setText(self._inspector_hint)
                self._render_view()  # un-zoom; date, strategy, and replay stay
                return
        super().keyPressEvent(event)

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
