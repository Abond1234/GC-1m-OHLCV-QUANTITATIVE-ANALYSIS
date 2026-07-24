"""Main window: chart + strategy replay + free-play + forensics.

Loads the Dev/Val universe and bars on a worker thread, renders one day at a time,
overlays VWAP and the New York session, and supports two modes:

* Replay - pick any catalog/peer strategy, plot its trades (coloured by outcome),
  click a trade to redraw its exact path and read why it worked or failed.
* Free-play - click the chart to place an entry (filled next bar), size the exit
  freely (ATR stop, R target, trailing, breakeven), and watch the outcome path.

All fills come from the verified flexible-exit engine, so what you see on screen is
what the research engine would record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..datalayer.catalog_service import StrategyReplayService
from ..datalayer.forensics import ForensicsService
from ..datalayer.paths import project_root
from ..datalayer.vwap import execution_session_vwap, research_day_vwap, rolling_vwap
from ..sim.exit_config import ExitConfig, frozen_config
from ..sim.flex_exit import single_flex_exit
from ..workers.tasks import Task
from .chart_widget import ChartWidget

_TRADE_COLUMNS = ["Strategy", "Entry (NY)", "Dir", "Exit", "R", "Held"]


def _session_code(minute_ny: np.ndarray) -> np.ndarray:
    code = np.zeros(len(minute_ny), dtype=np.int8)
    code[(minute_ny >= 180) & (minute_ny < 360)] = 1
    code[(minute_ny >= 420) & (minute_ny < 720)] = 2
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
    def __init__(self, data: dict | None = None):
        super().__init__()
        self.setWindowTitle("GC Trade Simulator - Development + Validation")
        self.resize(1500, 900)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._data: dict | None = None
        self._log: pd.DataFrame | None = None
        self._view_start = 0
        self._view_end = 0
        self._replay_cfg = frozen_config()

        self._build_ui()
        if data is not None:
            # Synchronous path for the offscreen render harness and tests.
            self._on_loaded(data)
            return
        self._set_status("Loading Development + Validation data ...")
        task = Task(load_app_data)
        task.signals.finished.connect(self._on_loaded)
        task.signals.error.connect(self._on_error)
        self._pool.start(task)

    # -- construction ------------------------------------------------------
    def _build_ui(self) -> None:
        self.chart = ChartWidget()
        self.chart.bar_clicked.connect(self._on_bar_clicked)

        self.date_combo = QtWidgets.QComboBox()
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
        for box in (
            self.vwap_check,
            self.vwap_day_check,
            self.vwap_session_check,
            self.session_check,
        ):
            box.stateChanged.connect(lambda _s: self._render_current_date())

        self.strategy_combo = QtWidgets.QComboBox()
        self.custom_check = QtWidgets.QCheckBox("Custom exits")
        self.replay_btn = QtWidgets.QPushButton("Replay")
        self.replay_btn.clicked.connect(self._on_replay)
        self.freeplay_check = QtWidgets.QCheckBox("Free-play (click to enter)")
        self.long_radio = QtWidgets.QRadioButton("Long")
        self.long_radio.setChecked(True)
        self.short_radio = QtWidgets.QRadioButton("Short")

        self.stop_spin = self._spin(0.1, 10.0, 1.5, "stop ATR x")
        self.target_spin = self._spin(0.1, 20.0, 2.0, "target R")
        self.trail_check = QtWidgets.QCheckBox("trail")
        self.trail_spin = self._spin(0.1, 10.0, 1.5, "trail ATR x")
        self.be_check = QtWidgets.QCheckBox("breakeven @R")
        self.be_spin = self._spin(0.1, 5.0, 1.0, "BE trigger R")

        controls = QtWidgets.QHBoxLayout()
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
            _sep(),
            self.freeplay_check,
            self.long_radio,
            self.short_radio,
            QtWidgets.QLabel("stop"),
            self.stop_spin,
            QtWidgets.QLabel("target"),
            self.target_spin,
            self.trail_check,
            self.trail_spin,
            self.be_check,
            self.be_spin,
        ):
            controls.addWidget(w)
        controls.addStretch(1)

        self.trade_table = QtWidgets.QTableWidget(0, len(_TRADE_COLUMNS))
        self.trade_table.setHorizontalHeaderLabels(_TRADE_COLUMNS)
        self.trade_table.horizontalHeader().setStretchLastSection(True)
        self.trade_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.trade_table.itemSelectionChanged.connect(self._on_row_selected)
        self.forensics = QtWidgets.QTextEdit()
        self.forensics.setReadOnly(True)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(self.trade_table)
        right.addWidget(self.forensics)
        right.setSizes([500, 400])
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.chart)
        split.addWidget(right)
        split.setSizes([1050, 450])

        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.addLayout(controls)
        outer.addWidget(split, 1)
        self.setCentralWidget(central)
        self._set_status = self.statusBar().showMessage

    @staticmethod
    def _spin(lo, hi, val, tip):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(0.1)
        spin.setValue(val)
        spin.setToolTip(tip)
        spin.setFixedWidth(70)
        return spin

    # -- loading -----------------------------------------------------------
    def _on_error(self, message: str) -> None:
        self._set_status("Error - see dialog")
        QtWidgets.QMessageBox.critical(self, "Load error", message)

    def _on_loaded(self, data: dict) -> None:
        self._data = data
        self.date_combo.blockSignals(True)
        self.date_combo.addItems([pd.Timestamp(d).strftime("%Y-%m-%d") for d in data["dates"]])
        self.date_combo.blockSignals(False)
        for spec in data["replay"].list_strategies():
            self.strategy_combo.addItem(spec.name)
        self._set_status(f"Loaded {data['replay'].bars.n_bars:,} GC bars (Dev+Val).")
        self._render_current_date()

    # -- rendering ---------------------------------------------------------
    def _render_current_date(self) -> None:
        if self._data is None or not self.date_combo.count():
            return
        bars = self._data["replay"].bars
        date = np.datetime64(pd.Timestamp(self.date_combo.currentText()))
        idx = np.nonzero(bars.trade_date == date)[0]
        if len(idx) == 0:
            return
        lo, hi = int(idx[0]), int(idx[-1])
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
        if self.vwap_check.isChecked():
            self.chart.add_vwap(self._data["vwap20"][sl], "rolling")
        if self.vwap_day_check.isChecked():
            self.chart.add_vwap(self._data["vwap_day"][sl], "day")
        if self.vwap_session_check.isChecked():
            self.chart.add_vwap(self._data["vwap_session"][sl], "session")
        if self.session_check.isChecked():
            self.chart.shade_sessions(self._data["session_code"][sl])
        self._draw_markers_for_view()

    def _draw_markers_for_view(self) -> None:
        if self._log is None or self._log.empty:
            return
        bars = self._data["replay"].bars
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

    # -- replay ------------------------------------------------------------
    def _exit_config(self) -> ExitConfig:
        return ExitConfig(
            stop_mode="atr",
            stop_value=self.stop_spin.value(),
            target_mode="r",
            target_value=self.target_spin.value(),
            trailing_enabled=self.trail_check.isChecked(),
            trailing_mode="atr",
            trailing_value=self.trail_spin.value(),
            breakeven_enabled=self.be_check.isChecked(),
            breakeven_trigger_r=self.be_spin.value(),
        )

    def _on_replay(self) -> None:
        if self._data is None:
            return
        name = self.strategy_combo.currentText()
        specs = {s.name: s for s in self._data["replay"].list_strategies()}
        if name not in specs:
            return
        self._replay_cfg = self._exit_config() if self.custom_check.isChecked() else frozen_config()
        cfg = self._replay_cfg if self.custom_check.isChecked() else None
        self._set_status(f"Replaying {name} ...")
        task = Task(self._data["replay"].replay, specs[name], cfg)
        task.signals.finished.connect(self._on_replayed)
        task.signals.error.connect(self._on_error)
        self._pool.start(task)

    def _on_replayed(self, log: pd.DataFrame) -> None:
        self._log = log
        self.trade_table.setRowCount(0)
        wins = int((log["gross_r"] > 0).sum()) if not log.empty else 0
        self._set_status(
            f"{len(log):,} trades, {wins:,} winners, mean R {log['gross_r'].mean():.3f}"
            if not log.empty
            else "No trades fired."
        )
        for _, row in log.head(500).iterrows():
            self._append_trade_row(row)
        self._render_current_date()

    def _append_trade_row(self, row) -> None:
        r = self.trade_table.rowCount()
        self.trade_table.insertRow(r)
        values = [
            str(row.get("strategy", "")),
            pd.Timestamp(self._data["replay"].bars.ts[int(row["entry_position"])]).strftime(
                "%m-%d %H:%M"
            ),
            "L" if row["direction"] > 0 else "S",
            str(row["exit_reason"]),
            f"{row['gross_r']:+.2f}",
            str(int(row["holding_minutes"])),
        ]
        for c, v in enumerate(values):
            item = QtWidgets.QTableWidgetItem(v)
            item.setData(QtCore.Qt.UserRole, int(row["entry_position"]))
            self.trade_table.setItem(r, c, item)

    def _on_row_selected(self) -> None:
        rows = self.trade_table.selectionModel().selectedRows()
        if not rows or self._log is None:
            return
        entry_position = self.trade_table.item(rows[0].row(), 0).data(QtCore.Qt.UserRole)
        match = self._log[self._log["entry_position"] == entry_position]
        if match.empty:
            return
        trade = match.iloc[0]
        self._focus_trade(trade)

    def _focus_trade(self, trade) -> None:
        bars = self._data["replay"].bars
        date = pd.Timestamp(trade["trade_date_ny"]).strftime("%Y-%m-%d")
        if self.date_combo.currentText() != date:
            self.date_combo.setCurrentText(date)  # triggers re-render
        result = single_flex_exit(
            int(trade["entry_position"]),
            int(trade["direction"]),
            float(trade["initial_stop_points"]),
            float(trade["initial_target_points"]),
            self._replay_cfg,
            bars.bar_arrays(),
            float(trade.get("trail_points", 0.0)),
        )
        self._draw_and_explain(result, observation_id=trade.get("observation_id"))

    # -- free-play ---------------------------------------------------------
    def _on_bar_clicked(self, global_index: int) -> None:
        if self._data is None or not self.freeplay_check.isChecked():
            return
        bars = self._data["replay"].bars
        entry = global_index + 1  # fill on the next bar's open
        if entry >= bars.n_bars:
            return
        direction = 1 if self.long_radio.isChecked() else -1
        cfg = self._exit_config()
        atr = float(bars.atr20[max(entry - 1, 0)])
        stop_pts = cfg.stop_value * atr
        target_pts = cfg.target_value * stop_pts
        trail_pts = cfg.trailing_value * atr if cfg.trailing_enabled else 0.0
        result = single_flex_exit(
            entry, direction, stop_pts, target_pts, cfg, bars.bar_arrays(), trail_pts
        )
        self._draw_and_explain(result, observation_id=None)

    # -- draw + forensics --------------------------------------------------
    def _draw_and_explain(self, result, observation_id) -> None:
        bars = self._data["replay"].bars
        if not (self._view_start <= result.entry_position <= self._view_end):
            date = pd.Timestamp(bars.trade_date[result.entry_position]).strftime("%Y-%m-%d")
            self.date_combo.setCurrentText(date)
        entry_local = result.entry_position - self._view_start
        exit_local = result.exit_position - self._view_start
        self.chart.clear_trades()
        self.chart.draw_trade(
            entry_local,
            exit_local,
            result.entry_price,
            result.stop_track,
            result.target_track,
            result.exit_price,
        )
        self.chart.center_on(entry_local)
        self.forensics.setHtml(self._forensics_html(result, observation_id))

    def _forensics_html(self, result, observation_id) -> str:
        side = "Long" if result.direction > 0 else "Short"
        lines = [
            f"<h3>{side} trade &middot; {result.exit_reason}</h3>",
            f"<p>Net path: <b>{result.gross_r:+.2f} R</b> over {result.holding_minutes} min "
            f"(stop {result.stop_ticks:.0f} ticks).</p>",
            f"<p>Within the trade it reached <b>+{result.mfe_r:.2f} R</b> in favour and "
            f"<b>-{result.mae_r:.2f} R</b> against.</p>",
        ]
        if result.gross_r <= 0 and result.mfe_r >= 1.0:
            lines.append(
                "<p><i>It was a winner on the hook - reached target-range profit "
                "before reversing. A trailing stop or breakeven move would have kept some.</i></p>"
            )
        if observation_id is not None and self._data["forensics"].available():
            ctx = self._data["forensics"].context_for(int(observation_id))
            h = ctx.excursions.get(60, {})
            lines.append("<hr><h4>Registered context (60m horizon)</h4>")
            if h:
                lines.append(
                    f"<p>Forward return {h['forward_return_atr']:+.2f} ATR; "
                    f"MFE {h['mfe_long_atr']:+.2f} / MAE {h['mae_long_atr']:+.2f} ATR; "
                    f"expanded: {h['expanded']}.</p>"
                )
            feats = ctx.features
            if feats:
                lines.append(
                    "<p>Entry context: VWAP dist "
                    f"{feats['distance_from_execution_session_vwap_atr']:+.2f} ATR, "
                    f"ATR ratio {feats['atr_ratio_5_20']:.2f}, "
                    f"efficiency {feats['efficiency_ratio_30']:.2f}, "
                    f"choppiness {feats['choppiness_14']:.0f}, "
                    f"session pos {feats['session_range_position']:.2f}, "
                    f"rel vol {feats['relative_volume_60']:.2f}.</p>"
                )
        return "".join(lines)


def _sep() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.VLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line
