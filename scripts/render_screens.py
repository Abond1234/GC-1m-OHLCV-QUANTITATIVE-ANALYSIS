"""Offscreen render harness for the trade-simulator UI.

Builds the real ``MainWindow`` under the Qt ``offscreen`` platform, wired to a
narrow bar load (a handful of recent Development days via ``date_floor``, so it
loads in well under a second rather than reading the full 600 MB bars table),
drives it through the same slots a user would, and writes PNG screenshots. This
is how UI changes are visually verified without a display.

Usage:
    python scripts/render_screens.py [OUT_DIR] [DATE_FLOOR]

Screenshots default to a git-ignored ``reports/ui_smoke/`` directory. Requires the
app deps (PySide6, pyqtgraph) and the local parquet artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Platform: leave to the OS default so real fonts render (Windows/macOS with a
# window station). In a truly headless env (Linux CI) export
# ``QT_QPA_PLATFORM=offscreen`` before running; text there falls back to boxes.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from src.app.ui import theme  # noqa: E402
from src.app.ui.main_window import MainWindow, load_app_data  # noqa: E402

# A month of Validation loads in a second or so and has dense trades to screenshot.
DEFAULT_DATE_FLOOR = pd.Timestamp("2024-12-01")


def _settle(app: QtWidgets.QApplication, rounds: int = 8) -> None:
    """Drain worker threads and pyqtgraph's deferred paints before a grab."""

    from PySide6 import QtCore

    QtCore.QThreadPool.globalInstance().waitForDone(3000)
    for _ in range(rounds):
        app.processEvents()


def _grab(widget, path: Path, app: QtWidgets.QApplication) -> Path:
    _settle(app)
    widget.grab().save(str(path))
    return path


def _first_non_empty_strategy(win: MainWindow):
    """Pick a strategy that actually fires in the narrow window (prefer a benchmark)."""

    specs = win._data["replay"].list_strategies()
    ordered = sorted(specs, key=lambda s: 0 if "long" in s.name.lower() else 1)
    for spec in ordered:
        log = win._data["replay"].replay(spec, None)
        if not log.empty:
            return spec, log
    return None, None


def _qd(ts):
    from PySide6 import QtCore

    t = pd.Timestamp(ts)
    return QtCore.QDate(t.year, t.month, t.day)


def _set_window(win: MainWindow, start_ts, end_ts) -> None:
    win._date_guard = True
    win.start_edit.setDate(_qd(start_ts))
    win.end_edit.setDate(_qd(end_ts))
    win._date_guard = False
    win._render_view()


def _set_day(win: MainWindow, ts) -> None:
    _set_window(win, ts, ts)


def render_all(out_dir: Path | None = None, date_floor: pd.Timestamp | None = None) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else PROJECT_ROOT / "reports" / "ui_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_floor = DEFAULT_DATE_FLOOR if date_floor is None else pd.Timestamp(date_floor)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, "dark")

    data = load_app_data(date_floor=date_floor)
    win = MainWindow(data=data)
    win.resize(1600, 940)
    win.show()

    saved: list[Path] = []

    # 1. Overview of the last loaded day with VWAP + session shading.
    _set_day(win, win.end_edit.date().toPython())  # already the most recent day
    win.indicators_panel.add_study("vwap_day")
    win.indicators_panel.add_study("vwap_session")
    saved.append(_grab(win, out_dir / "01_chart_overview.png", app))
    win.simulator_button.click()
    saved.append(_grab(win, out_dir / "01b_simulator_strategy.png", app))
    win.simulator_panel.select("Risk")
    saved.append(_grab(win, out_dir / "01c_simulator_risk.png", app))
    win.simulator_panel.select("Prop firm")
    saved.append(_grab(win, out_dir / "01d_simulator_prop_firm.png", app))

    # 2. Replay a strategy so outcome-coloured markers appear.
    spec, log = _first_non_empty_strategy(win)
    if spec is not None:
        win.strategy_browser.select(spec.name)
        win._on_replayed(log)
        # Jump to the day of the first trade and draw its markers.
        first = log.iloc[0]
        _set_day(win, pd.Timestamp(first["trade_date_ny"]))
        saved.append(_grab(win, out_dir / "02_replay_markers.png", app))

        # 3. Focus one trade: its path + forensics.
        win._focus_trade(first)
        saved.append(_grab(win, out_dir / "03_trade_detail.png", app))

    # 4. Free-play: place a long and a short on the current day; show the exit panel.
    win.freeplay_check.setChecked(True)
    win.exit_panel.trail_check.setChecked(True)  # a trailing stop to watch it ratchet
    lo, hi = win._view_start, win._view_end
    win.long_radio.setChecked(True)
    win._on_bar_clicked(lo + (hi - lo) // 3)
    win.short_radio.setChecked(True)
    win._on_bar_clicked(lo + (hi - lo) // 2)
    win.long_radio.setChecked(True)
    saved.append(_grab(win, out_dir / "04_freeplay_trades.png", app))

    # 5. Drag the active trade's stop further from entry (a simulated live drag).
    active = win._placed.get(win._active_id)
    if active is not None:
        entry = active.result.entry_price
        widen = active.stop_points * 1.8
        new_stop = entry - widen if active.direction > 0 else entry + widen
        win._on_level_dragged("stop", new_stop)
        saved.append(_grab(win, out_dir / "05_drag_stop.png", app))

    # 6. What-if exits: overlay alternative exits and the comparison table.
    win._on_whatif()
    saved.append(_grab(win, out_dir / "06_whatif.png", app))

    # 7. Exit-grid heatmap for the current trade.
    win._on_exit_grid()
    saved.append(_grab(win, out_dir / "07_exit_grid.png", app))

    # 8. Animated trade replay: load and step to a mid-frame (future hidden).
    result, _cfg = win._current()
    if result is not None:
        win._animator.load(result)
        win._animator.step((result.exit_position - result.entry_position) // 2)
        saved.append(_grab(win, out_dir / "08_replay_frame.png", app))

    # 8b. Day replay: the tape revealed to mid-day, no trade attached.
    win._replay_day()
    win._animator.pause()
    win._animator.step((win._view_end - win._view_start) // 2)
    saved.append(_grab(win, out_dir / "08b_day_replay.png", app))
    win._animator.stop()

    # 8c. Studies: EMA + Bollinger on price, an RSI pane, volume hidden.
    win.indicators_panel.add_study("ema")
    win.indicators_panel.add_study("bollinger")
    win.indicators_panel.add_study("rsi")
    win.indicators_panel.volume_check.setChecked(False)
    win._set_simulator_visible(False)
    win._select_workspace(0)  # the Indicators workspace on the left
    saved.append(_grab(win, out_dir / "08c_studies.png", app))
    win.indicators_panel.volume_check.setChecked(True)
    for inst in list(win.indicators_panel.instances):
        if inst.key != "vwap20":
            win.indicators_panel.remove_instance(inst.id)

    # 8d. Drawing tools: fib retracement, long position, and a measurement.
    win._select_workspace(1)  # the Drawing workspace on the left
    lo = win._view_start
    close = win._bars.close

    def _price_at(local_x: int) -> float:
        return float(close[lo + local_x])

    win.chart.place_drawing("fibret", 120.0, _price_at(120))
    win.chart.place_drawing("fibret", 420.0, _price_at(420))
    win.chart.place_drawing("longpos", 520.0, _price_at(520))
    win.chart.place_drawing("longpos", 700.0, _price_at(520) + 3.0)
    win.chart.place_drawing("measure", 40.0, _price_at(40))
    win.chart.place_drawing("measure", 110.0, _price_at(110))
    saved.append(_grab(win, out_dir / "08d_drawing_tools.png", app))
    win.chart.clear_drawings()
    win._select_workspace(1)  # collapse chart tools
    win._show_performance(win.trade_table)

    # 9. Light theme.
    win._set_theme("light")
    saved.append(_grab(win, out_dir / "09_light_theme.png", app))
    win._set_theme("dark")

    # 9b. Multi-day literal window at an aggregated timeframe: a week at 15m.
    end_day = win.end_edit.date().toPython()
    _set_window(win, end_day - pd.Timedelta(days=7), end_day)
    win.tf_combo.setCurrentText("15m")
    saved.append(_grab(win, out_dir / "09b_week_15m.png", app))
    _set_day(win, end_day)
    win.tf_combo.setCurrentText("1m")

    # 10-12. Narrow-display pass: 1366x768 at 125% DPI is ~1092x614 logical
    # pixels. The truncation bugs the wide render can never show live here.
    win.resize(1092, 614)
    win.simulator_panel.select("Strategy")
    saved.append(_grab(win, out_dir / "10a_narrow_simulator_strategy.png", app))
    win.simulator_panel.select("Risk")
    saved.append(_grab(win, out_dir / "10b_narrow_simulator_risk.png", app))
    win.simulator_panel.select("Prop firm")
    saved.append(_grab(win, out_dir / "10c_narrow_simulator_prop_firm.png", app))
    win._show_performance(win.trade_table)
    saved.append(_grab(win, out_dir / "10_narrow_overview.png", app))
    win._tabs.setCurrentWidget(win.whatif_table)
    saved.append(_grab(win, out_dir / "11_narrow_whatif.png", app))
    win._tabs.setCurrentWidget(win.heatmap)
    saved.append(_grab(win, out_dir / "12_narrow_heatmap.png", app))
    win.resize(1600, 940)

    for pth in saved:
        print(pth)
    win.close()
    _settle(app)
    return saved


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    floor = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else None
    render_all(out, floor)
