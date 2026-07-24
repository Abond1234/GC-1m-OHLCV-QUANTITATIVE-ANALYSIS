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
    win.date_combo.setCurrentIndex(win.date_combo.count() - 1)
    win.vwap_day_check.setChecked(True)
    win.vwap_session_check.setChecked(True)
    saved.append(_grab(win, out_dir / "01_chart_overview.png", app))

    # 2. Replay a strategy so outcome-coloured markers appear.
    spec, log = _first_non_empty_strategy(win)
    if spec is not None:
        win.strategy_combo.setCurrentText(spec.name)
        win._on_replayed(log)
        # Jump to the day of the first trade and draw its markers.
        first = log.iloc[0]
        win.date_combo.setCurrentText(pd.Timestamp(first["trade_date_ny"]).strftime("%Y-%m-%d"))
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

    for pth in saved:
        print(pth)
    return saved


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    floor = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else None
    render_all(out, floor)
