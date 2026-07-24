"""Blotter of placed free-play trades.

A selectable table of the trades the user has placed on the chart. Selecting a
row makes that trade active (its stop/target become draggable); Remove/Delete
drops it. It is a thin view over ``PlacedTrade`` objects - all sizing and
simulation live in ``analysis.placed_trade``.
"""

from __future__ import annotations

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis.placed_trade import PlacedTrade
from . import theme

_COLUMNS = ["#", "Entry (NY)", "Dir", "Exit", "R", "Held"]


class TradeBlotter(QtWidgets.QWidget):
    """Table of placed trades; emits selection and removal by trade id."""

    tradeSelected = QtCore.Signal(int)
    tradeRemoved = QtCore.Signal(int)
    cleared = QtCore.Signal()

    def __init__(self, time_label=None, parent=None):
        super().__init__(parent)
        self._time_label = time_label  # callable(entry_position) -> str
        self._build()

    def set_time_label(self, fn) -> None:
        self._time_label = fn

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        self._remove_btn = QtWidgets.QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected)
        self._clear_btn = QtWidgets.QPushButton("Clear all")
        self._clear_btn.clicked.connect(self.cleared.emit)
        bar.addWidget(QtWidgets.QLabel("Placed trades"))
        bar.addStretch(1)
        bar.addWidget(self._remove_btn)
        bar.addWidget(self._clear_btn)
        layout.addLayout(bar)

        self.table = QtWidgets.QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table)

    def _time(self, entry_position: int) -> str:
        if self._time_label is None:
            return str(entry_position)
        return self._time_label(entry_position)

    def set_trades(self, trades: list[PlacedTrade]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for t in trades:
            self._append_row(t)
        self.table.blockSignals(False)

    def _append_row(self, t: PlacedTrade) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        res = t.result
        values = [
            str(t.id),
            self._time(t.entry_position),
            "L" if t.direction > 0 else "S",
            str(res.exit_reason),
            f"{res.gross_r:+.2f}",
            str(res.holding_minutes),
        ]
        for c, v in enumerate(values):
            item = QtWidgets.QTableWidgetItem(v)
            item.setData(QtCore.Qt.UserRole, int(t.id))
            if c == 0 and t.color:
                item.setForeground(QtGui.QColor(t.color))
            if c == 4:
                p = theme.active()
                item.setForeground(QtGui.QColor(p.up if res.gross_r > 0 else p.down))
            self.table.setItem(r, c, item)

    def update_trade(self, t: PlacedTrade) -> None:
        """Refresh one row's Exit/R/Held cells in place (used during a drag)."""

        p = theme.active()
        res = t.result
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(QtCore.Qt.UserRole) != t.id:
                continue
            self.table.item(r, 3).setText(str(res.exit_reason))
            r_item = self.table.item(r, 4)
            r_item.setText(f"{res.gross_r:+.2f}")
            r_item.setForeground(QtGui.QColor(p.up if res.gross_r > 0 else p.down))
            self.table.item(r, 5).setText(str(res.holding_minutes))
            return

    def select_trade(self, trade_id: int) -> None:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(QtCore.Qt.UserRole) == trade_id:
                self.table.blockSignals(True)
                self.table.selectRow(r)
                self.table.blockSignals(False)
                return

    def selected_id(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(QtCore.Qt.UserRole)

    def _on_selection(self) -> None:
        tid = self.selected_id()
        if tid is not None:
            self.tradeSelected.emit(int(tid))

    def _remove_selected(self) -> None:
        tid = self.selected_id()
        if tid is not None:
            self.tradeRemoved.emit(int(tid))

    def keyPressEvent(self, event) -> None:
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self._remove_selected()
        else:
            super().keyPressEvent(event)


def ny_time_label(bars) -> callable:
    """Build a (entry_position -> 'MM-DD HH:MM' New York) formatter over a BarStore.

    Uses the NY trade date and minute-of-day so the label matches the chart's time
    axis (``bars.ts`` is UTC and would read five hours off).
    """

    def _fmt(entry_position: int) -> str:
        i = int(entry_position)
        day = pd.Timestamp(bars.trade_date[i]).strftime("%m-%d")
        m = int(bars.minute_ny[i])
        return f"{day} {m // 60:02d}:{m % 60:02d}"

    return _fmt
