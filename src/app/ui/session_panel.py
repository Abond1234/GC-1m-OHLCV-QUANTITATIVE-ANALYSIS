"""Session tab: dollar equity curve, desk stats, and evaluation-mode meters.

The money view of the free-play blotter. An account model (balance, sizing,
instrument economics) turns R multiples into dollars; the equity curve and stat
grid summarise the session; an optional prop-style evaluation preset arms hard
rules (daily loss, max drawdown, profit target, minimum days) with live meters
and a verdict banner. All simulation accounting - clearly labelled as such.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..analysis.account import (
    GC_SPEC,
    MGC_SPEC,
    AccountSettings,
    EquityCurve,
    InstrumentSpec,
    SessionStats,
)
from ..analysis.evaluation import PRESETS, EvaluationRules, EvaluationStatus
from . import theme

_STAT_ROWS = (
    ("trades", "Trades"),
    ("win_rate", "Win rate"),
    ("avg_r", "Avg R"),
    ("expectancy", "Expectancy"),
    ("profit_factor", "Profit factor"),
    ("total", "Total P&L"),
    ("equity", "Equity"),
    ("max_dd", "Max drawdown"),
    ("days", "Days traded"),
)


class SessionPanel(QtWidgets.QWidget):
    """Equity curve + stats + account settings + evaluation meters."""

    settingsChanged = QtCore.Signal()  # account or evaluation settings edited

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        caption = QtWidgets.QLabel(
            "Simulated fills on historical Dev+Val data - not live results and "
            "not a validated strategy."
        )
        caption.setWordWrap(True)
        caption.setProperty("role", "caption")
        layout.addWidget(caption)

        self.equity_plot = pg.PlotWidget()
        self.equity_plot.setMaximumHeight(170)
        self.equity_plot.setMouseEnabled(x=False, y=False)
        self.equity_plot.hideButtons()
        self.equity_plot.showGrid(x=False, y=True, alpha=0.12)
        self.equity_plot.setLabel("left", "$")
        layout.addWidget(self.equity_plot)

        stats_box = QtWidgets.QGroupBox("Session")
        grid = QtWidgets.QGridLayout(stats_box)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self._stat_values: dict[str, QtWidgets.QLabel] = {}
        for i, (key, label) in enumerate(_STAT_ROWS):
            row, col = divmod(i, 2)
            name = QtWidgets.QLabel(label)
            name.setProperty("role", "caption")
            value = QtWidgets.QLabel("-")
            grid.addWidget(name, row, col * 2)
            grid.addWidget(value, row, col * 2 + 1)
            self._stat_values[key] = value
        layout.addWidget(stats_box)

        account_box = QtWidgets.QGroupBox("Account")
        form = QtWidgets.QFormLayout(account_box)
        form.setContentsMargins(10, 6, 10, 8)
        form.setSpacing(6)
        self.balance_spin = QtWidgets.QDoubleSpinBox()
        self.balance_spin.setRange(1_000, 10_000_000)
        self.balance_spin.setDecimals(0)
        self.balance_spin.setSingleStep(1_000)
        self.balance_spin.setValue(100_000)
        self.balance_spin.setAccelerated(True)
        self.balance_spin.setToolTip("Starting balance for the session's accounting.")
        self.instrument_combo = QtWidgets.QComboBox()
        self._panel_specs: list[InstrumentSpec] = [GC_SPEC, MGC_SPEC]
        self.instrument_combo.addItems(["GC ($100/pt)", "MGC ($10/pt)"])
        self.instrument_combo.setToolTip(
            "Contract used to convert points to dollars. The chart's simulation\n"
            "is GC either way; MGC prices the same fills as micro contracts."
        )
        self.sizing_combo = QtWidgets.QComboBox()
        self.sizing_combo.addItems(["Risk % of equity", "Fixed contracts"])
        self.risk_spin = QtWidgets.QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 10.0)
        self.risk_spin.setSingleStep(0.1)
        self.risk_spin.setValue(1.0)
        self.risk_spin.setSuffix(" %")
        self.risk_spin.setAccelerated(True)
        self.risk_spin.setToolTip("Risk per trade as a percent of current equity.")
        self.contracts_spin = QtWidgets.QSpinBox()
        self.contracts_spin.setRange(1, 100)
        self.contracts_spin.setValue(1)
        self.contracts_spin.setAccelerated(True)
        form.addRow("balance $", self.balance_spin)
        form.addRow("instrument", self.instrument_combo)
        form.addRow("sizing", self.sizing_combo)
        form.addRow("risk / trade", self.risk_spin)
        form.addRow("contracts", self.contracts_spin)
        layout.addWidget(account_box)

        eval_box = QtWidgets.QGroupBox("Evaluation")
        ev = QtWidgets.QVBoxLayout(eval_box)
        ev.setContentsMargins(10, 6, 10, 8)
        ev.setSpacing(6)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(PRESETS))
        self.preset_combo.setToolTip(
            "Arm prop-firm style hard rules over the session: daily loss,\n"
            "max drawdown, profit target, minimum trading days."
        )
        ev.addWidget(self.preset_combo)
        self.verdict = QtWidgets.QLabel("Practice mode - no rules armed.")
        self.verdict.setWordWrap(True)
        ev.addWidget(self.verdict)
        self._meters: list[tuple[QtWidgets.QLabel, QtWidgets.QProgressBar, QtWidgets.QLabel]] = []
        for _ in range(4):
            row = QtWidgets.QHBoxLayout()
            name = QtWidgets.QLabel("")
            name.setMinimumWidth(84)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            detail = QtWidgets.QLabel("")
            detail.setProperty("role", "caption")
            row.addWidget(name)
            row.addWidget(bar, 1)
            widget = QtWidgets.QWidget()
            col = QtWidgets.QVBoxLayout(widget)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(1)
            col.addLayout(row)
            col.addWidget(detail)
            ev.addWidget(widget)
            self._meters.append((name, bar, detail))
        layout.addWidget(eval_box)
        layout.addStretch(1)

        for w in (self.balance_spin, self.risk_spin):
            w.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.contracts_spin.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        for c in (self.instrument_combo, self.sizing_combo, self.preset_combo):
            c.currentTextChanged.connect(lambda _t: self.settingsChanged.emit())
        self.sizing_combo.currentTextChanged.connect(self._sync_sizing)
        self._sync_sizing()

    def _sync_sizing(self, *_a) -> None:
        risk_mode = self.sizing_combo.currentIndex() == 0
        self.risk_spin.setEnabled(risk_mode)
        self.contracts_spin.setEnabled(not risk_mode)

    def sync_instruments(self, instruments, active_symbol: str) -> None:
        """Offer the available instruments' economics; select the active one."""

        self._panel_specs = [inst.spec for inst in instruments]
        self.instrument_combo.blockSignals(True)
        self.instrument_combo.clear()
        active_index = 0
        for i, inst in enumerate(instruments):
            per_point = inst.spec.dollars_per_point
            self.instrument_combo.addItem(f"{inst.symbol} (${per_point:g}/pt)")
            if inst.symbol == active_symbol:
                active_index = i
        self.instrument_combo.setCurrentIndex(active_index)
        self.instrument_combo.blockSignals(False)

    # -- settings ----------------------------------------------------------
    def account_settings(self) -> AccountSettings:
        return AccountSettings(
            starting_balance=float(self.balance_spin.value()),
            sizing_mode="risk_percent"
            if self.sizing_combo.currentIndex() == 0
            else "fixed_contracts",
            risk_percent=float(self.risk_spin.value()),
            fixed_contracts=int(self.contracts_spin.value()),
            instrument=self._panel_specs[
                max(0, min(self.instrument_combo.currentIndex(), len(self._panel_specs) - 1))
            ],
        )

    def evaluation_rules(self) -> EvaluationRules | None:
        return PRESETS.get(self.preset_combo.currentText())

    # -- rendering ---------------------------------------------------------
    def update_session(
        self,
        curve: EquityCurve,
        stats: SessionStats,
        status: EvaluationStatus | None,
    ) -> None:
        p = theme.active()
        self.equity_plot.clear()
        start = self.account_settings().starting_balance
        if curve.n_trades:
            x = np.arange(curve.n_trades + 1)
            y = np.concatenate(([start], curve.equity))
            self.equity_plot.addItem(
                pg.PlotDataItem(x, y, pen=pg.mkPen(p.gold, width=1.6), stepMode=None)
            )
        self.equity_plot.addItem(
            pg.InfiniteLine(pos=start, angle=0, pen=pg.mkPen(p.text_faint, width=1))
        )

        fmt = {
            "trades": f"{stats.n_trades}",
            "win_rate": f"{stats.win_rate * 100.0:.0f}%" if stats.n_trades else "-",
            "avg_r": f"{stats.avg_r:+.2f}" if stats.n_trades else "-",
            "expectancy": f"{stats.expectancy_usd:+,.0f} $/trade" if stats.n_trades else "-",
            "profit_factor": (
                "inf" if stats.profit_factor == float("inf") else f"{stats.profit_factor:.2f}"
            )
            if stats.n_trades
            else "-",
            "total": f"{stats.total_pnl:+,.0f} $",
            "equity": f"{stats.end_equity:,.0f} $",
            "max_dd": f"-{stats.max_drawdown_usd:,.0f} $ ({stats.max_drawdown_pct:.1f}%)",
            "days": f"{stats.days_traded}",
        }
        for key, label in self._stat_values.items():
            label.setText(fmt.get(key, "-"))
        colour = p.up if stats.total_pnl >= 0 else p.down
        self._stat_values["total"].setStyleSheet(f"color:{colour}")

        if status is None:
            self.verdict.setText("Practice mode - no rules armed.")
            self.verdict.setStyleSheet(f"color:{p.text_dim}")
            for name, bar, detail in self._meters:
                name.setText("")
                bar.setValue(0)
                bar.setVisible(False)
                detail.setText("")
            return
        for (name, bar, detail), rule in zip(self._meters, status.rules, strict=True):
            name.setText(rule.label)
            bar.setVisible(True)
            frac = min(1.0, rule.used / rule.limit) if rule.limit > 0 else 0.0
            bar.setValue(int(frac * 1000))
            detail.setText(rule.detail)
        if status.state == "PASSED":
            self.verdict.setText("EVALUATION PASSED - target reached within the rules.")
            self.verdict.setStyleSheet(f"color:{p.up}; font-weight:600")
        elif status.state == "FAILED":
            labels = {r.rule: r.label for r in status.rules}
            self.verdict.setText(
                f"EVALUATION FAILED - {labels.get(status.failed_rule, status.failed_rule)} "
                f"rule breached."
            )
            self.verdict.setStyleSheet(f"color:{p.down}; font-weight:600")
        else:
            self.verdict.setText("Evaluation in progress.")
            self.verdict.setStyleSheet(f"color:{p.text_dim}")

    def retheme(self) -> None:
        self.equity_plot.setBackground(theme.active().bg)
