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
from ..analysis.evaluation import POLICY_PRESETS, EvaluationReport, PropFirmPolicy
from . import theme

_OUTPUT_ROWS = (
    ("verdict_state", "Result"),
    ("breach", "Breach"),
    ("to_pass", "To pass"),
    ("payout_eligible", "Payout eligible"),
    ("payout", "Simulated payout"),
    ("post_payout", "Post-payout balance"),
)

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

        rules_box = QtWidgets.QGroupBox("Prop-firm rules")
        rf = QtWidgets.QFormLayout(rules_box)
        rf.setContentsMargins(10, 6, 10, 8)
        rf.setSpacing(6)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(POLICY_PRESETS))
        self.preset_combo.setToolTip("Load an evaluation preset; every field stays editable.")
        rf.addRow("preset", self.preset_combo)
        self.profit_spin = self._pct(8.0, "Profit target, % of starting balance.")
        self.daily_spin = self._pct(5.0, "Daily loss limit, % of starting balance.")
        self.dd_spin = self._pct(10.0, "Maximum drawdown, % of starting balance.")
        self.trailing_check = QtWidgets.QCheckBox("trailing (peak-relative)")
        self.trailing_check.setToolTip("Off = static floor below the start; on = trails the peak.")
        self.min_days_spin = self._intspin(5, 0, 60, "Minimum distinct trading days to pass.")
        self.max_days_spin = self._intspin(0, 0, 120, "Maximum evaluation days (0 = no cap).")
        self.consistency_spin = self._pct(0.0, "Max share of profit one day may make (0 = off).")
        self.contracts_cap_spin = self._intspin(0, 0, 100, "Max contracts per trade (0 = no cap).")
        self.payout_spin = self._pct(80.0, "Trader's share of profit at a payout.")
        rf.addRow("profit target", self.profit_spin)
        rf.addRow("daily loss", self.daily_spin)
        rf.addRow("max drawdown", self.dd_spin)
        rf.addRow("", self.trailing_check)
        rf.addRow("min days", self.min_days_spin)
        rf.addRow("max eval days", self.max_days_spin)
        rf.addRow("consistency", self.consistency_spin)
        rf.addRow("max contracts", self.contracts_cap_spin)
        rf.addRow("payout split", self.payout_spin)
        layout.addWidget(rules_box)

        out_box = QtWidgets.QGroupBox("Evaluation")
        og = QtWidgets.QGridLayout(out_box)
        og.setContentsMargins(10, 6, 10, 8)
        og.setHorizontalSpacing(10)
        og.setVerticalSpacing(4)
        self.verdict = QtWidgets.QLabel("Set the rules and place trades to evaluate.")
        self.verdict.setWordWrap(True)
        og.addWidget(self.verdict, 0, 0, 1, 2)
        self._outputs: dict[str, QtWidgets.QLabel] = {}
        for i, (key, label) in enumerate(_OUTPUT_ROWS):
            name = QtWidgets.QLabel(label)
            name.setProperty("role", "caption")
            value = QtWidgets.QLabel("-")
            og.addWidget(name, i + 1, 0)
            og.addWidget(value, i + 1, 1)
            self._outputs[key] = value
        layout.addWidget(out_box)

        self.util_plot = pg.PlotWidget()
        self.util_plot.setMaximumHeight(120)
        self.util_plot.setMouseEnabled(x=False, y=False)
        self.util_plot.hideButtons()
        self.util_plot.showGrid(x=False, y=True, alpha=0.12)
        self.util_plot.setLabel("left", "rule use")
        layout.addWidget(self.util_plot)

        meters_box = QtWidgets.QGroupBox("Rule meters")
        mv = QtWidgets.QVBoxLayout(meters_box)
        mv.setContentsMargins(10, 6, 10, 8)
        mv.setSpacing(6)
        self._meters: list[tuple[QtWidgets.QLabel, QtWidgets.QProgressBar, QtWidgets.QLabel]] = []
        for _ in range(5):
            row = QtWidgets.QHBoxLayout()
            name = QtWidgets.QLabel("")
            name.setMinimumWidth(96)
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
            mv.addWidget(widget)
            self._meters.append((name, bar, detail))
        layout.addWidget(meters_box)
        layout.addStretch(1)

        for w in (self.balance_spin, self.risk_spin):
            w.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.contracts_spin.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        for c in (self.instrument_combo, self.sizing_combo):
            c.currentTextChanged.connect(lambda _t: self.settingsChanged.emit())
        self.preset_combo.currentTextChanged.connect(self._load_preset)
        for s in (
            self.profit_spin,
            self.daily_spin,
            self.dd_spin,
            self.min_days_spin,
            self.max_days_spin,
            self.consistency_spin,
            self.contracts_cap_spin,
            self.payout_spin,
        ):
            s.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.trailing_check.toggled.connect(lambda _c: self.settingsChanged.emit())
        self.sizing_combo.currentTextChanged.connect(self._sync_sizing)
        self._sync_sizing()

    def _pct(self, value, tip) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 100.0)
        spin.setSingleStep(0.5)
        spin.setValue(value)
        spin.setSuffix(" %")
        spin.setAccelerated(True)
        spin.setToolTip(tip)
        return spin

    def _intspin(self, value, lo, hi, tip) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        spin.setAccelerated(True)
        spin.setToolTip(tip)
        return spin

    def _load_preset(self, name: str) -> None:
        policy = POLICY_PRESETS.get(name)
        if policy is None:
            return
        for widget in (
            self.profit_spin,
            self.daily_spin,
            self.dd_spin,
            self.trailing_check,
            self.min_days_spin,
            self.max_days_spin,
            self.consistency_spin,
            self.contracts_cap_spin,
            self.payout_spin,
        ):
            widget.blockSignals(True)
        self.profit_spin.setValue(policy.profit_target_pct)
        self.daily_spin.setValue(policy.daily_loss_limit_pct)
        self.dd_spin.setValue(policy.max_drawdown_pct)
        self.trailing_check.setChecked(policy.trailing_drawdown)
        self.min_days_spin.setValue(policy.min_trading_days)
        self.max_days_spin.setValue(policy.max_evaluation_days)
        self.consistency_spin.setValue(policy.consistency_max_daily_share * 100.0)
        self.contracts_cap_spin.setValue(policy.max_contracts)
        self.payout_spin.setValue(policy.payout_split_pct)
        for widget in (
            self.profit_spin,
            self.daily_spin,
            self.dd_spin,
            self.trailing_check,
            self.min_days_spin,
            self.max_days_spin,
            self.consistency_spin,
            self.contracts_cap_spin,
            self.payout_spin,
        ):
            widget.blockSignals(False)
        self.settingsChanged.emit()

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

    def policy(self) -> PropFirmPolicy:
        return PropFirmPolicy(
            name=self.preset_combo.currentText(),
            starting_balance=float(self.balance_spin.value()),
            profit_target_pct=float(self.profit_spin.value()),
            daily_loss_limit_pct=float(self.daily_spin.value()),
            max_drawdown_pct=float(self.dd_spin.value()),
            trailing_drawdown=self.trailing_check.isChecked(),
            min_trading_days=int(self.min_days_spin.value()),
            max_evaluation_days=int(self.max_days_spin.value()),
            consistency_max_daily_share=float(self.consistency_spin.value()) / 100.0,
            max_contracts=int(self.contracts_cap_spin.value()),
            payout_split_pct=float(self.payout_spin.value()),
        )

    def policy_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self.policy())

    def load_policy_dict(self, data: dict) -> None:
        """Restore a full custom policy into the controls (one settings emit)."""

        if not data:
            return
        spins = {
            self.balance_spin: data.get("starting_balance", 100_000.0),
            self.profit_spin: data.get("profit_target_pct", 8.0),
            self.daily_spin: data.get("daily_loss_limit_pct", 5.0),
            self.dd_spin: data.get("max_drawdown_pct", 10.0),
            self.min_days_spin: data.get("min_trading_days", 5),
            self.max_days_spin: data.get("max_evaluation_days", 0),
            self.consistency_spin: data.get("consistency_max_daily_share", 0.0) * 100.0,
            self.contracts_cap_spin: data.get("max_contracts", 0),
            self.payout_spin: data.get("payout_split_pct", 80.0),
        }
        for widget in (*spins, self.trailing_check):
            widget.blockSignals(True)
        for widget, value in spins.items():
            widget.setValue(value)
        self.trailing_check.setChecked(bool(data.get("trailing_drawdown", False)))
        for widget in (*spins, self.trailing_check):
            widget.blockSignals(False)
        self.settingsChanged.emit()

    # -- rendering ---------------------------------------------------------
    def update_session(
        self, curve: EquityCurve, stats: SessionStats, report: EvaluationReport
    ) -> None:
        p = theme.active()
        start = self.account_settings().starting_balance
        self.equity_plot.clear()
        if curve.n_trades:
            x = np.arange(curve.n_trades + 1)
            y = np.concatenate(([start], curve.equity))
            self.equity_plot.addItem(pg.PlotDataItem(x, y, pen=pg.mkPen(p.gold, width=1.6)))
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
        self._stat_values["total"].setStyleSheet(
            f"color:{p.up if stats.total_pnl >= 0 else p.down}"
        )

        # Rule-utilization curve: drawdown used as a fraction of the cap (1.0 = out).
        self.util_plot.clear()
        if report.trades_taken:
            x = np.arange(report.trades_taken)
            self.util_plot.addItem(
                pg.PlotDataItem(
                    x, np.clip(report.utilization, 0, 1.3), pen=pg.mkPen(p.gold, width=1.4)
                )
            )
        self.util_plot.addItem(
            pg.InfiniteLine(
                pos=1.0, angle=0, pen=pg.mkPen(p.down, width=1, style=QtCore.Qt.DashLine)
            )
        )

        for (name, bar, detail), rule in zip(self._meters, report.rules, strict=True):
            name.setText(rule.label)
            bar.setVisible(True)
            frac = min(1.0, rule.used / rule.limit) if rule.limit > 0 else 0.0
            bar.setValue(int(frac * 1000))
            detail.setText(rule.detail)

        banner = {
            "PASSED": (f"EVALUATION PASSED in {report.days_to_pass} days.", p.up),
            "FAILED": (f"EVALUATION FAILED - {report.breach_detail}.", p.down),
            "IN_PROGRESS": ("Evaluation in progress.", p.text_dim),
        }[report.state]
        self.verdict.setText(banner[0])
        self.verdict.setStyleSheet(f"color:{banner[1]}; font-weight:600")

        labels = {r.rule: r.label for r in report.rules}
        breach = (
            f"{labels.get(report.failed_rule, report.failed_rule)} on {report.breach_date}"
            if report.failed_rule
            else "none"
        )
        to_pass = (
            f"{report.days_to_pass} days / {report.trades_to_pass} trades"
            if report.days_to_pass >= 0
            else "-"
        )
        self._outputs["verdict_state"].setText(report.state.replace("_", " ").title())
        self._outputs["breach"].setText(breach)
        self._outputs["to_pass"].setText(to_pass)
        self._outputs["payout_eligible"].setText("Yes" if report.payout_eligible else "No")
        self._outputs["payout"].setText(f"{report.simulated_payout:,.0f} $")
        self._outputs["post_payout"].setText(f"{report.post_payout_balance:,.0f} $")

    def retheme(self) -> None:
        self.equity_plot.setBackground(theme.active().bg)
        self.util_plot.setBackground(theme.active().bg)
