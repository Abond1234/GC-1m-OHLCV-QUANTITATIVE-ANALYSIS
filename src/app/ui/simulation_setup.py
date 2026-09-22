"""Focused Prop firm and Risk controls for challenge simulations."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..analysis.challenge_simulator import ChallengeRules, PhaseRules, RiskRules


def _caption(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setProperty("role", "caption")
    return label


def _percent(value: float, minimum: float = 0.0, maximum: float = 100.0):
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(2)
    spin.setSingleStep(0.25)
    spin.setValue(value)
    spin.setSuffix(" %")
    spin.setAccelerated(True)
    return spin


class PropFirmPanel(QtWidgets.QWidget):
    """Account-pool and one/two-step challenge rules."""

    settingsChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(
            _caption(
                "Accounts trade serially: only one account receives a setup at a time. "
                "A phase pass upgrades that account; a firm loss breach retires it."
            )
        )

        pool = QtWidgets.QGroupBox("Challenge accounts")
        pool_form = QtWidgets.QFormLayout(pool)
        self.account_count = QtWidgets.QSpinBox()
        self.account_count.setRange(1, 100)
        self.account_count.setValue(10)
        self.account_count.setToolTip("Number of independent prop accounts in this simulation.")
        self.account_size = QtWidgets.QDoubleSpinBox()
        self.account_size.setRange(1_000, 10_000_000)
        self.account_size.setDecimals(0)
        self.account_size.setSingleStep(5_000)
        self.account_size.setValue(100_000)
        self.account_size.setPrefix("$ ")
        self.steps = QtWidgets.QComboBox()
        self.steps.addItems(["1 step", "2 step"])
        pool_form.addRow("accounts", self.account_count)
        pool_form.addRow("account size", self.account_size)
        pool_form.addRow("challenge", self.steps)
        layout.addWidget(pool)

        (
            self.phase1_box,
            self.phase1_target,
            self.phase1_daily,
            self.phase1_max,
            self.phase1_consistency_enabled,
            self.phase1_consistency,
        ) = self._phase_box("Phase 1 rules", 8.0, 5.0, 10.0)
        (
            self.phase2_box,
            self.phase2_target,
            self.phase2_daily,
            self.phase2_max,
            self.phase2_consistency_enabled,
            self.phase2_consistency,
        ) = self._phase_box("Phase 2 rules", 5.0, 5.0, 10.0)
        layout.addWidget(self.phase1_box)
        layout.addWidget(self.phase2_box)

        payout = QtWidgets.QGroupBox("Funded payout")
        payout_form = QtWidgets.QFormLayout(payout)
        self.payout_target = _percent(5.0, 3.0, 5.0)
        self.payout_target.setToolTip(
            "Funded-account profit required for a payout, constrained to 3-5% "
            "of the original account size."
        )
        payout_form.addRow("payout target", self.payout_target)
        (
            self.funded_consistency_enabled,
            self.funded_consistency,
            funded_consistency_row,
        ) = self._consistency_control()
        payout_form.addRow("consistency rule", funded_consistency_row)
        payout_form.addRow(
            "",
            _caption("The funded stage reuses the final challenge phase's loss limits."),
        )
        layout.addWidget(payout)
        layout.addStretch(1)

        self.steps.currentIndexChanged.connect(self._sync_steps)
        for widget in (
            self.account_count,
            self.account_size,
            self.phase1_target,
            self.phase1_daily,
            self.phase1_max,
            self.phase1_consistency,
            self.phase2_target,
            self.phase2_daily,
            self.phase2_max,
            self.phase2_consistency,
            self.payout_target,
            self.funded_consistency,
        ):
            widget.valueChanged.connect(lambda _value: self.settingsChanged.emit())
        for checkbox in (
            self.phase1_consistency_enabled,
            self.phase2_consistency_enabled,
            self.funded_consistency_enabled,
        ):
            checkbox.toggled.connect(lambda _checked: self.settingsChanged.emit())
        self.steps.currentIndexChanged.connect(lambda _index: self.settingsChanged.emit())
        self._sync_steps()

    @staticmethod
    def _consistency_control():
        enabled = QtWidgets.QCheckBox("enabled")
        percent = _percent(40.0, 1.0, 100.0)
        percent.setEnabled(False)
        tip = (
            "Require the largest profitable day to be no more than this percentage "
            "of total profit in this stage. The rule delays a pass or payout; it does "
            "not breach the account."
        )
        enabled.setToolTip(tip)
        percent.setToolTip(tip)
        enabled.toggled.connect(percent.setEnabled)
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(enabled)
        row_layout.addWidget(percent, 1)
        return enabled, percent, row

    @staticmethod
    def _phase_box(title: str, target: float, daily: float, maximum: float):
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        target_spin = _percent(target, 0.25, 100.0)
        daily_spin = _percent(daily, 0.25, 100.0)
        max_spin = _percent(maximum, 0.25, 100.0)
        target_spin.setToolTip("Profit required to pass this phase, % of original size.")
        daily_spin.setToolTip("Firm daily loss breach, % of original account size.")
        max_spin.setToolTip("Firm overall loss breach, % of original account size.")
        consistency_enabled, consistency, consistency_row = PropFirmPanel._consistency_control()
        form.addRow("profit target", target_spin)
        form.addRow("daily loss limit", daily_spin)
        form.addRow("overall loss limit", max_spin)
        form.addRow("consistency rule", consistency_row)
        return box, target_spin, daily_spin, max_spin, consistency_enabled, consistency

    def _sync_steps(self, *_args) -> None:
        self.phase2_box.setVisible(self.steps.currentIndex() == 1)

    def rules(self) -> ChallengeRules:
        phases = [
            PhaseRules(
                float(self.phase1_target.value()),
                float(self.phase1_daily.value()),
                float(self.phase1_max.value()),
                (
                    float(self.phase1_consistency.value())
                    if self.phase1_consistency_enabled.isChecked()
                    else None
                ),
            )
        ]
        if self.steps.currentIndex() == 1:
            phases.append(
                PhaseRules(
                    float(self.phase2_target.value()),
                    float(self.phase2_daily.value()),
                    float(self.phase2_max.value()),
                    (
                        float(self.phase2_consistency.value())
                        if self.phase2_consistency_enabled.isChecked()
                        else None
                    ),
                )
            )
        return ChallengeRules(
            account_count=int(self.account_count.value()),
            starting_balance=float(self.account_size.value()),
            phases=tuple(phases),
            payout_target_pct=float(self.payout_target.value()),
            funded_consistency_pct=(
                float(self.funded_consistency.value())
                if self.funded_consistency_enabled.isChecked()
                else None
            ),
        )


class RiskPanel(QtWidgets.QWidget):
    """Only the risk controls that drive the challenge rotation engine."""

    settingsChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(
            _caption(
                "These are trader-level daily stops. Reaching one pauses the active "
                "account for that day and rotates the next eligible account in."
            )
        )
        box = QtWidgets.QGroupBox("Risk plan")
        form = QtWidgets.QFormLayout(box)
        self.risk_per_trade = _percent(1.0, 0.05, 10.0)
        self.daily_profit_target = _percent(2.0, 0.0, 25.0)
        self.daily_loss_limit = _percent(2.0, 0.0, 25.0)
        self.max_trades = QtWidgets.QSpinBox()
        self.max_trades.setRange(0, 100)
        self.max_trades.setValue(0)
        self.max_trades.setSpecialValueText("No limit")
        self.risk_per_trade.setToolTip("Risk per trade as a percentage of current equity.")
        self.daily_profit_target.setToolTip(
            "Pause this account after reaching the daily profit amount; 0 disables it."
        )
        self.daily_loss_limit.setToolTip(
            "Pause this account after reaching the trader daily loss stop; 0 disables it."
        )
        self.max_trades.setToolTip("Maximum trades per account per day; 0 means no cap.")
        form.addRow("risk per trade", self.risk_per_trade)
        form.addRow("daily profit target", self.daily_profit_target)
        form.addRow("daily max loss", self.daily_loss_limit)
        form.addRow("max trades / day", self.max_trades)
        layout.addWidget(box)
        layout.addWidget(
            _caption(
                "Firm daily and overall breach limits are configured separately in Prop firm. "
                "Three consecutive firm breaches stop the entire simulation for that day."
            )
        )
        layout.addStretch(1)
        for widget in (
            self.risk_per_trade,
            self.daily_profit_target,
            self.daily_loss_limit,
            self.max_trades,
        ):
            widget.valueChanged.connect(lambda _value: self.settingsChanged.emit())

    def rules(self) -> RiskRules:
        return RiskRules(
            risk_per_trade_pct=float(self.risk_per_trade.value()),
            daily_profit_target_pct=float(self.daily_profit_target.value()),
            daily_loss_limit_pct=float(self.daily_loss_limit.value()),
            max_trades_per_day=int(self.max_trades.value()),
        )
