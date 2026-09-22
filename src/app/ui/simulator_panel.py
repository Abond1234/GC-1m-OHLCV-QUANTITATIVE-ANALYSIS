"""Resizable simulator workspace shell and summaries of existing replay results."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class SimulatorPanel(QtWidgets.QWidget):
    """Keep simulation controls mounted while the sidebar is hidden or resized."""

    closeRequested = QtCore.Signal()

    def __init__(self, simulate_button, simulation_period, pages, parent=None):
        super().__init__(parent)
        self.setObjectName("simulatorPanel")
        self.setMinimumWidth(370)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Simulator")
        title.setProperty("role", "heading")
        header.addWidget(title)
        header.addStretch(1)
        self.close_button = QtWidgets.QToolButton()
        self.close_button.setText("Close")
        self.close_button.setToolTip("Close Simulator and keep its settings.")
        self.close_button.clicked.connect(self.closeRequested.emit)
        header.addWidget(self.close_button)
        layout.addLayout(header)
        layout.addWidget(simulation_period)
        layout.addWidget(simulate_button)
        self.status = QtWidgets.QLabel("Choose a strategy, then press Simulate.")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        layout.addWidget(self.status)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("simulatorTabs")
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setDocumentMode(True)
        for name, page in pages:
            self.tabs.addTab(page, name)
        layout.addWidget(self.tabs, 1)

    def select(self, name: str) -> None:
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == name:
                self.tabs.setCurrentIndex(index)
                return


class ReplaySummary(QtWidgets.QGroupBox):
    """Challenge outcomes and gross strategy statistics for the latest run."""

    def __init__(self, parent=None):
        super().__init__("Simulation performance", parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        grid = QtWidgets.QGridLayout()
        self.values = {}
        metrics = (
            ("accounts", "Accounts"),
            ("phase_1", "Phase 1 passed"),
            ("phase_2", "Phase 2 passed"),
            ("funded", "Challenges passed"),
            ("payouts", "Payouts received"),
            ("breached", "Accounts breached"),
            ("trades", "Trades taken"),
            ("skipped", "Setups skipped"),
            ("stop_days", "3-breach stop days"),
            ("win_rate", "Win rate"),
            ("mean_r", "Mean gross R"),
            ("total_pnl", "Net simulated P&L"),
            ("payout_value", "Payout value"),
        )
        for index, (key, label) in enumerate(metrics):
            row, column = divmod(index, 3)
            caption = QtWidgets.QLabel(label)
            caption.setProperty("role", "caption")
            value = QtWidgets.QLabel("-")
            value.setProperty("role", "metric")
            grid.addWidget(caption, row * 2, column)
            grid.addWidget(value, row * 2 + 1, column)
            self.values[key] = value
        layout.addLayout(grid)
        self.scope = QtWidgets.QLabel()
        self.scope.setWordWrap(True)
        self.scope.setProperty("role", "caption")
        layout.addWidget(self.scope)
        self.account_table = QtWidgets.QTableWidget(0, 6)
        self.account_table.setHorizontalHeaderLabels(
            ["Account", "Status", "Stage", "Trades", "Balance", "Payout"]
        )
        self.account_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.account_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.account_table.verticalHeader().setVisible(False)
        self.account_table.horizontalHeader().setStretchLastSection(True)
        self.account_table.setMinimumHeight(150)
        layout.addWidget(self.account_table)
        self.clear()

    def clear(self) -> None:
        self.description.setText("No simulation yet.")
        for value in self.values.values():
            value.setText("-")
        self.scope.setText("Run a strategy to see its trades here.")
        self.account_table.setRowCount(0)

    def set_results(
        self,
        result,
        strategy: str | None = None,
        *,
        period: tuple[str, str] | None = None,
    ) -> None:
        name = strategy or "Selected strategy"
        self.description.setText(f"{name} | serial prop-firm challenge simulation")
        self.values["accounts"].setText(f"{result.accounts_requested:,}")
        self.values["phase_1"].setText(f"{result.phase1_passes:,}")
        self.values["phase_2"].setText(f"{result.phase2_passes:,}")
        self.values["funded"].setText(f"{result.challenges_passed:,}")
        self.values["payouts"].setText(f"{result.payouts_received:,}")
        self.values["breached"].setText(f"{result.accounts_breached:,}")
        self.values["trades"].setText(f"{result.trades_taken:,}")
        self.values["skipped"].setText(f"{result.skipped_trades:,}")
        self.values["stop_days"].setText(f"{result.three_breach_stop_days:,}")
        self.values["win_rate"].setText(f"{result.win_rate:.1%}" if result.trades_taken else "-")
        self.values["mean_r"].setText(f"{result.mean_r:+.3f}" if result.trades_taken else "-")
        self.values["total_pnl"].setText(f"${result.total_pnl:+,.2f}")
        self.values["payout_value"].setText(f"${result.total_payout:+,.2f}")
        scope = (
            f"Simulation period {period[0]} to {period[1]} (inclusive entry dates)."
            if period is not None
            else "Loaded Development + Validation data."
        )
        self.scope.setText(
            f"{scope} Closed-trade outcomes use the strategy's verified research exits. "
            "Prop-firm limits and trader daily stops are applied afterward."
        )
        self.account_table.setRowCount(len(result.accounts))
        for row_index, account in enumerate(result.accounts):
            values = (
                str(account.account_id),
                account.status,
                account.stage,
                f"{account.trades:,}",
                f"${account.equity:,.2f}",
                f"${account.payout:,.2f}",
            )
            for column, value in enumerate(values):
                self.account_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(value))
        self.account_table.resizeColumnsToContents()
