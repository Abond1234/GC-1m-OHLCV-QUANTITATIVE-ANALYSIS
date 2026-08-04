"""Strategy workspace browser: search, family groups, states, and details.

Replaces the flat editable combo. The full library renders as collapsible
family groups in a compact tree; a search bar and a family filter narrow it;
details (thesis, entry condition, parameters) appear only for the selected
strategy. Every row supports the four states the brief demands: selected
(native highlight), loading (while its replay computes), disabled (off-GC,
with the reason on screen), and error (a failed replay, message on the row).
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme


class StrategyBrowser(QtWidgets.QWidget):
    strategyActivated = QtCore.Signal(str)  # double-click / Return on a row

    def __init__(self, parent=None):
        super().__init__(parent)
        self._specs: dict[str, object] = {}
        self._rows: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._loading: str | None = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search strategies...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        top.addWidget(self.search, 1)
        self.family_combo = QtWidgets.QComboBox()
        self.family_combo.addItem("All families")
        self.family_combo.setToolTip("Show one strategy family, or all of them.")
        self.family_combo.currentTextChanged.connect(self._apply_filter)
        top.addWidget(self.family_combo)
        lay.addLayout(top)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)  # stays responsive with the full catalog
        self.tree.itemSelectionChanged.connect(self._on_selection)
        self.tree.itemActivated.connect(self._on_activated)
        lay.addWidget(self.tree, 1)

        self.reason = QtWidgets.QLabel("")
        self.reason.setWordWrap(True)
        self.reason.setProperty("role", "caption")
        self.reason.setVisible(False)
        lay.addWidget(self.reason)

        self.details = QtWidgets.QLabel("Select a strategy to see its thesis.")
        self.details.setWordWrap(True)
        self.details.setTextFormat(QtCore.Qt.RichText)
        self.details.setProperty("role", "caption")
        lay.addWidget(self.details)

    # -- population ---------------------------------------------------------
    def populate(self, specs) -> None:
        self.tree.clear()
        self._rows.clear()
        self._specs = {s.name: s for s in specs}
        families: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for spec in specs:
            family = spec.family or "Other"
            group = families.get(family)
            if group is None:
                group = QtWidgets.QTreeWidgetItem([family])
                group.setFlags(group.flags() & ~QtCore.Qt.ItemIsSelectable)
                families[family] = group
                self.tree.addTopLevelItem(group)
            row = QtWidgets.QTreeWidgetItem([spec.name])
            row.setToolTip(0, spec.thesis)
            group.addChild(row)
            self._rows[spec.name] = row
        for family, group in families.items():
            group.setText(0, f"{family} ({group.childCount()})")
            group.setData(0, QtCore.Qt.UserRole, family)
            group.setExpanded(len(families) <= 3)
        self.family_combo.blockSignals(True)
        while self.family_combo.count() > 1:
            self.family_combo.removeItem(1)
        self.family_combo.addItems(sorted(families))
        self.family_combo.blockSignals(False)

    # -- filtering ----------------------------------------------------------
    def _apply_filter(self, *_a) -> None:
        needle = self.search.text().strip().lower()
        family = self.family_combo.currentText()
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            group_family = group.data(0, QtCore.Qt.UserRole)
            family_ok = family == "All families" or group_family == family
            visible_children = 0
            for j in range(group.childCount()):
                row = group.child(j)
                match = family_ok and (not needle or needle in row.text(0).lower())
                row.setHidden(not match)
                visible_children += int(match)
            group.setHidden(visible_children == 0)
            if needle and visible_children:
                group.setExpanded(True)  # search opens the groups that match

    # -- selection and details ----------------------------------------------
    def current_name(self) -> str | None:
        items = self.tree.selectedItems()
        if not items or items[0].parent() is None:
            return None
        return items[0].text(0).replace(" (computing...)", "")

    def select(self, name: str) -> None:
        row = self._rows.get(name)
        if row is not None:
            self.tree.setCurrentItem(row)
            self.tree.scrollToItem(row)

    def _on_selection(self) -> None:
        name = self.current_name()
        spec = self._specs.get(name)
        if spec is None:
            self.details.setText("Select a strategy to see its thesis.")
            return
        p = theme.active()
        parts = [f"<b style='color:{p.text}'>{spec.name}</b>", spec.thesis]
        if spec.looks_for:
            parts.append(f"<i>Looks for:</i> {spec.looks_for}")
        if spec.parameters:
            args = ", ".join(f"{k}={v}" for k, v in spec.parameters.items())
            parts.append(f"<i>Parameters:</i> {args}")
        self.details.setText("<br>".join(parts))

    def _on_activated(self, item, _column) -> None:
        if item.parent() is not None:  # a strategy row, not a family group
            self.strategyActivated.emit(item.text(0).replace(" (computing...)", ""))

    # -- row states ---------------------------------------------------------
    def set_loading(self, name: str | None) -> None:
        """Mark one row as computing (italic + suffix); None clears it."""

        if self._loading and self._loading in self._rows:
            row = self._rows[self._loading]
            row.setText(0, self._loading)
            font = row.font(0)
            font.setItalic(False)
            row.setFont(0, font)
        self._loading = name
        if name and name in self._rows:
            row = self._rows[name]
            row.setText(0, f"{name} (computing...)")
            font = row.font(0)
            font.setItalic(True)
            row.setFont(0, font)

    def set_error(self, name: str, message: str) -> None:
        """Paint a failed replay's row red with the message on hover."""

        self.set_loading(None)
        row = self._rows.get(name)
        if row is None:
            return
        row.setForeground(0, QtGui.QBrush(QtGui.QColor(theme.active().down)))
        row.setToolTip(0, message)

    def clear_error(self, name: str) -> None:
        row = self._rows.get(name)
        spec = self._specs.get(name)
        if row is not None:
            row.setForeground(0, QtGui.QBrush())
            row.setToolTip(0, spec.thesis if spec is not None else "")

    def set_enabled_with_reason(self, on: bool, reason: str = "") -> None:
        """Disable the whole browser with the reason visible (off-GC state)."""

        self.tree.setEnabled(on)
        self.search.setEnabled(on)
        self.family_combo.setEnabled(on)
        self.reason.setText(reason)
        self.reason.setVisible(bool(reason) and not on)
