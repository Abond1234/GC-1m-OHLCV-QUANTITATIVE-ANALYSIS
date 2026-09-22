"""Indicators workspace: a searchable Available list and a compact Active list.

Replaces the old permanent checkbox cards. Studies activate from the searchable
list (double-click or the add button); each active study gets parameter,
style (colour), hide and remove actions. Only active studies are computed and
rendered - the panel just describes the active set, the main window evaluates
it against the displayed window and hands the series to the chart.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..analysis.indicators import INDICATORS, IndicatorInstance
from . import theme

_STUDY_HELP = {
    "sma": "Trend baseline on the displayed timeframe.",
    "ema": "Responsive trend baseline on the displayed timeframe.",
    "bollinger": "Volatility envelope around a rolling mean.",
    "atr": "Displayed-bar range and stop-distance context.",
    "rsi": "Momentum context with 30/70 reference levels.",
    "rvol": "Rolling variability of displayed-bar returns.",
    "obv": "Cumulative volume signed by price direction.",
    "vma": "Moving average of the optional volume pane.",
    "vwap_day": "Research-day VWAP from the verified 1-minute series.",
    "vwap_session": "Execution-session VWAP from the verified 1-minute series.",
}

# Default colour cycle for new studies, resolved from the active palette.
_COLOR_ATTRS = ("vwap_rolling", "vwap_day", "vwap_session", "gold", "hook", "up", "down")


class _ActiveRow(QtWidgets.QFrame):
    """One active study row: label, parameters, colour, hide and remove."""

    def __init__(self, panel: IndicatorsPanel, inst: IndicatorInstance):
        super().__init__()
        self._panel = panel
        self._inst = inst
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 4, 2)
        lay.setSpacing(4)
        self.label = QtWidgets.QLabel(inst.display_label())
        self.label.setStyleSheet(f"color:{inst.color}")
        lay.addWidget(self.label, 1)
        self._buttons: dict[str, QtWidgets.QToolButton] = {}
        for key, text, tip in (
            ("params", "=", "Edit this study's parameters."),
            ("style", "#", "Pick this study's colour."),
            ("hide", "-", "Hide/show this study without removing it."),
            ("remove", "x", "Remove this study."),
        ):
            btn = QtWidgets.QToolButton()
            btn.setText(text)
            btn.setToolTip(tip)
            btn.setAutoRaise(True)
            btn.setFixedSize(20, 20)
            lay.addWidget(btn)
            self._buttons[key] = btn
        if not inst.definition.params:
            self._buttons["params"].setEnabled(False)
            self._buttons["params"].setToolTip("This study has no parameters.")
        self._buttons["params"].clicked.connect(self._edit_params)
        self._buttons["style"].clicked.connect(self._pick_color)
        self._buttons["hide"].clicked.connect(lambda: panel.toggle_instance(inst.id))
        self._buttons["remove"].clicked.connect(lambda: panel.remove_instance(inst.id))
        self.refresh()

    def refresh(self) -> None:
        inst = self._inst
        p = theme.active()
        self.label.setText(inst.display_label())
        self.label.setStyleSheet(f"color:{inst.color if inst.visible else p.text_faint}")
        self._buttons["hide"].setText("-" if inst.visible else "+")

    def _edit_params(self) -> None:
        """Inline parameter editor in a popup menu; edits apply live."""

        menu = QtWidgets.QMenu(self)
        holder = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(holder)
        form.setContentsMargins(8, 6, 8, 6)
        for spec in self._inst.definition.params:
            spin = QtWidgets.QSpinBox() if spec.integer else QtWidgets.QDoubleSpinBox()
            spin.setRange(int(spec.lo) if spec.integer else spec.lo, spec.hi)
            if not spec.integer:
                spin.setSingleStep(spec.step)
                spin.setDecimals(2)
            spin.setValue(self._inst.params.get(spec.name, spec.default))
            spin.setAccelerated(True)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(
                lambda value, name=spec.name: self._panel.set_param(self._inst.id, name, value)
            )
            form.addRow(spec.label, spin)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(holder)
        menu.addAction(action)
        menu.exec(self._buttons["params"].mapToGlobal(QtCore.QPoint(0, 20)))

    def _pick_color(self) -> None:
        from PySide6 import QtGui

        colour = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self._inst.color), self, "Study colour"
        )
        if colour.isValid():
            self._panel.set_color(self._inst.id, colour.name())


class IndicatorsPanel(QtWidgets.QWidget):
    """The workspace widget owning the active-study set."""

    changed = QtCore.Signal()  # active set, parameters, or visibility changed
    volumeToggled = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.instances: list[IndicatorInstance] = []
        self._next_id = 1
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        title = QtWidgets.QLabel("Chart indicators")
        title.setProperty("role", "sectionTitle")
        lay.addWidget(title)
        note = QtWidgets.QLabel(
            "Add only the visual context you need. These studies never change "
            "strategy entries, fills, or simulator results."
        )
        note.setWordWrap(True)
        note.setProperty("role", "caption")
        lay.addWidget(note)

        self.volume_check = QtWidgets.QCheckBox("Show volume pane")
        self.volume_check.setChecked(False)
        self.volume_check.setToolTip(
            "Display exchange volume below price. It is hidden when the app opens."
        )
        self.volume_check.toggled.connect(self.volumeToggled.emit)
        lay.addWidget(self.volume_check)

        available_label = QtWidgets.QLabel("Available studies")
        available_label.setProperty("role", "caption")
        lay.addWidget(available_label)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search indicators...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_available)
        lay.addWidget(self.search)

        self.available = QtWidgets.QListWidget()
        self.available.setToolTip("Double-click (or press Add) to activate a study.")
        for key, definition in INDICATORS.items():
            item = QtWidgets.QListWidgetItem(definition.label)
            item.setData(QtCore.Qt.UserRole, key)
            item.setToolTip(_STUDY_HELP.get(key, "Chart-only display study."))
            self.available.addItem(item)
        self.available.itemDoubleClicked.connect(
            lambda item: self.add_study(item.data(QtCore.Qt.UserRole))
        )
        lay.addWidget(self.available, 2)

        add_btn = QtWidgets.QPushButton("Add study")
        add_btn.clicked.connect(self._add_selected)
        lay.addWidget(add_btn)

        active_label = QtWidgets.QLabel("Active studies")
        active_label.setProperty("role", "caption")
        lay.addWidget(active_label)
        self.active_list = QtWidgets.QVBoxLayout()
        self.active_list.setSpacing(2)
        active_holder = QtWidgets.QWidget()
        active_holder.setLayout(self.active_list)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(active_holder)
        lay.addWidget(scroll, 2)
        self.empty_hint = QtWidgets.QLabel("No active studies. Add one from the list above.")
        self.empty_hint.setProperty("role", "caption")
        self.empty_hint.setWordWrap(True)
        self.active_list.addWidget(self.empty_hint)
        self.active_list.addStretch(1)
        self._rows: dict[int, _ActiveRow] = {}

    # -- available-list behaviour ------------------------------------------
    def _filter_available(self, text: str) -> None:
        needle = text.strip().lower()
        for i in range(self.available.count()):
            item = self.available.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _add_selected(self) -> None:
        item = self.available.currentItem()
        if item is None:  # nothing chosen: first visible row is the search hit
            for i in range(self.available.count()):
                if not self.available.item(i).isHidden():
                    item = self.available.item(i)
                    break
        if item is not None:
            self.add_study(item.data(QtCore.Qt.UserRole))

    # -- active-set mutations ----------------------------------------------
    def add_study(self, key: str) -> IndicatorInstance:
        definition = INDICATORS[key]
        palette = theme.active()
        colour = getattr(palette, _COLOR_ATTRS[(self._next_id - 1) % len(_COLOR_ATTRS)])
        inst = IndicatorInstance(
            id=self._next_id,
            key=key,
            params={p.name: p.default for p in definition.params},
            color=colour,
        )
        self._next_id += 1
        self.instances.append(inst)
        row = _ActiveRow(self, inst)
        self._rows[inst.id] = row
        self.active_list.insertWidget(self.active_list.count() - 2, row)
        self.empty_hint.setVisible(False)
        self.changed.emit()
        return inst

    def _instance(self, inst_id: int) -> IndicatorInstance | None:
        return next((i for i in self.instances if i.id == inst_id), None)

    def remove_instance(self, inst_id: int) -> None:
        inst = self._instance(inst_id)
        if inst is None:
            return
        self.instances.remove(inst)
        row = self._rows.pop(inst_id, None)
        if row is not None:
            self.active_list.removeWidget(row)
            row.deleteLater()
        self.empty_hint.setVisible(not self.instances)
        self.changed.emit()

    def toggle_instance(self, inst_id: int) -> None:
        inst = self._instance(inst_id)
        if inst is None:
            return
        inst.visible = not inst.visible
        row = self._rows.get(inst_id)
        if row is not None:
            row.refresh()
        self.changed.emit()

    def set_param(self, inst_id: int, name: str, value) -> None:
        inst = self._instance(inst_id)
        if inst is None:
            return
        inst.params[name] = float(value)
        row = self._rows.get(inst_id)
        if row is not None:
            row.refresh()
        self.changed.emit()

    def set_color(self, inst_id: int, color: str) -> None:
        inst = self._instance(inst_id)
        if inst is None:
            return
        inst.color = color
        row = self._rows.get(inst_id)
        if row is not None:
            row.refresh()
        self.changed.emit()

    # -- consumption by the main window --------------------------------------
    def active_instances(self) -> list[IndicatorInstance]:
        return list(self.instances)

    def chip_rows(self) -> list[dict]:
        return [
            {"id": i.id, "label": i.display_label(), "color": i.color, "visible": i.visible}
            for i in self.instances
        ]

    # -- session persistence -------------------------------------------------
    def to_dicts(self) -> list[dict]:
        return [
            {
                "key": i.key,
                "params": dict(i.params),
                "color": i.color,
                "visible": bool(i.visible),
            }
            for i in self.instances
        ]

    def load_dicts(self, specs: list[dict]) -> None:
        for inst_id in [i.id for i in self.instances]:
            self.remove_instance(inst_id)
        for spec in specs:
            if spec.get("key") not in INDICATORS:
                continue  # a study this build does not know is skipped, not guessed
            inst = self.add_study(spec["key"])
            inst.params.update({k: float(v) for k, v in spec.get("params", {}).items()})
            inst.color = spec.get("color", inst.color)
            inst.visible = bool(spec.get("visible", True))
            self._rows[inst.id].refresh()
        self.changed.emit()
