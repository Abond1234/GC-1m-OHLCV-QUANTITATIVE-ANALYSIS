"""Exit-configuration panel: every ExitConfig field, in one dockable place.

The research notebooks fix the exit (1.5x ATR stop, 2R target, 120-min cap). This
app deliberately unfreezes all of it so a trade can be explored creatively. Each
control maps to one ExitConfig field; ``to_config`` assembles them and a debounced
``configChanged`` lets the chart recompute the live trade as you scrub. Tooltips
define the jargon because the user is new to markets.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..sim.exit_config import FORCED_EXIT_MINUTE_NY, ExitConfig, frozen_config

_STOP_MODES = ("atr", "ticks", "points")
_TARGET_MODES = ("r", "atr", "ticks", "points")

_TIP = {
    "stop_mode": "How the stop distance is measured. ATR = a multiple of the average\n"
    "bar range; ticks = fixed 0.1-point steps; points = raw price.",
    "stop_value": "Size of the stop. This is your risk, '1 R'. A 1.5x ATR stop risks\n"
    "1.5 average-bar-ranges before the trade is cut.",
    "target_mode": "How the profit target is measured. R = a multiple of the stop\n"
    "(risk); the others mirror the stop modes.",
    "target_value": "Size of the target. 2 R means you aim to make twice what you risk.",
    "trailing": "A stop that follows price up (long) once you're in profit, locking\n"
    "in gains instead of a fixed stop.",
    "trailing_value": "How far the trailing stop sits behind the best price reached.",
    "breakeven": "Move the stop to your entry once the trade is far enough in profit,\n"
    "so a winner cannot turn into a loser.",
    "be_trigger": "Favourable move (in R) that arms the breakeven stop.",
    "be_offset": "Where the stop parks when armed, in ticks from entry (0 = exactly\n"
    "breakeven; positive locks in a little profit).",
    "hold": "Maximum minutes to hold before the trade is closed at market.",
    "forced": "Close any open trade at 15:30 New York (the research session end).",
    "stop_first": "If a bar touches both stop and target, assume the stop hit first\n"
    "(the conservative research convention).",
}


class _ClampDoubleSpin(QtWidgets.QDoubleSpinBox):
    """A spinbox whose out-of-range typed values clamp instead of reverting.

    Stock Qt keeps out-of-range text "Intermediate" and silently restores the
    previous value on commit - exactly the "control ignores input" failure the
    UI review flags. Clamping honours the user's intent and the field's rules.
    """

    def fixup(self, text: str) -> str:  # noqa: N802 - Qt override
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            return text
        clamped = min(max(value, self.minimum()), self.maximum())
        return f"{clamped:.{self.decimals()}f}"


class _ClampIntSpin(QtWidgets.QSpinBox):
    """Integer variant of :class:`_ClampDoubleSpin`."""

    def fixup(self, text: str) -> str:  # noqa: N802 - Qt override
        try:
            value = int(float(text.replace(",", ".")))
        except ValueError:
            return text
        return str(min(max(value, self.minimum()), self.maximum()))


class ExitPanel(QtWidgets.QWidget):
    """All ExitConfig fields as grouped controls; emits a debounced configChanged."""

    configChanged = QtCore.Signal(object)  # emits an ExitConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(140)
        self._debounce.timeout.connect(lambda: self.configChanged.emit(self.to_config()))
        self._warn_labels: dict[str, QtWidgets.QLabel] = {}
        self._build()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Presets, in a 2x2 grid so none clips when the dock sits at its
        # minimum width on a narrow display.
        presets = QtWidgets.QGridLayout()
        presets.setHorizontalSpacing(6)
        presets.setVerticalSpacing(6)
        for i, (label, factory) in enumerate(
            (
                ("Frozen contract", frozen_config),
                ("Wider stop", self._preset_wider),
                ("Add trailing", self._preset_trailing),
                ("Longer cap", self._preset_longer),
            )
        ):
            btn = QtWidgets.QPushButton(label)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            btn.clicked.connect(lambda _c=False, f=factory: self.from_config(f()))
            presets.addWidget(btn, i // 2, i % 2)
        layout.addLayout(presets)

        # Stop.
        stop_box, stop_form = self._group("Stop (your risk = 1 R)", "stop")
        self.stop_mode = self._combo(_STOP_MODES, _TIP["stop_mode"])
        self.stop_value = self._spin(0.1, 200.0, 1.5, 0.1, _TIP["stop_value"])
        stop_form.addRow("mode", self.stop_mode)
        stop_form.addRow("value", self.stop_value)
        layout.addWidget(stop_box)

        # Target.
        tgt_box, tgt_form = self._group("Target (your reward)", "target")
        self.target_mode = self._combo(_TARGET_MODES, _TIP["target_mode"])
        self.target_value = self._spin(0.1, 200.0, 2.0, 0.1, _TIP["target_value"])
        tgt_form.addRow("mode", self.target_mode)
        tgt_form.addRow("value", self.target_value)
        layout.addWidget(tgt_box)

        # Trailing.
        trail_box, trail_form = self._group("Trailing stop", "trail")
        self.trail_check = QtWidgets.QCheckBox("enabled")
        self.trail_check.setToolTip(_TIP["trailing"])
        self.trail_mode = self._combo(_STOP_MODES, _TIP["trailing"])
        self.trail_value = self._spin(0.1, 200.0, 1.5, 0.1, _TIP["trailing_value"])
        trail_form.addRow(self.trail_check)
        trail_form.addRow("mode", self.trail_mode)
        trail_form.addRow("value", self.trail_value)
        layout.addWidget(trail_box)
        self._trail_form = trail_form

        # Breakeven.
        be_box, be_form = self._group("Breakeven move", "be")
        self.be_check = QtWidgets.QCheckBox("enabled")
        self.be_check.setToolTip(_TIP["breakeven"])
        self.be_trigger = self._spin(0.1, 10.0, 1.0, 0.1, _TIP["be_trigger"])
        self.be_offset = self._spin(0.0, 100.0, 0.0, 1.0, _TIP["be_offset"], decimals=0)
        be_form.addRow(self.be_check)
        be_form.addRow("trigger R", self.be_trigger)
        be_form.addRow("offset ticks", self.be_offset)
        layout.addWidget(be_box)
        self._be_form = be_form

        # Time / session.
        time_box, time_form = self._group("Time and session", "time")
        self.hold_spin = _ClampIntSpin()
        self.hold_spin.setRange(1, 1440)
        self.hold_spin.setValue(120)
        self.hold_spin.setToolTip(_TIP["hold"])
        self.hold_spin.setAccelerated(True)
        self._attach_range_feedback(self.hold_spin)
        self.forced_check = QtWidgets.QCheckBox("force flat 15:30 NY")
        self.forced_check.setChecked(True)
        self.forced_check.setToolTip(_TIP["forced"])
        self.stop_first_check = QtWidgets.QCheckBox("stop wins ties (conservative)")
        self.stop_first_check.setChecked(True)
        self.stop_first_check.setToolTip(_TIP["stop_first"])
        time_form.addRow("max hold (min)", self.hold_spin)
        time_form.addRow(self.forced_check)
        time_form.addRow(self.stop_first_check)
        layout.addWidget(time_box)
        layout.addStretch(1)

        self._wire_signals()

    def _group(self, title: str, key: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setContentsMargins(10, 6, 10, 8)
        form.setSpacing(6)
        # Every group carries an inline warning slot, so invalid or
        # self-defeating input produces feedback next to the field, never a
        # silently-ignored keystroke.
        warn = QtWidgets.QLabel("")
        warn.setProperty("role", "warn")
        warn.setWordWrap(True)
        warn.setVisible(False)
        form.addRow(warn)
        self._warn_labels[key] = warn
        return box, form

    def _set_warning(self, key: str, message: str) -> None:
        label = self._warn_labels[key]
        label.setText(message)
        label.setVisible(bool(message))

    def _combo(self, items, tip: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.setToolTip(tip)
        return combo

    def _spin(self, lo, hi, val, step, tip, decimals: int = 2) -> QtWidgets.QDoubleSpinBox:
        spin = _ClampDoubleSpin()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(val)
        spin.setToolTip(tip)
        spin.setAccelerated(True)  # press-and-hold steppers ramp up
        self._attach_range_feedback(spin)
        return spin

    def _attach_range_feedback(self, spin: QtWidgets.QDoubleSpinBox) -> None:
        """Typed values outside the range flag the field instead of vanishing.

        Qt clamps out-of-range text silently on commit; here the field turns
        invalid-red the moment the typed number leaves ``[min, max]``, with the
        allowed range shown inline, and clears once the value is legal again.
        """

        def _on_text(text: str, _spin=spin) -> None:
            try:
                value = float(text.replace(",", "."))
            except ValueError:
                return  # incomplete input ("", "-", "1."): let the user type
            bad = value < _spin.minimum() or value > _spin.maximum()
            self._mark_invalid(_spin, bad)

        spin.lineEdit().textEdited.connect(_on_text)
        spin.editingFinished.connect(lambda _spin=spin: self._mark_invalid(_spin, False))

    def _mark_invalid(self, spin: QtWidgets.QDoubleSpinBox, bad: bool) -> None:
        if bool(spin.property("invalid")) == bad:
            return
        spin.setProperty("invalid", bad)
        spin.style().unpolish(spin)
        spin.style().polish(spin)
        if bad:
            spin.setProperty("base_tip", spin.toolTip())
            spin.setToolTip(
                f"Allowed range: {spin.minimum():g} to {spin.maximum():g}. "
                "The value will clamp when you leave the field."
            )
        elif spin.property("base_tip"):
            spin.setToolTip(spin.property("base_tip"))

    def _wire_signals(self) -> None:
        for spin in (
            self.stop_value,
            self.target_value,
            self.trail_value,
            self.be_trigger,
            self.be_offset,
        ):
            spin.valueChanged.connect(self._queue)
        self.hold_spin.valueChanged.connect(self._queue)
        for combo in (self.stop_mode, self.target_mode, self.trail_mode):
            combo.currentTextChanged.connect(self._queue)
        for check in (
            self.trail_check,
            self.be_check,
            self.forced_check,
            self.stop_first_check,
        ):
            check.toggled.connect(self._queue)
        # Enable/disable dependent controls.
        self.trail_check.toggled.connect(self._sync_enabled)
        self.be_check.toggled.connect(self._sync_enabled)
        self._sync_enabled()

    def _queue(self, *_a) -> None:
        self._refresh_advisories()
        self._debounce.start()

    def _sync_enabled(self, *_a) -> None:
        on_trail = self.trail_check.isChecked()
        for widget in (self.trail_mode, self.trail_value):
            widget.setEnabled(on_trail)
            label = self._trail_form.labelForField(widget)
            if label is not None:  # dim the row label too: clearly off, not broken
                label.setEnabled(on_trail)
        on_be = self.be_check.isChecked()
        for widget in (self.be_trigger, self.be_offset):
            widget.setEnabled(on_be)
            label = self._be_form.labelForField(widget)
            if label is not None:
                label.setEnabled(on_be)
        self._refresh_advisories()

    def _refresh_advisories(self) -> None:
        """Cross-field sanity feedback: legal but self-defeating settings."""

        trail_warn = ""
        if (
            self.trail_check.isChecked()
            and self.trail_mode.currentText() == self.stop_mode.currentText()
            and self.trail_value.value() > self.stop_value.value()
        ):
            trail_warn = (
                "Trailing distance is wider than the initial stop - it cannot "
                "tighten anything until price has run far in your favour."
            )
        self._set_warning("trail", trail_warn)
        be_warn = ""
        if (
            self.be_check.isChecked()
            and self.target_mode.currentText() == "r"
            and self.be_trigger.value() >= self.target_value.value()
        ):
            be_warn = (
                "Breakeven triggers at or beyond the target - the trade will "
                "exit at the target before the stop ever moves."
            )
        self._set_warning("be", be_warn)

    # -- config <-> widgets ------------------------------------------------
    def to_config(self) -> ExitConfig:
        return ExitConfig(
            stop_mode=self.stop_mode.currentText(),
            stop_value=self.stop_value.value(),
            target_mode=self.target_mode.currentText(),
            target_value=self.target_value.value(),
            trailing_enabled=self.trail_check.isChecked(),
            trailing_mode=self.trail_mode.currentText(),
            trailing_value=self.trail_value.value(),
            breakeven_enabled=self.be_check.isChecked(),
            breakeven_trigger_r=self.be_trigger.value(),
            breakeven_offset_ticks=self.be_offset.value(),
            max_holding_minutes=self.hold_spin.value(),
            forced_exit_minute_ny=(
                FORCED_EXIT_MINUTE_NY if self.forced_check.isChecked() else None
            ),
            stop_first=self.stop_first_check.isChecked(),
        )

    def from_config(self, cfg: ExitConfig) -> None:
        """Load a config into the controls without firing a burst of signals."""

        self._debounce.stop()  # this immediate emission supersedes any queued edit
        blockers = [
            self.stop_mode,
            self.stop_value,
            self.target_mode,
            self.target_value,
            self.trail_check,
            self.trail_mode,
            self.trail_value,
            self.be_check,
            self.be_trigger,
            self.be_offset,
            self.hold_spin,
            self.forced_check,
            self.stop_first_check,
        ]
        for w in blockers:
            w.blockSignals(True)
        self.stop_mode.setCurrentText(cfg.stop_mode)
        self.stop_value.setValue(cfg.stop_value)
        self.target_mode.setCurrentText(cfg.target_mode)
        self.target_value.setValue(cfg.target_value)
        self.trail_check.setChecked(cfg.trailing_enabled)
        self.trail_mode.setCurrentText(cfg.trailing_mode)
        self.trail_value.setValue(cfg.trailing_value)
        self.be_check.setChecked(cfg.breakeven_enabled)
        self.be_trigger.setValue(cfg.breakeven_trigger_r)
        self.be_offset.setValue(cfg.breakeven_offset_ticks)
        self.hold_spin.setValue(cfg.max_holding_minutes)
        self.forced_check.setChecked(cfg.forced_exit_minute_ny is not None)
        self.stop_first_check.setChecked(cfg.stop_first)
        for w in blockers:
            w.blockSignals(False)
        self._sync_enabled()
        self.configChanged.emit(self.to_config())

    def _preset_wider(self) -> ExitConfig:
        return ExitConfig(stop_mode="atr", stop_value=2.5, target_mode="r", target_value=2.0)

    def _preset_trailing(self) -> ExitConfig:
        return ExitConfig(
            stop_mode="atr",
            stop_value=1.5,
            target_mode="r",
            target_value=3.0,
            trailing_enabled=True,
            trailing_mode="atr",
            trailing_value=1.5,
        )

    def _preset_longer(self) -> ExitConfig:
        return ExitConfig(
            stop_mode="atr",
            stop_value=1.5,
            target_mode="r",
            target_value=2.0,
            max_holding_minutes=360,
        )
