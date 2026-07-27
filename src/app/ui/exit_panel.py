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


class ExitPanel(QtWidgets.QWidget):
    """All ExitConfig fields as grouped controls; emits a debounced configChanged."""

    configChanged = QtCore.Signal(object)  # emits an ExitConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(140)
        self._debounce.timeout.connect(lambda: self.configChanged.emit(self.to_config()))
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
        stop_box, stop_form = self._group("Stop (your risk = 1 R)")
        self.stop_mode = self._combo(_STOP_MODES, _TIP["stop_mode"])
        self.stop_value = self._spin(0.1, 200.0, 1.5, 0.1, _TIP["stop_value"])
        stop_form.addRow("mode", self.stop_mode)
        stop_form.addRow("value", self.stop_value)
        layout.addWidget(stop_box)

        # Target.
        tgt_box, tgt_form = self._group("Target (your reward)")
        self.target_mode = self._combo(_TARGET_MODES, _TIP["target_mode"])
        self.target_value = self._spin(0.1, 200.0, 2.0, 0.1, _TIP["target_value"])
        tgt_form.addRow("mode", self.target_mode)
        tgt_form.addRow("value", self.target_value)
        layout.addWidget(tgt_box)

        # Trailing.
        trail_box, trail_form = self._group("Trailing stop")
        self.trail_check = QtWidgets.QCheckBox("enabled")
        self.trail_check.setToolTip(_TIP["trailing"])
        self.trail_mode = self._combo(_STOP_MODES, _TIP["trailing"])
        self.trail_value = self._spin(0.1, 200.0, 1.5, 0.1, _TIP["trailing_value"])
        trail_form.addRow(self.trail_check)
        trail_form.addRow("mode", self.trail_mode)
        trail_form.addRow("value", self.trail_value)
        layout.addWidget(trail_box)

        # Breakeven.
        be_box, be_form = self._group("Breakeven move")
        self.be_check = QtWidgets.QCheckBox("enabled")
        self.be_check.setToolTip(_TIP["breakeven"])
        self.be_trigger = self._spin(0.1, 10.0, 1.0, 0.1, _TIP["be_trigger"])
        self.be_offset = self._spin(0.0, 100.0, 0.0, 1.0, _TIP["be_offset"])
        be_form.addRow(self.be_check)
        be_form.addRow("trigger R", self.be_trigger)
        be_form.addRow("offset ticks", self.be_offset)
        layout.addWidget(be_box)

        # Time / session.
        time_box, time_form = self._group("Time and session")
        self.hold_spin = QtWidgets.QSpinBox()
        self.hold_spin.setRange(1, 1440)
        self.hold_spin.setValue(120)
        self.hold_spin.setToolTip(_TIP["hold"])
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

    def _group(self, title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setContentsMargins(10, 6, 10, 8)
        form.setSpacing(6)
        return box, form

    def _combo(self, items, tip: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.setToolTip(tip)
        return combo

    def _spin(self, lo, hi, val, step, tip) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(val)
        spin.setToolTip(tip)
        return spin

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
        self._debounce.start()

    def _sync_enabled(self, *_a) -> None:
        on_trail = self.trail_check.isChecked()
        self.trail_mode.setEnabled(on_trail)
        self.trail_value.setEnabled(on_trail)
        on_be = self.be_check.isChecked()
        self.be_trigger.setEnabled(on_be)
        self.be_offset.setEnabled(on_be)

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
