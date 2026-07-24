"""Bar-by-bar replay of a trade, driven by a QTimer.

Steps a marker and the stop/target lines from entry to exit so the failure (or
win) unfolds in time - you watch price approach the stop, a trailing stop ratchet,
the target get hit. It only calls setData/setPos on existing chart items each tick,
so playback stays smooth.
"""

from __future__ import annotations

from PySide6 import QtCore


class ReplayAnimator(QtCore.QObject):
    """Drives ChartWidget.replay_frame between a trade's entry and exit."""

    positionChanged = QtCore.Signal(int, int)  # (current_offset_min, total_min)
    stateChanged = QtCore.Signal(bool)  # True while playing

    def __init__(self, chart, parent=None):
        super().__init__(parent)
        self._chart = chart
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._interval = 120  # ms per bar
        self._result = None
        self._t = 0

    def load(self, result) -> None:
        """Attach a trade and show its first frame (paused)."""

        self.stop()
        self._result = result
        self._t = int(result.entry_position)
        self._chart.start_replay(result)
        self._render()

    @property
    def active(self) -> bool:
        return self._result is not None

    def play(self) -> None:
        if self._result is None:
            return
        if self._t >= self._result.exit_position:
            self._t = int(self._result.entry_position)  # replay from the start
        self._timer.start(self._interval)
        self.stateChanged.emit(True)

    def pause(self) -> None:
        self._timer.stop()
        self.stateChanged.emit(False)

    def toggle(self) -> None:
        self.pause() if self._timer.isActive() else self.play()

    def step(self, delta: int = 1) -> None:
        self.pause()
        self._advance(delta)

    def set_interval(self, ms: int) -> None:
        self._interval = max(10, int(ms))
        if self._timer.isActive():
            self._timer.start(self._interval)

    def stop(self) -> None:
        self._timer.stop()
        if self._result is not None:
            self._chart.stop_replay()
        self._result = None
        self.stateChanged.emit(False)

    # -- internals ---------------------------------------------------------
    def _tick(self) -> None:
        if self._result is None or self._t >= self._result.exit_position:
            self.pause()
            return
        self._advance(1)

    def _advance(self, delta: int) -> None:
        if self._result is None:
            return
        start = int(self._result.entry_position)
        end = int(self._result.exit_position)
        self._t = max(start, min(end, self._t + delta))
        self._render()

    def _render(self) -> None:
        self._chart.replay_frame(self._result, self._t)
        start = int(self._result.entry_position)
        end = int(self._result.exit_position)
        self.positionChanged.emit(self._t - start, end - start)
