"""Bar-by-bar replay driven by a QTimer, with the future hidden.

Two modes share one transport:

* Trade replay (``load``): steps a marker and the stop/target lines from a
  trade's entry to its exit, so the failure (or win) unfolds in time.
* Day replay (``load_day``): no trade required - the reveal curtain drops at the
  start of the window and the tape plays out candle by candle, TradingView
  replay style.

Each tick only calls setData/setPos/setRegion on existing chart items, so
playback stays smooth.
"""

from __future__ import annotations

from PySide6 import QtCore


class ReplayAnimator(QtCore.QObject):
    """Drives ChartWidget.replay_frame across a bar range, optionally with a trade."""

    positionChanged = QtCore.Signal(int, int)  # (current_offset_min, total_min)
    stateChanged = QtCore.Signal(bool)  # True while playing

    def __init__(self, chart, parent=None):
        super().__init__(parent)
        self._chart = chart
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._interval = 120  # ms per bar
        self._result = None  # trade being replayed, or None for a day replay
        self._start: int | None = None  # global bar range being animated
        self._end: int | None = None
        self._t = 0

    def load(self, result) -> None:
        """Attach a trade and show its first frame (paused)."""

        self.stop()
        self._result = result
        self._start = int(result.entry_position)
        self._end = int(result.exit_position)
        self._t = self._start
        self._chart.start_replay(result)
        self._render()

    def load_day(self, start_global: int, end_global: int) -> None:
        """Replay a bare bar window - no trade, just the tape revealing."""

        self.stop()
        self._result = None
        self._start = int(start_global)
        self._end = int(end_global)
        self._t = self._start
        self._chart.start_replay(None, start_global=self._start)
        self._render()

    @property
    def active(self) -> bool:
        return self._start is not None

    def play(self) -> None:
        if self._start is None:
            return
        if self._t >= self._end:
            self._t = self._start  # replay from the start
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
        if self._start is not None:
            self._chart.stop_replay()
        self._result = None
        self._start = None
        self._end = None
        self.stateChanged.emit(False)

    # -- internals ---------------------------------------------------------
    def _tick(self) -> None:
        if self._start is None or self._t >= self._end:
            self.pause()
            return
        self._advance(1)

    def _advance(self, delta: int) -> None:
        if self._start is None:
            return
        self._t = max(self._start, min(self._end, self._t + delta))
        self._render()

    def _render(self) -> None:
        self._chart.replay_frame(self._result, self._t, start_global=self._start)
        self.positionChanged.emit(self._t - self._start, self._end - self._start)
