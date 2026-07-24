"""QThreadPool task wrapper so data loads and replays run off the UI thread.

Workers only ever touch numpy/pandas/dataclasses and return plain results; the
main window builds all pyqtgraph items on the GUI thread when the ``finished``
signal fires. This keeps pan/zoom smooth while a full-universe replay runs.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable

from PySide6 import QtCore


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)


class Task(QtCore.QRunnable):
    """Run ``fn(*args, **kwargs)`` on the thread pool and emit its result."""

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception:  # noqa: BLE001 - surface any load/replay failure to the UI
            self.signals.error.emit(traceback.format_exc())
            return
        self.signals.finished.emit(result)
