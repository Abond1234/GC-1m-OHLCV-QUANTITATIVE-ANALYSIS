"""Qt-free analysis helpers for the trade simulator.

Pure numpy/pandas compute that every visualization shares - placed-trade
sizing/recompute, within-trade excursions, what-if exit runs, exit-grid sweeps,
and plain-language explanations. Keeping these free of PySide6/pyqtgraph makes
them unit-testable headlessly and safe to call from worker threads.
"""
