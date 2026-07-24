"""Native trade-simulation and visualization app for the GC research dataset.

A desktop (PySide6 + finplot) application that charts the gold-futures data,
replays any catalog strategy's trades, and lets a user place trades with
FLEXIBLE (non-frozen) exits to understand why a trade worked or failed. It
reuses the verified research engine in-process for accuracy and never reads or
renders the locked Final-test partition.

The simulation and data layers (``src.app.sim`` and ``src.app.data``) are pure
numpy/pandas and carry no Qt dependency, so they are unit-testable without a GUI.
The UI layer (``src.app.ui``, ``src.app.workers``) requires PySide6/finplot,
which live in ``requirements-app.txt`` (installed on top of the pinned core).
"""
