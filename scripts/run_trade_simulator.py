"""Launch the GC trade-simulation desktop app.

Requires the app dependencies:
    python -m pip install -r requirements.txt -r requirements-app.txt
    python scripts/run_trade_simulator.py

Shows the Gold Quant splash immediately, loads the Development+Validation data on
a worker thread behind it, then fades the splash into the main window.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from PySide6 import QtWidgets

    from src.app.ui import brand, theme
    from src.app.ui.main_window import MainWindow
    from src.app.ui.splash import GoldSplash

    app = QtWidgets.QApplication(sys.argv)
    theme.apply(app, "light")
    app.setWindowIcon(brand.app_icon())

    splash = GoldSplash()
    splash.show()
    app.processEvents()  # paint the splash before the (brief) window construction

    window = MainWindow()

    def _reveal(_ok: bool) -> None:
        window.show()
        splash.finish_into(window)

    window.loadFinished.connect(_reveal)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
