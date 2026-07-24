"""Launch the GC trade-simulation desktop app.

Requires the app dependencies:
    python -m pip install -r requirements.txt -r requirements-app.txt
    python scripts/run_trade_simulator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from PySide6 import QtWidgets

    from src.app.ui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
