"""Package the GC Trade Simulator into a standalone build with PyInstaller.

    python -m pip install -r requirements.txt -r requirements-app.txt
    python scripts/build_app.py

Produces ``dist/GCTradeSimulator/`` (a one-directory build). The executable does
NOT bundle the multi-hundred-MB parquet data (that stays in the repo per the data
governance); run it with the repo's ``data/processed`` reachable, either by
launching from the repo root or by setting ``GC_PROJECT_ROOT`` to the checkout:

    set GC_PROJECT_ROOT=C:\\path\\to\\project-1
    dist\\GCTradeSimulator\\GCTradeSimulator.exe

``dist/``, ``build/``, and the generated ``*.spec`` are git-ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRY = PROJECT_ROOT / "scripts" / "run_trade_simulator.py"


def _render_icon() -> Path:
    """Render the code-drawn Gold Quant mark to a .ico for the executable.

    Generated into the git-ignored ``build/`` directory at build time - the mark's
    single source of truth stays ``src/app/ui/brand.py``; no binary is committed.
    """

    import os

    from PySide6 import QtWidgets

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.app.ui import brand

    # Use the native platform so real fonts render into the icon; the offscreen
    # platform substitutes box glyphs for text. Fall back to offscreen only when
    # there is no window station at all (a headless builder).
    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except (RuntimeError, SystemError):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication([])
    icon_path = PROJECT_ROOT / "build" / "app_icon.ico"
    brand.write_ico(icon_path)
    del app
    return icon_path


def main() -> None:
    try:
        import PyInstaller.__main__ as pyi
    except ImportError:
        raise SystemExit(
            "PyInstaller is not installed. Run:\n  python -m pip install -r requirements-app.txt"
        ) from None

    icon_path = _render_icon()
    args = [
        str(ENTRY),
        "--name=GCTradeSimulator",
        "--noconfirm",
        "--windowed",  # no console window
        f"--icon={icon_path}",
        "--paths",
        str(PROJECT_ROOT),  # so 'src' is importable during analysis
        "--collect-submodules=src.app",
        "--collect-submodules=src.statistical_research",
        "--distpath",
        str(PROJECT_ROOT / "dist"),
        "--workpath",
        str(PROJECT_ROOT / "build"),
        "--specpath",
        str(PROJECT_ROOT),
    ]
    print("Running PyInstaller:\n  " + "\n  ".join(args))
    pyi.run(args)
    print("\nBuild complete -> dist/GCTradeSimulator/")


if __name__ == "__main__":
    if not ENTRY.exists():
        raise SystemExit(f"entry point not found: {ENTRY}")
    sys.exit(main())
