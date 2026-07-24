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


def main() -> None:
    try:
        import PyInstaller.__main__ as pyi
    except ImportError:
        raise SystemExit(
            "PyInstaller is not installed. Run:\n  python -m pip install -r requirements-app.txt"
        ) from None

    args = [
        str(ENTRY),
        "--name=GCTradeSimulator",
        "--noconfirm",
        "--windowed",  # no console window
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
