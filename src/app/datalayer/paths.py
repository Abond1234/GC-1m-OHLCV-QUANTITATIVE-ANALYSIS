"""Repository-relative dataset paths for the app (no absolute paths)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    """Repository root holding ``data/processed``.

    Source runs resolve it relative to this file. A packaged (PyInstaller) build
    has no such tree, so it looks next to the executable; ``GC_PROJECT_ROOT``
    overrides both, letting a built app point at a checkout's data.
    """

    override = os.environ.get("GC_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def processed_dir() -> Path:
    return project_root() / "data" / "processed"


def statistical_research_dir() -> Path:
    return processed_dir() / "statistical_research"


def research_bars_path() -> Path:
    return processed_dir() / "research_bars_gc_mgc_1m.parquet"


def feature_matrix_path() -> Path:
    return statistical_research_dir() / "feature_matrix_gc.parquet"


def forward_labels_path() -> Path:
    return statistical_research_dir() / "forward_labels_gc.parquet"


def signal_candidates_path() -> Path:
    return statistical_research_dir() / "signal_candidates_gc.parquet"
