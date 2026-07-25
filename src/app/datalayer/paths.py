"""Repository-relative dataset paths for the app (no absolute paths)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_data_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` (inclusive) that holds ``data/processed``."""

    for candidate in (start, *start.parents):
        if (candidate / "data" / "processed").is_dir():
            return candidate
    return None


def project_root() -> Path:
    """Repository root holding ``data/processed``.

    Source runs resolve it relative to this file. A packaged (PyInstaller) build
    has no such tree and does not bundle the data (per the data governance), so it
    discovers a checkout's data by walking up from the executable and then from the
    working directory - a build under a checkout's ``dist/`` therefore finds the
    repo's ``data/`` with no configuration. ``GC_PROJECT_ROOT`` overrides all of
    this, letting a build placed anywhere point at an explicit checkout.
    """

    override = os.environ.get("GC_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for start in (exe_dir, Path.cwd().resolve()):
            found = _find_data_root(start)
            if found is not None:
                return found
        return exe_dir
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
