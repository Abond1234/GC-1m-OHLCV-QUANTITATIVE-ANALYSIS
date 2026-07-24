"""App-wide configuration and convenience re-exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .datalayer import paths
from .datalayer.partitions import EVAL_CAP_DATE, FINAL_TEST_START
from .sim.exit_config import ExitConfig, frozen_config

__all__ = [
    "AppConfig",
    "EVAL_CAP_DATE",
    "FINAL_TEST_START",
    "ExitConfig",
    "frozen_config",
]


@dataclass(frozen=True)
class AppConfig:
    """Dataset locations and the Development+Validation date cap."""

    eval_cap_date: pd.Timestamp = EVAL_CAP_DATE

    @property
    def root(self) -> Path:
        return paths.project_root()

    @property
    def research_bars(self) -> Path:
        return paths.research_bars_path()

    @property
    def feature_matrix(self) -> Path:
        return paths.feature_matrix_path()

    @property
    def forward_labels(self) -> Path:
        return paths.forward_labels_path()

    @property
    def signal_candidates(self) -> Path:
        return paths.signal_candidates_path()
