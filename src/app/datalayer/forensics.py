"""Trade forensics: why did a trade work or fail.

For a given ``observation_id`` this joins two independently built research tables:
the forward-label excursions (how far the move went for and against each horizon,
when the peak/trough happened, whether it expanded) and the causal feature context
at entry (VWAP distance, volatility regime, trend quality, session position,
relative volume). Together they turn a red trade into an explanation - "you reached
+1.8R at minute 12 then gave it all back", "entry was 2 ATR extended from VWAP into
a choppy regime".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .paths import feature_matrix_path, forward_labels_path, project_root

_HORIZONS = (5, 15, 30, 60, 120, 180)
_LABEL_COLUMNS = ["observation_id", "entry_price", "decision_atr_20m"]
for _h in _HORIZONS:
    _LABEL_COLUMNS += [
        f"mfe_long_{_h}_atr",
        f"mae_long_{_h}_atr",
        f"mfe_short_{_h}_atr",
        f"mae_short_{_h}_atr",
        f"time_to_mfe_long_{_h}_minutes",
        f"time_to_mfe_short_{_h}_minutes",
        f"forward_return_{_h}_atr",
        f"future_range_{_h}_atr",
        f"expansion_label_{_h}",
    ]

_FEATURE_COLUMNS = [
    "observation_id",
    "distance_from_execution_session_vwap_atr",
    "distance_from_rolling_vwap_20_atr",
    "atr_ratio_5_20",
    "efficiency_ratio_30",
    "ols_r_squared_30",
    "choppiness_14",
    "session_range_position",
    "relative_volume_60",
    "signed_body_atr",
]


@dataclass
class ForensicsContext:
    observation_id: int
    excursions: dict = field(default_factory=dict)  # horizon -> {mfe/mae/return/range/expansion}
    features: dict = field(default_factory=dict)  # feature name -> value


class ForensicsService:
    """Point lookups into the forward-label and feature tables by observation."""

    def __init__(
        self, forward_labels: Path | None = None, feature_matrix: Path | None = None
    ) -> None:
        root = project_root()
        self._labels_path = forward_labels or forward_labels_path()
        self._features_path = feature_matrix or feature_matrix_path()
        self._root = root

    def available(self) -> bool:
        return self._labels_path.exists() and self._features_path.exists()

    def _read_row(self, path: Path, columns: list[str], observation_id: int) -> pd.Series | None:
        import pyarrow.parquet as pq

        table = pq.read_table(
            path, columns=columns, filters=[("observation_id", "==", int(observation_id))]
        ).to_pandas(ignore_metadata=True)
        if table.empty:
            return None
        return table.iloc[0]

    def context_for(self, observation_id: int) -> ForensicsContext:
        """Excursions and feature context for one observation (long-side framing)."""

        context = ForensicsContext(observation_id=int(observation_id))
        labels = self._read_row(self._labels_path, _LABEL_COLUMNS, observation_id)
        if labels is not None:
            for h in _HORIZONS:
                context.excursions[h] = {
                    "mfe_long_atr": float(labels[f"mfe_long_{h}_atr"]),
                    "mae_long_atr": float(labels[f"mae_long_{h}_atr"]),
                    "mfe_short_atr": float(labels[f"mfe_short_{h}_atr"]),
                    "mae_short_atr": float(labels[f"mae_short_{h}_atr"]),
                    "time_to_mfe_long_min": float(labels[f"time_to_mfe_long_{h}_minutes"]),
                    "forward_return_atr": float(labels[f"forward_return_{h}_atr"]),
                    "future_range_atr": float(labels[f"future_range_{h}_atr"]),
                    "expanded": bool(labels[f"expansion_label_{h}"]),
                }
        features = self._read_row(self._features_path, _FEATURE_COLUMNS, observation_id)
        if features is not None:
            context.features = {
                col: float(features[col]) for col in _FEATURE_COLUMNS if col != "observation_id"
            }
        return context
