"""Strategy replay: turn any catalog/peer strategy into trades on the chart.

Wraps the verified strategy library and catalog. ``replay`` computes a strategy's
signals over the eligible universe, maps fired entries to bar positions, and runs
the flexible-exit simulator - the frozen contract by default (reproducing the
research result), or any custom :class:`ExitConfig`. Each trade carries its
``observation_id`` so the forensics panel can explain it, and its initial
stop/target so a click can redraw the exact path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistical_research.strategy_catalog import catalog_library, load_catalog_universe
from src.statistical_research.strategy_lab import StrategyLabConfig, StrategySpec, strategy_library

from ..sim.exit_config import ExitConfig, frozen_config, resolve_levels
from ..sim.flex_exit import simulate_flex
from .bar_store import BarStore
from .paths import project_root, research_bars_path

_EVAL_PARTITIONS = {"Development", "Validation"}


class StrategyReplayService:
    """List strategies and replay their trades over the Dev/Val universe."""

    def __init__(self, universe: pd.DataFrame, bars: BarStore) -> None:
        observed = set(universe["research_partition"].astype(str).unique())
        if not observed.issubset(_EVAL_PARTITIONS):
            raise ValueError(f"replay universe contains locked partitions: {observed}")
        self.universe = universe.reset_index(drop=True)
        self.bars = bars
        self._cfg = StrategyLabConfig()
        self._entry_positions = pd.Index(bars.ts).get_indexer(self.universe["entry_timestamp_utc"])

    @classmethod
    def load(cls, root=None, *, date_floor=None) -> StrategyReplayService:
        root = root or project_root()
        universe = load_catalog_universe(root)
        bars = BarStore.load(research_bars_path(), date_floor=date_floor)
        return cls(universe, bars)

    def list_strategies(self) -> list[StrategySpec]:
        specs = list(strategy_library())
        seen = {s.name for s in specs}
        kept, _dropped = catalog_library(self.universe)
        specs.extend(s for s in kept if s.name not in seen)
        return specs

    def replay(self, spec: StrategySpec, exit_cfg: ExitConfig | None = None) -> pd.DataFrame:
        """Trade log for a strategy; frozen contract unless an ExitConfig is given."""

        directions = spec.signal(self.universe, self._cfg)
        fired = (directions != 0) & (self._entry_positions >= 0)
        idx = np.nonzero(fired)[0]
        entry_positions = self._entry_positions[idx]
        dirs = directions[idx].astype(np.int8)

        if exit_cfg is None:
            exit_cfg = frozen_config()
            stop = self.universe["stop_points"].to_numpy(dtype=np.float64)[idx]
            target = self.universe["target_points"].to_numpy(dtype=np.float64)[idx]
            trail = None
        else:
            stop, target, trail = resolve_levels(entry_positions, self.bars.atr20, exit_cfg)

        meta = self.universe.iloc[idx][
            ["research_partition", "entry_session", "trade_date_ny"]
        ].reset_index(drop=True)
        log = simulate_flex(
            entry_positions, dirs, stop, target, exit_cfg, meta, self.bars.bar_arrays(), trail
        )
        if log.empty:
            return log

        # Attach observation_id + initial levels so a clicked trade can be redrawn.
        lookup = pd.DataFrame(
            {
                "entry_position": entry_positions,
                "observation_id": self.universe["observation_id"].to_numpy()[idx],
                "initial_stop_points": stop,
                "initial_target_points": target,
                "trail_points": (trail if trail is not None else np.zeros(len(idx))),
            }
        )
        log = log.merge(lookup, on="entry_position", how="left")
        log.insert(0, "strategy", spec.name)
        return log
