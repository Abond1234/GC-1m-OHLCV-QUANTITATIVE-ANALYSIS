"""Exit configuration for the flexible-exit simulator.

The frozen research contract sizes stops from the decision-bar ATR (1.5x, clamped)
with a 2R target, a 120-minute cap, and a 15:30 forced exit. This config lets the
app express that exactly (via ``frozen_config`` + explicit per-trade levels) and
also express creative exits the notebooks never allowed: arbitrary stop/target
sizing, a trailing stop, and a breakeven move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORCED_EXIT_MINUTE_NY = 930  # 15:30 New York
DEFAULT_MAX_HOLDING_MINUTES = 120
DEFAULT_TICK_SIZE = 0.10


@dataclass(frozen=True)
class ExitConfig:
    """A trade's exit rule. Static sizing fields are used only when explicit
    per-trade levels are not supplied (i.e. free-play); replay passes the frozen
    ``stop_points``/``target_points`` directly.
    """

    # Static sizing (free-play). ``r`` target means a multiple of the stop.
    stop_mode: str = "atr"  # "points" | "ticks" | "atr"
    stop_value: float = 1.5
    target_mode: str = "r"  # "points" | "ticks" | "atr" | "r"
    target_value: float = 2.0
    # Path-dependent exits.
    trailing_enabled: bool = False
    trailing_mode: str = "atr"  # "points" | "ticks" | "atr"
    trailing_value: float = 1.5
    breakeven_enabled: bool = False
    breakeven_trigger_r: float = 1.0
    breakeven_offset_ticks: float = 0.0
    # Session / time.
    max_holding_minutes: int = DEFAULT_MAX_HOLDING_MINUTES
    forced_exit_minute_ny: int | None = FORCED_EXIT_MINUTE_NY  # None disables 15:30
    respect_boundary: bool = True
    stop_first: bool = True  # conservative stop-before-target on an ambiguous bar
    tick_size: float = DEFAULT_TICK_SIZE

    @property
    def is_dynamic(self) -> bool:
        return self.trailing_enabled or self.breakeven_enabled


def frozen_config() -> ExitConfig:
    """The config that reproduces the frozen Section 10/11 exit contract exactly."""

    return ExitConfig(
        trailing_enabled=False,
        breakeven_enabled=False,
        max_holding_minutes=DEFAULT_MAX_HOLDING_MINUTES,
        forced_exit_minute_ny=FORCED_EXIT_MINUTE_NY,
        respect_boundary=True,
        stop_first=True,
        tick_size=DEFAULT_TICK_SIZE,
    )


def _distance_points(mode: str, value: float, atr_at_decision: float, tick_size: float) -> float:
    if mode == "points":
        return float(value)
    if mode == "ticks":
        return float(value) * tick_size
    if mode == "atr":
        return float(value) * float(atr_at_decision)
    raise ValueError(f"unknown sizing mode: {mode!r}")


def resolve_levels(
    entry_positions: np.ndarray, atr20: np.ndarray, exit_cfg: ExitConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trade (stop_points, target_points, trail_points) from the config.

    ATR sizing uses the decision bar (``atr20[start-1]``), point-in-time.
    """

    n = len(entry_positions)
    stop = np.empty(n, dtype=np.float64)
    target = np.empty(n, dtype=np.float64)
    trail = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = int(entry_positions[i])
        atr = float(atr20[max(start - 1, 0)])
        stop_pts = _distance_points(
            exit_cfg.stop_mode, exit_cfg.stop_value, atr, exit_cfg.tick_size
        )
        if exit_cfg.target_mode == "r":
            target_pts = exit_cfg.target_value * stop_pts
        else:
            target_pts = _distance_points(
                exit_cfg.target_mode, exit_cfg.target_value, atr, exit_cfg.tick_size
            )
        stop[i] = stop_pts
        target[i] = target_pts
        if exit_cfg.trailing_enabled:
            trail[i] = _distance_points(
                exit_cfg.trailing_mode, exit_cfg.trailing_value, atr, exit_cfg.tick_size
            )
    return stop, target, trail
