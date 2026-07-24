"""A user-placed free-play trade and the pure logic to (re)simulate it.

A ``PlacedTrade`` bundles an entry with its exit rule and the resulting
``FlexTradeResult`` (which carries the per-bar stop/target tracks the chart
draws). Sizing reuses the verified ``resolve_levels`` and every fill goes through
the verified ``single_flex_exit``, so dragging a stop just re-runs the same engine
with explicit point levels - no separate maths to keep honest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..sim.exit_config import ExitConfig, resolve_levels
from ..sim.flex_exit import FlexTradeResult, single_flex_exit


@dataclass
class PlacedTrade:
    """One placed free-play trade: its entry, exit rule, sizing, and result."""

    id: int
    entry_position: int
    direction: int
    cfg: ExitConfig
    stop_points: float
    target_points: float
    trail_points: float
    result: FlexTradeResult
    color: str = ""
    observation_id: int | None = None


def resolve_one(
    entry_position: int, atr20: np.ndarray, cfg: ExitConfig
) -> tuple[float, float, float]:
    """Per-trade (stop, target, trail) point distances for a single entry."""

    stop, target, trail = resolve_levels(np.asarray([int(entry_position)]), atr20, cfg)
    return float(stop[0]), float(target[0]), float(trail[0])


def place_trade(
    trade_id: int,
    entry_position: int,
    direction: int,
    cfg: ExitConfig,
    atr20: np.ndarray,
    arrays: dict,
    *,
    observation_id: int | None = None,
    color: str = "",
) -> PlacedTrade:
    """Size a new trade from ``cfg`` and simulate it through the verified engine."""

    stop_pts, target_pts, trail_pts = resolve_one(entry_position, atr20, cfg)
    result = single_flex_exit(
        int(entry_position), int(direction), stop_pts, target_pts, cfg, arrays, trail_pts
    )
    return PlacedTrade(
        id=trade_id,
        entry_position=int(entry_position),
        direction=int(direction),
        cfg=cfg,
        stop_points=stop_pts,
        target_points=target_pts,
        trail_points=trail_pts,
        result=result,
        color=color,
        observation_id=observation_id,
    )


def recompute_levels(
    trade: PlacedTrade,
    arrays: dict,
    *,
    stop_points: float | None = None,
    target_points: float | None = None,
) -> PlacedTrade:
    """Return a copy of ``trade`` re-simulated with overridden stop/target points.

    Used by the draggable stop/target lines: the config still supplies trailing,
    breakeven, and the time exit, but the level itself is set explicitly.
    """

    stop_pts = trade.stop_points if stop_points is None else max(1e-9, float(stop_points))
    target_pts = trade.target_points if target_points is None else max(1e-9, float(target_points))
    result = single_flex_exit(
        trade.entry_position,
        trade.direction,
        stop_pts,
        target_pts,
        trade.cfg,
        arrays,
        trade.trail_points,
    )
    return replace(trade, stop_points=stop_pts, target_points=target_pts, result=result)


def recompute_config(
    trade: PlacedTrade, cfg: ExitConfig, atr20: np.ndarray, arrays: dict
) -> PlacedTrade:
    """Return a copy of ``trade`` re-sized and re-simulated under a new config."""

    stop_pts, target_pts, trail_pts = resolve_one(trade.entry_position, atr20, cfg)
    result = single_flex_exit(
        trade.entry_position, trade.direction, stop_pts, target_pts, cfg, arrays, trail_pts
    )
    return replace(
        trade,
        cfg=cfg,
        stop_points=stop_pts,
        target_points=target_pts,
        trail_points=trail_pts,
        result=result,
    )
