"""What-if exit overlays: the same entry under alternative exit rules.

Answers 'why did MY exit fail' by re-running one entry under a few alternative
ExitConfigs (a wider stop, a trailing stop, a longer cap, a tighter target) and
returning each outcome so the chart can overlay the alternative paths and a table
can compare them. Every run goes through the verified engine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..sim.exit_config import ExitConfig
from ..sim.flex_exit import FlexTradeResult, single_flex_exit
from .placed_trade import resolve_one


@dataclass
class WhatIfRun:
    label: str
    cfg: ExitConfig
    result: FlexTradeResult
    color: str

    @property
    def survived(self) -> bool:
        """Reached the profit target rather than being stopped or timed out."""

        return self.result.exit_reason == "target"


def default_variants(base: ExitConfig) -> list[tuple[str, ExitConfig]]:
    """A small, sensible set of alternatives derived from the current rule."""

    variants = [
        ("wider stop", replace(base, stop_value=base.stop_value * 1.5)),
        (
            "add trailing",
            replace(base, trailing_enabled=True, trailing_mode="atr", trailing_value=1.5),
        ),
        ("longer cap", replace(base, max_holding_minutes=base.max_holding_minutes * 2)),
        ("tighter target", replace(base, target_value=max(0.5, base.target_value * 0.6))),
    ]
    return variants


def run_whatifs(
    entry_position: int,
    direction: int,
    base_cfg: ExitConfig,
    atr20,
    arrays: dict,
    colors,
    variants=None,
) -> list[WhatIfRun]:
    """Run each alternative exit for one entry and return the outcomes."""

    variants = default_variants(base_cfg) if variants is None else variants
    runs: list[WhatIfRun] = []
    for i, (label, cfg) in enumerate(variants):
        stop_pts, target_pts, trail_pts = resolve_one(entry_position, atr20, cfg)
        result = single_flex_exit(
            int(entry_position), int(direction), stop_pts, target_pts, cfg, arrays, trail_pts
        )
        runs.append(WhatIfRun(label, cfg, result, colors[i % len(colors)]))
    return runs
