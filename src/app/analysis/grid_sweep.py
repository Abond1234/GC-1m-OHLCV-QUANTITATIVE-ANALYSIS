"""Exit-grid sweep: how the outcome changes across a grid of exit rules.

For one entry, sweep a 2D grid of stop-multiple x target-R (or trailing) and
record the realised R, whether it won, and the exit reason for every cell - the
'exit structure' axis the research left untested, made interactive. Every cell
goes through the verified single-flex engine.

This is exploratory. Sweeping K = rows x cols exits against the same Dev/Val bars
means the best-looking cell is partly luck, not a validated edge; the UI shows the
trial count K so that is never forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.exit_config import ExitConfig
from ..sim.flex_exit import single_flex_exit
from .placed_trade import resolve_one


@dataclass
class GridResult:
    """Outcome surfaces over a stop x target grid for a single entry."""

    stop_mults: np.ndarray  # row axis values (ATR multiples)
    target_rs: np.ndarray  # column axis values (R multiples)
    gross_r: np.ndarray  # [rows, cols] realised R
    win: np.ndarray  # [rows, cols] bool, gross_r > 0
    reason_code: np.ndarray  # [rows, cols] int code into EXIT_REASONS

    @property
    def trials(self) -> int:
        return int(self.gross_r.size)

    @property
    def best_cell(self) -> tuple[int, int]:
        idx = int(np.nanargmax(self.gross_r))
        return int(idx // self.gross_r.shape[1]), int(idx % self.gross_r.shape[1])


EXIT_REASONS = ("stop", "target", "forced_1530", "time_exit", "boundary_exit", "end_of_data")
_REASON_CODE = {r: i for i, r in enumerate(EXIT_REASONS)}


def sweep_entry(
    entry_position: int,
    direction: int,
    base_cfg: ExitConfig,
    atr20: np.ndarray,
    arrays: dict,
    stop_mults: np.ndarray,
    target_rs: np.ndarray,
) -> GridResult:
    """Sweep stop-multiple x target-R for one entry and return outcome surfaces."""

    stop_mults = np.asarray(stop_mults, dtype=float)
    target_rs = np.asarray(target_rs, dtype=float)
    rows, cols = len(stop_mults), len(target_rs)
    gross = np.full((rows, cols), np.nan)
    win = np.zeros((rows, cols), dtype=bool)
    reason = np.zeros((rows, cols), dtype=int)

    atr = float(atr20[max(int(entry_position) - 1, 0)])
    # Trailing is carried from the base config (its distance re-sized off ATR).
    _s, _t, trail_pts = resolve_one(entry_position, atr20, base_cfg)

    for i, sm in enumerate(stop_mults):
        stop_pts = sm * atr
        for j, tr in enumerate(target_rs):
            target_pts = tr * stop_pts
            res = single_flex_exit(
                int(entry_position),
                int(direction),
                stop_pts,
                target_pts,
                base_cfg,
                arrays,
                trail_pts,
            )
            gross[i, j] = res.gross_r
            win[i, j] = res.gross_r > 0
            reason[i, j] = _REASON_CODE.get(res.exit_reason, len(EXIT_REASONS) - 1)
    return GridResult(stop_mults, target_rs, gross, win, reason)


def default_axes() -> tuple[np.ndarray, np.ndarray]:
    """A sensible default grid: stop 0.5-3.0x ATR, target 0.5-4.0 R."""

    return np.round(np.arange(0.5, 3.01, 0.25), 3), np.round(np.arange(0.5, 4.01, 0.5), 3)
