"""Plain-language 'why it worked / failed' verdict for a simulated trade.

Turns the mechanical result (exit reason, R, MFE/MAE, the stop track) and the
optional registered feature context into a few sentences a newcomer can read.
Pure and testable; the forensics panel renders whatever this returns.
"""

from __future__ import annotations

import numpy as np

from .excursion import is_winner_on_the_hook


def _was_stop_raised(result) -> bool:
    """True if the exit stop sat better than the initial stop (trailing/breakeven)."""

    track = np.asarray(result.stop_track, dtype=float)
    if track.size == 0:
        return False
    entry = float(result.entry_price)
    init = (
        entry - result.initial_stop_points
        if result.direction > 0
        else entry + result.initial_stop_points
    )
    final = float(track[-1])
    return final > init + 1e-9 if result.direction > 0 else final < init - 1e-9


def verdict(result, ctx=None, hook_r: float = 1.0) -> list[str]:
    """A list of plain-language sentences explaining the trade's outcome."""

    r = result.gross_r
    held = result.holding_minutes
    lines: list[str] = []

    reason = result.exit_reason
    if reason == "target":
        lines.append(f"Hit the profit target for +{r:.2f} R in {held} minutes - a clean winner.")
    elif reason == "stop":
        if _was_stop_raised(result) and r >= 0:
            lines.append(
                f"A trailing/breakeven stop banked +{r:.2f} R after locking in from the "
                f"peak of +{result.mfe_r:.2f} R."
            )
        elif is_winner_on_the_hook(result, hook_r):
            lines.append(
                f"A winner on the hook: you were up +{result.mfe_r:.2f} R before it reversed "
                f"into the stop for {r:+.2f} R. A trailing stop or a breakeven move would "
                f"have kept some of it."
            )
        else:
            lines.append(f"Stopped out for {r:+.2f} R - the stop was hit before the target.")
    elif reason == "forced_1530":
        lines.append(
            f"Closed at the 15:30 New York session end for {r:+.2f} R - the session ran out "
            f"before the target was reached."
        )
    elif reason == "time_exit":
        lines.append(
            f"Hit the {held}-minute holding cap at {r:+.2f} R before reaching a stop or target."
        )
    elif reason == "boundary_exit":
        lines.append(f"Closed at a session/roll boundary for {r:+.2f} R.")
    else:
        lines.append(f"Ended at {r:+.2f} R ({reason}).")

    lines.append(
        f"Along the way it ran +{result.mfe_r:.2f} R in your favour and "
        f"-{result.mae_r:.2f} R against."
    )

    if ctx is not None:
        lines.extend(_context_lines(ctx))
    return lines


def _context_lines(ctx) -> list[str]:
    feats = ctx.features or {}
    out: list[str] = []

    eff = feats.get("efficiency_ratio_30")
    chop = feats.get("choppiness_14")
    if eff is not None and not np.isnan(eff) and chop is not None and not np.isnan(chop):
        if eff < 0.3 or chop > 60:
            out.append(
                f"Entered into a choppy, low-follow-through tape (efficiency {eff:.2f}, "
                f"choppiness {chop:.0f}) - momentum did not carry."
            )
        elif eff > 0.5:
            out.append(f"Entered into a clean, trending tape (efficiency {eff:.2f}).")

    dist = feats.get("distance_from_execution_session_vwap_atr")
    if dist is not None and not np.isnan(dist) and abs(dist) >= 1.5:
        side = "above" if dist > 0 else "below"
        out.append(
            f"Price was stretched {abs(dist):.1f} ATR {side} the session VWAP at entry - "
            f"an extended location that often mean-reverts."
        )

    rvol = feats.get("relative_volume_60")
    if rvol is not None and not np.isnan(rvol):
        if rvol >= 1.3:
            out.append(f"Volume was elevated ({rvol:.2f}x the hour's norm).")
        elif rvol <= 0.7:
            out.append(f"Volume was thin ({rvol:.2f}x the hour's norm) - moves are less reliable.")

    return out
