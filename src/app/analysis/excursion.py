"""Within-trade excursion geometry for the 'why it failed' visuals.

From a simulated trade and the bar highs/lows over its held window, build the
running favourable and adverse excursions (in R and in price), the peak/trough
points, and the time-to-peak. The chart draws these as ribbons and markers so a
'winner on the hook' - a trade that was well in profit before reversing into a
stop - is visible on the price path, not just described in text.

Everything is in R relative to the trade's initial stop (its risk), matching the
engine's convention; the realized peak/trough here equal the engine's
``mfe_r``/``mae_r`` (asserted in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Excursion:
    """Running favourable/adverse excursion of one trade over its held bars."""

    x: np.ndarray  # bar index (global) for each held bar
    fav_price: np.ndarray  # running best price in the trade's favour
    adv_price: np.ndarray  # running worst price against the trade
    fav_r: np.ndarray  # favourable excursion in R at each bar
    adv_r: np.ndarray  # adverse excursion in R at each bar
    entry_price: float
    mfe_r: float  # peak favourable (== result.mfe_r)
    mae_r: float  # peak adverse (== result.mae_r)
    peak_x: int  # bar index of the favourable peak
    trough_x: int  # bar index of the adverse peak
    peak_price: float
    trough_price: float
    time_to_peak_min: int  # bars from entry to the favourable peak


def compute_excursion(result, high: np.ndarray, low: np.ndarray) -> Excursion:
    """Build the running excursion for ``result`` from bar highs/lows."""

    start = int(result.entry_position)
    end = int(result.exit_position)
    is_long = result.direction > 0
    entry = float(result.entry_price)
    stop_pts = float(result.initial_stop_points)
    x = np.arange(start, end + 1)
    h = np.asarray(high[start : end + 1], dtype=float)
    low_arr = np.asarray(low[start : end + 1], dtype=float)

    if is_long:
        fav_price = np.maximum.accumulate(h)
        adv_price = np.minimum.accumulate(low_arr)
        fav_r = (fav_price - entry) / stop_pts
        adv_r = (entry - adv_price) / stop_pts
    else:
        fav_price = np.minimum.accumulate(low_arr)
        adv_price = np.maximum.accumulate(h)
        fav_r = (entry - fav_price) / stop_pts
        adv_r = (adv_price - entry) / stop_pts

    peak_i = int(np.argmax(fav_r))
    trough_i = int(np.argmax(adv_r))
    return Excursion(
        x=x,
        fav_price=fav_price,
        adv_price=adv_price,
        fav_r=fav_r,
        adv_r=adv_r,
        entry_price=entry,
        mfe_r=float(fav_r[peak_i]),
        mae_r=float(adv_r[trough_i]),
        peak_x=int(x[peak_i]),
        trough_x=int(x[trough_i]),
        peak_price=float(fav_price[peak_i]),
        trough_price=float(adv_price[trough_i]),
        time_to_peak_min=int(peak_i),
    )


def is_winner_on_the_hook(result, hook_r: float) -> bool:
    """A losing/flat trade that was at least ``hook_r`` in profit before reversing."""

    return result.gross_r <= 0 and result.mfe_r >= hook_r
