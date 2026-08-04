"""Measurement-tool arithmetic: pure functions, no Qt.

A measurement runs from an anchor to a destination in data coordinates
(price and bar position), so its numbers are identical at any zoom level by
construction. Tick counts use the active instrument's tick size - never a
hardcoded 0.1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Measurement:
    """The readout for one anchor-to-destination drag."""

    price_change: float  # signed, in points
    ticks: float  # signed, in instrument ticks
    percent: float  # signed, relative to the anchor price
    bars: int  # displayed bars spanned (current timeframe)
    minutes: int  # elapsed 1-minute bars spanned (timeframe-independent)


def measure(
    price_a: float,
    price_b: float,
    tick_size: float,
    bars: int,
    minutes: int,
) -> Measurement:
    """Numbers for a measurement from ``a`` (anchor) to ``b`` (destination)."""

    change = float(price_b) - float(price_a)
    ticks = change / float(tick_size) if tick_size else 0.0
    percent = (change / float(price_a) * 100.0) if price_a else 0.0
    return Measurement(
        price_change=change,
        ticks=ticks,
        percent=percent,
        bars=int(abs(bars)),
        minutes=int(abs(minutes)),
    )


def format_duration(minutes: int) -> str:
    """Compact elapsed time: 45m, 3h 20m, 2d 4h."""

    minutes = int(minutes)
    if minutes < 60:
        return f"{minutes}m"
    hours, rem = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {rem}m" if rem else f"{hours}h"
    days, hrem = divmod(hours, 24)
    return f"{days}d {hrem}h" if hrem else f"{days}d"


def format_measurement(m: Measurement, decimals: int) -> str:
    """The on-chart readout block (plain text, one line per fact)."""

    return (
        f"{m.price_change:+,.{decimals}f} pts ({m.percent:+.2f}%)\n"
        f"{m.ticks:+,.0f} ticks\n"
        f"{m.bars} bars, {format_duration(m.minutes)}"
    )
