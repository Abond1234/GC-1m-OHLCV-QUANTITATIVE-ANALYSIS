"""Fibonacci drawing-tool geometry: pure functions, no Qt.

Every tool is defined by two or three anchor points in data coordinates
(bar position, price) and expands deterministically into polylines with
optional labels. Rendering is a thin layer above this, so the geometry is
identical at any zoom and unit-testable without a GUI. Circles, arcs,
spirals and wedges are elliptical in data units (bars x price have no shared
scale), which keeps them zoom-stable and honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

RETRACEMENT_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
EXTENSION_LEVELS = (0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.618)
CHANNEL_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618)
TIME_SEQUENCE = (0, 1, 2, 3, 5, 8, 13, 21, 34)
TIME_RATIOS = (0.382, 0.618, 1.0, 1.382, 1.618, 2.0)
FAN_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
ARC_RATIOS = (0.382, 0.5, 0.618, 0.786, 1.0)
CIRCLE_RATIOS = (0.382, 0.5, 0.618, 1.0, 1.618)
GOLDEN = (1 + math.sqrt(5)) / 2

# How many anchor points each tool needs (the placement gesture reads this).
FIB_POINTS = {
    "fibret": 2,
    "fibext": 3,
    "fibchan": 3,
    "fibtime": 2,
    "fibfan": 2,
    "fibttime": 3,
    "fibcircles": 2,
    "fibspiral": 2,
    "fibarcs": 2,
    "fibwedge": 3,
    "pitchfan": 3,
}


@dataclass(frozen=True)
class FibLine:
    """One polyline of a tool's geometry, with an optional label at its head."""

    xs: tuple
    ys: tuple
    label: str | None = None


def _seg(x1, y1, x2, y2, label=None) -> FibLine:
    return FibLine((float(x1), float(x2)), (float(y1), float(y2)), label)


def _ellipse(cx, cy, rx, ry, t0, t1, n=64) -> tuple[tuple, tuple]:
    ts = [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]
    return (
        tuple(cx + rx * math.cos(t) for t in ts),
        tuple(cy + ry * math.sin(t) for t in ts),
    )


def geometry(kind: str, points, extend: float = 100_000.0) -> list[FibLine]:
    """The polylines for ``kind`` anchored at ``points`` [(x, y), ...]."""

    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < FIB_POINTS[kind]:
        raise ValueError(f"{kind} needs {FIB_POINTS[kind]} points, got {len(pts)}")
    (x1, y1), (x2, y2) = pts[0], pts[1]
    dx, dy = x2 - x1, y2 - y1
    out: list[FibLine] = []

    if kind == "fibret":
        left, right = min(x1, x2), max(x1, x2)
        for r in RETRACEMENT_LEVELS:
            y = y2 - r * dy  # 0 at the second point, 1 back at the first
            out.append(_seg(left, y, right, y, f"{r:g}  {y:,.1f}"))
    elif kind == "fibext":
        (x3, y3) = pts[2]
        left, right = min(x1, x3), max(x1, x3) + abs(dx)
        for r in EXTENSION_LEVELS:
            y = y3 + r * dy  # the p1->p2 move projected from the p3 pullback
            out.append(_seg(left, y, right, y, f"{r:g}  {y:,.1f}"))
    elif kind == "fibchan":
        (x3, y3) = pts[2]
        vx, vy = x3 - x1, y3 - y1
        ex, ey = dx * extend, dy * extend
        for r in CHANNEL_LEVELS:
            out.append(_seg(x1 + r * vx, y1 + r * vy, x2 + r * vx + ex, y2 + r * vy + ey, f"{r:g}"))
    elif kind == "fibtime":
        span = abs(dx) or 1.0
        top, bot = max(y1, y2), min(y1, y2)
        for n in TIME_SEQUENCE:
            x = x1 + n * span
            out.append(_seg(x, bot, x, top, str(n)))
    elif kind == "fibttime":
        (x3, _y3) = pts[2]
        span = abs(dx) or 1.0
        top, bot = max(y1, y2), min(y1, y2)
        for r in TIME_RATIOS:
            x = x3 + r * span
            out.append(_seg(x, bot, x, top, f"{r:g}"))
    elif kind == "fibfan":
        for r in FAN_RATIOS:
            ty = y1 + r * dy
            fx, fy = (x2 - x1) * extend, (ty - y1) * extend
            out.append(_seg(x1, y1, x1 + fx, y1 + fy, f"{r:g}"))
    elif kind == "fibcircles":
        for r in CIRCLE_RATIOS:
            xs, ys = _ellipse(x1, y1, abs(dx) * r or r, abs(dy) * r or r, 0, 2 * math.pi)
            out.append(FibLine(xs, ys, f"{r:g}"))
    elif kind == "fibarcs":
        side = 0.0 if dy >= 0 else math.pi  # arcs face the second point
        for r in ARC_RATIOS:
            xs, ys = _ellipse(x1, y1, abs(dx) * r or r, abs(dy) * r or r, side, side + math.pi)
            out.append(FibLine(xs, ys, f"{r:g}"))
    elif kind == "fibspiral":
        # A golden spiral: radius multiplies by phi each quarter turn, scaled
        # anisotropically by the anchor box so it stays zoom-stable.
        theta0 = math.atan2(dy, dx)
        b = math.log(GOLDEN) / (math.pi / 2)
        n = 240
        xs, ys = [], []
        for i in range(n):
            t = -3 * math.pi + (3.5 * math.pi) * i / (n - 1)  # ~1.75 turns out
            scale = math.exp(b * t)
            xs.append(x1 + (abs(dx) or 1.0) * scale * math.cos(t + theta0))
            ys.append(y1 + (abs(dy) or 1.0) * scale * math.sin(t + theta0))
        out.append(FibLine(tuple(xs), tuple(ys), None))
    elif kind == "fibwedge":
        (x3, y3) = pts[2]
        sx = max(abs(dx), abs(x3 - x1)) or 1.0
        sy = max(abs(dy), abs(y3 - y1)) or 1.0
        a2 = math.atan2((y2 - y1) / sy, (x2 - x1) / sx)
        a3 = math.atan2((y3 - y1) / sy, (x3 - x1) / sx)
        out.append(_seg(x1, y1, x2, y2))
        out.append(_seg(x1, y1, x3, y3))
        for r in ARC_RATIOS:
            xs, ys = _ellipse(x1, y1, sx * r, sy * r, a2, a3, n=48)
            out.append(FibLine(xs, ys, f"{r:g}"))
    elif kind == "pitchfan":
        (x3, y3) = pts[2]
        for r in (0.0, 0.25, 0.382, 0.5, 0.618, 0.75, 1.0):
            tx = x2 + r * (x3 - x2)
            ty = y2 + r * (y3 - y2)
            fx, fy = (tx - x1) * extend, (ty - y1) * extend
            out.append(_seg(x1, y1, x1 + fx, y1 + fy, f"{r:g}"))
    else:
        raise KeyError(f"unknown fib tool: {kind}")
    return out
