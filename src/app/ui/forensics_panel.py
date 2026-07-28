"""Forensics panel: why the trade worked or failed, shown not just told.

Three parts: a plain-language verdict (from ``analysis.explain``), a per-horizon
mini-chart of how far the move ran for and against over 5-180 minutes (so the
edge decay is visible at a glance), and labelled meters for the entry's feature
context. Replaces the previous single HTML blob.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..analysis.explain import verdict
from . import theme

_Meter = namedtuple("_Meter", "key label lo hi fmt tip")

_METERS = (
    _Meter(
        "efficiency_ratio_30",
        "Trend efficiency",
        0.0,
        1.0,
        "{:.2f}",
        "0 = pure noise, 1 = a straight line over the last 30 bars.",
    ),
    _Meter(
        "ols_r_squared_30",
        "Trend fit (R2)",
        0.0,
        1.0,
        "{:.2f}",
        "How linear the last 30 bars were (1 = perfectly straight).",
    ),
    _Meter(
        "choppiness_14",
        "Choppiness",
        0.0,
        100.0,
        "{:.0f}",
        "High = ranging/choppy tape; low = trending.",
    ),
    _Meter(
        "atr_ratio_5_20",
        "Volatility 5/20",
        0.0,
        3.0,
        "{:.2f}",
        "Short vs long average range; above 1 = volatility expanding.",
    ),
    _Meter(
        "relative_volume_60",
        "Relative volume",
        0.0,
        3.0,
        "{:.2f}",
        "Volume vs this hour's norm; above 1 = busier than usual.",
    ),
    _Meter(
        "session_range_position",
        "Session range pos",
        0.0,
        1.0,
        "{:.2f}",
        "0 = at the session low, 1 = at the session high.",
    ),
    _Meter(
        "distance_from_execution_session_vwap_atr",
        "VWAP distance",
        -3.0,
        3.0,
        "{:+.2f}",
        "ATR from the session VWAP; positive = above, negative = below.",
    ),
)


class ForensicsPanel(QtWidgets.QWidget):
    """Verdict text, a per-horizon excursion chart, and entry-context meters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._meters: dict[str, tuple] = {}
        self._last_html = None  # skip QTextDocument rebuilds during 60 Hz drags
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.verdict = QtWidgets.QTextEdit()
        self.verdict.setReadOnly(True)
        self.verdict.setMinimumHeight(120)
        layout.addWidget(self.verdict)

        self._horizon_box = QtWidgets.QGroupBox("How far it ran (per horizon, ATR)")
        hb = QtWidgets.QVBoxLayout(self._horizon_box)
        self.horizon_plot = pg.PlotWidget()
        self.horizon_plot.setMaximumHeight(160)
        self.horizon_plot.setMouseEnabled(x=False, y=False)
        self.horizon_plot.hideButtons()
        self.horizon_plot.showGrid(x=False, y=True, alpha=0.12)
        hb.addWidget(self.horizon_plot)
        layout.addWidget(self._horizon_box)

        self._meter_box = QtWidgets.QGroupBox("Entry context")
        grid = QtWidgets.QGridLayout(self._meter_box)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)
        # When the pane narrows, shrink the bar (column 1), never the labels.
        grid.setColumnStretch(1, 1)
        for row, spec in enumerate(_METERS):
            name = QtWidgets.QLabel(spec.label)
            name.setToolTip(spec.tip)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            bar.setToolTip(spec.tip)
            val = QtWidgets.QLabel("n/a")
            val.setMinimumWidth(52)
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(name, row, 0)
            grid.addWidget(bar, row, 1)
            grid.addWidget(val, row, 2)
            self._meters[spec.key] = (bar, val, spec)
        layout.addWidget(self._meter_box)
        layout.addStretch(1)
        self._show_context_widgets(False)

    def _show_context_widgets(self, on: bool) -> None:
        self._horizon_box.setVisible(on)
        self._meter_box.setVisible(on)

    # -- public API --------------------------------------------------------
    def show_realized(self, result) -> None:
        """Verdict from the realized path only (no registered context yet)."""

        html = self._verdict_html(result, ctx=None)
        if html != self._last_html:  # a drag re-explains per event; text rarely changes
            self._last_html = html
            self.verdict.setHtml(html)
        self._show_context_widgets(False)

    def show_context(self, result, ctx) -> None:
        """Full verdict plus the per-horizon chart and feature meters."""

        self._last_html = self._verdict_html(result, ctx=ctx)
        self.verdict.setHtml(self._last_html)
        self._draw_horizons(result.direction, ctx)
        self._fill_meters(ctx.features)
        self._show_context_widgets(True)

    def clear(self) -> None:
        self._last_html = None
        self.verdict.clear()
        self._show_context_widgets(False)

    def retheme(self) -> None:
        """Re-style the pieces that baked the previous palette in."""

        self.horizon_plot.setBackground(theme.active().bg)

    # -- rendering ---------------------------------------------------------
    def _verdict_html(self, result, ctx) -> str:
        p = theme.active()
        lines = verdict(result, ctx=ctx, hook_r=theme.HOOK_R)
        side = "Long" if result.direction > 0 else "Short"
        colour = p.up if result.gross_r > 0 else p.down
        head = (
            f"<div style='color:{colour}; font-size:14px; font-weight:600'>"
            f"{side} &middot; {result.exit_reason} &middot; {result.gross_r:+.2f} R</div>"
        )
        body = "".join(f"<p style='margin:4px 0'>{line}</p>" for line in lines)
        return head + body

    def _draw_horizons(self, direction, ctx) -> None:
        p = theme.active()
        self.horizon_plot.clear()
        horizons = theme.FORENSICS_HORIZONS
        mfe, mae, fret = [], [], []
        for h in horizons:
            e = ctx.excursions.get(h, {})
            if direction > 0:
                mfe.append(_num(e.get("mfe_long_atr")))
                mae.append(-abs(_num(e.get("mae_long_atr"))))
            else:
                mfe.append(_num(e.get("mfe_short_atr")))
                mae.append(-abs(_num(e.get("mae_short_atr"))))
            fret.append(_num(e.get("forward_return_atr")))
        x = np.arange(len(horizons))
        self.horizon_plot.addItem(
            pg.BarGraphItem(x=x, width=0.6, height=np.nan_to_num(mfe), brush=p.up, pen=None)
        )
        self.horizon_plot.addItem(
            pg.BarGraphItem(x=x, width=0.6, height=np.nan_to_num(mae), brush=p.down, pen=None)
        )
        self.horizon_plot.addItem(
            pg.ScatterPlotItem(x=x, y=fret, symbol="o", size=7, brush=pg.mkBrush(p.gold))
        )
        self.horizon_plot.addItem(
            pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(p.text_faint, width=1))
        )
        axis = self.horizon_plot.getAxis("bottom")
        axis.setTicks([[(i, f"{h}m") for i, h in enumerate(horizons)]])

    def _fill_meters(self, feats: dict) -> None:
        # No per-label stylesheets: the app-wide QSS colours QLabel, so a theme
        # switch restyles these automatically instead of leaving stale colours.
        feats = feats or {}
        for key, (bar, val, spec) in self._meters.items():
            v = feats.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                bar.setValue(0)
                val.setText("n/a")
                continue
            frac = (float(v) - spec.lo) / (spec.hi - spec.lo)
            bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
            val.setText(spec.fmt.format(v))


def _num(value) -> float:
    if value is None:
        return float("nan")
    return float(value)
