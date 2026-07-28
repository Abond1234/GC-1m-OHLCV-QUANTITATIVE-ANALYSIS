"""Edge context tab: the four validated research features, live per bar.

Shows the FES Project 1 advancing slate (Section 6, locked Validation) for the
selected day: current value at the crosshair/selected bar, a day sparkline, and
the citation (session, target, Dev IC to Validation IC). The header carries the
frozen verdict verbatim; features outside their validated entry session are
dimmed rather than hidden, so the session-specificity of the evidence is
impossible to miss. No thresholds, no buy/sell colouring - this is descriptive
research context, not a signal.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from ..datalayer.edge_context import EDGE_FEATURES, VERDICT_CAPTION, EdgeDayContext
from . import theme

_LONDON = (180, 360)  # [03:00, 06:00) NY minutes - London entry window
_NEW_YORK = (420, 720)  # [07:00, 12:00) NY minutes


def _entry_session(minute: int) -> str | None:
    if _LONDON[0] <= minute < _LONDON[1]:
        return "London"
    if _NEW_YORK[0] <= minute < _NEW_YORK[1]:
        return "New York"
    return None


class EdgeContextPanel(QtWidgets.QWidget):
    """Four feature rows: value, sparkline, citation; session-honest dimming."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._context: EdgeDayContext | None = None
        self._rows: dict[str, dict] = {}
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QtWidgets.QLabel(VERDICT_CAPTION)
        header.setWordWrap(True)
        header.setProperty("role", "caption")
        layout.addWidget(header)

        self.status = QtWidgets.QLabel("Toggle a date to compute the day's context.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        for info in EDGE_FEATURES:
            box = QtWidgets.QGroupBox(info.label)
            grid = QtWidgets.QVBoxLayout(box)
            grid.setContentsMargins(10, 6, 10, 8)
            grid.setSpacing(4)
            top = QtWidgets.QHBoxLayout()
            value = QtWidgets.QLabel("-")
            value.setToolTip(
                "Value at the selected/hovered bar, computed with the frozen\n"
                "research formulas (float32 research fidelity)."
            )
            session_tag = QtWidgets.QLabel("")
            session_tag.setProperty("role", "caption")
            top.addWidget(value)
            top.addStretch(1)
            top.addWidget(session_tag)
            grid.addLayout(top)
            spark = pg.PlotWidget()
            spark.setFixedHeight(44)
            spark.setMouseEnabled(x=False, y=False)
            spark.hideButtons()
            spark.hideAxis("left")
            spark.hideAxis("bottom")
            grid.addWidget(spark)
            cite = QtWidgets.QLabel(
                f"validated: {info.session} - target {info.target} - "
                f"IC {info.dev_ic:+.3f} (Dev) to {info.val_ic:+.3f} (Val) - ADVANCES"
            )
            cite.setProperty("role", "caption")
            cite.setWordWrap(True)
            grid.addWidget(cite)
            layout.addWidget(box)
            self._rows[info.key] = {
                "box": box,
                "value": value,
                "tag": session_tag,
                "spark": spark,
                "info": info,
            }
        layout.addStretch(1)

    # -- data --------------------------------------------------------------
    def set_context(self, context: EdgeDayContext) -> None:
        """Install a computed day and draw the sparklines."""

        p = theme.active()
        self._context = context
        self.status.setText("Computed for the selected day with the frozen research formulas.")
        for key, row in self._rows.items():
            spark: pg.PlotWidget = row["spark"]
            spark.clear()
            series = context.values.get(key)
            if series is None or not np.isfinite(series).any():
                continue
            x = np.arange(len(series), dtype=float)
            finite = np.isfinite(series)
            spark.addItem(
                pg.PlotDataItem(
                    x[finite], series[finite].astype(float), pen=pg.mkPen(p.gold, width=1.0)
                )
            )
        self.show_bar(len(context.minute_ny) - 1)

    def clear_context(self, message: str) -> None:
        self._context = None
        self.status.setText(message)
        for row in self._rows.values():
            row["spark"].clear()
            row["value"].setText("-")
            row["tag"].setText("")

    def show_bar(self, local_index: int) -> None:
        """Update the value readouts for one bar of the loaded day."""

        context = self._context
        if context is None:
            return
        n = len(context.minute_ny)
        i = max(0, min(int(local_index), n - 1))
        minute = int(context.minute_ny[i])
        session = _entry_session(minute)
        p = theme.active()
        for key, row in self._rows.items():
            info = row["info"]
            value = context.values[key][i]
            cause = context.causes[key][i]
            if np.isfinite(value):
                row["value"].setText(f"{float(value):+.4f}")
            else:
                row["value"].setText(cause.replace("_", " ").lower() or "-")
            if session is None:
                row["tag"].setText("no entry window")
                row["box"].setEnabled(False)
            elif session != info.session:
                row["tag"].setText(f"outside validated session ({info.session})")
                row["box"].setEnabled(False)
            else:
                row["tag"].setText(f"in validated session ({info.session})")
                row["box"].setEnabled(True)
            row["value"].setStyleSheet(
                f"color:{p.text}" if row["box"].isEnabled() else f"color:{p.text_faint}"
            )

    def retheme(self) -> None:
        p = theme.active()
        for row in self._rows.values():
            row["spark"].setBackground(p.bg)
