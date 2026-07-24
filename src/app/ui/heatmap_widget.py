"""Exit-grid heatmap: outcome across a stop x target grid for one entry.

Renders a GridResult as a single ImageItem (diverging colours centred on 0 R for
realised R), with the stop/target values on the axes, a hovered-cell readout, and
click-to-apply. A persistent caption states the trial count and that the surface
is exploratory - the honest counterweight the research governance requires.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..analysis.grid_sweep import EXIT_REASONS, GridResult
from . import theme


class HeatmapWidget(QtWidgets.QWidget):
    """A stop x target outcome heatmap; emits (stop_mult, target_r) on click."""

    cellChosen = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid: GridResult | None = None
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Colour by"))
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems(["Realised R", "Exit reason"])
        self.metric_combo.currentTextChanged.connect(lambda _t: self._render())
        top.addWidget(self.metric_combo)
        top.addStretch(1)
        self.readout = QtWidgets.QLabel("")
        top.addWidget(self.readout)
        layout.addLayout(top)

        self._glw = pg.GraphicsLayoutWidget()
        self.plot = self._glw.addPlot(row=0, col=0)
        self.plot.setLabel("bottom", "target (R multiple of stop)")
        self.plot.setLabel("left", "stop (x ATR)")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self._hover = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_hover
        )
        layout.addWidget(self._glw, 1)

        self.caption = QtWidgets.QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

    def set_grid(self, grid: GridResult) -> None:
        self._grid = grid
        self._render()

    # -- rendering ---------------------------------------------------------
    def _render(self) -> None:
        grid = self._grid
        if grid is None:
            return
        p = theme.active()
        metric = self.metric_combo.currentText()
        if metric == "Realised R":
            data = grid.gross_r
            m = float(np.nanmax(np.abs(data))) or 1.0
            levels = (-m, m)
            cmap = pg.ColorMap([0.0, 0.5, 1.0], [p.heat_neg, p.heat_mid, p.heat_pos])
        else:
            data = grid.reason_code.astype(float)
            levels = (0, len(EXIT_REASONS) - 1)
            cmap = pg.ColorMap(np.linspace(0, 1, 4), [p.stop, p.target, p.gold, p.vwap_rolling])
        self.img.setImage(data.T, autoLevels=False)  # x = target, y = stop
        self.img.setLevels(levels)
        self.img.setLookupTable(cmap.getLookupTable(nPts=256))
        self._set_rect(grid)
        rows, cols = grid.gross_r.shape
        self.caption.setText(
            f"Exploratory: {rows} x {cols} = {grid.trials} exits swept on this one entry. "
            f"The best-looking cell is partly luck, not a validated edge - nothing here "
            f"changes the frozen research verdict. Click a cell to apply that exit."
        )

    def _set_rect(self, grid: GridResult) -> None:
        s, t = grid.stop_mults, grid.target_rs
        ds = float(s[1] - s[0]) if len(s) > 1 else 1.0
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
        self.img.setRect(
            QtCore.QRectF(t[0] - dt / 2, s[0] - ds / 2, (t[-1] - t[0]) + dt, (s[-1] - s[0]) + ds)
        )

    # -- interaction -------------------------------------------------------
    def _cell_at(self, scene_pos):
        grid = self._grid
        if grid is None or not self.plot.sceneBoundingRect().contains(scene_pos):
            return None
        pt = self.plot.vb.mapSceneToView(scene_pos)
        j = int(np.argmin(np.abs(grid.target_rs - pt.x())))
        i = int(np.argmin(np.abs(grid.stop_mults - pt.y())))
        return i, j

    def _on_hover(self, evt) -> None:
        cell = self._cell_at(evt[0])
        if cell is None:
            return
        i, j = cell
        grid = self._grid
        self.readout.setText(
            f"stop {grid.stop_mults[i]:.2f}x, target {grid.target_rs[j]:.1f}R  ->  "
            f"{grid.gross_r[i, j]:+.2f} R"
        )

    def _on_click(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        cell = self._cell_at(event.scenePos())
        if cell is None:
            return
        i, j = cell
        self.cellChosen.emit(float(self._grid.stop_mults[i]), float(self._grid.target_rs[j]))
