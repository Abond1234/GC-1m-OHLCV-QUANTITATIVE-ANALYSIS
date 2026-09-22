"""Central design system for the trade simulator.

One home for every colour, font, spacing, and magic constant the UI uses, so the
app reads as a single coherent surface and a light/dark switch is a one-liner.
Widgets import tokens from here instead of hardcoding literals; ``apply`` pushes
the palette into both Qt (a stylesheet) and pyqtgraph (its global config).

Values are plain data (hex strings / RGBA tuples) so non-Qt code and tests can
read them without importing PySide6 or pyqtgraph. The only functions that touch
those libraries import them lazily.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# --- theme-independent constants (previously scattered magic numbers) --------
CANDLE_HALF_WIDTH = 0.36  # half the candle body width, in bar-index units
CANDLE_MIN_HEIGHT = 1e-6  # doji guard so a zero-range bar still paints
CHART_PAD_BARS = 60  # default bars of padding on each side when centering
HOOK_R = 1.0  # "winner on the hook": gross_r<=0 but mfe_r>=HOOK_R
TABLE_ROW_CAP = 500  # max rows rendered in the trade blotter at once
FORENSICS_HORIZONS = (5, 15, 30, 60, 120, 180)  # minutes; forward-label horizons

# Execution-session windows (minute-of-day, New York). Mirrors
# ``vwap._execution_session_code`` so shading and VWAP stay in lockstep.
ASIA_WINDOW = (180, 360)  # [03:00, 06:00)
NY_WINDOW = (420, 720)  # [07:00, 12:00)


@dataclass(frozen=True)
class Palette:
    """A full colour set for one theme mode."""

    name: str
    # surfaces
    bg: str
    panel: str
    panel_alt: str
    border: str
    grid: str
    # text
    text: str
    text_dim: str
    text_faint: str
    # brand / market
    gold: str
    up: str
    down: str
    # trade levels
    stop: str
    target: str
    entry: str
    hook: str
    mfe_fill: tuple[int, int, int, int]  # RGBA translucent favourable ribbon
    mae_fill: tuple[int, int, int, int]  # RGBA translucent adverse ribbon
    hook_fill: tuple[int, int, int, int]  # RGBA brighter fill for 'winner on the hook'
    # vwap family
    vwap_rolling: str
    vwap_day: str
    vwap_session: str
    # overlays / heatmap
    session_shade: tuple[int, int, int, int]  # RGBA session background
    whatif_cycle: tuple[str, ...]  # distinct path colours for what-if overlays
    heat_pos: str  # diverging colormap: positive end
    heat_mid: str  # diverging colormap: zero
    heat_neg: str  # diverging colormap: negative end


DARK = Palette(
    name="dark",
    bg="#0d0c0a",
    panel="#16130f",
    panel_alt="#1c1813",
    border="#2b261d",
    grid="#241f18",
    text="#e9e1d1",
    text_dim="#a99f8a",
    text_faint="#726a58",
    gold="#cfa94e",
    up="#63b491",
    down="#d0705d",
    stop="#e06a52",
    target="#5cc0a0",
    entry="#cfa94e",
    hook="#f0c65e",
    mfe_fill=(92, 192, 160, 45),
    mae_fill=(224, 106, 82, 40),
    hook_fill=(240, 198, 94, 66),
    vwap_rolling="#5aa0d6",
    vwap_day="#c58fd6",
    vwap_session="#d99a5c",
    session_shade=(207, 169, 78, 18),
    whatif_cycle=("#e0b74e", "#6fb3e0", "#c58fd6", "#7fc99a", "#e0876a"),
    heat_pos="#5cc0a0",
    heat_mid="#16130f",
    heat_neg="#e06a52",
)

LIGHT = Palette(
    name="light",
    bg="#f6f2e9",
    panel="#efe8d9",
    panel_alt="#e6ddca",
    border="#cabfa6",
    grid="#ddd3bd",
    text="#2a251d",
    text_dim="#5c5342",
    text_faint="#8a8069",
    gold="#a9812c",
    up="#2f8f6b",
    down="#bc4d38",
    stop="#c8462f",
    target="#2f8f6b",
    entry="#a9812c",
    hook="#b3841f",
    mfe_fill=(47, 143, 107, 55),
    mae_fill=(188, 77, 56, 50),
    hook_fill=(179, 132, 31, 80),
    vwap_rolling="#3a78ad",
    vwap_day="#9155ad",
    vwap_session="#b3701f",
    session_shade=(169, 129, 44, 26),
    whatif_cycle=("#b3841f", "#3a78ad", "#9155ad", "#3f8f5f", "#c05a34"),
    heat_pos="#2f8f6b",
    heat_mid="#efe8d9",
    heat_neg="#c8462f",
)

_MODES = {"dark": DARK, "light": LIGHT}
_active: Palette = DARK


def active() -> Palette:
    """The palette currently applied to the app."""

    return _active


def qss(p: Palette) -> str:
    """A cohesive Qt stylesheet for the given palette."""

    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[3]))
    plus_icon = (bundle_root / "assets" / "ui" / f"spin_plus_{p.name}.svg").as_posix()
    minus_icon = (bundle_root / "assets" / "ui" / f"spin_minus_{p.name}.svg").as_posix()
    return f"""
    QMainWindow, QWidget {{ background: {p.bg}; color: {p.text};
        font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 12px; }}
    QToolBar {{ background: {p.panel}; border: none; spacing: 6px; padding: 4px; }}
    QToolButton {{ background: transparent; border: 1px solid transparent;
        border-radius: 5px; padding: 5px 8px; color: {p.text}; }}
    QToolButton:hover {{ border-color: {p.gold}; }}
    QToolButton:checked {{ background: {p.gold}; color: {p.bg}; font-weight: 600; }}
    QStatusBar {{ background: {p.panel}; color: {p.text_dim}; }}
    QDockWidget {{ titlebar-close-icon: none; color: {p.text_dim}; }}
    QDockWidget::title {{ background: {p.panel_alt}; padding: 6px 10px;
        border-bottom: 1px solid {p.border}; font-weight: 600; }}
    QGroupBox {{ background: {p.panel}; border: 1px solid {p.border};
        border-radius: 6px; margin-top: 14px; padding: 8px 8px 6px 8px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
        color: {p.gold}; font-weight: 600; }}
    QLabel {{ background: transparent; }}
    QLabel[role="caption"] {{ color: {p.text_faint}; font-size: 11px; }}
    QLabel[role="heading"] {{ font-size: 16px; font-weight: 600; }}
    QLabel[role="metric"] {{ font-size: 18px; font-weight: 600; }}
    QPushButton {{ background: {p.panel_alt}; border: 1px solid {p.border};
        border-radius: 5px; padding: 5px 12px; color: {p.text}; }}
    QPushButton:hover {{ border-color: {p.gold}; }}
    QPushButton:pressed {{ background: {p.border}; }}
    QPushButton[accent="true"] {{ background: {p.gold}; color: {p.bg}; font-weight: 600;
        border: none; }}
    QPushButton[accent="true"]:hover {{ background: {p.hook}; }}
    QPushButton#simulatorToggle {{ background: #000000; color: #ffffff;
        font-weight: 600; border: 1px solid #555555; padding: 6px 18px; }}
    QPushButton#simulatorToggle:hover, QPushButton#simulatorToggle:checked {{
        background: #000000; color: #ffffff; border-color: {p.gold}; }}
    QComboBox, QDoubleSpinBox, QSpinBox {{ background: {p.panel_alt};
        border: 1px solid {p.border}; border-radius: 5px; padding: 3px 6px;
        color: {p.text}; selection-background-color: {p.gold}; }}
    QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{ border-color: {p.gold}; }}
    QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled,
    QPushButton:disabled {{ color: {p.text_faint}; background: {p.panel};
        border-color: {p.border}; }}
    QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled {{ color: {p.text_faint}; }}
    QDoubleSpinBox[invalid="true"], QSpinBox[invalid="true"] {{ border-color: {p.down}; }}
    QLabel[role="warn"] {{ color: {p.down}; font-size: 11px; }}
    QLineEdit {{ background: {p.panel_alt}; border: 1px solid {p.border};
        border-radius: 5px; padding: 2px 6px; color: {p.text};
        selection-background-color: {p.gold}; selection-color: {p.bg}; }}
    QComboBox QLineEdit {{ border: none; padding: 0; }}
    QDoubleSpinBox, QSpinBox {{ padding-right: 22px; }}
    QDoubleSpinBox::up-button, QSpinBox::up-button {{
        subcontrol-origin: border; subcontrol-position: top right;
        width: 18px; background: {p.panel_alt};
        border-left: 1px solid {p.border}; border-bottom: 1px solid {p.border};
        border-top-right-radius: 5px; }}
    QDoubleSpinBox::down-button, QSpinBox::down-button {{
        subcontrol-origin: border; subcontrol-position: bottom right;
        width: 18px; background: {p.panel_alt};
        border-left: 1px solid {p.border};
        border-bottom-right-radius: 5px; }}
    QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
    QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
        background: {p.border}; }}
    QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed,
    QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{
        background: {p.gold}; }}
    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
        image: url("{plus_icon}"); width: 10px; height: 10px; }}
    QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
        image: url("{minus_icon}"); width: 10px; height: 10px; }}
    QComboBox QAbstractItemView {{ background: {p.panel_alt}; color: {p.text};
        selection-background-color: {p.gold}; selection-color: {p.bg};
        border: 1px solid {p.border}; }}
    QCheckBox, QRadioButton {{ spacing: 6px; background: transparent; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px;
        border: 1px solid {p.border}; border-radius: 3px; background: {p.panel_alt}; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background: {p.gold}; border-color: {p.gold}; }}
    QTableWidget, QTableView {{ background: {p.panel}; alternate-background-color: {p.panel_alt};
        gridline-color: {p.border}; color: {p.text}; border: 1px solid {p.border};
        selection-background-color: {p.gold}; selection-color: {p.bg}; }}
    QHeaderView::section {{ background: {p.panel_alt}; color: {p.text_dim};
        padding: 4px 6px; border: none; border-right: 1px solid {p.border};
        border-bottom: 1px solid {p.border}; font-weight: 600; }}
    QTableView::item:selected {{ background: {p.gold}; color: {p.bg}; }}
    QSplitter::handle {{ background: {p.border}; }}
    QSplitter::handle:hover {{ background: {p.gold}; }}
    QScrollBar:vertical {{ background: {p.panel}; width: 16px; margin: 18px 0 18px 0; }}
    QScrollBar::handle:vertical {{ background: {p.text_faint}; border: 1px solid {p.border};
        border-radius: 7px; min-height: 32px; }}
    QScrollBar::handle:vertical:hover {{ background: {p.gold}; border-color: {p.gold}; }}
    QScrollBar::sub-line:vertical {{ background: {p.panel_alt}; height: 18px;
        subcontrol-position: top; subcontrol-origin: margin; border: 1px solid {p.border}; }}
    QScrollBar::add-line:vertical {{ background: {p.panel_alt}; height: 18px;
        subcontrol-position: bottom; subcontrol-origin: margin; border: 1px solid {p.border}; }}
    QScrollBar::sub-line:vertical:hover, QScrollBar::add-line:vertical:hover {{
        background: {p.gold}; border-color: {p.gold}; }}
    QScrollBar::sub-arrow:vertical {{ width: 0; height: 0;
        border-left: 4px solid transparent; border-right: 4px solid transparent;
        border-bottom: 6px solid {p.text}; }}
    QScrollBar::add-arrow:vertical {{ width: 0; height: 0;
        border-left: 4px solid transparent; border-right: 4px solid transparent;
        border-top: 6px solid {p.text}; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    QScrollBar:horizontal {{ background: {p.panel}; height: 16px; margin: 0 18px 0 18px; }}
    QScrollBar::handle:horizontal {{ background: {p.text_faint}; border: 1px solid {p.border};
        border-radius: 7px; min-width: 32px; }}
    QScrollBar::handle:horizontal:hover {{ background: {p.gold}; border-color: {p.gold}; }}
    QScrollBar::sub-line:horizontal {{ background: {p.panel_alt}; width: 18px;
        subcontrol-position: left; subcontrol-origin: margin; border: 1px solid {p.border}; }}
    QScrollBar::add-line:horizontal {{ background: {p.panel_alt}; width: 18px;
        subcontrol-position: right; subcontrol-origin: margin; border: 1px solid {p.border}; }}
    QScrollBar::sub-arrow:horizontal {{ width: 0; height: 0;
        border-top: 4px solid transparent; border-bottom: 4px solid transparent;
        border-right: 6px solid {p.text}; }}
    QScrollBar::add-arrow:horizontal {{ width: 0; height: 0;
        border-top: 4px solid transparent; border-bottom: 4px solid transparent;
        border-left: 6px solid {p.text}; }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
    QSlider::groove:horizontal {{ height: 4px; background: {p.border}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {p.gold}; width: 12px; margin: -5px 0;
        border-radius: 6px; }}
    QToolTip {{ background: {p.panel_alt}; color: {p.text}; border: 1px solid {p.gold};
        padding: 4px 6px; }}
    QProgressBar {{ background: {p.panel_alt}; border: 1px solid {p.border};
        border-radius: 4px; text-align: center; color: {p.text_dim}; }}
    QProgressBar::chunk {{ background: {p.gold}; border-radius: 3px; }}
    QTabWidget::pane {{ border: 1px solid {p.border}; }}
    QTabBar::tab {{ background: {p.panel}; color: {p.text_dim}; padding: 5px 12px;
        border: 1px solid {p.border}; border-bottom: none; }}
    QTabBar::tab:selected {{ background: {p.panel_alt}; color: {p.gold}; }}
    QTabWidget#simulatorTabs > QTabBar::tab {{ padding: 7px 8px; }}
    """


def apply(app, mode: str = "dark") -> Palette:
    """Apply ``mode`` to the QApplication and pyqtgraph; return the palette."""

    global _active
    import pyqtgraph as pg

    _active = _MODES.get(mode, DARK)
    pg.setConfigOptions(
        antialias=True,
        background=_active.bg,
        foreground=_active.text_faint,
    )
    app.setStyleSheet(qss(_active))
    return _active
