# GC Trade Simulator (app)

A native desktop app (PySide6 + pyqtgraph) for charting the GC dataset, replaying
any catalog/peer strategy, and placing trades with **flexible, non-frozen exits**
to understand - visually - why a trade worked or failed. It reuses the verified
research engine in-process, so a simulated fill is exactly what the research engine
would record. Development + Validation only; the Final-test partition is never
loaded or rendered.

Status: **runs on Python 3.14; backend and analysis fully tested; the UI is
verified with offscreen-rendered screenshots and a live run.**

## Install and run

All commands run from the repository root. Setting up from scratch:

```powershell
# 1. Create and activate a virtual environment (Python 3.14)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install the pinned research core plus the app dependencies
python -m pip install -r requirements.txt -r requirements-app.txt

# 3. Run the app from source
python scripts/run_trade_simulator.py
```

The app shows the Gold Quant splash while the Development+Validation bars load
(about ten seconds for the full window), then opens the main window. The local
parquet data under `data/processed/` must be present - it is never committed, so
a fresh clone needs the data copied in first.

`requirements-app.txt` (PySide6, pyqtgraph, and pyinstaller for packaging) is kept
out of the pinned research core so the deterministic install and CI are unaffected.
PySide6 6.11 ships an abi3 wheel and installs on the repo's Python 3.14 venv.

## Building the standalone app locally

```powershell
python scripts/build_app.py
```

This renders the code-drawn GQ icon into the git-ignored `build/` directory and
packages `dist/GCTradeSimulator/GCTradeSimulator.exe` (a one-directory
PyInstaller build, about 400 MB) with that icon embedded. The executable does
NOT bundle the multi-hundred-MB parquet data (per the data governance), so run
it where it can find a checkout's `data/processed`:

```powershell
# either: launch from anywhere inside the checkout (it walks up to find data/)
dist\GCTradeSimulator\GCTradeSimulator.exe

# or: point a build that lives elsewhere at a checkout explicitly
$env:GC_PROJECT_ROOT = "C:\path\to\project-1"
dist\GCTradeSimulator\GCTradeSimulator.exe
```

`dist/`, `build/`, and the generated `GCTradeSimulator.spec` are git-ignored;
rebuild whenever you want a fresh packaged artifact - day-to-day development
runs from source.

## Instruments and adding an instrument

The topbar's instrument picker switches the primary chart among assets with
local data. GC and MGC come from the research bar table (Development+Validation
capped). The registry also knows NQ ($20/pt, 0.25 tick), ES ($50/pt, 0.25), and
BTCUSD ($1/pt) - they appear automatically once you provide bars at:

```
data/processed/instruments/<SYMBOL>_1m.parquet
```

Required columns (enforced loudly; a wrong file is refused, never rendered):
`ts_event_utc, open, high, low, close, volume, trade_date_ny,
minute_of_day_ny, continuous_segment_id, rolling_atr_20m`. One row per minute;
`continuous_segment_id` must break across data gaps/rolls so no candle or
simulation window spans a discontinuity; `rolling_atr_20m` is a simple
20-minute mean of true range (the exit engine sizes ATR stops from it).
New instruments register in `src/app/datalayer/instruments.py`.

Research validity is instrument-scoped: the strategy catalog, replay markers,
and the Edge context slate are GC evidence, so they disable - with the reason
on screen - on every other instrument. Free-play, exits, what-ifs, exit grids,
drawings, replay-view, and the session money layer (which auto-syncs the
active contract's dollars-per-point) work on any loaded instrument.

## Verifying a change

```powershell
python -m unittest discover -s tests          # full suite (app tests included)
python -m ruff check . ; python -m ruff format --check .
python scripts/render_screens.py              # UI screenshots without clicking through
```

`render_screens.py` builds the real `MainWindow` under a narrow bar load, drives
it through the same slots a user would, and writes PNG screenshots (including a
narrow-display pass and a mid-replay frame). It uses the OS default platform so
real fonts render; set `QT_QPA_PLATFORM=offscreen` for a truly headless machine.

## What it does

- **Chart**: GC candlesticks with volume, three VWAP overlays (rolling / day /
  session), New York session shading, a crosshair with an O/H/L/C readout, and a
  dual price axis. Candles never draw across a continuous-segment break.
- **Replay**: pick a strategy (the 12-rule library or the ~200-strategy catalog),
  plot its trades coloured by outcome, and click any trade to redraw its exact path
  and read its forensics.
- **Free-play (no frozen exit)**: enable free-play and click to place trades that
  persist in a blotter. Size every exit field in the **Exit-rule** panel (stop and
  target mode/value, trailing, breakeven, holding cap, forced 15:30 exit), or just
  **drag the stop/target lines on the chart** and watch the outcome recompute live.
- **Why it failed, visually**:
  - MFE/MAE **excursion ribbons** on the price path with peak/trough markers and a
    time-to-peak label (gold for a *winner on the hook*).
  - **What-if exits**: re-run the same entry under a wider stop / trailing / longer
    cap / tighter target, overlaid on the chart with a comparison table.
  - **Exit-grid heatmap**: sweep stop x target for one entry and colour the outcome;
    click a cell to apply it. Labelled exploratory (it shows the trial count).
  - **Bar-by-bar replay with the future hidden**: animate a trade from entry to
    exit with play/step/speed while a reveal curtain hides everything after
    "now" - candles appear one by one, TradingView-replay style, and the view
    auto-scrolls with the tape. Shaded risk/reward zones between entry and the
    stop/target tracks show the position like a long/short tool (a trailing
    stop's ratchet is visible as the red zone tightening).
  - **Replay day**: the same tape reveal for a whole day with no trade attached
    - play the session from its first bar and place free-play entries on
    revealed bars as it runs.
  - **Forensics panel**: a plain-language verdict, a per-horizon MFE/MAE mini-chart,
    and labelled entry-context meters.
- **Drawing tools** (left Draw bar, View menu Ctrl+4): draggable price levels,
  trendlines (drag across the chart or click two points), shaded supply/demand
  zones, and vertical time markers - all with TradingView-style drag gestures, a
  live preview, right-click to cancel, and Undo/Clear. Drawings live in data
  coordinates (exact under any zoom), persist per day within the session, and
  stay visible above the replay curtain while the tape plays.
- **MGC mirror pane** (row-2 checkbox or Ctrl+5): the micro-gold tape for the
  same day rendered underneath, with the current GC trade's entry/exit and
  initial stop/target ghosted onto it by timestamp - a visual check of whether
  the pattern replicated on the execution instrument. MGC bars load on demand
  through the same Development+Validation cap.
- **Multi-day ranges and display timeframes**: a range preset (1D/1W/1M/custom)
  anchored at the selected date, and a timeframe selector (1m to 1D). The
  aggregation is display-only - every simulation, entry, exit, and replay stays
  on true 1-minute bars - and every overlay (trades, ribbons, replay curtain,
  drawings, free-play clicks) maps through a single ViewMap, so everything
  works identically at any timeframe. TradingView-style price-axis handling:
  left-drag the price axis to scale, double-click it to auto-fit.
- **Session tab (money layer)**: a dollar account model (GC $100/pt or MGC
  $10/pt, risk-percent or fixed-contract sizing) turns the blotter into an
  equity curve with desk stats, and an optional prop-style evaluation preset
  arms hard rules (daily loss, max drawdown, profit target, minimum days) with
  live meters and a pass/fail verdict. Simulated fills on historical Dev+Val
  data - explicitly not live results.
- **Edge context tab**: the four features that ADVANCED through the FES
  Project 1 locked Validation batch, computed per bar for the selected day with
  the research module's own frozen helpers (a golden test pins the app's
  values to ``build_scalar_feature_matrix`` exactly). Session-honest display -
  London-validated and New York-validated features dim outside their windows -
  under the verbatim verdict: PREDICTIVE_ONLY_NOT_DIRECTIONAL, FROZEN_NO_POLICY,
  not a trade signal.
- **Sessions (File menu)**: save/load the working session (trades by entry and
  exit config, drawings, view, account and evaluation settings) as versioned
  JSON under the git-ignored ``reports/sessions/``. Loaded trades are
  re-simulated through the verified engine - results are never read from disk.
- **Light / dark** theme (View menu). Every surface re-themes in place, including
  the heatmap's colour scale and the chart axes/crosshair.
- **Collapsible panels**: the Exit-rule dock, the analysis sidebar, and the replay
  transport each toggle from the View menu (Ctrl+1/2/3); the sidebar can also be
  dragged shut on its splitter. The layout holds together down to ~1092x614
  logical pixels (a 1366x768 display at 125% DPI).
- **Branded launch**: an animated Gold Quant splash (drawn in code from the theme
  palette - no image assets) covers the data load, and the same GQ coin mark is
  the window/taskbar icon and, via `scripts/build_app.py`, the packaged
  executable's icon.

## Architecture

- `src/app/sim/` - the flexible-exit engine. Reuses the verified per-bar exit
  ordering and adds trailing/breakeven/custom-time exits. Under the frozen config it
  reproduces `strategy_lab.simulate_positions` to 1e-12
  (`tests/test_app_flex_exit_tieback.py`).
- `src/app/analysis/` - Qt-free compute shared by the visuals: placed-trade sizing
  and recompute, excursion geometry, what-if runs, exit-grid sweeps, and the
  plain-language explanation engine. All unit-tested.
- `src/app/datalayer/` - BarStore (narrow GC load, Dev/Val cap, viewport slicing,
  optional `date_floor` fast path), VWAP reconstruction, and the replay and
  forensics services. No Qt dependency.
- `src/app/ui/`, `src/app/workers/` - the PySide6 widgets (theme, chart, exit panel,
  blotter, forensics panel, heatmap, replay animator, main window) and thread-pool
  workers.

## Deferred

A statistics view (deflated Sharpe / CSCV / equity curves), MGC and multi-instrument,
multiple simultaneous positions, and a per-strategy (rather than per-entry) exit-grid
sweep. The finplot backend is an optional future enhancement behind the same
`ChartWidget` interface.
