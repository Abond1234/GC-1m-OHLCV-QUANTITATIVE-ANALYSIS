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

Research validity is instrument-scoped: the strategy catalog and replay markers
are GC evidence, so they disable - with the reason on screen - on every other
instrument. Free-play, exits, what-ifs, exit grids,
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

- **Workspace shell**: the chart stays central; Indicators and Drawing remain in
  collapsible left workspaces (Ctrl+1/2). The resizable Simulator sidebar owns
  Strategy, Risk, Prop firm, and Performance. The global toolbar carries only
  the instrument, chart timeframe, overlay popover, and Simulator toggle.
- **Full-history chart and simulation dates**: the chart exposes the complete
  loaded Development + Validation history through cached, buffered viewport
  tiles, defaulting to a daily overview. Zoomed-out views use display-only
  aggregation and automatically restore the selected detail while zooming in.
  Start and End sit above the shared Simulate button and inclusively filter
  strategy entry dates. Simulation, entries, exits, and replay stay on true
  1-minute bars.
- **Chart**: GC candlesticks with volume hidden by default (an Indicators
  show/hide toggle), New
  York session shading, and one tick-formatted price scale on the right
  (left-drag to scale, double-click to auto-fit) carrying a last-price pill
  that rides the replay tape. The chart header holds the O/H/L/C-and-change
  readout, one chip per active study with hide/remove controls, and compact
  Undo/Redo/Clear icons - none of which can overlap. The crosshair pins an
  exact date/time pill beneath the vertical line and a tick-snapped price pill
  on the scale. Candles never draw across a continuous-segment break.
- **Indicator studies**: a searchable Available list activates SMA, EMA,
  Bollinger Bands, ATR, RSI, rolling volatility, OBV, volume MA and the three
  VWAP variants; each active study has parameter, colour, hide and remove
  actions, and only active studies are computed. Bounded studies open an
  oscillator pane that exists only while needed and honours the replay
  curtain. Studies are display aids at the displayed timeframe - never inputs
  to fills - and persist in session files.
- **Measurement**: Shift+Left-drag (or the Measure tool) reads signed price
  change, ticks from the instrument's tick size, percent, bars and elapsed
  time; Esc cancels, release commits, Undo removes. Values are identical at
  any zoom.
- **Trade inspector** (right): a single click on a strategy trade shows a compact
  overview; a double-click opens the full detail (chart focus, forensics verdict,
  horizon chart, entry context); Escape returns to the full chart.
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
- **Drawing tools** (Drawing workspace, grouped): Lines - Trendline, Ray, Info
  line (with a live measurement readout), Extended line, Trend angle,
  Horizontal line, Horizontal ray, Vertical line, Crossline; Position - Long
  position and Short position (entry/target/stop handles, shaded reward/risk
  zones, a live R:R label); Zones; and the Fibonacci set - Fib retracement,
  Trend-based fib extension, Fib channel, Fib time zone, Fib speed resistance
  fan, Trend-based fib time, Fib circles, Fib spiral, Fib speed resistance
  arcs, Fib wedge, and Pitchfan. Two-point tools place by drag or two clicks,
  three-point tools by three clicks with anchor dots; every tool has draggable
  anchors, right-click or Escape cancel, and Undo/Redo/Clear (Ctrl+Z/Ctrl+Y)
  from the chart header. Drawings live in data coordinates (exact under any
  zoom), survive timeframe switches, persist in sessions, and stay visible
  above the replay curtain.
- **Cross-asset comparison** (toolbar Overlay/Compare): pick a second instrument
  and a mode - Horizontal or Vertical split (each pane keeps its own price scale,
  the panes X-linked for time sync) or Normalized (the comparison rebased to the
  primary's first close and overlaid without distorting the primary scale). The
  current GC trade is ghosted onto a split pane by timestamp.
- **Prop-firm evaluation** (Prop firm workspace): a dollar account model (GC
  $100/pt or MGC $10/pt, risk-percent or fixed-contract sizing) turns the blotter
  into an equity curve with desk stats, and a fully adjustable prop-firm policy -
  profit target, daily-loss limit, static or trailing drawdown, minimum and
  maximum evaluation days, a consistency rule, a max-contracts cap, and a payout
  split - is judged over the session: the pass/fail/in-progress verdict, the first
  breach with its reason and date, days and trades to pass, payout eligibility
  with the simulated payout and post-payout balance, per-rule meters, and equity
  and rule-utilization curves. Simulated fills on historical Dev+Val data -
  explicitly not live results.
- **Indicators**: a curated set of chart-only trend, volatility, momentum,
  volume, daily-VWAP, and session-VWAP studies. Nothing is active at startup;
  adding, hiding, or removing a study never changes research or simulation
  inputs.
- **Sessions (File menu)**: save/load the working session (trades by entry and
  exit config, drawings, view, account and evaluation settings) as versioned
  JSON under the git-ignored ``reports/sessions/``. Loaded trades are
  re-simulated through the verified engine - results are never read from disk.
- **Light / dark** theme (View menu). Every surface re-themes in place, including
  the heatmap's colour scale and the chart axes/crosshair.
- **Collapsible layout**: the nav rail (Ctrl+1-5) opens or collapses each left
  workspace - re-clicking the open one returns the width to the chart - and the
  View menu toggles the trade inspector and the replay transport. Collapsing
  never loses chart, strategy, risk, or replay state. The layout holds together
  down to ~1092x614 logical pixels (1366x768 at 125% DPI).
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
  and recompute, excursion geometry, what-if runs, exit-grid sweeps, the
  plain-language explanation engine, the indicator-study engine, measurement
  arithmetic, and Fibonacci tool geometry. All unit-tested.
- `src/app/datalayer/` - BarStore (narrow GC load, Dev/Val cap, viewport slicing,
  optional `date_floor` fast path), VWAP reconstruction, and the replay and
  forensics services. No Qt dependency.
- `src/app/ui/`, `src/app/workers/` - the PySide6 widgets (theme, chart, exit panel,
  blotter, forensics panel, heatmap, replay animator, main window) and thread-pool
  workers.

## Deferred

Deflated-Sharpe / CSCV statistics for the session view, multiple simultaneous
positions, and a per-strategy (rather than per-entry) exit-grid sweep. The
finplot backend is an optional future enhancement behind the same `ChartWidget`
interface. (Formerly deferred and since shipped: the MGC mirror pane, the
multi-instrument layer, session equity curves, and the evaluation-account view.)

## Simulator workspace draft (September 2026)

The black **Simulator** button at the top right opens a sidebar with Prop firm,
Risk, Strategy, and Performance tabs. Drag its left divider to widen it; Close
and reopen retain the selected tab, settings, and width. Indicators and Drawing
remain on the left (Ctrl+1/2). Ctrl+Shift+S toggles Simulator.

The main chart has no Start/End controls: it loads and displays the complete
Development + Validation history, defaulting to a daily overview. The only date
selectors sit above Simulate and remain visible across every Simulator tab. They
define an inclusive strategy-entry period for each simulation and default to the
full loaded Dev+Val range.

**Simulate** first runs the selected strategy through the verified replay engine,
then applies the configured challenge rules to those closed-trade outcomes. Only
one account is active at a time. Reaching the trader daily profit/loss stop pauses
that account for the day and rotates the next eligible account; a firm daily or
overall loss breach retires it. Three consecutive account breaches stop the
simulation for that NY trade date.

Prop firm defines the pool size, original account balance, one- or two-step phase
targets and loss limits, and a funded payout target constrained to 3-5% of the
original balance. A successful phase advances and resets the account; the final
challenge phase upgrades it to funded, and the funded target records a payout.
Risk contains only risk per trade, trader daily profit target, trader daily max
loss, and an optional trade cap. The former custom-exit, trailing, breakeven,
frozen-contract, free-play, what-if, and exit-grid controls are not part of the
Simulator workflow.

Performance reports phase passes, challenges passed, payouts, breaches, trades,
skipped setups, three-breach stop days, gross strategy metrics, simulated P&L,
and an account-by-account summary. Its trade table directly shows the account,
stage, P&L, and account event; it no longer nests the old Free-play/What-if/Exit
grid/Trade detail/Free-play account tabs.

## Packaged review workflow (September 2026)

After app changes, run the full tests and lint, then rebuild with
`.venv/Scripts/python.exe scripts/build_app.py`. Open
`dist/GCTradeSimulator/GCTradeSimulator.exe` to review the updated app; keep its
`_internal` folder alongside it. No uninstall/reinstall is required. Startup now
defaults to light mode; the View menu still offers both themes.

The build runs PyInstaller in a fresh subprocess with a Windows/Python-only PATH
and clean dependency analysis. This prevents unrelated tools' DLLs (notably a
Poppler ICU library incompatible with Qt) from contaminating the package. Always
smoke-test the packaged executable, not just the source launcher.
