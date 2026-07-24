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

```
python -m pip install -r requirements.txt -r requirements-app.txt
python scripts/run_trade_simulator.py
```

`requirements-app.txt` (PySide6, pyqtgraph, and pyinstaller for packaging) is kept
out of the pinned research core so the deterministic install and CI are unaffected.
PySide6 6.11 ships an abi3 wheel and installs on the repo's Python 3.14 venv.

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
  - **Bar-by-bar replay**: animate a trade from entry to exit with play/step/speed.
  - **Forensics panel**: a plain-language verdict, a per-horizon MFE/MAE mini-chart,
    and labelled entry-context meters.
- **Light / dark** theme (View menu).

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

## Verifying the UI without a display

`scripts/render_screens.py` builds the real `MainWindow` under a narrow bar load and
writes PNG screenshots, so UI changes can be reviewed headlessly. It uses the OS
default platform (real fonts where a window station exists); set
`QT_QPA_PLATFORM=offscreen` for a truly headless machine.

## Packaging a standalone build

```
python scripts/build_app.py
```

Produces `dist/GCTradeSimulator/` via PyInstaller. The build does **not** bundle the
multi-hundred-MB parquet data (that stays in the repo per the data governance): run
the executable from the repo root, or set `GC_PROJECT_ROOT` to a checkout so it can
find `data/processed`. `dist/`, `build/`, and the generated `*.spec` are git-ignored.

## Deferred

A statistics view (deflated Sharpe / CSCV / equity curves), MGC and multi-instrument,
multiple simultaneous positions, and a per-strategy (rather than per-entry) exit-grid
sweep. The finplot backend is an optional future enhancement behind the same
`ChartWidget` interface.
