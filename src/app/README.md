# GC Trade Simulator (app)

A native desktop app (PySide6 + pyqtgraph) for charting the GC dataset, replaying
any catalog/peer strategy, and placing trades with **flexible, non-frozen exits**
to understand why a trade worked or failed. It reuses the verified research engine
in-process, so a simulated fill is exactly what the research engine would record.
Development + Validation only; the Final-test partition is never loaded or rendered.

Status: **backend complete and tested; UI is a first runnable version pending live
verification on a machine with the GUI deps installed.**

## Install and run

```
python -m pip install -r requirements.txt -r requirements-app.txt
python scripts/run_trade_simulator.py
```

`requirements-app.txt` (PySide6, pyqtgraph, finplot) is kept out of the pinned
research core so the deterministic install and CI are unaffected.

## What it does

- **Chart**: GC candlesticks with volume, a recomputed VWAP overlay, and New York
  session shading, one day at a time; candles never draw across a continuous-segment
  break.
- **Replay**: pick a strategy (the 12-rule library or the ~200-strategy catalog),
  plot its trades coloured by outcome, and click any trade to redraw its exact path
  and read its forensics.
- **Free-play**: enable free-play, click the chart to place an entry (filled on the
  next bar), size the exit freely (ATR stop, R target, trailing stop, breakeven), and
  watch the outcome path and excursions.
- **Forensics**: for a selected trade, the realised R, holding, and within-trade MFE/MAE,
  plus - for a registered observation - the forward-label excursions and the causal
  feature context at entry (VWAP distance, volatility regime, trend quality, session
  position, relative volume).

## Architecture

- `src/app/sim/` - the flexible-exit engine. Reuses the verified per-bar exit ordering
  and adds trailing/breakeven/custom-time exits. Under the frozen config it reproduces
  `strategy_lab.simulate_positions` to 1e-12 (`tests/test_app_flex_exit_tieback.py`).
- `src/app/datalayer/` - BarStore (narrow GC load, Dev/Val cap, viewport slicing), VWAP
  reconstruction (validated against the stored distance feature), and the replay and
  forensics services. No Qt dependency; fully unit-tested.
- `src/app/ui/`, `src/app/workers/` - the PySide6 UI and thread-pool workers.

## Manual smoke checklist (run after installing the app deps)

App launches and loads; candles render for a Development day; pan/zoom is smooth; no
candle draws across a segment break; VWAP and session shading toggle; pick a strategy →
Replay → markers appear → click a trade → path and forensics populate; enable free-play,
click an entry, set a 1.5 ATR stop / 2R target → path and exit reason draw; toggle
trailing/breakeven and watch the stop track; confirm the date list never reaches 2025.

## Deferred

Toward notebook parity and the post-approval repo restructure: a statistics view
(deflated Sharpe / CSCV / equity curves), an installer/build, MGC and multi-instrument,
and multiple simultaneous positions. The finplot backend is an optional future
enhancement behind the same `ChartWidget` interface.
