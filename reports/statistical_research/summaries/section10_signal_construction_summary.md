# Section 10 Signal Construction Summary

**Status:** READY

## What this section is, honestly

Section 7 approved no directional feature, so Section 10 does not invent a directional signal. It formalizes the candidate family the evidence supports for a sequential test, with every rule frozen in `Section10Config` (seed 20260723) before construction:

- Direction variants are declared benchmarks: `long_benchmark` and `short_benchmark`.
- The only evidence-based component is the opportunity gate: the Section 9 frozen per-session ridge model of 60-minute ATR-relative future range, thresholded at its Development 80th percentile (fitted on Development, applied unchanged to Validation).
- Stops: 1.5 times the decision-bar ATR(20m), clamped to 10 to 100 ticks. Targets: 2R. Holding cap: 120 minutes. Risk: one R per trade; contract-count sizing is an execution-phase concern and is recorded as deferred.
- Session, noon-entry, and 15:30 rules are inherited from the eligible-observation construction.

## Result

- 419,063 candidate observations (302,553 Development, 116,510 Validation); 85,365 pass the expansion gate.
- Development gate rates: 19.98 percent (London) and 19.99 percent (New York) against the declared 20 percent; Validation rates 21.9 and 21.0 percent as out-of-sample observations.
- Median stop 10.6 ticks; 46 percent of raw ATR stops hit the 10-tick minimum clamp (recorded).

## Verification

- 7/7 structural checks passed (partitions, dev-only gate fitting, gate rate at declared percentile, stop clamps, target scaling, entry prices, sessions).
- 8/8 new synthetic tests passed, including proof that mutating every Validation feature value leaves the frozen gate threshold untouched.

## Saved outputs (excluded from Git)

`signal_candidates_gc.parquet`, `signal_gate_thresholds_gc.parquet`.

## Exact next step

Section 11.0 - Independent Sequential Backtest of this candidate family under declared costs.

SECTION 10 STATUS: READY
