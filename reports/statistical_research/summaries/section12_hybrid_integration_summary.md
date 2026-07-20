# Section 12 Hybrid Integration Summary

**Status:** READY

## Question and governance

Does the frozen Branch B opportunity model add stable out-of-sample value beyond the Branch A True POI baseline, used as a filter (PRD Hybrid Mode B)? Both branches were complete and frozen before the integration boundary was crossed. Rules frozen in `Section12Config` (seed 20260726): outcome is Branch A's headline metric (capped 60-minute R, robust-capped at 5R); the gate is the saved Section 10 model applied unchanged - the full-coverage gate table is verified to reproduce the saved Development/Validation predictions exactly before use (population-matched per-date rank z-scores; the verification check caught and forced that population match); verdicts use Development and Validation only with a date-block bootstrap on the pooled gated-minus-baseline improvement and pre-declared floors (500/300 gated events, 150/100 trading dates); the Final test is read once, separately, after verdicts are fixed.

## Event population

- 358,072 POI outcome rows; 98.1 percent joined to the frozen gate at decision-bar identity (unjoined rows fail the shared complete-label rule).
- 27,000 Development, 22,738 Validation, and 301,610 Final-test events.
- Gate pass rate on Development POI events: 5.1 percent - POIs form in extended conditions where the model forecasts below-median relative expansion, so the frozen 80th-percentile threshold passes few of them.

## Result

| Family | Dev improvement | Val improvement | Verdict |
|---|---:|---:|---|
| continuation long | -0.073 | -0.160 | NO_INCREMENTAL_VALUE |
| continuation short | +1.170 [CI 0.09, 2.37] | +0.875 (retention 0.75) | NO_INCREMENTAL_VALUE (sample floors) |
| reversal long | -1.676 | -0.988 | NO_INCREMENTAL_VALUE |
| reversal short | -0.713 | +0.505 | NO_INCREMENTAL_VALUE |

Continuation short is the instructive case: dramatic Development and Validation improvements sat on only 410 and 241 gated events (33 and 30 trading dates), below the pre-declared floors, so it was not advanced.

**One-time frozen Final-test read (301,610 events, after verdicts were fixed):** the continuation-short improvement collapses to -0.024R (gated 0.176 versus baseline 0.200); no family exceeds +0.044R. The small-sample effect did not generalize - precisely the outcome the floors guarded against.

**Decision: the gate-filter integration form adds no confirmed incremental value to any POI direction family.** This closes PRD Phase 3 for Mode-B-as-filter with a defensible rejection. Open integration forms that remain unexplored: opportunity-based sizing, no-trade suppression in low-opportunity regimes, and feature-level interactions between POI context and the frozen statistical set.

## Verification

- 7/7 structural checks (verdicts computed from Development+Validation only; Final-test report structurally separated; frozen-gate reproduction enforced with an error).
- 11/11 new synthetic tests: planted gate effect advances only its family; null gate never advances; mutating Final-test outcomes cannot change verdicts; decision-bar join exactness; partition normalization; quintile monotonicity; robust cap; determinism.
- Full project suite green. Analysis runtime about 6 seconds.

## Saved outputs (excluded from Git)

`hybrid_family_results_gc.parquet`, `hybrid_quintile_results_gc.parquet`, `hybrid_verdicts_gc.parquet`, `hybrid_final_test_report_gc.parquet`, and the `tables/section12/` CSVs.

SECTION 12 STATUS: READY
