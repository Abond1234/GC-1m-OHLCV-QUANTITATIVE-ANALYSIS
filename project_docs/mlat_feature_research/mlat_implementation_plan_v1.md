# MLAT Implementation Plan v1

## Architecture

```text
trusted GC bars + eligible observation IDs
    -> GC-only, outcome-free feature construction
    -> registry/causality/boundary/save-reload validation
    -> frozen feature Parquet
    -> Development+Validation label join
    -> univariate, stability, overlap, and incremental evaluation
    -> GARCH audit and simple-anchor comparison
    -> feature verdicts and multivariate authorization gate
```

The notebook is an orchestrator. Typed reusable logic lives in new modules:

- `mlat_feature_registry.py`
- `mlat_feature_engineering.py`
- `mlat_feature_validation.py`
- `mlat_feature_evaluation.py`
- `mlat_volatility_models.py`
- `mlat_artifacts.py`

## Gates

1. Verify every upstream file against shape, schema fingerprint, timestamp
   range, IDs, partition coverage, and hash.
2. Instantiate and validate the frozen MLAT registry before constructing
   predictors.
3. Load only a strict GC causal-source whitelist; legacy forward columns never
   enter memory in the engineering module.
4. Construct rolling state on the full chronological GC sequence with the
   established continuity run; map only decision-bar values afterward.
5. Validate registry alignment, row order, timestamp mapping, finite values,
   ranges, missingness, duplicates, constants, and deterministic persistence.
6. Save and reload the feature matrix before loading labels.
7. Build an evaluation frame containing Development and Validation only.
8. Evaluate direction, expansion, future realized volatility, and path-risk
   targets using daily rank IC, date-block intervals, Development-fitted
   buckets, multiple-testing correction, session/year stability, and thinning.
9. Audit exact/near/monotonic overlap and partial information beyond `atr_20`
   and the frozen expansion set.
10. Audit the old GARCH implementation, reconstruct only a GC-only segmented
    diagnostic model, and compare it with simpler volatility anchors.
11. Authorize no multivariate work unless a small non-redundant shortlist
    survives every prior gate.
12. Execute the notebook in a fresh kernel, run new and affected tests, render
    final reports, and review GitNexus change impact.

## Performance policy

Vectorized NumPy/Pandas/PyArrow operations and shared rolling primitives are
used over bars. Python loops are limited to small collections such as
continuity segments, dates, features, model candidates, or bootstrap batches;
there is no Python loop over all minute rows. Float64 is retained for
calculations, with output downcasting only after validation.

