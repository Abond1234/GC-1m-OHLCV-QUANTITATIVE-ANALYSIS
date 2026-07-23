# MLAT Implementation Patterns

## Adopted patterns

1. Discover the repository root from the notebook path/current parents; use
   only repository-relative artifact paths.
2. Load Parquet columns with
   `pyarrow.parquet.read_table(path, columns=...).to_pandas(ignore_metadata=True)`.
3. Add the raw table's physical row number as `source_row_id` before filtering
   to GC; map `decision_bar_id` by exact ID and timestamp equality.
4. Use a strict causal-source whitelist. The engineering function has no label
   argument and rejects names containing forward/outcome tokens.
5. Reuse the established continuity-run contract. Compute rolling arrays on
   the full GC sequence, then select eligible decision rows.
6. Use complete trailing windows and preserve null warm-up/boundary values.
7. Instantiate and validate a frozen registry before construction. Predictor
   columns in the saved matrix must equal registry order exactly.
8. Keep sensitive arithmetic in float64; validate, then save feature outputs as
   float32 where the registry permits.
9. Save to `data/processed/statistical_research/mlat_feature_research/v1`,
   reload through PyArrow, and verify IDs, order, schema, nulls, and
   deterministic sample hashes.
10. Join labels only after the feature artifact has passed reload validation.
11. Build one Development+Validation frame and raise if any other partition is
   supplied to evaluation.
12. Fit buckets on Development per session, apply unchanged to Validation, and
   use daily rank IC/date-block intervals rather than minute-row t tests.
13. Persist each result table and concise manifest under the versioned report
   root; keep the notebook as narrative/orchestration.

## Patterns rejected for v1

- Copying notebook cells or appending to the old feature registry.
- Loading all raw columns and dropping forbidden fields later.
- Row-wise Python loops across the minute table.
- Centered windows, whole-sample smoothers, backward fill, or full-series
  wavelet reconstruction.
- Joint GC/MGC thresholds or parameter fits.
- Random CV, unrestricted hyperparameter search, and model importance as proof.
- Hard-coded absolute paths, environment-specific Python assertions, or hidden
  reliance on an already-running notebook kernel.
- Calling old save helpers whose paths would overwrite historical artifacts.
