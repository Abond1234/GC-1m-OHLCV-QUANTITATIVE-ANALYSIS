# Section 9 Multivariate Research Summary

**Status:** READY

## Scope and governance

- Question: does combining the Section 8 frozen 15-feature expansion set beat the anchor `atr_20` alone?
- Population: the Section 7 evaluation frame (419,393 observations, Development + Validation only; Final-test rows rejected).
- Inputs and outcomes enter as per-date cross-sectional rank z-scores, so the regression objective matches the daily-IC evidence unit used since Section 7. Models are fitted per session.
- Ridge strength is selected by expanding-window walk-forward inside Development only (year folds 2021 to 2022 and 2021-22 to 2023, one-trading-date embargo exceeding the longest 180-minute label). Validation is touched once per model after the selection is frozen.
- All rules frozen in `Section9Config` (seed 20260722) before computation. Tree models are deferred by contract until linear benchmarks earn them; the decision is recorded in the outputs.

## Regression result (Validation, base configuration)

| Session | Horizon | Model IC | Anchor IC | Improvement | 95% CI | Verdict |
|---|---:|---:|---:|---:|---|---|
| London | 60m | 0.593 | 0.548 | +0.045 | [0.028, 0.062] | MODEL_ADVANCES |
| London | 180m | 0.732 | 0.723 | +0.008 | [0.003, 0.013] | ANCHOR_SUFFICIENT |
| New York | 60m | 0.626 | 0.545 | +0.081 | [0.064, 0.097] | MODEL_ADVANCES |
| New York | 180m | 0.849 | 0.761 | +0.088 | [0.071, 0.104] | MODEL_ADVANCES |

The advancement margin (0.02 minimum improvement with the bootstrap interval of the paired daily difference above zero) is cleared in three of four cells. London 180m shows a real but economically thin improvement and is honestly recorded as anchor-sufficient.

## Classification result (Validation)

- Logistic AUCs: London 0.800/0.803, New York 0.809/0.928 at 60m/180m; the model exceeds the anchor AUC in every cell and improves every Brier score.
- Calibration: rank-correct for London (both horizons) and New York 180m. **New York 60m overstates its upper probability deciles** (predicted 0.65 to 0.82 against realized 0.41 to 0.55). These probabilities require recalibration before any sizing use; the flag is carried in the summary.

## Verification

- 9/9 structural checks passed (partitions, dev-only walk-forward, embargo on every fold, lambda from the declared grid, complete-row fraction below 5 percent, AUC bounds, verdict coverage).
- 12/12 new synthetic tests passed: a planted second driver makes the model advance; an anchor-only world yields ANCHOR_SUFFICIENT everywhere (the overfitting guard); coefficients favor true drivers; chronological folds with embargo verified; logistic recovery on separable data; Final-test rejection; determinism.
- Runtime about 6 seconds; peak working memory about 0.7 GB. Full project suite green.

## Saved outputs (excluded from Git)

`multivariate_regression_gc.parquet`, `multivariate_walk_forward_gc.parquet`, `multivariate_coefficients_gc.parquet`, `multivariate_classification_gc.parquet`, `multivariate_calibration_gc.parquet`, `multivariate_verdicts_gc.parquet`, and the `tables/section9/` CSVs.

## Exact next step

Section 10.0 - Statistical Signal Construction: formalize the candidate family the evidence supports (benchmark directions gated by the frozen opportunity model) for the Section 11 sequential test.

*Follow-up (2026-07-20): executed as declared - see `section10_signal_construction_summary.md`.*

SECTION 9 STATUS: READY
