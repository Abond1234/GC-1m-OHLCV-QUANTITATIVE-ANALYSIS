# Section 7 Univariate Feature Evaluation Summary

**Status:** READY

## Scope and governance

- Evaluation partitions: **Development + Validation only**. All **162,224 Final-test observations were excluded** when the evaluation frame was built; no feature-to-outcome relationship on the Final test was computed anywhere in Section 7.
- Evaluation frame: **419,393 observations** (302,774 Development, 116,619 Validation; 157,527 London, 261,866 New York; 884 trading dates). 4,913 Development/Validation rows were dropped because at least one horizon label was unavailable or ATR normalization was missing, so the same sample supports every horizon comparison.
- Evaluated predictors: **83** (80 numeric + 3 boolean). The 2 categorical context fields are excluded with recorded reasons (`entry_session` is a stratification dimension of the screen itself; Section 5 approved no weekday filter).
- Evidence unit: **per-NY-date cross-sectional Spearman IC** with date-block bootstrap intervals (seed 20260720, 2,000 replicates) — minute observations are correlated events, never independent trades.
- Multiple testing: **Benjamini–Hochberg q-values on the Development screen only**, within outcome-family × session families of 498 related tests each.
- All shortlist criteria were frozen in `Section7Config` before any result was computed; verdicts are mechanical.

## Outcome families

- `direction` — signed `forward_return_{h}_atr` (long/short framings are exact algebraic pairs per Section 5)
- `expansion` — unsigned `future_range_{h}_atr` (opportunity/volatility forecasting)

## Headline result

| Verdict | Features |
|---|---:|
| ADVANCE_DIRECTIONAL | **0** |
| ADVANCE_EXPANSION | **55** |
| WEAK_UNSTABLE | 28 |
| NO_EVIDENCE | 0 |

- **No feature earned a directional advancement.** Nothing cleared the full gate stack (Development q ≤ 0.10, sample/date floors, Validation sign agreement with ≥ 25% IC retention, |bucket monotonicity| ≥ 0.8, and a ≥ 2-tick Q-top-minus-Q-bottom spread in both partitions) for signed returns. This is consistent with the Section 5 baseline: unconditional gold minutes are near coin flips, and univariate OHLCV state does not overcome that.
- **Expansion structure is strong, stable, and heavily confirmed.** 55 features advanced for opportunity/volatility forecasting, led by volatility-state and session-clock families. The strongest cells reach |daily IC| ≈ 0.61–0.76 with Development→Validation retention near or above 1.0 and monotone quintile structure, e.g. `atr_20` vs 180-minute ATR-relative future range (Dev IC −0.734 / Val −0.763 New York; −0.744 / −0.723 London).
- 358 of 1,992 confirmation cells passed all criteria (session- and horizon-specific rows).

## Interpretation caveats (recorded before any later use)

1. **Normalization direction.** The expansion outcome is measured in units of decision-bar ATR. High current volatility predicting a *lower* ATR-relative future range reflects volatility mean-reversion plus the normalization denominator — it does not mean high-vol minutes move less in absolute ticks. Any sizing/no-trade rule built later must restate the effect in the units it will trade.
2. **Redundancy is visible and expected.** The session-clock features (`minutes_to_noon_entry_cutoff`, `minutes_to_1530_forced_exit`, `minute_from_execution_window_open`, `session_progress_fraction`) carry identical |IC| by construction, and the ATR/realized-volatility ladder is highly self-correlated. Section 8 redundancy clustering must reduce these to interpretable representatives before any multivariate step.
3. **Statistical vs economic significance.** With ~420k correlated observations, many tiny effects are BH-significant on Development; 28 features passed the Development screen somewhere yet failed confirmation (`WEAK_UNSTABLE`). The economic tick gate and retention requirement — not q-values — are what separated the shortlist.
4. Expansion advancement is **not** a trading edge. It is evidence that opportunity size is forecastable, useful later for sizing, regime, no-trade, target and holding-time decisions under the roadmap's signal-construction rules.

## Validation

- **12/12 structural checks passed** (partition containment, no Final-test rows, expected cell counts, q-values confined to the Development screen, shortlist consistency, no forward-looking column among predictors, IC bounds).
- **18/18 new synthetic engine tests passed**; the full project suite is **112/112**. Synthetic coverage includes: planted signal detected and noise rejected under BH; Development-only bin fitting (shifted Validation data cannot move edges); bootstrap determinism by seed; hand-checked BH q-values; Final-test rows raising an error; session-confined signals staying session-confined; the economic spread gate blocking statistically-strong but economically-thin direction cells.
- Engine determinism: rerunning the evaluation reproduces the shortlist exactly (fixed seeds).
- Runtime ≈ 125 s for the full screen on this host; peak working memory ≈ 1.4 GB.

## Saved outputs

All generated outputs remain excluded from Git per the project's data-governance boundary:

- `data/processed/statistical_research/univariate_results_gc.parquet` (3,984 screen cells)
- `data/processed/statistical_research/univariate_bucket_summary_gc.parquet` (18,888 bucket rows)
- `data/processed/statistical_research/feature_stability_gc.parquet` (7,968 yearly rows)
- `data/processed/statistical_research/candidate_feature_shortlist_gc.parquet` (358 cells)
- `reports/statistical_research/tables/section7/section7_candidate_feature_shortlist_gc.csv`, `section7_feature_verdicts_gc.csv`

This summary is the tracked record of the milestone.

## Exact next step

**Section 8.0 — Redundancy and Incremental Information**: correlation/clustering of the 55 expansion advancers, interpretable representatives per family, incremental-value tests, then a frozen candidate set. The Final test remains locked until that shortlist freeze.

SECTION 7 STATUS: READY
