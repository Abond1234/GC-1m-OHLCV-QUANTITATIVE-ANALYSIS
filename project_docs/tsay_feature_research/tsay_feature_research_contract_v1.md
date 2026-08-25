# Project One Tsay Feature Research Contract v1

Status: frozen before outcome access

Research ID: `tsay_feature_research`

Implementation version: `v1`

Milestone branch: `feature-research`

Starting commit: `362f4061616ff2bfc4a3e40803419c5afd22f967`
Deterministic seed: `20260824`

This contract asks whether the nine prescribed, causally implementable Tsay-derived
states and the fixed simple-model ladder add stable GC directional or opportunity
information beyond Project One's existing anchors. MGC is execution-only and remains
closed unless a frozen GC directional policy passes every Development and locked
Validation gate. Honest universal rejection is a valid result.

The implementation prompt
`project_docs/tsay_project1_implementation_agent_prompt.md` is the controlling
specification. This contract resolves its Stage 1 declarations without changing a
formula, window, threshold, target, model, or gate. The PDF is traceability only and
must not be researched or used to reinterpret this contract.

## 1. Immutable mandate and data boundary

- Signal/features: active/front continuous GC only. MGC is prohibited as an alpha,
  discovery, orientation, tuning, fitting, or selection input.
- Frequency: one-minute OHLCV. UTC is the immutable join key; business time is
  timezone-aware `America/New_York`.
- A decision feature at `t` uses only information known at the close of completed bar
  `t`; theoretical entry is the true next valid bar open `t+1`.
- Every forward path begins at `t+1`, contains exactly its declared number of
  consecutive valid one-minute bars, and may not cross New York date, contract,
  instrument, continuous segment, gap, roll, tradability, liquidity, or 15:30 forced
  exit boundaries. Invalid paths are unavailable, never shortened.
- Rolling states are built on the chronological GC bar stream through 2024-12-31,
  within the canonical continuity run and New York date, then mapped to eligible rows.
  No price, volume, feature, filtered state, or label is forward-filled across a reset.
- POI context is `[01:00,12:00)` New York. London entries are `[03:00,06:00)`;
  `[06:00,07:00)` is closed; New York entries are `[07:00,12:00)`; no entry is allowed
  at or after 12:00; mandatory flat is 15:30; overnight holding is forbidden.
- London and New York are separate evidence/model populations. A later policy sequences
  both globally with at most one open position.
- Registered horizons are 5, 15, 30, 60, 120, and 180 minutes. The only selection
  horizon is 60. Thirty minutes is sign/stability diagnostic only. The other horizons
  are registered but excluded from selection and tuning.
- Development ends 2023-12-31. Validation is 2024-01-01 through 2024-12-31. No row
  dated 2025-01-01 or later may be loaded, counted, summarized, plotted, hashed,
  engineered, modelled, or inspected in this research line.
- Every repository time-series read uses `tsay_access.py`, pushes the immutable New York
  cutoff into the scan, asserts the returned maximum date, and logs source, mode,
  columns, predicate, row groups, rows, products, and returned dates. Product-bearing
  sources push `product == GC` before Stage 6. Productless time series require existing
  GC-only manifest proof. Metadata tables cannot be treated as observations.
- The FES Validation scalar file receives path/footer-schema inspection only until Stage
  5; no row may materialize earlier. Labels are schema/identity metadata only in Stage 1,
  unopened in Stage 2, Development-only in Stages 3-4, and Validation-only once in Stage
  5 after frozen hashes pass.
- Mixed-date upstream files never receive a whole-file content hash. Stage 1 hashes only
  predicate-returned identity metadata samples plus footer schema/file identity. The
  Development-only FES and through-2024 MLAT artifacts may be verified against their
  existing whole-file hashes. The sealed FES Validation file is not rehashed in Stage 1.
- Research numerics use deterministic CPU float64. Saved scalar feature columns become
  float32 only after validation. Random row splits and iid-row resampling are forbidden.
- Market data, Parquet/CSV results, figures, fitted models, caches, and other generated
  artifacts remain ignored. Source/notebook paths are repository-relative.

## 2. Frozen targets and claim classes

Schema-only Stage 1 verification requires these exact fields:

| Use | Exact column |
|---|---|
| Directional primary | `forward_return_60_atr` |
| Directional diagnostic | `forward_return_30_atr` |
| Opportunity primary | `future_range_60_atr` |
| Opportunity classifier | `expansion_label_60` |
| Secondary directional magnitude | `forward_return_60_ticks` |
| Secondary opportunity magnitude | `future_range_60_ticks` |

The tick fields are secondary 2.0/1.0-tick spread gates, not selection targets. Later
Development QA checks their rowwise consistency with the label artifact's decision ATR;
an aggregate ATR effect is never multiplied by an average ATR.

Claims remain separate: directional return information; opportunity/range/path-risk
information; executable positive expectancy. Only the third permits P&L or Sharpe
language, and only after the sequential policy gates.

## 3. Frozen logical registry and overlap resolution

Define `r_t = 10000*log(C_t/C_(t-1))`, `d_t=(C_t-C_(t-1))/0.10`, `g_t` as canonical
`current_range_over_atr`, and `ATRticks_t=atr_20/0.10`. Recomputed `atr_20`,
`current_range_over_atr`, and `realized_volatility_15` must equal their canonical
eligible-row values. Their canonical continuity-run resets remain unchanged; every new
candidate window also stays within one New York date.

| ID | Physical column | Role/sign | Exact frozen definition |
|---|---|---|---|
| T01 | `tsay_ar5_cumulative_forecast_60_from_120_bps` | directional/+ | Fit float64 AR(5) with intercept on 120 response rows ending at `t`, using 126 consecutive closes. Recursively forecast 60 returns from `[r_t,...,r_(t-4)]` and sum. Require complete rows, full rank, condition number at most `1e12`, and every companion eigenvalue modulus below `0.999`. No regularization/order search or coefficient repair. |
| T02 | `tsay_arch_lm_r2_120_l5` | opportunity/+ | Regress squared residuals from T01's shared AR fit on intercept plus five squared-residual lags over 115 rows; return ordinary R-squared in `[0,1]`. Constant squares/rank failure are missing. `LM=115*R2` and chi-square(5) p-value are diagnostic only. T02 may use a full-rank AR residual fit when T01's eigenvalue forecast gate withholds T01. |
| T03 | `tsay_roll_spread_proxy_120_ticks` | opportunity/+ | On 120 `Delta C`, use the window mean and `gamma1=(1/119)*sum((DeltaC_i-mean)*(DeltaC_(i-1)-mean))`. If `gamma1<0`, return `2*sqrt(-gamma1)/0.10`; otherwise missing. Companion `tsay_roll_spread_identified_120` is `1` identified, `0` valid-not-identified, `-1` unavailable. It is an OHLCV proxy, not observed spread/cost. |
| T04 | `tsay_one_minute_close_zero_change_fraction_60` | opportunity/- | Exact fraction of the last 60 completed tick changes equal to zero, requiring all 60. It is an aggregated-bar analogue, not a transaction no-change probability. |
| T05 | `tsay_loss_es_120_atr` | opportunity/+ | For 120 tick changes define `L=-d`, deterministically sort, average the largest 12 losses, and divide by positive current `ATRticks`. No quantile interpolation/distribution fit. |
| T06 | `tsay_es_tail_balance_120` | directional/+ | Average the largest 12 `d` values as `ES_gain` and largest 12 `-d` values as `ES_loss`; return `(ES_gain-ES_loss)/(|ES_gain|+|ES_loss|)`, or zero for an exactly zero denominator. |
| T07 | `tsay_volume_lead_return_impulse_120_l5` | directional/+ | For each lag 1-5 calculate Pearson correlation of `r_i` and `zV_(i-k)` over exactly 120 response timestamps within an exact 125-timestamp span; return the unweighted mean correlation times `zV_t`. Any missing pair or zero lag variance makes it missing; never search farther back. |
| T08 | `tsay_range_state_lead_absreturn_impulse_120_l5` | opportunity/+ | For each lag 1-5 calculate Pearson correlation of `abs(r_i)` and `zG_(i-k)` over the exact 125-timestamp span; return root-mean-square correlation times `zG_t`. The same completeness/variance rules apply. |
| T09 | `tsay_extreme_cluster_ratio_120_q90` | opportunity/- | Sort 120 `abs(d)` values; threshold is the 108th one-based order statistic; exceedances are strictly greater. Consecutive exceedances form a cluster. Return clusters/exceedances; ties leaving no exceedance are missing. This is a runs-based proxy, not an estimated extremal index. |

All nine candidates are frozen as `RELATED_BUT_MATERIALLY_DISTINCT`: no existing
formula matches in formula, window, timing, reset, target/protocol, or transform. No
canonical column is aliased and no prior locked verdict removes a hypothesis. The exact
logical-to-physical map is in `tsay_feature_registry.py`. The confirmatory family is
therefore 9 logical candidates in each of two sessions, exactly 18 primary tests.

T03's state one-hots use `IDENTIFIED` as reference; the two model columns represent
valid-not-identified and unavailable. They are context, not hypotheses, and T03 receives
no generic missingness indicator. `zV` and `zG` are persisted provenance intermediates,
not hypotheses.

### 3.1 Outcome-free clock profiles

For each full GC minute, `clock_bin=15*floor(New_York_minute_of_day/15)`,
`vraw=log1p(volume)`, and `graw=log1p(max(g,0))`. For Development date `D`, each bin's
mean/sample standard deviation uses only strictly prior Development dates. Require 30
prior dates, 300 observations, and positive standard deviation. Validation uses moments
frozen from all Development and never updates. Then `zV=(vraw-mean)/std` and
`zG=(graw-mean)/std`. This authorized covariate-only online transform is identical in
every OOF path and never consumes current/future dates or labels.

### 3.2 Partial-information controls

Take the union of the session D1/O1 anchor with these controls, deduplicate, then add
fixed 15-minute clock-bin dummies with one reference bin dropped:

| ID | Additional controls |
|---|---|
| T01 | `efficiency_ratio_30`, `return_acf_energy_60` |
| T02 | `realized_volatility_ratio_15_60`, `volatility_of_volatility_60`, `return_acf_energy_60` |
| T03 | `amihud_illiquidity_60`, `volume_per_tick_range`, `relative_volume_60` |
| T04 | `relative_volume_20`, `return_sign_change_rate_30`, `amihud_illiquidity_60` |
| T05 | `realized_semivariance_balance_60`, `ret_outlier_fraction_60`, `bipower_jump_ratio_60` |
| T06 | `ret_tail_balance_60`, `realized_semivariance_balance_60`, `ordered_draw_balance_30` |
| T07 | `lagged_volume_return_spearman_30`, `volume_price_alignment_10` |
| T08 | `range_volume_spearman_30`, `realized_volatility_15`, `volatility_of_volatility_60` |
| T09 | `ret_outlier_fraction_60`, `return_acf_energy_60`, `bipower_jump_ratio_60` |

Within each session/date use the exact common finite panel. Candidate, target, and every
continuous control become average percentile ranks `(average_rank-0.5)/n`; clock dummies
remain binary. Drop within-date zero-variance controls. Fit candidate-rank and target-rank
float64 OLS projections on the same intercept-plus-control matrix. Require at least
`max(30, retained_controls+10)` rows, full design rank, and condition number at most
`1e12`. Daily partial IC is the Pearson correlation of residuals. Recompute unadjusted IC
on this same panel before the 25% retention rule.

## 4. Frozen anchors and model input lists

Directional D1 base: `return_5m_atr`, `return_15m_atr`, `return_30m_atr`,
`normalized_ols_slope_30`, `close_location_value`, `signed_volume_proxy`,
`distance_from_research_day_vwap_atr`, `return_autocorrelation_15`. New York D1 adds
`lagged_volume_return_spearman_30`; London adds no FES scalar.

Opportunity O1 base: `atr_20`, `atr_ratio_20_60`, `atr_ratio_5_20`,
`current_range_over_atr`, `minute_from_execution_window_open`,
`minutes_to_1530_forced_exit`, `efficiency_ratio_15`, `efficiency_ratio_30`,
`efficiency_ratio_60`, `ols_r_squared_30`, `choppiness_14`,
`return_sign_change_rate_30`, `relative_volume_20`, `volume_acceleration_5_20`,
`vwap_elasticity_30_exp`. London adds `return_acf_energy_60` and
`range_volume_spearman_30`; New York adds `volume_profile_slope_30`.

- D0: zero and Development-training-mean diagnostics. D2: T01/T06/T07. D3: D1 union
  D2. D4: GC-only VAR(2) on `[r,zG,zV]`. D5: q10/q50/q90 regularized linear
  conditional quantiles on D3 union O3. D6: two sign-of-current-`return_1m_bps` TARX
  regimes on D3 inputs, only after D3 authorization.
- O0: `atr_20`. O2: T02/T03/T04/T05/T08/T09 plus the two nonreference T03 state
  one-hots. O3: O1 union O2. O4: repaired q90-q10 width from D5/O4. O5: `atr_20` plus
  four fold-local causal Kalman outputs.
- E0: Development training prevalence. E1/E2/E3: logistic models on O1/O2/O3.

The exact session-resolved ordered inputs are machine-frozen in
`tsay_feature_registry.py`; univariate outcomes cannot alter them.

Ridge/VAR alpha grid is `[1e-4,1e-3,1e-2,1e-1,1,10]`. Logistic uses L2,
`lbfgs`, no class weights, `max_iter=5000`, and C `[0.01,0.1,1,10]`. Smooth-quantile
alpha is `[1e-4,1e-3,1e-2]`, taus are 0.10/0.50/0.90, kappa is `1e-4` ATR, analytic
gradients and deterministic CPU L-BFGS-B use `maxiter=500` and gradient infinity norm
`1e-8`. Kalman log-Q/log-R fits use the three prescribed starts, bounds
`[v*1e-6,v*1e2]`, `maxiter=500`, and tolerance `1e-6`. D4 and T01 stability require
companion eigenvalue modulus below `0.999`. Quantile predictions are sorted rowwise
before metrics/policy use; pre-repair crossing is reported.

## 5. Frozen folds, preprocessing, and resampling

Stage 1 freezes exact fold-date lists and a canonical SHA-256 under the ignored artifact
root. Per session, outer folds have first 126 Development dates for training, one full
date embargo, 21-date assessment, 21-date step, expanding training, and complete blocks
only. Training labels whose horizon exit is on or after the first assessment decision
timestamp are purged later. `attainable_dev_oof_dates` is the union of complete outer
assessment dates, `minimum_dev_oof_dates=ceil(0.90*attainable)`, and attainable must be
at least 180. Inner folds use first 63 dates, one-date embargo, 21-date assessment,
21-date step, at least two complete folds, and identical purge logic.

Every training fold replaces nonfinite with missing; median-imputes continuous inputs;
adds one missingness indicator per continuous input except T03; robust-scales by median
and IQR with training standard deviation only if IQR is zero; retains a recorded zero
column if both are zero; clips to `[-20,20]`; and standardizes continuous targets then
inverts predictions, except D5/O4 targets remain in original ATR units. Assessment and
Validation receive frozen training transforms.

All IC and paired-improvement inference uses ordered New York dates, stationary
bootstrap restart probability `0.20`, 2,000 replicates, and seed `20260824`. Feature
orientation is applied before testing. Intervals are uncentered 2.5/97.5 percentiles.
One-sided null p-values use mean-centered date series and add-one correction `/2001`.
BH is applied to the exact 18-test feature family. Model families use synchronized
max-statistic bootstrap paths for D2-D5, O2-O5, and E2-E3. Date, dependent date-block,
and synchronized stationary resampling are allowed; iid observation resampling is not.

Within-model hyperparameters use a single common inner panel and synchronized bootstrap
indices. Select strongest regularization within one best-model bootstrap standard error;
if zero, require equality within `1e-12`. Architecture selection similarly chooses the
first model in simplicity order `D2<D3<D4<D5<D6`, `O2<O3<O4<O5`, or `E2<E3` within
one best-candidate bootstrap standard error. No other one-standard-error rule exists.

## 6. Feature evidence gates

For each candidate/session at its 60-minute role target: daily Spearman IC requires 10
observations; report mean/median/std/sign fraction/year stability/coverage/missingness;
apply frozen bootstrap/BH; fit noninterpolated Development quintile edges at one-based
`ceil(k*n/5)` and assign `1+count(value>edge)`; require oriented quintile monotonicity at
least 0.80; report top-minus-bottom ATR and tick effects; run the frozen partial-IC test;
and show 30-minute sign/stability only.

`DEV_SHORTLISTED` requires at least 200 valid dates and 10,000 observations; expected
sign; 60-minute BH q at most 0.10; oriented bootstrap interval excluding zero;
monotonicity at least 0.80; directional spread at least 2.0 GC ticks or opportunity
future-range spread at least 1.0 tick; at least 25% partial absolute-IC retention with
partial interval excluding zero; and neither one year nor ten best dates contributing
more than half of positive oriented IC. Otherwise the mechanical status is
`DEV_REJECTED`, locked-verdict reuse, or `NOT_EVALUABLE` with reason.

Validation retention requires at least 150 dates/4,000 observations, frozen sign, q at
most 0.10, oriented interval excluding zero, at least 25% of Development absolute mean
IC, unchanged-edge monotonicity at least 0.80, the same tick-spread and partial-retention
gates, nonnegative oriented mean IC in both 2024 halves, and ten-date concentration at
most 50%. No replacement is selected after Validation.

## 7. Model metrics and authorization gates

All comparisons use exact candidate/anchor matched rows. Each outer fold retains at least
90% of anchor-valid rows and at least 2,500 London or 4,000 New York rows. Aggregate OOF
retains at least 90%, 10,000 observations per session, and the required date count.
Continuous models report daily IC/interval, paired IC improvement/interval, MAE, RMSE,
OOS R-squared, date-balanced calibration intercept/slope, support, failures, fold and
coefficient stability, and ten-date concentration by session/year. Classifiers report
date-balanced ROC-AUC, PR-AUC, Brier, log loss, fixed-0.50 balanced accuracy/MCC,
calibration/reliability/ECE, class balance, and paired E1 improvement.

Continuous calibration is date-balanced weighted OLS of observed target on prediction.
Classifier calibration is unpenalized weighted logistic recalibration on clipped
`[1e-6,1-1e-6]` logits. ECE has ten deterministic equal-count bins ordered by probability
then observation ID. Coefficient stability excludes intercepts, follows frozen input
order, uses the model-specific flattening rule, requires median pairwise cosine at least
0.50 and sign agreement at least 0.70 for coefficients whose median absolute value
exceeds 0.05; no qualifying coefficient makes the stability gate non-evaluable.

A directional/continuous model passes Development only with positive mean OOF IC; mean
paired daily-IC improvement at least 0.01; paired interval lower bound above zero;
coverage/support gates; ten-date contribution at most 50%; positive improvement in at
least 70% of complete folds; calibration slope `[0.50,1.50]`; absolute intercept at most
0.10 ATR; every required fit finite and successful; coefficient stability; quantile
models additionally improve exact pinball loss at every tau by at least 2%, have 80%
coverage error at most 0.03, and zero crossings after repair; and family max-statistic
adjusted p at most 0.05.

E2/E3 pass Development only with coverage/support; per-date Brier at least 0.002 below
E1 with interval lower bound above zero; ROC-AUC improvement at least 0.01; nondecreasing
PR-AUC; lower log loss; ECE at most 0.05; calibration slope `[0.75,1.25]`; absolute
intercept at most 0.10; ten-date concentration at most 50%; and adjusted p at most 0.05.

Validation continuous retention requires positive IC; paired improvement at least 0.02
with lower bound above zero; 150 dates and 90% anchor support; calibration limits;
positive improvement in both 2024 halves; concentration at most 50%; unchanged quantile
conditions; and no numerical failure. Validation classifier retention keeps the
Development classifier limits plus both-half positivity, 150 dates/90% support, and no
failure. A failed frozen role ends; no replacement is allowed.

Residual Ljung-Box, squared-residual, ARCH-LM, and portmanteau tests use Holm family alpha
0.05 and remain adequacy diagnostics, not discretionary gates. Declared numerical,
stability, optimizer, eigenvalue, quantile, or Kalman failures are hard gates.

## 8. Conditional policy, economics, and MGC gates

No policy exists unless a directional model passes Development. Development policy uses
outer-OOF predictions only. Each outer assessment fold receives training-only session
p90 absolute-direction cutoff from complete inner-OOF predictions and, if a classifier
passes, training-only p80 expansion cutoff. Order statistics are one-based `ceil(q*n)`,
no interpolation, ties eligible, with at least 42 inner-OOF dates and 1,000 scores.
Variants are exactly `direction_only_p90` and, only if authorized,
`direction_p90_and_expansion_p80`. No alternative threshold, stop, target, horizon,
session/side subset, or interaction is permitted.

The additive Tsay simulator uses GC true `t+1` open; stop distance
`clip(1.5*atr_20,1.0,10.0)` price points without tick rounding; target 2R; 120-minute
maximum hold; 15:30 flat; one global position; stop-first same-bar ambiguity; full
boundary validation before sequencing; and costs 0.0, 2.6, 4.6 ticks per trade.

Development policy authorization requires minimum OOF date coverage, at least 500
sequential trades, positive gross mean R/trade, positive base-cost net mean R/trade and
mean daily net R including eligible zero-trade dates, mean-daily-net-R bootstrap lower
bound above zero, synchronized maximum-Sharpe adjusted p at most 0.05, at least 60%
positive qualifying quarters (at least ten dates each), positive-quarter concentration
at most 50%, and no boundary/conflict/cost/reproducibility failure. The exact daily-net-R
Sharpe is `sqrt(252)*mean/std(ddof=1)`. At most one policy freezes; direction-only wins if
within one best-variant bootstrap standard error. Otherwise `FROZEN_NO_POLICY` and all
Validation economics display `N/A - no executable policy`.

Validation GC economics opens only for an authorized frozen policy and requires at least
60 dates/100 trades, positive base-cost net R/trade and daily net R, daily-mean interval
lower bound above zero, 60% positive qualifying quarters, concentration at most 50%, and
no execution breach. Pessimistic costs are always reported.

MGC transfer remains `NOT_AUTHORIZED` unless the frozen GC policy passes every locked
Validation predictive and economic gate. If later authorized, execution is at synchronized
MGC native `t+1` open with exact matching delivery year/month parsed from `active_symbol`,
MGC tick size 0.10/value $1.00, inherited 0.0/2.6/4.6-tick proxy costs plus one-tick
sensitivities, and the exact coverage/basis/tracking/economics gates in the implementation
prompt. GC price containment is diagnostic only. FR-09's `G5_PROVISIONAL_FAIL` remains
acknowledged. No live or capital approval can result from bar-proxy evidence.

## 9. Prohibited actions

Do not inspect the PDF beyond identity/hash; generate alternative features; change any
formula/window/threshold/model/gate; use MGC for alpha; read any 2025+ row; inspect
Validation outcomes before Stage 5; read labels in Stage 2; shorten a path; bridge a
boundary; forward-fill; split rows randomly; resample iid rows; fit outside Development;
tune across horizons; choose order/lag/threshold after results; smooth Kalman states;
repair unstable AR/VAR coefficients; add model families; modify frozen notebooks,
registries, FES/MLAT/POI code or verdicts; calculate policy P&L without directional
authorization; call opportunity evidence directional; infer quotes/fills/impact from
OHLCV; force-add ignored artifacts; or describe a research candidate as profitable,
tradable, execution-ready, or approved.

## 10. Stage protocol and frozen Stage 1 disposition

Exactly one stage runs per invocation. Every fresh-kernel checkpoint separates integrity
`STATUS: READY|BLOCKED` from empirical `RESEARCH_VERDICT`. Stage 1 may access only mandate,
provenance, footer schemas, authorized identity/date/session metadata, registries, and
frozen fold dates. It cannot construct features or read an outcome value.

Stage 1's empirical disposition is `RESEARCH_VERDICT: NOT_AUTHORIZED`: no feature,
model, policy, Validation outcome, or MGC transfer has been evaluated. If every Stage 1
integrity gate passes, only Stage 2 outcome-free feature construction becomes authorized,
and only after the user says `continue`.
