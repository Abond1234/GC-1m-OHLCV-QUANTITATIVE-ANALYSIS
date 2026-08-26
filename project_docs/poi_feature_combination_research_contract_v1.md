# POI Feature Combination Research Contract v1

Status: FROZEN BEFORE COMBINATION OUTCOME ACCESS

Research ID: `poi_feature_combination_v1`

Contract date: 2026-08-26

Repository baseline: `9b2fbe8`

## 1. Research question and claim boundary

This study asks two ordered questions:

1. Can previously researched, causally available features add stable directional or
   reaction-magnitude information to a True POI event at a next-bar decision?
2. If the POI-conditioned models fail, can the same previously frozen non-POI feature
   groups produce a general 60-minute directional model?

This is a new research family. It does not reopen, relabel, or tune the archived
`S7P02_NY_BEAR_CONT` family. It cannot authorize live trading, MGC execution, or a
production system. A passing result may receive only
`RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA`.

The following are not valid conclusions from this study:

- A forecast of future range is not a directional edge.
- Event-level return is not a sequential portfolio return.
- GC signal quality does not establish an MGC fill.
- A positive Development or 2024 result is not live-trading authorization.

## 2. Data boundary and partitions

Permitted outcome partitions are:

- Development: through 2023-12-31.
- Retrospective Validation: 2024-01-01 through 2024-12-31.

Every read of a table that can contain outcome values must use a physical partition or
date predicate that excludes 2025 onward. The runner must fail if a materialized row has
`trade_date_ny >= 2025-01-01`, a Final-test partition label, or a post-2024 timestamp.
Timestamp cutoffs are compared in UTC for UTC timestamp columns; the independent New
York trading-date cutoff is also enforced.

The 2025+ Final-test partition is not read, summarized, plotted, or used for any verdict.
There is no one-time Final-test read in this contract.

The study uses the following frozen inputs read-only:

- `data/processed/section7_true_poi_context_frame_gc.parquet`
- `data/processed/statistical_research/eligible_observations_gc.parquet`
- `data/processed/statistical_research/forward_labels_gc.parquet`
- `data/processed/statistical_research/feature_matrix_gc.parquet`
- `data/processed/statistical_research/frozen_expansion_feature_set_gc.parquet`
- `data/processed/statistical_research/fes_project1/v1/scalar_features_development_gc.parquet`
- `data/processed/statistical_research/fes_project1/v1/scalar_features_validation_sealed_gc.parquet`
- the corresponding frozen feature registries, manifests, and tracked summaries.

Input SHA-256 values, schemas, row counts, and the exact projected columns are recorded
before model fitting. Existing POI, Branch B, FES, MLAT, and Tsay modules and verdict
artifacts are not modified.

Byte-level SHA-256 hashing and Parquet-footer schema/total-row inspection are
non-materializing provenance operations and are exempt from the row-predicate rule.
They may not inspect values or partition statistics. Reported analytic row counts are
only the authorized physically filtered Development/Validation counts; no Final-test
count or distribution is reported.

## 3. Decision timing, entry, and targets

The decision timestamp is the close of completed POI retest bar `t`. All predictors must
be available no later than that close. Entry is the statistical research convention:
the open of bar `t+1`. This permits touch-close POI features without lookahead and avoids
pretending that a feature observed at bar close was known at the intrabar first contact.

The primary horizon is exactly 60 minutes. A row is usable only when the existing
60-minute label is available without shortening or crossing a prohibited boundary.

Primary directional targets:

- POI branch: `signed_continuation_return_60_atr = poi_direction_sign *
  forward_return_60_atr`, where bullish is `+1` and bearish is `-1`.
- POI branch economic target: the same orientation applied to
  `forward_return_60_ticks`.
- General branch: `forward_return_60_atr` and `forward_return_60_ticks` from the long
  perspective.

For the POI branch, a positive prediction means continuation and a negative prediction
means reversal. For the general branch, a positive prediction means long and a negative
prediction means short.

The secondary, non-authorizing opportunity target is `future_range_60_atr`. Opportunity
models are evaluated for information only and can never create trade direction.
Each named directional ridge is fitted only to its branch's ATR-normalized directional
target. Tick returns are economic-evaluation fields, never fit targets. A parallel
`OPPORTUNITY_DIAGNOSTIC` ridge may be fitted to `future_range_60_atr` under the same
feature ladder, but it is separately keyed, cannot enter selection, and cannot supply a
trade sign.

Exactly-zero realized directional returns are excluded from sign accuracy and balanced
accuracy. A positive prediction is positive; a zero or negative prediction is negative.
Balanced accuracy is undefined when only one realized class remains, and every
undefined gate statistic fails closed.

## 4. POI population and deterministic deduplication

The primary POI population is the New York execution-session True POI retest population
that joins exactly from `retest_bar_id` to the eligible observation
`decision_bar_id`. London POI results are a coverage diagnostic only because prior
Development date support is below the authorization floor.

The POI artifact's lowercase partition labels are mapped one way as
`development -> Development` and `validation -> Validation`; `final_test` is rejected.
The bridge is an inner join. Unmatched POI candidates are excluded and counted; join
coverage below 99% is an integrity failure. After the bridge, timestamp, New York date,
session, and canonical partition equality are asserted before observation-ID joins.
POI direction is mapped strictly as `bullish -> +1` and `bearish -> -1`; any other or
missing value is an integrity failure.

Only one POI opportunity may represent a decision bar. Before any outcome column is
read, candidates sharing `decision_bar_id` are ordered by:

1. `feat_first_touch=True` first;
2. `feat_15bar_structural_validation=True` first;
3. smaller finite `feat_poi_age_minutes` first, missing last;
4. smaller finite `feat_poi_width_atr` first, missing last;
5. lexicographically smaller `true_retest_id`.

The first row is retained. This rule is outcome-free and fixed. Repeated retests and all
same-date observations remain together in chronological folds.

## 5. Frozen feature groups

### 5.1 POI context group

`POI_CONTEXT_V1` contains exactly:

- `feat_poi_width_atr`
- `feat_confirmation_directional_close_location`
- `feat_displacement_efficiency`
- `feat_displacement_speed_to_approach_speed_ratio`
- `feat_approach_15m_directional_efficiency`
- `feat_approach_15m_range_compression_ratio`
- `feat_approach_15m_candle_overlap_ratio`
- `feat_approach_15m_relative_volume`
- `feat_touch_penetration_fraction`
- `feat_touch_rejection_wick_to_body_ratio`
- `feat_distance_from_vwap_atr`
- `feat_short_to_long_volatility_ratio`
- `feat_previous_touch_count`
- `feat_poi_age_minutes`

These cover formation, displacement, approach, touch, market state, and retest history.
They are not selected again from new outcomes.

### 5.2 Frozen Branch B opportunity group

`STAT15_V1` contains the exact fifteen rows marked `in_frozen_set=True` in
`frozen_expansion_feature_set_gc.parquet`:

- `atr_20`
- `minute_from_execution_window_open`
- `minutes_to_1530_forced_exit`
- `atr_ratio_20_60`
- `volume_acceleration_5_20`
- `efficiency_ratio_60`
- `return_sign_change_rate_30`
- `ols_r_squared_30`
- `atr_ratio_5_20`
- `efficiency_ratio_30`
- `relative_volume_20`
- `vwap_elasticity_30_exp`
- `current_range_over_atr`
- `choppiness_14`
- `efficiency_ratio_15`

This group previously predicted opportunity, not direction.

### 5.3 Frozen FES advancers

`FES4_V1` uses only the four previously frozen session-specific feature cells:

- London: `return_acf_energy_60`, `range_volume_spearman_30`.
- New York: `lagged_volume_return_spearman_30`, `volume_profile_slope_30`.

Only the two authorized features for the model's session enter that session model.

### 5.4 Predeclared POI-coordinate interactions

`POI_INTERACTIONS_V1` contains exactly:

- `int_poi_direction_x_lagged_volume = poi_direction_sign *
  lagged_volume_return_spearman_30`
- `int_poi_width_x_atr_ratio = feat_poi_width_atr * atr_ratio_20_60`
- `int_displacement_x_efficiency = feat_displacement_efficiency *
  efficiency_ratio_30`
- `int_approach_overlap_x_sign_change = feat_approach_15m_candle_overlap_ratio *
  return_sign_change_rate_30`
- `int_touch_penetration_x_current_range = feat_touch_penetration_fraction *
  current_range_over_atr`
- `int_approach_compression_x_acf = feat_approach_15m_range_compression_ratio *
  return_acf_energy_60`
- `int_retest_volume_x_range_volume = feat_approach_15m_relative_volume *
  range_volume_spearman_30`
- `int_poi_vwap_x_trend = poi_direction_sign * feat_distance_from_vwap_atr *
  normalized_ols_slope_30`

Interaction source values are causal completed-bar values. Raw products are formed
in float64 before the training-only preprocessing step. These are standalone engineered
products rather than hierarchical regression interactions: both parent main effects are
not required. All eight products are intentionally present in both session models;
session-specific FES authorization applies to raw FES main effects, not to these
predeclared products.

### 5.5 Excluded confirmatory groups

- MLAT v1 is catalogued but excluded because 0/12 features were authorized and its
  thinning gate was structurally non-evaluable.
- Tsay T01-T09 are catalogued but excluded because Stage 4 selected no predictive
  architecture, froze no policy, and left Validation unopened.
- The GARCH feature is excluded because the fitted specification was nonstationary or
  nonconvergent and its hypotheses were not evaluated.
- The old Section 10 opportunity score is excluded from executable rules because its
  same-date cross-sectional normalization is not point-in-time live information.

## 6. Frozen model ladder

All continuous models are deterministic ridge regressions with intercept and fixed
`alpha=10.0`, `solver="svd"`, float64 inputs, unweighted rows, and the repository's
pinned scikit-learn version. There is no hyperparameter search. Models are fitted separately by
session. Training-only preprocessing replaces nonfinite values with the training
median, scales by training median and interquartile range, uses scale 1 when the IQR is
not positive, and clips transformed values to `[-10, 10]`. Median and IQR are computed
from finite training values before imputation. A training feature with no finite value
is an integrity failure. Quartiles use NumPy's `method="linear"`; positive or negative
infinity is treated as missing, there are no missing indicators, and Validation uses
the one preprocessing state refit on all Development rows.

POI models:

- `POI1_CONTEXT`: `POI_CONTEXT_V1`.
- `POI2_CONTEXT_STAT15`: POI context plus `STAT15_V1`.
- `POI3_CONTEXT_FES4`: POI context plus session-authorized FES features.
- `POI4_CONTEXT_INTERACTIONS`: POI context plus the eight interactions.
- `POI5_ALL`: POI context, STAT15, session-authorized FES, and all interactions.

General non-POI models:

- `GEN1_STAT15`: `STAT15_V1`.
- `GEN2_FES4`: the two session-authorized FES features.
- `GEN3_STAT15_FES4`: the union.

No tree, boosting, neural, stepwise, interaction-mining, or feature-elimination model is
permitted in v1.

## 7. Chronological Development OOF protocol

For each session, sort unique Development New York trading dates. Starting after the
first 126 dates, insert a one-whole-date embargo and assess the next 42 dates. Advance
by 42 assessment dates until fewer than 21 assessment dates remain. A valid model needs
at least four assessment folds. Fold dates and a canonical SHA-256 are persisted before
labels are joined.

In zero-based index notation, fold origin `o` starts at 126 and advances as
`o = 126 + 42*k`: training dates are `dates[:o]`, embargo is `dates[o:o+1]`, and
assessment is `dates[o+1:min(o+43,n)]`. A final partial assessment block is retained
only when it has at least 21 dates.

Every preprocessing parameter and ridge fit is training-fold only. The OOF prediction
for a row is produced only by a model that did not train on that row's date. Validation
is evaluated once using preprocessing and coefficients refit on all Development rows.

## 8. Metrics and inference

For each model, target role, session, and partition report:

- observations and valid New York trading dates;
- mean, standard deviation, median, p05, and p95 target;
- daily cross-sectional Spearman IC using dates with at least ten finite pairs;
- ordered-date stationary-bootstrap 95% interval for mean daily IC;
- Pearson correlation as a secondary diagnostic;
- MAE and RMSE;
- directional sign accuracy and balanced accuracy for the directional target;
- paired daily-IC difference and interval versus `POI1_CONTEXT` or `GEN1_STAT15`;
- Development-to-Validation IC retention;
- best-ten-date positive-evidence concentration;
- for nonanchors, best-ten-date positive paired-IC-delta concentration;
- coefficient sign and magnitude tables.

Stationary bootstrap uses restart probability `0.20`, 2,000 replicates, and seed
`20260826`. The date is the dependence unit. No iid-row p-value is authoritative.
Evaluation-date counts use all dates with at least one finite prediction/target pair;
IC-date counts separately use the dates with at least ten finite pairs. The model gate's
date floor is the evaluation-date count. IC means and intervals use only IC dates.
Model IC concentration is the sum of the ten largest positive daily ICs divided by all
positive daily IC; paired concentration uses the analogous common-date IC delta. A zero
positive-evidence denominator is undefined and fails closed.

## 9. Development selection and Validation confirmation gates

A directional model is Development-eligible only if all apply:

- minimum four OOF folds;
- at least 180 valid OOF dates for POI or 300 for the general branch;
- at least 3,000 OOF rows for POI or 50,000 for the general branch;
- mean daily IC at least 0.02;
- stationary-bootstrap IC interval lower bound above zero;
- sign accuracy at least 0.51;
- best-ten-date positive-evidence concentration at most 0.50;
- for an additive model, paired mean daily-IC improvement over the branch anchor at
  least 0.01 with interval lower bound above zero.

Selection uses Development only. Among eligible models choose the highest Development
OOF mean daily IC. A model within 0.005 of the best loses to the earlier model in the
listed simplicity order. If no model qualifies, freeze `FROZEN_NO_DIRECTIONAL_MODEL`.

Selection is per `(branch, session)`. London POI is diagnostic-only and never
selectable; New York is the sole POI selection cell. For the general branch, London and
New York are selected independently on Development. The general branch's global policy
is the chronological union of whichever frozen session selections independently pass
Validation; it is not allowed to replace a failed session/model after seeing Validation.
The listed model orders are the intentional simplicity priorities even though predictor
counts are not monotone. `within 0.005` means an absolute difference less than or equal
to 0.005.

POI2 through POI5 owe paired improvement versus POI1. GEN3 owes paired improvement
versus GEN1. GEN2 is a nonnested alternative and does not owe the additive paired gate.
Every paired delta uses the exact intersection of finite, IC-eligible dates for the
model and anchor.

The frozen Development selection confirms on retrospective Validation only if:

- at least 120 valid dates and 2,000 rows for POI, or 150 dates and 40,000 rows for the
  general branch;
- Validation mean daily IC is positive and its interval lower bound is above zero;
- its sign agrees with Development and absolute IC retention is at least 0.25;
- sign accuracy is at least 0.51;
- for an additive model, paired Validation improvement over the anchor is at least
  0.005 and nonnegative at the interval lower bound;
- best-ten-date positive-evidence concentration is at most 0.50.

No replacement model is selected after Validation.
If Development selects no model for a cell, that cell's Validation outcomes remain
unopened. If Development selects a model, Validation scores only that frozen model and
its required anchor/comparator. No other ladder model receives a Validation diagnostic.

## 10. Diagnostic fixed-horizon policy

For every directional model, fit session-specific prediction thresholds from its
Development OOF predictions: the noninterpolated 10th and 90th percentiles using the
existing one-based `ceil(q * n)` order statistic. On
Development OOF and Validation, trade only predictions at or beyond those frozen
thresholds.

An upper-tail trade additionally requires a strictly positive prediction; a lower-tail
trade requires a strictly negative prediction. The policy is disabled when `p10 >= 0`,
`p90 <= 0`, or `p10 >= p90`. Threshold equality is included once these validity checks
pass, and nonfinite predictions never trade.

- POI prediction at or above p90: trade POI continuation.
- POI prediction at or below p10: trade POI reversal.
- General prediction at or above p90: long.
- General prediction at or below p10: short.
- Entry: recorded next-bar open.
- Exit: recorded fixed 60-minute exit.
- One global position per branch; overlapping candidates are skipped chronologically.
- An entry exactly equal to the previous exit is treated as overlapping and skipped.
- Same-timestamp ties use observation ID, then True retest ID where present.
- No stop or target is inferred from a fixed-horizon label.

Report frictionless, base 2.6-tick, and pessimistic 4.6-tick round-trip costs. Report
trades, dates, mean and median gross/net ticks, bootstrap interval, win rate, profit
factor, total net ticks, daily-net-tick Sharpe, maximum drawdown, and best-ten-date PnL
concentration. Eligible dates with no trade enter the daily Sharpe and drawdown series
as zero. Profit factor is positive gross profit divided by absolute gross loss and is
reported as infinity when there is positive profit and no loss, and undefined when
there is neither profit nor loss.
Point mean net ticks is total net ticks divided by trades. Its interval resamples ordered
evaluation dates in stationary blocks and recomputes the ratio. Maximum drawdown is the
minimum drawdown of cumulative daily net ticks. Null tie keys are integrity failures.

Economic authorization requires the frozen selected model to have at least 300/200
trades and 100/60 dates in Development OOF/Validation, positive base-cost mean net
ticks with interval lower bound above zero in both partitions, positive daily Sharpe in
both, and best-ten-date PnL concentration at most 0.50. Pessimistic results must be
shown but are not an additional pass gate.

If the predictive or economic gate fails, the exact policy verdict is
`NO_TRADING_SYSTEM_AUTHORIZED`.

The branch order is binding. The POI Development selection and, when applicable, its
single Validation/economic confirmation are persisted to a POI checkpoint first. The
general branch may open only if that checkpoint does not say
`RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA`. Both branches may be reported, but the
general branch can never be chosen merely because its already-observed result looks
better.

## 11. MGC and forward-data boundary

No MGC policy is run unless a GC model passes every predictive and economic gate. Even
then, prior FR-09 evidence prevents assuming a GC boundary price fills on MGC. The only
permitted proposed execution convention is synchronized MGC next-bar-open entry with
future telemetry-measured costs; it is not tested in this notebook.

Any research candidate requires at least 60 new trading dates and 100 shadow trades
collected after this contract date under the unchanged specification before deployment
review.

The reused STAT15 and FES4 groups were partly chosen with earlier 2024 evidence, so 2024
is retrospective reuse rather than independent confirmation here. The eight tested
directional model specifications have no multiplicity correction. These limitations
are reported prominently and enforce the forward-data claim cap.

## 12. Completion states

The notebook must end with one of:

- `RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA`
- `PREDICTIVE_ONLY_NOT_ECONOMIC`
- `FROZEN_NO_DIRECTIONAL_MODEL`
- `NO_TRADING_SYSTEM_AUTHORIZED`
- `BLOCKED_INTEGRITY_FAILURE`

`STATUS: READY` means the governed analysis completed. It does not mean a strategy
passed.

State precedence is: integrity failure first; then a fully predictive and economic
branch becomes `RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA`; a predictive model that
fails economics becomes `PREDICTIVE_ONLY_NOT_ECONOMIC`; no Development-eligible model
becomes `FROZEN_NO_DIRECTIONAL_MODEL`; every other completed failure becomes
`NO_TRADING_SYSTEM_AUTHORIZED`.

The POI target is a terminal 60-minute continuation/reversal return and the opportunity
target is total future range. Neither by itself proves a causal reaction at the level.

## 13. Frozen I/O and artifact namespace

The exact outcome-bearing projection is limited to:

- eligible observations: `observation_id`, `decision_bar_id`,
  `decision_timestamp_utc`, `entry_timestamp_utc`, `trade_date_ny`, `entry_session`,
  `research_partition`;
- 60-minute labels: those same identifiers plus `exit_timestamp_utc_60`,
  `forward_return_60_ticks`, `forward_return_60_atr`, `future_range_60_atr`, and
  `label_available_60`;
- statistical feature matrix: identifiers plus STAT15 and
  `normalized_ols_slope_30`;
- FES split tables: identifiers plus the four frozen FES source fields needed by
  session main effects/interactions;
- POI context: bridge/identity/date/session/partition/direction fields, the 14 POI
  context features, `feat_first_touch`, and `feat_15bar_structural_validation`.

No other horizon, MFE, MAE, stop, target, final-test report, or outcome column is read.
Every outcome-bearing Parquet read must pass a physical Development/Validation or
pre-2025 predicate and a projected column list. Candidate metadata is read and
deduplicated before the outcome loader is called.

Outputs live only under
`data/processed/statistical_research/poi_feature_combination/v1`,
`reports/statistical_research/poi_feature_combination/v1`, and the one new notebook.
They include access/hash manifests, the pre-outcome fold manifest, population and join
audits, model/daily-IC/coefficient tables, frozen selection checkpoints, policy trades
and metrics, and a final headline/checkpoint. Validation access is a one-time locked
batch after Development selection; no Validation statistic may change a frozen model or
threshold.
