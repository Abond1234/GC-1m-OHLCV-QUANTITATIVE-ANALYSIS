# Quant Project 1 — Statistical Feature Research Context Report

> **Primary notebook**
>
> ```text
> C:\Users\abond\Desktop\WORK FILES\Systemic\Project 1\notebooks\exploration\statistical_feature_research.ipynb
> ```
>
> **Purpose of this document**
>
> This is the dedicated handoff document for the independent statistical-feature research branch of Quant Project 1. It contains only the project context, completed work, rules, and research standards needed for this notebook.
>
> The agent should use this document as its primary context. It should not need to read the full `exp1.ipynb` context report unless an exact historical implementation detail is genuinely required.

---

# 1. Agent Role

Act as a **Senior Quantitative Researcher and Quantitative Developer** working to professional quantitative-research standards.

The responsibility is not merely to produce code that runs. The work must be:

- Statistically defensible
- Free from lookahead leakage
- Computationally and memory efficient
- Reproducible
- Modular and maintainable
- Clearly documented
- Suitable for later integration into a serious research and backtesting pipeline

Priority order:

```text
1. Correctness
2. Statistical validity
3. Computational efficiency
4. Memory efficiency
5. Scalability
6. Readability
7. Maintainability
8. Simplicity
```

Do not blindly implement weak methodology. Challenge fragile feature definitions, arbitrary thresholds, excessive parameter searches, invalid validation methods, and conclusions unsupported by out-of-sample evidence.

Assume tables may contain several million rows. Avoid:

- Repeated full-table scans
- Repeated sorting and groupbys
- Unnecessary dataframe copies
- Large duplicated intermediate tables
- Row-wise Python loops over bars
- Recomputing reusable arrays
- Materializing hundreds of features before their definitions are justified

Before returning important code, review it for correctness, leakage, complexity, memory use, and scalability.

---

# 2. Mission of the New Notebook

The notebook must research market features **independently of the discretionary POI system**.

Its mandate is:

```text
Identify stable and economically meaningful relationships between information
available at time t and future intraday GC price behaviour, without using POI information.
```

The research should determine whether historical 1-minute OHLCV and market-time context contain useful information about:

- Future signed direction
- Future movement magnitude
- Volatility expansion
- Maximum favourable excursion
- Maximum adverse excursion
- Continuation versus reversal
- Trade opportunity quality
- Appropriate stop, target, holding-time, or sizing conditions later in the process

The long-term objective is to:

1. Find features that work independently.
2. Build and test a purely statistical signal or system.
3. Freeze the strongest independent features and rules.
4. Only then compare or combine them with the discretionary POI system in a separate integration phase.

This notebook is not a continuation of the POI event study. It is a new research branch built on the same trusted market data.

---

# 3. Independence Boundary

## 3.1 What may be reused

The statistical notebook should reuse:

- The completed data-cleaning work
- The liquidity-selected active-contract research table
- Existing rollover and tradability controls
- New York timestamp and session fields
- The established trading-time constraints
- Broad market-wide EDA findings
- Existing engineering standards

## 3.2 What must not be used during independent feature discovery

Do not load or use any POI-derived tables, labels, features, filters, rankings, or conclusions when creating or evaluating the independent statistical system.

Do not use:

```text
section6_* POI tables
section6b_* structural POI tables
section6c_* refined POI tables
section7_* POI event-study outputs
True POI identifiers
True retest identifiers
POI geometry fields
POI structural-break fields
POI entry or stop variants
POI candidate rankings
POI-derived short-side or long-side conclusions
```

The statistical branch must not inherit a directional bias from the POI branch. Long, short, continuation, reversal, and no-trade behaviour must be evaluated independently.

POI information may be introduced only in the later integration phase, after the independent statistical feature set and statistical strategy rules have been frozen.

---

# 4. Project Location and Environment

Project root:

```text
C:\Users\abond\Desktop\WORK FILES\Systemic\Project 1
```

Notebook:

```text
notebooks\exploration\statistical_feature_research.ipynb
```

Virtual environment:

```text
.venv-1
```

Active interpreter:

```text
C:\Users\abond\Desktop\WORK FILES\Systemic\Project 1\.venv-1\Scripts\python.exe
```

Existing core packages include:

```text
numpy
pandas
scipy
statsmodels
matplotlib
seaborn
pyarrow
databento
jupyter
```

Do not reinstall libraries, rebuild project folders, reload the original DBN archive, or repeat completed environment explanations unless an actual missing dependency or path problem is confirmed.

Use `pathlib.Path` and project-root-relative paths. Do not make notebook logic depend on an assumed current working directory.

---

# 5. Trusted Starting Dataset

Start directly from:

```text
data\processed\research_bars_gc_mgc_1m.parquet
```

Do not restart from raw Databento data and do not rerun the completed exploratory-data or active-contract construction pipeline.

Dataset summary:

```text
Total rows: 3,487,656
GC rows:    1,759,671
MGC rows:   1,727,985
```

The dataset already contains:

- One liquidity-selected active contract per product and time
- GC and MGC outright bars only
- No spread instruments
- Raw, non-back-adjusted tradable prices
- Rollover warning fields
- Tradability controls
- UTC and New York timestamps
- Session fields
- Historical return, range, volume, and volatility fields

Important existing fields include:

```text
ts_event_utc
ts_event_ny
trade_date_utc
trade_date_ny
day_of_week
hour_ny
minute_ny
session_label

is_roll_day
is_pre_roll_day
is_post_roll_day
roll_window_flag
roll_window_type
days_since_roll
days_to_next_roll
tradable_research_flag
```

The exact saved schema must be inspected before coding. Do not silently assume that a particular column name or dtype exists.

## 5.1 Critical forward-column safeguard

The trusted table was built for the wider project and may already contain earlier forward-return or outcome columns.

These columns are **not features**.

At load time:

1. Inspect all column names.
2. Identify every existing forward-looking or outcome-derived field.
3. Quarantine those fields from the feature frame.
4. Rebuild the independent notebook's own labels using the rules in this report.
5. Use earlier forward columns only for optional validation checks, never as predictors.

No field based on future price, future range, future excursion, target hits, stop hits, or later bars may enter the feature namespace.

## 5.2 Initial instrument scope

Use **GC as the primary Phase 1 research instrument**.

Reasons:

- GC is the established analysis instrument in Project 1.
- It avoids duplicating nearly identical tests across correlated GC and MGC series.
- It reduces computation and multiple-testing burden during discovery.
- MGC can later be used for transfer validation, execution mapping, or robustness after a GC shortlist has been frozen.

Do not claim cross-product robustness until the shortlisted GC features have actually been tested on MGC under a separately defined validation procedure.

---

# 6. Completed Work That Should Not Be Repeated

The following work is already complete:

```text
Raw DBN loading
DBN-to-Parquet conversion
Data-quality assessment
Contract inventory
Liquidity analysis
Active-contract selection
Rollover identification
Continuous active-contract research table construction
Session-aware calendar fields
Broad return, volatility, volume, and microstructure EDA
```

The new notebook should not repeat broad EDA unless a specific feature definition requires a targeted diagnostic.

## 6.1 Broad findings already established

### Data quality

- Core OHLCV data is suitable for research.
- The final research table contains no missing core OHLCV values.
- It contains no duplicate timestamp-product rows.
- Timestamps are timezone-aware and minute-aligned.
- Futures-session gaps require session-aware handling.
- Rollover and tradability flags are already available.

### Returns

- One-minute signed returns are centered near zero.
- Raw signed-return autocorrelation is weak and unstable.
- One-minute returns are strongly fat-tailed.
- Simple unconditional direction is not an established edge.

### Volatility

- Volatility clusters at minute and daily horizons.
- Volatility regime is informative about future movement magnitude.
- Volatility is not automatically a directional signal.

### Volume

- Volume is strongly related to absolute movement and bar range.
- Volume alone has weak signed-direction information.
- Volume should initially be treated as activity, opportunity, regime, or risk context unless independent directional evidence is found.

### Sessions

- New York has the strongest combined liquidity and volatility.
- London is meaningful but behaves differently from New York.
- Session-specific reporting is mandatory.

### Naive signals

- Simple unconditional breakout continuation was weak.
- Raw volume-spike continuation was weak.
- Large-move continuation versus reversal depended on context.

The new notebook should therefore move from trusted data into an observation-and-label framework, not another general market-description section.

---

# 7. Locked Trading-Time Rules

The independent system may use historical information from earlier bars, earlier sessions, and earlier dates, provided the information was available before the decision.

Candidate entries must initially be restricted to:

```text
London:   03:00-06:00 New York time
New York: 07:00-12:00 New York time
```

Use these as half-open entry intervals:

```text
London:   [03:00, 06:00)
New York: [07:00, 12:00)
```

Therefore:

- No entry during 06:00-07:00 New York time.
- No new entry at or after 12:00pm New York time.
- Positions initiated before noon may continue after noon.
- Every path must end no later than 3:30pm New York time.
- No overnight holding.
- Outcomes must not cross into the next New York trading date.

Use `tradable_research_flag` and the existing rollover controls. Feature windows and labels must not improperly cross:

- Contract changes
- Continuous-segment boundaries
- Invalid rollover periods
- New York date boundaries where the definition is same-day
- The mandatory 3:30pm exit boundary

London and New York must be evaluated separately before any combined headline result is reported.

---

# 8. Research Sequence

The correct sequence is:

```text
Trusted market data
→ eligible observation frame
→ forward outcome labels
→ simple baseline behaviour
→ controlled feature engineering
→ univariate evaluation
→ redundancy and incremental-value tests
→ multivariate research
→ statistical signal rules
→ independent sequential backtest
→ POI integration research
```

Do not begin by generating a large feature library.

The research must first define:

```text
What is the observation?
When is the decision made?
Where is the theoretical entry?
What future outcome is being predicted?
When is that outcome valid or unavailable?
What baseline must a feature beat?
```

---

# 9. Observation and Execution Convention

Use one clear convention throughout the notebook.

Initial convention:

```text
Feature timestamp: completed bar t
Feature information: bars at or before t only
Theoretical entry: open of bar t+1
Forward path begins: bar t+1
```

Entry eligibility is determined by the timestamp of bar `t+1`, not merely by the timestamp of the feature bar.

This prevents calculating a feature with the completed close of bar `t` and then pretending the strategy entered at that same close without delay.

Each eligible observation should contain at least:

```text
decision_timestamp
entry_timestamp
product
symbol / active contract
trade_date_ny
session
entry_eligible
rollover / tradability fields
continuous-segment identifier if available
```

Only rows with a valid next-bar entry and valid future path become label-eligible observations.

---

# 10. Forward Outcome Labels

The independent statistical system has no natural stop distance at the start.

Therefore, do **not** begin with R-multiple labels. An R label requires a defined risk unit, and choosing an arbitrary stop before the research would contaminate the label design.

Begin with raw and normalized market outcomes.

## 10.1 Initial fixed horizons

Use:

```text
5 minutes
15 minutes
30 minutes
60 minutes
120 minutes
180 minutes
```

Do not add many extra horizons until these are validated.

A fixed-horizon label is unavailable when the complete requested path would cross:

- 3:30pm New York
- The next New York date
- A contract change
- A continuous-segment break
- An invalid tradability boundary
- Missing required bars

Do not silently shorten a fixed horizon and call it the requested horizon.

A separate forced-exit-capped label family may be added later, but it must be clearly distinguished from fixed-horizon labels.

## 10.2 Forward signed returns

For each horizon, calculate:

```text
forward_return_<h>_ticks
forward_return_<h>_bps
forward_return_<h>_atr
```

The return should be measured from the theoretical entry at the open of bar `t+1` to the defined horizon exit.

## 10.3 Path-dependent excursions

For each horizon, calculate clearly named long and short path outcomes:

```text
mfe_long_<h>_ticks
mae_long_<h>_ticks
mfe_long_<h>_atr
mae_long_<h>_atr

mfe_short_<h>_ticks
mae_short_<h>_ticks
mfe_short_<h>_atr
mae_short_<h>_atr
```

Also consider:

```text
time_to_mfe_<h>
time_to_mae_<h>
```

Explicit long and short naming is preferred even when one side can be algebraically derived from the other.

## 10.4 Expansion and volatility labels

Also calculate:

```text
future_range_<h>_ticks
future_range_<h>_atr
future_realized_volatility_<h>
future_range_relative_to_current_atr_<h>
```

This separates directional forecasting from opportunity or volatility forecasting.

A feature may be useless for direction but valuable for:

- Identifying movement expansion
- Choosing when not to trade
- Adjusting position size
- Selecting targets
- Selecting holding time

## 10.5 Classification labels

Classification labels may later include:

```text
Future return above a positive cost threshold
Future return below a negative cost threshold
Future range above a development-fitted percentile
Large expansion event
Favourable excursion reached before adverse excursion
Continuation versus reversal
```

All percentile cutoffs, thresholds, scalers, and bins must be fitted on development data only.

---

# 11. Leakage and Availability Rules

Every feature row at timestamp `t` may use only information available at or before `t`.

Forward labels must begin at `t+1`.

Forbidden feature inputs include:

- Future returns
- Future high or low
- Future range
- MFE or MAE
- Target or stop outcomes
- Later session information
- Full-day statistics not yet known at time `t`
- Percentiles fitted using validation or final-test data
- Time-of-day baselines fitted using future years
- Centered rolling windows
- Backward-filled future information
- Labels from the POI branch

For each feature, document its availability time.

Session-resetting features must not use values from the future portion of the same session.

Rolling features must not cross contract or continuous-segment boundaries.

Missing history must remain missing or follow an explicitly justified policy. Do not conceal insufficient history through unsafe forward or backward filling.

---

# 12. Chronological Research Partitions

Use chronological partitions. Never randomly split individual minute rows.

Freeze the following initial project-wide split before feature evaluation:

```text
Development: start of eligible data through 2023-12-31
Validation:  2024-01-01 through 2024-12-31
Final test:  2025-01-01 through the dataset end on 2026-05-22
```

Rules:

- Fit feature bins on development only.
- Fit scalers and transformations on development only.
- Select thresholds on development only.
- Use validation to choose among pre-defined candidates.
- Keep the final test untouched until feature definitions, shortlist criteria, and model rules are frozen.
- Do not repeatedly inspect the final test and revise the feature set.
- Report sample counts and trading-date coverage in every partition.

For model research, use walk-forward or expanding-window validation with appropriate purging or embargo when labels overlap in time.

---

# 13. Baseline Behaviour Before Feature Engineering

Before evaluating any feature, establish what an eligible GC minute naturally produces.

Baseline tables should include, for every initial horizon:

```text
Observation count
Mean forward return
Median forward return
Positive-return rate
Return standard deviation
Mean absolute return
Mean MFE
Mean MAE
Future range
Relevant downside quantiles
```

Report by:

- Full eligible population
- London
- New York
- Year
- Time-of-day bin
- Day of week
- Development, validation, and final-test partition
- Long and short framing where relevant

The baseline defines what a feature must beat.

A feature is not useful merely because one bucket has a positive mean. The effect must be large enough to matter after realistic cost and slippage assumptions and stable enough to survive time and session splits.

At the feature stage, use explicit tick or basis-point cost thresholds rather than prematurely assuming a complete trade-cost model. Final transaction-cost assumptions belong before the independent strategy backtest.

---

# 14. Controlled Feature Engineering

Do not begin with hundreds of features.

Start with approximately **40-80 interpretable features** across controlled families. Add new features only when they represent a distinct hypothesis or improve incremental information.

## 14.1 Initial feature families

### Price, return, and momentum

Examples:

```text
return_1m
return_3m
return_5m
return_15m
return_30m
absolute_return
cumulative_return
momentum_acceleration
directional_streak
distance_from_rolling_high
distance_from_rolling_low
rolling_range_position
```

### Volatility, range, and compression

Examples:

```text
atr_5
atr_15
atr_30
atr_60
realized_volatility_5
realized_volatility_15
realized_volatility_30
current_range_over_atr
short_vol_over_long_vol
range_compression_ratio
volatility_percentile
```

### Candle geometry

Examples:

```text
body_size
body_to_range
upper_wick_to_range
lower_wick_to_range
close_location_value
candle_direction
inside_bar
outside_bar
```

### Volume

Examples:

```text
volume_1m
rolling_volume_mean
relative_volume
volume_zscore
volume_acceleration
volume_per_tick_range
volume_percentile_by_time_of_day
```

Time-of-day normalization is essential. Volume at 8:30am New York should not be compared directly with volume at 4:15am New York without an appropriate historical baseline.

### VWAP and price location

Examples:

```text
session_vwap
distance_from_vwap_ticks
distance_from_vwap_atr
vwap_slope
time_above_vwap
time_below_vwap
session_open_distance
session_high_low_position
```

All session-state fields must use only information available up to the decision bar.

### Trend, efficiency, and market state

Examples:

```text
efficiency_ratio
trend_slope
slope_normalized_by_atr
directional_persistence
rolling_autocorrelation
choppiness_measure
```

## 14.2 Feature batches

Build and validate features in batches rather than all at once:

```text
Batch A: price and momentum
Batch B: volatility, range, and compression
Batch C: candle geometry
Batch D: volume and time-of-day-normalized activity
Batch E: VWAP and session location
Batch F: trend, efficiency, and chop
```

A batch should pass leakage, distribution, missingness, and availability checks before the next batch is added.

---

# 15. Feature Registry

Maintain a formal feature registry. Do not allow the notebook to become an unexplained collection of columns.

Each feature should record:

```text
feature_name
family
description
lookback
input columns
availability timestamp
signed or unsigned
expected interpretation
session-resetting flag
minimum history
missing-value policy
normalization method
source function
version
```

Example:

| Feature | Family | Lookback | Type | Description |
|---|---:|---:|---|---|
| `return_15m` | Momentum | 15 bars | Signed | Return over the previous 15 completed bars |
| `relative_volume_30` | Volume | 30 bars | Unsigned | Current volume relative to its historical expectation |
| `vwap_distance_atr` | Location | Session | Signed | Price distance from current session VWAP normalized by ATR |
| `efficiency_ratio_20` | Trend | 20 bars | Unsigned | Net movement divided by total absolute movement |

Feature names should be stable and unambiguous. Do not use vague temporary names in saved research tables.

---

# 16. Univariate Feature Evaluation

Initially evaluate each feature independently.

## 16.1 Quantile analysis

Fit quantile boundaries on development data only, then apply those unchanged to validation and final test.

A default starting point is five buckets:

```text
Q1: lowest 20%
Q2
Q3
Q4
Q5: highest 20%
```

For each feature, bucket, horizon, session, and partition, report:

```text
Count
Trading-date count
Mean forward return
Median forward return
Positive-return rate
Mean MFE
Mean MAE
Future range
Relevant quantiles
```

## 16.2 What constitutes evidence

A promising feature should have:

- A sensible market interpretation
- Adequate observations and trading-date coverage
- Stable direction across time
- Stability across relevant sessions or a clear session-specific thesis
- Meaningful effect size
- Preferably monotonic or ordered bucket behaviour
- Value beyond transaction-cost thresholds
- Incremental information beyond closely related features
- Results not dependent on one exceptional year or a tiny bucket

Do not rank features by the best-looking bucket alone.

## 16.3 Statistical safeguards

Because minute observations and overlapping labels are highly correlated:

- Do not treat every row as independent evidence.
- Use trading-date block bootstrap confidence intervals.
- Report trading-date coverage.
- Consider non-overlapping or thinned samples as a sensitivity check.
- Use Benjamini-Hochberg false-discovery control for broad feature screens.
- Treat multiple horizons as related tests, not independent confirmations.
- Report effect sizes, not only p-values.

---

# 17. Redundancy and Incremental Information

After univariate evaluation:

1. Examine feature correlations.
2. Group highly related features.
3. Retain interpretable representatives.
4. Test whether a feature adds information beyond its family.
5. Reject duplicate transformations that tell the same story.
6. Build a small frozen shortlist before multivariate modelling.

Useful tools may include:

```text
Spearman correlation
Hierarchical feature clustering
Conditional bucket analysis
Nested linear or logistic benchmarks
Out-of-sample incremental score improvement
Permutation importance on validation only
```

Do not use tree-model importance as proof of economic value. A feature must remain understandable and stable under direct diagnostics.

---

# 18. Multivariate Research Standards

Multivariate modelling begins only after the univariate shortlist and redundancy review.

Recommended sequence:

```text
Linear regression benchmarks
Logistic classification benchmarks
Regularized linear/logistic models
Simple tree-based models
Walk-forward validation
Probability calibration
Stability and sensitivity checks
```

Requirements:

- Chronological splits only
- Scaling fitted on training data only
- No random row split
- Purging or embargo for overlapping labels
- Compare against unconditional and simple-feature benchmarks
- Report calibration, not only accuracy
- Report performance by year and session
- Prefer simple stable models over complex models with marginal gains
- Keep the final test locked until model and thresholds are frozen

The goal is not to maximize an in-sample machine-learning score. The goal is to find stable and economically usable structure.

---

# 19. From Features to a Statistical Signal

Only after features or models show stable out-of-sample value should the notebook define:

- Entry conditions
- Long versus short direction
- Confidence thresholds
- No-trade conditions
- Stop families
- Target families
- Maximum holding times
- Position sizing
- Trade-conflict and overlap rules

At that stage, R-multiple labels become meaningful because a stop or risk unit has been explicitly defined.

Avoid selecting a stop, target, or holding time from the final-test result.

---

# 20. Independent Backtest Boundary

The independent sequential backtest belongs after the signal rules have been frozen.

It must include:

- Next-bar or otherwise realistic execution
- Costs and slippage
- Rollover and tradability exclusions
- No new entries after noon
- Mandatory 3:30pm exit
- No overnight positions
- Overlapping-signal rules
- One-position or portfolio-state logic
- Trade log
- Drawdown and tail analysis
- Year and session breakdown
- Parameter sensitivity
- Walk-forward or locked final-test evaluation

Event-level forward labels are not a sequential backtest.

Do not claim strategy profitability from feature-bucket or event-level averages alone.

---

# 21. POI Integration Boundary

Integration is a later, separate phase.

The comparison should eventually include:

```text
Statistical system alone
POI baseline alone
POI filtered by statistical features
POI ranked by statistical features
Incremental value of the combined system
```

The independent statistical feature set must be frozen before loading POI tables. This prevents selecting features because they happen to improve the known POI sample.

The integration question is not merely whether combined performance is higher. It is whether statistical features add stable out-of-sample value beyond the POI baseline.

---

# 22. Recommended Notebook Structure

Use the following reader-facing structure.

```text
0.0 Statistical Research Mandate

1.0 Configuration and Research Environment
    1.1 Imports
    1.2 Paths
    1.3 Instrument and Time Settings
    1.4 Random Seeds and Display Settings

2.0 Load Trusted Research Data
    2.1 Load Final GC/MGC Research Table
    2.2 Confirm Schema and Data Types
    2.3 Confirm Timestamp and Time-Zone Handling
    2.4 Confirm Date Coverage
    2.5 Confirm Rollover and Exclusion Flags
    2.6 Quarantine Existing Forward-Looking Columns

3.0 Construct the Eligible Observation Frame
    3.1 GC Phase 1 Research Population
    3.2 London Execution Observations
    3.3 New York Execution Observations
    3.4 Next-Bar Entry Convention
    3.5 Entry Cutoff Rules
    3.6 End-of-Day Exit Constraints
    3.7 Missing-Bar, Contract, Segment, and Abnormal-Day Handling
    3.8 Save Eligible Observation Index

4.0 Construct Forward Outcome Labels
    4.1 Forward Returns
    4.2 Maximum Favourable Excursion
    4.3 Maximum Adverse Excursion
    4.4 Future Range and Volatility
    4.5 Directional and Expansion Labels
    4.6 Label Availability and Boundary Rules
    4.7 Label Validation
    4.8 Save Forward Label Table

5.0 Establish Baseline Behaviour
    5.1 Unconditional Forward Outcomes
    5.2 Forward Outcomes by Session
    5.3 Forward Outcomes by Time of Day
    5.4 Long and Short Symmetry
    5.5 Year and Partition Stability
    5.6 Transaction-Cost Thresholds
    5.7 Baseline Summary

6.0 Feature Engineering Framework
    6.1 Feature Registry
    6.2 Price and Return Features
    6.3 Volatility and Range Features
    6.4 Candle Geometry Features
    6.5 Volume Features
    6.6 VWAP and Location Features
    6.7 Session and Time Features
    6.8 Trend and Market-State Features
    6.9 Feature Validation
    6.10 Save Feature Matrix

7.0 Univariate Feature Evaluation
    7.1 Feature Distribution Checks
    7.2 Development-Fitted Quantile Analysis
    7.3 Predictive Relationship
    7.4 Monotonicity
    7.5 Long and Short Behaviour
    7.6 Feature Stability by Year
    7.7 Feature Stability by Session
    7.8 Multiple-Testing Controls
    7.9 Candidate Feature Ranking

8.0 Redundancy and Incremental Information
    8.1 Feature Correlations
    8.2 Feature Clustering
    8.3 Conditional Analysis
    8.4 Incremental Predictive Value
    8.5 Frozen Candidate Feature Set

9.0 Multivariate Research
    9.1 Linear and Logistic Benchmarks
    9.2 Regularized Models
    9.3 Tree-Based Models
    9.4 Walk-Forward Validation
    9.5 Probability Calibration
    9.6 Model Stability

10.0 Statistical Signal Construction
    10.1 Entry Conditions
    10.2 Direction Selection
    10.3 Confidence Thresholds
    10.4 Stop and Target Research
    10.5 Position Sizing
    10.6 Trade-Conflict Rules

11.0 Independent Strategy Backtest
    11.1 Cost and Slippage Assumptions
    11.2 Trade Simulation
    11.3 Performance Results
    11.4 Yearly and Session Breakdown
    11.5 Drawdown and Tail Analysis
    11.6 Sensitivity Tests

12.0 Integration Research
    12.1 Statistical System Alone
    12.2 POI Baseline Alone
    12.3 POI with Statistical Filtering
    12.4 POI with Statistical Ranking
    12.5 Incremental Value Assessment

13.0 Final Conclusions
```

This is the full roadmap. Do not attempt to implement all sections in the first milestone.

---

# 23. Immediate First Milestone

The first implementation should stop after the observation, label, and baseline foundation is validated.

Complete:

```text
0.0 Statistical Research Mandate
1.0 Configuration and Research Environment
2.0 Load Trusted Research Data
3.0 Construct the Eligible Observation Frame
4.0 Construct Forward Outcome Labels
5.0 Establish Baseline Behaviour
```

The milestone is complete only when:

- The trusted GC data loads correctly.
- Existing forward-looking fields are quarantined.
- Every eligible decision maps to a valid next-bar entry.
- Session and noon cutoffs are correct.
- Labels never cross invalid boundaries.
- Fixed-horizon availability is explicit.
- MFE, MAE, returns, and future-range labels pass synthetic and sampled manual checks.
- Baseline tables are produced by horizon, session, year, and partition.
- Output tables are saved.
- No feature-selection conclusions have been made prematurely.

Only after this foundation is approved should the first controlled feature batch begin.

---

# 24. Recommended Research Code and Output Locations

Keep the notebook beside `exp1.ipynb`:

```text
notebooks/
    exploration/
        exp1.ipynb
        statistical_feature_research.ipynb
```

Create a separate reusable-code area as definitions stabilize:

```text
src/
    statistical_research/
        __init__.py
        eligibility.py
        labels.py
        baselines.py
        price_features.py
        volatility_features.py
        candle_features.py
        volume_features.py
        vwap_features.py
        trend_features.py
        feature_registry.py
        evaluation.py
        validation.py
```

Do not create every module on day one. Prototype definitions in the notebook, validate them, and move stable repeated logic into reusable modules.

Recommended generated outputs:

```text
data/
    processed/
        statistical_research/
            eligible_observations_gc.parquet
            forward_labels_gc.parquet
            baseline_summary_gc.parquet
            feature_matrix_gc.parquet
            univariate_results_gc.parquet
            feature_stability_gc.parquet
            candidate_feature_shortlist_gc.parquet
```

Recommended reports:

```text
reports/
    statistical_research/
        feature_registry.csv
        figures/
        tables/
        summaries/
```

Generated Parquet outputs should remain excluded from Git. Commit and push meaningful notebook, source-code, test, and documentation milestones after they run successfully.

---

# 25. Validation Requirements

At minimum, build automated or explicit checks for:

```text
Unique observation identifiers
Sorted timestamps
Expected product population
No spread instruments
Entry timestamp strictly after decision timestamp
Entry timestamp in an approved execution window
No entry at or after 12:00pm New York
No label path after 3:30pm New York
No path across New York dates
No path across contracts
No path across continuous segments
No invalid rollover rows
Positive and correct horizon lengths
Unavailable horizons marked unavailable
No forward fields in the feature matrix
Development-only bin fitting
Stable row counts after joins
No accidental duplicate observations
```

Path-dependent label logic should be tested with small synthetic bar sequences where the correct result is known.

Sampled visual or tabular manual audits should complement automated checks.

---

# 26. Standing Instructions for the Agent

When working in `statistical_feature_research.ipynb`:

1. Start from `research_bars_gc_mgc_1m.parquet`.
2. Use GC for the first discovery phase.
3. Do not load POI research outputs.
4. Do not repeat broad EDA.
5. Define observations and outcomes before features.
6. Use completed bar `t` and next-bar-open execution.
7. Preserve London, New York, noon-cutoff, 3:30pm-exit, rollover, contract, and segment rules.
8. Quarantine all existing forward-looking columns.
9. Use chronological development, validation, and locked final-test partitions.
10. Fit all transformations and thresholds on development data only.
11. Begin with a controlled, interpretable feature set.
12. Evaluate stability, economic significance, and incremental value.
13. Treat overlapping minute observations as correlated.
14. Do not confuse event-level statistics with a sequential backtest.
15. Do not combine with POI information until the independent statistical branch is frozen.
16. Save reproducible outputs and reusable code at meaningful milestones.
17. Update this context report after major statistical-research decisions or completed sections.

---

# 27. Current Project State for This Notebook

```text
Trusted cleaned market data:                 COMPLETE
Active-contract and rollover construction:   COMPLETE
Broad market EDA:                            COMPLETE
POI research branch:                         SEPARATE — DO NOT LOAD
Independent statistical observation frame:   NOT STARTED
Independent forward labels:                  NOT STARTED
Independent baseline behaviour:              NOT STARTED
Independent feature engineering:             NOT STARTED
Independent feature evaluation:              NOT STARTED
Independent statistical backtest:            NOT STARTED
POI integration research:                    FUTURE PHASE
```

The correct next action is:

```text
Build Sections 0.0-5.0 of statistical_feature_research.ipynb,
validate the independent eligible-observation and label framework,
save the resulting tables, and only then begin feature engineering.
```
