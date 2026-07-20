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
Independent statistical observation frame:   COMPLETE
Independent forward labels:                  COMPLETE
Independent baseline behaviour:              COMPLETE
Independent feature engineering:             COMPLETE
Independent univariate feature evaluation:   COMPLETE
Redundancy and incremental information:      COMPLETE
Multivariate research:                       COMPLETE
Statistical signal construction:             COMPLETE
Independent statistical backtest:            COMPLETE — standalone system REJECTED
POI integration research:                    NEXT PHASE
```

The correct next action is:

```text
Section 12.0 — Integration Research (POI direction candidates filtered and sized by the frozen statistical opportunity models)
```

---

## Section 3 — Eligible Observation Frame Completion

**Status:** COMPLETE — `SECTION 3 STATUS: READY`

Section 3 now defines one eligible observation as a completed GC decision bar `t` mapped to the true next one-minute GC bar `t+1`, whose open is the theoretical entry for later research. Entry eligibility and session assignment use the timezone-aware `entry_timestamp_ny` in `America/New_York`; UTC timestamps remain immutable join keys. The locked half-open entry windows are London `[03:00, 06:00)` and New York `[07:00, 12:00)`. No entry is permitted during 06:00-07:00 or at/after 12:00. Each observation also carries the same-date 15:30 New York forced-exit timestamp and schedule-based minutes remaining to that exit.

The trusted GC source contains **1,759,671** rows. The validated frame contains **586,530** eligible observations: **219,938 London** and **366,592 New York**. Frozen partition counts, assigned from the New York entry date, are **306,230 Development**, **118,076 Validation**, and **162,224 Final test**.

Decision-entry pairs must be strictly consecutive by one minute and remain within the same product, selected symbol/active contract, instrument identifier, continuous segment, and New York trading date. Both bars must pass `tradable_research_flag`; saved roll-window and liquidity controls are also checked explicitly. The exclusion funnel is sequential and fully reconciles to the GC source count.

The output is saved to `data/processed/statistical_research/eligible_observations_gc.parquet` and remains excluded from Git. Save/reload checks confirmed row and column preservation, UTC and New York timezone metadata, unique observation IDs and decision-entry pairs, complete saved fields, and absence of all quarantined forward columns. Synthetic boundary cases for session opens/closes, contract changes, missing minutes, and date crossings passed. No schema fallback was required because `symbol`, `active_symbol`, `instrument_id`, and `continuous_segment_id` were available. Fixed-horizon availability and price outcomes were intentionally outside the Section 3 gate and are now completed in Section 4 below.

**Next section:** Section 6.0 — Feature Engineering Framework.

---

## Section 4 — Forward Outcome Label Completion

**Status:** COMPLETE — `SECTION 4 STATUS: READY`

Section 4 preserves all **586,530** validated GC observations and produces a **586,530 × 169** forward-label table. The locked horizons are **5, 15, 30, 60, 120, and 180 minutes**. Decision information ends with completed bar `t`; entry is the open of bar `t+1`; an `h`-minute path contains exactly `h` consecutive one-minute bars beginning with the entry bar. The final source bar starts at `entry + (h - 1) minutes`, its close supplies the exit price, and the saved economic exit timestamp is `entry + h minutes`. Fixed horizons are never shortened.

All **1,759,671** GC open, high, low, and close values passed the **0.10** GC tick grid at a strict `1e-8`-tick floating tolerance, with zero off-grid values. Tick outcomes are saved as nullable 32-bit integers; one-based first-extreme times are nullable 16-bit integers. ATR-normalized outcomes use only `rolling_atr_20m` from decision bar `t`, saved as `decision_atr_20m`. ATR normalization is available for **586,382** observations (**99.9748%**); the remaining **148** retain valid raw labels when their paths are valid, while ATR outcomes and expansion labels are null.

Realized volatility is the unannualized total path measure `sqrt(sum(r_i²)) × 10,000` basis points. Its first component is `log(entry-bar close / entry open)` and later components are consecutive path-close log returns. No decision-close-to-entry-open return is included.

Expansion uses one frozen 80th-percentile threshold per horizon, fitted only on label-available Development observations with valid decision ATR and applied unchanged to every partition. The thresholds and Development fitting counts are:

| Horizon (minutes) | Threshold (decision ATR) | Development rows |
|---:|---:|---:|
| 5 | 3.111111 | 306,161 |
| 15 | 5.557772 | 306,108 |
| 30 | 8.000000 | 305,979 |
| 60 | 11.797753 | 305,603 |
| 120 | 17.413793 | 304,370 |
| 180 | 22.409639 | 302,774 |

Availability is weakly non-increasing and preserves every observation:

| Horizon (minutes) | Available | Unavailable | Availability rate |
|---:|---:|---:|---:|
| 5 | 586,468 | 62 | 99.9894% |
| 15 | 586,296 | 234 | 99.9601% |
| 30 | 585,958 | 572 | 99.9025% |
| 60 | 585,033 | 1,497 | 99.7448% |
| 120 | 582,471 | 4,059 | 99.3080% |
| 180 | 579,166 | 7,364 | 98.7445% |

The observed production unavailability reason at every horizon is `missing_or_nonconsecutive_minute`; synthetic tests also validate the full deterministic reason priority for insufficient rows, contract/instrument changes, segment changes, New York date changes, tradability/roll/liquidity failures, exact and breached 15:30 boundaries, and invalid prices. The executed notebook passed **20/20 synthetic tests**, **174/174 production checks**, **12/12 fixed-seed London/New York path reconstructions**, and **13/13 independent save/reload checks**. The saved schemas retain Arrow dtypes, dictionary meanings, null semantics, identifier order, and UTC/New York timezone metadata. The trusted `research_bars` dataframe was not mutated, existing legacy forward columns remained quarantined, and no Final-test values influenced threshold fitting.

Saved generated artifacts (excluded from Git):

```text
data/processed/statistical_research/forward_labels_gc.parquet
data/processed/statistical_research/forward_label_expansion_thresholds_gc.parquet
```

Reusable implementation and automated tests:

```text
src/statistical_research/__init__.py
src/statistical_research/labels.py
tests/test_statistical_research_labels.py
```

No POI-derived information, R-multiple labels, stops, targets, continuation/reversal labels, or trade simulations were introduced. Section 4 is the ready input to the completed Section 5 baseline below.

---

## Section 5 — Baseline Behaviour Completion

**Status:** COMPLETE — `SECTION 5 STATUS: READY`

The Section 5 milestone completed the exact sequence `5.0 Establish Baseline Behaviour`, `5.1 Baseline Scope, Metrics, and Comparison Rules`, `5.2 Unconditional Forward Outcome Distributions`, `5.3 Baseline Excursion, Range, and Volatility Behaviour`, `5.4 Forward Outcomes by Session`, `5.5 Forward Outcomes by Time of Day and Day of Week`, `5.6 Directional Symmetry and Tail Asymmetry`, `5.7 Year and Research-Partition Stability`, `5.8 Economic Significance and Transaction-Cost Thresholds`, `5.9 Baseline Validation and Save Outputs`, and `5.10 Baseline Summary and Benchmark Definition`. Section 6 was outside that milestone and is now complete below.

The sole primary input is `data/processed/statistical_research/forward_labels_gc.parquet` (**586,530 × 169**). Population counts remain **219,938 London**, **366,592 New York**, **306,230 Development**, **118,076 Validation**, and **162,224 Final test**. Horizon availability remains **586,468 / 586,296 / 585,958 / 585,033 / 582,471 / 579,166** at 5/15/30/60/120/180 minutes, respectively.

Primary generated outputs (excluded from Git as configured) are:

```text
data/processed/statistical_research/baseline_summary_gc.parquet                179,184 × 19
data/processed/statistical_research/baseline_cost_thresholds_gc.parquet            126 × 23
data/processed/statistical_research/baseline_validation_gc.parquet                 120 × 5
reports/statistical_research/tables/section5/*.csv
reports/statistical_research/figures/section5/*.png
reports/statistical_research/summaries/section5_baseline_summary.md
reports/statistical_research/summaries/section5_benchmark_definition.json
```

The unconditional mean signed returns in ticks at 5/15/30/60/120/180 minutes are **0.014, -0.036, -0.144, -0.536, -0.871, and -0.802**; date-block 95% intervals for daily-weighted mean returns include zero at every horizon. Median returns are **0, 0, 1, 1, 2, and 2 ticks**, while mean absolute returns expand from **14.192** to **82.643 ticks**. Mean future range expands from **28.710** to **175.074 ticks** and mean realized volatility from **6.370** to **41.112 bps**; fitted log-log horizon exponents are **0.505** and **0.520**, respectively, indicating approximately square-root rather than linear clock-time growth over this grid.

New York movement is materially larger than London: mean future range grows from **33.023 to 199.960 ticks** in New York versus **21.520 to 133.565 ticks** in London. Signed-return centers are small relative to dispersion and vary by session and partition; Development-to-Validation signs are not uniformly stable. Overall weekday cells cover **238–250 New York trading dates** and are saved both overall and by year; no weekday filter is approved.

Long and short signed returns are paired algebraic transformations, not independent samples, and long/short excursion identities reconcile exactly. Central p95/|p05| tail ratios remain near one (**1.000 to 0.963**), while p99/|p01| ratios fall from **0.959 to 0.843**, showing modest larger downside extreme-tail magnitude at longer horizons without establishing a tradable directional bias.

The fixed sensitivity grid is **0, 1, 2, 3, 4, 5, and 10 ticks**. At a five-tick hurdle, combined absolute exit-to-exit exceedance rises from **63.49%** at 5 minutes to **93.34%** at 180 minutes. Five-tick long directional exceedance rises from **32.11% to 48.06%** and short directional exceedance from **31.38% to 45.29%**. Exit-to-exit, MFE, MAE, and range hurdle results remain separate movement diagnostics and are not PnL or a cost backtest.

The notebook passed **120/120** Section 5 checks: 8 input/identity, 19 availability, 54 outcome-invariant, 12 path-boundary, 8 aggregation-reconciliation, 2 sampled statistical/visual audit, and 17 save/reload checks. Stable keys, schemas, null behavior, counts, quantile ordering, tick/bps/ATR transformations, classifications, session/time/weekday/year/partition reconciliation, forced-exit boundaries, and deterministic early/late London/New York samples all passed. The notebook was restarted and executed top-to-bottom with `.venv-1`; all **36/36 code cells** executed in order with no error output, ending in `SECTION 5 STATUS: READY`.

Final-test governance: Section 5 used a metric and code contract frozen before its one-time descriptive Final-test exposure. The exposed Final-test information includes return center, direction rates, magnitude, excursions, range, realized volatility, tails, availability, sessions, time-of-day, weekdays, symmetry, and fixed hurdle sensitivity. Later work must not describe the Final test as completely unseen and must not tune feature definitions, bins, horizons, thresholds, or models to these baseline values. Development and Validation remain the primary interpretation samples.

Reusable implementation and tests:

```text
src/statistical_research/baselines.py
tests/test_statistical_research_baselines.py
scripts/update_statistical_section5_notebook.py
```

Git commit information: this milestone is committed directly on `main` with message `Complete statistical baseline behaviour research`; the exact verified hash and `origin/main` push result are recorded in Git history and the final implementation handoff.

**Next section at the Section 5 milestone:** Section 6.0 — Feature Engineering Framework.

---

## Section 6 — Feature Engineering Framework Completion

**Status:** COMPLETE — `SECTION 6 STATUS: READY`

Section 6 now implements the exact reader sequence `6.0 Feature Engineering Framework` through `6.15 Save Outputs, Section Summary, and Completion Gate`. It does not start Section 7 or perform feature ranking, feature/outcome correlations, quantile outcome analysis, predictive modelling, label-based selection, strategy construction, PnL estimation, or backtesting.

The final GC feature matrix is **586,530 × 97**: **12 join/audit metadata columns** followed by **85 registry-authoritative predictors**. The predictor set contains **79 core** and **6 experimental** features; exactly **80 are numeric**, with three nullable booleans and two categorical context fields. Population order and counts remain **219,938 London**, **366,592 New York**, **306,230 Development**, **118,076 Validation**, and **162,224 Final test**. Matrix memory is **213.40 MiB** after optimization versus an estimated **414.21 MiB** unoptimized representation; estimated peak working memory is **634.05 MiB**.

Feature-family counts are:

| Family | Count |
|---|---:|
| Price, return, and momentum | 16 |
| Volatility and range state | 13 |
| Candle geometry | 10 |
| Volume and activity | 10 |
| VWAP and session state | 10 |
| Trend and persistence | 11 |
| Session, clock, and calendar | 9 |
| Experimental hypotheses | 6 |

The experimental features and frozen rationales are:

- `directional_energy_balance_15_exp` — emphasizes high-energy completed returns and measures whether recent movement energy was directionally coherent.
- `wick_pressure_balance_10_exp` — aggregates repeated upper/lower-tail rejection as an OHLCV proxy; it is not order flow or measured liquidity.
- `compression_age_exp` — distinguishes fresh compression from a prolonged stagnant regime using predefined volatility and ATR-ratio thresholds.
- `vwap_elasticity_30_exp` — estimates the completed historical return response to lagged research-day VWAP distance using 30 causal pairs.
- `liquidity_vacuum_score_exp` — tests whether an edge-closing range bar on relatively light activity proxies for low resistance; it is not market depth or actual liquidity.
- `pullback_tension_5_30_exp` — isolates a short completed counter-move inside a broader 30-minute displacement without exhaustive interactions.

Standing principle:

> The statistical research branch combines a disciplined, interpretable feature foundation with a small number of explicitly marked experimental hypotheses. Experimental features are identified with an asterisk in reader-facing documentation, include a written rationale, and are evaluated under exactly the same statistical and out-of-sample standards as conventional features.

Construction uses the full **1,759,671-row** trusted GC history before direct decision-bar mapping. Legacy saved features are not copied blindly. The approved source namespace contains only raw OHLCV, timestamps, product/contract/instrument/segment identifiers, and tradability/roll/liquidity controls. Continuity resets on non-one-minute timestamps, product or selected-contract changes, instrument changes, continuous-segment changes, and invalid tradability/roll/liquidity boundaries. Fixed windows require complete history and retain nulls after warm-up, gaps, resets, zero denominators, or invalid regression variation. Research-day state resets at 01:00 New York, London execution state at 03:00, and New York execution state at 07:00. The first eligible entry at an execution-session open honestly has null execution-session state because decision bar `t` precedes the anchor bar.

The time-of-day volume reference has **32** session/15-minute groups and is fitted only on Development. Development application is leave-one-New-York-trading-date-out using count/sum/squared-sum subtraction; Validation and Final test apply frozen full-Development parameters. Validation or Final-test volume cannot change the saved reference table. Partial 2021 and partial 2026 contribute only observed dates, with no annual reweighting.

Final-test governance remains strict. Frozen Final-test feature values are saved, but inspection is limited to schema, row count, null/finite rates, dtypes, and transformation integrity. No Final-test feature distribution or feature-to-outcome relationship informed a definition, threshold, inclusion, or removal. The feature builder has no label-table argument, no outcome or POI column entered its namespace, and entry-bar OHLCV is excluded by the future-mutation test.

Validation results:

- **14/14** new synthetic feature tests passed.
- **38/38** combined statistical-research feature, label, and baseline tests passed.
- **37/37** critical production, manual, diagnostic, and save/reload gates passed.
- **40/40** fixed-seed manual reconstructions passed across five timing/boundary cases and all eight feature families.
- No infinity, exact duplicate, constant, near-constant, invalid-range, extreme, or excessive-missingness feature was found.
- Maximum all-sample feature missingness is **0.966873%**; maximum Development missingness is **1.396989%**, explained by continuity resets, zero denominators, or complete-window requirements.
- The fresh `.venv-1` notebook execution completed **51/51 code cells** sequentially with zero error outputs and recorded `C:\Users\abond\Desktop\WORK FILES\Systemic\Project 1\.venv-1\Scripts\python.exe` as the active interpreter. Full-notebook runtime was **488.4 seconds**; the Section 6 block took **49.25 seconds**, including a **34.26-second** feature build.

Saved generated artifacts (Parquet files remain excluded from Git):

```text
data/processed/statistical_research/feature_matrix_gc.parquet
data/processed/statistical_research/feature_registry_gc.parquet
data/processed/statistical_research/feature_validation_gc.parquet
data/processed/statistical_research/feature_diagnostics_gc.parquet
data/processed/statistical_research/feature_reference_parameters_gc.parquet

reports/statistical_research/tables/section6/feature_registry_gc.csv
reports/statistical_research/tables/section6/feature_diagnostics_gc.csv
reports/statistical_research/tables/section6/section6_manual_feature_audit.csv
reports/statistical_research/summaries/section6_feature_engineering_summary.md
```

Reusable implementation and tests:

```text
src/statistical_research/feature_registry.py
src/statistical_research/feature_engineering.py
src/statistical_research/feature_validation.py
tests/test_statistical_research_features.py
scripts/update_statistical_section6_notebook.py
```

No feature has yet been shown to possess predictive value. Section 6 establishes only a valid candidate feature matrix.

**Exact next approved section:** Section 7.0 — Univariate Feature Evaluation.

---

## Section 7 — Univariate Feature Evaluation Completion

**Status:** COMPLETE — `SECTION 7 STATUS: READY`

Section 7 implements the reader sequence `7.0 Univariate Feature Evaluation` through `7.10 Section 7 Summary and Completion Gate`. It screens all **83** evaluable registered predictors (80 numeric, 3 boolean) against two outcome families — `direction` (signed `forward_return_{h}_atr`) and `expansion` (unsigned `future_range_{h}_atr`) — across all six frozen horizons, both execution sessions, and the Development/Validation partitions. The two categorical context predictors are excluded with recorded reasons: `entry_session` is a stratification dimension of the screen itself, and Section 5 approved no weekday filter.

Governance is unchanged and enforced mechanically. The evaluation frame contains **419,393** observations (302,774 Development, 116,619 Validation; 157,527 London, 261,866 New York; 884 trading dates); all **162,224 Final-test rows are excluded at frame construction**, and passing a frame containing Final-test rows to the evaluator raises an error. 4,913 Development/Validation rows lacking any horizon label or ATR normalization were dropped so an identical sample supports every horizon. Quintile buckets are fitted per session on Development only and applied unchanged to Validation. The evidence unit is the per-New-York-trading-date cross-sectional Spearman IC with date-block bootstrap intervals (seed 20260720, 2,000 replicates). Benjamini–Hochberg q-values exist only on the Development screen, within outcome-family × session families of 498 related tests.

Shortlist criteria were frozen in `Section7Config` before results were computed: Development q ≤ 0.10; ≥ 400 Development and ≥ 150 Validation trading dates; ≥ 10,000 Development and ≥ 4,000 Validation observations; Validation sign agreement with ≥ 25% IC retention; |Development bucket monotonicity| ≥ 0.8; and, for the direction family only, a ≥ 2-tick top-minus-bottom quintile spread in both partitions.

Headline result:

```text
ADVANCE_DIRECTIONAL:  0 features
ADVANCE_EXPANSION:   55 features
WEAK_UNSTABLE:       28 features
NO_EVIDENCE:          0 features
Shortlist cells:    358 of 1,992 confirmation cells
```

No univariate OHLCV/state feature produced an advancement-grade signed-return relationship — consistent with the Section 5 baseline. Expansion (opportunity/volatility) structure is strong and Validation-confirmed: leading volatility-state and session-clock features reach |daily IC| ≈ 0.61–0.76 with Development→Validation retention near or above 1.0 and monotone quintile structure (e.g. `atr_20` vs 180-minute ATR-relative future range: Dev −0.734 / Val −0.763 in New York). Recorded caveats: the expansion outcome is ATR-normalized, so negative volatility-feature ICs reflect volatility mean-reversion plus the normalization denominator; the session-clock features are mutually redundant by construction; expansion advancement is opportunity forecasting, not a trading edge.

Validation: **12/12** structural checks passed; **18/18** new synthetic engine tests passed (planted-signal detection, BH noise rejection, Development-only bin fitting, bootstrap determinism, hand-checked BH q-values, Final-test rejection, session confinement, economic-gate blocking); the full project suite is **112/112**. The screen runs in ≈ 125 seconds with ≈ 1.4 GB peak working memory.

Reusable implementation and tests:

```text
src/statistical_research/feature_evaluation.py
tests/test_statistical_research_feature_evaluation.py
scripts/update_statistical_section7_notebook.py
```

Saved generated artifacts (excluded from Git):

```text
data/processed/statistical_research/univariate_results_gc.parquet
data/processed/statistical_research/univariate_bucket_summary_gc.parquet
data/processed/statistical_research/feature_stability_gc.parquet
data/processed/statistical_research/candidate_feature_shortlist_gc.parquet
```

Generated shortlist/verdict CSVs under `reports/statistical_research/tables/section7/` also remain excluded from Git per the global CSV rule. The tracked record of the milestone is:

```text
reports/statistical_research/summaries/section7_univariate_evaluation_summary.md
```

The Final test remains locked. **Exact next section:** Section 8.0 — Redundancy and Incremental Information (cluster the 55 expansion advancers, select interpretable representatives, test incremental value, freeze the candidate set).

---

## Section 8 — Redundancy and Incremental Information Completion

**Status:** COMPLETE — `SECTION 8 STATUS: READY`

Section 8 reduces the 55 Section 7 expansion advancers to a frozen candidate set under rules declared in `Section8Config` (seed 20260721) before computation. Development-only Spearman correlations are clustered with average linkage on `1 − |ρ|` cut at |ρ| ≥ 0.7, yielding **30 clusters** (the largest, size 7, is the ATR/realized-volatility ladder). One representative per cluster is selected mechanically: best passing-cell Validation |IC| at the 60-minute primary horizon, ties broken core-over-experimental then by name. Every non-anchor representative must then demonstrate incremental information beyond the anchor — the per-NY-date **partial rank IC** controlling for the anchor — with Development |partial IC| ≥ 0.05, Validation sign agreement, ≥ 25% retention, and 400/150 date floors, at the 60- and 180-minute horizons.

Result: anchor `atr_20` plus **14 confirmed-incremental representatives = 15 frozen expansion features**, spanning volatility state, session clock, trend efficiency/structure, activity, and one surviving experimental hypothesis (`vwap_elasticity_30_exp`). Fifteen representatives were rejected for no confirmed incremental information beyond the anchor — including `log_volume` (raw Validation |IC| ≈ 0.29, fully redundant with volatility state). The frozen directional set is **explicitly empty** because Section 7 produced no directional advancer.

Governance: same Development+Validation frame as Section 7; Final-test rows excluded and rejected with an error; clustering provably fitted on Development only (synthetic test: a duplicate decorrelated in Validation still clusters with its Development twin). Validation: **8/8 structural checks**, **10/10 new synthetic tests**, **122/122 full suite**. Runtime ≈ 10.5 s.

Reusable implementation and tests:

```text
src/statistical_research/feature_redundancy.py
tests/test_statistical_research_redundancy.py
scripts/update_statistical_section8_notebook.py
```

Generated artifacts (excluded from Git): `feature_correlation_gc.parquet`, `feature_clusters_gc.parquet`, `feature_incremental_information_gc.parquet`, `frozen_expansion_feature_set_gc.parquet`, and the `reports/statistical_research/tables/section8/` CSVs. Tracked record: `reports/statistical_research/summaries/section8_redundancy_summary.md`.

The Final test remains locked. **Exact next section:** Section 9.0 — Multivariate Research, beginning with simple linear benchmarks on the frozen 15-feature set against the anchor-only baseline under chronological validation.

---

## Section 9 — Multivariate Research Completion

**Status:** COMPLETE — `SECTION 9 STATUS: READY`

Section 9 benchmarks the frozen 15-feature expansion set against the anchor `atr_20` alone. Inputs and outcomes enter as per-date cross-sectional rank z-scores; models are fitted per session; ridge strength is selected by expanding-window walk-forward inside Development only (year folds with a one-trading-date embargo); Validation is touched once per model. Rules were frozen in `Section9Config` (seed 20260722). Tree models are deferred by contract until linear benchmarks earn them.

Result: the model advances in three of four session/horizon cells (Validation IC improvement over the anchor of +0.045 London 60m, +0.081 New York 60m, +0.088 New York 180m, all with bootstrap intervals above zero); London 180m improves by only +0.008 and is recorded as anchor-sufficient. Logistic AUCs reach 0.80 to 0.93 on Validation and always exceed the anchor. Calibration is rank-correct except New York 60m, which overstates upper probability deciles and requires recalibration before any sizing use. Verification: 9/9 structural checks, 12/12 synthetic tests (including the anchor-only overfitting guard), full suite green. Implementation: `src/statistical_research/multivariate.py`, `tests/test_statistical_research_multivariate.py`, `scripts/update_statistical_section9_notebook.py`. Tracked record: `reports/statistical_research/summaries/section9_multivariate_summary.md`.

---

## Section 10 — Statistical Signal Construction Completion

**Status:** COMPLETE — `SECTION 10 STATUS: READY`

Because Section 7 approved no directional feature, Section 10 formalizes benchmark directions (long, short) with the only evidence-based component available: an opportunity gate from the Section 9 frozen per-session 60-minute ridge model at its Development 80th percentile. Stops are 1.5 x decision ATR clamped to 10-100 ticks, targets 2R, holding capped at 120 minutes, one R per trade; session, noon, and 15:30 rules are inherited. Rules frozen in `Section10Config` (seed 20260723).

Result: 419,063 candidates (85,365 gate-passing); Development gate rates 19.98/19.99 percent against the declared 20 percent; median stop 10.6 ticks with the 46 percent minimum-clamp rate recorded. Verification: 7/7 structural checks, 8/8 synthetic tests (including proof that mutating Validation data leaves the frozen gate threshold unchanged). Implementation: `src/statistical_research/signal_construction.py`, `tests/test_statistical_research_signals.py`, `scripts/update_statistical_section10_notebook.py`. Tracked record: `reports/statistical_research/summaries/section10_signal_construction_summary.md`.

---

## Section 11 — Independent Sequential Backtest Completion

**Status:** COMPLETE — `SECTION 11 STATUS: READY` — **standalone statistical system REJECTED**

A chronological single-position simulator executed 86,353 trades across the four declared variants under frictionless, base (2.6 ticks round trip), and pessimistic (4.6 ticks) cost scenarios. Entry fills are verified against bar opens for every trade; ambiguous bars are scored conservative stop-first (0.19 percent of trades); no trade crosses a date, segment, or the 15:30 boundary. Rules frozen in `Section11Config` (seed 20260724).

Result: every variant is rejected under the declared rule (positive net base-scenario expectancy in both partitions required); mean net R ranges -0.19 to -0.28. Win rates near one third at a 2R target are what a directionless market produces, and the expansion gate concentrates activity but supplies no direction. This is the defensible "none qualify" outcome the PRD treats as valid Phase 2 success: Branch B standalone is closed, and its validated opportunity-forecasting assets transfer to the Section 12 hybrid integration phase. Verification: 8/8 structural checks, 13/13 synthetic tests on hand-constructed bar paths, full suite green. Implementation: `src/statistical_research/sequential_backtest.py`, `tests/test_statistical_research_backtest.py`, `scripts/update_statistical_section11_notebook.py`. Tracked record: `reports/statistical_research/summaries/section11_sequential_backtest_summary.md`.

The Final test remains locked throughout Sections 7-11.
