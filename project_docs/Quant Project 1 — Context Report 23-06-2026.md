# Quant Project 1 — Context Report

> **Purpose:**
> This document serves as the authoritative project context for future AI sessions. Any AI assistant reading this file should assume all completed sections are already done and continue from the current project state unless instructed otherwise.

---

# AI Collaboration & Engineering Guidelines

## Purpose

This document provides context and engineering standards for any AI assistant contributing to this project. Before writing code or making recommendations, assume the role and responsibilities described below.

---

# Your Role

You are acting as a **Senior Quantitative Developer and Research Engineer** working at a professional quantitative trading firm.

Your responsibility is **not simply to produce working code**, but to help build a research codebase that is:

* Production-quality
* Computationally efficient
* Memory efficient
* Readable
* Well documented
* Statistically sound
* Easy to maintain and extend

Treat every notebook and module as though it will eventually become part of a live trading research pipeline.

---

# Project Objective

The objective of this project is to research, develop, and evaluate systematic trading strategies using CME Gold Futures (GC) and Micro Gold Futures (MGC) data obtained from Databento.

This project prioritizes:

* Data integrity
* Robust research methodology
* Correct statistical reasoning
* Efficient data processing
* Reproducible experiments
* Professional software engineering practices

---

# Engineering Philosophy

Always prioritize the following (highest priority first):

1. Correctness
2. Computational efficiency
3. Memory efficiency
4. Scalability
5. Readability
6. Maintainability
7. Simplicity

Working code alone is **not sufficient** if a significantly better implementation exists.

---

# Performance Requirements

Assume datasets will routinely contain **5–20 million rows**, and potentially more in future research.

Avoid implementations that repeatedly scan large datasets or perform unnecessary operations.

Whenever possible:

* Minimize dataframe copies.
* Avoid unnecessary `reset_index()`.
* Avoid unnecessary `.copy()`.
* Avoid repeated dataframe scans.
* Avoid repeated sorting.
* Avoid repeated groupby operations when results can be reused.
* Avoid repeated merges where mapping or indexing is more efficient.
* Minimize memory allocations.
* Prefer vectorized operations.
* Prefer efficient algorithms over convenient ones.
* Consider algorithmic complexity (Big-O) when proposing solutions.

If an operation is expected to be computationally expensive, explicitly explain:

* Why it is expensive.
* Its approximate computational complexity.
* Whether a faster alternative exists.

---

# Code Review Requirement

Before returning any code, perform an internal review.

Ask yourself:

* Is this the most efficient implementation?
* Can this be vectorized further?
* Can unnecessary dataframe copies be removed?
* Can memory usage be reduced?
* Can duplicate computations be eliminated?
* Is there a simpler algorithm?
* Will this scale to tens of millions of rows?
* Would I approve this in a professional code review?

If a substantially better implementation exists, provide that version instead.

---

# Research Standards

Never optimize solely for speed at the expense of correctness.

Every analytical step should be:

* Reproducible
* Explainable
* Statistically defensible

Do not make assumptions without stating them.

If multiple valid approaches exist, explain the trade-offs and recommend the most appropriate one.

---

# Coding Standards

Produce code that is:

* Modular
* Readable
* Well commented where appropriate
* Consistently formatted
* Easy to extend

Avoid unnecessary abstraction, but also avoid large monolithic code blocks.

Prefer reusable functions when logic will be repeated.

---

# Notebook Standards

Notebook cells should have a clear purpose.

Avoid redundant computations.

Where practical:

* Time expensive operations.
* Reuse intermediate results.
* Clearly separate exploration from reusable pipeline code.

---

# Communication Standards

Do not blindly satisfy requests.

If the requested implementation is inefficient, statistically unsound, or poor engineering practice:

1. Explain why.
2. Recommend a better approach.
3. Implement the improved solution unless explicitly instructed otherwise.

Challenge poor design decisions constructively.

The objective is to maximize the long-term quality of the project, not merely complete individual tasks.

---

# Continuous Improvement

When reviewing existing project code:

* Identify inefficiencies.
* Suggest refactoring opportunities.
* Recommend better algorithms where appropriate.
* Point out technical debt before it accumulates.

Treat the project as if you are a long-term contributor responsible for its future quality.

---

## Git & GitHub Workflow Rule

This project uses Git for version control and GitHub for remote backup.

The AI assistant working on this project must follow this workflow:

1. After every major milestone, check the current Git status.
2. Review the modified files before committing.
3. Do not commit raw data files, processed data files, virtual environments, cache files, or large generated outputs unless explicitly instructed.
4. Commit only meaningful project changes such as:

   * notebook updates
   * source code changes
   * documentation updates
   * analysis/report updates
   * configuration changes
5. Use clear commit messages that describe the milestone completed.

Examples:

* `Completed 3.1 Data Quality Assessment`
* `Completed 3.2 Contract Structure Analysis`
* `Added liquidity migration analysis`
* `Implemented front-month detection logic`
* `Updated project context report`

After committing, push the changes to the connected GitHub repository.

The AI should not push broken, untested, or partially completed work unless explicitly told to save a work-in-progress checkpoint.

Before pushing, the AI should confirm that:

* the notebook runs or the relevant section has been tested,
* outputs are reasonable,
* no sensitive information or API keys are included,
* large data files are excluded by `.gitignore`.

For work-in-progress commits, use the prefix:

`WIP:`

Example:

`WIP: liquidity analysis draft`

Major milestones should be committed and pushed immediately after completion.

---


# Final Principle

Always optimize for building a professional quantitative research codebase rather than simply completing the next notebook cell.

Every contribution should improve the overall quality, reliability, scalability, and maintainability of the project.


---

# 1. Project Overview

## Objective

Develop and research a systematic futures trading strategy using CME Gold Futures data.

### Primary Instruments

* Gold Futures (GC)
* Micro Gold Futures (MGC)

### Data Source

* Databento
* GLBX MDP3 Dataset
* 1-Minute OHLCV Bars

---

# 2. Project Directory Structure

```text
Project 1/

├── data/
│   ├── raw/
│   │   └── databento/
│   │       └── glbx-mdp3-20210524-20260524.ohlcv-1m.dbn
│   │
│   └── processed/
│       └── glbx-mdp3-20210524-20260524.ohlcv-1m.parquet
│
├── notebooks/
│   ├── exploration/
│   │   └── exploration.ipynb
│   │
│   └── strategy_research/
│
├── src/
│
├── reports/
│
└── project_docs/
    └── project_context.md
```

---

# 3. Environment

## Virtual Environment

```text
.venv-1
```

## Active Python Interpreter

```text
c:\Users\abond\Desktop\WORK FILES\Systemic\Project 1\.venv-1\Scripts\python.exe
```

### Environment Status

* VS Code connected to correct interpreter.
* Virtual environment verified working.

---

# 4. Libraries Installed

## Core Libraries

```python
numpy
pandas
matplotlib
seaborn
scipy
statsmodels
pyarrow
databento
jupyter
```

### Notes

* Imports already established.
* Do not repeat setup or installation instructions unless specifically requested.

---

# 5. Current Notebook

## Active Notebook

```text
notebooks/exploration/exp1.ipynb
```

---

# 6. Progress Log

---

## 1.0 Project Setup & Imports

### Status

✅ Complete

### Libraries Imported

```python
os
pathlib
sys
datetime
warnings

numpy
pandas
matplotlib
seaborn

databento
```

---

## 2.0 Load & Inspect Raw Databento Data

### Status

✅ Complete

---

### 2.1 Load DBN Store

```python
store = db.DBNStore.from_file(data_path)
```

Status: ✅ Complete

---

### Convert DBN to DataFrame

```python
df = store.to_df()
```

Status: ✅ Complete

---

### Convert DBN to Parquet

```python
df.to_parquet(...)
```

Output:

```text
data/processed/glbx-mdp3-20210524-20260524.ohlcv-1m.parquet
```

Status: ✅ Complete

### Notes

* Parquet is expected to become the primary research dataset.
* DBN remains the immutable raw source archive.

---

## 2.2 Dataset Structure Validation

### Status

✅ Complete

### Checks Performed

#### Dataset Shape

```python
df.shape
```

#### Column Inspection

```python
df.columns
```

#### Data Types

```python
df.dtypes
```

#### Missing Values

```python
df.isna().sum()
```

#### Index Inspection

```python
df.index
```

#### Date Range

```python
df.index.min()
df.index.max()
```

#### Timestamp Ordering

```python
df.index.is_monotonic_increasing
```

#### Duplicate Rows

```python
df.duplicated().sum()
```

---

## Timezone Exploration

### Status

🔄 Started

Attempted:

```python
df.index.tz_convert("America/New_York")
```

Purpose:

* Convert CME timestamps from UTC to New York time.
* Support session-based market analysis.

Not yet formalized into workflow.

---

## Symbol Separation

### Status

✅ Complete

### GC Dataset

```python
gc
```

Contains:

* GC futures contracts
* Spread products excluded

Logic:

```python
df["symbol"].str.startswith("GC")
```

while excluding symbols containing:

```text
-
```

---

### MGC Dataset

```python
mgc
```

Contains:

* Micro Gold futures contracts
* Spread products excluded

---

### Spread Dataset

```python
spreads
```

Contains:

* Inter-contract spread products

Current priority: Low

---

## Contract Investigation

### Status

✅ Initial Exploration Complete

---

### Unique GC Contracts

Created:

```python
gc_symbols
```

Purpose:

* Identify all available GC contracts.

---

### Contract Month Extraction

Created:

```python
contract_month
```

Using:

```python
GC([A-Z])
```

Example:

```text
F = January
G = February
H = March
J = April
K = May
M = June
N = July
Q = August
U = September
V = October
X = November
Z = December
```

---

### Month Groups

Created:

```python
month_groups
```

Structure:

```python
month_code -> dataframe
```

Example:

```python
month_groups["F"]
```

---

# 7. Dataset Understanding

Current understanding of the Databento export:

* Contains multiple years of history.
* Contains multiple GC contracts.
* Contains multiple MGC contracts.
* Continuous futures series is not pre-built.
* Individual contracts must be handled explicitly.
* Contract rollover methodology must be determined during research.

---

# 8. Decisions Made

## Data Source Trust

### Decision

Treat Databento data as authoritative unless clear anomalies are discovered.

### Rationale

Avoid unnecessary time validating vendor data before meaningful research begins.

---

## Preferred Working Format

### Decision

Parquet is the primary analysis format.

### Rationale

* Faster loading
* Smaller storage footprint
* Better notebook workflow

DBN remains the raw archive.

---

## Research Philosophy

Focus on understanding market structure first before developing signals.

---

# 9. Current Workflow

```text
Raw DBN
    ↓
Parquet
    ↓
Data Validation
    ↓
Contract Investigation
    ↓
Signal Research
    ↓
Strategy Research
    ↓
Backtesting
```

---

# 10. Next Planned Work

## 2.8 Reload Parquet Test

### Objectives

Verify:

```python
pd.read_parquet(...)
```

works correctly.

Confirm:

* Row count
* Column count
* Index integrity
* Data consistency versus original load

---

## 2.9 Initial Data Quality Checks

### Investigate

* Trading session gaps
* Unexpected missing bars
* Duplicate timestamps
* Symbol coverage
* Contract rollover behavior
* Daily volume patterns

### Goal

Understand market microstructure before signal development.

---

# 11. Instructions for Future AI Assistants

## Assume Completed

The following sections are complete:

* 1.0 Project Setup & Imports
* 2.0 Load & Inspect Raw Data
* 2.1 DBN Loading
* 2.2 Dataset Validation
* Symbol Separation
* Initial Contract Investigation

---

## Do NOT Repeat

Do not:

* Explain virtual environments again.
* Reinstall libraries.
* Recreate project folders.
* Re-import basic libraries.
* Re-explain DBN fundamentals.
* Re-explain Parquet fundamentals.
* Rebuild notebook structure already completed.

---

## Preferred Behavior

When helping with this project:

1. Read this document first.
2. Continue from the current progress point.
3. Challenge assumptions where appropriate.
4. Focus on quantitative research best practices.
5. Preserve project continuity.

---

# 12. Project Status

| Phase                | Status         |
| -------------------- | -------------- |
| Data Acquisition     | ✅ Complete     |
| Data Validation      | ✅ Complete     |
| Exploratory Analysis | ✅ Complete     |
| Signal Research      | 🔄 Next Phase   |
| Strategy Development | ⏳ Not Started  |
| Backtesting          | ⏳ Not Started  |
| Evaluation           | ⏳ Not Started  |

---

# Update Policy

This document should be updated after every major notebook section, research milestone, or architectural decision.

Its purpose is to allow future conversations and AI assistants to immediately continue from the current project state without rebuilding context from scratch.

---

# WHATS NEXT ? 3.0 EDA 

Here is the structure to use for 3.0 section 

3.0 Exploratory Data Analysis (EDA)

    3.1 Data Quality Assessment

        3.1.1 Reload Parquet Validation
        3.1.2 Dataset Shape & Coverage
        3.1.3 Missing Values
        3.1.4 Duplicate Records
        3.1.5 Timestamp Integrity
        3.1.6 Trading Session Gaps

    3.2 Contract Structure Analysis

        3.2.1 Contract Inventory
        3.2.2 Contract Month Distribution
        3.2.3 Contract Lifespans
        3.2.4 Contract Activity Timeline
        3.2.5 Major vs Minor Delivery Months

    3.3 Liquidity Analysis

        3.3.1 Volume by Contract
        3.3.2 Daily Volume Evolution
        3.3.3 Contract Dominance Over Time
        3.3.4 Liquidity Migration
        3.3.5 Preliminary Front-Month Investigation

    3.4 Session Analysis

        3.4.1 Volume by Hour
        3.4.2 Volatility by Hour
        3.4.3 London Session Behavior
        3.4.4 New York Session Behavior
        3.4.5 Session Comparison

    3.5 Volatility Analysis

        3.5.1 Return Distribution
        3.5.2 Intraday Volatility
        3.5.3 Daily Volatility
        3.5.4 Volatility Regimes
        3.5.5 Volatility Clustering

    3.6 Return Analysis

        3.6.1 Minute Returns
        3.6.2 Daily Returns
        3.6.3 Distribution Characteristics
        3.6.4 Fat Tails
        3.6.5 Trend vs Mean Reversion Tendencies

    3.7 Market Microstructure

        3.7.1 Typical Bar Characteristics
        3.7.2 Volume Spikes
        3.7.3 Large Move Analysis
        3.7.4 Volume–Price Relationships

    3.8 Preliminary Signal Exploration

        3.8.1 Volume-Based Ideas
        3.8.2 Volatility-Based Ideas
        3.8.3 Session-Based Ideas
        3.8.4 Breakout Behavior
        3.8.5 Mean Reversion Behavior

    3.9 EDA Findings & Research Implications

---

# 3.0 Exploratory Data Analysis (EDA) Progress

## 3.1 Data Quality Assessment

### Status

Complete in notebook:

```text
notebooks/exploration/exp1.ipynb
```

### Tracking Summary

* Parquet reload validation completed.
* Dataset shape, coverage, missing values, duplicate records, timestamp integrity, and session gaps checked.
* Processed dataset has 9,527,903 rows and 9 columns.
* No missing values found.
* No true duplicate records found when timestamp is included.
* Timestamps are UTC, monotonic, and minute-aligned.
* Duplicate timestamps are expected because the dataset contains multiple instruments.
* Spreads are present and should remain separated from outright GC/MGC directional research.
* Trading session gaps are expected and need CME session-aware handling.

### Next

Proceed to 3.2 Contract Structure Analysis in the notebook.

---


## 3.2 Contract Structure Analysis

### Status

Added to notebook:

```text
notebooks/exploration/exp1.ipynb
```

### Tracking Summary

* Contract inventory built for GC and MGC outrights.
* Contract month distribution added using standard futures month codes.
* Contract lifespan and active-day calculations added.
* Daily and monthly contract activity timeline tables added.
* Major vs minor delivery months classified by observed volume.

### Key Findings

#### 3.2.1 Contract Inventory

* GC outright universe contains 85 contracts, 3,422,034 bars, and 230,863,662 total volume.
* MGC outright universe contains 41 contracts, 2,934,871 bars, and 186,643,698 total volume.
* Both GC and MGC cover the full processed dataset window from 2021-05-24 to 2026-05-22.
* Not all listed contracts are meaningfully liquid. Some minor contracts have only a handful of bars, while major delivery contracts have hundreds of thousands of bars and millions of contracts traded.

#### 3.2.2 Contract Month Distribution

* GC volume is concentrated in:

```text
Z Dec, M Jun, J Apr, Q Aug, G Feb
```

* MGC volume is concentrated in:

```text
Z Dec, M Jun, J Apr, G Feb, Q Aug
```

* The same five delivery months dominate both products, with slightly different ordering.
* Minor months are present, especially in GC, but contribute very little volume compared with the major cycle.

#### 3.2.3 Contract Lifespans

* GC median contract lifespan is approximately 271 days.
* GC median active trading days is 116.
* MGC median contract lifespan is approximately 544 days.
* MGC median active trading days is 332.
* Top-volume contracts are mostly major delivery months, especially December contracts such as GCZ1, GCZ2, GCZ3, GCZ4, and GCZ5.

#### 3.2.4 Contract Activity Timeline

* GC has a median of 9 active contracts per trading day.
* MGC has a median of 7 active contracts per trading day.
* Many contracts are technically active at the same time, so contract existence alone is not enough for tradable series construction.
* Liquidity ranking and rollover logic will be needed before building a front-month or continuous series.

#### 3.2.5 Major vs Minor Delivery Months

Observed major months:

```text
GC:  Z, M, J, Q, G
MGC: Z, M, J, G, Q
```

Major/month volume concentration:

```text
GC major months:   39 contracts, 2,884,575 bars, 228,439,877 volume
GC minor months:   46 contracts,   537,459 bars,   2,423,785 volume

MGC major months:  34 contracts, 2,628,960 bars, 184,918,692 volume
MGC minor months:   7 contracts,   305,911 bars,   1,725,006 volume
```

### Research Implication

For early strategy research, focus on the major delivery cycle:

```text
G, J, M, Q, Z
```

Minor months should be excluded initially unless a specific spread, calendar, or microstructure question requires them. The next research step should study liquidity migration and develop rules for selecting the active/front-month contract.

### Next

Proceed to 3.3 Liquidity Analysis in the notebook.

---

## 3.4 Session Analysis

### Status

Added to notebook:

```text
notebooks/exploration/exp1.ipynb
```

### Tracking Summary

* Session analysis added using the liquidity-selected active/front contract series from 3.3.
* UTC timestamps converted to America/New_York for intraday session labels.
* Volume by hour added with line chart and heatmap.
* Volatility by hour added with line chart and heatmap.
* London session behavior added.
* New York session behavior added.
* Session comparison added with volume share, realized variance share, and boxplots.

### Session Definitions

```text
London:        03:00-08:19 New York time
New York:      08:20-13:29 New York time
Post-NY:       13:30-16:59 New York time
Overnight/Asia:18:00-02:59 New York time
```

### Preliminary Key Findings

* Active/front-contract session dataset contains 3,493,667 rows.
* New York session carries the largest volume share for both GC and MGC.
* New York session also carries the largest realized variance share for both products.
* London session is meaningful, but secondary to New York in both volume and realized variance.
* Overnight/Asia contributes substantial realized variance despite lower average volume per bar.

Session volume shares:

```text
GC New York:       49.1%
GC London:         20.6%
GC Overnight/Asia: 20.6%
GC Post-NY:         9.7%

MGC New York:       41.4%
MGC Overnight/Asia: 29.9%
MGC London:         18.8%
MGC Post-NY:         9.9%
```

Realized variance shares:

```text
GC New York:       40.3%
GC Overnight/Asia: 31.3%
GC London:         18.6%
GC Post-NY:         9.8%

MGC New York:       40.3%
MGC Overnight/Asia: 30.9%
MGC London:         18.6%
MGC Post-NY:        10.1%
```

### Research Implication

New York should be the first session studied for execution-aware directional signals because it has the strongest combination of volume and volatility. London and Overnight/Asia remain important for regime context, setup formation, and pre-New-York behavior.

### Next

Proceed to 3.5 Volatility Analysis in the notebook.

---

## 3.3 Liquidity Analysis

### Status

Added to notebook:

```text
notebooks/exploration/exp1.ipynb
```

### Tracking Summary

* Volume by contract tables added.
* Daily product-level volume evolution added.
* Daily dominant contract analysis added.
* 5-day rolling liquidity migration logic added.
* Preliminary liquidity-based front-month candidate series added.

### Key Findings

#### 3.3.1 Volume by Contract

Top GC contracts by total volume:

```text
GCZ5, GCZ4, GCZ1, GCZ3, GCZ2
```

Top MGC contracts by total volume:

```text
MGCZ5, MGCJ6, MGCG6, MGCM6, MGCM5
```

Liquidity remains highly concentrated in major delivery months, especially December contracts.

#### 3.3.2 Daily Volume Evolution

Daily volume summary:

```text
GC median daily volume:   156,668
GC mean daily volume:     148,465
GC max daily volume:      537,850

MGC median daily volume:   58,888
MGC mean daily volume:    120,028
MGC max daily volume:   1,536,127
```

GC has higher median daily volume, but MGC has very large volume spikes in parts of the sample.

#### 3.3.3 Contract Dominance Over Time

Daily dominant contract share:

```text
GC median dominant share:  97.0%
GC mean dominant share:    94.8%

MGC median dominant share: 97.5%
MGC mean dominant share:   95.4%
```

The most liquid contract usually captures nearly all daily product volume. This supports using a dominant-contract/front-month approach instead of spreading research across all listed contracts equally.

#### 3.3.4 Liquidity Migration

Using 5-day rolling volume to reduce one-day noise:

```text
GC migration events:  25
MGC migration events: 25
```

Liquidity migration occurs repeatedly through the sample, consistent with futures rollover behavior. A robust rollover rule is required before backtesting.

#### 3.3.5 Preliminary Front-Month Investigation

The preliminary front-month candidate is defined as the contract with the highest rolling 5-day volume for each product/date.

This is not yet a final continuous futures methodology, but it provides a practical starting point for identifying the active contract through time.

### Research Implication

The project should continue toward a liquidity-aware active-contract series. The next major decision is whether to build:

```text
GC-only front-month series
MGC-only front-month series
GC and MGC comparison series
```

Rollover logic should be based on observed liquidity migration rather than fixed calendar assumptions alone.

### Next

Proceed to 3.4 Session Analysis in the notebook.

---

## Engineering Review Pass — 2026-06-25

### Status

Completed targeted engineering review of:

```text
notebooks/exploration/exp1.ipynb
```

### Review Scope

Reviewed notebook code from setup through 3.4 Session Analysis for:

* Repeated dataframe scans
* Unnecessary `.copy()`
* Unnecessary `reset_index()`
* Repeated `groupby`, `merge`, and `sort_values`
* Memory-heavy intermediate dataframes
* Expensive operations that need timing checks

### Changes Made

* Refactored 3.4.1 active/front-contract dataset construction.
* Kept the active-front dataframe narrower for session analysis.
* Replaced an expensive wide reset/merge path with a narrower merge using only required columns.
* Added `time.perf_counter()` timing to the active-front build.
* Refactored 3.4.2 log-return calculation to compute `np.log(close)` once before `groupby.diff()`.
* Added timing around hourly volatility construction.
* Replaced row-wise Python `.apply()` session labeling with vectorized `np.select`.
* Removed unnecessary `.copy()` from London and New York session subsets.

### Verification

Old-vs-new equivalence checks passed:

```text
active_front row count equal:       True
active_front key values equal:      True
session labels equal:               True
session summary shape equal:        True
session summary numeric close:      True
hourly volume equal:                True
hourly volatility numeric close:    True
```

Performance check from targeted verification:

```text
Old active_front build: ~4.85 seconds
New active_front build: ~2.53 seconds
```

### Review Notes

* The main performance risk was 3.4.1, not the hourly aggregation itself.
* Sections 3.1 through 3.3 still contain some repeated setup scans for notebook independence, but those are acceptable for now because they preserve section-level rerunability.
* Future cleanup should introduce a small reusable preprocessing layer only after the EDA workflow stabilizes.
* Plotting and small summary cells were left unchanged because refactoring them would add churn without meaningful speed or memory benefit.

### Next

Continue to 3.5 Volatility Analysis.

---

## 3.5 Volatility Analysis

### Status

Added to notebook:

```text
notebooks/exploration/exp1.ipynb
```

### Tracking Summary

* Return distribution analysis added.
* Intraday volatility analysis added using 30-minute New York-time buckets.
* Daily realized volatility analysis added.
* Product-specific volatility regime classification added.
* Volatility clustering analysis added using autocorrelation of absolute returns, squared returns, and daily realized volatility.
* Expensive calculations include `time.perf_counter()` timing checks.

### Key Findings

#### 3.5.1 Return Distribution

1-minute active/front-contract returns are centered near zero but are strongly fat-tailed.

```text
GC mean return:      0.0038 bps
GC std return:       3.0782 bps
GC excess kurtosis:  120.74
GC mean abs return:  1.8415 bps

MGC mean return:      0.0039 bps
MGC std return:       3.1387 bps
MGC excess kurtosis:  126.31
MGC mean abs return:  1.8960 bps
```

Tail-event frequency:

```text
GC |return| > 10 bps:  1.24%
MGC |return| > 10 bps: 1.28%

GC |return| > 20 bps:  0.19%
MGC |return| > 20 bps: 0.20%
```

#### 3.5.2 Intraday Volatility

Intraday volatility is strongest during the New York morning, consistent with the session analysis from 3.4.

Volatility analysis is now grouped into 30-minute New York-time buckets to avoid overly noisy minute-of-day charts while preserving intraday structure.

#### 3.5.3 Daily Volatility

Daily realized volatility is similar across GC and MGC.

```text
GC median daily realized vol:  80.52 bps
GC mean daily realized vol:    89.69 bps
GC 90th percentile daily vol: 136.75 bps
GC max daily realized vol:    601.68 bps

MGC median daily realized vol:  82.24 bps
MGC mean daily realized vol:    91.00 bps
MGC 90th percentile daily vol: 137.87 bps
MGC max daily realized vol:    601.10 bps
```

#### 3.5.4 Volatility Regimes

Volatility regimes are product-specific and based on daily realized volatility thresholds:

```text
Low:     bottom 25%
Normal:  25%-75%
High:    75%-90%
Extreme: top 10%
```

Extreme regimes show much higher realized volatility and larger single-minute moves.

```text
GC extreme median daily vol:  166.19 bps
MGC extreme median daily vol: 169.35 bps
```

Median daily volume also rises materially in high/extreme volatility regimes, especially for MGC.

#### 3.5.5 Volatility Clustering

Volatility clustering is clearly present.

Daily realized volatility autocorrelation:

```text
GC lag 1:  0.7310
GC lag 5:  0.5012
GC lag 20: 0.2587

MGC lag 1:  0.7173
MGC lag 5:  0.4894
MGC lag 20: 0.2501
```

Minute-level absolute returns also show persistence:

```text
GC minute abs-return lag 1:  0.3727
MGC minute abs-return lag 1: 0.3641
```

### Research Implication

Volatility is not independent through time. It clusters strongly at both minute and daily horizons, so future signal research should use volatility-aware filters, regime labels, or adaptive position sizing.

The return distribution is too fat-tailed for naive normal assumptions. Backtests should model tail behavior explicitly and include realistic risk limits.

### Next

Proceed to 3.6 Return Analysis in the notebook.

---

## Engineering Fix — 3.5.1 Return Distribution

### Date

2026-06-28

### Issue

The first code cell under 3.5.1 Return Distribution appeared to hang in VS Code without producing an error. The underlying return-summary calculation is not inherently large enough to justify an overnight runtime.

### Cause

The 3.5.1 workflow had two practical notebook-performance problems:

* The return-summary cell did not print progress until the full calculation completed.
* The histogram cell passed millions of raw return observations directly into `seaborn.histplot`, which can be very slow in notebook rendering.

### Fix

Updated `notebooks/exploration/exp1.ipynb`:

* Added progress prints and timing checkpoints to the 3.5.1 return-summary cell.
* Replaced groupby lambda/apply summary logic with faster NumPy-based per-product calculations.
* Replaced raw `sns.histplot` on millions of rows with precomputed NumPy histogram bins.
* Vectorized tail-risk threshold calculations.
* Added `minute_ny` directly to the `returns` dataframe so 3.5.2 no longer needs a slow `active_front.loc[...]` lookup.

### Verification

Targeted verification from Parquet reload through 3.5.2 completed successfully.

Observed timings:

```text
Return distribution summary: 0.74 seconds
Precomputed histogram:       0.18 seconds
Tail risk summary:           0.17 seconds
Intraday volatility table:   3.17 seconds
```

### Result

3.5.1 should now produce visible progress quickly and should not appear to hang during the return-distribution analysis.

---

## 3.6 Return Analysis

### Status

Added to notebook:

```text
notebooks/exploration/exp1.ipynb
```

### Tracking Summary

* Minute return analysis added.
* Daily return analysis added using active/front contract bars.
* Distribution characteristics added across minute, daily open-to-close, and same-contract close-to-close returns.
* Fat-tail exceedance analysis added.
* Preliminary trend versus mean-reversion diagnostics added.
* Rollover-aware daily return handling added by separating open-to-close returns from same-contract close-to-close returns.

### Key Findings

#### 3.6.1 Minute Returns

Minute returns are centered near zero and are directionally balanced.

```text
GC mean minute return:   0.0038 bps
MGC mean minute return:  0.0039 bps

GC positive minute rate:  44.2%
GC negative minute rate:  43.9%
GC zero minute rate:      11.8%

MGC positive minute rate: 44.6%
MGC negative minute rate: 44.2%
MGC zero minute rate:     11.2%
```

Raw signed return autocorrelation is small, so there is no obvious simple directional edge from raw one-minute returns alone.

#### 3.6.2 Daily Returns

Daily open-to-close returns show a small positive average in this sample.

```text
GC mean open-to-close daily return:   5.46 bps
GC median open-to-close daily return: 3.38 bps
GC daily return std:                100.24 bps

MGC mean open-to-close daily return:   5.51 bps
MGC median open-to-close daily return: 3.32 bps
MGC daily return std:                100.44 bps
```

Rollover-aware close-to-close returns were also computed by excluding days where the active contract changed.

```text
Rollover days detected:
GC:  26
MGC: 26
```

#### 3.6.3 Distribution Characteristics

Daily returns remain fat-tailed, though much less extreme than one-minute returns.

```text
GC daily open-to-close excess kurtosis:  6.26
MGC daily open-to-close excess kurtosis: 6.12
```

Minute returns have much higher excess kurtosis, confirming that the most extreme non-normality is concentrated at the intraday level.

#### 3.6.4 Fat Tails

Tail exceedance frequencies:

```text
GC |minute return| > 10 bps:  1.24%
MGC |minute return| > 10 bps: 1.28%

GC |daily O-C return| > 100 bps:  21.98%
MGC |daily O-C return| > 100 bps: 21.98%

GC |daily O-C return| > 200 bps:  4.95%
MGC |daily O-C return| > 200 bps: 4.95%
```

Large daily moves are common enough that risk logic must explicitly handle tail behavior.

#### 3.6.5 Trend vs Mean Reversion Tendencies

Preliminary signed-return dependence is weak and mixed.

Minute return current/future correlations:

```text
GC 1-minute horizon:  -0.0145
MGC 1-minute horizon: -0.0215

GC 15-minute horizon:  0.0050
MGC 15-minute horizon: 0.0059
```

Daily return autocorrelation is also small and unstable across lags:

```text
GC lag 1 daily autocorr:   0.0366
GC lag 2 daily autocorr:  -0.0722

MGC lag 1 daily autocorr:  0.0354
MGC lag 2 daily autocorr: -0.0663
```

### Research Implication

Return direction is much less stable than volatility. Volatility clusters strongly, but raw signed returns show weak and inconsistent autocorrelation.

Future signal research should avoid assuming a simple unconditional trend or mean-reversion edge. Directional ideas should be tested conditionally by:

```text
Session
Volatility regime
Volume regime
Breakout context
Prior range structure
Large move / tail-event context
```

### Next

Proceed to 3.7 Market Microstructure in the notebook.

---

## 3.7 Market Microstructure

### Status

Added to notebook:

```text
notebooks/exploration/exp1.ipynb
```

### Tracking Summary

* Typical 1-minute bar characteristics added.
* Volume spike analysis added using product-specific 95th and 99th percentile volume thresholds.
* Large move analysis added using product-specific return quantile thresholds.
* Volume-price relationship analysis added using correlations and volume deciles.
* Analysis uses the active/front contract series, not the full multi-contract universe.

### Key Findings

#### 3.7.1 Typical Bar Characteristics

Typical active/front 1-minute bars are small but non-trivial.

```text
GC median 1-minute range:  2.45 bps
GC mean 1-minute range:    3.33 bps
GC median body:            1.11 bps
GC median volume:          65

MGC median 1-minute range: 2.31 bps
MGC mean 1-minute range:   3.24 bps
MGC median body:           1.10 bps
MGC median volume:         36
```

Zero-return rates:

```text
GC:  11.8%
MGC: 11.2%
```

#### 3.7.2 Volume Spikes

Volume spike thresholds:

```text
GC 95th percentile volume: 414
GC 99th percentile volume: 888

MGC 95th percentile volume: 414
MGC 99th percentile volume: 938
```

Volume spikes are associated with much larger price movement.

```text
GC avg abs return all bars:        1.84 bps
GC avg abs return on 95% spikes:   5.60 bps
GC avg abs return on 99% spikes:   8.91 bps

MGC avg abs return all bars:       1.90 bps
MGC avg abs return on 95% spikes:  6.45 bps
MGC avg abs return on 99% spikes: 10.95 bps
```

#### 3.7.3 Large Move Analysis

Large move thresholds:

```text
GC 99th percentile abs return:  10.84 bps
MGC 99th percentile abs return: 10.96 bps

GC 99.9th percentile abs return:  25.07 bps
MGC 99.9th percentile abs return: 25.35 bps
```

Large moves often overlap with volume spikes, but not always.

```text
GC 99% large-move bars that are also 99% volume spikes:  26.9%
MGC 99% large-move bars that are also 99% volume spikes: 36.3%
```

#### 3.7.4 Volume-Price Relationships

Volume has a strong positive relationship with absolute movement and bar range.

```text
GC corr(volume, abs return):  0.496
GC corr(volume, range):       0.645

MGC corr(volume, abs return): 0.571
MGC corr(volume, range):      0.760
```

Volume has little directional information by itself.

```text
GC corr(volume, signed return):  -0.023
MGC corr(volume, signed return): -0.043
```

The top volume decile has much larger price movement and a much higher large-move rate than lower deciles.

```text
GC top volume decile mean abs return:  4.57 bps
MGC top volume decile mean abs return: 5.02 bps

GC top volume decile large-move rate:  6.97%
MGC top volume decile large-move rate: 8.64%
```

### Research Implication

Volume is a strong activity and volatility signal, but not a standalone directional signal.

Future signal research should use volume as context:

```text
Volume expansion + breakout
Volume spike + continuation/fade test
Large move + volume confirmation
Session-specific volume spike behavior
Volume decile as a volatility/risk filter
```

### Next

Sections 3.8 and 3.9 have now been completed. Proceed next to structured signal research and first strategy prototypes.

---

## 3.8 Preliminary Signal Exploration

### Status

Section 3.8 has been added to:

```text
notebooks/exploration/exp1.ipynb
```

The section was designed as an early EDA signal lab, not a final strategy backtest.

It explores simple forward-return behavior after:

```text
High-volume bars
Volume spikes
Volatility regimes
Session windows
60-minute breakouts
Large one-minute moves
```

Forward returns were measured over:

```text
5 minutes
15 minutes
30 minutes
60 minutes
```

### Engineering Notes

The section reuses existing cleaned objects from prior EDA sections:

```text
bar_features
daily_vol
active_front
```

It creates one reusable signal dataframe:

```text
signal_frame
```

This avoids repeatedly rebuilding forward returns across each subsection.

The section includes timing checks around heavier preparation steps.

### 3.8.1 Volume-Based Ideas

Initial result:

```text
High volume and volume spikes do not create a clean standalone continuation edge.
```

Volume spike bars often show slight short-term fade behavior rather than strong follow-through.

Key output examples:

```text
99th percentile volume spike, 60-minute mean absolute forward move:
GC:  31.8 bps
MGC: 47.1 bps

99th percentile volume spike continuation rate:
GC 5m:  46.9%
GC 60m: 48.6%
MGC 5m:  47.1%
MGC 60m: 48.6%
```

This confirms the earlier 3.7 conclusion:

```text
Volume is useful as a volatility/activity context variable.
Volume is not enough by itself as a directional signal.
```

### 3.8.2 Volatility-Based Ideas

Forward absolute returns rise strongly by volatility regime.

Main finding:

```text
Low regime    = smallest forward movement
Normal regime = moderate movement
High regime   = larger movement
Extreme regime = largest movement
```

Key output examples:

```text
60-minute mean absolute forward move:
Low regime:     GC 8.65 bps,  MGC 8.81 bps
Normal regime:  GC 12.10 bps, MGC 12.25 bps
High regime:    GC 17.15 bps, MGC 17.31 bps
Extreme regime: GC 29.77 bps, MGC 29.95 bps
```

This makes volatility regime useful for:

```text
Position sizing
Stop placement
Target placement
Signal filtering
Risk throttling
```

The volatility regime is more useful as a risk and opportunity filter than as a direct directional signal.

### 3.8.3 Session-Based Ideas

Session remains important.

New York generally produces the largest forward absolute movement, followed by London.

The result supports treating sessions separately in later strategy research.

Useful future split:

```text
New York signals
London signals
Overnight signals
Post-New York signals
```

### 3.8.4 Breakout Behavior

Naive 60-minute breakouts did not show clean continuation.

Initial finding:

```text
Breakout follow-through rates were generally below 50%.
Mean directional follow-through was often negative.
```

Key output examples:

```text
Breakout rate:
GC:  5.24% of bars
MGC: 5.42% of bars

All 60-minute breakouts:
GC 5m mean follow-through:  -0.29 bps, follow-through rate 43.8%
GC 60m mean follow-through: -0.40 bps, follow-through rate 46.6%

MGC 5m mean follow-through:  -0.36 bps, follow-through rate 43.0%
MGC 60m mean follow-through: -0.42 bps, follow-through rate 46.5%
```

Volume-confirmed breakouts did not automatically solve this.

Interpretation:

```text
Simple breakout logic is likely too naive.
Breakouts may need session filters, volatility filters, retest logic, or trend context.
```

### 3.8.5 Mean Reversion Behavior

Large one-minute moves showed mixed behavior.

Short-horizon behavior often leaned slightly toward reversal, especially after large moves.

Key output examples:

```text
All 99th percentile large moves:
GC 5m continuation rate:  46.7%
GC 60m continuation rate: 48.6%

MGC 5m continuation rate:  46.9%
MGC 60m continuation rate: 48.7%
```

However, overnight large moves showed more evidence of continuation over longer horizons.

Interpretation:

```text
Large-move behavior is session-dependent.
New York large moves may be more two-way/noisy.
Overnight shocks may carry more continuation information.
```

### Research Implication

The strongest early signal research direction is not a single raw trigger.

The better path is conditional signal design:

```text
Volume spike + session
Breakout + session + volatility regime
Large move + volume confirmation
Large move + session-specific fade/continuation test
Volatility regime as position/risk filter
```

### EDA Status

Section 3.8 is complete in the notebook and has been followed by Section 3.9, the final EDA synthesis.

Promising first strategy-research candidates:

```text
Session-filtered breakout/fade tests
Overnight large-move continuation tests
New York large-move fade tests
Volatility-regime-aware position sizing
```

---

## 3.9 EDA Findings & Research Implications

### Status

Section 3.9 has been added to:

```text
notebooks/exploration/exp1.ipynb
```

This completes the EDA phase.

### Final EDA Thesis

The data does not point to a simple one-variable edge.

It points to conditional research:

```text
Session matters.
Volatility regime matters.
Volume predicts activity better than direction.
Naive breakout continuation is weak.
Large-move behavior depends on session and volatility context.
```

### 3.9.1 Data Quality Conclusion

The cleaned OHLCV data is suitable for research.

Key points:

```text
No missing values in core OHLCV columns.
No true duplicate timestamp-symbol rows.
Timestamps are minute-aligned and timezone-aware.
Observed gaps are mostly expected futures-session gaps.
```

### 3.9.2 Contract and Liquidity Conclusion

The active/front contract should be selected by liquidity.

GC and MGC both concentrate activity in a dominant contract most of the time, with migration around roll windows.

Research implication:

```text
Use active_front as the base tradable universe.
Avoid thin/inactive contracts.
Flag rollover periods before strategy testing.
```

### 3.9.3 Session Conclusion

New York is the main liquidity and volatility window.

London is active and useful, but its behavior is not identical to New York.

Overnight and Post-NY should be treated as separate market states.

Research implication:

```text
All future strategy results should be reported by session.
Signals should not be judged only on all-session averages.
```

### 3.9.4 Volatility Conclusion

Gold futures show fat tails and strong volatility clustering.

Volatility regimes clearly separate expected forward movement magnitude.

Research implication:

```text
Volatility regime should influence position size, stop distance, target distance, and signal permission.
```

### 3.9.5 Return and Microstructure Conclusion

Unconditional return autocorrelation is weak.

Volume is strongly related to range and absolute return, but weakly related to signed direction.

Research implication:

```text
Avoid unconditional trend or mean-reversion assumptions.
Use volume as a context variable, not a standalone buy/sell trigger.
```

### 3.9.6 Preliminary Signal Conclusion

Section 3.8 produced useful positive and negative evidence.

Weak direction:

```text
Naive 60-minute breakout continuation.
Raw volume-spike continuation.
Always-on trend or mean reversion.
```

Promising direction:

```text
New York large-move fade tests.
Overnight large-move continuation tests.
Breakout failure/retest tests.
Volume-spike filters for event intensity.
Volatility-regime-aware risk sizing.
```

### Recommended Next Phase

Move from EDA into structured signal research and first strategy prototypes.

Recommended base research objects:

```text
active_front
bar_features
signal_frame
```

Every prototype should include:

```text
Explicit entry rules
Explicit exit rules
Costs and slippage
Stop and target logic
Session filter
Rollover filter
Volatility-regime filter
Train/test or walk-forward validation
```

### EDA Completion Note

EDA is complete enough to proceed.

The next work should not add more broad descriptive EDA unless a specific strategy question requires it.

Next phase:

```text
Structured signal research and strategy prototype development.
```

---

# Forward Research Roadmap: Sections 4.0 to 10.0

This roadmap defines the intended structure for the remainder of the exploration notebook after EDA.

Future AI sessions should follow this sequence unless the user explicitly changes direction.

The core principle is:

```text
Do not jump straight from EDA into a full backtest.
First build the final research dataset.
Then formalize the discretionary system.
Then engineer the measurable features.
Then run event studies.
Then run the first simple research backtest.
Then validate robustness.
Then hand off strong candidates to strategy_research notebooks.
```

---

## 4.0 Research Dataset Construction & Continuous Active Contract Series

### Purpose

Build the final clean minute-level research table that every strategy will use going forward.

This section is foundational. If the active contract construction is weak, every later signal test and backtest becomes questionable.

### Required Subsections

```text
4.1 Final Active Contract Selection Rules
4.2 Rollover Safeguards
4.3 Continuous Front-Month Series Construction
4.4 Rollover-Day Exclusion / Warning Flags
4.5 Session-Aware Calendar Handling
4.6 Return, Range, Volume, and Volatility Feature Rebuild
4.7 Save Final Research Tables
4.8 Section 4 Summary
```

### Target Output

Create one final strategy-ready table named one of:

```text
active_front_final
research_bars
```

Preferred name:

```text
research_bars
```

### Required Properties

```text
One row = one minute
Product = GC or MGC
One selected active contract per product/date
No spread instruments
No multi-contract distortion
Session labels included
Rollover flags included
Return features included
Range features included
Volume features included
Volatility features included
```

### Engineering Instruction

Start from the EDA active-front logic, but add rollover safeguards before treating the table as final.

Do not rely only on rolling volume if contract migration is ambiguous.

---

## 5.0 Discretionary Strategy Formalization

### Purpose

Translate the user's discretionary trading logic into clear rules a computer can test.

This is not backtesting yet.

The goal is to convert subjective trading language into measurable definitions.

### Required Subsections

```text
5.1 Strategy Thesis
5.2 Market Bias / Trend Definition
5.3 POI Definition
5.4 Entry Trigger Definition
5.5 Stop-Loss Logic
5.6 Target / Exit Logic
5.7 Invalidation Rules
5.8 Session Filters
5.9 Volatility and Volume Filters
5.10 Strategy Rules in Plain English
5.11 Strategy Rules in Pseudocode
5.12 Section 5 Summary
```

### Translation Rule

Every discretionary concept must become measurable.

Examples:

```text
"Trend is bullish"
```

must become something like:

```text
close > rolling VWAP
20 EMA > 50 EMA
price above previous session midpoint
higher swing highs / higher swing lows
```

And:

```text
"Price reaches my POI"
```

must become something like:

```text
price enters previous session high/low zone
price retests broken range
price returns to VWAP band
price enters volatility-adjusted pullback zone
price trades near a swing high/low level
```

### Output

End Section 5 with:

```text
Plain-English strategy rules
Pseudocode strategy rules
Open assumptions that still need user confirmation
```

---

## 6.0 POI & Signal Feature Engineering

### Purpose

Build the actual columns needed to test the strategy defined in Section 5.

Section 5 defines the logic.

Section 6 makes the logic machine-readable.

### Required Subsections

```text
6.1 Trend / Bias Features
6.2 Session Range Features
6.3 Previous Day / Previous Session Levels
6.4 Swing High / Swing Low Detection
6.5 Pullback / Retest Features
6.6 Mean-Reversion Features Around POIs
6.7 Breakout Failure Features
6.8 Volume Spike / Large Move Features
6.9 Volatility Regime Features
6.10 Forward Return Labels
6.11 Build signal_frame
6.12 Section 6 Summary
```

### Research Framing

The goal is conditional signal design.

Bad framing:

```text
Fade every large move.
```

Better framing:

```text
Fade a large New York session move only if:
price is at a predefined POI,
volatility is elevated but not extreme,
the move is stretched relative to recent range,
volume suggests exhaustion,
the trade is not on a rollover day.
```

### Output

Build a strategy-level `signal_frame` from `research_bars`.

It should include:

```text
POI features
Trend/bias features
Session features
Volatility regime features
Volume spike features
Large move features
Breakout/retest/failure features
Forward return labels
```

---

## 7.0 Prototype Signal Research

### Purpose

Run controlled event studies on the strategy conditions.

This is not a full trading backtest.

The question is:

```text
When the strategy condition appears, what tends to happen after 5, 15, 30, and 60 minutes?
```

### Required Subsections

```text
7.1 Baseline Forward Return Study
7.2 POI Touch Event Study
7.3 POI Rejection Event Study
7.4 POI + Mean Reversion Event Study
7.5 POI + Breakout Failure Event Study
7.6 Session-Specific Results
7.7 Volatility-Regime-Specific Results
7.8 GC vs MGC Comparison
7.9 Candidate Signal Ranking
7.10 Section 7 Summary
```

### Goal

By the end of Section 7, determine:

```text
Which setups deserve a backtest
Which setups should be rejected
Which filters improve behavior
Which sessions matter
Whether each idea behaves more like fade, continuation, or failure/retest
```

---

## 8.0 First Research Backtest

### Purpose

Convert the best Section 7 candidate signals into trades.

Keep the first backtest simple, readable, and research-oriented.

Do not build a large professional backtesting engine yet.

### Required Subsections

```text
8.1 Backtest Design
8.2 Entry Rules
8.3 Exit Rules
8.4 Stop-Loss Rules
8.5 Target Rules
8.6 Time-Based Exit
8.7 Transaction Costs and Slippage
8.8 Position Sizing
8.9 Trade Log Construction
8.10 Performance Metrics
8.11 Equity Curve
8.12 Drawdown Analysis
8.13 Section 8 Summary
```

### Required Outputs

Measure performance in R-multiples first, dollars later.

Backtest output should include:

```text
Number of trades
Win rate
Average win
Average loss
Expectancy per trade
Profit factor
Max drawdown
Average R
Median R
Best trade
Worst trade
Performance by year
Performance by session
Performance by volatility regime
```

---

## 9.0 Robustness & Validation

### Purpose

Test whether the backtest result is stable or curve-fit.

The core question is:

```text
Did we find a real behavior, or did we overfit 2021-2026 gold data?
```

### Required Subsections

```text
9.1 Train/Test Split
9.2 Walk-Forward Testing
9.3 Year-by-Year Performance
9.4 Month-by-Month Performance
9.5 Session-by-Session Performance
9.6 Volatility-Regime Performance
9.7 Parameter Sensitivity
9.8 Transaction Cost Sensitivity
9.9 Trade Randomization / Monte Carlo
9.10 Failure Mode Analysis
9.11 Section 9 Summary
```

### Validation Rule

Treat a strategy with suspicion if it only works:

```text
in one year,
in one session,
in one volatility regime,
with one fragile parameter setting,
or on a tiny sample.
```

---

## 10.0 Final Research Conclusion & Handoff

### Purpose

Close the exploration notebook and decide what moves into dedicated strategy research.

### Required Subsections

```text
10.1 What We Learned
10.2 Best Candidate Strategy
10.3 Rejected Strategy Ideas
10.4 Dataset Objects to Preserve
10.5 Strategy Rules to Move Forward
10.6 Known Limitations
10.7 Next Notebook Plan
10.8 Final Go / No-Go Decision
```

### Handoff Rule

After Section 10, stop expanding the exploration notebook.

Move strong candidates into:

```text
notebooks/strategy_research/
```

Possible next notebooks:

```text
notebooks/strategy_research/01_poi_mean_reversion_strategy.ipynb
notebooks/strategy_research/02_breakout_failure_strategy.ipynb
notebooks/strategy_research/03_ny_large_move_fade.ipynb
```

---

## Correct Project Flow From Here

```text
3.0 EDA
  -> complete

4.0 Final research dataset / active-front construction
  -> create research_bars

5.0 Formalize discretionary POI system
  -> convert trader logic into measurable rules

6.0 Build POI, trend, mean-reversion, session, volatility features
  -> create strategy-level signal_frame

7.0 Event studies / signal diagnostics
  -> rank candidate signals

8.0 First research backtest
  -> convert best events into trades

9.0 Robustness testing
  -> test stability and overfitting risk

10.0 Final handoff
  -> decide what moves into strategy_research

strategy_research notebook
  -> refine the strongest candidate in a cleaner notebook

proper backtester / production-quality research code
  -> later phase, after the idea earns it
```

### Standing Instruction For Future AI Sessions

When continuing this project, do not skip ahead to backtesting unless Sections 4-7 have produced a clean dataset, formal strategy definitions, engineered signal features, and event-study evidence.

The exploration notebook should prove that a strategy idea deserves deeper work.

The strategy research notebook should refine, clean, and prepare that idea for a more serious backtesting framework.

---

# 4.0 Research Dataset Construction & Continuous Active Contract Series Progress

## 4.1 Final Active Contract Selection Rules

### Status

Added to:

```text
notebooks/exploration/exp1.ipynb
```

### Purpose

Section 4.1 formalizes the active-contract selection schedule that will be used to build the continuous research dataset.

This section does not yet create the final minute-level `research_bars` table. It creates the daily active-contract schedule that later Section 4 cells will use.

### Final Selection Rule

The active contract is selected by observed liquidity, not by hard-coded calendar roll dates.

Rules:

```text
Universe: GC and MGC outrights only; spread symbols excluded.
Selection frequency: one active contract per product per UTC trade date.
Primary liquidity measure: 5-trading-day rolling sum of daily contract volume.
Raw active candidate: contract with the highest rolling 5-day volume for each product/date.
Confirmed normal switch: accept a new raw candidate after it remains the raw winner for at least 2 consecutive product-days, has at least 50% of current-day product volume, and has at least a 1.10x rolling-volume lead over the previously selected contract.
Confirmed decisive switch: accept a new raw candidate immediately when it has at least 80% of current-day product volume, the previously selected contract has no more than 20% of current-day product volume, and the new candidate has at least a 1.10x rolling-volume lead.
Ambiguous transition: if the raw winner changes but neither confirmation path passes, retain the prior selected contract and flag the date.
Rollover handling: confirmed switch days and ambiguous transition days become diagnostics for Section 4.2 rollover safeguards.
```

### Objects Created

```text
active_contract_selection_rules
active_contract_rankings
raw_active_candidates
final_active_contract_schedule
active_contract_selection_summary
active_contract_roll_audit
```

### Validation Summary

Validated against the full processed parquet dataset:

```text
GC product-days:  1,555
MGC product-days: 1,555
GC confirmed switches:  25
MGC confirmed switches: 25
GC ambiguous transition days:  6
MGC ambiguous transition days: 4
```

The remaining ambiguous transition days are expected and useful diagnostics. They occur when the raw rolling-volume winner changes, but the new contract has not yet cleared either the persistence rule or the decisive same-day liquidity migration rule.

### Research Implication

The project now has a defensible daily active-contract selection schedule.

Next work should proceed to:

```text
4.2 Rollover Safeguards
```

Section 4.2 should convert confirmed switch days and ambiguous transition days into explicit rollover warning/exclusion windows before `research_bars` is constructed.

---

## 4.0 Section Completion Update

### Status

Section 4.0 is complete in:

```text
notebooks/exploration/exp1.ipynb
```

The notebook now contains the full required subsection sequence:

```text
4.1 Final Active Contract Selection Rules
4.2 Rollover Safeguards
4.3 Continuous Front-Month Series Construction
4.4 Rollover-Day Exclusion / Warning Flags
4.5 Session-Aware Calendar Handling
4.6 Return, Range, Volume, and Volatility Feature Rebuild
4.7 Save Final Research Tables
4.8 Section 4 Summary
```

### Final Objects

Section 4 creates the following core objects:

```text
active_contract_daily
rollover_summary
active_front_final
research_bars
```

`research_bars` is now the main dataset for Section 5 onward.

### Saved Tables

The final Section 4 tables were saved to:

```text
data/processed/active_contract_daily.parquet
data/processed/rollover_summary.parquet
data/processed/active_front_final.parquet
data/processed/research_bars_gc_mgc_1m.parquet
```

These are generated research artifacts and are excluded from Git by `.gitignore`.

### Validation Summary

Final saved table sizes:

```text
active_contract_daily:       3,110 rows
rollover_summary:               50 rows
active_front_final:      3,487,656 rows
research_bars:           3,487,656 rows
```

Product row counts in `research_bars`:

```text
GC:  1,759,671 rows
MGC: 1,727,985 rows
```

Final validation checks passed:

```text
Expected products only: GC and MGC
No spread symbols
No missing OHLCV values
No duplicate timestamp-product rows
Sorted by product/timestamp
Rollover flags present
Session labels present
Return/range/volume/volatility feature columns present
```

### Rollover Handling

Detected active-contract switches:

```text
GC rolls:  25
MGC rolls: 25
Total rolls: 50
```

Section 4 does not delete rollover periods. It flags them with:

```text
is_roll_day
is_pre_roll_day
is_post_roll_day
roll_window_flag
roll_window_type
days_since_roll
days_to_next_roll
tradable_research_flag
```

Default `tradable_research_flag` is conservative and excludes roll-window days, ambiguous rollover candidates, and low selected-volume-share days.

### Continuous Series Decision

The active-front series is a raw stitched active-contract series.

Prices were not back-adjusted. This is intentional at this stage because raw tradable prices plus explicit rollover flags are more transparent for signal research and backtesting diagnostics.

### Session Handling

UTC timestamps remain the canonical key. New York session fields were added:

```text
ts_event_utc
ts_event_ny
trade_date_utc
trade_date_ny
day_of_week
hour_ny
minute_ny
session_label
```

Session labels:

```text
Overnight/Asia
London
NY Morning
NY RTH
Late Session
CME Maintenance Break / Closed
```

### Feature Rebuild

`research_bars` includes:

```text
1-minute returns
5/15/30/60-minute forward returns
bar range
candle body
upper/lower wick
true range
rolling ATR-style range
rolling realized volatility
rolling high-low range
rolling volume
relative volume
volume z-score
selected contract volume share
rollover flags
session labels
tradable research flag
```

Return and rolling-feature calculations are separated by product and are prevented from crossing contract changes or major timestamp gaps.

### Next

Proceed to:

```text
5.0 Discretionary Strategy Formalization
```

Do not begin backtesting yet. Section 5 should translate the discretionary POI-based system into objective, measurable research rules using `research_bars` as the base dataset.

---

---

# 5.0 Discretionary Strategy Formalization Progress

## Status

Section 5.0 is complete in:

```text
notebooks/exploration/exp1.ipynb
```

This section formalizes the user's discretionary GC/MGC POI strategy into objective research rules. It intentionally does not implement signal generation, candidate-trade construction, or backtesting. Those belong in Section 6 onward.

## Completed Subsections

```text
5.1 Strategy Thesis
5.2 Instrument, Session, and Research Scope
5.3 Market Structure and Swing-Break Logic
5.4 POI Definition
5.5 POI Validity and Retest Logic
5.6 Entry Trigger Definition
5.7 Stop-Loss Logic
5.8 Target / Exit Logic
5.9 Invalidation Rules
5.10 Confluence and Feature Engineering Framework
5.11 Strategy Rules in Plain English
5.12 Strategy Rules in Pseudocode
5.13 Section 5 Summary
```

## Core Strategy Formalization

The strategy is now defined as an intraday GC/MGC POI-retest research model:

```text
GC = analysis / signal instrument
MGC = intended execution instrument
Timeframe = 1-minute OHLCV
Base dataset = research_bars from Section 4
Timezone for strategy logic = America/New_York
```

The core principle is:

```text
A valid POI creates a decision zone.
A retest creates a candidate event.
It is not an automatic trade.
```

## Locked Session Rules

POI search window:

```text
1:00am-12:00pm New York time
```

Execution windows:

```text
London:   3:00am-6:00am New York time
New York: 7:00am-12:00pm New York time
```

Expiry:

```text
All POIs expire at 12:00pm New York time.
No POI carries past 12:00pm.
No POI carries into the next New York calendar day.
```

## Market Structure Rules

Swing definition is treated as a research parameter.

Required swing variants:

```text
3-bar swings
5-bar swings
7-bar swings
```

Required swing-break variants:

```text
Wick break
Close break
```

Initial displacement definition:

```text
Bullish displacement = most recent confirmed swing low before the break through the candle that breaks the swing high.
Bearish displacement = most recent confirmed swing high before the break through the candle that breaks the swing low.
```

The POI direction must match the swing-breaking displacement direction.

## Final POI Definition

A valid POI requires both:

```text
1. Classic ICT wick-to-wick FVG.
2. Close-to-next-open gap in the same direction.
```

The POI zone is the full high-to-low range of the middle candle:

```text
POI_high = H[i]
POI_low  = L[i]
POI_mid  = (H[i] + L[i]) / 2
```

Bullish POI:

```text
L[i+1] > H[i-1]
AND
O[i+1] > C[i]
```

Bearish POI:

```text
H[i+1] < L[i-1]
AND
O[i+1] < C[i]
```

Gold futures tick size:

```text
tick_size = 0.10
```

Record:

```text
fvg_size_ticks
close_open_gap_ticks
poi_size_ticks
```

A one-tick minimum FVG and one-tick close-open gap is recommended as the first test, but remains a research parameter.

## POI Retest Rules

Retest/touch definition for a later candle `j`:

```text
high[j] >= POI_low
AND
low[j] <= POI_high
```

Multiple touches are allowed and should be recorded separately before expiry.

Required retest fields include:

```text
retest_number
first_touch_flag
time_since_poi_creation
candles_since_poi_creation
time_since_previous_touch
full_poi_cross_flag
```

For later trade simulation, overlapping trades from the same POI must be controlled unless a specific re-entry or pyramiding rule is explicitly being tested.

## Entry, Stop, and Exit Research Variables

Entry trigger variants:

```text
Boundary touch
50% POI touch
Distal edge touch
```

Stop variants:

```text
POI distal edge stop
Previous-candle adjacent stop
Next-candle adjacent stop
Conservative adjacent-candle stop
```

Stop constraints:

```text
Minimum stop = 25 ticks
Maximum stop = 100 ticks
```

Target variants:

```text
1R
2R
3R
4R
5R
```

Forced exit:

```text
All trades flat by 12:00pm New York time.
No overnight positions.
```

Bars where both stop and target are touched are path-ambiguous on 1-minute OHLCV data. The conservative default should mark those cases as ambiguous or assume stop-first, then test sensitivity later.

## Invalidation Rules

Invalid candidates include:

```text
Expired POI
Invalid stop size
Entry outside execution windows
Rollover danger period
Malformed or missing execution mapping data
```

Historical red-folder news filtering is deferred. Placeholder fields should be added in Section 6:

```text
is_red_folder_news_window
news_event_name
minutes_to_news
minutes_since_news
```

Full POI violation rules remain research variants to test:

```text
No full violation rule
Wick-through invalidation
Close-through invalidation
Full-zone traversal flag
```

## Confluence Framework

Section 5 defines the feature categories that Section 6 should engineer:

```text
VWAP
Volume
Volatility
Session context
Market structure
Chop / consolidation
Extension / exhaustion
POI quality metrics
```

These should initially be recorded as explanatory variables, not hard filters. The research task is to determine which combinations improve POI retest expectancy.

## Required Section 6 Outputs

The next section should implement the formalized rules into:

```text
poi_table
retest_table
candidate_trade_table
```

Section 6 should not jump directly to a full backtest. It should build machine-readable POI, retest, entry, stop, target, and confluence features first.

## Current Project Status

| Phase                         | Status      |
| ----------------------------- | ----------- |
| Data Acquisition              | Complete    |
| Data Validation               | Complete    |
| Exploratory Analysis          | Complete    |
| Research Dataset Construction | Complete    |
| Strategy Formalization        | Complete    |
| POI Feature Engineering       | Next Phase  |
| Event Studies                 | Not Started |
| Backtesting                   | Not Started |
| Robustness Testing            | Not Started |

## Next

Proceed to:

```text
6.0 POI & Signal Feature Engineering
```

The first implementation goal is to build `poi_table`, `retest_table`, and `candidate_trade_table` from `research_bars` using the Section 5 definitions.

---
