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
| Data Validation      | 🔄 In Progress |
| Exploratory Analysis | 🔄 In Progress |
| Signal Research      | ⏳ Not Started  |
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
