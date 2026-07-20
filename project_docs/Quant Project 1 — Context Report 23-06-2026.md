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
import time
from pathlib import Path
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

## Authoritative Reader Map

```text
Sections 1-5: Data acquisition, EDA, research dataset construction, and strategy formalization
Section 6: Final refined POI definition and machine-readable signal tables — complete
Section 7: True POI context and conditional-signal research — complete; no Section 8 candidate approved
Section 8: Future sequential backtesting phase — not started
Appendix A: Superseded prototype event study retained for research history
```

The main notebook uses normal sequential reader-facing numbering. Internal reproducibility names such as `section6c_*`, `section7r_*`, `poi_selection_refinement.py`, and `poi_event_study_refined.py` remain unchanged.

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

# 6.0 POI & Signal Feature Engineering Progress

## Status

Section 6.0 is complete in:

```text
notebooks/exploration/exp1.ipynb
```

The reusable implementation was added to:

```text
src/features/poi_signal_features.py
```

Section 6 converts the Section 5 discretionary POI rules into machine-readable research tables. It remains a feature-engineering and event-labeling section, not a backtest.

## Completed Subsections

```text
6.1 Implementation Contract and Parameters
6.2 Build Section 6 Research Tables
6.3 Output Validation
6.4 POI, Retest, and Candidate Diagnostics
6.5 Save Section 6 Research Tables
6.6 Section 6 Summary
```

## Implementation Scope

The first Section 6 implementation validates GC-only signal logic, matching the Section 5 research plan:

```text
GC = analysis / signal instrument
MGC = intended execution instrument for a later phase
```

MGC execution mapping remains deferred until Section 7 event studies identify which POI retest variants deserve trade simulation.

## Core Objects Created

```text
section6_feature_frame
poi_table
retest_table
candidate_trade_table
signal_frame
section6_summary
section6_validation
```

`candidate_trade_table` contains valid continuation entry/stop variants after Section 5 stop-distance filtering. Invalid stop variants are rejected rather than persisted as candidate trades.

## Saved Tables

The generated Section 6 research tables were saved to:

```text
data/processed/section6_poi_table_gc.parquet
data/processed/section6_retest_table_gc.parquet
data/processed/section6_candidate_trade_table_gc.parquet
data/processed/section6_signal_frame_gc.parquet
```

These are generated research artifacts and remain excluded from Git by `.gitignore`.

## Full-Run Counts

Validated against the saved Section 4 `research_bars` dataset:

```text
GC feature rows:            1,759,671
GC tradable rows:           1,676,591
POIs:                         202,466
POIs with retests:            181,630
Retests:                    6,684,664
First-touch retests:          181,630
Valid candidate variants:   1,597,373
Signal frame rows:          1,597,373
```

## POI Variant Counts

```text
swing_n  break_mode  direction
3        close       bearish      14,527
                     bullish      14,349
         wick        bearish      17,187
                     bullish      16,873
5        close       bearish      16,008
                     bullish      15,858
         wick        bearish      18,682
                     bullish      18,393
7        close       bearish      16,491
                     bullish      16,290
         wick        bearish      19,080
                     bullish      18,728
```

## Valid Candidate Variant Counts

```text
entry_variant  stop_model             valid_candidate_count
boundary       conservative_adjacent  700,230
boundary       poi_distal_edge        404,942
distal         conservative_adjacent   86,784
midpoint       conservative_adjacent  316,298
midpoint       poi_distal_edge         89,119
```

No valid `distal + poi_distal_edge` continuation variants survived the 25-100 tick stop-distance filter in the full run.

## Validation Checks Passed

```text
poi_ids_unique:                     True
retest_ids_unique:                  True
candidate_trade_ids_unique:         True
poi_activation_in_search_window:    True
retests_after_poi_activation:       True
retests_in_execution_window:        True
valid_signals_have_valid_stops:     True
```

## Key Engineering Decisions

POI activation is defined as:

```text
max(POI confirmation candle, swing-break candle)
```

Retests must occur after activation, on the same New York date, inside the London or New York execution windows, and before or at 12:00pm New York time.

The retest engine uses day-level vectorized interval checks rather than one scan per POI. This is necessary because the full run contains more than 200,000 variant-level POIs and more than 6.6 million retest rows.

`signal_frame` is the main Section 7 input. It contains:

```text
candidate parameters
entry, stop, and target prices
forward 5/15/30/60-minute R labels
POI quality features
swing-break metadata
retest context
trend, VWAP, volume, volatility, session, and extension features
news-filter placeholder fields
```

## Current Project Status

| Phase                         | Status      |
| ----------------------------- | ----------- |
| Data Acquisition              | Complete    |
| Data Validation               | Complete    |
| Exploratory Analysis          | Complete    |
| Research Dataset Construction | Complete    |
| Strategy Formalization        | Complete    |
| POI Feature Engineering       | Complete    |
| Event Studies                 | Next Phase  |
| Backtesting                   | Not Started |
| Robustness Testing            | Not Started |

## Next

Proceed to:

```text
7.0 Prototype Signal Research
```

Section 7 should run controlled event studies on `signal_frame`, especially:

```text
baseline forward R by horizon
POI touch behavior
first-touch versus later-touch behavior
swing window and break-mode comparisons
entry and stop model comparisons
session-specific results
volatility-regime-specific results
continuation versus reversal outcome labels
candidate signal ranking
```

Do not begin a full backtest until Section 7 has ranked which conditions show robust forward behavior.

---

# 6B Structural Swing Validation Refinement

## Status

Section 6B was added after the completed Section 6.0 baseline in:

```text
notebooks/exploration/exp1.ipynb
```

The implementation was added to:

```text
src/features/poi_signal_features.py
```

This is a non-destructive refinement. The original Section 6 Version A tables remain unchanged and continue to represent the baseline mechanical/local swing model.

## Version A Baseline

Version A remains:

```text
3-bar, 5-bar, and 7-bar local swing windows
wick-break and close-break variants
section6_poi_table_gc.parquet
section6_retest_table_gc.parquet
section6_candidate_trade_table_gc.parquet
section6_signal_frame_gc.parquet
```

## Version B Structural Layer

Version B starts from the existing Version A POIs and annotates each POI with higher-order structural swing validation.

Structural swing windows:

```text
15-bar swing
21-bar swing
31-bar swing
```

No previous-day highs/lows, session highs/lows, or calendar-derived levels are used. The structural definition is based only on confirmed price-action swing highs and lows inside the GC active-front series. Structural swing computation is continuous-segment aware so it does not cross contract/gap boundaries.

For each Version A POI:

```text
Bullish POI: check whether the displacement interval broke a prior structural swing high.
Bearish POI: check whether the displacement interval broke a prior structural swing low.
```

The structural check uses the POI's own `break_mode` (`wick` or `close`) so Version B remains directly comparable to the existing Version A wick/close variants.

If multiple structural windows are broken, the primary column stores the largest window broken. An additional helper column stores all windows broken.

## New Structural Columns

Added to Version B tables:

```text
structural_swing_break_flag
structural_swing_window_broken
structural_swing_windows_broken
structural_break_mode
structural_swing_bar_id
structural_swing_price
structural_break_bar_id
structural_break_distance_ticks
local_swing_only_flag
```

Column meanings:

```text
structural_swing_break_flag = True when the POI displacement broke a 15/21/31-bar structural swing.
structural_swing_window_broken = largest structural swing window broken.
structural_swing_windows_broken = all broken structural windows as a comma-separated label.
structural_break_mode = wick or close structural break mode.
structural_swing_bar_id = bar_id of the broken structural swing high/low.
structural_swing_price = price level of the broken structural swing.
structural_break_bar_id = bar_id of the candle that broke the structural swing.
structural_break_distance_ticks = distance beyond the structural swing level in ticks.
local_swing_only_flag = True when the Version A POI broke only the local 3/5/7-bar swing and no 15/21/31 structural swing.
```

## Saved Version B Tables

Saved separate enhanced outputs:

```text
data/processed/section6b_structural_poi_table_gc.parquet
data/processed/section6b_structural_retest_table_gc.parquet
data/processed/section6b_structural_candidate_trade_table_gc.parquet
data/processed/section6b_structural_signal_frame_gc.parquet
```

These files do not overwrite the Version A `section6_*` outputs.

## Full-Run Counts

Version A row counts preserved:

```text
Baseline POIs:          202,466
Baseline retests:     6,684,664
Baseline candidates:  1,597,373
Baseline signals:     1,597,373
```

Version B structural classification:

```text
Structurally validated POIs:          84,890
Local-swing-only POIs:               117,576
Structurally validated retests:    2,787,207
Local-swing-only retests:          3,897,457
Structurally validated candidates:   685,420
Local-swing-only candidates:         911,953
Structurally validated signals:      685,420
Local-swing-only signals:            911,953
```

Primary structural window counts:

```text
15-bar: 26,904
21-bar: 21,539
31-bar: 36,447
None:  117,576
```

## Validation Checks Passed

```text
poi_row_count_preserved:                      True
retest_row_count_preserved:                   True
candidate_row_count_preserved:                True
signal_row_count_preserved:                   True
structural_columns_present_poi:               True
structural_columns_present_retest:            True
structural_columns_present_candidate:         True
structural_columns_present_signal:            True
structural_partition_valid:                   True
validated_rows_have_structural_window:        True
local_only_rows_have_no_structural_window:    True
```

## Section 7 Implication

Section 7 should compare:

```text
all baseline POIs
local-swing-only POIs
structurally validated POIs
15-bar structural break POIs
21-bar structural break POIs
31-bar structural break POIs
```

The research question is whether higher-order structural breaks improve forward behavior versus small local 3/5/7-bar swing breaks.

Do not begin a full backtest until this structural split is evaluated through event studies.

---

# 7.0 Prototype Signal Research Progress

## Status

Section 7.0 is complete in:

```text
notebooks/exploration/exp1.ipynb
```

The reusable implementation was added to:

```text
src/features/poi_event_study.py
```

A reproducible runner was added to:

```text
scripts/run_section7_event_study.py
```

Section 7 is an event-study and signal-diagnostic section. It is not a full backtest and does not include an equity curve, dollar PnL simulation, prop-firm challenge simulation, trade sequencing, commissions, or slippage.

## Completed Subsections

```text
7.1 Section 7 Research Contract
7.2 Load and Validate Section 6 / 6B Signal Tables
7.3 Baseline Forward R Study
7.4 POI Touch Event Study
7.5 Version A vs Version B Structural Validation Study
7.6 Swing Window and Break Mode Study
7.7 Entry Variant and Stop Model Study
7.8 Session-Specific Results
7.9 Volatility-Regime-Specific Results
7.10 Volume / Relative Volume Filter Study
7.11 Trend, VWAP, Extension, and Exhaustion Context Study
7.12 Candidate Signal Ranking
7.13 Section 7 Summary and Backtest Candidates
```

## Inputs

Section 7 used the Section 6B structural signal frame as the main input:

```text
data/processed/section6b_structural_signal_frame_gc.parquet
```

It also used:

```text
data/processed/research_bars_gc_mgc_1m.parquet
```

The structural signal frame contains:

```text
Signal rows: 1,597,373
Columns:        129
```

The research bars file contains:

```text
Rows: 3,487,656
```

Section 7 reconstructs the GC-only `bar_id` ordering used by Section 6 before rebuilding forward path labels.

## Validation Checks Passed

```text
signal_required_columns_present:              True
bar_required_columns_present:                 True
signal_ids_unique:                            True
existing_5_15_30_60_forward_r_present:        True
structural_columns_present:                   True
structural_partition_valid:                   True
session_columns_present:                      True
entry_rows_in_approved_execution_windows:     True
no_candidate_entry_after_1200_ny:             True
risk_points_positive:                         True
valid_candidate_flag_all_true:                True
volatility_context_present:                   True
volume_context_present:                       True
gc_research_bars_available:                   True
```

## Forward Label Method

Forward labels were rebuilt for:

```text
5m, 15m, 30m, 60m, 120m, 180m, 240m, 360m
```

For each horizon, Section 7 calculates:

```text
fixed-horizon forward R
capped/forced-exit forward R
MFE_R
MAE_R
+1R, +2R, +3R, +4R, +5R, +7.5R, +10R, +15R hit rates
-1R hit rate
entry-bar and path-order ambiguity diagnostics
5R and 10R capped diagnostic versions
```

The holding-period rule is enforced:

```text
No candidate entry after 12:00pm New York time.
Positions entered before 12:00pm may continue after noon.
All event-study holding windows are capped at the same-day 3:30pm New York forced-exit boundary.
```

Fixed horizons are invalid when the requested horizon extends beyond the forced-exit boundary. Capped horizons use:

```text
effective_exit_time = min(entry_time + horizon, same-day 15:30 NY forced-exit time)
```

## Generated Outputs

Saved generated outputs:

```text
data/processed/section7_event_study_summary_gc.parquet
data/processed/section7_candidate_signal_ranking_gc.parquet
```

These are reproducible generated artifacts and remain excluded from Git.

Output sizes:

```text
section7_event_study_summary_gc.parquet:       176,560 rows
section7_candidate_signal_ranking_gc.parquet:      374 rows
```

Full Section 7 event-study runtime:

```text
593.71 seconds
```

## Horizon Validity

Capped horizon validity stayed high because late-morning entries were force-exit capped:

```text
5m capped valid rows:    1,597,238
15m capped valid rows:   1,596,853
30m capped valid rows:   1,596,406
60m capped valid rows:   1,594,815
120m capped valid rows:  1,590,235
180m capped valid rows:  1,583,476
240m capped valid rows:  1,576,988
360m capped valid rows:  1,578,933
```

Fixed horizons lose observations at longer holds:

```text
240m fixed valid rows: 1,442,819
360m fixed valid rows:   832,681
```

Forced-exit-capped rates:

```text
240m capped forced-exit rate:  8.40%
360m capped forced-exit rate: 46.72%
```

## Baseline Forward R Findings

The all-candidate POI stream is not a clean standalone edge.

Across capped horizons:

```text
Positive R rate: approximately 49.6% to 50.3%
Median R: near 0R across most horizons
Uncapped mean R: mixed to negative
5R-capped mean R: only slightly positive at most horizons
```

Runner potential is real, but adverse path risk is also large:

```text
+5R hit rate rises from 1.39% at 5m to 47.67% at capped 360m.
-1R path-hit rate rises from 36.65% at 5m to 87.69% at capped 360m.
```

Interpretation:

```text
Raw POI logic is an event generator, not a finished trading signal.
The discretionary observation that some POIs run massively is supported.
The data also shows that many candidates experience severe adverse excursion.
```

## First-Touch Findings

First touch is not automatically cleaner.

At the 60m capped horizon:

```text
First-touch valid candidates: 62,877
Mean R:                     -0.0533
5R-capped mean R:           -0.0093
Median R:                   -0.0286
Positive R rate:             49.34%
+5R hit rate:                15.94%
-1R path-hit rate:           72.98%
```

Interpretation:

```text
First-touch should be analyzed separately but should not be treated as a permission filter by default.
```

## Structural Validation Findings

Higher-order structural validation does not uniformly dominate local-only POIs.

At the 60m capped horizon:

```text
Structurally validated:
  5R-capped mean R:  0.0138
  Median R:         -0.0263
  -1R hit rate:      73.83%

Local-swing-only:
  5R-capped mean R:  0.0051
  Median R:          0.0182
  -1R hit rate:      75.83%
```

The 15-bar structural window was the strongest structural cohort at 60m:

```text
15-bar structural window:
  5R-capped mean R:  0.0968
  Median R:          0.1053
```

The 21-bar and 31-bar structural cohorts were weaker at the 60m horizon.

Interpretation:

```text
Structural validation should move forward as an interaction/filter candidate, especially the 15-bar cohort.
It should not be used as a blanket permission rule without a backtest.
```

## Swing Window and Break Mode Findings

The 60m capped study shows a strong side asymmetry:

```text
Bearish/short variants rank above bullish/long variants across swing and break-mode groups.
```

Examples at 60m capped:

```text
5-bar wick bearish:
  5R-capped mean R: 0.1316
  Median R:         0.0400

5-bar close bearish:
  5R-capped mean R: 0.1115
  Median R:         0.0217

Most bullish/long swing-break groups were negative after 5R capping.
```

Interpretation:

```text
Section 8 should start short-side-first rather than backtesting every direction equally.
```

## Entry Variant and Stop Model Findings

Short boundary entries are the cleanest broad entry family.

At 60m capped:

```text
Short boundary + POI distal edge:
  Valid rows:        198,571
  5R-capped mean R:  0.1261
  Median R:          0.0263
  +5R hit rate:      24.77%
  -1R hit rate:      74.52%

Short boundary + conservative adjacent:
  Valid rows:        340,089
  5R-capped mean R:  0.0895
  Median R:          0.0000
```

Midpoint/distal short entries can raise runner rates, but they also raise adverse path risk. Long entry/stop variants are broadly weaker.

## Session Findings

At 60m capped:

```text
London shorts:
  5R-capped mean R: 0.1748
  Median R:         0.1194

New York shorts:
  5R-capped mean R: 0.0683
  Median R:        -0.0390

London longs:
  5R-capped mean R: -0.1830
  Median R:         -0.1111
```

New York shorts improve at longer capped horizons:

```text
120m: 5R-capped mean R 0.1718
180m: 5R-capped mean R 0.2537
240m: 5R-capped mean R 0.2618
360m: 5R-capped mean R 0.2163
```

Interpretation:

```text
London short candidates behave more like quicker 60m reactions.
New York short candidates show more 2-4 hour runner behavior.
```

## Volatility Findings

Volatility is a meaningful interaction variable.

At 60m capped:

```text
Extreme-volatility shorts:
  5R-capped mean R: 0.1271
  Median R:         0.0286

Extreme-volatility longs:
  5R-capped mean R: -0.1105
  Median R:         -0.0328

Elevated-volatility longs:
  5R-capped mean R: 0.1313
  Median R:         0.0769
```

Interpretation:

```text
Extreme volatility favors short-side candidates in the broad event study.
Elevated-volatility longs are one of the few broad long-side families worth a limited Section 8 test.
```

## Volume Findings

Relative volume helps explain movement magnitude, especially on shorts.

At 60m capped:

```text
Normal relative-volume shorts:
  5R-capped mean R: 0.1636

High relative-volume shorts:
  5R-capped mean R: 0.1633

Spike relative-volume shorts:
  5R-capped mean R: 0.0343
  -1R hit rate:     78.69%
```

Interpretation:

```text
Volume spike flags alone increase movement but do not cleanly improve directional edge.
Volume should be tested as a sizing/target/hold-time interaction rather than as a standalone permission filter.
```

## VWAP, Trend, Extension, and Context Findings

At 60m capped:

```text
Shorts near VWAP:
  5R-capped mean R: 0.1787
  Median R:         0.1351

Shorts moderately extended from VWAP:
  5R-capped mean R: 0.1535
  Median R:         0.0800

Very-extended shorts:
  5R-capped mean R: 0.0085
  Median R:        -0.1071

Longs against VWAP:
  5R-capped mean R: -0.1590
```

Trend alignment is not a universal positive filter. Broad short candidates against the 20/50 trend bias performed better than aligned shorts at 60m capped.

POI-quality buckets generated some high-mean small-sample groups, but those are hypothesis-generation only until they pass minimum sample thresholds.

## Candidate Ranking Findings

Candidate ranking used:

```text
sample size
5R-capped mean R
median R
downside quantile
positive R rate
MFE/MAE behavior
+2R/+3R/+5R hit rates
-1R hit rate
consistency across horizons
```

Minimum sample threshold:

```text
1,000 rows
```

Important conclusion:

```text
No composite group earned a clean robust approval.
The highest-ranked groups still had negative robust scores because downside quantiles and -1R path-hit rates remained severe.
```

The top-ranked groups still showed positive capped mean/median behavior and strong runner rates, so they are useful Section 8 research candidates, not production rules.

## Section 8 Candidate Scope

Recommended first Section 8 research-backtest candidates:

```text
1. Short-only POI candidates, especially New York extreme-volatility shorts.
2. Short boundary entries with POI-distal or conservative stops.
3. London short 60m reaction candidates.
4. New York short 120-240m runner candidates with forced-exit-aware holding.
5. 15-bar structural validation as an interaction/filter candidate.
6. Elevated-volatility long candidates as the only broad long-side family worth limited testing.
```

Rejected or deprioritized for first Section 8 scope:

```text
1. The raw all-candidate POI stream.
2. First-touch as a standalone permission filter.
3. London long candidates.
4. Long candidates against VWAP.
5. Broad extreme-volatility long candidates.
6. Volume-spike-only permission filters.
7. Tiny POI-quality buckets with attractive means but insufficient sample size.
```

## Current Project Status

| Phase                         | Status      |
| ----------------------------- | ----------- |
| Data Acquisition              | Complete    |
| Data Validation               | Complete    |
| Exploratory Analysis          | Complete    |
| Research Dataset Construction | Complete    |
| Strategy Formalization        | Complete    |
| POI Feature Engineering       | Complete    |
| Structural Swing Validation   | Complete    |
| Event Studies                 | Complete    |
| Backtesting                   | Next Phase  |
| Robustness Testing            | Not Started |

## Next

Proceed to:

```text
8.0 First Research Backtest
```

Section 8 should not backtest every raw POI candidate. It should start with the narrow candidate scopes identified above, preserve the 12:00pm no-new-entry rule, and enforce the 3:30pm New York forced-exit rule.

---

# 6C POI Selection Refinement and 7R Refined Event Study

## Status

Section 6C and Section 7R are complete. Section 8 has not begun.

This work is non-destructive. The original Section 6 Version A, Section 6B Version B, and Section 7 files were signature-checked before and after the full run and remained untouched. Their results are retained for historical comparison, but the original Section 7 conclusions are now **legacy/superseded for Section 8 decision-making**.

Reusable implementation:

```text
src/features/poi_selection_refinement.py
src/features/poi_refinement_visuals.py
src/features/poi_event_study_refined.py
scripts/run_section6c_refinement.py
scripts/run_section7r_event_study.py
```

## Exact Section 6C Definition

For each possible formation:

```text
A = candle i-1
B = candle i, the POI middle candle
C = candle i+1, the confirmation candle
```

A/B/C must be consecutive one-minute GC observations from the same active contract, New York date, and continuous segment. They must pass the existing tradability and rollover-integrity requirements.

Classic FVG:

```text
Bullish: low[C] > high[A]
Bearish: high[C] < low[A]
```

Close-to-next-open gap:

```text
Bullish: open[C] > close[B]
Bearish: open[C] < close[B]
Minimum = 1 GC tick
```

Strict confirmation-body survival:

```text
Bullish: close[C] >= open[C]
Bearish: close[C] <= open[C]
Doji close at C's open passes.
```

If C closes against the opening-gap direction, the formation is rejected as:

```text
confirmation_close_inside_opening_gap
```

This rule is stricter than merely requiring the gap not to be fully filled.

## Tick-Safe FVG Threshold

GC tick size:

```text
1 tick = 0.10 points
3 ticks = 0.30 points
4 ticks = 0.40 points
5 ticks = 0.50 points
```

Prices are mapped to nearest integer tick indices before FVG threshold comparisons. Binary floating-point equality is not used.

Two-stage validity:

```text
formation_valid_flag = continuity + classic FVG + close-open gap +
                       matching direction + confirmation survival +
                       search-window + rollover integrity

final_poi_valid_flag = formation_valid_flag AND fvg_size_ticks >= 3
```

The default minimum is 3 ticks. The 4-tick and 5-tick flags are predefined sensitivity cohorts in the same output tables; they are not post-performance parameter choices.

## Case 1 / Case 2 / Case 3 Geometry

Case 1 expanded:

```text
Bullish when open[C] > high[B]: [low[B], open[C]]
Bearish when open[C] < low[B]:  [open[C], high[B]]
```

Case 2 standard:

```text
The opening gap is contained by B.
POI = [low[B], high[B]]
```

Case 3 invalid:

```text
C closes against its opening-gap direction.
No valid POI, retest, candidate, or signal row is produced.
```

Activation and no-lookahead:

```text
confirmation_bar_id = C
activation_bar_id = max(confirmation_bar_id, swing_break_bar_id)
retest_bar_id > activation_bar_id
```

## Canonical Identity

Variant rows remain available for `swing_n` and `break_mode` research, but the physical formation is identified independently of those parameters:

```text
canonical_poi_id = product + direction + A/B/C bar identity
poi_variant_id = local swing/break POI variant
canonical_retest_id = canonical POI + retest bar
candidate_variant_id = entry/stop candidate row
canonical_candidate_id = canonical retest + entry variant + stop model
```

Raw swing/break rows are correlated views of the same market event. Headline Section 7R statistics use canonical candidate variants.

## Saved Section 6C Outputs

```text
data/processed/section6c_refined_poi_audit_gc.parquet
data/processed/section6c_refined_poi_table_gc.parquet
data/processed/section6c_refined_retest_table_gc.parquet
data/processed/section6c_refined_candidate_trade_table_gc.parquet
data/processed/section6c_refined_signal_frame_gc.parquet
data/processed/section6c_refinement_summary_gc.parquet
```

Every Section 6C row carries:

```text
poi_definition_version = 6C_v1
```

## Full-Run Formation Funnel

```text
Variant proposals re-evaluated:             202,466
Genuine classic FVGs:                       202,466
Genuine close-open gaps:                    202,466
Rejected confirmation-body closures:         39,391
Formation-valid before 3-tick threshold:    163,075
Rejected below 3 ticks:                      70,106
Final >=3-tick POI variants:                 92,969
Final >=4-tick POI variants:                 70,810
Final >=5-tick POI variants:                 54,846
Case 1 expanded:                             28,973
Case 2 standard:                             63,996
Unique canonical physical POIs:              24,105
```

The audit also retains physical diagnostic rejections without multiplying them across correlated swing/break variants:

```text
no_classic_fvg:                             429,525
no_close_open_gap:                          114,371
rollover_integrity_failure:                  27,074
segment_boundary_crossed:                     1,732
confirmation_close_inside_opening_gap:       39,391
fvg_below_3_ticks:                           70,106
```

Final POI breakdown:

```text
Bullish:               46,345
Bearish:               46,624
London activation:     25,169
New York activation:   43,018
Outside execution:     24,782
3-bar variants:        28,556
5-bar variants:        31,684
7-bar variants:        32,729
Wick break:            49,422
Close break:           43,547
```

Exact FVG tick buckets:

```text
3 ticks:       22,159
4 ticks:       15,964
5 ticks:       11,474
6-9 ticks:     23,476
10-19 ticks:   14,421
20+ ticks:      5,475
```

## Structural and Downstream Counts

Structural POI variants:

```text
Structurally validated: 40,823
Local-swing-only:       52,146
Primary 15-bar:         12,806
Primary 21-bar:         10,383
Primary 31-bar:         17,634
```

Rebuilt downstream objects:

```text
POI variants with retests:             82,538
Retest variant rows:                 3,185,556
Unique canonical retests:              853,620
First-touch variant rows:               82,538
Later-touch variant rows:            3,103,018
Valid candidate/signal variants:     1,080,394
Unique canonical candidate variants:   302,020
Unique canonical POIs in signals:         7,433
Unique canonical retests in signals:    179,036
Unique trading dates:                       882
```

Structural downstream split:

```text
Retests — structural: 1,380,040; local-only: 1,805,516
Signals — structural:   471,164; local-only:   609,230
```

Version comparison, without treating the rows as independent:

```text
Version A POIs:        202,466
Version B POIs:        202,466
Version C POIs:         92,969

Version A signals:   1,597,373
Version B signals:   1,597,373
Version C signals:   1,080,394
```

## Nineteen-Case Regression and False-FVG Audit

Visual output:

```text
reports/figures/section6c_poi_refinement/
```

Artifacts:

```text
19 hand-labelled charts
hand_labelled_regression_manifest.csv
hand_labelled_abc_ohlc.csv
13-stratum random visual audit
random_stratified_audit_manifest.csv
```

Regression result:

```text
PASS:   16
REVIEW:  3
```

The three REVIEW cases are raw-OHLC label conflicts, not implementation exceptions:

### GC_SIG_00433067

```text
A 2025-06-13 09:11 NY: O=3457.7 H=3458.5 L=3456.6 C=3457.2
B 2025-06-13 09:12 NY: O=3457.0 H=3459.7 L=3457.0 C=3459.7
C 2025-06-13 09:13 NY: O=3459.9 H=3461.5 L=3459.7 C=3461.4
```

`low[C]=3459.7 > high[A]=3458.5`, and C closes above its open. Actual result: valid 12-tick bullish Case 1 expanded.

### GC_SIG_00384676

```text
A 2025-05-13 09:11 NY: O=3249.5 H=3250.2 L=3249.0 C=3249.0
B 2025-05-13 09:12 NY: O=3248.4 H=3249.1 L=3246.9 C=3247.6
C 2025-05-13 09:13 NY: O=3247.4 H=3248.6 L=3247.0 C=3248.6
```

`high[C]=3248.6 < low[A]=3249.0`, so a genuine 4-tick bearish FVG exists. The formation is still invalid because C closes above its open and therefore back inside the bearish opening gap.

### GC_SIG_01388071

```text
A 2026-04-09 08:45 NY: O=4769.6 H=4769.6 L=4767.4 C=4769.1
B 2026-04-09 08:46 NY: O=4768.7 H=4769.2 L=4767.0 C=4767.6
C 2026-04-09 08:47 NY: O=4767.1 H=4767.1 L=4762.6 C=4763.5
```

`high[C]=4767.1 < low[A]=4767.4`, so a genuine 3-tick bearish FVG exists. C closes below its open. Actual result: valid 3-tick bearish Case 2 standard.

The original audit plots marked the actual B candle correctly, but the title emphasized “POI created” (confirmation time) and did not shade or label the A/C FVG interval. The new charts distinguish B, C, activation, and retest times and explicitly draw the FVG, opening gap, middle range, corrected zone, and any extension.

## Performance

Section 6C stage timings:

```text
Feature preparation:       4.23 seconds
POI formation audit:      82.64 seconds
Final POI filtering:       0.24 seconds
Structural annotation:    88.93 seconds
Retest construction:      43.98 seconds
Candidate construction:   22.56 seconds
Full Section 6C:         249.96 seconds
```

The expensive stages are unavoidable interval/state calculations over 1.76 million bars. Formation measurements use shifted arrays and integer tick-space comparisons. Retests reuse grouped day-level interval matrices and chunking rather than scanning once per POI. The 3/4/5-tick cohorts are flags in one signal frame rather than duplicated materialized datasets.

## Section 7R Outputs

Section 7R input:

```text
data/processed/section6c_refined_signal_frame_gc.parquet
```

Saved outputs:

```text
data/processed/section7r_event_study_summary_gc.parquet
data/processed/section7r_candidate_signal_ranking_gc.parquet
data/processed/section7r_fvg_threshold_comparison_gc.parquet
```

Output sizes:

```text
Event-study summary rows:       291,984
Candidate-ranking rows:             248
FVG-threshold comparison rows:      576
Raw variant signal rows:      1,080,394
Canonical candidate variants:   302,020
```

Runtime:

```text
Path-label construction:  44.56 seconds
Full Section 7R:          661.30 seconds
```

## Section 7R Headline Findings

Headline population: canonical candidate variants.

Canonical baseline at capped 60 minutes:

```text
Valid rows:                  301,601
Mean R:                       0.0565
Mean R capped at +/-5:        0.0531
Median R:                     0.0545
Positive-R rate:             50.56%
+3R hit rate:                39.16%
+5R hit rate:                20.96%
-1R path-hit rate:           74.82%
```

The baseline retains mild positive central tendency but still has severe two-sided path risk. It is an event generator, not a clean standalone signal.

The refined data independently reproduces the broad short-side asymmetry at capped 60 minutes:

```text
Short mean R capped at +/-5:  0.1259
Long mean R capped at +/-5:  -0.0157
```

Across capped horizons, short robust mean R remains positive from 5 through 360 minutes; long robust mean R is negative at every tested horizon.

Geometry at capped 60 minutes:

```text
Case 2 standard short:  0.1666 capped mean R
Case 1 expanded short: -0.0383 capped mean R
Case 2 standard long:  -0.0144 capped mean R
Case 1 expanded long:  -0.0202 capped mean R
```

Correctly expanded Case 1 events do not inherit the stronger Case 2 short result. Case 1 short outcomes have positive uncapped mean but negative capped mean and median, indicating outlier-driven behavior.

Structural cohorts at capped 60 minutes:

```text
15-bar:     0.1226 capped mean R; 0.1622 median R
21-bar:     0.0326 capped mean R; 0.0000 median R
31-bar:     0.0302 capped mean R; -0.0370 median R
Local-only: 0.0533 capped mean R; 0.0923 median R
```

The 15-bar cohort is independently reproduced as the strongest 60-minute structural window. Structural validation in aggregate does not dominate local-only, so 15-bar remains an interaction hypothesis rather than a blanket permission rule.

FVG threshold comparison at capped 60 minutes:

```text
Short >=3 ticks: 0.1259 capped mean R
Short >=4 ticks: 0.1298 capped mean R
Short >=5 ticks: 0.1350 capped mean R

Long >=3 ticks: -0.0157 capped mean R
Long >=4 ticks: -0.0180 capped mean R
Long >=5 ticks: -0.0209 capped mean R
```

The exact tick buckets are non-monotonic. A stricter minimum modestly improves broad shorts while worsening broad longs; no universal threshold above the locked 3-tick default is selected.

Session at capped 60 minutes:

```text
London short:    0.2894 capped mean R
New York short:  0.0829 capped mean R
New York long:   0.0026 capped mean R
London long:    -0.0868 capped mean R
```

London short strength and London long weakness are independently reproduced.

Canonical versus raw variant baseline at capped 60 minutes:

```text
Canonical capped mean R: 0.0531
Raw-row capped mean R:   0.0289
Canonical median R:      0.0545
Raw-row median R:        0.0303
```

Duplicate variant weighting materially changes headline estimates. Raw swing/break rows must be described as correlated parameter variants, not independent market events.

Candidate ranking:

```text
No composite group has a positive robust rank score.
Top canonical rank score: -0.2471
```

The highest-ranked canonical group is a New York extreme-volatility short, local-only, 3-bar close-break, boundary entry, POI-distal stop, normal relative volume cohort. It remains a research hypothesis, not approval for Section 8.

## Validation

Passed:

```text
29 Section 6C synthetic regression tests
Legacy Section 6 tests
Legacy and refined Section 7 tests
All modified module/script py_compile checks
All Section 6C acceptance-oriented full-run checks
All Section 7R input, canonical-deduplication, session, path, and forced-exit checks
Legacy Version A/B and Section 7 outputs unchanged
Rejected formations absent from all downstream tables
```

## Current Project Status and Next Recommended Step

| Phase                         | Status                                      |
| ----------------------------- | ------------------------------------------- |
| Data Acquisition              | Complete                                    |
| Data Validation               | Complete                                    |
| Exploratory Analysis          | Complete                                    |
| Research Dataset Construction | Complete                                    |
| Strategy Formalization        | Complete                                    |
| POI Feature Engineering       | Section 6C complete                         |
| Structural Swing Validation   | Reapplied to Section 6C                     |
| Event Studies                 | Section 7R complete                         |
| Backtesting                   | Not started; awaiting review/authorization  |
| Robustness Testing            | Not started                                 |

Next recommended step:

```text
Review the three raw-OHLC label conflicts and the canonical Section 7R findings.
Do not begin Section 8 until that review is complete and Section 8 scope is explicitly authorized.
```

---

# Authoritative Current Project Map — 2026-07-12

This final section is the current-state override for older milestone/status statements retained above for research traceability.

```text
Section 6: Final refined POI definition and machine-readable signal tables — complete.
Section 7: True POI context and conditional-signal research — complete; no Section 8 candidate approved.
Original baseline and prototype event studies — retained for research history but superseded.
Section 8: Sequential backtesting — not started.
```

Notebook reading order:

```text
Sections 1-5 — Data, EDA, research dataset, and strategy formalization
Section 6 — Complete POI-engine development and authoritative refined baseline
Section 7 — Completed True POI context, first-passage, out-of-sample, and candidate-policy research
Appendix A — Legacy prototype event study, not current independent evidence
```

Current research contract:

```text
The refined True POI definition is frozen as the authoritative project baseline.
Section 7 used the refined Section 6 signal frame and completed the market-context research phase.
The research separates continuation, reversal, and no-trade conditions at the True POI/retest level.
No full sequential backtest has started, and no Section 8 candidate is approved.
```

---

# Authoritative True POI Context Research Completion — 2026-07-12

This entry supersedes older current-phase statements while preserving every historical progress entry above.

## Status, Terminology, and Locked Time Rules

```text
Section 6: complete and frozen.
Section 7: complete as an event-level context and first-passage policy study.
Section 8: not started.
Section 8 approval: none.
```

Reader-facing terminology is now `True POI`, `true_poi_id`, `true_retest_id`, and `true_trade_opportunity_id`. A True POI is one unique refined A/B/C market formation after duplicate swing-window and break-mode representations are removed. “True” means the unique underlying formation; it does not imply profitability, validation, or a guaranteed reaction. Existing `canonical_*` columns and internal `section6c_*` / `section7r_*` filenames remain legacy compatibility and reproducibility fields. Exact `true_*` aliases were added to reusable refinement code and all new Section 7 objects.

The authoritative timing distinction is:

```text
POI search: 1:00am-12:00pm New York
London entry: 3:00am-6:00am New York
New York entry: 7:00am-12:00pm New York
12:00pm: no new entries and True POI expiry
3:30pm: mandatory exit for positions already entered
No overnight positions
```

The stale noon-flat statement in notebook Section 5.8 was corrected without otherwise rewriting completed Section 5.

## Implementation and Generated Objects

New reusable implementation:

```text
src/features/poi_context_features.py
src/features/poi_first_passage.py
src/research/poi_context_event_study.py
scripts/run_section7_poi_context_research.py
scripts/update_section7_notebook.py
tests/test_poi_context_features.py
tests/test_poi_first_passage.py
tests/test_poi_context_event_study.py
```

Generated outputs:

```text
section7_true_poi_context_frame_gc.parquet
section7_true_poi_outcome_labels_gc.parquet
section7_true_poi_feature_study_summary_gc.parquet
section7_true_poi_interaction_summary_gc.parquet
section7_true_poi_stop_target_summary_gc.parquet
section7_true_poi_candidate_ranking_gc.parquet
section7_true_poi_backtest_candidate_registry_gc.parquet
section7_true_poi_location_quality_summary_gc.parquet
reports/section7_true_poi_feature_registry.csv
reports/section7_true_poi_backtest_candidate_registry_gc.csv
```

Generated Parquet and CSV artifacts remain excluded from Git.

## Full-Run Counts and Partitions

```text
Legacy refined signal rows read:       1,080,394
Unique True POIs:                           7,433
Unique True POI/retest opportunities:     179,036
Paired directional label rows:            358,072
Unique trading dates:                         882
Feature registry rows:                        264
```

Chronological partitions:

```text
Development through 2023-12-31: 776 True POIs; 13,643 retests; 352 dates
Validation calendar 2024:        694 True POIs; 11,487 retests; 194 dates
Final test through 2026-05-22:  5,963 True POIs; 153,906 retests; 336 dates
```

Actual eligible retests begin 2021-05-26. Research variants remain attributes and flags, never independent headline samples.

## Feature Registry and No-Lookahead Contract

The registry covers every one of the 264 delivered `feat_*` columns: 29 formation/geometry, 24 displacement, 132 approach, 23 touch-interaction, 49 market-context, four retest-state, and three session-context fields. Approach windows are 3/5/10/15/30 completed minutes ending at `touch_bar - 1`. Market context includes past-only VWAP, trend, volatility, volume, time-of-day baseline, session state, and extension/exhaustion.

Touch-close fields may only be used for next-bar confirmation entries. Automated checks require:

```text
formation availability bar < retest bar
pre-touch availability bar  < retest bar
touch-close availability bar == retest bar
```

Development quantiles and volatility thresholds are fit only on development data and applied unchanged to validation/test. Forward labels never enter the `feat_*` namespace.

## Location Quality and Directional Outcomes

True POI retests were compared with non-POI controls matched by partition, session, 30-minute time bin, development-fitted volatility bucket, and recent-move bucket.

Median 60-minute expansion in one-minute ATR units:

```text
                 True POI    Matched control
Development        5.196          4.778
Validation         5.256          5.099
Final test         5.944          5.560
```

True POIs are modestly better expansion locations, but the difference is small relative to matched state. Both samples exceed 0.5/1.0 one-minute ATR almost universally at 60 minutes, so those thresholds are not discriminative. Location quality does not imply direction.

Robust ±5R 60-minute directional means:

```text
Continuation short: development 0.094; validation 0.137; final test 0.185
Continuation long:  development 0.046; validation -0.064; final test 0.124
Reversal short:      development 0.195; validation 0.124; final test -0.105
Reversal long:       development 0.116; validation -0.185; final test -0.060
```

Broad reversal does not survive final test or ordered execution. Continuation short remains the strongest broad directional family, but a True POI touch is not an automatic short.

## First Passage, Stops, Targets, and Holding

The reusable first-passage engine records entry fill, target-before-stop, stop-before-target, time/mandatory exit before either, no fill, invalid path, same-bar and entry-bar ambiguity, barrier times, forced-exit R, and maximum favorable/adverse R. Paths cannot cross dates, contracts, continuous segments, or invalid entry/forced-exit boundaries.

Ambiguity treatments are conservative stop-first for headlines, ambiguity-excluded sensitivity, and optimistic target-first sensitivity. At the primary 2R True-POI-invalidation continuation policy, same-bar ambiguity was 4.16% development, 4.89% validation, and 17.13% final test. Final-test mean changed from -0.169R conservative to +0.054R excluded and +0.464R optimistic. That sensitivity blocks approval of fragile same-bar policies.

Stops tested:

```text
True POI invalidation plus one tick
Recent pre-entry micro-swing
Volatility-adjusted hybrid
Touch/rejection candle for next-bar entry
Legacy conservative-adjacent reference
```

Continuation at 2R and a 240-minute maximum hold:

```text
Volatility hybrid:  0.056R development; 0.039R validation; 0.043R test
Recent micro-swing: 0.020R development; 0.029R validation; 0.032R test
True POI stop:     -0.004R development; -0.034R validation; -0.169R test
Touch-candle stop: -0.108R development; -0.094R validation; -0.093R test
```

All broad medians and 25th percentiles remained -1R. Only about 34% of validation volatility-hybrid risks met the legacy 25-100 tick range; out-of-range rows were reported, not discarded.

Targets tested were 1/1.5/2/3/4/5/7.5/10/15R. Time exits were 15/30/60/120/180/240 minutes plus the 3:30pm mandatory exit. Higher target means were often rare-runner driven while median and downside stayed at the stop, so no universal target was selected.

## Conditional Features and Pre-Specified Interactions

Retained for conditional or risk use, not standalone direction:

```text
Opening-gap / POI-width ratio
15-minute candle overlap
Time-of-day-adjusted relative volume
Displacement efficiency and relative volume
POI age and prior-touch count
Volatility and session-extension measures
```

Low 15-minute overlap produced robust continuation means of 0.158R/0.231R/0.191R across development/validation/test, but final-test monotonicity failed. High time-of-day-adjusted relative volume produced 0.155R/0.206R/0.189R and was also non-monotonic.

Pre-specified continuation interactions that helped in validation and test were 15-bar structure x displacement quality, volatility x normalized stop width, early-London short context, and standard geometry x normalized FVG. Their strongest validation cohorts had limited date coverage and remain hypotheses. Displacement-volume x approach-volume, displacement-quality x approach-quality, compression x continuation, and generic trend alignment failed stability. Broad screens use trading-date block bootstrap intervals and Benjamini-Hochberg development q-values.

## Exact Candidate Policies and Decisions

Four policies passed minimum development/validation freeze coverage:

```text
S7P02_NY_BEAR_CONT — RESEARCH_ONLY
New York; bearish True POI continuation short; Case 2; compressed 15m approach;
boundary entry at first-contact edge; volatility-hybrid stop; 3R target;
240m maximum hold; 15:30 forced exit; conservative ambiguity.
Development +0.102R; validation +0.122R; final test +0.037R.
Median and Q25 = -1R in every partition. Final-test stop-before-target = 58.31%.

S7P05_NY_BULL_CONT — REJECT
Validation +0.018R; final test -0.077R.

S7P04_NY_BULL_REV_CONFIRM — REJECT
Development -0.153R; validation -0.059R; final test -0.072R.

S7P06_NY_BEAR_REV — REJECT
Development -0.132R; validation -0.154R; final test -0.152R.
```

Final decision:

```text
No Section 8 candidate approved.
```

The leading policy remains `RESEARCH_ONLY` because its positive mean is runner-dependent and its central/downside outcomes fail the advancement standard.

## Runtime, Validation, and Limitations

Main full-run runtime was 321.1 seconds. The matched-control pass added 8.7 seconds. Peak observed RSS was approximately 4.05 GB during first passage. The engine uses one sorted bar index, chunks event/path matrices, and reuses prepared bar arrays across policies.

Validation covered True POI/retest deduplication and alias consistency; formation/pre-touch/touch-close timing; continuation/reversal side mapping; first/later touch identity; entry fill and barrier ordering; same-bar/entry-bar ambiguity; time/forced exits; noon, date, contract, and segment boundaries; zero denominators and missing history; development-bin reuse; legacy Section 6 and refined Section 7 regression tests; compilation; and execution of every new notebook code cell against saved outputs.

Known limitations:

```text
1-minute OHLCV cannot resolve intrabar ordering.
OHLCV volume is not aggressor-side order-flow delta.
Retests from one True POI and overlapping paths remain correlated despite grouped/date-block summaries.
The final-test population is much larger because qualifying signals concentrate in 2025-2026.
The 25-100 tick compatibility range excludes much of several meaningful stop families.
Event-level expectancy is not a sequential backtest; re-entry and overlapping positions are not sequenced.
GC-to-MGC execution mapping, costs, sizing, portfolio state, and Section 8 remain out of scope.
```

---

# Authoritative Branch B Update — Section 7 Univariate Feature Evaluation Complete — 2026-07-20

This entry updates the independent statistical branch status without altering any POI-branch statement above.

```text
Branch B Section 7 (univariate feature evaluation): COMPLETE — SECTION 7 STATUS: READY
Evaluation partitions: Development + Validation only; all 162,224 Final-test rows excluded at frame construction.
Result: 0 ADVANCE_DIRECTIONAL, 55 ADVANCE_EXPANSION, 28 WEAK_UNSTABLE, 0 NO_EVIDENCE across 83 evaluated predictors.
No univariate feature produced an advancement-grade signed-return relationship; volatility/opportunity
forecasting structure is strong and Validation-confirmed.
Next Branch B step: Section 8.0 — Redundancy and Incremental Information.
Branch A status unchanged: no Section 8 POI backtest candidate approved.
```

Full details, criteria, caveats, and artifact paths: `project_docs/statistical_feature_research_context_report.md` (Section 7 completion entry) and `reports/statistical_research/summaries/section7_univariate_evaluation_summary.md`.

# Authoritative Branch B Update — Section 8 Redundancy and Incremental Information Complete — 2026-07-20

```text
Branch B Section 8 (redundancy and incremental information): COMPLETE — SECTION 8 STATUS: READY
55 expansion advancers -> 30 Development-fitted clusters -> 30 representatives ->
anchor (atr_20) + 14 confirmed-incremental representatives = 15 frozen expansion features.
Frozen directional feature set: explicitly empty (no Section 7 directional advancer).
Next Branch B step: Section 9.0 — Multivariate Research (linear benchmarks vs anchor-only baseline).
Branch A status unchanged: no Section 8 POI backtest candidate approved.
```

Details: `project_docs/statistical_feature_research_context_report.md` (Section 8 completion entry) and `reports/statistical_research/summaries/section8_redundancy_summary.md`.

# Authoritative Branch B Update — Sections 9-11 Complete; Standalone Statistical System Rejected — 2026-07-20

```text
Section 9 (multivariate benchmarks):    COMPLETE — model beats anchor in 3 of 4 cells; NY 60m calibration flagged.
Section 10 (signal construction):       COMPLETE — benchmark directions + frozen opportunity gate; rules frozen first.
Section 11 (sequential backtest):       COMPLETE — 86,353 trades, all four variants REJECTED under base costs.
Decision: the standalone Branch B statistical system has no directional edge and is REJECTED.
Validated assets that carry forward: the frozen 15-feature expansion set and per-session opportunity models.
Next: Section 12 integration research — POI direction candidates filtered/sized by statistical opportunity models.
Branch A status unchanged: no POI Section 8 backtest candidate approved. Final test locked throughout.
```

Details: the Section 9-11 completion entries in `project_docs/statistical_feature_research_context_report.md` and the section summaries under `reports/statistical_research/summaries/`.

# Authoritative Update — Section 12 Hybrid Integration Complete — 2026-07-20

```text
Section 12 (hybrid integration, gate-filter form): COMPLETE — NO CONFIRMED INCREMENTAL VALUE.
The frozen Branch B opportunity gate does not reliably improve any True POI direction family.
Continuation short looked strong on small Dev/Val gated samples but collapsed to -0.02R on the
one-time 301,610-event Final-test read; the pre-declared sample floors correctly withheld it.
Open next steps: sizing/no-trade integration forms, POI-x-statistical interactions, or the
Branch A Section 8 backtest authorization decision. Final test discipline maintained throughout.
```

Details: Section 12 completion entry in `project_docs/statistical_feature_research_context_report.md` and `reports/statistical_research/summaries/section12_hybrid_integration_summary.md`.

# Authoritative Update — Section 8 POI Sequential Backtest Complete — 2026-07-20

```text
Section 8 (S7P02 sequential research backtest): COMPLETE — SEQUENTIAL_REJECTED at base costs.
Frictionless +0.138/+0.117/+0.063 (Dev/Val/Final) reproduces the Section 7 event evidence; the
2.6-tick base cost load flips every partition negative (-0.003/-0.035/-0.059). 3,766 trades,
entry-realism enforced, conservative ambiguity, verdict fixed on Dev+Val before the Final read.
Phase 2 for Branch A is closed. Per project_docs/section8_authorization_memo.md, the family's
remaining path is the frozen Section 12B contract; failing that, Option 3 archives it.
```

Details: `reports/statistical_research/summaries/section8_poi_sequential_backtest_summary.md`.

# Authoritative Update — Section 12B Opportunity Conditioning Complete; S7P02 Family Archived — 2026-07-20

```text
Section 12B (opportunity-conditioned sizing/exits/suppression): COMPLETE — NO_ADVANCE on all
three pre-declared hypotheses. H1 proportional sizing shows the model's most persistent positive
signal (mean/MAD ratio effects with 0.81-1.07 Validation retention) but fails the base-cost
bootstrap interval and the 0.05R mean-shift materiality bound - the lift is high-quintile return
concentration, the effect Section 12 already rejected. H2 exit conditioning and H3 suppression
both flip sign in Validation. One-time Final-test read: every effect within 0.03R of zero.
Per the authorization memo linkage (Option 1 SEQUENTIAL_REJECTED + Section 12B negative),
Option 3 applies: the S7P02 continuation-short family is ARCHIVED with its evidence chain.
The Phase 2 research program is closed for both branches. Remaining PRD work is engineering:
GC-to-MGC transfer validation, prop-firm rules, forward-test scaffolding. Final test locked
throughout; any new research question requires a fresh pre-declared contract.
```

Details: Section 12B completion entry in `project_docs/statistical_feature_research_context_report.md` and `reports/statistical_research/summaries/section12b_opportunity_conditioning_summary.md`.

# Authoritative Update — Phase 4 Execution Layer: MGC Transfer Validated, Prop-Firm Engine Built — 2026-07-20

```text
FR-09 (GC-to-MGC transfer validation): COMPLETE — G5_PROVISIONAL_FAIL. Synchronization and
liquidity at the 179,036 POI decision bars are excellent (100 percent coverage, median basis one
tick, median 274 MGC contracts per decision minute, zero-volume never), but the GC first-contact
price trades on MGC in the same minute only 80.2 percent of the time - stable 75.5-80.9 percent
in every year, so the gap is structural, not legacy liquidity. Tick-precise GC entries cannot be
assumed fillable on MGC; an MGC-native entry treatment (tolerance band or marketable entry) with
telemetry-measured cost is required. Provisional thresholds were declared before computation;
final G5 numbers await lead approval.
FR-10 (prop-firm rules engine): COMPLETE - versioned policy layer (profit target, daily loss,
static/trailing/EOD drawdown with initial-balance cap, consistency, min/max days, internal risk
buffers), deterministic evaluation simulator with day-by-day ledgers, and seeded bootstrap
estimator of pass/breach probabilities. Bundled policies are illustrative templates; real firm
terms must be re-verified before use. No strategy is attached - none is approved.
FR-11 (Rithmic paper integration): NOT STARTED - requires credentials and platform approval.
Implementation: src/execution/{mgc_transfer.py, prop_firm_rules.py}, 18 synthetic tests,
scripts/run_mgc_transfer_validation.py, reports/execution/mgc_transfer_validation_summary.md.
```
