<p align="center">
  <img src="./assets/readme/hero.svg" width="100%"
       alt="Quant Project 1 — CME Gold Futures Research and Strategy Pipeline">
</p>

<p align="center">
  <a href="#-research-branches">Research Branches</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-testing">Testing</a> ·
  <a href="#-research-standards">Research Standards</a>
</p>

---

## What This Is

**Quant Project 1** is a professional-grade quantitative research pipeline for **CME Gold Futures (GC) and Micro Gold Futures (MGC)**.

It is not a trading bot. It is a rigorous, multi-stage research codebase designed to:

1. Detect and refine **Points of Interest (POIs)** from intraday swing-breaking displacement moves.
2. Build a **candidate-trade database** with quantitative confluence features for each POI retest.
3. Develop a **fully independent statistical signal system** from raw OHLCV data — without any POI information — to find relationships that survive out-of-sample scrutiny.
4. Lay the groundwork for an **integration phase** where both branches are merged into a complete, backtestable systematic strategy.

The primary research timeframe is **1-minute OHLCV bars** over a dataset of **3,487,656 rows** (1,759,671 GC + 1,727,985 MGC), sourced via [Databento](https://databento.com).

---

## 🔬 Research Branches

<p align="center">
  <img src="./assets/readme/pipeline.svg" width="100%"
       alt="Dual-branch architecture: POI strategy pipeline and independent statistical research pipeline converging at integration">
</p>

### Branch A — POI-Based Strategy

The POI branch converts a discretionary intraday gold strategy into a systematic, rule-based research process. A valid **Point of Interest (POI)** requires:

- A classic **wick-to-wick ICT Fair Value Gap (FVG)** across three consecutive candles.
- A **close-to-next-open gap** in the same displacement direction.

When both conditions hold, the **full high-to-low range of the middle candle** becomes the tradable zone. POIs are valid only on the trading day they are created (reset at midnight New York time).

**Execution windows:**

| Window | New York Time |
|---|---|
| London | 03:00 – 06:00 |
| New York | 07:00 – 12:00 |

**POI creation window:** 01:00 – 12:00 NY. Same-day POIs created at 02:00 are valid for the 07:00 New York window.

The research pipeline for this branch runs in five stages:

```
1. POI detection (swing break + FVG scan)          →  poi_signal_features.py
2. POI selection & geometry refinement             →  poi_selection_refinement.py
3. Retest identification & event study             →  poi_event_study.py
4. First-passage & excursion analysis              →  poi_first_passage.py
5. Decision-time context feature engineering       →  poi_context_features.py
```

Trade types under research: **continuation**, **reversal**, and **pullback** retests — treated as separate classifications, not pooled.

---

### Branch B — Independent Statistical Research

This branch deliberately ignores all POI information. Its mandate:

> Identify stable and economically meaningful relationships between information available at time *t* and future intraday GC price behaviour, without using POI information.

The research pipeline for this branch:

```
1. Forward label construction                      →  labels.py
2. Feature specification & registry               →  feature_registry.py
3. Lookahead-free feature matrix engineering      →  feature_engineering.py
4. Feature validation & diagnostics               →  feature_validation.py
5. Random-entry bootstrap baselines               →  baselines.py
```

**Forward label targets investigated:**

- Signed 1-minute direction
- Future movement magnitude (5m, 15m, 30m, 60m horizons)
- Maximum Favourable Excursion (MFE)
- Maximum Adverse Excursion (MAE)
- Volatility expansion
- Continuation vs. reversal classification

Only after independent features and rules are frozen will they be compared or combined with Branch A results.

---

## 🏗 Architecture

```
project-1/
│
├── src/
│   ├── features/                     # Branch A — POI strategy modules
│   │   ├── poi_signal_features.py    # FVG detection, swing-break detection
│   │   ├── poi_selection_refinement.py  # Section 6c geometry scoring
│   │   ├── poi_event_study.py        # Candidate-trade event study
│   │   ├── poi_event_study_refined.py
│   │   ├── poi_first_passage.py      # MFE / MAE first-passage analysis
│   │   ├── poi_context_features.py   # Decision-time feature engineering
│   │   ├── poi_selection_refinement.py
│   │   └── poi_refinement_visuals.py
│   │
│   ├── statistical_research/         # Branch B — independent statistical pipeline
│   │   ├── labels.py                 # Forward label construction (no leakage)
│   │   ├── feature_registry.py       # Feature spec catalogue
│   │   ├── feature_engineering.py    # Lookahead-free feature matrix
│   │   ├── feature_validation.py     # Diagnostics & audit
│   │   └── baselines.py              # Random-entry bootstrap baselines
│   │
│   └── research/
│       └── poi_context_event_study.py  # Cross-branch context study
│
├── notebooks/exploration/
│   ├── exp1.ipynb                    # Main POI research notebook
│   ├── exp1 appendix.ipynb
│   └── statistical_feature_research.ipynb  # Branch B notebook
│
├── scripts/
│   ├── run_section6c_refinement.py   # POI refinement runner
│   ├── run_section7_event_study.py   # Event study runner
│   ├── run_section7_poi_context_research.py
│   └── update_statistical_section*.py  # Notebook update scripts
│
├── tests/                            # Full unit test suite (9 modules)
│   ├── test_poi_signal_features.py
│   ├── test_poi_selection_refinement.py
│   ├── test_poi_event_study.py
│   ├── test_poi_context_features.py
│   ├── test_poi_first_passage.py
│   ├── test_poi_context_event_study.py
│   ├── test_statistical_research_baselines.py
│   ├── test_statistical_research_features.py
│   └── test_statistical_research_labels.py
│
└── project_docs/                     # Strategy specs and context reports
    ├── gc_mgc_poi_strategy_spec_v0_2*.txt
    ├── Quant Project 1 — Context Report*.md
    ├── statistical_feature_research_context_report.md
    └── Section 4 Follow-up Report*.md
```

---

## ⚡ Quick Start

### Prerequisites

```
Python 3.11+
numpy · pandas · scipy · statsmodels
matplotlib · seaborn · pyarrow
databento · jupyter
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Abond1234/project-1.git
cd project-1

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the test suite

```bash
pytest
```

All 9 test modules cover POI formation geometry, retest identification, first-passage analysis, context feature availability, label leakage checks, feature matrix validation, and bootstrap baselines.

### Open the research notebooks

```bash
jupyter notebook notebooks/exploration/exp1.ipynb
jupyter notebook notebooks/exploration/statistical_feature_research.ipynb
```

> **Note:** The notebooks require the processed dataset at `data/processed/research_bars_gc_mgc_1m.parquet`. This file is not committed to the repository. It is produced by the data cleaning and active-contract construction pipeline described in Section 4 of the context report.

---

## ✅ Testing

The test suite rigorously verifies the research pipeline's correctness:

| Module | What it covers |
|---|---|
| `test_poi_signal_features.py` | FVG geometry (wick/close modes), swing detection, variant correctness |
| `test_poi_selection_refinement.py` | Section 6c geometry scoring, validation, final POI selection |
| `test_poi_event_study.py` | Retest identification, candidate table construction, structural annotation |
| `test_poi_context_features.py` | Decision-time feature availability gating, zero-denominator safety, deduplication |
| `test_poi_first_passage.py` | MFE/MAE first-passage analysis, excursion tracking |
| `test_poi_context_event_study.py` | Cross-branch context study integrity |
| `test_statistical_research_labels.py` | Forward label lookahead checks, tick-grid validation |
| `test_statistical_research_features.py` | Feature matrix construction, continuity run IDs, NaN handling |
| `test_statistical_research_baselines.py` | Bootstrap random-entry baseline reproducibility |

---

## 📐 Research Standards

This codebase is built to **professional quantitative research standards**. Every module is held to these requirements:

### Correctness first

Priority order: **Correctness → Statistical validity → Computational efficiency → Memory efficiency → Scalability → Readability**.

Working code alone is insufficient if a materially better implementation exists.

### No lookahead leakage

All features are validated against strict availability timestamps. The feature engineering pipeline tracks formation, pre-touch, and touch-close availability separately. Features that require information not yet available at entry time are flagged and gated by the `validate_entry_feature_availability` system.

### Independence boundary

Branch B must not inherit directional bias from Branch A. All POI-derived tables, identifiers, geometry fields, and rankings are explicitly excluded from the statistical research pipeline. POI information may only enter after the independent statistical system is completely frozen.

### Performance for scale

Designed for datasets with **5–20 million rows**:
- Vectorized NumPy/Pandas operations throughout.
- No row-wise Python loops over bars.
- Reusable computed arrays; no repeated full-table scans.
- Memory-efficient intermediate representations.

### Reproducibility

All random operations use fixed seeds (`BASELINE_RANDOM_SEED`). Parquet outputs are deterministic. Continuous segment IDs track data continuity across contract rolls.

---

## 📊 Instruments

| Property | Value |
|---|---|
| Signal instrument | GC — CME Gold Futures (front-month active contract) |
| Execution instrument | MGC — CME Micro Gold Futures (front-month active contract) |
| Active contract selection | Liquidity-based rolling rule (5-day volume window) |
| Primary research timeframe | 1-minute OHLCV bars |
| Data source | Databento |
| Dataset size | 3,487,656 bars (GC: 1,759,671 · MGC: 1,727,985) |
| Research period | Multi-year intraday history |

---

## 📁 Data

The project uses a processed dataset, not raw exchange data:

```
data/processed/research_bars_gc_mgc_1m.parquet
```

This dataset is produced by the Section 4 data pipeline, which performs:

- Multi-contract GC/MGC bar ingestion from Databento
- Liquidity-based active contract selection (rolling 5-day volume rule)
- Contract roll continuity tracking
- New York session timestamp fields
- Tradability and roll-window flags
- Core derived bar fields (ATR, realized vol, relative volume, log returns)

> The raw data and processed parquet files are **not included** in this repository.

---

## 📄 License

This is a private research repository. All rights reserved.
