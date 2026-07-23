# Quant Project 1: PRD vs. Codebase Comparison Report

This report presents a thorough analysis of **Quant Project 1 PRD v1.0** and compares its specifications with the current level of implementation in the repository.

---

## 1. Executive Summary

Quant Project 1 is a private systematic trading research and engineering program for CME Gold Futures (**GC**) and Micro Gold Futures (**MGC**). The repository has evolved from an exploratory phase into a structured, highly validated research environment. 

The current codebase features:
* A **highly robust data foundation** utilizing Databento Parquet tables.
* A formalized **Branch A (True POI)** research pipeline and a decoupled **Branch B (Independent Statistical)** pipeline.
* A comprehensive, cost-aware **Sequential Backtest engine** for both branches.
* Core components of **Phase 4 (Execution Layer)** infrastructure: GC-to-MGC transfer validation and a versioned prop-firm rules engine.
* **Excellent test coverage** with **224 automated tests** passing successfully.

### The Critical Transition
Research on both **Branch A** and **Branch B** has been completed and **closed with a systematic rejection**. The only candidate policy that survived screening (`S7P02_NY_BEAR_CONT`) failed to show positive expectancy when factoring in realistic execution costs, and conditioning methods failed to survive Validation. 

Consequently, the project status is now focused on **engineering and execution infrastructure**. The codebase contains completed scaffolding for forward testing, but no trading strategy is currently approved to run.

---

## 2. Functional Requirements (FR) Gap Analysis

The PRD defines 17 Functional Requirements (**FR-01** to **FR-17**). The table below maps each requirement to its corresponding implementation in the codebase and highlights any existing gaps.

| ID | PRD Functional Requirement | Implemented Files | Status / Findings |
|---|---|---|---|
| **FR-01** | **Trusted Data Loading** | [resources.py](../src/resources.py), [feature_validation.py](../src/statistical_research/feature_validation.py) | **Complete**. Schema, timestamps, row counts, and dtypes are checked. Loads project-root-relative Parquet tables. |
| **FR-02** | **Active-Contract & Rollover Integrity** | [resources.py](../src/resources.py), [labels.py](../src/statistical_research/labels.py) | **Complete**. Rollover and contract transition schedules are liquidity-selected and preserved; paths do not cross invalid boundaries. |
| **FR-03** | **Causal Multi-Timeframe Features** | [feature_engineering.py](../src/statistical_research/feature_engineering.py), [poi_context_features.py](../src/features/poi_context_features.py) | **Complete**. Features are strictly causal (completing at bar $t$, entry at $t+1$) with no lookahead or centered-window leakage. |
| **FR-04** | **POI Research Branch** | `src/features/` | **Complete**. Formulations, retests, context features, controls, and first-passage studies are frozen. |
| **FR-05** | **Independent Statistical Branch** | `src/statistical_research/` | **Complete**. 85 predictors registered and evaluated without POI information. |
| **FR-06** | **Signal Specification** | [signal_construction.py](../src/statistical_research/signal_construction.py) | **Complete**. Represents candidate strategies as versioned machine-readable configurations. |
| **FR-07** | **Sequential Backtest** | [sequential_backtest.py](../src/statistical_research/sequential_backtest.py), [poi_sequential_backtest.py](../src/research/poi_sequential_backtest.py) | **Complete**. Simulates chronological account state, order fills, position tracking, daily limits, and forced exit at 15:30 NY time. |
| **FR-08** | **Execution-Cost Model** | [sequential_backtest.py](../src/statistical_research/sequential_backtest.py) | **Complete**. Factures in commissions, exchange fees, bid/ask spread, and slippage under base and pessimistic cases. |
| **FR-09** | **GC-to-MGC Transfer Validation** | [mgc_transfer.py](../src/execution/mgc_transfer.py) | **Complete (Provisional Fail)**. Verified that the GC first-contact price trades on MGC within the same minute only ~80.2% of the time, meaning direct tick-precise limit transfer is unsafe. |
| **FR-10** | **Prop-Firm Rules Engine** | [prop_firm_rules.py](../src/execution/prop_firm_rules.py) | **Complete**. Policies are treated as configurable configuration layers (profit target, daily loss, EOD/intraday trailing drawdowns, consistency). Estimates pass/breach probabilities via bootstrap. |
| **FR-11** | **Rithmic Paper Integration** | *None* | **Not Started**. Requires broker credentials, Rithmic API integration, live-data gateway, and host environment setup. |
| **FR-12** | **Independent Risk Engine** | [shadow_mode.py](../src/execution/shadow_mode.py), [prop_firm_rules.py](../src/execution/prop_firm_rules.py) | **Partially Complete**. Stale-data guards and internal daily loss/drawdown buffers are coded. Real-time API-level disconnects and emergency stop handlers require live broker integration. |
| **FR-13** | **Performance Analytics** | [sequential_backtest.py](../src/statistical_research/sequential_backtest.py), [prop_firm_rules.py](../src/execution/prop_firm_rules.py) | **Complete**. Reports path-level metrics, drawdowns, equity curves, session mix, and bootstrap/Monte Carlo diagnostics. |
| **FR-14** | **Experiment Registry** | *None* | **Partially Complete**. Managed informally via Markdown summaries and notebooks, but lacks a software API/DB registry. |
| **FR-15** | **Reproducible Run Interface** | `scripts/` | **Complete**. Automation scripts run and regenerate research outputs. Lacks a unified "master script" or entry command. |
| **FR-16** | **Monitoring Dashboard** | *None* | **Not Started**. No graphical monitoring UI exists (requires FR-11 live adapter). |
| **FR-17** | **TBBO/MBP/MBO Expansion** | *None* | **Non-Goal / Future Phase**. Deferred until storage, compute, data, and research budgets are approved. |

---

## 3. Key Scaffolding & Subsystem Analysis

### A. Prop-Firm Constraint Engine (`prop_firm_rules.py`)
This module satisfies **FR-10** by decoupling prop firm rules from core strategy code.
* **Versioned Policies:** Allows configuration of account size, target profit, daily loss limits, and drawdown types (e.g. `static`, `trailing_intraday`, `trailing_eod`).
* **Evaluation Simulator (`simulate_evaluation`):** Grouping trades by `trade_date` to check daily limits, watermarks, drawdown floors, consistency ratios, and minimum/maximum trading days.
* **Bootstrap Estimator (`estimate_evaluation_statistics`):** Draws daily P&L values with replacement over a configured horizon to estimate overall pass rates, drawdown breach probabilities, and average days to pass.

### B. MGC Transfer Validation (`mgc_transfer.py`)
This module satisfies **FR-09** and serves as the gate-check for **G5 (MGC Transfer)**.
* **The Core Gap:** The synchronization of minutes between GC and MGC is high (97.85%), but **GC entry price containment inside the MGC bar is only 80.2%** (against the PRD threshold of >=95%). 
* **Implication:** The system cannot assume that limit orders placed at GC-derived prices will execute on MGC. This provisional fail requires an MGC-native entry treatment (e.g. market order or limit price offset/tolerance band) for live execution.

### C. Shadow-Mode Forward Testing (`shadow_mode.py`)
Provides the scaffolding for **Phase 5 (Shadow Mode)**.
* **Tamper-Evident Logs:** Uses a SHA-256 hash chain on `ShadowDecision` logs (each record embeds the previous record's hash) to guarantee log integrity.
* **Stale-Data Guard:** Blocks entry if data age exceeds 120 seconds.
* **Reconciliation:** Reconciles theoretical entry fills against successor bar high/low ranges.

---

## 4. Checklist: Open Decisions from PRD Section 22

The PRD lists several open design choices that require technical/operational resolution before Phase 5 onwards:

1. **Role Allocation:** Partner roles and backups for Research, Platform, and Execution need assignment.
2. **First Target Prop Firm:** Selecting the specific firm (e.g., Apex, FTMO, Earn2Trade) to verify current terms and Automations Policies.
3. **Execution Language/Host:** Decisions on hosting environment (cloud/on-prem), execution language (Python vs C++ or others), and Rithmic API integration path.
4. **Subsystem Strategy:** Acknowledge S7P02's rejection and determine if new research contracts should target a POI-only, statistical-only, or hybrid design.
5. **Position Sizing Rules:** Defining risk units and stop thresholds.
6. **Holdout Generation:** Designing a method to acquire untouched holdout data (e.g., collecting new live paper-trading records) since the original Final Test partition has been descriptively exposed.

---

## 5. Non-Functional Requirements & Code Health

* **Correctness & Tests:** The codebase is extremely clean, passing all **224 tests** successfully in ~16.6 seconds.
* **Performance:** Relies heavily on vectorized PyArrow and Pandas operations; memory scaling is managed dynamically via `MemoryPlan` in `src/resources.py`, preventing out-of-memory errors on lower-tier machines.
* **Modularity:** Highly modular. Feature generation, labels, rules engine, and backtester are cleanly separated and individually tested.
* **Security:** Secrets and data files are excluded via `.gitignore`.
