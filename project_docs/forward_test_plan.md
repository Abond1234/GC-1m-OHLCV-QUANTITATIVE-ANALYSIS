# Forward Test Plan — Shadow Mode Through Controlled Deployment

**Status:** SCAFFOLDING READY (2026-07-20). The shadow-mode harness (`src/execution/shadow_mode.py`) is built and tested; no forward test is running because no strategy is approved and no Rithmic access exists yet.

## Preconditions, stated honestly

1. **An approved strategy.** The Phase 2 research program closed with the S7P02 family archived: a real frictionless event edge with no surviving execution, sizing, exit, or suppression form at realistic costs. There is currently nothing to forward test. Any future candidate must arrive through a fresh pre-declared research contract and survive its own sequential, cost-aware backtest.
2. **Rithmic access (FR-11).** Requires credentials, an approved platform path, and infrastructure decisions the PRD assigns to the team (hosting, monitoring stack, execution language). Until then, the identical decision-and-telemetry code path runs against historical bars as dry runs.
3. **A selected prop firm and rule set** loaded into the FR-10 policy engine as versioned configuration, re-verified against the firm's current published terms.

## Stages and exit criteria (from the PRD, operationalized)

| Stage | Purpose | Exit criteria |
|---|---|---|
| Shadow mode | Run signals and risk checks on live data; transmit nothing | Zero stale-data violations (`stale_data_violation`, declared 120s bound); deterministic decisions (re-run reproduces the log byte-for-byte given the same inputs); complete telemetry (hash chain verifies via `verify_decision_log`); theoretical fills reconciled (`reconcile_theoretical_fills`) |
| Rithmic simulation | Simulated orders against live exchange data | Frozen strategy and config hash; zero unexplained position mismatches; zero hard-rule breaches in the FR-10 engine run alongside; stable operations; pre-declared sample reached |
| Prop evaluation (minimum size) | Exact approved configuration under the firm's terms | Pass or fail per the rules with no mid-evaluation changes; any material change restarts validation |
| Funded account (controlled scale) | Real operational and payout behaviour at minimum contract size | Sustained rule compliance; expected-versus-realized execution within declared tolerance; no unresolved safety events |
| Scale-up | More accounts / MGC size / later GC | Separate risk approval on live evidence |

## Pre-declared sample gate

Per the PRD's provisional default, frozen here before any campaign: **at least 60 trading days and at least 100 executed (simulated) trades** before any stage promotion, with longer testing required for low-frequency systems. This number may be revised only before a campaign starts, never during one.

## Telemetry requirements

Every decision is a `ShadowDecision` record: UTC decision timestamp, signal id, instrument, action (`enter_short` / `enter_long` / `no_trade` / `risk_block`), entry and stop prices for entries, source-bar timestamp and age, config fingerprint (`config_fingerprint` of the frozen configuration), and code version. Records append to a JSONL log chained by SHA-256 (`ShadowModeLogger`); the chain makes any edit, deletion, or reordering detectable, which turns the log into evidence rather than notes. Fill reconciliation compares each entry decision against the successor bar's range with the same entry-realism rule the research backtests used.

## MGC execution caveat carried forward from FR-09

The transfer validation found that the GC first-contact price prints on MGC in the same minute only ~80 percent of the time (stable across all years). Any forward-tested MGC configuration must therefore declare an MGC-native entry treatment (tolerance band or marketable entry) up front, and the Rithmic stages exist precisely to measure its real cost with decision-to-fill telemetry.

FORWARD TEST PLAN STATUS: READY
