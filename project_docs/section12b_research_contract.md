# Section 12B Research Contract: Opportunity-Conditioned Sizing and Exits

**Status:** FROZEN BEFORE IMPLEMENTATION (declared 2026-07-20, per the PRD workflow rule that the research question, definitions, expected artifacts, and acceptance checks precede implementation)

## Research question

Section 12 established that the frozen Branch B opportunity model fails as an entry filter for POI events. This contract tests what the model actually forecasts - movement magnitude - in the roles magnitude information can legitimately play: position sizing, target and holding-time selection, and no-trade suppression, applied to the True POI continuation-short family.

## Hypotheses (pre-declared, each tested independently)

- H1 Sizing: scaling risk per trade inversely/directly with predicted opportunity (three declared schemes: constant 1R baseline; proportional to prediction quintile; inverse to prediction quintile) changes the family's risk-adjusted profile (mean/MAD ratio, drawdown of cumulative R) without materially changing mean R.
- H2 Exits: predicted-opportunity quintile conditions the optimal target/holding pairing from the already-computed Section 7 stop/target grid (no new grid search; only the existing frozen grid rows are reused).
- H3 Suppression: removing events in the bottom Development-fitted opportunity quintile improves the family's median R and -1R rate without reducing date coverage below floors.

## Frozen definitions

- Population: True POI continuation-short outcome rows joined to the full-coverage frozen gate table exactly as in Section 12 (98.1 percent coverage, decision-bar identity join, reproduction check enforced).
- Prediction quintiles: fitted on Development POI events only, applied unchanged elsewhere.
- Outcome basis: existing Section 7 labels only (capped 60m/120m/240m R, MFE/MAE); no new label construction.
- Partitions: criteria evaluated on Development and Validation only; one Final-test read after verdicts are fixed, reported separately.

## Acceptance criteria (all pre-declared)

- A hypothesis advances only if its Development effect passes a date-block bootstrap interval excluding zero, Validation agrees in sign with at least 25 percent retention, and the affected subsets satisfy floors of 1,000 Development and 700 Validation events across at least 200 and 120 trading dates respectively (floors sized to the whole family, not the gated slice, to avoid the Section 12 starvation).
- H1 additionally requires the risk-adjusted improvement to survive both cost scenarios of `Section11Config`.
- Any post-hoc metric, threshold, or scheme change after seeing results voids the run.

## Expected artifacts

`src/research/opportunity_conditioning.py`, `tests/test_opportunity_conditioning.py`, notebook Section 12B cells via the standard updater pattern, tracked summary `reports/statistical_research/summaries/section12b_opportunity_conditioning_summary.md`, generated parquet outputs excluded from Git.

## Relationship to Section 8 authorization

Results feed the decision recorded in `project_docs/section8_authorization_memo.md`: if any hypothesis advances, the Section 8 candidate becomes the conditioned family; if none advance, the memo's Option 1 proceeds (or Option 3 closes the family) with this contract's negative results attached.
