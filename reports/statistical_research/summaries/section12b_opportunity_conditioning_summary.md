# Section 12B Opportunity Conditioning Summary

**Status:** READY — **no hypothesis advances; the S7P02 family is archived per the memo linkage (Option 3)**

## Scope and governance

Executed exactly per the frozen contract in `project_docs/section12b_research_contract.md` (declared before implementation): can the frozen Branch B opportunity model add value to the True POI continuation-short family in the roles magnitude information can legitimately play - position sizing (H1), exit-horizon conditioning (H2), and no-trade suppression (H3)? The population is the Section 12 decision-bar join (85,336 events, 98.1 percent gate coverage, saved-prediction reproduction check enforced; 7,066 Development / 5,225 Validation / 73,045 Final test). Prediction quintiles were fitted on Development events only. Advancement required a Development date-block bootstrap interval excluding zero, Validation sign agreement with at least 25 percent retention, and floors of 1,000/700 events across 200/120 dates on the affected subsets; H1 additionally had to survive both `Section11Config` cost scenarios (base 2.6 / pessimistic 4.6 ticks round trip against each event's own risk) and a 0.05R mean-shift materiality bound. The Final test was read once after all verdicts were fixed.

## Results

| Hypothesis | Development | Validation | Verdict |
|---|---|---|---|
| H1 proportional sizing (base) | ratio +0.032, CI [-0.013, +0.078] | +0.030 (retention 0.93) | NO_ADVANCE |
| H1 proportional sizing (pessimistic) | ratio +0.038, CI [+0.000, +0.082] | +0.031 (retention 0.81) | NO_ADVANCE |
| H1 inverse sizing (both scenarios) | negative | negative | NO_ADVANCE |
| H2 exit conditioning vs 120m | +0.023R, CI [-0.110, +0.155] | -0.045R (retention -2.0) | NO_ADVANCE |
| H3 bottom-quintile suppression | median +0.061R, stop rate -0.59pp | median -0.003R, stop rate +0.29pp | NO_ADVANCE |

- **H1 is the instructive near-miss.** Proportional sizing improves the mean/MAD ratio in every scenario with 0.81-1.07 Validation retention - the most persistent positive signal the opportunity model has produced anywhere. It fails the contract twice over: the Development interval includes zero at base costs, and the scheme shifts mean R by +0.04 to +0.08R in both partitions, breaching the materiality bound. The lift comes from concentrating exposure in high-prediction quintiles - the same return effect Section 12 already rejected on floors and the final test, so the bound did its job: a "sizing" scheme that moves mean R is a return bet in disguise.
- **H2**: Development prefers 120m unconditionally; the Development-fitted quintile mapping (60m for the bottom quintile, 240m for quintiles 1-2) gains +0.023R in Development and flips to -0.045R in Validation.
- **H3**: both co-primary effects improve in Development and both flip sign in Validation.
- **One-time Final-test read** (after verdicts fixed, feeding no criterion): every effect within 0.03R of zero.

## Family closure

Per `project_docs/section8_authorization_memo.md`: Option 1 already returned SEQUENTIAL_REJECTED at base costs (Section 8 backtest), and this contract's conditioning forms are all negative. Option 3 therefore applies - **the S7P02 continuation-short family is archived** with the complete evidence chain: a real frictionless event-level edge (+0.138/+0.117/+0.063 across partitions in sequential form) that no tested execution, sizing, exit, or suppression form carries past realistic costs.

## Verification

9/9 structural checks (Development-only quintile edges reproduced, robust cap on finite labels, weight ladders averaging one on Development, H2 menu within the frozen grid horizons, final-test report isolated from verdicts, one verdict per hypothesis); 10 new synthetic tests with planted effects proving each pass/veto path (proportional advances and inverse fails on a constructed dispersion split, cost arithmetic exact, materiality veto, H2 mapping and tie-to-shortest rule, H3 co-primary veto, bootstrap seed determinism); full suite 198 tests green. Analysis runtime about 6 seconds.

## Saved outputs (excluded from Git)

`section12b_h1_sizing_results_gc.parquet`, `section12b_h1_sizing_effects_gc.parquet`, `section12b_h2_exit_mapping_gc.parquet`, `section12b_h2_exit_results_gc.parquet`, `section12b_h3_suppression_results_gc.parquet`, `section12b_verdicts_gc.parquet`, `section12b_final_test_report_gc.parquet`; tracked CSVs under `reports/statistical_research/tables/section12b/`.

SECTION 12B STATUS: READY
