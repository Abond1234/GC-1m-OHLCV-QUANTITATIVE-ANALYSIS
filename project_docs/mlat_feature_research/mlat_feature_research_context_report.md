# MLAT Feature Research — Context Report

Concise handoff for reproducing and extending the completed v1 experiment.

## Purpose

A separate, from-scratch experiment testing whether a bounded batch of book-derived/adapted features adds stable, economically meaningful Development-to-Validation information beyond the trusted one-minute GC feature matrix. It does not rewrite the historical statistical notebook.

## Upstream dependencies

| artifact_name | exact_path | row_count | column_count | research_partition_coverage | validation_status | mlat_use |
| --- | --- | --- | --- | --- | --- | --- |
| trusted_gc_mgc_bars | data/processed/research_bars_gc_mgc_1m.parquet | 3487656 | 82 | GC and MGC physical history | VALIDATED_WITH_COLUMN_WHITELIST | GC-only chronological OHLCV source for causal feature construction |
| eligible_gc_observations | data/processed/statistical_research/eligible_observations_gc.parquet | 586530 | 34 | Development 306230; Validation 118076; historical Final test 162224 | VALIDATED | Stable observation IDs and decision-bar mapping; Final-test metadata used only to enforce exclusion |
| forward_gc_labels | data/processed/statistical_research/forward_labels_gc.parquet | 586530 | 169 | Development; Validation; historical Final test | VALIDATED_DEV_VALIDATION_ONLY | Existing frozen 5/15/30/60/120/180-minute outcome grid; Final-test outcome columns are never loaded |
| existing_gc_feature_matrix | data/processed/statistical_research/feature_matrix_gc.parquet | 586530 | 97 | Development; Validation; historical Final test | VALIDATED | Development/Validation redundancy and incremental-information anchors; Final-test predictors excluded from outcome evaluation |
| existing_gc_feature_registry | data/processed/statistical_research/feature_registry_gc.parquet | 85 | 24 | Feature definitions are partition-independent | VALIDATED | Formula and semantic overlap audit against the 85 historical predictors |
| existing_feature_reference_parameters | data/processed/statistical_research/feature_reference_parameters_gc.parquet | 32 | 16 | Development fitted; applied unchanged later | VALIDATED | Audit Development-only fitting conventions; not refitted or overwritten |
| frozen_expansion_feature_set | data/processed/statistical_research/frozen_expansion_feature_set_gc.parquet | 30 | 7 | Development and Validation selection record | VALIDATED | ATR(20) and frozen 15-feature incremental-information anchors |
| historical_univariate_results | data/processed/statistical_research/univariate_results_gc.parquet | 3984 | 18 | Development and Validation only | VALIDATED | Context for historical 0 directional and 55 expansion advancers; not used to tune MLAT definitions |
| historical_baseline_summary | data/processed/statistical_research/baseline_summary_gc.parquet | 179184 | 19 | Development; Validation; historical Final test | VALIDATED_METADATA_ONLY | Development/Validation baseline context; historical Final-test metrics are not displayed or used for MLAT selection |
| forward_expansion_thresholds | data/processed/statistical_research/forward_label_expansion_thresholds_gc.parquet | 6 | 6 | Development fitted | VALIDATED | Apply the pre-existing Development-fitted expansion thresholds unchanged to Validation |

## Book coverage

Stage 1 is complete for all 858 physical pages, including 735 substantive preface/chapter/Appendix pages, 23 chapters, and the Appendix. Every chapter/Appendix has a page-traceable summary.

## Frozen hypothesis batch

12 of 29 preregistered hypotheses were selected before outcome evaluation. Batch SHA-256: `8c8b74267635566f07011c2789ad4d323647f19eba1f1f3426a154c1a850acc0`. Rejected/deferred hypotheses: 17.

## Architecture

The old notebook remains an immutable upstream producer. The new notebook loads hash-documented GC artifacts, builds registry-controlled causal features in reusable typed modules, evaluates only Development and Validation, persists versioned evidence, reload-verifies it, and records completion in an execution manifest.

- `src/statistical_research/mlat_artifacts.py`
- `src/statistical_research/mlat_feature_engineering.py`
- `src/statistical_research/mlat_feature_evaluation.py`
- `src/statistical_research/mlat_feature_registry.py`
- `src/statistical_research/mlat_feature_validation.py`
- `src/statistical_research/mlat_volatility_models.py`

## Implemented features

| hypothesis_id | feature_name | source_type | target_family | minimum_history | status |
| --- | --- | --- | --- | --- | --- |
| MLAT-H001 | bollinger_zscore_20 | MLAT-ADAPTED | direction | 20 | FROZEN_V1 |
| MLAT-H002 | bollinger_bandwidth_20 | MLAT-DIRECT | expansion | 20 | FROZEN_V1 |
| MLAT-H003 | cutler_rsi_14 | MLAT-ADAPTED | direction / risk state | 15 | FROZEN_V1 |
| MLAT-H004 | chaikin_money_flow_20 | MLAT-ADAPTED | direction / risk state | 20 | FROZEN_V1 |
| MLAT-H005 | amihud_illiquidity_60 | MLAT-ADAPTED | expansion / risk state | 61 | FROZEN_V1 |
| MLAT-H006 | parkinson_volatility_30 | PROJECT-ORIGINAL-EXTENSION | volatility / expansion | 30 | FROZEN_V1 |
| MLAT-H007 | rogers_satchell_volatility_30 | PROJECT-ORIGINAL-EXTENSION | volatility / expansion | 30 | FROZEN_V1 |
| MLAT-H008 | realized_semivariance_balance_60 | PROJECT-ORIGINAL-EXTENSION | risk state / direction | 61 | FROZEN_V1 |
| MLAT-H009 | bipower_jump_ratio_60 | PROJECT-ORIGINAL-EXTENSION | risk state / expansion | 62 | FROZEN_V1 |
| MLAT-H010 | variance_ratio_60_5 | MLAT-ADAPTED | direction / risk state | 65 | FROZEN_V1 |
| MLAT-H011 | return_sign_entropy_60 | PROJECT-ORIGINAL-EXTENSION | risk state / expansion | 61 | FROZEN_V1 |
| MLAT-H012 | volatility_of_volatility_60 | PROJECT-ORIGINAL-EXTENSION | risk state / expansion | 75 | FROZEN_V1 |

## Evaluation contract

Chronological Development and Validation; London/New York separation; New York dates as evidence units; daily Spearman IC; 2,000 date-block bootstrap iterations; Development-fitted quintiles applied unchanged to Validation; Benjamini-Hochberg control; year/session stability; horizon thinning; two-GC-tick directional economics; Development overlap veto; and 60/180-minute partial-IC tests beyond ATR(20) and the frozen existing set. Contract SHA-256: `31ea8ff8255285d580dc9c2cb71bcbff85d05011e771b49f2f09d24cad810320`. The frozen v1 contract was not relaxed after observing outcomes.

## Key results

The notebook completed 69 code cells with no error outputs and passed its engineering gate. 0 of 12 features received an advancement verdict. All otherwise valid, nonredundant candidates are retained as `RESEARCH_ONLY`: this is a structural authorization veto, not an empirical rejection of their descriptive evidence. The frozen v1 advancement gate is `False` because frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized The multivariate authorization gate is `False`.

| final_decision | feature_count |
| --- | --- |
| RESEARCH_ONLY | 12 |

## GARCH decision

`REJECTED_IMPLEMENTATION`. GARCH is audit-only and is not a v1 matrix feature. Critical failed checks: 3.

## Feature verdicts

| feature_name | descriptive_pre_authorization_verdict | final_decision | verdict_reason |
| --- | --- | --- | --- |
| bollinger_zscore_20 | ECONOMICALLY_TRIVIAL | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| bollinger_bandwidth_20 | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| cutler_rsi_14 | ECONOMICALLY_TRIVIAL | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| chaikin_money_flow_20 | ECONOMICALLY_TRIVIAL | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| amihud_illiquidity_60 | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| parkinson_volatility_30 | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| rogers_satchell_volatility_30 | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| realized_semivariance_balance_60 | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| bipower_jump_ratio_60 | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| variance_ratio_60_5 | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| return_sign_entropy_60 | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| volatility_of_volatility_60 | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |

## Artifact paths

- `data/processed/statistical_research/mlat_feature_research/v1`
- `reports/statistical_research/mlat_feature_research/v1`
- `reports/figures/mlat_feature_research/v1`
- `notebooks/exploration/mlat_feature_research.ipynb`

## Unresolved issues

- The frozen v1 advancement gate is structurally non-evaluable and remains fail-closed: frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized
- Outcomes have been viewed, so a feasible thinning replacement must be preregistered as v2 and evaluated on new data or a new holdout.
- The prior Final-test period is exposed and remains unavailable as a pristine holdout.
- MGC transfer validation was not authorized or inspected.
- No trading policy, transaction-cost model, PnL, or Sharpe result exists.
- GARCH critical check `fit_residual_diagnostics` failed: segment-aware Ljung-Box on residuals/squares and ARCH LM
- GARCH critical check `development_audit_residual_diagnostics` failed: segment-aware Ljung-Box on residuals/squares and ARCH LM
- GARCH critical check `validation_residual_diagnostics` failed: segment-aware Ljung-Box on residuals/squares and ARCH LM

## Next step

Preregister an MLAT v2 contract with a feasible non-overlap sensitivity statistic before examining new outcomes; retain v1 as fail-closed RESEARCH_ONLY evidence and require a new future holdout or live paper period.
