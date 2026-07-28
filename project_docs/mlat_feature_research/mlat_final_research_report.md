# MLAT Feature Research — Final Research Report

Generated strictly from the frozen Stage-1 record and the executed v1 evidence artifacts. Notebook evidence: 69 executed code cells, 0 error outputs.

## 1. Executive conclusion

The executed v1 notebook passed its engineering, evidence-integrity, Development/Validation-only, artifact-persistence, and governance checks. However, the frozen advancement design is structurally non-evaluable, so v1 fails closed: 0 of 12 features were authorized to advance. This is not a clean empirical rejection of the raw relationships. The authorization blocker is: frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized The multivariate authorization gate was CLOSED. The separately governed GARCH audit verdict was `REJECTED_IMPLEMENTATION`. These are feature-research findings, not a signal, PnL, or Sharpe claim.

| final_decision | feature_count |
| --- | --- |
| RESEARCH_ONLY | 12 |

## 2. Book-ingestion completion status

Stage 1 is `COMPLETE`. The persisted completion report, ingestion manifest, 24 chapter/Appendix summaries, concept registry, formula registry, hypothesis catalog, frozen batch, and frozen contract all passed the pre-outcome build controls.

- Frozen batch SHA-256: `8c8b74267635566f07011c2789ad4d323647f19eba1f1f3426a154c1a850acc0`
- Frozen contract SHA-256: `31ea8ff8255285d580dc9c2cb71bcbff85d05011e771b49f2f09d24cad810320`
- Concepts recorded: 30
- Formula/definition entries: 14
- Hypotheses recorded: 29

## 3. Page coverage

The coverage manifest accounts exactly once for all 858 physical PDF pages. Substantive preface/chapter/Appendix coverage comprises 735 pages, and 102 distinct pages containing important equations, tables, figures, or diagrams were separately rendered and inspected.

| section_type | pdf_page_start | pdf_page_end | page_count | records | complete_records |
| --- | --- | --- | --- | --- | --- |
| appendix | 735 | 764 | 30 | 1 | 1 |
| chapter | 42 | 734 | 693 | 23 | 23 |
| contents | 9 | 29 | 21 | 1 | 1 |
| front_matter | 1 | 8 | 8 | 1 | 1 |
| index | 780 | 858 | 79 | 1 | 1 |
| preface | 30 | 41 | 12 | 1 | 1 |
| references | 765 | 779 | 15 | 1 | 1 |

## 4. Chapter coverage

All 23 chapters and the Appendix have complete durable summaries with page maps and source citations.

| section_type | title | pdf_pages | status |
| --- | --- | --- | --- |
| chapter | Chapter 1 - Machine Learning for Trading - From Idea to Execution | 42-58 | COMPLETE |
| chapter | Chapter 2 - Market and Fundamental Data - Sources and Techniques | 59-94 | COMPLETE |
| chapter | Chapter 3 - Alternative Data for Finance - Categories and Use Cases | 95-114 | COMPLETE |
| chapter | Chapter 4 - Financial Feature Engineering - How to Research Alpha Factors | 115-152 | COMPLETE |
| chapter | Chapter 5 - Portfolio Optimization and Performance Evaluation | 153-178 | COMPLETE |
| chapter | Chapter 6 - The Machine Learning Process | 179-202 | COMPLETE |
| chapter | Chapter 7 - Linear Models - From Risk Factors to Return Forecasts | 203-248 | COMPLETE |
| chapter | Chapter 8 - The ML4T Workflow - From Model to Strategy Backtesting | 249-279 | COMPLETE |
| chapter | Chapter 9 - Time-Series Models for Volatility Forecasts and Statistical Arbitrage | 280-319 | COMPLETE |
| chapter | Chapter 10 - Bayesian ML - Dynamic Sharpe Ratios and Pairs Trading | 320-349 | COMPLETE |
| chapter | Chapter 11 - Random Forests - A Long-Short Strategy for Japanese Stocks | 350-386 | COMPLETE |
| chapter | Chapter 12 - Boosting Your Trading Strategy | 387-428 | COMPLETE |
| chapter | Chapter 13 - Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning | 429-460 | COMPLETE |
| chapter | Chapter 14 - Text Data for Trading - Sentiment Analysis | 461-485 | COMPLETE |
| chapter | Chapter 15 - Topic Modeling - Summarizing Financial News | 486-505 | COMPLETE |
| chapter | Chapter 16 - Word Embeddings for Earnings Calls and SEC Filings | 506-532 | COMPLETE |
| chapter | Chapter 17 - Deep Learning for Trading | 533-568 | COMPLETE |
| chapter | Chapter 18 - CNNs for Financial Time Series and Satellite Images | 569-606 | COMPLETE |
| chapter | Chapter 19 - RNNs for Multivariate Time Series and Sentiment Analysis | 607-638 | COMPLETE |
| chapter | Chapter 20 - Autoencoders for Conditional Risk Factors and Asset Pricing | 639-662 | COMPLETE |
| chapter | Chapter 21 - Generative Adversarial Networks for Synthetic Time-Series Data | 663-690 | COMPLETE |
| chapter | Chapter 22 - Deep Reinforcement Learning - Building a Trading Agent | 691-723 | COMPLETE |
| chapter | Chapter 23 - Conclusions and Next Steps | 724-734 | COMPLETE |
| appendix | Appendix - Alpha Factor Library | 735-764 | COMPLETE |

## 5. Most relevant book concepts

| concept_id | concept_name | chapter | pdf_page | category | applicability | recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| MLAT-C001 | Research-to-execution workflow | 1 | 42-58 | model validation | DIRECT | ADOPT |
| MLAT-C002 | Point-in-time market data | 2 | 59-94 | backtesting | DIRECT | ADOPT |
| MLAT-C004 | Lagged-return features | 4 | 130-132 | direction | DIRECT | REJECT_REDUNDANT |
| MLAT-C005 | Bollinger standardized location and bandwidth | 4 / Appendix | 131-133; 740-742 | direction / volatility | ADAPTED | IMPLEMENT_AND_TEST_OVERLAP |
| MLAT-C006 | Relative strength index | 4 | 132 | direction | ADAPTED | IMPLEMENT_CUTLER_VARIANT |
| MLAT-C007 | Kalman filtering | 4 | 133-136 | risk state | ADAPTED | DEFER |
| MLAT-C009 | Information coefficient and factor quantiles | 4 | 141-150 | model validation | DIRECT | ADOPT |
| MLAT-C012 | Bias-variance trade-off | 6 | 192-195 | model validation | DIRECT | ADOPT |
| MLAT-C013 | Purging and embargo | 6 | 199-200 | model validation | DIRECT | ADOPT_WHEN_MODELLING |
| MLAT-C016 | Stationarity and differencing | 9 | 280-290 | model validation | DIRECT | ADOPT |
| MLAT-C017 | AR and variance-ratio state | 9 | 290-296 | direction | ADAPTED | IMPLEMENT_VARIANCE_RATIO_AND_TEST_OVERLAP |
| MLAT-C018 | ARCH/GARCH conditional variance | 9 | 297-301 | volatility | ADAPTED | AUDIT_SEPARATELY |

## 6. Methods rejected as inapplicable

The applicability screen rejected, deferred, or moved methods to a different-data phase when they violated causality/governance, duplicated existing features, or required unavailable cross-sectional, text, quote, trade, order-book, macro, or image data.

| Idea | Source (chapter:PDF) | Classification | Reason |
| --- | --- | --- | --- |
| Lagged returns | 4:130-132 | REDUNDANT WITH EXISTING FEATURES | Already represented at frozen horizons. |
| Wavelet denoising | 4:137-140 | REJECTED DUE TO LEAKAGE | Book demonstration uses full-window decomposition/reconstruction. |
| Cointegration/pairs | 9-10:301-349 | REQUIRES DIFFERENT DATA | Single GC series cannot supply a pair. |
| Sentiment/topics/embeddings | 14-16:461-532 | REQUIRES DIFFERENT DATA | Unavailable in OHLCV. |
| GAN synthetic data | 21:663-690 | REJECTED DUE TO GOVERNANCE | Could manufacture or erase rare risk behavior. |
| Reinforcement learning | 22:691-723 | REJECTED DUE TO GOVERNANCE | No signal or simulator authorization. |
| ATR/NATR | App:754-755 | REDUNDANT WITH EXISTING FEATURES | Use frozen anchor rather than add another ATR. |
| Quotes/trades/order book/MBO | 2:59-94 | REQUIRES DIFFERENT DATA | Not present in one-minute OHLCV. |
| Macro/fundamental factors | 2-4:59-152 | REQUIRES DIFFERENT DATA | Unavailable and often equity-specific. |
| Satellite/imagery | 3/18:95-114;569-606 | IMPRACTICAL WITH CURRENT OHLCV | No relevant image source. |

## 7. Existing-feature overlap findings

The empirical Development overlap audit evaluated 12 candidates against 85 existing references. It found 0 exact or absolute-correlation-at-least-0.995 veto relationships. Final redundancy decisions are carried in the verdict table.

| candidate_feature | reference_feature | absolute_correlation | is_exact_duplicate |
| --- | --- | --- | --- |
| parkinson_volatility_30 | atr_20 | 0.972692 | NO |
| bollinger_zscore_20 | distance_from_rolling_vwap_20_atr | 0.963953 | NO |
| rogers_satchell_volatility_30 | atr_20 | 0.961703 | NO |
| cutler_rsi_14 | return_15m_atr | 0.948554 | NO |
| realized_semivariance_balance_60 | return_60m_atr | 0.87069 | NO |
| bollinger_bandwidth_20 | realized_volatility_15 | 0.808396 | NO |
| return_sign_entropy_60 | atr_60 | 0.715162 | NO |
| amihud_illiquidity_60 | time_of_day_sin | 0.634153 | NO |
| chaikin_money_flow_20 | distance_from_rolling_vwap_20_atr | 0.535819 | NO |
| volatility_of_volatility_60 | realized_volatility_60 | 0.284944 | NO |
| variance_ratio_60_5 | return_sign_change_rate_30 | 0.275637 | NO |
| bipower_jump_ratio_60 | return_sign_change_rate_30 | 0.108195 | NO |

## 8. Frozen MLAT feature batch

Membership and parameters were frozen before any MLAT feature/outcome relationship was calculated.

| hypothesis_id | feature_name | source_type | target_family | minimum_history | availability_timestamp | reset_boundary |
| --- | --- | --- | --- | --- | --- | --- |
| MLAT-H001 | bollinger_zscore_20 | MLAT-ADAPTED | direction | 20 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H002 | bollinger_bandwidth_20 | MLAT-DIRECT | expansion | 20 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H003 | cutler_rsi_14 | MLAT-ADAPTED | direction / risk state | 15 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H004 | chaikin_money_flow_20 | MLAT-ADAPTED | direction / risk state | 20 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H005 | amihud_illiquidity_60 | MLAT-ADAPTED | expansion / risk state | 61 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H006 | parkinson_volatility_30 | PROJECT-ORIGINAL-EXTENSION | volatility / expansion | 30 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H007 | rogers_satchell_volatility_30 | PROJECT-ORIGINAL-EXTENSION | volatility / expansion | 30 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H008 | realized_semivariance_balance_60 | PROJECT-ORIGINAL-EXTENSION | risk state / direction | 61 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H009 | bipower_jump_ratio_60 | PROJECT-ORIGINAL-EXTENSION | risk state / expansion | 62 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H010 | variance_ratio_60_5 | MLAT-ADAPTED | direction / risk state | 65 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H011 | return_sign_entropy_60 | PROJECT-ORIGINAL-EXTENSION | risk state / expansion | 61 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |
| MLAT-H012 | volatility_of_volatility_60 | PROJECT-ORIGINAL-EXTENSION | risk state / expansion | 75 | Close of completed decision bar t | Reset at missing-minute, product, selected-contract, instrument, continuous-segment, roll, tradability, or liquidity boundary. |

## 9. Exact formulas and sources

| hypothesis_id | feature_name | formula_or_definition | source_chapter | source_pdf_page |
| --- | --- | --- | --- | --- |
| MLAT-H001 | bollinger_zscore_20 | (close_t - population_mean_20(close)) / population_std_20(close) | 4 / Appendix | 131-133; 740-742 |
| MLAT-H002 | bollinger_bandwidth_20 | 4 * population_std_20(close) / population_mean_20(close) | Appendix | 740-742 |
| MLAT-H003 | cutler_rsi_14 | 100 * sum_14(max(delta_close, 0)) / (sum_14(max(delta_close, 0)) + sum_14(max(-delta_close, 0))) | 4 | 132 |
| MLAT-H004 | chaikin_money_flow_20 | sum_20(volume * ((2*close-high-low)/(high-low))) / sum_20(volume) | Appendix | 752-753 |
| MLAT-H005 | amihud_illiquidity_60 | 1e9 * mean_60(abs(log_return) / (close * volume)) | 20 / Appendix | 656; 752 |
| MLAT-H006 | parkinson_volatility_30 | 1e4 * sqrt(mean_30(log(high/low)^2) / (4*log(2))) | 9 | 297-301 |
| MLAT-H007 | rogers_satchell_volatility_30 | 1e4 * sqrt(mean_30(log(high/open)*log(high/close) + log(low/open)*log(low/close))) | 9 | 297-301 |
| MLAT-H008 | realized_semivariance_balance_60 | (sum_60(r^2 * 1[r>0]) - sum_60(r^2 * 1[r<0])) / sum_60(r^2) | 5 / 9 | 169-178; 297-301 |
| MLAT-H009 | bipower_jump_ratio_60 | max(RV-BV, 0)/RV; RV=sum_60(r^2); BV=(pi/2)*(60/59)*sum_60(abs(r_i)*abs(r_(i-1))) | 9 | 297-301 |
| MLAT-H010 | variance_ratio_60_5 | population_var_60(sum_5(log_return)) / (5*population_var_60(log_return)) | 9 | 280-296 |
| MLAT-H011 | return_sign_entropy_60 | -sum(p_s*log(p_s), s in {negative, zero, positive}) / log(3) | 6 | 192 |
| MLAT-H012 | volatility_of_volatility_60 | population_std_60(RV15) / mean_60(RV15), RV15=1e4*sqrt(mean_15(log_return^2)) | 9 | 297-301 |

The formula registry independently records causal history and boundary controls:

| formula_id | name | required_history | causal_availability | boundary_requirements | book_trace |
| --- | --- | --- | --- | --- | --- |
| MLAT-F002 | Bollinger z-score | 20 bars | close of t | Complete single continuity run. | Chapter 4 PDF 131-133; Appendix 740-742 |
| MLAT-F003 | Bollinger bandwidth | 20 bars | close of t | Complete single continuity run. | Appendix PDF 740-742 |
| MLAT-F004 | Cutler RSI | 15 bars | close of t | Complete single continuity run. | Chapter 4 PDF 132 (RSI example) |
| MLAT-F005 | Chaikin money flow adaptation | 20 bars | close of t | Complete single continuity run. | Appendix PDF 752-753 |
| MLAT-F006 | Amihud illiquidity adaptation | 60 returns (61 bars) | close of t | Complete single continuity run. | Chapter 20 PDF 656; Appendix 752 |
| MLAT-F007 | Parkinson range volatility | 30 bars | close of t | Complete single continuity run. | Chapter 9 context PDF 297-301 |
| MLAT-F008 | Rogers-Satchell range volatility | 30 bars | close of t | Complete single continuity run. | Chapter 9 context PDF 297-301 |
| MLAT-F009 | Realized semivariance balance | 60 returns (61 bars) | close of t | Complete single continuity run. | Chapters 5 and 9 context PDF 169-178 and 297-301 |
| MLAT-F010 | Bipower jump ratio | 61 returns (62 bars) | close of t | Complete single continuity run. | Chapter 9 context PDF 297-301 |
| MLAT-F011 | Variance ratio | 64 returns (65 bars) | close of t | Complete single continuity run. | Chapter 9 PDF 280-296 |
| MLAT-F012 | Return-sign entropy | 60 returns (61 bars) | close of t | Complete single continuity run. | Chapter 6 entropy context PDF 192 |
| MLAT-F013 | Volatility of volatility | 74 returns (75 bars) | close of t | Complete single continuity run. | Chapter 9 context PDF 297-301 |

## 10. Features implemented

The saved, registry-controlled feature matrix implements exactly 12 features. Every feature is available at the completed decision-bar close and is mapped to stable upstream observation IDs after chronological GC-only construction.

| feature_name | count | mean | std | min | max | missing_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| bollinger_zscore_20 | 424133 | 0.003867 | 1.322718 | -4.335307 | 4.319009 | 0.000408 |
| bollinger_bandwidth_20 | 424133 | 0.001811 | 0.001377 | 0.00014 | 0.024496 | 0.000408 |
| cutler_rsi_14 | 424180 | 50.147575 | 16.297583 | 0 | 100 | 0.000297 |
| chaikin_money_flow_20 | 424133 | 0.004692 | 0.16287 | -0.748814 | 0.732018 | 0.000408 |
| amihud_illiquidity_60 | 423702 | 0.888528 | 0.435145 | 0.13629 | 4.240716 | 0.001424 |
| parkinson_volatility_30 | 424047 | 2.654242 | 1.480329 | 0.47298 | 21.636385 | 0.00061 |
| rogers_satchell_volatility_30 | 424047 | 2.655442 | 1.501454 | 0.371083 | 23.668699 | 0.00061 |
| realized_semivariance_balance_60 | 423702 | 0.009332 | 0.255375 | -0.970532 | 0.992347 | 0.001424 |
| bipower_jump_ratio_60 | 423690 | 0.06756 | 0.09538 | 0 | 0.960133 | 0.001452 |
| variance_ratio_60_5 | 423654 | 0.889353 | 0.26888 | 0.134498 | 3.754612 | 0.001537 |
| return_sign_entropy_60 | 423702 | 0.840563 | 0.077986 | 0.556033 | 1 | 0.001424 |
| volatility_of_volatility_60 | 423544 | 0.23875 | 0.117792 | 0.042621 | 2.190053 | 0.001796 |

## 11. Features rejected before implementation

| ID | Feature | Source | Type | Target | Decision | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| MLAT-H013 | kalman_innovation_state | Ch. 4, PDF 133-136 | MLAT-ADAPTED | direction / risk state | DEFER | Noise parameters and Development cross-fitting are not justified in v1. |
| MLAT-H014 | wavelet_denoised_return | Ch. 4, PDF 137-140 | MLAT-ADAPTED | direction | REJECT | Book example uses a two-sided transform; causal reconstruction is not established. |
| MLAT-H015 | garch_variance_forecast | Ch. 9, PDF 297-301 | MLAT-DIRECT | volatility / risk state | AUDIT_ONLY | Prior implementation is invalid; keep outside frozen matrix pending segmented audit. |
| MLAT-H016 | ppo_12_26 | Ch. 11 / Appendix, PDF 355; 748-749 | MLAT-DIRECT | direction | REJECT | High formula and monotonic overlap with existing momentum/trend features. |
| MLAT-H017 | normalized_atr_14 | Ch. 11 / Appendix, PDF 355; 754-755 | MLAT-DIRECT | expansion | REJECT | Existing ATR anchors already dominate this hypothesis. |
| MLAT-H018 | kama_gap | Ch. Appendix, PDF 738-740 | MLAT-DIRECT | direction / risk state | DEFER | Recursive cost and near-direct overlap with existing efficiency ratios. |
| MLAT-H019 | on_balance_volume | Ch. Appendix, PDF 753 | MLAT-DIRECT | direction | REJECT | Cumulative scale/reset ambiguity and overlap with existing signed-volume features. |
| MLAT-H020 | rolling_ar1_coefficient | Ch. 9, PDF 290-296 | MLAT-DIRECT | direction / regime | REJECT | Formula-equivalent family already exists; variance ratio is the distinct selected adaptation. |
| MLAT-H021 | williams_r_14 | Ch. Appendix, PDF 751 | MLAT-DIRECT | direction | REJECT | Near formula-equivalent to existing rolling range position. |
| MLAT-H022 | intraday_seasonal_state | Ch. 4, PDF 130-131 | MLAT-ADAPTED | expansion | REJECT | Existing clock features are already strong frozen expansion representatives. |
| MLAT-H023 | cross_sectional_risk_factor | Ch. 4 / 13, PDF 115-130; 429-460 | MLAT-DIRECT | portfolio | DEFER_DIFFERENT_DATA | Single-instrument GC discovery has no cross-section. |
| MLAT-H024 | news_sentiment | Ch. 14-16, PDF 461-532 | MLAT-DIRECT | direction / risk | DEFER_DIFFERENT_DATA | No point-in-time news/text source is in the current OHLCV dataset. |
| MLAT-H025 | pca_regime_state | Ch. 13, PDF 429-460 | MLAT-ADAPTED | risk state | LATER_PHASE | Requires frozen engineered inputs and stability analysis first. |
| MLAT-H026 | tree_or_boosted_feature_importance | Ch. 11-12, PDF 350-428 | MLAT-DIRECT | model validation | LATER_PHASE | Importance is not a feature and cannot prove economic value. |
| MLAT-H027 | deep_sequence_model | Ch. 17-21, PDF 533-690 | MLAT-DIRECT | later modelling | LATER_PHASE | Simple features and linear benchmarks have not established directional information. |
| MLAT-H028 | reinforcement_learning_policy | Ch. 22, PDF 691-723 | MLAT-DIRECT | execution | REJECT_GOVERNANCE | No approved signal, reward, or validated simulator for this feature task. |
| MLAT-H029 | mgc_transfer_check | Ch. 1-2, PDF 42-94 | PROJECT-ORIGINAL-EXTENSION | transfer validation | LATER_PHASE | MGC is locked until GC definitions and verdicts are frozen. |

## 12. Leakage and boundary test results

All 23 feature-validation checks, 6 evaluation-scope checks, and 40 saved-artifact verification rows passed. The executed manifest additionally proves that Final-test outcomes and MGC were not loaded. These integrity checks do not open the separate advancement gate, which fails closed because frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized Synthetic and invariance coverage is recorded in the MLAT test suite; this report does not infer test outcomes from source presence.

| evidence_table | check | passed | details |
| --- | --- | --- | --- |
| validation_checks | registry_matrix_column_set | PASS |  |
| validation_checks | registry_feature_order | PASS |  |
| validation_checks | observation_id_unique | PASS |  |
| validation_checks | observation_id_order | PASS |  |
| validation_checks | eligible_observation_id_alignment | PASS |  |
| validation_checks | eligible_decision_timestamp_utc_alignment | PASS |  |
| validation_checks | eligible_decision_timestamp_ny_alignment | PASS |  |
| validation_checks | eligible_entry_timestamp_utc_alignment | PASS |  |
| validation_checks | eligible_entry_timestamp_ny_alignment | PASS |  |
| validation_checks | eligible_trade_date_ny_alignment | PASS |  |
| validation_checks | eligible_research_partition_alignment | PASS |  |
| validation_checks | eligible_product_alignment | PASS |  |
| validation_checks | gc_only | PASS |  |
| validation_checks | decision_timestamp_utc_valid_utc | PASS |  |
| validation_checks | entry_timestamp_utc_valid_utc | PASS |  |
| validation_checks | decision_precedes_entry | PASS |  |
| validation_checks | feature_dtypes | PASS |  |
| validation_checks | feature_missing_rate | PASS |  |
| validation_checks | feature_nonfinite_values | PASS |  |
| validation_checks | feature_validation_ranges | PASS |  |
| validation_checks | feature_constants | PASS |  |
| validation_checks | exact_duplicate_audit | PASS |  |
| validation_checks | candidate_near_duplicate_audit | PASS |  |
| evaluation_checks | gc_only | PASS |  |
| evaluation_checks | feature_label_id_alignment | PASS |  |
| evaluation_checks | feature_label_provenance_alignment | PASS |  |
| evaluation_checks | final_excluded | PASS |  |
| evaluation_checks | unknown_partition_excluded | PASS |  |
| evaluation_checks | new_york_date_evidence_present | PASS |  |

## 13. Development results

The table below reports the largest absolute Development daily Spearman IC cells from the complete preregistered screen. Ranking here is descriptive; advancement still requires every registered Validation, stability, economic, redundancy, and incremental gate.

| feature_name | target_family | horizon_minutes | entry_session | research_partition | eligible_dates | finite_observations | mean_daily_ic | q_value | absolute_monotonicity | top_bottom_tick_spread | thinned_mean_daily_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parkinson_volatility_30 | expansion | 180 | New York | Development | 638 | 188850 | -0.714274 | 0.001531 | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 180 | New York | Development | 638 | 188850 | -0.694345 | 0.001531 | 1 |  |  |
| parkinson_volatility_30 | path_risk | 180 | New York | Development | 638 | 188850 | -0.656673 | 0.001531 | 1 |  |  |
| parkinson_volatility_30 | expansion | 180 | London | Development | 636 | 113850 | -0.653863 | 0.001945 | 1 |  |  |
| parkinson_volatility_30 | expansion | 120 | New York | Development | 638 | 188850 | -0.642177 | 0.001531 | 1 |  |  |
| rogers_satchell_volatility_30 | path_risk | 180 | New York | Development | 638 | 188850 | -0.639355 | 0.001531 | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 120 | New York | Development | 638 | 188850 | -0.621491 | 0.001531 | 1 |  |  |
| amihud_illiquidity_60 | expansion | 180 | New York | Development | 638 | 188760 | 0.620271 | 0.001531 | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 180 | London | Development | 636 | 113850 | -0.604008 | 0.001945 | 1 |  |  |
| parkinson_volatility_30 | expansion | 120 | London | Development | 636 | 113850 | -0.583033 | 0.001945 | 1 |  |  |
| parkinson_volatility_30 | path_risk | 120 | New York | Development | 638 | 188850 | -0.582819 | 0.001531 | 1 |  |  |
| amihud_illiquidity_60 | expansion | 120 | New York | Development | 638 | 188760 | 0.579096 | 0.001531 | 1 |  |  |

## 14. Validation results

Validation used frozen Development transformations, quantile edges, and thresholds. The largest absolute Validation cells are shown without using the previously exposed Final-test outcomes.

| feature_name | target_family | horizon_minutes | entry_session | research_partition | eligible_dates | finite_observations | mean_daily_ic | q_value | absolute_monotonicity | top_bottom_tick_spread | thinned_mean_daily_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parkinson_volatility_30 | expansion | 180 | New York | Validation | 246 | 72980 | -0.741922 |  | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 180 | New York | Validation | 246 | 72980 | -0.72166 |  | 1 |  |  |
| parkinson_volatility_30 | expansion | 120 | New York | Validation | 246 | 72980 | -0.671155 |  | 1 |  |  |
| parkinson_volatility_30 | path_risk | 180 | New York | Validation | 246 | 72980 | -0.671065 |  | 1 |  |  |
| rogers_satchell_volatility_30 | path_risk | 180 | New York | Validation | 246 | 72980 | -0.654097 |  | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 120 | New York | Validation | 246 | 72980 | -0.65246 |  | 1 |  |  |
| parkinson_volatility_30 | expansion | 180 | London | Validation | 244 | 43612 | -0.63257 |  | 1 |  |  |
| amihud_illiquidity_60 | expansion | 180 | New York | Validation | 246 | 72895 | 0.625127 |  | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 180 | London | Validation | 244 | 43612 | -0.610769 |  | 1 |  |  |
| parkinson_volatility_30 | path_risk | 120 | New York | Validation | 246 | 72980 | -0.597433 |  | 1 |  |  |
| rogers_satchell_volatility_30 | path_risk | 120 | New York | Validation | 246 | 72980 | -0.584 |  | 1 |  |  |
| amihud_illiquidity_60 | path_risk | 180 | New York | Validation | 246 | 72895 | 0.574041 |  | 1 |  |  |

## 15. Directional findings

Directional advancement cells across final feature verdicts: 0. The tick-spread field is retained because a statistically detectable rank relationship was not allowed to substitute for the preregistered two-GC-tick economic screen. The closed structural advancement gate also prevents any v1 authorization.

| feature_name | target_family | horizon_minutes | entry_session | research_partition | eligible_dates | finite_observations | mean_daily_ic | q_value | absolute_monotonicity | top_bottom_tick_spread | thinned_mean_daily_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| realized_semivariance_balance_60 | direction | 180 | New York | Development | 638 | 188760 | -0.353803 | 0.002999 | 0.4 | 0.135562 |  |
| realized_semivariance_balance_60 | direction | 120 | London | Development | 635 | 113793 | -0.315205 | 0.002999 | 0.9 | 1.013779 |  |
| realized_semivariance_balance_60 | direction | 120 | New York | Development | 638 | 188760 | -0.298428 | 0.002999 | 0.3 | 1.280907 |  |
| realized_semivariance_balance_60 | direction | 180 | London | Development | 635 | 113793 | -0.296585 | 0.002999 | 0.4 | -0.44669 |  |
| realized_semivariance_balance_60 | direction | 60 | London | Development | 635 | 113793 | -0.295198 | 0.002999 | 0.2 | -0.407845 |  |
| realized_semivariance_balance_60 | direction | 120 | London | Validation | 244 | 43612 | -0.332328 |  | 0.7 | 1.544625 |  |
| realized_semivariance_balance_60 | direction | 180 | New York | Validation | 246 | 72895 | -0.319117 |  | 0.9 | 9.610012 |  |
| realized_semivariance_balance_60 | direction | 180 | London | Validation | 244 | 43612 | -0.293805 |  | 0.2 | 0.224669 |  |
| realized_semivariance_balance_60 | direction | 60 | London | Validation | 244 | 43612 | -0.287462 |  | 0.7 | 3.854934 |  |
| realized_semivariance_balance_60 | direction | 120 | New York | Validation | 246 | 72895 | -0.260566 |  | 0.8 | 5.08111 |  |

## 16. Expansion findings

Expansion advancement cells across final feature verdicts: 0. Expansion evidence is interpreted separately from signed direction and remains descriptive because the v1 advancement gate is structurally non-evaluable.

| feature_name | target_family | horizon_minutes | entry_session | research_partition | eligible_dates | finite_observations | mean_daily_ic | q_value | absolute_monotonicity | top_bottom_tick_spread | thinned_mean_daily_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parkinson_volatility_30 | expansion | 180 | New York | Development | 638 | 188850 | -0.714274 | 0.001531 | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 180 | New York | Development | 638 | 188850 | -0.694345 | 0.001531 | 1 |  |  |
| parkinson_volatility_30 | expansion | 180 | London | Development | 636 | 113850 | -0.653863 | 0.001945 | 1 |  |  |
| parkinson_volatility_30 | expansion | 120 | New York | Development | 638 | 188850 | -0.642177 | 0.001531 | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 120 | New York | Development | 638 | 188850 | -0.621491 | 0.001531 | 1 |  |  |
| parkinson_volatility_30 | expansion | 180 | New York | Validation | 246 | 72980 | -0.741922 |  | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 180 | New York | Validation | 246 | 72980 | -0.72166 |  | 1 |  |  |
| parkinson_volatility_30 | expansion | 120 | New York | Validation | 246 | 72980 | -0.671155 |  | 1 |  |  |
| rogers_satchell_volatility_30 | expansion | 120 | New York | Validation | 246 | 72980 | -0.65246 |  | 1 |  |  |
| parkinson_volatility_30 | expansion | 180 | London | Validation | 244 | 43612 | -0.63257 |  | 1 |  |  |

## 17. Risk-state findings

Volatility/path-risk advancement cells across final feature verdicts: 0. A risk-state feature need not predict signed returns, and no risk-state finding is presented as a trading policy. The raw relationships remain research-only under the closed v1 advancement gate.

| feature_name | target_family | horizon_minutes | entry_session | research_partition | eligible_dates | finite_observations | mean_daily_ic | q_value | absolute_monotonicity | top_bottom_tick_spread | thinned_mean_daily_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parkinson_volatility_30 | path_risk | 180 | New York | Development | 638 | 188850 | -0.656673 | 0.001531 | 1 |  |  |
| rogers_satchell_volatility_30 | path_risk | 180 | New York | Development | 638 | 188850 | -0.639355 | 0.001531 | 1 |  |  |
| parkinson_volatility_30 | path_risk | 120 | New York | Development | 638 | 188850 | -0.582819 | 0.001531 | 1 |  |  |
| amihud_illiquidity_60 | path_risk | 180 | New York | Development | 638 | 188760 | 0.576805 | 0.001531 | 1 |  |  |
| rogers_satchell_volatility_30 | path_risk | 120 | New York | Development | 638 | 188850 | -0.565257 | 0.001531 | 1 |  |  |
| parkinson_volatility_30 | path_risk | 180 | New York | Validation | 246 | 72980 | -0.671065 |  | 1 |  |  |
| rogers_satchell_volatility_30 | path_risk | 180 | New York | Validation | 246 | 72980 | -0.654097 |  | 1 |  |  |
| parkinson_volatility_30 | path_risk | 120 | New York | Validation | 246 | 72980 | -0.597433 |  | 1 |  |  |
| rogers_satchell_volatility_30 | path_risk | 120 | New York | Validation | 246 | 72980 | -0.584 |  | 1 |  |  |
| amihud_illiquidity_60 | path_risk | 180 | New York | Validation | 246 | 72895 | 0.574041 |  | 1 |  |  |

## 18. Session stability

London and New York were evaluated separately. The verdict-level session gate requires eligible evidence and Development-sign consistency across both sessions for the relevant family/horizon.

| session_stability | final_decision | feature_count |
| --- | --- | --- |
| MIXED: at least one family/horizon meets the cross-session gate and at least one does not | RESEARCH_ONLY | 12 |

## 19. Year stability

Annual sign agreement is evaluated against each corresponding full-period cell. Verdict-level results and the largest annual absolute-IC rows follow.

| year_stability | final_decision | feature_count |
| --- | --- | --- |
| MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | RESEARCH_ONLY | 12 |

| feature_name | target_family | horizon_minutes | entry_session | research_partition | year | eligible_dates | mean_daily_ic | full_period_sign_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parkinson_volatility_30 | expansion | 180 | New York | Development | 2023 | 242 | -0.748634 | PASS |
| parkinson_volatility_30 | expansion | 180 | New York | Validation | 2024 | 246 | -0.741922 | PASS |
| rogers_satchell_volatility_30 | expansion | 180 | New York | Development | 2023 | 242 | -0.737656 | PASS |
| rogers_satchell_volatility_30 | expansion | 180 | New York | Validation | 2024 | 246 | -0.72166 | PASS |
| parkinson_volatility_30 | expansion | 180 | New York | Development | 2022 | 244 | -0.695259 | PASS |
| parkinson_volatility_30 | path_risk | 180 | New York | Development | 2023 | 242 | -0.691248 | PASS |
| parkinson_volatility_30 | expansion | 180 | New York | Development | 2021 | 152 | -0.690093 | PASS |
| parkinson_volatility_30 | expansion | 180 | London | Development | 2021 | 151 | -0.686442 | PASS |
| parkinson_volatility_30 | expansion | 120 | New York | Development | 2023 | 242 | -0.686077 | PASS |
| rogers_satchell_volatility_30 | path_risk | 180 | New York | Development | 2023 | 242 | -0.684747 | PASS |
| rogers_satchell_volatility_30 | expansion | 180 | New York | Development | 2021 | 152 | -0.674113 | PASS |
| rogers_satchell_volatility_30 | expansion | 120 | New York | Development | 2023 | 242 | -0.671698 | PASS |
| parkinson_volatility_30 | expansion | 120 | New York | Validation | 2024 | 246 | -0.671155 | PASS |
| parkinson_volatility_30 | path_risk | 180 | New York | Validation | 2024 | 246 | -0.671065 | PASS |
| rogers_satchell_volatility_30 | expansion | 180 | New York | Development | 2022 | 244 | -0.663993 | PASS |
| parkinson_volatility_30 | expansion | 180 | London | Development | 2022 | 244 | -0.655506 | PASS |

## 20. Incremental information

Partial rank IC was evaluated at 60 and 180 minutes beyond both the ATR(20) anchor and the frozen existing-feature control set. Development magnitude, Validation sign/retention, and the registered thresholds jointly determine the incremental pass flag. The frozen set already contains ATR(20), so only the two distinct controls `atr_20` and `frozen_15` are reported. Incremental passes are descriptive: at the same 60/180-minute horizons, the frozen daily-thinning rule has no eligible dates under its at-least-10-observations requirement, so it cannot authorize advancement.

| control_set | tests | passing_tests | features |
| --- | --- | --- | --- |
| atr_20 | 192 | 100 | 12 |
| frozen_15 | 192 | 109 | 12 |

| feature_name | target_family | horizon_minutes | entry_session | control_set | mean_daily_partial_ic_development | mean_daily_partial_ic_validation | validation_magnitude_retention | incremental_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amihud_illiquidity_60 | volatility | 180 | New York | atr_20 | 0.505606 | 0.33244 | 0.657507 | PASS |
| realized_semivariance_balance_60 | direction | 180 | New York | frozen_15 | -0.372917 | -0.362593 | 0.972315 | PASS |
| amihud_illiquidity_60 | expansion | 180 | New York | atr_20 | 0.366626 | 0.253959 | 0.692691 | PASS |
| bollinger_zscore_20 | direction | 60 | London | frozen_15 | -0.358128 | -0.349886 | 0.976987 | PASS |
| bollinger_zscore_20 | direction | 180 | New York | frozen_15 | -0.340181 | -0.33065 | 0.971981 | PASS |
| bollinger_zscore_20 | direction | 180 | London | frozen_15 | -0.334113 | -0.323611 | 0.968566 | PASS |
| cutler_rsi_14 | direction | 180 | New York | frozen_15 | -0.326247 | -0.315409 | 0.96678 | PASS |
| amihud_illiquidity_60 | path_risk | 180 | New York | atr_20 | 0.326148 | 0.229308 | 0.703081 | PASS |
| realized_semivariance_balance_60 | direction | 180 | London | atr_20 | -0.32277 | -0.291235 | 0.902297 | PASS |
| cutler_rsi_14 | direction | 60 | London | frozen_15 | -0.322733 | -0.308436 | 0.9557 | PASS |
| realized_semivariance_balance_60 | direction | 60 | London | atr_20 | -0.322396 | -0.267188 | 0.828758 | PASS |
| realized_semivariance_balance_60 | direction | 60 | New York | frozen_15 | -0.315233 | -0.288387 | 0.914838 | PASS |
| realized_semivariance_balance_60 | direction | 60 | London | frozen_15 | -0.30955 | -0.30905 | 0.998385 | PASS |
| return_sign_entropy_60 | volatility | 180 | New York | atr_20 | 0.304515 | 0.253809 | 0.833488 | PASS |
| cutler_rsi_14 | direction | 180 | London | frozen_15 | -0.300316 | -0.290528 | 0.967409 | PASS |
| realized_semivariance_balance_60 | direction | 180 | New York | atr_20 | -0.28167 | -0.259092 | 0.919844 | PASS |

## 21. GARCH audit result

The governed audit verdict is `REJECTED_IMPLEMENTATION`. 3 critical audit checks failed, and 0 captured warnings were persisted. GARCH remains audit-only and outside the frozen feature matrix irrespective of diagnostic success.

### Audit checks

| category | check_name | critical | passed | details | advancement_authorized |
| --- | --- | --- | --- | --- | --- |
| garch_audit | fit_critical_checks | YES | PASS | all optimizer and parameter gates | NO |
| garch_audit | fit_residual_diagnostics | YES | FAIL | segment-aware Ljung-Box on residuals/squares and ARCH LM | NO |
| garch_audit | development_audit_residual_diagnostics | YES | FAIL | segment-aware Ljung-Box on residuals/squares and ARCH LM | NO |
| garch_audit | validation_residual_diagnostics | YES | FAIL | segment-aware Ljung-Box on residuals/squares and ARCH LM | NO |
| garch_audit | development_calibration_success | YES | PASS | multipliers fitted on 2023 only and frozen | NO |
| garch_audit | validation_improves_best_simple_anchor | YES | PASS | garch_rmse=0.0007403402599226602; anchor_rmse={'realized_volatility_anchor': 0.000798902868852521, 'atr_anchor': 0.0007543416482724045} | NO |
| garch_audit | audit_only_no_feature_authorization | NO | PASS | Passing diagnostics do not add GARCH to the frozen v1 matrix. | NO |

### Parameters

| model_name | model_version | mean | omega | alpha | beta | persistence | initial_variance | return_scale | fit_end_utc | n_fit_observations | n_fit_segments | optimizer_method | optimizer_success | optimizer_status | optimizer_message | objective_value | iterations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian GARCH(1,1) | mlat-garch-audit-v1 | 1.505e-07 | 3.475e-10 | 0.076002 | 0.913922 | 0.989924 | 3.449e-08 | 10000 | 2023-01-01 04:59:59+00:00 | 541160 | 1483 | L-BFGS-B | PASS | 0 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 2.112386 | 20 |

### Fit diagnostics

| category | check_name | critical | passed | numeric_value | threshold | details |
| --- | --- | --- | --- | --- | --- | --- |
| garch_fit | gc_only_input | YES | PASS | 1 | products == {'GC'} after explicit filtering | excluded_non_gc_rows=0 |
| garch_fit | fit_scope_ends_2022 | YES | PASS | 1.672e+18 | timestamp <= 2023-01-01T04:59:59+00:00 | max_fit_timestamp=2022-12-30T21:59:00+00:00 |
| garch_fit | optimizer_success | YES | PASS | 0 | scipy success == True | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH; iterations=20 |
| garch_fit | objective_finite | YES | PASS | 2.112386 | finite mean negative log likelihood | likelihood evaluated on scaled fit returns only |
| garch_fit | omega_positive | YES | PASS | 3.475e-10 | omega > 0 | raw log-return variance scale |
| garch_fit | alpha_nonnegative | YES | PASS | 0.076002 | alpha >= 0 | ARCH coefficient |
| garch_fit | beta_nonnegative | YES | PASS | 0.913922 | beta >= 0 | GARCH coefficient |
| garch_fit | persistence_below_one | YES | PASS | 0.989924 | alpha + beta < 1 | mathematical stationarity constraint |
| garch_fit | persistence_below_audit_limit | YES | PASS | 0.989924 | alpha + beta < 0.999 | frozen audit rejection threshold |

### Residual diagnostics

| period | diagnostic | lags | n_observations | statistic | p_value | significance | passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fit | ljung_box_standardized_residual | 10 | 541160 | 502.324046 | 1.407e-101 | 0.01 | FAIL |
| fit | ljung_box_squared_standardized_residual | 10 | 541160 | 73.030215 | 1.149e-11 | 0.01 | FAIL |
| fit | ljung_box_standardized_residual | 20 | 541160 | 538.116113 | 2.973e-101 | 0.01 | FAIL |
| fit | ljung_box_squared_standardized_residual | 20 | 541160 | 88.427279 | 1.394e-10 | 0.01 | FAIL |
| fit | arch_lm_standardized_residual | 20 | 518601 | 72.482941 | 7.125e-08 | 0.01 | FAIL |
| development_audit | ljung_box_standardized_residual | 10 | 330543 | 186.109072 | 1.26e-34 | 0.01 | FAIL |
| development_audit | ljung_box_squared_standardized_residual | 10 | 330543 | 35.575299 | 9.956e-05 | 0.01 | FAIL |
| development_audit | ljung_box_standardized_residual | 20 | 330543 | 211.084585 | 7.128e-34 | 0.01 | FAIL |
| development_audit | ljung_box_squared_standardized_residual | 20 | 330543 | 45.627583 | 0.000906 | 0.01 | FAIL |
| development_audit | arch_lm_standardized_residual | 20 | 313601 | 34.406714 | 0.023502 | 0.01 | PASS |
| validation | ljung_box_standardized_residual | 10 | 336415 | 90.233882 | 4.813e-15 | 0.01 | FAIL |
| validation | ljung_box_squared_standardized_residual | 10 | 336415 | 13.11923 | 0.217082 | 0.01 | PASS |
| validation | ljung_box_standardized_residual | 20 | 336415 | 122.571806 | 9.505e-17 | 0.01 | FAIL |
| validation | ljung_box_squared_standardized_residual | 20 | 336415 | 22.00809 | 0.340072 | 0.01 | PASS |
| validation | arch_lm_standardized_residual | 20 | 322020 | 20.217409 | 0.444406 | 0.01 | PASS |

### Development calibration

| method | raw_forecast_column | calibrated_forecast_column | calibration_method | calibration_start_utc | calibration_end_utc | n_fit_observations | multiplier | intercept | success | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| garch | garch_sigma_forecast_60m | garch_sigma_calibrated_60m | nonnegative_zero_intercept_least_squares | 2023-01-01 05:00:00+00:00 | 2024-01-01 04:59:59+00:00 | 295599 | 0.966844 | 0 | PASS | frozen 2023 zero-intercept least-squares scale |
| realized_volatility_anchor | rv_anchor_sigma_60m | rv_anchor_sigma_calibrated_60m | nonnegative_zero_intercept_least_squares | 2023-01-01 05:00:00+00:00 | 2024-01-01 04:59:59+00:00 | 275427 | 0.903134 | 0 | PASS | frozen 2023 zero-intercept least-squares scale |
| atr_anchor | atr_anchor_sigma_60m | atr_anchor_sigma_calibrated_60m | nonnegative_zero_intercept_least_squares | 2023-01-01 05:00:00+00:00 | 2024-01-01 04:59:59+00:00 | 288874 | 2.839833 | 0 | PASS | frozen 2023 zero-intercept least-squares scale |

### Simple-anchor comparison

| method | period | n_observations | rmse_sigma | mae_sigma | mean_forecast_sigma | mean_realized_sigma | mean_forecast_to_realized_ratio | qlike | pearson_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| garch | development_audit | 295599 | 0.000689 | 0.000397 | 0.001472 | 0.001519 | 0.968865 | 0.336723 | 0.640069 |
| garch | validation | 304407 | 0.00074 | 0.000423 | 0.001588 | 0.001703 | 0.932412 | 0.341708 | 0.653324 |
| realized_volatility_anchor | development_audit | 275427 | 0.000745 | 0.000439 | 0.001411 | 0.001552 | 0.909124 | 0.535279 | 0.630285 |
| realized_volatility_anchor | validation | 284697 | 0.000799 | 0.000475 | 0.001578 | 0.00174 | 0.906884 | 0.496411 | 0.640125 |
| atr_anchor | development_audit | 288874 | 0.000724 | 0.000429 | 0.001358 | 0.001531 | 0.887134 | 0.586556 | 0.676554 |
| atr_anchor | validation | 297808 | 0.000754 | 0.000451 | 0.001554 | 0.001716 | 0.905466 | 0.498062 | 0.692319 |

### Captured warnings

_The corresponding executed evidence table contains zero rows._

## 22. Feature verdicts

Every final decision below fails closed as `RESEARCH_ONLY` because the frozen v1 thinning gate is structurally non-evaluable. The preserved pre-authorization label summarizes other descriptive gates only and must not be read as an advancement decision.

| hypothesis_id | feature_name | book_chapter | source_pdf_page | development_result | validation_result | session_stability | year_stability | redundancy_result | incremental_information_result | economic_interpretation | descriptive_pre_authorization_verdict | final_decision | verdict_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLAT-H001 | bollinger_zscore_20 | 4 / Appendix | 131-133; 740-742 | 34/48 cells pass the Development screen; max \|Development daily IC\|=0.2242 | 34/34 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.2171 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 15/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | ECONOMICALLY_TRIVIAL | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H002 | bollinger_bandwidth_20 | Appendix | 740-742 | 35/48 cells pass the Development screen; max \|Development daily IC\|=0.5468 | 34/35 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.5570 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 20/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H003 | cutler_rsi_14 | 4 | 132 | 30/48 cells pass the Development screen; max \|Development daily IC\|=0.2267 | 30/30 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.2152 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 17/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | ECONOMICALLY_TRIVIAL | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H004 | chaikin_money_flow_20 | Appendix | 752-753 | 16/48 cells pass the Development screen; max \|Development daily IC\|=0.1311 | 16/16 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.1394 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 11/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | ECONOMICALLY_TRIVIAL | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H005 | amihud_illiquidity_60 | 20 / Appendix | 656; 752 | 34/48 cells pass the Development screen; max \|Development daily IC\|=0.6203 | 33/34 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.6251 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 19/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H006 | parkinson_volatility_30 | 9 | 297-301 | 35/48 cells pass the Development screen; max \|Development daily IC\|=0.7143 | 34/35 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.7419 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 24/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H007 | rogers_satchell_volatility_30 | 9 | 297-301 | 36/48 cells pass the Development screen; max \|Development daily IC\|=0.6943 | 34/36 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.7217 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 22/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H008 | realized_semivariance_balance_60 | 5 / 9 | 169-178; 297-301 | 24/48 cells pass the Development screen; max \|Development daily IC\|=0.3538 | 22/24 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.3323 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 15/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H009 | bipower_jump_ratio_60 | 9 | 297-301 | 10/48 cells pass the Development screen; max \|Development daily IC\|=0.0837 | 10/10 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.0630 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 10/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H010 | variance_ratio_60_5 | 9 | 280-296 | 31/48 cells pass the Development screen; max \|Development daily IC\|=0.1109 | 31/31 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.0801 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 20/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H011 | return_sign_entropy_60 | 6 | 192 | 35/48 cells pass the Development screen; max \|Development daily IC\|=0.4604 | 34/35 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.4429 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 21/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |
| MLAT-H012 | volatility_of_volatility_60 | 9 | 297-301 | 29/48 cells pass the Development screen; max \|Development daily IC\|=0.1800 | 29/29 Development-screened cells confirmed in Validation; max \|confirmed Validation daily IC\|=0.2082 | MIXED: at least one family/horizon meets the cross-session gate and at least one does not | MIXED: at least one evaluated cell has stable annual IC signs and at least one does not | PASS: no exact or >=0.995 existing-feature overlap | 15/32 60/180 horizon-control tests pass | No incremental, stable, nonredundant economic interpretation is supported. | WEAK_OR_UNSTABLE | RESEARCH_ONLY | frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized |

## 23. Limitations

- Minute observations and overlapping forward paths are dependent; dates, sessions, block uncertainty, and horizon thinning mitigate but do not erase that dependence.
- Frozen v1 advancement is structurally non-evaluable: frozen v1 horizon-thinning gate is structurally non-evaluable: 0/384 required 60/180-minute feature-family-session-partition cells have finite thinned daily-IC evidence under the >=10 observations-per-New-York-date rule, including 0 Development cells; advancement is not authorized
- Outcomes are now viewed, so changing the v1 thinning statistic would be post hoc; a feasible replacement belongs in a preregistered v2 contract.
- The historical Final-test interval was exposed in the prior research cycle and is not a pristine holdout.
- MGC transfer validation was locked and not inspected.
- The experiment uses one-minute OHLCV and cannot answer quote-, trade-, order-book-, macro-, text-, or cross-sectional hypotheses.
- No sequential entry/exit policy, transaction-cost model, PnL, or Sharpe analysis was authorized.
- Notebook execution recorded 2159.152 seconds and 1,726,779,392 final-process RSS bytes; these are machine/run observations, not universal performance benchmarks.

## 24. Final-test governance

The execution manifest records `development_validation_only=True`, `final_outcomes_loaded=False`, and `mgc_loaded=False`. The historical Final-test period was therefore excluded from feature/outcome evaluation, selection, parameter tuning, and displays. A new future holdout or live paper period is required for genuine final confirmation.

## 25. Recommended next research stage

Preregister an MLAT v2 contract with a feasible non-overlap sensitivity statistic before examining new outcomes; retain v1 as fail-closed RESEARCH_ONLY evidence and require a new future holdout or live paper period.

The executed multivariate authorization flag is `False`. No nonlinear-model or backtest stage is implied when that gate is closed.

## 26. Files created and modified

The inventory below is generated from the known MLAT documentation, notebook, source, test, data, report, and figure paths. 42 manifest-listed artifacts were hash-verified before this report was built. The historical statistical notebook and context report were required as present, read only, and are not write targets of this builder. Their hashes at report time were `53e7432882bd368727e78bf57d5df3b38a36520751ef5163f41a2e65fc0e02f1` and `05fcad59d1cd900c374d5cb93eabd42e95bcde1d13f056484d52004d171d4ed2`, respectively.

### Documentation

- `project_docs/mlat_feature_research/chapter_summaries/appendix_alpha_factor_library.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_01.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_02.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_03.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_04.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_05.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_06.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_07.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_08.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_09.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_10.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_11.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_12.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_13.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_14.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_15.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_16.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_17.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_18.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_19.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_20.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_21.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_22.md`
- `project_docs/mlat_feature_research/chapter_summaries/chapter_23.md`
- `project_docs/mlat_feature_research/mlat_book_coverage_log.md`
- `project_docs/mlat_feature_research/mlat_book_ingestion_manifest.csv`
- `project_docs/mlat_feature_research/mlat_book_map.md`
- `project_docs/mlat_feature_research/mlat_concept_registry.csv`
- `project_docs/mlat_feature_research/mlat_concept_registry.md`
- `project_docs/mlat_feature_research/mlat_current_state_audit.md`
- `project_docs/mlat_feature_research/mlat_existing_feature_overlap_audit.csv`
- `project_docs/mlat_feature_research/mlat_feature_batch_v1.md`
- `project_docs/mlat_feature_research/mlat_feature_hypothesis_catalog.md`
- `project_docs/mlat_feature_research/mlat_feature_research_context_report.md`
- `project_docs/mlat_feature_research/mlat_feature_research_contract_v1.md`
- `project_docs/mlat_feature_research/mlat_final_research_report.md`
- `project_docs/mlat_feature_research/mlat_final_test_governance_note.md`
- `project_docs/mlat_feature_research/mlat_formulas_and_definitions.md`
- `project_docs/mlat_feature_research/mlat_implementation_patterns.md`
- `project_docs/mlat_feature_research/mlat_implementation_plan_v1.md`
- `project_docs/mlat_feature_research/mlat_methodological_warnings.md`
- `project_docs/mlat_feature_research/mlat_project_applicability_matrix.md`
- `project_docs/mlat_feature_research/mlat_source_traceability.csv`
- `project_docs/mlat_feature_research/mlat_stage1_completion_report.md`
- `project_docs/mlat_feature_research/mlat_upstream_artifact_manifest.csv`
- `project_docs/mlat_feature_research/README.md`

### Notebook

- `notebooks/exploration/mlat_feature_research.ipynb`

### Reusable source

- `src/statistical_research/mlat_artifacts.py`
- `src/statistical_research/mlat_feature_engineering.py`
- `src/statistical_research/mlat_feature_evaluation.py`
- `src/statistical_research/mlat_feature_registry.py`
- `src/statistical_research/mlat_feature_validation.py`
- `src/statistical_research/mlat_volatility_models.py`

### Tests

- `tests/test_mlat_artifacts.py`
- `tests/test_mlat_feature_engineering.py`
- `tests/test_mlat_feature_evaluation.py`
- `tests/test_mlat_feature_registry.py`
- `tests/test_mlat_feature_validation.py`
- `tests/test_mlat_volatility_models.py`

### Build scripts

- `scripts/build_mlat_feature_research_notebook.py`
- `scripts/build_mlat_final_docs.py`
- `scripts/build_mlat_stage1_docs.py`

### Processed data

- `data/processed/statistical_research/mlat_feature_research/v1/daily_ic_evidence.parquet`
- `data/processed/statistical_research/mlat_feature_research/v1/feature_matrix_mlat_gc.parquet`
- `data/processed/statistical_research/mlat_feature_research/v1/feature_observation_audit.parquet`
- `data/processed/statistical_research/mlat_feature_research/v1/feature_registry_mlat_gc.parquet`
- `data/processed/statistical_research/mlat_feature_research/v1/incremental_daily_partial_ic.parquet`

### Report artifacts

- `reports/statistical_research/mlat_feature_research/v1/execution_manifest.json`
- `reports/statistical_research/mlat_feature_research/v1/table_manifest.json`
- `reports/statistical_research/mlat_feature_research/v1/tables/artifact_verification.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/candidate_correlation.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/construction_performance.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/evaluation_checks.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/evaluation_family_coverage.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/evaluation_target_catalog.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/feature_build_runtime.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/feature_build_timings.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/feature_construction_audit.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/feature_engineering_diagnostics.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/feature_persistence_validation.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/feature_reload_checks.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/feature_save_checks.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/feature_summary.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/feature_verdicts.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/figure_manifest.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/garch_audit_checks.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/garch_benchmarks.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/garch_calibration.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/garch_fit_diagnostics.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/garch_parameters.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/garch_residual_diagnostics.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/garch_warnings.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/incremental_summary.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/overlap_audit.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/quintile_edges.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/quintile_results.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/registry_save_checks.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/thinning_results.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/univariate_cells.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/upstream_integrity_checks.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/validation_checks.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/validation_coverage.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/validation_feature_diagnostics.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/validation_sample_hash.csv`
- `reports/statistical_research/mlat_feature_research/v1/tables/year_stability.csv`

### Figures

- `reports/figures/mlat_feature_research/v1/candidate_spearman_correlation.png`
- `reports/figures/mlat_feature_research/v1/development_validation_ic.png`
- `reports/figures/mlat_feature_research/v1/feature_missingness.png`
- `reports/figures/mlat_feature_research/v1/garch_validation_benchmarks.png`
