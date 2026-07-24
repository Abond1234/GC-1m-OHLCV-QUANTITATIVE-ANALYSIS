# MLAT Feature Hypothesis Catalog

This catalog was frozen before any MLAT feature/outcome relationship was
calculated. Project-original extensions cite the motivating book discussion
without claiming their exact estimator appears in the book.

| ID | Feature | Source | Type | Target | Overlap | Leakage | Cost | Decision |
|---|---|---|---|---|---|---|---|---|
| MLAT-H001 | bollinger_zscore_20 | Ch. 4 / Appendix, PDF 131-133; 740-742 | MLAT-ADAPTED | direction | range position, OLS slope | LOW | LOW | SELECT_V1 |
| MLAT-H002 | bollinger_bandwidth_20 | Ch. Appendix, PDF 740-742 | MLAT-DIRECT | expansion | compression ratio, ATR ratios | LOW | LOW | SELECT_V1 |
| MLAT-H003 | cutler_rsi_14 | Ch. 4, PDF 132 | MLAT-ADAPTED | direction / risk state | momentum, persistence | LOW | LOW | SELECT_V1 |
| MLAT-H004 | chaikin_money_flow_20 | Ch. Appendix, PDF 752-753 | MLAT-ADAPTED | direction / risk state | CLV, signed volume, alignment | LOW | LOW | SELECT_V1 |
| MLAT-H005 | amihud_illiquidity_60 | Ch. 20 / Appendix, PDF 656; 752 | MLAT-ADAPTED | expansion / risk state | volume_per_tick_range, liquidity_vacuum | LOW | LOW | SELECT_V1 |
| MLAT-H006 | parkinson_volatility_30 | Ch. 9, PDF 297-301 | PROJECT-ORIGINAL-EXTENSION | volatility / expansion | ATR, RV, range | LOW | LOW | SELECT_V1 |
| MLAT-H007 | rogers_satchell_volatility_30 | Ch. 9, PDF 297-301 | PROJECT-ORIGINAL-EXTENSION | volatility / expansion | ATR, RV, range | LOW | LOW | SELECT_V1 |
| MLAT-H008 | realized_semivariance_balance_60 | Ch. 5 / 9, PDF 169-178; 297-301 | PROJECT-ORIGINAL-EXTENSION | risk state / direction | directional energy balance | LOW | LOW | SELECT_V1 |
| MLAT-H009 | bipower_jump_ratio_60 | Ch. 9, PDF 297-301 | PROJECT-ORIGINAL-EXTENSION | risk state / expansion | current range/ATR, liquidity vacuum | LOW | MEDIUM | SELECT_V1 |
| MLAT-H010 | variance_ratio_60_5 | Ch. 9, PDF 280-296 | MLAT-ADAPTED | direction / risk state | return autocorrelation, sign change | LOW | LOW | SELECT_V1 |
| MLAT-H011 | return_sign_entropy_60 | Ch. 6, PDF 192 | PROJECT-ORIGINAL-EXTENSION | risk state / expansion | choppiness, sign change | LOW | LOW | SELECT_V1 |
| MLAT-H012 | volatility_of_volatility_60 | Ch. 9, PDF 297-301 | PROJECT-ORIGINAL-EXTENSION | risk state / expansion | ATR/RV ratios | LOW | LOW | SELECT_V1 |
| MLAT-H013 | kalman_innovation_state | Ch. 4, PDF 133-136 | MLAT-ADAPTED | direction / risk state | EMA/trend states | HIGH | HIGH | DEFER |
|  | Rejection/defer reason |  |  |  |  |  |  | Noise parameters and Development cross-fitting are not justified in v1. |
| MLAT-H014 | wavelet_denoised_return | Ch. 4, PDF 137-140 | MLAT-ADAPTED | direction | trend features | VERY HIGH | HIGH | REJECT |
|  | Rejection/defer reason |  |  |  |  |  |  | Book example uses a two-sided transform; causal reconstruction is not established. |
| MLAT-H015 | garch_variance_forecast | Ch. 9, PDF 297-301 | MLAT-DIRECT | volatility / risk state | ATR/RV anchors | HIGH | HIGH | AUDIT_ONLY |
|  | Rejection/defer reason |  |  |  |  |  |  | Prior implementation is invalid; keep outside frozen matrix pending segmented audit. |
| MLAT-H016 | ppo_12_26 | Ch. 11 / Appendix, PDF 355; 748-749 | MLAT-DIRECT | direction | multi-horizon momentum/OLS | LOW | MEDIUM | REJECT |
|  | Rejection/defer reason |  |  |  |  |  |  | High formula and monotonic overlap with existing momentum/trend features. |
| MLAT-H017 | normalized_atr_14 | Ch. 11 / Appendix, PDF 355; 754-755 | MLAT-DIRECT | expansion | atr_20 and existing ATR family | LOW | LOW | REJECT |
|  | Rejection/defer reason |  |  |  |  |  |  | Existing ATR anchors already dominate this hypothesis. |
| MLAT-H018 | kama_gap | Ch. Appendix, PDF 738-740 | MLAT-DIRECT | direction / risk state | efficiency ratios and slopes | MEDIUM | HIGH | DEFER |
|  | Rejection/defer reason |  |  |  |  |  |  | Recursive cost and near-direct overlap with existing efficiency ratios. |
| MLAT-H019 | on_balance_volume | Ch. Appendix, PDF 753 | MLAT-DIRECT | direction | signed volume proxy | MEDIUM | MEDIUM | REJECT |
|  | Rejection/defer reason |  |  |  |  |  |  | Cumulative scale/reset ambiguity and overlap with existing signed-volume features. |
| MLAT-H020 | rolling_ar1_coefficient | Ch. 9, PDF 290-296 | MLAT-DIRECT | direction / regime | return_autocorrelation_15 | LOW | MEDIUM | REJECT |
|  | Rejection/defer reason |  |  |  |  |  |  | Formula-equivalent family already exists; variance ratio is the distinct selected adaptation. |
| MLAT-H021 | williams_r_14 | Ch. Appendix, PDF 751 | MLAT-DIRECT | direction | rolling_range_position_15 | LOW | LOW | REJECT |
|  | Rejection/defer reason |  |  |  |  |  |  | Near formula-equivalent to existing rolling range position. |
| MLAT-H022 | intraday_seasonal_state | Ch. 4, PDF 130-131 | MLAT-ADAPTED | expansion | existing clock calendar family | LOW | LOW | REJECT |
|  | Rejection/defer reason |  |  |  |  |  |  | Existing clock features are already strong frozen expansion representatives. |
| MLAT-H023 | cross_sectional_risk_factor | Ch. 4 / 13, PDF 115-130; 429-460 | MLAT-DIRECT | portfolio | none | HIGH | HIGH | DEFER_DIFFERENT_DATA |
|  | Rejection/defer reason |  |  |  |  |  |  | Single-instrument GC discovery has no cross-section. |
| MLAT-H024 | news_sentiment | Ch. 14-16, PDF 461-532 | MLAT-DIRECT | direction / risk | none | HIGH | HIGH | DEFER_DIFFERENT_DATA |
|  | Rejection/defer reason |  |  |  |  |  |  | No point-in-time news/text source is in the current OHLCV dataset. |
| MLAT-H025 | pca_regime_state | Ch. 13, PDF 429-460 | MLAT-ADAPTED | risk state | existing clustering/frozen set | MEDIUM | HIGH | LATER_PHASE |
|  | Rejection/defer reason |  |  |  |  |  |  | Requires frozen engineered inputs and stability analysis first. |
| MLAT-H026 | tree_or_boosted_feature_importance | Ch. 11-12, PDF 350-428 | MLAT-DIRECT | model validation | none | HIGH | HIGH | LATER_PHASE |
|  | Rejection/defer reason |  |  |  |  |  |  | Importance is not a feature and cannot prove economic value. |
| MLAT-H027 | deep_sequence_model | Ch. 17-21, PDF 533-690 | MLAT-DIRECT | later modelling | none | VERY HIGH | VERY HIGH | LATER_PHASE |
|  | Rejection/defer reason |  |  |  |  |  |  | Simple features and linear benchmarks have not established directional information. |
| MLAT-H028 | reinforcement_learning_policy | Ch. 22, PDF 691-723 | MLAT-DIRECT | execution | none | VERY HIGH | VERY HIGH | REJECT_GOVERNANCE |
|  | Rejection/defer reason |  |  |  |  |  |  | No approved signal, reward, or validated simulator for this feature task. |
| MLAT-H029 | mgc_transfer_check | Ch. 1-2, PDF 42-94 | PROJECT-ORIGINAL-EXTENSION | transfer validation | n/a | MEDIUM | HIGH | LATER_PHASE |
|  | Rejection/defer reason |  |  |  |  |  |  | MGC is locked until GC definitions and verdicts are frozen. |

## Full definitions

### MLAT-H001 - `bollinger_zscore_20`

- Rationale: Standardized local price displacement may identify reversal/continuation states.
- Definition: Bollinger z-score in formula registry
- History/input: 20 bars; close
- Availability/reset: close of t; continuity run
- Target and expected shape: direction; nonlinear or monotone
- Existing overlap: range position, OLS slope
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H002 - `bollinger_bandwidth_20`

- Rationale: Narrow bands may precede expansion while wide bands may mean-revert.
- Definition: Normalized four-sigma band width
- History/input: 20 bars; close
- Availability/reset: close of t; continuity run
- Target and expected shape: expansion; possibly negative at short horizons and nonlinear
- Existing overlap: compression ratio, ATR ratios
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H003 - `cutler_rsi_14`

- Rationale: Bounded gain/loss balance may distinguish exhausted from persistent moves.
- Definition: Simple-rolling Cutler RSI
- History/input: 15 bars; close
- Availability/reset: close of t; continuity run
- Target and expected shape: direction / risk state; no fixed 30/70 sign assumed
- Existing overlap: momentum, persistence
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H004 - `chaikin_money_flow_20`

- Rationale: Close location weighted by volume may reveal pressure not present in either input alone.
- Definition: Rolling normalized money-flow volume
- History/input: 20 bars; OHLCV
- Availability/reset: close of t; continuity run
- Target and expected shape: direction / risk state; positive may indicate buying pressure
- Existing overlap: CLV, signed volume, alignment
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H005 - `amihud_illiquidity_60`

- Rationale: Movement per unit activity may identify fragile/liquidity-vacuum states.
- Definition: Mean absolute return divided by close-volume proxy
- History/input: 60 returns (61 bars); close, volume
- Availability/reset: close of t; continuity run
- Target and expected shape: expansion / risk state; higher may imply greater future risk
- Existing overlap: volume_per_tick_range, liquidity_vacuum
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H006 - `parkinson_volatility_30`

- Rationale: High-low information may estimate latent volatility more efficiently than close-only RV.
- Definition: Parkinson estimator
- History/input: 30 bars; high, low
- Availability/reset: close of t; continuity run
- Target and expected shape: volatility / expansion; positive monotone
- Existing overlap: ATR, RV, range
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H007 - `rogers_satchell_volatility_30`

- Rationale: Drift-robust OHLC variation may retain information beyond ATR and close-only RV.
- Definition: Rogers-Satchell estimator
- History/input: 30 bars; OHLC
- Availability/reset: close of t; continuity run
- Target and expected shape: volatility / expansion; positive monotone
- Existing overlap: ATR, RV, range
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H008 - `realized_semivariance_balance_60`

- Rationale: Asymmetry in recent signed variation may distinguish downside/upside risk state.
- Definition: Signed semivariance difference over total RV
- History/input: 60 returns (61 bars); close
- Availability/reset: close of t; continuity run
- Target and expected shape: risk state / direction; shape unknown
- Existing overlap: directional energy balance
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H009 - `bipower_jump_ratio_60`

- Rationale: The share of local variation attributable to jumps may alter subsequent expansion and risk.
- Definition: Positive RV-minus-bipower share
- History/input: 61 returns (62 bars); close
- Availability/reset: close of t; continuity run
- Target and expected shape: risk state / expansion; higher may mark stress then mean reversion or persistence
- Existing overlap: current range/ATR, liquidity vacuum
- Risks: leakage LOW; computation MEDIUM; numerical LOW
- Decision: SELECT_V1

### MLAT-H010 - `variance_ratio_60_5`

- Rationale: Deviation from local random-walk variance scaling may identify persistence or mean reversion.
- Definition: Five-minute to one-minute variance ratio
- History/input: 64 returns (65 bars); close
- Availability/reset: close of t; continuity run
- Target and expected shape: direction / risk state; below 1 mean-reverting; above 1 persistent
- Existing overlap: return autocorrelation, sign change
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H011 - `return_sign_entropy_60`

- Rationale: Low sign entropy may reflect directional organization; high entropy may reflect chop.
- Definition: Normalized three-state Shannon entropy
- History/input: 60 returns (61 bars); close
- Availability/reset: close of t; continuity run
- Target and expected shape: risk state / expansion; shape unknown
- Existing overlap: choppiness, sign change
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H012 - `volatility_of_volatility_60`

- Rationale: Instability of volatility, rather than its level, may identify transition risk.
- Definition: CV of trailing RV15 over 60 values
- History/input: 74 returns (75 bars); close
- Availability/reset: close of t; continuity run
- Target and expected shape: risk state / expansion; higher may imply unstable expansion
- Existing overlap: ATR/RV ratios
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: SELECT_V1

### MLAT-H013 - `kalman_innovation_state`

- Rationale: Sequential innovations may capture filtered state surprises.
- Definition: State-space innovation
- History/input: expanding; close
- Availability/reset: close of t; continuity run
- Target and expected shape: direction / risk state; unknown
- Existing overlap: EMA/trend states
- Risks: leakage HIGH; computation HIGH; numerical MEDIUM
- Decision: DEFER - Noise parameters and Development cross-fitting are not justified in v1.

### MLAT-H014 - `wavelet_denoised_return`

- Rationale: Multiscale denoising may isolate trend.
- Definition: Wavelet threshold and inverse transform
- History/input: transform dependent; close
- Availability/reset: uncertain; continuity run
- Target and expected shape: direction; unknown
- Existing overlap: trend features
- Risks: leakage VERY HIGH; computation HIGH; numerical HIGH
- Decision: REJECT - Book example uses a two-sided transform; causal reconstruction is not established.

### MLAT-H015 - `garch_variance_forecast`

- Rationale: Conditional variance may forecast future expansion beyond ATR.
- Definition: GARCH(1,1)
- History/input: expanding; close
- Availability/reset: before forecast bar; continuity run
- Target and expected shape: volatility / risk state; positive monotone
- Existing overlap: ATR/RV anchors
- Risks: leakage HIGH; computation HIGH; numerical HIGH
- Decision: AUDIT_ONLY - Prior implementation is invalid; keep outside frozen matrix pending segmented audit.

### MLAT-H016 - `ppo_12_26`

- Rationale: Normalized EMA spread captures trend.
- Definition: PPO/MACD family
- History/input: 26+ bars; close
- Availability/reset: close of t; continuity run
- Target and expected shape: direction; trend-following
- Existing overlap: multi-horizon momentum/OLS
- Risks: leakage LOW; computation MEDIUM; numerical LOW
- Decision: REJECT - High formula and monotonic overlap with existing momentum/trend features.

### MLAT-H017 - `normalized_atr_14`

- Rationale: Price-normalized ATR measures volatility.
- Definition: ATR divided by price
- History/input: 15 bars; OHLC
- Availability/reset: close of t; continuity run
- Target and expected shape: expansion; positive
- Existing overlap: atr_20 and existing ATR family
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: REJECT - Existing ATR anchors already dominate this hypothesis.

### MLAT-H018 - `kama_gap`

- Rationale: Adaptive smoothing may distinguish efficient trends from noise.
- Definition: KAMA gap
- History/input: adaptive; close
- Availability/reset: close of t; continuity run
- Target and expected shape: direction / risk state; unknown
- Existing overlap: efficiency ratios and slopes
- Risks: leakage MEDIUM; computation HIGH; numerical MEDIUM
- Decision: DEFER - Recursive cost and near-direct overlap with existing efficiency ratios.

### MLAT-H019 - `on_balance_volume`

- Rationale: Cumulative signed volume may lead price.
- Definition: OBV cumulative sum
- History/input: expanding; close, volume
- Availability/reset: close of t; contract/session reset unresolved
- Target and expected shape: direction; trend-following
- Existing overlap: signed volume proxy
- Risks: leakage MEDIUM; computation MEDIUM; numerical LOW
- Decision: REJECT - Cumulative scale/reset ambiguity and overlap with existing signed-volume features.

### MLAT-H020 - `rolling_ar1_coefficient`

- Rationale: Local AR coefficient estimates serial persistence.
- Definition: Rolling OLS r_t on r_t-1
- History/input: 60 returns; close
- Availability/reset: close of t; continuity run
- Target and expected shape: direction / regime; sign indicates persistence/reversion
- Existing overlap: return_autocorrelation_15
- Risks: leakage LOW; computation MEDIUM; numerical LOW
- Decision: REJECT - Formula-equivalent family already exists; variance ratio is the distinct selected adaptation.

### MLAT-H021 - `williams_r_14`

- Rationale: Price location in trailing high-low range may identify overbought/oversold state.
- Definition: Williams percent range
- History/input: 14 bars; HLC
- Availability/reset: close of t; continuity run
- Target and expected shape: direction; unknown
- Existing overlap: rolling_range_position_15
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: REJECT - Near formula-equivalent to existing rolling range position.

### MLAT-H022 - `intraday_seasonal_state`

- Rationale: Clock time may forecast volatility/activity.
- Definition: clock indicators
- History/input: none; timestamp
- Availability/reset: known at t; NY date/session
- Target and expected shape: expansion; session dependent
- Existing overlap: existing clock calendar family
- Risks: leakage LOW; computation LOW; numerical LOW
- Decision: REJECT - Existing clock features are already strong frozen expansion representatives.

### MLAT-H023 - `cross_sectional_risk_factor`

- Rationale: Cross-sectional factor exposure predicts relative returns.
- Definition: cross-sectional factor score
- History/input: multi-asset; universe
- Availability/reset: point in time; universe membership
- Target and expected shape: portfolio; relative
- Existing overlap: none
- Risks: leakage HIGH; computation HIGH; numerical HIGH
- Decision: DEFER_DIFFERENT_DATA - Single-instrument GC discovery has no cross-section.

### MLAT-H024 - `news_sentiment`

- Rationale: Text sentiment or semantic content may predict futures response.
- Definition: NLP feature
- History/input: publication history; text
- Availability/reset: publication timestamp; source dependent
- Target and expected shape: direction / risk; unknown
- Existing overlap: none
- Risks: leakage HIGH; computation HIGH; numerical HIGH
- Decision: DEFER_DIFFERENT_DATA - No point-in-time news/text source is in the current OHLCV dataset.

### MLAT-H025 - `pca_regime_state`

- Rationale: Low-dimensional components may summarize correlated states.
- Definition: Development-fitted PCA
- History/input: feature history; feature matrix
- Availability/reset: close of t; continuity inherited
- Target and expected shape: risk state; unknown
- Existing overlap: existing clustering/frozen set
- Risks: leakage MEDIUM; computation HIGH; numerical LOW
- Decision: LATER_PHASE - Requires frozen engineered inputs and stability analysis first.

### MLAT-H026 - `tree_or_boosted_feature_importance`

- Rationale: Nonlinear models may rank interactions.
- Definition: model importance
- History/input: training sample; feature matrix
- Availability/reset: after training; n/a
- Target and expected shape: model validation; n/a
- Existing overlap: none
- Risks: leakage HIGH; computation HIGH; numerical HIGH
- Decision: LATER_PHASE - Importance is not a feature and cannot prove economic value.

### MLAT-H027 - `deep_sequence_model`

- Rationale: Neural models may learn nonlinear temporal representations.
- Definition: NN/CNN/RNN/AE/GAN
- History/input: large sample; various
- Availability/reset: model dependent; model dependent
- Target and expected shape: later modelling; unknown
- Existing overlap: none
- Risks: leakage VERY HIGH; computation VERY HIGH; numerical HIGH
- Decision: LATER_PHASE - Simple features and linear benchmarks have not established directional information.

### MLAT-H028 - `reinforcement_learning_policy`

- Rationale: An agent may optimize sequential actions.
- Definition: MDP policy
- History/input: environment; state/actions/rewards
- Availability/reset: sequential; environment
- Target and expected shape: execution; n/a
- Existing overlap: none
- Risks: leakage VERY HIGH; computation VERY HIGH; numerical VERY HIGH
- Decision: REJECT_GOVERNANCE - No approved signal, reward, or validated simulator for this feature task.

### MLAT-H029 - `mgc_transfer_check`

- Rationale: A frozen GC feature may transfer to MGC.
- Definition: same frozen feature on MGC
- History/input: same as GC; MGC OHLCV
- Availability/reset: close of t; MGC continuity
- Target and expected shape: transfer validation; same expected shape
- Existing overlap: n/a
- Risks: leakage MEDIUM; computation HIGH; numerical LOW
- Decision: LATER_PHASE - MGC is locked until GC definitions and verdicts are frozen.
