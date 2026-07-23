# MLAT Formulas and Definitions

Book pages identify the motivating discussion. Entries explicitly marked
project-original do not imply that the displayed estimator appears in the book.

## MLAT-F001 - Log return

- Definition: `r_t = ln(C_t / C_{t-1})`
- Variables: C is close; prior close must be in the same continuity run.
- Required history: 1 prior bar
- Causal availability: close of t
- Parameters: none
- Scaling: dimensionless; output may be expressed in bps
- Stationarity assumptions: Local return distribution may change by regime.
- Boundary requirements: No gap, contract, segment, roll, or invalid boundary.
- Numerical edge cases: Non-positive prices and first-in-run are null.
- Book trace: Chapters 4 and 9, PDF 131 and 280-301
- GC adaptation: Use one-minute GC closes only.
- Status: IMPLEMENTED_PRIMITIVE

## MLAT-F002 - Bollinger z-score

- Definition: `z_t = (C_t - mean_20(C)_t) / std_20(C)_t`
- Variables: Trailing population standard deviation; project uses 20 complete one-minute bars.
- Required history: 20 bars
- Causal availability: close of t
- Parameters: window=20
- Scaling: unitless
- Stationarity assumptions: Local mean/std are descriptive, not stationary guarantees.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Zero standard deviation is null.
- Book trace: Chapter 4 PDF 131-133; Appendix 740-742
- GC adaptation: Use standardized location rather than equity trading-rule thresholds.
- Status: FROZEN_V1

## MLAT-F003 - Bollinger bandwidth

- Definition: `bandwidth_t = 4 * std_20(C)_t / mean_20(C)_t`
- Variables: Two-standard-deviation upper and lower bands imply total width 4 sigma.
- Required history: 20 bars
- Causal availability: close of t
- Parameters: window=20; band multiplier=2
- Scaling: fraction of price
- Stationarity assumptions: Only locally scaled.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Zero/non-positive mean is null.
- Book trace: Appendix PDF 740-742
- GC adaptation: Expansion-state hypothesis; compare with existing compression features.
- Status: FROZEN_V1

## MLAT-F004 - Cutler RSI

- Definition: `RSI_t = 100 * G_t / (G_t + L_t), where G_t=mean_14(max(dC,0)), L_t=mean_14(max(-dC,0))`
- Variables: This is the explicitly named simple-rolling Cutler variant, not TA-Lib Wilder recursion.
- Required history: 15 bars
- Causal availability: close of t
- Parameters: window=14 changes
- Scaling: 0 to 100
- Stationarity assumptions: Bounded transform does not create stationarity.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: If gains+losses are zero, value is null.
- Book trace: Chapter 4 PDF 132 (RSI example)
- GC adaptation: Avoid fixed 30/70 trade rules; test continuous rank relationship.
- Status: FROZEN_V1

## MLAT-F005 - Chaikin money flow adaptation

- Definition: `CMF_t = sum_20(V_i * ((2C_i-H_i-L_i)/(H_i-L_i))) / sum_20(V_i)`
- Variables: Rolling normalized adaptation of the Appendix A/D money-flow volume.
- Required history: 20 bars
- Causal availability: close of t
- Parameters: window=20
- Scaling: -1 to 1
- Stationarity assumptions: Volume comparability is local to GC.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Zero-range bars contribute neutral multiplier 0; zero total volume is null.
- Book trace: Appendix PDF 752-753
- GC adaptation: Treat as volume-price state; do not infer aggressor side.
- Status: FROZEN_V1

## MLAT-F006 - Amihud illiquidity adaptation

- Definition: `ILLIQ_t = 1e9 * mean_60(|r_i| / (C_i * V_i))`
- Variables: C*V is a within-GC notional-activity proxy; the fixed futures multiplier would only rescale ranks.
- Required history: 60 returns (61 bars)
- Causal availability: close of t
- Parameters: window=60
- Scaling: scaled inverse notional
- Stationarity assumptions: Comparable within GC, not across contracts/products without further scaling.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Zero volume or non-positive close is null.
- Book trace: Chapter 20 PDF 656; Appendix 752
- GC adaptation: One-hour adaptation of the book's rolling daily equity measure.
- Status: FROZEN_V1

## MLAT-F007 - Parkinson range volatility

- Definition: `sigma_P,t = 1e4 * sqrt(mean_30(ln(H_i/L_i)^2) / (4 ln 2))`
- Variables: Project-original OHLC estimator under the book's volatility-feature program; formula is not attributed to the book.
- Required history: 30 bars
- Causal availability: close of t
- Parameters: window=30
- Scaling: bps per one-minute interval
- Stationarity assumptions: Assumes a diffusion-like high-low process; microstructure and jumps violate it.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Non-positive prices are null.
- Book trace: Chapter 9 context PDF 297-301
- GC adaptation: Use as expansion/risk state and compare with ATR/RV.
- Status: FROZEN_V1

## MLAT-F008 - Rogers-Satchell range volatility

- Definition: `sigma_RS,t = 1e4 * sqrt(mean_30[ln(H/O)ln(H/C)+ln(L/O)ln(L/C)])`
- Variables: Project-original drift-robust OHLC estimator; formula is not attributed to the book.
- Required history: 30 bars
- Causal availability: close of t
- Parameters: window=30
- Scaling: bps per one-minute interval
- Stationarity assumptions: Requires valid OHLC and may be noisy at one-minute frequency.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Tiny negative round-off is clipped to zero; material negatives are null.
- Book trace: Chapter 9 context PDF 297-301
- GC adaptation: Use as expansion/risk state and compare with simpler anchors.
- Status: FROZEN_V1

## MLAT-F009 - Realized semivariance balance

- Definition: `SVB_t = (sum_60(r_i^2 1[r_i>0]) - sum_60(r_i^2 1[r_i<0])) / sum_60(r_i^2)`
- Variables: Project extension separating upside and downside realized variation.
- Required history: 60 returns (61 bars)
- Causal availability: close of t
- Parameters: window=60
- Scaling: -1 to 1
- Stationarity assumptions: Local sign asymmetry can change by session/regime.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Zero total variation is null.
- Book trace: Chapters 5 and 9 context PDF 169-178 and 297-301
- GC adaptation: Potential risk-state/directional-shape feature, not a signed-return claim.
- Status: FROZEN_V1

## MLAT-F010 - Bipower jump ratio

- Definition: `JR_t = max(RV_t-BV_t,0)/RV_t; RV=sum_60(r_i^2); BV=(pi/2)*(60/59)*sum_60(|r_i||r_{i-1}|)`
- Variables: Project extension distinguishing discontinuous variation from local continuous variation.
- Required history: 61 returns (62 bars)
- Causal availability: close of t
- Parameters: window=60
- Scaling: 0 to 1
- Stationarity assumptions: Bipower approximation can be distorted by microstructure and sparse bars.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Zero RV is null; negative numerator is floored at zero.
- Book trace: Chapter 9 context PDF 297-301
- GC adaptation: Risk/expansion-state candidate; exact incremental evidence required.
- Status: FROZEN_V1

## MLAT-F011 - Variance ratio

- Definition: `VR_t = Var_60(sum_5(r)_i) / (5 * Var_60(r_i))`
- Variables: Overlapping five-minute sums provide a local persistence/mean-reversion state.
- Required history: 64 returns (65 bars)
- Causal availability: close of t
- Parameters: outer=60; aggregation=5
- Scaling: unitless; random-walk reference near 1
- Stationarity assumptions: Overlapping observations bias naive inference; feature use is descriptive.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Zero one-minute variance is null.
- Book trace: Chapter 9 PDF 280-296
- GC adaptation: Compare with existing return autocorrelation and sign-change features.
- Status: FROZEN_V1

## MLAT-F012 - Return-sign entropy

- Definition: `H_t = -sum_{s in negative,zero,positive} p_s ln(p_s) / ln(3)`
- Variables: Rolling empirical probabilities use 60 causal return signs.
- Required history: 60 returns (61 bars)
- Causal availability: close of t
- Parameters: window=60
- Scaling: 0 to 1
- Stationarity assumptions: Discrete sign bins discard magnitude and depend on zero-tick frequency.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Terms with p=0 contribute zero.
- Book trace: Chapter 6 entropy context PDF 192
- GC adaptation: Project adaptation for choppiness/risk state; compare with sign-change rate.
- Status: FROZEN_V1

## MLAT-F013 - Volatility of volatility

- Definition: `VoV_t = std_60(RV15_i) / mean_60(RV15_i), RV15_i=1e4*sqrt(mean_15(r^2)_i)`
- Variables: Project extension measuring instability of a short realized-volatility state.
- Required history: 74 returns (75 bars)
- Causal availability: close of t
- Parameters: inner=15; outer=60
- Scaling: non-negative coefficient of variation
- Stationarity assumptions: Only a local state; denominator can approach zero.
- Boundary requirements: Complete single continuity run.
- Numerical edge cases: Zero/non-finite mean is null.
- Book trace: Chapter 9 context PDF 297-301
- GC adaptation: Risk-state candidate distinct from volatility level.
- Status: FROZEN_V1

## MLAT-F014 - GARCH(1,1) conditional variance

- Definition: `sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2`
- Variables: Gaussian diagnostic model with omega>0, alpha>=0, beta>=0, alpha+beta<1.
- Required history: expanding history
- Causal availability: before forecasted bar
- Parameters: Development fit schedule declared separately
- Scaling: return variance
- Stationarity assumptions: Conditional-variance specification and innovation distribution must be diagnosed.
- Boundary requirements: Recursion resets at every invalid continuity boundary.
- Numerical edge cases: Convergence, persistence, initialization, and scaling are explicit.
- Book trace: Chapter 9 PDF 297-301
- GC adaptation: GC-only audit; not part of the frozen v1 feature matrix.
- Status: AUDIT_ONLY
