# MLAT Methodological Warnings

## Book-derived warnings

| Warning | Source | MLAT consequence |
|---|---|---|
| Data quality and point-in-time integrity dominate model sophistication. | Chapters 1-3 and 23, PDF 42-114 and 724-734 | Verify physical artifacts and exclude legacy forward columns before engineering. |
| Lag and target alignment can create look-ahead bias. | Chapter 4, PDF 130-132 | Features end at completed `t`; labels and entry begin at `t+1`. |
| Technical indicators are hypotheses, not validated trading rules. | Chapter 4 and Appendix, PDF 131-152 and 735-764 | No 30/70 RSI or band-cross signal is assumed. |
| Bias-variance and repeated model selection can overfit noise. | Chapter 6, PDF 192-201 | Use a bounded frozen batch, BH correction, chronological Validation, and no broad tuning. |
| Ordinary shuffled CV is invalid for dependent financial samples. | Chapter 6, PDF 195-200 | Use chronological partitions, date evidence units, thinning, and purging/embargo for models. |
| Backtests require explicit timestamp, calendar, order, position, and cost mechanics. | Chapter 8, PDF 249-279 | Feature evidence does not authorize a strategy or Sharpe calculation. |
| ARIMA-type mean models assume stable structure; changing variance needs separate modelling. | Chapter 9, PDF 280-301 | Diagnose stationarity, volatility clustering, residuals, and forecast calibration. |
| A fitted GARCH model must leave standardized residuals close to white noise. | Chapter 9, PDF 297-301 | Record convergence, persistence, Ljung-Box/ARCH diagnostics, scaling, and anchor comparison. |
| Feature importance reflects model behavior, not causal or economic value. | Chapters 11-12, PDF 350-428 | No nonlinear model or importance plot is used to prove a feature edge. |
| Complex deep/RL systems require data, diagnostics, and valid environments. | Chapters 17-22, PDF 533-723 | These methods remain outside v1. |
| Backtest overfitting remains a central failure mode. | Chapter 23, PDF 724-734 | The exposed historical Final test is not reused for selection. |

## Project-specific additions

- GC alone defines every v1 feature, parameter, threshold, and verdict. MGC is
  inaccessible to discovery code.
- Rolling state resets on missing minutes, contract/product/instrument/segment
  changes, roll windows, invalid tradability, and low-liquidity warnings.
- Development-fitted buckets and transformations are applied unchanged to
  Validation.
- The historical Final-test partition may be counted to prove exclusion but
  cannot enter an outcome screen.
- Overlapping minute labels make rows invalid as independent evidence units.
  Daily rank IC, date-block intervals, and horizon thinning are mandatory.
- A statistically detectable direction effect below the frozen two-tick
  quintile-spread floor is economically trivial.
- Expansion and risk-state features need not predict signed returns, but must
  add information beyond `atr_20` and the frozen expansion set.
- Global clipping, scaling, quantiles, Kalman noise fits, wavelet thresholds,
  and GARCH parameters are leakage channels unless fitted under a declared
  chronological protocol.
- Missingness is information about warm-up/boundaries and is not silently
  backfilled.
- Final reports must retain negative and redundant results; feature count is
  never a completion metric.
