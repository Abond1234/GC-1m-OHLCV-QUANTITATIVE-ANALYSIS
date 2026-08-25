# Tsay Feature Research v1: Frozen Book-to-Project Map

This table is the complete source disposition for this research line. The supplied
PDF is a traceability source only. No chapter may be reopened to generate another
feature, formula, threshold, model, or gate.

| Book chapter | Physical PDF pages | Project One disposition |
|---|---:|---|
| 1. Financial Time Series | 28-55 | Use log-return and distribution definitions as foundations. Generic returns, sample moments, skew, kurtosis, mixtures, and likelihood diagnostics are already represented or descriptive; do not add renamed copies as alpha. Jarque-Bera is descriptive only. ATR normalization is an existing Project One convention, not a Chapter 1 attribution. |
| 2. Linear Time Series | 56-135 | Implement only the frozen rolling AR(5) cumulative forecast `T01`. Use ACF, residual Ljung-Box, deterministic clock controls, and HAC-aware summaries only as diagnostics. No ARMA order search, long-memory estimation, seasonal-AR search, unit-root search, or imported book coefficient is permitted. |
| 3. Conditional Heteroskedasticity | 136-201 | Implement the ARCH-dependence diagnostic state `T02` as an explicitly labelled Project One adaptation. Reuse ATR, realized/range volatility, EWMA, and MLAT bipower/jump states as comparators. Plain GARCH previously failed implementation governance; Gaussian, Student-t, EGARCH, TGARCH, stochastic-volatility, and other likelihood variants are excluded from v1 by scope and cost, not claimed empirically rejected. |
| 4. Nonlinear Models | 202-257 | Implement only fixed two-regime TARX comparator `D6`, and only if `D3` earns its Development authorization gate. Threshold search, neural networks, Markov switching, nonparametric bandwidth searches, additive models, and generic nonlinear sweeps are excluded. |
| 5. High-Frequency Data | 258-313 | Implement Roll proxy `T03` and aggregated one-minute zero-close-change fraction `T04`. Raw formulas remain states; clock-bin effects are controlled in partial evidence and models. One-minute OHLCV does not identify duration, ordered-probit, quote, spread, depth, trade-direction, queue, or market-impact models. |
| 6. Continuous-Time Models | 314-351 | No new model. Diffusion, stochastic differential-equation, option-pricing, and continuous-time parameter estimation do not answer the one-minute feature question. Existing discrete jump proxies are comparators only. |
| 7. Extreme Values and Quantiles | 352-415 | Implement empirical expected-shortfall states `T05/T06`, extreme-cluster proxy `T09`, and regularized conditional-quantile comparator `D5/O4`. No block maxima, GEV/GPD/POT fit, threshold search, or full VaR system is permitted. |
| 8. Multivariate Time Series | 416-493 | Implement GC-only cross-correlation states `T07/T08` and small GC-only VAR comparator `D4`. VARMA identification searches and impulse-response selection are excluded. GC-MGC cointegration, threshold cointegration, and pairs trading are prohibited as alpha. |
| 9. PCA and Factor Models | 494-531 | No new PCA/factor experiment. FES already evaluated fold-local PCA/PLS profiles without an advancing model. BARRA/Fama-French, APCA, rotations, cross-sectional equity factors, and untimestamped or revised macro data are out of scope. |
| 10. Multivariate Volatility | 532-583 | No VEC, BEKK, CCC, DCC, Cholesky-order search, factor-volatility, multivariate-t, or multivariate stochastic-volatility model. Existing EWMA/rolling states are diagnostics only. MGC is not an alpha input. |
| 11. State-Space Models | 584-639 | Implement causal local-level Kalman volatility transformer `O5`. Filtering and prediction only; smoothing is leakage. Fit variance parameters inside each training fold. A time-varying GC-MGC beta is deferred to execution diagnostics. |
| 12. MCMC | 640-699 | No MCMC, Gibbs, Metropolis, FFBS, Bayesian imputation/outlier replacement, stochastic volatility, posterior regime path, or Markov-switching GARCH. Full-sample latent states and backward sampling are noncausal. |

Source identity: Ruey S. Tsay, *Analysis of Financial Time Series*, Third
Edition; 714 physical pages; required SHA-256
`B5630A4774C8C23F6BB8F624D05B536E49F09E828F4DB3B01D00B5B38A119E33`.
