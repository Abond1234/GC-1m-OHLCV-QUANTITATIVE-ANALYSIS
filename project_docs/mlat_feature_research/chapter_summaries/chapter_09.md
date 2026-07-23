# Chapter 9 - Time-Series Models for Volatility Forecasts and Statistical Arbitrage

**Assigned source range:** PDF pages 280-319

**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`

**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`

**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Financial observations arrive as one ordered realization of a stochastic process, so their lag structure, stationarity, changing variance, and cross-series dependence must be modeled explicitly. The chapter develops a progression from time-series diagnostics and transformations through ARIMA-family mean models, ARCH/GARCH variance models, vector autoregressions, cointegration tests, and a pairs-trading illustration (PDF pp. 280-319).

The practical message is conditional: linear time-series models are interpretable baselines when their assumptions and residual diagnostics hold. They are not mechanically valid because a fitted model converges or a historical spread appears mean-reverting. Model order, transformations, fitting windows, multiple testing, structural stability, and trading frictions all determine whether an apparent relationship can survive out of sample (PDF pp. 285-306, 307-319).

## Concepts and definitions

- **Time series and lag:** a time series is a sequence of usually equally spaced observations. A lag is the number of periods separating observations; lagged outcomes can become predictors (PDF pp. 281-282).
- **White noise:** an IID sequence with finite mean and variance. Gaussian white noise additionally has a normal distribution, zero mean, and constant variance (PDF p. 281).
- **Trend, seasonality, cycle, and residual:** systematic components may be separated additively or multiplicatively, leaving a residual for subsequent modeling (PDF pp. 282-283).
- **Rolling and expanding statistics:** sequential summaries such as moving averages, exponentially weighted averages, rolling covariance, and rolling correlation can expose changing state or smooth noise (PDF pp. 283-284).
- **ACF and PACF:** the autocorrelation function measures direct and indirect linear dependence by lag; partial autocorrelation removes the influence of intervening lags. Correlograms help propose AR or MA orders and diagnose residual dependence (PDF pp. 284, 291-294).
- **Stationarity:** strict stationarity requires the joint distribution to be time invariant. Covariance stationarity, used by many classical models, requires stable mean, variance, and autocorrelation. Volatility models deliberately relax constant variance (PDF p. 285).
- **Unit root and integration:** a unit root produces non-stationary, persistent behavior. A series that becomes stationary after differencing \(d\) times is integrated of order \(d\), denoted \(I(d)\) (PDF pp. 285-288).
- **ADF test:** tests the null hypothesis of a unit root against stationarity using a regression in first differences with deterministic terms and lagged differences (PDF p. 287).
- **AR, MA, ARMA, and ARIMA:** AR terms use lagged series values; MA terms use lagged innovations; ARMA combines both for stationary data; ARIMA adds differencing. ARMAX adds exogenous inputs and SARIMAX adds seasonal orders (PDF pp. 290-296).
- **Ljung-Box Q test:** tests whether a set of residual autocorrelations is jointly zero. Residual dependence suggests omitted time-series structure (PDF pp. 291-292).
- **Conditional heteroskedasticity:** ARCH models variance as a function of lagged squared errors. GARCH adds lagged conditional variance, typically with fewer parameters than a high-order ARCH model (PDF pp. 297-301).
- **HAR/HARCH and EGARCH:** heterogeneous models combine multiple time scales; EGARCH permits asymmetric responses to positive and negative shocks (PDF pp. 298-299).
- **VAR and VECM:** VAR gives every series equations containing lags of every series. VECM is appropriate when non-stationary series share stationary linear combinations (PDF pp. 301-306).
- **Cointegration:** two or more individually integrated series are cointegrated when a linear combination has a lower integration order, commonly stationarity and mean reversion. Cointegration is not the same as correlation (PDF pp. 307-309).
- **Engle-Granger and Johansen tests:** Engle-Granger tests pairwise regression residuals for a unit root; Johansen tests the rank restrictions of a multivariate error-correction representation (PDF pp. 308-309).
- **Statistical arbitrage and pairs trading:** formation selects a stable relationship; trading enters on divergence and exits on convergence while attempting to control market exposure (PDF pp. 309-319).

## Formulas and notation

Only equations recoverable from the supplied PDF are transcribed. Displayed formulas whose symbols were lost in text extraction are described but not reconstructed.

- The visually verified augmented Dickey-Fuller regression is:

  \[
  \Delta y_t
  =
  \alpha+\beta t+\gamma y_{t-1}
  +\delta_1\Delta y_{t-1}
  +\cdots+
  \delta_{p-1}\Delta y_{t-p+1}
  +\epsilon_t
  \]

  The unit-root null corresponds to \(\gamma=0\); deterministic-term choices distinguish a random walk from a random walk with drift (PDF p. 287).

- The visually verified GARCH(\(p,q\)) specification for log return \(r_t\), mean \(\mu\), innovation \(\epsilon_t=r_t-\mu\), conditional scale \(\sigma_t\), and IID standardized shock \(z_t\) is:

  \[
  \epsilon_t=\sigma_t z_t,
  \qquad
  \sigma_t^2
  =
  \omega+
  \sum_{i=1}^{p}\alpha_i\epsilon_{t-i}^2
  +
  \sum_{j=1}^{q}\beta_j\sigma_{t-j}^2
  \]

  (PDF p. 298).

- The chapter estimates a spread half-life from a regression of the spread change on its lagged level:

  \[
  \text{half-life}\approx-\frac{\ln 2}{\hat\beta}
  \]

  where \(\hat\beta\) is the lagged-level coefficient. The implementation floors the rounded result at one period (PDF p. 315).

- A rolling spread z-score is the spread minus its rolling mean, divided by its rolling standard deviation. The book uses a window of up to twice the estimated half-life (PDF pp. 315-316).

- AR(\(p\)), MA(\(q\)), ARIMA, VAR, and error-correction equations appear on PDF pp. 286, 291-294, 302, and 308. Several subscripts and operators are absent from extraction, so no exact transcription is attempted here.

## Assumptions

- Sampling is ordered and sufficiently regular for the declared lag to have consistent meaning.
- Classical ARIMA and VAR inference assumes the modeled series is stationary after any declared transformation (PDF pp. 285-296, 302-303).
- Innovations and standardized residuals should be approximately serially uncorrelated; many specifications also impose distributional and constant-parameter assumptions.
- Model orders selected from ACF/PACF, AIC, BIC, or cross-validation remain stable enough to forecast later periods.
- GARCH assumes past squared innovations and conditional variances summarize forecastable volatility dynamics. Standard ARCH/GARCH responds symmetrically to positive and negative shocks (PDF pp. 297-299).
- Cointegration requires a long-run relationship that persists beyond the formation window. Historical test significance alone does not guarantee future convergence (PDF pp. 307-319).
- The pairs example assumes both legs can be traded at modeled prices with adequate liquidity, borrow availability, and acceptable transaction costs.

## Procedures described by the chapter

### Time-series preparation and model diagnosis

1. Plot and decompose the series into trend, seasonality, and residual components.
2. Apply economically justified transformations such as logs, deflation, detrending, ordinary differencing, or seasonal differencing.
3. inspect rolling statistics, Q-Q plots, ACF, and PACF.
4. Use an ADF test to evaluate remaining unit-root non-stationarity.
5. Select a parsimonious lag structure using diagnostics, information criteria, and preferably chronological out-of-sample prediction.
6. Fit the model and verify that residuals resemble white noise using plots and the Ljung-Box test.
7. Reverse transformations correctly when forecasts must be expressed in original units (PDF pp. 282-306).

### Volatility forecasting

1. Specify a mean model using ARMA structure only where return autocorrelation warrants it.
2. Test squared residuals for remaining ARCH effects with ACF/PACF and formal tests.
3. Predeclare a small conditional-variance family and lag grid.
4. Jointly estimate mean and variance equations.
5. Produce genuinely one-step-ahead forecasts from rolling or expanding training windows.
6. Compare forecast loss against simpler volatility baselines and inspect standardized residuals (PDF pp. 297-301).

### Cointegration and pairs trading

1. Define the tradable universe and formation window.
2. Use inexpensive drift, spread-volatility, or correlation heuristics only as a screening layer.
3. Test surviving candidates with Engle-Granger and/or Johansen procedures, accounting for multiplicity.
4. Estimate a hedge ratio and spread using only information available in the formation or rolling window.
5. Estimate mean-reversion speed and define entry, exit, stop, and reassessment rules before the trading sample.
6. Simulate both legs with exposure, overlap, concentration, costs, and failure-to-converge risk represented explicitly (PDF pp. 307-319).

## Feature and model examples

- Rolling mean, exponentially weighted mean, rolling variance, ACF/PACF coefficients, and differenced or seasonally differenced series (PDF pp. 282-290).
- AR lag coefficients, MA error corrections, seasonal AR/MA terms, and exogenous covariates (PDF pp. 290-296).
- Squared residual lags, conditional variance, standardized residuals, GARCH forecasts, and volatility-regime state (PDF pp. 297-301).
- HAR-style daily, weekly, and monthly volatility components as heterogeneous time-scale features (PDF p. 298).
- VAR cross-series lags, impulse responses, forecast-error variance decomposition, and Granger-predictive relationships (PDF pp. 301-306).
- Spread drift, spread volatility, price correlation, return correlation, cointegration statistics, rolling hedge ratios, mean-reversion half-life, and spread z-scores (PDF pp. 310-318).
- The macro example searches ordinary and seasonal AR/MA orders using a rolling ten-year window and one-step RMSE (PDF pp. 295-296).
- The volatility example searches GARCH orders 1-4 on rolling ten-year windows and reports GARCH(2,2) as the parsimonious RMSE winner for the illustrated NASDAQ sample (PDF pp. 299-301).

## Statistical, validation, and backtesting warnings

- Non-stationary levels can create spurious regression significance. Transformations and unit-root diagnostics must precede inference (PDF pp. 285-288, 307).
- ACF/PACF are model-design hints, not proof. AR and MA effects can cancel, and adding many terms raises overfitting and convergence risk (PDF pp. 291-294).
- AIC and BIC are in-sample criteria. Predictive claims require chronological out-of-sample evaluation (PDF pp. 293-296, 303).
- Volatility clustering, fat tails, and leverage effects can invalidate Gaussian, constant-variance, or symmetric-shock assumptions (PDF pp. 288-301, 305-306).
- The book winsorizes returns before its rolling GARCH comparison. The displayed code computes quantiles from the full series, which would be an unacceptable future-distribution dependency in a governed point-in-time experiment (PDF pp. 299-300).
- GARCH convergence and statistically significant parameters do not establish forecast superiority over ATR, realized volatility, or a seasonal baseline.
- Pair searches create a large multiple-testing problem. In the example, two tests agree on only 366 of more than 46,000 period-pair evaluations (PDF pp. 310-313).
- Class imbalance makes screening metrics deceptive: an AUC near 0.82 still produces many thousands of false positives in the chapter's illustration (PDF p. 313).
- Hedge ratios, cointegration rank, and mean-reversion speed can drift or break after formation.
- The chapter explicitly acknowledges lookahead bias and ignored transaction costs in the pairs backtest. Its reported Sharpe 0.75 and Sortino 1.05 are therefore demonstrations, not credible evidence of net alpha (PDF p. 318).
- Optimizing universe, formation window, test threshold, z-score threshold, stop, and exit jointly would constitute extensive data snooping (PDF pp. 310-319).

## Implementation patterns worth preserving

- Separate transformation, fitting, forecasting, residual diagnostics, and scoring.
- Store the exact ordered timestamps and continuity-run identity used by each model fit.
- Use rolling or expanding fits that stop before the forecast timestamp.
- Persist fitted parameters, convergence status, objective value, residual tests, and forecast timestamps.
- Compare candidates on identical forecast rows and an identical loss function.
- Use a cheap deterministic screen before expensive pairwise tests, while measuring the screen's false-negative and false-positive costs.
- Represent a multi-leg trade as a first-class object containing both sizes, hedge ratio, entry/exit timestamps, and realized spread.
- Keep forecast evaluation separate from the strategy simulator and cost model.

## Dated APIs and examples

The chapter reflects a 2020 Python ecosystem and should not be copied without checking current documentation.

- `statsmodels` version 0.11 is discussed explicitly; current `ARIMA`, `SARIMAX`, `VARMAX`, diagnostic, and result APIs differ from older examples (PDF pp. 293-306).
- `pandas-datareader` and FRED symbols are historical data-access examples, not Project 1 dependencies.
- The `arch` package remains relevant, but current result, forecast, convergence, distribution, and rescaling behavior must be verified (PDF pp. 298-301).
- `pykalman` is used for smoothing and dynamic hedge ratios; its maintenance status and initialization behavior require review (PDF pp. 314-315).
- The example uses removed or deprecated pandas patterns such as `Series.append` and `fillna(method='bfill')` (PDF pp. 316-317).
- Backtrader and Stooq are illustrative dependencies; current futures execution and data requirements differ materially (PDF pp. 311, 317-318).

## Project 1 relevance

- **High direct relevance:** volatility clustering, squared-residual diagnostics, GARCH/HAR forecasts, ACF/PACF state, stationarity checks, and strict chronological forecast evaluation.
- Project 1 already has causal GC one-minute bars, stable observation IDs, six fixed horizons, Development/Validation partitions, ATR features, and evidence that expansion information is stronger than direction. A volatility model should therefore be treated first as expansion or risk-state information.
- `atr_20`, realized-volatility features, and the frozen expansion model are mandatory baselines. Complexity is justified only by incremental Validation performance, calibration, and operational usefulness.
- Existing notebook-local GARCH results should not be inherited as established evidence. The model must be rebuilt under continuity, GC-only fitting, point-in-time initialization, convergence, residual, and save/reload controls.
- AR or ARMAX terms may supply bounded state features, but weak one-minute return autocorrelation argues for a very small predeclared order set.
- Cointegration and cross-sectional pairs trading are not part of the current GC discovery experiment.

## One-minute GC adaptation

Potential bounded hypotheses:

1. A continuity-safe GC GARCH or EGARCH one-step conditional-variance forecast adds stable information about future 60- or 180-minute movement beyond `atr_20`.
2. A heterogeneous volatility feature combining short, medium, and long causal realized-variation windows improves expansion calibration beyond a single ATR anchor.
3. A small predeclared rolling AR state or innovation statistic describes mean-reversion versus persistence, without presuming directional edge.

Adaptation requirements:

- Build log returns only within valid one-minute continuity runs. Reset at timestamp gaps, selected-contract, instrument, segment, rollover, tradability, or liquidity boundaries.
- Use only GC for definitions, parameter selection, thresholds, and advancement. MGC must not influence discovery.
- Fit all parameters on Development data under a declared initial window and refit schedule. Apply frozen or historically available parameters forward.
- Map forecasts to completed decision bar \(t\); no entry-bar or future-bar information may enter the feature.
- Normalize or stratify intraday variance by causally fitted New York time-of-day effects so GARCH does not merely rediscover the clock.
- Test convergence, persistence, positivity/stationarity constraints, standardized residual autocorrelation, squared-residual autocorrelation, remaining ARCH effects, distributional misspecification, and forecast calibration.
- Compare against ATR, simple exponentially weighted variance, and realized-volatility baselines on identical Development and Validation observations.
- Evaluate risk and expansion targets separately from signed returns. Do not report Sharpe before a sequential policy exists.

## Unsuitable or deferred ideas

- Engle-Granger/Johansen screening within the single-instrument GC feature batch.
- Using MGC as the second leg during GC discovery; this violates the project's transfer-isolation rule.
- Copying the equity/ETF pairs strategy, quarterly formation schedule, or two-standard-deviation thresholds into intraday futures.
- Full-sample winsorization, decomposition, scaling, or lag selection.
- Large ARIMA/GARCH order grids selected repeatedly on Validation.
- Centered moving averages or noncausal smoothing.
- Calling Granger predictiveness economic causality.
- Treating a high GARCH persistence estimate as evidence of incremental forecast value.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Time-series setup, decomposition, rolling statistics, ACF/PACF | 280-284 |
| Stationarity, transformations, unit roots, and ADF testing | 285-290 |
| AR, MA, ARIMA, ARMAX, SARIMAX, and diagnostics | 290-296 |
| ARCH/GARCH volatility forecasting | 297-301 |
| VAR, VECM, and macro forecast example | 301-306 |
| Cointegration concepts and tests | 307-309 |
| Pair selection, screening, and multiplicity | 309-313 |
| Kalman hedge ratios, half-life, spread rules, and backtest | 314-319 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 287, 298, 312, 318.
- **Confidence: high** for the chapter structure, definitions, workflows, examples, reported metrics, and warnings; all assigned pages 280-319 were read sequentially.
- **Confidence: high** for the ADF and GARCH equations; PDF pages **287 and 298** were visually inspected.
- **Confidence: high** for the displayed cointegration-screen implementation and the warning attached to the strategy performance; PDF pages **312 and 318** were visually inspected.
- **Confidence: medium** for ARIMA, VAR, and Johansen notation because text extraction dropped several operators and subscripts. Those equations were not guessed.
- The chapter does not specify a governed treatment of optimizer failures across a GARCH order search or uncertainty in conditional-variance estimates.
- The example does not provide a costed, lookahead-free pairs result or multiplicity-adjusted end-to-end backtest.
- Project 1 applicability remains hypothetical until a separately registered MLAT implementation passes causal construction and Development/Validation tests.
