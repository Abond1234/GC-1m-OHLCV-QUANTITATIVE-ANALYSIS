# Chapter 7 - Linear Models: From Risk Factors to Return Forecasts

**Assigned source range:** PDF pages 203-248  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; the frozen MLAT batch remains a pre-model feature experiment.

## Main argument

Linear regression and classification are useful first models because they are
interpretable, computationally efficient, and expose assumptions that remain
important for more complex learners. The chapter separates statistical
inference from prediction, derives ordinary least squares and logistic
regression, describes diagnostics and robust alternatives, and uses
regularization to trade a small increase in bias for lower variance
(PDF pp. 203-216, 222-225, 241-248).

For financial data, a strong in-sample fit or nominal coefficient significance
does not establish an out-of-sample signal. Residual dependence, non-normality,
multicollinearity, overlapping forward returns, scale-sensitive penalties, and
look-ahead in preprocessing can invalidate apparent evidence. The chapter's
equity examples therefore combine chronological cross-validation, train-only
scaling, daily rank information coefficients, and simple linear benchmarks
(PDF pp. 226-241).

The factor-model section also gives a distinct use of linear models: explaining
portfolio returns and estimating compensated risk exposures. Such risk
attribution is not the same task as predicting the next GC return
(PDF pp. 216-222).

## Concepts and definitions

- **Linear regression:** models a continuous response as an intercept plus a
  weighted sum of inputs and an error term (PDF pp. 204-207).
- **Inference versus prediction:** inference asks whether relationships and
  coefficient estimates are credible; prediction asks how well fitted values
  generalize to unseen observations (PDF pp. 203-205).
- **Ordinary least squares (OLS):** selects coefficients that minimize the
  residual sum of squares; under stronger distributional assumptions it also
  coincides with maximum-likelihood estimation (PDF pp. 205-208).
- **Gradient descent / stochastic gradient descent:** iterative optimization
  methods that use all or sampled observations to update parameters
  (PDF pp. 207-208, 215-216).
- **Gauss-Markov assumptions:** conditions concerning linear specification,
  exogeneity, full rank, and the error covariance under which OLS is the best
  linear unbiased estimator (PDF pp. 208-210).
- **Residual diagnostics:** normality, heteroskedasticity, serial correlation,
  and condition-number checks used to detect failures of inferential
  assumptions (PDF pp. 210-215).
- **Factor model:** represents asset or portfolio excess returns through
  exposures to common compensated sources of risk (PDF pp. 216-222).
- **Fama-MacBeth regression:** two-stage estimator that first obtains asset
  factor loadings through time-series regressions and then estimates period
  risk premia through cross-sectional regressions (PDF pp. 219-222).
- **Regularization / shrinkage:** adds a coefficient-size penalty to the loss
  to constrain variance and reduce overfitting (PDF pp. 222-225).
- **Ridge regression:** uses an L2 penalty and proportionally shrinks
  coefficients without normally setting them exactly to zero
  (PDF pp. 224, 236-238).
- **Lasso regression:** uses an L1 penalty and can set coefficients to zero,
  yielding continuous feature selection (PDF pp. 225, 238-241).
- **Logistic regression:** makes the log-odds of a class probability linear in
  the inputs and estimates coefficients by maximum likelihood
  (PDF pp. 241-248).
- **Information coefficient (IC):** rank correlation between a prediction or
  factor score and a future outcome; the chapter reports both daily and pooled
  variants (PDF pp. 230, 233-241).

## Formulas and notation

The linear model is

\[
y_i=\beta_0+\sum_{j=1}^{p}x_{ij}\beta_j+\epsilon_i,
\]

and OLS minimizes

\[
\operatorname{RSS}(\beta)
=\sum_{i=1}^{N}\left(y_i-\beta_0-\sum_{j=1}^{p}x_{ij}\beta_j\right)^2.
\]

Ridge regression adds \(S(\beta)=\sum_{j=1}^{p}\beta_j^2\), excluding the
intercept from the penalty. Rendered PDF p. 224 gives

\[
\hat{\beta}^{\mathrm{Ridge}}
=\left(X^\top X+\lambda I\right)^{-1}X^\top y.
\]

Lasso instead uses \(S(\beta)=\sum_j|\beta_j|\). Its non-differentiable L1
penalty has no analogous closed-form coefficient solution and can drive some
coefficients to zero (PDF p. 225).

For binary classification, rendered PDF p. 243 defines

\[
p(x)=\frac{e^{x\beta}}{1+e^{x\beta}},
\qquad
\log\!\left(\frac{p(x)}{1-p(x)}\right)=x\beta,
\]

and maximizes the Bernoulli log likelihood

\[
\sum_{i=1}^{N}
\left[y_i\log p(x_i,\beta)
+(1-y_i)\log\{1-p(x_i,\beta)\}\right].
\]

The book's feature examples include RSI, Bollinger Bands, ATR, MACD, lagged
returns, and forward-return targets (PDF pp. 228-230), but do not freeze the
exact Project 1 formulas. The MLAT formula registry is the authority for the
causal `bollinger_zscore_20`, `bollinger_bandwidth_20`, and
`cutler_rsi_14` adaptations.

## Assumptions

- Regressors are measured at the decision time and do not contain future
  outcomes or transforms fitted on the evaluation sample.
- Linear specification is adequate for the intended baseline, including any
  deliberately introduced basis functions or interactions
  (PDF pp. 204-205).
- Classical OLS inference additionally relies on exogeneity, nonsingular
  inputs, and an appropriate error covariance; normal errors support exact
  finite-sample tests (PDF pp. 208-210).
- Heteroskedasticity or dependence requires robust/clustered covariance,
  weighted/generalized least squares, or another valid design
  (PDF pp. 210-213).
- Scale-sensitive ridge, lasso, SGD, and regularized logistic models require
  standardization learned only from training data
  (PDF pp. 215-216, 224-225, 236-239, 246-247).
- Cross-validation must preserve chronology. The book's rolling time-series
  splits train only on earlier observations (PDF pp. 233-237).
- Overlapping targets induce residual autocorrelation and reduce effective
  evidence; row-level standard errors cannot be accepted uncritically
  (PDF pp. 232-233).
- A factor exposure is not automatically a predictive alpha, and a predictive
  coefficient is not automatically a causal effect.

## Procedures described

1. Define a linear response model, loss, and intended inference or prediction
   objective.
2. Fit with OLS, maximum likelihood, or iterative gradient methods and inspect
   coefficient and residual diagnostics.
3. For factor models, estimate asset loadings first and period risk premia
   second, then assess their time-series average.
4. Construct a point-in-time universe, causal features, lagged returns, and
   aligned forward targets.
5. Dummy-encode categories while dropping one level when an intercept is used.
6. Walk forward through chronological train/test windows.
7. Fit scalers and regularization parameters only inside each training fold.
8. Compare simple linear, ridge, lasso, and logistic predictions with
   out-of-sample metrics such as daily rank IC, RMSE, or AUC
   (PDF pp. 203-248).

Project 1 stops after the declared univariate and incremental feature gate. It
does not convert the chapter's prediction examples into a strategy, and no
multivariate model is authorized unless a small frozen shortlist survives
Validation.

## Feature and model examples

- Synthetic OLS and SGD regression with `statsmodels` and scikit-learn
  (PDF pp. 213-216).
- CAPM, Fama-French five-factor data, industry portfolios, and two-stage
  Fama-MacBeth estimation (PDF pp. 216-222).
- Daily equity liquidity selection using rolling dollar volume
  (PDF pp. 226-228).
- RSI, Bollinger Bands, ATR, MACD, geometric lagged returns, and time/sector
  indicators (PDF pp. 228-231).
- One-, five-, ten-, and 21-day forward returns produced by negative shifts
  (PDF p. 230); these are label constructions, never contemporaneous features.
- OLS residual diagnostics for overlapping five-day outcomes
  (PDF pp. 231-233).
- Walk-forward linear, ridge, and lasso return prediction with daily Spearman
  IC and RMSE (PDF pp. 233-241).
- Regularized logistic direction classification evaluated with AUC and IC
  (PDF pp. 241-248).

## Statistical, validation, and backtesting warnings

- OLS coefficient p-values are unreliable when residual covariance assumptions
  fail (PDF pp. 208-213).
- A high \(R^2\), favorable F-test, or low in-sample RMSE does not demonstrate
  out-of-sample predictive value.
- Multicollinearity makes coefficients unstable and can let large opposing
  coefficients cancel; regularization reduces but does not diagnose the
  economic redundancy (PDF pp. 214-225).
- The equity residuals are non-normal and serially correlated. Rendered PDF
  p. 233 explicitly connects autocorrelation through lag four to overlapping
  five-day labels.
- Feature standardization, winsorization, universe thresholds, and
  hyperparameter choice can leak if fitted using Validation observations.
- Selecting the best ridge/lasso/logistic penalty from many candidates uses
  validation information and must be governed as model selection.
- Pooled minute-level IC exaggerates independence; the Project 1 primary unit
  is the trading date, with date-block uncertainty.
- Lasso selection is sample-dependent when predictors are highly correlated;
  a zero coefficient does not prove that a feature contains no information.
- Nominal classification balance and AUC do not encode tick costs, asymmetric
  losses, calibration, or tradability.
- Factor portfolio examples use external equity data and do not justify
  applying equity risk premia to a single GC futures series.
- Turning predictions into portfolios or trading rules adds an additional
  layer of selection and backtest risk that is out of scope here.

## Implementation patterns worth preserving

- Keep feature construction, targets, splitters, metrics, and random seeds in
  a versioned experiment contract.
- Preserve a simple linear/anchor baseline before considering nonlinear
  models.
- Fit scalers and other learned preprocessing inside training partitions and
  serialize their parameters.
- Use chronological rolling/expanding splits and account explicitly for label
  overlap.
- Compute rank IC by date and display its distribution and time stability,
  rather than relying only on a pooled score.
- Diagnose residual distribution, dependence, and conditioning before
  interpreting coefficients.
- Treat regularization values as tuned hyperparameters and record every value
  tried.
- Compare candidate features incrementally against established anchors and
  reject near-duplicates before model escalation.
- Keep explanatory factor models, predictive models, and trading evaluation as
  separate research stages.

## Dated APIs and examples

The examples use older pandas, `pandas_datareader`, TA-Lib, `statsmodels`,
`linearmodels`, Alphalens, and scikit-learn APIs. Items such as
`SGDRegressor(loss='squared_loss')`, legacy `DataFrame.info(null_counts=...)`,
older `groupby().apply()` index behavior, HDFStore access, and the custom
`MultipleTimeSeriesCV` implementation require version-specific review. The
Quandl Wiki price dataset used in the example is discontinued. Project 1 does
not depend on those APIs or external datasets.

Several mathematical symbols and optimal hyperparameter values disappear from
plain-text extraction; equations on PDF pp. 224 and 243 were rendered and
checked directly. Missing extracted values are not reconstructed as facts.

## Project 1 relevance

This chapter directly supports three parts of MLAT v1:

- the bounded RSI and Bollinger-derived candidate features;
- daily rank IC as an interpretable screen, with chronology and overlap
  controls stronger than the book's example;
- a future multivariate linear residualization gate, but only after features
  pass univariate Validation and redundancy checks.

The chapter also reinforces the decision to compare every candidate with
`atr_20` and the 15-feature frozen expansion set. It does not authorize
directional model training, hyperparameter search, a classifier, or a strategy
backtest in the present notebook.

## One-minute GC adaptation

- Compute indicators on the full causal GC bar sequence and reset at explicit
  contract/segment/tradability boundaries, not at arbitrary session starts.
- Join feature values to eligible decision bars only after rolling history is
  computed; do not use forward-label columns as engineering inputs.
- Replace the book's cross-sectional daily equity IC with daily aggregation of
  intraday GC rank relationships within declared session/partition cells.
- Keep all six horizons on controlled complete samples and use horizon-thinned
  observations as a dependence sensitivity check.
- Fit Development quantile bins once and apply their edges unchanged to
  Validation.
- Use date-block bootstrap uncertainty and Benjamini-Hochberg correction for
  the frozen screen families.
- Reject candidates that are near-duplicates of the 85-feature matrix or fail
  incremental residual tests beyond `atr_20` and the frozen 15.
- Exclude MGC and the exposed historical Final partition from discovery.
- Preserve t-close availability and theoretical t+1-open entry semantics.

## Unsuitable or deferred ideas

- Fama-French and Fama-MacBeth factor estimation are unsuitable for the
  single-instrument intraday GC feature batch.
- TA-Lib defaults are not adopted without explicit causal formula and boundary
  tests.
- Broad lag expansion, unrestricted dummy variables, and automated lasso
  selection are rejected because they would enlarge the hypothesis family
  after seeing outcomes.
- Ridge, lasso, logistic regression, and Alphalens strategy analysis are
  deferred until the model-authorization gate is satisfied.
- Random row-level cross-validation and pooled minute-row inference are
  unsuitable.
- Classification thresholds, probability calibration, position sizing, and
  portfolio/backtest construction are outside this notebook.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Linear regression formulation, fitting, and assumptions | 203-210 |
| Diagnostics, robust alternatives, `statsmodels`, and SGD | 210-216 |
| CAPM/Fama-French factors and Fama-MacBeth estimation | 216-222 |
| Shrinkage, ridge, and lasso foundations | 222-225 |
| Universe, indicators, lagged returns, and target preparation | 226-231 |
| OLS stock-return inference and overlap diagnostics | 231-233 |
| Walk-forward linear regression and IC/RMSE | 233-236 |
| Ridge and lasso tuning/results | 236-241 |
| Logistic regression theory, inference, and price direction | 241-248 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 224, 233, and 243.
- **Confidence: high** for the model assumptions, diagnostics, shrinkage
  concepts, chronological evaluation, and overlap warning after sequentially
  reading the full assigned range.
- **Confidence: high** for the ridge equations on PDF p. 224, the residual
  distribution/autocorrelation figure and explanation on p. 233, and the
  logistic/log-likelihood equations on p. 243; these pages were visually
  inspected.
- **Confidence: medium** for several displayed equations and numerical
  hyperparameter optima whose symbols were omitted by text extraction. They
  are not needed for the frozen Project 1 contract.
- The book does not specify continuity-reset semantics for minute futures or a
  date-block bootstrap for dependent intraday IC.
- The exact TA-Lib indicator definitions/default warm-up rules are external to
  the chapter and are not treated as the Project 1 feature specification.
- The chapter's cross-validation does not fully specify purging/embargo for
  overlapping horizons; Chapter 6 supplies the stronger governance principle.
