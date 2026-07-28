# Chapter 10 - Bayesian ML: Dynamic Sharpe Ratios and Pairs Trading

**Assigned source range:** PDF pages 320-349

**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`

**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`

**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Bayesian machine learning treats unknown parameters as probability distributions and updates prior beliefs with observed evidence. This produces posterior distributions rather than only point estimates, making uncertainty explicit and allowing sequential updating as information arrives. The chapter develops that perspective from Bayes' theorem through exact and approximate inference, implements a probabilistic-programming workflow in PyMC3, and illustrates recession classification, Bayesian Sharpe-ratio comparison, time-varying pairs hedge ratios, and latent stochastic volatility (PDF pp. 320-349).

For trading research, the central benefit is not that Bayesian notation makes a model more reliable. It is that a declared likelihood, prior, and update rule expose assumptions and return uncertainty that can be propagated into decisions. A posterior remains conditional on model specification, data timing, prior choices, and sampler quality. Bayesian inference does not repair leakage, nonstationarity, a random time-series split, omitted costs, or an unstable economic relationship (PDF pp. 323-340, 341-349).

## Concepts and definitions

- **Frequentist versus Bayesian probability:** the frequentist view interprets probability as long-run frequency and parameters as fixed but unknown. The Bayesian view interprets probability as degree of belief, treats observed data as given, and models parameters as random variables with inferable distributions (PDF pp. 320-322).
- **Prior, likelihood, evidence, and posterior:** the prior expresses beliefs about parameter values; the likelihood scores the observed data under each parameter value; the evidence is the marginal probability of the observed data; the posterior combines all three (PDF pp. 321-323).
- **Latent parameter:** an unobserved random variable inferred from observed data, such as an expected return, regression coefficient, volatility state, covariance matrix, or model weight (PDF pp. 321-322).
- **Maximum likelihood estimation (MLE):** selects the parameter value that maximizes the observed-data likelihood. **Maximum a posteriori (MAP)** estimation selects the posterior mode and therefore also reflects the prior (PDF p. 323).
- **Objective, subjective, and empirical priors:** an objective or flat prior attempts to minimize prior influence over a relevant support; a subjective prior represents external beliefs; an empirical prior estimates prior parameters from historical data (PDF pp. 323-324).
- **Prior as regularization:** the prior restricts posterior support and pulls estimates toward values considered plausible, especially with little data. Its influence normally declines as relevant evidence accumulates (PDF p. 324).
- **Conjugate prior:** a prior-likelihood pairing whose posterior belongs to the same distribution family, allowing a closed-form update and reuse of one posterior as the next prior (PDF pp. 324-325).
- **Beta-binomial updating:** binary Bernoulli outcomes have a binomial count likelihood, and a beta prior yields a beta posterior. The example updates the estimated probability of a positive S&P 500 day (PDF pp. 324-325).
- **Approximate inference:** when the evidence integral or state summation is intractable, inference approximates the posterior using stochastic sampling or deterministic optimization (PDF pp. 325-329).
- **Markov chain Monte Carlo (MCMC):** constructs a memoryless transition process whose retained states approximate the posterior. Early samples influenced by initialization are treated as burn-in; retained samples form the trace (PDF pp. 326-328).
- **Gibbs, Metropolis-Hastings, HMC, and NUTS:** Gibbs samples one conditional dimension at a time; Metropolis-Hastings accepts or rejects proposals using a posterior-proportional ratio; Hamiltonian Monte Carlo uses gradients and momentum; the No-U-Turn Sampler adaptively tunes HMC trajectories (PDF pp. 327-328).
- **Variational inference (VI):** optimizes a tractable distribution to be close to the posterior, typically by minimizing Kullback-Leibler divergence. Automatic Differentiation Variational Inference (ADVI) automates much of the model-specific optimization (PDF pp. 328-329).
- **Probabilistic programming:** represents observed and unobserved random variables, priors, likelihoods, and deterministic transformations as a program, leaving inference machinery to a system such as the chapter's PyMC3 (PDF pp. 329-333).
- **Credible interval and highest posterior density (HPD) interval:** a credible interval directly summarizes posterior probability over parameter values. An HPD interval is the minimum-width region containing a selected posterior mass under the chapter's terminology (PDF pp. 335-337).
- **Effective sample size and R-hat:** effective sample size discounts redundant autocorrelated draws. The Gelman-Rubin R-hat statistic compares between-chain with within-chain variance and should be near one after convergence (PDF p. 337).
- **Posterior predictive check (PPC):** draws replicated observations from the fitted model using posterior parameter samples, permitting a comparison between model-implied and observed data (PDF p. 338).
- **Bayesian Sharpe ratio:** models the return distribution and its parameters jointly so that the Sharpe ratio, its components, and differences between return series have posterior distributions (PDF pp. 341-343).
- **Dynamic Bayesian regression:** lets regression coefficients evolve as latent stochastic processes. The pairs example uses Gaussian random walks for both intercept and slope (PDF pp. 343-346).
- **Bayesian stochastic volatility:** represents time-varying volatility as a latent random walk and models fat-tailed returns conditionally on that state (PDF pp. 346-349).

## Formulas and notation

Only equations recoverable from the supplied PDF or its displayed code are transcribed.

- Bayes' theorem for parameter vector or hypothesis \(\theta\) and observed data \(x\) is:

  \[
  p(\theta\mid x)
  =
  \frac{p(x\mid\theta)\,p(\theta)}{p(x)}
  \]

  Here \(p(\theta)\) is the prior, \(p(x\mid\theta)\) the likelihood, \(p(x)\) the evidence, and \(p(\theta\mid x)\) the posterior (PDF p. 322).

- For a beta prior \(\operatorname{Beta}(a,b)\), \(u\) observed positive outcomes, and \(d\) negative outcomes, the code evaluates the posterior as:

  \[
  p\mid x \sim \operatorname{Beta}(a+u,\ b+d)
  \]

  The illustration begins with \(a=b=1\), a uniform prior on \([0,1]\) (PDF pp. 324-325).

- The visually verified logistic model is:

  \[
  p(y_i=1\mid\boldsymbol{\beta})
  =
  \sigma\!\left(\beta_0+\beta_1x_{i1}+\cdots+\beta_kx_{ik}\right),
  \qquad
  \sigma(t)=\frac{1}{1+e^{-t}}
  \]

  with a Bernoulli likelihood for the binary outcome (PDF pp. 331-332).

- The single-series Bayesian Sharpe example models daily returns with a Student \(t\) distribution and defines the deterministic annualized statistic, with a zero risk-free rate, as:

  \[
  SR=\sqrt{252}\frac{\mu}{\sqrt{\operatorname{Var}(r)}}
  \]

  The code places a normal prior on the mean, a uniform prior on the standard deviation, and a shifted exponential prior on degrees of freedom (PDF pp. 341-342).

- The dynamic pairs regression lets intercept and slope follow independent Gaussian random walks:

  \[
  \alpha_t\sim\mathcal{N}(\alpha_{t-1},\sigma_\alpha^2),
  \qquad
  \beta_t\sim\mathcal{N}(\beta_{t-1},\sigma_\beta^2)
  \]

  and uses a conditional mean of \(\alpha_t+\beta_t x_t\) for the second asset (PDF pp. 344-345).

- The visually verified stochastic-volatility display is:

  \[
  \nu\sim\operatorname{Exponential}(0.1),
  \qquad
  \sigma\sim\operatorname{Exponential}(50),
  \]

  \[
  s_i\sim\mathcal{N}(s_{i-1},\sigma^{-2}),
  \qquad
  \log(r_i)\sim t\!\left(\nu,0,\exp(-2s_i)\right)
  \]

  The accompanying code passes `lam=exp(-2*s)` to `StudentT`, so the third term is the PyMC3 precision parameterization, not an ordinary standard deviation. The source first constructs `log_returns = np.log(prices).diff()`, despite the display writing \(\log(r_i)\) (PDF p. 347).

## Assumptions

- Prior distributions and their support are defensible for the parameter and are tested for sensitivity (PDF pp. 323-325).
- The likelihood adequately represents the observed outcome distribution, tails, dependence, and conditional structure.
- Posterior samplers have mixed across relevant modes, forgotten initialization, and produced enough effective draws for the reported quantity (PDF pp. 326-338).
- The recession indicators are available at the timestamp implied by the model and are aligned to the 12-month target without revised-data leakage (PDF pp. 330-340).
- Logistic coefficients are stable enough for the intended forecast period; quasi-separation and feature scale do not make posterior computation or interpretation pathological (PDF pp. 333-337).
- Student \(t\) returns are an adequate model for Bayesian Sharpe comparison and the analyzed returns are sufficiently stationary and comparable over the selected interval (PDF pp. 341-343).
- A Gaussian random walk is a reasonable evolution law for the regression intercept, hedge ratio, or latent log-volatility (PDF pp. 343-348).
- Posterior uncertainty is not decision uncertainty unless execution, costs, sizing, and regime failure are modeled separately.

## Procedures described by the chapter

### General Bayesian workflow

1. Define priors for each latent parameter and a likelihood for the observed data.
2. Obtain MAP point estimates, MCMC samples, or a variational approximation depending on model complexity and accuracy requirements.
3. Inspect multiple chains, traces, marginal distributions, effective sample sizes, R-hat, energy behavior, and divergences.
4. Run posterior predictive checks to test whether replicated data resemble observed data.
5. Replace training inputs with chronologically valid prediction inputs and draw from the posterior predictive distribution.
6. Report posterior summaries and uncertainty, not only a preferred point estimate (PDF pp. 329-340).

### Recession-classification example

1. Source NBER recession labels and four FRED indicators: the 10-year minus 3-month Treasury spread, consumer sentiment, the National Financial Conditions Index, and its nonfinancial-leverage subindex.
2. Resample to monthly observations, shift the target to ask whether the economy will be in recession 12 months later, and standardize predictors.
3. Assign normal priors to logistic coefficients and a Bernoulli likelihood to the recession outcome.
4. Compare MAP, MCMC/NUTS, and ADVI estimates.
5. Diagnose convergence and generate posterior-predictive probabilities (PDF pp. 330-340).

The book explicitly uses a stratified random train/test split and says it is ignoring the time-series nature for exposition. That procedure must not be transferred to a trading experiment (PDF pp. 338-340).

### Bayesian performance comparison

1. Specify a fat-tailed return likelihood and priors for mean, standard deviation, and degrees of freedom.
2. Derive the Sharpe ratio as a deterministic function of every posterior draw.
3. For two return series, model both parameter sets and calculate posterior distributions of differences in means, volatility, Sharpe ratios, and standardized effect size.
4. Calculate the posterior probability that one series exceeds the other rather than comparing two point estimates only (PDF pp. 341-343).

### Dynamic hedge ratio and stochastic volatility

1. Normalize aligned series and declare random-walk priors for time-varying coefficients.
2. Fit a conditional observation model and retain the coefficient posterior by timestamp.
3. For volatility, model log returns with a Student \(t\) likelihood and a latent random-walk state controlling precision.
4. Tune NUTS, inspect convergence, and compare latent volatility estimates with observed return clusters.
5. Evaluate predictive accuracy chronologically; the chapter explicitly describes its volatility plot as in-sample fit (PDF pp. 343-349).

## Feature and model examples

- Posterior mean, median, mode, credible bounds, interval width, tail probability, and probability that a parameter exceeds an economically meaningful threshold.
- Beta-binomial posterior probability of an up day and the contraction of uncertainty as observations accumulate (PDF pp. 324-325).
- Bayesian logistic posterior coefficients and posterior-predictive recession probabilities (PDF pp. 330-340).
- MCMC diagnostics: R-hat, effective sample size, trace autocorrelation, divergences, chain agreement, and energy diagnostics (PDF pp. 334-338).
- Posterior Sharpe-ratio distribution, difference in Sharpe ratios, mean-return difference, volatility difference, and standardized effect size (PDF pp. 341-343).
- Time-varying intercept, slope, hedge ratio, coefficient uncertainty, and coefficient innovation scale from rolling Bayesian regression (PDF pp. 343-346).
- Latent volatility level, volatility uncertainty, degrees of freedom, and posterior predictive tail probability (PDF pp. 346-349).

## Statistical, validation, and backtesting warnings

- A flat prior is not automatically neutral: support, scale, and parameterization can materially influence weak-data posteriors. Prior-predictive and sensitivity checks are required (PDF pp. 323-325).
- MAP discards most posterior uncertainty and can be misleading for skewed, multimodal, or weakly identified posteriors (PDF pp. 323, 325, 340).
- MCMC draw count is not effective sample size. Serial correlation, failed mixing, unvisited modes, and divergent transitions can leave a large trace with little usable information (PDF pp. 326-338).
- R-hat near one is necessary but not sufficient. It does not validate the likelihood, causal timing, prior, predictive calibration, or economic use.
- VI/ADVI optimizes within a restricted distribution family and is not guaranteed to match the true posterior. The chapter itself reports slightly less accurate results (PDF pp. 328-329, 335-336).
- The recession example contains quasi-separation: the yield curve perfectly predicts about 17 percent of observations, leaving MLE coefficients poorly defined (PDF p. 337).
- The in-sample PPC AUC near 0.95 is not an out-of-sample estimate. The later random split puts nearby months from the same recession in both train and test sets, which the chapter acknowledges makes prediction easier (PDF pp. 338-340).
- Random row splitting is invalid for overlapping horizons or serially dependent market observations. Use chronological splits with overlap-aware purging or embargo.
- Bayesian Sharpe estimates remain conditional on the return likelihood, selected period, zero risk-free rate, and stationarity. They do not solve selection bias, backtest overfitting, autocorrelation, changing exposure, or trading costs (PDF pp. 341-343).
- The pairs example uses price levels for a more striking visualization while explicitly noting that returns should be used to estimate the hedge ratio. Its evolving scatter relationship also warns against a fixed relationship (PDF pp. 343-346).
- A random-walk coefficient inferred with all observations can embed future data in earlier latent states if smoothed posterior estimates are used as historical features. Live features require one-sided filtering or a refit that stops before each decision.
- The stochastic-volatility result is explicitly in-sample, and the run reports fewer than 200 effective samples for some parameters despite 28,000 total draws (PDF pp. 348-349).
- Bayesian uncertainty is conditional on a chosen model. Misspecification and structural breaks can dominate the displayed credible interval.

## Implementation patterns worth preserving

- Declare every prior, likelihood, transformation, and deterministic output in a versioned model specification.
- Run prior-predictive simulation before fitting and posterior-predictive checks afterward.
- Preserve chain-specific seeds, package versions, sampler configuration, tuning draws, retained draws, divergences, R-hat, effective sample size, and wall-clock cost.
- Treat sampler health as a validation gate rather than a notebook annotation.
- Store posterior summaries with their exact as-of timestamp, training-window endpoint, feature set, prior version, and model version.
- Use one-sided filters for live latent-state features; keep full-sample smoothing confined to retrospective diagnostics.
- Compare MCMC and VI on a limited benchmark before substituting a faster approximation.
- Use posterior probabilities or credible-width features only when they are produced causally and improve proper out-of-sample scoring.
- Separate inferential uncertainty from execution and PnL simulation.

## Dated APIs and examples

The chapter reflects a 2019-2020 probabilistic-programming ecosystem and must be translated against current documentation before implementation.

- It uses **PyMC3** with **Theano** and mentions an alpha release of PyMC4. Those names, backends, trace formats, and compatibility expectations are historical (PDF pp. 329-340).
- Examples use `pm.glm.GLM.from_formula`, `pm.find_MAP`, `pm.sample`, `pm.fit`, `pm.sample_ppc`, `pm.summary`, `pm.model_to_graphviz`, and `pm.GaussianRandomWalk`. Current signatures and availability must be checked.
- Historical arguments include `sd`, `lam`, `nuts_kwargs`, `testval`, `cores`, `init='adapt_diag'`, and raw `MultiTrace`-style objects. Parameterization changes can silently alter a ported model.
- `pm.sample_ppc` and the chapter's HPD terminology should not be assumed to match current posterior-predictive and interval APIs.
- `theano.shared` is used to swap training and test arrays; current backend/shared-data mechanisms require verification (PDF pp. 338-339).
- `yfinance`, FRED access, `pandas.read_hdf`, scikit-learn scaling, and formula parsing are illustrative dependencies rather than Project 1 requirements.

## Project 1 relevance

- **High conceptual relevance:** causal uncertainty features, posterior predictive probabilities, latent volatility states, dynamic coefficients, prior sensitivity, and diagnostic gating.
- Project 1's one-minute GC observations, stable IDs, fixed horizons, continuity rules, and Development/Validation separation are stricter than the chapter demonstrations. Those controls must remain authoritative.
- Bayesian classification is most naturally evaluated for the frozen expansion targets, where posterior predictive probabilities and uncertainty width can complement ATR and realized-volatility baselines.
- A latent-volatility model is relevant only if its one-sided posterior state adds Validation information beyond `atr_20`, causal realized volatility, time-of-day seasonality, and a simple exponentially weighted variance.
- Bayesian Sharpe comparison is useful later for uncertainty-aware policy comparison, but it is not a feature and should not be used before a sequential, costed, locked out-of-sample policy exists.
- The dynamic hedge-ratio example suggests a general pattern for drifting coefficients. It does not justify bringing ETF pairs, price-level regressions, or MGC information into the current GC discovery stage.
- Computational cost and reproducibility are material at one-minute frequency. A bounded model or sequential approximation should be preferred over repeated full MCMC unless posterior quality changes a decision.

## One-minute GC adaptation

Potential bounded hypotheses:

1. A Bayesian logistic model for each frozen expansion horizon produces better calibrated causal probabilities than the existing non-Bayesian baseline.
2. A one-sided latent-volatility state or its posterior width adds stable expansion information beyond `atr_20`, realized volatility, and New York time-of-day controls.
3. A time-varying coefficient model detects a useful shift in the relationship between a predeclared GC state feature and future expansion without using cross-instrument data.

Adaptation requirements:

- Build every update within valid continuity runs and reset state at gaps, contract-selection changes, rollover, segment, tradability, or liquidity boundaries.
- Restrict prior selection, likelihood choice, hyperparameters, sampler choice, and threshold design to GC Development data.
- Freeze the model recipe before Validation; MGC remains excluded from feature discovery.
- Use only observations available through completed decision bar \(t\). Never populate historical rows with latent states smoothed using later observations.
- Predeclare weakly informative priors on standardized variables and run prior-sensitivity checks.
- Remove deterministic intraday volatility seasonality causally or condition on it explicitly.
- Use chronological, overlap-aware folds and identical evaluation rows across baselines.
- Evaluate log loss, Brier score, calibration, discrimination, posterior-width stability, and incremental value beyond frozen baselines.
- Persist reproducible inference diagnostics and fail closed on divergences, poor R-hat, inadequate effective sample size, or save/reload mismatch.
- Benchmark a deterministic approximation for runtime and numerical reproducibility before promoting an MCMC-derived feature.

## Unsuitable or deferred ideas

- A stratified random train/test split for one-minute or overlapping-horizon observations.
- Full-sample posterior smoothing used as a historical feature.
- Copying the chapter's normal priors with standard deviation 100 without scaling and prior-predictive checks.
- Treating Bayesian credible intervals as protection against leakage or model misspecification.
- Running a large MCMC model independently at every bar without a bounded operational design.
- Using the GLD/GFI price-level regression or its priors as a GC hedge-ratio feature.
- Using Bayesian Sharpe as a discovery target or selecting many strategies by their posterior probability of superiority without multiplicity control.
- Accepting an in-sample volatility fit as forecast evidence.
- Importing MGC or another instrument into the current GC-only feature-research stage.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Bayesian versus frequentist inference and Bayes' theorem | 320-323 |
| Prior selection, conjugacy, and beta-binomial updating | 323-325 |
| MCMC, Gibbs, Metropolis-Hastings, HMC, NUTS, and VI | 325-329 |
| PyMC3 workflow and recession data | 329-331 |
| Bayesian logistic model, MAP, MCMC, and ADVI | 331-336 |
| Diagnostics, posterior predictive checks, and prediction | 336-340 |
| Bayesian Sharpe ratio and return-series comparison | 340-343 |
| Dynamic Bayesian regression for pairs trading | 343-346 |
| Bayesian stochastic volatility | 346-349 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 322, 332, 337, 344, 347.
- **Confidence: high** for the chapter structure, concepts, examples, diagnostics, and warnings; all assigned pages 320-349 were read sequentially.
- **Confidence: high** for Bayes' theorem, the logistic link, the diagnostic table, and the dynamic-regression equations; PDF pages **322, 332, 337, and 344** were visually inspected.
- **Confidence: high** for the stochastic-volatility display and its code parameterization; PDF page **347** was visually inspected.
- **Confidence: medium** for the semantic interpretation of \(\sigma^{-2}\) and \(\exp(-2s_i)\) in the displayed stochastic-volatility notation because PyMC3's code uses precision parameter `lam`; the source's symbols are preserved rather than reinterpreted.
- The recession PPC snippet on PDF p. 338 uses `data.income` as `y_true`, which appears inconsistent with the recession outcome and may be a copied variable name.
- The out-of-sample AUC is shown as `0.8386` in code while the following prose calls it `0.86` (PDF p. 339).
- Sampling-count descriptions and figure captions vary: PDF p. 335 mentions an additional 20,000 samples but Figure 10.7 says 50,000; PDF p. 336 refers to an additional 200,000 samples.
- Project 1 applicability remains hypothetical until a separately registered implementation passes causal construction, inference-diagnostic, and Development/Validation tests.
