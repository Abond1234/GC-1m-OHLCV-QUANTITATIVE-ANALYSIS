# Chapter 12 - Boosting Your Trading Strategy

**Assigned source range:** PDF pages 387-428

**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`

**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`

**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Boosting builds an ensemble sequentially: each new learner responds to errors made by the current ensemble. AdaBoost does this by raising the weights of misclassified observations, while gradient boosting fits new regression trees to the direction that reduces a differentiable loss. Shrinkage, tree complexity, subsampling, regularization, ensemble size, and stopping rules govern whether this flexibility captures signal or noise (PDF pp. 387-400).

The chapter then surveys XGBoost, LightGBM, and CatBoost, interprets boosted-tree predictions with importance measures, partial dependence, and SHAP, and demonstrates daily and minute-frequency equity signals. These are useful research templates, not validated strategy evidence. The daily strategy has 127.3% out-of-sample daily turnover under generous cost assumptions, and the minute example reports a 0.5-basis-point gross spread but omits the backtest entirely. The minute decile construction also uses the full day's prediction distribution, which is unavailable to an earlier intraday decision (PDF pp. 400-428).

## Concepts and definitions

- **Boosting:** an ensemble method that trains base learners sequentially so each addition addresses shortcomings of the current ensemble (PDF pp. 387-389).
- **Weak learner:** a model that performs only slightly better than chance. AdaBoost often uses a decision stump, a tree with one split (PDF pp. 388-390).
- **Additive model:** represents the prediction as a sum of base-learner functions, optionally weighted or shrunk (PDF pp. 388-393).
- **AdaBoost:** iteratively fits a weighted classifier, increases the influence of misclassified samples, and weights each learner according to its training error (PDF pp. 389-392).
- **Gradient boosting machine (GBM):** fits each new regression tree to pseudo-residuals corresponding to the negative gradient of a differentiable loss, then finds its incremental contribution to the ensemble (PDF pp. 392-394).
- **Pseudo-residual:** the observation-level negative gradient of the current loss. It is a continuous target even for classification, so gradient boosting uses regression trees for both regression and classification (PDF p. 393).
- **Shrinkage and learning rate:** scale down each new tree's contribution. Lower learning rates usually require more trees but can improve generalization (PDF p. 395).
- **Early stopping:** halts training when validation loss stops improving. Repeated use of the same validation data adapts to that set and biases its performance estimate (PDF pp. 395, 408).
- **Stochastic gradient boosting:** samples training rows without replacement for each tree and is normally combined with shrinkage (PDF pp. 395-396).
- **Second-order loss approximation:** XGBoost-style training uses both gradients and Hessians to score leaf values and splits efficiently (PDF pp. 400-402).
- **Histogram or approximate split finding:** bins continuous feature values to reduce the number of candidate thresholds and memory use (PDF p. 402).
- **Gradient-based one-side sampling (GOSS):** LightGBM retains observations with large gradients and samples among smaller-gradient observations when estimating split gain (PDF p. 402).
- **Exclusive feature bundling:** combines nearly mutually exclusive sparse features to reduce effective dimensionality (PDF p. 402).
- **Depth-wise versus leaf-wise growth:** depth-wise growth expands a full level; LightGBM's leaf-wise mode chooses the leaf with highest gain and can produce deep, unbalanced branches (PDF pp. 402-403).
- **DART:** dropout for additive regression trees, which temporarily omits whole trees during training so later predictions do not depend excessively on a few ensemble members (PDF p. 403).
- **Categorical handling:** LightGBM groups categories to improve within-group outcome homogeneity; CatBoost uses smoothed, ordered outcome-dependent category statistics and feature combinations; the chapter's XGBoost workflow requires encoding (PDF pp. 403-407).
- **Monotonicity constraint:** restricts the model response to move only up or only down with a selected feature, encoding a justified external relationship (PDF p. 404).
- **Gain, split-count, and permutation importance:** gain totals loss reduction, split count totals feature uses, and permutation measures held-out error deterioration after disrupting a feature (PDF pp. 413-414).
- **Partial dependence:** averages model predictions over the empirical values of other features to display a marginal feature-response relationship (PDF pp. 414-417).
- **SHAP:** assigns each feature an additive contribution to an individual model output using Shapley-value principles; summaries aggregate local attributions across samples (PDF pp. 417-420).
- **Information coefficient (IC):** the chapter uses Spearman rank correlation between forecasts and forward returns, often scaled for display (PDF pp. 408-413, 425-427).

## Formulas and notation

Only equations visually recoverable from the supplied PDF or explicitly represented by displayed code are transcribed. Source notation is preserved where it is internally inconsistent.

- The displayed generic additive hypothesis is:

  \[
  H_M(x)=\sum_{m=1}^{M}h_t(x)
  \]

  The summation index is \(m\), but the displayed weak learner is labeled \(h_t\); the intended generic idea is a sum of \(M\) base learners (PDF p. 388).

- For AdaBoost base learner \(h_m\), weighted training error \(\epsilon_m\), and class labels in \(\{-1,1\}\), the displayed learner weight and ensemble prediction are:

  \[
  \alpha_m=\log\left(\frac{1-\epsilon_m}{\epsilon_m}\right),
  \]

  \[
  H(x)=\operatorname{sign}\left(
  \sum_{m=1}^{M}\alpha_m h_m(x)
  \right)
  \]

  Misclassified sample weights are multiplied by \(\exp(\alpha_m)\) before the next stage; the chapter initializes all weights at \(1/N\) (PDF p. 389).

- The displayed stagewise gradient-boosting update is:

  \[
  H_m(x)
  =
  H_{m-1}(x)+\gamma_m h_m(x)
  =
  H_{m-1}(x)+
  \underset{\gamma,h}{\operatorname{argmin}}
  \sum_{i=1}^{N}
  L\!\left(y_i,H_{m-1}(x_i)+h(x)\right)
  \]

  The source describes fitting \(h_m\) to the negative gradient and then selecting an optimal leaf prediction; the displayed `argmin` expression is preserved rather than algebraically repaired (PDF p. 393).

- The visually verified XGBoost-style tree-complexity penalty for a tree with \(T\) leaves and leaf-score vector \(w\) is:

  \[
  \Omega(h)=\gamma T+\frac{1}{2}\lambda\lVert w\rVert^2
  \]

  (PDF p. 401).

- The displayed second-order approximation uses first and second loss derivatives \(g_i\) and \(h_i\):

  \[
  \mathcal{L}^{(m)}
  \simeq
  \sum_{i=1}^{n}
  \left[
  g_i f_m(x_i)+\frac{1}{2}h_i f_m^2(x_i)
  \right]
  +
  \Omega(h_m),
  \]

  \[
  g_i=
  \partial_{\hat y_i^{(m-1)}}
  l\!\left(y_i,\hat y_i^{(m-1)}\right),
  \qquad
  h_i=
  \partial_{\hat y_i^{(m-1)}}^2
  l\!\left(y_i,\hat y_i^{(m-1)}\right)
  \]

  The approximation writes the new function as \(f_m\) while the penalty uses \(h_m\), another source-level notation change (PDF p. 401).

- For \(T\) trees, \(L\) leaves, maximum depth \(D\), and \(M\) features, the chapter reports that TreeSHAP reduces computational complexity from:

  \[
  O(TLDM)\quad\text{to}\quad O(TLD^2)
  \]

  (PDF p. 417).

- The model-evaluation code repeatedly computes:

  \[
  IC=\rho_{\text{Spearman}}(y,\hat y)
  \]

  across complete validation predictions and within calendar-day groups (PDF pp. 408-413, 425-427).

## Assumptions

- The chosen loss function corresponds to the economic prediction task and target distribution.
- Base trees, ensemble size, learning rate, sampling fractions, leaf support, and regularization are selected without adapting to the final test.
- Difficult or repeatedly misclassified observations contain learnable structure rather than data errors or irreducible noise; this is especially important for AdaBoost (PDF pp. 389-390).
- Validation folds respect event chronology, label overlap, availability timestamps, universe membership, and all target-dependent transformations.
- Approximate bins, GOSS sampling, leaf-wise growth, and categorical encodings preserve the relevant signal without creating thin or unstable regions.
- Feature-attribution methods explain a fitted model under a selected background/data distribution; they do not identify causal effects.
- The daily and minute equity signals can be converted into executable orders only after spreads, impact, turnover, borrow, latency, and risk constraints are modeled.

## Procedures described by the chapter

### AdaBoost and scikit-learn GBM baselines

1. Encode monthly US-equity features and define a binary price-direction target.
2. Fit AdaBoost with shallow trees, reweighting training observations after every stage.
3. Evaluate with a 12-fold rolling time-series splitter over the last 12 months.
4. Replace exponential AdaBoost loss with a differentiable task loss and fit gradient-boosting regression trees to pseudo-residuals.
5. Tune tree complexity, learning rate, ensemble size, row/feature subsampling, and minimum split quality.
6. Preserve a separate holdout after cross-validation and model selection (PDF pp. 388-400).

The displayed AdaBoost validation AUC is 0.5348 versus 0.5358 for a default random forest. Default scikit-learn gradient boosting reaches AUC 0.537. A 576-combination grid reports mean CV AUC 0.5569, while the first month of a seven-month holdout is reported at 0.5381 (PDF pp. 390-400).

### Optimized gradient-boosting workflow

1. Convert feature frames into LightGBM `Dataset` or CatBoost `Pool` objects.
2. Slice training and validation rows using the custom chronological splitter.
3. Search a bounded set of loss, tree-size, learning-rate, row/column sampling, leaf-support, regularization, and iteration choices.
4. Generate validation predictions for several ensemble sizes rather than using unrestricted stopping on the evaluation fold.
5. Compare overall and daily IC, inspect dispersion across configurations, and analyze parameter sensitivity.
6. Average selected model forecasts only after declaring an ensemble rule, then test it on a separately reserved period (PDF pp. 400-413).

The chapter acknowledges that the LightGBM-versus-CatBoost comparison is not fair because more LightGBM configurations were tried. Its selected models are generally shallow and use longer training histories. It averages the top five models for Alphalens, reporting a one-day validation top-bottom spread of 12.1654 basis points and annualized alpha 0.1759 (PDF pp. 409-413).

### Interpretation workflow

1. Compare gain and split-count importance.
2. Permute a feature only on held-out rows to evaluate predictive dependence.
3. Use one- and two-feature partial-dependence plots to inspect marginal response shapes.
4. Compute SHAP values for each feature and sample.
5. Inspect global SHAP summaries, local force plots, clustered force plots, and dependence/interaction plots.
6. Treat all explanations as model-behavior diagnostics and test stability across chronological folds (PDF pp. 413-420).

### Daily equity strategy

1. Select the ten best one-day LightGBM models from the validation analysis.
2. Average their predictions for validation and 2017 test rows.
3. Use a Zipline factor and pipeline to enter 25 long and 25 short daily positions.
4. Compare in-sample and out-of-sample returns and trade summaries (PDF pp. 404-405, 420-423).

The displayed out-of-sample annual return is 8.0%, Sharpe 0.61, maximum drawdown 9.8%, alpha 0.05, beta 0.22, and daily turnover 127.3%. The chapter explicitly says its transaction-cost assumptions are generous (PDF pp. 421-423).

### Minute-frequency equity signal

1. Use AlgoSeek minute bars for NASDAQ 100 names from 2013-2017, limited to official market hours.
2. Derive 20 features from ten return lags, uptick/downtick and repeated-price volume shares, ask-minus-bid trade-volume imbalance, and several technical indicators.
3. Shift technical features to avoid same/future-bar leakage; the displayed MFI example applies a one-row per-ticker shift.
4. Train LightGBM on 12 months, predict the next 21 trading days, and repeat for 24 splits covering two years.
5. Use a one-minute lookahead, 250 trees, and IC as an evaluation metric.
6. Inspect daily IC and realized next-minute returns by forecast decile (PDF pp. 423-427).

The reported overall IC is 1.90, daily mean 1.98, daily median 1.91, and gross top-bottom spread about 0.5 basis points per minute. The scale applied to IC is not explicitly stated in the prose. No executable strategy backtest is performed (PDF pp. 426-427).

## Feature and model examples

- AdaBoost observation weight, learner error, learner weight, and ensemble margin (PDF pp. 389-392).
- Gradient, Hessian, pseudo-residual, leaf value, split gain, tree complexity, and ensemble iteration (PDF pp. 392-403).
- Learning rate, tree count, depth, leaf count, minimum leaf support, feature fraction, row fraction, histogram-bin size, L1/L2 penalty, and DART dropout.
- Gain, split count, chronological permutation importance, partial dependence, local SHAP value, absolute mean SHAP, and SHAP interaction values (PDF pp. 413-420).
- Monthly/daily equity lagged returns, ATR/NATR, momentum, MACD, sector, and calendar indicators (PDF pp. 404-420).
- Minute return lags 1-10; uptick/downtick volume shares; repeated-after-up/down volume shares; at-ask minus at-bid volume share; Balance of Power; Commodity Channel Index; Stochastic RSI; and Money Flow Index (PDF pp. 423-425).
- Cross-tree or cross-model prediction dispersion as a possible uncertainty diagnostic, subject to causal generation and calibration.

## Statistical, validation, and backtesting warnings

- Boosting focuses sequentially on errors. Label noise, bad ticks, stale prices, corporate-action errors, and outliers can receive disproportionate influence (PDF pp. 389-390).
- The number of interacting hyperparameters creates a large implicit trial count. The chapter explicitly links wide searches to false discoveries and recommends low-cardinality, sequential tuning plus a holdout (PDF pp. 395-398).
- Early stopping is model selection. Reusing an outer validation fold to choose the stopping iteration makes that fold optimistic (PDF pp. 395, 408).
- The chapter says the optimized experiments will avoid early stopping because of this bias, but the minute model passes each test fold as a validation set with `early_stopping_rounds=50` (PDF pp. 408, 426).
- The displayed monthly holdout code calls `best_model.predict()` before AUC. For a classifier this may return hard labels rather than a ranking score; the current estimator behavior must be checked and `predict_proba` or `decision_function` used when appropriate (PDF pp. 399-400).
- The LightGBM-versus-CatBoost result is selection-count confounded because more LightGBM configurations were evaluated, as the chapter acknowledges (PDF p. 410).
- Choosing the top five or ten models using validation IC and then reporting the same validation factor spread compounds selection bias (PDF pp. 409-413, 420-421).
- Outcome-dependent categorical encodings must be trained inside each fold. Precomputing them across validation or future rows leaks target information (PDF pp. 403-407).
- Leaf-wise growth can create deep, thin branches even with a modest leaf count. `num_leaves` and realized depth/support must be constrained jointly (PDF pp. 402-403, 407-408).
- Gain and split-count importance are training-path summaries. Permutation importance is more directly predictive but is distorted by correlated substitutes and must use untouched chronological data.
- Partial dependence can evaluate unrealistic feature combinations when inputs are correlated. It is a marginal model diagnostic, not a causal response curve.
- SHAP explains the fitted model, not the market. The chapter explicitly warns that correlated variables can receive arbitrary attribution shares (PDF pp. 417-420).
- Calendar variables dominate the displayed importance and individual explanation examples. This can reflect sample-specific regimes rather than transferable seasonality (PDF pp. 414, 418).
- The daily strategy's out-of-sample period is only 2017 after broad model search. Its 127.3% daily turnover makes omitted or generous costs decisive; short trades lose on average (PDF pp. 420-423).
- The minute IC is computed over a cross-section of many equities and times. It is not evidence that the same model predicts a single futures instrument through time.
- The minute decile code groups forecasts by **calendar date**, not exact timestamp, before calling `qcut`. An early-minute decile therefore depends on the distribution of model scores from later that same day. This is noncausal for an intraday portfolio rule (PDF p. 427).
- The 0.5-basis-point minute spread is gross, and the chapter omits the backtest because it is costly. Spread, queue position, impact, turnover, latency, and adverse selection can easily dominate it (PDF p. 427).
- Cumulative returns formed from repeated one-minute decile means are not a substitute for orders, fills, capital constraints, or exposure accounting.

## Implementation patterns worth preserving

- Use a loss matched to the frozen target and monitor both loss and economically relevant out-of-sample metrics.
- Use nested chronology: tune stopping and hyperparameters inside Development folds, then evaluate one frozen model recipe on Validation.
- Bound the search before execution and log every attempted configuration, seed, fold, runtime, and failure.
- Persist library-specific binary-data metadata, ordered feature names, categorical mappings, model parameters, iteration count, and serialized-model hash.
- Fit every target-based categorical transformation inside the training fold.
- Monitor realized depth, leaf support, gradient/Hessian ranges, split gain, missing-value direction, and prediction distribution by fold.
- Require deterministic or reproducibly bounded training settings and save/reload prediction parity.
- Compute permutation and SHAP diagnostics on chronologically held-out rows, with a declared background sample.
- Compare explanations across folds and regimes; unstable attribution is a model-risk signal.
- Generate intraday ranks from the contemporaneously available cross-section only, or use thresholds frozen from prior data.
- Keep signal scoring separate from execution simulation, and require cost stress before policy promotion.

## Dated APIs and examples

The chapter reflects the 2019-2020 API surface of scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, and Quantopian-era tools.

- `AdaBoostClassifier(base_estimator=..., algorithm='SAMME.R')` uses historical estimator and algorithm parameters.
- `GradientBoostingClassifier(loss='deviance')`, `min_impurity_split`, and `presort='auto'` are historical names or options whose current availability must be checked.
- `HistGradientBoostingClassifier` is described as experimental in scikit-learn 0.21 (PDF p. 396).
- The examples use `fit_params` in `cross_validate`, `plot_partial_dependence`, the older `partial_dependence` return shape, and `joblib` persistence patterns.
- LightGBM examples use `verbose_eval`, `early_stopping_rounds`, `feval`, `num_boost_round`, `num_iteration`, `Dataset.subset`, and historical GPU-installation assumptions.
- SHAP examples use `TreeExplainer.shap_values`, `summary_plot`, `force_plot`, and `initjs`; output shapes, default perturbation/background behavior, and plotting APIs require verification.
- `DataFrame.append` in the prediction loader is a removed historical pandas pattern (PDF p. 421).
- Zipline, pyfolio, Alphalens, Quandl Wiki equity data, TA-Lib, and the specific AlgoSeek sample are historical research dependencies rather than Project 1 requirements.

## Project 1 relevance

- **Very high direct relevance:** the chapter contains a one-minute tree-boosting example, causal shifting intent, rolling train/test design, nonlinear interactions, and feature-attribution tools.
- The transfer is conceptual, not empirical. The book studies a cross-section of NASDAQ equities with trade-side aggregates; Project 1 studies a single GC futures series with its own continuity, rollover, session, and feature contracts.
- A compact boosted-tree model is a strong nonlinear challenger for the frozen GC statistical feature matrix, especially for expansion targets. It should be evaluated only after simpler linear, shallow-tree, and random-forest baselines.
- Gradient boosting's flexibility makes it especially vulnerable to the project's limited effective sample size from overlapping one-minute horizons. Bounded search and leaf support are mandatory.
- SHAP and permutation diagnostics may reveal interactions among ATR, realized volatility, return/range state, and time of day, but cannot advance a feature without independent chronological performance.
- The chapter's minute decile analysis cannot be inherited because it is cross-sectional, noncausal within the day, uncosted, and not a strategy backtest.

## One-minute GC adaptation

Potential bounded hypotheses:

1. A shallow LightGBM or histogram-gradient-boosting model improves frozen GC expansion probability or rank performance beyond the existing baseline feature set.
2. Nonlinear interactions between `atr_20`, causal realized volatility, recent range/return state, and New York time-of-day are stable across Development folds and Validation.
3. Chronological permutation and SHAP stability identify a small, robust interaction set without changing the frozen feature definitions.

Adaptation requirements:

- Consume the exact frozen observation IDs, continuity gates, labels, horizons, and partitions from the statistical feature-research contract.
- Reset rolling inputs at timestamp gaps, selected-contract changes, rollovers, segment changes, and tradability/liquidity boundaries.
- Use only information through completed decision bar \(t\); align every forward label to the declared execution timestamp.
- Restrict objective, feature set, categorical handling, hyperparameters, iteration rule, and ensemble rule to GC Development data. MGC remains isolated.
- Use an inner chronological split for early stopping. Never stop on the outer Validation fold.
- Purge or embargo by the full label-information interval, not merely one row, for all six fixed horizons.
- Predeclare a small search over learning rate, depth/leaves, minimum leaf support, feature fraction, and iteration count.
- Compare on identical rows against the frozen linear/statistical baselines, a shallow tree, and a random forest.
- Evaluate proper probability scores and calibration for classification, rank/return metrics where appropriate, and stability by horizon, session, volatility regime, and continuity run.
- Generate any thresholds or ranks from past data or the current timestamp only; never use the completed day's future prediction distribution.
- Persist deterministic seeds, thread/device settings, feature order, package versions, data hash, model hash, and save/reload prediction hash.
- Defer PnL claims until a separately specified sequential policy passes spread, fees, slippage, impact, latency, turnover, and risk stress.

## Unsuitable or deferred ideas

- Copying the NASDAQ-equity minute IC, decile spread, or default 250-tree model into GC.
- Grouping all predictions from a completed day to rank an earlier intraday observation.
- Early stopping on the same fold reported as out of sample.
- Selecting top-five or top-ten ensembles after viewing Validation performance.
- Large random or Cartesian hyperparameter searches without a registered trial budget.
- Using calendar year as a transferable predictor.
- Treating gain, split count, partial dependence, or SHAP as evidence of causality.
- Target-encoding categories outside the training fold.
- Accepting gross 0.5-basis-point signal spread as evidence of tradable edge.
- GPU acceleration or DART solely to enlarge the search space before a reproducible CPU baseline exists.
- Importing cross-sectional or MGC data into the current GC-only discovery stage.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Boosting versus bagging and AdaBoost | 387-392 |
| Gradient boosting mechanics and loss functions | 392-394 |
| Tree complexity, early stopping, shrinkage, and subsampling | 394-398 |
| scikit-learn grid search and holdout | 396-400 |
| XGBoost, LightGBM, and CatBoost innovations | 400-404 |
| Daily equity boosting strategy setup | 404-413 |
| Gain, split, permutation, and partial dependence | 413-417 |
| SHAP summaries, local explanations, and interactions | 417-420 |
| Daily long-short backtest and cost warning | 420-423 |
| Minute data and causal feature construction | 423-425 |
| Minute LightGBM validation and uncosted decile analysis | 425-428 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 388, 389, 393, 401, 411, 417, 422, 425, 427.
- **Confidence: high** for the chapter structure, algorithms, implementation comparisons, interpretation methods, experiments, reported results, and warnings; all assigned pages 387-428 were read sequentially.
- **Confidence: high** for the additive, AdaBoost, gradient-boosting, and XGBoost equations; PDF pages **388, 389, 393, and 401** were visually inspected.
- **Confidence: high** for the selected-model results, SHAP complexity statement, daily-strategy turnover, shifted MFI construction, and gross minute-decile spread; PDF pages **411, 417, 422, 425, and 427** were visually inspected.
- The generic additive formula sums over \(m\) but labels the learner \(h_t\). The XGBoost discussion alternates between \(h_m\) and \(f_m\). These source notational inconsistencies were not silently corrected.
- The 12-fold `OneStepTimeSeriesSplit` example sets `shuffle=True`; the effect on ordering is not explained and must not be assumed safe (PDF p. 391).
- The holdout contains seven months, but only the first month's AUC 0.5381 is described in the prose (PDF pp. 399-400).
- The heading "Use IC instead of information coefficient" appears self-contradictory and likely does not state the intended Alphalens topic (PDF p. 412).
- The optimized workflow says it will avoid early stopping, while the minute example later uses the test fold for early stopping (PDF pp. 408, 426).
- The IC values 1.90, 1.98, and 1.91 are reported without an explicit percent or scaling convention, although raw Spearman correlation is bounded by one (PDF pp. 426-427).
- The raw data description includes a 16:00 timestamp, while the code defines 390 bars as 9:30-15:59; endpoint and bar-label semantics are unresolved (PDF pp. 424-425).
- Project 1 applicability remains hypothetical until a separately registered implementation passes causal construction and Development/Validation tests.
