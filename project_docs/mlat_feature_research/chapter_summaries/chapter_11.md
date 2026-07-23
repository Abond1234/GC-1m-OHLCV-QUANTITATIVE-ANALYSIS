# Chapter 11 - Random Forests: A Long-Short Strategy for Japanese Stocks

**Assigned source range:** PDF pages 350-386

**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`

**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`

**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Decision trees learn nonlinear prediction rules by greedily partitioning feature space and assigning a constant prediction to each resulting region. Their transparency and ability to capture interactions make them useful baselines, but unrestricted trees naturally fit noise and have high variance. The chapter therefore develops regularization, pruning, chronological cross-validation, bootstrap aggregation, and random forests before applying a LightGBM random-forest model to a Japanese-equity long-short strategy (PDF pp. 350-386).

The trading lesson is more restrained than the headline strategy result. Complexity does not guarantee improvement: in the chapter's monthly-equity experiment, the tuned random forests underperform individual decision trees. Any apparent edge also depends on point-in-time data, leakage-safe validation, bounded model selection, realistic execution, and a genuinely untouched test. The chapter explicitly flags random splitting, out-of-bag testing, and validation-driven early stopping as unsafe or biased for time-series use (PDF pp. 357-376, 377-386).

## Concepts and definitions

- **Decision tree:** a supervised model that predicts by applying a sequence of feature-threshold rules. A regression leaf predicts the mean training outcome in its region; a classification leaf predicts a class frequency or the majority class (PDF pp. 351-353, 354-359).
- **Root, internal node, edge, and leaf:** the root contains all training observations, internal nodes apply split rules, edges route observations, and leaves contain the final regional predictions (PDF pp. 351-352).
- **Recursive binary splitting:** a top-down, greedy algorithm that repeatedly chooses the feature and threshold yielding the best immediate loss reduction. It does not search all possible trees or guarantee a globally optimal partition (PDF p. 352).
- **Node purity:** the concentration of observations from a single class in a node. Gini impurity and cross-entropy are more sensitive to improving purity than raw classification error (PDF pp. 356-357).
- **Tree depth and leaf support:** deeper trees create more regions and can fit more complex interactions, but leave fewer observations supporting each leaf prediction (PDF pp. 352, 359-365).
- **Pre-pruning or growth constraints:** `max_depth`, `max_leaf_nodes`, `min_samples_split`, `min_samples_leaf`, `max_features`, and impurity thresholds limit complexity while the tree grows (PDF pp. 359-361).
- **Cost-complexity pruning:** first grows a large tree and then creates nested subtrees by penalizing additional leaves. Cross-validation selects the penalty and subtree (PDF pp. 359-361).
- **Learning curve:** plots training and validation performance against training-set size to distinguish potentially data-limited variance from persistent bias (PDF pp. 365-366).
- **Impurity-based feature importance:** sums the sample-weighted objective reduction attributed to a feature's splits and normalizes importance across features; forest importance averages this quantity across trees (PDF pp. 366, 374-375).
- **Ensemble learning:** combines multiple base estimators to reduce generalization error. Members need useful individual predictions and sufficiently different errors (PDF pp. 367-369).
- **Averaging versus boosting:** averaging trains base models independently and combines predictions to reduce variance; boosting trains models sequentially to reduce ensemble bias (PDF p. 368).
- **Bootstrap aggregation (bagging):** trains base estimators on row samples drawn with replacement and averages or votes their predictions (PDF pp. 368-371).
- **Pasting, random subspaces, and random patches:** pasting samples rows without replacement; random subspaces sample features; random patches sample both observations and features (PDF p. 369).
- **Random forest:** combines bootstrap row sampling with random feature subsets, typically selected at each split, to decorrelate tree errors before averaging (PDF pp. 371-374).
- **Out-of-bag (OOB) observation:** a row absent from a tree's bootstrap sample. Predictions from trees that did not train on that row can estimate generalization error under exchangeable sampling assumptions (PDF pp. 375-376).
- **Information coefficient (IC):** the chapter uses Spearman rank correlation between predicted and realized returns as the regression scoring metric (PDF pp. 362, 381-383).
- **Long-short signal:** ranks predicted returns cross-sectionally, buys high-ranked assets, and shorts low-ranked assets to seek a spread with limited broad-market exposure (PDF pp. 376-386).

## Formulas and notation

Only equations visible in the supplied PDF or directly represented by displayed code are transcribed.

- A split on feature \(X_i\) at cut point \(s_j\) creates the two regions:

  \[
  R_1(i,s_j)=\{X\mid X_i<s_j\},
  \qquad
  R_2(i,s_j)=\{X\mid X_i>s_j\}
  \]

  The algorithm selects the feature-threshold pair that maximizes the reduction in the sample-weighted child loss relative to the parent (PDF pp. 351-354).

- For node \(m\), class \(k\), and within-node class proportion \(p_{mk}\), the visually verified classification losses are:

  \[
  \operatorname{Gini}(m)=\sum_k p_{mk}(1-p_{mk}),
  \]

  \[
  \operatorname{CrossEntropy}(m)
  =
  -\sum_k p_{mk}\log(p_{mk})
  \]

  Both are largest for a balanced binary node and fall to zero for a pure node (PDF pp. 356-357).

- The chapter's visually verified squared-error decomposition at test point \(x_0\) is:

  \[
  \mathbb{E}\left[y_0-\hat f(x_0)\right]^2
  =
  \operatorname{Var}\!\left(\hat f(x_0)\right)
  +
  \left[\operatorname{Bias}\!\left(\hat f(x_0)\right)\right]^2
  +
  \operatorname{Var}(\epsilon)
  \]

  Bagging targets the first term by averaging differently randomized high-variance learners (PDF pp. 370-371).

- For a bootstrap sample of size \(n\), the probability that a particular observation is never drawn is:

  \[
  \left(1-\frac{1}{n}\right)^n
  \longrightarrow
  \frac{1}{e}
  \]

  so roughly one-third of observations are out of bag for any one tree (PDF p. 375).

- The custom regression scorer is the Spearman rank correlation:

  \[
  IC=\rho_{\text{Spearman}}(y,\hat y)
  \]

  computed both across validation predictions and by day in the Japanese-equity experiment (PDF pp. 362, 381-383).

## Assumptions

- Training rows represent the future prediction environment after respecting chronology, continuity, and label horizons.
- Leaf means or class frequencies are stable enough to apply to later observations with similar features.
- The selected depth, minimum leaf size, and feature subset adequately balance bias against variance.
- Bootstrap randomization produces meaningfully different errors; averaging highly correlated or systematically biased trees will not remove the common error (PDF pp. 368-373).
- OOB error is only valid when row order is exchangeable for the evaluation question. The chapter explicitly says this is difficult for time series (PDF pp. 375-376).
- Impurity reduction on the training set is a useful but conditional description of model reliance, not causal or stable feature value.
- Japanese stock histories, corporate actions, volume, membership, and categorical values are correct and available point in time.
- The cross-sectional portfolio can obtain borrow, execute both sides at modeled prices, and control unwanted beta, sector, currency, and liquidity exposures.

## Procedures described by the chapter

### Train and tune a decision tree

1. Define a continuous return target or categorical direction target.
2. Fit preprocessing on the training fold only; the illustrated scikit-learn trees require missing values to be dropped and categories to be encoded.
3. Grow a tree by selecting the best feature-threshold split at each node.
4. Bound complexity with depth, minimum samples, feature sampling, leaf limits, impurity thresholds, or cost-complexity pruning.
5. Use chronological cross-validation to select hyperparameters.
6. Compare training and validation curves by depth, inspect leaf counts, and use learning curves to evaluate whether more data may help.
7. Visualize shallow rules and inspect model reliance, while treating importance as descriptive rather than definitive (PDF pp. 351-367).

The chapter first demonstrates a random `train_test_split` and reports AUC 0.6341, but explicitly states that the split does not protect against lookahead. Its later custom splitter uses ten folds, 60 months of training, six months of validation, and a one-period lookahead gap. The tuned single-tree results are IC 0.0829 for regression and AUC 0.5250 for classification (PDF pp. 357-365).

### Build and tune a random forest

1. Draw randomized row samples for individual trees.
2. Restrict candidate features at each split to decorrelate tree structures.
3. Grow many low-bias trees, usually deeper than a standalone regularized tree.
4. Average regression predictions or class probabilities/votes.
5. Tune tree count, depth, minimum leaf support, row fraction, and feature fraction using chronological validation.
6. Monitor validation performance against forest size and compare with simpler models (PDF pp. 368-376).

For the monthly-equity example, the best displayed forest uses 100 trees and minimum leaf size 5, with depth 5 for regression and 15 for classification. Its IC 0.0435 and AUC 0.5205 both trail the tuned individual trees (PDF pp. 373-374).

### Japanese-equity signal and backtest

1. Source 2010-2019 Stooq prices for roughly 3,000 Japanese stocks, remove tickers with more than five consecutive missing observations, and retain a liquid universe.
2. Build lagged returns, PPO, NATR, RSI, Bollinger-band ratios, calendar categories, and daily cross-sectional return ranks.
3. Create forward-return targets at four horizons up to 21 trading days, yielding approximately 2.3 million rows, 18 features, four outcomes, and 941 stocks.
4. Use the 250 most-traded 2010-2017 names for model-selection experiments; reserve 2018-2019 for the displayed out-of-sample period.
5. Search lookbacks of 63, 126, 252, 756, and 1,260 trading days; roll-forward periods of 5, 21, or 63 days; row and feature fractions; leaf sizes; and tree counts.
6. Evaluate daily and aggregate IC. Average the top three one-day models for Alphalens analysis, then use the top ten one-day models for the strategy.
7. Rebalance daily into up to 25 highest positive forecasts and 25 lowest negative forecasts, requiring at least 15 candidates per side and closing names that leave the selected tails.
8. Apply a commission of $0.05 per share, but no slippage, and evaluate with pyfolio (PDF pp. 376-386).

## Feature and model examples

- Lagged returns at 1, 3, 6, and 12 months; momentum ratios; ATR/NATR; RSI; Fama-French factor loadings; and year, month, and sector categories for the monthly-equity example (PDF pp. 353-366).
- A two-lag regression tree as a nonlinear counterpart to an AR(2)-style linear model (PDF pp. 354-356).
- Leaf class probabilities, decision-path membership, leaf sample counts, leaf impurity, and tree depth.
- Training-versus-validation score gap and leaf-count growth as overfitting diagnostics (PDF pp. 363-365).
- Impurity-based importance, split count, and forest-averaged feature reliance (PDF pp. 366, 374-375).
- Ensemble mean prediction, cross-tree dispersion, and row/feature sampling configuration.
- Japanese-equity lagged returns at 1, 5, 10, 21, and 63 days; PPO, NATR, RSI, Bollinger-band ratios; year, month, weekday; and daily return ranks (PDF pp. 377-378).
- Daily Spearman IC, IC distribution by configuration, and top-minus-bottom quintile return spread (PDF pp. 381-384).

## Statistical, validation, and backtesting warnings

- An unrestricted tree can drive training error toward zero by isolating observations. Small leaves produce unstable predictions and misleadingly precise-looking rules (PDF pp. 352, 359-367).
- Greedy splits optimize immediate loss and can miss a globally better tree; minor sample perturbations can materially change the chosen structure (PDF pp. 352, 367).
- The chapter's random 80/20 split ignores time order. Its AUC 0.6341 is demonstrative and not a credible trading estimate (PDF pp. 357-359).
- OOB predictions are not chronologically held out. The chapter explicitly warns that random OOB membership can create lookahead for time-series data (PDF pp. 375-376).
- Overlapping forward returns require more than a nominal lookahead offset: folds and scoring rows must prevent training labels from extending into the test information interval.
- Selecting the 250-stock research universe using average dollar-volume rank over 2010-2017 uses future liquidity information relative to earlier validation folds. A production universe must be reconstructed as of each decision date (PDF p. 379).
- Stooq data are free but the chapter states that sourcing and quality lack transparency. Corporate-action, delisting, missing-data, calendar, and survivorship audits are mandatory (PDF p. 377).
- Impurity importance is training-model specific. Features with many candidate split points, time categories, and correlated substitutes can receive distorted or unstable rankings; importance is not a causal attribution.
- Raw leaf frequencies and forest votes are not guaranteed to be calibrated probabilities.
- The search spans outcomes, horizons, lookbacks, roll-forward periods, tree counts, row fractions, feature fractions, leaf sizes, and ensembles. Choosing top models by the same validation IC creates material multiple-testing and selection risk (PDF pp. 378-384).
- The chapter itself warns that using early stopping on a validation outcome biases the cross-validation estimate upward (PDF p. 380).
- The Alphalens alpha 0.081, beta 0.083, and 5.16-basis-point top-bottom spread are validation-sample diagnostics after model selection, not independent evidence (PDF pp. 383-384).
- The strategy rebalances daily and assumes no slippage. A per-share commission alone omits spread, market impact, borrow fees and availability, short recalls, taxes, FX, and liquidity-dependent fills (PDF pp. 384-386).
- Equal long/short counts do not establish dollar, beta, sector, volatility, or factor neutrality.
- The reported 2018-2019 out-of-sample annual return 5.5%, Sharpe 0.61, alpha 0.06, and maximum drawdown 8.7% cover only 23 months after extensive research choices. They are illustrative, not sufficient evidence of a durable strategy (PDF pp. 385-386).

## Implementation patterns worth preserving

- Keep feature construction, labels, fold generation, fit, prediction, scoring, and strategy simulation as separate reproducible stages.
- Fit encoders, missing-value rules, universe filters, and ranks using only information available to the relevant fold and timestamp.
- Persist fold boundaries and assert that every training label's information interval ends before the validation interval.
- Use fixed seeds for row sampling, feature sampling, and model initialization; record library versions and thread settings.
- Compare a shallow tree, regularized forest, and simple linear baseline on identical rows before increasing complexity.
- Record leaf count, minimum realized leaf support, depth, training score, validation score, inference latency, and serialized-model hash.
- Use held-out permutation or drop-column importance within chronological folds to complement impurity importance.
- Track probability calibration separately from rank discrimination.
- Freeze the model and ensemble rule before the untouched test or production simulation.
- Put turnover, spread, impact, position limits, and exposure controls in the strategy layer rather than inferring tradability from IC.

## Dated APIs and examples

The chapter reflects scikit-learn, LightGBM, and backtesting APIs from approximately 2019-2020.

- `DecisionTreeRegressor(criterion='mse')` uses historical criterion naming; current accepted names must be checked.
- Displayed tree parameters include historical or removed options such as `min_impurity_split` and `presort`, and historical defaults such as `n_estimators=10` and `max_features='auto'`.
- `BaggingRegressor(base_estimator=...)` uses historical estimator-parameter naming.
- Cost-complexity pruning is described as newly added in scikit-learn 0.22 (PDF p. 361).
- LightGBM examples use `boosting_type='rf'`, `bagging_freq`, `bagging_fraction`, `feature_fraction`, `verbose_eval`, `num_boost_round`, and `num_iteration`; current parameter interactions, deterministic settings, and categorical handling require verification.
- Alphalens, Zipline, and pyfolio are historical Quantopian-era research tools. Their installation, maintenance, calendars, data bundles, and metric behavior should not be assumed current.
- Stooq downloads, TA-Lib indicators, and HDF5 storage are illustrative data dependencies rather than Project 1 requirements.

## Project 1 relevance

- **High direct relevance:** trees and random forests are strong nonlinear baselines for testing whether interactions among existing causal statistical features add incremental signal.
- Project 1 already defines one-minute GC observation IDs, fixed horizons, continuity gates, and Development/Validation separation. Tree models must consume those frozen rows rather than rebuilding an ad hoc sample.
- Expansion outcomes are a natural first use because volatility, ATR, time-of-day, and recent-state interactions are plausible. Directional claims should remain separate and face the existing weak-edge baseline.
- No scaling is generally required for tree splits, but all rolling statistics, imputations, clipping, time categories, and label construction still require point-in-time controls.
- Impurity importance may help inspect fitted behavior but cannot promote a feature. Advancement requires stable chronological Validation lift, permutation or ablation evidence, and robustness across horizons and regimes.
- OOB scoring must not replace the project's chronological split. Row bootstrapping inside a training fold may diversify trees, but evaluation remains strictly forward.
- The cross-sectional Japanese-equity strategy, ranking, and portfolio rules do not map to the current single-instrument GC feature batch.

## One-minute GC adaptation

Potential bounded hypotheses:

1. A shallow, regularized tree captures nonlinear interactions between `atr_20`, causal realized-volatility state, recent range/return state, and New York time-of-day for frozen expansion targets.
2. A modest random forest improves expansion probability or rank performance over the same frozen feature matrix without materially degrading calibration or reproducibility.
3. Stable decision-path or leaf-state identifiers provide interpretable regime tags, provided they are generated by a frozen model and have sufficient support.

Adaptation requirements:

- Use the exact frozen observation IDs, labels, horizons, and continuity eligibility from the statistical feature-research contract.
- Restrict definitions and hyperparameter selection to GC Development data; freeze them before Validation and keep MGC isolated.
- Predeclare a small grid over depth, minimum leaf support, feature fraction, and tree count. Avoid adaptive expansion after viewing Validation.
- Purge or embargo folds by the maximum forward-label information interval and preserve continuity-run boundaries.
- Keep all bootstrap rows within a training fold and disable OOB claims.
- Require minimum leaf support both globally and within important sessions or regimes; flag unseen or thinly supported paths.
- Evaluate Brier/log loss and calibration for probabilities, rank or AUC metrics where appropriate, and incremental lift over frozen baselines on identical rows.
- Record seeds, thread count, software versions, model hashes, prediction hashes, and save/reload parity.
- Examine chronological permutation/ablation importance and importance stability across folds rather than relying on one full-sample impurity ranking.
- Test latency and memory under the one-minute production budget before advancing an ensemble.

## Unsuitable or deferred ideas

- Random train/test splits or OOB error as evidence for a one-minute time-series model.
- Selecting tree depth, features, or ensembles from Validation after examining many alternatives.
- Fully grown trees with tiny leaves on overlapping one-minute observations.
- Treating impurity importance, tree position, or a visually plausible rule as evidence of causality.
- Cross-sectional daily ranks, Japanese-equity universe selection, or a 25-by-25 long-short portfolio in the current GC-only notebook.
- Copying the chapter's no-slippage execution assumption or per-share equity commission into futures.
- Using calendar year as a durable predictor of future one-minute behavior.
- Assuming forest probabilities are calibrated without chronological calibration tests.
- Importing MGC or another market as a feature during the current GC discovery phase.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Decision-tree intuition and recursive binary splitting | 350-353 |
| Regression and classification trees | 353-359 |
| Regularization, pruning, and hyperparameter controls | 359-363 |
| Tree structure, overfitting curves, and learning curves | 363-366 |
| Feature importance and tree strengths/weaknesses | 366-367 |
| Ensembles and bootstrap aggregation | 367-371 |
| Random-forest construction and tuning | 371-375 |
| OOB testing and time-series caveat | 375-376 |
| Japanese-equity data, features, and targets | 376-379 |
| LightGBM cross-validation and configuration analysis | 379-383 |
| Signal ensemble, Alphalens, Zipline, and pyfolio results | 383-386 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 352, 356, 357, 370, 375, 385.
- **Confidence: high** for the chapter structure, model mechanics, hyperparameters, experiments, reported metrics, and stated caveats; all assigned pages 350-386 were read sequentially.
- **Confidence: high** for the tree-learning diagram and classification-loss equations; PDF pages **352, 356, and 357** were visually inspected.
- **Confidence: high** for the bias-variance equation, OOB probability, importance chart, and strategy-performance table; PDF pages **370, 375, and 385** were visually inspected.
- The source's split notation uses strict less-than and greater-than regions and does not show how equality is routed; implementation-specific tie handling must be tested.
- Figure 11.7 is captioned as a classification-tree visualization even though the surrounding text describes leaf-count growth by depth (PDF pp. 363-364).
- The source calls the commission both "$0.05 per share" and "$0.05 cent per share"; these differ by a factor of 100. No cost conclusion should rely on the ambiguous wording (PDF pp. 384-385).
- The opening describes profitable signals over the last three years, while the detailed backtest reports 25 in-sample and 23 out-of-sample months (PDF pp. 350, 385).
- Project 1 applicability remains hypothetical until a separately registered implementation passes causal construction and Development/Validation tests.
