# Chapter 17 - Deep Learning for Trading

**Assigned source range:** PDF pages 533-568  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Deep learning is representation learning: a multilayer network composes simple nonlinear functions so that progressively more useful representations can be learned from the data rather than fully designed by hand. The chapter first develops feedforward neural-network mechanics, optimization, and regularization, then implements the same ideas with TensorFlow/Keras and PyTorch, and finally illustrates a cross-validated daily-equity return model and long-short backtest (PDF pp. 533-545, 554-568).

For financial applications, model capacity is not evidence of signal. The chapter repeatedly links successful use of neural networks to enough data, disciplined validation, regularization, and a model objective matched to the trading problem. Its trading example is illustrative and not directly transferable to one-minute GC futures (PDF pp. 541-545, 561-568).

## Concepts and definitions

- **Hierarchical representation learning:** deeper layers compose features learned by earlier layers, encoding a prior that a complex target can be represented by nested simpler functions (PDF pp. 534-538).
- **Feedforward neural network:** information moves from inputs through one or more hidden layers to an output layer without temporal recurrence. Depth, width, connectivity, activation, output, loss, and optimizer define the model (PDF pp. 538-541).
- **Activation function:** introduces nonlinearity. ReLU is recommended as a common default; sigmoid and `tanh` can saturate and produce small gradients (PDF p. 540).
- **Backpropagation:** applies the chain rule to propagate the loss gradient from the output toward hidden parameters (PDF pp. 548-553).
- **Generalization controls:** parameter-norm penalties, early stopping, and dropout reduce variance or effective capacity (PDF pp. 541-542).
- **Optimization:** stochastic gradient descent uses sample or minibatch gradients; momentum, Nesterov momentum, AdaGrad, RMSProp, and Adam modify the update or learning rate (PDF pp. 542-545).
- **Information coefficient (IC):** the trading example evaluates predictions with daily Spearman rank correlation rather than only point-error loss (PDF pp. 563-565).
- **Ensemble signal:** averaging predictions from several selected networks is used to reduce forecast variance and hedge against a single in-sample model choice (PDF p. 566).

## Formulas and notation

Only formulas directly recoverable from the displayed PDF text are transcribed
here. The backpropagation equation on PDF p. 549 was additionally verified
against a rendered page; simpler activation/output equations retain their exact
page citations but were not separately counted as visual inspections.

- ReLU activation for pre-activation \(z\):

  \[
  g(z)=\max(0,z)
  \]

  (PDF p. 540).

- For \(z=h(x)\) and \(y=o(h(x))=o(z)\), the scalar chain rule is:

  \[
  \frac{dy}{dx}=\frac{dy}{dz}\frac{dz}{dx}
  \]

  (PDF p. 549).

- For vector-valued \(z\), the component form and Jacobian form shown by the chapter are:

  \[
  \frac{\partial y}{\partial x_i}
  =\sum_j
  \frac{\partial y}{\partial z_j}
  \frac{\partial z_j}{\partial x_i},
  \qquad
  \nabla_x y=
  \left(\frac{dz}{dx}\right)^T\nabla_z y
  \]

  (PDF p. 549).

- For the chapter's softmax/cross-entropy example, the output-layer loss gradient is:

  \[
  \frac{\partial J}{\partial z_i^0}=\hat y_i-y_i,
  \qquad
  \nabla_{z^0}J=\hat Y-Y=\delta^0
  \]

  (PDF p. 549).

- Regression uses a linear output with a regression loss such as mean squared error; binary classification uses a sigmoid/Bernoulli output; multiclass classification uses softmax, generally with cross-entropy (PDF pp. 540-541).

The extracted text drops several displayed optimizer equations on PDF pp. 543-544. They are intentionally not reconstructed here from memory.

## Assumptions

- The target relationship can be represented by a hierarchy of reusable nonlinear features (PDF pp. 534-536).
- Training and validation samples are representative enough to distinguish signal from noise.
- Sample or minibatch gradients adequately approximate the useful optimization direction for SGD variants (PDF p. 543).
- Regularization and model capacity can be tuned without consuming the true out-of-sample test set (PDF pp. 541-545).
- The daily-equity example assumes that cross-sectional ranks of one-day-ahead forecasts can support simultaneous long and short portfolios (PDF pp. 561-567). That assumption does not hold automatically for a single GC stream.

## Procedures described by the chapter

### Generic neural-network workflow

1. Match output activation and loss to regression, binary classification, multiclass classification, or ranking.
2. Choose depth, width, connectivity, and hidden activations.
3. fit preprocessing parameters on training data only.
4. Train with backpropagation and a suitable optimizer.
5. Monitor training and validation error; stop or regularize when validation performance degrades.
6. Preserve checkpoints so a selected epoch can be restored without refitting.
7. Evaluate signal quality and then economic performance on held-out data (PDF pp. 540-560).

### Trading example

1. Load daily data for 995 US stocks from 2010-2017 with volatility, momentum, lagged-return, cross-sectional-rank, and sector-rank features.
2. Remove all forward outcome columns from the feature frame.
3. Search two-hidden-layer architectures, dropout rates, batch sizes, and epochs using rolling `MultipleTimeSeriesCV`.
4. Standardize validation data using a scaler fit on the corresponding training fold.
5. Score each epoch by daily Spearman IC and save checkpoints.
6. Average predictions from three selected models.
7. Inspect quintile spreads with Alphalens and backtest a daily top/bottom selection strategy with Zipline (PDF pp. 561-568).

## Feature and model examples

- Inputs: volatility and momentum factors, lagged returns, and cross-sectional/sector ranks (PDF p. 561).
- Model grid: hidden widths `(16, 8)`, `(32, 16)`, `(32, 32)`, `(64, 32)`; dropout `0`, `0.1`, or `0.2`; two batch sizes; and multiple epochs (PDF pp. 562-564).
- The five best reported configurations have median daily IC values of roughly 0.0236-0.0246 (PDF p. 564).
- An OLS diagnostic finds small contributions from batch size, dropout, and epoch, but near-zero \(R^2\), emphasizing that hyperparameter effects are noisy (PDF p. 565).
- The book's long-short illustration trades the 50 highest and 50 lowest forecasts when each side has at least 10 candidates (PDF p. 567).

## Statistical, validation, and backtesting warnings

- Using the eventual out-of-sample period for early stopping creates lookahead bias (PDF p. 542).
- Falling training error with rising validation error is evidence of overfitting; adding capacity is not the default response (PDF p. 545).
- The architecture/epoch/dropout search is itself a multiple-testing exercise. The book uses a holdout to limit false discoveries, but the reported top configurations remain selected from many related trials (PDF pp. 561-565).
- The trading results are explicitly **before transaction costs** (PDF p. 567). They do not establish executable net alpha.
- The reported out-of-sample Sharpe ratio of 2.15 is based on only 12 months, so sampling uncertainty is substantial even though the chapter labels the period out of sample (PDF p. 568).
- Averaging selected models can reduce variance but does not repair leakage, target contamination, or an invalid split.
- Neural networks can memorize low-signal financial data; dropout, L1/L2 penalties, and early stopping reduce but do not eliminate this risk (PDF pp. 541-545).

## Implementation patterns worth preserving

- Keep data construction, model construction, fitting, checkpointing, prediction, and evaluation as separate functions.
- fit scalers inside each training fold and transform validation/test data without refitting (PDF pp. 563, 566).
- Save model weights by fold and epoch; restore exact selected checkpoints for later predictions.
- Log training and validation diagnostics with a stable run identifier and fixed seed.
- Record the full search space and every result, not only the winners.
- Keep the model's predictive metric separate from the portfolio simulator and from cost assumptions.

## Dated APIs and examples

The book is a 2020-era implementation reference. Before reuse, verify current APIs and pin exact versions.

- TensorFlow 2/Keras `Sequential`, `Dense`, `Activation`, `Dropout`, `model.fit`, `save_weights`, and TensorBoard examples may require current keyword or checkpoint-format changes (PDF pp. 554-557, 561-566).
- PyTorch examples on PDF pp. 557-560 are conceptual starting points, not a pinned modern training stack.
- FastAI is mentioned as an abstraction layer on PDF pp. 560-561; its current tabular API should be checked independently.
- Alphalens and Zipline examples on PDF pp. 567-568 rely on ecosystems that have changed materially.
- The daily-equity HDF5/Quandl inputs are historical examples, not Project 1 dependencies.

## Project 1 relevance

- The strongest conceptual contribution is a disciplined nonlinear benchmark **after** causal features, labels, folds, and baselines are frozen.
- Project 1 already has a 586,530-observation eligible GC population, causal decision-bar features, fixed horizons, and Development/Validation/Final-test governance. A neural model must consume those verified artifacts rather than rebuild or relabel data ad hoc.
- Existing research found no validated directional univariate feature but did find strong expansion/opportunity structure. A first neural experiment would therefore be more defensible as a challenger for the frozen expansion target than as an unconstrained directional signal generator.
- The model must be benchmarked against the existing `atr_20` anchor and frozen linear ridge model; increased complexity is useful only if it adds stable Validation performance and economic usefulness.

## One-minute GC adaptation

Potential bounded hypothesis:

> A small, strongly regularized feedforward network may capture nonlinear interactions among a **predeclared** subset of causal GC state features and improve 60- or 180-minute expansion forecasts beyond the frozen anchor/ridge benchmark.

Adaptation requirements:

- Information ends at completed decision bar \(t\); theoretical entry remains the open of \(t+1\).
- Use only eligible London `[03:00, 06:00)` and New York `[07:00, 12:00)` observations and preserve the same-date 15:30 forced-exit boundary.
- Fit preprocessing, missing-value policy, feature selection, architecture selection, and early stopping only inside Development walk-forward folds with purging/embargo for overlapping labels.
- Keep London and New York models or diagnostics separate unless a pooled design is predeclared and session interactions are explicit.
- Start with a very small architecture and compare it with linear and constant baselines using the same rows and loss.
- Score both statistical ranking (daily/date-block IC and retention) and calibration/economic usefulness; do not infer a trade direction from expansion.
- Treat every architecture, epoch, seed, horizon, and loss variant as a trial in the research ledger.

## Unsuitable or deferred ideas

- Copying the 995-stock cross-sectional top/bottom portfolio logic to one GC stream.
- Random train/test splits, shuffled temporal folds, or early stopping on Validation/Final test.
- Importing the chapter's features or hyperparameters without a GC mechanism and predeclared contract.
- Using a large network to discover hundreds of feature interactions before a simpler benchmark earns escalation.
- Claiming reproducibility from a notebook alone without saved configs, environment lock, deterministic seeds, fold manifests, checkpoints, and artifact hashes.
- Treating the book's gross backtest statistics as evidence for Project 1.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Deep learning and hierarchical representations | 533-537 |
| Feedforward networks, activations, outputs, and losses | 538-541 |
| Regularization and optimization | 541-545 |
| NumPy forward/backpropagation implementation | 545-553 |
| TensorFlow/Keras, TensorBoard, PyTorch, FastAI | 554-561 |
| Daily-equity feature/model setup and rolling CV | 561-565 |
| Ensemble, Alphalens, Zipline backtest, and summary | 566-568 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF p. 549.
- **Confidence: high** for the chapter's argument, workflows, API examples, model grid, and reported metrics; these were text-extracted cleanly.
- **Confidence: high** for the equations transcribed above; PDF p. 549 was visually inspected.
- **Confidence: medium** for mathematical symbols on PDF pp. 538-544 because extraction omits some glyphs. No missing equation has been guessed.
- The book does not provide a multiple-testing-adjusted confidence interval for its architecture search or a costed version of the final strategy.
- The chapter does not establish whether the one-year holdout is representative across regimes.
- Project 1 applicability remains a research hypothesis; this summary authorizes no neural-network implementation or Final-test exposure.
