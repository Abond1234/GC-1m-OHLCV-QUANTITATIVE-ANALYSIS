# Chapter 18 - CNNs for Financial Time Series and Satellite Images

**Assigned source range:** PDF pages 569-606  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Convolutional neural networks encode locality, sparse connectivity, weight sharing, and optional downsampling. These assumptions make CNNs efficient for grid-like data: images, one-dimensional time series, multivariate time-feature grids, or other arrays in which nearby values have a meaningful relationship. The chapter develops convolutional building blocks, reviews image architectures and transfer learning, and then demonstrates one-dimensional and two-dimensional time-series models for return prediction (PDF pp. 569-576, 577-606).

The key limitation is structural: a CNN is useful only when neighborhood and translation assumptions are defensible. Arbitrarily reshaping indicators into an image can create synthetic locality rather than discover market structure. The chapter's own trading experiment is weak and sensitive to small design changes (PDF pp. 594-605).

## Concepts and definitions

- **Convolution/kernel:** a small learned filter is applied to overlapping local patches; each dot product becomes one activation in a feature map (PDF pp. 571-573).
- **Receptive field:** the contiguous region visible to a kernel at one layer (PDF pp. 571-573).
- **Sparse connectivity:** each activation depends on a local subset rather than every input.
- **Weight sharing:** the same kernel scans every location, reducing parameters and allowing a feature to be detected at different locations (PDF pp. 572-574).
- **Stride:** the scanning step size; smaller strides inspect more overlapping regions at greater cost (PDF p. 573).
- **Padding:** `valid`, `same`, `full`, and **causal** padding have different boundary behavior. Causal padding adds values only on the left so time-series outputs do not depend on later inputs (PDF pp. 573-574).
- **Detector/nonlinearity:** feature maps generally pass through ReLU or another nonlinear activation (PDF p. 574).
- **Pooling:** max, mean, or median downsampling reduces dimensionality and can add approximate translation invariance (PDF pp. 574-575).
- **Transfer learning:** reuse early layers from a pretrained image model and replace or fine-tune later layers for a new task, especially when labeled data are scarce (PDF pp. 575, 583-590).
- **CNN-TA:** a design that computes many technical indicators over many intervals, clusters similar rows/columns, and reshapes them into a two-dimensional grid for a CNN (PDF pp. 597-605).

## Formulas and notation

- For a local input patch \(X_{\text{patch}}\) and kernel \(K\), the chapter defines the convolution-stage activation as their flattened dot product:

  \[
  a=\operatorname{vec}(X_{\text{patch}})^T\operatorname{vec}(K)
  \]

  Figure 18.3 gives a binary \(3\times3\) example whose dot product equals 4 (PDF p. 573).

- ReLU detector:

  \[
  g(z)=\max(0,z)
  \]

  Reintroduced for the detector stage on PDF p. 574.

- The one-dimensional example uses an input of 12 lagged returns, 32 filters of kernel size 4, causal padding, max-pool size 4, flattening, batch normalization, and a linear scalar output (PDF pp. 596-597).

- The CNN-TA example forms a \(15\times15\) array: 15 selected indicators by 15 lookback intervals. This shape is a design choice, not a financial identity (PDF pp. 598-602).

## Assumptions

- Adjacent time steps or adjacent features contain local patterns relevant to the target (PDF pp. 570-574, 594).
- A learned pattern can be reused at other positions through shared weights.
- Pooling does not remove timing or magnitude information essential to the task.
- For a two-dimensional feature grid, row/column ordering gives true neighborhood meaning. The chapter uses hierarchical clustering to impose such an order, but this does not prove causal or stable locality (PDF pp. 600-602).
- Causal padding is required whenever a time-axis convolution could otherwise access future values (PDF p. 574).
- Transfer learning assumes lower image features from the source task are useful for the target; that assumption is not relevant to ordinary OHLCV unless the research input is genuinely image-like.

## Procedures described by the chapter

### Convolutional model design

1. Decide whether the input has a defensible one-, two-, or three-dimensional topology.
2. Choose kernel size, filter count, stride, padding, activation, pooling, and depth.
3. Keep time-axis operations causal.
4. Fit preprocessing on training data only.
5. Use rolling temporal validation, checkpoint epochs, and restore selected weights.
6. Compare predictive metrics and economic results against simple baselines (PDF pp. 572-576, 594-605).

### One-dimensional return example

1. Construct 12 lagged monthly returns and the next monthly return.
2. Apply one causal `Conv1D` layer, max pooling, flattening, batch normalization, and a linear output.
3. Train on rolling five-year windows and predict the next month repeatedly.
4. Use early stopping and evaluate IC (PDF pp. 594-597).

### Two-dimensional CNN-TA example

1. Compute 15 technical/risk indicators over 15 intervals.
2. Select indicators with mutual information estimated from a 100,000-row sample.
3. Standardize and hierarchically cluster indicator and interval dimensions.
4. Reorder the \(15\times15\) grid by dendrogram leaves.
5. Train a two-convolution model in rolling five-year train/three-month validation folds.
6. Ensemble selected epochs, inspect factor spreads, and backtest a daily long-short equity strategy (PDF pp. 598-605).

## Feature and model examples

- Technical inputs include WMA, EMA, ROC, CMO, Chaikin A/D oscillator, ADX, RSI, NATR, PPO, Bollinger Bands, and rolling Fama-French betas (PDF pp. 598-600).
- The one-dimensional model has 449 trainable parameters and predicts next-month return from 12 monthly lags (PDF pp. 596-597).
- The two-dimensional model has two \(3\times3\) convolutional layers with 16 and 32 filters, max pooling, dropout, a 32-unit dense layer, and 55,041 trainable parameters (PDF p. 602).
- Best CNN-TA epochs report a low daily average IC around 0.009 (PDF p. 603).
- The illustrative gross strategy reports a 35.6% cumulative return and a 0.53 Sharpe ratio before transaction costs (PDF p. 605).

## Statistical, validation, and backtesting warnings

- The one-dimensional example explicitly uses a positively biased early-stopping presentation and says the results are illustrative; cherry-picked epochs produce the displayed cumulative IC (PDF p. 597).
- Mutual-information feature selection on a random 100,000-row sample is computationally convenient but creates sampling and selection uncertainty (PDF p. 600).
- The CNN-TA result is not robust: small architecture or training changes can make performance materially worse or collapse predictions to a constant (PDF p. 605).
- The strategy statistics are before transaction costs (PDF p. 605).
- Indicator-window proliferation creates a large family of correlated trials; apparently consistent nearby filters or intervals are not independent confirmations.
- The two-dimensional arrangement is data-derived. If ordering is recomputed outside each training fold, validation information can leak into the representation.
- Pooling and symmetric padding can erase event timing or leak future observations; both require explicit tests on time-series data.

## Implementation patterns worth preserving

- Express time-series inputs as explicit tensors with named dimensions such as `sample x time x feature`.
- Use causal convolution or validated left-padding on every time axis.
- Build rolling windows only within a continuous contract/segment/session history; never bridge missing minutes or contract changes.
- Fit scalers, feature selectors, and clustering order inside each training fold.
- Save the grid order, input schema, kernel configuration, seed, fold boundaries, checkpoint, and library versions with each run.
- Add synthetic leakage tests that perturb future bars and prove earlier feature tensors do not change.
- Compare the CNN against a linear model and simple lag/volatility baselines on exactly the same rows.

## Dated APIs and examples

- TensorFlow 2/Keras `Conv1D`, `Conv2D`, `MaxPooling*`, `ImageDataGenerator`, and Functional/Sequential examples reflect the 2020 API; current serialization, optimizer, and input-shape behavior must be verified (PDF pp. 577-605).
- The Quandl Wiki equity dataset used by the examples is historical and is not a current Project 1 source.
- `pandas_datareader` Fama-French access, TA-Lib builds, SciPy clustering calls, and statsmodels `RollingOLS` should be version-pinned.
- Alphalens is used for factor spreads and is no substitute for the project's own costed, boundary-aware evaluator.
- Pretrained VGG16, DenseNet201, and related image architectures are discussed for image tasks (PDF pp. 583-590); they are not natural defaults for GC bars.

## Project 1 relevance

- A **small causal 1D CNN** is the only directly relevant CNN architecture for a first GC experiment.
- Project 1 has high-frequency sequential observations, but eligible decision bars are a sampled research population. Input windows must be rebuilt from trusted full-history GC bars and mapped to eligible observations without crossing continuity resets.
- Existing statistical research already shows strong volatility/expansion structure and no validated directional univariate alpha. A CNN challenger should first test whether local nonlinear temporal motifs add incremental expansion information beyond `atr_20` and the frozen ridge model.
- A 2D technical-indicator image would duplicate many correlated transformations and increase multiple-testing burden; it should be deferred until a 1D baseline earns escalation.

## One-minute GC adaptation

Potential bounded hypothesis:

> A causal 1D convolution over a short, predeclared sequence of normalized one-minute GC returns, ranges, and activity measures captures local volatility-transition motifs that improve 60- or 180-minute expansion ranking beyond the frozen baseline.

Minimum design:

- End every sequence at completed decision bar \(t\); prediction and label paths begin at \(t+1\).
- Candidate windows could be a small predeclared set tied to market mechanics, not a sweep over dozens of lengths.
- Channels should be scale-stable and causal, for example lagged return, true range/ATR ratio, log-volume surprise, close location in bar, and session-clock state.
- Reset sequences at nonconsecutive minutes, contract/instrument/continuous-segment changes, invalid tradability/roll boundaries, and research-day/session boundaries where the feature definition requires it.
- Train/evaluate London and New York separately or include a predeclared session interaction.
- Use Development-only walk-forward model selection with purging/embargo for overlapping 60/180-minute labels; touch Validation once per frozen candidate.
- Evaluate incremental date-block IC, calibration, stability by year/session, and resource cost. Final test remains locked.

## Unsuitable or deferred ideas

- Satellite-image transfer learning: no satellite or image source is in the current GC OHLCV scope.
- Symmetric or `same` time padding unless an automated test proves no right-side information is used.
- A \(15\times15\) indicator grid chosen after looking at GC outcomes.
- Treating nearby technical indicators as independent evidence.
- Pooling across the time axis when exact recency is economically important and not separately encoded.
- Daily-equity cross-sectional long-short construction or the chapter's gross performance claims.
- High-capacity ResNet/DenseNet architectures before a small causal CNN produces stable incremental validation evidence.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| CNN motivation and grid-like assumptions | 569-571 |
| Convolution, feature maps, stride, padding, detector, pooling | 572-575 |
| Architecture lessons and image use cases | 575-590 |
| Object detection example | 590-594 |
| 1D autoregressive CNN | 594-597 |
| CNN-TA indicators, feature selection, and grid construction | 597-602 |
| CNN training, ensembling, and backtest | 602-605 |
| Chapter summary | 606 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF p. 573.
- **Confidence: high** for the architecture concepts, causal-padding requirement, model structures, validation comments, and reported metrics.
- **Confidence: high** for Figure 18.3 and the convolution interpretation; PDF p. 573 was visually inspected.
- **Confidence: medium** for some displayed tensor dimensions because text extraction omits multiplication symbols; prose and visual structure were used conservatively.
- The book does not demonstrate that the clustered 2D grid order is stable across folds or regimes.
- The chapter does not adjust its model/epoch/indicator search for multiple testing.
- No evidence in this chapter establishes that CNN features improve one-minute GC direction or expansion. That remains a separate, predeclared Project 1 hypothesis.
