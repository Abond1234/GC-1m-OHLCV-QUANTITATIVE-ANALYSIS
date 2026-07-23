# Appendix - Alpha Factor Library

**Assigned source range:** PDF pages 735-764  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; all indicators remain hypotheses until checked for duplication and validated under the Project 1 contract.

## Main argument

Feature engineering, preprocessing, and denoising often matter more than model novelty. The appendix provides a reference library of market-data indicators from TA-Lib and examples from the "101 Formulaic Alphas" literature, then compares univariate information coefficient/mutual information with multivariate feature importance, SHAP values, and portfolio analysis (PDF pp. 735-764).

The appendix is highly relevant to one-minute OHLCV research, but most examples use daily US equities, cross-sectional ranks, and holding periods of roughly 0.6-6.4 days. Every candidate therefore needs an economic reformulation, time-scale decision, continuity/reset rule, redundancy audit, and Project 1 validation. Indicator popularity or a published formula is not evidence of GC alpha (PDF pp. 735-736, 756-764).

## Concepts and definitions

- **Alpha factor:** a measurable transformation proposed to predict returns or another economically useful outcome.
- **Factor zoo:** the large body of reported equity characteristics and price metrics, many vulnerable to data mining and weak replication (PDF p. 735).
- **TA-Lib groups:** overlap, momentum, volume, volatility, price transform, cycle, mathematical, and statistical functions (PDF p. 736).
- **Moving-average family:** SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, and MAMA trade smoothness against reaction speed (PDF pp. 736-739).
- **Overlap/trend studies:** Bollinger Bands, moving averages, Hilbert trendline, and Parabolic SAR (PDF pp. 740-743).
- **Momentum family:** directional movement/ADX, Aroon, balance of power, CCI, MACD, StochRSI, stochastic oscillator, ultimate oscillator, Williams %R, RSI, and related measures (PDF pp. 743-752).
- **Volume/liquidity family:** Chaikin A/D, Chaikin oscillator, OBV, turnover/dollar-volume concepts, and Amihud illiquidity (PDF pp. 752-754).
- **Volatility family:** true range, ATR, and normalized ATR (PDF p. 754).
- **Formulaic alpha:** a nested computational expression built from cross-sectional and rolling time-series operators, often without a clear risk-based economic interpretation (PDF pp. 755-760).
- **Evaluation views:** IC and mutual information for bivariate association; gradient-boosting importance and SHAP for conditional relevance; quantile portfolios for economic performance (PDF pp. 760-764).

## Formulas and notation

The appendix uses \(P_t\) for closing price, \(V_t\) for volume, and \(r_t\) for the simple period return. Open, high, low, and close may use superscripts; trailing series span \(t-d\) through \(t\) (PDF p. 735).

### Moving averages

For window length \(N\):

\[
\operatorname{SMA}(N)_t
=\frac{P_{t-N+1}+\cdots+P_t}{N}
=\frac{1}{N}\sum_{i=1}^{N}P_{t-N+i}
\]

\[
\operatorname{EMA}(N)_t
=\alpha P_t+(1-\alpha)\operatorname{EMA}(N)_{t-1},
\qquad
\alpha=\frac{2}{N+1}
\]

(PDF pp. 737-738).

\[
\operatorname{WMA}(N)_t
=
\frac{P_{t-N+1}+2P_{t-N+2}+\cdots+NP_t}
{N(N+1)/2}
\]

\[
\operatorname{DEMA}(N)_t
=2\operatorname{EMA}(N)_t-\operatorname{EMA}_2(N)_t
\]

\[
\operatorname{TEMA}(N)_t
=3\left[
\operatorname{EMA}(N)_t-\operatorname{EMA}_2(N)_t
\right]
+\operatorname{EMA}_3(N)_t
\]

\[
\operatorname{TRIMA}(N)_t
=\frac{1}{N}
\sum_{i=1}^{N}\operatorname{SMA}(N)_{t-N+i}
\]

(PDF p. 738).

### Normalized Bollinger outputs

If \(U_t,M_t,L_t\) are upper, middle, and lower bands:

\[
\text{BB\_UP}_t=\frac{U_t}{P_t}-1,\qquad
\text{BB\_LOW}_t=\frac{L_t}{P_t}-1,\qquad
\text{BB\_SQUEEZE}_t=\frac{U_t-L_t}{P_t}
\]

These are the chapter's code-level normalizations (PDF pp. 740-742).

### Selected momentum/volatility identities

- Balance of power:

  \[
  \operatorname{BOP}_t
  =\frac{P_t^{\text{close}}-P_t^{\text{open}}}
  {P_t^{\text{high}}-P_t^{\text{low}}}
  \]

  (PDF p. 747).

- MACD:

  \[
  \operatorname{MACD}_t
  =\operatorname{EMA}_{12,t}-\operatorname{EMA}_{26,t}
  \]

  with signal line \(\operatorname{EMA}_9(\operatorname{MACD})\) and histogram equal to their difference (PDF pp. 748-749).

- True range:

  \[
  \operatorname{TRANGE}_t=
  \max\left[
  P_t^{\text{high}}-P_t^{\text{low}},
  \left|P_t^{\text{high}}-P_{t-1}^{\text{close}}\right|,
  \left|P_t^{\text{low}}-P_{t-1}^{\text{close}}\right|
  \right]
  \]

- Normalized ATR:

  \[
  \operatorname{NATR}_t=
  \frac{\operatorname{ATR}_t(T)}
  {P_t^{\text{close}}}\times100
  \]

  (PDF p. 754).

### Formulaic operators and examples

Common operators include lag, delta, rolling rank, mean, linearly weighted mean, sum, product, standard deviation, extrema, extrema locations, and rolling correlation (PDF pp. 756-758).

The printed Alpha 001 expression is:

```text
rank(ts_argmax(power(((returns < 0) ? ts_std(returns, 20) : close), 2.), 5))
```

(PDF p. 758). The subsequent docstring/code differ in sign/offset behavior; see unresolved ambiguities below.

Alpha 054 is:

\[
-\frac{
(P_t^{\text{low}}-P_t^{\text{close}})
(P_t^{\text{open}})^5
}{
(P_t^{\text{low}}-P_t^{\text{high}})
(P_t^{\text{close}})^5
}
\]

(PDF pp. 759-760).

## Assumptions

- Lookback-window statistics remain informative at the prediction horizon.
- Indicator thresholds and smoothing conventions transfer across assets/frequencies only after validation.
- OHLC bars are an adequate summary of within-bar trading and volume.
- Cumulative volume indicators have a meaningful reset origin.
- Technical-feature denominators are nonzero and handle zero-range bars explicitly.
- Cross-sectional ranks assume a contemporaneous asset universe; this is not available for a single GC stream.
- Formulaic expressions can be evaluated without an economic story, but their false-discovery burden is correspondingly high (PDF p. 756).

## Procedures described by the appendix

1. Select a liquid universe and compute market-data indicators.
2. Normalize price-level indicators for comparability.
3. Implement reusable rolling and cross-sectional operators.
4. Compute conventional and formulaic features.
5. Evaluate bivariate IC and mutual information.
6. Fit a multivariate model and inspect feature importance/SHAP values.
7. Evaluate quantile portfolio performance and cumulative long-short returns (PDF pp. 735-764).

The appendix's equity implementation uses 500 heavily traded stocks over 2007-2016, approximately 130 factors, a LightGBM model with early stopping, and Alphalens for selected factors (PDF pp. 736, 758-764).

## Feature and model examples

### High-relevance OHLCV families for GC hypothesis generation

- volatility level and normalization: TRANGE, ATR, NATR;
- range compression/expansion: normalized Bollinger width;
- directional efficiency/trend state: ADX/DX family, Aroon, MA separation or slope;
- within-bar pressure/location: BOP and close position in the bar;
- momentum/oscillation: ROC, RSI, MACD, stochastic variants, Williams %R;
- price-volume state: A/D, ADOSC, OBV, dollar volume, and Amihud-like activity ratios;
- rolling operators: delta, rank, extrema timing, correlation, weighted mean, and standard deviation.

These are candidate families, not a recommendation to materialize the full TA-Lib library.

### Reported appendix examples

- Alpha 001: IC about -0.0099 and MI 0.0129 on the equity sample (PDF p. 759).
- Alpha 054: IC about 0.025 and MI about 0.005 (PDF p. 760).
- A LightGBM model using roughly 130 factors prints global IC 3.40 and mean
  daily IC 2.01 on the final year (PDF p. 761). These cannot be raw Spearman
  correlations because IC is bounded to [-1, 1]; they appear to be percentage
  points (approximately 0.0340 and 0.0201), but the unit is not explicit and
  the values are treated as ambiguous rather than silently rescaled.
- SHAP and conventional feature-importance rank correlation is 0.89; MI and IC feature rankings correlate only 0.16 (PDF pp. 761-762).
- Alpha 054 shows about a 1.5 bps daily top-bottom spread, yet its cumulative long-short return is negative (PDF pp. 763-764).

## Statistical, validation, and backtesting warnings

- The appendix explicitly warns that large-scale signal mining creates multiple-testing bias and false discoveries (PDF p. 756).
- More than 60 candlestick pattern functions are set aside because published predictive evidence is mixed (PDF p. 736).
- Indicator families are highly redundant; many differ mainly by smoothing or lookback.
- MI is computed on a 100,000-observation sample for cost reasons and appears sensitive to sample size (PDF pp. 760, 762-763).
- IC, MI, SHAP, and split-based importance answer different questions and can rank features very differently (PDF pp. 760-763).
- A statistically significant IC can be economically negligible or unstable.
- A favorable average quantile spread can coexist with a losing cumulative long-short path, as Alpha 054 demonstrates (PDF pp. 763-764).
- The appendix's thresholds such as RSI 70/30 or ADX bands are trading conventions, not validated universal laws.
- Applying many windows, transforms, sessions, outcomes, and horizons multiplies the effective trial count.

## Implementation patterns worth preserving

- Centralize rolling primitives with explicit minimum history, null, zero-denominator, dtype, and reset behavior.
- Prefer vectorized NumPy/pandas/Numba implementations over repeated per-feature groupby scans.
- Keep a registry containing formula, inputs, units, lookback, lag, reset scope, directionality, rationale, and source pages.
- Fit normalizations and bins on Development only.
- Add equivalence tests against TA-Lib for selected nonpathological fixtures if a custom implementation is used.
- Write edge-case tests for flat bars, zero volume, contract changes, missing minutes, warm-up, and session starts.
- Materialize only the predeclared candidate batch, not the full indicator zoo.
- Evaluate redundancy before incremental modeling.

## Dated APIs and examples

- TA-Lib names and numeric moving-average codes are presented on PDF pp. 736-737; library availability and exact smoothing/warm-up conventions must be pinned and tested.
- pandas behavior for `groupby.apply`, rolling correlation, ranking, and chained assignment has changed since the book's code.
- statsmodels `RollingOLS`, `pandas_datareader`, LightGBM, SHAP, and Alphalens examples reflect a 2020 stack.
- The formulaic-alpha notebooks use daily equity matrices with tickers in columns. This layout and cross-sectional API do not map directly to Project 1's single-instrument event population.

## Project 1 relevance

This is the highest-relevance appendix in the assigned range, but it overlaps heavily with completed statistical research.

- `atr_20` is already the frozen expansion anchor; ATR/NATR should not be reintroduced under a new name.
- Existing feature families already cover returns, ranges, realized volatility, trend/efficiency, volume/activity, VWAP/session location, and context. Every appendix-inspired formula needs a registry-level duplicate and correlation audit first.
- The prior notebook found no validated directional feature and strong expansion structure. Oscillator values must not be relabeled as directional alpha without new evidence.
- Formulaic operators are useful implementation primitives, but the 101 equity alphas rely heavily on cross-sectional ranks and daily universes unavailable in Phase 1 GC.
- The book's volume indicators may require research-day or execution-session resets for futures; an all-history cumulative series would make level depend on arbitrary start date and contract history.

## One-minute GC adaptation

Potential controlled feature families, only after duplication review:

1. **Range compression:** normalized Bollinger width or a simpler rolling dispersion ratio, tested as an expansion hypothesis.
2. **Trend-strength state:** a carefully implemented ADX/DX-like statistic or MA-slope acceleration, normalized and compared with existing trend-efficiency features.
3. **Within-bar pressure:** BOP/close-location variants with a zero-range policy, tested for both direction and expansion but expected to face high noise.
4. **Price-volume divergence:** session-reset A/D/OBV-style state or rolling price-volume correlation, normalized by time-of-day activity.
5. **Extrema recency:** causal time since trailing high/low, compared with existing session-location and trend features.

Adaptation rules:

- Interpret \(N\) as a predeclared number of consecutive one-minute bars, not a mechanical conversion from daily windows.
- Use only history through completed bar \(t\); entry and labels start at \(t+1\).
- Preserve raw tradable GC prices and reset at continuity/contract/instrument/segment boundaries.
- Make research-day, London, and New York reset semantics explicit for cumulative/session indicators.
- Normalize volume by a Development-fitted time-of-day baseline; do not compare 08:30 volume directly with 04:15 volume.
- Handle zero range/volume without arbitrary tiny constants that can create extreme artifacts.
- Evaluate the fixed 5/15/30/60/120/180-minute labels under the existing Development/Validation governance and related-test correction.
- Require incremental value beyond the existing frozen anchor/feature set before advancement.

## Unsuitable or deferred ideas

- Materializing all 150+ TA-Lib indicators or all computable 101 alphas.
- Candlestick-pattern functions given mixed evidence and enormous multiple-testing burden.
- Equity fundamental/style factors for an OHLCV-only GC experiment.
- Cross-sectional ranks across different GC timestamps.
- Approximate VWAP as the simple mean of open/high/low/close; Project 1 should use its trusted volume/price fields and existing VWAP definitions.
- Copying daily windows, RSI thresholds, or average holding periods directly to one-minute futures.
- Infinite-history OBV/A/D without a defined reset.
- Treating IC/MI significance or SHAP rank as sufficient evidence of a costed trade.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Purpose, notation, factor zoo, TA-Lib scope | 735-736 |
| Moving-average families | 736-739 |
| Bollinger Bands and Parabolic SAR | 740-743 |
| Momentum and directional indicators | 743-752 |
| Volume/liquidity indicators | 752-754 |
| ATR/NATR and fundamental factors | 754-755 |
| Formulaic-alpha operators and inputs | 755-758 |
| Alpha 001 and Alpha 054 | 758-760 |
| IC, MI, LightGBM, SHAP, Alphalens evaluation | 760-764 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 737, 738, 754, 759, and 760.
- **Confidence: high** for the topic map, indicator families, code examples, reported metrics, and evaluation warnings.
- **Confidence: high** for moving-average formulas (PDF pp. 737-738), ATR/NATR (PDF p. 754), and Alpha 054 (PDF pp. 759-760); these pages were visually inspected.
- **Confidence: medium** for several displayed oscillator equations on PDF pp. 743-753 because text extraction omits mathematical glyphs. They are described conceptually rather than reconstructed.
- PDF p. 738 says "DEMA needs \(3N-2\)" immediately after the TEMA formula; this appears to be a naming typo and must not be propagated into implementation.
- Alpha 001 has an apparent mismatch among the printed expression, prose, docstring, and Python operation on PDF pp. 758-759. A Project 1 implementation would need to consult the primary paper/repository and freeze one audited definition.
- PDF p. 754 describes ATR as an SMA of true range, while library implementations may have their own Wilder-style smoothing/warm-up conventions. Exact TA-Lib parity must be tested rather than assumed.
- No appendix feature is approved until checked against the existing Project 1 registry and evaluated under a new predeclared contract.
