# Chapter 4 - Financial Feature Engineering: How to Research Alpha Factors

**Assigned source range:** PDF pages 115-152  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature, factor, model, or strategy is approved by this note.

## Main argument

Feature engineering for trading begins with an economic or market-mechanism hypothesis and ends with causal, cost-aware evidence. Alpha factors transform market, fundamental, or alternative data into one value per asset and decision time. Decades of momentum, value, volatility, size, and quality research provide useful hypotheses, but published anomalies are correlated, regime-dependent, vulnerable to false discovery, and often unsuitable for direct transfer to a different asset or horizon (PDF pp. 115-127).

The chapter then demonstrates a practical factor workflow: manipulate panel data with pandas and NumPy, build lagged features and forward-return labels, use technical-analysis functions, smooth noisy series with a Kalman filter or wavelets, simulate point-in-time factor calculations with Zipline, and evaluate quantile returns, information coefficient, and turnover with Alphalens (PDF pp. 127-151). The transferable value is the research structure and diagnostics. The particular US-equity datasets, platforms, APIs, thresholds, dates, and reported results are historical examples, not evidence for one-minute GC.

## Concepts and definitions

- **Alpha:** the portion of return not explained by the selected benchmark or modeled risk exposures; later in the chapter it is operationalized as average return in excess of a benchmark (PDF pp. 115, 147).
- **Alpha factor:** a transformation of raw data intended to predict asset-price movement. It produces one value per asset whenever the strategy evaluates the factor (PDF pp. 115-117).
- **Risk factor / priced factor:** a systematic driver of returns for which investors may receive compensation, often because performance is poor in adverse states (PDF pp. 116-117).
- **Rational explanation:** a premium compensates for economic or systematic risk (PDF pp. 116, 119, 122, 124-126).
- **Behavioral explanation:** persistent biases, underreaction, overreaction, extrapolation, loss aversion, herding, lottery preference, or constraints prevent immediate arbitrage (PDF pp. 116, 119, 121-126).
- **Momentum factor:** takes positive exposure to recent winners and negative exposure to recent losers, relying on return continuation over the stated horizon (PDF pp. 118-121).
- **Sentiment factor:** measures attention, opinion, revisions, positioning, or other investor beliefs, including analyst changes, short interest, news, and social data (PDF pp. 119-121).
- **Value factor:** compares market price with an estimate of fair or fundamental value and relies on eventual convergence or compensation for risk (PDF pp. 121-124).
- **Statistical arbitrage:** in this chapter, a family of systematic relative-value strategies often implemented as market-neutral long/short portfolios (PDF p. 121).
- **Low-volatility and size factors:** empirical relationships between returns and volatility, beta, idiosyncratic risk, or market capitalization (PDF pp. 124-125).
- **Quality factor:** a quantitative proxy for profitability, operating efficiency, financial strength, earnings stability, or governance quality (PDF pp. 125-127).
- **Feature incrementality:** predictive information added beyond known factors and existing features, rather than another correlated representation of the same effect (PDF pp. 117, 127).
- **Winsorization:** capping observations at selected distribution quantiles to limit extreme values; the example uses 1% and 99% full-sample cutoffs (PDF pp. 128-129).
- **Forward return:** a future holding-period return aligned with features at the current observation. It is an outcome/label, not a tradable feature (PDF pp. 129, 131, 144-145).
- **Bollinger Bands:** a simple moving average with upper and lower bands at selected rolling-standard-deviation multiples; the example uses two (PDF pp. 131-133).
- **Relative Strength Index (RSI):** a bounded price-change indicator comparing average upward and downward changes over a lookback, commonly interpreted with 70/30 thresholds (PDF pp. 119-120, 131-133).
- **Kalman filter:** a sequential linear-Gaussian state-space model that alternates prediction and measurement updates to infer a continuous hidden state from noisy observations (PDF pp. 133-136).
- **Wavelet transform:** a representation of a signal using scaled and translated finite-length waveforms; thresholding coefficients can create a smoother reconstruction (PDF pp. 137-138).
- **Event-driven backtest:** a simulator that advances through time and exposes only data available at each event. The chapter uses Zipline for factor calculation (PDF pp. 139-143).
- **Factor quantile:** a group formed by ranking factor values, such as five cross-sectional groups from lowest to highest (PDF pp. 140-150).
- **Information coefficient (IC):** Spearman rank correlation between factor values and subsequent returns for the target horizon (PDF pp. 147-149).
- **Information ratio (IR):** average benchmark-relative return divided by tracking risk (PDF p. 147).
- **Factor turnover:** the share of assets in a current factor quantile that were not in the corresponding prior-period quantile (PDF p. 150).
- **Factor rank autocorrelation:** correlation of asset ranks across dates; greater stability generally implies less rebalancing pressure (PDF p. 150).

## Formulas and notation

### Relative Strength Index

Let \(\Delta_p^{up}\) and \(\Delta_p^{down}\) denote the chapter's average price changes over rising-price and falling-price observations. The displayed formula is:

\[
\mathrm{RSI}
=
100-\frac{100}{1+\frac{\Delta_p^{up}}{\Delta_p^{down}}}
\]

(PDF p. 120; visually inspected). The page does not specify zero-denominator handling or the exact averaging convention; a production implementation must fix those details.

### Normalized multi-period return and momentum

After calculating and clipping an \(L\)-period percentage return \(r_{t,L}\), the code converts it to a geometric per-period return:

\[
g_{t,L}=(1+r_{t,L})^{1/L}-1
\]

for \(L\in\{1,2,3,6,9,12\}\) months (PDF pp. 128-129; the code on p. 129 was visually inspected).

The displayed code then defines:

\[
m_{t,L}=g_{t,L}-g_{t,1},
\qquad L\in\{2,3,6,9,12\},
\]

and:

\[
m_{t,3\_12}=g_{t,12}-g_{t,3}
\]

(PDF p. 129).

### Forward-return labels

For each ticker, a negative shift aligns a future normalized holding-period return with the current row:

\[
y_{t,L}=g_{t+L,L}
\]

as implemented by grouping on ticker and applying `.shift(-L)` (PDF pp. 129, 131). The notation summarizes the code's alignment; the future value must remain exclusively a label.

### Mean-reversion factor

The Zipline `MeanReversion` example computes the last 21-day return's z-score relative to a 252-row window of overlapping 21-day returns:

\[
z_{i,t}
=
\frac{r^{(21)}_{i,t}-\overline{r^{(21)}_{i,t-251:t}}}
{s\!\left(r^{(21)}_{i,t-251:t}\right)}
\]

(PDF pp. 139-141). The strategy interprets a high value as a short candidate and a low value as a long candidate. The implementation must define behavior when the standard deviation is zero or observations are missing.

### Kalman state-space model

The initial hidden state is:

\[
x_0\sim\mathcal{N}(\mu_0,\Sigma_0).
\]

The transition model printed on the page is:

\[
x_{t+1}=A_tx_t+b_t+\epsilon^1_{t+1},
\qquad
\epsilon^1_t\sim\mathcal{N}(0,Q).
\]

The observation model is:

\[
z_t=C_tx_t+d_t+\epsilon^2_t,
\qquad
\epsilon^2_t\sim\mathcal{N}(0,R).
\]

Here \(A\) and \(b\) are transition parameters, \(C\) and \(d\) are observation parameters, and \(Q\) and \(R\) are the corresponding noise covariances (PDF pp. 134-135; p. 135 was visually inspected).

### Information and turnover metrics

The chapter defines IC as period-wise Spearman rank correlation between factor values and forward returns:

\[
\mathrm{IC}_t
=
\rho_{\mathrm{Spearman}}\!\left(f_{i,t},R_{i,t\rightarrow t+h}\right).
\]

This symbolic form follows the chapter's verbal definition; the page does not print a separate Spearman formula (PDF pp. 147-149).

The displayed risk-adjusted IC definition is:

\[
\mathrm{Risk\text{-}adjusted\ IC}
=
\frac{\operatorname{mean}(\mathrm{IC})}
{\operatorname{std}(\mathrm{IC})}
\]

(PDF p. 149; visually inspected).

The chapter's verbal information-ratio definition is:

\[
\mathrm{IR}
=
\frac{\text{average return in excess of benchmark}}
{\text{tracking risk}}
\]

(PDF p. 147). It notes that using the risk-free rate as benchmark yields the Sharpe-ratio setting.

For current quantile membership \(Q_t\) and prior membership \(Q_{t-h}\), the turnover definition can be represented directly as:

\[
\mathrm{Turnover}_{t,h}
=
\frac{|Q_t\setminus Q_{t-h}|}{|Q_t|}
\]

(PDF p. 150; visually inspected). The book reports the mean of this quantity over dates.

The chapter mentions the fundamental law of active management but does not display its equation in the assigned range, so it is not retranscribed here.

## Assumptions

- A factor has an economic, behavioral, institutional, or microstructure rationale that applies to the target asset and horizon.
- Inputs are point-in-time accurate and available before the modeled decision.
- A historical transformation is computed by instrument and does not cross contract switches, session breaks, or invalid gaps.
- Lookback, clipping, smoothing, ranking, and threshold parameters are selected without using final-test outcomes.
- Forward returns are correctly aligned and never re-enter the feature matrix.
- Cross-sectional ranks are supported by a genuine contemporaneous universe; temporal ranks require a separate interpretation.
- Published factor evidence survives differences in asset class, sample period, liquidity, implementation, and cost.
- Kalman transition/observation models are sufficiently close to linear and their noise assumptions sufficiently useful for the application.
- Wavelet basis, decomposition level, boundary handling, and threshold are determined without future leakage.
- IC observations and hypothesis tests account for serial dependence, overlapping labels, repeated assets, and multiple comparisons.
- Portfolio turnover can be translated into realistic trades, fees, spread, slippage, and market impact.
- Missing and dropped observations are not informative in a way that biases the reported factor results.

## Factor categories and mechanisms

### Momentum and sentiment

The chapter links momentum to underreaction followed by extrapolation, fear and greed, slow-moving supply/demand, feedback between asset prices and the economy, and mechanical trading rules such as stop losses, portfolio insurance, dynamic hedging, and risk-parity rebalancing. It also notes that feedback horizons differ by asset class and that commodities can reverse as supply responds (PDF pp. 118-121).

Examples include past return, volatility-adjusted return, price-trend acceleration, distance from a 52-week high, analyst coverage and revisions, target-price changes, share-count changes, short interest, and text-derived sentiment (PDF pp. 119-121).

### Value and relative value

Value factors compare price with a fair-value anchor. The anchor can be absolute, relative to another asset, or a range; mispricing may reverse after behavioral overreaction, liquidity shocks, or slow supply/demand adjustment. Examples include cash-flow yield, free-cash-flow yield, cash return on invested capital, earnings yield, PEG, sales yield, book yield, dividend yield, and cross-asset relationships (PDF pp. 121-124).

The listed accounting ratios are equity examples. The transferable commodity concept is a justified relative-value anchor, not a stock valuation ratio copied onto GC.

### Volatility, size, and quality

The chapter presents low volatility as an empirical anomaly and discusses lottery preference, representativeness, overconfidence, short-sale asymmetry, and crisis behavior as possible explanations. Volatility can be measured with realized dispersion, beta, implied volatility, or correlations (PDF pp. 124-125).

Quality examples include profitability, asset turnover and its change, liquidity ratios, interest coverage, leverage, payout, return on equity, and accruals. Quality is multidimensional, correlated with other factors, and difficult to reduce objectively to one score (PDF pp. 125-127).

## Procedures described by the chapter

### Engineer panel features with pandas and NumPy

1. Load typed point-in-time data with date and asset identifiers.
2. select the analysis interval and reshape prices into a date-by-asset panel.
3. resample only under a declared calendar rule, using the last valid observation for each period when appropriate.
4. calculate percentage returns for specified lags.
5. cap outliers using declared quantiles and convert multi-period returns to comparable geometric per-period values.
6. reshape to a long asset-date index.
7. derive momentum differences and explicit calendar indicators.
8. group by asset before shifting lagged features.
9. create forward returns with negative shifts only in a separately identified target table.
10. assert ordering, horizon alignment, missingness, and train/test separation before modeling (PDF pp. 127-131).

### Estimate rolling factor exposures

The example retrieves monthly Fama-French factors, joins them to asset returns, and applies a 24-month rolling regression by ticker to estimate market, size, value, profitability, and investment betas. Such betas can become model features, but the factor series, return frequency, window, intercept, missingness, and estimation timing must be explicit (PDF p. 130).

### Compute technical indicators

The TA-Lib example loads adjusted AAPL OHLCV, computes 21-period Bollinger Bands at two standard deviations and a 14-period RSI, and plots the results with 70/30 reference lines. The subsequent chart is deliberately mixed: both indicators can label an advancing market “overbought” while price continues higher (PDF pp. 131-133).

### Filter a series sequentially with a Kalman model

1. Define the hidden state, observation, transition, and observation equations.
2. set initial state mean/covariance and process/measurement covariances.
3. update the prior state estimate as each observation arrives.
4. use the measurement update to weight prior and observed information according to uncertainty.
5. retain both the one-step prediction and filtered state for diagnostics.
6. compare against simple baselines such as past-only moving averages.
7. validate residuals, parameter sensitivity, lag, and robustness to nonlinearity and heavy tails (PDF pp. 133-136).

The example initializes scalar unit matrices, a zero state mean, observation covariance \(1\), and transition covariance \(0.01\), then filters the 2008-2009 S&P 500 close (PDF pp. 135-136). Those values are demonstration choices, not estimated Project 1 parameters.

### Apply wavelet denoising

1. select a wavelet family and boundary mode.
2. decompose the signal into approximation/detail coefficients.
3. apply a declared threshold rule to the detail coefficients.
4. reconstruct the series with the inverse transform.
5. verify reconstructed length, alignment, boundary effects, lag, and causal availability.
6. compare multiple thresholds only inside training data (PDF pp. 137-138).

The example applies a Daubechies-6 transform to 2008-2009 S&P 500 daily returns with soft thresholds proportional to the sample maximum. It is a retrospective visualization, not a causal trading feature (PDF pp. 137-138; p. 138 was visually inspected).

### Simulate a factor pipeline

1. ingest a point-in-time data bundle.
2. define a `CustomFactor` with explicit inputs and lookback.
3. apply a predeclared liquid-universe screen.
4. calculate long/short membership and an evaluation rank.
5. run the pipeline at the scheduled decision time.
6. record factor values and prices before placing orders or constructing a portfolio.
7. persist outputs needed for independent factor evaluation (PDF pp. 139-143).

The single-factor example ranks a rolling mean-reversion z-score and uses the 1,000 stocks with highest 30-day average dollar volume, selecting 25 at each extreme. The multi-factor example simply adds ranks for profitability, return on invested capital, EBITDA yield, mean reversion, price momentum, and StockTwits sentiment. The chapter correctly describes this equal-rank sum as naive because it ignores scale, correlation, incrementality, and relative predictive value (PDF pp. 140-143).

### Evaluate factor signal with quantiles, IC, and turnover

1. align each factor value with valid forward returns for declared holding periods.
2. report rows lost during return construction or binning.
3. form quantiles and inspect monotonicity of mean forward returns.
4. inspect cumulative and full return distributions, not only means.
5. calculate period-wise Spearman IC and summarize it through time and by regime.
6. calculate quantile turnover and rank autocorrelation as trading-intensity diagnostics.
7. apply dependence-aware inference and a multiple-testing correction.
8. translate the signal into a costed execution simulation before claiming strategy value (PDF pp. 143-150).

The example evaluates 5-, 10-, 21-, and 42-day horizons, reports 14.5% row loss in forward-return construction, finds limited distributional separation despite different means, and reports high turnover. It explicitly defers transaction costs and slippage (PDF pp. 144-150; pp. 145, 149, and 150 were visually inspected).

## Feature and model examples

- Past returns over several horizons and contrasts between long and short lookbacks (PDF pp. 118-120, 128-131).
- Price acceleration, distance from a trailing high, Bollinger position, and RSI (PDF pp. 119-120, 131-133).
- Calendar indicators such as year and month (PDF p. 131).
- Rolling exposures to common risk factors (PDF p. 130).
- Fundamental valuation and quality ratios for equity panels (PDF pp. 121-127).
- Kalman filtered state, one-step prediction, innovation, or time-varying parameter estimates when specified causally (PDF pp. 133-136).
- Wavelet coefficient or reconstruction features only when a causal implementation is demonstrated (PDF pp. 137-138).
- Cross-sectional quantile/rank transforms, universe screens, and equal-rank composites (PDF pp. 139-150).
- IC, turnover, and rank autocorrelation as diagnostics, not predictive input features (PDF pp. 147-150).

## Statistical, validation, and backtesting warnings

- **Factor zoo / false discovery:** hundreds of published factors and countless transformations make a plausible in-sample result easy to find. Economic intuition narrows the search but does not validate a result (PDF pp. 116-118, 127).
- **Cross-asset transfer:** equity momentum, value, size, and quality evidence does not establish the same mechanism or horizon in gold futures.
- **Correlated factors:** factor categories overlap. Summing ranks can double-count one latent effect and make apparent diversity illusory (PDF pp. 117, 125-127, 143).
- **Global winsorization:** the example calculates 1%/99% cutoffs on the complete stacked sample before a modeled split. Reusing this literally would leak future distribution information into earlier rows (PDF pp. 128-129).
- **Calendar resampling:** `.resample('M').last()` must respect the intended exchange calendar and missing periods; month-end equity examples do not define CME session boundaries (PDF p. 128).
- **Shift direction:** positive shifts create lagged inputs; negative shifts create future labels. A misplaced sign or unsorted index causes direct look-ahead (PDF pp. 129, 131).
- **Overlapping outcomes:** multi-period returns and daily signals for 5-42 day holding periods overlap heavily. Naive row-level \(t\)-tests overstate effective sample size (PDF pp. 144-149).
- **Quantile monotonicity:** mean differences can coexist with strongly overlapping distributions. One favorable extreme group or period is not a stable rank relationship (PDF pp. 145-147).
- **Dropped rows:** the example loses 14.5% during forward-return construction. Loss must be explained by horizon endpoints or missing prices and tested for selection bias (PDF p. 144).
- **P-value presentation:** p-values printed as `0` are rounded output, not literal zero probability. The chapter's simple one-sample \(t\)-test does not address serial/cross-sectional dependence or research multiplicity (PDF p. 149).
- **IC interpretation:** a small IC can be economically useful only with genuinely independent breadth and implementable turnover; minute rows and overlapping labels are not independent bets (PDF pp. 147-150).
- **Turnover and costs:** the example reports high quantile turnover and explicitly omits transaction costs and slippage. Predictive separation is not net profitability (PDF pp. 147, 150).
- **Technical-indicator semantics:** “overbought” and “oversold” are descriptions, not validated directional trades. The chapter's own AAPL example shows trend persistence after such readings (PDF pp. 131-133).
- **Kalman misspecification:** linear transitions and Gaussian, uncorrelated, constant-covariance noise are strong assumptions for financial returns (PDF pp. 134-136).
- **Filter versus smoother:** a forward Kalman filter can be causal; a two-sided smoother or parameter fit using the full series is not.
- **Wavelet leakage:** decomposing and reconstructing the complete series uses coefficients and boundary calculations influenced by later observations. The shown code cannot be inserted directly as a live feature (PDF pp. 137-138).
- **Wavelet threshold wording:** p. 138 says coefficients “above” a threshold are filtered out, while the shown `pywt.threshold(..., mode='soft')` operation shrinks coefficients and zeros values below the threshold magnitude. The prose and operation should not be treated as equivalent.
- **Universe and survivorship:** a present-day stock universe or vendor bundle can omit delisted names and change historical coverage.
- **Point-in-time fundamentals:** `.latest` and trailing aggregation require actual filing availability and revision handling, not only a quarter label (PDF pp. 142-143).
- **Execution timing:** factor value and completed-bar price must be known before the modeled order; recording both in one event does not establish a fill at that same price.
- **Three-year example:** the 2015-2017 factor result is an illustration with uneven annual IC, not a robust general conclusion (PDF pp. 141, 148-150).

## Implementation patterns worth preserving

- Start each factor with a research contract: mechanism, exact formula, source columns, lookback, sign, target, horizon, decision time, entry time, and advancement criteria.
- Keep feature and label builders separate and suffix future outcomes unambiguously.
- Apply rolling calculations within product, selected contract, and valid continuity segment.
- Fit clipping thresholds, scalers, Kalman parameters, and any learned transform on the training partition only; freeze or update them by a declared causal rule.
- Add unit tests with synthetic monotonic series, gaps, contract changes, zero variance, and known shifts.
- Record effective start time caused by each lookback and show coverage loss by feature.
- Compare every new factor with simpler transforms and the existing matrix to quantify incrementality.
- Use temporal folds with purging/embargo appropriate to the longest label horizon.
- Report IC by fold, year/session/regime, horizon, and product; include block-bootstrap or HAC uncertainty where dependence remains.
- Include turnover, rank stability, signal-change frequency, and a realistic friction schedule before economic promotion.
- Preserve notebook parameters, dependency versions, input hashes, output hashes, and a machine-readable run manifest.
- Fail loudly if a forward label appears in the feature frame or if a transformation crosses a prohibited boundary.

## Dated APIs and examples

- Quantopian has closed; its research environment, `quantopian.*` APIs, `QTradableStocksUS`, Morningstar integration, and hosted StockTwits data are unavailable as described (PDF pp. 139-143).
- The community Quandl WIKI US-equity bundle was discontinued/frozen and is not a current live equity feed.
- Zipline, Alphalens, and pyfolio have fragmented into maintained forks and compatibility variants. The book's interfaces and dependency environment must not be assumed current.
- `alphatools`, `pyfinance.PandasRollingOLS`, `pykalman`, TA-Lib Python bindings, PyWavelets, and HDF5 integration each require current compatibility and license checks.
- pandas aliases and behavior have evolved; examples such as `.resample('M')`, older `groupby.apply` assumptions, and pickle/HDF outputs may warn or behave differently in current pandas.
- `pandas-datareader` access to Fama-French data and its return/index conventions are network- and version-dependent.
- QuantConnect, WorldQuant, Alpha Trading Labs, PyAlgoTrade, `pybacktest`, `ultrafinance`, Trading with Python, and Interactive Brokers are a publication-era resource list; their present status and interfaces require current verification (PDF p. 151).
- TA-Lib indicator defaults are an implementation contract, not a universal mathematical definition. Version, warm-up behavior, missing handling, and output alignment must be pinned.

## Project 1 relevance

This chapter is the most directly relevant of Chapters 1-4 because Project 1 already has a statistical feature-research workflow. That existing work includes causal return, range, volume, volatility, VWAP/context, session, roll, and continuity controls. Any new notebook should reproduce those safeguards and audit feature overlap before adding another momentum, mean-reversion, dispersion, or smoothing transform.

The chapter's evaluation stack needs adaptation:

- GC/MGC are one or two related time series, not a broad contemporaneous equity cross-section.
- A rank across thousands of stocks becomes a **past-only temporal percentile**, regime-conditioned score, or a GC-versus-MGC relative measure only when the economic interpretation is declared.
- Information coefficient across time must use dependence-aware inference; the number of minute bars is not the breadth assumed by cross-sectional factor research.
- Contract roll, exchange sessions, maintenance gaps, and next-bar entry replace month-end equity conventions.
- Forward returns must follow the existing decision-at-\(t\), entry-no-earlier-than-\(t+1\) rule.
- Turnover becomes signal-state or target-position change frequency and must be costed against futures tick size, spread/slippage assumptions, commissions, and roll exclusions.

Nothing in the chapter rescues an incomplete or failed existing candidate. It supplies candidate transformations and validation diagnostics, not retrospective permission to promote a feature.

## One-minute GC adaptation

A bounded adaptation is a **causal factor-diagnostics extension** centered on one genuinely incremental hypothesis:

### Candidate hypothesis: normalized Kalman innovation

1. Define a scalar or small state-space model on a stable GC input, such as log price or return.
2. estimate every hyperparameter on development data only or update it through a declared one-sided rule.
3. at completed minute \(t\), retain the one-step prediction \(\hat z_{t|t-1}\), innovation \(e_t=z_t-\hat z_{t|t-1}\), and innovation variance.
4. form a standardized innovation using only the prior and current completed observation.
5. reset or flag contract switches, invalid gaps, and session boundaries according to the existing data contract.
6. test separate predeclared mechanisms: large innovations may mean short-horizon continuation under information arrival or mean reversion under transient dislocation.
7. evaluate forward direction, magnitude, and conditional dispersion separately at fixed horizons with entry no earlier than \(t+1\).
8. compare against simpler rolling z-score and return/range/volatility baselines.
9. report coverage, parameter sensitivity, IC, conditional effects, stability, multiplicity-adjusted inference, turnover, and cost-aware performance.

The candidate advances only if it adds stable out-of-sample information beyond existing features. A filtered price that merely resembles a moving average is not an incremental factor.

### Supporting diagnostic adaptation

- Replace cross-sectional quintiles with training-fitted temporal bins conditioned on time of day and, if justified, volatility regime.
- Compute temporal Spearman association and monotonic bin outcomes using purged folds.
- Translate factor turnover into changes in the proposed discrete signal or target position.
- Treat RSI, Bollinger, and lookback-return variants as controls because they overlap common existing return/volatility features.
- Defer wavelet inputs unless a strictly one-sided rolling implementation is demonstrated against an equal-lag simple filter.

## Unsuitable or deferred ideas

- Direct use of equity book, earnings, sales, dividend, market-cap, quality, or analyst factors for one-minute GC.
- Fama-French cross-sectional equity betas as GC minute features without a new cross-asset research contract.
- Copying 12-month equity momentum, 52-week-high, 21-day Bollinger, or 14-day RSI parameters to minute bars by changing “day” to “minute.”
- Quantopian/Quandl WIKI pipelines as the reproducibility foundation.
- Summing ranks of correlated factors without training-fold weighting and incrementality tests.
- Creating features with `.shift(-n)` or any forward-return column.
- Full-sample winsorization, scaling, Kalman estimation, wavelet decomposition, or threshold selection.
- Using the retrospective wavelet reconstruction on p. 138 as a live feature.
- Claiming alpha from IC alone while ignoring overlap, turnover, costs, slippage, and effective sample size.
- Treating GC and MGC duplicate economic exposure as independent cross-sectional breadth.
- Assuming every low RSI or upper/lower Bollinger touch implies a trade.
- Promoting a feature because a library function is standardized or because a published factor worked in equities.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Chapter objective and research-tool overview | 115 |
| Alpha-factor workflow, definitions, rationale, and factor proliferation | 116-117 |
| Momentum and sentiment mechanisms, metrics, and examples | 118-120 |
| Value and relative-value factors and equity ratios | 121-123 |
| Volatility, size, and quality factors | 124-127 |
| pandas/NumPy panel transformations, returns, lags, targets, and betas | 128-131 |
| TA-Lib, Bollinger Bands, RSI, and denoising transition | 132-133 |
| Kalman model, assumptions, equations, and example | 134-136 |
| Wavelet concepts, thresholding, and example | 137-138 |
| Zipline single- and multi-factor pipelines | 139-143 |
| Alphalens forward returns, quantiles, and return distributions | 144-147 |
| Information coefficient and factor-turnover diagnostics | 148-150 |
| Publication-era alternative libraries and chapter summary | 151-152 |

## Unresolved ambiguities and extraction confidence

**Extraction confidence: high for the chapter's factor taxonomy, code-level transformations, Kalman equations, and Alphalens diagnostics; medium for chart-derived conclusions and historical platform behavior; low for current API compatibility.**

- **Visual inspection pages:** PDF pp. 120, 129, 135, 138, 145, 149, 150.

The assigned range was read sequentially. The following actual PDF pages were also rendered and visually inspected:

- **p. 120:** displayed RSI equation and momentum/sentiment factor table.
- **p. 129:** multi-period return normalization, momentum differences, and shift code.
- **p. 135:** initial-state, transition, and observation equations for the Kalman model.
- **p. 138:** Daubechies plots, wavelet-threshold code, and denoised reconstructions.
- **p. 145:** sample Alphalens factor/forward-return rows and quantile-return chart.
- **p. 149:** IC-by-year chart and IC summary-statistics table.
- **p. 150:** quantile-turnover and factor-rank-autocorrelation tables.

Remaining ambiguities:

- The RSI page displays average rising and falling price changes but does not specify Wilder smoothing, simple averaging, treatment of zeros, or warm-up output. TA-Lib behavior must be treated as versioned implementation detail.
- The transition equation prints \(\epsilon^1_{t+1}\) in the state update but \(\epsilon^1_t\) in the distribution statement. The notation is preserved rather than silently normalized.
- The p. 138 prose says coefficients above the threshold are filtered, whereas the shown soft-threshold operation removes/shrinks small-magnitude coefficients. This is a substantive wording/code mismatch.
- Figure 4.3's surrounding text on p. 124 describes VIX using S&P 100 options, which appears inconsistent with conventional VIX terminology; no correction is substituted without a separate authoritative check.
- The Alphalens table prints some p-values as `0`; output precision is not shown.
- Factor-return charts summarize one historical US-equity sample. Exact values were not digitized beyond the printed tables because they are not Project 1 evidence.
- Text extraction occasionally wraps code tokens and mathematical glyphs. Rendered inspection resolved the equations and key tables listed above, but exact runnable historical code should still come from the companion repository and be modernized in a pinned environment.
