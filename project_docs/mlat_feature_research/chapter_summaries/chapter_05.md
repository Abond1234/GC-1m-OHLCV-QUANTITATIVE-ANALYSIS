# Chapter 5 - Portfolio Optimization and Performance Evaluation

**Assigned source range:** PDF pages 153-178  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature or trading policy is approved by this note.

## Main argument

An alpha estimate is not a strategy result. Signals must become positions, and
the joint return, covariance, turnover, constraints, and execution costs of
those positions determine portfolio performance. The chapter moves from
risk-adjusted metrics through mean-variance and alternative allocation methods,
then demonstrates trade simulation and performance analysis with Zipline,
PyPortfolioOpt, and pyfolio (PDF pp. 153-177). Its most durable lesson for this
project is that sophisticated optimization is fragile when expected returns or
covariances are estimated poorly, so simple benchmarks and genuine
out-of-sample evidence are essential (PDF pp. 158-167).

## Concepts and definitions

- **Excess return:** portfolio return less the matching risk-free return
  (PDF p. 154).
- **Sharpe ratio:** expected excess return per unit of excess-return
  volatility; its common square-root annualization assumes IID returns and can
  be wrong under serial dependence (PDF pp. 154-155).
- **Information ratio:** benchmark-relative alpha divided by tracking error
  (PDF pp. 155-156).
- **Information coefficient and breadth:** forecasting skill and the effective
  number of independent bets jointly constrain attainable active performance;
  correlated observations do not create genuine breadth (PDF pp. 156-157).
- **Efficient frontier:** portfolios that maximize expected return for a fixed
  risk level, or minimize risk for a fixed expected return (PDF pp. 157-161).
- **CAPM and alternative risk factors:** expected return may compensate
  systematic exposures rather than represent manager-specific alpha
  (PDF pp. 158-159).
- **Kelly criterion:** a log-wealth growth rule for sequential sizing whose
  theoretical optimum can imply leverage and is highly sensitive to estimated
  probabilities and returns (PDF pp. 163-165).
- **Risk parity/factor investing:** allocation based on risk contribution or
  underlying risk drivers rather than noisy expected-return estimates
  (PDF pp. 166-167).
- **Hierarchical risk parity:** clustering, covariance quasi-diagonalization,
  and recursive bisection reduce optimizer degrees of freedom without inverting
  the full covariance matrix (PDF pp. 166-167).
- **Walk-forward testing:** an out-of-sample period simulates performance on
  observations not used to tailor the strategy (PDF pp. 172-173).
- **Drawdown and downside metrics:** max drawdown, Calmar, Omega, Sortino, tail
  ratio, and VaR describe risks hidden by mean/variance summaries
  (PDF pp. 173-176).

## Formulas and notation

Rendered PDF p. 155 displays the population Sharpe definition as

\[
SR=\frac{\mu-R_f}{\sigma_{R^e}},
\qquad R^e=R-R_f,
\]

and the information ratio as

\[
IR=\frac{\text{alpha}}{\text{tracking error}}.
\]

Rendered PDF p. 159 gives the portfolio quadratic form
\(\omega^\top\Sigma\omega\) as the variance objective. The displayed notation
on that page alternates between \(\sigma_{PF}\) and \(\sigma_{PF}^2\); the
quadratic form is therefore recorded as an objective without silently
repairing the symbol.

Two source anomalies on rendered PDF p. 155 matter for reproducibility:

- the historical Sharpe estimator is printed with a variance-like
  \(\hat{\sigma}^{2}\) denominator even though the surrounding definition and
  prose require standard deviation;
- the prose prints \(\sqrt{12}\) for both monthly-to-annual and
  daily-to-monthly scaling.

These are treated as apparent typesetting errors, not formulas to implement.
For Project 1, no Sharpe statistic is authorized in this feature-only
experiment because no sequential policy return series is defined.

## Assumptions

- Portfolio weights, expected returns, covariance estimates, risk-free rates,
  and benchmarks are measured on aligned frequencies.
- Square-root time aggregation requires IID returns; autocorrelation requires
  an adjusted estimator (PDF p. 155).
- Mean-variance optimization assumes inputs are sufficiently stable and the
  covariance matrix is numerically usable (PDF pp. 159-162).
- Breadth means independent bets, not a raw count of overlapping minute rows
  (PDF pp. 156-157).
- Kelly sizing assumes the estimated outcome distribution is trustworthy and
  the objective truly is long-run log wealth (PDF pp. 163-165).
- Backtest statistics assume realistic order timing, costs, slippage, and
  position accounting (PDF pp. 167-173).
- Cross-sectional allocation methods require multiple investable assets;
  one-instrument GC discovery does not supply that universe.

## Procedures described by the chapter

1. Define returns, risk-free or benchmark series, and evaluation frequency
   before calculating risk-adjusted metrics (PDF pp. 154-156).
2. Use information coefficient and effective breadth as an analytical
   decomposition, while accounting for dependence among forecasts
   (PDF pp. 156-157).
3. Treat mean-variance optimization as a benchmark, compare it with equal
   weighting, minimum variance, risk parity, and more robust allocation rules,
   and judge them out of sample (PDF pp. 159-167).
4. Schedule signal generation and execution explicitly; check tradability and
   open orders, and configure commission and slippage assumptions
   (PDF pp. 167-170).
5. Separate in-sample and out-of-sample performance, then examine cumulative
   returns, drawdowns, rolling statistics, factor exposure, turnover, and event
   periods rather than relying on one ratio (PDF pp. 171-177).

For MLAT Feature Research, the adapted procedure stops before portfolio
construction: evaluate information content on Development and Validation,
report dependence-aware effect sizes, and authorize a policy only in a later
experiment.

## Feature and model examples

- Benchmark-relative and risk-adjusted performance measures
  (PDF pp. 154-157).
- Mean-variance, maximum-Sharpe, global-minimum-variance, equal-weight,
  Black-Litterman, Kelly, risk-parity, factor-risk, and hierarchical-risk
  portfolios (PDF pp. 157-167).
- A monthly mean-reversion equity factor translated into equal-weight or
  optimized long/short positions (PDF pp. 167-171).
- Walk-forward return cones, bootstrapped performance statistics, drawdown
  plots, factor exposure, and event-risk analysis (PDF pp. 171-177).

The chapter motivates separate upside/downside and volatility-state research,
but it does not provide a distinct causal one-minute GC predictor. The frozen
`realized_semivariance_balance_60` is a Project-original risk-state adaptation,
not a claim that the chapter specifies that exact formula.

## Statistical, validation, and backtesting warnings

- Sharpe significance and annualization can be invalid under serially
  correlated returns (PDF p. 155).
- The number of minute observations is not the number of independent bets;
  overlapping horizons inflate apparent breadth (PDF pp. 156-157).
- Expected returns are very difficult to estimate and optimized weights are
  extremely sensitive to estimation error (PDF pp. 158-162).
- Correlated assets make covariance inversion ill-conditioned; added
  diversification can paradoxically make optimizer output less reliable
  (PDF pp. 162, 166-167).
- A simple 1/N portfolio can outperform sophisticated optimizers out of sample
  because estimation error overwhelms theoretical gains (PDF p. 162).
- An attractive in-sample efficient frontier is backward-looking evidence, not
  a forward performance guarantee (PDF pp. 161-162).
- Transaction costs, slippage, turnover, leverage, and short constraints must
  enter any policy evaluation (PDF pp. 167-175).
- A favorable point estimate should be accompanied by uncertainty, drawdown,
  tail, stability, and regime diagnostics (PDF pp. 173-177).
- The reported example statistics on rendered PDF p. 175 describe one toy
  equity strategy and are not transferable evidence for GC.

## Implementation patterns worth preserving

- Keep signal timestamps, order scheduling, tradability checks, and execution
  timing explicit.
- Store gross exposure, turnover, transaction costs, benchmark alignment, and
  in/out-of-sample flags alongside any future policy results.
- Compare complexity to simple baselines and report optimizer convergence and
  conditioning.
- Use resampling or other dependence-aware uncertainty estimates.
- Separate predictive-feature validation from sizing and portfolio decisions.
- Treat every fitted allocation or calibration object as partition-scoped and
  apply it unchanged out of sample.

## Dated APIs and examples

The chapter uses the former Quantopian ecosystem: Zipline Pipeline,
Alphalens, pyfolio, and empyrical, plus historical versions of
PyPortfolioOpt/SciPy APIs (PDF pp. 159-177). Quantopian-hosted assumptions,
calendars, data bundles, and function signatures are dated. The conceptual
requirements—explicit scheduling, costs, walk-forward separation, robust
baselines, and multi-metric diagnostics—remain current.

## Project 1 relevance

The chapter is moderately relevant to governance and interpretation, but not
to direct feature construction:

- daily/date-block evidence is preferable to treating overlapping minute rows
  as independent;
- directional association below plausible tick costs is economically trivial;
- risk/expansion features may be useful for later sizing even with no signed
  return information;
- no Sharpe, Kelly fraction, or optimized position is meaningful until a
  sequential execution policy is separately frozen;
- `realized_semivariance_balance_60` should be interpreted as a risk-state
  descriptor, not as a portfolio recommendation.

## One-minute GC adaptation

- Use New York trading dates as the primary resampling units and explicitly
  thin overlapping horizons.
- Report signed-return economic effects in GC ticks as well as normalized
  statistical units.
- Evaluate expansion, volatility, and adverse-path targets independently of
  direction.
- Compare complex volatility estimates with simple ATR and realized-volatility
  anchors before considering sizing.
- Defer Kelly, risk parity, hierarchical allocation, and portfolio covariance
  methods; GC is the sole discovery instrument and MGC is locked.
- Do not infer strategy performance from feature IC. A later approved policy
  must model t-to-t+1 sequencing, costs, turnover, stops, exits, and forced
  closure.

## Unsuitable or deferred ideas

- Cross-sectional mean-variance, CAPM, factor-risk, HRP, and Black-Litterman
  allocation are not directly applicable to a single GC feature matrix.
- Kelly sizing is deferred until calibrated probabilities/distributions and a
  governed sequential policy exist.
- Zipline/Alphalens equity examples do not define CME futures execution.
- Pyfolio tear sheets and Sharpe comparisons are premature in an univariate
  feature experiment.
- Historical equity benchmark/event examples cannot substitute for GC
  Development/Validation evidence.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Chapter mandate and performance objectives | 153-154 |
| Sharpe, information ratio, skill and effective breadth | 154-157 |
| Portfolio theory, CAPM, and allocation menu | 157-159 |
| Mean-variance formulation and efficient-frontier example | 159-162 |
| Simple/robust alternatives, Kelly, risk parity, factors, HRP | 162-167 |
| Zipline scheduling, costs, and position construction | 167-171 |
| pyfolio inputs, walk-forward analysis, and diagnostics | 171-177 |
| Chapter conclusions and handoff | 177-178 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 155, 159, and 175.
- **Confidence: high** for the chapter flow, optimization warnings,
  performance-metric definitions, and implementation examples.
- **Confidence: high** for the Sharpe/IR display on PDF p. 155, the
  mean-variance display on PDF p. 159, and the performance table on PDF p. 175;
  these pages were visually inspected.
- **Confidence: medium** for the two apparent p. 155 typesetting anomalies and
  the inconsistent variance symbol on p. 159; they are preserved as warnings
  rather than silently corrected.
- The chapter does not establish a reliable effective sample-size estimator for
  overlapping intraday bets.
- It does not demonstrate that any portfolio method improves one-minute GC
  feature information.
