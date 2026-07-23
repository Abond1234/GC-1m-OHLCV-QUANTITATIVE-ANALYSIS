# Chapter 8 - The ML4T Workflow: From Model to Strategy Backtesting

**Assigned source range:** PDF pages 249-279  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; MLAT v1 does not authorize a trading-signal or strategy backtest.

## Main argument

An ML trading project is an end-to-end system, not an isolated prediction
score. Point-in-time data, feature engineering, model fitting, signal
formation, position sizing, order execution, accounting, and performance
evaluation must be sequenced consistently. A historical simulation is useful
only when its data, market mechanics, costs, timestamps, and statistical
interpretation are plausible enough to inform a future deployment decision
(PDF pp. 249-258).

The chapter contrasts fast vectorized calculations with structured
event-driven simulation. Vectorized backtests are convenient for rough checks
but provide few safeguards against timestamp mistakes and commonly omit order,
cost, portfolio, and risk state. Event-driven engines explicitly advance
through a calendar and broker lifecycle, but their structure cannot guarantee
correct inputs or assumptions (PDF pp. 255-262).

The backtrader and Zipline examples demonstrate the architecture, not a claim
that a particular engine makes a strategy valid. The most durable lesson for
Project 1 is to keep research stages separate and preserve a causal,
auditable handoff from features to later predictions and, only if authorized,
to execution (PDF pp. 259-279).

## Concepts and definitions

- **ML4T workflow:** source data, engineer factors, train/evaluate models,
  convert predictions into decisions, size positions, simulate trades, and
  evaluate performance (PDF pp. 249-250).
- **Backtest:** historical simulation of a trading algorithm intended to
  estimate behavior under future market conditions (PDF pp. 250-251).
- **Look-ahead bias:** use of information at a historical decision time before
  it was actually available (PDF pp. 251-252).
- **Survivorship bias:** evaluating only assets that remain available today
  while omitting historical failures, delistings, or acquisitions
  (PDF p. 252).
- **Mark to market:** repeatedly value open positions so interim drawdown,
  volatility, and risk constraints are represented (PDF p. 253).
- **Transaction costs:** commissions, spread, slippage, market impact, borrow,
  and other execution frictions (PDF pp. 253, 258).
- **Backtest overfitting:** selecting among many tested strategies or informed
  variants until sample noise appears successful (PDF pp. 253-254).
- **Deflated Sharpe ratio:** significance adjustment for repeated trials,
  non-normal returns, and sample length (PDF p. 254).
- **Vectorized backtest:** aligns signal/position arrays with future-return
  arrays and computes performance using bulk numerical operations
  (PDF pp. 255-257).
- **Event-driven backtest:** processes time-stamped data, strategy decisions,
  orders, fills, positions, and performance sequentially
  (PDF pp. 257-258).
- **Data feed / line / indicator:** backtrader objects representing a security
  stream, an individual field, and derived time-series computations
  (PDF pp. 259-261).
- **Broker and order lifecycle:** engine components that accept, fill, reject,
  and account for orders and resulting positions (PDF pp. 261-262).
- **Zipline bundle, calendar, and Pipeline:** point-in-time OHLCV storage,
  exchange timing, and modular factor computation within Zipline's
  event-driven architecture (PDF pp. 266-273).

## Formulas and notation

The chapter's simplest vectorized performance idea is elementwise signal
alignment:

\[
r^{\text{strategy}}_{t+1}=w_t^\top r_{t+1},
\]

where \(w_t\) must be fully determined at time \(t\) and the return begins only
after a feasible execution time. The displayed code shifts next-day returns
back for multiplication with current predictions (PDF pp. 255-256). This
identity is not a substitute for execution simulation.

For model monitoring, the chapter computes Spearman IC, RMSE, MAE, and a
long-minus-short return spread after the relevant forward return has
materialized (PDF pp. 277-278). The MLAT notebook adopts daily Spearman IC for
feature screening but does not compute a strategy return or Sharpe ratio.

The deflated Sharpe ratio and minimum-backtest-length results are referenced,
not fully derived in the assigned pages (PDF p. 254). They are therefore
governance warnings rather than formulas implemented in MLAT v1.

## Assumptions

- Every data field has a verified availability timestamp and point-in-time
  meaning (PDF pp. 251-252, 257-258).
- The historical universe and market calendar match what could actually have
  been traded at each time.
- Outlier treatment does not erase plausible extreme events simply to improve
  performance (PDF p. 252).
- The sample contains relevant volatility, liquidity, and market regimes and
  is long enough for the number of research trials (PDF pp. 252, 254).
- A signal computed from a bar close cannot fill at that same close; execution
  must occur at a subsequent feasible price (PDF p. 253).
- Costs, slippage, liquidity, order types, and operational hours are
  represented conservatively (PDF pp. 253, 258, 261-262).
- Model training and transforms use only information available before their
  prediction timestamp.
- The number of trials, strategy variants, and prior data exposure is part of
  the statistical evidence (PDF pp. 253-254).
- An engine's timestamp enforcement reduces implementation risk but does not
  prove the absence of leakage (PDF p. 257).

## Procedures described

1. Specify the investment universe, horizon, data sources, and causal feature
   pipeline.
2. Fit and evaluate an ML model on appropriately separated historical data.
3. Translate predictions into explicit trading rules and portfolio targets.
4. Sequence signal arrival, order placement, fills, position accounting, and
   mark-to-market valuation.
5. Apply realistic trading calendars, costs, liquidity, and order constraints.
6. Evaluate both model quality and portfolio performance through time.
7. Report the research-trial count and interpret performance in light of
   selection bias.
8. Use either a quick vectorized calculation or a more realistic event-driven
   architecture according to the decision at hand (PDF pp. 249-279).

For Project 1, only steps 1-2 are partially in scope: the notebook evaluates a
frozen feature batch, not a predictive model or trading strategy. Later steps
remain gated and must not be smuggled into feature research through informal
thresholding.

## Feature and model examples

- Long the ten highest positive ridge predictions and short the ten lowest
  negative predictions in a daily equity universe (PDF pp. 255-256).
- backtrader `Cerebro`, custom pandas feeds, a parameterized `Strategy`,
  market/limit/stop orders, target weights, commission logic, and pyfolio
  analysis (PDF pp. 259-265).
- Zipline bundles, calendars, Algorithm API, custom minute bundles, and the
  Pipeline API (PDF pp. 266-274).
- `DataFrameLoader` plus a one-period `CustomFactor` that converts stored model
  predictions into long/short rankings (PDF pp. 271-274).
- In-backtest model training with ranked pipeline factors,
  `StandardScaler`, and an SGD ridge model (PDF pp. 275-278).
- Out-of-sample model monitoring after labels mature, including IC, error
  metrics, and sign-conditioned spreads (PDF pp. 277-278).

These examples use cross-sectional equities. Their universe selection,
portfolio construction, and engine-specific code are not directly portable to
one-minute GC futures.

## Statistical, validation, and backtesting warnings

- Point-in-time errors in prices, fundamentals, corporate actions, or
  predictions create look-ahead bias (PDF pp. 251-252).
- Survivorship filtering can make historical performance look systematically
  better (PDF p. 252).
- Removing legitimate tail observations understates the environment a live
  strategy must survive (PDF p. 252).
- A favorable period may omit future-like regimes and cannot establish
  generalization (PDF p. 252).
- Close-derived signals evaluated at the same close use an infeasible fill
  (PDF p. 253). MLAT preserves t-close availability and t+1-open theoretical
  entry semantics.
- Ignoring spreads, commissions, slippage, impact, liquidity, and rejected
  orders biases strategy performance upward (PDF pp. 253, 258, 261-262).
- Repeated trials on the same sample produce selection bias even if individual
  backtests are implemented correctly (PDF pp. 253-254).
- Prior knowledge from others' trials on the same market history is also data
  exposure; a nominally new script does not restore an untouched holdout
  (PDF p. 254).
- Vectorized alignment offers no inherent look-ahead safeguard and has no
  broker/position state (PDF p. 257).
- Event-driven engines impose useful order but cannot guarantee that data,
  features, costs, or code are correct (PDF p. 257).
- A `cheat_on_open`-style option explicitly sees the next bar and biases
  results (PDF p. 261).
- Training inside a backtest can silently refit scalers or models on
  inappropriate windows unless the information set is tested precisely.
- Rolling Sharpe plots and a single cumulative-return path do not correct
  multiple testing or regime dependence.

## Implementation patterns worth preserving

- Encode an explicit event chronology: bar close, feature availability,
  decision, next feasible entry, label maturation, and evaluation.
- Separate immutable market inputs, computed features, predictions, orders,
  fills, and performance artifacts.
- Store exact data/schema hashes, code revision, seeds, contract version, and
  split boundaries with every run.
- Use a feature registry and modular causal computation independent of the
  later execution engine.
- Validate timestamps and continuity boundaries before computing any rolling
  indicator.
- Track missing warm-up periods explicitly rather than silently trading
  incomplete indicators.
- Evaluate predictions only after their forward outcomes mature.
- Keep model fitting separate from strategy search when this reduces repeated
  tuning and improves auditability.
- Carry costs and execution assumptions as versioned inputs when strategy
  testing is eventually authorized.
- Record all attempted variants, not only the chosen backtest.

## Dated APIs and examples

The chapter documents the 2018-era Quantopian Zipline stack, bcolz bundles,
deprecated pandas `Panel`, the discontinued Quandl Wiki bundle, old IEX
benchmark hooks, `pandas_datareader`, pyfolio, Alphalens, empyrical, and
backtrader APIs. Quantopian's hosted platform is no longer operating in the
form described. Zipline forks, calendars, storage formats, and pandas
compatibility have changed substantially.

The printed snippets also contain names and rank conditions that appear
internally inconsistent in extraction (for example, singular/plural position
variables and the vectorized top-ten mask). They are illustrative pseudocode,
not a tested Project 1 dependency.

## Project 1 relevance

This chapter is primarily a governance and future-handoff reference:

- it reinforces strict point-in-time feature construction and explicit
  t-close/t+1-open chronology;
- it supports freezing hypotheses and counting trials before outcome review;
- it explains why the historically exposed Final partition cannot be reused as
  an untouched confirmation set;
- it motivates machine-readable boundaries between features, predictions,
  trades, and performance;
- it confirms that a feature notebook should not quietly become a strategy
  backtest.

MLAT v1 therefore reports feature-level direction, expansion, volatility, and
path-risk evidence only. No position sizing, cost model, Sharpe ratio, or live
deployment claim is produced.

## One-minute GC adaptation

- Use only GC rows and exclude MGC from research and volatility fitting.
- Compute features on the full causal GC sequence, reset at contract/segment
  and declared tradability boundaries, then select eligible decision bars.
- Whitelist market columns so embedded forward labels cannot enter feature
  engineering.
- Preserve UTC timestamps plus New York session/date fields and verify DST
  behavior.
- Treat the feature value as available after the decision bar closes; any
  theoretical entry begins at the next bar open.
- Never equate the labeled next-open reference with an executed price or
  strategy P&L.
- Aggregate primary evidence by New York date and use date-block resampling,
  because one-minute observations are highly dependent.
- Keep Development and Validation chronological, apply Development-fitted
  transforms unchanged, and exclude the historical Final partition.
- Save missingness, boundary-reset, session, year, and overlap diagnostics as
  durable artifacts before any future strategy gate.

## Unsuitable or deferred ideas

- Vectorized or event-driven trading backtests are both out of scope for the
  current feature-research notebook.
- The chapter's long-ten/short-ten equity portfolio is unsuitable for a
  single-instrument GC dataset.
- In-backtest model training, live broker integration, position sizing,
  stop-loss rules, and execution optimization are deferred.
- Backtrader, Zipline, bcolz, pyfolio, Quantopian, and external data bundles are
  not added as Project 1 dependencies.
- Deflated Sharpe calculations are unnecessary because MLAT v1 produces no
  strategy Sharpe; trial control is enforced earlier through the frozen batch
  and BH-adjusted feature screens.
- Synthetic-market backtesting would require a separately frozen generative
  model and validation plan.
- The exposed Final set remains excluded; no engine can make it untouched
  again.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| End-to-end ML4T workflow | 249-250 |
| Point-in-time, survivorship, outliers, and sample-period pitfalls | 251-252 |
| Mark-to-market, costs, decision timing, and multiple testing | 253-254 |
| Vectorized backtest and its limitations | 255-257 |
| Event-driven requirements and factor/model handoffs | 257-258 |
| backtrader architecture and worked example | 259-265 |
| Zipline architecture, bundles, calendars, and Algorithm API | 266-271 |
| Pipeline API and stored-prediction backtest | 271-275 |
| In-backtest model training and monitoring | 275-279 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 250, 257, and 275.
- **Confidence: high** for the workflow, point-in-time and multiple-testing
  warnings, vectorized/event-driven distinction, and engine architecture after
  sequentially reading the full assigned range.
- **Confidence: high** for the seven-stage ML4T workflow on rendered PDF
  p. 250, the vectorized-backtest limitations on visually inspected p. 257,
  and the Zipline result/training flow on rendered p. 275.
- **Confidence: medium** for exact behavior of legacy backtrader/Zipline code,
  which depends on old library versions and includes apparent printed or
  extraction-level inconsistencies.
- The chapter does not define Project 1's continuity runs, GC contract-roll
  policy, intraday dependence correction, or multi-horizon target contract.
- The minimum-backtest-length and deflated-Sharpe formulas are referenced but
  not fully reproduced in the assigned pages.
- No chapter example establishes that its displayed strategy performance
  survives a truly untouched post-publication market sample.
