# Strategy laboratory: findings

Durable record of the multi-strategy search built on the independently verified
sequential-backtest engine. Modules: `src/statistical_research/strategy_lab.py`
(simulation) and `strategy_evaluation.py` (anti-overfitting statistics). Runners:
`scripts/run_strategy_lab.py`, `scripts/run_strategy_evaluation.py`. Development
and Validation only; the Final-test partition is untouched.

## What was tested

Twelve pre-registered directional strategies plus the two null benchmarks, run
over the eligible GC universe (419,063 observations; Development and Validation).
Each strategy is a fixed rule reading only causal registered features, with round
thresholds taken from feature marginals - never from their relationship to
outcomes - and all share the frozen Section 10 stop/target contract (1.5x ATR
stop, 2R target) so the search space is not inflated by exit tuning. The families
span trend (`momentum_60m`, `trend_efficiency_60`, `vwap_trend_60`,
`streak_follow`), reversion (`session_extreme_fade`, `vwap_reversion_20`,
`wick_reversal`, `choppiness_reversion`), breakout (`range_breakout`), volatility
(`vol_expansion_momentum`), session (`opening_drive`), and volume
(`volume_momentum`). 302,958 trades were simulated.

## Result: no directional edge survives

**0 of 12 strategies advance.** Every anti-overfitting lens agrees:

- **Benjamini-Hochberg q = 1.0** for all twelve on the Development one-sided test
  for positive mean net R - no evidence of positive edge after false-discovery
  control.
- **Validation bootstrap 95% intervals lie entirely below zero** for all twelve
  (2,000 date-block resamples), so the losses are not sampling noise.
- **The deflated Sharpe ratio is effectively zero** for every strategy, including
  the least-negative one, after correcting for trial count, sample length, skew,
  and kurtosis.
- **CSCV probability of backtest overfitting is 0**, with **zero probability that
  the in-sample-best strategy is profitable out of sample**. The relative ranking
  of strategies is stable - it is set by cost and trade-frequency structure, not
  by fragile alpha - but the stably-best strategy still loses.

## Why: a signal wall, then a cost wall

The frictionless (zero-cost) decomposition is decisive. In Development, per-trade
gross mean net R is approximately zero for every strategy - the best is +0.006 R
(`choppiness_reversion`), the worst -0.046 R, essentially all within +/-0.02 R of
zero. Switching costs off does not reveal a hidden edge, because there is none:
this is a signal problem, not a cost problem, and it reconfirms the Section 7
univariate finding at the multivariate strategy level.

Costs are then a hard second wall. The base scenario is 2.6 ticks round trip; on
the median ~10-tick stop that is ~0.26 R of drag per trade, and frequency
multiplies it - the always-long benchmark trades ~36 times per day. Win rates sit
near 33% for every strategy, the structural consequence of a 2R target rather than
evidence of a directional read.

## What this adds, and where edge could live

The value delivered is threefold: a backtest engine now *verified* trade-for-trade
against the raw bars; a reusable strategy-search harness with the overfitting
controls a funded-competition entry requires; and a decisive, evidence-based
negative result that steers capital away from intraday directional GC rules, which
are now rejected at both the univariate (Section 7) and multivariate (this search)
levels.

The evidence points the next research cycle away from direction:

- **Expansion / volatility timing.** Section 9 already found predictable structure
  in future *range* where direction has none. A strategy that profits from
  expansion regardless of sign (volatility-timed entries, or a straddle-like
  construction) is the evidence-supported direction, not another directional rule.
- **Exit structure is secondary.** Every rule here used a fixed 2R target. Because
  the gross edge is ~0, changing the target only redistributes outcomes; it cannot
  manufacture edge. Exit-mapping work (Section 12b) is worth revisiting only in
  service of a signal that first clears the gross-edge bar.
- **Horizon and instrument.** One-minute bars may be too noisy to carry
  directional information; a lower-frequency horizon and the MGC cost profile are
  worth scoping, but only against the same gross-edge-first discipline.

Any future strategy must be declared before it is run, and must clear a positive
gross edge *before* costs are debated. A gate specification for the expansion line
should be frozen in a research contract prior to computation.
