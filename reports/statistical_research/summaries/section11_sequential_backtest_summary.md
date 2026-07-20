# Section 11 Sequential Backtest Summary

**Status:** READY

## Scope and governance

A chronological, single-position, cost-aware research backtest of the Section 10 candidate family (two benchmark directions, with and without the frozen expansion gate) over Development + Validation. Rules frozen in `Section11Config` (seed 20260724): entries fill at the recorded next-bar open with the bar-open price verified for every trade; exits are stop, 2R target, 120-minute holding cap, or the forced 15:30 close; ambiguous bars (stop and target in one bar) are scored stop-first and flagged; trades never cross a New York date or continuous segment. Costs are declared versioned assumptions: 0.6 ticks commission round trip plus 1 tick slippage per side (base, 2.6 ticks total) and 2 ticks per side (pessimistic, 4.6 ticks), reported alongside a frictionless case.

Declared decision rule: a variant advances only with positive net base-scenario expectancy in both partitions.

## Result

| Variant | Dev mean net R | Val mean net R | Verdict |
|---|---:|---:|---|
| long, ungated | -0.224 | -0.192 | REJECTED |
| long, expansion-gated | -0.268 | -0.281 | REJECTED |
| short, ungated | -0.211 | -0.202 | REJECTED |
| short, expansion-gated | -0.260 | -0.207 | REJECTED |

86,353 trades across the four variants; ambiguous-bar rate 0.19 percent; win rates near 33 percent at the 2R target, which is what a directionless market produces, with costs then setting the sign. The expansion gate concentrates trades in high-opportunity periods but cannot supply direction, so gated variants are not better.

**Decision: the standalone statistical system is REJECTED.** No variant shows positive net expectancy in both partitions under base costs. This closes the Branch B standalone loop with the outcome the evidence supported since Section 7, and it is the "defensible decision that none qualify" the project treats as a valid Phase 2 result. The branch's validated asset - opportunity forecasting - transfers to the hybrid integration phase, where the POI event stream supplies candidate direction and the statistical models supply opportunity quality, sizing context, and no-trade filters.

## Verification

- 8/8 structural checks passed (entry prices verified against bar opens for all trades, one-position no-overlap invariant, holding cap, all variants simulated, cost monotonicity, verdict coverage).
- 13/13 new synthetic tests passed: hand-constructed bar paths for target, stop, ambiguous (conservative stop-first), time exit, forced 15:30 exit; short-side symmetry; one-position exclusion of overlapping candidates; gate exclusion; exact cost arithmetic in R; verdict rule requiring both partitions positive; entry-price mismatch and Final-test rejection.
- Full run: about 5 seconds for 86,353 trades. Full project suite green.

## Saved outputs (excluded from Git)

`backtest_trade_log_gc.parquet`, `backtest_performance_gc.parquet`, `backtest_yearly_performance_gc.parquet`, `backtest_equity_curves_gc.parquet`, `backtest_verdicts_gc.parquet`, and the `tables/section11/` CSVs.

## Exact next step

Branch B standalone research is complete through its sequential test. The next roadmap step is Section 12 integration research: POI events as direction candidates, statistical opportunity models as filters and sizing context, evaluated against the POI baseline alone. The Final test remains locked.

*Follow-up (2026-07-20): executed as declared - see `section12_hybrid_integration_summary.md`.*

SECTION 11 STATUS: READY
