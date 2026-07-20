# Section 8 POI Sequential Backtest Summary

**Status:** READY — **S7P02_NY_BEAR_CONT is SEQUENTIAL_REJECTED at base costs**

## Scope and governance

The frozen S7P02 policy (its registry row states "Section 8 must sequence") executed as a chronological, single-position, cost-aware trade sequence. Entry/stop prices come from the frozen Section 7 machinery (`build_stop_policy_opportunities`: boundary-touch entry at the first-contact edge, volatility-hybrid stop); eligibility follows the registry exactly (bearish, New York, case-2 geometry, compressed 15-minute approach at the Development-fitted 40th-percentile threshold, recomputed deterministically). Sequencing rules mirror the verified Section 11 simulator: conservative stop-first ambiguity, 3R target, 240-minute cap, 15:30 forced exit, one position, entry-realism check (the edge price must have traded inside the retest bar; 4,227 unfillable rows dropped and counted). Costs: frictionless / base 2.6 ticks / pessimistic 4.6 ticks round trip. The declared rule: advancement requires positive net base expectancy in Development and Validation; the Final test is read once after the verdict is fixed. This run supplies the Option 1 evidence requested by `project_docs/section8_authorization_memo.md`.

## Result

20,780 eligible opportunities sequenced into 3,766 trades (505 Development, 382 Validation, 2,879 Final test).

| Scenario | Dev mean net R | Val mean net R | Final test (one-time read) |
|---|---:|---:|---:|
| Frictionless | +0.138 | +0.117 | +0.063 |
| Base | -0.003 | -0.035 | -0.059 |
| Pessimistic | -0.112 | -0.152 | -0.153 |

- Frictionless expectancy faithfully reproduces the Section 7 event-level findings, confirming the sequencer.
- **Base costs flip every partition negative.** The edge is real but thinner than a 2.6-tick toll against a ~10-30 tick stop. Median trade is a stop-out in every partition; win rates 27-30 percent at 3R.
- Yearly: 2021 strongly negative (-0.36), 2022-23 mildly positive, 2024-26 fading - the runner-dependence the Section 7 record warned about.

**Verdict: SEQUENTIAL_REJECTED.** PRD Phase 2 for Branch A is closed with sequential-trade evidence: the raw S7P02 family does not survive realistic execution. Per the memo's linkage, the family's remaining path is the frozen Section 12B contract (opportunity-conditioned sizing/exits/suppression); if 12B fails to improve it, Option 3 (archive) applies with this evidence attached.

## Verification

6/6 structural checks (one-position invariant, chronological entries, holding cap, entry realism, cost monotonicity, dev/val-only verdict); 6/6 new synthetic tests on hand-built bar paths (3R target, ambiguous conservative stop-first, unfillable entry dropped, one-position exclusion, exact cost arithmetic, verdict rule); full suite green. Runtime about 4 seconds.

## Saved outputs (excluded from Git)

`section8_poi_trade_log_gc.parquet`, `section8_poi_performance_gc.parquet`, `section8_poi_yearly_performance_gc.parquet`.

SECTION 8 POI BACKTEST STATUS: READY
