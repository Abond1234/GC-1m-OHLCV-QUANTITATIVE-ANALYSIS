# Independent backtest verification report

Durable record of the clean-room accuracy audit of the Section 10/11 statistical
signal-and-backtest chain. The audit answers one question before any new
strategy work is trusted: **did the recorded backtest simulate what the raw
market data actually permits?** Module: `src/statistical_research/backtest_verification.py`.
Runner: `scripts/run_backtest_verification.py`. Tests: `tests/test_backtest_verification.py`.

## Method

The verifier re-derives the entire backtest from the raw one-minute bars through
a **second, deliberately independent implementation** and reconciles it against
the recorded artifacts. It does not import the production simulator; a check that
reuses the code it audits proves nothing. Entry timestamps are mapped to bars by
a pandas index lookup (the production path uses `numpy.searchsorted`), and the
one-position chronological walk is re-implemented as an explicit bar-by-bar loop.

Three layers:

1. **Reconciliation** — every recorded trade's entry bar, exit bar, exit reason,
   holding time, stop size, ambiguity flag, and gross R, plus per-partition mean
   net R under all three cost scenarios.
2. **Physical-fidelity audits** independent of the simulator's own contract:
   fills equal raw bar opens; decision ATR is the raw rolling ATR at the bar
   strictly before entry (point-in-time sizing); stop/target fills are reachable
   inside the exit bar; close-based exits equal the raw close; no trade crosses a
   New York date, a continuous segment, or the 15:30 forced-exit minute.
3. **Cross-artifact triangulation** — entry price and decision ATR agree across
   three independently built tables (raw bars, signal candidates, forward labels).

## Result: the recorded backtest is faithful

All fifteen checks passed.

| Audit | Result |
| --- | --- |
| Independent vs recorded trades | 86,353 vs 86,353; all 4 variants reconciled; 0 field mismatches |
| Mean net R (4 variants x 2 partitions x 3 scenarios) | matches to floating-point round-off (max abs diff 0.0 at 1e-16) |
| Entry fills equal raw bar opens | 419,063 candidates, max error 0.0 |
| Sizing is point-in-time | decision ATR equals prior-bar raw ATR, max error 0.0; entry is always the next bar |
| Exit fills reachable | every exit reachable inside its bar |
| Boundary integrity | no trade crosses date, segment, or the holding cap |
| Gate flag consistency | flag equals `prediction >= session threshold` for all 419,063 rows |
| Cross-artifact triangulation | entry price and decision ATR agree across all three tables, max error 0.0 |
| Partition governance | Development and Validation only; no Final-test row |

## The one documented caveat: gap-through fills

33 of 86,353 trades (0.038%) exit on a bar that **gapped through** the stop or
target, so the recorded fill at the exact level is not a price the bar actually
traded. For the 18 gap-through stops the modelled fill is slightly optimistic (a
live fill would be at the gap, worse); for the 15 gap-through targets it is
slightly pessimistic. The effect is immaterial to the conclusions — all four
variants are rejected with negative net expectancy, and correcting the stops
would make them marginally more negative — but it is the correct refinement for
the execution phase: model gap-through fills at the bar open rather than the
level.

## What this certifies, and what it does not

- It certifies that the simulator faithfully implements its frozen Section 10/11
  contract and that the fills are consistent with the raw bars and with the
  independently built forward-label table. The rejected verdicts stand on
  accurate arithmetic.
- It does **not** re-open those verdicts, and it is not a signal-validity audit.
  One item is carried forward for the strategy lab: the expansion gate's
  `gate_prediction` is normalised per New York date across all intraday
  observations, so it is a research diagnostic, not a point-in-time live trigger.
  The gate flag was applied consistently (verified), but any future strategy that
  treats such a gate as tradeable must first re-derive it point-in-time.
