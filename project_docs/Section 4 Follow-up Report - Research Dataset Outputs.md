# Section 4 Follow-up Report - Research Dataset Outputs

## Purpose

This report explains the outputs produced by Section 4.0 of the exploration notebook:

```text
Research Dataset Construction & Continuous Active Contract Series
```

The purpose of Section 4 was not to create a trading strategy. It was to turn the cleaned multi-contract GC/MGC futures data into a research-ready active/front-month dataset that can support later POI strategy formalization, signal studies, and backtesting.

The most important final output is:

```text
research_bars
```

This should be the main dataset used from Section 5 onward.

---

## 1. Active Contract Selection Rules

### Output

```text
active_contract_selection_rules
```

The rule parameters were:

```text
rolling_window_days: 5
min_raw_winner_persistence_days: 2
min_candidate_daily_volume_share: 0.50
min_candidate_to_current_rolling_volume_ratio: 1.10
strong_candidate_daily_volume_share: 0.80
max_current_daily_volume_share_for_fast_switch: 0.20
```

### What Was Done

We defined a liquidity-based rule for choosing the active contract each day for GC and MGC separately.

The rule compares:

```text
1. Raw daily volume winner
2. 5-day rolling volume winner
3. Previously selected active contract
```

The final rule allows two types of roll confirmation:

```text
Normal roll:
The new rolling-volume winner persists for at least 2 product-days, has at least 50% of product volume, and has at least a 1.10x rolling-volume lead.

Decisive same-day roll:
The new rolling-volume winner has at least 80% of product volume, the old contract has no more than 20%, and the new contract has at least a 1.10x rolling-volume lead.
```

### Why It Matters

This prevents two major research errors:

```text
1. Rolling too early because of one noisy volume day.
2. Staying too long in an old contract after liquidity has clearly moved.
```

Liquidity-based selection is preferred because the active tradable contract is determined by market participation, not by a hard-coded calendar assumption.

### How We Use This To Our Advantage

Future signal research will operate on the contract where real trading activity is concentrated. That improves the realism of:

```text
entries
exits
volume filters
slippage assumptions
forward returns
session analysis
backtest diagnostics
```

---

## 2. Raw Rolling-Winner Diagnostics

### Output

```text
active_contract_candidate_quality
```

Results:

```text
GC product-days: 1,555
MGC product-days: 1,555
GC raw rolling contracts selected: 26
MGC raw rolling contracts selected: 26

GC median raw rolling-winner daily share: 97.01%
MGC median raw rolling-winner daily share: 97.49%

GC mean raw rolling-winner daily share: 93.35%
MGC mean raw rolling-winner daily share: 93.16%

GC minimum raw rolling-winner daily share: 0.29%
MGC minimum raw rolling-winner daily share: 0.28%
```

### What Was Done

The notebook calculated the highest 5-day rolling-volume contract for each product and date.

### What The Results Mean

Most of the time, the rolling-volume winner is also overwhelmingly dominant on the current day. Median daily volume share is around 97% for both GC and MGC.

The very low minimum values are important. They occur around rollover or unusual transition days, when the rolling-volume winner can still be mathematically ahead on a 5-day basis even though same-day liquidity may be temporarily distorted.

### How We Use This To Our Advantage

The high median dominance confirms that liquidity-based active-contract selection is reliable most of the time.

The low-minimum edge cases justify the extra rollover safeguards. Those edge cases are exactly why the final dataset needs rollover flags instead of blindly trusting a rolling-volume winner.

---

## 3. Final Active Contract Schedule

### Output

```text
active_contract_daily
active_contract_selection_summary
```

Results:

```text
GC product-days: 1,555
MGC product-days: 1,555

GC final active contracts: 26
MGC final active contracts: 26

GC confirmed switches: 25
MGC confirmed switches: 25

GC ambiguous roll days: 6
MGC ambiguous roll days: 4

GC median selected volume share: 97.01%
MGC median selected volume share: 97.48%

GC mean selected volume share: 93.01%
MGC mean selected volume share: 92.93%
```

### What Was Done

We created the daily active-contract table. It includes:

```text
product
trade_date
active_symbol
previous_active_symbol
total_product_volume
selected_contract_volume
selected_contract_volume_share
active_contracts
raw_daily_volume_winner
rolling_volume_winner
final_selected_contract
selection_method
roll diagnostics
tradable research flags
```

### What The Results Mean

The final rule selected 26 active contracts for each product across the full dataset. That implies 25 transitions per product, which matches the expected multi-year futures roll pattern.

The selected contract captures the dominant volume most of the time:

```text
GC median selected volume share: about 97.0%
MGC median selected volume share: about 97.5%
```

The ambiguous roll days are rare:

```text
GC: 6 days
MGC: 4 days
```

These days are not ignored. They are flagged for later exclusion or special analysis.

### How We Use This To Our Advantage

`active_contract_daily` becomes the control table for the entire research pipeline. It lets us:

```text
recreate the active-front series
audit every roll
filter ambiguous transition days
merge selected-contract volume share into minute bars
avoid using thin contracts in signal research
```

---

## 4. Selection Method Counts

### Output

Selection method counts from `active_contract_daily`.

Results:

```text
GC:
same_contract: 1,523 days
decisive_same_day_liquidity_migration: 19 days
confirmed_liquidity_migration: 6 days
retain_previous_contract_ambiguous_roll: 6 days
initial_selection: 1 day

MGC:
same_contract: 1,525 days
decisive_same_day_liquidity_migration: 21 days
confirmed_liquidity_migration: 4 days
retain_previous_contract_ambiguous_roll: 4 days
initial_selection: 1 day
```

### What The Results Mean

Most days do not involve any selection change. That is good. The active contract is stable.

Most rolls were decisive same-day liquidity migrations:

```text
GC: 19 of 25 rolls
MGC: 21 of 25 rolls
```

This means that when liquidity moved, it often moved clearly and quickly.

### How We Use This To Our Advantage

This tells us rollover behavior is usually clean enough to automate, but not clean enough to ignore. The small number of ambiguous days can be treated conservatively in strategy research.

---

## 5. Active Contract Validation

### Output

```text
active_contract_daily_validation
```

Results:

```text
Rows: 3,110
Products: 2
Duplicate product-dates: 0
Spread symbols selected: 0
GC/MGC symbol mismatch: 0
Missing selected volume: 0
Low-share days below 20%: 71
```

### What Was Done

The notebook checked whether the active-contract table is structurally safe.

### What The Results Mean

The important integrity checks passed:

```text
one active contract per product/date
no spreads selected
GC and MGC not mixed
selected volume exists
```

The 71 low-share days matter. They are mostly around unstable roll/transition conditions and are flagged rather than deleted.

### How We Use This To Our Advantage

This gives us a clean daily map from product/date to active symbol. It also gives future research a way to avoid weak liquidity days without permanently removing data.

---

## 6. Daily Winner vs Rolling Winner Match

### Output

Product-level match diagnostics.

Results:

```text
GC daily-winner match rate: 97.17%
MGC daily-winner match rate: 96.85%

GC rolling-winner match rate: 99.61%
MGC rolling-winner match rate: 99.74%
```

### What The Results Mean

The final selected active contract almost always matches the rolling-volume winner. It also usually matches the same-day volume winner.

The final rule differs from the daily winner more often than from the rolling winner. That is expected because the final rule intentionally smooths one-day noise.

### How We Use This To Our Advantage

The final rule is stable without being detached from actual same-day liquidity. This is the right balance for intraday research.

---

## 7. Rollover Summary

### Output

```text
rollover_summary
rollover_validation
```

Results:

```text
Total rollover rows: 50
GC rolls: 25
MGC rolls: 25
Missing old contract volume: 0
Missing new contract volume: 0
Unusual rolls: 0
Daily roll flags: 50
```

### What Was Done

The notebook detected every date where the selected active contract changed and created a rollover table with:

```text
product
roll_date
old_active_contract
new_active_contract
old_contract_volume
new_contract_volume
old_contract_volume_share
new_contract_volume_share
roll_transition_label
roll_notes
```

### What The Results Mean

The roll count is balanced and sensible:

```text
25 GC rolls
25 MGC rolls
```

No roll was flagged as unusual under the current thresholds. That means the selected new contract had sufficient liquidity and the old contract had already lost enough dominance on roll days.

### How We Use This To Our Advantage

`rollover_summary` gives us a clean audit trail. Later, if a strategy has suspicious trades near rolls, we can trace the exact old/new contract transition and inspect the liquidity state.

---

## 8. Continuous Active-Front Series

### Output

```text
active_front_final
active_front_validation
```

Results:

```text
Rows: 3,487,656
Products: 2
GC rows: 1,759,671
MGC rows: 1,727,985
Start: 2021-05-24 00:00:00 UTC
End: 2026-05-22 20:59:00 UTC
Duplicate timestamp-product rows: 0
Missing OHLCV values: 0
Spread symbols: 0
Symbol-active symbol mismatches: 0
```

### What Was Done

The daily active-contract table was joined back to minute-level GC/MGC outright bars. Only the selected active symbol for each product/date was kept.

### What The Results Mean

We now have a clean active-front dataset:

```text
one selected active contract per product/date
one row per timestamp/product where a bar exists
no spread instruments
no duplicate timestamp-product rows
no missing OHLCV
```

This is a raw stitched series, not a back-adjusted continuous futures series.

### How We Use This To Our Advantage

This is the first dataset in the project that behaves like a realistic tradable active contract stream. It allows later signal research to avoid multi-contract distortion.

---

## 9. Selected Contract Volume Share Distribution

### Output

Distribution of `selected_contract_volume_share`.

Results:

```text
GC mean: 93.01%
GC median: 97.01%
GC 5th percentile: 75.00%
GC 1st percentile: 5.93%

MGC mean: 92.93%
MGC median: 97.48%
MGC 5th percentile: 82.32%
MGC 1st percentile: 2.45%
```

### What The Results Mean

The selected active contract carries dominant liquidity most of the time.

The sharp drop in the 1st percentile confirms that a small number of days have poor selected-contract volume share. These are the days where rollover or unusual session effects matter.

### How We Use This To Our Advantage

This gives us a liquidity quality feature. Future strategies can:

```text
require selected_contract_volume_share above a threshold
exclude low-liquidity days
compare performance on high-liquidity vs low-liquidity days
apply stricter slippage assumptions when liquidity quality is weak
```

---

## 10. Rollover Warning and Exclusion Flags

### Output

```text
daily_roll_window_summary
minute_roll_window_summary
```

Daily results:

```text
GC product-days: 1,555
GC roll days: 25
GC pre-roll days: 25
GC post-roll days: 25
GC roll-window days: 75
GC low-liquidity days: 28
GC tradable research days: 1,472

MGC product-days: 1,555
MGC roll days: 25
MGC pre-roll days: 25
MGC post-roll days: 25
MGC roll-window days: 75
MGC low-liquidity days: 43
MGC tradable research days: 1,462
```

Minute-level results:

```text
GC rows: 1,759,671
GC roll-window rows: 77,651
GC roll-window row pct: 4.41%
GC tradable research rows: 1,676,591

MGC rows: 1,727,985
MGC roll-window rows: 68,741
MGC roll-window row pct: 3.98%
MGC tradable research rows: 1,649,276
```

### What Was Done

The notebook flagged:

```text
is_roll_day
is_pre_roll_day
is_post_roll_day
roll_window_flag
roll_window_type
days_since_roll
days_to_next_roll
low_liquidity_active_day_flag
tradable_research_flag
```

### What The Results Mean

Only about 4% of active-front rows fall inside the conservative roll window. That is small enough to exclude in initial research without losing too much data.

### How We Use This To Our Advantage

This gives future strategy research two clean modes:

```text
Conservative mode:
Use tradable_research_flag == True.

Diagnostic mode:
Include roll-window rows and compare performance degradation.
```

This is especially important for backtesting, where rollover-related price behavior can create false signals.

---

## 11. Session-Aware Calendar Outputs

### Output

```text
session_diagnostics
```

Results by volume share:

```text
GC:
NY Morning: 27.86% of volume, avg range 6.41 bps
London: 24.15% of volume, avg range 3.24 bps
NY RTH: 21.21% of volume, avg range 4.12 bps
Overnight/Asia: 17.05% of volume, avg range 2.60 bps
Late Session: 9.73% of volume, avg range 2.51 bps

MGC:
Overnight/Asia: 26.13% of volume, avg range 2.55 bps
NY Morning: 22.80% of volume, avg range 6.31 bps
London: 22.60% of volume, avg range 3.08 bps
NY RTH: 18.58% of volume, avg range 4.02 bps
Late Session: 9.88% of volume, avg range 2.46 bps
```

### What Was Done

The notebook preserved UTC timestamps and added New York time/session fields:

```text
ts_event_utc
ts_event_ny
trade_date_utc
trade_date_ny
day_of_week
hour_ny
minute_ny
session_label
```

### What The Results Mean

NY Morning is the most volatile and most concentrated session for GC. MGC has more volume share in Overnight/Asia than GC, but NY Morning still has the largest average bar range.

### How We Use This To Our Advantage

Session labels should become standard filters in Section 5+.

Early implication:

```text
NY Morning likely deserves separate strategy tests.
London is important but lower-volatility than NY Morning.
Overnight/Asia may behave differently, especially for MGC.
Late Session is lower activity and may need stricter filters.
```

---

## 12. Feature-Rich Research Table

### Output

```text
research_bars
feature_validation
forward_return_coverage
```

Final validation:

```text
Rows: 3,487,656
Products: 2
Duplicate timestamp-product rows: 0
Missing OHLCV values: 0
Non-null 1-minute returns: 3,454,742
Regular 1-minute bar rate: 99.06%
Roll-window rows: 146,392
Tradable research rows: 3,325,867
```

Forward return coverage:

```text
GC rows: 1,759,671
GC non-null 1m returns: 1,751,038
GC non-null 5m forward returns: 1,728,062
GC non-null 15m forward returns: 1,689,298
GC non-null 30m forward returns: 1,647,283
GC non-null 60m forward returns: 1,584,716

MGC rows: 1,727,985
MGC non-null 1m returns: 1,703,704
MGC non-null 5m forward returns: 1,639,637
MGC non-null 15m forward returns: 1,540,741
MGC non-null 30m forward returns: 1,448,381
MGC non-null 60m forward returns: 1,330,915
```

Median bar/range stats:

```text
GC median bar range: 2.45 bps
GC median true range: 2.55 bps
GC median relative volume 60m: 0.75

MGC median bar range: 2.32 bps
MGC median true range: 2.52 bps
MGC median relative volume 60m: 0.75
```

### What Was Done

The notebook rebuilt core research features:

```text
close-to-close return
1-minute return
log return
5/15/30/60-minute forward returns
bar range
candle body
upper/lower wick
true range
rolling ATR-style range
rolling realized volatility
rolling high-low range
rolling volume
relative volume
volume z-score
selected contract volume share
rollover flags
session labels
tradable research flag
```

Returns and rolling features are computed separately by product. Forward returns are set to missing when they would cross contract changes or major timestamp gaps.

### What The Results Mean

The final table is clean and usable. The high regular 1-minute bar rate means most rows can support intraday feature calculations.

The missing forward returns are expected. They occur near:

```text
contract changes
market closures
session gaps
dataset end
```

This is a feature, not a bug. It prevents contaminated forward-return labels.

### How We Use This To Our Advantage

`research_bars` is ready for:

```text
POI feature engineering
event studies
session-filtered signal tests
volatility-regime filters
rollover-safe backtests
volume/range context filters
forward-return diagnostics
```

---

## 13. Saved Research Tables

### Output

The notebook saved:

```text
data/processed/active_contract_daily.parquet
data/processed/rollover_summary.parquet
data/processed/active_front_final.parquet
data/processed/research_bars_gc_mgc_1m.parquet
```

Saved table sizes:

```text
active_contract_daily: 3,110 rows
rollover_summary: 50 rows
active_front_final: 3,487,656 rows
research_bars_gc_mgc_1m: 3,487,656 rows
```

### Why It Matters

The Section 4 outputs are now reusable. Future notebooks do not need to rebuild the full active-contract pipeline every time.

### How We Use This To Our Advantage

From Section 5 onward, we can load:

```python
research_bars = pd.read_parquet("data/processed/research_bars_gc_mgc_1m.parquet")
```

That gives us a consistent, validated research base.

---

## 14. Final Validation

### Output

```text
section4_final_validation
```

All checks passed:

```text
expected_products_only: True
no_spread_symbols: True
no_missing_ohlcv: True
no_duplicate_timestamp_product_rows: True
sorted_by_product_timestamp: True
active_contract_daily_columns_present: True
rollover_summary_columns_present: True
active_front_columns_present: True
research_bars_columns_present: True
rollover_flags_present: True
session_labels_present: True
```

### What The Results Mean

The Section 4 dataset passed the core integrity checks required before moving into strategy formalization.

### How We Use This To Our Advantage

We can proceed to Section 5 with confidence that the dataset is not mixing contracts, products, spread instruments, or contaminated rollover returns.

---

## Final Interpretation

Section 4 successfully transformed the raw multi-contract GC/MGC dataset into a professional research dataset.

The most important conclusions are:

```text
1. Liquidity-based active contract selection is reliable for GC and MGC.
2. The selected active contract carries around 97% median daily volume share.
3. Rollover events are cleanly detected and auditable.
4. Roll-window rows are flagged, not deleted.
5. The active-front series is raw stitched, not back-adjusted.
6. Session-aware fields confirm NY Morning is especially important.
7. research_bars is clean, validated, saved, and ready for Section 5.
```

The correct next step is:

```text
5.0 Discretionary Strategy Formalization
```

Section 5 should translate the discretionary POI-based trading logic into measurable rules using `research_bars` as the foundation.
