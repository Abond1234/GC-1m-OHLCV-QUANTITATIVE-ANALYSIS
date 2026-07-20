# Section 8 Redundancy and Incremental Information Summary

**Status:** READY

## Scope and governance

- Input population: the Section 7 evaluation frame — **Development + Validation only** (419,393 observations; Final-test rows excluded and rejected with an error if present).
- Candidate features: the **55** Section 7 `ADVANCE_EXPANSION` verdicts. The directional candidate set is **empty by evidence** (0 Section 7 directional advancers), so the frozen directional set is explicitly empty rather than silently absent.
- All rules were frozen in `Section8Config` before computation (seed 20260721): Development-only Spearman correlations; average-linkage clustering on `1 − |ρ|` cut at |ρ| ≥ 0.7; mechanical representative selection (best passing-cell Validation |IC| at the 60-minute primary horizon, ties → core over experimental, then name); incremental gates of Development |partial IC| ≥ 0.05 with Validation sign agreement, ≥ 25% retention, and 400/150 trading-date floors at 60- and 180-minute horizons.

## Results

| Stage | Count |
|---|---:|
| Expansion advancers in | 55 |
| Development-fitted clusters | 30 |
| Representatives (one per cluster) | 30 |
| Anchor | `atr_20` |
| Non-anchor representatives with confirmed incremental information | 14 |
| **Frozen expansion feature set** | **15** |
| Frozen directional feature set | 0 |

- The largest cluster (7 features) is the ATR/realized-volatility ladder, represented by `atr_20` — which is also the global anchor (best Validation |IC| ≈ 0.55 at 60 m).
- The frozen set spans distinct information families: volatility state (`atr_20`, `atr_ratio_20_60`, `atr_ratio_5_20`, `current_range_over_atr`), session clock (`minute_from_execution_window_open`, `minutes_to_1530_forced_exit`), trend efficiency/structure (`efficiency_ratio_15/30/60`, `ols_r_squared_30`, `choppiness_14`, `return_sign_change_rate_30`), activity (`relative_volume_20`, `volume_acceleration_5_20`), and one surviving experimental hypothesis (`vwap_elasticity_30_exp`).
- **Redundancy did real work:** 15 of 30 representatives were rejected for no confirmed incremental information beyond the anchor — including `log_volume`, whose raw Validation |IC| ≈ 0.29 is impressive univariately but adds nothing once volatility state is known. Two of the three surviving experimental representatives also died here.

## Verification

- **8/8 structural checks passed** (partition containment; correlation matrix square on advancers; every advancer in exactly one cluster; one representative per cluster; anchor in the frozen set; frozen set ⊆ representatives; directional set empty; incremental rows cover both partitions).
- **10/10 new synthetic tests passed** (duplicate features cluster together; representative and anchor mechanics; a planted independent driver confirms incremental value while a pure-noise representative is rejected with its reason recorded; Final-test rows raise; determinism; clustering provably fitted on Development only — a duplicate decorrelated in Validation still clusters with its Development twin). Full project suite: **122/122**.
- Runtime ≈ 10.5 s; peak working memory ≈ 0.6 GB.

## Saved outputs (excluded from Git)

- `data/processed/statistical_research/feature_correlation_gc.parquet`
- `data/processed/statistical_research/feature_clusters_gc.parquet`
- `data/processed/statistical_research/feature_incremental_information_gc.parquet`
- `data/processed/statistical_research/frozen_expansion_feature_set_gc.parquet`
- `reports/statistical_research/tables/section8/*.csv`

This summary is the tracked record of the milestone.

## Exact next step

**Section 9.0 — Multivariate Research**, beginning with simple linear benchmarks on the 15-feature frozen set under chronological validation, against the anchor-only baseline. The Final test remains locked.

SECTION 8 STATUS: READY
