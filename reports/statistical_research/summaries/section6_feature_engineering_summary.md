# Section 6 Feature Engineering Summary

**Status:** READY

- Feature matrix shape: **586,530 × 97**
- Metadata columns: **12**
- Registered predictors: **85** (**79 core**, **6 experimental**)
- Numeric predictors: **80**
- Critical validation checks passed: **37/37**
- Automated feature tests: **14/14 passed**
- Estimated unoptimized matrix memory: **414.21 MiB**
- Optimized matrix memory: **213.40 MiB**
- Estimated peak working memory: **634.05 MiB**

## Features by family

- candle_geometry: 10
- experimental_hypothesis: 6
- price_return_momentum: 16
- session_clock_calendar: 9
- trend_persistence: 11
- volatility_range_state: 13
- volume_activity: 10
- vwap_session_state: 10

## Missingness

Development missingness ranges from 0.000000% to 1.150116%. Nulls are retained for warm-up, reset boundaries, session openings, zero denominators, and insufficient fitted-reference support.

Highest Development missingness:

- `two_bar_directional_balance`: 1.150116%
- `distance_from_session_open_atr`: 0.453581%
- `session_range_over_atr`: 0.453581%
- `distance_from_execution_session_vwap_atr`: 0.453581%
- `session_range_position`: 0.416680%
- `liquidity_vacuum_score_exp`: 0.200503%
- `signed_volume_proxy`: 0.200503%
- `range_relative_to_previous_bar`: 0.170134%

## Experimental hypotheses (*)

- *** Directional Energy Balance — 15m** (`directional_energy_balance_15_exp`): Squared returns emphasize the bars contributing most to realized movement and test whether that energy was directionally coherent.
- *** Wick Pressure Balance — 10m** (`wick_pressure_balance_10_exp`): Repeated tail rejection may contain information beyond one-bar wick geometry; this is not measured order flow or liquidity.
- *** Compression Age** (`compression_age_exp`): Duration distinguishes fresh compression from a prolonged stagnant regime without selecting thresholds from outcomes.
- *** VWAP Elasticity — 30m** (`vwap_elasticity_30_exp`): The local slope estimates whether completed price responses have recently reverted toward or continued away from research-day VWAP.
- *** Liquidity Vacuum Proxy** (`liquidity_vacuum_score_exp`): A large edge-closing bar on relatively light activity may proxy for low resistance; it is not measured depth or actual liquidity.
- *** Pullback Tension Across Time Scales — 5/30m** (`pullback_tension_5_30_exp`): The feature isolates a short counter-move inside a broader completed directional displacement without exhaustive interactions.

## Important implementation decisions

- Features were constructed on the full trusted GC bar sequence and mapped to eligible completed decision bars only after calculation.
- Continuity resets on non-one-minute timestamps, product/contract/instrument/segment changes, and tradability/roll/liquidity boundaries.
- Research-day state resets at 01:00 New York; London execution state resets at 03:00; New York execution state resets at 07:00.
- The first eligible entry at each execution-session open honestly has null execution-session state because decision bar t precedes the opening bar.
- Time-of-day log-volume references use Development only. Development rows exclude their own New York trading date; Validation and Final test use frozen full-Development parameters.
- Partial 2021 and partial 2026 contribute only their observed dates; no annual reweighting or full-sample clock normalization is used.
- Final-test inspection is limited to schema, row counts, null/finite rates, and transformation integrity. No feature-outcome relationship is examined.
- Float64 is used for sensitive calculations; ordinary saved continuous features are float32, bounded counters are nullable small integers, and repeated strings are categorical/dictionary encoded.

## Saved outputs

- feature_matrix: `data\processed\statistical_research\feature_matrix_gc.parquet`
- feature_registry: `data\processed\statistical_research\feature_registry_gc.parquet`
- feature_validation: `data\processed\statistical_research\feature_validation_gc.parquet`
- feature_diagnostics: `data\processed\statistical_research\feature_diagnostics_gc.parquet`
- feature_reference_parameters: `data\processed\statistical_research\feature_reference_parameters_gc.parquet`
- registry_csv: `reports\statistical_research\tables\section6\feature_registry_gc.csv`
- diagnostics_csv: `reports\statistical_research\tables\section6\feature_diagnostics_gc.csv`
- manual_audit_csv: `reports\statistical_research\tables\section6\section6_manual_feature_audit.csv`
- summary_markdown: `reports\statistical_research\summaries\section6_feature_engineering_summary.md`

## Known limitations

- Signed-volume, wick-pressure, and liquidity-vacuum measures are OHLCV proxies, not order flow, queue state, market depth, or measured liquidity.
- Feature definitions are frozen for Section 7 evaluation, but no redundancy selection or predictive interpretation has been performed.
- MGC is intentionally excluded until the GC shortlist is frozen for later transfer validation.

**No feature has yet been shown to possess predictive value.** Section 6 establishes only a valid candidate feature matrix.

**Exact next section:** Section 7.0 — Univariate Feature Evaluation

*Follow-up (2026-07-20): executed as declared - see `section7_univariate_evaluation_summary.md`.*

SECTION 6 STATUS: READY
