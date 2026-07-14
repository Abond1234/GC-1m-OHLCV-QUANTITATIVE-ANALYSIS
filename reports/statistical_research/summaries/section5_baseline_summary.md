# Section 5 Baseline Summary and Benchmark Definition

## Scope and statistical interpretation

This report describes overlapping GC observation outcomes at 5, 15, 30, 60, 120, and 180 minutes. These are event-level labels, not independent trials or trades. MFE is not realized profit, MAE is not realized loss, and future range is not a tradable return. Raw calculations are never clipped; figure clipping, where used for display, does not alter the tables. Dependence-aware intervals resample New York trading-date aggregates with seed 20260714 and 2,000 replicates.

## Unconditional benchmark

- Available counts by horizon: 5m=586,468, 15m=586,296, 30m=585,958, 60m=585,033, 120m=582,471, 180m=579,166.
- Mean signed returns in ticks (5m through 180m): 0.014, -0.036, -0.144, -0.536, -0.871, -0.802.
- Median signed returns in ticks: 0.000, 0.000, 1.000, 1.000, 2.000, 2.000.
- Positive-return rates: 0.4857, 0.4968, 0.5007, 0.5065, 0.5087, 0.5107.
- Mean absolute returns in ticks: 14.192, 24.454, 34.492, 48.874, 68.019, 82.643.
- The p05/p95 tick pairs are -31.0/31.0, -55.0/54.0, -78.0/77.0, -111.0/108.0, -156.0/150.0, -190.0/183.0; tails and robust centers must accompany means in future comparisons.
- Mean future ranges in ticks are 28.710, 50.712, 72.373, 103.223, 144.445, 175.074; the fitted log-log horizon exponent is 0.505 (a transparent growth diagnostic, not a holding-period choice).
- Mean future realized volatility in basis points is 6.370, 11.590, 16.698, 23.959, 33.767, 41.112; its fitted log-log horizon exponent is 0.520. Movement growth is therefore measured directly rather than assumed proportional to clock time.

## Sessions, time of day, and weekdays

- London mean returns in ticks: 0.074, 0.113, 0.164, -0.025, 0.509, 0.863; New York: -0.022, -0.125, -0.329, -0.842, -1.696, -1.800.
- London positive rates: 0.4798, 0.4915, 0.4963, 0.5071, 0.5163, 0.5179; New York: 0.4892, 0.5000, 0.5033, 0.5061, 0.5042, 0.5064.
- London mean future ranges in ticks: 21.520, 37.743, 53.258, 74.683, 103.777, 133.565; New York: 33.023, 58.487, 83.824, 120.287, 168.744, 199.960.
- Availability, 15-minute entry bins, and minutes-since-session-open bins are saved explicitly. Long-horizon near-cutoff subsets must not be compared with early-session subsets without matching their availability and date coverage.
- Weekday estimates are descriptive and are saved both overall and by year with per-session date coverage (overall cells span 238 to 250 dates). No weekday is declared an effect from a single year or a small subset.

## Directional and tail symmetry

- Short signed returns are exactly the algebraic negative of long returns and are not independent samples.
- Full-population p95/|p05| tail ratios by horizon: 1.000, 0.982, 0.987, 0.973, 0.962, 0.963.
- Full-population p99/|p01| tail ratios by horizon: 0.959, 0.909, 0.873, 0.863, 0.859, 0.843.
- Long/short excursion identities reconcile exactly. Any modest unconditional tail imbalance remains descriptive and does not establish a directional edge.

## Year and research-partition stability

- 2021: 2021-05-24 to 2021-12-31, 152 dates (partial from dataset start).
- 2022: 2022-01-03 to 2022-12-30, 244 dates (complete observed year).
- 2023: 2023-01-03 to 2023-12-29, 242 dates (complete observed year).
- 2024: 2024-01-02 to 2024-12-31, 246 dates (complete observed year).
- 2025: 2025-01-02 to 2025-12-31, 245 dates (complete observed year).
- 2026: 2026-01-02 to 2026-05-22, 93 dates (partial through 2026-05-22).

Development and Validation are the primary interpretation samples. The baseline code path and metric contract were frozen before the one-time descriptive Final-test exposure. Section 5 exposes Final-test means, medians, positive rates, magnitude, excursion, range, volatility, tails, availability, session, time, weekday, symmetry, and fixed hurdle results. Future work must not tune definitions, horizons, cost hurdles, or features to those exposed values.

## Movement hurdles, not PnL

- At a fixed five-tick hurdle, combined absolute exit-to-exit exceedance rates (5m through 180m) are 0.6349, 0.7743, 0.8375, 0.8862, 0.9186, 0.9334.
- Five-tick long directional exceedance rates are 0.3211, 0.3941, 0.4270, 0.4545, 0.4719, 0.4806; short directional exceedance rates are 0.3138, 0.3802, 0.4105, 0.4317, 0.4466, 0.4529.
- Exit-to-exit return, MFE opportunity, MAE risk, and total high-low range hurdles are saved separately. Frequent intrahorizon exceedance is not evidence of realizable profit.

## Benchmark contract for future features

A future feature must be compared with the matching baseline using the same horizon, session, direction, research partition, availability rules, outcome definition, cost hurdle, and trading-date coverage. A positive bucket mean is insufficient. Evidence must include adequate observations and dates, economically meaningful effect size, Development-to-Validation consistency, reasonable year stability, session stability or a justified session thesis, friction-aware magnitude, tail robustness, date-clustered uncertainty, interpretability, and no leakage or Final-test tuning. No arbitrary pass/fail thresholds are introduced here.

Critical validation blockers: None.

SECTION 5 STATUS: READY
