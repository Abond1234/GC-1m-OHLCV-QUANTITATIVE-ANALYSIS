# Strategy catalog (strategy-lab branch)

A pre-registered catalog of ~200 rule-based trading strategies, each backtested through the branch's independently verified one-position engine and scored with a full anti-overfitting battery. This document is generated from the code and the run (`scripts/build_strategy_catalog_doc.py`) so the descriptions, parameters, and results below cannot drift from what was actually simulated.

**Peers: this is the menu.** Every row is a rule you can rerun, tweak, or use as a template for your own idea. Add a strategy by dropping a `StrategySpec` into `src/statistical_research/strategy_catalog.py` (or `strategy_lab.py`) and rerunning `scripts/run_strategy_catalog.py`; the evaluation deflates automatically for the new trial count.

## Headline result

- **196 strategies tested; 0 advanced.** Selection on Development, confirmation on Validation, Final-test partition untouched.
- **The gross (zero-cost) per-trade edge is ~0 for essentially every rule** - 49/196 have a positive frictionless edge, and the whole range is [-0.061, +0.027] R. This is a signal wall, not a cost wall.
- **The best deflated Sharpe across all 196 trials is 0.000** (1.0 would mean a real edge after correcting for the number tried). Benjamini-Hochberg q-values and Validation bootstrap intervals agree.
- **Probability of backtest overfitting (CSCV): 0.00**, with 0.00 probability the in-sample-best strategy is profitable out of sample - the ranking is stable but uniformly unprofitable.
- Widening the search from a dozen rules to ~200 did not surface an edge; it raised the deflation bar, exactly as intended. A broad honest search that finds nothing is the result.

## How each strategy is built and judged

- **Inputs are point-in-time.** Every rule reads only causal registered features (each available after the completed decision bar); fills are at the next bar's open; no forward label is ever read.
- **Thresholds are pre-registered, not fitted.** They are Development-only marginal quantiles of the feature (p20/p80, or p10/p90 for extremes) or fixed structural levels (0 for a slope, 1 for a variance ratio, 0.5 for a bounded oscillator) - never chosen from the relationship between the feature and the outcome.
- **One shared exit contract.** Every strategy inherits the frozen Section 10 stop and target (1.5x ATR stop, 2R target, 120-minute cap), so the search is purely over entries and every row is directly comparable. Varying the exit is a separate future axis.
- **Costs:** base scenario is 2.6 ticks round trip; the frictionless column isolates the raw signal from the cost drag.
- **Anti-overfitting:** date-block bootstrap CIs, Benjamini-Hochberg q-values across the family, the deflated Sharpe ratio (deflated by the full trial count), and the probability of backtest overfitting via CSCV. Numbers reproduce from `scripts/run_strategy_catalog.py`.
- **Near-dead rules dropped:** 1 pre-registered rule(s) whose condition almost never occurs on this data were dropped as structurally inapplicable (three_bar_drive_x).

Columns: **Looks for** = the entry trigger; **Params** = the frozen constants; **Trades/Win** are base-cost Development; **Gross** is the frictionless Development per-trade edge (signal only); **Dev R / Val R** are base-cost per-trade net R; **DSR** is the deflated Sharpe; **Verdict** is the advancement gate.

## Results by family

| Family | Strategies | Median Dev R (base) | Best gross edge | Best DSR | Advanced |
|---|---|---|---|---|---|
| Trend / momentum | 45 | -0.228 | +0.010 | 0.000 | 0 |
| Mean-reversion | 20 | -0.206 | +0.014 | 0.000 | 0 |
| Breakout / range | 10 | -0.235 | -0.008 | 0.000 | 0 |
| Volatility-regime | 2 | -0.215 | +0.001 | 0.000 | 0 |
| Volume / flow | 10 | -0.236 | +0.022 | 0.000 | 0 |
| Candle / price-action | 6 | -0.215 | +0.014 | 0.000 | 0 |
| VWAP-relative | 7 | -0.222 | +0.003 | 0.000 | 0 |
| Session / time-of-day | 7 | -0.264 | +0.015 | 0.000 | 0 |
| Regime-gated | 89 | -0.226 | +0.027 | 0.000 | 0 |
| Null benchmarks | 2 | -0.217 | - | nan | 0 |

### Trend / momentum (45)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `open_drive_follow_x` | Long when distance_from_session_open_atr > 8.94; short when < -8.74 (extreme tails). | feature=distance_from_session_open_atr; long_above=8.9412; short_below=-8.7407; thresholds=Dev p10 / p90 | 5,433 | 33.3% | -0.005 | -0.215 | -0.164 | 0.000 | REJECTED_NO_EDGE |
| `persistence_trend_x` | Long when directional_persistence_15 > 0.4; short when < 0 (extreme tails). | feature=directional_persistence_15; long_above=0.4; short_below=0; thresholds=Dev p10 / p90 | 5,936 | 33.4% | -0.002 | -0.214 | -0.195 | 0.000 | REJECTED_NO_EDGE |
| `slope_trend_60_x` | Long when normalized_ols_slope_60 > 0.118; short when < -0.113 (extreme tails). | feature=normalized_ols_slope_60; long_above=0.1175; short_below=-0.1126; thresholds=Dev p10 / p90 | 6,519 | 32.9% | -0.016 | -0.228 | -0.158 | 0.000 | REJECTED_NO_EDGE |
| `vwap_slope_trend_x` | Long when vwap_slope_15_atr > 0.506; short when < -0.473 (extreme tails). | feature=vwap_slope_15_atr; long_above=0.5057; short_below=-0.4727; thresholds=Dev p10 / p90 | 7,079 | 32.8% | -0.020 | -0.221 | -0.158 | 0.000 | REJECTED_NO_EDGE |
| `tsmom_60m_x` | Long when return_60m_atr > 6.51; short when < -6.3 (extreme tails). | feature=return_60m_atr; long_above=6.506; short_below=-6.2963; thresholds=Dev p10 / p90 | 8,119 | 32.8% | -0.020 | -0.234 | -0.166 | 0.000 | REJECTED_NO_EDGE |
| `accel_15_30_x` | Long when momentum_acceleration_15_30 > 0.612; short when < -0.609 (extreme tails). | feature=momentum_acceleration_15_30; long_above=0.6116; short_below=-0.6089; thresholds=Dev p10 / p90 | 11,501 | 33.2% | -0.007 | -0.197 | -0.181 | 0.000 | REJECTED_NO_EDGE |
| `vwap_slope_fast_x` | Long when vwap_slope_5_atr > 0.162; short when < -0.152 (extreme tails). | feature=vwap_slope_5_atr; long_above=0.1621; short_below=-0.1516; thresholds=Dev p10 / p90 | 9,129 | 33.3% | -0.004 | -0.210 | -0.175 | 0.000 | REJECTED_NO_EDGE |
| `slope_trend_30_x` | Long when normalized_ols_slope_30 > 0.165; short when < -0.159 (extreme tails). | feature=normalized_ols_slope_30; long_above=0.1651; short_below=-0.1593; thresholds=Dev p10 / p90 | 8,448 | 32.7% | -0.025 | -0.237 | -0.202 | 0.000 | REJECTED_NO_EDGE |
| `autocorr_trend_x` | Long when return_autocorrelation_15 > 0.241; short when < -0.411 (extreme tails). | feature=return_autocorrelation_15; long_above=0.2414; short_below=-0.4107; thresholds=Dev p10 / p90 | 12,106 | 33.2% | -0.005 | -0.221 | -0.183 | 0.000 | REJECTED_NO_EDGE |
| `open_drive_follow` | Long when distance_from_session_open_atr > 5.25; short when < -5.28. | feature=distance_from_session_open_atr; long_above=5.2525; short_below=-5.2843; thresholds=Dev p20 / p80 | 10,600 | 32.7% | -0.021 | -0.231 | -0.180 | 0.000 | REJECTED_NO_EDGE |
| `slope_trend_60` | Long when normalized_ols_slope_60 > 0.078; short when < -0.0758. | feature=normalized_ols_slope_60; long_above=0.078; short_below=-0.0758; thresholds=Dev p20 / p80 | 11,864 | 33.1% | -0.010 | -0.222 | -0.165 | 0.000 | REJECTED_NO_EDGE |
| `tsmom_30m_x` | Long when return_30m_atr > 4.57; short when < -4.42 (extreme tails). | feature=return_30m_atr; long_above=4.5669; short_below=-4.4172; thresholds=Dev p10 / p90 | 10,217 | 32.6% | -0.025 | -0.239 | -0.199 | 0.000 | REJECTED_NO_EDGE |
| `slope_trend_15_x` | Long when normalized_ols_slope_15 > 0.234; short when < -0.233 (extreme tails). | feature=normalized_ols_slope_15; long_above=0.2342; short_below=-0.2327; thresholds=Dev p10 / p90 | 11,786 | 32.7% | -0.022 | -0.237 | -0.182 | 0.000 | REJECTED_NO_EDGE |
| `vwap_slope_trend` | Long when vwap_slope_15_atr > 0.289; short when < -0.272. | feature=vwap_slope_15_atr; long_above=0.289; short_below=-0.2719; thresholds=Dev p20 / p80 | 12,216 | 33.1% | -0.011 | -0.218 | -0.175 | 0.000 | REJECTED_NO_EDGE |
| `accel_5_15_x` | Long when momentum_acceleration_5_15 > 1.22; short when < -1.22 (extreme tails). | feature=momentum_acceleration_5_15; long_above=1.2226; short_below=-1.2203; thresholds=Dev p10 / p90 | 14,937 | 32.7% | -0.021 | -0.219 | -0.180 | 0.000 | REJECTED_NO_EDGE |
| `persistence_trend` | Long when directional_persistence_15 > 0.333; short when < 0.0667. | feature=directional_persistence_15; long_above=0.3333; short_below=0.0667; thresholds=Dev p20 / p80 | 15,057 | 33.5% | +0.002 | -0.216 | -0.211 | 0.000 | REJECTED_NO_EDGE |
| `tsmom_15m_x` | Long when return_15m_atr > 3.22; short when < -3.19 (extreme tails). | feature=return_15m_atr; long_above=3.2161; short_below=-3.1944; thresholds=Dev p10 / p90 | 12,794 | 32.5% | -0.028 | -0.243 | -0.202 | 0.000 | REJECTED_NO_EDGE |
| `slope_trend_30` | Long when normalized_ols_slope_30 > 0.11; short when < -0.107. | feature=normalized_ols_slope_30; long_above=0.1101; short_below=-0.1072; thresholds=Dev p20 / p80 | 14,707 | 33.0% | -0.014 | -0.227 | -0.186 | 0.000 | REJECTED_NO_EDGE |
| `tsmom_60m` | Long when return_60m_atr > 4.35; short when < -4.24. | feature=return_60m_atr; long_above=4.3523; short_below=-4.2353; thresholds=Dev p20 / p80 | 13,735 | 32.7% | -0.023 | -0.237 | -0.176 | 0.000 | REJECTED_NO_EDGE |
| `autocorr_trend` | Long when return_autocorrelation_15 > 0.128; short when < -0.307. | feature=return_autocorrelation_15; long_above=0.1278; short_below=-0.3066; thresholds=Dev p20 / p80 | 17,239 | 33.4% | +0.001 | -0.214 | -0.201 | 0.000 | REJECTED_NO_EDGE |
| `vwap_slope_fast` | Long when vwap_slope_5_atr > 0.0891; short when < -0.0839. | feature=vwap_slope_5_atr; long_above=0.0891; short_below=-0.0839; thresholds=Dev p20 / p80 | 14,245 | 33.3% | -0.004 | -0.214 | -0.175 | 0.000 | REJECTED_NO_EDGE |
| `tsmom_30m` | Long when return_30m_atr > 3.05; short when < -2.96. | feature=return_30m_atr; long_above=3.0476; short_below=-2.963; thresholds=Dev p20 / p80 | 15,783 | 33.1% | -0.010 | -0.225 | -0.190 | 0.000 | REJECTED_NO_EDGE |
| `accel_15_30` | Long when momentum_acceleration_15_30 > 0.361; short when < -0.364. | feature=momentum_acceleration_15_30; long_above=0.3612; short_below=-0.3637; thresholds=Dev p20 / p80 | 18,059 | 32.8% | -0.017 | -0.223 | -0.188 | 0.000 | REJECTED_NO_EDGE |
| `slope_trend_15` | Long when normalized_ols_slope_15 > 0.155; short when < -0.153. | feature=normalized_ols_slope_15; long_above=0.1548; short_below=-0.1534; thresholds=Dev p20 / p80 | 17,874 | 33.0% | -0.014 | -0.229 | -0.189 | 0.000 | REJECTED_NO_EDGE |
| `tsmom_15m` | Long when return_15m_atr > 2.14; short when < -2.11. | feature=return_15m_atr; long_above=2.1384; short_below=-2.1053; thresholds=Dev p20 / p80 | 18,231 | 33.0% | -0.014 | -0.228 | -0.198 | 0.000 | REJECTED_NO_EDGE |
| `tsmom_5m_x` | Long when return_5m_atr > 1.88; short when < -1.88 (extreme tails). | feature=return_5m_atr; long_above=1.875; short_below=-1.8792; thresholds=Dev p10 / p90 | 17,532 | 32.3% | -0.031 | -0.248 | -0.183 | 0.000 | REJECTED_NO_EDGE |
| `stoch_cross` | Long when rolling_range_position_15 > rolling_range_position_60; short when rolling_range_position_15 < rolling_range_position_60. | fast=rolling_range_position_15; slow=rolling_range_position_60 | 22,381 | 33.3% | -0.003 | -0.218 | -0.196 | 0.000 | REJECTED_NO_EDGE |
| `return_cross_15_60` | Long when return_15m_atr > return_60m_atr; short when return_15m_atr < return_60m_atr. | fast=return_15m_atr; slow=return_60m_atr | 23,132 | 33.7% | +0.010 | -0.204 | -0.198 | 0.000 | REJECTED_NO_EDGE |
| `slope_sign_30` | Long when normalized_ols_slope_30 > 0; short when < 0. | feature=normalized_ols_slope_30; level=0 | 23,596 | 33.3% | -0.004 | -0.218 | -0.191 | 0.000 | REJECTED_NO_EDGE |
| `accel_5_15` | Long when momentum_acceleration_5_15 > 0.72; short when < -0.728. | feature=momentum_acceleration_5_15; long_above=0.7201; short_below=-0.7281; thresholds=Dev p20 / p80 | 20,542 | 32.5% | -0.025 | -0.235 | -0.178 | 0.000 | REJECTED_NO_EDGE |
| `ma_cross_30_60` | Long when normalized_ols_slope_30 > normalized_ols_slope_60; short when normalized_ols_slope_30 < normalized_ols_slope_60. | fast=normalized_ols_slope_30; slow=normalized_ols_slope_60 | 23,458 | 33.2% | -0.007 | -0.221 | -0.193 | 0.000 | REJECTED_NO_EDGE |
| `return_cross_5_30` | Long when return_5m_atr > return_30m_atr; short when return_5m_atr < return_30m_atr. | fast=return_5m_atr; slow=return_30m_atr | 22,994 | 33.4% | +0.000 | -0.213 | -0.190 | 0.000 | REJECTED_NO_EDGE |
| `open_side` | Long when distance_from_session_open_atr > 0; short when < 0. | feature=distance_from_session_open_atr; level=0 | 23,070 | 33.1% | -0.009 | -0.222 | -0.189 | 0.000 | REJECTED_NO_EDGE |
| `efficiency_cross` | Long when efficiency_ratio_15 > efficiency_ratio_60; short when efficiency_ratio_15 < efficiency_ratio_60. | fast=efficiency_ratio_15; slow=efficiency_ratio_60 | 23,205 | 33.0% | -0.013 | -0.227 | -0.186 | 0.000 | REJECTED_NO_EDGE |
| `efficiency_cross_15_30` | Long when efficiency_ratio_15 > efficiency_ratio_30; short when efficiency_ratio_15 < efficiency_ratio_30. | fast=efficiency_ratio_15; slow=efficiency_ratio_30 | 23,195 | 32.9% | -0.016 | -0.229 | -0.173 | 0.000 | REJECTED_NO_EDGE |
| `body_momentum_x` | Long when signed_body_atr > 0.836; short when < -0.833 (extreme tails). | feature=signed_body_atr; long_above=0.8364; short_below=-0.8333; thresholds=Dev p10 / p90 | 19,610 | 32.4% | -0.028 | -0.244 | -0.193 | 0.000 | REJECTED_NO_EDGE |
| `tsmom_5m` | Long when return_5m_atr > 1.22; short when < -1.22. | feature=return_5m_atr; long_above=1.2162; short_below=-1.2183; thresholds=Dev p20 / p80 | 21,618 | 32.4% | -0.029 | -0.243 | -0.188 | 0.000 | REJECTED_NO_EDGE |
| `three_bar_sign` | Long when three_bar_directional_balance > 0; short when < 0. | feature=three_bar_directional_balance; level=0 | 23,556 | 32.7% | -0.019 | -0.233 | -0.194 | 0.000 | REJECTED_NO_EDGE |
| `body_momentum` | Long when signed_body_atr > 0.533; short when < -0.531. | feature=signed_body_atr; long_above=0.5333; short_below=-0.531; thresholds=Dev p20 / p80 | 22,080 | 32.6% | -0.023 | -0.237 | -0.197 | 0.000 | REJECTED_NO_EDGE |
| `ma_cross_15_60` | Long when normalized_ols_slope_15 > normalized_ols_slope_60; short when normalized_ols_slope_15 < normalized_ols_slope_60. | fast=normalized_ols_slope_15; slow=normalized_ols_slope_60 | 23,574 | 32.8% | -0.020 | -0.233 | -0.203 | 0.000 | REJECTED_NO_EDGE |
| `accel_sign` | Long when momentum_acceleration_5_15 > 0; short when < 0. | feature=momentum_acceleration_5_15; level=0 | 23,675 | 32.6% | -0.022 | -0.236 | -0.194 | 0.000 | REJECTED_NO_EDGE |
| `three_bar_drive` | Long when three_bar_directional_balance > 0.733; short when < -0.75. | feature=three_bar_directional_balance; long_above=0.7333; short_below=-0.75; thresholds=Dev p20 / p80 | 21,026 | 32.8% | -0.018 | -0.232 | -0.193 | 0.000 | REJECTED_NO_EDGE |
| `ma_cross_15_30` | Long when normalized_ols_slope_15 > normalized_ols_slope_30; short when normalized_ols_slope_15 < normalized_ols_slope_30. | fast=normalized_ols_slope_15; slow=normalized_ols_slope_30 | 23,584 | 32.7% | -0.022 | -0.236 | -0.203 | 0.000 | REJECTED_NO_EDGE |
| `streak_sign` | Long when directional_streak > 0; short when < 0. | feature=directional_streak; level=0 | 23,583 | 32.4% | -0.029 | -0.243 | -0.182 | 0.000 | REJECTED_NO_EDGE |
| `two_bar_sign` | Long when two_bar_directional_balance > 0; short when < 0. | feature=two_bar_directional_balance; level=0 | 23,572 | 32.8% | -0.016 | -0.230 | -0.200 | 0.000 | REJECTED_NO_EDGE |

### Mean-reversion (20)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `vwap_revert_day_x` | Long when distance_from_research_day_vwap_atr < -6.81; short when > 7.31 (extreme tails). | feature=distance_from_research_day_vwap_atr; long_below=-6.8089; short_above=7.3123; thresholds=Dev p10 / p90 | 6,800 | 33.3% | -0.002 | -0.217 | -0.196 | 0.000 | REJECTED_NO_EDGE |
| `open_fade_x` | Long when distance_from_session_open_atr < -8.74; short when > 8.94 (extreme tails). | feature=distance_from_session_open_atr; long_below=-8.7407; short_above=8.9412; thresholds=Dev p10 / p90 | 5,729 | 33.0% | -0.011 | -0.221 | -0.197 | 0.000 | REJECTED_NO_EDGE |
| `vwap_revert_session_x` | Long when distance_from_execution_session_vwap_atr < -4.56; short when > 4.75 (extreme tails). | feature=distance_from_execution_session_vwap_atr; long_below=-4.5594; short_above=4.7547; thresholds=Dev p10 / p90 | 7,338 | 32.9% | -0.016 | -0.223 | -0.177 | 0.000 | REJECTED_NO_EDGE |
| `open_fade` | Long when distance_from_session_open_atr < -5.28 (stretched down); short when > 5.25. | feature=distance_from_session_open_atr; long_below=-5.2843; short_above=5.2525; thresholds=Dev p20 / p80 | 11,188 | 33.7% | +0.008 | -0.202 | -0.186 | 0.000 | REJECTED_NO_EDGE |
| `vwap_revert_day` | Long when distance_from_research_day_vwap_atr < -4.53 (stretched down); short when > 4.72. | feature=distance_from_research_day_vwap_atr; long_below=-4.5276; short_above=4.7185; thresholds=Dev p20 / p80 | 12,262 | 33.7% | +0.009 | -0.206 | -0.202 | 0.000 | REJECTED_NO_EDGE |
| `vwap_revert_60_x` | Long when distance_from_rolling_vwap_60_atr < -3.23; short when > 3.4 (extreme tails). | feature=distance_from_rolling_vwap_60_atr; long_below=-3.2299; short_above=3.3967; thresholds=Dev p10 / p90 | 11,664 | 33.5% | +0.003 | -0.213 | -0.219 | 0.000 | REJECTED_NO_EDGE |
| `vwap_revert_session` | Long when distance_from_execution_session_vwap_atr < -2.78 (stretched down); short when > 2.86. | feature=distance_from_execution_session_vwap_atr; long_below=-2.783; short_above=2.8601; thresholds=Dev p20 / p80 | 13,925 | 33.4% | +0.000 | -0.211 | -0.200 | 0.000 | REJECTED_NO_EDGE |
| `reversal_15m_x` | Long when return_15m_atr < -3.19; short when > 3.22 (extreme tails). | feature=return_15m_atr; long_below=-3.1944; short_above=3.2161; thresholds=Dev p10 / p90 | 15,034 | 33.9% | +0.014 | -0.201 | -0.196 | 0.000 | REJECTED_NO_EDGE |
| `vwap_revert_20_x` | Long when distance_from_rolling_vwap_20_atr < -1.92; short when > 1.94 (extreme tails). | feature=distance_from_rolling_vwap_20_atr; long_below=-1.9229; short_above=1.9425; thresholds=Dev p10 / p90 | 16,946 | 33.9% | +0.014 | -0.202 | -0.202 | 0.000 | REJECTED_NO_EDGE |
| `clv_fade` | Long when close_location_value < -0.714 (oversold); short when > 0.714 (overbought). | feature=close_location_value; long_below=-0.7143; short_above=0.7143; thresholds=Dev p20 / p80 | 19,263 | 33.8% | +0.013 | -0.201 | -0.186 | 0.000 | REJECTED_NO_EDGE |
| `vwap_revert_60` | Long when distance_from_rolling_vwap_60_atr < -2.19 (stretched down); short when > 2.31. | feature=distance_from_rolling_vwap_60_atr; long_below=-2.188; short_above=2.3078; thresholds=Dev p20 / p80 | 17,101 | 33.5% | +0.003 | -0.213 | -0.207 | 0.000 | REJECTED_NO_EDGE |
| `session_pos_fade` | Long when session_range_position < 0.203 (oversold); short when > 0.811 (overbought). | feature=session_range_position; long_below=0.2025; short_above=0.8112; thresholds=Dev p20 / p80 | 15,641 | 33.2% | -0.006 | -0.221 | -0.215 | 0.000 | REJECTED_NO_EDGE |
| `rsi60_fade` | Long when rolling_range_position_60 < 0.205 (oversold); short when > 0.81 (overbought). | feature=rolling_range_position_60; long_below=0.2048; short_above=0.8103; thresholds=Dev p20 / p80 | 17,465 | 33.7% | +0.008 | -0.207 | -0.210 | 0.000 | REJECTED_NO_EDGE |
| `reversal_15m` | Long when return_15m_atr < -2.11 (stretched down); short when > 2.14. | feature=return_15m_atr; long_below=-2.1053; short_above=2.1384; thresholds=Dev p20 / p80 | 19,816 | 33.7% | +0.010 | -0.204 | -0.205 | 0.000 | REJECTED_NO_EDGE |
| `reversal_5m_x` | Long when return_5m_atr < -1.88; short when > 1.88 (extreme tails). | feature=return_5m_atr; long_below=-1.8792; short_above=1.875; thresholds=Dev p10 / p90 | 18,303 | 33.7% | +0.009 | -0.206 | -0.210 | 0.000 | REJECTED_NO_EDGE |
| `body_fade_x` | Long when signed_body_atr < -0.833; short when > 0.836 (extreme tails). | feature=signed_body_atr; long_below=-0.8333; short_above=0.8364; thresholds=Dev p10 / p90 | 19,193 | 33.4% | -0.001 | -0.216 | -0.186 | 0.000 | REJECTED_NO_EDGE |
| `vwap_revert_20` | Long when distance_from_rolling_vwap_20_atr < -1.28 (stretched down); short when > 1.31. | feature=distance_from_rolling_vwap_20_atr; long_below=-1.2819; short_above=1.3121; thresholds=Dev p20 / p80 | 21,037 | 33.7% | +0.008 | -0.206 | -0.196 | 0.000 | REJECTED_NO_EDGE |
| `reversal_5m` | Long when return_5m_atr < -1.22 (stretched down); short when > 1.22. | feature=return_5m_atr; long_below=-1.2183; short_above=1.2162; thresholds=Dev p20 / p80 | 21,606 | 33.6% | +0.007 | -0.206 | -0.188 | 0.000 | REJECTED_NO_EDGE |
| `body_fade` | Long when signed_body_atr < -0.531 (stretched down); short when > 0.533. | feature=signed_body_atr; long_below=-0.531; short_above=0.5333; thresholds=Dev p20 / p80 | 21,519 | 33.9% | +0.014 | -0.200 | -0.190 | 0.000 | REJECTED_NO_EDGE |
| `rsi15_fade` | Long when rolling_range_position_15 < 0.192 (oversold); short when > 0.815 (overbought). | feature=rolling_range_position_15; long_below=0.1923; short_above=0.8148; thresholds=Dev p20 / p80 | 20,595 | 33.8% | +0.012 | -0.202 | -0.209 | 0.000 | REJECTED_NO_EDGE |

### Breakout / range (10)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `squeeze_release` | After a volatility squeeze, take the expansion in the direction of the slope. | long=compression<p20 & range>p50 & slope>0; short=mirror | 6,638 | 33.2% | -0.008 | -0.229 | -0.165 | 0.000 | REJECTED_NO_EDGE |
| `volume_breakout` | Long on a session high confirmed by a volume z-spike; short on a confirmed low. | long=new high & volume z>p80 & body>0; short=mirror | 7,473 | 32.8% | -0.020 | -0.235 | -0.200 | 0.000 | REJECTED_NO_EDGE |
| `session_break` | Long on a new session-range high with a positive body; short on a new low with a negative body. | long=session_range_position>p90 & body>0; short=mirror | 10,258 | 32.9% | -0.016 | -0.234 | -0.199 | 0.000 | REJECTED_NO_EDGE |
| `session_pos_follow` | Long when session_range_position > 0.9 (upper extreme); short when < 0.108. | feature=session_range_position; long_above=0.9; short_below=0.1084; thresholds=Dev p10 / p90 | 10,378 | 32.9% | -0.014 | -0.232 | -0.201 | 0.000 | REJECTED_NO_EDGE |
| `donchian60_break` | Long on a fresh 60-minute high with a positive body; short on a fresh low. | long=rolling_range_position_60>p90 & body>0; short=mirror | 11,464 | 32.8% | -0.019 | -0.236 | -0.189 | 0.000 | REJECTED_NO_EDGE |
| `rsi60_follow` | Long when rolling_range_position_60 > 0.9 (upper extreme); short when < 0.109. | feature=rolling_range_position_60; long_above=0.9; short_below=0.1091; thresholds=Dev p10 / p90 | 11,582 | 32.7% | -0.022 | -0.239 | -0.186 | 0.000 | REJECTED_NO_EDGE |
| `inside_bar_break` | On an inside bar, go with the sign of the body. | long=inside_bar & body>0; short=inside_bar & body<0 | 16,249 | 32.8% | -0.019 | -0.234 | -0.159 | 0.000 | REJECTED_NO_EDGE |
| `wide_range_bar` | Long on a bar more than twice the prior range closing up; short if closing down. | long=range>2x prior & body>0; short=range>2x prior & body<0 | 14,213 | 32.7% | -0.022 | -0.240 | -0.182 | 0.000 | REJECTED_NO_EDGE |
| `outside_bar_break` | On an outside (engulfing) bar, go with the sign of the body. | long=outside_bar & body>0; short=outside_bar & body<0 | 15,047 | 33.2% | -0.008 | -0.225 | -0.197 | 0.000 | REJECTED_NO_EDGE |
| `range_expansion` | Long on an expansion bar closing up; short on an expansion bar closing down. | long=current_range_over_atr>p80 & body>0; short=range>p80 & body<0 | 19,774 | 32.6% | -0.022 | -0.238 | -0.187 | 0.000 | REJECTED_NO_EDGE |

### Volatility-regime (2)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `atr_ratio_cross` | Long when atr_ratio_5_20 > atr_ratio_20_60; short when atr_ratio_5_20 < atr_ratio_20_60. | fast=atr_ratio_5_20; slow=atr_ratio_20_60 | 23,204 | 33.4% | +0.001 | -0.212 | -0.166 | 0.000 | REJECTED_NO_EDGE |
| `rvol_ratio_cross` | Long when realized_volatility_ratio_5_30 > realized_volatility_ratio_15_60; short when realized_volatility_ratio_5_30 < realized_volatility_ratio_15_60. | fast=realized_volatility_ratio_5_30; slow=realized_volatility_ratio_15_60 | 23,269 | 33.3% | -0.003 | -0.217 | -0.190 | 0.000 | REJECTED_NO_EDGE |

### Volume / flow (10)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `volume_dryup_reversion` | When participation dries up, fade the session-range extreme. | gate=relative_volume_20<p20; base=fade session position | 6,015 | 34.2% | +0.022 | -0.195 | -0.198 | 0.000 | REJECTED_NO_EDGE |
| `climax_fade` | Fade a blow-off: buy a high-volume down bar, sell a high-volume up bar. | long=volume z>p90 on a down bar (fade); short=volume z>p90 on an up bar | 13,998 | 33.9% | +0.017 | -0.197 | -0.170 | 0.000 | REJECTED_NO_EDGE |
| `vpa_follow_x` | Long when volume_price_alignment_10 > 0.505; short when < -0.487 (extreme tails). | feature=volume_price_alignment_10; long_above=0.5051; short_below=-0.4868; thresholds=Dev p10 / p90 | 13,714 | 32.4% | -0.030 | -0.243 | -0.234 | 0.000 | REJECTED_NO_EDGE |
| `rvol_momentum` | Follow the 15-minute move only when relative volume is elevated. | gate=relative_volume_20>p80; base=sign(return_15m) | 18,830 | 33.0% | -0.013 | -0.231 | -0.220 | 0.000 | REJECTED_NO_EDGE |
| `tod_volume_surprise` | Follow the body direction when volume exceeds its time-of-day norm. | gate=tod_log_volume_z>p80; base=sign(body) | 16,079 | 32.0% | -0.041 | -0.251 | -0.182 | 0.000 | REJECTED_NO_EDGE |
| `obv_follow_x` | Long when signed_volume_proxy > 0.891; short when < -0.883 (extreme tails). | feature=signed_volume_proxy; long_above=0.8913; short_below=-0.8828; thresholds=Dev p10 / p90 | 18,624 | 32.5% | -0.027 | -0.245 | -0.191 | 0.000 | REJECTED_NO_EDGE |
| `vpa_follow` | Long when volume_price_alignment_10 > 0.356; short when < -0.333. | feature=volume_price_alignment_10; long_above=0.3562; short_below=-0.3326; thresholds=Dev p20 / p80 | 18,436 | 32.6% | -0.026 | -0.239 | -0.224 | 0.000 | REJECTED_NO_EDGE |
| `obv_follow` | Long when signed_volume_proxy > 0.562; short when < -0.558. | feature=signed_volume_proxy; long_above=0.5616; short_below=-0.5584; thresholds=Dev p20 / p80 | 21,810 | 32.8% | -0.019 | -0.233 | -0.190 | 0.000 | REJECTED_NO_EDGE |
| `flow_sign` | Long when signed_volume_proxy > 0; short when < 0. | feature=signed_volume_proxy; level=0 | 23,324 | 33.0% | -0.011 | -0.225 | -0.182 | 0.000 | REJECTED_NO_EDGE |
| `volume_accel_momentum` | Follow the body direction when volume is accelerating. | gate=volume_acceleration_5_20>0; base=sign(body) | 23,513 | 32.4% | -0.029 | -0.243 | -0.182 | 0.000 | REJECTED_NO_EDGE |

### Candle / price-action (6)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `key_reversal` | Long when price makes a new low but closes near its high; short the mirror. | long=new short-term low but close high; short=new high but close low | 4,088 | 33.9% | +0.013 | -0.212 | -0.232 | 0.000 | REJECTED_NO_EDGE |
| `wick_rejection_level` | Buy a lower-wick rejection at the session low; sell an upper-wick rejection at the high. | long=lower wick at session low; short=upper wick at session high | 7,753 | 34.0% | +0.014 | -0.198 | -0.203 | 0.000 | REJECTED_NO_EDGE |
| `doji_fade` | Fade a small-bodied indecision bar back toward the mean from a range extreme. | long=tiny body at range low; short=tiny body at range high | 10,242 | 33.5% | +0.002 | -0.213 | -0.215 | 0.000 | REJECTED_NO_EDGE |
| `pin_bar` | Long a hammer (long lower wick, close near high); short a shooting star. | long=long lower wick & close high; short=long upper wick & close low | 11,257 | 33.5% | +0.002 | -0.216 | -0.208 | 0.000 | REJECTED_NO_EDGE |
| `marubozu` | Long a strong full-bodied up bar; short a full-bodied down bar. | long=body/range>p80 & body>0; short=body/range>p80 & body<0 | 17,047 | 32.8% | -0.017 | -0.233 | -0.206 | 0.000 | REJECTED_NO_EDGE |
| `clv_sign` | Long when close_location_value > 0; short when < 0. | feature=close_location_value; level=0 | 23,324 | 33.0% | -0.011 | -0.225 | -0.182 | 0.000 | REJECTED_NO_EDGE |

### VWAP-relative (7)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `vwap_trend_confluence` | Long when 60-bar VWAP distance>0 and VWAP slope>0; short when both<0. | features=vwap distance & slope | 18,219 | 33.2% | -0.007 | -0.220 | -0.171 | 0.000 | REJECTED_NO_EDGE |
| `above_session_vwap` | Long when distance_from_execution_session_vwap_atr > 0; short when < 0. | feature=distance_from_execution_session_vwap_atr; level=0 | 23,379 | 33.3% | -0.004 | -0.218 | -0.185 | 0.000 | REJECTED_NO_EDGE |
| `vwap_band_cross` | Long when distance_from_rolling_vwap_20_atr > distance_from_rolling_vwap_60_atr; short when distance_from_rolling_vwap_20_atr < distance_from_rolling_vwap_60_atr. | fast=distance_from_rolling_vwap_20_atr; slow=distance_from_rolling_vwap_60_atr | 23,103 | 33.1% | -0.008 | -0.222 | -0.187 | 0.000 | REJECTED_NO_EDGE |
| `vwap_slope_cross` | Long when vwap_slope_5_atr > vwap_slope_15_atr; short when vwap_slope_5_atr < vwap_slope_15_atr. | fast=vwap_slope_5_atr; slow=vwap_slope_15_atr | 23,147 | 33.5% | +0.003 | -0.211 | -0.217 | 0.000 | REJECTED_NO_EDGE |
| `above_rolling_vwap_60` | Long when distance_from_rolling_vwap_60_atr > 0; short when < 0. | feature=distance_from_rolling_vwap_60_atr; level=0 | 23,568 | 33.1% | -0.010 | -0.224 | -0.191 | 0.000 | REJECTED_NO_EDGE |
| `vwap_majority_side` | Long when fraction_above_vwap_15 > 0.5; short when < 0.5. | feature=fraction_above_vwap_15; level=0.5 | 23,389 | 33.1% | -0.010 | -0.223 | -0.178 | 0.000 | REJECTED_NO_EDGE |
| `open_vs_vwap_cross` | Long when distance_from_session_open_atr > distance_from_execution_session_vwap_atr; short when distance_from_session_open_atr < distance_from_execution_session_vwap_atr. | fast=distance_from_session_open_atr; slow=distance_from_execution_session_vwap_atr | 23,126 | 32.9% | -0.016 | -0.230 | -0.204 | 0.000 | REJECTED_NO_EDGE |

### Session / time-of-day (7)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `late_session_trend` | Follow normalized_ols_slope_30 only when session_progress_fraction indicates the trend late in the session. | gate=session_progress_fraction gt p80; base=sign(normalized_ols_slope_30) | 4,926 | 34.0% | +0.015 | -0.213 | -0.239 | 0.000 | REJECTED_NO_EDGE |
| `london_breakout` | Take session-range breakouts only during the London window. | session=London; base=session breakout | 3,636 | 32.8% | -0.019 | -0.264 | -0.238 | 0.000 | REJECTED_NO_EDGE |
| `newyork_breakout` | Take session-range breakouts only during the New York window. | session=New York; base=session breakout | 6,628 | 32.9% | -0.015 | -0.218 | -0.175 | 0.000 | REJECTED_NO_EDGE |
| `early_session_trend` | Follow return_30m_atr only when session_progress_fraction indicates momentum early in the session. | gate=session_progress_fraction lt p20; base=sign(return_30m_atr) | 5,411 | 32.0% | -0.042 | -0.287 | -0.233 | 0.000 | REJECTED_NO_EDGE |
| `opening_drive` | Follow return_5m_atr only when minute_from_execution_window_open indicates the opening drift, in the first minutes of the window. | gate=minute_from_execution_window_open lt p20; base=sign(return_5m_atr) | 5,448 | 31.3% | -0.061 | -0.306 | -0.233 | 0.000 | REJECTED_NO_EDGE |
| `london_momentum` | Follow the 30-minute move only during the London window. | session=London; base=sign(return_30m) | 7,925 | 32.3% | -0.032 | -0.277 | -0.228 | 0.000 | REJECTED_NO_EDGE |
| `newyork_momentum` | Follow the 30-minute move only during the New York window. | session=New York; base=sign(return_30m) | 15,687 | 33.6% | +0.004 | -0.194 | -0.178 | 0.000 | REJECTED_NO_EDGE |

### Regime-gated (89)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `fade_vwapS_highchop` | Fade distance_from_execution_session_vwap_atr only when choppiness_14 indicates a high-choppiness regime. | base=fade(distance_from_execution_session_vwap_atr); gate=choppiness_14 gt p80; gate_value=56.993 | 3,890 | 34.3% | +0.021 | -0.187 | -0.193 | 0.000 | REJECTED_NO_EDGE |
| `fade_ret15_highchop` | Fade return_15m_atr only when choppiness_14 indicates a high-choppiness regime. | base=fade(return_15m_atr); gate=choppiness_14 gt p80; gate_value=56.993 | 4,007 | 34.2% | +0.023 | -0.195 | -0.205 | 0.000 | REJECTED_NO_EDGE |
| `fade_vwapS_nontrend` | Fade distance_from_execution_session_vwap_atr only when ols_r_squared_30 indicates a non-trending (low R-squared) regime. | base=fade(distance_from_execution_session_vwap_atr); gate=ols_r_squared_30 lt p20; gate_value=0.09 | 4,217 | 33.6% | +0.004 | -0.214 | -0.204 | 0.000 | REJECTED_NO_EDGE |
| `fade_vwap20_highchop` | Fade distance_from_rolling_vwap_20_atr only when choppiness_14 indicates a high-choppiness regime. | base=fade(distance_from_rolling_vwap_20_atr); gate=choppiness_14 gt p80; gate_value=56.993 | 3,726 | 32.6% | -0.026 | -0.246 | -0.172 | 0.000 | REJECTED_NO_EDGE |
| `accel_widerange` | Follow momentum_acceleration_5_15 only when session_range_over_atr indicates a wide developing session range. | base=sign(momentum_acceleration_5_15); gate=session_range_over_atr gt p80; gate_value=17 | 5,240 | 33.9% | +0.014 | -0.203 | -0.179 | 0.000 | REJECTED_NO_EDGE |
| `body_widerange` | Follow signed_body_atr only when session_range_over_atr indicates a wide developing session range. | base=sign(signed_body_atr); gate=session_range_over_atr gt p80; gate_value=17 | 5,135 | 33.9% | +0.013 | -0.204 | -0.188 | 0.000 | REJECTED_NO_EDGE |
| `mom60_widerange` | Follow return_60m_atr only when session_range_over_atr indicates a wide developing session range. | base=sign(return_60m_atr); gate=session_range_over_atr gt p80; gate_value=17 | 5,313 | 34.2% | +0.020 | -0.197 | -0.192 | 0.000 | REJECTED_NO_EDGE |
| `slope60_widerange` | Follow normalized_ols_slope_60 only when session_range_over_atr indicates a wide developing session range. | base=sign(normalized_ols_slope_60); gate=session_range_over_atr gt p80; gate_value=17 | 5,297 | 34.0% | +0.016 | -0.202 | -0.205 | 0.000 | REJECTED_NO_EDGE |
| `fade_vwapS_lowvol` | Fade distance_from_execution_session_vwap_atr only when realized_volatility_ratio_5_30 indicates a contracting-volatility regime. | base=fade(distance_from_execution_session_vwap_atr); gate=realized_volatility_ratio_5_30 lt p20; gate_value=0.262 | 5,325 | 34.0% | +0.016 | -0.194 | -0.232 | 0.000 | REJECTED_NO_EDGE |
| `fade_vwap60_highchop` | Fade distance_from_rolling_vwap_60_atr only when choppiness_14 indicates a high-choppiness regime. | base=fade(distance_from_rolling_vwap_60_atr); gate=choppiness_14 gt p80; gate_value=56.993 | 4,459 | 33.7% | +0.004 | -0.212 | -0.213 | 0.000 | REJECTED_NO_EDGE |
| `fade_vwap60_nontrend` | Fade distance_from_rolling_vwap_60_atr only when ols_r_squared_30 indicates a non-trending (low R-squared) regime. | base=fade(distance_from_rolling_vwap_60_atr); gate=ols_r_squared_30 lt p20; gate_value=0.09 | 4,094 | 33.6% | +0.003 | -0.224 | -0.239 | 0.000 | REJECTED_NO_EDGE |
| `mom15_widerange` | Follow return_15m_atr only when session_range_over_atr indicates a wide developing session range. | base=sign(return_15m_atr); gate=session_range_over_atr gt p80; gate_value=17 | 5,221 | 34.2% | +0.021 | -0.196 | -0.203 | 0.000 | REJECTED_NO_EDGE |
| `mom30_widerange` | Follow return_30m_atr only when session_range_over_atr indicates a wide developing session range. | base=sign(return_30m_atr); gate=session_range_over_atr gt p80; gate_value=17 | 5,298 | 33.8% | +0.009 | -0.207 | -0.213 | 0.000 | REJECTED_NO_EDGE |
| `slope30_widerange` | Follow normalized_ols_slope_30 only when session_range_over_atr indicates a wide developing session range. | base=sign(normalized_ols_slope_30); gate=session_range_over_atr gt p80; gate_value=17 | 5,303 | 33.7% | +0.005 | -0.213 | -0.229 | 0.000 | REJECTED_NO_EDGE |
| `fade_vwap60_lowvol` | Fade distance_from_rolling_vwap_60_atr only when realized_volatility_ratio_5_30 indicates a contracting-volatility regime. | base=fade(distance_from_rolling_vwap_60_atr); gate=realized_volatility_ratio_5_30 lt p20; gate_value=0.262 | 6,055 | 33.6% | +0.005 | -0.212 | -0.240 | 0.000 | REJECTED_NO_EDGE |
| `fade_ret15_lowvol` | Fade return_15m_atr only when realized_volatility_ratio_5_30 indicates a contracting-volatility regime. | base=fade(return_15m_atr); gate=realized_volatility_ratio_5_30 lt p20; gate_value=0.262 | 7,066 | 33.5% | -0.000 | -0.216 | -0.203 | 0.000 | REJECTED_NO_EDGE |
| `fade_ret15_nontrend` | Fade return_15m_atr only when ols_r_squared_30 indicates a non-trending (low R-squared) regime. | base=fade(return_15m_atr); gate=ols_r_squared_30 lt p20; gate_value=0.09 | 6,990 | 34.4% | +0.027 | -0.194 | -0.223 | 0.000 | REJECTED_NO_EDGE |
| `fade_vwap20_lowvol` | Fade distance_from_rolling_vwap_20_atr only when realized_volatility_ratio_5_30 indicates a contracting-volatility regime. | base=fade(distance_from_rolling_vwap_20_atr); gate=realized_volatility_ratio_5_30 lt p20; gate_value=0.262 | 6,239 | 33.2% | -0.012 | -0.228 | -0.221 | 0.000 | REJECTED_NO_EDGE |
| `slope60_volx2` | Follow normalized_ols_slope_60 only when atr_ratio_20_60 indicates expanding medium-term volatility. | base=sign(normalized_ols_slope_60); gate=atr_ratio_20_60 gt p80; gate_value=1.1553 | 8,933 | 33.2% | -0.006 | -0.204 | -0.161 | 0.000 | REJECTED_NO_EDGE |
| `slope30_volx2` | Follow normalized_ols_slope_30 only when atr_ratio_20_60 indicates expanding medium-term volatility. | base=sign(normalized_ols_slope_30); gate=atr_ratio_20_60 gt p80; gate_value=1.1553 | 8,929 | 32.7% | -0.023 | -0.222 | -0.175 | 0.000 | REJECTED_NO_EDGE |
| `mom30_volx2` | Follow return_30m_atr only when atr_ratio_20_60 indicates expanding medium-term volatility. | base=sign(return_30m_atr); gate=atr_ratio_20_60 gt p80; gate_value=1.1553 | 8,947 | 32.7% | -0.021 | -0.220 | -0.170 | 0.000 | REJECTED_NO_EDGE |
| `fade_vwap20_nontrend` | Fade distance_from_rolling_vwap_20_atr only when ols_r_squared_30 indicates a non-trending (low R-squared) regime. | base=fade(distance_from_rolling_vwap_20_atr); gate=ols_r_squared_30 lt p20; gate_value=0.09 | 7,977 | 33.3% | -0.005 | -0.227 | -0.210 | 0.000 | REJECTED_NO_EDGE |
| `mom15_volx2` | Follow return_15m_atr only when atr_ratio_20_60 indicates expanding medium-term volatility. | base=sign(return_15m_atr); gate=atr_ratio_20_60 gt p80; gate_value=1.1553 | 8,935 | 32.7% | -0.022 | -0.220 | -0.174 | 0.000 | REJECTED_NO_EDGE |
| `body_linear` | Follow signed_body_atr only when ols_r_squared_30 indicates a linear price path. | base=sign(signed_body_atr); gate=ols_r_squared_30 gt p80; gate_value=0.7288 | 8,959 | 33.2% | -0.008 | -0.220 | -0.217 | 0.000 | REJECTED_NO_EDGE |
| `mom60_linear` | Follow return_60m_atr only when ols_r_squared_30 indicates a linear price path. | base=sign(return_60m_atr); gate=ols_r_squared_30 gt p80; gate_value=0.7288 | 8,816 | 32.9% | -0.016 | -0.228 | -0.180 | 0.000 | REJECTED_NO_EDGE |
| `mom60_volx2` | Follow return_60m_atr only when atr_ratio_20_60 indicates expanding medium-term volatility. | base=sign(return_60m_atr); gate=atr_ratio_20_60 gt p80; gate_value=1.1553 | 8,927 | 32.6% | -0.025 | -0.223 | -0.155 | 0.000 | REJECTED_NO_EDGE |
| `accel_abovevwap` | Follow momentum_acceleration_5_15 only when fraction_above_vwap_15 indicates price mostly above VWAP. | base=sign(momentum_acceleration_5_15); gate=fraction_above_vwap_15 gt p50; gate_value=0.4667 | 12,689 | 33.2% | -0.006 | -0.223 | -0.182 | 0.000 | REJECTED_NO_EDGE |
| `accel_linear` | Follow momentum_acceleration_5_15 only when ols_r_squared_30 indicates a linear price path. | base=sign(momentum_acceleration_5_15); gate=ols_r_squared_30 gt p80; gate_value=0.7288 | 9,253 | 33.1% | -0.009 | -0.221 | -0.211 | 0.000 | REJECTED_NO_EDGE |
| `slope60_linear` | Follow normalized_ols_slope_60 only when ols_r_squared_30 indicates a linear price path. | base=sign(normalized_ols_slope_60); gate=ols_r_squared_30 gt p80; gate_value=0.7288 | 8,892 | 32.9% | -0.014 | -0.227 | -0.184 | 0.000 | REJECTED_NO_EDGE |
| `slope30_linear` | Follow normalized_ols_slope_30 only when ols_r_squared_30 indicates a linear price path. | base=sign(normalized_ols_slope_30); gate=ols_r_squared_30 gt p80; gate_value=0.7288 | 8,636 | 32.8% | -0.020 | -0.232 | -0.209 | 0.000 | REJECTED_NO_EDGE |
| `mom30_linear` | Follow return_30m_atr only when ols_r_squared_30 indicates a linear price path. | base=sign(return_30m_atr); gate=ols_r_squared_30 gt p80; gate_value=0.7288 | 8,632 | 32.8% | -0.020 | -0.231 | -0.207 | 0.000 | REJECTED_NO_EDGE |
| `body_volx2` | Follow signed_body_atr only when atr_ratio_20_60 indicates expanding medium-term volatility. | base=sign(signed_body_atr); gate=atr_ratio_20_60 gt p80; gate_value=1.1553 | 8,894 | 31.8% | -0.045 | -0.244 | -0.180 | 0.000 | REJECTED_NO_EDGE |
| `accel_volx2` | Follow momentum_acceleration_5_15 only when atr_ratio_20_60 indicates expanding medium-term volatility. | base=sign(momentum_acceleration_5_15); gate=atr_ratio_20_60 gt p80; gate_value=1.1553 | 8,959 | 32.0% | -0.041 | -0.239 | -0.203 | 0.000 | REJECTED_NO_EDGE |
| `mom15_linear` | Follow return_15m_atr only when ols_r_squared_30 indicates a linear price path. | base=sign(return_15m_atr); gate=ols_r_squared_30 gt p80; gate_value=0.7288 | 8,606 | 32.5% | -0.028 | -0.239 | -0.207 | 0.000 | REJECTED_NO_EDGE |
| `body_abovevwap` | Follow signed_body_atr only when fraction_above_vwap_15 indicates price mostly above VWAP. | base=sign(signed_body_atr); gate=fraction_above_vwap_15 gt p50; gate_value=0.4667 | 12,448 | 33.2% | -0.007 | -0.223 | -0.214 | 0.000 | REJECTED_NO_EDGE |
| `mom30_abovevwap` | Follow return_30m_atr only when fraction_above_vwap_15 indicates price mostly above VWAP. | base=sign(return_30m_atr); gate=fraction_above_vwap_15 gt p50; gate_value=0.4667 | 12,562 | 33.8% | +0.011 | -0.205 | -0.204 | 0.000 | REJECTED_NO_EDGE |
| `slope60_abovevwap` | Follow normalized_ols_slope_60 only when fraction_above_vwap_15 indicates price mostly above VWAP. | base=sign(normalized_ols_slope_60); gate=fraction_above_vwap_15 gt p50; gate_value=0.4667 | 12,448 | 33.7% | +0.009 | -0.207 | -0.180 | 0.000 | REJECTED_NO_EDGE |
| `mom60_trendy` | Follow return_60m_atr only when efficiency_ratio_30 indicates a strong (efficient) trend. | base=sign(return_60m_atr); gate=efficiency_ratio_30 gt p80; gate_value=0.2865 | 10,218 | 32.8% | -0.018 | -0.229 | -0.216 | 0.000 | REJECTED_NO_EDGE |
| `accel_trendy` | Follow momentum_acceleration_5_15 only when efficiency_ratio_30 indicates a strong (efficient) trend. | base=sign(momentum_acceleration_5_15); gate=efficiency_ratio_30 gt p80; gate_value=0.2865 | 10,607 | 32.9% | -0.015 | -0.227 | -0.218 | 0.000 | REJECTED_NO_EDGE |
| `mom15_abovevwap` | Follow return_15m_atr only when fraction_above_vwap_15 indicates price mostly above VWAP. | base=sign(return_15m_atr); gate=fraction_above_vwap_15 gt p50; gate_value=0.4667 | 12,671 | 33.1% | -0.009 | -0.225 | -0.211 | 0.000 | REJECTED_NO_EDGE |
| `mom30_trendy` | Follow return_30m_atr only when efficiency_ratio_30 indicates a strong (efficient) trend. | base=sign(return_30m_atr); gate=efficiency_ratio_30 gt p80; gate_value=0.2865 | 10,012 | 32.8% | -0.021 | -0.232 | -0.203 | 0.000 | REJECTED_NO_EDGE |
| `slope30_trendy` | Follow normalized_ols_slope_30 only when efficiency_ratio_30 indicates a strong (efficient) trend. | base=sign(normalized_ols_slope_30); gate=efficiency_ratio_30 gt p80; gate_value=0.2865 | 10,012 | 32.8% | -0.021 | -0.232 | -0.204 | 0.000 | REJECTED_NO_EDGE |
| `mom15_trendy` | Follow return_15m_atr only when efficiency_ratio_30 indicates a strong (efficient) trend. | base=sign(return_15m_atr); gate=efficiency_ratio_30 gt p80; gate_value=0.2865 | 10,067 | 32.5% | -0.028 | -0.240 | -0.213 | 0.000 | REJECTED_NO_EDGE |
| `mom60_abovevwap` | Follow return_60m_atr only when fraction_above_vwap_15 indicates price mostly above VWAP. | base=sign(return_60m_atr); gate=fraction_above_vwap_15 gt p50; gate_value=0.4667 | 12,466 | 33.8% | +0.012 | -0.204 | -0.192 | 0.000 | REJECTED_NO_EDGE |
| `slope30_abovevwap` | Follow normalized_ols_slope_30 only when fraction_above_vwap_15 indicates price mostly above VWAP. | base=sign(normalized_ols_slope_30); gate=fraction_above_vwap_15 gt p50; gate_value=0.4667 | 12,549 | 33.3% | -0.002 | -0.218 | -0.198 | 0.000 | REJECTED_NO_EDGE |
| `slope60_trendy` | Follow normalized_ols_slope_60 only when efficiency_ratio_30 indicates a strong (efficient) trend. | base=sign(normalized_ols_slope_60); gate=efficiency_ratio_30 gt p80; gate_value=0.2865 | 10,393 | 32.8% | -0.019 | -0.231 | -0.210 | 0.000 | REJECTED_NO_EDGE |
| `slope60_lowchop` | Follow normalized_ols_slope_60 only when choppiness_14 indicates a low-choppiness regime. | base=sign(normalized_ols_slope_60); gate=choppiness_14 lt p20; gate_value=40.2383 | 12,526 | 33.5% | +0.003 | -0.212 | -0.193 | 0.000 | REJECTED_NO_EDGE |
| `slope30_lowchop` | Follow normalized_ols_slope_30 only when choppiness_14 indicates a low-choppiness regime. | base=sign(normalized_ols_slope_30); gate=choppiness_14 lt p20; gate_value=40.2383 | 12,425 | 33.1% | -0.011 | -0.226 | -0.206 | 0.000 | REJECTED_NO_EDGE |
| `mom60_lowchop` | Follow return_60m_atr only when choppiness_14 indicates a low-choppiness regime. | base=sign(return_60m_atr); gate=choppiness_14 lt p20; gate_value=40.2383 | 12,417 | 33.0% | -0.013 | -0.228 | -0.174 | 0.000 | REJECTED_NO_EDGE |
| `slope30_rvx` | Follow normalized_ols_slope_30 only when realized_volatility_ratio_5_30 indicates expanding realized volatility. | base=sign(normalized_ols_slope_30); gate=realized_volatility_ratio_5_30 gt p80; gate_value=0.5008 | 14,860 | 33.2% | -0.006 | -0.222 | -0.199 | 0.000 | REJECTED_NO_EDGE |
| `mom30_lowchop` | Follow return_30m_atr only when choppiness_14 indicates a low-choppiness regime. | base=sign(return_30m_atr); gate=choppiness_14 lt p20; gate_value=40.2383 | 12,296 | 32.7% | -0.022 | -0.237 | -0.188 | 0.000 | REJECTED_NO_EDGE |
| `mom15_lowchop` | Follow return_15m_atr only when choppiness_14 indicates a low-choppiness regime. | base=sign(return_15m_atr); gate=choppiness_14 lt p20; gate_value=40.2383 | 12,239 | 32.3% | -0.034 | -0.249 | -0.172 | 0.000 | REJECTED_NO_EDGE |
| `body_trendy` | Follow signed_body_atr only when efficiency_ratio_30 indicates a strong (efficient) trend. | base=sign(signed_body_atr); gate=efficiency_ratio_30 gt p80; gate_value=0.2865 | 10,232 | 32.1% | -0.040 | -0.251 | -0.212 | 0.000 | REJECTED_NO_EDGE |
| `body_lowchop` | Follow signed_body_atr only when choppiness_14 indicates a low-choppiness regime. | base=sign(signed_body_atr); gate=choppiness_14 lt p20; gate_value=40.2383 | 11,976 | 32.1% | -0.038 | -0.253 | -0.179 | 0.000 | REJECTED_NO_EDGE |
| `slope60_volx` | Follow normalized_ols_slope_60 only when atr_ratio_5_20 indicates expanding short-term volatility. | base=sign(normalized_ols_slope_60); gate=atr_ratio_5_20 gt p80; gate_value=1.1743 | 15,360 | 33.7% | +0.008 | -0.211 | -0.187 | 0.000 | REJECTED_NO_EDGE |
| `mom30_rvx` | Follow return_30m_atr only when realized_volatility_ratio_5_30 indicates expanding realized volatility. | base=sign(return_30m_atr); gate=realized_volatility_ratio_5_30 gt p80; gate_value=0.5008 | 14,853 | 33.1% | -0.009 | -0.226 | -0.198 | 0.000 | REJECTED_NO_EDGE |
| `mom60_volx` | Follow return_60m_atr only when atr_ratio_5_20 indicates expanding short-term volatility. | base=sign(return_60m_atr); gate=atr_ratio_5_20 gt p80; gate_value=1.1743 | 15,325 | 33.4% | -0.000 | -0.219 | -0.194 | 0.000 | REJECTED_NO_EDGE |
| `mom30_volx` | Follow return_30m_atr only when atr_ratio_5_20 indicates expanding short-term volatility. | base=sign(return_30m_atr); gate=atr_ratio_5_20 gt p80; gate_value=1.1743 | 15,395 | 33.0% | -0.013 | -0.231 | -0.183 | 0.000 | REJECTED_NO_EDGE |
| `slope30_activetod` | Follow normalized_ols_slope_30 only when tod_relative_volume indicates an active time-of-day window. | base=sign(normalized_ols_slope_30); gate=tod_relative_volume gt p80; gate_value=1.8732 | 16,503 | 33.1% | -0.012 | -0.222 | -0.197 | 0.000 | REJECTED_NO_EDGE |
| `slope30_volx` | Follow normalized_ols_slope_30 only when atr_ratio_5_20 indicates expanding short-term volatility. | base=sign(normalized_ols_slope_30); gate=atr_ratio_5_20 gt p80; gate_value=1.1743 | 15,448 | 33.1% | -0.010 | -0.228 | -0.188 | 0.000 | REJECTED_NO_EDGE |
| `accel_lowchop` | Follow momentum_acceleration_5_15 only when choppiness_14 indicates a low-choppiness regime. | base=sign(momentum_acceleration_5_15); gate=choppiness_14 lt p20; gate_value=40.2383 | 12,197 | 32.2% | -0.037 | -0.252 | -0.201 | 0.000 | REJECTED_NO_EDGE |
| `body_rvx` | Follow signed_body_atr only when realized_volatility_ratio_5_30 indicates expanding realized volatility. | base=sign(signed_body_atr); gate=realized_volatility_ratio_5_30 gt p80; gate_value=0.5008 | 14,799 | 33.0% | -0.012 | -0.227 | -0.190 | 0.000 | REJECTED_NO_EDGE |
| `accel_rvx` | Follow momentum_acceleration_5_15 only when realized_volatility_ratio_5_30 indicates expanding realized volatility. | base=sign(momentum_acceleration_5_15); gate=realized_volatility_ratio_5_30 gt p80; gate_value=0.5008 | 14,876 | 33.0% | -0.011 | -0.228 | -0.211 | 0.000 | REJECTED_NO_EDGE |
| `slope60_activetod` | Follow normalized_ols_slope_60 only when tod_relative_volume indicates an active time-of-day window. | base=sign(normalized_ols_slope_60); gate=tod_relative_volume gt p80; gate_value=1.8732 | 16,527 | 33.4% | -0.002 | -0.212 | -0.191 | 0.000 | REJECTED_NO_EDGE |
| `slope60_rvx` | Follow normalized_ols_slope_60 only when realized_volatility_ratio_5_30 indicates expanding realized volatility. | base=sign(normalized_ols_slope_60); gate=realized_volatility_ratio_5_30 gt p80; gate_value=0.5008 | 14,823 | 33.3% | -0.004 | -0.220 | -0.209 | 0.000 | REJECTED_NO_EDGE |
| `mom15_volx` | Follow return_15m_atr only when atr_ratio_5_20 indicates expanding short-term volatility. | base=sign(return_15m_atr); gate=atr_ratio_5_20 gt p80; gate_value=1.1743 | 15,347 | 32.9% | -0.014 | -0.233 | -0.193 | 0.000 | REJECTED_NO_EDGE |
| `mom15_rvx` | Follow return_15m_atr only when realized_volatility_ratio_5_30 indicates expanding realized volatility. | base=sign(return_15m_atr); gate=realized_volatility_ratio_5_30 gt p80; gate_value=0.5008 | 14,847 | 33.1% | -0.010 | -0.226 | -0.205 | 0.000 | REJECTED_NO_EDGE |
| `mom60_rvx` | Follow return_60m_atr only when realized_volatility_ratio_5_30 indicates expanding realized volatility. | base=sign(return_60m_atr); gate=realized_volatility_ratio_5_30 gt p80; gate_value=0.5008 | 14,814 | 33.0% | -0.012 | -0.229 | -0.194 | 0.000 | REJECTED_NO_EDGE |
| `mom30_activetod` | Follow return_30m_atr only when tod_relative_volume indicates an active time-of-day window. | base=sign(return_30m_atr); gate=tod_relative_volume gt p80; gate_value=1.8732 | 16,200 | 32.8% | -0.020 | -0.230 | -0.199 | 0.000 | REJECTED_NO_EDGE |
| `accel_volx` | Follow momentum_acceleration_5_15 only when atr_ratio_5_20 indicates expanding short-term volatility. | base=sign(momentum_acceleration_5_15); gate=atr_ratio_5_20 gt p80; gate_value=1.1743 | 15,447 | 32.7% | -0.021 | -0.240 | -0.219 | 0.000 | REJECTED_NO_EDGE |
| `mom15_activetod` | Follow return_15m_atr only when tod_relative_volume indicates an active time-of-day window. | base=sign(return_15m_atr); gate=tod_relative_volume gt p80; gate_value=1.8732 | 16,169 | 32.5% | -0.027 | -0.236 | -0.198 | 0.000 | REJECTED_NO_EDGE |
| `slope30_highvol` | Follow normalized_ols_slope_30 only when relative_volume_60 indicates elevated relative volume. | base=sign(normalized_ols_slope_30); gate=relative_volume_60 gt p80; gate_value=1.4599 | 18,091 | 33.2% | -0.007 | -0.222 | -0.203 | 0.000 | REJECTED_NO_EDGE |
| `mom60_activetod` | Follow return_60m_atr only when tod_relative_volume indicates an active time-of-day window. | base=sign(return_60m_atr); gate=tod_relative_volume gt p80; gate_value=1.8732 | 16,325 | 33.1% | -0.010 | -0.220 | -0.198 | 0.000 | REJECTED_NO_EDGE |
| `slope30_volsurge` | Follow normalized_ols_slope_30 only when volume_zscore_60 indicates a volume surge. | base=sign(normalized_ols_slope_30); gate=volume_zscore_60 gt p80; gate_value=0.9331 | 18,214 | 33.1% | -0.009 | -0.224 | -0.202 | 0.000 | REJECTED_NO_EDGE |
| `body_volx` | Follow signed_body_atr only when atr_ratio_5_20 indicates expanding short-term volatility. | base=sign(signed_body_atr); gate=atr_ratio_5_20 gt p80; gate_value=1.1743 | 15,271 | 32.3% | -0.033 | -0.251 | -0.172 | 0.000 | REJECTED_NO_EDGE |
| `slope60_highvol` | Follow normalized_ols_slope_60 only when relative_volume_60 indicates elevated relative volume. | base=sign(normalized_ols_slope_60); gate=relative_volume_60 gt p80; gate_value=1.4599 | 18,151 | 33.4% | -0.000 | -0.215 | -0.197 | 0.000 | REJECTED_NO_EDGE |
| `slope60_volsurge` | Follow normalized_ols_slope_60 only when volume_zscore_60 indicates a volume surge. | base=sign(normalized_ols_slope_60); gate=volume_zscore_60 gt p80; gate_value=0.9331 | 18,311 | 33.3% | -0.002 | -0.216 | -0.193 | 0.000 | REJECTED_NO_EDGE |
| `mom15_highvol` | Follow return_15m_atr only when relative_volume_60 indicates elevated relative volume. | base=sign(return_15m_atr); gate=relative_volume_60 gt p80; gate_value=1.4599 | 17,809 | 32.9% | -0.016 | -0.231 | -0.196 | 0.000 | REJECTED_NO_EDGE |
| `accel_highvol` | Follow momentum_acceleration_5_15 only when relative_volume_60 indicates elevated relative volume. | base=sign(momentum_acceleration_5_15); gate=relative_volume_60 gt p80; gate_value=1.4599 | 17,873 | 32.9% | -0.015 | -0.230 | -0.193 | 0.000 | REJECTED_NO_EDGE |
| `accel_volsurge` | Follow momentum_acceleration_5_15 only when volume_zscore_60 indicates a volume surge. | base=sign(momentum_acceleration_5_15); gate=volume_zscore_60 gt p80; gate_value=0.9331 | 18,004 | 32.9% | -0.016 | -0.230 | -0.198 | 0.000 | REJECTED_NO_EDGE |
| `mom15_volsurge` | Follow return_15m_atr only when volume_zscore_60 indicates a volume surge. | base=sign(return_15m_atr); gate=volume_zscore_60 gt p80; gate_value=0.9331 | 17,915 | 32.8% | -0.017 | -0.231 | -0.196 | 0.000 | REJECTED_NO_EDGE |
| `mom30_highvol` | Follow return_30m_atr only when relative_volume_60 indicates elevated relative volume. | base=sign(return_30m_atr); gate=relative_volume_60 gt p80; gate_value=1.4599 | 17,831 | 32.9% | -0.016 | -0.231 | -0.204 | 0.000 | REJECTED_NO_EDGE |
| `accel_activetod` | Follow momentum_acceleration_5_15 only when tod_relative_volume indicates an active time-of-day window. | base=sign(momentum_acceleration_5_15); gate=tod_relative_volume gt p80; gate_value=1.8732 | 16,337 | 32.4% | -0.030 | -0.240 | -0.197 | 0.000 | REJECTED_NO_EDGE |
| `body_activetod` | Follow signed_body_atr only when tod_relative_volume indicates an active time-of-day window. | base=sign(signed_body_atr); gate=tod_relative_volume gt p80; gate_value=1.8732 | 16,062 | 32.2% | -0.036 | -0.246 | -0.182 | 0.000 | REJECTED_NO_EDGE |
| `mom30_volsurge` | Follow return_30m_atr only when volume_zscore_60 indicates a volume surge. | base=sign(return_30m_atr); gate=volume_zscore_60 gt p80; gate_value=0.9331 | 17,935 | 32.9% | -0.017 | -0.232 | -0.202 | 0.000 | REJECTED_NO_EDGE |
| `mom60_highvol` | Follow return_60m_atr only when relative_volume_60 indicates elevated relative volume. | base=sign(return_60m_atr); gate=relative_volume_60 gt p80; gate_value=1.4599 | 17,982 | 33.0% | -0.012 | -0.227 | -0.205 | 0.000 | REJECTED_NO_EDGE |
| `mom60_volsurge` | Follow return_60m_atr only when volume_zscore_60 indicates a volume surge. | base=sign(return_60m_atr); gate=volume_zscore_60 gt p80; gate_value=0.9331 | 18,126 | 33.0% | -0.013 | -0.228 | -0.197 | 0.000 | REJECTED_NO_EDGE |
| `body_highvol` | Follow signed_body_atr only when relative_volume_60 indicates elevated relative volume. | base=sign(signed_body_atr); gate=relative_volume_60 gt p80; gate_value=1.4599 | 17,642 | 32.5% | -0.028 | -0.242 | -0.195 | 0.000 | REJECTED_NO_EDGE |
| `body_volsurge` | Follow signed_body_atr only when volume_zscore_60 indicates a volume surge. | base=sign(signed_body_atr); gate=volume_zscore_60 gt p80; gate_value=0.9331 | 17,748 | 32.5% | -0.027 | -0.242 | -0.195 | 0.000 | REJECTED_NO_EDGE |

### Null benchmarks (2)

| Strategy | Looks for | Params | Trades | Win | Gross | Dev R | Val R | DSR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `always_short` | Always short. | - | 23,365 | 33.5% | +0.003 | -0.211 | -0.202 | 0.000 | BENCHMARK |
| `always_long` | Always long. | - | 23,167 | 33.1% | -0.010 | -0.224 | -0.192 | 0.000 | BENCHMARK |

## Where this points next

- **Exit structure is the untested axis.** Every strategy here shares one exit (1.5x ATR stop, 2R target, 120-minute cap), so this was purely an entry search. Because a trade's exit depends only on its entry bar, direction, and stop/target - not on which strategy opened it - exit configurations can be precomputed once and swept cheaply across all entries. A future run should declare an exit grid (stop multiple, R-multiple, trailing and breakeven variants, time exits) in a frozen contract before running, and feed the grid size into the deflation.
- **Direction is the wrong question; expansion/volatility is the open one.** Section 9 already found predictable structure in future *range* where direction has none. The faint positive gross tilt in this search is concentrated in fade-in-choppy-regime rules, consistent with mean-reversion/range carrying more signal than direction at this horizon.
- **Tooling.** Taking this out of notebooks into an interactive chart-and-simulate surface (TradingView-style) is worth doing with open-source parts rather than rebuilding from scratch: TradingView Lightweight Charts for rendering, a vetted backtest/simulation engine (vectorbt, nautilus_trader, backtrader, or QuantConnect LEAN) behind this verified engine, and a thin UI (Streamlit or Dash). Scope a small spike before committing to one.

## Reproduce

```
python scripts/run_strategy_catalog.py       # build, simulate, evaluate, save
python scripts/build_strategy_catalog_doc.py  # regenerate this document
```

Generated tables (parquet/CSV) stay out of Git by policy; this markdown and the tracked summaries are the record.
