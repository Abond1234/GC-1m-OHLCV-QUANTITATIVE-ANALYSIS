# MLAT Feature Research Contract v1

Status: FROZEN BEFORE MLAT OUTCOME EVALUATION

## Scope

- GC only; MGC is inaccessible to feature construction and evaluation.
- Features use information available at or before the close of decision bar
  `t`; theoretical entry is the open of `t+1`.
- London and New York are reported separately.
- Six fixed horizons are 5, 15, 30, 60, 120, and 180 minutes.
- Development and Validation only. Historical Final-test outcomes are excluded.
- No signal construction, PnL backtest, or Sharpe is authorized.

## Engineering contract

- Predictor names and order equal the frozen registry exactly.
- Rolling calculations use full chronological GC history and reset at every
  missing-minute, product, contract, instrument, segment, roll, tradability, or
  liquidity boundary.
- No forward, MFE, MAE, future range/volatility, direction, expansion, target,
  stop, POI, or entry-bar OHLCV may enter engineering.
- Complete windows are required. No backward fill or cross-boundary fill.
- Float64 calculations precede validated float32 persistence.
- Feature artifact save/reload must preserve IDs, order, dtypes, timestamps,
  registry alignment, null counts, and deterministic sample hash.

## Evaluation frame

- Evaluation seed: `20260723`.
- Date-block bootstrap: 2,000 replicates, 95% percentile interval.
- Same complete sample is used across the six horizons within each target
  family; ATR-normalized targets require valid decision ATR.
- Targets:
  - direction: `forward_return_{h}_atr` with tick spread reported;
  - expansion: `future_range_{h}_atr`;
  - volatility: `future_realized_volatility_{h}_bps`;
  - path risk: `max(mae_long_{h}_atr, mae_short_{h}_atr)`.
- Primary evidence is the mean per-New-York-date Spearman rank IC.
- Each daily IC requires at least 10 observations.
- Development-fitted quintiles are session-specific and applied unchanged to
  Validation.
- Benjamini-Hochberg q-values are computed within Development
  target-family x session screens across the full batch and all horizons.
- Horizon-thinned sensitivity keeps chronologically spaced observations within
  date/session; sign disagreement blocks advancement.

## Minimum evidence

- Development q-value <= 0.10.
- At least 400 Development and 150 Validation trading dates.
- At least 10,000 Development and 4,000 Validation finite observations.
- Validation IC sign agreement and retention >= 25% of Development magnitude.
- Absolute Development quintile monotonicity >= 0.8.
- Direction also requires absolute top-minus-bottom spread >= 2 GC ticks in
  both Development and Validation.
- An exact duplicate or Development absolute Spearman >= 0.995 with an
  existing feature is `REDUNDANT_WITH_EXISTING`.
- Incremental screen at 60 and 180 minutes:
  - beyond `atr_20`: Development absolute partial daily IC >= 0.02;
  - beyond the frozen 15-feature set: Development absolute partial daily IC
    >= 0.01;
  - both require Validation sign agreement and >=25% retention.
- Coverage exceeding 5% missingness is `FAILED_VALIDATION` unless a stricter
  feature-specific registry policy is declared before outcomes.

## Verdict mapping

- `ADVANCE_DIRECTIONAL`: every direction and incremental/economic gate passes.
- `ADVANCE_EXPANSION`: an expansion or volatility target passes and adds
  incremental information beyond existing expansion anchors.
- `ADVANCE_RISK_STATE`: a path-risk/volatility state passes and is not merely a
  level replica.
- `REDUNDANT_WITH_EXISTING`: valid feature but overlap veto fires.
- `ECONOMICALLY_TRIVIAL`: statistically confirmed direction below the tick floor.
- `WEAK_OR_UNSTABLE`: Development evidence does not retain in Validation,
  sessions, years, or thinning.
- `FAILED_VALIDATION`: engineering, coverage, numerical, boundary, or
  persistence gate fails.
- `NO_EVIDENCE`: no predeclared screen passes.
- Rejected/deferred catalog entries retain their explicit
  `REJECTED_LEAKAGE`, `REJECTED_IMPLEMENTATION`, or `DEFER_DIFFERENT_DATA`
  reason.

## GARCH audit protocol

- Audit GC only; never pool an MGC value or threshold.
- Returns reset at the established continuity boundary.
- Parameters fit only on GC history ending 2022-12-31.
- 2023 is a Development audit period; 2024 is Validation.
- Forecast initialization is chronological and segment-specific.
- Record optimizer result, persistence, standardized-residual and squared-
  residual diagnostics, ARCH effects, calibration, and simple-anchor metrics.
- Reject advancement on optimizer failure, persistence >= 0.999, invalid
  parameters, material residual structure, or no Validation improvement over
  the simple frozen volatility anchor.
- The model is audit-only in v1, even if diagnostics pass; adding it to a
  registry requires a new frozen experiment.

## Multivariate authorization gate

At least three non-redundant MLAT features must survive univariate, stability,
thinning, and full-anchor incremental gates. If authorized, the next experiment
starts with a chronological regularized linear benchmark. No nonlinear model or
hyperparameter sweep is authorized here.
