# GARCH Volatility Feature Research Contract

**Status:** FROZEN BEFORE IMPLEMENTATION (declared 2026-07-21, per the PRD workflow rule that the research question, definitions, expected artifacts, and acceptance checks precede implementation)

## Research question

Can a GARCH(1,1) conditional volatility estimate — fitted on the development partition and applied forward without lookahead — provide a useful volatility-regime signal for risk management and feature engineering in the GC/MGC statistical research branch? Specifically: does the regime flag (`high_vol_regime`) identify periods where forward return distributions shift in mean, median, or dispersion in a direction useful for position sizing or trade suppression?

## Hypotheses (pre-declared, each tested independently)

- **H1 Regime shift in returns:** GC eligible observations flagged as `feat_garch_high_vol_regime = True` (conditional vol above development p75) show a materially different mean and median forward return (at the primary research horizon) compared to low-vol-regime observations, in both development and validation partitions.
- **H2 Dispersion shift:** Standard deviation and MAE of forward returns are higher in high-vol-regime bars than low-vol-regime bars (validates that GARCH is capturing genuine risk variation rather than noise).
- **H3 Persistence (α+β):** The fitted GARCH(1,1) persistence parameter α+β exceeds 0.90 for GC, consistent with the well-known high persistence of futures intraday volatility. This is a sanity check, not an acceptance criterion.

## Frozen definitions

- **Input data:** `research_bars` dataframe (trusted `data/processed/research_bars_gc_mgc_1m.parquet`), product GC only for hypothesis evaluation; MGC fitted separately for completeness.
- **Model:** GARCH(1,1) with constant mean, Normal distribution (`arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='Normal')`). No model search; GARCH(1,1) is the pre-declared specification.
- **Returns:** 1-minute log returns of `close` price, computed within each product group, scaled ×10,000 (basis points) for numerical optimizer stability during fitting, and unscaled back to log-return units.
- **Fitting partition:** Development partition bars only. Parameters (ω, α, β, μ) are estimated once and frozen.
- **Forward application:** GARCH recursion applied bar-by-bar forward (validation and final-test bars) using only past information — no future data enters. This makes `feat_garch_cond_vol` a leakage-free feature.
- **High-vol threshold:** Development-partition 75th percentile of `feat_garch_cond_vol`. Fixed before validation evaluation.
- **Eligible observations:** `tradable_research_flag == True` rows only, for hypothesis evaluation.
- **Partitions evaluated:** Development and Validation. Final-test partition is read once only, after development and validation verdicts are fixed.

## Acceptance criteria (all pre-declared)

- H1 advances if: the difference in mean forward return between regimes is consistent in sign across development and validation, and the development-partition difference exceeds 0.05R in absolute value (small but economically meaningful for risk conditioning).
- H2 advances if: high-vol-regime standard deviation of forward returns exceeds low-vol-regime standard deviation by ≥15% in both development and validation.
- H3 passes if: α+β > 0.90 for GC (informational only; does not gate H1 or H2).
- Any threshold change, horizon change, or subgroup definition change after seeing results voids the run.

## Expected artifacts

- New cells appended to `notebooks/exploration/statistical_feature_research.ipynb` (GARCH section, cells 187–190).
- Two new columns in `research_bars` at notebook runtime: `feat_garch_cond_vol`, `feat_garch_high_vol_regime`.
- No new source files required; GARCH fitting is self-contained in the notebook (exploration-phase research).
- If H1 or H2 advance: promote to `src/statistical_research/feature_engineering.py` and add corresponding tests in `tests/`.

## Relationship to forward test plan

GARCH conditional volatility is a **risk management input**, not a directional signal. A passing H1/H2 result would justify using `feat_garch_high_vol_regime` to condition position size or suppress entries in any future strategy candidate — it does not by itself constitute a new tradeable strategy requiring Section 8 authorization.
