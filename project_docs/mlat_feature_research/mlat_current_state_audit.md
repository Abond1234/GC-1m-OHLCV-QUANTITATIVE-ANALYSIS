# MLAT Current-State Audit

Audit date: 2026-07-23. Classifications reflect the current working tree and
the physical saved artifacts, not assumptions from notebook prose alone.

| Statement | Classification | Evidence and qualification |
|---|---|---|
| The trusted data are Databento CME futures one-minute OHLCV bars. | VERIFIED WITH QUALIFICATION | `data/processed/research_bars_gc_mgc_1m.parquet` contains 3,487,656 one-minute active-contract rows with OHLCV, Databento identifiers, CME/Globex lineage, and project-added controls. It is a processed active-contract research table, not raw trade, quote, MBO, or order-book data. |
| GC is the discovery instrument. | VERIFIED | The eligible-observation, label, feature, baseline, and frozen-feature artifacts are GC-only. The statistical context report freezes GC as Phase 1 discovery. |
| MGC is reserved for later transfer validation and execution mapping. | VERIFIED | The statistical context report and saved artifact names enforce this role. No MGC artifact is an authorized MLAT discovery input. |
| Feature information ends at completed decision bar `t`. | VERIFIED | `src/statistical_research/feature_engineering.py` builds predictors on full chronological history and maps the decision-bar values. Existing mutation tests prove later bars cannot change decision features. MLAT retains this contract. |
| The theoretical entry is the open of `t+1`. | VERIFIED | `eligible_observations_gc.parquet` stores distinct decision and entry IDs/timestamps; the label module measures paths from the next-bar entry. |
| Candidate entries are restricted to London and New York execution windows. | VERIFIED | Physical observations contain 219,938 London and 366,592 New York rows. Locked half-open windows are London `[03:00,06:00)` and New York `[07:00,12:00)` in New York time. |
| No new entries occur at or after 12:00 New York time. | VERIFIED | Enforced by the eligible-observation artifact and the statistical context contract. |
| Positions may continue after noon but close before 15:30 New York time. | VERIFIED WITH QUALIFICATION | The label artifact stores `forced_exit_timestamp_ny` and fixed-horizon availability. Paths may continue after noon, but they end **at or before** the frozen 15:30 boundary; no strategy is created in the MLAT feature task. |
| No forward path crosses an invalid contract, segment, rollover, tradability, missing-minute, or New York-date boundary. | VERIFIED | `src/statistical_research/labels.py` and its tests cover exact path availability and each boundary. Physical label-availability counts decline with horizon rather than silently shortening paths. |
| The previous notebook engineered and evaluated a substantial feature set. | VERIFIED | `feature_registry_gc.parquet` has 85 registered predictors; `feature_matrix_gc.parquet` has 586,530 rows and 85 predictor columns. |
| The previous research found substantially more expansion than stable direction information. | VERIFIED | The saved univariate cycle records 0 directional and 55 expansion advancers; redundancy reduced the expansion set to 15 features with `atr_20` as anchor. |
| The earlier Final-test period is pristine for this experiment. | INCORRECT | It was inspected in the earlier statistical, hybrid, and opportunity-conditioning cycles. MLAT excludes it from feature/outcome screens and recommends a new future holdout or live paper period. |
| The historical Final-test metadata may be loaded to enforce exclusion. | VERIFIED WITH QUALIFICATION | Partition metadata and feature engineering integrity may be checked, but MLAT must not compute or display Final-test feature/outcome relationships. |
| A later GARCH implementation exists in the previous notebook. | VERIFIED WITH QUALIFICATION | Cells 193-195 contain exploratory GC/MGC GARCH work, but there is no reusable module, test suite, saved parameter/forecast artifact, or completion gate. |
| The prior GARCH result is a trusted upstream feature. | INCORRECT | The implementation crosses gaps/contracts/segments, initializes the chronology incorrectly, uses a joint GC/MGC regime threshold, has an unresolved MGC convergence warning, and never evaluates its frozen H1/H2 contract. MLAT audits it independently. |
| Existing saved artifacts can be loaded with unrestricted `pandas.read_parquet`. | OUTDATED | With the pinned pandas/PyArrow combination, some column-scoped pandas reads fail on stored dictionary metadata. The safe pattern is `pyarrow.parquet.read_table(..., columns=...).to_pandas(ignore_metadata=True)`. |
| The old feature save/evaluation helpers can be reused for MLAT outputs. | INCORRECT | Their save paths overwrite historical statistical-research artifacts. MLAT uses separate `mlat_feature_research/v1` roots. |
| Every upstream artifact has a reusable source producer. | INCORRECT | Label construction is reusable, but eligible-observation construction and some invariant validation remain notebook-inline. A missing eligible artifact is therefore a genuine reconstruction blocker requiring a separate governed task. |
| The statistical context report is exact on every physical diagnostic. | VERIFIED WITH QUALIFICATION | It is the primary governance record, but its Development maximum-missingness statement has drifted from the current physical diagnostic, and it does not document the exploratory GARCH/Section 13 additions. Physical artifact fingerprints govern MLAT loading. |

## Physical upstream facts

- Trusted bars: 3,487,656 rows, 82 columns; GC 1,759,671 and MGC
  1,727,985.
- Eligible observations: 586,530 rows, 34 columns.
- Partitions: Development 306,230; Validation 118,076; historical Final test
  162,224.
- Forward-label horizons: 5, 15, 30, 60, 120, and 180 minutes.
- Complete Development+Validation six-horizon evaluation frame: 419,393 rows.
- Existing feature matrix: 586,530 rows, 97 columns (12 metadata and 85
  predictors).
- Frozen expansion set: 15 features; `atr_20` is the anchor.

