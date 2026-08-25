# Tsay Feature Research Status

## Stage 1 checkpoint

`STATUS: READY`

`RESEARCH_VERDICT: NOT_AUTHORIZED`

Stage 1 froze the mandate, provenance, footer schemas, overlap resolution, model
input lists, resampling seed, and exact outer-fold memberships without reading a
label value, Validation outcome, MGC research row, or any row dated 2025 onward.

- Milestone branch: `feature-research`, created from synchronized `main` commit
  `362f4061616ff2bfc4a3e40803419c5afd22f967`.
- The supplied Third Edition PDF identity matches SHA-256
  `B5630A4774C8C23F6BB8F624D05B536E49F09E828F4DB3B01D00B5B38A119E33`.
  Its contents were not researched.
- All six prescribed label fields exist by footer schema; their values remain
  unopened.
- The base 85, frozen 15, MLAT 12, and FES 10 were audited before outcomes. All
  T01-T09 are `RELATED_BUT_MATERIALLY_DISTINCT`; no locked verdict removes a
  candidate. The frozen primary family is 18 session-specific tests.
- Through-2024 eligible identity metadata contains 306,230 Development rows
  (114,834 London; 191,396 New York) across 638 dates, and 118,076 Validation
  rows (44,276 London; 73,800 New York) across 246 dates.
- Each session has 24 complete outer folds, 504 attainable Development OOF
  dates, and `minimum_dev_oof_dates=454`; the 180-date structural gate is
  attainable.
- Fold-membership SHA-256:
  `02e0e6dba59d9693288fc07ff29763267d497a9d42cfb5c5764f98671bc3e46b`.
- Frozen-contract SHA-256:
  `cc95d3254c9f271a60fd0fa0042a24c69328044748348802aea1ab2817dc6e0f`.

Generated Stage 1 JSON artifacts and access/hash manifests are local under the
ignored `data/processed/statistical_research/tsay_feature_research/v1/` root.

Stage 2 is authorized only for outcome-free causal feature construction, only
after the user says `continue`. Labels and every later stage remain closed.

## Stage 2 checkpoint addendum

`STATUS: READY`

`RESEARCH_VERDICT: NOT_AUTHORIZED`

Stage 2 constructed the frozen T01-T09 family, T03 companion state, and causal
clock-profile intermediates over 1,271,756 chronological GC minutes through
2024, then mapped them exactly to 424,306 eligible Development/Validation rows.
No label value, Validation outcome, MGC research row, or 2025+ row was read.

- The original Stage 1 contract and checkpoint remain unchanged. The separate
  `tsay_stage2_numerical_compatibility_amendment_v1.md` records the authorized
  legacy compatibility rule and commit `1871dc3` provenance.
- `atr_20` and `current_range_over_atr` match the canonical artifact exactly at
  `float32` on every eligible row, including exact null masks.
- `realized_volatility_15` has an exact null-mask match. Twenty-three rows differ
  from the preserved legacy artifact by exactly one `float32` ULP; maximum ULP
  distance is 1 and maximum absolute difference is
  `9.5367431640625e-07`.
- T01/T02 share one deterministic batched float64 SVD fit per eligible row.
  Candidate windows reset at canonical continuity boundaries and New York dates;
  clock moments are strictly prior-date on Development and all-Development frozen
  on Validation.
- All nine persisted feature columns are `float32`, contain no infinities, pass
  their frozen ranges, and survive exact save/reload comparison. The T03 companion
  persists as its prescribed three-state `int8` field.
- The guarded legacy-primitive slice SHA-256 is
  `37e1e5ab326c23fa4dbc90e8569a313b249fd240bd9eb0841afad721c18c378c`.
- The deterministic feature-matrix sample SHA-256 is
  `3c2574007e407d94463b25833c4f8a302ad6bd28784e35b1f8f7c0c366b84bcc`.

Stage 3 may begin only after a new explicit continuation instruction and is
limited to Development-only feature evidence. It has not begun here.

## Stage 3 checkpoint addendum

`STATUS: READY`

`RESEARCH_VERDICT: NOT_AUTHORIZED`

Stage 3 verified the Stage 2 checkpoint and hashes, then read only 306,230 GC
Development rows dated 2021-05-24 through 2023-12-29. The only label fields
materialized were the frozen identity/partition fields, decision ATR, the
60-minute directional/range ATR and tick targets, the 30-minute directional/range
ATR diagnostics, and the two availability flags. Validation label values,
Validation performance, MGC rows, models, policy P&L, and post-2023 rows remained
unopened.

- The complete 18-test family was retained. Six directional tests are
  `DEV_REJECTED`; all twelve opportunity tests are `NOT_EVALUABLE`; no test is
  `DEV_SHORTLISTED`.
- T01 and T06 have strongly negative oriented primary evidence in both sessions.
  T07 is weakly positive but fails BH, interval, monotonicity/tick-spread, and
  partial-interval gates.
- T05 and T08 have positive BH-significant primary opportunity evidence in one or
  both sessions, but fail other frozen gates. In particular, all opportunity
  partial-information panels are rank deficient under the exact frozen O1 plus
  explicit-control design, so their partial retention/interval is unavailable.
- T02, T03, T04, and T09 do not pass their complete primary and partial gate sets.
- The 30-minute results remain diagnostic only and have no selection authority.
- Both 60-minute ATR/tick label pairs pass rowwise consistency QA using each row's
  decision ATR. No aggregate average-ATR conversion was used.
- All evidence, daily IC, quintile-edge/summary, year-stability, partial-IC,
  redundancy, candidate-correlation, access, run, and hash artifacts survive
  exact save/reload verification under the pinned deterministic CPU environment.

Stage 4 was authorized only for the already fixed Development nested model
ladder and began after the user's explicit continuation instruction.

## Stage 4 checkpoint addendum

`STATUS: READY`

`RESEARCH_VERDICT: NO_PREDICTIVE_MODEL_AUTHORIZED`

Stage 4 verified the Stage 3 checkpoint, hashes, and access manifest, then fit the
frozen session-specific D0-D5, O0-O5, and E0-E3 ladders on Development only.
D3 did not mechanically authorize D6 in either session, so D6 is recorded as
`NOT_AUTHORIZED`. Validation labels/outcomes, MGC research rows, real bars,
strategy P&L, and post-2023 rows remained unopened.

- All directional candidates D2-D5, continuous-opportunity candidates O2-O5,
  and expansion candidates E2-E3 are `DEV_REJECTED` under their unchanged gates.
- The D2-D5, O2-O5, and E2-E3 families use synchronized stationary-bootstrap
  max-statistic correction with each candidate's fixed original studentizer.
- The exact two-session architecture one-standard-error procedure selects no
  directional, opportunity, or expansion architecture.
- D4 residual, squared-residual, and multivariate portmanteau diagnostics and O5
  innovation Ljung-Box/ARCH-LM diagnostics are Holm-corrected and retained as
  model-adequacy evidence. Quantile crossing, pinball, coverage, width, and
  year/session calibration evidence is retained separately.
- With no authorized directional architecture, Development real bars and policy
  economics were not opened. The immutable policy state is
  `FROZEN_NO_POLICY`, and Validation economics display as unavailable rather
  than zero.
- No all-Development deployment model is fit for a role whose ladder selects no
  architecture; the immutable deployment state hashes the null selections and
  the complete Development OOF prediction artifact.

Stage 5 is not authorized because no predictive architecture or executable
policy passed Development. Validation outcomes remain unopened.
