# MLAT feature research — integration status (2026-07-24)

Durable record of how the book-derived MLAT research line (branch `twatski`)
was brought onto `main`. The research findings themselves are in
`project_docs/mlat_feature_research/mlat_final_research_report.md` (script-generated;
do not hand-edit).

## What the research found

- **0 of 12 candidate features authorized — all `RESEARCH_ONLY`.** The line fails
  closed, but not by clean empirical rejection.
- The blocker is design, not data: the frozen v1 horizon-thinning gate is
  **structurally non-evaluable** — 0 of 384 required 60/180-minute
  feature-family-session-partition cells (0 of them in Development) keep finite
  thinned daily-IC evidence under the ≥10-observations-per-New-York-date rule, so
  the multivariate authorization gate is CLOSED.
- A separately governed audit records the plain GARCH(1,1) implementation as
  `REJECTED_IMPLEMENTATION`, consistent with Branch B's own boundary/explosive fit.
- Stage 1 (book ingestion) is COMPLETE: 858 pages accounted for once, 30 concepts,
  14 formula/definition entries, 29 hypotheses, frozen batch and contract hashes
  recorded.
- A v2 requires a gate specification that leaves evaluable evidence at the longer
  horizons, **declared before any recomputation** — choosing it after seeing v1 output
  would void the run.

## What was integrated, and how

- Merged `main` (PR #25: portable compute, pinned numerical stack, GARCH module,
  notebook portability) into the branch. Both merge conflicts —
  `statistical_feature_research.ipynb` and `section6_feature_engineering_summary.md` —
  were resolved to **main's canonical, reproducible values** (the branch copy carried
  divergent unpinned-environment values, e.g. Section 6 `two_bar_directional_balance`
  1.150116% vs the canonical 1.396989%).
- Removed ~190 MB of committed binary scratch (`work_quant_review/` Word-render
  captures, `tmp/pdfs/` book-page scans) and 41 force-added `.csv` result tables that
  the `*.csv` gitignore rule already excludes; extended `.gitignore` so they cannot
  recur. Merged by **squash** so `main` never carries the binaries in its history.
- Made the MLAT test suite run under the project runner: four modules were authored
  for pytest with a non-`src` import root and did not run under `python -m unittest`
  (CI installs no pytest). Converted them to `unittest`, and ruff-cleaned and formatted
  the whole contribution. Full suite: **316 tests, green**; ruff check and format clean.

## Deferred (flagged follow-up)

- **Sharpe-governance and external-figure cells** for the statistical notebook
  (`performance_diagnostics.py`, `research_figures.py`, and
  `scripts/update_statistical_sharpe_visualizations.py`) are merged at module level but
  not wired into the notebook. Their injection anchors on the pre-rewrite GARCH section
  (text markers and a `vol_threshold` variable) that PR #25 replaced; wiring them in
  requires reconciling those anchors with the current GARCH implementation and a
  full pinned-environment notebook re-execution. `research_figures.py` has no unit test
  and is exercised only by that notebook section, so it lands unexercised until then.
