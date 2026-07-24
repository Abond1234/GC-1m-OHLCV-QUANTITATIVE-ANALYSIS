# MLAT Feature Research

This directory is the durable research record for the separate experiment
orchestrated by `notebooks/exploration/mlat_feature_research.ipynb`.

The experiment treats `statistical_feature_research.ipynb` as an immutable
upstream producer. It consumes the trusted GC bar, observation, label, baseline,
feature, and expansion-anchor artifacts produced by that earlier cycle; it does
not copy the old notebook or append features to its registry.

## Governance

- Discovery instrument: GC only.
- Feature timestamp: completed decision bar `t`.
- Theoretical entry: open of bar `t+1`.
- Entry windows: London `[03:00, 06:00)` and New York `[07:00, 12:00)`.
- Forced path end: no later than 15:30 New York time.
- Development and Validation are the controlled research samples.
- The historical Final-test period has already been exposed and is excluded
  from MLAT feature selection and outcome displays.
- MGC may not influence definitions, parameters, thresholds, selection, or
  advancement decisions.
- Generated outputs use versioned `mlat_feature_research/v1` directories.

## Reading order

1. `mlat_current_state_audit.md`
2. `mlat_book_map.md`
3. `mlat_book_coverage_log.md`
4. `chapter_summaries/`
5. `mlat_concept_registry.md`
6. `mlat_formulas_and_definitions.md`
7. `mlat_project_applicability_matrix.md`
8. `mlat_feature_hypothesis_catalog.md`
9. `mlat_feature_batch_v1.md`
10. `mlat_feature_research_contract_v1.md`
11. `mlat_feature_research_context_report.md`
12. `mlat_final_research_report.md`

Machine-readable manifests and registries are CSV files in this directory.
Generated Parquet tables and figures remain outside Git under the versioned data
and report roots. Partners must obtain the ignored upstream Parquet artifacts
through the project's normal data handoff and verify them against
`mlat_upstream_artifact_manifest.csv` before execution.

## Reproducing the book ingestion

The supplied second-edition PDF is identified by 858 physical pages, a size of
27,293,878 bytes, and SHA-256
`4439a7be25210efb1173f30bcb75cb67743acff78de880578dc8a6de351aa39a`.
The absolute Downloads path used during the original review is not a project
dependency. Partners may place the same file anywhere, verify it with
`Get-FileHash -Algorithm SHA256 <pdf>`, and use Poppler commands such as:

```powershell
pdfinfo <pdf>
pdftotext -f 42 -l 58 -layout <pdf> -
pdftoppm -f 134 -l 134 -singlefile -png -r 150 <pdf> <temporary-output-prefix>
```

Text was extracted sequentially for all pages; layout-sensitive equations,
tables, diagrams, and workflows were checked from selected rendered pages.
Exact confirmed visual pages are recorded in
`mlat_book_ingestion_manifest.csv`. Full extracted text and rendered page
images are temporary copyrighted-source material under `tmp/`; they are not
research deliverables and must never be staged or committed.
