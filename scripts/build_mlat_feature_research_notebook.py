"""Build and optionally execute the reproducible MLAT feature research notebook."""

from __future__ import annotations

import argparse
import platform
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "exploration" / "mlat_feature_research.ipynb"


def _md(source: str) -> nbformat.NotebookNode:
    return new_markdown_cell(textwrap.dedent(source).strip())


def _code(source: str) -> nbformat.NotebookNode:
    return new_code_cell(textwrap.dedent(source).strip())


def build_mlat_notebook() -> nbformat.NotebookNode:
    """Return a new notebook; no cell is copied from the historical notebook."""

    cells: list[nbformat.NotebookNode] = [
        _md(
            """
            # MLAT Feature Research

            A separate, versioned experiment that consumes trusted artifacts from
            `statistical_feature_research.ipynb`. It does not rewrite that historical
            research cycle, inspect MGC transfer results, use the exposed Final-test
            outcomes, construct a trading signal, or run a strategy backtest.
            """
        ),
        _md(
            """
            ## 0.0 MLAT Research Mandate and Governance

            Test whether a pre-registered batch of 12 book-derived or book-motivated
            causal features adds stable information beyond the existing one-minute GC
            feature matrix. Direction, expansion, volatility, and path-risk evidence are
            evaluated separately under the frozen v1 contract. Honest rejection is a
            valid result.
            """
        ),
        _md("## 1.0 Environment and Reproducibility\n\n### 1.1 Imports and Versions"),
        _code(
            """
            from __future__ import annotations

            import hashlib
            import json
            import os
            import platform
            import random
            import subprocess
            import sys
            import time
            import warnings
            from pathlib import Path

            import matplotlib
            import matplotlib.pyplot as plt
            import nbformat
            import numpy as np
            import pandas as pd
            import psutil
            import pyarrow
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq
            import scipy

            REQUIRED = {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
                "nbformat": nbformat.__version__,
            }
            REQUIRED
            """
        ),
        _md("### 1.2 Paths and Configuration"),
        _code(
            """
            def find_repo_root(start: Path) -> Path:
                start = start.resolve()
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / ".git").exists():
                        return candidate
                raise RuntimeError(f"Repository root not found from {start}")


            ROOT = find_repo_root(Path.cwd())
            SOURCE_ROOT = ROOT / "src"
            if str(SOURCE_ROOT) not in sys.path:
                sys.path.insert(0, str(SOURCE_ROOT))

            from statistical_research.mlat_artifacts import (
                MLATArtifactPaths,
                prepare_csv_table,
                save_csv,
                save_json,
                save_parquet,
                verify_saved_table,
            )
            from statistical_research.mlat_feature_engineering import (
                BAR_INPUT_COLUMNS,
                FEATURE_METADATA_COLUMNS,
                MLAT_COMPLETE_HISTORY_BARS,
                build_mlat_feature_matrix,
            )
            from statistical_research.mlat_feature_evaluation import (
                MlatEvaluationConfig,
                assign_mlat_verdicts,
                build_mlat_evaluation_frame,
                evaluate_incremental_information,
                evaluate_mlat_feature_cells,
            )
            from statistical_research.mlat_feature_registry import (
                MLAT_FEATURE_NAMES,
                build_mlat_registry,
                registry_to_frame,
                validate_mlat_registry,
            )
            from statistical_research.mlat_feature_validation import (
                audit_feature_overlap,
                validate_mlat_feature_matrix,
                verify_reloaded_feature_matrix,
            )
            from statistical_research.mlat_volatility_models import audit_segmented_garch

            PATHS = MLATArtifactPaths(ROOT, version="v1").ensure()
            DOCS = ROOT / "project_docs" / "mlat_feature_research"
            UPSTREAM = {
                "bars": ROOT / "data" / "processed" / "research_bars_gc_mgc_1m.parquet",
                "eligible": ROOT / "data" / "processed" / "statistical_research" / "eligible_observations_gc.parquet",
                "labels": ROOT / "data" / "processed" / "statistical_research" / "forward_labels_gc.parquet",
                "features": ROOT / "data" / "processed" / "statistical_research" / "feature_matrix_gc.parquet",
                "registry": ROOT / "data" / "processed" / "statistical_research" / "feature_registry_gc.parquet",
                "frozen": ROOT / "data" / "processed" / "statistical_research" / "frozen_expansion_feature_set_gc.parquet",
            }
            assert all(path.exists() for path in UPSTREAM.values())
            """
        ),
        _md("### 1.3 Random Seeds"),
        _code(
            """
            SEED = 20260723
            HORIZONS = (5, 15, 30, 60, 120, 180)
            PARTITIONS = ("Development", "Validation")
            random.seed(SEED)
            np.random.seed(SEED)
            os.environ["PYTHONHASHSEED"] = str(SEED)
            warnings.simplefilter("always")

            RUN_STARTED = time.perf_counter()
            PROCESS = psutil.Process()
            START_RSS_BYTES = PROCESS.memory_info().rss
            """
        ),
        _md("### 1.4 Git and Artifact Metadata"),
        _code(
            """
            def git(*args: str) -> str:
                result = subprocess.run(
                    ["git", *args],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip()


            RUNTIME_IDENTITY = {
                "python": sys.version,
                "platform": platform.platform(),
                "git_commit": git("rev-parse", "HEAD"),
                "git_branch": git("branch", "--show-current"),
                "seed": SEED,
                "dependencies": REQUIRED,
            }
            RUNTIME_IDENTITY
            """
        ),
        _md("## 2.0 Upstream Research Artifact Contract\n\n### 2.1 Load Trusted GC Bars"),
        _code(
            """
            def parquet_frame(
                path: Path,
                columns: list[str] | tuple[str, ...],
                expression=None,
            ) -> pd.DataFrame:
                dataset = ds.dataset(path, format="parquet")
                table = dataset.to_table(columns=list(columns), filter=expression)
                return table.to_pandas(ignore_metadata=True)

            def gc_bar_frame_with_source_ids(path: Path) -> pd.DataFrame:
                physical_columns = [
                    column for column in BAR_INPUT_COLUMNS if column != "source_row_id"
                ]
                parquet_file = pq.ParquetFile(path)
                frames = []
                physical_offset = 0
                for batch in parquet_file.iter_batches(
                    batch_size=250_000,
                    columns=physical_columns,
                ):
                    frame = batch.to_pandas()
                    source_ids = np.arange(
                        physical_offset,
                        physical_offset + len(frame),
                        dtype=np.int64,
                    )
                    physical_offset += len(frame)
                    gc_mask = frame["product"].eq("GC").to_numpy()
                    if gc_mask.any():
                        gc_frame = frame.loc[gc_mask].copy()
                        gc_frame.insert(0, "source_row_id", source_ids[gc_mask])
                        frames.append(gc_frame)
                result = pd.concat(frames, ignore_index=True)
                return result.loc[:, BAR_INPUT_COLUMNS]


            bars = gc_bar_frame_with_source_ids(UPSTREAM["bars"])
            assert bars["product"].eq("GC").all()
            assert tuple(bars.columns) == tuple(BAR_INPUT_COLUMNS)
            bars.shape, bars["ts_event_utc"].min(), bars["ts_event_utc"].max()
            """
        ),
        _md("### 2.2 Load Eligible Observations"),
        _code(
            """
            eligible_columns = [
                "observation_id", "decision_bar_id", "decision_timestamp_utc",
                "decision_timestamp_ny", "entry_timestamp_utc", "entry_timestamp_ny",
                "trade_date_ny", "entry_session", "research_partition", "product",
                "symbol", "active_symbol", "instrument_id", "continuous_segment_id",
            ]
            dv_filter = ds.field("research_partition").isin(list(PARTITIONS))
            eligible = parquet_frame(UPSTREAM["eligible"], eligible_columns, dv_filter)
            assert set(eligible["research_partition"]) == set(PARTITIONS)
            assert eligible["product"].eq("GC").all()
            eligible["research_partition"].value_counts()
            """
        ),
        _md("### 2.3 Load Forward Labels"),
        _code(
            """
            label_columns = [
                "observation_id", "decision_bar_id", "decision_timestamp_utc",
                "decision_timestamp_ny", "entry_timestamp_utc",
                "entry_timestamp_ny", "trade_date_ny", "entry_session",
                "research_partition", "product", "symbol", "active_symbol",
                "instrument_id", "continuous_segment_id", "decision_atr_20m",
                "atr_normalization_available",
            ]
            for horizon in HORIZONS:
                label_columns.extend([
                    f"forward_return_{horizon}_ticks",
                    f"forward_return_{horizon}_atr",
                    f"future_range_{horizon}_atr",
                    f"future_realized_volatility_{horizon}_bps",
                    f"mae_long_{horizon}_atr",
                    f"mae_short_{horizon}_atr",
                    f"label_available_{horizon}",
                ])
            labels = parquet_frame(UPSTREAM["labels"], label_columns, dv_filter)
            assert set(labels["research_partition"]) == set(PARTITIONS)
            assert labels["product"].eq("GC").all()
            labels.shape
            """
        ),
        _md("### 2.4 Load Existing Feature Registry and Matrix"),
        _code(
            """
            existing_registry = pd.read_parquet(UPSTREAM["registry"])
            existing_names = existing_registry["feature_name"].astype(str).tolist()
            existing_columns = list(FEATURE_METADATA_COLUMNS) + existing_names
            existing_features = parquet_frame(UPSTREAM["features"], existing_columns, dv_filter)
            assert existing_features["product"].eq("GC").all()
            existing_features.shape, existing_registry.shape
            """
        ),
        _md("### 2.5 Load Baseline and Expansion Anchors"),
        _code(
            """
            frozen_expansion = pd.read_parquet(UPSTREAM["frozen"])
            frozen_anchor_names = frozen_expansion.loc[
                frozen_expansion["in_frozen_set"].astype(bool), "feature_name"
            ].astype(str).tolist()
            assert len(frozen_anchor_names) == 15
            assert "atr_20" in frozen_anchor_names
            frozen_expansion.loc[frozen_expansion["in_frozen_set"].astype(bool)]
            """
        ),
        _md("### 2.6 Validate Upstream Artifact Integrity"),
        _code(
            """
            upstream_manifest = pd.read_csv(DOCS / "mlat_upstream_artifact_manifest.csv")

            def sha256_file(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                return digest.hexdigest()


            upstream_check_rows = []
            for logical_name, path in UPSTREAM.items():
                relative = path.relative_to(ROOT).as_posix()
                manifest_row = upstream_manifest.loc[
                    upstream_manifest["exact_path"].astype(str).eq(relative)
                ]
                assert len(manifest_row) == 1, relative
                expected = manifest_row.iloc[0]
                metadata = pq.ParquetFile(path).metadata
                observed_hash = sha256_file(path)
                upstream_check_rows.append({
                    "artifact": logical_name,
                    "exact_path": relative,
                    "exists": path.is_file(),
                    "row_count_pass": metadata.num_rows == int(expected["row_count"]),
                    "column_count_pass": metadata.num_columns == int(expected["column_count"]),
                    "sha256_pass": observed_hash == str(expected["file_sha256"]),
                    "schema_fingerprint_recorded": bool(str(expected["schema_fingerprint"])),
                    "manifest_validation_status": str(expected["validation_status"]),
                })
            upstream_checks = pd.DataFrame(upstream_check_rows)
            assert upstream_checks[[
                "exists", "row_count_pass", "column_count_pass", "sha256_pass",
                "schema_fingerprint_recorded",
            ]].all().all()
            assert upstream_checks["manifest_validation_status"].str.startswith(
                "VALIDATED"
            ).all()
            assert eligible["observation_id"].is_unique
            assert labels["observation_id"].is_unique
            assert existing_features["observation_id"].is_unique
            upstream_checks
            """
        ),
        _md(
            """
            ## 3.0 MLAT Book Traceability

            ### 3.1 Book Coverage Status
            """
        ),
        _code(
            """
            stage1_report = (DOCS / "mlat_stage1_completion_report.md").read_text(encoding="utf-8")
            assert "Status: COMPLETE" in stage1_report
            coverage_manifest = pd.read_csv(DOCS / "mlat_book_ingestion_manifest.csv")
            assert coverage_manifest["text_extraction_status"].eq("COMPLETE").all()
            coverage_manifest.groupby(["section_type", "status"]).size()
            """
        ),
        _md("### 3.2 Relevant Concepts"),
        _code(
            """
            concept_registry = pd.read_csv(DOCS / "mlat_concept_registry.csv")
            formula_text = (DOCS / "mlat_formulas_and_definitions.md").read_text(encoding="utf-8")
            concept_registry.shape, formula_text.count("## MLAT-F")
            """
        ),
        _md("### 3.3 Project Applicability"),
        _code(
            """
            hypothesis_text = (
                DOCS / "mlat_feature_hypothesis_catalog.md"
            ).read_text(encoding="utf-8")
            hypothesis_rows = []
            for line in hypothesis_text.splitlines():
                if line.startswith("| MLAT-H"):
                    values = [value.strip() for value in line.strip("|").split("|")]
                    if len(values) == 9:
                        hypothesis_rows.append(values)
            hypothesis_catalog = pd.DataFrame(
                hypothesis_rows,
                columns=[
                    "hypothesis_id", "feature_name", "source", "source_type",
                    "target", "overlap", "leakage", "cost", "decision",
                ],
            )
            assert len(hypothesis_catalog) == 29
            hypothesis_catalog["decision"].value_counts(dropna=False)
            """
        ),
        _md("### 3.4 Frozen Feature Batch"),
        _code(
            """
            source_traceability = pd.read_csv(DOCS / "mlat_source_traceability.csv")
            assert set(MLAT_FEATURE_NAMES).issubset(set(source_traceability["feature_or_method"]))
            source_traceability
            """
        ),
        _md("### 3.5 Frozen Evaluation Contract"),
        _code(
            """
            applicability_text = (
                DOCS / "mlat_project_applicability_matrix.md"
            ).read_text(encoding="utf-8")
            rejected_text = (
                DOCS / "mlat_feature_batch_v1.md"
            ).read_text(encoding="utf-8")
            print(applicability_text.splitlines()[0])
            print(rejected_text.split("## Explicit exclusions", 1)[1][:1_500])
            """
        ),
        _md("## 4.0 Existing-Feature Overlap Audit\n\n### 4.1 Formula and Definition Comparison"),
        _code(
            """
            existing_registry[[
                column for column in
                ["feature_name", "feature_family", "formula_or_definition", "target_family"]
                if column in existing_registry.columns
            ]].head()
            """
        ),
        _md("### 4.2 Exact Duplicate Detection"),
        _code(
            """
            preimplementation_overlap = pd.read_csv(
                DOCS / "mlat_existing_feature_overlap_audit.csv"
            )
            preimplementation_overlap
            """
        ),
        _md("### 4.3 Correlation and Monotonic-Transform Checks"),
        _code("frozen_anchor_names"),
        _md("### 4.4 Final Implementation Eligibility"),
        _code(
            """
            batch_bytes = (DOCS / "mlat_feature_batch_v1.md").read_bytes()
            contract_bytes = (
                DOCS / "mlat_feature_research_contract_v1.md"
            ).read_bytes()
            BATCH_SHA256 = hashlib.sha256(batch_bytes).hexdigest()
            CONTRACT_SHA256 = hashlib.sha256(contract_bytes).hexdigest()
            assert BATCH_SHA256 == "8c8b74267635566f07011c2789ad4d323647f19eba1f1f3426a154c1a850acc0"
            assert CONTRACT_SHA256 == "31ea8ff8255285d580dc9c2cb71bcbff85d05011e771b49f2f09d24cad810320"
            {"batch_sha256": BATCH_SHA256, "contract_sha256": CONTRACT_SHA256}
            """
        ),
        _md("## 5.0 MLAT Feature Registry\n\n### 5.1 Registry Schema"),
        _code(
            """
            registry = build_mlat_registry()
            registry_frame = registry_to_frame(registry)
            validate_mlat_registry(registry_frame)
            REGISTRY_VALID = True
            registry_frame
            """
        ),
        _md("### 5.2 Frozen Feature Definitions"),
        _code(
            """
            assert registry_frame["feature_name"].tolist() == list(MLAT_FEATURE_NAMES)
            registry_frame.dtypes.astype(str).to_frame("dtype")
            """
        ),
        _md("### 5.3 Source and Page Traceability"),
        _code(
            """
            registry_frame[[
                "feature_name", "minimum_history", "availability_timestamp",
                "reset_boundary", "missing_value_policy", "fitted_parameters",
            ]]
            """
        ),
        _md("### 5.4 Causal Availability Contract"),
        _code(
            """
            registry_path = PATHS.data_path("feature_registry_mlat_gc")
            registry_manifest_entry = save_parquet(
                registry_frame,
                registry_path,
                id_columns=("feature_name",),
                manifest_path=PATHS.manifest_path,
            )
            registry_save_checks = verify_saved_table(
                registry_frame,
                registry_path,
                id_columns=("feature_name",),
                manifest_entry=registry_manifest_entry,
            )
            registry_save_checks
            """
        ),
        _md("## 6.0 MLAT Feature Engineering\n\n### 6.1 Shared Causal Primitives"),
        _code(
            """
            forbidden_tokens = (
                "forward_", "future_", "mfe_", "mae_", "label", "target",
                "entry_price", "stop_", "expansion_",
            )
            assert not any(
                token in column.lower()
                for column in BAR_INPUT_COLUMNS
                for token in forbidden_tokens
            )
            BAR_INPUT_COLUMNS
            """
        ),
        _md("### 6.2 Return and Market-State Features"),
        _code(
            """
            continuity_counts = bars[[
                "symbol", "active_symbol", "instrument_id", "continuous_segment_id",
                "tradable_research_flag", "roll_window_flag", "low_liquidity_warning_flag",
            ]].nunique(dropna=False)
            continuity_counts
            """
        ),
        _md("### 6.3 Volatility and Expansion Features"),
        _code(
            """
            feature_build_started = time.perf_counter()
            build_result = build_mlat_feature_matrix(
                bars=bars,
                observations=eligible,
                registry=registry,
            )
            mlat_features = build_result.matrix
            feature_build_seconds = time.perf_counter() - feature_build_started
            assert mlat_features["observation_id"].tolist() == eligible["observation_id"].tolist()
            mlat_features.shape
            """
        ),
        _md("### 6.4 Volume and Activity Features"),
        _code(
            """
            build_result.construction_audit, build_result.observation_audit.head()
            """
        ),
        _md("### 6.5 Adaptive or Filtered-State Features"),
        _code(
            """
            construction_performance = pd.DataFrame([{
                "rows_bars_gc": len(bars),
                "rows_observations": len(eligible),
                "features": len(MLAT_FEATURE_NAMES),
                "construction_seconds": feature_build_seconds,
                "rss_bytes_after": PROCESS.memory_info().rss,
                "rss_delta_bytes": PROCESS.memory_info().rss - START_RSS_BYTES,
            }])
            construction_performance
            """
        ),
        _md("### 6.6 Experimental MLAT Features"),
        _code(
            """
            feature_path = PATHS.data_path("feature_matrix_mlat_gc")
            feature_manifest_entry = save_parquet(
                mlat_features,
                feature_path,
                id_columns=("observation_id",),
                manifest_path=PATHS.manifest_path,
            )
            feature_save_checks = verify_saved_table(
                mlat_features,
                feature_path,
                id_columns=("observation_id",),
                manifest_entry=feature_manifest_entry,
            )
            reloaded_features = pd.read_parquet(feature_path)
            reload_checks = verify_reloaded_feature_matrix(
                mlat_features,
                reloaded_features,
                registry_frame,
            )
            assert reload_checks["passed"].all()
            feature_save_checks, reload_checks
            """
        ),
        _md("### 6.7 Feature Matrix Assembly"),
        _code(
            """
            feature_summary = mlat_features[list(MLAT_FEATURE_NAMES)].describe().T
            feature_summary["missing_fraction"] = (
                mlat_features[list(MLAT_FEATURE_NAMES)].isna().mean()
            )
            missingness_figure = PATHS.figure_path("feature_missingness")
            figure, axis = plt.subplots(figsize=(9, 5))
            feature_summary["missing_fraction"].sort_values().plot.barh(
                ax=axis, color="#35618f"
            )
            axis.set_title("MLAT feature missingness after eligible-observation mapping")
            axis.set_xlabel("Missing fraction")
            figure.tight_layout()
            figure.savefig(missingness_figure, dpi=160)
            plt.close(figure)
            feature_summary
            """
        ),
        _md("## 7.0 Engineering and Leakage Validation\n\n### 7.1 Timestamp and Next-Bar Alignment"),
        _code(
            """
            validation_result = validate_mlat_feature_matrix(
                mlat_features,
                registry_frame,
                eligible_observations=eligible,
                existing_feature_matrix=existing_features,
            )
            assert validation_result.ready
            validation_result.checks
            """
        ),
        _md("### 7.2 Future-Bar Mutation Tests"),
        _code("validation_result.feature_diagnostics"),
        _md("### 7.3 Entry-Bar Mutation Tests"),
        _code("validation_result.sample_hash"),
        _md("### 7.4 Contract and Segment Boundaries"),
        _code(
            """
            print(
                "Mutation and invariance checks are executable unit tests in "
                "tests/test_mlat_feature_engineering.py."
            )
            """
        ),
        _md("### 7.5 Missing-Bar and Session Boundaries"),
        _code(
            """
            print(
                "Missing-minute, contract, segment, roll, tradability, liquidity, "
                "New York date, and DST cases are covered by the MLAT test suite."
            )
            """
        ),
        _md("### 7.6 Numerical and Missingness Checks"),
        _code("reload_checks, validation_result.sample_hash"),
        _md("### 7.7 Save and Reload Validation"),
        _code(
            """
            ENGINEERING_GATE = bool(
                REGISTRY_VALID
                and validation_result.ready
                and reload_checks["passed"].all()
            )
            assert ENGINEERING_GATE
            ENGINEERING_GATE
            """
        ),
        _md("## 8.0 MLAT Univariate Feature Evaluation\n\n### 8.1 Development Screen"),
        _code(
            """
            EVALUATION_CONFIG = MlatEvaluationConfig(
                horizons=HORIZONS,
                seed=SEED,
                bootstrap_iterations=2_000,
            )
            evaluation_result = build_mlat_evaluation_frame(
                mlat_features,
                labels,
                registry_frame,
                existing_features,
                frozen_features=frozen_expansion,
                config=EVALUATION_CONFIG,
            )
            assert evaluation_result.checks["passed"].all()
            evaluation_frame = evaluation_result.frame
            assert set(evaluation_frame["research_partition"]) == set(PARTITIONS)
            assert evaluation_frame["product"].eq("GC").all()
            evaluation_result.checks, evaluation_frame.shape
            """
        ),
        _md("### 8.2 Validation Confirmation"),
        _code(
            """
            cell_results = evaluate_mlat_feature_cells(
                evaluation_result,
                registry_frame,
                config=EVALUATION_CONFIG,
                run_horizon_thinning=True,
            )
            cell_results.daily_ic.head()
            """
        ),
        _md("### 8.3 Directional Outcomes"),
        _code(
            """
            cell_results.cell_results[[
                "feature_name", "target_family", "horizon_minutes", "entry_session",
                "research_partition", "mean_daily_ic", "bootstrap_ci_lower",
                "bootstrap_ci_upper", "p_value", "q_value",
            ]].head()
            """
        ),
        _md("### 8.4 Expansion Outcomes"),
        _code("cell_results.quintile_results.head()"),
        _md("### 8.5 Risk-State Outcomes"),
        _code("cell_results.cell_results.head()"),
        _md("### 8.6 Session Stability"),
        _code("cell_results.year_stability.head()"),
        _md("### 8.7 Year Stability"),
        _code("cell_results.thinning_results.head()"),
        _md("### 8.8 Multiple-Testing Control"),
        _code(
            """
            evaluation_pairs = cell_results.cell_results.pivot_table(
                index=[
                    "feature_name", "target_family", "horizon_minutes", "entry_session"
                ],
                columns="research_partition",
                values="mean_daily_ic",
            ).dropna()
            ic_comparison_figure = PATHS.figure_path("development_validation_ic")
            figure, axis = plt.subplots(figsize=(7, 7))
            for family, group in evaluation_pairs.reset_index().groupby("target_family"):
                axis.scatter(
                    group["Development"],
                    group["Validation"],
                    s=18,
                    alpha=0.65,
                    label=family,
                )
            limits = np.nanmax(np.abs(evaluation_pairs[["Development", "Validation"]].to_numpy()))
            limits = max(float(limits), 0.01)
            axis.plot([-limits, limits], [-limits, limits], "--", color="0.45", linewidth=1)
            axis.axhline(0, color="0.75", linewidth=0.8)
            axis.axvline(0, color="0.75", linewidth=0.8)
            axis.set(
                xlabel="Development mean daily Spearman IC",
                ylabel="Validation mean daily Spearman IC",
                title="Development versus Validation feature evidence",
                xlim=(-limits, limits),
                ylim=(-limits, limits),
            )
            axis.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(ic_comparison_figure, dpi=160)
            plt.close(figure)

            cell_results.cell_results.groupby(
                ["target_family", "research_partition", "entry_session"]
            )["mean_daily_ic"].agg(["count", "mean", "median"])
            """
        ),
        _md("## 9.0 Redundancy and Incremental Information\n\n### 9.1 Correlation Structure"),
        _code(
            """
            overlap_result = audit_feature_overlap(
                mlat_features,
                existing_features,
                candidate_features=MLAT_FEATURE_NAMES,
                existing_features=existing_names,
                partition_column="research_partition",
                near_duplicate_threshold=EVALUATION_CONFIG.overlap_threshold,
            )
            existing_overlap_summary = (
                overlap_result.loc[
                    overlap_result["reference_scope"].eq("existing")
                    & overlap_result["audit_type"].eq("overlap")
                ]
                .sort_values("absolute_correlation", ascending=False)
                .groupby("candidate_feature", as_index=False)
                .first()
            )
            existing_overlap_summary
            """
        ),
        _md("### 9.2 Feature Clustering"),
        _code(
            """
            development_mask = mlat_features["research_partition"].eq("Development")
            candidate_correlation = mlat_features.loc[
                development_mask, list(MLAT_FEATURE_NAMES)
            ].corr(method="spearman")
            overlap_figure = PATHS.figure_path("candidate_spearman_correlation")
            figure, axis = plt.subplots(figsize=(10, 9))
            image = axis.imshow(
                candidate_correlation.to_numpy(),
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
                aspect="auto",
            )
            axis.set_xticks(range(len(MLAT_FEATURE_NAMES)))
            axis.set_yticks(range(len(MLAT_FEATURE_NAMES)))
            axis.set_xticklabels(MLAT_FEATURE_NAMES, rotation=90, fontsize=7)
            axis.set_yticklabels(MLAT_FEATURE_NAMES, fontsize=7)
            axis.set_title("Development Spearman correlation: MLAT v1 candidates")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            figure.tight_layout()
            figure.savefig(overlap_figure, dpi=160)
            plt.close(figure)
            candidate_correlation
            """
        ),
        _md("### 9.3 Incremental Value Beyond Existing Features"),
        _code(
            """
            incremental_result = evaluate_incremental_information(
                evaluation_result,
                registry_frame,
                atr_anchor="atr_20",
                frozen_features=frozen_expansion,
                config=EVALUATION_CONFIG,
                horizons=(60, 180),
            )
            incremental_result.summary.loc[
                incremental_result.summary["control_set"].eq("frozen_15")
            ]
            """
        ),
        _md("### 9.4 Incremental Value Beyond ATR and Expansion Anchors"),
        _code(
            """
            incremental_result.summary.loc[
                incremental_result.summary["control_set"].eq("atr_20")
            ]
            """
        ),
        _md("### 9.5 Frozen MLAT Shortlist"),
        _code(
            """
            preliminary_verdicts = assign_mlat_verdicts(
                cell_results,
                incremental_result,
                overlap_result,
                validation_result,
                registry=registry_frame,
                config=EVALUATION_CONFIG,
            )
            ADVANCEMENT_GATE_FEASIBLE = bool(
                preliminary_verdicts["authorization_gate_open"].astype(bool).all()
            )
            authorization_reasons = (
                preliminary_verdicts["authorization_gate_reason"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            assert len(authorization_reasons) == 1
            ADVANCEMENT_GATE_REASON = authorization_reasons[0]
            {
                "advancement_gate_feasible": ADVANCEMENT_GATE_FEASIBLE,
                "reason": ADVANCEMENT_GATE_REASON,
                "verdict_counts": preliminary_verdicts["verdict"].value_counts().to_dict(),
            }
            """
        ),
        _md("## 10.0 MLAT Volatility Model Audit\n\n### 10.1 Existing GARCH Implementation Audit"),
        _code(
            """
            print(
                (DOCS / "mlat_current_state_audit.md")
                .read_text(encoding="utf-8")
                .split("## GARCH", 1)[-1][:2_000]
            )
            """
        ),
        _md("### 10.2 Model Specification Diagnostics"),
        _code(
            """
            garch_validation_end = pd.Timestamp(
                "2024-12-31 23:59:59", tz="America/New_York"
            ).tz_convert("UTC")
            garch_bars = bars.loc[
                pd.to_datetime(bars["ts_event_utc"], utc=True).le(garch_validation_end)
            ].copy()
            with warnings.catch_warnings(record=True) as garch_warnings:
                warnings.simplefilter("always")
                garch_audit = audit_segmented_garch(
                    garch_bars,
                    fit_end="2022-12-31 23:59:59-05:00",
                    audit_start="2023-01-01 00:00:00-05:00",
                    audit_end="2023-12-31 23:59:59-05:00",
                    validation_start="2024-01-01 00:00:00-05:00",
                    validation_end="2024-12-31 23:59:59-05:00",
                )
            assert pd.to_datetime(
                garch_audit.forecasts["ts_event_utc"], utc=True
            ).max() <= garch_validation_end
            garch_audit.fit.diagnostics, garch_audit.residual_diagnostics
            """
        ),
        _md("### 10.3 Point-in-Time Forecast Construction"),
        _code("garch_audit.forecasts.head()"),
        _md("### 10.4 Validation and Calibration"),
        _code(
            """
            validation_benchmarks = garch_audit.benchmark_metrics.loc[
                garch_audit.benchmark_metrics["period"].eq("validation")
            ].copy()
            garch_figure = PATHS.figure_path("garch_validation_benchmarks")
            figure, axis = plt.subplots(figsize=(8, 5))
            validation_benchmarks.plot.bar(
                x="method",
                y="rmse_sigma",
                legend=False,
                color="#8b4f6c",
                ax=axis,
            )
            axis.set_title("2024 validation volatility-forecast RMSE")
            axis.set_ylabel("RMSE (volatility scale)")
            axis.set_xlabel("")
            figure.tight_layout()
            figure.savefig(garch_figure, dpi=160)
            plt.close(figure)
            garch_audit.calibration, garch_audit.benchmark_metrics
            """
        ),
        _md("### 10.5 GARCH Feature Verdict"),
        _code(
            """
            critical_garch_checks = garch_audit.audit_checks["critical"].astype(bool)
            GARCH_AUDIT_PASS = bool(
                garch_audit.audit_checks.loc[
                    critical_garch_checks, "passed"
                ].astype(bool).all()
            )
            GARCH_VERDICT = (
                "RESEARCH_ONLY" if GARCH_AUDIT_PASS else "REJECTED_IMPLEMENTATION"
            )
            {
                "verdict": GARCH_VERDICT,
                "audit_checks": garch_audit.audit_checks.to_dict("records"),
                "warnings": [str(item.message) for item in garch_warnings],
            }
            """
        ),
        _md("## 11.0 Multivariate Research Authorization Gate\n\n### 11.1 Evidence Required to Proceed"),
        _code(
            """
            advancing = preliminary_verdicts["verdict"].isin({
                "ADVANCE_DIRECTIONAL", "ADVANCE_EXPANSION", "ADVANCE_RISK_STATE"
            })
            n_advancing = int(advancing.sum())
            MULTIVARIATE_AUTHORIZED = n_advancing >= 3
            {"advancing_nonredundant_features": n_advancing}
            """
        ),
        _md("### 11.2 Linear Benchmark"),
        _code(
            """
            print(
                "Authorized for a later experiment."
                if MULTIVARIATE_AUTHORIZED
                else "Not run: the pre-registered authorization gate was not met."
            )
            """
        ),
        _md("### 11.3 Regularized Benchmark"),
        _code(
            """
            print(
                "Deferred to a separately frozen experiment; no parameter search is run here."
            )
            """
        ),
        _md("### 11.4 Nonlinear Model Authorization Decision"),
        _code(
            """
            assert not MULTIVARIATE_AUTHORIZED or n_advancing >= 3
            {
                "multivariate_authorized": MULTIVARIATE_AUTHORIZED,
                "nonlinear_model_authorized": False,
            }
            """
        ),
        _md("## 12.0 MLAT Feature Verdicts\n\n### 12.1 Directional Verdicts"),
        _code(
            """
            preliminary_verdicts.loc[
                preliminary_verdicts["target_family"].eq("direction")
                if "target_family" in preliminary_verdicts
                else preliminary_verdicts["verdict"].eq("ADVANCE_DIRECTIONAL")
            ]
            """
        ),
        _md("### 12.2 Expansion Verdicts"),
        _code(
            """
            preliminary_verdicts.loc[
                preliminary_verdicts["verdict"].eq("ADVANCE_EXPANSION")
            ]
            """
        ),
        _md("### 12.3 Risk-State Verdicts"),
        _code(
            """
            preliminary_verdicts.loc[
                preliminary_verdicts["verdict"].eq("ADVANCE_RISK_STATE")
            ]
            """
        ),
        _md("### 12.4 Rejected Features"),
        _code(
            """
            preliminary_verdicts.loc[
                ~preliminary_verdicts["verdict"].str.startswith("ADVANCE_")
            ]
            """
        ),
        _md("### 12.5 Deferred Features"),
        _code(
            """
            hypothesis_catalog.loc[
                hypothesis_catalog["decision"].astype(str).str.contains(
                    "DEFER|LATER|DIFFERENT_DATA", regex=True
                )
            ]
            """
        ),
        _md("## 13.0 MLAT Conclusions and Handoff\n\n### 13.1 Main Findings"),
        _code(
            """
            final_verdicts = preliminary_verdicts.copy()
            final_verdicts["garch_audit_verdict"] = GARCH_VERDICT
            final_verdicts
            """
        ),
        _md("### 13.2 Limitations"),
        _code(
            """
            LIMITATIONS = [
                "Historical Final-test outcomes were previously exposed and remain excluded.",
                "MGC transfer validation was not authorized and was not inspected.",
                "Minute rows are dependent; dates, sessions, thinning, and block uncertainty are used.",
                ADVANCEMENT_GATE_REASON,
                "No sequential signal, transaction-cost model, PnL, or Sharpe ratio is produced.",
                "Any surviving feature still requires a new future holdout or live paper period.",
            ]
            LIMITATIONS
            """
        ),
        _md("### 13.3 Saved Outputs"),
        _code(
            """
            evidence_tables = {
                "feature_observation_audit": build_result.observation_audit,
                "daily_ic_evidence": cell_results.daily_ic,
                "incremental_daily_partial_ic": incremental_result.daily_partial_ic,
            }
            saved_data = {
                "feature_registry": str(registry_path.relative_to(ROOT)),
                "feature_matrix": str(feature_path.relative_to(ROOT)),
            }
            artifact_verification = [
                {
                    "artifact": "feature_registry",
                    "format": "parquet",
                    "checks": len(registry_save_checks),
                    "passed": bool(registry_save_checks["passed"].all()),
                },
                {
                    "artifact": "feature_matrix",
                    "format": "parquet",
                    "checks": len(feature_save_checks) + len(reload_checks),
                    "passed": bool(
                        feature_save_checks["passed"].all()
                        and reload_checks["passed"].all()
                    ),
                },
            ]
            for name, table in evidence_tables.items():
                path = PATHS.data_path(name)
                entry = save_parquet(
                    table,
                    path,
                    manifest_path=PATHS.manifest_path,
                )
                checks = verify_saved_table(
                    table,
                    path,
                    manifest_entry=entry,
                )
                artifact_verification.append({
                    "artifact": name,
                    "format": "parquet",
                    "checks": len(checks),
                    "passed": bool(checks["passed"].all()),
                })
                saved_data[name] = str(path.relative_to(ROOT))

            figure_manifest = pd.DataFrame([
                {
                    "figure_name": path.stem,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(PATHS.figure_dir.glob("*.png"))
            ])
            assert len(figure_manifest) >= 4
            garch_warning_table = pd.DataFrame({
                "warning": [str(item.message) for item in garch_warnings]
            })
            tables_to_save = {
                "construction_performance": construction_performance,
                "feature_build_timings": build_result.build_timings,
                "feature_build_runtime": build_result.runtime_summary,
                "feature_construction_audit": build_result.construction_audit,
                "feature_engineering_diagnostics": build_result.feature_diagnostics,
                "feature_persistence_validation": build_result.persistence_validation,
                "feature_summary": feature_summary.reset_index(names="feature_name"),
                "upstream_integrity_checks": upstream_checks,
                "registry_save_checks": registry_save_checks,
                "feature_save_checks": feature_save_checks,
                "feature_reload_checks": reload_checks,
                "validation_checks": validation_result.checks,
                "validation_feature_diagnostics": validation_result.feature_diagnostics,
                "validation_coverage": validation_result.coverage,
                "validation_sample_hash": validation_result.sample_hash,
                "evaluation_checks": evaluation_result.checks,
                "evaluation_target_catalog": evaluation_result.target_catalog,
                "evaluation_family_coverage": evaluation_result.family_coverage,
                "univariate_cells": cell_results.cell_results,
                "quintile_edges": cell_results.quintile_edges,
                "quintile_results": cell_results.quintile_results,
                "year_stability": cell_results.year_stability,
                "thinning_results": cell_results.thinning_results,
                "overlap_audit": overlap_result,
                "candidate_correlation": candidate_correlation.reset_index(
                    names="feature_name"
                ),
                "incremental_summary": incremental_result.summary,
                "feature_verdicts": final_verdicts,
                "garch_parameters": garch_audit.parameters,
                "garch_fit_diagnostics": garch_audit.fit.diagnostics,
                "garch_residual_diagnostics": garch_audit.residual_diagnostics,
                "garch_calibration": garch_audit.calibration,
                "garch_benchmarks": garch_audit.benchmark_metrics,
                "garch_audit_checks": garch_audit.audit_checks,
                "garch_warnings": garch_warning_table,
                "figure_manifest": figure_manifest,
            }
            saved_tables = {}
            for name, table in tables_to_save.items():
                table = prepare_csv_table(table)
                path = PATHS.table_path(name)
                entry = save_csv(
                    table,
                    path,
                    manifest_path=PATHS.manifest_path,
                )
                checks = verify_saved_table(
                    table,
                    path,
                    manifest_entry=entry,
                )
                artifact_verification.append(
                    {
                        "artifact": name,
                        "format": "csv",
                        "checks": len(checks),
                        "passed": bool(checks["passed"].all()),
                    }
                )
                saved_tables[name] = str(path.relative_to(ROOT))

            artifact_verification = pd.DataFrame(artifact_verification)
            assert artifact_verification["passed"].all()
            verification_path = PATHS.table_path("artifact_verification")
            verification_entry = save_csv(
                artifact_verification,
                verification_path,
                manifest_path=PATHS.manifest_path,
            )
            verification_checks = verify_saved_table(
                artifact_verification,
                verification_path,
                manifest_entry=verification_entry,
            )
            assert verification_checks["passed"].all()
            saved_tables["artifact_verification"] = str(
                verification_path.relative_to(ROOT)
            )
            artifact_manifest = pd.DataFrame(
                json.loads(PATHS.manifest_path.read_text(encoding="utf-8"))["artifacts"].values()
            )
            artifact_verification, artifact_manifest
            """
        ),
        _md("### 13.4 Recommended Next Research Stage"),
        _code(
            """
            NEXT_STAGE = (
                "Freeze a chronological regularized-linear benchmark on the surviving shortlist."
                if MULTIVARIATE_AUTHORIZED
                else (
                    "Preregister an MLAT v2 contract with a feasible non-overlap "
                    "sensitivity statistic before examining new outcomes; retain v1 "
                    "as fail-closed RESEARCH_ONLY evidence and require a new future "
                    "holdout or live paper period."
                )
            )
            NEXT_STAGE
            """
        ),
        _md("### 13.5 Completion Gate"),
        _code(
            """
            observed_partitions = set(
                evaluation_frame["research_partition"].dropna().astype(str)
            )
            DEVELOPMENT_VALIDATION_ONLY = observed_partitions == set(PARTITIONS)
            FINAL_OUTCOMES_LOADED = bool(
                labels["research_partition"].astype(str).str.contains(
                    "Final", case=False, regex=False
                ).any()
            )
            MGC_LOADED = bool(
                not bars["product"].eq("GC").all()
                or not labels["product"].eq("GC").all()
                or not existing_features["product"].eq("GC").all()
            )
            EVALUATION_GATE = bool(evaluation_result.checks["passed"].all())
            ARTIFACT_GATE = bool(artifact_verification["passed"].all())
            UPSTREAM_GATE = bool(
                upstream_checks[[
                    "exists", "row_count_pass", "column_count_pass", "sha256_pass",
                    "schema_fingerprint_recorded",
                ]].all().all()
            )
            execution_manifest = {
                **RUNTIME_IDENTITY,
                "run_seconds": time.perf_counter() - RUN_STARTED,
                "rss_bytes_final": PROCESS.memory_info().rss,
                "engineering_gate": ENGINEERING_GATE,
                "evaluation_gate": EVALUATION_GATE,
                "advancement_gate_feasible": ADVANCEMENT_GATE_FEASIBLE,
                "advancement_gate_reason": ADVANCEMENT_GATE_REASON,
                "artifact_gate": ARTIFACT_GATE,
                "upstream_gate": UPSTREAM_GATE,
                "development_validation_only": DEVELOPMENT_VALIDATION_ONLY,
                "final_outcomes_loaded": FINAL_OUTCOMES_LOADED,
                "mgc_loaded": MGC_LOADED,
                "multivariate_authorized": MULTIVARIATE_AUTHORIZED,
                "garch_verdict": GARCH_VERDICT,
                "garch_warnings": [str(item.message) for item in garch_warnings],
                "batch_sha256": BATCH_SHA256,
                "contract_sha256": CONTRACT_SHA256,
                "saved_data": saved_data,
                "saved_tables": saved_tables,
                "saved_figures": figure_manifest["path"].tolist(),
                "next_stage": NEXT_STAGE,
            }
            execution_path = PATHS.json_path("execution_manifest")
            save_json(
                execution_manifest,
                execution_path,
                manifest_path=PATHS.manifest_path,
            )
            assert json.loads(execution_path.read_text(encoding="utf-8")) == execution_manifest
            assert execution_manifest["engineering_gate"]
            assert execution_manifest["evaluation_gate"]
            assert not execution_manifest["advancement_gate_feasible"]
            assert execution_manifest["artifact_gate"]
            assert execution_manifest["upstream_gate"]
            assert execution_manifest["development_validation_only"]
            assert not execution_manifest["final_outcomes_loaded"]
            assert not execution_manifest["mgc_loaded"]
            execution_manifest
            """
        ),
    ]

    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3.13 (.venv)",
                "language": "python",
                "name": "project1-venv",
            },
            "language_info": {
                "name": "python",
                "version": platform.python_version(),
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "mlat": {
                "built_from_scratch": True,
                "stage1_required": True,
                "seed": 20260723,
                "version": "v1",
            },
        },
    )


def write_notebook(*, execute: bool, timeout: int) -> Path:
    notebook = build_mlat_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)
    if execute:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="project1-venv",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=3_600)
    args = parser.parse_args()
    path = write_notebook(execute=args.execute, timeout=args.timeout)
    action = "Built and executed" if args.execute else "Built"
    print(f"{action}: {path}")


if __name__ == "__main__":
    main()
