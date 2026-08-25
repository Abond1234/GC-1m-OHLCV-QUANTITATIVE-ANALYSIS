"""Build and execute the reproducible Tsay notebook through Stage 4."""

from __future__ import annotations

import argparse
import platform
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "exploration" / "tsay_feature_research.ipynb"


def _md(source: str) -> nbformat.NotebookNode:
    return new_markdown_cell(textwrap.dedent(source).strip())


def _code(source: str) -> nbformat.NotebookNode:
    return new_code_cell(textwrap.dedent(source).strip())


def _build_stage3_notebook() -> nbformat.NotebookNode:
    """Return the concise notebook containing completed Stages 1 through 3."""

    cells = [
        _md(
            """
            # Tsay Feature Research

            Project One v1 through **Stage 3 only**. Stages 1 and 2 froze the
            contract and constructed T01-T09 without outcomes. Stage 3 opens only
            Development labels through 2023-12-31 for the prescribed 18-test
            confirmatory feature-evidence family.

            Validation labels/performance, MGC, every 2025+ row, predictive models,
            policy P&L, and Stage 4 remain unopened. The Tsay PDF remains
            traceability-only.
            """
        ),
        _md(
            """
            ## 1. Verified continuation and deterministic Stage 3 run

            The runner verifies the prior `STATUS: READY`, Stage 2 artifact hashes,
            contract/amendment hashes, and access manifest before the authorized
            Development label scan.
            """
        ),
        _code(
            """
            from __future__ import annotations

            import json
            import os
            import sys
            from pathlib import Path

            import pandas as pd


            def find_repo_root(start: Path) -> Path:
                for candidate in (start.resolve(), *start.resolve().parents):
                    if (candidate / ".git").exists() and (candidate / "CLAUDE.md").exists():
                        return candidate
                raise RuntimeError(f"Repository root not found from {start}")


            ROOT = find_repo_root(Path.cwd())
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            if str(ROOT / "src") not in sys.path:
                sys.path.insert(0, str(ROOT / "src"))
            os.environ["PROJECT_COMPUTE"] = "cpu"

            from scripts.run_tsay_feature_research import ARTIFACT_ROOT, run_stage3
            from src.statistical_research.tsay_artifacts import load_parquet

            CHECKPOINT = run_stage3()
            CHECKPOINT
            """
        ),
        _code(
            """
            def load_json(name: str):
                return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


            prior = load_json("stage2_continuation_verification_stage3.json")
            assert prior["status"] == "VERIFIED"
            assert prior["stage2_status"] == "READY"
            {
                "stage2_hash_manifest_sha256": prior["stage2_hash_manifest_sha256"],
                "contract_sha256": prior["contract_sha256"],
                "amendment_sha256": prior["amendment_sha256"],
                "feature_matrix_sha256": prior["feature_matrix_sha256"],
                "stage2_access_manifest_sha256": prior["access_manifest_sha256"],
            }
            """
        ),
        _md(
            """
            ## 2. Access boundary and label consistency

            Every returned time-series table is GC Development only, ends no later
            than 2023-12-31, and records its exact columns. Return/range ATR fields
            are checked rowwise against the corresponding tick field and decision ATR;
            no average-ATR conversion is used.
            """
        ),
        _code(
            """
            access = load_json("access_manifest_stage3.json")
            assert access["stage"] == 3
            assert access["materialization_cutoff_date_ny"] == "2023-12-31"
            assert all(
                record["returned_maximum_date"] is None
                or record["returned_maximum_date"] <= "2023-12-31"
                for record in access["records"]
            )
            assert all(
                record["returned_research_partition_set"] in (None, ["Development"])
                for record in access["records"]
            )
            pd.DataFrame(access["records"])[[
                "source", "requested_columns", "returned_row_count",
                "returned_research_partition_set", "returned_minimum_date",
                "returned_maximum_date",
            ]]
            """
        ),
        _code(
            """
            label_qa = load_parquet(
                ARTIFACT_ROOT / "development_label_atr_tick_qa.parquet"
            )
            assert label_qa["rowwise_consistent"].all()
            assert not label_qa["aggregate_average_atr_conversion_used"].any()
            label_qa
            """
        ),
        _md(
            """
            ## 3. Confirmatory evidence and mechanical gates

            The 18 primary tests use daily Spearman IC (minimum 10 observations),
            frozen orientation, a 2,000-replicate stationary date bootstrap with
            restart probability 0.20 and seed 20260824, and one BH family. Quintile
            edges are noninterpolated Development order statistics and are persisted
            for later unchanged use. Negative findings are retained without tuning or
            replacement.
            """
        ),
        _code(
            """
            evidence = load_parquet(
                ARTIFACT_ROOT / "development_feature_evidence_ledger.parquet"
            )
            assert len(evidence) == 18
            evidence[[
                "logical_id", "entry_session", "role", "valid_dates",
                "valid_observations", "mean_oriented_ic", "bh_q_value",
                "bootstrap_lower_oriented", "absolute_quintile_monotonicity",
                "oriented_top_bottom_ticks", "partial_absolute_retention",
                "partial_bootstrap_lower_oriented", "failed_gate_reasons", "status",
            ]]
            """
        ),
        _code(
            """
            evidence.groupby(["status"], observed=True).size().rename("tests").to_frame()
            """
        ),
        _md(
            """
            ## 4. Partial information, stability, and redundancy

            Partial ranks use the exact common candidate/target/control panel per
            session/date, the frozen anchor plus explicit controls, fixed 15-minute
            clock dummies, float64 OLS, and the contract's support/rank/condition gates.
            Missing partial results remain `NOT_EVALUABLE`. Redundancy tables are
            diagnostic and do not silently select candidates.
            """
        ),
        _code(
            """
            partial = load_parquet(ARTIFACT_ROOT / "development_partial_summary.parquet")
            stability = load_parquet(ARTIFACT_ROOT / "development_year_stability.parquet")
            redundancy = load_parquet(
                ARTIFACT_ROOT / "development_related_feature_redundancy.parquet"
            )
            display(partial[[
                "logical_id", "entry_session", "valid_dates",
                "mean_oriented_partial_ic", "absolute_retention",
                "bootstrap_lower_oriented", "unavailable_date_count",
            ]])
            display(stability.head(12))
            redundancy.loc[
                redundancy.groupby(["logical_id", "entry_session"])[
                    "pooled_spearman"
                ].apply(lambda values: values.abs().idxmax())
            ]
            """
        ),
        _md(
            """
            ## 5. Thirty-minute diagnostic only

            The 30-minute horizon reports sign and stability but has no selection
            authority and cannot rescue a failed 60-minute gate.
            """
        ),
        _code(
            """
            diagnostic = load_parquet(
                ARTIFACT_ROOT / "development_diagnostic_30m_summary.parquet"
            )
            assert not diagnostic["selection_authority"].any()
            diagnostic
            """
        ),
        _md(
            """
            ## 6. Persistence, hashes, and Stage 3 checkpoint

            Every Stage 3 Parquet artifact is saved, reloaded, and compared exactly.
            `STATUS` is reproducibility/integrity; the separate research verdict does
            not authorize a model or policy.
            """
        ),
        _code(
            """
            hashes = load_json("stage3_hash_manifest.json")
            assert hashes["access_manifest_development_cutoff_verified"] is True
            assert hashes["development_label_access_only"] is True
            assert hashes["validation_label_access_absent"] is True
            assert hashes["mgc_access_absent"] is True
            assert hashes["post_2023_materialization_absent"] is True
            {
                "persisted_artifacts": len(hashes["artifact_sha256"]),
                "evidence_ledger_sha256": CHECKPOINT["evidence_ledger_sha256"],
                "access_manifest_sha256": CHECKPOINT["access_manifest_sha256"],
                "stage3_hash_manifest_sha256": CHECKPOINT[
                    "stage3_hash_manifest_sha256"
                ],
            }
            """
        ),
        _code(
            """
            assert CHECKPOINT["status"] == "READY"
            assert CHECKPOINT["development_label_values_read"] is True
            assert CHECKPOINT["validation_label_values_read"] is False
            assert CHECKPOINT["validation_performance_evaluated"] is False
            assert CHECKPOINT["mgc_research_rows_read"] is False
            assert CHECKPOINT["post_2023_rows_materialized"] is False
            assert CHECKPOINT["models_fitted"] is False

            print(f"STATUS: {CHECKPOINT['status']}")
            print(f"RESEARCH_VERDICT: {CHECKPOINT['research_verdict']}")
            print(f"STAGE_4_AUTHORIZED: {CHECKPOINT['stage4_authorized']}")
            print(CHECKPOINT["stage4_authorization_scope"])
            """
        ),
        _md(
            """
            Stage 3 stops here. Stage 4 has not begun and may run only after an
            explicit continuation instruction.
            """
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Project 1 venv",
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
            "tsay": {
                "version": "v1",
                "completed_stage": 3,
                "stage2_present": True,
                "stage3_present": True,
                "development_outcomes_authorized": True,
                "validation_outcomes_authorized": False,
                "stage4_present": False,
                "seed": 20260824,
            },
        },
    )


def build_tsay_notebook() -> nbformat.NotebookNode:
    """Return the concise notebook containing completed Stages 1 through 4."""

    cells = [
        _md(
            """
            # Tsay Feature Research

            Project One v1 through **Stage 4 only**. Stages 1–3 froze the
            contract, built T01–T09 without outcomes, and evaluated the fixed
            Development feature-evidence family. Stage 4 fits only the frozen
            D0–D6, O0–O5, and E0–E3 nested ladder on Development through
            2023-12-31.

            Validation labels/performance, MGC research rows, and every 2024+
            outcome remain unopened. The Tsay PDF remains traceability-only.
            """
        ),
        _md(
            """
            ## 1. Verified continuation and deterministic Stage 4 run

            The runner first verifies the Stage 3 `STATUS: READY`, contract and
            amendment hashes, all Stage 3 artifact hashes, and the Development-only
            access manifest. It then executes the fixed nested folds on CPU.
            """
        ),
        _code(
            """
            from __future__ import annotations

            import json
            import os
            import sys
            from pathlib import Path

            import pandas as pd


            def find_repo_root(start: Path) -> Path:
                for candidate in (start.resolve(), *start.resolve().parents):
                    if (candidate / ".git").exists() and (candidate / "CLAUDE.md").exists():
                        return candidate
                raise RuntimeError(f"Repository root not found from {start}")


            ROOT = find_repo_root(Path.cwd())
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            if str(ROOT / "src") not in sys.path:
                sys.path.insert(0, str(ROOT / "src"))
            os.environ["PROJECT_COMPUTE"] = "cpu"

            from scripts.run_tsay_feature_research import ARTIFACT_ROOT, run_stage4
            from src.statistical_research.tsay_artifacts import load_parquet

            CHECKPOINT = run_stage4()
            CHECKPOINT
            """
        ),
        _code(
            """
            def load_json(name: str):
                return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


            prior = load_json("stage3_continuation_verification_stage4.json")
            assert prior["status"] == "VERIFIED"
            assert prior["stage3_status"] == "READY"
            prior
            """
        ),
        _md(
            """
            ## 2. Development access boundary

            Stage 4 reads only GC Development rows through 2023-12-31 and only
            the prescribed feature inputs, 60-minute targets, expansion label,
            availability flag, and exit timestamp required for label purging.
            """
        ),
        _code(
            """
            access = load_json("access_manifest_stage4.json")
            assert access["stage"] == 4
            assert access["materialization_cutoff_date_ny"] == "2023-12-31"
            assert all(
                record["returned_maximum_date"] is None
                or record["returned_maximum_date"] <= "2023-12-31"
                for record in access["records"]
            )
            pd.DataFrame(access["records"])[[
                "source", "requested_columns", "returned_row_count",
                "returned_research_partition_set", "returned_maximum_date",
            ]]
            """
        ),
        _md(
            """
            ## 3. Nested OOF model evidence

            Every candidate uses the frozen outer folds, purged inner folds,
            fold-local transforms, one-standard-error tuning, matched anchor rows,
            stationary-bootstrap inference, and family max-statistic correction.
            Negative and non-evaluable findings remain in the ledgers.
            """
        ),
        _code(
            """
            continuous = load_parquet(
                ARTIFACT_ROOT / "development_continuous_model_evidence.parquet"
            )
            classifier = load_parquet(
                ARTIFACT_ROOT / "development_classifier_model_evidence.parquet"
            )
            selection = load_parquet(
                ARTIFACT_ROOT / "development_architecture_selection.parquet"
            )
            display(continuous)
            display(classifier)
            display(load_parquet(ARTIFACT_ROOT / "development_model_max_stat_audit.parquet"))
            selection
            """
        ),
        _md(
            """
            ## 4. Specialized diagnostics and frozen policy state

            Quantile, VAR, Kalman, calibration, convergence, and coefficient
            stability diagnostics are persisted separately. Real bars and policy
            economics remain unopened unless a directional architecture passes all
            Development predictive gates.
            """
        ),
        _code(
            """
            display(load_parquet(ARTIFACT_ROOT / "development_quantile_diagnostics.parquet"))
            display(load_parquet(ARTIFACT_ROOT / "development_classifier_reliability.parquet"))
            display(load_parquet(ARTIFACT_ROOT / "development_var_diagnostics.parquet").head())
            display(load_parquet(ARTIFACT_ROOT / "development_kalman_diagnostics.parquet").head())
            policy = load_json("frozen_policy_state.json")
            policy
            """
        ),
        _md(
            """
            ## 5. Persistence, hashes, and Stage 4 checkpoint

            All Stage 4 artifacts were saved, reloaded, compared, and hashed before
            this checkpoint. `STATUS` reports integrity separately from the research
            verdict and authorization fields.
            """
        ),
        _code(
            """
            hashes = load_json("stage4_hash_manifest.json")
            assert hashes["development_label_access_only"] is True
            assert hashes["validation_label_access_absent"] is True
            assert hashes["mgc_access_absent"] is True
            assert hashes["post_2023_materialization_absent"] is True
            assert CHECKPOINT["status"] == "READY"
            assert CHECKPOINT["validation_outcomes_read"] is False
            assert CHECKPOINT["mgc_research_rows_read"] is False
            assert CHECKPOINT["post_2023_rows_materialized"] is False

            print(f"STATUS: {CHECKPOINT['status']}")
            print(f"RESEARCH_VERDICT: {CHECKPOINT['research_verdict']}")
            print(
                "PREDICTIVE_MODEL_AUTHORIZATION: "
                f"{CHECKPOINT['predictive_model_authorization']}"
            )
            print(f"POLICY_AUTHORIZED: {CHECKPOINT['policy_authorized']}")
            print(
                "VALIDATION_ECONOMICS_MAY_OPEN: "
                f"{CHECKPOINT['validation_economics_may_open']}"
            )
            print(f"STAGE_5_AUTHORIZED: {CHECKPOINT['stage5_authorized']}")
            """
        ),
        _md(
            """
            Stage 4 stops here. Stage 5 has not begun and may run only after a
            separate explicit continuation instruction and the recorded authorization.
            """
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Project 1 venv",
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
            "tsay": {
                "version": "v1",
                "completed_stage": 4,
                "stage2_present": True,
                "stage3_present": True,
                "stage4_present": True,
                "development_outcomes_authorized": True,
                "validation_outcomes_authorized": False,
                "seed": 20260824,
            },
        },
    )


def write_notebook(*, execute: bool = False, timeout: int = 3_600) -> Path:
    notebook = build_tsay_notebook()
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
