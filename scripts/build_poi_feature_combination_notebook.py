"""Build and execute the governed POI-first feature-combination notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/exploration/poi_feature_combination_research.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    """Return the concise executable notebook for the frozen v1 study."""

    cells = [
        new_markdown_cell(
            """# POI-first feature-combination research

This notebook tests whether previously researched features add directional value first at True POI retests and, only after the POI branch fails to produce a complete candidate, in the general GC observation stream. The frozen contract is `project_docs/poi_feature_combination_research_contract_v1.md`.

The decision is the completed retest bar close and entry is the next one-minute bar open. Development is used for chronological OOF selection. Retrospective 2024 Validation is opened only for a Development-selected model and its required anchor. The 2025+ Final-test partition is never opened. Opportunity/range predictions are diagnostic and never provide trade direction. No result here authorizes live GC or MGC execution."""
        ),
        new_code_cell(
            """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "CLAUDE.md").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate the Project 1 repository root")


ROOT = find_project_root(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.statistical_research.feature_combination_runner import run_feature_combination_research

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)
print(f"Repository: {ROOT}")"""
        ),
        new_markdown_cell(
            """## 1. Governed execution

The runner records byte hashes and projected schemas, fixes chronological folds before loading any outcome, deduplicates POI candidates before the label loader, persists the POI verdict before the general branch can open, and rejects any materialized 2025+ row."""
        ),
        new_code_cell(
            """result = run_feature_combination_research(ROOT)
display(pd.Series(result.headline, name="value").to_frame())"""
        ),
        new_markdown_cell("## 2. Population and chronological support"),
        new_code_cell(
            """display(result.population_summary)
display(result.fold_summary[[
    "branch", "session", "fold_id", "train_date_count", "train_end",
    "embargo_dates", "assessment_date_count", "assessment_start", "assessment_end"
]])"""
        ),
        new_markdown_cell(
            """## 3. Directional combination evidence

Daily rank IC uses New York dates with at least ten finite prediction/target pairs; date-block stationary bootstrap is the inference unit. The model gate separately reports all finite OOF evaluation dates so sparse POI dates are not silently discarded. Additive models must beat their frozen anchor on common IC dates."""
        ),
        new_code_cell(
            """directional = result.model_metrics.loc[
    result.model_metrics["target_role"].eq("DIRECTIONAL"),
    [
        "branch", "session", "model_id", "partition", "observation_count",
        "evaluation_date_count", "ic_date_count", "daily_ic_mean",
        "daily_ic_ci_low", "daily_ic_ci_high", "paired_daily_ic_improvement",
        "paired_daily_ic_ci_low", "sign_accuracy", "balanced_accuracy",
        "best_ten_date_positive_evidence_share"
    ],
].sort_values(["branch", "session", "partition", "model_id"])
display(directional)"""
        ),
        new_markdown_cell(
            """## 4. Opportunity diagnostics

These rows answer whether the combinations forecast future 60-minute range. They are explicitly nonselectable: an opportunity forecast is not a long/short signal."""
        ),
        new_code_cell(
            """opportunity = result.model_metrics.loc[
    result.model_metrics["target_role"].eq("OPPORTUNITY_DIAGNOSTIC"),
    [
        "branch", "session", "model_id", "partition", "observation_count",
        "evaluation_date_count", "ic_date_count", "daily_ic_mean",
        "daily_ic_ci_low", "daily_ic_ci_high", "paired_daily_ic_improvement"
    ],
].sort_values(["branch", "session", "model_id"])
display(opportunity)"""
        ),
        new_markdown_cell("## 5. Frozen selection and policy verdicts"),
        new_code_cell(
            """display(result.selection_verdicts)
display(result.policy_thresholds)
display(result.policy_metrics.sort_values(
    ["branch", "model_id", "partition", "round_trip_cost_ticks"]
))"""
        ),
        new_markdown_cell("## 6. Compact visual audit"),
        new_code_cell(
            """plot_table = directional.loc[directional["partition"].eq("Development OOF")].copy()
plot_table["cell"] = (
    plot_table["branch"] + " | " + plot_table["session"] + " | " + plot_table["model_id"]
)
plot_table = plot_table.sort_values("daily_ic_mean")

figure_dir = ROOT / "reports/statistical_research/poi_feature_combination/v1/figures"
figure_dir.mkdir(parents=True, exist_ok=True)
figure_path = figure_dir / "development_directional_daily_ic.png"

fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(plot_table))))
colors = np.where(plot_table["daily_ic_mean"] > 0, "#2f6b4f", "#a04444")
ax.barh(plot_table["cell"], plot_table["daily_ic_mean"], color=colors)
ax.axvline(0.0, color="#333333", linewidth=0.8)
ax.axvline(0.02, color="#c58b2b", linewidth=1.0, linestyle="--", label="Development IC gate")
ax.set_xlabel("Mean daily Spearman IC")
ax.set_title("Development OOF directional IC: POI-first and general combinations")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()
print(f"Saved: {figure_path.relative_to(ROOT)}")"""
        ),
        new_markdown_cell(
            """## 7. Senior quant interpretation

The decisive distinction is between *reaction opportunity* and *direction*. POI levels can coincide with larger subsequent movement without yielding a cost-surviving continuation/reversal forecast. The same distinction applies to the frozen Branch B, FES, and MLAT evidence: strong future-range association does not create a direction or an executable policy.

MGC must not inherit a GC boundary-price fill assumption. A future passing GC specification would require synchronized MGC next-bar-open execution and fresh telemetry for spread, slippage, and fill containment before deployment review."""
        ),
        new_code_cell(
            """assert result.headline["status"] == "READY"
assert result.headline["final_test_opened"] is False
assert result.headline["mgc_test_run"] is False
assert result.headline["poi_first_completed_before_general"] is True
assert not result.access_audit["maximum_trade_date"].dropna().ge(pd.Timestamp("2025-01-01")).any()

print(f"STATUS: {result.headline['status']}")
print(f"OVERALL STATE: {result.headline['overall_state']}")
print("FINAL TEST: UNOPENED")
print("LIVE GC/MGC AUTHORIZATION: NONE")"""
        ),
    ]
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "research_contract": "poi_feature_combination_v1",
        },
    )
    nbformat.validate(notebook)
    return notebook


def write_notebook(*, execute: bool = False, timeout: int = 7_200) -> Path:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)
    if execute:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=7_200)
    args = parser.parse_args()
    path = write_notebook(execute=args.execute, timeout=args.timeout)
    print(path)


if __name__ == "__main__":
    main()
