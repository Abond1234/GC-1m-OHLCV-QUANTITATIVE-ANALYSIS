"""Append the completed Section 6 record to feature_research_1.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/exploration/feature_research_1.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    joined_markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    if "# Section 6 - Single locked Validation batch and GC economics" in joined_markdown:
        raise RuntimeError("Notebook Section 6 already exists.")
    cells = [
        markdown(
            """# Section 6 - Single locked Validation batch and GC economics

The exact Section 5 command was executed once outside the notebook after all 33 frozen artifacts, the policy, model bundle, and base configuration hashes passed. The batch opened 118,076 retrospective Validation rows from 2024, scored the unchanged frozen objects, applied the Development-fitted quintiles and p90/p80 values, and wrote an immutable completion hash without an intermediate choice or refit.

Historical Final remains unread. Because Section 5 froze `FROZEN_NO_POLICY`, the batch did not open GC/MGC economic bars or manufacture direction from the unsigned opportunity model.
"""
        ),
        markdown(
            """### 6.1 One-time completion, hashes, and access boundary

The cells below only reload the completed immutable batch. They do not rerun Validation, refit a model, or alter a result.
"""
        ),
        code(
            """section6_report_dir = ROOT / REPORT_NAMESPACE
section6_data_dir = ROOT / DATA_NAMESPACE
completion6_path = section6_report_dir / 'section6_batch_completion.json'
checkpoint6_path = section6_report_dir / 'section6_checkpoint.json'
manifest6_path = section6_report_dir / 'section6_hash_manifest.json'
completion6 = json.loads(completion6_path.read_text(encoding='utf-8'))
checkpoint6 = json.loads(checkpoint6_path.read_text(encoding='utf-8'))
manifest6 = json.loads(manifest6_path.read_text(encoding='utf-8'))

assert completion6['status'] == 'COMPLETED_ONE_TIME_LOCKED_BATCH'
assert checkpoint6['single_locked_validation_batch_completed'] is True
assert checkpoint6['retrospective_validation_outcomes_read'] is True
assert checkpoint6['historical_final_outcomes_read'] is False
assert checkpoint6['gc_economic_bars_read'] is False
assert checkpoint6['gc_economics_executed'] is False
assert sha256_file(manifest6_path) == completion6['section6_manifest_sha256']
for artifact in manifest6['artifacts']:
    path = ROOT / artifact['path']
    assert path.stat().st_size == artifact['bytes']
    assert sha256_file(path) == artifact['sha256']

section6_access = pd.read_csv(section6_report_dir / 'section6_access_audit.csv')
assert section6_access.loc[
    section6_access['source'].eq('Historical Final outcomes'), 'rows_loaded'
].eq(0).all()
assert section6_access.loc[
    section6_access['source'].eq('GC/MGC economic bars'), 'rows_loaded'
].eq(0).all()

print(f"Section 6 artifacts verified: {len(manifest6['artifacts'])}")
print(f"Batch completion SHA-256: {completion6['batch_completion_sha256']}")
print(f"Section 6 manifest SHA-256: {completion6['section6_manifest_sha256']}")
display(section6_access)
"""
        ),
        markdown(
            """### 6.2 Final feature and interaction gates

Four preregistered base-feature/session hypotheses meet every frozen Development-plus-Validation feature gate. This is feature evidence, not a directional trading policy. No interaction meets the joint incremental gate.
"""
        ),
        code(
            """feature_gates6 = pd.read_csv(section6_report_dir / 'section6_final_feature_gates.csv')
interaction_gates6 = pd.read_csv(section6_report_dir / 'section6_final_interaction_gates.csv')
feature_advancers6 = feature_gates6.loc[
    feature_gates6['final_feature_label'].eq('FEATURE_ADVANCES')
].copy()
interaction_advancers6 = interaction_gates6.loc[
    interaction_gates6['final_interaction_label'].eq('INTERACTION_ADVANCES')
].copy()

assert len(feature_gates6) == 20
assert len(feature_advancers6) == 4
assert len(interaction_gates6) == 8
assert interaction_advancers6.empty
display(feature_advancers6[[
    'session', 'feature_name', 'target',
    'development_daily_ic_mean', 'validation_daily_ic_mean',
    'validation_daily_ic_ci_low', 'validation_daily_ic_ci_high',
    'validation_partial_absolute_retention',
    'development_top_bottom_spread_ticks',
    'validation_top_bottom_spread_ticks',
    'final_feature_label',
]])
display(interaction_gates6[[
    'session', 'interaction_name',
    'development_mean_daily_ic_delta',
    'validation_mean_daily_ic_delta',
    'validation_delta_ci_low', 'validation_delta_ci_high',
    'failed_gates', 'final_interaction_label',
]])
"""
        ),
        markdown(
            """### 6.3 Frozen-model Validation evidence and fixed score thresholds

All six selected continuous objects have positive retrospective Validation IC, but none passes the complete advancement contract. London M8 loses paired IC to D1 on both directional horizons in Validation; the other selected objects are anchors or show no qualifying incremental improvement. Every comparison also retains only 378 Development OOF dates versus the required 400.

The expansion classifiers remain useful descriptive opportunity scores but are ineligible: they are the O1 anchor (zero paired improvement), and their frozen session-specific training thresholds do not match the authoritative single Development-wide expansion-label threshold.
"""
        ),
        code(
            """continuous_validation6 = pd.read_csv(
    section6_report_dir / 'section6_validation_continuous_metrics.csv'
)
model_gates6 = pd.read_csv(section6_report_dir / 'section6_final_model_gates.csv')
expansion_validation6 = pd.read_csv(
    section6_report_dir / 'section6_validation_expansion_metrics.csv'
)
candidate_counts6 = pd.read_csv(section6_report_dir / 'section6_candidate_counts.csv')

assert len(continuous_validation6) == 6
assert not model_gates6['model_advances'].astype(bool).any()
assert not expansion_validation6['classifier_advances'].astype(bool).any()
assert not expansion_validation6['threshold_contract_match'].astype(bool).any()
assert not candidate_counts6['policy_execution_allowed'].astype(bool).any()
display(model_gates6[[
    'session', 'target', 'model_family',
    'development_daily_ic_mean', 'development_paired_delta',
    'development_valid_dates', 'validation_daily_ic_mean',
    'validation_daily_ic_ci_low', 'validation_daily_ic_ci_high',
    'validation_paired_delta', 'validation_paired_delta_ci_low',
    'validation_paired_delta_ci_high', 'validation_valid_dates',
    'failed_gates', 'model_advances',
]])
display(expansion_validation6)
display(candidate_counts6)
"""
        ),
        markdown(
            """### 6.4 GC economics, Sharpe, and exact verdict

The p90/p80 exceedance counts are retained for trial accounting, but they are inactive scores—not tradable candidates—because no 60-minute directional model passes. Therefore zero policies are sequenced, all three cost scenarios are explicitly not applicable, drawdown and daily-net-R Sharpe are undefined, the max-Sharpe adjusted p-value has a policy-family size of zero, and the GC gate cannot permit MGC work.
"""
        ),
        code(
            """gc_economics6 = pd.read_csv(section6_report_dir / 'section6_gc_economics.csv')
gc_drawdown6 = pd.read_csv(section6_report_dir / 'section6_gc_drawdown.csv')
sharpe6 = pd.read_csv(section6_report_dir / 'section6_daily_net_r_sharpe.csv')
gc_verdict6 = pd.read_csv(section6_report_dir / 'section6_gc_gate_verdict.csv')

assert gc_economics6['executed_trades'].eq(0).all()
assert gc_economics6['mean_net_r'].isna().all()
assert gc_drawdown6['maximum_drawdown_r'].isna().all()
assert sharpe6['daily_net_r_sharpe'].isna().all()
assert sharpe6['policies_in_family'].eq(0).all()
assert not gc_verdict6['gc_economics_executed'].astype(bool).any()
assert not gc_verdict6['mgc_work_permitted'].astype(bool).any()
assert gc_verdict6.loc[0, 'headline_verdict'] == 'PREDICTIVE_ONLY_NOT_DIRECTIONAL'
display(gc_economics6)
display(gc_drawdown6)
display(sharpe6)
display(gc_verdict6)
display(Markdown(
    f"**Checkpoint 6:** `{checkpoint6['status']}`  \\n"
    f"Batch: `{completion6['batch_completion_sha256']}`  \\n"
    f"Verdict: `{gc_verdict6.loc[0, 'headline_verdict']}`  \\n"
    f"MGC work permitted: `{bool(gc_verdict6.loc[0, 'mgc_work_permitted'])}`"
))
"""
        ),
        markdown(
            """### 6.5 Checkpoint 6 - stop before MGC transfer

The one-time retrospective Validation batch is complete. Four base-feature hypotheses survive their frozen evidence gates, but no interaction, continuous model, classifier, or directional policy advances. GC economics and Sharpe are not applicable under the preregistered no-direction rule; no economic bar was opened, and the GC gate does not permit Section 7 MGC work.

Headline verdict: **`PREDICTIVE_ONLY_NOT_DIRECTIONAL`**.

**Stop here. Historical Final remains unread. Do not begin MGC transfer before explicit Section 7 authorization.**
"""
        ),
    ]
    notebook["cells"].extend(cells)
    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
