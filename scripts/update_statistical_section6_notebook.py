"""Append the completed Section 6 reader flow to the statistical notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "exploration" / "statistical_feature_research.ipynb"


def md(source: str):
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip() + "\n")


notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
section_6_start = next(
    (
        i
        for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# 6.0 ")
    ),
    len(notebook.cells),
)
notebook.cells = notebook.cells[:section_6_start]

section_6_cells = [
    md(
        """
# 6.0 Feature Engineering Framework

Section 6 constructs a frozen, independent GC candidate-feature matrix. Every predictor is available by the close of completed decision bar `t`; theoretical execution remains the open of bar `t+1`. The section performs feature construction, lineage, dtype optimization, engineering diagnostics, and leakage/boundary validation only. It does not rank features, inspect feature-to-outcome relationships, select predictors, model returns, estimate PnL, or start Section 7.
"""
    ),
    md(
        """
## 6.1 Research Contract, Scope, and Leakage Boundary

The discovery instrument is GC only. MGC is reserved for later transfer validation after the GC shortlist is frozen. Section 6 does not load or use any POI artifact, identifier, retest, geometry, ranking, direction conclusion, or Section 7 POI output.

Feature information ends at completed bar `t`. No entry-bar OHLCV, forward return, future high/low/range/volatility, MFE, MAE, direction/expansion label, target/stop result, later-bar information, or Final-test outcome summary may enter construction. The reusable builder accepts only the trusted full GC bar sequence and the key/metadata eligible-observation frame; it has no label-table argument. Development and Validation are the engineering samples. Final-test feature values are saved under frozen definitions, while Final-test inspection is restricted to schema, row count, null/finite rate, dtype, and transformation-integrity checks.
"""
    ),
    code(
        """
from time import perf_counter
import inspect

from src.statistical_research.feature_engineering import (
    EXPECTED_OBSERVATION_ROWS,
    FEATURE_SOURCE_COLUMNS,
    METADATA_COLUMNS,
    build_feature_matrix,
)
from src.statistical_research.feature_registry import (
    EXPERIMENTAL_FEATURE_NAMES,
    FEATURE_NAMES,
    feature_registry_frame,
    validate_registry,
)
from src.statistical_research.feature_validation import (
    build_manual_feature_audit,
    compute_feature_diagnostics,
    save_feature_outputs,
    validate_feature_matrix,
)

assert section_5_ready, "Section 6 requires the validated Section 5 foundation."
assert EXPECTED_OBSERVATION_ROWS == 586_530
assert set(inspect.signature(build_feature_matrix).parameters) == {
    "full_gc_bars", "eligible_observations", "expected_rows"
}

SECTION_6_STARTED = perf_counter()
SECTION_6_FORBIDDEN_TOKENS = (
    "forward", "future", "mfe", "mae", "direction_label", "expansion_label",
    "target", "stop", "outcome", "poi", "entry_price",
)
section_6_contract = pd.Series(
    {
        "instrument_scope": "GC only",
        "decision_information_end": "completed bar t",
        "theoretical_entry": "open of bar t+1 (not a feature input)",
        "feature_builder_label_argument": False,
        "POI_inputs": False,
        "Final_test_fit_contribution": False,
        "Section_6_predictive_evaluation": False,
    },
    name="locked_value",
)
display(section_6_contract.to_frame())
"""
    ),
    md(
        """
## 6.2 Feature Registry and Hypothesis Inventory

The machine-readable registry is instantiated and validated before matrix assembly. It is the authoritative predictor list for later sections. Identifiers, timestamps, partition labels, contracts, and audit fields are not registered as predictors. Core features form the disciplined conventional foundation. The six experimental hypotheses use clean saved names and an asterisk only in reader-facing display names; every experimental row includes a written creative rationale and the same lineage, missingness, dtype, and out-of-sample contract as the core set.
"""
    ),
    code(
        """
feature_registry_gc = feature_registry_frame()
validate_registry(feature_registry_gc)
feature_inventory = (
    feature_registry_gc.groupby(["family", "feature_tier"], observed=True)
    .size()
    .rename("feature_count")
    .reset_index()
)
registry_counts = pd.Series(
    {
        "registered_predictors": len(feature_registry_gc),
        "core_features": int((~feature_registry_gc["is_experimental"]).sum()),
        "experimental_features": int(feature_registry_gc["is_experimental"].sum()),
        "numeric_features": int(feature_registry_gc["output_dtype"].isin(["float32", "Int16", "Int8"]).sum()),
        "boolean_features": int(feature_registry_gc["output_dtype"].eq("boolean").sum()),
        "categorical_features": int(feature_registry_gc["output_dtype"].eq("category").sum()),
    },
    name="count",
)
display(registry_counts.to_frame())
display(feature_inventory)
display(feature_registry_gc[[
    "feature_name", "display_name", "family", "feature_tier", "formula_or_definition",
    "lookback", "availability_timestamp", "reset_boundary", "output_dtype",
]].head(12))
"""
    ),
    md(
        """
## 6.3 Decision-Time Base Frame and Shared Causal Primitives

Rolling and anchored state is calculated on all 1,759,671 trusted GC bars—not only the 586,530 eligible entry observations. The source namespace is an explicit allow-list of raw OHLCV, timestamps, contract/instrument/segment identity, and tradability/roll/liquidity controls; legacy historical features and every saved forward field are excluded.

A continuity run breaks on a non-one-minute timestamp, product change, selected-contract change, instrument change, continuous-segment change, or tradability/roll/liquidity boundary. Complete-window primitives reuse float64 log returns, signed/absolute changes, range, true range, candle geometry, rolling sums/moments/extrema, regressions, price-volume cumulants, and session counters. Feature values are mapped to decision-bar identifiers only after full-history construction and are downcast only after calculation.
"""
    ),
    code(
        """
# Section 4/5 outcomes remain saved artifacts but are removed from the live Section 6
# namespace. Only the key/metadata eligible frame is passed to feature construction.
for large_label_name in (
    "forward_labels_gc", "reloaded_forward_labels_gc", "gc_path_source",
    "saved_forward_arrow", "reloaded_forward_arrow",
):
    globals().pop(large_label_name, None)

section_6_raw_columns = [name for name in FEATURE_SOURCE_COLUMNS if name != "source_row_id"]
assert not any(
    token in column.lower()
    for column in section_6_raw_columns
    for token in SECTION_6_FORBIDDEN_TOKENS
)
gc_feature_source = research_bars.loc[gc_source_mask, section_6_raw_columns].copy()
gc_feature_source.insert(0, "source_row_id", pd.array(gc_source_row_ids, dtype="int64[pyarrow]"))
assert tuple(gc_feature_source.columns) == FEATURE_SOURCE_COLUMNS
assert len(gc_feature_source) == 1_759_671
assert gc_feature_source["source_row_id"].is_unique
assert gc_feature_source["source_row_id"].is_monotonic_increasing

feature_build_result = build_feature_matrix(gc_feature_source, eligible_observations_gc)
feature_matrix_gc = feature_build_result.matrix
feature_reference_parameters_gc = feature_build_result.reference_parameters
display(feature_build_result.build_timings)
display(feature_build_result.construction_audit)
"""
    ),
    md(
        """
## 6.4 Price, Return, Momentum, and Multi-Horizon Alignment

Completed close-to-close log returns are stored in basis points at 1, 5, 15, 30, and 60 minutes. Multi-bar price displacement is separately normalized by the current causal 20-bar ATR. The compact family also includes one-minute absolute movement, two per-minute momentum-acceleration contrasts, a signed capped directional streak, 15/60-bar range position, and strict non-zero 5/30-minute directional alignment. High/low distance pairs and duplicate absolute transforms are omitted because they would be exact or near-deterministic representations of retained range-position and return fields.
"""
    ),
    code(
        """
price_registry = feature_registry_gc.loc[
    feature_registry_gc["family"].astype(str).eq("price_return_momentum"),
    ["feature_name", "formula_or_definition", "lookback", "expected_range", "output_dtype"],
]
display(price_registry)
display(feature_matrix_gc[["observation_id", *price_registry["feature_name"].head(6)]].head(3))
"""
    ),
    md(
        """
## 6.5 Volatility, Range, Compression, and Expansion State

This family describes only completed movement state: 5/20/60-bar true-range averages, causal ATR ratios, 5/15/30/60-bar realized log variation, two realized-volatility ratios, current completed range in ATR units, and 5/30-bar range compression. It does not use Section 4 expansion labels, future realized volatility, future ranges, or Development expansion thresholds.
"""
    ),
    code(
        """
volatility_registry = feature_registry_gc.loc[
    feature_registry_gc["family"].astype(str).eq("volatility_range_state"),
    ["feature_name", "formula_or_definition", "lookback", "expected_range"],
]
display(volatility_registry)
"""
    ),
    md(
        """
## 6.6 Candle Geometry and Local Price-Action State

Continuous body, wick, close-location, relative-range, and two/three-bar directional-balance measures take priority over named candlestick libraries. A zero-range bar produces null for undefined ratios. Inside/outside indicators use nullable boolean dtype and require a valid previous bar inside the same continuity run.
"""
    ),
    code(
        """
candle_registry = feature_registry_gc.loc[
    feature_registry_gc["family"].astype(str).eq("candle_geometry"),
    ["feature_name", "formula_or_definition", "missing_value_policy", "output_dtype"],
]
display(candle_registry)
"""
    ),
    md(
        """
## 6.7 Volume, Activity, and Time-of-Day Normalization

Volume is treated as participation and activity context, not true signed order flow. Rolling features use `log1p(volume)`, causal 20/60-bar relative volume, a 60-bar log-volume z-score, 5/20-bar acceleration, volume per GC range tick, an explicitly approximate close-location signed-volume proxy, and rolling return/log-volume correlation.

The New York clock reference groups `entry_session` plus a 15-minute entry bin. Counts, log-volume means, and standard deviations are fitted only on Development. Each Development row subtracts all observations from its own New York trading date before its reference moment is calculated. Validation and Final test use the frozen full-Development table unchanged. Unsupported groups would remain null. Partial 2021/2026 periods contribute only observed dates and receive no annual reweighting.
"""
    ),
    code(
        """
volume_registry = feature_registry_gc.loc[
    feature_registry_gc["family"].astype(str).eq("volume_activity"),
    ["feature_name", "formula_or_definition", "fit_scope", "expected_range"],
]
display(volume_registry)
display(feature_reference_parameters_gc.head(8))
"""
    ),
    md(
        """
## 6.8 VWAP, Anchored Location, and Session-State Features

VWAP uses one-minute typical price `(high + low + close) / 3`. Research-day state resets at 01:00 New York; London execution state at 03:00; New York execution state at 07:00. Rolling VWAP uses complete 20/60-bar continuity windows. All cumulative highs, lows, opens, ranges, and price-volume sums end at decision bar `t`. The first eligible entry exactly at 03:00 or 07:00 correctly has null execution-session state because its decision bar precedes the anchor bar.
"""
    ),
    code(
        """
vwap_registry = feature_registry_gc.loc[
    feature_registry_gc["family"].astype(str).eq("vwap_session_state"),
    ["feature_name", "formula_or_definition", "reset_boundary", "minimum_history"],
]
display(vwap_registry)
"""
    ),
    md(
        """
## 6.9 Trend, Persistence, Efficiency, and Mean-Reversion State

The compact trend-state family includes 15/30/60-bar path efficiency and ATR-normalized OLS slopes, 30-bar OLS R-squared, 15-bar directional persistence and adjacent-return autocorrelation, a 30-return sign-change rate, and 14-bar choppiness. Complete windows and non-zero regression denominators are required. These values describe recent state; Section 6 makes no claim that autocorrelation, trend, efficiency, or choppiness predicts an outcome.
"""
    ),
    code(
        """
trend_registry = feature_registry_gc.loc[
    feature_registry_gc["family"].astype(str).eq("trend_persistence"),
    ["feature_name", "formula_or_definition", "lookback", "expected_range"],
]
display(trend_registry)
"""
    ),
    md(
        """
## 6.10 Session, Clock, and Calendar Context

The scheduled timestamp of entry bar `t+1` is known at decision time and supplies London/New York session, minute from execution-window open, progress fraction, minutes to the noon entry cutoff, minutes to the 15:30 forced exit, New York minute-of-day, cyclical time encoding, and weekday. Session and weekday remain categorical; year, partition, row number, observation identifier, and contract identifiers are audit metadata rather than predictive numeric fields. No calendar or clock trading filter is created.
"""
    ),
    code(
        """
context_registry = feature_registry_gc.loc[
    feature_registry_gc["family"].astype(str).eq("session_clock_calendar"),
    ["feature_name", "formula_or_definition", "output_dtype", "expected_range"],
]
display(context_registry)
display(feature_matrix_gc[["observation_id", *context_registry["feature_name"]]].head(3))
"""
    ),
    md(
        """
## 6.11 Experimental and Niche Hypothesis Features (*)

The experimental batch is intentionally limited to six predefined OHLCV hypotheses. Each is causal, versioned, visibly marked with an asterisk below, and subject to the same validation and later out-of-sample standards as every core feature. Wick pressure and liquidity vacuum are explicitly proxies—not measured order flow, market depth, or liquidity.
"""
    ),
    code(
        """
experimental_registry = feature_registry_gc.loc[
    feature_registry_gc["is_experimental"],
    ["feature_name", "display_name", "formula_or_definition", "creative_rationale", "expected_range"],
]
assert experimental_registry["display_name"].astype(str).str.startswith("* ").all()
assert not experimental_registry["feature_name"].astype(str).str.contains(r"\\*").any()
display(experimental_registry)
"""
    ),
    md(
        """
## 6.12 Feature Matrix Assembly, Lineage, and Type Optimization

The matrix preserves one stable row per eligible observation and stores 12 join/audit metadata columns followed by the registry's authoritative predictor order. It is not joined to the forward-label table. Sensitive calculations use float64; saved ordinary continuous values use float32, binary patterns use nullable booleans, counters use small nullable integers, and repeated strings use categorical/dictionary encoding. The unoptimized figure below is an explicit float64/object estimate; the optimized figure is the measured in-memory matrix footprint.
"""
    ),
    code(
        """
matrix_engineering_summary = pd.Series(
    {
        "rows": feature_matrix_gc.shape[0],
        "columns": feature_matrix_gc.shape[1],
        "metadata_columns": len(METADATA_COLUMNS),
        "registered_features": len(FEATURE_NAMES),
        "core_features": len(FEATURE_NAMES) - len(EXPERIMENTAL_FEATURE_NAMES),
        "experimental_features": len(EXPERIMENTAL_FEATURE_NAMES),
        "estimated_unoptimized_MiB": feature_build_result.estimated_unoptimized_memory_bytes / 2**20,
        "optimized_MiB": feature_build_result.optimized_memory_bytes / 2**20,
        "estimated_peak_working_MiB": feature_build_result.estimated_peak_working_memory_bytes / 2**20,
    },
    name="value",
)
display(matrix_engineering_summary.to_frame())
display(pd.DataFrame({
    "column": feature_matrix_gc.columns,
    "dtype": feature_matrix_gc.dtypes.astype(str).to_numpy(),
}).head(24))
display(feature_matrix_gc[[
    "observation_id", "decision_timestamp_ny", "entry_timestamp_ny", "entry_session",
    "research_partition", "return_5m_bps", "current_range_over_atr",
    "relative_volume_20", "distance_from_research_day_vwap_atr",
    "efficiency_ratio_30", "directional_energy_balance_15_exp",
]].head(5))
"""
    ),
    md(
        """
## 6.13 Leakage, Boundary, Numerical, and Synthetic Validation

Reusable tests cover registry identity, future/entry-bar mutation invariance, gaps, contract/instrument/segment/tradability boundaries, research-day/London/New York resets, zero denominators, regression zero variance, experimental bounds, dtypes, Development leave-one-date-out exclusion, and Validation/Final-test reference independence. Production gates then check all rows, names, lineages, finite/range behavior, missingness, exact duplicates, and fitted-reference support. A fixed-seed independent reconstruction audits early/late London, early/late New York, and the eligible observation nearest a contract/segment boundary, with one representative calculation from every feature family.
"""
    ),
    code(
        """
import io
import unittest

section_6_test_stream = io.StringIO()
section_6_test_suite = unittest.defaultTestLoader.loadTestsFromName(
    "tests.test_statistical_research_features"
)
section_6_test_result = unittest.TextTestRunner(
    stream=section_6_test_stream, verbosity=2
).run(section_6_test_suite)
print(section_6_test_stream.getvalue())
assert section_6_test_result.wasSuccessful()

feature_validation_gc = validate_feature_matrix(feature_build_result, eligible_observations_gc)
manual_feature_audit = build_manual_feature_audit(
    gc_feature_source, eligible_observations_gc, feature_build_result, random_seed=RANDOM_SEED
)
assert manual_feature_audit["passed"].all(), manual_feature_audit.loc[~manual_feature_audit["passed"]]

test_and_manual_checks = pd.DataFrame(
    [
        {
            "category": "automated_tests",
            "check_name": f"synthetic_feature_suite_{section_6_test_result.testsRun}_of_{section_6_test_result.testsRun}",
            "critical": True,
            "passed": section_6_test_result.wasSuccessful(),
            "details": f"{section_6_test_result.testsRun}/{section_6_test_result.testsRun} tests passed; failures={len(section_6_test_result.failures)}; errors={len(section_6_test_result.errors)}",
        },
        {
            "category": "manual_audit",
            "check_name": "fixed_seed_manual_reconstruction_every_family",
            "critical": True,
            "passed": bool(manual_feature_audit["passed"].all()),
            "details": f"{int(manual_feature_audit['passed'].sum())}/{len(manual_feature_audit)} reconstructions passed across five timing/boundary cases and eight families",
        },
    ]
)
feature_validation_gc = pd.concat([feature_validation_gc, test_and_manual_checks], ignore_index=True)
display(feature_validation_gc)
display(manual_feature_audit.groupby(["audit_case", "family"], observed=True)["passed"].agg(["count", "sum"]))
assert feature_validation_gc.loc[feature_validation_gc["critical"], "passed"].all()
"""
    ),
    md(
        """
## 6.14 Feature Coverage and Engineering Diagnostics

Development receives per-feature non-null, finite, unique-count, mean, standard deviation, extrema, and 1st/median/99th-percentile diagnostics. Missingness is also reported by feature family, London/New York on the engineering samples, and Development/Validation partition. Final-test diagnostics contain only integrity-oriented row, non-null, and finite rates; no Final-test distribution, label relationship, ranking, tuning, correlation, or redundancy selection is performed.
"""
    ),
    code(
        """
feature_diagnostics_gc = compute_feature_diagnostics(feature_build_result)
development_feature_diagnostics = feature_diagnostics_gc.loc[
    feature_diagnostics_gc["analysis_type"].eq("Development_feature_summary")
].copy()
diagnostic_failure_columns = [
    "constant_flag", "exact_duplicate_group", "unexpectedly_extreme_flag",
    "excessive_missingness_flag", "invalid_range_flag",
]
diagnostic_gate_passed = bool(
    ~development_feature_diagnostics["constant_flag"].any()
    and development_feature_diagnostics["exact_duplicate_group"].eq("").all()
    and ~development_feature_diagnostics["unexpectedly_extreme_flag"].any()
    and ~development_feature_diagnostics["excessive_missingness_flag"].any()
    and ~development_feature_diagnostics["invalid_range_flag"].any()
)
diagnostic_check = pd.DataFrame([
    {
        "category": "engineering_diagnostics",
        "check_name": "no_constant_duplicate_extreme_excessive_missing_or_invalid_range_feature",
        "critical": True,
        "passed": diagnostic_gate_passed,
        "details": "Development-only diagnostic flags checked before save",
    }
])
feature_validation_gc = pd.concat([feature_validation_gc, diagnostic_check], ignore_index=True)
display(feature_inventory)
display(development_feature_diagnostics.sort_values("missing_rate", ascending=False)[[
    "feature_name", "family", "non_null_count", "missing_rate", "finite_rate",
    "unique_value_count", "mean", "standard_deviation", "minimum", "p01", "median", "p99", "maximum",
]].head(15))
display(feature_diagnostics_gc.loc[
    feature_diagnostics_gc["analysis_type"].isin(["family_missingness", "partition_integrity"]),
    ["analysis_type", "group_name", "group_value", "feature_name", "family", "row_count", "non_null_count", "missing_rate", "finite_rate"],
].head(30))
assert feature_validation_gc.loc[feature_validation_gc["critical"], "passed"].all()
"""
    ),
    md(
        """
## 6.15 Save Outputs, Section Summary, and Completion Gate

Approved outputs are written with Zstandard-compressed Parquet and compact readable CSV/Markdown reports. Independent reload gates verify shape, order, null counts, timezone metadata, registered dtypes, registry alignment, reference/diagnostic schemas, and deterministic sample values. `READY` is derived only from the complete critical validation table, automated tests, manual audits, engineering diagnostics, and save/reload checks.
"""
    ),
    code(
        """
section_6_test_details = f"{section_6_test_result.testsRun}/{section_6_test_result.testsRun} passed"
feature_save_result = save_feature_outputs(
    feature_build_result,
    feature_validation_gc,
    feature_diagnostics_gc,
    manual_feature_audit,
    project_root=PROJECT_ROOT,
    test_details=section_6_test_details,
)
feature_validation_gc = feature_save_result.validation
section_6_ready = bool(
    feature_save_result.ready
    and section_6_test_result.wasSuccessful()
    and manual_feature_audit["passed"].all()
    and diagnostic_gate_passed
    and feature_validation_gc.loc[feature_validation_gc["critical"], "passed"].all()
)
assert section_6_ready, feature_validation_gc.loc[
    feature_validation_gc["critical"] & ~feature_validation_gc["passed"]
]

section_6_runtime_seconds = perf_counter() - SECTION_6_STARTED
display(feature_save_result.reload_checks)
display(pd.Series(
    {
        "matrix_shape": str(feature_matrix_gc.shape),
        "metadata_columns": len(METADATA_COLUMNS),
        "core_features": 79,
        "experimental_features": 6,
        "critical_checks_passed": int(feature_validation_gc.loc[feature_validation_gc["critical"], "passed"].sum()),
        "critical_checks_total": int(feature_validation_gc["critical"].sum()),
        "Section_6_runtime_seconds": section_6_runtime_seconds,
        "optimized_matrix_MiB": feature_build_result.optimized_memory_bytes / 2**20,
    },
    name="value",
).to_frame())
display(Markdown(feature_save_result.report_markdown))
print("SECTION 6 STATUS: READY" if section_6_ready else "SECTION 6 STATUS: NOT READY")
"""
    ),
]

notebook.cells.extend(section_6_cells)
nbformat.write(notebook, NOTEBOOK_PATH)
print(
    f"Updated {NOTEBOOK_PATH} with {len(section_6_cells)} Section 6 cells; total={len(notebook.cells)}"
)
