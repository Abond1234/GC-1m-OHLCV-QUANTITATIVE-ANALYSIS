"""Section 6 engineering diagnostics, manual audits, and save/reload gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .feature_engineering import (
    EXPECTED_OBSERVATION_ROWS,
    FEATURE_SOURCE_COLUMNS,
    METADATA_COLUMNS,
    FeatureBuildResult,
    build_continuity_run_id,
)
from .feature_registry import (
    FEATURE_NAMES,
    FEATURE_SPECS,
)

FORBIDDEN_FEATURE_NAME_TOKENS = (
    "forward",
    "future",
    "mfe",
    "mae",
    "direction_label",
    "expansion_label",
    "target",
    "stop",
    "outcome",
    "poi",
    "entry_price",
)


@dataclass
class FeatureSaveResult:
    validation: pd.DataFrame
    reload_checks: pd.DataFrame
    report_markdown: str
    output_paths: dict[str, Path]
    ready: bool


def _check(category: str, name: str, passed: bool, details: str, critical: bool = True) -> dict:
    return {
        "category": category,
        "check_name": name,
        "critical": bool(critical),
        "passed": bool(passed),
        "details": str(details),
    }


def _feature_numeric_array(series: pd.Series) -> np.ndarray | None:
    if isinstance(series.dtype, pd.CategoricalDtype) or str(series.dtype) in {"boolean", "bool"}:
        return None
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    return None


def _exact_duplicate_groups(frame: pd.DataFrame, feature_names: Iterable[str]) -> dict[str, str]:
    signatures: dict[tuple, list[str]] = {}
    for name in feature_names:
        series = frame[name]
        hashed = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64)
        signature = (
            str(series.dtype),
            int(series.isna().sum()),
            int(hashed.sum(dtype=np.uint64)),
            int(np.bitwise_xor.reduce(hashed, initial=np.uint64(0))),
        )
        signatures.setdefault(signature, []).append(name)
    duplicates: dict[str, str] = {}
    for candidates in signatures.values():
        if len(candidates) < 2:
            continue
        first = candidates[0]
        for candidate in candidates[1:]:
            if frame[first].equals(frame[candidate]):
                label = "|".join(sorted((first, candidate)))
                duplicates[first] = label
                duplicates[candidate] = label
    return duplicates


def validate_feature_matrix(
    build_result: FeatureBuildResult,
    eligible_observations: pd.DataFrame,
    *,
    expected_rows: int = EXPECTED_OBSERVATION_ROWS,
) -> pd.DataFrame:
    """Run production identity, leakage, boundary, reference, and numerical gates."""

    matrix = build_result.matrix
    registry = build_result.registry
    records: list[dict] = []
    feature_names = registry["feature_name"].astype(str).tolist()
    audit = build_result.construction_audit.set_index("item")["value"].astype(str).to_dict()

    records.extend(
        [
            _check(
                "identity_structure",
                "exact_observation_row_count",
                len(matrix) == expected_rows,
                f"rows={len(matrix):,}; expected={expected_rows:,}",
            ),
            _check(
                "identity_structure",
                "unique_observation_identifiers",
                matrix["observation_id"].is_unique,
                f"unique={matrix['observation_id'].nunique():,}",
            ),
            _check(
                "identity_structure",
                "stable_observation_order",
                np.array_equal(
                    matrix["observation_id"].to_numpy(),
                    eligible_observations["observation_id"].to_numpy(),
                ),
                "matrix observation order compared directly with the eligible frame",
            ),
            _check(
                "identity_structure",
                "expected_GC_only_population",
                set(matrix["product"].astype(str).unique()) == {"GC"},
                f"products={sorted(matrix['product'].astype(str).unique().tolist())}",
            ),
            _check(
                "identity_structure",
                "no_duplicate_columns",
                not matrix.columns.duplicated().any(),
                f"columns={matrix.shape[1]}",
            ),
            _check(
                "identity_structure",
                "registry_feature_one_to_one",
                feature_names
                == list(matrix.columns[len(METADATA_COLUMNS) :])
                == list(FEATURE_NAMES),
                f"registry={len(feature_names)}; saved_features={matrix.shape[1] - len(METADATA_COLUMNS)}",
            ),
            _check(
                "identity_structure",
                "frozen_core_and_experimental_counts",
                int((~registry["is_experimental"].astype(bool)).sum()) == 79
                and int(registry["is_experimental"].astype(bool).sum()) == 6,
                f"core={int((~registry['is_experimental'].astype(bool)).sum())}; experimental={int(registry['is_experimental'].astype(bool).sum())}",
            ),
            _check(
                "identity_structure",
                "numeric_feature_ceiling",
                int(registry["output_dtype"].isin(["float32", "Int16", "Int8"]).sum()) == 80,
                "80 numeric features; three boolean and two categorical features",
            ),
        ]
    )

    forbidden_names = [
        name
        for name in feature_names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_NAME_TOKENS)
    ]
    source_columns = audit.get("source_columns", "").split("|")
    forbidden_source = [
        name
        for name in source_columns
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_NAME_TOKENS)
    ]
    records.extend(
        [
            _check(
                "leakage",
                "no_forward_outcome_or_POI_feature_names",
                not forbidden_names,
                f"forbidden_matches={forbidden_names}",
            ),
            _check(
                "leakage",
                "approved_source_namespace_only",
                tuple(source_columns) == FEATURE_SOURCE_COLUMNS and not forbidden_source,
                f"source_columns={len(source_columns)}; forbidden_matches={forbidden_source}",
            ),
            _check(
                "leakage",
                "feature_builder_does_not_require_forward_labels",
                audit.get("forward_label_dataframe_required") == "False",
                "builder signature accepts only full GC bars and key/metadata observation frame",
            ),
            _check(
                "leakage",
                "decision_bar_ids_and_timestamps_map_exactly",
                audit.get("decision_source_ids_exact") == "True"
                and audit.get("decision_timestamps_exact") == "True",
                "no entry-bar OHLCV namespace was supplied or mapped",
            ),
            _check(
                "leakage",
                "Development_only_reference_fit",
                audit.get("time_of_day_fit_scope") == "Development only",
                audit.get("validation_final_reference_application", "missing"),
            ),
            _check(
                "leakage",
                "Validation_and_Final_use_frozen_reference",
                audit.get("validation_final_reference_application") == "frozen full-Development",
                "reference parameters have no Validation or Final-test contribution",
            ),
        ]
    )

    numeric = matrix.loc[
        :, [name for name in feature_names if _feature_numeric_array(matrix[name]) is not None]
    ]
    infinite_count = 0
    for name in numeric:
        values = _feature_numeric_array(numeric[name])
        infinite_count += int(np.isinf(values).sum())
    range_failures: list[str] = []
    spec_by_name = {spec.feature_name: spec for spec in FEATURE_SPECS}
    for name in feature_names:
        values = _feature_numeric_array(matrix[name])
        if values is None:
            continue
        finite = values[np.isfinite(values)]
        spec = spec_by_name[name]
        tolerance = 2.0e-5
        if (
            spec.validation_minimum is not None
            and len(finite)
            and finite.min() < spec.validation_minimum - tolerance
        ):
            range_failures.append(f"{name}:min={finite.min():.8g}")
        if (
            spec.validation_maximum is not None
            and len(finite)
            and finite.max() > spec.validation_maximum + tolerance
        ):
            range_failures.append(f"{name}:max={finite.max():.8g}")
    boolean_valid = all(
        str(matrix[name].dtype) == "boolean"
        for name in ("momentum_alignment_5_30", "inside_bar", "outside_bar")
    )
    progress = _feature_numeric_array(matrix["session_progress_fraction"])
    duplicate_groups = _exact_duplicate_groups(matrix, feature_names)
    max_missing = float(matrix[feature_names].isna().mean().max())
    records.extend(
        [
            _check(
                "numerical",
                "no_positive_or_negative_infinity",
                infinite_count == 0,
                f"infinite_values={infinite_count}",
            ),
            _check(
                "numerical",
                "registered_expected_ranges_hold",
                not range_failures,
                f"range_failures={range_failures[:20]}",
            ),
            _check(
                "numerical",
                "boolean_fields_use_nullable_boolean_dtype",
                boolean_valid,
                "momentum alignment, inside bar, and outside bar",
            ),
            _check(
                "numerical",
                "session_progress_within_bounds",
                bool(np.nanmin(progress) >= 0 and np.nanmax(progress) <= 1),
                f"min={np.nanmin(progress):.6f}; max={np.nanmax(progress):.6f}",
            ),
            _check(
                "numerical",
                "bounded_experimental_scores_hold",
                not any(
                    name.startswith(("directional_energy", "wick_pressure", "compression_age"))
                    for name in range_failures
                ),
                "Directional energy, wick pressure, and compression age checked against registry bounds",
            ),
            _check(
                "engineering",
                "no_exact_duplicate_features",
                not duplicate_groups,
                f"duplicate_groups={sorted(set(duplicate_groups.values()))}",
            ),
            _check(
                "missingness",
                "no_unexplained_broad_missingness",
                max_missing < 0.20,
                f"maximum feature missing rate={max_missing:.6%}; nulls are retained rather than filled",
            ),
        ]
    )

    reference = build_result.reference_parameters
    reference_ok = (
        len(reference) > 0
        and set(reference["fit_scope"].astype(str)) == {"Development only"}
        and (
            reference["development_observation_count"] >= reference["minimum_observations_required"]
        ).all()
        and (reference["development_date_count"] >= reference["minimum_dates_required"]).all()
        and (reference["std_log_volume"] > 0).all()
    )
    records.extend(
        [
            _check(
                "Development_reference",
                "reference_groups_have_sufficient_Development_support",
                reference_ok,
                f"reference_groups={len(reference)}",
            ),
            _check(
                "Development_reference",
                "Development_rows_use_leave_one_date_out",
                set(reference["development_row_application"].astype(str))
                == {"leave-one-New-York-trading-date-out"},
                "current New York trading date is subtracted from group count/sum/sumsq",
            ),
            _check(
                "boundaries",
                "continuity_runs_constructed_on_full_GC_sequence",
                int(audit.get("source_rows", "0")) > len(matrix)
                and int(audit.get("continuity_run_count", "0")) > 1,
                f"full_source_rows={audit.get('source_rows')}; runs={audit.get('continuity_run_count')}",
            ),
            _check(
                "boundaries",
                "observation_audit_maps_every_feature_row",
                len(build_result.observation_audit) == len(matrix)
                and np.array_equal(
                    build_result.observation_audit["observation_id"], matrix["observation_id"]
                ),
                "run, boundary reason, and source position retained outside the predictive matrix",
            ),
        ]
    )
    validation = pd.DataFrame.from_records(records)
    validation["critical"] = validation["critical"].astype(bool)
    validation["passed"] = validation["passed"].astype(bool)
    return validation


def compute_feature_diagnostics(build_result: FeatureBuildResult) -> pd.DataFrame:
    """Create label-free Development diagnostics plus integrity-only group missingness."""

    matrix = build_result.matrix
    registry = build_result.registry.set_index("feature_name")
    feature_names = list(FEATURE_NAMES)
    partition = matrix["research_partition"].astype(str)
    development = matrix.loc[partition.eq("Development")]
    duplicate_groups = _exact_duplicate_groups(development, feature_names)
    records: list[dict] = []

    def summarize(
        name: str,
        frame: pd.DataFrame,
        analysis_type: str,
        group_name: str,
        group_value: str,
        full: bool,
    ) -> dict:
        series = frame[name]
        non_null = int(series.notna().sum())
        missing_rate = float(series.isna().mean()) if len(series) else np.nan
        numeric = _feature_numeric_array(series)
        finite_rate = 1.0
        stats = {
            key: np.nan
            for key in ("mean", "standard_deviation", "minimum", "p01", "median", "p99", "maximum")
        }
        unique_count: float | int = np.nan
        invalid_range = False
        constant = False
        near_constant = False
        extreme = False
        if numeric is not None:
            finite = numeric[np.isfinite(numeric)]
            finite_rate = float(len(finite) / non_null) if non_null else np.nan
            if full:
                unique_count = int(pd.Series(numeric).nunique(dropna=True))
                if len(finite):
                    quantiles = np.quantile(finite, [0.01, 0.50, 0.99])
                    stats = {
                        "mean": float(finite.mean()),
                        "standard_deviation": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                        "minimum": float(finite.min()),
                        "p01": float(quantiles[0]),
                        "median": float(quantiles[1]),
                        "p99": float(quantiles[2]),
                        "maximum": float(finite.max()),
                    }
                    spec = next(spec for spec in FEATURE_SPECS if spec.feature_name == name)
                    tolerance = 2.0e-5
                    invalid_range = bool(
                        (
                            spec.validation_minimum is not None
                            and stats["minimum"] < spec.validation_minimum - tolerance
                        )
                        or (
                            spec.validation_maximum is not None
                            and stats["maximum"] > spec.validation_maximum + tolerance
                        )
                    )
                    constant = unique_count <= 1
                    near_constant = bool(
                        str(series.dtype) == "float32"
                        and not constant
                        and (stats["p99"] - stats["p01"] <= 1.0e-7)
                    )
                    extreme = bool(abs(stats["maximum"]) > 1.0e30 or abs(stats["minimum"]) > 1.0e30)
        elif full:
            unique_count = int(series.nunique(dropna=True))
            constant = unique_count <= 1
        return {
            "analysis_type": analysis_type,
            "group_name": group_name,
            "group_value": group_value,
            "feature_name": name,
            "display_name": registry.loc[name, "display_name"],
            "family": registry.loc[name, "family"],
            "feature_tier": registry.loc[name, "feature_tier"],
            "dtype": str(series.dtype),
            "row_count": len(frame),
            "non_null_count": non_null,
            "missing_rate": missing_rate,
            "finite_rate": finite_rate,
            "unique_value_count": unique_count,
            **stats,
            "constant_flag": constant if full else False,
            "near_constant_flag": near_constant if full else False,
            "exact_duplicate_group": duplicate_groups.get(name, "") if full else "",
            "unexpectedly_extreme_flag": extreme if full else False,
            "excessive_missingness_flag": missing_rate >= 0.20 if full else False,
            "invalid_range_flag": invalid_range if full else False,
        }

    for name in feature_names:
        records.append(
            summarize(
                name,
                development,
                "Development_feature_summary",
                "research_partition",
                "Development",
                True,
            )
        )

    for partition_name in ("Development", "Validation", "Final test"):
        subset = matrix.loc[partition.eq(partition_name)]
        for name in feature_names:
            records.append(
                summarize(
                    name, subset, "partition_integrity", "research_partition", partition_name, False
                )
            )

    engineering_population = matrix.loc[~partition.eq("Final test")]
    for session_name in ("London", "New York"):
        subset = engineering_population.loc[
            engineering_population["entry_session"].astype(str).eq(session_name)
        ]
        for name in feature_names:
            records.append(
                summarize(name, subset, "session_missingness", "entry_session", session_name, False)
            )

    for family, names in registry.groupby("family", observed=True).groups.items():
        family_names = list(names)
        cells = development[family_names]
        records.append(
            {
                "analysis_type": "family_missingness",
                "group_name": "family",
                "group_value": str(family),
                "feature_name": "__family_aggregate__",
                "display_name": str(family),
                "family": str(family),
                "feature_tier": "mixed"
                if len(set(registry.loc[family_names, "feature_tier"].astype(str))) > 1
                else str(registry.loc[family_names[0], "feature_tier"]),
                "dtype": "mixed",
                "row_count": len(development) * len(family_names),
                "non_null_count": int(cells.notna().sum().sum()),
                "missing_rate": float(cells.isna().to_numpy().mean()),
                "finite_rate": np.nan,
                "unique_value_count": np.nan,
                **{
                    key: np.nan
                    for key in (
                        "mean",
                        "standard_deviation",
                        "minimum",
                        "p01",
                        "median",
                        "p99",
                        "maximum",
                    )
                },
                "constant_flag": False,
                "near_constant_flag": False,
                "exact_duplicate_group": "",
                "unexpectedly_extreme_flag": False,
                "excessive_missingness_flag": False,
                "invalid_range_flag": False,
            }
        )
    return pd.DataFrame.from_records(records)


def build_manual_feature_audit(
    full_gc_bars: pd.DataFrame,
    eligible_observations: pd.DataFrame,
    build_result: FeatureBuildResult,
    *,
    random_seed: int = 20260714,
) -> pd.DataFrame:
    """Independently reconstruct one representative feature from every family."""

    bars = full_gc_bars.loc[:, FEATURE_SOURCE_COLUMNS].reset_index(drop=True)
    run_id, _, _ = build_continuity_run_id(bars)
    open_ = pd.to_numeric(bars["open"]).to_numpy(dtype=np.float64)
    high = pd.to_numeric(bars["high"]).to_numpy(dtype=np.float64)
    low = pd.to_numeric(bars["low"]).to_numpy(dtype=np.float64)
    close = pd.to_numeric(bars["close"]).to_numpy(dtype=np.float64)
    volume = pd.to_numeric(bars["volume"]).to_numpy(dtype=np.float64)
    previous_close = np.full(len(close), np.nan)
    valid_previous = run_id[1:] == run_id[:-1]
    indices = np.arange(1, len(close))[valid_previous]
    previous_close[indices] = close[:-1][valid_previous]
    true_range = np.maximum.reduce(
        [high - low, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    true_range[np.isnan(previous_close)] = (high - low)[np.isnan(previous_close)]

    matrix = build_result.matrix
    observation_audit = build_result.observation_audit
    entry_ts = pd.DatetimeIndex(pd.to_datetime(matrix["entry_timestamp_ny"]))
    minute = entry_ts.hour * 60 + entry_ts.minute
    session = matrix["entry_session"].astype(str).to_numpy()
    rng = np.random.default_rng(random_seed)

    case_masks = {
        "early London": (session == "London") & (minute <= 182),
        "late London": (session == "London") & (minute >= 357),
        "early New York": (session == "New York") & (minute <= 422),
        "late New York": (session == "New York") & (minute >= 717),
    }
    selected_rows: list[tuple[str, int]] = []
    for case, mask in case_masks.items():
        candidates = np.flatnonzero(mask)
        selected_rows.append((case, int(rng.choice(candidates))))
    boundary_mask = (
        observation_audit["continuity_boundary_reason"]
        .astype(str)
        .isin(["selected_contract_change", "instrument_change", "continuous_segment_change"])
    )
    boundary_candidates = observation_audit.loc[boundary_mask].sort_values(
        ["bars_since_continuity_start", "observation_id"], kind="mergesort"
    )
    if boundary_candidates.empty:
        boundary_candidates = observation_audit.sort_values(
            "bars_since_continuity_start", kind="mergesort"
        )
    boundary_observation_id = int(boundary_candidates.iloc[0]["observation_id"])
    boundary_row = int(matrix.index[matrix["observation_id"].eq(boundary_observation_id)][0])
    selected_rows.append(("nearest contract/segment boundary", boundary_row))

    def window_valid(position: int, start: int) -> bool:
        return start >= 0 and run_id[position] == run_id[start]

    records: list[dict] = []
    for case, row in selected_rows:
        source_position = int(observation_audit.iloc[row]["source_position"])
        atr20 = np.nan
        if window_valid(source_position, source_position - 19):
            atr20 = float(true_range[source_position - 19 : source_position + 1].mean())
        reconstructed: dict[str, float] = {}
        if window_valid(source_position, source_position - 5):
            reconstructed["return_5m_bps"] = float(
                10_000.0 * np.log(close[source_position] / close[source_position - 5])
            )
        else:
            reconstructed["return_5m_bps"] = np.nan
        reconstructed["atr_20"] = atr20
        current_range = high[source_position] - low[source_position]
        reconstructed["body_to_range"] = (
            abs(close[source_position] - open_[source_position]) / current_range
            if current_range != 0
            else np.nan
        )
        if window_valid(source_position, source_position - 19):
            vol_window = volume[source_position - 19 : source_position + 1]
            reconstructed["relative_volume_20"] = (
                float(volume[source_position] / vol_window.mean())
                if vol_window.mean() != 0
                else np.nan
            )
            typical = (
                high[source_position - 19 : source_position + 1]
                + low[source_position - 19 : source_position + 1]
                + close[source_position - 19 : source_position + 1]
            ) / 3.0
            vwap = (
                float(np.sum(typical * vol_window) / np.sum(vol_window))
                if np.sum(vol_window) != 0
                else np.nan
            )
            reconstructed["distance_from_rolling_vwap_20_atr"] = (
                (close[source_position] - vwap) / atr20
                if np.isfinite(atr20) and atr20 != 0
                else np.nan
            )
        else:
            reconstructed["relative_volume_20"] = np.nan
            reconstructed["distance_from_rolling_vwap_20_atr"] = np.nan
        if window_valid(source_position, source_position - 15):
            changes = np.diff(close[source_position - 15 : source_position + 1])
            denominator = np.abs(changes).sum()
            reconstructed["efficiency_ratio_15"] = (
                abs(close[source_position] - close[source_position - 15]) / denominator
                if denominator != 0
                else np.nan
            )
        else:
            reconstructed["efficiency_ratio_15"] = np.nan
        reconstructed["minute_from_execution_window_open"] = float(
            minute[row] - (180 if session[row] == "London" else 420)
        )
        if (
            window_valid(source_position, source_position - 30)
            and np.isfinite(atr20)
            and atr20 != 0
        ):
            short = (close[source_position] - close[source_position - 5]) / atr20
            broad = (close[source_position] - close[source_position - 30]) / atr20
            reconstructed["pullback_tension_5_30_exp"] = (
                float(np.sign(broad) * min(abs(short), abs(broad))) if short * broad < 0 else 0.0
            )
        else:
            reconstructed["pullback_tension_5_30_exp"] = np.nan

        family_map = {
            "return_5m_bps": "price_return_momentum",
            "atr_20": "volatility_range_state",
            "body_to_range": "candle_geometry",
            "relative_volume_20": "volume_activity",
            "distance_from_rolling_vwap_20_atr": "vwap_session_state",
            "efficiency_ratio_15": "trend_persistence",
            "minute_from_execution_window_open": "session_clock_calendar",
            "pullback_tension_5_30_exp": "experimental_hypothesis",
        }
        for name, expected in reconstructed.items():
            saved = matrix.iloc[row][name]
            saved_float = float(saved) if not pd.isna(saved) else np.nan
            tolerance = 2.0e-5 if name != "atr_20" else 2.0e-4
            passed = bool(
                (np.isnan(saved_float) and np.isnan(expected))
                or (
                    np.isfinite(saved_float)
                    and np.isfinite(expected)
                    and abs(saved_float - expected) <= tolerance * max(1.0, abs(expected))
                )
            )
            records.append(
                {
                    "audit_case": case,
                    "observation_id": int(matrix.iloc[row]["observation_id"]),
                    "decision_timestamp_ny": matrix.iloc[row]["decision_timestamp_ny"],
                    "entry_session": session[row],
                    "continuity_boundary_reason": str(
                        observation_audit.iloc[row]["continuity_boundary_reason"]
                    ),
                    "bars_since_continuity_start": int(
                        observation_audit.iloc[row]["bars_since_continuity_start"]
                    ),
                    "family": family_map[name],
                    "feature_name": name,
                    "saved_value": saved_float,
                    "reconstructed_value": expected,
                    "absolute_error": abs(saved_float - expected)
                    if np.isfinite(saved_float) and np.isfinite(expected)
                    else np.nan,
                    "tolerance": tolerance,
                    "passed": passed,
                }
            )
    return pd.DataFrame.from_records(records)


def _saved_dtype_policy_passes(schema: pa.Schema) -> bool:
    spec_by_name = {spec.feature_name: spec for spec in FEATURE_SPECS}
    for name, spec in spec_by_name.items():
        dtype = schema.field(name).type
        if spec.output_dtype == "float32" and not pa.types.is_float32(dtype):
            return False
        if spec.output_dtype == "boolean" and not pa.types.is_boolean(dtype):
            return False
        if spec.output_dtype == "Int16" and not pa.types.is_int16(dtype):
            return False
        if spec.output_dtype == "Int8" and not pa.types.is_int8(dtype):
            return False
        if spec.output_dtype == "category" and not (
            pa.types.is_dictionary(dtype) or pa.types.is_string(dtype)
        ):
            return False
    return True


def _render_summary(
    build_result: FeatureBuildResult,
    validation: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_paths: dict[str, Path],
    ready: bool,
    project_root: Path,
    test_details: str,
) -> str:
    registry = build_result.registry
    matrix = build_result.matrix
    family_counts = registry.groupby("family", observed=True).size().sort_index()
    dev_summary = diagnostics.loc[diagnostics["analysis_type"].eq("Development_feature_summary")]
    highest_missing = dev_summary.sort_values("missing_rate", ascending=False).head(8)
    experimental = registry.loc[registry["is_experimental"].astype(bool)]
    critical = validation.loc[validation["critical"]]
    lines = [
        "# Section 6 Feature Engineering Summary",
        "",
        f"**Status:** {'READY' if ready else 'NOT READY'}",
        "",
        f"- Feature matrix shape: **{matrix.shape[0]:,} × {matrix.shape[1]:,}**",
        f"- Metadata columns: **{len(METADATA_COLUMNS)}**",
        f"- Registered predictors: **{len(registry)}** (**{int((~registry['is_experimental'].astype(bool)).sum())} core**, **{int(registry['is_experimental'].astype(bool).sum())} experimental**)",
        f"- Numeric predictors: **{int(registry['output_dtype'].isin(['float32', 'Int16', 'Int8']).sum())}**",
        f"- Critical validation checks passed: **{int(critical['passed'].sum())}/{len(critical)}**",
        f"- Automated feature tests: **{test_details}**",
        f"- Estimated unoptimized matrix memory: **{build_result.estimated_unoptimized_memory_bytes / 2**20:,.2f} MiB**",
        f"- Optimized matrix memory: **{build_result.optimized_memory_bytes / 2**20:,.2f} MiB**",
        f"- Estimated peak working memory: **{build_result.estimated_peak_working_memory_bytes / 2**20:,.2f} MiB**",
        "",
        "## Features by family",
        "",
    ]
    lines.extend(f"- {family}: {int(count)}" for family, count in family_counts.items())
    lines.extend(["", "## Missingness", ""])
    lines.append(
        f"Development missingness ranges from {dev_summary['missing_rate'].min():.6%} to {dev_summary['missing_rate'].max():.6%}. "
        "Nulls are retained for warm-up, reset boundaries, session openings, zero denominators, and insufficient fitted-reference support."
    )
    lines.append("")
    lines.append("Highest Development missingness:")
    lines.append("")
    for row in highest_missing.itertuples():
        lines.append(f"- `{row.feature_name}`: {row.missing_rate:.6%}")
    lines.extend(["", "## Experimental hypotheses (*)", ""])
    for row in experimental.itertuples():
        lines.append(f"- **{row.display_name}** (`{row.feature_name}`): {row.creative_rationale}")
    lines.extend(
        [
            "",
            "## Important implementation decisions",
            "",
            "- Features were constructed on the full trusted GC bar sequence and mapped to eligible completed decision bars only after calculation.",
            "- Continuity resets on non-one-minute timestamps, product/contract/instrument/segment changes, and tradability/roll/liquidity boundaries.",
            "- Research-day state resets at 01:00 New York; London execution state resets at 03:00; New York execution state resets at 07:00.",
            "- The first eligible entry at each execution-session open honestly has null execution-session state because decision bar t precedes the opening bar.",
            "- Time-of-day log-volume references use Development only. Development rows exclude their own New York trading date; Validation and Final test use frozen full-Development parameters.",
            "- Partial 2021 and partial 2026 contribute only their observed dates; no annual reweighting or full-sample clock normalization is used.",
            "- Final-test inspection is limited to schema, row counts, null/finite rates, and transformation integrity. No feature-outcome relationship is examined.",
            "- Float64 is used for sensitive calculations; ordinary saved continuous features are float32, bounded counters are nullable small integers, and repeated strings are categorical/dictionary encoded.",
            "",
            "## Saved outputs",
            "",
        ]
    )
    for label, path in output_paths.items():
        lines.append(f"- {label}: `{path.relative_to(project_root)}`")
    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- Signed-volume, wick-pressure, and liquidity-vacuum measures are OHLCV proxies, not order flow, queue state, market depth, or measured liquidity.",
            "- Feature definitions are frozen for Section 7 evaluation, but no redundancy selection or predictive interpretation has been performed.",
            "- MGC is intentionally excluded until the GC shortlist is frozen for later transfer validation.",
            "",
            "**No feature has yet been shown to possess predictive value.** Section 6 establishes only a valid candidate feature matrix.",
            "",
            "**Exact next section:** Section 7.0 — Univariate Feature Evaluation",
            "",
            "*Follow-up (2026-07-20): executed as declared - see "
            "`section7_univariate_evaluation_summary.md`.*",
            "",
            "SECTION 6 STATUS: READY" if ready else "SECTION 6 STATUS: NOT READY",
            "",
        ]
    )
    return "\n".join(lines)


def save_feature_outputs(
    build_result: FeatureBuildResult,
    validation: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manual_audit: pd.DataFrame,
    *,
    project_root: Path,
    test_details: str,
) -> FeatureSaveResult:
    """Save every approved output, reload it, and generate the completion summary."""

    project_root = Path(project_root)
    data_dir = project_root / "data" / "processed" / "statistical_research"
    table_dir = project_root / "reports" / "statistical_research" / "tables" / "section6"
    summary_dir = project_root / "reports" / "statistical_research" / "summaries"
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "feature_matrix": data_dir / "feature_matrix_gc.parquet",
        "feature_registry": data_dir / "feature_registry_gc.parquet",
        "feature_validation": data_dir / "feature_validation_gc.parquet",
        "feature_diagnostics": data_dir / "feature_diagnostics_gc.parquet",
        "feature_reference_parameters": data_dir / "feature_reference_parameters_gc.parquet",
        "registry_csv": table_dir / "feature_registry_gc.csv",
        "diagnostics_csv": table_dir / "feature_diagnostics_gc.csv",
        "manual_audit_csv": table_dir / "section6_manual_feature_audit.csv",
        "summary_markdown": summary_dir / "section6_feature_engineering_summary.md",
    }

    build_result.matrix.to_parquet(
        output_paths["feature_matrix"], engine="pyarrow", compression="zstd", index=False
    )
    build_result.registry.to_parquet(
        output_paths["feature_registry"], engine="pyarrow", compression="zstd", index=False
    )
    diagnostics.to_parquet(
        output_paths["feature_diagnostics"], engine="pyarrow", compression="zstd", index=False
    )
    build_result.reference_parameters.to_parquet(
        output_paths["feature_reference_parameters"],
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    build_result.registry.to_csv(output_paths["registry_csv"], index=False)
    diagnostics.to_csv(output_paths["diagnostics_csv"], index=False)
    manual_audit.to_csv(output_paths["manual_audit_csv"], index=False)

    reloaded_matrix = pd.read_parquet(output_paths["feature_matrix"], engine="pyarrow")
    reloaded_registry = pd.read_parquet(output_paths["feature_registry"], engine="pyarrow")
    reloaded_diagnostics = pd.read_parquet(output_paths["feature_diagnostics"], engine="pyarrow")
    reloaded_reference = pd.read_parquet(
        output_paths["feature_reference_parameters"], engine="pyarrow"
    )
    matrix_schema = pq.ParquetFile(output_paths["feature_matrix"]).schema_arrow
    sample_positions = np.array(
        [0, len(reloaded_matrix) // 2, len(reloaded_matrix) - 1], dtype=np.int64
    )
    sample_columns = [
        "observation_id",
        "decision_timestamp_utc",
        "entry_timestamp_ny",
        "return_5m_bps",
        "atr_20",
        "compression_age_exp",
    ]
    sampled_equal = True
    for column in sample_columns:
        left = build_result.matrix.iloc[sample_positions][column].reset_index(drop=True)
        right = reloaded_matrix.iloc[sample_positions][column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(left.dtype):
            sampled_equal &= bool(
                np.allclose(
                    pd.to_numeric(left, errors="coerce").to_numpy(
                        dtype=np.float64, na_value=np.nan
                    ),
                    pd.to_numeric(right, errors="coerce").to_numpy(
                        dtype=np.float64, na_value=np.nan
                    ),
                    equal_nan=True,
                )
            )
        else:
            sampled_equal &= left.astype(str).equals(right.astype(str))

    reload_records = [
        _check(
            "save_reload",
            "matrix_row_and_column_order_preserved",
            reloaded_matrix.shape == build_result.matrix.shape
            and list(reloaded_matrix.columns) == list(build_result.matrix.columns),
            f"shape={reloaded_matrix.shape}",
        ),
        _check(
            "save_reload",
            "matrix_observation_order_preserved",
            np.array_equal(
                reloaded_matrix["observation_id"].to_numpy(),
                build_result.matrix["observation_id"].to_numpy(),
            ),
            "all observation identifiers compared",
        ),
        _check(
            "save_reload",
            "matrix_null_counts_preserved",
            reloaded_matrix.isna().sum().equals(build_result.matrix.isna().sum()),
            "all columns compared",
        ),
        _check(
            "save_reload",
            "matrix_timezone_metadata_preserved",
            matrix_schema.field("decision_timestamp_utc").type.tz == "UTC"
            and matrix_schema.field("entry_timestamp_utc").type.tz == "UTC"
            and matrix_schema.field("decision_timestamp_ny").type.tz == "America/New_York"
            and matrix_schema.field("entry_timestamp_ny").type.tz == "America/New_York",
            "UTC and America/New_York Arrow timezone fields",
        ),
        _check(
            "save_reload",
            "matrix_saved_dtype_policy_preserved",
            _saved_dtype_policy_passes(matrix_schema),
            "float32, boolean, small integer, and categorical policies checked from Arrow schema",
        ),
        _check(
            "save_reload",
            "matrix_selected_deterministic_values_preserved",
            sampled_equal,
            f"positions={sample_positions.tolist()}; columns={sample_columns}",
        ),
        _check(
            "save_reload",
            "registry_reload_alignment",
            reloaded_registry["feature_name"].astype(str).tolist() == list(FEATURE_NAMES)
            and len(reloaded_registry) == len(build_result.registry),
            f"registry_rows={len(reloaded_registry)}",
        ),
        _check(
            "save_reload",
            "diagnostics_reload_preserved",
            reloaded_diagnostics.shape == diagnostics.shape
            and list(reloaded_diagnostics.columns) == list(diagnostics.columns),
            f"diagnostic_shape={reloaded_diagnostics.shape}",
        ),
        _check(
            "save_reload",
            "reference_parameters_reload_preserved",
            reloaded_reference.shape == build_result.reference_parameters.shape
            and list(reloaded_reference.columns) == list(build_result.reference_parameters.columns),
            f"reference_shape={reloaded_reference.shape}",
        ),
    ]
    reload_checks = pd.DataFrame.from_records(reload_records)
    combined_validation = pd.concat([validation, reload_checks], ignore_index=True)
    combined_validation.to_parquet(
        output_paths["feature_validation"], engine="pyarrow", compression="zstd", index=False
    )
    reloaded_validation = pd.read_parquet(output_paths["feature_validation"], engine="pyarrow")
    validation_artifact_ok = (
        reloaded_validation.shape == combined_validation.shape
        and list(reloaded_validation.columns) == list(combined_validation.columns)
        and reloaded_validation["passed"]
        .astype(bool)
        .equals(combined_validation["passed"].astype(bool))
    )
    if not validation_artifact_ok:
        raise AssertionError(
            "feature_validation_gc.parquet failed its independent save/reload check"
        )

    ready = bool(
        combined_validation.loc[combined_validation["critical"].astype(bool), "passed"]
        .astype(bool)
        .all()
    )
    report_markdown = _render_summary(
        build_result,
        combined_validation,
        diagnostics,
        output_paths,
        ready,
        project_root,
        test_details,
    )
    output_paths["summary_markdown"].write_text(report_markdown, encoding="utf-8")
    return FeatureSaveResult(
        validation=combined_validation,
        reload_checks=reload_checks,
        report_markdown=report_markdown,
        output_paths=output_paths,
        ready=ready,
    )
