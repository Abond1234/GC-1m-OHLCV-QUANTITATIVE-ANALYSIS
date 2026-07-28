"""Frozen contract and provenance helpers for the Project 1 FES study.

This module contains only deterministic, outcome-free configuration used by
Notebook Section 1. Feature construction begins in Notebook Section 2 and is
deliberately absent here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

RESEARCH_ID = "fes_project1"
ARTIFACT_VERSION = "v1"
DETERMINISTIC_SEED = 20260726
VALIDATION_BOOTSTRAP_SEED = 20260727
SPECIFICATION_PATH = "project_docs/feature_engineering_selection_project1_agent_prompt.md"
SPECIFICATION_SHA256 = "c7c87fe613788ecb4cbde8bb7973def9014b7572ebb9d5351d714f004342eb3a"
NOTEBOOK_PATH = "notebooks/exploration/feature_research_1.ipynb"
DATA_NAMESPACE = "data/processed/statistical_research/fes_project1/v1"
REPORT_NAMESPACE = "reports/statistical_research/fes_project1/v1"

TRUSTED_INPUT_PATHS = (
    "data/processed/research_bars_gc_mgc_1m.parquet",
    "data/processed/statistical_research/eligible_observations_gc.parquet",
    "data/processed/statistical_research/forward_labels_gc.parquet",
    "data/processed/statistical_research/feature_matrix_gc.parquet",
    "data/processed/statistical_research/feature_registry_gc.parquet",
    "data/processed/statistical_research/frozen_expansion_feature_set_gc.parquet",
)

PROVENANCE_INPUT_PATHS = (
    "AGENTS.md",
    SPECIFICATION_PATH,
    *TRUSTED_INPUT_PATHS,
    "src/statistical_research/sequential_backtest.py",
)

PLANNED_ADDITIVE_FILES = (
    NOTEBOOK_PATH,
    "src/statistical_research/fes_project1_config.py",
    "src/statistical_research/fes_project1_features.py",
    "src/statistical_research/fes_project1_modeling.py",
    "src/statistical_research/fes_project1_evaluation.py",
    "src/execution/fes_project1_mgc_adapter.py",
    "scripts/run_fes_project1_research.py",
    "tests/test_fes_project1_features.py",
    "tests/test_fes_project1_modeling.py",
    "tests/test_fes_project1_evaluation.py",
    "tests/test_fes_project1_mgc_adapter.py",
)

FEATURE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "F01",
        "name": "ret_trimmed_mean_30_bps",
        "history": "30 returns / 31 closes",
        "role": "directional",
        "range": "unbounded",
        "formula": "mean(sort(r[t-29:t]) after removing three values from each tail)",
    },
    {
        "id": "F02",
        "name": "ret_mad_30_bps",
        "history": "30 returns / 31 closes",
        "role": "opportunity_risk",
        "range": "[0,+inf)",
        "formula": "1.4826 * median(abs(r - median(r))); all-equal window -> 0",
    },
    {
        "id": "F03",
        "name": "ret_tail_balance_60",
        "history": "60 returns / 61 closes",
        "role": "directional",
        "range": "[-1,1]",
        "formula": "(abs(q90)-abs(q10))/(abs(q90)+abs(q10)); zero denominator -> 0",
    },
    {
        "id": "F04",
        "name": "ret_outlier_fraction_60",
        "history": "60 returns / 61 closes",
        "role": "opportunity_risk",
        "range": "[0,1]",
        "formula": "mean(abs(r-median(r)) > 3*(1.4826*MAD)); zero scale -> 0",
    },
    {
        "id": "F05",
        "name": "price_path_curvature_30_atr",
        "history": "31 closes and decision ATR20",
        "role": "directional_path_shape",
        "range": "unbounded",
        "formula": "quadratic beta2 from ATR-normalized close path on x in [-1,1]",
    },
    {
        "id": "F06",
        "name": "ordered_draw_balance_30",
        "history": "31 closes",
        "role": "directional_path_shape",
        "range": "[-1,1]",
        "formula": "(max_ordered_runup-max_ordered_drawdown)/(sum); zero sum -> 0",
    },
    {
        "id": "F07",
        "name": "return_acf_energy_60",
        "history": "60 returns / 61 closes",
        "role": "opportunity_regime",
        "range": "[0,1]",
        "formula": "sqrt(mean(rho_k**2 for Pearson ACF lags k=1..5))",
    },
    {
        "id": "F08",
        "name": "lagged_volume_return_spearman_30",
        "history": "30 lagged ZV/return pairs",
        "role": "directional_participation_lead",
        "range": "[-1,1]",
        "formula": "Spearman(ZV[i-1], r[i]) for i=t-29..t",
    },
    {
        "id": "F09",
        "name": "range_volume_spearman_30",
        "history": "30 complete bars",
        "role": "opportunity_liquidity_state",
        "range": "[-1,1]",
        "formula": "Spearman(TR[i]/ATR20[i], ZV[i]) for i=t-29..t",
    },
    {
        "id": "F10",
        "name": "volume_profile_slope_30",
        "history": "30 ZV values",
        "role": "opportunity_participation_regime",
        "range": "unbounded",
        "formula": "OLS beta for ZV profile on x in [-1,1]",
    },
)

INTERACTION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "H01",
        "name": "relative_volume_20_clipped",
        "formula": "clip(relative_volume_20, 0.5, 2.0)",
        "standalone_screen": False,
    },
    {
        "id": "I01",
        "name": "curvature_coherence_30",
        "formula": "price_path_curvature_30_atr * efficiency_ratio_30",
        "parents": ["price_path_curvature_30_atr", "efficiency_ratio_30"],
    },
    {
        "id": "I02",
        "name": "tail_pressure_activity_60",
        "formula": "ret_tail_balance_60 * relative_volume_20_clipped",
        "parents": ["ret_tail_balance_60", "relative_volume_20_clipped"],
    },
    {
        "id": "I03",
        "name": "lagged_volume_confirmation_30",
        "formula": "lagged_volume_return_spearman_30 * relative_volume_20_clipped",
        "parents": [
            "lagged_volume_return_spearman_30",
            "relative_volume_20_clipped",
        ],
    },
    {
        "id": "I04",
        "name": "vwap_trend_alignment_30",
        "formula": (
            "distance_from_research_day_vwap_atr * normalized_ols_slope_30"
        ),
        "parents": [
            "distance_from_research_day_vwap_atr",
            "normalized_ols_slope_30",
        ],
    },
)

DIRECTIONAL_FEATURES = (
    "ret_trimmed_mean_30_bps",
    "ret_tail_balance_60",
    "price_path_curvature_30_atr",
    "ordered_draw_balance_30",
    "lagged_volume_return_spearman_30",
)

OPPORTUNITY_FEATURES = (
    "ret_mad_30_bps",
    "ret_outlier_fraction_60",
    "return_acf_energy_60",
    "range_volume_spearman_30",
    "volume_profile_slope_30",
)

D1_FEATURES = (
    "return_5m_atr",
    "return_15m_atr",
    "return_30m_atr",
    "normalized_ols_slope_30",
    "close_location_value",
    "signed_volume_proxy",
    "distance_from_research_day_vwap_atr",
    "return_autocorrelation_15",
)

O1_FEATURES = (
    "atr_20",
    "atr_ratio_20_60",
    "atr_ratio_5_20",
    "current_range_over_atr",
    "minute_from_execution_window_open",
    "minutes_to_1530_forced_exit",
    "efficiency_ratio_15",
    "efficiency_ratio_30",
    "efficiency_ratio_60",
    "ols_r_squared_30",
    "choppiness_14",
    "return_sign_change_rate_30",
    "relative_volume_20",
    "volume_acceleration_5_20",
    "vwap_elasticity_30_exp",
)


def canonical_json(value: Mapping[str, Any] | list[Any]) -> str:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Return the hexadecimal SHA-256 digest for bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a file into a SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_frozen_config() -> dict[str, Any]:
    """Return the complete deterministic v1 research contract."""

    feature_names = [item["name"] for item in FEATURE_SPECS]
    interaction_names = [
        item["name"] for item in INTERACTION_SPECS if item["id"].startswith("I")
    ]
    return {
        "schema_version": "1.0.0",
        "research_id": RESEARCH_ID,
        "artifact_version": ARTIFACT_VERSION,
        "specification": {
            "path": SPECIFICATION_PATH,
            "sha256": SPECIFICATION_SHA256,
            "implementation_mode": "frozen_specification",
            "book_read_prohibited": True,
        },
        "notebook": {
            "path": NOTEBOOK_PATH,
            "filename_override": {
                "prompt_default": (
                    "notebooks/exploration/"
                    "feature_engineering_selection_project1.ipynb"
                ),
                "user_requested": NOTEBOOK_PATH,
                "reason": "explicit user filename instruction",
            },
            "section_boundary": 1,
            "section_2_started": False,
        },
        "paths": {
            "trusted_inputs": list(TRUSTED_INPUT_PATHS),
            "data_namespace": DATA_NAMESPACE,
            "report_namespace": REPORT_NAMESPACE,
            "planned_additive_files": list(PLANNED_ADDITIVE_FILES),
            "protected_notebooks": [
                "notebooks/exploration/exp1.ipynb",
                "notebooks/exploration/statistical_feature_research.ipynb",
                "notebooks/exploration/exp1 appendix.ipynb",
            ],
            "unverified_mlat_namespace": (
                "data/processed/statistical_research/mlat_feature_research/v1"
            ),
        },
        "randomness": {
            "primary_seed": DETERMINISTIC_SEED,
            "development_bootstrap_seed": DETERMINISTIC_SEED,
            "validation_bootstrap_seed": VALIDATION_BOOTSTRAP_SEED,
            "date_block_bootstrap_replicates": 2000,
            "confidence": 0.95,
        },
        "evidence_boundary": {
            "development": "through 2023",
            "retrospective_validation": "2024; one locked read in Section 6",
            "historical_final": "2025 through approximately 2026-05-22; unread by default",
            "feature_construction_validation_exception": (
                "deterministic outcome-free sealed features only"
            ),
            "sections_2_to_5_validation_outcomes_locked": True,
            "historical_final_always_locked_before_authorization": True,
            "new_forward_minimum": {"trading_days": 60, "simulated_trades": 100},
        },
        "market_contract": {
            "signal_instrument": "GC",
            "execution_instrument": "MGC",
            "bar_frequency": "1 minute",
            "timezone": "America/New_York",
            "decision_information_through": "completed GC close t",
            "earliest_entry": "true open of t+1",
            "exact_successor_minutes": 1,
            "schedule": {
                "poi_search": "[01:00,12:00]",
                "london_entry": "[03:00,06:00)",
                "no_entry_gap": "[06:00,07:00)",
                "new_york_entry": "[07:00,12:00)",
                "no_new_entry_at_or_after": "12:00",
                "mandatory_flat": "15:30",
                "overnight_positions": False,
            },
            "continuity_identity": [
                "product",
                "instrument/active symbol",
                "continuous segment",
                "New York trading date",
                "tradability/roll state",
                "exact regular one-minute successors",
            ],
            "join_key": "observation_id",
            "calculation_dtype": "float64",
            "saved_scalar_dtype": "float32",
        },
        "features": list(FEATURE_SPECS),
        "interactions": list(INTERACTION_SPECS),
        "interaction_rules": {
            "create_before_learned_preprocessing": True,
            "direction_only": True,
            "hierarchical_closure_required": True,
            "higher_order_terms": False,
            "all_pairs_search": False,
        },
        "profile": {
            "positions": 30,
            "predecessor_closes_required": 31,
            "channel_order": ["return_bps", "true_range_over_atr20", "volume_zscore_60"],
            "column_order": "[R_0..R_29,G_0..G_29,Q_0..Q_29]",
            "raw_columns": 90,
            "stored_dtype": "float32",
            "complete_only": True,
            "transforms": {
                "P0": {
                    "name": "raw_profile",
                    "columns": 90,
                    "fold_column_standardization": True,
                },
                "P1": {
                    "name": "row_robust_standardized",
                    "trim_each_tail": 3,
                    "scale_ddof": 0,
                    "zero_scale_output": 0.0,
                    "columns": 90,
                },
                "P2": {
                    "name": "first_difference_of_P1",
                    "columns": 87,
                },
                "P3": {
                    "name": "causal_trailing_median3_of_P1",
                    "columns": 84,
                    "centered_smoothing": False,
                },
            },
            "persist_full_data_components": False,
        },
        "targets": {
            "fit_separately_by_session": ["London", "New York"],
            "primary_directional": "forward_return_60_atr",
            "secondary_directional_diagnostic": "forward_return_30_atr",
            "primary_opportunity_continuous": "future_range_60_atr",
            "primary_opportunity_binary": "expansion_label_60",
            "expansion_quantile": 0.80,
            "screened_horizons": [30, 60],
            "excluded_horizons": [5, 15, 120, 180],
            "required_availability": [
                "label_available_h == True",
                "atr_normalization_available == True",
            ],
        },
        "baselines": {
            "D0": "session-specific training mean",
            "D1": list(D1_FEATURES),
            "O0": ["atr_20"],
            "O1": list(O1_FEATURES),
            "unverified_comparators": [
                "realized_semivariance_balance_60",
                "parkinson_volatility_30",
                "amihud_illiquidity_60",
            ],
        },
        "missingness": {
            "ordered_causes": [
                "SOURCE_START",
                "TIMESTAMP_GAP",
                "PRODUCT_CHANGE",
                "SELECTED_CONTRACT_CHANGE",
                "INSTRUMENT_CHANGE",
                "CONTINUOUS_SEGMENT_CHANGE",
                "TRADABILITY_ROLL_OR_LIQUIDITY_BOUNDARY",
                "NY_DATE_RESET",
                "SOURCE_INPUT_NULL",
                "NONPOSITIVE_ATR",
                "DEGENERATE_STATISTIC",
                "COMPLETE",
            ],
            "degenerate_imputation_only": [
                "return_acf_energy_60",
                "lagged_volume_return_spearman_30",
                "range_volume_spearman_30",
            ],
            "degenerate_imputer": "training-fold median",
            "degenerate_indicator": True,
            "warmup_boundary_indicators": False,
            "source_input_null_integrity_failure": True,
            "nonpositive_atr_integrity_failure": True,
        },
        "preprocessing": {
            "S0": {
                "continuous": "training-fold mean/population standard deviation",
                "constant_scale": 1.0,
            },
            "S1": {
                "power": "Yeo-Johnson, standardize=False, fold-local",
                "scale": "StandardScaler, fold-local",
                "bypass": ["categorical", "Boolean", "missing-indicator", "constant"],
            },
            "S2": {
                "implementation": "sklearn.preprocessing.SplineTransformer",
                "degree": 3,
                "n_knots": 4,
                "knots": "quantile",
                "extrapolation": "linear",
                "include_bias": False,
                "retain_linear_main_effects": True,
                "features": [
                    "ret_tail_balance_60",
                    "price_path_curvature_30_atr",
                    "ordered_draw_balance_30",
                    "return_acf_energy_60",
                    "distance_from_research_day_vwap_atr",
                    "efficiency_ratio_30",
                ],
            },
            "all_learned_operations_fold_local": True,
        },
        "support_masks": {
            "scalar_ladder": "target plus all scalar inputs for M0-M5",
            "profile_ladder": "target/anchor plus profile_complete_30",
            "M8": "frozen scalar-plus-profile support",
            "FINAL_HEAD_TO_HEAD_SUPPORT": (
                "target, anchor, all M1-M5 scalar inputs, profile_complete_30, "
                "and common OOF outer-assessment rows"
            ),
            "matched_rows_and_dates_required": True,
        },
        "resampling": {
            "unit": "New York trading date",
            "outer": {
                "initial_train_end_zero_based": 251,
                "embargo_dates": 1,
                "assessment_dates": 63,
                "step_dates": 64,
                "complete_blocks_only": True,
                "target_exit_purge": "exit_timestamp_utc_h >= first assessment decision",
            },
            "inner": {
                "initial_train_end_zero_based": 125,
                "embargo_dates": 1,
                "assessment_dates": 21,
                "step_dates": 22,
                "minimum_valid_assessment_folds": 2,
                "complete_blocks_only": True,
                "target_exit_purge": "exit_timestamp_utc_h >= first assessment decision",
            },
            "random_row_cv": False,
            "matched_folds": True,
            "fold_membership_hashed": True,
        },
        "model_ladder": {
            "M0": "null training mean/prevalence",
            "M1": "existing anchor ridge (D1; O0 and O1)",
            "M2": f"book scalar ridge ({','.join(feature_names)})",
            "M3": (
                "combined scalar ridge; direction D1+F01-F10+efficiency_ratio_30+"
                "H01+I01-I04; opportunity O1+F01-F10"
            ),
            "M4": "elastic-net support proposal plus hierarchy-closed ridge",
            "M5": "M3 plus fixed S2 spline bases, ridge",
            "M6": "P0-P3 fold-fitted PCA profile ridge",
            "M7": "P2 fold-fitted PLS regression",
            "M8": "nested combined scalar plus profile",
            "ridge_alpha_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
            "elastic_net": {
                "alpha_grid": [1e-4, 1e-3, 1e-2, 1e-1],
                "l1_ratio_grid": [0.1, 0.5, 0.9, 1.0],
                "selection_only": True,
                "max_iter": 5000,
                "tol": 1e-6,
                "selection": "cyclic",
            },
            "PCA": {
                "component_grid": [3, 5, 8, 12],
                "whiten": False,
                "svd_solver": "full",
                "deterministic_orientation": True,
            },
            "PLS": {
                "profile": "P2",
                "component_grid": [2, 3, 5, 8],
                "scale": False,
                "max_iter": 500,
                "tol": 1e-6,
                "deterministic_orientation": True,
            },
            "expansion_logistic": {
                "C_grid": [0.01, 0.1, 1.0, 10.0],
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": None,
                "max_iter": 2000,
                "tol": 1e-6,
            },
            "prohibited": [
                "trees",
                "boosting",
                "SVM",
                "neural networks",
                "autoencoders",
                "RFE",
                "stepwise",
                "MARS",
                "genetic algorithms",
                "simulated annealing",
            ],
        },
        "selection": {
            "primary_continuous_metric": "mean outer-fold daily Spearman IC",
            "inner_only_hyperparameters": True,
            "one_standard_error_rule": True,
            "simplicity_order": [
                "fewer raw predictors",
                "fewer components",
                "linear over spline",
                "PCA over PLS",
                "scalar-only over scalar-plus-profile",
                "lexicographically sorted serialized config",
            ],
            "complexity_modal_fraction_min": 0.60,
            "component_count_within_two_fraction_min": 0.75,
            "quarter_sign_fraction_min": 0.60,
            "largest_positive_quarter_share_max": 0.50,
            "quarter_min_dates": 10,
            "term_selection_frequency_min": 0.60,
            "term_sign_agreement_min": 0.75,
            "correlation_cluster_abs_spearman": 0.80,
            "random_subset_draws_per_outer_fold": 100,
            "interaction_names": interaction_names,
        },
        "feature_evidence": {
            "confirmatory_tests": 20,
            "exploratory_tests": 40,
            "interaction_incremental_tests": 8,
            "BH_q_max": 0.10,
            "development_dates_min": 400,
            "validation_dates_min": 150,
            "development_observations_min": 10000,
            "validation_observations_min": 4000,
            "validation_ic_retention_min": 0.25,
            "bucket_abs_monotonicity_min": 0.80,
            "directional_top_bottom_spread_ticks_min": 2.0,
            "partial_ic_abs_retention_min": 0.25,
            "interaction_dev_delta_min": 0.01,
            "interaction_validation_delta_min": 0.02,
        },
        "model_gate": {
            "development_paired_daily_ic_delta_min": 0.01,
            "validation_paired_daily_ic_delta_min": 0.02,
            "bootstrap_ci_lower_above_zero": True,
            "positive_model_ic_both_partitions": True,
            "development_oof_dates_min": 400,
            "validation_dates_min": 150,
            "best_10_dates_positive_improvement_share_max": 0.50,
            "passing_label": "RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA",
        },
        "policy": {
            "direction": "sign of frozen 60-minute directional prediction",
            "direction_score_threshold": {
                "quantile": 0.90,
                "fit_on": "Development outer-OOF absolute predictions by session",
            },
            "variants": [
                "direction_only_p90",
                "direction_p90_and_expansion_p80",
            ],
            "expansion_threshold": {
                "quantile": 0.80,
                "fit_on": "Development outer-OOF probabilities by session",
            },
            "global_london_new_york_sequencing": True,
        },
        "backtest": {
            "entry": "GC true t+1 open",
            "stop_atr_multiplier": 1.5,
            "stop_ticks_min": 10,
            "stop_ticks_max": 100,
            "target_R": 2.0,
            "max_holding_minutes": 120,
            "mandatory_flat": "15:30 America/New_York",
            "one_position_globally": True,
            "ambiguous_bar": "stop first",
            "cost_ticks": {
                "frictionless": 0.0,
                "base": 2.6,
                "pessimistic": 4.6,
            },
            "costs_are_proxy_assumptions": True,
        },
        "economic_gate": {
            "development_oof_dates_min": 400,
            "development_trades_min": 500,
            "validation_dates_min": 60,
            "validation_trades_min": 100,
            "positive_mean_net_R_both_partitions": True,
            "bootstrap_ci_lower_above_zero_both_partitions": True,
            "positive_quarter_fraction_min": 0.60,
            "quarter_min_dates": 10,
            "largest_positive_quarter_total_share_max": 0.50,
            "pessimistic_result_required_to_report": True,
        },
        "sharpe": {
            "name": "daily-net-R Sharpe",
            "daily_support_includes_zero_trade_dates": True,
            "annualization": 252,
            "risk_free_daily": 0.0,
            "formula": "sqrt(252)*mean(daily_net_R)/std(daily_net_R,ddof=1)",
            "partitions_separate": True,
            "stationary_bootstrap_restart_probability": 0.20,
            "stationary_bootstrap_replicates": 2000,
            "adjusted_p_denominator": 2001,
            "multiplicity_method": "max-Sharpe stationary-bootstrap adjusted p-value",
        },
        "mgc_transfer": {
            "adapter": "src/execution/fes_project1_mgc_adapter.py",
            "entry": "synchronized MGC t+1 open",
            "tick_size": 0.10,
            "tick_value_usd": 1.00,
            "native_open_structural_thresholds": {
                "minute_coverage_min": 0.98,
                "decision_coverage_min": 0.99,
                "median_abs_open_basis_ticks_max": 1.0,
                "p95_abs_open_basis_ticks_max": 3.0,
                "zero_volume_rate_max": 0.02,
                "median_decision_volume_min": 10,
            },
            "exact_gc_price_containment_min": 0.95,
            "bar_proxy_cost_ticks": {
                "frictionless": 0.0,
                "base": 2.6,
                "pessimistic": 4.6,
            },
            "spread_queue_impact_inference_prohibited": True,
            "strongest_without_order_telemetry": "GC_EDGE_MGC_BAR_PROXY_PROVISIONAL",
        },
        "final_verdict_vocabulary": [
            "REJECTED_NO_INCREMENTAL_FEATURE_VALUE",
            "PREDICTIVE_ONLY_NOT_DIRECTIONAL",
            "DIRECTIONAL_RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD_DATA",
            "DIRECTIONAL_PREDICTIVE_ONLY_GC_STRATEGY_REJECTED",
            "GC_EDGE_MGC_TRANSFER_FAILED",
            "GC_EDGE_MGC_BAR_PROXY_PROVISIONAL",
            "ADVANCE_TO_FORWARD_SHADOW_VALIDATION",
        ],
        "section_1_prohibitions": {
            "new_feature_construction": False,
            "target_association": False,
            "validation_outcome_inspection": False,
            "historical_final_outcome_inspection": False,
            "model_fit": False,
            "economic_evaluation": False,
            "mgc_policy_mining": False,
        },
    }


def config_sha256(config: Mapping[str, Any] | None = None) -> str:
    """Hash the canonical frozen config."""

    frozen = build_frozen_config() if config is None else dict(config)
    return sha256_bytes(canonical_json(frozen).encode("utf-8"))


def assert_frozen_specification(root: str | Path) -> str:
    """Assert that the implementation prompt still matches the frozen digest."""

    actual = sha256_file(Path(root) / SPECIFICATION_PATH)
    if actual != SPECIFICATION_SHA256:
        raise AssertionError(
            f"specification hash mismatch: expected {SPECIFICATION_SHA256}, got {actual}"
        )
    return actual


def build_trial_ledger_skeleton() -> list[dict[str, Any]]:
    """Predeclare all frozen evidence, model, policy, and economic trial families."""

    rows: list[dict[str, Any]] = []

    def add(
        trial_id: str,
        section: int,
        family: str,
        role: str,
        *,
        session: str = "ALL",
        target: str = "",
        candidate: str = "",
        comparator: str = "",
        multiplicity_family: str = "",
        status: str = "PLANNED",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "trial_id": trial_id,
                "notebook_section": section,
                "family": family,
                "analysis_role": role,
                "session": session,
                "target": target,
                "candidate": candidate,
                "comparator": comparator,
                "multiplicity_family": multiplicity_family,
                "status": status,
                "frozen_before_outcomes": True,
                "notes": notes,
            }
        )

    for session in ("London", "New York"):
        session_code = session.lower().replace(" ", "_")
        for feature in DIRECTIONAL_FEATURES:
            add(
                f"UCONF_{session_code}_{feature}",
                4,
                "univariate",
                "confirmatory",
                session=session,
                target="forward_return_60_atr",
                candidate=feature,
                comparator="zero daily IC",
                multiplicity_family="CONFIRMATORY_20",
            )
        for feature in OPPORTUNITY_FEATURES:
            add(
                f"UCONF_{session_code}_{feature}",
                4,
                "univariate",
                "confirmatory",
                session=session,
                target="future_range_60_atr",
                candidate=feature,
                comparator="zero daily IC",
                multiplicity_family="CONFIRMATORY_20",
            )
        for feature in DIRECTIONAL_FEATURES:
            for target in ("forward_return_30_atr", "future_range_60_atr"):
                add(
                    f"UEXP_{session_code}_{feature}_{target}",
                    4,
                    "univariate",
                    "exploratory",
                    session=session,
                    target=target,
                    candidate=feature,
                    comparator="zero daily IC",
                    multiplicity_family="EXPLORATORY_40",
                )
        for feature in OPPORTUNITY_FEATURES:
            for target in ("forward_return_60_atr", "forward_return_30_atr"):
                add(
                    f"UEXP_{session_code}_{feature}_{target}",
                    4,
                    "univariate",
                    "exploratory",
                    session=session,
                    target=target,
                    candidate=feature,
                    comparator="zero daily IC",
                    multiplicity_family="EXPLORATORY_40",
                )
        for item in INTERACTION_SPECS:
            if not item["id"].startswith("I"):
                continue
            add(
                f"INT_{session_code}_{item['id']}",
                4,
                "interaction",
                "confirmatory_incremental",
                session=session,
                target="forward_return_60_atr",
                candidate=item["name"],
                comparator="D1 plus required main effects",
                multiplicity_family="INTERACTION_8",
            )

    continuous_targets = (
        "forward_return_60_atr",
        "forward_return_30_atr",
        "future_range_60_atr",
    )
    for session in ("London", "New York"):
        session_code = session.lower().replace(" ", "_")
        for target in continuous_targets:
            for model in (f"M{index}" for index in range(9)):
                add(
                    f"MODEL_{session_code}_{target}_{model}",
                    5,
                    "continuous_model",
                    "nested_development_oof",
                    session=session,
                    target=target,
                    candidate=model,
                    comparator="relevant frozen anchor",
                    multiplicity_family="MODEL_LADDER",
                )
        for model in ("M0", "M1_O0", "M1_O1", "M2", "M3", "M_SELECTED_CONTINUOUS"):
            add(
                f"CLASS_{session_code}_{model}",
                5,
                "expansion_classifier",
                "nested_development_oof",
                session=session,
                target="expansion_label_60",
                candidate=model,
                comparator="prevalence and O1 baselines",
                multiplicity_family="EXPANSION_CLASSIFIER",
            )

    for policy in ("direction_only_p90", "direction_p90_and_expansion_p80"):
        add(
            f"POLICY_{policy}",
            5,
            "policy_freeze",
            "pre_validation",
            target="forward_return_60_atr",
            candidate=policy,
            comparator="no policy",
            multiplicity_family="POLICY_2",
        )
        for partition in ("Development OOF", "Retrospective Validation"):
            for cost in ("frictionless", "base", "pessimistic"):
                add(
                    f"ECON_{policy}_{partition.lower().replace(' ', '_')}_{cost}",
                    6,
                    "gc_economics",
                    "locked_batch",
                    target="daily net R",
                    candidate=policy,
                    comparator=cost,
                    multiplicity_family="POLICY_2",
                    notes=partition,
                )

    for verdict in ("NATIVE_OPEN_STRUCTURAL_PROXY_PASS", "EXACT_GC_PRICE_TRANSFER_PASS"):
        add(
            f"MGC_{verdict}",
            7,
            "mgc_transfer",
            "gated_structural",
            target="MGC mapping",
            candidate=verdict,
            comparator="frozen structural thresholds",
            multiplicity_family="MGC_TRANSFER",
        )

    return rows


def trial_ledger_sha256(rows: Iterable[Mapping[str, Any]] | None = None) -> str:
    """Hash the trial ledger records in their declared deterministic order."""

    records = build_trial_ledger_skeleton() if rows is None else [dict(row) for row in rows]
    return sha256_bytes(canonical_json(records).encode("utf-8"))


def write_contract_artifacts(
    root: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    ledger: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Write deterministic Section 1 frozen-config and ledger artifacts."""

    project_root = Path(root)
    output_dir = project_root / DATA_NAMESPACE
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = build_frozen_config() if config is None else dict(config)
    records = (
        build_trial_ledger_skeleton()
        if ledger is None
        else [dict(record) for record in ledger]
    )
    config_path = output_dir / "frozen_config.json"
    ledger_json_path = output_dir / "trial_ledger_skeleton.json"
    ledger_csv_path = output_dir / "trial_ledger_skeleton.csv"

    config_text = json.dumps(frozen, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ledger_json_text = (
        json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    ledger_csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        ledger_csv_buffer,
        fieldnames=list(records[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(records)

    for path, content in (
        (config_path, config_text),
        (ledger_json_path, ledger_json_text),
        (ledger_csv_path, ledger_csv_buffer.getvalue()),
    ):
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != content:
                raise FileExistsError(
                    f"refusing to overwrite conflicting frozen v1 artifact: {path}"
                )
            continue
        path.write_text(content, encoding="utf-8")

    return {
        "config_path": config_path.relative_to(project_root).as_posix(),
        "config_sha256": config_sha256(frozen),
        "trial_ledger_json_path": ledger_json_path.relative_to(project_root).as_posix(),
        "trial_ledger_csv_path": ledger_csv_path.relative_to(project_root).as_posix(),
        "trial_ledger_sha256": trial_ledger_sha256(records),
        "trial_count": str(len(records)),
    }
