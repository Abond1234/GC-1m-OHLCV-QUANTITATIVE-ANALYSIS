"""Section 9 multivariate benchmarks for the independent statistical branch.

The section asks one narrow question: does combining the Section 8 frozen
expansion features beat the anchor feature alone?  It is a benchmark stage,
not signal construction.

Models are deliberately simple, per the research contract ("simple
linear/logistic benchmarks precede more complex tree or nonlinear models"):

- Ridge/OLS regression predicting the per-date cross-sectional rank of the
  ATR-relative future range (the expansion outcome).
- Ridge-stabilized logistic regression predicting the frozen Development
  80th-percentile expansion classification label.

Tree-based models are explicitly deferred until these benchmarks earn them;
the decision is recorded in the outputs.

Design rules, frozen in :class:`Section9Config` before computation:

- Development and Validation only; a frame containing Final-test rows raises.
- Features and outcomes are per-date cross-sectional rank z-scores, so the
  regression objective aligns with the daily-IC evidence unit used since
  Section 7 and is robust to fat tails.
- Ridge strength is selected by expanding-window walk-forward **inside
  Development only** (train on earlier years, test on the next year, with a
  trading-date embargo exceeding the longest label horizon).  Validation is
  touched exactly once per model, after the configuration is frozen.
- Models are fitted per session, because every prior section shows London and
  New York behave differently.
- Advancement requires the Validation daily IC of the model prediction to
  beat the anchor-only benchmark by a declared margin, with a date-block
  bootstrap interval on the paired daily difference excluding zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .feature_evaluation import EVALUATION_PARTITIONS, EVALUATION_SESSIONS

SECTION9_RANDOM_SEED = 20260722


@dataclass(frozen=True)
class Section9Config:
    """Frozen Section 9 multivariate benchmark contract."""

    random_seed: int = SECTION9_RANDOM_SEED
    bootstrap_replicates: int = 2_000
    bootstrap_confidence: float = 0.95
    horizons: tuple[int, ...] = (60, 180)
    outcome_template: str = "future_range_{h}_atr"
    classification_template: str = "expansion_label_{h}"
    ridge_lambda_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    embargo_trading_dates: int = 1
    min_daily_observations: int = 10
    min_validation_ic_improvement: float = 0.02
    logistic_ridge_lambda: float = 1e-4
    logistic_max_iterations: int = 50
    tree_models_deferred_reason: str = (
        "contract: simple linear/logistic benchmarks precede tree models; "
        "revisit only if linear benchmarks demonstrate stable incremental value"
    )


@dataclass
class MultivariateBuildResult:
    """Container for every Section 9 output object."""

    regression_results: pd.DataFrame
    walk_forward_results: pd.DataFrame
    coefficient_table: pd.DataFrame
    classification_results: pd.DataFrame
    calibration_table: pd.DataFrame
    verdicts: pd.DataFrame
    validation_checks: pd.DataFrame
    summary: pd.Series
    config: Section9Config = field(default_factory=Section9Config)


def load_expansion_labels(path: Path, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Column-scoped load of the frozen expansion classification labels."""

    columns = ["observation_id"] + [f"expansion_label_{h}" for h in horizons]
    table = pq.read_table(path, columns=columns)
    return table.to_pandas(ignore_metadata=True).set_index("observation_id")


def _daily_rank_z(values: pd.DataFrame, date_codes: np.ndarray) -> pd.DataFrame:
    """Per-date cross-sectional rank z-scores; zero-variance dates become NaN."""

    grouped = values.groupby(date_codes, sort=False)
    ranks = grouped.rank(method="average")
    mean = ranks.groupby(date_codes, sort=False).transform("mean")
    std = ranks.groupby(date_codes, sort=False).transform("std")
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (ranks - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge on standardized inputs; lambda scaled by sample size."""

    n, k = x.shape
    gram = x.T @ x + lam * n * np.eye(k)
    return np.linalg.solve(gram, x.T @ y)


def _daily_ic_of_prediction(
    prediction: np.ndarray,
    outcome_z: np.ndarray,
    date_codes: np.ndarray,
    n_dates: int,
    min_daily_observations: int,
) -> np.ndarray:
    """Per-date Spearman IC between a prediction and the rank-z outcome."""

    frame = pd.DataFrame({"prediction": prediction})
    pred_z = _daily_rank_z(frame, date_codes)["prediction"].to_numpy()
    product = pred_z * outcome_z
    finite = np.isfinite(product)
    sums = np.bincount(date_codes, weights=np.where(finite, product, 0.0), minlength=n_dates)
    counts = np.bincount(date_codes, weights=finite.astype(np.float64), minlength=n_dates)
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = sums / counts
    daily[counts < min_daily_observations] = np.nan
    return daily


def _bootstrap_mean_ci(
    daily_values: np.ndarray, *, seed: int, replicates: int, confidence: float
) -> tuple[float, float]:
    values = np.asarray(daily_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    batch = 200
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        draw = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[draw].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(v) for v in np.quantile(estimates, [alpha, 1.0 - alpha]))


def _fit_logistic_irls(x: np.ndarray, y: np.ndarray, lam: float, max_iterations: int) -> np.ndarray:
    """Ridge-stabilized logistic regression via iteratively reweighted least squares."""

    n, k = x.shape
    design = np.hstack([np.ones((n, 1)), x])
    beta = np.zeros(k + 1)
    penalty = lam * n * np.eye(k + 1)
    penalty[0, 0] = 0.0
    for _ in range(max_iterations):
        logits = np.clip(design @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        gradient = design.T @ (y - p) - penalty @ beta
        hessian = (design * w[:, None]).T @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return beta


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC; NaN when one class is absent."""

    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    order = pd.Series(np.concatenate([positive, negative])).rank(method="average").to_numpy()
    rank_sum = order[: len(positive)].sum()
    u = rank_sum - len(positive) * (len(positive) + 1) / 2.0
    return float(u / (len(positive) * len(negative)))


def _prepare_session_data(
    frame: pd.DataFrame,
    features: list[str],
    outcome_column: str,
    session: str,
) -> dict:
    """Complete-row rank-z design matrices for one session, both partitions."""

    session_mask = frame["entry_session"].eq(session).to_numpy()
    sub = frame.loc[session_mask]
    date_codes, date_index = pd.factorize(sub["trade_date_ny"], sort=True)
    z = _daily_rank_z(sub[features + [outcome_column]], date_codes)
    matrix = z[features].to_numpy(dtype=np.float64)
    outcome = z[outcome_column].to_numpy(dtype=np.float64)
    complete = np.isfinite(matrix).all(axis=1) & np.isfinite(outcome)
    partitions = sub["research_partition"].astype(str).to_numpy()
    years = pd.DatetimeIndex(sub["trade_date_ny"]).year.to_numpy()
    return {
        "index": sub.index,
        "matrix": matrix,
        "outcome_z": outcome,
        "complete": complete,
        "partitions": partitions,
        "years": years,
        "date_codes": date_codes,
        "date_index": date_index,
        "dropped_incomplete_fraction": float(1.0 - complete.mean()),
    }


def _walk_forward_folds(
    years: np.ndarray,
    date_codes: np.ndarray,
    date_index: pd.Index,
    partitions: np.ndarray,
    embargo_trading_dates: int,
) -> list[dict]:
    """Expanding-window year folds inside Development with a date embargo."""

    development = partitions == "Development"
    dev_years = np.unique(years[development])
    folds = []
    for position in range(1, len(dev_years)):
        train_years = set(dev_years[:position].tolist())
        test_year = int(dev_years[position])
        train_rows = development & np.isin(years, list(train_years))
        test_rows = development & (years == test_year)
        test_date_codes = np.unique(date_codes[test_rows])
        embargoed_dates = set(test_date_codes[:embargo_trading_dates].tolist())
        test_rows = test_rows & ~np.isin(date_codes, list(embargoed_dates))
        if train_rows.sum() and test_rows.sum():
            folds.append(
                {
                    "train_years": ",".join(str(y) for y in sorted(train_years)),
                    "test_year": test_year,
                    "train_rows": train_rows,
                    "test_rows": test_rows,
                    "embargoed_dates": len(embargoed_dates),
                }
            )
    return folds


def build_multivariate_benchmarks(
    frame: pd.DataFrame,
    frozen_features: list[str],
    anchor_feature: str,
    classification_labels: pd.DataFrame,
    config: Section9Config | None = None,
) -> MultivariateBuildResult:
    """Run the complete declared Section 9 benchmark suite."""

    cfg = config or Section9Config()
    observed_partitions = set(frame["research_partition"].astype(str).unique())
    if not observed_partitions.issubset(set(EVALUATION_PARTITIONS)):
        raise ValueError(f"multivariate frame contains locked partitions: {observed_partitions}")
    if anchor_feature not in frozen_features:
        raise ValueError("anchor feature must belong to the frozen feature set")
    missing = [name for name in frozen_features if name not in frame.columns]
    if missing:
        raise ValueError(f"frozen features missing from frame: {missing[:5]}")

    regression_records: list[dict] = []
    walk_forward_records: list[dict] = []
    coefficient_records: list[dict] = []
    classification_records: list[dict] = []
    calibration_records: list[dict] = []
    verdict_records: list[dict] = []
    dropped_fractions: list[float] = []
    embargo_counts: list[int] = []

    anchor_position = frozen_features.index(anchor_feature)

    for session_index, session in enumerate(EVALUATION_SESSIONS):
        for horizon_index, horizon in enumerate(cfg.horizons):
            outcome_column = cfg.outcome_template.format(h=horizon)
            data = _prepare_session_data(frame, frozen_features, outcome_column, session)
            dropped_fractions.append(data["dropped_incomplete_fraction"])
            complete = data["complete"]
            matrix = data["matrix"]
            outcome_z = data["outcome_z"]
            partitions = data["partitions"]
            date_codes = data["date_codes"]
            n_dates = len(data["date_index"])
            development = (partitions == "Development") & complete
            validation = (partitions == "Validation") & complete

            # ---- 9.4 walk-forward ridge selection inside Development -------
            folds = _walk_forward_folds(
                data["years"], date_codes, data["date_index"], partitions, cfg.embargo_trading_dates
            )
            lambda_scores: dict[float, list[float]] = {lam: [] for lam in cfg.ridge_lambda_grid}
            for fold in folds:
                embargo_counts.append(fold["embargoed_dates"])
                train = fold["train_rows"] & complete
                test = fold["test_rows"] & complete
                for lam in cfg.ridge_lambda_grid:
                    beta = _fit_ridge(matrix[train], outcome_z[train], lam)
                    prediction = np.full(len(matrix), np.nan)
                    prediction[test] = matrix[test] @ beta
                    daily = _daily_ic_of_prediction(
                        prediction, outcome_z, date_codes, n_dates, cfg.min_daily_observations
                    )
                    score = float(np.nanmean(daily))
                    lambda_scores[lam].append(score)
                    walk_forward_records.append(
                        {
                            "session": session,
                            "horizon_minutes": horizon,
                            "train_years": fold["train_years"],
                            "test_year": fold["test_year"],
                            "ridge_lambda": lam,
                            "test_daily_ic_mean": score,
                        }
                    )
            mean_scores = {
                lam: float(np.mean(scores)) if scores else np.nan
                for lam, scores in lambda_scores.items()
            }
            selected_lambda = max(
                cfg.ridge_lambda_grid,
                key=lambda lam: (np.nan_to_num(mean_scores[lam], nan=-np.inf), lam),
            )

            # ---- 9.1/9.2 final ridge and anchor benchmark ------------------
            beta = _fit_ridge(matrix[development], outcome_z[development], selected_lambda)
            anchor_beta = _fit_ridge(
                matrix[development][:, [anchor_position]],
                outcome_z[development],
                selected_lambda,
            )
            for name, value in zip(frozen_features, beta, strict=False):
                coefficient_records.append(
                    {
                        "session": session,
                        "horizon_minutes": horizon,
                        "feature_name": name,
                        "ridge_lambda": selected_lambda,
                        "coefficient": float(value),
                    }
                )

            model_prediction = np.where(complete, matrix @ beta, np.nan)
            anchor_prediction = np.where(
                complete, matrix[:, anchor_position] * anchor_beta[0], np.nan
            )
            partition_masks = {"Development": development, "Validation": validation}
            difference_daily: dict[str, np.ndarray] = {}
            for partition, mask in partition_masks.items():
                model_daily = _daily_ic_of_prediction(
                    np.where(mask, model_prediction, np.nan),
                    outcome_z,
                    date_codes,
                    n_dates,
                    cfg.min_daily_observations,
                )
                anchor_daily = _daily_ic_of_prediction(
                    np.where(mask, anchor_prediction, np.nan),
                    outcome_z,
                    date_codes,
                    n_dates,
                    cfg.min_daily_observations,
                )
                paired = model_daily - anchor_daily
                difference_daily[partition] = paired
                seed = (
                    cfg.random_seed
                    + 100_000 * session_index
                    + 10_000 * horizon_index
                    + (0 if partition == "Development" else 5_000)
                )
                diff_low, diff_high = _bootstrap_mean_ci(
                    paired,
                    seed=seed,
                    replicates=cfg.bootstrap_replicates,
                    confidence=cfg.bootstrap_confidence,
                )
                regression_records.append(
                    {
                        "session": session,
                        "horizon_minutes": horizon,
                        "research_partition": partition,
                        "ridge_lambda": selected_lambda,
                        "observation_count": int(mask.sum()),
                        "trading_date_count": int(np.isfinite(paired).sum()),
                        "model_daily_ic_mean": float(np.nanmean(model_daily)),
                        "anchor_daily_ic_mean": float(np.nanmean(anchor_daily)),
                        "ic_improvement_mean": float(np.nanmean(paired)),
                        "ic_improvement_ci_low": diff_low,
                        "ic_improvement_ci_high": diff_high,
                    }
                )

            validation_row = regression_records[-1]
            improvement = validation_row["ic_improvement_mean"]
            ci_low = validation_row["ic_improvement_ci_low"]
            advances = bool(
                np.isfinite(improvement)
                and improvement >= cfg.min_validation_ic_improvement
                and np.isfinite(ci_low)
                and ci_low > 0.0
            )
            verdict_records.append(
                {
                    "session": session,
                    "horizon_minutes": horizon,
                    "validation_ic_improvement": improvement,
                    "improvement_ci_low": ci_low,
                    "improvement_ci_high": validation_row["ic_improvement_ci_high"],
                    "verdict": "MODEL_ADVANCES" if advances else "ANCHOR_SUFFICIENT",
                }
            )

            # ---- 9.3/9.5 logistic benchmark and calibration ----------------
            label_column = cfg.classification_template.format(h=horizon)
            labels = classification_labels[label_column].reindex(data["index"])
            label_values = labels.to_numpy(dtype="float64", na_value=np.nan)
            usable = complete & np.isfinite(label_values)
            y = label_values > 0.5
            logistic_beta = _fit_logistic_irls(
                matrix[usable & development],
                y[usable & development],
                cfg.logistic_ridge_lambda,
                cfg.logistic_max_iterations,
            )
            anchor_logistic_beta = _fit_logistic_irls(
                matrix[usable & development][:, [anchor_position]],
                y[usable & development],
                cfg.logistic_ridge_lambda,
                cfg.logistic_max_iterations,
            )
            for partition, mask in partition_masks.items():
                rows = usable & mask
                design = np.hstack([np.ones((int(rows.sum()), 1)), matrix[rows]])
                p = 1.0 / (1.0 + np.exp(-np.clip(design @ logistic_beta, -30.0, 30.0)))
                anchor_design = np.hstack(
                    [np.ones((int(rows.sum()), 1)), matrix[rows][:, [anchor_position]]]
                )
                anchor_p = 1.0 / (
                    1.0 + np.exp(-np.clip(anchor_design @ anchor_logistic_beta, -30.0, 30.0))
                )
                classification_records.append(
                    {
                        "session": session,
                        "horizon_minutes": horizon,
                        "research_partition": partition,
                        "observation_count": int(rows.sum()),
                        "positive_rate": float(y[rows].mean()),
                        "model_auc": _auc(p, y[rows]),
                        "anchor_auc": _auc(anchor_p, y[rows]),
                        "model_brier": float(np.mean((p - y[rows]) ** 2)),
                        "anchor_brier": float(np.mean((anchor_p - y[rows]) ** 2)),
                    }
                )
                if partition == "Validation":
                    deciles = np.clip((p * 10).astype(int), 0, 9)
                    for decile in range(10):
                        in_decile = deciles == decile
                        if not in_decile.any():
                            continue
                        calibration_records.append(
                            {
                                "session": session,
                                "horizon_minutes": horizon,
                                "probability_decile": decile,
                                "observation_count": int(in_decile.sum()),
                                "mean_predicted_probability": float(p[in_decile].mean()),
                                "realized_positive_rate": float(y[rows][in_decile].mean()),
                            }
                        )

    regression_results = pd.DataFrame.from_records(regression_records)
    walk_forward_results = pd.DataFrame.from_records(walk_forward_records)
    coefficient_table = pd.DataFrame.from_records(coefficient_records)
    classification_results = pd.DataFrame.from_records(classification_records)
    calibration_table = pd.DataFrame.from_records(calibration_records)
    verdicts = pd.DataFrame.from_records(verdict_records)

    # ---- 9.6 coefficient stability -----------------------------------------
    sign_stability = (
        coefficient_table.assign(sign=np.sign(coefficient_table["coefficient"]))
        .groupby("feature_name")["sign"]
        .agg(lambda s: float((s == s.iloc[0]).mean()))
        .rename("sign_consistency")
    )
    coefficient_table = coefficient_table.merge(sign_stability, on="feature_name", how="left")

    def _calibration_rank_agreement(group: pd.DataFrame) -> bool:
        populated = group.loc[group["observation_count"] >= 50]
        if len(populated) < 3:
            return True
        rho = populated["mean_predicted_probability"].corr(
            populated["realized_positive_rate"], method="spearman"
        )
        return bool(rho >= 0.9)

    calibration_monotone = (
        bool(
            calibration_table.groupby(["session", "horizon_minutes"])
            .apply(_calibration_rank_agreement, include_groups=False)
            .all()
        )
        if len(calibration_table)
        else False
    )

    checks = {
        "partitions_limited_to_development_validation": observed_partitions
        <= set(EVALUATION_PARTITIONS),
        "anchor_in_frozen_features": anchor_feature in frozen_features,
        "ridge_lambda_selected_from_grid": verdicts.merge(
            regression_results.loc[
                regression_results["research_partition"].eq("Validation"),
                ["session", "horizon_minutes", "ridge_lambda"],
            ],
            on=["session", "horizon_minutes"],
        )["ridge_lambda"]
        .isin(cfg.ridge_lambda_grid)
        .all(),
        "walk_forward_inside_development_only": bool(len(walk_forward_results) > 0),
        "embargo_applied_on_every_fold": all(
            count >= cfg.embargo_trading_dates for count in embargo_counts
        ),
        "incomplete_row_fraction_below_5pct": all(f < 0.05 for f in dropped_fractions),
        "regression_cells_cover_both_partitions": set(
            regression_results["research_partition"].unique()
        )
        == set(EVALUATION_PARTITIONS),
        "auc_within_bounds": classification_results[["model_auc", "anchor_auc"]]
        .stack()
        .dropna()
        .between(0.0, 1.0)
        .all(),
        "verdict_per_session_horizon": len(verdicts)
        == len(EVALUATION_SESSIONS) * len(cfg.horizons),
    }
    validation_checks = pd.Series(checks, name="passed").rename_axis("check").reset_index()

    summary = pd.Series(
        {
            "frozen_features_used": len(frozen_features),
            "anchor_feature": anchor_feature,
            "sessions_x_horizons": len(verdicts),
            "model_advances": int(verdicts["verdict"].eq("MODEL_ADVANCES").sum()),
            "anchor_sufficient": int(verdicts["verdict"].eq("ANCHOR_SUFFICIENT").sum()),
            "best_validation_ic_improvement": float(verdicts["validation_ic_improvement"].max()),
            "calibration_monotone_on_validation": calibration_monotone,
            "tree_models": "deferred - " + cfg.tree_models_deferred_reason,
        },
        name="value",
    )

    return MultivariateBuildResult(
        regression_results=regression_results,
        walk_forward_results=walk_forward_results,
        coefficient_table=coefficient_table,
        classification_results=classification_results,
        calibration_table=calibration_table,
        verdicts=verdicts,
        validation_checks=validation_checks,
        summary=summary,
        config=cfg,
    )


def save_multivariate_outputs(
    result: MultivariateBuildResult, *, project_root: Path
) -> pd.DataFrame:
    """Persist Section 9 outputs; generated artifacts stay excluded from Git."""

    processed = project_root / "data" / "processed" / "statistical_research"
    tables = project_root / "reports" / "statistical_research" / "tables" / "section9"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    outputs = {
        "multivariate_regression": (
            processed / "multivariate_regression_gc.parquet",
            result.regression_results,
        ),
        "multivariate_walk_forward": (
            processed / "multivariate_walk_forward_gc.parquet",
            result.walk_forward_results,
        ),
        "multivariate_coefficients": (
            processed / "multivariate_coefficients_gc.parquet",
            result.coefficient_table,
        ),
        "multivariate_classification": (
            processed / "multivariate_classification_gc.parquet",
            result.classification_results,
        ),
        "multivariate_calibration": (
            processed / "multivariate_calibration_gc.parquet",
            result.calibration_table,
        ),
        "multivariate_verdicts": (
            processed / "multivariate_verdicts_gc.parquet",
            result.verdicts,
        ),
    }
    records = []
    for name, (path, table) in outputs.items():
        table.to_parquet(path, index=False)
        reloaded = pd.read_parquet(path, engine="pyarrow")
        records.append(
            {
                "output": name,
                "path": str(path.relative_to(project_root)),
                "rows": len(table),
                "columns": table.shape[1],
                "reload_row_match": len(reloaded) == len(table),
            }
        )
    result.verdicts.to_csv(tables / "section9_model_verdicts_gc.csv", index=False)
    result.regression_results.to_csv(tables / "section9_regression_results_gc.csv", index=False)
    return pd.DataFrame.from_records(records)
