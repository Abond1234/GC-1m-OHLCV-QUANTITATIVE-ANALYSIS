"""Probability recalibration for the Section 9 classification benchmarks.

Section 9 recorded that the New York 60-minute logistic probabilities are
rank-correct but overstate their upper deciles.  This module provides
dependency-free isotonic recalibration: fit a monotone map from predicted to
realized probability on Development only, apply it unchanged elsewhere.  The
implementation is the standard pool-adjacent-violators algorithm with linear
interpolation between fitted points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IsotonicCalibration:
    """A fitted monotone probability map."""

    fitted_predictions: np.ndarray
    fitted_probabilities: np.ndarray

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        return np.interp(
            values,
            self.fitted_predictions,
            self.fitted_probabilities,
            left=float(self.fitted_probabilities[0]),
            right=float(self.fitted_probabilities[-1]),
        )


def _pool_adjacent_violators(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted isotonic regression producing a non-decreasing sequence."""

    means = values.astype(np.float64).copy()
    block_weight = weights.astype(np.float64).copy()
    block_end = np.arange(len(means))
    index = 0
    while index < len(means) - 1:
        if means[index] <= means[index + 1] + 1e-15:
            index += 1
            continue
        merged_weight = block_weight[index] + block_weight[index + 1]
        merged_mean = (
            means[index] * block_weight[index] + means[index + 1] * block_weight[index + 1]
        ) / merged_weight
        means[index] = merged_mean
        block_weight[index] = merged_weight
        block_end[index] = block_end[index + 1]
        means = np.delete(means, index + 1)
        block_weight = np.delete(block_weight, index + 1)
        block_end = np.delete(block_end, index + 1)
        index = max(index - 1, 0)
    return means, block_weight, block_end


def fit_isotonic_calibration(
    predicted_probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> IsotonicCalibration:
    """Fit a monotone predicted-to-realized map (Development data only)."""

    predictions = np.asarray(predicted_probabilities, dtype=np.float64)
    labels = np.asarray(outcomes, dtype=np.float64)
    finite = np.isfinite(predictions) & np.isfinite(labels)
    predictions, labels = predictions[finite], labels[finite]
    if len(predictions) < 2:
        raise ValueError("isotonic calibration requires at least two observations")
    order = np.argsort(predictions, kind="mergesort")
    predictions, labels = predictions[order], labels[order]
    unique_predictions, start_index = np.unique(predictions, return_index=True)
    grouped_means = np.add.reduceat(labels, start_index) / np.diff(
        np.append(start_index, len(labels))
    )
    weights = np.diff(np.append(start_index, len(labels))).astype(np.float64)
    means, _, block_end = _pool_adjacent_violators(grouped_means, weights)
    fitted_predictions = unique_predictions[block_end]
    return IsotonicCalibration(
        fitted_predictions=np.asarray(fitted_predictions, dtype=np.float64),
        fitted_probabilities=np.asarray(means, dtype=np.float64),
    )


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    return float(np.mean((p - y) ** 2))
