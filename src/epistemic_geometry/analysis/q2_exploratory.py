"""Reusable dyadic diagnostics for the post-hoc Q2 V2 principal review.

The helpers in this module preserve the controller-family split and item-level
dependence used by Q2 V2.  They intentionally expose no scientific decision
rule: every use in the principal review is exploratory and cannot alter the
frozen Q2 V2 classification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Spearman correlation without a SciPy dependency."""

    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if len(left_rank) < 3 or np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Pearson correlation, returning NaN for degenerate inputs."""

    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(left_array) < 3 or np.std(left_array) == 0.0 or np.std(right_array) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def upper_edges(matrix: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Return unique upper-triangle edges among ``indices``."""

    values = np.asarray(matrix, dtype=np.float64)
    selected = list(indices)
    return np.asarray(
        [
            values[left, right]
            for offset, left in enumerate(selected)
            for right in selected[offset + 1 :]
        ],
        dtype=np.float64,
    )


def fit_linear(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit an intercept plus a deliberately small frozen feature set."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) != len(y) or len(y) < x.shape[1] + 2:
        raise ValueError("linear calibration inputs are malformed")
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def linear_predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply coefficients returned by :func:`fit_linear`."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return np.column_stack([np.ones(len(x)), x]) @ np.asarray(coefficients)


def family_fold_edges(
    names: Sequence[str], family_by_name: Mapping[str, str], heldout_family: str
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return training indices and directed cross-family held-out edges.

    The edge convention exactly matches the frozen Q2 V2 analysis: calibration
    uses unique within-training-bank edges, while testing uses every edge from
    one held-out controller to every controller outside its family.
    """

    controller_names = list(names)
    train = [
        index
        for index, name in enumerate(controller_names)
        if family_by_name[name] != heldout_family
    ]
    heldout = [
        index
        for index, name in enumerate(controller_names)
        if family_by_name[name] == heldout_family
    ]
    test_edges = [
        (left, right)
        for left in heldout
        for right in train
        if left != right
    ]
    return train, test_edges


def _edge_feature_matrix(
    feature_matrices: Sequence[np.ndarray], edges: Sequence[tuple[int, int]]
) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray([matrix[left, right] for left, right in edges], dtype=np.float64)
            for matrix in feature_matrices
        ]
    )


def family_heldout_predictions(
    target: np.ndarray,
    feature_matrices: Sequence[np.ndarray],
    names: Sequence[str],
    family_by_name: Mapping[str, str],
) -> dict[str, Any]:
    """Fit on non-held-out families and persist every held-out prediction."""

    controller_names = list(names)
    families = sorted(set(family_by_name.values()))
    fold_records: dict[str, Any] = {}
    edge_records: list[dict[str, Any]] = []
    for family in families:
        train, test_edges = family_fold_edges(controller_names, family_by_name, family)
        train_edges = [
            (left, right)
            for offset, left in enumerate(train)
            for right in train[offset + 1 :]
        ]
        train_x = _edge_feature_matrix(feature_matrices, train_edges)
        train_y = np.asarray(
            [target[left, right] for left, right in train_edges], dtype=np.float64
        )
        test_x = _edge_feature_matrix(feature_matrices, test_edges)
        test_y = np.asarray(
            [target[left, right] for left, right in test_edges], dtype=np.float64
        )
        coefficients = fit_linear(train_x, train_y)
        predictions = linear_predict(test_x, coefficients)
        constant_predictions = np.full(len(test_y), float(np.mean(train_y)))
        rmse = float(np.sqrt(np.mean(np.square(predictions - test_y))))
        constant_rmse = float(
            np.sqrt(np.mean(np.square(constant_predictions - test_y)))
        )
        raw_feature_spearman = spearman(test_x[:, -1], test_y)
        raw_feature_pearson = pearson(test_x[:, -1], test_y)
        prediction_spearman = spearman(predictions, test_y)
        prediction_pearson = pearson(predictions, test_y)
        fold_records[family] = {
            "controllers": int(sum(family_by_name[name] == family for name in controller_names)),
            "train_edges": len(train_edges),
            "test_edges": len(test_edges),
            "spearman": (
                raw_feature_spearman if len(feature_matrices) == 1 else prediction_spearman
            ),
            "pearson": raw_feature_pearson if len(feature_matrices) == 1 else prediction_pearson,
            "raw_last_feature_spearman": raw_feature_spearman,
            "raw_last_feature_pearson": raw_feature_pearson,
            "prediction_spearman": prediction_spearman,
            "prediction_pearson": prediction_pearson,
            "rmse": rmse,
            "constant_rmse": constant_rmse,
            "rmse_ratio": rmse / constant_rmse if constant_rmse > 0 else None,
            "coefficients": coefficients.tolist(),
        }
        for index, ((left, right), observed, predicted, constant) in enumerate(
            zip(test_edges, test_y, predictions, constant_predictions, strict=True)
        ):
            edge_records.append(
                {
                    "heldout_family": family,
                    "left": controller_names[left],
                    "right": controller_names[right],
                    "edge_index": index,
                    "observed": float(observed),
                    "predicted": float(predicted),
                    "constant_prediction": float(constant),
                    "residual": float(observed - predicted),
                    "features": test_x[index].tolist(),
                }
            )
    mean_rmse = float(np.mean([fold["rmse"] for fold in fold_records.values()]))
    mean_constant = float(
        np.mean([fold["constant_rmse"] for fold in fold_records.values()])
    )
    return {
        "folds": fold_records,
        "aggregate": {
            "mean_spearman": float(
                np.nanmean([fold["spearman"] for fold in fold_records.values()])
            ),
            "median_spearman": float(
                np.nanmedian([fold["spearman"] for fold in fold_records.values()])
            ),
            "mean_pearson": float(
                np.nanmean([fold["pearson"] for fold in fold_records.values()])
            ),
            "mean_rmse": mean_rmse,
            "mean_constant_rmse": mean_constant,
            "rmse_ratio": mean_rmse / mean_constant if mean_constant > 0 else None,
        },
        "edge_records": edge_records,
    }


def family_heldout_incremental(
    target: np.ndarray,
    nuisance_matrices: Sequence[np.ndarray],
    candidate_matrix: np.ndarray,
    names: Sequence[str],
    family_by_name: Mapping[str, str],
) -> dict[str, Any]:
    """Compare a candidate geometry with frozen scalar nuisance predictors."""

    controller_names = list(names)
    folds: dict[str, Any] = {}
    for family in sorted(set(family_by_name.values())):
        train, test_edges = family_fold_edges(controller_names, family_by_name, family)
        train_edges = [
            (left, right)
            for offset, left in enumerate(train)
            for right in train[offset + 1 :]
        ]
        train_nuisance = _edge_feature_matrix(nuisance_matrices, train_edges)
        test_nuisance = _edge_feature_matrix(nuisance_matrices, test_edges)
        train_candidate = _edge_feature_matrix([candidate_matrix], train_edges)
        test_candidate = _edge_feature_matrix([candidate_matrix], test_edges)
        train_target = np.asarray(
            [target[left, right] for left, right in train_edges], dtype=np.float64
        )
        test_target = np.asarray(
            [target[left, right] for left, right in test_edges], dtype=np.float64
        )

        nuisance_coef = fit_linear(train_nuisance, train_target)
        nuisance_prediction = linear_predict(test_nuisance, nuisance_coef)
        augmented_train = np.column_stack([train_nuisance, train_candidate])
        augmented_test = np.column_stack([test_nuisance, test_candidate])
        augmented_coef = fit_linear(augmented_train, train_target)
        augmented_prediction = linear_predict(augmented_test, augmented_coef)

        candidate_on_nuisance = fit_linear(train_nuisance, train_candidate[:, 0])
        candidate_residual = test_candidate[:, 0] - linear_predict(
            test_nuisance, candidate_on_nuisance
        )
        target_residual = test_target - nuisance_prediction
        nuisance_rmse = float(
            np.sqrt(np.mean(np.square(nuisance_prediction - test_target)))
        )
        augmented_rmse = float(
            np.sqrt(np.mean(np.square(augmented_prediction - test_target)))
        )
        folds[family] = {
            "test_edges": len(test_edges),
            "residual_spearman": spearman(candidate_residual, target_residual),
            "residual_pearson": pearson(candidate_residual, target_residual),
            "nuisance_rmse": nuisance_rmse,
            "augmented_rmse": augmented_rmse,
            "augmented_to_nuisance_rmse_ratio": (
                augmented_rmse / nuisance_rmse if nuisance_rmse > 0 else None
            ),
        }
    mean_nuisance = float(np.mean([fold["nuisance_rmse"] for fold in folds.values()]))
    mean_augmented = float(np.mean([fold["augmented_rmse"] for fold in folds.values()]))
    return {
        "folds": folds,
        "aggregate": {
            "mean_residual_spearman": float(
                np.nanmean([fold["residual_spearman"] for fold in folds.values()])
            ),
            "mean_residual_pearson": float(
                np.nanmean([fold["residual_pearson"] for fold in folds.values()])
            ),
            "mean_nuisance_rmse": mean_nuisance,
            "mean_augmented_rmse": mean_augmented,
            "augmented_to_nuisance_rmse_ratio": (
                mean_augmented / mean_nuisance if mean_nuisance > 0 else None
            ),
        },
    }


def pair_feature_matrix(values: Sequence[float], operation: str) -> np.ndarray:
    """Create a symmetric pairwise scalar feature from controller values."""

    array = np.asarray(values, dtype=np.float64)
    if operation == "absolute_difference":
        matrix = np.abs(array[:, None] - array[None, :])
    elif operation == "mean":
        matrix = 0.5 * (array[:, None] + array[None, :])
    elif operation == "maximum":
        matrix = np.maximum(array[:, None], array[None, :])
    elif operation == "mismatch":
        matrix = (array[:, None] != array[None, :]).astype(np.float64)
    else:
        raise ValueError(f"unsupported pair feature operation: {operation}")
    np.fill_diagonal(matrix, 0.0)
    return matrix


def unbiased_error_distance(errors: np.ndarray) -> np.ndarray:
    """Compute the canonical two-rollout D matrix from controller errors."""

    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("errors must have controller x item x two-rollout shape")
    first = values[:, None, :, 0] - values[None, :, :, 0]
    second = values[:, None, :, 1] - values[None, :, :, 1]
    return np.mean(first * second, axis=2)


__all__ = [
    "average_ranks",
    "family_fold_edges",
    "family_heldout_incremental",
    "family_heldout_predictions",
    "fit_linear",
    "linear_predict",
    "pair_feature_matrix",
    "pearson",
    "spearman",
    "unbiased_error_distance",
    "upper_edges",
]
