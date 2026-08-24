"""Controller-held-out prediction and QAP inference for the Q2 pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from epistemic_geometry.analysis.rank_statistics import spearman_correlation


def edge_indices(
    controller_ids: Sequence[str], train_controllers: Sequence[str]
) -> dict[str, list[tuple[int, int]]]:
    """Return train/train calibration edges and every edge touching heldout controllers."""

    names = tuple(controller_ids)
    train = set(train_controllers)
    if not train < set(names):
        raise ValueError("train controllers must be a proper subset of the bank")
    train_edges: list[tuple[int, int]] = []
    test_edges: list[tuple[int, int]] = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            target = train_edges if names[left] in train and names[right] in train else test_edges
            target.append((left, right))
    return {"train": train_edges, "heldout": test_edges}


def edge_values(matrix: np.ndarray, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return np.asarray([values[left, right] for left, right in edges], dtype=np.float64)


def fit_affine(train_geometry: np.ndarray, train_target: np.ndarray) -> tuple[float, float]:
    """Fit D=a+b*d only on train-controller edges."""

    x = np.asarray(train_geometry, dtype=np.float64).reshape(-1)
    y = np.asarray(train_target, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("affine calibration inputs are malformed")
    design = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def heldout_prediction(
    geometry: np.ndarray,
    target: np.ndarray,
    controller_ids: Sequence[str],
    train_controllers: Sequence[str],
) -> dict[str, Any]:
    """Score one frozen geometry without exposing heldout targets to calibration."""

    edges = edge_indices(controller_ids, train_controllers)
    train_x = edge_values(geometry, edges["train"])
    train_y = edge_values(target, edges["train"])
    test_x = edge_values(geometry, edges["heldout"])
    test_y = edge_values(target, edges["heldout"])
    intercept, slope = fit_affine(train_x, train_y)
    predictions = intercept + slope * test_x
    train_scale = float(np.std(train_y, ddof=1))
    if not np.isfinite(train_scale) or train_scale <= 0:
        raise ValueError("train-edge D has no scale for standardized RMSE")
    rmse = float(np.sqrt(np.mean(np.square(predictions - test_y))))
    constant = np.full_like(test_y, float(np.mean(train_y)))
    constant_rmse = float(np.sqrt(np.mean(np.square(constant - test_y))))
    rho = spearman_correlation(test_x, test_y)
    return {
        "heldout_spearman_rho": rho,
        "heldout_edge_count": len(test_y),
        "train_edge_count": len(train_y),
        "affine_intercept": intercept,
        "affine_slope": slope,
        "heldout_rmse": rmse,
        "heldout_standardized_rmse": rmse / train_scale,
        "constant_heldout_rmse": constant_rmse,
        "rmse_ratio_to_constant": rmse / constant_rmse if constant_rmse > 0 else None,
        "train_target_mean": float(np.mean(train_y)),
        "train_target_sd": train_scale,
        "test_targets_used_in_fit": False,
    }


def qap_permutation(
    geometry: np.ndarray,
    target: np.ndarray,
    controller_ids: Sequence[str],
    train_controllers: Sequence[str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Permute controller labels of D, preserving dyadic dependence."""

    edges = edge_indices(controller_ids, train_controllers)["heldout"]
    fixed = edge_values(geometry, edges)
    observed = spearman_correlation(fixed, edge_values(target, edges))
    if observed is None:
        return {"observed_rho": None, "p_value_one_sided": None, "permutations": 0}
    rng = np.random.default_rng(seed)
    values = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        order = rng.permutation(len(controller_ids))
        permuted = np.asarray(target)[np.ix_(order, order)]
        statistic = spearman_correlation(fixed, edge_values(permuted, edges))
        values[index] = np.nan if statistic is None else statistic
    finite = values[np.isfinite(values)]
    exceedances = int(np.sum(finite >= observed - 1e-15))
    return {
        "observed_rho": float(observed),
        "p_value_one_sided": (exceedances + 1) / (len(finite) + 1),
        "permutations": int(len(finite)),
        "seed": int(seed),
        "null_q025": float(np.quantile(finite, 0.025)),
        "null_median": float(np.quantile(finite, 0.5)),
        "null_q975": float(np.quantile(finite, 0.975)),
        "permutation_unit": "controller label",
    }


def metric_signal(score: Mapping[str, Any], qap: Mapping[str, Any]) -> dict[str, bool]:
    checks = {
        "rho_at_least_0_30": bool(score["heldout_spearman_rho"] is not None)
        and float(score["heldout_spearman_rho"]) >= 0.30,
        "qap_one_sided_p_at_most_0_05": bool(qap["p_value_one_sided"] is not None)
        and float(qap["p_value_one_sided"]) <= 0.05,
        "rmse_at_least_10pct_better_than_constant": bool(
            score["rmse_ratio_to_constant"] is not None
        )
        and float(score["rmse_ratio_to_constant"]) <= 0.90,
    }
    checks["signal"] = all(checks.values())
    return checks


def classify_q2(metric_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen DEVELOPMENT hierarchy to M0/M1/M2."""

    signals = {
        name: metric_signal(record["score"], record["qap"])
        for name, record in metric_results.items()
    }
    flat = metric_results["M0_FLAT"]["score"]
    outperform: dict[str, bool] = {}
    for name in ("M1_WHITENED", "M2_FINITE_SECANT"):
        score = metric_results[name]["score"]
        outperform[name] = bool(
            signals[name]["signal"]
            and score["heldout_spearman_rho"] >= flat["heldout_spearman_rho"] + 0.15
            and score["heldout_rmse"] <= 0.90 * flat["heldout_rmse"]
        )
    if any(outperform.values()):
        classification = "Q2_PILOT_CONTROL_GEOMETRY_OUTPERFORMS_FLAT"
    elif signals["M0_FLAT"]["signal"]:
        classification = "Q2_PILOT_SIMPLE_GEOMETRY_SIGNAL"
    elif signals["M1_WHITENED"]["signal"] or signals["M2_FINITE_SECANT"]["signal"]:
        classification = "Q2_PILOT_HELDOUT_PREDICTION_SIGNAL"
    else:
        classification = "Q2_PILOT_NO_HELDOUT_GEOMETRY_SIGNAL"
    return {
        "classification": classification,
        "metric_signals": signals,
        "control_geometry_outperformance": outperform,
        "development_only": True,
    }


__all__ = [
    "classify_q2",
    "edge_indices",
    "edge_values",
    "fit_affine",
    "heldout_prediction",
    "metric_signal",
    "qap_permutation",
]
