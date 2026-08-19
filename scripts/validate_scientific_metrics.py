#!/usr/bin/env python3
"""Run deterministic analytic/synthetic checks for the canonical estimands."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.rank_statistics import label_permutation_test  # noqa: E402
from epistemic_geometry.metrics.reasoning import (  # noqa: E402
    SeedRegime,
    stochastic_complementarity_estimands,
    unbiased_two_rollout_propensity_distance,
)
from epistemic_geometry.metrics.uncertainty import cluster_bootstrap_mean  # noqa: E402


def _cyclic_distances(order: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        [
            min(abs(order[i] - order[j]), 7 - abs(order[i] - order[j]))
            for i in range(7)
            for j in range(i + 1, 7)
        ],
        dtype=float,
    )


def main() -> int:
    rng = np.random.default_rng(20260819)
    p0 = rng.uniform(0.05, 0.95, size=128)
    pj = rng.uniform(0.05, 0.95, size=128)
    estimands = stochastic_complementarity_estimands(p0, pj)
    if abs(estimands["decomposition_residual"]) > 1e-12:
        raise AssertionError("stochastic complementarity decomposition failed")

    true_distance = float(np.mean((p0 - pj) ** 2))
    estimates = []
    for _ in range(4_000):
        errors_0 = rng.random((p0.size, 2)) < p0[:, None]
        errors_j = rng.random((pj.size, 2)) < pj[:, None]
        estimates.append(
            unbiased_two_rollout_propensity_distance(
                errors_0,
                errors_j,
                seed_regime=SeedRegime.INDEPENDENT_PRIMARY,
            )
        )
    estimate_mean = float(np.mean(estimates))
    if abs(estimate_mean - true_distance) > 0.01:
        raise AssertionError("two-rollout distance Monte Carlo validation failed")

    labels = tuple(range(7))
    permutation = label_permutation_test(
        labels,
        _cyclic_distances(labels),
        _cyclic_distances,
        exact=True,
    )
    if permutation["permutations"] != math.factorial(7):
        raise AssertionError("weekday exact permutation did not enumerate 7!")

    matched_rejected = False
    try:
        unbiased_two_rollout_propensity_distance(
            errors_0,
            errors_j,
            seed_regime=SeedRegime.MATCHED_COUPLING_SECONDARY,
        )
    except ValueError:
        matched_rejected = True
    if not matched_rejected:
        raise AssertionError("matched coupling was accepted by an independence estimator")

    clustered = cluster_bootstrap_mean(
        [0.0] * 100 + [1.0],
        ["large-cluster"] * 100 + ["small-cluster"],
        seed=20260819,
        n_resamples=1_000,
    )
    if clustered["estimate"] != 0.5:
        raise AssertionError("cluster bootstrap treated nested rows as independent")

    summary = {
        "status": "PASS",
        "stochastic_decomposition_residual": estimands["decomposition_residual"],
        "two_rollout_true_distance": true_distance,
        "two_rollout_monte_carlo_mean": estimate_mean,
        "two_rollout_absolute_bias": abs(estimate_mean - true_distance),
        "weekday_exact_permutations": permutation["permutations"],
        "weekday_identity_p": permutation["p_value"],
        "matched_coupling_rejected": matched_rejected,
        "cluster_bootstrap_equal_problem_estimate": clustered["estimate"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
