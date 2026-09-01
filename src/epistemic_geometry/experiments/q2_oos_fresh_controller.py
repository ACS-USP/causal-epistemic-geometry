"""Outcome-free primitives for fresh-controller Q2 out-of-bank validation.

The functions in this module operate only on controller coefficients,
pre-outcome geometry, synthetic planning data, or future binary error arrays
supplied by an explicitly authorized analysis.  They never load semantic
outcomes or benchmark correctness.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np

from epistemic_geometry.experiments.q2_v4 import average_ranks

EXPERIMENT_ID = "Q2_OOS_FRESH_CONTROLLER_V1"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 27
SHELLS = ("MEDIUM", "STRONG")
SHELL_TARGETS = {"MEDIUM": 0.25, "STRONG": 0.50}
REFERENCE_COUNT = 31
PRIMARY_N = 300
ROLLOUTS = 2
QAP_MAPS = 50_000
BOOTSTRAP_RESAMPLES = 10_000
K_CANDIDATES = (6, 8, 10, 12, 16)
SEED_PREFIX = "Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V1|"


def protocol_seed(namespace: str, source_commit: str) -> int:
    """Return a stable 128-bit PCG64DXSM seed."""

    payload = f"{namespace}|{source_commit}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def fresh_candidate_bank(
    basis: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one normalized-Gaussian coefficient stream and map through Q."""

    q = np.asarray(basis, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 8:
        raise ValueError("fresh-controller basis must be ambient-by-8")
    if count < 8:
        raise ValueError("candidate stream must contain at least rank=8 rows")
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    gaussian = generator.standard_normal((count, 8))
    coefficients = gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)
    vectors = coefficients @ q.T
    return coefficients, vectors


def angular_cross_block(fresh: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return 1-cosine for fresh rows against frozen reference rows."""

    left = np.asarray(fresh, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("fresh/reference embeddings must share one feature dimension")
    left = left / np.linalg.norm(left, axis=1, keepdims=True)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)
    return 1.0 - np.clip(left @ right.T, -1.0, 1.0)


def spearman_flat(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation over a complete rectangular cross block."""

    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(x) != len(y):
        raise ValueError("cross-block arrays must have equal size")
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denominator)


def shell_mean_spearman(
    geometry_by_shell: dict[str, np.ndarray],
    outcome_by_shell: dict[str, np.ndarray],
) -> tuple[dict[str, float], float]:
    """Equal-weight shell-specific Spearman statistic."""

    if set(geometry_by_shell) != set(SHELLS) or set(outcome_by_shell) != set(SHELLS):
        raise ValueError("both frozen shells are required")
    shell = {
        name: spearman_flat(geometry_by_shell[name], outcome_by_shell[name]) for name in SHELLS
    }
    return shell, float(np.mean([shell[name] for name in SHELLS]))


def fresh_row_permutations(size: int, count: int, *, seed: int) -> np.ndarray:
    """Identity plus unique fresh-row permutations.

    For small K the complete group is returned when ``count`` exceeds K!.
    Reference columns are deliberately fixed.
    """

    if size < 3 or count < 2:
        raise ValueError("at least three fresh rows and two maps are required")
    target = min(count, math.factorial(size))
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    identity = tuple(range(size))
    seen = {identity}
    rows = [identity]
    while len(rows) < target:
        candidate = tuple(int(value) for value in generator.permutation(size))
        if candidate not in seen:
            seen.add(candidate)
            rows.append(candidate)
    return np.asarray(rows, dtype=np.int16)


def row_permutation_test(
    geometry_by_shell: dict[str, np.ndarray],
    outcome_by_shell: dict[str, np.ndarray],
    permutations: np.ndarray,
) -> dict[str, object]:
    """Fresh-row-label randomization with fixed reference columns."""

    observed_shell, observed = shell_mean_spearman(geometry_by_shell, outcome_by_shell)
    null = np.empty(len(permutations), dtype=np.float64)
    for index, permutation in enumerate(np.asarray(permutations, dtype=np.int64)):
        shell_values = [
            spearman_flat(geometry_by_shell[shell][permutation, :], outcome_by_shell[shell])
            for shell in SHELLS
        ]
        null[index] = float(np.mean(shell_values))
    return {
        "observed_shell_rho": observed_shell,
        "observed_aggregate_rho": observed,
        "permutation_statistics": null,
        "p_value": float(np.sum(null >= observed) / len(null)),
        "maps": int(len(null)),
    }


def leave_one_fresh_out(
    geometry_by_shell: dict[str, np.ndarray],
    outcome_by_shell: dict[str, np.ndarray],
) -> np.ndarray:
    """Return aggregate rho after deleting each complete fresh row."""

    size = next(iter(geometry_by_shell.values())).shape[0]
    values = []
    for omitted in range(size):
        keep = np.arange(size) != omitted
        _shell, aggregate = shell_mean_spearman(
            {name: geometry_by_shell[name][keep, :] for name in SHELLS},
            {name: outcome_by_shell[name][keep, :] for name in SHELLS},
        )
        values.append(aggregate)
    return np.asarray(values, dtype=np.float64)


def leave_one_reference_out(
    geometry_by_shell: dict[str, np.ndarray],
    outcome_by_shell: dict[str, np.ndarray],
) -> np.ndarray:
    """Return aggregate rho after deleting each frozen reference column."""

    size = next(iter(geometry_by_shell.values())).shape[1]
    values = []
    for omitted in range(size):
        keep = np.arange(size) != omitted
        _shell, aggregate = shell_mean_spearman(
            {name: geometry_by_shell[name][:, keep] for name in SHELLS},
            {name: outcome_by_shell[name][:, keep] for name in SHELLS},
        )
        values.append(aggregate)
    return np.asarray(values, dtype=np.float64)


def cross_block_shape(
    fresh_errors: np.ndarray,
    reference_errors: np.ndarray,
) -> np.ndarray:
    """Frozen N/(N-1) blind-spot-shape estimator for fresh×reference pairs."""

    fresh = np.asarray(fresh_errors, dtype=np.float64)
    reference = np.asarray(reference_errors, dtype=np.float64)
    if fresh.ndim != 3 or reference.ndim != 3 or fresh.shape[2] != 2:
        raise ValueError("errors must have controller x item x two-rollout shape")
    if reference.shape[2] != 2 or fresh.shape[1] != reference.shape[1]:
        raise ValueError("fresh/reference errors must share items and two rollouts")
    if np.any((fresh != 0.0) & (fresh != 1.0)) or np.any((reference != 0.0) & (reference != 1.0)):
        raise ValueError("error outcomes must be binary")
    items = fresh.shape[1]
    d0 = fresh[:, None, :, 0] - reference[None, :, :, 0]
    d1 = fresh[:, None, :, 1] - reference[None, :, :, 1]
    panel = np.mean(d0 * d1, axis=2) - np.mean(d0, axis=2) * np.mean(d1, axis=2)
    return panel * (items / (items - 1.0))


def semantic_schedule(
    item_ids: Sequence[str],
    fresh_ids: Sequence[str],
    lock_commit: str,
) -> list[dict[str, object]]:
    """Build the future schedule containing fresh conditions only."""

    conditions = [f"{controller}_{shell}" for controller in fresh_ids for shell in SHELLS]
    rows: list[dict[str, object]] = []
    for item_id in item_ids:
        for rollout in range(ROLLOUTS):
            order_seed = protocol_seed(f"Q2-OOS-CONDITION-ORDER|{item_id}|{rollout}", lock_commit)
            order_rng = np.random.Generator(np.random.PCG64DXSM(order_seed))
            for order, condition in enumerate(order_rng.permutation(conditions).tolist()):
                seed = protocol_seed(
                    f"Q2-OOS-SEMANTIC|{item_id}|{condition}|{rollout}", lock_commit
                ) & ((1 << 63) - 1)
                rows.append(
                    {
                        "item_id": item_id,
                        "condition": str(condition),
                        "rollout_index": rollout,
                        "condition_order": order,
                        "seed": seed,
                    }
                )
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("future fresh-controller schedule contains a seed collision")
    return rows


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "EXPERIMENT_ID",
    "K_CANDIDATES",
    "LAYER",
    "MODEL",
    "MODEL_REVISION",
    "PRIMARY_N",
    "QAP_MAPS",
    "REFERENCE_COUNT",
    "ROLLOUTS",
    "SHELLS",
    "SHELL_TARGETS",
    "angular_cross_block",
    "cross_block_shape",
    "fresh_candidate_bank",
    "fresh_row_permutations",
    "leave_one_fresh_out",
    "leave_one_reference_out",
    "protocol_seed",
    "row_permutation_test",
    "semantic_schedule",
    "shell_mean_spearman",
    "spearman_flat",
]
