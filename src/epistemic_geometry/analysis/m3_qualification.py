"""Prospective, outcome-free contracts for Q2 M3 engineering qualification.

The module defines a teacher-forced categorical-Fisher Gram on a small
controller span.  It contains no benchmark adapter, semantic evaluator, or
historical outcome reader.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 27
FIXTURE_NAMESPACE = "Q2-M3-QUALIFICATION-NONSCIENTIFIC-V1"
DIRECTION_NAMESPACE = "Q2-M3-QUALIFICATION-DIRECTIONS-V1"
FIXTURE_COUNT = 16
DIRECTION_COUNT = 6
HIDDEN_SIZE = 4096
CHECKPOINT_OFFSETS = (0, 1, 3, 7, 15, 31, 63)
EPSILONS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
LOCAL_EPSILON_MAX = 1.0
BF16_BRIDGE_EPSILON = 3.0
BRIDGE_FIXTURE_INDICES = (0, 3, 7, 11)
DIFFERENTIAL_FIXTURE_INDICES = (0, 3, 7, 11)
POLARIZATION_FIXTURE_INDICES = (0, 3, 7, 11)
EXACT_CROSSCHECK_CASES = ((0, 0), (3, 1), (7, 2), (11, 3))

THRESHOLDS: dict[str, float | int] = {
    "alpha_zero_max_js": 1e-12,
    "repeat_gram_relative_frobenius": 5e-5,
    "order_gram_relative_frobenius": 5e-5,
    "batch_gram_relative_frobenius": 5e-4,
    "independent_jvp_cosine": 0.99999,
    "independent_jvp_relative_norm": 0.005,
    "jvp_vjp_relative_error": 1e-4,
    "direct_polarization_relative_frobenius": 0.01,
    "psd_relative_negative_eigenvalue": 1e-8,
    "finite_window_length": 3,
    "finite_jvp_cosine": 0.999,
    "finite_relative_error": 0.05,
    "finite_angular_max_abs_error": 0.05,
    "finite_local_rms_logit_movement": 0.01,
    "fp32_sequence_top1": 1.0,
    "fp32_sequence_median_js": 1e-8,
    "fp32_sequence_p99_js": 1e-6,
    "fp32_sequence_median_target_logp": 1e-5,
    "fp32_sequence_max_target_logp": 1e-3,
    "fp32_sequence_median_logit_cosine": 0.999999,
    "bf16_bridge_top1": 0.99,
    "bf16_bridge_median_js": 1e-4,
    "bf16_bridge_p95_js": 5e-3,
    "bf16_bridge_radius_spearman": 0.95,
    "bf16_bridge_distance_spearman": 0.95,
    "bf16_bridge_curvature_median_relative": 0.15,
}

CLASSIFICATIONS = (
    "M3_DIRECTIONAL_ENGINE_QUALIFIED",
    "M3_FP32_COHERENT_BF16_SURROGATE_NOT_QUALIFIED",
    "M3_DERIVATIVE_IDENTITIES_FAILED",
    "M3_FINITE_LOCAL_WINDOW_FAILED",
    "M3_SEQUENCE_SEMANTICS_FAILED",
    "M3_ENGINE_FAILURE",
)


def stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()


def engineering_directions(
    count: int = DIRECTION_COUNT, hidden_size: int = HIDDEN_SIZE
) -> np.ndarray:
    """Return deterministic orthonormal engineering-only directions as rows."""

    seed = int(stable_digest(DIRECTION_NAMESPACE)[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((hidden_size, count))
    q, _ = np.linalg.qr(raw, mode="reduced")
    directions = q.T.astype(np.float64)
    for index in range(count):
        pivot = int(np.argmax(np.abs(directions[index])))
        if directions[index, pivot] < 0:
            directions[index] *= -1
    return directions


def _token_ids(seed: int, count: int, *, repeated: bool = False) -> list[int]:
    rng = np.random.default_rng(seed)
    if repeated:
        base = rng.integers(200, 20_000, size=max(2, min(5, count)))
        return [int(base[index % len(base)]) for index in range(count)]
    return [int(value) for value in rng.integers(200, 100_000, size=count)]


def engineering_fixtures() -> list[dict[str, Any]]:
    """Create 16 synthetic token fixtures with no task semantics or oracle."""

    lengths = (
        (4, 1),
        (8, 2),
        (16, 8),
        (32, 32),
        (64, 64),
        (96, 8),
        (8, 64),
        (48, 32),
        (12, 8),
        (24, 32),
        (80, 2),
        (6, 64),
        (16, 8),
        (20, 32),
        (72, 1),
        (10, 64),
    )
    fixtures: list[dict[str, Any]] = []
    for index, (prompt_n, continuation_n) in enumerate(lengths):
        seed = int(stable_digest(FIXTURE_NAMESPACE, index)[:16], 16) % (2**32)
        prompt = _token_ids(seed, prompt_n, repeated=index in {5, 8, 12})
        continuation = _token_ids(seed + 1, continuation_n, repeated=index in {6, 11, 15})
        offsets = [value for value in CHECKPOINT_OFFSETS if value <= continuation_n]
        fixture = {
            "fixture_id": f"Q2_M3_FIXTURE_{index:02d}",
            "source": "synthetic_non_benchmark_token_sequence",
            "prompt_token_ids": prompt,
            "continuation_token_ids": continuation,
            "checkpoint_offsets": offsets,
            "no_task_oracle": True,
            "semantic_correctness_available": False,
        }
        fixture["fixture_sha256"] = stable_digest(
            FIXTURE_NAMESPACE,
            json.dumps(fixture, sort_keys=True, separators=(",", ":")),
        )
        fixtures.append(fixture)
    return fixtures


def weighted_fisher_gram(logit_jvps: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Compute ``R(diag(p)-pp')R'`` without materializing the Fisher matrix."""

    rows = np.asarray(logit_jvps, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if rows.ndim != 2 or rows.shape[1] != p.size:
        raise ValueError("JVP rows and probabilities are incompatible")
    if not np.isclose(np.sum(p), 1.0, atol=1e-10) or np.any(p < 0):
        raise ValueError("probabilities must be nonnegative and sum to one")
    means = rows @ p
    second = (rows * p[None, :]) @ rows.T
    result = second - np.outer(means, means)
    return 0.5 * (result + result.T)


def relative_frobenius(observed: np.ndarray, expected: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(observed) - np.asarray(expected))
    denominator = max(np.linalg.norm(np.asarray(expected)), 1e-15)
    return float(numerator / denominator)


def gram_geometry(gram: np.ndarray) -> dict[str, np.ndarray | float]:
    values = 0.5 * (np.asarray(gram, dtype=np.float64) + np.asarray(gram).T)
    eig = np.linalg.eigvalsh(values)
    radii = np.sqrt(np.maximum(np.diag(values), 0.0))
    denom = np.outer(radii, radii)
    cosine = np.full_like(values, np.nan)
    mask = denom > 0
    cosine[mask] = np.clip(values[mask] / denom[mask], -1.0, 1.0)
    sqdist = np.maximum(np.diag(values)[:, None] + np.diag(values)[None, :] - 2 * values, 0)
    np.fill_diagonal(sqdist, 0.0)
    return {
        "eigenvalues": eig,
        "radii": radii,
        "cosine": cosine,
        "distances": np.sqrt(sqdist),
        "minimum_eigenvalue": float(eig[0]),
        "maximum_eigenvalue": float(eig[-1]),
    }


def stable_local_window(per_epsilon: Sequence[Mapping[str, float]]) -> list[float] | None:
    """Return the first prospectively valid three-scale local window."""

    needed = int(THRESHOLDS["finite_window_length"])
    ordered = sorted(per_epsilon, key=lambda row: row["epsilon"])
    eligible: list[bool] = []
    for row in ordered:
        eligible.append(
            bool(
                row["epsilon"] <= LOCAL_EPSILON_MAX
                and row["jvp_cosine"] >= THRESHOLDS["finite_jvp_cosine"]
                and row["fisher_relative_error"] <= THRESHOLDS["finite_relative_error"]
                and row["kl_relative_error"] <= THRESHOLDS["finite_relative_error"]
                and row["hellinger_relative_error"] <= THRESHOLDS["finite_relative_error"]
                and row["js_relative_error"] <= THRESHOLDS["finite_relative_error"]
                and row["gram_relative_error"] <= THRESHOLDS["finite_relative_error"]
                and row["radius_relative_error"] <= THRESHOLDS["finite_relative_error"]
                and row["angle_max_abs_error"] <= THRESHOLDS["finite_angular_max_abs_error"]
                and row["rms_logit_movement"] <= THRESHOLDS["finite_local_rms_logit_movement"]
            )
        )
    for start in range(len(ordered) - needed + 1):
        if all(eligible[start : start + needed]):
            return [float(row["epsilon"]) for row in ordered[start : start + needed]]
    return None


def classify_m3(
    *,
    sequence_pass: bool,
    derivative_pass: bool,
    finite_window_pass: bool,
    bf16_bridge_pass: bool,
    engine_ok: bool = True,
) -> str:
    if not engine_ok:
        return "M3_ENGINE_FAILURE"
    if not sequence_pass:
        return "M3_SEQUENCE_SEMANTICS_FAILED"
    if not derivative_pass:
        return "M3_DERIVATIVE_IDENTITIES_FAILED"
    if not finite_window_pass:
        return "M3_FINITE_LOCAL_WINDOW_FAILED"
    if not bf16_bridge_pass:
        return "M3_FP32_COHERENT_BF16_SURROGATE_NOT_QUALIFIED"
    return "M3_DIRECTIONAL_ENGINE_QUALIFIED"


__all__ = [
    "BF16_BRIDGE_EPSILON",
    "BRIDGE_FIXTURE_INDICES",
    "CHECKPOINT_OFFSETS",
    "CLASSIFICATIONS",
    "DIRECTION_COUNT",
    "DIFFERENTIAL_FIXTURE_INDICES",
    "EPSILONS",
    "FIXTURE_COUNT",
    "HIDDEN_SIZE",
    "LAYER",
    "LOCAL_EPSILON_MAX",
    "MODEL",
    "MODEL_REVISION",
    "POLARIZATION_FIXTURE_INDICES",
    "EXACT_CROSSCHECK_CASES",
    "THRESHOLDS",
    "classify_m3",
    "engineering_directions",
    "engineering_fixtures",
    "gram_geometry",
    "relative_frobenius",
    "stable_digest",
    "stable_local_window",
    "weighted_fisher_gram",
]
