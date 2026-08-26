"""Pure, outcome-free primitives for Q2 V4 Spark-1 qualification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

EXPERIMENT_ID = "Q2_V4_SPARK1_PRESEMANTIC"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_REPO = "cruxeval-org/cruxeval"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
LAYER = 27
LOCATIONS = ("PROMPT_BOUNDARY", "EXECUTION_BOUNDARY")
SHELLS = ("MEDIUM", "STRONG")
SHELL_TARGETS = {"MEDIUM": 0.25, "STRONG": 0.50}
SOURCE_FAMILIES = (
    "CONTROL_FLOW_PATH_COVERAGE",
    "MUTATION_ALIAS_CAUSALITY",
    "LOOP_BOUNDARY_ACCOUNTING",
    "HYPOTHESIS_BRANCH_ELIMINATION",
)
CANDIDATE_COUNT = 40
SELECTED_COUNT = 32
PRIMARY_N = 300
QAP_MAPS = 50_000
BOOTSTRAP_RESAMPLES = 10_000
SEED_PREFIX = "Q2-V4-INTERVENTION-SUBSPACE-DIRECTIONS-V1|"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def prelock_seed(prelock_commit: str) -> int:
    """Return the big-endian first 128 bits of the frozen seed payload."""

    if len(prelock_commit) != 40 or any(c not in "0123456789abcdef" for c in prelock_commit):
        raise ValueError("PRELOCK commit must be a full lowercase SHA-1")
    digest = hashlib.sha256(f"{SEED_PREFIX}{prelock_commit}".encode()).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def deterministic_seed(namespace: str, *parts: str | int) -> int:
    payload = "\x1f".join((namespace, *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def source_direction_id(family: str, location: str) -> str:
    if family not in SOURCE_FAMILIES or location not in LOCATIONS:
        raise ValueError("unknown V4 source direction")
    return f"MEAN_{family}_{location}"


def retained_subspace(vectors: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """SVD an ambient-by-source matrix under the prospective V4 gate."""

    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 8:
        raise ValueError("V4 source matrix must have eight columns")
    unit = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)
    q, singular, vh = np.linalg.svd(unit, full_matrices=False)
    retained = singular / singular[0] >= 1e-6
    rank = int(np.sum(retained))
    retained_singular = singular[retained]
    condition = float(retained_singular[0] / retained_singular[-1])
    probabilities = singular**2 / float(np.sum(singular**2))
    entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    stable_rank = float(np.sum(singular**2) / singular[0] ** 2)
    basis = q[:, retained]
    leverage_by_source = np.sum(vh[retained, :] ** 2, axis=0)
    concept_leverage = [
        float(leverage_by_source[2 * i] + leverage_by_source[2 * i + 1]) / rank for i in range(4)
    ]
    checks = {
        "retained_rank_at_least_6": rank >= 6,
        "condition_number_at_most_10": condition <= 10.0,
        "each_concept_leverage_at_least_0_01": min(concept_leverage) >= 0.01,
        "orthonormality_max_error_at_most_1e_10": float(
            np.max(np.abs(basis.T @ basis - np.eye(rank)))
        )
        <= 1e-10,
    }
    report = {
        "ambient_dimension": int(matrix.shape[0]),
        "source_count": int(matrix.shape[1]),
        "singular_values": singular.tolist(),
        "relative_singular_values": (singular / singular[0]).tolist(),
        "retained_rank": rank,
        "condition_number": condition,
        "entropy_effective_rank": entropy_rank,
        "stable_rank": stable_rank,
        "concept_leverage": concept_leverage,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return basis, report


def candidate_bank(basis: np.ndarray, prelock_commit: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate exactly forty isotropic coefficient rows from one stream."""

    q = np.asarray(basis, dtype=np.float64)
    seed = prelock_seed(prelock_commit)
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    gaussian = generator.standard_normal((CANDIDATE_COUNT, q.shape[1]))
    coefficients = gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)
    vectors = coefficients @ q.T
    return coefficients, vectors, seed


def bank_algebraic_checks(coefficients: np.ndarray, vectors: np.ndarray) -> dict[str, Any]:
    c = np.asarray(coefficients, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    singular = np.linalg.svd(c, compute_uv=False)
    gram = c @ c.T
    upper = np.abs(gram[np.triu_indices(len(c), 1)])
    eigen = singular**2
    probabilities = eigen / np.sum(eigen)
    entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    checks = {
        "finite": bool(np.isfinite(c).all() and np.isfinite(v).all()),
        "coefficient_norm_error_at_most_1e_12": float(
            np.max(np.abs(np.linalg.norm(c, axis=1) - 1.0))
        )
        <= 1e-12,
        "vector_norm_error_at_most_1e_10": float(np.max(np.abs(np.linalg.norm(v, axis=1) - 1.0)))
        <= 1e-10,
        "full_subspace_rank": int(np.linalg.matrix_rank(c, tol=1e-10)) == c.shape[1],
        "entropy_effective_rank_at_least_0_75r": entropy_rank >= 0.75 * c.shape[1],
        "condition_number_at_most_3": float(singular[0] / singular[-1]) <= 3.0,
        "max_absolute_pair_cosine_below_0_98": float(np.max(upper)) < 0.98,
    }
    return {
        "rank": int(np.linalg.matrix_rank(c, tol=1e-10)),
        "entropy_effective_rank": entropy_rank,
        "condition_number": float(singular[0] / singular[-1]),
        "max_absolute_pair_cosine": float(np.max(upper)),
        "checks": checks,
        "pass": all(checks.values()),
    }


def select_first_safe(safety: Mapping[str, Mapping[str, Any]]) -> list[str]:
    eligible = [f"V4_DIRECTION_{i:02d}" for i in range(CANDIDATE_COUNT)]
    selected = [name for name in eligible if bool(safety[name]["both_shells_pass"])]
    return selected[:SELECTED_COUNT]


def selected_bank_checks(coefficients: np.ndarray, amplitudes: np.ndarray) -> dict[str, Any]:
    c = np.asarray(coefficients, dtype=np.float64)
    a = np.asarray(amplitudes, dtype=np.float64)
    singular = np.linalg.svd(c, compute_uv=False)
    eigen = singular**2
    probs = eigen / np.sum(eigen)
    effective = float(np.exp(-np.sum(probs * np.log(probs))))
    cosine = np.abs(c @ c.T)[np.triu_indices(len(c), 1)]
    shell_cvs = np.std(a, axis=0) / np.mean(a, axis=0)
    a0 = 1.0 - c @ c.T
    spread = float(
        np.quantile(a0[np.triu_indices(len(c), 1)], 0.9)
        - np.quantile(a0[np.triu_indices(len(c), 1)], 0.1)
    )
    checks = {
        "selected_count_32": len(c) == SELECTED_COUNT,
        "full_subspace_rank": int(np.linalg.matrix_rank(c, tol=1e-10)) == c.shape[1],
        "entropy_effective_rank_at_least_0_75r": effective >= 0.75 * c.shape[1],
        "condition_number_at_most_3": float(singular[0] / singular[-1]) <= 3.0,
        "max_absolute_pair_cosine_below_0_98": float(np.max(cosine)) < 0.98,
        "a0_q90_q10_at_least_0_20": spread >= 0.20,
        "shell_amplitude_cv_at_most_0_03": float(np.max(shell_cvs)) <= 0.03,
    }
    return {
        "rank": int(np.linalg.matrix_rank(c, tol=1e-10)),
        "entropy_effective_rank": effective,
        "condition_number": float(singular[0] / singular[-1]),
        "max_absolute_pair_cosine": float(np.max(cosine)),
        "a0_q90_q10": spread,
        "shell_amplitude_cv": shell_cvs.tolist(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def mean_js(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    p = _probabilities(logits_a)
    q = _probabilities(logits_b)
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * (np.log(p) - np.log(m)), axis=-1)
    js += 0.5 * np.sum(q * (np.log(q) - np.log(m)), axis=-1)
    return float(np.mean(js))


def baseline_centered_angle(
    baseline: np.ndarray,
    fingerprints: Mapping[str, np.ndarray],
    *,
    noise_floor_squared: float,
) -> dict[str, Any]:
    names = list(fingerprints)
    radii2 = np.asarray([mean_js(fingerprints[name], baseline) for name in names])
    d2 = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            value = mean_js(fingerprints[left], fingerprints[names[j]])
            d2[i, j] = d2[j, i] = value
    gram = 0.5 * (radii2[:, None] + radii2[None, :] - d2)
    radii = np.sqrt(np.maximum(radii2, 0.0))
    denominator = radii[:, None] * radii[None, :]
    cosine = gram / denominator
    raw_min = float(np.min(cosine))
    raw_max = float(np.max(cosine))
    if raw_min < -1.0 - 1e-8 or raw_max > 1.0 + 1e-8:
        raise ValueError("A2 cosine exceeds frozen numerical tolerance")
    cosine = np.clip(cosine, -1.0, 1.0)
    np.fill_diagonal(cosine, 1.0)
    return {
        "names": names,
        "radii_squared": radii2,
        "distance_squared": d2,
        "gram": gram,
        "cosine": cosine,
        "dissimilarity": 1.0 - cosine,
        "raw_cosine_min": raw_min,
        "raw_cosine_max": raw_max,
        "gram_min_eigenvalue": float(np.min(np.linalg.eigvalsh(gram))),
        "radius_floor_pass": bool(np.all(radii2 > noise_floor_squared)),
    }


def unique_controller_permutations(prelock_commit: str) -> tuple[np.ndarray, int]:
    seed = int.from_bytes(
        hashlib.sha256(f"Q2-V4-QAP-V1|{prelock_commit}".encode()).digest()[:16],
        "big",
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    identity = tuple(range(SELECTED_COUNT))
    seen = {identity}
    rows = [identity]
    while len(rows) < QAP_MAPS:
        row = tuple(int(v) for v in rng.permutation(SELECTED_COUNT))
        if row not in seen:
            seen.add(row)
            rows.append(row)
    return np.asarray(rows, dtype=np.uint8), seed


def unique_shell_swaps(prelock_commit: str) -> tuple[np.ndarray, int]:
    """Return identity plus 49,999 unique paired medium/strong swap maps."""

    seed = int.from_bytes(
        hashlib.sha256(f"Q2-V4-RADIAL-SWAPS-V1|{prelock_commit}".encode()).digest()[:16],
        "big",
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    seen = {bytes(SELECTED_COUNT)}
    rows = [np.zeros(SELECTED_COUNT, dtype=np.uint8)]
    while len(rows) < QAP_MAPS:
        row = rng.integers(0, 2, size=SELECTED_COUNT, dtype=np.uint8)
        key = row.tobytes()
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return np.stack(rows), seed


def _wide_seed(namespace: str, *parts: str | int) -> int:
    payload = "\x1f".join((namespace, *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def semantic_schedule(
    item_ids: Sequence[str], selected_ids: Sequence[str], prelock_commit: str
) -> list[dict[str, Any]]:
    selected = list(selected_ids)
    if len(selected) != SELECTED_COUNT or len(set(selected)) != SELECTED_COUNT:
        raise ValueError("future semantic schedule requires 32 unique selected directions")
    allowed = {f"V4_DIRECTION_{index:02d}" for index in range(CANDIDATE_COUNT)}
    if not set(selected).issubset(allowed):
        raise ValueError("future semantic schedule contains an unknown direction")
    conditions = ["BASELINE"] + [
        f"{direction}_{shell}" for direction in selected for shell in SHELLS
    ]
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        for rollout in (0, 1):
            order_rng = np.random.Generator(
                np.random.PCG64DXSM(
                    _wide_seed("Q2-V4-CONDITION-ORDER", prelock_commit, item_id, rollout)
                )
            )
            for order, condition in enumerate(order_rng.permutation(conditions).tolist()):
                rows.append(
                    {
                        "item_id": item_id,
                        "condition": str(condition),
                        "rollout_index": rollout,
                        "condition_order": order,
                        "seed": _wide_seed(
                            "Q2-V4-INDEPENDENT-PRIMARY", prelock_commit, item_id, condition, rollout
                        ),
                    }
                )
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("future semantic schedule contains a seed collision")
    return rows


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "CANDIDATE_COUNT",
    "DATASET_REPO",
    "DATASET_REVISION",
    "EXPERIMENT_ID",
    "LAYER",
    "LOCATIONS",
    "MODEL",
    "MODEL_REVISION",
    "PRIMARY_N",
    "QAP_MAPS",
    "SELECTED_COUNT",
    "SHELLS",
    "SHELL_TARGETS",
    "SOURCE_FAMILIES",
    "bank_algebraic_checks",
    "baseline_centered_angle",
    "candidate_bank",
    "canonical_json_hash",
    "deterministic_seed",
    "prelock_seed",
    "retained_subspace",
    "selected_bank_checks",
    "select_first_safe",
    "semantic_schedule",
    "sha256_bytes",
    "source_direction_id",
    "unique_controller_permutations",
    "unique_shell_swaps",
]
