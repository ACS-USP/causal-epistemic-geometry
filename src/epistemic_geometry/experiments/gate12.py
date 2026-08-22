"""Pure mathematical and provenance contracts for Gate 12.

Gate 12 measures one-dimensional, local trajectory pullbacks along frozen
sustained-control directions.  It does not estimate a full pullback matrix.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

EXPERIMENT_ID = "GATE12_UTILITY_ALIGNED_PULLBACK"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 27
ETA_D75 = 9.637427952852196
REFERENCE_SCALE = 10.153299177386142
D75_SCALAR = ETA_D75 * REFERENCE_SCALE
CONTROLLER_HASH = "e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838"
CONTROL_ITEMS_PER_DOMAIN = 24
UTILITY_ITEMS_PER_DOMAIN = 32
CONTROL_SEQUENCE_CAP = 128
CONTROL_CHECKPOINTS = (-1, 0, 1, 3, 7, 15, 31, 63, 127)
DIRECTIONS_PER_DOMAIN = 5
EPSILON_Q = 1e-12
EPSILON_KL = 1e-12
FINITE_DIFFERENCE_DIVISORS = (128, 64, 32)
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20260826
CONTROL_BOOTSTRAP_SEED = 20260827
RAW_DTYPE = "float32"


def rank_utility_ids(domain: str, item_ids: Sequence[str]) -> list[str]:
    """Apply the frozen outcome-independent Gate-12 utility ranking."""

    label = "CRUX" if domain == "CRUXEval" else "CHARCOUNT"

    def key(item_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"GATE12-UTILITY-PREDICTION|{label}|{item_id}".encode()).hexdigest()
        return digest, str(item_id)

    return sorted(map(str, item_ids), key=key)


def canonical_answer(domain: str, reference: Any) -> str:
    """Render the globally frozen minimal correct continuation."""

    if domain == "CRUXEval":
        value = str(reference)
    elif domain == "CHARCOUNT":
        value = str(int(reference))
    else:
        raise ValueError(f"unsupported Gate-12 domain: {domain}")
    return f"FINAL: {value}"


def softmax64(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def fisher_energy(logits: np.ndarray, jvp: np.ndarray) -> np.ndarray:
    """Categorical Fisher directional energy using the stable variance identity."""

    p = softmax64(logits)
    r = np.asarray(jvp, dtype=np.float64)
    mean = np.sum(p * r, axis=-1)
    second = np.sum(p * np.square(r), axis=-1)
    return second - np.square(mean)


def utility_slope(logits: np.ndarray, jvp: np.ndarray, target_token_ids: np.ndarray) -> np.ndarray:
    p = softmax64(logits)
    r = np.asarray(jvp, dtype=np.float64)
    targets = np.asarray(target_token_ids, dtype=np.int64)
    if r.shape[:-1] != targets.shape:
        raise ValueError("target-token shape does not match JVP positions")
    mean = np.sum(p * r, axis=-1)
    selected = np.take_along_axis(r, targets[..., None], axis=-1)[..., 0]
    return selected - mean


def fisher_inner(logits: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = softmax64(logits)
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mean_a = np.sum(p * a, axis=-1)
    mean_b = np.sum(p * b, axis=-1)
    return np.sum(p * a * b, axis=-1) - mean_a * mean_b


def fisher_cosine(logits: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = fisher_inner(logits, left, right)
    left_q = fisher_energy(logits, left)
    right_q = fisher_energy(logits, right)
    denominator = np.sqrt(np.maximum(left_q * right_q, 0.0))
    result = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def geometry_summary(
    logits: np.ndarray,
    jvp: np.ndarray,
    target_token_ids: np.ndarray,
    careful_logits: np.ndarray,
) -> dict[str, float]:
    q = fisher_energy(logits, jvp)
    u = utility_slope(logits, jvp, target_token_ids)
    careful_delta = np.asarray(careful_logits, dtype=np.float64) - np.asarray(
        logits, dtype=np.float64
    )
    alignment = fisher_cosine(logits, jvp, careful_delta)
    q_mean = float(np.mean(q))
    return {
        "Q_local": q_mean,
        "Q_Hellinger": q_mean / 4.0,
        "U_mean": float(np.mean(u)),
        "U_sum": float(np.sum(u)),
        "eta_utility": float(np.mean(u) / np.sqrt(q_mean + EPSILON_Q)),
        "fisher_careful_alignment": float(np.nanmean(alignment)),
    }


def categorical_kl(baseline_logits: np.ndarray, condition_logits: np.ndarray) -> np.ndarray:
    left = np.asarray(baseline_logits, dtype=np.float64)
    right = np.asarray(condition_logits, dtype=np.float64)
    left_logp = left - np.logaddexp.reduce(left, axis=-1, keepdims=True)
    right_logp = right - np.logaddexp.reduce(right, axis=-1, keepdims=True)
    p = np.exp(left_logp)
    return np.sum(p * (left_logp - right_logp), axis=-1)


def centered_ranks(values: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    """Average ranks within domain, centered to zero within each domain."""

    x = np.asarray(values, dtype=np.float64)
    labels = np.asarray(list(groups), dtype=object)
    result = np.empty_like(x)
    for group in sorted(set(labels.tolist())):
        mask = labels == group
        ranks = average_ranks(x[mask])
        result[mask] = ranks - np.mean(ranks)
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Dependency-free average ranks, including exact tie handling."""

    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and x[order[stop]] == x[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    x = average_ranks(np.asarray(left, dtype=np.float64))
    y = average_ranks(np.asarray(right, dtype=np.float64))
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


def domain_centered_spearman(
    left: np.ndarray, right: np.ndarray, domains: Sequence[str]
) -> float | None:
    x = centered_ranks(np.asarray(left, dtype=np.float64), domains)
    y = centered_ranks(np.asarray(right, dtype=np.float64), domains)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


def classify(
    *, control_supported: bool, item_utility_supported: bool, domain_utility_supported: bool
) -> str:
    if control_supported and item_utility_supported and domain_utility_supported:
        return "GATE12_UTILITY_ALIGNED_PULLBACK_SUPPORTED"
    if control_supported and domain_utility_supported:
        return "GATE12_PULLBACK_CONTROL_WITH_DOMAIN_LEVEL_UTILITY_ALIGNMENT"
    if control_supported:
        return "GATE12_PULLBACK_CONTROL_WITHOUT_UTILITY_PREDICTION"
    if item_utility_supported or domain_utility_supported:
        return "GATE12_UTILITY_ALIGNMENT_WITHOUT_PULLBACK_CONTROL_PREDICTION"
    return "GATE12_LOCAL_GEOMETRY_NOT_PREDICTIVE"


def historical_utility_target(
    baseline_rows: Mapping[int, bool], condition_rows: Mapping[int, bool]
) -> float:
    if set(baseline_rows) != {0, 1} or set(condition_rows) != {0, 1}:
        raise ValueError("historical utility requires exactly rollouts 0 and 1")
    return float(
        np.mean([int(condition_rows[index]) - int(baseline_rows[index]) for index in (0, 1)])
    )


__all__ = [name for name in globals() if name.isupper()] + [
    "rank_utility_ids",
    "canonical_answer",
    "softmax64",
    "fisher_energy",
    "utility_slope",
    "fisher_inner",
    "fisher_cosine",
    "geometry_summary",
    "categorical_kl",
    "average_ranks",
    "spearman",
    "centered_ranks",
    "domain_centered_spearman",
    "classify",
    "historical_utility_target",
]
