"""Numerical qualification helpers for Gate 12.1.

This module contains no benchmark adapters, semantic evaluators, or historical
outcome readers.  It operates only on synthetic token fixtures and numerical
arrays produced by the engineering runner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 27
D75_ALPHA = 97.85168930581241
EPSILONS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SMALL_ALPHAS = (0.01, 0.1)
ALL_ALPHAS = (0.0, *SMALL_ALPHAS, D75_ALPHA)
ENGINE_DIRECTION_SEEDS = (1201001, 1201002)
FIXTURE_NAMESPACE = "GATE12.1-CONTINUOUS-GEOMETRY-ENGINE-V1"
CLASSIFICATIONS = (
    "GATE12_1_CONTINUOUS_GEOMETRY_ENGINE_QUALIFIED",
    "GATE12_1_SEQUENCE_SEMANTICS_BUG_REPAIRED_AND_ENGINE_QUALIFIED",
    "GATE12_1_SEQUENCE_SEMANTICS_BUG_FOUND_NOT_REPAIRED",
    "GATE12_1_FP32_ENGINE_QUALIFIED_BF16_BRIDGE_FAILED",
    "GATE12_1_DERIVATIVE_ENGINE_NOT_QUALIFIED",
    "GATE12_1_NUMERICAL_ANALYSIS_INCONCLUSIVE",
    "GATE12_1_ENGINE_FAILURE",
)


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tokens(seed: int, count: int, *, repeated: bool = False) -> list[int]:
    if repeated:
        return [1000 + seed % 97] * count
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(100, 30_000, size=count)]


def engineering_fixtures() -> list[dict[str, Any]]:
    """Return twelve deterministic, non-semantic token fixtures."""

    designs = (
        ("short_c1", "short_prompt_short_continuation", 6, 1, False),
        ("short_c2", "short_prompt_short_continuation", 7, 2, False),
        ("short_c8", "short_prompt_short_continuation", 8, 8, False),
        ("short_c32", "short_prompt_long_continuation", 9, 32, False),
        ("short_c64", "short_prompt_long_continuation", 10, 64, False),
        ("long_c1", "long_prompt_short_continuation", 96, 1, False),
        ("long_c8", "long_prompt_short_continuation", 112, 8, False),
        ("long_c32", "long_prompt_long_continuation", 128, 32, False),
        ("repeated_c64", "repeated_tokens", 48, 64, True),
        ("punctuation_code_c8", "punctuation_code_like_token_pattern", 40, 8, False),
        ("numeric_c32", "numeric_token_pattern", 44, 32, False),
        ("multilingual_neutral_c2", "multilingual_neutral_token_pattern", 52, 2, False),
    )
    fixtures: list[dict[str, Any]] = []
    for index, (name, category, prompt_len, continuation_len, repeated) in enumerate(designs):
        prompt = _tokens(12_100 + index, prompt_len, repeated=repeated)
        continuation = _tokens(12_200 + index, continuation_len, repeated=repeated)
        final_target = _tokens(12_300 + index, 1)[0]
        row: dict[str, Any] = {
            "fixture_id": f"G12_1_FIXTURE_{index:02d}",
            "name": name,
            "category": category,
            "prompt_token_ids": prompt,
            "continuation_token_ids": continuation,
            "target_token_ids": [*continuation, final_target],
            "direction_index": index % len(ENGINE_DIRECTION_SEEDS),
            "source": "synthetic_non_benchmark_token_sequence",
        }
        row["fixture_sha256"] = stable_sha256(row)
        fixtures.append(row)
    return fixtures


def engineering_directions(hidden_size: int = 4096) -> np.ndarray:
    records = []
    for seed in ENGINE_DIRECTION_SEEDS:
        vector = np.random.default_rng(seed).standard_normal(hidden_size).astype(np.float64)
        vector /= np.linalg.norm(vector)
        for earlier in records:
            vector -= np.dot(vector, earlier) * earlier
        vector /= np.linalg.norm(vector)
        records.append(vector)
    return np.stack(records)


def log_softmax64(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def js_divergence(left_logits: np.ndarray, right_logits: np.ndarray) -> np.ndarray:
    left_logp = log_softmax64(left_logits)
    right_logp = log_softmax64(right_logits)
    left = np.exp(left_logp)
    right = np.exp(right_logp)
    middle = 0.5 * (left + right)
    log_middle = np.log(middle)
    return 0.5 * (
        np.sum(left * (left_logp - log_middle), axis=-1)
        + np.sum(right * (right_logp - log_middle), axis=-1)
    )


def target_logp(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    logp = log_softmax64(logits)
    return np.take_along_axis(logp, np.asarray(targets)[..., None], axis=-1)[..., 0]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def relative_error(observed: float, expected: float, *, floor: float = 1e-12) -> float:
    return abs(float(observed) - float(expected)) / max(abs(float(expected)), floor)


def fisher_energy(logits: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    p = np.exp(log_softmax64(logits))
    r = np.asarray(derivative, dtype=np.float64)
    mean = np.sum(p * r, axis=-1)
    return np.sum(p * r * r, axis=-1) - mean * mean


def utility_slope(logits: np.ndarray, derivative: np.ndarray, target: int) -> float:
    p = np.exp(log_softmax64(logits))
    r = np.asarray(derivative, dtype=np.float64)
    return float(r[target] - np.sum(p * r))


def local_kl(logits: np.ndarray, moved_logits: np.ndarray) -> float:
    baseline_logp = log_softmax64(logits)
    moved_logp = log_softmax64(moved_logits)
    p = np.exp(baseline_logp)
    return float(np.sum(p * (baseline_logp - moved_logp)))


def stable_window(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_epsilon: list[dict[str, float]] = []
    for epsilon in EPSILONS:
        selected = [row for row in rows if float(row["epsilon"]) == epsilon]
        if not selected:
            continue
        by_epsilon.append(
            {
                "epsilon": epsilon,
                "jvp_cosine": float(np.nanmedian([row["jvp_cosine"] for row in selected])),
                "fisher_relative_error": float(
                    np.nanmedian([row["fisher_relative_error"] for row in selected])
                ),
                "utility_relative_error": float(
                    np.nanmedian([row["utility_relative_error"] for row in selected])
                ),
                "local_kl_relative_error": float(
                    np.nanmedian([row["local_kl_relative_error"] for row in selected])
                ),
            }
        )
    passing = [
        row["jvp_cosine"] >= 0.999
        and row["fisher_relative_error"] <= 0.05
        and row["utility_relative_error"] <= 0.05
        and row["local_kl_relative_error"] <= 0.05
        for row in by_epsilon
    ]
    windows = []
    for start in range(max(0, len(passing) - 2)):
        if all(passing[start : start + 3]):
            windows.append([by_epsilon[index]["epsilon"] for index in range(start, start + 3)])
    return {
        "per_epsilon_pooled_medians": by_epsilon,
        "three_consecutive_window": windows[0] if windows else None,
        "pass": bool(windows),
    }


def classify_qualification(
    *,
    semantic_bug_found: bool,
    semantic_bug_repaired: bool,
    fp32_sequence_pass: bool,
    bf16_bridge_pass: bool,
    derivative_pass: bool,
    inconclusive: bool = False,
) -> str:
    if semantic_bug_found and not semantic_bug_repaired:
        return "GATE12_1_SEQUENCE_SEMANTICS_BUG_FOUND_NOT_REPAIRED"
    if inconclusive:
        return "GATE12_1_NUMERICAL_ANALYSIS_INCONCLUSIVE"
    if not fp32_sequence_pass:
        return "GATE12_1_DERIVATIVE_ENGINE_NOT_QUALIFIED"
    if not derivative_pass:
        return "GATE12_1_DERIVATIVE_ENGINE_NOT_QUALIFIED"
    if not bf16_bridge_pass:
        return "GATE12_1_FP32_ENGINE_QUALIFIED_BF16_BRIDGE_FAILED"
    if semantic_bug_found:
        return "GATE12_1_SEQUENCE_SEMANTICS_BUG_REPAIRED_AND_ENGINE_QUALIFIED"
    return "GATE12_1_CONTINUOUS_GEOMETRY_ENGINE_QUALIFIED"


__all__ = [name for name in globals() if name.isupper()] + [
    "stable_sha256",
    "engineering_fixtures",
    "engineering_directions",
    "log_softmax64",
    "js_divergence",
    "target_logp",
    "cosine",
    "relative_error",
    "fisher_energy",
    "utility_slope",
    "local_kl",
    "stable_window",
    "classify_qualification",
]
