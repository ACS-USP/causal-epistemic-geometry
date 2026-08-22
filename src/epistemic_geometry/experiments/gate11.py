"""Pure contracts for the Gate 11 domain-conditioned control postmortem."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from epistemic_geometry.experiments.gate6_3 import (
    bank_geometry,
    single_layer_random_bank,
    vector_sha256,
)
from epistemic_geometry.experiments.gate7 import REFERENCE_SCALE
from epistemic_geometry.reproducibility import stable_seed

EXPERIMENT_ID = "GATE11_DOMAIN_CONDITIONED_CONTROL_POSTMORTEM"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CONTROLLER_HASH = "e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838"
ETA = 9.637427952852196
LAYER = 27
SOURCE_LAYERS = (8, 12, 17, 22, 27, 32)
PROPAGATION_LAYERS = (27, 28, 30, 32, 35)
CHECKPOINTS = (0, 1, 3, 7, 15, 31, 63, 127, 255)
SEQUENCE_CAP = 256
SOURCE_ITEM_COUNT = 40
PROPAGATION_ITEM_COUNT = 24
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20260825
RANDOM_NAMES = tuple(f"GATE11_RANDOM_R{i}" for i in range(4))
TF_BASELINE = "TF_BASELINE"
TF_TEXTUAL = "TF_TEXTUAL_CAREFUL"
TF_MEANINGFUL = "TF_MEANINGFUL_L27_D75"
TF_RANDOMS = tuple(f"TF_RANDOM_R{i}" for i in range(4))
TF_CONDITIONS = (TF_BASELINE, TF_TEXTUAL, TF_MEANINGFUL, *TF_RANDOMS)
PROMPT_VARIANTS = (
    "P0_ORDINARY",
    "P1_SOURCE_CAREFUL",
    "P2_SOURCE_DIRECT",
    "P3_DOMAIN_TEXTUAL_CAREFUL",
)
SYSTEM_CAREFUL = (
    "You are a meticulous program tracer. Carefully track every operation, mutation, "
    "intermediate value, branch, and loop. Verify the result before answering. End with "
    "exactly one line in the form FINAL: <answer>."
)
SYSTEM_DIRECT = (
    "Answer the program-output question immediately. Do not trace, deliberate, explain, "
    "or verify. End with exactly one line in the form FINAL: <answer>."
)
SYSTEM_CHARCOUNT_CAREFUL = (
    "You are a meticulous character counter. Scan the entire string systematically, "
    "keep a running count of every occurrence of the requested character, verify the "
    "count with an independent second pass, and end with exactly one line in the form "
    "FINAL: <integer>."
)


def ranked_item_ids(domain: str, item_ids: Sequence[str]) -> list[str]:
    """Rank IDs by the exact frozen SHA-256 rule, independent of outcomes."""

    def digest(item_id: str) -> str:
        value = f"GATE11-SOURCE-AXIS|{domain}|{item_id}".encode()
        return hashlib.sha256(value).hexdigest()

    return sorted(map(str, item_ids), key=lambda item_id: (digest(item_id), item_id))


def select_items(domain: str, item_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    ranked = ranked_item_ids(domain, item_ids)
    if len(ranked) < SOURCE_ITEM_COUNT:
        raise ValueError(f"Gate 11 requires at least {SOURCE_ITEM_COUNT} {domain} items")
    source = ranked[:SOURCE_ITEM_COUNT]
    return source, source[:PROPAGATION_ITEM_COUNT]


def choose_baseline_sequence(
    rows: Mapping[tuple[str, int], Mapping[str, Any]], item_id: str
) -> dict[str, Any]:
    """Apply the frozen rollout-0 then rollout-1 token-ID fallback."""

    for rollout in (0, 1):
        row = rows.get((item_id, rollout))
        if row is None:
            continue
        token_ids = [int(value) for value in row.get("generated_token_ids", [])]
        if token_ids:
            return {
                "available": True,
                "selected_rollout_index": rollout,
                "full_generated_token_count": len(token_ids),
                "continuation_token_ids": token_ids[:SEQUENCE_CAP],
                "continuation_length": min(len(token_ids), SEQUENCE_CAP),
                "truncated_at_cap": len(token_ids) > SEQUENCE_CAP,
                "source_status": row.get("status"),
            }
    return {
        "available": False,
        "selected_rollout_index": None,
        "full_generated_token_count": 0,
        "continuation_token_ids": [],
        "continuation_length": 0,
        "truncated_at_cap": False,
        "source_status": "TOKEN_IDS_MECHANICALLY_ABSENT",
    }


def random_bank(meaningful: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seeds = tuple(stable_seed("GATE11-SHARED-RANDOM-BANK-V1", i) for i in range(4))
    raw = single_layer_random_bank(np.asarray(meaningful, dtype=np.float64), seeds=seeds)
    vectors = {name: raw[f"R{i}"] for i, name in enumerate(RANDOM_NAMES)}
    geometry = bank_geometry(meaningful, vectors)
    if not all(
        geometry[key]
        for key in (
            "unit_norm_pass",
            "meaningful_orthogonality_pass",
            "random_pairwise_orthogonality_pass",
        )
    ):
        raise RuntimeError("Gate 11 random-bank geometry failed")
    records = {
        name: {
            "seed": int(seed),
            "canonical_float64_vector_sha256": vector_sha256(vectors[name]),
            "norm": float(np.linalg.norm(vectors[name])),
            "delta_norm": float(ETA * REFERENCE_SCALE),
        }
        for name, seed in zip(RANDOM_NAMES, seeds, strict=True)
    }
    return vectors, {"seeds": list(seeds), "records": records, "geometry": geometry}


def unit_direction(differences: np.ndarray) -> np.ndarray:
    values = np.asarray(differences, dtype=np.float64)
    mean = values.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("source direction must be finite and nonzero")
    return mean / norm


def _auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Pairwise AUROC with half credit for ties."""

    p = np.asarray(positive, dtype=np.float64)[:, None]
    n = np.asarray(negative, dtype=np.float64)[None, :]
    return float(np.mean((p > n) + 0.5 * (p == n)))


def source_axis_metrics(careful: np.ndarray, direct: np.ndarray) -> dict[str, Any]:
    careful = np.asarray(careful, dtype=np.float64)
    direct = np.asarray(direct, dtype=np.float64)
    direction = unit_direction(careful - direct)
    gaps = (careful - direct) @ direction
    gap_sd = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
    return {
        "direction": direction,
        "gaps": gaps,
        "mean_gap": float(gaps.mean()),
        "median_gap": float(np.median(gaps)),
        "positive_gap_fraction": float(np.mean(gaps > 0)),
        "paired_standardized_effect": float(gaps.mean() / gap_sd) if gap_sd > 0 else None,
        "auroc": _auroc(careful @ direction, direct @ direction),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else None


def relative_dose_geometry(
    ordinary: np.ndarray,
    careful: np.ndarray,
    direct: np.ndarray,
    frozen_direction: np.ndarray,
    delta_d75: float,
) -> dict[str, Any]:
    direction = np.asarray(frozen_direction, dtype=np.float64)
    z0 = np.asarray(ordinary, dtype=np.float64) @ direction
    zc = np.asarray(careful, dtype=np.float64) @ direction
    zd = np.asarray(direct, dtype=np.float64) @ direction
    natural_gap = float(np.mean(zc - zd))
    scale = float(np.std(z0, ddof=1))
    careful_centroid = float(zc.mean())
    before = np.abs(z0 - careful_centroid)
    after = np.abs(z0 + delta_d75 - careful_centroid)
    return {
        "natural_gap": natural_gap,
        "ordinary_projection_scale": scale,
        "delta_d75": float(delta_d75),
        "delta_over_gap": float(delta_d75 / natural_gap) if natural_gap != 0 else None,
        "delta_over_scale": float(delta_d75 / scale) if scale != 0 else None,
        "ordinary_to_careful_distance_before": float(before.mean()),
        "ordinary_to_careful_distance_after": float(after.mean()),
        "fraction_moved_toward_careful_centroid": float(np.mean(after < before)),
    }


def logit_metrics(
    baseline_logits: np.ndarray, condition_logits: np.ndarray, target_token: int | None
) -> dict[str, Any]:
    """Compute frozen full-vocabulary next-token diagnostics."""

    b = np.asarray(baseline_logits, dtype=np.float64)
    c = np.asarray(condition_logits, dtype=np.float64)
    b_logp = b - np.logaddexp.reduce(b)
    c_logp = c - np.logaddexp.reduce(c)
    pb = np.exp(b_logp)
    pc = np.exp(c_logp)
    log_mixture = np.logaddexp(b_logp, c_logp) - np.log(2.0)
    displacement = c - b
    result: dict[str, Any] = {
        "next_token_kl": float(np.sum(pb * (b_logp - c_logp))),
        "symmetric_js": float(
            0.5 * np.sum(pb * (b_logp - log_mixture))
            + 0.5 * np.sum(pc * (c_logp - log_mixture))
        ),
        "logit_l2": float(np.linalg.norm(displacement)),
        "logit_cosine": cosine(b, c),
        "top1_flip": bool(np.argmax(b) != np.argmax(c)),
        "baseline_top1": int(np.argmax(b)),
        "condition_top1": int(np.argmax(c)),
        "displacement_norm": float(np.linalg.norm(displacement)),
    }
    result["target_logprob_shift"] = (
        float(c_logp[target_token] - b_logp[target_token])
        if target_token is not None
        else None
    )
    return result


def classify_components(
    *,
    source_transfer: bool,
    control_gain_shift: bool,
    policy_realization_shift: bool,
    policy_utility_shift: bool,
) -> str:
    mismatches = (
        (not source_transfer),
        control_gain_shift,
        policy_realization_shift,
        policy_utility_shift,
    )
    if sum(mismatches) >= 2:
        return "GATE11_MULTIPLE_DOMAIN_CONDITIONING_FACTORS"
    if not source_transfer:
        return "GATE11_SOURCE_AXIS_DOMAIN_MISMATCH"
    if control_gain_shift:
        return "GATE11_DOWNSTREAM_CONTROL_GAIN_DOMAIN_MISMATCH"
    if policy_realization_shift:
        return "GATE11_POLICY_REALIZATION_DOMAIN_MISMATCH"
    if policy_utility_shift:
        return "GATE11_POLICY_UTILITY_DOMAIN_MISMATCH"
    return "GATE11_POSTMORTEM_INCONCLUSIVE"


__all__ = [name for name in globals() if name.isupper()] + [
    "ranked_item_ids",
    "select_items",
    "choose_baseline_sequence",
    "random_bank",
    "unit_direction",
    "source_axis_metrics",
    "cosine",
    "relative_dose_geometry",
    "logit_metrics",
    "classify_components",
]
