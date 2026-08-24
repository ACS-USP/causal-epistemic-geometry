"""Frozen contracts for the cross-model Q1 fixed-controller confirmation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from epistemic_geometry.experiments.gate6_3 import vector_sha256
from epistemic_geometry.experiments.gate6_3_v3 import audit_two_rollout_estimands
from epistemic_geometry.reproducibility import stable_digest, stable_seed

EXPERIMENT_ID = "Q1_CONFIRMATORY_FIXED_CONTROLLERS"
CONDITIONS = (
    "BASELINE",
    "TEXTUAL_CAREFUL",
    "MEANINGFUL_FIXED",
    "RANDOM_R0",
    "RANDOM_R1",
    "RANDOM_R2",
    "RANDOM_R3",
)
RANDOM_NAMES = ("RANDOM_R0", "RANDOM_R1", "RANDOM_R2", "RANDOM_R3")
N_ITEMS = 57
ROLLOUTS = 2
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEEDS = {"Qwen": 2026091101, "Ministral": 2026091102}


def _orthogonal_unit(vector: np.ndarray, bases: Sequence[np.ndarray]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
    for basis in bases:
        unit = np.asarray(basis, dtype=np.float64).reshape(-1)
        value -= float(np.dot(value, unit)) * unit
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("confirmatory null direction is degenerate")
    return value / norm


def build_null_bank(
    meaningful: np.ndarray,
    paired_differences: np.ndarray,
    *,
    model_role: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build two isotropic and two construction-matched shuffled nulls."""

    direction = np.asarray(meaningful, dtype=np.float64).reshape(-1)
    direction /= np.linalg.norm(direction)
    pairs = np.asarray(paired_differences, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != len(direction) or len(pairs) < 2:
        raise ValueError("paired differences must have shape (n_pairs, hidden_size)")
    bank: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    bases = [direction]
    for index, condition in enumerate(RANDOM_NAMES):
        if index < 2:
            kind = "ISOTROPIC"
            seed = stable_seed(EXPERIMENT_ID, model_role, kind, index)
            raw = np.random.default_rng(seed).normal(size=len(direction))
        else:
            kind = "CONSTRUCTION_MATCHED_SIGN_SHUFFLED"
            seed = stable_seed(EXPERIMENT_ID, model_role, kind, index)
            signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=len(pairs))
            raw = (pairs * signs[:, None]).mean(axis=0)
        value = _orthogonal_unit(raw, bases)
        bank[condition] = value
        bases.append(value)
        records[condition] = {
            "kind": kind,
            "seed": seed,
            "canonical_float64_vector_sha256": vector_sha256(value),
        }
    matrix = np.stack([direction, *bank.values()])
    cosine = matrix @ matrix.T
    if not np.allclose(np.diag(cosine), 1.0, atol=1e-10, rtol=0):
        raise AssertionError("confirmatory null bank contains a non-unit vector")
    if np.max(np.abs(cosine - np.eye(5))) > 1e-6:
        raise AssertionError("confirmatory null bank is not prospectively orthogonal")
    return bank, {
        "model_role": model_role,
        "meaningful_vector_hash": vector_sha256(direction),
        "paired_difference_count": len(pairs),
        "records": records,
        "cosine_matrix": cosine.tolist(),
        "construction_outcomes_used": False,
    }


def build_schedule(item_ids: Sequence[str], *, model_role: str) -> list[dict[str, Any]]:
    """Build the complete independent, outcome-free model schedule."""

    ids = tuple(str(value) for value in item_ids)
    if len(ids) != N_ITEMS or len(set(ids)) != N_ITEMS:
        raise ValueError("confirmatory schedule requires exactly 57 unique item IDs")
    rows: list[dict[str, Any]] = []
    for item_id in ids:
        for rollout in range(ROLLOUTS):
            ordered = sorted(
                CONDITIONS,
                key=lambda condition: stable_digest(
                    EXPERIMENT_ID, model_role, "CONDITION_ORDER", item_id, rollout, condition
                ),
            )
            for order, condition in enumerate(ordered):
                rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "model_role": model_role,
                        "item_id": item_id,
                        "condition": condition,
                        "rollout_index": rollout,
                        "condition_order": order,
                        "seed": stable_seed(
                            EXPERIMENT_ID, model_role, item_id, condition, rollout
                        ),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                    }
                )
    validate_schedule(rows, model_role=model_role, item_ids=ids)
    return rows


def validate_schedule(
    rows: Sequence[Mapping[str, Any]], *, model_role: str, item_ids: Sequence[str]
) -> None:
    expected = len(item_ids) * len(CONDITIONS) * ROLLOUTS
    if len(rows) != expected:
        raise RuntimeError("confirmatory schedule row count mismatch")
    keys = [
        (
            str(row["model_role"]),
            str(row["item_id"]),
            str(row["condition"]),
            int(row["rollout_index"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("confirmatory schedule contains duplicate logical keys")
    if {str(row["model_role"]) for row in rows} != {model_role}:
        raise RuntimeError("confirmatory schedule mixes model roles")
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("confirmatory independent schedule has a seed collision")
    if {str(row["condition"]) for row in rows} != set(CONDITIONS):
        raise RuntimeError("confirmatory schedule condition set mismatch")


def completed_keys(
    rows: Sequence[Mapping[str, Any]], *, source_commit: str | None = None
) -> set[tuple[str, str, str, int]]:
    keys = []
    for row in rows:
        if source_commit is not None and row.get("confirmatory_source_commit") != source_commit:
            raise RuntimeError("confirmatory journal mixes source commits")
        keys.append(
            (
                str(row["model_role"]),
                str(row["item_id"]),
                str(row["condition"]),
                int(row["rollout_index"]),
            )
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError("confirmatory journal contains duplicate logical keys")
    return set(keys)


def error_arrays(
    rows: Sequence[Mapping[str, Any]], item_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    lookup = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): int(
            not bool(row["correct"])
        )
        for row in rows
    }
    expected = len(item_ids) * len(CONDITIONS) * ROLLOUTS
    if len(lookup) != expected:
        raise RuntimeError("confirmatory rows are incomplete or duplicated")
    return {
        condition: np.asarray(
            [
                [lookup[(str(item), condition, rollout)] for rollout in range(ROLLOUTS)]
                for item in item_ids
            ],
            dtype=np.int8,
        )
        for condition in CONDITIONS
    }


def primary_estimands(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    baseline = np.asarray(arrays["BASELINE"])
    return {
        condition: audit_two_rollout_estimands(baseline, np.asarray(arrays[condition]))
        for condition in CONDITIONS[1:]
    }


def classify_model(
    *,
    summaries: Mapping[str, Mapping[str, float]],
    estimands: Mapping[str, Mapping[str, float]],
    intervals: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    meaningful = estimands["MEANINGFUL_FIXED"]
    null_c = [estimands[name]["C"] for name in RANDOM_NAMES]
    checks = {
        "P1_C_interval_lower_gt_zero": intervals["C_meaningful"]["q025"] > 0,
        "P2_delta_C_interval_lower_gt_zero": intervals["delta_C_nullmean"]["q025"] > 0,
        "P2_C_above_null_max": meaningful["C"] > max(null_c),
        "S1_commitment_validity": summaries["MEANINGFUL_FIXED"]["commitment_validity"]
        >= summaries["BASELINE"]["commitment_validity"] - 0.05,
        "S2_semantic_evaluability": summaries["MEANINGFUL_FIXED"]["semantic_evaluability"]
        >= summaries["BASELINE"]["semantic_evaluability"] - 0.05,
        "S3_competence": summaries["MEANINGFUL_FIXED"]["accuracy"]
        >= summaries["BASELINE"]["accuracy"] - 0.10,
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def cross_model_classification(qwen_pass: bool, ministral_pass: bool) -> str:
    if qwen_pass and ministral_pass:
        return "Q1_CONFIRMATORY_CROSS_MODEL_PASS"
    if qwen_pass:
        return "Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL"
    if ministral_pass:
        return "Q1_CONFIRMATORY_QWEN_FAIL_MINISTRAL_PASS"
    return "Q1_CONFIRMATORY_BOTH_FAIL"


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEEDS",
    "CONDITIONS",
    "EXPERIMENT_ID",
    "N_ITEMS",
    "RANDOM_NAMES",
    "build_null_bank",
    "build_schedule",
    "classify_model",
    "completed_keys",
    "cross_model_classification",
    "error_arrays",
    "primary_estimands",
    "validate_schedule",
]
