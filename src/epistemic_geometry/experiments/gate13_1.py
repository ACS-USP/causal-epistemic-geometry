"""Pure contracts for Gate 13.1 all-layer causal atlas."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from epistemic_geometry.experiments.gate6_3 import vector_sha256
from epistemic_geometry.reproducibility import stable_digest, stable_seed

EXPERIMENT_ID = "GATE13_1_ALL_LAYER_CAUSAL_ATLAS"
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
REVISION = "f6fae9795746f63c9be8344932f01275f3c63734"
NUM_LAYERS = 34
HIDDEN_SIZE = 4096
DOSE_FRACTIONS = {"D25": 0.25, "D50": 0.50, "D75": 0.75, "D100": 1.0}
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_RESAMPLES = 10_000
SWEEP_NAMESPACE = "GATE13.1-LAYER-DOSE"


def split_development_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(items) != 40:
        raise ValueError("Gate 13.1 development allocation must contain exactly 40 items")
    ranked = sorted(
        (dict(row) for row in items),
        key=lambda row: (
            hashlib.sha256(
                f"{SWEEP_NAMESPACE}|{row['item_id']}".encode()
            ).hexdigest(),
            str(row["item_id"]),
        ),
    )
    sweep, qualification = ranked[:12], ranked[12:]
    ids = [str(row["item_id"]) for row in ranked]
    if len(ids) != len(set(ids)) or len(sweep) != 12 or len(qualification) != 28:
        raise RuntimeError("Gate 13.1 split is incomplete or overlapping")
    return sweep, qualification


def _ordered_conditions(stage: str, item_id: str, conditions: Sequence[str]) -> list[str]:
    return sorted(
        conditions,
        key=lambda condition: stable_digest(
            EXPERIMENT_ID, stage, "CONDITION_ORDER", item_id, condition
        ),
    )


def build_sweep_schedule(item_ids: Sequence[str]) -> list[dict[str, Any]]:
    conditions = ["BASELINE", *(f"MEANINGFUL_L{layer}_D50" for layer in range(NUM_LAYERS))]
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        seed = stable_seed(EXPERIMENT_ID, "ALL_LAYER_SWEEP", item_id)
        for order, condition in enumerate(
            _ordered_conditions("ALL_LAYER_SWEEP", str(item_id), conditions)
        ):
            rows.append(
                {
                    "stage": "ALL_LAYER_SWEEP",
                    "model": MODEL,
                    "item_id": str(item_id),
                    "condition": condition,
                    "rollout_index": 0,
                    "condition_order": order,
                    "seed": seed,
                    "seed_regime": "MATCHED_COUPLING",
                }
            )
    _validate_schedule(rows, expected=len(item_ids) * 35, independent=False)
    return rows


def select_sweep_candidates(
    metrics: Mapping[int, Mapping[str, float]],
) -> tuple[list[int], dict[int, bool]]:
    eligible = {
        layer: bool(
            values["commitment_validity"] >= 0.75
            and values["semantic_evaluability"] >= 0.75
            and values["Q"] >= 0.10
        )
        for layer, values in metrics.items()
    }
    candidates: list[int] = []
    for quartile in np.array_split(np.arange(NUM_LAYERS), 4):
        available = [int(layer) for layer in quartile if eligible.get(int(layer), False)]
        if available:
            candidates.append(max(available, key=lambda layer: (metrics[layer]["Q"], -layer)))
    return candidates, eligible


def _orthogonal_unit(vector: np.ndarray, against: Sequence[np.ndarray]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
    for basis in against:
        unit = np.asarray(basis, dtype=np.float64).reshape(-1)
        value -= float(np.dot(value, unit)) * unit
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Gate 13.1 orthogonalized direction is degenerate")
    return value / norm


def stage_b_nulls(
    meaningful: np.ndarray, paired_differences: np.ndarray, layer: int
) -> dict[str, np.ndarray]:
    isotropic_rng = np.random.default_rng(
        stable_seed(EXPERIMENT_ID, "STAGE_B_ISOTROPIC", layer)
    )
    isotropic = _orthogonal_unit(isotropic_rng.normal(size=len(meaningful)), [meaningful])
    signs_rng = np.random.default_rng(
        stable_seed(EXPERIMENT_ID, "STAGE_B_SHUFFLED", layer)
    )
    signs = signs_rng.choice((-1.0, 1.0), size=len(paired_differences))
    shuffled = _orthogonal_unit(
        (np.asarray(paired_differences) * signs[:, None]).mean(axis=0),
        [meaningful, isotropic],
    )
    return {"ISOTROPIC_NULL": isotropic, "SHUFFLED_NULL": shuffled}


def build_layer_dose_schedule(
    item_ids: Sequence[str], candidate_layers: Sequence[int]
) -> list[dict[str, Any]]:
    conditions = ["BASELINE"]
    for layer in candidate_layers:
        for dose in DOSE_FRACTIONS:
            conditions.extend(
                (
                    f"MEANINGFUL_L{layer}_{dose}",
                    f"ISOTROPIC_NULL_L{layer}_{dose}",
                    f"SHUFFLED_NULL_L{layer}_{dose}",
                )
            )
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        seed = stable_seed(EXPERIMENT_ID, "LAYER_DOSE", item_id)
        for order, condition in enumerate(
            _ordered_conditions("LAYER_DOSE", str(item_id), conditions)
        ):
            rows.append(
                {
                    "stage": "LAYER_DOSE_QUALIFICATION",
                    "model": MODEL,
                    "item_id": str(item_id),
                    "condition": condition,
                    "rollout_index": 0,
                    "condition_order": order,
                    "seed": seed,
                    "seed_regime": "MATCHED_COUPLING",
                }
            )
    expected = len(item_ids) * (1 + len(candidate_layers) * 4 * 3)
    _validate_schedule(rows, expected=expected, independent=False)
    return rows


def cell_eligibility(values: Mapping[str, float]) -> dict[str, bool]:
    checks = {
        "commitment_validity": values["commitment_validity"] >= 0.90,
        "semantic_evaluability": values["semantic_evaluability"] >= 0.90,
        "competence_safety": values["accuracy"] >= values["baseline_accuracy"] - 0.10,
        "semantic_change": values["Q"] >= 0.15,
        "null_mean_specificity": values["Q"] - values["null_mean_Q"] >= 0.05,
        "null_max_specificity": values["Q"] > values["null_max_Q"],
    }
    return checks


def select_layer_dose(
    metrics: Mapping[tuple[int, str], Mapping[str, float]],
    source_effects: Mapping[int, float],
) -> tuple[tuple[int, str] | None, dict[str, dict[str, Any]]]:
    proof: dict[str, dict[str, Any]] = {}
    selected_by_layer: list[tuple[int, str]] = []
    dose_order = list(DOSE_FRACTIONS)
    layers = sorted({layer for layer, _dose in metrics})
    for layer in layers:
        for dose in dose_order:
            key = (layer, dose)
            if key not in metrics:
                continue
            checks = cell_eligibility(metrics[key])
            passed = all(checks.values())
            proof[f"L{layer}_{dose}"] = {"checks": checks, "eligible": passed}
            if passed and not any(candidate[0] == layer for candidate in selected_by_layer):
                selected_by_layer.append(key)
    if not selected_by_layer:
        return None, proof
    selected = max(
        selected_by_layer,
        key=lambda key: (
            metrics[key]["Q"] - metrics[key]["null_mean_Q"],
            metrics[key]["Q"],
            source_effects[key[0]],
            -DOSE_FRACTIONS[key[1]],
            -key[0],
        ),
    )
    return selected, proof


def final_null_bank(
    meaningful: np.ndarray, paired_differences: np.ndarray, layer: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bank: dict[str, np.ndarray] = {}
    bases = [np.asarray(meaningful, dtype=np.float64)]
    records: dict[str, Any] = {}
    for index in range(4):
        name = f"R{index}"
        if index < 2:
            kind = "ISOTROPIC"
            seed = stable_seed(EXPERIMENT_ID, "FINAL_ISOTROPIC", layer, index)
            raw = np.random.default_rng(seed).normal(size=len(meaningful))
        else:
            kind = "SHUFFLED"
            seed = stable_seed(EXPERIMENT_ID, "FINAL_SHUFFLED", layer, index)
            rng = np.random.default_rng(seed)
            signs = rng.choice((-1.0, 1.0), size=len(paired_differences))
            raw = (np.asarray(paired_differences) * signs[:, None]).mean(axis=0)
        value = _orthogonal_unit(raw, bases)
        bank[name] = value
        bases.append(value)
        records[name] = {"kind": kind, "seed": seed, "vector_hash": vector_sha256(value)}
    matrix = np.stack([np.asarray(meaningful), *bank.values()])
    metadata = {
        "records": records,
        "cosine_matrix": (matrix @ matrix.T).tolist(),
        "meaningful_vector_hash": vector_sha256(np.asarray(meaningful)),
    }
    return bank, metadata


def build_final_schedule(item_ids: Sequence[str]) -> list[dict[str, Any]]:
    conditions = (
        "BASELINE",
        "TEXTUAL_CAREFUL",
        "MEANINGFUL_SELECTED",
        "RANDOM_R0",
        "RANDOM_R1",
        "RANDOM_R2",
        "RANDOM_R3",
    )
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        for rollout in (0, 1):
            ordered = sorted(
                conditions,
                key=lambda condition: stable_digest(
                    EXPERIMENT_ID, "FINAL_ORDER", item_id, rollout, condition
                ),
            )
            for order, condition in enumerate(ordered):
                rows.append(
                    {
                        "stage": "FINAL_EVALUATION",
                        "model": MODEL,
                        "item_id": str(item_id),
                        "condition": condition,
                        "rollout_index": rollout,
                        "condition_order": order,
                        "seed": stable_seed(
                            EXPERIMENT_ID, "FINAL", item_id, condition, rollout
                        ),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                    }
                )
    _validate_schedule(rows, expected=len(item_ids) * 14, independent=True)
    return rows


def _validate_schedule(
    rows: Sequence[Mapping[str, Any]], *, expected: int, independent: bool
) -> None:
    if len(rows) != expected:
        raise RuntimeError("Gate 13.1 schedule row count mismatch")
    keys = [
        (
            str(row["stage"]),
            str(row["item_id"]),
            str(row["condition"]),
            int(row["rollout_index"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 13.1 schedule contains duplicate logical keys")
    if independent:
        seeds = [int(row["seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError("Gate 13.1 independent schedule has a seed collision")
