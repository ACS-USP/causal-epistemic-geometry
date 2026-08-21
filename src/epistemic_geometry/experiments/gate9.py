"""Pure, model-free contracts for the Gate 9 selected-D75 evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from epistemic_geometry.experiments.gate6_3 import (
    bank_geometry,
    single_layer_random_bank,
    standardized_delta,
    vector_sha256,
)
from epistemic_geometry.experiments.gate7 import (
    DATASET_REPO,
    DATASET_REVISION,
    LAYER,
    MAX_NEW_TOKENS,
    MODEL,
    MODEL_REVISION,
    PARSER_VERSION,
    REFERENCE_SCALE,
    file_sha256,
    historical_cruxeval_ids,
    task_prompt,
)
from epistemic_geometry.reproducibility import canonical_json, stable_digest, stable_seed

EXPERIMENT_ID = "GATE9_SELECTED_D75_EVALUATION"
SELECTION_NAMESPACE = "GATE9-FRESH-D75-EVALUATION-V1"
BASELINE = "BASELINE"
TEXTUAL = "TEXTUAL_CAREFUL_REFERENCE"
MEANINGFUL = "MEANINGFUL_L27_D75"
RANDOM_NAMES = tuple(f"RANDOM_L27_D75_R{i}" for i in range(4))
CONDITIONS = (BASELINE, TEXTUAL, MEANINGFUL, *RANDOM_NAMES)
ETA = 9.637427952852196
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260823
CONTROLLER_HASH = "e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838"


def _reference_type(reference: str) -> str:
    from epistemic_geometry.benchmarks.external.semantic_v3 import canonicalize_semantic_value

    return str(canonicalize_semantic_value(reference)[0])


def normalize_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    item_id = str(row.get("id", row.get("item_id")))
    prompt = (
        str(row["prompt"]) if "prompt" in row else task_prompt(str(row["code"]), str(row["input"]))
    )
    reference = str(row.get("output", row.get("reference_answer")))
    reference_type = _reference_type(reference)
    prompt_hash = stable_digest("GATE9-TASK-PROMPT", prompt)
    item_hash = stable_digest(
        "GATE9-ITEM", item_id, prompt_hash, reference, "python_literal", DATASET_REVISION
    )
    return {
        "allocation": "GATE9_EVALUATION",
        "item_id": item_id,
        "benchmark": "CRUXEval",
        "subtask": "output_prediction",
        "prompt": prompt,
        "reference_answer": reference,
        "reference_canonical_type": reference_type,
        "evaluator": "python_literal",
        "source_revision": DATASET_REVISION,
        "prompt_hash": prompt_hash,
        "item_hash": item_hash,
        "metadata": {
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "selection_namespace": SELECTION_NAMESPACE,
            "official_id": item_id,
            "reference_canonical_type": reference_type,
        },
    }


def allocate_fresh_items(
    candidates: Sequence[Mapping[str, Any]], historical_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Allocate exactly 100 fresh items and leave all remaining IDs untouched."""

    excluded = set(map(str, historical_ids))
    normalized = [normalize_dataset_row(row) for row in candidates]
    if len({row["item_id"] for row in normalized}) != len(normalized):
        raise RuntimeError("pinned CRUXEval candidates contain duplicate IDs")
    eligible = [row for row in normalized if row["item_id"] not in excluded]
    eligible.sort(
        key=lambda row: (stable_digest(SELECTION_NAMESPACE, row["item_id"]), row["item_id"])
    )
    if len(eligible) < 100:
        raise RuntimeError(f"GATE9_BLOCKED_INSUFFICIENT_FRESH_ITEMS: {len(eligible)} < 100")
    selected = eligible[:100]
    remaining = eligible[100:]
    type_counts: dict[str, int] = {}
    for row in selected:
        value = str(row["reference_canonical_type"])
        type_counts[value] = type_counts.get(value, 0) + 1
    return selected, {
        "requested_n": 100,
        "actual_n": len(selected),
        "eligible_before_allocation": len(eligible),
        "remaining_unallocated_n": len(remaining),
        "remaining_unallocated_ids": [row["item_id"] for row in remaining],
        "historical_excluded_count": len(excluded),
        "historical_exclusion_digest": stable_digest(
            SELECTION_NAMESPACE, "HISTORICAL-EXCLUSION", canonical_json(sorted(excluded))
        ),
        "manifest_hash": stable_digest("GATE9-EVALUATION-MANIFEST", canonical_json(selected)),
        "reference_type_distribution": dict(sorted(type_counts.items())),
        "selection_namespace": SELECTION_NAMESPACE,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
    }


def gate9_random_bank(meaningful: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seeds = tuple(stable_seed("GATE9-L27-RANDOM-BANK-V1", index) for index in range(4))
    short = single_layer_random_bank(meaningful, seeds=seeds)
    bank = {name: short[f"R{index}"] for index, name in enumerate(RANDOM_NAMES)}
    geometry = bank_geometry(meaningful, bank)
    required = (
        "unit_norm_pass",
        "meaningful_orthogonality_pass",
        "random_pairwise_orthogonality_pass",
    )
    if not all(geometry[key] for key in required):
        raise RuntimeError(f"Gate 9 random-bank geometry failed: {geometry}")
    records = {
        name: {
            "seed": int(seed),
            "vector_sha256": vector_sha256(bank[name]),
            "norm": float(np.linalg.norm(bank[name])),
            "delta_norm": float(
                np.linalg.norm(
                    standardized_delta(bank[name], eta=ETA, reference_scale=REFERENCE_SCALE)
                )
            ),
        }
        for name, seed in zip(RANDOM_NAMES, seeds, strict=True)
    }
    return bank, {"seeds": list(seeds), "records": records, "geometry": geometry}


def build_schedule(item_ids: Sequence[str]) -> list[dict[str, Any]]:
    logical: list[dict[str, Any]] = []
    for item_index, item_id in enumerate(item_ids):
        for rollout in (0, 1):
            order = sorted(
                CONDITIONS,
                key=lambda condition: (
                    stable_digest(SELECTION_NAMESPACE, "ORDER", item_id, rollout, condition),
                    condition,
                ),
            )
            for order_index, condition in enumerate(order):
                logical.append(
                    {
                        "phase": "GATE9_SELECTED_D75_EVALUATION",
                        "item_index": item_index,
                        "item_id": str(item_id),
                        "condition": condition,
                        "condition_order": order_index,
                        "rollout_index": rollout,
                        "seed": stable_seed(EXPERIMENT_ID, item_id, condition, rollout),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                    }
                )
    keys = [(row["item_id"], row["condition"], row["rollout_index"]) for row in logical]
    seeds = [int(row["seed"]) for row in logical]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 9 schedule contains duplicate logical keys")
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Gate 9 independent seed bank contains a collision")
    return logical


def classify_gate9(
    *,
    baseline: Mapping[str, float],
    controller: Mapping[str, float],
    controller_estimands: Mapping[str, float],
    random_summary: Mapping[str, Mapping[str, float]],
    bootstrap: Mapping[str, Mapping[str, float]],
    loo_sign_stable: Mapping[str, bool],
    controller_style_replicated: bool,
    source_replicated: bool = True,
) -> tuple[str, dict[str, Any]]:
    commitment = bool(
        controller["commitment_validity"] >= 0.90
        and controller["commitment_validity"] >= baseline["commitment_validity"] - 0.05
    )
    evaluability = bool(
        controller["semantic_evaluability"] >= 0.90
        and controller["semantic_evaluability"] >= baseline["semantic_evaluability"] - 0.05
    )
    competence = bool(controller["accuracy"] >= baseline["accuracy"] - 0.10)
    point = controller_estimands
    above_mean = {
        metric: point[metric] > random_summary[metric]["mean"] for metric in ("G", "C", "D")
    }
    above_max = {
        metric: point[metric] > random_summary[metric]["max"] for metric in ("G", "C", "D")
    }
    minimum = bool(
        commitment
        and evaluability
        and competence
        and all(point[metric] > 0 for metric in ("G", "C", "D"))
        and all(above_mean.values())
        and point["rescue"] > point["damage"]
        and sum(above_max.values()) >= 2
    )
    strong_names = (
        "meaningful:accuracy_change",
        "meaningful:G",
        "meaningful:C",
        "meaningful:G_minus_random_mean",
        "meaningful:C_minus_random_mean",
    )
    strong = bool(
        commitment
        and evaluability
        and competence
        and point["G"] >= 0.10
        and point["C"] >= 0.05
        and point["D"] >= 0.08
        and point["G"] - random_summary["G"]["mean"] >= 0.08
        and point["C"] - random_summary["C"]["mean"] >= 0.05
        and point["D"] - random_summary["D"]["mean"] >= 0.05
        and all(above_max.values())
        and point["rescue"] > point["damage"]
        and controller["accuracy"] - baseline["accuracy"] >= 0.05
        and all(float(bootstrap[name]["q025"]) > 0 for name in strong_names)
        and all(loo_sign_stable.get(metric, False) for metric in ("accuracy_change", "G", "C"))
    )
    movement = bool(
        commitment
        and evaluability
        and competence
        and point["D"] > 0
        and above_mean["D"]
        and above_max["D"]
    )
    if not source_replicated:
        classification = "GATE9_SOURCE_POLICY_NOT_REPLICATED"
    elif not (commitment and evaluability and competence):
        classification = "GATE9_SELECTED_DOSE_DESTRUCTIVE"
    elif strong:
        classification = "GATE9_STRONG_SAFE_SELECTED_DOSE_REPLICATION"
    elif minimum:
        classification = "GATE9_MINIMUM_SAFE_SELECTED_DOSE_SIGNAL"
    elif movement:
        classification = "GATE9_SAFE_ERROR_PROFILE_MOVEMENT_ONLY"
    elif controller_style_replicated:
        classification = "GATE9_CAREFUL_STYLE_CONTROL_WITHOUT_ERROR_CONTROL"
    else:
        classification = "GATE9_NO_SELECTED_DOSE_EFFECT"
    return classification, {
        "commitment_validity_guard": commitment,
        "semantic_evaluability_guard": evaluability,
        "competence_guard": competence,
        "source_policy_replicated": source_replicated,
        "minimum_safe_specific_control": minimum,
        "strong_safe_error_control_replication": strong,
        "safe_error_profile_movement": movement,
        "above_random_mean": above_mean,
        "above_random_max": above_max,
    }


__all__ = [
    "BASELINE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONDITIONS",
    "CONTROLLER_HASH",
    "DATASET_REPO",
    "DATASET_REVISION",
    "ETA",
    "EXPERIMENT_ID",
    "LAYER",
    "MAX_NEW_TOKENS",
    "MEANINGFUL",
    "MODEL",
    "MODEL_REVISION",
    "PARSER_VERSION",
    "RANDOM_NAMES",
    "REFERENCE_SCALE",
    "SELECTION_NAMESPACE",
    "TEXTUAL",
    "allocate_fresh_items",
    "build_schedule",
    "classify_gate9",
    "file_sha256",
    "gate9_random_bank",
    "historical_cruxeval_ids",
    "normalize_dataset_row",
    "vector_sha256",
]
