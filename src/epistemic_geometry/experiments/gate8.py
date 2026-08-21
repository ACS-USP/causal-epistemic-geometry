"""Pure, model-free contracts for Gate 8 L27 dose calibration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
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
    REFERENCE_SCALE,
    historical_cruxeval_ids,
    task_prompt,
)
from epistemic_geometry.reproducibility import canonical_json, stable_digest, stable_seed

EXPERIMENT_ID = "GATE8_L27_DOSE_CALIBRATION"
SELECTION_NAMESPACE = "GATE8-L27-DOSE-CALIBRATION-V1"
BASELINE = "BASELINE"
TEXTUAL = "TEXTUAL_CAREFUL_REFERENCE"
MEANINGFUL_VECTOR = "BEST_SINGLE_MEAN_PLUS"
PARSER_VERSION = "external-semantic-v3"
ETA_FULL = 12.849903937136261
DOSE_FRACTIONS = {"D25": 0.25, "D50": 0.50, "D75": 0.75, "D100": 1.0}
MEANINGFUL_CONDITIONS = tuple(f"MEAN_{dose}" for dose in DOSE_FRACTIONS)
RANDOM_VECTOR_NAMES = tuple(f"GATE8_RANDOM_R{i}" for i in range(4))
RANDOM_CONDITIONS = tuple(
    f"RANDOM_R{index}_{dose}" for dose in DOSE_FRACTIONS for index in range(4)
)
CONDITIONS = (BASELINE, TEXTUAL, *MEANINGFUL_CONDITIONS, *RANDOM_CONDITIONS)
SELECTABLE_DOSES = ("D25", "D50", "D75")
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20260822


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    prompt_hash = stable_digest("GATE8-TASK-PROMPT", prompt)
    item_hash = stable_digest(
        "GATE8-ITEM", item_id, prompt_hash, reference, "python_literal", DATASET_REVISION
    )
    return {
        "allocation": "GATE8_DOSE_CALIBRATION",
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


def allocate_calibration_items(
    candidates: Sequence[Mapping[str, Any]], historical_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze exactly 50 items while leaving at least 100 unseen and unallocated."""

    excluded = set(map(str, historical_ids))
    normalized = [normalize_dataset_row(row) for row in candidates]
    if len({row["item_id"] for row in normalized}) != len(normalized):
        raise RuntimeError("pinned CRUXEval candidates contain duplicate IDs")
    eligible = [row for row in normalized if row["item_id"] not in excluded]
    eligible.sort(
        key=lambda row: (stable_digest(SELECTION_NAMESPACE, row["item_id"]), row["item_id"])
    )
    if len(eligible) < 150:
        raise RuntimeError(f"GATE8_BLOCKED_INSUFFICIENT_FRESH_ITEMS: {len(eligible)} < 150")
    selected = eligible[:50]
    remaining = len(eligible) - len(selected)
    if remaining < 100:
        raise RuntimeError(f"GATE8_BLOCKED_INSUFFICIENT_FRESH_ITEMS: {remaining} < 100 remaining")
    type_counts: dict[str, int] = {}
    for row in selected:
        value = str(row["reference_canonical_type"])
        type_counts[value] = type_counts.get(value, 0) + 1
    return selected, {
        "requested_n": 50,
        "actual_n": len(selected),
        "eligible_before_allocation": len(eligible),
        "remaining_unallocated_n": remaining,
        "historical_excluded_count": len(excluded),
        "historical_exclusion_digest": stable_digest(
            SELECTION_NAMESPACE, "HISTORICAL-EXCLUSION", canonical_json(sorted(excluded))
        ),
        "manifest_hash": stable_digest("GATE8-CALIBRATION-MANIFEST", canonical_json(selected)),
        "reference_type_distribution": dict(sorted(type_counts.items())),
        "selection_namespace": SELECTION_NAMESPACE,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "future_evaluation_ids_allocated": False,
    }


def gate8_random_bank(meaningful: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seeds = tuple(stable_seed("GATE8-L27-RANDOM-BANK-V1", index) for index in range(4))
    short = single_layer_random_bank(meaningful, seeds=seeds)
    bank = {name: short[f"R{index}"] for index, name in enumerate(RANDOM_VECTOR_NAMES)}
    geometry = bank_geometry(meaningful, bank)
    if not all(
        geometry[key]
        for key in (
            "unit_norm_pass",
            "meaningful_orthogonality_pass",
            "random_pairwise_orthogonality_pass",
        )
    ):
        raise RuntimeError(f"Gate 8 random-bank geometry failed: {geometry}")
    records = {
        name: {
            "seed": int(seed),
            "vector_sha256": vector_sha256(bank[name]),
            "norm": float(np.linalg.norm(bank[name])),
            "full_dose_delta_norm": float(
                np.linalg.norm(
                    standardized_delta(bank[name], eta=ETA_FULL, reference_scale=REFERENCE_SCALE)
                )
            ),
        }
        for name, seed in zip(RANDOM_VECTOR_NAMES, seeds, strict=True)
    }
    return bank, {"seeds": list(seeds), "records": records, "geometry": geometry}


def condition_spec(condition: str) -> dict[str, Any]:
    if condition == BASELINE:
        return {"kind": "baseline", "dose": "D0", "fraction": 0.0, "eta": 0.0}
    if condition == TEXTUAL:
        return {"kind": "textual", "dose": "D0", "fraction": 0.0, "eta": 0.0}
    if condition.startswith("MEAN_"):
        dose = condition.removeprefix("MEAN_")
        return {
            "kind": "meaningful",
            "vector": MEANINGFUL_VECTOR,
            "dose": dose,
            "fraction": DOSE_FRACTIONS[dose],
            "eta": ETA_FULL * DOSE_FRACTIONS[dose],
        }
    if condition.startswith("RANDOM_R"):
        head, dose = condition.rsplit("_", 1)
        index = int(head.removeprefix("RANDOM_R"))
        return {
            "kind": "random",
            "vector": RANDOM_VECTOR_NAMES[index],
            "dose": dose,
            "fraction": DOSE_FRACTIONS[dose],
            "eta": ETA_FULL * DOSE_FRACTIONS[dose],
        }
    raise ValueError(f"unknown Gate 8 condition: {condition}")


def build_schedule(item_ids: Sequence[str]) -> list[dict[str, Any]]:
    logical: list[dict[str, Any]] = []
    for item_index, item_id in enumerate(item_ids):
        for rollout in (0, 1):
            shared_seed = stable_seed(EXPERIMENT_ID, item_id, rollout)
            condition_order = sorted(
                CONDITIONS,
                key=lambda condition: (
                    stable_digest(SELECTION_NAMESPACE, "ORDER", item_id, rollout, condition),
                    condition,
                ),
            )
            for order_index, condition in enumerate(condition_order):
                spec = condition_spec(condition)
                logical.append(
                    {
                        "phase": "GATE8_DOSE_CALIBRATION",
                        "item_index": item_index,
                        "item_id": str(item_id),
                        "condition": condition,
                        "condition_order": order_index,
                        "rollout_index": rollout,
                        "seed": shared_seed,
                        "seed_block_id": stable_digest(EXPERIMENT_ID, item_id, rollout),
                        "seed_regime": "MATCHED_COUPLING_CALIBRATION",
                        "dose": spec["dose"],
                        "dose_fraction": spec["fraction"],
                        "eta": spec["eta"],
                    }
                )
    keys = [(row["item_id"], row["condition"], row["rollout_index"]) for row in logical]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 8 schedule contains duplicate logical keys")
    for item_id in item_ids:
        for rollout in (0, 1):
            block = [
                row
                for row in logical
                if row["item_id"] == item_id and row["rollout_index"] == rollout
            ]
            if len(block) != len(CONDITIONS) or len({row["seed"] for row in block}) != 1:
                raise RuntimeError("Gate 8 matched seed block is malformed")
    return logical


def classify_source(baseline: Mapping[str, float], textual: Mapping[str, float]) -> str:
    replicated = bool(
        textual["commitment_validity"] >= 0.90
        and textual["semantic_evaluability"] >= 0.90
        and textual["mean_tokens"] >= 1.5 * baseline["mean_tokens"]
        and textual["median_tokens"] >= baseline["median_tokens"] + 10
    )
    return "CAREFUL_SOURCE_REPLICATED" if replicated else "CAREFUL_SOURCE_NOT_REPLICATED"


def dose_eligibility(
    *,
    baseline: Mapping[str, float],
    dose: Mapping[str, float],
    random_q: Mapping[str, float],
    source_replicated: bool,
) -> dict[str, bool]:
    commitment = bool(
        dose["commitment_validity"] >= 0.90
        and dose["commitment_validity"] >= baseline["commitment_validity"] - 0.05
    )
    evaluability = bool(
        dose["semantic_evaluability"] >= 0.90
        and dose["semantic_evaluability"] >= baseline["semantic_evaluability"] - 0.05
    )
    competence = bool(dose["accuracy"] >= baseline["accuracy"] - 0.10)
    first_stage = bool(
        dose["Q"] >= 0.15
        and dose["Q"] - random_q["mean"] >= 0.05
        and dose["Q"] > random_q["max"]
        and dose["rho_tokens"] >= 0.25
        and dose["rho_tokens"] <= 1.25
    )
    return {
        "source_replicated": source_replicated,
        "commitment_validity": commitment,
        "semantic_evaluability": evaluability,
        "competence_safety": competence,
        "behavioral_first_stage": first_stage,
        "eligible": bool(
            source_replicated and commitment and evaluability and competence and first_stage
        ),
    }


def select_dose(
    eligibility: Mapping[str, Mapping[str, bool]],
) -> tuple[str | None, str]:
    for dose in SELECTABLE_DOSES:
        if eligibility[dose]["eligible"]:
            return dose, "GATE8_SAFE_LOWER_DOSE_SELECTED"
    if eligibility["D100"]["eligible"]:
        return None, "GATE8_ORIGINAL_DOSE_ONLY_SPECIFIC"
    specific = [dose for dose in DOSE_FRACTIONS if eligibility[dose]["behavioral_first_stage"]]
    if specific and all(
        not (
            eligibility[dose]["commitment_validity"]
            and eligibility[dose]["semantic_evaluability"]
            and eligibility[dose]["competence_safety"]
        )
        for dose in specific
    ):
        return None, "GATE8_EFFECT_VALIDITY_TRADEOFF_CONFIRMED"
    return None, "GATE8_LOWER_DOSES_NONSPECIFIC_OR_INERT"


__all__ = [
    "BASELINE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONDITIONS",
    "DOSE_FRACTIONS",
    "ETA_FULL",
    "EXPERIMENT_ID",
    "MEANINGFUL_CONDITIONS",
    "MEANINGFUL_VECTOR",
    "PARSER_VERSION",
    "RANDOM_CONDITIONS",
    "RANDOM_VECTOR_NAMES",
    "SELECTABLE_DOSES",
    "SELECTION_NAMESPACE",
    "TEXTUAL",
    "allocate_calibration_items",
    "build_schedule",
    "classify_source",
    "condition_spec",
    "dose_eligibility",
    "file_sha256",
    "gate8_random_bank",
    "historical_cruxeval_ids",
    "select_dose",
    "vector_sha256",
]
