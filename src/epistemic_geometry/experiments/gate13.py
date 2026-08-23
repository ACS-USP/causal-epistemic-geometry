"""Pure contracts for Gate 13 cross-model Ministral-3 replication."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.gate6_3 import vector_sha256
from epistemic_geometry.experiments.gate8 import dose_eligibility
from epistemic_geometry.reproducibility import stable_digest, stable_seed

EXPERIMENT_ID = "GATE13_CROSS_MODEL_MINISTRAL3"
SELECTION_NAMESPACE = "GATE13-CROSS-MODEL"
DATASET_REPO = "cruxeval-org/cruxeval"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
PRIMARY_MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
PRIMARY_REVISION = "f6fae9795746f63c9be8344932f01275f3c63734"
FALLBACK_MODEL = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
FALLBACK_REVISION = "3cea74c1ebaf5ce5f5a2553de470e2ceab825142"
PARSER_VERSION = "external-semantic-v3"
MAX_NEW_TOKENS = 4096
NUM_LAYERS = 34
HIDDEN_SIZE = 4096
MODEL_LAYER_PATH = "model.language_model.layers"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260824

SOURCE_CAREFUL = (
    "You are a meticulous program tracer. Carefully track every operation, mutation, "
    "intermediate value, branch, and loop. Verify the result before answering. End with "
    "exactly one line in the form FINAL: <answer>."
)
SOURCE_DIRECT = (
    "Answer the program-output question immediately. Do not trace, deliberate, explain, or "
    "verify. End with exactly one line in the form FINAL: <answer>."
)
CAREFUL_CONCISE = (
    "You are a meticulous program tracer. Carefully track every operation, mutation, "
    "intermediate value, branch, and loop, and verify the result internally. Keep the "
    "response concise. End with exactly one line in the form FINAL: <answer>."
)
VERBOSE_DIRECT = (
    "Answer the program-output question directly without performing an independent "
    "verification pass. You may explain the answer in detail, but do not deliberately "
    "retrace the full computation. End with exactly one line in the form FINAL: <answer>."
)

SCREEN_CONDITIONS = (
    "BASELINE",
    "SOURCE_DIRECT",
    "SOURCE_CAREFUL",
    "CAREFUL_CONCISE",
    "VERBOSE_DIRECT",
)
ALLOCATION_COUNTS = (
    ("PRIMARY_8B_SUBSTRATE_SCREEN", 30),
    ("FALLBACK_14B_SUBSTRATE_SCREEN", 30),
    ("SOURCE_CONSTRUCTION", 64),
    ("SOURCE_VALIDATION", 32),
    ("LAYER_FIRST_STAGE", 24),
    ("DOSE_CALIBRATION", 40),
    ("FINAL_EVALUATION", 100),
)
DOSE_FRACTIONS = {"D25": 0.25, "D50": 0.50, "D75": 0.75, "D100": 1.0}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_record(row: Mapping[str, Any]) -> dict[str, Any]:
    item_id = str(row["item_id"])
    prompt = str(row["prompt"])
    reference = str(row["reference_answer"])
    from epistemic_geometry.benchmarks.external.semantic_v3 import canonicalize_semantic_value

    reference_type = str(canonicalize_semantic_value(reference)[0])
    return {
        "item_id": item_id,
        "benchmark": "CRUXEval",
        "subtask": "output_prediction",
        "prompt": prompt,
        "reference_answer": reference,
        "reference_canonical_type": reference_type,
        "evaluator": "python_literal",
        "source_revision": str(row.get("source_revision", DATASET_REVISION)),
        "prompt_hash": stable_digest("GATE13-TASK-PROMPT", prompt),
        "item_hash": stable_digest(
            "GATE13-REUSED-DEVELOPMENT-ITEM", item_id, prompt, reference, DATASET_REVISION
        ),
    }


def build_reused_development_pool(
    records: Sequence[Mapping[str, Any]], untouched_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Canonicalize outcome-free historical manifest records and protect untouched IDs."""

    untouched = set(map(str, untouched_ids))
    by_id: dict[str, dict[str, Any]] = {}
    for raw in records:
        row = _canonical_record(raw)
        if row["item_id"] in untouched:
            raise RuntimeError("Gate 13 development pool intersects the 57 untouched IDs")
        previous = by_id.get(row["item_id"])
        if previous is not None and (
            previous["prompt"] != row["prompt"]
            or previous["reference_answer"] != row["reference_answer"]
        ):
            raise RuntimeError(f"conflicting historical manifest content for {row['item_id']}")
        by_id[row["item_id"]] = row
    ranked = sorted(
        by_id.values(),
        key=lambda row: (
            hashlib.sha256(f"GATE13-CROSS-MODEL|{row['item_id']}".encode()).hexdigest(),
            row["item_id"],
        ),
    )
    if len(ranked) < sum(count for _name, count in ALLOCATION_COUNTS):
        raise RuntimeError("GATE13_REUSED_DEVELOPMENT_POOL is too small")
    return ranked


def allocate_pool(pool: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    allocations: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name, count in ALLOCATION_COUNTS:
        rows = [dict(row, allocation=name) for row in pool[offset : offset + count]]
        if len(rows) != count:
            raise RuntimeError(f"Gate 13 allocation {name} is incomplete")
        allocations[name] = rows
        offset += count
    ids = [row["item_id"] for rows in allocations.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Gate 13 allocations overlap")
    return allocations


def build_screen_schedule(item_ids: Sequence[str], model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        for rollout in (0, 1):
            order = sorted(
                SCREEN_CONDITIONS,
                key=lambda condition: stable_digest(
                    SELECTION_NAMESPACE, "SCREEN-ORDER", model_name, item_id, rollout, condition
                ),
            )
            for order_index, condition in enumerate(order):
                rows.append(
                    {
                        "stage": "SUBSTRATE_SCREEN",
                        "model": model_name,
                        "item_id": item_id,
                        "condition": condition,
                        "rollout_index": rollout,
                        "condition_order": order_index,
                        "seed": stable_seed(
                            EXPERIMENT_ID, "SCREEN", model_name, item_id, condition, rollout
                        ),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                    }
                )
    _validate_unique_schedule(rows, independent_seeds=True)
    return rows


def _validate_unique_schedule(
    rows: Sequence[Mapping[str, Any]], *, independent_seeds: bool
) -> None:
    keys = [
        (
            str(row["stage"]),
            str(row["model"]),
            str(row["item_id"]),
            str(row["condition"]),
            int(row["rollout_index"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 13 schedule contains duplicate logical keys")
    if independent_seeds:
        seeds = [int(row["seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError("Gate 13 independent schedule contains a seed collision")


def classify_substrate(metrics: Mapping[str, Mapping[str, float]]) -> tuple[str, dict[str, bool]]:
    baseline = metrics["BASELINE"]
    direct = metrics["SOURCE_DIRECT"]
    careful = metrics["SOURCE_CAREFUL"]
    mechanical = bool(
        baseline["commitment_validity"] >= 0.95
        and baseline["semantic_evaluability"] >= 0.95
        and direct["commitment_validity"] >= 0.95
        and careful["commitment_validity"] >= 0.95
        and max(
            baseline.get("truncation", 0.0),
            direct.get("truncation", 0.0),
            careful.get("truncation", 0.0),
        )
        <= 0.05
    )
    source_accuracy = bool(
        careful["accuracy"] >= direct["accuracy"] + 0.05
        and careful["accuracy"] >= baseline["accuracy"] + 0.03
    )
    source_behavior = bool(
        careful["mean_tokens"] >= 1.25 * direct["mean_tokens"]
        or careful["median_tokens"] >= direct["median_tokens"] + 20
        or metrics["SOURCE_CAREFUL"].get("semantic_change_vs_direct", 0.0) >= 0.15
    )
    floor = baseline["accuracy"] < 0.25
    ceiling = baseline["accuracy"] > 0.85
    passed = bool(mechanical and not floor and not ceiling and source_accuracy and source_behavior)
    if passed:
        classification = "MINISTRAL3_8B_SUBSTRATE_PASS"
    elif not mechanical:
        classification = "MINISTRAL3_8B_MECHANICAL_FAILURE"
    elif floor:
        classification = "MINISTRAL3_8B_COMPETENCE_FLOOR"
    elif ceiling:
        classification = "MINISTRAL3_8B_COMPETENCE_CEILING"
    else:
        classification = "MINISTRAL3_8B_SOURCE_POLICY_NOT_USEFUL"
    return classification, {
        "mechanical": mechanical,
        "competence_floor": floor,
        "competence_ceiling": ceiling,
        "source_accuracy": source_accuracy,
        "source_behavior": source_behavior,
        "pass": passed,
    }


def source_atlas(
    construction_careful: np.ndarray,
    construction_direct: np.ndarray,
    validation_careful: np.ndarray,
    validation_direct: np.ndarray,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    """Construct paired-mean directions and held-out source-only layer metrics."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            construction_careful,
            construction_direct,
            validation_careful,
            validation_direct,
        )
    )
    if any(value.ndim != 3 for value in arrays):
        raise ValueError("source activations must have shape [items, layers, hidden]")
    if any(value.shape[1:] != arrays[0].shape[1:] for value in arrays[1:]):
        raise ValueError("source activation layer/hidden shapes differ")
    train_diff = arrays[0] - arrays[1]
    valid_diff = arrays[2] - arrays[3]
    directions: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for layer in range(arrays[0].shape[1]):
        mean_difference = train_diff[:, layer, :].mean(axis=0)
        norm = float(np.linalg.norm(mean_difference))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"non-finite/zero source direction at layer {layer}")
        direction = mean_difference / norm
        directions[layer] = direction
        careful_projection = arrays[2][:, layer, :] @ direction
        direct_projection = arrays[3][:, layer, :] @ direction
        gaps = valid_diff[:, layer, :] @ direction
        gap_sd = float(np.std(gaps, ddof=1))
        effect = float(np.mean(gaps) / gap_sd) if gap_sd > 0 else float("inf")
        auroc = float(
            np.mean(
                (careful_projection[:, None] > direct_projection[None, :])
                + 0.5 * (careful_projection[:, None] == direct_projection[None, :])
            )
        )
        activation_norms = np.concatenate(
            (
                np.linalg.norm(arrays[2][:, layer, :], axis=1),
                np.linalg.norm(arrays[3][:, layer, :], axis=1),
            )
        )
        activation_scale = float(np.median(activation_norms))
        mean_gap = float(np.mean(gaps))
        rows.append(
            {
                "layer": layer,
                "paired_mean_gap": mean_gap,
                "positive_gap_fraction": float(np.mean(gaps > 0)),
                "standardized_paired_effect": effect,
                "auroc": auroc,
                "direction_pre_normalization_norm": norm,
                "direction_hash": vector_sha256(direction),
                "ordinary_activation_scale": activation_scale,
                "source_gap_activation_scale_ratio": mean_gap / activation_scale,
                "source_eligible": bool(
                    np.mean(gaps > 0) >= 0.80 and auroc >= 0.80 and mean_gap > 0
                ),
            }
        )
    return directions, rows


def shortlist_layers(atlas_rows: Sequence[Mapping[str, Any]], n_layers: int) -> list[int]:
    by_layer = {int(row["layer"]): row for row in atlas_rows}
    shortlist: list[int] = []
    for quartile in np.array_split(np.arange(n_layers), 4):
        eligible = [
            by_layer[int(layer)]
            for layer in quartile
            if by_layer[int(layer)]["source_eligible"]
        ]
        if eligible:
            selected = max(
                eligible,
                key=lambda row: (float(row["standardized_paired_effect"]), -int(row["layer"])),
            )
            shortlist.append(int(selected["layer"]))
    if len(shortlist) < 2:
        raise RuntimeError("GATE13_SOURCE_REPRESENTATION_NOT_QUALIFIED")
    return shortlist


def _orthogonal_unit(vector: np.ndarray, against: Sequence[np.ndarray]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
    for basis in against:
        unit = np.asarray(basis, dtype=np.float64).reshape(-1)
        value -= np.dot(value, unit) * unit
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("orthogonalized Gate 13 null direction is degenerate")
    return value / norm


def first_stage_nulls(
    meaningful: np.ndarray, paired_differences: np.ndarray, layer: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(stable_seed(EXPERIMENT_ID, "FIRST-STAGE-ISOTROPIC", layer))
    isotropic = _orthogonal_unit(rng.normal(size=len(meaningful)), [meaningful])
    signs_rng = np.random.default_rng(stable_seed(EXPERIMENT_ID, "FIRST-STAGE-SHUFFLED", layer))
    signs = signs_rng.choice((-1.0, 1.0), size=len(paired_differences))
    shuffled = _orthogonal_unit((paired_differences * signs[:, None]).mean(axis=0), [meaningful])
    return {"ISOTROPIC": isotropic, "SHUFFLED": shuffled}


def build_first_stage_schedule(
    item_ids: Sequence[str], model_name: str, layers: Sequence[int]
) -> list[dict[str, Any]]:
    conditions = ["BASELINE"]
    for layer in layers:
        conditions.extend(
            (f"MEANINGFUL_L{layer}_D50", f"ISOTROPIC_L{layer}_D50", f"SHUFFLED_L{layer}_D50")
        )
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        seed = stable_seed(EXPERIMENT_ID, "FIRST-STAGE", model_name, item_id)
        order = sorted(
            conditions,
            key=lambda condition: stable_digest(
                SELECTION_NAMESPACE, "FIRST-STAGE-ORDER", model_name, item_id, condition
            ),
        )
        for order_index, condition in enumerate(order):
            rows.append(
                {
                    "stage": "LAYER_FIRST_STAGE",
                    "model": model_name,
                    "item_id": item_id,
                    "condition": condition,
                    "rollout_index": 0,
                    "condition_order": order_index,
                    "seed": seed,
                    "seed_regime": "MATCHED_COUPLING_CALIBRATION",
                }
            )
    _validate_unique_schedule(rows, independent_seeds=False)
    return rows


def select_first_stage_layer(
    layer_metrics: Mapping[int, Mapping[str, float]], source_effects: Mapping[int, float]
) -> tuple[int, dict[int, bool]]:
    passed: dict[int, bool] = {}
    for layer, metric in layer_metrics.items():
        passed[layer] = bool(
            metric["commitment_validity"] >= 0.90
            and metric["semantic_evaluability"] >= 0.90
            and metric["accuracy"] >= metric["baseline_accuracy"] - 0.10
            and metric["Q"] >= 0.15
            and metric["Q"] - metric["null_mean_Q"] >= 0.05
            and metric["Q"] > metric["null_max_Q"]
        )
    candidates = [layer for layer, value in passed.items() if value]
    if not candidates:
        raise RuntimeError("GATE13_NO_CAUSAL_LAYER_FIRST_STAGE")
    selected = max(
        candidates,
        key=lambda layer: (
            layer_metrics[layer]["Q"] - layer_metrics[layer]["null_mean_Q"],
            source_effects[layer],
            -layer,
        ),
    )
    return selected, passed


def final_null_bank(
    meaningful: np.ndarray, paired_differences: np.ndarray, layer: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    basis = [np.asarray(meaningful, dtype=np.float64)]
    bank: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"layer": layer, "records": {}}
    for index in range(2):
        seed = stable_seed(EXPERIMENT_ID, "FINAL-ISOTROPIC", layer, index)
        rng = np.random.default_rng(seed)
        value = _orthogonal_unit(rng.normal(size=len(meaningful)), basis)
        name = f"R{index}"
        bank[name] = value
        basis.append(value)
        metadata["records"][name] = {"kind": "isotropic", "seed": seed}
    for index in range(2):
        seed = stable_seed(EXPERIMENT_ID, "FINAL-SHUFFLED", layer, index)
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=len(paired_differences))
        value = _orthogonal_unit((paired_differences * signs[:, None]).mean(axis=0), basis)
        name = f"R{index + 2}"
        bank[name] = value
        basis.append(value)
        metadata["records"][name] = {
            "kind": "construction_matched_shuffled",
            "seed": seed,
            "sign_pattern_sha256": hashlib.sha256(signs.astype(np.int8).tobytes()).hexdigest(),
        }
    vectors = {"MEANINGFUL": np.asarray(meaningful, dtype=np.float64), **bank}
    names = list(vectors)
    for name, value in vectors.items():
        if not np.isclose(np.linalg.norm(value), 1.0, atol=1e-10):
            raise RuntimeError(f"Gate 13 final null bank norm failed for {name}")
    cosines = {
        f"{left}__{right}": float(np.dot(vectors[left], vectors[right]))
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    if max(map(abs, cosines.values())) > 1e-6:
        raise RuntimeError("Gate 13 final null bank orthogonality failed")
    for name, value in bank.items():
        metadata["records"][name].update(
            {
                "canonical_float64_vector_sha256": vector_sha256(value),
                "norm": float(np.linalg.norm(value)),
                "source_axis_projection": float(np.dot(value, meaningful)),
            }
        )
    metadata["cosines"] = cosines
    return bank, metadata


def build_dose_schedule(item_ids: Sequence[str], model_name: str) -> list[dict[str, Any]]:
    conditions = ["BASELINE", "TEXTUAL_CAREFUL"]
    conditions.extend(f"MEAN_{dose}" for dose in DOSE_FRACTIONS)
    conditions.extend(f"R{index}_{dose}" for dose in DOSE_FRACTIONS for index in range(4))
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        for rollout in (0, 1):
            seed = stable_seed(EXPERIMENT_ID, "DOSE", model_name, item_id, rollout)
            order = sorted(
                conditions,
                key=lambda condition: stable_digest(
                    SELECTION_NAMESPACE, "DOSE-ORDER", model_name, item_id, rollout, condition
                ),
            )
            for order_index, condition in enumerate(order):
                rows.append(
                    {
                        "stage": "DOSE_CALIBRATION",
                        "model": model_name,
                        "item_id": item_id,
                        "condition": condition,
                        "rollout_index": rollout,
                        "condition_order": order_index,
                        "seed": seed,
                        "seed_regime": "MATCHED_COUPLING_CALIBRATION",
                    }
                )
    _validate_unique_schedule(rows, independent_seeds=False)
    return rows


def select_dose(
    baseline: Mapping[str, float],
    textual: Mapping[str, float],
    doses: Mapping[str, Mapping[str, float]],
    random_q: Mapping[str, Mapping[str, float]],
) -> tuple[str | None, dict[str, dict[str, bool]], str]:
    source_replicated = bool(
        textual["commitment_validity"] >= 0.90
        and textual["semantic_evaluability"] >= 0.90
        and textual["mean_tokens"] >= 1.5 * baseline["mean_tokens"]
        and textual["median_tokens"] >= baseline["median_tokens"] + 10
    )
    eligibility = {
        dose: dose_eligibility(
            baseline=baseline,
            dose=doses[dose],
            random_q=random_q[dose],
            source_replicated=source_replicated,
        )
        for dose in DOSE_FRACTIONS
    }
    for dose in ("D25", "D50", "D75"):
        if eligibility[dose]["eligible"]:
            return dose, eligibility, "GATE13_SAFE_DOSE_SELECTED"
    if eligibility["D100"]["eligible"]:
        return None, eligibility, "GATE13_ORIGINAL_DOSE_ONLY_SPECIFIC"
    specific = [dose for dose in DOSE_FRACTIONS if eligibility[dose]["behavioral_first_stage"]]
    if specific and all(
        not (
            eligibility[dose]["commitment_validity"]
            and eligibility[dose]["semantic_evaluability"]
            and eligibility[dose]["competence_safety"]
        )
        for dose in specific
    ):
        return None, eligibility, "GATE13_EFFECT_VALIDITY_TRADEOFF"
    return None, eligibility, "GATE13_DOSES_NONSPECIFIC_OR_INERT"


def build_final_schedule(item_ids: Sequence[str], model_name: str) -> list[dict[str, Any]]:
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
            order = sorted(
                conditions,
                key=lambda condition: stable_digest(
                    SELECTION_NAMESPACE, "FINAL-ORDER", model_name, item_id, rollout, condition
                ),
            )
            for order_index, condition in enumerate(order):
                rows.append(
                    {
                        "stage": "FINAL_EVALUATION",
                        "model": model_name,
                        "item_id": item_id,
                        "condition": condition,
                        "rollout_index": rollout,
                        "condition_order": order_index,
                        "seed": stable_seed(
                            EXPERIMENT_ID, "FINAL", model_name, item_id, condition, rollout
                        ),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                    }
                )
    _validate_unique_schedule(rows, independent_seeds=True)
    return rows


def map_gate9_classification(classification: str) -> str:
    mapping = {
        "GATE9_STRONG_SAFE_SELECTED_DOSE_REPLICATION": (
            "GATE13_STRONG_CROSS_MODEL_PROTOCOL_REPLICATION"
        ),
        "GATE9_MINIMUM_SAFE_SELECTED_DOSE_SIGNAL": (
            "GATE13_MINIMUM_CROSS_MODEL_PROTOCOL_REPLICATION"
        ),
        "GATE9_SAFE_ERROR_PROFILE_MOVEMENT_ONLY": (
            "GATE13_CROSS_MODEL_CONTROL_WITHOUT_USEFUL_COMPLEMENTARITY"
        ),
        "GATE9_CAREFUL_STYLE_CONTROL_WITHOUT_ERROR_CONTROL": "GATE13_CROSS_MODEL_NO_REPLICATION",
        "GATE9_NO_SELECTED_DOSE_EFFECT": "GATE13_CROSS_MODEL_NO_REPLICATION",
        "GATE9_SELECTED_DOSE_DESTRUCTIVE": "GATE13_CROSS_MODEL_DESTRUCTIVE",
        "GATE9_SOURCE_POLICY_NOT_REPLICATED": "GATE13_CROSS_MODEL_NO_REPLICATION",
    }
    try:
        return mapping[classification]
    except KeyError as exc:
        raise ValueError(f"unsupported Gate-9 structural classification: {classification}") from exc


__all__ = [
    "ALLOCATION_COUNTS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CAREFUL_CONCISE",
    "DATASET_REPO",
    "DATASET_REVISION",
    "DOSE_FRACTIONS",
    "EXPERIMENT_ID",
    "FALLBACK_MODEL",
    "FALLBACK_REVISION",
    "HIDDEN_SIZE",
    "MAX_NEW_TOKENS",
    "MODEL_LAYER_PATH",
    "NUM_LAYERS",
    "PARSER_VERSION",
    "PRIMARY_MODEL",
    "PRIMARY_REVISION",
    "SCREEN_CONDITIONS",
    "SELECTION_NAMESPACE",
    "SOURCE_CAREFUL",
    "SOURCE_DIRECT",
    "VERBOSE_DIRECT",
    "allocate_pool",
    "build_dose_schedule",
    "build_final_schedule",
    "build_first_stage_schedule",
    "build_reused_development_pool",
    "build_screen_schedule",
    "classify_substrate",
    "file_sha256",
    "final_null_bank",
    "first_stage_nulls",
    "map_gate9_classification",
    "select_dose",
    "select_first_stage_layer",
    "shortlist_layers",
    "source_atlas",
]
