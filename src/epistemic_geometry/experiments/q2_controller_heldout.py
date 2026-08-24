"""Prospective, model-free contracts for the first Q2 DEVELOPMENT pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from epistemic_geometry.experiments.gate6 import unit_vector
from epistemic_geometry.experiments.gate6_3 import vector_sha256
from epistemic_geometry.reproducibility import stable_digest, stable_seed

EXPERIMENT_ID = "Q2_CONTROLLER_HELDOUT_GEOMETRY_PILOT"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 27
N_ITEMS = 120
ROLLOUTS = 2
ETA = 9.637427952852196
REFERENCE_SCALE = 10.153299177386142
DELTA_NORM = ETA * REFERENCE_SCALE
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 2026082401
QAP_PERMUTATIONS = 10_000
QAP_SEED = 2026082402

BASELINE = "BASELINE"
LOCATIONS = ("PROMPT_BOUNDARY", "EXECUTION_BOUNDARY")
SIGNS = ("PLUS", "MINUS")


@dataclass(frozen=True)
class SourceAxis:
    """One pre-outcome conceptual source contrast."""

    axis_id: str
    positive_instruction: str
    negative_instruction: str
    construct: str


SOURCE_AXES = (
    SourceAxis(
        axis_id="INDEPENDENT_VERIFICATION",
        positive_instruction=(
            "Work carefully. After deriving a candidate answer, independently verify it "
            "with a second pass, correct any discrepancy, and end with exactly one line "
            "in the form FINAL: <answer>."
        ),
        negative_instruction=(
            "Answer from the first-pass result without an independent verification pass. "
            "End with exactly one line in the form FINAL: <answer>."
        ),
        construct="verification policy: independent second pass minus first pass",
    ),
    SourceAxis(
        axis_id="EXPLICIT_STATE_TRACKING",
        positive_instruction=(
            "Solve by explicitly tracking program state step by step, including every "
            "variable update and control-flow transition. End with exactly one line in "
            "the form FINAL: <answer>."
        ),
        negative_instruction=(
            "Solve holistically without writing an explicit step-by-step state trace. "
            "End with exactly one line in the form FINAL: <answer>."
        ),
        construct="explicit state tracking minus holistic first-pass reasoning",
    ),
    SourceAxis(
        axis_id="TYPE_REPRESENTATION_DISCIPLINE",
        positive_instruction=(
            "Explicitly track runtime types, exact representations, mutation, aliasing, "
            "and container semantics wherever relevant. End with exactly one line in the "
            "form FINAL: <answer>."
        ),
        negative_instruction=(
            "Focus on the apparent final value without explicitly tracking runtime types, "
            "representations, mutation, or aliasing. End with exactly one line in the form "
            "FINAL: <answer>."
        ),
        construct="type and representation discipline minus value-only reasoning",
    ),
)

EXECUTION_TEACHER_TEXT = (
    "I will now apply the requested reasoning policy to the program before committing "
    "to one answer."
)

NULL_FAMILIES = (
    "ISOTROPIC",
    "ISOTROPIC",
    "CONSTRUCTION_MATCHED_SIGN_SHUFFLED",
    "CONSTRUCTION_MATCHED_SIGN_SHUFFLED",
)


def source_axis_payload() -> list[dict[str, str]]:
    return [asdict(axis) for axis in SOURCE_AXES]


def meaningful_controller_id(axis_id: str, location: str, sign: str) -> str:
    if location not in LOCATIONS or sign not in SIGNS:
        raise ValueError("unknown controller location or sign")
    return f"MEAN_{axis_id}_{location}_{sign}"


def meaningful_controller_ids() -> tuple[str, ...]:
    return tuple(
        meaningful_controller_id(axis.axis_id, location, sign)
        for axis in SOURCE_AXES
        for location in LOCATIONS
        for sign in SIGNS
    )


NULL_IDS = tuple(f"NULL_{kind}_R{index}" for index, kind in enumerate(NULL_FAMILIES))
CONTROLLER_IDS = (*meaningful_controller_ids(), *NULL_IDS)
CONDITIONS = (BASELINE, *CONTROLLER_IDS)


def expand_meaningful_bank(
    base_directions: Mapping[tuple[str, str], np.ndarray],
) -> dict[str, np.ndarray]:
    """Expand six conceptual/location directions into both frozen signs."""

    expected = {(axis.axis_id, location) for axis in SOURCE_AXES for location in LOCATIONS}
    if set(base_directions) != expected:
        raise ValueError("base direction identities differ from the prospective six-family bank")
    output: dict[str, np.ndarray] = {}
    for axis in SOURCE_AXES:
        for location in LOCATIONS:
            vector = unit_vector(base_directions[(axis.axis_id, location)])
            output[meaningful_controller_id(axis.axis_id, location, "PLUS")] = vector
            output[meaningful_controller_id(axis.axis_id, location, "MINUS")] = -vector
    return output


def _orthogonal_unit(candidate: np.ndarray, basis: Sequence[np.ndarray]) -> np.ndarray:
    value = np.asarray(candidate, dtype=np.float64).reshape(-1).copy()
    # Modified Gram-Schmidt twice is deterministic and stable for the small basis.
    for _ in range(2):
        for previous in basis:
            unit = unit_vector(previous)
            value -= float(np.dot(value, unit)) * unit
    return unit_vector(value)


def build_null_bank(
    base_directions: Mapping[tuple[str, str], np.ndarray],
    paired_differences: Mapping[tuple[str, str], np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build two isotropic and two construction-matched shuffled nulls.

    Nulls are orthogonalized against the six-dimensional meaningful base span,
    not against the sign-expanded duplicate span. Construction-matched draws
    pool pairwise source differences after per-family RMS normalization.
    """

    expected = {(axis.axis_id, location) for axis in SOURCE_AXES for location in LOCATIONS}
    if set(base_directions) != expected or set(paired_differences) != expected:
        raise ValueError("null construction requires all six prospective source families")
    basis = [unit_vector(base_directions[key]) for key in sorted(expected)]
    hidden = len(basis[0])
    if any(len(vector) != hidden for vector in basis):
        raise ValueError("all Q2 controllers must occupy one residual coordinate space")
    normalized_pairs: list[np.ndarray] = []
    for key in sorted(expected):
        pairs = np.asarray(paired_differences[key], dtype=np.float64)
        if pairs.ndim != 2 or pairs.shape[1] != hidden or len(pairs) < 2:
            raise ValueError("paired source differences have invalid shape")
        rms = float(np.sqrt(np.mean(np.square(pairs))))
        if not np.isfinite(rms) or rms <= 0:
            raise ValueError("paired source differences are degenerate")
        normalized_pairs.append(pairs / rms)
    pooled = np.concatenate(normalized_pairs, axis=0)
    bank: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for controller_id, family in zip(NULL_IDS, NULL_FAMILIES, strict=True):
        seed = stable_seed(EXPERIMENT_ID, "NULL_BANK_V1", controller_id)
        rng = np.random.default_rng(seed)
        if family == "ISOTROPIC":
            raw = rng.standard_normal(hidden)
        else:
            signs = rng.choice((-1.0, 1.0), size=len(pooled))
            raw = np.mean(pooled * signs[:, None], axis=0)
        vector = _orthogonal_unit(raw, [*basis, *bank.values()])
        bank[controller_id] = vector
        records[controller_id] = {
            "family": family,
            "seed": int(seed),
            "canonical_float64_vector_sha256": vector_sha256(vector),
        }
    return bank, {"records": records, "pooled_pair_count": len(pooled)}


def validate_bank(bank: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate exact identities, unit norms, sign pairs, and base diversity."""

    if set(bank) != set(CONTROLLER_IDS):
        raise ValueError("Q2 bank must contain exactly 12 meaningful and four nulls")
    vectors = {name: unit_vector(value) for name, value in bank.items()}
    norms = {name: float(np.linalg.norm(value)) for name, value in vectors.items()}
    sign_errors: dict[str, float] = {}
    base_vectors: list[np.ndarray] = []
    base_names: list[str] = []
    for axis in SOURCE_AXES:
        for location in LOCATIONS:
            plus = meaningful_controller_id(axis.axis_id, location, "PLUS")
            minus = meaningful_controller_id(axis.axis_id, location, "MINUS")
            sign_errors[f"{axis.axis_id}:{location}"] = float(
                np.linalg.norm(vectors[plus] + vectors[minus])
            )
            base_names.append(plus)
            base_vectors.append(vectors[plus])
    base_cos = np.asarray(base_vectors) @ np.asarray(base_vectors).T
    off_diagonal = np.abs(base_cos - np.eye(len(base_vectors)))
    null_cos = {
        name: float(max(abs(np.dot(vectors[name], base)) for base in base_vectors))
        for name in NULL_IDS
    }
    null_pair_cos = {
        f"{left}__{right}": float(abs(np.dot(vectors[left], vectors[right])))
        for index, left in enumerate(NULL_IDS)
        for right in NULL_IDS[index + 1 :]
    }
    return {
        "controller_count": len(vectors),
        "meaningful_count": len(meaningful_controller_ids()),
        "null_count": len(NULL_IDS),
        "unit_norm_pass": all(abs(value - 1.0) <= 1e-10 for value in norms.values()),
        "sign_pair_pass": all(value <= 1e-10 for value in sign_errors.values()),
        "base_max_absolute_cosine": float(np.max(off_diagonal)),
        "base_diversity_pass": float(np.max(off_diagonal)) <= 0.98,
        "null_to_meaningful_max_absolute_cosines": null_cos,
        "null_pair_absolute_cosines": null_pair_cos,
        "null_orthogonality_pass": max([*null_cos.values(), *null_pair_cos.values()]) <= 1e-6,
        "norms": norms,
        "sign_pair_errors": sign_errors,
        "base_names": base_names,
        "base_absolute_cosine_matrix": np.abs(base_cos).tolist(),
        "hashes": {name: vector_sha256(value) for name, value in vectors.items()},
    }


def controller_split() -> dict[str, Any]:
    """Freeze a 10/6 source-family-held-out controller split."""

    heldout_axis = min(
        (axis.axis_id for axis in SOURCE_AXES),
        key=lambda value: stable_digest(EXPERIMENT_ID, "HELDOUT_AXIS", value),
    )
    isotropic = [
        name
        for name, kind in zip(NULL_IDS, NULL_FAMILIES, strict=True)
        if kind == "ISOTROPIC"
    ]
    shuffled = [
        name
        for name, kind in zip(NULL_IDS, NULL_FAMILIES, strict=True)
        if kind == "CONSTRUCTION_MATCHED_SIGN_SHUFFLED"
    ]
    heldout_nulls = {
        min(isotropic, key=lambda value: stable_digest(EXPERIMENT_ID, "HELDOUT_NULL", value)),
        min(shuffled, key=lambda value: stable_digest(EXPERIMENT_ID, "HELDOUT_NULL", value)),
    }
    test = {
        meaningful_controller_id(heldout_axis, location, sign)
        for location in LOCATIONS
        for sign in SIGNS
    } | heldout_nulls
    train = set(CONTROLLER_IDS) - test
    if len(train) != 10 or len(test) != 6:
        raise AssertionError("Q2 split must be exactly 10 train and six heldout controllers")
    return {
        "namespace": f"{EXPERIMENT_ID}:CONTROLLER_SPLIT_V1",
        "heldout_axis": heldout_axis,
        "train_controllers": sorted(train),
        "test_controllers": sorted(test),
        "train_n": len(train),
        "test_n": len(test),
        "train_edge_count": len(train) * (len(train) - 1) // 2,
        "heldout_edge_count": len(CONTROLLER_IDS) * (len(CONTROLLER_IDS) - 1) // 2
        - len(train) * (len(train) - 1) // 2,
    }


def build_schedule(item_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Build the frozen independent common-panel schedule."""

    ids = tuple(map(str, item_ids))
    if len(ids) != N_ITEMS or len(set(ids)) != N_ITEMS:
        raise ValueError("Q2 common panel requires exactly 120 unique items")
    rows: list[dict[str, Any]] = []
    for item_index, item_id in enumerate(ids):
        for rollout in range(ROLLOUTS):
            order = sorted(
                CONDITIONS,
                key=lambda condition: stable_digest(
                    EXPERIMENT_ID, "COMMON_PANEL_ORDER", item_id, rollout, condition
                ),
            )
            for condition_order, condition in enumerate(order):
                rows.append(
                    {
                        "phase": "Q2_COMMON_PANEL",
                        "item_index": item_index,
                        "item_id": item_id,
                        "condition": condition,
                        "controller_id": None if condition == BASELINE else condition,
                        "rollout_index": rollout,
                        "condition_order": condition_order,
                        "seed": stable_seed(EXPERIMENT_ID, item_id, condition, rollout),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                    }
                )
    validate_schedule(rows, ids)
    return rows


def validate_schedule(rows: Sequence[Mapping[str, Any]], item_ids: Sequence[str]) -> None:
    expected = len(item_ids) * len(CONDITIONS) * ROLLOUTS
    if len(rows) != expected:
        raise ValueError(f"Q2 schedule has {len(rows)} rows, expected {expected}")
    keys = [
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        for row in rows
    ]
    seeds = [int(row["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Q2 schedule has duplicate logical keys")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Q2 schedule has a seed collision")
    if {str(row["condition"]) for row in rows} != set(CONDITIONS):
        raise ValueError("Q2 schedule has the wrong condition set")


def error_arrays(
    rows: Sequence[Mapping[str, Any]], item_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    lookup: dict[tuple[str, str, int], int] = {}
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        if key in lookup:
            raise ValueError("duplicate Q2 outcome key")
        lookup[key] = int(not bool(row["correct"]))
    expected = len(item_ids) * len(CONDITIONS) * ROLLOUTS
    if len(lookup) != expected:
        raise ValueError("Q2 outcome journal is incomplete")
    return {
        condition: np.asarray(
            [
                [lookup[(str(item_id), condition, rollout)] for rollout in range(ROLLOUTS)]
                for item_id in item_ids
            ],
            dtype=np.int8,
        )
        for condition in CONDITIONS
    }


def pairwise_unbiased_distance_matrix(
    arrays: Mapping[str, np.ndarray], controller_ids: Sequence[str] = CONTROLLER_IDS
) -> np.ndarray:
    """Estimate D_ij without plug-in squared propensities."""

    names = tuple(controller_ids)
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    for left_index, left in enumerate(names):
        left_values = np.asarray(arrays[left], dtype=np.float64)
        if left_values.ndim != 2 or left_values.shape[1] != 2:
            raise ValueError("Q2 D requires exactly two rollouts per controller")
        for right_index in range(left_index + 1, len(names)):
            right_values = np.asarray(arrays[names[right_index]], dtype=np.float64)
            if right_values.shape != left_values.shape:
                raise ValueError("all Q2 controller arrays must have equal shape")
            estimate = float(
                np.mean(
                    (left_values[:, 0] - right_values[:, 0])
                    * (left_values[:, 1] - right_values[:, 1])
                )
            )
            matrix[left_index, right_index] = estimate
            matrix[right_index, left_index] = estimate
    return matrix


def qualification_decision(
    source_records: Mapping[str, Mapping[str, Any]],
    controller_records: Mapping[str, Mapping[str, Any]],
    bank_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply source and manipulation checks without correctness-based ranking."""

    source_pass = {
        axis.axis_id: bool(
            source_records[axis.axis_id]["positive_commitment_validity"] >= 0.90
            and source_records[axis.axis_id]["negative_commitment_validity"] >= 0.90
            and source_records[axis.axis_id]["positive_semantic_evaluability"] >= 0.90
            and source_records[axis.axis_id]["negative_semantic_evaluability"] >= 0.90
            and (
                (
                    source_records[axis.axis_id]["cross_disagreement"] >= 0.10
                    and source_records[axis.axis_id]["excess_disagreement"] >= 0.03
                )
                or (
                    source_records[axis.axis_id]["positive_negative_mean_token_ratio"]
                    >= 1.15
                    and source_records[axis.axis_id]["positive_minus_negative_median_tokens"]
                    >= 2
                )
            )
            and all(
                source_records[axis.axis_id]["activation"][location][
                    "standardized_mean_gap"
                ]
                >= 0.20
                and source_records[axis.axis_id]["activation"][location][
                    "positive_gap_fraction"
                ]
                >= 0.60
                for location in LOCATIONS
            )
        )
        for axis in SOURCE_AXES
    }
    controller_pass = {
        name: bool(
            controller_records[name]["commitment_validity"] >= 0.75
            and controller_records[name]["semantic_evaluability"] >= 0.75
            and controller_records[name]["semantic_change_rate"] >= 1.0 / 12.0
            and controller_records[name]["raw_sequence_change_rate"] >= 0.25
        )
        for name in CONTROLLER_IDS
    }
    geometry_pass = all(
        bool(bank_validation[key])
        for key in (
            "unit_norm_pass",
            "sign_pair_pass",
            "base_diversity_pass",
            "null_orthogonality_pass",
        )
    )
    qualified = bool(all(source_pass.values()) and all(controller_pass.values()) and geometry_pass)
    return {
        "qualified": qualified,
        "classification": (
            "Q2_CONTROLLER_BANK_QUALIFIED"
            if qualified
            else "Q2_CONTROLLER_BANK_NOT_QUALIFIED"
        ),
        "source_axis_pass": source_pass,
        "controller_pass": controller_pass,
        "representation_geometry_pass": geometry_pass,
        "accuracy_used_for_qualification": False,
        "G_C_D_used_for_qualification": False,
    }


__all__ = [
    "BASELINE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONDITIONS",
    "CONTROLLER_IDS",
    "DELTA_NORM",
    "ETA",
    "EXECUTION_TEACHER_TEXT",
    "EXPERIMENT_ID",
    "LAYER",
    "LOCATIONS",
    "MODEL",
    "MODEL_REVISION",
    "N_ITEMS",
    "NULL_FAMILIES",
    "NULL_IDS",
    "QAP_PERMUTATIONS",
    "QAP_SEED",
    "REFERENCE_SCALE",
    "ROLLOUTS",
    "SIGNS",
    "SOURCE_AXES",
    "build_null_bank",
    "build_schedule",
    "controller_split",
    "error_arrays",
    "expand_meaningful_bank",
    "meaningful_controller_id",
    "meaningful_controller_ids",
    "pairwise_unbiased_distance_matrix",
    "qualification_decision",
    "source_axis_payload",
    "validate_bank",
    "validate_schedule",
]
