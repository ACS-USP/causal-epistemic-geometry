"""Prospective, model-free contracts for the calibrated Q2-V2 pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from epistemic_geometry.experiments.gate6_3 import vector_sha256
from epistemic_geometry.reproducibility import canonical_json, stable_digest, stable_seed

EXPERIMENT_ID = "Q2_CONTROLLER_HELDOUT_GEOMETRY_V2"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 27
ROLLOUTS = 2
DOSE_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
DOSE_NAMES = ("D_LOW", "D_MEDIUM", "D_HIGH", "D_VERY_HIGH")
SOURCE_CONSTRUCTION_N = 24
SOURCE_VALIDATION_N = 24
DOSE_CALIBRATION_N = 12
COMMON_PANEL_N = 120
NULL_COUNT = 4
MIN_SOURCE_AXES = 4
MIN_MEANINGFUL = 14
MIN_FAMILIES = 4
MIN_CAUSAL_DIRECTIONS = 8
RAW_MOVEMENT_MIN = 0.15
SEMANTIC_MOVEMENT_MIN = 0.10
VALIDITY_MIN = 0.90
VALIDITY_DROP_MAX = 0.05
MAX_TRUNCATION_RATE = 0.10
ORTHOGONALITY_TOLERANCE = 1e-6
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026082401
QAP_PERMUTATIONS = 10_000
QAP_SEED = 2026082402
BASELINE = "BASELINE"
LOCATIONS = ("PROMPT_BOUNDARY", "EXECUTION_BOUNDARY")
SIGNS = ("PLUS", "MINUS")


@dataclass(frozen=True)
class SourceAxis:
    axis_id: str
    positive_instruction: str
    negative_instruction: str
    rationale: str


SOURCE_AXES = (
    SourceAxis(
        "INDEPENDENT_VERIFICATION",
        (
            "Work carefully. After deriving a candidate answer, independently verify it "
            "with a second pass, correct any discrepancy, and end with exactly one line "
            "in the form FINAL: <answer>."
        ),
        (
            "Answer from the first-pass result without an independent verification pass. "
            "End with exactly one line in the form FINAL: <answer>."
        ),
        "independent second-pass verification versus first-pass commitment",
    ),
    SourceAxis(
        "EXPLICIT_STATE_TRACKING",
        (
            "Solve by explicitly tracking program state step by step, including every "
            "variable update and control-flow transition. End with exactly one line in "
            "the form FINAL: <answer>."
        ),
        (
            "Solve holistically without writing an explicit step-by-step state trace. "
            "End with exactly one line in the form FINAL: <answer>."
        ),
        "state-transition bookkeeping versus holistic first-pass reasoning",
    ),
    SourceAxis(
        "TYPE_REPRESENTATION_DISCIPLINE",
        (
            "Explicitly track runtime types, exact representations, mutation, aliasing, "
            "and container semantics wherever relevant. End with exactly one line in "
            "the form FINAL: <answer>."
        ),
        (
            "Focus on the apparent final value without explicitly tracking runtime "
            "types, representations, mutation, or aliasing. End with exactly one line "
            "in the form FINAL: <answer>."
        ),
        "runtime type/representation tracking versus value-only reasoning",
    ),
    SourceAxis(
        "DECOMPOSE_THEN_SOLVE",
        (
            "First decompose the problem into named subproblems, solve each subproblem, "
            "and then compose the result. End with exactly one line in the form FINAL: "
            "<answer>."
        ),
        (
            "Solve the problem directly as one undivided task without first naming "
            "subproblems. End with exactly one line in the form FINAL: <answer>."
        ),
        "explicit subproblem decomposition versus undivided direct solving",
    ),
    SourceAxis(
        "INVARIANT_CHECKING",
        (
            "Identify the relevant loop or state invariant, use it while solving, and "
            "verify it at the end. End with exactly one line in the form FINAL: <answer>."
        ),
        (
            "Solve without formulating or checking a loop or state invariant. End with "
            "exactly one line in the form FINAL: <answer>."
        ),
        "invariant-based consistency checking versus no explicit invariant",
    ),
    SourceAxis(
        "COUNTERFACTUAL_CHECKING",
        (
            "After deriving the answer, test it against a nearby counterexample or "
            "alternative case and revise if needed. End with exactly one line in the "
            "form FINAL: <answer>."
        ),
        (
            "Commit the direct derivation without testing a counterexample or alternative "
            "case. End with exactly one line in the form FINAL: <answer>."
        ),
        "counterexample/alternative-case checking versus direct commitment",
    ),
)

EXECUTION_TEACHER_TEXT = (
    "I will now apply the requested reasoning policy to the program before committing "
    "to one answer."
)


def source_axis_payload() -> list[dict[str, str]]:
    return [asdict(axis) for axis in SOURCE_AXES]


def meaningful_controller_id(axis_id: str, location: str, sign: str) -> str:
    if location not in LOCATIONS or sign not in SIGNS:
        raise ValueError("unknown location or sign")
    return f"MEAN_{axis_id}_{location}_{sign}"


def meaningful_ids(axis_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        meaningful_controller_id(axis_id, location, sign)
        for axis_id in axis_ids
        for location in LOCATIONS
        for sign in SIGNS
    )


def dose_condition_id(controller: str, dose_name: str) -> str:
    if dose_name not in DOSE_NAMES:
        raise ValueError("unknown dose name")
    return f"CAL_{controller}_{dose_name}"


def controller_metadata(controller: str) -> dict[str, str]:
    parts = controller.split("_")
    if not controller.startswith("MEAN_"):
        return {"source_axis": "NULL", "source_location": "NULL", "sign": "NULL"}
    return {
        "source_axis": "_".join(parts[1:-3]),
        "source_location": "_".join(parts[-3:-1]),
        "sign": parts[-1],
    }


def orthonormal_basis(vectors: Sequence[np.ndarray], tolerance: float = 1e-10) -> np.ndarray:
    """Return an SVD basis for the span of the supplied columns."""

    if not vectors:
        raise ValueError("cannot form a basis from no vectors")
    matrix = np.column_stack(
        [np.asarray(vector, dtype=np.float64).reshape(-1) for vector in vectors]
    )
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if len(singular) == 0 or singular[0] <= 0:
        raise ValueError("degenerate meaningful span")
    rank = int(np.sum(singular > singular[0] * tolerance))
    return u[:, :rank]


def project_orthogonal(candidate: np.ndarray, basis: np.ndarray) -> np.ndarray:
    value = np.asarray(candidate, dtype=np.float64).reshape(-1)
    projected = value - basis @ (basis.T @ value)
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("random candidate is degenerate after span projection")
    return projected / norm


def build_null_bank(
    meaningful_vectors: Mapping[str, np.ndarray], seeds: Sequence[int]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(seeds) != NULL_COUNT:
        raise ValueError("V2 requires exactly four fresh null seeds")
    basis = orthonormal_basis(list(meaningful_vectors.values()))
    bank: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for index, seed in enumerate(seeds):
        name = f"NULL_V2_R{index}"
        raw = np.random.default_rng(int(seed)).standard_normal(basis.shape[0])
        vector = project_orthogonal(raw, np.column_stack([basis, *bank.values()]))
        bank[name] = vector
        records[name] = {
            "seed": int(seed),
            "vector_hash": vector_sha256(vector),
        }
    return bank, {
        "basis_rank": int(basis.shape[1]),
        "span_projector": "SVD orthonormal basis; r_perp=r-Q(Q^T r)",
        "records": records,
    }


def validate_null_bank(
    meaningful_vectors: Mapping[str, np.ndarray], null_vectors: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    basis = orthonormal_basis(list(meaningful_vectors.values()))
    span_cos = {
        name: float(np.max(np.abs(basis.T @ vector))) for name, vector in null_vectors.items()
    }
    names = list(null_vectors)
    pair_cos = {
        f"{left}__{right}": float(abs(np.dot(null_vectors[left], null_vectors[right])))
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    return {
        "meaningful_span_rank": int(basis.shape[1]),
        "unit_norm_pass": all(
            abs(np.linalg.norm(v) - 1.0) <= 1e-10
            for v in [*meaningful_vectors.values(), *null_vectors.values()]
        ),
        "span_orthogonality_max": max(span_cos.values()),
        "span_orthogonality_pass": max(span_cos.values()) <= ORTHOGONALITY_TOLERANCE,
        "pairwise_null_cosines": pair_cos,
        "pairwise_null_orthogonality_pass": max(pair_cos.values(), default=0.0)
        <= ORTHOGONALITY_TOLERANCE,
        "span_cosines": span_cos,
        "tolerance": ORTHOGONALITY_TOLERANCE,
    }


def source_pass(record: Mapping[str, Any]) -> bool:
    return bool(
        record["positive_validity"] >= VALIDITY_MIN
        and record["negative_validity"] >= VALIDITY_MIN
        and record["positive_evaluability"] >= VALIDITY_MIN
        and record["negative_evaluability"] >= VALIDITY_MIN
        and record["cross_disagreement"] >= 0.10
        and record["excess_disagreement"] >= 0.03
        and all(
            record["activation"][location]["standardized_gap"] >= 0.20
            and record["activation"][location]["positive_gap_fraction"] >= 0.60
            for location in LOCATIONS
        )
    )


def dose_is_safe(record: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return bool(
        record["validity"] >= VALIDITY_MIN
        and record["evaluability"] >= VALIDITY_MIN
        and record["validity"] >= baseline["validity"] - VALIDITY_DROP_MAX
        and record["evaluability"] >= baseline["evaluability"] - VALIDITY_DROP_MAX
        and record["truncation_rate"] <= MAX_TRUNCATION_RATE
    )


def dose_is_causal(record: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return bool(
        dose_is_safe(record, baseline)
        and record["raw_sequence_movement"] >= RAW_MOVEMENT_MIN
        and record["semantic_movement"] >= SEMANTIC_MOVEMENT_MIN
    )


def choose_operating_dose(
    records: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    causal = [name for name in DOSE_NAMES if dose_is_causal(records[name], baseline)]
    safe = [name for name in DOSE_NAMES if dose_is_safe(records[name], baseline)]
    selected = causal[0] if causal else (safe[0] if safe else None)
    return {
        "selected_dose": selected,
        "causal_doses": causal,
        "safe_doses": safe,
        "strongest_safe_causal_dose": causal[-1] if causal else None,
        "causal_pass": bool(causal),
    }


def bank_qualification(
    selected: Mapping[str, Mapping[str, Any]],
    source_records: Mapping[str, Mapping[str, Any]],
    null_checks: Mapping[str, Any],
) -> dict[str, Any]:
    families = sorted({record["source_axis"] for record in selected.values()})
    causal = [name for name, record in selected.items() if record["causal_pass"]]
    by_family = {
        family: sum(item["source_axis"] == family for item in selected.values())
        for family in families
    }
    dose_bins = sorted(
        {record["selected_dose"] for record in selected.values() if record["selected_dose"]}
    )
    qualified_sources = [axis for axis, record in source_records.items() if record["source_pass"]]
    checks = {
        "source_axes_at_least_minimum": len(qualified_sources) >= MIN_SOURCE_AXES,
        "meaningful_count_at_least_minimum": len(selected) >= MIN_MEANINGFUL,
        "families_at_least_minimum": len(families) >= MIN_FAMILIES,
        "causal_directions_at_least_minimum": len(causal) >= MIN_CAUSAL_DIRECTIONS,
        "each_family_has_two_directions": all(value >= 2 for value in by_family.values()),
        "dose_dynamic_range_non_degenerate": len(dose_bins) >= 2,
        "null_span_orthogonality": bool(null_checks["span_orthogonality_pass"]),
        "null_pair_orthogonality": bool(null_checks["pairwise_null_orthogonality_pass"]),
    }
    return {
        "classification": "Q2_V2_CONTROLLER_BANK_QUALIFIED"
        if all(checks.values())
        else "Q2_V2_CONTROLLER_BANK_NOT_QUALIFIED",
        "checks": checks,
        "qualified_source_axes": qualified_sources,
        "meaningful_count": len(selected),
        "causal_direction_count": len(causal),
        "family_count": len(families),
        "families": families,
        "directions_per_family": by_family,
        "selected_dose_bins": dose_bins,
        "accuracy_used": False,
        "G_C_D_used": False,
    }


def source_schedule(item_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        for rollout in range(ROLLOUTS):
            conditions = [(axis.axis_id, polarity) for axis in SOURCE_AXES for polarity in SIGNS]
            conditions.sort(
                key=lambda pair: stable_digest(
                    EXPERIMENT_ID, "SOURCE_ORDER", item_id, rollout, *pair
                )
            )
            for order, (axis_id, polarity) in enumerate(conditions):
                rows.append(
                    {
                        "phase": "V2_SOURCE_QUALIFICATION",
                        "item_id": item_id,
                        "axis_id": axis_id,
                        "polarity": polarity,
                        "rollout_index": rollout,
                        "condition_order": order,
                        "seed": stable_seed(
                            EXPERIMENT_ID, "SOURCE", item_id, axis_id, polarity, rollout
                        ),
                        "seed_regime": "INDEPENDENT_SOURCE_QUALIFICATION",
                    }
                )
    return rows


def calibration_schedule(
    item_ids: Sequence[str], controller_ids: Sequence[str]
) -> list[dict[str, Any]]:
    conditions = [BASELINE] + [
        dose_condition_id(controller, dose) for controller in controller_ids for dose in DOSE_NAMES
    ]
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        seed = stable_seed(EXPERIMENT_ID, "CALIBRATION", item_id)
        order = sorted(
            conditions,
            key=lambda condition: stable_digest(EXPERIMENT_ID, "CAL_ORDER", item_id, condition),
        )
        rows.extend(
            {
                "phase": "V2_DOSE_CALIBRATION",
                "item_id": item_id,
                "condition": condition,
                "condition_order": index,
                "rollout_index": 0,
                "seed": seed,
                "seed_regime": "MATCHED_COUPLING_CALIBRATION",
            }
            for index, condition in enumerate(order)
        )
    return rows


def common_schedule(item_ids: Sequence[str], controller_ids: Sequence[str]) -> list[dict[str, Any]]:
    conditions = [BASELINE, *controller_ids]
    rows: list[dict[str, Any]] = []
    for item_index, item_id in enumerate(item_ids):
        for rollout in range(ROLLOUTS):
            order = sorted(
                conditions,
                key=lambda condition: stable_digest(
                    EXPERIMENT_ID, "COMMON_ORDER", item_id, rollout, condition
                ),
            )
            rows.extend(
                {
                    "phase": "V2_COMMON_PANEL",
                    "item_index": item_index,
                    "item_id": item_id,
                    "condition": condition,
                    "rollout_index": rollout,
                    "condition_order": index,
                    "seed": stable_seed(EXPERIMENT_ID, "COMMON", item_id, condition, rollout),
                    "seed_regime": "INDEPENDENT_PRIMARY",
                }
                for index, condition in enumerate(order)
            )
    return rows


def validate_schedule(
    rows: Sequence[Mapping[str, Any]],
    expected_keys: Sequence[tuple[str, str, int]],
    *,
    require_unique_seeds: bool = True,
) -> None:
    keys = [(str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in rows]
    if (
        len(keys) != len(expected_keys)
        or len(keys) != len(set(keys))
        or set(keys) != set(expected_keys)
    ):
        raise ValueError("V2 schedule is incomplete or contains duplicate logical keys")
    seeds = [int(row["seed"]) for row in rows]
    if require_unique_seeds and len(seeds) != len(set(seeds)):
        raise ValueError("V2 schedule has a seed collision")


def pairwise_unbiased_distance_matrix(
    arrays: Mapping[str, np.ndarray], controller_ids: Sequence[str]
) -> np.ndarray:
    names = tuple(controller_ids)
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    for left_index, left in enumerate(names):
        left_values = np.asarray(arrays[left], dtype=np.float64)
        if left_values.ndim != 2 or left_values.shape[1] != 2:
            raise ValueError("V2 D requires exactly two independent rollouts")
        for right_index in range(left_index + 1, len(names)):
            right_values = np.asarray(arrays[names[right_index]], dtype=np.float64)
            matrix[left_index, right_index] = matrix[right_index, left_index] = float(
                np.mean(
                    (left_values[:, 0] - right_values[:, 0])
                    * (left_values[:, 1] - right_values[:, 1])
                )
            )
    return matrix


def canonical_controller_split(axis_ids: Sequence[str]) -> dict[str, Any]:
    ordered = sorted(axis_ids, key=lambda axis: stable_digest(EXPERIMENT_ID, "FAMILY_SPLIT", axis))
    folds = {axis: index for index, axis in enumerate(ordered)}
    return {
        "scheme": "leave_one_source_family_out",
        "family_order": ordered,
        "fold_by_family": folds,
        "assignment_digest": stable_digest(
            EXPERIMENT_ID, "FAMILY_SPLIT_ASSIGNMENT", canonical_json(folds)
        ),
    }


__all__ = [
    "BASELINE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "DOSE_FRACTIONS",
    "DOSE_NAMES",
    "EXECUTION_TEACHER_TEXT",
    "EXPERIMENT_ID",
    "LAYER",
    "LOCATIONS",
    "MODEL",
    "MODEL_REVISION",
    "ORTHOGONALITY_TOLERANCE",
    "RAW_MOVEMENT_MIN",
    "SEMANTIC_MOVEMENT_MIN",
    "SIGNS",
    "SOURCE_AXES",
    "SOURCE_CONSTRUCTION_N",
    "SOURCE_VALIDATION_N",
    "DOSE_CALIBRATION_N",
    "COMMON_PANEL_N",
    "bank_qualification",
    "build_null_bank",
    "canonical_controller_split",
    "calibration_schedule",
    "canonical_json",
    "common_schedule",
    "controller_metadata",
    "dose_condition_id",
    "dose_is_causal",
    "dose_is_safe",
    "meaningful_controller_id",
    "meaningful_ids",
    "orthonormal_basis",
    "project_orthogonal",
    "source_axis_payload",
    "source_pass",
    "source_schedule",
    "validate_null_bank",
    "validate_schedule",
    "pairwise_unbiased_distance_matrix",
    "vector_sha256",
    "stable_seed",
    "stable_digest",
]
