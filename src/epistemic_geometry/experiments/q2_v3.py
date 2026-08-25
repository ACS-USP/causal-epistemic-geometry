"""Prospectively frozen design primitives for Q2 V3.

This module contains no behavioral outcomes and performs no model inference.
It defines the scientific choices needed by a later, separately authorized
execution agent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import permutations, product
from typing import Any

import numpy as np

EXPERIMENT_ID = "Q2_V3_RADIAL_ANGULAR_OUT_OF_BANK"
DATASET_REPO = "cruxeval-org/cruxeval"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 27
LOCATIONS = ("PROMPT_BOUNDARY", "EXECUTION_BOUNDARY")
SHELLS = ("MEDIUM", "STRONG")
SHELL_TARGETS = {"MEDIUM": 0.25, "STRONG": 0.50}
PRIMARY_PANEL_NAMESPACE = "Q2-V3-HISTORICAL-C-PROSPECTIVE-CONTROLLER-V1"
SOURCE_CONSTRUCTION_NAMESPACE = "Q2-V3-SOURCE-CONSTRUCTION-V1"
SOURCE_VALIDATION_NAMESPACE = "Q2-V3-SOURCE-VALIDATION-V1"
SHELL_CALIBRATION_NAMESPACE = "Q2-V3-SHELL-CALIBRATION-V1"
M1_COVARIANCE_NAMESPACE = "Q2-V3-M1-COVARIANCE-V1"
M2_PROBE_NAMESPACE = "Q2-V3-M2-CLASS-B-PROBES-V1"
EVALUATION_SEED_NAMESPACE = "Q2-V3-INDEPENDENT-PRIMARY-V1"
SOURCE_SEED_NAMESPACE = "Q2-V3-SOURCE-SEPARATION-V1"
SHELL_SEED_NAMESPACE = "Q2-V3-SHELL-SAFETY-V1"
NULL_SEEDS = (2026082601, 2026082602)
BOOTSTRAP_SEED = 2026082603

EXECUTION_TEACHER_TEXT = (
    "I will now apply the requested reasoning policy to the program before committing "
    "to one answer."
)


@dataclass(frozen=True)
class SourceFamily:
    family_id: str
    positive_instruction: str
    negative_instruction: str
    rationale: str


SOURCE_FAMILIES = (
    SourceFamily(
        "CONTROL_FLOW_PATH_COVERAGE",
        (
            "Enumerate every feasible control-flow path that can affect the return value, "
            "trace each path to completion, reconcile the path results, and end with exactly "
            "one line in the form FINAL: <answer>."
        ),
        (
            "Follow only the most immediately apparent control-flow path without enumerating "
            "alternative paths, and end with exactly one line in the form FINAL: <answer>."
        ),
        "complete feasible-path coverage versus single apparent-path tracing",
    ),
    SourceFamily(
        "MUTATION_ALIAS_CAUSALITY",
        (
            "Track every mutation and alias relation causally: record which objects share "
            "storage, apply updates in execution order, and end with exactly one line in the "
            "form FINAL: <answer>."
        ),
        (
            "Treat each variable update locally without maintaining an explicit alias or "
            "shared-storage ledger, and end with exactly one line in the form FINAL: <answer>."
        ),
        "causal mutation/alias ledger versus local variable-update reasoning",
    ),
    SourceFamily(
        "API_CONTRACT_EXACTNESS",
        (
            "For every operation, check the exact API contract, runtime type, return value, "
            "side effect, and exceptional case before composing the result. End with exactly "
            "one line in the form FINAL: <answer>."
        ),
        (
            "Infer each operation's ordinary intent without checking its exact API contract, "
            "runtime type, side effects, or exceptional cases, and end with exactly one line "
            "in the form FINAL: <answer>."
        ),
        "exact operation contracts versus intent-level API reasoning",
    ),
    SourceFamily(
        "LOOP_BOUNDARY_ACCOUNTING",
        (
            "Audit every loop boundary explicitly: initialization, first iteration, last "
            "included iteration, update order, and termination condition. End with exactly "
            "one line in the form FINAL: <answer>."
        ),
        (
            "Reason from the loop's overall pattern without separately auditing initialization, "
            "first/last iterations, update order, or termination, and end with exactly one line "
            "in the form FINAL: <answer>."
        ),
        "explicit boundary accounting versus loop-gist reasoning",
    ),
    SourceFamily(
        "HYPOTHESIS_BRANCH_ELIMINATION",
        (
            "Maintain all plausible candidate outcomes, test each against every branch and "
            "operation, eliminate contradicted candidates, and end with exactly one line in "
            "the form FINAL: <answer>."
        ),
        (
            "Commit to the first plausible outcome without maintaining or eliminating an "
            "explicit set of alternatives, and end with exactly one line in the form FINAL: "
            "<answer>."
        ),
        "explicit candidate elimination versus first-plausible commitment",
    ),
)


def family_payload() -> list[dict[str, str]]:
    return [asdict(family) for family in SOURCE_FAMILIES]


def base_direction_id(family: str, location: str) -> str:
    if family not in {value.family_id for value in SOURCE_FAMILIES}:
        raise ValueError("unknown Q2 V3 family")
    if location not in LOCATIONS:
        raise ValueError("unknown Q2 V3 source location")
    return f"MEAN_{family}_{location}"


def meaningful_controller_id(family: str, location: str, shell: str) -> str:
    if shell not in SHELLS:
        raise ValueError("unknown Q2 V3 shell")
    return f"{base_direction_id(family, location)}_{shell}"


def meaningful_controller_ids() -> tuple[str, ...]:
    return tuple(
        meaningful_controller_id(family.family_id, location, shell)
        for shell in SHELLS
        for family in SOURCE_FAMILIES
        for location in LOCATIONS
    )


def null_controller_ids() -> tuple[str, ...]:
    return tuple(f"NULL_Q2_V3_R{index}_{shell}" for shell in SHELLS for index in range(2))


def condition_ids() -> tuple[str, ...]:
    return ("BASELINE", *meaningful_controller_ids(), *null_controller_ids())


def stable_rank(namespace: str, item_id: str) -> str:
    return hashlib.sha256(f"{namespace}\x1f{item_id}".encode()).hexdigest()


def stable_seed(namespace: str, *parts: str | int) -> int:
    payload = "\x1f".join([namespace, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**31 - 1)


def ordered_id_hash(item_ids: Sequence[str]) -> str:
    import json

    payload = json.dumps(list(item_ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def deterministic_allocate(
    rows: Iterable[Mapping[str, Any]],
    *,
    provenance_class: str,
    namespace: str,
    count: int,
    excluded: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_ids = excluded or set()
    eligible = [
        dict(row)
        for row in rows
        if row["provenance_class"] == provenance_class
        and str(row["item_id"]) not in excluded_ids
    ]
    eligible.sort(key=lambda row: (stable_rank(namespace, str(row["item_id"])), row["item_id"]))
    if len(eligible) < count:
        raise ValueError(f"insufficient Class-{provenance_class} items for {namespace}")
    return eligible[:count]


def exact_family_qap_permutations() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Enumerate 5! family mappings crossed with 2^5 within-family swaps."""

    return tuple(
        (family_permutation, swaps)
        for family_permutation in permutations(range(5))
        for swaps in product((0, 1), repeat=5)
    )


def identifiability_checks(
    *,
    meaningful_vectors: np.ndarray,
    implemented_amplitudes: Mapping[str, float],
    geometry_matrices: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute the frozen pre-outcome bank-identifiability checks.

    Matrices and amplitudes must be ordered as ``condition_ids()[1:]``.  This
    routine deliberately consumes no semantic outcome.
    """

    names = condition_ids()[1:]
    meaningful_names = meaningful_controller_ids()
    if tuple(implemented_amplitudes) != names:
        raise ValueError("implemented-amplitude ordering differs from frozen controller order")
    if set(geometry_matrices) != {"M0", "M1", "M2"}:
        raise ValueError("Q2 V3 identifiability requires exactly M0/M1/M2")
    vectors = np.asarray(meaningful_vectors, dtype=np.float64)
    if vectors.shape[0] != len(meaningful_names):
        raise ValueError("meaningful vector count differs from frozen design")
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    gram = unit @ unit.T
    eig = np.linalg.eigvalsh(gram)
    effective_rank = float(np.square(np.sum(eig)) / np.sum(np.square(eig)))
    offdiag = np.abs(gram[np.triu_indices(len(unit), 1)])
    checks: dict[str, Any] = {
        "direction_gram_effective_rank": effective_rank,
        "direction_gram_effective_rank_pass": effective_rank >= 5.0,
        "max_absolute_nonantipodal_cosine": float(np.max(offdiag)),
        "max_absolute_nonantipodal_cosine_pass": float(np.max(offdiag)) < 0.95,
    }
    for shell in SHELLS:
        shell_names = [name for name in names if name.endswith(f"_{shell}")]
        radii = np.asarray([implemented_amplitudes[name] for name in shell_names])
        cv = float(np.std(radii, ddof=0) / np.mean(radii))
        checks[f"{shell}.radius_cv"] = cv
        checks[f"{shell}.radius_cv_pass"] = cv <= 0.03
    checks["all_pass"] = all(
        value for key, value in checks.items() if key.endswith("_pass")
    )
    return checks


__all__ = [
    "BOOTSTRAP_SEED",
    "DATASET_REPO",
    "DATASET_REVISION",
    "EVALUATION_SEED_NAMESPACE",
    "EXECUTION_TEACHER_TEXT",
    "EXPERIMENT_ID",
    "LAYER",
    "LOCATIONS",
    "M1_COVARIANCE_NAMESPACE",
    "M2_PROBE_NAMESPACE",
    "MODEL",
    "MODEL_REVISION",
    "NULL_SEEDS",
    "PRIMARY_PANEL_NAMESPACE",
    "SHELLS",
    "SHELL_TARGETS",
    "SOURCE_FAMILIES",
    "base_direction_id",
    "condition_ids",
    "deterministic_allocate",
    "exact_family_qap_permutations",
    "family_payload",
    "meaningful_controller_ids",
    "null_controller_ids",
    "ordered_id_hash",
    "stable_rank",
    "stable_seed",
]
