"""Prospective contracts for the Q1 LiveCodeBench second-task design.

This module is deliberately model-free.  It defines the immutable benchmark
adapter, safe exact-value evaluator, deterministic split/schedule machinery,
the prospectively extended null bank, and the R=4 latent-propensity estimands.
It never reads Q2 artifacts or scientific model outcomes.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from epistemic_geometry.benchmarks.external.semantic_v3 import extract_final_commitment
from epistemic_geometry.experiments.gate6_3 import vector_sha256
from epistemic_geometry.reproducibility import stable_digest, stable_seed

EXPERIMENT_ID = "Q1_SECOND_TASK_LIVECODEBENCH_SPARK2"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_REVISION = MODEL_REVISION
LAYER = 27
ETA = 9.637427952852196
REFERENCE_SCALE = 10.153299177386142
EFFECTIVE_DELTA_NORM = 97.85168930581241
MEANINGFUL_VECTOR_HASH = "e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838"
MEANINGFUL_VECTOR_FILE_SHA256 = (
    "b1630039fcbb829028a0e8f9f521d7e87bb24e831bc81c74a1591a6c39f40772"
)
LIVECODEBENCH_REPOSITORY_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LIVECODEBENCH_DATASET_REVISION = "6f3ac40bbecf81eba15899139d279b077f2816fd"
LIVECODEBENCH_PARQUET_SHA256 = (
    "4826aa00c059d6d47a099606ceed2d0e51d3aeeb1868f1bcf349a038bb64b4b1"
)
CRUXEVAL_DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"

STAGE_A_N = 50
STAGE_A_ROLLOUTS = 2
STAGE_A_CONDITIONS = ("BASELINE", "TEXTUAL_CAREFUL")
STAGE_B_N = 200
STAGE_B_ROLLOUTS = 4
STAGE_B_CONDITIONS = (
    "BASELINE",
    "TEXTUAL_CAREFUL",
    "MEANINGFUL_FIXED_QWEN_L27_D75",
    "RANDOM_R0",
    "RANDOM_R1",
    "RANDOM_R2",
    "RANDOM_R3",
    "RANDOM_R4",
    "RANDOM_R5",
    "RANDOM_R6",
    "RANDOM_R7",
)
RANDOM_NAMES = tuple(name for name in STAGE_B_CONDITIONS if name.startswith("RANDOM_"))
NEW_RANDOM_NAMES = ("RANDOM_R4", "RANDOM_R5", "RANDOM_R6", "RANDOM_R7")
SPLIT_HALF_A = (0, 1)
SPLIT_HALF_B = (2, 3)


@dataclass(frozen=True)
class LiveCodeBenchItem:
    """One official test-output-prediction row after deterministic adaptation."""

    item_id: str
    question_id: str
    test_id: int
    prompt: str
    reference_json: str
    starter_code: str
    question_content: str
    test_input: str
    difficulty: str
    source_index: int

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            {
                "question_content": self.question_content,
                "starter_code": self.starter_code,
                "test_input": self.test_input,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def item_sha256(self) -> str:
        payload = json.dumps(
            {
                "item_id": self.item_id,
                "prompt_sha256": self.prompt_sha256,
                "reference_json": self.reference_json,
                "source_revision": LIVECODEBENCH_DATASET_REVISION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def public_manifest_record(self) -> dict[str, Any]:
        """Return a non-redistributive manifest record with hashes, not content."""

        return {
            "item_id": self.item_id,
            "question_id": self.question_id,
            "test_id": self.test_id,
            "source_index": self.source_index,
            "difficulty": self.difficulty,
            "prompt_sha256": self.prompt_sha256,
            "content_sha256": self.content_sha256,
            "item_sha256": self.item_sha256,
            "reference_type": type(json.loads(self.reference_json)).__name__,
        }


def _test_call(function_name: str, testcase_input: str) -> str:
    arguments = ", ".join(str(testcase_input).split("\n"))
    return f"assert {function_name}({arguments}) == # TODO"


def build_livecodebench_prompt(row: Mapping[str, Any]) -> str:
    """Adapt the official scenario to the frozen CEG final-commitment channel."""

    tests = json.loads(str(row["test"]))
    if not isinstance(tests, list) or len(tests) != 1 or not isinstance(tests[0], dict):
        raise ValueError("LiveCodeBench row must contain exactly one official test case")
    test = tests[0]
    return (
        f"Problem:\n{row['question_content']}\n"
        f"Function:\n```python\n{row['starter_code']}\n```\n\n"
        "Complete the following test case by determining its exact output:\n"
        f"```python\n{_test_call(str(row['function_name']), str(test['input']))}\n```\n\n"
        "End with exactly one line in the form FINAL: <the exact Python or JSON literal "
        "output>. Do not add text after FINAL."
    )


def normalize_livecodebench_row(row: Mapping[str, Any], source_index: int) -> LiveCodeBenchItem:
    tests = json.loads(str(row["test"]))
    if not isinstance(tests, list) or len(tests) != 1 or not isinstance(tests[0], dict):
        raise ValueError("LiveCodeBench row must contain exactly one test")
    test = tests[0]
    expected = json.loads(str(test["output"]))
    if type(expected) not in {bool, int, list, str}:  # noqa: E721 - bool is an int subclass
        raise ValueError(f"unsupported reference type: {type(expected).__name__}")
    item_id = f"{row['question_id']}:{int(row['test_id'])}"
    return LiveCodeBenchItem(
        item_id=item_id,
        question_id=str(row["question_id"]),
        test_id=int(row["test_id"]),
        prompt=build_livecodebench_prompt(row),
        reference_json=str(test["output"]),
        starter_code=str(row["starter_code"]),
        question_content=str(row["question_content"]),
        test_input=str(test["input"]),
        difficulty=str(row["difficulty"]),
        source_index=int(source_index),
    )


def _canonical_value(value: Any) -> Any:
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, list):
        return ["list", [_canonical_value(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [_canonical_value(item) for item in value]]
    if isinstance(value, dict):
        records = [
            [_canonical_value(key), _canonical_value(item)] for key, item in value.items()
        ]
        records.sort(key=lambda record: json.dumps(record[0], sort_keys=True))
        return ["dict", records]
    if isinstance(value, set):
        records = [_canonical_value(item) for item in value]
        records.sort(key=lambda record: json.dumps(record, sort_keys=True))
        return ["set", records]
    raise ValueError(f"unsupported literal type: {type(value).__name__}")


def parse_safe_literal(payload: str) -> Any:
    """Parse a literal without executing model-generated code."""

    try:
        value = ast.literal_eval(payload)
    except (SyntaxError, ValueError, TypeError):
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("payload is not a Python or JSON literal") from exc
    return _canonical_value(value)


def evaluate_livecodebench_output(
    raw_output: str,
    reference_json: str,
    *,
    truncated: bool = False,
    runtime_error: bool = False,
) -> dict[str, Any]:
    """Score one output with exact typed comparison and no ``eval``."""

    if runtime_error:
        return {
            "status": "RUNTIME_ERROR",
            "commitment_valid": False,
            "semantic_evaluable": False,
            "correct": False,
            "canonical_value": None,
            "failure_reason": "runtime error",
        }
    commitment = extract_final_commitment(raw_output, truncated=truncated)
    if not commitment.valid or commitment.payload is None:
        return {
            "status": "TRUNCATION" if truncated else "INVALID_FORMAT",
            "commitment_valid": False,
            "semantic_evaluable": False,
            "correct": False,
            "canonical_value": None,
            "failure_reason": commitment.failure_reason,
        }
    try:
        actual = parse_safe_literal(commitment.payload)
    except ValueError as exc:
        return {
            "status": "UNEVALUABLE",
            "commitment_valid": True,
            "semantic_evaluable": False,
            "correct": False,
            "canonical_value": None,
            "failure_reason": str(exc),
        }
    expected = _canonical_value(json.loads(reference_json))
    correct = actual == expected
    return {
        "status": "VALID_CORRECT" if correct else "VALID_WRONG",
        "commitment_valid": True,
        "semantic_evaluable": True,
        "correct": correct,
        "canonical_value": json.dumps(actual, ensure_ascii=False, separators=(",", ":")),
        "failure_reason": None,
    }


def _select_group_exactly(
    groups: Mapping[str, Sequence[LiveCodeBenchItem]],
    target: int,
    namespace: str,
) -> tuple[str, ...]:
    """Select whole question groups by a frozen hash order and exact-count DP."""

    ordered = sorted(groups, key=lambda key: stable_digest(namespace, key))
    solutions: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered:
        size = len(groups[group])
        for subtotal, selected in sorted(solutions.items(), reverse=True):
            candidate = subtotal + size
            if candidate <= target and candidate not in solutions:
                solutions[candidate] = (*selected, group)
    if target not in solutions:
        raise RuntimeError(f"cannot select exactly {target} rows without splitting a problem")
    return solutions[target]


def split_items(
    items: Sequence[LiveCodeBenchItem],
) -> tuple[list[LiveCodeBenchItem], list[LiveCodeBenchItem], list[LiveCodeBenchItem]]:
    """Build disjoint Stage A, Stage B, and reserve sets at question level."""

    groups: dict[str, list[LiveCodeBenchItem]] = {}
    for item in items:
        groups.setdefault(item.question_id, []).append(item)
    stage_a_groups = set(
        _select_group_exactly(groups, STAGE_A_N, f"{EXPERIMENT_ID}:STAGE_A")
    )
    remaining_groups = {key: value for key, value in groups.items() if key not in stage_a_groups}
    stage_b_groups = set(
        _select_group_exactly(remaining_groups, STAGE_B_N, f"{EXPERIMENT_ID}:STAGE_B")
    )

    def ordered(selected: set[str], namespace: str) -> list[LiveCodeBenchItem]:
        values = [item for item in items if item.question_id in selected]
        return sorted(values, key=lambda item: stable_digest(namespace, item.item_id))

    stage_a = ordered(stage_a_groups, f"{EXPERIMENT_ID}:STAGE_A:ITEM_ORDER")
    stage_b = ordered(stage_b_groups, f"{EXPERIMENT_ID}:STAGE_B:ITEM_ORDER")
    reserve_groups = set(groups) - stage_a_groups - stage_b_groups
    reserve = ordered(reserve_groups, f"{EXPERIMENT_ID}:RESERVE:ITEM_ORDER")
    if len(stage_a) != STAGE_A_N or len(stage_b) != STAGE_B_N:
        raise AssertionError("stage split has incorrect size")
    if ({item.question_id for item in stage_a} & {item.question_id for item in stage_b}) or (
        {item.item_id for item in stage_a} & {item.item_id for item in stage_b}
    ):
        raise AssertionError("stage split leaked a question or item")
    return stage_a, stage_b, reserve


def build_schedule(
    items: Sequence[LiveCodeBenchItem],
    *,
    stage: str,
    conditions: Sequence[str],
    rollouts: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        for rollout in range(rollouts):
            ordered = sorted(
                conditions,
                key=lambda condition: stable_digest(
                    EXPERIMENT_ID, stage, "CONDITION_ORDER", item.item_id, rollout, condition
                ),
            )
            for condition_order, condition in enumerate(ordered):
                rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "stage": stage,
                        "item_id": item.item_id,
                        "item_sha256": item.item_sha256,
                        "condition": condition,
                        "rollout_index": rollout,
                        "condition_order": condition_order,
                        "seed": stable_seed(
                            EXPERIMENT_ID, stage, item.item_id, condition, rollout
                        ),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                    }
                )
    validate_schedule(rows, items, conditions=conditions, rollouts=rollouts, stage=stage)
    return rows


def validate_schedule(
    rows: Sequence[Mapping[str, Any]],
    items: Sequence[LiveCodeBenchItem],
    *,
    conditions: Sequence[str],
    rollouts: int,
    stage: str,
) -> None:
    expected = len(items) * len(conditions) * rollouts
    if len(rows) != expected:
        raise RuntimeError("schedule row count mismatch")
    keys = [
        (str(row["stage"]), str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("schedule contains duplicate logical keys")
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("schedule contains seed collisions")
    if {str(row["stage"]) for row in rows} != {stage}:
        raise RuntimeError("schedule stage mismatch")
    if {str(row["condition"]) for row in rows} != set(conditions):
        raise RuntimeError("schedule condition mismatch")


def _orthogonal_unit(vector: np.ndarray, bases: Sequence[np.ndarray]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
    for basis in bases:
        unit = np.asarray(basis, dtype=np.float64).reshape(-1)
        value -= float(np.dot(value, unit)) * unit
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("prospective null direction is degenerate")
    return value / norm


def build_extended_null_bank(
    meaningful: np.ndarray,
    existing: Mapping[str, np.ndarray],
    paired_differences: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Preserve R0-R3 and add two isotropic plus two sign-shuffled nulls."""

    direction = np.asarray(meaningful, dtype=np.float64).reshape(-1)
    direction /= np.linalg.norm(direction)
    pairs = np.asarray(paired_differences, dtype=np.float64)
    if tuple(existing) != RANDOM_NAMES[:4]:
        raise ValueError("existing null order must be exactly RANDOM_R0..R3")
    bank = {name: np.asarray(existing[name], dtype=np.float64).reshape(-1) for name in existing}
    bases = [direction, *bank.values()]
    records: dict[str, Any] = {}
    for offset, name in enumerate(NEW_RANDOM_NAMES):
        if offset < 2:
            kind = "ISOTROPIC"
            seed = stable_seed(EXPERIMENT_ID, "Qwen", kind, name)
            raw = np.random.default_rng(seed).normal(size=len(direction))
        else:
            kind = "CONSTRUCTION_MATCHED_SIGN_SHUFFLED"
            seed = stable_seed(EXPERIMENT_ID, "Qwen", kind, name)
            signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=len(pairs))
            raw = (pairs * signs[:, None]).mean(axis=0)
        value = _orthogonal_unit(raw, bases)
        bank[name] = value
        bases.append(value)
        records[name] = {
            "kind": kind,
            "seed": seed,
            "canonical_float64_vector_sha256": vector_sha256(value),
        }
    matrix = np.stack([direction, *bank.values()])
    gram = matrix @ matrix.T
    if np.max(np.abs(gram - np.eye(len(gram)))) > 1e-6:
        raise AssertionError("extended null bank is not orthonormal")
    return bank, {
        "experiment_id": EXPERIMENT_ID,
        "meaningful_vector_hash": vector_sha256(direction),
        "existing_nulls_preserved": list(RANDOM_NAMES[:4]),
        "new_records": records,
        "cosine_matrix": gram.tolist(),
        "construction_outcomes_used": False,
    }


def _within_distinct_mean(values: np.ndarray) -> np.ndarray:
    total = values.sum(axis=1)
    return (total * total - np.square(values).sum(axis=1)) / (
        values.shape[1] * (values.shape[1] - 1)
    )


def r_rollout_estimands(
    baseline_errors: np.ndarray, condition_errors: np.ndarray
) -> dict[str, float]:
    """Unbiased latent-propensity estimands for any R>=2.

    The off-diagonal cross term in ``D`` is chosen so R=2 is exactly the
    historical canonical estimator, not merely equal in expectation.
    """

    baseline = np.asarray(baseline_errors, dtype=np.float64)
    condition = np.asarray(condition_errors, dtype=np.float64)
    if baseline.shape != condition.shape or baseline.ndim != 2:
        raise ValueError("error arrays must have identical shape (items, rollouts)")
    n, rollouts = baseline.shape
    if n < 2 or rollouts < 2:
        raise ValueError("estimands require at least two items and two rollouts")
    baseline_mean = baseline.mean(axis=1)
    condition_mean = condition.mean(axis=1)
    b00 = float(np.mean(_within_distinct_mean(baseline)))
    b0j = float(np.mean(baseline_mean * condition_mean))
    denominator = n * (n - 1)
    u00 = float(
        (baseline_mean.sum() ** 2 - np.square(baseline_mean).sum()) / denominator
    )
    u0j = float(
        (
            baseline_mean.sum() * condition_mean.sum()
            - np.dot(baseline_mean, condition_mean)
        )
        / denominator
    )
    full_cross = baseline[:, :, None] * condition[:, None, :]
    diagonal = np.einsum("ij,ij->i", baseline, condition)
    off_diagonal_cross = (full_cross.sum(axis=(1, 2)) - diagonal) / (
        rollouts * (rollouts - 1)
    )
    distance = float(
        np.mean(
            _within_distinct_mean(baseline)
            + _within_distinct_mean(condition)
            - 2.0 * off_diagonal_cross
        )
    )
    rescue = float(np.mean(baseline_mean * (1.0 - condition_mean)))
    damage = float(np.mean((1.0 - baseline_mean) * condition_mean))
    result = {
        "accuracy_baseline": float(1.0 - baseline.mean()),
        "accuracy_condition": float(1.0 - condition.mean()),
        "B00": b00,
        "O00": 1.0 - b00,
        "B0j": b0j,
        "O0j": 1.0 - b0j,
        "G": b00 - b0j,
        "U00": u00,
        "U0j": u0j,
        "C": b00 - b0j - u00 + u0j,
        "D": distance,
        "rescue": rescue,
        "damage": damage,
    }
    if not np.isclose(
        rescue - damage,
        result["accuracy_condition"] - result["accuracy_baseline"],
        atol=1e-12,
    ):
        raise AssertionError("rescue-damage identity failed")
    return result


def split_half_estimands(
    baseline_errors: np.ndarray, condition_errors: np.ndarray
) -> dict[str, dict[str, float]]:
    baseline = np.asarray(baseline_errors)
    condition = np.asarray(condition_errors)
    if baseline.shape[1] != 4:
        raise ValueError("split-half analysis requires exactly four rollouts")
    return {
        "A": r_rollout_estimands(baseline[:, SPLIT_HALF_A], condition[:, SPLIT_HALF_A]),
        "B": r_rollout_estimands(baseline[:, SPLIT_HALF_B], condition[:, SPLIT_HALF_B]),
    }


__all__ = [
    "CRUXEVAL_DATASET_REVISION",
    "EFFECTIVE_DELTA_NORM",
    "ETA",
    "EXPERIMENT_ID",
    "LAYER",
    "LIVECODEBENCH_DATASET_REVISION",
    "LIVECODEBENCH_PARQUET_SHA256",
    "LIVECODEBENCH_REPOSITORY_COMMIT",
    "LiveCodeBenchItem",
    "MEANINGFUL_VECTOR_FILE_SHA256",
    "MEANINGFUL_VECTOR_HASH",
    "MODEL",
    "MODEL_REVISION",
    "NEW_RANDOM_NAMES",
    "RANDOM_NAMES",
    "REFERENCE_SCALE",
    "SPLIT_HALF_A",
    "SPLIT_HALF_B",
    "STAGE_A_CONDITIONS",
    "STAGE_A_N",
    "STAGE_A_ROLLOUTS",
    "STAGE_B_CONDITIONS",
    "STAGE_B_N",
    "STAGE_B_ROLLOUTS",
    "TOKENIZER_REVISION",
    "build_extended_null_bank",
    "build_livecodebench_prompt",
    "build_schedule",
    "evaluate_livecodebench_output",
    "normalize_livecodebench_row",
    "parse_safe_literal",
    "r_rollout_estimands",
    "split_half_estimands",
    "split_items",
    "validate_schedule",
]
