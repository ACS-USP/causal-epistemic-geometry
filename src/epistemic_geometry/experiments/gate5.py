"""Pure Gate-5 source/duration bridge helpers.

The module contains no model-loading code.  It owns the frozen seed schedule,
random-controller construction, source-disagreement summaries, independent
two-rollout estimands, and outcome-independent classification rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from epistemic_geometry.experiments.micro_q1 import vector_sha256
from epistemic_geometry.reproducibility import stable_seed

MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
LAYER = 17
ALPHA = 8.39900588973121
CONDITIONS = (
    "BASELINE",
    "ONE_SHOT_PLUS",
    "ONE_SHOT_MINUS",
    "SUSTAINED_PLUS",
    "SUSTAINED_MINUS",
    "SUSTAINED_RANDOM_R0",
    "SUSTAINED_RANDOM_R1",
    "SUSTAINED_RANDOM_R2",
    "SUSTAINED_RANDOM_R3",
)
SOURCE_CONDITIONS = ("ORDINARY", "CAREFUL", "DIRECT")
SYSTEM_CAREFUL = (
    "You are a meticulous program tracer. Carefully track every operation, mutation, "
    "intermediate value, branch, and loop. Verify the result before answering. End with "
    "exactly one line in the form FINAL: <answer>."
)
SYSTEM_DIRECT = (
    "Answer the program-output question immediately. Do not trace, deliberate, explain, "
    "or verify. End with exactly one line in the form FINAL: <answer>."
)
RANDOM_SEEDS = (2026082001, 2026082002, 2026082003)


def source_seed(item_id: str, condition: str, rollout_index: int) -> int:
    return stable_seed(
        "GATE5-SOURCE-DURATION",
        "SOURCE_CHECK",
        item_id,
        condition,
        rollout_index,
        "INDEPENDENT_PRIMARY",
    )


def manipulation_seed(item_id: str) -> int:
    return stable_seed(
        "GATE5-SOURCE-DURATION",
        "SUSTAINED_MANIPULATION",
        item_id,
        "MATCHED_COUPLING_SECONDARY",
    )


def evaluation_seed(item_id: str, condition: str, rollout_index: int) -> int:
    return stable_seed(
        "GATE5-SOURCE-DURATION",
        "SUSTAINED_EVALUATION",
        item_id,
        condition,
        rollout_index,
        "INDEPENDENT_PRIMARY",
    )


def _orthogonalize(candidate: np.ndarray, basis: Sequence[np.ndarray]) -> np.ndarray:
    vector = np.asarray(candidate, dtype=np.float64).copy()
    for base in basis:
        base_array = np.asarray(base, dtype=np.float64)
        vector -= float(np.dot(vector, base_array)) * base_array
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("random controller became degenerate during Gram-Schmidt")
    return vector / norm


def random_controller_bank(
    meaningful: np.ndarray,
    r0: np.ndarray,
    *,
    seeds: Sequence[int] = RANDOM_SEEDS,
) -> dict[str, np.ndarray]:
    """Construct R0-R3 with deterministic Gaussian draws and Gram-Schmidt."""

    v = np.asarray(meaningful, dtype=np.float64).reshape(-1)
    # R0 is the exact Gate-4 random controller.  Do not re-normalize or
    # re-orthogonalize it, because even a numerically tiny rewrite changes its
    # frozen artifact hash.
    r0_array = np.asarray(r0, dtype=np.float64).reshape(-1).copy()
    if v.shape != r0_array.shape:
        raise ValueError("meaningful and R0 dimensions differ")
    if not np.isclose(np.linalg.norm(r0_array), 1.0, atol=1e-12):
        raise ValueError("Gate-4 R0 is not unit norm")
    if abs(float(np.dot(v, r0_array))) > 1e-6:
        raise ValueError("Gate-4 R0 is not orthogonal to the meaningful direction")
    bank: dict[str, np.ndarray] = {"R0": r0_array}
    basis = [v, r0_array]
    for index, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(int(seed))
        bank[f"R{index}"] = _orthogonalize(rng.standard_normal(v.size), basis)
        basis.append(bank[f"R{index}"])
    return bank


def controller_metadata(
    bank: Mapping[str, np.ndarray], *, seeds: Sequence[int] = RANDOM_SEEDS
) -> dict[str, Any]:
    names = tuple(bank)
    values: dict[str, Any] = {
        "dimension": int(next(iter(bank.values())).size),
        "seeds": {"R0": "GATE4"},
    }
    for index, name in enumerate(names):
        vector = np.asarray(bank[name], dtype=np.float64)
        values[name] = {
            "norm": float(np.linalg.norm(vector)),
            "sha256": vector_sha256(vector),
            "seed": None if name == "R0" else int(seeds[index - 1]),
        }
    values["pairwise_absolute_cosines"] = {
        f"{left}:{right}": float(abs(np.dot(bank[left], bank[right])))
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    return values


def _error(status: str) -> bool:
    return str(status) != "VALID_CORRECT"


def _matrix(rows: Sequence[Sequence[bool]]) -> np.ndarray:
    result = np.asarray(rows, dtype=bool)
    if result.ndim != 2 or result.shape[1] != 2 or result.shape[0] < 2:
        raise ValueError("Gate-5 independent estimands require [items, 2] matrices")
    return result


def independent_estimands(
    baseline_errors: Sequence[Sequence[bool]], condition_errors: Sequence[Sequence[bool]]
) -> dict[str, float]:
    """Compute Gate-4/5 independent two-rollout estimands directly."""

    baseline = _matrix(baseline_errors).astype(float)
    condition = _matrix(condition_errors).astype(float)
    if baseline.shape != condition.shape:
        raise ValueError("baseline and condition item banks must have equal shape")
    b1, b2 = baseline[:, 0], baseline[:, 1]
    j1, j2 = condition[:, 0], condition[:, 1]
    q0, qj = baseline.mean(axis=1), condition.mean(axis=1)
    n = len(q0)
    b00 = float(np.mean(b1 * b2))
    b0j = float(np.mean((b1 * j1 + b1 * j2 + b2 * j1 + b2 * j2) / 4.0))
    u00 = float((q0.sum() ** 2 - np.dot(q0, q0)) / (n * (n - 1)))
    u0j = float((q0.sum() * qj.sum() - np.dot(q0, qj)) / (n * (n - 1)))
    rescue = float(np.mean((b1 * (1 - j1) + b1 * (1 - j2) + b2 * (1 - j1) + b2 * (1 - j2)) / 4.0))
    damage = float(np.mean(((1 - b1) * j1 + (1 - b1) * j2 + (1 - b2) * j1 + (1 - b2) * j2) / 4.0))
    return {
        "B00": b00,
        "B0j": b0j,
        "O00": 1.0 - b00,
        "O0j": 1.0 - b0j,
        "G": b00 - b0j,
        "U00": u00,
        "U0j": u0j,
        "C": b00 - b0j - u00 + u0j,
        "D": float(np.mean(b1 * b2 + j1 * j2 - b1 * j2 - b2 * j1)),
        "rescue": rescue,
        "damage": damage,
        "accuracy_baseline": float(1.0 - baseline.mean()),
        "accuracy_condition": float(1.0 - condition.mean()),
    }


def source_disagreement(
    outcomes: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, float]:
    """Compute X, W, and S using semantic outcome/status categories."""

    item_ids = tuple(outcomes["CAREFUL"])
    if set(outcomes) != set(SOURCE_CONDITIONS):
        raise ValueError("source outcomes must contain ordinary, careful, and direct")
    careful = outcomes["CAREFUL"]
    direct = outcomes["DIRECT"]
    cross = []
    within_careful = []
    within_direct = []
    for item_id in item_ids:
        c = tuple(careful[item_id])
        d = tuple(direct[item_id])
        if len(c) != 2 or len(d) != 2:
            raise ValueError("source check requires two rollouts per textual condition")
        cross.extend(left != right for left in c for right in d)
        within_careful.append(c[0] != c[1])
        within_direct.append(d[0] != d[1])
    x = float(np.mean(cross))
    w = float(0.5 * (np.mean(within_careful) + np.mean(within_direct)))
    return {"X_cross_disagreement": x, "W_within_disagreement": w, "S_excess": x - w}


def classify_source(metrics: Mapping[str, float]) -> str:
    if (
        metrics["careful_validity"] >= 0.90
        and metrics["direct_validity"] >= 0.90
        and metrics["X_cross_disagreement"] >= 0.10
        and metrics["S_excess"] >= 0.05
    ):
        return "SOURCE_SEMANTIC_BEHAVIOR_PASS"
    if (
        metrics["careful_validity"] >= 0.90
        and metrics["direct_validity"] >= 0.90
        and metrics["careful_mean_tokens"] >= 1.25 * metrics["direct_mean_tokens"]
        and metrics["careful_median_tokens"] >= metrics["direct_median_tokens"] + 2
    ):
        return "SOURCE_COMPUTATION_STYLE_ONLY"
    return "SOURCE_NO_BEHAVIORAL_SEPARATION"


def classify_manipulation(metrics: Mapping[str, Mapping[str, float]]) -> bool:
    def at_least(value: float, threshold: float) -> bool:
        # The rates are exact rational counts, but their float representation
        # can land one ulp below a frozen boundary (for example 1/20).
        return value >= threshold or bool(np.isclose(value, threshold, rtol=0.0, atol=1e-12))

    random = [metrics[f"SUSTAINED_RANDOM_R{i}"]["semantic_change_rate"] for i in range(4)]
    mean_random = float(np.mean(random))
    for sign in ("PLUS", "MINUS"):
        sustained = metrics[f"SUSTAINED_{sign}"]
        one_shot = metrics[f"ONE_SHOT_{sign}"]
        if (
            at_least(sustained["validity"], 0.85)
            and at_least(sustained["semantic_change_rate"], 0.15)
            and at_least(
                sustained["semantic_change_rate"] - one_shot["semantic_change_rate"],
                0.05,
            )
            and at_least(sustained["semantic_change_rate"] - mean_random, 0.05)
        ):
            return True
    return False


def classify_gate5(
    estimands: Mapping[str, Mapping[str, float]],
    *,
    engineering_pass: bool,
    manipulation_pass: bool,
) -> str:
    if not engineering_pass:
        return "GATE5_SUSTAINED_ENGINE_FAILURE"
    if not manipulation_pass:
        return "GATE5_NO_BEHAVIORAL_FIRST_STAGE"
    movement = {}
    useful = {}
    random_names = [f"SUSTAINED_RANDOM_R{i}" for i in range(4)]
    random_d = [estimands[name]["D"] for name in random_names]
    random_c = [estimands[name]["C"] for name in random_names]
    for sign in ("PLUS", "MINUS"):
        name = f"SUSTAINED_{sign}"
        one_shot = estimands[f"ONE_SHOT_{sign}"]
        current = estimands[name]
        valid_guard = (
            current["validity"] >= 0.90
            and current["validity"] >= estimands["BASELINE"]["validity"] - 0.05
        )
        competence_guard = current["accuracy"] >= estimands["BASELINE"]["accuracy"] - 0.10
        movement[name] = bool(
            valid_guard
            and competence_guard
            and current["D"] >= 0.05
            and current["D"] - float(np.mean(random_d)) >= 0.05
            and current["D"] > max(random_d)
            and current["D"] - one_shot["D"] >= 0.03
        )
        useful[name] = bool(
            movement[name]
            and current["G"] >= 0.03
            and current["C"] >= 0.03
            and current["C"] - float(np.mean(random_c)) >= 0.05
            and current["C"] > max(random_c)
        )
    if any(useful.values()):
        return "GATE5_SUSTAINED_USEFUL_COMPLEMENTARITY_SIGNAL"
    if any(movement.values()):
        return "GATE5_SUSTAINED_ERROR_PROFILE_MOVEMENT_ONLY"
    if all(
        not (
            estimands[f"SUSTAINED_{sign}"]["validity"] >= 0.90
            and estimands[f"SUSTAINED_{sign}"]["validity"]
            >= estimands["BASELINE"]["validity"] - 0.05
            and estimands[f"SUSTAINED_{sign}"]["accuracy"]
            >= estimands["BASELINE"]["accuracy"] - 0.10
        )
        for sign in ("PLUS", "MINUS")
    ):
        return "GATE5_SUSTAINED_DESTRUCTIVE"
    if any(
        estimands[f"SUSTAINED_{sign}"]["D"] - estimands[f"ONE_SHOT_{sign}"]["D"] >= 0.03
        for sign in ("PLUS", "MINUS")
    ):
        return "GATE5_DURATION_EFFECT_BELOW_MOVEMENT_THRESHOLD"
    return "GATE5_NO_DURATION_EFFECT"
