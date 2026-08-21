"""Pure, model-free contracts for the Gate 7 fresh L27 replication."""

from __future__ import annotations

import hashlib
import re
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
from epistemic_geometry.experiments.gate6_3_v3 import audit_two_rollout_estimands
from epistemic_geometry.reproducibility import canonical_json, stable_digest, stable_seed

EXPERIMENT_ID = "GATE7_FRESH_SINGLE_L27_REPLICATION"
SELECTION_NAMESPACE = "GATE7-FRESH-L27-REPLICATION-V1"
DATASET_REPO = "cruxeval-org/cruxeval"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MEANINGFUL = "BEST_SINGLE_MEAN_PLUS"
TEXTUAL = "TEXTUAL_CAREFUL_REFERENCE"
BASELINE = "BASELINE"
RANDOM_NAMES = tuple(f"GATE7_RANDOM_R{i}" for i in range(4))
CONDITIONS = (BASELINE, TEXTUAL, MEANINGFUL, *RANDOM_NAMES)
PARSER_VERSION = "external-semantic-v3"
LAYER = 27
ETA = 12.849903937136261
REFERENCE_SCALE = 10.153299177386142
MAX_NEW_TOKENS = 4096
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260821
SAMPLE_ID = re.compile(r"\bsample_[0-9]+\b")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_cruxeval_ids(
    review_root: Path, *, gate7_output: Path | None = None
) -> tuple[str, ...]:
    """Return every historically allocated CRUXEval ID in preserved artifacts."""

    found: set[str] = set()
    output_resolved = gate7_output.resolve() if gate7_output is not None else None
    for path in review_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md"}:
            continue
        if output_resolved is not None and output_resolved in path.resolve().parents:
            continue
        try:
            found.update(SAMPLE_ID.findall(path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    if not found:
        raise RuntimeError("no historical CRUXEval IDs found")
    return tuple(sorted(found, key=lambda value: int(value.split("_")[1])))


def task_prompt(code: str, value: str) -> str:
    return (
        "Solve this Python code-output prediction problem.\n\n"
        "Function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    )


def normalize_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    item_id = str(row.get("id", row.get("item_id")))
    prompt = (
        str(row["prompt"]) if "prompt" in row else task_prompt(str(row["code"]), str(row["input"]))
    )
    reference = str(row.get("output", row.get("reference_answer")))
    reference_type = _reference_type(reference)
    prompt_hash = stable_digest("GATE7-TASK-PROMPT", prompt)
    item_hash = stable_digest(
        "GATE7-ITEM", item_id, prompt_hash, reference, "python_literal", DATASET_REVISION
    )
    return {
        "allocation": "GATE7_EVALUATION",
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


def _reference_type(reference: str) -> str:
    from epistemic_geometry.benchmarks.external.semantic_v3 import canonicalize_semantic_value

    return str(canonicalize_semantic_value(reference)[0])


def allocate_fresh_items(
    candidates: Sequence[Mapping[str, Any]], historical_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen 120/100 availability contingency without model outcomes."""

    excluded = set(map(str, historical_ids))
    normalized = [normalize_dataset_row(row) for row in candidates]
    if len({row["item_id"] for row in normalized}) != len(normalized):
        raise RuntimeError("pinned CRUXEval candidates contain duplicate IDs")
    eligible = [row for row in normalized if row["item_id"] not in excluded]
    eligible.sort(
        key=lambda row: (stable_digest(SELECTION_NAMESPACE, row["item_id"]), row["item_id"])
    )
    if len(eligible) >= 120:
        actual_n = 120
    elif len(eligible) >= 100:
        actual_n = 100
    else:
        raise RuntimeError(f"GATE7_BLOCKED_INSUFFICIENT_FRESH_ITEMS: {len(eligible)} < 100")
    selected = eligible[:actual_n]
    type_counts: dict[str, int] = {}
    for row in selected:
        value = str(row["reference_canonical_type"])
        type_counts[value] = type_counts.get(value, 0) + 1
    manifest_hash = stable_digest("GATE7-EVALUATION-MANIFEST", canonical_json(selected))
    summary = {
        "requested_n": 120,
        "actual_n": actual_n,
        "eligible_n": len(eligible),
        "historical_excluded_count": len(excluded),
        "historical_exclusion_digest": stable_digest(
            SELECTION_NAMESPACE, "HISTORICAL-EXCLUSION", canonical_json(sorted(excluded))
        ),
        "manifest_hash": manifest_hash,
        "reference_type_distribution": dict(sorted(type_counts.items())),
        "selection_namespace": SELECTION_NAMESPACE,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
    }
    return selected, summary


def gate7_random_bank(meaningful: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seeds = tuple(stable_seed("GATE7-L27-RANDOM-BANK-V1", index) for index in range(4))
    short_bank = single_layer_random_bank(meaningful, seeds=seeds)
    bank = {name: short_bank[f"R{index}"] for index, name in enumerate(RANDOM_NAMES)}
    geometry = bank_geometry(meaningful, bank)
    if not all(
        geometry[key]
        for key in (
            "unit_norm_pass",
            "meaningful_orthogonality_pass",
            "random_pairwise_orthogonality_pass",
        )
    ):
        raise RuntimeError(f"Gate 7 random-bank geometry failed: {geometry}")
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
    """Freeze a condition-interleaved schedule and globally unique independent seeds."""

    logical: list[dict[str, Any]] = []
    for item_index, item_id in enumerate(item_ids):
        for rollout in (0, 1):
            condition_order = sorted(
                CONDITIONS,
                key=lambda condition: (
                    stable_digest(SELECTION_NAMESPACE, "ORDER", item_id, rollout, condition),
                    condition,
                ),
            )
            for order_index, condition in enumerate(condition_order):
                logical.append(
                    {
                        "phase": "GATE7_PRIMARY_REPLICATION",
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
        raise RuntimeError("Gate 7 schedule contains duplicate logical keys")
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Gate 7 independent seed bank contains a collision")
    return logical


def classify_gate7(
    *,
    baseline: Mapping[str, float],
    controller: Mapping[str, float],
    controller_estimands: Mapping[str, float],
    random_summary: Mapping[str, Mapping[str, float]],
    bootstrap: Mapping[str, Mapping[str, float]],
    loo_sign_stable: Mapping[str, bool],
    controller_style_replicated: bool,
) -> tuple[str, dict[str, Any]]:
    commitment_guard = bool(
        controller["commitment_validity"] >= 0.90
        and controller["commitment_validity"] >= baseline["commitment_validity"] - 0.05
    )
    evaluability_guard = bool(
        controller["semantic_evaluability"] >= 0.90
        and controller["semantic_evaluability"] >= baseline["semantic_evaluability"] - 0.05
    )
    competence_guard = bool(controller["accuracy"] >= baseline["accuracy"] - 0.10)
    minimum = bool(
        commitment_guard
        and evaluability_guard
        and competence_guard
        and controller_estimands["G"] >= 0.10
        and controller_estimands["C"] >= 0.05
        and controller_estimands["D"] >= 0.08
        and controller_estimands["G"] - random_summary["G"]["mean"] >= 0.08
        and controller_estimands["C"] - random_summary["C"]["mean"] >= 0.05
        and controller_estimands["D"] - random_summary["D"]["mean"] >= 0.05
        and controller_estimands["G"] > random_summary["G"]["max"]
        and controller_estimands["C"] > random_summary["C"]["max"]
        and controller_estimands["D"] > random_summary["D"]["max"]
        and controller_estimands["rescue"] > controller_estimands["damage"]
    )
    strong_interval_names = (
        "meaningful:accuracy_change",
        "meaningful:G",
        "meaningful:C",
        "meaningful:G_minus_random_mean",
        "meaningful:C_minus_random_mean",
    )
    strong = bool(
        minimum
        and all(float(bootstrap[name]["q025"]) > 0 for name in strong_interval_names)
        and controller["accuracy"] - baseline["accuracy"] >= 0.08
        and all(loo_sign_stable.get(metric, False) for metric in ("accuracy_change", "G", "C"))
    )
    qualitative = bool(
        commitment_guard
        and evaluability_guard
        and competence_guard
        and all(controller_estimands[metric] > 0 for metric in ("G", "C", "D"))
        and all(
            controller_estimands[metric] > random_summary[metric]["mean"]
            for metric in ("G", "C", "D")
        )
        and controller_estimands["rescue"] > controller_estimands["damage"]
    )
    if not commitment_guard or not evaluability_guard or not competence_guard:
        classification = "GATE7_DESTRUCTIVE"
    elif strong:
        classification = "GATE7_STRONG_SINGLE_L27_REPLICATION"
    elif minimum:
        classification = "GATE7_MINIMUM_SINGLE_L27_REPLICATION"
    elif qualitative:
        classification = "GATE7_QUALITATIVE_PARTIAL_REPLICATION"
    elif controller_style_replicated:
        classification = "GATE7_CAREFUL_STYLE_CONTROL_WITHOUT_SPECIFIC_ERROR_CONTROL"
    else:
        classification = "GATE7_NO_REPLICATION"
    return classification, {
        "commitment_validity_guard": commitment_guard,
        "semantic_evaluability_guard": evaluability_guard,
        "competence_guard": competence_guard,
        "minimum_specific_replication": minimum,
        "strong_replication": strong,
        "qualitative_partial_replication": qualitative,
    }


def pseudo_replication_projection(
    baseline: np.ndarray,
    conditions: Mapping[str, np.ndarray],
    *,
    target_n: int,
    resamples: int = 2_000,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Describe precision scaling using only preserved Gate 6.3 item clusters."""

    base = np.asarray(baseline, dtype=np.int8)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(resamples):
        indices = rng.integers(0, len(base), size=target_n)
        points = {
            name: audit_two_rollout_estimands(base[indices], np.asarray(values)[indices])
            for name, values in conditions.items()
        }
        random_names = tuple(name for name in points if name.startswith("SINGLE_L27_RANDOM_"))
        meaningful = points[MEANINGFUL]
        for metric in ("G", "C", "D"):
            samples.setdefault(metric, []).append(float(meaningful[metric]))
            random_mean = float(np.mean([points[name][metric] for name in random_names]))
            samples.setdefault(f"{metric}_minus_random_mean", []).append(
                float(meaningful[metric] - random_mean)
            )
        samples.setdefault("accuracy_change", []).append(
            float(meaningful["accuracy_condition"] - meaningful["accuracy_baseline"])
        )
    return {
        "source_item_count": len(base),
        "target_item_count": target_n,
        "bootstrap_width_scaling_sqrt_60_over_n": float(np.sqrt(len(base) / target_n)),
        "resamples": resamples,
        "seed": seed,
        "distributions": {
            name: {
                "mean": float(np.mean(values)),
                "q025": float(np.quantile(values, 0.025)),
                "q50": float(np.quantile(values, 0.5)),
                "q975": float(np.quantile(values, 0.975)),
            }
            for name, values in sorted(samples.items())
        },
        "decision_effect": "none; sample-size rule was frozen independently",
    }


__all__ = [
    "BASELINE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONDITIONS",
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
    "classify_gate7",
    "file_sha256",
    "gate7_random_bank",
    "historical_cruxeval_ids",
    "normalize_dataset_row",
    "pseudo_replication_projection",
    "task_prompt",
    "vector_sha256",
]
