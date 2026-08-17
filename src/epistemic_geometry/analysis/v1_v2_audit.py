"""Independent, analysis-only audit of the Q1 V1.2 score aggregation.

This module intentionally does not import a model backend or a benchmark
loader.  It consumes only the frozen raw candidate-score artifact and
recomputes the two aggregation rules used by V1.2:

* S: mean centered candidate logits across cyclic orderings (primary);
* Q: mean per-ordering candidate probabilities (secondary diagnostic).

The separation from :mod:`experiments.q1_v1_2` is deliberate.  A future
review can therefore compare the stored run output with an independently
implemented calculation without performing new inference.
"""

# This report generator contains intentionally long Markdown/table literals.
# Keep ordinary Ruff checks active while exempting those presentation lines.
# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROLES = ("baseline", "pc1_minus", "pc1_plus", "probe_minus", "probe_plus")
PRIMARY_ROLES = ("baseline", "pc1_minus", "pc1_plus")
PROTOCOL = "Q1_DEVELOPMENT_PROTOCOL_V1_2"
EXPECTED_DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
EXPECTED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_PC1_HASH = "abca43ae3b9621614562798dbfbd8c3ad9932fc9fcb0cfd2c58d28adc48897c5"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(_json_safe(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(values.max())
    probabilities = np.exp(shifted)
    return probabilities / float(probabilities.sum())


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _argmax(values: list[float] | np.ndarray) -> int:
    """Use NumPy's first-maximum rule, matching the production implementation."""

    return int(np.argmax(np.asarray(values, dtype=np.float64)))


def validate_raw_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate raw scientific keys and reconstruct the item/order index."""

    if not rows:
        raise ValueError("Raw V1.2 artifact is empty")
    keys: set[tuple[str, int, str]] = set()
    item_order: list[str] = []
    item_options: dict[str, int] = {}
    item_targets: dict[str, int] = {}
    item_prompt_hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
    shifts_by_item: dict[str, set[int]] = defaultdict(set)
    roles = Counter()
    score_semantics = set()
    invalid_prediction_count = 0

    for row in rows:
        required = {
            "item_id",
            "cyclic_shift",
            "role",
            "option_count",
            "candidate_labels",
            "semantic_option_ids",
            "candidate_scores",
            "target_semantic_original_index",
            "predicted_semantic_original_index",
            "predicted_displayed_label",
            "target_displayed_label",
            "correct",
            "condition_spec",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"Raw row missing fields {sorted(missing)}")
        item_id = str(row["item_id"])
        shift = int(row["cyclic_shift"])
        role = str(row["role"])
        key = (item_id, shift, role)
        if key in keys:
            raise ValueError(f"Duplicate raw scientific key: {key}")
        keys.add(key)
        if item_id not in item_options:
            item_order.append(item_id)
        option_count = int(row["option_count"])
        labels = list(row["candidate_labels"])
        semantic_ids = [int(value) for value in row["semantic_option_ids"]]
        scores = row["candidate_scores"]
        if item_id in item_options and item_options[item_id] != option_count:
            raise ValueError(f"Option count changes within item {item_id}")
        if len(labels) != option_count or len(semantic_ids) != option_count:
            raise ValueError(f"Candidate length mismatch for {key}")
        if len(set(labels)) != option_count or sorted(semantic_ids) != list(range(option_count)):
            raise ValueError(f"Candidate labels/semantic IDs are not a permutation for {key}")
        if set(scores) != set(labels):
            raise ValueError(f"Candidate score labels mismatch for {key}")
        if role not in ROLES:
            raise ValueError(f"Unexpected raw role {role!r}")
        if shift < 0 or shift >= option_count:
            raise ValueError(f"Invalid cyclic shift for {key}")
        if row["condition"] != f"cyclic_{shift:02d}_{role}":
            raise ValueError(f"Condition name mismatch for {key}")
        spec = row["condition_spec"]
        if spec.get("role") != role or int(spec.get("cyclic_shift", -1)) != shift:
            raise ValueError(f"Condition spec mismatch for {key}")
        item_options[item_id] = option_count
        target = int(row["target_semantic_original_index"])
        if item_id in item_targets and item_targets[item_id] != target:
            raise ValueError(f"Target changes within item {item_id}")
        item_targets[item_id] = target
        item_prompt_hashes[(item_id, shift)].add(str(row.get("rendered_prompt_hash", "")))
        shifts_by_item[item_id].add(shift)
        roles[role] += 1
        score_semantics.add(row.get("candidate_score_semantics"))
        predicted_label = max(labels, key=lambda label: float(scores[label]))
        predicted_index = semantic_ids[labels.index(predicted_label)]
        expected_target_label = labels[semantic_ids.index(target)]
        if (
            row["predicted_displayed_label"] != predicted_label
            or int(row["predicted_semantic_original_index"]) != predicted_index
            or row["target_displayed_label"] != expected_target_label
            or bool(row["correct"]) != (predicted_index == target)
        ):
            invalid_prediction_count += 1

    if set(roles) != set(ROLES):
        raise ValueError(f"Raw roles differ from expected roles: {sorted(roles)}")
    for item_id in item_order:
        expected_shifts = set(range(item_options[item_id]))
        if shifts_by_item[item_id] != expected_shifts:
            raise ValueError(f"Cyclic shifts incomplete for {item_id}")
        for shift in shifts_by_item[item_id]:
            if len(item_prompt_hashes[(item_id, shift)]) != 1:
                raise ValueError(
                    f"Prompt hash changes across conditions for {item_id}, shift {shift}"
                )
    if invalid_prediction_count:
        raise ValueError(
            f"Raw prediction fields disagree with candidate scores: {invalid_prediction_count}"
        )
    return {
        "row_count": len(rows),
        "item_count": len(item_order),
        "item_order": item_order,
        "item_options": item_options,
        "item_targets": item_targets,
        "roles": dict(roles),
        "score_semantics": sorted(str(value) for value in score_semantics),
        "expected_row_count": sum(item_options.values()) * len(ROLES),
    }


def recompute_aggregates(
    rows: list[dict[str, Any]], index: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute S/Q rows and per-item cyclic instability without model code."""

    by_key = {
        (str(row["item_id"]), int(row["cyclic_shift"]), str(row["role"])): row for row in rows
    }
    sym_rows: list[dict[str, Any]] = []
    instability_rows: list[dict[str, Any]] = []
    for item_id in index["item_order"]:
        option_count = int(index["item_options"][item_id])
        target = int(index["item_targets"][item_id])
        for role in PRIMARY_ROLES:
            semantic_logits_by_shift: list[list[float]] = []
            semantic_probs_by_shift: list[list[float]] = []
            shift_s_predictions: list[int] = []
            shift_q_predictions: list[int] = []
            for shift in range(option_count):
                row = by_key[(item_id, shift, role)]
                labels = list(row["candidate_labels"])
                values = np.asarray([float(row["candidate_scores"][label]) for label in labels])
                centered = values - float(values.mean())
                probabilities = _softmax(values)
                semantic_ids = [int(value) for value in row["semantic_option_ids"]]
                semantic_logits = [0.0] * option_count
                semantic_probs = [0.0] * option_count
                for displayed_index, semantic_index in enumerate(semantic_ids):
                    semantic_logits[semantic_index] = float(centered[displayed_index])
                    semantic_probs[semantic_index] = float(probabilities[displayed_index])
                semantic_logits_by_shift.append(semantic_logits)
                semantic_probs_by_shift.append(semantic_probs)
                shift_s_predictions.append(_argmax(semantic_logits))
                shift_q_predictions.append(_argmax(semantic_probs))
            s_scores = np.mean(np.asarray(semantic_logits_by_shift), axis=0)
            q_scores = np.mean(np.asarray(semantic_probs_by_shift), axis=0)
            s_prediction = _argmax(s_scores)
            q_prediction = _argmax(q_scores)
            s_ordered = np.sort(s_scores)[::-1]
            q_ordered = np.sort(q_scores)[::-1]
            s_margin = float(s_ordered[0] - s_ordered[1])
            q_margin = float(q_ordered[0] - q_ordered[1])
            sym_rows.append(
                {
                    "item_id": item_id,
                    "condition": f"{role}_sym",
                    "role": role,
                    "option_count": option_count,
                    "centered_logit_scores": s_scores.tolist(),
                    "probability_mean_scores": q_scores.tolist(),
                    "predicted_semantic_original_index": s_prediction,
                    "probability_mean_prediction": q_prediction,
                    "target_semantic_original_index": target,
                    "correct": s_prediction == target,
                    "symmetrized_margin": s_margin,
                    "probability_mean_margin": q_margin,
                }
            )
            instability_rows.append(
                {
                    "item_id": item_id,
                    "role": role,
                    "option_count": option_count,
                    "s_score_sd_mean": float(
                        np.asarray(semantic_logits_by_shift).std(axis=0).mean()
                    ),
                    "s_score_sd_max": float(np.asarray(semantic_logits_by_shift).std(axis=0).max()),
                    "q_score_sd_mean": float(
                        np.asarray(semantic_probs_by_shift).std(axis=0).mean()
                    ),
                    "q_score_sd_max": float(np.asarray(semantic_probs_by_shift).std(axis=0).max()),
                    "s_prediction_mode_share": _mode_share(shift_s_predictions),
                    "q_prediction_mode_share": _mode_share(shift_q_predictions),
                    "s_unique_shift_predictions": len(set(shift_s_predictions)),
                    "q_unique_shift_predictions": len(set(shift_q_predictions)),
                }
            )
    return sym_rows, instability_rows


def _mode_share(values: list[int]) -> float:
    counts = Counter(values)
    return float(max(counts.values()) / len(values)) if values else float("nan")


def _phi(errors_a: np.ndarray, errors_b: np.ndarray) -> tuple[float | None, str]:
    a = errors_a.astype(np.float64)
    b = errors_b.astype(np.float64)
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator == 0:
        return None, "undefined_zero_variance"
    return float(np.dot(a_centered, b_centered) / denominator), "defined"


def paired_metrics(
    baseline: dict[str, bool], treatment: dict[str, bool], treatment_label: str
) -> dict[str, Any]:
    """Compute the Q1 paired metrics directly from correctness booleans."""

    if set(baseline) != set(treatment) or not baseline:
        raise ValueError("Paired metric inputs must have identical non-empty item IDs")
    item_ids = list(baseline)
    base_correct = np.asarray([baseline[item_id] for item_id in item_ids], dtype=bool)
    treat_correct = np.asarray([treatment[item_id] for item_id in item_ids], dtype=bool)
    base_errors = ~base_correct
    treat_errors = ~treat_correct
    counts = {
        "baseline_correct__treatment_correct": int(
            np.logical_and(base_correct, treat_correct).sum()
        ),
        "baseline_correct__treatment_wrong": int(np.logical_and(base_correct, treat_errors).sum()),
        "baseline_wrong__treatment_correct": int(np.logical_and(base_errors, treat_correct).sum()),
        "baseline_wrong__treatment_wrong": int(np.logical_and(base_errors, treat_errors).sum()),
    }
    phi, phi_status = _phi(base_errors, treat_errors)
    union = int(np.logical_or(base_errors, treat_errors).sum())
    intersection = int(np.logical_and(base_errors, treat_errors).sum())
    baseline_error_count = int(base_errors.sum())
    baseline_success_count = int(base_correct.sum())
    pair_oracle = float(np.logical_or(base_correct, treat_correct).mean())
    baseline_accuracy = float(base_correct.mean())
    treatment_accuracy = float(treat_correct.mean())
    return {
        "n_items": len(item_ids),
        "treatment_condition": treatment_label,
        "baseline_accuracy": baseline_accuracy,
        "treatment_accuracy": treatment_accuracy,
        "delta_accuracy": treatment_accuracy - baseline_accuracy,
        "error_correlation_phi": phi,
        "error_correlation_phi_status": phi_status,
        "error_jaccard": 1.0 if union == 0 else intersection / union,
        "disagreement_rate": float(np.not_equal(base_errors, treat_errors).mean()),
        "double_fault": float(np.logical_and(base_errors, treat_errors).mean()),
        "rescue_rate": (
            float(np.logical_and(base_errors, treat_correct).sum() / baseline_error_count)
            if baseline_error_count
            else None
        ),
        "damage_rate": (
            float(np.logical_and(base_correct, treat_errors).sum() / baseline_success_count)
            if baseline_success_count
            else None
        ),
        "pair_oracle_accuracy": pair_oracle,
        "complementarity_headroom": pair_oracle - max(baseline_accuracy, treatment_accuracy),
        "paired_2x2": counts,
    }


def _bootstrap(
    baseline: dict[str, bool], treatment: dict[str, bool], seed: int, n_resamples: int
) -> dict[str, Any]:
    item_ids = list(baseline)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    rescue_damage: list[float] = []
    headrooms: list[float] = []
    for _ in range(n_resamples):
        selected = rng.integers(0, len(item_ids), size=len(item_ids))
        base = np.asarray([baseline[item_ids[index]] for index in selected], dtype=bool)
        treat = np.asarray([treatment[item_ids[index]] for index in selected], dtype=bool)
        base_errors = ~base
        treat_errors = ~treat
        base_error_count = int(base_errors.sum())
        base_success_count = int(base.sum())
        delta = float(treat.mean() - base.mean())
        rescue = (
            float(np.logical_and(base_errors, treat).sum() / base_error_count)
            if base_error_count
            else 0.0
        )
        damage = (
            float(np.logical_and(base, treat_errors).sum() / base_success_count)
            if base_success_count
            else 0.0
        )
        oracle = float(np.logical_or(base, treat).mean())
        deltas.append(delta)
        rescue_damage.append(rescue - damage)
        headrooms.append(oracle - max(float(base.mean()), float(treat.mean())))

    def interval(values: list[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {
        "method": "independent_item_bootstrap_descriptive",
        "confidence": 0.95,
        "n_resamples": n_resamples,
        "seed": seed,
        "delta_accuracy_interval": interval(deltas),
        "rescue_minus_damage_interval": interval(rescue_damage),
        "complementarity_headroom_interval": interval(headrooms),
    }


def _compare_stored(
    stored_rows: list[dict[str, Any]], recomputed_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    stored = {(str(row["item_id"]), str(row["role"])): row for row in stored_rows}
    recomputed = {(str(row["item_id"]), str(row["role"])): row for row in recomputed_rows}
    missing = sorted(set(recomputed) - set(stored))
    extra = sorted(set(stored) - set(recomputed))
    prediction_mismatches: list[dict[str, Any]] = []
    numeric_diffs: list[float] = []
    q_prediction_mismatches: list[dict[str, Any]] = []
    for key in sorted(set(stored) & set(recomputed)):
        left = stored[key]
        right = recomputed[key]
        left_s = np.asarray(left["centered_logit_scores"], dtype=np.float64)
        right_s = np.asarray(right["centered_logit_scores"], dtype=np.float64)
        left_q = np.asarray(left["probability_mean_scores"], dtype=np.float64)
        right_q = np.asarray(right["probability_mean_scores"], dtype=np.float64)
        if left_s.shape != right_s.shape or not np.allclose(
            left_s, right_s, atol=1e-12, rtol=1e-12
        ):
            numeric_diffs.extend(np.abs(left_s - right_s).reshape(-1).tolist())
        if left_q.shape != right_q.shape or not np.allclose(
            left_q, right_q, atol=1e-12, rtol=1e-12
        ):
            numeric_diffs.extend(np.abs(left_q - right_q).reshape(-1).tolist())
        if int(left["predicted_semantic_original_index"]) != int(
            right["predicted_semantic_original_index"]
        ):
            prediction_mismatches.append(
                {
                    "item_id": key[0],
                    "role": key[1],
                    "stored": left["predicted_semantic_original_index"],
                    "recomputed": right["predicted_semantic_original_index"],
                }
            )
        if int(left["probability_mean_prediction"]) != int(right["probability_mean_prediction"]):
            q_prediction_mismatches.append(
                {
                    "item_id": key[0],
                    "role": key[1],
                    "stored": left["probability_mean_prediction"],
                    "recomputed": right["probability_mean_prediction"],
                }
            )
    return {
        "stored_row_count": len(stored_rows),
        "recomputed_row_count": len(recomputed_rows),
        "missing_keys": missing,
        "extra_keys": extra,
        "primary_prediction_mismatch_count": len(prediction_mismatches),
        "primary_prediction_mismatches": prediction_mismatches[:20],
        "secondary_prediction_mismatch_count": len(q_prediction_mismatches),
        "secondary_prediction_mismatches": q_prediction_mismatches[:20],
        "max_abs_numeric_difference": max(numeric_diffs, default=0.0),
        "stored_primary_rows_match": not missing and not extra and not prediction_mismatches,
        "stored_secondary_rows_match": not missing and not extra and not q_prediction_mismatches,
    }


def _s_q_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in PRIMARY_ROLES:
        selected = [row for row in rows if row["role"] == role]
        disagreements = [
            {
                "item_id": row["item_id"],
                "s_prediction": row["predicted_semantic_original_index"],
                "q_prediction": row["probability_mean_prediction"],
                "target": row["target_semantic_original_index"],
                "s_correct": row["predicted_semantic_original_index"]
                == row["target_semantic_original_index"],
                "q_correct": row["probability_mean_prediction"]
                == row["target_semantic_original_index"],
                "s_margin": row["symmetrized_margin"],
                "q_margin": row["probability_mean_margin"],
            }
            for row in selected
            if row["predicted_semantic_original_index"] != row["probability_mean_prediction"]
        ]
        result[role] = {
            "n_items": len(selected),
            "agreement_count": len(selected) - len(disagreements),
            "agreement_rate": (len(selected) - len(disagreements)) / len(selected),
            "disagreement_count": len(disagreements),
            "disagreements": disagreements,
        }
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _flip_robustness(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare S/Q semantic flips, rescues, damages, and transition types."""

    by_role_item = {(row["role"], row["item_id"]): row for row in rows}
    result: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    for role in ("pc1_minus", "pc1_plus"):
        s_flips: set[str] = set()
        q_flips: set[str] = set()
        s_rescues: set[str] = set()
        q_rescues: set[str] = set()
        s_damages: set[str] = set()
        q_damages: set[str] = set()
        s_transitions: Counter[str] = Counter()
        q_transitions: Counter[str] = Counter()
        wrong_to_wrong_s: set[str] = set()
        wrong_to_wrong_q: set[str] = set()
        correct_to_correct_flip_s = 0
        correct_to_correct_flip_q = 0
        for item_id in sorted({row["item_id"] for row in rows}):
            baseline = by_role_item[("baseline", item_id)]
            treatment = by_role_item[(role, item_id)]
            s_base = int(baseline["predicted_semantic_original_index"])
            s_treatment = int(treatment["predicted_semantic_original_index"])
            q_base = int(baseline["probability_mean_prediction"])
            q_treatment = int(treatment["probability_mean_prediction"])
            target = int(baseline["target_semantic_original_index"])
            s_base_correct = s_base == target
            s_treatment_correct = s_treatment == target
            q_base_correct = q_base == target
            q_treatment_correct = q_treatment == target
            s_transition = _transition(s_base_correct, s_treatment_correct)
            q_transition = _transition(q_base_correct, q_treatment_correct)
            s_transitions[s_transition] += 1
            q_transitions[q_transition] += 1
            if s_base != s_treatment:
                s_flips.add(item_id)
                if s_base_correct and s_treatment_correct:
                    correct_to_correct_flip_s += 1
                if not s_base_correct and not s_treatment_correct:
                    wrong_to_wrong_s.add(item_id)
            if q_base != q_treatment:
                q_flips.add(item_id)
                if q_base_correct and q_treatment_correct:
                    correct_to_correct_flip_q += 1
                if not q_base_correct and not q_treatment_correct:
                    wrong_to_wrong_q.add(item_id)
            if s_transition == "rescue":
                s_rescues.add(item_id)
            if q_transition == "rescue":
                q_rescues.add(item_id)
            if s_transition == "damage":
                s_damages.add(item_id)
            if q_transition == "damage":
                q_damages.add(item_id)
            if role == "pc1_plus":
                audit_rows.append(
                    {
                        "item_id": item_id,
                        "target_semantic_id": target,
                        "baseline_S": s_base,
                        "pc1plus_S": s_treatment,
                        "S_status": s_transition,
                        "baseline_Q": q_base,
                        "pc1plus_Q": q_treatment,
                        "Q_status": q_transition,
                        "same_baseline_prediction": s_base == q_base,
                        "same_treatment_prediction": s_treatment == q_treatment,
                        "same_flip": (s_base != s_treatment) == (q_base != q_treatment),
                        "same_correctness_transition": s_transition == q_transition,
                    }
                )
        result[role] = {
            "semantic_flip_sets": {
                "S_count": len(s_flips),
                "Q_count": len(q_flips),
                "intersection_count": len(s_flips & q_flips),
                "union_count": len(s_flips | q_flips),
                "jaccard": _jaccard(s_flips, q_flips),
            },
            "rescue_sets": {
                "S_count": len(s_rescues),
                "Q_count": len(q_rescues),
                "intersection_count": len(s_rescues & q_rescues),
                "union_count": len(s_rescues | q_rescues),
                "jaccard": _jaccard(s_rescues, q_rescues),
            },
            "damage_sets": {
                "S_count": len(s_damages),
                "Q_count": len(q_damages),
                "intersection_count": len(s_damages & q_damages),
                "union_count": len(s_damages | q_damages),
                "jaccard": _jaccard(s_damages, q_damages),
            },
            "primary_transition_counts": dict(s_transitions),
            "secondary_transition_counts": dict(q_transitions),
            "wrong_to_wrong_flip_overlap": {
                "S_count": len(wrong_to_wrong_s),
                "Q_count": len(wrong_to_wrong_q),
                "intersection_count": len(wrong_to_wrong_s & wrong_to_wrong_q),
                "union_count": len(wrong_to_wrong_s | wrong_to_wrong_q),
                "jaccard": _jaccard(wrong_to_wrong_s, wrong_to_wrong_q),
            },
            "correct_to_correct_flip_count_S": correct_to_correct_flip_s,
            "correct_to_correct_flip_count_Q": correct_to_correct_flip_q,
        }
    return result, audit_rows


def _margin_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in PRIMARY_ROLES:
        selected = [row for row in rows if row["role"] == role]
        groups: dict[str, list[dict[str, Any]]] = {"agreement": [], "disagreement": []}
        for row in selected:
            group = (
                "agreement"
                if row["predicted_semantic_original_index"] == row["probability_mean_prediction"]
                else "disagreement"
            )
            groups[group].append(row)
        result[role] = {
            group: {
                "n": len(group_rows),
                "S_margin": _summary([float(row["symmetrized_margin"]) for row in group_rows]),
                "Q_margin": _summary([float(row["probability_mean_margin"]) for row in group_rows]),
            }
            for group, group_rows in groups.items()
        }
    return result


def _instability_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_role = {
        role: {
            row["item_id"]
            for row in rows
            if row["role"] == role
            and row["predicted_semantic_original_index"] != row["probability_mean_prediction"]
        }
        for role in PRIMARY_ROLES
    }
    result: dict[str, Any] = {}
    item_count = len({row["item_id"] for row in rows})
    for treatment in ("pc1_minus", "pc1_plus"):
        baseline = by_role["baseline"]
        treated = by_role[treatment]
        result[treatment] = {
            "baseline_only_count": len(baseline - treated),
            "treatment_only_count": len(treated - baseline),
            "both_count": len(baseline & treated),
            "neither_count": item_count - len(baseline | treated),
            "baseline_disagreement_rate": len(baseline) / item_count,
            "treatment_disagreement_rate": len(treated) / item_count,
        }
    return result


def _score_geometry_diagnostic(
    raw_rows: list[dict[str, Any]], sym_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_key = {(row["item_id"], row["role"]): row for row in sym_rows}
    raw_by_item_role: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in raw_rows:
        values = np.asarray(
            [float(row["candidate_scores"][label]) for label in row["candidate_labels"]],
            dtype=np.float64,
        )
        raw_by_item_role[(row["item_id"], row["role"])].append(float(values.std()))
    result: dict[str, Any] = {}
    for role in PRIMARY_ROLES:
        groups: dict[str, list[float]] = {"S_Q_agreement": [], "S_Q_disagreement": []}
        spread_ranges: dict[str, list[float]] = {"S_Q_agreement": [], "S_Q_disagreement": []}
        spread_cvs: dict[str, list[float]] = {"S_Q_agreement": [], "S_Q_disagreement": []}
        for (item_id, row_role), spreads in raw_by_item_role.items():
            if row_role != role:
                continue
            aggregate = by_key[(item_id, role)]
            group = (
                "S_Q_agreement"
                if aggregate["predicted_semantic_original_index"]
                == aggregate["probability_mean_prediction"]
                else "S_Q_disagreement"
            )
            mean_spread = float(np.mean(spreads))
            groups[group].append(mean_spread)
            spread_ranges[group].append(float(np.max(spreads) - np.min(spreads)))
            spread_cvs[group].append(
                float(np.std(spreads) / mean_spread) if mean_spread else float("nan")
            )
        result[role] = {
            group: {
                "candidate_logit_sd_across_orders": _summary(values),
                "candidate_logit_sd_range_across_orders": _summary(spread_ranges[group]),
                "candidate_logit_sd_cv_across_orders": _summary(
                    [value for value in spread_cvs[group] if math.isfinite(value)]
                ),
            }
            for group, values in groups.items()
        }
    result["interpretation"] = (
        "This is a descriptive scale diagnostic only. It does not define or apply "
        "a third estimator."
    )
    return result


def _robustness_classification(
    metrics: dict[str, Any], flip_robustness: dict[str, Any]
) -> dict[str, Any]:
    """Apply the pre-specified descriptive ROBUST/SENSITIVE/NON-ROBUST rule."""

    plus_s = metrics["S"]["pc1_plus"]
    plus_q = metrics["Q"]["pc1_plus"]
    s_rescue_damage = (
        plus_s["paired_2x2"]["baseline_wrong__treatment_correct"]
        - plus_s["paired_2x2"]["baseline_correct__treatment_wrong"]
    )
    q_rescue_damage = (
        plus_q["paired_2x2"]["baseline_wrong__treatment_correct"]
        - plus_q["paired_2x2"]["baseline_correct__treatment_wrong"]
    )
    flip_jaccard = flip_robustness["pc1_plus"]["semantic_flip_sets"]["jaccard"]
    sign_reversals = {
        "delta_accuracy": np.sign(plus_s["delta_accuracy"]) != np.sign(plus_q["delta_accuracy"]),
        "rescue_minus_damage": np.sign(s_rescue_damage) != np.sign(q_rescue_damage),
    }
    if any(sign_reversals.values()):
        classification = "NON-ROBUST"
    elif flip_jaccard >= 0.5:
        classification = "ROBUST"
    else:
        classification = "ESTIMATOR-SENSITIVE"
    return {
        "classification": classification,
        "rule": {
            "sign_reversal_is_non_robust": True,
            "substantial_flip_overlap_threshold": 0.5,
            "almost_no_overlap_threshold": 0.25,
        },
        "pc1_plus": {
            "S_delta": plus_s["delta_accuracy"],
            "Q_delta": plus_q["delta_accuracy"],
            "S_rescue_minus_damage_count": s_rescue_damage,
            "Q_rescue_minus_damage_count": q_rescue_damage,
            "flip_jaccard": flip_jaccard,
            "sign_reversals": sign_reversals,
        },
        "interpretation": "Descriptive aggregator-sensitivity classification only; no scientific claim is frozen.",
    }


def _transition(base: bool, treatment: bool) -> str:
    if base and treatment:
        return "correct_to_correct"
    if base and not treatment:
        return "damage"
    if not base and treatment:
        return "rescue"
    return "wrong_to_wrong"


def _overlap(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    by_role_item = {(row["role"], row["item_id"]): row for row in rows}
    for role in ("pc1_minus", "pc1_plus"):
        transitions_s: list[str] = []
        transitions_q: list[str] = []
        matrix: Counter[tuple[str, str]] = Counter()
        prediction_flip_matrix: Counter[tuple[bool, bool]] = Counter()
        for item_id in sorted({row["item_id"] for row in rows}):
            base = by_role_item[("baseline", item_id)]
            treatment = by_role_item[(role, item_id)]
            s_transition = _transition(bool(base["correct"]), bool(treatment["correct"]))
            q_transition = _transition(
                int(base["probability_mean_prediction"])
                == int(base["target_semantic_original_index"]),
                int(treatment["probability_mean_prediction"])
                == int(treatment["target_semantic_original_index"]),
            )
            transitions_s.append(s_transition)
            transitions_q.append(q_transition)
            matrix[(s_transition, q_transition)] += 1
            prediction_flip_matrix[
                (
                    int(base["predicted_semantic_original_index"])
                    != int(treatment["predicted_semantic_original_index"]),
                    int(base["probability_mean_prediction"])
                    != int(treatment["probability_mean_prediction"]),
                )
            ] += 1
        result[role] = {
            "primary_transition_counts": dict(Counter(transitions_s)),
            "secondary_transition_counts": dict(Counter(transitions_q)),
            "transition_overlap_matrix": {
                f"S:{left}|Q:{right}": count for (left, right), count in sorted(matrix.items())
            },
            "transition_exact_overlap_count": sum(
                count for (left, right), count in matrix.items() if left == right
            ),
            "transition_disagreement_count": sum(
                count for (left, right), count in matrix.items() if left != right
            ),
            "prediction_flip_overlap": {
                f"S_flip_{left}|Q_flip_{right}": count
                for (left, right), count in sorted(prediction_flip_matrix.items())
            },
        }
    return result


def _margin_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in PRIMARY_ROLES:
        selected = [row for row in rows if row["role"] == role]
        result[role] = {
            "S_margin": _summary([float(row["symmetrized_margin"]) for row in selected]),
            "Q_margin": _summary([float(row["probability_mean_margin"]) for row in selected]),
        }
    return result


def _instability_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in PRIMARY_ROLES:
        selected = [row for row in rows if row["role"] == role]
        result[role] = {
            "s_score_sd_mean": _summary([float(row["s_score_sd_mean"]) for row in selected]),
            "s_score_sd_max": _summary([float(row["s_score_sd_max"]) for row in selected]),
            "q_score_sd_mean": _summary([float(row["q_score_sd_mean"]) for row in selected]),
            "q_score_sd_max": _summary([float(row["q_score_sd_max"]) for row in selected]),
            "s_prediction_mode_share": _summary(
                [float(row["s_prediction_mode_share"]) for row in selected]
            ),
            "q_prediction_mode_share": _summary(
                [float(row["q_prediction_mode_share"]) for row in selected]
            ),
            "s_unique_shift_predictions": Counter(
                int(row["s_unique_shift_predictions"]) for row in selected
            ),
            "q_unique_shift_predictions": Counter(
                int(row["q_unique_shift_predictions"]) for row in selected
            ),
        }
    return result


def _scale_diagnostics(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate_score_semantics": "candidate_logits_no_vocab_normalization"
    }
    all_centered_means: list[float] = []
    for role in ROLES:
        selected = [row for row in raw_rows if row["role"] == role]
        raw_means: list[float] = []
        raw_stds: list[float] = []
        raw_ranges: list[float] = []
        centered_means: list[float] = []
        winner_mismatches = 0
        for row in selected:
            values = np.asarray(
                [float(row["candidate_scores"][label]) for label in row["candidate_labels"]]
            )
            centered = values - float(values.mean())
            raw_means.append(float(values.mean()))
            raw_stds.append(float(values.std()))
            raw_ranges.append(float(values.max() - values.min()))
            centered_means.append(float(centered.mean()))
            all_centered_means.append(float(centered.mean()))
            if _argmax(values) != _argmax(centered):
                winner_mismatches += 1
        result[role] = {
            "raw_score_mean": _summary(raw_means),
            "raw_score_sd": _summary(raw_stds),
            "raw_score_range": _summary(raw_ranges),
            "centered_row_mean": _summary(centered_means),
            "centered_row_mean_abs_max": max((abs(value) for value in centered_means), default=0.0),
            "raw_vs_centered_winner_mismatch_count": winner_mismatches,
        }
    result["global_centered_row_mean_abs_max"] = max(
        (abs(value) for value in all_centered_means), default=0.0
    )
    result["interpretation"] = (
        "Candidate scores are logits for the allowed candidates only; they are not "
        "full-vocabulary normalized log probabilities. Centering removes a common "
        "per-row offset and cannot change the within-row winner."
    )
    return result


def _robust_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in PRIMARY_ROLES:
        selected = [row for row in rows if row["role"] == role]
        status_counts: Counter[str] = Counter()
        for row in selected:
            s_correct = bool(
                row["predicted_semantic_original_index"] == row["target_semantic_original_index"]
            )
            q_correct = bool(
                row["probability_mean_prediction"] == row["target_semantic_original_index"]
            )
            if s_correct and q_correct:
                status = "both_correct"
            elif not s_correct and not q_correct:
                status = "both_wrong"
            elif s_correct:
                status = "S_only_correct"
            else:
                status = "Q_only_correct"
            status_counts[status] += 1
        result[role] = {
            "classification_definition": "agreement between primary centered-logit S and secondary probability-mean Q correctness",
            "counts": dict(status_counts),
            "robust_count": int(status_counts["both_correct"] + status_counts["both_wrong"]),
            "robust_fraction": float(
                (status_counts["both_correct"] + status_counts["both_wrong"]) / len(selected)
            )
            if selected
            else None,
        }
    return result


def _split_provenance(
    raw_index: dict[str, Any],
    raw_path: Path,
    sym_path: Path,
    source_manifest: dict[str, Any],
    repo_root: Path,
    split_manifest_path: Path | None,
) -> dict[str, Any]:
    benchmark = source_manifest.get("benchmark_provenance", {})
    model = source_manifest.get("model_provenance", {})
    recorded_ids_hash = benchmark.get("item_ids_hash")
    calculated_ids_hash = _stable_digest(*raw_index["item_order"])
    try:
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        current_commit = None
    split_details: dict[str, Any] = {
        "available_locally": False,
        "logical_manifest_digest": None,
        "split_file_sha256": None,
        "dev_evaluation_ids_digest": None,
        "dev_evaluation_exact_id_match": None,
        "interpretation": (
            "The logical manifest digest, split-file byte SHA-256, and DEV_EVALUATION "
            "ID digest are distinct objects."
        ),
    }
    if split_manifest_path and split_manifest_path.is_file():
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        dev_ids = split_manifest.get("splits", {}).get("dev_evaluation", [])
        split_details.update(
            {
                "available_locally": True,
                "path": str(split_manifest_path),
                "logical_manifest_digest": split_manifest.get("manifest_sha256"),
                "split_file_sha256": _sha256(split_manifest_path),
                "dev_evaluation_ids_digest": _stable_digest(*dev_ids),
                "dev_evaluation_item_count": len(dev_ids),
                "dev_evaluation_exact_id_match": set(dev_ids) == set(raw_index["item_order"]),
                "dev_evaluation_ordered_id_match": list(dev_ids) == list(raw_index["item_order"]),
            }
        )
    checks = {
        "protocol": source_manifest.get("protocol") == PROTOCOL,
        "stage_development": source_manifest.get("protocol_stage") == "DEVELOPMENT",
        "dataset_revision": benchmark.get("dataset_revision") == EXPECTED_DATASET_REVISION,
        "model_revision": model.get("model_revision") == EXPECTED_MODEL_REVISION,
        "layer": source_manifest.get("layer") == 17,
        "token_scope": source_manifest.get("token_scope") == "last_token",
        "vector_hash": source_manifest.get("pc1_hash") == EXPECTED_PC1_HASH,
        "item_count": raw_index["item_count"] == 512,
        "item_ids_hash": recorded_ids_hash == calculated_ids_hash,
        "raw_hash": source_manifest.get("raw_scores_sha256") == _sha256(raw_path),
        "sym_hash": source_manifest.get("symmetrized_scores_sha256") == _sha256(sym_path),
        "holdout_access_forbidden": source_manifest.get("holdout_access") == "forbidden",
        "confirmatory_not_accessed": source_manifest.get("confirmatory_accessed") == "NO",
        "requested_split_dev_evaluation": benchmark.get("requested_split") == "dev_evaluation",
    }
    return {
        "source_manifest_protocol": source_manifest.get("protocol"),
        "source_run_status": source_manifest.get("status"),
        "source_run_timestamp_utc": source_manifest.get("timestamp_utc"),
        "source_run_git_commit": source_manifest.get("git", {}).get("git_commit"),
        "audit_code_git_commit": current_commit,
        "dataset_id": benchmark.get("dataset_id"),
        "dataset_revision": benchmark.get("dataset_revision"),
        "requested_split": benchmark.get("requested_split"),
        "item_count": raw_index["item_count"],
        "recorded_item_ids_hash": recorded_ids_hash,
        "calculated_item_ids_hash": calculated_ids_hash,
        "model_identifier": model.get("model_identifier"),
        "model_revision": model.get("model_revision"),
        "tokenizer_revision": model.get("tokenizer_revision"),
        "layer": source_manifest.get("layer"),
        "token_scope": source_manifest.get("token_scope"),
        "pc1_hash": source_manifest.get("pc1_hash"),
        "alpha_minus": source_manifest.get("alpha_minus"),
        "alpha_plus": source_manifest.get("alpha_plus"),
        "holdout_firewall": {
            "access_declared_forbidden": source_manifest.get("holdout_access") == "forbidden",
            "confirmatory_accessed_declared_no": source_manifest.get("confirmatory_accessed")
            == "NO",
            "audit_loaded_dataset_or_holdout": False,
            "status": "PASS_DECLARATION_ONLY",
        },
        "split_hashes": split_details,
        "checks": checks,
        "all_static_checks_pass": all(checks.values()),
    }


def _summary_markdown(
    source_manifest: dict[str, Any],
    provenance: dict[str, Any],
    metrics: dict[str, Any],
    agreement: dict[str, Any],
    overlap: dict[str, Any],
    comparison: dict[str, Any],
    flip_robustness: dict[str, Any],
    margin_sensitivity: dict[str, Any],
    robustness: dict[str, Any],
) -> str:
    lines = [
        "# Q1 V1.2 Aggregation Audit — Complete Review Bundle",
        "",
        "> ANALYSIS-ONLY AUDIT. No model, dataset, or new inference was run.",
        "> This document does not freeze or support a scientific claim.",
        "",
        "## Scope",
        "",
        "The audit independently recomputes the primary centered-logit aggregation (S) and the secondary probability-mean aggregation (Q) from the frozen raw cyclic candidate scores.",
        "The primary protocol remains Q1 V1.2 DEVELOPMENT on the 512-item DEV_EVALUATION split.",
        "",
        "## Provenance",
        "",
        f"- Source run status: `{source_manifest.get('status')}`",
        f"- Source model: `{provenance.get('model_identifier')}` revision `{provenance.get('model_revision')}`",
        f"- Dataset revision: `{provenance.get('dataset_revision')}`",
        f"- Split: `{provenance.get('requested_split')}`; items: `{provenance.get('item_count')}`",
        f"- Layer/scope: `{provenance.get('layer')}` / `{provenance.get('token_scope')}`",
        f"- Holdout firewall: `{provenance['holdout_firewall']['status']}`",
        f"- Static provenance checks: `{provenance.get('all_static_checks_pass')}`",
        "",
        "## Stored primary recomputation gate",
        "",
        f"- Stored S-row prediction mismatches: `{comparison['primary_prediction_mismatch_count']}`",
        f"- Stored Q-row prediction mismatches: `{comparison['secondary_prediction_mismatch_count']}`",
        f"- Maximum stored/recomputed numeric difference: `{comparison['max_abs_numeric_difference']:.3g}`",
        "",
        "The required primary discrete prediction mismatch gate is zero. Numeric tolerance is used only for floating-point serialization.",
        "",
        "## Paired metrics — primary S",
        "",
        "| condition | baseline accuracy | treatment accuracy | delta | phi | Jaccard | rescue | damage | oracle gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("pc1_minus", "pc1_plus"):
        row = metrics["S"][role]
        lines.append(
            f"| {role} | {row['baseline_accuracy']:.4f} | {row['treatment_accuracy']:.4f} | {row['delta_accuracy']:.4f} | {row['error_correlation_phi'] if row['error_correlation_phi'] is not None else 'null'} | {row['error_jaccard']:.4f} | {row['rescue_rate'] if row['rescue_rate'] is not None else 'null'} | {row['damage_rate'] if row['damage_rate'] is not None else 'null'} | {row['complementarity_headroom']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired metrics — secondary Q",
            "",
            "Q is a diagnostic aggregation only. It is not substituted for the frozen primary S result.",
            "",
            "| condition | baseline accuracy | treatment accuracy | delta | phi | Jaccard | rescue | damage | oracle gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for role in ("pc1_minus", "pc1_plus"):
        row = metrics["Q"][role]
        lines.append(
            f"| {role} | {row['baseline_accuracy']:.4f} | {row['treatment_accuracy']:.4f} | {row['delta_accuracy']:.4f} | {row['error_correlation_phi'] if row['error_correlation_phi'] is not None else 'null'} | {row['error_jaccard']:.4f} | {row['rescue_rate'] if row['rescue_rate'] is not None else 'null'} | {row['damage_rate'] if row['damage_rate'] is not None else 'null'} | {row['complementarity_headroom']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## S versus Q",
            "",
        ]
    )
    for role in PRIMARY_ROLES:
        row = agreement[role]
        lines.append(
            f"- `{role}`: agreement `{row['agreement_count']}/{row['n_items']}` ({row['agreement_rate']:.4f}); disagreements `{row['disagreement_count']}`."
        )
    lines.extend(
        [
            "",
            "## Rescue/damage overlap",
            "",
        ]
    )
    for role in ("pc1_minus", "pc1_plus"):
        row = overlap[role]
        lines.append(
            f"- `{role}`: exact S/Q transition overlap `{row['transition_exact_overlap_count']}`; transition disagreements `{row['transition_disagreement_count']}`."
        )
    plus_flips = flip_robustness["pc1_plus"]["semantic_flip_sets"]
    plus_rescues = flip_robustness["pc1_plus"]["rescue_sets"]
    plus_damages = flip_robustness["pc1_plus"]["damage_sets"]
    lines.extend(
        [
            "",
            "## PC1+ flip-set audit",
            "",
            f"- Semantic flips S/Q: `{plus_flips['S_count']}` / `{plus_flips['Q_count']}`; intersection `{plus_flips['intersection_count']}`; Jaccard `{plus_flips['jaccard']:.4f}`.",
            f"- Rescue overlap S/Q: `{plus_rescues['intersection_count']}`; Jaccard `{plus_rescues['jaccard']:.4f}`.",
            f"- Damage overlap S/Q: `{plus_damages['intersection_count']}`; Jaccard `{plus_damages['jaccard']:.4f}`.",
            f"- Aggregator-sensitivity classification: `{robustness['classification']}`.",
            "",
            "## Margin and scale diagnostics",
            "",
            f"- Baseline S-margin median, agreement/disagreement: `{margin_sensitivity['baseline']['agreement']['S_margin']['median']}` / `{margin_sensitivity['baseline']['disagreement']['S_margin']['median']}`.",
            f"- Baseline Q-margin median, agreement/disagreement: `{margin_sensitivity['baseline']['agreement']['Q_margin']['median']}` / `{margin_sensitivity['baseline']['disagreement']['Q_margin']['median']}`.",
            "- Candidate-logit spread diagnostics are in `score_geometry_diagnostic.json`; no third estimator was created.",
            "",
            "## Split hash objects",
            "",
            f"- Logical split manifest digest: `{provenance['split_hashes']['logical_manifest_digest']}`.",
            f"- Split file byte SHA-256: `{provenance['split_hashes']['split_file_sha256']}`.",
            f"- DEV_EVALUATION ID digest: `{provenance['split_hashes']['dev_evaluation_ids_digest']}`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- These are descriptive recomputations of an existing DEVELOPMENT artifact.",
            "- Low error similarity, if observed, cannot be interpreted without individual accuracy, rescues, damages, and the paired 2×2 table.",
            "- Q is a sensitivity diagnostic for aggregation choice, not a second scientific result.",
            "- No holdout data were loaded by this audit; the holdout status is a provenance declaration copied from the frozen run manifest.",
            "- Q2 geometry and any confirmatory campaign remain untouched.",
            "",
            "## Files",
            "",
            "See `q1_v1_v2_audit_metrics.json`, `s_q_agreement.json`, `flip_rescue_damage_overlap.json`, `margin_analysis.json`, `instability_analysis.json`, `scale_diagnostics.json`, `robust_classification.json`, and `split_provenance_audit.json` for machine-readable details.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(
    raw_path: Path,
    stored_sym_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    repo_root: Path,
    source_review_dir: Path | None = None,
    split_manifest_path: Path | None = None,
    bootstrap_resamples: int = 200,
    force: bool = False,
) -> dict[str, Any]:
    """Run the complete local V1.2 audit and write a self-contained bundle."""

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"Audit output is non-empty: {output_dir}; use --force explicitly")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _read_jsonl(raw_path)
    stored_sym_rows = _read_jsonl(stored_sym_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    index = validate_raw_rows(raw_rows)
    if len(stored_sym_rows) != index["item_count"] * len(PRIMARY_ROLES):
        raise ValueError("Stored symmetrized row count does not equal item_count × 3")
    sym_rows, instability_rows = recompute_aggregates(raw_rows, index)
    comparison = _compare_stored(stored_sym_rows, sym_rows)
    if comparison["primary_prediction_mismatch_count"]:
        raise ValueError("Stored primary S predictions differ from independent recomputation")

    by_role_item = {(row["role"], row["item_id"]): row for row in sym_rows}
    metrics: dict[str, Any] = {"S": {}, "Q": {}}
    for aggregator in ("S", "Q"):
        for role in ("pc1_minus", "pc1_plus"):
            baseline = {
                item_id: (
                    row["predicted_semantic_original_index"]
                    == row["target_semantic_original_index"]
                    if aggregator == "S"
                    else row["probability_mean_prediction"] == row["target_semantic_original_index"]
                )
                for (row_role, item_id), row in by_role_item.items()
                if row_role == "baseline"
            }
            treatment = {
                item_id: (
                    row["predicted_semantic_original_index"]
                    == row["target_semantic_original_index"]
                    if aggregator == "S"
                    else row["probability_mean_prediction"] == row["target_semantic_original_index"]
                )
                for (row_role, item_id), row in by_role_item.items()
                if row_role == role
            }
            metric = paired_metrics(baseline, treatment, f"{role}_{aggregator}")
            metric["bootstrap"] = _bootstrap(
                baseline,
                treatment,
                seed=int(source_manifest.get("experiment_seed", 0))
                + (0 if aggregator == "S" else 1),
                n_resamples=bootstrap_resamples,
            )
            metrics[aggregator][role] = metric

    agreement = _s_q_agreement(sym_rows)
    overlap = _overlap(sym_rows)
    flip_robustness, flip_audit_rows = _flip_robustness(sym_rows)
    margins = _margin_analysis(sym_rows)
    margin_sensitivity = _margin_sensitivity(sym_rows)
    instability = _instability_analysis(instability_rows)
    instability_overlap = _instability_overlap(sym_rows)
    scale = _scale_diagnostics(raw_rows)
    score_geometry = _score_geometry_diagnostic(raw_rows, sym_rows)
    robust = _robust_classification(sym_rows)
    robustness = _robustness_classification(metrics, flip_robustness)
    provenance = _split_provenance(
        raw_index=index,
        raw_path=raw_path,
        sym_path=stored_sym_path,
        source_manifest=source_manifest,
        repo_root=repo_root,
        split_manifest_path=split_manifest_path,
    )

    for source, name in (
        (raw_path, "raw_permutation_scores.jsonl"),
        (stored_sym_path, "symmetrized_scores.jsonl"),
    ):
        shutil.copy2(source, output_dir / name)
    if source_review_dir and source_review_dir.exists():
        for name in ("manifest.json", "config_resolved.yaml"):
            source = source_review_dir / name
            if source.exists():
                shutil.copy2(source, output_dir / name)
    _write_jsonl(output_dir / "recomputed_symmetrized_scores.jsonl", sym_rows)
    _write_jsonl(output_dir / "instability_by_item.jsonl", instability_rows)
    _write_json(output_dir / "q1_v1_v2_audit_metrics.json", metrics)
    _write_json(output_dir / "s_q_agreement.json", agreement)
    _write_json(output_dir / "flip_rescue_damage_overlap.json", overlap)
    _write_json(output_dir / "flip_set_robustness.json", flip_robustness)
    _write_json(output_dir / "margin_analysis.json", margins)
    _write_json(output_dir / "margin_sensitivity.json", margin_sensitivity)
    _write_json(output_dir / "instability_analysis.json", instability)
    _write_json(output_dir / "instability_overlap.json", instability_overlap)
    _write_json(output_dir / "scale_diagnostics.json", scale)
    _write_json(output_dir / "score_geometry_diagnostic.json", score_geometry)
    _write_json(output_dir / "robust_classification.json", robust)
    _write_json(output_dir / "robustness_classification.json", robustness)
    _write_json(output_dir / "split_provenance_audit.json", provenance)
    _write_json(output_dir / "stored_sym_comparison.json", comparison)
    with (output_dir / "primary_pc1_plus_flip_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = list(flip_audit_rows[0]) if flip_audit_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flip_audit_rows)
    audit_manifest = {
        "audit_type": "Q1_V1_2_ANALYSIS_ONLY_AGGREGATION_AUDIT",
        "audit_status": "COMPLETE",
        "scientific_result": None,
        "raw_source_sha256": _sha256(raw_path),
        "stored_sym_source_sha256": _sha256(stored_sym_path),
        "raw_row_count": len(raw_rows),
        "stored_sym_row_count": len(stored_sym_rows),
        "recomputed_sym_row_count": len(sym_rows),
        "primary_prediction_mismatch_count": comparison["primary_prediction_mismatch_count"],
        "secondary_prediction_mismatch_count": comparison["secondary_prediction_mismatch_count"],
        "source_manifest": str(source_manifest_path),
        "source_run_git_commit": source_manifest.get("git", {}).get("git_commit"),
        "audit_code_git_commit": provenance.get("audit_code_git_commit"),
        "no_model_loaded": True,
        "no_dataset_loaded": True,
        "holdout_loaded": False,
        "robustness_classification": robustness["classification"],
    }
    _write_json(output_dir / "audit_manifest.json", audit_manifest)
    (output_dir / "summary.md").write_text(
        _summary_markdown(
            source_manifest,
            provenance,
            metrics,
            agreement,
            overlap,
            comparison,
            flip_robustness,
            margin_sensitivity,
            robustness,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "audit_manifest": audit_manifest,
        "comparison": comparison,
        "provenance": provenance,
        "metrics": metrics,
    }


def validate_audit_dir(path: Path) -> dict[str, Any]:
    """Validate hashes, row counts, and the primary recomputation gate."""

    required = [
        "audit_manifest.json",
        "raw_permutation_scores.jsonl",
        "symmetrized_scores.jsonl",
        "recomputed_symmetrized_scores.jsonl",
        "stored_sym_comparison.json",
        "split_provenance_audit.json",
        "q1_v1_v2_audit_metrics.json",
        "flip_set_robustness.json",
        "primary_pc1_plus_flip_audit.csv",
        "margin_sensitivity.json",
        "instability_overlap.json",
        "score_geometry_diagnostic.json",
        "robustness_classification.json",
        "summary.md",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Audit bundle missing files: {missing}")
    manifest = json.loads((path / "audit_manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((path / "stored_sym_comparison.json").read_text(encoding="utf-8"))
    raw_path = path / "raw_permutation_scores.jsonl"
    sym_path = path / "symmetrized_scores.jsonl"
    recomputed_path = path / "recomputed_symmetrized_scores.jsonl"
    checks = {
        "raw_hash": manifest["raw_source_sha256"] == _sha256(raw_path),
        "sym_hash": manifest["stored_sym_source_sha256"] == _sha256(sym_path),
        "raw_rows": manifest["raw_row_count"] == len(_read_jsonl(raw_path)),
        "sym_rows": manifest["stored_sym_row_count"] == len(_read_jsonl(sym_path)),
        "recomputed_rows": manifest["recomputed_sym_row_count"]
        == len(_read_jsonl(recomputed_path)),
        "primary_prediction_mismatch_zero": comparison["primary_prediction_mismatch_count"] == 0,
        "audit_status_complete": manifest.get("audit_status") == "COMPLETE",
        "scientific_result_none": manifest.get("scientific_result") is None,
    }
    if not all(checks.values()):
        raise ValueError(f"V1.2 audit validation failed: {checks}")
    return {"status": "PASS", "checks": checks, "path": str(path)}
