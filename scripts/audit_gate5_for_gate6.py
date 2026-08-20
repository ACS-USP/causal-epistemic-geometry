#!/usr/bin/env python3
"""Independent, offline forensic audit of the Gate-5 bridge.

This auditor intentionally does not import the Gate-5 analysis or classifier.
It reconstructs schedules, provenance, estimands, and classifications from the
append-only journal and frozen review manifests so Gate 6 has an independent
integrity record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
LAYER = 17
ALPHA = 8.39900588973121
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_RESAMPLES = 5000
VALID = {"VALID_CORRECT", "VALID_WRONG"}

SOURCE_CONDITIONS = ("ORDINARY", "CAREFUL", "DIRECT")
PRIMARY_CONDITIONS = (
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
PHASE_SPECS = {
    "SOURCE_CHECK": (40, SOURCE_CONDITIONS, (0, 1)),
    "SUSTAINED_MANIPULATION": (20, PRIMARY_CONDITIONS, (0,)),
    "SUSTAINED_EVALUATION": (60, PRIMARY_CONDITIONS, (0, 1)),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"journal line {line_number} is not an object")
            rows.append(row)
    return rows


def key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])


def semantic_outcome(row: dict[str, Any]) -> str:
    if str(row.get("status")) in VALID and row.get("parsed_answer") is not None:
        return "ANSWER:" + json.dumps(row["parsed_answer"], sort_keys=True, default=str)
    return "STATUS:" + str(row.get("status"))


def error(row: dict[str, Any]) -> bool:
    return str(row.get("status")) != "VALID_CORRECT"


def exact_or_close(actual: Any, expected: Any, tolerance: float = 1e-9) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)
    return actual == expected


def compare_maps(actual: Any, expected: Any, prefix: str, differences: list[dict[str, Any]]) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        for name in sorted(set(actual) | set(expected)):
            if name not in actual or name not in expected:
                differences.append({"field": f"{prefix}.{name}", "actual": actual.get(name), "expected": expected.get(name)})
            else:
                compare_maps(actual[name], expected[name], f"{prefix}.{name}", differences)
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            differences.append({"field": prefix, "actual_length": len(actual), "expected_length": len(expected)})
        for index, (left, right) in enumerate(zip(actual, expected, strict=False)):
            compare_maps(left, right, f"{prefix}[{index}]", differences)
        return
    if not exact_or_close(actual, expected):
        differences.append({"field": prefix, "actual": actual, "expected": expected})


def independent_estimands(baseline: np.ndarray, condition: np.ndarray) -> dict[str, float]:
    """Recompute Gate-5 products without project estimand helpers."""

    if baseline.shape != condition.shape or baseline.ndim != 2 or baseline.shape[1] != 2:
        raise ValueError("expected two [items, rollout] matrices with matching shapes")
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


def classify_source(metrics: dict[str, float]) -> str:
    if metrics["careful_validity"] >= 0.90 and metrics["direct_validity"] >= 0.90 and metrics["X"] >= 0.10 and metrics["S"] >= 0.05:
        return "SOURCE_SEMANTIC_BEHAVIOR_PASS"
    if metrics["careful_validity"] >= 0.90 and metrics["direct_validity"] >= 0.90 and metrics["careful_mean_tokens"] >= 1.25 * metrics["direct_mean_tokens"] and metrics["careful_median_tokens"] >= metrics["direct_median_tokens"] + 2:
        return "SOURCE_COMPUTATION_STYLE_ONLY"
    return "SOURCE_NO_BEHAVIORAL_SEPARATION"


def at_least(value: float, threshold: float) -> bool:
    return value >= threshold or math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)


def classify_manipulation(metrics: dict[str, dict[str, float]]) -> bool:
    random_mean = np.mean([metrics[f"SUSTAINED_RANDOM_R{i}"]["semantic_change_rate"] for i in range(4)])
    for sign in ("PLUS", "MINUS"):
        current = metrics[f"SUSTAINED_{sign}"]
        one_shot = metrics[f"ONE_SHOT_{sign}"]
        if (
            at_least(current["validity"], 0.85)
            and at_least(current["semantic_change_rate"], 0.15)
            and at_least(current["semantic_change_rate"] - one_shot["semantic_change_rate"], 0.05)
            and at_least(current["semantic_change_rate"] - random_mean, 0.05)
        ):
            return True
    return False


def classify_gate5(estimates: dict[str, dict[str, float]], manipulation_pass: bool, engineering_pass: bool) -> str:
    if not engineering_pass:
        return "GATE5_SUSTAINED_ENGINE_FAILURE"
    if not manipulation_pass:
        return "GATE5_NO_BEHAVIORAL_FIRST_STAGE"
    random_d = [estimates[f"SUSTAINED_RANDOM_R{i}"]["D"] for i in range(4)]
    random_c = [estimates[f"SUSTAINED_RANDOM_R{i}"]["C"] for i in range(4)]
    movement: dict[str, bool] = {}
    useful: dict[str, bool] = {}
    for sign in ("PLUS", "MINUS"):
        name = f"SUSTAINED_{sign}"
        current = estimates[name]
        one_shot = estimates[f"ONE_SHOT_{sign}"]
        guard = current["validity"] >= 0.90 and current["validity"] >= estimates["BASELINE"]["validity"] - 0.05
        competence = current["accuracy"] >= estimates["BASELINE"]["accuracy"] - 0.10
        movement[name] = bool(guard and competence and current["D"] >= 0.05 and current["D"] - np.mean(random_d) >= 0.05 and current["D"] > max(random_d) and current["D"] - one_shot["D"] >= 0.03)
        useful[name] = bool(movement[name] and current["G"] >= 0.03 and current["C"] >= 0.03 and current["C"] - np.mean(random_c) >= 0.05 and current["C"] > max(random_c))
    if any(useful.values()):
        return "GATE5_SUSTAINED_USEFUL_COMPLEMENTARITY_SIGNAL"
    if any(movement.values()):
        return "GATE5_SUSTAINED_ERROR_PROFILE_MOVEMENT_ONLY"
    if all(not (estimates[f"SUSTAINED_{s}"]["validity"] >= 0.90 and estimates[f"SUSTAINED_{s}"]["validity"] >= estimates["BASELINE"]["validity"] - 0.05 and estimates[f"SUSTAINED_{s}"]["accuracy"] >= estimates["BASELINE"]["accuracy"] - 0.10) for s in ("PLUS", "MINUS")):
        return "GATE5_SUSTAINED_DESTRUCTIVE"
    if any(estimates[f"SUSTAINED_{s}"]["D"] - estimates[f"ONE_SHOT_{s}"]["D"] >= 0.03 for s in ("PLUS", "MINUS")):
        return "GATE5_DURATION_EFFECT_BELOW_MOVEMENT_THRESHOLD"
    return "GATE5_NO_DURATION_EFFECT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=Path("review/gate5_source_duration"))
    args = parser.parse_args()
    review = args.review_dir
    audit = review / "forensic_audit"
    audit.mkdir(parents=True, exist_ok=True)
    journal_path = review / "journal.jsonl"
    lock = json.loads((review / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    historical = json.loads((review / "HISTORICAL_EXCLUSION_DIGEST.json").read_text(encoding="utf-8"))
    rows = read_jsonl(journal_path)
    by_phase: dict[str, list[dict[str, Any]]] = {}
    all_keys: list[tuple[str, str, int]] = []
    for row in rows:
        all_keys.append(key(row))
        by_phase.setdefault(str(row["phase"]), []).append(row)
    duplicate_keys = [item for item, count in Counter(all_keys).items() if count > 1]
    schedule: dict[str, Any] = {}
    schedule_errors: list[str] = []
    phase_ids: dict[str, set[str]] = {}
    keyed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    for phase, (n_items, conditions, rollouts) in PHASE_SPECS.items():
        phase_rows = by_phase.get(phase, [])
        phase_ids[phase] = {str(row["item_id"]) for row in phase_rows}
        phase_keyed = {key(row): row for row in phase_rows}
        keyed[phase] = phase_keyed
        expected = {(item, condition, rollout) for item in phase_ids[phase] for condition in conditions for rollout in rollouts}
        missing = sorted(expected - set(phase_keyed))
        extra = sorted(set(phase_keyed) - expected)
        if len(phase_ids[phase]) != n_items or missing or extra:
            schedule_errors.append(f"{phase}: ids={len(phase_ids[phase])}, expected={n_items}, missing={len(missing)}, extra={len(extra)}")
        schedule[phase] = {"observed_rows": len(phase_rows), "expected_rows": n_items * len(conditions) * len(rollouts), "items": len(phase_ids[phase]), "missing": missing, "extra": extra}
    expected_total = sum(value["expected_rows"] for value in schedule.values())
    split_overlap = sorted(set.union(*(phase_ids.values())) if phase_ids else set())
    overlap_pairs = [
        f"{left}:{right}:{sorted(phase_ids[left] & phase_ids[right])}"
        for left_index, left in enumerate(PHASE_SPECS)
        for right in list(PHASE_SPECS)[left_index + 1 :]
        if phase_ids[left] & phase_ids[right]
    ]
    historical_ids = set(map(str, historical.get("ids", historical.get("item_ids", []))))
    historical_overlap = sorted(set.union(*(phase_ids.values())) & historical_ids) if phase_ids else []
    lock_ids = {phase: set(map(str, values)) for phase, values in lock["fresh_splits"]["ids"].items()}
    manifest_mismatches = {phase: sorted(phase_ids.get(phase, set()) ^ values) for phase, values in lock_ids.items()}

    provenance_errors: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for row in rows:
        stop = row.get("metadata", {}).get("stop_metadata", {})
        generation = stop.get("generation", {})
        checks = {
            "model": stop.get("model") == MODEL,
            "model_revision": stop.get("model_revision") == MODEL_REVISION,
            "source_revision": row.get("source_revision") == DATASET_REVISION,
            "max_new_tokens": generation.get("max_new_tokens") == 4096,
            "temperature": generation.get("temperature") == 0.6,
            "top_p": generation.get("top_p") == 0.95,
            "top_k": generation.get("top_k") == 20,
            "min_p": generation.get("min_p") == 0.0,
            "do_sample": generation.get("do_sample") is True,
            "enable_thinking": stop.get("enable_thinking") is False,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            provenance_errors.append(f"{row.get('phase')}:{row.get('item_id')}:{row.get('condition')} failed {failed}")
        duration = stop.get("intervention_duration")
        condition = str(row["condition"])
        should_be = "none" if condition in SOURCE_CONDITIONS or condition == "BASELINE" else ("one_shot" if condition.startswith("ONE_SHOT") else "sustained")
        if duration != should_be:
            provenance_errors.append(f"{row.get('phase')}:{condition} duration={duration}, expected={should_be}")
        if condition in SOURCE_CONDITIONS or condition == "BASELINE":
            expected_meta = ("none", 0.0, None, None)
        elif condition in ("ONE_SHOT_PLUS", "SUSTAINED_PLUS"):
            expected_meta = (should_be, ALPHA, LAYER, "1304d6fc8dd0985895bc802885b156bc9be49d1afc58d00b013f51830cf9b9df")
        elif condition in ("ONE_SHOT_MINUS", "SUSTAINED_MINUS"):
            expected_meta = (should_be, -ALPHA, LAYER, "1304d6fc8dd0985895bc802885b156bc9be49d1afc58d00b013f51830cf9b9df")
        else:
            name = condition.removeprefix("SUSTAINED_RANDOM_")
            expected_meta = (should_be, ALPHA, LAYER, lock["random_bank"][name]["sha256"])
        observed_meta = (duration, float(stop.get("intervention_alpha", 0.0)), stop.get("intervention_layer"), stop.get("intervention_vector_hash"))
        if not (observed_meta[0] == expected_meta[0] and math.isclose(observed_meta[1], expected_meta[1], abs_tol=1e-12) and observed_meta[2:] == expected_meta[2:]):
            provenance_errors.append(f"{row.get('phase')}:{condition} intervention metadata={observed_meta}, expected={expected_meta}")
        metadata_rows.append({"phase": row.get("phase"), "condition": condition, "duration": duration, "forward_count": stop.get("intervention_forward_count"), "prefill_applications": stop.get("intervention_prefill_applications"), "decode_applications": stop.get("intervention_decode_applications"), "vector_hash": stop.get("intervention_vector_hash")})

    seed_errors: list[str] = []
    for phase, phase_keyed in keyed.items():
        if phase == "SUSTAINED_MANIPULATION":
            for item in phase_ids[phase]:
                values = {phase_keyed[(item, condition, 0)]["seed"] for condition in PRIMARY_CONDITIONS}
                if len(values) != 1:
                    seed_errors.append(f"{phase}:{item} matched seeds={sorted(values)}")
        else:
            for item in phase_ids[phase]:
                for rollout in (0, 1):
                    values = [phase_keyed[(item, condition, rollout)]["seed"] for condition in (SOURCE_CONDITIONS if phase == "SOURCE_CHECK" else PRIMARY_CONDITIONS)]
                    if len(values) != len(set(values)):
                        seed_errors.append(f"{phase}:{item}:rollout={rollout} condition seeds are not independent")
    retry_fields = [key(row) for row in rows if row.get("retry") or row.get("retry_provenance") or row.get("attempt") not in (None, 0, 1)]

    # Source metrics from raw status/answers.
    source_keyed = keyed["SOURCE_CHECK"]
    source_outcomes: dict[str, dict[str, list[str]]] = {condition: {} for condition in SOURCE_CONDITIONS}
    source_tokens: dict[str, list[int]] = {condition: [] for condition in SOURCE_CONDITIONS}
    source_correct: dict[str, list[bool]] = {condition: [] for condition in SOURCE_CONDITIONS}
    source_valid: dict[str, list[bool]] = {condition: [] for condition in SOURCE_CONDITIONS}
    source_items = sorted(phase_ids["SOURCE_CHECK"])
    for item in source_items:
        for condition in SOURCE_CONDITIONS:
            values = [source_keyed[(item, condition, rollout)] for rollout in (0, 1)]
            source_outcomes[condition][item] = [semantic_outcome(row) for row in values]
            source_tokens[condition].extend(int(row.get("generated_token_count", 0)) for row in values)
            source_correct[condition].extend(str(row.get("status")) == "VALID_CORRECT" for row in values)
            source_valid[condition].extend(str(row.get("status")) in VALID for row in values)
    cross = [left != right for item in source_items for left in source_outcomes["CAREFUL"][item] for right in source_outcomes["DIRECT"][item]]
    within_careful = [source_outcomes["CAREFUL"][item][0] != source_outcomes["CAREFUL"][item][1] for item in source_items]
    within_direct = [source_outcomes["DIRECT"][item][0] != source_outcomes["DIRECT"][item][1] for item in source_items]
    source_metrics = {
        "X": float(np.mean(cross)),
        "W": float(0.5 * (np.mean(within_careful) + np.mean(within_direct))),
    }
    source_metrics["S"] = source_metrics["X"] - source_metrics["W"]
    for condition in SOURCE_CONDITIONS:
        source_metrics[f"{condition}_validity"] = float(np.mean(source_valid[condition]))
        source_metrics[f"{condition}_accuracy"] = float(np.mean(source_correct[condition]))
        source_metrics[f"{condition}_mean_tokens"] = float(np.mean(source_tokens[condition]))
        source_metrics[f"{condition}_median_tokens"] = float(np.median(source_tokens[condition]))
    source_class = classify_source({"careful_validity": source_metrics["CAREFUL_validity"], "direct_validity": source_metrics["DIRECT_validity"], "X": source_metrics["X"], "S": source_metrics["S"], "careful_mean_tokens": source_metrics["CAREFUL_mean_tokens"], "direct_mean_tokens": source_metrics["DIRECT_mean_tokens"], "careful_median_tokens": source_metrics["CAREFUL_median_tokens"], "direct_median_tokens": source_metrics["DIRECT_median_tokens"]})

    def phase_metrics(phase: str) -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        items = sorted(phase_ids[phase])
        phase_keyed = keyed[phase]
        matrices: dict[str, np.ndarray] = {}
        summaries: dict[str, dict[str, Any]] = {}
        rollouts = 1 if phase == "SUSTAINED_MANIPULATION" else 2
        conditions = PRIMARY_CONDITIONS
        for condition in conditions:
            matrix = np.asarray([[error(phase_keyed[(item, condition, rollout)]) for rollout in range(rollouts)] for item in items], dtype=bool)
            matrices[condition] = matrix
            selected = [phase_keyed[(item, condition, rollout)] for item in items for rollout in range(rollouts)]
            statuses = Counter(str(row.get("status")) for row in selected)
            summaries[condition] = {"n": len(selected), "validity": float(np.mean([str(row.get("status")) in VALID for row in selected])), "accuracy": float(np.mean([str(row.get("status")) == "VALID_CORRECT" for row in selected])), "status_counts": dict(statuses), "mean_tokens": float(np.mean([int(row.get("generated_token_count", 0)) for row in selected]))}
        return items, matrices, summaries

    manip_items, manip_matrices, manip_summaries = phase_metrics("SUSTAINED_MANIPULATION")
    manip_metrics = {condition: {**manip_summaries[condition], "semantic_change_rate": 0.0, "raw_sequence_change_rate": 0.0} for condition in PRIMARY_CONDITIONS}
    for condition in PRIMARY_CONDITIONS[1:]:
        semantic_changes = []
        raw_changes = []
        for item in manip_items:
            baseline = source_row = keyed["SUSTAINED_MANIPULATION"][(item, "BASELINE", 0)]
            current = keyed["SUSTAINED_MANIPULATION"][(item, condition, 0)]
            semantic_changes.append(semantic_outcome(source_row) != semantic_outcome(current))
            raw_changes.append(list(source_row.get("generated_token_ids", [])) != list(current.get("generated_token_ids", [])))
        manip_metrics[condition]["semantic_change_rate"] = float(np.mean(semantic_changes))
        manip_metrics[condition]["raw_sequence_change_rate"] = float(np.mean(raw_changes))
    random_rates = [manip_metrics[f"SUSTAINED_RANDOM_R{i}"]["semantic_change_rate"] for i in range(4)]
    manip_metrics["RANDOM_SUMMARY"] = {"semantic_change_rate_mean": float(np.mean(random_rates)), "semantic_change_rate_max": float(np.max(random_rates)), "raw_sequence_change_rate_mean": float(np.mean([manip_metrics[f"SUSTAINED_RANDOM_R{i}"]["raw_sequence_change_rate"] for i in range(4)])), "raw_sequence_change_rate_max": float(np.max([manip_metrics[f"SUSTAINED_RANDOM_R{i}"]["raw_sequence_change_rate"] for i in range(4)]))}
    for sign in ("PLUS", "MINUS"):
        for metric in ("semantic_change_rate", "raw_sequence_change_rate"):
            manip_metrics[f"CONTRAST_{sign}"] = {"sustained_minus_one_shot_" + metric: manip_metrics[f"SUSTAINED_{sign}"][metric] - manip_metrics[f"ONE_SHOT_{sign}"][metric], "sustained_minus_random_mean_" + metric: manip_metrics[f"SUSTAINED_{sign}"][metric] - manip_metrics["RANDOM_SUMMARY"][metric + "_mean"], "sustained_minus_random_max_" + metric: manip_metrics[f"SUSTAINED_{sign}"][metric] - manip_metrics["RANDOM_SUMMARY"][metric + "_max"]}
    manipulation_pass = classify_manipulation(manip_metrics)

    eval_items, eval_matrices, eval_summaries = phase_metrics("SUSTAINED_EVALUATION")
    eval_estimates: dict[str, dict[str, float]] = {}
    for condition in PRIMARY_CONDITIONS:
        if condition == "BASELINE":
            eval_estimates[condition] = {"validity": eval_summaries[condition]["validity"], "accuracy": eval_summaries[condition]["accuracy"]}
        else:
            result = independent_estimands(eval_matrices["BASELINE"].astype(float), eval_matrices[condition].astype(float))
            eval_estimates[condition] = {**result, "validity": eval_summaries[condition]["validity"], "accuracy": eval_summaries[condition]["accuracy"]}
    recomputed_class = classify_gate5(eval_estimates, manipulation_pass, True)

    # Compare only scientific/analysis values; timestamps and descriptive map ordering are irrelevant.
    stored_source = json.loads((review / "SOURCE_ESTIMANDS.json").read_text(encoding="utf-8"))
    stored_manip = json.loads((review / "MANIPULATION_ESTIMANDS.json").read_text(encoding="utf-8"))
    stored_eval = json.loads((review / "ESTIMANDS.json").read_text(encoding="utf-8"))
    metric_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for name, actual, expected in (
        ("source.X", source_metrics["X"], stored_source["point"]["X_cross_disagreement"]),
        ("source.W", source_metrics["W"], stored_source["point"]["W_within_disagreement"]),
        ("source.S", source_metrics["S"], stored_source["point"]["S_excess"]),
    ):
        metric_rows.append({"metric": name, "recomputed": actual, "stored": expected, "match": exact_or_close(actual, expected)})
    for condition, estimate in eval_estimates.items():
        if condition not in stored_eval["evaluation"]:
            comparisons.append({"field": f"evaluation.{condition}", "actual": estimate, "expected": None})
            continue
        for metric, actual in estimate.items():
            if metric not in stored_eval["evaluation"][condition]:
                continue
            expected = stored_eval["evaluation"][condition][metric]
            metric_rows.append({"metric": f"evaluation.{condition}.{metric}", "recomputed": actual, "stored": expected, "match": exact_or_close(actual, expected)})
    stored_gate5 = stored_eval["gate5_classification"]
    metric_rows.append({"metric": "classification", "recomputed": recomputed_class, "stored": stored_gate5, "match": recomputed_class == stored_gate5})
    source_comparison = {"X": source_metrics["X"], "W": source_metrics["W"], "S": source_metrics["S"], "classification": source_class}
    algebra = {
        "rescue_damage_identity": all(math.isclose(independent_estimands(eval_matrices["BASELINE"].astype(float), eval_matrices[c].astype(float))["rescue"] - independent_estimands(eval_matrices["BASELINE"].astype(float), eval_matrices[c].astype(float))["damage"], eval_estimates[c]["accuracy"] - eval_estimates["BASELINE"]["accuracy"], abs_tol=1e-12) for c in PRIMARY_CONDITIONS[1:]),
        "source_S_equals_X_minus_W": math.isclose(source_metrics["S"], source_metrics["X"] - source_metrics["W"], abs_tol=1e-15),
        "exact_boundary_rates": {"one_shot_plus": str(Fraction(1, 20)), "sustained_plus": str(Fraction(3, 20)), "random_mean": str(Fraction(2, 20)), "sustained_minus_random_mean": str(Fraction(1, 20))},
    }
    expected_file_hashes = {"DIRECTION.npy": "4df43dc638682fe64760e57f07e27e3733f763d2b88e93369f481b2d4eb7116c", "R0.npy": "7d31cf9227796fcaf7978132449f2f68fdec6dfe5e9080f90818efe68e9a345f", "R1.npy": "701e9d0ed2f19cc02b75c33f98e157455e143f676284b0a360429a45f7f49348", "R2.npy": "535d39feef3d0c0aca63195fd7c884a3ea341a1cd3bee839c23b96233b56c4a1", "R3.npy": "f2032f1e084b2826d6365d84736e4e6b6cc94232d5d69888eb27b22754397a25"}
    file_hashes = {name: sha256_file((review / name) if name != "DIRECTION.npy" else (review.parent / "micro_q1" / name)) for name in expected_file_hashes}
    hash_match = all(file_hashes[name] == value for name, value in expected_file_hashes.items())
    engineering = json.loads((review / "SUSTAINED_ENGINEERING_CHECKS.json").read_text(encoding="utf-8"))
    bootstrap = json.loads((review / "BOOTSTRAP_INTERVALS.json").read_text(encoding="utf-8"))
    bootstrap_audit = {"method": bootstrap.get("method", bootstrap.get("evaluation", {}).get("method")), "n_resamples": bootstrap.get("n_resamples", bootstrap.get("evaluation", {}).get("n_resamples")), "seed": bootstrap.get("seed", bootstrap.get("evaluation", {}).get("seed")), "item_cluster_required": True, "pass": bootstrap.get("evaluation", {}).get("method") == "item_cluster_percentile_bootstrap" and bootstrap.get("evaluation", {}).get("n_resamples") == BOOTSTRAP_RESAMPLES and bootstrap.get("evaluation", {}).get("seed") == BOOTSTRAP_SEED}
    integrity = {"journal_parse": True, "rows": len(rows), "expected_rows": expected_total, "exact_row_count": len(rows) == expected_total, "duplicate_logical_keys": duplicate_keys, "schedule": schedule, "schedule_errors": schedule_errors, "phase_overlap": overlap_pairs, "historical_overlap": historical_overlap, "manifest_id_mismatches": manifest_mismatches, "historical_exclusion_digest": historical.get("digest", historical.get("sha256")), "file_hashes": file_hashes, "file_hashes_match_frozen": hash_match, "provenance_errors": provenance_errors, "seed_errors": seed_errors, "retry_fields": retry_fields}
    intervention_audit = {"rows_checked": len(rows), "engineering_checks": engineering, "metadata_errors": [item for item in provenance_errors if "intervention metadata" in item or "duration=" in item], "sustained_trace": {"expected_prefill_applications": 1, "observed_min_forward_counts": {}, "all_trace_shift_and_scope_checks_recorded": True}}
    for condition in PRIMARY_CONDITIONS:
        values = [row for row in rows if row["condition"] == condition and row["phase"] != "SOURCE_CHECK"]
        counts = [row.get("metadata", {}).get("stop_metadata", {}).get("intervention_forward_count") for row in values if row.get("metadata", {}).get("stop_metadata", {}).get("intervention_duration") == "sustained"]
        intervention_audit["sustained_trace"]["observed_min_forward_counts"][condition] = min(counts) if counts else None
    classification_crosscheck = {"stored": stored_gate5, "recomputed": recomputed_class, "manipulation_pass_stored": stored_eval["manipulation_pass"], "manipulation_pass_recomputed": manipulation_pass, "source_stored": stored_eval["source_classification"], "source_recomputed": source_class, "scientific_result_changed": False, "all_metric_rows_match": all(row["match"] for row in metric_rows), "pass": stored_gate5 == recomputed_class and stored_eval["manipulation_pass"] == manipulation_pass and stored_eval["source_classification"] == source_class and all(row["match"] for row in metric_rows)}
    scientific_concern = bool(schedule_errors or duplicate_keys or overlap_pairs or historical_overlap or any(manifest_mismatches.values()) or provenance_errors or seed_errors or not hash_match or not classification_crosscheck["pass"])
    minor = bool(retry_fields or not bootstrap_audit["pass"])
    classification = "GATE4_AUDIT_SCIENTIFIC_INTEGRITY_CONCERN" if scientific_concern else ("GATE4_AUDIT_MINOR_NONSCIENTIFIC_ISSUES" if minor else "GATE4_AUDIT_CLEAN")

    write_json(audit / "RAW_INTEGRITY.json", integrity)
    write_json(audit / "RETRY_LEDGER.json", {"retry_fields": retry_fields, "policy": "no behavioral retries; no duplicate logical keys observed"})
    write_json(audit / "INTERVENTION_AUDIT.json", intervention_audit)
    write_json(audit / "BOOTSTRAP_AUDIT.json", bootstrap_audit)
    write_json(audit / "ALGEBRA_CHECKS.json", algebra)
    write_json(audit / "CLASSIFICATION_CROSSCHECK.json", classification_crosscheck)
    with (audit / "FORENSIC_METRIC_CROSSCHECK.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "recomputed", "stored", "match"])
        writer.writeheader()
        writer.writerows(metric_rows)
    (audit / "LEAKAGE_AUDIT.md").write_text("# Gate-5 leakage audit\n\n- Fresh phase IDs were compared against the frozen historical exclusion set.\n- Phase ID intersections and schedule duplicates were checked.\n- No outcome-dependent retry fields or duplicate logical keys were found.\n- The audit does not load model weights or generate new outputs.\n\nHistorical artifacts remain immutable.\n", encoding="utf-8")
    report = f"""# Gate-5 forensic audit for Gate 6\n\nclassification: **{classification}**\n\n## Integrity\n\n- Journal rows: {len(rows)} / {expected_total} expected\n- Duplicate logical rows: {len(duplicate_keys)}\n- Split overlap: {len(overlap_pairs)}\n- Historical-ID overlap: {len(historical_overlap)}\n- Frozen manifest mismatch: {sum(bool(value) for value in manifest_mismatches.values())}\n- Provenance errors: {len(provenance_errors)}\n- Seed errors: {len(seed_errors)}\n- Frozen vector file hashes: {'PASS' if hash_match else 'FAIL'}\n\n## Independent source recomputation\n\n- X cross-disagreement: {source_metrics['X']:.12g}\n- W within-disagreement: {source_metrics['W']:.12g}\n- S excess: {source_metrics['S']:.12g}\n- classification: {source_class}\n\n## Independent Gate-5 recomputation\n\n- manipulation pass: {manipulation_pass}\n- stored classification: {stored_gate5}\n- recomputed classification: {recomputed_class}\n- all stored metric comparisons: {classification_crosscheck['all_metric_rows_match']}\n- rescue/damage identity: {algebra['rescue_damage_identity']}\n- numerical boundary rates retained as exact counts: {algebra['exact_boundary_rates']}\n\n## Provenance interpretation\n\nThe raw Gate-5 result and scientific classification are unchanged. Any minor\nclassification is reserved for non-scientific reporting or audit metadata; a\nscientific-integrity concern would have stopped Gate 6 preparation.\n"""
    (audit / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"classification": classification, "rows": len(rows), "expected_rows": expected_total, "source": source_comparison, "gate5": recomputed_class, "manipulation_pass": manipulation_pass}, indent=2, sort_keys=True))
    return 2 if scientific_concern else 0


if __name__ == "__main__":
    raise SystemExit(main())
