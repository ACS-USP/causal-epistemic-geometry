#!/usr/bin/env python3
"""Independent forensic audit of the completed Q2 V4.1 semantic campaign.

This audit deliberately does not call the primary semantic-analysis helpers.
It reads the sealed raw journal, applies the frozen parser for a row-level
cross-check, and recomputes the two-rollout estimands, shape matrices, QAP
statistics, and classification inputs with local implementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    evaluate_external_answer_v3,
)

REVIEW = ROOT / "review/q2_v4_1_prediction_lock"
EXECUTION = ROOT / "review/q2_v4_1_semantic_execution"
PANEL = REVIEW / "SEMANTIC_PANEL_MANIFEST.json"
SCHEDULE = REVIEW / "FUTURE_SEMANTIC_SCHEDULE.json"
JOURNAL = EXECUTION / "journal.jsonl"
SCORES = EXECUTION / "SEMANTIC_SCORES.jsonl"
ESTIMANDS = EXECUTION / "ESTIMANDS.json"
BOOTSTRAP = EXECUTION / "BOOTSTRAP_INTERVALS.json"
MATRICES = REVIEW / "PREDICTION_MATRICES.npz"
QAP = REVIEW / "QAP_CONTROLLER_PERMUTATIONS.npy"
CONTROLLER_COUNT = 31
N = 300
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1_885_846_737_463_784_981
BASELINE = "BASELINE"
SHELLS = ("MEDIUM", "STRONG")
METRICS = ("A0", "A1", "A2")
EXPECTED_CLASSIFICATIONS = {
    "Q2_V4_1_G0",
    "Q2_V4_1_G1",
    "Q2_V4_1_G2",
    "Q2_V4_1_G3",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_schedule() -> tuple[list[dict[str, Any]], list[str]]:
    payload = read_json(SCHEDULE)
    rows = list(payload["rows"])
    keys = [(str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in rows]
    if payload["status"] != "FROZEN_NOT_AUTHORIZED_NOT_RUN":
        raise RuntimeError("frozen schedule status changed")
    if len(rows) != 37_800 or len(set(keys)) != 37_800:
        raise RuntimeError("schedule cardinality/uniqueness failure")
    return rows, [str(value) for value in payload["conditions"]]


def load_panel() -> tuple[list[str], dict[str, dict[str, Any]]]:
    payload = read_json(PANEL)
    items = {str(row["item_id"]): row for row in payload["items"]}
    ordered = [str(value) for value in payload["item_ids"]]
    if payload["item_count"] != N or len(items) != N or ordered != list(items):
        raise RuntimeError("panel order/count failure")
    return ordered, items


def load_journal() -> dict[tuple[str, str, int], dict[str, Any]]:
    if not JOURNAL.is_file():
        raise RuntimeError("semantic journal missing")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    identity: dict[str, Any] | None = None
    with JOURNAL.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            wrapper = json.loads(line)
            if wrapper.get("version") != "research-os-jsonl-v1":
                raise RuntimeError(f"journal wrapper at line {line_number}")
            if wrapper.get("key_fields") != ["item_id", "condition", "rollout_index"]:
                raise RuntimeError(f"journal key fields at line {line_number}")
            if identity is None:
                identity = dict(wrapper.get("identity", {}))
            if wrapper.get("identity") != identity:
                raise RuntimeError("journal identity changed")
            row = wrapper["row"]
            key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
            if tuple(wrapper.get("key", ())) != key or key in records:
                raise RuntimeError(f"journal key mismatch/duplicate: {key}")
            records[key] = row
    if identity is None:
        raise RuntimeError("empty journal")
    return records


def classify_raw(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    parsed = evaluate_external_answer_v3(
        str(row.get("raw_output", "")),
        str(item["reference_answer"]),
        truncated=bool(row.get("truncated", False)),
        runtime_error=bool(row.get("runtime_error")),
    )
    return {
        "commitment_valid": bool(parsed.commitment_valid),
        "semantic_evaluable": bool(parsed.semantic_evaluable),
        "correct": bool(parsed.correct),
        "status": (
            "VALID_CORRECT"
            if parsed.correct
            else "VALID_WRONG"
            if parsed.commitment_valid and parsed.semantic_evaluable
            else "RUNTIME_ERROR"
            if parsed.failure_reason == "runtime error"
            else "TRUNCATED"
            if parsed.failure_reason == "truncated or unclosed response"
            else "INVALID_FORMAT"
        ),
        "failure_reason": parsed.failure_reason,
        "canonical_value": parsed.canonical_value,
        "value_type": parsed.value_type,
        "parsed_answer": parsed.payload,
    }


def load_primary_scores() -> dict[tuple[str, str, int], dict[str, Any]]:
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    with SCORES.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
            if key in scores:
                raise RuntimeError(f"duplicate primary score at line {line_number}")
            scores[key] = row
    return scores


def errors_by_condition(
    item_ids: list[str],
    conditions: list[str],
    journal: dict[tuple[str, str, int], dict[str, Any]],
    items: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], int]:
    errors: dict[str, np.ndarray] = {}
    parser_differences = 0
    primary_scores = load_primary_scores()
    for condition in conditions:
        values = np.zeros((N, 2), dtype=np.float64)
        for item_index, item_id in enumerate(item_ids):
            for rollout in (0, 1):
                key = (item_id, condition, rollout)
                raw = journal[key]
                audit_score = classify_raw(raw, items[item_id])
                primary_score = primary_scores[key]
                for field in ("commitment_valid", "semantic_evaluable", "correct", "status"):
                    if audit_score[field] != primary_score[field]:
                        parser_differences += 1
                values[item_index, rollout] = float(not audit_score["correct"])
        errors[condition] = values
    return errors, parser_differences


def explicit_estimand(base: np.ndarray, condition: np.ndarray) -> dict[str, float]:
    b00 = float(np.mean(base[:, 0] * base[:, 1]))
    cross = (
        base[:, 0] * condition[:, 0]
        + base[:, 0] * condition[:, 1]
        + base[:, 1] * condition[:, 0]
        + base[:, 1] * condition[:, 1]
    ) / 4.0
    b0j = float(np.mean(cross))
    q0 = np.mean(base, axis=1)
    qj = np.mean(condition, axis=1)
    denominator = N * (N - 1)
    u00 = float((q0.sum() ** 2 - np.square(q0).sum()) / denominator)
    u0j = float((q0.sum() * qj.sum() - np.dot(q0, qj)) / denominator)
    distance = float(
        np.mean(
            base[:, 0] * base[:, 1]
            + condition[:, 0] * condition[:, 1]
            - base[:, 0] * condition[:, 1]
            - base[:, 1] * condition[:, 0]
        )
    )
    rescue = float(
        np.mean(
            (
                base[:, 0] * (1 - condition[:, 0])
                + base[:, 0] * (1 - condition[:, 1])
                + base[:, 1] * (1 - condition[:, 0])
                + base[:, 1] * (1 - condition[:, 1])
            )
            / 4.0
        )
    )
    damage = float(
        np.mean(
            (
                (1 - base[:, 0]) * condition[:, 0]
                + (1 - base[:, 0]) * condition[:, 1]
                + (1 - base[:, 1]) * condition[:, 0]
                + (1 - base[:, 1]) * condition[:, 1]
            )
            / 4.0
        )
    )
    return {
        "accuracy_baseline": float(1.0 - np.mean(base)),
        "accuracy_condition": float(1.0 - np.mean(condition)),
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


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_rank -= np.mean(left_rank)
    right_rank -= np.mean(right_rank)
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    return float(np.dot(left_rank, right_rank) / denominator) if denominator else math.nan


def shape_matrix(values: np.ndarray) -> np.ndarray:
    delta0 = values[:, None, :, 0] - values[None, :, :, 0]
    delta1 = values[:, None, :, 1] - values[None, :, :, 1]
    total = np.mean(delta0 * delta1, axis=2)
    mean_product = np.mean(delta0, axis=2) * np.mean(delta1, axis=2)
    return (total - mean_product) * (N / (N - 1.0))


def load_geometries() -> dict[str, dict[str, np.ndarray]]:
    with np.load(MATRICES, allow_pickle=False) as archive:
        return {
            metric: {
                shell: np.asarray(archive[f"{metric}_{shell}"], dtype=np.float64)
                for shell in SHELLS
            }
            for metric in METRICS
        }


def outcome_geometry(
    errors: dict[str, np.ndarray], controller_ids: list[str]
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for shell in SHELLS:
        values = np.stack([errors[BASELINE]] + [errors[f"{cid}_{shell}"] for cid in controller_ids])
        result[shell] = shape_matrix(values)[1:, 1:]
    return result


def qap_recompute(
    geometries: dict[str, dict[str, np.ndarray]], outcomes: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    permutations = np.load(QAP, allow_pickle=False)
    if permutations.shape != (50_000, CONTROLLER_COUNT):
        raise RuntimeError("QAP shape changed")
    upper = np.triu_indices(CONTROLLER_COUNT, 1)
    null = np.empty((len(permutations), len(METRICS)), dtype=np.float64)
    for permutation_index, permutation in enumerate(permutations):
        for metric_index, metric in enumerate(METRICS):
            shell_values = []
            for shell in SHELLS:
                permuted = geometries[metric][shell][np.ix_(permutation, permutation)]
                shell_values.append(spearman(permuted[upper], outcomes[shell][upper]))
            null[permutation_index, metric_index] = float(np.mean(shell_values))
    observed = {metric: float(null[0, index]) for index, metric in enumerate(METRICS)}
    maximum = np.max(null, axis=1)
    adjusted = {metric: float(np.mean(maximum >= observed[metric])) for metric in METRICS}
    summary = {
        "observed": observed,
        "raw_p": {
            metric: float(np.mean(null[:, index] >= observed[metric]))
            for index, metric in enumerate(METRICS)
        },
        "maxT_adjusted_p": adjusted,
        "maps": int(len(permutations)),
    }
    return summary, {metric: null[:, index] for index, metric in enumerate(METRICS)}


def classify_metrics(
    geometries: dict[str, dict[str, np.ndarray]],
    outcomes: dict[str, np.ndarray],
    qap: dict[str, Any],
    primary_bootstrap: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    upper = np.triu_indices(CONTROLLER_COUNT, 1)
    values: dict[str, Any] = {}
    for metric in METRICS:
        shell_rho = {
            shell: spearman(geometries[metric][shell][upper], outcomes[shell][upper])
            for shell in SHELLS
        }
        qualifies = bool(
            shell_rho["MEDIUM"] > 0.0
            and shell_rho["STRONG"] > 0.0
            and np.mean(list(shell_rho.values())) >= 0.2
            and primary_bootstrap[metric]["q025"] > 0.0
            and qap["maxT_adjusted_p"][metric] <= 0.05
        )
        values[metric] = {"shell_rho": shell_rho, "qualifies": qualifies}
    if values["A2"]["qualifies"]:
        classification = "Q2_V4_1_G2"
    elif values["A0"]["qualifies"] or values["A1"]["qualifies"]:
        classification = "Q2_V4_1_G1"
    else:
        classification = "Q2_V4_1_G0"
    return values, classification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="reserved for CLI compatibility")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    schedule, conditions = load_schedule()
    item_ids, items = load_panel()
    journal = load_journal()
    expected = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in schedule
    }
    if set(journal) != expected:
        raise RuntimeError("journal does not match frozen schedule")
    errors, parser_differences = errors_by_condition(item_ids, conditions, journal, items)
    primary = read_json(ESTIMANDS)
    if primary.get("classification") not in EXPECTED_CLASSIFICATIONS:
        raise RuntimeError("primary classification is not a frozen G-class")
    independent_estimands = {
        condition: explicit_estimand(errors[BASELINE], errors[condition])
        for condition in conditions
        if condition != BASELINE
    }
    base = errors[BASELINE]
    independent_baseline = {
        "B00": float(np.mean(base[:, 0] * base[:, 1])),
        "O00": float(1.0 - np.mean(base[:, 0] * base[:, 1])),
    }
    primary_differences = []
    for condition, computed in independent_estimands.items():
        for name, value in computed.items():
            primary_value = float(primary["estimands"][condition][name])
            primary_differences.append(
                {
                    "object": "estimand",
                    "condition": condition,
                    "field": name,
                    "primary": primary_value,
                    "audit": value,
                    "absolute_difference": abs(primary_value - value),
                }
            )
    for name, value in independent_baseline.items():
        primary_value = float(primary["estimands"][BASELINE][name])
        primary_differences.append(
            {
                "object": "baseline",
                "condition": BASELINE,
                "field": name,
                "primary": primary_value,
                "audit": value,
                "absolute_difference": abs(primary_value - value),
            }
        )
    controller_ids = [
        condition.removesuffix("_MEDIUM")
        for condition in conditions
        if condition.endswith("_MEDIUM")
    ]
    outcomes = outcome_geometry(errors, controller_ids)
    geometries = load_geometries()
    qap, _qap_null = qap_recompute(geometries, outcomes)
    primary_bootstrap = read_json(BOOTSTRAP)
    metric_audit, classification = classify_metrics(geometries, outcomes, qap, primary_bootstrap)
    crosscheck_rows = list(primary_differences)
    for metric in METRICS:
        for shell in SHELLS:
            observed = float(primary["metrics"][metric]["shell_rho"][shell])
            audit = float(metric_audit[metric]["shell_rho"][shell])
            crosscheck_rows.append(
                {
                    "object": "metric",
                    "condition": metric,
                    "field": f"{shell}_rho",
                    "primary": observed,
                    "audit": audit,
                    "absolute_difference": abs(observed - audit),
                }
            )
        observed = float(primary["qap"]["observed"][metric])
        audit = float(qap["observed"][metric])
        crosscheck_rows.append(
            {
                "object": "qap",
                "condition": metric,
                "field": "observed",
                "primary": observed,
                "audit": audit,
                "absolute_difference": abs(observed - audit),
            }
        )
    max_difference = max(float(row["absolute_difference"]) for row in crosscheck_rows)
    write_csv(EXECUTION / "METRIC_CROSSCHECK.csv", crosscheck_rows)
    retry_rows = []
    for key, row in sorted(journal.items()):
        if int(row.get("retry_count", 0)) or row.get("retry_reasons"):
            retry_rows.append(
                {
                    "key": list(key),
                    "retry_count": row.get("retry_count", 0),
                    "reasons": row.get("retry_reasons", []),
                }
            )
    write_json(
        EXECUTION / "RETRY_LEDGER.json",
        {
            "rows_with_retries": len(retry_rows),
            "rows": retry_rows,
            "policy": "frozen normative lock",
        },
    )
    write_json(
        EXECUTION / "CLASSIFICATION_CROSSCHECK.json",
        {
            "primary": primary["classification"],
            "audit": classification,
            "metric_audit": metric_audit,
            "qap_observed": qap["observed"],
            "agreement": primary["classification"] == classification,
        },
    )
    report = (
        "# Q2 V4.1 semantic forensic audit\n\n"
        f"Primary classification: {primary['classification']}.\n\n"
        f"Independent classification: {classification}.\n\n"
        f"Maximum recomputed metric difference: {max_difference:.12g}.\n\n"
        f"Independent parser field differences: {parser_differences}.\n\n"
        "The audit read only the completed raw journal and frozen label-free "
        "matrices. No model inference or replacement output was performed.\n"
    )
    (EXECUTION / "FORENSIC_AUDIT.md").write_text(report, encoding="utf-8")
    write_json(
        EXECUTION / "FORENSIC_AUDIT.json",
        {
            "classification": (
                "Q2_V4_1_SEMANTIC_FORENSIC_CLEAN"
                if primary["classification"] == classification and parser_differences == 0
                else "Q2_V4_1_SEMANTIC_FORENSIC_INTEGRITY_CONCERN"
            ),
            "primary_classification": primary["classification"],
            "audit_classification": classification,
            "primary_audit_agreement": primary["classification"] == classification,
            "maximum_difference": max_difference,
            "parser_field_differences": parser_differences,
            "journal_sha256": sha256_file(JOURNAL),
            "scores_sha256": sha256_file(SCORES),
            "semantic_outcomes": len(journal),
            "correctness_inspected": True,
            "new_model_inference": False,
            "new_gpu_inference": False,
            "q3": "NOT_RUN",
            "git_head": git_head(),
        },
    )
    print(
        json.dumps(
            {"status": "FORENSIC_AUDIT_COMPLETE", "max_difference": max_difference},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
