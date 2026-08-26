#!/usr/bin/env python3
"""Seal and analyze the complete frozen Q2 V4.1 semantic campaign.

The ``seal`` command is outcome-blind: it checks only journal completeness and
freezes the raw campaign.  The ``analyze`` command is permitted only after the
seal and computes the hash-pinned semantic endpoint and classification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    evaluate_external_answer_v3,
)
from epistemic_geometry.experiments import q2_v4  # noqa: E402
from epistemic_geometry.experiments.gate6_3_v3 import (  # noqa: E402
    audit_two_rollout_estimands,
    item_contributions,
)

REVIEW = ROOT / "review/q2_v4_1_prediction_lock"
EXECUTION_REVIEW = ROOT / "review/q2_v4_1_semantic_execution"
PROTOCOL = REVIEW / "PROTOCOL_LOCK.json"
NORMATIVE = REVIEW / "Q2_V4_1_NORMATIVE_EXECUTION_AND_ANALYSIS_LOCK.json"
PANEL = REVIEW / "SEMANTIC_PANEL_MANIFEST.json"
SCHEDULE = REVIEW / "FUTURE_SEMANTIC_SCHEDULE.json"
QAP = REVIEW / "QAP_CONTROLLER_PERMUTATIONS.npy"
MATRICES = REVIEW / "PREDICTION_MATRICES.npz"
MATRIX_METADATA = REVIEW / "PREDICTION_MATRIX_METADATA.json"
JOURNAL = EXECUTION_REVIEW / "journal.jsonl"
COMPLETENESS = EXECUTION_REVIEW / "CAMPAIGN_COMPLETENESS.json"
RAW_SEAL = EXECUTION_REVIEW / "RAW_DATA_SEAL.json"
SCORES = EXECUTION_REVIEW / "SEMANTIC_SCORES.jsonl"

N = 300
CONTROLLER_COUNT = 31
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1_885_846_737_463_784_981
RADIAL_MAPS = 50_000
PRELOCK = "99782d6f4f3ce1ca52d2cf6caeacafd4d0de9081"
KEY_FIELDS = ("item_id", "condition", "rollout_index")
BASELINE = "BASELINE"


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


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_panel() -> tuple[list[str], dict[str, dict[str, Any]]]:
    payload = read_json(PANEL)
    items = {str(row["item_id"]): row for row in payload["items"]}
    if payload["item_count"] != N or len(items) != N:
        raise RuntimeError("semantic panel count is not frozen at 300")
    ordered = [str(value) for value in payload["item_ids"]]
    if ordered != [str(row["item_id"]) for row in payload["items"]]:
        raise RuntimeError("panel order is inconsistent")
    return ordered, items


def load_schedule() -> tuple[list[dict[str, Any]], list[str]]:
    payload = read_json(SCHEDULE)
    if payload["status"] != "FROZEN_NOT_AUTHORIZED_NOT_RUN":
        raise RuntimeError("semantic schedule status changed")
    rows = list(payload["rows"])
    keys = [(str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in rows]
    if len(rows) != 37_800 or len(set(keys)) != 37_800:
        raise RuntimeError("semantic schedule is incomplete or duplicated")
    conditions = [str(value) for value in payload["conditions"]]
    if len(conditions) != 63 or conditions[0] != BASELINE:
        raise RuntimeError("semantic schedule condition contract changed")
    return rows, conditions


def unwrap_journal() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not JOURNAL.is_file():
        raise RuntimeError("semantic journal is absent")
    rows: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    key_set: set[tuple[str, str, int]] = set()
    with JOURNAL.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            wrapper = json.loads(line)
            if wrapper.get("version") != "research-os-jsonl-v1":
                raise RuntimeError(f"unsupported journal wrapper at line {line_number}")
            if wrapper.get("key_fields") != list(KEY_FIELDS):
                raise RuntimeError(f"journal key contract changed at line {line_number}")
            if identity is None:
                identity = dict(wrapper.get("identity", {}))
            if wrapper.get("identity") != identity:
                raise RuntimeError(f"journal identity changed at line {line_number}")
            row = wrapper.get("row")
            if not isinstance(row, dict):
                raise RuntimeError(f"invalid journal row at line {line_number}")
            key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
            if tuple(wrapper.get("key", ())) != key:
                raise RuntimeError(f"journal key mismatch at line {line_number}")
            if key in key_set:
                raise RuntimeError(f"duplicate journal key {key}")
            key_set.add(key)
            rows.append(row)
    if identity is None:
        raise RuntimeError("empty semantic journal")
    return rows, identity


def expected_keys(schedule: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    return {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in schedule
    }


def validate_complete_campaign() -> tuple[
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]
]:
    schedule, _conditions = load_schedule()
    rows, identity = unwrap_journal()
    wanted = expected_keys(schedule)
    actual = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in rows
    }
    if len(rows) != 37_800 or actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        raise RuntimeError(
            f"campaign incomplete: rows={len(rows)} missing={len(missing)} extra={len(extra)}"
        )
    for row in rows:
        for field in (
            "raw_output",
            "generated_token_ids",
            "generated_token_count",
            "truncated",
            "seed",
        ):
            if field not in row:
                raise RuntimeError(
                    f"raw row missing {field}: {row.get('item_id')}/{row.get('condition')}"
                )
        if row.get("semantic_scoring") != "DEFERRED_UNTIL_COMPLETE":
            raise RuntimeError("semantic scoring marker was changed during collection")
    return rows, identity, schedule


def seal() -> None:
    rows, identity, schedule = validate_complete_campaign()
    completeness = {
        "schema_version": "q2-v4.1-campaign-completeness-v1",
        "status": "COMPLETE_RAW_UNSCORED_CAMPAIGN",
        "expected_logical_rows": 37_800,
        "observed_logical_rows": len(rows),
        "unique_logical_keys": len(rows),
        "duplicates": 0,
        "missing": 0,
        "unexpected": 0,
        "replacements": 0,
        "semantic_scoring": "NOT_PERFORMED",
        "correctness_inspected": False,
        "journal_sha256": sha256_file(JOURNAL),
        "journal_identity": identity,
        "schedule_sha256": sha256_file(SCHEDULE),
        "code_commit": git_head(),
    }
    write_json(COMPLETENESS, completeness)
    raw_seal = {
        "schema_version": "q2-v4.1-raw-semantic-data-seal-v1",
        "status": "RAW_DATA_SEALED_BEFORE_ANALYSIS",
        "journal": {"path": str(JOURNAL.relative_to(ROOT)), "sha256": sha256_file(JOURNAL)},
        "campaign_completeness": {
            "path": str(COMPLETENESS.relative_to(ROOT)),
            "sha256": sha256_file(COMPLETENESS),
        },
        "schedule_sha256": sha256_file(SCHEDULE),
        "raw_outputs_present": True,
        "parser_scoring": "NOT_PERFORMED",
        "correctness_inspected": False,
        "rows": len(rows),
        "sealed_by_commit": git_head(),
        "identity": identity,
    }
    write_json(RAW_SEAL, raw_seal)
    print(json.dumps({"status": completeness["status"], "rows": len(rows)}, sort_keys=True))


def classify_row(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_external_answer_v3(
        str(row.get("raw_output", "")),
        str(item["reference_answer"]),
        truncated=bool(row.get("truncated", False)),
        runtime_error=bool(row.get("runtime_error")),
    )
    if result.correct:
        status = "VALID_CORRECT"
    elif result.commitment_valid and result.semantic_evaluable:
        status = "VALID_WRONG"
    elif result.failure_reason == "runtime error":
        status = "RUNTIME_ERROR"
    elif result.failure_reason == "truncated or unclosed response":
        status = "TRUNCATED"
    else:
        status = "INVALID_FORMAT"
    return {
        "item_id": str(row["item_id"]),
        "condition": str(row["condition"]),
        "rollout_index": int(row["rollout_index"]),
        "commitment_valid": bool(result.commitment_valid),
        "semantic_evaluable": bool(result.semantic_evaluable),
        "correct": bool(result.correct),
        "status": status,
        "value_type": result.value_type,
        "canonical_value": result.canonical_value,
        "parsed_answer": result.payload,
        "failure_reason": result.failure_reason,
        "generated_token_count": int(row.get("generated_token_count", 0)),
        "truncated": bool(row.get("truncated", False)),
        "raw_output_sha256": hashlib.sha256(str(row.get("raw_output", "")).encode()).hexdigest(),
    }


def write_scores(
    rows: list[dict[str, Any]], items: dict[str, dict[str, Any]]
) -> dict[tuple[str, str, int], dict[str, Any]]:
    parsed: dict[tuple[str, str, int], dict[str, Any]] = {}
    temporary = SCORES.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
            score = classify_row(row, items[str(row["item_id"])])
            parsed[key] = score
            handle.write(json.dumps(score, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(SCORES)
    return parsed


def condition_summary(
    rows: list[dict[str, Any]],
    parsed: dict[tuple[str, str, int], dict[str, Any]],
    conditions: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    output: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float]] = {}
    for condition in conditions:
        selected = [row for row in rows if str(row["condition"]) == condition]
        results = [
            parsed[(str(row["item_id"]), condition, int(row["rollout_index"]))] for row in selected
        ]
        tokens = np.asarray(
            [int(row["generated_token_count"]) for row in selected], dtype=np.float64
        )
        statuses = Counter(str(result["status"]) for result in results)
        record: dict[str, Any] = {
            "condition": condition,
            "n": len(selected),
            "commitment_valid": int(sum(bool(result["commitment_valid"]) for result in results)),
            "commitment_validity": float(
                np.mean([result["commitment_valid"] for result in results])
            ),
            "semantic_evaluable": int(
                sum(bool(result["semantic_evaluable"]) for result in results)
            ),
            "semantic_evaluability": float(
                np.mean([result["semantic_evaluable"] for result in results])
            ),
            "correct": int(sum(bool(result["correct"]) for result in results)),
            "accuracy": float(np.mean([result["correct"] for result in results])),
            "error_count": int(sum(not bool(result["correct"]) for result in results)),
            "mean_tokens": float(np.mean(tokens)),
            "median_tokens": float(np.median(tokens)),
            "p75_tokens": float(np.quantile(tokens, 0.75)),
            "p90_tokens": float(np.quantile(tokens, 0.90)),
            "max_tokens": int(np.max(tokens)),
            "truncated": int(statuses["TRUNCATED"]),
            "runtime_error": int(statuses["RUNTIME_ERROR"]),
            "invalid_format": int(statuses["INVALID_FORMAT"]),
            "valid_wrong": int(statuses["VALID_WRONG"]),
            "status_counts": json.dumps(dict(sorted(statuses.items())), sort_keys=True),
        }
        output.append(record)
        metrics[condition] = {
            key: float(record[key])
            for key in (
                "commitment_validity",
                "semantic_evaluability",
                "accuracy",
                "mean_tokens",
                "median_tokens",
            )
        }
    return output, metrics


def error_arrays(
    item_ids: list[str], conditions: list[str], parsed: dict[tuple[str, str, int], dict[str, Any]]
) -> dict[str, np.ndarray]:
    return {
        condition: np.asarray(
            [
                [int(not parsed[(item, condition, rollout)]["correct"]) for rollout in (0, 1)]
                for item in item_ids
            ],
            dtype=np.float64,
        )
        for condition in conditions
    }


def pair_estimand(baseline: np.ndarray, condition: np.ndarray) -> dict[str, float]:
    return audit_two_rollout_estimands(baseline, condition)


def metric_arrays() -> dict[str, dict[str, np.ndarray]]:
    archive = np.load(MATRICES, allow_pickle=False)
    return {
        "A0": {
            "MEDIUM": np.asarray(archive["A0_MEDIUM"], dtype=np.float64),
            "STRONG": np.asarray(archive["A0_STRONG"], dtype=np.float64),
        },
        "A1": {
            "MEDIUM": np.asarray(archive["A1_MEDIUM"], dtype=np.float64),
            "STRONG": np.asarray(archive["A1_STRONG"], dtype=np.float64),
        },
        "A2": {
            "MEDIUM": np.asarray(archive["A2_MEDIUM"], dtype=np.float64),
            "STRONG": np.asarray(archive["A2_STRONG"], dtype=np.float64),
        },
        "D2": {
            "MEDIUM": np.asarray(archive["D2_MEDIUM"], dtype=np.float64),
            "STRONG": np.asarray(archive["D2_STRONG"], dtype=np.float64),
        },
    }


def semantic_distance_arrays(
    errors: dict[str, np.ndarray], controller_ids: list[str]
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    outcome_shape: dict[str, np.ndarray] = {}
    outcome_total: dict[str, np.ndarray] = {}
    radial_shape: dict[str, np.ndarray] = {}
    radial_total: dict[str, np.ndarray] = {}
    for shell in ("MEDIUM", "STRONG"):
        values = np.stack(
            [errors[BASELINE], *(errors[f"{cid}_{shell}"] for cid in controller_ids)], axis=0
        )
        distances = q2_v4.blind_spot_shape_matrices(values)
        outcome_shape[shell] = distances["shape_item_population"][1:, 1:]
        outcome_total[shell] = distances["total"][0, 1:]
        radial_shape[shell] = distances["shape_item_population"][0, 1:]
        radial_total[shell] = distances["total"][0, 1:]
    return outcome_shape, outcome_total, radial_shape, radial_total


def finite_shape(errors: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = errors[:, indices, :]
    d0 = values[:, None, :, 0] - values[None, :, :, 0]
    d1 = values[:, None, :, 1] - values[None, :, :, 1]
    total = np.mean(d0 * d1, axis=2)
    m0 = np.mean(d0, axis=2)
    m1 = np.mean(d1, axis=2)
    return (total - m0 * m1) * (len(indices) / (len(indices) - 1.0))


def bootstrap(
    errors_by_condition: dict[str, np.ndarray],
    item_ids: list[str],
    controller_ids: list[str],
    geometries: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    del item_ids
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    upper = np.triu_indices(CONTROLLER_COUNT, 1)
    samples: dict[str, list[float]] = {name: [] for name in ("A0", "A1", "A2")}
    samples.update({"A2_minus_A0": [], "A2_minus_A1": []})
    base_stack = errors_by_condition[BASELINE]
    shell_errors = {
        shell: np.stack([errors_by_condition[f"{cid}_{shell}"] for cid in controller_ids], axis=0)
        for shell in ("MEDIUM", "STRONG")
    }
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, N, size=N)
        base = base_stack[indices]
        values = {
            shell: finite_shape(
                np.concatenate((base[None, :, :], shell_errors[shell][:, indices, :]), axis=0),
                np.arange(N),
            )[1:, 1:]
            for shell in ("MEDIUM", "STRONG")
        }
        rhos: dict[str, float] = {}
        for metric in ("A0", "A1", "A2"):
            shell_rhos = [
                float(q2_v4.spearman(geometries[metric][shell][upper], values[shell][upper]))
                for shell in ("MEDIUM", "STRONG")
            ]
            rhos[metric] = float(np.mean(shell_rhos))
            samples[metric].append(rhos[metric])
        samples["A2_minus_A0"].append(rhos["A2"] - rhos["A0"])
        samples["A2_minus_A1"].append(rhos["A2"] - rhos["A1"])
    return {
        name: {
            "estimate": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        }
        for name, values in samples.items()
    }


def qap_summary(
    geometries: dict[str, dict[str, np.ndarray]], outcomes: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    permutations = np.load(QAP, allow_pickle=False)
    if permutations.shape != (50_000, CONTROLLER_COUNT):
        raise RuntimeError("frozen QAP has wrong shape")
    result = q2_v4.shell_coupled_qap_statistics(geometries, outcomes, permutations)
    metric_order = [str(value) for value in result["metric_order"]]
    null = np.asarray(result["null"], dtype=np.float64)
    observed = {name: float(result["observed"][name]) for name in metric_order}
    summary = {
        "maps": int(len(permutations)),
        "identity_first": bool(np.array_equal(permutations[0], np.arange(CONTROLLER_COUNT))),
        "unique_maps": int(len({row.tobytes() for row in permutations})),
        "metric_order": metric_order,
        "observed": observed,
        "raw_p": {
            name: float(np.mean(null[:, metric_order.index(name)] >= observed[name]))
            for name in metric_order
        },
        "maxT_adjusted_p": {name: float(result["maxT_adjusted_p"][name]) for name in metric_order},
        "multiplicity": "single-step maxT across A0/A1/A2",
    }
    return summary, {name: null[:, metric_order.index(name)] for name in metric_order}


def leave_one_out(
    geometries: dict[str, dict[str, np.ndarray]], outcomes: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for metric in geometries:
        values: list[dict[str, float]] = []
        for dropped in range(CONTROLLER_COUNT):
            keep = np.arange(CONTROLLER_COUNT) != dropped
            upper = np.triu_indices(int(np.sum(keep)), 1)
            shell = [
                q2_v4.spearman(
                    geometries[metric][name][np.ix_(keep, keep)][upper],
                    outcomes[name][np.ix_(keep, keep)][upper],
                )
                for name in ("MEDIUM", "STRONG")
            ]
            values.append(
                {
                    "dropped_index": dropped,
                    "MEDIUM": float(shell[0]),
                    "STRONG": float(shell[1]),
                    "aggregate": float(np.mean(shell)),
                }
            )
        result[metric] = {
            "values": values,
            "all_sign_stable": bool(
                all(value["MEDIUM"] > 0 and value["STRONG"] > 0 for value in values)
            ),
        }
    return result


def g3_qap(qap_null: dict[str, np.ndarray], qap_observed: dict[str, float]) -> dict[str, Any]:
    a0 = qap_null["A0"]
    a1 = qap_null["A1"]
    a2 = qap_null["A2"]
    null = np.column_stack((a2 - a0, a2 - a1))
    observed = np.asarray(
        (qap_observed["A2"] - qap_observed["A0"], qap_observed["A2"] - qap_observed["A1"])
    )
    max_null = np.max(null, axis=1)
    p = [float(np.mean(max_null >= value)) for value in observed]
    return {
        "observed": {"A2_minus_A0": float(observed[0]), "A2_minus_A1": float(observed[1])},
        "maxT_superiority_p": {"A2_minus_A0": p[0], "A2_minus_A1": p[1]},
        "maps": int(len(null)),
        "multiplicity": "single-step maxT across the two frozen superiority contrasts",
    }


def radial_analysis(
    errors_by_condition: dict[str, np.ndarray], controller_ids: list[str]
) -> dict[str, Any]:
    baseline = errors_by_condition[BASELINE]
    med_values = np.stack([errors_by_condition[f"{cid}_MEDIUM"] for cid in controller_ids], axis=0)
    strong_values = np.stack(
        [errors_by_condition[f"{cid}_STRONG"] for cid in controller_ids], axis=0
    )

    def pair_distance(condition: np.ndarray, *, shape: bool) -> np.ndarray:
        values = np.stack([baseline, condition], axis=0)
        distance = q2_v4.blind_spot_shape_matrices(values)
        return distance["shape_item_population"][0, 1] if shape else distance["total"][0, 1]

    def paired_distance_sample(
        baseline_sample: np.ndarray, condition_sample: np.ndarray, *, shape: bool
    ) -> float:
        delta = baseline_sample - condition_sample
        total = float(np.mean(delta[:, 0] * delta[:, 1]))
        if not shape:
            return total
        mean_product = float(np.mean(delta[:, 0]) * np.mean(delta[:, 1]))
        return float((total - mean_product) * (len(delta) / (len(delta) - 1.0)))

    # Preserve the original V4 swap construction, generalized only to the
    # frozen K=31 controller bank; it is label-free and runs after completion.
    seed = int.from_bytes(
        hashlib.sha256(f"Q2-V4-RADIAL-SWAPS-V1|{PRELOCK}".encode()).digest()[:16], "big"
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    swaps = np.zeros((RADIAL_MAPS, CONTROLLER_COUNT), dtype=np.uint8)
    seen = {swaps[0].tobytes()}
    index = 1
    while index < RADIAL_MAPS:
        candidate = rng.integers(0, 2, size=CONTROLLER_COUNT, dtype=np.uint8)
        if candidate.tobytes() not in seen:
            seen.add(candidate.tobytes())
            swaps[index] = candidate
            index += 1
    result: dict[str, Any] = {
        "maps": RADIAL_MAPS,
        "unique_maps": len(seen),
        "seed": str(seed),
        "identity_first": True,
    }
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    for name, use_shape in (("R_shape", True), ("R_total", False)):
        med = np.asarray(
            [pair_distance(med_values[index], shape=use_shape) for index in range(CONTROLLER_COUNT)]
        )
        strong = np.asarray(
            [
                pair_distance(strong_values[index], shape=use_shape)
                for index in range(CONTROLLER_COUNT)
            ]
        )
        observed = strong - med
        null_medians = np.median((1.0 - 2.0 * swaps) * observed[None, :], axis=1)
        positive = int(np.sum(observed > 0.0))
        bootstrap_medians: list[float] = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            indices = bootstrap_rng.integers(0, N, size=N)
            baseline_sample = baseline[indices]
            med_sample = np.asarray(
                [
                    paired_distance_sample(
                        baseline_sample, med_values[index][indices], shape=use_shape
                    )
                    for index in range(CONTROLLER_COUNT)
                ],
                dtype=np.float64,
            )
            strong_sample = np.asarray(
                [
                    paired_distance_sample(
                        baseline_sample, strong_values[index][indices], shape=use_shape
                    )
                    for index in range(CONTROLLER_COUNT)
                ],
                dtype=np.float64,
            )
            bootstrap_medians.append(float(np.median(strong_sample - med_sample)))
        bootstrap_q025 = float(np.quantile(bootstrap_medians, 0.025))
        permutation_p = float(np.mean(null_medians >= np.median(observed)))
        result[name] = {
            "values": observed.tolist(),
            "median": float(np.median(observed)),
            "positive_directions": positive,
            "permutation_p": permutation_p,
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "median_estimate": float(np.median(bootstrap_medians)),
                "q025": bootstrap_q025,
                "q975": float(np.quantile(bootstrap_medians, 0.975)),
            },
            "positive_gate": bool(
                np.median(observed) > 0
                and positive >= 22
                and permutation_p <= 0.05
                and bootstrap_q025 > 0.0
            ),
        }
        result[name]["classification"] = (
            "RS+" if use_shape and result[name]["positive_gate"] else None
        )
        if not use_shape:
            result[name]["classification"] = "RT+" if result[name]["positive_gate"] else "RT-"
        elif result[name]["classification"] is None:
            result[name]["classification"] = "RS-"
    return result


def analyze() -> None:
    if not COMPLETENESS.is_file() or not RAW_SEAL.is_file():
        raise RuntimeError("raw campaign must be sealed before semantic analysis")
    completeness = read_json(COMPLETENESS)
    if completeness.get("status") != "COMPLETE_RAW_UNSCORED_CAMPAIGN":
        raise RuntimeError("invalid campaign completeness seal")
    if completeness.get("journal_sha256") != sha256_file(JOURNAL):
        raise RuntimeError("raw journal changed after completeness seal")
    if RAW_SEAL.is_file():
        raw_seal = read_json(RAW_SEAL)
        if raw_seal.get("journal", {}).get("sha256") != sha256_file(JOURNAL):
            raise RuntimeError("raw journal changed after raw-data seal")
        if raw_seal.get("campaign_completeness", {}).get("sha256") != sha256_file(COMPLETENESS):
            raise RuntimeError("campaign completeness seal changed after raw-data seal")
    rows, _identity, schedule = validate_complete_campaign()
    item_ids, items = load_panel()
    _schedule_rows, conditions = load_schedule()
    if [str(row["item_id"]) for row in schedule[:N]] and set(item_ids) != {
        str(row["item_id"]) for row in schedule
    }:
        raise RuntimeError("panel/schedule item mismatch")
    parsed = write_scores(rows, items)
    summary_rows, summaries = condition_summary(rows, parsed, conditions)
    write_csv(EXECUTION_REVIEW / "CONDITION_SUMMARY.csv", summary_rows)
    errors = error_arrays(item_ids, conditions, parsed)
    controller_ids = [
        condition
        for condition in conditions
        if condition != BASELINE and condition.endswith("_MEDIUM")
    ]
    controller_ids = [condition.removesuffix("_MEDIUM") for condition in controller_ids]
    if len(controller_ids) != CONTROLLER_COUNT or conditions != [
        BASELINE,
        *(f"{cid}_{shell}" for cid in controller_ids for shell in ("MEDIUM", "STRONG")),
    ]:
        # The order is frozen in the schedule; only the bank identity is used
        # for matrix indexing.  This check catches accidental bank changes but
        # does not impose a new execution order.
        expected = {
            BASELINE,
            *(f"{cid}_{shell}" for cid in controller_ids for shell in ("MEDIUM", "STRONG")),
        }
        if set(conditions) != expected:
            raise RuntimeError("condition/controller contract mismatch")
    estimands: dict[str, Any] = {}
    baseline_errors = errors[BASELINE]
    for condition in conditions[1:]:
        estimands[condition] = pair_estimand(baseline_errors, errors[condition])
    baseline_b00 = float(np.mean(baseline_errors[:, 0] * baseline_errors[:, 1]))
    estimands[BASELINE] = {
        "accuracy": summaries[BASELINE]["accuracy"],
        "B00": baseline_b00,
        "O00": 1.0 - baseline_b00,
        "baseline_resampling_gain": (1.0 - baseline_b00) - summaries[BASELINE]["accuracy"],
    }
    random_summary: dict[str, Any] = {}
    geometries = metric_arrays()
    outcome_shape, outcome_total, radial_shape, radial_total = semantic_distance_arrays(
        errors, controller_ids
    )
    del outcome_total, radial_shape, radial_total
    qap, qap_null = qap_summary(
        {name: geometries[name] for name in ("A0", "A1", "A2")}, outcome_shape
    )
    g3_qap_result = g3_qap(qap_null, qap["observed"])
    loo = leave_one_out({name: geometries[name] for name in ("A0", "A1", "A2")}, outcome_shape)
    bootstrap_result = bootstrap(errors, item_ids, controller_ids, geometries)
    write_json(EXECUTION_REVIEW / "BOOTSTRAP_INTERVALS.json", bootstrap_result)
    write_json(EXECUTION_REVIEW / "RADIAL_RESULTS.json", radial_analysis(errors, controller_ids))

    metric_results: dict[str, Any] = {}
    for metric in ("A0", "A1", "A2"):
        shell_rho = {
            shell: float(
                q2_v4.spearman(
                    geometries[metric][shell][np.triu_indices(CONTROLLER_COUNT, 1)],
                    outcome_shape[shell][np.triu_indices(CONTROLLER_COUNT, 1)],
                )
            )
            for shell in ("MEDIUM", "STRONG")
        }
        aggregate = float(np.mean(list(shell_rho.values())))
        metric_results[metric] = {
            "shell_rho": shell_rho,
            "aggregate_rho": aggregate,
            "qap": {
                "raw_p": qap["raw_p"][metric],
                "maxT_adjusted_p": qap["maxT_adjusted_p"][metric],
            },
            "bootstrap": bootstrap_result[metric],
            "leave_one_controller_out": loo[metric],
            "qualifies": bool(
                shell_rho["MEDIUM"] > 0
                and shell_rho["STRONG"] > 0
                and aggregate >= 0.2
                and loo[metric]["all_sign_stable"]
                and bootstrap_result[metric]["q025"] > 0
                and qap["maxT_adjusted_p"][metric] <= 0.05
            ),
        }

    random_conditions = [condition for condition in conditions if condition != BASELINE]
    for field in ("G", "C", "D", "rescue", "damage"):
        values = [float(estimands[condition][field]) for condition in random_conditions]
        random_summary[field] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    accuracy_change = {
        condition: summaries[condition]["accuracy"] - summaries[BASELINE]["accuracy"]
        for condition in random_conditions
    }
    random_summary["accuracy_change"] = {
        "mean": float(np.mean(list(accuracy_change.values()))),
        "median": float(np.median(list(accuracy_change.values()))),
        "min": float(np.min(list(accuracy_change.values()))),
        "max": float(np.max(list(accuracy_change.values()))),
    }

    a2_a0 = bootstrap_result["A2_minus_A0"]
    a2_a1 = bootstrap_result["A2_minus_A1"]
    g3 = bool(
        metric_results["A2"]["qualifies"]
        and g3_qap_result["observed"]["A2_minus_A0"] >= 0.10
        and g3_qap_result["observed"]["A2_minus_A1"] >= 0.10
        and a2_a0["q025"] > 0
        and a2_a1["q025"] > 0
        and g3_qap_result["maxT_superiority_p"]["A2_minus_A0"] <= 0.05
        and g3_qap_result["maxT_superiority_p"]["A2_minus_A1"] <= 0.05
    )
    if g3:
        classification = "Q2_V4_1_G3"
    elif metric_results["A2"]["qualifies"]:
        classification = "Q2_V4_1_G2"
    elif metric_results["A0"]["qualifies"] or metric_results["A1"]["qualifies"]:
        classification = "Q2_V4_1_G1"
    else:
        classification = "Q2_V4_1_G0"

    estimands_payload = {
        "classification": classification,
        "parser_version": PARSER_VERSION,
        "conditions": conditions,
        "controller_order": controller_ids,
        "summaries": summaries,
        "estimands": estimands,
        "random_summary": random_summary,
        "semantic_distance": {
            "D_shape_superpopulation": {
                shell: outcome_shape[shell].tolist() for shell in outcome_shape
            }
        },
        "metrics": metric_results,
        "qap": qap,
        "g3_superiority": g3_qap_result,
        "g3_power_characterization_is_planning_only": True,
        "radial": "see RADIAL_RESULTS.json; independent secondary result",
        "historical_v4_classification_preserved": "Q2_V4_SAFE_BANK_INSUFFICIENT",
        "q2_prior_state": "UNTESTED",
        "q3": "NOT_RUN",
    }
    write_json(EXECUTION_REVIEW / "ESTIMANDS.json", estimands_payload)
    contributions: list[dict[str, Any]] = []
    for condition in conditions[1:]:
        for item_id, value in zip(
            item_ids, item_contributions(baseline_errors, errors[condition]), strict=True
        ):
            contributions.append({"item_id": item_id, "condition": condition, **value})
    write_csv(EXECUTION_REVIEW / "ITEM_CONTRIBUTIONS.csv", contributions)
    loo_rows: list[dict[str, Any]] = []
    for condition in conditions[1:]:
        for item_id, contribution in zip(
            item_ids, item_contributions(baseline_errors, errors[condition]), strict=True
        ):
            loo_rows.append({"item_id": item_id, "condition": condition, **contribution})
    write_csv(EXECUTION_REVIEW / "LOO_SENSITIVITY.csv", loo_rows)
    total_elapsed = float(sum(float(row.get("elapsed_seconds", 0.0)) for row in rows))
    write_json(
        EXECUTION_REVIEW / "COST.json",
        {
            "scientific_trajectories": len(rows),
            "summed_generation_seconds": total_elapsed,
            "summed_generation_hours": total_elapsed / 3600.0,
            "cost_status": "record_remote_lifecycle_after_collection",
            "q2_v4_1_hard_ceiling": "not applicable",
        },
    )
    report = (
        f"# Q2 V4.1 Semantic Execution\n\n"
        f"Primary frozen relational classification: `{classification}`.\n\n"
        "The Q2 V4.1 semantic panel was executed only after the principal "
        "authorization, with 31 frozen controllers, two shells, N=300 items, "
        "and two independent rollouts. The original V4 classification remains "
        "`Q2_V4_SAFE_BANK_INSUFFICIENT`; this is a distinct V4.1 result.\n\n"
        "## Frozen endpoint\n\n"
        "For each controller pair, the analysis uses "
        "`D_shape_superpopulation = N/(N-1) * (D_total - m0*m1)` with binary "
        "error `e=1` for every non-correct outcome. Negative finite-sample "
        "estimates are retained.\n\n"
        "## Classification\n\n"
        "A0/A1/A2 qualification inputs and the G3 superiority contrasts are in "
        "`ESTIMANDS.json`; the 50,000-map QAP and 10,000 item-cluster bootstrap "
        "are frozen and recorded there. G3 planning power remains planning-only "
        "and was not used to modify the decision rule.\n\n"
        "## Firewall\n\n"
        "No Q3 was run. No controller, shell, item, seed, endpoint, metric, "
        "QAP map, or threshold was changed after opening.\n"
    )
    (EXECUTION_REVIEW / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": "q2-v4.1-semantic-analysis-manifest-v1",
        "analysis_commit": git_head(),
        "raw_seal_sha256": sha256_file(RAW_SEAL),
        "journal_sha256": sha256_file(JOURNAL),
        "scores_sha256": sha256_file(SCORES),
        "schedule_sha256": sha256_file(SCHEDULE),
        "parser_version": PARSER_VERSION,
        "classification": classification,
        "semantic_outcomes_now_exist": True,
    }
    write_json(EXECUTION_REVIEW / "manifest_hashes.json", manifest)
    print(
        json.dumps(
            {"status": "ANALYSIS_COMPLETE", "classification": classification}, sort_keys=True
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("seal", "analyze"), required=True)
    args = parser.parse_args()
    if args.mode == "seal":
        seal()
    else:
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
