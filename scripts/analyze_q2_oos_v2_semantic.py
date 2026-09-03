#!/usr/bin/env python3
"""Run the frozen post-collection Q2 OOS V2 analysis.

This module is intentionally separate from the semantic collector.  It can
only run against a complete, hash-pinned raw seal and applies the already
frozen external-semantic-v3 scorer and Q2 OOS V2 estimands.  Raw outputs and
the complete scored private dataset stay in the caller-selected private
artifact directory; the release directory receives aggregate, non-content
provenance only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from epistemic_geometry.experiments.heterogeneity_robust import (  # noqa: E402
    exact_positive_sign_test,
    node_jackknife_test,
    studentized_mean_test,
)
from epistemic_geometry.experiments.q2_oos_fresh_controller import (  # noqa: E402
    QAP_MAPS,
    cross_block_shape,
    fresh_row_permutations,
    row_permutation_test,
    spearman_flat,
)

OOS_REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout"
PREDICTION_LOCK = OOS_REVIEW / "PREDICTION_LOCK.json"
SCHEDULE = OOS_REVIEW / "FUTURE_SEMANTIC_SCHEDULE.json"
PANEL = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"
MATRICES = OOS_REVIEW / "PREDICTION_MATRICES.npz"
MATRIX_METADATA = OOS_REVIEW / "PREDICTION_MATRIX_METADATA.json"
COLLECTOR = ROOT / "scripts/execute_q2_oos_v2_semantic.py"
PARSER_SOURCE = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"

EXPECTED_JOURNAL_SHA256 = "24fdd1c818c6e507f2e1999ce6e5da380405bc533af60723da01c1ec2bd66a40"
EXPECTED_HISTORICAL_SCORES_SHA256 = (
    "a6a9f4b419d4531716337d2277688063d5655167a5d7b1a9bd85b34217f8a33f"
)
EXPECTED_SCHEDULE_SHA256 = "dac5c284b90c726016968f31d25200a362c42d96f63b63d730665f3f47e85ec5"
EXPECTED_PANEL_SHA256 = "c127cf3594e8ea849dbd038492606b3afaaac406feb4146188769c04d6691187"
EXPECTED_SELECTED_BANK_SHA256 = "9a544b4ec6d43ec1c3530feb963cd0340db516e82f91a40c2624300483e2e0fd"
EXPECTED_MATRIX_ARCHIVE_SHA256 = "b4ec00985e750c5bb8fd7fd49228267ec576bf6c2ad2ac3984f6f2390d927703"
EXPECTED_MATRIX_METADATA_SHA256 = "39ceaa889abc30a6740ab60ef4bdd7c24197b6ea2cc5189245bf365b9edd3b06"
EXPECTED_PREDICTION_LOCK_SHA256 = "825d6e3536b51a31956cbd5c9e75bedfed38f9e3df5da05a4452a5681f65f9bb"
EXPECTED_INFERENCE_LOCK_PARENT = "170dd50925c35e32a2439576f901bab1cf31eb7d"
EXPECTED_CODE_COMMIT = "71fff6d075c1e2bca0f701109b66a97bda9ecaec"
EXPECTED_ENVIRONMENT = "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
EXPECTED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_CONTROLLER_COUNT = 16
EXPECTED_REFERENCE_COUNT = 31
EXPECTED_ITEM_COUNT = 300
EXPECTED_ROLLOUTS = (0, 1)
SHELLS = ("MEDIUM", "STRONG")
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 8_570_345_065_466_006_396
CLUSTER_BOOTSTRAP_RESAMPLES = 10_000
CLUSTER_BOOTSTRAP_SEED = 118_613_282_769_869_400_430_583_486_296_130_981_279
# The historical score path is private and must be supplied by the execution
# environment; no infrastructure path is embedded in the repository.
HISTORICAL_SCORE_PATH_DEFAULT = Path("SEMANTIC_SCORES.jsonl")
FRESH_IDS = (
    "Q2_OOS_V2_DIRECTION_01",
    "Q2_OOS_V2_DIRECTION_02",
    "Q2_OOS_V2_DIRECTION_03",
    "Q2_OOS_V2_DIRECTION_04",
    "Q2_OOS_V2_DIRECTION_05",
    "Q2_OOS_V2_DIRECTION_06",
    "Q2_OOS_V2_DIRECTION_07",
    "Q2_OOS_V2_DIRECTION_08",
    "Q2_OOS_V2_DIRECTION_09",
    "Q2_OOS_V2_DIRECTION_11",
    "Q2_OOS_V2_DIRECTION_13",
    "Q2_OOS_V2_DIRECTION_14",
    "Q2_OOS_V2_DIRECTION_15",
    "Q2_OOS_V2_DIRECTION_16",
    "Q2_OOS_V2_DIRECTION_17",
    "Q2_OOS_V2_DIRECTION_18",
)
REFERENCE_IDS = (
    "V4_DIRECTION_00",
    "V4_DIRECTION_01",
    "V4_DIRECTION_02",
    "V4_DIRECTION_03",
    "V4_DIRECTION_04",
    "V4_DIRECTION_06",
    "V4_DIRECTION_07",
    "V4_DIRECTION_08",
    "V4_DIRECTION_09",
    "V4_DIRECTION_10",
    "V4_DIRECTION_11",
    "V4_DIRECTION_13",
    "V4_DIRECTION_15",
    "V4_DIRECTION_17",
    "V4_DIRECTION_18",
    "V4_DIRECTION_19",
    "V4_DIRECTION_20",
    "V4_DIRECTION_22",
    "V4_DIRECTION_23",
    "V4_DIRECTION_24",
    "V4_DIRECTION_26",
    "V4_DIRECTION_28",
    "V4_DIRECTION_29",
    "V4_DIRECTION_30",
    "V4_DIRECTION_31",
    "V4_DIRECTION_32",
    "V4_DIRECTION_33",
    "V4_DIRECTION_34",
    "V4_DIRECTION_35",
    "V4_DIRECTION_37",
    "V4_DIRECTION_39",
)
KEY_FIELDS = ("item_id", "condition", "rollout_index")
TERMINAL_FAILURES = {
    "EXTREME_MECHANICAL_REPETITION_V1": "REPETITION_STOP",
    "max_new_tokens": "HARD_CAP",
    "model_runtime_error": "RUNTIME_ERROR",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def key_of(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])


def load_panel() -> tuple[list[str], dict[str, dict[str, Any]]]:
    if sha256_file(PANEL) != EXPECTED_PANEL_SHA256:
        raise RuntimeError("frozen panel hash mismatch")
    payload = read_json(PANEL)
    items = {str(row["item_id"]): row for row in payload["items"]}
    item_ids = [str(value) for value in payload["item_ids"]]
    if payload.get("item_count") != EXPECTED_ITEM_COUNT or len(items) != EXPECTED_ITEM_COUNT:
        raise RuntimeError("frozen panel count mismatch")
    if item_ids != [str(row["item_id"]) for row in payload["items"]]:
        raise RuntimeError("frozen panel order mismatch")
    return item_ids, items


def load_schedule() -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    if sha256_file(SCHEDULE) != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("frozen semantic schedule hash mismatch")
    payload = read_json(SCHEDULE)
    rows = list(payload["rows"])
    if payload.get("row_count") != 19_200 or len(rows) != 19_200:
        raise RuntimeError("frozen semantic schedule count mismatch")
    expected_conditions = {f"{controller}_{shell}" for controller in FRESH_IDS for shell in SHELLS}
    if payload.get("selected_controller_order") != list(FRESH_IDS):
        raise RuntimeError("frozen selected-controller order mismatch")
    observed_conditions = {str(row["condition"]) for row in rows}
    if observed_conditions != expected_conditions:
        raise RuntimeError("frozen semantic condition set mismatch")
    by_key = {key_of(row): row for row in rows}
    if len(by_key) != len(rows) or len({int(row["seed"]) for row in rows}) != len(rows):
        raise RuntimeError("frozen schedule duplicate key or seed")
    if set(by_key) != {
        (item, condition, rollout)
        for item in [str(row["item_id"]) for row in read_json(PANEL)["items"]]
        for condition in expected_conditions
        for rollout in EXPECTED_ROLLOUTS
    }:
        raise RuntimeError("frozen schedule coverage mismatch")
    return rows, by_key


def load_raw_rows(
    raw_dir: Path,
    schedule_by_key: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    journal = raw_dir / "journal.jsonl"
    seal_path = raw_dir / "COLLECTION_COMPLETE_SEAL.json"
    if not journal.is_file() or not seal_path.is_file():
        raise RuntimeError("complete raw collection seal is required")
    observed_hash = sha256_file(journal)
    if observed_hash != EXPECTED_JOURNAL_SHA256:
        raise RuntimeError("raw journal SHA-256 differs from accepted seal")
    seal = read_json(seal_path)
    required = {
        "status": "COLLECTION_COMPLETE_RAW_UNSCORED",
        "completed_rows": 19_200,
        "expected_rows": 19_200,
        "missing_rows": 0,
        "unexpected_rows": 0,
        "duplicate_keys": 0,
        "replacements": 0,
        "retry_row_count": 0,
        "runtime_error_count": 0,
        "raw_journal_sha256": EXPECTED_JOURNAL_SHA256,
        "raw_journal_bytes": 127_968_010,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "selected_controller_bank_sha256": EXPECTED_SELECTED_BANK_SHA256,
        "semantic_scoring": "NOT_RUN",
        "correctness_inspected": False,
        "semantic_outcomes": 0,
        "code_commit": EXPECTED_CODE_COMMIT,
        "prediction_lock_parent_head": EXPECTED_INFERENCE_LOCK_PARENT,
    }
    for key, expected in required.items():
        if seal.get(key) != expected:
            raise RuntimeError(f"raw seal mismatch: {key}")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str, int]] = set()
    identity: dict[str, Any] | None = None
    with journal.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            wrapper = json.loads(line)
            if wrapper.get("version") != "research-os-jsonl-v1":
                raise RuntimeError(f"raw wrapper version mismatch at {line_number}")
            if wrapper.get("key_fields") != list(KEY_FIELDS):
                raise RuntimeError(f"raw key contract mismatch at {line_number}")
            row = wrapper.get("row")
            if not isinstance(row, dict):
                raise RuntimeError(f"raw row is not an object at {line_number}")
            key = key_of(row)
            if tuple(wrapper.get("key", ())) != key or key in keys:
                raise RuntimeError(f"raw logical-key integrity failure at {line_number}")
            expected = schedule_by_key.get(key)
            if expected is None:
                raise RuntimeError("unexpected raw logical key")
            keys.add(key)
            if identity is None:
                identity = dict(wrapper.get("identity", {}))
            if wrapper.get("identity") != identity:
                raise RuntimeError("raw journal identity changed")
            for field in (
                "seed",
                "candidate_id",
                "shell",
                "alpha",
                "layer",
                "duration",
                "prompt_sha256",
                "controller_vector_hash",
            ):
                if row.get(field) != expected.get(field) and field != "alpha":
                    raise RuntimeError(f"raw schedule provenance mismatch: {field}")
            if field == "alpha" and not math.isclose(
                float(row["alpha"]), float(expected["alpha"]), rel_tol=0.0, abs_tol=0.0
            ):
                raise RuntimeError("raw schedule provenance mismatch: alpha")
            if int(row.get("seed")) != int(expected["seed"]):
                raise RuntimeError("raw seed mismatch")
            if row.get("semantic_scoring") != "DEFERRED_UNTIL_COMPLETE":
                raise RuntimeError("raw scoring marker was changed")
            if row.get("code_commit") != EXPECTED_CODE_COMMIT:
                raise RuntimeError("raw code commit mismatch")
            if (
                row.get("model_revision") != EXPECTED_MODEL_REVISION
                or row.get("tokenizer_revision") != EXPECTED_MODEL_REVISION
            ):
                raise RuntimeError("raw model revision mismatch")
            rows.append(row)
    if len(rows) != 19_200 or keys != set(schedule_by_key):
        raise RuntimeError("raw schedule coverage is not exact")
    if identity is None:
        raise RuntimeError("raw journal identity is missing")
    return rows, seal, identity


def classify_row(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    terminal = str(row.get("terminal_reason", ""))
    if terminal in TERMINAL_FAILURES:
        result = {
            "commitment_valid": False,
            "semantic_evaluable": False,
            "correct": False,
            "value_type": None,
            "canonical_value": None,
            "parsed_answer": None,
            "failure_reason": terminal,
            "status": TERMINAL_FAILURES[terminal],
        }
    else:
        parsed = evaluate_external_answer_v3(
            str(row.get("raw_output", "")),
            str(item["reference_answer"]),
            truncated=bool(row.get("truncated", False)),
            runtime_error=bool(row.get("runtime_error")),
        )
        if parsed.correct:
            status = "VALID_CORRECT"
        elif parsed.commitment_valid and parsed.semantic_evaluable:
            status = "VALID_WRONG"
        elif parsed.failure_reason == "truncated or unclosed response":
            status = "TRUNCATED"
        elif parsed.failure_reason == "runtime error":
            status = "RUNTIME_ERROR"
        else:
            status = "INVALID_FORMAT"
        result = {
            "commitment_valid": bool(parsed.commitment_valid),
            "semantic_evaluable": bool(parsed.semantic_evaluable),
            "correct": bool(parsed.correct),
            "value_type": parsed.value_type,
            "canonical_value": parsed.canonical_value,
            "parsed_answer": parsed.payload,
            "failure_reason": parsed.failure_reason,
            "status": status,
        }
    return {
        "item_id": str(row["item_id"]),
        "condition": str(row["condition"]),
        "rollout_index": int(row["rollout_index"]),
        **result,
        "generated_token_count": int(row.get("generated_token_count", 0)),
        "truncated": bool(row.get("truncated", False)),
        "raw_output_sha256": hashlib.sha256(str(row.get("raw_output", "")).encode()).hexdigest(),
    }


def score_rows(
    rows: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    items: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    by_key = {key_of(row): row for row in rows}
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    score_path = output_dir / "SEMANTIC_SCORES.jsonl"
    temporary = score_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for schedule_row in schedule:
            key = key_of(schedule_row)
            score = classify_row(by_key[key], items[key[0]])
            scores[key] = score
            handle.write(json.dumps(score, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(score_path)
    return scores


def condition_summaries(
    rows: list[dict[str, Any]], scores: dict[tuple[str, str, int], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append((row, scores[key_of(row)]))
    output: dict[str, dict[str, Any]] = {}
    for condition, pairs in sorted(grouped.items()):
        tokens = np.asarray(
            [int(row["generated_token_count"]) for row, _ in pairs], dtype=np.float64
        )
        counts = Counter(str(score["status"]) for _, score in pairs)
        output[condition] = {
            "rows": len(pairs),
            "commitment_valid": int(sum(bool(score["commitment_valid"]) for _, score in pairs)),
            "commitment_validity": float(
                np.mean([bool(score["commitment_valid"]) for _, score in pairs])
            ),
            "semantic_evaluable": int(sum(bool(score["semantic_evaluable"]) for _, score in pairs)),
            "semantic_evaluability": float(
                np.mean([bool(score["semantic_evaluable"]) for _, score in pairs])
            ),
            "correct": int(sum(bool(score["correct"]) for _, score in pairs)),
            "accuracy": float(np.mean([bool(score["correct"]) for _, score in pairs])),
            "mean_tokens": float(np.mean(tokens)),
            "median_tokens": float(np.median(tokens)),
            "p90_tokens": float(np.quantile(tokens, 0.90)),
            "max_tokens": int(np.max(tokens)),
            "status_counts": dict(sorted(counts.items())),
        }
    return output


def build_error_arrays(
    item_ids: list[str], scores: dict[tuple[str, str, int], dict[str, Any]]
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    errors: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    evaluable: dict[str, np.ndarray] = {}
    correct: dict[str, np.ndarray] = {}
    conditions = [f"{controller}_{shell}" for controller in FRESH_IDS for shell in SHELLS]
    for condition in conditions:
        errors[condition] = np.asarray(
            [
                [
                    int(not scores[(item, condition, rollout)]["correct"])
                    for rollout in EXPECTED_ROLLOUTS
                ]
                for item in item_ids
            ],
            dtype=np.float64,
        )
        valid[condition] = np.asarray(
            [
                [
                    int(scores[(item, condition, rollout)]["commitment_valid"])
                    for rollout in EXPECTED_ROLLOUTS
                ]
                for item in item_ids
            ],
            dtype=np.float64,
        )
        evaluable[condition] = np.asarray(
            [
                [
                    int(scores[(item, condition, rollout)]["semantic_evaluable"])
                    for rollout in EXPECTED_ROLLOUTS
                ]
                for item in item_ids
            ],
            dtype=np.float64,
        )
        correct[condition] = 1.0 - errors[condition]
    return errors, valid, evaluable, correct


def load_reference_errors(
    path: Path, item_ids: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if sha256_file(path) != EXPECTED_HISTORICAL_SCORES_SHA256:
        raise RuntimeError("historical semantic-score hash mismatch")
    item_index = {item: index for index, item in enumerate(item_ids)}
    arrays = {
        shell: np.empty((EXPECTED_REFERENCE_COUNT, EXPECTED_ITEM_COUNT, 2), dtype=np.float64)
        for shell in SHELLS
    }
    seen: set[tuple[str, str, int]] = set()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            condition = str(row["condition"])
            if condition == "BASELINE":
                continue
            if not condition.endswith(("_MEDIUM", "_STRONG")):
                raise RuntimeError("unexpected historical condition")
            shell = "MEDIUM" if condition.endswith("_MEDIUM") else "STRONG"
            controller = condition[: -(len(shell) + 1)]
            if controller not in REFERENCE_IDS:
                raise RuntimeError("historical reference order mismatch")
            key = (str(row["item_id"]), condition, int(row["rollout_index"]))
            if key in seen:
                raise RuntimeError("duplicate historical score key")
            seen.add(key)
            arrays[shell][
                REFERENCE_IDS.index(controller), item_index[key[0]], int(row["rollout_index"])
            ] = int(not bool(row["correct"]))
            count += 1
    expected = EXPECTED_REFERENCE_COUNT * EXPECTED_ITEM_COUNT * 2 * 2
    if count != expected or len(seen) != expected:
        raise RuntimeError("historical reference score coverage mismatch")
    return arrays, {
        "path_sha256": sha256_file(path),
        "rows": count,
        "controller_order": list(REFERENCE_IDS),
    }


def load_prediction_matrices() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if sha256_file(MATRICES) != EXPECTED_MATRIX_ARCHIVE_SHA256:
        raise RuntimeError("prediction matrix archive hash mismatch")
    if sha256_file(MATRIX_METADATA) != EXPECTED_MATRIX_METADATA_SHA256:
        raise RuntimeError("prediction matrix metadata hash mismatch")
    metadata = read_json(MATRIX_METADATA)
    if metadata.get("fresh_controller_order") != list(FRESH_IDS) or metadata.get(
        "reference_controller_order"
    ) != list(REFERENCE_IDS):
        raise RuntimeError("prediction matrix controller order mismatch")
    archive = np.load(MATRICES, allow_pickle=False)
    matrices = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    expected_hashes = metadata["matrix_hashes"]
    if set(matrices) != set(expected_hashes):
        raise RuntimeError("prediction matrix key set mismatch")
    for name, expected in expected_hashes.items():
        if array_hash(matrices[name]) != expected:
            raise RuntimeError(f"prediction matrix array hash mismatch: {name}")
    return matrices, metadata


def direct_cross_total(fresh: np.ndarray, reference: np.ndarray) -> np.ndarray:
    d0 = fresh[:, None, :, 0] - reference[None, :, :, 0]
    d1 = fresh[:, None, :, 1] - reference[None, :, :, 1]
    return np.mean(d0 * d1, axis=2)


def compute_distances(
    errors: dict[str, np.ndarray], reference_errors: dict[str, np.ndarray]
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    shape_ref: dict[str, np.ndarray] = {}
    total_ref: dict[str, np.ndarray] = {}
    shape_ff: dict[str, np.ndarray] = {}
    total_ff: dict[str, np.ndarray] = {}
    for shell in SHELLS:
        fresh = np.stack([errors[f"{controller}_{shell}"] for controller in FRESH_IDS], axis=0)
        reference = reference_errors[shell]
        shape_ref[shell] = cross_block_shape(fresh, reference)
        total_ref[shell] = direct_cross_total(fresh, reference)
        ff = q2_v4.blind_spot_shape_matrices(fresh)
        shape_ff[shell] = ff["shape_item_population"]
        total_ff[shell] = ff["total"]
    return shape_ref, total_ref, shape_ff, total_ff


def row_associations(
    geometry: dict[str, np.ndarray], outcomes: dict[str, np.ndarray]
) -> np.ndarray:
    values = []
    for index in range(EXPECTED_CONTROLLER_COUNT):
        shell_values = [
            spearman_flat(geometry[shell][index], outcomes[shell][index]) for shell in SHELLS
        ]
        values.append(
            float(np.mean(shell_values)) if all(np.isfinite(shell_values)) else float("nan")
        )
    return np.asarray(values, dtype=np.float64)


def global_association(
    geometry: dict[str, np.ndarray], outcomes: dict[str, np.ndarray]
) -> dict[str, Any]:
    shell = {name: float(spearman_flat(geometry[name], outcomes[name])) for name in SHELLS}
    return {"shell": shell, "equal_shell_mean": float(np.mean(list(shell.values())))}


def total_distance_summary(
    geometry: dict[str, np.ndarray], errors: dict[str, np.ndarray]
) -> dict[str, Any]:
    values = []
    for shell in SHELLS:
        fresh = np.stack([errors[f"{controller}_{shell}"] for controller in FRESH_IDS], axis=0)
        d = q2_v4.blind_spot_shape_matrices(fresh)["total"]
        upper = np.triu_indices(EXPECTED_CONTROLLER_COUNT, 1)
        values.append(float(spearman_flat(geometry[shell][upper], d[upper])))
    return {
        "shell": dict(zip(SHELLS, values, strict=True)),
        "equal_shell_mean": float(np.mean(values)),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda pair: pair[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted[name] = running
    return adjusted


def controller_cluster_bootstrap(
    geometry: dict[str, np.ndarray], outcomes: dict[str, np.ndarray], estimate: float
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64DXSM(CLUSTER_BOOTSTRAP_SEED))
    values = np.empty(CLUSTER_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(CLUSTER_BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, EXPECTED_CONTROLLER_COUNT, size=EXPECTED_CONTROLLER_COUNT)
        values[index] = float(
            np.mean(
                [
                    spearman_flat(geometry[shell][sample], outcomes[shell][sample])
                    for shell in SHELLS
                ]
            )
        )
    q025, q975 = np.quantile(values, [0.025, 0.975])
    return {
        "resamples": CLUSTER_BOOTSTRAP_RESAMPLES,
        "seed": str(CLUSTER_BOOTSTRAP_SEED),
        "estimate": estimate,
        "percentile_95": [float(q025), float(q975)],
        "basic_95": [float(2 * estimate - q975), float(2 * estimate - q025)],
    }


def item_bootstrap(
    fresh_errors: dict[str, np.ndarray],
    reference_errors: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED))
    global_values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    median_values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    differences: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for shell in SHELLS:
        fresh = fresh_errors[shell]
        reference = reference_errors[shell]
        d0 = fresh[:, None, :, 0] - reference[None, :, :, 0]
        d1 = fresh[:, None, :, 1] - reference[None, :, :, 1]
        differences[shell] = (d0, d1, d0 * d1)

    batch_size = 16
    written = 0
    while written < BOOTSTRAP_RESAMPLES:
        count = min(batch_size, BOOTSTRAP_RESAMPLES - written)
        item_indices = rng.integers(0, EXPECTED_ITEM_COUNT, size=(count, EXPECTED_ITEM_COUNT))
        shape_batch: dict[str, np.ndarray] = {}
        for shell in SHELLS:
            d0, d1, product = differences[shell]
            sampled_d0 = np.take(d0, item_indices, axis=2).transpose(2, 0, 1, 3)
            sampled_d1 = np.take(d1, item_indices, axis=2).transpose(2, 0, 1, 3)
            sampled_product = np.take(product, item_indices, axis=2).transpose(2, 0, 1, 3)
            panel = sampled_product.mean(axis=-1)
            mean_product = sampled_d0.mean(axis=-1) * sampled_d1.mean(axis=-1)
            shape_batch[shell] = (panel - mean_product) * (
                EXPECTED_ITEM_COUNT / (EXPECTED_ITEM_COUNT - 1.0)
            )
        for offset in range(count):
            shape = {shell: shape_batch[shell][offset] for shell in SHELLS}
            rows = row_associations(geometry, shape)
            shell_values = [spearman_flat(geometry[shell], shape[shell]) for shell in SHELLS]
            global_values[written + offset] = float(np.mean(shell_values))
            median_values[written + offset] = float(np.median(rows))
        written += count
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": str(BOOTSTRAP_SEED),
        "global_equal_shell_mean": {
            "q025": float(np.quantile(global_values, 0.025)),
            "q50": float(np.quantile(global_values, 0.50)),
            "q975": float(np.quantile(global_values, 0.975)),
        },
        "median_row_association": {
            "q025": float(np.quantile(median_values, 0.025)),
            "q50": float(np.quantile(median_values, 0.50)),
            "q975": float(np.quantile(median_values, 0.975)),
        },
        "role": "UNCERTAINTY_AND_SENSITIVITY_NOT_PRIMARY_SIGN_TEST_REPLACEMENT",
    }


def lofo(geometry: dict[str, np.ndarray], outcomes: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for omitted in range(EXPECTED_CONTROLLER_COUNT):
        keep = np.arange(EXPECTED_CONTROLLER_COUNT) != omitted
        kept_geometry = {shell: geometry[shell][keep] for shell in SHELLS}
        kept_outcomes = {shell: outcomes[shell][keep] for shell in SHELLS}
        associations = row_associations(kept_geometry, kept_outcomes)
        sign = exact_positive_sign_test(associations)
        rows.append(
            {
                "omitted_controller": FRESH_IDS[omitted],
                "median": float(np.median(associations)),
                "mean": float(np.mean(associations)),
                "positive_count": int(sign["positives"]),
                "p_value": float(sign["p_value"]),
                "positive_sign_pass": bool(sign["reject_0_05"]),
            }
        )
    return rows


def safe_metric_result(
    metric: str,
    geometry: dict[str, np.ndarray],
    outcomes: dict[str, np.ndarray],
) -> dict[str, Any]:
    associations = row_associations(geometry, outcomes)
    sign = exact_positive_sign_test(associations)
    return {
        "metric": metric,
        "row_associations": associations.tolist(),
        "row_association_summary": {
            "median": float(np.median(associations)),
            "mean": float(np.mean(associations)),
            **{
                key: int(value)
                for key, value in sign.items()
                if key in ("positives", "zeros", "negatives")
            },
        },
        "exact_sign": sign,
        "global": global_association(geometry, outcomes),
        "studentized_mean": studentized_mean_test(associations),
    }


def primary_classification(associations: np.ndarray) -> tuple[str, dict[str, Any]]:
    sign = exact_positive_sign_test(associations)
    if not bool(sign["finite"] == EXPECTED_CONTROLLER_COUNT):
        return "Q2_OOS_V2_INFERENCE_DEGENERATE", sign
    if float(np.median(associations)) > 0.0 and bool(sign["reject_0_05"]):
        return "Q2_OOS_V2_A0_PASS", sign
    if float(np.median(associations)) > 0.0:
        return "Q2_OOS_V2_ASSOCIATION_INCOMPLETE", sign
    return "Q2_OOS_V2_NO_REPLICATION", sign


def independent_forensic(
    raw_dir: Path,
    item_ids: list[str],
    items: dict[str, dict[str, Any]],
    schedule: list[dict[str, Any]],
    primary_errors: dict[str, np.ndarray],
    primary_shape: dict[str, np.ndarray],
    primary_r: np.ndarray,
    primary_class: str,
) -> dict[str, Any]:
    raw_rows, _seal, _identity = load_raw_rows(raw_dir, {key_of(row): row for row in schedule})
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in raw_rows:
        scores[key_of(row)] = classify_row(row, items[str(row["item_id"])])
    errors, _valid, _evaluable, _correct = build_error_arrays(item_ids, scores)
    reference_errors, _ = load_reference_errors(HISTORICAL_SCORE_PATH_DEFAULT, item_ids)
    shape, _total, _shape_ff, _total_ff = compute_distances(errors, reference_errors)
    geometry, _metadata = load_prediction_matrices()
    forensic_r = row_associations(
        {shell: geometry[f"A0_{shell}_FRESH_REFERENCE"] for shell in SHELLS}, shape
    )
    max_error = max(float(np.max(np.abs(errors[key] - primary_errors[key]))) for key in errors)
    max_shape = max(float(np.max(np.abs(shape[shell] - primary_shape[shell]))) for shell in SHELLS)
    max_r = float(np.max(np.abs(forensic_r - primary_r)))
    forensic_class, forensic_sign = primary_classification(forensic_r)
    return {
        "status": "Q2_OOS_V2_FORENSIC_CLEAN"
        if max(max_error, max_shape, max_r) == 0.0 and forensic_class == primary_class
        else "Q2_OOS_V2_FORENSIC_DISAGREEMENT",
        "raw_journal_reopened": True,
        "max_error_array_difference": max_error,
        "max_dshape_difference": max_shape,
        "max_primary_r_difference": max_r,
        "primary_classification": primary_class,
        "forensic_classification": forensic_class,
        "classification_agreement": forensic_class == primary_class,
        "forensic_sign": forensic_sign,
        "scored_rows_reconstructed": len(scores),
    }


def write_summary_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    fields = [
        "condition",
        "rows",
        "commitment_validity",
        "semantic_evaluability",
        "accuracy",
        "mean_tokens",
        "median_tokens",
        "p90_tokens",
        "max_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, summary in sorted(summaries.items()):
            writer.writerow(
                {"condition": condition, **{field: summary[field] for field in fields[1:]}}
            )


def analyze(
    raw_dir: Path, historical_scores: Path, output_dir: Path, release_dir: Path
) -> dict[str, Any]:
    global HISTORICAL_SCORE_PATH_DEFAULT
    HISTORICAL_SCORE_PATH_DEFAULT = historical_scores
    if sha256_file(PREDICTION_LOCK) != EXPECTED_PREDICTION_LOCK_SHA256:
        raise RuntimeError("prediction lock hash mismatch")
    lock = read_json(PREDICTION_LOCK)
    if lock.get("semantic_execution_authorized") is not False:
        raise RuntimeError("prediction lock authorization state changed")
    item_ids, items = load_panel()
    schedule, schedule_by_key = load_schedule()
    raw_rows, raw_seal, raw_identity = load_raw_rows(raw_dir, schedule_by_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = score_rows(raw_rows, schedule, items, output_dir)
    summaries = condition_summaries(raw_rows, scores)
    write_summary_csv(output_dir / "CONDITION_SUMMARY.csv", summaries)
    errors, valid, evaluable, correct = build_error_arrays(item_ids, scores)
    np.savez_compressed(
        output_dir / "ERROR_ARRAYS.npz",
        **{f"error__{k}": v for k, v in errors.items()},
        **{f"valid__{k}": v for k, v in valid.items()},
        **{f"evaluable__{k}": v for k, v in evaluable.items()},
        **{f"correct__{k}": v for k, v in correct.items()},
    )
    reference_errors, reference_meta = load_reference_errors(historical_scores, item_ids)
    matrices, matrix_meta = load_prediction_matrices()
    shape_ref, total_ref, shape_ff, total_ff = compute_distances(errors, reference_errors)
    np.savez_compressed(
        output_dir / "D_SHAPE.npz",
        **{f"fresh_reference__{k}": v for k, v in shape_ref.items()},
        **{f"fresh_fresh__{k}": v for k, v in shape_ff.items()},
    )
    np.savez_compressed(
        output_dir / "D_TOTAL.npz",
        **{f"fresh_reference__{k}": v for k, v in total_ref.items()},
        **{f"fresh_fresh__{k}": v for k, v in total_ff.items()},
    )
    primary_geometry = {shell: matrices[f"A0_{shell}_FRESH_REFERENCE"] for shell in SHELLS}
    primary_r = row_associations(primary_geometry, shape_ref)
    primary_class, primary_sign = primary_classification(primary_r)
    error_hash = sha256_file(output_dir / "ERROR_ARRAYS.npz")
    dshape_hash = sha256_file(output_dir / "D_SHAPE.npz")
    dtotal_hash = sha256_file(output_dir / "D_TOTAL.npz")
    primary_result = {
        "schema_version": "q2-oos-v2-primary-result-seal-v1",
        "status": "PRIMARY_RESULT_SEALED_BEFORE_SECONDARIES",
        "raw_journal_sha256": EXPECTED_JOURNAL_SHA256,
        "scored_dataset_sha256": sha256_file(output_dir / "SEMANTIC_SCORES.jsonl"),
        "error_arrays_sha256": error_hash,
        "dshape_artifact_sha256": dshape_hash,
        "dtotal_artifact_sha256": dtotal_hash,
        "primary_r_i": primary_r.tolist(),
        "positive_count": int(np.sum(primary_r > 0.0)),
        "zero_count": int(np.sum(primary_r == 0.0)),
        "negative_count": int(np.sum(primary_r < 0.0)),
        "exact_binomial": primary_sign,
        "primary_classification": primary_class,
        "saved_before_secondary_analyses": True,
        "correctness_inspected_only_after_raw_seal": True,
    }
    write_json(output_dir / "PRIMARY_RESULT_SEAL.json", primary_result)
    primary_seal_hash = sha256_file(output_dir / "PRIMARY_RESULT_SEAL.json")

    metric_results: dict[str, Any] = {}
    for metric in ("A0", "A1", "A2", "D2"):
        metric_geometry = {shell: matrices[f"{metric}_{shell}_FRESH_REFERENCE"] for shell in SHELLS}
        metric_results[metric] = safe_metric_result(metric, metric_geometry, shape_ref)
    pvals = {
        metric: float(metric_results[metric]["exact_sign"]["p_value"])
        for metric in ("A1", "A2", "D2")
    }
    holm = holm_adjust(pvals)
    for metric, adjusted in holm.items():
        metric_results[metric]["holm_adjusted_exact_sign_p"] = adjusted
        metric_results[metric]["role"] = "SECONDARY_ONLY_CANNOT_RESCUE_A0"

    primary_permutations_seed = int.from_bytes(
        hashlib.sha256(f"Q2-OOS-V2-ROW-QAP|{EXPECTED_PREDICTION_LOCK_SHA256}".encode()).digest()[
            :16
        ],
        "big",
    )
    permutations = fresh_row_permutations(
        EXPECTED_CONTROLLER_COUNT, QAP_MAPS, seed=primary_permutations_seed
    )
    qap = row_permutation_test(primary_geometry, shape_ref, permutations)
    cluster = controller_cluster_bootstrap(
        primary_geometry, shape_ref, metric_results["A0"]["global"]["equal_shell_mean"]
    )
    item_uncertainty = item_bootstrap(
        {
            shell: np.stack([errors[f"{controller}_{shell}"] for controller in FRESH_IDS], axis=0)
            for shell in SHELLS
        },
        reference_errors,
        primary_geometry,
    )
    lofo_results = lofo(primary_geometry, shape_ref)
    fresh_fresh_geometry = {shell: matrices[f"A0_{shell}_FRESH_FRESH"] for shell in SHELLS}
    fresh_fresh_jackknife = node_jackknife_test(fresh_fresh_geometry, shape_ff)
    secondary = {
        "global_fresh_reference": metric_results,
        "original_row_qap": {
            "role": "DIAGNOSTIC_ONLY",
            "maps": QAP_MAPS,
            "seed": str(primary_permutations_seed),
            "observed_equal_shell_mean": float(qap["observed_aggregate_rho"]),
            "observed_shell_rho": qap["observed_shell_rho"],
            "p_value": float(qap["p_value"]),
            "degenerate_fail_closed": bool(qap["degenerate_fail_closed"]),
        },
        "studentized_controller_mean": metric_results["A0"]["studentized_mean"],
        "controller_cluster_bootstrap": cluster,
        "item_bootstrap": item_uncertainty,
        "lofo": lofo_results,
        "fresh_fresh_node_jackknife": {
            **{
                key: value
                for key, value in fresh_fresh_jackknife.items()
                if key not in {"leave_one_out", "pseudovalues"}
            },
            "role": "SECONDARY_ONLY_CANNOT_RESCUE_PRIMARY",
            "method": "NODE_JACKKNIFE_PSEUDOVALUE_T",
        },
    }
    result = {
        "schema_version": "q2-oos-v2-semantic-analysis-v1",
        "status": "ANALYSIS_COMPLETE",
        "raw_seal": {
            "status": raw_seal["status"],
            "journal_sha256": EXPECTED_JOURNAL_SHA256,
            "journal_bytes": int(raw_seal["raw_journal_bytes"]),
            "rows": int(raw_seal["completed_rows"]),
            "missing": int(raw_seal["missing_rows"]),
            "unexpected": int(raw_seal["unexpected_rows"]),
            "duplicates": int(raw_seal["duplicate_keys"]),
            "replacements": int(raw_seal["replacements"]),
            "retries": int(raw_seal["retry_row_count"]),
            "runtime_errors": int(raw_seal["runtime_error_count"]),
            "repetition_stops": int(raw_seal["repetition_stop_count"]),
            "hard_cap_stops": int(raw_seal["hard_cap_count"]),
            "wall_seconds": float(raw_seal["elapsed_seconds"]),
        },
        "provenance": {
            "code_commit": EXPECTED_CODE_COMMIT,
            "prediction_lock_sha256": EXPECTED_PREDICTION_LOCK_SHA256,
            "prediction_lock_parent": EXPECTED_INFERENCE_LOCK_PARENT,
            "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
            "panel_sha256": EXPECTED_PANEL_SHA256,
            "selected_bank_sha256": EXPECTED_SELECTED_BANK_SHA256,
            "model_revision": EXPECTED_MODEL_REVISION,
            "environment_fingerprint": EXPECTED_ENVIRONMENT,
            "parser_version": PARSER_VERSION,
            "parser_source_sha256": sha256_file(PARSER_SOURCE),
            "collector_source_sha256": sha256_file(COLLECTOR),
            "reference_scores_sha256": reference_meta["path_sha256"],
            "reference_order": list(REFERENCE_IDS),
            "fresh_order": list(FRESH_IDS),
            "matrix_archive_sha256": EXPECTED_MATRIX_ARCHIVE_SHA256,
            "matrix_metadata_sha256": EXPECTED_MATRIX_METADATA_SHA256,
            "raw_identity": raw_identity,
        },
        "scoring": {
            "scored_rows": len(scores),
            "correctness_inspected_only_after_raw_seal": True,
            "complete_case_filtering": False,
            "condition_summaries": summaries,
        },
        "primary": {
            "r_i": primary_r.tolist(),
            "positive_count": int(np.sum(primary_r > 0.0)),
            "zero_count": int(np.sum(primary_r == 0.0)),
            "negative_count": int(np.sum(primary_r < 0.0)),
            "median": float(np.median(primary_r)),
            "mean": float(np.mean(primary_r)),
            "exact_binomial": primary_sign,
            "classification": primary_class,
            "result_seal_sha256": primary_seal_hash,
        },
        "secondary": secondary,
        "efficient_termination": {
            "repetition_stops": 359,
            "hard_cap_stops": 2,
            "wall_seconds": 33924.293892600996,
            "role": "OPERATIONAL_DIAGNOSTIC_ONLY",
        },
        "private_artifacts": {
            "error_arrays_sha256": error_hash,
            "dshape_sha256": dshape_hash,
            "dtotal_sha256": dtotal_hash,
            "scored_dataset_sha256": sha256_file(output_dir / "SEMANTIC_SCORES.jsonl"),
        },
    }
    forensic = independent_forensic(
        raw_dir, item_ids, items, schedule, errors, shape_ref, primary_r, primary_class
    )
    result["forensic"] = forensic
    write_json(output_dir / "ANALYSIS_RESULTS.json", result)
    release_dir.mkdir(parents=True, exist_ok=True)
    release_result = dict(result)
    release_result["provenance"] = {
        key: value for key, value in result["provenance"].items() if key != "raw_identity"
    }
    release_result["scoring"] = {
        "scored_rows": len(scores),
        "correctness_inspected_only_after_raw_seal": True,
        "complete_case_filtering": False,
        "condition_summaries": summaries,
    }
    write_json(release_dir / "Q2_OOS_V2_SEMANTIC_ANALYSIS.json", release_result)
    write_json(release_dir / "Q2_OOS_V2_PRIMARY_RESULT_SEAL.json", primary_result)
    write_json(release_dir / "Q2_OOS_V2_FORENSIC_AUDIT.json", forensic)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--historical-scores", type=Path, default=HISTORICAL_SCORE_PATH_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=ROOT / "review/q2_oos_fresh_controller_design/v2_semantic_execution",
    )
    args = parser.parse_args()
    result = analyze(args.raw_dir, args.historical_scores, args.output_dir, args.release_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary": result["primary"]["classification"],
                "forensic": result["forensic"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
