#!/usr/bin/env python3
"""Outcome-free Spark-1 safety qualification for the frozen Q2 OOS V2 stream.

This runner never evaluates benchmark correctness.  It reuses the immutable
V4 baseline safety rows and denominator, then executes exactly 34 x 2 x 12 x 2
new controller-shell trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_q2_v3 import _calibrate_alpha, _mechanical_parse  # noqa: E402
from run_q2_v4_presemantic import (  # noqa: E402
    MAX_NEW_TOKENS,
    _condition_context,
    build_v4_backend,
    items,
)

from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.experiments.q2_oos_fresh_controller import (  # noqa: E402
    coefficient_bank_diagnostics,
    cross_block_diagnostics,
)
from epistemic_geometry.research.reliability import CrashSafeJournal  # noqa: E402

REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
HISTORICAL = ROOT / "review/q2_v4_spark1_presemantic"
REFERENCE = ROOT / "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
STREAM_MANIFEST = REVIEW / "V2_CANDIDATE_BANK_MANIFEST.json"
PROTOCOL_LOCK = REVIEW / "V2_FINAL_PROTOCOL_LOCK.json"
PRELOCK = REVIEW / "V2_PRELOCK.json"
SCHEDULE = REVIEW / "V2_SAFETY_SCHEDULE.json"
EXECUTION_LOCK = REVIEW / "V2_SAFETY_EXECUTION_LOCK.json"
HISTORICAL_JOURNAL = HISTORICAL / "CANDIDATE_SAFETY_JOURNAL.jsonl"
HISTORICAL_JOURNAL_SHA256 = "6dbaa8977c28c00508849953008972c098c3cbb0f0f8fda973c5718845cd2fb5"
QUALIFIED_ENVIRONMENT_PROFILE = "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
SHELLS = ("MEDIUM", "STRONG")
SHELL_TARGETS = {"MEDIUM": 0.25, "STRONG": 0.50}
EXPECTED_ROWS = 34 * 2 * 12 * 2


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def load_candidates() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = read_json(STREAM_MANIFEST)
    if manifest["candidate_count"] != 34 or manifest["generated_once"] is not True:
        raise RuntimeError("Q2_OOS_V2_CANDIDATE_STREAM_INTEGRITY_FAILURE")
    vectors: dict[str, np.ndarray] = {}
    for expected_index, record in enumerate(manifest["candidates"]):
        if record["candidate_index"] != expected_index:
            raise RuntimeError("Q2_OOS_V2_CANDIDATE_ORDER_FAILURE")
        path = ROOT / record["path"]
        if sha256(path) != record["file_sha256"]:
            raise RuntimeError(f"Q2_OOS_V2_VECTOR_HASH_MISMATCH:{record['candidate_id']}")
        vector = np.load(path, allow_pickle=False)
        if vector.shape != (4096,) or vector.dtype != np.float64:
            raise RuntimeError(f"Q2_OOS_V2_VECTOR_FORMAT_MISMATCH:{record['candidate_id']}")
        if vector_sha256(vector) != record["vector_array_sha256"]:
            raise RuntimeError(f"Q2_OOS_V2_VECTOR_ARRAY_HASH_MISMATCH:{record['candidate_id']}")
        vectors[record["candidate_id"]] = vector
    return vectors, manifest


def historical_baselines() -> dict[tuple[str, int], dict[str, Any]]:
    if sha256(HISTORICAL_JOURNAL) != HISTORICAL_JOURNAL_SHA256:
        raise RuntimeError("Q2_OOS_V2_HISTORICAL_BASELINE_HASH_MISMATCH")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    with HISTORICAL_JOURNAL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)["row"]
            if row["condition"] != "BASELINE":
                continue
            key = (str(row["item_id"]), int(row["rollout_index"]))
            if key in result:
                raise RuntimeError("Q2_OOS_V2_DUPLICATE_HISTORICAL_BASELINE")
            if row.get("correctness_evaluated") is not False:
                raise RuntimeError("Q2_OOS_V2_CORRECTNESS_FIREWALL_FAILURE")
            result[key] = {
                "matched_seed": int(row["matched_seed"]),
                "commitment_valid": bool(row["commitment_valid"]),
                "semantic_evaluable": bool(row["semantic_evaluable"]),
                "generated_token_ids": list(row["generated_token_ids"]),
            }
    if len(result) != 24:
        raise RuntimeError("Q2_OOS_V2_HISTORICAL_BASELINE_COUNT_MISMATCH")
    return result


def build_schedule() -> list[dict[str, Any]]:
    vectors, manifest = load_candidates()
    del vectors
    baseline = historical_baselines()
    item_ids = [item.item_id for item in items("SHELL_CALIBRATION_MANIFEST.json")]
    candidate_ids = [row["candidate_id"] for row in manifest["candidates"]]
    conditions = [f"{candidate}_{shell}" for candidate in candidate_ids for shell in SHELLS]
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        for rollout in (0, 1):
            key = (item_id, rollout)
            order_seed = stable_seed(
                "Q2-OOS-V2-SAFETY-ORDER", manifest["prelock_commit"], item_id, rollout
            )
            generator = np.random.Generator(np.random.PCG64DXSM(order_seed))
            for condition_order, condition in enumerate(generator.permutation(conditions).tolist()):
                candidate_id, shell = str(condition).rsplit("_", 1)
                rows.append(
                    {
                        "item_id": item_id,
                        "candidate_id": candidate_id,
                        "shell": shell,
                        "condition": str(condition),
                        "rollout_index": rollout,
                        "matched_seed": baseline[key]["matched_seed"],
                        "condition_order": condition_order,
                    }
                )
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("Q2_OOS_V2_SAFETY_SCHEDULE_COUNT_MISMATCH")
    keys = {(row["item_id"], row["condition"], row["rollout_index"]) for row in rows}
    if len(keys) != EXPECTED_ROWS:
        raise RuntimeError("Q2_OOS_V2_SAFETY_SCHEDULE_DUPLICATE")
    return rows


def prepare() -> None:
    protocol = read_json(PROTOCOL_LOCK)
    prelock = read_json(PRELOCK)
    if protocol["status"] != "Q2_OOS_V2_FINAL_PROTOCOL_FROZEN":
        raise RuntimeError("Q2_OOS_V2_PROTOCOL_NOT_FROZEN")
    if prelock["status"] != "Q2_OOS_V2_PRELOCK_READY_FOR_COMMIT":
        raise RuntimeError("Q2_OOS_V2_PRELOCK_NOT_FROZEN")
    rows = build_schedule()
    write_json(
        SCHEDULE,
        {
            "schema_version": "q2-oos-v2-safety-schedule-v1",
            "prelock_commit": prelock["prelock_commit"],
            "selection_forbidden": True,
            "correctness": "FORBIDDEN",
            "historical_baseline_sha256": HISTORICAL_JOURNAL_SHA256,
            "expected_rows": EXPECTED_ROWS,
            "rows": rows,
        },
    )
    source_hash = sha256(Path(__file__))
    write_json(
        EXECUTION_LOCK,
        {
            "schema_version": "q2-oos-v2-safety-execution-lock-v1",
            "status": "Q2_OOS_V2_SAFETY_EXECUTION_FROZEN_NOT_RUN",
            "protocol_lock_sha256": sha256(PROTOCOL_LOCK),
            "prelock_sha256": sha256(PRELOCK),
            "candidate_manifest_sha256": sha256(STREAM_MANIFEST),
            "safety_schedule_sha256": sha256(SCHEDULE),
            "runner_path": str(Path(__file__).relative_to(ROOT)),
            "runner_sha256": source_hash,
            "historical_baseline_sha256": HISTORICAL_JOURNAL_SHA256,
            "qualified_environment_profile": QUALIFIED_ENVIRONMENT_PROFILE,
            "planned_new_model_trajectories": EXPECTED_ROWS,
            "new_baseline_trajectories": 0,
            "correctness": "FORBIDDEN",
            "semantic_panel": "FORBIDDEN",
            "Spark1": "ONLY",
            "Spark2": "FORBIDDEN",
            "RunPod": "FORBIDDEN",
        },
    )
    print(
        json.dumps(
            {
                "schedule_rows": len(rows),
                "schedule_sha256": sha256(SCHEDULE),
                "execution_lock_sha256": sha256(EXECUTION_LOCK),
            },
            sort_keys=True,
        )
    )


def validate_static(expected_commit: str) -> dict[str, Any]:
    if git_head() != expected_commit:
        raise RuntimeError("Q2_OOS_V2_EXECUTION_COMMIT_MISMATCH")
    lock = read_json(EXECUTION_LOCK)
    checks = {
        "protocol_lock_sha256": sha256(PROTOCOL_LOCK) == lock["protocol_lock_sha256"],
        "prelock_sha256": sha256(PRELOCK) == lock["prelock_sha256"],
        "candidate_manifest_sha256": sha256(STREAM_MANIFEST) == lock["candidate_manifest_sha256"],
        "safety_schedule_sha256": sha256(SCHEDULE) == lock["safety_schedule_sha256"],
        "runner_sha256": sha256(Path(__file__)) == lock["runner_sha256"],
        "historical_baseline_sha256": sha256(HISTORICAL_JOURNAL) == HISTORICAL_JOURNAL_SHA256,
        "schedule_rows_1632": len(read_json(SCHEDULE)["rows"]) == EXPECTED_ROWS,
        "candidate_count_34": len(load_candidates()[0]) == 34,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Q2_OOS_V2_PREOPEN_FAILURE:{checks}")
    return checks


def environment_preflight(model_path: Path) -> dict[str, Any]:
    import torch
    import transformers

    checks = {
        "hostname_spark1": platform.node() == "spark1",
        "architecture_aarch64": platform.machine() == "aarch64",
        "python_3_12_3": platform.python_version() == "3.12.3",
        "torch_exact": torch.__version__ == "2.13.0+cu130",
        "transformers_exact": transformers.__version__ == "4.57.6",
        "cuda_available": bool(torch.cuda.is_available()),
        "one_gpu": torch.cuda.device_count() == 1,
        "gpu_gb10": torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GB10",
        "bf16_supported": torch.cuda.is_available() and bool(torch.cuda.is_bf16_supported()),
        "model_path_exists": model_path.is_dir(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Q2_OOS_V2_SEMANTIC_ENVIRONMENT_DRIFT:{checks}")
    return {
        "checks": checks,
        "qualified_environment_profile": QUALIFIED_ENVIRONMENT_PROFILE,
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "dtype": "bfloat16",
        "attention": "sdpa",
    }


def deployment(vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    historical = read_json(HISTORICAL / "SHELL_CALIBRATION_MANIFEST_RESULT.json")
    denominator = float(historical["denominator_mean_squared_norm"])
    result = {}
    for candidate_id, vector in vectors.items():
        for shell in SHELLS:
            condition = f"{candidate_id}_{shell}"
            result[condition] = {
                "condition": condition,
                "candidate_id": candidate_id,
                "shell": shell,
                "target_amplitude": SHELL_TARGETS[shell],
                "vector_hash": vector_sha256(vector),
                **_calibrate_alpha(vector, SHELL_TARGETS[shell], denominator),
            }
    return result


def selected_gate(
    selected_ids: list[str], manifest: dict[str, Any], deployed: dict[str, Any]
) -> dict[str, Any]:
    records = {row["candidate_id"]: row for row in manifest["candidates"]}
    coefficients = np.asarray(
        [records[name]["coefficients"] for name in selected_ids], dtype=np.float64
    )
    references = np.asarray(
        [row["coefficients"] for row in read_json(REFERENCE)["directions"]], dtype=np.float64
    )
    metrics = coefficient_bank_diagnostics(coefficients)
    cross = cross_block_diagnostics(coefficients, references)
    cvs = {}
    for shell in SHELLS:
        amplitudes = np.asarray(
            [deployed[f"{name}_{shell}"]["implemented_amplitude"] for name in selected_ids]
        )
        cvs[shell] = float(np.std(amplitudes, ddof=0) / np.mean(amplitudes))
    shell_cv = max(cvs.values())
    lofo_spreads = []
    for index in range(len(selected_ids)):
        reduced = np.delete(coefficients, index, axis=0)
        lofo_spreads.append(float(cross_block_diagnostics(reduced, references)["a0_q90_minus_q10"]))
    checks = {
        "count_16": metrics["count"] == 16,
        "rank_8": metrics["rank"] == 8,
        "effective_rank_at_least_4_8": metrics["effective_rank"] >= 4.8,
        "maximum_absolute_pair_cosine_below_0_98": metrics["maximum_absolute_pair_cosine"] < 0.98,
        "cross_block_A0_q90_minus_q10_at_least_0_20": cross["a0_q90_minus_q10"] >= 0.20,
        "shell_amplitude_cv_at_most_0_03": shell_cv <= 0.03,
    }
    return {
        "metrics": {
            **metrics,
            **cross,
            "condition_number_status": "DESCRIPTIVE_ONLY",
            "shell_amplitude_cv": shell_cv,
            "shell_amplitude_cv_by_shell": cvs,
            "lofo_a0_q90_minus_q10_min": min(lofo_spreads),
            "lofo_a0_q90_minus_q10_max": max(lofo_spreads),
            "lofo_status": "DESCRIPTIVE_REQUIRED",
        },
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def safety(model_path: Path, output_dir: Path, expected_commit: str) -> None:
    static_checks = validate_static(expected_commit)
    environment = environment_preflight(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "V2_SAFETY_JOURNAL.jsonl"
    preopen_path = output_dir / "V2_SAFETY_PREOPEN_SEAL.json"
    if journal_path.exists() and journal_path.stat().st_size > 0:
        # Resume is allowed only through the immutable journal identity below.
        preexisting = True
    else:
        preexisting = False
    write_json(
        preopen_path,
        {
            "source_commit": expected_commit,
            "static_checks": static_checks,
            "environment": environment,
            "journal_preexisting_for_resume": preexisting,
            "correctness_inspected": False,
            "semantic_outcomes": 0,
        },
    )
    vectors, manifest = load_candidates()
    deployed = deployment(vectors)
    schedule = read_json(SCHEDULE)["rows"]
    baseline = historical_baselines()
    item_map = {item.item_id: item for item in items("SHELL_CALIBRATION_MANIFEST.json")}
    journal = CrashSafeJournal(
        journal_path,
        identity={
            "experiment_id": "Q2_OOS_V2_PRESEMANTIC",
            "phase": "SAFETY",
            "source_commit": expected_commit,
            "schedule_sha256": sha256(SCHEDULE),
        },
        key_fields=("item_id", "condition", "rollout_index"),
    )
    backend = build_v4_backend(str(model_path))
    started = time.monotonic()
    generated_tokens = 0
    for row in schedule:
        key = (row["item_id"], row["condition"], row["rollout_index"])
        if key in journal.rows:
            continue
        context, model_row = _condition_context(
            backend, item_map[row["item_id"]], row["condition"], vectors, deployed
        )
        with context:
            output = backend.generate_reasoning(
                model_row,
                sampling_seed=int(row["matched_seed"]),
                max_new_tokens=MAX_NEW_TOKENS,
                intervention_metadata={
                    "phase": "Q2_OOS_V2_SAFETY",
                    "correctness": "FORBIDDEN",
                },
            )
        count = int(output.metadata.get("generated_token_count", 0))
        generated_tokens += count
        journal.append(
            {
                **row,
                **_mechanical_parse(output.raw_output, count),
                "raw_output": output.raw_output,
                "generated_token_ids": output.metadata.get("generated_token_ids", []),
                "generated_token_count": count,
                "truncated": count >= MAX_NEW_TOKENS,
                "correctness_evaluated": False,
            }
        )
    rows = list(journal.rows.values())
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("Q2_OOS_V2_SAFETY_EXECUTION_INCOMPLETE")
    baseline_validity = float(np.mean([row["commitment_valid"] for row in baseline.values()]))
    baseline_evaluability = float(np.mean([row["semantic_evaluable"] for row in baseline.values()]))
    candidate_results = {}
    for candidate_id in vectors:
        shell_results = {}
        for shell in SHELLS:
            condition = f"{candidate_id}_{shell}"
            selected = [row for row in rows if row["condition"] == condition]
            movement = float(
                np.mean(
                    [
                        row["generated_token_ids"]
                        != baseline[(row["item_id"], row["rollout_index"])]["generated_token_ids"]
                        for row in selected
                    ]
                )
            )
            validity = float(np.mean([row["commitment_valid"] for row in selected]))
            evaluability = float(np.mean([row["semantic_evaluable"] for row in selected]))
            truncation = float(np.mean([row["truncated"] for row in selected]))
            calibration = deployed[condition]
            checks = {
                "validity_at_least_0_90": validity >= 0.90,
                "validity_within_baseline_margin": validity >= baseline_validity - 0.05,
                "evaluability_at_least_0_90": evaluability >= 0.90,
                "evaluability_within_baseline_margin": evaluability >= baseline_evaluability - 0.05,
                "truncation_at_most_0_05": truncation <= 0.05,
                "movement": movement >= (0.10 if shell == "MEDIUM" else 0.15),
                "relative_amplitude_error_at_most_0_005": (
                    calibration["relative_target_error"] <= 0.005
                ),
            }
            shell_results[shell] = {
                "validity": validity,
                "evaluability": evaluability,
                "truncation": truncation,
                "raw_sequence_movement": movement,
                "implemented_amplitude": calibration["implemented_amplitude"],
                "relative_target_error": calibration["relative_target_error"],
                "checks": checks,
                "pass": bool(all(checks.values())),
            }
        candidate_results[candidate_id] = {
            "shells": shell_results,
            "both_shells_pass": all(record["pass"] for record in shell_results.values()),
        }
    safe_ids = [name for name in vectors if candidate_results[name]["both_shells_pass"]]
    selected_ids = safe_ids[:16]
    if len(selected_ids) < 16:
        classification = "Q2_OOS_V2_SAFE_BANK_INSUFFICIENT"
        gate = None
    else:
        gate = selected_gate(selected_ids, manifest, deployed)
        classification = (
            "Q2_OOS_V2_SELECTED_BANK_GATE_PASS"
            if gate["pass"]
            else "Q2_OOS_V2_SELECTED_BANK_GATE_FAIL"
        )
    journal_hash = sha256(journal_path)
    report = {
        "schema_version": "q2-oos-v2-safety-result-v1",
        "source_commit": expected_commit,
        "schedule_sha256": sha256(SCHEDULE),
        "journal_sha256": journal_hash,
        "expected_rows": EXPECTED_ROWS,
        "completed_rows": len(rows),
        "baseline_source": "IMMUTABLE_V4_PRESEMANTIC_BASELINE",
        "baseline_journal_sha256": HISTORICAL_JOURNAL_SHA256,
        "baseline_validity": baseline_validity,
        "baseline_evaluability": baseline_evaluability,
        "candidates": candidate_results,
        "safe_count": len(safe_ids),
        "safe_ids_in_stream_order": safe_ids,
        "selected_first_16_safe": selected_ids,
        "selected_bank_gate": gate,
        "classification": classification,
        "correctness_used": False,
        "semantic_outcomes": 0,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "new_generated_tokens_this_invocation": generated_tokens,
    }
    write_json(output_dir / "V2_SAFETY_RESULT.json", report)
    if classification == "Q2_OOS_V2_SAFE_BANK_INSUFFICIENT":
        raise RuntimeError(classification)
    if classification != "Q2_OOS_V2_SELECTED_BANK_GATE_PASS":
        raise RuntimeError(classification)
    write_json(
        output_dir / "V2_SELECTED_CONTROLLER_BANK.json",
        {
            "schema_version": "q2-oos-v2-selected-bank-v1",
            "source_commit": expected_commit,
            "selection_rule": "first 16 candidates passing both frozen shell gates",
            "selected_ids": selected_ids,
            "controllers": {
                f"{name}_{shell}": deployed[f"{name}_{shell}"]
                for name in selected_ids
                for shell in SHELLS
            },
            "selected_bank_gate": gate,
            "classification": "Q2_OOS_V2_SELECTED_BANK_GATE_PASS",
        },
    )
    print(
        json.dumps(
            {
                "classification": classification,
                "safe_count": len(safe_ids),
                "selected_ids": selected_ids,
                "journal_sha256": journal_hash,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--expected-commit", required=True)
    preflight_parser.add_argument("--model-path", required=True, type=Path)
    safety_parser = sub.add_parser("safety")
    safety_parser.add_argument("--expected-commit", required=True)
    safety_parser.add_argument("--model-path", required=True, type=Path)
    safety_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "preflight":
        print(
            json.dumps(
                {
                    "static": validate_static(args.expected_commit),
                    "environment": environment_preflight(args.model_path),
                },
                sort_keys=True,
            )
        )
    else:
        safety(args.model_path, args.output_dir, args.expected_commit)


if __name__ == "__main__":
    main()
