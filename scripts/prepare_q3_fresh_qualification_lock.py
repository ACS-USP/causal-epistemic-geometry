#!/usr/bin/env python3
"""Freeze the Q3.4 qualification schedule and execution contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_fresh_instrument_qualification"
SYSTEM = ROOT / "review/q3_final_system_and_evaluation_supply/FINAL_Q3_CANDIDATE_SYSTEM_DRAFT.json"
DATASET_SEAL = REVIEW / "Q3_FRESH_INSTRUMENT_DATASET_SEAL.json"
QUALIFICATION_MANIFEST = REVIEW / "QUALIFICATION_FAMILY_MANIFEST.json"
RUNNER = ROOT / "scripts/execute_q3_fresh_qualification.py"
TESTS = ROOT / "tests/test_q3_fresh_qualification.py"

EXPECTED_SYSTEM_SHA = "d8128e4ef4bf9459977cc46a3c9698b36c96afb8a2a388428f5daf03ac6e78f0"
EXPECTED_ROUTER_SHA = "269dc116c70b64dd47cf59340b07dbe558ec8c0f13be8410ed97017310ebad3d"
EXPECTED_DATASET_SHA = "c791e38c29d36a43fbac8ce00412e4c77d533665e0b8cb9eef8fa12fb918ac1d"
EXPECTED_MODEL_MANIFEST_SHA = "cedc88ba2f732baea6bb71f5e6d7f6bc3aad00d302c3456d208a21687c9e069c"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seed_for(family_id: str, condition: str, rollout: int) -> int:
    payload = (
        f"Q3-FRESH-QUALIFICATION-V1|{EXPECTED_DATASET_SHA}|"
        f"{EXPECTED_SYSTEM_SHA}|{family_id}|{condition}|{rollout}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def policy_records(system: dict[str, Any]) -> list[dict[str, Any]]:
    historical = read_json(ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json")
    fresh = read_json(
        ROOT
        / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
        / "V2_CANDIDATE_BANK_MANIFEST.json"
    )
    historical_rows = {row["candidate_id"]: row for row in historical["candidates"]}
    fresh_rows = {row["candidate_id"]: row for row in fresh["candidates"]}
    historical_shells = read_json(
        ROOT / "review/q2_v4_spark1_presemantic/SHELL_CALIBRATION_MANIFEST_RESULT.json"
    )["controllers"]
    fresh_shells = read_json(
        ROOT
        / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout"
        / "V2_SELECTED_CONTROLLER_BANK.json"
    )["controllers"]
    wanted = [*system["portfolio"]["policies"], system["champion"]]
    records: list[dict[str, Any]] = []
    for order, policy in enumerate(wanted):
        candidate = policy["controller_id"]
        condition = policy["policy_id"]
        source = fresh_rows if candidate.startswith("Q2_OOS") else historical_rows
        shells = fresh_shells if candidate.startswith("Q2_OOS") else historical_shells
        vector = source[candidate]
        canonical = vector.get("vector_array_sha256", vector.get("canonical_vector_hash"))
        if (
            canonical != policy["vector_sha256"]
            or vector["file_sha256"] != policy["vector_file_sha256"]
        ):
            raise RuntimeError(f"frozen policy identity mismatch: {condition}")
        records.append(
            {
                "order": order,
                "role": "BANK" if order < 8 else "CHAMPION",
                "policy_id": condition,
                "controller_id": candidate,
                "shell": policy["shell"],
                "alpha": float(shells[condition]["alpha"]),
                "vector_path": vector["path"],
                "vector_file_sha256": vector["file_sha256"],
                "vector_sha256": canonical,
                "layer": 27,
                "duration": "sustained_current_token",
            }
        )
    if [row["policy_id"] for row in records[:8]] != [
        row["policy_id"] for row in system["portfolio"]["policies"]
    ]:
        raise RuntimeError("frozen bank order mismatch")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-qualification", type=Path, required=True)
    parser.add_argument("--private-prompt-output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(SYSTEM) != EXPECTED_SYSTEM_SHA:
        raise SystemExit("candidate-system hash mismatch")
    if not RUNNER.is_file() or not TESTS.is_file():
        raise SystemExit("qualification implementation is incomplete")
    dataset = read_json(DATASET_SEAL)
    if dataset["status"] != "DATASET_COMPLETE_RAW_UNOPENED_TO_QWEN":
        raise SystemExit("dataset is not sealed and unopened")
    manifest = read_json(QUALIFICATION_MANIFEST)
    if manifest["private_dataset"]["sha256"] != EXPECTED_DATASET_SHA:
        raise SystemExit("qualification dataset identity mismatch")
    if sha256_file(args.private_qualification) != EXPECTED_DATASET_SHA:
        raise SystemExit("private qualification dataset hash mismatch")
    private_rows = [
        json.loads(line)
        for line in args.private_qualification.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(private_rows) != 300:
        raise SystemExit("private qualification dataset row count mismatch")
    prompt_rows = []
    for public, private in zip(manifest["families"], private_rows, strict=True):
        if public["family_id"] != private["family_id"]:
            raise SystemExit("private/public family order mismatch")
        prompt_hash = hashlib.sha256(private["prompt"].encode()).hexdigest()
        if prompt_hash != public["prompt_sha256"]:
            raise SystemExit("private/public prompt hash mismatch")
        prompt_rows.append(
            {
                "family_id": private["family_id"],
                "prompt": private["prompt"],
                "prompt_sha256": prompt_hash,
            }
        )
    args.private_prompt_output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    prompt_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in prompt_rows)
    if args.private_prompt_output.exists():
        if args.private_prompt_output.read_text(encoding="utf-8") != prompt_bytes:
            raise SystemExit("existing private prompt-only output differs")
    else:
        args.private_prompt_output.write_text(prompt_bytes, encoding="utf-8")
    prompt_only_sha = sha256_file(args.private_prompt_output)
    system = read_json(SYSTEM)
    if system["router"]["private_parameter_manifest"]["sha256"] != EXPECTED_ROUTER_SHA:
        raise SystemExit("private-router identity mismatch")
    policies = policy_records(system)
    conditions = [row["policy_id"] for row in policies] + ["ONLINE_ROUTED"]
    rows: list[dict[str, Any]] = []
    for family in manifest["families"]:
        for rollout in (0, 1):
            for condition in conditions:
                rows.append(
                    {
                        "family_id": family["family_id"],
                        "family_order": int(family["order"]),
                        "condition": condition,
                        "rollout_index": rollout,
                        "seed": seed_for(family["family_id"], condition, rollout),
                        "prompt_sha256": family["prompt_sha256"],
                    }
                )
    keys = {(r["family_id"], r["condition"], r["rollout_index"]) for r in rows}
    seeds = {r["seed"] for r in rows}
    if len(rows) != 6000 or len(keys) != 6000 or len(seeds) != 6000:
        raise RuntimeError("schedule identity/seed collision")
    schedule = {
        "schema_version": "q3-fresh-qualification-schedule-v1",
        "status": "FROZEN_NOT_RUN",
        "family_manifest_sha256": sha256_file(QUALIFICATION_MANIFEST),
        "private_qualification_dataset_sha256": EXPECTED_DATASET_SHA,
        "private_prompt_only_dataset": {
            "sha256": prompt_only_sha,
            "rows": 300,
            "contains_references": False,
            "tracked_in_git": False,
        },
        "families": 300,
        "conditions": conditions,
        "rollouts": [0, 1],
        "logical_generations": 6000,
        "exact_policy_sharing": False,
        "rows": rows,
    }
    schedule_path = REVIEW / "Q3_FRESH_QUALIFICATION_SCHEDULE.json"
    write_json(schedule_path, schedule)
    parent = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    lock = {
        "schema_version": "q3-fresh-qualification-execution-lock-v1",
        "status": "FROZEN_BEFORE_QWEN_QUALIFICATION",
        "source_parent": parent,
        "scientific_outcomes_before_lock": 0,
        "qualification_generations_before_lock": 0,
        "confirmation_generations": 0,
        "reserve_generations": 0,
        "candidate_system": {"path": str(SYSTEM.relative_to(ROOT)), "sha256": EXPECTED_SYSTEM_SHA},
        "private_router_sha256": EXPECTED_ROUTER_SHA,
        "dataset_seal_sha256": sha256_file(DATASET_SEAL),
        "qualification_manifest_sha256": sha256_file(QUALIFICATION_MANIFEST),
        "private_qualification_dataset_sha256": EXPECTED_DATASET_SHA,
        "private_prompt_only_dataset": {
            "sha256": prompt_only_sha,
            "rows": 300,
            "contains_references": False,
            "tracked_in_git": False,
        },
        "schedule_sha256": sha256_file(schedule_path),
        "implementation": {
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": sha256_file(RUNNER),
            "tests": str(TESTS.relative_to(ROOT)),
            "tests_sha256": sha256_file(TESTS),
        },
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "model_byte_manifest_sha256": EXPECTED_MODEL_MANIFEST_SHA,
            "dtype": "BF16",
            "attention": "SDPA",
        },
        "generation": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_new_tokens": 4096,
            "enable_thinking": False,
            "termination": "EXTREME_MECHANICAL_REPETITION_V1",
        },
        "policies": policies,
        "online_router": {
            "feature": "unsteered layer-27 block input at final non-padding prompt token",
            "select_once": "during prefill before the same layer-27 invocation output hook",
            "selection": "argmax frozen geometry-blind policy-identity score",
            "tie_break": "frozen policy order",
            "fallback": system["champion"]["policy_id"],
            "sampling_rng_consumed": False,
        },
        "qualification_gates": read_json(
            ROOT
            / "review/q3_final_system_and_evaluation_supply/Q3_FRESH_INSTRUMENT_DESIGN_DRAFT.json"
        )["instrument_qualification"]["gates"],
        "aggregation": {
            "validity_evaluability_repetition": "pooled across 600 rows per condition",
            "champion_accuracy": "pooled across 600 rows",
            "oracle_headroom": (
                "mean over 300 families of max bank two-rollout correctness mean "
                "minus champion two-rollout correctness mean"
            ),
            "routed_gain_is_gate": False,
        },
        "firewall": {
            "collection_imports_parser": False,
            "collection_loads_reference": False,
            "confirmation_qwen_access": False,
            "reserve_qwen_access": False,
            "spark1_only": True,
            "spark2": False,
            "runpod": False,
        },
    }
    write_json(REVIEW / "Q3_FRESH_QUALIFICATION_EXECUTION_LOCK.json", lock)
    print(json.dumps({"schedule_sha256": lock["schedule_sha256"], "rows": len(rows)}))


if __name__ == "__main__":
    main()
