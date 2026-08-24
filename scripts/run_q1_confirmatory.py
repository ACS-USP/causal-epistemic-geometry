#!/usr/bin/env python3
"""Crash-safe two-model runner for the locked Q1 confirmatory evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_gate7_fresh_l27_replication as qwen_runner  # noqa: E402
import run_gate13_cross_model_ministral3 as ministral_runner  # noqa: E402

from epistemic_geometry.benchmarks.external.semantic_v3 import PARSER_VERSION  # noqa: E402
from epistemic_geometry.experiments import gate9  # noqa: E402
from epistemic_geometry.experiments import q1_confirmatory as q1  # noqa: E402
from epistemic_geometry.experiments.gate6 import SYSTEM_CAREFUL  # noqa: E402
from epistemic_geometry.experiments.gate13 import SOURCE_CAREFUL  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402

REVIEW = ROOT / "review/q1_confirmatory_fixed_controllers"
HOLDOUT_ACCESS_ENV = "Q1_CONFIRMATORY_HOLDOUT_ACCESS"
HOLDOUT_ACCESS_VALUE = "AUTHORIZED_AFTER_COST_GATE"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(source_commit: str) -> dict[str, Any]:
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    if lock["status"] != "CONFIRMATORY_LOCKED_PRE_HOLDOUT":
        raise RuntimeError("Q1 confirmatory protocol is not finally locked")
    if git_commit() != source_commit:
        raise RuntimeError("runtime checkout differs from confirmatory source commit")
    parser = read_json(REVIEW / "RESPONSE_PARSER_LOCK.json")
    module = ROOT / parser["module"]
    if parser["version"] != PARSER_VERSION or sha256(module) != parser["module_sha256"]:
        raise RuntimeError("confirmatory parser differs from the frozen lock")
    return lock


def _require_holdout_access() -> None:
    if os.environ.get(HOLDOUT_ACCESS_ENV) != HOLDOUT_ACCESS_VALUE:
        raise RuntimeError("holdout content access requires the post-cost-gate environment lock")


def materialize_holdout() -> dict[str, Any]:
    """First content access: materialize only the prospectively assigned 57 IDs."""

    _require_holdout_access()
    require_remote_hf_execution("Q1 confirmatory holdout materialization")
    from datasets import load_dataset

    identity = read_json(REVIEW / "HOLDOUT_IDENTITY_LOCK.json")
    audit = read_json(REVIEW / "HOLDOUT_PROVENANCE_AUDIT.json")
    ordered_ids = [str(value) for value in audit["reserved_cruxeval_57"]["ids"]]
    if len(ordered_ids) != 57 or identity["ordered_id_list_sha256"] != (
        "a012b4d203d88d807a146ebbe8429c55a1834c6b8e0df5751a12b677ff7b2462"
    ):
        raise RuntimeError("sealed holdout identity mismatch")
    candidates = load_dataset(
        gate9.DATASET_REPO,
        split="test",
        revision=gate9.DATASET_REVISION,
    )
    normalized_rows = [gate9.normalize_dataset_row(row) for row in candidates]
    normalized = {row["item_id"]: row for row in normalized_rows}
    if any(item_id not in normalized for item_id in ordered_ids):
        raise RuntimeError("pinned dataset is missing a sealed holdout ID")
    selected = [normalized[item_id] for item_id in ordered_ids]
    payload = {
        "experiment_id": q1.EXPERIMENT_ID,
        "role": identity["role"],
        "first_content_access_after_cost_gate": True,
        "dataset_repo": gate9.DATASET_REPO,
        "dataset_revision": gate9.DATASET_REVISION,
        "n_items": len(selected),
        "items": selected,
    }
    write_json(REVIEW / "HOLDOUT_CONTENT_MANIFEST.json", payload)
    write_json(
        REVIEW / "HOLDOUT_ACCESS_EVENT.json",
        {
            "status": "CONTENT_MATERIALIZED_FOR_AUTHORIZED_CONFIRMATORY_COLLECTION",
            "n_items": 57,
            "manifest_sha256": sha256(REVIEW / "HOLDOUT_CONTENT_MANIFEST.json"),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "scientific_metrics_inspected": False,
        },
    )
    return payload


def vectors(model_role: str) -> tuple[dict[str, np.ndarray], dict[str, str], float, int]:
    controllers = read_json(REVIEW / "CONTROLLER_IDENTITY_LOCK.json")["controllers"]
    record = controllers[model_role]
    values = {
        "MEANINGFUL_FIXED": np.load(ROOT / record["vector_path"], allow_pickle=False).astype(
            np.float64
        )
    }
    bank = read_json(REVIEW / f"NULL_BANK_LOCK_{model_role.upper()}.json")
    for condition in q1.RANDOM_NAMES:
        values[condition] = np.load(
            ROOT / bank["records"][condition]["vector_path"], allow_pickle=False
        ).astype(np.float64)
    hashes = {
        "MEANINGFUL_FIXED": record["vector_hash"],
        **{
            condition: bank["records"][condition]["canonical_float64_vector_sha256"]
            for condition in q1.RANDOM_NAMES
        },
    }
    from epistemic_geometry.experiments.gate6_3 import vector_sha256

    if any(vector_sha256(value) != hashes[name] for name, value in values.items()):
        raise RuntimeError("confirmatory controller/null hash mismatch")
    scale = float(record["eta"]) * float(record["reference_scale"])
    return {name: value * scale for name, value in values.items()}, hashes, scale, 27


def build_backend(model_role: str, model_path: str) -> Any:
    if model_role == "Qwen":
        return qwen_runner.build_backend(model_path)
    record = read_json(REVIEW / "CONTROLLER_IDENTITY_LOCK.json")["controllers"][model_role]
    return ministral_runner.build_backend(model_path, record["model"], record["revision"])


def engineering(model_role: str, model_path: str) -> None:
    backend = build_backend(model_role, model_path)
    if model_role == "Qwen":
        deltas, hashes, _scale, _layer = vectors(model_role)
        qwen_runner.REVIEW = REVIEW
        qwen_runner.LAYER = 27
        qwen_runner.MEANINGFUL = "MEANINGFUL_FIXED"
        qwen_runner.RANDOM_NAMES = q1.RANDOM_NAMES
        result = qwen_runner.engineering_gate(
            backend,
            REVIEW,
            {"model": {"environment_profile": "CORE_QWEN"}},
            deltas,
            hashes,
        )
        write_json(REVIEW / "ENGINEERING_CHECKS_QWEN.json", result)
    else:
        record = read_json(REVIEW / "CONTROLLER_IDENTITY_LOCK.json")["controllers"][model_role]
        ministral_runner.engineering_gate(
            backend, REVIEW, record["model"], record["revision"]
        )
        result = read_json(REVIEW / "ENGINEERING_CHECKS.json")
        write_json(REVIEW / "ENGINEERING_CHECKS_MINISTRAL.json", result)


def _condition_context(
    backend: Any,
    item: Any,
    model_role: str,
    condition: str,
    deltas: dict[str, np.ndarray],
    hashes: dict[str, str],
) -> tuple[Any, Any, str, dict[str, Any]]:
    if model_role == "Qwen":
        system = SYSTEM_CAREFUL if condition == "TEXTUAL_CAREFUL" else None
        row = qwen_runner.model_item(item, system)
        prompt_ids, _rendered, rendered_hash = qwen_runner.prompt_tokens(backend, row)
    else:
        system = SOURCE_CAREFUL if condition == "TEXTUAL_CAREFUL" else None
        row = ministral_runner.model_item(item, system)
        prompt_ids, rendered_hash = ministral_runner.prompt_tokens(backend, row)
    if condition not in deltas:
        return nullcontext(), row, rendered_hash, {"intervention": "none"}
    tensor = backend.torch.tensor(
        deltas[condition], dtype=backend.torch.float32, device=backend.device
    ).view(1, 1, -1)
    context = Gate6HookTrace(
        layers={27: backend.layer_module(27)},
        deltas={27: tensor},
        target_positions=[len(prompt_ids) - 1],
    )
    return context, row, rendered_hash, {
        "intervention": condition,
        "intervention_layer": 27,
        "intervention_duration": "sustained_current_token",
        "intervention_scope": "final_prompt_token_then_current_decode_token",
        "intervention_vector_hash": hashes[condition],
        "effective_delta_norm": float(np.linalg.norm(deltas[condition])),
    }


def collect(model_role: str, model_path: str, source_commit: str) -> None:
    _require_holdout_access()
    manifest = REVIEW / "HOLDOUT_CONTENT_MANIFEST.json"
    if not manifest.exists():
        materialize_holdout()
    backend = build_backend(model_role, model_path)
    loader = qwen_runner.load_external if model_role == "Qwen" else ministral_runner.load_external
    items = loader(manifest)
    item_by_id = {item.item_id: item for item in items}
    schedule = read_json(REVIEW / "SEED_SCHEDULE_LOCK.json")["schedules"][model_role]
    journal = REVIEW / f"journal_{model_role.lower()}.jsonl"
    existing = [] if not journal.exists() else [
        json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line
    ]
    complete = q1.completed_keys(existing, source_commit=source_commit)
    deltas, hashes, _scale, layer = vectors(model_role)
    controller = read_json(REVIEW / "CONTROLLER_IDENTITY_LOCK.json")["controllers"][model_role]
    for schedule_index, planned in enumerate(schedule):
        key = (
            model_role,
            str(planned["item_id"]),
            str(planned["condition"]),
            int(planned["rollout_index"]),
        )
        if key in complete:
            continue
        item = item_by_id[key[1]]
        context, row, rendered_hash, context_metadata = _condition_context(
            backend, item, model_role, key[2], deltas, hashes
        )
        started = time.perf_counter()
        with context as trace:
            if model_role == "Qwen":
                output = backend.generate_reasoning(
                    row,
                    sampling_seed=int(planned["seed"]),
                    max_new_tokens=4096,
                    intervention_metadata=context_metadata,
                )
                vision_calls = 0
            else:
                output, vision_calls = ministral_runner.generate_with_vision_audit(
                    backend,
                    row,
                    seed=int(planned["seed"]),
                    max_new_tokens=4096,
                    intervention_metadata={**context_metadata, "vision_invoked": False},
                )
        metadata = dict(output.metadata)
        if key[2] in deltas:
            metadata["intervention_forward_trace"] = trace.metadata()
        token_count = int(metadata.get("generated_token_count", 0))
        scored = (
            qwen_runner.score(output.raw_output, item.reference_answer, token_count)
            if model_role == "Qwen"
            else ministral_runner.score(output.raw_output, item.reference_answer, token_count)
        )
        output_hash = hashlib.sha256(output.raw_output.encode()).hexdigest()
        record = {
            **planned,
            **scored,
            "confirmatory_source_commit": source_commit,
            "runtime_commit": git_commit(),
            "model": controller["model"],
            "model_revision": controller["revision"],
            "tokenizer_revision": controller["tokenizer_revision"],
            "parser_version": PARSER_VERSION,
            "raw_output": output.raw_output,
            "output_sha256": output_hash,
            "generated_token_ids": metadata.get("generated_token_ids", []),
            "generated_token_count": token_count,
            "reference_answer": item.reference_answer,
            "reference_canonical_type": item.metadata.get("reference_canonical_type"),
            "prompt_hash": item.prompt_hash,
            "rendered_prompt_hash": rendered_hash,
            "condition_metadata": context_metadata,
            "backend_metadata": metadata,
            "vision_tower_calls": vision_calls,
            "elapsed_seconds": time.perf_counter() - started,
            "retry_count": 0,
            "schedule_index": schedule_index,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        append_jsonl(journal, record)
        complete.add(key)
        if len(complete) % 100 == 0:
            print(json.dumps({"model_role": model_role, "rows": len(complete)}), flush=True)
    write_json(
        REVIEW / f"COLLECTION_METADATA_{model_role.upper()}.json",
        {
            "model_role": model_role,
            "completed_rows": len(complete),
            "expected_rows": 798,
            "confirmatory_source_commit": source_commit,
            "scientific_metrics_inspected": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("materialize", "engineering", "collect"), required=True)
    parser.add_argument("--model-role", choices=("Qwen", "Ministral"))
    parser.add_argument("--model-path")
    parser.add_argument("--confirmatory-source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution(f"Q1 confirmatory {args.mode}")
    load_protocol(args.confirmatory_source_commit)
    if args.mode == "materialize":
        materialize_holdout()
        return 0
    if not args.model_role or not args.model_path:
        raise SystemExit("engineering/collect require --model-role and --model-path")
    if args.mode == "engineering":
        engineering(args.model_role, args.model_path)
    else:
        collect(args.model_role, args.model_path, args.confirmatory_source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
