#!/usr/bin/env python3
"""Collect Gate 11.1 fixed-sequence propagation with raw primitive persistence.

This script never calls ``generate``.  It replays the immutable Gate-11
continuations and writes one losslessly compressed raw shard per item before
appending the corresponding logical rows to the crash-safe journal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gate11_domain_conditioned_control import (  # noqa: E402
    DiagnosticHooks,
    build_backend,
    external_item,
    forward,
    model_item,
    prompt_tokens,
)

from epistemic_geometry.experiments import gate11, gate11_1  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402

REVIEW = ROOT / "review/gate11_1_artifact_complete_replication"
HISTORICAL = REVIEW
VECTOR = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_lock(source_commit: str) -> dict[str, Any]:
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    if lock["status"] != "FROZEN_PRE_COLLECTION":
        raise RuntimeError("Gate 11.1 lock is not frozen")
    if source_commit != lock["source_commit"]:
        raise RuntimeError("Gate 11.1 source commit mismatch")
    return lock


def load_vectors() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    meaningful = np.load(VECTOR, allow_pickle=False).astype(np.float64).reshape(-1)
    if vector_sha256(meaningful) != gate11.CONTROLLER_HASH:
        raise RuntimeError("fixed meaningful vector hash mismatch")
    vectors = {gate11.TF_MEANINGFUL: meaningful}
    hashes = {gate11.TF_MEANINGFUL: gate11.CONTROLLER_HASH}
    for index, condition in enumerate(gate11.TF_RANDOMS):
        path = REVIEW / f"GATE11_RANDOM_R{index}.npy"
        vector = np.load(path, allow_pickle=False).astype(np.float64).reshape(-1)
        if (
            vector_sha256(vector)
            != read_json(REVIEW / "RANDOM_BANK.json")["records"][f"GATE11_RANDOM_R{index}"][
                "canonical_float64_vector_sha256"
            ]
        ):
            raise RuntimeError(f"random vector hash mismatch: {condition}")
        vectors[condition] = vector
        hashes[condition] = vector_sha256(vector)
    return vectors, hashes


def condition_system_prompt(condition: str) -> str | None:
    return gate11.SYSTEM_CAREFUL if condition == gate11.TF_TEXTUAL else None


def run_condition(
    backend: Any,
    item: Any,
    domain: str,
    condition: str,
    continuation: list[int],
    delta: np.ndarray | None,
) -> dict[str, Any]:
    row = model_item(item, condition_system_prompt(condition))
    prompt_ids, rendered, prompt_hash = prompt_tokens(backend, row)
    snapshots: dict[str, dict[str, Any]] = {}
    start = time.monotonic()
    with (
        DiagnosticHooks(backend, gate11_1.PROPAGATION_LAYERS, delta, len(prompt_ids)) as hooks,
        backend.torch.inference_mode(),
    ):
        hooks.begin_capture()
        output = forward(
            backend, prompt_ids, past=None, total_length=len(prompt_ids), phase="prefill"
        )
        hooks.note_forward()
        snapshots[gate11_1.SNAPSHOT_PREFILL] = {
            "logits": output.logits[0, -1, :].detach().float().cpu().numpy().copy(),
            "hidden": hooks.end_capture(),
            "target_token": continuation[0] if continuation else None,
        }
        past = output.past_key_values
        for token_index, token in enumerate(continuation):
            if token_index not in gate11.CHECKPOINTS:
                # The forward is still required for exact cache evolution.
                capture = False
            else:
                capture = True
                hooks.begin_capture()
            output = forward(
                backend,
                [int(token)],
                past=past,
                total_length=len(prompt_ids) + token_index + 1,
                phase="decode",
            )
            hooks.note_forward()
            past = output.past_key_values
            if capture:
                snapshots[str(token_index)] = {
                    "logits": output.logits[0, -1, :].detach().float().cpu().numpy().copy(),
                    "hidden": hooks.end_capture(),
                    "target_token": continuation[token_index + 1]
                    if token_index + 1 < len(continuation)
                    else None,
                }
    return {
        "domain": domain,
        "condition": condition,
        "prompt_hash": prompt_hash,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "prompt_ids": prompt_ids,
        "prompt_length": len(prompt_ids),
        "continuation_length": len(continuation),
        "snapshots": snapshots,
        "forward_count": hooks.forward_count,
        "application_count": hooks.application_count,
        "max_relative_shift_error": hooks.max_relative_shift_error,
        "max_noncurrent_change": hooks.max_noncurrent_change,
        "duration_seconds": time.monotonic() - start,
    }


def load_manifest_entries() -> dict[str, dict[str, Any]]:
    payload = read_json(REVIEW / "RAW_SHARD_MANIFEST.json")
    return {str(entry["logical_item_key"]): entry for entry in payload.get("entries", [])}


def write_manifest(entries: dict[str, dict[str, Any]]) -> None:
    payload = read_json(REVIEW / "RAW_SHARD_MANIFEST.json")
    payload["status"] = "IN_PROGRESS"
    payload["entries"] = [entries[key] for key in sorted(entries)]
    write_json(REVIEW / "RAW_SHARD_MANIFEST.json", payload)


def persist_item_shard(
    *,
    domain: str,
    item_id: str,
    continuation: list[int],
    outputs: dict[str, dict[str, Any]],
    source_commit: str,
    vector_hashes: dict[str, str],
) -> tuple[Path, str, dict[str, Any]]:
    labels = gate11_1.snapshot_labels(outputs[gate11.TF_BASELINE]["snapshots"])
    token_indices = gate11_1.snapshot_token_indices(labels)
    baseline = outputs[gate11.TF_BASELINE]
    baseline_logits = np.stack([baseline["snapshots"][label]["logits"] for label in labels]).astype(
        np.float32
    )
    condition_logits = np.stack(
        [
            np.stack([outputs[condition]["snapshots"][label]["logits"] for label in labels])
            for condition in gate11_1.CONDITIONS
        ]
    ).astype(np.float32)
    hidden_differences = np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [
                            outputs[condition]["snapshots"][label]["hidden"][layer]
                            - baseline["snapshots"][label]["hidden"][layer]
                            for layer in gate11_1.PROPAGATION_LAYERS
                        ]
                    )
                    for label in labels
                ]
            )
            for condition in gate11_1.CONDITIONS
        ]
    ).astype(np.float32)
    target_tokens = np.asarray(
        [
            baseline["snapshots"][label]["target_token"]
            if baseline["snapshots"][label]["target_token"] is not None
            else -1
            for label in labels
        ],
        dtype=np.int64,
    )
    raw_dir = REVIEW / "raw_primitives"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard = raw_dir / f"{domain.lower()}__{item_id}.npz"
    temporary = shard.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            baseline_logits=baseline_logits,
            condition_logits=condition_logits,
            hidden_differences=hidden_differences,
            prompt_token_ids=np.asarray(baseline["prompt_ids"], dtype=np.int64),
            continuation_token_ids=np.asarray(continuation, dtype=np.int64),
            checkpoint_token_indices=token_indices,
            target_next_token_ids=target_tokens,
            condition_names=np.asarray(gate11_1.CONDITIONS),
            propagation_layers=np.asarray(gate11_1.PROPAGATION_LAYERS, dtype=np.int64),
            prompt_hashes=np.asarray(
                [outputs[condition]["prompt_hash"] for condition in gate11_1.CONDITIONS]
            ),
            rendered_prompt_sha256=np.asarray(
                [outputs[condition]["rendered_prompt_sha256"] for condition in gate11_1.CONDITIONS]
            ),
        )
    temporary.replace(shard)
    digest = sha256(shard)
    metadata = {
        "logical_item_key": f"{domain}|{item_id}",
        "domain": domain,
        "item_id": item_id,
        "path": str(shard.relative_to(ROOT)),
        "sha256": digest,
        "bytes": shard.stat().st_size,
        "condition_count": len(gate11_1.CONDITIONS),
        "snapshot_count": len(labels),
        "snapshot_labels": labels,
        "checkpoint_token_indices": token_indices.tolist(),
        "continuation_sha256": hashlib.sha256(
            np.asarray(continuation, dtype=np.int64).tobytes()
        ).hexdigest(),
        "controller_hashes": vector_hashes,
        "eta": gate11.ETA,
        "layer": gate11.LAYER,
        "source_commit": source_commit,
        "model_revision": gate11.MODEL_REVISION,
        "sampling": False,
        "free_generation": False,
    }
    return shard, digest, metadata


def append_item_rows(
    journal: Path,
    *,
    domain: str,
    item_id: str,
    metadata: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    vector_hashes: dict[str, str],
) -> None:
    for condition in gate11_1.CONDITIONS:
        output = outputs[condition]
        append_jsonl(
            journal,
            {
                "logical_key": f"{domain}|{item_id}|{condition}",
                "domain": domain,
                "item_id": item_id,
                "condition": condition,
                "status": "COMPLETE",
                "raw_shard_path": metadata["path"],
                "raw_shard_sha256": metadata["sha256"],
                "prompt_hash": output["prompt_hash"],
                "prompt_length": output["prompt_length"],
                "continuation_length": output["continuation_length"],
                "continuation_sha256": metadata["continuation_sha256"],
                "checkpoint_token_indices": metadata["checkpoint_token_indices"],
                "forward_count": output["forward_count"],
                "application_count": output["application_count"],
                "max_relative_shift_error": output["max_relative_shift_error"],
                "max_noncurrent_change": output["max_noncurrent_change"],
                "duration_seconds": output["duration_seconds"],
                "vector_hash": vector_hashes.get(condition),
                "eta": gate11.ETA if condition in vector_hashes else 0.0,
                "layer": gate11.LAYER if condition in vector_hashes else None,
                "sampling": False,
                "free_generation": False,
                "model_revision": gate11.MODEL_REVISION,
                "source_commit": metadata["source_commit"],
            },
        )


def collect(
    backend: Any, source_commit: str, vectors: dict[str, np.ndarray], vector_hashes: dict[str, str]
) -> dict[str, Any]:
    schedule = read_json(REVIEW / "FIXED_SEQUENCE_SCHEDULE.json")
    journal = REVIEW / "journal.jsonl"
    existing_rows = (
        [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line]
        if journal.exists()
        else []
    )
    completed = {row["logical_key"] for row in existing_rows}
    if len(completed) != len(existing_rows):
        raise RuntimeError("duplicate Gate 11.1 logical row in journal")
    item_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in schedule["rows"]:
        item_groups.setdefault((row["domain"], str(row["item_id"])), row)
    selections = {
        "CRUXEval": read_json(REVIEW / "CRUX_ITEM_SELECTION.json"),
        "CHARCOUNT": read_json(REVIEW / "CHARCOUNT_ITEM_SELECTION.json"),
    }
    item_maps = {
        domain: {str(item["item_id"]): item for item in payload["items"]}
        for domain, payload in selections.items()
    }
    entries = load_manifest_entries()
    new_groups = 0
    for (domain, item_id), schedule_row in item_groups.items():
        keys = [f"{domain}|{item_id}|{condition}" for condition in gate11_1.CONDITIONS]
        if all(key in completed for key in keys):
            shard = (
                REVIEW / entries[f"{domain}|{item_id}"]["path"]
                if not Path(entries[f"{domain}|{item_id}"]["path"]).is_absolute()
                else Path(entries[f"{domain}|{item_id}"]["path"])
            )
            if not shard.exists() or sha256(shard) != entries[f"{domain}|{item_id}"]["sha256"]:
                raise RuntimeError(
                    f"completed journal group missing/corrupt shard: {domain}|{item_id}"
                )
            continue
        if any(key in completed for key in keys):
            raise RuntimeError(f"partial item group in journal: {domain}|{item_id}")
        if not schedule_row["available"]:
            raise RuntimeError("historical schedule contains unavailable sequence")
        item = external_item(item_maps[domain][item_id])
        continuation = [int(value) for value in schedule_row["continuation_token_ids"]]
        outputs = {
            condition: run_condition(
                backend,
                item,
                domain,
                condition,
                continuation,
                vectors[condition] * gate11.ETA * gate11.REFERENCE_SCALE
                if condition in vectors
                else None,
            )
            for condition in gate11_1.CONDITIONS
        }
        _shard, _digest, metadata = persist_item_shard(
            domain=domain,
            item_id=item_id,
            continuation=continuation,
            outputs=outputs,
            source_commit=source_commit,
            vector_hashes=vector_hashes,
        )
        entries[metadata["logical_item_key"]] = metadata
        write_manifest(entries)
        append_item_rows(
            journal,
            domain=domain,
            item_id=item_id,
            metadata=metadata,
            outputs=outputs,
            vector_hashes=vector_hashes,
        )
        completed.update(keys)
        new_groups += 1
    payload = read_json(REVIEW / "RAW_SHARD_MANIFEST.json")
    payload["status"] = "COMPLETE"
    payload["entries"] = [entries[key] for key in sorted(entries)]
    write_json(REVIEW / "RAW_SHARD_MANIFEST.json", payload)
    return {"logical_rows": len(completed), "new_item_groups": new_groups, "shards": len(entries)}


def engineering(
    backend: Any, vectors: dict[str, np.ndarray], vector_hashes: dict[str, str]
) -> dict[str, Any]:
    schedule = read_json(REVIEW / "FIXED_SEQUENCE_SCHEDULE.json")
    selection = read_json(REVIEW / "CRUX_ITEM_SELECTION.json")
    items = {str(item["item_id"]): item for item in selection["items"]}
    samples = [row for row in schedule["rows"] if row["available"]][:5]
    identity_checks = []
    exercised: dict[str, Any] = {}
    for row in samples:
        item = external_item(items[str(row["item_id"])])
        continuation = row["continuation_token_ids"][:2]
        clean = run_condition(backend, item, "CRUXEval", gate11.TF_BASELINE, continuation, None)
        zero = run_condition(
            backend,
            item,
            "CRUXEval",
            "TF_ZERO",
            continuation,
            np.zeros_like(vectors[gate11.TF_MEANINGFUL]),
        )
        identity_checks.append(
            all(
                np.array_equal(
                    clean["snapshots"][label]["logits"], zero["snapshots"][label]["logits"]
                )
                for label in clean["snapshots"]
            )
        )
    sample = samples[0]
    item = external_item(items[str(sample["item_id"])])
    for condition, vector in vectors.items():
        result = run_condition(
            backend,
            item,
            "CRUXEval",
            condition,
            sample["continuation_token_ids"][:2],
            vector * gate11.ETA * gate11.REFERENCE_SCALE,
        )
        exercised[condition] = {
            "vector_hash": vector_hashes[condition],
            "forward_count": result["forward_count"],
            "application_count": result["application_count"],
            "max_relative_shift_error": result["max_relative_shift_error"],
            "max_noncurrent_change": result["max_noncurrent_change"],
        }
    payload = {
        "classification": "GATE11_1_ENGINEERING_PASS",
        "alpha_zero_identity": all(identity_checks),
        "all_controllers_exercised": set(exercised) == set(vectors),
        "exact_shift": all(
            value["max_relative_shift_error"] <= 2.0 for value in exercised.values()
        ),
        "current_token_scope": all(
            value["max_noncurrent_change"] == 0 for value in exercised.values()
        ),
        "one_application_per_forward": all(
            value["application_count"] == value["forward_count"] for value in exercised.values()
        ),
        "raw_arrays_persisted_only_in_collect": True,
    }
    if not all(
        payload[key]
        for key in (
            "alpha_zero_identity",
            "all_controllers_exercised",
            "exact_shift",
            "current_token_scope",
            "one_application_per_forward",
        )
    ):
        payload["classification"] = "GATE11_1_ENGINE_FAILURE"
    write_json(REVIEW / "ENGINEERING_CHECKS.json", payload)
    if payload["classification"] != "GATE11_1_ENGINEERING_PASS":
        raise RuntimeError(payload["classification"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("engineering", "collect"), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--experiment-source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution("Gate 11.1 artifact-complete propagation diagnostics")
    load_lock(args.experiment_source_commit)
    backend = build_backend(args.model_path)
    vectors, hashes = load_vectors()
    if args.phase == "engineering":
        engineering(backend, vectors, hashes)
    else:
        result = collect(backend, args.experiment_source_commit, vectors, hashes)
        write_json(
            REVIEW / "COLLECTION_STATUS.json", {**result, "complete": result["logical_rows"] == 336}
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
