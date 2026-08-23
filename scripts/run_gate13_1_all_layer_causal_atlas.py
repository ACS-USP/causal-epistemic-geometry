#!/usr/bin/env python3
"""Crash-safe staged runner for Gate 13.1."""

from __future__ import annotations

import argparse
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

import run_gate13_cross_model_ministral3 as parent_runner  # noqa: E402

from epistemic_geometry.benchmarks.external.semantic_v3 import PARSER_VERSION  # noqa: E402
from epistemic_geometry.experiments import gate13, gate13_1  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402

REVIEW = ROOT / "review/gate13_1_all_layer_causal_atlas"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def source_is_ancestor(source_commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def load_lock(source_commit: str) -> dict[str, Any]:
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    if lock["status"] != "FROZEN_PRE_STAGE_A":
        raise RuntimeError("Gate 13.1 protocol is not prospectively frozen")
    if lock["experiment_source_commit"] != source_commit or not source_is_ancestor(
        source_commit
    ):
        raise RuntimeError("Gate 13.1 source commit mismatch")
    parser = read_json(ROOT / "review/gate13_cross_model_ministral3/RESPONSE_PARSER_LOCK.json")
    semantic_path = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    if parser["version"] != PARSER_VERSION or parser["module_sha256"] != gate13.file_sha256(
        semantic_path
    ):
        raise RuntimeError("Gate 13.1 parser provenance mismatch")
    source = read_json(REVIEW / "SOURCE_DIRECTION_MANIFEST.json")
    for record in source["layers"]:
        if gate13.file_sha256(ROOT / record["vector_path"]) != record["file_sha256"]:
            raise RuntimeError("Gate 13.1 source vector file hash mismatch")
    return lock


def completed_keys(path: Path, source_commit: str) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row["experiment_source_commit"] != source_commit:
            raise RuntimeError("Gate 13.1 journal mixes source commits")
        keys.append(
            (
                str(row["stage"]),
                str(row["item_id"]),
                str(row["condition"]),
                int(row["rollout_index"]),
            )
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 13.1 journal contains duplicate logical keys")
    return set(keys)


def stage_paths(stage: str) -> tuple[Path, Path]:
    return {
        "sweep": (
            REVIEW / "ALL_LAYER_SWEEP_ITEMS.json",
            REVIEW / "ALL_LAYER_SWEEP_SCHEDULE.json",
        ),
        "layer-dose": (
            REVIEW / "LAYER_DOSE_ITEMS.json",
            REVIEW / "LAYER_DOSE_SCHEDULE.json",
        ),
        "final": (
            REVIEW / "FINAL_EVALUATION_ITEMS.json",
            REVIEW / "FINAL_EVALUATION_SCHEDULE.json",
        ),
    }[stage]


def source_layer_record(layer: int) -> dict[str, Any]:
    manifest = read_json(REVIEW / "SOURCE_DIRECTION_MANIFEST.json")
    return next(row for row in manifest["layers"] if int(row["layer"]) == layer)


def condition_delta(
    stage: str, condition: str
) -> tuple[int | None, np.ndarray | None, str | None, float | None]:
    if condition in {"BASELINE", "TEXTUAL_CAREFUL"}:
        return None, None, None, None
    if stage == "sweep":
        layer = int(condition.split("_L", 1)[1].split("_", 1)[0])
        source = source_layer_record(layer)
        vector = np.load(ROOT / source["vector_path"], allow_pickle=False).astype(np.float64)
        alpha = 0.5 * float(source["D100"])
        return layer, vector * alpha, source["canonical_vector_hash"], alpha
    if stage == "layer-dose":
        record = read_json(REVIEW / "STAGE_B_DIRECTION_MANIFEST.json")["conditions"]
        selected = next(row for row in record if row["condition"] == condition)
        vector = np.load(REVIEW / selected["vector_path"], allow_pickle=False).astype(
            np.float64
        )
        alpha = float(selected["alpha"])
        return int(selected["layer"]), vector * alpha, selected["vector_hash"], alpha
    selected = read_json(REVIEW / "SELECTED_LAYER_DOSE_LOCK.json")
    layer = int(selected["selected_layer"])
    alpha = float(selected["selected_alpha"])
    if condition == "MEANINGFUL_SELECTED":
        path = ROOT / selected["meaningful_vector_path"]
        vector_hash = selected["meaningful_vector_hash"]
    else:
        name = condition.removeprefix("RANDOM_")
        record = read_json(REVIEW / "FINAL_RANDOM_BANK.json")["records"][name]
        path = REVIEW / record["vector_path"]
        vector_hash = record["vector_hash"]
    vector = np.load(path, allow_pickle=False).astype(np.float64)
    return layer, vector * alpha, vector_hash, alpha


def collect(backend: Any, stage: str, source_commit: str) -> None:
    manifest_path, schedule_path = stage_paths(stage)
    items = parent_runner.load_external(manifest_path)
    item_by_id = {item.item_id: item for item in items}
    schedule = read_json(schedule_path)
    journal = REVIEW / "journal.jsonl"
    completed = completed_keys(journal, source_commit)
    for schedule_index, planned in enumerate(schedule):
        key = (
            str(planned["stage"]),
            str(planned["item_id"]),
            str(planned["condition"]),
            int(planned["rollout_index"]),
        )
        if key in completed:
            continue
        item = item_by_id[key[1]]
        system = gate13.SOURCE_CAREFUL if key[2] == "TEXTUAL_CAREFUL" else None
        model_row = parent_runner.model_item(item, system)
        prompt_ids, rendered_hash = parent_runner.prompt_tokens(backend, model_row)
        layer, delta, vector_hash, alpha = condition_delta(stage, key[2])
        if delta is None:
            context: Any = nullcontext()
        else:
            tensor = backend.torch.tensor(
                delta, dtype=backend.torch.float32, device=backend.device
            ).view(1, 1, -1)
            context = Gate6HookTrace(
                layers={int(layer): backend.layer_module(int(layer))},
                deltas={int(layer): tensor},
                target_positions=[len(prompt_ids) - 1],
            )
        started = time.perf_counter()
        with context as trace:
            output, vision_calls = parent_runner.generate_with_vision_audit(
                backend,
                model_row,
                seed=int(planned["seed"]),
                max_new_tokens=gate13.MAX_NEW_TOKENS,
                intervention_metadata={
                    "experiment_id": gate13_1.EXPERIMENT_ID,
                    "stage": planned["stage"],
                    "condition": key[2],
                    "intervention": key[2] if delta is not None else "none",
                    "intervention_layer": layer,
                    "intervention_vector_hash": vector_hash,
                    "intervention_alpha": alpha,
                    "intervention_duration": (
                        "sustained_current_token" if delta is not None else "none"
                    ),
                    "parser_version": PARSER_VERSION,
                    "environment_profile": "CORE_MINISTRAL3",
                    "vision_invoked": False,
                },
            )
        elapsed = time.perf_counter() - started
        metadata = dict(output.metadata)
        if delta is not None:
            metadata["intervention_forward_trace"] = trace.metadata()
        token_count = int(metadata["generated_token_count"])
        scored = parent_runner.score(output.raw_output, item.reference_answer, token_count)
        record = {
            **planned,
            **scored,
            "model": gate13_1.MODEL,
            "model_revision": gate13_1.REVISION,
            "tokenizer_revision": gate13_1.REVISION,
            "experiment_source_commit": source_commit,
            "runtime_commit": git_commit(),
            "raw_output": output.raw_output,
            "generated_token_ids": metadata["generated_token_ids"],
            "generated_token_count": token_count,
            "reference_answer": item.reference_answer,
            "reference_canonical_type": item.metadata["reference_canonical_type"],
            "prompt_hash": item.prompt_hash,
            "rendered_prompt_hash": rendered_hash,
            "parser_version": PARSER_VERSION,
            "layer": layer,
            "vector_hash": vector_hash,
            "alpha": alpha,
            "vision_tower_calls": vision_calls,
            "backend_metadata": metadata,
            "elapsed_seconds": elapsed,
            "schedule_index": schedule_index,
            "retry_count": 0,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        append_jsonl(journal, record)
        completed.add(key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("engineering", "sweep", "layer-dose", "final"),
        required=True,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution(f"Gate 13.1 {args.mode}")
    load_lock(args.source_commit)
    backend = parent_runner.build_backend(args.model_path, gate13_1.MODEL, gate13_1.REVISION)
    if args.mode == "engineering":
        parent_runner.engineering_gate(
            backend, REVIEW, gate13_1.MODEL, gate13_1.REVISION
        )
    else:
        collect(backend, args.mode, args.source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
