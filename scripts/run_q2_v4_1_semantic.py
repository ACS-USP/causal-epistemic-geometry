#!/usr/bin/env python3
"""Frozen Q2 V4.1 semantic collector.

This module has two intentionally separate responsibilities:

* ``preflight`` validates the already sealed V4.1 objects and records a
  machine-readable authorization/preflight manifest without loading a model.
* ``collect`` consumes the hash-pinned schedule on Spark 1 and writes raw,
  unscored trajectories to an append-only journal.  It deliberately imports no
  semantic parser and performs no correctness inspection during collection.

The analysis phase is a separate post-completion operation so that a partial
campaign cannot accidentally become a scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.external.base import ExternalItem  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.research.reliability import CrashSafeJournal  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

REVIEW = ROOT / "review/q2_v4_1_prediction_lock"
EXECUTION_REVIEW = ROOT / "review/q2_v4_1_semantic_execution"
PROTOCOL = REVIEW / "PROTOCOL_LOCK.json"
NORMATIVE = REVIEW / "Q2_V4_1_NORMATIVE_EXECUTION_AND_ANALYSIS_LOCK.json"
PANEL = REVIEW / "SEMANTIC_PANEL_MANIFEST.json"
SCHEDULE = REVIEW / "FUTURE_SEMANTIC_SCHEDULE.json"
SAFE_BANK = ROOT / "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
HISTORICAL_CANDIDATES = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json"
JOURNAL = EXECUTION_REVIEW / "journal.jsonl"
MAX_NEW_TOKENS = 4096
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_ENVIRONMENT = "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
EXPECTED_BRANCH = "experiment/q2-v4-1-prediction-lock"
EXPECTED_PARENT_HEAD = "8f0a6db7ca0f9fbac3b4530f200a8ed01e2b5156"
EXPERIMENT_SOURCE_COMMIT = "4b6423dd909d3f7f10ffd27acd64c06c9f97c3dc"
PARSER_VERSION = "external-semantic-v3"
KEY_FIELDS = ("item_id", "condition", "rollout_index")
MAX_INFRASTRUCTURE_ATTEMPTS = 3


def model_item(item: ExternalItem) -> BenchmarkItem:
    """Convert one frozen panel row without importing an outcome scorer."""

    return BenchmarkItem(
        id=item.item_id,
        prompt=item.prompt,
        target=item.reference_answer,
        metadata={
            "source_prompt_hash": item.prompt_hash,
            "response_channel": item.metadata.get("response_channel", "cruxeval_semantic"),
        },
    )


def prompt_tokens(backend: HuggingFaceBackend, item: BenchmarkItem) -> tuple[list[int], str, str]:
    encoded, rendered, prompt_hash = backend._encode_item(item)  # noqa: SLF001
    values = encoded["input_ids"][0].detach().cpu().tolist()
    return [int(value) for value in values], rendered, prompt_hash


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


def git_branch() -> str:
    return subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def build_backend(model_path: str) -> HuggingFaceBackend:
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL,
        model_path=model_path,
        model_revision=MODEL_REVISION,
        tokenizer_id=model_path,
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        layer=27,
        layer_path="model.model.layers",
        prompt_mode="chat",
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        enable_thinking=False,
        attention_implementation="sdpa",
        execution_mode="serial_reference",
        batch_size=1,
        item_batch_size=1,
        condition_chunk_size=1,
    )
    return HuggingFaceBackend(
        config,
        model_identifier=MODEL,
        tokenizer_identifier=model_path,
        model_revision=MODEL_REVISION,
    )


def verify_spark1_environment(model_path: str) -> dict[str, Any]:
    """Fail closed before loading weights unless the locked Spark-1 profile holds."""

    if platform.node().split(".", 1)[0] != "spark1" or platform.machine() != "aarch64":
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT")
    if not model_path.startswith("/srv/shared/modelos/") or not Path(model_path).is_dir():
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT")
    if os.environ.get("CEG_EXECUTION_PROFILE") != "SPARK1":
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT")
    if not str(Path.cwd().resolve()).startswith("/home/gabriel.alexandre/projects/"):
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - remote-only guard
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT") from exc
    if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT")
    if importlib.metadata.version("transformers") != "4.57.6":
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT")
    device_name = torch.cuda.get_device_name(0)
    if "GB10" not in device_name or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT")
    return {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "gpu": device_name,
        "cuda_device_count": int(torch.cuda.device_count()),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "model_path": model_path,
        "qualified_environment_fingerprint": EXPECTED_ENVIRONMENT,
    }


def load_items() -> dict[str, ExternalItem]:
    payload = read_json(PANEL)
    if payload["status"] != "FROZEN_CONTENT_NOT_AUTHORIZED_FOR_INFERENCE":
        raise RuntimeError("Q2_V4_1_SEMANTIC_ENVIRONMENT_DRIFT: panel status changed")
    items: dict[str, ExternalItem] = {}
    for row in payload["items"]:
        item = ExternalItem(
            item_id=str(row["item_id"]),
            benchmark="CRUXEval",
            subtask="output_prediction",
            prompt=str(row["prompt"]),
            reference_answer=str(row["reference_answer"]),
            evaluator="python_literal",
            source_revision=str(payload["dataset_revision"]),
            metadata={
                "official_index": int(row["official_index"]),
                "provenance_class": row["provenance_class"],
                "reference_type": row["role"],
            },
        )
        if item.item_id in items:
            raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: duplicate panel item")
        if sha256_bytes(item.prompt.encode("utf-8")) != row["prompt_sha256"]:
            raise RuntimeError(f"Q2_V4_1_INSTRUMENT_FAILURE: prompt hash {item.item_id}")
        if sha256_bytes(item.reference_answer.encode("utf-8")) != row["reference_sha256"]:
            raise RuntimeError(f"Q2_V4_1_INSTRUMENT_FAILURE: reference hash {item.item_id}")
        items[item.item_id] = item
    if list(items) != payload["item_ids"] or len(items) != 300:
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: panel order/count")
    return items


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_schedule() -> list[dict[str, Any]]:
    payload = read_json(SCHEDULE)
    if payload["status"] != "FROZEN_NOT_AUTHORIZED_NOT_RUN":
        raise RuntimeError("Q2_V4_1_POST_OPENING_PROTOCOL_DEFECT: schedule status changed")
    rows = list(payload["rows"])
    keys = [(row["item_id"], row["condition"], int(row["rollout_index"])) for row in rows]
    if len(rows) != 37_800 or len(set(keys)) != 37_800:
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: schedule completeness")
    if len({int(row["seed"]) for row in rows}) != 37_800:
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: seed uniqueness")
    if {int(row["rollout_index"]) for row in rows} != {0, 1}:
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: rollout blocks")
    return rows


def load_vectors() -> dict[str, np.ndarray]:
    safe = read_json(SAFE_BANK)
    historical = read_json(HISTORICAL_CANDIDATES)
    safe_ids = [str(value) for value in safe["candidate_order"]]
    if len(safe_ids) != 31 or safe["safe_count"] != 31:
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: safe-bank count")
    historical_by_id = {str(row["candidate_id"]): row for row in historical["candidates"]}
    vectors: dict[str, np.ndarray] = {}
    for candidate_id in safe_ids:
        row = historical_by_id[candidate_id]
        path = ROOT / row["path"]
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"Q2_V4_1_INSTRUMENT_FAILURE: vector file {candidate_id}")
        vector = np.load(path, allow_pickle=False).astype(np.float64)
        from epistemic_geometry.experiments.gate6 import vector_sha256

        if vector_sha256(vector) != row["canonical_vector_hash"]:
            raise RuntimeError(f"Q2_V4_1_INSTRUMENT_FAILURE: vector canonical hash {candidate_id}")
        vectors[candidate_id] = vector
    return vectors


def required_hashes() -> dict[str, str]:
    lock = read_json(PROTOCOL)
    hashes = dict(lock["artifact_hashes"])
    hashes["PROTOCOL_LOCK.json"] = (
        "0adc2d04e314bca4bf488595cdbd171da1a47f439b90170cb8125c9def35d278"
    )
    hashes["Q2_V4_1_NORMATIVE_EXECUTION_AND_ANALYSIS_LOCK.json"] = (
        "ee70e60b9d2f64ee8d4a8e59afb8635a80f20bfd06da72600703e50fbdf33d8a"
    )
    hashes["SAFE_31_IMMUTABLE_MANIFEST.json"] = (
        "a641d612628c4f9eff2ae9fdf12d3ad17af5a3e921ec726d31c208ee5e030447"
    )
    return hashes


def validate_frozen_objects() -> dict[str, Any]:
    lock = read_json(PROTOCOL)
    normative = read_json(NORMATIVE)
    if git_branch() != EXPECTED_BRANCH:
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: branch")
    if lock["status"] != "Q2_V4_1_PRESEMANTIC_PROTOCOL_LOCKED":
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: protocol status")
    if lock["semantic_outcomes"] != 0 or lock["correctness_inspected"]:
        raise RuntimeError("Q2_V4_1_POST_OPENING_PROTOCOL_DEFECT: semantic firewall")
    if normative["status"] != "Q2_V4_1_NORMATIVE_LOCK_COMPLETE":
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: normative status")
    if normative["semantic_firewall"]["v4_v4_1_semantic_trajectories"] != 0:
        raise RuntimeError("Q2_V4_1_POST_OPENING_PROTOCOL_DEFECT: normative firewall")
    paths = {
        "PROTOCOL_LOCK.json": PROTOCOL,
        "Q2_V4_1_NORMATIVE_EXECUTION_AND_ANALYSIS_LOCK.json": NORMATIVE,
        "SAFE_31_IMMUTABLE_MANIFEST.json": SAFE_BANK,
        "FUTURE_SEMANTIC_SCHEDULE.json": SCHEDULE,
        "SEMANTIC_PANEL_MANIFEST.json": PANEL,
        "QAP_CONTROLLER_PERMUTATIONS.npy": REVIEW / "QAP_CONTROLLER_PERMUTATIONS.npy",
        "QAP_SCHEDULE.json": REVIEW / "QAP_SCHEDULE.json",
        "PREDICTION_MATRICES.npz": REVIEW / "PREDICTION_MATRICES.npz",
        "PREDICTION_MATRIX_METADATA.json": REVIEW / "PREDICTION_MATRIX_METADATA.json",
    }
    observed_hashes: dict[str, str] = {}
    for name, expected in required_hashes().items():
        path = paths.get(name, REVIEW / name)
        if not path.is_file():
            raise RuntimeError(f"Q2_V4_1_INSTRUMENT_FAILURE: missing {path}")
        observed_hashes[name] = sha256_file(path)
        if observed_hashes[name] != expected:
            raise RuntimeError(f"Q2_V4_1_INSTRUMENT_FAILURE: hash {path}")
    items = load_items()
    rows = load_schedule()
    vectors = load_vectors()
    scheduled_conditions = {str(row["condition"]) for row in rows}
    expected_conditions = {"BASELINE"}
    expected_conditions.update(
        f"{candidate}_{shell}" for candidate in vectors for shell in ("MEDIUM", "STRONG")
    )
    if scheduled_conditions != expected_conditions:
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: condition set")
    if {row["item_id"] for row in rows} != set(items):
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: schedule/panel mismatch")
    for row in rows:
        item = items[row["item_id"]]
        if row["prompt_sha256"] != sha256_bytes(item.prompt.encode("utf-8")):
            raise RuntimeError(f"Q2_V4_1_INSTRUMENT_FAILURE: schedule prompt {row['item_id']}")
        if row["condition"] != "BASELINE":
            candidate = row["condition"].rsplit("_", 1)[0]
            from epistemic_geometry.experiments.gate6 import vector_sha256

            if row["controller_vector_hash"] != vector_sha256(vectors[candidate]):
                raise RuntimeError(f"Q2_V4_1_INSTRUMENT_FAILURE: schedule vector {candidate}")
            if int(row["layer"]) != 27 or row["duration"] != "sustained_current_token":
                raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: intervention metadata")
    return {
        "branch": git_branch(),
        "head": git_head(),
        "expected_parent_head": EXPECTED_PARENT_HEAD,
        "hashes": observed_hashes,
        "panel_count": len(items),
        "schedule_count": len(rows),
        "unique_logical_keys": len(
            {(r["item_id"], r["condition"], r["rollout_index"]) for r in rows}
        ),
        "unique_seeds": len({int(r["seed"]) for r in rows}),
        "conditions": sorted(scheduled_conditions),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "BF16",
        "attention": "SDPA",
        "layer": 27,
        "parser_version": PARSER_VERSION,
        "environment_fingerprint_expected": EXPECTED_ENVIRONMENT,
        "semantic_outcomes_before_execution": 0,
        "correctness_inspected_before_execution": False,
        "spark1_only": True,
        "spark2_used": False,
        "runpod_used": False,
        "authorization": "PRINCIPAL_AUTHORIZATION_Q2_V4_1_SEMANTIC_EXECUTION",
    }


def record_preflight() -> None:
    payload = validate_frozen_objects()
    payload.update(
        {
            "schema_version": "q2-v4.1-semantic-preflight-v1",
            "status": "AUTHORIZED_PRE_EXECUTION_NO_SEMANTIC_OUTPUTS",
            "journal_path": str(JOURNAL.relative_to(ROOT)),
            "journal_exists_before_first_output": JOURNAL.exists(),
            "journal_sha256_before_first_output": sha256_file(JOURNAL)
            if JOURNAL.exists()
            else None,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    if JOURNAL.exists() and JOURNAL.stat().st_size:
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: pre-existing semantic completions")
    write_json(EXECUTION_REVIEW / "PRE_EXECUTION_MANIFEST.json", payload)
    print(json.dumps({"status": payload["status"], "rows": payload["schedule_count"]}))


def build_identity(code_commit: str) -> dict[str, Any]:
    return {
        "experiment_id": "Q2_V4_1_SEMANTIC_EXECUTION",
        "phase": "SEMANTIC_EXECUTION",
        "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
        "code_commit": code_commit,
        "protocol_lock_sha256": sha256_file(PROTOCOL),
        "normative_lock_sha256": sha256_file(NORMATIVE),
        "schedule_sha256": sha256_file(SCHEDULE),
        "panel_sha256": sha256_file(PANEL),
        "semantic_outcomes_during_collection": "NOT_INSPECTED",
    }


def condition_context(
    backend: HuggingFaceBackend,
    item: ExternalItem,
    row: dict[str, Any],
    vectors: dict[str, np.ndarray],
) -> tuple[Any, BenchmarkItem, dict[str, Any]]:
    model_row = model_item(item)
    prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, model_row)
    if row["condition"] == "BASELINE":
        return (
            nullcontext(),
            model_row,
            {
                "intervention": "none",
                "layer": None,
                "duration": "none",
                "prompt_length": len(prompt_ids),
                "prompt_hash": prompt_hash,
                "alpha": 0.0,
                "controller_vector_hash": None,
            },
        )
    candidate = str(row["condition"].rsplit("_", 1)[0])
    delta = backend.torch.tensor(
        vectors[candidate] * float(row["alpha"]), dtype=backend.torch.float32, device=backend.device
    ).view(1, 1, -1)
    return (
        Gate6HookTrace(
            layers={27: backend.layer_module(27)},
            deltas={27: delta},
            target_positions=[len(prompt_ids) - 1],
        ),
        model_row,
        {
            "intervention": str(row["condition"]),
            "layer": 27,
            "duration": "sustained_current_token",
            "prompt_length": len(prompt_ids),
            "prompt_hash": prompt_hash,
            "alpha": float(row["alpha"]),
            "controller_vector_hash": row["controller_vector_hash"],
        },
    )


def _is_operational_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError, EOFError))


def collect(model_path: str, code_commit: str) -> None:
    preflight = validate_frozen_objects()
    if preflight["head"] != code_commit:
        raise RuntimeError("Q2_V4_1_INSTRUMENT_FAILURE: code commit mismatch")
    items = load_items()
    schedule = load_schedule()
    vectors = load_vectors()
    identity = build_identity(code_commit)
    journal = CrashSafeJournal(JOURNAL, identity=identity, key_fields=KEY_FIELDS)
    if len(journal.rows) > len(schedule):
        raise RuntimeError("Q2_V4_1_POST_OPENING_PROTOCOL_DEFECT: extra journal rows")
    verify_spark1_environment(model_path)
    backend = build_backend(model_path)
    started = time.monotonic()
    for schedule_index, row in enumerate(schedule):
        key = (row["item_id"], row["condition"], int(row["rollout_index"]))
        if key in journal.rows:
            continue
        retry_count = 0
        retry_reasons: list[str] = []
        condition_meta: dict[str, Any] | None = None
        trace: Any | None = None
        while True:
            try:
                context, model_row, condition_meta = condition_context(
                    backend, items[row["item_id"]], row, vectors
                )
                trajectory_started = time.perf_counter()
                with context as trace:
                    output = backend.generate_reasoning(
                        model_row,
                        sampling_seed=int(row["seed"]),
                        max_new_tokens=MAX_NEW_TOKENS,
                        intervention_metadata={
                            "experiment_id": "Q2_V4_1_SEMANTIC_EXECUTION",
                            "phase": "SEMANTIC_EXECUTION",
                            "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
                            "parser_version": PARSER_VERSION,
                            "environment_profile": "CORE_QWEN",
                            **condition_meta,
                        },
                    )
                metadata = dict(output.metadata)
                journal.append(
                    {
                        **row,
                        "schedule_index": schedule_index,
                        "raw_output": output.raw_output,
                        "generated_token_ids": metadata.get("generated_token_ids", []),
                        "generated_token_count": int(metadata.get("generated_token_count", 0)),
                        "truncated": int(metadata.get("generated_token_count", 0))
                        >= MAX_NEW_TOKENS,
                        "condition_metadata": condition_meta,
                        "hook_trace": trace.metadata() if trace is not None else None,
                        "model": MODEL,
                        "model_revision": MODEL_REVISION,
                        "tokenizer_revision": MODEL_REVISION,
                        "parser_version": PARSER_VERSION,
                        "environment_profile": "CORE_QWEN",
                        "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
                        "code_commit": code_commit,
                        "seed": int(row["seed"]),
                        "retry_count": retry_count,
                        "retry_reasons": retry_reasons,
                        "elapsed_seconds": time.perf_counter() - trajectory_started,
                        "runtime_error": None,
                        "semantic_scoring": "DEFERRED_UNTIL_COMPLETE",
                    }
                )
                break
            except Exception as exc:  # noqa: BLE001 - preserve operational boundary below
                if _is_operational_retryable(exc) and retry_count + 1 < MAX_INFRASTRUCTURE_ATTEMPTS:
                    retry_reasons.append(type(exc).__name__)
                    retry_count += 1
                    continue
                # A generation/runtime failure is a terminal scientific row.  A
                # process crash before this append is recovered by resume and is
                # therefore never confused with a recorded model outcome.
                journal.append(
                    {
                        **row,
                        "schedule_index": schedule_index,
                        "raw_output": "",
                        "generated_token_ids": [],
                        "generated_token_count": 0,
                        "truncated": False,
                        "condition_metadata": condition_meta,
                        "hook_trace": None,
                        "model": MODEL,
                        "model_revision": MODEL_REVISION,
                        "tokenizer_revision": MODEL_REVISION,
                        "parser_version": PARSER_VERSION,
                        "environment_profile": "CORE_QWEN",
                        "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
                        "code_commit": code_commit,
                        "seed": int(row["seed"]),
                        "retry_count": retry_count,
                        "retry_reasons": retry_reasons,
                        "elapsed_seconds": 0.0,
                        "runtime_error": f"{type(exc).__name__}: {exc}",
                        "semantic_scoring": "DEFERRED_UNTIL_COMPLETE",
                    }
                )
                break
        if len(journal.rows) % 100 == 0:
            print(
                json.dumps(
                    {
                        "health": "running",
                        "completed_logical_keys": len(journal.rows),
                        "expected_logical_keys": len(schedule),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if len(journal.rows) != len(schedule):
        raise RuntimeError("Q2_V4_1_SEMANTIC_EXECUTION_INCOMPLETE")
    print(
        json.dumps({"health": "complete", "completed_logical_keys": len(journal.rows)}), flush=True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "collect"), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--code-commit", default=git_head())
    args = parser.parse_args()
    if args.mode == "preflight":
        record_preflight()
        return 0
    if not args.model_path:
        raise RuntimeError("--model-path is required for collect")
    collect(args.model_path, args.code_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
