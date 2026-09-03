#!/usr/bin/env python3
"""Execute the already frozen Q2 OOS V2 semantic schedule.

This is an execution-layer companion to ``run_q2_oos_v2_semantic.py``.  The
original file is a hash-pinned, pre-opening terminal-policy primitive and is
intentionally preserved unchanged.  This module adds only the missing
schedule/journal orchestration; it does not score outputs or import a parser.

``preflight`` performs all frozen-object and Spark-1 environment checks without
loading model weights.  ``collect`` loads the model only after those checks,
then writes private, unscored rows to ``CrashSafeJournal``.  Progress output is
operational only and never prints generated text or semantic fields.
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
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_q2_oos_v2_semantic import (  # noqa: E402
    EXTREME_REPETITION_NAME,
    SEMANTIC_MAX_NEW_TOKENS,
    extreme_mechanical_repetition_v1,
    frozen_terminal_metadata,
)

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.external.base import ExternalItem  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.experiments.gate6 import vector_sha256  # noqa: E402
from epistemic_geometry.research.reliability import (  # noqa: E402
    CrashSafeJournal,
    validate_logical_rows,
)
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402
from epistemic_geometry.types import BackendOutput, BenchmarkItem  # noqa: E402

REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout"
V2_STREAM = ROOT / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
PANEL = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"
PANEL_HASH_KEY = "SEMANTIC_PANEL_MANIFEST.json"
SCHEDULE = REVIEW / "FUTURE_SEMANTIC_SCHEDULE.json"
LOCK = REVIEW / "PREDICTION_LOCK.json"
LOCK_HASHES = REVIEW / "PREDICTION_LOCK_HASHES.json"
INFERENCE_LOCK = REVIEW / "INFERENCE_LOCK.json"
RUNTIME_LOCK = REVIEW / "RUNTIME_MONITOR_LOCK.json"
MATRIX_METADATA = REVIEW / "PREDICTION_MATRIX_METADATA.json"
SELECTED = REVIEW / "V2_SELECTED_CONTROLLER_BANK.json"
CANDIDATE_MANIFEST = V2_STREAM / "V2_CANDIDATE_BANK_MANIFEST.json"
V2_FINAL_PROTOCOL = V2_STREAM / "V2_FINAL_PROTOCOL_LOCK.json"
AMENDED_EXECUTION_LOCK = V2_STREAM / "Q2_OOS_V2_AMENDED_SEMANTIC_EXECUTION_LOCK.json"
MODEL_MANIFEST = ROOT / "review/q2_v4_spark1_presemantic/EXACT_MODEL_MANIFEST.json"

MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_ENVIRONMENT = "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
EXPECTED_BRANCH = "research/q2-fresh-controller-oos-design"
PREDICTION_LOCK_PARENT_HEAD = "170dd50925c35e32a2439576f901bab1cf31eb7d"
EXPECTED_PANEL_SHA256 = "c127cf3594e8ea849dbd038492606b3afaaac406feb4146188769c04d6691187"
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "3d19c8d71fe86329026d113312f5088468f45d956dd860f97d340da60a093479"
)
EXPECTED_V2_FINAL_PROTOCOL_SHA256 = (
    "dd6b327243d29690a451ecf3ea93395406a2d17f6582d9cae0b68e312e51ddd4"
)
EXPECTED_AMENDED_EXECUTION_LOCK_SHA256 = (
    "4e5a6bfbd6af174d4095b68cee4d0e20759ee312320d3d9afaa5eaa6b6eeaa4d"
)
EXPECTED_SELECTED_BANK_SHA256 = "9a544b4ec6d43ec1c3530feb963cd0340db516e82f91a40c2624300483e2e0fd"
EXPECTED_SCHEDULE_SHA256 = "dac5c284b90c726016968f31d25200a362c42d96f63b63d730665f3f47e85ec5"
EXPECTED_LOCK_SHA256 = "825d6e3536b51a31956cbd5c9e75bedfed38f9e3df5da05a4452a5681f65f9bb"
EXPECTED_LOCK_HASHES_SHA256 = "2b02c2a6e0fa14a1d6760e384d726787d54c3e8c66b1d81585be914e500e9f68"
EXPECTED_INFERENCE_LOCK_SHA256 = "a8d9ead49d9265211906a0f367ac3062d03b32c924e8606a2f8c12caaf3fbea1"
EXPECTED_RUNTIME_LOCK_SHA256 = "fadafb50b9c26c42bfb2abd4dcdeb7a93870ac46c3a1e0925b4ba8fc3707ea8d"
EXPECTED_MATRIX_METADATA_SHA256 = "39ceaa889abc30a6740ab60ef4bdd7c24197b6ea2cc5189245bf365b9edd3b06"
EXPECTED_PREDICTION_MATRIX_SHA256 = (
    "b4ec00985e750c5bb8fd7fd49228267ec576bf6c2ad2ac3984f6f2390d927703"
)
EXPECTED_A2_RAW_MANIFEST_SHA256 = "54f4d2b8e3699d4d9bcce3b102a6ca23b6e01112a5b037cf00db6df8beb987d6"
EXPECTED_MODEL_MANIFEST_SHA256 = "cedc88ba2f732baea6bb71f5e6d7f6bc3aad00d302c3456d208a21687c9e069c"
PARSER_VERSION = "external-semantic-v3"
KEY_FIELDS = ("item_id", "condition", "rollout_index")
ROLL_OUTS = (0, 1)
SHELLS = ("MEDIUM", "STRONG")
MAX_INFRASTRUCTURE_ATTEMPTS = 3
EXPECTED_SCHEDULE_ROWS = 19_200
EXPECTED_ITEMS = 300
EXPECTED_CONTROLLERS = 16
EXPECTED_CONDITIONS = EXPECTED_CONTROLLERS * len(SHELLS)


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_qualified_model_bytes(model_path: str) -> dict[str, Any]:
    """Verify every qualified model/tokenizer file before model construction."""

    model_root = Path(model_path)
    if sha256_file(MODEL_MANIFEST) != EXPECTED_MODEL_MANIFEST_SHA256:
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    try:
        manifest = read_json(MODEL_MANIFEST)
        expected_rows = list(manifest["files"])
        expected_paths = {str(row["path"]) for row in expected_rows}
        expected_count = int(manifest["file_count"])
        expected_total_bytes = int(manifest["total_bytes"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT") from exc
    if manifest.get("model") != MODEL or manifest.get("revision") != MODEL_REVISION:
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    if len(expected_paths) != len(expected_rows) or expected_count != len(expected_rows):
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    observed_paths = {
        str(path.relative_to(model_root))
        for path in model_root.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    if observed_paths != expected_paths:
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    observed_total_bytes = 0
    try:
        for row in expected_rows:
            path = model_root / str(row["path"])
            observed_total_bytes += path.stat().st_size
            if path.stat().st_size != int(row["bytes"]):
                raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
            if sha256_file(path) != str(row["sha256"]):
                raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT") from exc
    if observed_total_bytes != expected_total_bytes:
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    return {
        "manifest_sha256": EXPECTED_MODEL_MANIFEST_SHA256,
        "manifest_inner_sha256": str(manifest["manifest_sha256"]),
        "file_count": len(expected_rows),
        "total_bytes": int(manifest["total_bytes"]),
        "model": MODEL,
        "revision": MODEL_REVISION,
    }


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_branch() -> str:
    return subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def model_item(item: ExternalItem) -> BenchmarkItem:
    return BenchmarkItem(
        id=item.item_id,
        prompt=item.prompt,
        target=item.reference_answer,
        metadata={
            "source_prompt_hash": item.metadata["prompt_sha256"],
            "response_channel": "cruxeval_semantic",
        },
    )


def prompt_tokens(backend: HuggingFaceBackend, item: BenchmarkItem) -> tuple[list[int], str, str]:
    encoded, rendered, prompt_hash = backend._encode_item(item)  # noqa: SLF001
    values = encoded["input_ids"][0].detach().cpu().tolist()
    return [int(value) for value in values], rendered, prompt_hash


def load_items() -> dict[str, ExternalItem]:
    payload = read_json(PANEL)
    if payload.get("status") != "FROZEN_CONTENT_NOT_AUTHORIZED_FOR_INFERENCE":
        raise RuntimeError("Q2_OOS_V2_POST_OPENING_PROTOCOL_DEFECT: panel status")
    if len(payload.get("items", [])) != EXPECTED_ITEMS:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: panel count")
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
                "prompt_sha256": row["prompt_sha256"],
            },
        )
        if item.item_id in items:
            raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: duplicate panel item")
        if sha256_bytes(item.prompt.encode("utf-8")) != row["prompt_sha256"]:
            raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: prompt hash {item.item_id}")
        if sha256_bytes(item.reference_answer.encode("utf-8")) != row["reference_sha256"]:
            raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: reference hash {item.item_id}")
        items[item.item_id] = item
    if list(items) != payload["item_ids"]:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: panel order")
    return items


def load_schedule() -> list[dict[str, Any]]:
    payload = read_json(SCHEDULE)
    if payload.get("status") != "FROZEN_NOT_AUTHORIZED_NOT_RUN":
        raise RuntimeError("Q2_OOS_V2_POST_OPENING_PROTOCOL_DEFECT: schedule status")
    rows = list(payload.get("rows", []))
    keys = [(r["item_id"], r["condition"], int(r["rollout_index"])) for r in rows]
    if len(rows) != EXPECTED_SCHEDULE_ROWS or len(set(keys)) != EXPECTED_SCHEDULE_ROWS:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: schedule completeness")
    if len({int(r["seed"]) for r in rows}) != EXPECTED_SCHEDULE_ROWS:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: seed uniqueness")
    if {int(r["rollout_index"]) for r in rows} != set(ROLL_OUTS):
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: rollout set")
    if {str(r["shell"]) for r in rows} != set(SHELLS):
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: shell set")
    return rows


def load_selected_vectors() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = read_json(SELECTED)
    if selected.get("classification") != "Q2_OOS_V2_SELECTED_BANK_GATE_PASS":
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: selected bank status")
    selected_ids = [str(value) for value in selected["selected_ids"]]
    if len(selected_ids) != EXPECTED_CONTROLLERS or len(set(selected_ids)) != EXPECTED_CONTROLLERS:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: selected bank count")
    selected_hashes = {
        str(row["candidate_id"]): str(row["vector_hash"])
        for row in selected["controllers"].values()
    }
    manifest = read_json(CANDIDATE_MANIFEST)
    if manifest.get("classification") != "Q2_OOS_V2_CANDIDATE_STREAM_INTEGRITY_PASS":
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: candidate stream status")
    by_id = {str(row["candidate_id"]): row for row in manifest["candidates"]}
    if len(by_id) != 34:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: candidate stream count")
    vectors: dict[str, np.ndarray] = {}
    for candidate_id in selected_ids:
        row = by_id.get(candidate_id)
        if row is None:
            raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: missing vector {candidate_id}")
        path = ROOT / str(row["path"])
        if not path.is_file() or sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: vector file {candidate_id}")
        vector = np.load(path, allow_pickle=False).astype(np.float64)
        if vector_sha256(vector) != row["vector_array_sha256"]:
            raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: vector hash {candidate_id}")
        if vector_sha256(vector) != selected_hashes[candidate_id]:
            raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: selected vector hash {candidate_id}")
        vectors[candidate_id] = vector
    return vectors, selected


def frozen_file_hashes() -> dict[str, str]:
    lock_hashes = read_json(LOCK_HASHES)
    expected = dict(lock_hashes["files"])
    expected.update(
        {
            str(LOCK_HASHES.relative_to(ROOT)): EXPECTED_LOCK_HASHES_SHA256,
            PANEL_HASH_KEY: EXPECTED_PANEL_SHA256,
            "V2_CANDIDATE_BANK_MANIFEST.json": EXPECTED_CANDIDATE_MANIFEST_SHA256,
            "V2_FINAL_PROTOCOL_LOCK.json": EXPECTED_V2_FINAL_PROTOCOL_SHA256,
            "Q2_OOS_V2_AMENDED_SEMANTIC_EXECUTION_LOCK.json": (
                EXPECTED_AMENDED_EXECUTION_LOCK_SHA256
            ),
            str(MODEL_MANIFEST.relative_to(ROOT)): EXPECTED_MODEL_MANIFEST_SHA256,
        }
    )
    paths: dict[str, Path] = {
        name: ROOT / name for name in lock_hashes["files"]
    }
    paths.update(
        {
            str(LOCK_HASHES.relative_to(ROOT)): LOCK_HASHES,
            PANEL_HASH_KEY: PANEL,
            "V2_CANDIDATE_BANK_MANIFEST.json": CANDIDATE_MANIFEST,
            "V2_FINAL_PROTOCOL_LOCK.json": V2_FINAL_PROTOCOL,
            "Q2_OOS_V2_AMENDED_SEMANTIC_EXECUTION_LOCK.json": AMENDED_EXECUTION_LOCK,
            str(MODEL_MANIFEST.relative_to(ROOT)): MODEL_MANIFEST,
        }
    )
    observed: dict[str, str] = {}
    for name, expected_hash in expected.items():
        path = paths[name]
        if not path.is_file():
            raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: missing {path}")
        observed[name] = sha256_file(path)
        if observed[name] != expected_hash:
            raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: hash {name}")
    return observed


def validate_frozen_objects() -> dict[str, Any]:
    if git_branch() != EXPECTED_BRANCH:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: branch")
    current_head = git_head()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREDICTION_LOCK_PARENT_HEAD, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: prediction-lock ancestry")
    lock = read_json(LOCK)
    inference = read_json(INFERENCE_LOCK)
    if lock.get("status") != "Q2_OOS_V2_READY_FOR_PREDICTION_LOCK":
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: prediction-lock status")
    if lock.get("semantic_trajectories") != 0 or lock.get("correctness_inspected"):
        raise RuntimeError("Q2_OOS_V2_POST_OPENING_PROTOCOL_DEFECT: prediction firewall")
    if inference.get("status") != "FROZEN_NOT_RUN" or inference.get("semantic_outcomes") != 0:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: inference lock status")
    observed_hashes = frozen_file_hashes()
    if observed_hashes[str(LOCK.relative_to(ROOT))] != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: prediction lock hash")
    if observed_hashes[str(LOCK_HASHES.relative_to(ROOT))] != EXPECTED_LOCK_HASHES_SHA256:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: hash-manifest hash")
    if observed_hashes[str(INFERENCE_LOCK.relative_to(ROOT))] != EXPECTED_INFERENCE_LOCK_SHA256:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: inference lock hash")
    if observed_hashes[str(RUNTIME_LOCK.relative_to(ROOT))] != EXPECTED_RUNTIME_LOCK_SHA256:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: runtime lock hash")
    if observed_hashes[str(MATRIX_METADATA.relative_to(ROOT))] != EXPECTED_MATRIX_METADATA_SHA256:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: matrix metadata hash")
    if observed_hashes[
        str((REVIEW / "PREDICTION_MATRICES.npz").relative_to(ROOT))
    ] != EXPECTED_PREDICTION_MATRIX_SHA256:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: prediction matrix hash")
    if observed_hashes[
        str((REVIEW / "A2_FRESH_RAW_ARCHIVE_HASHES.json").relative_to(ROOT))
    ] != EXPECTED_A2_RAW_MANIFEST_SHA256:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: A2 raw manifest hash")
    items = load_items()
    schedule = load_schedule()
    vectors, selected = load_selected_vectors()
    order = [str(value) for value in selected["selected_ids"]]
    expected_conditions = {f"{candidate}_{shell}" for candidate in order for shell in SHELLS}
    if {str(row["condition"]) for row in schedule} != expected_conditions:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: condition set")
    if {row["item_id"] for row in schedule} != set(items):
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: item set")
    per_key_counts: dict[tuple[str, str], int] = {}
    for row in schedule:
        item = items[row["item_id"]]
        if row["prompt_sha256"] != item.metadata["prompt_sha256"]:
            raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: prompt binding {row['item_id']}")
        candidate = str(row["candidate_id"])
        condition = str(row["condition"])
        if candidate not in vectors or condition != f"{candidate}_{row['shell']}":
            raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: controller binding")
        selected_meta = selected["controllers"].get(condition)
        if selected_meta is None:
            raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: selected condition metadata")
        for field in ("candidate_id", "shell", "vector_hash"):
            if str(row[field] if field != "vector_hash" else row["controller_vector_hash"]) != str(
                selected_meta[field]
            ):
                raise RuntimeError(f"Q2_OOS_V2_INSTRUMENT_FAILURE: schedule {field}")
        if not np.isclose(float(row["alpha"]), float(selected_meta["alpha"]), rtol=0.0, atol=0.0):
            raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: schedule alpha")
        if int(row["layer"]) != 27 or row["duration"] != "sustained_current_token":
            raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: intervention metadata")
        pair = (str(row["item_id"]), str(row["condition"]))
        per_key_counts[pair] = per_key_counts.get(pair, 0) + 1
    if set(per_key_counts.values()) != {2} or len(per_key_counts) != 9_600:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: item-condition rollout coverage")
    return {
        "branch": git_branch(),
        "head": current_head,
        "prediction_lock_parent_head": PREDICTION_LOCK_PARENT_HEAD,
        "hashes": observed_hashes,
        "panel_count": len(items),
        "schedule_count": len(schedule),
        "unique_logical_keys": len(
            {(r["item_id"], r["condition"], int(r["rollout_index"])) for r in schedule}
        ),
        "unique_seeds": len({int(r["seed"]) for r in schedule}),
        "fresh_controller_order": order,
        "conditions": sorted(expected_conditions),
        "rollouts": list(ROLL_OUTS),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "BF16",
        "attention": "SDPA",
        "layer": 27,
        "parser_version": PARSER_VERSION,
        "max_new_tokens": SEMANTIC_MAX_NEW_TOKENS,
        "repetition_policy": EXTREME_REPETITION_NAME,
        "repetition_policy_parameters": {
            "minimum_generated_tokens": 256,
            "tail_window_tokens": 1024,
            "maximum_period_tokens": 64,
            "periodic_match_threshold": 0.9,
            "dominant_token_share_threshold": 0.5,
        },
        "environment_fingerprint_expected": EXPECTED_ENVIRONMENT,
        "semantic_outcomes_before_execution": 0,
        "correctness_inspected_before_execution": False,
        "spark1_only": True,
        "spark2_used": False,
        "runpod_used": False,
        "controller_stream_changed": False,
        "redraws": 0,
        "replacements": 0,
        "authorization": "PRINCIPAL_AUTHORIZATION_Q2_OOS_V2_SEMANTIC_EXECUTION",
    }


def verify_spark1_environment(model_path: str) -> dict[str, Any]:
    """Fail closed before model loading unless qualified Spark 1 holds."""

    if platform.node().split(".", 1)[0] != "spark1" or platform.machine() != "aarch64":
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    if not Path(model_path).is_dir():
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    model_bytes = verify_qualified_model_bytes(model_path)
    if os.environ.get("CEG_EXECUTION_PROFILE") != "SPARK1":
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - remote-only
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT") from exc
    if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    if importlib.metadata.version("transformers") != "4.57.6":
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    device_name = torch.cuda.get_device_name(0)
    if "GB10" not in device_name or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
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
        "sdpa_available": True,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "qualified_environment_fingerprint": EXPECTED_ENVIRONMENT,
        "model_bytes": model_bytes,
    }


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
        max_new_tokens=SEMANTIC_MAX_NEW_TOKENS,
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


def condition_context(
    backend: HuggingFaceBackend,
    item: ExternalItem,
    row: dict[str, Any],
    vectors: dict[str, np.ndarray],
) -> tuple[Any, BenchmarkItem, dict[str, Any]]:
    model_row = model_item(item)
    prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, model_row)
    candidate = str(row["candidate_id"])
    delta = backend.torch.tensor(
        vectors[candidate] * float(row["alpha"]),
        dtype=backend.torch.float32,
        device=backend.device,
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
            "controller_vector_hash": str(row["controller_vector_hash"]),
        },
    )


def _is_operational_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError, EOFError))


def build_identity(code_commit: str, execution_environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": "Q2_OOS_V2_SEMANTIC_EXECUTION",
        "phase": "SEMANTIC_EXECUTION",
        "authorization": "PRINCIPAL_AUTHORIZATION_Q2_OOS_V2_SEMANTIC_EXECUTION",
        "prediction_lock_parent_head": PREDICTION_LOCK_PARENT_HEAD,
        "code_commit": code_commit,
        "schedule_sha256": sha256_file(SCHEDULE),
        "panel_sha256": sha256_file(PANEL),
        "selected_controller_bank_sha256": sha256_file(SELECTED),
        "model_revision": MODEL_REVISION,
        "environment_fingerprint": EXPECTED_ENVIRONMENT,
        "parser_version": PARSER_VERSION,
        "semantic_outcomes_during_collection": "NOT_INSPECTED",
        "environment_observation": execution_environment,
    }


def _runtime_error_row(
    row: dict[str, Any],
    schedule_index: int,
    condition_meta: dict[str, Any] | None,
    code_commit: str,
    retry_count: int,
    retry_reasons: list[str],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        **row,
        "schedule_index": schedule_index,
        "raw_output": "",
        "generated_token_ids": [],
        "generated_token_count": 0,
        "truncated": False,
        "terminal_reason": "model_runtime_error",
        "terminal_answer_channel_failure": True,
        "commitment_valid_if_terminal_failure": False,
        "semantic_evaluable_if_terminal_failure": False,
        "binary_error_e_if_terminal_failure": 1,
        "condition_metadata": condition_meta,
        "hook_trace": None,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "parser_version": PARSER_VERSION,
        "environment_profile": "CORE_QWEN",
        "experiment_id": "Q2_OOS_V2_SEMANTIC_EXECUTION",
        "prediction_lock_parent_head": PREDICTION_LOCK_PARENT_HEAD,
        "code_commit": code_commit,
        "seed": int(row["seed"]),
        "retry_count": retry_count,
        "retry_reasons": retry_reasons,
        "elapsed_seconds": 0.0,
        "runtime_error": f"{type(exc).__name__}: {exc}",
        "semantic_scoring": "DEFERRED_UNTIL_COMPLETE",
    }


def _successful_row(
    row: dict[str, Any],
    schedule_index: int,
    output: BackendOutput,
    condition_meta: dict[str, Any],
    trace: Any,
    code_commit: str,
    retry_count: int,
    retry_reasons: list[str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    output_meta = dict(output.metadata)
    terminal = frozen_terminal_metadata(output)
    return {
        **row,
        "schedule_index": schedule_index,
        "raw_output": output.raw_output,
        "generated_token_ids": output_meta.get("generated_token_ids", []),
        **terminal,
        "truncated": bool(terminal["truncated"]),
        "condition_metadata": condition_meta,
        "hook_trace": trace.metadata() if trace is not None else None,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "parser_version": PARSER_VERSION,
        "environment_profile": "CORE_QWEN",
        "experiment_id": "Q2_OOS_V2_SEMANTIC_EXECUTION",
        "prediction_lock_parent_head": PREDICTION_LOCK_PARENT_HEAD,
        "code_commit": code_commit,
        "seed": int(row["seed"]),
        "retry_count": retry_count,
        "retry_reasons": retry_reasons,
        "elapsed_seconds": elapsed_seconds,
        "generation_seconds": output_meta.get("timing", {}).get("generation_seconds"),
        "peak_memory_allocated_bytes": output_meta.get("timing", {}).get(
            "peak_memory_allocated_bytes"
        ),
        "runtime_error": None,
        "semantic_scoring": "DEFERRED_UNTIL_COMPLETE",
    }


def complete_collection_seal(
    journal: CrashSafeJournal,
    schedule: list[dict[str, Any]],
    execution_dir: Path,
    preflight: dict[str, Any],
    environment: dict[str, Any],
    code_commit: str,
    started_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    rows = list(journal.rows.values())
    expected_keys = [
        (r["item_id"], r["condition"], int(r["rollout_index"])) for r in schedule
    ]
    coverage = validate_logical_rows(rows, key_fields=KEY_FIELDS, expected_keys=expected_keys)
    if not coverage.valid or len(rows) != EXPECTED_SCHEDULE_ROWS:
        raise RuntimeError("Q2_OOS_V2_SEMANTIC_EXECUTION_INCOMPLETE")
    journal_sha = sha256_file(journal.path)
    token_counts = [int(row.get("generated_token_count", 0)) for row in rows]
    seal = {
        "schema_version": "q2-oos-v2-collection-complete-seal-v1",
        "status": "COLLECTION_COMPLETE_RAW_UNSCORED",
        "experiment_id": "Q2_OOS_V2_SEMANTIC_EXECUTION",
        "code_commit": code_commit,
        "prediction_lock_parent_head": PREDICTION_LOCK_PARENT_HEAD,
        "schedule_sha256": sha256_file(SCHEDULE),
        "panel_sha256": sha256_file(PANEL),
        "selected_controller_bank_sha256": sha256_file(SELECTED),
        "expected_rows": EXPECTED_SCHEDULE_ROWS,
        "completed_rows": len(rows),
        "missing_rows": len(coverage.missing_keys),
        "unexpected_rows": len(coverage.unexpected_keys),
        "duplicate_keys": len(coverage.duplicate_keys),
        "replacements": 0,
        "raw_journal_sha256": journal_sha,
        "raw_journal_bytes": journal.path.stat().st_size,
        "generated_token_total": sum(token_counts),
        "generated_token_min": min(token_counts),
        "generated_token_max": max(token_counts),
        "repetition_stop_count": sum(
            row.get("terminal_reason") == EXTREME_REPETITION_NAME for row in rows
        ),
        "hard_cap_count": sum(row.get("terminal_reason") == "max_new_tokens" for row in rows),
        "runtime_error_count": sum(row.get("runtime_error") is not None for row in rows),
        "retry_row_count": sum(int(row.get("retry_count", 0)) > 0 for row in rows),
        "semantic_scoring": "NOT_RUN",
        "correctness_inspected": False,
        "semantic_outcomes": 0,
        "environment": environment,
        "preflight": preflight,
        "started_at_utc": started_at_utc,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed_seconds,
        "raw_artifacts": "PRIVATE_HASH_PINNED_NOT_IN_GIT",
    }
    write_json(execution_dir / "COLLECTION_COMPLETE_SEAL.json", seal)
    return seal


def record_preflight(execution_dir: Path, model_path: str, code_commit: str) -> dict[str, Any]:
    execution_dir.mkdir(parents=True, exist_ok=True)
    if git_head() != code_commit:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: code commit mismatch")
    journal_path = execution_dir / "journal.jsonl"
    if journal_path.exists() and journal_path.stat().st_size:
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: pre-existing semantic completions")
    if (execution_dir / "COLLECTION_COMPLETE_SEAL.json").exists():
        raise RuntimeError("Q2_OOS_V2_INSTRUMENT_FAILURE: existing collection seal")
    frozen = validate_frozen_objects()
    environment = verify_spark1_environment(model_path)
    payload = {
        "schema_version": "q2-oos-v2-semantic-preopen-seal-v1",
        "status": "AUTHORIZED_PREOPEN_NO_SEMANTIC_OUTPUTS",
        "authorization": "PRINCIPAL_AUTHORIZATION_Q2_OOS_V2_SEMANTIC_EXECUTION",
        "code_commit": code_commit,
        "frozen": frozen,
        "environment": environment,
        "model_load_performed": False,
        "journal_path_is_empty": not journal_path.exists() or journal_path.stat().st_size == 0,
        "journal_sha256_before_first_output": sha256_file(journal_path)
        if journal_path.exists()
        else None,
        "semantic_outcomes_before_execution": 0,
        "correctness_inspected_before_execution": False,
        "pre_existing_semantic_rows": 0,
        "status_contract": "AUTHORIZED_PREOPEN_NO_SEMANTIC_OUTPUTS",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(execution_dir / "PREOPEN_SEAL.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "head": frozen["head"],
                "schedule_count": frozen["schedule_count"],
                "semantic_outcomes": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def validate_preopen_seal(
    execution_dir: Path, code_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a current, exact persisted pre-open seal before any model load."""

    seal_path = execution_dir / "PREOPEN_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("Q2_OOS_V2_PREOPEN_SEAL_REQUIRED")
    try:
        seal = read_json(seal_path)
        frozen = validate_frozen_objects()
        current_head = git_head()
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Q2_OOS_V2_PREOPEN_SEAL_INVALID") from exc
    if (
        not isinstance(seal, dict)
        or seal.get("status") != "AUTHORIZED_PREOPEN_NO_SEMANTIC_OUTPUTS"
        or seal.get("status_contract") != "AUTHORIZED_PREOPEN_NO_SEMANTIC_OUTPUTS"
        or seal.get("authorization")
        != "PRINCIPAL_AUTHORIZATION_Q2_OOS_V2_SEMANTIC_EXECUTION"
        or seal.get("code_commit") != code_commit
        or current_head != code_commit
        or frozen.get("head") != code_commit
        or seal.get("frozen") != frozen
        or seal.get("semantic_outcomes_before_execution") != 0
        or seal.get("pre_existing_semantic_rows") != 0
        or seal.get("correctness_inspected_before_execution") is not False
        or seal.get("model_load_performed") is not False
        or seal.get("journal_path_is_empty") is not True
    ):
        raise RuntimeError("Q2_OOS_V2_PREOPEN_SEAL_INVALID")
    required_hashes = {
        str(SCHEDULE.relative_to(ROOT)): EXPECTED_SCHEDULE_SHA256,
        str(SELECTED.relative_to(ROOT)): EXPECTED_SELECTED_BANK_SHA256,
        PANEL_HASH_KEY: EXPECTED_PANEL_SHA256,
    }
    frozen_hashes = seal.get("frozen", {}).get("hashes")
    if not isinstance(frozen_hashes, dict):
        raise RuntimeError("Q2_OOS_V2_PREOPEN_SEAL_INVALID")
    for key, expected in required_hashes.items():
        if frozen_hashes.get(key) != expected:
            raise RuntimeError("Q2_OOS_V2_PREOPEN_SEAL_INVALID")
    environment = seal.get("environment")
    model_bytes = environment.get("model_bytes") if isinstance(environment, dict) else None
    if (
        not isinstance(environment, dict)
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("tokenizer_revision") != MODEL_REVISION
        or environment.get("qualified_environment_fingerprint") != EXPECTED_ENVIRONMENT
        or not isinstance(model_bytes, dict)
        or model_bytes.get("manifest_sha256") != EXPECTED_MODEL_MANIFEST_SHA256
    ):
        raise RuntimeError("Q2_OOS_V2_PREOPEN_SEAL_INVALID")
    return seal, frozen


def collect(execution_dir: Path, model_path: str, code_commit: str) -> dict[str, Any]:
    execution_dir.mkdir(parents=True, exist_ok=True)
    preopen, preflight = validate_preopen_seal(execution_dir, code_commit)
    seal_path = execution_dir / "COLLECTION_COMPLETE_SEAL.json"
    if seal_path.exists():
        raise RuntimeError("Q2_OOS_V2_POST_OPENING_PROTOCOL_DEFECT: collection already sealed")
    items = load_items()
    schedule = load_schedule()
    vectors, _selected = load_selected_vectors()
    environment = verify_spark1_environment(model_path)
    if environment != preopen["environment"]:
        raise RuntimeError("Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_DRIFT")
    identity = build_identity(code_commit, environment)
    journal = CrashSafeJournal(
        execution_dir / "journal.jsonl", identity=identity, key_fields=KEY_FIELDS
    )
    if len(journal.rows) > len(schedule):
        raise RuntimeError("Q2_OOS_V2_POST_OPENING_PROTOCOL_DEFECT: extra journal rows")
    backend = build_backend(model_path)
    started = time.monotonic()
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for schedule_index, row in enumerate(schedule):
        key = (row["item_id"], row["condition"], int(row["rollout_index"]))
        if key in journal.rows:
            continue
        retry_count = 0
        retry_reasons: list[str] = []
        condition_meta: dict[str, Any] | None = None
        while True:
            trace: Any | None = None
            try:
                context, model_row, condition_meta = condition_context(
                    backend, items[row["item_id"]], row, vectors
                )
                trajectory_started = time.perf_counter()
                with context as trace:
                    output = backend.generate_reasoning(
                        model_row,
                        sampling_seed=int(row["seed"]),
                        max_new_tokens=SEMANTIC_MAX_NEW_TOKENS,
                        intervention_metadata={
                            "experiment_id": "Q2_OOS_V2_SEMANTIC_EXECUTION",
                            "phase": "SEMANTIC_EXECUTION",
                            "prediction_lock_parent_head": PREDICTION_LOCK_PARENT_HEAD,
                            "parser_version": PARSER_VERSION,
                            "environment_profile": "CORE_QWEN",
                            **condition_meta,
                        },
                        token_stop_predicate=extreme_mechanical_repetition_v1,
                        token_stop_name=EXTREME_REPETITION_NAME,
                    )
                journal.append(
                    _successful_row(
                        row,
                        schedule_index,
                        output,
                        condition_meta,
                        trace,
                        code_commit,
                        retry_count,
                        retry_reasons,
                        time.perf_counter() - trajectory_started,
                    )
                )
                break
            except Exception as exc:  # noqa: BLE001 - exact retry boundary is frozen
                if _is_operational_retryable(exc) and retry_count + 1 < MAX_INFRASTRUCTURE_ATTEMPTS:
                    retry_reasons.append(type(exc).__name__)
                    retry_count += 1
                    continue
                journal.append(
                    _runtime_error_row(
                        row,
                        schedule_index,
                        condition_meta,
                        code_commit,
                        retry_count,
                        retry_reasons,
                        exc,
                    )
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
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = complete_collection_seal(
        journal,
        schedule,
        execution_dir,
        preflight,
        environment,
        code_commit,
        started_at_utc,
        time.monotonic() - started,
    )
    print(
        json.dumps(
            {
                "health": "complete",
                "completed_logical_keys": result["completed_rows"],
                "semantic_scoring": "NOT_RUN",
                "semantic_outcomes": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "collect"), required=True)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-commit", default=git_head())
    args = parser.parse_args()
    if args.mode == "preflight":
        record_preflight(args.execution_dir, args.model_path, args.code_commit)
    else:
        collect(args.execution_dir, args.model_path, args.code_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
