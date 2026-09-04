#!/usr/bin/env python3
"""Fail-closed Spark-1 engine validation and blind Q3.4 qualification collector.

This module deliberately has no semantic parser import and never loads a
reference answer.  Scoring is a separate post-seal program.
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
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from execute_q2_oos_v2_semantic import verify_qualified_model_bytes  # noqa: E402
from run_q2_oos_v2_semantic import (  # noqa: E402
    EXTREME_REPETITION_NAME,
    extreme_mechanical_repetition_v1,
    frozen_terminal_metadata,
)

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.q3_fresh.instrument import build_family  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.experiments.gate6 import vector_sha256  # noqa: E402
from epistemic_geometry.research.reliability import (  # noqa: E402
    CrashSafeJournal,
    validate_logical_rows,
)
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

REVIEW = ROOT / "review/q3_fresh_instrument_qualification"
LOCK = REVIEW / "Q3_FRESH_QUALIFICATION_EXECUTION_LOCK.json"
SCHEDULE = REVIEW / "Q3_FRESH_QUALIFICATION_SCHEDULE.json"
SYSTEM = ROOT / "review/q3_final_system_and_evaluation_supply/FINAL_Q3_CANDIDATE_SYSTEM_DRAFT.json"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_ENVIRONMENT = "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
EXPECTED_BRANCH = "research/q3-fresh-instrument-qualification"
PRIVATE_ROUTER_SHA = "269dc116c70b64dd47cf59340b07dbe558ec8c0f13be8410ed97017310ebad3d"
EXPECTED_ROWS = 6000
KEY_FIELDS = ("family_id", "condition", "rollout_index")
MAX_INFRASTRUCTURE_ATTEMPTS = 3


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_branch() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()


def verify_environment(model_path: str) -> dict[str, Any]:
    if platform.node().split(".", 1)[0] != "spark1" or platform.machine() != "aarch64":
        raise RuntimeError("Q3_FRESH_QUALIFICATION_ENVIRONMENT_DRIFT")
    if os.environ.get("CEG_EXECUTION_PROFILE") != "SPARK1":
        raise RuntimeError("Q3_FRESH_QUALIFICATION_ENVIRONMENT_DRIFT")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Spark-only
        raise RuntimeError("Q3_FRESH_QUALIFICATION_ENVIRONMENT_DRIFT") from exc
    if (
        platform.python_version() != "3.12.3"
        or torch.__version__ != "2.13.0+cu130"
        or torch.version.cuda != "13.0"
        or importlib.metadata.version("transformers") != "4.57.6"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or "GB10" not in torch.cuda.get_device_name(0)
        or not torch.cuda.is_bf16_supported()
        or not hasattr(torch.nn.functional, "scaled_dot_product_attention")
    ):
        raise RuntimeError("Q3_FRESH_QUALIFICATION_ENVIRONMENT_DRIFT")
    model_bytes = verify_qualified_model_bytes(model_path)
    return {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "sdpa_available": True,
        "qualified_environment_fingerprint": EXPECTED_ENVIRONMENT,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "model_bytes": model_bytes,
    }


def load_schedule() -> list[dict[str, Any]]:
    payload = read_json(SCHEDULE)
    rows = list(payload.get("rows", []))
    keys = [(r["family_id"], r["condition"], int(r["rollout_index"])) for r in rows]
    if (
        payload.get("status") != "FROZEN_NOT_RUN"
        or len(rows) != EXPECTED_ROWS
        or len(set(keys)) != EXPECTED_ROWS
        or len({int(r["seed"]) for r in rows}) != EXPECTED_ROWS
        or {int(r["rollout_index"]) for r in rows} != {0, 1}
        or len(set(payload.get("conditions", []))) != 10
    ):
        raise RuntimeError("Q3_FRESH_QUALIFICATION_SCHEDULE_INVALID")
    return rows


def validate_frozen_objects(
    private_prompts: Path, private_router: Path, *, require_current_head: bool = True
) -> dict[str, Any]:
    lock = read_json(LOCK)
    if lock.get("status") != "FROZEN_BEFORE_QWEN_QUALIFICATION":
        raise RuntimeError("Q3_FRESH_QUALIFICATION_LOCK_INVALID")
    if git_branch() != EXPECTED_BRANCH:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_BRANCH_INVALID")
    head = git_head()
    if (
        require_current_head
        and subprocess.run(
            ["git", "merge-base", "--is-ancestor", lock["source_parent"], head], cwd=ROOT
        ).returncode
    ):
        raise RuntimeError("Q3_FRESH_QUALIFICATION_LINEAGE_INVALID")
    expected_runner = lock["implementation"]["runner_sha256"]
    if sha256_file(Path(__file__).resolve()) != expected_runner:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_EXECUTOR_HASH_MISMATCH")
    if sha256_file(SCHEDULE) != lock["schedule_sha256"]:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_SCHEDULE_HASH_MISMATCH")
    if sha256_file(SYSTEM) != lock["candidate_system"]["sha256"]:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_SYSTEM_HASH_MISMATCH")
    if sha256_file(private_prompts) != lock["private_prompt_only_dataset"]["sha256"]:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_PROMPT_DATA_HASH_MISMATCH")
    if sha256_file(private_router) != PRIVATE_ROUTER_SHA:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_ROUTER_HASH_MISMATCH")
    rows = load_schedule()
    return {
        "head": head,
        "lock_sha256": sha256_file(LOCK),
        "schedule_sha256": sha256_file(SCHEDULE),
        "schedule_rows": len(rows),
        "unique_seeds": len({int(r["seed"]) for r in rows}),
        "private_prompt_sha256": sha256_file(private_prompts),
        "private_router_sha256": sha256_file(private_router),
        "confirmation_qwen_access": 0,
        "reserve_qwen_access": 0,
    }


def load_prompts(path: Path) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 300:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_PROMPT_COUNT_INVALID")
    prompts: dict[str, str] = {}
    for row in rows:
        if set(row) != {"family_id", "prompt", "prompt_sha256"}:
            raise RuntimeError("Q3_FRESH_QUALIFICATION_REFERENCE_FIREWALL_FAILURE")
        if sha256_text(row["prompt"]) != row["prompt_sha256"]:
            raise RuntimeError("Q3_FRESH_QUALIFICATION_PROMPT_HASH_INVALID")
        prompts[row["family_id"]] = row["prompt"]
    if len(prompts) != 300:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_PROMPT_ID_INVALID")
    return prompts


def load_vectors() -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], list[str]]:
    lock = read_json(LOCK)
    vectors: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in lock["policies"]:
        policy = str(row["policy_id"])
        path = ROOT / row["vector_path"]
        if sha256_file(path) != row["vector_file_sha256"]:
            raise RuntimeError(f"Q3_FRESH_QUALIFICATION_VECTOR_FILE_MISMATCH: {policy}")
        vector = np.load(path, allow_pickle=False).astype(np.float64)
        if vector_sha256(vector) != row["vector_sha256"]:
            raise RuntimeError(f"Q3_FRESH_QUALIFICATION_VECTOR_HASH_MISMATCH: {policy}")
        vectors[policy] = vector
        metadata[policy] = dict(row)
        if row["role"] == "BANK":
            order.append(policy)
    if len(vectors) != 9 or len(order) != 8:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_POLICY_SET_INVALID")
    return vectors, metadata, order


def load_router(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        router = {name: data[name].astype(np.float64) for name in data.files}
    shapes = {
        "pca_mean": (4096,),
        "pca_components": (8, 4096),
        "pca_scale": (8,),
        "router_u": (8, 2),
        "router_v": (8, 2),
        "router_a": (8,),
        "router_b": (8,),
    }
    if set(router) != set(shapes) or any(router[k].shape != v for k, v in shapes.items()):
        raise RuntimeError("Q3_FRESH_QUALIFICATION_ROUTER_SHAPE_INVALID")
    if not all(np.isfinite(value).all() for value in router.values()):
        raise RuntimeError("Q3_FRESH_QUALIFICATION_ROUTER_NONFINITE")
    return router


def select_policy(feature: np.ndarray, router: dict[str, np.ndarray], order: list[str]) -> str:
    x = (feature.astype(np.float64) - router["pca_mean"]) @ router["pca_components"].T
    x = x / router["pca_scale"]
    scores = (
        x @ router["router_a"]
        + router["router_b"]
        + (x @ router["router_u"]) @ router["router_v"].T
    )
    if not np.isfinite(scores).all():
        return "V4_DIRECTION_02_MEDIUM"
    return order[int(np.argmax(scores))]


def build_backend(model_path: str) -> HuggingFaceBackend:
    return HuggingFaceBackend(
        BackendConfig(
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
            max_new_tokens=4096,
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
        ),
        model_identifier=MODEL,
        tokenizer_identifier=model_path,
        model_revision=MODEL_REVISION,
    )


def model_item(family_id: str, prompt: str) -> BenchmarkItem:
    return BenchmarkItem(
        id=family_id,
        prompt=prompt,
        target="REFERENCE_NOT_LOADED_DURING_COLLECTION",
        metadata={"source_prompt_hash": sha256_text(prompt), "response_channel": "q3_fresh"},
    )


class RoutedHook(AbstractContextManager["RoutedHook"]):
    def __init__(
        self,
        backend: HuggingFaceBackend,
        router: dict[str, np.ndarray],
        order: list[str],
        vectors: dict[str, np.ndarray],
        policy_meta: dict[str, dict[str, Any]],
        final_index: int,
    ) -> None:
        self.backend, self.router, self.order = backend, router, order
        self.vectors, self.policy_meta, self.final_index = vectors, policy_meta, final_index
        self.selected_policy: str | None = None
        self.feature: np.ndarray | None = None
        self.selection_count = 0
        self.forward_count = 0
        self.applications = 0
        self._handles: list[Any] = []

    def __enter__(self) -> RoutedHook:
        module = self.backend.layer_module(27)

        def pre(_module: Any, inputs: Any) -> None:
            hidden = inputs[0]
            if self.selected_policy is None:
                if hidden.shape[1] <= self.final_index:
                    raise RuntimeError("Q3_FRESH_ROUTER_PREFILL_POSITION_INVALID")
                self.feature = hidden[0, self.final_index].detach().float().cpu().numpy()
                self.selected_policy = select_policy(self.feature, self.router, self.order)
                self.selection_count += 1

        def post(_module: Any, _inputs: Any, output: Any) -> Any:
            if self.selected_policy is None:
                raise RuntimeError("Q3_FRESH_ROUTER_SELECTION_MISSING")
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            position = 0 if hidden.shape[1] == 1 else self.final_index
            row = self.policy_meta[self.selected_policy]
            delta = self.backend.torch.tensor(
                self.vectors[self.selected_policy] * float(row["alpha"]),
                dtype=self.backend.torch.float32,
                device=hidden.device,
            ).to(dtype=hidden.dtype)
            updated = hidden.clone()
            updated[0, position, :] = hidden[0, position, :] + delta
            self.forward_count += 1
            self.applications += 1
            if isinstance(output, tuple):
                return (updated, *output[1:])
            if isinstance(output, list):
                return [updated, *output[1:]]
            return updated

        self._handles = [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def metadata(self) -> dict[str, Any]:
        return {
            "selected_policy": self.selected_policy,
            "selection_count": self.selection_count,
            "forward_count": self.forward_count,
            "applications": self.applications,
            "same_policy_throughout_decode": self.selection_count == 1,
        }


def generation_context(
    backend: HuggingFaceBackend,
    item: BenchmarkItem,
    condition: str,
    vectors: dict[str, np.ndarray],
    policy_meta: dict[str, dict[str, Any]],
    order: list[str],
    router: dict[str, np.ndarray],
) -> tuple[AbstractContextManager[Any], dict[str, Any]]:
    encoded, _rendered, prompt_hash = backend._encode_item(item)  # noqa: SLF001
    final_index = int(encoded["input_ids"].shape[1] - 1)
    if condition == "ONLINE_ROUTED":
        return RoutedHook(backend, router, order, vectors, policy_meta, final_index), {
            "condition": condition,
            "prompt_hash": prompt_hash,
            "layer": 27,
            "duration": "sustained_current_token",
            "router": "FROZEN_Q3_GEOMETRY_BLIND_POLICY_ID",
        }
    row = policy_meta[condition]
    delta = backend.torch.tensor(
        vectors[condition] * float(row["alpha"]),
        dtype=backend.torch.float32,
        device=backend.device,
    ).view(1, 1, -1)
    return Gate6HookTrace(
        layers={27: backend.layer_module(27)}, deltas={27: delta}, target_positions=[final_index]
    ), {
        "condition": condition,
        "prompt_hash": prompt_hash,
        "layer": 27,
        "duration": "sustained_current_token",
        "alpha": row["alpha"],
        "vector_sha256": row["vector_sha256"],
    }


def run_engine_test(
    execution_dir: Path, model_path: str, private_prompts: Path, private_router: Path
) -> dict[str, Any]:
    frozen = validate_frozen_objects(private_prompts, private_router)
    environment = verify_environment(model_path)
    vectors, metadata, order = load_vectors()
    router = load_router(private_router)
    backend = build_backend(model_path)
    results = []
    generations = 0
    forwards = 0
    for index in range(2):
        fixture = build_family("excluded-fixture", index, 0x5133)
        item = model_item(f"ENGINEERING_ONLY_{index}", fixture.prompt)
        context, meta = generation_context(
            backend, item, "ONLINE_ROUTED", vectors, metadata, order, router
        )
        with context as routed:
            output = backend.generate_reasoning(
                item,
                sampling_seed=730000 + index,
                max_new_tokens=32,
                token_stop_predicate=extreme_mechanical_repetition_v1,
                token_stop_name=EXTREME_REPETITION_NAME,
            )
        generations += 1
        forwards += int(routed.forward_count)
        selected = str(routed.selected_policy)
        if routed.selection_count != 1 or routed.feature is None:
            raise RuntimeError("Q3_FRESH_ENGINE_ROUTER_LIFECYCLE_FAILURE")
        if select_policy(routed.feature, router, order) != selected:
            raise RuntimeError("Q3_FRESH_ENGINE_ONLINE_OFFLINE_MISMATCH")
        fixed_context, _ = generation_context(
            backend, item, selected, vectors, metadata, order, router
        )
        with fixed_context as fixed:
            replay = backend.generate_reasoning(
                item,
                sampling_seed=730000 + index,
                max_new_tokens=32,
                token_stop_predicate=extreme_mechanical_repetition_v1,
                token_stop_name=EXTREME_REPETITION_NAME,
            )
        generations += 1
        forwards += int(fixed.forward_count)
        if output.metadata["generated_token_ids"] != replay.metadata["generated_token_ids"]:
            raise RuntimeError("Q3_FRESH_ENGINE_FIXED_POLICY_REPLAY_MISMATCH")
        results.append(
            {"fixture": index, "selected_policy": selected, "prompt_hash": meta["prompt_hash"]}
        )
    report = {
        "schema_version": "q3-fresh-engine-validation-v1",
        "status": "Q3_FRESH_ROUTER_ENGINE_VALIDATED",
        "frozen": frozen,
        "environment": environment,
        "engineering_fixtures": 2,
        "engineering_generations": generations,
        "engineering_model_forwards": forwards,
        "online_offline_selection_agreement": True,
        "fixed_policy_replay_agreement": True,
        "sampling_rng_consumed_by_router": False,
        "references_or_scorer_available": False,
        "results": results,
    }
    write_json(execution_dir / "ENGINE_VALIDATION.json", report)
    return report


def record_preopen(
    execution_dir: Path, model_path: str, private_prompts: Path, private_router: Path
) -> dict[str, Any]:
    journal = execution_dir / "journal.jsonl"
    if journal.exists() and journal.stat().st_size:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_PREEXISTING_ROWS")
    frozen = validate_frozen_objects(private_prompts, private_router)
    environment = verify_environment(model_path)
    engine_path = execution_dir / "ENGINE_VALIDATION.json"
    if (
        not engine_path.is_file()
        or read_json(engine_path).get("status") != "Q3_FRESH_ROUTER_ENGINE_VALIDATED"
    ):
        raise RuntimeError("Q3_FRESH_QUALIFICATION_ENGINE_VALIDATION_REQUIRED")
    seal = {
        "schema_version": "q3-fresh-qualification-preopen-v1",
        "status": "AUTHORIZED_PREOPEN_NO_QUALIFICATION_OUTPUTS",
        "code_commit": git_head(),
        "frozen": frozen,
        "environment": environment,
        "engine_validation_sha256": sha256_file(engine_path),
        "journal_rows": 0,
        "correctness_inspected": False,
        "qualification_outcomes": 0,
        "confirmation_qwen_access": 0,
        "reserve_qwen_access": 0,
        "model_load_during_preopen": False,
    }
    write_json(execution_dir / "PREOPEN_SEAL.json", seal)
    return seal


def validate_preopen(
    execution_dir: Path, model_path: str, private_prompts: Path, private_router: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = execution_dir / "PREOPEN_SEAL.json"
    if not path.is_file():
        raise RuntimeError("Q3_FRESH_QUALIFICATION_PREOPEN_REQUIRED")
    seal = read_json(path)
    frozen = validate_frozen_objects(private_prompts, private_router)
    environment = verify_environment(model_path)
    if (
        seal.get("status") != "AUTHORIZED_PREOPEN_NO_QUALIFICATION_OUTPUTS"
        or seal.get("code_commit") != git_head()
        or seal.get("frozen") != frozen
        or seal.get("environment") != environment
        or seal.get("journal_rows") != 0
        or seal.get("correctness_inspected") is not False
        or seal.get("qualification_outcomes") != 0
        or seal.get("confirmation_qwen_access") != 0
        or seal.get("reserve_qwen_access") != 0
    ):
        raise RuntimeError("Q3_FRESH_QUALIFICATION_PREOPEN_INVALID")
    return seal, environment


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError, EOFError))


def collect(
    execution_dir: Path, model_path: str, private_prompts: Path, private_router: Path
) -> dict[str, Any]:
    preopen, environment = validate_preopen(
        execution_dir, model_path, private_prompts, private_router
    )
    if (execution_dir / "COLLECTION_COMPLETE_SEAL.json").exists():
        raise RuntimeError("Q3_FRESH_QUALIFICATION_ALREADY_SEALED")
    prompts = load_prompts(private_prompts)
    schedule = load_schedule()
    vectors, metadata, order = load_vectors()
    router = load_router(private_router)
    identity = {
        "experiment": "Q3_FRESH_INSTRUMENT_QUALIFICATION",
        "code_commit": git_head(),
        "lock_sha256": sha256_file(LOCK),
        "schedule_sha256": sha256_file(SCHEDULE),
        "private_prompt_sha256": sha256_file(private_prompts),
        "private_router_sha256": sha256_file(private_router),
        "environment": environment,
        "semantic_scoring": "DEFERRED_UNTIL_RAW_SEAL",
    }
    journal = CrashSafeJournal(
        execution_dir / "journal.jsonl", identity=identity, key_fields=KEY_FIELDS
    )
    backend = build_backend(model_path)
    started = time.monotonic()
    for index, row in enumerate(schedule):
        key = (row["family_id"], row["condition"], int(row["rollout_index"]))
        if key in journal.rows:
            continue
        item = model_item(row["family_id"], prompts[row["family_id"]])
        attempts = 0
        retry_reasons: list[str] = []
        while True:
            try:
                context, condition_meta = generation_context(
                    backend, item, row["condition"], vectors, metadata, order, router
                )
                trajectory_started = time.perf_counter()
                with context as trace:
                    output = backend.generate_reasoning(
                        item,
                        sampling_seed=int(row["seed"]),
                        max_new_tokens=4096,
                        token_stop_predicate=extreme_mechanical_repetition_v1,
                        token_stop_name=EXTREME_REPETITION_NAME,
                    )
                elapsed = time.perf_counter() - trajectory_started
                terminal = frozen_terminal_metadata(output)
                trace_meta = trace.metadata()
                if row["condition"] == "ONLINE_ROUTED" and trace_meta["selection_count"] != 1:
                    raise RuntimeError("Q3_FRESH_ROUTER_SELECTION_COUNT_INVALID")
                journal.append(
                    {
                        **row,
                        "schedule_index": index,
                        "raw_output": output.raw_output,
                        "generated_token_ids": output.metadata["generated_token_ids"],
                        **terminal,
                        "condition_metadata": condition_meta,
                        "hook_trace": trace_meta,
                        "model": MODEL,
                        "model_revision": MODEL_REVISION,
                        "seed": int(row["seed"]),
                        "retry_count": attempts,
                        "retry_reasons": retry_reasons,
                        "elapsed_seconds": elapsed,
                        "runtime_error": None,
                        "semantic_scoring": "DEFERRED_UNTIL_COMPLETE_RAW_SEAL",
                    }
                )
                break
            except BaseException as exc:
                if _retryable(exc) and attempts + 1 < MAX_INFRASTRUCTURE_ATTEMPTS:
                    attempts += 1
                    retry_reasons.append(f"{type(exc).__name__}: {exc}")
                    continue
                raise
        completed = len(journal.rows)
        if completed == 1 or completed % 100 == 0:
            print(
                json.dumps({"completed": completed, "pending": EXPECTED_ROWS - completed}),
                flush=True,
            )
    rows = list(journal.rows.values())
    expected = [(r["family_id"], r["condition"], int(r["rollout_index"])) for r in schedule]
    coverage = validate_logical_rows(rows, key_fields=KEY_FIELDS, expected_keys=expected)
    if not coverage.valid or len(rows) != EXPECTED_ROWS:
        raise RuntimeError("Q3_FRESH_QUALIFICATION_EXECUTION_INCOMPLETE")
    tokens = [int(row["generated_token_count"]) for row in rows]
    seal = {
        "schema_version": "q3-fresh-qualification-collection-seal-v1",
        "status": "COLLECTION_COMPLETE_RAW_UNSCORED",
        "completed": len(rows),
        "expected": EXPECTED_ROWS,
        "missing": len(coverage.missing_keys),
        "unexpected": len(coverage.unexpected_keys),
        "duplicates": len(coverage.duplicate_keys),
        "replacements": 0,
        "retry_rows": sum(int(r["retry_count"]) > 0 for r in rows),
        "runtime_errors": sum(r["runtime_error"] is not None for r in rows),
        "repetition_stops": sum(r["terminal_reason"] == EXTREME_REPETITION_NAME for r in rows),
        "hard_caps": sum(r["terminal_reason"] == "max_new_tokens" for r in rows),
        "generated_tokens": sum(tokens),
        "generated_token_mean": float(np.mean(tokens)),
        "generated_token_median": float(np.median(tokens)),
        "elapsed_seconds": time.monotonic() - started,
        "journal_sha256": sha256_file(journal.path),
        "journal_bytes": journal.path.stat().st_size,
        "preopen_sha256": sha256_file(execution_dir / "PREOPEN_SEAL.json"),
        "correctness_inspected": False,
        "semantic_scoring": "NOT_RUN",
        "confirmation_qwen_access": 0,
        "reserve_qwen_access": 0,
        "environment": environment,
    }
    write_json(execution_dir / "COLLECTION_COMPLETE_SEAL.json", seal)
    return seal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("engine-test", "preflight", "collect"), required=True)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--private-prompts", type=Path, required=True)
    parser.add_argument("--private-router", type=Path, required=True)
    args = parser.parse_args()
    args.execution_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "engine-test":
        result = run_engine_test(
            args.execution_dir, args.model_path, args.private_prompts, args.private_router
        )
    elif args.mode == "preflight":
        result = record_preopen(
            args.execution_dir, args.model_path, args.private_prompts, args.private_router
        )
    else:
        result = collect(
            args.execution_dir, args.model_path, args.private_prompts, args.private_router
        )
    print(
        json.dumps(
            {k: result[k] for k in result if k in {"status", "completed", "expected"}},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
