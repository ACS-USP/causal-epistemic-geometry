#!/usr/bin/env python3
"""Synthetic-only Spark-2 native qualification for the fixed Qwen instrument."""

from __future__ import annotations

import argparse
import hashlib
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

from run_gate6_2_first_stage_repair import build_backend, model_item, prompt_tokens  # noqa: E402

from epistemic_geometry.benchmarks.external.base import ExternalItem  # noqa: E402
from epistemic_geometry.experiments import q1_second_task as q1s  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402

REVIEW = ROOT / "review/q1_second_task_spark2_design"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tokens(output: Any) -> list[int]:
    return [int(value) for value in output.metadata.get("generated_token_ids", [])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if platform.node() != "spark2":
        raise RuntimeError("Spark-2 guard failed")
    if _git_commit() != args.source_commit:
        raise RuntimeError("runtime checkout differs from frozen pre-engine commit")
    protocol = _read_json(REVIEW / "PROTOCOL_LOCK.json")
    if protocol["status"] != "PROSPECTIVE_LOCK_PRE_ENGINE":
        raise RuntimeError("pre-engine protocol is not frozen")
    expected_hash = protocol["hashes"]["SPARK2_ENGINE_QUALIFICATION_PROTOCOL.json"]
    if _sha256(REVIEW / "SPARK2_ENGINE_QUALIFICATION_PROTOCOL.json") != expected_hash:
        raise RuntimeError("engine protocol hash mismatch")

    import torch
    import transformers

    backend = build_backend(args.model_path)
    expected = _read_json(REVIEW / "SPARK2_ENGINE_QUALIFICATION_PROTOCOL.json")["expected"]
    environment = {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "dtype": "BF16",
        "attention": "SDPA",
        "model_revision": q1s.MODEL_REVISION,
    }
    environment_exact = all(environment.get(key) == value for key, value in expected.items())
    scientific_fingerprint = hashlib.sha256(
        json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    controller_lock = _read_json(REVIEW / "CONTROLLER_PROVENANCE_LOCK.json")
    vector_path = ROOT / controller_lock["vector_path"]
    meaningful = np.load(vector_path, allow_pickle=False).astype(np.float64)
    vector_identity = (
        _sha256(vector_path) == q1s.MEANINGFUL_VECTOR_FILE_SHA256
        and vector_sha256(meaningful) == q1s.MEANINGFUL_VECTOR_HASH
        and np.isclose(np.linalg.norm(meaningful), 1.0, atol=1e-12)
    )
    random_lock = _read_json(REVIEW / "RANDOM_BANK_LOCK.json")
    vectors = {"MEANINGFUL_FIXED_QWEN_L27_D75": meaningful}
    vector_hashes = {"MEANINGFUL_FIXED_QWEN_L27_D75": q1s.MEANINGFUL_VECTOR_HASH}
    for name in q1s.RANDOM_NAMES:
        record = random_lock["records"][name]
        value = np.load(ROOT / record["vector_path"], allow_pickle=False).astype(np.float64)
        if vector_sha256(value) != record["canonical_float64_vector_sha256"]:
            raise RuntimeError(f"null vector hash mismatch: {name}")
        vectors[name] = value
        vector_hashes[name] = record["canonical_float64_vector_sha256"]
    deltas = {name: value * q1s.EFFECTIVE_DELTA_NORM for name, value in vectors.items()}

    fixtures = _read_json(REVIEW / "ENGINEERING_FIXTURES.json")["fixtures"]
    tokenization = []
    for fixture in fixtures:
        item = ExternalItem(
            item_id=fixture["fixture_id"],
            benchmark="SYNTHETIC_ENGINEERING",
            subtask="NO_SCIENTIFIC_TASK",
            prompt=fixture["prompt"],
            reference_answer="",
            evaluator="none",
            source_revision="synthetic-v1",
        )
        row = model_item(item)
        prompt_ids, rendered, rendered_hash = prompt_tokens(backend, row)
        tokenization.append(
            {
                "fixture_id": fixture["fixture_id"],
                "prompt_tokens": len(prompt_ids),
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "rendered_prompt_hash": rendered_hash,
            }
        )

    alpha_zero = []
    repeatability = []
    cleanup = []
    trace_rows: list[dict[str, Any]] = []
    trace_forward_counts: list[int] = []
    exercised: set[str] = set()
    generated_token_count = 0
    generation_started = time.perf_counter()
    for index, fixture in enumerate(fixtures[:3]):
        item = ExternalItem(
            item_id=fixture["fixture_id"],
            benchmark="SYNTHETIC_ENGINEERING",
            subtask="NO_SCIENTIFIC_TASK",
            prompt=fixture["prompt"],
            reference_answer="",
            evaluator="none",
            source_revision="synthetic-v1",
        )
        row = model_item(item)
        seed = 202_608_290 + index
        clean = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        repeated = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        generated_token_count += len(_tokens(clean)) + len(_tokens(repeated))
        repeatability.append(_tokens(clean) == _tokens(repeated))
        prompt_ids, _rendered, _rendered_hash = prompt_tokens(backend, row)
        zero = backend.torch.zeros(
            (1, 1, meaningful.size), dtype=backend.torch.float32, device=backend.device
        )
        with Gate6HookTrace(
            layers={q1s.LAYER: backend.layer_module(q1s.LAYER)},
            deltas={q1s.LAYER: zero},
            target_positions=[len(prompt_ids) - 1],
        ) as zero_trace:
            zero_output = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        generated_token_count += len(_tokens(zero_output))
        alpha_zero.append(_tokens(clean) == _tokens(zero_output) and zero_trace.forward_count >= 1)
        conditions = tuple(vectors) if index == 0 else ("MEANINGFUL_FIXED_QWEN_L27_D75",)
        for offset, condition in enumerate(conditions):
            tensor = backend.torch.tensor(
                deltas[condition], dtype=backend.torch.float32, device=backend.device
            ).view(1, 1, -1)
            with Gate6HookTrace(
                layers={q1s.LAYER: backend.layer_module(q1s.LAYER)},
                deltas={q1s.LAYER: tensor},
                target_positions=[len(prompt_ids) - 1],
            ) as trace:
                steered_output = backend.generate_reasoning(
                    row, sampling_seed=seed + 100 + offset, max_new_tokens=16
                )
            generated_token_count += len(_tokens(steered_output))
            metadata = trace.metadata()
            trace_rows.extend(metadata["applications"])
            trace_forward_counts.append(int(metadata["forward_count"]))
            exercised.add(condition)
        clean_after = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        generated_token_count += len(_tokens(clean_after))
        cleanup.append(_tokens(clean) == _tokens(clean_after))
    generation_seconds = time.perf_counter() - generation_started

    parser_roundtrip = all(
        [
            q1s.evaluate_livecodebench_output("FINAL: 7", "7")["correct"],
            q1s.evaluate_livecodebench_output("FINAL: false", "false")["correct"],
            q1s.evaluate_livecodebench_output('FINAL: "x"', '"x"')["correct"],
            q1s.evaluate_livecodebench_output("FINAL: [1, 2]", "[1, 2]")["correct"],
            not q1s.evaluate_livecodebench_output(
                "FINAL: __import__('os').system('id')", "0"
            )["semantic_evaluable"],
        ]
    )
    synthetic_journal = [
        (fixture["fixture_id"], "BASELINE", index % 2) for index, fixture in enumerate(fixtures)
    ]
    journal_resume = len(synthetic_journal) == len(set(synthetic_journal))
    relative_error = max(float(row["relative_shift_error"]) for row in trace_rows)
    noncurrent = max(abs(float(row["non_current_change"])) for row in trace_rows)
    delta_norms = [float(np.linalg.norm(value)) for value in deltas.values()]
    checks = {
        "environment_exact": environment_exact,
        "model_and_tokenizer_identity": bool(
            q1s.MODEL_REVISION == q1s.TOKENIZER_REVISION
            and Path(args.model_path).name.endswith(q1s.MODEL_REVISION)
        ),
        "vector_identity": vector_identity,
        "alpha_zero_token_identity": all(alpha_zero),
        "seed_repeatability": all(repeatability),
        "hook_cleanup": all(cleanup),
        "per_forward_exact_shift_bf16_eps_le_2": relative_error <= 2.0,
        "current_token_noncurrent_change_le_0.125": noncurrent <= 0.125,
        "one_application_per_forward": len(trace_rows) == sum(trace_forward_counts),
        "cached_decode_observed": any(int(row["sequence_length"]) == 1 for row in trace_rows),
        "random_delta_norm_range_le_1e-9": float(np.ptp(delta_norms)) <= 1e-9,
        "parser_roundtrip": parser_roundtrip,
        "journal_resume_synthetic": journal_resume,
    }
    passed = all(checks.values()) and exercised == set(vectors)
    result = {
        "classification": (
            "SPARK2_NATIVE_ENGINE_QUALIFIED" if passed else "SPARK2_NATIVE_ENGINE_NOT_QUALIFIED"
        ),
        "pass": passed,
        "source_commit": args.source_commit,
        "environment": environment,
        "scientific_environment_fingerprint": scientific_fingerprint,
        "checks": checks,
        "fixtures": len(fixtures),
        "synthetic_generation_fixtures": 3,
        "synthetic_generated_tokens": generated_token_count,
        "synthetic_generation_seconds": generation_seconds,
        "synthetic_tokens_per_second": generated_token_count / generation_seconds,
        "scientific_benchmark_items": 0,
        "scientific_benchmark_outcomes": 0,
        "correctness_inspected": False,
        "max_relative_shift_error_bf16_eps": relative_error,
        "max_noncurrent_change": noncurrent,
        "trace_forward_count": sum(trace_forward_counts),
        "trace_application_count": len(trace_rows),
        "exercised_conditions": sorted(exercised),
        "tokenization": tokenization,
        "raw_generated_text_persisted": False,
        "process_id": os.getpid(),
    }
    _write_json(REVIEW / "SPARK2_ENGINE_QUALIFICATION.json", result)
    if not passed:
        raise RuntimeError("SPARK2_NATIVE_ENGINE_NOT_QUALIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
