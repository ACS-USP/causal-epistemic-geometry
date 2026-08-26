#!/usr/bin/env python3
"""Spark-1 runner for Q2 V4 presemantic qualification only."""

from __future__ import annotations

import argparse
import hashlib
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

from run_gate6_2_first_stage_repair import model_item, prompt_tokens  # noqa: E402
from run_gate11_domain_conditioned_control import forward  # noqa: E402
from run_q2_v3 import (  # noqa: E402
    EXECUTION_TEACHER_TEXT,
    _calibrate_alpha,
    _capture_boundaries,
    _mechanical_parse,
)

from epistemic_geometry.analysis.q2_geometries import fit_whitening, whitened_geometry  # noqa: E402
from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.external.base import ExternalItem  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.experiments.q2_v3 import SOURCE_FAMILIES as V3_FAMILIES  # noqa: E402
from epistemic_geometry.experiments.q2_v3_prompt_provenance import (  # noqa: E402
    canonical_q2_v3_task_prompt,
)
from epistemic_geometry.experiments.q2_v4_presemantic import (  # noqa: E402
    CANDIDATE_COUNT,
    EXPERIMENT_ID,
    LAYER,
    LOCATIONS,
    MODEL,
    MODEL_REVISION,
    SELECTED_COUNT,
    SHELL_TARGETS,
    SHELLS,
    SOURCE_FAMILIES,
    bank_algebraic_checks,
    baseline_centered_angle,
    candidate_bank,
    deterministic_seed,
    retained_subspace,
    select_first_safe,
    selected_bank_checks,
    source_direction_id,
)
from epistemic_geometry.research.reliability import CrashSafeJournal  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

REVIEW = ROOT / "review/q2_v4_spark1_presemantic"
OFFICIAL = ROOT / "review/q2_v3_provenance_reconciliation/OFFICIAL_SOURCE_RECORDS.jsonl"
MAX_NEW_TOKENS = 4096


def build_v4_backend(model_path: str) -> HuggingFaceBackend:
    """Build the exact Spark-1 V4 backend without inheriting an old layer field."""

    config = BackendConfig(
        type="huggingface",
        model_id=MODEL,
        model_path=model_path,
        model_revision=MODEL_REVISION,
        tokenizer_id=model_path,
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        layer=LAYER,
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


def require_lock() -> dict[str, Any]:
    lock = read_json(REVIEW / "QUALIFICATION_PROTOCOL_LOCK.json")
    if lock["status"] != "Q2_V4_SPARK1_QUALIFICATION_PROTOCOL_FROZEN":
        raise RuntimeError("Q2_V4_PREDICTION_LOCK_FAILED: qualification lock status")
    if lock["spark1_only"] is not True or lock["spark2_forbidden"] is not True:
        raise RuntimeError("Q2_V4_SPARK1_ENGINE_NOT_QUALIFIED: backend policy")
    for name, expected in lock["artifact_hashes"].items():
        if sha256(REVIEW / name) != expected:
            raise RuntimeError(f"Q2_V4_PREDICTION_LOCK_FAILED: hash {name}")
    if len(read_json(REVIEW / "PRIMARY_PANEL_MANIFEST.json")["items"]) != 300:
        raise RuntimeError("Q2_V4_PREDICTION_LOCK_FAILED: panel size")
    return lock


def _official() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with OFFICIAL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records[str(row["id"])] = row
    return records


def items(allocation_file: str) -> list[ExternalItem]:
    manifest = read_json(REVIEW / allocation_file)
    source = _official()
    result = []
    for frozen in manifest["items"]:
        row = source[str(frozen["item_id"])]
        prompt = canonical_q2_v3_task_prompt(str(row["code"]), str(row["input"]))
        reference = str(row["output"])
        result.append(
            ExternalItem(
                item_id=str(row["id"]),
                benchmark="CRUXEval",
                subtask="output_prediction",
                prompt=prompt,
                reference_answer=reference,
                evaluator="python_literal",
                source_revision=str(row["dataset_revision"]),
                metadata={
                    "allocation": manifest["allocation"],
                    "official_index": int(row["official_index"]),
                    "provenance_class": frozen["provenance_class"],
                },
            )
        )
    if [item.item_id for item in result] != manifest["item_ids"]:
        raise RuntimeError(f"Q2_V4_PREDICTION_LOCK_FAILED: order {allocation_file}")
    return result


def _instruction(family: str, polarity: str) -> str:
    record = next(row for row in V3_FAMILIES if row.family_id == family)
    return record.positive_instruction if polarity == "POSITIVE" else record.negative_instruction


def model_manifest(model_path: Path) -> dict[str, Any]:
    records = []
    for path in sorted(model_path.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            records.append(
                {
                    "path": str(path.relative_to(model_path)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    required = {"config.json", "tokenizer.json", "model.safetensors.index.json"}
    present = {row["path"] for row in records}
    if not required.issubset(present):
        raise RuntimeError("Q2_V4_SPARK1_ENGINE_NOT_QUALIFIED: incomplete model manifest")
    config = read_json(model_path / "config.json")
    if config.get("_name_or_path") not in {MODEL, str(model_path)}:
        config_name = str(config.get("_name_or_path"))
    else:
        config_name = MODEL
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "local_path": str(model_path),
        "config_model_identity": config_name,
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "manifest_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def environment_manifest(backend: Any, model_path: Path) -> dict[str, Any]:
    torch = backend.torch
    python_include = Path(os.environ.get("CEG_SPARK_PYTHON_INCLUDE", "/usr/include/python3.12"))
    python_header = python_include / "Python.h"
    if not python_header.is_file():
        raise RuntimeError("Q2_V4_SPARK1_ENGINE_NOT_QUALIFIED: Python.h missing")
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,utilization.gpu,temperature.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "hostname": platform.node(),
        "required_hostname": "spark1",
        "architecture": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": smi,
        "packages": packages,
        "python_build_include": str(python_include),
        "python_header_sha256": sha256(python_header),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "model_path": str(model_path),
        "source_commit": git_head(),
        "spark2_used": False,
    }
    payload["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def engine_phase(backend: Any, model_path: Path) -> None:
    require_lock()
    started = time.monotonic()
    fixtures = read_json(REVIEW / "TECHNICAL_FIXTURES.json")["fixtures"]
    tokenizer_checks = []
    deterministic_checks = []
    alpha_zero_checks = []
    hook_checks = []
    cache_checks = []
    technical_logit_checks = []
    throughput = []
    ordered_outputs: dict[str, list[int]] = {}
    for index, fixture in enumerate(fixtures):
        item = BenchmarkItem(
            id=fixture["fixture_id"], prompt=fixture["prompt"], target="ENGINEERING_ONLY"
        )
        first_ids, first_rendered, first_hash = prompt_tokens(backend, item)
        second_ids, second_rendered, second_hash = prompt_tokens(backend, item)
        tokenizer_checks.append(
            first_ids == second_ids
            and first_rendered == second_rendered
            and first_hash == second_hash
        )
        if index >= 5:
            continue
        seed = deterministic_seed("Q2-V4-ENGINE", fixture["fixture_id"])
        before = time.monotonic()
        clean = backend.generate_reasoning(item, sampling_seed=seed, max_new_tokens=32)
        elapsed = time.monotonic() - before
        repeated = backend.generate_reasoning(item, sampling_seed=seed, max_new_tokens=32)
        deterministic_checks.append(
            clean.metadata.get("generated_token_ids")
            == repeated.metadata.get("generated_token_ids")
        )
        ordered_outputs[item.id] = list(clean.metadata.get("generated_token_ids", []))
        prompt_ids, _rendered, _hash = prompt_tokens(backend, item)
        with backend.torch.inference_mode():
            logits_first = (
                forward(
                    backend,
                    prompt_ids,
                    past=None,
                    total_length=len(prompt_ids),
                    phase="prefill",
                )
                .logits[0, -1]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            logits_second = (
                forward(
                    backend,
                    prompt_ids,
                    past=None,
                    total_length=len(prompt_ids),
                    phase="prefill",
                )
                .logits[0, -1]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        technical_logit_checks.append(bool(np.array_equal(logits_first, logits_second)))
        zero = backend.torch.zeros((1, 1, 4096), dtype=backend.torch.float32, device=backend.device)
        trace = Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: zero},
            target_positions=[len(prompt_ids) - 1],
        )
        with trace:
            zero_output = backend.generate_reasoning(item, sampling_seed=seed, max_new_tokens=32)
        alpha_zero_checks.append(
            clean.metadata.get("generated_token_ids")
            == zero_output.metadata.get("generated_token_ids")
        )
        rng = np.random.default_rng(1000 + index)
        delta_np = rng.normal(size=4096)
        delta_np = 0.01 * delta_np / np.linalg.norm(delta_np)
        delta = backend.torch.tensor(
            delta_np, dtype=backend.torch.float32, device=backend.device
        ).view(1, 1, -1)
        shifted = Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: delta},
            target_positions=[len(prompt_ids) - 1],
        )
        with shifted:
            backend.generate_reasoning(item, sampling_seed=seed, max_new_tokens=8)
        metadata = shifted.metadata()
        hook_checks.append(
            metadata["forward_count"] >= 1
            and metadata["forward_count"] == len(metadata["applications"])
            and all(row["relative_shift_error"] <= 2.0 for row in metadata["applications"])
            and all(abs(row["non_current_change"]) <= 0.125 for row in metadata["applications"])
        )
        cache_checks.append(
            metadata["applications"][0]["sequence_length"] == len(prompt_ids)
            and all(row["sequence_length"] == 1 for row in metadata["applications"][1:])
        )
        throughput.append(
            {
                "fixture_id": fixture["fixture_id"],
                "seconds": elapsed,
                "generated_tokens": int(clean.metadata.get("generated_token_count", 0)),
            }
        )
    cleanup_item = BenchmarkItem(
        id="cleanup", prompt="Finish with FINAL: cleanup.", target="ENGINEERING_ONLY"
    )
    cleanup = backend.generate_reasoning(
        cleanup_item, sampling_seed=deterministic_seed("Q2-V4-ENGINE", "cleanup"), max_new_tokens=16
    )
    reverse_outputs = {}
    for fixture in reversed(fixtures[:5]):
        item = BenchmarkItem(
            id=fixture["fixture_id"],
            prompt=fixture["prompt"],
            target="ENGINEERING_ONLY",
        )
        seed = deterministic_seed("Q2-V4-ENGINE", fixture["fixture_id"])
        output = backend.generate_reasoning(item, sampling_seed=seed, max_new_tokens=32)
        reverse_outputs[item.id] = list(output.metadata.get("generated_token_ids", []))
    resume_path = REVIEW / "ENGINE_RESUME_FIXTURE.jsonl"
    resume_identity = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "ENGINE_RESUME_FIXTURE",
        "source_commit": git_head(),
    }
    resume = CrashSafeJournal(
        resume_path,
        identity=resume_identity,
        key_fields=("fixture_id",),
    )
    if ("ENGINE_RESUME_0",) not in resume.rows:
        resume.append({"fixture_id": "ENGINE_RESUME_0", "value": 1})
    reopened = CrashSafeJournal(
        resume_path,
        identity=resume_identity,
        key_fields=("fixture_id",),
    )
    checks = {
        "hostname_spark1": platform.node() == "spark1",
        "architecture_aarch64": platform.machine() == "aarch64",
        "one_cuda_device": backend.torch.cuda.device_count() == 1,
        "gpu_gb10": "GB10" in backend.torch.cuda.get_device_name(0),
        "bf16_supported": bool(backend.torch.cuda.is_bf16_supported()),
        "tokenization_repeatability": all(tokenizer_checks),
        "deterministic_seed_repeatability": all(deterministic_checks),
        "technical_logit_repeatability": all(technical_logit_checks),
        "batch_order_independence": ordered_outputs == reverse_outputs,
        "resume_identity_and_no_duplicate": (
            len(reopened.rows) == 1 and ("ENGINE_RESUME_0",) in reopened.rows
        ),
        "alpha_zero_identity": all(alpha_zero_checks),
        "hook_scope_shift_forward": all(hook_checks),
        "kv_cache_current_token_semantics": all(cache_checks),
        "hook_cleanup": cleanup.metadata.get("intervention", "none") == "none",
    }
    write_json(REVIEW / "SPARK1_ENVIRONMENT_LOCK.json", environment_manifest(backend, model_path))
    write_json(REVIEW / "EXACT_MODEL_MANIFEST.json", model_manifest(model_path))
    write_json(
        REVIEW / "SPARK1_ENGINE_QUALIFICATION.json",
        {
            "checks": checks,
            "throughput_fixtures": throughput,
            "elapsed_seconds": time.monotonic() - started,
            "classification": "Q2_V4_SPARK1_ENGINE_QUALIFIED"
            if all(checks.values())
            else "Q2_V4_SPARK1_ENGINE_NOT_QUALIFIED",
        },
    )
    if not all(checks.values()):
        raise RuntimeError("Q2_V4_SPARK1_ENGINE_NOT_QUALIFIED")


def source_phase(backend: Any) -> None:
    if read_json(REVIEW / "SPARK1_ENGINE_QUALIFICATION.json")["classification"] != (
        "Q2_V4_SPARK1_ENGINE_QUALIFIED"
    ):
        raise RuntimeError("Q2_V4_SPARK1_ENGINE_NOT_QUALIFIED")
    construction = items("SOURCE_CONSTRUCTION_MANIFEST.json")
    validation = items("SOURCE_VALIDATION_MANIFEST.json")
    arrays: dict[str, np.ndarray] = {}
    archive_path = REVIEW / "SOURCE_ACTIVATIONS.npz"
    if not archive_path.exists():
        for split, split_items in (("construction", construction), ("validation", validation)):
            for family in SOURCE_FAMILIES:
                for polarity in ("POSITIVE", "NEGATIVE"):
                    captures = [
                        _capture_boundaries(backend, item, _instruction(family, polarity))
                        for item in split_items
                    ]
                    for location in LOCATIONS:
                        arrays[f"{split}__{family}__{polarity}__{location}"] = np.stack(
                            [row[location] for row in captures]
                        ).astype(np.float32)
        np.savez_compressed(archive_path, **arrays)
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "SPARK1_SOURCE_QUALIFICATION",
        "source_commit": git_head(),
    }
    journal = CrashSafeJournal(
        REVIEW / "SOURCE_JOURNAL.jsonl",
        identity=identity,
        key_fields=("item_id", "family", "polarity", "rollout_index"),
    )
    schedule = read_json(REVIEW / "SOURCE_QUALIFICATION_SCHEDULE.json")["rows"]
    item_map = {item.item_id: item for item in validation}
    phase_started = time.monotonic()
    generated_tokens = 0
    for row in schedule:
        key = (row["item_id"], row["family"], row["polarity"], row["rollout_index"])
        if key in journal.rows:
            continue
        output = backend.generate_reasoning(
            model_item(item_map[row["item_id"]], _instruction(row["family"], row["polarity"])),
            sampling_seed=int(row["seed"]),
            max_new_tokens=MAX_NEW_TOKENS,
            intervention_metadata={
                "phase": "SPARK1_SOURCE_QUALIFICATION",
                "correctness": "FORBIDDEN",
            },
        )
        token_count = int(output.metadata.get("generated_token_count", 0))
        generated_tokens += token_count
        journal.append(
            {
                **row,
                **_mechanical_parse(output.raw_output, token_count),
                "raw_output": output.raw_output,
                "generated_token_ids": output.metadata.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "correctness_evaluated": False,
            }
        )
    if len(journal.rows) != len(schedule):
        raise RuntimeError("Q2_V4_SPARK1_ENGINE_NOT_QUALIFIED: source journal incomplete")
    finalize_source(identity)
    write_json(
        REVIEW / "SOURCE_THROUGHPUT.json",
        {
            "rows": len(schedule),
            "new_generated_tokens": generated_tokens,
            "elapsed_seconds": time.monotonic() - phase_started,
        },
    )


def finalize_source(identity: dict[str, Any]) -> None:
    rows = list(
        CrashSafeJournal(
            REVIEW / "SOURCE_JOURNAL.jsonl",
            identity=identity,
            key_fields=("item_id", "family", "polarity", "rollout_index"),
        ).rows.values()
    )
    activations = np.load(REVIEW / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    vector_dir = REVIEW / "SOURCE_DIRECTIONS"
    vector_dir.mkdir(exist_ok=True)
    records: dict[str, Any] = {}
    vector_meta: dict[str, Any] = {}
    ordered_vectors = []
    for family in SOURCE_FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        by_key = {(r["item_id"], r["polarity"], r["rollout_index"]): r for r in selected}
        item_ids = sorted({row["item_id"] for row in selected})
        cross = []
        within = []
        for item_id in item_ids:
            pos = [by_key[(item_id, "POSITIVE", r)]["canonical_value"] for r in (0, 1)]
            neg = [by_key[(item_id, "NEGATIVE", r)]["canonical_value"] for r in (0, 1)]
            cross.extend(float(left != right) for left in pos for right in neg)
            within.extend((float(pos[0] != pos[1]), float(neg[0] != neg[1])))
        pos_rows = [row for row in selected if row["polarity"] == "POSITIVE"]
        neg_rows = [row for row in selected if row["polarity"] == "NEGATIVE"]
        record: dict[str, Any] = {
            "positive_validity": float(np.mean([r["commitment_valid"] for r in pos_rows])),
            "negative_validity": float(np.mean([r["commitment_valid"] for r in neg_rows])),
            "positive_evaluability": float(np.mean([r["semantic_evaluable"] for r in pos_rows])),
            "negative_evaluability": float(np.mean([r["semantic_evaluable"] for r in neg_rows])),
            "cross_disagreement": float(np.mean(cross)),
            "within_disagreement": float(np.mean(within)),
            "excess_disagreement": float(np.mean(cross) - np.mean(within)),
            "locations": {},
        }
        for location in LOCATIONS:
            cpos = activations[f"construction__{family}__POSITIVE__{location}"].astype(np.float64)
            cneg = activations[f"construction__{family}__NEGATIVE__{location}"].astype(np.float64)
            raw = np.mean(cpos - cneg, axis=0)
            raw_norm = float(np.linalg.norm(raw))
            direction = raw / raw_norm
            vpos = activations[f"validation__{family}__POSITIVE__{location}"].astype(np.float64)
            vneg = activations[f"validation__{family}__NEGATIVE__{location}"].astype(np.float64)
            gaps = (vpos - vneg) @ direction
            sd = float(np.std(gaps, ddof=1))
            direction_id = source_direction_id(family, location)
            path = vector_dir / f"{direction_id}.npy"
            np.save(path, direction)
            meta = {
                "family": family,
                "location": location,
                "path": str(path.relative_to(ROOT)),
                "file_sha256": sha256(path),
                "canonical_vector_hash": vector_sha256(direction),
                "raw_norm": raw_norm,
                "standardized_gap": float(np.mean(gaps) / max(sd, 1e-12)),
                "positive_projection_fraction": float(np.mean(gaps > 0)),
            }
            record["locations"][location] = meta
            vector_meta[direction_id] = meta
            ordered_vectors.append(direction)
        behavior = (
            min(
                record["positive_validity"],
                record["negative_validity"],
                record["positive_evaluability"],
                record["negative_evaluability"],
            )
            >= 0.90
            and record["cross_disagreement"] >= 0.10
            and record["excess_disagreement"] >= 0.03
        )
        representation = all(
            row["raw_norm"] >= 1e-6
            and row["standardized_gap"] >= 0.20
            and row["positive_projection_fraction"] >= 0.60
            for row in record["locations"].values()
        )
        record["pass"] = bool(behavior and representation)
        records[family] = record
    source_pass = len(vector_meta) == 8 and all(record["pass"] for record in records.values())
    write_json(
        REVIEW / "SPARK1_SOURCE_BASIS_QUALIFICATION.json",
        {
            "families": records,
            "directions": vector_meta,
            "all_four_families_pass": source_pass,
            "correctness_used": False,
            "classification": "Q2_V4_SPARK1_SOURCE_BASIS_QUALIFIED"
            if source_pass
            else "Q2_V4_SPARK1_SOURCE_BASIS_NOT_QUALIFIED",
        },
    )
    if not source_pass:
        raise RuntimeError("Q2_V4_SPARK1_SOURCE_BASIS_NOT_QUALIFIED")
    source_matrix = np.stack(ordered_vectors, axis=1)
    basis, report = retained_subspace(source_matrix)
    np.save(REVIEW / "SPARK1_SOURCE_MATRIX.npy", source_matrix)
    np.save(REVIEW / "SPARK1_SUBSPACE_Q.npy", basis)
    report.update(
        {
            "source_matrix_sha256": sha256(REVIEW / "SPARK1_SOURCE_MATRIX.npy"),
            "Q_sha256": sha256(REVIEW / "SPARK1_SUBSPACE_Q.npy"),
            "direction_order": list(vector_meta),
            "classification": "Q2_V4_SUBSPACE_QUALIFIED"
            if report["pass"]
            else "Q2_V4_SUBSPACE_NOT_QUALIFIED",
        }
    )
    write_json(REVIEW / "SPARK1_SUBSPACE_QUALIFICATION.json", report)
    if not report["pass"]:
        raise RuntimeError("Q2_V4_SUBSPACE_NOT_QUALIFIED")


def derive_bank_phase(prelock_commit: str) -> None:
    require_lock()
    subspace = read_json(REVIEW / "SPARK1_SUBSPACE_QUALIFICATION.json")
    if subspace["classification"] != "Q2_V4_SUBSPACE_QUALIFIED":
        raise RuntimeError("Q2_V4_SUBSPACE_NOT_QUALIFIED")
    basis = np.load(REVIEW / "SPARK1_SUBSPACE_Q.npy", allow_pickle=False)
    coefficients, vectors, seed = candidate_bank(basis, prelock_commit)
    checks = bank_algebraic_checks(coefficients, vectors)
    bank_dir = REVIEW / "CANDIDATE_DIRECTIONS"
    bank_dir.mkdir(exist_ok=True)
    candidates = []
    for index in range(CANDIDATE_COUNT):
        name = f"V4_DIRECTION_{index:02d}"
        path = bank_dir / f"{name}.npy"
        np.save(path, vectors[index])
        candidates.append(
            {
                "candidate_id": name,
                "generation_index": index,
                "coefficients": coefficients[index].tolist(),
                "path": str(path.relative_to(ROOT)),
                "file_sha256": sha256(path),
                "canonical_vector_hash": vector_sha256(vectors[index]),
            }
        )
    write_json(
        REVIEW / "CANDIDATE_BANK_MANIFEST.json",
        {
            "schema_version": "q2-v4-candidate-bank-v1",
            "prelock_commit": prelock_commit,
            "seed": str(seed),
            "seed_hex_128": f"{seed:032x}",
            "byte_order": "big",
            "rng": "NumPy PCG64DXSM",
            "candidate_count": CANDIDATE_COUNT,
            "candidates": candidates,
            "algebraic_gate": checks,
            "redraw_permitted": False,
            "classification": "Q2_V4_CANDIDATE_BANK_ALGEBRAIC_PASS"
            if checks["pass"]
            else "Q2_V4_CANDIDATE_BANK_ALGEBRAIC_FAIL",
        },
    )
    if not checks["pass"]:
        raise RuntimeError("Q2_V4_BANK_IDENTIFIABILITY_FAILED")


def _candidate_vectors() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = read_json(REVIEW / "CANDIDATE_BANK_MANIFEST.json")
    vectors = {}
    for row in manifest["candidates"]:
        vector = np.load(ROOT / row["path"], allow_pickle=False).astype(np.float64)
        if sha256(ROOT / row["path"]) != row["file_sha256"]:
            raise RuntimeError("Q2_V4_PREDICTION_LOCK_FAILED: candidate hash")
        vectors[row["candidate_id"]] = vector
    return vectors, manifest


def _baseline_denominator(backend: Any, split_items: list[ExternalItem]) -> tuple[float, list[Any]]:
    teacher = [
        int(v) for v in backend.tokenizer.encode(EXECUTION_TEACHER_TEXT, add_special_tokens=False)
    ]
    squared = []
    records = []
    for item in split_items:
        row = model_item(item)
        prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
        ids = backend.torch.tensor(
            [prompt_ids + teacher], dtype=backend.torch.long, device=backend.device
        )
        captured = []

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            start: int = len(prompt_ids) - 1,
            sink: list[np.ndarray] = captured,
        ) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            sink.append(hidden[0, start:, :].detach().float().cpu().numpy())

        handle = backend.layer_module(LAYER).register_forward_hook(hook)
        try:
            with backend.torch.inference_mode():
                backend._forward(  # noqa: SLF001
                    backend.model,
                    {
                        "input_ids": ids,
                        "attention_mask": backend.torch.ones_like(ids),
                        "use_cache": False,
                        "return_dict": True,
                    },
                    "prefill",
                )
        finally:
            handle.remove()
        values = captured[0].astype(np.float64)
        squared.extend(np.sum(values**2, axis=1).tolist())
        records.append(
            {"item_id": item.item_id, "prompt_hash": prompt_hash, "positions": len(values)}
        )
    return float(np.mean(squared)), records


def _condition_context(
    backend: Any,
    item: ExternalItem,
    condition: str,
    vectors: dict[str, np.ndarray],
    deployment: dict[str, Any],
) -> tuple[Any, BenchmarkItem]:
    row = model_item(item)
    if condition == "BASELINE":
        return nullcontext(), row
    prompt_ids, _rendered, _hash = prompt_tokens(backend, row)
    record = deployment[condition]
    delta = backend.torch.tensor(
        vectors[record["candidate_id"]] * float(record["alpha"]),
        dtype=backend.torch.float32,
        device=backend.device,
    ).view(1, 1, -1)
    return (
        Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: delta},
            target_positions=[len(prompt_ids) - 1],
        ),
        row,
    )


def safety_phase(backend: Any, prelock_commit: str) -> None:
    vectors, bank = _candidate_vectors()
    if bank["prelock_commit"] != prelock_commit:
        raise RuntimeError("Q2_V4_PREDICTION_LOCK_FAILED: PRELOCK mismatch")
    split_items = items("SHELL_CALIBRATION_MANIFEST.json")
    denominator, denominator_records = _baseline_denominator(backend, split_items)
    deployment = {}
    for candidate, vector in vectors.items():
        for shell in SHELLS:
            condition = f"{candidate}_{shell}"
            deployment[condition] = {
                "condition": condition,
                "candidate_id": candidate,
                "shell": shell,
                "target_amplitude": SHELL_TARGETS[shell],
                "vector_hash": vector_sha256(vector),
                **_calibrate_alpha(vector, SHELL_TARGETS[shell], denominator),
            }
    write_json(
        REVIEW / "SHELL_CALIBRATION_MANIFEST_RESULT.json",
        {
            "prelock_commit": prelock_commit,
            "denominator_mean_squared_norm": denominator,
            "denominator_records": denominator_records,
            "controllers": deployment,
            "classification": "Q2_V4_SHELL_CALIBRATION_COMPLETE",
        },
    )
    schedule = []
    conditions = ["BASELINE", *deployment]
    for item in split_items:
        for rollout in (0, 1):
            seed = deterministic_seed("Q2-V4-SHELL-SAFETY", prelock_commit, item.item_id, rollout)
            order_rng = np.random.Generator(np.random.PCG64DXSM(seed))
            for order, condition in enumerate(order_rng.permutation(conditions).tolist()):
                schedule.append(
                    {
                        "item_id": item.item_id,
                        "condition": str(condition),
                        "rollout_index": rollout,
                        "matched_seed": seed,
                        "condition_order": order,
                    }
                )
    write_json(REVIEW / "CANDIDATE_SAFETY_SCHEDULE.json", {"rows": schedule})
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "CANDIDATE_SAFETY",
        "prelock_commit": prelock_commit,
    }
    journal = CrashSafeJournal(
        REVIEW / "CANDIDATE_SAFETY_JOURNAL.jsonl",
        identity=identity,
        key_fields=("item_id", "condition", "rollout_index"),
    )
    item_map = {item.item_id: item for item in split_items}
    started = time.monotonic()
    generated = 0
    for row in schedule:
        key = (row["item_id"], row["condition"], row["rollout_index"])
        if key in journal.rows:
            continue
        context, model_row = _condition_context(
            backend, item_map[row["item_id"]], row["condition"], vectors, deployment
        )
        with context:
            output = backend.generate_reasoning(
                model_row,
                sampling_seed=int(row["matched_seed"]),
                max_new_tokens=MAX_NEW_TOKENS,
                intervention_metadata={"phase": "CANDIDATE_SAFETY", "correctness": "FORBIDDEN"},
            )
        count = int(output.metadata.get("generated_token_count", 0))
        generated += count
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
    if len(rows) != len(schedule):
        raise RuntimeError("Q2_V4_SPARK1_ENGINE_NOT_QUALIFIED: safety journal incomplete")
    baseline = [row for row in rows if row["condition"] == "BASELINE"]
    baseline_valid = float(np.mean([row["commitment_valid"] for row in baseline]))
    baseline_eval = float(np.mean([row["semantic_evaluable"] for row in baseline]))
    by_key = {(row["item_id"], row["condition"], row["rollout_index"]): row for row in rows}
    results = {}
    for candidate in vectors:
        shell_records = {}
        for shell in SHELLS:
            condition = f"{candidate}_{shell}"
            selected = [row for row in rows if row["condition"] == condition]
            movement = float(
                np.mean(
                    [
                        row["generated_token_ids"]
                        != by_key[(row["item_id"], "BASELINE", row["rollout_index"])][
                            "generated_token_ids"
                        ]
                        for row in selected
                    ]
                )
            )
            validity = float(np.mean([row["commitment_valid"] for row in selected]))
            evaluability = float(np.mean([row["semantic_evaluable"] for row in selected]))
            truncation = float(np.mean([row["truncated"] for row in selected]))
            calibration = deployment[condition]
            passed = (
                validity >= 0.90
                and validity >= baseline_valid - 0.05
                and evaluability >= 0.90
                and evaluability >= baseline_eval - 0.05
                and truncation <= 0.05
                and movement >= (0.10 if shell == "MEDIUM" else 0.15)
                and calibration["relative_target_error"] <= 0.005
            )
            shell_records[shell] = {
                "validity": validity,
                "evaluability": evaluability,
                "truncation": truncation,
                "raw_sequence_movement": movement,
                "implemented_amplitude": calibration["implemented_amplitude"],
                "relative_target_error": calibration["relative_target_error"],
                "pass": passed,
            }
        results[candidate] = {
            "shells": shell_records,
            "both_shells_pass": all(row["pass"] for row in shell_records.values()),
        }
    selected_names = select_first_safe(results)
    classification = (
        "Q2_V4_SAFE_BANK_QUALIFIED"
        if len(selected_names) == SELECTED_COUNT
        else "Q2_V4_SAFE_BANK_INSUFFICIENT"
    )
    write_json(
        REVIEW / "CANDIDATE_SAFETY_REPORT.json",
        {
            "baseline_validity": baseline_valid,
            "baseline_evaluability": baseline_eval,
            "candidates": results,
            "safe_count": sum(bool(row["both_shells_pass"]) for row in results.values()),
            "selected_first_32_safe": selected_names,
            "classification": classification,
            "correctness_used": False,
            "elapsed_seconds": time.monotonic() - started,
            "new_generated_tokens": generated,
        },
    )
    if classification != "Q2_V4_SAFE_BANK_QUALIFIED":
        raise RuntimeError("Q2_V4_SAFE_BANK_INSUFFICIENT")
    coefficients = np.asarray(
        [
            next(row["coefficients"] for row in bank["candidates"] if row["candidate_id"] == name)
            for name in selected_names
        ]
    )
    amplitudes = np.asarray(
        [
            [deployment[f"{name}_{shell}"]["implemented_amplitude"] for shell in SHELLS]
            for name in selected_names
        ]
    )
    identifiability = selected_bank_checks(coefficients, amplitudes)
    write_json(
        REVIEW / "SELECTED_CONTROLLER_BANK.json",
        {
            "prelock_commit": prelock_commit,
            "selection_rule": "first 32 candidates passing both frozen shell gates",
            "selected_ids": selected_names,
            "controllers": {
                f"{name}_{shell}": deployment[f"{name}_{shell}"]
                for name in selected_names
                for shell in SHELLS
            },
            "identifiability": identifiability,
            "classification": "Q2_V4_BANK_IDENTIFIABILITY_PASS"
            if identifiability["pass"]
            else "Q2_V4_BANK_IDENTIFIABILITY_FAILED",
        },
    )
    if not identifiability["pass"]:
        raise RuntimeError("Q2_V4_BANK_IDENTIFIABILITY_FAILED")


def _capture_covariance(backend: Any) -> np.ndarray:
    captured_rows = []
    metadata = []
    for item in items("M1_COVARIANCE_MANIFEST.json"):
        row = model_item(item)
        prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
        ids = backend.torch.tensor([prompt_ids], dtype=backend.torch.long, device=backend.device)
        capture = []

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            sink: list[np.ndarray] = capture,
        ) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            sink.append(hidden[0, -1].detach().float().cpu().numpy())

        handle = backend.layer_module(LAYER).register_forward_hook(hook)
        try:
            with backend.torch.inference_mode():
                backend._forward(  # noqa: SLF001
                    backend.model,
                    {
                        "input_ids": ids,
                        "attention_mask": backend.torch.ones_like(ids),
                        "use_cache": False,
                        "return_dict": True,
                    },
                    "prefill",
                )
        finally:
            handle.remove()
        captured_rows.append(capture[0])
        metadata.append({"item_id": item.item_id, "prompt_hash": prompt_hash})
    array = np.stack(captured_rows).astype(np.float32)
    np.savez_compressed(REVIEW / "A1_COVARIANCE_ACTIVATIONS.npz", activations=array)
    write_json(REVIEW / "A1_COVARIANCE_METADATA.json", metadata)
    return array


def _checkpoint_indices(length: int) -> tuple[int, int, int]:
    if length < 3:
        raise RuntimeError("Q2_V4_A2_INSTRUMENT_NOT_QUALIFIED: continuation too short")
    indices = (length // 3, (2 * length) // 3, length - 1)
    if len(set(indices)) != 3:
        raise RuntimeError("Q2_V4_A2_INSTRUMENT_NOT_QUALIFIED: duplicate checkpoints")
    return indices


def _fingerprint(
    backend: Any,
    item: ExternalItem,
    candidate: str | None,
    alpha: float,
    vectors: dict[str, np.ndarray],
    continuation: list[int],
) -> np.ndarray:
    row = model_item(item)
    prompt_ids, _rendered, _hash = prompt_tokens(backend, row)
    if candidate is None:
        context = nullcontext()
    else:
        delta = backend.torch.tensor(
            vectors[candidate] * alpha,
            dtype=backend.torch.float32,
            device=backend.device,
        ).view(1, 1, -1)
        context = Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: delta},
            target_positions=[len(prompt_ids) - 1],
        )
    snapshots = []
    checkpoints = _checkpoint_indices(len(continuation))
    with context, backend.torch.inference_mode():
        output = forward(
            backend,
            prompt_ids,
            past=None,
            total_length=len(prompt_ids),
            phase="prefill",
        )
        snapshots.append(output.logits[0, -1].detach().float().cpu().numpy())
        past = output.past_key_values
        for index, token in enumerate(continuation):
            output = forward(
                backend,
                [token],
                past=past,
                total_length=len(prompt_ids) + index + 1,
                phase="decode",
            )
            past = output.past_key_values
            if index in checkpoints:
                snapshots.append(output.logits[0, -1].detach().float().cpu().numpy())
    return np.stack(snapshots).astype(np.float32)


def geometry_phase(backend: Any, prelock_commit: str) -> None:
    selected = read_json(REVIEW / "SELECTED_CONTROLLER_BANK.json")
    if selected["classification"] != "Q2_V4_BANK_IDENTIFIABILITY_PASS":
        raise RuntimeError("Q2_V4_BANK_IDENTIFIABILITY_FAILED")
    vectors, _bank = _candidate_vectors()
    covariance = _capture_covariance(backend)
    fit = fit_whitening(covariance.astype(np.float64), regularization_fraction=0.10)
    np.savez_compressed(
        REVIEW / "A1_COVARIANCE_FIT.npz",
        mean=fit.mean,
        right_singular_vectors=fit.right_singular_vectors,
        eigenvalues=fit.eigenvalues,
        isotropic_variance=np.asarray([fit.isotropic_variance]),
        regularization_fraction=np.asarray([fit.regularization_fraction]),
        regularization_value=np.asarray([fit.regularization_value]),
    )
    names = selected["selected_ids"]
    coefficients = np.asarray(
        [
            next(
                row["coefficients"]
                for row in read_json(REVIEW / "CANDIDATE_BANK_MANIFEST.json")["candidates"]
                if row["candidate_id"] == name
            )
            for name in names
        ],
        dtype=np.float64,
    )
    a0 = 1.0 - coefficients @ coefficients.T
    direction_rows = np.stack([vectors[name] for name in names])
    a1 = np.asarray(whitened_geometry(direction_rows, fit)["cosine_distance"], dtype=np.float64)
    repeated_fit = fit_whitening(covariance.astype(np.float64), regularization_fraction=0.10)
    a1_checks = {
        "activation_shape_64_by_4096": covariance.shape == (64, 4096),
        "activations_finite": bool(np.isfinite(covariance).all()),
        "regularization_value_positive": fit.regularization_value > 0.0,
        "condition_number_finite_at_most_1e6": bool(
            np.isfinite(fit.condition_number) and fit.condition_number <= 1e6
        ),
        "effective_rank_at_least_2": fit.effective_rank >= 2.0,
        "deterministic_fit_hash": fit.fit_hash == repeated_fit.fit_hash,
        "matrix_finite": bool(np.isfinite(a1).all()),
        "matrix_symmetry_error_at_most_1e10": float(np.max(np.abs(a1 - a1.T))) <= 1e-10,
        "matrix_diagonal_error_at_most_1e10": float(np.max(np.abs(np.diag(a1)))) <= 1e-10,
        "cosine_distance_range": float(np.min(a1)) >= -1e-10 and float(np.max(a1)) <= 2.0 + 1e-10,
    }
    a1_pass = all(a1_checks.values())
    write_json(
        REVIEW / "A1_INSTRUMENT_QUALIFICATION.json",
        {
            "activation_archive_sha256": sha256(REVIEW / "A1_COVARIANCE_ACTIVATIONS.npz"),
            "fit_sha256": sha256(REVIEW / "A1_COVARIANCE_FIT.npz"),
            "fit_hash": fit.fit_hash,
            "lambda": 0.10,
            "regularization_value": fit.regularization_value,
            "effective_rank": fit.effective_rank,
            "condition_number": fit.condition_number,
            "checks": a1_checks,
            "classification": "Q2_V4_A1_INSTRUMENT_QUALIFIED"
            if a1_pass
            else "Q2_V4_A1_INSTRUMENT_NOT_QUALIFIED",
        },
    )
    if not a1_pass:
        raise RuntimeError("Q2_V4_A1_INSTRUMENT_NOT_QUALIFIED")
    continuation = [
        int(v) for v in backend.tokenizer.encode(EXECUTION_TEACHER_TEXT, add_special_tokens=False)
    ]
    probes = items("M2_PROBE_MANIFEST.json")
    deployment = selected["controllers"]
    raw_dir = REVIEW / "A2_FINGERPRINTS"
    raw_dir.mkdir(exist_ok=True)
    repeat_dir = REVIEW / "A2_REPEAT_FINGERPRINTS"
    repeat_dir.mkdir(exist_ok=True)
    baseline_rows = []
    repeated_baseline_rows = []
    condition_rows = {f"{name}_{shell}": [] for name in names for shell in SHELLS}
    repeated_condition_rows = {f"{name}_{shell}": [] for name in names for shell in SHELLS}
    for item in probes:
        baseline = _fingerprint(backend, item, None, 0.0, vectors, continuation)
        repeated_baseline = _fingerprint(backend, item, None, 0.0, vectors, continuation)
        baseline_rows.append(baseline)
        repeated_baseline_rows.append(repeated_baseline)
        arrays = {"BASELINE": baseline}
        repeated_arrays = {"BASELINE": repeated_baseline}
        for name in names:
            for shell in SHELLS:
                condition = f"{name}_{shell}"
                value = _fingerprint(
                    backend,
                    item,
                    name,
                    float(deployment[condition]["alpha"]),
                    vectors,
                    continuation,
                )
                repeated_value = _fingerprint(
                    backend,
                    item,
                    name,
                    float(deployment[condition]["alpha"]),
                    vectors,
                    continuation,
                )
                condition_rows[condition].append(value)
                repeated_condition_rows[condition].append(repeated_value)
                arrays[condition] = value
                repeated_arrays[condition] = repeated_value
        np.savez_compressed(raw_dir / f"{item.item_id}.npz", **arrays)
        np.savez_compressed(repeat_dir / f"{item.item_id}.npz", **repeated_arrays)
    baseline = np.concatenate(baseline_rows, axis=0).astype(np.float64)
    repeated_baseline = np.concatenate(repeated_baseline_rows, axis=0).astype(np.float64)
    from epistemic_geometry.experiments.q2_v4_presemantic import mean_js

    repeat_js = mean_js(baseline, repeated_baseline)
    noise_floor = max(1e-12, 100.0 * repeat_js)
    matrices = {"A0_MEDIUM": a0, "A0_STRONG": a0, "A1_MEDIUM": a1, "A1_STRONG": a1}
    a2_reports = {}
    for shell in SHELLS:
        fingerprints = {
            name: np.concatenate(condition_rows[f"{name}_{shell}"], axis=0).astype(np.float64)
            for name in names
        }
        repeated_fingerprints = {
            name: np.concatenate(repeated_condition_rows[f"{name}_{shell}"], axis=0).astype(
                np.float64
            )
            for name in names
        }
        result = baseline_centered_angle(baseline, fingerprints, noise_floor_squared=noise_floor)
        repeated_result = baseline_centered_angle(
            repeated_baseline,
            repeated_fingerprints,
            noise_floor_squared=noise_floor,
        )
        upper = np.triu_indices(len(names), 1)
        distance_relative = float(
            np.max(
                np.abs(result["distance_squared"] - repeated_result["distance_squared"])
                / np.maximum(np.abs(result["distance_squared"]), 1e-12)
            )
        )
        radius_relative = float(
            np.max(
                np.abs(result["radii_squared"] - repeated_result["radii_squared"])
                / np.maximum(np.abs(result["radii_squared"]), 1e-12)
            )
        )
        left = result["dissimilarity"][upper]
        right = repeated_result["dissimilarity"][upper]
        angular_correlation = float(
            np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1]
        )
        baseline_identity_error = float(
            np.max(
                np.abs(
                    result["radii_squared"]
                    - np.asarray([mean_js(fingerprints[name], baseline) for name in names])
                )
            )
        )
        checks = {
            "radius_floor": result["radius_floor_pass"],
            "symmetry": float(
                np.max(np.abs(result["distance_squared"] - result["distance_squared"].T))
            )
            <= 1e-12,
            "diagonal": float(np.max(np.abs(np.diag(result["distance_squared"])))) <= 1e-12,
            "baseline_identity": baseline_identity_error <= 1e-10,
            "gram_psd": result["gram_min_eigenvalue"] >= -1e-8,
            "cosine_bounds": result["raw_cosine_min"] >= -1.0 - 1e-8
            and result["raw_cosine_max"] <= 1.0 + 1e-8,
            "repeat_radius": radius_relative <= 1e-6,
            "repeat_distance": distance_relative <= 1e-6,
            "repeat_angular_rank": angular_correlation >= 0.999,
        }
        matrices[f"A2_{shell}"] = result["dissimilarity"]
        matrices[f"D2_{shell}"] = np.sqrt(np.maximum(result["distance_squared"], 0.0))
        a2_reports[shell] = {
            "radii_squared": result["radii_squared"].tolist(),
            "gram_min_eigenvalue": result["gram_min_eigenvalue"],
            "cosine_range": [result["raw_cosine_min"], result["raw_cosine_max"]],
            "repeat_radius_relative_error_max": radius_relative,
            "repeat_distance_relative_error_max": distance_relative,
            "repeat_angular_rank_correlation": angular_correlation,
            "baseline_identity_max_error": baseline_identity_error,
            "checks": checks,
            "pass": all(checks.values()),
        }
    np.savez_compressed(REVIEW / "PREDICTION_MATRICES.npz", **matrices)
    instrument_pass = all(row["pass"] for row in a2_reports.values())
    write_json(
        REVIEW / "A2_INSTRUMENT_QUALIFICATION.json",
        {
            "prelock_commit": prelock_commit,
            "probe_count": len(probes),
            "checkpoint_count_per_probe": 4,
            "repeat_baseline_mean_JS": repeat_js,
            "noise_floor_squared": noise_floor,
            "shells": a2_reports,
            "classification": "Q2_V4_A2_INSTRUMENT_QUALIFIED"
            if instrument_pass
            else "Q2_V4_A2_INSTRUMENT_NOT_QUALIFIED",
        },
    )
    write_json(
        REVIEW / "PREDICTION_MATRIX_METADATA.json",
        {
            "prelock_commit": prelock_commit,
            "controller_order": names,
            "matrix_archive_sha256": sha256(REVIEW / "PREDICTION_MATRICES.npz"),
            "matrix_hashes": {
                key: hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()
                for key, value in matrices.items()
            },
            "semantic_outcomes": 0,
        },
    )
    if not instrument_pass:
        raise RuntimeError("Q2_V4_A2_INSTRUMENT_NOT_QUALIFIED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("engine", "source", "derive-bank", "safety", "geometry"))
    parser.add_argument("--model-path")
    parser.add_argument("--prelock-commit")
    args = parser.parse_args()
    require_lock()
    if args.phase == "derive-bank":
        if not args.prelock_commit:
            parser.error("--prelock-commit is required")
        derive_bank_phase(args.prelock_commit)
        return
    if not args.model_path:
        parser.error("--model-path is required")
    backend = build_v4_backend(args.model_path)
    model_path = Path(args.model_path)
    if args.phase == "engine":
        engine_phase(backend, model_path)
    elif args.phase == "source":
        source_phase(backend)
    elif args.phase == "safety":
        if not args.prelock_commit:
            parser.error("--prelock-commit is required")
        safety_phase(backend, args.prelock_commit)
    elif args.phase == "geometry":
        if not args.prelock_commit:
            parser.error("--prelock-commit is required")
        geometry_phase(backend, args.prelock_commit)


if __name__ == "__main__":
    main()
