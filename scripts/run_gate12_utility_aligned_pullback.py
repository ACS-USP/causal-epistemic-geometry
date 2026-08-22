#!/usr/bin/env python3
"""Collect exact Gate-12 JVP geometry without reading historical outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from run_gate6_2_first_stage_repair import build_backend, model_item, prompt_tokens  # noqa: E402
from run_gate11_domain_conditioned_control import (  # noqa: E402
    external_item,
    run_condition,
)

from epistemic_geometry.experiments import gate11, gate12  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402

REVIEW = ROOT / "review/gate12_utility_aligned_pullback"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_is_ancestor(source: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source, "HEAD"], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def load_lock(source_commit: str) -> dict[str, Any]:
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    if lock["status"] != "FROZEN_PRE_GEOMETRY":
        raise RuntimeError("Gate-12 lock is not frozen")
    if lock["experiment_source_commit"] != source_commit or not source_is_ancestor(source_commit):
        raise RuntimeError("Gate-12 source-commit provenance mismatch")
    return lock


def split_output(output: Any) -> tuple[Any, tuple[Any, ...] | None, bool]:
    if hasattr(output, "shape"):
        return output, None, False
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0], tuple(output[1:]), isinstance(output, tuple)
    raise TypeError("unexpected transformer block output")


def join_output(hidden: Any, remainder: tuple[Any, ...] | None, was_tuple: bool) -> Any:
    if remainder is None:
        return hidden
    return (hidden, *remainder) if was_tuple else [hidden, *remainder]


class AlphaHook(AbstractContextManager["AlphaHook"]):
    """Apply scalar alpha times one unit direction at frozen sequence positions."""

    def __init__(self, backend: Any, alpha: Any, vector: Any, mask: Any) -> None:
        self.backend = backend
        self.alpha = alpha
        self.vector = vector
        self.mask = mask
        self.handle: Any | None = None
        self.application_count = 0

    def __enter__(self) -> AlphaHook:
        self.handle = self.backend.layer_module(gate12.LAYER).register_forward_hook(self.hook)
        return self

    def hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden, remainder, was_tuple = split_output(output)
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        mask = self.mask.to(device=hidden.device, dtype=hidden.dtype).view(1, -1, 1)
        updated = hidden + mask * self.alpha.to(dtype=hidden.dtype) * vector
        self.application_count += 1
        return join_output(updated, remainder, was_tuple)

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class FullSequenceJVP:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.torch = backend.torch

    def tensors(
        self, prompt_ids: list[int], continuation_inputs: list[int], output_offsets: list[int]
    ) -> dict[str, Any]:
        values = prompt_ids + continuation_inputs
        torch = self.torch
        ids = torch.tensor([values], dtype=torch.long, device=self.backend.device)
        attention = torch.ones_like(ids)
        positions = torch.arange(len(values), dtype=torch.long, device=self.backend.device)[None, :]
        mask = torch.zeros(len(values), dtype=torch.float32, device=self.backend.device)
        mask[len(prompt_ids) - 1 :] = 1.0
        output_positions = torch.tensor(
            [len(prompt_ids) - 1 + offset for offset in output_offsets],
            dtype=torch.long,
            device=self.backend.device,
        )
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "position_ids": positions,
            "intervention_mask": mask,
            "output_positions": output_positions,
        }

    def _base_logits(self, tensors: dict[str, Any], alpha: Any, vector: Any) -> Any:
        with AlphaHook(self.backend, alpha, vector, tensors["intervention_mask"]) as hook:
            output = self.backend.model.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                position_ids=tensors["position_ids"],
                use_cache=False,
                return_dict=True,
            )
            selected = output.last_hidden_state[:, tensors["output_positions"], :]
            logits = self.backend.model.lm_head(selected)[0]
        if hook.application_count != 1:
            raise RuntimeError("full-sequence intervention must apply once per model forward")
        return logits

    def regular(self, tensors: dict[str, Any], alpha: float, vector: np.ndarray) -> np.ndarray:
        torch = self.torch
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=self.backend.device)
        vector_tensor = torch.tensor(vector, dtype=torch.float32, device=self.backend.device)
        with torch.inference_mode():
            logits = self._base_logits(tensors, alpha_tensor, vector_tensor)
        return logits.detach().float().cpu().numpy()

    def jvp(self, tensors: dict[str, Any], vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact forward-mode derivative with respect to scalar alpha at zero."""

        torch = self.torch
        vector_tensor = torch.tensor(vector, dtype=torch.float32, device=self.backend.device)
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            primal_alpha = torch.tensor(0.0, dtype=torch.float32, device=self.backend.device)
            tangent_alpha = torch.tensor(1.0, dtype=torch.float32, device=self.backend.device)
            dual_alpha = torch.autograd.forward_ad.make_dual(primal_alpha, tangent_alpha)
            dual_logits = self._base_logits(tensors, dual_alpha, vector_tensor)
            primal, tangent = torch.autograd.forward_ad.unpack_dual(dual_logits)
        if tangent is None:
            raise RuntimeError("exact forward-mode JVP returned no tangent")
        return (
            primal.detach().float().cpu().numpy(),
            tangent.detach().float().cpu().numpy(),
        )


def load_directions(component: str, domain: str) -> tuple[list[str], list[str], list[np.ndarray]]:
    payload = read_json(REVIEW / "HISTORICAL_DIRECTION_MANIFEST.json")
    records = (
        payload["control_validation"]
        if component == "CONTROL_VALIDATION"
        else payload["utility_prediction"][domain]
    )
    labels: list[str] = []
    hashes: list[str] = []
    vectors: list[np.ndarray] = []
    for record in records:
        path = ROOT / record["vector_path"]
        if sha256(path) != record["file_sha256"]:
            raise RuntimeError(f"direction file hash mismatch: {path}")
        vector = np.load(path, allow_pickle=False).astype(np.float64).reshape(-1)
        if vector_sha256(vector) != record["canonical_float64_sha256"]:
            raise RuntimeError(f"canonical direction mismatch: {record['label']}")
        labels.append(record["label"])
        hashes.append(record["canonical_float64_sha256"])
        vectors.append(vector)
    return labels, hashes, vectors


def system_careful(domain: str) -> str:
    return gate11.SYSTEM_CAREFUL if domain == "CRUXEval" else gate11.SYSTEM_CHARCOUNT_CAREFUL


def encode_prompts(backend: Any, item_row: dict[str, Any]) -> tuple[list[int], list[int], str, str]:
    item = external_item(item_row)
    ordinary = model_item(item, None)
    careful = model_item(
        item, system_careful("CRUXEval" if item.benchmark == "CRUXEval" else "CHARCOUNT")
    )
    ordinary_ids, _ordinary_rendered, ordinary_hash = prompt_tokens(backend, ordinary)
    careful_ids, _careful_rendered, careful_hash = prompt_tokens(backend, careful)
    return ordinary_ids, careful_ids, ordinary_hash, careful_hash


def persist_shard(path: Path, arrays: dict[str, Any]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256(path), path.stat().st_size


def manifest_entries() -> dict[str, dict[str, Any]]:
    path = REVIEW / "RAW_GEOMETRY_MANIFEST.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return {entry["logical_item_key"]: entry for entry in payload.get("entries", [])}


def update_manifest(entries: dict[str, dict[str, Any]], status: str) -> None:
    write_json(
        REVIEW / "RAW_GEOMETRY_MANIFEST.json",
        {
            "schema_version": 1,
            "status": status,
            "expected_shards": 112,
            "raw_dtype": gate12.RAW_DTYPE,
            "entries": [entries[key] for key in sorted(entries)],
        },
    )


def collect_control(backend: Any, engine: FullSequenceJVP, source_commit: str) -> int:
    payload = read_json(REVIEW / "CONTROL_VALIDATION_ITEMS.json")
    labels, hashes, vectors = load_directions("CONTROL_VALIDATION", "CRUXEval")
    entries = manifest_entries()
    created = 0
    for row in payload["items"]:
        domain = row["domain"]
        item_id = str(row["item_id"])
        key = f"CONTROL_VALIDATION|{domain}|{item_id}"
        if key in entries:
            path = ROOT / entries[key]["path"]
            if path.exists() and sha256(path) == entries[key]["sha256"]:
                continue
            raise RuntimeError(f"corrupt completed geometry shard: {key}")
        ordinary_ids, careful_ids, ordinary_hash, careful_hash = encode_prompts(
            backend, row["item"]
        )
        continuation = [int(value) for value in row["continuation_token_ids"]]
        checkpoints = [int(value) for value in row["checkpoints"]]
        offsets = [0 if value == -1 else value + 1 for value in checkpoints]
        ordinary_tensors = engine.tensors(ordinary_ids, continuation, offsets)
        careful_tensors = engine.tensors(careful_ids, continuation, offsets)
        primals = []
        derivatives = []
        for vector in vectors:
            primal, derivative = engine.jvp(ordinary_tensors, vector)
            primals.append(primal)
            derivatives.append(derivative)
        baseline = primals[0]
        if not all(np.array_equal(baseline, value) for value in primals[1:]):
            raise RuntimeError("direction-dependent alpha-zero primal")
        careful = engine.regular(careful_tensors, 0.0, vectors[0])
        path = REVIEW / "raw_geometry" / f"control__{domain.lower()}__{item_id}.npz"
        digest, size = persist_shard(
            path,
            {
                "baseline_logits": baseline.astype(np.float32),
                "careful_logits": careful.astype(np.float32),
                "jvp_vectors": np.stack(derivatives).astype(np.float32),
                "direction_labels": np.asarray(labels),
                "direction_hashes": np.asarray(hashes),
                "prompt_token_ids": np.asarray(ordinary_ids, dtype=np.int64),
                "careful_prompt_token_ids": np.asarray(careful_ids, dtype=np.int64),
                "continuation_token_ids": np.asarray(continuation, dtype=np.int64),
                "target_token_ids": np.full(len(checkpoints), -1, dtype=np.int64),
                "checkpoint_indices": np.asarray(checkpoints, dtype=np.int64),
                "attention_mask": ordinary_tensors["attention_mask"].cpu().numpy(),
                "position_ids": ordinary_tensors["position_ids"].cpu().numpy(),
                "intervention_mask": ordinary_tensors["intervention_mask"].cpu().numpy(),
            },
        )
        entries[key] = {
            "logical_item_key": key,
            "component": "CONTROL_VALIDATION",
            "domain": domain,
            "item_id": item_id,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "bytes": size,
            "positions": len(checkpoints),
            "directions": len(labels),
            "ordinary_prompt_hash": ordinary_hash,
            "careful_prompt_hash": careful_hash,
            "source_commit": source_commit,
            "model_revision": gate12.MODEL_REVISION,
        }
        update_manifest(entries, "IN_PROGRESS")
        created += 1
    return created


def collect_utility(backend: Any, engine: FullSequenceJVP, source_commit: str) -> int:
    payload = read_json(REVIEW / "UTILITY_PREDICTION_ITEMS.json")
    entries = manifest_entries()
    created = 0
    for domain in ("CRUXEval", "CHARCOUNT"):
        labels, hashes, vectors = load_directions("UTILITY_PREDICTION", domain)
        for row in payload[domain]["items"]:
            item_id = str(row["item_id"])
            key = f"UTILITY_PREDICTION|{domain}|{item_id}"
            if key in entries:
                path = ROOT / entries[key]["path"]
                if path.exists() and sha256(path) == entries[key]["sha256"]:
                    continue
                raise RuntimeError(f"corrupt completed geometry shard: {key}")
            ordinary_ids, careful_ids, ordinary_hash, careful_hash = encode_prompts(backend, row)
            continuation = backend.tokenizer(
                row["canonical_correct_continuation"], add_special_tokens=False
            ).input_ids
            continuation = [int(value) for value in continuation]
            if not continuation:
                raise RuntimeError(f"empty canonical answer tokenization: {domain}|{item_id}")
            continuation_inputs = continuation[:-1]
            offsets = list(range(len(continuation)))
            ordinary_tensors = engine.tensors(ordinary_ids, continuation_inputs, offsets)
            careful_tensors = engine.tensors(careful_ids, continuation_inputs, offsets)
            primals = []
            derivatives = []
            for vector in vectors:
                primal, derivative = engine.jvp(ordinary_tensors, vector)
                primals.append(primal)
                derivatives.append(derivative)
            baseline = primals[0]
            if not all(np.array_equal(baseline, value) for value in primals[1:]):
                raise RuntimeError("direction-dependent utility alpha-zero primal")
            careful = engine.regular(careful_tensors, 0.0, vectors[0])
            path = REVIEW / "raw_geometry" / f"utility__{domain.lower()}__{item_id}.npz"
            digest, size = persist_shard(
                path,
                {
                    "baseline_logits": baseline.astype(np.float32),
                    "careful_logits": careful.astype(np.float32),
                    "jvp_vectors": np.stack(derivatives).astype(np.float32),
                    "direction_labels": np.asarray(labels),
                    "direction_hashes": np.asarray(hashes),
                    "prompt_token_ids": np.asarray(ordinary_ids, dtype=np.int64),
                    "careful_prompt_token_ids": np.asarray(careful_ids, dtype=np.int64),
                    "continuation_token_ids": np.asarray(continuation, dtype=np.int64),
                    "target_token_ids": np.asarray(continuation, dtype=np.int64),
                    "checkpoint_indices": np.arange(len(continuation), dtype=np.int64),
                    "attention_mask": ordinary_tensors["attention_mask"].cpu().numpy(),
                    "position_ids": ordinary_tensors["position_ids"].cpu().numpy(),
                    "intervention_mask": ordinary_tensors["intervention_mask"].cpu().numpy(),
                },
            )
            entries[key] = {
                "logical_item_key": key,
                "component": "UTILITY_PREDICTION",
                "domain": domain,
                "item_id": item_id,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "bytes": size,
                "positions": len(continuation),
                "directions": len(labels),
                "ordinary_prompt_hash": ordinary_hash,
                "careful_prompt_hash": careful_hash,
                "canonical_continuation_sha256": hashlib.sha256(
                    row["canonical_correct_continuation"].encode()
                ).hexdigest(),
                "source_commit": source_commit,
                "model_revision": gate12.MODEL_REVISION,
            }
            update_manifest(entries, "IN_PROGRESS")
            created += 1
    return created


def relative_median(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.maximum(np.abs(right), 1e-12)
    return float(np.median(np.abs(left - right) / denominator))


def engineering(backend: Any, engine: FullSequenceJVP) -> dict[str, Any]:
    control = read_json(REVIEW / "CONTROL_VALIDATION_ITEMS.json")["items"]
    selected = []
    for domain in ("CRUXEval", "CHARCOUNT"):
        selected.extend([row for row in control if row["domain"] == domain][:2])
    labels, hashes, vectors = load_directions("CONTROL_VALIDATION", "CRUXEval")
    vector = vectors[0]
    jvp_cosines: list[float] = []
    q_relative: list[float] = []
    u_relative: list[float] = []
    quadratic_relative: list[float] = []
    equivalence_logit_max: list[float] = []
    equivalence_kl_max: list[float] = []
    for row in selected:
        domain = row["domain"]
        continuation = [int(value) for value in row["continuation_token_ids"]][:8]
        checkpoints = [-1, 0, 1, 3, 7]
        checkpoints = [value for value in checkpoints if value == -1 or value < len(continuation)]
        offsets = [0 if value == -1 else value + 1 for value in checkpoints]
        item = external_item(row["item"])
        ordinary_ids, _careful_ids, _ordinary_hash, _careful_hash = encode_prompts(
            backend, row["item"]
        )
        tensors = engine.tensors(ordinary_ids, continuation, offsets)
        baseline, derivative = engine.jvp(tensors, vector)
        finite_d75 = engine.regular(tensors, gate12.D75_SCALAR, vector)
        sequential_base = run_condition(
            backend, item, domain, gate11.TF_BASELINE, continuation, None
        )
        sequential_d75 = run_condition(
            backend,
            item,
            domain,
            gate11.TF_MEANINGFUL,
            continuation,
            vector * gate12.D75_SCALAR,
        )
        sequence_labels = ["PREFILL" if value == -1 else str(value) for value in checkpoints]
        seq_base = np.stack(
            [sequential_base["snapshots"][value]["logits"] for value in sequence_labels]
        )
        seq_d75 = np.stack(
            [sequential_d75["snapshots"][value]["logits"] for value in sequence_labels]
        )
        equivalence_logit_max.append(float(np.max(np.abs(baseline - seq_base))))
        equivalence_logit_max.append(float(np.max(np.abs(finite_d75 - seq_d75))))
        equivalence_kl_max.append(
            float(
                np.max(
                    np.abs(
                        gate12.categorical_kl(baseline, finite_d75)
                        - gate12.categorical_kl(seq_base, seq_d75)
                    )
                )
            )
        )
        target_ids = np.asarray(
            [
                continuation[0]
                if value == -1
                else continuation[min(value + 1, len(continuation) - 1)]
                for value in checkpoints
            ],
            dtype=np.int64,
        )
        q = gate12.fisher_energy(baseline, derivative)
        u = gate12.utility_slope(baseline, derivative, target_ids)
        for divisor in gate12.FINITE_DIFFERENCE_DIVISORS:
            epsilon = gate12.D75_SCALAR / divisor
            plus = engine.regular(tensors, epsilon, vector)
            minus = engine.regular(tensors, -epsilon, vector)
            finite = (plus - minus) / (2 * epsilon)
            flat_left = derivative.reshape(-1)
            flat_right = finite.reshape(-1)
            jvp_cosines.append(
                float(
                    np.dot(flat_left, flat_right)
                    / (np.linalg.norm(flat_left) * np.linalg.norm(flat_right))
                )
            )
            q_relative.append(relative_median(gate12.fisher_energy(baseline, finite), q))
            u_relative.append(
                relative_median(gate12.utility_slope(baseline, finite, target_ids), u)
            )
            quadratic = 2 * gate12.categorical_kl(baseline, plus) / epsilon**2
            quadratic_relative.append(relative_median(quadratic, q))
    payload = {
        "classification": "GATE12_DIFFERENTIABLE_ENGINEERING_PASS",
        "full_sequence_kv_cache": {
            "max_abs_logit_difference": max(equivalence_logit_max),
            "max_abs_kl_difference": max(equivalence_kl_max),
            "pass": max(equivalence_logit_max) <= 0.25 and max(equivalence_kl_max) <= 0.01,
        },
        "exact_jvp": True,
        "jvp_method": "torch.autograd.forward_ad",
        "median_jvp_cosine": float(np.median(jvp_cosines)),
        "median_relative_fisher_difference": float(np.median(q_relative)),
        "median_relative_utility_difference": float(np.median(u_relative)),
        "median_local_kl_quadratic_difference": float(np.median(quadratic_relative)),
        "direction_hashes": hashes,
        "outcome_firewall": "runner source contains no historical journal path",
    }
    passed = (
        payload["full_sequence_kv_cache"]["pass"]
        and payload["median_jvp_cosine"] >= 0.995
        and payload["median_relative_fisher_difference"] <= 0.10
        and payload["median_relative_utility_difference"] <= 0.10
        and payload["median_local_kl_quadratic_difference"] <= 0.10
    )
    if not passed:
        payload["classification"] = "GATE12_JVP_ENGINE_FAILURE"
    write_json(REVIEW / "NUMERICAL_VALIDATION.json", payload)
    if not passed:
        raise RuntimeError("GATE12_JVP_ENGINE_FAILURE")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("engineering", "collect"), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--experiment-source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution("Gate 12 exact directional JVP geometry")
    load_lock(args.experiment_source_commit)
    backend = build_backend(args.model_path)
    engine = FullSequenceJVP(backend)
    started = time.monotonic()
    if args.phase == "engineering":
        result = engineering(backend, engine)
    else:
        created_control = collect_control(backend, engine, args.experiment_source_commit)
        created_utility = collect_utility(backend, engine, args.experiment_source_commit)
        entries = manifest_entries()
        update_manifest(entries, "COMPLETE" if len(entries) == 112 else "IN_PROGRESS")
        result = {
            "status": "COMPLETE" if len(entries) == 112 else "IN_PROGRESS",
            "shards": len(entries),
            "new_control_shards": created_control,
            "new_utility_shards": created_utility,
        }
    result["duration_seconds"] = time.monotonic() - started
    result["effective_checkout"] = git_commit()
    write_json(REVIEW / f"{args.phase.upper()}_STATUS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
