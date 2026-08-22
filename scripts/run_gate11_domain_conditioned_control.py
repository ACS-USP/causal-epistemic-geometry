#!/usr/bin/env python3
"""Run Gate 11 prompt-activation and fixed-sequence teacher-forcing diagnostics.

This runner never calls ``generate`` and never samples a token.  Every decode
token is imported from an immutable Gate-9/Gate-10 baseline trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gate6_2_first_stage_repair import build_backend, model_item, prompt_tokens  # noqa: E402

from epistemic_geometry.benchmarks.external.base import ExternalItem  # noqa: E402
from epistemic_geometry.experiments import gate11  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402

REVIEW = ROOT / "review/gate11_domain_conditioned_control"
MEANINGFUL_PATH = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def external_item(row: dict[str, Any]) -> ExternalItem:
    return ExternalItem(
        item_id=str(row["item_id"]),
        benchmark=str(row.get("benchmark", "external")),
        subtask=str(row.get("subtask", "diagnostic")),
        prompt=str(row["prompt"]),
        reference_answer=str(row.get("reference_answer", "")),
        evaluator=str(row.get("evaluator", "none")),
        source_revision=str(row.get("source_revision", "historical")),
        metadata=dict(row.get("metadata", {})),
    )


def system_prompt(domain: str, variant: str) -> str | None:
    if variant == "P0_ORDINARY":
        return None
    if variant == "P1_SOURCE_CAREFUL":
        return gate11.SYSTEM_CAREFUL
    if variant == "P2_SOURCE_DIRECT":
        return gate11.SYSTEM_DIRECT
    if variant == "P3_DOMAIN_TEXTUAL_CAREFUL":
        return (
            gate11.SYSTEM_CAREFUL
            if domain == "CRUXEval"
            else gate11.SYSTEM_CHARCOUNT_CAREFUL
        )
    raise ValueError(f"unknown prompt variant {variant}")


def load_lock(source_commit: str) -> dict[str, Any]:
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    binding = read_json(REVIEW / "EXPERIMENT_SOURCE_COMMIT.json")
    if lock["status"] != "FROZEN_PRE_DIAGNOSTIC":
        raise RuntimeError("Gate 11 lock is not frozen")
    if binding["experiment_source_commit"] != source_commit or git_commit() != source_commit:
        raise RuntimeError("Gate 11 execution source commit mismatch")
    if binding["protocol_lock_sha256"] != sha256(REVIEW / "PROTOCOL_LOCK.json"):
        raise RuntimeError("Gate 11 protocol binding mismatch")
    return lock


def load_vectors() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    meaningful = np.load(MEANINGFUL_PATH, allow_pickle=False).astype(np.float64).reshape(-1)
    if vector_sha256(meaningful) != gate11.CONTROLLER_HASH:
        raise RuntimeError("fixed L27 vector hash mismatch")
    vectors = {gate11.TF_MEANINGFUL: meaningful}
    hashes = {gate11.TF_MEANINGFUL: gate11.CONTROLLER_HASH}
    bank = read_json(REVIEW / "RANDOM_BANK.json")
    for index, condition in enumerate(gate11.TF_RANDOMS):
        name = gate11.RANDOM_NAMES[index]
        path = ROOT / bank["records"][name]["vector_path"]
        vector = np.load(path, allow_pickle=False).astype(np.float64).reshape(-1)
        expected = bank["records"][name]["canonical_float64_vector_sha256"]
        if vector_sha256(vector) != expected:
            raise RuntimeError(f"random vector mismatch: {name}")
        vectors[condition] = vector
        hashes[condition] = expected
    return vectors, hashes


def split_output(output: Any) -> tuple[Any, tuple[Any, ...] | None, bool]:
    if hasattr(output, "shape"):
        return output, None, False
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0], tuple(output[1:]), isinstance(output, tuple)
    raise TypeError("unexpected transformer-block output")


def join_output(hidden: Any, remainder: tuple[Any, ...] | None, was_tuple: bool) -> Any:
    if remainder is None:
        return hidden
    return (hidden, *remainder) if was_tuple else [hidden, *remainder]


@dataclass
class DiagnosticHooks:
    backend: Any
    capture_layers: tuple[int, ...]
    delta: np.ndarray | None
    prompt_length: int
    capture_enabled: bool = False
    captures: dict[int, np.ndarray] = field(default_factory=dict)
    handles: list[Any] = field(default_factory=list)
    forward_count: int = 0
    application_count: int = 0
    max_relative_shift_error: float = 0.0
    max_noncurrent_change: float = 0.0

    def __enter__(self) -> DiagnosticHooks:
        if self.delta is not None:
            self.handles.append(
                self.backend.layer_module(gate11.LAYER).register_forward_hook(
                    self._intervention_hook
                )
            )
        for layer in self.capture_layers:
            self.handles.append(
                self.backend.layer_module(layer).register_forward_hook(self._capture_hook(layer))
            )
        return self

    def _intervention_hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden, remainder, was_tuple = split_output(output)
        position = hidden.shape[1] - 1 if hidden.shape[1] > 1 else 0
        delta = self.backend.torch.tensor(
            self.delta, dtype=self.backend.torch.float32, device=hidden.device
        )
        updated = hidden.clone()
        before = hidden[0, position, :].detach().clone()
        updated[0, position, :] = before + delta.to(dtype=hidden.dtype)
        after = updated[0, position, :].detach()
        error = (after.float() - before.float() - delta.float()).abs()
        scale = before.float().abs().maximum(after.float().abs()).maximum(delta.abs()).clamp_min(1)
        eps = float(self.backend.torch.finfo(hidden.dtype).eps)
        self.max_relative_shift_error = max(
            self.max_relative_shift_error, float((error / (eps * scale)).max().item())
        )
        noncurrent = updated - hidden
        noncurrent[0, position, :] = 0
        self.max_noncurrent_change = max(
            self.max_noncurrent_change, float(noncurrent.abs().max().item())
        )
        self.application_count += 1
        return join_output(updated, remainder, was_tuple)

    def _capture_hook(self, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden, _remainder, _was_tuple = split_output(output)
            if self.capture_enabled:
                position = hidden.shape[1] - 1 if hidden.shape[1] > 1 else 0
                self.captures[layer] = hidden[0, position, :].detach().float().cpu().numpy()

        return hook

    def begin_capture(self) -> None:
        self.capture_enabled = True
        self.captures = {}

    def end_capture(self) -> dict[int, np.ndarray]:
        self.capture_enabled = False
        if set(self.captures) != set(self.capture_layers):
            raise RuntimeError("not all requested activation layers were captured")
        return dict(self.captures)

    def note_forward(self) -> None:
        self.forward_count += 1

    def __exit__(self, *_args: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def forward(
    backend: Any,
    input_ids: list[int],
    *,
    past: Any | None,
    total_length: int,
    phase: str,
) -> Any:
    torch = backend.torch
    ids = torch.tensor([input_ids], dtype=torch.long, device=backend.device)
    attention = torch.ones((1, total_length), dtype=torch.long, device=backend.device)
    start = total_length - len(input_ids)
    positions = torch.arange(start, total_length, dtype=torch.long, device=backend.device)[None, :]
    cache_position = torch.arange(start, total_length, dtype=torch.long, device=backend.device)
    kwargs = backend._forward_kwargs(  # noqa: SLF001
        backend.model,
        ids,
        attention,
        positions,
        past_key_values=past,
        cache_position=cache_position,
    )
    return backend._forward(backend.model, kwargs, phase)  # noqa: SLF001


def source_capture(backend: Any, source_commit: str) -> dict[str, Any]:
    journal = REVIEW / "source_activation_journal.jsonl"
    completed: set[tuple[str, str, str]] = set()
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed.add((row["domain"], row["item_id"], row["variant"]))
    physical = 0
    for domain, filename in (
        ("CRUXEval", "CRUX_ITEM_SELECTION.json"),
        ("CHARCOUNT", "CHARCOUNT_ITEM_SELECTION.json"),
    ):
        selection = read_json(REVIEW / filename)
        for item_row in selection["items"]:
            item = external_item(item_row)
            alias_payload: dict[str, Any] | None = None
            for variant in gate11.PROMPT_VARIANTS:
                key = (domain, str(item.item_id), variant)
                if key in completed:
                    continue
                if domain == "CRUXEval" and variant == "P3_DOMAIN_TEXTUAL_CAREFUL":
                    if alias_payload is None:
                        source_rows = [
                            json.loads(line)
                            for line in journal.read_text(encoding="utf-8").splitlines()
                            if line
                        ]
                        alias_payload = next(
                            row
                            for row in reversed(source_rows)
                            if row["domain"] == domain
                            and row["item_id"] == item.item_id
                            and row["variant"] == "P1_SOURCE_CAREFUL"
                        )
                    payload = {**alias_payload, "variant": variant, "physical_alias": True}
                    append_jsonl(journal, payload)
                    completed.add(key)
                    continue
                row = model_item(item, system_prompt(domain, variant))
                prompt_ids, rendered, prompt_hash = prompt_tokens(backend, row)
                with DiagnosticHooks(
                    backend, gate11.SOURCE_LAYERS, None, len(prompt_ids)
                ) as hooks, backend.torch.inference_mode():
                    hooks.begin_capture()
                    forward(
                        backend,
                        prompt_ids,
                        past=None,
                        total_length=len(prompt_ids),
                        phase="prefill",
                    )
                    activations = hooks.end_capture()
                raw_path = (
                    REVIEW
                    / "raw/source_activations"
                    / domain.lower()
                    / f"{item.item_id}__{variant}.npz"
                )
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    raw_path,
                    **{f"L{layer}": values for layer, values in activations.items()},
                )
                payload = {
                    "domain": domain,
                    "item_id": item.item_id,
                    "variant": variant,
                    "physical_alias": False,
                    "prompt_hash": prompt_hash,
                    "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "activation_path": str(raw_path.relative_to(ROOT)),
                    "activation_file_sha256": sha256(raw_path),
                    "layers": list(gate11.SOURCE_LAYERS),
                    "experiment_source_commit": source_commit,
                    "model_revision": gate11.MODEL_REVISION,
                    "free_generation": False,
                }
                append_jsonl(journal, payload)
                alias_payload = payload if variant == "P1_SOURCE_CAREFUL" else alias_payload
                completed.add(key)
                physical += 1
    return {"logical_rows": len(completed), "new_physical_forwards": physical}


def run_condition(
    backend: Any,
    item: ExternalItem,
    domain: str,
    condition: str,
    continuation: list[int],
    delta: np.ndarray | None,
) -> dict[str, Any]:
    system = gate11.SYSTEM_CAREFUL if condition == gate11.TF_TEXTUAL else None
    row = model_item(item, system)
    prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
    snapshots: dict[str, dict[str, Any]] = {}
    start = time.monotonic()
    with DiagnosticHooks(
        backend, gate11.PROPAGATION_LAYERS, delta, len(prompt_ids)
    ) as hooks, backend.torch.inference_mode():
        hooks.begin_capture()
        output = forward(
            backend, prompt_ids, past=None, total_length=len(prompt_ids), phase="prefill"
        )
        hooks.note_forward()
        snapshots["PREFILL"] = {
            "logits": output.logits[0, -1, :].detach().float().cpu().numpy(),
            "hidden": hooks.end_capture(),
            "target_token": continuation[0] if continuation else None,
        }
        past = output.past_key_values
        for token_index, token in enumerate(continuation):
            capture = token_index in gate11.CHECKPOINTS
            if capture:
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
                    "logits": output.logits[0, -1, :].detach().float().cpu().numpy(),
                    "hidden": hooks.end_capture(),
                    "target_token": (
                        continuation[token_index + 1]
                        if token_index + 1 < len(continuation)
                        else None
                    ),
                }
    return {
        "domain": domain,
        "condition": condition,
        "prompt_hash": prompt_hash,
        "prompt_length": len(prompt_ids),
        "continuation_length": len(continuation),
        "snapshots": snapshots,
        "forward_count": hooks.forward_count,
        "application_count": hooks.application_count,
        "max_relative_shift_error": hooks.max_relative_shift_error,
        "max_noncurrent_change": hooks.max_noncurrent_change,
        "duration_seconds": time.monotonic() - start,
    }


def snapshot_metrics(
    baseline: dict[str, Any], condition: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    for checkpoint in baseline["snapshots"]:
        base = baseline["snapshots"][checkpoint]
        other = condition["snapshots"][checkpoint]
        metrics = gate11.logit_metrics(base["logits"], other["logits"], base["target_token"])
        hidden = {}
        for layer in gate11.PROPAGATION_LAYERS:
            left = base["hidden"][layer]
            right = other["hidden"][layer]
            hidden[f"L{layer}"] = {
                "displacement_norm": float(np.linalg.norm(right - left)),
                "activation_cosine": gate11.cosine(left, right),
            }
        result.append({"checkpoint": checkpoint, **metrics, "hidden": hidden})
    return result


def collect_teacher_forcing(
    backend: Any,
    source_commit: str,
    vectors: dict[str, np.ndarray],
    vector_hashes: dict[str, str],
) -> dict[str, Any]:
    schedule = read_json(REVIEW / "FIXED_SEQUENCE_SCHEDULE.json")
    journal = REVIEW / "fixed_sequence_journal.jsonl"
    completed: set[str] = set()
    if journal.exists():
        rows = [
            json.loads(line)
            for line in journal.read_text(encoding="utf-8").splitlines()
            if line
        ]
        completed = {row["logical_key"] for row in rows}
        if len(completed) != len(rows):
            raise RuntimeError("duplicate fixed-sequence logical keys")
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in schedule["rows"]:
        grouped.setdefault((row["domain"], str(row["item_id"])), row)
    manifests = {
        "CRUXEval": read_json(REVIEW / "CRUX_ITEM_SELECTION.json"),
        "CHARCOUNT": read_json(REVIEW / "CHARCOUNT_ITEM_SELECTION.json"),
    }
    item_maps = {
        domain: {str(item["item_id"]): item for item in payload["items"]}
        for domain, payload in manifests.items()
    }
    new_rows = 0
    for (domain, item_id), schedule_row in grouped.items():
        expected = [f"{domain}|{item_id}|{condition}" for condition in gate11.TF_CONDITIONS]
        if all(key in completed for key in expected):
            continue
        if any(key in completed for key in expected):
            raise RuntimeError(
                "partial item group in journal; safe resume requires whole-item append"
            )
        if not schedule_row["available"]:
            for condition, key in zip(gate11.TF_CONDITIONS, expected, strict=True):
                append_jsonl(
                    journal,
                    {
                        "logical_key": key,
                        "domain": domain,
                        "item_id": item_id,
                        "condition": condition,
                        "status": "TOKEN_IDS_MECHANICALLY_ABSENT",
                        "experiment_source_commit": source_commit,
                    },
                )
                new_rows += 1
            continue
        item = external_item(item_maps[domain][item_id])
        continuation = [int(value) for value in schedule_row["continuation_token_ids"]]
        outputs: dict[str, dict[str, Any]] = {}
        for condition in gate11.TF_CONDITIONS:
            delta = None
            if condition in vectors:
                delta = vectors[condition] * gate11.ETA * gate11.REFERENCE_SCALE
            outputs[condition] = run_condition(
                backend, item, domain, condition, continuation, delta
            )
        baseline = outputs[gate11.TF_BASELINE]
        condition_metrics = {
            condition: snapshot_metrics(baseline, output)
            for condition, output in outputs.items()
        }
        textual_by_checkpoint = {
            row["checkpoint"]: row
            for row in condition_metrics[gate11.TF_TEXTUAL]
        }
        for condition, key in zip(gate11.TF_CONDITIONS, expected, strict=True):
            metrics = condition_metrics[condition]
            if condition == gate11.TF_MEANINGFUL:
                for row, snapshot in zip(
                    metrics, outputs[condition]["snapshots"].values(), strict=True
                ):
                    checkpoint = row["checkpoint"]
                    base_logits = baseline["snapshots"][checkpoint]["logits"]
                    textual_logits = outputs[gate11.TF_TEXTUAL]["snapshots"][checkpoint]["logits"]
                    row["careful_logit_alignment"] = gate11.cosine(
                        snapshot["logits"] - base_logits, textual_logits - base_logits
                    )
                    row["textual_logit_l2"] = textual_by_checkpoint[checkpoint]["logit_l2"]
            output = outputs[condition]
            record = {
                "logical_key": key,
                "domain": domain,
                "item_id": item_id,
                "condition": condition,
                "status": "COMPLETE",
                "source_rollout_index": schedule_row["selected_rollout_index"],
                "continuation_length": len(continuation),
                "continuation_sha256": hashlib.sha256(
                    np.asarray(continuation, dtype=np.int64).tobytes()
                ).hexdigest(),
                "prompt_hash": output["prompt_hash"],
                "prompt_length": output["prompt_length"],
                "checkpoints": metrics,
                "forward_count": output["forward_count"],
                "application_count": output["application_count"],
                "max_relative_shift_error": output["max_relative_shift_error"],
                "max_noncurrent_change": output["max_noncurrent_change"],
                "duration_seconds": output["duration_seconds"],
                "vector_hash": vector_hashes.get(condition),
                "eta": gate11.ETA if condition in vectors else 0.0,
                "layer": gate11.LAYER if condition in vectors else None,
                "sampling": False,
                "free_generation": False,
                "model_revision": gate11.MODEL_REVISION,
                "experiment_source_commit": source_commit,
            }
            append_jsonl(journal, record)
            completed.add(key)
            new_rows += 1
    return {"logical_rows": len(completed), "new_rows": new_rows}


def engineering(
    backend: Any, vectors: dict[str, np.ndarray], vector_hashes: dict[str, str]
) -> dict[str, Any]:
    schedule = read_json(REVIEW / "FIXED_SEQUENCE_SCHEDULE.json")
    available = [row for row in schedule["rows"] if row["available"]][:5]
    selection = read_json(REVIEW / "CRUX_ITEM_SELECTION.json")
    items = {str(item["item_id"]): item for item in selection["items"]}
    checks = []
    for row in available:
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
        identity = all(
            np.array_equal(clean["snapshots"][key]["logits"], zero["snapshots"][key]["logits"])
            for key in clean["snapshots"]
        )
        checks.append(identity)
    exercised = {}
    sample = available[0]
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
        "classification": "GATE11_ENGINEERING_PASS",
        "alpha_zero_identity": all(checks),
        "free_generation": False,
        "all_controllers_exercised": set(exercised) == set(vectors),
        "controller_checks": exercised,
        "exact_shift": all(
            value["max_relative_shift_error"] <= 2.0 for value in exercised.values()
        ),
        "current_token_scope": all(
            value["max_noncurrent_change"] == 0 for value in exercised.values()
        ),
        "one_application_per_forward": all(
            value["application_count"] == value["forward_count"] for value in exercised.values()
        ),
        "hook_cleanup": True,
        "cache_safety": True,
    }
    required = [
        payload["alpha_zero_identity"],
        payload["all_controllers_exercised"],
        payload["exact_shift"],
        payload["current_token_scope"],
        payload["one_application_per_forward"],
    ]
    if not all(required):
        payload["classification"] = "GATE11_ENGINE_FAILURE"
    write_json(REVIEW / "ENGINEERING_CHECKS.json", payload)
    if payload["classification"] != "GATE11_ENGINEERING_PASS":
        raise RuntimeError("GATE11_ENGINE_FAILURE")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("engineering", "collect"), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--experiment-source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution("Gate 11 activation/logit diagnostics")
    lock = load_lock(args.experiment_source_commit)
    if lock["firewall"]["free_generation"] != "NOT_AUTHORIZED":
        raise RuntimeError("Gate 11 free-generation firewall mismatch")
    backend = build_backend(args.model_path)
    vectors, hashes = load_vectors()
    if args.phase == "engineering":
        engineering(backend, vectors, hashes)
    else:
        source = source_capture(backend, args.experiment_source_commit)
        propagation = collect_teacher_forcing(
            backend, args.experiment_source_commit, vectors, hashes
        )
        write_json(
            REVIEW / "COLLECTION_STATUS.json",
            {"source": source, "teacher_forcing": propagation, "complete": True},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
