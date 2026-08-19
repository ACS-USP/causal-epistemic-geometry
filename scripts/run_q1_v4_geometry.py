#!/usr/bin/env python3
"""Collect the bounded V4 geometry activation/behavior diagnostic on RunPod."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.prompts import render_prompt  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    git_metadata,
    require_remote_hf_execution,
    stable_digest,
)
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    expected = stable_digest("V4-GEOMETRY-MANIFEST", canonical_json(items))
    if expected != manifest.get("manifest_hash"):
        raise ValueError("geometry manifest hash mismatch")
    if len(items) != 94 or len({item["item_id"] for item in items}) != 94:
        raise ValueError("geometry manifest must contain 94 unique items")
    return manifest


def _candidate_ids(tokenizer: Any, labels: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for label in labels:
        choices = []
        for candidate in (f" {label}", label):
            ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if len(ids) == 1:
                choices.append(int(ids[0]))
        result[label] = list(dict.fromkeys(choices))
    return result


def _forward_capture(
    backend: HuggingFaceBackend,
    item: BenchmarkItem,
    *,
    enable_thinking: bool,
    labels: list[str],
) -> tuple[np.ndarray, list[float] | None, dict[str, Any]]:
    torch = backend.torch
    rendered = render_prompt(
        item,
        mode="chat",
        tokenizer=backend.tokenizer,
        enable_thinking=enable_thinking,
    )
    encoded = backend.tokenizer(rendered.text, return_tensors="pt")
    encoded = {key: value.to(backend.device) for key, value in encoded.items()}
    last_index = int(encoded["attention_mask"][0].sum().item()) - 1
    captured: list[torch.Tensor] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(hidden[:, last_index, :].detach().float().cpu())
        return output

    handle = backend.layer_module(31).register_forward_hook(capture)
    try:
        with torch.inference_mode():
            output = backend.model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("geometry activation hook captured no output")
    activation = captured[0][0].numpy().copy()
    if not enable_thinking:
        logits = output.logits[0, last_index].float()
        candidates = _candidate_ids(backend.tokenizer, labels)
        represented: dict[str, int] = {}
        scores: list[torch.Tensor] = []
        used: set[int] = set()
        for label in labels:
            ids = candidates[label]
            if ids:
                token_id = ids[0]
                represented[label] = token_id
                used.add(token_id)
                scores.append(logits[token_id])
        if len(scores) != len(labels):
            probabilities = None
        else:
            score_tensor = torch.stack(scores)
            other_mask = torch.ones_like(logits, dtype=torch.bool)
            for token_id in used:
                other_mask[token_id] = False
            other = torch.logsumexp(logits[other_mask], dim=0)
            probabilities = torch.softmax(torch.cat((score_tensor, other.view(1))), dim=0)
            probabilities = [float(value) for value in probabilities.cpu().tolist()]
        return activation, probabilities, {
            "rendered_prompt_hash": rendered.hash,
            "candidate_token_ids": candidates,
            "represented_token_ids": represented,
            "all_candidates_single_token": len(scores) == len(labels),
        }
    return activation, None, {"rendered_prompt_hash": rendered.hash}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "instrument": "Q1_V4_GEOMETRY",
        "manifest_hash": manifest["manifest_hash"],
        "layer": 31,
        "views": ["THINKING_PROMPT_BOUNDARY_ACTIVATIONS", "DIRECT_GEOMETRY_POSITIVE_CONTROL"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bf16",
        "attention_implementation": "sdpa",
        "source_commit": (
            os.environ.get("CEG_SOURCE_COMMIT") or git_metadata(ROOT).get("git_commit")
        ),
        "steering": False,
        "generation": False,
        "holdout": False,
    }
    identity_hash = stable_digest("V4-GEOMETRY-RUN", canonical_json(identity))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity_hash") != identity_hash:
            raise RuntimeError("refusing resume: V4 geometry identity changed")
    _atomic_json(
        manifest_path,
        {
            "status": "RUNNING",
            "identity": identity,
            "identity_hash": identity_hash,
            "started_utc": datetime.now(UTC).isoformat(),
        },
    )
    require_remote_hf_execution("Q1 V4 geometry forward passes")
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        prompt_mode="chat",
        max_new_tokens=1,
        do_sample=False,
        attention_implementation="sdpa",
        enable_thinking=True,
        execution_mode="serial_reference",
        item_batch_size=1,
        batch_size=1,
        layer=31,
    )
    backend = HuggingFaceBackend(config)
    rows: list[dict[str, Any]] = []
    activations: dict[str, np.ndarray] = {}
    direct_tokenization: dict[str, Any] = {}
    for item in manifest["items"]:
        domain = str(item["domain"])
        labels = (
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if domain == "WEEKDAYS"
            else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        )
        benchmark_item = BenchmarkItem(
            id=str(item["item_id"]),
            prompt=str(item["prompt"]),
            target=str(item["answer"]),
        )
        started = time.perf_counter()
        thinking_activation, _, thinking_meta = _forward_capture(
            backend, benchmark_item, enable_thinking=True, labels=labels
        )
        direct_activation, direct_probs, direct_meta = _forward_capture(
            backend, benchmark_item, enable_thinking=False, labels=labels
        )
        activations[f"thinking__{item['item_id']}"] = thinking_activation
        activations[f"direct__{item['item_id']}"] = direct_activation
        direct_tokenization[domain] = direct_meta.get("candidate_token_ids")
        rows.append(
            {
                "item_id": item["item_id"],
                "domain": domain,
                "answer": item["answer"],
                "conceptual_index": item["conceptual_index"],
                "prompt_hash": item["prompt_hash"],
                "thinking_activation_key": f"thinking__{item['item_id']}",
                "direct_activation_key": f"direct__{item['item_id']}",
                "direct_probabilities": direct_probs,
                "thinking_meta": thinking_meta,
                "direct_meta": direct_meta,
                "forward_seconds": time.perf_counter() - started,
            }
        )
        print(f"completed {item['item_id']}", flush=True)
    np.savez_compressed(output / "activations.npz", **activations)
    (output / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    _atomic_json(output / "tokenization.json", direct_tokenization)
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior.update(
        {
            "status": "COMPLETE",
            "completed_utc": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
            "activation_file": "activations.npz",
        }
    )
    _atomic_json(manifest_path, prior)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
