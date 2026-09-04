#!/usr/bin/env python3
"""Capture label-free Q3.1 prompt representations on the closed Q2 panel.

The runner has no semantic-generation path and reads a private prompt-only
manifest that contains neither references nor correctness.  It verifies the
prospective precheck, the qualified Spark-1 stack, and every frozen model byte
before loading Qwen.  The only model operation is an unsteered prompt forward.
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

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

REVIEW = ROOT / "review/q3_route_a_prompt_representation"
PRECHECK = REVIEW / "Q3_ROUTE_A_PROMPT_REPRESENTATION_PRECHECK.json"
MODEL_MANIFEST = ROOT / "review/q2_v4_spark1_presemantic/EXACT_MODEL_MANIFEST.json"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_MANIFEST_SHA256 = "cedc88ba2f732baea6bb71f5e6d7f6bc3aad00d302c3456d208a21687c9e069c"
ENVIRONMENT_FINGERPRINT = "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
EXPECTED_WIDTH = 4096
LAYER = 27


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_model_bytes(model_path: Path) -> dict[str, Any]:
    if sha256_file(MODEL_MANIFEST) != MODEL_MANIFEST_SHA256:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: model manifest")
    manifest = read_json(MODEL_MANIFEST)
    if manifest.get("model") != MODEL or manifest.get("revision") != MODEL_REVISION:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: model identity")
    rows = list(manifest["files"])
    expected_paths = {str(row["path"]) for row in rows}
    observed_paths = {
        str(path.relative_to(model_path))
        for path in model_path.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    if observed_paths != expected_paths:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: model file set")
    total = 0
    for row in rows:
        path = model_path / str(row["path"])
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: model bytes")
        total += path.stat().st_size
    if total != int(manifest["total_bytes"]):
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: model size")
    return {
        "manifest_sha256": MODEL_MANIFEST_SHA256,
        "file_count": len(rows),
        "total_bytes": total,
    }


def verify_environment(model_path: Path) -> dict[str, Any]:
    import torch

    checks = {
        "hostname_spark1": platform.node().split(".", 1)[0] == "spark1",
        "architecture_aarch64": platform.machine() == "aarch64",
        "python_3_12_3": platform.python_version() == "3.12.3",
        "torch_exact": torch.__version__ == "2.13.0+cu130",
        "torch_cuda_13": torch.version.cuda == "13.0",
        "transformers_exact": importlib.metadata.version("transformers") == "4.57.6",
        "cuda_available": bool(torch.cuda.is_available()),
        "one_gpu": torch.cuda.device_count() == 1,
        "gpu_gb10": torch.cuda.device_count() == 1 and "GB10" in torch.cuda.get_device_name(0),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "sdpa_available": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
        "execution_profile": os.environ.get("CEG_EXECUTION_PROFILE") == "SPARK1",
        "hf_home": os.environ.get("HF_HOME") == "/srv/shared/hf-cache",
        "model_path": model_path.is_dir(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: environment {checks}")
    return {
        "checks": checks,
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "sdpa_available": True,
        "qualified_environment_fingerprint": ENVIRONMENT_FINGERPRINT,
        "model_bytes": verify_model_bytes(model_path),
    }


def build_backend(model_path: Path) -> HuggingFaceBackend:
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL,
        model_path=str(model_path),
        model_revision=MODEL_REVISION,
        tokenizer_id=str(model_path),
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        layer=LAYER,
        layer_path="model.model.layers",
        prompt_mode="chat",
        max_new_tokens=1,
        do_sample=False,
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
        tokenizer_identifier=str(model_path),
        model_revision=MODEL_REVISION,
    )


def verify_single_forward_hook_mechanics() -> dict[str, Any]:
    """Verify that a pre-hook choice can affect the same module invocation."""

    import torch

    module = torch.nn.Identity()
    state: dict[str, Any] = {"selected": None, "pre_calls": 0, "post_calls": 0}
    delta = torch.tensor([[[0.25, -0.5]]], dtype=torch.float32)

    def choose(_module: Any, inputs: Any) -> None:
        state["pre_calls"] += 1
        state["selected"] = delta.to(inputs[0])

    def intervene(_module: Any, _inputs: Any, output: Any) -> Any:
        state["post_calls"] += 1
        updated = output.clone()
        updated[:, -1:, :] += state["selected"]
        return updated

    pre = module.register_forward_pre_hook(choose)
    post = module.register_forward_hook(intervene)
    try:
        source = torch.zeros((1, 3, 2), dtype=torch.float32)
        observed = module(source)
    finally:
        post.remove()
        pre.remove()
    expected = source.clone()
    expected[:, -1:, :] += delta
    passed = (
        state["pre_calls"] == 1
        and state["post_calls"] == 1
        and torch.equal(observed, expected)
        and torch.equal(source, torch.zeros_like(source))
    )
    if not passed:
        raise RuntimeError("Q3_ROUTE_A_SINGLE_FORWARD_DEPLOYMENT_INFEASIBLE")
    return {
        "passed": True,
        "pre_hook_selection_precedes_same_call_output_hook": True,
        "only_current_final_position_changed": True,
        "source_tensor_not_retroactively_modified": True,
    }


def verify_precheck(prompt_manifest: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    precheck = read_json(PRECHECK)
    if precheck.get("status") != "Q3_ROUTE_A_PROMPT_REPRESENTATION_PRECHECK_FROZEN":
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: precheck status")
    if sha256_file(Path(__file__).resolve()) != precheck["implementation"]["capture_runner_sha256"]:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: runner hash")
    if sha256_file(prompt_manifest) != precheck["capture"]["private_prompt_manifest_sha256"]:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: prompt manifest hash")
    prompts = read_json(prompt_manifest)
    forbidden = {"reference_answer", "correct", "outcome", "generated_text", "raw_output"}
    if forbidden.intersection(prompts):
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: forbidden top-level field")
    if len(prompts.get("items", [])) != 300:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: prompt count")
    if list(prompts.get("item_ids", [])) != precheck["capture"]["item_ids"]:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: item order")
    for row in prompts["items"]:
        if forbidden.intersection(row):
            raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: forbidden item field")
        if sha256_bytes(str(row["prompt"]).encode()) != row["prompt_sha256"]:
            raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: prompt bytes")
    return precheck, prompts


def capture_one(backend: HuggingFaceBackend, row: dict[str, Any]) -> dict[str, Any]:
    import torch

    item = BenchmarkItem(
        id=str(row["item_id"]),
        prompt=str(row["prompt"]),
        target="LABEL_FREE_NOT_LOADED",
        metadata={"source_prompt_hash": row["prompt_sha256"]},
    )
    encoded, _rendered, rendered_hash = backend._encode_item(item)  # noqa: SLF001
    input_ids = encoded["input_ids"]
    attention = encoded.get("attention_mask")
    final_index = (
        int(attention[0].sum().item() - 1) if attention is not None else input_ids.shape[1] - 1
    )
    layer26: list[np.ndarray] = []
    layer27_input: list[np.ndarray] = []

    def output_hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        layer26.append(hidden[0, final_index].detach().float().cpu().numpy())

    def input_hook(_module: Any, inputs: Any) -> None:
        hidden = inputs[0]
        layer27_input.append(hidden[0, final_index].detach().float().cpu().numpy())

    h26 = backend.layer_module(26).register_forward_hook(output_hook)
    h27 = backend.layer_module(27).register_forward_pre_hook(input_hook)
    try:
        with torch.inference_mode():
            backend.model(**encoded, use_cache=False, return_dict=True)
    finally:
        h27.remove()
        h26.remove()
    if len(layer26) != 1 or len(layer27_input) != 1:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: hook count")
    a = layer26[0]
    b = layer27_input[0]
    if a.shape != (EXPECTED_WIDTH,) or b.shape != (EXPECTED_WIDTH,):
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: width")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: nonfinite")
    return {
        "item_id": row["item_id"],
        "source_prompt_sha256": row["prompt_sha256"],
        "rendered_prompt_sha256": rendered_hash,
        "input_token_count": int(input_ids.shape[1]),
        "final_nonpadding_index": final_index,
        "layer26_output": a,
        "layer27_input": b,
        "equivalence_max_abs": float(np.max(np.abs(a - b))),
    }


def run_capture(model_path: Path, prompt_manifest: Path, output_dir: Path) -> dict[str, Any]:
    precheck, prompts = verify_precheck(prompt_manifest)
    environment = verify_environment(model_path)
    single_forward = verify_single_forward_hook_mechanics()
    subset = set(precheck["capture"]["repeat_subset_item_ids"])
    by_id = {row["item_id"]: row for row in prompts["items"]}
    started = time.monotonic()
    backend = build_backend(model_path)
    forensic_first = [
        capture_one(backend, by_id[item_id])
        for item_id in precheck["capture"]["repeat_subset_item_ids"]
    ]
    forensic_second = [
        capture_one(backend, by_id[item_id])
        for item_id in precheck["capture"]["repeat_subset_item_ids"]
    ]
    repeat_diffs = [
        float(np.max(np.abs(a["layer27_input"] - b["layer27_input"])))
        for a, b in zip(forensic_first, forensic_second, strict=True)
    ]
    full = [capture_one(backend, row) for row in prompts["items"]]
    matrix = np.stack([row["layer27_input"] for row in full]).astype(np.float32)
    if matrix.shape != (300, EXPECTED_WIDTH):
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: matrix shape")
    if set(row["item_id"] for row in forensic_first) != subset:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: repeat subset")
    if max(repeat_diffs, default=0.0) > precheck["capture"]["repeat_max_abs_tolerance"]:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: repeatability")
    equivalence = [row["equivalence_max_abs"] for row in forensic_first + forensic_second + full]
    if max(equivalence, default=0.0) > precheck["capture"]["site_equivalence_max_abs_tolerance"]:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: site equivalence")

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "Q3_PROMPT_REPRESENTATIONS_LAYER27_INPUT_FLOAT32.npy"
    np.save(matrix_path, matrix, allow_pickle=False)
    rows_path = output_dir / "Q3_PROMPT_REPRESENTATION_ROWS.json"
    rows_payload = {
        "item_ids": [row["item_id"] for row in full],
        "source_prompt_sha256": [row["source_prompt_sha256"] for row in full],
        "rendered_prompt_sha256": [row["rendered_prompt_sha256"] for row in full],
        "input_token_count": [row["input_token_count"] for row in full],
        "final_nonpadding_index": [row["final_nonpadding_index"] for row in full],
    }
    write_json(rows_path, rows_payload)
    result = {
        "schema_version": "q3-route-a-prompt-representation-capture-v1",
        "status": "Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_COMPLETE",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "representation_site": "layer_27_block_input_final_nonpadding_prompt_token",
        "equivalent_site_checked": "layer_26_block_output_final_nonpadding_prompt_token",
        "no_steering": True,
        "semantic_generation": 0,
        "candidate_answers": 0,
        "reference_or_correctness_loaded": False,
        "prompt_only_forward_count": len(forensic_first) + len(forensic_second) + len(full),
        "development_family_count": len(full),
        "repeat_subset_count": len(forensic_first),
        "repeat_max_abs_difference": max(repeat_diffs, default=0.0),
        "site_equivalence_max_abs_difference": max(equivalence, default=0.0),
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "matrix_sha256": sha256_file(matrix_path),
        "row_metadata_sha256": sha256_file(rows_path),
        "private_prompt_manifest_sha256": sha256_file(prompt_manifest),
        "environment": environment,
        "single_forward_hook_mechanics": single_forward,
        "elapsed_seconds": time.monotonic() - started,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
    }
    result_path = output_dir / "Q3_PROMPT_REPRESENTATION_CAPTURE_RESULT.json"
    write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_capture(args.model_path, args.prompt_manifest, args.output_dir)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "prompt_only_forward_count", "matrix_sha256")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
