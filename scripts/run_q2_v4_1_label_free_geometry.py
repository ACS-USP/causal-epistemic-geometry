#!/usr/bin/env python3
"""Spark-1-only, label-free A1/A2 materialization for Q2 V4.1.

This runner intentionally has no path to the semantic panel, semantic parser,
correctness labels, or free generation.  It captures only the frozen M1
activation covariance and M2 teacher-forced output-response probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from run_gate6_2_first_stage_repair import prompt_tokens  # noqa: E402
from run_gate11_domain_conditioned_control import forward  # noqa: E402
from run_q2_v3 import EXECUTION_TEACHER_TEXT  # noqa: E402

from epistemic_geometry.analysis.q2_geometries import fit_whitening, whitened_geometry  # noqa: E402
from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.external.base import ExternalItem  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.experiments.q2_v4_1 import (  # noqa: E402
    EXPECTED_SAFE_IDS,
    sha256_file,
)
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

REVIEW = ROOT / "review/q2_v4_1_prediction_lock"
OLD_REVIEW = ROOT / "review/q2_v4_1_31_safe_bank_review"
VECTOR_DIR = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_DIRECTIONS"
LAYER = 27
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
EXPECTED_ENVIRONMENT = "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
SHELLS = ("MEDIUM", "STRONG")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


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
        layer=LAYER,
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
    )
    return HuggingFaceBackend(
        config,
        model_identifier=MODEL,
        tokenizer_identifier=model_path,
        model_revision=MODEL_REVISION,
    )


def official_records() -> dict[str, dict[str, Any]]:
    path = ROOT / "review/q2_v3_provenance_reconciliation/OFFICIAL_SOURCE_RECORDS.jsonl"
    records = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            records[str(row["id"])] = {**row, "official_index": index}
    return records


def manifest_items(name: str) -> list[Any]:
    manifest = read_json(REVIEW / name)
    source = official_records()
    values = []
    for frozen in manifest["items"]:
        row = source[str(frozen["item_id"])]
        prompt = str(frozen["prompt"])
        reference = str(frozen.get("reference_answer", row["output"]))
        if hashlib.sha256(prompt.encode()).hexdigest() != frozen["prompt_sha256"]:
            raise RuntimeError("label-free manifest prompt hash mismatch")
        values.append(
            ExternalItem(
                item_id=str(frozen["item_id"]),
                benchmark="CRUXEval",
                subtask="output_prediction",
                prompt=prompt,
                reference_answer=reference,
                evaluator="python_literal",
                source_revision=DATASET_REVISION,
                metadata={
                    "allocation": manifest["allocation"],
                    "official_index": row["official_index"],
                },
            )
        )
    if [item.item_id for item in values] != manifest["item_ids"]:
        raise RuntimeError(f"manifest order mismatch: {name}")
    return values


def require_lock() -> dict[str, Any]:
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    if lock["status"] != "Q2_V4_1_PRESEMANTIC_PROTOCOL_LOCKED":
        raise RuntimeError("Q2_V4.1 presemantic protocol lock is not active")
    if lock["semantic_execution_authorized"] or lock["semantic_outcomes"] != 0:
        raise RuntimeError("semantic firewall state is invalid")
    if lock["controller_count"] != 31 or lock["layer"] != LAYER:
        raise RuntimeError("V4.1 lock dimension/layer mismatch")
    return lock


def load_vectors() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = read_json(OLD_REVIEW / "SAFE_31_IMMUTABLE_MANIFEST.json")
    if (
        sha256_file(OLD_REVIEW / "SAFE_31_IMMUTABLE_MANIFEST.json")
        != "a641d612628c4f9eff2ae9fdf12d3ad17af5a3e921ec726d31c208ee5e030447"
    ):
        raise RuntimeError("immutable safe-bank hash mismatch")
    vectors = {}
    for row in manifest["directions"]:
        # The immutable V4.1 safe-bank manifest intentionally records the
        # vector hash but not a path. Reconstruct only the historical,
        # deterministic candidate path from the immutable candidate_id.
        path = VECTOR_DIR / f"{row['candidate_id']}.npy"
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"controller file hash mismatch: {row['candidate_id']}")
        vectors[row["candidate_id"]] = np.load(path, allow_pickle=False).astype(np.float64)
    if list(vectors) != list(EXPECTED_SAFE_IDS):
        raise RuntimeError("controller order mismatch")
    return vectors, manifest


def current_environment(model_path: Path, source_commit: str) -> dict[str, Any]:
    import torch
    import transformers

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
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": smi,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "model_path": str(model_path),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "source_commit": source_commit,
        "expected_parent_fingerprint": EXPECTED_ENVIRONMENT,
    }
    static_checks = {
        "hostname_spark1": payload["hostname"].split(".", 1)[0] == "spark1",
        "architecture_aarch64": payload["architecture"] == "aarch64",
        "python_3_12_3": payload["python"] == "3.12.3",
        "torch_exact": payload["torch"] == "2.13.0+cu130",
        "transformers_exact": payload["transformers"] == "4.57.6",
        "cuda_13": payload["torch_cuda"] == "13.0",
        "cuda_available": payload["cuda_available"],
        "gpu_gb10": "GB10" in payload["gpu"],
        "one_gpu": torch.cuda.device_count() == 1,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "model_path_exists": model_path.is_dir(),
    }
    payload["static_checks"] = static_checks
    payload["qualified_environment_profile"] = EXPECTED_ENVIRONMENT
    payload["profile_pass"] = bool(all(static_checks.values()))
    write_json(REVIEW / "ENVIRONMENT_PROVENANCE.json", payload)
    if not payload["profile_pass"]:
        raise RuntimeError("Q2_V4_1_ENVIRONMENT_DRIFT")
    return payload


def capture_covariance(backend: Any) -> np.ndarray:
    rows = []
    metadata = []
    for item in manifest_items("A1_COVARIANCE_MANIFEST.json"):
        bench = BenchmarkItem(id=item.item_id, prompt=item.prompt, target="LABEL_FREE")
        prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, bench)
        ids = backend.torch.tensor([prompt_ids], dtype=backend.torch.long, device=backend.device)
        capture: list[np.ndarray] = []

        def make_hook(storage: list[np.ndarray]) -> Any:
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                storage.append(hidden[0, -1].detach().float().cpu().numpy())

            return hook

        handle = backend.layer_module(LAYER).register_forward_hook(make_hook(capture))
        try:
            with backend.torch.inference_mode():
                backend._forward(
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
        if len(capture) != 1:
            raise RuntimeError("A1 layer capture count mismatch")
        rows.append(capture[0])
        metadata.append(
            {
                "item_id": item.item_id,
                "prompt_sha256": prompt_hash,
                "activation_site": "model.model.layers.27.output.final_prompt_token",
            }
        )
    values = np.stack(rows).astype(np.float32)
    np.savez_compressed(REVIEW / "A1_COVARIANCE_ACTIVATIONS.npz", activations=values)
    write_json(
        REVIEW / "A1_COVARIANCE_METADATA.json",
        {"rows": metadata, "shape": list(values.shape), "semantic_outcomes": 0},
    )
    return values


def checkpoints(length: int) -> tuple[int, int, int]:
    if length < 3:
        raise RuntimeError("A2 continuation too short")
    values = (length // 3, (2 * length) // 3, length - 1)
    if len(set(values)) != 3:
        raise RuntimeError("A2 checkpoint collision")
    return values


def fingerprint(
    backend: Any,
    item: Any,
    candidate: str | None,
    alpha: float,
    vectors: dict[str, np.ndarray],
    continuation: list[int],
) -> np.ndarray:
    bench = BenchmarkItem(id=item.item_id, prompt=item.prompt, target="LABEL_FREE")
    prompt_ids, _rendered, _prompt_hash = prompt_tokens(backend, bench)
    if candidate is None:
        context = nullcontext()
    else:
        delta = backend.torch.tensor(
            vectors[candidate] * alpha, dtype=backend.torch.float32, device=backend.device
        ).view(1, 1, -1)
        context = Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: delta},
            target_positions=[len(prompt_ids) - 1],
        )
    snapshots = []
    selected_checkpoints = checkpoints(len(continuation))
    with context, backend.torch.inference_mode():
        output = forward(
            backend, prompt_ids, past=None, total_length=len(prompt_ids), phase="prefill"
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
            if index in selected_checkpoints:
                snapshots.append(output.logits[0, -1].detach().float().cpu().numpy())
    return np.stack(snapshots).astype(np.float32)


def mean_js(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - np.logaddexp.reduce(a, axis=-1, keepdims=True)
    b = b - np.logaddexp.reduce(b, axis=-1, keepdims=True)
    m = np.logaddexp(a, b) - np.log(2.0)
    pa, pb = np.exp(a), np.exp(b)
    return float(np.mean(0.5 * np.sum(pa * (a - m), axis=-1) + 0.5 * np.sum(pb * (b - m), axis=-1)))


def a2_report(
    baseline: np.ndarray,
    fingerprints: dict[str, np.ndarray],
    repeated_baseline: np.ndarray,
    repeated_fingerprints: dict[str, np.ndarray],
    noise_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    names = list(fingerprints)
    radii2 = np.asarray([mean_js(fingerprints[name], baseline) for name in names])
    d2 = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            d2[i, j] = d2[j, i] = mean_js(fingerprints[left], fingerprints[names[j]])
    gram = 0.5 * (radii2[:, None] + radii2[None, :] - d2)
    radii = np.sqrt(np.maximum(radii2, 0.0))
    cosine = gram / np.outer(radii, radii)
    raw_min, raw_max = float(np.min(cosine)), float(np.max(cosine))
    if raw_min < -1.0 - 1e-8 or raw_max > 1.0 + 1e-8:
        raise RuntimeError("A2 cosine bounds failed")
    cosine = np.clip(cosine, -1.0, 1.0)
    np.fill_diagonal(cosine, 1.0)
    repeat_radii = np.asarray(
        [mean_js(repeated_fingerprints[name], repeated_baseline) for name in names]
    )
    repeat_d2 = np.zeros_like(d2)
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            repeat_d2[i, j] = repeat_d2[j, i] = mean_js(
                repeated_fingerprints[left], repeated_fingerprints[names[j]]
            )
    repeat_gram = 0.5 * (repeat_radii[:, None] + repeat_radii[None, :] - repeat_d2)
    repeat_cosine = repeat_gram / np.outer(
        np.sqrt(np.maximum(repeat_radii, 0.0)), np.sqrt(np.maximum(repeat_radii, 0.0))
    )
    upper = np.triu_indices(len(names), 1)
    rank = np.corrcoef(
        np.argsort(np.argsort(1.0 - cosine[upper])),
        np.argsort(np.argsort(1.0 - np.clip(repeat_cosine, -1.0, 1.0)[upper])),
    )[0, 1]
    direct_sum_error = float(np.max(np.abs(d2 - (radii2[:, None] + radii2[None, :] - 2.0 * gram))))
    checks = {
        "radius_floor": bool(np.all(radii2 > noise_floor)),
        "symmetry": float(np.max(np.abs(d2 - d2.T))) <= 1e-12,
        "diagonal": float(np.max(np.abs(np.diag(d2)))) <= 1e-12,
        "baseline_identity": float(
            np.max(
                np.abs(
                    radii2 - np.asarray([mean_js(fingerprints[name], baseline) for name in names])
                )
            )
        )
        <= 1e-10,
        "gram_psd": float(np.min(np.linalg.eigvalsh(gram))) >= -1e-8,
        "cosine_bounds": raw_min >= -1.0 - 1e-8 and raw_max <= 1.0 + 1e-8,
        "repeat_radius": float(
            np.max(np.abs(radii2 - repeat_radii) / np.maximum(np.abs(radii2), 1e-12))
        )
        <= 1e-6,
        "repeat_distance": float(np.max(np.abs(d2 - repeat_d2) / np.maximum(np.abs(d2), 1e-12)))
        <= 1e-6,
        "repeat_angular_rank": float(rank) >= 0.999,
        "direct_sum_hilbert_identity": direct_sum_error <= 1e-12,
    }
    report = {
        "radii_squared": radii2.tolist(),
        "gram_min_eigenvalue": float(np.min(np.linalg.eigvalsh(gram))),
        "cosine_range": [raw_min, raw_max],
        "noise_floor_squared": noise_floor,
        "repeat_radius_relative_error_max": float(
            np.max(np.abs(radii2 - repeat_radii) / np.maximum(np.abs(radii2), 1e-12))
        ),
        "repeat_distance_relative_error_max": float(
            np.max(np.abs(d2 - repeat_d2) / np.maximum(np.abs(d2), 1e-12))
        ),
        "repeat_angular_rank_correlation": float(rank),
        "direct_sum_hilbert_identity_max_error": direct_sum_error,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    return 1.0 - cosine, {
        "report": report,
        "distance_squared": d2,
        "radii_squared": radii2,
        "gram": gram,
    }


def run(model_path: str) -> None:
    lock = require_lock()
    REVIEW.mkdir(parents=True, exist_ok=True)
    vectors, bank = load_vectors()
    source_commit = lock["source_commit"]
    environment = current_environment(Path(model_path), source_commit)
    backend = build_backend(model_path)
    covariance = capture_covariance(backend)
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
    names = list(EXPECTED_SAFE_IDS)
    direction_rows = np.stack([vectors[name] for name in names])
    a0 = (
        1.0
        - np.asarray([bank["directions"][index]["coefficients"] for index in range(len(names))])
        @ np.asarray([bank["directions"][index]["coefficients"] for index in range(len(names))]).T
    )
    np.fill_diagonal(a0, 0.0)
    a1 = np.asarray(whitened_geometry(direction_rows, fit)["cosine_distance"], dtype=np.float64)
    a1_checks = {
        "activation_shape_64_by_4096": covariance.shape == (64, 4096),
        "activations_finite": bool(np.isfinite(covariance).all()),
        "regularization_positive": fit.regularization_value > 0.0,
        "fit_condition_finite": bool(
            np.isfinite(fit.condition_number) and fit.condition_number <= 1e6
        ),
        "effective_rank_at_least_2": fit.effective_rank >= 2.0,
        "matrix_finite": bool(np.isfinite(a1).all()),
        "matrix_symmetry": float(np.max(np.abs(a1 - a1.T))) <= 1e-10,
        "matrix_diagonal": float(np.max(np.abs(np.diag(a1)))) <= 1e-10,
        "cosine_distance_range": float(np.min(a1)) >= -1e-10 and float(np.max(a1)) <= 2.0 + 1e-10,
    }
    write_json(
        REVIEW / "A1_INSTRUMENT_QUALIFICATION.json",
        {
            "activation_archive_sha256": sha256_file(REVIEW / "A1_COVARIANCE_ACTIVATIONS.npz"),
            "fit_sha256": sha256_file(REVIEW / "A1_COVARIANCE_FIT.npz"),
            "fit_hash": fit.fit_hash,
            "lambda": 0.10,
            "regularization_value": fit.regularization_value,
            "effective_rank": fit.effective_rank,
            "condition_number": fit.condition_number,
            "checks": a1_checks,
            "classification": "Q2_V4_1_A1_INSTRUMENT_QUALIFIED"
            if all(a1_checks.values())
            else "Q2_V4_1_A1_INSTRUMENT_NOT_QUALIFIED",
        },
    )
    if not all(a1_checks.values()):
        raise RuntimeError("Q2_V4_1_A1_INSTRUMENT_NOT_QUALIFIED")
    continuation = [
        int(value)
        for value in backend.tokenizer.encode(EXECUTION_TEACHER_TEXT, add_special_tokens=False)
    ]
    probes = manifest_items("A2_PROBE_MANIFEST.json")
    deployment = lock["deployment"]
    raw_dir = REVIEW / "A2_FINGERPRINTS"
    repeat_dir = REVIEW / "A2_REPEAT_FINGERPRINTS"
    raw_dir.mkdir(exist_ok=True)
    repeat_dir.mkdir(exist_ok=True)
    baseline_rows: list[np.ndarray] = []
    repeated_baseline_rows: list[np.ndarray] = []
    condition_rows = {f"{name}_{shell}": [] for name in names for shell in SHELLS}
    repeated_condition_rows = {f"{name}_{shell}": [] for name in names for shell in SHELLS}
    for item in probes:
        baseline = fingerprint(backend, item, None, 0.0, vectors, continuation)
        repeated_baseline = fingerprint(backend, item, None, 0.0, vectors, continuation)
        baseline_rows.append(baseline)
        repeated_baseline_rows.append(repeated_baseline)
        arrays = {"BASELINE": baseline}
        repeated_arrays = {"BASELINE": repeated_baseline}
        for name in names:
            for shell in SHELLS:
                condition = f"{name}_{shell}"
                alpha = float(deployment[condition]["alpha"])
                value = fingerprint(backend, item, name, alpha, vectors, continuation)
                repeated_value = fingerprint(backend, item, name, alpha, vectors, continuation)
                condition_rows[condition].append(value)
                repeated_condition_rows[condition].append(repeated_value)
                arrays[condition] = value
                repeated_arrays[condition] = repeated_value
        np.savez_compressed(raw_dir / f"{item.item_id}.npz", **arrays)
        np.savez_compressed(repeat_dir / f"{item.item_id}.npz", **repeated_arrays)
    baseline = np.concatenate(baseline_rows, axis=0).astype(np.float64)
    repeated_baseline = np.concatenate(repeated_baseline_rows, axis=0).astype(np.float64)
    repeat_js = mean_js(baseline, repeated_baseline)
    noise_floor = max(1e-12, 100.0 * repeat_js)
    matrices = {"A0_MEDIUM": a0, "A0_STRONG": a0, "A1_MEDIUM": a1, "A1_STRONG": a1}
    reports = {}
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
        dissimilarity, result = a2_report(
            baseline, fingerprints, repeated_baseline, repeated_fingerprints, noise_floor
        )
        matrices[f"A2_{shell}"] = dissimilarity
        matrices[f"D2_{shell}"] = np.sqrt(np.maximum(result["distance_squared"], 0.0))
        reports[shell] = result["report"]
        if not result["report"]["pass"]:
            raise RuntimeError("Q2_V4_1_A2_INSTRUMENT_NOT_QUALIFIED")
    np.savez_compressed(REVIEW / "PREDICTION_MATRICES.npz", **matrices)
    write_json(
        REVIEW / "A2_INSTRUMENT_QUALIFICATION.json",
        {
            "probe_count": len(probes),
            "checkpoint_count_per_probe": 4,
            "repeat_baseline_mean_JS": repeat_js,
            "noise_floor_squared": noise_floor,
            "shells": reports,
            "classification": "Q2_V4_1_A2_INSTRUMENT_QUALIFIED",
        },
    )
    write_json(
        REVIEW / "PREDICTION_MATRIX_METADATA.json",
        {
            "controller_order": names,
            "matrix_archive_sha256": sha256_file(REVIEW / "PREDICTION_MATRICES.npz"),
            "matrix_hashes": {
                name: array_hash(value.astype(np.float64)) for name, value in matrices.items()
            },
            "semantic_outcomes": 0,
            "correctness_inspected": False,
            "environment_fingerprint_profile": EXPECTED_ENVIRONMENT,
        },
    )
    write_json(
        REVIEW / "LABEL_FREE_GEOMETRY_RUN.json",
        {
            "status": "COMPLETE",
            "A1": "Q2_V4_1_A1_INSTRUMENT_QUALIFIED",
            "A2": "Q2_V4_1_A2_INSTRUMENT_QUALIFIED",
            "semantic_outcomes": 0,
            "correctness_inspected": False,
            "primary_panel_processed": False,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "source_commit": source_commit,
            "elapsed_seconds": None,
            "environment": environment,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    run(args.model_path)
    elapsed = time.monotonic() - started
    value = read_json(REVIEW / "LABEL_FREE_GEOMETRY_RUN.json")
    value["elapsed_seconds"] = elapsed
    write_json(REVIEW / "LABEL_FREE_GEOMETRY_RUN.json", value)
    print(json.dumps({"status": "COMPLETE", "elapsed_seconds": elapsed, "semantic_outcomes": 0}))


if __name__ == "__main__":
    main()
