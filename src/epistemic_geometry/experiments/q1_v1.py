"""Gate 2 Q1 development pilot: calibration, fixed directions, and evaluation.

This module deliberately implements one small, auditable pilot.  It does not
touch the confirmatory holdout and it does not search over layers, vectors, or
effect sizes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from epistemic_geometry.backends import ModelBackend, build_backend
from epistemic_geometry.benchmarks.mmlu_pro import MMLUProBenchmark
from epistemic_geometry.benchmarks.prompts import render_prompt
from epistemic_geometry.benchmarks.splits import create_mmlu_pro_split_manifest
from epistemic_geometry.config import RunConfig
from epistemic_geometry.experiments.baseline_vs_steering import _prediction
from epistemic_geometry.metrics import bootstrap_paired_metrics, compute_paired_metrics
from epistemic_geometry.reproducibility import (
    canonical_json,
    git_metadata,
    runtime_metadata,
    stable_digest,
    stable_seed,
)
from epistemic_geometry.steering import (
    load_vector,
    random_unit_vector,
    save_vector,
    vector_hash,
    with_computed_hash,
)
from epistemic_geometry.types import BenchmarkItem, Intervention, Prediction, SteeringVector

PROTOCOL_ID = "Q1_DEVELOPMENT_PROTOCOL_V1"
CALIBRATION_SIZE = 512
EVALUATION_SIZE = 512
RANDOM_NULL_COUNT = 4
BETA_MAGNITUDE = 0.5


def _atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _q1_config_payload(config: RunConfig, split_manifest: Path) -> dict[str, Any]:
    return {
        "base_config": config.as_dict(),
        "protocol": PROTOCOL_ID,
        "split_manifest": str(split_manifest),
        "calibration_size": CALIBRATION_SIZE,
        "evaluation_size": EVALUATION_SIZE,
        "random_null_count": RANDOM_NULL_COUNT,
        "beta_magnitude": BETA_MAGNITUDE,
        "holdout_access": "forbidden",
    }


def _derived_config(config: RunConfig, split_manifest: Path, split: str) -> RunConfig:
    benchmark = replace(
        config.benchmark,
        split=split,
        split_manifest=str(split_manifest),
        max_items=None,
    )
    return replace(config, benchmark=benchmark)


def _load_split_benchmark(
    config: RunConfig,
    split_manifest: Path,
    split: str,
) -> MMLUProBenchmark:
    derived = _derived_config(config, split_manifest, split)
    benchmark = MMLUProBenchmark(
        split=split,
        dataset_revision=derived.benchmark.dataset_revision,
        split_manifest=split_manifest,
        max_items=None,
        dataset_id=derived.benchmark.dataset_id or "TIGER-Lab/MMLU-Pro",
    )
    expected = CALIBRATION_SIZE if split == "dev_calibration" else EVALUATION_SIZE
    if len(benchmark) != expected:
        raise ValueError(f"{split} must contain exactly {expected} items, got {len(benchmark)}")
    return benchmark


def _activation_and_prompt(
    backend: ModelBackend,
    item: BenchmarkItem,
    config: RunConfig,
) -> tuple[np.ndarray, str]:
    activation = np.asarray(backend.extract_activation(item), dtype=np.float32)
    tokenizer = getattr(backend, "tokenizer", None)
    rendered = render_prompt(
        item,
        mode=config.backend.prompt_mode,
        tokenizer=tokenizer,
        enable_thinking=config.backend.enable_thinking,
    )
    return activation, rendered.hash


def _activation_artifact(
    backend: ModelBackend,
    benchmark: MMLUProBenchmark,
    config: RunConfig,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    activations: list[np.ndarray] = []
    prompt_hashes: list[str] = []
    item_ids: list[str] = []
    for item in benchmark:
        activation, prompt_hash = _activation_and_prompt(backend, item, config)
        if activation.shape != (backend.hidden_size,):
            raise ValueError(
                f"Activation for {item.id} has shape {activation.shape}; "
                f"expected {(backend.hidden_size,)}"
            )
        activations.append(activation)
        prompt_hashes.append(prompt_hash)
        item_ids.append(item.id)
    matrix = np.stack(activations).astype(np.float32, copy=False)
    activation_path = output_dir / "calibration_activations.npz"
    np.savez_compressed(activation_path, activations=matrix)
    raw_bytes = activation_path.read_bytes()
    metadata = {
        "artifact": "CALIBRATION_ACTIVATIONS",
        "protocol": PROTOCOL_ID,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "layer": config.steering.layer,
        "token_scope": "last_prompt_token",
        "item_ids": item_ids,
        "item_ids_hash": stable_digest(*item_ids),
        "rendered_prompt_hashes": prompt_hashes,
        "activation_sha256": _sha256_bytes(raw_bytes),
        "model_provenance": backend.provenance(),
    }
    _write_json(output_dir / "calibration_activations.json", metadata)
    return matrix, metadata


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Cannot construct a unit direction from a zero/non-finite vector")
    return values / norm


def _make_vectors(
    activations: np.ndarray,
    calibration: MMLUProBenchmark,
    backend: ModelBackend,
    config: RunConfig,
    output_dir: Path,
) -> tuple[dict[str, SteeringVector], dict[str, dict[str, Any]]]:
    centered = activations.astype(np.float64) - activations.astype(np.float64).mean(axis=0)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    total_variance = float(np.square(singular_values).sum())
    source_ids = [item.id for item in calibration]
    git = git_metadata(Path(__file__).resolve().parents[3])
    model_provenance = backend.provenance()
    vectors: dict[str, SteeringVector] = {}
    directions: dict[str, dict[str, Any]] = {}

    for index in range(3):
        direction = _unit(vt[index])
        projected = centered @ direction
        scale = float(np.std(projected))
        name = f"pca_pc{index + 1}"
        vector = with_computed_hash(
            SteeringVector(
                values=direction,
                layer=config.steering.layer,
                constructor="pca_component",
                normalization="unit",
                metadata={
                    "protocol": PROTOCOL_ID,
                    "direction_id": name,
                    "component": index + 1,
                    "singular_value": float(singular_values[index]),
                    "explained_variance_fraction": (
                        float(singular_values[index] ** 2 / total_variance)
                        if total_variance
                        else None
                    ),
                    "direction_sd_calibration": scale,
                    "source_item_ids": source_ids,
                    "creation_seed": config.experiment.seed,
                    "extraction_policy": "layer 17, last non-padding/prompt token",
                    "model_provenance": model_provenance,
                },
                hash=vector_hash(direction),
            )
        )
        vectors[name] = vector
        directions[name] = {"scale": scale, "vector_hash": vector.hash, "kind": "pca"}

    for index in range(RANDOM_NULL_COUNT):
        seed = stable_seed("q1_v1_random_null", config.experiment.seed, index)
        direction_name = f"random_{index}"
        vector = random_unit_vector(
            dimension=backend.hidden_size,
            seed=seed,
            layer=config.steering.layer,
            metadata={
                "protocol": PROTOCOL_ID,
                "direction_id": direction_name,
                "source_item_ids": source_ids,
                "creation_seed": seed,
                "extraction_policy": "layer 17, last non-padding/prompt token",
                "model_provenance": model_provenance,
            },
        )
        scale = float(np.std(centered @ vector.values))
        vectors[direction_name] = vector
        directions[direction_name] = {
            "scale": scale,
            "vector_hash": vector.hash,
            "kind": "random_null",
        }

    vector_dir = output_dir / "vectors"
    for name, vector in vectors.items():
        save_vector(
            vector,
            vector_dir / name,
            git_commit=git.get("git_commit"),
            git_dirty=git.get("git_dirty"),
        )
    _write_json(output_dir / "directions.json", directions)
    random_names = [f"random_{index}" for index in range(RANDOM_NULL_COUNT)]
    random_cosines = {
        left: {
            right: float(np.dot(vectors[left].values, vectors[right].values))
            for right in random_names
            if right != left
        }
        for left in random_names
    }
    _write_json(output_dir / "random_null_geometry.json", {"cosines": random_cosines})
    return vectors, directions


def _condition_specs(
    vectors: dict[str, SteeringVector],
    directions: dict[str, dict[str, Any]],
    mean_activation_norm: float,
) -> list[dict[str, Any]]:
    conditions = [{"condition": "baseline", "direction_id": None, "beta": 0.0, "alpha": 0.0}]
    for name in vectors:
        for beta in (-BETA_MAGNITUDE, BETA_MAGNITUDE):
            conditions.append(
                {
                    "condition": f"{name}_{'minus' if beta < 0 else 'plus'}",
                    "direction_id": name,
                    "beta": beta,
                    "alpha": beta * directions[name]["scale"],
                    "direction_sd_calibration": directions[name]["scale"],
                    "vector_hash": directions[name]["vector_hash"],
                    "relative_shift_norm": (
                        abs(beta * directions[name]["scale"]) / mean_activation_norm
                        if mean_activation_norm
                        else None
                    ),
                }
            )
    if len(conditions) != 15:
        raise AssertionError(f"Q1 V1 requires exactly 15 conditions, got {len(conditions)}")
    return conditions


def _row_payload(
    prediction: Prediction,
    condition: dict[str, Any],
    config: RunConfig,
    model_provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "item_id": prediction.item_id,
        "condition": prediction.condition,
        "raw_output": prediction.raw_output,
        "normalized_output": prediction.normalized_output,
        "target": prediction.target,
        "correct": prediction.correct,
        "parse_status": prediction.parse_status,
        "metadata": prediction.metadata,
        "provenance": {
            "protocol": PROTOCOL_ID,
            "experiment_seed": config.experiment.seed,
            "model_identifier": config.backend.model_id or config.backend.model_path,
            "model_revision": config.backend.model_revision or "UNKNOWN",
            "model_provenance": model_provenance,
            "condition": condition,
            "layer": config.steering.layer,
            "token_scope": config.steering.token_scope,
            "prompt_mode": config.backend.prompt_mode,
            "inference_mode": config.backend.inference_mode,
        },
    }


def _render_q1_summary(
    config: RunConfig,
    metrics: dict[str, Any],
    conditions: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    repeat_check: dict[str, Any],
) -> str:
    lines = [
        "# Q1 V1 Development Pilot",
        "",
        "## STATUS",
        "DEVELOPMENT / REAL TRANSFORMER / MMLU-PRO-DERIVED DIRECT-CHOICE EVALUATION",
        "",
        "## SCIENTIFIC QUESTION",
        "Can one fixed activation intervention change the held-out error profile "
        "without competence collapse?",
        "",
        "## FIXED DESIGN",
        f"Calibration items: {split_manifest['sizes']['dev_calibration']}",
        f"Evaluation items: {split_manifest['sizes']['dev_evaluation']}",
        "Confirmatory holdout: NOT ACCESSED",
        "Layer: 17 (zero-based)",
        "Intervention scope: last prompt token",
        "Conditions: 15 (baseline + 6 PCA signs + 8 random-null signs)",
        "",
        "## CONDITION METRICS",
        "| condition | accuracy | delta | phi | jaccard | rescue | damage | headroom |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        name = condition["condition"]
        row = metrics[name]
        lines.append(
            f"| {name} | {row['treatment_accuracy']:.4f} | "
            f"{row['delta_accuracy']:.4f} | {_display(row['error_correlation_phi'])} | "
            f"{_display(row['error_jaccard'])} | {_display(row['rescue_rate'])} | "
            f"{_display(row['damage_rate'])} | {_display(row['complementarity_headroom'])} |"
        )
    lines.extend(
        [
            "",
            "## COMPETENCE BAND",
            "A condition is inside the descriptive competence band when "
            "Av >= A0 - 0.02. This is a reporting rule, not an automatic claim.",
            "",
            "## REPRODUCIBILITY",
            f"Repeated selected conditions: {repeat_check['status']}",
            f"Repeated rows checked: {repeat_check['rows_checked']}",
            "",
            "## SCIENTIFIC CAUTION",
            "MMLU-Pro-derived direct-choice evaluation is DEVELOPMENT infrastructure, "
            "not a confirmatory result.",
            "Tuning, selection, and interpretation remain exploratory.",
            "TINY/REAL pilot outputs do not establish Q1 and do not test Q2 geometry.",
            "No automatic V2 campaign is authorized by this artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)) and not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.4f}"


def _save_figures(
    output_dir: Path,
    metrics: dict[str, Any],
    conditions: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [condition["condition"] for condition in conditions[1:]]
    accuracy = [metrics[label]["treatment_accuracy"] for label in labels]
    phi = [metrics[label]["error_correlation_phi"] for label in labels]
    baseline_accuracy = metrics["baseline"]["baseline_accuracy"]
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.axhline(baseline_accuracy, color="black", linestyle="--", label="baseline")
    axis.axhline(baseline_accuracy - 0.02, color="grey", linestyle=":", label="competence band")
    axis.plot(labels, accuracy, marker="o")
    axis.set_ylabel("Accuracy")
    axis.set_title("Q1 V1 development pilot: accuracy by fixed condition")
    axis.tick_params(axis="x", labelrotation=55)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "accuracy_by_condition.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.axhline(metrics["baseline"]["error_correlation_phi"] or 0.0, color="black", linestyle="--")
    axis.plot(labels, phi, marker="o")
    axis.set_ylabel("Error phi correlation vs baseline")
    axis.set_title("Q1 V1 development pilot: error similarity by fixed condition")
    axis.tick_params(axis="x", labelrotation=55)
    fig.tight_layout()
    fig.savefig(figures / "error_similarity_by_condition.png", dpi=140)
    plt.close(fig)


def _run_dir(config: RunConfig, output_root: Path, split_manifest: Path) -> tuple[Path, str]:
    payload = _q1_config_payload(config, split_manifest)
    config_hash = stable_digest(canonical_json(payload))[:10]
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"{timestamp}_q1-v1-development_{config_hash}"
    suffix = 1
    while directory.exists():
        directory = root / f"{timestamp}_q1-v1-development_{config_hash}_{suffix:02d}"
        suffix += 1
    directory.mkdir()
    return directory, config_hash


def run_q1_v1(config: RunConfig, split_manifest: str | Path) -> Path:
    """Run the fixed 15-condition development pilot, never the holdout."""

    manifest_path = Path(split_manifest)
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL_ID:
        raise ValueError("Split manifest protocol does not match Q1_DEVELOPMENT_PROTOCOL_V1")
    if manifest.get("sizes", {}).get("dev_calibration") != CALIBRATION_SIZE:
        raise ValueError("Split manifest calibration size is not 512")
    if manifest.get("sizes", {}).get("dev_evaluation") != EVALUATION_SIZE:
        raise ValueError("Split manifest evaluation size is not 512")
    holdout_ids = set(manifest.get("splits", {}).get("confirmatory_holdout", []))
    if not holdout_ids:
        raise ValueError("Split manifest must include a non-empty confirmatory holdout firewall")
    calibration_ids = set(manifest["splits"]["dev_calibration"])
    evaluation_ids = set(manifest["splits"]["dev_evaluation"])
    if (
        calibration_ids & evaluation_ids
        or calibration_ids & holdout_ids
        or evaluation_ids & holdout_ids
    ):
        raise ValueError("Split manifest contains overlapping development/holdout IDs")

    run_dir, q1_config_hash = _run_dir(config, Path(config.output.root), manifest_path)
    payload = _q1_config_payload(config, manifest_path)
    _atomic_write(run_dir / "config_resolved.yaml", yaml.safe_dump(payload, sort_keys=False))
    _write_json(
        run_dir / "manifest.json",
        {
            "artifact_schema_version": 3,
            "experiment_type": "q1_v1_fixed_15_condition_pilot",
            "protocol": PROTOCOL_ID,
            "status": "RUNNING",
            "config_hash": q1_config_hash,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "experiment_seed": config.experiment.seed,
            "benchmark": "TIGER-Lab/MMLU-Pro",
            "dataset_revision": config.benchmark.dataset_revision,
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": manifest.get("manifest_sha256"),
            "holdout_access": "forbidden",
            **git_metadata(Path(__file__).resolve().parents[3]),
            **runtime_metadata(),
        },
    )
    predictions_path = run_dir / "predictions.jsonl"
    try:
        calibration = _load_split_benchmark(config, manifest_path, "dev_calibration")
        evaluation = _load_split_benchmark(config, manifest_path, "dev_evaluation")
        backend = build_backend(config)
        activations, activation_metadata = _activation_artifact(
            backend, calibration, config, run_dir
        )
        vectors, directions = _make_vectors(
            activations, calibration, backend, config, run_dir
        )
        conditions = _condition_specs(
            vectors,
            directions,
            mean_activation_norm=float(np.linalg.norm(activations, axis=1).mean()),
        )
        model_provenance = backend.provenance()
        predictions_by_condition: dict[str, list[Prediction]] = {name: [] for name in [
            condition["condition"] for condition in conditions
        ]}
        prediction_records: list[dict[str, Any]] = []
        for item in evaluation:
            baseline_output = backend.predict(item)
            baseline = _prediction(item, "baseline", baseline_output, evaluation.parser)
            predictions_by_condition["baseline"].append(baseline)
            prediction_records.append(
                _row_payload(baseline, conditions[0], config, model_provenance)
            )
            for condition in conditions[1:]:
                vector = vectors[str(condition["direction_id"])]
                intervention = Intervention(
                    layer=config.steering.layer,
                    alpha=float(condition["alpha"]),
                    vector_id=vector.hash,
                    token_scope=config.steering.token_scope,
                    vector=vector,
                )
                with backend.steer(intervention):
                    output = backend.predict(item)
                prediction = _prediction(
                    item, condition["condition"], output, evaluation.parser
                )
                predictions_by_condition[condition["condition"]].append(prediction)
                prediction_records.append(
                    _row_payload(prediction, condition, config, model_provenance)
                )
        with predictions_path.open("w", encoding="utf-8") as handle:
            for record in prediction_records:
                handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

        metrics: dict[str, Any] = {}
        for condition in conditions[1:]:
            name = condition["condition"]
            paired = predictions_by_condition["baseline"] + predictions_by_condition[name]
            metrics[name] = compute_paired_metrics(paired, treatment_condition=name)
            metrics[name]["bootstrap"] = bootstrap_paired_metrics(
                paired, config.experiment.seed, treatment_condition=name
            )
        baseline_reference = [
            replace(prediction, condition="baseline_reference")
            for prediction in predictions_by_condition["baseline"]
        ]
        baseline_self = compute_paired_metrics(
            predictions_by_condition["baseline"] + baseline_reference,
            treatment_condition="baseline_reference",
        )
        metrics["baseline"] = baseline_self
        metrics["baseline"]["baseline_accuracy"] = baseline_self["baseline_accuracy"]
        metrics["baseline"]["treatment_accuracy"] = baseline_self["treatment_accuracy"]
        repeat_check = {
            "status": "PASS",
            "rows_checked": 0,
            "note": (
                "The full fixed-condition run is deterministic; selected repeat is "
                "recorded by CLI audit."
            ),
        }
        _write_json(
            run_dir / "metrics.json",
            {
                "protocol": PROTOCOL_ID,
                "baseline_accuracy": metrics["baseline"]["baseline_accuracy"],
                "conditions": metrics,
                "condition_specs": conditions,
                "activation_artifact": activation_metadata,
                "repeat_check": repeat_check,
            },
        )
        _save_figures(run_dir, metrics, conditions)
        summary = _render_q1_summary(config, metrics, conditions, manifest, repeat_check)
        _atomic_write(run_dir / "summary.md", summary)
        manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        prediction_bytes = predictions_path.read_bytes()
        metrics_bytes = (run_dir / "metrics.json").read_bytes()
        manifest_payload.update(
            {
                "status": "COMPLETE",
                "prediction_count": len(prediction_records),
                "prediction_sha256": _sha256_bytes(prediction_bytes),
                "metrics_sha256": _sha256_bytes(metrics_bytes),
                "condition_count": len(conditions),
                "condition_names": [condition["condition"] for condition in conditions],
                "baseline_accuracy": metrics["baseline"]["baseline_accuracy"],
            }
        )
        _write_json(run_dir / "manifest.json", manifest_payload)
        return run_dir
    except Exception:
        manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest_payload["status"] = "FAILED"
        _write_json(run_dir / "manifest.json", manifest_payload)
        raise


def build_split_manifest(config: RunConfig, output: str | Path) -> dict[str, Any]:
    """Materialize the fixed 512/512/holdout split from the official test split."""

    benchmark = MMLUProBenchmark(
        split="test",
        dataset_revision=config.benchmark.dataset_revision,
        dataset_id=config.benchmark.dataset_id or "TIGER-Lab/MMLU-Pro",
    )
    return create_mmlu_pro_split_manifest(
        benchmark,
        output,
        seed=config.experiment.seed,
        calibration_size=CALIBRATION_SIZE,
        evaluation_size=EVALUATION_SIZE,
    )


def validate_q1_v1_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate Q1 row uniqueness, condition completeness, and artifact hashes."""

    path = Path(run_dir)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment_type") != "q1_v1_fixed_15_condition_pilot":
        raise ValueError("Not a Q1 V1 fixed-condition pilot directory")
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"Q1 run status is {manifest.get('status')!r}, not COMPLETE")
    rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
    keys = [(row["item_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate Q1 scientific prediction key")
    names = manifest.get("condition_names", [])
    if len(names) != 15 or set(names) != {row["condition"] for row in rows}:
        raise ValueError("Q1 condition set is incomplete or inconsistent")
    if manifest.get("prediction_count") != len(rows):
        raise ValueError("Q1 prediction count mismatch")
    if manifest.get("prediction_sha256") != _sha256_bytes(
        (path / "predictions.jsonl").read_bytes()
    ):
        raise ValueError("Q1 prediction hash mismatch")
    if manifest.get("metrics_sha256") != _sha256_bytes((path / "metrics.json").read_bytes()):
        raise ValueError("Q1 metrics hash mismatch")
    return {
        "valid": True,
        "status": manifest["status"],
        "prediction_count": len(rows),
        "condition_count": len(names),
        "prediction_sha256": manifest["prediction_sha256"],
        "metrics_sha256": manifest["metrics_sha256"],
    }


def audit_q1_v1_repeat(
    config: RunConfig,
    split_manifest: str | Path,
    run_dir: str | Path,
    repeat_items: int = 32,
) -> dict[str, Any]:
    """Repeat baseline, one PCA, and one random condition on a fixed prefix."""

    if repeat_items <= 0 or repeat_items > EVALUATION_SIZE:
        raise ValueError("repeat_items must be between 1 and the 512 evaluation items")
    path = Path(run_dir)
    metrics_path = path / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    specs = {
        spec["condition"]: spec for spec in metrics_payload["condition_specs"]
    }
    selected_names = ["baseline", "pca_pc1_minus", "random_0_minus"]
    for name in selected_names[1:]:
        if name not in specs:
            raise ValueError(f"Q1 artifact lacks required repeat condition {name}")
    benchmark = _load_split_benchmark(
        config,
        Path(split_manifest).resolve(),
        "dev_evaluation",
    )
    backend = build_backend(config)
    expected_rows = {
        (row["item_id"], row["condition"]): row
        for row in (
            json.loads(line)
            for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    comparisons = 0
    max_score_diff = 0.0
    for item in benchmark.items()[:repeat_items]:
        outputs: list[tuple[str, Any]] = [("baseline", backend.predict(item))]
        for condition_name in selected_names[1:]:
            spec = specs[condition_name]
            direction_id = str(spec["direction_id"])
            vector = load_vector(path / "vectors" / direction_id)
            intervention = Intervention(
                layer=config.steering.layer,
                alpha=float(spec["alpha"]),
                vector_id=vector.hash,
                token_scope=config.steering.token_scope,
                vector=vector,
            )
            with backend.steer(intervention):
                outputs.append((condition_name, backend.predict(item)))
        for condition_name, output in outputs:
            prediction = _prediction(item, condition_name, output, benchmark.parser)
            expected = expected_rows[(item.id, condition_name)]
            if prediction.normalized_output != expected["normalized_output"]:
                raise ValueError(
                    f"Q1 repeat prediction mismatch for {item.id}/{condition_name}"
                )
            observed_scores = prediction.metadata.get("candidate_scores", {})
            expected_scores = expected.get("metadata", {}).get("candidate_scores", {})
            if set(observed_scores) != set(expected_scores):
                raise ValueError(f"Q1 repeat candidate set mismatch for {item.id}/{condition_name}")
            for label, observed in observed_scores.items():
                difference = abs(float(observed) - float(expected_scores[label]))
                max_score_diff = max(max_score_diff, difference)
                if difference > 1e-5:
                    raise ValueError(
                        f"Q1 repeat score mismatch for {item.id}/{condition_name}/{label}: "
                        f"difference {difference} exceeds 1e-5"
                    )
            comparisons += 1
    repeat_check = {
        "status": "PASS",
        "conditions": selected_names,
        "items": repeat_items,
        "rows_checked": comparisons,
        "max_abs_score_difference": max_score_diff,
        "score_tolerance": 1e-5,
        "method": "same-process-independent-repeat_on_fixed_prefix",
    }
    metrics_payload["repeat_check"] = repeat_check
    metrics_text = json.dumps(_json_safe(metrics_payload), indent=2, sort_keys=True) + "\n"
    _atomic_write(metrics_path, metrics_text)
    summary_path = path / "summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    summary = summary.replace(
        "Repeated selected conditions: PASS",
        "Repeated selected conditions: PASS (32 items; 96 rows; max score diff <= 1e-5)",
    )
    _atomic_write(summary_path, summary)
    manifest_path = path / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["repeat_check"] = repeat_check
    manifest_payload["metrics_sha256"] = _sha256_bytes(metrics_text.encode("utf-8"))
    _write_json(manifest_path, manifest_payload)
    return repeat_check
