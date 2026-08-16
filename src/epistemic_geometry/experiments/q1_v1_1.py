"""Controlled Q1 V1.1 follow-up.

V1.1 reuses the frozen V1 vectors and evaluation IDs. It adds only the
pre-registered numerical, equal-norm random, and option-permutation controls.
It never loads or evaluates the confirmatory holdout.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from epistemic_geometry.backends import ModelBackend, build_backend
from epistemic_geometry.benchmarks.mmlu_pro import LABELS, MMLUProBenchmark
from epistemic_geometry.benchmarks.permutations import permute_mmlu_item, permute_mmlu_items
from epistemic_geometry.config import RunConfig
from epistemic_geometry.experiments.baseline_vs_steering import _prediction
from epistemic_geometry.metrics import bootstrap_paired_metrics, compute_paired_metrics
from epistemic_geometry.reproducibility import (
    canonical_json,
    git_metadata,
    runtime_metadata,
    stable_digest,
)
from epistemic_geometry.steering import load_vector
from epistemic_geometry.types import BenchmarkItem, Intervention, Prediction, SteeringVector

PROTOCOL_ID = "Q1_DEVELOPMENT_PROTOCOL_V1_1"
V1_PROTOCOL_ID = "Q1_DEVELOPMENT_PROTOCOL_V1"
EVALUATION_SIZE = 512
PERMUTATION_IDS = ("permutation_0", "permutation_1", "permutation_2", "permutation_3")
ORIGINAL_VECTOR_HASHES = {
    "pca_pc1": "abca43ae3b9621614562798dbfbd8c3ad9932fc9fcb0cfd2c58d28adc48897c5",
    "random_0": "d6ef7d2c8146196330fb14aa2b1e1d6e7d94177b9e2033a6eeda82bc64d00a28",
    "random_1": "8d440a17db54034db10fe52ed1237cd821c763df68aa4f8a9b51181a6956d853",
    "random_2": "1951e428065d639ce4308da3c3ebf41c250678c12141dc1415fcd08c1db7f8ab",
    "random_3": "28847139eca3c31f40253e4496eb85814bd28d3e3b232d77964272b7f255ac6b",
}
V1_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
V1_DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
V1_SPLIT_HASH = "84982e4c72e230ffff78363f085d4d5c53447fd1e248e5e170ed5e8c508d343e"


def _atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or Path.cwd()) / path


def _v11_options(config: RunConfig) -> dict[str, Any]:
    options = dict(config.q1_v1_1)
    if options.get("protocol", PROTOCOL_ID) != PROTOCOL_ID:
        raise ValueError("q1_v1_1.protocol must be Q1_DEVELOPMENT_PROTOCOL_V1_1")
    return options


def _load_v1_reference(config: RunConfig) -> dict[str, Any]:
    options = _v11_options(config)
    if not options.get("v1_run_dir"):
        raise ValueError("q1_v1_1.v1_run_dir is required")
    path = _resolve_path(str(options["v1_run_dir"]))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("V1 reference run is not COMPLETE")
    if manifest.get("protocol") != V1_PROTOCOL_ID:
        raise ValueError("V1 reference protocol mismatch")
    if manifest.get("holdout_access") != "forbidden":
        raise ValueError("V1 reference does not carry the holdout firewall")
    if manifest.get("dataset_revision") != V1_DATASET_REVISION:
        raise ValueError("V1 dataset revision differs from the frozen revision")
    if manifest.get("split_manifest_sha256") != V1_SPLIT_HASH:
        raise ValueError("V1 split hash differs from the frozen split")
    vectors: dict[str, SteeringVector] = {}
    for name, expected_hash in ORIGINAL_VECTOR_HASHES.items():
        vector = load_vector(path / "vectors" / name)
        if vector.hash != expected_hash:
            raise ValueError(f"V1 vector hash mismatch for {name}")
        if vector.layer != 17 or vector.dimension != 4096:
            raise ValueError(f"V1 vector shape/layer mismatch for {name}")
        vectors[name] = vector
    old_rows = {
        (row["item_id"], row["condition"]): row
        for row in (
            json.loads(line)
            for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    required = {"baseline", "pca_pc1_minus", "pca_pc1_plus"}
    if not required.issubset({condition for _item, condition in old_rows}):
        raise ValueError("V1 predictions lack baseline and both PC1 conditions")
    specs = {spec["condition"]: spec for spec in metrics["condition_specs"]}
    alpha_plus = float(specs["pca_pc1_plus"]["alpha"])
    alpha_minus = float(specs["pca_pc1_minus"]["alpha"])
    if not math.isclose(alpha_plus, -alpha_minus, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("V1 PC1 alpha signs are not symmetric")
    activations_path = path / "calibration_activations.npz"
    with np.load(activations_path, allow_pickle=False) as archive:
        activations = np.asarray(archive["activations"], dtype=np.float64)
    return {
        "path": path,
        "manifest": manifest,
        "metrics": metrics,
        "vectors": vectors,
        "old_rows": old_rows,
        "condition_specs": specs,
        "alpha_pc1_plus": alpha_plus,
        "alpha_pc1_minus": alpha_minus,
        "calibration_median_hidden_norm": float(np.median(np.linalg.norm(activations, axis=1))),
    }


def _load_evaluation(
    config: RunConfig, split_manifest: Path
) -> tuple[MMLUProBenchmark, dict[str, Any]]:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != V1_SPLIT_HASH:
        raise ValueError("V1.1 split hash differs from the frozen V1 split")
    if manifest.get("dataset_revision") != V1_DATASET_REVISION:
        raise ValueError("V1.1 dataset revision differs from V1")
    evaluation_ids = set(manifest.get("splits", {}).get("dev_evaluation", []))
    holdout_ids = set(manifest.get("splits", {}).get("confirmatory_holdout", []))
    if len(evaluation_ids) != EVALUATION_SIZE or not evaluation_ids:
        raise ValueError("V1.1 DEV_EVALUATION must contain exactly 512 IDs")
    if evaluation_ids & holdout_ids:
        raise ValueError("V1.1 evaluation IDs overlap CONFIRMATORY_HOLDOUT")
    benchmark = MMLUProBenchmark(
        split="dev_evaluation",
        dataset_revision=V1_DATASET_REVISION,
        split_manifest=split_manifest,
        dataset_id="TIGER-Lab/MMLU-Pro",
    )
    actual_ids = {item.id for item in benchmark}
    if actual_ids != evaluation_ids:
        raise ValueError("Loaded V1.1 item IDs differ from DEV_EVALUATION")
    if actual_ids & holdout_ids:
        raise ValueError("V1.1 loaded an item from CONFIRMATORY_HOLDOUT")
    return benchmark, manifest


def _condition(
    name: str,
    family: str,
    vector_id: str | None,
    vector_hash_value: str | None,
    alpha: float,
    baseline_condition: str,
    direction_sd: float | None = None,
    permutation_id: str | None = None,
    beta: float | None = None,
    vector_norm: float | None = None,
    median_hidden_norm: float | None = None,
) -> dict[str, Any]:
    norm = vector_norm if vector_norm is not None else 0.0
    intervention_norm = abs(alpha) * norm
    return {
        "condition": name,
        "family": family,
        "baseline_condition": baseline_condition,
        "direction_id": vector_id,
        "vector_hash": vector_hash_value,
        "alpha": float(alpha),
        "beta": beta,
        "direction_sd_calibration": direction_sd,
        "intervention_norm": intervention_norm,
        "relative_intervention_norm": (
            intervention_norm / median_hidden_norm
            if median_hidden_norm and intervention_norm
            else 0.0
        ),
        "permutation_id": permutation_id,
        "layer": 17,
        "token_scope": "last_token",
    }


def _frozen_conditions(
    reference: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, SteeringVector]]:
    vectors = reference["vectors"]
    median_norm = reference["calibration_median_hidden_norm"]
    specs: list[dict[str, Any]] = [
        _condition(
            "baseline_fp32",
            "original_fp32",
            None,
            None,
            0.0,
            "baseline_fp32",
            median_hidden_norm=median_norm,
        ),
        _condition(
            "pca_pc1_minus_fp32",
            "original_fp32",
            "pca_pc1",
            vectors["pca_pc1"].hash,
            reference["alpha_pc1_minus"],
            "baseline_fp32",
            reference["condition_specs"]["pca_pc1_minus"]["direction_sd_calibration"],
            beta=-0.5,
            vector_norm=float(np.linalg.norm(vectors["pca_pc1"].values)),
            median_hidden_norm=median_norm,
        ),
        _condition(
            "pca_pc1_plus_fp32",
            "original_fp32",
            "pca_pc1",
            vectors["pca_pc1"].hash,
            reference["alpha_pc1_plus"],
            "baseline_fp32",
            reference["condition_specs"]["pca_pc1_plus"]["direction_sd_calibration"],
            beta=0.5,
            vector_norm=float(np.linalg.norm(vectors["pca_pc1"].values)),
            median_hidden_norm=median_norm,
        ),
    ]
    for index in range(4):
        name = f"random_{index}"
        direction_sd = reference["condition_specs"][f"{name}_plus"]["direction_sd_calibration"]
        for sign, suffix in ((-1.0, "neg"), (1.0, "pos")):
            specs.append(
                _condition(
                    f"{name}_native_scale_{suffix}",
                    "random_native_scale",
                    name,
                    vectors[name].hash,
                    sign * float(direction_sd),
                    "baseline_fp32",
                    direction_sd,
                    vector_norm=float(np.linalg.norm(vectors[name].values)),
                    median_hidden_norm=median_norm,
                )
            )
    alpha_match = abs(reference["alpha_pc1_plus"])
    for index in range(4):
        name = f"random_{index}"
        native_sd = reference["condition_specs"][f"{name}_plus"]["direction_sd_calibration"]
        for sign, suffix in ((-1.0, "neg"), (1.0, "pos")):
            specs.append(
                _condition(
                    f"{name}_normmatched_pc1_{suffix}",
                    "random_pc1_normmatched",
                    name,
                    vectors[name].hash,
                    sign * alpha_match,
                    "baseline_fp32",
                    native_sd,
                    vector_norm=float(np.linalg.norm(vectors[name].values)),
                    median_hidden_norm=median_norm,
                )
            )
    if len(specs) != 19:
        raise AssertionError(f"V1.1 original-order condition count is {len(specs)}, expected 19")
    return specs, vectors


def _permutation_conditions(
    reference: dict[str, Any],
    permutation_id: str,
) -> list[dict[str, Any]]:
    vector = reference["vectors"]["pca_pc1"]
    return [
        _condition(
            f"{permutation_id}_baseline",
            "option_permutation",
            None,
            None,
            0.0,
            f"{permutation_id}_baseline",
            permutation_id=permutation_id,
        ),
        _condition(
            f"{permutation_id}_pc1_minus",
            "option_permutation",
            "pca_pc1",
            vector.hash,
            reference["alpha_pc1_minus"],
            f"{permutation_id}_baseline",
            reference["condition_specs"]["pca_pc1_minus"]["direction_sd_calibration"],
            permutation_id=permutation_id,
            beta=-0.5,
            vector_norm=float(np.linalg.norm(vector.values)),
            median_hidden_norm=reference["calibration_median_hidden_norm"],
        ),
        _condition(
            f"{permutation_id}_pc1_plus",
            "option_permutation",
            "pca_pc1",
            vector.hash,
            reference["alpha_pc1_plus"],
            f"{permutation_id}_baseline",
            reference["condition_specs"]["pca_pc1_plus"]["direction_sd_calibration"],
            permutation_id=permutation_id,
            beta=0.5,
            vector_norm=float(np.linalg.norm(vector.values)),
            median_hidden_norm=reference["calibration_median_hidden_norm"],
        ),
    ]


def _semantic_prediction(prediction: Prediction, item: BenchmarkItem) -> Prediction:
    metadata = dict(prediction.metadata)
    item_metadata = dict(metadata.get("item_metadata", {}))
    semantic_ids = item_metadata.get("semantic_option_ids")
    if not isinstance(semantic_ids, list):
        semantic_ids = list(range(len(item_metadata.get("options", []))))
    try:
        displayed_index = LABELS.index(prediction.normalized_output)
    except ValueError:
        displayed_index = None
    semantic_prediction = (
        semantic_ids[displayed_index]
        if displayed_index is not None and displayed_index < len(semantic_ids)
        else None
    )
    target_index = item_metadata.get("permuted_target_index", item_metadata.get("answer_index"))
    semantic_target = (
        semantic_ids[int(target_index)]
        if target_index is not None and int(target_index) < len(semantic_ids)
        else None
    )
    metadata.update(
        {
            "displayed_prediction_label": prediction.normalized_output,
            "displayed_target_label": item.target,
            "semantic_prediction_original_index": semantic_prediction,
            "semantic_target_original_index": semantic_target,
            "semantic_option_ids": semantic_ids,
        }
    )
    return replace(prediction, metadata=metadata)


def _run_one(
    backend: ModelBackend,
    item: BenchmarkItem,
    spec: dict[str, Any],
    parser: Any,
    vector: SteeringVector | None,
) -> Prediction:
    if vector is None:
        output = backend.predict(item)
    else:
        intervention = Intervention(
            layer=17,
            alpha=float(spec["alpha"]),
            vector_id=vector.hash,
            token_scope="last_token",
            vector=vector,
        )
        with backend.steer(intervention):
            output = backend.predict(item)
    return _semantic_prediction(_prediction(item, spec["condition"], output, parser), item)


def _row_payload(
    prediction: Prediction,
    spec: dict[str, Any],
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
            "confirmatory_accessed": "NO",
            "experiment_seed": config.experiment.seed,
            "model_identifier": config.backend.model_id or config.backend.model_path,
            "model_revision": config.backend.model_revision,
            "model_provenance": model_provenance,
            "condition": spec,
            "layer": 17,
            "token_scope": "last_token",
            "prompt_mode": config.backend.prompt_mode,
            "inference_mode": config.backend.inference_mode,
            "execution_engine": config.backend.execution_mode,
            "candidate_head_mode": config.backend.candidate_head_mode,
            "attention_implementation": config.backend.attention_implementation,
            "torch_compile": config.backend.torch_compile,
            "cuda_graphs": config.backend.cuda_graphs,
            "item_batch_size": config.backend.item_batch_size,
            "condition_chunk_size": config.backend.condition_chunk_size,
            "max_prefill_tokens": config.backend.max_prefill_tokens,
            "padding_side": config.backend.padding_side,
            "serial_shape_reference": config.backend.serial_shape_reference,
        },
    }


def _paired_metrics(
    predictions_by_condition: dict[str, list[Prediction]],
    baseline_name: str,
    treatment_name: str,
    seed: int,
    bootstrap: bool,
) -> dict[str, Any]:
    baseline = [
        replace(item, condition="baseline") for item in predictions_by_condition[baseline_name]
    ]
    treatment = [
        replace(item, condition=treatment_name) for item in predictions_by_condition[treatment_name]
    ]
    paired = baseline + treatment
    metrics = compute_paired_metrics(paired, treatment_condition=treatment_name)
    if bootstrap:
        metrics["bootstrap"] = bootstrap_paired_metrics(
            paired, seed, treatment_condition=treatment_name
        )
    return metrics


def _summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
    }


def _margins_for_condition(
    baseline_rows: list[Prediction],
    treatment_rows: list[Prediction],
) -> dict[str, Any]:
    base_by_id = {row.item_id: row for row in baseline_rows}
    treat_by_id = {row.item_id: row for row in treatment_rows}
    groups: dict[str, list[float]] = {"unchanged": [], "changed": [], "rescues": [], "damages": []}
    margins: list[float] = []
    by_item: list[dict[str, Any]] = []
    for item_id, baseline in base_by_id.items():
        scores = baseline.metadata.get("candidate_scores", {})
        ordered = sorted((float(value) for value in scores.values()), reverse=True)
        if len(ordered) < 2:
            continue
        margin = ordered[0] - ordered[1]
        treatment = treat_by_id[item_id]
        margins.append(margin)
        if baseline.metadata.get("semantic_prediction_original_index") == treatment.metadata.get(
            "semantic_prediction_original_index"
        ):
            groups["unchanged"].append(margin)
            status = "unchanged"
        else:
            groups["changed"].append(margin)
            if not baseline.correct and treatment.correct:
                groups["rescues"].append(margin)
                status = "rescue"
            elif baseline.correct and not treatment.correct:
                groups["damages"].append(margin)
                status = "damage"
            else:
                status = "changed_other"
        by_item.append({"item_id": item_id, "margin": margin, "status": status})
    return {
        "groups": {name: _summary_stats(values) for name, values in groups.items()},
        "items": by_item,
    }


def _old_margin_groups(
    old_rows: dict[tuple[str, str], dict[str, Any]],
    new_rows: list[Prediction],
    old_condition: str,
) -> dict[str, Any]:
    """Summarize whether FP32 prediction changes occur near old score margins."""

    groups: dict[str, list[float]] = {"unchanged": [], "changed": []}
    for prediction in new_rows:
        old = old_rows[(prediction.item_id, old_condition)]
        scores = old.get("metadata", {}).get("candidate_scores", {})
        ordered = sorted((float(value) for value in scores.values()), reverse=True)
        if len(ordered) < 2:
            continue
        margin = ordered[0] - ordered[1]
        status = (
            "changed"
            if prediction.normalized_output != old.get("normalized_output")
            else "unchanged"
        )
        groups[status].append(margin)
    return {name: _summary_stats(values) for name, values in groups.items()}


def _margin_quartiles(margin_items: list[dict[str, Any]]) -> dict[str, Any]:
    margins = np.asarray([float(item["margin"]) for item in margin_items], dtype=float)
    boundaries = np.percentile(margins, [25, 50, 75]).tolist()
    bins: dict[str, dict[str, int]] = {}
    for item in margin_items:
        quartile = int(np.digitize(float(item["margin"]), boundaries, right=True)) + 1
        key = f"Q{quartile}"
        row = bins.setdefault(key, {"n": 0, "changed": 0, "rescued": 0, "damaged": 0})
        row["n"] += 1
        if item["status"] in {"changed_other", "rescue", "damage"}:
            row["changed"] += 1
        if item["status"] == "rescue":
            row["rescued"] += 1
        if item["status"] == "damage":
            row["damaged"] += 1
    for row in bins.values():
        n = row["n"]
        for key in ("changed", "rescued", "damaged"):
            row[f"{key}_rate"] = row[key] / n if n else None
    return {"boundaries": boundaries, "rates": bins}


def _option_bias(
    predictions_by_condition: dict[str, list[Prediction]],
    ordering_conditions: dict[str, dict[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ordering, conditions in ordering_conditions.items():
        result[ordering] = {}
        for role, name in conditions.items():
            rows = predictions_by_condition[name]
            prediction_counts = Counter(row.normalized_output for row in rows)
            target_counts = Counter(row.target for row in rows)
            semantic_prediction_counts = Counter(
                str(row.metadata.get("semantic_prediction_original_index")) for row in rows
            )
            semantic_target_counts = Counter(
                str(row.metadata.get("semantic_target_original_index")) for row in rows
            )
            n = len(rows)
            result[ordering][role] = {
                "condition": name,
                "prediction_letter_distribution": {
                    label: prediction_counts.get(label, 0) / n for label in LABELS[:10]
                },
                "target_letter_distribution": {
                    label: target_counts.get(label, 0) / n for label in LABELS[:10]
                },
                "semantic_prediction_distribution": {
                    index: semantic_prediction_counts.get(str(index), 0) / n for index in range(10)
                },
                "semantic_target_distribution": {
                    index: semantic_target_counts.get(str(index), 0) / n for index in range(10)
                },
            }
    return result


def _category_analysis(
    baseline_rows: list[Prediction],
    treatment_rows: list[Prediction],
) -> dict[str, Any]:
    base = {row.item_id: row for row in baseline_rows}
    treat = {row.item_id: row for row in treatment_rows}
    categories: dict[str, dict[str, int]] = {}
    for item_id, row in base.items():
        category = str(row.metadata.get("item_metadata", {}).get("category", "UNKNOWN"))
        target = categories.setdefault(
            category,
            {"n": 0, "baseline_errors": 0, "treatment_errors": 0, "rescues": 0, "damages": 0},
        )
        target["n"] += 1
        target["baseline_errors"] += int(not row.correct)
        target["treatment_errors"] += int(not treat[item_id].correct)
        target["rescues"] += int(not row.correct and treat[item_id].correct)
        target["damages"] += int(row.correct and not treat[item_id].correct)
    return categories


def _save_figures(
    output_dir: Path,
    metrics: dict[str, Any],
    permutation_metrics: dict[str, Any],
    margin_quartiles: dict[str, Any],
    option_bias: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    original = [
        name
        for name, row in metrics.items()
        if name != "baseline_fp32" and row.get("treatment_condition")
    ]
    colors = ["tab:blue" if "normmatched" not in name else "tab:orange" for name in original]
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.scatter(
        [metrics[name]["delta_accuracy"] for name in original],
        [metrics[name]["complementarity_headroom"] for name in original],
        c=colors,
    )
    for name in original:
        axis.annotate(
            name,
            (metrics[name]["delta_accuracy"], metrics[name]["complementarity_headroom"]),
            fontsize=6,
        )
    axis.set_xlabel("Delta accuracy")
    axis.set_ylabel("Complementarity headroom")
    axis.set_title("Q1 V1.1: accuracy versus complementarity headroom")
    fig.tight_layout()
    fig.savefig(figures / "delta_accuracy_vs_headroom.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.axline((0, 0), slope=1.0, color="black", linestyle="--", label="rescues = damages")
    axis.scatter(
        [metrics[name]["rescue_rate"] for name in original],
        [metrics[name]["damage_rate"] for name in original],
        c=colors,
    )
    for name in original:
        axis.annotate(
            name, (metrics[name]["rescue_rate"], metrics[name]["damage_rate"]), fontsize=6
        )
    axis.set_xlabel("Rescue rate")
    axis.set_ylabel("Damage rate")
    axis.set_title("Q1 V1.1: rescues versus damages")
    fig.tight_layout()
    fig.savefig(figures / "rescues_vs_damages.png", dpi=140)
    plt.close(fig)

    labels = list(permutation_metrics)
    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.36
    axis.bar(
        x - width / 2,
        [permutation_metrics[name]["plus"]["delta_accuracy"] for name in labels],
        width,
        label="PC1+",
    )
    axis.bar(
        x + width / 2,
        [permutation_metrics[name]["minus"]["delta_accuracy"] for name in labels],
        width,
        label="PC1-",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Delta accuracy")
    axis.set_title("Q1 V1.1: option permutation PC1 deltas")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "option_permutation_results.png", dpi=140)
    plt.close(fig)

    orderings = ["original", *labels]
    fig, axis = plt.subplots(figsize=(10, 5))
    for _index, ordering in enumerate(orderings):
        values = option_bias[ordering]["pc1_plus"]["prediction_letter_distribution"]
        axis.plot(
            list(LABELS[:10]), [values[label] for label in LABELS[:10]], marker="o", label=ordering
        )
    axis.set_xlabel("Displayed prediction letter")
    axis.set_ylabel("Proportion")
    axis.set_title("Q1 V1.1: baseline/PC1+ displayed-letter distributions")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "prediction_letter_distributions.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    quartile_labels = sorted(margin_quartiles["rates"])
    axis.plot(
        quartile_labels,
        [margin_quartiles["rates"][key]["changed_rate"] for key in quartile_labels],
        marker="o",
        label="changed",
    )
    axis.plot(
        quartile_labels,
        [margin_quartiles["rates"][key]["rescued_rate"] for key in quartile_labels],
        marker="o",
        label="rescued",
    )
    axis.plot(
        quartile_labels,
        [margin_quartiles["rates"][key]["damaged_rate"] for key in quartile_labels],
        marker="o",
        label="damaged",
    )
    axis.set_xlabel("Baseline score-margin quartile")
    axis.set_ylabel("Rate")
    axis.set_title("Q1 V1.1: PC1+ changes by baseline margin quartile")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "margin_quartile_rates.png", dpi=140)
    plt.close(fig)


def estimate_v1_v1(config: RunConfig) -> dict[str, Any]:
    """Estimate V1.1 workload from the recorded V1 scoring duration."""

    options = _v11_options(config)
    original_conditions = 19
    permutation_conditions = len(options.get("permutation_ids", PERMUTATION_IDS)) * 3
    condition_count = original_conditions + permutation_conditions
    n_items = EVALUATION_SIZE
    candidate_count = len(config.backend.candidate_labels)
    candidate_forwards = condition_count * n_items * candidate_count
    observed_v1_minutes = float(options.get("observed_v1_scoring_minutes", 80.85))
    estimated_minutes = observed_v1_minutes * condition_count / 15.0 * 1.10
    hourly_rate = float(options.get("a40_hourly_usd_assumption", 0.40))
    item_batch_size = max(1, int(config.backend.item_batch_size))
    condition_chunk_size = max(1, int(config.backend.condition_chunk_size))
    ordering_count = 1 + len(options.get("permutation_ids", PERMUTATION_IDS))
    optimized_prefill_batches = math.ceil(n_items / item_batch_size) * ordering_count
    optimized_decode_batches = math.ceil(n_items / item_batch_size) * (
        math.ceil(original_conditions / condition_chunk_size)
        + len(options.get("permutation_ids", PERMUTATION_IDS))
        * math.ceil(3 / condition_chunk_size)
    )
    return {
        "items": n_items,
        "original_order_conditions": original_conditions,
        "permutation_conditions": permutation_conditions,
        "total_conditions": condition_count,
        "item_condition_evaluations": condition_count * n_items,
        "candidate_labels": candidate_count,
        "candidate_forward_passes": candidate_forwards,
        "observed_v1_scoring_minutes": observed_v1_minutes,
        "overhead_factor": 1.10,
        "estimated_minutes": estimated_minutes,
        "a40_hourly_usd_assumption": hourly_rate,
        "estimated_a40_cost_usd": estimated_minutes / 60.0 * hourly_rate,
        "cost_gate_usd": 2.0,
        "cost_gate_pass": estimated_minutes / 60.0 * hourly_rate <= 2.0,
        "execution_engine_requested": config.backend.execution_mode,
        "optimized_plan": {
            "ordering_count": ordering_count,
            "prefill_batches_upper_bound": optimized_prefill_batches,
            "decode_batches_upper_bound": optimized_decode_batches,
            "candidate_forward_passes_avoided_by_single_token_fast_path": candidate_forwards,
            "note": "Engineering estimate only; target-GPU benchmark remains required.",
        },
    }


def _run_dir(config: RunConfig) -> tuple[Path, str]:
    payload = {"config": config.as_dict(), "protocol": PROTOCOL_ID}
    config_hash = stable_digest(canonical_json(payload))[:10]
    root = _resolve_path(config.output.root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{timestamp}_q1-v1-1-development_{config_hash}"
    suffix = 1
    while path.exists():
        path = root / f"{timestamp}_q1-v1-1-development_{config_hash}_{suffix:02d}"
        suffix += 1
    path.mkdir()
    return path, config_hash


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path / "manifest.json", payload)


def _summary(
    status: str,
    numerical_audit: dict[str, Any],
    metrics: dict[str, Any],
    permutation_metrics: dict[str, Any],
    margin_analysis: dict[str, Any],
    estimate: dict[str, Any],
) -> str:
    lines = [
        "# Q1 DEVELOPMENT V1.1",
        "",
        "## STATUS",
        f"DEVELOPMENT FOLLOW-UP / {status} / NOT CONFIRMATORY",
        "",
        "## FIREWALL",
        "Confirmatory holdout accessed: NO",
        "Model, dataset, split, layer, token scope, and prompt semantics are frozen to V1.",
        "",
        "## CONTROL A — FP32 NUMERICAL AUDIT",
    ]
    for name, row in numerical_audit.get("conditions", {}).items():
        lines.append(
            f"- {name}: prediction differences {row['prediction_differences']}/512; "
            f"max score difference {_display(row['max_absolute_score_difference'])}; "
            f"median score difference {_display(row['median_absolute_score_difference'])}; "
            f"old accuracy {row['old_accuracy']:.4f}; new accuracy {row['new_accuracy']:.4f}"
        )
    lines.extend([f"- numerical audit status: {numerical_audit.get('status')}", ""])
    if status != "COMPLETE":
        lines.extend(["No remaining V1.1 controls were run after the numerical stop rule.", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## ORIGINAL-ORDER METRICS",
            "",
            "| condition | accuracy | delta | rescues | damages | headroom |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in metrics.items():
        if name == "baseline_fp32":
            continue
        lines.append(
            "| {name} | {accuracy:.4f} | {delta:.4f} | {rescues} | {damages} | {headroom} |".format(
                name=name,
                accuracy=row["treatment_accuracy"],
                delta=row["delta_accuracy"],
                rescues=row["paired_2x2"]["baseline_wrong__treatment_correct"],
                damages=row["paired_2x2"]["baseline_correct__treatment_wrong"],
                headroom=_display(row["complementarity_headroom"]),
            )
        )
    lines.extend(
        [
            "",
            "## OPTION PERMUTATIONS",
            "",
            "| ordering | PC1- delta | PC1+ delta | PC1+ rescues | PC1+ damages |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in permutation_metrics.items():
        lines.append(
            "| {name} | {minus:.4f} | {plus:.4f} | {rescues} | {damages} |".format(
                name=name,
                minus=row["minus"]["delta_accuracy"],
                plus=row["plus"]["delta_accuracy"],
                rescues=row["plus"]["paired_2x2"]["baseline_wrong__treatment_correct"],
                damages=row["plus"]["paired_2x2"]["baseline_correct__treatment_wrong"],
            )
        )
    lines.extend(
        [
            "",
            "## MARGIN ANALYSIS",
            f"PC1+ groups: {json.dumps(margin_analysis['pca_pc1_plus']['groups'], sort_keys=True)}",
            "PC1+ quartile rates: "
            + json.dumps(margin_analysis["pca_pc1_plus_quartiles"], sort_keys=True),
            "",
            "## COST ESTIMATE",
            f"Candidate forward passes: {estimate['candidate_forward_passes']}",
            f"Estimated runtime minutes: {estimate['estimated_minutes']:.2f}",
            f"A40 hourly-rate assumption: ${estimate['a40_hourly_usd_assumption']:.2f}",
            f"Estimated A40 cost: ${estimate['estimated_a40_cost_usd']:.2f}",
            "",
            "## SCIENTIFIC STATUS",
            "This is descriptive DEVELOPMENT evidence only. No condition is labeled successful.",
            "Q2 geometry was not run. V1.2 is not authorized by this artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _display(value: Any) -> str:
    if value is None or (isinstance(value, (float, int)) and not math.isfinite(float(value))):
        return "n/a"
    return f"{float(value):.4f}"


def _recover_q1_prediction_journal(path: Path) -> None:
    """Quarantine an interrupted final JSONL record without losing prior rows."""

    if not path.exists() or not path.stat().st_size:
        return
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if lines[-1].endswith((b"\n", b"\r")):
        return
    try:
        json.loads(lines[-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        quarantine = path.with_suffix(".quarantine.jsonl")
        suffix = 1
        while quarantine.exists():
            quarantine = path.with_suffix(f".quarantine.{suffix}.jsonl")
            suffix += 1
        quarantine.write_bytes(lines[-1])
        _atomic_write(path, b"".join(lines[:-1]).decode("utf-8"))
    else:
        _atomic_write(path, raw.decode("utf-8") + "\n")


def _load_q1_prediction_journal(
    path: Path,
) -> tuple[dict[tuple[str, str], Prediction], list[dict[str, Any]]]:
    _recover_q1_prediction_journal(path)
    if not path.exists():
        path.touch()
    predictions: dict[tuple[str, str], Prediction] = {}
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid V1.1 prediction journal row at line {line_number}") from exc
        key = (str(row["item_id"]), str(row["condition"]))
        if key in predictions:
            raise ValueError(f"Duplicate V1.1 prediction key: {key}")
        predictions[key] = Prediction(
            item_id=key[0],
            condition=key[1],
            raw_output=str(row["raw_output"]),
            normalized_output=str(row["normalized_output"]),
            target=str(row["target"]),
            correct=bool(row["correct"]),
            parse_status=str(row.get("parse_status", "OK")),
            metadata=dict(row.get("metadata", {})),
        )
        records.append(row)
    return predictions, records


def _append_q1_prediction(path: Path, row: dict[str, Any]) -> None:
    """Append one complete row and fsync it before the next condition proceeds."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class _BufferedQ1Journal:
    """Buffer complete rows while retaining crash-safe chunk boundaries."""

    def __init__(self, path: Path, chunk_size: int = 32) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.path = path
        self.chunk_size = chunk_size
        self._rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for row in self._rows:
                handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._rows.clear()


def _assert_q1_resume_compatible(existing: Prediction, current: Prediction) -> None:
    """Reject conflicting recomputation rather than silently keeping one row."""

    fields = ("raw_output", "normalized_output", "target", "correct", "parse_status")
    if any(getattr(existing, field) != getattr(current, field) for field in fields):
        raise ValueError(
            f"Conflicting recomputed V1.1 prediction for {existing.item_id}/{existing.condition}"
        )
    for name in ("rendered_prompt_hash", "candidate_score_semantics"):
        if existing.metadata.get(name) != current.metadata.get(name):
            raise ValueError(
                f"Conflicting V1.1 provenance for {existing.item_id}/{existing.condition}: {name}"
            )


def run_q1_v1_1(
    config: RunConfig,
    split_manifest: str | Path,
    resume_dir: str | Path | None = None,
    equivalence_only: bool = False,
) -> Path:
    """Run V1.1 or its pre-run 512-item engine-equivalence gate."""

    if config.experiment.stage != "development":
        raise ValueError("Q1 V1.1 is development-only")
    if config.backend.model_revision != V1_MODEL_REVISION:
        raise ValueError("V1.1 model revision differs from V1")
    if config.benchmark.dataset_revision != V1_DATASET_REVISION:
        raise ValueError("V1.1 dataset revision differs from V1")
    manifest_path = _resolve_path(split_manifest)
    reference = _load_v1_reference(config)
    benchmark, split = _load_evaluation(config, manifest_path)
    estimate = estimate_v1_v1(config)
    if not estimate["cost_gate_pass"]:
        raise RuntimeError(
            "V1.1 projected A40 cost "
            f"${estimate['estimated_a40_cost_usd']:.2f} exceeds $2.00 stop rule"
        )
    if equivalence_only and resume_dir is not None:
        raise ValueError("The equivalence-only gate cannot resume a scientific run")
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing V1.1 run directory: {run_dir}")
        existing_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config_hash = stable_digest(
            canonical_json({"config": config.as_dict(), "protocol": PROTOCOL_ID})
        )[:10]
        if existing_manifest.get("config_hash") != config_hash:
            raise ValueError("V1.1 resume refused: resolved config hash does not match")
        if existing_manifest.get("status") == "COMPLETE":
            raise ValueError("V1.1 run is already COMPLETE; choose a new run")
        manifest_payload = existing_manifest
        manifest_payload["status"] = "RUNNING"
    else:
        run_dir, config_hash = _run_dir(config)
        _atomic_write(
            run_dir / "config_resolved.yaml", yaml.safe_dump(config.as_dict(), sort_keys=False)
        )
        _write_json(
            run_dir / "v1_reference.json",
            {
                "run_id": reference["path"].name,
                "path": str(reference["path"]),
                "manifest": {
                    "git_commit": reference["manifest"].get("git_commit"),
                    "timestamp_utc": reference["manifest"].get("timestamp_utc"),
                    "prediction_sha256": reference["manifest"].get("prediction_sha256"),
                    "metrics_sha256": reference["manifest"].get("metrics_sha256"),
                },
                "model_revision": V1_MODEL_REVISION,
                "dataset_revision": V1_DATASET_REVISION,
                "split_manifest_sha256": V1_SPLIT_HASH,
                "vector_hashes": ORIGINAL_VECTOR_HASHES,
                "alpha_pc1_plus": reference["alpha_pc1_plus"],
                "alpha_pc1_minus": reference["alpha_pc1_minus"],
            },
        )
        manifest_payload = {
            "artifact_schema_version": 1,
            "experiment_type": "q1_v1_1_controlled_followup",
            "protocol": PROTOCOL_ID,
            "status": "RUNNING",
            "config_hash": config_hash,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "experiment_seed": config.experiment.seed,
            "benchmark": "TIGER-Lab/MMLU-Pro",
            "dataset_revision": V1_DATASET_REVISION,
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": V1_SPLIT_HASH,
            "split": "DEV_EVALUATION",
            "evaluation_item_count": EVALUATION_SIZE,
            "confirmatory_accessed": "NO",
            "holdout_access": "forbidden",
            "v1_reference_run_id": reference["path"].name,
            "v1_reference_path": str(reference["path"]),
            "v1_vector_hashes": ORIGINAL_VECTOR_HASHES,
            "inference_engine": config.backend.execution_mode,
            "candidate_head_mode": config.backend.candidate_head_mode,
            "attention_implementation_requested": config.backend.attention_implementation,
            "torch_compile": config.backend.torch_compile,
            "cuda_graphs": config.backend.cuda_graphs,
            "item_batch_size": config.backend.item_batch_size,
            "condition_chunk_size": config.backend.condition_chunk_size,
            "max_prefill_tokens": config.backend.max_prefill_tokens,
            "padding_side": config.backend.padding_side,
            "workload_estimate": estimate,
            **git_metadata(Path(__file__).resolve().parents[3]),
            **runtime_metadata(),
        }
    _write_manifest(run_dir, manifest_payload)
    try:
        backend = build_backend(config)
        model_provenance = backend.provenance()
        manifest_payload["model_provenance"] = model_provenance
        manifest_payload["benchmark_provenance"] = benchmark.provenance()
        _write_manifest(run_dir, manifest_payload)
        original_specs, vectors = _frozen_conditions(reference)
        prediction_journal = run_dir / "predictions.jsonl"
        existing_predictions, records = _load_q1_prediction_journal(prediction_journal)
        journal = _BufferedQ1Journal(prediction_journal)
        predictions_by_condition: dict[str, list[Prediction]] = {}
        for existing_prediction in existing_predictions.values():
            predictions_by_condition.setdefault(existing_prediction.condition, []).append(
                existing_prediction
            )

        def record_prediction(
            prediction: Prediction, spec: dict[str, Any]
        ) -> None:
            key = (prediction.item_id, prediction.condition)
            row = _row_payload(prediction, spec, config, model_provenance)
            if key in existing_predictions:
                _assert_q1_resume_compatible(existing_predictions[key], prediction)
                return
            journal.append(row)
            existing_predictions[key] = prediction
            records.append(row)
            predictions_by_condition.setdefault(spec["condition"], []).append(prediction)

        def evaluate_specs(items: list[BenchmarkItem], specs: list[dict[str, Any]]) -> None:
            if config.backend.execution_mode == "serial_reference":
                for item in items:
                    for spec in specs:
                        vector = (
                            vectors.get(str(spec["direction_id"]))
                            if spec["direction_id"]
                            else None
                        )
                        prediction = _run_one(backend, item, spec, benchmark.parser, vector)
                        record_prediction(prediction, spec)
                return
            if not hasattr(backend, "prepare_choice_items") or not hasattr(
                backend, "predict_choice_batch"
            ):
                raise TypeError(
                    "An optimized V1.1 execution engine requires a HuggingFace-style "
                    "prepared choice batch backend"
                )
            prepared = backend.prepare_choice_items(items)  # type: ignore[attr-defined]
            condition_inputs = [
                (
                    spec,
                    vectors.get(str(spec["direction_id"]))
                    if spec["direction_id"]
                    else None,
                )
                for spec in specs
            ]
            item_by_id = {item.id: item for item in items}
            batch_outputs = backend.predict_choice_batch(  # type: ignore[attr-defined]
                prepared, condition_inputs
            )
            for prepared_item, spec, output in batch_outputs:
                item = item_by_id[prepared_item.item_id]
                prediction = _semantic_prediction(
                    _prediction(item, spec["condition"], output, benchmark.parser), item
                )
                record_prediction(prediction, spec)

        numerical_specs = original_specs[:3]
        evaluate_specs(benchmark.items(), numerical_specs)
        journal.flush()
        numerical_audit: dict[str, Any] = {"status": "PASS", "conditions": {}}
        for spec in numerical_specs:
            name = spec["condition"]
            old_name = {
                "baseline_fp32": "baseline",
                "pca_pc1_minus_fp32": "pca_pc1_minus",
                "pca_pc1_plus_fp32": "pca_pc1_plus",
            }[name]
            differences: list[float] = []
            changed = 0
            old_correct = []
            new_correct = []
            for prediction in predictions_by_condition[name]:
                old = reference["old_rows"][(prediction.item_id, old_name)]
                old_scores = old.get("metadata", {}).get("candidate_scores", {})
                new_scores = prediction.metadata.get("candidate_scores", {})
                differences.extend(
                    abs(float(new_scores[label]) - float(old_scores[label])) for label in new_scores
                )
                if set(new_scores) != set(old_scores):
                    raise ValueError(f"Candidate score labels changed for {prediction.item_id}")
                changed += int(prediction.normalized_output != old.get("normalized_output"))
                old_correct.append(bool(old.get("correct")))
                new_correct.append(bool(prediction.correct))
            numerical_audit["conditions"][name] = {
                "v1_condition": old_name,
                "prediction_differences": changed,
                "prediction_difference_rate": changed / EVALUATION_SIZE,
                "score_comparison": (
                    "NOT_COMPARABLE_LOGITS_VS_LOG_PROBABILITIES"
                    if config.backend.candidate_head_mode == "candidate_only"
                    else "FULL_VOCAB_LOG_PROBABILITY"
                ),
                "max_absolute_score_difference": (
                    None
                    if config.backend.candidate_head_mode == "candidate_only"
                    else max(differences) if differences else 0.0
                ),
                "median_absolute_score_difference": (
                    None
                    if config.backend.candidate_head_mode == "candidate_only"
                    else float(np.median(differences)) if differences else 0.0
                ),
                "old_accuracy": float(np.mean(old_correct)),
                "new_accuracy": float(np.mean(new_correct)),
                "old_score_margin_groups": _old_margin_groups(
                    reference["old_rows"], predictions_by_condition[name], old_name
                ),
            }
        numerical_audit["max_prediction_difference_rate"] = max(
            row["prediction_difference_rate"] for row in numerical_audit["conditions"].values()
        )
        if numerical_audit["max_prediction_difference_rate"] > 0.01:
            numerical_audit["status"] = "STOP"
            _write_json(run_dir / "numerical_audit.json", numerical_audit)
            _atomic_write(
                run_dir / "predictions.jsonl",
                "".join(
                    json.dumps(_json_safe(record), sort_keys=True) + "\n" for record in records
                ),
            )
            manifest_payload.update(
                {"status": "STOPPED_NUMERICAL", "numerical_audit": numerical_audit}
            )
            _write_manifest(run_dir, manifest_payload)
            _atomic_write(
                run_dir / "summary.md",
                _summary("STOPPED_NUMERICAL", numerical_audit, {}, {}, {}, estimate),
            )
            return run_dir

        if equivalence_only:
            prediction_hash = _sha256_bytes((run_dir / "predictions.jsonl").read_bytes())
            equivalence_metrics = {
                "protocol": PROTOCOL_ID,
                "scientific_result": "NONE",
                "equivalence_only": True,
                "compared_conditions": [spec["condition"] for spec in numerical_specs],
                "numerical_audit": numerical_audit,
            }
            _write_json(run_dir / "metrics.json", equivalence_metrics)
            manifest_payload.update(
                {
                    "status": "EQUIVALENCE_COMPLETE",
                    "equivalence_only": True,
                    "scientific_result": "NONE",
                    "prediction_count": len(records),
                    "prediction_sha256": prediction_hash,
                    "numerical_audit": numerical_audit,
                }
            )
            _write_manifest(run_dir, manifest_payload)
            _atomic_write(
                run_dir / "summary.md",
                "# Q1 V1.1 Engine Equivalence Gate\n\n"
                "This artifact is an engineering gate only. It contains the three "
                "original-order conditions on DEV_EVALUATION and is not a scientific result.\n\n"
                f"Prediction rows: {len(records)}\n\n"
                f"Maximum prediction difference rate: "
                f"{numerical_audit['max_prediction_difference_rate']:.6f}\n\n"
                "Confirmatory access: NO\n",
            )
            return run_dir

        evaluate_specs(benchmark.items(), original_specs[3:])
        journal.flush()
        permutation_manifests: dict[str, list[dict[str, Any]]] = {}
        ordering_conditions: dict[str, dict[str, str]] = {
            "original": {
                "baseline": "baseline_fp32",
                "pc1_minus": "pca_pc1_minus_fp32",
                "pc1_plus": "pca_pc1_plus_fp32",
            }
        }
        permutation_ids = tuple(_v11_options(config).get("permutation_ids", PERMUTATION_IDS))
        if permutation_ids != PERMUTATION_IDS:
            raise ValueError(
                "V1.1 permutation IDs are frozen to permutation_0 through permutation_3"
            )
        for permutation_id in permutation_ids:
            permuted_items, permutation_manifest = permute_mmlu_items(
                benchmark.items(), config.experiment.seed, permutation_id
            )
            permutation_manifests[permutation_id] = permutation_manifest
            specs = _permutation_conditions(reference, permutation_id)
            evaluate_specs(permuted_items, specs)
            ordering_conditions[permutation_id] = {
                "baseline": f"{permutation_id}_baseline",
                "pc1_minus": f"{permutation_id}_pc1_minus",
                "pc1_plus": f"{permutation_id}_pc1_plus",
            }
        _write_json(run_dir / "permutation_manifests.json", permutation_manifests)
        _write_json(run_dir / "numerical_audit.json", numerical_audit)
        metrics: dict[str, Any] = {}
        for spec in original_specs:
            name = spec["condition"]
            if name == "baseline_fp32":
                continue
            metrics[name] = _paired_metrics(
                predictions_by_condition, "baseline_fp32", name, config.experiment.seed, True
            )
        baseline_self = [
            replace(row, condition="baseline_self")
            for row in predictions_by_condition["baseline_fp32"]
        ]
        metrics["baseline_fp32"] = compute_paired_metrics(
            [
                replace(row, condition="baseline")
                for row in predictions_by_condition["baseline_fp32"]
            ]
            + baseline_self,
            treatment_condition="baseline_self",
        )
        permutation_metrics: dict[str, Any] = {}
        for permutation_id in permutation_ids:
            base_name = f"{permutation_id}_baseline"
            minus_name = f"{permutation_id}_pc1_minus"
            plus_name = f"{permutation_id}_pc1_plus"
            minus = _paired_metrics(
                predictions_by_condition, base_name, minus_name, config.experiment.seed, False
            )
            plus = _paired_metrics(
                predictions_by_condition, base_name, plus_name, config.experiment.seed, False
            )
            permutation_metrics[permutation_id] = {"minus": minus, "plus": plus}

        margin_analysis = {
            "pca_pc1_plus": _margins_for_condition(
                predictions_by_condition["baseline_fp32"],
                predictions_by_condition["pca_pc1_plus_fp32"],
            ),
            "pca_pc1_minus": _margins_for_condition(
                predictions_by_condition["baseline_fp32"],
                predictions_by_condition["pca_pc1_minus_fp32"],
            ),
        }
        margin_analysis["pca_pc1_plus_quartiles"] = _margin_quartiles(
            margin_analysis["pca_pc1_plus"]["items"]
        )
        option_bias = _option_bias(predictions_by_condition, ordering_conditions)
        category_analysis = _category_analysis(
            predictions_by_condition["baseline_fp32"], predictions_by_condition["pca_pc1_plus_fp32"]
        )
        metric_payload = {
            "protocol": PROTOCOL_ID,
            "baseline_condition": "baseline_fp32",
            "conditions": metrics,
            "permutation_metrics": permutation_metrics,
            "condition_specs": original_specs
            + [
                spec
                for permutation_id in permutation_ids
                for spec in _permutation_conditions(reference, permutation_id)
            ],
            "numerical_audit": numerical_audit,
            "option_bias": option_bias,
            "margin_analysis": margin_analysis,
            "category_analysis_pc1_plus": category_analysis,
        }
        _write_json(run_dir / "metrics.json", metric_payload)
        _atomic_write(
            run_dir / "predictions.jsonl",
            "".join(
                json.dumps(_json_safe(record), sort_keys=True) + "\n" for record in records
            ),
        )
        _save_figures(
            run_dir,
            metrics,
            permutation_metrics,
            margin_analysis["pca_pc1_plus_quartiles"],
            option_bias,
        )
        prediction_hash = _sha256_bytes((run_dir / "predictions.jsonl").read_bytes())
        metrics_hash = _sha256_bytes((run_dir / "metrics.json").read_bytes())
        manifest_payload.update(
            {
                "status": "COMPLETE",
                "condition_count": len(original_specs) + len(permutation_ids) * 3,
                "condition_names": [
                    spec["condition"] for spec in metric_payload["condition_specs"]
                ],
                "prediction_count": len(records),
                "prediction_sha256": prediction_hash,
                "metrics_sha256": metrics_hash,
                "numerical_audit": numerical_audit,
                "permutation_count": len(permutation_ids),
                "permutation_manifest_sha256": _sha256_bytes(
                    (run_dir / "permutation_manifests.json").read_bytes()
                ),
            }
        )
        _write_manifest(run_dir, manifest_payload)
        _atomic_write(
            run_dir / "summary.md",
            _summary(
                "COMPLETE", numerical_audit, metrics, permutation_metrics, margin_analysis, estimate
            ),
        )
        return run_dir
    except Exception:
        journal.flush()
        manifest_payload["status"] = "FAILED"
        _write_manifest(run_dir, manifest_payload)
        raise


def repair_q1_v1_1_finalization(run_dir: str | Path) -> Path:
    """Finalize a row-complete run after a post-inference reporting failure.

    This operation never performs inference. It is deliberately limited to a
    run marked ``FAILED`` whose canonical prediction, metrics, audit, and
    permutation artifacts are already present and internally hashed.
    """

    path = Path(run_dir)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_type") != "q1_v1_1_controlled_followup":
        raise ValueError("Not a Q1 V1.1 run directory")
    if manifest.get("status") != "FAILED":
        raise ValueError("Only a FAILED Q1 V1.1 run can be repaired")
    required = (
        "config_resolved.yaml",
        "predictions.jsonl",
        "metrics.json",
        "numerical_audit.json",
        "permutation_manifests.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Cannot repair incomplete run; missing: {', '.join(missing)}")

    rows = [
        json.loads(line)
        for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != EVALUATION_SIZE * 31:
        raise ValueError("Cannot repair run with incomplete prediction rows")
    keys = [(row["item_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Cannot repair run with duplicate scientific prediction keys")

    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    numerical_audit = json.loads((path / "numerical_audit.json").read_text(encoding="utf-8"))
    if numerical_audit.get("status") != "PASS":
        raise ValueError("Cannot repair run whose numerical audit did not pass")
    prediction_hash = _sha256_bytes((path / "predictions.jsonl").read_bytes())
    metrics_hash = _sha256_bytes((path / "metrics.json").read_bytes())
    if manifest.get("prediction_sha256") != prediction_hash:
        raise ValueError("Prediction hash mismatch; refusing repair")
    if manifest.get("metrics_sha256") != metrics_hash:
        raise ValueError("Metrics hash mismatch; refusing repair")
    if len(metrics.get("condition_specs", [])) != 31:
        raise ValueError("Cannot repair run with an incomplete condition table")

    summary = _summary(
        "COMPLETE",
        numerical_audit,
        metrics["conditions"],
        metrics["permutation_metrics"],
        metrics["margin_analysis"],
        manifest["workload_estimate"],
    )
    _atomic_write(path / "summary.md", summary)
    manifest.update(
        {
            "status": "COMPLETE",
            "prediction_count": len(rows),
            "condition_count": 31,
            "prediction_sha256": prediction_hash,
            "metrics_sha256": metrics_hash,
            "numerical_audit": numerical_audit,
            "permutation_manifest_sha256": _sha256_bytes(
                (path / "permutation_manifests.json").read_bytes()
            ),
        }
    )
    _write_manifest(path, manifest)
    return path


def validate_q1_v1_1_run(run_dir: str | Path, split_manifest: str | Path) -> dict[str, Any]:
    """Validate hashes, firewall, row uniqueness, conditions, and metrics."""

    path = Path(run_dir)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment_type") != "q1_v1_1_controlled_followup":
        raise ValueError("Not a Q1 V1.1 run directory")
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"V1.1 run status is {manifest.get('status')!r}, not COMPLETE")
    if (
        manifest.get("confirmatory_accessed") != "NO"
        or manifest.get("holdout_access") != "forbidden"
    ):
        raise ValueError("V1.1 confirmatory firewall is not intact")
    resolved_config = yaml.safe_load((path / "config_resolved.yaml").read_text(encoding="utf-8"))
    expected_config_hash = stable_digest(
        canonical_json({"config": resolved_config, "protocol": PROTOCOL_ID})
    )[:10]
    if manifest.get("config_hash") != expected_config_hash:
        raise ValueError("V1.1 resolved config hash mismatch")
    if manifest.get("split_manifest_sha256") != V1_SPLIT_HASH:
        raise ValueError("V1.1 split manifest hash mismatch")
    split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    evaluation_ids = set(split["splits"]["dev_evaluation"])
    holdout_ids = set(split["splits"]["confirmatory_holdout"])
    rows = [
        json.loads(line)
        for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != EVALUATION_SIZE * 31:
        raise ValueError("V1.1 prediction row count is not 512 items x 31 conditions")
    keys = [(row["item_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate V1.1 scientific prediction key")
    if set(row["item_id"] for row in rows) != evaluation_ids:
        raise ValueError("V1.1 predictions are not exactly DEV_EVALUATION IDs")
    if set(row["item_id"] for row in rows) & holdout_ids:
        raise ValueError("V1.1 predictions contain CONFIRMATORY_HOLDOUT IDs")
    if manifest.get("prediction_count") != len(rows):
        raise ValueError("V1.1 manifest prediction count mismatch")
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    specs = {spec["condition"]: spec for spec in metrics["condition_specs"]}
    if len(specs) != 31 or manifest.get("condition_count") != 31:
        raise ValueError("V1.1 condition set is incomplete")
    if set(row["condition"] for row in rows) != set(specs):
        raise ValueError("V1.1 row conditions do not match the frozen condition set")
    prediction_objects = [
        Prediction(
            item_id=str(row["item_id"]),
            condition=str(row["condition"]),
            raw_output=str(row["raw_output"]),
            normalized_output=str(row["normalized_output"]),
            target=str(row["target"]),
            correct=bool(row["correct"]),
            parse_status=str(row.get("parse_status", "OK")),
            metadata=dict(row.get("metadata", {})),
        )
        for row in rows
    ]
    by_condition: dict[str, list[Prediction]] = {}
    for prediction in prediction_objects:
        by_condition.setdefault(prediction.condition, []).append(prediction)
    if any(
        len(rows_for_condition) != EVALUATION_SIZE for rows_for_condition in by_condition.values()
    ):
        raise ValueError("Every V1.1 condition must contain exactly 512 rows")
    for name, spec in specs.items():
        direction_id = spec.get("direction_id")
        if direction_id is not None and spec["vector_hash"] != ORIGINAL_VECTOR_HASHES.get(
            direction_id
        ):
            raise ValueError(f"Unknown V1 vector hash in {name}")
        if spec["layer"] != 17 or spec["token_scope"] != "last_token":
            raise ValueError(f"Layer/token scope mismatch in {name}")
    for name, stored in metrics["conditions"].items():
        if name == "baseline_fp32":
            baseline = [replace(row, condition="baseline") for row in by_condition["baseline_fp32"]]
            treatment = [
                replace(row, condition="baseline_self") for row in by_condition["baseline_fp32"]
            ]
            recomputed = compute_paired_metrics(
                baseline + treatment, treatment_condition="baseline_self"
            )
        else:
            baseline_name = specs[name]["baseline_condition"]
            baseline = [replace(row, condition="baseline") for row in by_condition[baseline_name]]
            treatment = [replace(row, condition=name) for row in by_condition[name]]
            recomputed = compute_paired_metrics(baseline + treatment, treatment_condition=name)
        if "bootstrap" in stored:
            recomputed["bootstrap"] = bootstrap_paired_metrics(
                baseline + treatment,
                int(manifest["experiment_seed"]),
                treatment_condition=("baseline_self" if name == "baseline_fp32" else name),
            )
        if _json_safe(stored) != _json_safe(recomputed):
            raise ValueError(f"V1.1 metric mismatch for {name}")
    for permutation_id in PERMUTATION_IDS:
        for role in ("minus", "plus"):
            treatment_name = f"{permutation_id}_pc1_{role}"
            baseline = [
                replace(row, condition="baseline")
                for row in by_condition[f"{permutation_id}_baseline"]
            ]
            treatment = [
                replace(row, condition=treatment_name) for row in by_condition[treatment_name]
            ]
            recomputed = compute_paired_metrics(
                baseline + treatment, treatment_condition=treatment_name
            )
            stored = metrics["permutation_metrics"][permutation_id][role]
            if _json_safe(stored) != _json_safe(recomputed):
                raise ValueError(f"V1.1 permutation metric mismatch for {treatment_name}")
    permutation_manifests = json.loads(
        (path / "permutation_manifests.json").read_text(encoding="utf-8")
    )
    if manifest.get("permutation_manifest_sha256") != _sha256_bytes(
        (path / "permutation_manifests.json").read_bytes()
    ):
        raise ValueError("V1.1 permutation manifest hash mismatch")
    for permutation_id in PERMUTATION_IDS:
        if len(permutation_manifests.get(permutation_id, [])) != EVALUATION_SIZE:
            raise ValueError(f"Permutation manifest incomplete for {permutation_id}")
        for row in by_condition[f"{permutation_id}_baseline"]:
            row_metadata = row.metadata
            metadata = row_metadata.get("item_metadata", {})
            semantic_ids = metadata.get("semantic_option_ids")
            if (
                not isinstance(semantic_ids, list)
                or metadata.get("permutation_id") != permutation_id
            ):
                raise ValueError(f"Permutation metadata missing for {permutation_id}/{row.item_id}")
            if metadata.get("permuted_target_index") is None:
                raise ValueError(
                    f"Permutation target remapping missing for {permutation_id}/{row.item_id}"
                )
            if semantic_ids != metadata.get("option_order_original_indices"):
                raise ValueError(
                    f"Permutation semantic order mismatch for {permutation_id}/{row.item_id}"
                )
            if row_metadata.get("semantic_target_original_index") != metadata.get(
                "original_target_index"
            ):
                raise ValueError(
                    f"Permutation target identity mismatch for {permutation_id}/{row.item_id}"
                )
        for condition in (
            f"{permutation_id}_baseline",
            f"{permutation_id}_pc1_minus",
            f"{permutation_id}_pc1_plus",
        ):
            if len(by_condition[condition]) != EVALUATION_SIZE:
                raise ValueError(f"Condition row count mismatch for {condition}")
    if manifest.get("prediction_sha256") != _sha256_bytes(
        (path / "predictions.jsonl").read_bytes()
    ):
        raise ValueError("V1.1 prediction hash mismatch")
    if manifest.get("metrics_sha256") != _sha256_bytes((path / "metrics.json").read_bytes()):
        raise ValueError("V1.1 metrics hash mismatch")
    return {
        "valid": True,
        "status": manifest["status"],
        "prediction_count": len(rows),
        "condition_count": len(specs),
        "prediction_sha256": manifest["prediction_sha256"],
        "metrics_sha256": manifest["metrics_sha256"],
        "confirmatory_accessed": "NO",
    }


def audit_q1_v1_1_repeat(
    config: RunConfig,
    split_manifest: str | Path,
    run_dir: str | Path,
    repeat_items: int = 16,
) -> dict[str, Any]:
    """Repeat four fixed conditions on a deterministic DEV_EVALUATION prefix."""

    if repeat_items <= 0 or repeat_items > EVALUATION_SIZE:
        raise ValueError("repeat_items must be between 1 and 512")
    reference = _load_v1_reference(config)
    benchmark, _split = _load_evaluation(config, _resolve_path(split_manifest))
    backend = build_backend(config)
    run_path = Path(run_dir)
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    specs = {spec["condition"]: spec for spec in metrics["condition_specs"]}
    names = [
        "baseline_fp32",
        "pca_pc1_plus_fp32",
        "random_0_normmatched_pc1_pos",
        "permutation_0_pc1_plus",
    ]
    expected = {
        (row["item_id"], row["condition"]): row
        for row in (
            json.loads(line) for line in (run_path / "predictions.jsonl").read_text().splitlines()
        )
    }
    max_score_difference = 0.0
    comparisons = 0
    for item in benchmark.items()[:repeat_items]:
        for name in names:
            spec = specs[name]
            actual_item = item
            permutation_id = spec.get("permutation_id")
            if permutation_id:
                actual_item, _manifest = permute_mmlu_item(
                    item, config.experiment.seed, permutation_id
                )
            vector = (
                reference["vectors"].get(str(spec["direction_id"]))
                if spec["direction_id"]
                else None
            )
            prediction = _run_one(backend, actual_item, spec, benchmark.parser, vector)
            old = expected[(item.id, name)]
            if prediction.normalized_output != old["normalized_output"]:
                raise ValueError(f"V1.1 repeat prediction mismatch for {item.id}/{name}")
            observed = prediction.metadata.get("candidate_scores", {})
            stored = old.get("metadata", {}).get("candidate_scores", {})
            for label in observed:
                max_score_difference = max(
                    max_score_difference, abs(float(observed[label]) - float(stored[label]))
                )
            comparisons += 1
    result = {
        "status": "PASS",
        "conditions": names,
        "items": repeat_items,
        "rows_checked": comparisons,
        "max_abs_score_difference": max_score_difference,
        "score_tolerance": 1e-5,
    }
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["repeat_check"] = result
    _write_manifest(run_path, manifest)
    return result
