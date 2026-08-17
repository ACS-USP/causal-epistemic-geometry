"""Q1 V1.2 label/position-bias deconfounding experiment.

V1.2 keeps the frozen Q1 scientific objects and replaces outcome-selected
permutations with an exact cyclic balance. The module stores raw displayed
candidate scores before computing centered semantic aggregates, so the
symmetrization is auditable and reproducible.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from epistemic_geometry.backends import build_backend
from epistemic_geometry.benchmarks.mmlu_pro import LABELS, MMLUProBenchmark
from epistemic_geometry.benchmarks.permutations import (
    cyclic_mmlu_item,
    cyclic_option_order,
    validate_cyclic_balance,
)
from epistemic_geometry.config import RunConfig
from epistemic_geometry.experiments.q1_v1_1 import (
    EVALUATION_SIZE,
    ORIGINAL_VECTOR_HASHES,
    V1_DATASET_REVISION,
    V1_SPLIT_HASH,
)
from epistemic_geometry.metrics import bootstrap_paired_metrics, compute_paired_metrics
from epistemic_geometry.reproducibility import (
    canonical_json,
    git_metadata,
    runtime_metadata,
    stable_digest,
    stable_seed,
)
from epistemic_geometry.steering import load_vector
from epistemic_geometry.types import BenchmarkItem, Prediction, PreparedChoiceItem, SteeringVector

PROTOCOL_ID = "Q1_DEVELOPMENT_PROTOCOL_V1_2"
PC1_HASH = ORIGINAL_VECTOR_HASHES["pca_pc1"]
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BETA_PROBE = 0.05
MAIN_ROLES = ("baseline", "pc1_minus", "pc1_plus")
PROBE_ROLES = ("probe_minus", "probe_plus")
ALL_ROLES = MAIN_ROLES + PROBE_ROLES


def _atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(_json_safe(row), sort_keys=True) + "\n" for row in rows),
    )


def _resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base or Path.cwd()) / path


def _q12_options(config: RunConfig) -> dict[str, Any]:
    options = dict(config.q1_v1_2)
    if options.get("protocol", PROTOCOL_ID) != PROTOCOL_ID:
        raise ValueError(f"q1_v1_2.protocol must be {PROTOCOL_ID}")
    if config.experiment.stage != "development":
        raise ValueError("V1.2 is development-only")
    if config.backend.execution_mode != "full_prompt_batched":
        raise ValueError("V1.2 requires the approved full_prompt_batched engine")
    if (
        not config.backend.serial_shape_reference
        or config.backend.candidate_head_mode != "candidate_only"
    ):
        raise ValueError("V1.2 requires serial_shape_reference and candidate_only")
    if config.backend.layer != 17 or config.steering.layer != 17:
        raise ValueError("V1.2 freezes layer 17")
    if config.steering.token_scope != "last_token":
        raise ValueError("V1.2 freezes token_scope=last_token")
    if config.backend.model_revision != MODEL_REVISION:
        raise ValueError("V1.2 model revision differs from the frozen revision")
    if config.benchmark.dataset_revision != V1_DATASET_REVISION:
        raise ValueError("V1.2 dataset revision differs from the frozen revision")
    if tuple(config.backend.candidate_labels) != LABELS:
        raise ValueError("V1.2 freezes the ten candidate labels A-J")
    return options


def _run_dir(config: RunConfig, split_manifest: Path) -> tuple[Path, str]:
    payload = {
        "config": config.as_dict(),
        "protocol": PROTOCOL_ID,
        "split_manifest_sha256": _sha256_bytes(split_manifest.read_bytes()),
    }
    config_hash = stable_digest(canonical_json(payload))[:10]
    root = _resolve_path(config.output.root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{timestamp}_q1-v1-2-development_{config_hash}"
    suffix = 1
    while path.exists():
        path = root / f"{timestamp}_q1-v1-2-development_{config_hash}_{suffix:02d}"
        suffix += 1
    path.mkdir()
    return path, config_hash


def _load_benchmark(config: RunConfig, split_manifest: Path) -> MMLUProBenchmark:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != V1_SPLIT_HASH:
        raise ValueError("V1.2 split manifest hash differs from frozen V1")
    if manifest.get("dataset_revision") != V1_DATASET_REVISION:
        raise ValueError("V1.2 dataset revision differs from frozen V1")
    evaluation_ids = set(manifest.get("splits", {}).get("dev_evaluation", []))
    holdout_ids = set(manifest.get("splits", {}).get("confirmatory_holdout", []))
    if len(evaluation_ids) != EVALUATION_SIZE or evaluation_ids & holdout_ids:
        raise ValueError("V1.2 DEV_EVALUATION firewall is invalid")
    benchmark = MMLUProBenchmark(
        split="dev_evaluation",
        dataset_revision=V1_DATASET_REVISION,
        split_manifest=split_manifest,
        dataset_id=config.benchmark.dataset_id or "TIGER-Lab/MMLU-Pro",
    )
    actual_ids = {item.id for item in benchmark}
    if actual_ids != evaluation_ids or actual_ids & holdout_ids:
        raise ValueError("V1.2 loaded item IDs violate the frozen split")
    if any(len(item.metadata.get("options", [])) != len(LABELS) for item in benchmark):
        raise ValueError("V1.2 requires exactly ten MMLU-Pro options for every item")
    return benchmark


def _load_source(config: RunConfig) -> dict[str, Any]:
    options = _q12_options(config)
    source_dir = _resolve_path(str(options.get("v1_v1_1_run_dir", "")))
    reference_dir = _resolve_path(str(options.get("v1_reference_run_dir", "")))
    if not source_dir.is_dir() or not reference_dir.is_dir():
        raise ValueError("V1.2 requires the complete V1.1 run and V1 vector run")
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    source_metrics = json.loads((source_dir / "metrics.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETE":
        raise ValueError("V1.1 source run is not COMPLETE")
    if source_manifest.get("model_provenance", {}).get("model_revision") != MODEL_REVISION:
        raise ValueError("V1.1 source model revision mismatch")
    if source_manifest.get("dataset_revision") != V1_DATASET_REVISION:
        raise ValueError("V1.1 source dataset revision mismatch")
    if source_manifest.get("candidate_head_mode") != "candidate_only":
        raise ValueError("V1.1 source candidate semantics are not candidate_only")
    vector = load_vector(reference_dir / "vectors" / "pca_pc1")
    if vector.hash != PC1_HASH or vector.layer != 17 or vector.dimension != 4096:
        raise ValueError("Frozen PC1 vector hash, layer, or dimension mismatch")
    specs = {spec["condition"]: spec for spec in source_metrics["condition_specs"]}
    for name in ("pca_pc1_minus_fp32", "pca_pc1_plus_fp32"):
        if specs[name]["vector_hash"] != PC1_HASH:
            raise ValueError(f"V1.1 source vector hash mismatch in {name}")
    rows = [
        json.loads(line)
        for line in (source_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    old_rows = {(row["item_id"], row["condition"]): row for row in rows}
    if len(rows) != 15872:
        raise ValueError("V1.1 source does not contain the complete 31-condition artifact")
    return {
        "source_dir": source_dir,
        "reference_dir": reference_dir,
        "manifest": source_manifest,
        "metrics": source_metrics,
        "old_rows": old_rows,
        "vector": vector,
        "alpha_minus": float(specs["pca_pc1_minus_fp32"]["alpha"]),
        "alpha_plus": float(specs["pca_pc1_plus_fp32"]["alpha"]),
        "direction_sd": float(specs["pca_pc1_plus_fp32"]["direction_sd_calibration"]),
    }


def _condition_specs(
    shift: int,
    alpha_minus: float,
    alpha_plus: float,
    epsilon: float,
    vector: SteeringVector,
) -> list[tuple[dict[str, Any], SteeringVector | None]]:
    prefix = f"cyclic_{shift:02d}"
    values = [
        ("baseline", 0.0, None, None),
        ("pc1_minus", alpha_minus, -0.5, vector),
        ("pc1_plus", alpha_plus, 0.5, vector),
        ("probe_minus", -epsilon, -BETA_PROBE, vector),
        ("probe_plus", epsilon, BETA_PROBE, vector),
    ]
    return [
        (
            {
                "condition": f"{prefix}_{role}",
                "role": role,
                "cyclic_shift": shift,
                "family": "v1_2_cyclic",
                "alpha": float(alpha),
                "beta": beta,
                "layer": 17,
                "token_scope": "last_token",
                "vector_hash": vector.hash if role != "baseline" else None,
                "direction_id": "pca_pc1" if role != "baseline" else None,
                "probe_beta": BETA_PROBE if role.startswith("probe") else None,
            },
            selected_vector,
        )
        for role, alpha, beta, selected_vector in values
    ]


def _item_target_index(item: BenchmarkItem) -> int:
    value = item.metadata.get("answer_index", item.metadata.get("original_target_index"))
    if value is None:
        raise ValueError(f"Item {item.id} lacks a semantic target index")
    return int(value)


def _raw_record(
    item: BenchmarkItem,
    prepared: PreparedChoiceItem,
    spec: dict[str, Any],
    scores: dict[str, float],
    cache_status: str,
    source_condition: str | None = None,
) -> dict[str, Any]:
    labels = list(prepared.candidate_labels)
    semantic_ids = list(prepared.semantic_option_ids)
    predicted_label = max(scores, key=scores.get)
    predicted_index = labels.index(predicted_label)
    predicted_semantic = int(semantic_ids[predicted_index])
    target_semantic = _item_target_index(item)
    target_index = semantic_ids.index(target_semantic)
    displayed_candidates = [
        {
            "displayed_label": label,
            "displayed_position": index,
            "semantic_original_index": int(semantic_ids[index]),
            "candidate_score": float(scores[label]),
        }
        for index, label in enumerate(labels)
    ]
    result = {
        "item_id": item.id,
        "cyclic_shift": int(spec["cyclic_shift"]),
        "condition": spec["condition"],
        "role": spec["role"],
        "option_count": len(labels),
        "candidate_labels": labels,
        "semantic_option_ids": semantic_ids,
        "candidate_scores": {label: float(scores[label]) for label in labels},
        "displayed_candidates": displayed_candidates,
        "target_semantic_original_index": target_semantic,
        "target_displayed_label": labels[target_index],
        "predicted_displayed_label": predicted_label,
        "predicted_semantic_original_index": predicted_semantic,
        "correct": bool(predicted_semantic == target_semantic),
        "rendered_prompt_hash": prepared.rendered_prompt_hash,
        "candidate_score_semantics": "candidate_logits_no_vocab_normalization",
        "cache_status": cache_status,
        "source_condition": source_condition,
        "condition_spec": spec,
    }
    return result


def _reused_record(
    item: BenchmarkItem,
    prepared: PreparedChoiceItem,
    spec: dict[str, Any],
    old_row: dict[str, Any],
    source_condition: str,
) -> dict[str, Any] | None:
    metadata = old_row.get("metadata", {})
    if metadata.get("rendered_prompt_hash") != prepared.rendered_prompt_hash:
        return None
    if metadata.get("candidate_score_semantics") != "candidate_logits_no_vocab_normalization":
        return None
    scores = metadata.get("candidate_scores")
    if not isinstance(scores, dict) or set(scores) != set(prepared.candidate_labels):
        return None
    row_provenance = old_row.get("provenance", {})
    if row_provenance.get("model_revision") != MODEL_REVISION:
        return None
    condition = row_provenance.get("condition", {})
    if source_condition != condition.get("condition"):
        return None
    if condition.get("vector_hash") not in {None, PC1_HASH}:
        return None
    if float(condition.get("alpha", float("nan"))) != float(spec["alpha"]):
        return None
    if condition.get("layer") != 17 or condition.get("token_scope") != "last_token":
        return None
    model_provenance = row_provenance.get("model_provenance", {})
    if model_provenance.get("model_revision") != MODEL_REVISION:
        return None
    if model_provenance.get("tokenizer_revision") != MODEL_REVISION:
        return None
    if row_provenance.get("candidate_head_mode") != "candidate_only":
        return None
    return _raw_record(
        item,
        prepared,
        spec,
        {key: float(value) for key, value in scores.items()},
        "CACHE_REUSED_EXACT",
        source_condition,
    )


def _symmetrize(
    raw_rows: list[dict[str, Any]], items: list[BenchmarkItem]
) -> tuple[list[dict[str, Any]], dict[str, list[Prediction]], dict[str, float]]:
    by_key = {(row["item_id"], int(row["cyclic_shift"]), row["role"]): row for row in raw_rows}
    item_by_id = {item.id: item for item in items}
    sym_rows: list[dict[str, Any]] = []
    predictions: dict[str, list[Prediction]] = {role: [] for role in MAIN_ROLES}
    agreement_counts = {role: 0.0 for role in MAIN_ROLES}
    item_ids = [item.id for item in items]
    for item_id in item_ids:
        item = item_by_id[item_id]
        option_count = len(item.metadata["options"])
        shifts = range(option_count)
        target = _item_target_index(item)
        for role in MAIN_ROLES:
            semantic_logits: list[float] = []
            semantic_probs: list[float] = []
            for semantic_index in range(option_count):
                logits: list[float] = []
                probs: list[float] = []
                for shift in shifts:
                    row = by_key[(item_id, shift, role)]
                    values = np.asarray(
                        [row["candidate_scores"][label] for label in row["candidate_labels"]],
                        dtype=np.float64,
                    )
                    centered = values - float(values.mean())
                    probabilities = np.exp(values - float(values.max()))
                    probabilities /= float(probabilities.sum())
                    displayed_index = row["semantic_option_ids"].index(semantic_index)
                    logits.append(float(centered[displayed_index]))
                    probs.append(float(probabilities[displayed_index]))
                semantic_logits.append(float(np.mean(logits)))
                semantic_probs.append(float(np.mean(probs)))
            primary_prediction = int(np.argmax(semantic_logits))
            probability_prediction = int(np.argmax(semantic_probs))
            agreement_counts[role] += float(primary_prediction == probability_prediction)
            ordered = sorted(semantic_logits, reverse=True)
            margin = ordered[0] - ordered[1] if len(ordered) > 1 else float("nan")
            sym_rows.append(
                {
                    "item_id": item_id,
                    "condition": f"{role}_sym",
                    "role": role,
                    "option_count": option_count,
                    "centered_logit_scores": semantic_logits,
                    "probability_mean_scores": semantic_probs,
                    "predicted_semantic_original_index": primary_prediction,
                    "probability_mean_prediction": probability_prediction,
                    "target_semantic_original_index": target,
                    "correct": primary_prediction == target,
                    "symmetrized_margin": margin,
                }
            )
            predictions[role].append(
                Prediction(
                    item_id=item_id,
                    condition="baseline" if role == "baseline" else f"{role}_sym",
                    raw_output=str(primary_prediction),
                    normalized_output=str(primary_prediction),
                    target=str(target),
                    correct=primary_prediction == target,
                    metadata={"symmetrized_margin": margin},
                )
            )
    agreement = {role: agreement_counts[role] / len(item_ids) for role in MAIN_ROLES}
    return sym_rows, predictions, agreement


def _paired_metrics(
    baseline: list[Prediction], treatment: list[Prediction], seed: int, resamples: int
) -> dict[str, Any]:
    treatment_condition = treatment[0].condition
    paired = [replace(row, condition="baseline") for row in baseline] + treatment
    metrics = compute_paired_metrics(paired, treatment_condition=treatment_condition)
    metrics["bootstrap"] = bootstrap_paired_metrics(
        paired, seed, n_resamples=resamples, treatment_condition=treatment_condition
    )
    return metrics


def _summary_stats(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q1": float(np.percentile(finite, 25)),
        "q3": float(np.percentile(finite, 75)),
    }


def _group_bootstrap_interval(
    values_by_item: dict[str, list[float]], seed: int, resamples: int
) -> list[float | None]:
    item_ids = sorted(values_by_item)
    if not item_ids:
        return [None, None]
    rng = np.random.default_rng(
        stable_seed("q1_v1_2_group_bootstrap", seed, len(item_ids), resamples)
    )
    means: list[float] = []
    for _ in range(resamples):
        selected = rng.integers(0, len(item_ids), size=len(item_ids))
        values = [value for index in selected for value in values_by_item[item_ids[index]]]
        finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        if finite.size:
            means.append(float(finite.mean()))
    if not means:
        return [None, None]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _directional_analysis(
    raw_rows: list[dict[str, Any]], items: list[BenchmarkItem], seed: int, resamples: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(row["item_id"], int(row["cyclic_shift"]), row["role"]): row for row in raw_rows}
    directional_rows: list[dict[str, Any]] = []
    slot_values: dict[str, list[float]] = defaultdict(list)
    slot_by_item: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    semantic_by_item: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    main_delta: dict[str, dict[str, list[float]]] = {
        "pc1_minus": defaultdict(list),
        "pc1_plus": defaultdict(list),
    }
    for item in items:
        option_count = len(item.metadata["options"])
        for shift in range(option_count):
            plus = by_key[(item.id, shift, "probe_plus")]
            minus = by_key[(item.id, shift, "probe_minus")]
            epsilon = abs(float(plus["condition_spec"]["alpha"]))
            if epsilon == 0:
                raise ValueError("V1.2 probe epsilon must be non-zero")
            labels = plus["candidate_labels"]
            raw = [
                (plus["candidate_scores"][label] - minus["candidate_scores"][label])
                / (2.0 * epsilon)
                for label in labels
            ]
            centered = np.asarray(raw, dtype=np.float64) - float(np.mean(raw))
            for index, label in enumerate(labels):
                value = float(centered[index])
                slot_values[label].append(value)
                slot_by_item[item.id][label].append(value)
                semantic_index = int(plus["semantic_option_ids"][index])
                semantic_by_item[item.id][semantic_index].append(value)
            directional_rows.append(
                {
                    "item_id": item.id,
                    "cyclic_shift": shift,
                    "displayed_labels": labels,
                    "semantic_option_ids": plus["semantic_option_ids"],
                    "raw_directional_response": raw,
                    "centered_directional_response": centered.tolist(),
                    "epsilon": epsilon,
                }
            )
            baseline = by_key[(item.id, shift, "baseline")]
            for role in ("pc1_minus", "pc1_plus"):
                intervention = by_key[(item.id, shift, role)]
                for label in labels:
                    main_delta[role][label].append(
                        intervention["candidate_scores"][label]
                        - baseline["candidate_scores"][label]
                    )
    slot_summary: dict[str, Any] = {}
    for label in sorted(slot_values, key=LABELS.index):
        stats = _summary_stats(slot_values[label])
        stats["bootstrap_interval"] = _group_bootstrap_interval(
            {item_id: values[label] for item_id, values in slot_by_item.items()}, seed, resamples
        )
        slot_summary[label] = stats
    all_means = {label: slot_summary[label]["mean"] for label in slot_summary}
    a_value = all_means.get("A")
    rest = [value for label, value in all_means.items() if label != "A" and value is not None]
    a_vs_rest = float(a_value - np.mean(rest)) if a_value is not None and rest else None
    semantic_means: dict[str, float] = {}
    for semantic_index in range(max(len(item.metadata["options"]) for item in items)):
        values = [
            float(np.mean(item_values[semantic_index]))
            for item_values in semantic_by_item.values()
            if semantic_index in item_values
        ]
        if values:
            semantic_means[str(semantic_index)] = float(np.mean(values))
    slot_mean_values = [value for value in all_means.values() if value is not None]
    semantic_mean_values = list(semantic_means.values())
    total = np.asarray([value for values in slot_values.values() for value in values], dtype=float)
    overall = float(np.mean(total)) if total.size else 0.0
    between = sum(
        len(slot_values[label]) * (float(all_means[label]) - overall) ** 2
        for label in all_means
        if all_means[label] is not None
    )
    total_ss = float(np.square(total - overall).sum()) if total.size else 0.0
    response = {
        "beta_probe": BETA_PROBE,
        "slot_summary": slot_summary,
        "semantic_mean_response": semantic_means,
        "a_vs_rest": a_vs_rest,
        "strongest_slot_by_absolute_mean": (
            max(all_means, key=lambda label: abs(all_means[label])) if slot_mean_values else None
        ),
        "slot_effect_range": float(max(slot_mean_values) - min(slot_mean_values))
        if slot_mean_values
        else None,
        "semantic_effect_range": float(max(semantic_mean_values) - min(semantic_mean_values))
        if semantic_mean_values
        else None,
        "displayed_slot_variance_fraction": between / total_ss if total_ss else None,
        "main_alpha_delta_by_displayed_label": {
            role: {label: _summary_stats(values) for label, values in by_label.items()}
            for role, by_label in main_delta.items()
        },
    }
    return directional_rows, response


def _margin_analysis(sym_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["item_id"], row["role"]): row for row in sym_rows}
    groups: dict[str, dict[str, list[float]]] = {
        "unchanged": {"baseline": [], "pc1_plus": []},
        "changed": {"baseline": [], "pc1_plus": []},
        "rescues": {"baseline": [], "pc1_plus": []},
        "damages": {"baseline": [], "pc1_plus": []},
    }
    for item_id in sorted({row["item_id"] for row in sym_rows}):
        baseline = by_key[(item_id, "baseline")]
        treatment = by_key[(item_id, "pc1_plus")]
        if (
            baseline["predicted_semantic_original_index"]
            == treatment["predicted_semantic_original_index"]
        ):
            group = "unchanged"
        elif not baseline["correct"] and treatment["correct"]:
            group = "rescues"
        elif baseline["correct"] and not treatment["correct"]:
            group = "damages"
        else:
            group = "changed"
        groups[group]["baseline"].append(float(baseline["symmetrized_margin"]))
        groups[group]["pc1_plus"].append(float(treatment["symmetrized_margin"]))
    return {
        group: {
            condition: _summary_stats(values) for condition, values in values_by_condition.items()
        }
        for group, values_by_condition in groups.items()
    }


def _category_analysis(
    sym_rows: list[dict[str, Any]], items: list[BenchmarkItem]
) -> dict[str, Any]:
    by_key = {(row["item_id"], row["role"]): row for row in sym_rows}
    categories: dict[str, dict[str, int]] = {}
    for item in items:
        baseline = by_key[(item.id, "baseline")]
        treatment = by_key[(item.id, "pc1_plus")]
        category = str(item.metadata.get("category", "UNKNOWN"))
        row = categories.setdefault(
            category,
            {"n": 0, "baseline_errors": 0, "treatment_errors": 0, "rescues": 0, "damages": 0},
        )
        row["n"] += 1
        row["baseline_errors"] += int(not baseline["correct"])
        row["treatment_errors"] += int(not treatment["correct"])
        row["rescues"] += int(not baseline["correct"] and treatment["correct"])
        row["damages"] += int(baseline["correct"] and not treatment["correct"])
    return categories


def _display_distributions(
    raw_rows: list[dict[str, Any]], sym_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in MAIN_ROLES:
        rows = [row for row in raw_rows if row["role"] == role]
        displayed = Counter(row["predicted_displayed_label"] for row in rows)
        semantic = Counter(str(row["predicted_semantic_original_index"]) for row in rows)
        n = len(rows)
        result[role] = {
            "raw_displayed_pooled": {label: displayed[label] / n for label in LABELS if n},
            "raw_semantic_pooled": {
                str(index): semantic[str(index)] / n for index in range(10) if n
            },
        }
    for role in MAIN_ROLES:
        rows = [row for row in sym_rows if row["role"] == role]
        counts = Counter(str(row["predicted_semantic_original_index"]) for row in rows)
        n = len(rows)
        result[f"{role}_sym"] = {
            "semantic_pooled": {str(index): counts[str(index)] / n for index in range(10) if n}
        }
    return result


def _save_figures(
    output_dir: Path,
    raw_reference: dict[str, Any],
    sym_metrics: dict[str, Any],
    directional: dict[str, Any],
    margins: dict[str, Any],
    distributions: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    labels = ["PC1+", "PC1-"]
    raw_delta = [
        float(raw_reference["pca_pc1_plus_fp32"]["delta_accuracy"]),
        float(raw_reference["pca_pc1_minus_fp32"]["delta_accuracy"]),
    ]
    sym_delta = [
        float(sym_metrics["pc1_plus"]["delta_accuracy"]),
        float(sym_metrics["pc1_minus"]["delta_accuracy"]),
    ]
    plt.figure(figsize=(6, 4))
    x = np.arange(2)
    plt.bar(x - 0.18, raw_delta, 0.36, label="V1.1 raw")
    plt.bar(x + 0.18, sym_delta, 0.36, label="V1.2 sym")
    plt.xticks(x, labels)
    plt.ylabel("Delta accuracy")
    plt.title("Raw versus slot-symmetrized PC1 effect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "01_raw_vs_symmetrized_effect.png", dpi=140)
    plt.close()

    slot_labels = list(directional["slot_summary"])
    slot_means = [directional["slot_summary"][label]["mean"] for label in slot_labels]
    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="black", linewidth=0.7)
    plt.bar(slot_labels, slot_means)
    plt.ylabel("Centered directional response")
    plt.title("Displayed-slot directional response")
    plt.tight_layout()
    plt.savefig(figures / "02_displayed_slot_directional_response.png", dpi=140)
    plt.close()

    a = directional.get("a_vs_rest")
    full_plus = [directional["main_alpha_delta_by_displayed_label"]["pc1_plus"]["A"]["mean"]]
    full_minus = [directional["main_alpha_delta_by_displayed_label"]["pc1_minus"]["A"]["mean"]]
    plt.figure(figsize=(5, 4))
    plt.bar(
        ["local A-vs-rest", "full PC1+ A", "full PC1- A"],
        [a or 0.0, full_plus[0] or 0.0, full_minus[0] or 0.0],
    )
    plt.xticks(rotation=20)
    plt.ylabel("Response")
    plt.title("A-slot mechanistic contrast")
    plt.tight_layout()
    plt.savefig(figures / "03_a_vs_rest_mechanistic_contrast.png", dpi=140)
    plt.close()

    rescues = [
        sym_metrics[role]["paired_2x2"]["baseline_wrong__treatment_correct"]
        for role in ("pc1_plus", "pc1_minus")
    ]
    damages = [
        sym_metrics[role]["paired_2x2"]["baseline_correct__treatment_wrong"]
        for role in ("pc1_plus", "pc1_minus")
    ]
    plt.figure(figsize=(5, 4))
    plt.bar(x - 0.18, rescues, 0.36, label="rescues")
    plt.bar(x + 0.18, damages, 0.36, label="damages")
    plt.xticks(x, labels)
    plt.legend()
    plt.title("Symmetrized rescue versus damage")
    plt.tight_layout()
    plt.savefig(figures / "04_symmetrized_rescue_damage.png", dpi=140)
    plt.close()

    group_labels = list(margins)
    baseline_medians = [margins[group]["baseline"]["median"] or 0.0 for group in group_labels]
    treatment_medians = [margins[group]["pc1_plus"]["median"] or 0.0 for group in group_labels]
    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(len(group_labels)) - 0.18, baseline_medians, 0.36, label="baseline")
    plt.bar(np.arange(len(group_labels)) + 0.18, treatment_medians, 0.36, label="PC1+")
    plt.xticks(np.arange(len(group_labels)), group_labels, rotation=20)
    plt.ylabel("Symmetrized margin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "05_symmetrized_margins.png", dpi=140)
    plt.close()

    raw_base = distributions["baseline"]["raw_displayed_pooled"]
    raw_plus = distributions["pc1_plus"]["raw_displayed_pooled"]
    plt.figure(figsize=(8, 4))
    plt.bar(
        x=np.arange(10) - 0.18,
        height=[raw_base.get(label, 0.0) for label in LABELS],
        width=0.36,
        label="baseline",
    )
    plt.bar(
        x=np.arange(10) + 0.18,
        height=[raw_plus.get(label, 0.0) for label in LABELS],
        width=0.36,
        label="PC1+",
    )
    plt.xticks(np.arange(10), LABELS)
    plt.ylabel("Displayed prediction proportion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "06_raw_displayed_prediction_distribution.png", dpi=140)
    plt.close()


def _summary(
    manifest: dict[str, Any],
    balance: dict[str, Any],
    metrics: dict[str, Any],
    directional: dict[str, Any],
    margins: dict[str, Any],
    raw_reference: dict[str, Any],
) -> str:
    plus = metrics["pc1_plus"]
    minus = metrics["pc1_minus"]
    raw_plus = raw_reference["pca_pc1_plus_fp32"]
    plus_counts = plus["paired_2x2"]
    raw_plus_counts = raw_plus["paired_2x2"]
    lines = [
        "# Q1 DEVELOPMENT V1.2 — LABEL / POSITION BIAS DECONFOUNDING",
        "",
        "**DEVELOPMENT FOLLOW-UP — NOT CONFIRMATORY**",
        "",
        "## STATUS",
        f"{manifest['status']} / confirmatory holdout accessed: NO",
        "",
        "## PROTOCOL",
        f"Model revision: {manifest['model_revision']}",
        f"Dataset revision: {manifest['dataset_revision']}",
        f"DEV items: {manifest['item_count']}",
        f"PC1 hash: {manifest['pc1_hash']}",
        f"Layer/token scope: {manifest['layer']} / {manifest['token_scope']}",
        f"Inference engine: {manifest['inference_engine']}",
        "",
        "## BALANCED DESIGN",
        f"Balance validation: {balance['status']}",
        f"Cyclic orderings: {manifest['cyclic_ordering_count']} per item",
        "Every semantic option visited every displayed slot: YES",
        "",
        "## RAW V1.1 REFERENCE",
        f"Baseline accuracy: {raw_reference['baseline_fp32']['treatment_accuracy']:.4f}",
        f"PC1+ accuracy: {raw_plus['treatment_accuracy']:.4f}",
        f"PC1+ delta: {raw_plus['delta_accuracy']:.4f}",
        "PC1+ rescues/damages: "
        f"{raw_plus_counts['baseline_wrong__treatment_correct']}/"
        f"{raw_plus_counts['baseline_correct__treatment_wrong']}",
        "",
        "## SYMMETRIZED PRIMARY RESULT",
        f"baseline_sym accuracy: {plus['baseline_accuracy']:.4f}",
        f"PC1+_sym accuracy: {plus['treatment_accuracy']:.4f}",
        f"PC1+_sym delta: {plus['delta_accuracy']:.4f}",
        "PC1+_sym rescues/damages: "
        f"{plus_counts['baseline_wrong__treatment_correct']}/"
        f"{plus_counts['baseline_correct__treatment_wrong']}",
        f"PC1+_sym error phi: {_display(plus['error_correlation_phi'])}",
        "PC1+_sym pair oracle/headroom: "
        f"{plus['pair_oracle_accuracy']:.4f} / "
        f"{_display(plus['complementarity_headroom'])}",
        "PC1-_sym accuracy/delta: "
        f"{minus['treatment_accuracy']:.4f} / {minus['delta_accuracy']:.4f}",
        "",
        "## SECONDARY AGGREGATOR",
        f"Centered-logit versus probability-mean agreement: {manifest['secondary_agreement']}",
        "",
        "## SLOT DIRECTIONAL RESPONSE",
        f"beta_probe: {directional['beta_probe']}",
        f"A-vs-rest: {_display(directional['a_vs_rest'])}",
        "strongest displayed slot by absolute mean: "
        f"{directional['strongest_slot_by_absolute_mean']}",
        "displayed-slot variance fraction: "
        f"{_display(directional['displayed_slot_variance_fraction'])}",
    ]
    for label, row in directional["slot_summary"].items():
        lines.append(f"mean centered response {label}: {_display(row['mean'])}")
    lines.extend(
        [
            "",
            "## MARGINS",
            f"unchanged baseline median: {_display(margins['unchanged']['baseline']['median'])}",
            f"changed baseline median: {_display(margins['changed']['baseline']['median'])}",
            f"rescue baseline median: {_display(margins['rescues']['baseline']['median'])}",
            f"damage baseline median: {_display(margins['damages']['baseline']['median'])}",
            "",
            "## SCIENTIFIC QUESTIONS",
            "Q1.2-A — accuracy after symmetrization: reported descriptively above.",
            "Q1.2-B — semantic rescues/damages: reported descriptively above.",
            "Q1.2-C — complementarity after symmetrization: reported descriptively above.",
            "Q1.2-D — displayed-slot directional response: see slot table above.",
            "Q1.2-E — slot versus semantic-content tracking: see stored ranges "
            "and variance fraction.",
            "Q1.2-F — low-margin concentration: see symmetrized margin groups above.",
            "",
            "## SCIENTIFIC DISCIPLINE",
            "This artifact is DEVELOPMENT evidence only. It does not establish "
            "a semantic mechanism, useful diversity, or a Q1 claim. V1.3 and "
            "Q2 were not run.",
        ]
    )
    return "\n".join(lines) + "\n"


def _display(value: Any) -> str:
    if value is None or (isinstance(value, (float, int)) and not math.isfinite(float(value))):
        return "n/a"
    return f"{float(value):.6f}"


def run_q1_v1_2(config: RunConfig, split_manifest: str | Path) -> Path:
    """Run the frozen V1.2 development deconfounding protocol."""

    split_path = _resolve_path(split_manifest)
    _q12_options(config)
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    run_dir, config_hash = _run_dir(config, split_path)
    resolved_config = config.as_dict()
    _atomic_write(run_dir / "config_resolved.yaml", yaml.safe_dump(resolved_config, sort_keys=True))
    manifest: dict[str, Any] = {
        "artifact_schema_version": 1,
        "status": "RUNNING",
        "protocol": PROTOCOL_ID,
        "protocol_stage": "DEVELOPMENT",
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config_hash": config_hash,
        "git": git_metadata(Path(__file__).resolve().parents[3]),
        "experiment_seed": config.experiment.seed,
        "split_manifest": str(split_path),
        "model_revision": MODEL_REVISION,
        "dataset_revision": V1_DATASET_REVISION,
        "split_manifest_sha256": _sha256_bytes(split_path.read_bytes()),
        "pc1_hash": PC1_HASH,
        "layer": 17,
        "token_scope": "last_token",
        "inference_engine": config.backend.execution_mode,
        "candidate_head_mode": config.backend.candidate_head_mode,
        "confirmatory_accessed": "NO",
        "holdout_access": "forbidden",
        "scientific_result": None,
    }
    _write_json(run_dir / "manifest.json", manifest)
    try:
        from epistemic_geometry.reproducibility import seed_everything

        seed_everything(config.experiment.seed)
        benchmark = _load_benchmark(config, split_path)
        items = benchmark.items()
        balance = validate_cyclic_balance(items)
        _write_json(run_dir / "balance_validation.json", balance)
        source = _load_source(config)
        vector = source["vector"]
        epsilon = BETA_PROBE * source["direction_sd"]
        if not np.isfinite(epsilon) or epsilon == 0:
            raise ValueError("V1.2 probe epsilon is non-finite or zero")
        backend = build_backend(config)
        model_provenance = backend.provenance()
        raw_rows: list[dict[str, Any]] = []
        cyclic_manifests: dict[str, list[dict[str, Any]]] = {}
        reuse_counts = Counter()
        max_options = max(len(item.metadata["options"]) for item in items)
        original_by_id = {item.id: item for item in items}
        for shift in range(max_options):
            shifted_pairs = [
                cyclic_mmlu_item(item, shift)
                for item in items
                if len(item.metadata["options"]) > shift
            ]
            if not shifted_pairs:
                continue
            shifted_items = [pair[0] for pair in shifted_pairs]
            cyclic_manifests[str(shift)] = [pair[1] for pair in shifted_pairs]
            prepared_items = backend.prepare_choice_items(shifted_items)  # type: ignore[attr-defined]
            prepared_by_id = {prepared.item_id: prepared for prepared in prepared_items}
            specs_and_vectors = _condition_specs(
                shift, source["alpha_minus"], source["alpha_plus"], epsilon, vector
            )
            compute_conditions = specs_and_vectors
            if shift == 0 and config.q1_v1_2.get("reuse_original_order", True):
                reused_rows: list[dict[str, Any]] = []
                reused_counts = Counter()
                all_reusable = True
                for shifted_item in shifted_items:
                    prepared = prepared_by_id[shifted_item.id]
                    original_item = original_by_id[shifted_item.id]
                    for role, source_condition in (
                        ("baseline", "baseline_fp32"),
                        ("pc1_minus", "pca_pc1_minus_fp32"),
                        ("pc1_plus", "pca_pc1_plus_fp32"),
                    ):
                        spec = next(
                            spec for spec, _vec in specs_and_vectors if spec["role"] == role
                        )
                        reused = _reused_record(
                            original_item,
                            prepared,
                            spec,
                            source["old_rows"][(original_item.id, source_condition)],
                            source_condition,
                        )
                        if reused is None:
                            all_reusable = False
                            break
                        reused_rows.append(reused)
                        reused_counts[role] += 1
                    if not all_reusable:
                        break
                if all_reusable:
                    raw_rows.extend(reused_rows)
                    reuse_counts.update(reused_counts)
                    compute_conditions = [
                        pair for pair in specs_and_vectors if pair[0]["role"] in PROBE_ROLES
                    ]
            if compute_conditions:
                outputs = backend.predict_choice_batch(
                    prepared_items, compute_conditions, mode=config.backend.execution_mode
                )  # type: ignore[attr-defined]
                prepared_map = {item.item_id: item for item in prepared_items}
                for prepared, spec, output in outputs:
                    item = original_by_id[prepared.item_id]
                    scores = output.metadata.get("candidate_scores")
                    if not isinstance(scores, dict):
                        raise ValueError(f"No candidate scores returned for {item.id}")
                    raw_rows.append(
                        _raw_record(
                            item,
                            prepared_map[prepared.item_id],
                            spec,
                            {key: float(value) for key, value in scores.items()},
                            "RECOMPUTED",
                        )
                    )
            _write_jsonl(run_dir / "raw_permutation_scores.jsonl", raw_rows)
        _write_json(run_dir / "cyclic_permutation_manifests.json", cyclic_manifests)
        expected_rows = sum(len(item.metadata["options"]) * len(ALL_ROLES) for item in items)
        if len(raw_rows) != expected_rows:
            raise ValueError(f"V1.2 raw row count {len(raw_rows)} != expected {expected_rows}")
        keys = [(row["item_id"], row["cyclic_shift"], row["role"]) for row in raw_rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate V1.2 item/shift/condition key")
        sym_rows, predictions, agreement = _symmetrize(raw_rows, items)
        sym_metrics = {
            "pc1_plus": _paired_metrics(
                predictions["baseline"],
                predictions["pc1_plus"],
                config.experiment.seed,
                int(config.q1_v1_2.get("bootstrap_resamples", 200)),
            ),
            "pc1_minus": _paired_metrics(
                predictions["baseline"],
                predictions["pc1_minus"],
                config.experiment.seed,
                int(config.q1_v1_2.get("bootstrap_resamples", 200)),
            ),
        }
        directional_rows, directional = _directional_analysis(
            raw_rows,
            items,
            config.experiment.seed,
            int(config.q1_v1_2.get("bootstrap_resamples", 200)),
        )
        margins = _margin_analysis(sym_rows)
        categories = _category_analysis(sym_rows, items)
        distributions = _display_distributions(raw_rows, sym_rows)
        raw_reference = source["metrics"]["conditions"]
        _write_jsonl(run_dir / "symmetrized_scores.jsonl", sym_rows)
        _write_jsonl(run_dir / "directional_responses.jsonl", directional_rows)
        _write_json(run_dir / "paired_metrics.json", sym_metrics)
        _write_json(run_dir / "slot_response_summary.json", directional)
        _write_json(run_dir / "margin_analysis.json", margins)
        _write_json(run_dir / "category_analysis.json", categories)
        _write_json(run_dir / "prediction_distributions.json", distributions)
        _save_figures(run_dir, raw_reference, sym_metrics, directional, margins, distributions)
        manifest.update(
            {
                "status": "COMPLETE",
                "item_count": len(items),
                "option_count_distribution": dict(
                    Counter(len(item.metadata["options"]) for item in items)
                ),
                "cyclic_ordering_count": max_options,
                "raw_row_count": len(raw_rows),
                "symmetrized_row_count": len(sym_rows),
                "probe_epsilon": epsilon,
                "pc1_direction_sd": source["direction_sd"],
                "alpha_minus": source["alpha_minus"],
                "alpha_plus": source["alpha_plus"],
                "secondary_agreement": agreement,
                "reuse_counts": dict(reuse_counts),
                "model_provenance": model_provenance,
                "benchmark_provenance": benchmark.provenance(),
                "runtime": runtime_metadata(),
                "source_v1_v1_1_run": str(source["source_dir"]),
                "source_v1_reference_run": str(source["reference_dir"]),
                "raw_scores_sha256": _sha256_bytes(
                    (run_dir / "raw_permutation_scores.jsonl").read_bytes()
                ),
                "symmetrized_scores_sha256": _sha256_bytes(
                    (run_dir / "symmetrized_scores.jsonl").read_bytes()
                ),
                "paired_metrics_sha256": _sha256_bytes(
                    (run_dir / "paired_metrics.json").read_bytes()
                ),
                "scientific_result": None,
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        _atomic_write(
            run_dir / "summary.md",
            _summary(manifest, balance, sym_metrics, directional, margins, raw_reference),
        )
        return run_dir
    except Exception:
        manifest["status"] = "FAILED"
        _write_json(run_dir / "manifest.json", manifest)
        raise


def validate_q1_v1_2_run(run_dir: str | Path, split_manifest: str | Path) -> dict[str, Any]:
    """Validate V1.2 provenance, raw rows, and derived artifacts from scratch.

    This validator deliberately does not load the model or the remote dataset.
    It reconstructs the lightweight semantic item metadata from the raw score
    artifact and recomputes the symmetrization and paired metrics. That makes a
    completed remote run auditable on the local machine without reproducing
    model inference.
    """

    path = Path(run_dir)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL_ID or manifest.get("status") != "COMPLETE":
        raise ValueError("V1.2 run is not a complete V1.2 artifact")
    if (
        manifest.get("item_count") != EVALUATION_SIZE
        or manifest.get("cyclic_ordering_count") != len(LABELS)
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("dataset_revision") != V1_DATASET_REVISION
    ):
        raise ValueError("V1.2 manifest does not match the frozen workload")
    if (
        manifest.get("confirmatory_accessed") != "NO"
        or manifest.get("holdout_access") != "forbidden"
    ):
        raise ValueError("V1.2 firewall is not intact")
    split = Path(split_manifest)
    if _sha256_bytes(split.read_bytes()) != V1_SPLIT_HASH:
        raise ValueError("V1.2 split hash mismatch")
    resolved_config = yaml.safe_load(
        (path / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    expected_config_hash = stable_digest(
        canonical_json(
            {
                "config": resolved_config,
                "protocol": PROTOCOL_ID,
                "split_manifest_sha256": V1_SPLIT_HASH,
            }
        )
    )[:10]
    if manifest.get("config_hash") != expected_config_hash:
        raise ValueError("V1.2 resolved config hash mismatch")
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    expected_item_id_order = split_payload.get("splits", {}).get("dev_evaluation", [])
    expected_item_ids = set(expected_item_id_order)
    holdout_ids = set(split_payload.get("splits", {}).get("confirmatory_holdout", []))
    if len(expected_item_ids) != EVALUATION_SIZE or expected_item_ids & holdout_ids:
        raise ValueError("V1.2 split manifest violates the development firewall")
    balance = json.loads((path / "balance_validation.json").read_text(encoding="utf-8"))
    if balance.get("status") != "PASS":
        raise ValueError("V1.2 balance validation did not pass")
    raw_rows = [
        json.loads(line)
        for line in (path / "raw_permutation_scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    keys = [(row["item_id"], row["cyclic_shift"], row["role"]) for row in raw_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate V1.2 raw scientific key")
    if manifest.get("raw_row_count") != len(raw_rows):
        raise ValueError("V1.2 raw row count mismatch")
    if manifest.get("raw_scores_sha256") != _sha256_bytes(
        (path / "raw_permutation_scores.jsonl").read_bytes()
    ):
        raise ValueError("V1.2 raw score hash mismatch")
    item_ids = {row["item_id"] for row in raw_rows}
    if item_ids != expected_item_ids or item_ids & holdout_ids:
        raise ValueError("V1.2 raw rows do not match exactly DEV_EVALUATION IDs")
    if manifest.get("symmetrized_scores_sha256") != _sha256_bytes(
        (path / "symmetrized_scores.jsonl").read_bytes()
    ):
        raise ValueError("V1.2 symmetrized score hash mismatch")
    if manifest.get("paired_metrics_sha256") != _sha256_bytes(
        (path / "paired_metrics.json").read_bytes()
    ):
        raise ValueError("V1.2 paired metrics hash mismatch")
    if (
        manifest.get("pc1_hash") != PC1_HASH
        or manifest.get("layer") != 17
        or manifest.get("token_scope") != "last_token"
    ):
        raise ValueError("V1.2 frozen intervention identity mismatch")
    expected_roles = set(ALL_ROLES)
    if {row["role"] for row in raw_rows} != expected_roles:
        raise ValueError("V1.2 condition roles are incomplete")
    if manifest.get("raw_row_count") != EVALUATION_SIZE * 10 * len(ALL_ROLES):
        raise ValueError("V1.2 raw row count is incompatible with the frozen 10-way design")
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_item[row["item_id"]].append(row)
        option_count = int(row["option_count"])
        shift = int(row["cyclic_shift"])
        if (
            option_count != len(LABELS)
            or row["candidate_labels"] != list(LABELS)
            or len(row["semantic_option_ids"]) != len(LABELS)
        ):
            raise ValueError(f"V1.2 candidate semantics mismatch for {row['item_id']}")
        if row["semantic_option_ids"] != cyclic_option_order(option_count, shift):
            raise ValueError(f"V1.2 cyclic mapping mismatch for {row['item_id']} shift {shift}")
        if row.get("candidate_score_semantics") != "candidate_logits_no_vocab_normalization":
            raise ValueError(f"V1.2 score semantics mismatch for {row['item_id']}")
        scores = {label: float(row["candidate_scores"][label]) for label in LABELS}
        predicted_label = max(scores, key=scores.get)
        predicted_position = LABELS.index(predicted_label)
        predicted_semantic = int(row["semantic_option_ids"][predicted_position])
        target_semantic = int(row["target_semantic_original_index"])
        target_position = row["semantic_option_ids"].index(target_semantic)
        if (
            row["predicted_displayed_label"] != predicted_label
            or int(row["predicted_semantic_original_index"]) != predicted_semantic
            or row["target_displayed_label"] != LABELS[target_position]
            or bool(row["correct"]) != (predicted_semantic == target_semantic)
        ):
            raise ValueError(
                f"V1.2 raw prediction fields are not reproducible for {row['item_id']}"
            )
    for item_id in expected_item_id_order:
        rows = by_item[item_id]
        if len({int(row["target_semantic_original_index"]) for row in rows}) != 1:
            raise ValueError(f"V1.2 target semantic identity changed for {item_id}")
        expected_keys = {
            (shift, role)
            for shift in range(10)
            for role in ALL_ROLES
        }
        actual_keys = {(int(row["cyclic_shift"]), row["role"]) for row in rows}
        if actual_keys != expected_keys:
            raise ValueError(f"V1.2 incomplete item/shift/role grid for {item_id}")

    lightweight_items = []
    for item_id in expected_item_id_order:
        first = by_item[item_id][0]
        option_count = int(first["option_count"])
        target = int(first["target_semantic_original_index"])
        lightweight_items.append(
            BenchmarkItem(
                id=item_id,
                prompt="reconstructed from V1.2 raw scores",
                target=LABELS[target],
                metadata={
                    "options": [f"option-{index}" for index in range(option_count)],
                    "answer_index": target,
                },
            )
        )
    stored_sym_rows = [
        json.loads(line)
        for line in (path / "symmetrized_scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    recomputed_sym_rows, recomputed_predictions, _agreement = _symmetrize(
        raw_rows, lightweight_items
    )
    if canonical_json(_json_safe(stored_sym_rows)) != canonical_json(
        _json_safe(recomputed_sym_rows)
    ):
        raise ValueError("V1.2 symmetrized scores are not reproducible from raw scores")
    if len(stored_sym_rows) != EVALUATION_SIZE * len(MAIN_ROLES):
        raise ValueError("V1.2 symmetrized row count mismatch")
    stored_metrics = json.loads((path / "paired_metrics.json").read_text(encoding="utf-8"))
    bootstrap_resamples = int(resolved_config.get("q1_v1_2", {}).get("bootstrap_resamples", 200))
    recomputed_metrics = {
        "pc1_plus": _paired_metrics(
            recomputed_predictions["baseline"],
            recomputed_predictions["pc1_plus"],
            int(manifest["experiment_seed"]),
            bootstrap_resamples,
        ),
        "pc1_minus": _paired_metrics(
            recomputed_predictions["baseline"],
            recomputed_predictions["pc1_minus"],
            int(manifest["experiment_seed"]),
            bootstrap_resamples,
        ),
    }
    if canonical_json(_json_safe(stored_metrics)) != canonical_json(
        _json_safe(recomputed_metrics)
    ):
        raise ValueError("V1.2 paired metrics are not reproducible from raw scores")
    directional_path = path / "directional_responses.jsonl"
    directional_rows = [
        json.loads(line) for line in directional_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(directional_rows) != EVALUATION_SIZE * len(LABELS):
        raise ValueError("V1.2 directional response row count mismatch")
    for artifact_name in (
        "cyclic_permutation_manifests.json",
        "balance_validation.json",
        "slot_response_summary.json",
        "margin_analysis.json",
        "category_analysis.json",
        "prediction_distributions.json",
        "summary.md",
    ):
        if not (path / artifact_name).is_file():
            raise ValueError(f"V1.2 required artifact is missing: {artifact_name}")
    return {
        "valid": True,
        "status": manifest["status"],
        "item_count": manifest["item_count"],
        "raw_row_count": len(raw_rows),
        "cyclic_ordering_count": manifest["cyclic_ordering_count"],
        "symmetrized_row_count": len(stored_sym_rows),
        "derived_artifacts_recomputed": True,
        "confirmatory_accessed": "NO",
        "raw_scores_sha256": manifest["raw_scores_sha256"],
    }


def estimate_q1_v1_2(config: RunConfig) -> dict[str, Any]:
    """Estimate V1.2 cost from the approved V1.1 A40 observation only."""

    _q12_options(config)
    items = EVALUATION_SIZE
    option_count = len(config.backend.candidate_labels)
    total_rows = items * option_count * len(ALL_ROLES)
    reused_rows = items * len(MAIN_ROLES)
    computed_rows = total_rows - reused_rows
    reference_minutes = 17.479421
    reference_rows = 15872
    estimated_minutes = computed_rows / reference_rows * reference_minutes
    hourly_rate = float(config.q1_v1_2.get("a40_hourly_usd_assumption", 0.44))
    return {
        "items": items,
        "option_count": option_count,
        "cyclic_orderings_per_item": option_count,
        "conditions_per_ordering": len(ALL_ROLES),
        "total_item_condition_rows": total_rows,
        "exact_original_rows_reused": reused_rows,
        "estimated_computed_rows": computed_rows,
        "reference_runtime_minutes": reference_minutes,
        "estimated_runtime_minutes": estimated_minutes,
        "a40_hourly_usd_assumption": hourly_rate,
        "estimated_a40_cost_usd": estimated_minutes / 60.0 * hourly_rate,
        "cost_gate_usd": 1.0,
        "cost_gate_pass": estimated_minutes / 60.0 * hourly_rate <= 1.0,
        "note": "Engineering estimate only; no inference or dataset/model load performed.",
    }
