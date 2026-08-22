#!/usr/bin/env python3
"""Run the Gate 6.2 source repair and its gated development phases.

The SOURCE phase consumes the immutable Gate 6.1 source trajectories and
activations.  It performs source-label-only RFM CV, recomputes causal
teacher-forced first-stage scores with the corrected suffix window, and freezes
the paired-mean bridge selection.  The manipulation/evaluation phases are
separate, crash-safe, and may not start until the source phase has selected a
controller set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.external.base import (  # noqa: E402
    ExternalItem,
    score_external_response,
)
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.experiments.gate6 import (  # noqa: E402
    ALPHA_GATE5,
    DATASET_REVISION,
    LAYERS,
    SOURCE_LOCATIONS,
    SYSTEM_CAREFUL,
    SYSTEM_DIRECT,
    RFMConfig,
    evaluation_seed,
    paired_mean_direction,
    rfm_agop_direction,
    source_readout_metrics,
    standardize_scale,
    standardized_budget,
    symmetric_first_stage_contributions,
    vector_sha256,
)
from epistemic_geometry.experiments.gate6_2 import (  # noqa: E402
    config_product,
    orthogonal_random_bank,
    paired_stratified_kfold_indices,
    select_source_cv_config,
    teacher_forced_score_window,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    git_metadata,
    require_remote_hf_execution,
    stable_seed,
)
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MAX_NEW_TOKENS = 4096
SOURCE_TRAIN_COUNT = 104
SOURCE_VALIDATION_COUNT = 32
ETA_REFERENCE_LAYER = 17
MEAN_LAYERS = (22, 27, 32)
PARSER_VERSION = "external-semantic-v1"
SOURCE_OUTPUT_NAMES = {
    "CONTROLLERS_RAW_CORRECTED.json",
    "MEAN_CONTROLLERS_RAW_CORRECTED.json",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_external(path: Path) -> list[ExternalItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        ExternalItem(
            item_id=str(row["item_id"]),
            benchmark=str(row.get("benchmark", "CRUXEval")),
            subtask=str(row.get("subtask", "output_prediction")),
            prompt=str(row["prompt"]),
            reference_answer=str(row["reference_answer"]),
            evaluator=str(row.get("evaluator", "python_literal")),
            source_revision=str(row["source_revision"]),
            metadata=dict(row.get("metadata", {})),
        )
        for row in payload["items"]
    ]


def model_item(item: ExternalItem, system_prompt: str | None = None) -> BenchmarkItem:
    metadata = {
        "source_prompt_hash": item.prompt_hash,
        "response_channel": item.metadata.get("response_channel", "cruxeval_semantic"),
    }
    if system_prompt is not None:
        metadata["system_prompt"] = system_prompt
    return BenchmarkItem(
        id=item.item_id,
        prompt=item.prompt,
        target=item.reference_answer,
        metadata=metadata,
    )


def build_backend(model_path: str | None) -> HuggingFaceBackend:
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL,
        model_path=model_path,
        model_revision=MODEL_REVISION,
        tokenizer_id=model_path or MODEL,
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        layer=17,
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
        tokenizer_identifier=model_path or MODEL,
        model_revision=MODEL_REVISION,
    )


def prompt_tokens(backend: HuggingFaceBackend, item: BenchmarkItem) -> tuple[list[int], str, str]:
    encoded, rendered, prompt_hash = backend._encode_item(item)  # noqa: SLF001
    values = encoded["input_ids"][0].detach().cpu().tolist()
    return [int(value) for value in values], rendered, prompt_hash


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_source_items(review: Path, split: str) -> list[ExternalItem]:
    path = review / f"SOURCE_SELECTED_{split.upper()}.json"
    return load_external(path)


def _load_source_rows(review: Path) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(review / "SOURCE_GENERATIONS.jsonl")
    result = {str(row["item_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate source generation key")
    return result


def _activation_arrays(
    review: Path,
    split: str,
    items: list[ExternalItem],
    location: str,
    layer: int,
    ordinary_cache: dict[tuple[str, str, int], np.ndarray] | None = None,
    require_ordinary: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(review / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    careful = np.stack(
        [archive[f"{split}__{location}__careful__{layer}__{item.item_id}"] for item in items]
    ).astype(np.float64)
    direct = np.stack(
        [archive[f"{split}__{location}__direct__{layer}__{item.item_id}"] for item in items]
    ).astype(np.float64)
    ordinary_key = f"{split}__ordinary__{location}__{layer}"
    ordinary = archive[ordinary_key]
    if ordinary.shape != (len(items), 4096) and ordinary_cache is not None:
        ordinary = np.stack(
            [ordinary_cache[(item.item_id, location, layer)] for item in items]
        ).astype(np.float64)
    if careful.shape != direct.shape or (require_ordinary and ordinary.shape != (len(items), 4096)):
        raise RuntimeError(f"malformed source activation arrays {split}:{location}:L{layer}")
    return careful, direct, ordinary


def _capture_prompt_layers(
    backend: HuggingFaceBackend, item: ExternalItem, layers: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Re-extract only missing ordinary prompt states, never source generations."""

    model_row = model_item(item)
    prompt_ids, _rendered, _prompt_hash = prompt_tokens(backend, model_row)
    input_ids = backend.torch.tensor([prompt_ids], dtype=backend.torch.long, device=backend.device)
    attention_mask = backend.torch.ones_like(input_ids)
    target = len(prompt_ids) - 1
    captured: dict[int, np.ndarray] = {}

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer] = hidden[0, target, :].detach().float().cpu().numpy().copy()
            return output

        return hook

    handles = [
        backend.layer_module(layer).register_forward_hook(make_hook(layer)) for layer in layers
    ]
    try:
        with backend.torch.inference_mode():
            backend._forward(  # noqa: SLF001
                backend.model,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                    "return_dict": True,
                },
                "prefill",
            )
    finally:
        for handle in reversed(handles):
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError("ordinary source activation re-extraction missed a layer")
    return captured


def _torch_pair(
    backend: HuggingFaceBackend, careful: np.ndarray, direct: np.ndarray, indices: np.ndarray
) -> tuple[Any, Any]:
    x = np.concatenate((careful[indices], direct[indices]))
    y = np.concatenate((np.ones(len(indices)), np.zeros(len(indices))))
    return (
        backend.torch.tensor(x, dtype=backend.torch.float32, device=backend.device),
        backend.torch.tensor(y, dtype=backend.torch.float32, device=backend.device),
    )


def _fit_rfm_with_source_cv(
    backend: HuggingFaceBackend,
    careful: np.ndarray,
    direct: np.ndarray,
    location: str,
    layer: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    configs = config_product(iters=(8,), bandwidth=(10.0,), exponent=(1.0,), regularization=(1e-3,))
    folds = paired_stratified_kfold_indices(
        len(careful), n_splits=4, seed=stable_seed("GATE6-2-RFM-CV", location, layer)
    )
    fold_results: list[dict[str, Any]] = []
    for config_dict in configs:
        config = RFMConfig(**config_dict)
        for fold_index, (train_indices, validation_indices) in enumerate(folds):
            x_train, y_train = _torch_pair(backend, careful, direct, train_indices)
            x_validation, y_validation = _torch_pair(backend, careful, direct, validation_indices)
            fit = rfm_agop_direction(
                x_train,
                y_train,
                x_validation,
                y_validation,
                config=config,
            )
            direction = np.asarray(fit["direction"], dtype=np.float64)
            train_careful = careful[train_indices]
            train_direct = direct[train_indices]
            gap = float(np.mean((train_careful - train_direct) @ direction))
            if gap < 0:
                direction = -direction
            metrics = source_readout_metrics(
                direction, careful[validation_indices], direct[validation_indices]
            )
            fold_results.append(
                {
                    "location": location,
                    "layer": layer,
                    "config": config_dict,
                    "fold": fold_index,
                    "train_item_indices": train_indices.tolist(),
                    "validation_item_indices": validation_indices.tolist(),
                    "auroc": metrics["auroc"],
                    "positive_gap_fraction": metrics["positive_gap_fraction"],
                    "best_iter": getattr(fit["rfm"], "best_iter", None),
                }
            )
    selected = select_source_cv_config(fold_results)
    selected_base = {
        key: selected[key] for key in ("iters", "bandwidth", "exponent", "regularization")
    }
    config = RFMConfig(**selected_base)
    # The final fit sees every SOURCE_TRAIN item, but its validation argument
    # is the first deterministic inner fold, never SOURCE_VALIDATION.
    train_indices, validation_indices = folds[0]
    x_full, y_full = _torch_pair(backend, careful, direct, np.arange(len(careful)))
    x_inner, y_inner = _torch_pair(backend, careful, direct, validation_indices)
    fit = rfm_agop_direction(
        x_full,
        y_full,
        x_inner,
        y_inner,
        config=config,
    )
    direction = np.asarray(fit["direction"], dtype=np.float64)
    if float(np.mean((careful - direct) @ direction)) < 0:
        direction = -direction
    return direction, {
        "location": location,
        "layer": layer,
        "cv_folds": fold_results,
        "selected_config": selected,
        "final_fit_inner_validation_indices": validation_indices.tolist(),
        "final_fit_unused_train_fold_indices": train_indices.tolist(),
        "final_best_iter": getattr(fit["rfm"], "best_iter", None),
        "vector_sha256": vector_sha256(direction),
        "semantic_outcomes_used": False,
    }


def _gate4_reference_scale(review: Path, ordinary_l17: np.ndarray) -> float:
    direction_path = ROOT / "review" / "micro_q1" / "DIRECTION.npy"
    expected_hash = "1304d6fc8dd0985895bc802885b156bc9be49d1afc58d00b013f51830cf9b9df"
    direction = np.load(direction_path, allow_pickle=False).astype(np.float64)
    if vector_sha256(direction) != expected_hash:
        raise RuntimeError("Gate-4 reference direction hash changed")
    return standardize_scale(direction, ordinary_l17)


def _source_marker(row: dict[str, Any], condition: str) -> int:
    value = row.get(f"{condition.lower()}_marker_token_index")
    if value is None:
        raise RuntimeError(f"missing source marker {row['item_id']}:{condition}")
    return int(value)


def _teacher_forced_loglik_window(
    backend: HuggingFaceBackend,
    item: BenchmarkItem,
    continuation: list[int],
    source_location: str,
    marker_token_index: int | None,
    delta: np.ndarray | None,
    layer: int,
) -> tuple[float, dict[str, Any]]:
    prompt_ids, _rendered, _prompt_hash = prompt_tokens(backend, item)
    if not continuation:
        raise ValueError("teacher-forced continuation cannot be empty")
    window = teacher_forced_score_window(
        source_location=source_location,
        continuation_length=len(continuation),
        marker_token_index=marker_token_index,
    )
    full = prompt_ids + continuation
    torch = backend.torch
    input_ids = torch.tensor([full[:-1]], dtype=torch.long, device=backend.device)
    attention_mask = torch.ones_like(input_ids)
    target_position = len(prompt_ids) - 1 + window.intervention_token_index
    context = (
        Gate6HookTrace(
            layers={layer: backend.layer_module(layer)},
            deltas={layer: torch.tensor(delta, dtype=torch.float32, device=backend.device)},
            target_positions=[target_position],
        )
        if delta is not None
        else nullcontext()
    )
    with context:
        with torch.inference_mode():
            result = backend._forward(  # noqa: SLF001
                backend.model,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                    "return_dict": True,
                },
                "prefill",
            )
    logit_start = len(prompt_ids) - 1 + window.score_start_index
    logit_end = len(prompt_ids) - 1 + window.score_end_index
    logits = result.logits[0, logit_start:logit_end].float()
    if logits.shape[0] != window.scored_token_count:
        raise RuntimeError("teacher-forced logits do not cover the frozen scoring window")
    targets = torch.tensor(
        continuation[window.score_start_index : window.score_end_index],
        dtype=torch.long,
        device=backend.device,
    )
    selected = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return float(selected.mean().item()), {
        **window.as_dict(),
        "per_token_normalized_log_likelihood": selected.detach().cpu().numpy().tolist(),
        "target_position": target_position,
    }


def _first_stage_for_candidate(
    backend: HuggingFaceBackend,
    validation: list[ExternalItem],
    source_rows: dict[str, dict[str, Any]],
    ordinary_activations: np.ndarray,
    location: str,
    layer: int,
    direction: np.ndarray,
    eta0: float,
    random_directions: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_directions = {"MEANINGFUL": direction, **random_directions}
    scores: dict[str, list[float]] = {name: [] for name in all_directions}
    windows: dict[str, list[dict[str, Any]]] = {name: [] for name in all_directions}
    for name, candidate in all_directions.items():
        delta = standardized_budget(candidate, ordinary_activations, eta0, 1)
        for item in validation:
            row = source_rows[item.item_id]
            careful = model_item(item, SYSTEM_CAREFUL)
            direct = model_item(item, SYSTEM_DIRECT)
            careful_result = []
            direct_result = []
            for sign in (1.0, -1.0):
                plus_careful, careful_window = _teacher_forced_loglik_window(
                    backend,
                    careful,
                    list(map(int, row["careful_token_ids"])),
                    location,
                    _source_marker(row, "careful") if location == "EXECUTION_BOUNDARY" else None,
                    sign * delta,
                    layer,
                )
                plus_direct, direct_window = _teacher_forced_loglik_window(
                    backend,
                    direct,
                    list(map(int, row["direct_token_ids"])),
                    location,
                    _source_marker(row, "direct") if location == "EXECUTION_BOUNDARY" else None,
                    sign * delta,
                    layer,
                )
                if sign > 0:
                    careful_result.append(plus_careful)
                    direct_result.append(plus_direct)
                    windows[name].append(
                        {
                            "item_id": item.item_id,
                            "plus_careful": careful_window,
                            "plus_direct": direct_window,
                        }
                    )
                else:
                    careful_result.append(plus_careful)
                    direct_result.append(plus_direct)
            scores[name].append(
                float(
                    symmetric_first_stage_contributions(
                        [careful_result[0]],
                        [direct_result[0]],
                        [careful_result[1]],
                        [direct_result[1]],
                    )[0]
                )
            )
    meaningful = np.asarray(scores["MEANINGFUL"], dtype=np.float64)
    random_values = np.asarray([scores[name] for name in random_directions], dtype=np.float64)
    random_mean = float(np.mean(random_values, axis=1).mean())
    random_max = float(np.max(np.mean(random_values, axis=1)))
    record = {
        "location": location,
        "layer": layer,
        "F": float(np.mean(meaningful)),
        "positive_count": int(np.sum(meaningful > 0)),
        "corrupt_count": int(np.sum(~np.isfinite(meaningful))),
        "random_mean_F": random_mean,
        "random_max_F": random_max,
        "random_F_values": [float(np.mean(values)) for values in random_values],
        "source_location_scoring_window": (
            "all_continuation_tokens"
            if location == "PROMPT_BOUNDARY"
            else "final_marker_suffix_only"
        ),
    }
    return record, {"candidate": scores, "windows": windows}


def _readout_pass(readout: dict[str, Any]) -> bool:
    return bool(readout["auroc"] >= 0.80 and readout["positive_gap_fraction"] * 32 >= 24)


def source_phase(backend: HuggingFaceBackend, review: Path) -> None:
    train = _load_source_items(review, "train")
    validation = _load_source_items(review, "validation")
    if len(train) != SOURCE_TRAIN_COUNT or len(validation) != SOURCE_VALIDATION_COUNT:
        raise RuntimeError("Gate 6.2 source counts must remain 104/32")
    source_rows = _load_source_rows(review)
    ordinary_cache: dict[tuple[str, str, int], np.ndarray] = {}
    # Gate 6.1's compact NPZ accidentally retained only the last ordinary
    # vector for each split/location/layer.  Re-extracting these prompt-only
    # states is deterministic and does not regenerate or reinterpret a source
    # trajectory; it restores the scale rows required for Gate 6.2.
    for item in train:
        for location in SOURCE_LOCATIONS:
            captured = _capture_prompt_layers(backend, item, tuple(LAYERS))
            for layer, value in captured.items():
                ordinary_cache[(item.item_id, location, layer)] = value
    ordinary_l17 = _activation_arrays(
        review, "train", train, "PROMPT_BOUNDARY", 17, ordinary_cache
    )[2]
    reference_scale = _gate4_reference_scale(review, ordinary_l17)
    eta0 = ALPHA_GATE5 / reference_scale
    rfm_records: dict[str, Any] = {}
    mean_records: dict[str, Any] = {}
    cv_records: dict[str, Any] = {}
    first_stage: dict[str, Any] = {}
    first_stage_details: dict[str, Any] = {}
    for location in SOURCE_LOCATIONS:
        for layer in LAYERS:
            train_careful, train_direct, ordinary = _activation_arrays(
                review, "train", train, location, layer, ordinary_cache
            )
            validation_careful, validation_direct, _ = _activation_arrays(
                review,
                "validation",
                validation,
                location,
                layer,
                require_ordinary=False,
            )
            rfm_direction, cv_metadata = _fit_rfm_with_source_cv(
                backend, train_careful, train_direct, location, layer
            )
            mean_direction, mean_delta, _raw = paired_mean_direction(train_careful, train_direct)
            rfm_readout = source_readout_metrics(
                rfm_direction, validation_careful, validation_direct
            )
            mean_readout = source_readout_metrics(
                mean_direction, validation_careful, validation_direct
            )
            key = f"{location}:L{layer}"
            rfm_path = review / "RFM_DIRECTIONS" / location / f"L{layer}.npy"
            mean_path = review / "PAIRED_MEAN_DIRECTIONS" / location / f"L{layer}.npy"
            rfm_path.parent.mkdir(parents=True, exist_ok=True)
            mean_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(rfm_path, rfm_direction)
            np.save(mean_path, mean_direction)
            rfm_records[key] = {
                "key": key,
                "location": location,
                "layer": layer,
                "constructor": "RFM_AGOP_SOURCE_TRAIN_CV",
                "direction_path": str(rfm_path.relative_to(ROOT)),
                "vector_hash": vector_sha256(rfm_direction),
                "readout": rfm_readout,
                "scale": standardize_scale(rfm_direction, ordinary),
                "semantic_outcomes_used": False,
            }
            mean_records[key] = {
                "key": key,
                "location": location,
                "layer": layer,
                "constructor": "PAIRED_MEAN_DIFFERENCE_SOURCE_ONLY",
                "direction_path": str(mean_path.relative_to(ROOT)),
                "vector_hash": vector_sha256(mean_direction),
                "readout": mean_readout,
                "scale": standardize_scale(mean_direction, ordinary),
                "mean_delta_train": mean_delta,
                "semantic_outcomes_used": False,
            }
            cv_records[key] = cv_metadata
            # Corrected causal first-stage is needed for all readout-eligible
            # candidates and the three pre-registered mean bridge candidates.
            candidates: list[tuple[str, np.ndarray, dict[str, Any]]] = []
            if _readout_pass(rfm_readout):
                candidates.append((f"RFM:{key}", rfm_direction, rfm_records[key]))
            if location == "PROMPT_BOUNDARY" and layer in MEAN_LAYERS:
                candidates.append((f"MEAN:{key}", mean_direction, mean_records[key]))
            for candidate_name, candidate_direction, candidate_record in candidates:
                random_bank = orthogonal_random_bank(
                    candidate_direction,
                    seeds=[
                        stable_seed("GATE6-2-RANDOM-SOURCE", candidate_name, index)
                        for index in range(4)
                    ],
                    additional_basis=(mean_direction,),
                )
                result, details = _first_stage_for_candidate(
                    backend,
                    validation,
                    source_rows,
                    ordinary,
                    location,
                    layer,
                    candidate_direction,
                    eta0,
                    random_bank,
                )
                result.update(
                    {
                        "candidate": candidate_name,
                        "constructor": candidate_record["constructor"],
                        "readout": candidate_record["readout"],
                        "readout_pass": _readout_pass(candidate_record["readout"]),
                        "pass": bool(
                            _readout_pass(candidate_record["readout"])
                            and result["F"] > 0
                            and result["positive_count"] >= 22
                            and result["F"] >= result["random_mean_F"] + 0.01
                            and result["F"] > result["random_max_F"]
                            and result["corrupt_count"] == 0
                        ),
                        "random_bank": {
                            name: {
                                "seed": stable_seed("GATE6-2-RANDOM-SOURCE", candidate_name, index),
                                "vector_sha256": vector_sha256(value),
                            }
                            for index, (name, value) in enumerate(random_bank.items())
                        },
                    }
                )
                first_stage[candidate_name] = result
                first_stage_details[candidate_name] = details
    write_json(review / "RFM_CV_RESULTS.json", cv_records)
    write_json(review / "CONTROLLERS_RAW_CORRECTED.json", rfm_records)
    write_json(review / "MEAN_CONTROLLERS_RAW_CORRECTED.json", mean_records)
    write_json(review / "FIRST_STAGE_RESULTS_CORRECTED.json", first_stage)
    write_json(review / "FIRST_STAGE_DETAILS.json", first_stage_details)
    selection = select_controller_hierarchy(rfm_records, mean_records, first_stage, eta0)
    write_json(review / "CONTROLLER_SELECTION_CORRECTED.json", selection)
    write_json(
        review / "SOURCE_PHASE_DECISION_CORRECTED.json",
        {
            "source_only_audit": "GATE6_1_SOURCE_ONLY_AUDIT_CLEAN",
            "rfm_passes": selection["rfm_passes"],
            "paired_mean_passes": selection["paired_mean_passes"],
            "continue_to_manipulation": bool(selection["continue_to_manipulation"]),
            "semantic_outcomes_used_for_selection": False,
            "source_validation_used_for": ["held_out_readout", "causal_first_stage", "random_null"],
        },
    )
    if not selection["continue_to_manipulation"]:
        raise RuntimeError("GATE6_2_NO_SOURCE_ONLY_CONTROLLER_PASSED")


def select_controller_hierarchy(
    rfm_records: dict[str, Any],
    mean_records: dict[str, Any],
    first_stage: dict[str, Any],
    eta0: float,
) -> dict[str, Any]:
    rfm_passes = [
        (key, value)
        for key, value in first_stage.items()
        if key.startswith("RFM:") and value["pass"]
    ]
    rfm_passes.sort(key=lambda value: (-value[1]["F"], value[0]))
    rfm_source = None
    rfm_layers: list[int] = []
    rfm_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for candidate, value in rfm_passes:
        base_key = candidate.removeprefix("RFM:")
        rfm_groups.setdefault(rfm_records[base_key]["location"], []).append((candidate, value))
    eligible_groups = [
        (float(np.mean([value["F"] for _candidate, value in group])), source, group)
        for source, group in rfm_groups.items()
        if len(group) >= 2
    ]
    if eligible_groups:
        _mean_f, rfm_source, selected_group = sorted(
            eligible_groups, key=lambda value: (-value[0], value[1])
        )[0]
        rfm_layers = sorted(
            int(candidate.removeprefix("RFM:").split(":L", 1)[1])
            for candidate, _value in selected_group
        )
    mean_keys = [f"PROMPT_BOUNDARY:L{layer}" for layer in MEAN_LAYERS]
    mean_passes = [
        key for key in mean_keys if first_stage.get(f"MEAN:{key}", {}).get("pass") is True
    ]
    if not all(key in mean_records for key in mean_keys):
        raise RuntimeError("paired-mean bridge records are incomplete")
    selected = {
        "eta0": eta0,
        "rfm_passes": [{"candidate": key, **value} for key, value in rfm_passes],
        "paired_mean_passes": [f"MEAN:{key}" for key in mean_passes],
        "rfm_selected_source": rfm_source,
        "rfm_selected_layers": rfm_layers,
        "best_single_mean": "PROMPT_BOUNDARY:L27",
        "multilayer_mean": mean_keys,
        "continue_to_manipulation": bool(len(mean_passes) == 3 or bool(eligible_groups)),
        "selection_rule": {
            "rfm": "if at least two pass, largest source mean F then all passing layers",
            "mean": (
                "historical source-only prompt L22/L27/L32; L27 single and all three multilayer"
            ),
            "semantic_outcomes_used": False,
        },
    }
    return selected


def _read_vectors(review: Path) -> dict[str, np.ndarray]:
    selection = json.loads((review / "CONTROLLER_SELECTION_CORRECTED.json").read_text())
    records = json.loads((review / "CONTROLLERS_RAW_CORRECTED.json").read_text())
    mean_records = json.loads((review / "MEAN_CONTROLLERS_RAW_CORRECTED.json").read_text())
    vectors: dict[str, np.ndarray] = {}
    for key, record in records.items():
        vectors[f"RFM:{key}"] = np.load(ROOT / record["direction_path"], allow_pickle=False).astype(
            np.float64
        )
    for key, record in mean_records.items():
        vectors[f"MEAN:{key}"] = np.load(
            ROOT / record["direction_path"], allow_pickle=False
        ).astype(np.float64)
    for layer in MEAN_LAYERS:
        key = f"PROMPT_BOUNDARY:L{layer}"
        meaningful = vectors[f"MEAN:{key}"]
        random_bank = orthogonal_random_bank(
            meaningful,
            seeds=[stable_seed("GATE6-2-RANDOM-MEAN-BANK", layer, index) for index in range(4)],
            additional_basis=(vectors[f"RFM:{key}"],),
        )
        for name, vector in random_bank.items():
            vectors[f"RANDOM_MEAN:{key}:{name}"] = vector
    if not selection["continue_to_manipulation"]:
        raise RuntimeError("cannot load controller vectors without a source-only pass")
    return vectors


def _delta_map(
    review: Path, vectors: dict[str, np.ndarray], controller: str
) -> dict[int, np.ndarray]:
    selection = json.loads((review / "CONTROLLER_SELECTION_CORRECTED.json").read_text())
    records = json.loads((review / "CONTROLLERS_RAW_CORRECTED.json").read_text())
    mean_records = json.loads((review / "MEAN_CONTROLLERS_RAW_CORRECTED.json").read_text())
    eta0 = float(selection["eta0"])
    if controller == "BEST_SINGLE_MEAN_PLUS":
        keys = ["PROMPT_BOUNDARY:L27"]
        prefix = "MEAN:"
        signs = 1.0
    elif controller in {"MULTILAYER_MEAN_PLUS", "MULTILAYER_MEAN_MINUS"}:
        keys = [f"PROMPT_BOUNDARY:L{layer}" for layer in MEAN_LAYERS]
        prefix = "MEAN:"
        signs = 1.0 if controller.endswith("PLUS") else -1.0
    elif controller.startswith("MULTILAYER_RANDOM_MEAN_R"):
        keys = [f"PROMPT_BOUNDARY:L{layer}" for layer in MEAN_LAYERS]
        prefix = "RANDOM_MEAN:"
        signs = 1.0
    elif controller == "BEST_SINGLE_RFM_PLUS":
        if not selection["rfm_passes"]:
            raise KeyError(controller)
        eligible = [
            entry
            for entry in selection["rfm_passes"]
            if records[entry["candidate"].removeprefix("RFM:")]["location"]
            == selection["rfm_selected_source"]
        ]
        if not eligible:
            raise KeyError(controller)
        key = sorted(eligible, key=lambda entry: (-entry["F"], entry["candidate"]))[0][
            "candidate"
        ].removeprefix("RFM:")
        keys = [key]
        prefix = "RFM:"
        signs = 1.0
    elif controller in {"MULTILAYER_RFM_PLUS", "MULTILAYER_RFM_MINUS"}:
        keys = [
            f"{selection['rfm_selected_source']}:L{layer}"
            for layer in selection["rfm_selected_layers"]
        ]
        prefix = "RFM:"
        signs = 1.0 if controller.endswith("PLUS") else -1.0
    else:
        raise KeyError(controller)
    output: dict[int, np.ndarray] = {}
    for key in keys:
        if prefix == "RANDOM_MEAN:":
            random_name = controller.removeprefix("MULTILAYER_RANDOM_MEAN_")
            vector = vectors[f"RANDOM_MEAN:{key}:{random_name}"]
            scale = float(mean_records[key]["scale"])
            layer = int(mean_records[key]["layer"])
        elif prefix == "MEAN:":
            vector = vectors[f"MEAN:{key}"]
            scale = float(mean_records[key]["scale"])
            layer = int(mean_records[key]["layer"])
        else:
            vector = vectors[f"RFM:{key}"]
            scale = float(records[key]["scale"])
            layer = int(records[key]["layer"])
        n_layers = len(keys) if len(keys) > 1 else 1
        output[layer] = signs * vector * (eta0 * scale / np.sqrt(n_layers))
    return output


def _condition_spec(review: Path) -> list[str]:
    selection = json.loads((review / "CONTROLLER_SELECTION_CORRECTED.json").read_text())
    conditions = [
        "BASELINE",
        "TEXTUAL_CAREFUL_REFERENCE",
        "TEXTUAL_DIRECT_REFERENCE",
        "BEST_SINGLE_MEAN_PLUS",
        "MULTILAYER_MEAN_PLUS",
        "MULTILAYER_MEAN_MINUS",
        "MULTILAYER_RANDOM_MEAN_R0",
        "MULTILAYER_RANDOM_MEAN_R1",
        "MULTILAYER_RANDOM_MEAN_R2",
        "MULTILAYER_RANDOM_MEAN_R3",
    ]
    if len(selection["rfm_selected_layers"]) >= 2:
        conditions.extend(("BEST_SINGLE_RFM_PLUS", "MULTILAYER_RFM_PLUS", "MULTILAYER_RFM_MINUS"))
    return conditions


def _completed(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    rows = _load_jsonl(path)
    return {(str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in rows}


def _condition_context(
    backend: HuggingFaceBackend,
    item: ExternalItem,
    condition: str,
    deltas: dict[int, np.ndarray] | None,
) -> tuple[Any, BenchmarkItem, dict[str, Any]]:
    system = None
    if condition == "TEXTUAL_CAREFUL_REFERENCE":
        system = SYSTEM_CAREFUL
    elif condition == "TEXTUAL_DIRECT_REFERENCE":
        system = SYSTEM_DIRECT
    model_row = model_item(item, system)
    prompt_ids, _rendered, _hash = prompt_tokens(backend, model_row)
    if not deltas:
        return nullcontext(), model_row, {"prompt_length": len(prompt_ids), "system_prompt": system}
    torch = backend.torch
    delta_tensors = {
        layer: torch.tensor(value, dtype=torch.float32, device=backend.device).view(1, 1, -1)
        for layer, value in deltas.items()
    }
    context = Gate6HookTrace(
        layers={layer: backend.layer_module(layer) for layer in delta_tensors},
        deltas=delta_tensors,
        target_positions=[len(prompt_ids) - 1],
    )
    return (
        context,
        model_row,
        {
            "prompt_length": len(prompt_ids),
            "system_prompt": system,
            "intervention_duration": "one_shot_prefill",
        },
    )


def execute_phase(
    backend: HuggingFaceBackend,
    review: Path,
    phase: str,
    manifest: Path,
) -> None:
    require_remote_hf_execution(f"Gate 6.2 {phase} inference")
    items = load_external(manifest)
    conditions = _condition_spec(review)
    rollouts = (0,) if phase == "MANIPULATION" else (0, 1)
    phase_name = f"CONTROLLER_{phase}"
    schedule = [
        {
            "phase": phase_name,
            "item_id": item.item_id,
            "condition": condition,
            "rollout_index": rollout,
            "seed": (
                stable_seed("GATE6-2-MANIPULATION", item.item_id)
                if phase == "MANIPULATION"
                else evaluation_seed(item.item_id, condition, rollout)
            ),
            "seed_regime": (
                "MATCHED_COUPLING_SECONDARY" if phase == "MANIPULATION" else "INDEPENDENT_PRIMARY"
            ),
        }
        for item in items
        for condition in conditions
        for rollout in rollouts
    ]
    schedule_path = review / f"{phase_name}_SCHEDULE.json"
    if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule:
        raise RuntimeError("frozen Gate 6.2 schedule differs from existing schedule")
    write_json(schedule_path, schedule)
    journal = review / "journal.jsonl"
    completed = _completed(journal)
    vectors = _read_vectors(review)
    for row in schedule:
        key = (row["item_id"], row["condition"], row["rollout_index"])
        if key in completed:
            continue
        item = next(item for item in items if item.item_id == row["item_id"])
        delta_map = None
        if row["condition"] not in {
            "BASELINE",
            "TEXTUAL_CAREFUL_REFERENCE",
            "TEXTUAL_DIRECT_REFERENCE",
        }:
            delta_map = _delta_map(review, vectors, row["condition"])
        context, model_row, context_meta = _condition_context(
            backend, item, row["condition"], delta_map
        )
        started = time.perf_counter()
        try:
            with context as trace:
                output = backend.generate_reasoning(
                    model_row,
                    sampling_seed=int(row["seed"]),
                    max_new_tokens=MAX_NEW_TOKENS,
                    intervention_metadata={
                        "gate6_2_phase": phase_name,
                        "condition": row["condition"],
                        "intervention": row["condition"] if delta_map else "none",
                        "intervention_duration": "one_shot_prefill" if delta_map else "none",
                        "intervention_layers": sorted(delta_map) if delta_map else [],
                        "intervention_vector_hashes": (
                            {str(layer): vector_sha256(value) for layer, value in delta_map.items()}
                            if delta_map
                            else {}
                        ),
                        "alpha_reference": ALPHA_GATE5,
                        "source_controller_selection": "source_only_corrected_gate6_2",
                    },
                )
            elapsed = time.perf_counter() - started
            output_metadata = dict(output.metadata)
            if delta_map:
                output_metadata["intervention_forward_trace"] = trace.metadata()
            token_count = int(output_metadata.get("generated_token_count", 0))
            scored = score_external_response(
                item,
                output.raw_output,
                rollout_seed=int(row["seed"]),
                truncated=token_count >= MAX_NEW_TOKENS,
                token_count=token_count,
                metadata={
                    "phase": phase_name,
                    "condition": row["condition"],
                    "rollout_index": row["rollout_index"],
                    "generation_seconds": elapsed,
                    "stop_metadata": output_metadata,
                },
            )
            record = {
                **row,
                "status": scored.status.value.replace("TRUNCATED_THINKING", "TRUNCATED"),
                "correct": bool(scored.correct),
                "parsed_answer": scored.parsed_answer,
                "reference_answer": item.reference_answer,
                "raw_output": scored.raw_output,
                "generated_token_ids": output_metadata.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "prompt_hash": item.prompt_hash,
                "rendered_prompt_hash": output_metadata.get("rendered_prompt_hash"),
                "source_revision": DATASET_REVISION,
                "evaluator": item.evaluator,
                "metadata": scored.metadata,
                "context_metadata": context_meta,
                "elapsed_seconds": elapsed,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
        except RuntimeError as exc:
            record = {
                **row,
                "status": "RUNTIME_ERROR",
                "correct": False,
                "parsed_answer": None,
                "raw_output": "",
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        append_jsonl(journal, record)
        if record["status"] == "RUNTIME_ERROR":
            raise RuntimeError(f"Gate 6.2 runtime failure for {key}: {record['error']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("SOURCE", "MANIPULATION", "EVALUATION"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=ROOT / "review" / "gate6_2_first_stage_repair_mean_bridge",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    require_remote_hf_execution(f"Gate 6.2 {args.phase} setup")
    backend = build_backend(args.model_path)
    review = args.review_dir.resolve()
    if args.phase == "SOURCE":
        source_phase(backend, review)
    elif args.phase == "MANIPULATION":
        execute_phase(
            backend,
            review,
            "MANIPULATION",
            args.manifest or review / "MANIPULATION_MANIFEST.json",
        )
    else:
        execute_phase(
            backend,
            review,
            "EVALUATION",
            args.manifest or review / "EVALUATION_MANIFEST.json",
        )
    write_json(
        review / f"RUN_METADATA_{args.phase}.json",
        {
            "phase": args.phase,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "source_commit": git_metadata(ROOT).get("git_commit"),
            "worktree_dirty": git_metadata(ROOT).get("git_dirty"),
            "semantic_outcomes_used_for_controller_selection": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
