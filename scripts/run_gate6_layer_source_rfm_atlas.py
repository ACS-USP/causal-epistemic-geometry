#!/usr/bin/env python3
"""Execute the frozen Gate-6 source atlas and gated controller phases.

The source phase is the only phase allowed to construct controllers.  It uses
careful/direct behavioral instructions as labels, never benchmark correctness.
The manipulation and evaluation phases consume the source-only controller
selection and journal every trajectory under its complete logical key.
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
from epistemic_geometry.experiments.gate6 import (  # noqa: E402
    ALPHA_GATE5,
    BOOTSTRAP_SEED,
    DATASET_REVISION,
    LAYERS,
    SOURCE_LOCATIONS,
    SYSTEM_CAREFUL,
    SYSTEM_DIRECT,
    RFMConfig,
    activation_pcs,
    covariance_spectrum,
    direction_alignment,
    eigenvalue_spectrum,
    evaluation_seed,
    manipulation_seed,
    orthogonal_random_bank,
    paired_mean_direction,
    rfm_agop_direction,
    source_readout_metrics,
    source_seed,
    standardize_scale,
    standardized_budget,
    symmetric_first_stage_contributions,
    vector_sha256,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    require_remote_hf_execution,
    stable_digest,
    stable_seed,
)
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GATE4_DIRECTION_HASH = "1304d6fc8dd0985895bc802885b156bc9be49d1afc58d00b013f51830cf9b9df"
MAX_NEW_TOKENS = 4096
HIDDEN_LAYERS = tuple(LAYERS)
PARSER_VERSION = "external-semantic-v1"
SOURCE_TRAIN_COUNT = 104
FRESH_PHASES = {
    "SOURCE_VALIDATION": (32, ("ORDINARY", "CAREFUL", "DIRECT"), (0, 1)),
    "CONTROLLER_MANIPULATION": (
        20,
        (
            "BASELINE",
            "TEXTUAL_CAREFUL_REFERENCE",
            "TEXTUAL_DIRECT_REFERENCE",
            "BEST_SINGLE_RFM_PLUS",
            "MULTILAYER_MEAN_PLUS",
            "MULTILAYER_RFM_PLUS",
            "MULTILAYER_RFM_MINUS",
            "MULTILAYER_RANDOM_R0",
            "MULTILAYER_RANDOM_R1",
            "MULTILAYER_RANDOM_R2",
            "MULTILAYER_RANDOM_R3",
        ),
        (0,),
    ),
    "CONTROLLER_EVALUATION": (
        60,
        (
            "BASELINE",
            "TEXTUAL_CAREFUL_REFERENCE",
            "BEST_SINGLE_RFM_PLUS",
            "MULTILAYER_MEAN_PLUS",
            "MULTILAYER_RFM_PLUS",
            "MULTILAYER_RFM_MINUS",
            "MULTILAYER_RANDOM_R0",
            "MULTILAYER_RANDOM_R1",
            "MULTILAYER_RANDOM_R2",
            "MULTILAYER_RANDOM_R3",
        ),
        (0, 1),
    ),
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
            benchmark="CRUXEval",
            subtask="output_prediction",
            prompt=str(row["prompt"]),
            reference_answer=str(row["reference_answer"]),
            evaluator="python_literal",
            source_revision=str(row["source_revision"]),
            metadata=dict(row.get("metadata", {})),
        )
        for row in payload["items"]
    ]


def load_source_training(review: Path) -> list[ExternalItem]:
    paths = (
        review.parent / "micro_q1" / "CONSTRUCTION_MANIFEST.json",
        review.parent / "gate5_source_duration" / "SOURCE_CHECK.json",
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["items"])
    if len(rows) != SOURCE_TRAIN_COUNT or len({str(row["item_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Gate-6 source training pool is not exactly 104 unique items")
    return [
        ExternalItem(
            item_id=str(row["item_id"]),
            benchmark="CRUXEval",
            subtask="output_prediction",
            prompt=str(row["prompt"]),
            reference_answer=str(row.get("reference_answer", row.get("target", ""))),
            evaluator="python_literal",
            source_revision=DATASET_REVISION,
            metadata=dict(row.get("metadata", {})),
        )
        for row in rows
    ]


def gate5_reference_scale(review: Path, ordinary_l17: np.ndarray) -> tuple[float, str]:
    """Recover the frozen Gate-5 equivalent scale from the Gate-4 controller.

    Gate 5 intervened with the frozen Gate-4 unit vector at layer 17.  Its
    standardized Gate-6 reference scale is therefore the projection spread of
    that exact vector on the ordinary Gate-6 source-training prompts, rather
    than the scale of whichever newly learned RFM happens to be encountered
    first.  This is determined before any Gate-6 source outcome is used.
    """

    path = review.parent / "micro_q1" / "DIRECTION.npy"
    vector = np.load(path, allow_pickle=False).astype(np.float64)
    digest = vector_sha256(vector)
    if digest != GATE4_DIRECTION_HASH:
        raise RuntimeError(
            "Gate-4 direction hash changed while computing the frozen Gate-5 reference scale"
        )
    return standardize_scale(vector, ordinary_l17), digest


def model_item(item: ExternalItem, system_prompt: str | None = None) -> BenchmarkItem:
    metadata = {"source_prompt_hash": item.prompt_hash, "response_channel": "cruxeval_semantic"}
    if system_prompt:
        metadata["system_prompt"] = system_prompt
    return BenchmarkItem(
        id=item.item_id,
        prompt=item.prompt,
        target=item.reference_answer,
        metadata=metadata,
    )


def build_backend(model_path: str | None) -> HuggingFaceBackend:
    from epistemic_geometry.config import BackendConfig

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


def marker_position(tokenizer: Any, continuation: list[int], prompt_length: int) -> int:
    start = marker_start(tokenizer, continuation)
    if start is not None:
        return prompt_length + start - 1 if start else prompt_length - 1
    return prompt_length - 1


def marker_start(tokenizer: Any, continuation: list[int]) -> int | None:
    marker = tokenizer.encode("FINAL:", add_special_tokens=False)
    if not marker:
        return None
    for start in range(len(continuation)):
        if continuation[start : start + len(marker)] == marker:
            return start
    return None


def capture_layers(
    backend: HuggingFaceBackend,
    item: BenchmarkItem,
    layers: tuple[int, ...],
    continuation: list[int] | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    prompt_ids, rendered, prompt_hash = prompt_tokens(backend, item)
    full_ids = prompt_ids + (continuation or [])
    target = (
        marker_position(backend.tokenizer, continuation or [], len(prompt_ids))
        if continuation is not None
        else len(prompt_ids) - 1
    )
    torch = backend.torch
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=backend.device)
    attention_mask = torch.ones_like(input_ids)
    captured: dict[int, np.ndarray] = {}

    def make_capture(layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer] = hidden[0, target, :].detach().float().cpu().numpy().copy()
            return output

        return capture

    handles = [
        backend.layer_module(layer).register_forward_hook(make_capture(layer)) for layer in layers
    ]
    try:
        with torch.inference_mode():
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
        raise RuntimeError("Gate-6 source capture missed one or more layers")
    return captured, {
        "rendered_prompt_hash": prompt_hash,
        "rendered_prompt": rendered,
        "prompt_token_count": len(prompt_ids),
        "sequence_token_count": len(full_ids),
        "target_position": target,
    }


def generate_source(
    backend: HuggingFaceBackend,
    item: ExternalItem,
    condition: str,
    phase: str,
) -> tuple[list[int], str, dict[str, Any]]:
    system = {"CAREFUL": SYSTEM_CAREFUL, "DIRECT": SYSTEM_DIRECT}.get(condition)
    seed = source_seed(item.item_id, "GENERATION", condition)
    output = backend.generate_reasoning(
        model_item(item, system),
        sampling_seed=seed,
        max_new_tokens=MAX_NEW_TOKENS,
        intervention_metadata={
            "gate6_phase": phase,
            "source_condition": condition,
            "source_generation_only": True,
        },
    )
    return (
        list(map(int, output.metadata["generated_token_ids"])),
        output.raw_output,
        output.metadata,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _next_token_log_probs(
    backend: HuggingFaceBackend,
    item: BenchmarkItem,
    continuation: list[int] | None,
    delta_map: dict[int, np.ndarray] | None,
    target_position: int | None,
) -> np.ndarray:
    """Return next-token log probabilities at a frozen source position."""

    prompt_ids, _rendered, _prompt_hash = prompt_tokens(backend, item)
    full_ids = prompt_ids + (continuation or [])
    input_ids = full_ids if continuation is None else full_ids[:-1]
    target = len(prompt_ids) - 1 if target_position is None else target_position
    torch = backend.torch
    tensors = (
        {
            layer: torch.tensor(value, dtype=torch.float32, device=backend.device).view(1, 1, -1)
            for layer, value in (delta_map or {}).items()
        }
    )
    context = (
        Gate6HookTrace(
            layers={layer: backend.layer_module(layer) for layer in tensors},
            deltas=tensors,
            target_positions=[target],
        )
        if tensors
        else nullcontext()
    )
    with context:
        with torch.inference_mode():
            result = backend._forward(  # noqa: SLF001
                backend.model,
                {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long, device=backend.device),
                    "attention_mask": torch.ones(
                        (1, len(input_ids)), dtype=torch.long, device=backend.device
                    ),
                    "use_cache": False,
                    "return_dict": True,
                },
                "prefill",
            )
    return result.logits[0, target].float().log_softmax(dim=-1).detach().cpu().numpy()


def local_control_gain_rows(
    backend: HuggingFaceBackend,
    validation: list[ExternalItem],
    ordinary: dict[str, Any],
    continuation_rows: list[dict[str, Any]],
    ordinary_values: np.ndarray,
    location: str,
    layer: int,
    direction: np.ndarray,
    constructor: str,
    *,
    eta: float,
) -> list[dict[str, Any]]:
    """Measure symmetric label-free next-token KL sensitivity on ordinary inputs."""

    rows: list[dict[str, Any]] = []
    for item in validation:
        item_row = model_item(item)
        continuation = next(
            (
                row["ordinary_token_ids"]
                for row in continuation_rows
                if row["item_id"] == item.item_id and row["split"] == "validation"
            ),
            None,
        )
        if continuation is None:
            raise RuntimeError(f"missing ordinary source continuation for {item.item_id}")
        target = (
            marker_position(
                backend.tokenizer, continuation, len(prompt_tokens(backend, item_row)[0])
            )
            if location == "EXECUTION_BOUNDARY"
            else None
        )
        delta = standardized_budget(direction, ordinary_values, eta, 1)
        teacher_tokens = continuation if location == "EXECUTION_BOUNDARY" else None
        plus = _next_token_log_probs(backend, item_row, teacher_tokens, {layer: delta}, target)
        minus = _next_token_log_probs(backend, item_row, teacher_tokens, {layer: -delta}, target)
        base = _next_token_log_probs(backend, item_row, teacher_tokens, None, target)
        p = np.exp(base)
        kl_plus = float(np.sum(p * (base - plus)))
        kl_minus = float(np.sum(p * (base - minus)))
        rows.append({
            "item_id": item.item_id,
            "source_location": location,
            "layer": layer,
            "constructor": constructor,
            "eta": eta,
            "kl_plus": kl_plus,
            "kl_minus": kl_minus,
            "symmetric_kl": 0.5 * (kl_plus + kl_minus),
            "local_control_gain": 0.5 * (kl_plus + kl_minus) / (eta * eta),
            "vector_sha256": vector_sha256(direction),
        })
    return rows


def save_activation_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def teacher_forced_loglik(
    backend: HuggingFaceBackend,
    item: BenchmarkItem,
    continuation: list[int],
    deltas: dict[int, Any] | None = None,
    target_position: int | None = None,
) -> tuple[float, np.ndarray]:
    prompt_ids, _rendered, _hash = prompt_tokens(backend, item)
    if not continuation:
        raise ValueError("teacher-forced continuation cannot be empty")
    full = prompt_ids + continuation
    torch = backend.torch
    input_ids = torch.tensor([full[:-1]], dtype=torch.long, device=backend.device)
    attention_mask = torch.ones_like(input_ids)
    target = len(prompt_ids) - 1 if target_position is None else target_position
    context = (
        Gate6HookTrace(
            layers={layer: backend.layer_module(layer) for layer in (deltas or {})},
            deltas=deltas,
            target_positions=[target],
        )
        if deltas
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
    start = len(prompt_ids) - 1
    logits = result.logits[0, start : start + len(continuation)].float()
    targets = torch.tensor(continuation, dtype=torch.long, device=backend.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    selected = log_probs.gather(1, targets[:, None]).squeeze(1)
    return float(selected.mean().item()), selected.detach().cpu().numpy()


def source_pipeline(backend: HuggingFaceBackend, review: Path, validation_path: Path) -> None:
    train = load_source_training(review)
    validation = load_external(validation_path)
    all_items = [("train", item) for item in train] + [("validation", item) for item in validation]
    continuation_rows: list[dict[str, Any]] = []
    ordinary: dict[str, dict[str, dict[int, np.ndarray]]] = {"train": {}, "validation": {}}
    paired: dict[str, dict[str, dict[str, dict[int, np.ndarray]]]] = {
        "train": {location: {} for location in SOURCE_LOCATIONS},
        "validation": {location: {} for location in SOURCE_LOCATIONS},
    }
    source_journal = review / "SOURCE_GENERATION_JOURNAL.jsonl"
    for split, item in all_items:
        item_row: dict[str, Any] = {"split": split, "item_id": item.item_id}
        ordinary_item = model_item(item)
        ordinary_tokens, ordinary_raw, ordinary_generation_meta = generate_source(
            backend, item, "ORDINARY", split
        )
        ordinary_prompt, _rendered, _hash = prompt_tokens(backend, ordinary_item)
        ordinary_captured, ordinary_prompt_meta = capture_layers(
            backend, ordinary_item, HIDDEN_LAYERS
        )
        ordinary_execution, ordinary_execution_meta = capture_layers(
            backend, ordinary_item, HIDDEN_LAYERS, ordinary_tokens
        )
        ordinary[split][item.item_id] = {
            "PROMPT_BOUNDARY": ordinary_captured,
            "EXECUTION_BOUNDARY": ordinary_execution,
        }
        item_row["ordinary_prompt_token_count"] = len(ordinary_prompt)
        item_row["ordinary_token_ids"] = ordinary_tokens
        item_row["ordinary_raw_output"] = ordinary_raw
        item_row["ordinary_generation_metadata"] = ordinary_generation_meta
        item_row["ordinary_prompt_meta"] = ordinary_prompt_meta
        item_row["ordinary_execution_meta"] = ordinary_execution_meta
        item_row["ordinary_final_marker_found"] = marker_start(
            backend.tokenizer, ordinary_tokens
        ) is not None
        for condition in ("CAREFUL", "DIRECT"):
            tokens, raw, generation_meta = generate_source(backend, item, condition, split)
            item_row[f"{condition.lower()}_token_ids"] = tokens
            item_row[f"{condition.lower()}_raw_output"] = raw
            condition_item = model_item(
                item, SYSTEM_CAREFUL if condition == "CAREFUL" else SYSTEM_DIRECT
            )
            prompt_captured, prompt_meta = capture_layers(backend, condition_item, HIDDEN_LAYERS)
            execution_captured, execution_meta = capture_layers(
                backend, condition_item, HIDDEN_LAYERS, tokens
            )
            for location, captured in (
                ("PROMPT_BOUNDARY", prompt_captured),
                ("EXECUTION_BOUNDARY", execution_captured),
            ):
                paired[split][location].setdefault(item.item_id, {})[condition.lower()] = captured
            item_row[f"{condition.lower()}_generation_metadata"] = generation_meta
            item_row[f"{condition.lower()}_prompt_meta"] = prompt_meta
            item_row[f"{condition.lower()}_execution_meta"] = execution_meta
            item_row[f"{condition.lower()}_final_marker_found"] = marker_start(
                backend.tokenizer, tokens
            ) is not None
        item_row["source_pair_eligible"] = bool(
            item_row["careful_final_marker_found"]
            and item_row["direct_final_marker_found"]
        )
        continuation_rows.append(item_row)
        append_jsonl(source_journal, item_row)
        if not item_row["source_pair_eligible"]:
            raise RuntimeError(
                f"GATE6_SOURCE_TRAJECTORY_INELIGIBLE: FINAL marker missing for {item.item_id}"
            )
    save_activation_archive(
        review / "SOURCE_ACTIVATIONS.npz",
        {
            f"{split}__ordinary__{location}__{layer}": ordinary[split][item_id][location][layer]
            for split, item_id in ((split, item.item_id) for split, item in all_items)
            for location in ordinary[split][item_id]
            for layer in HIDDEN_LAYERS
        }
        | {
            f"{split}__{location}__{condition}__{layer}__{item_id}": paired[split][location][
                item_id
            ][condition][layer]
            for split, item in all_items
            for location in SOURCE_LOCATIONS
            for condition in ("careful", "direct")
            for layer in HIDDEN_LAYERS
            for item_id in (item.item_id,)
        },
    )
    (review / "SOURCE_GENERATIONS.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in continuation_rows) + "\n",
        encoding="utf-8",
    )
    train_data: dict[str, Any] = {}
    validation_data: dict[str, Any] = {}
    controller_records: dict[str, Any] = {}
    mean_records: dict[str, Any] = {}
    source_metrics: dict[str, Any] = {}
    readout_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    local_control_rows: list[dict[str, Any]] = []
    activation_metadata: list[dict[str, Any]] = []
    ordinary_l17 = np.stack(
        [ordinary["train"][item.item_id]["PROMPT_BOUNDARY"][17] for item in train]
    )
    reference_scale, reference_hash = gate5_reference_scale(review, ordinary_l17)
    for location in SOURCE_LOCATIONS:
        train_data[location] = {}
        validation_data[location] = {}
        for layer in HIDDEN_LAYERS:
            train_careful = np.stack(
                [paired["train"][location][item.item_id]["careful"][layer] for item in train]
            )
            train_direct = np.stack(
                [paired["train"][location][item.item_id]["direct"][layer] for item in train]
            )
            validation_careful = np.stack(
                [
                    paired["validation"][location][item.item_id]["careful"][layer]
                    for item in validation
                ]
            )
            validation_direct = np.stack(
                [
                    paired["validation"][location][item.item_id]["direct"][layer]
                    for item in validation
                ]
            )
            train_data[location][layer] = (train_careful, train_direct)
            validation_data[location][layer] = (validation_careful, validation_direct)
            x_train = backend.torch.tensor(
                np.concatenate((train_careful, train_direct)),
                dtype=backend.torch.float32,
                device=backend.device,
            )
            y_train = backend.torch.tensor(
                np.concatenate((np.ones(len(train_careful)), np.zeros(len(train_direct)))),
                dtype=backend.torch.float32,
                device=backend.device,
            )
            fit = rfm_agop_direction(
                x_train,
                y_train,
                x_train,
                y_train,
                config=RFMConfig(),
            )
            direction = np.asarray(fit["direction"], dtype=np.float64)
            if float(np.mean((validation_careful - validation_direct) @ direction)) < 0:
                direction = -direction
            mean_direction, mean_delta, _mean_raw = paired_mean_direction(
                train_careful, train_direct
            )
            if float(np.mean((validation_careful - validation_direct) @ mean_direction)) < 0:
                mean_direction = -mean_direction
            readout = source_readout_metrics(direction, validation_careful, validation_direct)
            mean_readout = source_readout_metrics(
                mean_direction, validation_careful, validation_direct
            )
            ordinary_values = np.stack(
                [ordinary["train"][item.item_id][location][layer] for item in train]
            )
            scale = standardize_scale(direction, ordinary_values)
            mean_scale = standardize_scale(mean_direction, ordinary_values)
            key = f"{location}:L{layer}"
            source_metrics[key] = {
                "readout": readout,
                "mean_readout": mean_readout,
                "scale_ordinary_prompt_reference": scale,
                "mean_scale_ordinary_prompt_reference": mean_scale,
                "vector_norm": float(np.linalg.norm(direction)),
                "vector_hash": vector_sha256(direction),
                "mean_vector_hash": vector_sha256(mean_direction),
                "rfm_spectrum": eigenvalue_spectrum(fit["eigenvalues"]),
                "activation_spectrum": covariance_spectrum(
                    np.concatenate((train_careful, train_direct))
                ),
                "mean_rfm_alignment": direction_alignment(mean_direction, direction),
                "mean_delta_train": mean_delta,
            }
            direction_path = review / "DIRECTIONS" / location / f"L{layer}.npy"
            direction_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(direction_path, direction)
            mean_path = review / "MEAN_DIRECTIONS" / location / f"L{layer}.npy"
            mean_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mean_path, mean_direction)
            controller_records[key] = {
                "location": location,
                "layer": layer,
                "constructor": "RFM_AGOP",
                "direction_path": str(direction_path.relative_to(ROOT)),
                "vector_hash": source_metrics[key]["vector_hash"],
                "readout": readout,
                "scale": scale,
                "rfm_config": fit["config"],
            }
            mean_records[key] = {
                "location": location,
                "layer": layer,
                "constructor": "PAIRED_MEAN_DIFFERENCE",
                "direction_path": str(mean_path.relative_to(ROOT)),
                "vector_hash": source_metrics[key]["mean_vector_hash"],
                "readout": mean_readout,
                "scale": mean_scale,
            }
            for constructor, candidate, candidate_readout, candidate_scale in (
                ("RFM_AGOP", direction, readout, scale),
                ("PAIRED_MEAN_DIFFERENCE", mean_direction, mean_readout, mean_scale),
            ):
                readout_rows.append(
                    {
                        "source_location": location,
                        "layer": layer,
                        "constructor": constructor,
                        **candidate_readout,
                        "ordinary_prompt_projection_scale": candidate_scale,
                        "vector_sha256": vector_sha256(candidate),
                    }
                )
                pooled = np.concatenate((train_careful, train_direct))
                pcs = activation_pcs(pooled)
                spectral_rows.append(
                    {
                        "source_location": location,
                        "layer": layer,
                        "constructor": constructor,
                        "vector_sha256": vector_sha256(candidate),
                        "mean_rfm_alignment": direction_alignment(candidate, direction),
                        "top_pc_abs_alignment": [
                            float(abs(np.dot(candidate, pc))) for pc in pcs["components"]
                        ],
                        "rfm_spectrum": source_metrics[key]["rfm_spectrum"],
                        "activation_spectrum": source_metrics[key]["activation_spectrum"],
                    }
                )
            activation_metadata.append(
                {
                    "source_location": location,
                    "layer": layer,
                    "train_activation_shape": list(
                        np.concatenate((train_careful, train_direct)).shape
                    ),
                    "validation_careful_shape": list(validation_careful.shape),
                    "validation_direct_shape": list(validation_direct.shape),
                    "train_careful_hash": stable_digest(
                        "GATE6-ACTIVATION",
                        location,
                        layer,
                        "train-careful",
                        train_careful.tobytes(),
                    ),
                    "train_direct_hash": stable_digest(
                        "GATE6-ACTIVATION",
                        location,
                        layer,
                        "train-direct",
                        train_direct.tobytes(),
                    ),
                }
            )
            for constructor, candidate, _candidate_scale in (
                ("RFM_AGOP", direction, scale),
                ("PAIRED_MEAN_DIFFERENCE", mean_direction, mean_scale),
            ):
                local_control_rows.extend(
                    local_control_gain_rows(
                        backend,
                        validation,
                        ordinary,
                        continuation_rows,
                        ordinary_values,
                        location,
                        layer,
                        candidate,
                        constructor,
                        eta=0.05,
                    )
                )
    write_json(review / "SOURCE_METRICS.json", source_metrics)
    write_json(review / "CONTROLLERS_RAW.json", controller_records)
    write_json(review / "MEAN_CONTROLLERS_RAW.json", mean_records)
    write_json(
        review / "ACTIVATION_METADATA.json",
        {
            "model": MODEL,
            "revision": MODEL_REVISION,
            "layers": list(HIDDEN_LAYERS),
            "source_locations": list(SOURCE_LOCATIONS),
            "records": activation_metadata,
        },
    )
    _write_csv(review / "READOUT_ATLAS.csv", readout_rows)
    _write_csv(review / "SPECTRAL_ATLAS.csv", spectral_rows)
    write_json(review / "SPECTRAL_ATLAS.json", spectral_rows)
    _write_csv(review / "LOCAL_CONTROL_GAIN.csv", local_control_rows)
    write_json(
        review / "GATE5_REFERENCE_SCALE.json",
        {
            "alpha_gate5": ALPHA_GATE5,
            "s17_gate5": reference_scale,
            "eta0": ALPHA_GATE5 / reference_scale,
            "gate4_direction_sha256": reference_hash,
            "ordinary_source_training_reference": True,
        },
    )
    _teacher_forced_selection(
        backend,
        review,
        validation,
        validation_data,
        ordinary,
        controller_records,
        mean_records,
        continuation_rows,
        reference_scale,
    )


def _teacher_forced_selection(
    backend: HuggingFaceBackend,
    review: Path,
    validation: list[ExternalItem],
    validation_data: dict[str, Any],
    ordinary: dict[str, Any],
    controller_records: dict[str, Any],
    mean_records: dict[str, Any],
    continuation_rows: list[dict[str, Any]],
    reference_scale: float,
) -> None:
    eta0 = ALPHA_GATE5 / reference_scale
    continuations = {
        row["item_id"]: {"careful": row["careful_token_ids"], "direct": row["direct_token_ids"]}
        for row in continuation_rows
        if row["split"] == "validation"
    }
    random_results: dict[str, list[float]] = {}
    random_bank_metadata: dict[str, Any] = {}
    first_stage: dict[str, Any] = {}

    candidate_records = {
        **controller_records,
        **{f"MEAN:{key}": value for key, value in mean_records.items()},
    }
    first_stage_rows: list[dict[str, Any]] = []
    for key, record in candidate_records.items():
        base_key = key.removeprefix("MEAN:")
        location = record["location"]
        layer = int(record["layer"])
        direction = np.load(ROOT / record["direction_path"], allow_pickle=False)
        direction = np.asarray(direction, dtype=np.float64)
        paired_mean = np.load(
            ROOT / mean_records[base_key]["direction_path"], allow_pickle=False
        ).astype(np.float64)
        orthogonal_basis = (
            [paired_mean]
            if record["constructor"] == "RFM_AGOP"
            else [
                np.load(ROOT / controller_records[base_key]["direction_path"], allow_pickle=False)
                .astype(np.float64)
            ]
        )
        random_bank = orthogonal_random_bank(
            direction,
            additional_basis=orthogonal_basis,
            seeds=[stable_seed("GATE6-RANDOM", key, index) for index in range(4)],
        )
        random_results[key] = []
        ordinary_values = np.stack(
            [
                ordinary["train"][train_id][location][layer]
                for train_id in ordinary["train"]
            ]
        )

        def symmetric_first_stage(
            candidate: np.ndarray,
            *,
            bound_location: str = location,
            bound_layer: int = layer,
            bound_ordinary_values: np.ndarray = ordinary_values,
        ) -> tuple[float, int, int]:
            contributions: list[float] = []
            corrupt = 0
            delta = standardized_budget(candidate, bound_ordinary_values, eta0, 1)
            delta_tensor = backend.torch.tensor(
                delta, dtype=backend.torch.float32, device=backend.device
            ).view(1, 1, -1)
            for item in validation:
                careful_item = model_item(item, SYSTEM_CAREFUL)
                direct_item = model_item(item, SYSTEM_DIRECT)
                careful_tokens = continuations[item.item_id]["careful"]
                direct_tokens = continuations[item.item_id]["direct"]
                _, careful_meta = capture_layers(
                    backend,
                    careful_item,
                    (bound_layer,),
                    careful_tokens if bound_location == "EXECUTION_BOUNDARY" else None,
                )
                _, direct_meta = capture_layers(
                    backend,
                    direct_item,
                    (bound_layer,),
                    direct_tokens if bound_location == "EXECUTION_BOUNDARY" else None,
                )
                target_careful = (
                    careful_meta["target_position"]
                    if bound_location == "EXECUTION_BOUNDARY"
                    else None
                )
                target_direct = (
                    direct_meta["target_position"]
                    if bound_location == "EXECUTION_BOUNDARY"
                    else None
                )
                plus_careful, _ = teacher_forced_loglik(
                    backend,
                    careful_item,
                    careful_tokens,
                    {bound_layer: delta_tensor},
                    target_careful,
                )
                plus_direct, _ = teacher_forced_loglik(
                    backend,
                    direct_item,
                    direct_tokens,
                    {bound_layer: delta_tensor},
                    target_direct,
                )
                minus_careful, _ = teacher_forced_loglik(
                    backend,
                    careful_item,
                    careful_tokens,
                    {bound_layer: -delta_tensor},
                    target_careful,
                )
                minus_direct, _ = teacher_forced_loglik(
                    backend,
                    direct_item,
                    direct_tokens,
                    {bound_layer: -delta_tensor},
                    target_direct,
                )
                contribution = symmetric_first_stage_contributions(
                    [plus_careful], [plus_direct], [minus_careful], [minus_direct]
                )[0]
                if not np.isfinite(contribution):
                    corrupt += 1
                contributions.append(float(contribution))
            values = np.asarray(contributions, dtype=np.float64)
            return float(np.mean(values)), int(np.sum(values > 0)), corrupt

        f_value, positive_count, corrupt_count = symmetric_first_stage(direction)
        first_stage[key] = {
            "constructor": record["constructor"],
            "location": location,
            "layer": layer,
            "F": f_value,
            "positive_count": positive_count,
            "corrupt_count": corrupt_count,
            "readout": record["readout"],
        }
        for random_name, random_direction in random_bank.items():
            random_f, _positive, random_corrupt = symmetric_first_stage(random_direction)
            if random_corrupt:
                raise RuntimeError(f"Gate-6 random first-stage corruption for {key}:{random_name}")
            random_results[key].append(random_f)
        random_metadata = {
            name: {
                "seed_namespace": "GATE6-RANDOM-BANK",
                "vector_hash": vector_sha256(value),
            }
            for name, value in random_bank.items()
        }
        random_bank_metadata[key] = random_metadata
        record["random_bank"] = random_metadata
        for name, value in random_bank.items():
            path = (
                review
                / "RANDOM_DIRECTIONS"
                / location
                / record["constructor"]
                / f"L{layer}_{name}.npy"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, value)
    for key, values in first_stage.items():
        random_mean = float(np.mean(random_results[key]))
        random_max = float(np.max(random_results[key]))
        values["random_mean_F"] = random_mean
        values["random_max_F"] = random_max
        values["readout_pass"] = bool(
            values["readout"]["auroc"] >= 0.80
            and values["readout"]["positive_gap_fraction"] * len(validation) >= 24
        )
        values["pass"] = bool(
            values["readout_pass"]
            and values["F"] > 0
            and values["positive_count"] >= 22
            and values["F"] >= random_mean + 0.01
            and values["F"] > random_max
            and values["corrupt_count"] == 0
        )
        first_stage_rows.append(
            {
                "key": key,
                **values,
                "random_F_values": random_results[key],
            }
        )
    selected = _select_controllers(controller_records, mean_records, first_stage)
    selected["eta0"] = eta0
    selected["reference_scale_layer17"] = reference_scale
    selected["gate4_direction_sha256"] = GATE4_DIRECTION_HASH
    write_json(review / "FIRST_STAGE_RESULTS.json", first_stage)
    write_json(review / "CONTROLLER_SELECTION.json", selected)
    write_json(review / "CONTROLLERS_RAW.json", controller_records)
    write_json(review / "MEAN_CONTROLLERS_RAW.json", mean_records)
    write_json(review / "RANDOM_CONTROLLER_RESULTS.json", random_results)
    write_json(review / "RANDOM_BANK_METADATA.json", random_bank_metadata)
    _write_csv(review / "TEACHER_FORCED_FIRST_STAGE.csv", first_stage_rows)
    if selected.get("best_single"):
        _engineering_gate(backend, review, validation[0], selected)


def _select_controllers(
    records: dict[str, Any], mean_records: dict[str, Any], first_stage: dict[str, Any]
) -> dict[str, Any]:
    passing = [
        (key, values)
        for key, values in first_stage.items()
        if key in records and records[key]["constructor"] == "RFM_AGOP" and values["pass"]
    ]
    passing.sort(key=lambda value: (-value[1]["F"], value[0]))
    best = passing[0] if passing else None
    layers = sorted(int(records[key]["layer"]) for key, _values in passing)
    selected = {
        "source_only_passes": [
            {"key": key, **values} for key, values in passing
        ],
        "best_single": {"key": best[0]} if best else None,
        "multilayer_keys": [
            key
            for key, record in records.items()
            if int(record["layer"]) in layers and first_stage[key]["pass"]
        ],
        "selected_source": records[best[0]]["location"] if best else None,
        "eta_ladder": ["eta0", "2eta0"],
        "eta_selected": "eta0",
        "selection_rule": (
            "RFM-only source readout and symmetric teacher-forced first-stage "
            "gates; lexical tie-break"
        ),
        "source_outcome_labels_used": False,
        "paired_mean_baseline_available": bool(mean_records),
    }
    return selected


def _engineering_gate(
    backend: HuggingFaceBackend,
    review: Path,
    item: ExternalItem,
    selection: dict[str, Any],
) -> None:
    """Run the pre-evaluation identity, scope, and lifecycle checks."""

    deltas = selected_delta_maps(review, load_vectors(review))
    delta_map = deltas["BEST_SINGLE_RFM_PLUS"]
    model_row = model_item(item)
    seed = stable_seed("GATE6-ENGINEERING", item.item_id)
    baseline = backend.generate_reasoning(model_row, sampling_seed=seed, max_new_tokens=32)
    zero = {layer: np.zeros_like(value, dtype=np.float64) for layer, value in delta_map.items()}
    prompt_ids, _rendered, _hash = prompt_tokens(backend, model_row)
    torch = backend.torch
    zero_tensors = {
        layer: torch.zeros((1, 1, backend.hidden_size), dtype=torch.float32, device=backend.device)
        for layer in zero
    }
    with Gate6HookTrace(
        layers={layer: backend.layer_module(layer) for layer in zero_tensors},
        deltas=zero_tensors,
        target_positions=[len(prompt_ids) - 1],
    ) as zero_trace:
        zero_output = backend.generate_reasoning(model_row, sampling_seed=seed, max_new_tokens=32)
    cleanup = backend.generate_reasoning(model_row, sampling_seed=seed, max_new_tokens=32)
    shift_traces: dict[str, dict[str, Any]] = {}
    for condition, condition_delta_map in deltas.items():
        nonzero_tensors = {
            layer: torch.tensor(value, dtype=torch.float32, device=backend.device).view(1, 1, -1)
            for layer, value in condition_delta_map.items()
        }
        with Gate6HookTrace(
            layers={layer: backend.layer_module(layer) for layer in nonzero_tensors},
            deltas=nonzero_tensors,
            target_positions=[len(prompt_ids) - 1],
        ) as shift_trace:
            _ = backend.generate_reasoning(
                model_row,
                sampling_seed=seed,
                max_new_tokens=8,
                intervention_metadata={
                    "gate6_phase": "ENGINEERING",
                    "condition": condition,
                    "intervention_duration": "sustained_current_token",
                    "intervention_layers": sorted(condition_delta_map),
                    "controller_source": "engineering_identity_check",
                },
            )
        shift_traces[condition] = shift_trace.metadata()
    applications = [
        entry
        for trace in shift_traces.values()
        for entry in trace["applications"]
    ]
    records = json.loads((review / "CONTROLLERS_RAW.json").read_text(encoding="utf-8"))
    selection_records = json.loads(
        (review / "MEAN_CONTROLLERS_RAW.json").read_text(encoding="utf-8")
    )
    selection_data = json.loads((review / "CONTROLLER_SELECTION.json").read_text(encoding="utf-8"))
    scales = {key: float(value["scale"]) for key, value in records.items()}
    scales.update(
        {f"MEAN:{key}": float(value["scale"]) for key, value in selection_records.items()}
    )
    energy_errors: dict[str, float] = {}
    eta0 = float(selection_data["eta0"])

    def record_layer(key: str) -> int:
        record_key = key.removeprefix("MEAN:")
        source = selection_records if key.startswith("MEAN:") else records
        return int(source[record_key]["layer"])

    for condition, condition_delta_map in deltas.items():
        normalized_energy = 0.0
        if condition == "BEST_SINGLE_RFM_PLUS":
            scale_keys = [selection_data["best_single"]["key"]]
        elif condition == "MULTILAYER_MEAN_PLUS":
            scale_keys = [f"MEAN:{key}" for key in selection_data["multilayer_keys"]]
        else:
            scale_keys = list(selection_data["multilayer_keys"])
        for layer, value in condition_delta_map.items():
            key = next(key for key in scale_keys if record_layer(key) == layer)
            normalized_energy += (float(np.linalg.norm(value)) / scales[key]) ** 2
        energy_errors[condition] = abs(normalized_energy - eta0**2)
    checks = {
        "alpha_zero_token_identity": baseline.metadata["generated_token_ids"]
        == zero_output.metadata["generated_token_ids"],
        "hook_cleanup_after_identity": baseline.metadata["generated_token_ids"]
        == cleanup.metadata["generated_token_ids"],
        "forward_count_positive": zero_trace.forward_count > 0,
        "one_application_per_layer_forward": all(
            count == zero_trace.forward_count for count in zero_trace.forward_counts.values()
        ),
        "exact_shift_finite": bool(applications)
        and all(
            np.isfinite(entry["shift_error"]) and entry["shift_error"] < 1e-4
            for entry in applications
        ),
        "non_current_scope": bool(applications)
        and all(abs(entry["non_current_change"]) < 1e-6 for entry in applications),
        "all_selected_conditions_shifted": set(deltas).issubset(shift_traces),
        "distributed_energy_matching": all(error < 1e-4 for error in energy_errors.values()),
    }
    payload = {
        "checks": checks,
        "pass": all(checks.values()),
        "layers": sorted(delta_map),
        "seed": seed,
        "zero_trace": zero_trace.metadata(),
        "shift_traces": shift_traces,
        "normalized_energy_errors": energy_errors,
        "scientific_outcomes_collected": False,
    }
    write_json(review / "ENGINEERING_CHECKS.json", payload)
    if not payload["pass"]:
        raise RuntimeError("GATE6_INTERVENTION_ENGINE_FAILURE")


def read_completed(path: Path) -> set[tuple[str, str, int, str]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed.add(
                    (
                        str(row["phase"]),
                        str(row["item_id"]),
                        str(row["condition"]),
                        int(row["rollout_index"]),
                    )
                )
    return completed


def load_vectors(review: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    selected = json.loads((review / "CONTROLLER_SELECTION.json").read_text(encoding="utf-8"))
    records = json.loads((review / "CONTROLLERS_RAW.json").read_text(encoding="utf-8"))
    mean_records = json.loads((review / "MEAN_CONTROLLERS_RAW.json").read_text(encoding="utf-8"))
    for key in selected.get("multilayer_keys", []):
        record = records[key]
        values[key] = np.load(ROOT / record["direction_path"], allow_pickle=False).astype(
            np.float64
        )
        values[f"MEAN:{key}"] = np.load(
            ROOT / mean_records[key]["direction_path"], allow_pickle=False
        ).astype(np.float64)
    for path in (review / "DIRECTIONS").rglob("L*.npy"):
        location = path.parent.name
        layer = path.stem.removeprefix("L")
        values[f"{location}:L{layer}"] = np.load(path, allow_pickle=False).astype(np.float64)
    for path in (review / "RANDOM_DIRECTIONS").rglob("*.npy"):
        constructor = path.parent.name
        location = path.parent.parent.name
        values[f"RANDOM:{location}:{constructor}:{path.stem}"] = np.load(
            path, allow_pickle=False
        ).astype(np.float64)
    return values


def selected_delta_maps(
    review: Path, vectors: dict[str, np.ndarray]
) -> dict[str, dict[int, np.ndarray]]:
    selection = json.loads((review / "CONTROLLER_SELECTION.json").read_text(encoding="utf-8"))
    records = json.loads((review / "CONTROLLERS_RAW.json").read_text(encoding="utf-8"))
    mean_records = json.loads((review / "MEAN_CONTROLLERS_RAW.json").read_text(encoding="utf-8"))
    if not selection.get("best_single"):
        raise RuntimeError("Gate-6 has no source-only passing controller")
    best_key = selection["best_single"]["key"]
    best_record = records[best_key]
    scales = {key: float(value["scale"]) for key, value in records.items()}
    eta = float(selection.get("eta0", 1.0))
    single = {
        int(best_record["layer"]): vectors[best_key] * scales[best_key] * eta
    }
    multi_keys = selection.get("multilayer_keys", [])
    layer_count = max(1, len(multi_keys))
    multi_rfm = {
        int(records[key]["layer"]): vectors[key] * scales[key] * eta / np.sqrt(layer_count)
        for key in multi_keys
    }
    multi_mean = (
        {
            int(records[key]["layer"]): vectors[f"MEAN:{key}"]
            * float(mean_records[key]["scale"])
            * eta
            / np.sqrt(layer_count)
            for key in multi_keys
        }
        if multi_keys
        else {}
    )
    output = {
        "BEST_SINGLE_RFM_PLUS": single,
        "MULTILAYER_MEAN_PLUS": multi_mean,
        "MULTILAYER_RFM_PLUS": multi_rfm,
        "MULTILAYER_RFM_MINUS": {layer: -value for layer, value in multi_rfm.items()},
    }
    for index in range(4):
        output[f"MULTILAYER_RANDOM_R{index}"] = {
            int(records[key]["layer"]): vectors[
                f"RANDOM:{records[key]['location']}:{records[key]['constructor']}"
                f":L{records[key]['layer']}_R{index}"
            ]
            * scales[key]
            * eta
            / np.sqrt(layer_count)
            for key in multi_keys
        }
    return output


def condition_context(
    backend: HuggingFaceBackend,
    item: ExternalItem,
    condition: str,
    delta_map: dict[int, np.ndarray] | None,
) -> tuple[Any, dict[str, Any]]:
    system = None
    if condition == "TEXTUAL_CAREFUL_REFERENCE":
        system = SYSTEM_CAREFUL
    if condition == "TEXTUAL_DIRECT_REFERENCE":
        system = SYSTEM_DIRECT
    model_row = model_item(item, system)
    prompt_ids, _rendered, _hash = prompt_tokens(backend, model_row)
    if not delta_map:
        return nullcontext(), {
            "item": model_row,
            "prompt_length": len(prompt_ids),
            "system_prompt": system,
        }
    torch = backend.torch
    deltas = {
        layer: torch.tensor(value, dtype=torch.float32, device=backend.device).view(1, 1, -1)
        for layer, value in delta_map.items()
    }
    context = Gate6HookTrace(
        layers={layer: backend.layer_module(layer) for layer in deltas},
        deltas=deltas,
        target_positions=[len(prompt_ids) - 1],
    )
    return context, {"item": model_row, "prompt_length": len(prompt_ids), "system_prompt": system}


def execute_phase(backend: HuggingFaceBackend, review: Path, phase: str, manifest: Path) -> None:
    require_remote_hf_execution(f"Gate 6 {phase} inference")
    items = load_external(manifest)
    n_items, conditions, rollouts = FRESH_PHASES[phase]
    if len(items) != n_items:
        raise RuntimeError(f"{phase} expected {n_items} items, found {len(items)}")
    schedule = [
        {
            "phase": phase,
            "item_id": item.item_id,
            "condition": condition,
            "rollout_index": rollout,
            "seed": manipulation_seed(item.item_id)
            if len(rollouts) == 1
            else evaluation_seed(item.item_id, condition, rollout),
            "seed_regime": "MATCHED_COUPLING_SECONDARY"
            if len(rollouts) == 1
            else "INDEPENDENT_PRIMARY",
        }
        for item in items
        for condition in conditions
        for rollout in rollouts
    ]
    schedule_path = review / f"{phase}_SCHEDULE.json"
    if schedule_path.exists() and json.loads(schedule_path.read_text(encoding="utf-8")) != schedule:
        raise RuntimeError(f"frozen {phase} schedule differs from existing schedule")
    write_json(schedule_path, schedule)
    journal = review / "journal.jsonl"
    completed = read_completed(journal)
    vectors = load_vectors(review)
    deltas = selected_delta_maps(review, vectors)
    by_id = {item.item_id: item for item in items}
    for row in schedule:
        key = (phase, row["item_id"], row["condition"], row["rollout_index"])
        if key in completed:
            continue
        condition = row["condition"]
        if condition in {"BASELINE", "TEXTUAL_CAREFUL_REFERENCE", "TEXTUAL_DIRECT_REFERENCE"}:
            delta_map = None
        else:
            delta_map = deltas[condition]
        context, context_meta = condition_context(
            backend, by_id[row["item_id"]], condition, delta_map
        )
        started = time.perf_counter()
        try:
            with context as trace:
                output = backend.generate_reasoning(
                    context_meta["item"],
                    sampling_seed=int(row["seed"]),
                    max_new_tokens=MAX_NEW_TOKENS,
                    intervention_metadata={
                        "gate6_phase": phase,
                        "condition": condition,
                        "intervention_duration": "sustained_current_token" if delta_map else "none",
                        "intervention_layers": sorted(delta_map) if delta_map else [],
                        "controller_source": "source_only_rfm_agop" if delta_map else "none",
                        "alpha_gate5_reference": ALPHA_GATE5,
                    },
                )
            elapsed = time.perf_counter() - started
            output_meta = dict(output.metadata)
            if delta_map:
                output_meta["sustained_hook_trace"] = trace.metadata()
            token_count = int(output_meta.get("generated_token_count", 0))
            scored = score_external_response(
                by_id[row["item_id"]],
                output.raw_output,
                rollout_seed=int(row["seed"]),
                truncated=token_count >= MAX_NEW_TOKENS,
                token_count=token_count,
                metadata={
                    "phase": phase,
                    "condition": condition,
                    "rollout_index": row["rollout_index"],
                    "generation_seconds": elapsed,
                    "stop_metadata": output_meta,
                },
            )
            record = {
                **row,
                "status": scored.status.value.replace("TRUNCATED_THINKING", "TRUNCATED"),
                "correct": bool(scored.correct),
                "parsed_answer": scored.parsed_answer,
                "reference_answer": by_id[row["item_id"]].reference_answer,
                "raw_output": scored.raw_output,
                "generated_token_ids": output_meta.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "prompt_hash": by_id[row["item_id"]].prompt_hash,
                "rendered_prompt_hash": output_meta.get("rendered_prompt_hash"),
                "source_revision": DATASET_REVISION,
                "evaluator": by_id[row["item_id"]].evaluator,
                "metadata": scored.metadata,
                "elapsed_seconds": elapsed,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
        except RuntimeError as exc:
            record = {
                **row,
                "status": "RUNTIME_ERROR",
                "correct": False,
                "raw_output": "",
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        append_jsonl(journal, record)
        if record["status"] == "RUNTIME_ERROR":
            raise RuntimeError(f"Gate-6 runtime failure for {key}: {record['error']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("SOURCE", "CONTROLLER_MANIPULATION", "CONTROLLER_EVALUATION"),
        required=True,
    )
    parser.add_argument("--model-path")
    parser.add_argument(
        "--review-dir", type=Path, default=ROOT / "review" / "gate6_layer_source_rfm_atlas"
    )
    parser.add_argument("--source-validation", type=Path)
    parser.add_argument("--manipulation-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    args = parser.parse_args()
    require_remote_hf_execution("Gate 6 real-model execution")
    backend = build_backend(args.model_path)
    review = args.review_dir
    if args.phase == "SOURCE":
        validation = args.source_validation or review / "SOURCE_VALIDATION.json"
        source_pipeline(backend, review, validation)
        selection = json.loads((review / "CONTROLLER_SELECTION.json").read_text(encoding="utf-8"))
        write_json(
            review / "SOURCE_PHASE_DECISION.json",
            {
                "readout_and_first_stage_passes": len(selection.get("source_only_passes", [])),
                "individual_rfm_first_stage_pass": bool(selection.get("source_only_passes")),
                "multilayer_rfm_available": len(selection.get("multilayer_keys", [])) >= 2,
                "semantic_outcomes_used_for_selection": False,
                "continue_to_manipulation": len(selection.get("multilayer_keys", [])) >= 2,
            },
        )
    elif args.phase == "CONTROLLER_MANIPULATION":
        execute_phase(
            backend,
            review,
            "CONTROLLER_MANIPULATION",
            args.manipulation_manifest or review / "CONTROLLER_MANIPULATION.json",
        )
    else:
        execute_phase(
            backend,
            review,
            "CONTROLLER_EVALUATION",
            args.evaluation_manifest or review / "CONTROLLER_EVALUATION.json",
        )
    write_json(
        review / "RUN_METADATA.json",
        {
            "phase": args.phase,
            "model": MODEL,
            "revision": MODEL_REVISION,
            "layers": list(LAYERS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "source_commit": stable_digest("GATE6-SOURCE-COMMIT", Path.cwd().name),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
