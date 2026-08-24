#!/usr/bin/env python3
"""Remote Q2 source qualification, geometry capture, and common-panel runner."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from run_gate6_2_first_stage_repair import (  # noqa: E402
    build_backend,
    load_external,
    model_item,
    prompt_tokens,
)
from run_gate11_domain_conditioned_control import DiagnosticHooks, forward  # noqa: E402

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    canonicalize_semantic_value,
    evaluate_external_answer_v3,
    extract_final_commitment,
)
from epistemic_geometry.experiments.gate6 import (  # noqa: E402
    paired_mean_direction,
    unit_vector,
)
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.experiments.q2_controller_heldout import (  # noqa: E402
    BASELINE,
    CONDITIONS,
    CONTROLLER_IDS,
    DELTA_NORM,
    EXECUTION_TEACHER_TEXT,
    EXPERIMENT_ID,
    LAYER,
    LOCATIONS,
    MODEL,
    MODEL_REVISION,
    SOURCE_AXES,
    build_null_bank,
    expand_meaningful_bank,
    qualification_decision,
    validate_bank,
)
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from epistemic_geometry.research.reliability import CrashSafeJournal  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402

REVIEW = ROOT / "review/q2_controller_heldout_geometry"
MAX_NEW_TOKENS = 4096


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def candidate_lock(review: Path) -> dict[str, Any]:
    lock = read_json(review / "CANDIDATE_PROTOCOL_LOCK.json")
    if lock["status"] != "FROZEN_PRE_QUALIFICATION":
        raise RuntimeError("Q2 candidate protocol is not frozen")
    if lock["model"]["id"] != MODEL or lock["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("Q2 model differs from the candidate lock")
    return lock


def final_lock(review: Path, source_commit: str) -> dict[str, Any]:
    lock = read_json(review / "PROTOCOL_LOCK.json")
    if lock["status"] != "FROZEN_PRE_COMMON_PANEL":
        raise RuntimeError("Q2 final bank/geometry lock is not frozen")
    if source_commit != lock["experiment_source_commit"] or git_head() != source_commit:
        raise RuntimeError("Q2 execution source commit mismatch")
    if sha256(review / "PROTOCOL_LOCK.json") != read_json(
        review / "EXPERIMENT_SOURCE_COMMIT.json"
    )["protocol_lock_sha256"]:
        raise RuntimeError("Q2 protocol/source binding mismatch")
    return lock


def _source_instruction(axis_id: str, polarity: str) -> str:
    axis = next(axis for axis in SOURCE_AXES if axis.axis_id == axis_id)
    if polarity == "POSITIVE":
        return axis.positive_instruction
    if polarity == "NEGATIVE":
        return axis.negative_instruction
    raise ValueError("unknown source polarity")


def _capture_source_pair(backend: Any, item: Any, system: str) -> dict[str, np.ndarray]:
    row = model_item(item, system)
    prompt_ids, _rendered, _prompt_hash = prompt_tokens(backend, row)
    teacher_ids = backend.tokenizer.encode(EXECUTION_TEACHER_TEXT, add_special_tokens=False)
    full_ids = prompt_ids + [int(token) for token in teacher_ids]
    torch = backend.torch
    ids = torch.tensor([full_ids], dtype=torch.long, device=backend.device)
    attention = torch.ones_like(ids)
    positions = {
        "PROMPT_BOUNDARY": len(prompt_ids) - 1,
        "EXECUTION_BOUNDARY": len(full_ids) - 1,
    }
    captures: dict[str, np.ndarray] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        for location, position in positions.items():
            captures[location] = hidden[0, position, :].detach().float().cpu().numpy().copy()

    handle = backend.layer_module(LAYER).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            backend._forward(  # noqa: SLF001
                backend.model,
                {
                    "input_ids": ids,
                    "attention_mask": attention,
                    "use_cache": False,
                    "return_dict": True,
                },
                "prefill",
            )
    finally:
        handle.remove()
    if set(captures) != set(LOCATIONS):
        raise RuntimeError("Q2 source capture missed a frozen boundary")
    return captures


def _mechanical_parse(raw_output: str, token_count: int) -> dict[str, Any]:
    commitment = extract_final_commitment(
        raw_output, truncated=token_count >= MAX_NEW_TOKENS
    )
    if not commitment.valid or commitment.payload is None:
        return {
            "commitment_valid": False,
            "semantic_evaluable": False,
            "canonical_value": None,
            "parse_reason": commitment.failure_reason,
        }
    canonical = canonicalize_semantic_value(commitment.payload)
    return {
        "commitment_valid": True,
        "semantic_evaluable": True,
        "canonical_value": json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "parse_reason": None,
    }


def source_qualification(backend: Any, review: Path, source_commit: str) -> None:
    candidate_lock(review)
    construction = load_external(review / "SOURCE_CONSTRUCTION_MANIFEST.json")
    validation = load_external(review / "SOURCE_VALIDATION_MANIFEST.json")
    arrays_path = review / "SOURCE_ACTIVATIONS.npz"
    if not arrays_path.exists():
        arrays: dict[str, np.ndarray] = {}
        for split_name, items in (("construction", construction), ("validation", validation)):
            for axis in SOURCE_AXES:
                for polarity in ("POSITIVE", "NEGATIVE"):
                    rows = [
                        _capture_source_pair(
                            backend, item, _source_instruction(axis.axis_id, polarity)
                        )
                        for item in items
                    ]
                    for location in LOCATIONS:
                        arrays[f"{split_name}__{axis.axis_id}__{polarity}__{location}"] = (
                            np.stack([row[location] for row in rows]).astype(np.float32)
                        )
        np.savez_compressed(arrays_path, **arrays)

    journal = CrashSafeJournal(
        review / "source_behavior_journal.jsonl",
        identity={
            "experiment_id": EXPERIMENT_ID,
            "phase": "SOURCE_BEHAVIOR_QUALIFICATION",
            "source_commit": source_commit,
        },
        key_fields=("item_id", "axis_id", "polarity", "rollout_index"),
    )
    schedule = read_json(review / "SOURCE_BEHAVIOR_SCHEDULE.json")
    items = {item.item_id: item for item in validation}
    for row in schedule:
        key = (
            row["item_id"],
            row["axis_id"],
            row["polarity"],
            row["rollout_index"],
        )
        if key in journal.rows:
            continue
        item = items[row["item_id"]]
        output = backend.generate_reasoning(
            model_item(item, _source_instruction(row["axis_id"], row["polarity"])),
            sampling_seed=int(row["seed"]),
            max_new_tokens=MAX_NEW_TOKENS,
            intervention_metadata={
                "experiment_id": EXPERIMENT_ID,
                "phase": "SOURCE_BEHAVIOR_QUALIFICATION",
                "axis_id": row["axis_id"],
                "polarity": row["polarity"],
                "correctness_not_evaluated": True,
            },
        )
        metadata = dict(output.metadata)
        token_count = int(metadata.get("generated_token_count", 0))
        journal.append(
            {
                **row,
                **_mechanical_parse(output.raw_output, token_count),
                "raw_output": output.raw_output,
                "generated_token_ids": metadata.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "parser_version": PARSER_VERSION,
                "source_commit": source_commit,
                "correctness_evaluated": False,
            }
        )
    finalize_source_bank(review)
    print(json.dumps({"phase": "source_qualification", "rows": len(journal.rows)}), flush=True)


def _journal_rows(path: Path, identity: dict[str, Any], keys: tuple[str, ...]) -> list[dict]:
    return list(CrashSafeJournal(path, identity=identity, key_fields=keys).rows.values())


def _disagreement_metrics(rows: list[dict[str, Any]], axis_id: str) -> dict[str, float]:
    selected = [row for row in rows if row["axis_id"] == axis_id]
    by_key = {
        (row["item_id"], row["polarity"], row["rollout_index"]): row for row in selected
    }
    item_ids = sorted({row["item_id"] for row in selected})
    cross: list[float] = []
    within_positive: list[float] = []
    within_negative: list[float] = []
    for item_id in item_ids:
        pos = [by_key[(item_id, "POSITIVE", rollout)]["canonical_value"] for rollout in (0, 1)]
        neg = [by_key[(item_id, "NEGATIVE", rollout)]["canonical_value"] for rollout in (0, 1)]
        cross.extend(float(left != right) for left in pos for right in neg)
        within_positive.append(float(pos[0] != pos[1]))
        within_negative.append(float(neg[0] != neg[1]))
    within = 0.5 * (float(np.mean(within_positive)) + float(np.mean(within_negative)))
    return {
        "cross_disagreement": float(np.mean(cross)),
        "within_disagreement": within,
        "excess_disagreement": float(np.mean(cross)) - within,
    }


def finalize_source_bank(review: Path) -> None:
    source_commit = candidate_lock(review)["source_commit_at_preparation"]
    # Remote-safe source bundles may be executed at a later commit with identical frozen files.
    source_rows = _journal_rows(
        review / "source_behavior_journal.jsonl",
        identity={
            "experiment_id": EXPERIMENT_ID,
            "phase": "SOURCE_BEHAVIOR_QUALIFICATION",
            "source_commit": git_head(),
        },
        keys=("item_id", "axis_id", "polarity", "rollout_index"),
    )
    del source_commit
    archive = np.load(review / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    base: dict[tuple[str, str], np.ndarray] = {}
    pairs: dict[tuple[str, str], np.ndarray] = {}
    source_records: dict[str, Any] = {}
    vector_dir = review / "CONTROLLER_VECTORS"
    vector_dir.mkdir(parents=True, exist_ok=True)
    for axis in SOURCE_AXES:
        selected = [row for row in source_rows if row["axis_id"] == axis.axis_id]
        pos_rows = [row for row in selected if row["polarity"] == "POSITIVE"]
        neg_rows = [row for row in selected if row["polarity"] == "NEGATIVE"]
        pos_tokens = np.asarray([row["generated_token_count"] for row in pos_rows])
        neg_tokens = np.asarray([row["generated_token_count"] for row in neg_rows])
        record: dict[str, Any] = {
            "positive_commitment_validity": float(
                np.mean([row["commitment_valid"] for row in pos_rows])
            ),
            "negative_commitment_validity": float(
                np.mean([row["commitment_valid"] for row in neg_rows])
            ),
            "positive_semantic_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in pos_rows])
            ),
            "negative_semantic_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in neg_rows])
            ),
            "positive_negative_mean_token_ratio": float(np.mean(pos_tokens) / np.mean(neg_tokens)),
            "positive_minus_negative_median_tokens": float(
                np.median(pos_tokens) - np.median(neg_tokens)
            ),
            "activation": {},
            **_disagreement_metrics(source_rows, axis.axis_id),
        }
        for location in LOCATIONS:
            construct_pos = archive[f"construction__{axis.axis_id}__POSITIVE__{location}"]
            construct_neg = archive[f"construction__{axis.axis_id}__NEGATIVE__{location}"]
            direction, raw_gap, _raw = paired_mean_direction(construct_pos, construct_neg)
            validation_pos = archive[f"validation__{axis.axis_id}__POSITIVE__{location}"]
            validation_neg = archive[f"validation__{axis.axis_id}__NEGATIVE__{location}"]
            gaps = (validation_pos - validation_neg) @ direction
            pooled_projection = np.concatenate(
                (validation_pos @ direction, validation_neg @ direction)
            )
            scale = float(np.std(pooled_projection, ddof=1))
            base[(axis.axis_id, location)] = direction
            pairs[(axis.axis_id, location)] = construct_pos - construct_neg
            record["activation"][location] = {
                "construction_raw_mean_gap": raw_gap,
                "validation_mean_gap": float(np.mean(gaps)),
                "validation_projection_sd": scale,
                "standardized_mean_gap": float(np.mean(gaps) / scale),
                "positive_gap_fraction": float(np.mean(gaps > 0)),
                "canonical_float64_vector_sha256": vector_sha256(direction),
            }
        source_records[axis.axis_id] = record
    bank = expand_meaningful_bank(base)
    nulls, null_metadata = build_null_bank(base, pairs)
    bank.update(nulls)
    bank_checks = validate_bank(bank)
    for name, vector in bank.items():
        np.save(vector_dir / f"{name}.npy", unit_vector(vector).astype(np.float64))
    write_json(review / "SOURCE_QUALIFICATION.json", source_records)
    write_json(review / "NULL_BANK.json", null_metadata)
    write_json(review / "BANK_VALIDATION.json", bank_checks)
    write_json(
        review / "CONTROLLER_BANK.json",
        {
            "controller_ids": list(CONTROLLER_IDS),
            "vectors": {
                name: {
                    "path": str((vector_dir / f"{name}.npy").relative_to(ROOT)),
                    "canonical_float64_vector_sha256": vector_sha256(vector),
                    "delta_norm": DELTA_NORM,
                }
                for name, vector in bank.items()
            },
            "source_qualification_file": "SOURCE_QUALIFICATION.json",
            "bank_validation_file": "BANK_VALIDATION.json",
            "correctness_used": False,
        },
    )


def load_bank(review: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    metadata = read_json(review / "CONTROLLER_BANK.json")
    vectors: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for name in CONTROLLER_IDS:
        record = metadata["vectors"][name]
        values = np.load(ROOT / record["path"], allow_pickle=False).astype(np.float64)
        if vector_sha256(values) != record["canonical_float64_vector_sha256"]:
            raise RuntimeError(f"Q2 controller vector hash mismatch: {name}")
        vectors[name] = unit_vector(values)
        hashes[name] = vector_sha256(values)
    return vectors, hashes


def _condition_context(
    backend: Any, item: Any, condition: str, vectors: dict[str, np.ndarray], hashes: dict[str, str]
) -> tuple[Any, Any, dict[str, Any]]:
    row = model_item(item)
    prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
    if condition == BASELINE:
        return nullcontext(), row, {
            "intervention": "none",
            "prompt_length": len(prompt_ids),
            "rendered_prompt_hash_preflight": prompt_hash,
        }
    delta = vectors[condition] * DELTA_NORM
    tensor = backend.torch.tensor(
        delta, dtype=backend.torch.float32, device=backend.device
    ).view(1, 1, -1)
    return (
        Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: tensor},
            target_positions=[len(prompt_ids) - 1],
        ),
        row,
        {
            "intervention": condition,
            "layer": LAYER,
            "duration": "sustained_current_token",
            "scope": "final_prompt_token_then_current_decode_token",
            "vector_hash": hashes[condition],
            "delta_norm": float(np.linalg.norm(delta)),
            "prompt_length": len(prompt_ids),
            "rendered_prompt_hash_preflight": prompt_hash,
        },
    )


def engineering_gate(backend: Any, review: Path) -> None:
    candidate_lock(review)
    vectors, hashes = load_bank(review)
    fixtures = load_external(
        ROOT / "review/gate6_2_first_stage_repair_mean_bridge/MANIPULATION_MANIFEST.json"
    )[:5]
    identity: list[bool] = []
    cleanup: list[bool] = []
    traces: list[dict[str, Any]] = []
    exercised: set[str] = set()
    for index, item in enumerate(fixtures):
        seed = 8_240_000 + index
        row = model_item(item)
        clean = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        prompt_ids, _rendered, _hash = prompt_tokens(backend, row)
        zero = backend.torch.zeros(
            (1, 1, len(next(iter(vectors.values())))),
            dtype=backend.torch.float32,
            device=backend.device,
        )
        with Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: zero},
            target_positions=[len(prompt_ids) - 1],
        ):
            zero_output = backend.generate_reasoning(
                row, sampling_seed=seed, max_new_tokens=16
            )
        identity.append(
            clean.metadata.get("generated_token_ids")
            == zero_output.metadata.get("generated_token_ids")
        )
        conditions = CONTROLLER_IDS if index == 0 else CONTROLLER_IDS[:1]
        for offset, condition in enumerate(conditions):
            context, model_row, _metadata = _condition_context(
                backend, item, condition, vectors, hashes
            )
            with context as trace:
                backend.generate_reasoning(
                    model_row, sampling_seed=seed + 100 + offset, max_new_tokens=16
                )
            traces.append(trace.metadata())
            exercised.add(condition)
        clean_after = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        cleanup.append(
            clean.metadata.get("generated_token_ids")
            == clean_after.metadata.get("generated_token_ids")
        )
    applications = [row for trace in traces for row in trace["applications"]]
    result = {
        "alpha_zero_identity": all(identity),
        "hook_cleanup": all(cleanup),
        "per_forward_exact_shift": bool(applications)
        and max(float(row["relative_shift_error"]) for row in applications) <= 2.0,
        "current_token_scope": bool(applications)
        and max(abs(float(row["non_current_change"])) for row in applications) <= 0.125,
        "one_application_per_forward": sum(trace["forward_count"] for trace in traces)
        == len(applications),
        "cache_safety": any(int(row["sequence_length"]) == 1 for row in applications),
        "all_controllers_exercised": exercised == set(CONTROLLER_IDS),
        "common_delta_norm": max(
            abs(float(np.linalg.norm(vector * DELTA_NORM)) - DELTA_NORM)
            for vector in vectors.values()
        )
        <= 1e-9,
        "controller_hashes": hashes,
    }
    result["pass"] = all(value for key, value in result.items() if key != "controller_hashes")
    result["classification"] = (
        "Q2_ENGINEERING_PASS" if result["pass"] else "Q2_ENGINE_FAILURE"
    )
    write_json(review / "ENGINEERING_CHECKS.json", result)
    if not result["pass"]:
        raise RuntimeError("Q2_ENGINE_FAILURE")
    print(json.dumps({"classification": result["classification"]}), flush=True)


def manipulation_qualification(backend: Any, review: Path, source_commit: str) -> None:
    engineering = read_json(review / "ENGINEERING_CHECKS.json")
    if engineering["classification"] != "Q2_ENGINEERING_PASS":
        raise RuntimeError("Q2 manipulation requires a passed engineering gate")
    vectors, hashes = load_bank(review)
    items = {
        item.item_id: item for item in load_external(review / "MANIPULATION_MANIFEST.json")
    }
    schedule = read_json(review / "MANIPULATION_SCHEDULE.json")
    journal = CrashSafeJournal(
        review / "manipulation_journal.jsonl",
        identity={
            "experiment_id": EXPERIMENT_ID,
            "phase": "CONTROLLER_MANIPULATION_QUALIFICATION",
            "source_commit": source_commit,
        },
        key_fields=("item_id", "condition", "rollout_index"),
    )
    for row in schedule:
        key = (row["item_id"], row["condition"], row["rollout_index"])
        if key in journal.rows:
            continue
        context, model_row, metadata = _condition_context(
            backend, items[row["item_id"]], row["condition"], vectors, hashes
        )
        with context as trace:
            output = backend.generate_reasoning(
                model_row,
                sampling_seed=int(row["seed"]),
                max_new_tokens=MAX_NEW_TOKENS,
                intervention_metadata={
                    "experiment_id": EXPERIMENT_ID,
                    "phase": "CONTROLLER_MANIPULATION_QUALIFICATION",
                    **metadata,
                    "correctness_not_evaluated": True,
                },
            )
        backend_metadata = dict(output.metadata)
        token_count = int(backend_metadata.get("generated_token_count", 0))
        journal.append(
            {
                **row,
                **_mechanical_parse(output.raw_output, token_count),
                "raw_output": output.raw_output,
                "generated_token_ids": backend_metadata.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "condition_metadata": metadata,
                "hook_trace": trace.metadata() if row["condition"] != BASELINE else None,
                "source_commit": source_commit,
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "parser_version": PARSER_VERSION,
                "correctness_evaluated": False,
            }
        )
    finalize_manipulation(review, source_commit)
    print(json.dumps({"phase": "manipulation", "rows": len(journal.rows)}), flush=True)


def finalize_manipulation(review: Path, source_commit: str) -> None:
    rows = _journal_rows(
        review / "manipulation_journal.jsonl",
        identity={
            "experiment_id": EXPERIMENT_ID,
            "phase": "CONTROLLER_MANIPULATION_QUALIFICATION",
            "source_commit": source_commit,
        },
        keys=("item_id", "condition", "rollout_index"),
    )
    lookup = {(row["item_id"], row["condition"]): row for row in rows}
    items = sorted({row["item_id"] for row in rows})
    records: dict[str, Any] = {}
    for name in CONTROLLER_IDS:
        selected = [lookup[(item_id, name)] for item_id in items]
        baseline = [lookup[(item_id, BASELINE)] for item_id in items]
        records[name] = {
            "commitment_validity": float(np.mean([row["commitment_valid"] for row in selected])),
            "semantic_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in selected])
            ),
            "semantic_change_rate": float(
                np.mean(
                    [
                        row["canonical_value"] != base["canonical_value"]
                        for row, base in zip(selected, baseline, strict=True)
                    ]
                )
            ),
            "raw_sequence_change_rate": float(
                np.mean(
                    [
                        row["generated_token_ids"] != base["generated_token_ids"]
                        for row, base in zip(selected, baseline, strict=True)
                    ]
                )
            ),
        }
    decision = qualification_decision(
        read_json(review / "SOURCE_QUALIFICATION.json"),
        records,
        read_json(review / "BANK_VALIDATION.json"),
    )
    write_json(review / "MANIPULATION_QUALIFICATION.json", records)
    write_json(review / "BANK_QUALIFICATION.json", decision)


def _capture_covariance(backend: Any, review: Path) -> None:
    items = load_external(review / "COVARIANCE_MANIFEST.json")
    values: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for item in items:
        row = model_item(item)
        prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
        torch = backend.torch
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=backend.device)
        attention = torch.ones_like(ids)
        captured: list[np.ndarray] = []

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            destination: list[np.ndarray] = captured,
        ) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            destination.append(hidden[0, -1, :].detach().float().cpu().numpy().copy())

        handle = backend.layer_module(LAYER).register_forward_hook(hook)
        try:
            with torch.inference_mode():
                backend._forward(  # noqa: SLF001
                    backend.model,
                    {
                        "input_ids": ids,
                        "attention_mask": attention,
                        "use_cache": False,
                        "return_dict": True,
                    },
                    "prefill",
                )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Q2 covariance capture count mismatch")
        values.append(captured[0])
        metadata.append({"item_id": item.item_id, "prompt_hash": prompt_hash})
    np.savez_compressed(
        review / "COVARIANCE_ACTIVATIONS.npz",
        activations=np.stack(values).astype(np.float32),
    )
    write_json(review / "COVARIANCE_CAPTURE_METADATA.json", metadata)


def _checkpoint_indices(length: int) -> tuple[int, ...]:
    if length < 3:
        raise RuntimeError("Q2 finite-secant continuation is too short")
    return tuple(sorted({0, max(0, length // 3), max(0, (2 * length) // 3), length - 1}))


def _secant_logits(
    backend: Any, item: Any, delta: np.ndarray, continuation: list[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    row = model_item(item)
    prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
    checkpoints = _checkpoint_indices(len(continuation))
    snapshots: list[np.ndarray] = []
    with (
        DiagnosticHooks(backend, (), delta, len(prompt_ids)) as hooks,
        backend.torch.inference_mode(),
    ):
        output = forward(
            backend, prompt_ids, past=None, total_length=len(prompt_ids), phase="prefill"
        )
        hooks.note_forward()
        snapshots.append(output.logits[0, -1, :].detach().float().cpu().numpy().copy())
        past = output.past_key_values
        for token_index, token in enumerate(continuation):
            output = forward(
                backend,
                [int(token)],
                past=past,
                total_length=len(prompt_ids) + token_index + 1,
                phase="decode",
            )
            hooks.note_forward()
            past = output.past_key_values
            if token_index in checkpoints[1:]:
                snapshots.append(
                    output.logits[0, -1, :].detach().float().cpu().numpy().copy()
                )
    if len(snapshots) != 4:
        raise RuntimeError(f"Q2 finite-secant captured {len(snapshots)} checkpoints, expected 4")
    return np.stack(snapshots), {
        "item_id": item.item_id,
        "prompt_hash": prompt_hash,
        "continuation_length": len(continuation),
        "continuation_checkpoint_indices": [-1, *checkpoints[1:]],
        "forward_count": hooks.forward_count,
        "application_count": hooks.application_count,
        "max_relative_shift_error": hooks.max_relative_shift_error,
        "max_noncurrent_change": hooks.max_noncurrent_change,
    }


def geometry_capture(backend: Any, review: Path) -> None:
    if read_json(review / "BANK_QUALIFICATION.json")["classification"] != (
        "Q2_CONTROLLER_BANK_QUALIFIED"
    ):
        raise RuntimeError("Q2_CONTROLLER_BANK_NOT_QUALIFIED")
    vectors, hashes = load_bank(review)
    if not (review / "COVARIANCE_ACTIVATIONS.npz").exists():
        _capture_covariance(backend, review)
    items = load_external(review / "FINITE_SECANT_PROBE_MANIFEST.json")
    continuation = [
        int(value)
        for value in backend.tokenizer.encode(
            EXECUTION_TEACHER_TEXT, add_special_tokens=False
        )
    ]
    raw_dir = review / "finite_secant_logits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in items:
        path = raw_dir / f"{item.item_id}.npz"
        if path.exists():
            records.append(
                {
                    "item_id": item.item_id,
                    "path": str(path.relative_to(review)),
                    "sha256": sha256(path),
                    "resumed": True,
                }
            )
            continue
        arrays: dict[str, np.ndarray] = {}
        item_metadata: dict[str, Any] = {}
        for name in CONTROLLER_IDS:
            logits, metadata = _secant_logits(
                backend, item, vectors[name] * DELTA_NORM, continuation
            )
            # The persisted float16 logits are the exact representation used by M2.
            arrays[name] = logits.astype(np.float16)
            item_metadata[name] = {**metadata, "vector_hash": hashes[name]}
        np.savez_compressed(path, **arrays)
        metadata_path = raw_dir / f"{item.item_id}.json"
        write_json(metadata_path, item_metadata)
        records.append(
            {
                "item_id": item.item_id,
                "path": str(path.relative_to(review)),
                "sha256": sha256(path),
                "metadata_path": str(metadata_path.relative_to(review)),
                "metadata_sha256": sha256(metadata_path),
                "resumed": False,
            }
        )
    write_json(
        review / "FINITE_SECANT_ARCHIVE.json",
        {
            "records": records,
            "controller_ids": list(CONTROLLER_IDS),
            "representation_dtype": "float16 logits; dequantized to float64 for JS",
            "teacher_forced_text": EXECUTION_TEACHER_TEXT,
            "teacher_token_ids": continuation,
            "checkpoint_count": 4,
            "correctness_labels_used": False,
        },
    )
    print(json.dumps({"phase": "geometry_capture", "probes": len(records)}), flush=True)


def preflight(backend: Any, review: Path) -> None:
    vectors, hashes = load_bank(review)
    items = load_external(
        ROOT / "review/gate6_2_first_stage_repair_mean_bridge/MANIPULATION_MANIFEST.json"
    )[:5]
    durations: list[float] = []
    tokens: list[int] = []
    for item in items:
        for condition in CONDITIONS:
            context, row, _metadata = _condition_context(
                backend, item, condition, vectors, hashes
            )
            started = time.monotonic()
            with context:
                output = backend.generate_reasoning(
                    row,
                    sampling_seed=91_000_000 + len(durations),
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            durations.append(time.monotonic() - started)
            tokens.append(int(output.metadata.get("generated_token_count", 0)))
    seconds_per_row = float(np.mean(durations))
    projected_seconds = seconds_per_row * 4080 * 1.20
    result = {
        "fixture_rows": len(durations),
        "mean_seconds_per_row": seconds_per_row,
        "median_seconds_per_row": float(np.median(durations)),
        "mean_tokens": float(np.mean(tokens)),
        "max_tokens": int(max(tokens)),
        "projected_seconds_with_20pct_margin": projected_seconds,
        "projected_hours_with_20pct_margin": projected_seconds / 3600.0,
        "hourly_price_required_for_cost_projection": True,
        "scientific_items_used": False,
    }
    write_json(review / "THROUGHPUT_PREFLIGHT.json", result)
    print(json.dumps(result), flush=True)


def common_panel(backend: Any, review: Path, source_commit: str) -> None:
    final_lock(review, source_commit)
    if read_json(review / "BANK_QUALIFICATION.json")["classification"] != (
        "Q2_CONTROLLER_BANK_QUALIFIED"
    ):
        raise RuntimeError("Q2_CONTROLLER_BANK_NOT_QUALIFIED")
    if read_json(review / "ENGINEERING_CHECKS.json")["classification"] != (
        "Q2_ENGINEERING_PASS"
    ):
        raise RuntimeError("Q2 common panel requires passed engineering")
    vectors, hashes = load_bank(review)
    items = {
        item.item_id: item
        for item in load_external(review / "DEVELOPMENT_PANEL_MANIFEST.json")
    }
    schedule = read_json(review / "COMMON_PANEL_SCHEDULE.json")
    if len(schedule) != 4080:
        raise RuntimeError("Q2 common-panel schedule is not 4,080 rows")
    journal = CrashSafeJournal(
        review / "journal.jsonl",
        identity={
            "experiment_id": EXPERIMENT_ID,
            "phase": "COMMON_PANEL",
            "source_commit": source_commit,
            "protocol_lock_sha256": sha256(review / "PROTOCOL_LOCK.json"),
        },
        key_fields=("item_id", "condition", "rollout_index"),
    )
    for schedule_index, row in enumerate(schedule):
        key = (row["item_id"], row["condition"], row["rollout_index"])
        if key in journal.rows:
            continue
        item = items[row["item_id"]]
        context, model_row, condition_metadata = _condition_context(
            backend, item, row["condition"], vectors, hashes
        )
        started = time.monotonic()
        with context as trace:
            output = backend.generate_reasoning(
                model_row,
                sampling_seed=int(row["seed"]),
                max_new_tokens=MAX_NEW_TOKENS,
                intervention_metadata={
                    "experiment_id": EXPERIMENT_ID,
                    "phase": "COMMON_PANEL",
                    "experiment_source_commit": source_commit,
                    **condition_metadata,
                    "parser_version": PARSER_VERSION,
                    "environment_profile": "CORE_QWEN",
                },
            )
        backend_metadata = dict(output.metadata)
        token_count = int(backend_metadata.get("generated_token_count", 0))
        result = evaluate_external_answer_v3(
            output.raw_output,
            item.reference_answer,
            truncated=token_count >= MAX_NEW_TOKENS,
        )
        journal.append(
            {
                **row,
                "correct": result.correct,
                "commitment_valid": result.commitment_valid,
                "semantic_evaluable": result.semantic_evaluable,
                "canonical_value": result.canonical_value,
                "parsed_answer": result.payload,
                "parse_reason": result.failure_reason,
                "raw_output": output.raw_output,
                "generated_token_ids": backend_metadata.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "reference_answer": item.reference_answer,
                "reference_canonical_type": item.metadata.get("reference_canonical_type"),
                "prompt_hash": item.prompt_hash,
                "condition_metadata": condition_metadata,
                "hook_trace": trace.metadata() if row["condition"] != BASELINE else None,
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "parser_version": PARSER_VERSION,
                "experiment_source_commit": source_commit,
                "schedule_index": schedule_index,
                "elapsed_seconds": time.monotonic() - started,
                "retry_count": 0,
            }
        )
        if len(journal.rows) % 100 == 0:
            print(
                json.dumps({"health": "running", "completed_rows": len(journal.rows)}),
                flush=True,
            )
    write_json(
        review / "COLLECTION_METADATA.json",
        {
            "completed_rows": len(journal.rows),
            "expected_rows": 4080,
            "source_commit": source_commit,
            "protocol_lock_sha256": sha256(review / "PROTOCOL_LOCK.json"),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "source",
            "engineering",
            "manipulation",
            "geometry",
            "preflight",
            "collect",
        ),
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution(f"Q2 {args.mode}")
    review = args.review_dir.resolve()
    backend = build_backend(args.model_path)
    if args.mode == "source":
        source_qualification(backend, review, args.source_commit)
    elif args.mode == "engineering":
        engineering_gate(backend, review)
    elif args.mode == "manipulation":
        manipulation_qualification(backend, review, args.source_commit)
    elif args.mode == "geometry":
        geometry_capture(backend, review)
    elif args.mode == "preflight":
        preflight(backend, review)
    else:
        common_panel(backend, review, args.source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
