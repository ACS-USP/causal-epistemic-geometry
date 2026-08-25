#!/usr/bin/env python3
"""Mechanical executor for the prospectively frozen Q2 V3 protocol.

The scientific choices live in ``review/q2_v3_radial_angular_freeze``.  This
runner only materializes the pinned public rows and executes the frozen phases.
It deliberately has no outcome-dependent controller, dose, or metric choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from run_gate6_2_first_stage_repair import build_backend, model_item, prompt_tokens  # noqa: E402
from run_gate11_domain_conditioned_control import DiagnosticHooks, forward  # noqa: E402

from epistemic_geometry.analysis.q2_geometries import (  # noqa: E402
    finite_secant_geometry,
    fit_whitening,
    flat_geometry,
    whitened_geometry,
)
from epistemic_geometry.benchmarks.external.base import ExternalItem  # noqa: E402
from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    canonicalize_semantic_value,
    evaluate_external_answer_v3,
    extract_final_commitment,
)
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.experiments.gate7 import task_prompt  # noqa: E402
from epistemic_geometry.experiments.q2_v3 import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    EXECUTION_TEACHER_TEXT,
    EXPERIMENT_ID,
    LAYER,
    LOCATIONS,
    MODEL,
    MODEL_REVISION,
    NULL_SEEDS,
    SHELL_TARGETS,
    SHELLS,
    SOURCE_FAMILIES,
    base_direction_id,
    condition_ids,
    meaningful_controller_ids,
    null_controller_ids,
    ordered_id_hash,
)
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from epistemic_geometry.research.reliability import CrashSafeJournal  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402

REVIEW = ROOT / "review/q2_v3_radial_angular_freeze"
MAX_NEW_TOKENS = 4096
EXPERIMENT_SOURCE_COMMIT = "9a748de3706a788f8c6c5a1d12c09489808006e8"
FROZEN_HEAD = "c9292d2baecb41de786912b77c39734855ed46cb"
MATERIALIZED = "Q2_V3_MATERIALIZED_ITEMS.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _assert_frozen(review: Path) -> dict[str, Any]:
    lock = read_json(review / "PROTOCOL_LOCK.json")
    if lock["status"] != "Q2_V3_FROZEN_NOT_RUN":
        raise RuntimeError("Q2_V3_INSTRUMENT_FAILURE: protocol status changed")
    if lock["experiment_source_commit"] != EXPERIMENT_SOURCE_COMMIT:
        raise RuntimeError("Q2_V3_INSTRUMENT_FAILURE: source commit mismatch")
    if lock["M3"] != "EXCLUDED_NOT_QUALIFIED_M3_DERIVATIVE_IDENTITIES_FAILED":
        raise RuntimeError("Q2_V3_INSTRUMENT_FAILURE: M3 exclusion mismatch")
    for name, expected in lock["artifact_hashes"].items():
        if sha256(review / name) != expected:
            raise RuntimeError(f"Q2_V3_INSTRUMENT_FAILURE: frozen hash mismatch: {name}")
    return lock


def _normalize_public(row: dict[str, Any]) -> dict[str, Any]:
    item_id = str(row.get("id", row.get("item_id")))
    prompt = (
        str(row["prompt"]) if "prompt" in row else task_prompt(str(row["code"]), str(row["input"]))
    )
    reference = str(row.get("output", row.get("reference_answer")))
    canonical = canonicalize_semantic_value(reference)
    return {
        "item_id": item_id,
        "benchmark": "CRUXEval",
        "subtask": "output_prediction",
        "prompt": prompt,
        "reference_answer": reference,
        "evaluator": "python_literal",
        "source_revision": DATASET_REVISION,
        "metadata": {
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "reference_canonical_type": str(canonical[0]),
        },
    }


def materialize_phase(review: Path) -> None:
    _assert_frozen(review)
    from datasets import load_dataset

    source = list(load_dataset(DATASET_REPO, split="test", revision=DATASET_REVISION))
    public = {
        _normalize_public(dict(row))["item_id"]: _normalize_public(dict(row)) for row in source
    }
    filenames = (
        "SOURCE_CONSTRUCTION_MANIFEST.json",
        "SOURCE_VALIDATION_MANIFEST.json",
        "SHELL_CALIBRATION_MANIFEST.json",
        "M1_COVARIANCE_MANIFEST.json",
        "M2_PROBE_MANIFEST.json",
        "PRIMARY_PANEL_MANIFEST.json",
    )
    allocations: dict[str, list[dict[str, Any]]] = {}
    all_ids: list[str] = []
    for filename in filenames:
        manifest = read_json(review / filename)
        rows: list[dict[str, Any]] = []
        for frozen in manifest["items"]:
            item_id = str(frozen["item_id"])
            if item_id not in public:
                raise RuntimeError(f"Q2_V3_PANEL_PROVENANCE_MISMATCH: missing {item_id}")
            item = dict(public[item_id])
            if bytes_sha256(item["prompt"]) != frozen["prompt_sha256"]:
                raise RuntimeError(f"Q2_V3_PANEL_PROVENANCE_MISMATCH: prompt {item_id}")
            if bytes_sha256(item["reference_answer"]) != frozen["reference_sha256"]:
                raise RuntimeError(f"Q2_V3_PANEL_PROVENANCE_MISMATCH: reference {item_id}")
            item["metadata"] = {
                **item["metadata"],
                "allocation": frozen["allocation"],
                "provenance_class": frozen["provenance_class"],
                "official_index": frozen["official_index"],
            }
            rows.append(item)
        if [row["item_id"] for row in rows] != manifest["item_ids"]:
            raise RuntimeError(f"Q2_V3_PANEL_PROVENANCE_MISMATCH: order {filename}")
        if ordered_id_hash([row["item_id"] for row in rows]) != manifest["ordered_ids_sha256"]:
            raise RuntimeError(f"Q2_V3_PANEL_PROVENANCE_MISMATCH: hash {filename}")
        allocations[manifest["allocation"]] = rows
        all_ids.extend(row["item_id"] for row in rows)
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Q2_V3_PANEL_PROVENANCE_MISMATCH: allocation overlap")
    write_json(
        review / MATERIALIZED,
        {
            "schema_version": "q2-v3-materialized-public-items-v1",
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "dataset_rows": len(source),
            "allocations": allocations,
            "primary_ordered_ids_sha256": ordered_id_hash(
                [row["item_id"] for row in allocations["PRIMARY_SEMANTIC_PANEL"]]
            ),
            "classification": "Q2_V3_PANEL_PROVENANCE_PASS",
        },
    )
    print(json.dumps({"phase": "materialize", "classification": "Q2_V3_PANEL_PROVENANCE_PASS"}))


def _items(review: Path, allocation: str) -> list[ExternalItem]:
    payload = read_json(review / MATERIALIZED)
    rows = payload["allocations"][allocation]
    return [
        ExternalItem(
            item_id=row["item_id"],
            benchmark=row["benchmark"],
            subtask=row["subtask"],
            prompt=row["prompt"],
            reference_answer=row["reference_answer"],
            evaluator=row["evaluator"],
            source_revision=row["source_revision"],
            metadata=dict(row["metadata"]),
        )
        for row in rows
    ]


def _instruction(family_id: str, polarity: str) -> str:
    family = next(value for value in SOURCE_FAMILIES if value.family_id == family_id)
    return family.positive_instruction if polarity == "POSITIVE" else family.negative_instruction


def _capture_boundaries(backend: Any, item: ExternalItem, system: str) -> dict[str, np.ndarray]:
    row = model_item(item, system)
    prompt_ids, _rendered, _hash = prompt_tokens(backend, row)
    teacher = [
        int(value)
        for value in backend.tokenizer.encode(EXECUTION_TEACHER_TEXT, add_special_tokens=False)
    ]
    full_ids = prompt_ids + teacher
    torch = backend.torch
    ids = torch.tensor([full_ids], dtype=torch.long, device=backend.device)
    attention = torch.ones_like(ids)
    captures: dict[str, np.ndarray] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captures["PROMPT_BOUNDARY"] = hidden[0, len(prompt_ids) - 1].detach().float().cpu().numpy()
        captures["EXECUTION_BOUNDARY"] = hidden[0, -1].detach().float().cpu().numpy()

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
        raise RuntimeError("Q2_V3_ENGINE_FAILURE: boundary capture")
    return captures


def _mechanical_parse(raw: str, token_count: int) -> dict[str, Any]:
    commitment = extract_final_commitment(raw, truncated=token_count >= MAX_NEW_TOKENS)
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


def _journal_rows(
    path: Path, identity: dict[str, Any], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    return list(CrashSafeJournal(path, identity=identity, key_fields=keys).rows.values())


def source_phase(backend: Any, review: Path, code_commit: str) -> None:
    _assert_frozen(review)
    construction = _items(review, "SOURCE_CONSTRUCTION")
    validation = _items(review, "SOURCE_VALIDATION")
    archive_path = review / "Q2_V3_SOURCE_ACTIVATIONS.npz"
    if not archive_path.exists():
        arrays: dict[str, np.ndarray] = {}
        for split, items in (("construction", construction), ("validation", validation)):
            for family in SOURCE_FAMILIES:
                for polarity in ("POSITIVE", "NEGATIVE"):
                    captures = [
                        _capture_boundaries(backend, item, _instruction(family.family_id, polarity))
                        for item in items
                    ]
                    for location in LOCATIONS:
                        arrays[f"{split}__{family.family_id}__{polarity}__{location}"] = np.stack(
                            [row[location] for row in captures]
                        ).astype(np.float32)
        np.savez_compressed(archive_path, **arrays)
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "SOURCE_QUALIFICATION",
        "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
        "code_commit": code_commit,
    }
    journal = CrashSafeJournal(
        review / "Q2_V3_SOURCE_JOURNAL.jsonl",
        identity=identity,
        key_fields=("item_id", "family_id", "polarity", "rollout_index"),
    )
    schedule = read_json(review / "SOURCE_QUALIFICATION_SCHEDULE.json")["rows"]
    item_map = {item.item_id: item for item in validation}
    for row in schedule:
        key = (row["item_id"], row["family"], row["polarity"], row["rollout_index"])
        if key in journal.rows:
            continue
        output = backend.generate_reasoning(
            model_item(item_map[row["item_id"]], _instruction(row["family"], row["polarity"])),
            sampling_seed=int(row["seed"]),
            max_new_tokens=MAX_NEW_TOKENS,
            intervention_metadata={
                "experiment_id": EXPERIMENT_ID,
                "phase": "SOURCE_QUALIFICATION",
                "correctness_forbidden": True,
            },
        )
        metadata = dict(output.metadata)
        token_count = int(metadata.get("generated_token_count", 0))
        journal.append(
            {
                **row,
                "family_id": row["family"],
                **_mechanical_parse(output.raw_output, token_count),
                "raw_output": output.raw_output,
                "generated_token_ids": metadata.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "parser_version": PARSER_VERSION,
                "correctness_evaluated": False,
            }
        )
    if len(journal.rows) != len(schedule):
        raise RuntimeError("Q2_V3_ENGINE_FAILURE: source journal incomplete")
    finalize_source(review, identity)


def finalize_source(review: Path, identity: dict[str, Any]) -> None:
    rows = _journal_rows(
        review / "Q2_V3_SOURCE_JOURNAL.jsonl",
        identity,
        ("item_id", "family_id", "polarity", "rollout_index"),
    )
    activations = np.load(review / "Q2_V3_SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    records: dict[str, Any] = {}
    vectors: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    vector_dir = review / "Q2_V3_VECTORS"
    vector_dir.mkdir(exist_ok=True)
    for family in SOURCE_FAMILIES:
        selected = [row for row in rows if row["family_id"] == family.family_id]
        by_key = {(row["item_id"], row["polarity"], row["rollout_index"]): row for row in selected}
        item_ids = sorted({row["item_id"] for row in selected})
        cross: list[float] = []
        within: list[float] = []
        for item_id in item_ids:
            pos = [by_key[(item_id, "POSITIVE", r)]["canonical_value"] for r in (0, 1)]
            neg = [by_key[(item_id, "NEGATIVE", r)]["canonical_value"] for r in (0, 1)]
            cross.extend(float(a != b) for a in pos for b in neg)
            within.extend((float(pos[0] != pos[1]), float(neg[0] != neg[1])))
        pos_rows = [row for row in selected if row["polarity"] == "POSITIVE"]
        neg_rows = [row for row in selected if row["polarity"] == "NEGATIVE"]
        record: dict[str, Any] = {
            "positive_validity": float(np.mean([row["commitment_valid"] for row in pos_rows])),
            "negative_validity": float(np.mean([row["commitment_valid"] for row in neg_rows])),
            "positive_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in pos_rows])
            ),
            "negative_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in neg_rows])
            ),
            "cross_disagreement": float(np.mean(cross)),
            "within_disagreement": float(np.mean(within)),
            "excess_disagreement": float(np.mean(cross) - np.mean(within)),
            "locations": {},
        }
        for location in LOCATIONS:
            cpos = activations[f"construction__{family.family_id}__POSITIVE__{location}"].astype(
                np.float64
            )
            cneg = activations[f"construction__{family.family_id}__NEGATIVE__{location}"].astype(
                np.float64
            )
            raw = np.mean(cpos - cneg, axis=0)
            raw_norm = float(np.linalg.norm(raw))
            direction = raw / raw_norm
            vpos = activations[f"validation__{family.family_id}__POSITIVE__{location}"].astype(
                np.float64
            )
            vneg = activations[f"validation__{family.family_id}__NEGATIVE__{location}"].astype(
                np.float64
            )
            gaps = (vpos - vneg) @ direction
            gap_sd = float(np.std(gaps, ddof=1))
            standardized = float(np.mean(gaps) / max(gap_sd, 1e-12))
            positive_fraction = float(np.mean(gaps > 0))
            direction_id = base_direction_id(family.family_id, location)
            path = vector_dir / f"{direction_id}.npy"
            np.save(path, direction.astype(np.float64))
            vectors[direction_id] = direction
            metadata[direction_id] = {
                "family_id": family.family_id,
                "location": location,
                "path": str(path.relative_to(ROOT)),
                "vector_hash": vector_sha256(direction),
                "raw_norm": raw_norm,
                "standardized_gap": standardized,
                "positive_projection_fraction": positive_fraction,
            }
            record["locations"][location] = metadata[direction_id]
        source_behavior_pass = (
            all(
                record[key] >= 0.90
                for key in (
                    "positive_validity",
                    "negative_validity",
                    "positive_evaluability",
                    "negative_evaluability",
                )
            )
            and record["cross_disagreement"] >= 0.10
            and record["excess_disagreement"] >= 0.03
        )
        representation_pass = all(
            record["locations"][location]["raw_norm"] >= 1e-6
            and record["locations"][location]["standardized_gap"] >= 0.20
            and record["locations"][location]["positive_projection_fraction"] >= 0.60
            for location in LOCATIONS
        )
        record["pass"] = bool(source_behavior_pass and representation_pass)
        records[family.family_id] = record
    passed = all(record["pass"] for record in records.values()) and len(vectors) == 10
    write_json(
        review / "Q2_V3_SOURCE_QUALIFICATION.json",
        {
            "families": records,
            "all_five_families_pass": passed,
            "correctness_used": False,
            "classification": "Q2_V3_CONTROLLER_QUALIFICATION_PASS"
            if passed
            else "Q2_V3_CONTROLLER_QUALIFICATION_FAILED",
        },
    )
    write_json(
        review / "Q2_V3_DIRECTION_BANK.json",
        {
            "directions": metadata,
            "controller_order_base": list(metadata),
            "classification": "Q2_V3_CONTROLLER_QUALIFICATION_PASS"
            if passed
            else "Q2_V3_CONTROLLER_QUALIFICATION_FAILED",
        },
    )
    if not passed:
        raise RuntimeError("Q2_V3_CONTROLLER_QUALIFICATION_FAILED")


def _load_base_vectors(review: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bank = read_json(review / "Q2_V3_DIRECTION_BANK.json")
    vectors: dict[str, np.ndarray] = {}
    for name, record in bank["directions"].items():
        vector = np.load(ROOT / record["path"], allow_pickle=False).astype(np.float64)
        if (
            vector_sha256(vector) != record["vector_hash"]
            or abs(np.linalg.norm(vector) - 1.0) > 1e-10
        ):
            raise RuntimeError(f"Q2_V3_INSTRUMENT_FAILURE: vector {name}")
        vectors[name] = vector
    return vectors, bank["directions"]


def _construct_nulls(
    vectors: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ordered = np.stack(
        [
            vectors[base_direction_id(f.family_id, loc)]
            for f in SOURCE_FAMILIES
            for loc in LOCATIONS
        ],
        axis=1,
    )
    u, singular, _vh = np.linalg.svd(ordered, full_matrices=False)
    rank = int(np.sum(singular > 1e-10))
    q = u[:, :rank]
    nulls: dict[str, np.ndarray] = {}
    for index, seed in enumerate(NULL_SEEDS):
        candidate = np.random.Generator(np.random.PCG64(seed)).standard_normal(ordered.shape[0])
        candidate -= q @ (q.T @ candidate)
        for prior in nulls.values():
            candidate -= prior * float(prior @ candidate)
        candidate /= np.linalg.norm(candidate)
        nulls[f"NULL_Q2_V3_R{index}"] = candidate
    max_span = max(float(np.linalg.norm(q.T @ value)) for value in nulls.values())
    pairwise = abs(float(nulls["NULL_Q2_V3_R0"] @ nulls["NULL_Q2_V3_R1"]))
    if max_span > 1e-6 or pairwise > 1e-6:
        raise RuntimeError("Q2_V3_ANGULAR_IDENTIFIABILITY_FAILED: null orthogonality")
    return nulls, {
        "seeds": list(NULL_SEEDS),
        "svd_rank": rank,
        "max_span_absolute_cosine": max_span,
        "pairwise_absolute_cosine": pairwise,
    }


def _baseline_denominator(
    backend: Any, items: list[ExternalItem]
) -> tuple[float, list[dict[str, Any]]]:
    teacher = [
        int(v) for v in backend.tokenizer.encode(EXECUTION_TEACHER_TEXT, add_special_tokens=False)
    ]
    squared: list[float] = []
    records: list[dict[str, Any]] = []
    for item in items:
        row = model_item(item)
        prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
        full = prompt_ids + teacher
        torch = backend.torch
        ids = torch.tensor([full], dtype=torch.long, device=backend.device)
        attention = torch.ones_like(ids)
        captured: list[np.ndarray] = []

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            destination: list[np.ndarray] = captured,
            start: int = len(prompt_ids) - 1,
        ) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            destination.append(hidden[0, start:, :].detach().float().cpu().numpy())

        handle = backend.layer_module(LAYER).register_forward_hook(hook)
        try:
            with torch.inference_mode():
                backend._forward(
                    backend.model,
                    {
                        "input_ids": ids,
                        "attention_mask": attention,
                        "use_cache": False,
                        "return_dict": True,
                    },
                    "prefill",
                )  # noqa: SLF001
        finally:
            handle.remove()
        values = captured[0]
        squared.extend(np.sum(values.astype(np.float64) ** 2, axis=1).tolist())
        records.append(
            {"item_id": item.item_id, "prompt_hash": prompt_hash, "positions": len(values)}
        )
    return float(np.mean(squared)), records


def _amplitude(direction: np.ndarray, alpha: float, denominator: float) -> float:
    import torch

    delta = (
        torch.tensor(direction * alpha, dtype=torch.float64)
        .to(torch.bfloat16)
        .float()
        .numpy()
        .astype(np.float64)
    )
    return float(math.sqrt(float(delta @ delta) / denominator))


def _calibrate_alpha(direction: np.ndarray, target: float, denominator: float) -> dict[str, float]:
    low, high = 0.0, 256.0
    visited: list[tuple[float, float]] = []
    for _ in range(40):
        midpoint = 0.5 * (low + high)
        value = _amplitude(direction, midpoint, denominator)
        visited.append((midpoint, value))
        if value < target:
            low = midpoint
        else:
            high = midpoint
    alpha, value = min(visited, key=lambda pair: (abs(pair[1] - target), pair[0]))
    relative = abs(value - target) / target
    if relative > 0.005:
        raise RuntimeError("Q2_V3_CONTROLLER_BANK_DESTRUCTIVE: shell root failure")
    return {
        "alpha": float(alpha),
        "implemented_amplitude": float(value),
        "relative_target_error": float(relative),
    }


def _condition_context(
    backend: Any,
    item: ExternalItem,
    condition: str,
    vectors: dict[str, np.ndarray],
    deployment: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    row = model_item(item)
    prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
    if condition == "BASELINE":
        return (
            nullcontext(),
            row,
            {"intervention": "none", "prompt_length": len(prompt_ids), "prompt_hash": prompt_hash},
        )
    record = deployment[condition]
    delta = vectors[record["base_direction_id"]] * float(record["alpha"])
    tensor = backend.torch.tensor(delta, dtype=backend.torch.float32, device=backend.device).view(
        1, 1, -1
    )
    return (
        Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: tensor},
            target_positions=[len(prompt_ids) - 1],
        ),
        row,
        {
            **record,
            "intervention": condition,
            "layer": LAYER,
            "duration": "sustained_current_token",
            "prompt_length": len(prompt_ids),
            "prompt_hash": prompt_hash,
        },
    )


def shell_phase(backend: Any, review: Path, code_commit: str) -> None:
    _assert_frozen(review)
    vectors, records = _load_base_vectors(review)
    nulls, null_geometry = _construct_nulls(vectors)
    vectors.update(nulls)
    items = _items(review, "SHELL_CALIBRATION")
    denominator, denominator_records = _baseline_denominator(backend, items)
    deployment: dict[str, Any] = {}
    vector_meta: dict[str, Any] = dict(records)
    for name, vector in nulls.items():
        vector_meta[name] = {
            "family_id": "NULL",
            "location": "NULL",
            "vector_hash": vector_sha256(vector),
        }
        path = review / "Q2_V3_VECTORS" / f"{name}.npy"
        np.save(path, vector.astype(np.float64))
        vector_meta[name]["path"] = str(path.relative_to(ROOT))
    for shell in SHELLS:
        for family in SOURCE_FAMILIES:
            for location in LOCATIONS:
                base = base_direction_id(family.family_id, location)
                condition = f"{base}_{shell}"
                deployment[condition] = {
                    "condition": condition,
                    "base_direction_id": base,
                    "family_id": family.family_id,
                    "location": location,
                    "shell": shell,
                    "target_amplitude": SHELL_TARGETS[shell],
                    "vector_hash": vector_meta[base]["vector_hash"],
                    **_calibrate_alpha(vectors[base], SHELL_TARGETS[shell], denominator),
                }
        for index in range(2):
            base = f"NULL_Q2_V3_R{index}"
            condition = f"{base}_{shell}"
            deployment[condition] = {
                "condition": condition,
                "base_direction_id": base,
                "family_id": "NULL",
                "location": "NULL",
                "shell": shell,
                "target_amplitude": SHELL_TARGETS[shell],
                "vector_hash": vector_meta[base]["vector_hash"],
                **_calibrate_alpha(vectors[base], SHELL_TARGETS[shell], denominator),
            }
    write_json(
        review / "Q2_V3_CONTROLLER_DEPLOYMENT.json",
        {
            "denominator_mean_squared_norm": denominator,
            "denominator_records": denominator_records,
            "vectors": vector_meta,
            "controllers": deployment,
            "null_geometry": null_geometry,
            "classification": "Q2_V3_SHELL_CALIBRATION_PASS",
        },
    )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "SHELL_SAFETY",
        "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
        "code_commit": code_commit,
    }
    journal = CrashSafeJournal(
        review / "Q2_V3_SHELL_JOURNAL.jsonl",
        identity=identity,
        key_fields=("item_id", "condition", "rollout_index"),
    )
    schedule = read_json(review / "SHELL_CALIBRATION_SCHEDULE.json")["rows"]
    item_map = {item.item_id: item for item in items}
    meaningful_set = set(meaningful_controller_ids())
    for row in schedule:
        key = (row["item_id"], row["condition"], row["rollout_index"])
        if key in journal.rows:
            continue
        if row["condition"] != "BASELINE" and row["condition"] not in meaningful_set:
            raise RuntimeError("Q2_V3_INSTRUMENT_FAILURE: shell schedule condition")
        context, model_row, condition_meta = _condition_context(
            backend, item_map[row["item_id"]], row["condition"], vectors, deployment
        )
        with context:
            output = backend.generate_reasoning(
                model_row,
                sampling_seed=int(row["matched_seed"]),
                max_new_tokens=MAX_NEW_TOKENS,
                intervention_metadata={
                    "experiment_id": EXPERIMENT_ID,
                    "phase": "SHELL_SAFETY",
                    "correctness_forbidden": True,
                    **condition_meta,
                },
            )
        meta = dict(output.metadata)
        token_count = int(meta.get("generated_token_count", 0))
        journal.append(
            {
                **row,
                **_mechanical_parse(output.raw_output, token_count),
                "raw_output": output.raw_output,
                "generated_token_ids": meta.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "truncated": token_count >= MAX_NEW_TOKENS,
                "condition_metadata": condition_meta,
                "correctness_evaluated": False,
            }
        )
    if len(journal.rows) != len(schedule):
        raise RuntimeError("Q2_V3_ENGINE_FAILURE: shell journal incomplete")
    finalize_shell(review, identity)


def finalize_shell(review: Path, identity: dict[str, Any]) -> None:
    rows = _journal_rows(
        review / "Q2_V3_SHELL_JOURNAL.jsonl", identity, ("item_id", "condition", "rollout_index")
    )
    by_key = {(row["item_id"], row["condition"], row["rollout_index"]): row for row in rows}
    baseline = [row for row in rows if row["condition"] == "BASELINE"]
    baseline_validity = float(np.mean([row["commitment_valid"] for row in baseline]))
    baseline_eval = float(np.mean([row["semantic_evaluable"] for row in baseline]))
    records: dict[str, Any] = {}
    all_pass = True
    for condition in meaningful_controller_ids():
        selected = [row for row in rows if row["condition"] == condition]
        movement = float(
            np.mean(
                [
                    row["generated_token_ids"]
                    != by_key[(row["item_id"], "BASELINE", row["rollout_index"])][
                        "generated_token_ids"
                    ]
                    for row in selected
                ]
            )
        )
        validity = float(np.mean([row["commitment_valid"] for row in selected]))
        evaluability = float(np.mean([row["semantic_evaluable"] for row in selected]))
        truncation = float(np.mean([row["truncated"] for row in selected]))
        threshold = 0.10 if condition.endswith("_MEDIUM") else 0.15
        passed = (
            validity >= 0.90
            and validity >= baseline_validity - 0.05
            and evaluability >= 0.90
            and evaluability >= baseline_eval - 0.05
            and truncation <= 0.05
            and movement >= threshold
        )
        records[condition] = {
            "validity": validity,
            "evaluability": evaluability,
            "truncation": truncation,
            "raw_sequence_movement": movement,
            "movement_threshold": threshold,
            "pass": passed,
        }
        all_pass = all_pass and passed
    classification = "Q2_V3_SHELL_SAFETY_PASS" if all_pass else "Q2_V3_CONTROLLER_BANK_DESTRUCTIVE"
    write_json(
        review / "Q2_V3_SHELL_SAFETY.json",
        {
            "baseline_validity": baseline_validity,
            "baseline_evaluability": baseline_eval,
            "controllers": records,
            "all_20_pass": all_pass,
            "classification": classification,
        },
    )
    if not all_pass:
        raise RuntimeError("Q2_V3_CONTROLLER_BANK_DESTRUCTIVE")


def _load_deployment(review: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = read_json(review / "Q2_V3_CONTROLLER_DEPLOYMENT.json")
    vectors: dict[str, np.ndarray] = {}
    for name, record in payload["vectors"].items():
        vector = np.load(ROOT / record["path"], allow_pickle=False).astype(np.float64)
        if vector_sha256(vector) != record["vector_hash"]:
            raise RuntimeError(f"Q2_V3_INSTRUMENT_FAILURE: deployed vector {name}")
        vectors[name] = vector
    return vectors, payload["controllers"]


def _capture_covariance(backend: Any, review: Path) -> np.ndarray:
    values: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for item in _items(review, "M1_COVARIANCE"):
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
            destination.append(hidden[0, -1].detach().float().cpu().numpy())

        handle = backend.layer_module(LAYER).register_forward_hook(hook)
        try:
            with torch.inference_mode():
                backend._forward(
                    backend.model,
                    {
                        "input_ids": ids,
                        "attention_mask": attention,
                        "use_cache": False,
                        "return_dict": True,
                    },
                    "prefill",
                )  # noqa: SLF001
        finally:
            handle.remove()
        values.append(captured[0])
        records.append({"item_id": item.item_id, "prompt_hash": prompt_hash})
    array = np.stack(values).astype(np.float32)
    np.savez_compressed(review / "Q2_V3_M1_ACTIVATIONS.npz", activations=array)
    write_json(review / "Q2_V3_M1_CAPTURE_METADATA.json", records)
    return array


def _checkpoint_indices(length: int) -> tuple[int, int, int]:
    return (length // 3, (2 * length) // 3, length - 1)


def _secant_logits(
    backend: Any,
    item: ExternalItem,
    condition: str,
    vectors: dict[str, np.ndarray],
    deployment: dict[str, Any],
    continuation: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    row = model_item(item)
    prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
    delta = (
        None
        if condition == "BASELINE"
        else vectors[deployment[condition]["base_direction_id"]]
        * float(deployment[condition]["alpha"])
    )
    snapshots: list[np.ndarray] = []
    context = (
        nullcontext() if delta is None else DiagnosticHooks(backend, (), delta, len(prompt_ids))
    )
    checkpoints = _checkpoint_indices(len(continuation))
    with context as hooks, backend.torch.inference_mode():
        output = forward(
            backend, prompt_ids, past=None, total_length=len(prompt_ids), phase="prefill"
        )
        if hooks is not None:
            hooks.note_forward()
        snapshots.append(output.logits[0, -1].detach().float().cpu().numpy())
        past = output.past_key_values
        for index, token in enumerate(continuation):
            output = forward(
                backend,
                [int(token)],
                past=past,
                total_length=len(prompt_ids) + index + 1,
                phase="decode",
            )
            if hooks is not None:
                hooks.note_forward()
            past = output.past_key_values
            if index in checkpoints:
                snapshots.append(output.logits[0, -1].detach().float().cpu().numpy())
    if len(snapshots) != 4:
        raise RuntimeError("Q2_V3_ENGINE_FAILURE: M2 checkpoints")
    return np.stack(snapshots).astype(np.float32), {
        "item_id": item.item_id,
        "prompt_hash": prompt_hash,
        "checkpoint_indices": [-1, *checkpoints],
        "condition": condition,
    }


def geometry_phase(backend: Any, review: Path, code_commit: str) -> None:
    if (
        read_json(review / "Q2_V3_SOURCE_QUALIFICATION.json")["classification"]
        != "Q2_V3_CONTROLLER_QUALIFICATION_PASS"
    ):
        raise RuntimeError("Q2_V3_CONTROLLER_QUALIFICATION_FAILED")
    if read_json(review / "Q2_V3_SHELL_SAFETY.json")["classification"] != "Q2_V3_SHELL_SAFETY_PASS":
        raise RuntimeError("Q2_V3_CONTROLLER_BANK_DESTRUCTIVE")
    vectors, deployment = _load_deployment(review)
    covariance = _capture_covariance(backend, review)
    continuation = [
        int(v) for v in backend.tokenizer.encode(EXECUTION_TEACHER_TEXT, add_special_tokens=False)
    ]
    raw_dir = review / "Q2_V3_M2_LOGITS"
    raw_dir.mkdir(exist_ok=True)
    archive_records: list[dict[str, Any]] = []
    for item in _items(review, "M2_LABEL_FREE_PROBES"):
        path = raw_dir / f"{item.item_id}.npz"
        metadata_path = raw_dir / f"{item.item_id}.json"
        if not (path.exists() and metadata_path.exists()):
            arrays: dict[str, np.ndarray] = {}
            metadata: dict[str, Any] = {}
            for condition in condition_ids():
                arrays[condition], metadata[condition] = _secant_logits(
                    backend, item, condition, vectors, deployment, continuation
                )
            np.savez_compressed(path, **arrays)
            write_json(metadata_path, metadata)
        archive_records.append(
            {
                "item_id": item.item_id,
                "path": str(path.relative_to(review)),
                "sha256": sha256(path),
                "metadata_path": str(metadata_path.relative_to(review)),
                "metadata_sha256": sha256(metadata_path),
            }
        )
    archive = {
        "records": archive_records,
        "conditions": list(condition_ids()),
        "checkpoint_count": 4,
        "storage_dtype": "float32",
        "correctness_labels_used": False,
    }
    write_json(review / "Q2_V3_M2_ARCHIVE.json", archive)
    names = list(condition_ids()[1:])
    base_rows = np.stack([vectors[deployment[name]["base_direction_id"]] for name in names])
    m0 = np.asarray(flat_geometry(base_rows)["normalized_euclidean"], dtype=np.float64)
    fit = fit_whitening(covariance.astype(np.float64), shrinkage=0.10)
    m1 = np.asarray(whitened_geometry(base_rows, fit)["normalized_euclidean"], dtype=np.float64)
    condition_logits: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for record in archive_records:
        values = np.load(review / record["path"], allow_pickle=False)
        for name in names:
            condition_logits[name].append(values[name].astype(np.float64))
    stacked = {name: np.concatenate(condition_logits[name], axis=0) for name in names}
    m2 = np.asarray(finite_secant_geometry(stacked, names)["sqrt_mean_js"], dtype=np.float64)
    np.savez_compressed(review / "Q2_V3_PREDICTION_MATRICES.npz", M0=m0, M1=m1, M2=m2)
    fit_path = review / "Q2_V3_M1_FIT.npz"
    np.savez_compressed(
        fit_path,
        mean=fit.mean,
        basis=fit.basis,
        inverse_eigenvalues=fit.inverse_eigenvalues,
        ridge=np.asarray([fit.ridge]),
    )
    matrices_path = review / "Q2_V3_PREDICTION_MATRICES.npz"
    metadata = {
        "source_commit": EXPERIMENT_SOURCE_COMMIT,
        "code_commit": code_commit,
        "controller_order": names,
        "controller_order_hash": bytes_sha256(json.dumps(names, separators=(",", ":"))),
        "controller_vector_hashes": {name: deployment[name]["vector_hash"] for name in names},
        "implemented_alphas_and_amplitudes": {
            name: {
                key: deployment[name][key]
                for key in (
                    "alpha",
                    "implemented_amplitude",
                    "target_amplitude",
                    "relative_target_error",
                )
            }
            for name in names
        },
        "calibration_manifest_hashes": {
            name: sha256(review / name)
            for name in (
                "SOURCE_CONSTRUCTION_MANIFEST.json",
                "SOURCE_VALIDATION_MANIFEST.json",
                "SHELL_CALIBRATION_MANIFEST.json",
                "M1_COVARIANCE_MANIFEST.json",
                "M2_PROBE_MANIFEST.json",
            )
        },
        "M1_fit_hash": sha256(fit_path),
        "M2_archive_hash": sha256(review / "Q2_V3_M2_ARCHIVE.json"),
        "matrix_hashes": {
            "archive": sha256(matrices_path),
            "M0": bytes_sha256(m0.tobytes().hex()),
            "M1": bytes_sha256(m1.tobytes().hex()),
            "M2": bytes_sha256(m2.tobytes().hex()),
        },
        "geometry_data_hash": bytes_sha256(
            json.dumps(archive_records, sort_keys=True, separators=(",", ":"))
            + sha256(review / "Q2_V3_M1_ACTIVATIONS.npz")
        ),
    }
    write_json(review / "Q2_V3_PREDICTION_MATRICES.json", metadata)
    _identifiability(review, base_rows, m0, m1, m2, deployment)


def _cross_edges(shell_names: list[str], deployment: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(len(shell_names))
        for j in range(i + 1, len(shell_names))
        if deployment[shell_names[i]]["family_id"] != deployment[shell_names[j]]["family_id"]
    ]


def _ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    prediction = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 0.0 if total <= 1e-15 else float(1.0 - np.sum((y - prediction) ** 2) / total)


def _identifiability(
    review: Path,
    base_rows: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    deployment: dict[str, Any],
) -> None:
    names = list(condition_ids()[1:])
    meaningful = list(meaningful_controller_ids())
    ten = base_rows[:10]
    gram = ten @ ten.T
    eigen = np.linalg.eigvalsh(gram)
    effective_rank = float(np.sum(eigen) ** 2 / np.sum(eigen**2))
    max_cos = float(np.max(np.abs(gram[np.triu_indices(10, 1)])))
    checks: dict[str, Any] = {
        "direction_gram_effective_rank": effective_rank,
        "direction_gram_effective_rank_pass": effective_rank >= 5.0,
        "max_absolute_nonantipodal_cosine": max_cos,
        "max_absolute_nonantipodal_cosine_pass": max_cos < 0.95,
    }
    matrices = {"M0": m0, "M1": m1, "M2": m2}
    for shell in SHELLS:
        shell_meaningful = [name for name in meaningful if name.endswith(f"_{shell}")]
        shell_all = [name for name in names if name.endswith(f"_{shell}")]
        radii = np.asarray([deployment[name]["implemented_amplitude"] for name in shell_all])
        cv = float(np.std(radii) / np.mean(radii))
        checks[f"{shell}.radius_cv"] = cv
        checks[f"{shell}.radius_cv_pass"] = cv <= 0.03
        global_median = float(
            np.median([deployment[name]["implemented_amplitude"] for name in shell_meaningful])
        )
        deviations = []
        for family in SOURCE_FAMILIES:
            family_radii = [
                deployment[name]["implemented_amplitude"]
                for name in shell_meaningful
                if deployment[name]["family_id"] == family.family_id
            ]
            deviations.append(abs(float(np.median(family_radii)) - global_median) / global_median)
        checks[f"{shell}.max_family_median_deviation"] = max(deviations)
        checks[f"{shell}.family_median_deviation_pass"] = max(deviations) <= 0.03
        indices = [names.index(name) for name in shell_meaningful]
        edges = _cross_edges(shell_meaningful, deployment)
        checks[f"{shell}.cross_family_dyads"] = len(edges)
        checks[f"{shell}.cross_family_dyads_pass"] = len(edges) == 40
        features: list[np.ndarray] = []
        for metric, matrix in matrices.items():
            values = np.asarray([matrix[indices[i], indices[j]] for i, j in edges])
            if metric in {"M0", "M1"}:
                spread = float(np.quantile(values, 0.9) - np.quantile(values, 0.1))
                checks[f"{shell}.{metric}.q90_q10"] = spread
                checks[f"{shell}.{metric}.angular_spread_pass"] = spread >= 0.20
            radius_left = np.asarray(
                [deployment[shell_meaningful[i]]["implemented_amplitude"] for i, _j in edges]
            )
            radius_right = np.asarray(
                [deployment[shell_meaningful[j]]["implemented_amplitude"] for _i, j in edges]
            )
            design = np.column_stack(
                (
                    np.ones(len(edges)),
                    np.abs(radius_left - radius_right),
                    0.5 * (radius_left + radius_right),
                )
            )
            r2 = _ols_r2(values, design)
            checks[f"{shell}.{metric}.radial_nuisance_r2"] = r2
            checks[f"{shell}.{metric}.radial_nuisance_pass"] = r2 <= 0.10
            z = (values - np.mean(values)) / max(float(np.std(values)), 1e-12)
            leverage_denominator = 2.0 * float(np.sum(z**2))
            leverage = []
            for family in SOURCE_FAMILIES:
                incident = [
                    k
                    for k, (i, j) in enumerate(edges)
                    if deployment[shell_meaningful[i]]["family_id"] == family.family_id
                    or deployment[shell_meaningful[j]]["family_id"] == family.family_id
                ]
                leverage.append(float(np.sum(z[incident] ** 2) / max(leverage_denominator, 1e-12)))
            checks[f"{shell}.{metric}.max_family_leverage"] = max(leverage)
            checks[f"{shell}.{metric}.family_leverage_pass"] = max(leverage) <= 0.30
            features.append(z)
        feature_matrix = np.column_stack(features)
        singular = np.linalg.svd(feature_matrix, compute_uv=False)
        condition_number = float(singular[0] / singular[-1])
        checks[f"{shell}.geometry_condition_number"] = condition_number
        checks[f"{shell}.geometry_condition_number_pass"] = condition_number <= 30.0
    deployment_payload = read_json(review / "Q2_V3_CONTROLLER_DEPLOYMENT.json")
    null_geometry = deployment_payload["null_geometry"]
    max_target = max(record["relative_target_error"] for record in deployment.values())
    checks["null_span_cosine_pass"] = null_geometry["max_span_absolute_cosine"] <= 1e-6
    checks["null_pairwise_cosine_pass"] = null_geometry["pairwise_absolute_cosine"] <= 1e-6
    checks["target_relative_error_pass"] = max_target <= 0.005
    passed = all(value for key, value in checks.items() if key.endswith("_pass"))
    write_json(
        review / "Q2_V3_IDENTIFIABILITY.json",
        {
            "checks": checks,
            "all_pass": passed,
            "classification": "Q2_V3_ANGULAR_IDENTIFIABILITY_PASS"
            if passed
            else "Q2_V3_ANGULAR_IDENTIFIABILITY_FAILED",
        },
    )
    if not passed:
        raise RuntimeError("Q2_V3_ANGULAR_IDENTIFIABILITY_FAILED")


def prediction_lock_phase(review: Path, code_commit: str, seal_commit: str | None) -> None:
    if (
        read_json(review / "Q2_V3_IDENTIFIABILITY.json")["classification"]
        != "Q2_V3_ANGULAR_IDENTIFIABILITY_PASS"
    ):
        raise RuntimeError("Q2_V3_ANGULAR_IDENTIFIABILITY_FAILED")
    arrays = np.load(review / "Q2_V3_PREDICTION_MATRICES.npz", allow_pickle=False)
    if set(arrays.files) != {"M0", "M1", "M2"} or any(
        arrays[key].shape != (24, 24) for key in arrays.files
    ):
        raise RuntimeError("Q2_V3_PREDICTION_LOCK_FAILED")
    metadata = read_json(review / "Q2_V3_PREDICTION_MATRICES.json")
    if metadata["code_commit"] != code_commit or metadata["controller_order"] != list(
        condition_ids()[1:]
    ):
        raise RuntimeError("Q2_V3_PREDICTION_LOCK_FAILED")
    payload = {
        "schema_version": "q2-v3-prediction-lock-v1",
        "classification": "Q2_V3_PREDICTION_LOCK_PASS",
        "source_commit": EXPERIMENT_SOURCE_COMMIT,
        "code_commit": code_commit,
        "seal_commit": seal_commit or "PENDING_COMMIT",
        "matrix_archive_sha256": sha256(review / "Q2_V3_PREDICTION_MATRICES.npz"),
        "matrix_hashes": metadata["matrix_hashes"],
        "controller_order_hash": metadata["controller_order_hash"],
        "geometry_data_hash": metadata["geometry_data_hash"],
        "semantic_rows_at_lock": 0,
    }
    write_json(review / "Q2_V3_PREDICTION_LOCK.json", payload)
    print(json.dumps(payload))


def engineering_phase(backend: Any, review: Path) -> None:
    vectors, deployment = _load_deployment(review)
    items = _items(review, "SHELL_CALIBRATION")[:2]
    identities: list[bool] = []
    cleanups: list[bool] = []
    traces: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        row = model_item(item)
        seed = 92_000_000 + index
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
            zero_output = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        identities.append(
            clean.metadata.get("generated_token_ids")
            == zero_output.metadata.get("generated_token_ids")
        )
        for condition in (
            meaningful_controller_ids() if index == 0 else meaningful_controller_ids()[:1]
        ):
            context, model_row, _meta = _condition_context(
                backend, item, condition, vectors, deployment
            )
            with context as trace:
                backend.generate_reasoning(model_row, sampling_seed=seed + 100, max_new_tokens=16)
            traces.append(trace.metadata())
        after = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        cleanups.append(
            clean.metadata.get("generated_token_ids") == after.metadata.get("generated_token_ids")
        )
    applications = [row for trace in traces for row in trace["applications"]]
    result = {
        "alpha_zero_identity": all(identities),
        "hook_cleanup": all(cleanups),
        "per_forward_exact_shift": bool(applications)
        and max(float(row["relative_shift_error"]) for row in applications) <= 2.0,
        "current_token_scope": bool(applications)
        and max(abs(float(row["non_current_change"])) for row in applications) <= 0.125,
        "one_application_per_forward": sum(trace["forward_count"] for trace in traces)
        == len(applications),
        "cache_safety": any(int(row["sequence_length"]) == 1 for row in applications),
        "all_meaningful_controllers_exercised": len(traces) >= 20,
        "scientific_items_used": False,
    }
    result["pass"] = all(
        value for key, value in result.items() if key not in {"scientific_items_used"}
    )
    result["classification"] = (
        "Q2_V3_ENGINEERING_PASS" if result["pass"] else "Q2_V3_ENGINE_FAILURE"
    )
    write_json(review / "Q2_V3_ENGINEERING_CHECKS.json", result)
    if not result["pass"]:
        raise RuntimeError("Q2_V3_ENGINE_FAILURE")


def preflight_phase(backend: Any, review: Path, wallet_balance: float, hourly_rate: float) -> None:
    vectors, deployment = _load_deployment(review)
    items = _items(review, "SHELL_CALIBRATION")[:2]
    durations: list[float] = []
    tokens: list[int] = []
    for item in items:
        for condition in (
            "BASELINE",
            meaningful_controller_ids()[0],
            meaningful_controller_ids()[-1],
            null_controller_ids()[0],
        ):
            context, model_row, _meta = _condition_context(
                backend, item, condition, vectors, deployment
            )
            started = time.monotonic()
            with context:
                output = backend.generate_reasoning(
                    model_row,
                    sampling_seed=93_000_000 + len(durations),
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            durations.append(time.monotonic() - started)
            tokens.append(int(output.metadata.get("generated_token_count", 0)))
    mean = float(np.mean(durations))
    projected_hours = mean * 10_000 * 1.50 / 3600.0
    projected_cost = projected_hours * hourly_rate
    pass_gate = wallet_balance >= 18.0 and projected_cost <= 15.0
    payload = {
        "wallet_balance_usd": wallet_balance,
        "hourly_rate_usd": hourly_rate,
        "fixture_rows": len(durations),
        "mean_seconds_per_row": mean,
        "mean_tokens": float(np.mean(tokens)),
        "projected_hours_with_50pct_tail": projected_hours,
        "projected_cost_with_50pct_tail_usd": projected_cost,
        "wallet_minimum_usd": 18.0,
        "hard_ceiling_usd": 15.0,
        "classification": "Q2_V3_WALLET_GATE_PASS" if pass_gate else "Q2_V3_WALLET_GATE_FAILED",
    }
    write_json(review / "Q2_V3_COST_PREFLIGHT.json", payload)
    if not pass_gate:
        raise RuntimeError("Q2_V3_WALLET_GATE_FAILED")
    print(json.dumps(payload))


def collect_phase(
    backend: Any, review: Path, code_commit: str, prediction_seal_commit: str
) -> None:
    prediction = read_json(review / "Q2_V3_PREDICTION_LOCK.json")
    if prediction["classification"] != "Q2_V3_PREDICTION_LOCK_PASS":
        raise RuntimeError("Q2_V3_PREDICTION_LOCK_FAILED")
    if prediction["seal_commit"] not in {prediction_seal_commit, "PENDING_COMMIT"}:
        raise RuntimeError("Q2_V3_PREDICTION_LOCK_FAILED: seal commit")
    if (
        read_json(review / "Q2_V3_ENGINEERING_CHECKS.json")["classification"]
        != "Q2_V3_ENGINEERING_PASS"
    ):
        raise RuntimeError("Q2_V3_ENGINE_FAILURE")
    if (
        read_json(review / "Q2_V3_COST_PREFLIGHT.json")["classification"]
        != "Q2_V3_WALLET_GATE_PASS"
    ):
        raise RuntimeError("Q2_V3_WALLET_GATE_FAILED")
    vectors, deployment = _load_deployment(review)
    items = {item.item_id: item for item in _items(review, "PRIMARY_SEMANTIC_PANEL")}
    schedule = read_json(review / "EVALUATION_SCHEDULE.json")["rows"]
    if (
        len(schedule) != 10_000
        or len({(r["item_id"], r["condition"], r["rollout_index"]) for r in schedule}) != 10_000
    ):
        raise RuntimeError("Q2_V3_INSTRUMENT_FAILURE: schedule")
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "PRIMARY_SEMANTIC_PANEL",
        "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
        "code_commit": code_commit,
        "prediction_seal_commit": prediction_seal_commit,
        "prediction_lock_sha256": sha256(review / "Q2_V3_PREDICTION_LOCK.json"),
    }
    journal = CrashSafeJournal(
        review / "Q2_V3_SEMANTIC_JOURNAL.jsonl",
        identity=identity,
        key_fields=("item_id", "condition", "rollout_index"),
    )
    for schedule_index, row in enumerate(schedule):
        key = (row["item_id"], row["condition"], row["rollout_index"])
        if key in journal.rows:
            continue
        context, model_row, condition_meta = _condition_context(
            backend, items[row["item_id"]], row["condition"], vectors, deployment
        )
        started = time.monotonic()
        with context as trace:
            output = backend.generate_reasoning(
                model_row,
                sampling_seed=int(row["seed"]),
                max_new_tokens=MAX_NEW_TOKENS,
                intervention_metadata={
                    "experiment_id": EXPERIMENT_ID,
                    "phase": "PRIMARY_SEMANTIC_PANEL",
                    "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
                    "prediction_seal_commit": prediction_seal_commit,
                    "parser_version": PARSER_VERSION,
                    "environment_profile": "CORE_QWEN",
                    **condition_meta,
                },
            )
        meta = dict(output.metadata)
        token_count = int(meta.get("generated_token_count", 0))
        result = evaluate_external_answer_v3(
            output.raw_output,
            items[row["item_id"]].reference_answer,
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
                "generated_token_ids": meta.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "reference_answer": items[row["item_id"]].reference_answer,
                "prompt_hash": items[row["item_id"]].prompt_hash,
                "condition_metadata": condition_meta,
                "hook_trace": trace.metadata() if row["condition"] != "BASELINE" else None,
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "parser_version": PARSER_VERSION,
                "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
                "code_commit": code_commit,
                "prediction_seal_commit": prediction_seal_commit,
                "schedule_index": schedule_index,
                "elapsed_seconds": time.monotonic() - started,
                "retry_count": 0,
            }
        )
        if len(journal.rows) % 100 == 0:
            print(
                json.dumps(
                    {
                        "health": "running",
                        "completed_rows": len(journal.rows),
                        "expected_rows": 10_000,
                    }
                ),
                flush=True,
            )
    if len(journal.rows) != 10_000:
        raise RuntimeError("Q2_V3_ENGINE_FAILURE: semantic journal incomplete")
    write_json(
        review / "Q2_V3_COLLECTION_METADATA.json",
        {
            "completed_rows": 10_000,
            "expected_rows": 10_000,
            "logical_keys": 10_000,
            "prediction_lock_sha256": sha256(review / "Q2_V3_PREDICTION_LOCK.json"),
            "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
            "code_commit": code_commit,
            "prediction_seal_commit": prediction_seal_commit,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "materialize",
            "source",
            "shell",
            "geometry",
            "prediction-lock",
            "engineering",
            "preflight",
            "collect",
        ),
    )
    parser.add_argument("--model-path")
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--code-commit", default=git_head())
    parser.add_argument("--prediction-seal-commit")
    parser.add_argument("--wallet-balance", type=float)
    parser.add_argument("--hourly-rate", type=float, default=0.44)
    args = parser.parse_args()
    review = args.review_dir.resolve()
    if args.mode == "materialize":
        require_remote_hf_execution("Q2 V3 public materialization")
        materialize_phase(review)
        return 0
    if args.mode == "prediction-lock":
        prediction_lock_phase(review, args.code_commit, args.prediction_seal_commit)
        return 0
    require_remote_hf_execution(f"Q2 V3 {args.mode}")
    if not args.model_path:
        raise RuntimeError("--model-path is required for model phases")
    backend = build_backend(args.model_path)
    if args.mode == "source":
        source_phase(backend, review, args.code_commit)
    elif args.mode == "shell":
        shell_phase(backend, review, args.code_commit)
    elif args.mode == "geometry":
        geometry_phase(backend, review, args.code_commit)
    elif args.mode == "engineering":
        engineering_phase(backend, review)
    elif args.mode == "preflight":
        if args.wallet_balance is None:
            raise RuntimeError("--wallet-balance is required")
        preflight_phase(backend, review, args.wallet_balance, args.hourly_rate)
    else:
        if not args.prediction_seal_commit:
            raise RuntimeError("--prediction-seal-commit is required")
        collect_phase(backend, review, args.code_commit, args.prediction_seal_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
