#!/usr/bin/env python3
"""Phased Q2-V2 runner with a strict pre-common-panel firewall."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    canonicalize_semantic_value,
    extract_final_commitment,
)
from epistemic_geometry.experiments.gate6 import paired_mean_direction  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.experiments.q2_controller_heldout_v2 import (  # noqa: E402
    EXECUTION_TEACHER_TEXT,
    EXPERIMENT_ID,
    LAYER,
    LOCATIONS,
    MODEL,
    MODEL_REVISION,
    SIGNS,
    SOURCE_AXES,
    source_pass,
)
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from epistemic_geometry.research.reliability import CrashSafeJournal  # noqa: E402

REVIEW = ROOT / "review/q2_controller_bank_v2"
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
    marker = REVIEW / "EXECUTION_SOURCE_COMMIT.txt"
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_lock(review: Path) -> dict[str, Any]:
    lock = read_json(review / "PROTOCOL_LOCK.json")
    if lock["status"] != "FROZEN_PRE_SOURCE_QUALIFICATION":
        raise RuntimeError("Q2 V2 source lock is not frozen")
    if lock["model"]["id"] != MODEL or lock["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("Q2 V2 model differs from the source lock")
    return lock


def _source_instruction(axis_id: str, polarity: str) -> str:
    axis = next(axis for axis in SOURCE_AXES if axis.axis_id == axis_id)
    if polarity == "PLUS":
        return axis.positive_instruction
    if polarity == "MINUS":
        return axis.negative_instruction
    raise ValueError(f"unknown V2 source sign: {polarity}")


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
        raise RuntimeError("V2 source capture missed a frozen boundary")
    return captures


def _mechanical_parse(raw_output: str, token_count: int) -> dict[str, Any]:
    commitment = extract_final_commitment(raw_output, truncated=token_count >= MAX_NEW_TOKENS)
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


def _disagreement_metrics(rows: list[dict[str, Any]], axis_id: str) -> dict[str, float]:
    selected = [row for row in rows if row["axis_id"] == axis_id]
    by_key = {(row["item_id"], row["polarity"], row["rollout_index"]): row for row in selected}
    item_ids = sorted({row["item_id"] for row in selected})
    cross: list[float] = []
    within: list[float] = []
    for item_id in item_ids:
        plus = [by_key[(item_id, "PLUS", rollout)]["canonical_value"] for rollout in (0, 1)]
        minus = [by_key[(item_id, "MINUS", rollout)]["canonical_value"] for rollout in (0, 1)]
        cross.extend(float(left != right) for left in plus for right in minus)
        within.append(0.5 * float(plus[0] != plus[1]) + 0.5 * float(minus[0] != minus[1]))
    cross_mean = float(np.mean(cross))
    within_mean = float(np.mean(within))
    return {
        "cross_disagreement": cross_mean,
        "within_disagreement": within_mean,
        "excess_disagreement": cross_mean - within_mean,
    }


def source_phase(backend: Any, review: Path, source_commit: str) -> None:
    lock = source_lock(review)
    construction = load_external(review / lock["allocations"]["V2_SOURCE_CONSTRUCTION"]["file"])
    validation = load_external(review / lock["allocations"]["V2_SOURCE_VALIDATION"]["file"])
    arrays_path = review / "V2_SOURCE_ACTIVATIONS.npz"
    if not arrays_path.exists():
        arrays: dict[str, np.ndarray] = {}
        for split_name, items in (("construction", construction), ("validation", validation)):
            for axis in SOURCE_AXES:
                for polarity in SIGNS:
                    rows = [
                        _capture_source_pair(
                            backend, item, _source_instruction(axis.axis_id, polarity)
                        )
                        for item in items
                    ]
                    for location in LOCATIONS:
                        arrays[f"{split_name}__{axis.axis_id}__{polarity}__{location}"] = np.stack(
                            [row[location] for row in rows]
                        ).astype(np.float32)
        np.savez_compressed(arrays_path, **arrays)

    journal = CrashSafeJournal(
        review / "V2_SOURCE_JOURNAL.jsonl",
        identity={
            "experiment_id": EXPERIMENT_ID,
            "phase": "V2_SOURCE_QUALIFICATION",
            "source_commit": source_commit,
        },
        key_fields=("item_id", "axis_id", "polarity", "rollout_index"),
    )
    schedule = read_json(review / "V2_SOURCE_SCHEDULE.json")
    items = {item.item_id: item for item in validation}
    for row in schedule:
        key = (row["item_id"], row["axis_id"], row["polarity"], row["rollout_index"])
        if key in journal.rows:
            continue
        item = items[row["item_id"]]
        output = backend.generate_reasoning(
            model_item(item, _source_instruction(row["axis_id"], row["polarity"])),
            sampling_seed=int(row["seed"]),
            max_new_tokens=MAX_NEW_TOKENS,
            intervention_metadata={
                "experiment_id": EXPERIMENT_ID,
                "phase": "V2_SOURCE_QUALIFICATION",
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
    if len(journal.rows) != len(schedule):
        raise RuntimeError("V2 source journal is incomplete")
    finalize_source(review, source_commit)
    print(json.dumps({"phase": "source", "rows": len(journal.rows)}), flush=True)


def finalize_source(review: Path, source_commit: str) -> None:
    lock = source_lock(review)
    rows = _journal_rows(
        review / "V2_SOURCE_JOURNAL.jsonl",
        identity={
            "experiment_id": EXPERIMENT_ID,
            "phase": "V2_SOURCE_QUALIFICATION",
            "source_commit": source_commit,
        },
        keys=("item_id", "axis_id", "polarity", "rollout_index"),
    )
    archive = np.load(review / "V2_SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    source_records: dict[str, Any] = {}
    directions: dict[str, dict[str, Any]] = {}
    vector_dir = review / "V2_SOURCE_VECTORS"
    vector_dir.mkdir(parents=True, exist_ok=True)
    for axis in SOURCE_AXES:
        selected = [row for row in rows if row["axis_id"] == axis.axis_id]
        plus_rows = [row for row in selected if row["polarity"] == "PLUS"]
        minus_rows = [row for row in selected if row["polarity"] == "MINUS"]
        plus_tokens = np.asarray([row["generated_token_count"] for row in plus_rows])
        minus_tokens = np.asarray([row["generated_token_count"] for row in minus_rows])
        record: dict[str, Any] = {
            "positive_validity": float(np.mean([row["commitment_valid"] for row in plus_rows])),
            "negative_validity": float(np.mean([row["commitment_valid"] for row in minus_rows])),
            "positive_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in plus_rows])
            ),
            "negative_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in minus_rows])
            ),
            "positive_mean_tokens": float(np.mean(plus_tokens)),
            "negative_mean_tokens": float(np.mean(minus_tokens)),
            "positive_negative_mean_token_ratio": float(
                np.mean(plus_tokens) / max(np.mean(minus_tokens), 1e-12)
            ),
            "positive_minus_negative_median_tokens": float(
                np.median(plus_tokens) - np.median(minus_tokens)
            ),
            **_disagreement_metrics(rows, axis.axis_id),
            "activation": {},
        }
        direction_candidates: dict[str, tuple[np.ndarray, float]] = {}
        for location in LOCATIONS:
            construction_plus = archive[f"construction__{axis.axis_id}__PLUS__{location}"].astype(
                np.float64
            )
            construction_minus = archive[f"construction__{axis.axis_id}__MINUS__{location}"].astype(
                np.float64
            )
            direction, raw_gap, _raw = paired_mean_direction(construction_plus, construction_minus)
            validation_plus = archive[f"validation__{axis.axis_id}__PLUS__{location}"].astype(
                np.float64
            )
            validation_minus = archive[f"validation__{axis.axis_id}__MINUS__{location}"].astype(
                np.float64
            )
            gaps = (validation_plus - validation_minus) @ direction
            projections = np.concatenate(
                (validation_plus @ direction, validation_minus @ direction)
            )
            scale = float(np.std(projections, ddof=1))
            standardized_gap = float(np.mean(gaps) / max(scale, 1e-12))
            positive_gap_fraction = float(np.mean(gaps > 0))
            record["activation"][location] = {
                "construction_raw_mean_gap": float(raw_gap),
                "validation_mean_gap": float(np.mean(gaps)),
                "validation_projection_sd": scale,
                "standardized_gap": standardized_gap,
                "positive_gap_fraction": positive_gap_fraction,
                "vector_hash": vector_sha256(direction),
            }
            direction_candidates[location] = (direction, scale)
        record["source_pass"] = source_pass(record)
        if record["source_pass"]:
            for location, (direction, scale) in direction_candidates.items():
                for sign, vector in (("PLUS", direction), ("MINUS", -direction)):
                    controller = f"MEAN_{axis.axis_id}_{location}_{sign}"
                    path = vector_dir / f"{controller}.npy"
                    np.save(path, vector.astype(np.float64))
                    directions[controller] = {
                        "controller": controller,
                        "source_axis": axis.axis_id,
                        "source_location": location,
                        "sign": sign,
                        "path": str(path.relative_to(ROOT)),
                        "vector_hash": vector_sha256(vector),
                        "reference_scale": scale,
                        "source_commit": source_commit,
                    }
        source_records[axis.axis_id] = record

    qualified_axes = [axis for axis, record in source_records.items() if record["source_pass"]]
    # A source axis qualifies as a family only when both locations pass the frozen rule.
    directions = {
        name: value
        for name, value in directions.items()
        if source_records[value["source_axis"]]["source_pass"]
    }
    write_json(review / "V2_SOURCE_QUALIFICATION.json", source_records)
    write_json(
        review / "V2_SOURCE_DIRECTION_BANK.json",
        {
            "qualified_axes": qualified_axes,
            "qualified_axis_count": len(qualified_axes),
            "directions": directions,
            "correctness_used": False,
            "status": "QUALIFIED_FOR_DOSE_CALIBRATION"
            if len(qualified_axes) >= 4
            else "Q2_V2_SOURCE_BANK_TOO_NARROW",
        },
    )
    write_json(
        review / "V2_SOURCE_PROVENANCE.json",
        {
            "source_commit": source_commit,
            "source_activation_sha256": sha256(review / "V2_SOURCE_ACTIVATIONS.npz"),
            "source_journal_sha256": sha256(review / "V2_SOURCE_JOURNAL.jsonl"),
            "parser_version": PARSER_VERSION,
            "correctness_used": False,
            "qualified_axes": qualified_axes,
            "direction_count": len(directions),
            "minimum_qualified_axes": lock["source_axes"]["minimum_qualified_axes"],
        },
    )
    if len(qualified_axes) < 4:
        raise RuntimeError("Q2_V2_SOURCE_BANK_TOO_NARROW")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("source",))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution(f"Q2 V2 {args.mode}")
    review = args.review_dir.resolve()
    backend = build_backend(args.model_path)
    source_phase(backend, review, args.source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
