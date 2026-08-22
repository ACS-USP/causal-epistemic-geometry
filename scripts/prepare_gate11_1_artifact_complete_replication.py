#!/usr/bin/env python3
"""Freeze the Gate 11.1 artifact-complete forensic replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate11, gate11_1  # noqa: E402
from epistemic_geometry.reproducibility import canonical_json, stable_digest  # noqa: E402

HISTORICAL = ROOT / "review/gate11_domain_conditioned_control"
REVIEW = ROOT / "review/gate11_1_artifact_complete_replication"
VECTOR = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_frozen(name: str) -> None:
    source = HISTORICAL / name
    target = REVIEW / name
    if not source.exists():
        raise RuntimeError(f"historical Gate 11 artifact missing: {source}")
    shutil.copy2(source, target)
    if sha256(source) != sha256(target):
        raise RuntimeError(f"frozen artifact copy mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    REVIEW.mkdir(parents=True, exist_ok=True)

    required = [
        "CRUX_ITEM_SELECTION.json",
        "CHARCOUNT_ITEM_SELECTION.json",
        "FIXED_SEQUENCE_SCHEDULE.json",
        "RANDOM_BANK.json",
    ]
    for name in required:
        copy_frozen(name)
    for index in range(4):
        copy_frozen(f"GATE11_RANDOM_R{index}.npy")

    schedule = read_json(REVIEW / "FIXED_SEQUENCE_SCHEDULE.json")
    if schedule["logical_rows"] != 336 or schedule["available_item_sequences"] != 48:
        raise RuntimeError("historical Gate 11 schedule is not the frozen 336-row design")
    if schedule["conditions"] != list(gate11_1.CONDITIONS):
        raise RuntimeError("historical Gate 11 conditions changed")

    historical_names = [
        "PROTOCOL_LOCK.json",
        "EXPERIMENT_SOURCE_COMMIT.json",
        "CRUX_ITEM_SELECTION.json",
        "CHARCOUNT_ITEM_SELECTION.json",
        "FIXED_SEQUENCE_SCHEDULE.json",
        "RANDOM_BANK.json",
        "PROMPT_VARIANTS.json",
        "SOURCE_ACTIVATION_SCHEDULE.json",
        "manifest_hashes.json",
    ]
    historical_hashes = {
        name: {
            "path": str((HISTORICAL / name).relative_to(ROOT)),
            "sha256": sha256(HISTORICAL / name),
        }
        for name in historical_names
    }
    historical_hashes["controller"] = {
        "path": str(VECTOR.relative_to(ROOT)),
        "sha256": sha256(VECTOR),
        "canonical_float64_sha256": gate11.CONTROLLER_HASH,
    }
    write_json(REVIEW / "HISTORICAL_HASH_CROSSCHECK.json", historical_hashes)

    copied_hashes = {
        name: {"path": str((REVIEW / name).relative_to(ROOT)), "sha256": sha256(REVIEW / name)}
        for name in [*required, *(f"GATE11_RANDOM_R{index}.npy" for index in range(4))]
    }

    lock = {
        "schema_version": 1,
        "experiment_id": gate11_1.EXPERIMENT_ID,
        "historical_experiment_id": gate11.EXPERIMENT_ID,
        "status": "FROZEN_PRE_COLLECTION",
        "lifecycle": "PROSPECTIVE_LOCK",
        "stage": "DEVELOPMENT_FORENSIC_REPLICATION",
        "source_commit": args.source_commit,
        "source_commit_note": (
            "implementation/source commit; lock is committed as a descendant before collection"
        ),
        "model": {
            "id": gate11.MODEL,
            "revision": gate11.MODEL_REVISION,
            "tokenizer_revision": gate11.MODEL_REVISION,
            "dtype": "bf16",
            "quantization": "none",
            "attention": "sdpa",
            "enable_thinking": False,
            "environment": "CORE_QWEN",
        },
        "design": {
            "free_generation": False,
            "new_items": False,
            "new_controller": False,
            "new_dose": False,
            "new_layer": False,
            "new_random_bank": False,
            "semantic_evaluator": "NOT_RUN",
            "domains": ["CRUXEval", "CHARCOUNT"],
            "propagation_items_per_domain": gate11.PROPAGATION_ITEM_COUNT,
            "conditions": list(gate11_1.CONDITIONS),
            "logical_rows": 336,
            "sequence_cap": gate11.SEQUENCE_CAP,
            "checkpoints": list(gate11.CHECKPOINTS),
            "captured_layers": list(gate11.PROPAGATION_LAYERS),
            "source_sequences": "historical Gate-9/Gate-10 baseline rollout fallback, unchanged",
        },
        "controller": {
            "canonical_float64_hash": gate11.CONTROLLER_HASH,
            "file_sha256": sha256(VECTOR),
            "layer": gate11.LAYER,
            "eta": gate11.ETA,
            "reference_scale": gate11.REFERENCE_SCALE,
            "delta_norm": gate11.ETA * gate11.REFERENCE_SCALE,
            "duration": "sustained_current_token",
            "source": "CRUXEval careful-minus-direct prompt-boundary paired mean",
        },
        "raw_persistence": {
            "logits": "float32 complete vocabulary arrays before softmax",
            "hidden": (
                "float32 complete current-token differences relative to baseline at "
                "L27/L28/L30/L32/L35"
            ),
            "metadata": [
                "prompt_token_ids",
                "continuation_token_ids",
                "checkpoint_token_indices",
                "target_next_token_ids",
                "attention/position policy",
                "condition/vector/eta/layer/source sequence hashes",
            ],
            "storage": "one losslessly compressed shard per domain/item",
        },
        "audit": {
            "primary_reads_raw_shards_only": True,
            "independent_reads_raw_shards_only": True,
            "abs_tolerance": gate11_1.AUDIT_ABS_TOL,
            "relative_tolerance": gate11_1.AUDIT_REL_TOL,
        },
        "firewall": {
            "q2": "NOT_RUN",
            "holdout": "UNTOUCHED",
            "free_generation": "NOT_AUTHORIZED",
            "new_semantic_evaluation": "NOT_RUN",
        },
        "historical_copies": copied_hashes,
        "cost": {"target_usd": 0.60, "hard_ceiling_usd": 1.25},
        "manifest_digest": stable_digest(
            "GATE11.1-HISTORICAL-DESIGN", canonical_json(copied_hashes)
        ),
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", lock)
    (REVIEW / "PROTOCOL_LOCK.md").write_text(
        "# Gate 11.1 prospective protocol lock\n\n"
        "The historical Gate 11 design is reused byte-for-byte. Only the fixed-sequence "
        "propagation runner changes: full vocabulary logits and complete hidden-difference "
        "vectors are persisted per item shard. No free generation or new semantic outcome "
        "is authorized. The historical Gate 11 result and forensic concern remain immutable.\n",
        encoding="utf-8",
    )
    spec = {
        "experiment_id": gate11_1.EXPERIMENT_ID,
        "status": "FROZEN_PROSPECTIVE",
        "stage": "DEVELOPMENT_FORENSIC_REPLICATION",
        "free_generation": False,
        "historical_gate11_head": "55c895b12e1b8e65dac72eb8c27e312e13fd70b3",
        "propagation_items_per_domain": 24,
        "conditions": list(gate11_1.CONDITIONS),
        "captured_layers": list(gate11_1.PROPAGATION_LAYERS),
        "raw_dtype": gate11_1.RAW_DTYPE,
        "bootstrap": (
            "not required for primitive agreement; historical scalar aggregation reproduced"
        ),
        "q2": "NOT_RUN",
        "holdout": "UNTOUCHED",
        "hard_cost_ceiling_usd": 1.25,
    }
    (ROOT / "experiments/specs/gate11_1_artifact_complete_replication.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    write_json(
        REVIEW / "RAW_SHARD_MANIFEST.json",
        {
            "schema_version": gate11_1.RAW_SCHEMA_VERSION,
            "status": "PRE_COLLECTION",
            "expected_shards": 48,
            "expected_logical_rows": 336,
            "expected_conditions": list(gate11_1.CONDITIONS),
            "expected_checkpoints": list(gate11.CHECKPOINTS),
            "captured_layers": list(gate11_1.PROPAGATION_LAYERS),
            "dtype": gate11_1.RAW_DTYPE,
            "entries": [],
        },
    )
    write_json(
        REVIEW / "artifact_hashes_preoutcome.json",
        {
            name: sha256(REVIEW / name)
            for name in [
                "PROTOCOL_LOCK.json",
                "PROTOCOL_LOCK.md",
                "HISTORICAL_HASH_CROSSCHECK.json",
                "RAW_SHARD_MANIFEST.json",
                *required,
                *(f"GATE11_RANDOM_R{index}.npy" for index in range(4)),
            ]
        },
    )
    print(
        json.dumps(
            {"classification": "PREMORTEM_PASS", "expected_shards": 48, "logical_rows": 336},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
