#!/usr/bin/env python3
"""Freeze Gate 8 items, random bank, matched schedule, and protocol lock."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate7 import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    LAYER,
    MAX_NEW_TOKENS,
    MODEL,
    MODEL_REVISION,
    REFERENCE_SCALE,
)
from epistemic_geometry.experiments.gate8 import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    DOSE_FRACTIONS,
    ETA_FULL,
    EXPERIMENT_ID,
    MEANINGFUL_VECTOR,
    PARSER_VERSION,
    RANDOM_VECTOR_NAMES,
    SELECTABLE_DOSES,
    SELECTION_NAMESPACE,
    allocate_calibration_items,
    build_schedule,
    condition_spec,
    file_sha256,
    gate8_random_bank,
    historical_cruxeval_ids,
    vector_sha256,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
)

OUTPUT = ROOT / "review/gate8_l27_dose_calibration"
CONTROLLER_PATH = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)
GATE7_LOCK = ROOT / "review/gate7_fresh_l27_replication/PROTOCOL_LOCK.json"
V3_MODULE = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
V3_SPEC = ROOT / "review/gate6_3_semantic_validity_audit/SEMANTIC_V3_SPEC.md"
V3_CORPUS = ROOT / "review/gate6_3_semantic_validity_audit/BLINDED_CORPUS.jsonl"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_exclusion(output: Path) -> dict[str, Any]:
    ids = historical_cruxeval_ids(ROOT / "review", gate7_output=output)
    payload = {
        "benchmark": "CRUXEval",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "historical_ids": list(ids),
        "historical_count": len(ids),
        "historical_exclusion_digest": stable_digest(
            SELECTION_NAMESPACE, "HISTORICAL-EXCLUSION", canonical_json(ids)
        ),
        "source": "all preserved local manifests, journals, reserves, drafts, and allocations",
        "reserve_ids_consumed": True,
        "future_or_confirmatory_allocations_excluded_if_present": True,
    }
    write_json(output / "HISTORICAL_EXCLUSION_DIGEST.json", payload)
    return payload


def _controller_identity() -> tuple[np.ndarray, dict[str, Any]]:
    gate7 = json.loads(GATE7_LOCK.read_text(encoding="utf-8"))
    vector = np.load(CONTROLLER_PATH, allow_pickle=False).astype(np.float64).reshape(-1)
    canonical_hash = vector_sha256(vector)
    expected = "e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838"
    if (
        canonical_hash != expected
        or canonical_hash != gate7["controller"]["canonical_float64_vector_sha256"]
    ):
        raise RuntimeError("Gate 8 meaningful controller identity mismatch")
    if not np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-12):
        raise RuntimeError("Gate 8 meaningful controller is not unit norm")
    if float(gate7["controller"]["eta"]) != ETA_FULL:
        raise RuntimeError("Gate 7 eta differs from Gate 8 full-dose eta")
    return vector, {
        "name": MEANINGFUL_VECTOR,
        "source": "PROMPT_BOUNDARY",
        "layer": LAYER,
        "constructor": "PAIRED_MEAN_DIFFERENCE",
        "sign": "PLUS",
        "duration": "sustained_current_token",
        "vector_path": str(CONTROLLER_PATH.relative_to(ROOT)),
        "vector_file_sha256": file_sha256(CONTROLLER_PATH),
        "canonical_float64_vector_sha256": canonical_hash,
        "vector_norm": float(np.linalg.norm(vector)),
        "eta_full": ETA_FULL,
        "reference_scale": REFERENCE_SCALE,
        "full_dose_delta_norm": float(np.linalg.norm(vector * ETA_FULL * REFERENCE_SCALE)),
        "gate7_protocol": str(GATE7_LOCK.relative_to(ROOT)),
    }


def _require_premortem(output: Path) -> None:
    payload = json.loads((output / "PREMORTEM.json").read_text(encoding="utf-8"))
    if payload.get("classification") != "PREMORTEM_PASS":
        raise RuntimeError("Gate 8 premortem is not passed")


def freeze(candidates: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    _require_premortem(output)
    exclusion = write_exclusion(output)
    selected, allocation = allocate_calibration_items(candidates, exclusion["historical_ids"])
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "allocation": "GATE8_DOSE_CALIBRATION",
        "items": selected,
        **allocation,
        "selection_outcome_independent": True,
        "future_evaluation_ids_allocated": False,
    }
    write_json(output / "CALIBRATION_MANIFEST.json", manifest)
    write_json(
        output / "REMAINING_FRESH_AVAILABILITY.json",
        {
            "eligible_before_calibration": allocation["eligible_before_allocation"],
            "calibration_allocated": allocation["actual_n"],
            "remaining_unseen_unallocated": allocation["remaining_unallocated_n"],
            "minimum_required_for_future_evaluation": 100,
            "passes": allocation["remaining_unallocated_n"] >= 100,
            "future_ids_allocated_or_inspected": False,
        },
    )

    meaningful, controller = _controller_identity()
    bank, bank_metadata = gate8_random_bank(meaningful)
    random_records: dict[str, Any] = {}
    for name in RANDOM_VECTOR_NAMES:
        path = output / f"{name}.npy"
        np.save(path, bank[name].astype(np.float64))
        random_records[name] = {
            **bank_metadata["records"][name],
            "vector_path": str(path.relative_to(ROOT)),
            "vector_file_sha256": file_sha256(path),
            "canonical_float64_vector_sha256": vector_sha256(bank[name]),
            "layer": LAYER,
            "reference_scale": REFERENCE_SCALE,
            "duration": "sustained_current_token",
        }
    random_payload = {
        "schema_version": 1,
        "namespace": "GATE8-L27-RANDOM-BANK-V1",
        "construction": "new deterministic Gaussian orthonormal bank",
        "meaningful_controller": controller,
        "random_vectors": random_records,
        "geometry": bank_metadata["geometry"],
        "same_vectors_across_all_doses": True,
        "gate6_3_or_gate7_randoms_reused": False,
        "outcome_independent": True,
    }
    write_json(output / "RANDOM_BANK.json", random_payload)

    schedule = build_schedule([row["item_id"] for row in selected])
    write_json(output / "CALIBRATION_SCHEDULE.json", schedule)
    parser = {
        "version": PARSER_VERSION,
        "module_sha256": file_sha256(V3_MODULE),
        "specification_sha256": file_sha256(V3_SPEC),
        "blinded_test_corpus_sha256": file_sha256(V3_CORPUS),
        "condition_invariance_test": "PASS",
        "frozen_before_gate8_outputs": True,
    }
    dose_grid = {
        dose: {
            "fraction": fraction,
            "eta": ETA_FULL * fraction,
            "selectable": dose in SELECTABLE_DOSES,
        }
        for dose, fraction in DOSE_FRACTIONS.items()
    }
    projected_seconds_per_row = 6.306324623713432
    lock = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_PRE_OUTCOME",
        "lifecycle": "PROSPECTIVE_LOCK",
        "stage": "DEVELOPMENT_CALIBRATION",
        "lock_preparation_source_commit": git_commit(),
        "experiment_source_commit_binding": {
            "file": "EXPERIMENT_SOURCE_COMMIT.json",
            "timing": "after lock commit and before Gate-8 model outputs",
        },
        "model": {
            "id": MODEL,
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dtype": "bf16",
            "quantization": "none",
            "enable_thinking": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "attention": "sdpa",
            "environment_profile": "CORE_QWEN",
            "sampling": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
            },
        },
        "instrument": {
            "benchmark": "CRUXEval semantic output prediction",
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "evaluator": parser,
        },
        "sample": {
            **allocation,
            "manifest_file_sha256": file_sha256(output / "CALIBRATION_MANIFEST.json"),
            "future_evaluation_ids_allocated": False,
        },
        "controller": controller,
        "dose_grid": dose_grid,
        "random_bank": {
            "vectors": list(RANDOM_VECTOR_NAMES),
            "file_sha256": file_sha256(output / "RANDOM_BANK.json"),
            "records": random_records,
            "geometry": bank_metadata["geometry"],
        },
        "conditions": list(CONDITIONS),
        "condition_specs": {condition: condition_spec(condition) for condition in CONDITIONS},
        "rollouts_per_item_condition": 2,
        "seed_regime": "MATCHED_COUPLING_CALIBRATION",
        "schedule": {
            "logical_rows": len(schedule),
            "file_sha256": file_sha256(output / "CALIBRATION_SCHEDULE.json"),
            "outcome_independent_interleaving": True,
            "same_seed_within_item_rollout_block": True,
        },
        "primary_outcome": "invalid_as_error; correctness only is e=0",
        "source_gate": {
            "commitment_minimum": 0.90,
            "evaluability_minimum": 0.90,
            "mean_token_ratio_minimum": 1.5,
            "median_token_increase_minimum": 10,
        },
        "safety_guards": {
            "commitment_minimum": 0.90,
            "commitment_drop_max": 0.05,
            "evaluability_minimum": 0.90,
            "evaluability_drop_max": 0.05,
            "accuracy_drop_max": 0.10,
        },
        "first_stage_gate": {
            "Q_minimum": 0.15,
            "Q_minus_random_mean_minimum": 0.05,
            "Q_greater_than_random_max": True,
            "rho_tokens_minimum": 0.25,
            "rho_tokens_maximum": 1.25,
        },
        "selection_rule": {
            "selectable": list(SELECTABLE_DOSES),
            "priority": list(SELECTABLE_DOSES),
            "D100_selectable": False,
            "objective": "lowest eligible lower dose",
            "accuracy_G_C_D_forbidden_as_optimization_objectives": True,
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "item_cluster_all_22_conditions_both_rollout_blocks",
            "interval": "percentile_95",
        },
        "classifications": [
            "GATE8_SAFE_LOWER_DOSE_SELECTED",
            "GATE8_ORIGINAL_DOSE_ONLY_SPECIFIC",
            "GATE8_EFFECT_VALIDITY_TRADEOFF_CONFIRMED",
            "GATE8_LOWER_DOSES_NONSPECIFIC_OR_INERT",
            "GATE8_SOURCE_POLICY_NOT_REPLICATED",
            "GATE8_INSTRUMENT_FAILURE",
            "GATE8_ENGINE_FAILURE",
        ],
        "cost": {
            "target_usd": 2.50,
            "hard_stop_usd": 4.00,
            "projected_rows": len(schedule),
            "conservative_gate7_seconds_per_row": projected_seconds_per_row,
            "projected_generation_hours": len(schedule) * projected_seconds_per_row / 3600,
            "projected_generation_cost_usd_at_0_44": (
                len(schedule) * projected_seconds_per_row / 3600 * 0.44
            ),
        },
        "firewall": {
            "calibration_only": True,
            "future_dose_evaluation": "NOT_RUN",
            "G_C_D_primary_evidence": "NOT_RUN",
            "q2": "NOT_RUN",
            "character_count": "NOT_RUN",
            "confirmatory_holdout": "UNTOUCHED",
        },
    }
    write_json(output / "PROTOCOL_LOCK.json", lock)
    (output / "PROTOCOL_LOCK.md").write_text(
        "# Gate 8 prospective protocol lock\n\n"
        "Status: `FROZEN_PRE_OUTCOME`. Gate 8 is calibration only. It reuses the exact "
        "frozen L27 plus controller and changes only the prospectively fixed scalar dose. "
        "The 50-item matched-coupling schedule contains 2,200 rows across 22 conditions. "
        "Dose selection uses safety, semantic Q specificity, and CAREFUL token recovery; "
        "accuracy and G/C/D are not optimization objectives. D100 is diagnostic and cannot "
        "be selected. No future evaluation IDs are allocated.\n\n"
        f"Controller hash: `{controller['canonical_float64_vector_sha256']}`.\n\n"
        f"Calibration manifest hash: `{allocation['manifest_hash']}`.\n\n"
        f"Remaining unseen/unallocated IDs: `{allocation['remaining_unallocated_n']}`.\n\n"
        "Q2 and character count are NOT RUN; confirmatory holdout is UNTOUCHED.\n",
        encoding="utf-8",
    )
    names = (
        "PREMORTEM.md",
        "PREMORTEM.json",
        "PROTOCOL_LOCK.md",
        "PROTOCOL_LOCK.json",
        "HISTORICAL_EXCLUSION_DIGEST.json",
        "CALIBRATION_MANIFEST.json",
        "RANDOM_BANK.json",
        "CALIBRATION_SCHEDULE.json",
        "REMAINING_FRESH_AVAILABILITY.json",
        *(f"{name}.npy" for name in RANDOM_VECTOR_NAMES),
    )
    write_json(
        output / "artifact_hashes_preoutcome.json",
        {name: file_sha256(output / name) for name in names},
    )
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--write-exclusion-only", action="store_true")
    parser.add_argument("--from-items", type=Path)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--bind-source-commit")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.bind_source_commit:
        if git_commit() != args.bind_source_commit:
            raise RuntimeError("source binding must equal current checkout")
        payload = {
            "experiment_source_commit": args.bind_source_commit,
            "bound_before_model_outputs": True,
            "protocol_lock_sha256": file_sha256(output / "PROTOCOL_LOCK.json"),
        }
        write_json(output / "EXPERIMENT_SOURCE_COMMIT.json", payload)
        print(json.dumps(payload, indent=2))
        return 0
    if args.write_exclusion_only:
        payload = write_exclusion(output)
        print(json.dumps({"historical_count": payload["historical_count"]}, indent=2))
        return 0
    if args.from_items:
        payload = json.loads(args.from_items.read_text(encoding="utf-8"))
        candidates = list(payload["items"] if isinstance(payload, dict) else payload)
    elif args.remote:
        require_remote_hf_execution("Gate 8 pinned CRUXEval manifest preparation")
        from datasets import load_dataset

        candidates = list(load_dataset(DATASET_REPO, split="test", revision=DATASET_REVISION))
    else:
        raise SystemExit("provide --from-items or --remote on authorized host")
    lock = freeze(candidates, output)
    print(
        json.dumps(
            {
                "status": lock["status"],
                "actual_n": lock["sample"]["actual_n"],
                "remaining": lock["sample"]["remaining_unallocated_n"],
                "logical_rows": lock["schedule"]["logical_rows"],
                "projected_cost_usd": lock["cost"]["projected_generation_cost_usd_at_0_44"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
