#!/usr/bin/env python3
"""Freeze Gate 9 items, random bank, schedule, and selected-D75 protocol."""

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

from epistemic_geometry.experiments.gate9 import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    CONTROLLER_HASH,
    DATASET_REPO,
    DATASET_REVISION,
    ETA,
    EXPERIMENT_ID,
    LAYER,
    MAX_NEW_TOKENS,
    MEANINGFUL,
    MODEL,
    MODEL_REVISION,
    PARSER_VERSION,
    RANDOM_NAMES,
    REFERENCE_SCALE,
    SELECTION_NAMESPACE,
    allocate_fresh_items,
    build_schedule,
    file_sha256,
    gate9_random_bank,
    historical_cruxeval_ids,
    vector_sha256,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
)

OUTPUT = ROOT / "review/gate9_selected_d75_evaluation"
CONTROLLER_PATH = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)
GATE8_SELECTION = ROOT / "review/gate8_l27_dose_calibration/DOSE_SELECTION.json"
GATE8_LOCK = ROOT / "review/gate8_l27_dose_calibration/PROTOCOL_LOCK.json"
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


def _require_premortem(output: Path) -> None:
    payload = json.loads((output / "PREMORTEM.json").read_text(encoding="utf-8"))
    if payload.get("classification") != "PREMORTEM_PASS":
        raise RuntimeError("Gate 9 premortem is not passed")


def _controller_identity() -> tuple[np.ndarray, dict[str, Any]]:
    selection = json.loads(GATE8_SELECTION.read_text(encoding="utf-8"))
    gate8_lock = json.loads(GATE8_LOCK.read_text(encoding="utf-8"))
    if (
        selection.get("selected_dose") != "D75"
        or selection.get("classification") != "GATE8_SAFE_LOWER_DOSE_SELECTED"
    ):
        raise RuntimeError("Gate 8 did not prospectively select D75")
    if float(gate8_lock["dose_grid"]["D75"]["eta"]) != ETA:
        raise RuntimeError("Gate 8 selected eta differs from Gate 9 D75 eta")
    vector = np.load(CONTROLLER_PATH, allow_pickle=False).astype(np.float64).reshape(-1)
    canonical_hash = vector_sha256(vector)
    if canonical_hash != CONTROLLER_HASH:
        raise RuntimeError("Gate 9 meaningful controller identity mismatch")
    if not np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-12):
        raise RuntimeError("Gate 9 meaningful controller is not unit norm")
    return vector, {
        "name": MEANINGFUL,
        "source": "PROMPT_BOUNDARY",
        "layer": LAYER,
        "constructor": "PAIRED_MEAN_DIFFERENCE",
        "sign": "PLUS",
        "dose": "D75",
        "eta": ETA,
        "reference_scale": REFERENCE_SCALE,
        "duration": "sustained_current_token",
        "scope": "final_prompt_token_then_current_decode_token",
        "vector_path": str(CONTROLLER_PATH.relative_to(ROOT)),
        "vector_file_sha256": file_sha256(CONTROLLER_PATH),
        "canonical_float64_vector_sha256": canonical_hash,
        "vector_norm": float(np.linalg.norm(vector)),
        "delta_norm": float(np.linalg.norm(vector * ETA * REFERENCE_SCALE)),
        "gate8_selection_path": str(GATE8_SELECTION.relative_to(ROOT)),
        "gate8_selection_sha256": file_sha256(GATE8_SELECTION),
        "gate8_selection_rule": selection["selection_rule"],
        "gate8_accuracy_G_C_D_used_for_selection": selection["accuracy_G_C_D_used_for_selection"],
    }


def _historical_exclusion(output: Path) -> dict[str, Any]:
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
        "reserve_ids_consumed": True,
        "gate8_calibration_excluded": True,
        "holdout_allocations_excluded_if_present": True,
        "source": "all preserved local manifests, journals, reserves, drafts, and allocations",
    }
    write_json(output / "HISTORICAL_EXCLUSION_DIGEST.json", payload)
    return payload


def freeze(candidates: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    _require_premortem(output)
    exclusion = _historical_exclusion(output)
    selected, allocation = allocate_fresh_items(candidates, exclusion["historical_ids"])
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "allocation": "GATE9_EVALUATION",
        "items": selected,
        **{key: value for key, value in allocation.items() if key != "remaining_unallocated_ids"},
        "selection_outcome_independent": True,
    }
    write_json(output / "EVALUATION_MANIFEST.json", manifest)
    write_json(
        output / "REMAINING_FRESH_AVAILABILITY.json",
        {
            "eligible_before_gate9": allocation["eligible_before_allocation"],
            "gate9_allocated": allocation["actual_n"],
            "remaining_unseen_unallocated": allocation["remaining_unallocated_n"],
            "remaining_ids": allocation["remaining_unallocated_ids"],
            "remaining_ids_allocated_or_inspected_for_outcomes": False,
        },
    )

    meaningful, controller = _controller_identity()
    bank, bank_metadata = gate9_random_bank(meaningful)
    records: dict[str, Any] = {}
    for name in RANDOM_NAMES:
        path = output / f"{name}.npy"
        np.save(path, bank[name].astype(np.float64))
        records[name] = {
            **bank_metadata["records"][name],
            "vector_path": str(path.relative_to(ROOT)),
            "vector_file_sha256": file_sha256(path),
            "canonical_float64_vector_sha256": vector_sha256(bank[name]),
            "layer": LAYER,
            "eta": ETA,
            "reference_scale": REFERENCE_SCALE,
            "duration": "sustained_current_token",
            "scope": "final_prompt_token_then_current_decode_token",
        }
    random_payload = {
        "schema_version": 1,
        "namespace": "GATE9-L27-RANDOM-BANK-V1",
        "construction": "new deterministic Gaussian orthonormal bank",
        "meaningful_controller_hash": CONTROLLER_HASH,
        "random_vectors": records,
        "geometry": bank_metadata["geometry"],
        "prior_gate_randoms_reused": False,
        "outcome_independent": True,
    }
    write_json(output / "RANDOM_BANK.json", random_payload)

    schedule = build_schedule([row["item_id"] for row in selected])
    write_json(output / "EVALUATION_SCHEDULE.json", schedule)
    parser = {
        "version": PARSER_VERSION,
        "module_sha256": file_sha256(V3_MODULE),
        "specification_sha256": file_sha256(V3_SPEC),
        "blinded_test_corpus_sha256": file_sha256(V3_CORPUS),
        "condition_invariance_test": "PASS",
        "frozen_before_gate9_outputs": True,
    }
    seconds_per_row = 6.306324623713432
    lock = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_PRE_OUTCOME",
        "lifecycle": "PROSPECTIVE_LOCK",
        "stage": "DEVELOPMENT_SELECTED_DOSE_EVALUATION",
        "lock_preparation_source_commit": git_commit(),
        "experiment_source_commit_binding": {
            "file": "EXPERIMENT_SOURCE_COMMIT.json",
            "timing": "after lock commit and before Gate-9 model outputs",
        },
        "model": {
            "id": MODEL,
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dtype": "bf16",
            "quantization": "none",
            "enable_thinking": False,
            "attention": "sdpa",
            "environment_profile": "CORE_QWEN",
            "max_new_tokens": MAX_NEW_TOKENS,
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
            **{
                key: value
                for key, value in allocation.items()
                if key != "remaining_unallocated_ids"
            },
            "manifest_file_sha256": file_sha256(output / "EVALUATION_MANIFEST.json"),
            "remaining_ids_unallocated": True,
        },
        "controller": controller,
        "random_bank": {
            "vectors": list(RANDOM_NAMES),
            "file_sha256": file_sha256(output / "RANDOM_BANK.json"),
            "records": records,
            "geometry": bank_metadata["geometry"],
        },
        "conditions": list(CONDITIONS),
        "rollouts_per_item_condition": 2,
        "seed_regime": "INDEPENDENT_PRIMARY",
        "schedule": {
            "logical_rows": len(schedule),
            "file_sha256": file_sha256(output / "EVALUATION_SCHEDULE.json"),
            "outcome_independent_interleaving": True,
            "globally_distinct_seeds": True,
        },
        "primary_outcome": "invalid_as_error; correctness only is e=0",
        "source_policy_gate": {
            "commitment_minimum": 0.90,
            "evaluability_minimum": 0.90,
            "mean_token_ratio_minimum": 1.5,
            "median_token_increase_minimum": 10,
        },
        "guards": {
            "commitment_minimum": 0.90,
            "commitment_drop_max": 0.05,
            "evaluability_minimum": 0.90,
            "evaluability_drop_max": 0.05,
            "accuracy_drop_max": 0.10,
        },
        "strong_thresholds": {
            "G_minimum": 0.10,
            "C_minimum": 0.05,
            "D_minimum": 0.08,
            "G_minus_random_mean_minimum": 0.08,
            "C_minus_random_mean_minimum": 0.05,
            "D_minus_random_mean_minimum": 0.05,
            "all_G_C_D_above_random_max": True,
            "rescue_greater_than_damage": True,
            "accuracy_gain_minimum": 0.05,
            "bootstrap_positive": [
                "accuracy_gain",
                "G",
                "C",
                "G_minus_random_mean",
                "C_minus_random_mean",
            ],
            "loo_sign_stable": ["accuracy_gain", "G", "C"],
        },
        "minimum_thresholds": {
            "positive_G_C_D": True,
            "G_C_D_above_random_mean": True,
            "rescue_greater_than_damage": True,
            "at_least_two_G_C_D_above_random_max": True,
        },
        "careful_style_control_definition": {
            "source_policy_replicated": True,
            "mean_token_gain_fraction_minimum": 0.50,
            "median_token_gain_fraction_minimum": 0.50,
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "item_cluster_all_7_conditions_both_rollouts",
            "interval": "percentile_95",
        },
        "classifications": [
            "GATE9_STRONG_SAFE_SELECTED_DOSE_REPLICATION",
            "GATE9_MINIMUM_SAFE_SELECTED_DOSE_SIGNAL",
            "GATE9_SAFE_ERROR_PROFILE_MOVEMENT_ONLY",
            "GATE9_CAREFUL_STYLE_CONTROL_WITHOUT_ERROR_CONTROL",
            "GATE9_NO_SELECTED_DOSE_EFFECT",
            "GATE9_SELECTED_DOSE_DESTRUCTIVE",
            "GATE9_SOURCE_POLICY_NOT_REPLICATED",
            "GATE9_INSTRUMENT_FAILURE",
            "GATE9_ENGINE_FAILURE",
        ],
        "cost": {
            "target_usd": 1.75,
            "hard_stop_usd": 3.50,
            "projected_rows": len(schedule),
            "conservative_gate7_seconds_per_row": seconds_per_row,
            "projected_generation_hours": len(schedule) * seconds_per_row / 3600,
            "projected_generation_cost_usd_at_0_44": len(schedule) * seconds_per_row / 3600 * 0.44,
        },
        "firewall": {
            "dose_search": "NOT_RUN",
            "controller_search": "NOT_RUN",
            "q2": "NOT_RUN",
            "character_count": "NOT_RUN",
            "confirmatory_holdout": "UNTOUCHED",
            "gate10": "NOT_RUN",
        },
    }
    write_json(output / "PROTOCOL_LOCK.json", lock)
    (output / "PROTOCOL_LOCK.md").write_text(
        "# Gate 9 prospective protocol lock\n\n"
        "Status: `FROZEN_PRE_OUTCOME`. Gate 9 evaluates the exact frozen L27 plus "
        "controller at the D75 dose selected prospectively in Gate 8. Exactly 100 fresh "
        "CRUXEval items, seven conditions, and two independent rollouts form a 1,400-row "
        "outcome-independent schedule. There is no controller, layer, dose, or item search.\n\n"
        f"Controller hash: `{CONTROLLER_HASH}`. Eta: `{ETA}`.\n\n"
        f"Manifest hash: `{allocation['manifest_hash']}`. Remaining fresh IDs: "
        f"`{allocation['remaining_unallocated_n']}`.\n\n"
        "Q2 and character count are NOT RUN; confirmatory holdout is UNTOUCHED.\n",
        encoding="utf-8",
    )
    names = (
        "PREMORTEM.md",
        "PREMORTEM.json",
        "PROTOCOL_LOCK.md",
        "PROTOCOL_LOCK.json",
        "HISTORICAL_EXCLUSION_DIGEST.json",
        "EVALUATION_MANIFEST.json",
        "RANDOM_BANK.json",
        "EVALUATION_SCHEDULE.json",
        "REMAINING_FRESH_AVAILABILITY.json",
        *(f"{name}.npy" for name in RANDOM_NAMES),
    )
    write_json(
        output / "artifact_hashes_preoutcome.json",
        {name: file_sha256(output / name) for name in names},
    )
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
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
    if args.from_items:
        payload = json.loads(args.from_items.read_text(encoding="utf-8"))
        candidates = list(payload["items"] if isinstance(payload, dict) else payload)
    elif args.remote:
        require_remote_hf_execution("Gate 9 pinned CRUXEval manifest preparation")
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
