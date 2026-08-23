#!/usr/bin/env python3
"""Prepare the outcome-free Gate 13 reused-development allocation and master lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate13  # noqa: E402
from epistemic_geometry.reproducibility import canonical_json, stable_digest  # noqa: E402

REVIEW = ROOT / "review/gate13_cross_model_ministral3"
HISTORICAL_MANIFESTS = (
    "review/full_nonthinking_smoke/cruxeval.json",
    "review/substrate_race/CRUXEVAL_MANIFEST.json",
    "review/micro_q1/CONSTRUCTION_MANIFEST.json",
    "review/micro_q1/VALIDATION_MANIFEST.json",
    "review/micro_q1/EVALUATION_MANIFEST.json",
    "review/gate5_source_duration/SOURCE_CHECK.json",
    "review/gate5_source_duration/SUSTAINED_MANIPULATION.json",
    "review/gate5_source_duration/SUSTAINED_EVALUATION.json",
    "review/gate6_layer_source_rfm_atlas/SOURCE_CANDIDATE_ORDER.json",
    "review/gate6_2_first_stage_repair_mean_bridge/MANIPULATION_MANIFEST.json",
    "review/gate6_2_first_stage_repair_mean_bridge/EVALUATION_MANIFEST.json",
    "review/gate7_fresh_l27_replication/EVALUATION_MANIFEST.json",
    "review/gate8_l27_dose_calibration/CALIBRATION_MANIFEST.json",
    "review/gate9_selected_d75_evaluation/EVALUATION_MANIFEST.json",
)
MANIFEST_NAMES = {
    "PRIMARY_8B_SUBSTRATE_SCREEN": "SUBSTRATE_SCREEN_MANIFEST.json",
    "FALLBACK_14B_SUBSTRATE_SCREEN": "FALLBACK_SCREEN_MANIFEST.json",
    "SOURCE_CONSTRUCTION": "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION": "SOURCE_VALIDATION_MANIFEST.json",
    "LAYER_FIRST_STAGE": "LAYER_FIRST_STAGE_MANIFEST.json",
    "DOSE_CALIBRATION": "DOSE_CALIBRATION_MANIFEST.json",
    "FINAL_EVALUATION": "FINAL_EVALUATION_MANIFEST.json",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nested_item_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        candidate = value.get("item", value)
        if isinstance(candidate, dict) and all(
            key in candidate for key in ("item_id", "prompt", "reference_answer")
        ):
            records.append(candidate)
        for child in value.values():
            records.extend(nested_item_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(nested_item_records(child))
    return records


def historical_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for relative in HISTORICAL_MANIFESTS:
        path = ROOT / relative
        raw = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in raw.splitlines() if line.strip()]
        found = nested_item_records(payload)
        records.extend(found)
        sources[relative] = {
            "sha256": gate13.file_sha256(path),
            "records_with_prompt_and_reference": len(found),
            "contains_scientific_outcomes": False,
        }
    return records, sources


def manifest_payload(name: str, rows: list[dict[str, Any]], pool_digest: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": gate13.EXPERIMENT_ID,
        "allocation": name,
        "n_items": len(rows),
        "dataset_repo": gate13.DATASET_REPO,
        "dataset_revision": gate13.DATASET_REVISION,
        "selection_rule": 'SHA256("GATE13-CROSS-MODEL|" + item_id)',
        "reused_development_pool_digest": pool_digest,
        "fresh_model_reused_development_items": True,
        "items": rows,
        "manifest_hash": stable_digest("GATE13-MANIFEST", name, canonical_json(rows)),
    }


def prepare(source_commit: str) -> dict[str, Any]:
    REVIEW.mkdir(parents=True, exist_ok=True)
    untouched_path = ROOT / "review/gate9_selected_d75_evaluation/REMAINING_FRESH_AVAILABILITY.json"
    untouched_payload = json.loads(untouched_path.read_text(encoding="utf-8"))
    untouched = list(map(str, untouched_payload["remaining_ids"]))
    if (
        len(untouched) != 57
        or untouched_payload["remaining_ids_allocated_or_inspected_for_outcomes"]
    ):
        raise RuntimeError("Gate 13 requires the exact sealed list of 57 untouched CRUXEval IDs")
    records, sources = historical_records()
    pool = gate13.build_reused_development_pool(records, untouched)
    allocations = gate13.allocate_pool(pool)
    pool_ids = [row["item_id"] for row in pool]
    pool_digest = stable_digest("GATE13-REUSED-DEVELOPMENT-POOL", canonical_json(pool_ids))
    pool_payload = {
        "schema_version": 1,
        "name": "GATE13_REUSED_DEVELOPMENT_POOL",
        "n_items": len(pool),
        "item_ids": pool_ids,
        "pool_digest": pool_digest,
        "historical_manifest_sources": sources,
        "historical_manifest_source_count": len(sources),
        "outcome_files_read": False,
        "confirmatory_items_included": False,
        "untouched_ids_included": False,
        "untouched_id_count": 57,
        "untouched_ids_sha256": stable_digest("GATE13-UNTOUCHED-57", canonical_json(untouched)),
    }
    write_json(REVIEW / "DEVELOPMENT_REUSE_POOL.json", pool_payload)
    manifest_hashes: dict[str, str] = {}
    for name, rows in allocations.items():
        payload = manifest_payload(name, rows, pool_digest)
        path = REVIEW / MANIFEST_NAMES[name]
        write_json(path, payload)
        manifest_hashes[name] = gate13.file_sha256(path)
    all_allocated = [row["item_id"] for rows in allocations.values() for row in rows]
    allocation_payload = {
        "schema_version": 1,
        "selection_namespace": gate13.SELECTION_NAMESPACE,
        "pool_n": len(pool),
        "allocations": {name: len(rows) for name, rows in allocations.items()},
        "allocated_item_ids": all_allocated,
        "allocated_n": len(all_allocated),
        "disjoint": len(all_allocated) == len(set(all_allocated)),
        "untouched_intersection": sorted(set(all_allocated) & set(untouched)),
        "manifest_file_sha256": manifest_hashes,
    }
    write_json(REVIEW / "ALLOCATION_MANIFEST.json", allocation_payload)
    screen = gate13.build_screen_schedule(
        [row["item_id"] for row in allocations["PRIMARY_8B_SUBSTRATE_SCREEN"]],
        gate13.PRIMARY_MODEL,
    )
    write_json(REVIEW / "SUBSTRATE_SCREEN_SCHEDULE.json", screen)

    semantic_module = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    semantic_spec = ROOT / "review/gate6_3_semantic_validity_audit/SEMANTIC_V3_SPEC.md"
    semantic_tests = ROOT / "review/gate6_3_semantic_validity_audit/SEMANTIC_V3_TESTS.json"
    parser_lock = {
        "version": gate13.PARSER_VERSION,
        "module": str(semantic_module.relative_to(ROOT)),
        "module_sha256": gate13.file_sha256(semantic_module),
        "spec_sha256": gate13.file_sha256(semantic_spec),
        "test_corpus_sha256": gate13.file_sha256(semantic_tests),
        "condition_invariance_required": True,
    }
    write_json(REVIEW / "RESPONSE_PARSER_LOCK.json", parser_lock)
    candidates = {
        "primary": {
            "id": gate13.PRIMARY_MODEL,
            "revision": gate13.PRIMARY_REVISION,
            "architecture": "Mistral3ForConditionalGeneration",
            "family": "Ministral 3 / Mistral",
            "license": "apache-2.0",
            "precision": "BF16",
            "quantization": "none",
            "gated": False,
            "parameter_count": 8_918_026_240,
        },
        "fallback": {
            "id": gate13.FALLBACK_MODEL,
            "revision": gate13.FALLBACK_REVISION,
            "architecture": "Mistral3ForConditionalGeneration",
            "family": "Ministral 3 / Mistral",
            "license": "apache-2.0",
            "precision": "BF16",
            "quantization": "none",
            "gated": False,
            "parameter_count": 13_945_031_680,
            "screen_rule": "only after MINISTRAL3_8B_COMPETENCE_FLOOR plus all Section-13A gates",
        },
    }
    write_json(REVIEW / "MODEL_CANDIDATES.json", candidates)
    engine_spec = {
        "model_loader": "AutoModelForImageTextToText",
        "text_only": True,
        "vision_inputs": False,
        "vision_tower_call_count_required": 0,
        "layer_path": gate13.MODEL_LAYER_PATH,
        "expected_layers": gate13.NUM_LAYERS,
        "expected_hidden_size": gate13.HIDDEN_SIZE,
        "engine": "serial_transformers_generate",
        "attention": "SDPA",
        "dtype": "BF16",
        "environment_profile": "CORE_MINISTRAL3",
        "generation": {
            "do_sample": True,
            "temperature": 0.30,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.0,
            "max_new_tokens": gate13.MAX_NEW_TOKENS,
            "enable_thinking": None,
            "native_no_reasoning_channel": True,
            "tools": False,
            "vision": False,
        },
    }
    write_json(REVIEW / "MODEL_ENGINE_SPEC.json", engine_spec)
    cost = {
        "wallet_balance_reported_by_principal_usd": 10.79,
        "a40_hourly_rate_assumption_usd": 0.44,
        "target_total_usd": 6.50,
        "hard_total_usd": 9.50,
        "safety_margin": 0.25,
        "maximum_trajectory_counts": {
            "substrate_screen": 300,
            "layer_first_stage_at_four_candidates": 312,
            "dose_calibration": 1760,
            "final_evaluation": 1400,
            "total_generation": 3772,
        },
        "source_activation_forwards": 192,
        "initial_projected_total_usd_with_25pct_margin": 5.75,
        "projected_under_hard_ceiling": True,
        "recompute_before_each_stage": True,
    }
    write_json(REVIEW / "COST_PROJECTION.json", cost)
    premortem = {
        "classification": "PREMORTEM_PASS",
        "risks": {
            "adapter": "multimodal wrapper must remain text-only and prove zero vision calls",
            "source_verbosity": "concise-careful and verbose-direct are diagnostic only",
            "selection_leakage": "allocation, source shortlist, layer, and dose rules are frozen",
            "freshness": "fresh-model evidence reuses consumed DEVELOPMENT items only",
            "untouched_firewall": "explicit 57-ID intersection is empty",
            "fallback": "14B is conditional only on the exact 8B floor rule",
            "cost": "25% projection recomputed before every stage",
            "resume": "stage/model/item/condition/rollout keys are append-only",
            "parser": "external-semantic-v3 code/spec/tests frozen before outputs",
        },
        "unresolved_class_d_ambiguities": [],
        "scientific_outputs_observed": False,
    }
    write_json(REVIEW / "PREMORTEM.json", premortem)
    (REVIEW / "PREMORTEM.md").write_text(
        "# Gate 13 adversarial premortem\n\n"
        "Classification: `PREMORTEM_PASS`.\n\n"
        "The multimodal wrapper is accepted only through a text-only adapter with a zero-call "
        "vision assertion. Allocation uses outcome-free historical manifests and excludes the "
        "explicit 57 untouched IDs. The layer shortlist uses source labels only; layer and dose "
        "transitions are mechanical. The 14B fallback is conditional on the frozen competence-"
        "floor rule. Journals are append-only and cost is reprojected with 25% margin before "
        "each stage. No unresolved scientific-design ambiguity remains.\n",
        encoding="utf-8",
    )
    lock = {
        "schema_version": 1,
        "experiment_id": gate13.EXPERIMENT_ID,
        "status": "FROZEN_PRE_SCREEN",
        "lifecycle": "PROSPECTIVE_MASTER_LOCK",
        "experiment_source_commit": source_commit,
        "accepted_main_head": "675e3f4a536203de107d4087234f24f3cb2fd040",
        "model_candidates": candidates,
        "engine": engine_spec,
        "instrument": {
            "dataset_repo": gate13.DATASET_REPO,
            "dataset_revision": gate13.DATASET_REVISION,
            "evaluator": parser_lock,
            "fresh_model_reused_development_items": True,
            "untouched_cruxeval_ids": 57,
        },
        "source_prompts": {
            "SOURCE_CAREFUL": gate13.SOURCE_CAREFUL,
            "SOURCE_DIRECT": gate13.SOURCE_DIRECT,
            "CAREFUL_CONCISE": gate13.CAREFUL_CONCISE,
            "VERBOSE_DIRECT": gate13.VERBOSE_DIRECT,
        },
        "allocation": allocation_payload,
        "screen": {
            "conditions": list(gate13.SCREEN_CONDITIONS),
            "items": 30,
            "rollouts": 2,
            "logical_rows": len(screen),
            "schedule_sha256": gate13.file_sha256(REVIEW / "SUBSTRATE_SCREEN_SCHEDULE.json"),
            "thresholds": {
                "validity": 0.95,
                "baseline_accuracy_min": 0.25,
                "baseline_accuracy_max": 0.85,
                "careful_minus_direct_accuracy": 0.05,
                "careful_minus_baseline_accuracy": 0.03,
                "behavioral_source": "token ratio 1.25 or median +20 or semantic change 0.15",
                "max_truncation": 0.05,
            },
        },
        "source_atlas": {
            "construction_items": 64,
            "validation_items": 32,
            "layers": "all discovered language layers",
            "activation": "final non-padding prompt-token residual output",
            "constructor": "paired careful-minus-direct mean difference",
            "eligibility": {"positive_gap_fraction": 0.80, "auroc": 0.80, "mean_gap": ">0"},
            "quartile_selection": (
                "highest held-out standardized paired effect per non-empty quartile"
            ),
            "minimum_candidates": 2,
        },
        "layer_first_stage": {
            "items": 24,
            "dose": "0.50 * held-out mean source projection gap",
            "nulls_per_layer": ["isotropic_orthogonal", "sign_shuffled_paired_mean_orthogonal"],
            "seed_regime": "MATCHED_COUPLING_CALIBRATION",
            "selection": "largest meaningful Q - null mean; then source effect; then lower layer",
        },
        "dose_calibration": {
            "items": 40,
            "conditions": 22,
            "rollouts": 2,
            "logical_rows": 1760,
            "fractions": gate13.DOSE_FRACTIONS,
            "selection": "lowest eligible dose under accepted Gate-8 gates",
        },
        "final_evaluation": {
            "items": 100,
            "conditions": 7,
            "rollouts": 2,
            "logical_rows": 1400,
            "seed_regime": "INDEPENDENT_PRIMARY",
            "estimands": ["B00", "O00", "B0j", "O0j", "G", "C", "D", "rescue", "damage"],
            "classification": (
                "accepted Gate-9 strong/minimum/safety/random rules mapped to Gate 13"
            ),
            "bootstrap_resamples": gate13.BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": gate13.BOOTSTRAP_SEED,
        },
        "cost": cost,
        "firewall": {
            "untouched_cruxeval_ids": 57,
            "q2": "NOT_RUN",
            "q3": "NOT_RUN",
            "confirmatory_holdout": "UNTOUCHED",
            "qwen_vector_transport": False,
            "geometry": "NOT_RUN",
        },
    }
    write_json(REVIEW / "MASTER_PROTOCOL_LOCK.json", lock)
    (REVIEW / "MASTER_PROTOCOL_LOCK.md").write_text(
        "# Gate 13 master protocol lock\n\n"
        f"Experiment source commit: `{source_commit}`.\n\n"
        "Fresh-model outputs use only reused DEVELOPMENT CRUXEval items. The exact 57 untouched "
        "IDs remain sealed. Ministral-3 8B is primary; 14B is conditional only on the frozen "
        "competence-floor rule. Source prompts, all allocations, source-only layer shortlist, "
        "matched first-stage, Gate-8 dose calibration, Gate-9 final classification mapping, "
        "parser, generation policy, cost ceiling, and scientific firewalls are frozen before "
        "the 8B screen.\n",
        encoding="utf-8",
    )
    preoutcome = {
        path.name: gate13.file_sha256(path)
        for path in sorted(REVIEW.iterdir())
        if path.is_file() and path.name != "artifact_hashes_preoutcome.json"
    }
    write_json(REVIEW / "artifact_hashes_preoutcome.json", preoutcome)
    return {
        "pool_n": len(pool),
        "allocated_n": len(all_allocated),
        "untouched_n": len(untouched),
        "screen_rows": len(screen),
        "experiment_source_commit": source_commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_commit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
