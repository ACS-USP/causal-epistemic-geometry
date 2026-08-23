#!/usr/bin/env python3
"""Prepare the prospective Gate 13.1 split, manifests, and protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate13, gate13_1  # noqa: E402
from epistemic_geometry.reproducibility import stable_seed  # noqa: E402

PARENT = ROOT / "review/gate13_cross_model_ministral3"
REVIEW = ROOT / "review/gate13_1_all_layer_causal_atlas"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def manifest(name: str, items: list[dict[str, Any]], source: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": gate13_1.EXPERIMENT_ID,
        "allocation": name,
        "n_items": len(items),
        "items": items,
        "source_allocation": source,
        "fresh_relative_to_gate13_1_causal_measurement": True,
        "historical_outcomes_used_for_selection": False,
        "dataset_repo": gate13.DATASET_REPO,
        "dataset_revision": gate13.DATASET_REVISION,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", default=git_head())
    args = parser.parse_args()
    audit = read_json(PARENT / "FORENSIC_AUDIT.json")
    if audit["classification"] not in {
        "GATE13_FORENSIC_CLEAN",
        "GATE13_FORENSIC_MINOR_NONSCIENTIFIC_ISSUES",
    }:
        raise RuntimeError("Gate 13.1 requires a clean Gate-13 forensic closeout")
    if read_json(PARENT / "LAYER_FIRST_STAGE_REPORT.json")["classification"] != (
        "GATE13_NO_CAUSAL_LAYER_FIRST_STAGE"
    ):
        raise RuntimeError("Gate 13 historical classification mismatch")

    REVIEW.mkdir(parents=True, exist_ok=True)
    parent_development = read_json(PARENT / "DOSE_CALIBRATION_MANIFEST.json")["items"]
    sweep, qualification = gate13_1.split_development_items(parent_development)
    final_items = read_json(PARENT / "FINAL_EVALUATION_MANIFEST.json")["items"]
    if len(final_items) != 100:
        raise RuntimeError("Gate 13.1 final evaluation allocation is not exactly 100 items")
    sets = [
        {str(row["item_id"]) for row in sweep},
        {str(row["item_id"]) for row in qualification},
        {str(row["item_id"]) for row in final_items},
    ]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Gate 13.1 item firewall overlap")

    write_json(
        REVIEW / "ALL_LAYER_SWEEP_ITEMS.json",
        manifest("ALL_LAYER_SWEEP", sweep, "GATE13_DOSE_CALIBRATION"),
    )
    write_json(
        REVIEW / "LAYER_DOSE_ITEMS.json",
        manifest("LAYER_DOSE_QUALIFICATION", qualification, "GATE13_DOSE_CALIBRATION"),
    )
    write_json(
        REVIEW / "FINAL_EVALUATION_ITEMS.json",
        manifest("FINAL_EVALUATION", final_items, "GATE13_FINAL_EVALUATION"),
    )
    write_json(
        REVIEW / "ALL_LAYER_SWEEP_SCHEDULE.json",
        gate13_1.build_sweep_schedule([str(row["item_id"]) for row in sweep]),
    )

    atlas = read_json(PARENT / "SOURCE_ATLAS.json")["layers"]
    source_records = []
    for row in atlas:
        layer = int(row["layer"])
        path = PARENT / f"SOURCE_DIRECTIONS/L{layer}.npy"
        source_records.append(
            {
                "layer": layer,
                "vector_path": str(path.relative_to(ROOT)),
                "file_sha256": sha256(path),
                "canonical_vector_hash": row["direction_hash"],
                "D100": float(row["paired_mean_gap"]),
                "source_effect": float(row["standardized_paired_effect"]),
                "source_auroc": float(row["auroc"]),
                "source_eligible": bool(row["source_eligible"]),
            }
        )
    activation_archive = PARENT / "SOURCE_ACTIVATIONS.npz"
    write_json(
        REVIEW / "SOURCE_DIRECTION_MANIFEST.json",
        {
            "model": gate13_1.MODEL,
            "revision": gate13_1.REVISION,
            "layers": source_records,
            "all_layers_source_eligible": all(row["source_eligible"] for row in source_records),
            "source_activation_archive": str(activation_archive.relative_to(ROOT)),
            "source_activation_archive_sha256": sha256(activation_archive),
            "directions_rebuilt": False,
            "source_activations_recollected": False,
        },
    )
    write_json(
        REVIEW / "DOSE_DEFINITION.json",
        {
            "D100_definition": "historical held-out mean careful-minus-direct projection gap",
            "fractions": gate13_1.DOSE_FRACTIONS,
            "per_layer": {
                f"L{row['layer']}": {
                    dose: float(row["D100"]) * fraction
                    for dose, fraction in gate13_1.DOSE_FRACTIONS.items()
                }
                for row in source_records
            },
        },
    )
    write_json(
        REVIEW / "SWEEP_SELECTION_RULE.json",
        {
            "layers": list(range(34)),
            "dose": "D50",
            "candidate_generation_only": True,
            "eligibility": {
                "commitment_validity_min": 0.75,
                "semantic_evaluability_min": 0.75,
                "Q_min": 0.10,
            },
            "selection": "maximum Q within each historical depth quartile; lower layer tie-break",
            "correctness_used_for_ranking": False,
            "minimum_candidates_to_continue": 2,
        },
    )
    write_json(
        REVIEW / "LAYER_DOSE_SELECTION_RULE.json",
        {
            "doses": list(gate13_1.DOSE_FRACTIONS),
            "nulls_per_cell": ["ISOTROPIC_NULL", "SHUFFLED_NULL"],
            "eligibility": {
                "commitment_validity_min": 0.90,
                "semantic_evaluability_min": 0.90,
                "competence_tolerance": -0.10,
                "Q_min": 0.15,
                "Q_minus_null_mean_min": 0.05,
                "Q_strictly_above_null_max": True,
            },
            "within_layer": "lowest eligible dose",
            "between_layers": "maximum Q-minus-null-mean",
            "tie_breaks": [
                "larger Q",
                "larger historical source effect",
                "lower dose fraction",
                "lower layer index",
            ],
            "accuracy_used_for_ranking": False,
        },
    )
    random_seeds = {
        f"L{layer}": {
            "stage_b_isotropic": stable_seed(
                gate13_1.EXPERIMENT_ID, "STAGE_B_ISOTROPIC", layer
            ),
            "stage_b_shuffled": stable_seed(
                gate13_1.EXPERIMENT_ID, "STAGE_B_SHUFFLED", layer
            ),
            "final": {
                f"R{index}": stable_seed(
                    gate13_1.EXPERIMENT_ID,
                    "FINAL_ISOTROPIC" if index < 2 else "FINAL_SHUFFLED",
                    layer,
                    index,
                )
                for index in range(4)
            },
        }
        for layer in range(34)
    }
    write_json(
        REVIEW / "COST_PROJECTION.json",
        {
            "a40_hourly_rate_usd": 0.45,
            "historical_gate13_cost_usd_estimate": 0.90,
            "stage_a_max_trajectories": 420,
            "stage_b_max_trajectories": 1372,
            "stage_c_max_trajectories": 1400,
            "maximum_gate13_1_trajectories": 3192,
            "target_incremental_usd": 4.50,
            "hard_incremental_usd": 7.00,
            "minimum_wallet_reserve_usd": 0.50,
            "wallet_balance_pre_lock_usd": 9.89,
            "projected_incremental_usd_with_25pct_margin": 3.60,
            "passes_pre_stage_a_cost_gate": True,
        },
    )
    premortem = {
        "classification": "PREMORTEM_PASS",
        "risks": {
            "post_hoc_L22_L26_cherry_picking": "all 34 layers required",
            "decode_control_conflation": "source metrics cannot select causal layer",
            "dose_layer_interaction": "Stage A fixed D50; Stage B joint prospective grid",
            "small_sweep_sample": "candidate generation only",
            "null_specificity": "two matched null families per Stage-B cell",
            "safety": "validity/evaluability/competence gates frozen",
            "outcome_leakage": "final 100 items remain unopened until cell lock",
            "model_shopping": "Ministral-3 8B only; no fallback",
            "budget": "reprojection before every later stage",
        },
        "unresolved_class_a": [],
        "unresolved_class_b": [],
    }
    write_json(REVIEW / "PREMORTEM.json", premortem)
    (REVIEW / "PREMORTEM.md").write_text(
        "# Gate 13.1 adversarial premortem\n\n"
        "Classification: `PREMORTEM_PASS`. All 34 layers are tested; source readout cannot "
        "select the causal layer; Stage A is candidate generation only; Stage B separates "
        "layer and dose with isotropic and construction-shuffled nulls; safety, item, model, "
        "and cost firewalls are frozen before output.\n",
        encoding="utf-8",
    )

    lock = {
        "schema_version": 1,
        "status": "FROZEN_PRE_STAGE_A",
        "lifecycle": "PROSPECTIVE_LOCK",
        "experiment_id": gate13_1.EXPERIMENT_ID,
        "accepted_main_head": "a3f68a52f76ac68170d99b1fcc1a3e63f78bbe55",
        "experiment_source_commit": args.source_commit,
        "historical_gate13": {
            "classification": "GATE13_NO_CAUSAL_LAYER_FIRST_STAGE",
            "forensic": audit["classification"],
            "result_modified": False,
        },
        "model": {
            "id": gate13_1.MODEL,
            "revision": gate13_1.REVISION,
            "tokenizer_revision": gate13_1.REVISION,
            "dtype": "BF16",
            "engine": "serial Transformers",
            "environment": "CORE_MINISTRAL3",
            "fix_mistral_regex": True,
        },
        "source_direction_manifest_sha256": sha256(REVIEW / "SOURCE_DIRECTION_MANIFEST.json"),
        "item_manifests": {
            name: sha256(REVIEW / name)
            for name in (
                "ALL_LAYER_SWEEP_ITEMS.json",
                "LAYER_DOSE_ITEMS.json",
                "FINAL_EVALUATION_ITEMS.json",
            )
        },
        "stage_a": {
            "items": 12,
            "layers": list(range(34)),
            "dose": "D50",
            "trajectories": 420,
            "seed_regime": "MATCHED_COUPLING",
        },
        "stage_b": {
            "items": 28,
            "doses": list(gate13_1.DOSE_FRACTIONS),
            "maximum_trajectories": 1372,
            "seed_regime": "MATCHED_COUPLING",
            "random_seeds": random_seeds,
        },
        "stage_c": {
            "items": 100,
            "conditions": 7,
            "rollouts": 2,
            "trajectories": 1400,
            "seed_regime": "INDEPENDENT_PRIMARY",
            "bootstrap_seed": gate13_1.BOOTSTRAP_SEED,
            "bootstrap_resamples": gate13_1.BOOTSTRAP_RESAMPLES,
        },
        "cost": read_json(REVIEW / "COST_PROJECTION.json"),
        "firewall": {
            "untouched_cruxeval_ids": 57,
            "q2": "NOT_RUN",
            "q3": "NOT_RUN",
            "holdout": "UNTOUCHED",
            "no_other_model": True,
        },
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", lock)
    (REVIEW / "PROTOCOL_LOCK.md").write_text(
        "# Gate 13.1 prospective protocol lock\n\n"
        f"Source commit: `{args.source_commit}`. All 34 layers, the D25/D50/D75/D100 grid, "
        "item partitions, null-generation namespaces, safety gates, selection rules, final "
        "evaluation, bootstrap, and cost ceiling are frozen before Stage-A outputs.\n",
        encoding="utf-8",
    )
    write_json(
        REVIEW / "HISTORICAL_GATE13_CLOSEOUT.json",
        {
            "accepted_head": "a3f68a52f76ac68170d99b1fcc1a3e63f78bbe55",
            "classification": "GATE13_NO_CAUSAL_LAYER_FIRST_STAGE",
            "forensic": audit["classification"],
            "historical_result_modified": False,
            "volume_handoff": "PRINCIPAL_AUTHORIZED_GATE13_TO_GATE13_1",
        },
    )
    spec = {
        "id": gate13_1.EXPERIMENT_ID,
        "status": "FROZEN_PROSPECTIVE_LOCK",
        "stage": "DEVELOPMENT_CROSS_MODEL_LAYER_DOSE_IDENTIFICATION",
        "model": gate13_1.MODEL,
        "revision": gate13_1.REVISION,
        "source_commit": args.source_commit,
        "protocol": "review/gate13_1_all_layer_causal_atlas/PROTOCOL_LOCK.json",
        "scientific_firewall": {
            "untouched_cruxeval_ids": 57,
            "q2": "NOT_RUN",
            "q3": "NOT_RUN",
            "holdout": "UNTOUCHED",
        },
    }
    spec_path = ROOT / "experiments/specs/gate13_1_all_layer_causal_atlas.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    hashes = {
        path.name: sha256(path)
        for path in sorted(REVIEW.iterdir())
        if path.is_file() and path.name != "artifact_hashes_preoutcome.json"
    }
    write_json(REVIEW / "artifact_hashes_preoutcome.json", hashes)
    print(json.dumps({"source_commit": args.source_commit, "review": str(REVIEW)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
