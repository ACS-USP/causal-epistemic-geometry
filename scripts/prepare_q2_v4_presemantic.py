#!/usr/bin/env python3
"""Materialize the outcome-free Q2 V4 Spark-1 qualification lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.q2_v3_prompt_provenance import (  # noqa: E402
    canonical_q2_v3_task_prompt,
)
from epistemic_geometry.experiments.q2_v4_presemantic import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    CANDIDATE_COUNT,
    DATASET_REPO,
    DATASET_REVISION,
    LAYER,
    LOCATIONS,
    MODEL,
    MODEL_REVISION,
    PRIMARY_N,
    QAP_MAPS,
    SELECTED_COUNT,
    SHELL_TARGETS,
    SOURCE_FAMILIES,
    deterministic_seed,
)

REVIEW = ROOT / "review/q2_v4_spark1_presemantic"
V3_FREEZE = ROOT / "review/q2_v3_amendment1_freeze"
V3_PANEL = ROOT / "review/q2_v3_four_family_statistical_redesign"
OFFICIAL = ROOT / "review/q2_v3_provenance_reconciliation/OFFICIAL_SOURCE_RECORDS.jsonl"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def inherited_manifest(name: str, *, allocation: str) -> dict[str, Any]:
    source = json.loads((V3_FREEZE / name).read_text())
    return {
        **source,
        "schema_version": "q2-v4-spark1-allocation-manifest-v1",
        "status": "FROZEN_PRESEMANTIC",
        "allocation": allocation,
        "v4_native_spark1": True,
        "inherited_source_path": str((V3_FREEZE / name).relative_to(ROOT)),
        "inherited_source_sha256": sha256(V3_FREEZE / name),
        "outcome_values_read_or_used": False,
    }


def official_rows(dataset_path: Path) -> dict[str, dict[str, Any]]:
    if not dataset_path.is_file():
        raise RuntimeError(f"exact official dataset file is missing: {dataset_path}")
    rows: dict[str, dict[str, Any]] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for official_index, line in enumerate(handle):
            row = json.loads(line)
            item_id = str(row.get("id", f"sample_{official_index}"))
            rows[item_id] = {
                **row,
                "id": item_id,
                "official_index": official_index,
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
            }
    if len(rows) != 800:
        raise RuntimeError("exact CRUXEval revision must contain 800 unique rows")
    tracked = {
        str(row["id"]): row
        for row in (json.loads(line) for line in OFFICIAL.read_text(encoding="utf-8").splitlines())
    }
    for item_id, expected in tracked.items():
        observed = rows[item_id]
        if any(str(observed[key]) != str(expected[key]) for key in ("code", "input", "output")):
            raise RuntimeError(f"official dataset mismatch against tracked record: {item_id}")
    return rows


def primary_panel_manifest(dataset_path: Path) -> dict[str, Any]:
    design = json.loads((V3_PANEL / "FOUR_FAMILY_PRIMARY_PANEL_MANIFEST.json").read_text())
    source = official_rows(dataset_path)
    selected = list(design["selected_ids"])
    if len(selected) != PRIMARY_N or len(set(selected)) != PRIMARY_N:
        raise RuntimeError("V4 primary selection is not exactly 300 unique IDs")
    items = []
    for order, item_id in enumerate(selected):
        row = source[item_id]
        prompt = canonical_q2_v3_task_prompt(str(row["code"]), str(row["input"]))
        reference = str(row["output"])
        items.append(
            {
                "item_id": item_id,
                "order": order,
                "official_index": int(row["official_index"]),
                "provenance_class": "C",
                "role": "PRIMARY_SEMANTIC_PANEL_FUTURE_NOT_RUN",
                "prompt": prompt,
                "reference_answer": reference,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
            }
        )
    return {
        "schema_version": "q2-v4-primary-panel-v1",
        "status": "FROZEN_CONTENT_NOT_AUTHORIZED_FOR_INFERENCE",
        "evidence_class": "historical-item/prospective-controller same-domain validation",
        "provenance_class": "C",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_file_sha256": sha256(dataset_path),
        "selection_rule": "exact inherited deterministic 300-item V3-design order",
        "inherited_manifest": str(
            (V3_PANEL / "FOUR_FAMILY_PRIMARY_PANEL_MANIFEST.json").relative_to(ROOT)
        ),
        "inherited_manifest_sha256": sha256(V3_PANEL / "FOUR_FAMILY_PRIMARY_PANEL_MANIFEST.json"),
        "item_count": len(items),
        "item_ids": selected,
        "items": items,
        "semantic_outcomes": 0,
    }


def technical_fixtures() -> dict[str, Any]:
    texts = (
        "Return exactly FINAL: ok.",
        "Repeat the token alpha twice and finish with FINAL: alpha alpha.",
        "Punctuation fixture: []{}()!?; finish with FINAL: punctuation.",
        "Numeric fixture 000123456789; finish with FINAL: 123456789.",
        "Code-like fixture: for i in range(3): pass; finish with FINAL: done.",
        "Unicode-neutral fixture cafe naive resume; finish with FINAL: done.",
        "Short prompt. Finish with FINAL: short.",
        "A longer deterministic engineering prompt containing repeated words repeated words "
        "and no benchmark question. Finish with FINAL: long.",
        "List A B C D E and finish with FINAL: E.",
        "Whitespace fixture with one sentence. Finish with FINAL: whitespace.",
        "Cache-position fixture. Produce a brief preface then FINAL: cache.",
        "Intervention-scope fixture. Produce two brief clauses then FINAL: scope.",
    )
    return {
        "schema_version": "q2-v4-technical-fixtures-v1",
        "scientific_items": False,
        "correctness_oracle": None,
        "fixtures": [
            {
                "fixture_id": f"TECH_{i:02d}",
                "prompt": text,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            for i, text in enumerate(texts)
        ],
    }


def source_schedule(validation: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for item_id in validation["item_ids"]:
        for family in SOURCE_FAMILIES:
            for polarity in ("POSITIVE", "NEGATIVE"):
                for rollout in (0, 1):
                    rows.append(
                        {
                            "item_id": item_id,
                            "family": family,
                            "polarity": polarity,
                            "rollout_index": rollout,
                            "seed": deterministic_seed(
                                "Q2-V4-SPARK1-SOURCE", item_id, family, polarity, rollout
                            ),
                        }
                    )
    return {"schema_version": "q2-v4-source-schedule-v1", "rows": rows}


def protocol_lock(source_commit: str) -> dict[str, Any]:
    return {
        "schema_version": "q2-v4-spark1-qualification-lock-v1",
        "status": "Q2_V4_SPARK1_QUALIFICATION_PROTOCOL_FROZEN",
        "source_commit": source_commit,
        "backend": "V4_NATIVE_SPARK1",
        "spark1_only": True,
        "spark2_forbidden": True,
        "max_gb10": 1,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dtype": "BF16",
        "attention": "SDPA",
        "layer": LAYER,
        "source_families": list(SOURCE_FAMILIES),
        "source_locations": list(LOCATIONS),
        "source_gate": {
            "validity_min": 0.90,
            "evaluability_min": 0.90,
            "cross_disagreement_min": 0.10,
            "excess_disagreement_min": 0.03,
            "raw_norm_min": 1e-6,
            "standardized_gap_min": 0.20,
            "positive_projection_fraction_min": 0.60,
            "all_four_families_required": True,
        },
        "subspace_gate": {
            "relative_singular_threshold": 1e-6,
            "retained_rank_min": 6,
            "condition_number_max": 10.0,
            "concept_leverage_min": 0.01,
        },
        "candidate_policy": {
            "count": CANDIDATE_COUNT,
            "rng": "NumPy PCG64DXSM",
            "seed_rule": (
                "big-endian first 128 bits SHA256('Q2-V4-INTERVENTION-SUBSPACE-"
                "DIRECTIONS-V1|' + PRELOCK_COMMIT)"
            ),
            "draw": "g~N(0,I_r); c=g/||g||; v=Qc",
            "redraw": "FORBIDDEN",
            "algebraic_gate": {
                "coefficient_norm_error_max": 1e-12,
                "vector_norm_error_max": 1e-10,
                "rank": "retained subspace rank",
                "entropy_effective_rank_min_fraction": 0.75,
                "condition_number_max": 3.0,
                "max_absolute_pair_cosine": 0.98,
            },
        },
        "shell_calibration": {
            "targets": SHELL_TARGETS,
            "bounds": [0.0, 256.0],
            "iterations": 40,
            "BF16_aware": True,
            "relative_target_error_max": 0.005,
        },
        "safety_gate": {
            "items": 12,
            "rollouts": 2,
            "matched_seed_with_baseline": True,
            "validity_min": 0.90,
            "relative_validity_drop_max": 0.05,
            "evaluability_min": 0.90,
            "relative_evaluability_drop_max": 0.05,
            "truncation_max": 0.05,
            "movement_min": {"MEDIUM": 0.10, "STRONG": 0.15},
            "candidate_requires_both_shells": True,
            "selection": "first 32 safe in generation order",
            "minimum_safe": SELECTED_COUNT,
            "candidates_41_plus": "FORBIDDEN",
            "correctness": "FORBIDDEN",
        },
        "selected_bank_gate": {
            "rank": "retained subspace rank",
            "entropy_effective_rank_min_fraction": 0.75,
            "condition_number_max": 3.0,
            "max_absolute_pair_cosine": 0.98,
            "A0_q90_minus_q10_min": 0.20,
            "shell_amplitude_cv_max": 0.03,
        },
        "A1": {"covariance_items": 64, "shrinkage_lambda": 0.10},
        "A2": {
            "natural_log_JS": True,
            "probe_items": 12,
            "checkpoints_per_probe": 4,
            "full_vocabulary": True,
            "noise_floor_squared": "max(1e-12,100*max repeated-baseline mean-JS)",
            "symmetry_error_max": 1e-12,
            "diagonal_error_max": 1e-12,
            "baseline_identity_error_max": 1e-10,
            "gram_min_eigenvalue_min": -1e-8,
            "cosine_tolerance": 1e-8,
            "repeat_radius_relative_error_max": 1e-6,
            "repeat_distance_relative_error_max": 1e-6,
            "repeat_angular_spearman_min": 0.999,
            "failure_downgrade": "FORBIDDEN",
        },
        "primary_panel": {"N": PRIMARY_N, "class": "C", "rollouts": 2},
        "shape_estimator": "N/(N-1)*(Dtotal-m0*m1); negative values retained",
        "QAP": {
            "maps_total": QAP_MAPS,
            "identity_included": True,
            "same_controller_permutation_across_shells_and_metrics": True,
            "p_value": "count(T_perm>=T_obs)/50000",
            "multiplicity": "single-step maxT over A0/A1/A2",
            "G3_margin": 0.10,
        },
        "bootstrap": {"item_cluster_resamples": BOOTSTRAP_RESAMPLES},
        "radial": {
            "R_total": "Dtotal(BASELINE,strong)-Dtotal(BASELINE,medium)",
            "R_shape": "Dshape(BASELINE,strong)-Dshape(BASELINE,medium)",
            "maps_total": QAP_MAPS,
            "paired_direction_shell_swaps": True,
            "positive_direction_count_min": 22,
            "independent_statuses": ["RT+/-", "RS+/-"],
        },
        "semantic_inference_authorized": False,
        "Q3": "NOT_RUN",
        "presemantic_gpu_hour_soft_ceiling": 8.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-dataset",
        type=Path,
        required=True,
        help="test.jsonl from the exact frozen CRUXEval dataset revision",
    )
    args = parser.parse_args()
    REVIEW.mkdir(parents=True, exist_ok=True)
    source_commit = git_head()
    manifests = {
        "SOURCE_CONSTRUCTION_MANIFEST.json": inherited_manifest(
            "SOURCE_CONSTRUCTION_MANIFEST.json", allocation="SOURCE_CONSTRUCTION"
        ),
        "SOURCE_VALIDATION_MANIFEST.json": inherited_manifest(
            "SOURCE_VALIDATION_MANIFEST.json", allocation="SOURCE_VALIDATION"
        ),
        "SHELL_CALIBRATION_MANIFEST.json": inherited_manifest(
            "SHELL_CALIBRATION_MANIFEST.json", allocation="SHELL_CALIBRATION"
        ),
        "M1_COVARIANCE_MANIFEST.json": inherited_manifest(
            "M1_COVARIANCE_MANIFEST.json", allocation="M1_COVARIANCE"
        ),
        "M2_PROBE_MANIFEST.json": inherited_manifest(
            "M2_PROBE_MANIFEST.json", allocation="M2_LABEL_FREE_PROBES"
        ),
    }
    for name, value in manifests.items():
        write_json(REVIEW / name, value)
    panel = primary_panel_manifest(args.official_dataset)
    write_json(REVIEW / "PRIMARY_PANEL_MANIFEST.json", panel)
    fixtures = technical_fixtures()
    write_json(REVIEW / "TECHNICAL_FIXTURES.json", fixtures)
    write_json(
        REVIEW / "SOURCE_QUALIFICATION_SCHEDULE.json",
        source_schedule(manifests["SOURCE_VALIDATION_MANIFEST.json"]),
    )
    role_sets = {
        name.removesuffix("_MANIFEST.json"): value["item_ids"] for name, value in manifests.items()
    }
    role_sets["PRIMARY_PANEL"] = panel["item_ids"]
    overlap = {
        f"{left}__{right}": sorted(set(role_sets[left]) & set(role_sets[right]))
        for i, left in enumerate(role_sets)
        for right in list(role_sets)[i + 1 :]
    }
    if any(overlap.values()):
        raise RuntimeError("Q2 V4 data-purpose overlap")
    write_json(
        REVIEW / "DATA_PURPOSE_LEDGER.json",
        {
            "schema_version": "q2-v4-data-purpose-ledger-v1",
            "roles": {
                key: {"count": len(value), "item_ids": value} for key, value in role_sets.items()
            },
            "technical_fixture_ids": [row["fixture_id"] for row in fixtures["fixtures"]],
            "pairwise_overlaps": overlap,
            "all_disjoint": True,
            "semantic_panel_touched": False,
        },
    )
    lock = protocol_lock(source_commit)
    lock["artifact_hashes"] = {
        name: sha256(REVIEW / name)
        for name in (
            "SOURCE_CONSTRUCTION_MANIFEST.json",
            "SOURCE_VALIDATION_MANIFEST.json",
            "SHELL_CALIBRATION_MANIFEST.json",
            "M1_COVARIANCE_MANIFEST.json",
            "M2_PROBE_MANIFEST.json",
            "PRIMARY_PANEL_MANIFEST.json",
            "TECHNICAL_FIXTURES.json",
            "SOURCE_QUALIFICATION_SCHEDULE.json",
            "DATA_PURPOSE_LEDGER.json",
        )
    }
    write_json(REVIEW / "QUALIFICATION_PROTOCOL_LOCK.json", lock)
    (REVIEW / "QUALIFICATION_PROTOCOL_LOCK.md").write_text(
        "# Q2 V4 Spark-1 presemantic qualification lock\n\n"
        "Status: `Q2_V4_SPARK1_QUALIFICATION_PROTOCOL_FROZEN`.\n\n"
        "This lock authorizes Spark-1 technical, source, shell, M1, and A2 qualification only. "
        "The 300-item semantic panel is content-frozen and may not be processed. A PRELOCK commit "
        "will be created only after native source/subspace qualification; no random bank exists "
        "before that commit.\n",
        encoding="utf-8",
    )
    spec = ROOT / "experiments/specs/q2_v4_spark1_presemantic.yaml"
    spec.write_text(
        "schema_version: 1\n"
        "experiment_id: Q2_V4_SPARK1_PRESEMANTIC\n"
        "status: PROSPECTIVE_PRESEMANTIC_QUALIFICATION\n"
        "stage: DEVELOPMENT_PRESEMANTIC_QUALIFICATION\n"
        "backend: V4_NATIVE_SPARK1\n"
        "spark1_only: true\n"
        "spark2_used: false\n"
        "model: Qwen/Qwen3-8B\n"
        f"model_revision: {MODEL_REVISION}\n"
        "layer: 27\n"
        "candidate_count: 40\n"
        "selected_count: 32\n"
        "shell_targets: [0.25, 0.50]\n"
        "primary_panel_n: 300\n"
        "semantic_inference: forbidden\n"
        "q3: not_run\n",
        encoding="utf-8",
    )
    print(json.dumps({"review": str(REVIEW), "source_commit": source_commit, "panel_n": PRIMARY_N}))


if __name__ == "__main__":
    main()
