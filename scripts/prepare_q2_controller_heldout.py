#!/usr/bin/env python3
"""Prepare the pre-qualification lock for the first Q2 DEVELOPMENT pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import PARSER_VERSION  # noqa: E402
from epistemic_geometry.experiments.gate7 import DATASET_REPO, DATASET_REVISION  # noqa: E402
from epistemic_geometry.experiments.q2_controller_heldout import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    CONTROLLER_IDS,
    DELTA_NORM,
    ETA,
    EXECUTION_TEACHER_TEXT,
    EXPERIMENT_ID,
    LAYER,
    LOCATIONS,
    MODEL,
    MODEL_REVISION,
    N_ITEMS,
    NULL_FAMILIES,
    QAP_PERMUTATIONS,
    QAP_SEED,
    REFERENCE_SCALE,
    SOURCE_AXES,
    build_schedule,
    controller_split,
    source_axis_payload,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    stable_digest,
    stable_seed,
)

REVIEW = ROOT / "review/q2_controller_heldout_geometry"
NAMESPACE = "Q2-CONTROLLER-HELDOUT-GEOMETRY-V1"
ROLE_COUNTS = (
    ("SOURCE_CONSTRUCTION", 24),
    ("SOURCE_VALIDATION", 12),
    ("MANIPULATION_QUALIFICATION", 12),
    ("COVARIANCE_POOL", 64),
    ("FINITE_SECANT_PROBES", 12),
    ("COMMON_PANEL", 120),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _walk_items(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            output.extend(_walk_items(item))
    elif isinstance(value, dict):
        if {"item_id", "prompt", "reference_answer"} <= set(value):
            output.append(dict(value))
        else:
            for item in value.values():
                output.extend(_walk_items(item))
    return output


def historical_manifest_catalog() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct CRUXEval content from manifests only, never historical journals."""

    holdout_path = ROOT / "review/q1_confirmatory_fixed_controllers/HOLDOUT_CONTENT_MANIFEST.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdout_ids = {str(item["item_id"]) for item in holdout["items"]}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest_paths: list[str] = []
    for path in sorted((ROOT / "review").rglob("*.json")):
        upper = path.name.upper()
        if "MANIFEST" not in upper and "ITEMS" not in upper:
            continue
        if path == holdout_path or REVIEW in path.parents:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        items = _walk_items(value)
        used = False
        for item in items:
            item_id = str(item["item_id"])
            if not item_id.startswith("sample_"):
                continue
            if item.get("benchmark", "CRUXEval") != "CRUXEval":
                continue
            candidates[item_id].append(item)
            used = True
        if used:
            manifest_paths.append(str(path.relative_to(ROOT)))
    canonical: list[dict[str, Any]] = []
    ambiguous: dict[str, int] = {}
    for item_id, rows in candidates.items():
        if item_id in holdout_ids:
            continue
        signatures = Counter(
            (str(row["prompt"]), str(row["reference_answer"])) for row in rows
        )
        (prompt, reference), frequency = min(
            signatures.items(),
            key=lambda item: (-item[1], stable_digest(NAMESPACE, item_id, *item[0])),
        )
        if len(signatures) > 1:
            ambiguous[item_id] = len(signatures)
        exemplar = next(
            row
            for row in rows
            if str(row["prompt"]) == prompt and str(row["reference_answer"]) == reference
        )
        prompt_hash = stable_digest("Q2-TASK-PROMPT", prompt)
        canonical.append(
            {
                "item_id": item_id,
                "benchmark": "CRUXEval",
                "subtask": "output_prediction",
                "prompt": prompt,
                "reference_answer": reference,
                "reference_canonical_type": exemplar.get(
                    "reference_canonical_type",
                    exemplar.get("metadata", {}).get("reference_canonical_type"),
                ),
                "evaluator": "python_literal",
                "source_revision": DATASET_REVISION,
                "prompt_hash": prompt_hash,
                "item_hash": stable_digest(
                    "Q2-CRUXEVAL-ITEM", item_id, prompt_hash, reference, DATASET_REVISION
                ),
                "metadata": {
                    "dataset_repo": DATASET_REPO,
                    "dataset_revision": DATASET_REVISION,
                    "official_id": item_id,
                    "catalog_signature_frequency": frequency,
                },
            }
        )
    canonical.sort(
        key=lambda row: (
            stable_digest(NAMESPACE, "POOL", row["item_id"]),
            row["item_id"],
        )
    )
    required = sum(count for _role, count in ROLE_COUNTS)
    if len(canonical) < required:
        raise RuntimeError(
            f"Q2 requires {required} non-holdout manifest items; found {len(canonical)}"
        )
    return canonical, {
        "catalog_items": len(canonical),
        "holdout_excluded_count": len(holdout_ids),
        "holdout_identity_digest": stable_digest(
            NAMESPACE, "CLOSED_HOLDOUT_IDS", canonical_json(sorted(holdout_ids))
        ),
        "holdout_content_manifest_sha256": sha256(holdout_path),
        "source_manifest_count": len(manifest_paths),
        "source_manifests_digest": stable_digest(
            NAMESPACE, "SOURCE_MANIFESTS", canonical_json(manifest_paths)
        ),
        "ambiguous_historical_prompt_ids": ambiguous,
        "selection_used_historical_outcomes": False,
        "historical_journals_read": False,
    }


def allocate_roles(catalog: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    allocated: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for role, count in ROLE_COUNTS:
        rows = []
        for source in catalog[cursor : cursor + count]:
            row = dict(source)
            row["allocation"] = role
            row["metadata"] = {**row["metadata"], "q2_allocation": role}
            rows.append(row)
        allocated[role] = rows
        cursor += count
    all_ids = [row["item_id"] for rows in allocated.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Q2 role allocations are not disjoint")
    return allocated


def source_schedule(item_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        for rollout in range(2):
            conditions = [
                (axis.axis_id, polarity)
                for axis in SOURCE_AXES
                for polarity in ("POSITIVE", "NEGATIVE")
            ]
            conditions.sort(
                key=lambda pair: stable_digest(
                    NAMESPACE, "SOURCE_ORDER", item_id, rollout, *pair
                )
            )
            for order, (axis_id, polarity) in enumerate(conditions):
                rows.append(
                    {
                        "phase": "SOURCE_BEHAVIOR_QUALIFICATION",
                        "item_id": item_id,
                        "axis_id": axis_id,
                        "polarity": polarity,
                        "rollout_index": rollout,
                        "condition_order": order,
                        "seed": stable_seed(
                            EXPERIMENT_ID, "SOURCE_BEHAVIOR", item_id, axis_id, polarity, rollout
                        ),
                        "seed_regime": "INDEPENDENT_SOURCE_QUALIFICATION",
                    }
                )
    return rows


def manipulation_schedule(item_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        seed = stable_seed(EXPERIMENT_ID, "MANIPULATION_MATCHED", item_id)
        order = sorted(
            CONDITIONS,
            key=lambda condition: stable_digest(
                NAMESPACE, "MANIPULATION_ORDER", item_id, condition
            ),
        )
        rows.extend(
            {
                "phase": "CONTROLLER_MANIPULATION_QUALIFICATION",
                "item_id": item_id,
                "condition": condition,
                "rollout_index": 0,
                "condition_order": index,
                "seed": seed,
                "seed_regime": "MATCHED_COUPLING_QUALIFICATION",
            }
            for index, condition in enumerate(order)
        )
    return rows


def premortem() -> tuple[dict[str, Any], str]:
    risks = {
        "PSEUDO_GENERALIZATION": (
            "The test split holds out one entire conceptual source axis and two null-family "
            "members; signs and locations of the held-out axis cannot leak into train."
        ),
        "SOURCE_AXIS_CONFLATION": (
            "Three exact prompt pairs are frozen; source qualification uses behavior and "
            "held-out activation separation, never correctness or Q2 utility."
        ),
        "DOSE_CONFOUND": (
            "Every controller has one common absolute L27 displacement norm; there are no "
            "dose variants in the bank."
        ),
        "NULL_DISTORTION": (
            "Two isotropic and two sign-shuffled construction nulls are frozen before Q2 "
            "outcomes and orthogonalized against the six meaningful base span. This is "
            "reported as a design property, not hidden."
        ),
        "WHITENING_TUNING": (
            "The covariance pool is label-free and disjoint; lambda=0.10 times mean "
            "coordinate variance is fixed without error outcomes."
        ),
        "FINITE_SECANT_LEAKAGE": (
            "M2 uses 12 disjoint label-free probes, one fixed arbitrary continuation, four "
            "relative checkpoints, equal weights, and full-vocabulary JS."
        ),
        "DYADIC_PSEUDOREPLICATION": (
            "The unit permuted is the controller label; item bootstrap moves all conditions "
            "and both rollouts together."
        ),
        "NOISY_D": (
            "Unbiased two-rollout D may be negative. Negative-edge rate, bootstrap rank "
            "stability, interval width, and split-half matrix agreement are mandatory."
        ),
        "MIDRUN_PEEKING": (
            "Until 4,080 rows exist, monitoring is restricted to counts, integrity, process, "
            "GPU, disk, runtime, and cost."
        ),
        "COST": (
            "A condition-mix throughput preflight plus 20% margin must project no more than "
            "US$15 before common-panel collection."
        ),
        "FIREWALL": (
            "The 57 confirmatory identities are excluded; their outcomes are never read. "
            "Q3, JVP, Fisher, pullback, and manifold work remain closed."
        ),
    }
    data = {
        "experiment_id": EXPERIMENT_ID,
        "classification": "PREMORTEM_PASS",
        "risks": risks,
        "unresolved_scientific_ambiguities": [],
        "q1_result_changed": False,
        "q3_authorized": False,
    }
    lines = ["# Q2 controller-held-out pilot premortem", "", "`PREMORTEM_PASS`", ""]
    for name, mitigation in risks.items():
        lines.extend((f"## {name}", "", mitigation, ""))
    return data, "\n".join(lines)


def main() -> int:
    REVIEW.mkdir(parents=True, exist_ok=True)
    catalog, provenance = historical_manifest_catalog()
    roles = allocate_roles(catalog)
    filenames = {
        "SOURCE_CONSTRUCTION": "SOURCE_CONSTRUCTION_MANIFEST.json",
        "SOURCE_VALIDATION": "SOURCE_VALIDATION_MANIFEST.json",
        "MANIPULATION_QUALIFICATION": "MANIPULATION_MANIFEST.json",
        "COVARIANCE_POOL": "COVARIANCE_MANIFEST.json",
        "FINITE_SECANT_PROBES": "FINITE_SECANT_PROBE_MANIFEST.json",
        "COMMON_PANEL": "DEVELOPMENT_PANEL_MANIFEST.json",
    }
    for role, rows in roles.items():
        write_json(REVIEW / filenames[role], rows)
    split = controller_split()
    write_json(REVIEW / "CONTROLLER_SPLIT_LOCK.json", split)
    write_json(
        REVIEW / "SOURCE_BEHAVIOR_SCHEDULE.json",
        source_schedule([row["item_id"] for row in roles["SOURCE_VALIDATION"]]),
    )
    write_json(
        REVIEW / "MANIPULATION_SCHEDULE.json",
        manipulation_schedule(
            [row["item_id"] for row in roles["MANIPULATION_QUALIFICATION"]]
        ),
    )
    write_json(
        REVIEW / "COMMON_PANEL_SCHEDULE.json",
        build_schedule([row["item_id"] for row in roles["COMMON_PANEL"]]),
    )
    holdout_digest = {
        **provenance,
        "role_counts": dict(ROLE_COUNTS),
        "allocated_ids_digest": stable_digest(
            NAMESPACE,
            "ALLOCATED_IDS",
            canonical_json(
                {
                    role: [row["item_id"] for row in rows]
                    for role, rows in roles.items()
                }
            ),
        ),
        "allocations_pairwise_disjoint": True,
    }
    write_json(REVIEW / "DEVELOPMENT_PROVENANCE.json", holdout_digest)
    pre_json, pre_md = premortem()
    write_json(REVIEW / "PREMORTEM.json", pre_json)
    (REVIEW / "PREMORTEM.md").write_text(pre_md, encoding="utf-8")
    parser_path = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    geometry_lock = {
        "M0_FLAT": {
            "primary": "normalized Euclidean between unit controller directions",
            "secondary": "cosine distance",
            "identity": "d_euclidean^2 = 2 * d_cosine",
        },
        "M1_WHITENED": {
            "covariance_pool": "COVARIANCE_MANIFEST.json",
            "layer": LAYER,
            "token_position": "final non-padding prompt token",
            "estimator": "sample covariance low-rank SVD",
            "regularization": "Sigma_lambda=(1-0.10)Sigma+0.10*mean_variance*I",
            "regularization_fraction": 0.10,
            "distance": "normalized Euclidean in inverse-covariance inner product",
        },
        "M2_FINITE_SECANT": {
            "probe_panel": "FINITE_SECANT_PROBE_MANIFEST.json",
            "teacher_forced_text": EXECUTION_TEACHER_TEXT,
            "checkpoints": ["first", "one_third", "two_thirds", "last"],
            "distribution": "full vocabulary next-token softmax",
            "aggregation": "equal-weight mean full-vocabulary JS over probes/checkpoints",
            "distance": "sqrt(mean JS)",
            "exact_local_geometry_claimed": False,
        },
    }
    lock = {
        "schema_version": "q2-controller-heldout-prequalification-v1",
        "status": "FROZEN_PRE_QUALIFICATION",
        "lifecycle": "PROSPECTIVE_LOCK",
        "experiment_id": EXPERIMENT_ID,
        "development_only": True,
        "source_commit_at_preparation": git_head(),
        "model": {
            "id": MODEL,
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dtype": "BF16",
            "quantization": "none",
            "enable_thinking": False,
            "sampling": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "max_new_tokens": 4096,
            },
            "attention": "sdpa",
            "environment_profile": "CORE_QWEN",
        },
        "instrument": {
            "benchmark": "CRUXEval semantic output prediction",
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "evaluator": PARSER_VERSION,
            "evaluator_sha256": sha256(parser_path),
            "invalid_as_error": True,
            "confirmatory_holdout_outcomes_read": False,
        },
        "allocations": {
            role: {
                "n": len(rows),
                "file": filenames[role],
                "file_sha256": sha256(REVIEW / filenames[role]),
            }
            for role, rows in roles.items()
        },
        "controller_candidates": {
            "common_layer": LAYER,
            "source_axes": source_axis_payload(),
            "locations": list(LOCATIONS),
            "signs": ["PLUS", "MINUS"],
            "meaningful_n": 12,
            "null_n": 4,
            "null_families": list(NULL_FAMILIES),
            "controller_ids": list(CONTROLLER_IDS),
            "eta": ETA,
            "reference_scale": REFERENCE_SCALE,
            "common_absolute_delta_norm": DELTA_NORM,
            "dose_variants": False,
            "construction_uses_correctness": False,
        },
        "qualification": {
            "source_validity_min": 0.90,
            "source_cross_disagreement_min": 0.10,
            "source_excess_disagreement_min": 0.03,
            "source_token_ratio_alternative_min": 1.15,
            "source_median_token_difference_alternative_min": 2,
            "activation_standardized_gap_min": 0.20,
            "activation_positive_gap_fraction_min": 0.60,
            "base_direction_max_absolute_cosine": 0.98,
            "controller_validity_min": 0.75,
            "controller_semantic_change_min": 1.0 / 12.0,
            "controller_raw_sequence_change_min": 0.25,
            "accuracy_ranking_forbidden": True,
            "G_C_D_ranking_forbidden": True,
            "failure_classification": "Q2_CONTROLLER_BANK_NOT_QUALIFIED",
        },
        "geometries": geometry_lock,
        "controller_split": split,
        "common_panel": {
            "n": N_ITEMS,
            "conditions": len(CONDITIONS),
            "controller_bank_k": len(CONTROLLER_IDS),
            "rollouts": 2,
            "expected_rows": N_ITEMS * len(CONDITIONS) * 2,
            "seed_regime": "INDEPENDENT_PRIMARY",
            "schedule_file": "COMMON_PANEL_SCHEDULE.json",
            "schedule_sha256": sha256(REVIEW / "COMMON_PANEL_SCHEDULE.json"),
        },
        "prediction": {
            "primary": "heldout-controller-edge Spearman rho",
            "secondary": "train-calibrated heldout standardized RMSE",
            "train_calibration_edges": 45,
            "heldout_prediction_edges": 75,
            "qap_permutations": QAP_PERMUTATIONS,
            "qap_seed": QAP_SEED,
            "qap_unit": "controller label",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "item",
        },
        "classification_thresholds": {
            "metric_signal": {
                "rho_min": 0.30,
                "one_sided_qap_p_max": 0.05,
                "rmse_ratio_to_constant_max": 0.90,
            },
            "control_geometry_outperformance": {
                "rho_gain_over_flat_min": 0.15,
                "rmse_ratio_to_flat_max": 0.90,
            },
            "names": [
                "Q2_PILOT_HELDOUT_PREDICTION_SIGNAL",
                "Q2_PILOT_SIMPLE_GEOMETRY_SIGNAL",
                "Q2_PILOT_CONTROL_GEOMETRY_OUTPERFORMS_FLAT",
                "Q2_PILOT_NO_HELDOUT_GEOMETRY_SIGNAL",
                "Q2_CONTROLLER_BANK_NOT_QUALIFIED",
            ],
        },
        "cost": {
            "target_usd": 8.50,
            "soft_ceiling_usd": 12.0,
            "hard_ceiling_usd": 15.0,
            "projection_margin": 0.20,
            "full_collection_blocked_if_projection_exceeds_hard_ceiling": True,
        },
        "firewall": {
            "Q1": "IMMUTABLE",
            "Q3": "NOT AUTHORIZED",
            "confirmatory_holdout": "CLOSED; identities excluded; outcomes unread",
            "JVP_Fisher_pullback_manifold": "NOT RUN",
        },
    }
    write_json(REVIEW / "CANDIDATE_PROTOCOL_LOCK.json", lock)
    report = f"""# Q2 controller-held-out geometry pilot — candidate lock

Status: `FROZEN_PRE_QUALIFICATION`  
Role: `DEVELOPMENT`  
Q1: immutable  
Q3: not authorized

The bank is prospectively defined as three conceptual axes (verification,
explicit state tracking, and type/representation discipline), two L27 source
locations, both signs, and four nulls. Every controller uses the same absolute
L27 sustained-current-token displacement norm `{DELTA_NORM:.15f}`. Accuracy,
G, C, D, rescue, damage, and complementarity are forbidden during bank
qualification.

The controller-held-out split is already frozen at 10/6 and holds out the full
`{split['heldout_axis']}` conceptual family plus one null from each null family.
The three geometry definitions and predictive thresholds are frozen here. A
second final bank lock with vector/archive hashes is required before the 4,080
common-panel rows may begin.
"""
    (REVIEW / "CANDIDATE_PROTOCOL_LOCK.md").write_text(report, encoding="utf-8")
    hashes = {
        path.name: sha256(path)
        for path in sorted(REVIEW.iterdir())
        if path.is_file() and path.name != "artifact_hashes_prequalification.json"
    }
    write_json(REVIEW / "artifact_hashes_prequalification.json", hashes)
    print(
        json.dumps(
            {
                "classification": "Q2_CANDIDATE_PROTOCOL_LOCK_PREPARED",
                "catalog_items": len(catalog),
                "allocated_items": sum(dict(ROLE_COUNTS).values()),
                "common_panel_rows": lock["common_panel"]["expected_rows"],
                "heldout_axis": split["heldout_axis"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
