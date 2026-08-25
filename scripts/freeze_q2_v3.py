#!/usr/bin/env python3
# ruff: noqa: E501
"""Materialize the inference-free prospective Q2 V3 freeze artifacts."""

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

from epistemic_geometry.experiments.q2_v3 import (  # noqa: E402
    BOOTSTRAP_SEED,
    DATASET_REPO,
    DATASET_REVISION,
    EVALUATION_SEED_NAMESPACE,
    EXECUTION_TEACHER_TEXT,
    EXPERIMENT_ID,
    LAYER,
    M1_COVARIANCE_NAMESPACE,
    M2_PROBE_NAMESPACE,
    MODEL,
    MODEL_REVISION,
    NULL_SEEDS,
    PRIMARY_PANEL_NAMESPACE,
    SHELL_CALIBRATION_NAMESPACE,
    SHELL_SEED_NAMESPACE,
    SHELL_TARGETS,
    SOURCE_CONSTRUCTION_NAMESPACE,
    SOURCE_FAMILIES,
    SOURCE_SEED_NAMESPACE,
    SOURCE_VALIDATION_NAMESPACE,
    condition_ids,
    deterministic_allocate,
    family_payload,
    meaningful_controller_ids,
    null_controller_ids,
    ordered_id_hash,
    stable_rank,
    stable_seed,
)

REVIEW = ROOT / "review/q2_v3_radial_angular_freeze"
PROVENANCE = ROOT / "review/q2_m3_qualification_cruxeval_provenance"
SPEC = ROOT / "experiments/specs/q2_v3_radial_angular_geometry.yaml"
REVIEWED_BASE = "9c2ea31c449a6874d4ae47337dd3b913d5bee559"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def read_ledger() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (PROVENANCE / "CRUXEVAL_PROVENANCE_LEDGER.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(rows) != 800 or len({row["item_id"] for row in rows}) != 800:
        raise RuntimeError("CRUXEval provenance ledger is incomplete or duplicated")
    return rows


def historical_content_fallback() -> dict[str, dict[str, str]]:
    rows: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if str(value.get("item_id", "")).startswith("sample_"):
                rows.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    source_root = ROOT / "review/external_benchmark_qualification"
    paths = sorted(source_root.rglob("journal.jsonl")) + sorted(
        source_root.rglob("results.json")
    )
    for path in paths:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    walk(json.loads(line))
        else:
            walk(json.loads(path.read_text(encoding="utf-8")))
    return {
        str(row["item_id"]): {
            "prompt_sha256": str(row["prompt_hash"]),
            "reference_sha256": sha256_bytes(str(row["reference_answer"]).encode()),
        }
        for row in rows
        if row.get("prompt_hash") and row.get("reference_answer") is not None
    }


def manifest_record(
    row: dict[str, Any],
    namespace: str,
    allocation: str,
    fallback: dict[str, dict[str, str]],
) -> dict[str, Any]:
    content = row["canonical_content"] or fallback.get(str(row["item_id"]))
    if content is None:
        raise RuntimeError(f"missing frozen content hashes for {row['item_id']}")
    return {
        "allocation": allocation,
        "item_id": row["item_id"],
        "official_index": row["official_index"],
        "provenance_class": row["provenance_class"],
        "prompt_sha256": content["prompt_sha256"],
        "reference_sha256": content["reference_sha256"],
        "selection_rank": stable_rank(namespace, row["item_id"]),
    }


def make_manifest(
    rows: list[dict[str, Any]],
    namespace: str,
    allocation: str,
    provenance_class: str,
    fallback: dict[str, dict[str, str]],
) -> dict[str, Any]:
    items = [manifest_record(row, namespace, allocation, fallback) for row in rows]
    ids = [item["item_id"] for item in items]
    return {
        "schema_version": "q2-v3-allocation-v1",
        "status": "FROZEN_NOT_RUN",
        "allocation": allocation,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "provenance_class": provenance_class,
        "selection_namespace": namespace,
        "selection_algorithm": (
            "filter exact provenance class and prior exclusions; sort ascending by "
            "SHA256(namespace + U+001F + item_id), then item_id; take first N"
        ),
        "rng": "NONE_SHA256_TOTAL_ORDER",
        "outcome_values_read_or_used": False,
        "item_count": len(items),
        "item_ids": ids,
        "ordered_ids_sha256": ordered_id_hash(ids),
        "items": items,
    }


def schedule() -> dict[str, Any]:
    panel = json.loads((REVIEW / "PRIMARY_PANEL_MANIFEST.json").read_text(encoding="utf-8"))
    conditions = condition_ids()
    rows: list[dict[str, Any]] = []
    for item_id in panel["item_ids"]:
        for rollout in (0, 1):
            ordered_conditions = sorted(
                conditions,
                key=lambda condition: (
                    stable_rank(
                        f"{EVALUATION_SEED_NAMESPACE}-ORDER-{item_id}-{rollout}", condition
                    ),
                    condition,
                ),
            )
            for order_in_block, condition in enumerate(ordered_conditions):
                rows.append(
                    {
                        "item_id": item_id,
                        "condition": condition,
                        "rollout_index": rollout,
                        "seed": stable_seed(
                            EVALUATION_SEED_NAMESPACE, item_id, condition, rollout
                        ),
                        "order_in_item_rollout_block": order_in_block,
                    }
                )
    logical = [(row["item_id"], row["condition"], row["rollout_index"]) for row in rows]
    seeds = [row["seed"] for row in rows]
    if len(rows) != 10_000 or len(set(logical)) != 10_000 or len(set(seeds)) != 10_000:
        raise RuntimeError("frozen Q2 V3 evaluation schedule is not unique and complete")
    return {
        "schema_version": "q2-v3-evaluation-schedule-v1",
        "status": "FROZEN_NOT_RUN",
        "seed_regime": "INDEPENDENT_PRIMARY",
        "seed_namespace": EVALUATION_SEED_NAMESPACE,
        "condition_interleaving": "SHA256 permutation inside every item-rollout block",
        "expected_rows": 10_000,
        "rows": rows,
    }


def source_schedule(item_ids: list[str]) -> dict[str, Any]:
    rows = []
    for item_id in item_ids:
        for family in SOURCE_FAMILIES:
            for polarity in ("POSITIVE", "NEGATIVE"):
                for rollout in (0, 1):
                    rows.append(
                        {
                            "item_id": item_id,
                            "family": family.family_id,
                            "polarity": polarity,
                            "rollout_index": rollout,
                            "seed": stable_seed(
                                SOURCE_SEED_NAMESPACE,
                                item_id,
                                family.family_id,
                                polarity,
                                rollout,
                            ),
                        }
                    )
    if len(rows) != 480 or len({row["seed"] for row in rows}) != 480:
        raise RuntimeError("source schedule is not the frozen 480-row design")
    return {
        "schema_version": "q2-v3-source-schedule-v1",
        "status": "FROZEN_NOT_RUN",
        "seed_namespace": SOURCE_SEED_NAMESPACE,
        "expected_rows": 480,
        "correctness_forbidden": True,
        "rows": rows,
    }


def shell_schedule(item_ids: list[str]) -> dict[str, Any]:
    conditions = ("BASELINE", *meaningful_controller_ids())
    rows = []
    for item_id in item_ids:
        for rollout in (0, 1):
            seed = stable_seed(SHELL_SEED_NAMESPACE, item_id, rollout)
            for condition in sorted(
                conditions,
                key=lambda name: stable_rank(
                    f"{SHELL_SEED_NAMESPACE}-ORDER-{item_id}-{rollout}", name
                ),
            ):
                rows.append(
                    {
                        "item_id": item_id,
                        "condition": condition,
                        "rollout_index": rollout,
                        "matched_seed": seed,
                    }
                )
    if len(rows) != 504:
        raise RuntimeError("shell calibration schedule is not the frozen 504-row design")
    return {
        "schema_version": "q2-v3-shell-calibration-schedule-v1",
        "status": "FROZEN_NOT_RUN",
        "coupling": "MATCHED_WITHIN_ITEM_ROLLOUT_CALIBRATION_ONLY",
        "seed_namespace": SHELL_SEED_NAMESPACE,
        "expected_rows": 504,
        "correctness_forbidden": True,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", default=git_head())
    args = parser.parse_args()
    source_commit = str(args.source_commit)
    ledger = read_ledger()
    fallback = historical_content_fallback()
    proposed = json.loads(
        (PROVENANCE / "Q2_V3_PROPOSED_PRIMARY_PANEL.json").read_text(encoding="utf-8")
    )
    class_counts = {
        key: sum(row["provenance_class"] == key for row in ledger)
        for key in ("A", "B", "C", "D", "UNRESOLVED")
    }
    expected_primary = deterministic_allocate(
        ledger,
        provenance_class="C",
        namespace=PRIMARY_PANEL_NAMESPACE,
        count=200,
    )
    expected_ids = [row["item_id"] for row in expected_primary]
    if expected_ids != proposed["item_ids"]:
        raise RuntimeError("Q2_V3_FREEZE_BLOCKED: OUTCOME_DEPENDENT_PANEL_SELECTION")
    if ordered_id_hash(expected_ids) != (
        "969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf"
    ):
        raise RuntimeError("Q2 V3 primary panel hash mismatch")

    REVIEW.mkdir(parents=True, exist_ok=True)
    primary = make_manifest(
        expected_primary,
        PRIMARY_PANEL_NAMESPACE,
        "PRIMARY_SEMANTIC_PANEL",
        "C",
        fallback,
    )
    write_json(REVIEW / "PRIMARY_PANEL_MANIFEST.json", primary)
    excluded = set(expected_ids)
    allocations: dict[str, dict[str, Any]] = {}
    for filename, namespace, allocation, count in (
        ("SOURCE_CONSTRUCTION_MANIFEST.json", SOURCE_CONSTRUCTION_NAMESPACE, "SOURCE_CONSTRUCTION", 24),
        ("SOURCE_VALIDATION_MANIFEST.json", SOURCE_VALIDATION_NAMESPACE, "SOURCE_VALIDATION", 24),
        ("SHELL_CALIBRATION_MANIFEST.json", SHELL_CALIBRATION_NAMESPACE, "SHELL_CALIBRATION", 12),
        ("M1_COVARIANCE_MANIFEST.json", M1_COVARIANCE_NAMESPACE, "M1_COVARIANCE", 64),
    ):
        chosen = deterministic_allocate(
            ledger,
            provenance_class="C",
            namespace=namespace,
            count=count,
            excluded=excluded,
        )
        manifest = make_manifest(chosen, namespace, allocation, "C", fallback)
        write_json(REVIEW / filename, manifest)
        allocations[allocation] = manifest
        excluded.update(manifest["item_ids"])
    m2_rows = deterministic_allocate(
        ledger, provenance_class="B", namespace=M2_PROBE_NAMESPACE, count=12
    )
    m2_manifest = make_manifest(
        m2_rows, M2_PROBE_NAMESPACE, "M2_LABEL_FREE_PROBES", "B", fallback
    )
    write_json(REVIEW / "M2_PROBE_MANIFEST.json", m2_manifest)

    proof = {
        "schema_version": "q2-v3-panel-provenance-proof-v1",
        "classification_counts": class_counts,
        "classification_uses_outcome_values": False,
        "eligibility": "provenance_class == C",
        "ordering": "SHA256(namespace + U+001F + item_id), then item_id",
        "selection_namespace": PRIMARY_PANEL_NAMESPACE,
        "rng_algorithm": "NONE; cryptographic hash total order",
        "seed_equivalent": PRIMARY_PANEL_NAMESPACE,
        "selected_n": 200,
        "selected_ordered_ids_sha256": primary["ordered_ids_sha256"],
        "matches_proposed_panel_byte_order": True,
        "code_paths": [
            "scripts/audit_cruxeval_provenance.py",
            "src/epistemic_geometry/analysis/cruxeval_provenance.py",
            "src/epistemic_geometry/experiments/q2_v3.py",
        ],
        "forbidden_fields_loaded_by_selector": [],
        "explicitly_not_used": [
            "historical accuracy",
            "baseline correctness",
            "item difficulty",
            "controller performance",
            "complementarity",
            "error profiles",
            "M0/M1/M2 behavior",
            "manual examples",
            "outcome variance",
        ],
        "selection_outcome_independent": True,
        "content_hash_fallback": {
            "role": "post-selection manifest completion only",
            "source": "tracked external_benchmark_qualification journal/results artifacts",
            "fields_read": ["item_id", "prompt_hash", "reference_answer"],
            "outcome_fields_read_or_used": False,
        },
        "classification": "Q2_V3_PANEL_SELECTION_PROVENANCE_CLEAN",
    }
    write_json(REVIEW / "PRIMARY_PANEL_PROVENANCE_PROOF.json", proof)

    controller_spec = {
        "schema_version": "q2-v3-controller-bank-spec-v1",
        "status": "FROZEN_NOT_RUN",
        "layer_zero_based": LAYER,
        "constructor": "PAIRED_MEAN_DIFFERENCE",
        "orientation": "POSITIVE_MINUS_NEGATIVE",
        "normalization": "float64 Euclidean unit norm after paired mean",
        "duration": "SUSTAINED_CURRENT_TOKEN",
        "prefill_scope": "final non-padding prompt token",
        "decode_scope": "current token exactly once per forward; cached history unchanged",
        "source_locations": {
            "PROMPT_BOUNDARY": "layer-27 block output at final non-padding rendered-prompt token",
            "EXECUTION_BOUNDARY": (
                "layer-27 block output at final token of the frozen teacher continuation appended "
                "to the same rendered prompt"
            ),
        },
        "execution_teacher_text": EXECUTION_TEACHER_TEXT,
        "families": family_payload(),
        "construction": {
            "items": 24,
            "pairing": "same item under positive and negative instructions",
            "formula": "v_raw=mean_x(h_positive(x)-h_negative(x)); v=v_raw/||v_raw||_2",
            "precision": "capture float32; pair and average float64; persist vector float32",
            "minimum_raw_norm": 1e-6,
            "finite_required": True,
        },
        "heldout_representation_qualification": {
            "items": 24,
            "standardized_gap_formula": (
                "mean_x(<h_pos-h_neg,v>)/std_x(<h_pos-h_neg,v>,ddof=1)"
            ),
            "standardized_gap_min": 0.20,
            "positive_projection_fraction_min": 0.60,
            "applies_to_each_of_ten_base_directions": True,
        },
        "label_free_source_behavior_qualification": {
            "rollouts_per_item_polarity": 2,
            "commitment_validity_each_polarity_min": 0.90,
            "semantic_evaluability_each_polarity_min": 0.90,
            "cross_disagreement_min": 0.10,
            "excess_disagreement_min": 0.03,
            "cross_disagreement": "mean of the four positive-vs-negative rollout comparisons",
            "within_disagreement": (
                "mean of positive rollout0-vs-rollout1 and negative rollout0-vs-rollout1"
            ),
            "excess_disagreement": "cross_disagreement-within_disagreement",
            "correctness_reference_access": "FORBIDDEN",
        },
        "candidate_count": 10,
        "usable_rule": (
            "all five families must pass source behavior and both location-specific representation "
            "gates; any failure stops the bank with no replacement, sign flip, or exclusion"
        ),
        "shells": SHELL_TARGETS,
        "meaningful_controller_count": 20,
        "meaningful_controller_ids": list(meaningful_controller_ids()),
        "nulls": {
            "base_direction_count": 2,
            "controller_count": 4,
            "seeds": list(NULL_SEEDS),
            "algorithm": (
                "Gaussian PCG64; project with SVD orthonormal basis Q of the ten meaningful "
                "directions as r-Q(Q^T r); Gram-Schmidt against prior null; normalize"
            ),
            "svd_rank_tolerance": 1e-10,
            "span_and_pairwise_absolute_cosine_max": 1e-6,
            "ids": list(null_controller_ids()),
        },
        "semantic_correctness_used": False,
        "failure": "Q2_V3_CONTROLLER_QUALIFICATION_FAILED; semantic panel forbidden",
    }
    write_json(REVIEW / "CONTROLLER_BANK_SPEC.json", controller_spec)

    shell_spec = {
        "schema_version": "q2-v3-shell-calibration-spec-v1",
        "status": "FROZEN_NOT_RUN",
        "terminology": "coordinate-space implemented intervention amplitude",
        "not_intrinsic_geometry": True,
        "formula": (
            "r_impl(delta)=sqrt(mean_{x,k}||BF16(delta)||_2^2 / "
            "mean_{x,k}||h_L27(x,k)||_2^2)"
        ),
        "delta_cast_timing": "alpha times float64 unit direction, then cast once to BF16 and back to float32 for measurement and injection",
        "activation_denominator": (
            "baseline layer-27 block-output current-token residuals on 12 calibration prompts, "
            "sequential KV teacher forcing; k is final prompt token plus every token of the "
            "frozen teacher continuation; squared norms accumulated float64"
        ),
        "targets": SHELL_TARGETS,
        "target_justification": (
            "0.25 and 0.50 span a twofold implemented-amplitude contrast inside the historical "
            "label-free V2 safe/manipulation displacement range while remaining below the "
            "historical high-dose anchor; no V3 outcome informed them"
        ),
        "root_finding": {
            "method": "deterministic bisection over alpha with BF16-cast amplitude evaluated at every visited point",
            "initial_interval": [0.0, 256.0],
            "maximum_iterations": 40,
            "relative_target_error_max": 0.005,
            "selection": "visited alpha with minimum absolute amplitude error; ties choose lower alpha",
            "failure": "stop before label-free shell-safety generation; no extrapolation or target change",
        },
        "label_free_safety": {
            "items": 12,
            "rollouts": 2,
            "matched_seed_across_conditions": True,
            "validity_min": 0.90,
            "evaluability_min": 0.90,
            "relative_validity_drop_max": 0.05,
            "relative_evaluability_drop_max": 0.05,
            "truncation_rate_max": 0.05,
            "raw_sequence_movement_min": {"MEDIUM": 0.10, "STRONG": 0.15},
            "correctness_forbidden": True,
            "all_20_meaningful_controllers_must_pass": True,
        },
        "seed_namespace": SHELL_SEED_NAMESPACE,
    }
    write_json(REVIEW / "SHELL_CALIBRATION_SPEC.json", shell_spec)

    identifiability = {
        "schema_version": "q2-v3-identifiability-gate-v1",
        "status": "FROZEN_NOT_RUN",
        "population": "ten meaningful base directions; shell checks include ten meaningful plus two null directions",
        "criteria": {
            "radius_cv_each_shell": {"formula": "population_std(r_impl)/mean(r_impl)", "max": 0.03},
            "family_median_deviation_each_shell": {
                "formula": "abs(median(two family radii)-median(ten meaningful radii))/global_median",
                "max": 0.03,
            },
            "cross_family_dyads_each_shell": {"exact": 40},
            "direction_gram_effective_rank": {
                "formula": "(sum eigenvalues)^2/sum(eigenvalues^2) for 10x10 Euclidean unit-vector Gram",
                "min": 5.0,
            },
            "nonantipodal_absolute_cosine": {"population": "all 45 distinct base-direction pairs", "max_strict": 0.95},
            "angular_q90_minus_q10": {"population": "40 cross-family dyads separately for M0 and M1 in each shell", "min": 0.20},
            "radial_nuisance_r2": {
                "formula": "OLS R2(metric distance ~ 1 + abs radius difference + mean radius)",
                "population": "40 cross-family dyads, separately by shell and M0/M1/M2",
                "max": 0.10,
            },
            "family_angular_leverage": {
                "formula": "incident squared centered z-distance divided by total across families; each dyad counted for both incident families",
                "population": "each metric and shell",
                "max": 0.30,
            },
            "standardized_geometry_feature_condition_number": {
                "formula": "largest/smallest singular value of centered unit-SD columns [M0,M1,M2] over 40 dyads",
                "population": "each shell",
                "max": 30.0,
            },
            "null_matching": {
                "target_relative_error_max": 0.005,
                "span_absolute_cosine_max": 1e-6,
                "pairwise_null_absolute_cosine_max": 1e-6,
            },
        },
        "undefined_or_nonfinite": "FAIL",
        "all_criteria_required": True,
        "failure_classification": "Q2_V3_ANGULAR_IDENTIFIABILITY_FAILED",
        "failure_action": "do not run semantic panel; do not redraw or alter thresholds",
    }
    write_json(REVIEW / "IDENTIFIABILITY_GATE.json", identifiability)

    geometry = {
        "schema_version": "q2-v3-geometry-definitions-v1",
        "status": "FROZEN_NOT_RUN",
        "controller_order": list((*meaningful_controller_ids(), *null_controller_ids())),
        "M0": {
            "name": "normalized coordinate-space angular chord",
            "formula": "sqrt(2-2*clip(<v_i/||v_i||,v_j/||v_j||>,-1,1))",
            "input": "ten meaningful and two null Euclidean unit base directions, duplicated by shell",
            "implementation": "src/epistemic_geometry/analysis/q2_geometries.py::flat_geometry",
        },
        "M1": {
            "name": "regularized covariance-whitened angular chord",
            "covariance_source": "64 disjoint label-free Class-C baseline prompt-boundary L27 activations",
            "covariance_formula": "Sigma_lambda=(1-lambda)*sample_covariance+lambda*mean_coordinate_variance*I",
            "lambda": 0.10,
            "solver": "float64 thin SVD low-rank inverse plus isotropic ridge complement; no explicit dense inverse",
            "normalization": "unit norm in the regularized inverse-covariance inner product",
            "formula": "sqrt(2-2*whitened_cosine(v_i,v_j))",
            "implementation": "src/epistemic_geometry/analysis/q2_geometries.py::fit_whitening,whitened_geometry",
        },
        "M2": {
            "name": "finite output-response Jensen-Shannon pseudometric",
            "not_an_inner_product_angle": True,
            "probe_source": "12 disjoint provenance-Class-B label-free CRUXEval prompts",
            "teacher_forced_text": EXECUTION_TEACHER_TEXT,
            "conditions": ["BASELINE", *meaningful_controller_ids(), *null_controller_ids()],
            "distribution": "full-vocabulary softmax of BF16-model logits, dequantized to float64",
            "checkpoints": (
                "four per probe: prefill final-prompt next-token logits, then continuation "
                "indices floor(L/3), floor(2L/3), and L-1 under sequential KV teacher forcing"
            ),
            "js": "natural-log Jensen-Shannon; logsumexp float64; no probability clipping; nonfinite input fails",
            "aggregation": "equal-weight arithmetic mean over 12x4 probe/checkpoint rows",
            "square_root_placement": "sqrt after the mean: d_ij=sqrt(mean_r JS(P_i,r,P_j,r))",
            "baseline": "captured for diagnostic response radius only; not converted into an angle",
            "storage": "compressed NPZ float32 logits with controller/probe/checkpoint metadata",
            "implementation": "src/epistemic_geometry/analysis/q2_geometries.py::finite_secant_geometry with float32 persistence amendment",
        },
        "M3": {
            "status": "EXCLUDED_NOT_QUALIFIED",
            "classification": "M3_DERIVATIVE_IDENTITIES_FAILED",
            "interpretation": "instrument nonqualification for the BF16 stack, not evidence against Fisher/pullback geometry",
            "runtime_reenable_forbidden": True,
        },
        "semantic_labels_or_outcomes_used": False,
    }
    write_json(REVIEW / "GEOMETRY_DEFINITIONS.json", geometry)

    prediction_lock = {
        "schema_version": "q2-v3-prediction-lock-v1",
        "status": "PROCEDURE_FROZEN_ARTIFACTS_NOT_YET_COMPUTED",
        "chronology": [
            "construct and qualify source directions",
            "calibrate and qualify implemented-amplitude shells",
            "construct fresh nulls and apply identifiability prerequisites",
            "capture M1 covariance and M2 label-free finite responses",
            "compute M0/M1/M2 matrices",
            "write matrices and metadata atomically",
            "verify all identifiability gates",
            "hash and commit prediction lock",
            "only then authorize semantic panel collection",
        ],
        "primary_semantic_panel_disjoint_from_geometry_inputs": True,
        "files": {
            "arrays": "Q2_V3_PREDICTION_MATRICES.npz",
            "metadata": "Q2_V3_PREDICTION_MATRICES.json",
            "lock": "Q2_V3_PREDICTION_LOCK.json",
        },
        "npz_keys": ["M0", "M1", "M2"],
        "shape_each": [24, 24],
        "dtype": "float64 matrices",
        "controller_order": list((*meaningful_controller_ids(), *null_controller_ids())),
        "required_metadata": [
            "source_commit",
            "code_commit",
            "controller_vector_hashes",
            "implemented_alphas_and_amplitudes",
            "calibration_manifest_hashes",
            "M1_fit_hash",
            "M2_archive_hash",
            "matrix_hashes",
            "controller_order_hash",
        ],
        "post_outcome_recomputation": "forensic exact reproduction from identical frozen inputs only",
        "hash_or_order_mismatch": "stop before semantic inference",
    }
    write_json(REVIEW / "PREDICTION_LOCK_SPEC.json", prediction_lock)

    statistics = {
        "schema_version": "q2-v3-statistical-analysis-plan-v1",
        "status": "FROZEN_NOT_RUN",
        "primary_claim": "RELATIONAL_GEOMETRY_WITHIN_MATCHED_IMPLEMENTED_AMPLITUDE_SHELLS",
        "error": "e=0 iff external-semantic-v3 correct; valid-wrong, invalid, unevaluable, truncation, and model runtime outcome are e=1",
        "distance": {
            "formula": "D_ij=N^-1 sum_t (e_i,t,0-e_j,t,0)*(e_i,t,1-e_j,t,1)",
            "rollouts": "two independent draws per item-condition",
            "shrinkage": "none",
            "missingness": "no complete-case filtering; model outcomes retained as errors",
        },
        "primary_pairs": "40 cross-family base-direction dyads per shell; 80 shell-stratified dyads",
        "spearman": {
            "rank_ties": "average ranks",
            "correlation": "Pearson correlation of average ranks",
            "family_summary": "for each family and shell, rho over 16 incident cross-family dyads",
            "shell_summary": "arithmetic mean of five family incident-dyad rhos",
            "aggregate": "arithmetic mean of MEDIUM and STRONG shell summaries",
            "undefined": "metric fails relational gate",
        },
        "multiplicity": {
            "choice": "OMNIBUS_EXISTENCE_THEN_METRIC_IDENTIFICATION",
            "justification": (
                "M0/M1/M2 are three competing pre-outcome hypotheses; an exact max-statistic "
                "family QAP controls their shared existence claim before metric attribution"
            ),
            "qap_space": "exact 5! family-block permutations x 2^5 within-family location swaps = 3840",
            "shell_handling": "same family/location mapping applied to both shells",
            "max_statistic": "maximum aggregate family-balanced rho across M0/M1/M2",
            "p_value": "number of exact null statistics >= observed divided by 3840, identity included",
            "alternative": "one-sided positive association",
            "rng": "none; exhaustive lexicographic enumeration",
        },
        "bootstrap": {
            "resamples": 10_000,
            "seed": BOOTSTRAP_SEED,
            "unit": "item_id",
            "dependence": "move all 25 conditions and both rollouts for each sampled item together",
            "ci": "percentile [2.5%,97.5%]",
        },
        "leave_one_direction_out": "ten diagnostics; remove one base direction and all incident pairs, recompute aggregate family-balanced rho",
        "relational_gate": {
            "aggregate_family_balanced_rho_min": 0.25,
            "max_qap_corrected_p_max": 0.05,
            "item_bootstrap_lower_bound_strictly_greater_than": 0.0,
            "positive_family_summaries_min": 4,
            "positive_family_summary_definition": "mean of that family's two shell-specific incident-dyad rhos > 0",
            "positive_shells_required": 2,
            "leave_one_direction_out_all_strictly_positive": True,
        },
        "threshold_justifications": {
            "rho_0_25": "A moderate rank effect is required because N=200 chiefly reduces item noise; prior dependence-preserving simulations showed family novelty remains dominant and did not justify a smaller scientific effect.",
            "qap_0_05": "The exact family-constrained max statistic provides finite-sample family-wise control across all three prespecified metrics.",
            "bootstrap_lower_0": "Item-cluster uncertainty must exclude a nonpositive aggregate association while preserving all condition/rollout dependence.",
            "four_of_five_families": "A relational claim must be distributed across new conceptual families rather than driven by one or two families.",
            "both_shells": "The relation must survive the twofold radial manipulation and therefore cannot be a single-shell accident.",
            "lodo_positive": "No one of the ten prospectively constructed directions may determine the sign.",
            "m2_delta_0_10": "Specific necessity of finite output response requires a scientifically material paired rank improvement, not merely nominal significance.",
        },
        "m2_required": {
            "M2_relational_gate_pass": True,
            "delta_rho_over_M0_min": 0.10,
            "delta_rho_over_M1_min": 0.10,
            "paired_item_bootstrap_lower_bound_each_strictly_greater_than": 0.0,
            "exact_stepdown_family_qap_p_each_max": 0.05,
        },
        "radial_claim": {
            "direction_value": "D(strong,baseline)-D(medium,baseline)",
            "aggregate": "median over ten base directions",
            "positive_directions_min": 8,
            "aggregate_strictly_positive": True,
            "item_bootstrap_lower_bound_strictly_greater_than": 0.0,
            "permutation": "exact 2^5 family-block sign flips; p=count(null>=observed)/32",
            "p_max": 0.05,
            "role": "secondary control claim, reported independently as R+ or R-",
        },
        "secondary": ["all metric-specific rhos", "same-direction cross-shell distances", "null dyads", "RMSE/calibration without fitted primary coefficients"],
    }
    write_json(REVIEW / "STATISTICAL_ANALYSIS_PLAN.json", statistics)

    taxonomy = {
        "schema_version": "q2-v3-classification-taxonomy-v1",
        "status": "FROZEN_NOT_RUN",
        "prepanel_terminal_states": [
            "Q2_V3_PANEL_PROVENANCE_MISMATCH",
            "Q2_V3_CONTROLLER_QUALIFICATION_FAILED",
            "Q2_V3_CONTROLLER_BANK_DESTRUCTIVE",
            "Q2_V3_ANGULAR_IDENTIFIABILITY_FAILED",
            "Q2_V3_PREDICTION_LOCK_FAILED",
            "Q2_V3_INSTRUMENT_FAILURE",
            "Q2_V3_ENGINE_FAILURE",
            "Q2_V3_WALLET_GATE_FAILED",
        ],
        "scientific_hierarchy": {
            "Q2_V3_G3_FINITE_RESPONSE_GEOMETRY_REQUIRED": "M2 passes and every M2-required superiority criterion passes",
            "Q2_V3_G2_FINITE_RESPONSE_RELATIONAL_GEOMETRY_SUPPORTED": "M2 passes, neither M0 nor M1 passes, and M2-required superiority does not pass",
            "Q2_V3_G1_GENERIC_COORDINATE_STATISTICAL_RELATIONAL_GEOMETRY": "M0 or M1 passes and M2-required superiority does not pass, whether or not M2 also passes",
            "Q2_V3_G0_NO_RELATIONAL_GEOMETRY": "none of M0/M1/M2 passes the relational gate",
        },
        "radial_suffix": {"R+": "radial claim passes", "R-": "radial claim fails"},
        "precedence": ["engine/instrument/provenance", "controller safety/qualification", "identifiability", "prediction lock", "G3", "G2", "G1", "G0"],
        "no_signal_wording_for_subcriterion_failure_forbidden": True,
    }
    write_json(REVIEW / "CLASSIFICATION_TAXONOMY.json", taxonomy)

    evaluation_schedule = schedule()
    write_json(REVIEW / "EVALUATION_SCHEDULE.json", evaluation_schedule)
    write_json(
        REVIEW / "SOURCE_QUALIFICATION_SCHEDULE.json",
        source_schedule(allocations["SOURCE_VALIDATION"]["item_ids"]),
    )
    write_json(
        REVIEW / "SHELL_CALIBRATION_SCHEDULE.json",
        shell_schedule(allocations["SHELL_CALIBRATION"]["item_ids"]),
    )

    execution = {
        "schema_version": "q2-v3-execution-plan-v1",
        "status": "FROZEN_NOT_RUN_EXECUTION_REQUIRES_SEPARATE_AUTHORIZATION",
        "model": {
            "id": MODEL,
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dtype": "BF16",
            "quantization": "none",
            "enable_thinking": False,
            "attention": "SDPA",
            "environment": "CORE_QWEN",
        },
        "sampling": {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_new_tokens": 4096},
        "hardware": "NVIDIA A40 48GB; equivalent substitution requires principal review",
        "batch_strategy": "one logical trajectory at a time with KV cache; schedule order exact; no cross-condition batching",
        "phases": {
            "source_construction": "240 paired textual forward captures; no free-generation endpoint",
            "source_qualification": "480 label-free textual-policy trajectories",
            "shell_calibration": "504 matched label-free safety/manipulation trajectories",
            "M1": "64 baseline activation captures",
            "M2": "300 teacher-forced condition-probe captures (25x12), not free generation",
            "semantic_panel": "10,000 trajectories (25x200x2)",
        },
        "nominal_semantic_trajectories": 10_000,
        "trajectory_derivation": "25 conditions x 200 items x 2 independent rollouts",
        "conditions": list(condition_ids()),
        "retry_policy": (
            "only infrastructure failures; same logical key and seed; append retry provenance; "
            "invalid/truncated/model-runtime scientific outputs are retained and never redrawn; "
            "maximum three infrastructure attempts per logical key then ENGINE_FAILURE"
        ),
        "journal": "append, flush, fsync after every row; logical key item_id/condition/rollout_index; resume never duplicates",
        "no_midrun_peeking": "counts, duplicate keys, process/GPU/disk/cost only until all 10,000 rows complete",
        "cost": {
            "q2_v2_observed_reference": "6,960 rows; 2.6602 A40 billed hours; US$1.1853 panel pod cost",
            "linear_10000_row_panel_usd": 1.682,
            "panel_with_25pct_margin_usd": 2.103,
            "expected_full_program_usd": [4.0, 7.0],
            "conservative_projected_cost_usd": 12.0,
            "hard_ceiling_usd": 15.0,
            "safety_reserve_usd": 6.0,
            "wallet_gate_minimum_usd": 18.0,
            "wallet_gate": "available wallet >= US$18.00 AND projected cost with 50% tail margin <= US$15.00",
            "wallet_value_must_be_queried_live": True,
        },
        "stop_rules": [
            "wallet gate failure before provisioning",
            "dataset/model/tokenizer/source commit mismatch",
            "primary panel ID/order/hash/provenance mismatch",
            "source or controller qualification failure",
            "shell target/root/safety failure",
            "null orthogonality or identifiability failure",
            "M0/M1/M2 prediction-lock hash failure",
            "any M3 enablement",
            "schedule missing/duplicate key or seed mismatch",
            "more than three infrastructure attempts for a key",
            "scientific outcome inspection before complete collection",
            "projected hard-ceiling breach",
            "environment or intervention engineering mismatch",
        ],
        "recovery_boundary": "Class-A repair may restore identical frozen environment and resume immutable journal; no item/controller/threshold substitution",
    }
    write_json(REVIEW / "EXECUTION_COST_PLAN.json", execution)

    provenance_text = f"""# Q2 V3 provenance statement

The primary evidence class is **historical-item / prospective-controller
same-domain validation**. Q2 V3 tests prospective generalization to previously
unseen causal controllers on a historical same-domain item distribution whose
items were not implicated in the Q2 V2 geometry discovery that motivated the
radial/angular redesign.

The official CRUXEval census is A={class_counts['A']}, B={class_counts['B']},
C={class_counts['C']}, D={class_counts['D']}, unresolved={class_counts['UNRESOLVED']}.
The 200 primary items are Class C, selected only by provenance eligibility and
the SHA-256 total order under `{PRIMARY_PANEL_NAMESPACE}`. No accuracy,
correctness, difficulty, controller response, complementarity, error profile,
geometry result, manual example, or outcome variance enters selection. Their
ordered-ID SHA-256 is `{primary['ordered_ids_sha256']}`.

The primary panel is disjoint from source construction, source validation,
shell calibration, M1 covariance, and M2 probes. M2 uses 12 Class-B label-free
probes and is computed, hashed, and committed before primary semantic outcomes.
The primary panel must not be called pristine, fresh-item, untouched holdout, or
new-item confirmation.
"""
    (REVIEW / "PROVENANCE_STATEMENT.md").write_text(provenance_text, encoding="utf-8")

    protocol = """# Q2 V3 prospective protocol lock

Status: `Q2_V3_FROZEN_NOT_RUN`

Execution authorization: **NONE in this sprint**.

## Scientific question

After matching implemented intervention amplitude, does pre-outcome internal
geometry prospectively predict which semantic blind spots differ across
genuinely new causal controllers?

## Evidence boundary

This is historical-item / prospective-controller same-domain validation. The
200-item Class-C panel is not fresh-item confirmation. Selection is proven
outcome-independent in `PRIMARY_PANEL_PROVENANCE_PROOF.json`.

## Bank and radial control

Five frozen source families each yield positive-minus-negative paired-mean L27
directions at prompt and execution boundaries. Ten base directions at
implemented-amplitude targets 0.25 and 0.50 produce 20 meaningful controllers.
Two SVD-span-orthogonal Gaussian bases at both shells produce four nulls.
Implemented amplitude is a coordinate-space matching variable, not an intrinsic
model metric. All source, shell, safety, null, and identifiability criteria are
mechanical and precede the semantic panel.

## Competing geometries

M0 is normalized coordinate angular chord, M1 is lambda=0.10 regularized
covariance-whitened angular chord, and M2 is a finite full-vocabulary
Jensen-Shannon response pseudometric on a separate 12-item Class-B probe set.
M2 is not an inner-product angle. M3 is excluded as
`M3_DERIVATIVE_IDENTITIES_FAILED / NOT_QUALIFIED`; this is instrument
nonqualification, not evidence against Fisher or pullback geometry.

## Inference

The primary endpoint is the canonical unbiased two-independent-rollout error
distance on 40 cross-family dyads in each shell. Family-balanced Spearman,
exhaustive 3,840-map family QAP max-statistic inference, 10,000 item-cluster
bootstraps, and ten leave-one-direction-out diagnostics are frozen in
`STATISTICAL_ANALYSIS_PLAN.json`. The relational threshold remains rho >=0.25,
corrected p<=0.05, bootstrap lower>0, 4/5 positive families, both shells
positive, and every LODO estimate positive. M2 is specifically required only
with >=0.10 rho improvement over both simpler metrics and both frozen paired
uncertainty tests. The radial claim is separate and reported as R+/R-.

## Execution boundary

The semantic panel is exactly 25 conditions x 200 items x 2 independent
rollouts = 10,000 trajectories. No trajectory is authorized or created by this
freeze. Before any later run, controller vectors, alphas, amplitudes, nulls,
M0/M1/M2 arrays, metadata, and hashes must pass and be committed in a separate
prediction lock. The wallet must be queried live and satisfy the frozen US$18
gate. Q3 remains not run.
"""
    (REVIEW / "PROTOCOL_LOCK.md").write_text(protocol, encoding="utf-8")

    checklist = """# Q2 V3 mechanical execution checklist

Every box is mandatory. Any failure invokes the named frozen stop state; the
execution agent has no authority to alter a scientific choice.

- [ ] exact execution source commit and clean checkout
- [ ] Qwen/Qwen3-8B and tokenizer exact frozen revision
- [ ] CORE_QWEN BF16/SDPA environment and sustained-current-token hook pass
- [ ] wallet queried live; balance >= US$18 and 50%-tail projection <= US$15
- [ ] all panel/allocation IDs, content hashes, and provenance hashes match
- [ ] source construction and 480-row label-free source qualification complete
- [ ] all five families and ten base directions pass without substitution
- [ ] denominator capture and deterministic roots hit r_impl 0.25/0.50
- [ ] complete 504-row shell safety/manipulation schedule passes
- [ ] two null bases pass SVD-span and pairwise orthogonality
- [ ] M1 covariance and M2 Class-B probe archives complete and hash-clean
- [ ] exactly M0/M1/M2 24x24 matrices produced; M3 absent
- [ ] every identifiability criterion passes
- [ ] prediction arrays/metadata/lock committed and pushed before outcomes
- [ ] 10,000-row schedule and all seeds match; journal starts empty
- [ ] no scientific peeking until exactly 10,000 unique rows
- [ ] raw recovery, complete primary analysis, independent forensic audit
- [ ] GPU and retained volumes removed after verified recovery
- [ ] Q3 not run
"""
    (REVIEW / "EXECUTION_CHECKLIST.md").write_text(checklist, encoding="utf-8")

    referenced = [
        "PRIMARY_PANEL_MANIFEST.json",
        "PRIMARY_PANEL_PROVENANCE_PROOF.json",
        "SOURCE_CONSTRUCTION_MANIFEST.json",
        "SOURCE_VALIDATION_MANIFEST.json",
        "SHELL_CALIBRATION_MANIFEST.json",
        "M1_COVARIANCE_MANIFEST.json",
        "M2_PROBE_MANIFEST.json",
        "CONTROLLER_BANK_SPEC.json",
        "SHELL_CALIBRATION_SPEC.json",
        "IDENTIFIABILITY_GATE.json",
        "GEOMETRY_DEFINITIONS.json",
        "PREDICTION_LOCK_SPEC.json",
        "STATISTICAL_ANALYSIS_PLAN.json",
        "CLASSIFICATION_TAXONOMY.json",
        "EVALUATION_SCHEDULE.json",
        "SOURCE_QUALIFICATION_SCHEDULE.json",
        "SHELL_CALIBRATION_SCHEDULE.json",
        "EXECUTION_COST_PLAN.json",
        "PROVENANCE_STATEMENT.md",
        "EXECUTION_CHECKLIST.md",
    ]
    hashes = {name: sha256(REVIEW / name) for name in referenced}
    lock = {
        "schema_version": "q2-v3-prospective-lock-v1",
        "status": "Q2_V3_FROZEN_NOT_RUN",
        "experiment_id": EXPERIMENT_ID,
        "reviewed_base_commit": REVIEWED_BASE,
        "experiment_source_commit": source_commit,
        "execution_authorized": False,
        "scientific_question": "After matching implemented intervention amplitude, does pre-outcome internal geometry prospectively predict which semantic blind spots differ across genuinely new causal controllers?",
        "evidence_class": "HISTORICAL_ITEM_PROSPECTIVE_CONTROLLER_SAME_DOMAIN_VALIDATION",
        "model": {"id": MODEL, "revision": MODEL_REVISION, "layer": LAYER},
        "panel": {"n": 200, "class": "C", "ordered_ids_sha256": primary["ordered_ids_sha256"]},
        "bank": {"families": 5, "base_directions": 10, "shells": SHELL_TARGETS, "meaningful": 20, "nulls": 4},
        "geometries": ["M0", "M1", "M2"],
        "M3": "EXCLUDED_NOT_QUALIFIED_M3_DERIVATIVE_IDENTITIES_FAILED",
        "semantic_panel": {"conditions": 25, "items": 200, "rollouts": 2, "rows": 10_000, "executed": False},
        "source_or_shell_outputs_in_this_freeze": 0,
        "semantic_outcomes_in_this_freeze": 0,
        "q3": "NOT_RUN",
        "artifact_hashes": hashes,
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", lock)

    spec_text = f"""experiment_id: {EXPERIMENT_ID}
status: Q2_V3_FROZEN_NOT_RUN
execution_authorized: false
experiment_source_commit: {source_commit}
evidence_class: HISTORICAL_ITEM_PROSPECTIVE_CONTROLLER_SAME_DOMAIN_VALIDATION
model:
  id: {MODEL}
  revision: {MODEL_REVISION}
  tokenizer_revision: {MODEL_REVISION}
  dtype: BF16
  attention: SDPA
  enable_thinking: false
  max_new_tokens: 4096
controller_bank:
  layer: 27
  families: 5
  source_locations: [PROMPT_BOUNDARY, EXECUTION_BOUNDARY]
  shells: {{MEDIUM: 0.25, STRONG: 0.50}}
  meaningful_controllers: 20
  null_controllers: 4
geometries: [M0, M1, M2]
m3: EXCLUDED_NOT_QUALIFIED
primary_panel:
  provenance_class: C
  items: 200
  ordered_ids_sha256: {primary['ordered_ids_sha256']}
common_panel:
  conditions: 25
  rollouts_per_item_condition: 2
  expected_rows: 10000
  seed_regime: INDEPENDENT_PRIMARY
cost:
  expected_usd: [4.0, 7.0]
  conservative_usd: 12.0
  hard_ceiling_usd: 15.0
  wallet_gate_minimum_usd: 18.0
firewall:
  semantic_panel_run: false
  q3: NOT_RUN
"""
    SPEC.parent.mkdir(parents=True, exist_ok=True)
    SPEC.write_text(spec_text, encoding="utf-8")

    final_hashes = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(REVIEW.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    final_hashes[str(SPEC.relative_to(ROOT))] = {
        "bytes": SPEC.stat().st_size,
        "sha256": sha256(SPEC),
    }
    write_json(REVIEW / "artifact_hashes.json", final_hashes)
    print(
        json.dumps(
            {
                "status": "Q2_V3_FREEZE_ARTIFACTS_MATERIALIZED_NOT_RUN",
                "source_commit": source_commit,
                "primary_panel_hash": primary["ordered_ids_sha256"],
                "semantic_trajectories": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
