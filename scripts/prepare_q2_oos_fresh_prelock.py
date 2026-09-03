#!/usr/bin/env python3
"""Prepare the prospective Q2 fresh-controller presemantic lock."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from epistemic_geometry.experiments.q2_oos_fresh_controller import (
    BOOTSTRAP_RESAMPLES,
    EXPERIMENT_ID,
    LAYER,
    MODEL,
    MODEL_REVISION,
    PRIMARY_N,
    QAP_MAPS,
    REFERENCE_COUNT,
    ROLLOUTS,
    SHELL_TARGETS,
    SHELLS,
)

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review" / "q2_oos_fresh_controller_design"
REFERENCE = ROOT / "review" / "q2_v4_1_prediction_lock"
V4 = ROOT / "review" / "q2_v4_spark1_presemantic"
SAFE = ROOT / "review" / "q2_v4_1_31_safe_bank_review"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_hashes() -> dict[str, str]:
    paths = {
        "reference_closeout": ROOT
        / "review/q2_v4_1_semantic_execution/Q2_V4_1_SEMANTIC_CLOSEOUT.json",
        "reference_safe_bank": SAFE / "SAFE_31_IMMUTABLE_MANIFEST.json",
        "reference_candidate_stream": V4 / "CANDIDATE_BANK_MANIFEST.json",
        "source_subspace_q": V4 / "SPARK1_SUBSPACE_Q.npy",
        "source_subspace_qualification": V4 / "SPARK1_SUBSPACE_QUALIFICATION.json",
        "safety_items": V4 / "SHELL_CALIBRATION_MANIFEST.json",
        "safety_protocol": V4 / "QUALIFICATION_PROTOCOL_LOCK.json",
        "a1_covariance_manifest": REFERENCE / "A1_COVARIANCE_MANIFEST.json",
        "a1_covariance_fit": REFERENCE / "A1_COVARIANCE_FIT.npz",
        "a2_probe_manifest": REFERENCE / "A2_PROBE_MANIFEST.json",
        "a2_raw_archive_hashes": REFERENCE / "A2_RAW_ARCHIVE_HASHES.json",
        "a0_medium": REFERENCE / "A0_MEDIUM.npy",
        "a0_strong": REFERENCE / "A0_STRONG.npy",
        "reference_prediction_matrices": REFERENCE / "PREDICTION_MATRICES.npz",
        "semantic_panel_manifest": REFERENCE / "SEMANTIC_PANEL_MANIFEST.json",
        "environment_profile": REFERENCE / "ENVIRONMENT_PROVENANCE_SPARK1_CAPTURE.json",
        "power_precision": REVIEW / "POWER_PRECISION.json",
        "power_table": REVIEW / "POWER_PRECISION.csv",
        "power_summary": REVIEW / "POWER_PRECISION.md",
        "dependence_sensitivity": REVIEW / "CONTROLLER_DEPENDENCE_SENSITIVITY.csv",
        "reserve_feasibility": REVIEW / "RESERVE_FEASIBILITY.json",
        "outcome_free_primitives": ROOT
        / "src/epistemic_geometry/experiments/q2_oos_fresh_controller.py",
        "power_code": ROOT / "scripts/design_q2_oos_fresh_controller.py",
        "prelock_preparer": ROOT / "scripts/prepare_q2_oos_fresh_prelock.py",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing required prelock sources: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def write_design_documents(k: int, candidates: int) -> None:
    (REVIEW / "DESIGN_REVIEW.md").write_text(
        f"""# Q2 fresh-controller out-of-bank design

This prospective experiment tests whether the closed Q2 V4.1 A0 relational
association generalizes to entirely fresh controller identities sampled from
the same frozen rank-8 intervention subspace.  It does not test new items,
tasks, models, subspaces, global smoothness, or utility.

The design uses **K={k}** fresh safe controllers selected as the first safe
members of one immutable {candidates}-candidate PCG64DXSM stream.  It never
optimizes the bank against old or new semantic outcomes.  Future semantic
inference, which is not authorized here, would add {1200 * k:,} trajectories
and reuse the sealed 31-controller reference outcomes and sealed baseline.

The primary metric is A0.  The statistic is the equal-weight mean of MEDIUM
and STRONG Spearman correlations over the complete K×31 cross block.  The
randomization permutes complete fresh-controller rows while keeping frozen
reference columns fixed; this directly tests fresh-identity alignment while
preserving row dependence.  A1 and A2 are predeclared secondary metrics and
do not create an A2-superiority claim.

The primary A0 replication requires positive aggregate rho, one-sided
fresh-row permutation p<=0.05, positive 2.5th-percentile item-bootstrap bound,
and positive aggregate rho in every leave-one-fresh-controller-out fold.
Leave-one-reference-out is descriptive stability, not an extra terminal gate.
"""
    )
    (REVIEW / "MATCHED_RANDOM_SUBSPACE_DESIGN_MEMO.md").write_text(
        """# Matched random-subspace control — model-free future design

This memo specifies a later alternative-explanation test and does not generate
or execute a random subspace in the current sprint.

Construct an ambient L27 rank-8 orthonormal basis from one prospectively frozen
isotropic random draw, conditioned only on numerical orthonormality.  Match the
meaningful-subspace experiment on model revision, layer, rank, candidate count,
PCG64DXSM controller construction, MEDIUM/STRONG implemented norm targets,
safety data and gates, safety attrition rule, safe K, 300-item panel, two
rollouts, Dshape estimator, cross-block statistic, row-label permutation,
item bootstrap, and stability analyses.  Fit no transform to semantic data.

Because safety conditioning can differ between subspaces, predeclare the same
reserve policy and report attrition rather than redrawing.  A fair comparison
must either qualify both banks under identical gates or stop.  The future
contrast is whether a matched random subspace yields comparable relational
association; it is not authorized and must not delay the fresh-controller
validation in the frozen meaningful subspace.
"""
    )


def main() -> None:
    power = read_json(REVIEW / "POWER_PRECISION.json")
    k = int(power["recommended_K"])
    candidates = int(power["recommended_candidate_count"])
    if (k, candidates) != (10, 19):
        raise RuntimeError(f"unexpected mechanical design recommendation: {(k, candidates)}")
    hashes = source_hashes()
    write_design_documents(k, candidates)
    inference = {
        "primary_metric": "A0",
        "secondary_metrics": ["A1", "A2"],
        "statistic": "equal-weight mean of shell-specific cross-block Spearman correlations",
        "cross_block_shape": [k, REFERENCE_COUNT],
        "permutation": {
            "group": "complete fresh-controller row-label permutations",
            "reference_columns": "fixed",
            "justification": (
                "conditions on the sealed reference bank and tests whether fresh identity-specific "
                "geometry rows align with fresh semantic rows; individual dyads are never permuted"
            ),
            "maps": QAP_MAPS,
            "identity_included": True,
            "nonidentity_unique": QAP_MAPS - 1,
            "same_map_both_shells_all_metrics": True,
            "seed_rule": "first128bits SHA256('Q2-OOS-FRESH-ROW-QAP-V1|' + PRELOCK_COMMIT)",
            "p_value": "count(T_perm >= T_observed) / 50000",
            "primary_multiplicity": "none; A0 is the single prospectively primary metric",
            "secondary_multiplicity": (
                "Holm correction across A1 and A2; descriptive and non-primary"
            ),
        },
        "bootstrap": {
            "unit": "item",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed_rule": "first128bits SHA256('Q2-OOS-ITEM-BOOTSTRAP-V1|' + PRELOCK_COMMIT)",
            "interval": "percentile [0.025,0.975]",
            "cluster": (
                "all 31 reference controllers, all fresh controllers, both shells, both rollouts"
            ),
            "duplicates": "sampled items retain their bootstrap multiplicity",
            "negative_Dshape": "retained",
        },
        "stability": {
            "leave_one_fresh_out": "required: every aggregate A0 rho strictly positive",
            "leave_one_reference_out": "reported descriptively; not a terminal gate",
        },
        "radial_secondary": {
            "baseline": "reuse the sealed V4.1 baseline; no baseline rerun",
            "reason": (
                "same model, panel, parser, generation contract and two independent "
                "baseline rollouts already sealed"
            ),
            "statistic": (
                "median over fresh controllers of Dshape(BASELINE,STRONG)-Dshape(BASELINE,MEDIUM)"
            ),
            "positive_rule": (
                "median > 0 AND exact one-sided sign p<=0.05 (at K=10 requires >=9 positives) "
                "AND item-bootstrap lower bound > 0"
            ),
            "continuous_dose_response_claim": False,
        },
        "terminal_states": {
            "Q2_OOS_FRESH_CONTROLLER_A0_PASS": "P1 AND P2 AND P3 AND P4",
            "Q2_OOS_FRESH_CONTROLLER_ASSOCIATION_INCOMPLETE": (
                "aggregate A0 rho > 0 and at least one of permutation/bootstrap/LOFO fails"
            ),
            "Q2_OOS_FRESH_CONTROLLER_NO_REPLICATION": (
                "aggregate A0 rho <= 0 or permutation p > 0.05"
            ),
            "Q2_OOS_FRESH_CONTROLLER_INSTRUMENT_NOT_QUALIFIED": (
                "presemantic bank or A1/A2 qualification fails"
            ),
            "Q2_OOS_FRESH_CONTROLLER_EXECUTION_INCOMPLETE": (
                "future semantic schedule cannot be completed"
            ),
        },
    }
    safety = {
        "inherits_exact_v4_1_gate": True,
        "items": 12,
        "rollouts": 2,
        "matched_seed_with_baseline": True,
        "correctness": "FORBIDDEN",
        "validity_min": 0.90,
        "relative_validity_drop_max": 0.05,
        "evaluability_min": 0.90,
        "relative_evaluability_drop_max": 0.05,
        "truncation_max": 0.05,
        "movement_min": {"MEDIUM": 0.10, "STRONG": 0.15},
        "relative_target_error_max": 0.005,
        "candidate_requires_both_shells": True,
        "selection": "first 10 safe candidates in immutable generation order",
        "minimum_safe": k,
        "candidates_after_stream": "FORBIDDEN",
        "redraw": "FORBIDDEN",
    }
    lock = {
        "schema_version": "q2-oos-fresh-controller-presemantic-lock-v1",
        "status": "Q2_OOS_FRESH_CONTROLLER_PROTOCOL_FROZEN",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": git_head(),
        "prelock_commit": "THE_COMMIT_CONTAINING_THIS_ARTIFACT",
        "reference_result": "Q2_V4_1_G2",
        "reference_radial": ["RS+", "RT+"],
        "reference_controllers": REFERENCE_COUNT,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "layer": LAYER,
        "timing": "sustained-current-token",
        "subspace": "exact frozen rank-8 SPARK1_SUBSPACE_Q.npy",
        "fresh_controller_policy": {
            "K": k,
            "candidate_count": candidates,
            "reserve": candidates - k,
            "namespace": "Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V1",
            "draw": "g~N(0,I_8); c=g/||g||; v=Qc",
            "rng": "NumPy PCG64DXSM",
            "seed_rule": (
                "big-endian first 128 bits "
                "SHA256('Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V1|' + PRELOCK_COMMIT)"
            ),
            "generate_once": True,
            "redraw": "FORBIDDEN",
            "selection": "first K safe in generation order",
            "historical_overlap": "FORBIDDEN",
        },
        "candidate_stream_gate": {
            "rank": 8,
            "entropy_effective_rank_min": 6.0,
            "condition_number_max": 3.0,
            "maximum_absolute_pair_cosine_max": 0.98,
            "unit_norm_error_max": 1e-12,
        },
        "selected_bank_gate": {
            "count": k,
            "rank": 8,
            "entropy_effective_rank_min": 4.8,
            "condition_number_max": 10.0,
            "maximum_absolute_pair_cosine_max": 0.98,
            "cross_block_A0_q90_minus_q10_min": 0.20,
            "shell_amplitude_cv_max": 0.03,
            "purpose": (
                "gross multidimensional identifiability, not spherical aesthetic optimization"
            ),
        },
        "shells": SHELL_TARGETS,
        "safety_gate": safety,
        "geometry": {
            "A0": "primary coordinate-space angular dissimilarity",
            "A1": "secondary frozen V4.1 whitening fit",
            "A2": (
                "secondary fresh label-free finite-response capture; "
                "pre-semantic-outcome but not pre-intervention"
            ),
            "A2_superiority_test": False,
        },
        "inference": inference,
        "future_semantic": {
            "authorized": False,
            "panel_N": PRIMARY_N,
            "rollouts": ROLLOUTS,
            "fresh_conditions_only": k * len(SHELLS),
            "trajectories": k * len(SHELLS) * PRIMARY_N * ROLLOUTS,
            "old_reference_rerun": False,
            "baseline_rerun": False,
        },
        "backend": {
            "presemantic": "Spark 1 only",
            "Spark2": "FORBIDDEN",
            "RunPod": "FORBIDDEN",
            "semantic_panel": "FORBIDDEN",
            "correctness": "FORBIDDEN",
        },
        "source_hashes": hashes,
        "historical_results_mutable": False,
        "LiveCodeBench_outputs_accessed": False,
        "Q3": "NOT_RUN",
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", lock)
    prelock = {
        "schema_version": "q2-oos-fresh-controller-prelock-v1",
        "status": "Q2_OOS_FRESH_CONTROLLER_PRELOCK_READY_FOR_COMMIT",
        "prelock_commit": "THE_COMMIT_CONTAINING_THIS_ARTIFACT",
        "K": k,
        "candidate_count": candidates,
        "candidate_bank_exists": False,
        "semantic_outcomes": 0,
        "correctness_inspected": False,
        "protocol_lock_sha256": sha256(REVIEW / "PROTOCOL_LOCK.json"),
        "design_review_sha256": sha256(REVIEW / "DESIGN_REVIEW.md"),
        "random_subspace_memo_sha256": sha256(REVIEW / "MATCHED_RANDOM_SUBSPACE_DESIGN_MEMO.md"),
        "source_hashes": hashes,
    }
    write_json(REVIEW / "PRELOCK.json", prelock)
    print(json.dumps({"status": prelock["status"], "K": k, "candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
