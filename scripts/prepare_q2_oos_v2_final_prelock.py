#!/usr/bin/env python3
"""Create the final prospective Q2 OOS V2 protocol lock and PRELOCK."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
OOS = ROOT / "review/q2_oos_fresh_controller_design"
ROBUST = OOS / "heterogeneity_robust_inference"
V4 = ROOT / "review/q2_v4_spark1_presemantic"
V41 = ROOT / "review/q2_v4_1_prediction_lock"
SAFE = ROOT / "review/q2_v4_1_31_safe_bank_review"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_hashes() -> dict[str, str]:
    sources = {
        "rank8_Q_basis": V4 / "SPARK1_SUBSPACE_Q.npy",
        "rank8_Q_qualification": V4 / "SPARK1_SUBSPACE_QUALIFICATION.json",
        "V4_candidate_manifest": V4 / "CANDIDATE_BANK_MANIFEST.json",
        "V1_candidate_manifest": OOS / "CANDIDATE_BANK_MANIFEST.json",
        "historical_safe31_manifest": SAFE / "SAFE_31_IMMUTABLE_MANIFEST.json",
        "safety_items": V4 / "SHELL_CALIBRATION_MANIFEST.json",
        "safety_protocol": V4 / "QUALIFICATION_PROTOCOL_LOCK.json",
        "A1_covariance_manifest": V41 / "A1_COVARIANCE_MANIFEST.json",
        "A1_frozen_fit": V41 / "A1_COVARIANCE_FIT.npz",
        "A2_probe_manifest": V41 / "A2_PROBE_MANIFEST.json",
        "A2_raw_archive_hashes": V41 / "A2_RAW_ARCHIVE_HASHES.json",
        "environment_profile": V41 / "ENVIRONMENT_PROVENANCE_SPARK1_CAPTURE.json",
        "semantic_panel_manifest": V41 / "SEMANTIC_PANEL_MANIFEST.json",
        "accepted_revised_design": ROBUST / "REVISED_OOS_PROTOCOL_DRAFT.json",
        "zero_erratum_precheck": ROBUST / "ZERO_HANDLING_SIGN_TEST_ERRATUM_PRECHECK.json",
        "zero_recalibration": ROBUST / "ZERO_HANDLING_SIGN_TEST_RECALIBRATION.json",
        "runtime_autopsy": ROBUST / "HISTORICAL_Q2_RUNTIME_AUTOPSY.json",
        "sign_test_code": ROOT / "src/epistemic_geometry/experiments/heterogeneity_robust.py",
        "sign_test_tests": ROOT / "tests/test_heterogeneity_robust.py",
        "stream_materializer": ROOT / "scripts/materialize_q2_oos_v2_candidates.py",
        "prelock_preparer": ROOT / "scripts/prepare_q2_oos_v2_final_prelock.py",
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing V2 PRELOCK sources: {missing}")
    return {name: sha256(path) for name, path in sources.items()}


def main() -> None:
    recalibration = read_json(ROBUST / "ZERO_HANDLING_SIGN_TEST_RECALIBRATION.json")
    if recalibration["classification"] != "Q2_OOS_V2_SIGN_TEST_CALIBRATED":
        raise RuntimeError("Q2_OOS_V2_SIGN_TEST_CALIBRATION_BLOCKED")
    hashes = source_hashes()
    lock = {
        "schema_version": "q2-oos-v2-final-presemantic-protocol-lock-v1",
        "status": "Q2_OOS_V2_FINAL_PROTOCOL_FROZEN",
        "source_commit_before_prelock": git_head(),
        "prelock_commit": "THE_COMMIT_CONTAINING_THIS_ARTIFACT",
        "historical_states_immutable": [
            "Q2_OOS_FRESH_CONTROLLER_DESIGN_BLOCKED",
            "Q2_OOS_V2_NULL_CALIBRATION_BLOCKED",
            "Q2_V4_1_G2",
            "RS+",
            "RT+",
            "Q2_V4_1_HETEROGENEITY_ROBUST_SUPPORT"
        ],
        "design": {
            "K": 16,
            "candidate_count": 34,
            "reference_controllers": 31,
            "items": 300,
            "shells": ["MEDIUM", "STRONG"],
            "rollouts": 2,
            "future_semantic_trajectories": 19200,
            "future_streams": 1
        },
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "layer": 27,
            "timing": "sustained-current-token",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "backend": "Spark 1 only"
        },
        "candidate_generation": {
            "namespace": "Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V2",
            "basis": "exact frozen rank-8 SPARK1_SUBSPACE_Q.npy",
            "rng": "NumPy PCG64DXSM",
            "draw": "g~N(0,I_8); c=g/||g||; v=Qc",
            "seed_rule": (
                "big-endian first 128 bits "
                "SHA256('Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V2|' + PRELOCK_COMMIT)"
            ),
            "generate_once": True,
            "redraws": 0,
            "replacement": "FORBIDDEN",
            "additional_candidates": "FORBIDDEN",
            "exclude_all_V1_controllers": True,
            "publish_before_model_use": True
        },
        "candidate_stream_gate": {
            "finite": True,
            "rank": 8,
            "coefficient_unit_norm_error_max": 1e-12,
            "vector_unit_norm_error_max": 1e-10,
            "maximum_absolute_pair_cosine_max_exclusive": 0.98,
            "V1_exact_or_hash_overlap": "FORBIDDEN",
            "effective_rank": "DESCRIPTIVE_ONLY",
            "condition_number": "DESCRIPTIVE_ONLY"
        },
        "safety": {
            "candidates": 34,
            "shells": {"MEDIUM": 0.25, "STRONG": 0.50},
            "items": 12,
            "rollouts": 2,
            "planned_trajectories": 1632,
            "correctness": "FORBIDDEN",
            "inherits_exact_V4_1_gate": True,
            "candidate_requires_both_shells": True,
            "selection": "first 16 candidates passing both shells in immutable order",
            "minimum_safe": 16,
            "replacement": "FORBIDDEN",
            "runtime_based_selection": "FORBIDDEN"
        },
        "selected_bank_gate": {
            "count": 16,
            "rank": 8,
            "effective_rank_min": 4.8,
            "maximum_absolute_pair_cosine_max_exclusive": 0.98,
            "fresh_by_reference_A0_q90_minus_q10_min": 0.20,
            "shell_amplitude_cv_max": 0.03,
            "condition_number": "DESCRIPTIVE_ONLY",
            "cross_block_row_diversity": "DESCRIPTIVE_REQUIRED",
            "fresh_by_reference_geometry_LOFO": "DESCRIPTIVE_REQUIRED",
            "predicted_power": "MANDATORY_DIAGNOSTIC_ONLY_NO_TERMINAL_THRESHOLD"
        },
        "label_free_geometry": {
            "A0": "primary coordinate angular dissimilarity",
            "A1": "secondary using exact frozen V4.1 whitening fit",
            "A2": "secondary fresh label-free finite-response capture",
            "D2": "inherited secondary",
            "A1_A2_qualification": "inherit V4.1 algebraic and forensic gates",
            "correctness": "FORBIDDEN",
            "semantic_panel": "FORBIDDEN"
        },
        "fresh_old_primary": {
            "row_statistic": (
                "r_i = 0.5*(Spearman_j(A0_MEDIUM(i,j),Dshape_MEDIUM(i,j)) + "
                "Spearman_j(A0_STRONG(i,j),Dshape_STRONG(i,j)))"
            ),
            "reference_order": "exact frozen 31-controller V4.1 order",
            "bernoulli": "X_i=1 iff finite r_i>0; X_i=0 iff finite r_i<=0",
            "nonfinite": "Q2_OOS_V2_INFERENCE_DEGENERATE",
            "null": "P(r_i>0)<=0.5",
            "alternative": "P(r_i>0)>0.5",
            "test": "exact one-sided Binomial upper-tail",
            "denominator": 16,
            "positive_count_min": 12,
            "alpha": 0.05,
            "mandatory": ["all 16 r_i finite", "positive_count>=12", "exact_p<=0.05"],
            "descriptive": [
                "complete r_i distribution", "median r_i", "mean r_i",
                "positive zero negative counts", "global cross-block rho",
                "shell-specific global rho"
            ],
            "sensitivities": [
                "original row-QAP diagnostic only",
                "studentized mean controller-level sensitivity",
                "controller-cluster bootstrap",
                "item bootstrap",
                "LOFO"
            ],
            "rescue_by_sensitivity": "FORBIDDEN"
        },
        "fresh_fresh_secondary": {
            "method": "NODE_JACKKNIFE_PSEUDOVALUE_T",
            "role": "SECONDARY_ONLY_CANNOT_RESCUE_PRIMARY"
        },
        "runtime_reference_hours": {
            "normal_superpopulation": {
                "mean": 44.15,
                "p50": 43.84,
                "p80": 48.52,
                "p90": 51.17,
                "p95": 53.42,
                "p99": 57.86
            },
            "cap_stress_1_5x": {"p50": 62.25, "p95": 76.78},
            "cap_stress_2x": {"p50": 80.66, "p95": 100.17},
            "cannot_affect_selection": True
        },
        "future_schedule": {
            "rows": 19200,
            "balance": "deterministically interleave controller shell rollout and item region",
            "runtime_or_semantic_adaptation": "FORBIDDEN",
            "generation": "exact inherited V4.1 normative generation contract",
            "retry_resume": "exact inherited V4.1 operational-only retry and missing-key resume"
        },
        "terminal_states": [
            "Q2_OOS_V2_STREAM_INTEGRITY_BLOCKED",
            "Q2_OOS_V2_SAFE_BANK_INSUFFICIENT",
            "Q2_OOS_V2_SELECTED_BANK_NOT_QUALIFIED",
            "Q2_OOS_V2_LABEL_FREE_INSTRUMENT_NOT_QUALIFIED",
            "Q2_OOS_V2_READY_FOR_PREDICTION_LOCK"
        ],
        "source_hashes": hashes,
        "semantic_execution_authorized": False,
        "semantic_N300_trajectories": 0,
        "correctness_inspected": False,
        "Spark2": "FORBIDDEN",
        "RunPod": "FORBIDDEN",
        "Q3": "NOT_RUN"
    }
    prelock = {
        "schema_version": "q2-oos-v2-final-prelock-v1",
        "status": "Q2_OOS_V2_PRELOCK_READY_FOR_COMMIT",
        "prelock_commit": "THE_COMMIT_CONTAINING_THIS_ARTIFACT",
        "protocol_lock_sha256": "PENDING_WRITE",
        "K": 16,
        "candidate_count": 34,
        "namespace": "Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V2",
        "candidate_stream_exists": False,
        "actual_seed_derived": False,
        "new_controller_streams": 0,
        "semantic_N300_trajectories": 0,
        "correctness_inspected": False,
        "source_hashes": hashes
    }
    write_json(REVIEW / "V2_FINAL_PROTOCOL_LOCK.json", lock)
    prelock["protocol_lock_sha256"] = sha256(REVIEW / "V2_FINAL_PROTOCOL_LOCK.json")
    write_json(REVIEW / "V2_PRELOCK.json", prelock)
    print(
        json.dumps(
            {
                "status": prelock["status"],
                "K": 16,
                "candidate_count": 34,
                "candidate_stream_exists": False,
                "actual_seed_derived": False
            },
            indent=2
        )
    )


if __name__ == "__main__":
    main()
