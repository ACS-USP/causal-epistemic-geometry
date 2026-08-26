#!/usr/bin/env python3
"""Seal the source-qualified V4 protocol immediately before bank generation."""

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

from epistemic_geometry.experiments.q2_v4_presemantic import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    CANDIDATE_COUNT,
    PRIMARY_N,
    QAP_MAPS,
    SELECTED_COUNT,
    SHELL_TARGETS,
)

REVIEW = ROOT / "review/q2_v4_spark1_presemantic"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def analysis_plan() -> dict[str, Any]:
    return {
        "schema_version": "q2-v4-statistical-analysis-plan-v1",
        "status": "FROZEN_PRE_OUTCOME",
        "semantic_execution_authorized": False,
        "panel": {
            "N": PRIMARY_N,
            "conditions": 65,
            "rollouts": 2,
            "expected_rows": 39_000,
            "seed_regime": "INDEPENDENT_PRIMARY",
        },
        "endpoint": {
            "D_total": "mean_t[(e_i,t,0-e_j,t,0)*(e_i,t,1-e_j,t,1)]",
            "mean_shift_product": "mean_t(d_t,0)*mean_t(d_t,1)",
            "D_shape_panel": "D_total-mean_shift_product",
            "D_shape_superpopulation": "N/(N-1)*D_shape_panel",
            "negative_estimates": "RETAIN",
        },
        "angular_metrics": ["A0", "A1", "A2"],
        "secondary_metric": "D2",
        "primary_statistic": (
            "equal-weight mean of shell-specific upper-triangle Spearman correlations"
        ),
        "metric_gate": {
            "maxT_adjusted_p_max": 0.05,
            "shell_mean_rho_min": 0.20,
            "both_shell_rhos_strictly_positive": True,
            "item_bootstrap_lower_bound_strictly_positive": True,
            "delete_one_controller_sign_stable": True,
        },
        "QAP": {
            "maps_total": QAP_MAPS,
            "identity_first": True,
            "unique_sampled_maps": QAP_MAPS - 1,
            "rng": "NumPy PCG64DXSM",
            "seed": "first128bits SHA256('Q2-V4-QAP-V1|' + PRELOCK_COMMIT), big-endian",
            "same_permutation_both_shells_all_metrics": True,
            "p_value": "count(T_perm>=T_observed)/50000",
            "multiplicity": "single-step maxT across A0/A1/A2",
        },
        "G3": {
            "A2_full_gate_required": True,
            "rho_A2_minus_A0_min": 0.10,
            "rho_A2_minus_A1_min": 0.10,
            "paired_bootstrap_lower_bounds_positive": True,
            "two_contrast_single_step_maxT_p_max": 0.05,
        },
        "classification": {
            "V4-G0": "no primary angular metric qualifies",
            "V4-G1": "A0 and/or A1 qualifies without separate A2 support",
            "V4-G2": "A2 qualifies",
            "V4-G3": "A2 qualifies and passes frozen superiority requirements",
        },
        "radial": {
            "R_total": "D_total(BASELINE,strong)-D_total(BASELINE,medium)",
            "R_shape": "D_shape(BASELINE,strong)-D_shape(BASELINE,medium)",
            "maps_total": QAP_MAPS,
            "paired_shell_swaps": True,
            "rng": "NumPy PCG64DXSM",
            "seed": ("first128bits SHA256('Q2-V4-RADIAL-SWAPS-V1|' + PRELOCK_COMMIT), big-endian"),
            "positive_gate": {
                "median_strictly_positive": True,
                "permutation_p_max": 0.05,
                "item_bootstrap_lower_bound_strictly_positive": True,
                "positive_directions_min": 22,
            },
            "statuses": ["RT+", "RT-", "RS+", "RS-"],
            "independent_of_G_classification": True,
        },
        "bootstrap": {
            "item_cluster_resamples": BOOTSTRAP_RESAMPLES,
            "unit": "item; move all 65 conditions and both rollouts together",
        },
        "Q3": "NOT_RUN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-execution-commit", required=True)
    args = parser.parse_args()
    required = {
        "SPARK1_ENGINE_QUALIFICATION.json": "Q2_V4_SPARK1_ENGINE_QUALIFIED",
        "SPARK1_SOURCE_BASIS_QUALIFICATION.json": ("Q2_V4_SPARK1_SOURCE_BASIS_QUALIFIED"),
        "SPARK1_SUBSPACE_QUALIFICATION.json": "Q2_V4_SUBSPACE_QUALIFIED",
    }
    for name, classification in required.items():
        if read_json(REVIEW / name)["classification"] != classification:
            raise RuntimeError(f"prelock gate failed: {name}")
    forbidden = (
        REVIEW / "CANDIDATE_BANK_MANIFEST.json",
        REVIEW / "CANDIDATE_DIRECTIONS",
        REVIEW / "CANDIDATE_SAFETY_JOURNAL.jsonl",
        REVIEW / "FUTURE_SEMANTIC_JOURNAL.jsonl",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("a bank or future semantic artifact exists before PRELOCK")
    source_rows = []
    with (REVIEW / "SOURCE_JOURNAL.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            source_rows.append(json.loads(line)["row"])
    keys = {
        (row["item_id"], row["family"], row["polarity"], row["rollout_index"])
        for row in source_rows
    }
    if len(source_rows) != 384 or len(keys) != 384:
        raise RuntimeError("source journal is incomplete or duplicated")
    if any(row.get("correctness_evaluated") is not False for row in source_rows):
        raise RuntimeError("source qualification touched correctness")
    panel = read_json(REVIEW / "PRIMARY_PANEL_MANIFEST.json")
    if panel["item_count"] != PRIMARY_N or panel["semantic_outcomes"] != 0:
        raise RuntimeError("future semantic panel firewall failed")
    plan = analysis_plan()
    write_json(REVIEW / "V4_STATISTICAL_ANALYSIS_PLAN.json", plan)
    (REVIEW / "V4_STATISTICAL_ANALYSIS_PLAN.md").write_text(
        "# Q2 V4 statistical analysis plan\n\n"
        "Status: `FROZEN_PRE_OUTCOME`. Semantic execution is not authorized.\n\n"
        "The primary endpoint is the N/(N-1)-corrected two-independent-rollout "
        "blind-spot-shape distance. A0, A1, and A2 are all reported under one "
        "50,000-map shell-coupled controller-label QAP with single-step maxT. "
        "R-total and R-shape are distinct and receive independent RT/RS statuses. "
        "The machine-readable JSON is authoritative.\n",
        encoding="utf-8",
    )
    paths = (
        "QUALIFICATION_PROTOCOL_LOCK.json",
        "SPARK1_ENVIRONMENT_LOCK.json",
        "EXACT_MODEL_MANIFEST.json",
        "SPARK1_ENGINE_QUALIFICATION.json",
        "SOURCE_ACTIVATIONS.npz",
        "SOURCE_JOURNAL.jsonl",
        "SPARK1_SOURCE_BASIS_QUALIFICATION.json",
        "SPARK1_SOURCE_MATRIX.npy",
        "SPARK1_SUBSPACE_Q.npy",
        "SPARK1_SUBSPACE_QUALIFICATION.json",
        "PRIMARY_PANEL_MANIFEST.json",
        "DATA_PURPOSE_LEDGER.json",
        "V4_STATISTICAL_ANALYSIS_PLAN.json",
    )
    hashes = {name: sha256(REVIEW / name) for name in paths}
    payload = {
        "schema_version": "q2-v4-prelock-v1",
        "status": "Q2_V4_PRELOCK_READY_FOR_COMMIT",
        "prelock_commit": "THE_COMMIT_CONTAINING_THIS_ARTIFACT",
        "generated_from_local_parent": git_head(),
        "source_execution_commit": args.source_execution_commit,
        "source_basis": "QUALIFIED",
        "subspace": "QUALIFIED",
        "candidate_bank_exists": False,
        "candidate_count_frozen": CANDIDATE_COUNT,
        "selected_count_frozen": SELECTED_COUNT,
        "shell_targets": SHELL_TARGETS,
        "primary_panel_N": PRIMARY_N,
        "semantic_outcomes": 0,
        "Q3": "NOT_RUN",
        "artifact_hashes": hashes,
    }
    write_json(REVIEW / "PRELOCK.json", payload)
    (REVIEW / "PRELOCK.md").write_text(
        "# Q2 V4 PRELOCK\n\n"
        "Status: `Q2_V4_PRELOCK_READY_FOR_COMMIT`.\n\n"
        "The commit containing this artifact is the unique PRELOCK. No candidate "
        "bank exists before it. The 40-direction seed is derived from that commit "
        "exactly once. Source and subspace gates passed; semantic outcomes remain zero.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "hashes": len(hashes)}))


if __name__ == "__main__":
    main()
