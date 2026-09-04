#!/usr/bin/env python3
"""Build release-safe Q3.1 closeout artifacts from sealed private results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_route_a_prompt_representation"
PRECHECK = REVIEW / "Q3_ROUTE_A_PROMPT_REPRESENTATION_PRECHECK.json"
SUMMARY = REVIEW / "Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json"
FORENSICS = REVIEW / "Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_FORENSICS.json"
RELEASE = REVIEW / "Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SAFETY.json"
HASHES = REVIEW / "Q3_ROUTE_A_PROMPT_REPRESENTATION_ARTIFACT_HASHES.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def public_model_summary(model: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"itemwise_routed_correctness", "itemwise_champion_correctness"}
    return {key: value for key, value in model.items() if key not in forbidden}


def public_bank_summary(bank: dict[str, Any]) -> dict[str, Any]:
    return {
        "bank_method": bank["bank_method"],
        "fold_banks": bank["fold_banks"],
        "hyperparameters": bank["hyperparameters"],
        "models": {
            name: public_model_summary(value) for name, value in bank["models"].items()
        },
    }


def build(
    full_result_path: Path,
    capture_result_path: Path,
    matrix_path: Path,
    row_metadata_path: Path,
) -> dict[str, Any]:
    precheck = read_json(PRECHECK)
    full = read_json(full_result_path)
    capture = read_json(capture_result_path)
    expected_status = "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL"
    if precheck.get("status") != "Q3_ROUTE_A_PROMPT_REPRESENTATION_PRECHECK_FROZEN":
        raise RuntimeError("Q3.1 precheck is not frozen")
    if full.get("status") != expected_status:
        raise RuntimeError("unexpected frozen Q3.1 ruling")
    if capture.get("status") != "Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_COMPLETE":
        raise RuntimeError("capture is not complete")
    if capture.get("prompt_only_forward_count") != 332:
        raise RuntimeError("unexpected prompt-only forward count")
    if capture.get("semantic_generation") != 0 or capture.get("candidate_answers") != 0:
        raise RuntimeError("capture crossed the semantic firewall")
    if capture.get("reference_or_correctness_loaded") is not False:
        raise RuntimeError("capture loaded forbidden reference/correctness data")
    if sha256_file(matrix_path) != capture["matrix_sha256"]:
        raise RuntimeError("private representation matrix hash mismatch")
    if sha256_file(row_metadata_path) != capture["row_metadata_sha256"]:
        raise RuntimeError("private row metadata hash mismatch")
    if full["capture"]["matrix_sha256"] != capture["matrix_sha256"]:
        raise RuntimeError("analysis/capture matrix mismatch")
    if full["fresh_holdout_outcomes_inspected"] is not False:
        raise RuntimeError("fresh holdout firewall violated")

    summary = {
        "schema_version": "q3-route-a-prompt-representation-release-summary-v1",
        "status": full["status"],
        "evidence_class": full["evidence_class"],
        "scientific_state": "Q3_NOT_RUN_DEVELOPMENT_ONLY",
        "primary_bank": public_bank_summary(full["primary_bank"]),
        "secondary_banks_cannot_rescue_primary": {
            name: public_bank_summary(bank)
            for name, bank in full["secondary_banks_cannot_rescue_primary"].items()
        },
        "q3_0_deterministic_prompt_structure_control": full[
            "q3_0_deterministic_prompt_structure_control"
        ],
        "gate_results": full["gate_results"],
        "capture": full["capture"],
        "source_counts": full["source_counts"],
        "private_artifact_identity": {
            "full_result_sha256": sha256_file(full_result_path),
            "capture_result_sha256": sha256_file(capture_result_path),
            "representation_matrix_sha256": sha256_file(matrix_path),
            "row_metadata_sha256": sha256_file(row_metadata_path),
        },
        "firewall": {
            "new_semantic_trajectories": 0,
            "fresh_evaluation_outcomes_inspected": False,
            "reference_or_correctness_used_in_capture": False,
            "q3_confirmatory_experiment": "NOT_RUN",
            "q1_q2_classifications_changed": False,
        },
        "control_completeness": {
            "capacity_matched_geometry_blind": "COMPUTED",
            "fixed_coordinate_permutation": "COMPUTED",
            "q3_0_prompt_structure": "COMPUTED_FROM_FROZEN_Q3_0_ARTIFACT",
            "global_champion": "PRIMARY_COMPARATOR",
            "policy_prior_random_router": "NOT_INSTANTIATED_PRECHECK_DID_NOT_FIX_PROBABILITY_RULE",
            "effect_on_terminal_ruling": (
                "NONE_PRIMARY_AND_INCREMENTAL_GATES_DO_NOT_DEPEND_ON_RANDOM_ROUTER"
            ),
        },
    }
    write_json(SUMMARY, summary)

    forensics = {
        "schema_version": "q3-route-a-prompt-representation-capture-forensics-v1",
        "status": "Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_FORENSIC_CLEAN",
        "development_families": capture["development_family_count"],
        "prompt_only_forwards": capture["prompt_only_forward_count"],
        "repeat_subset": capture["repeat_subset_count"],
        "repeat_max_abs_difference": capture["repeat_max_abs_difference"],
        "site_equivalence_max_abs_difference": capture[
            "site_equivalence_max_abs_difference"
        ],
        "matrix_shape": capture["matrix_shape"],
        "matrix_dtype": capture["matrix_dtype"],
        "matrix_sha256": capture["matrix_sha256"],
        "row_metadata_sha256": capture["row_metadata_sha256"],
        "private_prompt_manifest_sha256": capture["private_prompt_manifest_sha256"],
        "model": capture["model"],
        "model_revision": capture["model_revision"],
        "tokenizer_revision": capture["tokenizer_revision"],
        "representation_site": capture["representation_site"],
        "equivalent_site_checked": capture["equivalent_site_checked"],
        "single_forward_hook_mechanics": capture["single_forward_hook_mechanics"],
        "qualified_environment_fingerprint": capture["environment"][
            "qualified_environment_fingerprint"
        ],
        "model_manifest_sha256": capture["environment"]["model_bytes"][
            "manifest_sha256"
        ],
        "elapsed_seconds": capture["elapsed_seconds"],
        "git_head": capture["git_head"],
        "semantic_generation": capture["semantic_generation"],
        "candidate_answers": capture["candidate_answers"],
        "reference_or_correctness_loaded": capture["reference_or_correctness_loaded"],
    }
    write_json(FORENSICS, forensics)

    release = {
        "schema_version": "q3-route-a-prompt-representation-release-safety-v1",
        "status": "PASS",
        "tracked": [
            str(PRECHECK.relative_to(ROOT)),
            str(SUMMARY.relative_to(ROOT)),
            str(FORENSICS.relative_to(ROOT)),
            "aggregate development reports",
            "capture and analysis source",
        ],
        "private_hash_pinned_not_tracked": [
            {
                "artifact": "prompt-only manifest containing benchmark prompts",
                "sha256": capture["private_prompt_manifest_sha256"],
            },
            {
                "artifact": "300x4096 float32 prompt-representation matrix",
                "sha256": capture["matrix_sha256"],
            },
            {
                "artifact": "representation row metadata",
                "sha256": capture["row_metadata_sha256"],
            },
            {
                "artifact": "full itemwise development result",
                "sha256": sha256_file(full_result_path),
            },
        ],
        "excluded": [
            "benchmark prompt text",
            "hidden representation values",
            "itemwise routed/champion correctness arrays",
            "raw model outputs",
            "references",
            "credentials and infrastructure identifiers",
        ],
        "raw_model_text_present": False,
        "credentials_present": False,
        "private_infrastructure_present": False,
    }
    write_json(RELEASE, release)

    files = [
        PRECHECK,
        ROOT / "scripts/run_q3_prompt_representation_capture.py",
        ROOT / "scripts/analyze_q3_prompt_representation.py",
        SUMMARY,
        FORENSICS,
        RELEASE,
    ]
    hashes = {
        "schema_version": "q3-route-a-prompt-representation-artifact-hashes-v1",
        "artifacts": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in files
        },
        "private_hash_pinned": release["private_hash_pinned_not_tracked"],
    }
    write_json(HASHES, hashes)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-result", type=Path, required=True)
    parser.add_argument("--capture-result", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--row-metadata", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.full_result, args.capture_result, args.matrix, args.row_metadata)
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
