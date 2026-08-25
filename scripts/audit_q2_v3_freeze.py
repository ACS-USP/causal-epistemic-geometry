#!/usr/bin/env python3
"""Independent, inference-free audit of the prospective Q2 V3 freeze."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_v3_radial_angular_freeze"
EXPECTED_PANEL_HASH = "969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf"


def read_json(name: str) -> Any:
    return json.loads((REVIEW / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*parts: str) -> str:
    return subprocess.run(
        list(parts), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    clean_at_start = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode == 0 and (
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0
    )
    head = command("git", "rev-parse", "HEAD")
    lock = read_json("PROTOCOL_LOCK.json")
    panel = read_json("PRIMARY_PANEL_MANIFEST.json")
    proof = read_json("PRIMARY_PANEL_PROVENANCE_PROOF.json")
    bank = read_json("CONTROLLER_BANK_SPEC.json")
    shell = read_json("SHELL_CALIBRATION_SPEC.json")
    ident = read_json("IDENTIFIABILITY_GATE.json")
    geometry = read_json("GEOMETRY_DEFINITIONS.json")
    prediction = read_json("PREDICTION_LOCK_SPEC.json")
    stats = read_json("STATISTICAL_ANALYSIS_PLAN.json")
    taxonomy = read_json("CLASSIFICATION_TAXONOMY.json")
    execution = read_json("EXECUTION_COST_PLAN.json")
    schedule = read_json("EVALUATION_SCHEDULE.json")
    source_schedule = read_json("SOURCE_QUALIFICATION_SCHEDULE.json")
    shell_schedule = read_json("SHELL_CALIBRATION_SCHEDULE.json")
    artifacts = read_json("artifact_hashes.json")

    journal_like = [
        path.name
        for path in REVIEW.iterdir()
        if path.is_file()
        and any(token in path.name.upper() for token in ("JOURNAL", "RESULT", "OUTCOME"))
    ]
    logical = {
        (row["item_id"], row["condition"], int(row["rollout_index"]))
        for row in schedule["rows"]
    }
    seeds = {int(row["seed"]) for row in schedule["rows"]}
    referenced_hashes_pass = all(
        (ROOT / path).exists() and sha256(ROOT / path) == record["sha256"]
        for path, record in artifacts.items()
    )
    source_commit = lock["experiment_source_commit"]
    source_is_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, head], cwd=ROOT
        ).returncode
        == 0
    )
    required_thresholds = {
        "rho": stats["relational_gate"].get("aggregate_family_balanced_rho_min"),
        "qap": stats["relational_gate"].get("max_qap_corrected_p_max"),
        "bootstrap": stats["bootstrap"].get("resamples"),
        "families": stats["relational_gate"].get("positive_family_summaries_min"),
        "shells": stats["relational_gate"].get("positive_shells_required"),
        "m2_delta_m0": stats["m2_required"].get("delta_rho_over_M0_min"),
        "m2_delta_m1": stats["m2_required"].get("delta_rho_over_M1_min"),
    }
    checks = {
        "no_q2_v3_semantic_outcomes_exist": journal_like == [],
        "semantic_trajectory_count_zero": lock["semantic_outcomes_in_this_freeze"] == 0,
        "primary_panel_hash_matches": panel["ordered_ids_sha256"] == EXPECTED_PANEL_HASH,
        "primary_panel_n_200": panel["item_count"] == 200,
        "primary_panel_class_c": panel["provenance_class"] == "C",
        "selection_provenance_clean": proof["selection_outcome_independent"] is True
        and proof["forbidden_fields_loaded_by_selector"] == [],
        "m3_excluded": geometry["M3"]["runtime_reenable_forbidden"] is True
        and set(geometry) >= {"M0", "M1", "M2", "M3"}
        and lock["geometries"] == ["M0", "M1", "M2"],
        "all_thresholds_specified": all(value is not None for value in required_thresholds.values())
        and "TBD" not in json.dumps([ident, stats, taxonomy]),
        "all_rng_seeds_specified": len(seeds) == 10_000
        and source_schedule["expected_rows"] == 480
        and shell_schedule["expected_rows"] == 504
        and stats["bootstrap"]["seed"] is not None
        and len(bank["nulls"]["seeds"]) == 2,
        "controller_construction_fully_specified": bank["candidate_count"] == 10
        and bank["meaningful_controller_count"] == 20
        and len(bank["families"]) == 5
        and bank["usable_rule"],
        "shell_calibration_fully_specified": shell["targets"] == {"MEDIUM": 0.25, "STRONG": 0.5}
        and shell["root_finding"]["maximum_iterations"] == 40,
        "geometry_inputs_fully_specified": geometry["M1"]["lambda"] == 0.10
        and "Class-B" in geometry["M2"]["probe_source"]
        and prediction["shape_each"] == [24, 24],
        "geometry_semantic_separation": prediction[
            "primary_semantic_panel_disjoint_from_geometry_inputs"
        ]
        is True,
        "metric_and_analysis_fully_specified": stats["distance"]["shrinkage"] == "none"
        and stats["multiplicity"]["qap_space"].endswith("3840")
        and stats["bootstrap"]["resamples"] == 10_000,
        "stop_rules_specified": len(execution["stop_rules"]) >= 10,
        "schedule_complete_unique": len(schedule["rows"]) == 10_000
        and len(logical) == 10_000
        and len(seeds) == 10_000,
        "execution_requires_no_scientific_judgment": taxonomy["precedence"]
        and execution["recovery_boundary"]
        and prediction["post_outcome_recomputation"],
        "source_commit_is_committed_ancestor": source_is_ancestor,
        "source_state_clean_at_audit_start": clean_at_start,
        "artifact_hashes_match": referenced_hashes_pass,
        "q3_not_run": lock["q3"] == "NOT_RUN",
    }
    passed = all(bool(value) for value in checks.values())
    classification = "Q2_V3_FREEZE_AUDIT_PASS" if passed else "Q2_V3_FREEZE_BLOCKED"
    audit = {
        "schema_version": "q2-v3-independent-freeze-audit-v1",
        "classification": classification,
        "checks": checks,
        "head_at_audit": head,
        "experiment_source_commit": source_commit,
        "journal_or_outcome_files_found": journal_like,
        "required_thresholds": required_thresholds,
        "semantic_trajectories": 0,
        "model_inference_performed": False,
        "q2_v3_status": "Q2_V3_FROZEN_NOT_RUN" if passed else "Q2_V3_FREEZE_BLOCKED",
    }
    (REVIEW / "FREEZE_AUDIT.md").write_text(
        "# Q2 V3 independent freeze audit\n\n"
        f"Classification: `{classification}`\n\n"
        + "\n".join(
            f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in checks.items()
        )
        + "\n\nNo model inference or semantic outcome evaluation was performed.\n",
        encoding="utf-8",
    )
    (REVIEW / "FREEZE_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    all_files = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(REVIEW.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    spec = ROOT / "experiments/specs/q2_v3_radial_angular_geometry.yaml"
    all_files[str(spec.relative_to(ROOT))] = {
        "bytes": spec.stat().st_size,
        "sha256": sha256(spec),
    }
    (REVIEW / "artifact_hashes.json").write_text(
        json.dumps(all_files, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"classification": classification, "checks": checks}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
