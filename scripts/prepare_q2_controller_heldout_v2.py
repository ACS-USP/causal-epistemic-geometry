#!/usr/bin/env python3
"""Prepare the outcome-free pre-source lock for Q2 controller-bank V2."""

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
from epistemic_geometry.experiments.q2_controller_heldout_v2 import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COMMON_PANEL_N,
    DOSE_CALIBRATION_N,
    DOSE_FRACTIONS,
    DOSE_NAMES,
    EXPERIMENT_ID,
    LOCATIONS,
    MAX_TRUNCATION_RATE,
    MIN_CAUSAL_DIRECTIONS,
    MIN_FAMILIES,
    MIN_MEANINGFUL,
    MIN_SOURCE_AXES,
    NULL_COUNT,
    ORTHOGONALITY_TOLERANCE,
    RAW_MOVEMENT_MIN,
    SEMANTIC_MOVEMENT_MIN,
    SIGNS,
    SOURCE_AXES,
    SOURCE_CONSTRUCTION_N,
    SOURCE_VALIDATION_N,
    canonical_json,
    source_axis_payload,
    source_schedule,
    stable_digest,
)

REVIEW = ROOT / "review/q2_controller_bank_v2"
V1_REVIEW = ROOT / "review/q2_controller_heldout_geometry"
V1_COMMON = V1_REVIEW / "DEVELOPMENT_PANEL_MANIFEST.json"
NAMESPACE = "Q2-CONTROLLER-HELDOUT-GEOMETRY-V2"
ROLE_COUNTS = (
    ("V2_SOURCE_CONSTRUCTION", SOURCE_CONSTRUCTION_N),
    ("V2_SOURCE_VALIDATION", SOURCE_VALIDATION_N),
    ("V2_DOSE_CALIBRATION", DOSE_CALIBRATION_N),
    ("V2_COVARIANCE_POOL", 64),
    ("V2_FINITE_SECANT_PROBES", 12),
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


def historical_catalog() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a content catalog from manifests, never from outcome journals."""

    holdout_path = ROOT / "review/q1_confirmatory_fixed_controllers/HOLDOUT_CONTENT_MANIFEST.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdout_ids = {str(item["item_id"]) for item in holdout["items"]}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest_paths: list[str] = []
    for path in sorted((ROOT / "review").rglob("*.json")):
        upper = path.name.upper()
        if "MANIFEST" not in upper and "ITEMS" not in upper:
            continue
        if path == holdout_path or V1_REVIEW in path.parents or REVIEW in path.parents:
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
        signatures = Counter((str(row["prompt"]), str(row["reference_answer"])) for row in rows)
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
        prompt_hash = stable_digest("Q2-V2-TASK-PROMPT", prompt)
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
                    "Q2-V2-CRUXEVAL-ITEM", item_id, prompt_hash, reference, DATASET_REVISION
                ),
                "metadata": {
                    "dataset_repo": DATASET_REPO,
                    "dataset_revision": DATASET_REVISION,
                    "official_id": item_id,
                    "catalog_signature_frequency": frequency,
                    "selection_namespace": NAMESPACE,
                },
            }
        )
    canonical.sort(
        key=lambda row: (stable_digest(NAMESPACE, "POOL", row["item_id"]), row["item_id"])
    )

    v1_ids: set[str] = set()
    for path in V1_REVIEW.glob("*MANIFEST*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        v1_ids.update(str(item["item_id"]) for item in _walk_items(value))
    eligible = [row for row in canonical if row["item_id"] not in v1_ids]
    required = sum(count for _role, count in ROLE_COUNTS)
    if len(eligible) < required:
        raise RuntimeError(f"Q2 V2 requires {required} fresh items; found {len(eligible)}")
    return eligible, {
        "catalog_items_after_holdout": len(canonical),
        "historical_q2_v1_ids_excluded": len(v1_ids),
        "eligible_fresh_items": len(eligible),
        "holdout_excluded_count": len(holdout_ids),
        "holdout_content_manifest_sha256": sha256(holdout_path),
        "source_manifest_count": len(manifest_paths),
        "source_manifests_digest": stable_digest(
            NAMESPACE, "SOURCE_MANIFESTS", canonical_json(manifest_paths)
        ),
        "ambiguous_historical_prompt_ids": ambiguous,
        "selection_used_historical_outcomes": False,
        "historical_journals_read": False,
    }


def allocate(eligible: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    allocations: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for role, count in ROLE_COUNTS:
        rows = []
        for item in eligible[cursor : cursor + count]:
            row = dict(item)
            row["allocation"] = role
            row["metadata"] = {**row["metadata"], "v2_allocation": role}
            rows.append(row)
        allocations[role] = rows
        cursor += count
    ids = [row["item_id"] for rows in allocations.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("V2 fresh allocations are not disjoint")
    return allocations


def common_panel_reuse() -> dict[str, Any]:
    if not V1_COMMON.exists():
        raise RuntimeError("the frozen V1 common-panel manifest is missing")
    original = json.loads(V1_COMMON.read_text(encoding="utf-8"))
    items = original["items"]
    if len(items) != COMMON_PANEL_N:
        raise RuntimeError(f"V1 common panel has {len(items)} items, expected {COMMON_PANEL_N}")
    ids = [str(item["item_id"]) for item in items]
    hashes = [str(item["item_hash"]) for item in items]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise RuntimeError("V1 common-panel IDs or item hashes are duplicated")

    # The V1 pilot's only journals were source/manipulation qualification journals.
    # Check filenames only: no outcome rows are opened or parsed here.
    suspicious = [
        str(path.relative_to(ROOT))
        for path in V1_REVIEW.glob("*common*journal*.jsonl")
        if path.is_file()
    ]
    if suspicious:
        raise RuntimeError(f"possible V1 common-panel journal exists: {suspicious}")
    return {
        "schema_version": 1,
        "reuse_status": "V1_COMMON_PANEL_REUSED_WITHOUT_OUTCOMES",
        "source_manifest": str(V1_COMMON.relative_to(ROOT)),
        "source_manifest_sha256": sha256(V1_COMMON),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "selection_namespace": NAMESPACE,
        "item_count": len(items),
        "item_ids": ids,
        "item_hashes": hashes,
        "items": items,
        "outcome_rows_verified_absent_by_filename_check": True,
        "scientific_outcomes_read": False,
    }


def premortem() -> tuple[dict[str, Any], str]:
    risks = {
        "V1_POSTMORTEM_FIREWALL": (
            "The V1 evidence postmortem reads only source/manipulation qualification "
            "artifacts; correctness, G/C/D, rescue, damage, and the never-run common "
            "panel are excluded."
        ),
        "SOURCE_AXIS_CONFLATION": (
            "Six conceptually distinct instruction contrasts are frozen before source "
            "outputs, with rationale recorded; qualification cannot use correctness labels."
        ),
        "FRESHNESS": (
            "All V1 Q2 manifest IDs are excluded from the new source, calibration, "
            "covariance, and finite-secant pools. The existing V1 common panel is reused "
            "only because its common-panel outcome journal is absent."
        ),
        "NULL_GEOMETRY": (
            "Nulls will be projected against an SVD orthonormal basis of the full "
            "meaningful span, then Gram-Schmidt orthogonalized against earlier nulls; "
            "correlated raw-vector sequential projection is forbidden."
        ),
        "DOSE_LEAKAGE": (
            "Each signed direction receives the same prospective four-bin dose grid but "
            "its operating dose is selected only from label-free movement, "
            "validity/evaluability, truncation, and token diagnostics."
        ),
        "BANK_SELECTION": (
            "The bank-level rule requires at least four source families, fourteen "
            "meaningful controllers, eight causal directions, two selected dose bins, "
            "and full null orthogonality; accuracy/G/C/D cannot qualify or rank controllers."
        ),
        "FAMILY_HELD_OUT": (
            "The family-level split is deterministic and frozen before common-panel "
            "outcomes; dose variants cannot substitute for an unseen source family."
        ),
        "GEOMETRY_SCOPE": (
            "V2 is limited to flat, covariance-whitened, and finite-secant geometry. "
            "JVP, Fisher, pullback, and manifold metrics are forbidden."
        ),
        "COST": (
            "The full design is projected before GPU collection against the US$15 soft "
            "and US$25 hard Q2-V2 envelope; insufficient wallet balance causes a clean "
            "stop rather than a reduced design."
        ),
        "FIREWALL": (
            "Q1 is immutable, Q3 is not run, confirmatory IDs remain excluded, and no "
            "scientific common-panel row is collected before the final bank lock."
        ),
    }
    data = {
        "experiment_id": EXPERIMENT_ID,
        "classification": "PREMORTEM_PASS",
        "risks": risks,
        "unresolved_scientific_ambiguities": [],
        "v1_common_panel_outcomes_read": False,
        "q1_changed": False,
        "q3_authorized": False,
    }
    lines = ["# Q2 V2 calibrated controller-bank premortem", "", "`PREMORTEM_PASS`", ""]
    for name, mitigation in risks.items():
        lines.extend((f"## {name}", "", mitigation, ""))
    return data, "\n".join(lines)


def engineering_fixtures() -> dict[str, Any]:
    prompts = (
        "Continue the neutral token pattern A B A B and end with FINAL: done.",
        "Read this punctuation-only sample: []{}(),.; and end with FINAL: done.",
        "Inspect the numeric string 001122334455 and end with FINAL: done.",
        "Repeat the neutral word twice and end with FINAL: done.",
        "Use the short code-like text x = [1, 2, 3] and end with FINAL: done.",
        "Read the multilingual-neutral pattern alpha beta gamma and end with FINAL: done.",
    )
    return {
        "schema_version": 1,
        "allocation": "Q2_V2_ENGINEERING_ONLY",
        "scientific_items": False,
        "oracles": False,
        "items": [
            {
                "item_id": f"q2_v2_engineering_{index}",
                "benchmark": "SYNTHETIC_ENGINEERING",
                "subtask": "no_oracle_fixture",
                "prompt": prompt,
                "reference_answer": "ENGINEERING_NO_ORACLE",
                "evaluator": "none",
                "source_revision": "q2-v2-engineering-fixtures-v1",
                "prompt_hash": stable_digest(NAMESPACE, "ENGINEERING_PROMPT", prompt),
                "item_hash": stable_digest(NAMESPACE, "ENGINEERING_ITEM", index, prompt),
                "metadata": {"scientific_item": False, "oracle_exists": False},
            }
            for index, prompt in enumerate(prompts)
        ],
    }


def main() -> int:
    REVIEW.mkdir(parents=True, exist_ok=True)
    fixtures = engineering_fixtures()
    write_json(REVIEW / "V2_ENGINEERING_FIXTURES.json", fixtures)
    eligible, provenance = historical_catalog()
    allocations = allocate(eligible)
    filenames = {
        "V2_SOURCE_CONSTRUCTION": "V2_SOURCE_CONSTRUCTION_MANIFEST.json",
        "V2_SOURCE_VALIDATION": "V2_SOURCE_VALIDATION_MANIFEST.json",
        "V2_DOSE_CALIBRATION": "V2_DOSE_CALIBRATION_MANIFEST.json",
        "V2_COVARIANCE_POOL": "V2_COVARIANCE_MANIFEST.json",
        "V2_FINITE_SECANT_PROBES": "V2_FINITE_SECANT_MANIFEST.json",
    }
    for role, rows in allocations.items():
        write_json(
            REVIEW / filenames[role],
            {
                "schema_version": 1,
                "allocation": role,
                "selection_namespace": NAMESPACE,
                "items": rows,
            },
        )

    common = common_panel_reuse()
    write_json(REVIEW / "V2_COMMON_PANEL_MANIFEST.json", common)
    write_json(
        REVIEW / "V2_SOURCE_SCHEDULE.json",
        source_schedule([row["item_id"] for row in allocations["V2_SOURCE_VALIDATION"]]),
    )
    write_json(
        REVIEW / "V2_CALIBRATION_TEMPLATE_SCHEDULE.json",
        {
            "status": "TEMPLATE_BEFORE_SOURCE_QUALIFICATION",
            "note": (
                "The controller list is populated only after source qualification; "
                "no calibration output exists."
            ),
            "items": [row["item_id"] for row in allocations["V2_DOSE_CALIBRATION"]],
            "dose_names": list(DOSE_NAMES),
            "dose_fractions": list(DOSE_FRACTIONS),
            "rollouts": 1,
        },
    )
    write_json(
        REVIEW / "V2_COMMON_PANEL_SCHEDULE_TEMPLATE.json",
        {
            "status": "TEMPLATE_BEFORE_BANK_QUALIFICATION",
            "note": (
                "The controller list is populated only after label-free source/dose qualification."
            ),
            "items": common["item_ids"],
            "rollouts": 2,
        },
    )

    pre_json, pre_md = premortem()
    write_json(REVIEW / "PREMORTEM.json", pre_json)
    (REVIEW / "PREMORTEM.md").write_text(pre_md, encoding="utf-8")

    parser_path = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    lock = {
        "schema_version": "q2-controller-heldout-geometry-v2-pre-source-v1",
        "status": "FROZEN_PRE_SOURCE_QUALIFICATION",
        "lifecycle": "PROSPECTIVE_LOCK",
        "experiment_id": EXPERIMENT_ID,
        "development_only": True,
        "source_commit_at_preparation": git_head(),
        "v1_status": {
            "classification": "Q2_CONTROLLER_BANK_NOT_QUALIFIED",
            "common_panel_ran": False,
            "predictive_outcomes_exist": False,
            "postmortem": "V1_EVIDENCE_POSTMORTEM.md",
        },
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
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
            "source_qualification_correctness_forbidden": True,
            "v1_common_panel_outcomes_read": False,
        },
        "engineering_fixtures": {
            "file": "V2_ENGINEERING_FIXTURES.json",
            "sha256": sha256(REVIEW / "V2_ENGINEERING_FIXTURES.json"),
            "n": len(fixtures["items"]),
            "scientific_items": False,
            "oracles": False,
        },
        "source_axes": {
            "axes": source_axis_payload(),
            "locations": list(LOCATIONS),
            "signs": list(SIGNS),
            "axis_count": len(SOURCE_AXES),
            "conceptual_distinctness_rule": (
                "No cosmetic paraphrase qualifies as a distinct family; rationale and "
                "held-out activation/behavioral separation are required."
            ),
            "minimum_qualified_axes": MIN_SOURCE_AXES,
            "narrow_bank_stop_if_below": 4,
            "construction_uses_correctness": False,
        },
        "allocations": {
            role: {
                "n": len(rows),
                "file": filenames[role],
                "file_sha256": sha256(REVIEW / filenames[role]),
            }
            for role, rows in allocations.items()
        },
        "common_panel_reuse": {
            "manifest": "V2_COMMON_PANEL_MANIFEST.json",
            "manifest_sha256": sha256(REVIEW / "V2_COMMON_PANEL_MANIFEST.json"),
            "source_manifest": str(V1_COMMON.relative_to(ROOT)),
            "source_manifest_sha256": sha256(V1_COMMON),
            "n": COMMON_PANEL_N,
            "outcome_rows": 0,
            "reused_without_outcome_inspection": True,
        },
        "source_qualification": {
            "validity_min": 0.90,
            "evaluability_min": 0.90,
            "cross_disagreement_min": 0.10,
            "excess_disagreement_min": 0.03,
            "activation_standardized_gap_min": 0.20,
            "activation_positive_gap_fraction_min": 0.60,
            "selection_signals": [
                "behavioral separation",
                "activation separation",
                "validity/evaluability",
                "source geometry",
            ],
            "correctness_used": False,
        },
        "dose_calibration": {
            "dose_names": list(DOSE_NAMES),
            "fractions_of_reference_scale": list(DOSE_FRACTIONS),
            "calibration_items": DOSE_CALIBRATION_N,
            "matched_rollout_blocks": 1,
            "label_free_only": True,
            "safe_validity_min": 0.90,
            "safe_evaluability_min": 0.90,
            "max_validity_or_evaluability_drop": 0.05,
            "max_truncation_rate": MAX_TRUNCATION_RATE,
            "causal_raw_sequence_movement_min": RAW_MOVEMENT_MIN,
            "causal_semantic_movement_min": SEMANTIC_MOVEMENT_MIN,
            "selection_rule": (
                "lowest causal dose; otherwise lowest safe dose; no accuracy/G/C/D/ranking"
            ),
        },
        "bank_level_rule": {
            "minimum_meaningful": MIN_MEANINGFUL,
            "minimum_source_families": MIN_FAMILIES,
            "minimum_causal_directions": MIN_CAUSAL_DIRECTIONS,
            "minimum_dose_bins": 2,
            "minimum_directions_per_family": 2,
            "null_count": NULL_COUNT,
            "null_projector": (
                "SVD orthonormal basis of full meaningful span, then prior-null Gram-Schmidt"
            ),
            "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,
            "accuracy_used": False,
            "G_C_D_used": False,
        },
        "geometry": {
            "M0": "normalized Euclidean/cosine",
            "M1": (
                "activation-covariance-whitened with frozen regularization before common outcomes"
            ),
            "M2": "finite behavioral secant; full-vocabulary JS over frozen label-free probes",
            "JVP_Fisher_pullback_forbidden": True,
        },
        "family_split": {
            "scheme": "leave_one_source_family_out",
            "assignment": (
                "deterministic and frozen after source qualification, before common outcomes"
            ),
            "primary_population": "meaningful controllers only",
        },
        "uncertainty": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "item",
        },
        "cost": {
            "soft_envelope_usd": 15.0,
            "hard_ceiling_usd": 25.0,
            "design_may_not_be_reduced_for_wallet": True,
            "stop_if_projected_cost_exceeds_hard_ceiling": True,
        },
        "firewall": {
            "Q1": "IMMUTABLE",
            "Q3": "NOT RUN",
            "confirmatory_holdout": "EXCLUDED; no outcomes read",
            "scientific_common_panel": "BLOCKED UNTIL FINAL BANK LOCK",
            "JVP_Fisher_pullback": "NOT RUN",
        },
        "allocation_provenance": provenance,
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", lock)
    (REVIEW / "PROTOCOL_LOCK.md").write_text(
        "# Q2 V2 pre-source protocol lock\n\n"
        "Status: `FROZEN_PRE_SOURCE_QUALIFICATION`\n\n"
        "This is a DEVELOPMENT bank-rebuild lock. Six conceptual source axes, two "
        "locations, both signs, a four-bin per-direction dose grid, the SVD null "
        "projector, and the source/calibration/geometry allocations are frozen before "
        "new model outputs. Source and dose qualification cannot inspect correctness, "
        "G, C, D, rescue, damage, or common-panel outcomes. The reused 120-item V1 "
        "common panel has no outcome journal and remains blocked until a later final "
        "bank lock. Q1 is immutable, Q3 is not run, and JVP/Fisher/pullback geometry "
        "is forbidden in V2.\n",
        encoding="utf-8",
    )

    hashes = {
        path.name: sha256(path)
        for path in sorted(REVIEW.iterdir())
        if path.is_file() and path.name != "artifact_hashes_pre_source.json"
    }
    write_json(REVIEW / "artifact_hashes_pre_source.json", hashes)
    print(
        json.dumps(
            {
                "classification": "Q2_V2_PRE_SOURCE_LOCK_PREPARED",
                "fresh_qualification_items": len(eligible),
                "allocated_qualification_items": sum(dict(ROLE_COUNTS).values()),
                "reused_common_panel_items": COMMON_PANEL_N,
                "source_rows": SOURCE_VALIDATION_N * 2 * len(SOURCE_AXES) * len(SIGNS),
                "common_panel_outcomes": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
