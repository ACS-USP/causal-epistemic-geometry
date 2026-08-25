#!/usr/bin/env python3
"""Build an outcome-value-free, item-level CRUXEval provenance ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.cruxeval_provenance import (  # noqa: E402
    classify_item,
    deterministic_panel,
    eligibility_claim,
)

REVIEW = ROOT / "review/q2_m3_qualification_cruxeval_provenance"
DATASET_REPO = "cruxeval-org/cruxeval"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
UNIVERSE = tuple(f"sample_{index}" for index in range(800))
PANEL_NAMESPACE = "Q2-V3-HISTORICAL-C-PROSPECTIVE-CONTROLLER-V1"

EXPERIMENT_ORDER = {
    "external_benchmark_qualification": 10,
    "full_nonthinking_smoke": 20,
    "full_nonthinking_smoke_reanalysis": 21,
    "substrate_race": 30,
    "micro_q1": 40,
    "gate5_source_duration": 50,
    "gate6_layer_source_rfm_atlas": 60,
    "gate6_2_first_stage_repair_mean_bridge": 61,
    "gate6_3_single_mean_semantic_evaluation": 62,
    "gate6_3_semantic_validity_audit": 63,
    "gate7_fresh_l27_replication": 70,
    "gate8_l27_dose_calibration": 80,
    "gate9_selected_d75_evaluation": 90,
    "gate10_cross_domain_charcount": 100,
    "gate11_domain_conditioned_control": 110,
    "gate11_1_artifact_complete_replication": 111,
    "gate12_utility_aligned_pullback": 120,
    "gate12_1_continuous_geometry_engine": 121,
    "gate13_cross_model_ministral3": 130,
    "gate13_1_all_layer_causal_atlas": 131,
    "q1_confirmatory_fixed_controllers": 140,
    "q2_controller_heldout_geometry": 150,
    "q2_controller_bank_v2": 160,
}

LABEL_FREE_EXPERIMENTS = {
    "gate11_domain_conditioned_control",
    "gate11_1_artifact_complete_replication",
    "gate12_utility_aligned_pullback",
    "gate12_1_continuous_geometry_engine",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def walk_item_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if str(value.get("item_id", "")).startswith("sample_"):
            rows.append(value)
        for child in value.values():
            rows.extend(walk_item_rows(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(walk_item_rows(child))
    return rows


def read_structured(path: Path) -> list[Any]:
    if path.suffix == ".json":
        return [json.loads(path.read_text(encoding="utf-8"))]
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def canonical_role(row: dict[str, Any], path: Path) -> str:
    for field in ("allocation", "phase", "role", "split"):
        if row.get(field):
            return str(row[field]).upper()
    return path.stem.upper()


def event_flags(experiment: str, role: str, path: Path, q2_common_ids: set[str]) -> dict[str, Any]:
    upper = f"{role} {path.name}".upper()
    source_axis = any(
        token in upper
        for token in ("SOURCE_CONSTRUCTION", "SOURCE_VALIDATION", "SOURCE_ACTIVATION")
    )
    covariance = any(
        token in upper for token in ("COVARIANCE", "FINITE_SECANT", "GEOMETRY", "FIXED_SEQUENCE")
    )
    allocated = any(
        token in upper for token in ("MANIFEST", "SCHEDULE", "ITEMS", "ALLOCATION", "RESERVE")
    )
    q2_label_free = (
        experiment in {"q2_controller_heldout_geometry", "q2_controller_bank_v2"}
        and "COMMON_PANEL" not in upper
    )
    label_free = experiment in LABEL_FREE_EXPERIMENTS or q2_label_free or source_axis or covariance
    generated = any(
        token in upper
        for token in ("JOURNAL", "RESULT", "EVALUATION", "CALIBRATION", "MANIPULATION", "SCREEN")
    )
    behavioral = generated and not label_free
    correctness = behavioral and any(
        token in upper for token in ("EVALUATION", "RESULT", "HOLDOUT", "COMMON_PANEL", "SCREEN")
    )
    return {
        "activation_only": bool(covariance and not generated),
        "source_axis_construction": source_axis,
        "covariance_geometry_calibration": covariance,
        "reserved_or_allocated": allocated,
        "label_free_generation": bool(generated and label_free),
        "free_generation_inference": bool(generated and not covariance),
        "behavioral_outcome_inspected": behavioral,
        "semantic_correctness_scored": correctness,
        "semantic_outcome_inspected": correctness,
        "used_for_controller_selection": bool(
            generated
            and any(
                token in upper
                for token in ("SOURCE", "CALIBRATION", "MANIPULATION", "SELECTION", "SWEEP")
            )
        ),
        "used_for_hyperparameter_selection": "CALIBRATION" in upper,
        "used_for_metric_selection": experiment == "q2_controller_bank_v2"
        and "COMMON_PANEL" in upper,
        "used_for_threshold_calibration": "CALIBRATION" in upper,
        "q1_confirmatory": experiment == "q1_confirmatory_fixed_controllers",
        "q2_v2_common_panel": experiment == "q2_controller_bank_v2" and "COMMON_PANEL" in upper,
        "q2_geometry_discovery": False,
        "manual_inspection_known": "POSTHOC" in upper or "BLINDED_CORPUS" in upper,
    }


def source_paths() -> list[Path]:
    paths: list[Path] = []
    for path in sorted((ROOT / "review").rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        if REVIEW in path.parents or path.stat().st_size > 100_000_000:
            continue
        paths.append(path)
    return paths


def build_ledger() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q2_common_path = ROOT / "review/q2_controller_bank_v2/V2_COMMON_PANEL_MANIFEST.json"
    q2_common_payload = json.loads(q2_common_path.read_text(encoding="utf-8"))
    q2_common_ids = {str(value) for value in q2_common_payload["item_ids"]}
    events: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    content: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned: list[dict[str, Any]] = []

    for path in source_paths():
        try:
            payloads = read_structured(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error):
            continue
        experiment = path.relative_to(ROOT / "review").parts[0]
        found = 0
        for payload in payloads:
            for row in walk_item_rows(payload):
                item_id = str(row["item_id"])
                if item_id not in UNIVERSE:
                    continue
                found += 1
                if row.get("prompt") and row.get("reference_answer") is not None:
                    content[item_id].append(
                        {
                            "prompt": str(row["prompt"]),
                            "reference_answer": str(row["reference_answer"]),
                            "source_path": str(path.relative_to(ROOT)),
                        }
                    )
                role = canonical_role(row, path)
                key = (experiment, role)
                event = events[item_id].setdefault(
                    key,
                    {
                        "experiment": experiment,
                        "role": role,
                        "order": EXPERIMENT_ORDER.get(experiment, 10_000),
                        "evidence_paths": set(),
                        **event_flags(experiment, role, path, q2_common_ids),
                    },
                )
                event["evidence_paths"].add(str(path.relative_to(ROOT)))
                fresh_flags = event_flags(experiment, role, path, q2_common_ids)
                for field, value in fresh_flags.items():
                    event[field] = bool(event[field] or value)
        if found:
            scanned.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": file_sha256(path),
                    "structured_item_rows": found,
                }
            )

    for item_id in q2_common_ids:
        key = ("q2_controller_bank_v2", "V2_COMMON_PANEL")
        event = events[item_id].setdefault(
            key,
            {
                "experiment": "q2_controller_bank_v2",
                "role": "V2_COMMON_PANEL",
                "order": EXPERIMENT_ORDER["q2_controller_bank_v2"],
                "evidence_paths": {str(q2_common_path.relative_to(ROOT))},
                **event_flags(
                    "q2_controller_bank_v2", "V2_COMMON_PANEL", q2_common_path, q2_common_ids
                ),
            },
        )
        event["q2_geometry_discovery"] = True
        event["semantic_outcome_inspected"] = True
        event["behavioral_outcome_inspected"] = True
        event["semantic_correctness_scored"] = True
        event["used_for_metric_selection"] = True

    ledger: list[dict[str, Any]] = []
    ambiguous_content: dict[str, int] = {}
    for item_id in UNIVERSE:
        item_events = []
        for event in events[item_id].values():
            clean = dict(event)
            clean["evidence_paths"] = sorted(clean["evidence_paths"])
            item_events.append(clean)
        item_events.sort(key=lambda event: (event["order"], event["experiment"], event["role"]))
        provenance_class, reason = classify_item(item_events)
        signatures = Counter(
            (value["prompt"], value["reference_answer"]) for value in content[item_id]
        )
        if len(signatures) > 1:
            ambiguous_content[item_id] = len(signatures)
        canonical = None
        if signatures:
            (prompt, reference), _count = min(
                signatures.items(),
                key=lambda pair: (-pair[1], sha256_bytes(repr(pair[0]).encode())),
            )
            canonical = {
                "prompt_sha256": sha256_bytes(prompt.encode()),
                "reference_sha256": sha256_bytes(reference.encode()),
                "signature_count": len(signatures),
            }
        ledger.append(
            {
                "item_id": item_id,
                "official_index": int(item_id.split("_")[1]),
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
                "source_split": "test/output_prediction",
                "first_experimental_appearance": item_events[0]["experiment"]
                if item_events
                else None,
                "experiments": sorted({event["experiment"] for event in item_events}),
                "roles": sorted({event["role"] for event in item_events}),
                "activation_only_exposure": any(event["activation_only"] for event in item_events),
                "source_axis_construction": any(
                    event["source_axis_construction"] for event in item_events
                ),
                "covariance_geometry_calibration": any(
                    event["covariance_geometry_calibration"] for event in item_events
                ),
                "free_generation_inference": any(
                    event["free_generation_inference"] for event in item_events
                ),
                "controllers_or_conditions_applied": sorted(
                    {
                        event["experiment"]
                        for event in item_events
                        if event["free_generation_inference"] or event["label_free_generation"]
                    }
                ),
                "semantic_correctness_scored": any(
                    event["semantic_correctness_scored"] for event in item_events
                ),
                "outcome_inspected_by_researchers": any(
                    event["behavioral_outcome_inspected"] for event in item_events
                ),
                "used_for_controller_selection": any(
                    event["used_for_controller_selection"] for event in item_events
                ),
                "used_for_hyperparameter_selection": any(
                    event["used_for_hyperparameter_selection"] for event in item_events
                ),
                "used_for_metric_selection": any(
                    event["used_for_metric_selection"] for event in item_events
                ),
                "used_for_threshold_calibration": any(
                    event["used_for_threshold_calibration"] for event in item_events
                ),
                "used_in_q1_confirmatory": any(event["q1_confirmatory"] for event in item_events),
                "used_in_q2_v2_common_panel": item_id in q2_common_ids,
                "used_in_radial_angular_posthoc": item_id in q2_common_ids,
                "known_manual_inspection": any(
                    event["manual_inspection_known"] for event in item_events
                ),
                "provenance_class": provenance_class,
                "classification_reason": reason,
                "eligible_claim": eligibility_claim(provenance_class),
                "selection_rank": sha256_bytes(f"{PANEL_NAMESPACE}\x1f{item_id}".encode()),
                "canonical_content": canonical,
                "events": item_events,
            }
        )

    counts = Counter(row["provenance_class"] for row in ledger)
    if len(ledger) != 800 or len({row["item_id"] for row in ledger}) != 800:
        raise RuntimeError("CRUXEval universe reconstruction is incomplete or duplicated")
    if len(q2_common_ids) != 120 or counts["D"] != 120:
        raise RuntimeError("Q2 V2 common-panel provenance is inconsistent")
    unresolved = [row["item_id"] for row in ledger if row["provenance_class"] == "UNRESOLVED"]
    summary = {
        "schema_version": 1,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "universe_count": len(ledger),
        "class_counts": {key: counts.get(key, 0) for key in ("A", "B", "C", "D", "UNRESOLVED")},
        "q2_v2_common_panel_count": len(q2_common_ids),
        "q2_v2_common_panel_manifest_sha256": file_sha256(q2_common_path),
        "ambiguous_content_signatures": ambiguous_content,
        "unresolved_ids": unresolved,
        "source_file_count": len(scanned),
        "source_files_digest": sha256_bytes(json.dumps(scanned, sort_keys=True).encode()),
        "classification_uses_outcome_values": False,
    }
    return ledger, {"summary": summary, "source_files": scanned}


def write_outputs(ledger: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    ledger_path = REVIEW / "CRUXEVAL_PROVENANCE_LEDGER.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(REVIEW / "CRUXEVAL_PROVENANCE_AUDIT.json", audit)

    counts = audit["summary"]["class_counts"]
    if counts["A"] >= 200:
        panel_class = "A"
        claim = "fresh-item/new-controller prospective validation"
        panel = deterministic_panel(ledger, allowed_classes={"A"}, count=200)
    elif counts["C"] >= 200:
        panel_class = "C"
        claim = "historical-item/prospective-controller same-domain validation"
        panel = deterministic_panel(ledger, allowed_classes={"C"}, count=200)
    else:
        panel_class = None
        claim = "same-domain primary panel unavailable"
        panel = []
    panel_payload = {
        "status": "PROPOSED_NOT_FROZEN_NOT_RUN",
        "selection_namespace": PANEL_NAMESPACE,
        "provenance_class": panel_class,
        "claim": claim,
        "item_count": len(panel),
        "item_ids": [row["item_id"] for row in panel],
        "selection_used_outcome_values": False,
    }
    panel_payload["ordered_ids_sha256"] = sha256_bytes(
        json.dumps(panel_payload["item_ids"], separators=(",", ":")).encode()
    )
    write_json(REVIEW / "Q2_V3_PROPOSED_PRIMARY_PANEL.json", panel_payload)

    report = f"""# CRUXEval item-level provenance audit

The finite official output-prediction universe contains **800** items. “Used” is
not treated as a single epistemic category. Classification is based only on
tracked exposure provenance, never on prior accuracy, difficulty, error
patterns, complementarity, or geometry performance.

## Census

- Class A — pristine: **{counts["A"]}**
- Class B — representation-only / label-free / allocation exposure: **{counts["B"]}**
- Class C — historical behavioral exposure outside Q2 geometry discovery: **{counts["C"]}**
- Class D — directly implicated in Q2 V2/V3 discovery: **{counts["D"]}**
- unresolved: **{counts["UNRESOLVED"]}**

The 120 Q2 V2 common-panel items are Class D because their observed error
geometry and the post-hoc magnitude analysis directly motivated the current
radial/angular redesign. They are excluded from primary Q2 V3 evaluation.

## Proposed Q2 V3 panel

Provenance class: **{panel_class or "NONE"}**. Proposed N: **{len(panel)}**.
Exact claim: **{claim}**.

The panel is ordered by a frozen SHA-256 namespace over item ID. Historical
performance is not read or used. Class C evidence is legitimate prospective
prediction along the controller axis, but it is not fresh-item confirmation.
The proposal remains a draft and no Q2 V3 behavioral outcome has been run.
"""
    (REVIEW / "CRUXEVAL_PROVENANCE_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ledger, audit = build_ledger()
    write_outputs(ledger, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
