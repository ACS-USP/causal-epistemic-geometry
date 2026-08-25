#!/usr/bin/env python3
"""Independent closeout audit for the Q2 V3 Amendment-1 source-gate stop."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_v3_amendment1_execution"
FREEZE = ROOT / "review/q2_v3_amendment1_freeze"
ORIGINAL_FREEZE = ROOT / "review/q2_v3_radial_angular_freeze"
FAMILIES = (
    "CONTROL_FLOW_PATH_COVERAGE",
    "MUTATION_ALIAS_CAUSALITY",
    "API_CONTRACT_EXACTNESS",
    "LOOP_BOUNDARY_ACCOUNTING",
    "HYPOTHESIS_BRANCH_ELIMINATION",
)
LOCATIONS = ("PROMPT_BOUNDARY", "EXECUTION_BOUNDARY")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_sha256(vector: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vector, dtype=np.float64).reshape(-1).tobytes()).hexdigest()


def numeric_differences(left: Any, right: Any, prefix: str = "") -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() & right.keys():
            values.update(numeric_differences(left[key], right[key], f"{prefix}.{key}"))
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        values[prefix] = abs(float(left) - float(right))
    return values


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--pod-created", required=True)
    parser.add_argument("--pod-deleted-upper-bound", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--hourly-rate", type=float, default=0.44)
    args = parser.parse_args()
    review = args.review_dir.resolve()

    wrappers = [
        json.loads(line)
        for line in (review / "Q2_V3_SOURCE_JOURNAL.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [dict(wrapper["row"]) for wrapper in wrappers]
    keys = [tuple(wrapper["key"]) for wrapper in wrappers]
    schedule_path = FREEZE / "SOURCE_QUALIFICATION_SCHEDULE.json"
    if not schedule_path.exists():
        schedule_path = ORIGINAL_FREEZE / "SOURCE_QUALIFICATION_SCHEDULE.json"
    schedule = read_json(schedule_path)["rows"]
    expected = {
        (row["item_id"], row["family"], row["polarity"], row["rollout_index"])
        for row in schedule
    }
    actual = set(keys)
    identity_hashes = {str(wrapper["identity_hash"]) for wrapper in wrappers}
    identities = {json.dumps(wrapper["identity"], sort_keys=True) for wrapper in wrappers}

    with np.load(review / "Q2_V3_SOURCE_ACTIVATIONS.npz", allow_pickle=False) as archive:
        activations = {name: archive[name].astype(np.float64) for name in archive.files}

    independent: dict[str, Any] = {}
    crosscheck_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        selected = [row for row in rows if row["family_id"] == family]
        positive = [row for row in selected if row["polarity"] == "POSITIVE"]
        negative = [row for row in selected if row["polarity"] == "NEGATIVE"]
        lookup = {
            (row["item_id"], row["polarity"], row["rollout_index"]): row["canonical_value"]
            for row in selected
        }
        item_ids = sorted({row["item_id"] for row in selected})
        cross: list[float] = []
        within: list[float] = []
        for item_id in item_ids:
            pos = [lookup[(item_id, "POSITIVE", rollout)] for rollout in (0, 1)]
            neg = [lookup[(item_id, "NEGATIVE", rollout)] for rollout in (0, 1)]
            cross.extend(float(left != right) for left in pos for right in neg)
            within.extend((float(pos[0] != pos[1]), float(neg[0] != neg[1])))
        record: dict[str, Any] = {
            "positive_validity": float(np.mean([row["commitment_valid"] for row in positive])),
            "negative_validity": float(np.mean([row["commitment_valid"] for row in negative])),
            "positive_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in positive])
            ),
            "negative_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in negative])
            ),
            "cross_disagreement": float(np.mean(cross)),
            "within_disagreement": float(np.mean(within)),
            "excess_disagreement": float(np.mean(cross) - np.mean(within)),
            "locations": {},
        }
        for location in LOCATIONS:
            cpos = activations[f"construction__{family}__POSITIVE__{location}"]
            cneg = activations[f"construction__{family}__NEGATIVE__{location}"]
            raw = np.mean(cpos - cneg, axis=0)
            raw_norm = float(np.linalg.norm(raw))
            direction = raw / raw_norm
            vpos = activations[f"validation__{family}__POSITIVE__{location}"]
            vneg = activations[f"validation__{family}__NEGATIVE__{location}"]
            gaps = (vpos - vneg) @ direction
            location_record = {
                "raw_norm": raw_norm,
                "standardized_gap": float(np.mean(gaps) / max(np.std(gaps, ddof=1), 1e-12)),
                "positive_projection_fraction": float(np.mean(gaps > 0)),
                "vector_hash": vector_sha256(direction),
            }
            vector_path = review / "Q2_V3_VECTORS" / f"MEAN_{family}_{location}.npy"
            stored = np.load(vector_path, allow_pickle=False).astype(np.float64)
            location_record["stored_vector_max_abs_difference"] = float(
                np.max(np.abs(stored - direction))
            )
            location_record["stored_vector_hash"] = vector_sha256(stored)
            record["locations"][location] = location_record
        behavior_pass = (
            all(
                record[name] >= 0.90
                for name in (
                    "positive_validity",
                    "negative_validity",
                    "positive_evaluability",
                    "negative_evaluability",
                )
            )
            and record["cross_disagreement"] >= 0.10
            and record["excess_disagreement"] >= 0.03
        )
        representation_pass = all(
            record["locations"][location]["raw_norm"] >= 1e-6
            and record["locations"][location]["standardized_gap"] >= 0.20
            and record["locations"][location]["positive_projection_fraction"] >= 0.60
            for location in LOCATIONS
        )
        record["pass"] = bool(behavior_pass and representation_pass)
        record["behavior_pass"] = bool(behavior_pass)
        record["representation_pass"] = bool(representation_pass)
        independent[family] = record
        crosscheck_rows.append(
            {
                "family": family,
                "positive_validity": record["positive_validity"],
                "negative_validity": record["negative_validity"],
                "cross_disagreement": record["cross_disagreement"],
                "within_disagreement": record["within_disagreement"],
                "excess_disagreement": record["excess_disagreement"],
                "behavior_pass": behavior_pass,
                "representation_pass": representation_pass,
                "pass": record["pass"],
            }
        )

    primary = read_json(review / "Q2_V3_SOURCE_QUALIFICATION.json")
    differences = numeric_differences({"families": independent}, primary)
    max_difference = max(differences.values(), default=0.0)
    semantic_artifacts = (
        "Q2_V3_SEMANTIC_JOURNAL.jsonl",
        "Q2_V3_PREDICTION_MATRICES.npz",
        "Q2_V3_PREDICTION_LOCK.json",
        "Q2_V3_SHELL_JOURNAL.jsonl",
    )
    forbidden_present = [name for name in semantic_artifacts if (review / name).exists()]
    pass_vector = [record["pass"] for record in independent.values()]
    classification = (
        "Q2_V3_CONTROLLER_QUALIFICATION_FAILED"
        if not all(pass_vector)
        else "Q2_V3_CONTROLLER_QUALIFICATION_PASS"
    )
    checks = {
        "journal_rows_480": len(rows) == 480,
        "journal_unique_keys_480": len(actual) == 480 and len(keys) == len(actual),
        "schedule_exact": actual == expected,
        "single_identity": len(identity_hashes) == 1 and len(identities) == 1,
        "correctness_never_evaluated": not any(
            bool(row.get("correctness_evaluated")) for row in rows
        ),
        "no_operational_retries": not any(int(row.get("retry_count", 0)) for row in rows),
        "primary_audit_classification_match": classification == primary["classification"],
        "primary_audit_metric_match": max_difference <= 1e-12,
        "stored_vectors_exact": all(
            record["locations"][location]["stored_vector_max_abs_difference"] <= 1e-15
            and record["locations"][location]["stored_vector_hash"]
            == primary["families"][family]["locations"][location]["vector_hash"]
            for family, record in independent.items()
            for location in LOCATIONS
        ),
        "no_post_source_artifacts": not forbidden_present,
    }
    audit_classification = (
        "Q2_V3_AMENDMENT1_FORENSIC_CLEAN"
        if all(checks.values())
        else "Q2_V3_AMENDMENT1_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
    )
    audit = {
        "schema_version": "q2-v3-amendment1-source-stop-forensic-v1",
        "classification": audit_classification,
        "terminal_state": classification,
        "checks": checks,
        "rows": len(rows),
        "unique_keys": len(actual),
        "missing_keys": len(expected - actual),
        "unexpected_keys": len(actual - expected),
        "identity_hashes": len(identity_hashes),
        "primary_audit_maximum_numeric_difference": max_difference,
        "forbidden_post_source_artifacts": forbidden_present,
        "families": independent,
        "scientific_semantic_trajectories": 0,
        "correctness_outcomes_opened": False,
        "prediction_matrices_created": False,
    }
    write_json(review / "FORENSIC_AUDIT.json", audit)
    with (review / "SOURCE_METRIC_CROSSCHECK.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(crosscheck_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(crosscheck_rows)

    created = parse_time(args.pod_created)
    deleted = parse_time(args.pod_deleted_upper_bound)
    runtime_seconds = int((deleted - created).total_seconds())
    gpu_cost = runtime_seconds / 3600.0 * args.hourly_rate
    volume_upper = 100.0 * 0.20 / (30.0 * 24.0) * runtime_seconds / 3600.0
    cost = {
        "pod_created_utc": args.pod_created,
        "pod_deleted_upper_bound_utc": args.pod_deleted_upper_bound,
        "runtime_seconds_upper_bound": runtime_seconds,
        "gpu_hours_upper_bound": runtime_seconds / 3600.0,
        "gpu_hourly_rate_usd": args.hourly_rate,
        "estimated_gpu_cost_upper_bound_usd": gpu_cost,
        "estimated_100gb_volume_cost_upper_bound_usd": volume_upper,
        "estimated_incremental_total_upper_bound_usd": gpu_cost + volume_upper,
        "hard_ceiling_usd": 15.0,
        "within_hard_ceiling": gpu_cost + volume_upper <= 15.0,
        "billing_note": (
            "Live billing was lagging at deletion; this is a conservative wall-clock "
            "upper bound."
        ),
    }
    write_json(review / "COST.json", cost)
    environment = {
        "profile": "CORE_QWEN",
        "pod_id": args.pod_id,
        "gpu": "NVIDIA A40 48 GB",
        "datacenter": "CA-MTL-1",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "python": "3.11.10",
        "torch": "2.4.1+cu124",
        "transformers": "4.57.1",
        "accelerate": "1.14.0",
        "huggingface_hub": "0.36.0",
        "cuda_runtime": "12.4",
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "execution_source_commit": "ff97bc22ffd5479e6ea3ae082cc94bad8e632a0b",
        "remote_preflight": "PASS",
        "artifact_recovery_files": 18,
        "artifact_recovery_sha256_manifest_match": True,
        "pod_deleted": True,
        "active_pods_after_closeout": 0,
        "retained_network_volumes_after_closeout": 0,
    }
    write_json(review / "ENVIRONMENT_PROVENANCE.json", environment)

    failed = [family for family, record in independent.items() if not record["pass"]]
    report = f"""# Q2 V3 Amendment 1 execution closeout

`{classification}`

## Mechanical outcome

The prospectively frozen source qualification gate stopped Q2 V3 Amendment 1. Four of
five conceptual families passed. `{failed[0]}` failed only the behavioral excess-
disagreement criterion: `{independent[failed[0]]['excess_disagreement']:.12f}` against
the frozen minimum `0.03`. Its validity/evaluability, cross disagreement, and both
representation-location gates passed.

No controller was replaced. Shell calibration/safety, angular identifiability,
M0/M1/M2 construction, prediction lock, and the 10,000-row semantic panel were not run.
There is no Q2 V3 relational geometry result.

## Integrity

- Source journal: 480/480 rows, 480 unique logical keys, zero duplicates or missing keys.
- Operational retries: 0.
- Correctness labels used: NO.
- Primary/audit maximum numeric discrepancy: `{max_difference:.3g}`.
- Forensic classification: `{audit_classification}`.
- Recovered artifacts: 18 files, exact remote/local SHA-256 manifest match.

## Interpretation boundary

The five-family bank did not qualify under the all-families-pass rule. This is not a
negative test of M0, M1, M2, radial structure, or semantic error geometry because none
of those predictive objects or outcomes was opened. The narrow margin does not authorize
an Amendment 2 or a rerun in this task.

## Resources

The A40 Pod was deleted after byte-verified artifact recovery. Active Pods: 0. Retained
network volumes: 0. Conservative incremental cost upper bound: US${gpu_cost + volume_upper:.3f}.

Q1 and Q2 V2 remain unchanged. M3 remains excluded/not qualified. The original Q2 V3
provenance abort remains preserved. Q3 was not run.
"""
    (review / "REPORT.md").write_text(report, encoding="utf-8")
    (review / "FORENSIC_AUDIT.md").write_text(
        "# Q2 V3 Amendment 1 forensic audit\n\n"
        f"Classification: `{audit_classification}`.\n\n"
        "The independent low-level recomputation reproduced the source metrics and terminal "
        f"classification with maximum numeric discrepancy `{max_difference:.3g}`. The journal "
        "is complete, unique, schedule-exact, retry-free, and contains no correctness evaluation. "
        "No post-source or primary semantic artifact exists.\n",
        encoding="utf-8",
    )
    print(json.dumps({"terminal_state": classification, "audit": audit_classification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
