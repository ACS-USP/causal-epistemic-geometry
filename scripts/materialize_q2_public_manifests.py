#!/usr/bin/env python3
"""Materialize Q2 manifests from the pinned public CRUXEval dataset on RunPod."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate9 import normalize_dataset_row  # noqa: E402

FILENAMES = {
    "SOURCE_CONSTRUCTION": "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION": "SOURCE_VALIDATION_MANIFEST.json",
    "MANIPULATION_QUALIFICATION": "MANIPULATION_MANIFEST.json",
    "COVARIANCE_POOL": "COVARIANCE_MANIFEST.json",
    "FINITE_SECANT_PROBES": "FINITE_SECANT_PROBE_MANIFEST.json",
    "COMMON_PANEL": "DEVELOPMENT_PANEL_MANIFEST.json",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    review = args.review_dir.resolve()
    allocation = json.loads(
        (review / "REMOTE_PUBLIC_DATASET_ALLOCATION.json").read_text(encoding="utf-8")
    )
    from datasets import load_dataset

    source = list(
        load_dataset(
            allocation["dataset_repo"],
            split=allocation["split"],
            revision=allocation["dataset_revision"],
        )
    )
    normalized = {row["item_id"]: row for row in map(normalize_dataset_row, source)}
    allocated_ids: list[str] = []
    summaries: dict[str, Any] = {}
    for role, ids in allocation["allocations"].items():
        if role not in FILENAMES:
            raise RuntimeError(f"unknown Q2 allocation role: {role}")
        missing = sorted(set(ids) - normalized.keys())
        if missing:
            raise RuntimeError(f"pinned public dataset is missing Q2 IDs: {missing}")
        rows = []
        for item_id in ids:
            row = dict(normalized[item_id])
            row["allocation"] = role
            row["metadata"] = {**row["metadata"], "q2_allocation": role}
            rows.append(row)
        # Match the canonical external-manifest envelope consumed by
        # ``load_external``. The ID-only allocation deliberately has a
        # different shape and must never leak into the execution interface.
        write_json(
            review / FILENAMES[role],
            {
                "schema_version": 1,
                "allocation": role,
                "dataset_repo": allocation["dataset_repo"],
                "dataset_revision": allocation["dataset_revision"],
                "items": rows,
            },
        )
        allocated_ids.extend(ids)
        summaries[role] = {
            "n": len(rows),
            "first_id": ids[0],
            "last_id": ids[-1],
        }
    if len(allocated_ids) != len(set(allocated_ids)):
        raise RuntimeError("materialized Q2 allocations are not disjoint")
    result = {
        "classification": "Q2_PUBLIC_DATASET_MATERIALIZATION_PASS",
        "dataset_repo": allocation["dataset_repo"],
        "dataset_revision": allocation["dataset_revision"],
        "dataset_rows": len(source),
        "allocated_rows": len(allocated_ids),
        "allocations": summaries,
        "model_outputs_existed": False,
    }
    write_json(review / "PUBLIC_DATASET_MATERIALIZATION.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
