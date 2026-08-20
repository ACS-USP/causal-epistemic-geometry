#!/usr/bin/env python3
"""Create the Gate-6 machine-readable protocol lock after fresh allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review" / "gate6_layer_source_rfm_atlas"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Gate-6 lock generation requires PyYAML") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    review = args.review_dir.resolve()
    spec = load_yaml(ROOT / "experiments" / "specs" / "gate6_layer_source_rfm_atlas.yaml")
    summary = json.loads((review / "ALLOCATION_SUMMARY.json").read_text(encoding="utf-8"))
    exclusion = json.loads(
        (review / "HISTORICAL_EXCLUSION_DIGEST.json").read_text(encoding="utf-8")
    )
    manifest_names = {
        "SOURCE_VALIDATION": "SOURCE_VALIDATION.json",
        "CONTROLLER_MANIPULATION": "CONTROLLER_MANIPULATION.json",
        "CONTROLLER_EVALUATION": "CONTROLLER_EVALUATION.json",
    }
    manifests = {}
    for allocation, filename in manifest_names.items():
        path = review / filename
        if not path.exists():
            raise RuntimeError(f"missing frozen Gate-6 manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests[allocation] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "manifest_hash": payload.get("manifest_hash"),
            "item_ids": [str(row["item_id"]) for row in payload["items"]],
        }
    all_ids = [item_id for values in summary["groups"].values() for item_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Gate-6 fresh manifests are not disjoint")
    source_items = []
    source_records: list[dict[str, Any]] = []
    for path in (
        ROOT / "review" / "micro_q1" / "CONSTRUCTION_MANIFEST.json",
        review.parent / "gate5_source_duration" / "SOURCE_CHECK.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_records.extend(payload["items"])
        source_items.extend(str(row["item_id"]) for row in payload["items"])
    if len(source_items) != 104 or len(source_items) != len(set(source_items)):
        raise RuntimeError("Gate-6 source-training pool is not exactly 104 unique items")
    source_manifest = {
        "allocation": "SOURCE_TRAIN",
        "items": source_records,
        "n_items": len(source_records),
        "dataset_repo": spec["instrument"]["dataset_repo"],
        "dataset_revision": spec["instrument"]["dataset_revision"],
        "source": "historical non-evaluation source manifests; outcome-independent",
    }
    source_manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(source_manifest["items"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    source_manifest_path = review / "SOURCE_TRAIN_MANIFEST.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    lock = {
        "schema_version": 1,
        "status": "FROZEN_PRE_OUTCOME",
        "experiment": spec["experiment_id"],
        "source_commit": source_commit,
        "model": spec["model"],
        "policy": spec["policy"],
        "instrument": spec["instrument"],
        "upstream": spec["upstream"],
        "layers": spec["layers"],
        "source_training": {
            "count": len(source_items),
            "item_ids": source_items,
            "manifest_paths": [
                "review/micro_q1/CONSTRUCTION_MANIFEST.json",
                "review/gate5_source_duration/SOURCE_CHECK.json",
            ],
            "manifest_path": str(source_manifest_path.relative_to(ROOT)),
            "manifest_sha256": digest(source_manifest_path),
            "locations": spec["source_training"]["locations"],
            "labels": "careful_1_direct_0",
            "outcome_labels_used": False,
        },
        "fresh_splits": {
            "allocation_summary_sha256": digest(review / "ALLOCATION_SUMMARY.json"),
            "historical_exclusion": exclusion,
            "groups": manifests,
            "all_ids": all_ids,
            "all_ids_digest": summary["all_ids_digest"],
        },
        "constructors": spec["constructors"],
        "source_gate": spec["source_gate"],
        "teacher_forced_first_stage": spec["teacher_forced_first_stage"],
        "standardized_budget": spec["standardized_budget"],
        "local_control_gain": spec["local_control_gain"],
        "manipulation": spec["manipulation"],
        "evaluation": spec["evaluation"],
        "estimands": spec["estimands"],
        "firewall": spec["firewall"],
        "cost_gate": spec["cost_gate"],
    }
    review.mkdir(parents=True, exist_ok=True)
    (review / "PROTOCOL_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (review / "PROTOCOL_LOCK.md").write_text(
        "# Gate 6 protocol lock\n\n"
        "This lock was generated before any Gate-6 model outcome. It freezes the\n"
        "model, upstream RFM provenance, source/evaluation IDs, layer set,\n"
        "teacher-forced gates, standardized budgets, conditions, estimands,\n"
        "cost gate, and holdout firewall. See `PROTOCOL_LOCK.json` for the\n"
        "complete machine-readable record.\n\n"
        f"Source commit: `{source_commit}`\n\n"
        "Gate 5 artifacts and classification remain immutable.\n",
        encoding="utf-8",
    )
    (review / "manifest_hashes.json").write_text(
        json.dumps(
            {
                **{name: data["sha256"] for name, data in manifests.items()},
                "SOURCE_TRAIN": digest(source_manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    spec["status"] = "FROZEN_PRE_OUTCOME"
    spec["source_commit"] = source_commit
    spec["fresh_splits"] = lock["fresh_splits"]
    (ROOT / "experiments" / "specs" / "gate6_layer_source_rfm_atlas.yaml").write_text(
        yaml_dump(spec), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_commit": source_commit,
                "fresh_ids": len(all_ids),
                "lock": str(review / "PROTOCOL_LOCK.json"),
            },
            indent=2,
        )
    )
    return 0


def yaml_dump(value: Any) -> str:
    import yaml

    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    raise SystemExit(main())
