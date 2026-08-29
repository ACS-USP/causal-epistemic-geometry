#!/usr/bin/env python3
"""Independent model-free audit of the Q1 hierarchical-unit amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_second_task as base  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_hierarchical as h  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_hierarchical_power as hp  # noqa: E402
from epistemic_geometry.reproducibility import stable_digest  # noqa: E402

PARENT = ROOT / "review/q1_second_task_spark2_design"
REVIEW = PARENT / "amendment1_hierarchical_unit"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pages(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(
        directory.glob("rows_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    rows = [entry for path in paths for entry in read_json(path)["rows"]]
    rows.sort(key=lambda entry: int(entry["row_idx"]))
    return [dict(entry["row"]) for entry in rows]


def audit_schedule(
    path: Path,
    *,
    expected_rows: int,
    expected_families: int,
    expected_conditions: int,
    expected_rollouts: int,
) -> dict[str, Any]:
    rows = read_json(path)
    keys = [
        (
            row["family_id"],
            row["condition"],
            int(row["rollout_index"]),
        )
        for row in rows
    ]
    seeds = [int(row["seed"]) for row in rows]
    result = {
        "rows": len(rows),
        "families": len({row["family_id"] for row in rows}),
        "conditions": len({row["condition"] for row in rows}),
        "rollouts": len({int(row["rollout_index"]) for row in rows}),
        "duplicate_logical_keys": len(keys) - len(set(keys)),
        "duplicate_seeds": len(seeds) - len(set(seeds)),
    }
    expected = {
        "rows": expected_rows,
        "families": expected_families,
        "conditions": expected_conditions,
        "rollouts": expected_rollouts,
        "duplicate_logical_keys": 0,
        "duplicate_seeds": 0,
    }
    if result != expected:
        raise RuntimeError(f"schedule mismatch for {path.name}: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-pages", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.parquet) != base.LIVECODEBENCH_PARQUET_SHA256:
        raise RuntimeError("pinned parquet mismatch")
    lock = read_json(REVIEW / "AMENDMENT_LOCK.json")
    parent_lock = read_json(PARENT / "PRINCIPAL_REVIEW_LOCK.json")
    if parent_lock["scientific_benchmark_outcomes"] != 0:
        raise RuntimeError("parent scientific firewall changed")
    if parent_lock["correctness_inspected"]:
        raise RuntimeError("parent correctness firewall changed")

    raw_rows = load_pages(args.dataset_pages)
    items = [base.normalize_livecodebench_row(row, index) for index, row in enumerate(raw_rows)]
    groups = h.group_families(items)
    if Counter(len(rows) for rows in groups.values()) != Counter({2: 105, 3: 76, 4: 1}):
        raise RuntimeError("family structure mismatch")

    stage_a = read_json(REVIEW / "STAGE_A_FAMILY_MANIFEST.json")
    stage_b = read_json(REVIEW / "STAGE_B_FAMILY_MANIFEST.json")
    reserve = read_json(REVIEW / "RESERVE_FAMILY_MANIFEST.json")
    selected_records = [*stage_a["ordered_families"], *stage_b["ordered_families"]]
    selected_family_ids = [record["family_id"] for record in selected_records]
    reserve_family_ids = [record["family_id"] for record in reserve["families"]]
    if len(selected_family_ids) != len(set(selected_family_ids)):
        raise RuntimeError("selected family duplication")
    if set(selected_family_ids) & set(reserve_family_ids):
        raise RuntimeError("selected/reserve family overlap")
    if set(selected_family_ids) | set(reserve_family_ids) != set(groups):
        raise RuntimeError("family partition is incomplete")

    for record in selected_records:
        family_id = record["family_id"]
        selected_id = record["selected_item"]["item_id"]
        expected = min(
            groups[family_id],
            key=lambda item: stable_digest(
                h.EXPERIMENT_ID,
                "REPRESENTATIVE_ROW",
                family_id,
                item.item_id,
            ),
        )
        if selected_id != expected.item_id:
            raise RuntimeError("representative-row rule mismatch")

    siblings = read_json(REVIEW / "EXCLUDED_SIBLING_ROWS_MANIFEST.json")
    selected_rows = len(selected_records)
    reserve_rows = int(reserve["n_raw_rows"])
    if selected_rows + int(siblings["n_rows"]) + reserve_rows != 442:
        raise RuntimeError("raw-row accounting mismatch")

    schedule_a = audit_schedule(
        REVIEW / "STAGE_A_SCHEDULE.json",
        expected_rows=128,
        expected_families=32,
        expected_conditions=2,
        expected_rollouts=2,
    )
    schedule_b = audit_schedule(
        REVIEW / "STAGE_B_SCHEDULE.json",
        expected_rows=5720,
        expected_families=130,
        expected_conditions=11,
        expected_rollouts=4,
    )
    all_seeds = [
        row["seed"]
        for path in (REVIEW / "STAGE_A_SCHEDULE.json", REVIEW / "STAGE_B_SCHEDULE.json")
        for row in read_json(path)
    ]
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("cross-stage seed collision")

    for name, expected_hash in lock["artifact_hashes"].items():
        if sha256(REVIEW / name) != expected_hash:
            raise RuntimeError(f"amendment artifact hash mismatch: {name}")
    for name, expected_hash in lock["inherited_lock_hashes"].items():
        if sha256(PARENT / name) != expected_hash:
            raise RuntimeError(f"inherited lock hash mismatch: {name}")

    supersession = read_json(REVIEW / "PREOUTCOME_SUPERSESSION.json")
    for name, record in supersession["old_artifacts"].items():
        if record["status"] != "SUPERSEDED_PRE_OUTCOME_NEVER_EXECUTED":
            raise RuntimeError(f"old artifact is not explicitly superseded: {name}")
        if sha256(PARENT / name) != record["sha256"]:
            raise RuntimeError(f"old artifact changed: {name}")

    power_first = hp.simulate_one_row_per_family(
        130,
        transfer_fraction=1.0,
        replicates=10_000,
        seed=987654,
    )
    power_second = hp.simulate_one_row_per_family(
        130,
        transfer_fraction=1.0,
        replicates=10_000,
        seed=987654,
    )
    if power_first != power_second:
        raise RuntimeError("power simulation is not deterministic")

    forbidden_outputs = [
        path.name
        for path in REVIEW.iterdir()
        if path.name.lower().startswith(("journal", "outcome", "result"))
    ]
    if forbidden_outputs:
        raise RuntimeError(f"unexpected scientific output artifact: {forbidden_outputs}")

    audit = {
        "classification": "Q1_SECOND_TASK_HIERARCHICAL_DESIGN_FORENSIC_CLEAN",
        "family_structure": {
            "rows": 442,
            "families": 182,
            "size_counts": {"2": 105, "3": 76, "4": 1},
        },
        "partition": {
            "stage_a_families": 32,
            "stage_b_families": 130,
            "reserve_families": 20,
            "selected_rows": selected_rows,
            "excluded_sibling_rows": siblings["n_rows"],
            "reserve_raw_rows": reserve_rows,
            "raw_rows_accounted": 442,
        },
        "stage_a_schedule": schedule_a,
        "stage_b_schedule": schedule_b,
        "cross_stage_unique_seeds": len(all_seeds),
        "representative_rule_recomputed_independently": True,
        "amendment_hashes_verified": len(lock["artifact_hashes"]),
        "inherited_lock_hashes_verified": len(lock["inherited_lock_hashes"]),
        "old_artifacts_immutable_and_superseded": True,
        "power_determinism_crosscheck": True,
        "scientific_inference": 0,
        "correctness_inspected": False,
        "q2_outputs_inspected": False,
        "spark1_used": False,
        "spark2_scientific_inference": False,
    }
    output = REVIEW / "INDEPENDENT_DESIGN_AUDIT.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code_paths = (
        ROOT / "experiments/specs/q1_second_task_spark2_hierarchical_amendment.yaml",
        ROOT / "scripts/prepare_q1_second_task_hierarchical_amendment.py",
        ROOT / "scripts/audit_q1_second_task_hierarchical_amendment.py",
        ROOT / "src/epistemic_geometry/experiments/q1_second_task_hierarchical.py",
        ROOT / "src/epistemic_geometry/experiments/q1_second_task_hierarchical_power.py",
        ROOT / "tests/test_q1_second_task_hierarchical.py",
        ROOT / "tests/test_q1_second_task_hierarchical_artifacts.py",
        ROOT / "tests/test_q1_second_task_hierarchical_power.py",
    )
    review_paths = tuple(
        sorted(
            path
            for path in REVIEW.iterdir()
            if path.is_file() and path.name != "artifact_hashes.json"
        )
    )
    artifact_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in (*code_paths, *review_paths)
    }
    (REVIEW / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
