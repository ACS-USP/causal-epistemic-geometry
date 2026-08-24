#!/usr/bin/env python3
"""Freeze the V2 dose-calibration phase after source qualification only."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_bank_v2"
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.q2_controller_heldout_v2 import (  # noqa: E402
    DOSE_FRACTIONS,
    DOSE_NAMES,
    EXPERIMENT_ID,
    LAYER,
    calibration_schedule,
    stable_digest,
)


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
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    source_lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    if source_lock["status"] != "FROZEN_PRE_SOURCE_QUALIFICATION":
        raise RuntimeError("the V2 pre-source lock is not frozen")
    source_bank = read_json(REVIEW / "V2_SOURCE_DIRECTION_BANK.json")
    source_provenance = read_json(REVIEW / "V2_SOURCE_PROVENANCE.json")
    qualified_axes = list(source_bank["qualified_axes"])
    directions = source_bank["directions"]
    if source_bank["status"] != "QUALIFIED_FOR_DOSE_CALIBRATION":
        raise RuntimeError("Q2_V2_SOURCE_BANK_TOO_NARROW")
    if len(qualified_axes) < 4:
        raise RuntimeError("Q2_V2_SOURCE_BANK_TOO_NARROW")
    controller_ids = sorted(directions)
    items = read_json(REVIEW / "V2_DOSE_CALIBRATION_MANIFEST.json")["items"]
    item_ids = [str(item["item_id"]) for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise RuntimeError("dose calibration items are not unique")
    schedule = calibration_schedule(item_ids, controller_ids)
    expected_rows = len(item_ids) * (1 + len(controller_ids) * len(DOSE_NAMES))
    if len(schedule) != expected_rows:
        raise RuntimeError("dose calibration schedule has unexpected row count")
    write_json(REVIEW / "V2_CALIBRATION_SCHEDULE.json", schedule)
    lock = {
        "schema_version": "q2-controller-heldout-geometry-v2-dose-lock-v1",
        "status": "FROZEN_PRE_DOSE_CALIBRATION",
        "lifecycle": "PROSPECTIVE_LOCK",
        "experiment_id": EXPERIMENT_ID,
        "source_lock_sha256": sha256(REVIEW / "PROTOCOL_LOCK.json"),
        "source_commit": source_provenance["source_commit"],
        "lock_preparation_commit": git_head(),
        "model": source_lock["model"],
        "layer": LAYER,
        "qualified_source_axes": qualified_axes,
        "meaningful_controller_ids": controller_ids,
        "direction_provenance": directions,
        "dose_grid": {
            "names": list(DOSE_NAMES),
            "fractions_of_direction_reference_scale": list(DOSE_FRACTIONS),
            "actual_delta_norm": "dose_fraction * direction.reference_scale",
        },
        "calibration_manifest": {
            "file": "V2_DOSE_CALIBRATION_MANIFEST.json",
            "sha256": sha256(REVIEW / "V2_DOSE_CALIBRATION_MANIFEST.json"),
            "n": len(item_ids),
        },
        "schedule": {
            "file": "V2_CALIBRATION_SCHEDULE.json",
            "sha256": sha256(REVIEW / "V2_CALIBRATION_SCHEDULE.json"),
            "expected_rows": expected_rows,
            "rollout_blocks": 1,
            "seed_regime": "MATCHED_COUPLING_CALIBRATION",
            "same_item_seed_across_conditions": True,
        },
        "label_free_selection": {
            "allowed": [
                "raw_sequence_movement",
                "semantic_change_without_correctness",
                "commitment_validity",
                "semantic_evaluability",
                "truncation_rate",
                "token_movement",
            ],
            "forbidden": [
                "accuracy",
                "G",
                "C",
                "D",
                "rescue",
                "damage",
                "error_correlation",
                "common_panel_outcomes",
            ],
        },
        "operating_dose_rule": (
            "For each signed direction choose the lowest causal dose; if none is "
            "causal, choose the lowest safe dose and mark the direction weak/inert."
        ),
        "bank_rule_after_calibration": source_lock["bank_level_rule"],
        "correctness_used": False,
        "source_qualification_outcomes_read": True,
        "common_panel_outcomes_read": False,
        "Q1": "IMMUTABLE",
        "Q3": "NOT RUN",
    }
    write_json(REVIEW / "V2_DOSE_CALIBRATION_LOCK.json", lock)
    (REVIEW / "V2_DOSE_CALIBRATION_LOCK.md").write_text(
        "# Q2 V2 dose-calibration lock\n\n"
        "Status: `FROZEN_PRE_DOSE_CALIBRATION`.\n\n"
        f"The source phase qualified {len(qualified_axes)} conceptual families and "
        f"{len(controller_ids)} signed/location directions. Each receives the frozen "
        f"dose grid {', '.join(DOSE_NAMES)} scaled to its own source reference scale. "
        "The operating dose is selected per direction using only label-free movement, "
        "validity/evaluability, truncation, and token diagnostics. Accuracy, G/C/D, "
        "rescue, damage, and common-panel outcomes are forbidden. The next required "
        "transition is the complete matched calibration schedule; no common panel is "
        "authorized until a later final bank lock.\n",
        encoding="utf-8",
    )
    write_json(
        REVIEW / "V2_DOSE_CALIBRATION_PROVENANCE.json",
        {
            "source_bank_sha256": sha256(REVIEW / "V2_SOURCE_DIRECTION_BANK.json"),
            "source_provenance_sha256": sha256(REVIEW / "V2_SOURCE_PROVENANCE.json"),
            "qualified_axis_digest": stable_digest(
                EXPERIMENT_ID, "QUALIFIED_AXES", *qualified_axes
            ),
            "controller_digest": stable_digest(EXPERIMENT_ID, "DIRECTIONS", *controller_ids),
            "correctness_used": False,
            "common_panel_outcomes_read": False,
        },
    )
    print(
        json.dumps(
            {
                "classification": "Q2_V2_DOSE_CALIBRATION_LOCK_PREPARED",
                "qualified_axes": len(qualified_axes),
                "directions": len(controller_ids),
                "expected_rows": expected_rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
