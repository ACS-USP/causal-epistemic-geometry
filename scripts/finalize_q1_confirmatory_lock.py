#!/usr/bin/env python3
"""Finalize the Q1 pre-holdout lock after a passing dress rehearsal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q1_confirmatory_fixed_controllers"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    rehearsal = read_json(REVIEW / "DRESS_REHEARSAL.json")
    power = read_json(REVIEW / "POWER_ANALYSIS.json")
    identity = read_json(REVIEW / "HOLDOUT_IDENTITY_LOCK.json")
    if rehearsal["classification"] != "DRESS_REHEARSAL_PASS":
        raise RuntimeError("cannot finalize after a failed dress rehearsal")
    if power["classification"] != "Q1_CONFIRMATORY_N57_POWER_QUALIFIED":
        raise RuntimeError("cannot finalize without N=57 power qualification")
    if identity["status"] != "SEALED_ASSIGNED_UNACCESSED":
        raise RuntimeError("cannot finalize after holdout content access")

    premortem = read_json(REVIEW / "PREMORTEM.json")
    premortem["classification"] = "PREMORTEM_PASS"
    premortem["dress_rehearsal"] = "DRESS_REHEARSAL_PASS"
    write_json(REVIEW / "PREMORTEM.json", premortem)
    premortem_md = (REVIEW / "PREMORTEM.md").read_text(encoding="utf-8")
    (REVIEW / "PREMORTEM.md").write_text(
        premortem_md.replace(
            "PREMORTEM_PASS_FOR_DRESS_REHEARSAL", "PREMORTEM_PASS"
        ).replace(
            "# Q1 Confirmatory Premortem\n\n",
            "# Q1 Confirmatory Premortem\n\nDress rehearsal: `DRESS_REHEARSAL_PASS`.\n\n",
        ),
        encoding="utf-8",
    )

    protocol = read_json(REVIEW / "PROTOCOL_LOCK.json")
    protocol["status"] = "CONFIRMATORY_LOCKED_PRE_HOLDOUT"
    protocol["lifecycle"] = "PROSPECTIVE_LOCK"
    protocol["dress_rehearsal"] = {
        "classification": rehearsal["classification"],
        "maximum_primary_independent_metric_difference": rehearsal[
            "maximum_primary_independent_metric_difference"
        ],
    }
    protocol["holdout_content_accessed"] = False
    protocol["confirmatory_source_commit_binding"] = {
        "file": "CONFIRMATORY_SOURCE_COMMIT.json",
        "timing": "after this lock commit and before any holdout content access",
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", protocol)
    (REVIEW / "PROTOCOL_LOCK.md").write_text(
        "# Q1 Fixed-Controller Confirmatory Protocol Lock\n\n"
        "Status: `CONFIRMATORY_LOCKED_PRE_HOLDOUT`.\n\n"
        "The exact prospectively assigned 57-ID set remains "
        "`SEALED_ASSIGNED_UNACCESSED`. Offline power qualified for Qwen and Ministral, "
        "and the synthetic dress rehearsal passed every frozen engineering and analysis "
        "check. Each model has seven conditions and two independent rollouts (798 rows; "
        "1,596 total). The primary endpoint is the two-sided 95% item-percentile "
        "bootstrap lower bound for `C_meaningful`, with 50,000 resamples. Both models "
        "must independently pass primary, null-specificity, and safety criteria.\n\n"
        "No holdout prompt, reference, outcome, or model inference was accessed while "
        "creating this lock. A complete two-model cost preflight with a 25% margin is "
        "required before first content access.\n",
        encoding="utf-8",
    )

    spec_path = ROOT / "experiments/specs/q1_confirmatory_fixed_controllers.yaml"
    spec = spec_path.read_text(encoding="utf-8")
    spec = spec.replace(
        "status: PROSPECTIVE_CONFIRMATORY_LOCK_PRE_HOLDOUT",
        "status: FROZEN_CONFIRMATORY_LOCK_PRE_HOLDOUT",
    ).replace(
        "stage: CONFIRMATORY_PRE_HOLDOUT_LOCK_PREPARATION",
        "stage: CONFIRMATORY_PROSPECTIVE_LOCK",
    )
    spec_path.write_text(spec, encoding="utf-8")

    names = [
        "HOLDOUT_PROVENANCE_AUDIT.json",
        "HOLDOUT_IDENTITY_LOCK.json",
        "POWER_ANALYSIS_LOCK.json",
        "POWER_ANALYSIS.json",
        "HYPOTHESIS_LOCK.md",
        "ANALYSIS_LOCK.json",
        "CONTROLLER_IDENTITY_LOCK.json",
        "NULL_BANK_LOCK_QWEN.json",
        "NULL_BANK_LOCK_MINISTRAL.json",
        "RESPONSE_PARSER_LOCK.json",
        "SEED_SCHEDULE_LOCK.json",
        "COST_LOCK.json",
        "PREMORTEM.json",
        "DRESS_REHEARSAL.json",
        "PROTOCOL_LOCK.json",
    ]
    for role in ("QWEN", "MINISTRAL"):
        names.extend(
            f"NULL_DIRECTIONS_{role}/RANDOM_R{index}.npy" for index in range(4)
        )
    hashes = {name: sha256(REVIEW / name) for name in names}
    write_json(
        REVIEW / "artifact_hashes_preholdout.json",
        {
            "status": "FROZEN_BEFORE_HOLDOUT_CONTENT_ACCESS",
            "artifacts": hashes,
            "holdout_content_accessed": False,
        },
    )
    print(json.dumps({"status": protocol["status"], "artifacts": len(hashes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
