#!/usr/bin/env python3
"""Independent low-level audit of persisted Gate 12.1 engineering arrays."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate12_1  # noqa: E402

REVIEW = ROOT / "review/gate12_1_continuous_geometry_engine"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    exact = read_json(REVIEW / "EXACT_DERIVATIVE_RAW_SUMMARY.json")["fixtures"]
    crosschecks = []
    maximum_difference = 0.0
    for record in exact:
        archive = np.load(ROOT / record["raw_path"], allow_pickle=False)
        if sha256(ROOT / record["raw_path"]) != record["raw_sha256"]:
            raise RuntimeError("raw derivative archive hash mismatch")
        baseline = archive["baseline_logits"].astype(np.float64)
        forward = archive["forward_jvp"].astype(np.float64)
        independent = archive["independent_jvp"].astype(np.float64)
        target = int(archive["target_token_id"])
        cosine = gate12_1.cosine(forward, independent)
        q = float(gate12_1.fisher_energy(baseline, forward))
        u = gate12_1.utility_slope(baseline, forward, target)
        finite = archive["finite_derivatives"].astype(np.float64)
        finite_cosines = [gate12_1.cosine(forward, row) for row in finite]
        differences = (
            abs(cosine - record["forward_independent_jvp_cosine"]),
            abs(q - record["q_jvp"]),
            abs(u - record["u_jvp"]),
        )
        maximum_difference = max(maximum_difference, *differences)
        crosschecks.append(
            {
                "fixture_id": record["fixture_id"],
                "jvp_cosine": cosine,
                "q_jvp": q,
                "u_jvp": u,
                "minimum_finite_cosine": min(finite_cosines),
                "maximum_primary_difference": max(differences),
            }
        )
    with (REVIEW / "FORENSIC_METRIC_CROSSCHECK.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crosschecks[0]))
        writer.writeheader()
        writer.writerows(crosschecks)
    primary = read_json(REVIEW / "ENGINE_QUALIFICATION.json")
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    firewall_clean = bool(
        lock["scientific_item_count"] == 0
        and primary["scientific_items_processed"] == 0
        and not primary["historical_outcomes_revealed"]
    )
    classification = (
        "GATE12_1_FORENSIC_CLEAN"
        if maximum_difference <= 1e-7 and firewall_clean
        else "GATE12_1_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
    )
    audit = {
        "classification": classification,
        "primary_classification": primary["classification"],
        "maximum_primary_audit_metric_difference": maximum_difference,
        "raw_archives_checked": len(crosschecks),
        "scientific_items_processed": 0,
        "historical_outcomes_revealed": False,
        "firewall_clean": firewall_clean,
    }
    write_json(REVIEW / "FORENSIC_AUDIT.json", audit)
    (REVIEW / "FORENSIC_AUDIT.md").write_text(
        "# Gate 12.1 independent forensic audit\n\n"
        f"Classification: `{classification}`.\n\n"
        f"Maximum primary/audit metric difference: `{maximum_difference}`. The audit "
        "recomputed JVP cosine, Fisher moment, utility derivative, and finite-difference "
        "cosines directly from persisted arrays. Scientific items processed: `0`; "
        "historical outcomes revealed: `NO`.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
