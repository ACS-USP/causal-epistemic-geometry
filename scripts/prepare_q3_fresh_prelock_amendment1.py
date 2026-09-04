#!/usr/bin/env python3
"""Freeze a pre-generation behavioral-signature resolution amendment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_fresh_instrument_qualification"
ORIGINAL_COMMIT = "013305064d75cd4638826b1c4c18a3407457d5b4"
AMENDMENT_IMPLEMENTATION_COMMIT = "439967aaa946cc98df860cd36d76b26a0c403aa6"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", ORIGINAL_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise SystemExit("original generator prelock is not an ancestor")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        != AMENDMENT_IMPLEMENTATION_COMMIT
    ):
        raise SystemExit("amendment artifact must be prepared from the exact implementation commit")
    original = REVIEW / "Q3_FRESH_INSTRUMENT_GENERATOR_PRELOCK.json"
    module = ROOT / "src/epistemic_geometry/benchmarks/q3_fresh/instrument.py"
    test = ROOT / "tests/test_q3_fresh_instrument.py"
    artifact = {
        "schema_version": "q3-fresh-instrument-prelock-amendment-1",
        "status": "FROZEN_BEFORE_SCIENTIFIC_GENERATION",
        "classification": "IMPLEMENTATION_ONLY_BEHAVIORAL_SIGNATURE_RESOLUTION",
        "original_prelock_commit": ORIGINAL_COMMIT,
        "amendment_commit": AMENDMENT_IMPLEMENTATION_COMMIT,
        "reason_for_amendment": "make the frozen behavioral deduplication rule executable",
        "scientific_generation_before_amendment": 0,
        "scientific_outcomes_before_amendment": 0,
        "original_prelock": {
            "path": str(original.relative_to(ROOT)),
            "sha256": sha256(original),
            "commit": ORIGINAL_COMMIT,
            "preserved": True,
        },
        "pre_amendment_state": {
            "scientific_families_generated": 0,
            "experimental_seeds_derived": 0,
            "model_outcomes": 0,
            "correctness_inspected": False,
        },
        "defect": (
            "Four final-output probes provide at most 16 distinct signatures for boolean "
            "outputs, so signature equality alone cannot serve as a mathematically viable "
            "global family-identity criterion."
        ),
        "repair": (
            "Use 32 fixed generator-only probe inputs. Treat signature equality as a "
            "near-duplicate diagnostic that triggers rejection only together with the "
            "frozen structural near-duplicate rule; exact canonical skeleton identity "
            "continues to reject unconditionally. No allocated input, model outcome, or "
            "reference value influenced this amendment."
        ),
        "unchanged": [
            "generator productions and probabilities",
            "candidate stream and namespace law",
            "family canonicalization",
            "allocation",
            "near-duplicate threshold",
            "qualification conditions and gates",
            "candidate router, bank, and champion",
            "model and generation contract",
        ],
        "effective_code_identity": {
            "generator_module": {
                "path": str(module.relative_to(ROOT)),
                "sha256": sha256(module),
            },
            "regression_test": {
                "path": str(test.relative_to(ROOT)),
                "sha256": sha256(test),
            },
            "behavioral_probe_count": 32,
        },
        "amended_acceptance_rule": {
            "exact_canonical_skeleton_collision": "REJECT",
            "structural_near_duplicate": "REJECT",
            "behavioral_signature_collision_alone": "FLAG_NOT_REJECT",
            "behavioral_plus_structural_near_duplicate": "REJECT",
            "near_duplicate_rate_gate": 0.01,
        },
        "effective_seed_derivation": (
            "first 63 bits of SHA256('Q3-FRESH-V1' || this_amendment_sha256 || "
            "containing_commit || namespace)"
        ),
    }
    destination = REVIEW / "Q3_FRESH_INSTRUMENT_PRELOCK_AMENDMENT_1.json"
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "sha256": sha256(destination)}))


if __name__ == "__main__":
    main()
