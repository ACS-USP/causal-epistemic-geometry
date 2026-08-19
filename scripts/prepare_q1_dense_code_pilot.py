#!/usr/bin/env python3
"""Run the model-free safety gate for the dense-code micro-pilot.

This command intentionally stops before item selection when no approved
isolated evaluator is available. It never downloads a benchmark or model and
never starts a Pod. The generated review directory is ignored by repository
policy because it is a run artifact, while this gate remains versioned here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "review/q1_v4_microbench/DENSE_CODE_FAILURE_VECTOR_AUDIT.md"
STATUS = "DENSE_CODE_PILOT_BLOCKED_BY_EVALUATOR"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(output: Path) -> dict[str, object]:
    """Materialize a reproducible pre-GPU blocked-gate bundle."""

    audit_exists = AUDIT_PATH.is_file()
    docker = shutil.which("docker")
    podman = shutil.which("podman")
    firejail = shutil.which("firejail")
    manifest: dict[str, object] = {
        "campaign": "q1-v4-dense-code-micro-pilot",
        "status": STATUS,
        "stage": "PHASE_A_MODEL_FREE_GATE",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "historical_audit": str(AUDIT_PATH.relative_to(ROOT)),
        "historical_audit_present": audit_exists,
        "candidate_selection": None,
        "item_selection": None,
        "model_inference": False,
        "pod_started": False,
        "steering": False,
        "geometry": False,
        "confirmatory_accessed": False,
        "official_evaluator": {
            "livecodebench": "NOT_VERIFIED_PER_TEST_ARTIFACT",
            "evalplus": "PER_TEST_DETAILS_AVAILABLE_BUT_SECURITY_SANDBOX_NOT_APPROVED",
            "evalplus_version_inspected": "0.3.1",
        },
        "available_isolation_commands": {
            "docker": docker,
            "podman": podman,
            "firejail": firejail,
        },
        "reason": (
            "The official EvalPlus detail vector is available, but its local "
            "reliability_guard is explicitly not a security sandbox. No approved "
            "Docker/Podman/Firejail execution boundary is available on this Mac."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "PROTOCOL.md").write_text(
        "# Q1 V4 dense-code micro-pilot\n\n"
        f"Status: **{STATUS}**\n\n"
        "Phase A stopped before GPU. The historical LiveCodeBench audit did not "
        "verify per-test vectors. EvalPlus v0.3.1 exposes test details, but its "
        "own reliability guard is not a security sandbox. The host has no Docker, "
        "Podman, or Firejail executable. No benchmark items were selected and no "
        "model operation was attempted.\n",
        encoding="utf-8",
    )
    (output / "DENSE_CODE_PILOT_SELECTION.md").write_text(
        "# Dense-code pilot selection\n\n"
        f"**{STATUS}**\n\n"
        "No benchmark was selected. LiveCodeBench remains unverified for official "
        "per-test vectors in the repository. EvalPlus HumanEval+ was audited as a "
        "prospective fallback, but safe execution was not approved on this host. "
        "This decision was made before any model outcome and before any item was "
        "selected.\n",
        encoding="utf-8",
    )
    (output / "DENSE_CODE_EVALUATOR_AUDIT.md").write_text(
        "# Dense-code evaluator audit\n\n"
        "## LiveCodeBench\n\n"
        "The checked-in normalized material contains a reference answer but no "
        "verified official per-test execution artifact. The historical audit is "
        "preserved and remains the authority for this candidate.\n\n"
        "## EvalPlus HumanEval+ v0.3.1\n\n"
        "The official evaluator exposes `(status, details)` where `details` is a "
        "boolean result for each input when test-details mode is enabled. Its "
        "documentation and source explicitly state that `reliability_guard` is "
        "not a security sandbox. The current Mac has no Docker, Podman, or Firejail "
        "available, so the required no-network/no-secrets/no-arbitrary-filesystem "
        "boundary is not validated.\n\n"
        "Result: evaluator gate blocked before GPU.\n",
        encoding="utf-8",
    )
    _write_json(output / "MANIFEST.json", manifest)
    (output / "GENERATION_RESULTS.csv").write_text(
        "item_id,rollout_seed,status,executable,tests_passed,tests_total\n",
        encoding="utf-8",
    )
    (output / "TEST_VECTOR_RESULTS.jsonl").write_text("", encoding="utf-8")
    (output / "journal.jsonl").write_text(
        json.dumps(
            {
                "event": "PILOT_BLOCKED_BEFORE_GPU",
                "status": STATUS,
                "timestamp_utc": manifest["timestamp_utc"],
                "model_inference": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "PILOT_REPORT.md").write_text(
        "# Dense-code micro-pilot report\n\n"
        f"**{STATUS}**\n\n"
        "The five-problem baseline-only pilot was not launched. There are no "
        "generation results, no test vectors, and no scientific conclusion.\n",
        encoding="utf-8",
    )
    (output / "COST_REPORT.md").write_text(
        "# Dense-code pilot cost\n\n"
        "GPU runtime: 0 seconds. GPU cost: US$0.00. The RunPod Pod was not started.\n",
        encoding="utf-8",
    )
    report = (
        "# Final report — Q1 V4 dense-code micro-pilot\n\n"
        f"## Decision\n\n**{STATUS}**\n\n"
        "The evaluator gate failed before item selection. No code generated by a "
        "model was executed. Historical artifacts are untouched.\n\n"
        "## Next action\n\n"
        "Provide an approved isolated evaluator, preferably EvalPlus HumanEval+ "
        "inside the official Docker execution path, then rerun Phase A from a "
        "fresh manifest. Do not start steering or select items before that gate.\n"
    )
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    payload = json.dumps(manifest, sort_keys=True).encode()
    (output / "manifest_hash.txt").write_text(_sha256_bytes(payload) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "review/q1_dense_code_pilot",
    )
    args = parser.parse_args()
    manifest = prepare(args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
