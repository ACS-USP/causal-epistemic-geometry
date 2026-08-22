#!/usr/bin/env python3
"""Gate 10 runner using the audited sustained-current-token lifecycle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_gate7_fresh_l27_replication as core  # noqa: E402

from epistemic_geometry.benchmarks.v4.character_semantic_v3 import (  # noqa: E402
    evaluate_character_count_answer_v3,
)
from epistemic_geometry.experiments import gate10  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402

REVIEW = ROOT / "review/gate10_cross_domain_charcount"


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def configure() -> None:
    core.REVIEW = REVIEW
    core.CONDITIONS = gate10.CONDITIONS
    core.ETA = gate10.ETA
    core.LAYER = gate10.LAYER
    core.MAX_NEW_TOKENS = gate10.MAX_NEW_TOKENS
    core.MEANINGFUL = gate10.MEANINGFUL
    core.MODEL = gate10.MODEL
    core.MODEL_REVISION = gate10.MODEL_REVISION
    core.RANDOM_NAMES = gate10.RANDOM_NAMES
    core.REFERENCE_SCALE = gate10.REFERENCE_SCALE
    core.TEXTUAL = gate10.TEXTUAL
    core.SYSTEM_CAREFUL = gate10.SYSTEM_CAREFUL
    core.PARSER_VERSION = gate10.PARSER_VERSION
    core.evaluate_external_answer_v3 = evaluate_character_count_answer_v3


def load_lock(review: Path, source_commit: str) -> dict[str, Any]:
    lock = json.loads((review / "PROTOCOL_LOCK.json").read_text())
    binding = json.loads((review / "EXPERIMENT_SOURCE_COMMIT.json").read_text())
    checks = (
        lock["status"] == "FROZEN_PRE_OUTCOME",
        lock["lifecycle"] == "PROSPECTIVE_LOCK",
        git_commit() == source_commit,
        binding["experiment_source_commit"] == source_commit,
        binding["protocol_lock_sha256"] == gate10.file_sha256(review / "PROTOCOL_LOCK.json"),
        lock["instrument"]["evaluator"]["version"] == gate10.PARSER_VERSION,
        lock["instrument"]["evaluator"]["module_sha256"]
        == gate10.file_sha256(
            ROOT / "src/epistemic_geometry/benchmarks/v4/character_semantic_v3.py"
        ),
        lock["model"]["id"] == gate10.MODEL,
        lock["model"]["revision"] == gate10.MODEL_REVISION,
    )
    if not all(checks):
        raise RuntimeError("Gate 10 lock/source/parser provenance mismatch")
    return lock


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("engineering", "collect"), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--review-dir", type=Path, default=REVIEW)
    p.add_argument("--experiment-source-commit", required=True)
    args = p.parse_args()
    require_remote_hf_execution(f"Gate 10 {args.mode}")
    configure()
    review = args.review_dir.resolve()
    lock = load_lock(review, args.experiment_source_commit)
    vectors = core.load_vectors(review, lock)
    deltas = core.deltas(vectors)
    hashes = {name: gate10.vector_sha256(vector) for name, vector in vectors.items()}
    backend = core.build_backend(args.model_path)
    if args.mode == "engineering":
        result = core.engineering_gate(backend, review, lock, deltas, hashes)
        manifest = json.loads((review / "EVALUATION_MANIFEST.json").read_text())
        result["generator_integrity"] = all(
            x["text"].count(x["target_character"]) == x["answer"] for x in manifest["items"]
        )
        result["parser_integrity"] = json.loads((review / "PARSER_VALIDATION.json").read_text())[
            "pass"
        ]
        result["classification"] = (
            "GATE10_ENGINEERING_PASS"
            if result["pass"] and result["generator_integrity"] and result["parser_integrity"]
            else "GATE10_ENGINE_FAILURE"
        )
        core.write_json(review / "ENGINEERING_CHECKS.json", result)
        print(json.dumps({"classification": result["classification"]}, indent=2))
        return 0
    engineering = json.loads((review / "ENGINEERING_CHECKS.json").read_text())
    if engineering["classification"] != "GATE10_ENGINEERING_PASS":
        raise RuntimeError("Gate 10 engineering gate not passed")
    core.collect(backend, review, lock, deltas, hashes, args.experiment_source_commit)
    print(
        json.dumps({"collection": "complete", "rows": lock["schedule"]["logical_rows"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
