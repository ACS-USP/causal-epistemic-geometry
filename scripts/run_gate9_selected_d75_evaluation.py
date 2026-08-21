#!/usr/bin/env python3
"""Gate 9 runner using the already-audited Gate 7 sustained hook lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_gate7_fresh_l27_replication as core  # noqa: E402

from epistemic_geometry.experiments import gate9  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402

REVIEW = ROOT / "review/gate9_selected_d75_evaluation"


def _configure_core() -> None:
    core.REVIEW = REVIEW
    core.CONDITIONS = gate9.CONDITIONS
    core.ETA = gate9.ETA
    core.LAYER = gate9.LAYER
    core.MAX_NEW_TOKENS = gate9.MAX_NEW_TOKENS
    core.MEANINGFUL = gate9.MEANINGFUL
    core.MODEL = gate9.MODEL
    core.MODEL_REVISION = gate9.MODEL_REVISION
    core.RANDOM_NAMES = gate9.RANDOM_NAMES
    core.REFERENCE_SCALE = gate9.REFERENCE_SCALE
    core.TEXTUAL = gate9.TEXTUAL


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("engineering", "collect"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--experiment-source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution(f"Gate 9 {args.mode}")
    _configure_core()
    review = args.review_dir.resolve()
    lock = core.load_lock(review, args.experiment_source_commit)
    vectors = core.load_vectors(review, lock)
    controller_deltas = core.deltas(vectors)
    controller_hashes = {name: gate9.vector_sha256(vector) for name, vector in vectors.items()}
    backend = core.build_backend(args.model_path)
    if args.mode == "engineering":
        result = core.engineering_gate(backend, review, lock, controller_deltas, controller_hashes)
        result["classification"] = (
            "GATE9_ENGINEERING_PASS" if result["pass"] else "GATE9_ENGINE_FAILURE"
        )
        core.write_json(review / "ENGINEERING_CHECKS.json", result)
        print(json.dumps({"classification": result["classification"]}, indent=2))
        return 0
    engineering = json.loads((review / "ENGINEERING_CHECKS.json").read_text(encoding="utf-8"))
    if engineering.get("classification") != "GATE9_ENGINEERING_PASS":
        raise RuntimeError("Gate 9 collection requires a passed engineering gate")
    core.collect(
        backend,
        review,
        lock,
        controller_deltas,
        controller_hashes,
        args.experiment_source_commit,
    )
    print(json.dumps({"collection": "complete", "rows": lock["schedule"]["logical_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
