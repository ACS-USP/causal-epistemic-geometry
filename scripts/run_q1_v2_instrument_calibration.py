#!/usr/bin/env python3
"""Run E3-10 baseline-only calibration on the approved remote model.

This is intentionally the only Q1 V2 execution script at this stage.  It does
not construct directions, extract activations, or access fresh scientific
splits.  The HuggingFace backend enforces the RunPod/HF_HOME location guard
before loading weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from epistemic_geometry.backends.huggingface import HuggingFaceBackend
from epistemic_geometry.benchmarks.e3.base import GENERATOR_VERSION, LatentItem
from epistemic_geometry.benchmarks.e3.calibration import run_baseline_calibration
from epistemic_geometry.benchmarks.e3.qualification import select_cells, summarize_cell
from epistemic_geometry.benchmarks.e3.splits import SplitManifest
from epistemic_geometry.config import load_config


def _load_manifests(path: Path) -> list[SplitManifest]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "INSTRUMENT_CALIBRATION":
        raise ValueError("calibration runner accepts only INSTRUMENT_CALIBRATION manifests")
    if payload.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("calibration manifest uses a different generator version")
    if payload.get("model_outcomes"):
        raise ValueError("calibration manifest provenance is invalid: model outcomes already used")
    eligible_cells = set(payload.get("structural_gate", {}).get("eligible_cells", []))
    if not eligible_cells:
        raise ValueError("calibration manifest is missing structural eligibility")
    manifests: list[SplitManifest] = []
    for key, record in payload.get("manifests", {}).items():
        if not isinstance(record, dict):
            raise ValueError("malformed family/cell manifest")
        if key not in eligible_cells:
            raise ValueError(f"structurally ineligible cell was scheduled for calibration: {key}")
        manifests.append(
            SplitManifest(
                split_name=str(record["split_name"]),
                suite_version=str(record["suite_version"]),
                generator_version=str(record["generator_version"]),
                seed=int(record["seed"]),
                items=tuple(LatentItem.from_record(item) for item in record["items"]),
                metadata=dict(record["metadata"]),
            )
        )
    if not manifests:
        raise ValueError("calibration manifest contains no family/cell records")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.experiment.stage != "development" or config.steering.enabled:
        raise ValueError("E3-10 calibration requires development stage and steering.enabled=false")
    if config.backend.candidate_head_mode != "candidate_only":
        raise ValueError("E3-10 calibration requires candidate_head_mode=candidate_only")
    manifests = _load_manifests(args.manifest)
    backend = HuggingFaceBackend(config.backend)
    rows = []
    for manifest in manifests:
        rows.extend(
            run_baseline_calibration(
                backend,
                manifest,
                execution_mode=config.backend.execution_mode,
                layer=config.backend.layer,
            )
        )
    grouped: dict[tuple[str, str], list] = {}
    for row in rows:
        grouped.setdefault((row.family, row.cell), []).append(row)
    summaries = [summarize_cell(group) for group in grouped.values()]
    selected = select_cells(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores_path = args.output / "baseline_score_vectors.jsonl"
    with scores_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_record(), sort_keys=True) + "\n")
    (args.output / "qualification.json").write_text(
        json.dumps({"summaries": summaries, "selected": selected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} baseline score rows to {args.output}")
    print("STEERING_PERFORMED: NO")
    print("FRESH_SCIENTIFIC_SPLITS_ACCESSED: NO")


if __name__ == "__main__":
    main()
