#!/usr/bin/env python3
"""Generate baseline-calibration latent manifests without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from epistemic_geometry.benchmarks.e3.splits import (
    CALIBRATION_SPLIT,
    FAMILY_CELLS,
    generate_calibration_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--n-items", type=int, default=200)
    args = parser.parse_args()
    manifests = {}
    for family, cells in FAMILY_CELLS.items():
        for cell in cells:
            manifest = generate_calibration_manifest(
                family, cell, seed=args.seed, n_items=args.n_items
            )
            manifests[f"{family}/{cell}"] = manifest.to_record()
    payload = {
        "suite": "E3-10",
        "split": CALIBRATION_SPLIT,
        "model_outcomes": False,
        "manifests": manifests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(manifests)} family/cell manifests)")


if __name__ == "__main__":
    main()
