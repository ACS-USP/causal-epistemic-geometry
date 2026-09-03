#!/usr/bin/env python3
"""Generate the hash-validated Q2 V4.1 paper figures and figure tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.publication.q2.loaders import validate_frozen_sources  # noqa: E402
from epistemic_geometry.publication.q2.pipeline import generate_visual_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.validate_only:
        result = {"validated_sources": validate_frozen_sources()}
    else:
        generated = generate_visual_evidence(ROOT)
        result = {
            "tables": {
                name: str(path.relative_to(ROOT)) for name, path in generated["tables"].items()
            },
            "figures": {
                name: {suffix: str(path.relative_to(ROOT)) for suffix, path in outputs.items()}
                for name, outputs in generated["figures"].items()
            },
            "figure_data_manifest": str(generated["figure_data_manifest"].relative_to(ROOT)),
            "source_manifest": str(generated["source_manifest"].relative_to(ROOT)),
        }
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Q2 V4.1 publication source validation: PASS")
        if not args.validate_only:
            print(f"Generated {len(result['figures'])} figures and {len(result['tables'])} tables")


if __name__ == "__main__":
    main()
