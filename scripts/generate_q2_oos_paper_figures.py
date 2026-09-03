#!/usr/bin/env python3
"""Generate deterministic Q2 OOS V2 publication figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.publication.q2_oos.loaders import validate_sources  # noqa: E402
from epistemic_geometry.publication.q2_oos.pipeline import generate_visual_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.validate_only:
        result: dict[str, object] = {"validated_sources": validate_sources()}
    else:
        generated = generate_visual_evidence(ROOT)
        result = {
            "figures": {
                figure_id: {suffix: str(path.relative_to(ROOT)) for suffix, path in outputs.items()}
                for figure_id, outputs in generated["figures"].items()
            },
            "figure_data_manifest": str(generated["figure_data_manifest"].relative_to(ROOT)),
            "source_manifest": str(generated["source_manifest"].relative_to(ROOT)),
        }
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Q2 OOS visual evidence: PASS")


if __name__ == "__main__":
    main()
