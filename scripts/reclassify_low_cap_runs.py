#!/usr/bin/env python3
"""Relabel preserved 2048-token runs as low-cap diagnostics.

Only the manifest classification is updated.  Journals, results, and raw model
outputs are never rewritten or deleted.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs="+")
    args = parser.parse_args()
    for run in args.run:
        manifest_path = run / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest.get("identity", {})
        generation = identity.get("generation_config", {})
        if generation.get("max_new_tokens") != 2048:
            raise ValueError(f"{run}: refusing to relabel non-2048 run")
        manifest.update(
            {
                "status": "LOW_CAP_DIAGNOSTIC",
                "classification": "LOW_CAP_DIAGNOSTIC_NOT_SCIENTIFIC_FAILURE",
                "scientific_qualification_eligible": False,
                "reclassified_utc": datetime.now(UTC).isoformat(),
                "reclassification_reason": (
                    "2048-token thinking cap is an operational diagnostic; "
                    "truncations are not benchmark evidence"
                ),
            }
        )
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(manifest_path)
        print(f"reclassified {run} as LOW_CAP_DIAGNOSTIC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
