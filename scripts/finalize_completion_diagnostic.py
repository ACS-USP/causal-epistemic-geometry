#!/usr/bin/env python3
"""Finalize a complete diagnostic whose stdout connection was lost.

This command is model-free. It changes only manifest bookkeeping after the
journal and cap recommendation have been independently inspected.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--recommendation", type=Path)
    args = parser.parse_args()
    manifest_path = args.run / "manifest.json"
    recommendation_path = args.recommendation or args.run / "cap_recommendation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    if recommendation.get("proposed_cap") is None:
        raise ValueError("cannot finalize a diagnostic without a proposed cap")
    if manifest.get("identity", {}).get("stage") != "completion_diagnostic":
        raise ValueError("run is not a completion diagnostic")
    if not (args.run / "journal.jsonl").exists():
        raise FileNotFoundError(args.run / "journal.jsonl")
    manifest.update(
        {
            "status": "COMPLETE",
            "classification": "DEVELOPMENT_ONLY_NOT_SCIENTIFIC_OUTCOMES",
            "completed_utc": datetime.now(UTC).isoformat(),
            "rows": sum(
                1
                for line in (args.run / "journal.jsonl").read_text().splitlines()
                if line.strip()
            ),
            "cap_recommendation": recommendation,
            "recovered_from_interrupted": manifest.get("status") == "INTERRUPTED",
        }
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"finalized {args.run} with cap {recommendation['proposed_cap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
