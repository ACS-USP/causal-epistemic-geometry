#!/usr/bin/env python3
"""Hash every persisted Gate 12.1 engineering artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/gate12_1_continuous_geometry_engine"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    records = {}
    for path in sorted(REVIEW.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json":
            relative = str(path.relative_to(REVIEW))
            records[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (REVIEW / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "classification": "GATE12_1_DERIVATIVE_ENGINE_NOT_QUALIFIED",
                "scientific_items_processed": 0,
                "historical_outcomes_revealed": False,
                "artifacts": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
