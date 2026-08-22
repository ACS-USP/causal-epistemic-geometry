#!/usr/bin/env python3
"""Hash the compact Gate-12 closeout artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/gate12_utility_aligned_pullback"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    entries = {}
    for path in sorted(REVIEW.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            entries[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (REVIEW / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "classification": "GATE12_JVP_ENGINE_FAILURE",
                "raw_scientific_shards": 0,
                "artifacts": entries,
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
