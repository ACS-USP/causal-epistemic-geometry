#!/usr/bin/env python3
"""Hash the complete local Q2 V2 review directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_bank_v2"
MANIFEST = REVIEW / "artifact_hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    records = {
        str(path.relative_to(REVIEW)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(REVIEW.rglob("*"))
        if path.is_file() and path != MANIFEST
    }
    MANIFEST.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    total_bytes = sum(record["bytes"] for record in records.values())
    print(
        json.dumps(
            {
                "artifact_count": len(records),
                "total_bytes": total_bytes,
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "manifest_sha256": sha256(MANIFEST),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
