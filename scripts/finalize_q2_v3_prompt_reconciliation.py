#!/usr/bin/env python3
"""Write the self-excluding hash ledger for Q2 V3 provenance artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_v3_provenance_reconciliation"
DESTINATION = REVIEW / "artifact_hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    records = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(REVIEW.iterdir())
        if path.is_file() and path != DESTINATION
    }
    DESTINATION.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"artifacts": len(records), "ledger": str(DESTINATION)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
