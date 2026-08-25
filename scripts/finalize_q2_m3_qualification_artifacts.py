#!/usr/bin/env python3
"""Write a stable SHA-256 manifest for the Q2 M3/provenance closeout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_m3_qualification_cruxeval_provenance"
MANIFEST = REVIEW / "artifact_hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    hashes = {
        str(path.relative_to(REVIEW)): sha256(path)
        for path in sorted(REVIEW.rglob("*"))
        if path.is_file() and path != MANIFEST
    }
    MANIFEST.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifacts": len(hashes), "manifest": str(MANIFEST)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
