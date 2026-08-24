#!/usr/bin/env python3
"""Hash Q2 bank-qualification artifacts and create a verified local bundle."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_heldout_geometry"
BUNDLE = ROOT / "review/q2_controller_heldout_geometry_bundle.tar.gz"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    excluded = {"artifact_hashes.json", "BUNDLE_METADATA.json"}
    records = {
        str(path.relative_to(REVIEW)): {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(REVIEW.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    manifest = REVIEW / "artifact_hashes.json"
    manifest.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with tarfile.open(BUNDLE, "w:gz") as archive:
        archive.add(REVIEW, arcname=REVIEW.name)

    metadata = {
        "artifact_count": len(records),
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "bundle_bytes": BUNDLE.stat().st_size,
        "bundle_sha256": digest(BUNDLE),
        "manifest_sha256": digest(manifest),
    }
    (REVIEW / "BUNDLE_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
