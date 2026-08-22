#!/usr/bin/env python3
"""Hash Gate-11 closeout artifacts and create the verified review bundle."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/gate11_domain_conditioned_control"
BUNDLE = ROOT / "review/gate11_domain_conditioned_control.tar.gz"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    excluded = {"manifest_hashes.json"}
    records = {
        str(path.relative_to(REVIEW)): {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(REVIEW.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    manifest = REVIEW / "manifest_hashes.json"
    manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tarfile.open(BUNDLE, "w:gz") as archive:
        archive.add(REVIEW, arcname=REVIEW.name)
    metadata = {
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "bundle_sha256": digest(BUNDLE),
        "bundle_bytes": BUNDLE.stat().st_size,
        "manifest_sha256": digest(manifest),
        "artifact_count": len(records),
    }
    (REVIEW / "BUNDLE_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
