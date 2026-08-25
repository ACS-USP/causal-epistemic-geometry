#!/usr/bin/env python3
"""Write or verify the self-excluding Q2 V3 Amendment-1 artifact ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_v3_amendment1_freeze"
LEDGER = REVIEW / "artifact_hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload() -> dict[str, object]:
    files = sorted(path for path in REVIEW.iterdir() if path.is_file() and path != LEDGER)
    return {
        "schema_version": "q2-v3-amendment1-artifact-hashes-v1",
        "self_excluding": True,
        "artifacts": {path.name: sha256(path) for path in files},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = payload()
    if args.check:
        actual = json.loads(LEDGER.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("Q2 V3 Amendment-1 artifact ledger mismatch")
    else:
        LEDGER.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"artifacts": len(expected["artifacts"]), "ledger": str(LEDGER)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
