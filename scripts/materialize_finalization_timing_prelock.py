#!/usr/bin/env python3
"""Materialize outcome-free artifacts for Q1 D75 finalization timing.

The command reads only caller-supplied manifest files and their frozen SHA-256
values. It extracts latent IDs, never journals or outcomes, then emits a new
manifest, paired schedule, and provenance record exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.reasoning.finalization_timing import (  # noqa: E402
    CAPS,
    DEFAULT_LATENTS_PER_CELL,
    NAMESPACE,
    build_manifest,
    build_schedule,
    extract_historical_latent_ids,
)
from epistemic_geometry.reproducibility import canonical_json, stable_digest  # noqa: E402

MANIFEST_FILENAME = "MANIFEST.json"
SCHEDULE_FILENAME = "SCHEDULE.json"
PROVENANCE_FILENAME = "PROVENANCE.json"
OUTPUT_FILENAMES = (MANIFEST_FILENAME, SCHEDULE_FILENAME, PROVENANCE_FILENAME)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"source manifest is not valid JSON: {path}") from exc


def _flat_manifest_ids(value: Any) -> frozenset[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    records = tuple(value)
    if not records or not all(isinstance(row, Mapping) for row in records):
        return None
    ids = [row.get("latent_id") for row in records]
    if not all(isinstance(identifier, str) and identifier for identifier in ids):
        return None
    if len(set(ids)) != len(ids):
        raise ValueError("flat source manifest has duplicate latent IDs")
    return frozenset(ids)


def extract_source_ids(value: Any) -> frozenset[str]:
    """Extract only latent identities from either supported manifest shape."""

    flat = _flat_manifest_ids(value)
    if flat is not None:
        return flat
    return extract_historical_latent_ids(value)


def _schedule_digest(schedule: list[dict[str, Any]]) -> str:
    return stable_digest(NAMESPACE, "SCHEDULE", canonical_json(schedule))


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prelock artifact: {path.name}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize(
    sources: Mapping[str | Path, str], output_dir: str | Path
) -> dict[str, Any]:
    """Build one 192-latent schedule after validating every exclusion source."""

    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must map one or more manifest paths to SHA-256 values")
    output = Path(output_dir)
    destination = tuple(output / name for name in OUTPUT_FILENAMES)
    existing = [path.name for path in destination if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {', '.join(existing)}")

    source_rows: list[dict[str, Any]] = []
    excluded: set[str] = set()
    for raw_path, expected_digest in sorted(sources.items(), key=lambda item: str(item[0])):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"exclusion manifest is not a file: {path}")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError(f"source digest must be SHA-256 hex: {path}")
        payload = path.read_bytes()
        observed_digest = _sha256(payload)
        if observed_digest != expected_digest:
            raise ValueError(f"source SHA-256 mismatch: {path}")
        ids = extract_source_ids(_read_json(path))
        overlap = excluded & ids
        if overlap:
            raise ValueError("exclusion sources contain overlapping latent IDs")
        excluded.update(ids)
        source_rows.append({"path": str(path), "sha256": observed_digest, "id_count": len(ids)})

    manifest = build_manifest(excluded, n_per_cell=DEFAULT_LATENTS_PER_CELL)
    schedule = build_schedule(manifest)
    manifest_hashes = {row.get("manifest_hash") for row in manifest}
    if len(manifest_hashes) != 1 or not isinstance(next(iter(manifest_hashes)), str):
        raise ValueError("generated manifest has no unique manifest hash")
    provenance = {
        "schema_version": "q1-finalization-timing-prelock-provenance-v1",
        "namespace": NAMESPACE,
        "sources": source_rows,
        "excluded_id_count": len(excluded),
        "manifest_hash": next(iter(manifest_hashes)),
        "schedule_digest": _schedule_digest(schedule),
        "N": len(manifest),
        "n_per_cell": DEFAULT_LATENTS_PER_CELL,
        "call_count": len(schedule),
        "calls_per_cap": len(schedule) // len(CAPS),
    }
    output.mkdir(parents=True, exist_ok=True)
    for path, value in zip(destination, (manifest, schedule, provenance), strict=True):
        _write_new_json(path, value)
    return {"manifest": manifest, "schedule": schedule, "provenance": provenance}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--source-sha256", action="append", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if len(args.source) != len(args.source_sha256):
        raise ValueError("each --source requires one --source-sha256")
    result = materialize(dict(zip(args.source, args.source_sha256, strict=True)), args.output_dir)
    print(
        json.dumps(
            {
                "N": result["provenance"]["N"],
                "call_count": result["provenance"]["call_count"],
                "namespace": result["provenance"]["namespace"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
