#!/usr/bin/env python3
"""Materialize the outcome-free Q1 budget interaction prelock artifacts.

This command is deliberately an invocation boundary.  It reads only the
explicitly supplied historical Stage-A manifest JSON, extracts latent IDs from
that already-loaded object, and writes the deterministic prospective manifest,
schedule, and provenance records.  It never opens a journal or outcome file
and does not perform model or benchmark access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.reasoning.budget_interaction import (  # noqa: E402
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _schedule_digest(schedule: list[dict[str, Any]]) -> str:
    """Hash the canonical schedule content with the experiment namespace."""

    return stable_digest(NAMESPACE, "SCHEDULE", canonical_json(schedule))


def materialize(
    historical_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build and write the final prospective artifacts from one source file.

    The source bytes are hashed before JSON decoding so the provenance records
    the exact file supplied by the caller.  All destination names are checked
    before creating or replacing a file.  ``overwrite`` is intentionally opt
    in and defaults to ``False``.
    """

    source = Path(historical_manifest_path)
    output = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"historical Stage-A manifest is not a file: {source}")

    source_bytes = source.read_bytes()
    try:
        source_payload = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"historical Stage-A manifest is invalid JSON: {source}") from exc

    destination_paths = tuple(output / name for name in OUTPUT_FILENAMES)
    existing = tuple(path.name for path in destination_paths if path.exists())
    if existing and not overwrite:
        names = ", ".join(existing)
        raise FileExistsError(
            f"refusing to overwrite existing prelock artifact(s): {names}; "
            "pass --overwrite to replace them"
        )

    excluded_ids = extract_historical_latent_ids(source_payload)
    manifest = build_manifest(excluded_ids, n_per_cell=DEFAULT_LATENTS_PER_CELL)
    schedule = build_schedule(manifest)

    manifest_hashes = {row.get("manifest_hash") for row in manifest}
    if len(manifest_hashes) != 1:
        raise ValueError("materialized manifest has inconsistent manifest hashes")
    manifest_hash = next(iter(manifest_hashes))
    if not isinstance(manifest_hash, str):
        raise ValueError("materialized manifest has no manifest hash")

    provenance: dict[str, Any] = {
        "schema_version": "q1-budget-interaction-prelock-provenance-v1",
        "source_sha256": _sha256_bytes(source_bytes),
        "extracted_id_count": len(excluded_ids),
        "manifest_hash": manifest_hash,
        "schedule_digest": _schedule_digest(schedule),
        "namespace": NAMESPACE,
        "N": len(manifest),
        "n_per_cell": DEFAULT_LATENTS_PER_CELL,
        "call_count": len(schedule),
        "calls_per_cap": len(schedule) // len(CAPS),
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(destination_paths[0], manifest)
    _write_json(destination_paths[1], schedule)
    _write_json(destination_paths[2], provenance)
    return {
        "output_dir": str(output),
        "manifest_rows": len(manifest),
        "schedule_rows": len(schedule),
        "provenance": provenance,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize outcome-free Q1 budget interaction prelock artifacts."
    )
    parser.add_argument(
        "historical_manifest",
        type=Path,
        help="explicit historical Stage-A manifest JSON path",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="explicit output directory for the three prelock artifacts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing MANIFEST.json, SCHEDULE.json, or PROVENANCE.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = materialize(args.historical_manifest, args.output_dir, overwrite=args.overwrite)
    # Keep stdout free of latent IDs.  The returned counts are enough to audit
    # that the requested finite artifact set was written.
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "manifest_rows": result["manifest_rows"],
                "schedule_rows": result["schedule_rows"],
                "namespace": result["provenance"]["namespace"],
                "N": result["provenance"]["N"],
                "call_count": result["provenance"]["call_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
