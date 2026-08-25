#!/usr/bin/env python3
"""CPU-only, all-or-nothing prompt preflight for Q2 V3 Amendment 1."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.q2_v3 import ordered_id_hash  # noqa: E402
from epistemic_geometry.experiments.q2_v3_prompt_provenance import (  # noqa: E402
    AMENDMENT1_PROVENANCE_SCHEMA,
    GATE7_CANONICAL_TEMPLATE_VERSION,
    canonical_q2_v3_task_prompt,
    raw_utf8_sha256,
    source_record_sha256,
)

DEFAULT_REVIEW = ROOT / "review/q2_v3_amendment1_freeze"
DEFAULT_SOURCE = ROOT / "review/q2_v3_provenance_reconciliation/OFFICIAL_SOURCE_RECORDS.jsonl"
EXPECTED_PRIMARY_HASH = "969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf"
MANIFESTS = (
    "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION_MANIFEST.json",
    "SHELL_CALIBRATION_MANIFEST.json",
    "M1_COVARIANCE_MANIFEST.json",
    "M2_PROBE_MANIFEST.json",
    "PRIMARY_PANEL_MANIFEST.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_digest(record: dict[str, Any]) -> str:
    payload = {
        "provenance_schema_version": record["provenance_schema_version"],
        "template_version": record["template_version"],
        "purpose": record["purpose"],
        "item_id": record["item_id"],
        "exact_model_visible_utf8_base64": record["model_visible_prompt"]["utf8_base64"],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_preflight(review: Path, source_path: Path) -> dict[str, Any]:
    lock = read_json(review / "PROTOCOL_LOCK.json")
    failures: list[str] = []
    if lock.get("status") != "Q2_V3_AMENDMENT1_FROZEN_NOT_RUN":
        failures.append("protocol status is not Amendment-1 frozen/not-run")
    for name, expected in lock.get("artifact_hashes", {}).items():
        path = review / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"frozen artifact hash mismatch: {name}")
    for relative, expected in lock.get("inherited_artifact_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"inherited frozen artifact hash mismatch: {relative}")

    source_rows = [
        json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line
    ]
    source = {str(row["id"]): row for row in source_rows}
    if len(source_rows) != 336 or len(source) != 336:
        failures.append("source subset is not 336 unique rows")

    seen: set[str] = set()
    counts: Counter[str] = Counter()
    legacy_executable = 0
    for filename in MANIFESTS:
        manifest = read_json(review / filename)
        if manifest.get("template_version") is not None:
            failures.append(f"ambiguous top-level template field: {filename}")
        if manifest.get("canonical_prompt_template") != GATE7_CANONICAL_TEMPLATE_VERSION:
            failures.append(f"template version mismatch: {filename}")
        if manifest.get("provenance_schema_version") != AMENDMENT1_PROVENANCE_SCHEMA:
            failures.append(f"provenance schema mismatch: {filename}")
        ids = [str(row["item_id"]) for row in manifest["items"]]
        if ids != manifest["item_ids"]:
            failures.append(f"item order mismatch: {filename}")
        if ordered_id_hash(ids) != manifest["ordered_ids_sha256"]:
            failures.append(f"ordered ID hash mismatch: {filename}")
        for row in manifest["items"]:
            item_id = str(row["item_id"])
            if item_id in seen:
                failures.append(f"duplicate item: {item_id}")
                continue
            seen.add(item_id)
            if "prompt_sha256" in row:
                failures.append(f"ambiguous prompt_sha256 field survived: {item_id}")
            record = row["prompt_provenance"]
            public = source.get(item_id)
            if public is None:
                failures.append(f"missing source: {item_id}")
                continue
            expected_prompt = canonical_q2_v3_task_prompt(
                str(public["code"]), str(public["input"])
            ).encode("utf-8")
            try:
                locked_prompt = base64.b64decode(
                    record["model_visible_prompt"]["utf8_base64"], validate=True
                )
            except Exception:  # noqa: BLE001
                failures.append(f"invalid prompt base64: {item_id}")
                continue
            checks = {
                "purpose": record["purpose"] == row["allocation"],
                "template": record["template_version"] == GATE7_CANONICAL_TEMPLATE_VERSION,
                "schema": record["provenance_schema_version"] == AMENDMENT1_PROVENANCE_SCHEMA,
                "bytes": locked_prompt == expected_prompt,
                "hash": record["model_visible_prompt"]["prompt_bytes_sha256"]
                == hashlib.sha256(expected_prompt).hexdigest(),
                "reference": record["reference"]["utf8_sha256"]
                == raw_utf8_sha256(str(public["output"])),
                "source": record["source_artifact"]["source_record_sha256"]
                == source_record_sha256(
                    item_id=item_id,
                    code=str(public["code"]),
                    value=str(public["input"]),
                    reference=str(public["output"]),
                ),
                "digest": record["provenance_digest_sha256"] == contract_digest(record),
            }
            for name, passed in checks.items():
                if not passed:
                    failures.append(f"{name} mismatch: {item_id}")
            if record["template_version"] != GATE7_CANONICAL_TEMPLATE_VERSION:
                legacy_executable += 1
            counts[str(row["allocation"])] += 1

    aggregate = read_json(review / "AMENDED_PROMPT_MANIFEST.json")
    if aggregate["record_count"] != 336 or len(aggregate["records"]) != 336:
        failures.append("aggregate prompt manifest count mismatch")
    aggregate_ids = [str(row["item_id"]) for row in aggregate["records"]]
    if set(aggregate_ids) != seen or len(aggregate_ids) != len(set(aggregate_ids)):
        failures.append("aggregate/allocation manifest identity mismatch")
    primary = read_json(review / "PRIMARY_PANEL_MANIFEST.json")
    if (
        primary["item_count"] != 200
        or ordered_id_hash(primary["item_ids"]) != EXPECTED_PRIMARY_HASH
    ):
        failures.append("primary panel identity changed")
    if seen != set(source):
        failures.append("source subset and purpose coverage differ")
    if legacy_executable:
        failures.append(f"legacy executable prompt records: {legacy_executable}")
    return {
        "schema_version": "q2-v3-amendment1-cpu-preflight-v1",
        "classification": (
            "Q2_V3_AMENDMENT1_CPU_PREFLIGHT_PASS"
            if not failures
            else "Q2_V3_AMENDMENT1_CPU_PREFLIGHT_FAIL"
        ),
        "records_checked": len(seen),
        "purpose_counts": dict(sorted(counts.items())),
        "template_version": GATE7_CANONICAL_TEMPLATE_VERSION,
        "provenance_schema_version": AMENDMENT1_PROVENANCE_SCHEMA,
        "legacy_executable_records": legacy_executable,
        "mismatch_count": len(failures),
        "failures": failures,
        "pass": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    result = run_preflight(args.review_dir.resolve(), args.source_jsonl.resolve())
    if args.write_report:
        args.write_report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
