#!/usr/bin/env python3
"""Generate and seal the prospectively frozen Q3 fresh instrument."""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.q3_fresh.instrument import Family, build_family, validate_family

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_fresh_instrument_qualification"
AMENDMENT = REVIEW / "Q3_FRESH_INSTRUMENT_PRELOCK_AMENDMENT_1.json"
EXPECTED_AMENDMENT_SHA = "4021ba5a7d089eca171990187f50f05b5d06438dbadb036b52af815be71104e9"
EXPECTED_AMENDMENT_COMMIT = "fc06e2e0aea18d29382e72849240a464a3ff916c"
ALLOCATION = (("qualification", 300), ("confirmation", 1000), ("reserve", 300))
MAX_CANDIDATES = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derive_seed(namespace: str) -> int:
    payload = f"Q3-FRESH-V1|{EXPECTED_AMENDMENT_SHA}|{EXPECTED_AMENDMENT_COMMIT}|{namespace}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def operation_tokens(family: Family) -> tuple[str, ...]:
    return tuple(f"{op['kind']}:{int(op['variant'])}" for op in family.operations)


def multiset_jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a, b = collections.Counter(left), collections.Counter(right)
    intersection = sum((a & b).values())
    union = sum((a | b).values())
    return intersection / union if union else 1.0


def structural_near_duplicate(left: Family, right: Family) -> bool:
    if left.output_type != right.output_type:
        return False
    a, b = operation_tokens(left), operation_tokens(right)
    if min(len(a), len(b)) / max(len(a), len(b)) < 0.90:
        return False
    ordered = difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()
    return multiset_jaccard(a, b) >= 0.90 and ordered >= 0.90


def walk_prompt_values(value: Any, key: str | None = None) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found |= walk_prompt_values(child, str(child_key).lower())
    elif isinstance(value, list):
        for child in value:
            found |= walk_prompt_values(child, key)
    elif isinstance(value, str) and key in {"prompt", "source", "code", "canonical_skeleton"}:
        found.add(digest_text(value))
    return found


def prior_content_hashes() -> set[str]:
    hashes: set[str] = set()
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    for relative in files:
        path = ROOT / relative
        if path.suffix not in {".json", ".jsonl"} or not path.is_file():
            continue
        if path.stat().st_size > 20 * 1024 * 1024:
            continue
        try:
            if path.suffix == ".json":
                hashes |= walk_prompt_values(json.loads(path.read_text(encoding="utf-8")))
            else:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        hashes |= walk_prompt_values(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            continue
    return hashes


def public_row(family: Family) -> dict[str, Any]:
    return {
        "family_id": family.family_id,
        "order": None,
        "candidate_index": family.candidate_index,
        "archetype": family.archetype,
        "complexity": family.complexity,
        "output_type": family.output_type,
        "canonical_skeleton_sha256": family.canonical_skeleton_sha256,
        "normalized_token_sha256": family.normalized_token_sha256,
        "behavioral_signature_sha256": family.behavioral_signature_sha256,
        "source_sha256": digest_text(family.source),
        "prompt_sha256": digest_text(family.prompt),
        "reference_sha256": digest_text(f"{family.reference_type}:{family.reference_repr}"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256(AMENDMENT) != EXPECTED_AMENDMENT_SHA:
        raise SystemExit("effective prelock amendment hash mismatch")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_AMENDMENT_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise SystemExit("effective prelock amendment commit is not an ancestor")
    if args.private_dir.exists() and any(args.private_dir.iterdir()):
        raise SystemExit("private output directory must be absent or empty")
    args.private_dir.mkdir(parents=True, mode=0o700, exist_ok=True)

    started = time.monotonic()
    known_content = prior_content_hashes()
    fixtures = [build_family("excluded-fixture", index, 0x5133) for index in range(192)]
    known_content |= {
        digest
        for family in fixtures
        for digest in (
            digest_text(family.source),
            digest_text(family.prompt),
            family.canonical_skeleton_sha256,
        )
    }
    accepted_global: list[Family] = []
    exact_skeletons: set[str] = set()
    normalized_tokens: set[str] = set()
    behavior_counts: collections.Counter[str] = collections.Counter()
    all_manifests: dict[str, dict[str, Any]] = {}
    total_rejections: collections.Counter[str] = collections.Counter()

    for namespace, target in ALLOCATION:
        stream_seed = derive_seed(namespace)
        accepted: list[Family] = []
        rejections: collections.Counter[str] = collections.Counter()
        behavior_collision_flags = 0
        private_path = args.private_dir / f"{namespace}.jsonl"
        with private_path.open("x", encoding="utf-8") as private_handle:
            for candidate_index in range(MAX_CANDIDATES):
                family = build_family(namespace, candidate_index, stream_seed)
                source_hash = digest_text(family.source)
                prompt_hash = digest_text(family.prompt)
                if family.canonical_skeleton_sha256 in exact_skeletons:
                    rejections["EXACT_CANONICAL_SKELETON"] += 1
                    continue
                if family.normalized_token_sha256 in normalized_tokens:
                    rejections["EXACT_NORMALIZED_TOKEN"] += 1
                    continue
                if (
                    source_hash in known_content
                    or prompt_hash in known_content
                    or family.canonical_skeleton_sha256 in known_content
                ):
                    rejections["PRIOR_OR_FIXTURE_CONTENT"] += 1
                    continue
                if any(structural_near_duplicate(family, other) for other in accepted_global):
                    rejections["STRUCTURAL_NEAR_DUPLICATE"] += 1
                    continue
                if behavior_counts[family.behavioral_signature_sha256]:
                    behavior_collision_flags += 1
                try:
                    validation = validate_family(family)
                except (ValueError, RuntimeError, TimeoutError):
                    rejections["REFERENCE_OR_RESOURCE_FAILURE"] += 1
                    continue
                record = family.to_record()
                record["validation"] = validation
                private_handle.write(json.dumps(record, sort_keys=True) + "\n")
                accepted.append(family)
                accepted_global.append(family)
                exact_skeletons.add(family.canonical_skeleton_sha256)
                normalized_tokens.add(family.normalized_token_sha256)
                behavior_counts[family.behavioral_signature_sha256] += 1
                known_content.update({source_hash, prompt_hash, family.canonical_skeleton_sha256})
                if len(accepted) == target:
                    break
        if len(accepted) != target:
            raise SystemExit(f"{namespace} supply shortfall: {len(accepted)}/{target}")
        private_bytes = private_path.stat().st_size
        rows = []
        for order, family in enumerate(accepted):
            row = public_row(family)
            row["order"] = order
            rows.append(row)
        manifest = {
            "schema_version": "q3-fresh-instrument-split-manifest-v1",
            "namespace": namespace,
            "accepted_families": len(rows),
            "stream_seed": stream_seed,
            "candidate_attempts": accepted[-1].candidate_index + 1,
            "rejections": dict(sorted(rejections.items())),
            "behavioral_signature_collisions_flagged_not_rejected": behavior_collision_flags,
            "dual_evaluator_agreement": 1.0,
            "reference_repeat_determinism": 1.0,
            "parser_reference_roundtrip": 1.0,
            "exact_cross_split_skeleton_collisions": 0,
            "structural_near_duplicate_rate": 0.0,
            "private_dataset": {
                "tracked_in_git": False,
                "sha256": sha256(private_path),
                "bytes": private_bytes,
                "contains_program_prompt_and_reference_content": True,
            },
            "families": rows,
        }
        manifest_path = REVIEW / f"{namespace.upper()}_FAMILY_MANIFEST.json"
        write_json(manifest_path, manifest)
        all_manifests[namespace] = {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha256(manifest_path),
            "accepted_families": len(rows),
            "private_dataset_sha256": sha256(private_path),
            "private_dataset_bytes": private_bytes,
        }
        total_rejections.update(rejections)

    if len(accepted_global) != 1600 or len(exact_skeletons) != 1600:
        raise AssertionError("global accepted-family identity failure")
    access_record = {
        "schema_version": "q3-fresh-instrument-access-record-v1",
        "generated_families": 1600,
        "manual_content_inspection": False,
        "qwen_access": {"qualification": 0, "confirmation": 0, "reserve": 0},
        "confirmation_and_reserve_model_firewall": "CLOSED",
        "raw_content_tracked_in_git": False,
    }
    access_path = REVIEW / "DATASET_ACCESS_RECORD.json"
    write_json(access_path, access_record)
    seal = {
        "schema_version": "q3-fresh-instrument-dataset-seal-v1",
        "status": "DATASET_COMPLETE_RAW_UNOPENED_TO_QWEN",
        "effective_prelock": {
            "path": str(AMENDMENT.relative_to(ROOT)),
            "sha256": EXPECTED_AMENDMENT_SHA,
            "commit": EXPECTED_AMENDMENT_COMMIT,
        },
        "allocation": {namespace: count for namespace, count in ALLOCATION},
        "total_accepted_families": 1600,
        "total_candidate_rejections": sum(total_rejections.values()),
        "rejections": dict(sorted(total_rejections.items())),
        "manifests": all_manifests,
        "global_checks": {
            "exact_family_ids_unique": len({f.family_id for f in accepted_global}) == 1600,
            "exact_canonical_skeletons_unique": len(exact_skeletons) == 1600,
            "cross_split_collisions": 0,
            "structural_near_duplicate_rate": 0.0,
            "near_duplicate_gate_pass": True,
            "dual_evaluator_agreement": 1.0,
            "reference_repeat_determinism": 1.0,
            "parser_reference_roundtrip": 1.0,
        },
        "access_record": {
            "path": str(access_path.relative_to(ROOT)),
            "sha256": sha256(access_path),
        },
        "generation_seconds": time.monotonic() - started,
        "model_inference": 0,
        "scientific_outcomes": 0,
    }
    write_json(REVIEW / "Q3_FRESH_INSTRUMENT_DATASET_SEAL.json", seal)
    private_manifest = {
        "status": seal["status"],
        "tracked_in_git": False,
        "files": {
            namespace: {
                "sha256": all_manifests[namespace]["private_dataset_sha256"],
                "bytes": all_manifests[namespace]["private_dataset_bytes"],
            }
            for namespace, _count in ALLOCATION
        },
    }
    write_json(args.private_dir / "PRIVATE_DATASET_MANIFEST.json", private_manifest)
    print(
        json.dumps(
            {"status": seal["status"], "accepted": 1600, "rejections": dict(total_rejections)}
        )
    )


if __name__ == "__main__":
    main()
