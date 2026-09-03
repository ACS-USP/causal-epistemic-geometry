#!/usr/bin/env python3
"""Capture fresh-controller A2 fingerprints for Q2 OOS V2.

This runner is deliberately label-free.  It teacher-forces the already frozen
V4.1 continuation on the already frozen V4.1 A2 probes and persists logits for
the prospectively selected 16 fresh controllers.  It cannot access the N=300
semantic panel, references, correctness, or free generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_q2_oos_v2_presemantic import environment_preflight  # noqa: E402
from run_q2_v3 import EXECUTION_TEACHER_TEXT  # noqa: E402
from run_q2_v4_1_label_free_geometry import (  # noqa: E402
    SHELLS,
    build_backend,
    fingerprint,
    manifest_items,
)

from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402

REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout"
V2_STREAM = ROOT / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
SELECTED_BANK = REVIEW / "V2_SELECTED_CONTROLLER_BANK.json"
CAPTURE_LOCK = REVIEW / "LABEL_FREE_CAPTURE_LOCK.json"
STREAM_MANIFEST = V2_STREAM / "V2_CANDIDATE_BANK_MANIFEST.json"
VECTOR_DIR = V2_STREAM / "CANDIDATE_DIRECTIONS"
EXPECTED_CONDITIONS = 16 * 2
EXPECTED_PROBES = 12


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_selected() -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    selected = read_json(SELECTED_BANK)
    if selected["classification"] != "Q2_OOS_V2_SELECTED_BANK_GATE_PASS":
        raise RuntimeError("Q2_OOS_V2_SELECTED_BANK_NOT_QUALIFIED")
    names = [str(value) for value in selected["selected_ids"]]
    if len(names) != 16 or len(set(names)) != 16:
        raise RuntimeError("Q2_OOS_V2_SELECTED_BANK_IDENTITY_FAILURE")
    stream = read_json(STREAM_MANIFEST)
    records = {row["candidate_id"]: row for row in stream["candidates"]}
    vectors: dict[str, np.ndarray] = {}
    for name in names:
        row = records[name]
        path = ROOT / row["path"]
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"Q2_OOS_V2_VECTOR_FILE_HASH_MISMATCH:{name}")
        vector = np.load(path, allow_pickle=False).astype(np.float64)
        if vector_sha256(vector) != row["vector_array_sha256"]:
            raise RuntimeError(f"Q2_OOS_V2_VECTOR_ARRAY_HASH_MISMATCH:{name}")
        vectors[name] = vector
    return names, vectors, selected


def validate_lock(expected_commit: str) -> dict[str, Any]:
    if git_head() != expected_commit:
        raise RuntimeError("Q2_OOS_V2_LABEL_FREE_EXECUTION_COMMIT_MISMATCH")
    lock = read_json(CAPTURE_LOCK)
    checks = {
        "status": lock["status"] == "FROZEN_LABEL_FREE_CAPTURE_NOT_RUN",
        "selected_bank_hash": sha256_file(SELECTED_BANK) == lock["selected_bank_sha256"],
        "stream_manifest_hash": sha256_file(STREAM_MANIFEST) == lock["stream_manifest_sha256"],
        "capture_runner_hash": sha256_file(Path(__file__)) == lock["capture_runner_sha256"],
        "probe_manifest_hash": sha256_file(
            ROOT / "review/q2_v4_1_prediction_lock/A2_PROBE_MANIFEST.json"
        )
        == lock["a2_probe_manifest_sha256"],
        "semantic_outcomes_zero": lock["semantic_outcomes"] == 0,
        "correctness_forbidden": lock["correctness"] == "FORBIDDEN",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Q2_OOS_V2_LABEL_FREE_PREOPEN_FAILURE:{checks}")
    return checks


def existing_probe_valid(path: Path, expected_keys: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return set(archive.files) == expected_keys and all(
                archive[key].ndim == 2 and archive[key].shape[0] == 4
                for key in archive.files
            )
    except Exception:
        return False


def run(model_path: Path, output_dir: Path, expected_commit: str) -> None:
    lock_checks = validate_lock(expected_commit)
    environment = environment_preflight(model_path)
    names, vectors, selected = load_selected()
    condition_names = [f"{name}_{shell}" for name in names for shell in SHELLS]
    if len(condition_names) != EXPECTED_CONDITIONS:
        raise RuntimeError("Q2_OOS_V2_LABEL_FREE_CONDITION_COUNT_FAILURE")
    expected_keys = {"BASELINE", *condition_names}
    probes = manifest_items("A2_PROBE_MANIFEST.json")
    if len(probes) != EXPECTED_PROBES:
        raise RuntimeError("Q2_OOS_V2_LABEL_FREE_PROBE_COUNT_FAILURE")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "A2_FRESH_FINGERPRINTS"
    repeat_dir = output_dir / "A2_FRESH_REPEAT_FINGERPRINTS"
    preopen = output_dir / "LABEL_FREE_PREOPEN_SEAL.json"
    if any(raw_dir.glob("*.npz")) or any(repeat_dir.glob("*.npz")):
        preexisting = True
    else:
        preexisting = False
    atomic_json(
        preopen,
        {
            "source_commit": expected_commit,
            "lock_checks": lock_checks,
            "environment": environment,
            "selected_ids": names,
            "output_preexisting_for_resume": preexisting,
            "free_generation": False,
            "correctness": "FORBIDDEN",
            "correctness_inspected": False,
            "semantic_outcomes": 0,
        },
    )
    backend = build_backend(str(model_path))
    continuation = [
        int(value)
        for value in backend.tokenizer.encode(EXECUTION_TEACHER_TEXT, add_special_tokens=False)
    ]
    started = time.monotonic()
    deployment = selected["controllers"]
    for probe in probes:
        raw_path = raw_dir / f"{probe.item_id}.npz"
        repeat_path = repeat_dir / f"{probe.item_id}.npz"
        raw_ok = existing_probe_valid(raw_path, expected_keys)
        repeat_ok = existing_probe_valid(repeat_path, expected_keys)
        if raw_ok and repeat_ok:
            continue
        if raw_ok != repeat_ok:
            raise RuntimeError(f"Q2_OOS_V2_LABEL_FREE_PARTIAL_PROBE:{probe.item_id}")
        baseline = fingerprint(backend, probe, None, 0.0, vectors, continuation)
        repeated_baseline = fingerprint(backend, probe, None, 0.0, vectors, continuation)
        arrays = {"BASELINE": baseline}
        repeated_arrays = {"BASELINE": repeated_baseline}
        for name in names:
            for shell in SHELLS:
                condition = f"{name}_{shell}"
                alpha = float(deployment[condition]["alpha"])
                arrays[condition] = fingerprint(
                    backend, probe, name, alpha, vectors, continuation
                )
                repeated_arrays[condition] = fingerprint(
                    backend, probe, name, alpha, vectors, continuation
                )
        atomic_npz(raw_path, arrays)
        atomic_npz(repeat_path, repeated_arrays)
    files = sorted([*raw_dir.glob("*.npz"), *repeat_dir.glob("*.npz")])
    if len(files) != 2 * EXPECTED_PROBES:
        raise RuntimeError("Q2_OOS_V2_LABEL_FREE_EXECUTION_INCOMPLETE")
    hashes = {str(path.relative_to(output_dir)): sha256_file(path) for path in files}
    aggregate = hashlib.sha256(
        "".join(f"{name}:{hashes[name]}\n" for name in sorted(hashes)).encode()
    ).hexdigest()
    atomic_json(
        output_dir / "A2_FRESH_RAW_ARCHIVE_HASHES.json",
        {
            "schema_version": "q2-oos-v2-a2-fresh-raw-hashes-v1",
            "source_commit": expected_commit,
            "file_count": len(files),
            "files": hashes,
            "aggregate_sha256": aggregate,
            "selected_ids": names,
            "probe_count": len(probes),
            "condition_count": len(condition_names),
            "repeat_capture": True,
            "correctness_inspected": False,
            "semantic_outcomes": 0,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(
        json.dumps(
            {
                "status": "Q2_OOS_V2_LABEL_FREE_CAPTURE_COMPLETE",
                "file_count": len(files),
                "aggregate_sha256": aggregate,
                "semantic_outcomes": 0,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(args.model_path, args.output_dir, args.expected_commit)


if __name__ == "__main__":
    main()
