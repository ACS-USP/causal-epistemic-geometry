#!/usr/bin/env python3
"""Validate raw Q2 V4.1 A2 hashes, ordering, and an independent JS reference.

This is a small label-free check.  It reads one persisted probe and never runs
the model, parses a response, or accesses a correctness/semantic outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from epistemic_geometry.experiments.q2_v4_1 import EXPECTED_SAFE_IDS, sha256_file  # noqa: E402

REVIEW = ROOT / "review/q2_v4_1_prediction_lock"
SHELLS = ("MEDIUM", "STRONG")
REFERENCE_TOLERANCE = 1e-11


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def stable_logsumexp(values: np.ndarray) -> float:
    maximum = max(float(value) for value in values)
    return maximum + math.log(sum(math.exp(float(value) - maximum) for value in values))


def reference_js(left: np.ndarray, right: np.ndarray) -> tuple[float, list[float]]:
    rows = []
    for left_row, right_row in zip(left, right, strict=True):
        left_log_z = stable_logsumexp(left_row)
        right_log_z = stable_logsumexp(right_row)
        left_logs = [float(value) - left_log_z for value in left_row]
        right_logs = [float(value) - right_log_z for value in right_row]
        terms = []
        for left_log, right_log in zip(left_logs, right_logs, strict=True):
            mixture_log = math.log((math.exp(left_log) + math.exp(right_log)) / 2.0)
            terms.append(
                0.5 * math.exp(left_log) * (left_log - mixture_log)
                + 0.5 * math.exp(right_log) * (right_log - mixture_log)
            )
        rows.append(math.fsum(terms))
    return math.fsum(rows) / len(rows), rows


def vectorized_js(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - np.logaddexp.reduce(a, axis=-1, keepdims=True)
    b = b - np.logaddexp.reduce(b, axis=-1, keepdims=True)
    m = np.logaddexp(a, b) - np.log(2.0)
    return float(
        np.mean(
            0.5 * np.sum(np.exp(a) * (a - m), axis=-1)
            + 0.5 * np.sum(np.exp(b) * (b - m), axis=-1)
        )
    )


def main() -> None:
    manifest = read_json(REVIEW / "A2_PROBE_MANIFEST.json")
    probe_ids = [str(value) for value in manifest["item_ids"]]
    names = list(EXPECTED_SAFE_IDS)
    expected_order = ["BASELINE"] + [
        f"{name}_{shell}" for name in names for shell in SHELLS
    ]
    raw_hashes: dict[str, str] = {}
    all_paths = []
    for directory_name in ("A2_FINGERPRINTS", "A2_REPEAT_FINGERPRINTS"):
        directory = REVIEW / directory_name
        files = sorted(directory.glob("*.npz"))
        if [path.stem for path in files] != sorted(probe_ids):
            raise RuntimeError(f"unexpected {directory_name} probe set")
        for path in files:
            relative = f"{directory_name}/{path.name}"
            raw_hashes[relative] = sha256_file(path)
            all_paths.append((relative, raw_hashes[relative]))
    if len(raw_hashes) != 24:
        raise RuntimeError("expected exactly 24 raw A2 files")
    digest = hashlib.sha256()
    for relative, value in sorted(all_paths):
        digest.update(f"{value}  {relative}\n".encode())

    probe_id = probe_ids[0]
    with np.load(REVIEW / "A2_FINGERPRINTS" / f"{probe_id}.npz", allow_pickle=False) as archive:
        if archive.files != expected_order:
            raise RuntimeError("raw A2 condition/controller ordering mismatch")
        pairs = [
            (archive[f"{names[0]}_MEDIUM"], archive[f"{names[1]}_MEDIUM"]),
            (archive[f"{names[0]}_STRONG"], archive[f"{names[1]}_STRONG"]),
        ]
    pair_results = []
    for left, right in pairs:
        reference_mean, row_values = reference_js(left, right)
        vectorized_mean = vectorized_js(left, right)
        if abs(reference_mean - vectorized_mean) > REFERENCE_TOLERANCE:
            raise RuntimeError("independent JS reference mismatch")
        if abs(reference_mean - sum(row_values) / len(row_values)) > REFERENCE_TOLERANCE:
            raise RuntimeError("checkpoint weighting is not equal-weight mean")
        pair_results.append(
            {
                "rows": int(left.shape[0]),
                "vocabulary": int(left.shape[1]),
                "reference_mean_js": reference_mean,
                "vectorized_mean_js": vectorized_mean,
                "absolute_difference": abs(reference_mean - vectorized_mean),
            }
        )
    p = np.asarray([0.25, 0.75], dtype=np.float64)
    q = np.asarray([0.75, 0.25], dtype=np.float64)
    hand_mean = 0.5 * np.sum(p * (np.log(p) - np.log((p + q) / 2.0))) + 0.5 * np.sum(
        q * (np.log(q) - np.log((p + q) / 2.0))
    )
    if not math.isfinite(hand_mean) or hand_mean <= 0.0:
        raise RuntimeError("natural-log JS hand check failed")
    write = {
        "schema_version": "q2-v4.1-a2-raw-reference-validation-v1",
        "validation_commit": git_head(),
        "semantic_outcomes": 0,
        "correctness_inspected": False,
        "raw_file_count": len(raw_hashes),
        "raw_file_sha256": dict(sorted(raw_hashes.items())),
        "raw_archive_aggregate_sha256": digest.hexdigest(),
        "probe_id": probe_id,
        "condition_order": expected_order,
        "shell_order": list(SHELLS),
        "controller_order": names,
        "log_base": "natural_log",
        "js_weighting": "0.5 KL(p||m) + 0.5 KL(q||m)",
        "checkpoint_aggregation": "equal_weight_mean",
        "reference_tolerance": REFERENCE_TOLERANCE,
        "independent_pair_checks": pair_results,
        "hand_distribution_check": {
            "p": p.tolist(),
            "q": q.tolist(),
            "js_nats": float(hand_mean),
        },
        "pass": True,
    }
    path = REVIEW / "A2_OFFLINE_REFERENCE_VALIDATION.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(write, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    hash_path = REVIEW / "A2_RAW_ARCHIVE_HASHES.json"
    hash_payload = {
        "schema_version": "q2-v4.1-a2-raw-archive-hashes-v1",
        "validation_commit": git_head(),
        "file_count": len(raw_hashes),
        "files": dict(sorted(raw_hashes.items())),
        "aggregate_sha256": digest.hexdigest(),
        "semantic_outcomes": 0,
        "correctness_inspected": False,
    }
    hash_tmp = hash_path.with_suffix(hash_path.suffix + ".tmp")
    hash_tmp.write_text(json.dumps(hash_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hash_tmp.replace(hash_path)
    print(json.dumps({"pass": True, "raw_files": len(raw_hashes), "aggregate": digest.hexdigest()}))


if __name__ == "__main__":
    main()
