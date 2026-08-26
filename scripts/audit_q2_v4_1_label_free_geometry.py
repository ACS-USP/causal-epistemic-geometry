#!/usr/bin/env python3
"""Independent forensic audit of the persisted label-free A1/A2 artifacts.

The audit intentionally does not import the primary A2 consolidator.  It
recomputes the finite-vocabulary JS distances from raw arrays using an
independent log-sum-exp implementation and checks the persisted matrices,
hashes, ordering, and scientific-firewall state.  No model, generation,
parser, correctness label, or semantic outcome is accessed.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from epistemic_geometry.experiments.q2_v4_1 import EXPECTED_SAFE_IDS, sha256_file  # noqa: E402

REVIEW = ROOT / "review/q2_v4_1_prediction_lock"
OLD_REVIEW = ROOT / "review/q2_v4_1_31_safe_bank_review"
VECTOR_DIR = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_DIRECTIONS"
SHELLS = ("MEDIUM", "STRONG")
SAFE_MANIFEST_SHA256 = "a641d612628c4f9eff2ae9fdf12d3ad17af5a3e921ec726d31c208ee5e030447"
EXPECTED_ENVIRONMENT = "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
AUDIT_MATRIX_TOLERANCE = 1e-8
AUDIT_SCALAR_TOLERANCE = 1e-10


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def direct_js(left: np.ndarray, right: np.ndarray) -> float:
    """Independent JS calculation using explicit max-shift normalization."""

    row_values = []
    for left_row, right_row in zip(left, right, strict=True):
        left_values = np.asarray(left_row, dtype=np.float64)
        right_values = np.asarray(right_row, dtype=np.float64)
        left_max = float(np.max(left_values))
        right_max = float(np.max(right_values))
        left_exp = np.exp(left_values - left_max)
        right_exp = np.exp(right_values - right_max)
        left_log = left_values - (left_max + math.log(float(np.sum(left_exp))))
        right_log = right_values - (right_max + math.log(float(np.sum(right_exp))))
        left_prob = np.exp(left_log)
        right_prob = np.exp(right_log)
        mixture_log = np.logaddexp(left_log, right_log) - math.log(2.0)
        row_values.append(
            float(
                0.5 * np.sum(left_prob * (left_log - mixture_log))
                + 0.5 * np.sum(right_prob * (right_log - mixture_log))
            )
        )
    return float(np.mean(np.asarray(row_values, dtype=np.float64)))


def pairwise(names: list[str], arrays: dict[str, np.ndarray], workers: int) -> np.ndarray:
    result = np.zeros((len(names), len(names)), dtype=np.float64)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]

    def calculate(pair: tuple[int, int]) -> tuple[int, int, float]:
        i, j = pair
        return i, j, direct_js(arrays[names[i]], arrays[names[j]])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, j, value in executor.map(calculate, pairs):
            result[i, j] = result[j, i] = value
    return result


def raw_hashes() -> dict[str, str]:
    archive = read_json(REVIEW / "A2_RAW_ARCHIVE_HASHES.json")
    values = {}
    for relative, expected in archive["files"].items():
        actual = sha256_file(REVIEW / relative)
        if actual != expected:
            raise RuntimeError(f"raw hash mismatch: {relative}")
        values[relative] = actual
    if len(values) != 24:
        raise RuntimeError("raw A2 archive does not contain 24 files")
    for probe_id in {Path(relative).stem for relative in values}:
        first = values[f"A2_FINGERPRINTS/{probe_id}.npz"]
        repeat = values[f"A2_REPEAT_FINGERPRINTS/{probe_id}.npz"]
        if first != repeat:
            raise RuntimeError(f"raw/repeat A2 archive differs: {probe_id}")
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    pinned = lock["label_free_offline_consolidation"]["raw_file_sha256"]
    if pinned != values:
        raise RuntimeError("PROTOCOL_LOCK raw hash map differs from persisted archive")
    return values


def load_shell(shell: str) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    manifest = read_json(REVIEW / "A2_PROBE_MANIFEST.json")
    probe_ids = [str(value) for value in manifest["item_ids"]]
    names = list(EXPECTED_SAFE_IDS)
    first_path = REVIEW / "A2_FINGERPRINTS" / f"{probe_ids[0]}.npz"
    with np.load(first_path, allow_pickle=False) as first:
        baseline_shape = first["BASELINE"].shape
    total_rows = len(probe_ids) * baseline_shape[0]
    baseline = np.empty((total_rows, baseline_shape[1]), dtype=np.float32)
    arrays = {name: np.empty_like(baseline) for name in names}
    for probe_index, probe_id in enumerate(probe_ids):
        path = REVIEW / "A2_FINGERPRINTS" / f"{probe_id}.npz"
        with np.load(path, allow_pickle=False) as archive:
            start = probe_index * baseline_shape[0]
            stop = start + baseline_shape[0]
            expected_keys = ["BASELINE"] + [
                f"{name}_{candidate_shell}" for name in names for candidate_shell in SHELLS
            ]
            if archive.files != expected_keys:
                raise RuntimeError(f"A2 key/order mismatch: {probe_id}")
            baseline[start:stop] = archive["BASELINE"]
            for name in names:
                arrays[name][start:stop] = archive[f"{name}_{shell}"]
    return names, baseline, arrays


def vector_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = OLD_REVIEW / "SAFE_31_IMMUTABLE_MANIFEST.json"
    if sha256_file(manifest_path) != SAFE_MANIFEST_SHA256:
        raise RuntimeError("safe-bank manifest hash mismatch")
    manifest = read_json(manifest_path)
    coefficients = np.asarray(
        [row["coefficients"] for row in manifest["directions"]], dtype=np.float64
    )
    vectors = []
    for row in manifest["directions"]:
        path = VECTOR_DIR / f"{row['candidate_id']}.npy"
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"vector hash mismatch: {row['candidate_id']}")
        vectors.append(np.load(path, allow_pickle=False).astype(np.float64))
    return coefficients, np.asarray(vectors, dtype=np.float64)


def manual_whitened(vectors: np.ndarray, fit_archive: dict[str, np.ndarray]) -> np.ndarray:
    basis = fit_archive["right_singular_vectors"]
    eigenvalues = fit_archive["eigenvalues"]
    fraction = float(fit_archive["regularization_fraction"][0])
    ridge = float(fit_archive["regularization_value"][0])
    projected = vectors @ basis.T
    adjusted = (1.0 - fraction) * eigenvalues + ridge
    gram = (vectors @ vectors.T) / ridge
    gram += (projected * ((1.0 / adjusted) - (1.0 / ridge))[None, :]) @ projected.T
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    cosine = np.clip(gram / np.outer(norms, norms), -1.0, 1.0)
    distance = 1.0 - cosine
    np.fill_diagonal(distance, 0.0)
    return distance


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    raw = raw_hashes()
    with np.load(REVIEW / "PREDICTION_MATRICES.npz", allow_pickle=False) as archive:
        primary = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    coefficients, vectors = vector_geometry()
    a0 = 1.0 - (coefficients / np.linalg.norm(coefficients, axis=1)[:, None]) @ (
        coefficients / np.linalg.norm(coefficients, axis=1)[:, None]
    ).T
    np.fill_diagonal(a0, 0.0)
    with np.load(REVIEW / "A1_COVARIANCE_FIT.npz", allow_pickle=False) as archive:
        fit_archive = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    a1 = manual_whitened(vectors, fit_archive)
    crosscheck_rows = []
    max_difference = 0.0
    for key, recomputed in {
        "A0_MEDIUM": a0,
        "A0_STRONG": a0,
        "A1_MEDIUM": a1,
        "A1_STRONG": a1,
    }.items():
        difference = float(np.max(np.abs(recomputed - primary[key])))
        max_difference = max(max_difference, difference)
        crosscheck_rows.append(
            {
                "artifact": key,
                "primary": "PREDICTION_MATRICES.npz",
                "max_abs_difference": difference,
            }
        )
    shell_reports = {}
    for shell in SHELLS:
        names, baseline, arrays = load_shell(shell)
        radii = np.asarray([direct_js(arrays[name], baseline) for name in names])
        d2 = pairwise(names, arrays, args.workers)
        gram = 0.5 * (radii[:, None] + radii[None, :] - d2)
        norm = np.sqrt(np.maximum(radii, 0.0))
        cosine = np.clip(gram / np.outer(norm, norm), -1.0, 1.0)
        np.fill_diagonal(cosine, 1.0)
        a2 = 1.0 - cosine
        np.fill_diagonal(a2, 0.0)
        d2_distance = np.sqrt(np.maximum(d2, 0.0))
        for key, recomputed in {
            f"A2_{shell}": a2,
            f"D2_{shell}": d2_distance,
        }.items():
            difference = float(np.max(np.abs(recomputed - primary[key])))
            max_difference = max(max_difference, difference)
            crosscheck_rows.append(
                {
                    "artifact": key,
                    "primary": "PREDICTION_MATRICES.npz",
                    "max_abs_difference": difference,
                }
            )
        shell_reports[shell] = {
            "rows": int(baseline.shape[0]),
            "vocabulary": int(baseline.shape[1]),
            "recomputed_gram_min_eigenvalue": float(np.min(np.linalg.eigvalsh(gram))),
            "recomputed_cosine_range": [float(np.min(cosine)), float(np.max(cosine))],
            "matrix_max_abs_difference": max(
                row["max_abs_difference"]
                for row in crosscheck_rows
                if row["artifact"] in {f"A2_{shell}", f"D2_{shell}"}
            ),
        }
    environment = read_json(REVIEW / "ENVIRONMENT_PROVENANCE.json")
    environment_pass = (
        environment.get("qualified_environment_profile") == EXPECTED_ENVIRONMENT
        and environment.get("profile_pass") is True
        and environment.get("model_revision") == "b968826d9c46dd6066d109eabc6255188de91218"
        and environment.get("torch") == "2.13.0+cu130"
        and environment.get("transformers") == "4.57.6"
        and environment.get("dtype") == "bfloat16"
        and environment.get("attention") == "sdpa"
        and environment.get("gpu") == "NVIDIA GB10"
        and environment.get("access_path") == "direct_ssh_spark1_no_local_dstack"
    )
    run_state = read_json(REVIEW / "LABEL_FREE_GEOMETRY_RUN.json")
    firewall_pass = (
        run_state.get("status") == "COMPLETE"
        and run_state.get("semantic_outcomes") == 0
        and run_state.get("correctness_inspected") is False
        and run_state.get("primary_panel_processed") is False
    )
    pass_audit = bool(
        len(raw) == 24
        and max_difference <= AUDIT_MATRIX_TOLERANCE
        and environment_pass
        and firewall_pass
    )
    audit_classification = (
        "GATE12_1_FORENSIC_CLEAN"
        if pass_audit
        else "GATE12_1_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
    )
    with (REVIEW / "METRIC_CROSSCHECK.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact", "primary", "max_abs_difference"])
        writer.writeheader()
        writer.writerows(crosscheck_rows)
    report = {
        "schema_version": "q2-v4.1-label-free-forensic-audit-v1",
        "audit_commit": git_head(),
        "classification": audit_classification,
        "scientific_items_processed": 0,
        "semantic_outcomes": 0,
        "correctness_inspected": False,
        "raw_file_count": len(raw),
        "raw_archive_hashes_verified": True,
        "raw_repeat_byte_identity_verified": True,
        "independent_js": {
            "log_base": "natural_log",
            "weighting": "0.5 KL(p||m) + 0.5 KL(q||m)",
            "aggregation": "equal_weight_mean_over_48_probe_checkpoint_rows",
            "implementation": (
                "explicit max-shift normalization, independent from primary consolidator"
            ),
        },
        "matrix_max_abs_difference": max_difference,
        "matrix_tolerance": AUDIT_MATRIX_TOLERANCE,
        "shells": shell_reports,
        "environment_pass": environment_pass,
        "firewall_pass": firewall_pass,
        "no_model_execution": True,
        "no_semantic_parser_execution": True,
        "no_gpu_used_for_consolidation": True,
        "tolerance_note": "Audit tolerance was frozen in the source before this recomputation.",
    }
    (REVIEW / "FORENSIC_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (REVIEW / "FORENSIC_AUDIT.md").write_text(
        "# Q2 V4.1 label-free forensic audit\n\n"
        f"Classification: `{audit_classification}`.\n\n"
        "This audit independently recomputed A0, A1, A2, and D2 from the "
        "persisted arrays. It used an explicit max-shift natural-log JS "
        "implementation, equal weighting over 48 probe/checkpoint rows, and "
        "did not import the primary A2 consolidator.\n\n"
        f"Maximum matrix absolute difference: `{max_difference:.12g}` "
        f"(frozen audit tolerance `{AUDIT_MATRIX_TOLERANCE}`).\n\n"
        f"Raw files verified: `{len(raw)}/24`; raw/repeat byte identity: `PASS`; "
        f"environment: `{'PASS' if environment_pass else 'FAIL'}`; "
        f"scientific firewall: `{'PASS' if firewall_pass else 'FAIL'}`.\n\n"
        "Scientific items processed: `0`. Semantic outcomes: `0`. Correctness "
        "inspected: `False`.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"classification": audit_classification, "max_abs_difference": max_difference}
        )
    )


if __name__ == "__main__":
    main()
