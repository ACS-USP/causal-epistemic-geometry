#!/usr/bin/env python3
"""Freeze qualified Q2 vectors and outcome-free geometry before common outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.q2_geometries import (  # noqa: E402
    finite_secant_geometry,
    fit_whitening,
    flat_geometry,
    matrix_checks,
    whitened_geometry,
)
from epistemic_geometry.experiments.q2_controller_heldout import (  # noqa: E402
    CONTROLLER_IDS,
)

REVIEW = ROOT / "review/q2_controller_heldout_geometry"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_vectors(review: Path) -> np.ndarray:
    metadata = read_json(review / "CONTROLLER_BANK.json")
    return np.stack(
        [
            np.load(ROOT / metadata["vectors"][name]["path"], allow_pickle=False)
            for name in CONTROLLER_IDS
        ]
    ).astype(np.float64)


def load_secant_logits(review: Path) -> dict[str, np.ndarray]:
    archive = read_json(review / "FINITE_SECANT_ARCHIVE.json")
    values = {name: [] for name in CONTROLLER_IDS}
    for record in archive["records"]:
        path = review / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"finite-secant archive hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            for name in CONTROLLER_IDS:
                values[name].append(shard[name].astype(np.float64))
    return {name: np.concatenate(rows, axis=0) for name, rows in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--experiment-source-commit", required=True)
    parser.add_argument("--hourly-price-usd", type=float, required=True)
    parser.add_argument("--incurred-cost-usd", type=float, default=0.0)
    args = parser.parse_args()
    review = args.review_dir.resolve()
    if git_head() != args.experiment_source_commit:
        raise RuntimeError("final Q2 lock must bind the current source commit")
    qualification = read_json(review / "BANK_QUALIFICATION.json")
    if qualification["classification"] != "Q2_CONTROLLER_BANK_QUALIFIED":
        raise RuntimeError("Q2_CONTROLLER_BANK_NOT_QUALIFIED")
    candidate = read_json(review / "CANDIDATE_PROTOCOL_LOCK.json")
    vectors = load_vectors(review)
    flat = flat_geometry(vectors)
    covariance_path = review / "COVARIANCE_ACTIVATIONS.npz"
    with np.load(covariance_path, allow_pickle=False) as archive:
        covariance_activations = archive["activations"].astype(np.float64)
    whitening = fit_whitening(covariance_activations, regularization_fraction=0.10)
    whitened = whitened_geometry(vectors, whitening)
    secant = finite_secant_geometry(load_secant_logits(review), CONTROLLER_IDS)
    matrices_path = review / "GEOMETRY_MATRICES.npz"
    np.savez_compressed(
        matrices_path,
        M0_FLAT=flat["normalized_euclidean"],
        M0_COSINE=flat["cosine_distance"],
        M1_WHITENED=whitened["normalized_euclidean"],
        M1_WHITENED_COSINE=whitened["cosine_distance"],
        M2_FINITE_SECANT=secant["sqrt_mean_js"],
        M2_MEAN_JS=secant["mean_js"],
    )
    geometry_metadata = {
        "controller_order": list(CONTROLLER_IDS),
        "M0": {
            "algebraic_identity_max_error": flat["algebraic_identity_max_error"],
            "checks": matrix_checks(flat["normalized_euclidean"]),
        },
        "M1": {
            "fit": {
                **asdict(whitening),
                "mean": None,
                "right_singular_vectors": None,
                "eigenvalues": whitening.eigenvalues.tolist(),
            },
            "checks": matrix_checks(whitened["normalized_euclidean"]),
            "covariance_archive_sha256": sha256(covariance_path),
        },
        "M2": {
            **secant["metadata"],
            "checks": matrix_checks(secant["sqrt_mean_js"]),
            "finite_secant_archive_sha256": sha256(review / "FINITE_SECANT_ARCHIVE.json"),
            "persisted_representation": "float16 full-vocabulary logits",
        },
        "geometry_matrices_sha256": sha256(matrices_path),
        "semantic_correctness_used": False,
    }
    write_json(review / "GEOMETRY_LOCK.json", geometry_metadata)
    preflight = read_json(review / "THROUGHPUT_PREFLIGHT.json")
    projected_main = (
        float(preflight["projected_hours_with_20pct_margin"]) * args.hourly_price_usd
    )
    projected_cumulative = args.incurred_cost_usd + projected_main
    cost = {
        "hourly_price_usd": args.hourly_price_usd,
        "incurred_pre_main_usd": args.incurred_cost_usd,
        "projected_main_with_20pct_margin_usd": projected_main,
        "projected_cumulative_usd": projected_cumulative,
        "target_usd": 8.50,
        "soft_ceiling_usd": 12.0,
        "hard_ceiling_usd": 15.0,
        "cost_gate_pass": projected_cumulative <= 15.0,
    }
    write_json(review / "COST_GATE.json", cost)
    if not cost["cost_gate_pass"]:
        raise RuntimeError("Q2_BLOCKED_PROJECTED_COST")
    bank_hashes = {
        name: candidate_record["canonical_float64_vector_sha256"]
        for name, candidate_record in read_json(review / "CONTROLLER_BANK.json")[
            "vectors"
        ].items()
    }
    lock = {
        **candidate,
        "schema_version": "q2-controller-heldout-final-v1",
        "status": "FROZEN_PRE_COMMON_PANEL",
        "experiment_source_commit": args.experiment_source_commit,
        "controller_bank": {
            "controller_order": list(CONTROLLER_IDS),
            "vector_hashes": bank_hashes,
            "bank_qualification_sha256": sha256(review / "BANK_QUALIFICATION.json"),
            "bank_validation_sha256": sha256(review / "BANK_VALIDATION.json"),
            "source_qualification_sha256": sha256(review / "SOURCE_QUALIFICATION.json"),
            "manipulation_qualification_sha256": sha256(
                review / "MANIPULATION_QUALIFICATION.json"
            ),
        },
        "geometry_lock": {
            "geometry_lock_sha256": sha256(review / "GEOMETRY_LOCK.json"),
            "geometry_matrices_sha256": sha256(matrices_path),
            "captured_before_common_panel": True,
        },
        "cost_gate": cost,
        "common_panel_outcomes_existing_at_lock": False,
    }
    write_json(review / "PROTOCOL_LOCK.json", lock)
    binding = {
        "experiment_source_commit": args.experiment_source_commit,
        "protocol_lock_sha256": sha256(review / "PROTOCOL_LOCK.json"),
        "bank_qualification_sha256": sha256(review / "BANK_QUALIFICATION.json"),
        "geometry_matrices_sha256": sha256(matrices_path),
    }
    write_json(review / "EXPERIMENT_SOURCE_COMMIT.json", binding)
    write_json(
        review / "artifact_hashes_preoutcome.json",
        {
            path.name: sha256(path)
            for path in sorted(review.iterdir())
            if path.is_file() and path.name != "artifact_hashes_preoutcome.json"
        },
    )
    (review / "PROTOCOL_LOCK.md").write_text(
        "# Q2 final bank and geometry lock\n\n"
        "`FROZEN_PRE_COMMON_PANEL`\n\n"
        f"Source commit: `{args.experiment_source_commit}`.\n\n"
        "The qualified K=16 L27 bank, 10/6 source-family split, M0/M1/M2 "
        "geometry matrices, 120-item panel, 4,080-row independent schedule, "
        "QAP/bootstrap rules, thresholds, and cost gate are immutable before "
        "common-panel semantic outcomes.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "classification": "Q2_FINAL_PROTOCOL_LOCKED",
                "experiment_source_commit": args.experiment_source_commit,
                "projected_cumulative_usd": projected_cumulative,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
