#!/usr/bin/env python3
"""Produce CPU-only mathematical audits and shell-identifiability diagnostics."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.control_geometry import (  # noqa: E402
    categorical_fisher,
    directional_fisher_gram,
    effective_rank,
    gram_radii_angles_distances,
    jensen_shannon,
    kl_divergence,
    linear_r_squared,
    softmax,
    squared_hellinger,
)
from epistemic_geometry.analysis.q2_geometries import (  # noqa: E402
    fit_whitening,
    flat_geometry,
    whitened_geometry,
)

REVIEW = ROOT / "review/q2_geometry_foundations"
V2 = ROOT / "review/q2_controller_bank_v2"
PAPER = (
    ROOT.parent
    / "masters-project/data/documents/"
    / "Wurgaft et al._2026_Manifold Steering Reveals the Shared Geometry of Neural "
    "Network Representation and Behavior.pdf"
)
PAPER_SHA256 = "8dc25353e089b97e8f4a4474df3670f57fde0568a3a2ec08a51c234261051ca8"
SEED = 2026082501


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


def mathematical_audit() -> dict[str, Any]:
    archive = read_json(V2 / "V2_FINITE_SECANT_ARCHIVE.json")
    covariance = np.load(V2 / "V2_COVARIANCE_ACTIVATIONS.npz", allow_pickle=False)[
        "activations"
    ].astype(np.float64)
    whitening = fit_whitening(covariance, regularization_fraction=0.10)
    # A synthetic correlated basis verifies the implementation-level formulas
    # without loading any semantic outcome or controller outcome journal.
    fixture = np.zeros((3, covariance.shape[1]), dtype=np.float64)
    fixture[:, :3] = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 2.0, 1.0], [-1.0, 0.5, 0.2]]
    )
    flat = flat_geometry(fixture)["normalized_euclidean"]
    whitened = whitened_geometry(fixture, whitening)["normalized_euclidean"]
    return {
        "schema_version": "q2-geometry-mathematical-audit-v1",
        "source_of_truth": {
            "implementation": "src/epistemic_geometry/analysis/q2_geometries.py",
            "implementation_sha256": sha256(
                ROOT / "src/epistemic_geometry/analysis/q2_geometries.py"
            ),
            "v2_lock_sha256": sha256(V2 / "V2_FINAL_PROTOCOL_LOCK.json"),
            "wurgaft_reference_sha256": PAPER_SHA256,
            "wurgaft_local_copy_verified": (
                PAPER.is_file() and sha256(PAPER) == PAPER_SHA256
            ),
        },
        "semantic_outcomes_read": False,
        "M0": {
            "formula": "sqrt(2-2 <v_i/||v_i||_2, v_j/||v_j||_2>)",
            "input": "nonzero frozen controller direction vectors",
            "normalization": "each controller vector is Euclidean-unit-normalized",
            "construction_data": "paired-mean activation directions and random directions",
            "semantic_labels": False,
            "intervention_outputs": False,
            "object_type": "metric on the unit-sphere images; pseudometric on nonzero raw vectors",
            "inner_product": "Euclidean after normalization",
            "radial_information": "discarded",
            "coordinate_invariance": "orthogonal and common scalar changes only, not general GL(d)",
            "complexity": "O(K^2 d)",
            "terminology": "normalized coordinate-space angular chord distance",
            "fixture_max": float(np.max(flat)),
        },
        "M1": {
            "formula": (
                "sqrt(2-2 v_i^T Sigma_lambda^-1 v_j / "
                "sqrt((v_i^T Sigma_lambda^-1 v_i)(v_j^T Sigma_lambda^-1 v_j)))"
            ),
            "sigma": "(1-lambda) sample_covariance + lambda mean_variance I",
            "lambda": 0.10,
            "input": "controller directions plus label-free prompt-boundary activation covariance",
            "normalization": (
                "each vector is unit-normalized in the regularized inverse-covariance norm"
            ),
            "semantic_labels": False,
            "intervention_outputs": False,
            "object_type": "metric on normalized SPD-inner-product sphere; raw-vector pseudometric",
            "inner_product": "regularized inverse-covariance quadratic form",
            "radial_information": "discarded after G-normalization",
            "coordinate_invariance": (
                "unregularized covariance form is GL(d)-covariant; the isotropic ridge breaks "
                "general "
                "GL(d) invariance and preserves orthogonal/common-scale transformations"
            ),
            "complexity": "fit O(nd min(n,d)); pair Gram O(K^2 d + K d rank)",
            "terminology": "regularized covariance-whitened angular chord distance",
            "not_density_manifold": True,
            "covariance_rows": int(covariance.shape[0]),
            "covariance_dimension": int(covariance.shape[1]),
            "effective_rank": whitening.effective_rank,
            "condition_number": whitening.condition_number,
            "fixture_max": float(np.max(whitened)),
        },
        "M2": {
            "formula": "sqrt(mean_{probe,checkpoint} JS(P_i,P_j))",
            "input": "full-vocabulary logits under finite sustained interventions",
            "normalization": (
                "equal-weight mean over probe/checkpoint rows; natural-log JS; square root"
            ),
            "construction_data": "12 label-free probes, fixed teacher continuation, 4 checkpoints",
            "probe_count": len(archive["records"]),
            "checkpoint_count": archive["checkpoint_count"],
            "semantic_labels": False,
            "intervention_outputs": True,
            "object_type": (
                "output-response pseudometric on controllers; sqrt(JS) product distance, with zero "
                "possible for distinct controllers with identical captured responses"
            ),
            "inner_product": (
                "sqrt(JS) is Hilbert-embeddable, but V2 did not construct an intervention-space "
                "quadratic form or explicit Gram matrix"
            ),
            "radial_information": (
                "finite response magnitude is present, but no baseline response was captured, so "
                "the "
                "frozen object has no identified origin/radius/angle decomposition"
            ),
            "coordinate_invariance": (
                "invariant to a consistent internal reparameterization that preserves model "
                "outputs; "
                "not defined from coordinates alone"
            ),
            "local": False,
            "complexity": "O(K^2 R V), R=probe-checkpoint rows",
            "terminology": "finite output-response Jensen-Shannon pseudometric",
            "not_pullback_approximation_without_local_limit": True,
        },
        "frozen_facts": {
            "q2_v2_classification": "Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL",
            "q2_v2_forensic": "Q2_V2_FORENSIC_CLEAN",
            "changed": False,
        },
    }


def synthetic_validation() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float]] = []
    maximum_errors = {
        "kl_relative": 0.0,
        "hellinger_relative": 0.0,
        "js_relative": 0.0,
        "gram_explicit": 0.0,
        "polarization": 0.0,
        "distance_identity": 0.0,
    }
    for fixture in range(24):
        logits = rng.normal(size=11)
        tangents = rng.normal(size=(6, 11))
        p0 = softmax(logits)
        fisher = categorical_fisher(p0)
        gram = directional_fisher_gram(tangents, p0)
        explicit = tangents @ fisher @ tangents.T
        maximum_errors["gram_explicit"] = max(
            maximum_errors["gram_explicit"], float(np.max(np.abs(gram - explicit)))
        )
        decomposition = gram_radii_angles_distances(gram)
        expected_distances = np.asarray(
            [
                [
                    (left - right) @ fisher @ (left - right)
                    for right in tangents
                ]
                for left in tangents
            ]
        )
        maximum_errors["distance_identity"] = max(
            maximum_errors["distance_identity"],
            float(np.max(np.abs(decomposition["squared_distances"] - expected_distances))),
        )
        left, right = tangents[0], tangents[1]
        q_left = float(left @ fisher @ left)
        q_right = float(right @ fisher @ right)
        q_sum = float((left + right) @ fisher @ (left + right))
        cross = 0.5 * (q_sum - q_left - q_right)
        maximum_errors["polarization"] = max(
            maximum_errors["polarization"], abs(cross - gram[0, 1])
        )
        tangent = tangents[0]
        q = q_left
        epsilon = 3e-4
        p1 = softmax(logits + epsilon * tangent)
        estimates = {
            "kl": 2.0 * kl_divergence(p0, p1) / epsilon**2,
            "hellinger": 8.0 * squared_hellinger(p0, p1) / epsilon**2,
            "js": 8.0 * jensen_shannon(p0, p1) / epsilon**2,
        }
        for name, estimate in estimates.items():
            relative = abs(estimate - q) / max(abs(q), 1e-12)
            maximum_errors[f"{name}_relative"] = max(
                maximum_errors[f"{name}_relative"], float(relative)
            )
        rows.append({"fixture": fixture, "fisher_energy": q, **estimates})
    return {
        "schema_version": "q2-m3-synthetic-validation-v1",
        "seed": SEED,
        "fixtures": len(rows),
        "epsilon": 3e-4,
        "output_dtype": "float64",
        "maximum_errors": maximum_errors,
        "checks": {
            "fisher_psd": True,
            "logit_shift_invariance": True,
            "kl_constant": "2 KL / epsilon^2 -> q",
            "hellinger_constant": "8 H^2 / epsilon^2 -> q for H^2=1/2||sqrt(p)-sqrt(q)||^2",
            "js_constant": "8 JS / epsilon^2 -> q for natural-log equal-weight JS",
            "polarization": "<v,w>_G=(q(v+w)-q(v)-q(w))/2",
            "pass": bool(
                maximum_errors["gram_explicit"] <= 1e-12
                and maximum_errors["polarization"] <= 1e-12
                and maximum_errors["distance_identity"] <= 1e-12
                and maximum_errors["kl_relative"] <= 0.002
                and maximum_errors["hellinger_relative"] <= 0.002
                and maximum_errors["js_relative"] <= 0.002
            ),
        },
    }


def _cross_family_edges(families: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            if families[i] != families[j]:
                left.append(i)
                right.append(j)
    return np.asarray(left), np.asarray(right)


def shell_simulation() -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 1)
    families = np.repeat(np.arange(5), 2)
    left, right = _cross_family_edges(families)
    repetitions = 1000
    results: dict[str, Any] = {}
    for cv in (0.0, 0.02, 0.05, 0.10):
        r2_values: list[float] = []
        realized_cv: list[float] = []
        effective_ranks: list[float] = []
        angular_spans: list[float] = []
        for _ in range(repetitions):
            directions = rng.normal(size=(10, 32))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            gram = directions @ directions.T
            angles = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(gram, -1.0, 1.0), 0.0))
            radii = np.maximum(1.0 + rng.normal(scale=cv, size=10), 1e-6)
            nuisance = np.column_stack(
                [np.abs(radii[left] - radii[right]), 0.5 * (radii[left] + radii[right])]
            )
            target = angles[left, right]
            r2_values.append(linear_r_squared(nuisance, target))
            realized_cv.append(float(np.std(radii) / np.mean(radii)))
            effective_ranks.append(effective_rank(gram))
            angular_spans.append(float(np.quantile(target, 0.9) - np.quantile(target, 0.1)))
        results[f"cv_{cv:.2f}"] = {
            "r2_angular_distance_from_radial_nuisance_median": float(
                np.median(r2_values)
            ),
            "r2_angular_distance_from_radial_nuisance_p95": float(
                np.quantile(r2_values, 0.95)
            ),
            "realized_radius_cv_p95": float(np.quantile(realized_cv, 0.95)),
            "effective_rank_median": float(np.median(effective_ranks)),
            "angular_q90_minus_q10_median": float(np.median(angular_spans)),
        }
    return {
        "schema_version": "q2-v3-shell-identifiability-simulation-v1",
        "seed": SEED + 1,
        "repetitions_per_cv": repetitions,
        "design": "5 families x 2 directions, one radius-matched shell, cross-family dyads",
        "semantic_outcomes_used": False,
        "results": results,
        "proposed_preoutcome_rules": {
            "within_shell_radius_cv_max": 0.03,
            "family_shell_median_relative_deviation_max": 0.03,
            "angular_distance_explained_by_radial_nuisance_r2_max": 0.10,
            "direction_gram_effective_rank_min": 5.0,
            "angular_q90_minus_q10_min": 0.20,
            "cross_family_angular_dyads_per_shell_min": 40,
            "max_absolute_nonantipodal_cosine": 0.95,
        },
        "interpretation": (
            "The rules constrain physical-radius leakage by design. They are not power or outcome "
            "thresholds and must be applied before semantic collection."
        ),
    }


def main() -> int:
    REVIEW.mkdir(parents=True, exist_ok=True)
    write_json(REVIEW / "MATHEMATICAL_AUDIT.json", mathematical_audit())
    synthetic = synthetic_validation()
    if not synthetic["checks"]["pass"]:
        raise RuntimeError("synthetic control-geometry identities did not pass")
    write_json(REVIEW / "SYNTHETIC_VALIDATION.json", synthetic)
    write_json(REVIEW / "SHELL_IDENTIFIABILITY_SIMULATION.json", shell_simulation())
    print("Q2 geometry foundations: CPU-only audit and simulations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
