#!/usr/bin/env python3
"""Independent audit for the outcome-free Q2 V4.1 bank review.

This deliberately does not import the primary review runner. It recomputes
the frozen-bank geometry, reserve probabilities, and the mechanical decision
from the immutable V4 artifacts and persisted planning CSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_v4_1_31_safe_bank_review"
CANDIDATES = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json"
SAFETY = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_SAFETY_REPORT.json"
V4_AUDIT = ROOT / "review/q2_v4_spark1_presemantic/SAFETY_FORENSIC_AUDIT.json"
EXPECTED = {
    "V4_DIRECTION_00", "V4_DIRECTION_01", "V4_DIRECTION_02", "V4_DIRECTION_03",
    "V4_DIRECTION_04", "V4_DIRECTION_06", "V4_DIRECTION_07", "V4_DIRECTION_08",
    "V4_DIRECTION_09", "V4_DIRECTION_10", "V4_DIRECTION_11", "V4_DIRECTION_13",
    "V4_DIRECTION_15", "V4_DIRECTION_17", "V4_DIRECTION_18", "V4_DIRECTION_19",
    "V4_DIRECTION_20", "V4_DIRECTION_22", "V4_DIRECTION_23", "V4_DIRECTION_24",
    "V4_DIRECTION_26", "V4_DIRECTION_28", "V4_DIRECTION_29", "V4_DIRECTION_30",
    "V4_DIRECTION_31", "V4_DIRECTION_32", "V4_DIRECTION_33", "V4_DIRECTION_34",
    "V4_DIRECTION_35", "V4_DIRECTION_37", "V4_DIRECTION_39",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry(values: np.ndarray) -> dict[str, Any]:
    unit = values / np.linalg.norm(values, axis=1, keepdims=True)
    singular = np.linalg.svd(unit, compute_uv=False)
    gram = unit @ unit.T
    upper = gram[np.triu_indices(len(unit), 1)]
    a0 = 1.0 - upper
    energy = singular**2
    probability = energy / np.sum(energy)
    effective = float(np.exp(-np.sum(probability * np.log(probability))))
    return {
        "rank": int(np.linalg.matrix_rank(unit, tol=1e-10)),
        "singular_values": singular.tolist(),
        "effective_rank": effective,
        "stable_rank": float(np.sum(energy) / energy[0]),
        "condition_number": float(singular[0] / singular[-1]),
        "max_absolute_pair_cosine": float(np.max(np.abs(upper))),
        "a0_q90_minus_q10": float(np.quantile(a0, 0.90) - np.quantile(a0, 0.10)),
    }


def binomial_tail(n: int, p: float, minimum: int = 32) -> float:
    return float(sum(
        math.comb(n, k) * p**k * (1 - p) ** (n - k)
        for k in range(minimum, n + 1)
    ))


def read_power() -> list[dict[str, str]]:
    with (REVIEW / "POWER_SIMULATION.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row(rows: list[dict[str, str]], k: int, n: int, rho: float) -> dict[str, str]:
    for value in rows:
        if int(value["K"]) == k and int(value["N"]) == n and float(value["target_rho"]) == rho:
            return value
    raise AssertionError((k, n, rho))


def main() -> None:
    manifest = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    safety = json.loads(SAFETY.read_text(encoding="utf-8"))
    historical_audit = json.loads(V4_AUDIT.read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    assert len(candidates) == 40
    assert safety["safe_count"] == 31
    assert safety["classification"] == "Q2_V4_SAFE_BANK_INSUFFICIENT"
    assert safety["correctness_used"] is False
    ids = [candidate["candidate_id"] for candidate in candidates]
    assert ids == [f"V4_DIRECTION_{i:02d}" for i in range(40)]
    safe_ids = [
        candidate_id for candidate_id in ids
        if bool(safety["candidates"][candidate_id]["both_shells_pass"])
    ]
    assert len(safe_ids) == 31
    assert set(safe_ids) == EXPECTED
    assert historical_audit["classification"] == "Q2_V4_PRESEMANTIC_FORENSIC_CLEAN"

    all_values = np.asarray(
        [candidate["coefficients"] for candidate in candidates], dtype=np.float64
    )
    safe_values = np.asarray([
        candidate["coefficients"] for candidate in candidates
        if candidate["candidate_id"] in EXPECTED
    ], dtype=np.float64)
    all_geometry = geometry(all_values)
    safe_geometry = geometry(safe_values)
    amplitudes = np.asarray([
        [
            safety["candidates"][candidate["candidate_id"]]["shells"]["MEDIUM"]["implemented_amplitude"],
            safety["candidates"][candidate["candidate_id"]]["shells"]["STRONG"]["implemented_amplitude"],
        ]
        for candidate in candidates if candidate["candidate_id"] in EXPECTED
    ])
    amplitude_cv = np.std(amplitudes, axis=0) / np.mean(amplitudes, axis=0)
    checks = {
        "selected_count_31": len(safe_values) == 31,
        "full_subspace_rank": safe_geometry["rank"] == 8,
        "entropy_effective_rank_at_least_0_75r": safe_geometry["effective_rank"] >= 6.0,
        "condition_number_at_most_3": safe_geometry["condition_number"] <= 3.0,
        "max_absolute_pair_cosine_below_0_98": safe_geometry["max_absolute_pair_cosine"] < 0.98,
        "a0_q90_q10_at_least_0_20": safe_geometry["a0_q90_minus_q10"] >= 0.20,
        "shell_amplitude_cv_at_most_0_03": bool(np.max(amplitude_cv) <= 0.03),
    }

    power_rows = read_power()
    k31 = row(power_rows, 31, 300, 0.25)
    k32 = row(power_rows, 32, 300, 0.25)
    null31 = row(power_rows, 31, 300, 0.0)
    power_checks = {
        "coverage_all_pass": all(checks.values()),
        "k31_omnibus_power_at_least_0_80": float(k31["omnibus_rate"]) >= 0.80,
        "k31_fpr_in_prespecified_range": 0.025 <= float(null31["omnibus_rate"]) <= 0.075,
        "absolute_omnibus_power_loss_vs_k32_at_most_0_10": (
            float(k32["omnibus_rate"]) - float(k31["omnibus_rate"]) <= 0.10
        ),
        "ci_width_ratio_vs_k32_at_most_1_10": (
            float(k31["a2_rho_mc95_width"]) / float(k32["a2_rho_mc95_width"]) <= 1.10
        ),
    }
    decision = (
        "Q2_V4_1_31_SAFE_BANK_ADEQUATE"
        if all(power_checks.values())
        else "Q2_V4_1_31_SAFE_BANK_INADEQUATE"
    )
    primary = json.loads((REVIEW / "ADEQUACY_DECISION.json").read_text(encoding="utf-8"))
    assert primary["decision"] == decision
    reserve_rows = []
    for probability in (0.70, 0.75, 0.775, 0.80, 0.85, 0.90):
        tail = binomial_tail(40, probability)
        for count in range(32, 201):
            if binomial_tail(count, probability) >= 0.95:
                needed = count
                break
        reserve_rows.append({
            "p_safe": probability,
            "probability_at_least_minimum": tail,
            "minimum_candidates_for_95pct": needed,
        })
    persisted_reserve = json.loads((REVIEW / "RESERVE_FRAGILITY.json").read_text(encoding="utf-8"))
    assert persisted_reserve["rows"] == [
        {"candidate_count": 40, **value} for value in reserve_rows
    ]

    crosscheck_rows = []
    for name, left, right in (
        ("rank", all_geometry["rank"], json.loads(
            (REVIEW / "SAFETY_ATTRITION_GEOMETRY.json").read_text(encoding="utf-8")
        )["all_40"]["rank"]),
        ("safe_rank", safe_geometry["rank"], json.loads(
            (REVIEW / "SAFETY_ATTRITION_GEOMETRY.json").read_text(encoding="utf-8")
        )["safe_31"]["rank"]),
        ("safe_effective_rank", safe_geometry["effective_rank"], json.loads(
            (REVIEW / "SAFETY_ATTRITION_GEOMETRY.json").read_text(encoding="utf-8")
        )["safe_31"]["effective_rank"]),
        ("safe_condition_number", safe_geometry["condition_number"], json.loads(
            (REVIEW / "SAFETY_ATTRITION_GEOMETRY.json").read_text(encoding="utf-8")
        )["safe_31"]["condition_number"]),
    ):
        crosscheck_rows.append({
            "metric": name,
            "independent": left,
            "primary": right,
            "absolute_difference": abs(left - right),
        })
    with (REVIEW / "FORENSIC_METRIC_CROSSCHECK.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(crosscheck_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(crosscheck_rows)
    result = {
        "classification": "Q2_V4_1_FORENSIC_CLEAN",
        "decision_recomputed": decision,
        "candidate_count": 40,
        "safe_count": 31,
        "safe_ids_match_original_order_set": True,
        "coverage_checks_recomputed": checks,
        "power_checks_recomputed": power_checks,
        "max_primary_metric_difference": float(max(
            value["absolute_difference"] for value in crosscheck_rows
        )),
        "reserve_tail_recomputed": reserve_rows,
        "historical_v4_classification_unchanged": safety["classification"],
        "historical_v4_forensic_classification": historical_audit["classification"],
        "semantic_outcomes": 0,
        "correctness_inspected": False,
        "model_inference": False,
        "gpu_used": False,
        "new_candidates_generated": 0,
        "independent_method": "direct JSON/CSV parsing and standalone NumPy/math recomputation",
        "source_hashes": {
            "candidate_manifest": sha256(CANDIDATES),
            "safety_report": sha256(SAFETY),
            "historical_audit": sha256(V4_AUDIT),
        },
    }
    (REVIEW / "FORENSIC_AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (REVIEW / "FORENSIC_AUDIT.md").write_text(
        "# Q2 V4.1 independent forensic audit\n\n"
        "Classification: Q2_V4_1_FORENSIC_CLEAN.\n\n"
        "A separate implementation re-read only the immutable V4 candidate and "
        "safety artifacts plus the persisted synthetic planning CSV. It "
        "recomputed candidate/safe counts, order/set identity, coefficient "
        "geometry, amplitude CV, decision criteria, power comparison, and "
        "reserve tails. The maximum crosscheck difference against the primary "
        f"geometry values was {result['max_primary_metric_difference']:.3g}. "
        "No model output, semantic correctness, GPU, A1/A2 outcome, or "
        "semantic panel was accessed. The historical V4 classification remains "
        "Q2_V4_SAFE_BANK_INSUFFICIENT.\n",
        encoding="utf-8",
    )
    if decision == "Q2_V4_1_31_SAFE_BANK_ADEQUATE":
        stale = REVIEW / "FUTURE_RESERVE_RECOMMENDATION.json"
        if stale.exists():
            stale.unlink()
    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(REVIEW.iterdir(), key=lambda item: item.name)
        if path.is_file()
        and path.name not in {
            "artifact_hashes.json",
            "MANIFEST_AND_HASHES.json",
            "Q2_V4_1_REVIEW_BUNDLE.tar.gz",
            "BUNDLE_SHA256.txt",
        }
    }
    (REVIEW / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (REVIEW / "MANIFEST_AND_HASHES.json").write_text(
        json.dumps(
            {
                "review_artifact_hash": hashlib.sha256(
                    json.dumps(
                        artifact_hashes, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "historical_candidate_manifest_sha256": sha256(CANDIDATES),
                "historical_safety_report_sha256": sha256(SAFETY),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
