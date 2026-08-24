#!/usr/bin/env python3
"""Build the final V2 bank lock after label-free dose calibration."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_bank_v2"
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.experiments.q2_controller_heldout_v2 import (  # noqa: E402
    BASELINE,
    COMMON_PANEL_N,
    EXECUTION_TEACHER_TEXT,
    EXPERIMENT_ID,
    LAYER,
    NULL_COUNT,
    bank_qualification,
    build_null_bank,
    canonical_controller_split,
    common_schedule,
    stable_digest,
    stable_seed,
    validate_null_bank,
    validate_schedule,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def main() -> int:
    source_lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    read_json(REVIEW / "V2_DOSE_CALIBRATION_LOCK.json")
    calibration = read_json(REVIEW / "V2_DOSE_CALIBRATION.json")
    if calibration["status"] != "COMPLETE_LABEL_FREE_CALIBRATION":
        raise RuntimeError("V2 dose calibration is not complete")
    if source_lock["instrument"]["v1_common_panel_outcomes_read"] is not False:
        raise RuntimeError("V1 common-panel outcome firewall changed")

    all_records = calibration["controllers"]
    selected: dict[str, dict[str, Any]] = {}
    for controller, record in sorted(all_records.items()):
        dose = record["selected_dose"]
        if dose is None:
            continue
        dose_record = record["doses"][dose]
        selected[controller] = {
            "controller": controller,
            "source_axis": record["source_axis"],
            "source_location": record["source_location"],
            "sign": record["sign"],
            "selected_dose": dose,
            "causal_pass": bool(record["causal_pass"]),
            "reference_scale": record["reference_scale"],
            "delta_norm": dose_record["delta_norm"],
            "vector_hash": read_json(REVIEW / "V2_SOURCE_DIRECTION_BANK.json")["directions"][
                controller
            ]["vector_hash"],
        }

    source_bank = read_json(REVIEW / "V2_SOURCE_DIRECTION_BANK.json")
    source_records = read_json(REVIEW / "V2_SOURCE_QUALIFICATION.json")
    meaningful_vectors: dict[str, np.ndarray] = {}
    for controller in selected:
        metadata = source_bank["directions"][controller]
        vector = np.load(ROOT / metadata["path"], allow_pickle=False).astype(np.float64)
        if vector_sha256(vector) != metadata["vector_hash"]:
            raise RuntimeError(f"meaningful vector hash mismatch: {controller}")
        meaningful_vectors[controller] = vector

    null_seeds = [stable_seed(EXPERIMENT_ID, "V2_FINAL_NULL", index) for index in range(NULL_COUNT)]
    nulls, null_metadata = build_null_bank(meaningful_vectors, null_seeds)
    null_checks = validate_null_bank(meaningful_vectors, nulls)
    null_dir = REVIEW / "V2_FINAL_NULL_VECTORS"
    null_dir.mkdir(parents=True, exist_ok=True)
    null_delta_norm = float(np.median([record["delta_norm"] for record in selected.values()]))
    null_records: dict[str, Any] = {}
    for index, (name, vector) in enumerate(sorted(nulls.items())):
        path = null_dir / f"{name}.npy"
        np.save(path, vector.astype(np.float64))
        null_records[name] = {
            "controller": name,
            "seed": null_seeds[index],
            "path": str(path.relative_to(ROOT)),
            "vector_hash": vector_sha256(vector),
            "delta_norm": null_delta_norm,
            "layer": LAYER,
            "duration": "sustained_current_token",
            "scope": "final_prompt_token_then_current_decode_token",
        }
    write_json(
        REVIEW / "V2_FINAL_RANDOM_BANK.json",
        {
            "controllers": null_records,
            "construction": null_metadata,
            "checks": null_checks,
            "source_meaningful_count": len(meaningful_vectors),
            "delta_norm_rule": "median selected meaningful delta norm",
            "correctness_used": False,
        },
    )

    qualification = bank_qualification(
        selected,
        source_records,
        null_checks,
    )
    write_json(REVIEW / "V2_BANK_QUALIFICATION.json", qualification)
    if qualification["classification"] != "Q2_V2_CONTROLLER_BANK_QUALIFIED":
        write_json(
            REVIEW / "V2_FINAL_BANK_STATUS.json",
            {
                "classification": qualification["classification"],
                "common_panel_run": False,
                "common_panel_rows": 0,
                "correctness_used": False,
            },
        )
        print(json.dumps(qualification, indent=2, sort_keys=True))
        return 0

    meaningful_ids = sorted(selected)
    null_ids = sorted(null_records)
    controller_ids = [*meaningful_ids, *null_ids]
    common_manifest = read_json(REVIEW / "V2_COMMON_PANEL_MANIFEST.json")
    item_ids = list(common_manifest["item_ids"])
    schedule = common_schedule(item_ids, controller_ids)
    expected_keys = [
        (item_id, condition, rollout)
        for item_id in item_ids
        for condition in [BASELINE, *controller_ids]
        for rollout in (0, 1)
    ]
    validate_schedule(schedule, expected_keys)
    write_json(REVIEW / "V2_COMMON_PANEL_SCHEDULE.json", schedule)

    family_split = canonical_controller_split(
        sorted({selected[name]["source_axis"] for name in meaningful_ids})
    )
    meaningful_metadata = {
        name: {
            **selected[name],
            "path": source_bank["directions"][name]["path"],
            "layer": LAYER,
            "duration": "sustained_current_token",
            "scope": "final_prompt_token_then_current_decode_token",
        }
        for name in meaningful_ids
    }
    final_lock = {
        "schema_version": "q2-controller-heldout-geometry-v2-final-lock-v1",
        "status": "FROZEN_PRE_COMMON_PANEL",
        "lifecycle": "PROSPECTIVE_LOCK",
        "experiment_id": EXPERIMENT_ID,
        "experiment_source_commit": git_head(),
        "source_lock_sha256": sha256(REVIEW / "PROTOCOL_LOCK.json"),
        "dose_lock_sha256": sha256(REVIEW / "V2_DOSE_CALIBRATION_LOCK.json"),
        "calibration_sha256": sha256(REVIEW / "V2_DOSE_CALIBRATION.json"),
        "model": source_lock["model"],
        "layer": LAYER,
        "meaningful_controllers": meaningful_metadata,
        "random_controllers": null_records,
        "controller_ids": controller_ids,
        "meaningful_count": len(meaningful_ids),
        "null_count": len(null_ids),
        "bank_qualification": qualification,
        "random_bank_checks": null_checks,
        "source_families": qualification["families"],
        "family_split": family_split,
        "common_panel": {
            "manifest": "V2_COMMON_PANEL_MANIFEST.json",
            "manifest_sha256": sha256(REVIEW / "V2_COMMON_PANEL_MANIFEST.json"),
            "schedule": "V2_COMMON_PANEL_SCHEDULE.json",
            "schedule_sha256": sha256(REVIEW / "V2_COMMON_PANEL_SCHEDULE.json"),
            "n": COMMON_PANEL_N,
            "conditions": len(controller_ids) + 1,
            "rollouts": 2,
            "expected_rows": len(schedule),
            "seed_regime": "INDEPENDENT_PRIMARY",
        },
        "geometry": {
            "M0": "normalized Euclidean/cosine on unit controller directions",
            "M1": {
                "type": "activation-covariance-whitened",
                "covariance_manifest": "V2_COVARIANCE_MANIFEST.json",
                "regularization": "Sigma_lambda=(1-lambda)Sigma+lambda*mean_variance*I",
                "lambda": 0.10,
                "construction_outcomes_forbidden": True,
            },
            "M2": {
                "type": "finite behavioral secant",
                "probe_manifest": "V2_FINITE_SECANT_MANIFEST.json",
                "teacher_forced_text": EXECUTION_TEACHER_TEXT,
                "checkpoints": ["first", "one_third", "two_thirds", "last"],
                "distribution": "full-vocabulary next-token JS",
                "exact_local_geometry_claimed": False,
            },
            "JVP_Fisher_pullback_forbidden": True,
        },
        "estimands": {
            "primary_population": "meaningful_controllers_only",
            "target": "D_ij=E[(p_i-p_j)^2] unbiased two-independent-rollout estimator",
            "secondary_population": "meaningful_plus_nulls",
            "bootstrap_unit": "item",
        },
        "prediction": {
            "family_split": "leave_one_source_family_out",
            "primary_score": "family-held-out Spearman",
            "secondary_score": "train-calibrated held-out RMSE",
            "qap_permutations": 10000,
            "qap_seed": 2026082402,
            "qap_unit": "controller-family-aware label permutation",
            "classification_thresholds": {
                "spearman_min": 0.30,
                "qap_one_sided_p_max": 0.05,
                "rmse_ratio_to_constant_max": 0.90,
            },
        },
        "correctness_used_for_bank": False,
        "common_panel_outcomes_read": False,
        "Q1": "IMMUTABLE",
        "Q3": "NOT RUN",
    }
    write_json(REVIEW / "V2_FINAL_PROTOCOL_LOCK.json", final_lock)
    write_json(
        REVIEW / "V2_FINAL_PROTOCOL_PROVENANCE.json",
        {
            "protocol_lock_sha256": sha256(REVIEW / "V2_FINAL_PROTOCOL_LOCK.json"),
            "common_schedule_sha256": sha256(REVIEW / "V2_COMMON_PANEL_SCHEDULE.json"),
            "bank_qualification_sha256": sha256(REVIEW / "V2_BANK_QUALIFICATION.json"),
            "controller_digest": stable_digest(EXPERIMENT_ID, "CONTROLLERS", *controller_ids),
            "meaningful_count": len(meaningful_ids),
            "null_count": len(null_ids),
            "common_panel_outcomes_read": False,
        },
    )
    (REVIEW / "V2_FINAL_PROTOCOL_LOCK.md").write_text(
        "# Q2 V2 final controller-bank and geometry lock\n\n"
        "Status: `FROZEN_PRE_COMMON_PANEL`.\n\n"
        f"The label-free calibration produced {len(meaningful_ids)} selected meaningful "
        f"controllers across {len(qualification['families'])} source families and "
        f"{len(null_ids)} fresh SVD-span-orthogonal nulls. The common-panel schedule "
        f"contains {len(schedule)} independent scientific rows. M0, M1, and M2, the "
        "family-held-out split, D estimator, and prediction procedure are frozen here. "
        "No correctness outcome was used for bank construction or selection. JVP, "
        "Fisher, pullback, manifold geometry, Q3, and holdout access remain forbidden.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "classification": "Q2_V2_CONTROLLER_BANK_QUALIFIED",
                "meaningful": len(meaningful_ids),
                "families": len(qualification["families"]),
                "causal": qualification["causal_direction_count"],
                "common_rows": len(schedule),
                "null_span_orthogonality": null_checks["span_orthogonality_max"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
