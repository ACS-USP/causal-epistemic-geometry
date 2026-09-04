#!/usr/bin/env python3
# ruff: noqa: E501
"""Materialize the single Q3 candidate system from closed development data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_q3_prompt_representation as q31  # noqa: E402
import design_q3_realizable_utility as q3  # noqa: E402

REVIEW = ROOT / "review/q3_final_system_and_evaluation_supply"
PRECHECK = REVIEW / "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK.json"
STEER = REVIEW / "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK_ADDITIVE_STEER.json"
PANEL = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"
Q31 = (
    ROOT
    / "review/q3_route_a_prompt_representation/Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json"
)
HIST = ROOT / "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
FRESH = (
    ROOT
    / "review/q2_oos_fresh_controller_design/v2_final_presemantic/V2_CANDIDATE_BANK_MANIFEST.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fit_pca_persist(raw: np.ndarray, dimension: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mean = raw.mean(axis=0)
    centered = raw - mean
    gram = centered @ centered.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    keep = min(dimension, int(np.sum(values > 1e-10)))
    components = (vectors[:, :keep].T @ centered) / np.sqrt(values[:keep, None])
    scores = centered @ components.T
    if keep < dimension:
        components = np.pad(components, ((0, dimension - keep), (0, 0)))
        scores = np.pad(scores, ((0, 0), (0, dimension - keep)))
    scale = scores.std(axis=0)
    scale[scale < 1e-8] = 1.0
    transformed = scores / scale
    reference, _ = q31.fit_pca(raw, raw[:1], dimension)
    if not np.allclose(transformed, reference, atol=1e-12, rtol=1e-12):
        raise RuntimeError("persisted PCA does not reproduce frozen transformation")
    return transformed, {"mean": mean, "components": components, "scale": scale}


def vector_records() -> dict[str, dict[str, Any]]:
    records = {}
    for row in read_json(HIST)["directions"]:
        records[row["candidate_id"]] = {
            "vector_sha256": row["canonical_vector_hash"],
            "vector_file_sha256": row["file_sha256"],
            "source": str(HIST.relative_to(ROOT)),
        }
    for row in read_json(FRESH)["candidates"]:
        records[row["candidate_id"]] = {
            "vector_sha256": row["vector_array_sha256"],
            "vector_file_sha256": row["file_sha256"],
            "source": str(FRESH.relative_to(ROOT)),
        }
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--fresh-scores", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()

    precheck = read_json(PRECHECK)
    steer = read_json(STEER)
    if precheck["status"] != "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK_FROZEN":
        raise RuntimeError("Q3.3 base precheck is not frozen")
    if steer["base_precheck_sha256"] != sha256_file(PRECHECK):
        raise RuntimeError("Q3.3 additive steer does not bind the base precheck")
    q3.verify_inputs(args.historical_scores, args.fresh_scores)
    expected_representation = "3612a645e3739e3cf7bf4d32f1f808034b15604a1e7f99e784c45e04b49d81ac"
    if sha256_file(args.representations) != expected_representation:
        raise RuntimeError("prompt-representation hash mismatch")

    q31_summary = read_json(Q31)
    fold_hp = [
        q31_summary["primary_bank"]["hyperparameters"][f"fold_{fold}_GEOMETRY_BLIND_POLICY_ID"]
        for fold in range(5)
    ]
    consensus = {
        "dimension": int(np.median([row["dimension"] for row in fold_hp])),
        "rank": int(np.median([row["rank"] for row in fold_hp])),
        "l2": float(10 ** np.median(np.log10([row["l2"] for row in fold_hp]))),
    }
    frozen = precheck["final_system_materialization"]["router"]
    expected_consensus = {
        "dimension": frozen["pca_dimension"],
        "rank": frozen["interaction_rank"],
        "l2": frozen["l2"],
    }
    if consensus != expected_consensus:
        raise RuntimeError(f"hyperparameter consensus mismatch: {consensus}")

    item_ids = list(read_json(PANEL)["item_ids"])
    data, source_counts = q3.load_outcomes(args.historical_scores, args.fresh_scores, item_ids)
    controllers, _ = q3.load_controller_coordinates()
    geometry = q3.combined_geometry()
    all_items = np.arange(len(item_ids))
    shell_policies = q3.choose_shells(controllers, data, all_items)
    bank = q3.select_bank(
        "A0_MAXIMIN", 8, False, controllers, shell_policies, geometry, data, all_items
    )
    candidate_policies = sorted(p for p in data if p != "BASELINE")
    champion = q3.global_champion({p: data[p] for p in candidate_policies}, all_items)

    raw = np.load(args.representations, allow_pickle=False).astype(np.float64)
    if raw.shape != (300, 4096) or not np.isfinite(raw).all():
        raise RuntimeError("invalid prompt-representation matrix")
    x, pca = fit_pca_persist(raw, consensus["dimension"])
    y = np.stack([data[policy]["correct"].mean(axis=1) for policy in bank], axis=1)
    c = np.zeros((len(bank), 8), dtype=np.float64)
    fitted = q31.fit_low_rank_logistic(
        x,
        c,
        y,
        consensus["rank"],
        consensus["l2"],
        "BLIND",
        frozen["initialization_seed"],
        frozen["steps"],
        frozen["learning_rate"],
    )
    probabilities = q31.predict_probabilities(x, c, fitted, "BLIND")
    if probabilities.shape != (300, 8) or not np.isfinite(probabilities).all():
        raise RuntimeError("invalid fitted final router")

    args.private_output.mkdir(parents=True, exist_ok=True)
    private_npz = args.private_output / "Q3_FINAL_ROUTER_PARAMETERS_PRIVATE.npz"
    np.savez(
        private_npz,
        pca_mean=pca["mean"].astype(np.float32),
        pca_components=pca["components"].astype(np.float32),
        pca_scale=pca["scale"].astype(np.float32),
        router_u=fitted["u"].astype(np.float32),
        router_v=fitted["v"].astype(np.float32),
        router_a=fitted["a"].astype(np.float32),
        router_b=fitted["b"].astype(np.float32),
    )
    private_manifest = {
        "artifact": private_npz.name,
        "sha256": sha256_file(private_npz),
        "bytes": private_npz.stat().st_size,
        "tracked_in_git": False,
        "reason": "derived prompt-activation and closed-outcome router parameters are private/hash-pinned",
    }
    write_json(args.private_output / "Q3_FINAL_ROUTER_PRIVATE_MANIFEST.json", private_manifest)

    vectors = vector_records()
    policy_records = []
    for order, policy in enumerate(bank):
        controller, shell = policy.rsplit("_", 1)
        policy_records.append(
            {
                "order": order,
                "policy_id": policy,
                "controller_id": controller,
                "shell": shell,
                **vectors[controller],
            }
        )
    champion_controller, champion_shell = champion.rsplit("_", 1)
    draft = {
        "schema_version": "q3-final-candidate-system-draft-v1",
        "status": "DEVELOPMENT_SELECTED_NOT_EVALUATED",
        "development_phase_closed": True,
        "evidence_class": "DEVELOPMENT_ONLY",
        "source_counts": source_counts,
        "development_families": len(item_ids),
        "portfolio": {
            "method": "A0_MAXIMIN",
            "size": len(bank),
            "policies": policy_records,
            "coordinates_used_for_construction_only": True,
        },
        "representation": {
            "source": "ordinary unsteered layer-27 block input at final non-padding prompt token",
            "input_dimension": 4096,
            "pca_dimension": consensus["dimension"],
            "preprocessing": "subtract frozen full-development mean; project onto frozen dual-SVD components; divide by frozen full-development score SD with <1e-8 mapped to 1",
            "source_matrix_sha256": expected_representation,
        },
        "router": {
            "family": frozen["family"],
            "mode": frozen["mode"],
            "controller_coordinates_as_input": False,
            "interaction_rank": consensus["rank"],
            "l2": consensus["l2"],
            "optimizer": frozen["optimizer"],
            "steps": frozen["steps"],
            "learning_rate": frozen["learning_rate"],
            "initialization_seed": frozen["initialization_seed"],
            "target": frozen["targets"],
            "calibration": "native sigmoid scores; no post-hoc recalibration selected",
            "selection": "argmax predicted policy correctness; ties resolved by frozen policy order",
            "fallback": champion,
            "private_parameter_manifest": private_manifest,
        },
        "champion": {
            "policy_id": champion,
            "controller_id": champion_controller,
            "shell": champion_shell,
            **vectors[champion_controller],
            "selection_rule": precheck["final_system_materialization"]["champion"]["rule"],
        },
        "deployment": {
            "answer_generations": 1,
            "same_forward": True,
            "pseudocode": [
                "run unsteered prefill through layer 27 block input",
                "read final non-padding prompt-token representation",
                "apply frozen PCA and frozen learned-policy-identity router",
                "select argmax policy, with frozen champion fallback on non-finite router state",
                "apply selected policy's frozen sustained-current-token intervention from that same prefill point",
                "decode exactly one answer with the frozen generation contract",
            ],
        },
        "upper_bound_clarification": read_json(STEER)["upper_bound_clarification"]
        if "upper_bound_clarification" in read_json(STEER)
        else {
            "q3_2_outcome_optimized_bank": "bank-opportunity upper bound only",
            "not_routed_accuracy_upper_bound": True,
        },
        "future_data_may_change_system": False,
        "firewall": {
            "new_semantic_trajectories": 0,
            "new_qwen_forwards": 0,
            "fresh_outcomes_inspected": False,
        },
    }
    public_path = REVIEW / "FINAL_Q3_CANDIDATE_SYSTEM_DRAFT.json"
    write_json(public_path, draft)
    print(
        json.dumps(
            {
                "bank": bank,
                "champion": champion,
                "draft_sha256": sha256_file(public_path),
                "private_sha256": private_manifest["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
