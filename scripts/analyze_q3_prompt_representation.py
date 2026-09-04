#!/usr/bin/env python3
"""Run the frozen Q3.1 nested development tournament on captured prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import design_q3_realizable_utility as q3  # noqa: E402

REVIEW = ROOT / "review/q3_route_a_prompt_representation"
PRECHECK = REVIEW / "Q3_ROUTE_A_PROMPT_REPRESENTATION_PRECHECK.json"
PANEL = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"
Q3_RESULTS = ROOT / "review/q3_realizable_utility_design/CROSS_FIT_RESULTS.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fit_pca(train: np.ndarray, test: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Dual-SVD PCA fit only on training rows, followed by train-only scaling."""

    mean = train.mean(axis=0)
    centered = train - mean
    gram = centered @ centered.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    keep = min(dimension, int(np.sum(values > 1e-10)))
    components = (vectors[:, :keep].T @ centered) / np.sqrt(values[:keep, None])
    train_scores = centered @ components.T
    test_scores = (test - mean) @ components.T
    if keep < dimension:
        train_scores = np.pad(train_scores, ((0, 0), (0, dimension - keep)))
        test_scores = np.pad(test_scores, ((0, 0), (0, dimension - keep)))
    scale = train_scores.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return train_scores / scale, test_scores / scale


def normalize_coordinates(c: np.ndarray) -> np.ndarray:
    c = c.copy().astype(np.float64)
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    c /= np.maximum(norms, 1e-12)
    mean = c.mean(axis=0, keepdims=True)
    std = c.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (c - mean) / std


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def fit_low_rank_logistic(
    x: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    rank: int,
    l2: float,
    mode: str,
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    k = y.shape[1]
    u = rng.normal(scale=0.02, size=(d, rank))
    v = rng.normal(scale=0.02, size=((k if mode == "BLIND" else c.shape[1]), rank))
    a = np.zeros(d)
    b = np.zeros(k)
    params = [u, v, a, b]
    first = [np.zeros_like(value) for value in params]
    second = [np.zeros_like(value) for value in params]
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        qx = x @ u
        qk = v if mode == "BLIND" else c @ v
        scores = x @ a[:, None] + b[None, :] + qx @ qk.T
        error = (sigmoid(scores) - y) / (n * k)
        grad_a = x.T @ error.sum(axis=1)
        grad_b = error.sum(axis=0)
        grad_u = x.T @ (error @ qk) + l2 * u / (n * k)
        if mode == "BLIND":
            grad_v = error.T @ qx + l2 * v / (n * k)
        else:
            grad_v = c.T @ (error.T @ qx) + l2 * v / (n * k)
        gradients = [grad_u, grad_v, grad_a + l2 * a / (n * k), grad_b]
        for index, (parameter, gradient) in enumerate(zip(params, gradients, strict=True)):
            first[index] = beta1 * first[index] + (1 - beta1) * gradient
            second[index] = beta2 * second[index] + (1 - beta2) * gradient * gradient
            corrected_first = first[index] / (1 - beta1**step)
            corrected_second = second[index] / (1 - beta2**step)
            parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return {"u": u, "v": v, "a": a, "b": b}


def predict_probabilities(
    x: np.ndarray, c: np.ndarray, fitted: dict[str, np.ndarray], mode: str
) -> np.ndarray:
    qx = x @ fitted["u"]
    qk = fitted["v"] if mode == "BLIND" else c @ fitted["v"]
    return sigmoid(x @ fitted["a"][:, None] + fitted["b"][None, :] + qx @ qk.T)


def calibration(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    brier = float(np.mean((clipped - y) ** 2))
    log_loss = float(np.mean(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))))
    bins = np.minimum((clipped * 10).astype(int), 9)
    ece = 0.0
    for value in range(10):
        mask = bins == value
        if mask.any():
            ece += float(mask.mean() * abs(clipped[mask].mean() - y[mask].mean()))
    return {"brier": brier, "log_loss": log_loss, "ece_10": ece}


def permuted_coordinates(
    bank: list[str], controllers: list[str], coords: dict[str, np.ndarray], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(controllers))
    mapping = {
        controller: controllers[int(permutation[i])] for i, controller in enumerate(controllers)
    }
    values = [coords[mapping[policy.rsplit("_", 1)[0]]] for policy in bank]
    return normalize_coordinates(np.stack(values))


def true_coordinates(bank: list[str], coords: dict[str, np.ndarray]) -> np.ndarray:
    return normalize_coordinates(np.stack([coords[p.rsplit("_", 1)[0]] for p in bank]))


def choose_hyperparameters(
    raw: np.ndarray,
    y: np.ndarray,
    item_ids: list[str],
    c: np.ndarray,
    mode: str,
    model_seed: int,
    precheck: dict[str, Any],
) -> dict[str, Any]:
    inner = q3.balanced_hash_folds(item_ids, 4, f"q3-route-a-representation-{model_seed}")
    candidates = []
    for dimension in precheck["preprocessing"]["representation_dimensions"]:
        pca_cache = {}
        for fold in range(4):
            train = inner != fold
            valid = inner == fold
            pca_cache[fold] = fit_pca(raw[train], raw[valid], dimension)
        for rank in precheck["models"]["interaction_ranks"]:
            for l2 in precheck["models"]["l2_grid"]:
                correct = []
                for fold in range(4):
                    train = inner != fold
                    valid = inner == fold
                    tx, vx = pca_cache[fold]
                    fitted = fit_low_rank_logistic(
                        tx,
                        c,
                        y[train],
                        rank,
                        l2,
                        mode,
                        model_seed + 1009 * fold + 17 * dimension + rank,
                        precheck["models"]["optimizer"]["steps"],
                        precheck["models"]["optimizer"]["learning_rate"],
                    )
                    probabilities = predict_probabilities(vx, c, fitted, mode)
                    chosen = probabilities.argmax(axis=1)
                    correct.extend(y[valid, chosen].tolist())
                score = float(np.mean(correct))
                candidates.append((score, -dimension, -rank, l2, dimension, rank, l2))
    best = max(candidates)
    return {"inner_accuracy": best[0], "dimension": best[4], "rank": best[5], "l2": best[6]}


def evaluate_bank(
    bank_method: str,
    raw: np.ndarray,
    data: dict[str, dict[str, np.ndarray]],
    item_ids: list[str],
    controllers: list[str],
    coords: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    precheck: dict[str, Any],
    primary_hyperparameters: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    outer = q3.balanced_hash_folds(item_ids, 5, "q3-realizable-utility-v1")
    models = {
        "TRUE_GEOMETRY": "GEOMETRY",
        "GEOMETRY_BLIND_POLICY_ID": "BLIND",
        "PERMUTED_COORDINATES": "GEOMETRY",
    }
    accumulators = {
        name: {
            "routed": [],
            "champion": [],
            "oracle": [],
            "valid": [],
            "evaluable": [],
            "prob": [],
            "target": [],
            "folds": [],
            "chosen": [],
        }
        for name in models
    }
    hyperparameters: dict[tuple[int, str], dict[str, Any]] = {}
    fold_banks = []
    for fold in range(5):
        train = np.flatnonzero(outer != fold)
        test = np.flatnonzero(outer == fold)
        shell_policies = q3.choose_shells(controllers, data, train)
        bank = q3.select_bank(
            f"{bank_method}_MAXIMIN", 8, False, controllers, shell_policies, geometry, data, train
        )
        fold_banks.append(bank)
        champion = q3.global_champion(data, train)
        y_train = np.stack([data[p]["correct"][train].mean(axis=1) for p in bank], axis=1)
        y_test = np.stack([data[p]["correct"][test].mean(axis=1) for p in bank], axis=1)
        c_true = true_coordinates(bank, coords)
        c_permuted = permuted_coordinates(
            bank, controllers, coords, precheck["models"]["coordinate_permutation_seed"]
        )
        for model_name, mode in models.items():
            c = c_permuted if model_name == "PERMUTED_COORDINATES" else c_true
            if primary_hyperparameters is None:
                selected = choose_hyperparameters(
                    raw[train],
                    y_train,
                    [item_ids[i] for i in train],
                    c,
                    mode,
                    precheck["models"]["optimizer"]["initialization_seed"]
                    + 10000 * fold
                    + len(model_name),
                    precheck,
                )
            else:
                selected = dict(primary_hyperparameters[(fold, model_name)])
            hyperparameters[(fold, model_name)] = selected
            tx, vx = fit_pca(raw[train], raw[test], selected["dimension"])
            fitted = fit_low_rank_logistic(
                tx,
                c,
                y_train,
                selected["rank"],
                selected["l2"],
                mode,
                precheck["models"]["optimizer"]["initialization_seed"]
                + 20000 * fold
                + len(model_name),
                precheck["models"]["optimizer"]["steps"],
                precheck["models"]["optimizer"]["learning_rate"],
            )
            probabilities = predict_probabilities(vx, c, fitted, mode)
            chosen = probabilities.argmax(axis=1)
            routed = y_test[np.arange(len(test)), chosen]
            champion_values = data[champion]["correct"][test].mean(axis=1)
            oracle = y_test.max(axis=1)
            selected_probability = probabilities[np.arange(len(test)), chosen]
            valid = np.stack([data[p]["valid"][test].mean(axis=1) for p in bank], axis=1)[
                np.arange(len(test)), chosen
            ]
            evaluable = np.stack([data[p]["evaluable"][test].mean(axis=1) for p in bank], axis=1)[
                np.arange(len(test)), chosen
            ]
            acc = accumulators[model_name]
            acc["routed"].extend(routed.tolist())
            acc["champion"].extend(champion_values.tolist())
            acc["oracle"].extend(oracle.tolist())
            acc["valid"].extend(valid.tolist())
            acc["evaluable"].extend(evaluable.tolist())
            acc["prob"].extend(selected_probability.tolist())
            acc["target"].extend(routed.tolist())
            acc["folds"].append(float((routed - champion_values).mean()))
            acc["chosen"].extend([bank[int(index)] for index in chosen])
    results = {}
    for name, values in accumulators.items():
        routed = np.asarray(values["routed"])
        champion = np.asarray(values["champion"])
        oracle = np.asarray(values["oracle"])
        gain = routed - champion
        headroom = float((oracle - champion).mean())
        counts = Counter(values["chosen"])
        results[name] = {
            "routed_accuracy": float(routed.mean()),
            "champion_accuracy": float(champion.mean()),
            "absolute_gain": float(gain.mean()),
            "oracle_headroom": headroom,
            "fraction_oracle_headroom_realized": float(gain.mean() / headroom),
            "positive_outer_folds": int(sum(value > 0 for value in values["folds"])),
            "worst_outer_fold_gain": float(min(values["folds"])),
            "outer_fold_gains": values["folds"],
            "fold_gain_standard_deviation": float(np.std(values["folds"], ddof=1)),
            "commitment_validity": float(np.mean(values["valid"])),
            "semantic_evaluability": float(np.mean(values["evaluable"])),
            "calibration": calibration(np.asarray(values["target"]), np.asarray(values["prob"])),
            "policy_selection_counts": dict(sorted(counts.items())),
            "maximum_single_policy_selection_share": max(counts.values()) / len(item_ids),
            "distinct_selected_policies": len(counts),
            "itemwise_routed_correctness": routed.tolist(),
            "itemwise_champion_correctness": champion.tolist(),
        }
    return {
        "bank_method": bank_method,
        "fold_banks": fold_banks,
        "models": results,
        "hyperparameters": {
            f"fold_{fold}_{model}": value for (fold, model), value in hyperparameters.items()
        },
    }


def assess(primary: dict[str, Any], precheck: dict[str, Any]) -> tuple[dict[str, Any], str]:
    true = primary["models"]["TRUE_GEOMETRY"]
    blind = primary["models"]["GEOMETRY_BLIND_POLICY_ID"]
    permuted = primary["models"]["PERMUTED_COORDINATES"]
    gates = precheck["feasibility_gates"]
    realization = {
        "absolute_gain": true["absolute_gain"] >= gates["absolute_routed_gain_min"],
        "oracle_fraction": true["fraction_oracle_headroom_realized"]
        >= gates["oracle_headroom_fraction_min"],
        "positive_folds": true["positive_outer_folds"] >= gates["positive_outer_folds_min"],
        "worst_fold": true["worst_outer_fold_gain"] >= gates["worst_fold_gain_min"],
        "selection_concentration": true["maximum_single_policy_selection_share"]
        <= gates["maximum_overall_single_policy_selection_share"],
        "selection_diversity": true["distinct_selected_policies"]
        >= gates["minimum_distinct_selected_policies"],
    }
    true_fold = np.asarray(true["outer_fold_gains"])
    blind_fold = np.asarray(blind["outer_fold_gains"])
    perm_fold = np.asarray(permuted["outer_fold_gains"])
    true_minus_blind = true["absolute_gain"] - blind["absolute_gain"]
    true_minus_permuted = true["absolute_gain"] - permuted["absolute_gain"]
    incremental = {
        "true_minus_blind_gain": true_minus_blind,
        "true_minus_permuted_gain": true_minus_permuted,
        "true_over_blind_threshold": true_minus_blind
        >= gates["incremental_true_geometry_over_blind_gain_min"],
        "true_over_permuted_threshold": true_minus_permuted
        >= gates["incremental_true_geometry_over_permuted_gain_min"],
        "true_over_blind_nonnegative_folds": int(np.sum(true_fold - blind_fold >= 0)),
        "true_over_permuted_nonnegative_folds": int(np.sum(true_fold - perm_fold >= 0)),
    }
    incremental["fold_consistency"] = (
        incremental["true_over_blind_nonnegative_folds"]
        >= gates["incremental_nonnegative_outer_folds_min"]
        and incremental["true_over_permuted_nonnegative_folds"]
        >= gates["incremental_nonnegative_outer_folds_min"]
    )
    if not all(realization.values()):
        ruling = "Q3_ROUTE_A_REPRESENTATION_UNSTABLE"
    elif not all(
        incremental[key]
        for key in ("true_over_blind_threshold", "true_over_permuted_threshold", "fold_consistency")
    ):
        ruling = "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL"
    else:
        ruling = "Q3_ROUTE_A_REPRESENTATION_GEOMETRY_READY_FOR_HOLDOUT_DESIGN"
    return {"realization": realization, "incremental_geometry": incremental}, ruling


def q3_0_structure_control() -> dict[str, Any]:
    rows = read_json(Q3_RESULTS)["route_a_results"]
    matches = [
        row
        for row in rows
        if row["bank_method"] == "A0_MAXIMIN"
        and row["K"] == 8
        and row["baseline_included"] is False
        and row["mechanism"] == "GEOMETRY_BILINEAR"
    ]
    if len(matches) != 1:
        raise RuntimeError("Q3.0 deterministic structure control not unique")
    row = matches[0]
    return {
        key: row[key]
        for key in (
            "routed_accuracy",
            "development_selected_champion_accuracy",
            "absolute_gain",
            "fraction_oracle_headroom_realized",
            "positive_outer_folds",
            "worst_outer_fold_gain",
            "outer_fold_gains",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--capture-result", type=Path, required=True)
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--fresh-scores", type=Path, required=True)
    args = parser.parse_args()
    precheck = read_json(PRECHECK)
    if (
        sha256_file(Path(__file__).resolve())
        != precheck["implementation"]["analysis_runner_sha256"]
    ):
        raise RuntimeError("Q3.1 analysis runner hash mismatch")
    capture = read_json(args.capture_result)
    if capture.get("status") != "Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_COMPLETE":
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID")
    if sha256_file(args.representations) != capture["matrix_sha256"]:
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: matrix hash")
    if (
        capture["site_equivalence_max_abs_difference"]
        > precheck["capture"]["site_equivalence_max_abs_tolerance"]
    ):
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: site equivalence")
    if not capture["single_forward_hook_mechanics"]["passed"]:
        raise RuntimeError("Q3_ROUTE_A_SINGLE_FORWARD_DEPLOYMENT_INFEASIBLE")
    raw = np.load(args.representations, allow_pickle=False).astype(np.float64)
    if raw.shape != (300, 4096) or not np.isfinite(raw).all():
        raise RuntimeError("Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID: matrix")
    q3.verify_inputs(args.historical_scores, args.fresh_scores)
    panel = read_json(PANEL)
    item_ids = list(panel["item_ids"])
    data, source_counts = q3.load_outcomes(args.historical_scores, args.fresh_scores, item_ids)
    controllers, coords = q3.load_controller_coordinates()
    geometry = q3.combined_geometry()
    primary = evaluate_bank("A0", raw, data, item_ids, controllers, coords, geometry, precheck)
    hyper = {
        (fold, model): primary["hyperparameters"][f"fold_{fold}_{model}"]
        for fold in range(5)
        for model in ("TRUE_GEOMETRY", "GEOMETRY_BLIND_POLICY_ID", "PERMUTED_COORDINATES")
    }
    secondary = {
        name: evaluate_bank(
            name, raw, data, item_ids, controllers, coords, geometry, precheck, hyper
        )
        for name in ("A1", "A2")
    }
    gate_results, ruling = assess(primary, precheck)
    result = {
        "schema_version": "q3-route-a-prompt-representation-analysis-v1",
        "status": ruling,
        "evidence_class": ["DEVELOPMENT_ONLY", "POST_CLOSED_RESULT_PLANNING"],
        "primary_bank": primary,
        "secondary_banks_cannot_rescue_primary": secondary,
        "q3_0_deterministic_prompt_structure_control": q3_0_structure_control(),
        "gate_results": gate_results,
        "capture": {
            "matrix_sha256": capture["matrix_sha256"],
            "prompt_only_forward_count": capture["prompt_only_forward_count"],
            "representation_site": capture["representation_site"],
            "single_forward_deployment_feasible": capture["single_forward_hook_mechanics"][
                "passed"
            ],
        },
        "source_counts": source_counts,
        "fresh_holdout_outcomes_inspected": False,
        "new_semantic_trajectories": 0,
        "q3_confirmatory_experiment": "NOT_RUN",
    }
    output = REVIEW / "Q3_ROUTE_A_PROMPT_REPRESENTATION_RESULTS.json"
    write_json(output, result)
    print(
        json.dumps(
            {
                "status": ruling,
                "primary": primary["models"]["TRUE_GEOMETRY"],
                "incremental": gate_results["incremental_geometry"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
