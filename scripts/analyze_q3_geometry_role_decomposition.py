#!/usr/bin/env python3
"""Closed-data Q3.2 geometry-role decomposition.

This development-only analysis reads scored closed outcomes and a private,
hash-pinned prompt-representation matrix. It performs no model inference and
does not read raw generation text.
"""

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

import analyze_q3_prompt_representation as q31  # noqa: E402
import design_q3_realizable_utility as q3  # noqa: E402

REVIEW = ROOT / "review/q3_geometry_role_decomposition"
PRECHECK = REVIEW / "Q3_GEOMETRY_ROLE_DECOMPOSITION_PRECHECK.json"
Q31_SUMMARY = (
    ROOT
    / "review/q3_route_a_prompt_representation"
    / "Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json"
)
PANEL = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"


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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def controller_from_policy(policy: str) -> str:
    return policy.rsplit("_", 1)[0]


def shell_from_policy(policy: str) -> str:
    return policy.rsplit("_", 1)[1]


def own_champion(bank: list[str], data: dict[str, dict[str, np.ndarray]], train: np.ndarray) -> str:
    return max(
        bank,
        key=lambda policy: (
            float(data[policy]["correct"][train].mean()),
            float(data[policy]["evaluable"][train].mean()),
            -float(data[policy]["tokens"][train].mean()),
            policy,
        ),
    )


def bank_features(
    bank: list[str],
    data: dict[str, dict[str, np.ndarray]],
    train: np.ndarray,
    geometry: np.ndarray,
    controller_order: list[str],
) -> np.ndarray:
    means = []
    for key in ("correct", "valid", "evaluable", "tokens"):
        means.append(float(np.mean([data[policy][key][train].mean() for policy in bank])))
    indices = [controller_order.index(controller_from_policy(policy)) for policy in bank]
    distances = [geometry[i, j] for offset, i in enumerate(indices) for j in indices[:offset]]
    return np.asarray([*means, float(np.mean(distances))], dtype=np.float64)


def candidate_controller_banks(
    controller_count: int, pool_size: int, fold: int, seed: int
) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed + 104729 * fold)
    seen: set[tuple[int, ...]] = set()
    banks: list[tuple[int, ...]] = []
    while len(seen) < pool_size:
        candidate = tuple(sorted(int(value) for value in rng.choice(controller_count, 8, False)))
        if candidate not in seen:
            seen.add(candidate)
            banks.append(candidate)
    return banks


def greedy_oracle_bank(
    policies: list[str], data: dict[str, dict[str, np.ndarray]], train: np.ndarray
) -> list[str]:
    selected: list[str] = []
    remaining = sorted(policies)
    while len(selected) < 8:
        scored = []
        for policy in remaining:
            bank = selected + [policy]
            correctness = np.stack([data[value]["correct"][train] for value in bank], axis=1)
            oracle = float(correctness.max(axis=1).mean())
            mean_accuracy = float(np.mean([data[value]["correct"][train].mean() for value in bank]))
            valid = float(np.mean([data[value]["valid"][train].mean() for value in bank]))
            tokens = float(np.mean([data[value]["tokens"][train].mean() for value in bank]))
            scored.append((oracle, mean_accuracy, valid, -tokens, policy))
        chosen = max(scored)[-1]
        selected.append(chosen)
        chosen_controller = controller_from_policy(chosen)
        remaining = [
            policy for policy in remaining if controller_from_policy(policy) != chosen_controller
        ]
    return selected


def fit_blind_bank_fold(
    raw: np.ndarray,
    data: dict[str, dict[str, np.ndarray]],
    item_ids: list[str],
    train: np.ndarray,
    test: np.ndarray,
    bank: list[str],
    hyperparameter: dict[str, Any],
    fold: int,
    precheck: dict[str, Any],
) -> dict[str, Any]:
    y_train = np.stack([data[policy]["correct"][train].mean(axis=1) for policy in bank], axis=1)
    y_test = np.stack([data[policy]["correct"][test].mean(axis=1) for policy in bank], axis=1)
    tx, vx = q31.fit_pca(raw[train], raw[test], int(hyperparameter["dimension"]))
    c = np.zeros((len(bank), 8), dtype=np.float64)
    model_name = "GEOMETRY_BLIND_POLICY_ID"
    fitted = q31.fit_low_rank_logistic(
        tx,
        c,
        y_train,
        int(hyperparameter["rank"]),
        float(hyperparameter["l2"]),
        "BLIND",
        int(precheck["part_a"]["router_initialization_seed"]) + 20000 * fold + len(model_name),
        int(precheck["router"]["optimizer_steps"]),
        float(precheck["router"]["learning_rate"]),
    )
    probability = q31.predict_probabilities(vx, c, fitted, "BLIND")
    chosen = probability.argmax(axis=1)
    routed = y_test[np.arange(len(test)), chosen]
    champion = own_champion(bank, data, train)
    champion_values = data[champion]["correct"][test].mean(axis=1)
    oracle = y_test.max(axis=1)
    valid = np.stack([data[policy]["valid"][test].mean(axis=1) for policy in bank], axis=1)[
        np.arange(len(test)), chosen
    ]
    evaluable = np.stack([data[policy]["evaluable"][test].mean(axis=1) for policy in bank], axis=1)[
        np.arange(len(test)), chosen
    ]
    tokens = np.stack([data[policy]["tokens"][test].mean(axis=1) for policy in bank], axis=1)[
        np.arange(len(test)), chosen
    ]
    selected = [bank[int(index)] for index in chosen]
    return {
        "item_ids": [item_ids[int(index)] for index in test],
        "routed": routed.tolist(),
        "champion": champion_values.tolist(),
        "oracle": oracle.tolist(),
        "valid": valid.tolist(),
        "evaluable": evaluable.tolist(),
        "tokens": tokens.tolist(),
        "selected": selected,
        "gain": float((routed - champion_values).mean()),
        "headroom": float((oracle - champion_values).mean()),
    }


def aggregate_bank_folds(
    folds: list[dict[str, Any]], fold_banks: list[list[str]]
) -> dict[str, Any]:
    routed = np.asarray([value for fold in folds for value in fold["routed"]])
    champion = np.asarray([value for fold in folds for value in fold["champion"]])
    oracle = np.asarray([value for fold in folds for value in fold["oracle"]])
    gain = routed - champion
    headroom = float((oracle - champion).mean())
    selected = [value for fold in folds for value in fold["selected"]]
    counts = Counter(selected)
    return {
        "routed_accuracy": float(routed.mean()),
        "own_champion_accuracy": float(champion.mean()),
        "routed_gain": float(gain.mean()),
        "oracle_accuracy": float(oracle.mean()),
        "oracle_headroom": headroom,
        "oracle_fraction_realized": float(gain.mean() / headroom) if headroom > 0 else None,
        "positive_fold_count": int(sum(fold["gain"] > 0 for fold in folds)),
        "worst_fold_gain": float(min(fold["gain"] for fold in folds)),
        "fold_gains": [fold["gain"] for fold in folds],
        "fold_gain_sd": float(np.std([fold["gain"] for fold in folds], ddof=1)),
        "commitment_validity": float(np.mean([x for fold in folds for x in fold["valid"]])),
        "semantic_evaluability": float(np.mean([x for fold in folds for x in fold["evaluable"]])),
        "mean_generated_tokens": float(np.mean([x for fold in folds for x in fold["tokens"]])),
        "maximum_policy_share": float(max(counts.values()) / len(selected)),
        "distinct_selected_policies": len(counts),
        "fold_banks": fold_banks,
    }


def part_a_analysis(
    raw: np.ndarray,
    data: dict[str, dict[str, np.ndarray]],
    item_ids: list[str],
    controllers: list[str],
    geometry: dict[str, np.ndarray],
    q31_summary: dict[str, Any],
    precheck: dict[str, Any],
) -> dict[str, Any]:
    outer = q3.balanced_hash_folds(item_ids, 5, "q3-realizable-utility-v1")
    hyperparameters = {
        fold: q31_summary["primary_bank"]["hyperparameters"][
            f"fold_{fold}_GEOMETRY_BLIND_POLICY_ID"
        ]
        for fold in range(5)
    }
    fold_contexts = []
    pool_size = int(precheck["part_a"]["candidate_random_bank_pool_per_fold"])
    distribution_size = int(precheck["part_a"]["evaluated_banks_per_distribution"])
    for fold in range(5):
        train = np.flatnonzero(outer != fold)
        test = np.flatnonzero(outer == fold)
        shells = q3.choose_shells(controllers, data, train)
        fixed = {
            name: q3.select_bank(
                f"{name}_MAXIMIN", 8, False, controllers, shells, geometry, data, train
            )
            for name in ("A0", "A1", "A2")
        }
        candidates = candidate_controller_banks(
            len(controllers), pool_size, fold, int(precheck["part_a"]["random_bank_seed"])
        )
        candidate_banks = [[shells[index] for index in values] for values in candidates]
        feature_rows = np.stack(
            [
                bank_features(bank, data, train, geometry["A0"], controllers)
                for bank in candidate_banks
            ]
        )
        target = bank_features(fixed["A0"], data, train, geometry["A0"], controllers)
        scale = feature_rows[:, :4].std(axis=0)
        scale[scale < 1e-8] = 1.0
        match_score = np.sqrt(np.sum(((feature_rows[:, :4] - target[:4]) / scale) ** 2, axis=1))
        match_order = np.lexsort((np.arange(pool_size), match_score))
        matched_indices = match_order[:distribution_size]
        competence_pool = match_order[: int(precheck["part_a"]["low_diversity_match_pool"])]
        low_order = sorted(
            (int(index) for index in competence_pool),
            key=lambda index: (feature_rows[index, 4], match_score[index], index),
        )
        low_indices = np.asarray(low_order[:distribution_size], dtype=int)
        deterministic_indices = np.arange(distribution_size, dtype=int)
        oracle = greedy_oracle_bank(shells, data, train)
        fold_contexts.append(
            {
                "fold": fold,
                "train": train,
                "test": test,
                "fixed": fixed,
                "candidate_banks": candidate_banks,
                "deterministic_indices": deterministic_indices,
                "matched_indices": matched_indices,
                "low_indices": low_indices,
                "oracle": oracle,
                "target_features": target.tolist(),
            }
        )

    fixed_results = {}
    for name in ("A0", "A1", "A2", "OUTCOME_OPTIMIZED_UPPER_BOUND"):
        folds = []
        banks = []
        for context in fold_contexts:
            fold = context["fold"]
            bank = context["oracle"] if name.startswith("OUTCOME") else context["fixed"][name]
            banks.append(bank)
            folds.append(
                fit_blind_bank_fold(
                    raw,
                    data,
                    item_ids,
                    context["train"],
                    context["test"],
                    bank,
                    hyperparameters[fold],
                    fold,
                    precheck,
                )
            )
        fixed_results[name] = aggregate_bank_folds(folds, banks)

    distributions = {}
    for distribution, key in (
        ("DETERMINISTIC_RANDOM", "deterministic_indices"),
        ("COMPETENCE_MATCHED_RANDOM", "matched_indices"),
        ("LOW_A0_DIVERSITY", "low_indices"),
    ):
        rows = []
        for replicate in range(distribution_size):
            folds = []
            banks = []
            for context in fold_contexts:
                fold = context["fold"]
                index = int(context[key][replicate])
                bank = context["candidate_banks"][index]
                banks.append(bank)
                folds.append(
                    fit_blind_bank_fold(
                        raw,
                        data,
                        item_ids,
                        context["train"],
                        context["test"],
                        bank,
                        hyperparameters[fold],
                        fold,
                        precheck,
                    )
                )
            row = aggregate_bank_folds(folds, banks)
            row.pop("fold_banks")
            rows.append(row)
        distributions[distribution] = rows

    matched = distributions["COMPETENCE_MATCHED_RANDOM"]
    observed = fixed_results["A0"]
    gains = np.asarray([row["routed_gain"] for row in matched])
    headrooms = np.asarray([row["oracle_headroom"] for row in matched])
    fold_matrix = np.asarray([row["fold_gains"] for row in matched])
    observed_gain = float(observed["routed_gain"])
    observed_headroom = float(observed["oracle_headroom"])
    percentile_gain = float(np.mean(gains <= observed_gain))
    percentile_headroom = float(np.mean(headrooms <= observed_headroom))
    p_gain = float((1 + np.sum(gains >= observed_gain)) / (len(gains) + 1))
    p_headroom = float((1 + np.sum(headrooms >= observed_headroom)) / (len(headrooms) + 1))
    median_fold = np.median(fold_matrix, axis=0)
    gates = {
        "a0_realization_gain": observed_gain >= precheck["part_a"]["realization_gain_min"],
        "gain_percentile": percentile_gain >= precheck["part_a"]["minimum_percentile"],
        "headroom_percentile": percentile_headroom >= precheck["part_a"]["minimum_percentile"],
        "gain_randomization_p": p_gain <= precheck["part_a"]["randomization_alpha"],
        "headroom_randomization_p": p_headroom <= precheck["part_a"]["randomization_alpha"],
        "gain_over_matched_median": (
            observed_gain - float(np.median(gains))
            >= precheck["part_a"]["minimum_gain_over_matched_median"]
        ),
        "fold_consistency": int(np.sum(np.asarray(observed["fold_gains"]) - median_fold >= 0))
        >= precheck["part_a"]["minimum_nonnegative_folds"],
    }
    ruling = (
        "GEOMETRY_BANK_SELECTION_SUPPORTED"
        if all(gates.values())
        else "GEOMETRY_BANK_SELECTION_NOT_SUPPORTED"
    )
    return {
        "fixed_banks": fixed_results,
        "distributions": {
            name: {
                "replicates": len(rows),
                "routed_gain_quantiles": np.quantile(
                    [row["routed_gain"] for row in rows], [0.025, 0.5, 0.95, 0.975]
                ).tolist(),
                "oracle_headroom_quantiles": np.quantile(
                    [row["oracle_headroom"] for row in rows], [0.025, 0.5, 0.95, 0.975]
                ).tolist(),
                "routed_accuracy_quantiles": np.quantile(
                    [row["routed_accuracy"] for row in rows], [0.025, 0.5, 0.95, 0.975]
                ).tolist(),
                "rows": rows,
            }
            for name, rows in distributions.items()
        },
        "attribution": {
            "a0_gain_percentile_matched_random": percentile_gain,
            "a0_headroom_percentile_matched_random": percentile_headroom,
            "gain_randomization_p": p_gain,
            "headroom_randomization_p": p_headroom,
            "a0_minus_matched_median_gain": observed_gain - float(np.median(gains)),
            "matched_median_fold_gains": median_fold.tolist(),
            "a0_nonnegative_fold_contrasts": int(
                np.sum(np.asarray(observed["fold_gains"]) - median_fold >= 0)
            ),
            "gates": gates,
        },
        "ruling": ruling,
    }


def policy_descriptors(
    controller_order: list[str],
    coords: dict[str, np.ndarray],
    mode: str,
    precheck: dict[str, Any],
) -> tuple[list[str], np.ndarray]:
    if mode == "TRUE":
        directions = {
            key: value / max(np.linalg.norm(value), 1e-12) for key, value in coords.items()
        }
    elif mode == "PERMUTED":
        rng = np.random.default_rng(precheck["part_b"]["coordinate_permutation_seed"])
        permutation = rng.permutation(len(controller_order))
        directions = {
            controller: coords[controller_order[int(permutation[index])]]
            / max(np.linalg.norm(coords[controller_order[int(permutation[index])]]), 1e-12)
            for index, controller in enumerate(controller_order)
        }
    elif mode == "RANDOM":
        rng = np.random.default_rng(precheck["part_b"]["random_coordinate_seed"])
        directions = {}
        for controller in controller_order:
            value = rng.normal(size=8)
            directions[controller] = value / np.linalg.norm(value)
    elif mode == "AGNOSTIC":
        directions = {controller: np.empty(0) for controller in controller_order}
    else:
        raise ValueError(mode)
    policies = []
    values = []
    for controller in controller_order:
        for shell, amplitude in (("MEDIUM", 0.25), ("STRONG", 0.50)):
            policies.append(f"{controller}_{shell}")
            if mode == "AGNOSTIC":
                values.append(np.asarray([amplitude]))
            else:
                values.append(np.concatenate([amplitude * directions[controller], [amplitude]]))
    return policies, np.stack(values)


def standardize_descriptors(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def fit_transfer_logistic(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    rank: int,
    l2: float,
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    k, gdim = z.shape
    u = rng.normal(scale=0.02, size=(d, rank))
    v = rng.normal(scale=0.02, size=(gdim, rank))
    a = np.zeros(d)
    g = np.zeros(gdim)
    bias = np.zeros(1)
    params = [u, v, a, g, bias]
    first = [np.zeros_like(value) for value in params]
    second = [np.zeros_like(value) for value in params]
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        qx = x @ u
        qz = z @ v
        score = bias[0] + x @ a[:, None] + (z @ g)[None, :] + qx @ qz.T
        error = (q31.sigmoid(score) - y) / (n * k)
        gradients = [
            x.T @ (error @ qz) + l2 * u / (n * k),
            z.T @ (error.T @ qx) + l2 * v / (n * k),
            x.T @ error.sum(axis=1) + l2 * a / (n * k),
            z.T @ error.sum(axis=0) + l2 * g / (n * k),
            np.asarray([error.sum()]),
        ]
        for index, (parameter, gradient) in enumerate(zip(params, gradients, strict=True)):
            first[index] = beta1 * first[index] + (1 - beta1) * gradient
            second[index] = beta2 * second[index] + (1 - beta2) * gradient * gradient
            corrected_first = first[index] / (1 - beta1**step)
            corrected_second = second[index] / (1 - beta2**step)
            parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return {"u": u, "v": v, "a": a, "g": g, "bias": bias}


def transfer_probabilities(
    x: np.ndarray, z: np.ndarray, fitted: dict[str, np.ndarray]
) -> np.ndarray:
    return q31.sigmoid(
        fitted["bias"][0]
        + x @ fitted["a"][:, None]
        + (z @ fitted["g"])[None, :]
        + (x @ fitted["u"]) @ (z @ fitted["v"]).T
    )


def fractional_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    value = np.clip(probability, 1e-8, 1 - 1e-8)
    return float(np.mean(-(y * np.log(value) + (1 - y) * np.log(1 - value))))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    a = rankdata(values_a)
    b = rankdata(values_b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def transfer_hyperparameters(
    raw: np.ndarray,
    y: np.ndarray,
    item_ids: list[str],
    z: np.ndarray,
    mode_index: int,
    precheck: dict[str, Any],
) -> dict[str, Any]:
    inner = q3.balanced_hash_folds(item_ids, 4, f"q3-role-transfer-{mode_index}")
    candidates = []
    for dimension in precheck["router"]["representation_dimensions"]:
        pca = {}
        for fold in range(4):
            pca[fold] = q31.fit_pca(raw[inner != fold], raw[inner == fold], int(dimension))
        for rank in precheck["router"]["interaction_ranks"]:
            for l2 in precheck["router"]["l2_grid"]:
                losses = []
                briers = []
                for fold in range(4):
                    tx, vx = pca[fold]
                    fitted = fit_transfer_logistic(
                        tx,
                        z,
                        y[inner != fold],
                        int(rank),
                        float(l2),
                        precheck["part_b"]["model_initialization_seed"]
                        + 10000 * mode_index
                        + 1009 * fold
                        + int(dimension)
                        + int(rank),
                        precheck["router"]["transfer_optimizer_steps"],
                        precheck["router"]["learning_rate"],
                    )
                    probability = transfer_probabilities(vx, z, fitted)
                    target = y[inner == fold]
                    losses.append(fractional_log_loss(target, probability))
                    briers.append(float(np.mean((probability - target) ** 2)))
                candidates.append(
                    (
                        -float(np.mean(losses)),
                        -float(np.mean(briers)),
                        -int(dimension),
                        -int(rank),
                        float(l2),
                        int(dimension),
                        int(rank),
                    )
                )
    best = max(candidates)
    return {
        "inner_log_loss": -best[0],
        "inner_brier": -best[1],
        "dimension": best[5],
        "rank": best[6],
        "l2": best[4],
    }


def calibration(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    bins = np.minimum((np.clip(probability, 0, 1) * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if np.any(mask):
            ece += float(mask.mean() * abs(probability[mask].mean() - y[mask].mean()))
    return {
        "log_loss": fractional_log_loss(y, probability),
        "brier": float(np.mean((probability - y) ** 2)),
        "ece_10": ece,
    }


def route_fresh_policies(
    mode: str,
    probability: np.ndarray,
    y_test: np.ndarray,
    fresh_policies: list[str],
) -> tuple[np.ndarray, list[int | str]]:
    """Route without using an arbitrary controller tie for the agnostic model."""
    if mode != "AGNOSTIC":
        chosen = probability.argmax(axis=1)
        return y_test[np.arange(len(y_test)), chosen], chosen.tolist()
    shell_columns = {
        shell: np.asarray(
            [
                index
                for index, policy in enumerate(fresh_policies)
                if shell_from_policy(policy) == shell
            ]
        )
        for shell in ("MEDIUM", "STRONG")
    }
    shell_scores = np.stack(
        [probability[:, shell_columns[shell]].mean(axis=1) for shell in shell_columns],
        axis=1,
    )
    chosen_shells = np.asarray(["MEDIUM", "STRONG"])[shell_scores.argmax(axis=1)]
    routed = np.asarray(
        [y_test[row, shell_columns[str(shell)]].mean() for row, shell in enumerate(chosen_shells)]
    )
    return routed, [f"UNIFORM_TIE_{shell}" for shell in chosen_shells]


def part_b_analysis(
    raw: np.ndarray,
    data: dict[str, dict[str, np.ndarray]],
    item_ids: list[str],
    controllers: list[str],
    coords: dict[str, np.ndarray],
    precheck: dict[str, Any],
) -> dict[str, Any]:
    historical = controllers[:31]
    fresh = controllers[31:]
    outer = q3.balanced_hash_folds(item_ids, 5, "q3-realizable-utility-v1")
    mode_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in precheck["part_b"]["modes"]
    }
    for fold in range(5):
        train = np.flatnonzero(outer != fold)
        test = np.flatnonzero(outer == fold)
        for mode_index, mode in enumerate(precheck["part_b"]["modes"]):
            policies, descriptors = policy_descriptors(controllers, coords, mode, precheck)
            historical_policies = [
                policy for policy in policies if controller_from_policy(policy) in historical
            ]
            fresh_policies = [
                policy for policy in policies if controller_from_policy(policy) in fresh
            ]
            index = {policy: offset for offset, policy in enumerate(policies)}
            z_train_raw = np.stack([descriptors[index[policy]] for policy in historical_policies])
            z_test_raw = np.stack([descriptors[index[policy]] for policy in fresh_policies])
            z_train, z_test = standardize_descriptors(z_train_raw, z_test_raw)
            y_train = np.stack(
                [data[policy]["correct"][train].mean(axis=1) for policy in historical_policies],
                axis=1,
            )
            y_test = np.stack(
                [data[policy]["correct"][test].mean(axis=1) for policy in fresh_policies], axis=1
            )
            selected = transfer_hyperparameters(
                raw[train],
                y_train,
                [item_ids[int(value)] for value in train],
                z_train,
                mode_index,
                precheck,
            )
            tx, vx = q31.fit_pca(raw[train], raw[test], selected["dimension"])
            fitted = fit_transfer_logistic(
                tx,
                z_train,
                y_train,
                selected["rank"],
                selected["l2"],
                precheck["part_b"]["model_initialization_seed"] + 20000 * fold + mode_index,
                precheck["router"]["transfer_optimizer_steps"],
                precheck["router"]["learning_rate"],
            )
            probability = transfer_probabilities(vx, z_test, fitted)
            routed, chosen = route_fresh_policies(mode, probability, y_test, fresh_policies)
            random_policy = y_test.mean(axis=1)
            oracle = y_test.max(axis=1)
            ranking = [spearman(probability[row], y_test[row]) for row in range(len(test))]
            finite_ranking = [value for value in ranking if value is not None]
            top_k = {}
            for k in (1, 3, 5):
                if mode == "AGNOSTIC":
                    top_k[str(k)] = None
                    continue
                hits = []
                for row in range(len(test)):
                    predicted = set(np.argsort(probability[row], kind="mergesort")[-k:])
                    target = set(np.flatnonzero(y_test[row] == y_test[row].max()))
                    hits.append(bool(predicted.intersection(target)))
                top_k[str(k)] = float(np.mean(hits))
            mode_results[mode].append(
                {
                    "fold": fold,
                    "hyperparameters": selected,
                    "routed_accuracy": float(routed.mean()),
                    "random_policy_accuracy": float(random_policy.mean()),
                    "oracle_accuracy": float(oracle.mean()),
                    "routing_gain": float((routed - random_policy).mean()),
                    "predictive": calibration(y_test, probability),
                    "mean_item_policy_rank_correlation": (
                        float(np.mean(finite_ranking)) if finite_ranking else None
                    ),
                    "rank_degenerate_items": len(ranking) - len(finite_ranking),
                    "top_k_policy_recall": top_k,
                    "selected_policies": chosen,
                    "routed": routed.tolist(),
                    "random_policy": random_policy.tolist(),
                    "oracle": oracle.tolist(),
                    "probability": probability.tolist(),
                    "target": y_test.tolist(),
                    "test_item_indices": test.tolist(),
                    "fresh_policies": fresh_policies,
                }
            )

    summarized = {}
    for mode, folds in mode_results.items():
        routed = np.asarray([value for fold in folds for value in fold["routed"]])
        random_policy = np.asarray([value for fold in folds for value in fold["random_policy"]])
        oracle = np.asarray([value for fold in folds for value in fold["oracle"]])
        probability = np.concatenate([np.asarray(fold["probability"]) for fold in folds], axis=0)
        target = np.concatenate([np.asarray(fold["target"]) for fold in folds], axis=0)
        gains = [fold["routing_gain"] for fold in folds]
        headroom = float((oracle - random_policy).mean())
        controller_residuals = {}
        fresh_policies = folds[0]["fresh_policies"]
        for controller in fresh:
            cols = [
                i
                for i, policy in enumerate(fresh_policies)
                if controller_from_policy(policy) == controller
            ]
            controller_residuals[controller] = float(
                np.mean(probability[:, cols] - target[:, cols])
            )
        summarized[mode] = {
            "routed_accuracy": float(routed.mean()),
            "random_policy_accuracy": float(random_policy.mean()),
            "routing_gain": float((routed - random_policy).mean()),
            "oracle_accuracy": float(oracle.mean()),
            "oracle_headroom_over_random": headroom,
            "oracle_fraction_realized": float((routed - random_policy).mean() / headroom),
            "positive_fold_count": int(sum(value > 0 for value in gains)),
            "worst_fold_gain": float(min(gains)),
            "fold_gains": gains,
            "fold_gain_sd": float(np.std(gains, ddof=1)),
            "predictive": calibration(target, probability),
            "mean_item_policy_rank_correlation": float(
                np.mean(
                    [
                        fold["mean_item_policy_rank_correlation"]
                        for fold in folds
                        if fold["mean_item_policy_rank_correlation"] is not None
                    ]
                )
            ),
            "rank_degenerate_items": int(sum(fold["rank_degenerate_items"] for fold in folds)),
            "top_k_policy_recall": {
                str(k): (
                    None
                    if mode == "AGNOSTIC"
                    else float(np.mean([fold["top_k_policy_recall"][str(k)] for fold in folds]))
                )
                for k in (1, 3, 5)
            },
            "per_fresh_controller_mean_prediction_residual": controller_residuals,
            "fold_hyperparameters": [fold["hyperparameters"] for fold in folds],
        }

    # Descriptor-only true-A0 kernel prior: historical competence is smoothed
    # to fresh policies with no fresh outcome and no item-specific label.
    kernel_folds = []
    true_policies, true_descriptors = policy_descriptors(controllers, coords, "TRUE", precheck)
    pindex = {policy: index for index, policy in enumerate(true_policies)}
    historical_policies = [p for p in true_policies if controller_from_policy(p) in historical]
    fresh_policies = [p for p in true_policies if controller_from_policy(p) in fresh]
    for fold in range(5):
        train = np.flatnonzero(outer != fold)
        test = np.flatnonzero(outer == fold)
        zh_raw = np.stack([true_descriptors[pindex[p]] for p in historical_policies])
        zf_raw = np.stack([true_descriptors[pindex[p]] for p in fresh_policies])
        zh, zf = standardize_descriptors(zh_raw, zf_raw)
        pair = np.linalg.norm(zh[:, None, :] - zh[None, :, :], axis=2)
        bandwidth = float(np.median(pair[np.triu_indices_from(pair, k=1)]))
        weights = np.exp(
            -0.5 * (np.linalg.norm(zf[:, None, :] - zh[None, :, :], axis=2) / bandwidth) ** 2
        )
        competence = np.asarray([data[p]["correct"][train].mean() for p in historical_policies])
        prediction = weights @ competence / weights.sum(axis=1)
        chosen = int(np.argmax(prediction))
        y = np.stack([data[p]["correct"][test].mean(axis=1) for p in fresh_policies], axis=1)
        routed = y[:, chosen]
        random_policy = y.mean(axis=1)
        kernel_folds.append(
            {
                "fold": fold,
                "selected_policy": fresh_policies[chosen],
                "routing_accuracy": float(routed.mean()),
                "random_policy_accuracy": float(random_policy.mean()),
                "routing_gain": float((routed - random_policy).mean()),
                "bandwidth": bandwidth,
            }
        )
    kernel_gain = [row["routing_gain"] for row in kernel_folds]
    kernel = {
        "routed_accuracy": float(np.mean([row["routing_accuracy"] for row in kernel_folds])),
        "random_policy_accuracy": float(
            np.mean([row["random_policy_accuracy"] for row in kernel_folds])
        ),
        "routing_gain": float(np.mean(kernel_gain)),
        "positive_fold_count": int(sum(value > 0 for value in kernel_gain)),
        "worst_fold_gain": float(min(kernel_gain)),
        "folds": kernel_folds,
    }

    # Controller-agnostic global prior: use historical outer-training outcomes
    # only to choose one shell, then average uniformly over fresh controllers.
    global_prior_folds = []
    for fold in range(5):
        train = np.flatnonzero(outer != fold)
        test = np.flatnonzero(outer == fold)
        historical_shell_accuracy = {
            shell: float(
                np.mean(
                    [
                        data[f"{controller}_{shell}"]["correct"][train].mean()
                        for controller in historical
                    ]
                )
            )
            for shell in ("MEDIUM", "STRONG")
        }
        preferred_shell = max(
            ("MEDIUM", "STRONG"),
            key=lambda shell: (historical_shell_accuracy[shell], shell == "MEDIUM"),
        )
        fresh_shell_policies = [f"{controller}_{preferred_shell}" for controller in fresh]
        routed = np.stack(
            [data[policy]["correct"][test].mean(axis=1) for policy in fresh_shell_policies],
            axis=1,
        ).mean(axis=1)
        all_fresh_policies = [
            f"{controller}_{shell}" for controller in fresh for shell in ("MEDIUM", "STRONG")
        ]
        random_policy = np.stack(
            [data[policy]["correct"][test].mean(axis=1) for policy in all_fresh_policies],
            axis=1,
        ).mean(axis=1)
        global_prior_folds.append(
            {
                "fold": fold,
                "preferred_shell": preferred_shell,
                "historical_shell_accuracy": historical_shell_accuracy,
                "routing_accuracy": float(routed.mean()),
                "random_policy_accuracy": float(random_policy.mean()),
                "routing_gain": float((routed - random_policy).mean()),
            }
        )
    global_prior = {
        "routed_accuracy": float(np.mean([row["routing_accuracy"] for row in global_prior_folds])),
        "random_policy_accuracy": float(
            np.mean([row["random_policy_accuracy"] for row in global_prior_folds])
        ),
        "routing_gain": float(np.mean([row["routing_gain"] for row in global_prior_folds])),
        "folds": global_prior_folds,
    }

    true = summarized["TRUE"]
    controls = ["PERMUTED", "RANDOM", "AGNOSTIC"]
    differences = {}
    gates = {
        "true_realization_gain": true["routing_gain"] >= precheck["part_b"]["realization_gain_min"],
        "true_positive_folds": true["positive_fold_count"]
        >= precheck["part_b"]["minimum_positive_folds"],
        "true_worst_fold": true["worst_fold_gain"] >= precheck["part_b"]["worst_fold_gain_min"],
    }
    for control in controls:
        other = summarized[control]
        route_difference = true["routing_gain"] - other["routing_gain"]
        log_loss_difference = other["predictive"]["log_loss"] - true["predictive"]["log_loss"]
        fold_difference = np.asarray(true["fold_gains"]) - np.asarray(other["fold_gains"])
        differences[control] = {
            "routing_gain_difference": route_difference,
            "predictive_log_loss_improvement": log_loss_difference,
            "nonnegative_fold_count": int(np.sum(fold_difference >= 0)),
            "fold_differences": fold_difference.tolist(),
        }
        gates[f"routing_over_{control.lower()}"] = (
            route_difference >= precheck["part_b"]["minimum_routing_gain_over_control"]
        )
        gates[f"log_loss_over_{control.lower()}"] = (
            log_loss_difference >= precheck["part_b"]["minimum_log_loss_improvement"]
        )
        gates[f"folds_over_{control.lower()}"] = (
            int(np.sum(fold_difference >= 0)) >= precheck["part_b"]["minimum_nonnegative_folds"]
        )

    def realization(value: dict[str, Any]) -> bool:
        return bool(
            value["routing_gain"] >= precheck["part_b"]["realization_gain_min"]
            and value["positive_fold_count"] >= precheck["part_b"]["minimum_positive_folds"]
            and value["worst_fold_gain"] >= precheck["part_b"]["worst_fold_gain_min"]
        )

    if all(gates.values()):
        ruling = "CONTROLLER_OOS_GEOMETRY_TRANSFER_SUPPORTED"
    elif any(realization(summarized[name]) for name in ("TRUE", "PERMUTED", "RANDOM", "AGNOSTIC")):
        ruling = "CONTROLLER_OOS_SELECTABILITY_WITHOUT_GEOMETRY"
    else:
        ruling = "CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED"
    return {
        "fixed_controller_split": {"training_historical": historical, "held_out_fresh": fresh},
        "models": summarized,
        "true_geometry_kernel_prior": kernel,
        "historical_global_shell_prior": global_prior,
        "attribution": {"differences": differences, "gates": gates},
        "ruling": ruling,
    }


def high_level_ruling(part_a: str, part_b: str) -> tuple[str, str]:
    if (
        part_a == "GEOMETRY_BANK_SELECTION_SUPPORTED"
        and part_b == "CONTROLLER_OOS_GEOMETRY_TRANSFER_SUPPORTED"
    ):
        return (
            "Q3_GEOMETRY_BRIDGE_SUPPORTED_READY_FOR_FRESH_INSTRUMENT_DESIGN",
            "Geometry supports both portfolio construction and controller-OOS routing transfer.",
        )
    if part_a == "GEOMETRY_BANK_SELECTION_SUPPORTED":
        return (
            "Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING",
            "Geometry supports portfolio construction but not held-out-controller routing.",
        )
    if part_b == "CONTROLLER_OOS_GEOMETRY_TRANSFER_SUPPORTED":
        return (
            "Q3_GEOMETRY_ROLE_UNRESOLVED",
            "Controller-OOS transfer is positive, but bank-construction specificity is absent.",
        )
    return (
        "Q3_CAUSAL_POLICY_SELECTABILITY_WITHOUT_GEOMETRY_ATTRIBUTION",
        "Fixed-bank selectability remains, but neither tested geometry role is "
        "adequately supported.",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--fresh-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    precheck = read_json(PRECHECK)
    if precheck.get("status") != "Q3_GEOMETRY_ROLE_DECOMPOSITION_PRECHECK_FROZEN":
        raise RuntimeError("Q3.2 precheck is not frozen")
    if sha256_file(Path(__file__).resolve()) != precheck["implementation"]["analysis_sha256"]:
        raise RuntimeError("Q3.2 analysis source hash mismatch")
    q3.verify_inputs(args.historical_scores, args.fresh_scores)
    if sha256_file(args.representations) != precheck["sources"]["representation_matrix_sha256"]:
        raise RuntimeError("Q3.2 representation matrix hash mismatch")
    q31_summary = read_json(Q31_SUMMARY)
    if q31_summary["status"] != precheck["immutable_q3_1"]["classification"]:
        raise RuntimeError("Q3.1 classification changed")
    raw = np.load(args.representations, allow_pickle=False).astype(np.float64)
    if raw.shape != (300, 4096) or not np.isfinite(raw).all():
        raise RuntimeError("invalid prompt-representation matrix")
    item_ids = list(read_json(PANEL)["item_ids"])
    data, source_counts = q3.load_outcomes(args.historical_scores, args.fresh_scores, item_ids)
    controllers, coords = q3.load_controller_coordinates()
    geometry = q3.combined_geometry()
    part_a = part_a_analysis(raw, data, item_ids, controllers, geometry, q31_summary, precheck)
    part_b = part_b_analysis(raw, data, item_ids, controllers, coords, precheck)
    ruling, narrative = high_level_ruling(part_a["ruling"], part_b["ruling"])
    result = {
        "schema_version": "q3-geometry-role-decomposition-results-v1",
        "status": ruling,
        "evidence_class": ["DEVELOPMENT_ONLY", "POST_CLOSED_RESULT_PLANNING"],
        "immutable_q3_1": precheck["immutable_q3_1"],
        "part_a": part_a,
        "part_b": part_b,
        "narrative": narrative,
        "source_counts": source_counts,
        "firewall": {
            "new_semantic_trajectories": 0,
            "new_qwen_forwards": 0,
            "fresh_evaluation_outcomes_inspected": False,
            "q3_confirmatory_experiment": "NOT_RUN",
            "q1_q2_q3_1_classifications_changed": False,
        },
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": ruling,
                "part_a": part_a["ruling"],
                "part_b": part_b["ruling"],
                "a0_percentile": part_a["attribution"]["a0_gain_percentile_matched_random"],
                "true_controller_oos_gain": part_b["models"]["TRUE"]["routing_gain"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
