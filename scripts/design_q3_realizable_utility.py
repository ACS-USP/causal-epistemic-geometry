#!/usr/bin/env python3
"""Build the Q3.0 realizable-utility design artifacts from closed data.

This is a development-only, outcome-read-only analysis.  It never reads raw
generation text, never opens a future evaluation outcome, and never performs
model inference.  Private scored artifacts are supplied explicitly and are
verified against their frozen SHA-256 identities.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/q3_realizable_utility_design"
PRECHECK = OUT / "Q3_REALIZABLE_UTILITY_DESIGN_PRECHECK.json"
PANEL = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"
PROVENANCE = (
    ROOT / "review/q2_m3_qualification_cruxeval_provenance" / "CRUXEVAL_PROVENANCE_LEDGER.jsonl"
)
HIST_BANK = ROOT / "review/q2_v4_1_31_safe_bank_review" / "SAFE_31_IMMUTABLE_MANIFEST.json"
FRESH_BANK = (
    ROOT
    / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout"
    / "V2_SELECTED_CONTROLLER_BANK.json"
)
FRESH_CANDIDATES = (
    ROOT
    / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
    / "V2_CANDIDATE_BANK_MANIFEST.json"
)
HIST_MATRICES = ROOT / "review/q2_v4_1_prediction_lock/PREDICTION_MATRICES.npz"
FRESH_MATRICES = (
    ROOT
    / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout"
    / "PREDICTION_MATRICES.npz"
)

EXPECTED = {
    "historical_scores": "a6a9f4b419d4531716337d2277688063d5655167a5d7b1a9bd85b34217f8a33f",
    "fresh_scores": "9f03d96d40839e228d6cfb55408ea056e262fbf7e9aef2e863080e035e4b721b",
    "panel": "c127cf3594e8ea849dbd038492606b3afaaac406feb4146188769c04d6691187",
    "provenance": "3f8c22baa6fee56fd292cc9dabb9bcd8500c5b09038e2cd7e4715c1668f8a7cc",
    "historical_matrices": "723348bddd1de0b482bc859f888f982c8a3dd99e406331cb9a4642ea2c7dcae8",
    "fresh_matrices": "b4ec00985e750c5bb8fd7fd49228267ec576bf6c2ad2ac3984f6f2390d927703",
}
LABELS = ["DEVELOPMENT_ONLY", "POST_CLOSED_RESULT_PLANNING"]
K_VALUES = (2, 4, 8)
BANK_METHODS = ("A0_MAXIMIN", "A1_MAXIMIN", "A2_MAXIMIN", "ACCURACY_QUALIFIED_A0_MAXIMIN")
MODEL_NAMES = ("GEOMETRY_BILINEAR", "GEOMETRY_NONLINEAR_16", "GEOMETRY_BLIND_MATCHED")
LAMBDAS = (0.01, 0.1, 1.0, 10.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def verify_inputs(hist_scores: Path, fresh_scores: Path) -> None:
    checks = {
        "historical_scores": hist_scores,
        "fresh_scores": fresh_scores,
        "panel": PANEL,
        "provenance": PROVENANCE,
        "historical_matrices": HIST_MATRICES,
        "fresh_matrices": FRESH_MATRICES,
    }
    bad = {
        name: {"expected": EXPECTED[name], "observed": sha256_file(path)}
        for name, path in checks.items()
        if sha256_file(path) != EXPECTED[name]
    }
    if bad:
        raise RuntimeError(f"frozen input hash mismatch: {bad}")
    precheck = read_json(PRECHECK)
    if precheck.get("classification") != "Q3_REALIZABLE_UTILITY_DESIGN_PRECHECK_FROZEN":
        raise RuntimeError("Q3 precheck is not frozen")


def hash_fold(identifier: str, folds: int, salt: str) -> int:
    value = hashlib.sha256(f"{identifier}|{salt}".encode()).digest()
    return int.from_bytes(value[:8], "big") % folds


def balanced_hash_folds(ids: list[str], folds: int, salt: str) -> np.ndarray:
    ranked = sorted(
        range(len(ids)), key=lambda i: hashlib.sha256(f"{ids[i]}|{salt}".encode()).hexdigest()
    )
    out = np.empty(len(ids), dtype=int)
    for rank, index in enumerate(ranked):
        out[index] = rank % folds
    return out


def structural_hash(prompt: str) -> str:
    normalized = re.sub(r"\b\d+\b", "<NUM>", prompt)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def prompt_features(prompt: str) -> list[float]:
    lines = prompt.splitlines()
    words = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", prompt)
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", prompt, flags=re.S | re.I)
    code = max(code_blocks, key=len) if code_blocks else prompt
    ast_counts = Counter()
    try:
        tree = ast.parse(code)
        ast_counts.update(type(node).__name__ for node in ast.walk(tree))
        ast_ok = 1.0
    except SyntaxError:
        ast_ok = 0.0
    max_indent = max((len(x) - len(x.lstrip(" ")) for x in code.splitlines()), default=0)
    features = [
        len(prompt),
        len(lines),
        len(words),
        len(set(words)),
        len(code),
        len(code.splitlines()),
        prompt.count("("),
        prompt.count("["),
        prompt.count("{"),
        prompt.count("="),
        prompt.count("+") + prompt.count("-") + prompt.count("*") + prompt.count("/"),
        sum(ch.isdigit() for ch in prompt),
        max_indent,
        ast_ok,
        ast_counts["If"],
        ast_counts["For"],
        ast_counts["While"],
        ast_counts["Call"],
        ast_counts["List"],
        ast_counts["Dict"],
        ast_counts["Set"],
        ast_counts["FunctionDef"],
        ast_counts["Subscript"],
        ast_counts["Compare"],
        ast_counts["BinOp"],
        ast_counts["BoolOp"],
    ]
    return [float(x) for x in features]


def build_exposure_ledger(panel_ids: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = read_jsonl(PROVENANCE)
    entries = []
    exact_prompt = Counter(
        row["canonical_content"]["prompt_sha256"]
        for row in rows
        if isinstance(row.get("canonical_content"), dict)
        and row["canonical_content"].get("prompt_sha256")
    )
    q2_ids = panel_ids
    for row in rows:
        content = row.get("canonical_content") or {}
        item_id = str(row["item_id"])
        globally_untouched = not (
            row["free_generation_inference"]
            or row["semantic_correctness_scored"]
            or row["outcome_inspected_by_researchers"]
        )
        candidate_policy_outcome_observed = item_id in q2_ids
        entries.append(
            {
                "dataset": row["dataset_repo"],
                "dataset_revision": row["dataset_revision"],
                "split": row["source_split"],
                "item_id": item_id,
                "official_index": row["official_index"],
                "family_id": item_id,
                "prompt_sha256": content.get("prompt_sha256"),
                "reference_sha256": content.get("reference_sha256"),
                "roles": row["roles"],
                "model_output_generated": bool(row["free_generation_inference"]),
                "correctness_scored": bool(row["semantic_correctness_scored"]),
                "outcome_inspected": bool(row["outcome_inspected_by_researchers"]),
                "candidate_q3_policy_outcome_observed": candidate_policy_outcome_observed,
                "globally_untouched": globally_untouched,
                "eligible_q3_development": candidate_policy_outcome_observed,
                "eligible_fresh_evaluation_tier_a": globally_untouched
                and not candidate_policy_outcome_observed,
                "eligible_fresh_evaluation_tier_b": not candidate_policy_outcome_observed,
                "exact_prompt_duplicate_count": exact_prompt[content.get("prompt_sha256")]
                if content.get("prompt_sha256")
                else None,
            }
        )
    tier_a = [x for x in entries if x["eligible_fresh_evaluation_tier_a"]]
    tier_b = [x for x in entries if x["eligible_fresh_evaluation_tier_b"]]
    ledger = {
        "schema_version": "1.0",
        "labels": LABELS,
        "source_sha256": EXPECTED["provenance"],
        "release_safe": True,
        "raw_prompt_or_reference_content_included": False,
        "family_definition": (
            "CRUXEval output-prediction item; no multi-row family structure is "
            "present in the official item axis."
        ),
        "summary": {
            "total_items": len(entries),
            "q2_closed_development_items": len(q2_ids),
            "globally_untouched_items": len(tier_a),
            "no_candidate_policy_outcome_items": len(tier_b),
            "exact_prompt_duplicate_groups": sum(v > 1 for v in exact_prompt.values()),
        },
        "items": entries,
    }
    feasibility = {
        "schema_version": "1.0",
        "labels": LABELS,
        "future_outcomes_inspected": False,
        "future_holdout_permanently_allocated": False,
        "tier_a": {
            "definition": (
                "No prior free generation, scored correctness, inspected outcome, "
                "or candidate-policy outcome."
            ),
            "available_families": len(tier_a),
            "candidate_ids": [x["item_id"] for x in tier_a],
            "status": "PREFERRED_BUT_NUMERICALLY_INADEQUATE" if len(tier_a) < 200 else "AVAILABLE",
        },
        "tier_b": {
            "definition": (
                "No prior outcome from any of the exact 47 Q2 candidate "
                "controllers, but other historical exposure may exist."
            ),
            "available_families": len(tier_b),
            "candidate_ids_not_permanently_allocated": [x["item_id"] for x in tier_b],
            "limitation": (
                "Not globally untouched; prior project exposure can induce "
                "benchmark-level adaptation risk."
            ),
        },
        "family_overlap_with_q2_development": 0,
        "exact_prompt_duplicates_across_tier_b_and_development": 0,
        "reference_or_prompt_content_opened": False,
        "provisional_ruling": "POWER_AUDIT_REQUIRED",
    }
    return ledger, feasibility


def load_outcomes(
    hist_path: Path, fresh_path: Path, item_ids: list[str]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    index = {item: i for i, item in enumerate(item_ids)}
    data: dict[str, dict[str, np.ndarray]] = {}
    sources = [("historical", read_jsonl(hist_path)), ("fresh", read_jsonl(fresh_path))]
    source_counts = {}
    for source, rows in sources:
        source_counts[source] = len(rows)
        for row in rows:
            # Do not load or propagate answer/reference fields.
            cond = str(row["condition"])
            if cond not in data:
                data[cond] = {
                    key: np.full((len(item_ids), 2), np.nan)
                    for key in ("correct", "valid", "evaluable", "tokens")
                }
                data[cond]["answer"] = np.full((len(item_ids), 2), None, dtype=object)
            i = index[str(row["item_id"])]
            r = int(row["rollout_index"])
            data[cond]["correct"][i, r] = float(bool(row["correct"]))
            data[cond]["valid"][i, r] = float(bool(row["commitment_valid"]))
            data[cond]["evaluable"][i, r] = float(bool(row["semantic_evaluable"]))
            data[cond]["tokens"][i, r] = float(row["generated_token_count"])
            if row.get("parsed_answer") is not None:
                data[cond]["answer"][i, r] = json.dumps(
                    row["parsed_answer"], sort_keys=True, ensure_ascii=False
                )
    if any(
        np.isnan(values[key]).any()
        for values in data.values()
        for key in ("correct", "valid", "evaluable", "tokens")
    ):
        raise RuntimeError("closed development outcome matrix has missing cells")
    expected_conditions = 1 + 31 * 2 + 16 * 2
    if len(data) != expected_conditions:
        raise RuntimeError(f"expected {expected_conditions} policies, found {len(data)}")
    return data, source_counts


def load_controller_coordinates() -> tuple[list[str], dict[str, np.ndarray]]:
    historical = read_json(HIST_BANK)
    fresh_manifest = read_json(FRESH_CANDIDATES)
    selected = set(read_json(FRESH_BANK)["selected_ids"])
    coords = {
        row["candidate_id"]: np.asarray(row["coefficients"], dtype=float)
        for row in historical["directions"]
    }
    coords.update(
        {
            row["candidate_id"]: np.asarray(row["coefficients"], dtype=float)
            for row in fresh_manifest["candidates"]
            if row["candidate_id"] in selected
        }
    )
    order = list(
        read_json(ROOT / "review/q2_v4_1_prediction_lock/PREDICTION_MATRIX_METADATA.json")[
            "controller_order"
        ]
    )
    order += list(
        read_json(
            ROOT
            / "review/q2_oos_fresh_controller_design"
            / "v2_presemantic_closeout/PREDICTION_MATRIX_METADATA.json"
        )["fresh_controller_order"]
    )
    if set(order) != set(coords) or len(order) != 47:
        raise RuntimeError("controller-coordinate population mismatch")
    return order, coords


def combined_geometry() -> dict[str, np.ndarray]:
    hist = np.load(HIST_MATRICES)
    fresh = np.load(FRESH_MATRICES)
    out = {}
    for name in ("A0", "A1", "A2"):
        h = 0.5 * (hist[f"{name}_MEDIUM"] + hist[f"{name}_STRONG"])
        ff = 0.5 * (fresh[f"{name}_MEDIUM_FRESH_FRESH"] + fresh[f"{name}_STRONG_FRESH_FRESH"])
        fr = 0.5 * (
            fresh[f"{name}_MEDIUM_FRESH_REFERENCE"] + fresh[f"{name}_STRONG_FRESH_REFERENCE"]
        )
        matrix = np.block([[h, fr.T], [fr, ff]])
        if matrix.shape != (47, 47) or not np.allclose(matrix, matrix.T):
            raise RuntimeError(f"invalid combined {name} geometry")
        out[name] = matrix
    return out


def condition_for(controller: str, shell: str) -> str:
    return f"{controller}_{shell}"


def choose_shells(
    controllers: list[str], data: dict[str, dict[str, np.ndarray]], train: np.ndarray
) -> list[str]:
    policies = []
    for controller in controllers:
        choices = []
        for shell in ("MEDIUM", "STRONG"):
            cond = condition_for(controller, shell)
            values = data[cond]
            choices.append(
                (
                    float(values["correct"][train].mean()),
                    float(values["evaluable"][train].mean()),
                    -float(values["tokens"][train].mean()),
                    shell == "MEDIUM",
                    cond,
                )
            )
        policies.append(max(choices)[-1])
    return policies


def maximin_select(eligible: list[int], distance: np.ndarray, k: int, ids: list[str]) -> list[int]:
    if len(eligible) < k:
        raise RuntimeError("insufficient eligible controllers")
    first = min(eligible, key=lambda i: (float(distance[i, eligible].mean()), ids[i]))
    selected = [first]
    while len(selected) < k:
        remaining = [i for i in eligible if i not in selected]
        chosen = max(
            remaining, key=lambda i: (float(distance[i, selected].min()), -ord(ids[i][0]), ids[i])
        )
        # Resolve equal distances explicitly by lexicographic identity.
        score = max(float(distance[i, selected].min()) for i in remaining)
        tied = sorted(
            i for i in remaining if abs(float(distance[i, selected].min()) - score) <= 1e-12
        )
        selected.append(tied[0] if len(tied) else chosen)
    return selected


def select_bank(
    method: str,
    k: int,
    include_baseline: bool,
    controllers: list[str],
    shell_policies: list[str],
    geometry: dict[str, np.ndarray],
    data: dict[str, dict[str, np.ndarray]],
    train: np.ndarray,
) -> list[str]:
    eligible = list(range(len(controllers)))
    if method == "ACCURACY_QUALIFIED_A0_MAXIMIN":
        all_acc = {p: float(data[p]["correct"][train].mean()) for p in shell_policies}
        champion = max(all_acc.values())
        baseline_valid = float(data["BASELINE"]["valid"][train].mean())
        baseline_eval = float(data["BASELINE"]["evaluable"][train].mean())
        eligible = [
            i
            for i, p in enumerate(shell_policies)
            if all_acc[p] >= max(0.50, champion - 0.10)
            and float(data[p]["valid"][train].mean()) >= baseline_valid - 0.05
            and float(data[p]["evaluable"][train].mean()) >= baseline_eval - 0.05
        ]
        matrix = geometry["A0"]
    else:
        matrix = geometry[method.split("_")[0]]
    target = k - int(include_baseline)
    selected = maximin_select(eligible, matrix, target, controllers)
    bank = [shell_policies[i] for i in selected]
    if include_baseline:
        bank = ["BASELINE"] + bank
    return bank


def policy_coords(
    bank: list[str], coords: dict[str, np.ndarray], blind: bool = False
) -> np.ndarray:
    values = []
    for policy in bank:
        if policy == "BASELINE":
            vector = np.zeros(8)
        else:
            controller = policy.rsplit("_", 1)[0]
            vector = coords[controller].copy()
            vector /= max(np.linalg.norm(vector), 1e-12)
        if blind:
            seed = int.from_bytes(hashlib.sha256(f"q3-blind|{policy}".encode()).digest()[:8], "big")
            vector = np.random.default_rng(seed).normal(size=8)
            vector /= np.linalg.norm(vector)
        values.append(vector)
    return np.stack(values)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-9] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def design_matrix(x: np.ndarray, c: np.ndarray, nonlinear: bool, seed: int) -> np.ndarray:
    n, d = x.shape
    k, g = c.shape
    interaction = np.einsum("nd,kg->nkdg", x, c).reshape(n * k, d * g)
    policy_bias = np.tile(np.eye(k), (n, 1))
    pieces = [np.ones((n * k, 1)), policy_bias, interaction]
    if nonlinear:
        rng = np.random.default_rng(seed)
        projection = rng.normal(
            scale=1 / math.sqrt(max(1, interaction.shape[1])), size=(interaction.shape[1], 16)
        )
        pieces.append(np.tanh(interaction @ projection))
    return np.concatenate(pieces, axis=1)


def fit_ridge_router(
    x: np.ndarray, c: np.ndarray, y: np.ndarray, lam: float, nonlinear: bool, seed: int
) -> np.ndarray:
    z = design_matrix(x, c, nonlinear, seed)
    target = y.reshape(-1)
    penalty = np.eye(z.shape[1]) * lam
    penalty[0, 0] = 0.0
    return np.linalg.solve(z.T @ z + penalty, z.T @ target)


def predict_router(
    x: np.ndarray, c: np.ndarray, coef: np.ndarray, nonlinear: bool, seed: int
) -> np.ndarray:
    scores = design_matrix(x, c, nonlinear, seed) @ coef
    return scores.reshape(len(x), len(c)).argmax(axis=1)


def select_lambda(
    x: np.ndarray, c: np.ndarray, y: np.ndarray, item_ids: list[str], model: str, seed: int
) -> float:
    folds = balanced_hash_folds(item_ids, 4, f"q3-inner-{seed}")
    blind = model == "GEOMETRY_BLIND_MATCHED"
    c_model = policy_coords_from_array(c, bank_size=len(c), blind=blind, seed=seed)
    nonlinear = model == "GEOMETRY_NONLINEAR_16"
    scores = []
    for lam in LAMBDAS:
        vals = []
        for fold in range(4):
            tr = folds != fold
            va = folds == fold
            tx, vx = standardize(x[tr], x[va])
            coef = fit_ridge_router(tx, c_model, y[tr], lam, nonlinear, seed)
            chosen = predict_router(vx, c_model, coef, nonlinear, seed)
            vals.extend(y[va, chosen].tolist())
        scores.append((float(np.mean(vals)), -lam, lam))
    return max(scores)[2]


def policy_coords_from_array(c: np.ndarray, bank_size: int, blind: bool, seed: int) -> np.ndarray:
    if not blind:
        return c
    values = []
    for k in range(bank_size):
        rng = np.random.default_rng(seed + 104729 * (k + 1))
        v = rng.normal(size=c.shape[1])
        values.append(v / np.linalg.norm(v))
    return np.stack(values)


def paired_interval(values: np.ndarray, seed: int, draws: int = 50000) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    estimates = np.empty(draws)
    for start in range(0, draws, 2000):
        count = min(2000, draws - start)
        idx = rng.integers(0, n, size=(count, n))
        estimates[start : start + count] = values[idx].mean(axis=1)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.5, 0.975])]


def bank_metrics(
    bank: list[str],
    data: dict[str, dict[str, np.ndarray]],
    test: np.ndarray,
    geometry: dict[str, np.ndarray],
    controller_order: list[str],
    champion: str,
) -> dict[str, Any]:
    correct = np.stack([data[p]["correct"][test].mean(axis=1) for p in bank], axis=1)
    valid = np.stack([data[p]["valid"][test].mean(axis=1) for p in bank], axis=1)
    evaluable = np.stack([data[p]["evaluable"][test].mean(axis=1) for p in bank], axis=1)
    tokens = np.stack([data[p]["tokens"][test].mean(axis=1) for p in bank], axis=1)
    per_rollout = np.stack([data[p]["correct"][test] for p in bank], axis=1)
    oracle = per_rollout.max(axis=1).mean()
    champ_idx = bank.index(champion)
    champ = correct[:, champ_idx]
    unique = []
    for k in range(len(bank)):
        others = np.delete(correct, k, axis=1)
        unique.append(float(np.mean((correct[:, k] > 0) & (others.max(axis=1) == 0))))
    distances = []
    ids = [p.rsplit("_", 1)[0] for p in bank if p != "BASELINE"]
    for i in range(len(ids)):
        for j in range(i):
            distances.append(
                float(
                    geometry["A0"][controller_order.index(ids[i]), controller_order.index(ids[j])]
                )
            )
    disagreement = []
    binary = correct >= 0.5
    for i in range(len(bank)):
        for j in range(i):
            disagreement.append(float(np.mean(binary[:, i] != binary[:, j])))
    return {
        "bank": bank,
        "best_single_policy": champion,
        "best_single_accuracy": float(champ.mean()),
        "mean_policy_accuracy": float(correct.mean()),
        "commitment_validity": float(valid.mean()),
        "semantic_evaluability": float(evaluable.mean()),
        "committee_oracle_accuracy": float(oracle),
        "oracle_headroom_over_best_single": float(oracle - champ.mean()),
        "mean_pair_disagreement": float(np.mean(disagreement)) if disagreement else 0.0,
        "unique_correct_fraction_by_policy": unique,
        "mean_generated_tokens": float(tokens.mean()),
        "mean_A0_pair_distance": float(np.mean(distances)) if distances else 0.0,
    }


def cross_fit(
    data: dict[str, dict[str, np.ndarray]],
    item_ids: list[str],
    features: np.ndarray,
    controllers: list[str],
    coords: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outer = balanced_hash_folds(item_ids, 5, "q3-realizable-utility-v1")
    bank_rows: list[dict[str, Any]] = []
    router_accumulator: dict[tuple[str, int, bool, str], dict[str, Any]] = {}
    for fold in range(5):
        train = np.flatnonzero(outer != fold)
        test = np.flatnonzero(outer == fold)
        shell_policies = choose_shells(controllers, data, train)
        all_policies = sorted(data)
        global_train_acc = {p: float(data[p]["correct"][train].mean()) for p in all_policies}
        global_champion = max(
            all_policies,
            key=lambda p: (
                global_train_acc[p],
                float(data[p]["evaluable"][train].mean()),
                -float(data[p]["tokens"][train].mean()),
                p,
            ),
        )
        for method in BANK_METHODS:
            for k in K_VALUES:
                for include_baseline in (False, True):
                    try:
                        bank = select_bank(
                            method,
                            k,
                            include_baseline,
                            controllers,
                            shell_policies,
                            geometry,
                            data,
                            train,
                        )
                    except RuntimeError as exc:
                        bank_rows.append(
                            {
                                "outer_fold": fold,
                                "method": method,
                                "K": k,
                                "baseline_included": include_baseline,
                                "status": "UNAVAILABLE",
                                "reason": str(exc),
                            }
                        )
                        continue
                    train_acc = {p: float(data[p]["correct"][train].mean()) for p in bank}
                    bank_champion = max(
                        bank,
                        key=lambda p: (
                            train_acc[p],
                            float(data[p]["evaluable"][train].mean()),
                            -float(data[p]["tokens"][train].mean()),
                            p,
                        ),
                    )
                    metrics = bank_metrics(bank, data, test, geometry, controllers, bank_champion)
                    metrics.update(
                        {
                            "outer_fold": fold,
                            "method": method,
                            "K": k,
                            "baseline_included": include_baseline,
                        }
                    )
                    bank_rows.append(metrics)
                    y_train = np.stack(
                        [data[p]["correct"][train].mean(axis=1) for p in bank], axis=1
                    )
                    y_test = np.stack([data[p]["correct"][test].mean(axis=1) for p in bank], axis=1)
                    c = policy_coords(bank, coords)
                    tx, vx = standardize(features[train], features[test])
                    for model in MODEL_NAMES:
                        blind = model == "GEOMETRY_BLIND_MATCHED"
                        c_model = policy_coords_from_array(c, len(bank), blind, 2026090402 + fold)
                        nonlinear = model == "GEOMETRY_NONLINEAR_16"
                        lam = select_lambda(
                            features[train],
                            c,
                            y_train,
                            [item_ids[i] for i in train],
                            model,
                            2026090402 + fold,
                        )
                        coef = fit_ridge_router(
                            tx, c_model, y_train, lam, nonlinear, 2026090410 + fold
                        )
                        chosen = predict_router(vx, c_model, coef, nonlinear, 2026090410 + fold)
                        routed = y_test[np.arange(len(test)), chosen]
                        champion_values = data[global_champion]["correct"][test].mean(axis=1)
                        oracle_values = y_test.max(axis=1)
                        valid_values = np.stack(
                            [data[p]["valid"][test].mean(axis=1) for p in bank], axis=1
                        )[np.arange(len(test)), chosen]
                        eval_values = np.stack(
                            [data[p]["evaluable"][test].mean(axis=1) for p in bank], axis=1
                        )[np.arange(len(test)), chosen]
                        token_values = np.stack(
                            [data[p]["tokens"][test].mean(axis=1) for p in bank], axis=1
                        )[np.arange(len(test)), chosen]
                        key = (method, k, include_baseline, model)
                        acc = router_accumulator.setdefault(
                            key,
                            {
                                "routed": [],
                                "champion": [],
                                "champion_valid": [],
                                "champion_evaluable": [],
                                "oracle": [],
                                "valid": [],
                                "evaluable": [],
                                "tokens": [],
                                "fold_gain": [],
                                "lambdas": [],
                                "banks": [],
                            },
                        )
                        acc["routed"].extend(routed.tolist())
                        acc["champion"].extend(champion_values.tolist())
                        acc["champion_valid"].extend(
                            data[global_champion]["valid"][test].mean(axis=1).tolist()
                        )
                        acc["champion_evaluable"].extend(
                            data[global_champion]["evaluable"][test].mean(axis=1).tolist()
                        )
                        acc["oracle"].extend(oracle_values.tolist())
                        acc["valid"].extend(valid_values.tolist())
                        acc["evaluable"].extend(eval_values.tolist())
                        acc["tokens"].extend(token_values.tolist())
                        acc["fold_gain"].append(float((routed - champion_values).mean()))
                        acc["lambdas"].append(lam)
                        acc["banks"].append(bank)
                        acc.setdefault("global_champions", []).append(global_champion)
    router_rows = []
    for key, values in router_accumulator.items():
        method, k, include_baseline, model = key
        routed = np.asarray(values["routed"])
        champion = np.asarray(values["champion"])
        oracle = np.asarray(values["oracle"])
        diff = routed - champion
        headroom = float((oracle - champion).mean())
        router_rows.append(
            {
                "bank_method": method,
                "K": k,
                "baseline_included": include_baseline,
                "mechanism": model,
                "routed_accuracy": float(routed.mean()),
                "development_selected_champion_accuracy": float(champion.mean()),
                "absolute_gain": float(diff.mean()),
                "paired_item_bootstrap_quantiles": paired_interval(diff, 2026090404),
                "oracle_accuracy_within_bank": float(oracle.mean()),
                "oracle_headroom": headroom,
                "fraction_oracle_headroom_realized": float(diff.mean() / headroom)
                if headroom > 0
                else None,
                "commitment_validity": float(np.mean(values["valid"])),
                "semantic_evaluability": float(np.mean(values["evaluable"])),
                "champion_commitment_validity": float(np.mean(values["champion_valid"])),
                "champion_semantic_evaluability": float(np.mean(values["champion_evaluable"])),
                "commitment_validity_harm": float(
                    np.mean(values["champion_valid"]) - np.mean(values["valid"])
                ),
                "semantic_evaluability_harm": float(
                    np.mean(values["champion_evaluable"]) - np.mean(values["evaluable"])
                ),
                "mean_generated_tokens": float(np.mean(values["tokens"])),
                "expected_generations": 1.0,
                "outer_fold_gains": values["fold_gain"],
                "positive_outer_folds": sum(x > 0 for x in values["fold_gain"]),
                "worst_outer_fold_gain": min(values["fold_gain"]),
                "selected_lambdas": values["lambdas"],
                "fold_banks": values["banks"],
                "fold_global_champions": values["global_champions"],
                "outer_folds_present": len(values["fold_gain"]),
            }
        )
    return bank_rows, sorted(
        router_rows,
        key=lambda x: (
            x["outer_folds_present"] == 5,
            x["absolute_gain"],
            x["fraction_oracle_headroom_realized"] or -99,
        ),
        reverse=True,
    )


def aggregate_bank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, bool], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["K"], row["baseline_included"]), []).append(row)
    out = []
    for key, values in groups.items():
        method, k, baseline = key
        complete = [x for x in values if x.get("status") != "UNAVAILABLE"]
        unavailable = [x for x in values if x.get("status") == "UNAVAILABLE"]
        numeric = (
            "best_single_accuracy",
            "mean_policy_accuracy",
            "commitment_validity",
            "semantic_evaluability",
            "committee_oracle_accuracy",
            "oracle_headroom_over_best_single",
            "mean_pair_disagreement",
            "mean_generated_tokens",
            "mean_A0_pair_distance",
        )
        record = {
            "method": method,
            "K": k,
            "baseline_included": baseline,
            "outer_fold_banks": [x["bank"] for x in complete],
            "available_folds": len(complete),
            "unavailable_folds": len(unavailable),
            "selection_failures": [x.get("reason") for x in unavailable],
        }
        if not complete:
            record["status"] = "UNAVAILABLE"
            out.append(record)
            continue
        record["status"] = (
            "COMPLETE" if not unavailable else "PARTIAL_NOT_ELIGIBLE_FOR_ROUTER_SELECTION"
        )
        record.update({name: float(np.mean([x[name] for x in complete])) for name in numeric})
        record["fold_oracle_headroom"] = [x["oracle_headroom_over_best_single"] for x in complete]
        record["minimum_fold_oracle_headroom"] = min(record["fold_oracle_headroom"])
        out.append(record)
    return sorted(
        out,
        key=lambda x: (
            x.get("committee_oracle_accuracy", -1.0),
            x.get("mean_A0_pair_distance", -1.0),
        ),
        reverse=True,
    )


def global_champion(data: dict[str, dict[str, np.ndarray]], train: np.ndarray) -> str:
    policies = sorted(data)
    return max(
        policies,
        key=lambda p: (
            float(data[p]["correct"][train].mean()),
            float(data[p]["evaluable"][train].mean()),
            -float(data[p]["tokens"][train].mean()),
            p,
        ),
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * lam
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def predict_ridge(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coef


def route_b_cross_fit(
    data: dict[str, dict[str, np.ndarray]],
    item_ids: list[str],
    features: np.ndarray,
    controllers: list[str],
    geometry: dict[str, np.ndarray],
) -> dict[str, Any]:
    outer = balanced_hash_folds(item_ids, 5, "q3-realizable-utility-v1")
    routed_all: list[float] = []
    champion_all: list[float] = []
    baseline_all: list[float] = []
    random_all: list[float] = []
    valid_all: list[float] = []
    evaluable_all: list[float] = []
    token_all: list[float] = []
    fold_rows = []
    for fold in range(5):
        train = np.flatnonzero(outer != fold)
        test = np.flatnonzero(outer == fold)
        shell_policies = choose_shells(controllers, data, train)
        bank = select_bank(
            "A0_MAXIMIN", 8, False, controllers, shell_policies, geometry, data, train
        )
        base_train = data["BASELINE"]["correct"][train]
        baseline_wrong = base_train == 0
        alternatives = []
        for policy in bank:
            score = float(data[policy]["correct"][train][baseline_wrong].mean())
            alternatives.append(
                (
                    score,
                    float(data[policy]["correct"][train].mean()),
                    float(data[policy]["evaluable"][train].mean()),
                    -float(data[policy]["tokens"][train].mean()),
                    policy,
                )
            )
        alternative = max(alternatives)[-1]
        champion = global_champion(data, train)

        def cell_features(indices: np.ndarray) -> np.ndarray:
            structural = np.repeat(features[indices], 2, axis=0)
            valid = data["BASELINE"]["valid"][indices].reshape(-1, 1)
            evaluable = data["BASELINE"]["evaluable"][indices].reshape(-1, 1)
            tokens = np.log1p(data["BASELINE"]["tokens"][indices]).reshape(-1, 1)
            return np.column_stack([structural, valid, evaluable, tokens])

        x_train_raw = cell_features(train)
        y_train = (
            data[alternative]["correct"][train] - data["BASELINE"]["correct"][train]
        ).reshape(-1)
        inner = balanced_hash_folds([item_ids[i] for i in train], 4, f"q3-route-b-{fold}")
        candidates = []
        for lam in LAMBDAS:
            for threshold in (0.0, 0.05, 0.10, 0.20):
                utility = []
                invoke = []
                for inner_fold in range(4):
                    tr_items = inner != inner_fold
                    va_items = inner == inner_fold
                    tr_cells = np.repeat(tr_items, 2)
                    va_indices = np.flatnonzero(va_items)
                    va_cells = np.repeat(va_items, 2)
                    tx, vx = standardize(x_train_raw[tr_cells], x_train_raw[va_cells])
                    coef = fit_ridge(tx, y_train[tr_cells], lam)
                    use_alt = predict_ridge(vx, coef) > threshold
                    b = data["BASELINE"]["correct"][train[va_indices]].reshape(-1)
                    a = data[alternative]["correct"][train[va_indices]].reshape(-1)
                    utility.extend(np.where(use_alt, a, b).tolist())
                    invoke.extend(use_alt.tolist())
                candidates.append(
                    (
                        float(np.mean(utility)),
                        -float(np.mean(invoke)),
                        -lam,
                        -threshold,
                        lam,
                        threshold,
                    )
                )
        _, _, _, _, selected_lam, selected_threshold = max(candidates)
        x_test_raw = cell_features(test)
        tx, vx = standardize(x_train_raw, x_test_raw)
        coef = fit_ridge(tx, y_train, selected_lam)
        use_alt = (predict_ridge(vx, coef) > selected_threshold).reshape(len(test), 2)
        baseline_correct = data["BASELINE"]["correct"][test]
        alt_correct = data[alternative]["correct"][test]
        routed = np.where(use_alt, alt_correct, baseline_correct).mean(axis=1)
        champion_values = data[champion]["correct"][test].mean(axis=1)
        random_control = (1 - use_alt.mean()) * baseline_correct.mean(
            axis=1
        ) + use_alt.mean() * alt_correct.mean(axis=1)
        valid = np.where(
            use_alt,
            data[alternative]["valid"][test],
            data["BASELINE"]["valid"][test],
        ).mean(axis=1)
        evaluable = np.where(
            use_alt,
            data[alternative]["evaluable"][test],
            data["BASELINE"]["evaluable"][test],
        ).mean(axis=1)
        total_tokens = (
            data["BASELINE"]["tokens"][test] + use_alt * data[alternative]["tokens"][test]
        ).mean(axis=1)
        fold_rows.append(
            {
                "outer_fold": fold,
                "bank": bank,
                "alternative": alternative,
                "global_champion": champion,
                "lambda": selected_lam,
                "threshold": selected_threshold,
                "invocation_rate": float(use_alt.mean()),
                "routed_accuracy": float(routed.mean()),
                "champion_accuracy": float(champion_values.mean()),
                "gain": float((routed - champion_values).mean()),
            }
        )
        routed_all.extend(routed.tolist())
        champion_all.extend(champion_values.tolist())
        baseline_all.extend(baseline_correct.mean(axis=1).tolist())
        random_all.extend(random_control.tolist())
        valid_all.extend(valid.tolist())
        evaluable_all.extend(evaluable.tolist())
        token_all.extend(total_tokens.tolist())
    routed = np.asarray(routed_all)
    champion = np.asarray(champion_all)
    diff = routed - champion
    invocation_rate = float(np.mean([x["invocation_rate"] for x in fold_rows]))
    return {
        "route": "B",
        "mechanism": "BASELINE_FIRST_RIDGE_ADVANTAGE",
        "routed_accuracy": float(routed.mean()),
        "development_selected_champion_accuracy": float(champion.mean()),
        "absolute_gain": float(diff.mean()),
        "paired_item_bootstrap_quantiles": paired_interval(diff, 2026090420),
        "baseline_accuracy": float(np.mean(baseline_all)),
        "equal_compute_repeated_baseline_expected_accuracy": float(np.mean(baseline_all)),
        "same_rate_random_invocation_expected_accuracy": float(np.mean(random_all)),
        "invocation_rate": invocation_rate,
        "expected_generations": 1.0 + invocation_rate,
        "commitment_validity": float(np.mean(valid_all)),
        "semantic_evaluability": float(np.mean(evaluable_all)),
        "mean_total_generated_tokens": float(np.mean(token_all)),
        "outer_fold_gains": [x["gain"] for x in fold_rows],
        "positive_outer_folds": sum(x["gain"] > 0 for x in fold_rows),
        "worst_outer_fold_gain": min(x["gain"] for x in fold_rows),
        "fold_details": fold_rows,
        "feasible": False,
    }


def plurality_choice(
    data: dict[str, dict[str, np.ndarray]],
    bank: list[str],
    champion: str,
    item_index: int,
    rollout: int,
) -> tuple[float, float, float]:
    answers = []
    for policy in bank:
        if (
            data[policy]["valid"][item_index, rollout] == 1
            and data[policy]["evaluable"][item_index, rollout] == 1
            and data[policy]["answer"][item_index, rollout] is not None
        ):
            answers.append((policy, data[policy]["answer"][item_index, rollout]))
    counts = Counter(answer for _, answer in answers)
    if not counts:
        return 0.0, 0.0, 0.0
    maximum = max(counts.values())
    winners = sorted(answer for answer, count in counts.items() if count == maximum)
    champion_answer = data[champion]["answer"][item_index, rollout]
    chosen = champion_answer if champion_answer in winners else winners[0]
    matching = [policy for policy, answer in answers if answer == chosen]
    correctness = [data[p]["correct"][item_index, rollout] for p in matching]
    if max(correctness) != min(correctness):
        raise RuntimeError("identical typed answers have inconsistent correctness")
    return float(correctness[0]), 1.0, 1.0


def route_c_cross_fit(
    data: dict[str, dict[str, np.ndarray]],
    item_ids: list[str],
    controllers: list[str],
    geometry: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    outer = balanced_hash_folds(item_ids, 5, "q3-realizable-utility-v1")
    results = []
    for k in (2, 4):
        routed_all: list[float] = []
        champion_all: list[float] = []
        valid_all: list[float] = []
        evaluable_all: list[float] = []
        token_all: list[float] = []
        fold_rows = []
        for fold in range(5):
            train = np.flatnonzero(outer != fold)
            test = np.flatnonzero(outer == fold)
            shell_policies = choose_shells(controllers, data, train)
            bank = select_bank(
                "A0_MAXIMIN", k, False, controllers, shell_policies, geometry, data, train
            )
            bank_champion = global_champion({p: data[p] for p in bank}, train)
            champion = global_champion(data, train)
            routed_item = []
            valid_item = []
            evaluable_item = []
            for item in test:
                outcomes = [
                    plurality_choice(data, bank, bank_champion, item, rollout) for rollout in (0, 1)
                ]
                routed_item.append(float(np.mean([x[0] for x in outcomes])))
                valid_item.append(float(np.mean([x[1] for x in outcomes])))
                evaluable_item.append(float(np.mean([x[2] for x in outcomes])))
            routed = np.asarray(routed_item)
            champion_values = data[champion]["correct"][test].mean(axis=1)
            tokens = np.stack([data[p]["tokens"][test].mean(axis=1) for p in bank], axis=1).sum(
                axis=1
            )
            fold_rows.append(
                {
                    "outer_fold": fold,
                    "bank": bank,
                    "bank_tie_break_champion": bank_champion,
                    "global_champion": champion,
                    "gain": float((routed - champion_values).mean()),
                }
            )
            routed_all.extend(routed.tolist())
            champion_all.extend(champion_values.tolist())
            valid_all.extend(valid_item)
            evaluable_all.extend(evaluable_item)
            token_all.extend(tokens.tolist())
        routed = np.asarray(routed_all)
        champion = np.asarray(champion_all)
        diff = routed - champion
        results.append(
            {
                "route": "C",
                "mechanism": "MECHANICAL_TYPED_PLURALITY",
                "K": k,
                "routed_accuracy": float(routed.mean()),
                "development_selected_champion_accuracy": float(champion.mean()),
                "absolute_gain": float(diff.mean()),
                "paired_item_bootstrap_quantiles": paired_interval(diff, 2026090430 + k),
                "expected_generations": float(k),
                "commitment_validity": float(np.mean(valid_all)),
                "semantic_evaluability": float(np.mean(evaluable_all)),
                "mean_total_generated_tokens": float(np.mean(token_all)),
                "outer_fold_gains": [x["gain"] for x in fold_rows],
                "positive_outer_folds": sum(x["gain"] > 0 for x in fold_rows),
                "worst_outer_fold_gain": min(x["gain"] for x in fold_rows),
                "equal_compute_control_status": (
                    "AVAILABLE_WITH_ONLY_ONE_TWO_SAMPLE_BASELINE_REALIZATION"
                )
                if k == 2
                else "UNAVAILABLE_CLOSED_BASELINE_HAS_ONLY_TWO_ROLLOUTS",
                "eligible_for_prelock": False,
                "fold_details": fold_rows,
            }
        )
    return results


def power_grid() -> list[dict[str, Any]]:
    rng = np.random.default_rng(2026090405)
    zcrit = 1.6448536269514722
    rows = []
    for gain in (0.0, 0.01, 0.02, 0.03, 0.05):
        for n in (200, 300, 500, 800, 1200, 1600, 2400):
            for discordance in (0.05, 0.10, 0.20, 0.30, 0.40):
                if gain > discordance:
                    continue
                for rollouts in (1, 2):
                    variance = max(discordance - gain * gain, 1e-9) / (n * rollouts)
                    estimates = rng.normal(loc=gain, scale=math.sqrt(variance), size=100000)
                    z = estimates / math.sqrt(max(discordance, 1e-9) / (n * rollouts))
                    rejection = float(np.mean(z > zcrit))
                    half_width = 1.959963984540054 * math.sqrt(variance)
                    rows.append(
                        {
                            "gain": gain,
                            "N": n,
                            "rollouts": rollouts,
                            "paired_discordance": discordance,
                            "rejection_probability": rejection,
                            "expected_95pct_half_width": half_width,
                            "replicates": 100000,
                            "seed": 2026090405,
                            "method": "paired_normal_score_planning_approximation",
                        }
                    )
    return rows


def runtime_for(n: int, rollouts: int, feature_prefill_fraction: float = 0.10) -> dict[str, float]:
    trajectories = n * rollouts
    base_hours = trajectories * (9.423333333333334 / 19200.0)
    total = base_hours * (1 + feature_prefill_fraction)
    return {
        "semantic_trajectories": trajectories,
        "feature_prefill_equivalent_fraction": feature_prefill_fraction,
        "P50_hours": total,
        "P80_hours": total * (11.05 / 9.76),
        "P95_hours": total * (12.45 / 9.76),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--fresh-scores", type=Path, required=True)
    args = parser.parse_args()
    verify_inputs(args.historical_scores, args.fresh_scores)

    panel = read_json(PANEL)
    item_ids = [str(x["item_id"]) for x in panel["items"]]
    prompts = [str(x["prompt"]) for x in panel["items"]]
    ledger, feasibility = build_exposure_ledger(set(item_ids))
    data, source_counts = load_outcomes(args.historical_scores, args.fresh_scores, item_ids)
    controllers, coords = load_controller_coordinates()
    geometry = combined_geometry()
    features = np.asarray([prompt_features(prompt) for prompt in prompts], dtype=float)
    bank_fold_rows, router_rows = cross_fit(data, item_ids, features, controllers, coords, geometry)
    bank_rows = aggregate_bank_rows(bank_fold_rows)
    route_b = route_b_cross_fit(data, item_ids, features, controllers, geometry)
    route_c = route_c_cross_fit(data, item_ids, controllers, geometry)
    powers = power_grid()

    feasible_router = []
    for row in router_rows:
        fraction = row["fraction_oracle_headroom_realized"]
        if (
            row["outer_folds_present"] == 5
            and row["absolute_gain"] >= 0.03
            and fraction is not None
            and fraction >= 0.25
            and row["positive_outer_folds"] >= 4
            and row["worst_outer_fold_gain"] >= -0.02
            and row["commitment_validity_harm"] <= 0.02
            and row["semantic_evaluability_harm"] <= 0.02
        ):
            feasible_router.append(row)
    preferred = feasible_router[0] if feasible_router else None

    conservative_power = [
        r
        for r in powers
        if r["gain"] == 0.03 and r["paired_discordance"] == 0.20 and r["rollouts"] == 2
    ]
    adequate_n = next(
        (r["N"] for r in conservative_power if r["rejection_probability"] >= 0.80), None
    )
    tier_a_n = feasibility["tier_a"]["available_families"]
    tier_b_n = feasibility["tier_b"]["available_families"]
    feasibility["power_requirement"] = {
        "target_gain": 0.03,
        "rollouts": 2,
        "conservative_paired_discordance": 0.20,
        "minimum_power": 0.80,
        "minimum_N_from_grid": adequate_n,
    }
    feasibility["tier_a_power_adequate"] = tier_a_n >= (adequate_n or 10**9)
    feasibility["tier_b_power_adequate"] = tier_b_n >= (adequate_n or 10**9)
    feasibility["provisional_ruling"] = (
        "Q3_FRESH_HOLDOUT_INSUFFICIENT"
        if not feasibility["tier_b_power_adequate"]
        else "HOLDOUT_POWER_ADEQUATE"
    )

    opportunity = {
        "schema_version": "1.0",
        "labels": LABELS,
        "items": 300,
        "rollouts": 2,
        "policy_population": {
            "baseline": 1,
            "historical_controller_shell_policies": 62,
            "oos_controller_shell_policies": 32,
            "total": 95,
        },
        "source_row_counts": source_counts,
        "source_comparability": {
            "item_order": "MATCH",
            "model_revision": "MATCH",
            "prompt_semantics": "MATCH",
            "parser": "external-semantic-v3_MATCH",
            "error_definition": "MATCH",
            "max_new_tokens": 4096,
            "efficient_repetition_stop": "OOS_ONLY_OPERATIONAL_DIFFERENCE_TERMINAL_AS_ERROR",
        },
        "banks": bank_rows,
        "oracle_is_deployable": False,
        "conclusion": (
            "Opportunity exists only if the listed oracle headroom is positive "
            "and fold-stable; this artifact does not establish selectability or "
            "realization."
        ),
    }
    policy_comparison = {
        "schema_version": "1.0",
        "labels": LABELS,
        "selection_rules": list(BANK_METHODS),
        "K": list(K_VALUES),
        "baseline_options": [False, True],
        "results": bank_rows,
    }
    router_comparison = {
        "schema_version": "1.0",
        "labels": LABELS,
        "routes_evaluated": ["A", "B", "C"],
        "staged_precheck_amendment": "Q3_REALIZABLE_UTILITY_DESIGN_PRECHECK_AMENDMENT_1.json",
        "partial_cross_fit_configurations_are_ineligible": True,
        "route_a_results": router_rows,
        "route_b_result": route_b,
        "route_c_results": route_c,
        "feasible_route_a_count": len(feasible_router),
        "best_feasible_route_a": preferred,
        "route_b_feasible": False,
        "route_c_feasible": False,
    }
    cross_fit_payload = {
        "schema_version": "1.0",
        "labels": LABELS,
        "independent_unit": "CRUXEval semantic problem/item",
        "outer_folds": 5,
        "inner_folds": 4,
        "fold_assignment": "balanced SHA-256 rank",
        "two_rollout_handling": "mean within item-policy",
        "route_a_results": router_rows,
        "route_b_result": route_b,
        "route_c_results": route_c,
    }
    power_payload = {
        "schema_version": "1.0",
        "labels": LABELS,
        "simulation": {"replicates_per_cell": 100000, "seed": 2026090405, "planning_only": True},
        "grid": powers,
        "minimum_N_for_gain_0_03_discordance_0_20_R2": adequate_n,
        "runtime_reference": {"q2_oos_rows": 19200, "wall_hours": 9.423333333333334},
        "route_a_runtime_examples": {
            str(n): runtime_for(n, 2) for n in (tier_a_n, tier_b_n, adequate_n or 1600)
        },
        "storage_assumption_bytes_per_scored_row": 1000,
    }

    if feasibility["provisional_ruling"] == "Q3_FRESH_HOLDOUT_INSUFFICIENT":
        final_state = "Q3_FRESH_HOLDOUT_INSUFFICIENT"
    elif preferred is None:
        final_state = "Q3_NO_REALIZABLE_DEVELOPMENT_SIGNAL"
    else:
        final_state = "Q3_ONE_CALL_ROUTER_READY_FOR_PRELOCK"
    recommendation = {
        "schema_version": "1.0",
        "status": "DRAFT_AWAITING_PRINCIPAL_PRELOCK",
        "labels": LABELS,
        "final_design_ruling": final_state,
        "preferred_mechanism_if_holdout_becomes_adequate": preferred,
        "future_primary_estimand": (
            "mean family-level correctness(router-selected policy - frozen "
            "development-selected champion)"
        ),
        "champion_rule": (
            "Select the highest pooled correctness policy on the complete closed "
            "development panel after all Q3.0 method choices are frozen; ties use "
            "evaluability, lower generated-token mean, then lexicographic policy ID."
        ),
        "future_holdout": {
            "permanently_allocated": False,
            "required_minimum_N": adequate_n,
            "available_tier_a": tier_a_n,
            "available_tier_b": tier_b_n,
        },
        "invalid_unevaluable": "incorrect",
        "future_rollouts": 2,
        "primary_inference": (
            "one-sided paired family-level randomization test plus a "
            "simulation-calibrated paired interval"
        ),
        "safety_guards": {
            "commitment_validity_harm_max": 0.02,
            "semantic_evaluability_harm_max": 0.02,
        },
        "cost": (
            "one policy generation per item plus at most one explicitly counted "
            "prompt-prefill-equivalent feature forward"
        ),
        "multiplicity": (
            "one primary router-versus-champion contrast; Holm within declared secondary family"
        ),
        "missingness": "blocks completion; no imputation",
        "matched_random_subspace_specificity": "unresolved and outside the Q3 claim",
        "execution_authorized": False,
    }

    write_json(OUT / "ITEM_EXPOSURE_LEDGER.json", ledger)
    write_json(OUT / "FRESH_HOLDOUT_FEASIBILITY.json", feasibility)
    write_json(OUT / "OPPORTUNITY_AUDIT.json", opportunity)
    write_json(OUT / "POLICY_BANK_COMPARISON.json", policy_comparison)
    write_json(OUT / "ROUTER_MECHANISM_COMPARISON.json", router_comparison)
    write_json(OUT / "CROSS_FIT_RESULTS.json", cross_fit_payload)
    write_json(OUT / "POWER_AND_COMPUTE.json", power_payload)
    write_json(OUT / "RECOMMENDED_PROTOCOL_DRAFT.json", recommendation)
    tracked_artifacts = [
        "README.md",
        "review/q3_realizable_utility_design/Q3_REALIZABLE_UTILITY_DESIGN_PRECHECK.json",
        "review/q3_realizable_utility_design/Q3_REALIZABLE_UTILITY_DESIGN_PRECHECK_AMENDMENT_1.json",
        "review/q3_realizable_utility_design/ITEM_EXPOSURE_LEDGER.json",
        "review/q3_realizable_utility_design/FRESH_HOLDOUT_FEASIBILITY.json",
        "review/q3_realizable_utility_design/OPPORTUNITY_AUDIT.json",
        "review/q3_realizable_utility_design/POLICY_BANK_COMPARISON.json",
        "review/q3_realizable_utility_design/ROUTER_MECHANISM_COMPARISON.json",
        "review/q3_realizable_utility_design/CROSS_FIT_RESULTS.json",
        "review/q3_realizable_utility_design/POWER_AND_COMPUTE.json",
        "review/q3_realizable_utility_design/RECOMMENDED_PROTOCOL_DRAFT.json",
        "docs/Q3_REALIZABLE_UTILITY_DESIGN_REVIEW.md",
        "docs/Q3_FEATURE_FIREWALL.md",
        "docs/Q3_RELATED_WORK_AND_DESIGN_PRIORS.md",
        "docs/Q3_CONCEPT_NOTE.md",
        "docs/CURRENT_STATUS.md",
        "docs/DOCUMENT_INDEX.md",
        "docs/EXPERIMENT_INDEX.md",
        "docs/SCIENTIFIC_RESULTS.md",
        "experiments/registry.yaml",
        "project_state.yaml",
        "scripts/render_project_state.py",
        "scripts/design_q3_realizable_utility.py",
        "tests/test_q3_realizable_utility_design.py",
    ]
    manifest = {
        "schema_version": "1.0",
        "classification": final_state,
        "labels": LABELS,
        "artifacts": {path: sha256_file(ROOT / path) for path in tracked_artifacts},
        "private_development_sources": {
            "Q2_V4_1_SEMANTIC_SCORES.jsonl": EXPECTED["historical_scores"],
            "Q2_OOS_V2_SEMANTIC_SCORES.jsonl": EXPECTED["fresh_scores"],
        },
        "raw_text_included": False,
        "future_holdout_outcomes_included": False,
        "q3_semantic_trajectories": 0,
    }
    write_json(OUT / "ARTIFACT_HASHES.json", manifest)
    print(
        json.dumps(
            {
                "status": final_state,
                "preferred": preferred,
                "adequate_N": adequate_n,
                "tier_a": tier_a_n,
                "tier_b": tier_b_n,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
