#!/usr/bin/env python3
"""Prepare the Q2 V4.1 presemantic prediction lock.

This program is deliberately outcome-free.  It consumes only the immutable V4
coefficient/safety records and the already frozen, label-free M1/M2/panel
manifests.  The G3 section is synthetic planning only; it never imports a
model runner or a semantic parser.
"""

from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(ROOT / "scripts"))

from design_q2_v4_intervention_subspace import (  # noqa: E402
    _angular,
    _candidate_embeddings,
    _finite_specific_embeddings,
    _normalized_ranks,
    _qap_cache,
    _sigmoid,
    _unit_rows,
)

from epistemic_geometry.experiments.q2_v4 import (  # noqa: E402
    controller_permutations,
    protocol_seed,
    spearman,
)
from epistemic_geometry.experiments.q2_v4_1 import (  # noqa: E402
    EXPECTED_SAFE_IDS,
    V4_CANDIDATE_COMMIT,
    V4_CLASSIFICATION,
    V4_FINAL_COMMIT,
    V4_PRELOCK,
    bank_geometry,
    load_frozen_candidates,
    sha256_file,
)

REVIEW = ROOT / "review/q2_v4_1_prediction_lock"
V4_REVIEW = ROOT / "review/q2_v4_spark1_presemantic"
OLD_REVIEW = ROOT / "review/q2_v4_1_31_safe_bank_review"
CANDIDATES = V4_REVIEW / "CANDIDATE_BANK_MANIFEST.json"
SAFETY = V4_REVIEW / "CANDIDATE_SAFETY_REPORT.json"
M1_SOURCE = V4_REVIEW / "M1_COVARIANCE_MANIFEST.json"
M2_SOURCE = V4_REVIEW / "M2_PROBE_MANIFEST.json"
PANEL_SOURCE = V4_REVIEW / "PRIMARY_PANEL_MANIFEST.json"
SHELL_RESULT = V4_REVIEW / "SHELL_CALIBRATION_MANIFEST_RESULT.json"
V4_ENVIRONMENT = V4_REVIEW / "SPARK1_ENVIRONMENT_LOCK.json"
SAFE_MANIFEST = OLD_REVIEW / "SAFE_31_IMMUTABLE_MANIFEST.json"
V4_AUDIT = V4_REVIEW / "SAFETY_FORENSIC_AUDIT.json"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
LAYER = 27
SHELLS = ("MEDIUM", "STRONG")
SHELL_TARGETS = {"MEDIUM": 0.25, "STRONG": 0.50}
N = 300
K = 31
QAP_MAPS = 50_000
G3_MARGIN = 0.10
G3_DELTAS = (0.0, 0.05, 0.10, 0.15, 0.20)
G3_REPLICATES = 500
G3_BOOTSTRAP_RESAMPLES = 32
G3_PLANNING_QAP_MAPS = 499


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def wide_seed(namespace: str, *parts: str | int) -> int:
    payload = "\x1f".join((namespace, *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def safe_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, safe = load_frozen_candidates(CANDIDATES, SAFETY)
    if [row["candidate_id"] for row in safe] != list(EXPECTED_SAFE_IDS):
        raise RuntimeError("immutable safe-bank order changed")
    if (
        sha256_file(SAFE_MANIFEST)
        != "a641d612628c4f9eff2ae9fdf12d3ad17af5a3e921ec726d31c208ee5e030447"
    ):
        raise RuntimeError("immutable V4.1 safe-bank manifest hash changed")
    return candidates, safe


def write_safety_erratum(safe: list[dict[str, Any]]) -> None:
    safety = read_json(SAFETY)
    d38 = safety["candidates"]["V4_DIRECTION_38"]
    expected = {
        "MEDIUM": {
            "pass": False,
            "validity": 1.0,
            "evaluability": 1.0,
            "raw_sequence_movement": 0.08333333333333333,
            "truncation": 0.0,
        },
        "STRONG": {
            "pass": False,
            "validity": 0.9166666666666666,
            "evaluability": 0.9166666666666666,
            "raw_sequence_movement": 0.6666666666666666,
            "truncation": 0.0,
        },
    }
    for shell, fields in expected.items():
        for key, value in fields.items():
            if not math.isclose(
                float(d38["shells"][shell][key]), float(value), rel_tol=0, abs_tol=1e-15
            ):
                raise RuntimeError(f"immutable D38 record differs at {shell}.{key}")
    if bool(d38["both_shells_pass"]):
        raise RuntimeError("D38 unexpectedly safe")
    value = {
        "schema_version": "q2-v4.1-safety-history-erratum-v1",
        "classification": "DOCUMENTATION_TYPO_CONFIRMED",
        "authority": {
            "candidate_safety_sha256": sha256_file(SAFETY),
            "v4_forensic_audit_sha256": sha256_file(V4_AUDIT),
            "safe_manifest_sha256": sha256_file(SAFE_MANIFEST),
            "raw_artifacts_modified": False,
        },
        "candidate_id": "V4_DIRECTION_38",
        "raw_truth": {
            "medium_pass": False,
            "strong_pass": False,
            "both_shells_pass": False,
            "medium_reason_labels": ["raw_sequence_movement_below_0.10"],
            "strong_reason_labels": [
                "validity_below_relative_0.05_guard",
                "evaluability_below_relative_0.05_guard",
            ],
            "truncation_at_both_shells": False,
        },
        "reason_code_note": (
            "The immutable V4 JSON has no normalized reason-code field; labels "
            "above are this erratum's deterministic normalization."
        ),
        "v4_1_manifest_unaffected": [row["candidate_id"] for row in safe]
        == list(EXPECTED_SAFE_IDS),
    }
    write_json(REVIEW / "SAFETY_HISTORY_ERRATUM.json", value)
    (REVIEW / "SAFETY_HISTORY_ERRATUM.md").write_text(
        "# V4 safety-history erratum: V4_DIRECTION_38\n\n"
        "The immutable safety JSON and journal resolve the narrative discrepancy. "
        "V4_DIRECTION_38 fails both shells: MEDIUM raw sequence movement is "
        "0.08333333333333333 (<0.10), while STRONG validity/evaluability are "
        "0.9166666666666666 and the relative-drop guard fails. Neither shell is "
        "truncated. The raw V4 artifacts were not changed, and the 31-safe "
        "manifest is byte-for-byte unaffected. Normalized reason labels here are "
        "an addendum, not historical reason-code replacements.\n\n"
        "Classification: `DOCUMENTATION_TYPO_CONFIRMED`.\n",
        encoding="utf-8",
    )


def write_leverage_ruling(safe: list[dict[str, Any]]) -> None:
    coefficients = np.asarray([row["coefficients"] for row in safe], dtype=np.float64)
    geometry = bank_geometry(coefficients)
    value = {
        "schema_version": "q2-v4.1-leverage-ruling-v1",
        "classification": "LEVERAGE_DESCRIPTIVE_ONLY",
        "applicable_frozen_controller_level_threshold": None,
        "v3_family_leverage_threshold_transfer": False,
        "reason": (
            "V4/V4.1 froze no controller-level leverage gate; the V3 family-level "
            "concept is not the same estimand."
        ),
        "safe_bank_leverage": geometry["leverage"],
        "safe_bank_leverage_values": geometry["leverage_values"],
        "semantic_outcomes": 0,
        "selection_used": False,
    }
    write_json(REVIEW / "LEVERAGE_RULING.json", value)
    (REVIEW / "LEVERAGE_RULING.md").write_text(
        "# V4.1 leverage ruling\n\n"
        "No prospectively applicable V4/V4.1 controller-level leverage threshold "
        "exists. The V3 ~0.40 quantity was a family-level diagnostic and is not "
        "transferred. The safe-bank maximum is reported descriptively only.\n\n"
        "Classification: `LEVERAGE_DESCRIPTIVE_ONLY`.\n",
        encoding="utf-8",
    )


def _shape_matrix(errors: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(errors, dtype=np.float64)
    if indices is not None:
        values = values[:, indices, :]
    count = values.shape[1]
    d0 = values[:, None, :, 0] - values[None, :, :, 0]
    d1 = values[:, None, :, 1] - values[None, :, :, 1]
    total = np.mean(d0 * d1, axis=2)
    shifted = np.mean(d0, axis=2) * np.mean(d1, axis=2)
    result = (total - shifted) * count / (count - 1.0)
    np.fill_diagonal(result, 0.0)
    return result


def _edge(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], 1)]


def _target_embedding(
    metrics: dict[str, np.ndarray], seed: int, requested_delta: float
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    # Delta=0 is a genuine no-superiority planning null. It must not be
    # represented by an A2-anchored target whose geometry can make A2 the
    # best metric before any simulated outcomes exist.
    anchor_name = "A0" if requested_delta == 0.0 else "A2"
    anchor = metrics[anchor_name]
    nuisance = _unit_rows(rng.standard_normal(anchor.shape))
    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []
    for weight in np.linspace(0.0, 1.0, 201):
        target = _unit_rows(
            np.concatenate([np.sqrt(weight) * anchor, np.sqrt(1.0 - weight) * nuisance], axis=1)
        )
        target_geometry = _angular(target)
        rhos = {
            name: float(spearman(_edge(_angular(value)), _edge(target_geometry)))
            for name, value in metrics.items()
        }
        achieved_delta = rhos["A2"] - max(rhos["A0"], rhos["A1"])
        if requested_delta == 0.0:
            # Prefer a non-positive achieved delta near the common rho_A2=.25
            # planning scale. Positive deltas are strongly penalized so the
            # Delta=0 row is not mislabeled as a superiority alternative.
            score = abs(rhos["A2"] - 0.25) + abs(achieved_delta) + 100.0 * max(0.0, achieved_delta)
        else:
            score = abs(rhos["A2"] - 0.25) + 2.0 * abs(achieved_delta - requested_delta)
        candidates.append((score, target, {**rhos, "delta": achieved_delta, "weight": weight}))
    candidates.sort(key=lambda entry: entry[0])
    _score, target, achieved = candidates[0]
    achieved["target_anchor"] = anchor_name
    return target, achieved


def _bootstrap_superiority(
    errors: np.ndarray,
    metric_edges: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = []
    count = errors.shape[1]
    for _ in range(G3_BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, count, size=count)
        shape = _shape_matrix(errors, indices)
        outcome = _normalized_ranks(_edge(shape))
        rhos = {name: float(np.dot(edge, outcome)) for name, edge in metric_edges.items()}
        values.append(
            {
                "a2_minus_a0": rhos["A2"] - rhos["A0"],
                "a2_minus_a1": rhos["A2"] - rhos["A1"],
            }
        )
    margin = [row["a2_minus_a0"] >= G3_MARGIN and row["a2_minus_a1"] >= G3_MARGIN for row in values]
    positive = [row["a2_minus_a0"] > 0 and row["a2_minus_a1"] > 0 for row in values]
    return {
        "resamples": G3_BOOTSTRAP_RESAMPLES,
        "margin_support_fraction": float(np.mean(margin)),
        "positive_support_fraction": float(np.mean(positive)),
        "margin_supported": bool(np.mean(margin) >= 0.95),
        "positive_supported": bool(np.mean(positive) >= 0.95),
    }


def _simulate_g3_once(
    metrics: dict[str, np.ndarray],
    target: np.ndarray,
    qap_cache: np.ndarray,
    metric_names: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    size = len(target)
    features = rng.standard_normal((N, target.shape[1]))
    base_logit = rng.normal(0.0, 0.75, size=N)
    response = target @ features.T
    probabilities = {
        "MEDIUM": _sigmoid(base_logit[None, :] + 0.75 * response),
        "STRONG": _sigmoid(base_logit[None, :] + 1.15 * response),
    }
    errors_by_shell = {
        shell: (rng.random((size, N, 2)) < probabilities[shell][:, :, None]).astype(np.float64)
        for shell in SHELLS
    }
    shell_ranks = []
    for shell in SHELLS:
        shell_ranks.append(_normalized_ranks(_edge(_shape_matrix(errors_by_shell[shell]))))
    outcome_rank = np.mean(np.stack(shell_ranks), axis=0)
    null = np.einsum("mpe,e->pm", qap_cache, outcome_rank)
    observed = null[0]
    max_null = np.max(null, axis=1)
    adjusted = np.asarray([np.mean(max_null >= value) for value in observed])
    indices = {name: metric_names.index(name) for name in ("A0", "A1", "A2")}
    differences = np.asarray(
        [
            observed[indices["A2"]] - observed[indices["A0"]],
            observed[indices["A2"]] - observed[indices["A1"]],
        ]
    )
    difference_null = np.stack(
        [
            null[:, indices["A2"]] - null[:, indices["A0"]],
            null[:, indices["A2"]] - null[:, indices["A1"]],
        ],
        axis=1,
    )
    max_difference_null = np.max(difference_null, axis=1)
    superiority_adjusted = np.asarray(
        [np.mean(max_difference_null >= value) for value in differences]
    )
    metric_edges = {
        name: _normalized_ranks(_edge(_angular(value))) for name, value in metrics.items()
    }
    bootstrap = _bootstrap_superiority(errors_by_shell["MEDIUM"], metric_edges, rng)
    bootstrap_strong = _bootstrap_superiority(errors_by_shell["STRONG"], metric_edges, rng)
    bootstrap_support = bool(bootstrap["margin_supported"] and bootstrap_strong["margin_supported"])
    a2_index = indices["A2"]
    a2_qualifies = bool(observed[a2_index] > 0.0 and adjusted[a2_index] <= 0.05)
    effect = bool(np.all(differences >= G3_MARGIN))
    permutation_support = bool(np.all(superiority_adjusted <= 0.05))
    return {
        "a2_qualifies": a2_qualifies,
        "both_effect_margin": effect,
        "corrected_permutation_support": permutation_support,
        "paired_bootstrap_support": bootstrap_support,
        "g3_classification": bool(
            a2_qualifies and effect and permutation_support and bootstrap_support
        ),
        "a2_observed_rho": float(observed[a2_index]),
        "a0_observed_rho": float(observed[indices["A0"]]),
        "a1_observed_rho": float(observed[indices["A1"]]),
        "a2_minus_a0": float(differences[0]),
        "a2_minus_a1": float(differences[1]),
        "a2_superiority_p_a0": float(superiority_adjusted[0]),
        "a2_superiority_p_a1": float(superiority_adjusted[1]),
        "bootstrap_medium": bootstrap,
        "bootstrap_strong": bootstrap_strong,
    }


def g3_power() -> None:
    coefficients = np.asarray([row["coefficients"] for row in safe_rows()[1]], dtype=np.float64)
    seed = protocol_seed("Q2-V4.1-G3-PLANNING-BANK-K31", V4_FINAL_COMMIT)
    ordinary = _candidate_embeddings(coefficients, seed ^ 0xA2A2)
    specific = _finite_specific_embeddings(coefficients, seed ^ 0xA2A2)
    metrics = {"A0": coefficients, "A1": ordinary["A1"], "A2": specific["A2"]}
    geometries = {name: _angular(value) for name, value in metrics.items()}
    permutations = controller_permutations(
        K,
        G3_PLANNING_QAP_MAPS,
        seed=protocol_seed("Q2-V4.1-G3-QAP-K31", V4_FINAL_COMMIT),
    )
    metric_names, qap_cache = _qap_cache(geometries, permutations)
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for delta in G3_DELTAS:
        target, achieved = _target_embedding(metrics, seed ^ int(delta * 10_000) ^ 0xD3, delta)
        results = [
            _simulate_g3_once(
                metrics,
                target,
                qap_cache,
                metric_names,
                wide_seed("Q2-V4.1-G3-SIM", V4_FINAL_COMMIT, "K31", N, delta, index),
            )
            for index in range(G3_REPLICATES)
        ]
        row = {
            "K": K,
            "N": N,
            "requested_delta": delta,
            "achieved_true_rho_A0": achieved["A0"],
            "achieved_true_rho_A1": achieved["A1"],
            "achieved_true_rho_A2": achieved["A2"],
            "achieved_true_delta": achieved["delta"],
            "target_mixture_weight": achieved["weight"],
            "replicates": G3_REPLICATES,
            "planning_qap_maps": G3_PLANNING_QAP_MAPS,
            "a2_qualification_probability": float(np.mean([r["a2_qualifies"] for r in results])),
            "effect_margin_probability": float(np.mean([r["both_effect_margin"] for r in results])),
            "corrected_permutation_support_probability": float(
                np.mean([r["corrected_permutation_support"] for r in results])
            ),
            "paired_bootstrap_support_probability": float(
                np.mean([r["paired_bootstrap_support"] for r in results])
            ),
            "g3_classification_probability": float(
                np.mean([r["g3_classification"] for r in results])
            ),
            "fpr_at_delta_zero": float(np.mean([r["g3_classification"] for r in results]))
            if delta == 0
            else None,
            "observed_a2_rho_mean": float(np.mean([r["a2_observed_rho"] for r in results])),
            "observed_superiority_margin_mean": float(
                np.mean([min(r["a2_minus_a0"], r["a2_minus_a1"]) for r in results])
            ),
            "monte_carlo_se_g3": math.sqrt(
                max(0.0, float(np.mean([r["g3_classification"] for r in results])))
                * max(0.0, 1.0 - float(np.mean([r["g3_classification"] for r in results])))
                / G3_REPLICATES
            ),
        }
        rows.append(row)
        detail[str(delta)] = {"achieved": achieved, "replicates": results}
    write_csv(REVIEW / "G3_POWER_CHARACTERIZATION.csv", rows)
    write_json(
        REVIEW / "G3_POWER_CHARACTERIZATION.json",
        {
            "schema_version": "q2-v4.1-g3-power-characterization-v1",
            "cpu_only": True,
            "semantic_outcomes": 0,
            "correctness_inspected": False,
            "K": K,
            "N": N,
            "delta_grid": list(G3_DELTAS),
            "base_relational_target": 0.25,
            "superiority_margin": G3_MARGIN,
            "qap": {
                "planning_maps": G3_PLANNING_QAP_MAPS,
                "final_maps_preserved": QAP_MAPS,
                "same_permutation_across_shells": True,
                "maxT": True,
                "permutation_unit": "controller label",
            },
            "bootstrap": {
                "resamples_per_synthetic_replicate": G3_BOOTSTRAP_RESAMPLES,
                "unit": "item cluster",
                "support_rule": (
                    "both A2-minus-A0 and A2-minus-A1 >= 0.10 in at least 95% of "
                    "paired item resamples, separately per shell"
                ),
            },
            "method": (
                "fixed synthetic target embedding selected on a predeclared weight "
                "grid to approximate rho_A2=0.25 and requested Delta; no observed "
                "bank or semantic outcome enters"
            ),
            "rows": rows,
            "detail": detail,
        },
    )


def copy_label_free_manifests() -> None:
    for source, target_name, allocation in (
        (M1_SOURCE, "A1_COVARIANCE_MANIFEST.json", "M1_COVARIANCE"),
        (M2_SOURCE, "A2_PROBE_MANIFEST.json", "M2_LABEL_FREE_PROBES"),
    ):
        value = read_json(source)
        value["status"] = "FROZEN_PRESEMANTIC_V4_1"
        value["allocation"] = allocation
        value["v4_1_review"] = True
        value["outcome_values_read_or_used"] = False
        value["inherited_source_path"] = str(source.relative_to(ROOT))
        value["inherited_source_sha256"] = sha256_file(source)
        write_json(REVIEW / target_name, value)


def copy_semantic_panel() -> dict[str, Any]:
    source = read_json(PANEL_SOURCE)
    if source["item_count"] != N or source["semantic_outcomes"] != 0:
        raise RuntimeError("historical V4 panel is not a clean future panel")
    panel = dict(source)
    panel["schema_version"] = "q2-v4.1-primary-panel-v1"
    panel["status"] = "FROZEN_CONTENT_NOT_AUTHORIZED_FOR_INFERENCE"
    panel["v4_1_namespace"] = "Q2-V4.1-SEMANTIC-PANEL-V1"
    panel["semantic_outcomes"] = 0
    panel["correctness_inspected"] = False
    write_json(REVIEW / "SEMANTIC_PANEL_MANIFEST.json", panel)
    return panel


def make_deployment(safe: list[dict[str, Any]]) -> dict[str, Any]:
    value = read_json(SHELL_RESULT)
    controllers = value["controllers"]
    deployment = {}
    for row in safe:
        name = row["candidate_id"]
        for shell in SHELLS:
            key = f"{name}_{shell}"
            if key not in controllers:
                raise RuntimeError(f"missing immutable shell deployment: {key}")
            deployment[key] = controllers[key]
            if deployment[key]["vector_hash"] != row["canonical_vector_hash"]:
                raise RuntimeError(f"deployment/vector mismatch: {key}")
    return deployment


def make_schedule(panel: dict[str, Any], deployment: dict[str, Any]) -> dict[str, Any]:
    conditions = ["BASELINE"] + [
        f"{name}_{shell}" for name in EXPECTED_SAFE_IDS for shell in SHELLS
    ]
    rows = []
    for item_id in panel["item_ids"]:
        for rollout in (0, 1):
            order_rng = np.random.Generator(
                np.random.PCG64DXSM(
                    wide_seed("Q2-V4.1-CONDITION-ORDER", V4_PRELOCK, item_id, rollout)
                )
            )
            for order, condition in enumerate(order_rng.permutation(conditions).tolist()):
                controller = None if condition == "BASELINE" else condition
                rows.append(
                    {
                        "item_id": item_id,
                        "condition": condition,
                        "rollout_index": rollout,
                        "condition_order": order,
                        "seed": wide_seed(
                            "Q2-V4.1-INDEPENDENT-PRIMARY", V4_PRELOCK, item_id, condition, rollout
                        ),
                        "prompt_sha256": next(
                            item["prompt_sha256"]
                            for item in panel["items"]
                            if item["item_id"] == item_id
                        ),
                        "reference_type": "CRUXEval.output",
                        "controller_vector_hash": None
                        if controller is None
                        else deployment[controller]["vector_hash"],
                        "alpha": 0.0 if controller is None else deployment[controller]["alpha"],
                        "layer": None if controller is None else LAYER,
                        "duration": "none" if controller is None else "sustained_current_token",
                    }
                )
    keys = {(row["item_id"], row["condition"], row["rollout_index"]) for row in rows}
    seeds = [row["seed"] for row in rows]
    if len(rows) != 63 * N * 2 or len(keys) != len(rows) or len(seeds) != len(set(seeds)):
        raise RuntimeError("V4.1 future schedule cardinality or seed collision")
    result = {
        "schema_version": "q2-v4.1-future-semantic-schedule-v1",
        "status": "FROZEN_NOT_AUTHORIZED_NOT_RUN",
        "namespace": "Q2-V4.1-SEMANTIC-PANEL-V1",
        "prelock_source": V4_PRELOCK,
        "item_count": N,
        "condition_count": 63,
        "rollouts": 2,
        "row_count": len(rows),
        "unique_logical_keys": len(keys),
        "unique_seeds": len(set(seeds)),
        "conditions": conditions,
        "controller_order": list(EXPECTED_SAFE_IDS),
        "semantic_outcomes": 0,
        "rows": rows,
    }
    write_json(REVIEW / "FUTURE_SEMANTIC_SCHEDULE.json", result)
    return result


def make_qap() -> None:
    seed = protocol_seed("Q2-V4-QAP-V1-K31", V4_PRELOCK)
    permutations = controller_permutations(K, QAP_MAPS, seed=seed)
    if not np.array_equal(permutations[0], np.arange(K)):
        raise RuntimeError("QAP identity is not first")
    if len({row.tobytes() for row in permutations}) != QAP_MAPS:
        raise RuntimeError("QAP permutation uniqueness failure")
    np.save(REVIEW / "QAP_CONTROLLER_PERMUTATIONS.npy", permutations)
    write_json(
        REVIEW / "QAP_SCHEDULE.json",
        {
            "schema_version": "q2-v4.1-qap-v1",
            "maps": QAP_MAPS,
            "controller_count": K,
            "identity_first": True,
            "unique_maps": QAP_MAPS,
            "seed": str(seed),
            "rng": "NumPy PCG64DXSM",
            "seed_source": "Q2-V4-QAP-V1-K31|V4_PRELOCK",
            "same_permutation_across_shells_and_A0_A1_A2": True,
            "p_value": "count(T_perm>=T_observed)/50000",
            "multiplicity": "single-step maxT across A0/A1/A2",
            "array_sha256": sha256_file(REVIEW / "QAP_CONTROLLER_PERMUTATIONS.npy"),
        },
    )


def write_a0(safe: list[dict[str, Any]]) -> None:
    coefficients = np.asarray([row["coefficients"] for row in safe], dtype=np.float64)
    a0 = 1.0 - coefficients @ coefficients.T
    np.fill_diagonal(a0, 0.0)
    np.save(REVIEW / "A0_MEDIUM.npy", a0)
    np.save(REVIEW / "A0_STRONG.npy", a0)
    write_json(
        REVIEW / "A0_METADATA.json",
        {
            "definition": "coordinate-space angular dissimilarity 1-cosine",
            "controller_order": list(EXPECTED_SAFE_IDS),
            "shell_invariance_expected": True,
            "medium_array_sha256": sha256_file(REVIEW / "A0_MEDIUM.npy"),
            "strong_array_sha256": sha256_file(REVIEW / "A0_STRONG.npy"),
            "cross_shell_max_absolute_difference": 0.0,
            "semantic_outcomes": 0,
        },
    )


def write_protocol_lock(
    source_commit: str,
    safe: list[dict[str, Any]],
    deployment: dict[str, Any],
    panel: dict[str, Any],
    schedule: dict[str, Any],
) -> None:
    source_env = read_json(V4_ENVIRONMENT)
    value = {
        "schema_version": "q2-v4.1-prediction-lock-protocol-v1",
        "status": "Q2_V4_1_PRESEMANTIC_PROTOCOL_LOCKED",
        "source_commit": source_commit,
        "historical_v4_prelock": V4_PRELOCK,
        "historical_v4_candidate_bank_commit": V4_CANDIDATE_COMMIT,
        "historical_v4_final_commit": V4_FINAL_COMMIT,
        "historical_v4_classification": V4_CLASSIFICATION,
        "v4_1_adequacy": "Q2_V4_1_31_SAFE_BANK_ADEQUATE",
        "backend": "V4_NATIVE_SPARK1",
        "spark1_only": True,
        "spark2_forbidden": True,
        "runpod_forbidden": True,
        "max_gb10": 1,
        "expected_environment_fingerprint": source_env["fingerprint_sha256"],
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "BF16",
        "attention": "SDPA",
        "layer": LAYER,
        "controller_count": K,
        "controller_order": list(EXPECTED_SAFE_IDS),
        "controller_population": (
            "original isotropic V4 candidates conditioned on frozen two-shell safety eligibility"
        ),
        "shells": SHELL_TARGETS,
        "deployment": deployment,
        "A0": {
            "definition": "coordinate-space angular dissimilarity",
            "status": "MATERIALIZED_OUTCOME_FREE",
        },
        "A1": {
            "definition": "regularized covariance-whitened angular dissimilarity",
            "lambda": 0.10,
            "status": "TO_BE_MATERIALIZED_AFTER_LOCK",
        },
        "A2": {
            "definition": "baseline-centered natural-log full-vocabulary JS response angle",
            "status": "TO_BE_MATERIALIZED_AFTER_LOCK",
        },
        "D2": {
            "definition": "finite response total distance",
            "status": "TO_BE_MATERIALIZED_AFTER_LOCK",
        },
        "endpoint": "N/(N-1)*(Dtotal - m0*m1); negative estimates retained",
        "panel": {
            "manifest": "SEMANTIC_PANEL_MANIFEST.json",
            "N": N,
            "semantic_outcomes": 0,
            "status": panel["status"],
        },
        "schedule": {
            "manifest": "FUTURE_SEMANTIC_SCHEDULE.json",
            "rows": schedule["row_count"],
            "conditions": 63,
            "rollouts": 2,
        },
        "QAP": {
            "manifest": "QAP_SCHEDULE.json",
            "maps": QAP_MAPS,
            "identity_first": True,
            "same_map_across_shells_and_metrics": True,
            "maxT": True,
        },
        "bootstrap": {
            "item_cluster_resamples": 10_000,
            "seed": wide_seed("Q2-V4.1-BOOTSTRAP", V4_PRELOCK),
        },
        "G3": {
            "margin": G3_MARGIN,
            "characterization": "G3_POWER_CHARACTERIZATION.json",
            "no_redesign": True,
        },
        "leverage": "LEVERAGE_DESCRIPTIVE_ONLY",
        "semantic_execution_authorized": False,
        "semantic_outcomes": 0,
        "correctness_inspected": False,
        "Q3": "NOT_RUN",
        "Q2": "UNTESTED",
        "artifact_hashes": {
            name: sha256_file(REVIEW / name)
            for name in (
                "PREMORTEM.md",
                "PREMORTEM.json",
                "SAFETY_HISTORY_ERRATUM.md",
                "SAFETY_HISTORY_ERRATUM.json",
                "LEVERAGE_RULING.md",
                "LEVERAGE_RULING.json",
                "G3_POWER_CHARACTERIZATION.csv",
                "G3_POWER_CHARACTERIZATION.json",
                "A1_COVARIANCE_MANIFEST.json",
                "A2_PROBE_MANIFEST.json",
                "SEMANTIC_PANEL_MANIFEST.json",
                "FUTURE_SEMANTIC_SCHEDULE.json",
                "QAP_CONTROLLER_PERMUTATIONS.npy",
                "QAP_SCHEDULE.json",
                "A0_MEDIUM.npy",
                "A0_STRONG.npy",
                "A0_METADATA.json",
                "PREPARATION_AUDIT.json",
            )
        },
        "safe_bank_manifest": {
            "path": str(SAFE_MANIFEST.relative_to(ROOT)),
            "sha256": sha256_file(SAFE_MANIFEST),
        },
        "spec_sha256": sha256_file(ROOT / "experiments/specs/q2_v4_1_prediction_lock.yaml"),
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", value)
    (REVIEW / "PROTOCOL_LOCK.md").write_text(
        "# Q2 V4.1 final presemantic protocol lock\n\n"
        "Status: `Q2_V4_1_PRESEMANTIC_PROTOCOL_LOCKED`. This commit freezes "
        "the immutable 31-safe bank, inherited shell deployments, label-free "
        "M1/M2 manifests, A0, 300-item future panel, 63-condition/37,800-row "
        "future schedule, QAP, estimands, bootstrap, and G3 characterization. "
        "A1/A2 will be materialized on Spark 1 only after this lock. Semantic "
        "execution is not authorized and semantic outcomes remain zero.\n",
        encoding="utf-8",
    )


def write_spec(source_commit: str) -> None:
    path = ROOT / "experiments/specs/q2_v4_1_prediction_lock.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "schema_version: 1\n"
        "experiment_id: Q2_V4_1_PREDICTION_LOCK\n"
        "status: PROSPECTIVE_PRESEMANTIC_LOCK\n"
        "stage: DEVELOPMENT_PRESEMANTIC_GEOMETRY_FREEZE\n"
        "source_commit: " + source_commit + "\n"
        "backend: V4_NATIVE_SPARK1\n"
        "spark1_only: true\n"
        "spark2_forbidden: true\n"
        "runpod_forbidden: true\n"
        "model: Qwen/Qwen3-8B\n"
        f"model_revision: {MODEL_REVISION}\n"
        "layer: 27\n"
        "controllers: 31\n"
        "shells: [MEDIUM, STRONG]\n"
        "primary_n: 300\n"
        "conditions: 63\n"
        "future_semantic_trajectories: 37800\n"
        "qap_maps: 50000\n"
        "semantic_execution: forbidden\n"
        "q3: not_run\n",
        encoding="utf-8",
    )


def prepare() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    source_commit = git_head()
    candidates, safe = safe_rows()
    if len(candidates) != 40 or len(safe) != 31:
        raise RuntimeError("immutable V4 bank count changed")
    write_safety_erratum(safe)
    write_leverage_ruling(safe)
    g3_power()
    copy_label_free_manifests()
    panel = copy_semantic_panel()
    deployment = make_deployment(safe)
    schedule = make_schedule(panel, deployment)
    make_qap()
    write_a0(safe)
    write_spec(source_commit)
    write_json(
        REVIEW / "PREPARATION_AUDIT.json",
        {
            "source_commit": source_commit,
            "historical_v4_candidate_count": len(candidates),
            "safe_count": len(safe),
            "safe_order": list(EXPECTED_SAFE_IDS),
            "semantic_panel_rows": 0,
            "semantic_outcomes": 0,
            "correctness_inspected": False,
            "new_model_inference": False,
            "new_gpu_inference": False,
            "A1_A2_materialized": False,
            "Q2": "UNTESTED",
            "Q3": "NOT_RUN",
        },
    )
    write_protocol_lock(source_commit, safe, deployment, panel, schedule)
    print(json.dumps({"status": "PREPARED", "source_commit": source_commit, "review": str(REVIEW)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare",))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()


if __name__ == "__main__":
    main()
