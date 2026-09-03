#!/usr/bin/env python3
"""Blind historical prefix replay for the Q2 OOS V2 efficiency amendment.

This script never runs a model and never emits raw text, token IDs, item IDs,
reference answers, or controller identities.  It uses the sealed Q2 V4.1
journal to select a prospective hard cap on a controller-development split,
opens one held-out controller-validation split exactly once, and then performs
the frozen complete endpoint-equivalence certification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    evaluate_external_answer_v3,
)
from epistemic_geometry.experiments import q2_v4  # noqa: E402
from epistemic_geometry.experiments.heterogeneity_robust import (  # noqa: E402
    node_jackknife_test,
)

EXPECTED_JOURNAL_SHA256 = "d726b473feca8c6922b545bdf8a217e8171c8267697ff2b9714b14e1a0363a99"
EXPECTED_SCORES_SHA256 = "a6a9f4b419d4531716337d2277688063d5655167a5d7b1a9bd85b34217f8a33f"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
PARTITION_NAMESPACE = "Q2-OOS-V2-EFFICIENCY-PARTITION-V1"
CAPS = (4096, 2048, 1024, 512, 256)
SHELLS = ("MEDIUM", "STRONG")
METRICS = ("A0", "A1", "A2")
BASELINE = "BASELINE"
N = 300
K = 31
RUNTIME_SEED = 181879861245714525386395017528639201702
T975_DF30 = 2.042272456

PANEL = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"
MATRICES = ROOT / "review/q2_v4_1_prediction_lock/PREDICTION_MATRICES.npz"
ESTIMANDS = ROOT / "review/q2_v4_1_semantic_execution/ESTIMANDS.json"
RADIAL = ROOT / "review/q2_v4_1_semantic_execution/RADIAL_RESULTS.json"
ROBUST = (
    ROOT
    / "review/q2_oos_fresh_controller_design/heterogeneity_robust_inference/"
    "POST_HOC_Q2_HETEROGENEITY_ROBUST_SENSITIVITY.json"
)
PRIMARY_SCRIPT = ROOT / "scripts/analyze_q2_v4_1_semantic.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_primary_module() -> Any:
    spec = importlib.util.spec_from_file_location("q2_v4_1_primary_frozen", PRIMARY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen primary analysis")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unwrap_journal(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != EXPECTED_JOURNAL_SHA256:
        raise RuntimeError("historical journal SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            wrapper = json.loads(line)
            row = wrapper.get("row")
            if not isinstance(row, dict):
                raise RuntimeError(f"invalid historical wrapper at line {line_number}")
            key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
            if key in keys:
                raise RuntimeError("duplicate historical logical key")
            keys.add(key)
            rows.append(row)
    if len(rows) != 37_800:
        raise RuntimeError("historical journal row count mismatch")
    return rows


def load_scores(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    if sha256_file(path) != EXPECTED_SCORES_SHA256:
        raise RuntimeError("historical semantic-scores SHA-256 mismatch")
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
            if key in scores:
                raise RuntimeError("duplicate historical score key")
            scores[key] = row
    if len(scores) != 37_800:
        raise RuntimeError("historical semantic-scores row count mismatch")
    return scores


def load_references() -> tuple[list[str], dict[str, str]]:
    payload = read_json(PANEL)
    ids = [str(value) for value in payload["item_ids"]]
    references = {str(row["item_id"]): str(row["reference_answer"]) for row in payload["items"]}
    if len(ids) != N or len(references) != N or set(ids) != set(references):
        raise RuntimeError("historical panel identity mismatch")
    return ids, references


def controller_from_condition(condition: str) -> str | None:
    if condition == BASELINE:
        return None
    for shell in SHELLS:
        suffix = f"_{shell}"
        if condition.endswith(suffix):
            return condition[: -len(suffix)]
    raise RuntimeError("unrecognized historical condition")


def controller_partition(controller_ids: list[str]) -> tuple[set[str], set[str], dict[str, str]]:
    digests = {
        controller: hashlib.sha256(
            f"{PARTITION_NAMESPACE}|{controller}".encode()
        ).hexdigest()
        for controller in controller_ids
    }
    ordered = sorted(controller_ids, key=lambda value: digests[value])
    return set(ordered[:16]), set(ordered[16:]), digests


def endpoint_tuple(score: dict[str, Any]) -> tuple[bool, bool, bool, int]:
    correct = bool(score["correct"])
    return (
        bool(score["commitment_valid"]),
        bool(score["semantic_evaluable"]),
        correct,
        int(not correct),
    )


def classify_text(
    text: str,
    reference: str,
    *,
    truncated: bool,
    runtime_error: bool,
) -> tuple[bool, bool, bool, int]:
    parsed = evaluate_external_answer_v3(
        text,
        reference,
        truncated=truncated,
        runtime_error=runtime_error,
    )
    return (
        bool(parsed.commitment_valid),
        bool(parsed.semantic_evaluable),
        bool(parsed.correct),
        int(not parsed.correct),
    )


def replay_endpoint(
    row: dict[str, Any],
    reference: str,
    cap: int,
    decode: Callable[[list[int]], str],
) -> tuple[bool, bool, bool, int]:
    count = int(row["generated_token_count"])
    runtime_error = bool(row.get("runtime_error"))
    if count > cap:
        token_ids = [int(value) for value in row["generated_token_ids"][:cap]]
        if len(token_ids) != cap:
            raise RuntimeError("stored generated-token sequence is shorter than its count")
        text = decode(token_ids)
        try:
            return classify_text(text, reference, truncated=True, runtime_error=runtime_error)
        finally:
            del text, token_ids
    return classify_text(
        str(row.get("raw_output", "")),
        reference,
        truncated=bool(row.get("truncated", False)),
        runtime_error=runtime_error,
    )


def compare_cap(
    rows: list[dict[str, Any]],
    scores: dict[tuple[str, str, int], dict[str, Any]],
    references: dict[str, str],
    cap: int,
    decode: Callable[[list[int]], str],
) -> tuple[dict[str, int], dict[tuple[str, str, int], tuple[bool, bool, bool, int]]]:
    differences = {
        "commitment_valid": 0,
        "semantic_evaluable": 0,
        "correct": 0,
        "binary_error_e": 0,
    }
    replayed: dict[tuple[str, str, int], tuple[bool, bool, bool, int]] = {}
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        old = endpoint_tuple(scores[key])
        new = replay_endpoint(row, references[key[0]], cap, decode)
        replayed[key] = new
        for index, field in enumerate(differences):
            differences[field] += int(old[index] != new[index])
    return differences, replayed


def exact(differences: dict[str, int]) -> bool:
    return all(value == 0 for value in differences.values())


def valid_lengths(
    rows: list[dict[str, Any]], scores: dict[tuple[str, str, int], dict[str, Any]]
) -> np.ndarray:
    values = []
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        score = scores[key]
        if (
            not bool(row.get("truncated", False))
            and not bool(row.get("runtime_error"))
            and bool(score["commitment_valid"])
            and bool(score["semantic_evaluable"])
        ):
            values.append(int(row["generated_token_count"]))
    if not values:
        raise RuntimeError("no naturally terminated valid/evaluable historical outputs")
    return np.asarray(values, dtype=np.int64)


def length_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(values)),
        "p99": float(np.quantile(values, 0.99)),
        "p99_9": float(np.quantile(values, 0.999)),
        "maximum": int(np.max(values)),
        "required_cap_at_2x_max": int(2 * np.max(values)),
    }


def replay_to_score_rows(
    scores: dict[tuple[str, str, int], dict[str, Any]],
    replayed: dict[tuple[str, str, int], tuple[bool, bool, bool, int]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, values in replayed.items():
        old = scores[key]
        output[key] = {
            **old,
            "commitment_valid": values[0],
            "semantic_evaluable": values[1],
            "correct": values[2],
        }
    return output


def errors_by_condition(
    item_ids: list[str],
    controller_ids: list[str],
    scores: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, np.ndarray]:
    conditions = [
        BASELINE,
        *(f"{controller}_{shell}" for controller in controller_ids for shell in SHELLS),
    ]
    return {
        condition: np.asarray(
            [
                [int(not scores[(item, condition, rollout)]["correct"]) for rollout in (0, 1)]
                for item in item_ids
            ],
            dtype=np.float64,
        )
        for condition in conditions
    }


def max_abs(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise RuntimeError("scientific object shape mismatch")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, key in enumerate(ordered):
        running = max(running, (len(ordered) - index) * pvalues[key])
        adjusted[key] = min(running, 1.0)
    return adjusted


def robust_recompute(
    geometries: dict[str, dict[str, np.ndarray]], outcomes: dict[str, np.ndarray]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    for metric in METRICS:
        audit = node_jackknife_test(geometries[metric], outcomes)
        pseudo = np.asarray(audit["pseudovalues"], dtype=np.float64)
        estimate = float(np.mean(pseudo))
        se = float(audit["jackknife_standard_error"])
        p = float(audit["p_value"])
        pvalues[metric] = p
        results[metric] = {
            "historical_full_association": float(audit["full_association"]),
            "jackknife_pseudovalue_mean": estimate,
            "jackknife_standard_error": se,
            "CI95": [estimate - T975_DF30 * se, estimate + T975_DF30 * se],
            "t": float(audit["t"]),
            "one_sided_p": p,
            "leave_one_node_min": float(np.min(audit["leave_one_out"])),
            "leave_one_node_max": float(np.max(audit["leave_one_out"])),
            "leave_one_node_all_positive": bool(np.all(audit["leave_one_out"] > 0.0)),
        }
    adjusted = holm(pvalues)
    for metric in METRICS:
        results[metric]["Holm_p"] = adjusted[metric]
        results[metric]["robust_support"] = bool(
            adjusted[metric] <= 0.05 and results[metric]["CI95"][0] > 0.0
        )
    if all(results[metric]["robust_support"] for metric in METRICS):
        classification = "Q2_V4_1_HETEROGENEITY_ROBUST_SUPPORT"
    elif any(results[metric]["robust_support"] for metric in METRICS):
        classification = "Q2_V4_1_HETEROGENEITY_SENSITIVITY_MIXED"
    else:
        classification = "Q2_V4_1_HETEROGENEITY_ROBUST_SUPPORT_NOT_ESTABLISHED"
    return {"results": results, "classification": classification}


def scientific_certification(
    item_ids: list[str],
    controller_ids: list[str],
    historical_scores: dict[tuple[str, str, int], dict[str, Any]],
    replayed: dict[tuple[str, str, int], tuple[bool, bool, bool, int]],
) -> dict[str, Any]:
    primary = load_primary_module()
    canonical = read_json(ESTIMANDS)
    canonical_radial = read_json(RADIAL)
    canonical_robust = read_json(ROBUST)
    replay_scores = replay_to_score_rows(historical_scores, replayed)
    old_errors = errors_by_condition(item_ids, controller_ids, historical_scores)
    new_errors = errors_by_condition(item_ids, controller_ids, replay_scores)
    error_difference = max(
        max_abs(old_errors[key], new_errors[key]) for key in old_errors
    )
    geometries = primary.metric_arrays()
    old_shape, _old_total_baseline, _, _ = primary.semantic_distance_arrays(
        old_errors, controller_ids
    )
    new_shape, _new_total_baseline, _, _ = primary.semantic_distance_arrays(
        new_errors, controller_ids
    )
    dshape_difference = max(max_abs(old_shape[shell], new_shape[shell]) for shell in SHELLS)
    dtotal_difference = 0.0
    for shell in SHELLS:
        old_stack = np.stack(
            [old_errors[BASELINE], *(old_errors[f"{cid}_{shell}"] for cid in controller_ids)]
        )
        new_stack = np.stack(
            [new_errors[BASELINE], *(new_errors[f"{cid}_{shell}"] for cid in controller_ids)]
        )
        old_total = q2_v4.blind_spot_shape_matrices(old_stack)["total"]
        new_total = q2_v4.blind_spot_shape_matrices(new_stack)["total"]
        dtotal_difference = max(dtotal_difference, max_abs(old_total, new_total))

    upper = np.triu_indices(K, 1)
    relational: dict[str, Any] = {}
    for metric in METRICS:
        shell_rho = {
            shell: float(q2_v4.spearman(geometries[metric][shell][upper], new_shape[shell][upper]))
            for shell in SHELLS
        }
        aggregate = float(np.mean(list(shell_rho.values())))
        expected = canonical["metrics"][metric]
        relational[metric] = {
            "shell_rho": shell_rho,
            "aggregate_rho": aggregate,
            "maximum_absolute_difference": max(
                max_abs([shell_rho[shell]], [expected["shell_rho"][shell]])
                for shell in SHELLS
            ),
            "aggregate_absolute_difference": abs(aggregate - float(expected["aggregate_rho"])),
        }

    qap, qap_null = primary.qap_summary(
        {metric: geometries[metric] for metric in METRICS}, new_shape
    )
    bootstrap = primary.bootstrap(new_errors, item_ids, controller_ids, geometries)
    loo = primary.leave_one_out(
        {metric: geometries[metric] for metric in METRICS}, new_shape
    )
    qualifications: dict[str, bool] = {}
    for metric in METRICS:
        rhos = relational[metric]["shell_rho"]
        qualifications[metric] = bool(
            rhos["MEDIUM"] > 0
            and rhos["STRONG"] > 0
            and relational[metric]["aggregate_rho"] >= 0.2
            and loo[metric]["all_sign_stable"]
            and bootstrap[metric]["q025"] > 0
            and qap["maxT_adjusted_p"][metric] <= 0.05
        )
    g3_qap = primary.g3_qap(qap_null, qap["observed"])
    g3 = bool(
        qualifications["A2"]
        and g3_qap["observed"]["A2_minus_A0"] >= 0.10
        and g3_qap["observed"]["A2_minus_A1"] >= 0.10
        and bootstrap["A2_minus_A0"]["q025"] > 0
        and bootstrap["A2_minus_A1"]["q025"] > 0
        and g3_qap["maxT_superiority_p"]["A2_minus_A0"] <= 0.05
        and g3_qap["maxT_superiority_p"]["A2_minus_A1"] <= 0.05
    )
    classification = (
        "Q2_V4_1_G3"
        if g3
        else "Q2_V4_1_G2"
        if qualifications["A2"]
        else "Q2_V4_1_G1"
        if qualifications["A0"] or qualifications["A1"]
        else "Q2_V4_1_G0"
    )
    radial = primary.radial_analysis(new_errors, controller_ids)
    robust = robust_recompute(
        {metric: geometries[metric] for metric in METRICS}, new_shape
    )

    qap_difference = max(
        abs(float(qap["observed"][metric]) - float(canonical["qap"]["observed"][metric]))
        for metric in METRICS
    )
    bootstrap_difference = max(
        abs(
            float(bootstrap[metric][field])
            - float(canonical["metrics"][metric]["bootstrap"][field])
        )
        for metric in METRICS
        for field in ("estimate", "q025", "q975")
    )
    radial_difference = max(
        abs(float(radial[name][field]) - float(canonical_radial[name][field]))
        for name in ("R_shape", "R_total")
        for field in ("median", "permutation_p")
    )
    robust_difference = max(
        abs(
            float(robust["results"][metric][field])
            - float(canonical_robust["results"][metric][field])
        )
        for metric in METRICS
        for field in (
            "historical_full_association",
            "jackknife_pseudovalue_mean",
            "jackknife_standard_error",
            "one_sided_p",
            "Holm_p",
        )
    )
    result = {
        "controller_item_error_arrays_maximum_absolute_difference": error_difference,
        "Dtotal_maximum_absolute_difference": dtotal_difference,
        "Dshape_maximum_absolute_difference": dshape_difference,
        "relational": relational,
        "QAP_observed_maximum_absolute_difference": qap_difference,
        "bootstrap_maximum_absolute_difference": bootstrap_difference,
        "classification": classification,
        "classification_matches": classification == canonical["classification"] == "Q2_V4_1_G2",
        "radial": {
            "R_shape": radial["R_shape"]["classification"],
            "R_total": radial["R_total"]["classification"],
            "maximum_absolute_difference": radial_difference,
            "matches": radial["R_shape"]["classification"] == "RS+"
            and radial["R_total"]["classification"] == "RT+",
        },
        "heterogeneity_robust": {
            "classification": robust["classification"],
            "maximum_absolute_difference": robust_difference,
            "matches": robust["classification"]
            == canonical_robust["classification"]
            == "Q2_V4_1_HETEROGENEITY_ROBUST_SUPPORT",
        },
    }
    result["exact_scientific_equivalence"] = bool(
        error_difference == 0.0
        and dtotal_difference == 0.0
        and dshape_difference == 0.0
        and max(row["maximum_absolute_difference"] for row in relational.values()) == 0.0
        and qap_difference == 0.0
        and bootstrap_difference == 0.0
        and result["classification_matches"]
        and radial_difference == 0.0
        and result["radial"]["matches"]
        and robust_difference == 0.0
        and result["heterogeneity_robust"]["matches"]
    )
    return result


def fit_runtime(rows: list[dict[str, Any]]) -> tuple[float, float]:
    natural = [row for row in rows if not bool(row.get("truncated", False))]
    x = np.asarray([int(row["generated_token_count"]) for row in natural], dtype=np.float64)
    y = np.asarray([float(row["elapsed_seconds"]) for row in natural], dtype=np.float64)
    design = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def runtime_validation(
    rows: list[dict[str, Any]], intercept: float, slope: float
) -> dict[str, float]:
    natural = [row for row in rows if not bool(row.get("truncated", False))]
    x = np.asarray([int(row["generated_token_count"]) for row in natural], dtype=np.float64)
    y = np.asarray([float(row["elapsed_seconds"]) for row in natural], dtype=np.float64)
    predicted = intercept + slope * x
    residual = y - predicted
    denominator = np.sum((y - np.mean(y)) ** 2)
    return {
        "rows": int(len(y)),
        "R2": float(1.0 - np.sum(residual**2) / denominator),
        "MAE_seconds": float(np.mean(np.abs(residual))),
        "median_absolute_error_seconds": float(np.median(np.abs(residual))),
        "MAPE": float(np.mean(np.abs(residual) / np.maximum(np.abs(y), 1e-12))),
    }


def counterfactual_runtime(
    rows: list[dict[str, Any]], cap: int, intercept: float, slope: float
) -> np.ndarray:
    return np.asarray(
        [
            float(row["elapsed_seconds"])
            if int(row["generated_token_count"]) <= cap
            else max(0.0, intercept + slope * cap)
            for row in rows
        ],
        dtype=np.float64,
    )


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "P50": float(np.quantile(values, 0.50)),
        "P80": float(np.quantile(values, 0.80)),
        "P90": float(np.quantile(values, 0.90)),
        "P95": float(np.quantile(values, 0.95)),
        "P99": float(np.quantile(values, 0.99)),
    }


def runtime_report(
    rows: list[dict[str, Any]],
    development: set[str],
    validation: set[str],
    selected_cap: int,
) -> dict[str, Any]:
    dev_rows = [
        row
        for row in rows
        if controller_from_condition(str(row["condition"])) in development
    ]
    val_rows = [
        row
        for row in rows
        if controller_from_condition(str(row["condition"])) in validation
    ]
    intercept, slope = fit_runtime(dev_rows)
    validation_metrics = runtime_validation(val_rows, intercept, slope)
    old_seconds = float(sum(float(row["elapsed_seconds"]) for row in rows))
    candidates: dict[str, Any] = {}
    for cap in CAPS:
        values = counterfactual_runtime(rows, cap, intercept, slope)
        total = float(np.sum(values))
        tail = np.asarray(
            [
                value
                for value, row in zip(values, rows, strict=True)
                if int(row["generated_token_count"]) > cap
            ],
            dtype=np.float64,
        )
        candidates[str(cap)] = {
            "counterfactual_hours": total / 3600.0,
            "hours_saved": (old_seconds - total) / 3600.0,
            "percent_saved": 100.0 * (old_seconds - total) / old_seconds,
            "rows_hard_stopped": int(len(tail)),
            "counterfactual_tail_hours": float(np.sum(tail) / 3600.0),
        }

    selected_times = counterfactual_runtime(rows, selected_cap, intercept, slope)
    by_controller: dict[str, list[float]] = defaultdict(list)
    cap_by_controller: dict[str, list[bool]] = defaultdict(list)
    for row, elapsed in zip(rows, selected_times, strict=True):
        controller = controller_from_condition(str(row["condition"]))
        if controller is not None:
            by_controller[controller].append(float(elapsed))
            cap_by_controller[controller].append(int(row["generated_token_count"]) > selected_cap)
    controller_ids = sorted(by_controller)
    if len(controller_ids) != K or any(
        len(by_controller[value]) != 1200 for value in controller_ids
    ):
        raise RuntimeError("future runtime controller-profile contract mismatch")
    profile_totals = np.asarray([sum(by_controller[value]) for value in controller_ids])
    rng = np.random.Generator(np.random.PCG64DXSM(RUNTIME_SEED))
    draws = rng.integers(0, K, size=(100_000, 16))
    future = np.sum(profile_totals[draws], axis=1)

    stress: dict[str, Any] = {}
    for multiplier in (1.5, 2.0):
        expected_profiles = []
        for controller in controller_ids:
            values = np.asarray(by_controller[controller], dtype=np.float64)
            capped = np.asarray(cap_by_controller[controller], dtype=bool)
            rate = float(np.mean(capped))
            capped_mean = (
                float(np.mean(values[capped]))
                if np.any(capped)
                else max(0.0, intercept + slope * selected_cap)
            )
            noncapped_mean = (
                float(np.mean(values[~capped]))
                if np.any(~capped)
                else float(np.mean(values))
            )
            stressed_rate = min(1.0, multiplier * rate)
            expected_profiles.append(
                1200.0
                * (stressed_rate * capped_mean + (1.0 - stressed_rate) * noncapped_mean)
            )
        stress[str(multiplier)] = distribution(
            np.sum(np.asarray(expected_profiles)[draws], axis=1)
        )

    return {
        "model": {
            "fit_population": "naturally terminating DEVELOPMENT-controller rows",
            "intercept_seconds": intercept,
            "per_generated_token_seconds": slope,
            "validation": validation_metrics,
        },
        "observed_historical_hours": old_seconds / 3600.0,
        "candidate_counterfactuals": candidates,
        "selected_future_19200_seconds": distribution(future),
        "stress_future_19200_seconds": stress,
        "draws": 100_000,
        "seed": str(RUNTIME_SEED),
        "runtime_used_for_policy_selection": False,
        "interpretation": "MODEL_ESTIMATED_COUNTERFACTUAL_RUNTIME_NOT_OBSERVED_EXECUTION",
    }


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    journal = Path(args.journal)
    scores_path = Path(args.scores)
    output = Path(args.output)
    rows = unwrap_journal(journal)
    scores = load_scores(scores_path)
    item_ids, references = load_references()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    decode = lambda values: tokenizer.decode(values, skip_special_tokens=True)  # noqa: E731

    controller_ids = sorted(
        {
            controller
            for row in rows
            if (controller := controller_from_condition(str(row["condition"]))) is not None
        }
    )
    if len(controller_ids) != K:
        raise RuntimeError("historical controller count mismatch")
    development, validation, partition_digests = controller_partition(controller_ids)
    dev_rows = [
        row
        for row in rows
        if controller_from_condition(str(row["condition"])) in development
    ]
    validation_rows = [
        row
        for row in rows
        if controller_from_condition(str(row["condition"])) in validation
    ]
    if len(dev_rows) != 19_200 or len(validation_rows) != 18_000:
        raise RuntimeError("controller-held-out replay partition row count mismatch")

    dev_lengths = valid_lengths(dev_rows, scores)
    development_results: dict[str, Any] = {}
    qualifying: list[int] = []
    for cap in CAPS:
        differences, _ = compare_cap(dev_rows, scores, references, cap, decode)
        margin = cap >= 2 * int(np.max(dev_lengths))
        development_results[str(cap)] = {
            "row_level_differences": differences,
            "exact_endpoint_equivalence": exact(differences),
            "valid_output_2x_margin": margin,
        }
        if cap < 4096 and exact(differences) and margin:
            qualifying.append(cap)
    if not qualifying:
        write_json(
            output,
            {
                "schema_version": "q2-oos-v2-semantic-efficiency-replay-v1",
                "status": "Q2_OOS_V2_NO_ENDPOINT_EQUIVALENT_STOP_POLICY",
                "development": development_results,
                "development_valid_lengths": length_summary(dev_lengths),
                "validation_opened": False,
                "model_inference": 0,
                "raw_text_persisted_or_printed": False,
            },
        )
        print(json.dumps({"status": "Q2_OOS_V2_NO_ENDPOINT_EQUIVALENT_STOP_POLICY"}))
        return

    selected_cap = min(qualifying)
    validation_differences, _ = compare_cap(
        validation_rows, scores, references, selected_cap, decode
    )
    validation_lengths = valid_lengths(validation_rows, scores)
    validation_margin = selected_cap >= 2 * int(np.max(validation_lengths))
    if not exact(validation_differences) or not validation_margin:
        status = "Q2_OOS_V2_NO_ENDPOINT_EQUIVALENT_STOP_POLICY"
        write_json(
            output,
            {
                "schema_version": "q2-oos-v2-semantic-efficiency-replay-v1",
                "status": status,
                "selected_on_development": selected_cap,
                "development": development_results,
                "development_valid_lengths": length_summary(dev_lengths),
                "validation": {
                    "row_level_differences": validation_differences,
                    "exact_endpoint_equivalence": exact(validation_differences),
                    "valid_output_2x_margin": validation_margin,
                    "valid_lengths": length_summary(validation_lengths),
                },
                "validation_opened_once": True,
                "no_fallback_after_validation_failure": True,
                "model_inference": 0,
                "raw_text_persisted_or_printed": False,
            },
        )
        print(json.dumps({"status": status}))
        return

    full_differences, replayed = compare_cap(
        rows, scores, references, selected_cap, decode
    )
    full_lengths = valid_lengths(rows, scores)
    full_margin = selected_cap >= 2 * int(np.max(full_lengths))
    if not exact(full_differences) or not full_margin:
        raise RuntimeError("selected held-out policy failed complete historical certification")
    scientific = scientific_certification(item_ids, controller_ids, scores, replayed)
    if not scientific["exact_scientific_equivalence"]:
        raise RuntimeError("complete historical scientific endpoint equivalence failed")
    runtimes = runtime_report(rows, development, validation, selected_cap)
    result = {
        "schema_version": "q2-oos-v2-semantic-efficiency-replay-v1",
        "status": "Q2_OOS_V2_ENDPOINT_EQUIVALENT_HARD_CAP_QUALIFIED",
        "historical_inputs": {
            "journal_sha256": EXPECTED_JOURNAL_SHA256,
            "semantic_scores_sha256": EXPECTED_SCORES_SHA256,
            "rows": len(rows),
        },
        "controller_partition": {
            "namespace": PARTITION_NAMESPACE,
            "development_controllers": len(development),
            "validation_controllers": len(validation),
            "development_rows": len(dev_rows),
            "validation_rows": len(validation_rows),
            "digest_commitment_sha256": hashlib.sha256(
                json.dumps(partition_digests, sort_keys=True).encode()
            ).hexdigest(),
            "controller_identities_reported": False,
        },
        "development": development_results,
        "development_valid_lengths": length_summary(dev_lengths),
        "selected_on_development": selected_cap,
        "held_out_validation": {
            "opened_once": True,
            "tested_cap_only": selected_cap,
            "row_level_differences": validation_differences,
            "exact_endpoint_equivalence": exact(validation_differences),
            "valid_output_2x_margin": validation_margin,
            "valid_lengths": length_summary(validation_lengths),
        },
        "full_historical_certification": {
            "row_level_differences": full_differences,
            "exact_endpoint_equivalence": exact(full_differences),
            "valid_output_2x_margin": full_margin,
            "valid_lengths": length_summary(full_lengths),
            "scientific": scientific,
        },
        "selected_policy": {
            "type": "HARD_CAP_ONLY",
            "semantic_max_new_tokens": selected_cap,
            "repetition_stop": False,
            "safety_max_new_tokens_unchanged": 4096,
        },
        "runtime": runtimes,
        "V2_safety_results_used": False,
        "V2_semantic_outcomes_observed": 0,
        "model_inference": 0,
        "raw_text_persisted_or_printed": False,
        "benchmark_content_persisted_or_printed": False,
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_semantic_max_new_tokens": selected_cap,
                "exact_full_endpoint_equivalence": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
