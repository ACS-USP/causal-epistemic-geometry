#!/usr/bin/env python3
"""Primary offline analysis for Gate 10 cross-domain character counting."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_gate7_fresh_l27_replication as core  # noqa: E402

from epistemic_geometry.benchmarks.v4.character_semantic_v3 import (  # noqa: E402
    evaluate_character_count_answer_v3,
)
from epistemic_geometry.experiments import gate10  # noqa: E402
from epistemic_geometry.experiments.gate6_3_v3 import (  # noqa: E402
    audit_two_rollout_estimands,
)

REVIEW = ROOT / "review/gate10_cross_domain_charcount"
ORIGINAL_POD_UPTIME_SECONDS = 1177
RESUME_POD_UPTIME_SECONDS = 41299
A40_HOURLY_RATE_USD = 0.44

_OPPORTUNITY: dict[str, Any] = {}
_SOURCE: dict[str, Any] = {}


def _reparse(row: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_character_count_answer_v3(
        str(row.get("raw_output", "")),
        str(row["reference_answer"]),
        truncated=int(row.get("generated_token_count", 0)) >= gate10.MAX_NEW_TOKENS,
        runtime_error=str(row.get("status")) == "RUNTIME_ERROR",
    )
    if result.correct:
        status = "VALID_CORRECT"
    elif result.commitment_valid and result.semantic_evaluable:
        status = "VALID_WRONG"
    elif result.failure_reason == "truncated or unclosed response":
        status = "TRUNCATED"
    elif result.failure_reason == "runtime error":
        status = "RUNTIME_ERROR"
    else:
        status = "INVALID_FORMAT"
    return {
        "status": status,
        "correct": bool(result.correct),
        "commitment_valid": bool(result.commitment_valid),
        "semantic_evaluable": bool(result.semantic_evaluable),
        "canonical_value": result.canonical_value,
        "parsed_answer": result.payload,
        "failure_reason": result.failure_reason,
    }


def _configure_core() -> None:
    core.REVIEW = REVIEW
    core.BASELINE = gate10.BASELINE
    core.BOOTSTRAP_RESAMPLES = gate10.BOOTSTRAP_RESAMPLES
    core.BOOTSTRAP_SEED = gate10.BOOTSTRAP_SEED
    core.CONDITIONS = gate10.CONDITIONS
    core.MAX_NEW_TOKENS = gate10.MAX_NEW_TOKENS
    core.MEANINGFUL = gate10.MEANINGFUL
    core.RANDOM_NAMES = gate10.RANDOM_NAMES
    core.TEXTUAL = gate10.TEXTUAL
    core._reparse = _reparse
    core.classify_gate7 = _classify


def _classify(
    *,
    baseline: dict[str, float],
    controller: dict[str, float],
    controller_estimands: dict[str, float],
    random_summary: dict[str, dict[str, float]],
    bootstrap: dict[str, dict[str, float]],
    loo_sign_stable: dict[str, bool],
    controller_style_replicated: bool,
) -> tuple[str, dict[str, Any]]:
    del controller_style_replicated
    point = dict(controller_estimands)
    point["G_norm"] = point["G"] / point["B00"] if point["B00"] > 0 else None
    classification, gates = gate10.classify_gate10(
        baseline=baseline,
        controller=controller,
        point=point,
        random_summary=random_summary,
        bootstrap=bootstrap,
        loo=loo_sign_stable,
        opportunity_pass=bool(_OPPORTUNITY["pass"]),
        style_transfer=bool(_SOURCE["style_transfer"]),
        accuracy_bootstrap_positive=bootstrap["meaningful:accuracy_change"]["q025"] > 0,
    )
    gates["commitment_validity_guard"] = gates["commitment_guard"]
    gates["semantic_evaluability_guard"] = gates["evaluability_guard"]
    return classification, gates


def _context(review: Path) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, dict[str, float]],
]:
    rows = core.read_journal(review / "journal.jsonl")
    parsed = {
        (row["item_id"], row["condition"], int(row["rollout_index"])): _reparse(row)
        for row in rows
    }
    item_ids = sorted({str(row["item_id"]) for row in rows})
    _, summaries = core._condition_summary(rows, parsed)
    arrays = core._arrays(item_ids, parsed)
    return rows, item_ids, parsed, arrays, summaries


def _source_summary(
    item_ids: list[str],
    parsed: dict[tuple[str, str, int], dict[str, Any]],
    summaries: dict[str, dict[str, float]],
    lock: dict[str, Any],
) -> dict[str, Any]:
    baseline = summaries[gate10.BASELINE]
    textual = summaries[gate10.TEXTUAL]
    meaningful = summaries[gate10.MEANINGFUL]
    replicated = bool(
        textual["commitment_validity"] >= 0.95
        and textual["semantic_evaluability"] >= 0.95
        and textual["accuracy"] >= baseline["accuracy"] - 0.03
        and (
            textual["mean_tokens"] >= 1.5 * baseline["mean_tokens"]
            or textual["median_tokens"] >= baseline["median_tokens"] + 10
            or textual["accuracy"] >= baseline["accuracy"] + 0.03
        )
    )
    token_denominator = textual["mean_tokens"] - baseline["mean_tokens"]
    token_recovery = (
        (meaningful["mean_tokens"] - baseline["mean_tokens"]) / token_denominator
        if token_denominator > 0
        else None
    )
    accuracy_denominator = textual["accuracy"] - baseline["accuracy"]
    accuracy_recovery = (
        (meaningful["accuracy"] - baseline["accuracy"]) / accuracy_denominator
        if accuracy_denominator > 0
        else None
    )
    threshold = float(lock["style_transfer_threshold"]["mean_token_recovery_fraction_min"])
    return {
        "classification": "CHARCOUNT_CAREFUL_SOURCE_REPLICATED"
        if replicated
        else "CHARCOUNT_CAREFUL_SOURCE_NOT_REPLICATED",
        "style_transfer": bool(
            replicated and token_recovery is not None and token_recovery >= threshold
        ),
        "style_transfer_threshold": threshold,
        "accuracy_gain_recovered_fraction": accuracy_recovery,
        "token_gain_recovered_fraction": token_recovery,
        "meaningful_textual_semantic_agreement": core._answer_agreement(
            item_ids, gate10.MEANINGFUL, gate10.TEXTUAL, parsed
        ),
        "meaningful_baseline_semantic_agreement": core._answer_agreement(
            item_ids, gate10.MEANINGFUL, gate10.BASELINE, parsed
        ),
    }


def _g_norm_bootstrap(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    rng = np.random.default_rng(gate10.BOOTSTRAP_SEED)
    values: list[float] = []
    n_items = len(arrays[gate10.BASELINE])
    for _ in range(gate10.BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, n_items, size=n_items)
        point = audit_two_rollout_estimands(
            arrays[gate10.BASELINE][indices], arrays[gate10.MEANINGFUL][indices]
        )
        if point["B00"] > 0:
            values.append(float(point["G"] / point["B00"]))
    if not values:
        raise RuntimeError("Gate 10 G_norm bootstrap has no finite resamples")
    data = np.asarray(values, dtype=np.float64)
    return {
        "estimate": float(np.median(data)),
        "q025": float(np.quantile(data, 0.025)),
        "q975": float(np.quantile(data, 0.975)),
        "resamples": gate10.BOOTSTRAP_RESAMPLES,
        "finite_resamples": len(values),
    }


def _bin_rows(
    review: Path, item_ids: list[str], arrays: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    manifest = core.read_json(review / "EVALUATION_MANIFEST.json")
    items = {item["item_id"]: item for item in manifest["items"]}
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item_id in enumerate(item_ids):
        metadata = items[item_id]["metadata"]
        length = int(metadata["length"])
        density = float(metadata["target_density"])
        labels = {
            "target_count": str(int(metadata["exact_oracle"])),
            "string_length": "31-40" if length <= 40 else "41-50" if length <= 50 else "51-60",
            "target_density": "<0.05"
            if density < 0.05
            else "0.05-<0.10"
            if density < 0.10
            else ">=0.10",
            "alphabet_size": str(int(metadata["alphabet_size"])),
        }
        for kind, label in labels.items():
            grouped[(kind, label)].append(index)
    baseline = arrays[gate10.BASELINE].astype(np.float64)
    meaningful = arrays[gate10.MEANINGFUL].astype(np.float64)
    result: list[dict[str, Any]] = []
    for (kind, label), indices in sorted(grouped.items()):
        b = baseline[indices]
        j = meaningful[indices]
        rescue = (
            b[:, 0] * (1 - j[:, 0])
            + b[:, 0] * (1 - j[:, 1])
            + b[:, 1] * (1 - j[:, 0])
            + b[:, 1] * (1 - j[:, 1])
        ) / 4.0
        damage = (
            (1 - b[:, 0]) * j[:, 0]
            + (1 - b[:, 0]) * j[:, 1]
            + (1 - b[:, 1]) * j[:, 0]
            + (1 - b[:, 1]) * j[:, 1]
        ) / 4.0
        result.append(
            {
                "bin_type": kind,
                "bin_label": label,
                "n_items": len(indices),
                "baseline_error_rate": float(b.mean()),
                "meaningful_error_rate": float(j.mean()),
                "rescue": float(rescue.mean()),
                "damage": float(damage.mean()),
            }
        )
    return result


def _report(review: Path, result: dict[str, Any]) -> None:
    summaries = result["summaries"]
    point = result["estimands"][gate10.MEANINGFUL]
    contrasts = result["meaningful_random_contrasts"]
    opportunity = core.read_json(review / "BASELINE_OPPORTUNITY.json")
    source = core.read_json(review / "SOURCE_REFERENCE_SUMMARY.json")
    intervals = core.read_json(review / "BOOTSTRAP_INTERVALS.json")
    cost = core.read_json(review / "COST.json")
    accuracy_difference = (
        summaries[gate10.MEANINGFUL]["accuracy"] - summaries[gate10.BASELINE]["accuracy"]
    )
    accuracy_recovery = source["accuracy_gain_recovered_fraction"]
    accuracy_recovery_display = "N/A" if accuracy_recovery is None else f"{accuracy_recovery:.6f}"
    contrast_lines = {
        metric: (
            f"{contrasts[metric]['minus_random_mean']:.6f} / "
            f"{contrasts[metric]['minus_random_max']:.6f}"
        )
        for metric in ("G", "C", "D")
    }
    rows = "".join(
        f"| `{name}` | {summaries[name]['commitment_validity']:.4f} | "
        f"{summaries[name]['semantic_evaluability']:.4f} | "
        f"{summaries[name]['accuracy']:.4f} | {summaries[name]['mean_tokens']:.2f} / "
        f"{summaries[name]['median_tokens']:.1f} / {summaries[name]['max_tokens']:.0f} |\n"
        for name in gate10.CONDITIONS
    )
    random_rows = "".join(
        f"| {metric} | {result['random_summary'][metric]['mean']:.6f} | "
        f"{result['random_summary'][metric]['median']:.6f} | "
        f"{result['random_summary'][metric]['min']:.6f} | "
        f"{result['random_summary'][metric]['max']:.6f} |\n"
        for metric in ("G", "C", "D")
    )
    interval_rows = "".join(
        f"| `{name}` | {intervals[name]['q025']:.6f} | {intervals[name]['q975']:.6f} |\n"
        for name in (
            "meaningful:accuracy_change",
            "meaningful:commitment_validity_change",
            "meaningful:semantic_evaluability_change",
            "meaningful:G",
            "meaningful:C",
            "meaningful:D",
            "meaningful:G_norm",
            "meaningful:G_minus_random_mean",
            "meaningful:C_minus_random_mean",
            "meaningful:D_minus_random_mean",
            "meaningful:rescue",
            "meaningful:damage",
        )
    )
    text = f"""# GATE 10 — CROSS-DOMAIN CHARACTER COUNTING

Primary classification: `{result['classification']}`.

Opportunity classification: `{result['opportunity_classification']}`.

Source-policy classification: `{result['source_policy_classification']}`.

This frozen DEVELOPMENT evaluation used 200 fresh `FRESH_PSEUDOWORD_LONG`
items, seven conditions, and two independent rollouts: 2,800 trajectories.

## Conditions

| condition | commitment | evaluability | accuracy | mean / median / max tokens |
|---|---:|---:|---:|---:|
{rows}

## Baseline opportunity

- B00 / O00: {opportunity['B00']:.6f} / {opportunity['O00']:.6f}
- Double-wrong items: {opportunity['double_wrong_items']}
- Correct in at least one rollout: {opportunity['correct_in_at_least_one_items']}

## Meaningful fixed L27-D75 controller

- Accuracy difference: {accuracy_difference:.6f}
- G / C / D: {point['G']:.6f} / {point['C']:.6f} / {point['D']:.6f}
- G_norm: {point['G_norm']:.6f}
- Rescue / damage: {point['rescue']:.6f} / {point['damage']:.6f}
- G minus random mean/max: {contrast_lines['G']}
- C minus random mean/max: {contrast_lines['C']}
- D minus random mean/max: {contrast_lines['D']}

Safety guards: commitment={result['gates']['commitment_guard']},
evaluability={result['gates']['evaluability_guard']}, and
competence={result['gates']['competence_guard']}.

## Random-controller null

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
{random_rows}

## Textual CAREFUL reference

- Classification: `{source['classification']}`
- Accuracy-gain fraction recovered: {accuracy_recovery_display}
- Token-increase fraction recovered: {source['token_gain_recovered_fraction']}

## Item-cluster bootstrap (10,000 resamples)

| estimand | 2.5% | 97.5% |
|---|---:|---:|
{interval_rows}

## Cost and firewall

- Scientific trajectories: {cost['scientific_trajectories']}
- Total A40 runtime: {cost['total_a40_runtime_hours']:.6f} hours
- Incremental GPU cost: US${cost['incremental_gpu_cost_usd']:.4f}
- RunPod: `{cost['runpod_status']}`
- Q2: `NOT RUN`
- Confirmatory holdout: `UNTOUCHED`
- Gate 11: `DRAFTED, NOT RUN`

## Interpretation boundary

The classification was applied mechanically after all frozen rows were
collected. This is cross-domain DEVELOPMENT evidence only. It is not Q2,
confirmatory holdout evidence, or execution of Gate 11. Descriptive generator
bins were not used for selection or classification.
"""
    (review / "REPORT.md").write_text(text, encoding="utf-8")


def analyze(review: Path) -> dict[str, Any]:
    global _OPPORTUNITY, _SOURCE
    _configure_core()
    lock = core.read_json(review / "PROTOCOL_LOCK.json")
    rows, item_ids, parsed, arrays, summaries = _context(review)
    _OPPORTUNITY = gate10.opportunity(arrays[gate10.BASELINE], summaries[gate10.BASELINE])
    _SOURCE = _source_summary(item_ids, parsed, summaries, lock)
    result = core.analyze(review)

    b00 = float(result["estimands"][gate10.BASELINE]["B00"])
    mu0 = 1.0 - summaries[gate10.BASELINE]["accuracy"]
    normalized: dict[str, dict[str, float | None]] = {}
    for condition in gate10.CONDITIONS[1:]:
        point = result["estimands"][condition]
        normalized[condition] = {
            "G_norm": point["G"] / b00 if b00 > 0 else None,
            "rescue_efficiency": point["rescue"] / mu0 if mu0 > 0 else None,
            "damage_on_correct": point["damage"] / (1 - mu0) if mu0 < 1 else None,
        }
        point.update(normalized[condition])
    core.write_json(review / "NORMALIZED_ESTIMANDS.json", normalized)
    core.write_json(review / "BASELINE_OPPORTUNITY.json", _OPPORTUNITY)
    core.write_json(review / "SOURCE_REFERENCE_SUMMARY.json", _SOURCE)

    intervals = core.read_json(review / "BOOTSTRAP_INTERVALS.json")
    intervals["meaningful:G_norm"] = _g_norm_bootstrap(arrays)
    core.write_json(review / "BOOTSTRAP_INTERVALS.json", intervals)

    loo_rows: list[dict[str, Any]] = []
    loo_signs: dict[str, list[float]] = defaultdict(list)
    for index, item_id in enumerate(item_ids):
        keep = np.arange(len(item_ids)) != index
        point = audit_two_rollout_estimands(
            arrays[gate10.BASELINE][keep], arrays[gate10.MEANINGFUL][keep]
        )
        record = {
            "left_out_item_id": item_id,
            "accuracy_change": point["accuracy_condition"] - point["accuracy_baseline"],
            "G": point["G"],
            "C": point["C"],
            "D": point["D"],
            "G_norm": point["G"] / point["B00"] if point["B00"] > 0 else None,
        }
        loo_rows.append(record)
        for metric in ("accuracy_change", "G", "C", "D"):
            loo_signs[metric].append(float(record[metric]))
    core.write_csv(review / "LOO_SENSITIVITY.csv", loo_rows, list(loo_rows[0]))
    result["loo_sign_stable"] = {
        metric: all(value > 0 for value in values) for metric, values in loo_signs.items()
    }
    bin_rows = _bin_rows(review, item_ids, arrays)
    core.write_csv(review / "GENERATOR_BIN_SUMMARY.csv", bin_rows, list(bin_rows[0]))

    result.pop("historical_gate6_3_classification", None)
    result["source_policy_classification"] = _SOURCE["classification"]
    result["opportunity_classification"] = _OPPORTUNITY["classification"]
    result["normalized_estimands"] = normalized
    result["scientific_design_changes_after_outcomes"] = []
    core.write_json(review / "ESTIMANDS.json", result)

    total_runtime = ORIGINAL_POD_UPTIME_SECONDS + RESUME_POD_UPTIME_SECONDS
    cost = {
        "scientific_trajectories": len(rows),
        "summed_generation_seconds": sum(float(row.get("elapsed_seconds", 0)) for row in rows),
        "original_pod_runtime_seconds": ORIGINAL_POD_UPTIME_SECONDS,
        "resume_pod_runtime_seconds": RESUME_POD_UPTIME_SECONDS,
        "total_a40_runtime_seconds": total_runtime,
        "total_a40_runtime_hours": total_runtime / 3600,
        "hourly_gpu_rate_usd": A40_HOURLY_RATE_USD,
        "incremental_gpu_cost_usd": total_runtime / 3600 * A40_HOURLY_RATE_USD,
        "target_usd": 2.5,
        "original_hard_ceiling_usd": 5.0,
        "principal_amended_hard_ceiling_usd": 6.0,
        "cost_gate": "PASS_UNDER_PRINCIPAL_AMENDED_CEILING",
        "runpod_status": "STOPPED",
    }
    core.write_json(review / "COST.json", cost)
    _report(review, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    result = analyze(args.review_dir.resolve())
    print(json.dumps({"classification": result["classification"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
