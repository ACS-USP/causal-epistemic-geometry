#!/usr/bin/env python3
"""Prospectively staged primary analysis and transition locks for Gate 13."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_gate8_l27_dose_calibration as dose_core  # noqa: E402

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    evaluate_external_answer_v3,
)
from epistemic_geometry.experiments import gate13  # noqa: E402
from epistemic_geometry.experiments.gate6_3_v3 import (  # noqa: E402
    audit_two_rollout_estimands,
    item_contributions,
)
from epistemic_geometry.experiments.gate9 import classify_gate9  # noqa: E402

REVIEW = ROOT / "review/gate13_cross_model_ministral3"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Gate 13 CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def journal_rows(review: Path, stage: str, model: str | None = None) -> list[dict[str, Any]]:
    path = review / "journal.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [
        row
        for row in rows
        if row["stage"] == stage and (model is None or row["model"] == model)
    ]


def reparsed(row: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_external_answer_v3(
        str(row.get("raw_output", "")),
        str(row["reference_answer"]),
        truncated=int(row.get("generated_token_count", 0)) >= gate13.MAX_NEW_TOKENS,
        runtime_error=str(row.get("status")) == "RUNTIME_ERROR",
    )
    status = (
        "VALID_CORRECT"
        if result.correct
        else "VALID_WRONG"
        if result.commitment_valid and result.semantic_evaluable
        else "TRUNCATED"
        if result.failure_reason == "truncated or unclosed response"
        else "RUNTIME_ERROR"
        if result.failure_reason == "runtime error"
        else "INVALID_FORMAT"
    )
    return {
        "status": status,
        "correct": bool(result.correct),
        "commitment_valid": bool(result.commitment_valid),
        "semantic_evaluable": bool(result.semantic_evaluable),
        "canonical_value": result.canonical_value,
        "failure_reason": result.failure_reason,
    }


def semantic_outcome(parsed: dict[str, Any]) -> str:
    if parsed["commitment_valid"] and parsed["semantic_evaluable"]:
        return "VALUE:" + json.dumps(
            parsed["canonical_value"], sort_keys=True, ensure_ascii=False
        )
    return f"MECHANICAL:{parsed['status']}:{parsed['failure_reason']}"


def assert_complete(
    rows: list[dict[str, Any]], schedule: list[dict[str, Any]], model: str
) -> None:
    expected = Counter(
        (
            row["stage"],
            model,
            row["item_id"],
            row["condition"],
            int(row["rollout_index"]),
        )
        for row in schedule
    )
    observed = Counter(
        (
            row["stage"],
            row["model"],
            row["item_id"],
            row["condition"],
            int(row["rollout_index"]),
        )
        for row in rows
    )
    if expected != observed:
        raise RuntimeError("Gate 13 stage journal does not match its frozen schedule")


def condition_summary(
    rows: list[dict[str, Any]], parsed: dict[tuple[str, str, int], dict[str, Any]]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        values = [
            parsed[(str(row["item_id"]), condition, int(row["rollout_index"]))]
            for row in selected
        ]
        tokens = np.asarray([int(row["generated_token_count"]) for row in selected])
        output[condition] = {
            "n": len(selected),
            "commitment_validity": float(np.mean([row["commitment_valid"] for row in values])),
            "semantic_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in values])
            ),
            "accuracy": float(np.mean([row["correct"] for row in values])),
            "mean_tokens": float(np.mean(tokens)),
            "median_tokens": float(np.median(tokens)),
            "max_tokens": float(np.max(tokens)),
            "truncation": float(np.mean([row["status"] == "TRUNCATED" for row in values])),
            "no_commitment": float(
                np.mean([not row["commitment_valid"] for row in values])
            ),
        }
    return output


def analyze_screen(review: Path, model_role: str) -> dict[str, Any]:
    if model_role == "primary":
        model = gate13.PRIMARY_MODEL
        schedule_path = review / "SUBSTRATE_SCREEN_SCHEDULE.json"
    else:
        model = gate13.FALLBACK_MODEL
        schedule_path = review / "FALLBACK_SCREEN_SCHEDULE.json"
    schedule = read_json(schedule_path)
    rows = journal_rows(review, "SUBSTRATE_SCREEN", model)
    assert_complete(rows, schedule, model)
    parsed = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): reparsed(row)
        for row in rows
    }
    summaries = condition_summary(rows, parsed)
    item_ids = sorted({str(row["item_id"]) for row in rows})
    changes = []
    for item_id in item_ids:
        for careful_rollout in (0, 1):
            for direct_rollout in (0, 1):
                changes.append(
                    semantic_outcome(parsed[(item_id, "SOURCE_CAREFUL", careful_rollout)])
                    != semantic_outcome(parsed[(item_id, "SOURCE_DIRECT", direct_rollout)])
                )
    summaries["SOURCE_CAREFUL"]["semantic_change_vs_direct"] = float(np.mean(changes))
    classification, gates = gate13.classify_substrate(summaries)
    cost = read_json(review / "COST_PROJECTION.json")
    fallback_authorized = bool(
        model_role == "primary"
        and classification == "MINISTRAL3_8B_COMPETENCE_FLOOR"
        and gates["mechanical"]
        and summaries["SOURCE_CAREFUL"]["accuracy"]
        >= summaries["SOURCE_DIRECT"]["accuracy"] + 0.05
        and gates["source_behavior"]
        and cost["initial_projected_total_usd_with_25pct_margin"] <= 9.50
    )
    result = {
        "model": model,
        "model_role": model_role,
        "classification": classification,
        "gates": gates,
        "condition_summaries": summaries,
        "fallback_authorized": fallback_authorized,
        "diagnostics_do_not_select": {
            "careful_concise": summaries["CAREFUL_CONCISE"],
            "verbose_direct": summaries["VERBOSE_DIRECT"],
        },
    }
    write_json(review / "SUBSTRATE_SCREEN_REPORT.json", result)
    (review / "SUBSTRATE_SCREEN_REPORT.md").write_text(
        "# Gate 13 substrate screen\n\n"
        f"Model: `{model}`. Classification: `{classification}`.\n\n"
        "CAREFUL_CONCISE and VERBOSE_DIRECT remain diagnostic only and did not select "
        "the model, layer, direction, or dose.\n",
        encoding="utf-8",
    )
    if fallback_authorized:
        manifest = read_json(review / "FALLBACK_SCREEN_MANIFEST.json")
        schedule = gate13.build_screen_schedule(
            [row["item_id"] for row in manifest["items"]], gate13.FALLBACK_MODEL
        )
        write_json(review / "FALLBACK_SCREEN_SCHEDULE.json", schedule)
    elif classification.endswith("SUBSTRATE_PASS"):
        selected = {
            "status": "FROZEN_PRE_SOURCE_ATLAS",
            "model": model,
            "revision": (
                gate13.PRIMARY_REVISION if model_role == "primary" else gate13.FALLBACK_REVISION
            ),
            "screen_classification": classification,
            "engine": read_json(review / "MODEL_ENGINE_SPEC.json"),
            "parser": read_json(review / "RESPONSE_PARSER_LOCK.json"),
            "no_later_model_switch": True,
        }
        write_json(review / "SELECTED_MODEL_LOCK.json", selected)
        (review / "SELECTED_MODEL_LOCK.md").write_text(
            "# Gate 13 selected model lock\n\n"
            f"Selected `{model}` at `{selected['revision']}` after the complete frozen screen.\n",
            encoding="utf-8",
        )
    return result


def analyze_source_atlas(review: Path) -> dict[str, Any]:
    archive = np.load(review / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    directions, atlas = gate13.source_atlas(
        archive["construction_careful"],
        archive["construction_direct"],
        archive["validation_careful"],
        archive["validation_direct"],
    )
    direction_dir = review / "SOURCE_DIRECTIONS"
    direction_dir.mkdir(exist_ok=True)
    for layer, vector in directions.items():
        np.save(direction_dir / f"L{layer}.npy", vector.astype(np.float64), allow_pickle=False)
    write_csv(review / "SOURCE_ATLAS.csv", atlas)
    write_json(review / "SOURCE_ATLAS.json", {"layers": atlas})
    shortlist = gate13.shortlist_layers(atlas, len(atlas))
    model_lock = read_json(review / "SELECTED_MODEL_LOCK.json")
    conditions: list[dict[str, Any]] = []
    null_dir = review / "LAYER_FIRST_STAGE_DIRECTIONS"
    null_dir.mkdir(exist_ok=True)
    construction_diff = archive["construction_careful"] - archive["construction_direct"]
    by_layer = {int(row["layer"]): row for row in atlas}
    for layer in shortlist:
        meaningful = directions[layer]
        nulls = gate13.first_stage_nulls(meaningful, construction_diff[:, layer, :], layer)
        alpha = 0.5 * float(by_layer[layer]["paired_mean_gap"])
        records = {"MEANINGFUL": meaningful, **nulls}
        for kind, vector in records.items():
            path = null_dir / f"L{layer}_{kind}.npy"
            np.save(path, vector.astype(np.float64), allow_pickle=False)
            conditions.append(
                {
                    "condition": f"{kind}_L{layer}_D50",
                    "layer": layer,
                    "alpha": alpha,
                    "vector_path": str(path.relative_to(review)),
                    "vector_hash": gate13.vector_sha256(vector),
                }
            )
    schedule = gate13.build_first_stage_schedule(
        [
            row["item_id"]
            for row in read_json(review / "LAYER_FIRST_STAGE_MANIFEST.json")["items"]
        ],
        model_lock["model"],
        shortlist,
    )
    lock = {
        "status": "FROZEN_PRE_FIRST_STAGE",
        "model": model_lock["model"],
        "revision": model_lock["revision"],
        "candidate_layers": shortlist,
        "conditions": conditions,
        "selection_rule": "largest meaningful Q-null mean; source effect; lower layer",
        "source_labels_only": True,
        "semantic_correctness_used": False,
    }
    write_json(review / "LAYER_SHORTLIST_LOCK.json", lock)
    write_json(review / "LAYER_FIRST_STAGE_SCHEDULE.json", schedule)
    return {"eligible_layers": [row["layer"] for row in atlas if row["source_eligible"]], **lock}


def matched_q(
    rows: list[dict[str, Any]], baseline: str, condition: str
) -> tuple[float, dict[str, float]]:
    parsed = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): reparsed(row)
        for row in rows
    }
    selected = [row for row in rows if row["condition"] == condition]
    changes = []
    for row in selected:
        key = (str(row["item_id"]), baseline, int(row["rollout_index"]))
        current = (str(row["item_id"]), condition, int(row["rollout_index"]))
        changes.append(semantic_outcome(parsed[key]) != semantic_outcome(parsed[current]))
    summaries = condition_summary(rows, parsed)
    return float(np.mean(changes)), summaries[condition]


def analyze_first_stage(review: Path) -> dict[str, Any]:
    selected_model = read_json(review / "SELECTED_MODEL_LOCK.json")
    rows = journal_rows(review, "LAYER_FIRST_STAGE", selected_model["model"])
    schedule = read_json(review / "LAYER_FIRST_STAGE_SCHEDULE.json")
    assert_complete(rows, schedule, selected_model["model"])
    lock = read_json(review / "LAYER_SHORTLIST_LOCK.json")
    _baseline_q, baseline = matched_q(rows, "BASELINE", "BASELINE")
    atlas = {int(row["layer"]): row for row in read_json(review / "SOURCE_ATLAS.json")["layers"]}
    layer_metrics: dict[int, dict[str, float]] = {}
    detail: dict[str, Any] = {}
    for layer in lock["candidate_layers"]:
        meaningful_name = f"MEANINGFUL_L{layer}_D50"
        meaningful_q, meaningful = matched_q(rows, "BASELINE", meaningful_name)
        null_q = [
            matched_q(rows, "BASELINE", f"{kind}_L{layer}_D50")[0]
            for kind in ("ISOTROPIC", "SHUFFLED")
        ]
        layer_metrics[int(layer)] = {
            **meaningful,
            "baseline_accuracy": baseline["accuracy"],
            "Q": meaningful_q,
            "null_mean_Q": float(np.mean(null_q)),
            "null_max_Q": float(np.max(null_q)),
        }
        detail[str(layer)] = {**layer_metrics[int(layer)], "null_Q": null_q}
    selected, passed = gate13.select_first_stage_layer(
        layer_metrics,
        {layer: float(atlas[layer]["standardized_paired_effect"]) for layer in layer_metrics},
    )
    direction_source = review / f"SOURCE_DIRECTIONS/L{selected}.npy"
    selected_path = review / "SOURCE_DIRECTIONS/SELECTED_MEANINGFUL.npy"
    shutil.copyfile(direction_source, selected_path)
    archive = np.load(review / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    paired = archive["construction_careful"][:, selected, :] - archive[
        "construction_direct"
    ][:, selected, :]
    meaningful = np.load(selected_path, allow_pickle=False)
    bank, metadata = gate13.final_null_bank(meaningful, paired, selected)
    random_dir = review / "FINAL_RANDOM_DIRECTIONS"
    random_dir.mkdir(exist_ok=True)
    for name, vector in bank.items():
        path = random_dir / f"{name}.npy"
        np.save(path, vector.astype(np.float64), allow_pickle=False)
        metadata["records"][name]["file"] = str(path.relative_to(review))
        metadata["records"][name]["file_sha256"] = gate13.file_sha256(path)
    write_json(review / "FINAL_RANDOM_BANK.json", metadata)
    full = float(atlas[selected]["paired_mean_gap"])
    selected_lock = {
        "status": "FROZEN_PRE_DOSE_CALIBRATION",
        "selected_layer": selected,
        "full_source_displacement": full,
        "meaningful_vector_path": str(selected_path.relative_to(review)),
        "meaningful_vector_hash": gate13.vector_sha256(meaningful),
        "selection_metric": "meaningful Q - null mean Q",
        "semantic_correctness_not_used_for_ranking": True,
        "candidate_pass": {str(key): value for key, value in passed.items()},
    }
    write_json(review / "SELECTED_LAYER_LOCK.json", selected_lock)
    (review / "SELECTED_LAYER_LOCK.md").write_text(
        f"# Gate 13 selected layer lock\n\nLayer `{selected}` is frozen before dose outputs.\n",
        encoding="utf-8",
    )
    schedule = gate13.build_dose_schedule(
        [row["item_id"] for row in read_json(review / "DOSE_CALIBRATION_MANIFEST.json")["items"]],
        selected_model["model"],
    )
    write_json(review / "DOSE_CALIBRATION_SCHEDULE.json", schedule)
    result = {
        "classification": "GATE13_CAUSAL_LAYER_FIRST_STAGE_PASS",
        "layer_metrics": detail,
        "selected_layer": selected,
        "selected_layer_lock": selected_lock,
    }
    write_json(review / "LAYER_FIRST_STAGE_REPORT.json", result)
    (review / "LAYER_FIRST_STAGE_REPORT.md").write_text(
        "# Gate 13 causal layer first-stage\n\n"
        f"Classification: `GATE13_CAUSAL_LAYER_FIRST_STAGE_PASS`; layer `{selected}` frozen.\n",
        encoding="utf-8",
    )
    return result


def analyze_dose(review: Path) -> dict[str, Any]:
    model = read_json(review / "SELECTED_MODEL_LOCK.json")["model"]
    rows = journal_rows(review, "DOSE_CALIBRATION", model)
    schedule = read_json(review / "DOSE_CALIBRATION_SCHEDULE.json")
    assert_complete(rows, schedule, model)
    aliases = []
    for row in rows:
        alias = dict(row)
        if alias["condition"] == "TEXTUAL_CAREFUL":
            alias["condition"] = "TEXTUAL_CAREFUL_REFERENCE"
        elif alias["condition"].startswith("R"):
            alias["condition"] = "RANDOM_" + alias["condition"]
        aliases.append(alias)
    item_ids = sorted({str(row["item_id"]) for row in aliases})
    point = dose_core.point_estimates(aliases, item_ids)
    selected, eligibility, classification = gate13.select_dose(
        point["summaries"]["BASELINE"],
        point["summaries"]["TEXTUAL_CAREFUL_REFERENCE"],
        point["doses"],
        {dose: value["random_Q"] for dose, value in point["doses"].items()},
    )
    result = {
        "classification": classification,
        "selected_dose": selected,
        "summaries": point["summaries"],
        "doses": point["doses"],
        "eligibility": eligibility,
        "selection_rule": "lowest eligible dose: D25, D50, D75, then D100",
        "accuracy_used_only_for_safety": True,
        "G_C_D_used_for_selection": False,
    }
    write_json(review / "DOSE_CALIBRATION_REPORT.json", result)
    (review / "DOSE_CALIBRATION_REPORT.md").write_text(
        "# Gate 13 dose calibration\n\n"
        f"Classification: `{classification}`. Selected dose: `{selected or 'NONE'}`.\n",
        encoding="utf-8",
    )
    if selected is not None:
        layer = read_json(review / "SELECTED_LAYER_LOCK.json")
        alpha = float(layer["full_source_displacement"]) * gate13.DOSE_FRACTIONS[selected]
        lock = {
            "status": "FROZEN_PRE_FINAL_EVALUATION",
            "selected_dose": selected,
            "selected_alpha": alpha,
            "selected_layer": layer["selected_layer"],
            "selection_rule": "lowest eligible prospective Gate-8 dose",
            "accuracy_G_C_D_not_used_for_selection": True,
        }
        write_json(review / "SELECTED_DOSE_LOCK.json", lock)
        (review / "SELECTED_DOSE_LOCK.md").write_text(
            f"# Gate 13 selected dose lock\n\n`{selected}` at alpha `{alpha}` is frozen.\n",
            encoding="utf-8",
        )
        final_schedule = gate13.build_final_schedule(
            [
                row["item_id"]
                for row in read_json(review / "FINAL_EVALUATION_MANIFEST.json")["items"]
            ],
            model,
        )
        write_json(review / "FINAL_EVALUATION_SCHEDULE.json", final_schedule)
    return result


def _final_arrays(
    items: list[str],
    conditions: tuple[str, ...],
    parsed: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        condition: np.asarray(
            [
                [int(not parsed[(item, condition, rollout)]["correct"]) for rollout in (0, 1)]
                for item in items
            ],
            dtype=np.int8,
        )
        for condition in conditions
    }


def _bootstrap_final(
    arrays: dict[str, np.ndarray],
    commitment: dict[str, np.ndarray],
    evaluability: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    baseline = "BASELINE"
    meaningful = "MEANINGFUL_SELECTED"
    randoms = tuple(f"RANDOM_R{i}" for i in range(4))
    rng = np.random.default_rng(gate13.BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = defaultdict(list)
    n = len(arrays[baseline])
    for _ in range(gate13.BOOTSTRAP_RESAMPLES):
        index = rng.integers(0, n, size=n)
        points = {
            condition: audit_two_rollout_estimands(
                arrays[baseline][index], arrays[condition][index]
            )
            for condition in arrays
            if condition != baseline
        }
        point = points[meaningful]
        samples["meaningful:accuracy_change"].append(
            float(arrays[baseline][index].mean() - arrays[meaningful][index].mean())
        )
        samples["meaningful:commitment_validity_change"].append(
            float(commitment[meaningful][index].mean() - commitment[baseline][index].mean())
        )
        samples["meaningful:semantic_evaluability_change"].append(
            float(evaluability[meaningful][index].mean() - evaluability[baseline][index].mean())
        )
        for metric in ("G", "C", "D", "rescue", "damage"):
            samples[f"meaningful:{metric}"].append(float(point[metric]))
        for metric in ("G", "C", "D"):
            values = [points[name][metric] for name in randoms]
            samples[f"meaningful:{metric}_minus_random_mean"].append(
                float(point[metric] - np.mean(values))
            )
            samples[f"meaningful:{metric}_minus_random_max"].append(
                float(point[metric] - np.max(values))
            )
    return {
        name: {
            "estimate": float(np.mean(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "resamples": gate13.BOOTSTRAP_RESAMPLES,
        }
        for name, values in sorted(samples.items())
    }


def analyze_final(review: Path) -> dict[str, Any]:
    model = read_json(review / "SELECTED_MODEL_LOCK.json")["model"]
    rows = journal_rows(review, "FINAL_EVALUATION", model)
    schedule = read_json(review / "FINAL_EVALUATION_SCHEDULE.json")
    assert_complete(rows, schedule, model)
    conditions = (
        "BASELINE",
        "TEXTUAL_CAREFUL",
        "MEANINGFUL_SELECTED",
        "RANDOM_R0",
        "RANDOM_R1",
        "RANDOM_R2",
        "RANDOM_R3",
    )
    parsed = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): reparsed(row)
        for row in rows
    }
    summaries = condition_summary(rows, parsed)
    item_ids = sorted({str(row["item_id"]) for row in rows})
    arrays = _final_arrays(item_ids, conditions, parsed)
    estimands = {
        condition: audit_two_rollout_estimands(arrays["BASELINE"], arrays[condition])
        for condition in conditions[1:]
    }
    b00 = float(np.mean(arrays["BASELINE"][:, 0] * arrays["BASELINE"][:, 1]))
    estimands["BASELINE"] = {
        "B00": b00,
        "O00": 1.0 - b00,
        "baseline_resampling_gain": 1.0 - b00 - summaries["BASELINE"]["accuracy"],
    }
    randoms = tuple(f"RANDOM_R{i}" for i in range(4))
    random_summary: dict[str, dict[str, float]] = {}
    for metric in ("G", "C", "D", "rescue", "damage"):
        values = [float(estimands[name][metric]) for name in randoms]
        random_summary[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    intervals = _bootstrap_final(
        arrays,
        {
            condition: np.asarray(
                [
                    [
                        int(parsed[(item, condition, rollout)]["commitment_valid"])
                        for rollout in (0, 1)
                    ]
                    for item in item_ids
                ]
            )
            for condition in conditions
        },
        {
            condition: np.asarray(
                [
                    [
                        int(parsed[(item, condition, rollout)]["semantic_evaluable"])
                        for rollout in (0, 1)
                    ]
                    for item in item_ids
                ]
            )
            for condition in conditions
        },
    )
    write_json(review / "BOOTSTRAP_INTERVALS.json", intervals)
    point = estimands["MEANINGFUL_SELECTED"]
    full = {
        "accuracy_change": summaries["MEANINGFUL_SELECTED"]["accuracy"]
        - summaries["BASELINE"]["accuracy"],
        "G": point["G"],
        "C": point["C"],
        "D": point["D"],
    }
    loo_rows: list[dict[str, Any]] = []
    loo_sign: dict[str, list[float]] = defaultdict(list)
    for item_index, item_id in enumerate(item_ids):
        keep = np.arange(len(item_ids)) != item_index
        value = audit_two_rollout_estimands(
            arrays["BASELINE"][keep], arrays["MEANINGFUL_SELECTED"][keep]
        )
        record = {
            "left_out_item_id": item_id,
            "accuracy_change": value["accuracy_condition"] - value["accuracy_baseline"],
            "G": value["G"],
            "C": value["C"],
            "D": value["D"],
        }
        loo_rows.append(record)
        for metric in full:
            loo_sign[metric].append(float(record[metric]))
    write_csv(review / "LOO_SENSITIVITY.csv", loo_rows)
    sign_stable = {
        metric: all(value > 0 for value in values) if full[metric] > 0 else False
        for metric, values in loo_sign.items()
    }
    source_replicated = bool(
        summaries["TEXTUAL_CAREFUL"]["commitment_validity"] >= 0.90
        and summaries["TEXTUAL_CAREFUL"]["semantic_evaluability"] >= 0.90
        and summaries["TEXTUAL_CAREFUL"]["mean_tokens"]
        >= 1.5 * summaries["BASELINE"]["mean_tokens"]
        and summaries["TEXTUAL_CAREFUL"]["median_tokens"]
        >= summaries["BASELINE"]["median_tokens"] + 10
    )
    token_denominator = (
        summaries["TEXTUAL_CAREFUL"]["mean_tokens"] - summaries["BASELINE"]["mean_tokens"]
    )
    token_recovery = (
        (summaries["MEANINGFUL_SELECTED"]["mean_tokens"] - summaries["BASELINE"]["mean_tokens"])
        / token_denominator
        if token_denominator > 0
        else None
    )
    style = bool(source_replicated and token_recovery is not None and token_recovery >= 0.50)
    gate9_class, gates = classify_gate9(
        baseline=summaries["BASELINE"],
        controller=summaries["MEANINGFUL_SELECTED"],
        controller_estimands=point,
        random_summary=random_summary,
        bootstrap=intervals,
        loo_sign_stable=sign_stable,
        controller_style_replicated=style,
        source_replicated=source_replicated,
    )
    classification = gate13.map_gate9_classification(gate9_class)
    contrasts = {
        metric: {
            "minus_random_mean": point[metric] - random_summary[metric]["mean"],
            "minus_random_max": point[metric] - random_summary[metric]["max"],
        }
        for metric in ("G", "C", "D")
    }
    summary_rows = [{"condition": key, **value} for key, value in summaries.items()]
    write_csv(review / "CONDITION_SUMMARY.csv", summary_rows)
    contributions = []
    for condition in conditions[1:]:
        for item, contribution in zip(
            item_ids,
            item_contributions(arrays["BASELINE"], arrays[condition]),
            strict=True,
        ):
            contributions.append({"item_id": item, "condition": condition, **contribution})
    write_csv(review / "ITEM_CONTRIBUTIONS.csv", contributions)
    normalized = {
        "G_over_B00": point["G"] / b00 if b00 > 0 else None,
        "rescue_over_baseline_error": point["rescue"] / arrays["BASELINE"].mean(),
        "damage_over_baseline_success": point["damage"] / (1 - arrays["BASELINE"].mean()),
    }
    write_json(review / "NORMALIZED_ESTIMANDS.json", normalized)
    result = {
        "classification": classification,
        "gate9_structural_classification": gate9_class,
        "gates": gates,
        "summaries": summaries,
        "estimands": estimands,
        "random_summary": random_summary,
        "meaningful_random_contrasts": contrasts,
        "source_policy_replicated": source_replicated,
        "token_regime_recovery": token_recovery,
        "loo_sign_stable": sign_stable,
        "normalized_estimands": normalized,
    }
    write_json(review / "ESTIMANDS.json", result)
    (review / "REPORT.md").write_text(
        "# Gate 13 — Cross-model Ministral 3 replication\n\n"
        f"Primary classification: `{classification}`.\n\n"
        "This is fresh-model evidence on reused DEVELOPMENT items. The 57 untouched "
        "CRUXEval IDs, Q2, Q3, and the confirmatory holdout remain untouched.\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("screen", "source-atlas", "first-stage", "dose", "final"), required=True
    )
    parser.add_argument("--model-role", choices=("primary", "fallback"), default="primary")
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    review = args.review_dir.resolve()
    result = {
        "screen": lambda: analyze_screen(review, args.model_role),
        "source-atlas": lambda: analyze_source_atlas(review),
        "first-stage": lambda: analyze_first_stage(review),
        "dose": lambda: analyze_dose(review),
        "final": lambda: analyze_final(review),
    }[args.stage]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
