"""Pure contracts for Gate 10 cross-domain character-count replication."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.benchmarks.v4.character_count import _make_item
from epistemic_geometry.experiments.gate6_3 import (
    bank_geometry,
    single_layer_random_bank,
    standardized_delta,
    vector_sha256,
)
from epistemic_geometry.experiments.gate7 import REFERENCE_SCALE, file_sha256
from epistemic_geometry.reproducibility import canonical_json, stable_digest, stable_seed

EXPERIMENT_ID = "GATE10_CROSS_DOMAIN_CHARCOUNT"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 27
MAX_NEW_TOKENS = 4096
SELECTION_NAMESPACE = "GATE10-CROSS-DOMAIN-CHARCOUNT-V1"
GENERATOR_VERSION = "v4-charcount-gate10-cross-domain-1"
PARSER_VERSION = "character-count-semantic-v3"
BASELINE = "BASELINE"
TEXTUAL = "TEXTUAL_CAREFUL_CHARCOUNT_REFERENCE"
MEANINGFUL = "MEANINGFUL_L27_D75"
RANDOM_NAMES = tuple(f"RANDOM_L27_D75_R{i}" for i in range(4))
CONDITIONS = (BASELINE, TEXTUAL, MEANINGFUL, *RANDOM_NAMES)
ETA = 9.637427952852196
CONTROLLER_HASH = "e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260824
SYSTEM_CAREFUL = (
    "You are a meticulous character counter. Scan the entire string systematically, "
    "keep a running count of every occurrence of the requested character, verify the "
    "count with an independent second pass, and end with exactly one line in the form "
    "FINAL: <integer>."
)


def _walk_records(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_records(child)


def historical_charcount_records(
    review: Path, *, gate10_output: Path | None = None
) -> dict[str, Any]:
    ids: set[str] = set()
    seeds: set[int] = set()
    texts: set[str] = set()
    hashes: set[str] = set()
    files: list[str] = []
    for path in sorted(review.rglob("*.json*")):
        if gate10_output is not None and (path == gate10_output or gate10_output in path.parents):
            continue
        if path.name.endswith(".tar.gz"):
            continue
        try:
            if path.suffix == ".jsonl":
                payloads = [json.loads(line) for line in path.read_text().splitlines() if line]
            else:
                payloads = [json.loads(path.read_text())]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        matched = False
        for payload in payloads:
            for row in _walk_records(payload):
                is_char = bool(
                    row.get("stratum") == "FRESH_PSEUDOWORD_LONG"
                    or "charcount" in str(row.get("item_id", "")).lower()
                    or (
                        "target_character" in row
                        and ("text" in row or "exact_string" in row)
                        and ("answer" in row or "exact_oracle" in row)
                    )
                )
                if not is_char:
                    continue
                matched = True
                if row.get("item_id") is not None:
                    ids.add(str(row["item_id"]))
                for key in ("seed", "generator_seed"):
                    if isinstance(row.get(key), int):
                        seeds.add(int(row[key]))
                for key in ("text", "exact_string"):
                    if isinstance(row.get(key), str):
                        texts.add(str(row[key]))
                if row.get("item_hash") is not None:
                    hashes.add(str(row["item_hash"]))
        if matched:
            files.append(str(path))
    return {
        "item_ids": sorted(ids),
        "generator_seeds": sorted(seeds),
        "exact_strings": sorted(texts),
        "item_hashes": sorted(hashes),
        "source_files": files,
    }


def generate_fresh_manifest(
    historical: Mapping[str, Sequence[Any]], n_items: int = 200
) -> dict[str, Any]:
    excluded_ids = set(map(str, historical["item_ids"]))
    excluded_seeds = set(map(int, historical["generator_seeds"]))
    excluded_texts = set(map(str, historical["exact_strings"]))
    excluded_hashes = set(map(str, historical["item_hashes"]))
    items: list[dict[str, Any]] = []
    source_index = 0
    while len(items) < n_items:
        seed = stable_seed(SELECTION_NAMESPACE, "ITEM", source_index)
        item = _make_item("FRESH_PSEUDOWORD_LONG", source_index, seed=seed)
        record = item.to_record()
        record.update(
            {
                "item_id": f"gate10_charcount_{len(items):03d}",
                "source_index": source_index,
                "generator_seed": seed,
                "generator_version": GENERATOR_VERSION,
                "reference_answer": str(item.answer),
                "reference_canonical_type": "int",
                "source_revision": GENERATOR_VERSION,
                "evaluator": PARSER_VERSION,
                "benchmark": "FRESH_PSEUDOWORD_LONG",
                "subtask": "procedural_character_count",
            }
        )
        record["prompt"] = (
            f"How many times does the letter '{item.target_character}' appear in "
            f"'{item.text}'?\nReturn exactly one integer in this form:\nFINAL: <integer>"
        )
        record["prompt_hash"] = stable_digest("V4-CHARCOUNT-PROMPT", record["prompt"])
        record["item_hash"] = stable_digest("V4-CHARCOUNT-ITEM", canonical_json(record))
        collision = (
            record["item_id"] in excluded_ids
            or seed in excluded_seeds
            or item.text in excluded_texts
            or record["item_hash"] in excluded_hashes
            or any(existing["text"] == item.text for existing in items)
        )
        source_index += 1
        if collision:
            continue
        record["metadata"] = {
            "generator_version": GENERATOR_VERSION,
            "generator_seed": seed,
            "stratum": "FRESH_PSEUDOWORD_LONG",
            "text": item.text,
            "target_character": item.target_character,
            "exact_oracle": item.answer,
            "item_hash": record["item_hash"],
            "length": len(item.text),
            "target_density": item.answer / len(item.text),
            "alphabet_size": len(set(item.text)),
            "source_index": record["source_index"],
            "reference_canonical_type": "int",
            "response_channel": "character_count_semantic",
        }
        items.append(record)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "instrument": "FRESH_PSEUDOWORD_LONG",
        "generator_version": GENERATOR_VERSION,
        "namespace": SELECTION_NAMESPACE,
        "n_items": len(items),
        "items": items,
        "manifest_hash": stable_digest("GATE10-CHARCOUNT-MANIFEST", canonical_json(items)),
    }


def gate10_random_bank(meaningful: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seeds = tuple(stable_seed("GATE10-L27-RANDOM-BANK-V1", index) for index in range(4))
    raw = single_layer_random_bank(meaningful, seeds=seeds)
    bank = {name: raw[f"R{i}"] for i, name in enumerate(RANDOM_NAMES)}
    geometry = bank_geometry(meaningful, bank)
    if not all(
        geometry[key]
        for key in (
            "unit_norm_pass",
            "meaningful_orthogonality_pass",
            "random_pairwise_orthogonality_pass",
        )
    ):
        raise RuntimeError("Gate 10 random-bank geometry failed")
    records = {
        name: {
            "seed": int(seed),
            "vector_sha256": vector_sha256(bank[name]),
            "norm": float(np.linalg.norm(bank[name])),
            "delta_norm": float(
                np.linalg.norm(
                    standardized_delta(bank[name], eta=ETA, reference_scale=REFERENCE_SCALE)
                )
            ),
        }
        for name, seed in zip(RANDOM_NAMES, seeds, strict=True)
    }
    return bank, {"seeds": list(seeds), "records": records, "geometry": geometry}


def build_schedule(item_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_index, item_id in enumerate(item_ids):
        for rollout in (0, 1):
            order = sorted(
                CONDITIONS,
                key=lambda c: (stable_digest(SELECTION_NAMESPACE, "ORDER", item_id, rollout, c), c),
            )
            for order_index, condition in enumerate(order):
                rows.append(
                    {
                        "phase": EXPERIMENT_ID,
                        "item_index": item_index,
                        "item_id": item_id,
                        "condition": condition,
                        "condition_order": order_index,
                        "rollout_index": rollout,
                        "seed": stable_seed(EXPERIMENT_ID, item_id, condition, rollout),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                    }
                )
    keys = [(r["item_id"], r["condition"], r["rollout_index"]) for r in rows]
    seeds = [r["seed"] for r in rows]
    if len(keys) != len(set(keys)) or len(seeds) != len(set(seeds)):
        raise RuntimeError("Gate 10 schedule key or seed collision")
    return rows


def opportunity(baseline_errors: np.ndarray, summary: Mapping[str, float]) -> dict[str, Any]:
    errors = np.asarray(baseline_errors, dtype=np.int8)
    b00 = float(np.mean(errors[:, 0] * errors[:, 1]))
    double_wrong = int(np.sum(errors[:, 0] * errors[:, 1]))
    any_correct = int(np.sum(np.any(errors == 0, axis=1)))
    passed = bool(
        summary["commitment_validity"] >= 0.95
        and summary["semantic_evaluability"] >= 0.95
        and 0.55 <= summary["accuracy"] <= 0.95
        and b00 >= 0.04
        and double_wrong >= 8
        and any_correct >= 20
    )
    return {
        "classification": "CHARCOUNT_OPPORTUNITY_PASS" if passed else "CHARCOUNT_OPPORTUNITY_FAIL",
        "pass": passed,
        "B00": b00,
        "O00": 1 - b00,
        "double_wrong_items": double_wrong,
        "correct_in_at_least_one_items": any_correct,
        "pooled_accuracy": summary["accuracy"],
        "pooled_error_rate": 1 - summary["accuracy"],
    }


def classify_gate10(
    *,
    baseline: Mapping[str, float],
    controller: Mapping[str, float],
    point: Mapping[str, float],
    random_summary: Mapping[str, Mapping[str, float]],
    bootstrap: Mapping[str, Mapping[str, float]],
    loo: Mapping[str, bool],
    opportunity_pass: bool,
    style_transfer: bool,
    accuracy_bootstrap_positive: bool,
) -> tuple[str, dict[str, Any]]:
    commitment = (
        controller["commitment_validity"] >= 0.95
        and controller["commitment_validity"] >= baseline["commitment_validity"] - 0.03
    )
    evaluability = (
        controller["semantic_evaluability"] >= 0.95
        and controller["semantic_evaluability"] >= baseline["semantic_evaluability"] - 0.03
    )
    competence = controller["accuracy"] >= baseline["accuracy"] - 0.05
    safe = bool(commitment and evaluability and competence)
    above_mean = {m: point[m] > random_summary[m]["mean"] for m in ("G", "C", "D")}
    above_max = {m: point[m] > random_summary[m]["max"] for m in ("G", "C", "D")}
    g_norm = point["G_norm"]
    strong_keys = (
        "meaningful:G",
        "meaningful:C",
        "meaningful:D",
        "meaningful:G_minus_random_mean",
        "meaningful:C_minus_random_mean",
    )
    strong = bool(
        opportunity_pass
        and safe
        and point["G"] >= 0.03
        and point["C"] >= 0.015
        and point["D"] >= 0.04
        and point["G"] - random_summary["G"]["mean"] >= 0.025
        and point["C"] - random_summary["C"]["mean"] >= 0.015
        and point["D"] - random_summary["D"]["mean"] >= 0.03
        and all(above_max.values())
        and g_norm >= 0.15
        and point["rescue"] >= point["damage"]
        and all(bootstrap[k]["q025"] > 0 for k in strong_keys)
        and all(loo.get(k, False) for k in ("G", "C", "D"))
    )
    minimum = bool(
        opportunity_pass
        and safe
        and all(point[m] > 0 for m in ("G", "C", "D"))
        and all(above_mean.values())
        and sum(above_max.values()) >= 2
        and g_norm >= 0.08
        and point["rescue"] >= point["damage"]
    )
    movement = bool(
        opportunity_pass and safe and point["D"] >= 0.03 and above_mean["D"] and above_max["D"]
    )
    acc_gain = controller["accuracy"] - baseline["accuracy"]
    if not opportunity_pass:
        classification = "GATE10_INSTRUMENT_CEILING_OR_FLOOR"
    elif not safe:
        classification = "GATE10_CROSS_DOMAIN_DESTRUCTIVE"
    elif strong:
        classification = "GATE10_STRONG_CROSS_DOMAIN_USEFUL_COMPLEMENTARITY"
    elif minimum:
        classification = "GATE10_MINIMUM_CROSS_DOMAIN_CONTROL_SIGNAL"
    elif movement:
        classification = "GATE10_CROSS_DOMAIN_ERROR_PROFILE_MOVEMENT_ONLY"
    elif acc_gain > 0 and accuracy_bootstrap_positive:
        classification = "GATE10_CROSS_DOMAIN_COMPETENCE_GAIN_WITHOUT_COMPLEMENTARITY"
    elif style_transfer:
        classification = "GATE10_CAREFUL_STYLE_TRANSFER_ONLY"
    else:
        classification = "GATE10_NO_CROSS_DOMAIN_TRANSFER"
    return classification, {
        "opportunity": opportunity_pass,
        "commitment_guard": bool(commitment),
        "evaluability_guard": bool(evaluability),
        "competence_guard": bool(competence),
        "strong": strong,
        "minimum": minimum,
        "movement": movement,
        "above_random_mean": above_mean,
        "above_random_max": above_max,
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "historical_charcount_records",
    "generate_fresh_manifest",
    "gate10_random_bank",
    "build_schedule",
    "opportunity",
    "classify_gate10",
    "file_sha256",
    "vector_sha256",
]
