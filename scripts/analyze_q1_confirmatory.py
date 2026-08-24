#!/usr/bin/env python3
"""Frozen primary analysis for complete two-model Q1 confirmatory journals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_confirmatory as q1  # noqa: E402
from epistemic_geometry.experiments.q1_confirmatory_power import (  # noqa: E402
    c_from_feature_sums,
    c_sufficient_features,
)

REVIEW = ROOT / "review/q1_confirmatory_fixed_controllers"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty confirmatory CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_complete(
    rows: list[dict[str, Any]], schedule: list[dict[str, Any]], model_role: str
) -> list[str]:
    expected = {
        (
            str(row["model_role"]),
            str(row["item_id"]),
            str(row["condition"]),
            int(row["rollout_index"]),
        )
        for row in schedule
        if row["model_role"] == model_role
    }
    observed = q1.completed_keys(rows)
    if observed != expected or len(rows) != 798:
        raise RuntimeError(f"{model_role} journal is not the frozen complete 798-row set")
    if any(str(row.get("parser_version")) != "external-semantic-v3" for row in rows):
        raise RuntimeError("confirmatory journal parser provenance mismatch")
    return sorted({str(row["item_id"]) for row in rows})


def condition_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for condition in q1.CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        tokens = np.asarray([int(row.get("generated_token_count", 0)) for row in selected])
        output[condition] = {
            "n": len(selected),
            "commitment_validity": float(
                np.mean([bool(row["commitment_valid"]) for row in selected])
            ),
            "semantic_evaluability": float(
                np.mean([bool(row["semantic_evaluable"]) for row in selected])
            ),
            "accuracy": float(np.mean([bool(row["correct"]) for row in selected])),
            "mean_tokens": float(np.mean(tokens)),
            "median_tokens": float(np.median(tokens)),
            "max_tokens": float(np.max(tokens)),
        }
    return output


def bootstrap_primary(
    arrays: dict[str, np.ndarray], *, seed: int, resamples: int
) -> dict[str, dict[str, float | int]]:
    conditions = ("MEANINGFUL_FIXED", *q1.RANDOM_NAMES)
    features = c_sufficient_features(arrays, baseline="BASELINE", conditions=conditions)
    rng = np.random.default_rng(seed)
    n = len(features)
    meaningful_values = np.empty(resamples, dtype=np.float64)
    specificity_values = np.empty(resamples, dtype=np.float64)
    batch_size = 2_000
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(0, n, size=(stop - start, n))
        estimates = c_from_feature_sums(features[indices].sum(axis=1), n)
        meaningful_values[start:stop] = estimates[:, 0]
        specificity_values[start:stop] = estimates[:, 0] - estimates[:, 1:].mean(axis=1)

    def summary(values: np.ndarray) -> dict[str, float | int]:
        return {
            "estimate": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "resamples": resamples,
            "seed": seed,
        }

    return {
        "C_meaningful": summary(meaningful_values),
        "delta_C_nullmean": summary(specificity_values),
    }


def analyze_model(
    rows: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    model_role: str,
    *,
    bootstrap_resamples: int = q1.BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    item_ids = assert_complete(rows, schedule, model_role)
    arrays = q1.error_arrays(rows, item_ids)
    summaries = condition_summaries(rows)
    estimands = q1.primary_estimands(arrays)
    intervals = bootstrap_primary(
        arrays,
        seed=q1.BOOTSTRAP_SEEDS[model_role],
        resamples=bootstrap_resamples,
    )
    decision = q1.classify_model(
        summaries=summaries, estimands=estimands, intervals=intervals
    )
    null_c = [estimands[name]["C"] for name in q1.RANDOM_NAMES]
    meaningful = estimands["MEANINGFUL_FIXED"]
    loo: list[dict[str, Any]] = []
    for index, item_id in enumerate(item_ids):
        keep = np.arange(len(item_ids)) != index
        point = q1.primary_estimands({name: values[keep] for name, values in arrays.items()})
        current = point["MEANINGFUL_FIXED"]
        current_null = [point[name]["C"] for name in q1.RANDOM_NAMES]
        loo.append(
            {
                "model_role": model_role,
                "left_out_item_id": item_id,
                "C": current["C"],
                "delta_C_nullmean": current["C"] - float(np.mean(current_null)),
                "G": current["G"],
                "D": current["D"],
            }
        )
    return {
        "model_role": model_role,
        "n_items": len(item_ids),
        "summaries": summaries,
        "estimands": estimands,
        "intervals": intervals,
        "meaningful_C": meaningful["C"],
        "null_C_values": dict(zip(q1.RANDOM_NAMES, null_c, strict=True)),
        "delta_C_nullmean": meaningful["C"] - float(np.mean(null_c)),
        "model_pass": decision["pass"],
        "decision_checks": decision["checks"],
        "loo": loo,
    }


def analyze(review: Path = REVIEW) -> dict[str, Any]:
    lock = json.loads((review / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if lock["status"] != "CONFIRMATORY_LOCKED_PRE_HOLDOUT":
        raise RuntimeError("confirmatory analysis requires the final pre-holdout lock")
    schedule_lock = json.loads(
        (review / "SEED_SCHEDULE_LOCK.json").read_text(encoding="utf-8")
    )
    models = {}
    loo_rows = []
    for model_role in ("Qwen", "Ministral"):
        journal = review / f"journal_{model_role.lower()}.jsonl"
        rows = read_jsonl(journal)
        result = analyze_model(rows, schedule_lock["schedules"][model_role], model_role)
        loo_rows.extend(result.pop("loo"))
        models[model_role] = result
    classification = q1.cross_model_classification(
        bool(models["Qwen"]["model_pass"]), bool(models["Ministral"]["model_pass"])
    )
    result = {
        "classification": classification,
        "models": models,
        "analysis_lock": "ANALYSIS_LOCK.json",
        "holdout_outcome_reveal_after_both_journals_complete": True,
    }
    write_json(review / "CONFIRMATORY_RESULTS.json", result)
    write_csv(review / "LOO_SENSITIVITY.csv", loo_rows)
    for model_role, model in models.items():
        write_json(review / f"{model_role.upper()}_CONFIRMATORY_REPORT.json", model)
    write_json(
        review / "CROSS_MODEL_CONFIRMATORY_REPORT.json",
        {
            "classification": classification,
            "qwen_pass": models["Qwen"]["model_pass"],
            "ministral_pass": models["Ministral"]["model_pass"],
        },
    )
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
