#!/usr/bin/env python3
"""Independent read-only forensic recomputation of Q1 LiveCodeBench Stage B."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_q1_second_task_stage_a2 import score as independent_score  # noqa: E402

from epistemic_geometry.experiments import q1_second_task_stage_b as stage_b  # noqa: E402

REVIEW = ROOT / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit"
SCHEDULE = REVIEW / "STAGE_B_SCHEDULE.json"
MANIFEST = REVIEW / "STAGE_B_FAMILY_MANIFEST.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["stage"]),
        str(row["family_id"]),
        str(row["condition"]),
        int(row["rollout_index"]),
    )


def independent_estimands(baseline: np.ndarray, condition: np.ndarray) -> dict[str, float]:
    b = np.asarray(baseline, dtype=np.float64)
    x = np.asarray(condition, dtype=np.float64)
    n, r = b.shape
    if x.shape != (n, r) or n != 130 or r not in {2, 4}:
        raise ValueError("independent estimator received unexpected dimensions")
    bmean = b.sum(axis=1) / r
    xmean = x.sum(axis=1) / r
    within_b = np.asarray(
        [
            sum(row[a] * row[c] for a in range(r) for c in range(r) if a != c) / (r * (r - 1))
            for row in b
        ]
    )
    within_x = np.asarray(
        [
            sum(row[a] * row[c] for a in range(r) for c in range(r) if a != c) / (r * (r - 1))
            for row in x
        ]
    )
    cross_off = np.asarray(
        [
            sum(brow[a] * xrow[c] for a in range(r) for c in range(r) if a != c) / (r * (r - 1))
            for brow, xrow in zip(b, x, strict=True)
        ]
    )
    u00 = sum(bmean[a] * bmean[c] for a in range(n) for c in range(n) if a != c) / (n * (n - 1))
    u0x = sum(bmean[a] * xmean[c] for a in range(n) for c in range(n) if a != c) / (n * (n - 1))
    b00 = float(within_b.mean())
    b0x = float(np.mean(bmean * xmean))
    rescue = float(np.mean(bmean * (1.0 - xmean)))
    damage = float(np.mean((1.0 - bmean) * xmean))
    return {
        "accuracy_baseline": float(1.0 - b.mean()),
        "accuracy_condition": float(1.0 - x.mean()),
        "B00": b00,
        "O00": 1.0 - b00,
        "B0j": b0x,
        "O0j": 1.0 - b0x,
        "G": b00 - b0x,
        "U00": float(u00),
        "U0j": float(u0x),
        "C": b00 - b0x - float(u00) + float(u0x),
        "D": float(np.mean(within_b + within_x - 2.0 * cross_off)),
        "rescue": rescue,
        "damage": damage,
    }


def independent_bootstrap(
    baseline: np.ndarray, conditions: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(stage_b.BOOTSTRAP_SEED)
    meaningful = np.empty(stage_b.BOOTSTRAP_RESAMPLES, dtype=np.float64)
    contrast = np.empty(stage_b.BOOTSTRAP_RESAMPLES, dtype=np.float64)
    names = ("MEANINGFUL_FIXED_QWEN_L27_D75", *stage_b.RANDOM_NAMES)
    bmean = baseline.mean(axis=1)
    within_b = np.asarray(
        [
            sum(row[a] * row[c] for a in range(4) for c in range(4) if a != c) / 12
            for row in baseline
        ],
        dtype=np.float64,
    )
    condition_means = {name: values.mean(axis=1) for name, values in conditions.items()}
    cursor = 0
    while cursor < stage_b.BOOTSTRAP_RESAMPLES:
        size = min(500, stage_b.BOOTSTRAP_RESAMPLES - cursor)
        indices = rng.integers(0, 130, size=(size, 130))
        values = np.empty((size, 9), dtype=np.float64)
        sampled_bmean = bmean[indices]
        bsum = sampled_bmean.sum(axis=1)
        b00 = within_b[indices].mean(axis=1)
        u00 = (bsum * bsum - (sampled_bmean * sampled_bmean).sum(axis=1)) / (130 * 129)
        for column, name in enumerate(names):
            sampled_xmean = condition_means[name][indices]
            xsum = sampled_xmean.sum(axis=1)
            paired = (sampled_bmean * sampled_xmean).sum(axis=1)
            b0x = paired / 130
            u0x = (bsum * xsum - paired) / (130 * 129)
            values[:, column] = b00 - b0x - u00 + u0x
        meaningful[cursor : cursor + size] = values[:, 0]
        contrast[cursor : cursor + size] = values[:, 0] - values[:, 1:].mean(axis=1)
        cursor += size
    return {
        "C_meaningful": {
            "q025": float(np.quantile(meaningful, 0.025)),
            "q975": float(np.quantile(meaningful, 0.975)),
        },
        "delta_C_nullmean": {
            "q025": float(np.quantile(contrast, 0.025)),
            "q975": float(np.quantile(contrast, 0.975)),
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    tokens = [int(row["generated_token_count"]) for row in rows]
    n = len(rows)
    return {
        "n": n,
        "commitment_validity": sum(bool(row["commitment_valid"]) for row in rows) / n,
        "semantic_evaluability": sum(bool(row["semantic_evaluable"]) for row in rows) / n,
        "accuracy": sum(bool(row["correct"]) for row in rows) / n,
        "generated_tokens_mean": float(sum(tokens) / n),
        "generated_tokens_median": float(median(tokens)),
        "generated_tokens_p90": float(np.quantile(tokens, 0.90)),
        "generated_tokens_p95": float(np.quantile(tokens, 0.95)),
        "generated_tokens_max": int(max(tokens)),
    }


def scalar_differences(left: Any, right: Any, prefix: str = "") -> list[tuple[str, float]]:
    differences: list[tuple[str, float]] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for name in sorted(set(left) & set(right)):
            differences.extend(scalar_differences(left[name], right[name], f"{prefix}.{name}"))
    elif (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        differences.append((prefix.lstrip("."), abs(float(left) - float(right))))
    return differences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = read_jsonl(args.raw_dir / "journal.jsonl")
    schedule = read_json(SCHEDULE)
    expected = {key(row): row for row in schedule}
    observed = {key(row): row for row in raw}
    if len(raw) != 5720 or len(observed) != 5720 or set(observed) != set(expected):
        raise RuntimeError("Stage-B independent schedule completeness failure")
    for logical, row in observed.items():
        locked = expected[logical]
        for field in ("family_id", "item_id", "item_sha256", "seed"):
            if row[field] != locked[field]:
                raise RuntimeError(f"Stage-B independent lock mismatch: {field}")

    parsed = [{**row, **independent_score(row)} for row in raw]
    primary_parsed = read_jsonl(args.analysis_dir / "PARSED_STAGE_B_RECORDS.jsonl")
    primary_by_key = {key(row): row for row in primary_parsed}
    parser_disagreements = [
        list(logical)
        for logical, row in ((key(entry), entry) for entry in parsed)
        if any(
            bool(row[field]) != bool(primary_by_key[logical][field])
            for field in ("commitment_valid", "semantic_evaluable", "correct")
        )
    ]
    summaries = {
        condition: summarize([row for row in parsed if row["condition"] == condition])
        for condition in stage_b.CONDITIONS
    }
    family_order = [row["family_id"] for row in read_json(MANIFEST)["ordered_families"]]
    errors: dict[str, np.ndarray] = {}
    for condition in stage_b.CONDITIONS:
        lookup = {
            (row["family_id"], int(row["rollout_index"])): float(not row["correct"])
            for row in parsed
            if row["condition"] == condition
        }
        errors[condition] = np.asarray(
            [[lookup[(family, rollout)] for rollout in range(4)] for family in family_order]
        )
    baseline = errors["BASELINE"]
    estimands = {
        condition: independent_estimands(baseline, errors[condition])
        for condition in stage_b.CONDITIONS
        if condition != "BASELINE"
    }
    primary_names = ("MEANINGFUL_FIXED_QWEN_L27_D75", *stage_b.RANDOM_NAMES)
    primary_conditions = {name: errors[name] for name in primary_names}
    intervals = independent_bootstrap(baseline, primary_conditions)
    split_halves: dict[str, dict[str, Any]] = {}
    halves = {"A": (0, 1), "B": (2, 3)}
    for half, rollout_ids in halves.items():
        meaningful_c = independent_estimands(
            baseline[:, rollout_ids], errors["MEANINGFUL_FIXED_QWEN_L27_D75"][:, rollout_ids]
        )["C"]
        null_values = {
            name: independent_estimands(baseline[:, rollout_ids], errors[name][:, rollout_ids])["C"]
            for name in stage_b.RANDOM_NAMES
        }
        null_mean = float(np.mean(list(null_values.values())))
        checks = {
            "C_meaningful_gt_zero": meaningful_c > 0,
            "delta_C_nullmean_gt_zero": meaningful_c - null_mean > 0,
            "C_meaningful_gt_mean_nulls": meaningful_c > null_mean,
        }
        split_halves[half] = {
            "C_meaningful": meaningful_c,
            "null_C_values": null_values,
            "null_C_mean": null_mean,
            "delta_C_nullmean": meaningful_c - null_mean,
            "checks": checks,
            "passes": all(checks.values()),
        }
    meaningful_c = estimands["MEANINGFUL_FIXED_QWEN_L27_D75"]["C"]
    null_cs = [estimands[name]["C"] for name in stage_b.RANDOM_NAMES]
    scientific_checks = {
        "P1_C_lower_gt_zero": intervals["C_meaningful"]["q025"] > 0,
        "P2a_delta_C_nullmean_lower_gt_zero": intervals["delta_C_nullmean"]["q025"] > 0,
        "P2b_C_meaningful_gt_every_null": meaningful_c > max(null_cs),
        "split_half_A": split_halves["A"]["passes"],
        "split_half_B": split_halves["B"]["passes"],
    }
    safety_checks = {
        "commitment_validity": summaries["MEANINGFUL_FIXED_QWEN_L27_D75"]["commitment_validity"]
        >= summaries["BASELINE"]["commitment_validity"] - 0.05,
        "semantic_evaluability": summaries["MEANINGFUL_FIXED_QWEN_L27_D75"]["semantic_evaluability"]
        >= summaries["BASELINE"]["semantic_evaluability"] - 0.05,
        "accuracy": summaries["MEANINGFUL_FIXED_QWEN_L27_D75"]["accuracy"]
        >= summaries["BASELINE"]["accuracy"] - 0.10,
    }
    if all(scientific_checks.values()) and all(safety_checks.values()):
        classification = "Q1_SECOND_TASK_FIXED_CONTROLLER_PASS"
    elif all(scientific_checks.values()):
        classification = "Q1_SECOND_TASK_COMPLEMENTARITY_WITH_SAFETY_FAIL"
    else:
        classification = "Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY"
    audit = {
        "journal_sha256": sha256(args.raw_dir / "journal.jsonl"),
        "parser_disagreements": parser_disagreements,
        "summaries": summaries,
        "estimands": estimands,
        "intervals": intervals,
        "split_halves": split_halves,
        "scientific_checks": scientific_checks,
        "safety_checks": safety_checks,
        "classification": classification,
    }
    primary = read_json(args.analysis_dir / "PRIMARY_STAGE_B_RESULTS.json")
    comparisons = scalar_differences(
        {
            "summaries": summaries,
            "estimands": estimands,
            "intervals": intervals,
            "split_halves": split_halves,
        },
        {
            "summaries": primary["summaries"],
            "estimands": primary["estimands"],
            "intervals": {
                "C_meaningful": primary["primary"]["C_meaningful_95_percentile_CI"],
                "delta_C_nullmean": primary["primary"]["delta_C_nullmean_95_percentile_CI"],
            },
            "split_halves": primary["split_halves"],
        },
    )
    max_difference = max((value for _, value in comparisons), default=0.0)
    clean = (
        not parser_disagreements
        and max_difference <= 1e-12
        and classification == primary["decision"]["classification"]
        and scientific_checks == primary["decision"]["scientific_checks"]
        and safety_checks == primary["decision"]["safety_checks"]
    )
    audit.update(
        {
            "maximum_primary_audit_metric_difference": max_difference,
            "classification_agreement": classification == primary["decision"]["classification"],
            "forensic_classification": (
                "Q1_SECOND_TASK_STAGE_B_FORENSIC_CLEAN"
                if clean
                else "Q1_SECOND_TASK_STAGE_B_FORENSIC_DISAGREEMENT"
            ),
        }
    )
    write_json(args.analysis_dir / "INDEPENDENT_STAGE_B_FORENSIC_AUDIT.json", audit)
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
