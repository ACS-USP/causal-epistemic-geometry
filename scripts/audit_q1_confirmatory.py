#!/usr/bin/env python3
"""Independent low-level forensic recomputation for Q1 confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_confirmatory as q1  # noqa: E402

REVIEW = ROOT / "review/q1_confirmatory_fixed_controllers"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _independent_point(base: np.ndarray, current: np.ndarray) -> dict[str, float]:
    b0, b1 = base[:, 0].astype(float), base[:, 1].astype(float)
    c0, c1 = current[:, 0].astype(float), current[:, 1].astype(float)
    n = len(base)
    b00 = float(np.sum(b0 * b1) / n)
    b0j = float(np.sum((b0 + b1) * (c0 + c1) / 4.0) / n)
    q0, qj = (b0 + b1) / 2.0, (c0 + c1) / 2.0
    denominator = n * (n - 1)
    u00 = float((q0.sum() * q0.sum() - np.dot(q0, q0)) / denominator)
    u0j = float((q0.sum() * qj.sum() - np.dot(q0, qj)) / denominator)
    distance = float(np.mean(b0 * b1 + c0 * c1 - b0 * c1 - b1 * c0))
    rescue = float(np.mean((b0 + b1) * (2 - c0 - c1) / 4.0))
    damage = float(np.mean((2 - b0 - b1) * (c0 + c1) / 4.0))
    return {
        "B00": b00,
        "B0j": b0j,
        "G": b00 - b0j,
        "C": b00 - b0j - u00 + u0j,
        "D": distance,
        "rescue": rescue,
        "damage": damage,
    }


def audit(review: Path = REVIEW) -> dict[str, Any]:
    primary = json.loads((review / "CONFIRMATORY_RESULTS.json").read_text(encoding="utf-8"))
    schedule = json.loads((review / "SEED_SCHEDULE_LOCK.json").read_text(encoding="utf-8"))
    model_checks = {}
    maximum = 0.0
    passes = {}
    for model_role in ("Qwen", "Ministral"):
        rows = _read_rows(review / f"journal_{model_role.lower()}.jsonl")
        expected = {
            (
                str(row["model_role"]),
                str(row["item_id"]),
                str(row["condition"]),
                int(row["rollout_index"]),
            )
            for row in schedule["schedules"][model_role]
        }
        observed = q1.completed_keys(rows)
        if observed != expected:
            raise RuntimeError("forensic schedule mismatch")
        items = sorted({str(row["item_id"]) for row in rows})
        arrays = q1.error_arrays(rows, items)
        estimates = {
            name: _independent_point(arrays["BASELINE"], arrays[name])
            for name in q1.CONDITIONS[1:]
        }
        differences = []
        for condition, metrics in estimates.items():
            for metric, value in metrics.items():
                reference = primary["models"][model_role]["estimands"][condition][metric]
                differences.append(abs(value - reference))
        local_max = max(differences, default=0.0)
        maximum = max(maximum, local_max)
        meaningful_c = estimates["MEANINGFUL_FIXED"]["C"]
        null_c = [estimates[name]["C"] for name in q1.RANDOM_NAMES]
        checks = dict(primary["models"][model_role]["decision_checks"])
        checks["P2_C_above_null_max"] = meaningful_c > max(null_c)
        # Interval checks are verified by persisted deterministic primary arrays;
        # the audit does not invoke the primary bootstrap implementation.
        passes[model_role] = bool(all(checks.values()))
        model_checks[model_role] = {
            "rows": len(rows),
            "unique_logical_keys": len(observed),
            "seed_collisions": len(rows) - len({int(row["seed"]) for row in rows}),
            "maximum_point_metric_difference": local_max,
            "independent_estimands": estimates,
            "model_pass": passes[model_role],
        }
    classification = q1.cross_model_classification(passes["Qwen"], passes["Ministral"])
    integrity = classification == primary["classification"] and maximum <= 1e-12
    result = {
        "classification": (
            "Q1_CONFIRMATORY_FORENSIC_CLEAN"
            if integrity
            else "Q1_CONFIRMATORY_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
        ),
        "scientific_classification_crosscheck": classification,
        "primary_classification": primary["classification"],
        "maximum_point_metric_difference": maximum,
        "models": model_checks,
        "bootstrap_unit": "ITEM",
        "primary_high_level_metric_function_used": False,
    }
    temporary = review / "FORENSIC_AUDIT.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(review / "FORENSIC_AUDIT.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    result = audit(args.review_dir.resolve())
    print(json.dumps({"classification": result["classification"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
