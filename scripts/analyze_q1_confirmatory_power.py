#!/usr/bin/env python3
"""Run the sealed-holdout Q1 offline power qualification from DEVELOPMENT rows."""

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
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate6_3_v3 import (  # noqa: E402
    audit_two_rollout_estimands,
)
from epistemic_geometry.experiments.q1_confirmatory_power import (  # noqa: E402
    c_sufficient_features,
    nested_item_bootstrap_power,
    point_c,
)

REVIEW = ROOT / "review/q1_confirmatory_fixed_controllers"
LOCK = REVIEW / "POWER_ANALYSIS_LOCK.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_development_errors(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    path = (ROOT / spec["journal_path"]).resolve()
    permitted = {
        (ROOT / "review/gate9_selected_d75_evaluation/journal.jsonl").resolve(),
        (ROOT / "review/gate13_1_all_layer_causal_atlas/journal.jsonl").resolve(),
    }
    if path not in permitted:
        raise RuntimeError("power analysis source is not an authorized DEVELOPMENT journal")
    if sha256(path) != spec["journal_sha256"]:
        raise RuntimeError(f"DEVELOPMENT journal hash mismatch: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            stage = str(row.get("stage", row.get("phase", "")))
            if stage == spec["stage"] and (
                spec.get("model") is None or str(row.get("model")) == spec["model"]
            ):
                rows.append(row)
    allowed_conditions = [spec["baseline"], spec["meaningful"], *spec["nulls"]]
    rows = [row for row in rows if str(row["condition"]) in allowed_conditions]
    logical = Counter(
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        for row in rows
    )
    if logical and max(logical.values()) != 1:
        raise RuntimeError("duplicate DEVELOPMENT logical rows")
    item_ids = sorted({str(row["item_id"]) for row in rows})
    expected = len(item_ids) * len(allowed_conditions) * 2
    if len(item_ids) != spec["development_n"] or len(rows) != expected:
        raise RuntimeError("incomplete DEVELOPMENT final-evaluation rows")
    if any(int(row.get("retry_count", 0)) != 0 for row in rows):
        raise RuntimeError("unexpected DEVELOPMENT retry provenance")
    by_key = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): row
        for row in rows
    }
    return {
        condition: np.asarray(
            [
                [int(not bool(by_key[(item, condition, rollout)]["correct"])) for rollout in (0, 1)]
                for item in item_ids
            ],
            dtype=np.int8,
        )
        for condition in allowed_conditions
    }


def _crosscheck_points(
    arrays: dict[str, np.ndarray], spec: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    conditions = [spec["meaningful"], *spec["nulls"]]
    direct = np.asarray(
        [
            audit_two_rollout_estimands(arrays[spec["baseline"]], arrays[name])["C"]
            for name in conditions
        ]
    )
    features = c_sufficient_features(
        arrays,
        baseline=spec["baseline"],
        conditions=conditions,
    )
    sufficient = point_c(features)
    if not np.allclose(direct, sufficient, atol=1e-12, rtol=0):
        raise AssertionError("power sufficient-statistic C does not match canonical estimator")
    expected = np.asarray(spec["expected_c_values"], dtype=np.float64)
    if not np.allclose(direct, expected, atol=1e-12, rtol=0):
        raise AssertionError("DEVELOPMENT C values do not match the immutable closeout")
    return features, direct


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Q1 Confirmatory Offline Power Qualification",
        "",
        f"Classification: `{result['classification']}`.",
        "",
        "The exact 57-ID confirmatory set remained sealed. This analysis used only binary",
        "item-level outcomes from the immutable Gate-9 and Gate-13.1 DEVELOPMENT final",
        "evaluations. No prompt, reference, model weight, or holdout outcome was accessed.",
        "",
        "## Frozen planning design",
        "",
        f"- Target N: {result['planning']['n_items']}",
        f"- Outer pseudoexperiments: {result['planning']['outer_replications']}",
        "- Inner item-bootstrap resamples per pseudoexperiment: "
        f"{result['planning']['inner_resamples']}",
        f"- Planning seed: {result['planning']['seed']}",
        "- Interval: two-sided 95% item-percentile bootstrap",
        "- Primary endpoint: C_meaningful > 0",
        "- Adequacy threshold: estimated power >= 0.80 for both models",
        "",
        "## Results",
        "",
        "| model | C power | expected C CI width | specificity power | "
        "expected specificity CI width |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, model in result["models"].items():
        lines.append(
            f"| {name} | {model['meaningful_c_power_lower_bound_gt_zero']:.4f} | "
            f"{model['meaningful_c_expected_interval_width']:.6f} | "
            f"{model['specificity_c_power_lower_bound_gt_zero']:.4f} | "
            f"{model['specificity_c_expected_interval_width']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Null-specificity power is descriptive only and did not alter the frozen primary",
            "endpoint or adequacy decision. Safety-pass probability was not used as a reason",
            "to avoid the confirmatory test.",
            "",
            "## Decision",
            "",
            result["decision_explanation"],
            "",
        ]
    )
    return "\n".join(lines)


def analyze(lock_path: Path = LOCK) -> dict[str, Any]:
    lock = read_json(lock_path)
    if lock["holdout_access_permitted"]:
        raise RuntimeError("offline power lock must prohibit holdout access")
    models: dict[str, Any] = {}
    for index, (name, spec) in enumerate(lock["development_sources"].items()):
        arrays = load_development_errors(spec)
        features, direct = _crosscheck_points(arrays, spec)
        simulation = nested_item_bootstrap_power(
            features,
            n_items=lock["n_items"],
            outer_replications=lock["outer_replications"],
            inner_resamples=lock["inner_resamples_per_pseudoexperiment"],
            seed=lock["planning_seed"] + 1_000_003 * index,
            batch_size=lock["batch_size"],
        )
        simulation["development_n"] = spec["development_n"]
        simulation["development_point_c_meaningful"] = float(direct[0])
        simulation["development_point_c_nulls"] = [float(value) for value in direct[1:]]
        simulation["development_point_c_minus_null_mean"] = float(
            direct[0] - direct[1:].mean()
        )
        models[name] = simulation
    threshold = float(lock["adequacy_threshold"])
    qualified = all(
        model["meaningful_c_power_lower_bound_gt_zero"] >= threshold
        for model in models.values()
    )
    classification = (
        "Q1_CONFIRMATORY_N57_POWER_QUALIFIED"
        if qualified
        else "Q1_CONFIRMATORY_BLOCKED_INSUFFICIENT_N57_POWER"
    )
    result = {
        "classification": classification,
        "planning": {
            "n_items": lock["n_items"],
            "outer_replications": lock["outer_replications"],
            "inner_resamples": lock["inner_resamples_per_pseudoexperiment"],
            "seed": lock["planning_seed"],
            "adequacy_threshold": threshold,
        },
        "models": models,
        "holdout": {
            "status": "SEALED_ASSIGNED_UNACCESSED",
            "content_accessed": False,
            "outcomes_accessed": False,
            "model_inference": False,
        },
        "decision_explanation": (
            "Both models meet the prospectively frozen N=57 primary-C power threshold; "
            "dress-rehearsal and lock preparation may proceed without holdout access."
            if qualified
            else "At least one model fails the prospectively frozen N=57 primary-C power "
            "threshold; stop before holdout access."
        ),
    }
    write_json(REVIEW / "POWER_ANALYSIS.json", result)
    (REVIEW / "POWER_ANALYSIS.md").write_text(render_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=LOCK)
    args = parser.parse_args()
    result = analyze(args.lock.resolve())
    print(
        json.dumps(
            {"classification": result["classification"], "models": result["models"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
