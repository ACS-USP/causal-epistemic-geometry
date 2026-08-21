#!/usr/bin/env python3
"""Primary Gate 9 analysis built on the canonical two-rollout implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_gate7_fresh_l27_replication as core  # noqa: E402

from epistemic_geometry.experiments import gate9  # noqa: E402

REVIEW = ROOT / "review/gate9_selected_d75_evaluation"


def _configure_core() -> None:
    core.REVIEW = REVIEW
    core.BASELINE = gate9.BASELINE
    core.BOOTSTRAP_RESAMPLES = gate9.BOOTSTRAP_RESAMPLES
    core.BOOTSTRAP_SEED = gate9.BOOTSTRAP_SEED
    core.CONDITIONS = gate9.CONDITIONS
    core.MAX_NEW_TOKENS = gate9.MAX_NEW_TOKENS
    core.MEANINGFUL = gate9.MEANINGFUL
    core.RANDOM_NAMES = gate9.RANDOM_NAMES
    core.TEXTUAL = gate9.TEXTUAL
    core.classify_gate7 = gate9.classify_gate9


def _write_gate9_report(review: Path, result: dict[str, Any]) -> None:
    summaries = result["summaries"]
    estimands = result["estimands"]
    contrasts = result["meaningful_random_contrasts"]
    source = json.loads((review / "SOURCE_REFERENCE_SUMMARY.json").read_text(encoding="utf-8"))
    binding = json.loads((review / "EXPERIMENT_SOURCE_COMMIT.json").read_text(encoding="utf-8"))
    condition_rows = "".join(
        f"| `{condition}` | {summaries[condition]['commitment_validity']:.4f} | "
        f"{summaries[condition]['semantic_evaluability']:.4f} | "
        f"{summaries[condition]['accuracy']:.4f} | {summaries[condition]['mean_tokens']:.2f} / "
        f"{summaries[condition]['median_tokens']:.1f} / "
        f"{summaries[condition]['max_tokens']:.0f} |\n"
        for condition in gate9.CONDITIONS
    )
    point = estimands[gate9.MEANINGFUL]
    accuracy_change = (
        summaries[gate9.MEANINGFUL]["accuracy"] - summaries[gate9.BASELINE]["accuracy"]
    )
    contrast_lines = {
        metric: (
            f"{contrasts[metric]['minus_random_mean']:.6f} / "
            f"{contrasts[metric]['minus_random_max']:.6f}"
        )
        for metric in ("G", "C", "D")
    }
    report = f"""# Gate 9 — Fresh D75 Selected-Dose Evaluation

Primary classification: `{result["classification"]}`.

Source-policy classification: `{result["source_policy_classification"]}`.

Experiment source commit: `{binding["experiment_source_commit"]}`.

This is a 100-item independent DEVELOPMENT evaluation of the exact frozen L27
paired-mean plus controller at eta `{gate9.ETA}`. Seven conditions and two
independent rollouts produce 1,400 scientific trajectories. No controller,
layer, dose, item, or parser selection occurred after outcomes.

## Conditions

| condition | commitment | evaluability | accuracy | mean / median / max tokens |
|---|---:|---:|---:|---:|
{condition_rows}

## Meaningful D75 estimands

- Accuracy difference: {accuracy_change:.6f}
- G: {point["G"]:.6f}
- C: {point["C"]:.6f}
- D: {point["D"]:.6f}
- Rescue: {point["rescue"]:.6f}
- Damage: {point["damage"]:.6f}
- G minus random mean/max: {contrast_lines["G"]}
- C minus random mean/max: {contrast_lines["C"]}
- D minus random mean/max: {contrast_lines["D"]}

## Frozen guards and source anchor

- Commitment-validity guard: {result["gates"]["commitment_validity_guard"]}
- Semantic-evaluability guard: {result["gates"]["semantic_evaluability_guard"]}
- Competence guard: {result["gates"]["competence_guard"]}
- CAREFUL source: `{source["classification"]}`
- CAREFUL accuracy-gain fraction recovered: {source["accuracy_gain_recovered_fraction"]}
- CAREFUL token-increase fraction recovered: {source["token_gain_recovered_fraction"]}

## Interpretation boundary

The classification above was applied mechanically after all rows were collected.
This is DEVELOPMENT evidence only. It is not Q2, character-count replication,
confirmatory holdout evidence, or execution of Gate 10.
"""
    (review / "REPORT.md").write_text(report, encoding="utf-8")


def analyze(review: Path) -> dict[str, Any]:
    _configure_core()
    result = core.analyze(review)
    source = json.loads((review / "SOURCE_REFERENCE_SUMMARY.json").read_text(encoding="utf-8"))
    if source["classification"] != "CAREFUL_SOURCE_REPLICATED":
        classification, gates = gate9.classify_gate9(
            baseline=result["summaries"][gate9.BASELINE],
            controller=result["summaries"][gate9.MEANINGFUL],
            controller_estimands=result["estimands"][gate9.MEANINGFUL],
            random_summary=result["random_summary"],
            bootstrap=json.loads((review / "BOOTSTRAP_INTERVALS.json").read_text(encoding="utf-8")),
            loo_sign_stable=result["loo_sign_stable"],
            controller_style_replicated=source["activation_controller_style_regime_replicated"],
            source_replicated=False,
        )
        result["classification"] = classification
        result["gates"] = gates
    result.pop("historical_gate6_3_classification", None)
    result["gate8_selected_dose"] = "D75"
    result["gate8_selected_eta"] = gate9.ETA
    result["source_policy_classification"] = source["classification"]
    core.write_json(review / "ESTIMANDS.json", result)
    cost = json.loads((review / "COST.json").read_text(encoding="utf-8"))
    cost["hard_ceiling_usd"] = 3.50
    cost["target_usd"] = 1.75
    core.write_json(review / "COST.json", cost)
    _write_gate9_report(review, result)
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
