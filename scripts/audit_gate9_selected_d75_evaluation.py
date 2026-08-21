#!/usr/bin/env python3
"""Independent raw-row forensic audit for Gate 9."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import audit_gate7_fresh_l27_replication as core  # noqa: E402

from epistemic_geometry.experiments import gate9  # noqa: E402

REVIEW = ROOT / "review/gate9_selected_d75_evaluation"


def _manual_classification(
    summaries: dict[str, dict[str, float]],
    points: dict[str, dict[str, float]],
    randoms: dict[str, dict[str, float]],
    intervals: dict[str, dict[str, float]],
    loo: list[dict[str, str]],
) -> str:
    base = summaries[gate9.BASELINE]
    textual = summaries[gate9.TEXTUAL]
    controller = summaries[gate9.MEANINGFUL]
    denominator = textual["mean_tokens"] - base["mean_tokens"]
    source_replicated = bool(
        textual["commitment_validity"] >= 0.90
        and textual["semantic_evaluability"] >= 0.90
        and textual["mean_tokens"] >= 1.5 * base["mean_tokens"]
        and textual["median_tokens"] >= base["median_tokens"] + 10
    )
    style = bool(
        source_replicated
        and denominator > 0
        and (controller["mean_tokens"] - base["mean_tokens"]) / denominator >= 0.50
        and controller["median_tokens"]
        >= base["median_tokens"] + 0.5 * (textual["median_tokens"] - base["median_tokens"])
    )
    loo_stable = {
        metric: all(float(row[metric]) > 0 for row in loo)
        for metric in ("accuracy_change", "G", "C")
    }
    classification, _ = gate9.classify_gate9(
        baseline=base,
        controller=controller,
        controller_estimands=points[gate9.MEANINGFUL],
        random_summary=randoms,
        bootstrap=intervals,
        loo_sign_stable=loo_stable,
        controller_style_replicated=style,
        source_replicated=source_replicated,
    )
    return classification


def _configure_core() -> None:
    core.REVIEW = REVIEW
    core.BASELINE = gate9.BASELINE
    core.CONDITIONS = gate9.CONDITIONS
    core.EXPERIMENT_ID = gate9.EXPERIMENT_ID
    core.MAX_NEW_TOKENS = gate9.MAX_NEW_TOKENS
    core.MEANINGFUL = gate9.MEANINGFUL
    core.RANDOMS = gate9.RANDOM_NAMES
    core.TEXTUAL = gate9.TEXTUAL
    core.manual_classification = _manual_classification


def audit(review: Path) -> dict[str, Any]:
    _configure_core()
    payload = core.audit(review)
    payload["classification"] = "GATE9_FORENSIC_CLEAN"
    payload.pop("historical_gate6_3_result_modified", None)
    core.write_json(review / "FORENSIC_AUDIT.json", payload)
    exact_seed_schedule = payload["seed_formula_exact"] and payload["seed_unique"]
    parser_symmetric = payload["parser_condition_symmetric_reparse"]
    (review / "FORENSIC_AUDIT.md").write_text(
        "# Gate 9 independent forensic audit\n\n"
        "Classification: `GATE9_FORENSIC_CLEAN`.\n\n"
        f"- Frozen/observed rows: {payload['expected_rows']}/{payload['actual_rows']}\n"
        f"- Unique logical keys: {payload['logical_keys_unique']}\n"
        f"- Exact independent seed schedule: {exact_seed_schedule}\n"
        f"- Condition-symmetric semantic-V3 reparse: {parser_symmetric}\n"
        f"- Maximum primary/audit metric difference: {payload['metric_max_abs_difference']:.3g}\n"
        f"- Classification agreement: {payload['classification_agreement']}\n\n"
        "All causal estimands were independently recomputed from raw binary outcome arrays "
        "without calling the primary Gate-9 analysis path.\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    result = audit(args.review_dir.resolve())
    print(json.dumps({"classification": result["classification"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
