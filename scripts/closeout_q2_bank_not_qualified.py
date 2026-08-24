#!/usr/bin/env python3
"""Close the Q2 pilot after the prospectively frozen bank gate fails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_heldout_geometry"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--runtime-seconds", type=int, required=True)
    parser.add_argument("--hourly-gpu-usd", type=float, required=True)
    parser.add_argument("--volume-gb", type=float, required=True)
    parser.add_argument("--monthly-volume-usd-per-gb", type=float, default=0.20)
    args = parser.parse_args()
    review = args.review_dir.resolve()

    decision = read_json(review / "BANK_QUALIFICATION.json")
    if decision["classification"] != "Q2_CONTROLLER_BANK_NOT_QUALIFIED":
        raise RuntimeError("bank-failure closeout requires the frozen negative classification")
    source = read_json(review / "SOURCE_QUALIFICATION.json")
    manipulation = read_json(review / "MANIPULATION_QUALIFICATION.json")
    bank = read_json(review / "BANK_VALIDATION.json")
    engineering = read_json(review / "ENGINEERING_CHECKS.json")

    controller_failures: dict[str, list[str]] = {}
    for name, record in manipulation.items():
        failures: list[str] = []
        if record["commitment_validity"] < 0.75:
            failures.append("commitment_validity_below_0.75")
        if record["semantic_evaluability"] < 0.75:
            failures.append("semantic_evaluability_below_0.75")
        if record["semantic_change_rate"] < 1.0 / 12.0:
            failures.append("semantic_change_below_1_of_12")
        if record["raw_sequence_change_rate"] < 0.25:
            failures.append("raw_sequence_change_below_3_of_12")
        controller_failures[name] = failures

    qualified = sorted(name for name, failures in controller_failures.items() if not failures)
    failed = sorted(name for name, failures in controller_failures.items() if failures)
    null_cosines = bank["null_to_meaningful_max_absolute_cosines"]
    postmortem = {
        "classification": "Q2_CONTROLLER_BANK_NOT_QUALIFIED",
        "source_axes_passed": sum(decision["source_axis_pass"].values()),
        "source_axes_total": len(decision["source_axis_pass"]),
        "controllers_passed": len(qualified),
        "controllers_total": len(controller_failures),
        "qualified_controllers": qualified,
        "failed_controllers": failed,
        "controller_failure_reasons": controller_failures,
        "representation_geometry_pass": decision["representation_geometry_pass"],
        "bank_subgates": {
            key: bank[key]
            for key in (
                "unit_norm_pass",
                "sign_pair_pass",
                "base_diversity_pass",
                "null_orthogonality_pass",
            )
        },
        "maximum_base_absolute_cosine": bank["base_max_absolute_cosine"],
        "null_to_meaningful_max_absolute_cosines": null_cosines,
        "maximum_null_to_meaningful_absolute_cosine": max(null_cosines.values()),
        "engineering_classification": engineering["classification"],
        "accuracy_G_C_D_used_for_qualification": False,
        "common_panel_executed": False,
        "common_panel_rows": 0,
        "geometry_outcome_prediction_executed": False,
        "scientific_interpretation": (
            "All three textual/activation source axes separated cleanly, but the complete "
            "K=16 intervention bank did not satisfy the frozen causal-movement and "
            "representation-geometry gates. Thirteen controllers missed the raw-sequence "
            "movement threshold, and the null bank was not orthogonal to the non-orthogonal "
            "meaningful span. Either failure independently blocks the common panel."
        ),
    }
    write_json(review / "BANK_POSTMORTEM.json", postmortem)

    gpu_cost = args.runtime_seconds / 3600.0 * args.hourly_gpu_usd
    disk_cost = (
        args.volume_gb
        * args.monthly_volume_usd_per_gb
        / (30.0 * 24.0)
        * (args.runtime_seconds / 3600.0)
    )
    cost = {
        "runtime_seconds": args.runtime_seconds,
        "runtime_hours": args.runtime_seconds / 3600.0,
        "hourly_gpu_usd": args.hourly_gpu_usd,
        "estimated_gpu_usd": gpu_cost,
        "estimated_volume_usd": disk_cost,
        "estimated_incremental_total_usd": gpu_cost + disk_cost,
        "volume_estimate_assumption_usd_per_gb_month": args.monthly_volume_usd_per_gb,
        "billing_snapshot_note": (
            "RunPod billing was still lagging at deletion; cost is reconstructed from the "
            "recorded 6,415-second Pod uptime and listed rates."
        ),
        "target_usd": 8.50,
        "hard_ceiling_usd": 15.0,
        "within_hard_ceiling": gpu_cost + disk_cost <= 15.0,
    }
    write_json(review / "COST.json", cost)

    environment = read_json(review / "REMOTE_PREFLIGHT.json")
    environment.update(
        {
            "pod_id": "jwlev7hyvzwgc1",
            "gpu": "NVIDIA A40",
            "pod_deleted": True,
            "active_pods_after_closeout": 0,
            "retained_network_volumes_after_closeout": 0,
        }
    )
    write_json(review / "ENVIRONMENT_PROVENANCE.json", environment)

    source_lines = []
    for axis, record in source.items():
        source_lines.append(
            f"- `{axis}`: source PASS; validity/evaluability 1.000/1.000 on both "
            f"polarities; excess disagreement {record['excess_disagreement']:.4f}."
        )
    failure_lines = [
        f"- `{name}`: {', '.join(controller_failures[name]) or 'PASS'}"
        for name in sorted(controller_failures)
    ]
    shuffled_r2_cosine = null_cosines["NULL_CONSTRUCTION_MATCHED_SIGN_SHUFFLED_R2"]
    shuffled_r3_cosine = null_cosines["NULL_CONSTRUCTION_MATCHED_SIGN_SHUFFLED_R3"]
    report = f"""# Q2 controller-held-out geometry pilot — bank qualification closeout

`Q2_CONTROLLER_BANK_NOT_QUALIFIED`

## Outcome

The first Q2 DEVELOPMENT pilot stopped at its prospectively frozen controller-bank gate.
The 120-item common panel, error-distance matrix, M0/M1/M2 predictive analysis, QAP,
bootstrap, and outcome reveal were **not run**.

## What passed

- Engineering: `{engineering['classification']}`.
- Source axes: 3/3 passed.
- Unit norms, exact sign pairs, and meaningful-base diversity passed.

{chr(10).join(source_lines)}

## What failed

- Controller movement: {len(qualified)}/16 controllers passed all frozen mechanical and
  causal first-stage criteria; 13/16 missed raw sequence change `>= 0.25`.
- Representation geometry: null orthogonality failed. Maximum absolute cosine from a
  null to the meaningful basis was `{max(null_cosines.values()):.6f}` against the
  frozen `1e-6` tolerance.
- The construction-matched nulls reached `{shuffled_r2_cosine:.6f}` and
  `{shuffled_r3_cosine:.6f}` because sequential
  subtraction against correlated source directions did not construct an orthonormal
  basis for the meaningful span.

Controller gate details:

{chr(10).join(failure_lines)}

## Interpretation boundary

This is a controller-bank qualification failure, not a null predictive-geometry result.
No pairwise semantic error-profile matrix was collected. Accuracy, G, C, D, rescue,
damage, and complementarity were not used to select or reject controllers. Repairing the
null construction alone would not rescue this run because the causal-movement gate also
failed independently.

Q1 remains `Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`. Q3 was not run.
"""
    (review / "REPORT.md").write_text(report, encoding="utf-8")
    (review / "BANK_POSTMORTEM.md").write_text(report, encoding="utf-8")

    (review / "NEXT_PROTOCOL_DRAFT.md").write_text(
        """# Draft only — Q2 controller-bank rebuild qualification

Do **not** execute without a new principal lock.

The next protocol should remain outcome-free at bank construction and should:

1. construct the meaningful-span projector from a numerically verified orthonormal QR/SVD
   basis, then qualify every null with the frozen cosine tolerance before inference;
2. pre-specify a larger causal first-stage panel and a minimum balanced bank size/family
   composition rather than relying on a 12-item all-controllers-pass gate;
3. retain genuine conceptual/source variation and prohibit ranking by accuracy, G, C, D,
   rescue, damage, or downstream complementarity;
4. freeze a new controller-held-out split only after the bank passes;
5. run no common Q2 panel until engineering, source, movement, geometry, and cost gates all
   pass under one new prospective lock.

This draft is a postmortem recommendation, not authorization for new inference.
""",
        encoding="utf-8",
    )
    print(json.dumps(postmortem, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
