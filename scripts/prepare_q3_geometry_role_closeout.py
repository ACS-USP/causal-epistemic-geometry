#!/usr/bin/env python3
"""Prepare release-safe Q3.2 closeout artifacts from the private full result."""

# ruff: noqa: E501 -- Markdown table rows are intentionally kept as source rows.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_geometry_role_decomposition"
PRECHECK = REVIEW / "Q3_GEOMETRY_ROLE_DECOMPOSITION_PRECHECK.json"
AMENDMENT = REVIEW / "Q3_GEOMETRY_ROLE_EXECUTION_AMENDMENT.json"
SUMMARY = REVIEW / "Q3_GEOMETRY_ROLE_DECOMPOSITION_RELEASE_SUMMARY.json"
SAFETY = REVIEW / "Q3_GEOMETRY_ROLE_RELEASE_SAFETY.json"
HASHES = REVIEW / "Q3_GEOMETRY_ROLE_ARTIFACT_HASHES.json"
REPORT = ROOT / "docs/Q3_GEOMETRY_ROLE_DECOMPOSITION_REVIEW.md"
ROADMAP = ROOT / "docs/Q3_FRESH_EVALUATION_INSTRUMENT_ROADMAP.md"
EXPECTED_RESULT_SHA256 = "d1913c4e2b4f500ecece62da83598f5ca157455a788f6b7fc3a45f166c86c71e"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def release_summary(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    part_a = result["part_a"]
    part_b = result["part_b"]
    return {
        "schema_version": "q3-geometry-role-decomposition-release-summary-v1",
        "status": result["status"],
        "scientific_state": "Q3_NOT_RUN_DEVELOPMENT_ONLY",
        "evidence_class": result["evidence_class"],
        "immutable_q3_1": result["immutable_q3_1"],
        "sources": {
            "private_full_result_sha256": sha256_file(result_path),
            "private_full_result_tracked": False,
            "precheck_sha256": sha256_file(PRECHECK),
            "execution_amendment_sha256": sha256_file(AMENDMENT),
        },
        "part_a": {
            "question": "geometry as policy-bank construction",
            "fixed_banks": part_a["fixed_banks"],
            "distributions": {
                name: {key: value for key, value in distribution.items() if key != "rows"}
                for name, distribution in part_a["distributions"].items()
            },
            "attribution": part_a["attribution"],
            "ruling": part_a["ruling"],
        },
        "part_b": {
            "question": "historical-to-fresh controller routing transfer",
            "fixed_controller_split": part_b["fixed_controller_split"],
            "models": part_b["models"],
            "true_geometry_kernel_prior": part_b["true_geometry_kernel_prior"],
            "historical_global_shell_prior": part_b["historical_global_shell_prior"],
            "attribution": part_b["attribution"],
            "ruling": part_b["ruling"],
        },
        "surviving_claim": (
            "On the closed 300-family development panel, A0-maximin bank construction "
            "supported more realizable prompt-representation routing utility than the "
            "prospectively frozen competence-matched random-bank distribution; true "
            "coordinates did not support useful routing transfer to held-out fresh "
            "controller identities."
        ),
        "future_instrument": {
            "minimum_family_count": 800,
            "items_generated": 0,
            "holdout_allocated": False,
            "future_outcomes_inspected": False,
        },
        "firewall": result["firewall"],
    }


def report_markdown(summary: dict[str, Any]) -> str:
    a = summary["part_a"]
    b = summary["part_b"]
    a0 = a["fixed_banks"]["A0"]
    matched = a["distributions"]["COMPETENCE_MATCHED_RANDOM"]
    attr = a["attribution"]
    true = b["models"]["TRUE"]
    controls = b["models"]
    differences = b["attribution"]["differences"]
    return f"""# Q3.2 Geometry-Role Decomposition Review

## 1. Immutable Q3.1 result

Q3.1 remains permanently
`Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL`.
It showed stable prompt-representation policy selectability (+0.0533 over its
cross-fitted champion; 5/5 positive folds), while true A0 coordinates added
only +0.0033 over learned policy identity, below the frozen +0.01 criterion.
Q3 remains `NOT_RUN`; Q3.2 is closed-data DEVELOPMENT only.

## 2. Prospective development precheck

The precheck was frozen and pushed before outcomes were opened. Its SHA-256 is
`{summary["sources"]["precheck_sha256"]}`. A pre-result implementation-only
amendment cached the identical fold PCA and changed no scientific object; its
SHA-256 is `{summary["sources"]["execution_amendment_sha256"]}`. The full
private result was reproduced byte-for-byte twice and has SHA-256
`{summary["sources"]["private_full_result_sha256"]}`.

## 3. A0-maximin bank performance

With the Q3.1 geometry-blind learned-policy-identity router, A0-maximin K=8
achieved routed accuracy {fmt(a0["routed_accuracy"])}, versus its own
outer-training-selected champion at {fmt(a0["own_champion_accuracy"])}: gain
{a0["routed_gain"]:+.4f}. Oracle headroom was {fmt(a0["oracle_headroom"])}, of
which {fmt(a0["oracle_fraction_realized"])} was realized. Fold gains were
`{", ".join(f"{value:+.4f}" for value in a0["fold_gains"])}`; 4/5 were positive
and the worst was {a0["worst_fold_gain"]:+.4f}. Validity and evaluability were
both {fmt(a0["commitment_validity"])}.

## 4. Matched random-bank distribution

There were {matched["replicates"]} prospectively frozen competence-matched
random-bank procedures. Their routed-gain q2.5/median/q95/q97.5 values were
`{", ".join(fmt(value) for value in matched["routed_gain_quantiles"])}`. A0's
gain percentile was {fmt(attr["a0_gain_percentile_matched_random"])}; its
plus-one upper-tail diagnostic p was {fmt(attr["gain_randomization_p"], 6)}.
A0 exceeded the matched median gain by {attr["a0_minus_matched_median_gain"]:+.4f}.
These banks share items/controllers and are a paired development diagnostic,
not IID scientific replications.

## 5. Low-diversity and alternative-geometry banks

| Bank/design | Routed accuracy | Gain | Headroom | Positive folds | Worst fold |
|---|---:|---:|---:|---:|---:|
| A0-maximin | {a0["routed_accuracy"]:.4f} | {a0["routed_gain"]:+.4f} | {a0["oracle_headroom"]:.4f} | {a0["positive_fold_count"]}/5 | {a0["worst_fold_gain"]:+.4f} |
| A1-maximin | {a["fixed_banks"]["A1"]["routed_accuracy"]:.4f} | {a["fixed_banks"]["A1"]["routed_gain"]:+.4f} | {a["fixed_banks"]["A1"]["oracle_headroom"]:.4f} | {a["fixed_banks"]["A1"]["positive_fold_count"]}/5 | {a["fixed_banks"]["A1"]["worst_fold_gain"]:+.4f} |
| A2-maximin | {a["fixed_banks"]["A2"]["routed_accuracy"]:.4f} | {a["fixed_banks"]["A2"]["routed_gain"]:+.4f} | {a["fixed_banks"]["A2"]["oracle_headroom"]:.4f} | {a["fixed_banks"]["A2"]["positive_fold_count"]}/5 | {a["fixed_banks"]["A2"]["worst_fold_gain"]:+.4f} |

The low-A0-diversity distribution had median gain
{a["distributions"]["LOW_A0_DIVERSITY"]["routed_gain_quantiles"][1]:.4f}; the
unmatched deterministic-random distribution had median gain
{a["distributions"]["DETERMINISTIC_RANDOM"]["routed_gain_quantiles"][1]:.4f}.
The outcome-optimized bank remains an oracle upper bound, not a deployable bank.

## 6. Geometry bank-selection attribution

A0 passed every frozen gate: ≥0.03 realization gain, ≥95th-percentile gain and
headroom, both plus-one p-values ≤.05, ≥0.01 above matched median gain, and
nonnegative fold contrast in at least 4/5 folds (observed 5/5). Part A is:

`{a["ruling"]}`

This supports geometry's role in constructing the portfolio on closed data; it
does not show that coordinates improve routing within a fixed bank.

## 7. Historical-to-fresh controller transfer design

The model was trained on the fixed 31 historical controllers and evaluated on
the fixed 16 fresh OOS controllers, with simultaneous five-fold item-family
cross-fitting. PCA, scaling, hyperparameters and all model fitting used
historical-controller outer-training data only. The primary descriptor was
`[amplitude × unit rank-8 coordinates, amplitude]`; MEDIUM=0.25 and STRONG=0.50.
No policy-identity embedding or fresh-controller outcome entered fitting.

## 8. True-coordinate controller-OOS results

True coordinates achieved routing accuracy {true["routed_accuracy"]:.4f},
against uniform fresh-policy accuracy {true["random_policy_accuracy"]:.4f}, for
gain {true["routing_gain"]:+.4f}. Fold gains were
`{", ".join(f"{value:+.4f}" for value in true["fold_gains"])}`: 3/5 positive,
with worst fold {true["worst_fold_gain"]:+.4f}. Log loss was
{true["predictive"]["log_loss"]:.4f}, Brier {true["predictive"]["brier"]:.4f},
and mean itemwise policy-ranking Spearman {true["mean_item_policy_rank_correlation"]:.4f}.

## 9. Permuted/random/agnostic controls

| Representation | Routing gain | Positive folds | Worst fold | Log loss |
|---|---:|---:|---:|---:|
| True coordinates | {true["routing_gain"]:+.4f} | {true["positive_fold_count"]}/5 | {true["worst_fold_gain"]:+.4f} | {true["predictive"]["log_loss"]:.4f} |
| Permuted coordinates | {controls["PERMUTED"]["routing_gain"]:+.4f} | {controls["PERMUTED"]["positive_fold_count"]}/5 | {controls["PERMUTED"]["worst_fold_gain"]:+.4f} | {controls["PERMUTED"]["predictive"]["log_loss"]:.4f} |
| Random coordinates | {controls["RANDOM"]["routing_gain"]:+.4f} | {controls["RANDOM"]["positive_fold_count"]}/5 | {controls["RANDOM"]["worst_fold_gain"]:+.4f} | {controls["RANDOM"]["predictive"]["log_loss"]:.4f} |
| Controller-agnostic | {controls["AGNOSTIC"]["routing_gain"]:+.4f} | {controls["AGNOSTIC"]["positive_fold_count"]}/5 | {controls["AGNOSTIC"]["worst_fold_gain"]:+.4f} | {controls["AGNOSTIC"]["predictive"]["log_loss"]:.4f} |

True-minus-control routing-gain differences were
{differences["PERMUTED"]["routing_gain_difference"]:+.4f} (permuted),
{differences["RANDOM"]["routing_gain_difference"]:+.4f} (random), and
{differences["AGNOSTIC"]["routing_gain_difference"]:+.4f} (agnostic). The first
two crossed +0.01; the agnostic contrast did not. True coordinates improved log
loss over every control, but this predictive advantage did not satisfy the
routing-utility gates.

## 10. Controller-OOS routing utility

The true model failed the ≥0.03 realization gate, the ≥4/5 positive-fold gate,
and the −0.02 worst-fold gate. It also failed routing and fold-consistency
attribution against the agnostic control. The descriptor-only A0 kernel prior
gained {b["true_geometry_kernel_prior"]["routing_gain"]:+.4f}; the historical
global shell prior gained {b["historical_global_shell_prior"]["routing_gain"]:+.4f}.
Part B is:

`{b["ruling"]}`

## 11. Geometry-role ruling

Part A supports geometry for bank construction. Part B does not support useful
controller-OOS routing transfer. Q3.1 already did not support incremental
geometry for fixed-bank routing. The high-level ruling is:

`{summary["status"]}`

## 12. Q3 narrative implication

The surviving development claim is narrow: A0 geometry may design a
complementary portfolio, while prompt representations plus learned policy
identity perform routing for known policies. True coordinates did not support
deployment to unseen controller identities. This is not realized Q3 utility,
and Q3 remains `NOT_RUN`.

## 13. Fresh-evaluation instrument roadmap

No future item was generated, selected or scored. The minimum proposed supply
is 800 family-independent units. The roadmap compares: newly authored
CRUXEval-like executable traces; a family-disjoint public exact-evaluator
benchmark; and a separately generated deterministic program-execution
benchmark. See [the instrument roadmap](Q3_FRESH_EVALUATION_INSTRUMENT_ROADMAP.md).

## 14. Reviewer/fragility audit

- Closed-data development analysis cannot establish prospective utility.
- Part-A banks reuse controllers/items and are not IID bank replications.
- Competence matching is finite-pool and cannot remove all bank-composition
  confounding.
- The 31→16 split is provenance-defined, but the fresh population is
  safety-conditioned and from the same Qwen/CRUXEval/rank-8 laboratory.
- Part B has only 16 held-out controller identities and five item folds.
- True-coordinate prediction improved log loss, but routing benefit was small
  and fold-unstable; this must not be narrated as successful transfer.
- The result does not address cross-model/task generalization or a fresh Q3
  holdout.

## 15. Repository/resource state

- New semantic trajectories: **0**.
- New Qwen forwards: **0**.
- Closed historical/fresh controller outcomes used: **YES, development only**.
- Future fresh-evaluation outcomes inspected: **NO**.
- A0 bank percentile among matched random banks: **{attr["a0_gain_percentile_matched_random"]:.6f}**.
- Part-A ruling: **{a["ruling"]}**.
- True-coordinate controller-OOS gain: **{true["routing_gain"]:+.6f}**.
- Part-B ruling: **{b["ruling"]}**.
- Minimum proposed fresh-family count: **800**.
- Q1/Q2/Q3.1 classifications changed: **NO**.
- Q3 confirmatory experiment: **NOT_RUN**.
- Spark 1 GPU used: **NO**.
- Spark 2 used: **NO**.
- RunPod used: **NO**.
- Personal handbook/paper workspace modified: **NO**.

`{summary["status"]}`
"""


def roadmap_markdown() -> str:
    return """# Q3 Fresh-Evaluation Instrument Roadmap

This is a design-only supply roadmap following Q3.2. It does not allocate a
holdout, select a source, generate items, inspect future correctness or
authorize Q3. The minimum target remains **800 family-independent evaluation
units**, pending a separate prospectively justified power review.

| Route | Task match | Exact evaluator | Family unit | Main risk | Licensing/reviewer posture |
|---|---|---|---|---|---|
| New executable CRUXEval-like traces | Closest to the closed program-execution laboratory | Sandboxed deterministic execution with typed exact output | One independently generated source program/problem | Template leakage and synthetic-family dependence | Clean only if generator, code license and release rights are frozen before creation |
| Family-disjoint public benchmark | Potentially strong if output prediction and family IDs are native | Official deterministic scorer only | Official problem/source-program family | Contamination, hidden siblings and evaluator drift | Strongest external credibility if redistribution and exact instrument are unambiguous |
| Separately generated deterministic benchmark | Tunable task and difficulty match | Frozen generator plus executable reference oracle | One generator-independent program family | Generator artifacts may dominate semantics | Credible only with diverse templates, held-out generators and public auditability |

## Required qualification sequence

1. Freeze task definition, family ontology, generator/source provenance,
   deduplication, license and evaluator before model contact.
2. Construct a development pool separate from the permanent evaluation pool.
3. Qualify commitment validity, semantic evaluability, difficulty, family
   independence and policy opportunity on development families only.
4. Run an outcome-independent power review for the surviving Q3 claim.
5. Allocate and hash at least 800 evaluation families only after all gates pass.
6. Keep policy-bank construction and router training outside the evaluation
   families; open the holdout exactly once under a new confirmatory lock.

## Runtime envelope

At the observed Q2 OOS rate (19,200 trajectories in 9.42 h), a minimal
8-policy × 2-rollout evaluation over 800 families would contain 12,800 semantic
trajectories and scale to roughly 6.3 Spark-1 hours before qualification,
prompt capture, retries or safety overhead. This is an operational extrapolation,
not a runtime lock.

## Decision still required

No source should be chosen merely for convenience. A future principal review
must select the instrument only after model-free provenance, licensing,
family-independence, exact-evaluator and contamination audits. No candidate
future correctness has been inspected here.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-result", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.full_result) != EXPECTED_RESULT_SHA256:
        raise RuntimeError("Q3.2 private full-result hash mismatch")
    result = read_json(args.full_result)
    if result["status"] != "Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING":
        raise RuntimeError("unexpected Q3.2 mechanical ruling")
    summary = release_summary(result, args.full_result)
    write_json(SUMMARY, summary)
    REPORT.write_text(report_markdown(summary), encoding="utf-8")
    ROADMAP.write_text(roadmap_markdown(), encoding="utf-8")
    safety = {
        "status": "Q3_GEOMETRY_ROLE_RELEASE_SAFETY_PASS",
        "raw_benchmark_text_included": False,
        "raw_model_outputs_included": False,
        "prompt_representation_values_included": False,
        "private_itemwise_outcomes_included": False,
        "credentials_or_infrastructure_included": False,
        "new_semantic_outcomes": 0,
        "new_qwen_forwards": 0,
        "private_full_result_sha256": EXPECTED_RESULT_SHA256,
        "private_full_result_tracked": False,
    }
    write_json(SAFETY, safety)
    artifacts = [
        PRECHECK,
        AMENDMENT,
        SUMMARY,
        SAFETY,
        REPORT,
        ROADMAP,
        ROOT / "scripts/analyze_q3_geometry_role_decomposition.py",
        ROOT / "scripts/prepare_q3_geometry_role_precheck.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_q3_geometry_role_decomposition.py",
    ]
    write_json(
        HASHES,
        {
            "schema_version": "q3-geometry-role-artifact-hashes-v1",
            "classification": result["status"],
            "artifacts": {str(path.relative_to(ROOT)): sha256_file(path) for path in artifacts},
            "private_hash_pinned": {
                "Q3_GEOMETRY_ROLE_DECOMPOSITION_FULL_RESULT.json": (EXPECTED_RESULT_SHA256)
            },
            "raw_text_included": False,
            "q3": "NOT_RUN",
        },
    )
    print(sha256_file(SUMMARY))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
