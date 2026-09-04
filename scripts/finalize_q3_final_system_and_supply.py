#!/usr/bin/env python3
# ruff: noqa: E501
"""Finalize release-safe Q3.3 planning artifacts and the unified review."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_final_system_and_evaluation_supply"
DOC = ROOT / "docs/Q3_FINAL_SYSTEM_AND_EVALUATION_SUPPLY_REVIEW.md"


def read_json(name: str) -> Any:
    return json.loads((REVIEW / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(name: str, value: Any) -> None:
    (REVIEW / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def power_row(rows: list[dict[str, str]], n: int, r: int, gain: float) -> dict[str, float]:
    row = next(
        value
        for value in rows
        if value["scenario"] == "CONSERVATIVE_COMBINED"
        and value["seed_regime"] == "INDEPENDENT"
        and int(value["N"]) == n
        and int(value["R"]) == r
        and float(value["gain"]) == gain
    )
    return {
        "power": float(row["t_power"]),
        "mean_half_width": float(row["mean_half_width"]),
        "utility_trajectories_max": int(row["utility_trajectories_max"]),
        "p50_hours": float(row["frozen_p50_hours_utility"]),
        "p80_hours": float(row["frozen_p80_hours_utility"]),
        "p95_hours": float(row["frozen_p95_hours_utility"]),
        "expected_tokens": float(row["expected_generated_tokens_utility"]),
        "expected_storage_bytes": float(row["expected_storage_bytes_utility"]),
    }


def scaled_runtime(trajectories: int) -> dict[str, float | int]:
    return {
        "trajectories": trajectories,
        "expected_generated_tokens": trajectories * 381630 / 19200,
        "expected_storage_bytes": trajectories * 127968010 / 19200,
        "observed_rate_hours": trajectories * 33924.293892600996 / 19200 / 3600,
        "p50_hours": 9.76 * trajectories / 19200,
        "p80_hours": 11.05 * trajectories / 19200,
        "p95_hours": 12.45 * trajectories / 19200,
    }


def main() -> int:
    system = read_json("FINAL_Q3_CANDIDATE_SYSTEM_DRAFT.json")
    tier = read_json("Q3_TIER_B_EXPOSURE_SEVERITY_AUDIT.json")
    power_summary = read_json("Q3_UTILITY_POWER_PRECISION_SUMMARY.json")
    source_audit = read_json("Q3_FRESH_INSTRUMENT_SOURCE_AUDIT.json")
    instrument = read_json("Q3_FRESH_INSTRUMENT_DESIGN_DRAFT.json")
    with (REVIEW / "Q3_UTILITY_POWER_PRECISION.csv").open() as handle:
        power_rows = list(csv.DictReader(handle))

    selected = power_row(power_rows, 1000, 2, 0.03)
    comparisons = {
        "N800_R2": power_row(power_rows, 800, 2, 0.03),
        "N800_R4": power_row(power_rows, 800, 4, 0.03),
        "N1000_R2_SELECTED": selected,
        "N500_R4": power_row(power_rows, 500, 4, 0.03),
        "N1200_R2": power_row(power_rows, 1200, 2, 0.03),
    }
    runtime = {
        "schema_version": "q3-runtime-accounting-v1",
        "source": "linear scaling of sealed Q2 OOS V2 19,200-row runtime/tokens/storage and frozen tail forecast",
        "utility_only_selected_N1000_R2": scaled_runtime(1000 * 2 * 2),
        "full_bank_diagnostic_N1000_R2": scaled_runtime(1000 * 2 * 8),
        "qualification_full_bank_plus_champion_plus_baseline_N300_R2": scaled_runtime(300 * 2 * 10),
        "portfolio_attribution_N1000_R2": {
            str(s): scaled_runtime(1000 * 2 * (s + 2)) for s in (8, 12, 16, 19, 20)
        },
        "interpretation": {
            "deployment_calls": 1,
            "champion_generation": "experimental comparator only",
            "full_bank_required_for_primary_utility": False,
            "portfolio_banks_are_deployed_one-call_systems": True,
            "research_cost_is_not_deployment_cost": True,
        },
    }
    write_json("Q3_RUNTIME_ACCOUNTING.json", runtime)

    decision = {
        "schema_version": "q3-evaluation-supply-decision-v1",
        "status": "Q3_FRESH_INSTRUMENT_DESIGN_READY_FOR_PRELOCK",
        "development_phase_closed": True,
        "final_candidate_system": {
            "status": system["status"],
            "draft_sha256": sha256_file(REVIEW / "FINAL_Q3_CANDIDATE_SYSTEM_DRAFT.json"),
            "bank": [row["policy_id"] for row in system["portfolio"]["policies"]],
            "champion": system["champion"]["policy_id"],
        },
        "claim_states": {
            "U": "DEVELOPMENT_SUPPORTED_NOT_CONFIRMATORY",
            "P": "DEVELOPMENT_SUPPORTED_NOT_CONFIRMATORY",
            "G": "NOT_SUPPORTED",
            "C_OOS": "NOT_SUPPORTED",
        },
        "paper_strategy": {
            "selected": "A_CONFIRM_REALIZED_UTILITY_ONLY",
            "reason": "Claim U is the direct one-call utility claim; Claim P already has development support but a valid multi-bank confirmation would cost much more and is not necessary for U.",
            "portfolio_geometry": "retain as development evidence; do not make co-primary",
        },
        "tier_b": {
            "counts": tier["stratum_counts"],
            "confirmatory_eligible": tier["eligibility"]["confirmatory_families"],
            "internal_validation_eligible": tier["eligibility"][
                "bounded_internal_validation_families"
            ],
            "route_I": "REJECT_NUMERICALLY_AND_PROVENANCE_INADEQUATE",
        },
        "routes": {
            "I_TIER_B_ONLY": "REJECT",
            "II_FULLY_FRESH": "SELECT",
            "III_TIER_B_THEN_FRESH": "REJECT; 11 eligible families cannot meaningfully de-risk the frozen system and repeated opening would add adaptation pressure",
        },
        "fresh_supply": {
            "instrument": source_audit["recommended_route"],
            "allocation": instrument["allocation"],
            "selected_confirmatory_design": {"N": 1000, "R": 2, **selected},
            "why_not_N800_R2": "conservative +0.03 power is below 0.80",
            "minimum_proposed_fresh_family_count": 1600,
            "final_ids_generated": 0,
            "final_seeds_generated": 0,
            "holdout_allocated": 0,
        },
        "primary_inference": {
            "method": power_summary["selected_primary"],
            "confidence_interval": power_summary["selected_ci"],
            "calibration_regime": power_summary["selected_regime"],
            "max_fpr": power_summary["selected_regime_t_max_fpr"],
            "minimum_coverage": power_summary["selected_regime_t_min_coverage"],
            "matched_seeds_for_distinct_policies": "REJECTED",
            "same_exact_policy_generation_reuse": "ALLOWED_AND_DIFFERENCE_ZERO",
        },
        "future_action": "prepare a separate generator/evaluator/qualification prelock; do not generate final items under Q3.3",
        "q3": "NOT_RUN",
        "firewall": {
            "new_semantic_trajectories": 0,
            "new_qwen_forwards": 0,
            "fresh_evaluation_outcomes_inspected": False,
            "spark1_gpu": False,
            "spark2": False,
        },
    }
    write_json("Q3_EVALUATION_SUPPLY_DECISION.json", decision)

    q32 = json.loads(
        (
            ROOT
            / "review/q3_geometry_role_decomposition/Q3_GEOMETRY_ROLE_DECOMPOSITION_RELEASE_SUMMARY.json"
        ).read_text()
    )
    red_team = [
        (
            "Router overfit to 300 families",
            "Development gain may not transfer",
            "Evaluate the frozen system on the untouched fresh confirmation split",
            "Any router/bank adaptation after opening qualification or evaluation",
        ),
        (
            "A0-bank selection overfit",
            "One deterministic bank was selected using closed development data",
            "Keep P development-only or run a separately powered multi-bank design",
            "Calling one bank a population-level geometry attribution",
        ),
        (
            "Synthetic artifacts",
            "A generator may create shortcuts",
            "Public generator, construct quotas, human-blind structural audit, and public-benchmark bridge",
            "Qualification reveals template shortcuts or narrow skeleton dominance",
        ),
        (
            "Benchmark contamination",
            "Public tasks may be in pretraining",
            "Generate all final source/input pairs after prelock",
            "Any evidence final items existed before freeze",
        ),
        (
            "Pseudo-replication/family leakage",
            "Inputs or templates may masquerade as independent families",
            "One source skeleton per family and zero cross-split canonical overlap",
            "Duplicate skeleton or shared source across splits",
        ),
        (
            "Champion leakage",
            "Comparator could be selected on confirmation",
            "Freeze V4_DIRECTION_02_MEDIUM now",
            "Any confirmatory-outcome-dependent comparator change",
        ),
        (
            "Compute unfairness",
            "Router and champion may receive unequal calls",
            "Same generation contract and one answer each; share only identical-policy calls",
            "Unequal generation budgets or adaptive retries",
        ),
        (
            "Matched-seed validity",
            "Different policies consume randomness differently",
            "Use independent frozen seeds for distinct policies",
            "Power claim relies on unverified common-random-number coupling",
        ),
        (
            "Random-bank population",
            "Shared policies/families make banks dependent",
            "Separate bank-level prelock with prospectively sampled constructions",
            "Treating controller pairs or banks as IID",
        ),
        (
            "Repeated Tier-B testing",
            "Historical exposure invites adaptation",
            "Skip Tier B",
            "Any Tier-B outcome is opened before the fresh system is permanently frozen",
        ),
        (
            "Distribution shift",
            "Fresh generated tasks may differ from CRUXEval",
            "Qualification reports difficulty, validity, constructs, and a public descriptive bridge",
            "Qualification task is too easy/hard or answer contract fails",
        ),
        (
            "Practical +3 pp",
            "Statistical positivity may be too small to matter",
            "Report CI, +3 pp benchmark, latency, tokens, and failure modes separately",
            "Narrative claims practical benefit when point estimate is below +3 pp",
        ),
        (
            "Same-forward one-call",
            "Implementation might secretly require a second prefill",
            "Synthetic engine test traces one prefill and one decode",
            "Any second model forward solely for routing",
        ),
        (
            "Paper scope",
            "Q3 may dilute the closed Q1/Q2 claim",
            "Treat Q3 confirmation as an extension unless fresh evaluation is clean and decisive",
            "Pressing Q3 into the current paper before fresh confirmation",
        ),
    ]
    summary = {
        "schema_version": "q3-final-system-evaluation-supply-release-summary-v1",
        "status": decision["status"],
        "immutable_q3_2": {
            "classification": q32["status"],
            "a0_bank_percentile": q32["part_a"]["attribution"]["a0_gain_percentile_matched_random"],
            "a0_routed_gain": q32["part_a"]["fixed_banks"]["A0"]["routed_gain"],
            "controller_oos_true_gain": q32["part_b"]["models"]["TRUE"]["routing_gain"],
            "part_a": q32["part_a"]["ruling"],
            "part_b": q32["part_b"]["ruling"],
        },
        "final_system": decision["final_candidate_system"],
        "tier_b": decision["tier_b"],
        "power_comparisons": comparisons,
        "selected_route": decision["routes"],
        "fresh_instrument": decision["fresh_supply"],
        "claims": decision["claim_states"],
        "reviewer_red_team": [
            {"attack": a, "fragility": f, "cheapest_check": c, "stop_condition": s}
            for a, f, c, s in red_team
        ],
        "firewall": decision["firewall"],
    }
    write_json("Q3_FINAL_SYSTEM_AND_SUPPLY_RELEASE_SUMMARY.json", summary)

    bank_lines = "\n".join(
        f"{row['order'] + 1}. `{row['policy_id']}` — vector `{row['vector_sha256']}`"
        for row in system["portfolio"]["policies"]
    )
    rt = runtime["utility_only_selected_N1000_R2"]
    report = f"""# Q3.3 Final-System Freeze and Evaluation-Supply Review

## 1. Immutable Q3.2 result

`Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING` remains unchanged. Part A was `GEOMETRY_BANK_SELECTION_SUPPORTED`: the A0-maximin routed gain was +0.0400, at percentile {q32["part_a"]["attribution"]["a0_gain_percentile_matched_random"]:.6f} among 512 competence-matched banks. Part B was `CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED`: true-coordinate gain was {q32["part_b"]["models"]["TRUE"]["routing_gain"]:+.6f}, with 3/5 positive folds. These are closed-data DEVELOPMENT results; Q3 remains `NOT_RUN`.

The Q3.2 “outcome-optimized upper bound” bounds bank opportunity/construction only. It is not an upper bound on routed accuracy, and no historical result is changed by this editorial clarification.

## 2. Prospective development closure

The base precheck and additive steer were frozen and pushed before Q3.3 analyses. `DEVELOPMENT_PHASE_CLOSED = YES`: no further architecture, hyperparameter, or open-ended bank tournament is permitted on the 300-family panel.

## 3. Final candidate deployment system

Status: `DEVELOPMENT_SELECTED_NOT_EVALUATED`.

{bank_lines}

The frozen champion is `{system["champion"]["policy_id"]}`. The router uses the ordinary unsteered layer-27 block-input representation at the final non-padding prompt token, PCA dimension 8, a rank-2 learned-policy-identity interaction, L2=1, 400 deterministic full-batch Adam steps at learning rate 0.03, and seed 2026090511. Those values are the componentwise fold consensus (8/2/1), not the best apparent full-panel fit. Controller coordinates constructed the portfolio but are not router inputs.

Deployment is one-call: the unsteered prefill reaches L27, the frozen router chooses one policy, and sustained-current-token steering continues from the same prefill into one answer decode. The private fitted parameters are hash-pinned as `{system["router"]["private_parameter_manifest"]["sha256"]}` and are not stored in Git.

## 4. Tier-B exposure-severity audit

Of 500 candidate families: A={tier["stratum_counts"]["A"]}, B={tier["stratum_counts"]["B"]}, C={tier["stratum_counts"]["C"]}, D={tier["stratum_counts"]["D"]}, E={tier["stratum_counts"]["E"]}, F={tier["stratum_counts"]["F"]}. Thus zero are confirmatory-eligible and only 11 are eligible for bounded internal validation. No Tier-B correctness was opened, no IDs were allocated, and the public audit contains only counts and set hashes. Route I is not scientifically or numerically viable.

## 5. Power and precision

The model-free tournament used 20,000 null panels and 10,000 alternative panels per cell over N={{23,100,250,400,500,800,1000,1200}}, R={{1,2,4,6,8}}, gains +1/+2/+3/+4/+5 pp, variable discordance, family difficulty/effect heterogeneity, rare harm, and independent versus common-uniform seeds.

For the selected independent-seed N≥800 regime, the paired family-level studentized t test had maximum FPR {power_summary["selected_regime_t_max_fpr"]:.5f} and minimum 95% interval coverage {power_summary["selected_regime_t_min_coverage"]:.3f}. Common seeds are rejected for distinct autoregressive policies. At +3 pp under the conservative combined scenario: N=800/R=2 power={comparisons["N800_R2"]["power"]:.3f}; N=800/R=4 power={comparisons["N800_R4"]["power"]:.3f}; N=1000/R=2 power={selected["power"]:.3f}. N=1000/R=2 is preferred because it clears 0.80 while using more independent families and only 4,000 maximum trajectories.

## 6. Frozen primary inference

The future primary is Claim U: `Delta_utility`, the family-weighted mean of rollout-mean correctness(router-selected policy) minus correctness(frozen champion), with invalid/unevaluable outputs incorrect and missing rows blocking completion. The test is one-sided paired family-level studentized t; the interval is the two-sided 95% family-level t interval. Family is the independent unit and rollout is nested replication.

If router and champion select the exact same policy, one frozen-seed generation may back both experimental roles and contributes paired difference zero. Distinct policies use independent seeds. This preserves the estimand without pretending divergent autoregressive paths are common-random-number coupled.

## 7. Utility, full-bank, and portfolio-attribution designs

- Design U (primary): N×R×2 maximum trajectories; one deployed answer plus an experimental champion comparator.
- Design F (diagnostic): N×R×8 trajectories; needed only for oracle/headroom and full counterfactual policy matrices.
- Design P (optional): compare the learned bank with S random-bank systems. Banks share families and may share policies, so they are not IID controller/dyad observations.

For N=1000/R=2, U is 4,000 trajectories (P50/P80/P95 {rt["p50_hours"]:.2f}/{rt["p80_hours"]:.2f}/{rt["p95_hours"]:.2f} Spark-1 hours, about {rt["expected_generated_tokens"]:.0f} generated tokens and {rt["expected_storage_bytes"] / 1e6:.1f} MB). F is 16,000 trajectories, not a requirement for Claim U. P with S=20 is at most 44,000 trajectories and remains a separate research cost, never a 20-call deployment claim.

## 8. Claim precedence and paper strategy

- U: `DEVELOPMENT_SUPPORTED_NOT_CONFIRMATORY` — future primary.
- P: `DEVELOPMENT_SUPPORTED_NOT_CONFIRMATORY` — retain as development evidence.
- G: `NOT_SUPPORTED`.
- C-OOS: `NOT_SUPPORTED`.

Recommended paper strategy A: confirm realized one-call utility and keep portfolio geometry as development evidence. Strategy B would require a separately calibrated and much larger multi-bank campaign; it is not needed to establish U and should not be co-primary by default.

## 9. Fresh-instrument source audit

The official [CRUXEval repository](https://github.com/facebookresearch/cruxeval) documents 800 short Python function/input/output samples, execution-derived references, output prediction, and an MIT license. It is the task anchor but its current 800 families have been exhausted by prior project exposure. [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) officially supports test-output prediction, but the project’s pinned instrument has only 182 independent families and is already a closed negative Q1 development path.

[BigCodeBench](https://github.com/bigcode-project/bigcodebench) supplies 1,140 Apache-2.0 executable code-generation tasks; [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) has a 500-problem official test split; and [HumanEval](https://github.com/openai/human-eval) has an MIT executable harness. All change the response contract to code generation and carry substantial public-training exposure. [Project CodeNet](https://github.com/IBM/Project_CodeNet) and [CodeContests](https://github.com/google-deepmind/code_contests) offer thousands of natural programs/problems, but input-domain reconstruction, mixed third-party provenance, source exposure, and family definition make the hybrid route less clean.

## 10. Recommended supply route

Route II is selected: a genuinely fresh, separately generated deterministic program-execution instrument. Route III is rejected because 11 eligible Tier-B families cannot de-risk the system and opening them would add adaptation pressure. Proposed supply is 1,600 independent families: 300 qualification, 1,000 confirmation, 300 untouched reserve. No IDs, seeds, items, or permanent split allocation were generated in Q3.3.

## 11. Generator and evaluator design

Use a model-free typed restricted-Python AST grammar. Every family has an independently generated program skeleton and one allocated input; another input to the same program is nested, not a new family. Allow bounded mutation/aliasing, branching, loops, nested control flow, containers, pure helpers, and depth-bounded recursion. Forbid filesystem, network, imports, reflection, dynamic code, randomness, ambient state, and unbounded computation.

Reference outputs require exact agreement between a restricted-AST interpreter and sandboxed pinned CPython, repeated twice under deterministic locale/hash settings, 2-second timeout, 256 MB memory, bounded recursion/iterations/container size/integer magnitude. Deduplication combines canonical AST, token MinHash, and private multi-input behavioral signatures. Generator namespaces and grammar productions are split before generation. Nonscientific fixtures are permanently excluded.

## 12. Qualification and training contract

The 300 qualification families may test evaluator determinism, parser roundtrip, validity/evaluability, champion difficulty, frozen-bank opportunity, independence, near duplicates, runtime, and repetition. They cannot be reused for confirmation, and routed gain is not a qualification gate. The final router remains trained only on the original 300 CRUXEval development families. Refitting on a new-development split would create a new candidate system and require a new untouched confirmation split; reopening architecture selection would be a new development phase.

## 13. Safety, economics, and controls

A future positive claim requires positive routed utility, commitment validity and semantic evaluability no worse than champion by 3 pp, one-call accounting, token/latency and routing-concentration reporting, and no frozen pathological failure mode. Primary comparator is the frozen champion. Baseline, random routing in the same bank, prompt-only control, and optional matched-random bank are secondary; oracle/headroom is diagnostic only.

## 14. Reviewer/fragility audit

The release summary records fourteen attacks with the fragility, cheapest discriminative check, and stop condition. The central ones are development overfit, synthetic shortcuts, family leakage, comparator leakage, invalid matched-seed coupling, and confusing one-call deployment with evaluation cost. The cheapest decisive check is the untouched, independently generated family-level confirmation. Q3 should remain an extension to the Q1/Q2 paper until that check exists.

## 15. Repository and resource state

- New semantic trajectories: 0
- New Qwen forwards: 0
- Fresh evaluation outcomes inspected: NO
- Q3.2 classification changed: NO
- Q3: `NOT_RUN`
- Spark 1 GPU used: NO
- Spark 2 used: NO
- RunPod used: NO
- Final items/seeds generated: 0
- Holdout permanently allocated: NO

`Q3_FRESH_INSTRUMENT_DESIGN_READY_FOR_PRELOCK`
"""
    DOC.write_text(report, encoding="utf-8")

    release_candidates = [
        *sorted(REVIEW.glob("*")),
        DOC,
    ]
    excluded = ("/Users/", "/private/tmp/", "BEGIN PRIVATE KEY", "ssh-rsa ", "hf_")
    findings = []
    for path in release_candidates:
        if not path.is_file() or path.name == "Q3_FINAL_SYSTEM_RELEASE_SAFETY.json":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in excluded:
            if pattern in text:
                findings.append({"path": str(path.relative_to(ROOT)), "pattern": pattern})
    release_safety = {
        "schema_version": "q3-final-system-release-safety-v1",
        "status": "PASS" if not findings else "FAIL",
        "files_scanned": len(release_candidates),
        "forbidden_pattern_findings": findings,
        "raw_benchmark_text": False,
        "raw_model_output": False,
        "fresh_correctness": False,
        "credentials_or_private_infrastructure": False if not findings else None,
        "private_router_parameters": "HASH_PINNED_NOT_TRACKED",
        "personal_handbook_modified": False,
    }
    write_json("Q3_FINAL_SYSTEM_RELEASE_SAFETY.json", release_safety)
    if findings:
        raise RuntimeError(f"release-safety findings: {findings}")

    artifacts = [
        "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK.json",
        "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK_ADDITIVE_STEER.json",
        "FINAL_Q3_CANDIDATE_SYSTEM_DRAFT.json",
        "Q3_TIER_B_EXPOSURE_SEVERITY_AUDIT.json",
        "Q3_UTILITY_INFERENCE_CALIBRATION.csv",
        "Q3_UTILITY_POWER_PRECISION.csv",
        "Q3_UTILITY_POWER_PRECISION_SUMMARY.json",
        "Q3_FRESH_INSTRUMENT_SOURCE_AUDIT.json",
        "Q3_FRESH_INSTRUMENT_DESIGN_DRAFT.json",
        "Q3_RUNTIME_ACCOUNTING.json",
        "Q3_EVALUATION_SUPPLY_DECISION.json",
        "Q3_FINAL_SYSTEM_AND_SUPPLY_RELEASE_SUMMARY.json",
        "Q3_FINAL_SYSTEM_RELEASE_SAFETY.json",
    ]
    manifest = {
        "schema_version": "q3-final-system-and-supply-artifact-hashes-v1",
        "artifacts": {
            name: {"sha256": sha256_file(REVIEW / name), "bytes": (REVIEW / name).stat().st_size}
            for name in artifacts
        },
        "review": {
            "path": str(DOC.relative_to(ROOT)),
            "sha256": sha256_file(DOC),
            "bytes": DOC.stat().st_size,
        },
        "private_router": system["router"]["private_parameter_manifest"],
    }
    write_json("Q3_FINAL_SYSTEM_ARTIFACT_HASHES.json", manifest)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "system": decision["final_candidate_system"],
                "tier_b": decision["tier_b"],
                "selected": selected,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
