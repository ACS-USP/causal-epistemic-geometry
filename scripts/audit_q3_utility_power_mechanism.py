#!/usr/bin/env python3
"""Model-free audit of Q3 oracle optimism and utility power mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_fresh_instrument_qualification"
PRECHECK = REVIEW / "Q3_EXTERNAL_REVIEW_INTEGRITY_AUDIT_PRECHECK.json"
FINAL_SYSTEM = (
    ROOT
    / "review/q3_final_system_and_evaluation_supply/FINAL_Q3_CANDIDATE_SYSTEM_DRAFT.json"
)
SUPPORT = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def t_quantile(probability: float, degrees: int) -> float:
    z = NormalDist().inv_cdf(probability)
    nu = float(degrees)
    return (
        z
        + (z**3 + z) / (4 * nu)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * nu**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * nu**3)
    )


def wilson(successes: int, total: int) -> list[float]:
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def support_probabilities(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    zero = 1.0 - positive - negative
    probabilities = np.asarray(
        [
            np.mean(negative**2),
            np.mean(2 * negative * zero),
            np.mean(zero**2 + 2 * positive * negative),
            np.mean(2 * positive * zero),
            np.mean(positive**2),
        ]
    )
    probabilities /= probabilities.sum()
    return probabilities


def family_probabilities(
    rng: np.random.Generator, law: str, mean: float, pool: int
) -> np.ndarray:
    if law == "CONSTANT":
        return np.full(pool, mean)
    if law.startswith("LOGIT_NORMAL_"):
        sd = float(law.rsplit("_", 1)[1])
        base = (mean - 0.05) / 0.90
        logit = math.log(base / (1 - base))
        return 0.05 + 0.90 / (1 + np.exp(-(logit + rng.normal(0, sd, pool))))
    if law.startswith("BETA_"):
        concentration = float(law.rsplit("_", 1)[1])
        base = (mean - 0.05) / 0.90
        return 0.05 + 0.90 * rng.beta(
            base * concentration, (1 - base) * concentration, pool
        )
    raise ValueError(law)


def panel_simulation(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    n: int,
    replicates: int,
    target: float,
) -> dict[str, Any]:
    counts = rng.multinomial(n, probabilities, size=replicates)
    sums = counts @ SUPPORT
    sums2 = counts @ (SUPPORT * SUPPORT)
    means = sums / n
    variance = np.maximum((sums2 - sums * sums / n) / (n - 1), 0)
    se = np.sqrt(variance / n)
    statistic = np.divide(means, se, out=np.zeros_like(means), where=se > 0)
    reject = statistic > t_quantile(0.95, n - 1)
    critical = t_quantile(0.975, n - 1)
    lower, upper = means - critical * se, means + critical * se
    covered = (lower <= target) & (target <= upper)
    return {
        "rejection_rate": float(reject.mean()),
        "rejection_wilson_95": wilson(int(reject.sum()), replicates),
        "ci_coverage": float(covered.mean()),
        "mean_estimate": float(means.mean()),
        "mean_standard_error": float(se.mean()),
        "mean_ci_width": float(np.mean(upper - lower)),
        "degenerate_rate": float(np.mean(se == 0)),
    }


def independent_cell(
    rng: np.random.Generator,
    law: str,
    p_mean: float,
    delta: float,
    n: int,
    replicates: int,
    pool: int,
) -> dict[str, Any]:
    champion = family_probabilities(rng, law, p_mean, pool)
    router = champion + delta
    if np.any((router <= 0) | (router >= 1)):
        raise ValueError("probability law exceeds frozen safe range")
    positive = router * (1 - champion)
    negative = champion * (1 - router)
    probabilities = support_probabilities(positive, negative)
    target = float(np.mean(router - champion))
    result = panel_simulation(rng, probabilities, n, replicates, target)
    empirical_variance = float(probabilities @ ((SUPPORT - target) ** 2))
    identity_variance = float(
        np.var(router - champion)
        + np.mean(router * (1 - router) + champion * (1 - champion)) / 2
    )
    result.update(
        {
            "mechanism": "CONDITIONALLY_INDEPENDENT_BERNOULLI",
            "law": law,
            "p_mean_requested": p_mean,
            "p_champion_mean": float(champion.mean()),
            "delta": delta,
            "N": n,
            "R": 2,
            "q_identity": float(np.mean(positive + negative)),
            "support_probabilities": probabilities.tolist(),
            "variance_empirical_distribution": empirical_variance,
            "variance_identity": identity_variance,
            "variance_identity_abs_difference": abs(empirical_variance - identity_variance),
        }
    )
    return result


def historical_ternary_cell(
    rng: np.random.Generator, n: int, replicates: int, gain: float, discordance: float
) -> dict[str, Any]:
    positive = np.asarray([(discordance + gain) / 2])
    negative = np.asarray([(discordance - gain) / 2])
    probabilities = support_probabilities(positive, negative)
    result = panel_simulation(rng, probabilities, n, replicates, gain)
    result.update(
        {
            "mechanism": "HISTORICAL_TERNARY_FIXED_DISCORDANCE_COMPARATOR",
            "law": "FIXED_Q",
            "delta": gain,
            "discordance": discordance,
            "N": n,
            "R": 2,
            "support_probabilities": probabilities.tolist(),
        }
    )
    return result


def conservative_combined_independent(
    rng: np.random.Generator, n: int, replicates: int, pool: int
) -> dict[str, Any]:
    concentration = 1 / 0.35 - 1
    champion = rng.beta(0.60 * concentration, 0.40 * concentration, pool)
    harm = rng.random(pool) < 0.10
    delta = rng.normal((0.03 + 0.10 * 0.10) / 0.90, 0.05, pool)
    delta[harm] = rng.normal(-0.10, 0.02, int(harm.sum()))
    delta = delta - delta.mean() + 0.03
    router = np.clip(champion + delta, 0.001, 0.999)
    # Preserve the declared +.03 planning estimand after probability clipping.
    realized = router - champion
    router = np.clip(router + (0.03 - realized.mean()), 0.001, 0.999)
    target = float(np.mean(router - champion))
    positive = router * (1 - champion)
    negative = champion * (1 - router)
    probabilities = support_probabilities(positive, negative)
    result = panel_simulation(rng, probabilities, n, replicates, target)
    result.update(
        {
            "mechanism": "CONDITIONALLY_INDEPENDENT_BERNOULLI",
            "law": "CONSERVATIVE_COMBINED_MATCH_TO_HISTORICAL",
            "delta": target,
            "N": n,
            "R": 2,
            "q_identity": float(np.mean(positive + negative)),
            "support_probabilities": probabilities.tolist(),
        }
    )
    return result


def oracle_audit(rng: np.random.Generator, replicates: int) -> dict[str, Any]:
    policies = rng.binomial(2, 0.5, size=(replicates, 8)) / 2
    champion = rng.binomial(2, 0.5, size=replicates) / 2
    empirical = policies.max(axis=1) - champion
    analytic_max = 1 - 0.5 * 0.75**8 - 0.5 * 0.25**8
    return {
        "K": 8,
        "R": 2,
        "p": 0.5,
        "replicates": replicates,
        "analytic_expected_empirical_max": analytic_max,
        "analytic_expected_apparent_headroom": analytic_max - 0.5,
        "simulated_expected_empirical_max": float(policies.max(axis=1).mean()),
        "simulated_apparent_headroom": float(empirical.mean()),
        "true_policy_specialization": 0.0,
        "historical_gate_changed": False,
        "corrected_interpretation": (
            "OPPORTUNITY_UPPER_BOUND_DIAGNOSTIC_ONLY; passing alone does not establish "
            "repeatable complementarity or selectability"
        ),
        "rollout_transfer_diagnostic": (
            "FROZEN_FOR_POST_COMPLETION_EXPLORATORY_USE_ONLY; not run on partial collection"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, default=1_000_000)
    args = parser.parse_args()
    precheck = json.loads(PRECHECK.read_text(encoding="utf-8"))
    if precheck["status"] != "FROZEN_BEFORE_PRIVATE_STRUCTURAL_AUDIT":
        raise SystemExit("audit precheck is not frozen")
    cfg = precheck["power_audit"]
    if cfg["replicates_per_scenario"] != 50_000:
        raise SystemExit("replicate count drift")
    master = np.random.SeedSequence(cfg["master_seed"])
    rng_oracle, rng_cells = (np.random.default_rng(seed) for seed in master.spawn(2))
    cells: list[dict[str, Any]] = []
    laws = ["CONSTANT", "LOGIT_NORMAL_0.5", "LOGIT_NORMAL_1.0", "BETA_4", "BETA_12"]
    for n in (800, 1000):
        for p_mean in (0.35, 0.50, 0.70):
            cells.append(independent_cell(rng_cells, "CONSTANT", p_mean, 0.0, n, 50_000, args.pool))
        for law in laws[1:]:
            cells.append(independent_cell(rng_cells, law, 0.50, 0.0, n, 50_000, args.pool))
        for delta in (0.01, 0.02, 0.03, 0.04):
            for law in ("CONSTANT", "LOGIT_NORMAL_1.0", "BETA_4"):
                cells.append(independent_cell(rng_cells, law, 0.50, delta, n, 50_000, args.pool))
        cells.append(historical_ternary_cell(rng_cells, n, 50_000, 0.03, 0.35))
        cells.append(conservative_combined_independent(rng_cells, n, 50_000, args.pool))

    final_system = json.loads(FINAL_SYSTEM.read_text(encoding="utf-8"))
    bank = [row["policy_id"] for row in final_system["portfolio"]["policies"]]
    champion = final_system["champion"]["policy_id"]
    exact_policy_sharing_possible = champion in bank
    null_cells = [row for row in cells if row["delta"] == 0.0]
    target_cells = [
        row
        for row in cells
        if row["N"] == 1000
        and abs(row["delta"] - 0.03) < 1e-6
        and row["mechanism"] == "CONDITIONALLY_INDEPENDENT_BERNOULLI"
    ]
    output = {
        "schema_version": "q3-external-review-oracle-power-audit-v1",
        "classification": "POST_MATERIALIZATION_MODEL_FREE_DIAGNOSTIC",
        "precheck_sha256": sha256(PRECHECK),
        "oracle": oracle_audit(rng_oracle, precheck["oracle_audit"]["simulation_replicates"]),
        "power_mechanism": {
            "historical_independent_branch_actual_mechanism": (
                "TERNARY_DIFFERENCE_WITH_USER_CHOSEN_DISCORDANCE; not two conditionally "
                "independent Bernoulli correctness streams"
            ),
            "intended_mechanism": cfg["independent_mechanism"],
            "formula_checks_max_abs_difference": max(
                row.get("variance_identity_abs_difference", 0.0) for row in cells
            ),
            "null_max_fpr": max(row["rejection_rate"] for row in null_cells),
            "null_min_coverage": min(row["ci_coverage"] for row in null_cells),
            "n1000_delta_003_power_range_under_simple_independent_laws": [
                min(row["rejection_rate"] for row in target_cells),
                max(row["rejection_rate"] for row in target_cells),
            ],
            "historical_reported_power_0_8233_defensible": (
                "ONLY_FOR_THE_HISTORICAL_TERNARY_CONSERVATIVE_COMBINED_PLANNING_MODEL; "
                "not as a mechanism-free guarantee for the new generator"
            ),
            "confirmation_design_changed": False,
            "confirmation_authorization_implication": (
                "qualification may proceed; confirmation power justification requires explicit "
                "review of the independent-Bernoulli range before confirmation opens"
            ),
            "exact_policy_sharing": {
                "champion_in_eight_policy_bank": exact_policy_sharing_possible,
                "planned_shared_execution": False,
                "interpretation": (
                    "The external champion is absent from the eight-policy bank. A non-finite "
                    "fallback does not authorize outcome reuse, and distinct conditions retain "
                    "independent frozen seeds."
                ),
            },
            "replicates_per_cell": 50_000,
            "family_probability_pool": args.pool,
            "cells": cells,
        },
        "qualification_outcomes_used": False,
        "confirmation_outcomes_used": False,
        "qwen_inference": 0,
    }
    path = REVIEW / "Q3_EXTERNAL_REVIEW_ORACLE_POWER_AUDIT.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "oracle": output["oracle"],
                "power_summary": {
                    key: value
                    for key, value in output["power_mechanism"].items()
                    if key != "cells"
                },
                "cells": len(cells),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
