#!/usr/bin/env python3
"""Independent raw-row forensic audit for Gate 10.

The audit reuses only Gate 7's raw-row traversal shell. Character-count
parsing, the frozen Gate-10 decision rule, opportunity checks, normalized
estimands, and oracle checks are independently implemented here.
"""

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

from epistemic_geometry.benchmarks.v4.character_semantic_v3 import (  # noqa: E402
    evaluate_character_count_answer_v3,
)
from epistemic_geometry.experiments import gate10  # noqa: E402

REVIEW = ROOT / "review/gate10_cross_domain_charcount"


def _reparse(row: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_character_count_answer_v3(
        str(row.get("raw_output", "")),
        str(row["reference_answer"]),
        truncated=int(row.get("generated_token_count", 0)) >= gate10.MAX_NEW_TOKENS,
        runtime_error=str(row.get("status")) == "RUNTIME_ERROR",
    )
    return {
        "correct": bool(result.correct),
        "commitment_valid": bool(result.commitment_valid),
        "semantic_evaluable": bool(result.semantic_evaluable),
    }


def _manual_classification(
    summaries: dict[str, dict[str, float]],
    points: dict[str, dict[str, float]],
    randoms: dict[str, dict[str, float]],
    intervals: dict[str, dict[str, float]],
    loo: list[dict[str, str]],
) -> str:
    baseline = summaries[gate10.BASELINE]
    textual = summaries[gate10.TEXTUAL]
    controller = summaries[gate10.MEANINGFUL]
    point = points[gate10.MEANINGFUL]

    b00 = point["B00"]
    errors_wrong_twice = round(b00 * 200)
    opportunity = bool(
        baseline["commitment_validity"] >= 0.95
        and baseline["semantic_evaluability"] >= 0.95
        and 0.55 <= baseline["accuracy"] <= 0.95
        and b00 >= 0.04
        and errors_wrong_twice >= 8
        and 200 - errors_wrong_twice >= 20
    )
    commitment = bool(
        controller["commitment_validity"] >= 0.95
        and controller["commitment_validity"] >= baseline["commitment_validity"] - 0.03
    )
    evaluability = bool(
        controller["semantic_evaluability"] >= 0.95
        and controller["semantic_evaluability"]
        >= baseline["semantic_evaluability"] - 0.03
    )
    competence = controller["accuracy"] >= baseline["accuracy"] - 0.05
    safe = commitment and evaluability and competence
    above_mean = {name: point[name] > randoms[name]["mean"] for name in ("G", "C", "D")}
    above_max = {name: point[name] > randoms[name]["max"] for name in ("G", "C", "D")}
    g_norm = point["G"] / b00 if b00 > 0 else None
    loo_positive = {
        name: all(float(row[name]) > 0 for row in loo) for name in ("G", "C", "D")
    }
    strong = bool(
        opportunity
        and safe
        and point["G"] >= 0.030
        and point["C"] >= 0.015
        and point["D"] >= 0.040
        and point["G"] - randoms["G"]["mean"] >= 0.025
        and point["C"] - randoms["C"]["mean"] >= 0.015
        and point["D"] - randoms["D"]["mean"] >= 0.030
        and all(above_max.values())
        and g_norm is not None
        and g_norm >= 0.15
        and point["rescue"] >= point["damage"]
        and all(
            intervals[name]["q025"] > 0
            for name in (
                "meaningful:G",
                "meaningful:C",
                "meaningful:D",
                "meaningful:G_minus_random_mean",
                "meaningful:C_minus_random_mean",
            )
        )
        and all(loo_positive.values())
    )
    minimum = bool(
        opportunity
        and safe
        and all(point[name] > 0 for name in ("G", "C", "D"))
        and all(above_mean.values())
        and sum(above_max.values()) >= 2
        and g_norm is not None
        and g_norm >= 0.08
        and point["rescue"] >= point["damage"]
    )
    movement = bool(
        opportunity and safe and point["D"] >= 0.030 and above_mean["D"] and above_max["D"]
    )
    accuracy_change = controller["accuracy"] - baseline["accuracy"]
    accuracy_positive = intervals["meaningful:accuracy_change"]["q025"] > 0
    source = bool(
        textual["commitment_validity"] >= 0.95
        and textual["semantic_evaluability"] >= 0.95
        and textual["accuracy"] >= baseline["accuracy"] - 0.03
        and (
            textual["mean_tokens"] >= 1.5 * baseline["mean_tokens"]
            or textual["median_tokens"] >= baseline["median_tokens"] + 10
            or textual["accuracy"] >= baseline["accuracy"] + 0.03
        )
    )
    token_denominator = textual["mean_tokens"] - baseline["mean_tokens"]
    style = bool(
        source
        and token_denominator > 0
        and (controller["mean_tokens"] - baseline["mean_tokens"]) / token_denominator >= 0.25
    )

    if not opportunity:
        return "GATE10_INSTRUMENT_CEILING_OR_FLOOR"
    if not safe:
        return "GATE10_CROSS_DOMAIN_DESTRUCTIVE"
    if strong:
        return "GATE10_STRONG_CROSS_DOMAIN_USEFUL_COMPLEMENTARITY"
    if minimum:
        return "GATE10_MINIMUM_CROSS_DOMAIN_CONTROL_SIGNAL"
    if movement:
        return "GATE10_CROSS_DOMAIN_ERROR_PROFILE_MOVEMENT_ONLY"
    if accuracy_change > 0 and accuracy_positive:
        return "GATE10_CROSS_DOMAIN_COMPETENCE_GAIN_WITHOUT_COMPLEMENTARITY"
    if style:
        return "GATE10_CAREFUL_STYLE_TRANSFER_ONLY"
    return "GATE10_NO_CROSS_DOMAIN_TRANSFER"


def _configure_core() -> None:
    core.REVIEW = REVIEW
    core.BASELINE = gate10.BASELINE
    core.CONDITIONS = gate10.CONDITIONS
    core.EXPERIMENT_ID = gate10.EXPERIMENT_ID
    core.MAX_NEW_TOKENS = gate10.MAX_NEW_TOKENS
    core.MEANINGFUL = gate10.MEANINGFUL
    core.RANDOMS = gate10.RANDOM_NAMES
    core.TEXTUAL = gate10.TEXTUAL
    core.reparse = _reparse
    core.manual_classification = _manual_classification


def _oracle_checks(review: Path) -> dict[str, Any]:
    manifest = core.read_json(review / "EVALUATION_MANIFEST.json")
    historical = core.read_json(review / "HISTORICAL_CHARCOUNT_EXCLUSION_DIGEST.json")
    journal = core.read_jsonl(review / "journal.jsonl")
    items = manifest["items"]
    item_by_id = {item["item_id"]: item for item in items}
    ids = [item["item_id"] for item in items]
    strings = [item["text"] for item in items]
    hashes = [item["item_hash"] for item in items]
    seeds = [int(item["generator_seed"]) for item in items]
    exact_oracles = all(
        item["text"].count(item["target_character"]) == int(item["answer"])
        == int(item["metadata"]["exact_oracle"])
        for item in items
    )
    journal_binding = all(
        str(row["reference_answer"]) == str(item_by_id[row["item_id"]]["answer"])
        and row["item_metadata"]["item_hash"] == item_by_id[row["item_id"]]["item_hash"]
        and row["item_metadata"]["text"] == item_by_id[row["item_id"]]["text"]
        and row["item_metadata"]["target_character"]
        == item_by_id[row["item_id"]]["target_character"]
        for row in journal
    )
    checks = {
        "manifest_n_200": len(items) == 200,
        "unique_item_ids": len(ids) == len(set(ids)),
        "unique_strings": len(strings) == len(set(strings)),
        "unique_item_hashes": len(hashes) == len(set(hashes)),
        "unique_generator_seeds": len(seeds) == len(set(seeds)),
        "exact_integer_oracles": exact_oracles,
        "no_historical_id_collision": not set(ids).intersection(historical["historical_item_ids"]),
        "no_historical_string_collision": not set(strings).intersection(
            historical["historical_exact_strings"]
        ),
        "no_historical_hash_collision": not set(hashes).intersection(
            historical["historical_item_hashes"]
        ),
        "no_historical_seed_collision": not set(seeds).intersection(
            historical["historical_generator_seeds"]
        ),
        "journal_manifest_binding": journal_binding,
    }
    return {"classification": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def audit(review: Path) -> dict[str, Any]:
    _configure_core()
    payload = core.audit(review)
    oracle = _oracle_checks(review)
    primary = core.read_json(review / "ESTIMANDS.json")
    point = primary["estimands"][gate10.MEANINGFUL]
    primary_g_norm = float(point["G_norm"])
    audited_g_norm = float(point["G"] / point["B00"])
    g_norm_difference = audited_g_norm - primary_g_norm
    integrity = bool(
        payload["classification"] == "GATE7_FORENSIC_CLEAN"
        and oracle["classification"] == "PASS"
        and abs(g_norm_difference) <= 1e-12
    )
    payload["classification"] = (
        "GATE10_FORENSIC_CLEAN"
        if integrity
        else "GATE10_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
    )
    payload["oracle_generator_crosscheck"] = oracle
    payload["G_norm_crosscheck"] = {
        "primary": primary_g_norm,
        "audit": audited_g_norm,
        "difference": g_norm_difference,
    }
    payload.pop("historical_gate6_3_result_modified", None)
    core.write_json(review / "ORACLE_CROSSCHECK.json", oracle)
    core.write_json(review / "FORENSIC_AUDIT.json", payload)
    exact_seed_schedule = payload["seed_formula_exact"] and payload["seed_unique"]
    (review / "FORENSIC_AUDIT.md").write_text(
        "# Gate 10 independent forensic audit\n\n"
        f"Classification: `{payload['classification']}`.\n\n"
        f"- Frozen/observed rows: {payload['expected_rows']}/{payload['actual_rows']}\n"
        f"- Unique logical keys: {payload['logical_keys_unique']}\n"
        f"- Exact independent seed schedule: {exact_seed_schedule}\n"
        f"- Condition-symmetric character semantic-v3 reparse: "
        f"{payload['parser_condition_symmetric_reparse']}\n"
        f"- Maximum primary/audit metric difference: "
        f"{payload['metric_max_abs_difference']:.3g}\n"
        f"- G_norm difference: {g_norm_difference:.3g}\n"
        f"- Generator/oracle crosscheck: {oracle['classification']}\n"
        f"- Classification agreement: {payload['classification_agreement']}\n\n"
        "All causal estimands were independently recomputed from raw binary outcome "
        "arrays without calling the Gate-10 primary analysis path. The exact integer "
        "oracle and historical collision firewall were also independently checked.\n",
        encoding="utf-8",
    )
    if not integrity:
        raise RuntimeError("GATE10_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN")
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
