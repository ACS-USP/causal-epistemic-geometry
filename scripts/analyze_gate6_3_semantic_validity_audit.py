#!/usr/bin/env python3
"""Condition-symmetric offline semantic-V3 audit of preserved Gate 6.3 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.base import evaluate_external_answer  # noqa: E402
from epistemic_geometry.benchmarks.external.semantic_v2 import (  # noqa: E402
    parse_external_answer_v2,
)
from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    canonicalize_semantic_value,
    evaluate_external_answer_v3,
)
from epistemic_geometry.experiments.gate6_3_v3 import (  # noqa: E402
    audit_two_rollout_estimands,
    classify_semantic_v3,
    item_contributions,
    random_metric_summary,
)

SOURCE = ROOT / "review/gate6_3_single_mean_semantic_evaluation"
AUDIT = ROOT / "review/gate6_3_semantic_validity_audit"
HISTORICAL_CLASSIFICATION = "GATE6_3_SINGLE_MEAN_DESTRUCTIVE"
MEANINGFUL = "BEST_SINGLE_MEAN_PLUS"
BASELINE = "BASELINE"
TEXTUAL = "TEXTUAL_CAREFUL_REFERENCE"
RANDOMS = tuple(f"SINGLE_L27_RANDOM_R{i}" for i in range(4))
CONDITIONS = (BASELINE, TEXTUAL, MEANINGFUL, *RANDOMS)
VALID_V2 = {"VALID_CORRECT", "VALID_WRONG"}
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 6313003

csv.field_size_limit(sys.maxsize)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_journal() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (SOURCE / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def v2_score(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("status")) == "RUNTIME_ERROR":
        return {
            "status": "RUNTIME_ERROR",
            "parsed_answer": None,
            "correct": False,
            "parse_reason": str(row.get("error") or "runtime error"),
        }
    token_count = int(row.get("generated_token_count", 0))
    parsed = parse_external_answer_v2(
        str(row.get("raw_output", "")),
        truncated=str(row.get("status")) == "TRUNCATED" or token_count >= 4096,
    )
    if parsed.status is not None:
        return {
            "status": parsed.status.value.replace("TRUNCATED_THINKING", "TRUNCATED"),
            "parsed_answer": parsed.answer_text,
            "correct": False,
            "parse_reason": parsed.parse_reason,
        }
    try:
        correct = evaluate_external_answer(
            parsed.answer_text or "",
            str(row["reference_answer"]),
            str(row["evaluator"]),
        )
    except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as exc:
        return {
            "status": "INVALID_FORMAT",
            "parsed_answer": parsed.answer_text,
            "correct": False,
            "parse_reason": f"typed evaluator rejected answer: {type(exc).__name__}",
        }
    return {
        "status": "VALID_CORRECT" if correct else "VALID_WRONG",
        "parsed_answer": parsed.answer_text,
        "correct": bool(correct),
        "parse_reason": None,
    }


def v3_score(row: dict[str, Any]) -> dict[str, Any]:
    token_count = int(row.get("generated_token_count", 0))
    runtime_error = str(row.get("status")) == "RUNTIME_ERROR"
    truncated = str(row.get("status")) == "TRUNCATED" or token_count >= 4096
    result = evaluate_external_answer_v3(
        str(row.get("raw_output", "")),
        str(row.get("reference_answer", "")),
        truncated=truncated,
        runtime_error=runtime_error,
    )
    if runtime_error:
        status = "RUNTIME_ERROR"
    elif truncated:
        status = "TRUNCATED"
    elif not result.commitment_valid or not result.semantic_evaluable:
        status = "INVALID_FORMAT"
    else:
        status = "VALID_CORRECT" if result.correct else "VALID_WRONG"
    return {
        "status": status,
        "commitment_valid": result.commitment_valid,
        "semantic_evaluable": result.semantic_evaluable,
        "value_type": result.value_type,
        "canonical_value": result.canonical_value,
        "correct": result.correct,
        "failure_reason": result.failure_reason,
        "payload": result.payload,
    }


def reason_for_change(v2: dict[str, Any], v3: dict[str, Any]) -> str:
    if (
        v2["status"] == v3["status"]
        and bool(v2["correct"]) == bool(v3["correct"])
        and (v2["parsed_answer"] or None) == (v3["payload"] or None)
    ):
        return "unchanged"
    if v2["status"] == "INVALID_FORMAT" and v3["status"] == "VALID_CORRECT":
        return "V3 represents one unambiguous commitment that matches the reference"
    if v2["status"] == "INVALID_FORMAT" and v3["status"] == "VALID_WRONG":
        return "V3 represents one unambiguous wrong commitment instead of mechanical invalidity"
    if v2["status"] == "VALID_WRONG" and v3["status"] == "VALID_CORRECT":
        return "V3 typed canonicalization resolves an equivalent semantic value"
    if v2["status"] in VALID_V2 and v3["status"] == "INVALID_FORMAT":
        return "V3 rejects an ambiguous or multiply committed final section"
    if v2["status"] == v3["status"] and (v2["parsed_answer"] or None) != (v3["payload"] or None):
        return "V3 extracts a different globally specified final payload"
    return f"V2 {v2['status']} -> V3 {v3['status']}"


def verify_lock() -> dict[str, Any]:
    lock = load_json(AUDIT / "AUDIT_LOCK.json")
    parser_path = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    checks = {
        "parser": digest(parser_path) == lock["parser"]["module_sha256"],
        "spec": digest(AUDIT / "SEMANTIC_V3_SPEC.md") == lock["parser"]["spec_sha256"],
        "tests": digest(ROOT / "tests/test_semantic_v3.py") == lock["parser"]["tests_sha256"],
        "corpus": digest(AUDIT / "BLINDED_CORPUS.jsonl") == lock["blinded_corpus_sha256"],
        "journal": digest(SOURCE / "journal.jsonl") == lock["source_journal_sha256"],
        "historical_classification": lock["historical_classification"]
        == HISTORICAL_CLASSIFICATION,
    }
    for name, expected in lock["immutable_source_files"].items():
        checks[f"immutable:{name}"] = digest(SOURCE / name) == expected
    if not all(checks.values()):
        raise RuntimeError(f"semantic V3 audit lock verification failed: {checks}")
    return {"lock": lock, "checks": checks}


def historical_v2_index() -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = list(csv.DictReader((SOURCE / "EVALUATION_RESULTS.csv").open(encoding="utf-8")))
    if len(rows) != 840:
        raise RuntimeError(f"expected 840 historical V2 result rows, found {len(rows)}")
    return {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): {
            "status": str(row["v2_status"]),
            "parsed_answer": row["v2_parsed_answer"] or None,
            "correct": str(row["v2_correct"]).lower() == "true",
        }
        for row in rows
    }


def score_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    historical = historical_v2_index()
    output: list[dict[str, Any]] = []
    v2_crosscheck_failures: list[dict[str, Any]] = []
    unchanged_correctness_failures: list[dict[str, Any]] = []
    for row in rows:
        phase = str(row.get("phase"))
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        v2 = v2_score(row)
        if phase == "GATE6_3_PRIMARY_EVALUATION":
            stored = historical[key]
            if any(
                (
                    stored["status"] != v2["status"],
                    stored["parsed_answer"] != (v2["parsed_answer"] or None),
                    stored["correct"] != bool(v2["correct"]),
                )
            ):
                v2_crosscheck_failures.append({"key": key, "stored": stored, "recomputed": v2})
        v3 = v3_score(row)
        reason = reason_for_change(v2, v3)
        if v2["status"] == v3["status"] and bool(v2["correct"]) != bool(v3["correct"]):
            unchanged_correctness_failures.append({"key": key, "v2": v2, "v3": v3})
        output.append(
            {
                "phase": phase,
                "item_id": key[0],
                "condition": key[1],
                "rollout_index": key[2],
                "v2_status": v2["status"],
                "v2_parsed_answer": v2["parsed_answer"],
                "v2_correct": bool(v2["correct"]),
                "v3_status": v3["status"],
                "v3_commitment_valid": bool(v3["commitment_valid"]),
                "v3_semantic_evaluable": bool(v3["semantic_evaluable"]),
                "v3_value_type": v3["value_type"],
                "v3_canonical_value": v3["canonical_value"],
                "v3_payload": v3["payload"],
                "v3_correct": bool(v3["correct"]),
                "v3_failure_reason": v3["failure_reason"],
                "reason_for_change": reason,
                "reference_type": str(
                    canonicalize_semantic_value(str(row.get("reference_answer", "")))[0]
                ),
                "generated_token_count": int(row.get("generated_token_count", 0)),
            }
        )
    if v2_crosscheck_failures or unchanged_correctness_failures:
        raise RuntimeError(
            "V2 or unchanged-row strict crosscheck failed: "
            f"v2={len(v2_crosscheck_failures)} unchanged={len(unchanged_correctness_failures)}"
        )
    return output, {
        "historical_v2_rows_crosschecked": 840,
        "historical_v2_crosscheck_failures": 0,
        "unchanged_row_correctness_failures": 0,
    }


def condition_summary(
    condition: str,
    rows: list[dict[str, Any]],
    historical: dict[str, dict[str, str]],
) -> dict[str, Any]:
    commitment = sum(bool(row["v3_commitment_valid"]) for row in rows)
    evaluable = sum(bool(row["v3_semantic_evaluable"]) for row in rows)
    correct = sum(bool(row["v3_correct"]) for row in rows)
    truncated = sum(row["v3_status"] == "TRUNCATED" for row in rows)
    runtime = sum(row["v3_status"] == "RUNTIME_ERROR" for row in rows)
    no_commitment = sum(
        row["v3_failure_reason"] in {"no final commitment", "empty final commitment"}
        for row in rows
    )
    ambiguous = sum(
        not bool(row["v3_commitment_valid"])
        and row["v3_status"] == "INVALID_FORMAT"
        and row["v3_failure_reason"] not in {"no final commitment", "empty final commitment"}
        for row in rows
    )
    tokens = [int(row["generated_token_count"]) for row in rows]
    n = len(rows)
    return {
        "condition": condition,
        "n": n,
        "v2_historical_valid": int(historical[condition]["valid"]),
        "v2_historical_validity": float(historical[condition]["validity"]),
        "commitment_valid": commitment,
        "commitment_validity": commitment / n,
        "semantic_evaluable": evaluable,
        "semantic_evaluability": evaluable / n,
        "correct": correct,
        "wrong": evaluable - correct,
        "no_commitment": no_commitment,
        "ambiguous_commitment": ambiguous,
        "truncated": truncated,
        "runtime_error": runtime,
        "accuracy": correct / n,
        "mean_tokens": statistics.mean(tokens),
        "median_tokens": statistics.median(tokens),
        "max_tokens": max(tokens),
    }


def arrays_for(
    rows: list[dict[str, Any]], *, field: str
) -> tuple[list[str], dict[str, np.ndarray]]:
    item_ids = sorted({str(row["item_id"]) for row in rows})
    lookup = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): row
        for row in rows
    }
    arrays = {
        condition: np.asarray(
            [
                [int(bool(lookup[(item, condition, rollout)][field])) for rollout in (0, 1)]
                for item in item_ids
            ],
            dtype=np.int8,
        )
        for condition in CONDITIONS
    }
    return item_ids, arrays


def bootstrap_intervals(
    errors: dict[str, np.ndarray],
    commitment: dict[str, np.ndarray],
    evaluable: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: defaultdict[str, list[float]] = defaultdict(list)
    item_count = len(errors[BASELINE])
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, item_count, size=item_count)
        point: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS[1:]:
            estimate = audit_two_rollout_estimands(
                errors[BASELINE][indices], errors[condition][indices]
            )
            point[condition] = estimate
            samples[f"{condition}:accuracy_difference"].append(
                estimate["accuracy_condition"] - estimate["accuracy_baseline"]
            )
            samples[f"{condition}:commitment_validity_difference"].append(
                float(commitment[condition][indices].mean())
                - float(commitment[BASELINE][indices].mean())
            )
            samples[f"{condition}:semantic_evaluability_difference"].append(
                float(evaluable[condition][indices].mean())
                - float(evaluable[BASELINE][indices].mean())
            )
            for metric in ("G", "C", "D", "rescue", "damage"):
                samples[f"{condition}:{metric}"].append(estimate[metric])
        for metric in ("G", "C", "D"):
            random_values = [point[name][metric] for name in RANDOMS]
            samples[f"{MEANINGFUL}:{metric}_minus_random_mean"].append(
                point[MEANINGFUL][metric] - float(np.mean(random_values))
            )
            samples[f"{MEANINGFUL}:{metric}_minus_random_max"].append(
                point[MEANINGFUL][metric] - float(np.max(random_values))
            )
    return {
        key: {
            "estimate": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "resamples": BOOTSTRAP_RESAMPLES,
            "cluster": "item_id",
        }
        for key, values in sorted(samples.items())
    }


def leave_one_out(
    item_ids: list[str],
    errors: dict[str, np.ndarray],
    commitment: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    full = audit_two_rollout_estimands(errors[BASELINE], errors[MEANINGFUL])
    full_accuracy_difference = full["accuracy_condition"] - full["accuracy_baseline"]
    full_commitment = float(commitment[MEANINGFUL].mean())
    records: list[dict[str, Any]] = []
    for index, item_id in enumerate(item_ids):
        mask = np.arange(len(item_ids)) != index
        estimate = audit_two_rollout_estimands(
            errors[BASELINE][mask], errors[MEANINGFUL][mask]
        )
        accuracy_difference = estimate["accuracy_condition"] - estimate["accuracy_baseline"]
        commitment_validity = float(commitment[MEANINGFUL][mask].mean())
        records.append(
            {
                "left_out_item_id": item_id,
                "accuracy_difference": accuracy_difference,
                "accuracy_difference_delta_from_full": accuracy_difference
                - full_accuracy_difference,
                "controller_commitment_validity": commitment_validity,
                "commitment_validity_delta_from_full": commitment_validity - full_commitment,
                "G": estimate["G"],
                "G_delta_from_full": estimate["G"] - full["G"],
                "C": estimate["C"],
                "C_delta_from_full": estimate["C"] - full["C"],
                "D": estimate["D"],
                "D_delta_from_full": estimate["D"] - full["D"],
            }
        )
    return records


def contribution_rows(
    changed: list[dict[str, Any]],
    item_ids: list[str],
    v2_errors: dict[str, np.ndarray],
    v3_errors: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    item_index = {item: index for index, item in enumerate(item_ids)}
    v2_contributions = {
        condition: item_contributions(v2_errors[BASELINE], v2_errors[condition])
        for condition in CONDITIONS[1:]
    }
    v3_contributions = {
        condition: item_contributions(v3_errors[BASELINE], v3_errors[condition])
        for condition in CONDITIONS[1:]
    }
    output: list[dict[str, Any]] = []
    for row in changed:
        condition = str(row["condition"])
        affected = CONDITIONS[1:] if condition == BASELINE else (condition,)
        index = item_index[str(row["item_id"])]
        deltas = {
            name: {
                metric: v3_contributions[name][index][metric]
                - v2_contributions[name][index][metric]
                for metric in ("G", "C", "D", "rescue", "damage")
            }
            for name in affected
        }
        output.append(
            {
                "item_id": row["item_id"],
                "condition": condition,
                "rollout_index": row["rollout_index"],
                "affects_commitment_validity": (row["v2_status"] in VALID_V2)
                != bool(row["v3_commitment_valid"]),
                "affects_semantic_evaluability": (row["v2_status"] in VALID_V2)
                != bool(row["v3_semantic_evaluable"]),
                "affects_correctness": bool(row["v2_correct"]) != bool(row["v3_correct"]),
                "affects_G_C_D": bool(row["v2_correct"]) != bool(row["v3_correct"]),
                "affects_rescue_damage": bool(row["v2_correct"]) != bool(row["v3_correct"]),
                "item_metric_delta_json": json.dumps(deltas, sort_keys=True),
            }
        )
    return output


def next_protocol(classification: str) -> tuple[str, str]:
    if classification == "GATE6_3_V3_STRONG_SPECIFIC_CONTROL_SIGNAL":
        name = "GATE7_FRESH_SINGLE_L27_REPLICATION_PROTOCOL"
        body = """# Gate 7 — Fresh Single-L27 Replication Protocol (Draft Only)

Status: **DRAFT; NOT AUTHORIZED FOR EXECUTION**.

- Reuse Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Reuse the frozen L27 paired-mean controller and exact Gate 6.3 eta.
- Freeze `external-semantic-v3` before collecting any new output.
- Allocate fresh unseen DEVELOPMENT CRUXEval items with no controller search.
- Conditions: baseline, textual careful, meaningful controller, and at least four
  new architecture-matched random directions.
- Use two independent rollouts per item and item-cluster inference.
- Justify sample size prospectively from Gate 6.3 item-cluster variance.
- Preserve competence, commitment-validity, semantic-evaluability, random-null,
  Q2, character-count, and holdout firewalls.

This is replication, not rediscovery. Principal authorization and a new
prospective lock are required before collection.
"""
    elif classification == "GATE6_3_V3_VALIDITY_COST_CONFIRMED":
        name = "GATE7_PROSPECTIVE_DOSE_CALIBRATION_PROTOCOL"
        body = """# Gate 7 — Prospective Dose Calibration Protocol (Draft Only)

Status: **DRAFT; NOT AUTHORIZED FOR EXECUTION**.

- Use a fresh calibration split, separate from every later evaluation split.
- Test eta fractions 0.25, 0.50, and 0.75 of the frozen Gate 6.3 eta.
- Use source/teacher-forced first-stage metrics, commitment validity, and
  architecture-matched random controls.
- Select the smallest dose satisfying frozen first-stage and validity criteria.
- Do not access or render a later evaluation split during dose selection.
- Require separate principal authorization and a new lock before evaluation.
"""
    else:
        name = "GATE7_PROGRAM_B_C_PIVOT_RECOMMENDATION"
        body = """# Gate 7 — Program B/C Pivot Recommendation (Draft Only)

Status: **DRAFT; NOT AUTHORIZED FOR EXECUTION**.

The semantic audit does not justify automatic replication or dose search.
Principal review should choose between Program B (readout/style control versus
error control) and Program C (geometry of controllability). Any new model output,
controller, layer, dose, Q2 analysis, character count, or holdout use requires a
new prospective protocol and explicit authorization.
"""
    return name, body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=AUDIT)
    args = parser.parse_args()
    output = args.output.resolve()
    lock_audit = verify_lock()
    journal = read_journal()
    if len(journal) != 920:
        raise RuntimeError(f"expected 920 journal rows, found {len(journal)}")
    scored, crosscheck = score_rows(journal)
    primary = [row for row in scored if row["phase"] == "GATE6_3_PRIMARY_EVALUATION"]
    matched = [row for row in scored if row["phase"] == "GATE6_3_MATCHED_RANDOM_SUPPLEMENT"]
    if len(primary) != 840 or len(matched) != 80:
        raise RuntimeError("primary/matched phase partition is not 840/80")

    historical_rows = list(
        csv.DictReader((SOURCE / "CONDITION_SUMMARY.csv").open(encoding="utf-8"))
    )
    historical = {row["condition"]: row for row in historical_rows}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        grouped[str(row["condition"])].append(row)
    summaries = {
        condition: condition_summary(condition, grouped[condition], historical)
        for condition in CONDITIONS
    }

    item_ids, v3_correct = arrays_for(primary, field="v3_correct")
    _item_ids, v2_correct = arrays_for(primary, field="v2_correct")
    _item_ids, commitment = arrays_for(primary, field="v3_commitment_valid")
    _item_ids, evaluable = arrays_for(primary, field="v3_semantic_evaluable")
    v3_errors = {name: 1 - value for name, value in v3_correct.items()}
    v2_errors = {name: 1 - value for name, value in v2_correct.items()}
    estimands = {
        condition: audit_two_rollout_estimands(v3_errors[BASELINE], v3_errors[condition])
        for condition in CONDITIONS[1:]
    }
    random_summary = random_metric_summary(estimands, RANDOMS)
    classification, guards = classify_semantic_v3(
        baseline_summary=summaries[BASELINE],
        controller_summary=summaries[MEANINGFUL],
        controller_estimands=estimands[MEANINGFUL],
        random_summary=random_summary,
    )
    estimands[BASELINE] = {
        "B00": float(np.mean(v3_errors[BASELINE][:, 0] * v3_errors[BASELINE][:, 1])),
        "O00": float(1 - np.mean(v3_errors[BASELINE][:, 0] * v3_errors[BASELINE][:, 1])),
        "accuracy": summaries[BASELINE]["accuracy"],
        "commitment_validity": summaries[BASELINE]["commitment_validity"],
        "semantic_evaluability": summaries[BASELINE]["semantic_evaluability"],
    }
    meaningful_contrasts = {
        metric: {
            "minus_random_mean": estimands[MEANINGFUL][metric]
            - random_summary[metric]["mean"],
            "minus_random_max": estimands[MEANINGFUL][metric]
            - random_summary[metric]["max"],
        }
        for metric in ("G", "C", "D")
    }
    bootstrap = bootstrap_intervals(v3_errors, commitment, evaluable)
    loo = leave_one_out(item_ids, v3_errors, commitment)

    changed = [row for row in scored if row["reason_for_change"] != "unchanged"]
    primary_changed = [row for row in changed if row["phase"] == "GATE6_3_PRIMARY_EVALUATION"]
    contributions = contribution_rows(primary_changed, item_ids, v2_errors, v3_errors)
    matched_summary = {
        "rows": len(matched),
        "rows_reclassified": sum(row["reason_for_change"] != "unchanged" for row in matched),
        "commitment_valid": sum(bool(row["v3_commitment_valid"]) for row in matched),
        "semantic_evaluable": sum(bool(row["v3_semantic_evaluable"]) for row in matched),
        "correct": sum(bool(row["v3_correct"]) for row in matched),
    }
    result = {
        "schema_version": 1,
        "parser_version": PARSER_VERSION,
        "historical_classification": HISTORICAL_CLASSIFICATION,
        "historical_result_modified": False,
        "diagnostic_classification": classification,
        "lock_verification": lock_audit["checks"],
        "v2_crosscheck": crosscheck,
        "summaries": summaries,
        "estimands": estimands,
        "random_summary": random_summary,
        "meaningful_random_contrasts": meaningful_contrasts,
        "guards": guards,
        "matched_consistency": matched_summary,
        "rows_reclassified": len(changed),
        "primary_rows_reclassified": len(primary_changed),
        "reason_counts": dict(sorted(Counter(row["reason_for_change"] for row in changed).items())),
        "model_inference": False,
        "gpu_cost_usd": 0.0,
    }

    row_fields = list(scored[0])
    write_csv(output / "ROW_REANALYSIS_V3.csv", scored, row_fields)
    write_csv(
        output / "CONDITION_SUMMARY_V3.csv",
        [summaries[name] for name in CONDITIONS],
        list(summaries[BASELINE]),
    )
    write_json(output / "ESTIMANDS_V3.json", result)
    write_json(output / "BOOTSTRAP_INTERVALS_V3.json", bootstrap)
    disagreement_fields = list(primary_changed[0]) if primary_changed else row_fields
    write_csv(output / "V2_V3_DISAGREEMENTS.csv", primary_changed, disagreement_fields)
    write_csv(
        output / "ROW_CONTRIBUTIONS.csv",
        contributions,
        list(contributions[0]) if contributions else ["item_id"],
    )
    write_csv(output / "LOO_SENSITIVITY.csv", loo, list(loo[0]))

    draft_name, draft_body = next_protocol(classification)
    (output / "NEXT_PROTOCOL_DRAFT.md").write_text(
        f"<!-- {draft_name} -->\n\n{draft_body}", encoding="utf-8"
    )
    report = [
        "# Gate 6.3 Semantic-Validity Audit",
        "",
        f"Historical frozen result: `{HISTORICAL_CLASSIFICATION}` (unchanged).",
        f"Offline V3 diagnostic: `{classification}`.",
        "",
        "No model inference, RunPod access, Q2, character count, or holdout access occurred.",
        "",
        "## Condition-symmetric V3 summary",
        "",
        "| Condition | V2 validity | Commitment validity | Semantic evaluability | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        summary = summaries[condition]
        report.append(
            f"| `{condition}` | {summary['v2_historical_validity']:.3f} | "
            f"{summary['commitment_validity']:.3f} | "
            f"{summary['semantic_evaluability']:.3f} | {summary['accuracy']:.3f} |"
        )
    single = estimands[MEANINGFUL]
    report.extend(
        [
            "",
            "## Primary audit estimands",
            "",
            f"- Meaningful `G`: {single['G']:.6f}",
            f"- Meaningful `C`: {single['C']:.6f}",
            f"- Meaningful `D`: {single['D']:.6f}",
            f"- Random mean/max `G`: {random_summary['G']['mean']:.6f} / "
            f"{random_summary['G']['max']:.6f}",
            f"- Random mean/max `C`: {random_summary['C']['mean']:.6f} / "
            f"{random_summary['C']['max']:.6f}",
            f"- Random mean/max `D`: {random_summary['D']['mean']:.6f} / "
            f"{random_summary['D']['max']:.6f}",
            "",
            "## Reclassification audit",
            "",
            f"- All rows reclassified or re-extracted: {len(changed)} / 920",
            f"- Primary rows affected: {len(primary_changed)} / 840",
            f"- Matched consistency rows affected: {matched_summary['rows_reclassified']} / 80",
            f"- V2 historical crosscheck failures: "
            f"{crosscheck['historical_v2_crosscheck_failures']}",
            "",
            "Every condition was processed by the same frozen parser hash. The historical V2 "
            "classification remains the registered result; V3 is an additive offline diagnostic.",
            "",
            "## Next protocol",
            "",
            f"Drafted `{draft_name}` only. It was not executed.",
        ]
    )
    (output / "AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    tracked_outputs = (
        "AUDIT_LOCK.md",
        "AUDIT_LOCK.json",
        "SEMANTIC_V3_SPEC.md",
        "SEMANTIC_V3_TESTS.json",
        "BLINDED_CORPUS.jsonl",
        "BLINDED_CORPUS_PROVENANCE.json",
        "ROW_REANALYSIS_V3.csv",
        "CONDITION_SUMMARY_V3.csv",
        "ESTIMANDS_V3.json",
        "BOOTSTRAP_INTERVALS_V3.json",
        "V2_V3_DISAGREEMENTS.csv",
        "ROW_CONTRIBUTIONS.csv",
        "LOO_SENSITIVITY.csv",
        "AUDIT_REPORT.md",
        "NEXT_PROTOCOL_DRAFT.md",
    )
    write_json(
        output / "artifact_hashes.json",
        {name: digest(output / name) for name in tracked_outputs},
    )
    print(
        json.dumps(
            {
                "historical_classification": HISTORICAL_CLASSIFICATION,
                "diagnostic_classification": classification,
                "rows_reclassified": len(changed),
                "primary_rows_reclassified": len(primary_changed),
                "next_protocol": draft_name,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
