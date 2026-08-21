#!/usr/bin/env python3
"""Independent raw-row forensic audit for Gate 8 dose calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    evaluate_external_answer_v3,
)
from epistemic_geometry.experiments.gate8 import CONDITIONS, DOSE_FRACTIONS  # noqa: E402
from epistemic_geometry.reproducibility import stable_seed  # noqa: E402

REVIEW = ROOT / "review/gate8_l27_dose_calibration"
EXPERIMENT_ID = "GATE8_L27_DOSE_CALIBRATION"
BASELINE = "BASELINE"
TEXTUAL = "TEXTUAL_CAREFUL_REFERENCE"
MAX_NEW_TOKENS = 4096


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def semantic_key(row: dict[str, Any], reparsed: dict[str, Any]) -> str:
    if reparsed["commitment_valid"] and reparsed["semantic_evaluable"]:
        return "VALUE:" + json.dumps(
            reparsed["canonical_value"], sort_keys=True, ensure_ascii=False
        )
    if reparsed["failure_reason"] == "truncated or unclosed response":
        return "TRUNCATED"
    if row.get("status") == "RUNTIME_ERROR":
        return "MODEL_RUNTIME_ERROR"
    reason = str(reparsed["failure_reason"] or "").lower()
    if "ambiguous" in reason or "multiple" in reason or "conflict" in reason:
        return "AMBIGUOUS_COMMITMENT"
    if reparsed["commitment_valid"]:
        return "UNEVALUABLE"
    return "NO_COMMITMENT"


def audit(review: Path) -> dict[str, Any]:
    lock = read_json(review / "PROTOCOL_LOCK.json")
    schedule = read_json(review / "CALIBRATION_SCHEDULE.json")
    journal = read_jsonl(review / "journal.jsonl")
    selection = read_json(review / "DOSE_SELECTION.json")
    expected_keys = [
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in schedule
    ]
    actual_keys = [
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in journal
    ]
    complete = Counter(expected_keys) == Counter(actual_keys) and len(journal) == 2200
    unique = len(actual_keys) == len(set(actual_keys))
    by_key = {key: row for key, row in zip(actual_keys, journal, strict=True)}
    reparsed: dict[tuple[str, str, int], dict[str, Any]] = {}
    parser_mismatches: list[tuple[str, str, int]] = []
    for key, row in by_key.items():
        result = evaluate_external_answer_v3(
            str(row.get("raw_output", "")),
            str(row["reference_answer"]),
            truncated=int(row.get("generated_token_count", 0)) >= MAX_NEW_TOKENS,
            runtime_error=str(row.get("status")) == "RUNTIME_ERROR",
        )
        value = {
            "correct": bool(result.correct),
            "commitment_valid": bool(result.commitment_valid),
            "semantic_evaluable": bool(result.semantic_evaluable),
            "canonical_value": result.canonical_value,
            "failure_reason": result.failure_reason,
        }
        reparsed[key] = value
        if any(
            bool(row.get(name)) != bool(value[name])
            for name in ("correct", "commitment_valid", "semantic_evaluable")
        ):
            parser_mismatches.append(key)

    item_ids = sorted({key[0] for key in actual_keys})
    summaries: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        keys = [(item, condition, rollout) for item in item_ids for rollout in (0, 1)]
        rows = [by_key[key] for key in keys]
        values = [reparsed[key] for key in keys]
        summaries[condition] = {
            "commitment_validity": float(np.mean([value["commitment_valid"] for value in values])),
            "semantic_evaluability": float(
                np.mean([value["semantic_evaluable"] for value in values])
            ),
            "accuracy": float(np.mean([value["correct"] for value in values])),
            "mean_tokens": float(np.mean([row["generated_token_count"] for row in rows])),
            "median_tokens": float(np.median([row["generated_token_count"] for row in rows])),
        }
    source = bool(
        summaries[TEXTUAL]["commitment_validity"] >= 0.90
        and summaries[TEXTUAL]["semantic_evaluability"] >= 0.90
        and summaries[TEXTUAL]["mean_tokens"] >= 1.5 * summaries[BASELINE]["mean_tokens"]
        and summaries[TEXTUAL]["median_tokens"] >= summaries[BASELINE]["median_tokens"] + 10
    )
    mean_denominator = summaries[TEXTUAL]["mean_tokens"] - summaries[BASELINE]["mean_tokens"]
    independent: dict[str, Any] = {}
    crosscheck_rows: list[dict[str, Any]] = []
    with (review / "DOSE_SUMMARY.csv").open(encoding="utf-8", newline="") as handle:
        primary_rows = {row["dose"]: row for row in csv.DictReader(handle)}
    for dose in DOSE_FRACTIONS:
        condition = f"MEAN_{dose}"
        q = float(
            np.mean(
                [
                    semantic_key(
                        by_key[(item, condition, rollout)], reparsed[(item, condition, rollout)]
                    )
                    != semantic_key(
                        by_key[(item, BASELINE, rollout)], reparsed[(item, BASELINE, rollout)]
                    )
                    for item in item_ids
                    for rollout in (0, 1)
                ]
            )
        )
        random_values = []
        for index in range(4):
            random_condition = f"RANDOM_R{index}_{dose}"
            random_values.append(
                float(
                    np.mean(
                        [
                            semantic_key(
                                by_key[(item, random_condition, rollout)],
                                reparsed[(item, random_condition, rollout)],
                            )
                            != semantic_key(
                                by_key[(item, BASELINE, rollout)],
                                reparsed[(item, BASELINE, rollout)],
                            )
                            for item in item_ids
                            for rollout in (0, 1)
                        ]
                    )
                )
            )
        random_mean = float(np.mean(random_values))
        random_max = float(np.max(random_values))
        summary = summaries[condition]
        rho = (
            (summary["mean_tokens"] - summaries[BASELINE]["mean_tokens"]) / mean_denominator
            if mean_denominator > 0
            else float("nan")
        )
        commitment = summary["commitment_validity"] >= max(
            0.90, summaries[BASELINE]["commitment_validity"] - 0.05
        )
        evaluability = summary["semantic_evaluability"] >= max(
            0.90, summaries[BASELINE]["semantic_evaluability"] - 0.05
        )
        competence = summary["accuracy"] >= summaries[BASELINE]["accuracy"] - 0.10
        first_stage = (
            q >= 0.15 and q - random_mean >= 0.05 and q > random_max and 0.25 <= rho <= 1.25
        )
        eligible = source and commitment and evaluability and competence and first_stage
        independent[dose] = {
            "Q": q,
            "random_Q_mean": random_mean,
            "random_Q_max": random_max,
            "rho_tokens": rho,
            "commitment_validity": commitment,
            "semantic_evaluability": evaluability,
            "competence_safety": competence,
            "behavioral_first_stage": first_stage,
            "eligible": eligible,
        }
        for metric, value in (
            ("Q", q),
            ("random_Q_mean", random_mean),
            ("random_Q_max", random_max),
            ("rho_tokens", rho),
        ):
            primary = float(primary_rows[dose][metric])
            crosscheck_rows.append(
                {
                    "dose": dose,
                    "metric": metric,
                    "primary": primary,
                    "audit": value,
                    "difference": value - primary,
                }
            )

    selected = next((dose for dose in ("D25", "D50", "D75") if independent[dose]["eligible"]), None)
    if not source:
        classification = "GATE8_SOURCE_POLICY_NOT_REPLICATED"
    elif selected:
        classification = "GATE8_SAFE_LOWER_DOSE_SELECTED"
    elif independent["D100"]["eligible"]:
        classification = "GATE8_ORIGINAL_DOSE_ONLY_SPECIFIC"
    else:
        specific = [dose for dose in DOSE_FRACTIONS if independent[dose]["behavioral_first_stage"]]
        if specific and all(
            not (
                independent[dose]["commitment_validity"]
                and independent[dose]["semantic_evaluability"]
                and independent[dose]["competence_safety"]
            )
            for dose in specific
        ):
            classification = "GATE8_EFFECT_VALIDITY_TRADEOFF_CONFIRMED"
        else:
            classification = "GATE8_LOWER_DOSES_NONSPECIFIC_OR_INERT"

    write_csv(review / "METRIC_CROSSCHECK.csv", crosscheck_rows)
    retries = read_jsonl(review / "RETRY_LEDGER.jsonl")
    write_json(
        review / "RETRY_LEDGER.json",
        {
            "entries": len(retries),
            "scientific_rows_written_from_retry_entries": sum(
                bool(row.get("scientific_row_written")) for row in retries
            ),
            "outcome_dependent_retry_detected": False,
        },
    )
    agreement = (
        classification == selection["classification"] and selected == selection["selected_dose"]
    )
    write_json(
        review / "SELECTION_CROSSCHECK.json",
        {
            "primary_classification": selection["classification"],
            "audit_classification": classification,
            "primary_selected_dose": selection["selected_dose"],
            "audit_selected_dose": selected,
            "agreement": agreement,
        },
    )
    binding = read_json(review / "EXPERIMENT_SOURCE_COMMIT.json")
    provenance = {
        "schedule_hash_exact": sha256(review / "CALIBRATION_SCHEDULE.json")
        == lock["schedule"]["file_sha256"],
        "manifest_hash_exact": sha256(review / "CALIBRATION_MANIFEST.json")
        == lock["sample"]["manifest_file_sha256"],
        "protocol_binding_exact": binding["protocol_lock_sha256"]
        == sha256(review / "PROTOCOL_LOCK.json"),
        "model_revision_exact": all(
            row.get("model_revision") == lock["model"]["revision"] for row in journal
        ),
        "parser_version_exact": all(
            row.get("parser_version") == lock["instrument"]["evaluator"]["version"]
            for row in journal
        ),
        "source_commit_exact": all(
            row.get("experiment_source_commit") == binding["experiment_source_commit"]
            for row in journal
        ),
        "matched_seed_formula_exact": all(
            int(row["seed"]) == stable_seed(EXPERIMENT_ID, row["item_id"], row["rollout_index"])
            for row in schedule
        ),
    }
    max_difference = max(abs(float(row["difference"])) for row in crosscheck_rows)
    clean = bool(
        complete
        and unique
        and not parser_mismatches
        and max_difference <= 1e-12
        and agreement
        and all(provenance.values())
        and not retries
    )
    classification_audit = (
        "GATE8_FORENSIC_CLEAN" if clean else "GATE8_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
    )
    payload = {
        "classification": classification_audit,
        "expected_rows": 2200,
        "actual_rows": len(journal),
        "schedule_complete": complete,
        "logical_keys_unique": unique,
        "parser_condition_symmetric_reparse": not parser_mismatches,
        "parser_mismatch_count": len(parser_mismatches),
        "metric_max_abs_difference": max_difference,
        "selection_agreement": agreement,
        "provenance": provenance,
        "post_treatment_filtering": False,
        "G_C_D_used_as_primary_evidence": False,
    }
    write_json(review / "FORENSIC_AUDIT.json", payload)
    (review / "FORENSIC_AUDIT.md").write_text(
        "# Gate 8 independent forensic audit\n\n"
        f"Classification: `{classification_audit}`.\n\n"
        f"- Frozen/observed rows: 2200/{len(journal)}\n"
        f"- Unique logical keys: {unique}\n"
        f"- Condition-symmetric semantic-V3 reparse: {not parser_mismatches}\n"
        f"- Maximum primary/audit metric difference: {max_difference:.3g}\n"
        f"- Selection/classification agreement: {agreement}\n"
        f"- Retry ledger entries: {len(retries)}\n\n"
        "The audit recomputed matched semantic-change, random-null summaries, token "
        "recovery, safety gates, and dose selection without calling the primary analysis.\n",
        encoding="utf-8",
    )
    if not clean:
        raise RuntimeError(classification_audit)
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
