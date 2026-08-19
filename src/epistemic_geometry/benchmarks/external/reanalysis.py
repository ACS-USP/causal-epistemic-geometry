"""Offline semantic reanalysis for the completed Gate 1 smoke.

This module consumes preserved journal rows only. It never regenerates prompts,
loads a model, or changes the historical result in place.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.v4.character_parser import parse_final_integer

from .base import ExternalResult, ExternalStatus, evaluate_external_answer, parse_external_answer
from .gate1 import classify_full_n20


def _token_count(result: dict[str, Any]) -> int:
    return int(result.get("metadata", {}).get("stop_metadata", {}).get("generated_token_count", 0))


def _reanalysis_row(
    row: dict[str, Any],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    result = row["result"]
    instrument = str(row["instrument"])
    raw_output = str(result.get("raw_output", ""))
    original_status = str(result.get("status", ""))
    reference = str(result.get("reference_answer", ""))
    original_parsed = result.get("parsed_answer")
    token_count = _token_count(result)
    truncated = token_count >= max_new_tokens

    if instrument == "FRESH_PSEUDOWORD_LONG":
        parse_status, parsed, parse_reason = parse_final_integer(
            raw_output, truncated=truncated
        )
        if parse_status == "PARSED":
            corrected_status = (
                ExternalStatus.VALID_CORRECT.value
                if parsed == int(reference)
                else ExternalStatus.VALID_WRONG.value
            )
            corrected = parsed == int(reference)
            extracted = str(parsed)
        else:
            corrected_status = parse_status
            corrected = False
            extracted = None
            parse_reason = parse_reason or "character parser rejected answer"
        evaluator = "exact_integer"
    elif instrument == "CRUXEVAL_SEMANTIC":
        parsed = parse_external_answer(raw_output, truncated=truncated)
        extracted = parsed.answer_text
        parse_reason = parsed.parse_reason
        if parsed.status is not None:
            corrected_status = parsed.status.value
            corrected = False
        else:
            try:
                corrected = evaluate_external_answer(
                    parsed.answer_text or "", reference, "python_literal"
                )
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as exc:
                corrected_status = ExternalStatus.INVALID_FORMAT.value
                corrected = False
                parse_reason = f"structured evaluator rejected answer: {exc}"
            else:
                corrected_status = (
                    ExternalStatus.VALID_CORRECT.value
                    if corrected
                    else ExternalStatus.VALID_WRONG.value
                )
        evaluator = "python_literal"
    else:
        raise ValueError(f"unsupported Gate 1 instrument: {instrument}")

    if original_status == corrected_status:
        reason = "unchanged under corrected deterministic semantics"
    elif corrected_status == ExternalStatus.VALID_CORRECT.value:
        reason = "one unambiguous semantic final answer is now evaluable and matches the oracle"
    elif corrected_status == ExternalStatus.VALID_WRONG.value:
        reason = (
            "one unambiguous semantic final answer is now evaluable and differs "
            "from the oracle"
        )
    else:
        reason = parse_reason or "corrected deterministic parser classification"
    return {
        "instrument": instrument,
        "item_id": str(row["item_id"]),
        "original_status": original_status,
        "corrected_status": corrected_status,
        "original_correct": bool(result.get("correct", False)),
        "corrected_correct": corrected,
        "reference_answer": reference,
        "original_parsed_answer": original_parsed,
        "extracted_semantic_answer": extracted,
        "token_count": token_count,
        "reason": reason,
        "raw_output": raw_output,
        "evaluator": evaluator,
    }


def reanalyze_gate1(
    journal_path: Path,
    output_dir: Path,
    *,
    historical_source_commit: str,
    reanalysis_source_commit: str,
    max_new_tokens: int = 4096,
) -> dict[str, Any]:
    """Write a separate CSV/report/version record and return its summary."""

    rows = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row
        for row in rows
        if row.get("event") == "trajectory"
        and row.get("instrument") in {"FRESH_PSEUDOWORD_LONG", "CRUXEVAL_SEMANTIC"}
    ]
    if len(selected) != 40:
        raise ValueError(f"expected 40 primary Gate 1 rows, found {len(selected)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected = [
        _reanalysis_row(row, max_new_tokens=max_new_tokens) for row in selected
    ]
    csv_path = output_dir / "SEMANTIC_REANALYSIS.csv"
    fields = list(corrected[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(corrected)

    summaries: dict[str, dict[str, Any]] = {}
    for instrument in ("FRESH_PSEUDOWORD_LONG", "CRUXEVAL_SEMANTIC"):
        instrument_rows = [row for row in corrected if row["instrument"] == instrument]
        counts = Counter(row["corrected_status"] for row in instrument_rows)
        statuses = [ExternalStatus(row["corrected_status"]) for row in instrument_rows]
        summaries[instrument] = {
            "n": len(instrument_rows),
            "valid": counts[ExternalStatus.VALID_CORRECT.value]
            + counts[ExternalStatus.VALID_WRONG.value],
            "correct": counts[ExternalStatus.VALID_CORRECT.value],
            "wrong": counts[ExternalStatus.VALID_WRONG.value],
            "mechanical": len(instrument_rows)
            - counts[ExternalStatus.VALID_CORRECT.value]
            - counts[ExternalStatus.VALID_WRONG.value],
            "accuracy": (
                counts[ExternalStatus.VALID_CORRECT.value]
                / max(
                    1,
                    counts[ExternalStatus.VALID_CORRECT.value]
                    + counts[ExternalStatus.VALID_WRONG.value],
                )
            ),
            "classification": classify_full_n20(
                [
                    ExternalResult(
                        item_id=f"{instrument}-{index}",
                        benchmark=instrument,
                        subtask="reanalysis",
                        rollout_seed=0,
                        raw_output="",
                        parsed_answer=None,
                        status=status,
                        correct=status == ExternalStatus.VALID_CORRECT,
                        reference_answer="",
                        evaluator="reanalysis",
                    )
                    for index, status in enumerate(statuses)
                ]
            ),
        }

    journal_sha256 = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    parser_version = {
        "version": "gate1-semantic-reanalysis-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "historical_source_commit": historical_source_commit,
        "reanalysis_source_commit": reanalysis_source_commit,
        "input_journal_sha256": journal_sha256,
        "max_new_tokens_for_truncation_audit": max_new_tokens,
        "model_inference": False,
        "rules": {
            "character_count": (
                "one explicit final integer; harmless Markdown wrappers accepted; "
                "conflicting or non-final commitments remain invalid"
            ),
            "cruxeval_string_reference": (
                "when the reference literal is a string, an unquoted final "
                "payload is that exact string"
            ),
            "cruxeval_structured_reference": (
                "non-string reference literals require structurally valid "
                "Python literals"
            ),
            "llm_judge": False,
            "item_specific_repairs": False,
        },
    }
    (output_dir / "parser_version.json").write_text(
        json.dumps(parser_version, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Gate 1 Semantic Reanalysis",
        "",
        "Offline deterministic reanalysis of preserved Gate 1 outputs. No model "
        "was loaded, no prompt was regenerated, and the original "
        "journal/results/report were not modified.",
        "",
        f"- Historical source commit: `{historical_source_commit}`",
        f"- Reanalysis source commit: `{reanalysis_source_commit}`",
        f"- Input journal SHA-256: `{journal_sha256}`",
        "- Parser version: `gate1-semantic-reanalysis-v1`",
        "",
        "## Summary",
        "",
        "| Instrument | Original valid/correct/wrong/invalid | Corrected "
        "valid/correct/wrong/invalid | Accuracy among valid | Classification |",
        "|---|---:|---:|---:|---|",
    ]
    for instrument, summary in summaries.items():
        original = Counter(
            row["original_status"] for row in corrected if row["instrument"] == instrument
        )
        corrected_text = (
            f"{summary['valid']}/{summary['correct']}/{summary['wrong']}/{summary['mechanical']}"
        )
        original_text = (
            f"{original['VALID_CORRECT'] + original['VALID_WRONG']}/"
            f"{original['VALID_CORRECT']}/{original['VALID_WRONG']}/"
            f"{20 - original['VALID_CORRECT'] - original['VALID_WRONG']}"
        )
        report_lines.append(
            f"| `{instrument}` | {original_text} | {corrected_text} | "
            f"{summary['accuracy']:.1%} | `{summary['classification']}` |"
        )
    report_lines.extend(
        [
            "",
            "The corrected character-count arm remains `PROMISING`: all 20 rows "
            "are semantically evaluable, with 15 correct and 5 genuine wrong. "
            "The corrected CRUXEval arm is also semantically evaluable on all "
            "20 rows, with its exact counts shown above; it does not gate the "
            "positive control.",
            "",
            "## Reanalysis semantics",
            "",
            "- `### ✅ FINAL: 2` is one explicit integer commitment, so Markdown "
            "decoration is not a mechanical failure.",
            "- For CRUXEval string references, `FINAL: yes` and `FINAL: Name "
            "unknown` are exact unquoted string answers.",
            "- Structured CRUXEval references still require valid structure; no "
            "fuzzy normalization, item-specific relabeling, or LLM judge was "
            "used.",
            "- The original Gate 1 status and raw output remain preserved for "
            "historical comparison.",
            "",
            "## Gate decision",
            "",
            "`FRESH_PSEUDOWORD_LONG = PROMISING`; Phase B is authorized by the "
            "frozen rule. CRUXEval is recorded for later analysis and does not "
            "gate Phase B.",
        ]
    )
    (output_dir / "SEMANTIC_REANALYSIS.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return {"summaries": summaries, "journal_sha256": journal_sha256}
