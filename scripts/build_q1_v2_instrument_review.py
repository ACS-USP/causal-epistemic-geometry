#!/usr/bin/env python3
"""Build a self-contained CPU review bundle for E3-10 calibration.

This tool consumes only procedural manifests and stored baseline score vectors.
It never imports Torch, Transformers, datasets, or model weights.  It enriches
the raw rows with direct semantic observables, independently recomputes the
qualification decision, validates latent/view identity, and produces the
pre-registered calibration figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epistemic_geometry.benchmarks.e3.base import NUMBER_WORDS, LatentItem
from epistemic_geometry.benchmarks.e3.benchmark import views_for_item
from epistemic_geometry.benchmarks.e3.qualification import (
    MIN_ACCURACY,
    MIN_NORMALIZED_ENTROPY,
    MIN_SURFACE_AGREEMENT,
    MIN_WORD_AGREEMENT,
    CalibrationScoreRow,
    score_row,
    select_cells,
    summarize_cell,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _approximately_equal(left: Any, right: Any) -> bool:
    """Compare recomputation outputs without mistaking floating roundoff for drift."""

    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _approximately_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _approximately_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _load_expected_views(manifest_path: Path) -> tuple[dict[str, Any], dict[str, LatentItem]]:
    payload = _read_json(manifest_path)
    if payload.get("split") != "INSTRUMENT_CALIBRATION":
        raise ValueError("review builder accepts only INSTRUMENT_CALIBRATION")
    if payload.get("model_outcomes"):
        raise ValueError("calibration manifest must not contain model outcomes")
    views: dict[str, Any] = {}
    latents: dict[str, LatentItem] = {}
    for record in payload.get("manifests", {}).values():
        for item_record in record.get("items", []):
            item = LatentItem.from_record(item_record)
            if item.latent_id in latents:
                raise ValueError(f"duplicate latent ID in calibration manifest: {item.latent_id}")
            latents[item.latent_id] = item
            for view in views_for_item(item):
                if view.view_id in views:
                    raise ValueError(f"duplicate view ID in calibration manifest: {view.view_id}")
                views[view.view_id] = view
    if not views:
        raise ValueError("calibration manifest contains no views")
    return views, latents


def _load_rows(path: Path) -> tuple[list[CalibrationScoreRow], list[dict[str, Any]]]:
    rows: list[CalibrationScoreRow] = []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                rows.append(CalibrationScoreRow.from_record(record))
                records.append(record)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid score row at line {line_number}: {exc}") from exc
    return rows, records


def _load_tokenization_ids(path: Path) -> dict[str, dict[str, list[int]]]:
    """Read the JSON payload after the audit's human-readable host header."""

    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    if start < 0:
        raise ValueError("tokenization audit does not contain a JSON payload")
    payload = json.loads(text[start:])
    result: dict[str, dict[str, list[int]]] = {}
    for channel, report in payload.get("reports", {}).items():
        result[channel] = {
            label: list(candidate["context_compatible_token_ids"])
            for label, candidate in report.get("candidates", {}).items()
        }
    if set(result) != {"decimal", "number_word"}:
        raise ValueError("tokenization audit is missing a response channel")
    return result


def _labels(channel: str) -> list[str]:
    return [str(digit) for digit in range(10)] if channel == "decimal" else list(NUMBER_WORDS)


def _enrich_rows(
    rows: list[CalibrationScoreRow],
    records: list[dict[str, Any]],
    views: dict[str, Any],
    latents: dict[str, LatentItem],
    tokenization_ids: dict[str, dict[str, list[int]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(records):
        raise ValueError("score row and raw record counts differ")
    enriched: list[dict[str, Any]] = []
    seen_views: set[str] = set()
    errors: list[str] = []
    for row, raw in zip(rows, records, strict=True):
        view_id = str(row.metadata.get("view_id", ""))
        view = views.get(view_id)
        if view is None:
            errors.append(f"unknown view_id {view_id!r}")
            continue
        if view_id in seen_views:
            errors.append(f"duplicate scientific view row {view_id}")
        seen_views.add(view_id)
        latent = latents[row.latent_id]
        if (row.family, row.cell, row.target) != (view.family, view.cell, view.target):
            errors.append(f"identity mismatch for {view_id}")
        source_prompt_hash = str(row.metadata.get("source_prompt_hash", ""))
        if source_prompt_hash != view.prompt_hash:
            errors.append(f"source prompt hash mismatch for {view_id}")
        labels = _labels(row.response_channel)
        token_ids = raw.get("metadata", {}).get("candidate_token_ids", {})
        if not token_ids:
            token_ids = tokenization_ids[row.response_channel]
        if set(token_ids) != set(labels):
            errors.append(f"candidate token metadata mismatch for {view_id}")
        result = score_row(row)
        enriched_row = row.to_record()
        enriched_row.update(
            {
                "view_id": view_id,
                "latent_hash": latent.latent_hash,
                "latent_seed": latent.latent_seed,
                "difficulty": latent.difficulty,
                "source_prompt": view.prompt,
                "source_prompt_hash": source_prompt_hash,
                "rendered_prompt_hash": row.prompt_hash,
                "target_text": view.target_text,
                "candidate_labels": labels,
                "candidate_token_ids": token_ids,
                "candidate_token_counts": {label: len(token_ids[label]) for label in labels},
                "semantic_option_ids": list(range(10)),
                "candidate_logits": list(row.scores),
                "semantic_probabilities": result["probabilities"],
                "prediction": result["prediction"],
                "correct": result["correct"],
                "top1_score": result["top1_score"],
                "top2_score": result["top2_score"],
                "margin": result["margin"],
                "true_answer_logit": result["true_answer_logit"],
                "best_wrong_logit": result["best_wrong_logit"],
                "true_answer_margin": result["true_answer_margin"],
                "nll": result["nll"],
                "brier": result["brier"],
                "normalized_entropy": result["normalized_entropy"],
                "scientific_key": f"{row.latent_id}|{view_id}|baseline",
            }
        )
        enriched.append(enriched_row)
    expected = set(views)
    missing = sorted(expected - seen_views)
    if missing:
        errors.append(f"missing {len(missing)} expected view rows")
    return enriched, {"ok": not errors, "errors": errors, "expected_views": len(expected)}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _summary_rows(enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[CalibrationScoreRow]] = defaultdict(list)
    for record in enriched:
        grouped[(record["family"], record["cell"])].append(CalibrationScoreRow.from_record(record))
    summaries = [summarize_cell(rows) for rows in grouped.values()]
    return sorted(summaries, key=lambda value: (value["family"], value["cell"]))


def _difficulty(enriched: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for record in enriched:
        key = (record["family"], record["cell"])
        result.setdefault(key, json.dumps(record["difficulty"], sort_keys=True))
    return result


def _write_table(
    path: Path, summaries: list[dict[str, Any]], difficulty: dict[tuple[str, str], str]
) -> None:
    fields = [
        "family",
        "cell",
        "structural_difficulty",
        "accuracy",
        "nll",
        "brier",
        "median_margin",
        "normalized_prediction_entropy",
        "decimal_word_agreement",
        "surface_twin_agreement",
        "qualified",
        "selected",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    **{field: summary.get(field, "") for field in fields},
                    "structural_difficulty": difficulty[(summary["family"], summary["cell"])],
                    "qualified": summary["qualification"]["qualified"],
                    "selected": summary.get("selected", False),
                }
            )


def _plot_lines(
    summaries: list[dict[str, Any]],
    value_key: str,
    ylabel: str,
    filename: Path,
    threshold: float | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    families = sorted({summary["family"] for summary in summaries})
    for family in families:
        family_rows = [summary for summary in summaries if summary["family"] == family]
        axis.plot(
            range(len(family_rows)),
            [summary[value_key] for summary in family_rows],
            marker="o",
            label=family,
        )
    if threshold is not None:
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")
    axis.set_xticks(range(4), ["cell 1", "cell 2", "cell 3", "cell 4"])
    axis.set_ylabel(ylabel)
    axis.set_xlabel("difficulty-cell order within family")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def _plot_selection(summaries: list[dict[str, Any]], filename: Path) -> None:
    labels = [f"{s['family']}\n{s['cell']}" for s in summaries]
    values = [summary["accuracy"] for summary in summaries]
    colors = [
        "#2ca02c" if summary["qualification"]["qualified"] else "#d62728" for summary in summaries
    ]
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.bar(range(len(labels)), values, color=colors)
    axis.axhspan(MIN_ACCURACY, 0.75, color="#2ca02c", alpha=0.08)
    axis.axhline(MIN_ACCURACY, color="black", linestyle="--", linewidth=1)
    axis.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    axis.axhline(0.75, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    axis.set_ylabel("canonical decimal accuracy")
    axis.set_title("Qualification region; green = qualified, red = failed")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def _plot_distributions(
    enriched: list[dict[str, Any]], summaries: list[dict[str, Any]], filename: Path
) -> None:
    selected = [summary for summary in summaries if summary.get("selected")]
    families = [summary["family"] for summary in selected]
    if not selected:
        families = sorted({summary["family"] for summary in summaries})
        selected = [
            next(summary for summary in summaries if summary["family"] == family)
            for family in families
        ]
        title = "No selected cells; first cell shown for each family (suite failed)"
    else:
        title = "Predicted-digit distributions for selected cells"
    fig, axes = plt.subplots(
        max(1, len(selected)), 1, figsize=(10, 3.2 * max(1, len(selected))), squeeze=False
    )
    for axis, summary in zip(axes[:, 0], selected, strict=True):
        predictions = [
            record["prediction"]
            for record in enriched
            if record["family"] == summary["family"]
            and record["cell"] == summary["cell"]
            and record["surface"] == "canonical"
            and record["response_channel"] == "decimal"
        ]
        counts = np.bincount(predictions, minlength=10) / len(predictions)
        axis.bar(range(10), counts, alpha=0.8, label="predicted")
        axis.axhline(0.1, color="black", linestyle="--", label="uniform target reference")
        axis.set_ylim(0, 1)
        axis.set_ylabel(summary["family"])
        axis.set_xticks(range(10))
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[-1, 0].set_xlabel("semantic digit")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def _write_summary(
    path: Path,
    summaries: list[dict[str, Any]],
    validator: dict[str, Any],
    remote_hashes: dict[str, str],
) -> None:
    qualified = [summary for summary in summaries if summary["qualification"]["qualified"]]
    lines = [
        "# Q1 V2 — E3-10 Instrument Review",
        "",
        "## Status",
        "",
        "E3-10 calibration is **NOT QUALIFIED** under the frozen baseline-only rule.",
        "",
        "`E3_10_INSTRUMENT_NOT_QUALIFIED`",
        "",
        "This is an instrument result, not a steering result. No activation directions,",
        "steering conditions, DEV evaluation, or confirmatory holdout were accessed.",
        "",
        "## Qualification",
        "",
        f"- Calibration cells evaluated: {len(summaries)}",
        f"- Qualifying cells: {len(qualified)}",
        "- Required qualifying families: at least 2",
        "- Fresh scientific splits: NOT GENERATED because the suite failed qualification",
        "",
        "All calibration cells are retained in `instrument_review_table.csv`; failed cells",
        "are not hidden and were not manually retuned.",
        "",
        "## Frozen thresholds",
        "",
        f"- Accuracy: {MIN_ACCURACY:.2f} to 0.75",
        f"- Decimal/word agreement: >= {MIN_WORD_AGREEMENT:.2f}",
        f"- Surface-twin agreement: >= {MIN_SURFACE_AGREEMENT:.2f}",
        f"- Normalized prediction entropy: >= {MIN_NORMALIZED_ENTROPY:.2f}",
        "",
        "## Validation",
        "",
        f"- CPU validator: {'PASS' if validator['ok'] else 'FAIL'}",
        f"- Enriched rows: {validator['row_count']}",
        f"- Unique scientific view rows: {validator['unique_view_count']}",
        "",
        "## Remote artifact hashes",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in sorted(remote_hashes.items()))
    lines.extend(
        [
            "",
            "## Scientific firewall",
            "",
            "The Q1 V2 instrument is not qualified. Do not construct steering vectors",
            "or run DEV/holdout inference from this bundle. Principal review is required",
            "before any redesign or new calibration protocol.",
            "",
            "MOCK RESULTS and this calibration are software/instrument validation only;",
            "they do not establish a Q1 scientific result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tokenization-audit", type=Path, required=True)
    parser.add_argument("--remote-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    views, latents = _load_expected_views(args.manifest)
    rows, records = _load_rows(args.scores)
    tokenization_ids = _load_tokenization_ids(args.tokenization_audit)
    enriched, identity = _enrich_rows(rows, records, views, latents, tokenization_ids)
    if not identity["ok"]:
        raise ValueError(
            "calibration identity validation failed: " + "; ".join(identity["errors"][:5])
        )
    enriched_path = args.output / "baseline_score_rows_enriched.jsonl"
    _write_jsonl(enriched_path, enriched)

    summaries = _summary_rows(enriched)
    selected = select_cells(summaries)
    remote = _read_json(args.remote_qualification)
    remote_summaries = {(s["family"], s["cell"]): s for s in remote["summaries"]}
    recomputed_summaries = {(s["family"], s["cell"]): s for s in summaries}
    qualification_match = (
        _approximately_equal(remote_summaries, recomputed_summaries)
        and remote.get("selected", {}) == selected
    )

    # Validate exact calibration cardinalities and target balance.
    canonical = [
        record
        for record in enriched
        if record["surface"] == "canonical" and record["response_channel"] == "decimal"
    ]
    by_cell: Counter[tuple[str, str]] = Counter(
        (record["family"], record["cell"]) for record in enriched
    )
    target_counts: dict[str, dict[str, int]] = {}
    for record in canonical:
        target_counts.setdefault(f"{record['family']}/{record['cell']}", Counter())[
            str(record["target"])
        ] += 1
    balance_ok = all(sorted(counts.values()) == [20] * 10 for counts in target_counts.values())
    validator = {
        "ok": qualification_match and balance_ok and len(enriched) == len(views),
        "row_count": len(enriched),
        "expected_row_count": len(views),
        "unique_view_count": len({record["view_id"] for record in enriched}),
        "cell_counts": {
            f"{family}/{cell}": count for (family, cell), count in sorted(by_cell.items())
        },
        "target_balance_ok": balance_ok,
        "qualification_recomputation_matches_remote": qualification_match,
        "holdout_accessed": False,
        "model_outcomes_in_manifest": False,
        "errors": [] if qualification_match and balance_ok else ["validation mismatch"],
    }
    _write_json(args.output / "validator_report.json", validator)
    _write_json(
        args.output / "qualification_recomputed.json",
        {"summaries": summaries, "selected": selected},
    )
    _write_json(
        args.output / "instrument_selection.json",
        {
            "selected": selected,
            "qualifying_family_count": len(selected),
            "suite_qualified": len(selected) >= 2,
        },
    )
    _write_json(
        args.output / "fresh_splits_status.json",
        {
            "generated": False,
            "reason": "E3_10_INSTRUMENT_NOT_QUALIFIED",
            "geometry_calibration": 0,
            "dev_evaluation": 0,
            "confirmatory_holdout": 0,
        },
    )
    _write_json(
        args.output / "review_metadata.json",
        {
            "suite": "E3-10",
            "phase": "baseline-only instrument calibration",
            "steering_performed": False,
            "activation_extraction_performed": False,
            "dev_evaluation_accessed": False,
            "confirmatory_holdout_accessed": False,
            "scores_sha256": _sha256(args.scores),
            "manifest_sha256": _sha256(args.manifest),
            "remote_qualification_sha256": _sha256(args.remote_qualification),
        },
    )
    _write_table(args.output / "instrument_review_table.csv", summaries, _difficulty(enriched))
    _plot_lines(
        summaries,
        "accuracy",
        "canonical decimal accuracy",
        figures / "figure_1_accuracy_vs_difficulty.png",
        threshold=MIN_ACCURACY,
    )
    _plot_lines(
        summaries,
        "decimal_word_agreement",
        "decimal/word semantic agreement",
        figures / "figure_2_decimal_word_agreement.png",
        threshold=MIN_WORD_AGREEMENT,
    )
    _plot_lines(
        summaries,
        "surface_twin_agreement",
        "surface-twin semantic agreement",
        figures / "figure_3_surface_twin_agreement.png",
        threshold=MIN_SURFACE_AGREEMENT,
    )
    _plot_lines(
        summaries,
        "normalized_prediction_entropy",
        "normalized prediction entropy",
        figures / "figure_4_prediction_entropy.png",
        threshold=MIN_NORMALIZED_ENTROPY,
    )
    _plot_selection(summaries, figures / "figure_5_qualification_selection.png")
    _plot_distributions(enriched, summaries, figures / "figure_6_predicted_digit_distributions.png")

    remote_hashes = {
        "baseline_score_vectors.jsonl": _sha256(args.scores),
        "calibration_manifest.json": _sha256(args.manifest),
        "remote_qualification.json": _sha256(args.remote_qualification),
    }
    _write_summary(args.output / "summary.md", summaries, validator, remote_hashes)
    print(
        json.dumps(
            {
                "rows": len(enriched),
                "cells": len(summaries),
                "selected": sorted(selected),
                "suite_qualified": len(selected) >= 2,
                "validator": validator["ok"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
