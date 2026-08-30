#!/usr/bin/env python3
"""Post-hoc structural generation diagnostic for the sealed Q2 V4.1 campaign.

This utility never emits generated text, item identities, controller identities,
or correctness.  It is descriptive only and cannot alter the frozen Q2 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

MAX_NEW_TOKENS = 4096
MIN_REPETITION_TOKENS = 256
TAIL_WINDOW_TOKENS = 1024
MAX_PERIOD_TOKENS = 64
PERIODIC_MATCH_THRESHOLD = 0.90
DOMINANT_TOKEN_THRESHOLD = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[int], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def is_extreme_mechanical_repetition(token_ids: list[int]) -> bool:
    """Detect only conspicuous token-level repetition, without decoding text.

    A sequence is flagged when it has at least 256 generated tokens and either:

    * one token occupies at least 50% of the inspected tail; or
    * for some period from 1 through 64, at least 90% of comparable positions
      in the final at-most-1,024 tokens equal the token one period earlier.

    The thresholds are intentionally stringent.  This is a newly persisted
    post-hoc diagnostic definition, not an online stop rule or scoring rule.
    """

    if len(token_ids) < MIN_REPETITION_TOKENS:
        return False
    tail = token_ids[-TAIL_WINDOW_TOKENS:]
    dominant_share = max(Counter(tail).values()) / len(tail)
    if dominant_share >= DOMINANT_TOKEN_THRESHOLD:
        return True
    for period in range(1, min(MAX_PERIOD_TOKENS, len(tail) - 1) + 1):
        comparisons = len(tail) - period
        matches = sum(tail[index] == tail[index - period] for index in range(period, len(tail)))
        if comparisons and matches / comparisons >= PERIODIC_MATCH_THRESHOLD:
            return True
    return False


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            wrapper = json.loads(line)
            row = wrapper.get("row")
            if not isinstance(row, dict):
                raise RuntimeError(f"invalid journal row at line {line_number}")
            rows.append(row)
    return rows


def load_scores(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
            if key in scores:
                raise RuntimeError(f"duplicate score key at line {line_number}")
            scores[key] = row
    return scores


def shell_group(condition: str) -> str:
    if condition == "BASELINE":
        return "BASELINE"
    if condition.endswith("_MEDIUM"):
        return "MEDIUM"
    if condition.endswith("_STRONG"):
        return "STRONG"
    raise RuntimeError("unexpected frozen condition naming")


def relationship(flags: list[dict[str, bool]], name: str) -> dict[str, Any]:
    selected = [row for row in flags if row[name]]
    count = len(selected)
    return {
        "count": count,
        "fraction": count / len(flags),
        "commitment_valid_count": sum(row["commitment_valid"] for row in selected),
        "commitment_validity": (
            sum(row["commitment_valid"] for row in selected) / count if count else None
        ),
        "semantic_evaluable_count": sum(row["semantic_evaluable"] for row in selected),
        "semantic_evaluability": (
            sum(row["semantic_evaluable"] for row in selected) / count if count else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.journal)
    scores = load_scores(args.scores)
    if len(rows) != 37_800 or len(scores) != 37_800:
        raise RuntimeError("diagnostic requires the complete 37,800-row campaign")

    seen: set[tuple[str, str, int]] = set()
    lengths: list[int] = []
    flags: list[dict[str, Any]] = []
    shell_counts: dict[str, Counter[str]] = {
        "BASELINE": Counter(),
        "MEDIUM": Counter(),
        "STRONG": Counter(),
    }
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        if key in seen or key not in scores:
            raise RuntimeError("journal/score key mismatch")
        seen.add(key)
        score = scores[key]
        token_ids = [int(value) for value in row["generated_token_ids"]]
        token_count = int(row["generated_token_count"])
        if token_count != len(token_ids):
            raise RuntimeError("generated-token count does not match persisted token IDs")
        capped = token_count == MAX_NEW_TOKENS
        truncated = bool(row["truncated"])
        repeated = is_extreme_mechanical_repetition(token_ids)
        shell = shell_group(str(row["condition"]))
        shell_counts[shell]["rows"] += 1
        shell_counts[shell]["capped"] += capped
        shell_counts[shell]["truncated"] += truncated
        shell_counts[shell]["extreme_mechanical_repetition"] += repeated
        lengths.append(token_count)
        flags.append(
            {
                "capped": capped,
                "truncated": truncated,
                "extreme_mechanical_repetition": repeated,
                "capped_or_repetition": capped or repeated,
                "commitment_valid": bool(score["commitment_valid"]),
                "semantic_evaluable": bool(score["semantic_evaluable"]),
            }
        )

    if len(seen) != len(scores):
        raise RuntimeError("scores contain unexpected logical keys")

    by_shell: dict[str, dict[str, Any]] = {}
    for shell, counts in shell_counts.items():
        n = counts["rows"]
        by_shell[shell] = {
            "rows": n,
            "capped_count": counts["capped"],
            "capped_fraction": counts["capped"] / n,
            "truncated_count": counts["truncated"],
            "truncated_fraction": counts["truncated"] / n,
            "extreme_mechanical_repetition_count": counts["extreme_mechanical_repetition"],
            "extreme_mechanical_repetition_fraction": (counts["extreme_mechanical_repetition"] / n),
        }

    payload = {
        "schema_version": "q2-v4.1-posthoc-generation-diagnostic-v1",
        "label": "POST_HOC / DIAGNOSTIC",
        "scientific_classification_mutable": False,
        "frozen_relational_classification": "Q2_V4_1_G2",
        "frozen_radial_classifications": {"shape": "RS+", "total": "RT+"},
        "raw_text_printed_or_persisted": False,
        "item_or_controller_identities_printed_or_persisted": False,
        "journal": {"rows": len(rows), "sha256": sha256_file(args.journal)},
        "scores": {"rows": len(scores), "sha256": sha256_file(args.scores)},
        "lengths": {
            "p50": median(lengths),
            "p90": quantile(lengths, 0.90),
            "p95": quantile(lengths, 0.95),
            "p99": quantile(lengths, 0.99),
            "max": max(lengths),
        },
        "criterion": {
            "name": "EXTREME_MECHANICAL_REPETITION_V1",
            "minimum_generated_tokens": MIN_REPETITION_TOKENS,
            "tail_window_tokens": TAIL_WINDOW_TOKENS,
            "maximum_period_tokens": MAX_PERIOD_TOKENS,
            "periodic_match_threshold": PERIODIC_MATCH_THRESHOLD,
            "dominant_token_share_threshold": DOMINANT_TOKEN_THRESHOLD,
            "status": "NEWLY_PERSISTED_POST_HOC_DEFINITION_NOT_ONLINE_STOP_RULE",
        },
        "relationships": {
            name: relationship(flags, name)
            for name in (
                "capped",
                "truncated",
                "extreme_mechanical_repetition",
                "capped_or_repetition",
            )
        },
        "by_shell_descriptive": by_shell,
        "interpretation": (
            "Structural long-tail and repetition diagnostics are post-hoc and descriptive. "
            "Every terminal row remains included unchanged in the frozen analysis."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rel = payload["relationships"]
    lines = [
        "# Q2 V4.1 post-hoc generation diagnostic",
        "",
        "**POST_HOC / DIAGNOSTIC.** This report cannot alter `Q2_V4_1_G2`, `RS+`, or `RT+`.",
        "It prints no generated text, examples, item identities, or controller identities.",
        "",
        "## Structural summary",
        "",
        f"- Rows: {len(rows):,}",
        f"- Length p50/p90/p95/p99/max: {payload['lengths']['p50']:.1f} / "
        f"{payload['lengths']['p90']:.1f} / {payload['lengths']['p95']:.1f} / "
        f"{payload['lengths']['p99']:.1f} / {payload['lengths']['max']}",
        f"- Frozen-cap rows: {rel['capped']['count']:,} ({rel['capped']['fraction']:.3%})",
        f"- Recorded truncations: {rel['truncated']['count']:,} "
        f"({rel['truncated']['fraction']:.3%})",
        "- Extreme mechanical repetition under the newly persisted V1 criterion: "
        f"{rel['extreme_mechanical_repetition']['count']:,} "
        f"({rel['extreme_mechanical_repetition']['fraction']:.3%})",
        "",
        "## Relationship to answer-channel validity",
        "",
        "| Structural subset | n | Commitment validity | Semantic evaluability |",
        "|---|---:|---:|---:|",
    ]
    for name in ("capped", "truncated", "extreme_mechanical_repetition", "capped_or_repetition"):
        row = rel[name]
        lines.append(
            f"| `{name}` | {row['count']:,} | {row['commitment_validity']:.3%} | "
            f"{row['semantic_evaluability']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The repetition rule is a transparent post-hoc structural definition, not a "
            "reconstructed online criterion. No row was filtered, censored, regenerated, or "
            "reclassified. The frozen "
            "primary and forensic analyses include every terminal trajectory exactly as persisted.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
