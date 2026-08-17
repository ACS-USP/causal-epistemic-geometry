"""Model-free validation for the Q1 V3 reasoning suite."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from .base import ReasoningItem
from .families import FAMILY_CELLS, generate_item, oracle_for
from .rendering import render_reasoning
from .structural import shallow_shortcut_audit

ANSWER_COLLAPSE_WARNING = 0.25
ANSWER_COLLAPSE_FAILURE = 0.40
SHORTCUT_WARNING = 0.25
SHORTCUT_FAILURE = 0.40


def answer_distribution(items: list[ReasoningItem] | tuple[ReasoningItem, ...]) -> dict[str, Any]:
    counts = Counter(item.answer for item in items)
    total = len(items)
    probabilities = np.asarray([count / total for count in counts.values()], dtype=float)
    entropy = float(-(probabilities * np.log(probabilities)).sum()) if total else float("nan")
    modal_count = max(counts.values(), default=0)
    modal_frequency = modal_count / total if total else float("nan")
    return {
        "n_items": total,
        "support_size": len(counts),
        "counts": {str(answer): count for answer, count in sorted(counts.items())},
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(len(counts)) if len(counts) > 1 else 0.0,
        "modal_answer_frequency": modal_frequency,
        "top_5": [{"answer": answer, "count": count} for answer, count in counts.most_common(5)],
        "status": (
            "ANSWER_COLLAPSE_FAILURE"
            if modal_frequency >= ANSWER_COLLAPSE_FAILURE
            else "ANSWER_COLLAPSE_WARNING"
            if modal_frequency >= ANSWER_COLLAPSE_WARNING
            else "PASS"
        ),
    }


def validate_item(item: ReasoningItem) -> dict[str, Any]:
    if oracle_for(item) != item.answer:
        raise AssertionError(f"oracle mismatch for {item.latent_id}")
    if ReasoningItem.from_record(item.to_record()) != item:
        raise AssertionError(f"serialization changed {item.latent_id}")
    canonical = render_reasoning(item)
    twin = render_reasoning(item, surface="surface_twin")
    if canonical.answer != twin.answer or canonical.latent_id != twin.latent_id:
        raise AssertionError(f"surface-twin identity mismatch for {item.latent_id}")
    if canonical.prompt == twin.prompt:
        raise AssertionError(f"surface twin did not change presentation for {item.latent_id}")
    return {
        "latent_id": item.latent_id,
        "answer": item.answer,
        "surface_twin_oracle_equal": True,
        "roundtrip": True,
    }


def validate_family(
    family: str, cells: tuple[str, ...], *, n_per_cell: int = 5000
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for cell in cells:
        items = [generate_item(family, cell, seed) for seed in range(n_per_cell)]
        if len({item.latent_id for item in items}) != n_per_cell:
            raise AssertionError(f"duplicate latent IDs in {family}/{cell}")
        for item in items:
            validate_item(item)
        distribution = answer_distribution(items)
        shortcut = shallow_shortcut_audit(items)
        shortcut_accuracy = shortcut["depth_4_tree_accuracy_holdout"]
        shortcut["status"] = (
            "SHORTCUT_FAILURE"
            if shortcut_accuracy >= SHORTCUT_FAILURE
            else "SHORTCUT_WARNING"
            if shortcut_accuracy >= SHORTCUT_WARNING
            else "PASS"
        )
        reports[cell] = {
            "count": n_per_cell,
            "distribution": distribution,
            "shortcut_audit": shortcut,
            "structural_status": "PASS" if shortcut["status"] != "SHORTCUT_FAILURE" else "FAIL",
        }
    return reports


def validate_suite(*, n_per_cell: int = 5000) -> dict[str, Any]:
    reports = {
        family: validate_family(family, cells, n_per_cell=n_per_cell)
        for family, cells in FAMILY_CELLS.items()
    }
    failed = [
        f"{family}/{cell}"
        for family, cells in reports.items()
        for cell, report in cells.items()
        if report["distribution"]["status"] == "ANSWER_COLLAPSE_FAILURE"
    ]
    shortcut_failures = [
        f"{family}/{cell}"
        for family, cells in reports.items()
        for cell, report in cells.items()
        if report["shortcut_audit"]["status"] == "SHORTCUT_FAILURE"
    ]
    return {
        "suite_version": "Q1-V3-REASONING-v1",
        "generator_validation": "PASS",
        "cells": reports,
        "answer_collapse_failures": failed,
        "shortcut_failures": shortcut_failures,
        "status": "PASS" if not failed and not shortcut_failures else "FAIL",
    }
