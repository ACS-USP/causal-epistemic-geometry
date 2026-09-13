"""Focused tests for the independent Q1 prompt semantic auditor."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import pytest

from epistemic_geometry.benchmarks.reasoning.families import generate_item
from epistemic_geometry.benchmarks.reasoning.rendering import render_reasoning
from epistemic_geometry.benchmarks.reasoning.semantic_audit import (
    audit_reasoning_view,
    validate_reasoning_view,
)


def _with_prompt(view, prompt: str):
    return replace(view, prompt=prompt, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest())


@pytest.mark.parametrize(
    ("family", "cell"),
    [
        ("MODREG-R", "depth_4"),
        ("FSM-R", "length_8"),
        ("SATCOUNT-R", "vars5_clauses8"),
    ],
)
@pytest.mark.parametrize("surface", ["canonical", "surface_twin"])
def test_auditor_parses_both_surfaces_and_recomputes_answer(family, cell, surface) -> None:
    item = generate_item(family, cell, seed=19)
    view = render_reasoning(item, surface=surface)
    report = audit_reasoning_view(item, view)

    assert report.valid
    assert report.parsed_spec == item.spec
    assert report.recomputed_answer == item.answer
    assert validate_reasoning_view(item, view) == report


def test_auditor_rejects_mutated_modreg_operation_and_query() -> None:
    item = generate_item("MODREG-R", "depth_4", seed=3)
    view = render_reasoning(item)
    lines = view.prompt.splitlines()
    operation_index = next(index for index, line in enumerate(lines) if line.startswith("1. "))
    operation_mutation = lines.copy()
    operation_mutation[operation_index] = operation_mutation[operation_index].replace(
        "mod 10", "mod 9"
    )
    query_mutation_lines = view.prompt.splitlines()
    query_line_index = next(
        index for index, line in enumerate(query_mutation_lines) if line.startswith("What is")
    )
    query_mutation_lines[query_line_index] = re.sub(
        r"R[0-3](?=\?)",
        f"R{(int(item.spec['query'][1]) + 1) % 4}",
        query_mutation_lines[query_line_index],
    )
    query_mutation = "\n".join(query_mutation_lines)

    assert not audit_reasoning_view(item, _with_prompt(view, "\n".join(operation_mutation))).valid
    assert not audit_reasoning_view(item, _with_prompt(view, query_mutation)).valid
    with pytest.raises(ValueError, match="semantic audit failed"):
        validate_reasoning_view(item, _with_prompt(view, "\n".join(operation_mutation)))


def test_auditor_rejects_mutated_fsm_transition_and_sequence() -> None:
    item = generate_item("FSM-R", "length_4", seed=7)
    view = render_reasoning(item, surface="surface_twin")
    lines = view.prompt.splitlines()
    transition_mutation = lines.copy()
    fields = transition_mutation[3].split(" | ")
    fields[1] = str((int(fields[1]) + 1) % 10)
    transition_mutation[3] = " | ".join(fields)
    sequence_mutation = view.prompt.replace(
        "Apply this symbol sequence in order: ",
        "Apply this symbol sequence in reverse order: ",
    )

    assert not audit_reasoning_view(item, _with_prompt(view, "\n".join(transition_mutation))).valid
    assert not audit_reasoning_view(item, _with_prompt(view, sequence_mutation)).valid


def test_auditor_rejects_mutated_satcount_clause_and_question() -> None:
    item = generate_item("SATCOUNT-R", "vars4_clauses6", seed=11)
    view = render_reasoning(item, surface="surface_twin")
    clause_mutation = view.prompt.replace("x1", "x2", 1)
    question_mutation = view.prompt.replace(
        "How many assignments satisfy the formula?", "How many assignments satisfy one clause?"
    )

    assert not audit_reasoning_view(item, _with_prompt(view, clause_mutation)).valid
    assert not audit_reasoning_view(item, _with_prompt(view, question_mutation)).valid
