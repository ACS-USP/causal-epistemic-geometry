"""Independent, model-free semantic audit of Q1 reasoning prompts.

The parser in this module intentionally does not use the reasoning renderers or
the family oracle functions.  It reads the small, frozen textual grammar of
each Q1 family, maps surface-twin names back to their latent names, and runs a
small interpreter over the parsed representation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any

from .base import FINAL_ANSWER_INSTRUCTION, ReasoningItem, ReasoningView

_FAMILIES = {"MODREG-R", "FSM-R", "SATCOUNT-R"}
_SURFACES = {"canonical", "surface_twin"}
_REGISTER_RE = r"(?:R[0-3]|X[0-3])"
_SYMBOL_RE = r"[ABC]"
_LITERAL_RE = r"(?:x[1-9][0-9]*|NOT x[1-9][0-9]*)"


@dataclass(frozen=True)
class SemanticAudit:
    """Result of independently parsing and evaluating one rendered view."""

    valid: bool
    family: str
    surface: str
    recomputed_answer: int | None
    parsed_spec: dict[str, Any] | None
    errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.valid

    def __bool__(self) -> bool:
        return self.valid

    def to_record(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "family": self.family,
            "surface": self.surface,
            "recomputed_answer": self.recomputed_answer,
            "parsed_spec": self.parsed_spec,
            "errors": list(self.errors),
        }


class _ParseError(ValueError):
    pass


def _fail(message: str) -> None:
    raise _ParseError(message)


def _modreg_register_map(surface: str) -> tuple[dict[str, str], dict[str, str]]:
    if surface == "canonical":
        shown = {f"R{i}": f"R{i}" for i in range(4)}
    elif surface == "surface_twin":
        shown = {f"R{i}": f"X{(i + 1) % 4}" for i in range(4)}
    else:
        _fail(f"unsupported surface {surface!r}")
    return shown, {displayed: latent for latent, displayed in shown.items()}


def _parse_modreg(prompt: str, surface: str) -> dict[str, Any]:
    lines = prompt.split("\n")
    if len(lines) < 5:
        _fail("MODREG-R prompt is too short")
    shown, inverse = _modreg_register_map(surface)
    header = (
        "Registers R0, R1, R2, and R3 hold integers modulo 10."
        if surface == "canonical"
        else "Four registers X1, X2, X3, and X0 hold integers modulo 10."
    )
    if lines[0] != header or lines[2] != "Operations:":
        _fail("MODREG-R header or operations marker does not match")
    initial_match = re.fullmatch(
        rf"Initial values: ({_REGISTER_RE})=(-?\d+), ({_REGISTER_RE})=(-?\d+), "
        rf"({_REGISTER_RE})=(-?\d+), ({_REGISTER_RE})=(-?\d+)\.",
        lines[1],
    )
    if not initial_match:
        _fail("MODREG-R initial values are malformed")
    initial: list[int | None] = [None] * 4
    for name, value in zip(
        initial_match.group(1, 3, 5, 7), initial_match.group(2, 4, 6, 8), strict=True
    ):
        latent = inverse.get(name)
        if latent is None or initial[int(latent[1])] is not None:
            _fail("MODREG-R initial values contain an invalid or duplicate register")
        initial[int(latent[1])] = int(value)
    if any(value is None for value in initial):
        _fail("MODREG-R initial values omit a register")

    operation_lines = lines[3:-2]
    operations: list[dict[str, Any]] = []
    for expected_index, line in enumerate(operation_lines, start=1):
        numbered = re.fullmatch(r"(\d+)\. (.+)\.", line)
        if not numbered or int(numbered.group(1)) != expected_index:
            _fail("MODREG-R operation numbering is malformed")
        body = numbered.group(2)
        add_const = re.fullmatch(
            rf"(?P<r>{_REGISTER_RE}) <- \((?P=r) \+ (?P<c>\d+)\) mod 10", body
        )
        mul_unit = re.fullmatch(
            rf"(?P<r>{_REGISTER_RE}) <- \((?P<u>\d+) \* (?P=r)\) mod 10", body
        )
        reg_op = re.fullmatch(
            rf"(?P<dst>{_REGISTER_RE}) <- \((?P=dst) (?P<sign>[+-]) "
            rf"(?P<src>{_REGISTER_RE})\) mod 10",
            body,
        )
        swap = re.fullmatch(
            rf"swap (?P<r1>{_REGISTER_RE}) and (?P<r2>{_REGISTER_RE})", body
        )
        if add_const:
            operations.append(
                {"op": "ADD_CONST", "r": inverse[add_const["r"]], "c": int(add_const["c"])}
            )
        elif mul_unit:
            operations.append(
                {"op": "MUL_UNIT", "r": inverse[mul_unit["r"]], "u": int(mul_unit["u"])}
            )
        elif reg_op:
            operations.append(
                {
                    "op": "ADD_REG" if reg_op["sign"] == "+" else "SUB_REG",
                    "dst": inverse[reg_op["dst"]],
                    "src": inverse[reg_op["src"]],
                }
            )
        elif swap:
            operations.append(
                {"op": "SWAP", "r1": inverse[swap["r1"]], "r2": inverse[swap["r2"]]}
            )
        else:
            _fail(f"MODREG-R operation {expected_index} is malformed")
    if len(lines) < 5:
        _fail("MODREG-R query is missing")
    query_match = re.fullmatch(
        rf"What is the final value of (?P<query>{_REGISTER_RE})\?", lines[-2]
    )
    if not query_match or lines[-1] != FINAL_ANSWER_INSTRUCTION:
        _fail("MODREG-R query or final-answer instruction is malformed")
    return {
        "initial": [int(value) for value in initial],
        "operations": operations,
        "query": inverse[query_match["query"]],
    }


def _parse_fsm(prompt: str, surface: str) -> dict[str, Any]:
    lines = prompt.split("\n")
    if len(lines) != 17:
        _fail("FSM-R must contain exactly ten table rows and four trailer lines")
    intro = (
        "A deterministic machine has states 0 through 9 and symbols A, B, and C."
        if surface == "canonical"
        else "A deterministic machine has ten states numbered 0 through 9 and symbols C, A, and B."
    )
    shown_symbols = ("A", "B", "C") if surface == "canonical" else ("C", "A", "B")
    if lines[:2] != [intro, "For each row state and each symbol, the table gives the next state:"]:
        _fail("FSM-R introduction is malformed")
    if lines[2] != "state | " + " | ".join(shown_symbols):
        _fail("FSM-R table header does not match the surface")
    original_for_shown = {
        shown: original for original, shown in zip(("A", "B", "C"), shown_symbols, strict=True)
    }
    transitions: dict[str, list[int]] = {symbol: [0] * 10 for symbol in ("A", "B", "C")}
    expected_rows = list(range(10)) if surface == "canonical" else list(reversed(range(10)))
    for line, expected_state in zip(lines[3:13], expected_rows, strict=True):
        fields = line.split(" | ")
        if len(fields) != 4 or fields[0] != str(expected_state):
            _fail("FSM-R table rows are malformed or out of order")
        try:
            values = [int(value) for value in fields[1:]]
        except ValueError:
            _fail("FSM-R table contains a non-integer transition")
        if any(value not in range(10) for value in values):
            _fail("FSM-R transition leaves the state range")
        for shown, value in zip(shown_symbols, values, strict=True):
            transitions[original_for_shown[shown]][expected_state] = value

    start_match = re.fullmatch(r"Start at state (\d+)\.", lines[13])
    sequence_match = re.fullmatch(
        rf"Apply this symbol sequence in order: (?P<sequence>{_SYMBOL_RE}(?: {_SYMBOL_RE})*)\.",
        lines[14],
    )
    if not start_match or not sequence_match:
        _fail("FSM-R start or symbol sequence is malformed")
    if lines[15:] != ["What is the final state?", FINAL_ANSWER_INSTRUCTION]:
        _fail("FSM-R question or final-answer instruction is malformed")
    sequence = [original_for_shown[symbol] for symbol in sequence_match["sequence"].split()]
    return {"transitions": transitions, "start": int(start_match[1]), "sequence": sequence}


def _parse_satcount(prompt: str, surface: str) -> dict[str, Any]:
    lines = prompt.split("\n")
    if len(lines) != 5:
        _fail("SATCOUNT-R prompt must contain five lines")
    intro = (
        "Consider the Boolean formula below."
        if surface == "canonical"
        else "Consider the Boolean formula below; clauses and literals use an alternate order."
    )
    if (
        lines[0] != intro
        or lines[3] != "How many assignments satisfy the formula?"
        or lines[4] != FINAL_ANSWER_INSTRUCTION
    ):
        _fail("SATCOUNT-R introduction, question, or final-answer instruction is malformed")
    variables = re.fullmatch(r"It uses Boolean variables x1 through x(\d+)\.", lines[1])
    if not variables:
        _fail("SATCOUNT-R variable declaration is malformed")
    n_variables = int(variables[1])
    formula = lines[2]
    clause_pattern = rf"\(({_LITERAL_RE}(?: OR {_LITERAL_RE})*)\)"
    if not re.fullmatch(rf"{clause_pattern}(?: AND {clause_pattern})*\.", formula):
        _fail("SATCOUNT-R formula is malformed")
    clauses: list[list[int]] = []
    for clause_text in re.findall(r"\(([^()]*)\)", formula):
        clause: list[int] = []
        for literal_text in clause_text.split(" OR "):
            negated = literal_text.startswith("NOT ")
            number = int(literal_text[5 if negated else 1 :])
            if number > n_variables:
                _fail("SATCOUNT-R formula references an undeclared variable")
            clause.append(-number if negated else number)
        clauses.append(clause)
    if surface == "surface_twin":
        clauses = [list(reversed(clause)) for clause in reversed(clauses)]
    return {"n_variables": n_variables, "clauses": clauses}


def _execute_modreg(spec: dict[str, Any]) -> int:
    registers = {f"R{i}": int(value) % 10 for i, value in enumerate(spec["initial"])}
    for operation in spec["operations"]:
        kind = operation["op"]
        if kind == "ADD_CONST":
            register = operation["r"]
            registers[register] = (registers[register] + int(operation["c"])) % 10
        elif kind == "MUL_UNIT":
            register = operation["r"]
            registers[register] = (int(operation["u"]) * registers[register]) % 10
        elif kind in {"ADD_REG", "SUB_REG"}:
            destination, source = operation["dst"], operation["src"]
            delta = registers[source] if kind == "ADD_REG" else -registers[source]
            registers[destination] = (registers[destination] + delta) % 10
        elif kind == "SWAP":
            first, second = operation["r1"], operation["r2"]
            registers[first], registers[second] = registers[second], registers[first]
        else:
            _fail(f"unknown parsed MODREG-R operation {kind!r}")
    return registers[spec["query"]]


def _execute_fsm(spec: dict[str, Any]) -> int:
    state = int(spec["start"])
    for symbol in spec["sequence"]:
        state = int(spec["transitions"][symbol][state])
    return state


def _execute_satcount(spec: dict[str, Any]) -> int:
    n_variables = int(spec["n_variables"])
    if n_variables > 20:
        _fail("SATCOUNT-R variable count is too large for the bounded auditor")
    return sum(
        all(
            any(assignment[abs(literal) - 1] == (literal > 0) for literal in clause)
            for clause in spec["clauses"]
        )
        for assignment in product((False, True), repeat=n_variables)
    )


def audit_reasoning_view(item: ReasoningItem, view: ReasoningView) -> SemanticAudit:
    """Parse ``view.prompt`` independently and verify its latent semantics.

    Invalid prompts return a false-valued report with human-readable errors;
    this makes batch audits able to retain all failures.  Use
    :func:`validate_reasoning_view` when an exception is preferred.
    """

    errors: list[str] = []
    parsed: dict[str, Any] | None = None
    recomputed: int | None = None
    family, surface = str(view.family), str(view.surface)
    if family not in _FAMILIES:
        errors.append(f"unsupported family {family!r}")
    if surface not in _SURFACES:
        errors.append(f"unsupported surface {surface!r}")
    if view.latent_id != item.latent_id:
        errors.append("view latent_id does not match item")
    if view.family != item.family or view.cell != item.cell:
        errors.append("view family/cell does not match item")
    if view.answer != item.answer:
        errors.append("view answer does not match item")
    if not errors:
        try:
            parsed = {
                "MODREG-R": _parse_modreg,
                "FSM-R": _parse_fsm,
                "SATCOUNT-R": _parse_satcount,
            }[family](view.prompt, surface)
        except (KeyError, _ParseError, ValueError) as exc:
            errors.append(str(exc))
    if parsed is not None and parsed != item.spec:
        errors.append("parsed prompt specification does not match item specification")
    if parsed is not None:
        try:
            recomputed = {
                "MODREG-R": _execute_modreg,
                "FSM-R": _execute_fsm,
                "SATCOUNT-R": _execute_satcount,
            }[family](parsed)
        except (KeyError, _ParseError, ValueError, IndexError) as exc:
            errors.append(str(exc))
    if recomputed is not None and (recomputed != item.answer or recomputed != view.answer):
        errors.append(
            f"recomputed answer {recomputed} does not match stored answer {item.answer}"
        )
    return SemanticAudit(not errors, family, surface, recomputed, parsed, tuple(errors))


def validate_reasoning_view(item: ReasoningItem, view: ReasoningView) -> SemanticAudit:
    """Raise ``ValueError`` unless the independent semantic audit passes."""

    report = audit_reasoning_view(item, view)
    if not report.valid:
        raise ValueError("reasoning view semantic audit failed: " + "; ".join(report.errors))
    return report


def audit_reasoning_views(
    item: ReasoningItem, views: Iterable[ReasoningView]
) -> tuple[SemanticAudit, ...]:
    """Audit a collection of views, usually the canonical and surface twin."""

    return tuple(audit_reasoning_view(item, view) for view in views)
