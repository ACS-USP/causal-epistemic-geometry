"""Deterministic reasoning prompts and semantically equivalent surface twins."""

from __future__ import annotations

import hashlib
from typing import Any

from .base import FINAL_ANSWER_INSTRUCTION, ReasoningItem, ReasoningView


def _modreg(item: ReasoningItem, surface: str) -> tuple[str, str]:
    spec = item.spec
    if surface == "canonical":
        names = {name: name for name in ("R0", "R1", "R2", "R3")}
        header = "Registers R0, R1, R2, and R3 hold integers modulo 10."
        template = "MODREG-R/canonical/v1"
    else:
        names = {name: f"X{(int(name[1]) + 1) % 4}" for name in ("R0", "R1", "R2", "R3")}
        header = "Four registers X1, X2, X3, and X0 hold integers modulo 10."
        template = "MODREG-R/twin/v1"
    lines = [
        header,
        "Initial values: "
        + ", ".join(f"{names[f'R{i}']}={spec['initial'][i]}" for i in range(4))
        + ".",
        "Operations:",
    ]
    for index, operation in enumerate(spec["operations"], start=1):
        kind = operation["op"]
        if kind == "ADD_CONST":
            text = f"{names[operation['r']]} <- ({names[operation['r']]} + {operation['c']}) mod 10"
        elif kind == "MUL_UNIT":
            text = f"{names[operation['r']]} <- ({operation['u']} * {names[operation['r']]}) mod 10"
        elif kind in {"ADD_REG", "SUB_REG"}:
            symbol = "+" if kind == "ADD_REG" else "-"
            text = (
                f"{names[operation['dst']]} <- ({names[operation['dst']]} {symbol} "
                f"{names[operation['src']]}) mod 10"
            )
        else:
            text = f"swap {names[operation['r1']]} and {names[operation['r2']]}"
        lines.append(f"{index}. {text}.")
    lines.extend(
        [
            f"What is the final value of {names[spec['query']]}?",
            FINAL_ANSWER_INSTRUCTION,
        ]
    )
    return "\n".join(lines), template


def _fsm(item: ReasoningItem, surface: str) -> tuple[str, str]:
    spec = item.spec
    transitions = spec["transitions"]
    if surface == "canonical":
        symbols = ("A", "B", "C")
        rows = list(range(10))
        template = "FSM-R/canonical/v1"
        intro = "A deterministic machine has states 0 through 9 and symbols A, B, and C."
    else:
        symbols = ("C", "A", "B")
        rows = list(reversed(range(10)))
        template = "FSM-R/twin/v1"
        intro = (
            "A deterministic machine has ten states numbered 0 through 9 and symbols C, A, and B."
        )
    symbol_map = dict(zip(("A", "B", "C"), symbols, strict=True))
    lines = [
        intro,
        "For each row state and each symbol, the table gives the next state:",
        "state | " + " | ".join(symbols),
    ]
    inverse = {shown: original for original, shown in symbol_map.items()}
    for state in rows:
        lines.append(
            f"{state} | "
            + " | ".join(str(transitions[inverse[symbol]][state]) for symbol in symbols)
        )
    sequence = " ".join(symbol_map[symbol] for symbol in spec["sequence"])
    lines.extend(
        [
            f"Start at state {spec['start']}.",
            f"Apply this symbol sequence in order: {sequence}.",
            "What is the final state?",
            FINAL_ANSWER_INSTRUCTION,
        ]
    )
    return "\n".join(lines), template


def _satcount(item: ReasoningItem, surface: str) -> tuple[str, str]:
    spec = item.spec
    clauses = [[int(literal) for literal in clause] for clause in spec["clauses"]]
    if surface == "surface_twin":
        clauses = [list(reversed(clause)) for clause in reversed(clauses)]
        template = "SATCOUNT-R/twin/v1"
        intro = "Consider the Boolean formula below; clauses and literals use an alternate order."
    else:
        template = "SATCOUNT-R/canonical/v1"
        intro = "Consider the Boolean formula below."
    rendered = [
        "("
        + " OR ".join(
            f"x{abs(literal)}" if literal > 0 else f"NOT x{abs(literal)}" for literal in clause
        )
        + ")"
        for clause in clauses
    ]
    prompt = "\n".join(
        [
            intro,
            f"It uses Boolean variables x1 through x{spec['n_variables']}.",
            " AND ".join(rendered) + ".",
            "How many assignments satisfy the formula?",
            FINAL_ANSWER_INSTRUCTION,
        ]
    )
    return prompt, template


def render_reasoning(item: ReasoningItem, *, surface: str = "canonical") -> ReasoningView:
    if surface not in {"canonical", "surface_twin"}:
        raise ValueError("reasoning surface must be canonical or surface_twin")
    builders: dict[str, Any] = {
        "MODREG-R": _modreg,
        "FSM-R": _fsm,
        "SATCOUNT-R": _satcount,
    }
    prompt, template_name = builders[item.family](item, surface)
    template_hash = hashlib.sha256(
        f"{template_name}\n{FINAL_ANSWER_INSTRUCTION}".encode()
    ).hexdigest()
    return ReasoningView(
        latent_id=item.latent_id,
        view_id=f"{item.latent_id}:{surface}",
        family=item.family,
        cell=item.cell,
        surface=surface,
        answer=item.answer,
        prompt=prompt,
        prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        template_hash=template_hash,
        metadata={
            "suite_version": item.suite_version,
            "generator_version": item.generator_version,
            "latent_hash": item.latent_hash,
            "template_name": template_name,
        },
    )
