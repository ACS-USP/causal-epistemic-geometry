"""Deterministic canonical, surface-twin, and response-channel renderings."""

from __future__ import annotations

import hashlib

from .base import (
    DECIMAL_ANSWER_INSTRUCTION,
    NUMBER_WORD_ANSWER_INSTRUCTION,
    NUMBER_WORDS,
    LatentItem,
    RenderedView,
    ResponseChannel,
    Surface,
)


def _template_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _answer_instruction(channel: ResponseChannel) -> str:
    return DECIMAL_ANSWER_INSTRUCTION if channel == "decimal" else NUMBER_WORD_ANSWER_INSTRUCTION


def _target_text(target: int, channel: ResponseChannel) -> str:
    return str(target) if channel == "decimal" else NUMBER_WORDS[target]


def _modreg_prompt(item: LatentItem, surface: Surface) -> tuple[str, str]:
    spec = item.spec
    if surface == "canonical":
        names = {name: name for name in ("R0", "R1", "R2", "R3")}
        header = "Registers R0, R1, R2, and R3 hold integers modulo 10."
        template = "MODREG10/canonical/v1"
    else:
        names = {name: f"X{(int(name[1]) + 1) % 4}" for name in ("R0", "R1", "R2", "R3")}
        header = "Four registers X1, X2, X3, and X0 hold integers modulo 10."
        template = "MODREG10/twin/v1"
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
    lines.extend([f"What is the final value of {names[spec['query']]}?", "{answer_instruction}"])
    return "\n".join(lines), template


def _fsm_prompt(item: LatentItem, surface: Surface) -> tuple[str, str]:
    spec = item.spec
    transitions = spec["transitions"]
    if surface == "canonical":
        symbol_map = {symbol: symbol for symbol in ("A", "B", "C")}
        rows = list(range(10))
        template = "FSM10/canonical/v1"
        intro = "A deterministic machine has states 0 through 9 and symbols A, B, and C."
    else:
        symbol_map = {"A": "C", "B": "A", "C": "B"}
        rows = list(reversed(range(10)))
        template = "FSM10/twin/v1"
        intro = (
            "A deterministic machine has ten states numbered 0 through 9 and symbols C, A, and B."
        )
    lines = [
        intro,
        "For each row state and each symbol, the table gives the next state:",
        "state | A | B | C",
    ]
    for state in rows:
        lines.append(
            f"{state} | {transitions['A'][state]} | {transitions['B'][state]} "
            f"| {transitions['C'][state]}"
        )
    shown_sequence = [symbol_map[symbol] for symbol in spec["sequence"]]
    lines.extend(
        [
            f"Start at state {spec['start']}.",
            "Apply this symbol sequence in order: " + " ".join(shown_sequence) + ".",
            "What is the final state?",
            "{answer_instruction}",
        ]
    )
    # A symbol renaming must also rename table columns.  Rebuild those headings
    # and values in the twin text while retaining the same transition function.
    if surface == "surface_twin":
        header = "state | C | A | B"
        rebuilt = [
            intro,
            "For each row state and each symbol, the table gives the next state:",
            header,
        ]
        for state in rows:
            rebuilt.append(
                f"{state} | {transitions['A'][state]} | {transitions['B'][state]} "
                f"| {transitions['C'][state]}"
            )
        rebuilt.extend(lines[-4:])
        lines = rebuilt
    return "\n".join(lines), template


def _reachcount_prompt(item: LatentItem, surface: Surface) -> tuple[str, str]:
    spec = item.spec
    edges = [[int(a), int(b)] for a, b in spec["edges"]]
    if surface == "surface_twin":
        edges = list(reversed(edges))
        template = "REACHCOUNT10/twin/v1"
        intro = (
            "A directed graph has nodes 0 through 9. Its directed edges are "
            "listed below in reverse presentation order:"
        )
    else:
        template = "REACHCOUNT10/canonical/v1"
        intro = "A directed graph has nodes 0 through 9. Its directed edges are listed below:"
    edge_text = ", ".join(f"{source}->{target}" for source, target in edges) or "none"
    prompt = "\n".join(
        [
            intro,
            edge_text + ".",
            f"Starting from node {spec['source']}, how many distinct other nodes are "
            f"reachable using at most {spec['max_hops']} directed edges?",
            "{answer_instruction}",
        ]
    )
    return prompt, template


def _satcount_prompt(item: LatentItem, surface: Surface) -> tuple[str, str]:
    spec = item.spec
    clauses = [[int(literal) for literal in clause] for clause in spec["clauses"]]
    if surface == "surface_twin":
        clauses = [list(reversed(clause)) for clause in reversed(clauses)]
        template = "SATCOUNT10/twin/v1"
        intro = (
            "Consider the Boolean formula below; its clauses and literals are "
            "shown in an alternate order."
        )
    else:
        template = "SATCOUNT10/canonical/v1"
        intro = "Consider the Boolean formula below."
    rendered_clauses = [
        "("
        + " OR ".join(
            (f"x{abs(literal)}" if literal > 0 else f"NOT x{abs(literal)}") for literal in clause
        )
        + ")"
        for clause in clauses
    ]
    formula = " AND ".join(rendered_clauses)
    prompt = "\n".join(
        [
            intro,
            f"It uses Boolean variables x1 through x{spec['n_variables']}.",
            f"{formula}.",
            "How many assignments satisfy the formula, modulo 10?",
            "{answer_instruction}",
        ]
    )
    return prompt, template


def render_latent(
    item: LatentItem,
    *,
    surface: Surface = "canonical",
    response_channel: ResponseChannel = "decimal",
) -> RenderedView:
    """Render one latent item without changing its oracle or identity."""

    builders = {
        "MODREG10": _modreg_prompt,
        "FSM10": _fsm_prompt,
        "REACHCOUNT10": _reachcount_prompt,
        "SATCOUNT10": _satcount_prompt,
    }
    try:
        prompt_template, template_name = builders[item.family](item, surface)
    except KeyError as exc:
        raise ValueError(f"unsupported E3-10 family {item.family!r}") from exc
    instruction = _answer_instruction(response_channel)
    prompt = prompt_template.replace("{answer_instruction}", instruction)
    template_hash = _template_hash(template_name + "\n" + instruction)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    view_id = f"{item.latent_id}:{surface}:{response_channel}"
    return RenderedView(
        latent_id=item.latent_id,
        view_id=view_id,
        family=item.family,
        cell=item.cell,
        surface=surface,
        response_channel=response_channel,
        target=item.target,
        target_text=_target_text(item.target, response_channel),
        prompt=prompt,
        prompt_hash=prompt_hash,
        template_hash=template_hash,
        metadata={
            "suite_version": item.suite_version,
            "generator_version": item.generator_version,
            "latent_hash": item.latent_hash,
            "template_name": template_name,
        },
    )


def template_hashes() -> dict[str, str]:
    """Return frozen hashes for the four family templates and two channels."""

    result: dict[str, str] = {}
    for family in ("MODREG10", "FSM10", "REACHCOUNT10", "SATCOUNT10"):
        for surface in ("canonical", "surface_twin"):
            for channel, instruction in (
                ("decimal", DECIMAL_ANSWER_INSTRUCTION),
                ("number_word", NUMBER_WORD_ANSWER_INSTRUCTION),
            ):
                result[f"{family}:{surface}:{channel}"] = _template_hash(
                    f"{family}/{surface}/v1\n{instruction}"
                )
    return result
