"""Deterministic restricted-Python generator and independent reference paths.

The scientific family is the normalized program skeleton, not a rendered
program/input pair. Constants and identifiers are deliberately excluded from
the canonical skeleton. The module never consumes model outcomes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from epistemic_geometry.reproducibility import canonical_json

GENERATOR_VERSION = "q3-restricted-python-generator-v1"
REFERENCE_VERSION = "q3-dual-reference-v1"
PROMPT_TEMPLATE_VERSION = "q3-restricted-python-output-v1"
INTEGER_BOUND = 1_000_000_000
CONTAINER_BOUND = 128
LOOP_BOUND = 10_000
RECURSION_BOUND = 32
REFERENCE_TIMEOUT_SECONDS = 2.0
REFERENCE_MEMORY_MB = 256
BEHAVIORAL_PROBE_COUNT = 32

ARCHETYPES = (
    "ARITHMETIC",
    "BRANCHING",
    "SEQUENCE_ALIASING",
    "TEXT",
    "MAPPING",
    "NESTED_CONTROL",
    "BOUNDED_RECURSION",
    "MIXED",
)
OUTPUT_TYPES = ("int", "bool", "str", "list", "tuple", "dict")
OP_KINDS = ("AFFINE", "FOLD", "BRANCH", "MUTATE", "TEXT", "DICT", "NESTED", "RECURSE")
_TEXTS = ("amber", "cobalt", "delta", "ember", "forest", "indigo", "lumen", "quartz")


@dataclass(frozen=True)
class Family:
    """One accepted independently generated program skeleton and typed input."""

    family_id: str
    namespace: str
    candidate_index: int
    archetype: str
    complexity: int
    operations: tuple[dict[str, int | str], ...]
    output_type: str
    input_value: dict[str, Any]
    source: str
    prompt: str
    reference_type: str
    reference_repr: str
    canonical_skeleton: str
    canonical_skeleton_sha256: str
    normalized_token_sha256: str
    behavioral_signature_sha256: str

    def to_record(self) -> dict[str, Any]:
        row = asdict(self)
        row["operations"] = list(self.operations)
        return row


def _sha(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _typed(value: Any) -> tuple[str, str]:
    if type(value) not in {int, bool, str, list, tuple, dict}:  # noqa: E721
        raise TypeError(f"unsupported output type {type(value).__name__}")
    if isinstance(value, (list, tuple)) and len(value) > CONTAINER_BOUND:
        raise ValueError("container output exceeds bound")
    if isinstance(value, dict) and (
        len(value) > CONTAINER_BOUND or not all(isinstance(k, str) for k in value)
    ):
        raise ValueError("dictionary output violates contract")
    return type(value).__name__, repr(value)


def _clamp(value: int) -> int:
    return int(value) % INTEGER_BOUND


def custom_reference(
    operations: tuple[dict[str, int | str], ...], output_type: str, data: dict[str, Any]
) -> Any:
    """Interpret the generator IR without compiling or executing Python source."""

    n = int(data["n"])
    xs = [int(value) for value in data["values"]]
    text = str(data["text"])
    acc = _clamp(n)
    bag = {"a": len(xs), "b": n % 7}
    total_iterations = 0
    for op in operations:
        kind = str(op["kind"])
        a, b, c = int(op["a"]), int(op["b"]), int(op["c"])
        variant = int(op["variant"])
        if kind == "AFFINE":
            acc = _clamp(acc * a + b + len(xs) * c)
        elif kind == "FOLD":
            for index, value in enumerate(xs):
                total_iterations += 1
                term = (value * a + index * b) % c
                acc = _clamp(acc + term if (index + variant) % 2 == 0 else acc - term)
        elif kind == "BRANCH":
            if acc % a < b % a:
                acc = _clamp(acc + c * (len(text) + variant))
            else:
                acc = _clamp(acc * c - b)
        elif kind == "MUTATE":
            if xs:
                position = (acc + a + variant) % len(xs)
                xs[position] = _clamp(xs[position] * b + c) % 997
                acc = _clamp(acc + xs[position])
        elif kind == "TEXT":
            if text:
                shift = (a + variant) % len(text)
                text = text[shift:] + text[:shift]
                if b % 2:
                    text = text[::-1]
            acc = _clamp(acc + len(text) * c + sum(ord(ch) for ch in text[:4]))
        elif kind == "DICT":
            key = "a" if variant % 2 == 0 else "b"
            bag[key] = (bag[key] * a + acc + b) % c
            acc = _clamp(acc + bag["a"] - bag["b"])
        elif kind == "NESTED":
            outer = a % 4 + 1
            inner = b % 4 + 1
            for i in range(outer):
                for j in range(inner):
                    total_iterations += 1
                    if (i + j + variant) % 2:
                        acc = _clamp(acc + (i + 1) * (j + c))
                    else:
                        acc = _clamp(acc - (j + 1) * (i + c))
        elif kind == "RECURSE":
            value = abs(acc + a * n) % INTEGER_BOUND
            depth = 0
            digit_sum = 0
            while value and depth < RECURSION_BOUND:
                digit_sum += value % 10
                value //= 10
                depth += 1
            acc = _clamp(acc + digit_sum * b + c)
        else:
            raise ValueError(f"unknown IR operation {kind}")
        if total_iterations > LOOP_BOUND:
            raise ValueError("loop bound exceeded")

    if output_type == "int":
        return int(acc % 100_003) - 50_001
    if output_type == "bool":
        return bool((acc + len(xs) + len(text)) % 2)
    if output_type == "str":
        return f"{text}:{acc % 1000}"
    if output_type == "list":
        return [int(acc % 101), *[int(value % 101) for value in xs[:4]]]
    if output_type == "tuple":
        return (int(acc % 997), len(text), int(sum(xs) % 997))
    if output_type == "dict":
        return {"acc": int(acc % 997), "size": len(xs), "text": text[:4]}
    raise ValueError(f"unsupported output type {output_type}")


def render_program(operations: tuple[dict[str, int | str], ...], output_type: str) -> str:
    """Render the IR as a restricted standalone Python function."""

    lines = [
        "def solve(data):",
        "    n = int(data['n'])",
        "    xs = [int(value) for value in data['values']]",
        "    text = str(data['text'])",
        f"    acc = n % {INTEGER_BOUND}",
        "    bag = {'a': len(xs), 'b': n % 7}",
    ]
    for index, op in enumerate(operations):
        kind = str(op["kind"])
        a, b, c, variant = (int(op[key]) for key in ("a", "b", "c", "variant"))
        lines.append(f"    # step {index}: {kind}")
        if kind == "AFFINE":
            lines.append(f"    acc = (acc * {a} + {b} + len(xs) * {c}) % {INTEGER_BOUND}")
        elif kind == "FOLD":
            lines.extend(
                [
                    "    for index, value in enumerate(xs):",
                    f"        term = (value * {a} + index * {b}) % {c}",
                    f"        if (index + {variant}) % 2 == 0:",
                    f"            acc = (acc + term) % {INTEGER_BOUND}",
                    "        else:",
                    f"            acc = (acc - term) % {INTEGER_BOUND}",
                ]
            )
        elif kind == "BRANCH":
            lines.extend(
                [
                    f"    if acc % {a} < {b % a}:",
                    f"        acc = (acc + {c} * (len(text) + {variant})) % {INTEGER_BOUND}",
                    "    else:",
                    f"        acc = (acc * {c} - {b}) % {INTEGER_BOUND}",
                ]
            )
        elif kind == "MUTATE":
            lines.extend(
                [
                    "    if xs:",
                    f"        position = (acc + {a} + {variant}) % len(xs)",
                    f"        xs[position] = ((xs[position] * {b} + {c}) % {INTEGER_BOUND}) % 997",
                    f"        acc = (acc + xs[position]) % {INTEGER_BOUND}",
                ]
            )
        elif kind == "TEXT":
            lines.extend(
                [
                    "    if text:",
                    f"        shift = ({a} + {variant}) % len(text)",
                    "        text = text[shift:] + text[:shift]",
                ]
            )
            if b % 2:
                lines.append("        text = text[::-1]")
            lines.extend(
                [
                    "    char_total = 0",
                    "    for ch in text[:4]:",
                    "        char_total += ord(ch)",
                    f"    acc = (acc + len(text) * {c} + char_total) % {INTEGER_BOUND}",
                ]
            )
        elif kind == "DICT":
            key = "a" if variant % 2 == 0 else "b"
            lines.extend(
                [
                    f"    bag['{key}'] = (bag['{key}'] * {a} + acc + {b}) % {c}",
                    f"    acc = (acc + bag['a'] - bag['b']) % {INTEGER_BOUND}",
                ]
            )
        elif kind == "NESTED":
            lines.extend(
                [
                    f"    for i in range({a % 4 + 1}):",
                    f"        for j in range({b % 4 + 1}):",
                    f"            if (i + j + {variant}) % 2:",
                    f"                acc = (acc + (i + 1) * (j + {c})) % {INTEGER_BOUND}",
                    "            else:",
                    f"                acc = (acc - (j + 1) * (i + {c})) % {INTEGER_BOUND}",
                ]
            )
        elif kind == "RECURSE":
            lines.extend(
                [
                    f"    value = abs(acc + {a} * n) % {INTEGER_BOUND}",
                    "    depth = 0",
                    "    digit_sum = 0",
                    f"    while value and depth < {RECURSION_BOUND}:",
                    "        digit_sum += value % 10",
                    "        value //= 10",
                    "        depth += 1",
                    f"    acc = (acc + digit_sum * {b} + {c}) % {INTEGER_BOUND}",
                ]
            )
        else:
            raise ValueError(f"unknown IR operation {kind}")
    returns = {
        "int": "int(acc % 100003) - 50001",
        "bool": "bool((acc + len(xs) + len(text)) % 2)",
        "str": "f'{text}:{acc % 1000}'",
        "list": "[int(acc % 101)] + [int(value % 101) for value in xs[:4]]",
        "tuple": "(int(acc % 997), len(text), int(sum(xs) % 997))",
        "dict": "{'acc': int(acc % 997), 'size': len(xs), 'text': text[:4]}",
    }
    lines.append(f"    return {returns[output_type]}")
    return "\n".join(lines) + "\n"


_ALLOWED_NODES = {
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.AnnAssign,
    ast.Return,
    ast.If,
    ast.For,
    ast.While,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.ListComp,
    ast.comprehension,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Subscript,
    ast.Slice,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.AugAssign,
    ast.JoinedStr,
    ast.FormattedValue,
}
_ALLOWED_CALLS = {"abs", "bool", "enumerate", "int", "len", "ord", "range", "str", "sum"}


def validate_restricted_source(source: str) -> None:
    """Fail closed unless generated source remains in the exact safe subset."""

    tree = ast.parse(source, mode="exec")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "solve" or len(tree.body) != 1:
        raise ValueError("source must contain exactly one solve function")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"forbidden AST production: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
                raise ValueError("forbidden call target")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden")


_WORKER = r"""import ast, json, sys
payload = json.loads(sys.stdin.read())
source = payload["source"]
data = payload["input"]
blocked = ("open", "socket", "subprocess", "os.system", "os.spawn", "ctypes", "import")
def audit(event, args):
    if event.startswith(blocked):
        raise RuntimeError("blocked audit event: " + event)
sys.addaudithook(audit)
scope = {"__builtins__": {"abs": abs, "bool": bool, "enumerate": enumerate,
         "int": int, "len": len, "ord": ord, "range": range, "str": str, "sum": sum}}
exec(compile(source, "<q3-generated>", "exec"), scope, scope)
value = scope["solve"](data)
if type(value) not in {int, bool, str, list, tuple, dict}:
    raise TypeError(type(value).__name__)
print(json.dumps({"type": type(value).__name__, "repr": repr(value)}, sort_keys=True))
"""


def _limit_reference_worker() -> None:
    cpu = max(1, math.ceil(REFERENCE_TIMEOUT_SECONDS))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if sys.platform != "darwin":
        memory = REFERENCE_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))


def _communicate_with_rss_cap(
    process: subprocess.Popen[str], payload: str
) -> tuple[str, str, bool]:
    """Enforce the 256 MiB resident-memory cap on macOS.

    macOS rejects lowering RLIMIT_AS below the framework Python's already
    mapped virtual address space. A 2 ms watchdog enforces the scientifically
    specified resident-memory bound instead and fails closed on inspection
    errors. Linux retains the kernel RLIMIT_AS cap in ``preexec_fn``.
    """

    if sys.platform != "darwin":
        stdout, stderr = process.communicate(payload, timeout=REFERENCE_TIMEOUT_SECONDS)
        return stdout, stderr, False
    import psutil

    violation = threading.Event()
    finished = threading.Event()

    def monitor() -> None:
        target = psutil.Process(process.pid)
        cap = REFERENCE_MEMORY_MB * 1024 * 1024
        try:
            while not finished.is_set() and process.poll() is None:
                if target.memory_info().rss > cap:
                    violation.set()
                    process.kill()
                    return
                time.sleep(0.002)
        except psutil.NoSuchProcess:
            return
        except (psutil.AccessDenied, OSError):
            violation.set()
            if process.poll() is None:
                process.kill()

    watcher = threading.Thread(target=monitor, name="q3-reference-rss-watchdog", daemon=True)
    watcher.start()
    try:
        stdout, stderr = process.communicate(payload, timeout=REFERENCE_TIMEOUT_SECONDS)
    finally:
        finished.set()
        watcher.join(timeout=0.1)
    return stdout, stderr, violation.is_set()


def sandboxed_cpython_reference(
    source: str, data: dict[str, Any], *, python: str | None = None
) -> tuple[str, str]:
    """Execute generator-owned restricted source in a resource-limited process."""

    validate_restricted_source(source)
    executable = python or sys.executable
    with tempfile.TemporaryDirectory(prefix="q3-ref-") as directory:
        path = Path(directory)
        os.chmod(path, 0o500)
        process = subprocess.Popen(
            [executable, "-I", "-S", "-c", _WORKER],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=path,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0"},
            close_fds=True,
            preexec_fn=_limit_reference_worker if os.name == "posix" else None,
        )
        stdout, stderr, memory_violation = _communicate_with_rss_cap(
            process, canonical_json({"source": source, "input": data})
        )
    if memory_violation:
        raise RuntimeError("reference worker exceeded memory cap or could not be monitored")
    if process.returncode != 0:
        raise RuntimeError(f"reference worker failed rc={process.returncode}: {stderr.strip()}")
    result = json.loads(stdout)
    return str(result["type"]), str(result["repr"])


def canonical_skeleton(operations: tuple[dict[str, int | str], ...], output_type: str) -> str:
    """Normalize identifiers and literals while preserving executable structure."""

    tokens = [f"{op['kind']}:{int(op['variant'])}" for op in operations]
    return f"{GENERATOR_VERSION}|{'/'.join(tokens)}|RETURN:{output_type}"


def _behavioral_signature(operations: tuple[dict[str, int | str], ...], output_type: str) -> str:
    probes = (
        {"n": -7, "values": [0, 1, -2], "text": "probe"},
        {"n": 0, "values": [4, 4, 4, 4], "text": "axis"},
        {"n": 19, "values": [-5, 2, 9], "text": "quartz"},
        {"n": 41, "values": [1, -1, 2, -2, 3], "text": "z"},
        {"n": -43, "values": [7, 0, -7, 14], "text": "ember"},
        {"n": -18, "values": [3, 1, 4, 1, 5], "text": "indigo"},
        {"n": -1, "values": [-9, -3, 0, 3, 9], "text": "amber"},
        {"n": 2, "values": [8, 6, 7, 5, 3, 0, 9], "text": "cobalt"},
        {"n": 5, "values": [2, -4, 8, -16], "text": "delta"},
        {"n": 11, "values": [11, 0, 11], "text": "forest"},
        {"n": 23, "values": [-8, 13, -21, 34], "text": "lumen"},
        {"n": 37, "values": [6, 2, 6, 4, 3], "text": "quartz"},
        {"n": 50, "values": [-40, 40, -20, 20], "text": "amber"},
        {"n": -29, "values": [1, 1, 2, 3, 5, 8], "text": "forest"},
        {"n": 31, "values": [10, -10, 30, -30, 0], "text": "indigo"},
        {"n": 47, "values": [-1, -1, -2, -3, -5], "text": "cobalt"},
        {"n": -50, "values": [12, 24, 36], "text": "delta"},
        {"n": -37, "values": [-2, 4, -6, 8], "text": "lumen"},
        {"n": -23, "values": [17, -13, 11, -7], "text": "quartz"},
        {"n": -11, "values": [0, 2, 0, 2, 0], "text": "ember"},
        {"n": 1, "values": [39, -39, 13, -13], "text": "forest"},
        {"n": 3, "values": [-6, -4, -2, 0, 2, 4, 6], "text": "amber"},
        {"n": 7, "values": [5, 10, 15, 20], "text": "cobalt"},
        {"n": 13, "values": [-31, 0, 31], "text": "indigo"},
        {"n": 17, "values": [4, 16, -4, -16], "text": "lumen"},
        {"n": 29, "values": [7, 14, 21, 28], "text": "delta"},
        {"n": 43, "values": [-11, 22, -33], "text": "quartz"},
        {"n": 49, "values": [9, 8, 7, 6, 5, 4], "text": "ember"},
        {"n": -47, "values": [37, 23, 11, 5], "text": "amber"},
        {"n": -31, "values": [-17, -19, -23], "text": "forest"},
        {"n": -13, "values": [32, -16, 8, -4, 2], "text": "cobalt"},
        {"n": 0, "values": [-1, 0, 1, 0, -1], "text": "indigo"},
    )
    if len(probes) != BEHAVIORAL_PROBE_COUNT:
        raise AssertionError("behavioral probe count drifted")
    values = [_typed(custom_reference(operations, output_type, probe)) for probe in probes]
    return _sha(canonical_json(values))


def _required_ops(archetype: str) -> list[str]:
    return {
        "ARITHMETIC": ["AFFINE", "FOLD"],
        "BRANCHING": ["BRANCH", "AFFINE"],
        "SEQUENCE_ALIASING": ["MUTATE", "FOLD"],
        "TEXT": ["TEXT", "BRANCH"],
        "MAPPING": ["DICT", "AFFINE"],
        "NESTED_CONTROL": ["NESTED", "BRANCH"],
        "BOUNDED_RECURSION": ["RECURSE", "AFFINE"],
        "MIXED": ["MUTATE", "TEXT", "DICT", "NESTED", "RECURSE"],
    }[archetype]


def build_family(namespace: str, candidate_index: int, stream_seed: int) -> Family:
    """Build one deterministic candidate; acceptance is handled by the caller."""

    rng = random.Random(_sha(f"{stream_seed}:{namespace}:{candidate_index}"))
    archetype = ARCHETYPES[candidate_index % len(ARCHETYPES)]
    complexity = (6, 8, 10, 12)[(candidate_index // len(ARCHETYPES)) % 4]
    output_type = OUTPUT_TYPES[(candidate_index // (len(ARCHETYPES) * 4)) % len(OUTPUT_TYPES)]
    kinds = _required_ops(archetype)
    while len(kinds) < complexity:
        kinds.append(OP_KINDS[rng.randrange(len(OP_KINDS))])
    rng.shuffle(kinds)
    operations: list[dict[str, int | str]] = []
    for kind in kinds:
        operations.append(
            {
                "kind": kind,
                "variant": rng.randrange(4),
                "a": rng.randrange(2, 17),
                "b": rng.randrange(1, 31),
                "c": rng.randrange(3, 37),
            }
        )
    ops = tuple(operations)
    input_value = {
        "n": rng.randrange(-50, 51),
        "values": [rng.randrange(-40, 41) for _ in range(rng.randrange(3, 9))],
        "text": _TEXTS[rng.randrange(len(_TEXTS))],
    }
    source = render_program(ops, output_type)
    value = custom_reference(ops, output_type, input_value)
    reference_type, reference_repr = _typed(value)
    skeleton = canonical_skeleton(ops, output_type)
    skeleton_sha = _sha(skeleton)
    family_id = f"Q3F-{namespace.upper()}-{skeleton_sha[:20]}"
    prompt = (
        "Predict the exact output of this deterministic Python function.\n\n"
        f"```python\n{source}```\n\n"
        f"Input: {repr(input_value)}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    )
    normalized_tokens = " ".join(skeleton.replace("|", " ").replace("/", " ").split())
    return Family(
        family_id=family_id,
        namespace=namespace,
        candidate_index=candidate_index,
        archetype=archetype,
        complexity=complexity,
        operations=ops,
        output_type=output_type,
        input_value=input_value,
        source=source,
        prompt=prompt,
        reference_type=reference_type,
        reference_repr=reference_repr,
        canonical_skeleton=skeleton,
        canonical_skeleton_sha256=skeleton_sha,
        normalized_token_sha256=_sha(normalized_tokens),
        behavioral_signature_sha256=_behavioral_signature(ops, output_type),
    )


def validate_family(family: Family, *, python: str | None = None) -> dict[str, Any]:
    """Require two independent CPython repetitions and custom-reference agreement."""

    validate_restricted_source(family.source)
    expected = (family.reference_type, family.reference_repr)
    custom = _typed(custom_reference(family.operations, family.output_type, family.input_value))
    first = sandboxed_cpython_reference(family.source, family.input_value, python=python)
    second = sandboxed_cpython_reference(family.source, family.input_value, python=python)
    if custom != expected or first != expected or second != expected:
        raise ValueError(
            "dual reference disagreement "
            f"custom={custom} first={first} second={second} expected={expected}"
        )
    parsed = ast.literal_eval(family.reference_repr)
    if _typed(parsed) != expected:
        raise ValueError("typed parser/reference roundtrip failed")
    return {
        "dual_evaluator_agreement": True,
        "reference_repeat_determinism": True,
        "parser_reference_roundtrip": True,
        "reference_type": family.reference_type,
    }
