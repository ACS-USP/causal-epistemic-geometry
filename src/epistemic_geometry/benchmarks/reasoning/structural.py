"""Shallow, model-free shortcut audits for reasoning-suite cells."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .base import ReasoningItem


def shallow_features(item: ReasoningItem) -> dict[str, int | str]:
    """Return features that do not execute the full oracle trajectory."""

    spec = item.spec
    if item.family == "MODREG-R":
        operations = spec["operations"]
        counts = Counter(operation["op"] for operation in operations)
        return {
            "query": str(spec["query"]),
            "initial_sum": int(sum(spec["initial"])),
            "initial_query": int(spec["initial"][int(str(spec["query"])[1])]),
            **{
                f"op_{name}": int(counts.get(name, 0))
                for name in ("ADD_CONST", "MUL_UNIT", "ADD_REG", "SUB_REG", "SWAP")
            },
        }
    if item.family == "FSM-R":
        counts = Counter(spec["sequence"])
        return {
            "start": int(spec["start"]),
            "sequence_length": len(spec["sequence"]),
            "symbol_A": int(counts.get("A", 0)),
            "symbol_B": int(counts.get("B", 0)),
            "symbol_C": int(counts.get("C", 0)),
            "first_symbol": str(spec["sequence"][0]),
            "last_symbol": str(spec["sequence"][-1]),
        }
    if item.family == "SATCOUNT-R":
        clauses = spec["clauses"]
        occurrences = Counter(abs(int(literal)) for clause in clauses for literal in clause)
        positives = sum(int(literal) > 0 for clause in clauses for literal in clause)
        return {
            "n_variables": int(spec["n_variables"]),
            "n_clauses": len(clauses),
            "positive_literal_count": int(positives),
            "width_2": sum(len(clause) == 2 for clause in clauses),
            "width_3": sum(len(clause) == 3 for clause in clauses),
            "max_variable_occurrences": max(occurrences.values(), default=0),
        }
    raise ValueError(f"unknown family {item.family!r}")


def mode_accuracy(items: list[ReasoningItem]) -> float:
    if not items:
        return float("nan")
    mode = Counter(item.answer for item in items).most_common(1)[0][0]
    return sum(item.answer == mode for item in items) / len(items)


def _majority(labels: list[int]) -> int:
    return Counter(labels).most_common(1)[0][0]


def _gini(labels: list[int]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def _best_split(rows: list[tuple[dict[str, int | str], int]]) -> str | None:
    if not rows:
        return None
    features = sorted(rows[0][0])
    parent = _gini([label for _features, label in rows])
    best: tuple[float, str] | None = None
    for feature in features:
        groups: dict[object, list[int]] = defaultdict(list)
        for values, label in rows:
            groups[values[feature]].append(label)
        if len(groups) < 2:
            continue
        weighted = sum(len(labels) / len(rows) * _gini(labels) for labels in groups.values())
        gain = parent - weighted
        candidate = (gain, feature)
        if best is None or candidate > best:
            best = candidate
    return best[1] if best and best[0] > 0 else None


def _tree_fit_predict(
    train: list[tuple[dict[str, int | str], int]],
    query: list[dict[str, int | str]],
    *,
    max_depth: int,
) -> list[int]:
    def build(rows: list[tuple[dict[str, int | str], int]], depth: int) -> Any:
        labels = [label for _features, label in rows]
        node = {"majority": _majority(labels), "feature": None, "children": {}}
        if depth >= max_depth or len(set(labels)) <= 1:
            return node
        feature = _best_split(rows)
        if feature is None:
            return node
        node["feature"] = feature
        groups: dict[object, list[tuple[dict[str, int | str], int]]] = defaultdict(list)
        for values, label in rows:
            groups[values[feature]].append((values, label))
        node["children"] = {
            value: build(group, depth + 1) for value, group in sorted(groups.items(), key=str)
        }
        return node

    tree = build(train, 0)
    predictions: list[int] = []
    for values in query:
        node = tree
        while node["feature"] is not None:
            child = node["children"].get(values[node["feature"]])
            if child is None:
                break
            node = child
        predictions.append(int(node["majority"]))
    return predictions


def shallow_shortcut_audit(items: list[ReasoningItem]) -> dict[str, Any]:
    """Report weak structural predictors without using oracle trajectories."""

    if len(items) < 2:
        raise ValueError("shortcut audit needs at least two items")
    rows = [(shallow_features(item), item.answer) for item in items]
    split = max(1, len(rows) * 4 // 5)
    train, test = rows[:split], rows[split:]
    predictions = _tree_fit_predict(train, [features for features, _label in test], max_depth=4)
    tree_accuracy = sum(
        prediction == label
        for prediction, (_features, label) in zip(predictions, test, strict=True)
    ) / len(test)
    return {
        "mode_accuracy": mode_accuracy(items),
        "depth_4_tree_accuracy_holdout": tree_accuracy,
        "feature_names": sorted(rows[0][0]),
        "feature_policy": "shallow surface/spec statistics only; no oracle trajectory",
    }
