"""Candidate benchmark metadata and normalized JSONL adapters.

The adapters do not fetch data.  A remote command must explicitly materialize
an official dataset into a normalized JSONL file and pass that file in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .base import ExternalItem, record_digest, validate_item


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    subtask: str
    official_source: str
    source_revision_policy: str
    objective_evaluation: str
    q1_max_new_tokens: int
    status: str = "UNTESTED"
    notes: str = ""


_SPECS = (
    CandidateSpec(
        name="RE2-Bench",
        subtask="output_prediction",
        official_source="https://arxiv.org/abs/2512.14917",
        source_revision_policy="record official release/repository revision on RunPod",
        objective_evaluation="official executable output evaluator required",
        q1_max_new_tokens=2048,
        notes="Q0 remains unresolved until a stable official public artifact/evaluator is located.",
    ),
    CandidateSpec(
        name="LiveCodeBench",
        subtask="test_output_prediction",
        official_source="https://github.com/LiveCodeBench/LiveCodeBench",
        source_revision_policy="record dataset release and evaluator commit",
        objective_evaluation="official deterministic test-output evaluator",
        q1_max_new_tokens=2048,
    ),
    CandidateSpec(
        name="CRUXEval",
        subtask="output_prediction",
        official_source="https://github.com/facebookresearch/cruxeval",
        source_revision_policy="record dataset revision and evaluator commit",
        objective_evaluation="deterministic Python output evaluator",
        q1_max_new_tokens=2048,
        notes="Established control; possible contamination risk is documented, not hidden.",
    ),
    CandidateSpec(
        name="LiveBench",
        subtask="objective_subtask",
        official_source="https://github.com/livebench/livebench",
        source_revision_policy="record LiveBench release date and evaluator commit",
        objective_evaluation="objective subtasks only; reject LLM-judge tasks",
        q1_max_new_tokens=2048,
    ),
)


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return _SPECS


class ExternalBenchmarkAdapter:
    """Small adapter boundary used by Q0/Q1/Q2 runners."""

    spec: ClassVar[CandidateSpec]

    def load_items(self, path: Path, *, limit: int | None = None) -> list[ExternalItem]:
        raise NotImplementedError

    def validate(self, items: list[ExternalItem]) -> dict[str, Any]:
        if not items:
            raise ValueError(f"{self.spec.name} adapter received no items")
        seen: set[str] = set()
        for item in items:
            validate_item(item)
            if item.item_id in seen:
                raise ValueError(f"duplicate external item id: {item.item_id}")
            seen.add(item.item_id)
            if item.benchmark != self.spec.name or item.subtask != self.spec.subtask:
                raise ValueError(
                    f"item {item.item_id} does not match {self.spec.name}/{self.spec.subtask}"
                )
        return {
            "adapter": self.spec.name,
            "subtask": self.spec.subtask,
            "n_items": len(items),
            "item_ids_unique": True,
            "item_digest": record_digest(items),
            "objective_evaluator": self.spec.objective_evaluation,
        }


class JsonlExternalAdapter(ExternalBenchmarkAdapter):
    """Load normalized records produced by an official remote adapter.

    The normalized schema is intentionally boring and auditable.  It lets the
    model runner remain identical across candidates without pretending that
    their native official schemas are interchangeable.
    """

    def __init__(self, spec: CandidateSpec):
        self.spec = spec

    def load_items(self, path: Path, *, limit: int | None = None) -> list[ExternalItem]:
        if not path.exists():
            raise FileNotFoundError(path)
        items: list[ExternalItem] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
                required = {
                    "item_id",
                    "benchmark",
                    "subtask",
                    "prompt",
                    "reference_answer",
                    "evaluator",
                    "source_revision",
                }
                missing = required - record.keys()
                if missing:
                    raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
                items.append(
                    ExternalItem(
                        item_id=str(record["item_id"]),
                        benchmark=str(record["benchmark"]),
                        subtask=str(record["subtask"]),
                        prompt=str(record["prompt"]),
                        reference_answer=str(record["reference_answer"]),
                        evaluator=str(record["evaluator"]),
                        source_revision=str(record["source_revision"]),
                        metadata=dict(record.get("metadata", {})),
                    )
                )
                if limit is not None and len(items) >= limit:
                    break
        self.validate(items)
        return items


def adapter_for(name: str) -> JsonlExternalAdapter:
    for spec in _SPECS:
        if spec.name == name:
            return JsonlExternalAdapter(spec)
    raise KeyError(f"unknown candidate benchmark: {name}")
