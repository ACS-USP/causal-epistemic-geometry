"""JSONL benchmark adapter with strict validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.base import AnswerParser, Benchmark
from epistemic_geometry.types import BenchmarkItem


class JsonlBenchmark(Benchmark):
    """Read ``id``, ``prompt``, ``target``, and optional ``metadata`` records."""

    def __init__(
        self,
        path: str | Path,
        allowed_targets: list[str] | None = None,
        max_items: int | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Benchmark JSONL does not exist: {self.path}")
        self.parser = AnswerParser(set(allowed_targets) if allowed_targets else None)
        self._items = self._read(max_items)

    def _read(self, max_items: int | None) -> list[BenchmarkItem]:
        items: list[BenchmarkItem] = []
        seen: set[str] = set()
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Line {line_number} must contain a JSON object")
                for key in ("id", "prompt", "target"):
                    if not isinstance(record.get(key), str) or not record[key].strip():
                        raise ValueError(
                            f"Line {line_number} requires non-empty string field {key!r}"
                        )
                item_id = record["id"]
                if item_id in seen:
                    raise ValueError(f"Duplicate benchmark item id: {item_id!r}")
                metadata = record.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError(f"Line {line_number} metadata must be an object")
                item = BenchmarkItem(
                    id=item_id,
                    prompt=record["prompt"],
                    target=record["target"],
                    metadata=metadata,
                )
                self.parser.validate(item)
                seen.add(item_id)
                items.append(item)
                if max_items is not None and len(items) >= max_items:
                    break
        if not items:
            raise ValueError(f"Benchmark JSONL is empty: {self.path}")
        return items

    def items(self) -> list[BenchmarkItem]:
        return list(self._items)
