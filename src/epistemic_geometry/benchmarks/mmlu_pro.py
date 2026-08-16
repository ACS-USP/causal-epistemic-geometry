"""Strict adapter for the official TIGER-Lab/MMLU-Pro HuggingFace dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.base import AnswerParser, Benchmark
from epistemic_geometry.reproducibility import require_remote_hf_execution, stable_digest
from epistemic_geometry.types import BenchmarkItem

MMLU_PRO_ID = "TIGER-Lab/MMLU-Pro"
LABELS = tuple("ABCDEFGHIJ")
PROMPT_TEMPLATE_ID = "Q1_V1_MMLU_PRO_DIRECT_CHOICE_V1"


def render_mmlu_pro_question(question: str, options: list[str]) -> str:
    """Render the fixed V1 direct-choice prompt without chain-of-thought text."""

    lines = [
        "Choose the correct answer to the following multiple-choice question.",
        "Respond with only the answer letter.",
        "",
        "Question:",
        question.strip(),
        "",
    ]
    lines.extend(
        f"{label}. {option}"
        for label, option in zip(LABELS[: len(options)], options, strict=True)
    )
    return "\n".join(lines)


def _answer_index(row: dict[str, Any], option_count: int) -> int:
    value = row.get("answer_index")
    if value is None:
        value = row.get("answer")
    if isinstance(value, str) and value.strip().upper() in LABELS:
        index = LABELS.index(value.strip().upper())
    else:
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("MMLU-Pro row has no valid answer_index/answer") from exc
    if index < 0 or index >= option_count:
        raise ValueError(f"MMLU-Pro answer index {index} is outside 0..{option_count - 1}")
    return index


def row_to_item(row: dict[str, Any], dataset_split: str) -> BenchmarkItem:
    """Convert one dataset row while deliberately ignoring ``cot_content``."""

    question_id = str(row.get("question_id", "")).strip()
    question = str(row.get("question", "")).strip()
    options = row.get("options")
    if not question_id or not question or not isinstance(options, list) or not options:
        raise ValueError("MMLU-Pro rows require question_id, question, and non-empty options")
    options = [str(option).strip() for option in options]
    if len(options) > len(LABELS):
        raise ValueError(f"MMLU-Pro row {question_id} has more than ten options")
    answer_index = _answer_index(row, len(options))
    item_id = f"{dataset_split}:{question_id}"
    return BenchmarkItem(
        id=item_id,
        prompt=render_mmlu_pro_question(question, options),
        target=LABELS[answer_index],
        metadata={
            "dataset_id": MMLU_PRO_ID,
            "dataset_split": dataset_split,
            "question_id": question_id,
            "category": str(row.get("category", "UNKNOWN")),
            "src": str(row.get("src", "UNKNOWN")),
            "options": options,
            "candidate_labels": list(LABELS[: len(options)]),
            "answer_index": answer_index,
        },
    )


class MMLUProBenchmark(Benchmark):
    """Load official MMLU-Pro rows or deterministic IDs from a split manifest."""

    def __init__(
        self,
        split: str,
        dataset_revision: str | None = None,
        split_manifest: str | Path | None = None,
        max_items: int | None = None,
        dataset_id: str = MMLU_PRO_ID,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        if dataset_id != MMLU_PRO_ID:
            raise ValueError(f"Only {MMLU_PRO_ID} is supported by this adapter")
        if split not in {"validation", "test", "dev_calibration", "dev_evaluation"}:
            raise ValueError(f"Unsupported MMLU-Pro split: {split}")
        self.dataset_id = dataset_id
        self.requested_split = split
        self.dataset_revision = dataset_revision or "UNKNOWN"
        self.dataset_fingerprint = "UNKNOWN"
        self.parser = AnswerParser(set(LABELS))
        if rows is None:
            rows, fingerprint = self._load_rows(dataset_revision)
            self.dataset_fingerprint = fingerprint
        base_split = "test" if split in {"dev_calibration", "dev_evaluation"} else split
        items_by_id = {
            item.id: item for item in (row_to_item(row, base_split) for row in rows)
        }
        if split in {"dev_calibration", "dev_evaluation"}:
            if split_manifest is None:
                raise ValueError(f"{split} requires split_manifest")
            manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
            ids = manifest.get("splits", {}).get(split)
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"Split manifest lacks non-empty {split} IDs")
            missing = [item_id for item_id in ids if item_id not in items_by_id]
            if missing:
                raise ValueError(f"Split manifest contains unknown MMLU-Pro IDs: {missing[:3]}")
            selected = [items_by_id[item_id] for item_id in ids]
        else:
            selected = list(items_by_id.values())
        self._items = selected[:max_items] if max_items is not None else selected
        if not self._items:
            raise ValueError(f"MMLU-Pro {split} selection is empty")

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]], split: str = "validation") -> MMLUProBenchmark:
        """Build a deterministic fixture without importing the datasets package."""

        return cls(split=split, rows=rows, dataset_revision="fixture")

    def _load_rows(self, revision: str | None) -> tuple[list[dict[str, Any]], str]:
        require_remote_hf_execution("MMLU-Pro dataset loading")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "MMLU-Pro requires the optional datasets package; install with "
                "pip install -e '.[hf]'"
            ) from exc
        source_split = "validation" if self.requested_split == "validation" else "test"
        dataset = load_dataset(self.dataset_id, split=source_split, revision=revision)
        fingerprint = str(getattr(dataset, "_fingerprint", "UNKNOWN"))
        rows = [dict(row) for row in dataset]
        return rows, fingerprint

    def items(self) -> list[BenchmarkItem]:
        return list(self._items)

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "requested_split": self.requested_split,
            "dataset_revision": self.dataset_revision,
            "dataset_fingerprint": self.dataset_fingerprint,
            "item_count": len(self._items),
            "item_ids_hash": stable_digest(*[item.id for item in self._items]),
        }
