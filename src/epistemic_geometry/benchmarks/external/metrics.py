"""Small, pre-specified qualification summaries for external benchmarks."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .base import ExternalResult, ExternalStatus


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class QualificationSummary:
    n_results: int
    n_items: int
    valid_count: int
    correct_count: int
    wrong_count: int
    invalid_count: int
    truncated_count: int
    runtime_error_count: int
    valid_completion: float
    conditional_accuracy: float
    raw_accuracy: float
    valid_interval: tuple[float, float] | None
    conditional_accuracy_interval: tuple[float, float] | None
    raw_accuracy_interval: tuple[float, float] | None
    item_outcomes: dict[str, str]
    paired_counts: dict[str, int]
    seed_accuracy: tuple[float, ...]
    seed_accuracy_gap: float | None
    stable_hard_count: int
    seed_sensitive_count: int
    pair_oracle_accuracy: float | None
    resampling_gain: float | None

    def to_record(self) -> dict[str, object]:
        return {
            "n_results": self.n_results,
            "n_items": self.n_items,
            "valid_count": self.valid_count,
            "correct_count": self.correct_count,
            "wrong_count": self.wrong_count,
            "invalid_count": self.invalid_count,
            "truncated_count": self.truncated_count,
            "runtime_error_count": self.runtime_error_count,
            "valid_completion": self.valid_completion,
            "conditional_accuracy": self.conditional_accuracy,
            "raw_accuracy": self.raw_accuracy,
            "valid_interval_95": self.valid_interval,
            "conditional_accuracy_interval_95": self.conditional_accuracy_interval,
            "raw_accuracy_interval_95": self.raw_accuracy_interval,
            "item_outcomes": self.item_outcomes,
            "paired_counts": self.paired_counts,
            "seed_accuracy": self.seed_accuracy,
            "seed_accuracy_gap": self.seed_accuracy_gap,
            "stable_hard_count": self.stable_hard_count,
            "seed_sensitive_count": self.seed_sensitive_count,
            "pair_oracle_accuracy": self.pair_oracle_accuracy,
            "resampling_gain": self.resampling_gain,
        }


def summarize_qualification(results: Iterable[ExternalResult]) -> QualificationSummary:
    rows = list(results)
    if not rows:
        raise ValueError("cannot summarize empty external results")
    valid = [row for row in rows if row.status in {
        ExternalStatus.VALID_CORRECT,
        ExternalStatus.VALID_WRONG,
    }]
    correct = sum(row.status == ExternalStatus.VALID_CORRECT for row in rows)
    wrong = sum(row.status == ExternalStatus.VALID_WRONG for row in rows)
    invalid = sum(row.status == ExternalStatus.INVALID_FORMAT for row in rows)
    truncated = sum(row.status == ExternalStatus.TRUNCATED_THINKING for row in rows)
    runtime = sum(row.status == ExternalStatus.RUNTIME_ERROR for row in rows)
    n = len(rows)
    valid_n = len(valid)
    valid_rate = valid_n / n
    conditional = correct / valid_n if valid_n else float("nan")
    raw = correct / n
    valid_interval = wilson_interval(valid_n, n)
    cond_interval = wilson_interval(correct, valid_n) if valid_n else None
    raw_interval = wilson_interval(correct, n)

    by_item: dict[str, list[ExternalResult]] = defaultdict(list)
    for row in rows:
        by_item[row.item_id].append(row)
    item_outcomes: dict[str, str] = {}
    paired = Counter()
    seed_correct = [0, 0]
    seed_valid = [0, 0]
    for item_id, item_rows in sorted(by_item.items()):
        ordered = sorted(item_rows, key=lambda row: row.rollout_seed)
        if len(ordered) >= 2:
            first, second = ordered[:2]
            first_valid = first.status in {
                ExternalStatus.VALID_CORRECT,
                ExternalStatus.VALID_WRONG,
            }
            second_valid = second.status in {
                ExternalStatus.VALID_CORRECT,
                ExternalStatus.VALID_WRONG,
            }
            if first_valid and second_valid:
                for position, row in enumerate((first, second)):
                    seed_valid[position] += 1
                    seed_correct[position] += int(row.correct)
                key = ("cc", "cw", "wc", "ww")[(not first.correct) * 2 + (not second.correct)]
                paired[key] += 1
                if first.correct and second.correct:
                    item_outcomes[item_id] = "stable-easy"
                elif not first.correct and not second.correct:
                    item_outcomes[item_id] = "stable-hard"
                else:
                    item_outcomes[item_id] = "seed-sensitive"
            else:
                item_outcomes[item_id] = "invalid-or-truncated"
    two_seed_items = sum(paired.values())
    seed_accuracy = tuple(
        seed_correct[position] / seed_valid[position]
        if seed_valid[position]
        else float("nan")
        for position in range(2)
    )
    seed_gap = abs(seed_accuracy[0] - seed_accuracy[1]) if all(seed_valid) else None
    pair_oracle = (
        (paired["cc"] + paired["cw"] + paired["wc"]) / two_seed_items
        if two_seed_items
        else None
    )
    mean_single = (
        (paired["cc"] + 0.5 * (paired["cw"] + paired["wc"])) / two_seed_items
        if two_seed_items
        else None
    )
    gain = (
        pair_oracle - mean_single
        if pair_oracle is not None and mean_single is not None
        else None
    )
    return QualificationSummary(
        n_results=n,
        n_items=len(by_item),
        valid_count=valid_n,
        correct_count=correct,
        wrong_count=wrong,
        invalid_count=invalid,
        truncated_count=truncated,
        runtime_error_count=runtime,
        valid_completion=valid_rate,
        conditional_accuracy=conditional,
        raw_accuracy=raw,
        valid_interval=valid_interval,
        conditional_accuracy_interval=cond_interval,
        raw_accuracy_interval=raw_interval,
        item_outcomes=item_outcomes,
        paired_counts=dict(paired),
        seed_accuracy=seed_accuracy,
        seed_accuracy_gap=seed_gap,
        stable_hard_count=sum(value == "stable-hard" for value in item_outcomes.values()),
        seed_sensitive_count=sum(
            value == "seed-sensitive" for value in item_outcomes.values()
        ),
        pair_oracle_accuracy=pair_oracle,
        resampling_gain=gain,
    )
