"""Baseline-only E3-10 calibration metrics and mechanical selection rules."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import DIGITS

MIN_ACCURACY = 0.30
MAX_ACCURACY = 0.75
MIN_WORD_AGREEMENT = 0.85
MIN_SURFACE_AGREEMENT = 0.80
MIN_NORMALIZED_ENTROPY = 0.80


@dataclass(frozen=True)
class CalibrationScoreRow:
    """One baseline semantic score vector for one rendered view."""

    latent_id: str
    family: str
    cell: str
    surface: str
    response_channel: str
    target: int
    scores: tuple[float, ...]
    prompt_hash: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.scores) != 10:
            raise ValueError("E3-10 requires exactly ten semantic candidate scores")
        if self.target not in DIGITS:
            raise ValueError("calibration target must be a digit")
        if self.surface not in {"canonical", "surface_twin"}:
            raise ValueError("unknown calibration surface")
        if self.response_channel not in {"decimal", "number_word"}:
            raise ValueError("unknown response channel")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CalibrationScoreRow:
        scores = record.get("scores", record.get("candidate_logits"))
        if scores is None:
            raise ValueError("calibration row is missing scores")
        return cls(
            latent_id=str(record["latent_id"]),
            family=str(record["family"]),
            cell=str(record["cell"]),
            surface=str(record["surface"]),
            response_channel=str(record["response_channel"]),
            target=int(record["target"]),
            scores=tuple(float(value) for value in scores),
            prompt_hash=str(record.get("prompt_hash", "")),
            metadata=dict(record.get("metadata", {})),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "family": self.family,
            "cell": self.cell,
            "surface": self.surface,
            "response_channel": self.response_channel,
            "target": self.target,
            "scores": list(self.scores),
            "prompt_hash": self.prompt_hash,
            "metadata": self.metadata or {},
        }


def semantic_probabilities(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(scores), dtype=np.float64)
    if values.shape != (10,):
        raise ValueError("semantic score vector must have shape (10,)")
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def score_row(row: CalibrationScoreRow) -> dict[str, Any]:
    scores = np.asarray(row.scores, dtype=np.float64)
    probabilities = semantic_probabilities(scores)
    order = np.argsort(-scores, kind="stable")
    prediction = int(order[0])
    top2 = float(scores[order[1]])
    true_margin = float(scores[row.target] - np.max(np.delete(scores, row.target)))
    nll = float(-math.log(max(float(probabilities[row.target]), np.finfo(float).tiny)))
    one_hot = np.zeros(10, dtype=np.float64)
    one_hot[row.target] = 1.0
    entropy = float(
        -(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny))).sum()
    )
    return {
        "latent_id": row.latent_id,
        "prediction": prediction,
        "correct": prediction == row.target,
        "probabilities": probabilities.tolist(),
        "top1_score": float(scores[order[0]]),
        "top2_score": top2,
        "margin": float(scores[order[0]] - top2),
        "true_answer_logit": float(scores[row.target]),
        "best_wrong_logit": float(np.max(np.delete(scores, row.target))),
        "true_answer_margin": true_margin,
        "nll": nll,
        "brier": float(np.square(probabilities - one_hot).sum()),
        "normalized_entropy": entropy / math.log(10),
    }


def _centered_cosine(left: Iterable[float], right: Iterable[float]) -> float | None:
    a = np.asarray(tuple(left), dtype=np.float64)
    b = np.asarray(tuple(right), dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return None
    return float(np.dot(a, b) / denominator)


def _agreement(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["prediction"]) == int(right["prediction"])


def summarize_cell(rows: Iterable[CalibrationScoreRow]) -> dict[str, Any]:
    """Summarize one family/cell without selecting individual examples."""

    materialized = tuple(rows)
    if not materialized:
        raise ValueError("cannot summarize an empty calibration cell")
    keys = {(row.family, row.cell) for row in materialized}
    if len(keys) != 1:
        raise ValueError(f"expected one family/cell, got {keys}")
    scored = [score_row(row) for row in materialized]
    canonical_decimal = {
        row.latent_id: (row, scored[index])
        for index, row in enumerate(materialized)
        if row.surface == "canonical" and row.response_channel == "decimal"
    }
    twin_decimal = {
        row.latent_id: (row, scored[index])
        for index, row in enumerate(materialized)
        if row.surface == "surface_twin" and row.response_channel == "decimal"
    }
    word = {
        row.latent_id: (row, scored[index])
        for index, row in enumerate(materialized)
        if row.surface == "canonical" and row.response_channel == "number_word"
    }
    if not canonical_decimal:
        raise ValueError("cell requires canonical decimal rows")
    baseline = [entry[1] for entry in canonical_decimal.values()]
    predictions = [int(entry["prediction"]) for entry in baseline]
    target_by_id = {row.latent_id: row.target for row in materialized}
    confusion = {str(target): {str(prediction): 0 for prediction in DIGITS} for target in DIGITS}
    for row_id, (_, result) in canonical_decimal.items():
        confusion[str(target_by_id[row_id])][str(result["prediction"])] += 1
    surface_pairs = [
        (canonical_decimal[key], twin_decimal[key])
        for key in canonical_decimal.keys() & twin_decimal.keys()
    ]
    word_pairs = [
        (canonical_decimal[key], word[key]) for key in canonical_decimal.keys() & word.keys()
    ]
    if not surface_pairs or not word_pairs:
        raise ValueError("cell requires paired decimal/twin and decimal/word views")
    surface_agreement = sum(_agreement(left[1], right[1]) for left, right in surface_pairs) / len(
        surface_pairs
    )
    word_agreement = sum(_agreement(left[1], right[1]) for left, right in word_pairs) / len(
        word_pairs
    )
    similarities = [
        _centered_cosine(left[0].scores, right[0].scores) for left, right in surface_pairs
    ]
    similarities = [value for value in similarities if value is not None]
    pred_counts = np.bincount(predictions, minlength=10).astype(float)
    pred_distribution = pred_counts / len(predictions)
    normalized_entropy = float(
        -(
            pred_distribution[pred_distribution > 0]
            * np.log(pred_distribution[pred_distribution > 0])
        ).sum()
        / math.log(10)
    )
    margins = [float(result["margin"]) for result in baseline]
    return {
        "family": materialized[0].family,
        "cell": materialized[0].cell,
        "n_canonical_decimal": len(canonical_decimal),
        "accuracy": float(np.mean([result["correct"] for result in baseline])),
        "nll": float(np.mean([result["nll"] for result in baseline])),
        "brier": float(np.mean([result["brier"] for result in baseline])),
        "median_margin": float(np.median(margins)),
        "margin_quartiles": [float(value) for value in np.quantile(margins, [0.25, 0.5, 0.75])],
        "prediction_distribution": {
            str(digit): float(pred_distribution[digit]) for digit in DIGITS
        },
        "normalized_prediction_entropy": normalized_entropy,
        "confusion_matrix": confusion,
        "surface_twin_agreement": float(surface_agreement),
        "decimal_word_agreement": float(word_agreement),
        "centered_logit_cosine_surface_mean": float(np.mean(similarities))
        if similarities
        else None,
        "qualification": qualify_cell(
            accuracy=float(np.mean([result["correct"] for result in baseline])),
            word_agreement=float(word_agreement),
            surface_agreement=float(surface_agreement),
            normalized_entropy=normalized_entropy,
        ),
    }


def qualify_cell(
    *, accuracy: float, word_agreement: float, surface_agreement: float, normalized_entropy: float
) -> dict[str, Any]:
    reasons = {
        "accuracy": MIN_ACCURACY <= accuracy <= MAX_ACCURACY,
        "decimal_word_agreement": word_agreement >= MIN_WORD_AGREEMENT,
        "surface_twin_agreement": surface_agreement >= MIN_SURFACE_AGREEMENT,
        "normalized_prediction_entropy": normalized_entropy >= MIN_NORMALIZED_ENTROPY,
    }
    return {"qualified": all(reasons.values()), "thresholds": reasons}


def select_cells(summaries: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select the qualifying cell closest to 50% independently per family."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[str(summary["family"])].append(summary)
    selected: dict[str, dict[str, Any]] = {}
    for family, family_summaries in grouped.items():
        candidates = [
            summary for summary in family_summaries if summary["qualification"]["qualified"]
        ]
        for summary in family_summaries:
            summary["selected"] = False
        if candidates:
            chosen = min(
                candidates,
                key=lambda summary: (
                    abs(summary["accuracy"] - 0.5),
                    _difficulty_rank(summary["cell"]),
                ),
            )
            chosen["selected"] = True
            selected[family] = chosen
    return selected


def _difficulty_rank(cell: str) -> int:
    digits = "".join(character for character in cell if character.isdigit())
    return int(digits or "0")
