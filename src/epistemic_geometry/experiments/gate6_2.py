"""CPU-only contracts for the Gate 6.2 first-stage repair.

Gate 6.1 remains historical.  This module contains the new causal scoring
window and the deterministic, source-only cross-validation bookkeeping used by
the prospective Gate 6.2 runner.  It deliberately has no model or dataset
imports so the contracts can be tested on a laptop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TeacherForcedScoreWindow:
    """Continuation-token interval scored by a teacher-forced intervention."""

    intervention_token_index: int
    score_start_index: int
    score_end_index: int

    @property
    def scored_token_count(self) -> int:
        return self.score_end_index - self.score_start_index

    def as_dict(self) -> dict[str, int]:
        return {
            "intervention_token_index": self.intervention_token_index,
            "score_start_index": self.score_start_index,
            "score_end_index": self.score_end_index,
            "scored_token_count": self.scored_token_count,
        }


def teacher_forced_score_window(
    *,
    source_location: str,
    continuation_length: int,
    marker_token_index: int | None = None,
) -> TeacherForcedScoreWindow:
    """Return the causal continuation interval for a source intervention.

    ``marker_token_index`` is indexed in the generated continuation and points
    to the first token of the final ``FINAL`` marker.  The execution-boundary
    intervention is applied to the state immediately before that token; its
    score therefore starts at the marker token and excludes every earlier
    continuation token.  The end index is exclusive.
    """

    if continuation_length <= 0:
        raise ValueError("continuation_length must be positive")
    if source_location == "PROMPT_BOUNDARY":
        intervention_index = 0
    elif source_location == "EXECUTION_BOUNDARY":
        if marker_token_index is None:
            raise ValueError("execution-boundary scoring requires marker_token_index")
        intervention_index = int(marker_token_index)
    else:
        raise ValueError(f"unknown source location: {source_location}")
    if intervention_index < 0 or intervention_index >= continuation_length:
        raise ValueError("intervention token index is outside the continuation")
    return TeacherForcedScoreWindow(
        intervention_token_index=intervention_index,
        score_start_index=intervention_index,
        score_end_index=continuation_length,
    )


def score_teacher_forced_window(
    log_probs: np.ndarray,
    target_token_ids: Sequence[int],
    window: TeacherForcedScoreWindow,
) -> tuple[float, np.ndarray]:
    """Select only causal token log-probabilities and return their mean.

    ``log_probs[i]`` must be the distribution used to score continuation token
    ``target_token_ids[i]``.  The function reports the per-token values as well
    as the normalized mean, making the scoring window auditable.
    """

    values = np.asarray(log_probs, dtype=np.float64)
    targets = np.asarray(target_token_ids, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != len(targets):
        raise ValueError("log_probs must be [continuation_tokens, vocabulary]")
    if not 0 <= window.score_start_index < window.score_end_index <= len(targets):
        raise ValueError("score window is outside target continuation")
    row_indices = np.arange(window.score_start_index, window.score_end_index)
    selected = values[
        row_indices,
        targets[window.score_start_index : window.score_end_index],
    ]
    if selected.size == 0:
        raise ValueError("score window cannot be empty")
    return float(np.mean(selected)), selected.copy()


def stratified_kfold_indices(
    labels: Sequence[int], *, n_splits: int, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create deterministic stratified folds without using model outcomes."""

    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("labels cannot be empty")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    classes, counts = np.unique(values, return_counts=True)
    if np.any(counts < n_splits):
        raise ValueError("every class must have at least n_splits examples")
    rng = np.random.default_rng(int(seed))
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in classes:
        indices = np.flatnonzero(values == cls)
        shuffled = indices[rng.permutation(len(indices))]
        for offset, index in enumerate(shuffled):
            buckets[offset % n_splits].append(int(index))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(values), dtype=np.int64)
    for bucket in buckets:
        validation = np.asarray(sorted(bucket), dtype=np.int64)
        mask = np.ones(len(values), dtype=bool)
        mask[validation] = False
        train = all_indices[mask]
        folds.append((train, validation))
    return tuple(folds)


def source_cv_config_grid(configs: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Canonicalize a frozen RFM config grid with deterministic lexical order."""

    if not configs:
        raise ValueError("RFM config grid cannot be empty")
    required = {"iters", "bandwidth", "exponent", "regularization"}
    normalized: list[dict[str, Any]] = []
    for config in configs:
        if not required.issubset(config):
            raise ValueError(f"RFM config is missing {sorted(required - set(config))}")
        normalized.append(dict(config))
    return tuple(
        sorted(
            normalized,
            key=lambda value: tuple(
                (str(key), repr(value[key])) for key in sorted(value)
            ),
        )
    )


def select_source_cv_config(
    fold_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Select a source-only RFM configuration by mean validation AUROC.

    Ties are broken deterministically by config serialization.  ``fold_results``
    contains only source labels and inner-fold metrics; semantic outcomes are
    intentionally not representable in this function's input contract.
    """

    if not fold_results:
        raise ValueError("fold_results cannot be empty")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in fold_results:
        if "config" not in result or "auroc" not in result:
            raise ValueError("each fold result needs config and auroc")
        config_key = repr(sorted(dict(result["config"]).items()))
        grouped.setdefault(config_key, []).append(result)
    candidates: list[tuple[float, str, dict[str, Any], list[dict[str, Any]]]] = []
    for key, results in grouped.items():
        scores = np.asarray([float(result["auroc"]) for result in results], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("CV AUROC must be finite")
        candidates.append((float(np.mean(scores)), key, dict(results[0]["config"]), results))
    best_score, _key, config, results = sorted(
        candidates, key=lambda value: (-value[0], value[1])
    )[0]
    best_iters = [result.get("best_iter") for result in results]
    selected = dict(config)
    selected.update(
        {
            "selected_mean_inner_auroc": best_score,
            "selected_best_iter_values": best_iters,
            "cv_folds": len(results),
        }
    )
    return selected


def config_product(
    *,
    iters: Sequence[int],
    bandwidth: Sequence[float],
    exponent: Sequence[float],
    regularization: Sequence[float],
) -> tuple[dict[str, Any], ...]:
    """Build a frozen Cartesian config grid for protocol manifests."""

    return source_cv_config_grid(
        [
            {
                "iters": int(a),
                "bandwidth": float(b),
                "exponent": float(c),
                "regularization": float(d),
            }
            for a, b, c, d in product(iters, bandwidth, exponent, regularization)
        ]
    )


def paired_stratified_kfold_indices(
    n_items: int, *, n_splits: int, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Partition paired careful/direct source items into balanced folds.

    The two source conditions for an item are kept in the same fold.  Since
    every item contributes one careful and one direct example, this is a
    deterministic stratified split for the binary source label without ever
    splitting a pair across folds.
    """

    if n_items < n_splits or n_splits < 2:
        raise ValueError("n_items must be at least n_splits >= 2")
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(rng.permutation(n_items), dtype=np.int64)
    buckets = [shuffled[offset::n_splits] for offset in range(n_splits)]
    return tuple(
        (
            np.sort(
                np.concatenate(
                    [bucket for index, bucket in enumerate(buckets) if index != fold]
                )
            ),
            np.sort(buckets[fold]),
        )
        for fold in range(n_splits)
    )


def orthogonal_random_bank(
    meaningful: np.ndarray,
    *,
    seeds: Sequence[int],
    additional_basis: Sequence[np.ndarray] = (),
    namespace: str = "GATE6-2-RANDOM-MEAN-BANK",
) -> dict[str, np.ndarray]:
    """Create a deterministic Gram-Schmidt random bank in a fixed subspace."""

    basis = [np.asarray(meaningful, dtype=np.float64).reshape(-1)]
    basis.extend(np.asarray(value, dtype=np.float64).reshape(-1) for value in additional_basis)
    basis = [_unit(value) for value in basis]
    output: dict[str, np.ndarray] = {}
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        candidate = rng.standard_normal(len(basis[0])).astype(np.float64)
        for previous in basis:
            candidate -= np.dot(candidate, previous) * previous
        candidate = _unit(candidate)
        output[f"R{index}"] = candidate
        basis.append(candidate)
    return output


def _unit(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("vector must have finite non-zero norm")
    return vector / norm
