"""Pure Gate-6 layer/source/RFM atlas utilities.

The module contains no dataset loading and no model execution.  It owns the
frozen layer set, source prompts, deterministic controller mathematics,
source-only readout metrics, standardized distributed budgets, and the
independent two-rollout estimands used by the Gate-6 runner.

The RFM adapter intentionally imports the pinned optional ``xRFM`` package only
when called.  Its call shape follows the public ``neural_controllers``
implementation at the locked upstream commit; the repository remains usable on
CPU-only development machines.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from epistemic_geometry.reproducibility import stable_seed

MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_REPO = "cruxeval-org/cruxeval"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
LAYERS = (8, 12, 17, 22, 27, 32)
LAYER_PATH = "model.model.layers"
ACTIVATION = "block output residual stream"
TOKEN_POSITION_PROMPT = "final non-padding prompt token"
TOKEN_POSITION_EXECUTION = "first token of final FINAL marker; preceding state"
ALPHA_GATE5 = 8.39900588973121
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20260820

SYSTEM_CAREFUL = (
    "You are a meticulous program tracer. Carefully track every operation, mutation, "
    "intermediate value, branch, and loop. Verify the result before answering. End with "
    "exactly one line in the form FINAL: <answer>."
)
SYSTEM_DIRECT = (
    "Answer the program-output question immediately. Do not trace, deliberate, explain, "
    "or verify. End with exactly one line in the form FINAL: <answer>."
)

SOURCE_LOCATIONS = ("PROMPT_BOUNDARY", "EXECUTION_BOUNDARY")
SOURCE_PHASE = "SOURCE_VALIDATION"
CONTROLLER_MANIPULATION_PHASE = "CONTROLLER_MANIPULATION"
CONTROLLER_EVALUATION_PHASE = "CONTROLLER_EVALUATION"


@dataclass(frozen=True)
class RFMConfig:
    """Outcome-independent parameters passed to the pinned upstream RFM."""

    iters: int = 8
    bandwidth: float = 10.0
    exponent: float = 1.0
    regularization: float = 1e-3
    m_batch_size: int = 2048
    n_components: int = 1
    tuning_metric: str = "auc"
    kernel: str = "l2_high_dim"


def vector_sha256(values: np.ndarray) -> str:
    """Hash canonical float64 vector bytes."""

    return hashlib.sha256(np.asarray(values, dtype=np.float64).reshape(-1).tobytes()).hexdigest()


def matrix_sha256(values: np.ndarray) -> str:
    """Hash canonical float64 matrix bytes and shape."""

    array = np.asarray(values, dtype=np.float64)
    payload = str(array.shape).encode("ascii") + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def unit_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("vector must have finite non-zero norm")
    return vector / norm


def paired_mean_direction(
    careful: np.ndarray, direct: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """Construct a unit careful-minus-direct difference-of-means direction."""

    positive = np.asarray(careful, dtype=np.float64)
    negative = np.asarray(direct, dtype=np.float64)
    if positive.ndim != 2 or negative.shape != positive.shape:
        raise ValueError("careful and direct arrays must have equal two-dimensional shapes")
    raw = np.mean(positive - negative, axis=0)
    direction = unit_vector(raw)
    delta = float(np.mean((positive - negative) @ direction))
    return direction, delta, raw


def orient_direction(
    direction: np.ndarray, careful: np.ndarray, direct: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Orient an arbitrary learned vector toward positive held-out careful gaps."""

    vector = unit_vector(direction)
    gaps = (np.asarray(careful, dtype=np.float64) - np.asarray(direct, dtype=np.float64)) @ vector
    if float(np.mean(gaps)) < 0.0:
        vector = -vector
        gaps = -gaps
    return vector, gaps, float(np.mean(gaps))


def _rfm_factory() -> Any:
    try:
        from xrfm import RFM
    except ImportError as exc:  # pragma: no cover - exercised only in remote env
        raise RuntimeError(
            "Gate 6 requires the pinned xRFM package at commit 773fae8; "
            "it is intentionally not a core local dependency"
        ) from exc
    return RFM


def rfm_agop_direction(
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    validation_y: Any,
    *,
    config: RFMConfig = RFMConfig(),
    rfm_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Fit the pinned neural_controllers RFM and return its top AGOP vector.

    Labels are the source condition only (careful=1, direct=0).  No correctness
    or semantic answer is accepted by this function.
    """

    factory = rfm_factory or _rfm_factory()
    device = getattr(train_x, "device", "cpu")
    model = factory(
        kernel=config.kernel,
        iters=config.iters,
        bandwidth=config.bandwidth,
        exponent=config.exponent,
        bandwidth_mode="constant",
        device=device,
        diag=False,
        verbose=False,
        tuning_metric=config.tuning_metric,
    )
    model.fit(
        (train_x, train_y),
        (validation_x, validation_y),
        iters=config.iters,
        return_best_params=True,
        get_agop_best_model=True,
        M_batch_size=config.m_batch_size,
        reg=config.regularization,
        early_stop_rfm=False,
        verbose=False,
    )
    agop = getattr(model, "agop_best_model", None)
    if agop is None:
        raise RuntimeError("pinned RFM did not expose agop_best_model")
    import torch

    agop_tensor = agop.detach().float()
    eigenvalues, eigenvectors = torch.linalg.eigh(agop_tensor)
    order = torch.argsort(eigenvalues, descending=True)
    top = eigenvectors[:, order[0]].detach().cpu().numpy().astype(np.float64)
    spectrum = eigenvalues[order].detach().cpu().numpy().astype(np.float64)
    return {
        "direction": unit_vector(top),
        "agop": agop_tensor.detach().cpu().numpy().astype(np.float64),
        "eigenvalues": spectrum,
        "rfm": model,
        "config": config.__dict__.copy(),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, with ties receiving their average rank."""

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def auroc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Tie-aware AUROC without an sklearn dependency."""

    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(s) or not np.isin(y, [0, 1]).all():
        raise ValueError("labels and scores must be equal-length binary arrays")
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def source_readout_metrics(
    direction: np.ndarray, careful: np.ndarray, direct: np.ndarray
) -> dict[str, float]:
    """Compute the frozen label-free held-out source validation metrics."""

    vector = unit_vector(direction)
    positive = np.asarray(careful, dtype=np.float64)
    negative = np.asarray(direct, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim != 2:
        raise ValueError("source arrays must have equal two-dimensional shapes")
    scores = np.concatenate((positive @ vector, negative @ vector))
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    gaps = (positive - negative) @ vector
    predictions = scores >= np.median(scores)
    balanced_accuracy = 0.5 * (
        np.mean(predictions[: len(positive)]) + np.mean(~predictions[len(positive) :])
    )
    pooled_scale = float(np.std(scores, ddof=1))
    return {
        "auroc": auroc(labels, scores),
        "balanced_accuracy": float(balanced_accuracy),
        "mean_signed_gap": float(np.mean(gaps)),
        "positive_gap_fraction": float(np.mean(gaps > 0)),
        "projection_effect_size": float(np.mean(gaps) / pooled_scale)
        if pooled_scale
        else float("nan"),
    }


def covariance_spectrum(values: np.ndarray) -> dict[str, Any]:
    """Return the spectral atlas for a sample-by-hidden activation matrix."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("activation matrix must be [samples, hidden] with at least two samples")
    centered = x - x.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(x) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    top16 = eigenvalues[:16]
    positive = eigenvalues[eigenvalues > max(total, 1.0) * 1e-14]
    return {
        "eigenvalues_top16": top16.tolist(),
        "top1_top2_gap": float(eigenvalues[0] - eigenvalues[1])
        if len(eigenvalues) > 1
        else float("nan"),
        "top16_cumulative_energy": float(top16.sum() / total) if total else 0.0,
        "effective_rank": float((positive.sum() ** 2) / np.square(positive).sum())
        if len(positive)
        else 0.0,
        "participation_ratio": float((eigenvalues.sum() ** 2) / np.square(eigenvalues).sum())
        if eigenvalues.any()
        else 0.0,
        "covariance_sha256": matrix_sha256(covariance),
    }


def direction_alignment(left: np.ndarray, right: np.ndarray) -> float:
    a = unit_vector(left)
    b = unit_vector(right)
    return float(abs(np.dot(a, b)))


def standardize_scale(direction: np.ndarray, ordinary_activations: np.ndarray) -> float:
    vector = unit_vector(direction)
    projections = np.asarray(ordinary_activations, dtype=np.float64) @ vector
    scale = float(np.std(projections, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("ordinary activation projection scale must be finite and positive")
    return scale


def standardized_budget(
    direction: np.ndarray, ordinary_activations: np.ndarray, eta: float, n_layers: int = 1
) -> np.ndarray:
    """Return a single-layer or distributed BF16-ready delta with eta energy."""

    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    vector = unit_vector(direction)
    scale = standardize_scale(vector, ordinary_activations)
    return vector * (float(eta) * scale / math.sqrt(n_layers))


def orthogonal_random_bank(
    meaningful: np.ndarray,
    *,
    seeds: Sequence[int],
    additional_basis: Sequence[np.ndarray] = (),
) -> dict[str, np.ndarray]:
    """Create deterministic Gram-Schmidt random controls without outcomes."""

    basis = [unit_vector(meaningful)] + [unit_vector(value) for value in additional_basis]
    output: dict[str, np.ndarray] = {}
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(stable_seed("GATE6-RANDOM-BANK", int(seed), len(basis[0])))
        candidate = rng.standard_normal(len(basis[0])).astype(np.float64)
        for previous in basis:
            candidate -= np.dot(candidate, previous) * previous
        candidate = unit_vector(candidate)
        output[f"R{index}"] = candidate
        basis.append(candidate)
    return output


def source_seed(item_id: str, location: str, condition: str) -> int:
    return stable_seed(
        "GATE6-LAYER-SOURCE-RFM-ATLAS",
        SOURCE_PHASE,
        item_id,
        location,
        condition,
        "INDEPENDENT_PRIMARY",
    )


def evaluation_seed(item_id: str, condition: str, rollout: int) -> int:
    return stable_seed(
        "GATE6-LAYER-SOURCE-RFM-ATLAS",
        CONTROLLER_EVALUATION_PHASE,
        item_id,
        condition,
        rollout,
        "INDEPENDENT_PRIMARY",
    )


def manipulation_seed(item_id: str) -> int:
    return stable_seed(
        "GATE6-LAYER-SOURCE-RFM-ATLAS",
        CONTROLLER_MANIPULATION_PHASE,
        item_id,
        "MATCHED_COUPLING_SECONDARY",
    )


def two_rollout_estimands(
    baseline_errors: np.ndarray, condition_errors: np.ndarray
) -> dict[str, float]:
    """Gate-6 unbiased two-rollout G/C/D/rescue/damage estimands."""

    b = np.asarray(baseline_errors, dtype=np.float64)
    j = np.asarray(condition_errors, dtype=np.float64)
    if b.shape != j.shape or b.ndim != 2 or b.shape[1] != 2 or b.shape[0] < 2:
        raise ValueError("error banks must have shape [items, 2]")
    b1, b2 = b[:, 0], b[:, 1]
    j1, j2 = j[:, 0], j[:, 1]
    q0, qj = b.mean(axis=1), j.mean(axis=1)
    n = len(b)
    b00 = float(np.mean(b1 * b2))
    b0j = float(np.mean((b1 * j1 + b1 * j2 + b2 * j1 + b2 * j2) / 4))
    u00 = float((q0.sum() ** 2 - np.dot(q0, q0)) / (n * (n - 1)))
    u0j = float((q0.sum() * qj.sum() - np.dot(q0, qj)) / (n * (n - 1)))
    rescue = float(np.mean((b1 * (1 - j1) + b1 * (1 - j2) + b2 * (1 - j1) + b2 * (1 - j2)) / 4))
    damage = float(np.mean(((1 - b1) * j1 + (1 - b1) * j2 + (1 - b2) * j1 + (1 - b2) * j2) / 4))
    return {
        "B00": b00,
        "B0j": b0j,
        "O00": 1 - b00,
        "O0j": 1 - b0j,
        "G": b00 - b0j,
        "U00": u00,
        "U0j": u0j,
        "C": b00 - b0j - u00 + u0j,
        "D": float(np.mean(b1 * b2 + j1 * j2 - b1 * j2 - b2 * j1)),
        "rescue": rescue,
        "damage": damage,
        "accuracy_baseline": float(1 - b.mean()),
        "accuracy_condition": float(1 - j.mean()),
    }


def item_cluster_bootstrap(
    baseline: np.ndarray,
    conditions: Mapping[str, np.ndarray],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Descriptive item-cluster percentile intervals for Gate-6 estimands."""

    base = np.asarray(baseline, dtype=np.int8)
    values: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in ("accuracy_change", "G", "C", "D", "rescue", "damage")}
        for name in conditions
    }
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        indices = rng.integers(0, len(base), size=len(base))
        sampled_base = base[indices]
        base_accuracy = 1 - float(sampled_base.mean())
        for name, array in conditions.items():
            result = two_rollout_estimands(sampled_base, np.asarray(array, dtype=np.int8)[indices])
            values[name]["accuracy_change"].append(result["accuracy_condition"] - base_accuracy)
            for metric in ("G", "C", "D", "rescue", "damage"):
                values[name][metric].append(result[metric])
    return {
        name: {
            metric: (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))
            for metric, samples in metrics.items()
        }
        for name, metrics in values.items()
    }


def classify_gate6_movement(
    meaningful: Mapping[str, float],
    *,
    baseline: Mapping[str, float],
    random: Sequence[Mapping[str, float]],
    best_single: Mapping[str, float],
    multilayer_mean: Mapping[str, float],
) -> tuple[bool, bool]:
    """Return movement/useful flags using only frozen Gate-6 thresholds."""

    validity_guard = (
        meaningful["validity"] >= 0.90 and meaningful["validity"] >= baseline["validity"] - 0.05
    )
    competence_guard = meaningful["accuracy"] >= baseline["accuracy"] - 0.10
    random_d = [entry["D"] for entry in random]
    random_c = [entry["C"] for entry in random]
    movement = bool(
        validity_guard
        and competence_guard
        and meaningful["D"] >= 0.05
        and meaningful["D"] - float(np.mean(random_d)) >= 0.05
        and meaningful["D"] > max(random_d)
        and meaningful["D"] - best_single["D"] >= 0.02
        and meaningful["D"] - multilayer_mean["D"] >= 0.02
    )
    useful = bool(
        movement
        and meaningful["G"] >= 0.03
        and meaningful["C"] >= 0.03
        and meaningful["C"] - float(np.mean(random_c)) >= 0.05
        and meaningful["C"] > max(random_c)
    )
    return movement, useful
