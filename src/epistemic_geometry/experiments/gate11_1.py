"""Contracts and raw-array metrics for the Gate 11.1 forensic replication.

Gate 11.1 deliberately reuses the historical Gate 11 design.  This module is
new so the artifact-complete computation cannot accidentally fall back to the
historical scalar journal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from epistemic_geometry.experiments import gate11

EXPERIMENT_ID = "GATE11_1_ARTIFACT_COMPLETE_FORENSIC_REPLICATION"
HISTORICAL_EXPERIMENT_ID = gate11.EXPERIMENT_ID
RAW_SCHEMA_VERSION = 1
RAW_DTYPE = "float32"
AUDIT_ABS_TOL = 1e-10
AUDIT_REL_TOL = 1e-8
SNAPSHOT_PREFILL = "PREFILL"
CONDITIONS = gate11.TF_CONDITIONS
PROPAGATION_LAYERS = gate11.PROPAGATION_LAYERS


def snapshot_labels(snapshot_map: Mapping[str, Any]) -> list[str]:
    """Return the frozen historical snapshot order, prefill then checkpoints."""

    labels = [SNAPSHOT_PREFILL]
    labels.extend(str(checkpoint) for checkpoint in gate11.CHECKPOINTS)
    return [label for label in labels if label in snapshot_map]


def snapshot_token_indices(labels: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [-1 if label == SNAPSHOT_PREFILL else int(label) for label in labels], dtype=np.int64
    )


def _log_probs(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.logaddexp.reduce(values, axis=-1, keepdims=True)
    return shifted


def logit_metrics_from_arrays(
    baseline_logits: np.ndarray,
    condition_logits: np.ndarray,
    target_token_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all historical next-token primitives from persisted arrays only."""

    baseline = np.asarray(baseline_logits, dtype=np.float64)
    condition = np.asarray(condition_logits, dtype=np.float64)
    targets = np.asarray(target_token_ids, dtype=np.int64)
    baseline_logp = _log_probs(baseline)
    condition_logp = _log_probs(condition)
    baseline_prob = np.exp(baseline_logp)
    condition_prob = np.exp(condition_logp)
    mixture_logp = np.logaddexp(baseline_logp, condition_logp) - np.log(2.0)
    displacement = condition - baseline
    baseline_top1 = np.argmax(baseline, axis=-1)
    condition_top1 = np.argmax(condition, axis=-1)
    target_shifts = (
        condition_logp[np.arange(len(targets)), targets]
        - baseline_logp[np.arange(len(targets)), targets]
    )
    baseline_norm = np.linalg.norm(baseline, axis=-1)
    condition_norm = np.linalg.norm(condition, axis=-1)
    return {
        "next_token_kl": np.sum(baseline_prob * (baseline_logp - condition_logp), axis=-1),
        "symmetric_js": 0.5 * np.sum(baseline_prob * (baseline_logp - mixture_logp), axis=-1)
        + 0.5 * np.sum(condition_prob * (condition_logp - mixture_logp), axis=-1),
        "logit_l2": np.linalg.norm(displacement, axis=-1),
        "logit_cosine": np.sum(baseline * condition, axis=-1)
        / np.maximum(baseline_norm * condition_norm, np.finfo(np.float64).tiny),
        "top1_flip": (baseline_top1 != condition_top1).astype(np.int8),
        "baseline_top1": baseline_top1.astype(np.int64),
        "condition_top1": condition_top1.astype(np.int64),
        "target_logprob_shift": target_shifts,
        "displacement_norm": np.linalg.norm(displacement, axis=-1),
    }


def hidden_metrics_from_differences(hidden_differences: np.ndarray) -> dict[str, np.ndarray]:
    differences = np.asarray(hidden_differences, dtype=np.float64)
    return {
        f"A{layer}": np.linalg.norm(differences[:, layer_index, :], axis=-1)
        for layer_index, layer in enumerate(PROPAGATION_LAYERS)
    }


def item_metric_rows(
    *,
    baseline_logits: np.ndarray,
    condition_logits: np.ndarray,
    hidden_differences: np.ndarray,
    target_token_ids: np.ndarray,
    condition_index: int,
) -> list[dict[str, Any]]:
    metrics = logit_metrics_from_arrays(
        baseline_logits,
        condition_logits[condition_index],
        target_token_ids,
    )
    metrics.update(hidden_metrics_from_differences(hidden_differences[condition_index]))
    return [
        {
            "snapshot_index": int(index),
            **{
                key: (
                    int(value[index])
                    if key in {"top1_flip", "baseline_top1", "condition_top1"}
                    else float(value[index])
                )
                for key, value in metrics.items()
            },
        }
        for index in range(len(target_token_ids))
    ]


def raw_manifest_entry(
    path: str, *, sha256: str, bytes_count: int, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256,
        "bytes": int(bytes_count),
        "schema_version": RAW_SCHEMA_VERSION,
        "dtype": RAW_DTYPE,
        **dict(metadata),
    }


__all__ = [
    "EXPERIMENT_ID",
    "HISTORICAL_EXPERIMENT_ID",
    "RAW_SCHEMA_VERSION",
    "RAW_DTYPE",
    "AUDIT_ABS_TOL",
    "AUDIT_REL_TOL",
    "SNAPSHOT_PREFILL",
    "CONDITIONS",
    "PROPAGATION_LAYERS",
    "snapshot_labels",
    "snapshot_token_indices",
    "logit_metrics_from_arrays",
    "hidden_metrics_from_differences",
    "item_metric_rows",
    "raw_manifest_entry",
]
