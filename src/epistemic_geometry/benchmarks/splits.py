"""Deterministic category-stratified development split manifests."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.mmlu_pro import MMLUProBenchmark
from epistemic_geometry.reproducibility import canonical_json, stable_digest


def _quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    if total <= 0 or total > sum(counts.values()):
        raise ValueError("Requested split size is outside available item count")
    raw = {category: total * count / sum(counts.values()) for category, count in counts.items()}
    result = {category: min(counts[category], int(value)) for category, value in raw.items()}
    remaining = total - sum(result.values())
    ranked = sorted(
        counts,
        key=lambda category: (-(raw[category] - int(raw[category])), category),
    )
    for category in ranked:
        if remaining == 0:
            break
        if result[category] < counts[category]:
            result[category] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Could not allocate category-stratified split quota")
    return result


def create_mmlu_pro_split_manifest(
    benchmark: MMLUProBenchmark,
    output: str | Path,
    seed: int,
    calibration_size: int = 512,
    evaluation_size: int = 512,
) -> dict[str, Any]:
    """Create calibration/evaluation/holdout IDs without storing labels."""

    items = benchmark.items()
    if benchmark.requested_split != "test":
        raise ValueError("Split manifests must be created from the official MMLU-Pro test split")
    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in items:
        category = str(item.metadata.get("category", "UNKNOWN"))
        by_category[category].append((stable_digest(seed, item.id), item.id))
    for values in by_category.values():
        values.sort()
    counts = {category: len(values) for category, values in by_category.items()}
    calibration_quota = _quotas(counts, calibration_size)
    remaining_counts = {
        category: counts[category] - calibration_quota[category]
        for category in counts
    }
    evaluation_quota = _quotas(remaining_counts, evaluation_size)
    selected: dict[str, list[str]] = {
        "dev_calibration": [],
        "dev_evaluation": [],
        "confirmatory_holdout": [],
    }
    for category in sorted(by_category):
        ids = [item_id for _digest, item_id in by_category[category]]
        calibration_count = calibration_quota[category]
        evaluation_count = evaluation_quota[category]
        selected["dev_calibration"].extend(ids[:calibration_count])
        selected["dev_evaluation"].extend(
            ids[calibration_count : calibration_count + evaluation_count]
        )
        selected["confirmatory_holdout"].extend(ids[calibration_count + evaluation_count :])
    for values in selected.values():
        values.sort()
    manifest: dict[str, Any] = {
        "protocol": "Q1_DEVELOPMENT_PROTOCOL_V1",
        "dataset_id": benchmark.dataset_id,
        "dataset_revision": benchmark.dataset_revision,
        "dataset_fingerprint": benchmark.dataset_fingerprint,
        "source_split": "test",
        "split_seed": seed,
        "sizes": {name: len(values) for name, values in selected.items()},
        "splits": selected,
    }
    manifest["manifest_sha256"] = stable_digest(canonical_json(manifest))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
