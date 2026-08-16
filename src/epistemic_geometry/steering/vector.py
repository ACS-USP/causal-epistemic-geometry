"""Safe, inspectable ``.npz`` vector serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.types import SteeringVector


def vector_hash(values: np.ndarray) -> str:
    """Hash canonical float64 bytes, independent of opaque Python objects."""

    canonical = np.asarray(values, dtype=np.float64).reshape(-1)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _paths(path: str | Path, metadata_path: str | Path | None) -> tuple[Path, Path]:
    vector_path = Path(path)
    if vector_path.suffix != ".npz":
        vector_path = vector_path.with_suffix(".npz")
    meta_path = Path(metadata_path) if metadata_path else vector_path.with_suffix(".json")
    return vector_path, meta_path


def save_vector(
    vector: SteeringVector,
    path: str | Path,
    metadata_path: str | Path | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> tuple[Path, Path]:
    """Save values in NumPy format and provenance in adjacent JSON."""

    vector_path, meta_path = _paths(path, metadata_path)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    digest = vector.hash or vector_hash(vector.values)
    np.savez_compressed(vector_path, values=np.asarray(vector.values, dtype=np.float64))
    model_provenance = vector.metadata.get("model_provenance", {})
    if not isinstance(model_provenance, dict):
        model_provenance = {}
    positive_items = vector.metadata.get("positive_items", [])
    negative_items = vector.metadata.get("negative_items", [])
    source_item_ids = vector.metadata.get(
        "source_item_ids",
        list(positive_items) + list(negative_items)
        if isinstance(positive_items, list) and isinstance(negative_items, list)
        else [],
    )
    metadata: dict[str, Any] = {
        "vector_hash": digest,
        "dimension": vector.dimension,
        "layer": vector.layer,
        "constructor": vector.constructor,
        "normalization": vector.normalization,
        "creation_seed": vector.metadata.get("creation_seed"),
        "source_item_ids": source_item_ids,
        "extraction_policy": vector.metadata.get("extraction_policy", "UNKNOWN"),
        "model_identifier": model_provenance.get("model_identifier", "UNKNOWN"),
        "model_revision": model_provenance.get("model_revision", "UNKNOWN"),
        "tokenizer_identifier": model_provenance.get("tokenizer_identifier", "UNKNOWN"),
        "tokenizer_revision": model_provenance.get("tokenizer_revision", "UNKNOWN"),
        "metadata": vector.metadata,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return vector_path, meta_path


def load_vector(path: str | Path, metadata_path: str | Path | None = None) -> SteeringVector:
    """Load a vector and verify its stored hash before returning it."""

    vector_path, meta_path = _paths(path, metadata_path)
    if not vector_path.exists():
        raise FileNotFoundError(f"Steering vector does not exist: {vector_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Steering vector metadata does not exist: {meta_path}")
    with np.load(vector_path, allow_pickle=False) as archive:
        if "values" not in archive:
            raise ValueError(f"Vector archive lacks 'values': {vector_path}")
        values = np.asarray(archive["values"], dtype=np.float64)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_hash = vector_hash(values)
    if metadata.get("vector_hash") != actual_hash:
        raise ValueError(f"Vector hash mismatch for {vector_path}")
    return SteeringVector(
        values=values,
        layer=int(metadata["layer"]),
        constructor=str(metadata["constructor"]),
        normalization=str(metadata["normalization"]),
        metadata=dict(metadata.get("metadata", {})),
        hash=actual_hash,
    )


def with_computed_hash(vector: SteeringVector) -> SteeringVector:
    """Return an equivalent vector with its content hash populated."""

    return replace(vector, hash=vector_hash(vector.values))
