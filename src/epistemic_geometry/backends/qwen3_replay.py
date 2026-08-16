"""Strict Qwen3 suffix-replay boundary.

Suffix replay is intentionally isolated because it depends on the exact
Transformers Qwen3 decoder-layer and cache APIs.  This module validates the
model before any replay is attempted and fails closed when the installed
stack is not the audited one.  It must not silently approximate a Qwen3 pass.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any


class SuffixReplayUnavailable(RuntimeError):
    """Raised when architecture/version guards do not permit exact replay."""


@dataclass(frozen=True)
class SuffixReplayStatus:
    supported: bool
    reason: str
    model_class: str
    transformers_version: str


class Qwen3CachedSuffixReplayEngine:
    """Architecture gate for a future exact Qwen3 suffix implementation.

    The production path remains ``cached_decode`` until a real Qwen3 model on
    the approved RunPod environment passes numerical and discrete-equivalence
    tests.  Keeping this boundary explicit prevents an attractive but
    scientifically unsafe hand-written attention approximation.
    """

    SUPPORTED_TRANSFORMERS_PREFIX = "4.57."

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        model = backend.model
        config = getattr(model, "config", None)
        model_type = str(getattr(config, "model_type", "UNKNOWN"))
        model_class = model.__class__.__name__
        try:
            transformers_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            transformers_version = "UNKNOWN"
        reasons: list[str] = []
        if not model_type.startswith("qwen3") and "Qwen3" not in model_class:
            reasons.append(f"model is {model_class}/{model_type}, not Qwen3")
        if not transformers_version.startswith(self.SUPPORTED_TRANSFORMERS_PREFIX):
            reasons.append(
                f"Transformers {transformers_version} is not the audited "
                f"{self.SUPPORTED_TRANSFORMERS_PREFIX} series"
            )
        if getattr(backend, "_resolved_layer_path", "") != "model.model.layers":
            reasons.append(
                "resolved layer path is not model.model.layers; set and audit the exact path"
            )
        self.status = SuffixReplayStatus(
            supported=not reasons,
            reason="; ".join(reasons) if reasons else "guards passed; equivalence still required",
            model_class=model_class,
            transformers_version=transformers_version,
        )

    def require_supported(self) -> None:
        """Fail before inference when exact replay is not safe to attempt."""

        if not self.status.supported:
            raise SuffixReplayUnavailable(
                "Qwen3 cached suffix replay is unavailable: " + self.status.reason
            )
        raise SuffixReplayUnavailable(
            "Qwen3 suffix replay guards passed, but the real-model equivalence "
            "audit has not approved this engine; use cached_decode"
        )

    def replay(self, *_args: Any, **_kwargs: Any) -> Any:
        """Reserved exact replay entry point; never returns an approximation."""

        self.require_supported()
