"""Optional runtime accelerators with conservative fail-closed behavior."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def maybe_compile(function: Callable[..., Any], enabled: bool) -> Callable[..., Any]:
    """Compile a stable-shape callable only when explicitly requested."""

    if not enabled:
        return function
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch.compile requested but Torch is not installed") from exc
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requested but this Torch build has no compile API")
    return torch.compile(function, dynamic=False)


class CudaGraphRunner:
    """Small boundary for an optional fixed-shape CUDA graph prototype."""

    def __init__(self, enabled: bool) -> None:
        try:
            import torch
        except ImportError as exc:
            if enabled:
                raise RuntimeError("CUDA graphs requested but Torch is not installed") from exc
            self.enabled = False
            return
        self.enabled = bool(enabled)
        if self.enabled and not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs requested but CUDA is unavailable")
        self._graph = None

    def capture(self, function: Callable[[], Any]) -> None:
        """Capture only after callers provide stable buffers and shapes."""

        if not self.enabled:
            raise RuntimeError("CUDA graph capture was not enabled")
        import torch

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        self._graph = graph

    def replay(self) -> None:
        if not self.enabled or self._graph is None:
            raise RuntimeError("No CUDA graph has been captured")
        self._graph.replay()
