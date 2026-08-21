"""Generic multi-layer Gate-6 activation capture and current-token hooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _split_output(output: Any) -> tuple[Any, tuple[Any, ...] | None, bool]:
    """Return hidden output, remainder, and whether the original was a tuple."""

    if hasattr(output, "shape") and len(output.shape) >= 3:
        return output, None, False
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0], tuple(output[1:]), isinstance(output, tuple)
    raise TypeError("Gate-6 layer hook expected a tensor or tuple/list beginning with a tensor")


def _join_output(hidden: Any, remainder: tuple[Any, ...] | None, was_tuple: bool) -> Any:
    if remainder is None:
        return hidden
    return (hidden, *remainder) if was_tuple else [hidden, *remainder]


@dataclass
class Gate6HookTrace:
    """Lifecycle-safe one-shot capture or current-token intervention.

    ``target_positions`` refers to the final non-padding prompt position for a
    prefill.  During a cached decode, the only processed token is position zero.
    """

    layers: Mapping[int, Any]
    deltas: Mapping[int, Any] | None = None
    target_positions: Sequence[int] | None = None
    capture: bool = False
    capture_once: bool = True
    handles: list[Any] = field(default_factory=list, init=False)
    captured: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    applications: list[dict[str, Any]] = field(default_factory=list, init=False)
    forward_counts: dict[int, int] = field(default_factory=dict, init=False)

    def __enter__(self) -> Gate6HookTrace:
        if not self.layers:
            raise ValueError("Gate-6 hook requires at least one layer")
        if self.deltas is not None and set(self.deltas) != set(self.layers):
            raise ValueError("deltas must be provided for exactly the configured layers")
        for layer, module in self.layers.items():
            self.forward_counts[int(layer)] = 0
            self.handles.append(module.register_forward_hook(self._make_hook(int(layer))))
        return self

    def _positions(self, hidden: Any) -> Any:
        if hidden.shape[1] == 1:
            return [0] * hidden.shape[0]
        if self.target_positions is None or len(self.target_positions) != hidden.shape[0]:
            raise ValueError("prefill target_positions must match hook batch size")
        return [int(value) for value in self.target_positions]

    def _make_hook(self, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden, remainder, was_tuple = _split_output(output)
            if hidden.ndim != 3:
                raise ValueError("Gate-6 hook hidden output must be [batch, sequence, hidden]")
            positions = self._positions(hidden)
            self.forward_counts[layer] += 1
            updated = hidden
            if self.deltas is not None:
                updated = hidden.clone()
                delta = self.deltas[layer].to(device=hidden.device, dtype=hidden.dtype)
                for row, position in enumerate(positions):
                    before = hidden[row, position, :].detach().clone()
                    updated[row, position, :] = before + delta
                    after = updated[row, position, :].detach()
                    absolute_error = (after.float() - before.float() - delta.float()).abs()
                    bf16_scale = hidden[row, position, :].float().abs()
                    bf16_scale = (
                        bf16_scale.maximum(after.float().abs())
                        .maximum(delta.float().abs())
                        .clamp_min(1.0)
                    )
                    relative_error = absolute_error / (
                        self._dtype_epsilon(hidden.dtype) * bf16_scale
                    )
                    self.applications.append(
                        {
                            "layer": layer,
                            "batch_row": row,
                            "sequence_length": int(hidden.shape[1]),
                            "token_position": int(position),
                            "shift_error": float(absolute_error.max().item()),
                            "relative_shift_error": float(relative_error.max().item()),
                            "non_current_change": float(
                                (updated[row, :, :] - hidden[row, :, :]).abs().sum().item()
                                - (after - before).abs().sum().item()
                            ),
                        }
                    )
            if self.capture and (not self.captured or not self.capture_once):
                self.captured[layer] = np.stack(
                    [
                        updated[row, position, :].detach().float().cpu().numpy()
                        for row, position in enumerate(positions)
                    ]
                )
            return _join_output(updated, remainder, was_tuple)

        return hook

    @staticmethod
    def _dtype_epsilon(dtype: Any) -> float:
        import torch

        return float(torch.finfo(dtype).eps)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    @property
    def forward_count(self) -> int:
        values = set(self.forward_counts.values())
        if len(values) > 1:
            raise RuntimeError(f"layer forward counts diverged: {self.forward_counts}")
        return next(iter(values), 0)

    def metadata(self) -> dict[str, Any]:
        return {
            "layers": sorted(self.layers),
            "forward_counts": dict(self.forward_counts),
            "forward_count": self.forward_count,
            "applications": list(self.applications),
            "capture_layers": sorted(self.captured),
        }
