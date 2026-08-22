#!/usr/bin/env python3
"""Run Gate 12.1 numerical qualification on synthetic token fixtures only.

The source intentionally contains no benchmark adapter, semantic evaluator, or
historical outcome path.  It never calls ``generate``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.experiments import gate12_1  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402

REVIEW = ROOT / "review/gate12_1_continuous_geometry_engine"
RAW = REVIEW / "raw_engineering"
FP32_TOLERANCE = 1e-6
BF16_TOLERANCE = 1e-2
NEAR_ZERO_ABSOLUTE_TOLERANCE = 1e-7


def build_backend(model_path: str | None) -> HuggingFaceBackend:
    config = BackendConfig(
        type="huggingface",
        model_id=gate12_1.MODEL,
        model_path=model_path,
        model_revision=gate12_1.MODEL_REVISION,
        tokenizer_id=model_path or gate12_1.MODEL,
        tokenizer_revision=gate12_1.MODEL_REVISION,
        device="auto",
        dtype="bf16",
        layer=gate12_1.LAYER,
        layer_path="model.model.layers",
        prompt_mode="raw",
        max_new_tokens=1,
        do_sample=False,
        enable_thinking=False,
        attention_implementation="sdpa",
        execution_mode="serial_reference",
        batch_size=1,
        item_batch_size=1,
        condition_chunk_size=1,
    )
    return HuggingFaceBackend(
        config,
        model_identifier=gate12_1.MODEL,
        tokenizer_identifier=model_path or gate12_1.MODEL,
        model_revision=gate12_1.MODEL_REVISION,
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_is_ancestor(source: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def split_output(output: Any) -> tuple[Any, tuple[Any, ...] | None, bool]:
    if hasattr(output, "shape"):
        return output, None, False
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0], tuple(output[1:]), isinstance(output, tuple)
    raise TypeError("unexpected module output")


def join_output(hidden: Any, remainder: tuple[Any, ...] | None, was_tuple: bool) -> Any:
    if remainder is None:
        return hidden
    return (hidden, *remainder) if was_tuple else [hidden, *remainder]


class AlphaHook(AbstractContextManager["AlphaHook"]):
    def __init__(self, backend: Any, alpha: Any, vector: Any, *, full_mask: Any | None) -> None:
        self.backend = backend
        self.alpha = alpha
        self.vector = vector
        self.full_mask = full_mask
        self.handle: Any | None = None
        self.applications = 0

    def __enter__(self) -> AlphaHook:
        self.handle = self.backend.layer_module(gate12_1.LAYER).register_forward_hook(self.hook)
        return self

    def hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden, remainder, was_tuple = split_output(output)
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        updated = hidden.clone()
        if self.full_mask is None:
            updated[:, -1:, :] = updated[:, -1:, :] + self.alpha.to(hidden.dtype) * vector
        else:
            mask = self.full_mask.to(device=hidden.device, dtype=hidden.dtype).view(1, -1, 1)
            updated = updated + mask * self.alpha.to(hidden.dtype) * vector
        self.applications += 1
        return join_output(updated, remainder, was_tuple)

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class CaptureHooks(AbstractContextManager["CaptureHooks"]):
    """Capture residual, attention, and MLP outputs for every layer/call."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.handles: list[Any] = []
        self.values: dict[str, list[np.ndarray]] = defaultdict(list)

    def _capture(self, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor, _remainder, _was_tuple = split_output(output)
            self.values[name].append(tensor.detach().float().cpu().numpy()[0])

        return hook

    def __enter__(self) -> CaptureHooks:
        for layer_index in range(len(self.backend._layer_stack)):  # noqa: SLF001
            layer = self.backend.layer_module(layer_index)
            self.handles.append(
                layer.register_forward_hook(self._capture(f"L{layer_index}:residual"))
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(self._capture(f"L{layer_index}:attention"))
            )
            self.handles.append(
                layer.mlp.register_forward_hook(self._capture(f"L{layer_index}:mlp"))
            )
        return self

    def concatenated(self) -> dict[str, np.ndarray]:
        return {key: np.concatenate(values, axis=0) for key, values in self.values.items()}

    def __exit__(self, *_args: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def set_attention(backend: Any, implementation: str) -> None:
    backend.model.config._attn_implementation = implementation  # noqa: SLF001
    backend.model.model.config._attn_implementation = implementation  # noqa: SLF001


@contextmanager
def kernel_context(backend: Any, kernel: str) -> Iterator[None]:
    if kernel == "math":
        with backend.torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            yield
    else:
        yield


def tensors(backend: Any, fixture: dict[str, Any]) -> dict[str, Any]:
    torch = backend.torch
    prompt = [int(value) for value in fixture["prompt_token_ids"]]
    continuation = [int(value) for value in fixture["continuation_token_ids"]]
    values = prompt + continuation
    ids = torch.tensor([values], dtype=torch.long, device=backend.device)
    attention = torch.ones_like(ids)
    positions = torch.arange(len(values), dtype=torch.long, device=backend.device)[None, :]
    mask = torch.zeros(len(values), dtype=torch.float32, device=backend.device)
    mask[len(prompt) - 1 :] = 1
    output_positions = torch.arange(
        len(prompt) - 1, len(values), dtype=torch.long, device=backend.device
    )
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "position_ids": positions,
        "intervention_mask": mask,
        "output_positions": output_positions,
    }


def _forward_base(
    backend: Any,
    input_ids: Any,
    attention_mask: Any,
    position_ids: Any,
    *,
    past: Any | None,
    use_cache: bool,
) -> Any:
    start = int(position_ids[0, 0].item())
    cache_position = backend.torch.arange(
        start, start + input_ids.shape[1], dtype=backend.torch.long, device=backend.device
    )
    return backend.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=past,
        use_cache=use_cache,
        return_dict=True,
    )


def full_logits(
    backend: Any,
    fixture: dict[str, Any],
    alpha: Any,
    vector: Any,
    *,
    implementation: str,
    kernel: str,
    hook: bool,
    capture: bool = False,
) -> tuple[Any, dict[str, np.ndarray] | None]:
    data = tensors(backend, fixture)
    set_attention(backend, implementation)
    hook_context: Any = (
        AlphaHook(backend, alpha, vector, full_mask=data["intervention_mask"])
        if hook
        else nullcontext()
    )
    with kernel_context(backend, kernel), hook_context:
        capture_context: Any = CaptureHooks(backend) if capture else nullcontext()
        with capture_context as captured:
            output = _forward_base(
                backend,
                data["input_ids"],
                data["attention_mask"],
                data["position_ids"],
                past=None,
                use_cache=False,
            )
            hidden = output.last_hidden_state[:, data["output_positions"], :]
            logits = backend.model.lm_head(hidden)[0]
        values = captured.concatenated() if capture else None
    return logits, values


def sequential_logits(
    backend: Any,
    fixture: dict[str, Any],
    alpha: Any,
    vector: Any,
    *,
    implementation: str,
    kernel: str,
    hook: bool,
    capture: bool = False,
) -> tuple[Any, dict[str, np.ndarray] | None]:
    torch = backend.torch
    prompt = [int(value) for value in fixture["prompt_token_ids"]]
    continuation = [int(value) for value in fixture["continuation_token_ids"]]
    set_attention(backend, implementation)
    hook_context: Any = (
        AlphaHook(backend, alpha, vector, full_mask=None) if hook else nullcontext()
    )
    all_logits = []
    past = None
    with kernel_context(backend, kernel), hook_context:
        capture_context: Any = CaptureHooks(backend) if capture else nullcontext()
        with capture_context as captured:
            calls = [prompt, *[[value] for value in continuation]]
            for call_index, token_values in enumerate(calls):
                total_length = len(prompt) + max(0, call_index)
                ids = torch.tensor([token_values], dtype=torch.long, device=backend.device)
                attention = torch.ones((1, total_length), dtype=torch.long, device=backend.device)
                start = total_length - len(token_values)
                positions = torch.arange(
                    start, total_length, dtype=torch.long, device=backend.device
                )[None, :]
                output = _forward_base(
                    backend,
                    ids,
                    attention,
                    positions,
                    past=past,
                    use_cache=True,
                )
                past = output.past_key_values
                all_logits.append(backend.model.lm_head(output.last_hidden_state[:, -1:, :])[0, 0])
            values = captured.concatenated() if capture else None
    return torch.stack(all_logits), values


def numpy_logits(values: Any) -> np.ndarray:
    return values.detach().float().cpu().numpy()


def comparison_rows(
    fixture: dict[str, Any],
    comparison: str,
    alpha_label: str,
    left: np.ndarray,
    right: np.ndarray,
) -> list[dict[str, Any]]:
    targets = np.asarray(fixture["target_token_ids"], dtype=np.int64)
    js = gate12_1.js_divergence(left, right)
    left_target = gate12_1.target_logp(left, targets)
    right_target = gate12_1.target_logp(right, targets)
    rows = []
    for index in range(len(left)):
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "comparison": comparison,
                "alpha": alpha_label,
                "output_index": index,
                "top1_agreement": int(np.argmax(left[index]) == np.argmax(right[index])),
                "vocabulary_js": float(js[index]),
                "target_logp_abs_difference": float(abs(left_target[index] - right_target[index])),
                "logit_cosine": gate12_1.cosine(left[index], right[index]),
                "max_abs_logit_difference": float(np.max(np.abs(left[index] - right[index]))),
            }
        )
    return rows


def hash_state(model: Any, torch: Any, *, cast_back_bf16: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        tensor = value.detach()
        if cast_back_bf16 and tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16)
        tensor = tensor.contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def first_divergence_rows(
    fixture: dict[str, Any],
    dtype_label: str,
    sequential: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(sequential) & set(full)):
        left = sequential[key]
        right = full[key]
        if left.shape != right.shape:
            raise RuntimeError(f"capture shape mismatch {key}: {left.shape} != {right.shape}")
        layer, component = key.split(":")
        for token_index in range(len(left)):
            difference = np.abs(left[token_index].astype(np.float64) - right[token_index])
            rows.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "dtype": dtype_label,
                    "layer": int(layer[1:]),
                    "component": component,
                    "token_index": token_index,
                    "max_abs_difference": float(np.max(difference)),
                    "rms_difference": float(np.sqrt(np.mean(difference**2))),
                }
            )
    return rows


def exact_derivative_case(
    backend: Any,
    fixture: dict[str, Any],
    vector: np.ndarray,
    implementation: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = backend.torch
    vector_tensor = torch.tensor(vector, dtype=torch.float32, device=backend.device)
    output_index = 0

    def function(alpha: Any) -> Any:
        logits, _capture = full_logits(
            backend,
            fixture,
            alpha,
            vector_tensor,
            implementation=implementation,
            kernel="math" if implementation == "sdpa" else "eager",
            hook=True,
        )
        return logits[output_index]

    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        primal_alpha = torch.tensor(0.0, dtype=torch.float32, device=backend.device)
        tangent_alpha = torch.tensor(1.0, dtype=torch.float32, device=backend.device)
        dual_alpha = torch.autograd.forward_ad.make_dual(primal_alpha, tangent_alpha)
        dual_logits = function(dual_alpha)
        primal, forward_jvp = torch.autograd.forward_ad.unpack_dual(dual_logits)
    alpha0 = torch.tensor(0.0, dtype=torch.float32, device=backend.device)
    tangent = torch.tensor(1.0, dtype=torch.float32, device=backend.device)
    independent_primal, independent_jvp = torch.autograd.functional.jvp(
        function, alpha0, tangent, create_graph=False, strict=True
    )
    cotangent_rng = np.random.default_rng(
        12_140_000 + int(fixture["fixture_id"].rsplit("_", 1)[1])
    )
    cotangent_np = cotangent_rng.standard_normal(primal.numel()).astype(np.float32)
    cotangent_np /= np.linalg.norm(cotangent_np)
    cotangent = torch.tensor(cotangent_np, device=backend.device)
    alpha_vjp = torch.tensor(0.0, dtype=torch.float32, device=backend.device, requires_grad=True)
    vjp_logits = function(alpha_vjp)
    vjp_scalar = torch.sum(vjp_logits.double() * cotangent.double())
    vjp = torch.autograd.grad(vjp_scalar, alpha_vjp)[0]
    jvp_dot = torch.sum(forward_jvp.double() * cotangent.double())

    baseline64 = primal.detach().double()
    p0 = torch.softmax(baseline64, dim=-1).detach()
    logp0 = torch.log_softmax(baseline64, dim=-1).detach()

    def kl_function(alpha: Any) -> Any:
        moved = function(alpha).double()
        return torch.sum(p0 * (logp0 - torch.log_softmax(moved, dim=-1)))

    alpha_hessian = torch.tensor(
        0.0, dtype=torch.float32, device=backend.device, requires_grad=True
    )
    kl_value = kl_function(alpha_hessian)
    kl_first = torch.autograd.grad(kl_value, alpha_hessian, create_graph=True)[0]
    q_hessian = torch.autograd.grad(kl_first, alpha_hessian)[0]

    target = int(fixture["target_token_ids"][output_index])
    alpha_utility = torch.tensor(
        0.0, dtype=torch.float32, device=backend.device, requires_grad=True
    )
    utility_value = torch.log_softmax(function(alpha_utility).double(), dim=-1)[target]
    utility_autograd = torch.autograd.grad(utility_value, alpha_utility)[0]

    baseline = numpy_logits(primal)
    derivative = numpy_logits(forward_jvp)
    independent = numpy_logits(independent_jvp)
    q_jvp = float(gate12_1.fisher_energy(baseline, derivative))
    u_jvp = gate12_1.utility_slope(baseline, derivative, target)
    finite_rows = []
    finite_derivatives = []
    for epsilon in gate12_1.EPSILONS:
        with torch.inference_mode():
            plus = numpy_logits(function(torch.tensor(epsilon, device=backend.device)))
            minus = numpy_logits(function(torch.tensor(-epsilon, device=backend.device)))
        finite = (plus.astype(np.float64) - minus.astype(np.float64)) / (2 * epsilon)
        finite_derivatives.append(finite.astype(np.float32))
        q_finite = float(gate12_1.fisher_energy(baseline, finite))
        u_finite = gate12_1.utility_slope(baseline, finite, target)
        local_q = 2 * gate12_1.local_kl(baseline, plus) / epsilon**2
        finite_rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "direction_index": fixture["direction_index"],
                "epsilon": epsilon,
                "jvp_cosine": gate12_1.cosine(derivative, finite),
                "jvp_norm_relative_error": gate12_1.relative_error(
                    np.linalg.norm(finite), np.linalg.norm(derivative)
                ),
                "fisher_relative_error": gate12_1.relative_error(q_finite, q_jvp),
                "utility_relative_error": gate12_1.relative_error(
                    u_finite, u_jvp, floor=NEAR_ZERO_ABSOLUTE_TOLERANCE
                ),
                "local_kl_relative_error": gate12_1.relative_error(local_q, q_jvp),
                "q_finite": q_finite,
                "u_finite": u_finite,
                "local_kl_q": local_q,
            }
        )
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"derivative__{fixture['fixture_id']}.npz"
    np.savez_compressed(
        raw_path,
        baseline_logits=baseline.astype(np.float32),
        forward_jvp=derivative.astype(np.float32),
        independent_jvp=independent.astype(np.float32),
        cotangent=cotangent_np,
        finite_derivatives=np.stack(finite_derivatives),
        epsilons=np.asarray(gate12_1.EPSILONS),
        target_token_id=np.asarray(target, dtype=np.int64),
    )
    exact = {
        "fixture_id": fixture["fixture_id"],
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha256(raw_path),
        "forward_independent_primal_max_abs": float(
            torch.max(torch.abs(primal.float() - independent_primal.float())).item()
        ),
        "forward_independent_jvp_cosine": gate12_1.cosine(derivative, independent),
        "forward_jvp_norm": float(np.linalg.norm(derivative)),
        "independent_jvp_norm": float(np.linalg.norm(independent)),
        "relative_jvp_norm_difference": gate12_1.relative_error(
            np.linalg.norm(independent), np.linalg.norm(derivative)
        ),
        "jvp_vjp_left": float(jvp_dot.item()),
        "jvp_vjp_right": float(vjp.item()),
        "jvp_vjp_relative_error": gate12_1.relative_error(
            float(jvp_dot.item()), float(vjp.item())
        ),
        "q_jvp": q_jvp,
        "q_hessian": float(q_hessian.item()),
        "q_hessian_relative_error": gate12_1.relative_error(float(q_hessian.item()), q_jvp),
        "u_jvp": u_jvp,
        "u_autograd": float(utility_autograd.item()),
        "u_autograd_relative_error": gate12_1.relative_error(
            float(utility_autograd.item()), u_jvp, floor=NEAR_ZERO_ABSOLUTE_TOLERANCE
        ),
    }
    return exact, finite_rows


def run(args: argparse.Namespace) -> int:
    require_remote_hf_execution("Gate 12.1 numerical qualification")
    if not source_is_ancestor(args.experiment_source_commit):
        raise RuntimeError("Gate 12.1 source commit is not an ancestor of the execution checkout")
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    if lock["experiment_source_commit"] != args.experiment_source_commit:
        raise RuntimeError("Gate 12.1 source lock mismatch")
    fixtures_payload = read_json(REVIEW / "ENGINEERING_FIXTURES.json")
    fixtures = fixtures_payload["fixtures"]
    directions = np.load(REVIEW / "ENGINEERING_DIRECTIONS.npz", allow_pickle=False)["directions"]
    backend = build_backend(args.model_path)
    torch = backend.torch
    started = time.time()
    before_hash = hash_state(backend.model, torch)
    sequence_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []
    bridge_cache: dict[tuple[str, str], np.ndarray] = {}
    capture_ids = {fixtures[index]["fixture_id"] for index in (0, 7, 9, 10)}

    for fixture in fixtures:
        vector = torch.tensor(
            directions[int(fixture["direction_index"])],
            dtype=torch.float32,
            device=backend.device,
        )
        with torch.inference_mode():
            e0_nohook, _ = sequential_logits(
                backend,
                fixture,
                torch.tensor(0.0, device=backend.device),
                vector,
                implementation="sdpa",
                kernel="default",
                hook=False,
            )
            e1_nohook, _ = sequential_logits(
                backend,
                fixture,
                torch.tensor(0.0, device=backend.device),
                vector,
                implementation="sdpa",
                kernel="math",
                hook=False,
            )
            e2_nohook, _ = full_logits(
                backend,
                fixture,
                torch.tensor(0.0, device=backend.device),
                vector,
                implementation="sdpa",
                kernel="math",
                hook=False,
            )
        e0_no = numpy_logits(e0_nohook)
        e1_no = numpy_logits(e1_nohook)
        e2_no = numpy_logits(e2_nohook)
        sequence_rows.extend(comparison_rows(fixture, "E0_vs_E1", "NO_HOOK", e0_no, e1_no))
        sequence_rows.extend(comparison_rows(fixture, "E1_vs_E2", "NO_HOOK", e1_no, e2_no))
        for alpha in gate12_1.ALL_ALPHAS:
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=backend.device)
            with torch.inference_mode():
                e0, _ = sequential_logits(
                    backend,
                    fixture,
                    alpha_tensor,
                    vector,
                    implementation="sdpa",
                    kernel="default",
                    hook=True,
                )
                e1, e1_capture = sequential_logits(
                    backend,
                    fixture,
                    alpha_tensor,
                    vector,
                    implementation="sdpa",
                    kernel="math",
                    hook=True,
                    capture=fixture["fixture_id"] in capture_ids and alpha == 0.0,
                )
                e2, e2_capture = full_logits(
                    backend,
                    fixture,
                    alpha_tensor,
                    vector,
                    implementation="sdpa",
                    kernel="math",
                    hook=True,
                    capture=fixture["fixture_id"] in capture_ids and alpha == 0.0,
                )
            e0_np, e1_np, e2_np = map(numpy_logits, (e0, e1, e2))
            label = format(alpha, ".12g")
            sequence_rows.extend(comparison_rows(fixture, "E0_vs_E1", label, e0_np, e1_np))
            sequence_rows.extend(comparison_rows(fixture, "E1_vs_E2", label, e1_np, e2_np))
            if alpha in (0.0, gate12_1.D75_ALPHA):
                bridge_cache[(fixture["fixture_id"], label)] = e0_np
            if e1_capture is not None and e2_capture is not None:
                divergence_rows.extend(
                    first_divergence_rows(fixture, "BF16", e1_capture, e2_capture)
                )
            if alpha == 0.0:
                sequence_rows.extend(
                    comparison_rows(fixture, "E0_NOHOOK_vs_ALPHA0", label, e0_no, e0_np)
                )
                sequence_rows.extend(
                    comparison_rows(fixture, "E1_NOHOOK_vs_ALPHA0", label, e1_no, e1_np)
                )
                sequence_rows.extend(
                    comparison_rows(fixture, "E2_NOHOOK_vs_ALPHA0", label, e2_no, e2_np)
                )

    backend.model.float()
    backend.device = next(backend.model.parameters()).device
    after_hash = hash_state(backend.model, torch)
    roundtrip_hash = hash_state(backend.model, torch, cast_back_bf16=True)
    if roundtrip_hash != before_hash:
        raise RuntimeError("FP32 lift is not an exact cast of the loaded BF16 values")

    eager_available = True
    eager_error = None
    try:
        fixture = fixtures[0]
        vector = torch.tensor(directions[0], dtype=torch.float32, device=backend.device)
        with torch.inference_mode():
            full_logits(
                backend,
                fixture,
                torch.tensor(0.0, device=backend.device),
                vector,
                implementation="eager",
                kernel="eager",
                hook=False,
            )
    except Exception as exc:  # pragma: no cover - remote architecture-dependent path
        eager_available = False
        eager_error = f"{type(exc).__name__}: {exc}"
        torch.cuda.empty_cache()
    fp32_implementation = "eager" if eager_available else "sdpa"
    fp32_kernel = "eager" if eager_available else "math"
    write_json(
        REVIEW / "ENGINE_AVAILABILITY.json",
        {
            "eager_available": eager_available,
            "eager_error": eager_error,
            "frozen_fallback_rule": "use FP32 SDPA math only if eager availability probe fails",
            "selected_fp32_attention": fp32_implementation,
            "qualification_values_inspected_before_selection": False,
        },
    )

    for fixture in fixtures:
        vector = torch.tensor(
            directions[int(fixture["direction_index"])],
            dtype=torch.float32,
            device=backend.device,
        )
        with torch.inference_mode():
            e3_nohook, _ = sequential_logits(
                backend,
                fixture,
                torch.tensor(0.0, device=backend.device),
                vector,
                implementation=fp32_implementation,
                kernel=fp32_kernel,
                hook=False,
            )
            e4_nohook, _ = full_logits(
                backend,
                fixture,
                torch.tensor(0.0, device=backend.device),
                vector,
                implementation=fp32_implementation,
                kernel=fp32_kernel,
                hook=False,
            )
        e3_no, e4_no = map(numpy_logits, (e3_nohook, e4_nohook))
        sequence_rows.extend(comparison_rows(fixture, "E3_vs_E4", "NO_HOOK", e3_no, e4_no))
        for alpha in gate12_1.ALL_ALPHAS:
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=backend.device)
            with torch.inference_mode():
                e3, e3_capture = sequential_logits(
                    backend,
                    fixture,
                    alpha_tensor,
                    vector,
                    implementation=fp32_implementation,
                    kernel=fp32_kernel,
                    hook=True,
                    capture=fixture["fixture_id"] in capture_ids and alpha == 0.0,
                )
                e4, e4_capture = full_logits(
                    backend,
                    fixture,
                    alpha_tensor,
                    vector,
                    implementation=fp32_implementation,
                    kernel=fp32_kernel,
                    hook=True,
                    capture=fixture["fixture_id"] in capture_ids and alpha == 0.0,
                )
            e3_np, e4_np = map(numpy_logits, (e3, e4))
            label = format(alpha, ".12g")
            sequence_rows.extend(comparison_rows(fixture, "E3_vs_E4", label, e3_np, e4_np))
            if alpha in (0.0, gate12_1.D75_ALPHA):
                e0_np = bridge_cache[(fixture["fixture_id"], label)]
                sequence_rows.extend(comparison_rows(fixture, "E0_vs_E3", label, e0_np, e3_np))
            if e3_capture is not None and e4_capture is not None:
                divergence_rows.extend(
                    first_divergence_rows(fixture, "FP32", e3_capture, e4_capture)
                )
            if alpha == 0.0:
                sequence_rows.extend(
                    comparison_rows(fixture, "E3_NOHOOK_vs_ALPHA0", label, e3_no, e3_np)
                )
                sequence_rows.extend(
                    comparison_rows(fixture, "E4_NOHOOK_vs_ALPHA0", label, e4_no, e4_np)
                )

    write_csv(REVIEW / "ENGINE_MATRIX_RESULTS.csv", sequence_rows)
    write_csv(REVIEW / "FIRST_DIVERGENCE_ARRAYS.csv", divergence_rows)

    exact_rows = []
    finite_rows = []
    for fixture in fixtures:
        vector = directions[int(fixture["direction_index"])]
        exact, finite = exact_derivative_case(
            backend, fixture, vector, implementation=fp32_implementation
        )
        exact_rows.append(exact)
        finite_rows.extend(finite)
        write_json(REVIEW / "DERIVATIVE_PROGRESS.json", {"completed": len(exact_rows)})
        torch.cuda.empty_cache()
    write_json(REVIEW / "EXACT_DERIVATIVE_RAW_SUMMARY.json", {"fixtures": exact_rows})
    write_csv(REVIEW / "FINITE_DIFFERENCE_LADDER.csv", finite_rows)
    write_json(
        REVIEW / "REMOTE_RUN_METADATA.json",
        {
            "source_commit": args.experiment_source_commit,
            "model": gate12_1.MODEL,
            "revision": gate12_1.MODEL_REVISION,
            "parameter_hash_bf16": before_hash,
            "parameter_hash_fp32_lift": after_hash,
            "parameter_hash_fp32_roundtrip_bf16": roundtrip_hash,
            "fp32_attention": fp32_implementation,
            "scientific_items_processed": 0,
            "historical_outcomes_revealed": False,
            "free_generation_outputs": 0,
            "elapsed_seconds": time.time() - started,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-source-commit", required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
