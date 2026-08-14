"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a run configuration is incomplete or scientifically ambiguous."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    stage: str
    seed: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("experiment.name must be non-empty")
        if self.stage not in {"development", "confirmatory"}:
            raise ConfigError("experiment.stage must be 'development' or 'confirmatory'")


@dataclass(frozen=True)
class BackendConfig:
    type: str
    model_id: str | None = None
    model_path: str | None = None
    model_revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    hidden_size: int = 8
    n_classes: int = 4
    device: str = "auto"
    dtype: str = "auto"
    layer: int = 0
    layer_path: str | None = None
    prompt_mode: str = "plain"
    max_new_tokens: int = 4
    do_sample: bool = False
    temperature: float = 1.0
    device_map: str | dict[str, Any] | None = None
    batch_size: int = 1
    quantization: str = "none"

    def __post_init__(self) -> None:
        if self.type not in {"mock", "huggingface", "tiny_transformer"}:
            raise ConfigError("backend.type must be mock, huggingface, or tiny_transformer")
        if not isinstance(self.layer, int) or self.layer < 0:
            raise ConfigError("backend.layer must be a non-negative integer")
        if self.hidden_size <= 0 or self.n_classes <= 1:
            raise ConfigError("backend hidden_size/n_classes are invalid")
        if self.max_new_tokens <= 0:
            raise ConfigError("backend.max_new_tokens must be positive")
        if self.do_sample and self.temperature <= 0:
            raise ConfigError("backend.temperature must be positive when sampling")
        if self.prompt_mode not in {"plain", "chat"}:
            raise ConfigError("backend.prompt_mode must be plain or chat")
        if self.batch_size <= 0:
            raise ConfigError("backend.batch_size must be positive")
        if self.quantization != "none":
            raise ConfigError("Only quantization: none is implemented; choose it explicitly later")


@dataclass(frozen=True)
class BenchmarkConfig:
    type: str
    n_items: int = 32
    path: str | None = None
    max_items: int | None = None
    allowed_targets: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])

    def __post_init__(self) -> None:
        if self.type not in {"mock", "jsonl"}:
            raise ConfigError("benchmark.type must be 'mock' or 'jsonl'")
        if self.type == "mock" and self.n_items <= 0:
            raise ConfigError("benchmark.n_items must be positive")
        if self.type == "jsonl" and not self.path:
            raise ConfigError("benchmark.path is required for JSONL benchmarks")
        if not self.allowed_targets:
            raise ConfigError("benchmark.allowed_targets must not be empty")
        if self.max_items is not None and (
            not isinstance(self.max_items, int) or self.max_items <= 0
        ):
            raise ConfigError("benchmark.max_items must be a positive integer when provided")


@dataclass(frozen=True)
class SteeringConfig:
    enabled: bool = True
    layer: int = 0
    alpha: float | list[float] = 0.0
    token_scope: str = "last_token"
    vector_path: str | None = None
    constructor: str = "random_unit"
    vector_seed: int | None = None
    vector_dimension: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layer, int) or self.layer < 0:
            raise ConfigError("steering.layer must be non-negative")
        if self.token_scope not in {"all_tokens", "last_token"}:
            raise ConfigError("steering.token_scope must be all_tokens or last_token")
        values = self.alpha if isinstance(self.alpha, list) else [self.alpha]
        if not values or not all(isinstance(value, (int, float)) for value in values):
            raise ConfigError("steering.alpha must be a number or list of numbers")

    def alpha_values(self) -> list[float]:
        values = self.alpha if isinstance(self.alpha, list) else [self.alpha]
        return [float(value) for value in values]


@dataclass(frozen=True)
class OutputConfig:
    root: str = "runs"
    save_figures: bool = True


@dataclass(frozen=True)
class RunConfig:
    experiment: ExperimentConfig
    backend: BackendConfig
    benchmark: BenchmarkConfig
    steering: SteeringConfig
    output: OutputConfig
    source_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        return data


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> RunConfig:
    """Load and validate one YAML run configuration."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")

    experiment = ExperimentConfig(**_section(raw, "experiment"))
    backend = BackendConfig(**_section(raw, "backend"))
    benchmark = BenchmarkConfig(**_section(raw, "benchmark"))
    steering = SteeringConfig(**_section(raw, "steering"))
    output = OutputConfig(**_section(raw, "output"))
    return RunConfig(
        experiment=experiment,
        backend=backend,
        benchmark=benchmark,
        steering=steering,
        output=output,
        source_path=str(config_path.resolve()),
    )
