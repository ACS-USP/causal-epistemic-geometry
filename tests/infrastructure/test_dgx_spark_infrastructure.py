from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from tools.infrastructure.dgx_spark_doctor import (
    attach_fingerprint,
    canonical_json,
    parse_os_release,
    render_summary,
)
from tools.infrastructure.dgx_spark_sharding import assign_node, partition, recombine
from tools.infrastructure.scan_secrets import scan_paths


def synthetic_report() -> dict:
    return {
        "identity": {
            "hostname": "spark-test",
            "architecture": "aarch64",
            "kernel": {"system": "Linux", "release": "test", "version": "test-version"},
        },
        "os_release": {"ID": "ubuntu", "VERSION_ID": "24.04"},
        "cpu": {"Architecture": "aarch64", "CPU(s)": "20"},
        "memory": {"MemTotal": 128_000_000_000, "MemAvailable": 120_000_000_000},
        "nvidia": {"gpus": [{"name": "NVIDIA GB10", "driver_version": "test"}]},
        "native_nvidia_packages": [{"name": "nvidia-test", "version": "1"}],
        "python": {"version": "3.12.3", "implementation": "CPython", "executable": "/venv/python"},
        "python_packages": [{"name": "torch", "version": "test"}],
        "torch": {
            "installed": True,
            "version": "test",
            "cuda_runtime": "test",
            "cuda_available": True,
            "bf16_supported": True,
        },
        "git": {"head": "a" * 40, "clean": True},
        "shared_storage": {"exists": True},
    }


def test_os_release_parser_does_not_evaluate_shell() -> None:
    parsed = parse_os_release('NAME="Ubuntu"\nVERSION_ID="24.04"\nBAD=$(touch /tmp/nope)\n')
    assert parsed == {"NAME": "Ubuntu", "VERSION_ID": "24.04", "BAD": "$(touch /tmp/nope)"}


def test_environment_serialization_and_fingerprint_are_canonical() -> None:
    left = attach_fingerprint(synthetic_report())
    right = attach_fingerprint(synthetic_report())
    assert left["fingerprint"]["sha256"] == right["fingerprint"]["sha256"]
    metadata = left["fingerprint"]["canonical_metadata"]
    expected = hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest()
    assert left["fingerprint"]["sha256"] == expected
    assert json.loads(json.dumps(left)) == left
    assert "NVIDIA GB10" in render_summary(left)


def logical_keys() -> list[tuple[str, str, str]]:
    return [
        (condition, f"item-{item:02d}", f"rollout-{rollout:02d}")
        for condition in ("technical-a", "technical-b")
        for item in range(11)
        for rollout in range(3)
    ]


def test_sharding_has_complete_unique_stable_coverage() -> None:
    keys = logical_keys()
    first = partition(keys)
    second = partition(reversed(keys))
    flattened = [key for node_keys in first.values() for key in node_keys]
    assert len(flattened) == len(keys)
    assert len(set(flattened)) == len(keys)
    assert set(flattened) == set(keys)
    first_mapping = {key: node for node, node_keys in first.items() for key in node_keys}
    second_mapping = {key: node for node, node_keys in second.items() for key in node_keys}
    assert first_mapping == second_mapping
    for key in keys:
        assert first_mapping[key] == assign_node(key)


def test_sharding_rejects_duplicate_keys() -> None:
    key = ("technical", "item", "rollout")
    with pytest.raises(ValueError, match="duplicate logical key"):
        partition([key, key])


def test_recombination_is_order_independent_and_duplicate_safe() -> None:
    rows = [{"logical_key": list(key), "value": index} for index, key in enumerate(logical_keys())]
    assert recombine(rows) == recombine(reversed(rows))
    with pytest.raises(ValueError, match="duplicate result logical key"):
        recombine([rows[0], rows[0]])


def test_secret_scanner_detects_private_keys_and_tokens(tmp_path: Path) -> None:
    clean = tmp_path / "clean.txt"
    clean.write_text("dstack token committed: NO\n", encoding="utf-8")
    private_key = tmp_path / "key.txt"
    private_key.write_text("-----BEGIN " + "OPENSSH PRIVATE KEY-----\nsecret\n", encoding="utf-8")
    token = tmp_path / "token.txt"
    token.write_text("DSTACK_" + "TOKEN=abcdefghijklmnop123456\n", encoding="utf-8")
    assert scan_paths([clean]) == []
    patterns = {pattern for _, pattern in scan_paths([private_key, token])}
    assert patterns == {"private-key", "dstack-token-assignment"}


def test_dstack_smoke_yaml_has_required_arm_gpu_and_shared_mount() -> None:
    path = Path("infra/dstack/dgx-spark-smoke.dstack.yml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["type"] == "task"
    assert config["resources"]["cpu"] == "arm:2.."
    assert config["resources"]["gpu"] == "GB10:1"
    assert "/srv/shared:/shared" in config["volumes"]
    assert config["image"] == "nvcr.io/nvidia/vllm:26.05-py3"
