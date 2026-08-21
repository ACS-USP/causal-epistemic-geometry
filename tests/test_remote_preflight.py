from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from epistemic_geometry.research.preflight import (
    SystemProbe,
    load_environment_spec,
    run_preflight,
)


class FakeProbe(SystemProbe):
    def __init__(
        self,
        *,
        packages: Mapping[str, str | None],
        cuda: bool = True,
        dirty: bool = False,
    ) -> None:
        self.packages = dict(packages)
        self.cuda = cuda
        self.dirty = dirty

    def python_version(self) -> str:
        return "3.11.10"

    def package_version(self, distribution: str) -> str | None:
        return self.packages.get(distribution)

    def torch_facts(self) -> Mapping[str, Any]:
        return {
            "installed": self.packages.get("torch") is not None,
            "cuda_available": self.cuda,
            "cuda_runtime": "12.4" if self.cuda else None,
            "gpu_models": ["NVIDIA A40"] if self.cuda else [],
        }

    def disk_free_gb(self, path: Path) -> float:
        del path
        return 100.0

    def git_facts(self, root: Path) -> Mapping[str, Any]:
        del root
        return {"commit": "abc1234", "dirty": self.dirty}


SPEC = {
    "python": ">=3.11,<3.12",
    "packages": {
        "torch": {"distribution": "torch", "version": "==2.4.1+cu124"},
        "transformers": {"distribution": "transformers", "version": "==4.57.1"},
        "accelerate": {"distribution": "accelerate", "version": "==1.14.0"},
        "huggingface_hub": {"distribution": "huggingface-hub", "version": "==0.36.0"},
    },
    "require_cuda": True,
    "cuda_runtime": "==12.4",
    "minimum_disk_free_gb": 20,
    "hf_cache": {"required": True},
    "models": {"Qwen/Qwen3-8B": {"revision": "model-rev"}},
}


def _packages() -> dict[str, str]:
    return {
        "torch": "2.4.1+cu124",
        "transformers": "4.57.1",
        "accelerate": "1.14.0",
        "huggingface-hub": "0.36.0",
    }


def test_mocked_remote_preflight_validates_all_required_facts(tmp_path) -> None:
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    report = run_preflight(
        SPEC,
        root=tmp_path,
        probe=FakeProbe(packages=_packages()),
        hf_cache_path=cache,
        model_id="Qwen/Qwen3-8B",
        model_revision="model-rev",
        expected_source_commit="abc1234",
    )
    assert report.ready
    assert {check.name for check in report.checks} >= {
        "python_version",
        "torch_cuda_available",
        "cuda_runtime",
        "gpu_model",
        "disk_free_gb",
        "hf_cache_path",
        "model_revision",
        "source_git_commit",
        "source_worktree_clean",
    }


def test_missing_dependency_and_absent_cache_fail_without_network(tmp_path) -> None:
    packages = _packages()
    packages.pop("accelerate")
    report = run_preflight(
        SPEC,
        root=tmp_path,
        probe=FakeProbe(packages=packages),
        hf_cache_path=tmp_path / "absent-cache",
    )
    assert not report.ready
    failed = {check.name for check in report.checks if check.required and not check.passed}
    assert {"package:accelerate", "hf_cache_path"} <= failed


def test_dirty_source_and_wrong_model_revision_fail_closed(tmp_path) -> None:
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    report = run_preflight(
        SPEC,
        root=tmp_path,
        probe=FakeProbe(packages=_packages(), dirty=True),
        hf_cache_path=cache,
        model_id="Qwen/Qwen3-8B",
        model_revision="wrong",
    )
    assert not report.ready
    failed = {check.name for check in report.checks if not check.passed}
    assert {"model_revision", "source_worktree_clean"} <= failed


def test_named_profiles_select_core_and_separate_rfm_environment() -> None:
    root = Path(__file__).resolve().parents[1]
    _payload, core = load_environment_spec(root / "remote_environment.yaml", "CORE_QWEN")
    _payload, rfm = load_environment_spec(root / "remote_environment.yaml", "RFM_COMPAT")
    assert core["packages"]["transformers"]["version"] == "==4.57.1"
    assert core["attention_backend"] == "SDPA"
    assert rfm["packages"]["transformers"]["version"] == "==4.47.0"
    assert core != rfm


def test_version_mismatch_reports_observed_and_expected(tmp_path) -> None:
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    packages = _packages()
    packages["transformers"] = "4.57.6"
    report = run_preflight(
        SPEC,
        root=tmp_path,
        probe=FakeProbe(packages=packages),
        hf_cache_path=cache,
    )
    mismatch = next(check for check in report.checks if check.name == "package:transformers")
    assert not mismatch.passed
    assert mismatch.observed == "4.57.6"
    assert mismatch.expected == "==4.57.1"
