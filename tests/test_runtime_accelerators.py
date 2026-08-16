import pytest

from epistemic_geometry.inference.runtime import CudaGraphRunner, maybe_compile


def test_compile_disabled_is_identity() -> None:
    def function(value):
        return value + 1

    assert maybe_compile(function, enabled=False) is function
    assert function(2) == 3


def test_cuda_graphs_fail_closed_when_disabled_or_unavailable() -> None:
    runner = CudaGraphRunner(enabled=False)
    assert runner.enabled is False
    with pytest.raises(RuntimeError, match="not enabled"):
        runner.capture(lambda: None)
