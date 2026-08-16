"""Tests for local-only RunPod helpers using temporary SSH configuration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(
    script: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ssh_config_helper_is_scoped_idempotent_and_dry_run(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.write_text(
        "Host keep-me\n    HostName keep.example\n\n"
        "Host runpod-a40\n    HostName old.example\n",
        encoding="utf-8",
    )
    dry = _run(
        "configure_runpod_ssh.sh",
        "--ssh-config",
        str(config),
        "--host",
        "203.0.113.10",
        "--port",
        "12345",
        "--dry-run",
    )
    assert dry.returncode == 0
    assert config.read_text(encoding="utf-8").count("Host runpod-ceg") == 0

    applied = _run(
        "configure_runpod_ssh.sh",
        "--ssh-config",
        str(config),
        "--host",
        "203.0.113.10",
        "--port",
        "12345",
    )
    assert applied.returncode == 0
    second = _run(
        "configure_runpod_ssh.sh",
        "--ssh-config",
        str(config),
        "--host",
        "203.0.113.11",
        "--port",
        "12346",
    )
    assert second.returncode == 0
    text = config.read_text(encoding="utf-8")
    assert text.count("Host runpod-ceg") == 1
    assert "Host keep-me" in text
    assert "Host runpod-a40" in text
    assert "HostName 203.0.113.11" in text
    assert list(tmp_path.glob("config.bak.*"))


def test_connection_check_explains_unconfigured_alias(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.write_text("Host keep-me\n    HostName keep.example\n", encoding="utf-8")
    env = dict(os.environ, SSH_CONFIG_PATH=str(config))
    result = _run("check_runpod_connection.sh", env=env)
    assert result.returncode != 0
    assert "not configured" in result.stderr.lower()


def test_connection_check_default_ssh_options_do_not_raise_unbound_error(
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env.pop("SSH_CONFIG_PATH", None)
    env["HOME"] = str(tmp_path)
    env["RUNPOD_SSH_HOST"] = "ceg-test-unconfigured"
    result = _run("check_runpod_connection.sh", env=env)
    assert result.returncode != 0
    assert "unbound variable" not in result.stderr.lower()
    assert "not configured" in result.stderr.lower()


def test_transfer_helpers_are_explicit_and_non_destructive_by_default() -> None:
    for script in ("sync_to_runpod.sh", "sync_from_runpod.sh"):
        result = _run(script, "--help")
        assert result.returncode == 0
    result = _run("sync_to_runpod.sh", "--unknown")
    assert result.returncode == 2
    assert "Unknown argument" in result.stderr
