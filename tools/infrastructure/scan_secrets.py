#!/usr/bin/env python3
"""Fail when repository candidate files contain common credential material."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

PATTERNS = {
    "private-key": re.compile(rb"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
    "authorization-bearer": re.compile(rb"Authorization\s*:\s*Bearer\s+(?![<{[])\S+", re.I),
    "huggingface-token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "dstack-token-assignment": re.compile(
        rb"(?:DSTACK_TOKEN|dstack[_-]?token)\s*[:=]\s*[\"']?(?!<|REDACTED|NOT_COMMITTED)[A-Za-z0-9_.-]{16,}",
        re.I,
    ),
    "password-assignment": re.compile(
        rb"(?:password|passwd)\s*[:=]\s*[\"']?(?!<|REDACTED|CHANGEME|NOT_COMMITTED)[^\s\"']{8,}",
        re.I,
    ),
    "runpod-key": re.compile(rb"\brpa_[A-Za-z0-9]{20,}\b"),
}


def repository_files(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return [root / item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def scan_paths(paths: Iterable[Path], *, max_bytes: int = 5_000_000) -> list[tuple[Path, str]]:
    findings: list[tuple[Path, str]] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size > max_bytes:
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(data):
                findings.append((path, name))
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = [path.resolve() for path in args.paths] if args.paths else repository_files(root)
    findings = scan_paths(paths)
    if findings:
        for path, pattern in findings:
            try:
                display = path.relative_to(root)
            except ValueError:
                display = path
            print(f"SECRET_SCAN_FAIL {pattern} {display}")
        return 1
    print(f"SECRET_SCAN_PASS files={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
