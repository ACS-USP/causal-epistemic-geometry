"""Read and present already-generated run artifacts."""

from __future__ import annotations

from pathlib import Path


def read_summary(run_dir: str | Path) -> str:
    """Return the stored summary without recomputing or changing the run."""

    path = Path(run_dir) / "summary.md"
    if not path.exists():
        raise FileNotFoundError(f"Run summary does not exist: {path}")
    return path.read_text(encoding="utf-8")

