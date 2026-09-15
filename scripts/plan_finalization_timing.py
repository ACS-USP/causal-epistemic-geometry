#!/usr/bin/env python3
"""Write an outcome-free sensitivity grid for the finalization-timing endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

N_LATENTS = (96, 144, 192, 288)
MEAN_SHORTENING = (0.0, 0.05, 0.10, 0.15, 0.20)
LATENT_CONTRAST_SD = (0.05, 0.15, 0.30)
DELTA = 0.10
THRESHOLD = 60.0
REPLICATIONS = 10_000
SEED = 20260915


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(*, replications: int = REPLICATIONS) -> dict[str, Any]:
    """Simulate bounded latent-level timing contrasts, without empirical inputs."""

    if isinstance(replications, bool) or not isinstance(replications, int) or replications <= 0:
        raise ValueError("replications must be a positive integer")
    rng = np.random.default_rng(SEED)
    log_threshold = float(np.log(THRESHOLD))
    rows: list[dict[str, float | int]] = []
    for n_latents in N_LATENTS:
        for mean in MEAN_SHORTENING:
            for sd in LATENT_CONTRAST_SD:
                contrasts = np.clip(
                    rng.normal(loc=mean, scale=sd, size=(replications, n_latents)),
                    -1.0,
                    1.0,
                )
                early_log_e = np.log1p((contrasts - DELTA) / 2.0).sum(axis=1)
                late_log_e = np.log1p((-contrasts - DELTA) / 2.0).sum(axis=1)
                exclude_early = np.log1p((DELTA - contrasts) / 2.0).sum(axis=1)
                exclude_late = np.log1p((DELTA + contrasts) / 2.0).sum(axis=1)
                rows.append(
                    {
                        "n_latents": n_latents,
                        "mean_shortening": mean,
                        "latent_contrast_sd": sd,
                        "replications": replications,
                        "earlier_by_delta_rate": float(np.mean(early_log_e >= log_threshold)),
                        "later_by_delta_rate": float(np.mean(late_log_e >= log_threshold)),
                        "exclude_both_directions_rate": float(
                            np.mean(
                                (exclude_early >= log_threshold)
                                & (exclude_late >= log_threshold)
                            )
                        ),
                    }
                )
    return {
        "schema_version": "q1-finalization-timing-planning-v1",
        "status": "OUTCOME_FREE_SENSITIVITY_COMPLETE",
        "scope": (
            "synthetic bounded latent-contrast sensitivity only; no Qwen prompt, latent, "
            "seed, response, token sequence, or outcome was used"
        ),
        "decision_rule": {
            "contrast": "mean((time_baseline - time_D75) / 4096)",
            "delta": DELTA,
            "per_declaration_evalue_threshold": THRESHOLD,
            "declarations": [
                "D75 closes thinking at least delta earlier",
                "D75 closes thinking at least delta later",
                "both directional shifts of delta excluded",
            ],
        },
        "simulation": {
            "replications_per_cell": replications,
            "seed": SEED,
            "n_latents": list(N_LATENTS),
            "mean_shortening": list(MEAN_SHORTENING),
            "latent_contrast_sd": list(LATENT_CONTRAST_SD),
            "interpretation": (
                "normal values are clipped to the bounded estimand range and are a "
                "sensitivity grid, not an assumed Qwen timing distribution"
            ),
        },
        "implementation_sha256": _sha256(
            ROOT / "src/epistemic_geometry/analysis/finalization_timing.py"
        ),
        "planner_sha256": _sha256(Path(__file__).resolve()),
        "rows": rows,
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing planning artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = plan()
    _write_new_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "rows": len(payload["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
