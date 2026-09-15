#!/usr/bin/env python3
"""Write a fixed, outcome-free sensitivity grid for late-D75 ablation planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.late_exposure_ablation import (  # noqa: E402
    PlanningCell,
    estimate_operating_characteristics,
)

N_LATENTS = (96, 144, 192, 240, 288)
P_EARLY_STOP = (0.4, 0.6, 0.8)
LATE_EXPOSURE_HARM = (0.0, 0.05, 0.10, 0.15, 0.20)
SHARED_RANDOMNESS = (0.0, 0.5, 1.0)
DELTA = 0.10
THRESHOLD = 40.0
REPLICATIONS = 1_000
SEED = 20260915


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(*, replications: int = REPLICATIONS) -> dict[str, Any]:
    """Run the complete fixed planning grid without prompts or model outcomes."""

    cells = [
        PlanningCell(n, p, harm, shared)
        for n in N_LATENTS
        for p in P_EARLY_STOP
        for harm in LATE_EXPOSURE_HARM
        for shared in SHARED_RANDOMNESS
    ]
    rows = estimate_operating_characteristics(
        cells,
        delta=DELTA,
        threshold=THRESHOLD,
        replications=replications,
        seed=SEED,
    )
    return {
        "schema_version": "q1-d75-late-exposure-planning-v1",
        "status": "OUTCOME_FREE_SENSITIVITY_COMPLETE",
        "scope": (
            "synthetic paired-Bernoulli sensitivity only; no Qwen prompt, latent, seed, "
            "response, or outcome was used"
        ),
        "decision_rule": {
            "contrast": "mean(correct_D75_off_after_2048 - correct_D75_full_4096)",
            "delta": DELTA,
            "per_declaration_evalue_threshold": THRESHOLD,
            "declarations": [
                "support relevant late-exposure harm",
                "exclude relevant late-exposure harm",
            ],
        },
        "simulation": {
            "replications_per_cell": replications,
            "seed": SEED,
            "n_latents": list(N_LATENTS),
            "p_early_stop": list(P_EARLY_STOP),
            "late_exposure_harm": list(LATE_EXPOSURE_HARM),
            "shared_randomness": list(SHARED_RANDOMNESS),
            "interpretation": (
                "shared_randomness is a sensitivity parameter, not an estimate of future "
                "Qwen trajectory dependence"
            ),
        },
        "implementation_sha256": _sha256(
            ROOT / "src/epistemic_geometry/analysis/late_exposure_ablation.py"
        ),
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
