#!/usr/bin/env python3
"""Frozen future Q2 OOS V2 semantic generation primitive.

The 19,200-row schedule does not yet exist and semantic execution is not
authorized.  This module binds the endpoint-equivalent terminal policy to the
already qualified serial reference engine so that the primitive can be tested
before any benchmark row is opened.  The CLI deliberately fails closed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.inference.terminal_policies import (  # noqa: E402
    EXTREME_REPETITION_NAME,
    extreme_mechanical_repetition_v1,
    extreme_repetition_policy_identity,
)
from epistemic_geometry.types import BackendOutput, BenchmarkItem  # noqa: E402

SEMANTIC_MAX_NEW_TOKENS = 4096
SAFETY_MAX_NEW_TOKENS = 4096


def generate_frozen_semantic_output(
    backend: Any,
    item: BenchmarkItem,
    *,
    sampling_seed: int,
    intervention_metadata: dict[str, Any],
) -> BackendOutput:
    """Generate one future row with the frozen endpoint-equivalent stop."""

    return backend.generate_reasoning(
        item,
        sampling_seed=sampling_seed,
        max_new_tokens=SEMANTIC_MAX_NEW_TOKENS,
        intervention_metadata=intervention_metadata,
        token_stop_predicate=extreme_mechanical_repetition_v1,
        token_stop_name=EXTREME_REPETITION_NAME,
    )


def frozen_terminal_metadata(output: BackendOutput) -> dict[str, Any]:
    """Translate engine metadata into the immutable terminal-row contract."""

    metadata = dict(output.metadata)
    policy = dict(metadata.get("terminal_policy", {}))
    generated = int(metadata.get("generated_token_count", 0))
    repetition_stop = bool(policy.get("triggered", False))
    hard_cap_stop = generated >= SEMANTIC_MAX_NEW_TOKENS
    terminal_failure = repetition_stop or hard_cap_stop
    return {
        "generated_token_count": generated,
        "truncated": terminal_failure,
        "terminal_reason": (
            EXTREME_REPETITION_NAME
            if repetition_stop
            else "max_new_tokens"
            if hard_cap_stop
            else "natural_completion"
        ),
        "terminal_answer_channel_failure": terminal_failure,
        "commitment_valid_if_terminal_failure": False if terminal_failure else None,
        "semantic_evaluable_if_terminal_failure": False if terminal_failure else None,
        "binary_error_e_if_terminal_failure": 1 if terminal_failure else None,
        "terminal_policy": policy,
    }


def main() -> int:
    print(
        json.dumps(
            {
                "status": "Q2_OOS_V2_SEMANTIC_EXECUTION_NOT_AUTHORIZED",
                "future_schedule": "NOT_FROZEN",
                "semantic_trajectories": 0,
                "policy": extreme_repetition_policy_identity(),
            },
            sort_keys=True,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
