#!/usr/bin/env python3
"""Audit E3-10 decimal and number-word candidates on an explicitly loaded model.

This script is intentionally a remote operation.  The backend's existing
remote-only guard must print the execution host and HF_HOME before any model
load, so running it locally cannot download weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from epistemic_geometry.backends.huggingface import HuggingFaceBackend
from epistemic_geometry.benchmarks.e3.benchmark import view_to_benchmark_item
from epistemic_geometry.benchmarks.e3.rendering import render_latent
from epistemic_geometry.benchmarks.e3.splits import generate_latent
from epistemic_geometry.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    backend = HuggingFaceBackend(config.backend)
    item = generate_latent("MODREG10", "depth_4", 20260817)
    reports = {}
    for channel in ("decimal", "number_word"):
        view = render_latent(item, response_channel=channel)
        prepared = backend.prepare_choice_item(view_to_benchmark_item(view))
        reports[channel] = backend.candidate_token_audit(prepared)
    decimal = reports["decimal"]["candidates"]
    token_ids = [entry["context_compatible_token_ids"] for entry in decimal.values()]
    passed = all(len(ids) == 1 for ids in token_ids) and len({ids[0] for ids in token_ids}) == 10
    print(json.dumps({"primary_decimal_single_token": passed, "reports": reports}, indent=2))
    if not passed:
        print("E3_10_PRIMARY_CHANNEL_TOKENIZATION_FAILED")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
