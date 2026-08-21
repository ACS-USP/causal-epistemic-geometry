#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Repository wrapper for the no-inference remote preflight."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.research.preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
