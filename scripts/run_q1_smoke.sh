#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:?Usage: scripts/run_q1_smoke.sh path/to/reviewed-config.yaml}"
cd "${REPO_ROOT}"
CEG_BIN="${CEG_BIN:-ceg}"
if [[ -x "${REPO_ROOT}/.venv/bin/ceg" ]]; then
  CEG_BIN="${REPO_ROOT}/.venv/bin/ceg"
fi

"${CEG_BIN}" doctor --config "${CONFIG_PATH}"
"${CEG_BIN}" preflight "${CONFIG_PATH}"
"${CEG_BIN}" run "${CONFIG_PATH}"
