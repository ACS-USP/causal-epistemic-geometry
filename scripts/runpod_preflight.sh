#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/runpod_q1_smoke.example.yaml}"
cd "${REPO_ROOT}"
CEG_BIN="${CEG_BIN:-ceg}"
if [[ -x "${REPO_ROOT}/.venv/bin/ceg" ]]; then
  CEG_BIN="${REPO_ROOT}/.venv/bin/ceg"
fi

echo "== doctor =="
"${CEG_BIN}" doctor --config "${CONFIG_PATH}" || true
echo
echo "== preflight =="
"${CEG_BIN}" preflight "${CONFIG_PATH}"
