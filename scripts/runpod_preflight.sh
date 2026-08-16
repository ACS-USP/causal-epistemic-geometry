#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/runpod_q1_smoke.example.yaml}"
cd "${REPO_ROOT}"
CEG_BIN="${CEG_BIN:-ceg}"
if [[ -x "${REPO_ROOT}/.venv/bin/ceg" ]]; then
  CEG_BIN="${REPO_ROOT}/.venv/bin/ceg"
fi

echo "== environment =="
if [[ -f "scripts/runpod_environment.sh" && ( -d /workspace || "${CEG_FORCE_RUNPOD_ENV:-0}" == "1" ) ]]; then
  # shellcheck disable=SC1091
  source scripts/runpod_environment.sh
else
  echo "Local preflight: not sourcing the Pod environment helper."
fi
echo
echo "== doctor (no download) =="
"${CEG_BIN}" doctor --config "${CONFIG_PATH}" || true
echo
echo "== preflight (no inference/download) =="
"${CEG_BIN}" preflight "${CONFIG_PATH}"
