#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/runpod_q1_smoke.example.yaml}"
cd "${REPO_ROOT}"

echo "== doctor =="
ceg doctor --config "${CONFIG_PATH}" || true
echo
echo "== preflight =="
ceg preflight "${CONFIG_PATH}"
