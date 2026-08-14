#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:?Usage: scripts/run_q1_smoke.sh path/to/reviewed-config.yaml}"
cd "${REPO_ROOT}"

ceg doctor --config "${CONFIG_PATH}"
ceg preflight "${CONFIG_PATH}"
ceg run "${CONFIG_PATH}"
