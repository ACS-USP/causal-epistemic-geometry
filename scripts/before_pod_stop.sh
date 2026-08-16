#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "== completed runs on this Pod =="
shopt -s nullglob
run_dirs=(runs/*)
if [[ ${#run_dirs[@]} -eq 0 ]]; then
  echo "No local runs directory entries found."
else
  for run_dir in "${run_dirs[@]}"; do
    if [[ -d "${run_dir}" && -f "${run_dir}/manifest.json" ]]; then
      status="$(python - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "manifest.json"
try:
    print(json.loads(path.read_text(encoding="utf-8")).get("status", "UNKNOWN"))
except (OSError, ValueError):
    print("INVALID")
PY
)"
      echo "${run_dir}: ${status}"
    fi
  done
fi

echo
echo "== local repository changes on Pod =="
git status --short
echo
echo "== disk usage =="
if command -v df >/dev/null 2>&1; then
  df -h / || true
  [[ -d /workspace ]] && df -h /workspace || echo "/workspace: NOT FOUND"
fi
du -sh "${REPO_ROOT}" 2>/dev/null || true
du -sh "${HF_HOME:-/workspace/hf-cache}" 2>/dev/null || true

cat <<'EOF'

BEFORE TERMINATING:
  1. Run the local pull helper on the Mac before stopping the Pod.
  2. Pull runs, vectors, manifests, and logs that matter.
  3. Confirm code changes are committed and synced.
  4. Confirm no valuable data exists only under /workspace.
  5. Only then stop/terminate the Pod.

This script does not contact the local Mac and does not terminate anything.
EOF
