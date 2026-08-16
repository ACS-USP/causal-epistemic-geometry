#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_ALIAS="${RUNPOD_SSH_HOST:-runpod-ceg}"
REMOTE_ROOT="${RUNPOD_REMOTE_ROOT:-/workspace/causal-epistemic-geometry}"
dry_run=0
delete_remote=0

usage() {
  cat <<'EOF'
Usage: sync_to_runpod.sh [--dry-run] [--delete]

Environment:
  RUNPOD_SSH_HOST   SSH alias (default: runpod-ceg)
  RUNPOD_REMOTE_ROOT remote repository root (default: /workspace/causal-epistemic-geometry)

Default sync is additive and never deletes remote files. --delete is an
explicit destructive remote cleanup option and should be used only after review.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --delete) delete_remote=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v rsync >/dev/null 2>&1 || { echo "rsync is required." >&2; exit 1; }

# Keep flags compatible with the older rsync shipped on some macOS systems.
flags=(-a --partial --human-readable --progress)
if [[ "${dry_run}" == "1" ]]; then
  flags+=(--dry-run --itemize-changes)
fi
if [[ "${delete_remote}" == "1" ]]; then
  flags+=(--delete)
  echo "WARNING: remote files absent locally may be deleted (--delete)." >&2
fi

excludes=(
  --exclude .venv/
  --exclude venv/
  --exclude __pycache__/
  --exclude .pytest_cache/
  --exclude .ruff_cache/
  --exclude runs/
  --exclude models/
  --exclude checkpoints/
  --exclude hf-cache/
  --exclude huggingface/
  --exclude .cache/
  --exclude .env
  --exclude .env.*
  --exclude '*.safetensors'
  --exclude '*.bin'
  --exclude '*.pt'
  --exclude '*.pth'
  --exclude '*.ckpt'
  --exclude '*.token'
)

echo "Sync source: ${REPO_ROOT}/"
echo "Sync target: ${SSH_ALIAS}:${REMOTE_ROOT}/"
echo "Excluded: virtualenvs, caches, runs, model material, and secrets"
rsync "${flags[@]}" "${excludes[@]}" \
  -e 'ssh -o BatchMode=yes -o ConnectTimeout=8' \
  "${REPO_ROOT}/" "${SSH_ALIAS}:${REMOTE_ROOT}/"
