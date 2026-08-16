#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_ALIAS="${RUNPOD_SSH_HOST:-runpod-ceg}"
REMOTE_ROOT="${RUNPOD_REMOTE_ROOT:-/workspace/causal-epistemic-geometry}"
remote_path="${REMOTE_ROOT}/runs/"
destination="${REPO_ROOT}/runs/runpod/"
dry_run=0
allow_existing=0

usage() {
  cat <<'EOF'
Usage: sync_from_runpod.sh [options]

Options:
  --source PATH          Remote path (default: /workspace/.../runs/)
  --destination PATH     Local destination (default: runs/runpod/)
  --allow-existing       Permit merging into an existing destination
  --dry-run              Show rsync changes without copying
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) [[ $# -ge 2 ]] || { echo "--source requires a value" >&2; exit 2; }; remote_path="$2"; shift 2 ;;
    --destination) [[ $# -ge 2 ]] || { echo "--destination requires a value" >&2; exit 2; }; destination="$2"; shift 2 ;;
    --allow-existing) allow_existing=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v rsync >/dev/null 2>&1 || { echo "rsync is required." >&2; exit 1; }
if [[ -e "${destination}" && "${allow_existing}" != "1" ]]; then
  echo "Refusing to merge into existing destination: ${destination}" >&2
  echo "Review it, then rerun with --allow-existing (or choose another destination)." >&2
  exit 1
fi
temporary_destination=""
if [[ "${dry_run}" == "1" && ! -e "${destination}" ]]; then
  temporary_destination="$(mktemp -d "${TMPDIR:-/tmp}/ceg-pull-dry-run.XXXXXX")"
  trap 'rm -rf "${temporary_destination}"' EXIT
  rsync_destination="${temporary_destination}/"
  echo "Dry run uses a temporary destination; no local destination is created."
else
  mkdir -p "${destination}"
  rsync_destination="${destination}"
fi

# Keep flags compatible with the older rsync shipped on some macOS systems.
flags=(-a --partial --human-readable --progress)
if [[ "${dry_run}" == "1" ]]; then
  flags+=(--dry-run --itemize-changes)
fi

echo "Pull source: ${SSH_ALIAS}:${remote_path}"
echo "Pull target: ${destination}"
echo "Remote files are never deleted; local merge requires --allow-existing."
rsync "${flags[@]}" \
  -e 'ssh -o BatchMode=yes -o ConnectTimeout=8' \
  "${SSH_ALIAS}:${remote_path}" "${rsync_destination}"
