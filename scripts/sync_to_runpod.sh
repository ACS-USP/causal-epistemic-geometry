#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_ALIAS="${RUNPOD_SSH_HOST:-runpod-ceg}"
REMOTE_ROOT="${RUNPOD_REMOTE_ROOT:-/workspace/causal-epistemic-geometry}"
dry_run=0
delete_remote=0
stage_a_manifest=""

usage() {
  cat <<'EOF'
Usage: sync_to_runpod.sh [--dry-run] [--delete] [--stage-a-manifest PATH]

Environment:
  RUNPOD_SSH_HOST   SSH alias (default: runpod-ceg)
  RUNPOD_REMOTE_ROOT remote repository root (default: /workspace/causal-epistemic-geometry)

  --stage-a-manifest PATH
                    Explicitly transfer the small Q1 V3 Stage-A manifest.
                    The normal sync excludes review/ artifacts.

Default sync is additive and never deletes remote files. --delete is an
explicit destructive remote cleanup option and should be used only after review.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --delete) delete_remote=1; shift ;;
    --stage-a-manifest)
      [[ $# -ge 2 ]] || { echo "--stage-a-manifest requires a path." >&2; exit 2; }
      stage_a_manifest="$2"
      shift 2
      ;;
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
  --exclude review/
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

tar_excludes=(
  --exclude=./.venv
  --exclude=./venv
  --exclude=./__pycache__
  --exclude=./.pytest_cache
  --exclude=./.ruff_cache
  --exclude=./runs
  --exclude=./review
  --exclude=./models
  --exclude=./checkpoints
  --exclude=./hf-cache
  --exclude=./huggingface
  --exclude=./.cache
  --exclude=./.env
  --exclude=./.env.*
  --exclude='*.safetensors'
  --exclude='*.bin'
  --exclude='*.pt'
  --exclude='*.pth'
  --exclude='*.ckpt'
  --exclude='*.token'
)

create_tar() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --no-acls --no-fflags "$@"
  else
    tar "$@"
  fi
}

if [[ ! "${REMOTE_ROOT}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
  echo "Unsafe RUNPOD_REMOTE_ROOT; use an absolute path without shell metacharacters." >&2
  exit 2
fi

if [[ -n "${stage_a_manifest}" ]]; then
  if [[ "${stage_a_manifest}" != /* ]]; then
    stage_a_manifest="${REPO_ROOT}/${stage_a_manifest}"
  fi
  stage_a_manifest="$(cd "$(dirname "${stage_a_manifest}")" && pwd)/$(basename "${stage_a_manifest}")"
  expected_manifest="${REPO_ROOT}/review/q1_v3_reasoning_instrument/stage_a_manifest.json"
  if [[ "${stage_a_manifest}" != "${expected_manifest}" ]]; then
    echo "--stage-a-manifest must point to review/q1_v3_reasoning_instrument/stage_a_manifest.json." >&2
    exit 2
  fi
  [[ -f "${stage_a_manifest}" ]] || {
    echo "Stage-A manifest not found: ${stage_a_manifest}" >&2
    exit 1
  }
fi

sync_stage_a_manifest() {
  [[ -n "${stage_a_manifest}" ]] || return 0
  local remote_dir="${REMOTE_ROOT}/review/q1_v3_reasoning_instrument"
  local remote_path="${remote_dir}/stage_a_manifest.json"
  local remote_tmp="${remote_path}.tmp.$$.json"
  echo "Sync explicit Stage-A manifest: ${stage_a_manifest} -> ${SSH_ALIAS}:${remote_path}"
  if [[ "${dry_run}" == "1" ]]; then
    echo "DRY RUN: would atomically install Stage-A manifest on remote."
    return 0
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_ALIAS}" "mkdir -p -- '${remote_dir}'"
  scp -q -o BatchMode=yes -o ConnectTimeout=8 "${stage_a_manifest}" "${SSH_ALIAS}:${remote_tmp}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_ALIAS}" "mv -- '${remote_tmp}' '${remote_path}'"
}

echo "Sync source: ${REPO_ROOT}/"
echo "Sync target: ${SSH_ALIAS}:${REMOTE_ROOT}/"
echo "Excluded: virtualenvs, caches, runs, model material, and secrets"
remote_rsync="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_ALIAS}" \
  'command -v rsync >/dev/null 2>&1 && echo yes || echo no')" || {
  echo "Unable to inspect remote rsync; SSH authentication or connection failed." >&2
  exit 1
}

if [[ "${remote_rsync}" == "yes" ]]; then
  rsync "${flags[@]}" "${excludes[@]}" \
    -e 'ssh -o BatchMode=yes -o ConnectTimeout=8' \
    "${REPO_ROOT}/" "${SSH_ALIAS}:${REMOTE_ROOT}/"
  sync_stage_a_manifest
  exit 0
fi

if [[ "${delete_remote}" == "1" ]]; then
  echo "Cannot use --delete when remote rsync is unavailable; install rsync or review a different workflow." >&2
  exit 1
fi

echo "Remote rsync is unavailable; using additive tar-over-SSH fallback."
if [[ "${dry_run}" == "1" ]]; then
  archive="$(mktemp "${TMPDIR:-/tmp}/ceg-sync-dry-run.XXXXXX")"
  trap 'rm -f "${archive}"' EXIT
  create_tar -C "${REPO_ROOT}" -cf "${archive}" "${tar_excludes[@]}" .
  tar -tf "${archive}"
  sync_stage_a_manifest
  exit 0
fi

create_tar -C "${REPO_ROOT}" -cf - "${tar_excludes[@]}" . | \
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_ALIAS}" \
  "mkdir -p -- '${REMOTE_ROOT}' && tar --no-same-owner --overwrite -xpf - -C '${REMOTE_ROOT}'"
sync_stage_a_manifest
