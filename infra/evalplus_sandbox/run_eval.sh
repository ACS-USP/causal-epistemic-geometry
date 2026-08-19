#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_eval.sh IMAGE WORKSPACE RESULTS DATASET
#
# WORKSPACE must contain only the minimal samples file and benchmark material
# required for this evaluation. It must never be the repository or a home
# directory. RESULTS is a disposable output directory.
IMAGE=${1:?image digest required}
WORKSPACE=${2:?minimal workspace required}
RESULTS=${3:?disposable results directory required}
DATASET=${4:?dataset required}

[[ -d "$WORKSPACE" ]] || { echo "workspace missing: $WORKSPACE" >&2; exit 2; }
[[ -d "$RESULTS" ]] || { echo "results directory missing: $RESULTS" >&2; exit 2; }

[[ "$IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "image must include a reviewed immutable sha256 digest" >&2
  exit 2
}
command -v docker >/dev/null 2>&1 || {
  echo "docker executable unavailable; evaluator remains fail-closed" >&2
  exit 2
}

WORKSPACE_REAL=$(cd "$WORKSPACE" && pwd -P)
RESULTS_REAL=$(cd "$RESULTS" && pwd -P)
SCRIPT_ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
[[ "$WORKSPACE_REAL" != "$RESULTS_REAL" ]] || {
  echo "workspace and results must be distinct" >&2
  exit 2
}
case "$WORKSPACE_REAL" in
  "$SCRIPT_ROOT"|"$SCRIPT_ROOT"/*)
    echo "refusing to mount the repository as an untrusted-code workspace" >&2
    exit 2
    ;;
esac

if command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT=(gtimeout --signal=TERM --kill-after=2s 180s)
elif command -v timeout >/dev/null 2>&1; then
  TIMEOUT=(timeout --signal=TERM --kill-after=2s 180s)
else
  echo "GNU timeout/gtimeout unavailable; evaluator remains fail-closed" >&2
  exit 2
fi

"${TIMEOUT[@]}" docker run --rm \
  --network none \
  --user 65532:65532 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 64 \
  --cpus 1 \
  --memory 1g \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
  --tmpfs /run:rw,nosuid,nodev,noexec,size=16m \
  -v "$WORKSPACE:/work:ro" \
  -v "$RESULTS:/results:rw" \
  "$IMAGE" \
  --dataset "$DATASET" \
  --samples /work/samples.jsonl \
  --test-details \
  --i-just-wanna-run
