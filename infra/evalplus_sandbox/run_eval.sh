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

timeout --signal=TERM --kill-after=2s 180s docker run --rm \
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
