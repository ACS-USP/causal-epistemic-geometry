#!/usr/bin/env bash

# Source this on a Pod. It sets only non-secret, persistent-volume paths.
# CEG_HF_HOME is an explicit override for a different persistent mount.
export CEG_ROOT="${CEG_ROOT:-/workspace/causal-epistemic-geometry}"
export HF_HOME="${CEG_HF_HOME:-/workspace/hf-cache}"
export CEG_RUN_ROOT="${CEG_RUN_ROOT:-${CEG_ROOT}/runs}"
mkdir -p "${HF_HOME}" "${CEG_RUN_ROOT}"

if [[ "${TRANSFORMERS_CACHE:-}" == /root/.cache/* ]]; then
  echo "WARNING: TRANSFORMERS_CACHE points at ephemeral /root/.cache; HF_HOME remains canonical." >&2
fi

echo "CEG_ROOT=${CEG_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "CEG_RUN_ROOT=${CEG_RUN_ROOT}"
