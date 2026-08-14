#!/usr/bin/env bash
set -euo pipefail

# Conservative setup: this script never downloads a model and never replaces
# an existing Torch/CUDA installation.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
INSTALL_HF="${INSTALL_HF:-0}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}[dev]"

if [[ "${INSTALL_HF}" == "1" ]]; then
  python -m pip install -e "${REPO_ROOT}[hf]"
fi

echo
echo "Repository installed in ${VENV_DIR}."
echo "No model was downloaded and no Torch wheel was installed or replaced."
if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "Torch is not importable. Install a Torch build compatible with this machine before HF mode."
fi
echo "Next checks:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  ceg doctor"
echo "  ceg doctor --config configs/runpod_qwen3_8b.example.yaml"
echo "Model download/cache setup is an explicit later step; see docs/RUNPOD.md."

