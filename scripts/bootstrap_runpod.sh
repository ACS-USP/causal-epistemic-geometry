#!/usr/bin/env bash
set -euo pipefail

# Conservative setup: this script never downloads a model and never replaces
# an existing Torch/CUDA installation.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
INSTALL_HF="${INSTALL_HF:-0}"

echo "== host checks =="
python3 --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "nvidia-smi: NOT FOUND (this is expected on a CPU-only machine)"
fi

# shellcheck disable=SC1091
if [[ -f "${REPO_ROOT}/scripts/runpod_environment.sh" ]]; then
  source "${REPO_ROOT}/scripts/runpod_environment.sh"
fi

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
if python -c 'import torch' >/dev/null 2>&1; then
  python -c 'import torch; print(f"torch={torch.__version__}; cuda={torch.cuda.is_available()}; devices={torch.cuda.device_count()}")'
else
  echo "Torch is not importable. Install a Torch build compatible with this machine before HF mode."
fi
echo "Next checks:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  ceg doctor"
echo "  ceg storage-check"
echo "  ceg doctor --config configs/runpod_qwen3_8b.example.yaml"
if python -c 'import torch, transformers' >/dev/null 2>&1; then
  echo "Running cheap local technical checks (no model download)..."
  python -m pytest -q
  ceg run configs/tiny_transformer_smoke.yaml
else
  echo "Skipping tiny-transformer smoke: Torch and/or Transformers is missing."
fi
echo "Model download/cache setup is an explicit later step; see docs/RUNPOD.md."
