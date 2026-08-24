#!/usr/bin/env bash
set -u

network_checks=0
if [[ "${1:-}" == "--network" ]]; then
  network_checks=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--network]" >&2
  exit 2
fi

section() {
  printf '\n[%s]\n' "$1"
}

run_if_present() {
  local command_name="$1"
  shift
  if command -v "$command_name" >/dev/null 2>&1; then
    "$command_name" "$@" 2>&1 || true
  else
    printf '%s: not found\n' "$command_name"
  fi
}

section "identity"
printf 'hostname: '
hostname 2>/dev/null || true
printf 'kernel: '
uname -srmo 2>/dev/null || uname -a 2>/dev/null || true
printf 'architecture: '
uname -m 2>/dev/null || true

section "operating-system"
if [[ -r /etc/os-release ]]; then
  awk -F= '/^(NAME|VERSION|ID|VERSION_ID)=/ {print}' /etc/os-release
else
  sw_vers 2>/dev/null || true
fi

section "gpu-driver-cuda"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader 2>&1 || true
  nvidia-smi 2>&1 || true
else
  echo "nvidia-smi: not found"
fi
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"

section "memory"
if command -v free >/dev/null 2>&1; then
  free -h 2>&1 || true
else
  vm_stat 2>/dev/null || true
fi

section "disk"
df -h 2>&1 || true

section "mounts"
if command -v findmnt >/dev/null 2>&1; then
  findmnt -r -o TARGET,SOURCE,FSTYPE,OPTIONS 2>&1 || true
else
  mount 2>&1 || true
fi

section "python-environments"
run_if_present python --version
run_if_present python3 --version
run_if_present uv --version
run_if_present conda --version

section "docker"
run_if_present docker --version

section "pytorch"
python_bin=""
if command -v python >/dev/null 2>&1; then
  python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
fi
if [[ -n "$python_bin" ]]; then
  "$python_bin" -c 'import importlib.util; print("torch_installed:", importlib.util.find_spec("torch") is not None)' 2>&1 || true
  "$python_bin" -c 'import torch; print("torch_version:", torch.__version__); print("torch_cuda_version:", torch.version.cuda); print("cuda_available:", torch.cuda.is_available()); print("cuda_device_count:", torch.cuda.device_count()); print("bf16_supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())' 2>&1 || true
else
  echo "python: not found; PyTorch query skipped"
fi

section "git"
run_if_present git --version

section "hugging-face-cache-locations"
printf 'HF_HOME: %s\n' "${HF_HOME:-<unset>}"
printf 'HF_HUB_CACHE: %s\n' "${HF_HUB_CACHE:-<unset>}"
printf 'TRANSFORMERS_CACHE: %s\n' "${TRANSFORMERS_CACHE:-<unset>}"
for cache_path in "${HF_HOME:-}" "${HF_HUB_CACHE:-}" "${TRANSFORMERS_CACHE:-}" "$HOME/.cache/huggingface"; do
  if [[ -n "$cache_path" && -e "$cache_path" ]]; then
    printf 'existing_cache: %s\n' "$cache_path"
  fi
done

section "network"
if [[ $network_checks -eq 1 ]]; then
  echo "Network checks explicitly enabled."
  if command -v curl >/dev/null 2>&1; then
    curl --head --silent --show-error --max-time 5 https://huggingface.co/ 2>&1 | head -n 5 || true
    curl --head --silent --show-error --max-time 5 https://github.com/ 2>&1 | head -n 5 || true
  else
    echo "curl: not found"
  fi
else
  echo "skipped (pass --network only when policy permits)"
fi

section "doctor-status"
echo "READ_ONLY_DIAGNOSTIC_COMPLETE"
