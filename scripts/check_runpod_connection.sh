#!/usr/bin/env bash
set -euo pipefail

alias_name="${RUNPOD_SSH_HOST:-runpod-ceg}"
ssh_command=(ssh)
if [[ -n "${SSH_CONFIG_PATH:-}" ]]; then
  ssh_command+=(-F "${SSH_CONFIG_PATH}")
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: check_runpod_connection.sh [SSH_ALIAS]"
  echo "Default alias: ${alias_name}"
  exit 0
fi
if [[ $# -gt 1 ]]; then
  echo "Expected zero or one SSH alias." >&2
  exit 2
fi

if ! ssh_config="$("${ssh_command[@]}" -G "${alias_name}" 2>/dev/null)"; then
  echo "Unable to inspect SSH configuration for ${alias_name}." >&2
  exit 1
fi
resolved_host="$(printf '%s\n' "${ssh_config}" | awk '$1 == "hostname" {print $2; exit}')"
if [[ -z "${resolved_host}" || "${resolved_host}" == "${alias_name}" ]]; then
  echo "SSH alias ${alias_name} is not configured yet." >&2
  echo "After the Pod supplies its public host and port, run:" >&2
  echo "  scripts/configure_runpod_ssh.sh --host PUBLIC_IP --port PUBLIC_PORT" >&2
  exit 1
fi

echo "Checking ${alias_name} (${resolved_host}) with BatchMode and no remote writes..."
if ! "${ssh_command[@]}" \
  -o BatchMode=yes \
  -o ConnectTimeout=8 \
  -o ServerAliveInterval=10 \
  -o ServerAliveCountMax=2 \
  "${alias_name}" \
  'hostname; whoami; pwd; if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; else echo "nvidia-smi: NOT FOUND"; fi'; then
  echo "SSH connection failed. Verify the Pod is running, the exposed port is reachable, and the key is authorized." >&2
  exit 1
fi
