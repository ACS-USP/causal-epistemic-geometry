#!/usr/bin/env bash
set -euo pipefail

# Add or replace only the runpod-ceg host block. This script never reads the
# private key and supports SSH_CONFIG_PATH for safe local testing.
usage() {
  cat <<'EOF'
Usage: configure_runpod_ssh.sh --host HOST --port PORT [options]

Options:
  --host HOST        RunPod public IP address or DNS name
  --port PORT        RunPod exposed SSH TCP port (1-65535)
  --user USER        SSH user (default: root)
  --identity PATH    SSH identity path (default: ~/.ssh/id_ed25519_runpod)
  --ssh-config PATH  Config file (default: ~/.ssh/config, or SSH_CONFIG_PATH)
  --dry-run          Show the config diff without changing the file
  -h, --help         Show this help
EOF
}

host_name=""
port=""
user_name="root"
identity_file="~/.ssh/id_ed25519_runpod"
config_file="${SSH_CONFIG_PATH:-${HOME}/.ssh/config}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || { echo "--host requires a value" >&2; exit 2; }
      host_name="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      port="$2"
      shift 2
      ;;
    --user)
      [[ $# -ge 2 ]] || { echo "--user requires a value" >&2; exit 2; }
      user_name="$2"
      shift 2
      ;;
    --identity)
      [[ $# -ge 2 ]] || { echo "--identity requires a value" >&2; exit 2; }
      identity_file="$2"
      shift 2
      ;;
    --ssh-config)
      [[ $# -ge 2 ]] || { echo "--ssh-config requires a value" >&2; exit 2; }
      config_file="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${host_name}" || -z "${port}" ]]; then
  echo "Both --host and --port are required." >&2
  usage >&2
  exit 2
fi
if [[ ! "${host_name}" =~ ^[A-Za-z0-9._:-]+$ ]]; then
  echo "Invalid host: ${host_name}" >&2
  exit 2
fi
if [[ ! "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "Invalid port: ${port}; expected 1-65535." >&2
  exit 2
fi
if [[ ! "${user_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid SSH user: ${user_name}" >&2
  exit 2
fi
identity_resolved="${identity_file/#\~/${HOME}}"
if [[ ! -f "${identity_resolved}" ]]; then
  echo "SSH identity file does not exist: ${identity_resolved}" >&2
  echo "Use --identity PATH if the dedicated RunPod key is stored elsewhere." >&2
  exit 1
fi

config_dir="$(dirname "${config_file}")"
block_file="$(mktemp "${TMPDIR:-/tmp}/ceg-ssh-block.XXXXXX")"
candidate_file="$(mktemp "${TMPDIR:-/tmp}/ceg-ssh-config.XXXXXX")"
trap 'rm -f "${block_file}" "${candidate_file}"' EXIT

cat > "${block_file}" <<EOF

Host runpod-ceg
    HostName ${host_name}
    User ${user_name}
    Port ${port}
    IdentityFile ${identity_file}
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
EOF

if [[ -e "${config_file}" ]]; then
  if [[ ! -f "${config_file}" ]]; then
    echo "SSH config path is not a regular file: ${config_file}" >&2
    exit 1
  fi
  # Remove only a Host block containing the exact runpod-ceg pattern. All
  # unrelated hosts, comments, and options pass through unchanged.
  awk '
    function host_line(line) { return line ~ /^[[:space:]]*Host[[:space:]]+/ }
    {
      if (host_line($0)) {
        in_target = 0
        count = split($0, fields, /[[:space:]]+/)
        for (i = 1; i <= count; i++) {
          if (fields[i] == "runpod-ceg") in_target = 1
        }
      }
      if (!in_target) print
    }
  ' "${config_file}" > "${candidate_file}"
else
  : > "${candidate_file}"
fi
cat "${block_file}" >> "${candidate_file}"

if [[ "${dry_run}" == "1" ]]; then
  echo "DRY RUN: ${config_file} would be updated with host runpod-ceg."
  if [[ -e "${config_file}" ]]; then
    diff -u "${config_file}" "${candidate_file}" || true
  else
    sed 's/^/+ /' "${candidate_file}"
  fi
  exit 0
fi

mkdir -p "${config_dir}"
chmod 700 "${config_dir}"
if [[ -e "${config_file}" ]]; then
  backup="${config_file}.bak.$(date -u +%Y%m%dT%H%M%SZ)"
  backup_index=1
  while [[ -e "${backup}" ]]; do
    backup="${config_file}.bak.$(date -u +%Y%m%dT%H%M%SZ).${backup_index}"
    backup_index=$((backup_index + 1))
  done
  cp -p "${config_file}" "${backup}"
  echo "Backed up existing SSH config to ${backup}"
fi
mv "${candidate_file}" "${config_file}"
chmod 600 "${config_file}"
echo "Configured Host runpod-ceg in ${config_file}."
echo "IdentityFile is ${identity_file}; private key contents were not read."
