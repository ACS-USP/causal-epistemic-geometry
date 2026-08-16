#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" && -x "${REPO_ROOT}/.venv/bin/python" ]]; then PYTHON="${REPO_ROOT}/.venv/bin/python"; fi
PYTHON="${PYTHON:-python3}"
RUFF="${RUFF:-}"
if [[ -z "${RUFF}" && -x "${REPO_ROOT}/.venv/bin/ruff" ]]; then RUFF="${REPO_ROOT}/.venv/bin/ruff"; fi
RUFF="${RUFF:-ruff}"
CEG="${CEG_BIN:-}"
if [[ -z "${CEG}" && -x "${REPO_ROOT}/.venv/bin/ceg" ]]; then CEG="${REPO_ROOT}/.venv/bin/ceg"; fi
CEG="${CEG:-ceg}"

software_failed=0
run_check() {
  local label="$1"
  shift
  printf '%-28s' "${label}:"
  if "$@" >/tmp/ceg-predeploy-output.$$ 2>&1; then
    echo "PASS"
  else
    echo "FAIL"
    software_failed=1
    sed -n '1,18p' /tmp/ceg-predeploy-output.$$ >&2
  fi
  rm -f /tmp/ceg-predeploy-output.$$
}

echo "PREDEPLOY GATE"
echo "======================="
echo "Git status (must be reviewed before deployment):"
git status --short
echo
run_check "tests" "${PYTHON}" -m pytest -q
run_check "ruff" "${RUFF}" check .
run_check "compileall" "${PYTHON}" -m compileall -q src
run_check "mock" make smoke
run_check "tiny transformer" make tiny-smoke
run_check "hooks/vector/resume" "${PYTHON}" -m pytest -q tests/test_huggingface_tiny.py tests/test_resume_and_validation.py tests/test_steering.py tests/test_parsing_and_geometry.py

latest_tiny="$(find runs -maxdepth 1 -type d -name '*tiny-random-transformer*' -print 2>/dev/null | sort | tail -n 1)"
if [[ -n "${latest_tiny}" ]]; then
  run_check "run validator" "${CEG}" validate-run "${latest_tiny}"
else
  echo "run validator: FAIL (no tiny-transformer artifact found)"
  software_failed=1
fi
run_check "mock preflight" "${CEG}" preflight configs/mock_smoke.yaml

printf '%-28s' "RunPod preflight template:"
if "${CEG}" preflight configs/runpod_q1_smoke.example.yaml >/tmp/ceg-predeploy-output.$$ 2>&1; then
  echo "FAIL (template unexpectedly ready)"
  software_failed=1
else
  if rg -q "PREFLIGHT: NOT READY|BLOCKER" /tmp/ceg-predeploy-output.$$; then
    echo "PASS (expected placeholders reported)"
  else
    echo "FAIL (unexpected preflight failure)"
    software_failed=1
    sed -n '1,18p' /tmp/ceg-predeploy-output.$$ >&2
  fi
fi
rm -f /tmp/ceg-predeploy-output.$$

echo
echo "SSH key found: $([[ -f "${HOME}/.ssh/id_ed25519_runpod" ]] && echo YES || echo NO)"
if [[ -f "${HOME}/.ssh/id_ed25519_runpod.pub" ]]; then
  echo "RunPod public key fingerprint: $(ssh-keygen -lf "${HOME}/.ssh/id_ed25519_runpod.pub" | awk '{print $2}')"
else
  echo "RunPod public key fingerprint: UNKNOWN (public key not found)"
fi
alias_host=""
if alias_config="$(ssh -G runpod-ceg 2>/dev/null)"; then
  alias_host="$(printf '%s\n' "${alias_config}" | awk '$1 == "hostname" {print $2; exit}')"
fi
if [[ -n "${alias_host}" && "${alias_host}" != "runpod-ceg" ]]; then
  echo "runpod-ceg alias configured: YES (${alias_host})"
else
  echo "runpod-ceg alias configured: NO (configure after Pod supplies host/port)"
fi

if [[ "${software_failed}" == "0" ]]; then
  echo "GPU Pod required yet: YES (software gate passed; real-model review remains)"
  exit 0
fi
echo "GPU Pod required yet: NO (local software gate has failures)"
exit 1
