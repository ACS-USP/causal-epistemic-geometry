#!/usr/bin/env bash
set -euo pipefail

# Benign probes only. This script intentionally does not inspect or execute
# any model-generated program. Run it from a disposable directory on the host.
IMAGE=${1:?sandbox image digest required}

run_probe() {
  local name=$1
  local code=$2
  printf 'PROBE %s: ' "$name"
  if timeout --signal=TERM --kill-after=2s 10s docker run --rm \
      --network none \
      --user 65532:65532 \
      --cap-drop ALL \
      --security-opt no-new-privileges \
      --pids-limit 32 \
      --cpus 1 \
      --memory 256m \
      --read-only \
      --tmpfs /tmp:rw,nosuid,nodev,noexec,size=32m \
      "$IMAGE" python -c "$code" >/tmp/ceg-sandbox-probe.out 2>&1; then
    echo PASS
  else
    echo EXPECTED-FAIL
  fi
}

run_probe normal 'print("sandbox-ok")'
run_probe network 'import socket; socket.create_connection(("example.com", 80), 1)'
run_probe host_files 'from pathlib import Path; print(Path("/Users/costaleirbag/.ssh").exists())'
run_probe docker_socket 'from pathlib import Path; print(Path("/var/run/docker.sock").exists())'
run_probe write_boundary 'from pathlib import Path; Path("/outside").write_text("x")'
run_probe timeout 'while True: pass'
run_probe memory 'x = bytearray(2 * 1024 * 1024 * 1024)'
run_probe process_limit 'import os; [os.fork() for _ in range(1000)]'
