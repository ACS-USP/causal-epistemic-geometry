# EvalPlus disposable sandbox

This directory is an invocation template, not a scientific result. The image
must be built and pinned on a host with a functioning Docker Engine before the
dense-code pilot can pass its sandbox gate.

The base image digest is intentionally a placeholder. Replace it with a
reviewed immutable digest during provisioning and record the final image digest
in the pilot manifest. Do not commit generated programs, benchmark caches, or
credentials.

Required runtime properties:

- `--network none`
- non-root UID `65532`
- `--cap-drop ALL`
- `--security-opt no-new-privileges`
- `--pids-limit` and explicit CPU/memory limits
- `--read-only` root filesystem
- only a purpose-built temporary workspace mounted read-only
- a separate disposable results directory mounted read-write
- no home-directory, repository, SSH, credential, Docker-socket, or secret
  mounts
- `--rm` and a hard external wall-time timeout

The official EvalPlus executor must be run with test-details enabled. Its
`(status, details)` result is the source of the nested per-test vector. The
container wrapper must preserve test ordering and hash the exact benchmark
inputs before any model output is evaluated.

The image is not yet approved. Run `sandbox_probe.sh` and the official fixture
audit on the target host, then freeze the image digest and invocation flags.
