# Old RunPod workflow audit

Audit date: 2026-08-16. The legacy repository at
`/Users/costaleirbag/dev/masters-project` was inspected read-only. It was not
modified, copied from, imported, or committed.

## What existed

- `setup_remote.sh` was the main remote bootstrap entry point.
- The old project documented using SSH/VS Code on a Pod, but did not contain a
  reusable repository-local rsync push/pull workflow.
- The bootstrap used `uv`, installed Node/NVM and Gemini CLI, installed `btop`,
  removed an existing `.venv`, synchronized the old project environment, then
  force-reinstalled a CUDA 12.4 Torch/torchvision/torchaudio set.
- The local SSH configuration contains a pre-existing `runpod-a40` host with a
  RunPod endpoint, root user, exposed port, the dedicated
  `~/.ssh/id_ed25519_runpod` identity, `IdentitiesOnly yes`, and agent
  forwarding. The endpoint is treated as historical and is not reused as the
  new experiment alias.

## Reusable ideas

- Keep a dedicated Ed25519 identity for RunPod.
- Use an SSH host alias rather than repeating a public host, port, and identity
  on every command.
- Treat the remote Pod as a persistent workspace and perform environment checks
  immediately after connection.
- Check the installed Torch/CUDA combination before making package changes.

## Obsolete or unsafe for this repository

- Removing `.venv` is destructive and unnecessary.
- Force-reinstalling a CUDA-specific Torch wheel can break an official image or
  replace a compatible build.
- Installing Node, Gemini CLI, and unrelated system packages expands scope and
  cost without helping the Q1 harness.
- The old cache strategy does not define `/workspace/hf-cache` as the canonical
  persistent HuggingFace location.

## New workflow

This repository adds a conservative, explicit workflow:

- `scripts/configure_runpod_ssh.sh` updates only `Host runpod-ceg`, with a
  timestamped backup, `--dry-run`, and a temporary-config test path.
- `scripts/check_runpod_connection.sh` performs a BatchMode, read-only health
  check after the researcher supplies the Pod host and port.
- `scripts/sync_to_runpod.sh` transfers Git history, code, configs, docs,
  scripts, tests, and small examples while excluding virtualenvs, caches, run
  outputs, model files, and secrets. It never deletes remote files by default.
- `scripts/sync_from_runpod.sh` pulls runs by default and requires explicit
  permission to merge into an existing local destination.
- `scripts/runpod_environment.sh` makes `/workspace/hf-cache` the canonical
  cache and keeps runs below the persistent workspace.
- `scripts/bootstrap_runpod.sh` preserves existing Torch and never downloads a
  model. Model download is a later, reviewed cost gate.

No legacy code or model settings were adopted as scientific choices.
