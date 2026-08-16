# Codex Remote SSH workflow

The repository does not install Codex CLI on the Pod. Once the Pod exists:

1. Receive its SSH-over-exposed-TCP command.
2. Preview the local alias change without touching the real config:

   ```bash
   scripts/configure_runpod_ssh.sh --host PUBLIC_IP --port PUBLIC_PORT --dry-run
   ```

3. After reviewing the diff, add the alias:

   ```bash
   scripts/configure_runpod_ssh.sh --host PUBLIC_IP --port PUBLIC_PORT
   scripts/check_runpod_connection.sh
   ```

4. In the Codex desktop app, choose the Remote SSH host `runpod-ceg`.
5. Open `/workspace/causal-epistemic-geometry`.
6. Use `source scripts/runpod_environment.sh` and the cost-gated workflow in
   [RUNPOD_COST_GATES.md](RUNPOD_COST_GATES.md).

For artifact transfer from the Pod back to the Mac:

```bash
scripts/sync_from_runpod.sh
scripts/sync_from_runpod.sh --source /workspace/causal-epistemic-geometry/vectors/ \
  --destination vectors/runpod --allow-existing
```

The alias helper backs up `~/.ssh/config` and replaces only its
`Host runpod-ceg` block. It does not alter the existing `runpod-a40` alias.
