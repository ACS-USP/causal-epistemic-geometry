# Local SSH audit

Audit date: 2026-08-16. Only SSH configuration and public-key metadata were
inspected. No private key contents were read or printed.

## Key material

- Private key file: `~/.ssh/id_ed25519_runpod` — present.
- Public key file: `~/.ssh/id_ed25519_runpod.pub` — present.
- Public-key fingerprint: `SHA256:LKZJHh5XTUMJlll9mZbhUyBVIMT99rUGQmyTIo7oPXE`
  (ED25519, comment `runpod`).
- The private key remains in `~/.ssh` and is not part of this repository.

## Existing relevant configuration

The current `~/.ssh/config` contains:

```text
Host runpod-a40
    HostName 194.68.245.105
    User root
    Port 22039
    IdentityFile ~/.ssh/id_ed25519_runpod
    IdentitiesOnly yes
    ForwardAgent yes
```

This is useful evidence that the dedicated key workflow has worked before, but
the address and port are not assumed to identify the future Pod. There is no
`runpod-ceg` host block yet, and no SSH `Include` directive was present in the
relevant file.

## Reuse decision

The existing identity is reusable. The new alias will be added only after the
researcher receives the new Pod's public host and exposed SSH port:

```bash
scripts/configure_runpod_ssh.sh --host PUBLIC_IP --port PUBLIC_PORT --dry-run
scripts/configure_runpod_ssh.sh --host PUBLIC_IP --port PUBLIC_PORT
scripts/check_runpod_connection.sh
```

This audit did not modify `~/.ssh/config`.
