# DGX Spark shared-storage qualification

## Current status

`SHARED_STORAGE_PASS`: both nodes expose `~/shared` as a symlink to their local
`/srv/shared`. The expected `modelos`, `datasets`, and `checkpoints` directories
exist and are group-writable by `sparkusers`.

Spark 1 has an additive `spark-shared.service` oneshot running
`/usr/local/bin/spark-shared sync`, plus a timer scheduled approximately every
15 minutes. Spark 2 has no local unit. This is consistent with Spark 1
orchestrating the bidirectional union, but only Spark 1 to Spark 2 propagation
was exercised.

## Read-only census on each node

```bash
ls -lah ~/shared
ls -ld ~/shared/modelos ~/shared/datasets ~/shared/checkpoints
mount
df -h ~/shared
systemctl status --no-pager spark-shared.service
systemctl is-active spark-shared.service
systemctl is-enabled spark-shared.service
```

Record whether `~/shared` is a symlink, bind mount, ordinary directory, or
another filesystem presentation. Compare filesystem source, size, and service
state on both nodes.

## Observed bounded additive-sync probe

One 50-byte marker was created on Spark 1:

`ceg-infra-sync-probe-20260826T005700Z-spark1-01a03b82.txt`

Its SHA-256 on Spark 1 was:

`70da98ff399b5010a4013d7ad3819a09bacfbe923403d44e53f8ae3597cc378a`

`sudo -n systemctl start spark-shared.service` succeeded with service result 0.
The file was then immediately visible on Spark 2 with the same 50-byte size,
timestamp, and SHA-256. The marker is deliberately retained because deletion
propagation is outside scope. No large file or throughput test was run.

## Operational caveat

Treat the two paths as eventually consistent local copies, not a concurrent
POSIX shared filesystem. Jobs must use unique,
logical-key-derived output paths; a node must not expect the other node's write
to be immediately visible. Additive/no-delete sync also means temporary files
and failed-job artifacts require an administrator-reviewed retention policy.
