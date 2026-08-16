# Before terminating a Pod

Run the remote read-only checklist:

```bash
cd /workspace/causal-epistemic-geometry
source scripts/runpod_environment.sh
bash scripts/before_pod_stop.sh
```

Before termination:

1. On the Mac, pull completed `runs/` with
   `scripts/sync_from_runpod.sh`.
2. Pull any vector `.npz` and adjacent JSON metadata needed for review.
3. Validate each completed run with `ceg validate-run`.
4. Inspect `git status` on the Pod and sync/commit intentional code changes.
5. Confirm no valuable artifacts remain only under `/workspace`.
6. Confirm the persistent cache is where intended; do not delete it as part of
   this checklist.
7. Stop/terminate the Pod only after the above is recorded.

`before_pod_stop.sh` does not terminate anything and cannot push files into the
Mac. The Mac-side pull is an explicit step.
