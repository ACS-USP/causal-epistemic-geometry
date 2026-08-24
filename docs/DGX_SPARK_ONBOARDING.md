# DGX Spark onboarding

## Purpose

This checklist prepares a future DGX Spark host for reproducible project work.
It does not authorize scientific execution, package installation, model
downloads, driver changes, or remote access. Credentials and an experiment lock
must arrive separately.

## Before connecting

Record, out of band:

- owner and access policy;
- hostname and approved connection route;
- storage quota and persistence policy;
- whether outbound network and Hugging Face access are permitted;
- expected GPU, driver, CUDA, OS, and container runtime;
- billing or allocation boundary;
- approved locations for source, virtual environments, caches, and artifacts.

Never place credentials, tokens, private keys, or signed URLs in the repository
or diagnostic output.

## Read-only discovery

After access is explicitly authorized, copy or check out only the approved
source revision and run:

```bash
bash scripts/dgx_spark_doctor.sh
```

The doctor is read-only. It reports host, OS, architecture, GPU/driver, memory,
disk, mounts, Python/environment managers, Docker, PyTorch/CUDA visibility, Git,
and likely Hugging Face cache paths. It does not install, pull, mutate, or run a
benchmark. Network probes are skipped by default; `--network` enables only
short reachability checks when local policy permits them.

Review the output for secrets before preserving it as environment provenance.
The script intentionally avoids dumping the environment.

## Tiny explicit smoke

Only after the doctor is reviewed, run the opt-in tensor smoke:

```bash
python scripts/dgx_spark_smoke.py
```

It imports PyTorch, prints version/device properties, performs a tiny matrix
multiplication on one visible CUDA device, synchronizes, and exits. It loads no
model and downloads no data. A missing PyTorch or CUDA device is a diagnostic
failure, not permission to install or repair automatically.

## Reproducible environment acceptance

Before any future scientific protocol:

1. pin a source commit and require a clean checkout;
2. compare Python, PyTorch, CUDA, tokenizer, and model-library versions with the
   protocol's named environment;
3. define immutable model/cache and writable journal/artifact paths;
4. verify available disk for cache plus worst-case raw artifacts;
5. run repository CPU checks and the tiny smoke;
6. run the protocol-specific engineering gate on non-scientific fixtures;
7. record hashes and environment provenance before outcomes;
8. freeze recovery, resume, budget, and teardown rules.

Environment compatibility is not scientific authorization. A prospective lock
and principal-approved cost envelope remain required.

## Data and teardown policy

- Keep raw outputs and benchmark content in approved private storage.
- Push only artifacts allowed by repository egress policy.
- Journal atomically and make resume logical-key safe.
- Recover and verify artifacts before releasing compute.
- At closeout, terminate compute and remove retained storage only when its
  contents are verified elsewhere and deletion is authorized.

## Current status

The doctor and smoke are prepared locally but have not been run on a DGX Spark.
No credentials have been requested or used, and no remote machine has been
modified.

