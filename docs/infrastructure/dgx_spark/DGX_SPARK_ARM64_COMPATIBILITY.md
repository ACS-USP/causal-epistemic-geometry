# DGX Spark ARM64 compatibility audit

## Scope and evidence boundary

This audit covers repository commit
`2fbcc98fa6026b844074bdaef2ba311b360b7f81` and actual wheel-only installation
in dedicated Python 3.12 venvs on both Sparks. `pip check` and 29 tiny
Torch/Transformers hook tests passed on each node without pretrained weights.

The project core declares NumPy, pandas, PyYAML, Typer, Rich, Matplotlib, and
Packaging. Its optional Hugging Face group declares Transformers, Accelerate,
and Datasets. PyTorch is deliberately supplied by the machine/container rather
than declared by the project, so that installing CEG does not replace a
compatible CUDA build.

## Compatibility matrix

| Dependency | Needed by CEG? | ARM64 packaging expectation | GB10/CUDA concern | Source build required? | Current status |
| --- | --- | --- | --- | --- | --- |
| Python >=3.11 | Yes | Native system interpreter | None | No | PASS: 3.12.3 |
| NumPy | Yes | AArch64 wheel | CPU native code only | No | PASS: 2.5.2 |
| pandas | Yes | AArch64 wheel | Depends on NumPy | No | PASS: 3.0.5 |
| PyYAML | Yes | AArch64 wheel | None | No | PASS: 6.0.3 |
| Typer, Rich, Packaging | Yes | Pure Python | None | No | PASS |
| Matplotlib | Yes for analysis/plots | AArch64 wheel | Native FreeType/Qhull dependencies | No | PASS: 3.11.1 |
| PyTorch | Yes for inference | AArch64 CUDA wheel | GB10 capability 12.1, CUDA 13.0 | No | PASS: 2.13.0+cu130 |
| Transformers 4.x | Yes for Qwen | Pure Python plus binary deps | Must remain `<5` | No | PASS: 4.57.6 |
| tokenizers | Transitive, yes | AArch64 Rust wheel | CPU only | No | PASS: 0.22.2 |
| Accelerate | Optional HF stack | Pure Python | Torch integration | No | PASS: 1.14.0 |
| Datasets | Dataset preparation only | Python plus PyArrow | No GPU path | No | PASS: 5.0.1 |
| PyArrow | Transitive via Datasets | AArch64 wheel | CPU native code | No | PASS: 25.0.1 |
| SciPy / scikit-learn | Not imported by CEG core | AArch64 wheels generally exist | CPU native code | Not normally | Not needed; untested |
| Triton | Only if optional compiled Torch paths use it | AArch64 wheel installed by Torch | Blackwell/GB10 compiler execution not tested | No install build | IMPORT PASS: 3.7.1; execution untested |
| Flash Attention | Not declared or imported | ARM64/CUDA extension is highly version-specific | GB10 compute capability and CUDA toolchain | Often | Not needed; do not compile in sprint |
| bitsandbytes | Not declared or imported | Platform/CUDA extension sensitive | GB10 kernels may differ | Possibly | Not needed; untested |
| vLLM | Not used by the canonical CEG backend | NVIDIA ARM image | Container Torch CUDA 13.2 sees GB10 | No host build | Import/device PASS on Spark 1; serving untested |
| Custom CUDA kernels | None found in repository | N/A | N/A | N/A | Not present |

## Import-path findings

- The canonical backend imports NumPy, PyTorch, and Transformers.
- `datasets` is used by data-preparation entry points, not the minimal cached
  model smoke.
- No direct imports of Flash Attention, bitsandbytes, vLLM, Triton, SciPy,
  scikit-learn, or PyArrow were found in project source or execution scripts.
- `torch.compile` and CUDA graph support exist as optional performance paths;
  neither is required for initial eager/SDPA instrument qualification.
- Qwen3 cached-suffix replay depends on Transformers internals and therefore
  needs an explicit version/import test even if basic model loading succeeds.

## Safe node procedure

The executed setup was one venv per node; system Python remained untouched:

```bash
mkdir -p ~/projects
python3 -m venv ~/projects/ceg-infra-venv
source ~/projects/ceg-infra-venv/bin/activate
python -m pip install --only-binary=:all: torch
python -m pip install --only-binary=:all: -e '.[hf,dev]'
python -m pip check
python -m pip freeze --all
```

Before installation, preserve the system/container Torch census. Do not proceed
with a source build, broad upgrade, CUDA/driver change, or hours-long extension
compilation without principal-researcher review.

## Acceptance checks

The declared stack is `ARM_DEPENDENCY_STACK_PASS`. Before scientific use it
still requires:

1. Qwen backend construction from an already cached exact revision, without
   semantic inference;
2. explicit replay/attention-path testing against that exact model;
3. prospective review of the recorded node fingerprint differences.
