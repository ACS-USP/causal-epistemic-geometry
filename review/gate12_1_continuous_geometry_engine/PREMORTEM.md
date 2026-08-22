# Gate 12.1 adversarial premortem

Classification: `PREMORTEM_PASS`.

The qualification isolates sequence indexing, hook registration, cache mode, attention kernel, and dtype. It treats automatic differentiation of the FP32 computational lift as distinct from finite differences of rounded BF16 execution. The epsilon ladder is absolute and local, and JVP/VJP, Fisher/Hessian, utility, and KL identities are independently checked.

Only synthetic token fixtures and frozen engineering random directions are visible to the runner. No benchmark manifest, semantic outcome, scientific item, free generation, Q2 analysis, or holdout access is permitted. A passing engine would qualify only the local directional geometry of the FP32 computational lift of the frozen BF16-valued parameters.
