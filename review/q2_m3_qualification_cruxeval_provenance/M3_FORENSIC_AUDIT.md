# M3 forensic audit

Classification: `M3_FORENSIC_CLEAN`.

- primary/audit classification: `M3_DERIVATIVE_IDENTITIES_FAILED` / `M3_DERIVATIVE_IDENTITIES_FAILED`
- maximum independently recomputed metric difference: `2.117e-08`
- sequence / derivative: `True` / `False`
- finite-window / BF16-bridge: `False` / `False`
- scientific items processed: `0`
- semantic outcomes read: `False`

The audit independently reloaded the immutable sufficient-statistics and JVP
arrays, recomputed Gram algebra, PSD, finite-window eligibility, bridge gates,
and the frozen classification without importing the primary analysis module.
The JVP/VJP scalar error was threshold-checked from the technical runner record;
its two scalar operands were not separately archived. This is a reproducibility
limitation, not semantic-outcome leakage.
