# Q2 matched random rank-8 control — model-free planning precheck

Status: `FROZEN_MODEL_FREE_PLANNING_PRECHECK`

This precheck freezes the design alternatives, simulation grid, selection
precedence, and prohibitions before the CPU-only planning simulation. It does
not freeze or generate an experimental random basis or an experimental seed.

The scientific unit for the proposed specificity claim is one independently
sampled random subspace orientation. Controllers and controller dyads are
nested observations; neither may be treated as independent evidence about a
population of subspace orientations.

The required matched object contains the exact 47 frozen coefficient
identities: 31 historical-reference identities and 16 fresh identities. For
an orthonormal candidate basis `Q_random`, each controller would be mapped as
`v_k_random = Q_random c_k`, preserving the complete coefficient-space Gram
matrix and A0 geometry. No observed semantic distance may enter basis
construction, safety screening, or method selection.

The planning simulation compares three execution routes and three candidate
random-subspace families. It uses the closed learned-subspace statistic only
as an explicit planning effect. Its seed is a simulation seed, not a future
experimental seed.

No Qwen load, GPU use, safety inference, semantic trajectory, final random
basis, or Q3 work is authorized by this artifact.
