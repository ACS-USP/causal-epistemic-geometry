# Q2 V3 replacement-family conceptual inventory

Status: `CPU_ONLY_DESIGN_ONLY`  
Behavioral candidate search: `NONE`  
Decision population: the 88 CRUXEval records in the existing 336-record
provenance bundle that are disjoint from the unopened 200-item primary panel
and the earlier 24/24 source construction/validation sets.

Only `id`, official index, code, input, dataset identity, and dataset revision
were admitted to the static analysis. The benchmark `output` field, model
generations, source metrics, correctness, G/C/D, and semantic-panel outcomes
were excluded. Static counts are not predictions of model qualification.

## Surviving basis

The exact surviving family IDs are:

1. `CONTROL_FLOW_PATH_COVERAGE` — complete feasible-path coverage versus one
   apparent path;
2. `MUTATION_ALIAS_CAUSALITY` — causal mutation/alias ledger versus local
   variable updates;
3. `LOOP_BOUNDARY_ACCOUNTING` — explicit initialization/iteration/update/
   termination accounting versus loop gist;
4. `HYPOTHESIS_BRANCH_ELIMINATION` — maintain and eliminate candidate outcomes
   versus first-plausible commitment.

Their numerical qualification margins were not used to construct, rank, or
reject any replacement concept.

## Inventory

“Static +/−” counts feature-present and feature-absent programs among the 88
disjoint records. The intended source contrast would still require a paired
positive/negative reasoning instruction and a separate prospective lock.

| Candidate | Positive concept | Negative concept | Static + / − | Distinct from survivors? | Prior behavioral contamination? | Decision |
|---|---|---|---:|---|---|---|
| `INTERMEDIATE_DATAFLOW_COMPOSITION` | propagate exact intermediate return values through nested dependencies | infer the overall computation without an explicit dependency ledger | 30 / 58 | Partly; narrower than mutation and control flow, but adjacent to general state tracing/decomposition | Yes: Q2 V2 already ran `STATE_TRACE` and `DECOMPOSE` source policies | Reject: prospective choice could be biased by an already behaviorally exposed neighboring axis |
| `SCOPE_BINDING_SHADOWING` | resolve lexical/comprehension bindings and shadowed names explicitly | treat repeated names as context-obvious local values | 6 / 82 | Yes | No exact prior source policy found | Reject: too few unambiguous feature-positive examples for a 24-item construction side, let alone held-out qualification |
| `SHORT_CIRCUIT_EVALUATION` | track evaluation order and skipped operands | evaluate Boolean conditions as an undifferentiated truth test | 6 / 82 | No; primarily a control-flow/path subtype | No exact prior source policy found | Reject: sparse and redundant with `CONTROL_FLOW_PATH_COVERAGE` |
| `TYPE_COERCION_SEMANTICS` | track runtime type changes/coercions | reason from value gist across conversions | 14 / 74 | Weak; overlaps exact API contracts and representation tracking | Yes: Q2 V2 ran `TYPE_REP`; Amendment 1 ran API exactness | Reject: contaminated and conceptually redundant |
| `EXCEPTION_ERROR_PATH` | propagate exception/error paths | assume ordinary return-path execution | 1 / 87 | Weak; exception branches are control-flow and API-contract subcases | No exact prior source policy found | Reject: one unambiguous example and strong redundancy |
| `RECURSION_BASE_CASE` | unfold recursive calls and verify the base case | reason from recursive intent | 1 / 87 | Yes | No exact prior source policy found | Reject: one unambiguous example |
| `ORDERING_SIDE_EFFECT_DEPENDENCE` | preserve operation order when side effects interact | treat operations as order-insensitive local updates | 10 / 78 | No; overlaps mutation/alias causality and loop update order | No exact prior source policy found | Reject: sparse and redundant |
| `DATA_STRUCTURE_INVARIANT` | maintain global size/order/multiplicity invariants | compute local values without an invariant ledger | 22 / 66 | Partly | Yes: Q2 V2 ran an `INVARIANT` source policy | Reject: below the 24-example construction target, behaviorally exposed neighbor, and partial mutation/loop overlap |

## Availability and matching assessment

- Input provenance records: 336.
- Unopened primary-panel records excluded: 200.
- Prior source-construction records excluded: 24.
- Prior source-validation records excluded: 24.
- Disjoint static inventory records: 88.
- Static projection SHA-256:
  `c8cbe0a271091e07d9090bb79d389549ab6bb0af0ccf44cdeb37e8be1ead970d`.

Only `INTERMEDIATE_DATAFLOW_COMPOSITION` has at least 24 feature-positive and
24 feature-negative records. It is not selected because its nearest clean
mechanistic interpretation sits between two behaviorally exposed Q2 V2 source
policies (`STATE_TRACE`, `DECOMPOSE`). Selecting it now would not be a clean
counterfactual fifth axis independent of prior model behavior.

The other candidates either lack enough source support or collapse into a
surviving family. There is therefore no unique concept that dominates on
coverage, distinctness, and prospective cleanliness.

## Counterfactual audit

If `API_CONTRACT_EXACTNESS` had never been proposed, the substrate would still
suggest several possible fifth axes, not one uniquely natural axis. The one
well-populated candidate is historically exposed at the policy-family level;
the clean candidates are too sparse. Choosing one merely because API failed
would add researcher degrees of freedom rather than conceptual coverage.

Recommendation: `PATH_B_FOUR_FAMILY`.
