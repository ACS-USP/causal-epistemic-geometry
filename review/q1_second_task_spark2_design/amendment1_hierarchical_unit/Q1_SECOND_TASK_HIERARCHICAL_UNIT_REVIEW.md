# Q1 Second-Task Hierarchical-Unit Review

## 1. What a LiveCodeBench question family is

The pinned `livecodebench/test_generation` pool uses `question_id` as the
question-family identity. All rows in one family share the complete problem
statement, source/starter program, function name, title, contest, date, and
difficulty metadata. They differ in `test_id` and in the exact test input and
reference output. Thus they are repeated probes of one program-execution
problem, not independent programs. The audit used only the pinned official
dataset revision and no model output, correctness, or historical LiveCodeBench
result.

The scientifically defensible superpopulation unit is therefore the question
family. A row remains an objective test-output observation nested within that
family.

## 2. Family-size distribution

The pool contains 442 rows nested in 182 families.

| Rows per family | Families | Rows contributed |
|---:|---:|---:|
| 2 | 105 | 210 |
| 3 | 76 | 228 |
| 4 | 1 | 4 |

The minimum, median, mean, p90, and maximum are respectively 2, 2, 2.429, 3,
and 4 rows. The 77 families with at least three rows contribute 232/442 rows
(52.49%). Every family has unique test IDs and distinct test/reference payloads.

## 3. Problem with or defense of row-level independence

The old row design is not defensible as the primary uncertainty model. It gives
more weight to questions with more tests and a row bootstrap breaks the shared
program/problem cluster. Positive within-family dependence is structurally
plausible even though its empirical magnitude cannot be estimated without
opening model outcomes.

In the old Stage B, 150 rows came from only 62 families: 37 families supplied
two rows, 24 supplied three, and one supplied four. Equal family weight would
be 1.613%; row weighting ranged from 1.333% to 2.667%, a 2x ratio. Under
intraclass correlation rho of 0.25, 0.50, 0.75, and 1.00, the row-weighted
effective independent counts fall from 150 to 108.43, 84.91, 69.77, and 59.21.

The corresponding old Stage A was 50 rows/21 families (maximum family weight
6.0% versus 4.76% equal weight), and reserve was 242 rows/99 families. The old
item/row bootstrap is therefore classified as structurally anti-conservative
when within-family dependence is positive, and it is rejected prospectively.

This ruling also agrees with the repository's existing canonical test
`test_cluster_bootstrap_uses_problem_not_test_as_scientific_unit`.

## 4. Design A — row-level

Design A retains 150 test rows from 62 families and bootstraps rows. Its
estimand is the average over test cases, implicitly weighting each question by
its number of available tests. That could be a legitimate descriptive
test-case estimand if uncertainty were clustered, but the proposed independent
row bootstrap is not legitimate for the question-level scientific claim.

Its previously reported 86.2% full-transfer power assumes 150 independent
units and must not be cited for this benchmark design. Dependence sensitivity
reduces joint full-transfer planning power from 86.3% at rho=0 to 74.5%, 63.8%,
54.2%, and 46.5% at rho=0.25, 0.50, 0.75, and 1.00. Design A is rejected.

## 5. Design B — family-clustered

Design B is statistically valid if each family is reduced first and families
are then equally weighted. For family `f`, test row `t`, rollout `r`, and binary
error `e`, define:

`ebar[f,r] = (1/m_f) sum_t e[f,t,r]`.

The canonical pooled-R4 formulas then operate on `ebar` at the family level:
within-condition products average over `r != s`; cross-condition products use
independent rollout banks; between-family U-statistics exclude identical
families; and all rows, conditions, and rollouts of a bootstrapped family move
together. Rescue and damage are likewise computed from family rollout means.
The resulting estimand is the equally weighted average question-family
propensity, not a row-weighted test-case propensity.

For 32 Stage-A and 130 Stage-B families, Design B would process 84 and 310 raw
rows, producing 336 and 13,640 trajectories. It is valid and can gain precision
when tests within a family provide nonredundant information, but its advantage
depends on unknown intrafamily correlation. It is more than twice as expensive
as Design C and does not increase the number of independent question families.

## 6. Design C — one row per family

Design C deterministically chooses exactly one row from every allocated family.
The representative is the minimum stable SHA-based digest of the frozen
experiment namespace, family ID, and stable item ID. This is preferable to the
smallest test ID because it avoids systematic dependence on upstream test
ordering while remaining deterministic and auditable. No outcome, correctness,
difficulty, token length, or semantic property enters selection.

The amended split is:

| Role | Families | Selected rows | Rollouts | Conditions | Trajectories |
|---|---:|---:|---:|---:|---:|
| Stage A | 32 | 32 | 2 | 2 | 128 |
| Stage B | 130 | 130 | 4 | 11 | 5,720 |
| Reserve | 20 | 48 raw rows retained | - | - | 0 |

All 130/130 selected Stage-B rows are distinct question families by design;
the scientific and bootstrap unit is `QUESTION_FAMILY`, equally weighted. The
canonical R4 G/C/D/rescue/damage algebra applies unchanged because one selected
row represents each family. All conditions and rollouts for a family move
together in the 50,000-resample bootstrap. The predesignated rollout halves
remain `{0,1}` and `{2,3}`, and negative finite-sample D remains untrimmed.

## 7. Power comparison

Planning uses the same frozen historical Qwen effect and eight-null rule as the
parent design, solely as planning inputs. Each cell has 100,000 deterministic
Monte Carlo replicates. Null false-positive below is the full frozen conjunctive
rule, not an isolated nominal test.

### Design C: independent families, R=4

| Families | C CI width | Delta-C CI width | Joint power 100% | Joint power 75% | Null false-positive |
|---:|---:|---:|---:|---:|---:|
| 80 | 0.0684 | 0.0569 | 0.605 | 0.347 | 0.0020 |
| 100 | 0.0612 | 0.0509 | 0.710 | 0.443 | 0.0023 |
| 120 | 0.0558 | 0.0465 | 0.786 | 0.523 | 0.0021 |
| **130** | **0.0537** | **0.0446** | **0.815** | **0.557** | **0.0022** |
| 140 | 0.0517 | 0.0430 | 0.842 | 0.590 | 0.0022 |

At N=130, component planning probabilities under full transfer are 97.8% for
positive C lower bound, 92.4% for positive meaningful-minus-null-mean lower
bound, 85.7% for exceeding every random point estimate, and 99.9% for the
predesignated split-half sign rule. Monte Carlo standard error for the 81.5%
joint estimate is about 0.12 percentage point.

N=130 is the smallest evaluated independent-family design exceeding the
parent's prospective approximately-80% full-transfer target. N=140 adds only
2.7 percentage points of full-transfer joint power while consuming half of the
remaining 20-family reserve. At 75% transfer, power remains only 55.7%; this is
an explicit limitation, not a reason to change the rule.

### Design B: 130 equal-weight families with all 310 nested rows

| Within-family rho | Effective units | Joint power 100% | Joint power 75% |
|---:|---:|---:|---:|
| 0.00 | 298.24 | 0.980 | 0.875 |
| 0.25 | 225.33 | 0.951 | 0.790 |
| 0.50 | 181.07 | 0.912 | 0.705 |
| 0.75 | 151.34 | 0.866 | 0.627 |
| 1.00 | 130.00 | 0.815 | 0.556 |

Design B is robust if analyzed hierarchically, but its precision gain is
unknown before outcomes and vanishes under perfect within-family dependence.
Design C makes no assumption about that dependence.

## 8. Runtime comparison

Projections use the already-qualified synthetic Spark-2 throughput of 11.528
generated tokens/s and the frozen 25% safety margin. They are planning values,
not scientific stop rules.

| Design | Stage A rows | Stage B rows | Combined hours at 128 / 256 / 512 tokens | Storage reservation |
|---|---:|---:|---:|---:|
| A: old row-level | 200 | 6,600 | 26.2 / 52.4 / 104.9 | 1.5 GB |
| B: all rows, family-balanced | 336 | 13,640 | 53.9 / 107.8 / 215.5 | 3.1 GB |
| **C: one row/family** | **128** | **5,720** | **22.5 / 45.1 / 90.2** | **1.3 GB** |

Design C reduces Stage-B trajectories by 13.3% relative to the old proposal
while increasing independent Stage-B families from 62 to 130.

## 9. Stage-A implications

Stage A now contains 32 independent families, one selected row each. Its
baseline opportunity statistics are therefore family-level without fractional
or nested weighting: pooled accuracy and B00 average 32 equally weighted
families; “wrong twice” and “correct at least once” count families.

The frozen validity/evaluability minima (0.95), accuracy interval [0.25, 0.90],
and B00 minimum 0.05 are retained. The old count proportions translate
mechanically before outcomes: 5/50 wrong twice becomes `ceil(0.10*32)=4`, and
10/50 correct at least once becomes `ceil(0.20*32)=7`. No controller or random
condition is opened in Stage A. Stage A remains unauthorized.

## 10. Textual CAREFUL gate review

The textual gate is retained exactly from the historical Gate-10 source-anchor
logic:

- commitment validity and semantic evaluability >=0.95;
- textual accuracy >= baseline accuracy -0.03;
- and at least one of accuracy gain >=0.03, mean tokens >=1.5x baseline, or
  median tokens >= baseline +10.

This does not claim that verbosity is an accuracy benefit. It asks whether the
source policy visibly manifests while preventing material competence harm. A
purely verbose condition that loses more than three accuracy points fails—as
the Gate-10 precedent did. Requiring accuracy improvement in all cases would
replace the intended behavioral source-anchor gate with an efficacy gate and
is not justified pre-outcome.

## 11. Recommended design

Select Design C with 32 Stage-A and 130 Stage-B families. It targets the
scientifically natural equally weighted question-family estimand, eliminates
the unresolved dependence assumption, clears the inherited approximately-80%
full-transfer planning target, retains 20 untouched reserve families, and uses
fewer trajectories than the superseded row design.

No scientific claim is weakened or expanded. Controller, eight-null bank,
rollouts, generation, parser, safety rules, primary endpoints, and terminal
classifications remain inherited unchanged.

## 12. Prospective amendment

Amendment 1 records:

- benchmark outcomes before amendment: 0;
- correctness inspected: NO;
- reason: hierarchical dependence discovered during principal design review;
- old design: 50/150 rows, 21/62 families, row-level bootstrap;
- new design: 32/130 independently selected families, one row/family;
- reserve: 20 complete families/48 raw rows;
- bootstrap and estimand: equal-weight question family;
- Stage-A/Stage-B rows: 128/5,720;
- all old schedules: `SUPERSEDED_PRE_OUTCOME_NEVER_EXECUTED`.

The old lock and manifests remain immutable for provenance. This amendment does
not authorize Stage A or Stage B.

## 13. New frozen manifests and hashes, if amended

Canonical files are in `review/q1_second_task_spark2_design/amendment1_hierarchical_unit/`.
The amendment lock hash-pins the family structure audit, design comparison,
hierarchical estimator specification, 32/130/20 manifests, excluded siblings,
both schedules, power grid/method, runtime comparison, textual-gate review, and
the old-design supersession record. It also pins the inherited controller,
random-bank, estimator/decision, engine, and model-free instrument locks.

The frozen schedules contain 128 and 5,720 unique logical keys; all 5,848
cross-stage seeds are unique. Exact SHA-256 values are in `AMENDMENT_LOCK.json`
and the repository-wide amendment hash manifest.

## 14. Validation

The independent audit reconstructed all 182 families directly from the pinned
442-row source, reapplied the representative-row rule independently, accounted
for all 442 raw rows, verified disjoint family partitions, checked every lock
hash, confirmed old schedules are unchanged and superseded, checked both new
schedules and all seeds, and repeated a deterministic power cell. Its result is
`Q1_SECOND_TASK_HIERARCHICAL_DESIGN_FORENSIC_CLEAN`.

No inference or outcome was required for any validation.

## 15. Repository state

This amendment is isolated on `research/q1-second-task-spark2-design`. The
historical Q1 result, open Q2 campaign, Research OS scientific state, and parent
pre-outcome design artifacts remain unmodified. Stage A has not run.

- LiveCodeBench scientific inference: **0**
- LiveCodeBench correctness inspected: **NO**
- Q2 outputs inspected: **NO**
- Spark 1 used: **NO**
- Spark 2 scientific inference: **NO**
- historical Q1 modified: **NO**

`Q1_SECOND_TASK_HIERARCHICAL_DESIGN_READY_FOR_PRINCIPAL_REVIEW`
