# Paper 1 claims and evidence matrix

Status vocabulary:

- `CONFIRMATORY`: supported by the frozen Q1 holdout rule.
- `DEVELOPMENT`: supported on prospectively separated development data.
- `POST_HOC_DESCRIPTIVE`: discovered after outcomes and not claim-promoting.
- `NEGATIVE_RESULT`: a frozen test failed or bounded a claim.
- `NOT_ESTABLISHED`: not tested adequately or not run.

| Claim | Status | Supporting experiments and metrics | Exact caveat |
|---|---|---|---|
| The fixed Qwen controller produces causal complementarity on CRUXEval | `CONFIRMATORY` | Q1 holdout: C=0.05435, 95% interval [0.01441, 0.09680]; C exceeds all four random controls; C-minus-random-mean interval positive | Model-, task-, controller-, dose-, and policy-specific |
| Qwen complementarity is safety-qualified | `CONFIRMATORY` | Commitment/evaluability 0.97368 versus baseline 0.98246; frozen absolute, relative, and competence guards pass | Does not imply safety under other doses, tasks, or deployment settings |
| The fixed Ministral controller produces causal complementarity on CRUXEval | `CONFIRMATORY` for the complementarity components | C=0.07299, interval [0.02177, 0.12281]; delta-C versus null mean interval [0.02573, 0.10491]; C above every random; accuracy +0.07895 | The model-level composite decision fails because safety guards fail; do not call this a model pass |
| Ministral complementarity is safety-qualified | `NEGATIVE_RESULT` | Commitment validity and evaluability 0.88596 versus baseline 0.96491 | Below both frozen absolute and relative guards; no post-hoc recovery changes this |
| Both model families safely realize complementarity | `NEGATIVE_RESULT` | Terminal classification `Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL` | The conjunctive cross-model safety claim failed |
| A cross-model causal complementarity phenomenon exists on CRUXEval | `CONFIRMATORY` for positive/null-specific C components, not for joint safety | Qwen and Ministral both have positive C intervals and exceed their frozen random banks | Phrase as cross-model complementarity evidence, not cross-model safe-controller confirmation |
| Ministral invalidity is primarily commitment/generation instability | `POST_HOC_DESCRIPTIVE` | Aggregate audit: 3 token-cap, 10 commitment-structure failures; 9 recoverable correct; invalid rows zero rescues and 11/16 damage pairs | Same-analyst taxonomy after outcomes; mechanism requires prospective replication |
| The Qwen L27-D75 controller is domain-general | `NEGATIVE_RESULT` | Gate 10 long character count: G/C/D=-0.01625/-0.01230/-0.025, below random mean; rescue < damage | Negative for this controller/domain transfer, not all possible controllers or domains |
| Cross-domain transfer to long character counting | `NEGATIVE_RESULT` | Gate 10 opportunity and safety pass, transfer classification `GATE10_NO_CROSS_DOMAIN_TRANSFER` | Character counting may reward a different policy; no task adaptation was allowed |
| Strong readout implies strong causal controllability | `NEGATIVE_RESULT` as a ranking heuristic | Gate 6 source/RFM atlas and Gate 13 shortlist: widespread readout eligibility did not identify the final causal layer reliably | Does not show readout is useless; it shows readout alone does not rank control handles |
| Finite-displacement output or hidden movement is exact pullback/Fisher geometry | `NOT_ESTABLISHED` | Gate 11/11.1 preserved KL/JS and hidden displacement; Gate 12/12.1 stopped at engineering qualification | These are finite-shift control diagnostics, not a measured exact local metric |
| Internal geometry predicts error-profile geometry | `NOT_ESTABLISHED` | No scientific Q2 outcome exists | Gate 12 collected zero scientific geometry shards; Q2 is not run |
| Euclidean geometry is inferior to pullback geometry | `NOT_ESTABLISHED` | No prospectively held-out metric comparison | Neither positive nor negative Q2 evidence exists |
| The controllers implement a transferable careful-computation policy | `NEGATIVE_RESULT` in its broad form | CRUXEval benefits versus character-count null/harm; Gate 11 policy-utility mismatch | A domain-conditioned program-tracing policy remains plausible in development |
| Complementarity can be converted into implementable collective utility | `NOT_ESTABLISHED` | Pair-oracle headroom and rescue/damage are measured | No deployable selector/router/committee has been tested; Q3 is not run |
| The project has a universal controller | `NOT_ESTABLISHED` | Architecture-specific Qwen and Ministral vectors/layers/doses were selected separately | Similar source concept is not vector identity or cross-architecture transport |

## Paper-level claim boundary

The strongest current paper is not “safe cross-model steering succeeded” and
not “cross-model replication failed.” It is:

> Semantic error-profile complementarity is causally controllable and
> confirmatory-safe in Qwen; the complementarity component reproduces in
> Ministral while safe commitment realization does not. Cross-domain evidence
> bounds the current controllers as task-conditioned.

No sentence in the manuscript should imply that Q2 geometry prediction or Q3
deployable utility has been established.
