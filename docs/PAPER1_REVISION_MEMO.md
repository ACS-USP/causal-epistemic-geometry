# Paper 1 revision memo

No manuscript source (`.tex`, `.qmd`, `.docx`, or manuscript Markdown) is
tracked in the repository at the Q1 canonical closeout. This memo therefore
provides section-level replacement text for the external draft; it does not
pretend to reconstruct that draft.

## Recommended title

**Steering Blind Spots: Causal Control of Semantic Error Profiles in Language
Models**

This title is preferable to *Causal Control of Language-Model Blind Spots*.
“Steering Blind Spots” is memorable, while “semantic error profiles” states the
measured object and limits the claim. The shorter alternative is stronger
rhetorically but can be read as claiming domain-general or arbitrary control,
which Gate 10 does not support.

## Recommended central claim

Prospectively fixed, architecture-specific activation controllers can causally
reorganize semantic error profiles beyond repeated sampling and matched random
directions. Qwen satisfies the full frozen confirmatory criterion, including
safety. Ministral reproduces positive and random-specific complementarity but
fails the frozen commitment-validity/evaluability criterion. Thus
complementarity replicated more robustly than safe policy realization. The
phenomenon is strong within CRUXEval but not established as domain-general.

## Replacement abstract

Language models can have similar average accuracy while failing on different
items, creating potential complementarity. We ask whether a frozen model's
semantic error profile can be changed causally by an internal intervention,
rather than by ordinary stochastic resampling or generic perturbation. We
construct careful-versus-direct prompt-boundary directions and apply sustained
current-token activation steering in two 8B instruction-tuned model families.
Development experiments identify architecture-specific layer and dose choices;
the final controllers are then frozen before a prospectively assigned 57-item
CRUXEval holdout. In Qwen3-8B, steering increases competence-adjusted
complementarity to C=0.0544 (95% item-bootstrap interval 0.0144 to 0.0968),
exceeds every frozen random control, and passes commitment-validity,
evaluability, and competence safeguards. In Ministral-3-8B, C=0.0730 (95%
interval 0.0218 to 0.1228), the meaningful-minus-random-mean contrast is
positive, and accuracy improves, but commitment validity and semantic
evaluability fall to 0.886 and fail the frozen safety rule. The conjunctive
cross-model classification is therefore
`Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`. A labeled post-hoc analysis suggests
that Ministral's invalidity is dominated by commitment-format and generation
instability rather than semantic degeneration; this interpretation is
descriptive and does not alter the confirmatory decision. A fixed Qwen
controller also fails to transfer to long character counting, bounding the
effect to the tested task family. These results establish safe confirmatory
control in Qwen, cross-model evidence for causal complementarity with a safety
dissociation in Ministral, and a clear separation between readable directions,
causal control, and domain-general utility.

## Introduction thesis paragraph

Most activation-steering studies ask whether an intervention raises the mean
rate of a desired behavior. We study a different causal object: *where* a model
fails. Let each intervention induce an item-level error-propensity profile. Two
variants can be useful together when their errors overlap less than expected
from their individual competence, even if neither dominates every item. The
central question is therefore not only whether steering changes accuracy, but
whether a prospectively specified internal intervention moves semantic blind
spots beyond repeated baseline sampling and architecture-matched random
perturbations while preserving a reliable answer channel. This distinction
turns activation steering from scalar behavior control into causal control of
error-profile structure.

## Contribution list

1. **Estimand.** A two-rollout, item-clustered framework separates stochastic
   baseline resampling, raw error-profile movement, competence-adjusted
   complementarity, rescue, damage, and mechanical validity.
2. **Causal development sequence.** Published-method positive control,
   sustained-current-token engineering, source validation, prospective dose
   calibration, and fresh random-bank evaluations isolate a reproducible Qwen
   controller without outcome-dependent rescue.
3. **Model-specific confirmation.** A frozen Qwen controller passes all
   confirmatory safety, positive-C, and random-null criteria on the sealed
   holdout.
4. **Cross-model dissociation.** A separately developed Ministral controller
   shows confirmatory positive and null-specific complementarity and improved
   accuracy, yet fails the frozen commitment/evaluability safety rule.
5. **Boundary evidence.** The Qwen controller's long character-count transfer
   is negative, and source/readout strength does not reliably rank causal
   controllability.
6. **Reproducibility.** Logical schedules, independent seeds, controller and
   parser locks, random banks, item-cluster bootstrap, and independent forensic
   recomputation are preserved for every terminal claim.

## Central Results narrative

### 1. Development identifies a safe Qwen controller

Initial one-shot and fixed-site directions were inert or destructive. A
sustained layer-27 paired-mean controller produced large semantic movement at
full dose but harmed validity. Prospective matched calibration selected D75 as
the lowest safe dose without using G, C, D, or accuracy as optimization
objectives. On 100 fresh development items, that fixed choice increased
accuracy from 0.47 to 0.60 and produced G/C/D=0.1325/0.0643/0.1200, exceeding
four new matched random directions with rescue 0.1525 versus damage 0.0225.

### 2. Cross-model development requires architecture-specific identification

Directly transplanting a layer or source-decoding shortlist was insufficient
in Ministral. A prospectively frozen all-layer causal atlas and disjoint
layer-dose qualification selected L27-D25 without ranking by accuracy. On 100
untouched development items, accuracy rose from 0.445 to 0.575 and
G/C/D=0.1650/0.0940/0.1400 exceeded four fresh random controls. Validity passed
at the exact allowed relative boundary, motivating but not replacing the final
holdout.

### 3. Qwen passes the frozen confirmatory test

On the shared 57-item holdout, Qwen's fixed controller increased accuracy from
0.4386 to 0.5000. C was 0.05435, its item-bootstrap interval was [0.01441,
0.09680], and the meaningful-minus-random-mean C interval was also positive.
Meaningful C exceeded every frozen random C. Commitment validity and semantic
evaluability were 0.97368 versus baseline 0.98246, satisfying both absolute and
relative guards; competence also passed. Qwen therefore meets the complete
model-specific confirmatory criterion.

### 4. Ministral confirms complementarity but not safe realization

Ministral's fixed controller increased accuracy from 0.5702 to 0.6491. C was
0.07299 with interval [0.02177, 0.12281]; the C-minus-random-mean interval was
[0.02573, 0.10491], and C exceeded every frozen random controller. Rescue was
0.1491 versus damage 0.0702. However, commitment validity and semantic
evaluability fell from 0.96491 to 0.88596, below the frozen absolute and
relative safeguards. The model-specific confirmatory decision is therefore a
fail, and the conjunctive classification remains
`Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`.

This result should not be called simply a “failed replication.” The causal
complementarity dimensions replicate; safe policy realization does not. A
post-hoc descriptive audit attributes the 13 invalid meaningful rows to three
token-cap cases and ten commitment-structure failures. Nine contained a
human-recoverable correct answer, but those recoveries never enter the frozen
metrics. Invalid rows generated zero rescues and 11 of 16 damage pairs, so they
attenuated rather than manufactured complementarity.

### 5. The effect is not established as domain-general

The exact Qwen L27-D75 controller was transported without adaptation to 200
fresh long character-count items. Baseline opportunity and safety passed, but
G/C/D were -0.01625/-0.01230/-0.025 and below the random mean, with rescue below
damage. This negative result rules out a broad claim that the controller
implements a task-independent careful-computation policy.

## Discussion replacement

The results support a causal phenomenon narrower and more informative than a
generic steering success. Internal interventions can change the overlap
structure of semantic errors, and this effect can survive architecture-specific
development into a sealed holdout. Yet complementarity, answer-channel safety,
and cross-domain utility are separable. Qwen realizes all three requirements on
the tested semantic task. Ministral realizes complementarity and competence but
not the frozen commitment/evaluability standard. Character counting shows that
careful-like computation can have different utility across tasks.

This separation also clarifies what representation evidence can and cannot do.
Careful/direct states are readable across many layers, but the most readable
layers did not consistently provide causal control. Finite-displacement hidden
and output changes likewise did not establish an exact local pullback/Fisher
geometry or predict utility. Readout, control gain, policy realization, and task
utility should therefore be measured as distinct objects.

The natural next question is geometric but remains untested: can a metric on a
diverse intervention bank predict held-out distances between error-propensity
profiles? The present paper supplies controlled interventions and audited error
profiles; it does not claim a geometry-prediction result.

## Limitations replacement

- The full safety-qualified confirmatory result is model-specific to Qwen; the
  conjunctive two-model rule fails because Ministral's commitment channel is
  fragile under steering.
- The confirmatory holdout has 57 items. Item-cluster bootstrap and independent
  forensic recomputation address uncertainty and integrity, not breadth.
- Both positive controllers derive from careful-versus-direct prompting on
  CRUXEval. Architecture-specific layer/dose identification limits claims of a
  universal vector.
- Cross-domain transfer to long character counting is negative. A second
  positive objective task is not yet available.
- Post-hoc human recovery and failure taxonomy are descriptive; they cannot
  repair validity or establish a safer Ministral policy.
- Random directions are strong matched controls, but four randoms per final
  test characterize only a finite null bank.
- Longer generations co-occur with meaningful steering. Token length is a
  mechanistic correlate and possible mediator/confound, not evidence that
  verbosity itself causes complementarity.
- Readout-controllability and geometry-utility relationships remain
  unestablished. Q2 and Q3 have not been run.

## Confirmatory-status language

Use:

> Qwen passed the complete prospectively frozen model-specific confirmatory
> criterion. Ministral showed confirmatory positive and random-specific
> complementarity but failed the frozen commitment-validity/evaluability safety
> criterion. The conjunctive cross-model decision therefore failed.

Avoid:

- “Both models passed.”
- “Cross-model confirmation failed, so the effect did not replicate.”
- “Ministral partially passed.”
- “The parser caused Ministral to fail.”
- “The controller is domain-general.”
- “Geometry explains controllability.”

The compact formula is: **complementarity replicated more robustly than safe
policy realization**, with the failure-mode explanation labeled post-hoc.
