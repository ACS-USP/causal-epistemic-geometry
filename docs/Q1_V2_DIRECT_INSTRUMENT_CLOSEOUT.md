# Q1 V2 — Direct-Answer Instrument Closeout

Q1 V2 / E3-10 is formally closed as a **DEVELOPMENT measurement-instrument
ablation**. It is not V1.3, and it does not replace the project-level question.

The instrument removed multiple-choice answer slots and scored the first
response-state logits for the semantic candidates. On the frozen Qwen3-8B
snapshot, with thinking disabled, those direct candidate scores were close to
chance and the required output-surface stability did not hold. The frozen
qualification rule therefore returned:

```text
E3_10_INSTRUMENT_NOT_QUALIFIED
```

This result means that this **direct first-state readout** was not a clean
measurement instrument for the present model and task families. It does not
show that procedural reasoning tasks are unsuitable for a reasoning model. It
does not establish that activation steering cannot change semantic error
structure. No Q1 scientific result is frozen.

The following remain true:

- the earlier MMLU-Pro V1–V1.2 multiple-choice series is closed as DEVELOPMENT;
- the E3-10 generators, oracles, and model-free audits remain useful software
  and ablation infrastructure;
- no activation direction, PCA, random control, DEV evaluation, or
  confirmatory holdout was run under E3-10;
- the confirmatory holdout remains untouched.

The structural lesson is to measure the model's sampled reasoning policy when
the model is configured to reason, rather than infer semantic competence from a
single direct answer-logit slice. That motivates the separate Q1 V3 protocol
in [Q1_V3_REASONING_AGENT_PROTOCOL.md](Q1_V3_REASONING_AGENT_PROTOCOL.md).

This closeout must not be read as a positive or negative result about
complementarity. It is an instrument decision only.
