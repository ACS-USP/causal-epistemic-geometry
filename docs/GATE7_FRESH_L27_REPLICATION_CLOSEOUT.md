# Gate 7 fresh single-L27 replication closeout

Gate 7 is closed as an independently audited DEVELOPMENT replication with the
frozen classification:

`GATE7_DESTRUCTIVE`

This classification is not a claim that the controller was behaviorally inert.
It is the pre-registered consequence of a strong, specific causal movement
signal accompanied by excessive loss of mechanical validity.

## Frozen design

- Qwen3-8B at revision `b968826d9c46dd6066d109eabc6255188de91218`;
- full non-thinking BF16/SDPA generation;
- 120 fresh CRUXEval items excluded from 473 historical/reserved IDs;
- exact Gate-6.3 paired-mean plus controller at block 27;
- eta `12.849903937136261`, sustained current-token intervention;
- baseline, textual CAREFUL, meaningful, and four new matched random controls;
- two independent rollouts per item-condition, 1,680 rows total;
- `external-semantic-v3` frozen before collection;
- 10,000 item-cluster bootstrap resamples.

The experiment source commit is
`0dc9c3156bc86aebada93388a4e2fa28b2345f95`.

## Result

Baseline accuracy was 0.3917. The meaningful controller reached 0.5375, an
absolute gain of 0.1458. Its primary profile estimands were:

- G = 0.2375;
- C = 0.1501;
- D = 0.2167;
- rescue = 0.2458;
- damage = 0.1000.

The meaningful-minus-random-mean contrasts were 0.2057 for G, 0.1321 for C,
and 0.1729 for D. Each point estimate also exceeded the maximum of the four
random controllers. The item-cluster bootstrap 95% intervals were positive for
accuracy gain `[0.0542, 0.2375]`, G `[0.1708, 0.3083]`, C `[0.1127, 0.1863]`,
and D `[0.1417, 0.2917]`.

However, commitment validity and semantic evaluability were both 0.9000 under
the meaningful controller versus 0.9917 at baseline. The frozen relative guard
required at least 0.9417. Both guards therefore failed, while the competence
guard passed. The result cannot be reclassified as useful or non-destructive
merely because accuracy and complementarity estimands improved.

The textual CAREFUL source replicated. Its accuracy was 0.6792 and its mean
generated length was 313.4 tokens versus 12.1 at baseline. The activation
controller reproduced a still longer regime, averaging 413.0 tokens.

## Audit and interpretation

The independent audit returned `GATE7_FORENSIC_CLEAN`: all 1,680 logical rows
were present and unique, seeds and schedule matched the lock, semantic V3
reparsing was condition-symmetric, no retries occurred, and independently
recomputed metrics differed by exactly zero.

Gate 7 establishes a replicable development-level tradeoff: the fixed L27
controller causes specific semantic-error movement and improves measured
accuracy, but at the frozen dose it also causes too many ambiguous or
unevaluable final commitments. It does not establish useful
competence-preserving control, Q2 geometry, cross-domain generalization, or a
confirmatory result.

The only drafted next protocol is a separate fresh dose calibration. It has not
been authorized or executed. Character count and the confirmatory holdout were
not accessed.
