# External Semantic Evaluator V3

Version: `external-semantic-v3`

Status: offline diagnostic instrument. It does not modify the historical
`external-semantic-v2` result or `GATE6_3_SINGLE_MEAN_DESTRUCTIVE` classification.

## Three independent axes

1. **Commitment validity** — exactly one unambiguous final commitment exists.
2. **Semantic evaluability** — that commitment has a deterministic tagged value.
3. **Correctness** — the tagged value equals the tagged reference value.

A wrong-type value is a valid, evaluable, wrong commitment. Mechanical failure
is reserved for no final commitment, multiple final commitments, an empty or
ambiguous final section, truncation/unclosed thinking, runtime failure, or an
unmatched syntactic delimiter.

## Final commitment grammar

V3 accepts one complete `FINAL: value` line with optional matched Markdown
heading/list/checkmark, emphasis, inline-code, or enclosing code-fence syntax.
It also accepts one terminal `Final Answer:` or decorated `### Final:` section
whose payload begins on the next line and may be fenced or multiline. Such a
section may contain one subordinate `FINAL: value` line; this pair is one
commitment. An inline `FINAL:` may continue as a multiline string only when
every subsequent non-empty payload line is indented; unindented suffix prose
remains invalid.

An inline final commitment must be followed only by whitespace and matched
fence/emphasis closure. Substantive suffix text is invalid. A final section
extends to the end of the visible response; after its matched closure only
whitespace is allowed. Multiple or conflicting final markers are invalid.
Arbitrary values are never extracted from reasoning prose.

## Whitespace and delimiter normalization

Only syntactic outer whitespace, empty separator lines, and matched Markdown
fence/emphasis delimiters are removed. Internal newlines and each payload line's
leading/trailing spaces are preserved. V3 does not collapse spaces inside string
values.

## Typed canonical values

`ast.literal_eval` is the only Python-literal evaluator. Generated code is never
executed. Tagged canonical forms cover `None`, `bool`, `int`, `float`, `str`,
`bytes`, `bytearray`, `list`, `tuple`, `dict`, `set`, and `frozenset`. Dictionary
and set entries receive deterministic ordering. Bytes are represented by hex.

Safe explicit adapters recognize `bytearray(<bytes literal>)` and
`frozenset(<literal collection>)`; their inner value is still parsed only by
`ast.literal_eval`. Any other non-literal but unambiguous payload becomes an
exact tagged string. Therefore `FINAL: IndexError` against a list reference is
semantically evaluable and wrong, not mechanically invalid.

## Condition blindness

The parser API receives only raw output, reference text, and truncation/runtime
flags. It has no item-ID or condition argument. Rules were implemented and
tested against synthetic cases and a condition-masked corpus before condition
labels were restored for aggregation. No LLM judge or row-specific exception is
allowed.
