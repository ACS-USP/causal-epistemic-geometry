# Stage-B parser static semantic comparison

This is a post-closeout, outcome-independent comparison. The hash-pinned primary
parser remains authoritative. The historical independent audit is preserved.

| Logical stage | Frozen primary behavior | Historical audit behavior | Normative ruling |
|---|---|---|---|
| Think visibility | Closed `<think>` blocks are removed | Equivalent | `IMPLEMENTATIONS_EQUIVALENT_ON_THIS_RULE` |
| Unclosed think | Fails closed | Equivalent | `IMPLEMENTATIONS_EQUIVALENT_ON_THIS_RULE` |
| Truncation | Terminal truncation fails closed | Equivalent in parser path | `IMPLEMENTATIONS_EQUIVALENT_ON_THIS_RULE` |
| Marker recognition | Uses the shared external-semantic-v3 marker grammar | Audit used narrower independent regular expressions for repair | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` on repair edges |
| `FINAL` / `FINAL ANSWER` / `FINAL SECTION` | Distinguishes all three; heading-only `FINAL` is `FINAL_SECTION` | Collapses empty headings into one regular expression | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` on the exact repair contract |
| Empty headings | Allows only mechanically empty `FINAL ANSWER` or `FINAL SECTION` before one terminal `FINAL` | Allows regex-matched empty headings | Equivalent on 29 realized two-marker forms |
| Direct commitment | external-semantic-v3 result is retained | Uses the same frozen low-level extractor | Equivalent until literal/type decisions |
| Repair fallback | If the direct commitment is not semantically evaluable, the frozen repair is still attempted | Repair was attempted only when no direct payload existed | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` |
| Fenced literals | Only supported typed candidates (`bool`, `int`, `list`, `str`) enter repair competition | Any Python/JSON canonical type entered competition | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` |
| Standalone literals | Only supported typed candidates enter repair competition | Tuples/dicts/sets/floats could enter competition | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` |
| Competing candidates | A distinct supported typed literal rejects repair | Any distinct canonical literal rejected repair | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` |
| Identical candidates | Identical supported candidates do not compete | Identical canonical candidates do not compete | `IMPLEMENTATIONS_EQUIVALENT_ON_THIS_RULE` |
| Python parsing | `ast.literal_eval`, then JSON fallback | Same order | `IMPLEMENTATIONS_EQUIVALENT_ON_THIS_RULE` |
| JSON parsing | Deterministic JSON fallback | Equivalent | `IMPLEMENTATIONS_EQUIVALENT_ON_THIS_RULE` |
| Canonical types | Type-tagged canonical representation | Equivalent representation, but broader repair domain | Representation equivalent; repair domain deviates |
| Expected reference type | Applied only after a repaired candidate is selected; a direct type mismatch remains evaluable and wrong | Applied to every parsed payload and invalidated direct type mismatches | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` |
| Expected reference value | Used only for final exact correctness comparison | Same | `IMPLEMENTATIONS_EQUIVALENT_ON_THIS_RULE` |
| Mechanical repetition | Blocks repair when the frozen direct path is not evaluable | Historical auditor omitted this gate | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` |
| Commitment validity | Retains a recognized direct commitment even if its literal is unevaluable | Could reset validity on direct type mismatch | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` |
| Semantic evaluability / correctness | Exact typed parsing; wrong type/value is evaluable-wrong on the direct path | Direct type mismatch was invalid/unevaluable | `PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES` |

Neither implementation receives condition identity. Neither uses correctness or
the reference value to select a candidate. The primary executable source hash
matches the prospective lock; no primary implementation drift was found.
