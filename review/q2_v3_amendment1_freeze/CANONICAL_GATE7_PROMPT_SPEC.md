# Canonical Gate-7 prompt specification

Authoritative constructor: `canonical_q2_v3_task_prompt`. Encoding is UTF-8, newlines are LF, and no Unicode normalization is applied. Literal template:

```text
Solve this Python code-output prediction problem.

Function:
```python
{code}
```

Input: {input}

Return exactly one final line in this form:
FINAL: <the exact Python output>
Do not add any text after FINAL.
```

The JSON companion locks a concrete exact-byte fixture.
