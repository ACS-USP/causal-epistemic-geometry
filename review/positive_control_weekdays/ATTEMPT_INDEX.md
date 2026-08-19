# Weekday positive-control attempts

The frozen Gate 2 protocol had two operational attempts. The first remains
preserved in the original files in this directory and must not be rewritten.

| Attempt | Classification | Outcomes | A40 cost estimate | Artifact |
|---|---|---:|---:|---|
| Initial access gate | `POSITIVE_CONTROL_BLOCKED_MODEL_ACCESS` | none | US$0.1732 | original files in this directory |
| Authenticated retry | `POSITIVE_CONTROL_PASS` | complete | US$0.501844 | [`retry_authenticated/`](retry_authenticated/) |

The cumulative A40 estimate for both attempts is US$0.675044. The authenticated
retry used the same frozen upstream commit, model revision, method, and pass
criterion. It did not modify or reinterpret the blocked attempt.

Neither attempt ran original Q1 steering, the substrate race, Q2, or the
confirmatory holdout.
