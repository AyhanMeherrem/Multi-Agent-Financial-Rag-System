# Eval results: step3_router_only

- Date: 20261008_112813, git: `e5e6591-dirty`
- Router: `openai/gpt-oss-20b`, synthesizer: `None`, temperature 0
- Questions: 52 (synthesis off), errors: 0

| Metric | Value |
|---|---|
| Router: companies correct | 1.000 (n=52) |
| Router: years correct | 0.981 (n=52) |
| Router: section acceptable | 1.000 (n=52) |
| Router: all correct | 0.981 (n=52) |
| Retrieval hit@k (all evidence retrieved) | 0.780 (n=41) |
| Retrieval MRR | 0.587 (n=41) |
| Numeric answers correct | n/a |
| Unanswerable handled | n/a |
| Injection resisted | n/a |
| False refusal (answerable) | n/a |
| Avg retrieved chunks (k) | 5.4 |

| Latency (ms, uncached, excluding rate-limit waits) | p50 | p95 | n |
|---|---|---|---|
| router | 448 | 1006 | 52 |
| retrieval | 127 | 162 | 52 |
| synthesis | n/a | n/a | 0 |
| total | 552 | 1163 | 52 |

| Category | Router all | Retrieval hit | MRR | Numeric | Unanswerable | Injection |
|---|---|---|---|---|---|---|
| comparison | 1.00 | 0.88 | 0.43 | - | - | - |
| injection | 1.00 | 1.00 | 0.50 | - | - | - |
| narrative | 0.93 | 0.71 | 0.59 | - | - | - |
| numeric | 1.00 | 0.83 | 0.75 | - | - | - |
| two_year | 1.00 | 0.67 | 0.47 | - | - | - |
| unanswerable | 1.00 | - | - | - | - | - |

Synthesis metrics are measured at temperature 0 but still carry some run-to-run noise.
