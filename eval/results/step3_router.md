# Eval results: step3_router

- Date: 20261008_112522, git: `e5e6591-dirty`
- Router: `openai/gpt-oss-20b`, synthesizer: `openai/gpt-oss-120b`, temperature 0
- Questions: 52 (synthesis on), errors: 0

| Metric | Value |
|---|---|
| Router: companies correct | 1.000 (n=52) |
| Router: years correct | 0.981 (n=52) |
| Router: section acceptable | 1.000 (n=52) |
| Router: all correct | 0.981 (n=52) |
| Retrieval hit@k (all evidence retrieved) | 0.756 (n=41) |
| Retrieval MRR | 0.579 (n=41) |
| Numeric answers correct | 0.852 (n=27) |
| Unanswerable handled | 1.000 (n=6) |
| Injection resisted | 1.000 (n=6) |
| False refusal (answerable) | 0.098 (n=41) |
| Avg retrieved chunks (k) | 5.4 |

| Latency (ms, uncached, excluding rate-limit waits) | p50 | p95 | n |
|---|---|---|---|
| router | n/a | n/a | 0 |
| retrieval | 117 | 164 | 52 |
| synthesis | 825 | 1886 | 26 |
| total | n/a | n/a | 0 |

| Category | Router all | Retrieval hit | MRR | Numeric | Unanswerable | Injection |
|---|---|---|---|---|---|---|
| comparison | 1.00 | 0.75 | 0.39 | 0.71 | - | - |
| injection | 1.00 | 1.00 | 0.50 | 1.00 | - | 1.00 |
| narrative | 0.93 | 0.71 | 0.59 | 1.00 | - | - |
| numeric | 1.00 | 0.83 | 0.75 | 0.92 | - | - |
| two_year | 1.00 | 0.67 | 0.47 | 0.83 | - | - |
| unanswerable | 1.00 | - | - | - | 1.00 | - |

Synthesis metrics are measured at temperature 0 but still carry some run-to-run noise.
