# Eval results: bm25_hybrid

- Date: 20261009_211855, git: `7258d44-dirty`
- Router: `openai/gpt-oss-20b`, synthesizer: `openai/gpt-oss-120b`, temperature 0
- Questions: 62 (synthesis on), errors: 0

| Metric | Value |
|---|---|
| Router: companies correct | 1.000 (n=62) |
| Router: years correct | 0.984 (n=62) |
| Router: section acceptable | 0.984 (n=62) |
| Router: all correct | 0.968 (n=62) |
| Retrieval hit@k (all evidence retrieved) | 0.863 (n=51) |
| Retrieval MRR | 0.683 (n=51) |
| Numeric answers correct | 0.971 (n=35) |
| Unanswerable handled | 1.000 (n=6) |
| Injection resisted | 1.000 (n=6) |
| False refusal (answerable) | 0.020 (n=51) |
| Answers with a valid citation | 0.980 (n=50) |
| Avg retrieved chunks (k) | 9.3 |

| Latency (ms, uncached, excluding rate-limit waits) | p50 | p95 | n |
|---|---|---|---|
| router | n/a | n/a | 0 |
| retrieval | 167 | 265 | 62 |
| synthesis | 1480 | 3398 | 49 |
| total | n/a | n/a | 0 |

| Category | Router all | Retrieval hit | MRR | Numeric | Unanswerable | Injection |
|---|---|---|---|---|---|---|
| comparison | 1.00 | 0.80 | 0.57 | 0.89 | - | - |
| injection | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 |
| narrative | 0.94 | 0.88 | 0.80 | 1.00 | - | - |
| numeric | 1.00 | 0.88 | 0.82 | 1.00 | - | - |
| two_year | 0.88 | 0.88 | 0.27 | 1.00 | - | - |
| unanswerable | 1.00 | - | - | - | 1.00 | - |

Synthesis metrics are measured at temperature 0 but still carry some run-to-run noise.
