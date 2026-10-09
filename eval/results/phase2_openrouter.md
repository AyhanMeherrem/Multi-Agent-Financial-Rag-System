# Eval results: phase2_openrouter

- Date: 20261009_190737, git: `c4f9ecc-dirty`
- Router: `openai/gpt-oss-20b`, synthesizer: `openai/gpt-oss-120b`, temperature 0
- Questions: 62 (synthesis on), errors: 0

| Metric | Value |
|---|---|
| Router: companies correct | 1.000 (n=62) |
| Router: years correct | 0.984 (n=62) |
| Router: section acceptable | 0.984 (n=62) |
| Router: all correct | 0.968 (n=62) |
| Retrieval hit@k (all evidence retrieved) | 0.686 (n=51) |
| Retrieval MRR | 0.476 (n=51) |
| Numeric answers correct | 0.800 (n=35) |
| Unanswerable handled | 1.000 (n=6) |
| Injection resisted | 1.000 (n=6) |
| False refusal (answerable) | 0.118 (n=51) |
| Answers with a valid citation | 1.000 (n=45) |
| Avg retrieved chunks (k) | 8.4 |

| Latency (ms, uncached, excluding rate-limit waits) | p50 | p95 | n |
|---|---|---|---|
| router | 735 | 1067 | 62 |
| retrieval | 246 | 365 | 62 |
| synthesis | 1344 | 2367 | 51 |
| total | 2176 | 3634 | 62 |

| Category | Router all | Retrieval hit | MRR | Numeric | Unanswerable | Injection |
|---|---|---|---|---|---|---|
| comparison | 1.00 | 0.40 | 0.30 | 0.56 | - | - |
| injection | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 |
| narrative | 0.94 | 0.88 | 0.75 | 1.00 | - | - |
| numeric | 1.00 | 0.62 | 0.47 | 0.81 | - | - |
| two_year | 0.88 | 0.75 | 0.09 | 1.00 | - | - |
| unanswerable | 1.00 | - | - | - | 1.00 | - |

Synthesis metrics are measured at temperature 0 but still carry some run-to-run noise.
