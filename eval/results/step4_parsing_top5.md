# Eval results: step4_parsing_top5 (parser changes, top_k 5 per combination)

- Date: 20261008_120847, git: `f67273b-dirty`
- Router: `openai/gpt-oss-20b`, synthesizer: `openai/gpt-oss-120b`, temperature 0
- Questions: 52 (synthesis on), errors: 0

| Metric | Value |
|---|---|
| Router: companies correct | 1.000 (n=52) |
| Router: years correct | 0.981 (n=52) |
| Router: section acceptable | 1.000 (n=52) |
| Router: all correct | 0.981 (n=52) |
| Retrieval hit@k (all evidence retrieved) | 0.707 (n=41) |
| Retrieval MRR | 0.537 (n=41) |
| Numeric answers correct | 0.778 (n=27) |
| Unanswerable handled | 1.000 (n=6) |
| Injection resisted | 1.000 (n=6) |
| False refusal (answerable) | 0.122 (n=41) |
| Avg retrieved chunks (k) | 5.3 |

| Latency (ms, uncached, excluding rate-limit waits) | p50 | p95 | n |
|---|---|---|---|
| router | 492 | 871 | 52 |
| retrieval | 117 | 209 | 52 |
| synthesis | 963 | 2475 | 41 |
| total | 1431 | 3084 | 52 |

| Category | Router all | Retrieval hit | MRR | Numeric | Unanswerable | Injection |
|---|---|---|---|---|---|---|
| comparison | 1.00 | 0.38 | 0.34 | 0.43 | - | - |
| injection | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 |
| narrative | 0.93 | 0.93 | 0.81 | 1.00 | - | - |
| numeric | 1.00 | 0.58 | 0.52 | 0.83 | - | - |
| two_year | 1.00 | 0.83 | 0.12 | 1.00 | - | - |
| unanswerable | 1.00 | - | - | - | 1.00 | - |

Synthesis metrics are measured at temperature 0 but still carry some run-to-run noise.
