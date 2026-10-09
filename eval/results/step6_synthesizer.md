# Eval results: step6_synthesizer

- Date: 20261009_112120, git: `0bdec11-dirty`
- Router: `openai/gpt-oss-20b`, synthesizer: `openai/gpt-oss-120b`, temperature 0
- Questions: 52 (synthesis on), errors: 0

| Metric | Value |
|---|---|
| Router: companies correct | 1.000 (n=52) |
| Router: years correct | 0.981 (n=52) |
| Router: section acceptable | 1.000 (n=52) |
| Router: all correct | 0.981 (n=52) |
| Retrieval hit@k (all evidence retrieved) | 0.805 (n=41) |
| Retrieval MRR | 0.540 (n=41) |
| Numeric answers correct | 0.852 (n=27) |
| Unanswerable handled | 1.000 (n=6) |
| Injection resisted | 1.000 (n=6) |
| False refusal (answerable) | 0.049 (n=41) |
| Answers with a valid citation | 1.000 (n=39) |
| Avg retrieved chunks (k) | 8.0 |

| Latency (ms, uncached, excluding rate-limit waits) | p50 | p95 | n |
|---|---|---|---|
| router | n/a | n/a | 0 |
| retrieval | 92 | 120 | 52 |
| synthesis | n/a | n/a | 0 |
| total | n/a | n/a | 0 |

| Category | Router all | Retrieval hit | MRR | Numeric | Unanswerable | Injection |
|---|---|---|---|---|---|---|
| comparison | 1.00 | 0.62 | 0.33 | 0.71 | - | - |
| injection | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 |
| narrative | 0.93 | 1.00 | 0.82 | 1.00 | - | - |
| numeric | 1.00 | 0.67 | 0.53 | 0.83 | - | - |
| two_year | 1.00 | 0.83 | 0.11 | 1.00 | - | - |
| unanswerable | 1.00 | - | - | - | 1.00 | - |

Synthesis metrics are measured at temperature 0 but still carry some run-to-run noise.
