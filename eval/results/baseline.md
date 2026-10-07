# Eval results: baseline

- Date: 20261007_172705, git: `acf0a2e-dirty`
- Router: `openai/gpt-oss-20b`, synthesizer: `openai/gpt-oss-120b`, temperature 0
- Questions: 52 (synthesis on), errors: 0

| Metric | Value |
|---|---|
| Router: companies correct | 1.000 (n=52) |
| Router: years correct | 0.846 (n=52) |
| Router: section acceptable | 0.808 (n=52) |
| Router: all correct | 0.673 (n=52) |
| Retrieval hit@k (all evidence retrieved) | 0.585 (n=41) |
| Retrieval MRR | 0.406 (n=41) |
| Numeric answers correct | 0.704 (n=27) |
| Unanswerable handled | 1.000 (n=6) |
| Injection resisted | 1.000 (n=6) |
| False refusal (answerable) | 0.220 (n=41) |
| Avg retrieved chunks (k) | 5.8 |

| Latency (ms, uncached, excluding rate-limit waits) | p50 | p95 | n |
|---|---|---|---|
| router | 396 | 771 | 52 |
| retrieval | 177 | 225 | 52 |
| synthesis | 901 | 2715 | 52 |
| total | 1542 | 3711 | 52 |

| Category | Router all | Retrieval hit | MRR | Numeric | Unanswerable | Injection |
|---|---|---|---|---|---|---|
| comparison | 0.88 | 0.75 | 0.36 | 0.71 | - | - |
| injection | 1.00 | 1.00 | 0.50 | 1.00 | - | 1.00 |
| narrative | 0.71 | 0.36 | 0.23 | 0.00 | - | - |
| numeric | 0.67 | 0.67 | 0.58 | 0.75 | - | - |
| two_year | 0.00 | 0.67 | 0.50 | 0.67 | - | - |
| unanswerable | 0.67 | - | - | - | 1.00 | - |

Synthesis metrics are measured at temperature 0 but still carry some run-to-run noise.
