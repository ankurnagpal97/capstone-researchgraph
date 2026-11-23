# Chunking experiment (2026-11-23 23:07)

Same parsed PDFs, three chunk sizes, each with its own BM25 + bge-small index; retrieval only. '@budget' = the hits that fit the generator's 3,000-token evidence budget, which is the fair comparison (bigger chunks hold more text in a top 10). A gold passage counts as found if one retrieved chunk contains at least half of it. Mean [95% bootstrap CI].

| Variant | Target / overlap tokens | Chunks | Tokens p50 / p95 |
|---|---|---|---|
| small | 150 / 25 | 18,071 | 145 / 223 |
| base (production) | 300 / 50 | 11,148 | 252 / 414 |
| large | 450 / 75 | 8,406 | 348 / 519 |

**curated, B3 Hybrid+rerank**

| Variant | Passages in budget | Gold passage found @budget | Paper recall @budget | Reference ROUGE-L recall @budget | Gold passage found @10 |
|---|---|---|---|---|---|
| small | 10.19 [8.81, 11.50] | 0.72 [0.58, 0.86] | 0.88 [0.79, 0.95] | 0.54 [0.50, 0.58] | 0.72 [0.58, 0.86] |
| base | 8.08 [7.25, 8.83] | 0.86 [0.75, 0.97] | 0.83 [0.73, 0.91] | 0.58 [0.55, 0.61] | 0.89 [0.78, 0.97] |
| large | 6.89 [6.39, 7.31] | 0.81 [0.67, 0.92] | 0.80 [0.70, 0.90] | 0.59 [0.56, 0.63] | 0.83 [0.69, 0.94] |

**curated, B2 Hybrid**

| Variant | Passages in budget | Gold passage found @budget | Paper recall @budget | Reference ROUGE-L recall @budget | Gold passage found @10 |
|---|---|---|---|---|---|
| small | 13.56 [12.58, 14.28] | 0.44 [0.28, 0.61] | 0.89 [0.81, 0.96] | 0.56 [0.52, 0.60] | 0.44 [0.28, 0.61] |
| base | 9.61 [9.14, 10.08] | 0.64 [0.47, 0.78] | 0.84 [0.75, 0.93] | 0.58 [0.54, 0.61] | 0.64 [0.47, 0.78] |
| large | 7.69 [7.33, 8.14] | 0.67 [0.50, 0.83] | 0.85 [0.76, 0.93] | 0.59 [0.56, 0.63] | 0.67 [0.50, 0.83] |

**synthetic_test, B3 Hybrid+rerank**

| Variant | Passages in budget | Gold passage found @budget | Paper recall @budget | Reference ROUGE-L recall @budget | Gold passage found @10 |
|---|---|---|---|---|---|
| small | 11.67 [10.49, 12.82] | 0.53 [0.40, 0.67] | 1.00 [1.00, 1.00] | 0.78 [0.72, 0.83] | 0.53 [0.40, 0.67] |
| base | 8.60 [7.98, 9.18] | 0.87 [0.76, 0.96] | 1.00 [1.00, 1.00] | 0.81 [0.76, 0.85] | 0.87 [0.76, 0.96] |
| large | 6.91 [6.58, 7.22] | 0.84 [0.73, 0.93] | 1.00 [1.00, 1.00] | 0.81 [0.76, 0.85] | 0.84 [0.73, 0.93] |

**synthetic_test, B2 Hybrid**

| Variant | Passages in budget | Gold passage found @budget | Paper recall @budget | Reference ROUGE-L recall @budget | Gold passage found @10 |
|---|---|---|---|---|---|
| small | 13.89 [13.36, 14.33] | 0.51 [0.36, 0.64] | 1.00 [1.00, 1.00] | 0.78 [0.73, 0.83] | 0.51 [0.36, 0.64] |
| base | 9.33 [9.00, 9.69] | 0.82 [0.71, 0.93] | 0.98 [0.93, 1.00] | 0.80 [0.76, 0.85] | 0.84 [0.73, 0.93] |
| large | 7.38 [7.13, 7.64] | 0.78 [0.64, 0.89] | 0.98 [0.93, 1.00] | 0.78 [0.73, 0.83] | 0.78 [0.64, 0.89] |
