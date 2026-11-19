# AI ResearchGraph — evaluation results

Run 2026-11-14 17:40 · corpus 146 papers, 11148 chunks · values are means with 95% bootstrap confidence intervals over questions.

**Read with care.** Curated gold papers were drafted from paper titles and still need team review. Synthetic questions were written by Qwen3-8B from one known passage each, so they echo that passage's wording and are easier than real questions. With n under 50 the intervals are wide: differences whose intervals overlap are not established.

## Retrieval — curated (n = 36)

| System | Paper Recall@5 | Paper Recall@10 | Paper MRR | Passage Hit@10 | Passage MRR |
|---|---|---|---|---|---|
| B0 BM25 | 0.74 [0.62, 0.85] | 0.81 [0.71, 0.91] | 0.79 [0.68, 0.89] | – | – |
| B1 Dense | 0.79 [0.68, 0.88] | 0.82 [0.72, 0.92] | 0.84 [0.74, 0.94] | – | – |
| B2 Hybrid | 0.80 [0.69, 0.89] | 0.86 [0.77, 0.94] | 0.89 [0.81, 0.97] | – | – |
| B3 Hybrid+rerank | 0.75 [0.64, 0.85] | 0.85 [0.76, 0.93] | 0.83 [0.72, 0.92] | – | – |
| B4 +1-hop graph | 0.75 [0.64, 0.85] | 0.86 [0.77, 0.94] | 0.83 [0.72, 0.92] | – | – |
| B5 +PageRank graph | 0.75 [0.64, 0.85] | 0.86 [0.77, 0.94] | 0.83 [0.72, 0.92] | – | – |

Graph forced on for the 19 multi-hop, comparison and disagreement questions:

| System | Paper Recall@10 | Paper MRR |
|---|---|---|
| B3 no graph | 0.71 [0.57, 0.83] | 0.73 [0.57, 0.89] |
| B4 1-hop forced | 0.72 [0.58, 0.85] | 0.73 [0.57, 0.89] |
| B5 PPR forced | 0.72 [0.58, 0.85] | 0.73 [0.57, 0.89] |

Paper Recall@10 by question type:

| Type | n | B0 BM25 | B1 Dense | B2 Hybrid | B3 Hybrid+rerank | B4 +1-hop graph | B5 +PageRank graph |
|---|---|---|---|---|---|---|---|
| comparison | 9 | 0.74 | 0.70 | 0.83 | 0.78 | 0.78 | 0.78 |
| disagreement | 3 | 0.22 | 0.44 | 0.33 | 0.33 | 0.33 | 0.33 |
| metadata | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| multihop | 7 | 0.70 | 0.70 | 0.79 | 0.79 | 0.83 | 0.83 |
| single_paper | 14 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Retrieval — synthetic_dev (n = 105)

| System | Paper Recall@5 | Paper Recall@10 | Paper MRR | Passage Hit@10 | Passage MRR |
|---|---|---|---|---|---|
| B0 BM25 | 0.95 [0.91, 0.99] | 0.96 [0.92, 0.99] | 0.91 [0.86, 0.96] | 0.84 [0.76, 0.91] | 0.69 [0.62, 0.77] |
| B1 Dense | 0.91 [0.85, 0.96] | 0.93 [0.88, 0.97] | 0.85 [0.79, 0.91] | 0.61 [0.51, 0.70] | 0.43 [0.35, 0.52] |
| B2 Hybrid | 0.96 [0.92, 0.99] | 0.98 [0.95, 1.00] | 0.91 [0.86, 0.96] | 0.75 [0.67, 0.83] | 0.60 [0.52, 0.68] |
| B3 Hybrid+rerank | 0.95 [0.91, 0.99] | 0.98 [0.95, 1.00] | 0.92 [0.87, 0.96] | 0.81 [0.73, 0.89] | 0.69 [0.61, 0.77] |
| B4 +1-hop graph | 0.95 [0.91, 0.99] | 0.98 [0.95, 1.00] | 0.92 [0.87, 0.96] | 0.81 [0.73, 0.89] | 0.69 [0.61, 0.77] |
| B5 +PageRank graph | 0.95 [0.91, 0.99] | 0.98 [0.95, 1.00] | 0.92 [0.87, 0.96] | 0.81 [0.73, 0.89] | 0.69 [0.61, 0.77] |

## Retrieval — synthetic_test (n = 45)

| System | Paper Recall@5 | Paper Recall@10 | Paper MRR | Passage Hit@10 | Passage MRR |
|---|---|---|---|---|---|
| B0 BM25 | 0.93 [0.84, 1.00] | 0.98 [0.93, 1.00] | 0.87 [0.78, 0.95] | 0.80 [0.67, 0.91] | 0.67 [0.55, 0.79] |
| B1 Dense | 0.93 [0.84, 1.00] | 0.96 [0.89, 1.00] | 0.83 [0.73, 0.92] | 0.69 [0.56, 0.82] | 0.53 [0.41, 0.66] |
| B2 Hybrid | 0.98 [0.93, 1.00] | 1.00 [1.00, 1.00] | 0.92 [0.85, 0.98] | 0.84 [0.73, 0.93] | 0.65 [0.53, 0.76] |
| B3 Hybrid+rerank | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.91 [0.83, 0.97] | 0.87 [0.76, 0.96] | 0.73 [0.61, 0.84] |
| B4 +1-hop graph | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.91 [0.83, 0.97] | 0.87 [0.76, 0.96] | 0.73 [0.61, 0.84] |
| B5 +PageRank graph | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.91 [0.83, 0.97] | 0.87 [0.76, 0.96] | 0.73 [0.61, 0.84] |

## External benchmark — QASPER v0.3 test (CC BY 4.0)

194 questions over 60 papers (seed 7). Each question's own paper is searched, and success means ranking an annotator-marked evidence paragraph near the top. This measures within-paper evidence retrieval on NLP papers, not cross-paper comparison or the citation graph. Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

| System | Hit@1 | Hit@3 | Hit@5 | Evidence Recall@5 | MRR |
|---|---|---|---|---|---|
| B0 BM25 | 0.309 | 0.531 | 0.670 | 0.465 | 0.468 |
| B1 Dense | 0.366 | 0.639 | 0.768 | 0.537 | 0.538 |
| B2 Hybrid | 0.412 | 0.675 | 0.789 | 0.564 | 0.571 |
| B3 Hybrid+rerank | 0.454 | 0.747 | 0.845 | 0.599 | 0.622 |

## End-to-end answers — curated set, B4 +1-hop graph (n = 42)

| Measure | Value |
|---|---|
| Declined unanswerable questions | 2/6 |
| Answered answerable questions | 36/36 |
| Invalid JSON replies | 0 |
| Citation IDs that exist in the evidence | 1.0 |
| Claims without a valid citation | 0/110 |
| Gold papers among cited passages | 0.655 |
| Auto-judged: supported / partial / not supported | 0.955 / 0.036 / 0.009 |
| Latency p50 / p95 (s) | 6.35 / 10.57 |
| Mean tokens per answer | 2262 |

*auto_support_* are Qwen3-8B judging its own answers: an estimate. Human labels in reports/citation_audit_sample.csv are the real measure.*

Human citation-support labels: fill in `reports/citation_audit_sample.csv` (column `human_label`), then report the share labelled supported.

### Answer failures to analyse

| Question | Type | Expected answerable | Model said | Note |
|---|---|---|---|---|
| q15 | unanswerable | False | True | The best RAG method for medical diagnosis in hospitals is not explicitly stated in the evidence, but the Best Performanc |
| q40 | unanswerable | False | True | The evidence does not directly state which RAG method has the lowest carbon footprint per query. However, it mentions th |
| q41 | unanswerable | False | True | The evidence suggests that OpenAI uses dynamic retrieval calls via system prompts for ChatGPT's browsing feature, as ind |
| q42 | unanswerable | False | True | The exact recall@10 of RAPTOR on the Natural Questions test set with a 70B reader is not directly provided in the eviden |
