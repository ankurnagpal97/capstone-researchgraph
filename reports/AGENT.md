# Agent vs pipeline (curated set, 2026-11-23 23:20)

Agent `agent-v1`: planner → executor (tools) → generator → critic, at most 2 rounds, 8 tool calls and 20,000 tokens per task. Pipeline: B3 hybrid + rerank + per-entity retrieval, one answer call. Both answer with Qwen3-8B (prompt answer-v3) and pass the same code checks. Mean [95% bootstrap CI].

| Metric | Pipeline (B3) | Agent |
|---|---|---|
| End-to-end accuracy (decision right + a gold paper cited) | 0.86 [0.74, 0.95] | 0.88 [0.76, 0.98] |
| Task completion rate | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Tool calls per task | 1.00 [1.00, 1.00] | 2.40 [2.07, 2.74] |
| LLM calls per task | 1.00 [1.00, 1.00] | 3.62 [3.48, 3.76] |
| Tokens per query (cost) | 2522.38 [2382.29, 2662.33] | 5073.55 [4564.40, 5571.52] |
| Latency p50 (s) | 8.0 | 24.8 |
| Latency p95 (s) | 12.2 | 38.4 |
| Gold paper in evidence pack | 0.84 [0.74, 0.92] | 0.80 [0.69, 0.90] |
| Gold passage in evidence pack | 0.86 [0.75, 0.97] | 0.61 [0.44, 0.78] |
| ROUGE-L F1 vs reference | 0.25 [0.23, 0.27] | 0.23 [0.21, 0.26] |
| Token F1 vs reference | 0.30 [0.28, 0.33] | 0.29 [0.27, 0.32] |
| Semantic similarity vs reference | 0.89 [0.88, 0.90] | 0.88 [0.86, 0.90] |
| Citation IDs valid | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Claims supported (Qwen self-judge) | 0.929 | 0.888 |
| Abstained on unanswerable | 4/6 | 5/6 |
| Answered when answerable | 35/36 | 34/36 |

**Accuracy by question type**

| Type | Pipeline | Agent |
|---|---|---|
| comparison | 9/9 | 9/9 |
| disagreement | 2/3 | 2/3 |
| metadata | 2/3 | 2/3 |
| multihop | 5/7 | 6/7 |
| single_paper | 14/14 | 13/14 |
| unanswerable | 4/6 | 5/6 |

**Failure taxonomy** (questions not end-to-end correct, first matching cause)

| Cause | Pipeline | Agent |
|---|---|---|
| tool-call error | 0  | 0  |
| retrieval miss | 1 q36 | 2 q04, q36 |
| hallucination | 2 q15, q41 | 2 q33, q41 |
| reasoning error | 3 q32, q33, q37 | 1 q14 |

**Agent behaviour:** retry rate 0.619, stop reasons {'accepted': 16, 'max rounds': 26}, tools used {'search_papers': 98, 'citation_neighbours': 3}, tool errors 0, malformed JSON 0 (all repaired or fell back), fallbacks 0.

**Cost:** the course endpoint has no per-token price; cost is reported as tokens and LLM calls per query. Every LLM call holds the group's single request slot, so latency is also the cost to teammates.

Curated gold labels and reference answers are drafts until the team review is imported; n = 42, so differences under about 0.10 are not established.
