# Agent vs pipeline (curated set, 2026-11-24 00:05)

Agent `agent-v1`: planner → executor (tools) → generator → critic, at most 2 rounds, 8 tool calls and 20,000 tokens per task. Pipeline: B3 hybrid + rerank + per-entity retrieval, one answer call. Both answer with Qwen3-8B (prompt answer-v3) and pass the same code checks. Mean [95% bootstrap CI].

| Metric | Pipeline (B3) | Agent |
|---|---|---|
| End-to-end accuracy (decision right + a gold paper cited) | 0.90 [0.82, 0.97] | 0.92 [0.85, 0.98] |
| Task completion rate | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Tool calls per task | 1.00 [1.00, 1.00] | 2.15 [1.87, 2.42] |
| LLM calls per task | 1.00 [1.00, 1.00] | 3.48 [3.37, 3.61] |
| Tokens per query (cost) | 2488.60 [2360.27, 2611.31] | 4522.31 [4109.32, 4971.95] |
| Latency p50 (s) | 7.5 | 15.7 |
| Latency p95 (s) | 12.0 | 36.8 |
| Gold paper in evidence pack | 0.88 [0.81, 0.94] | 0.86 [0.77, 0.93] |
| Gold passage in evidence pack | 0.77 [0.65, 0.89] | 0.64 [0.50, 0.77] |
| ROUGE-L F1 vs reference | 0.26 [0.24, 0.28] | 0.26 [0.23, 0.28] |
| Token F1 vs reference | 0.31 [0.29, 0.34] | 0.31 [0.29, 0.33] |
| Semantic similarity vs reference | 0.89 [0.87, 0.90] | 0.88 [0.86, 0.90] |
| Citation IDs valid | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Claims supported (Qwen self-judge) | 0.911 | 0.91 |
| Abstained on unanswerable | 7/9 | 8/9 |
| Answered when answerable | 52/53 | 51/53 |

**Accuracy by question type**

| Type | Pipeline | Agent |
|---|---|---|
| comparison | 13/13 | 13/13 |
| disagreement | 2/3 | 2/3 |
| metadata | 4/5 | 4/5 |
| multihop | 8/10 | 9/10 |
| single_paper | 22/22 | 21/22 |
| unanswerable | 7/9 | 8/9 |

**Failure taxonomy** (questions not end-to-end correct, first matching cause)

| Cause | Pipeline | Agent |
|---|---|---|
| tool-call error | 0  | 0  |
| retrieval miss | 1 q36 | 2 q04, q36 |
| hallucination | 2 q15, q41 | 2 q33, q41 |
| reasoning error | 3 q32, q33, q37 | 1 q14 |

**Agent behaviour:** retry rate 0.484, stop reasons {'accepted': 32, 'max rounds': 30}, tools used {'search_papers': 129, 'citation_neighbours': 3, 'paper_info': 1}, tool errors 0, malformed JSON 0 (all repaired or fell back), fallbacks 0.

**Cost:** the course endpoint has no per-token price; cost is reported as tokens and LLM calls per query. Every LLM call holds the group's single request slot, so latency is also the cost to teammates.

Curated reference answers were drafted by Claude from the gold passages; n = 62, so differences under about 0.10 are not established.
