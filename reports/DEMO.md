# Demo script (5 questions, about 6 minutes)

Run the app with `.venv/bin/streamlit run app.py --server.address localhost` and open http://localhost:8501.
All five questions were run live on 19 Nov 2026; the timings and behaviour below are from that run. Each answer takes about 5–11 s.
**Before the demo, warn your group not to use the API key during it:** one request at a time per key.

| # | Question (paste into **Ask**) | Retrieval | What to point out |
|---|---|---|---|
| 1 | How does Self-RAG use reflection tokens to decide when to retrieve and to critique its own generations? | B3 (default) | **Single-paper lookup.** 3 claims, each cited to Self-RAG §1 with page numbers. Open an evidence card to show the exact passage, then click the arXiv link. About 5 s. |
| 2 | How do RAGAS and ARES differ in how they evaluate RAG systems? | B3 | **Cross-paper comparison.** It cites ARES itself plus two other papers that discuss both. The answer adds a limitation: different metrics and datasets. |
| 3 | Do long-context LLMs make retrieval-augmented generation unnecessary? | B3 | **Disagreement question.** It gives a nuanced "no, depends on the setting" with 7 claims from 3 papers (an LC-vs-RAG study, LongRAG, a survey). **Honest caveat to mention:** it labels the comparison "directly comparable", which is too generous because the claims come from different studies. That's a known weakness (report §5). |
| 4 | How does HippoRAG 2 build on the original HippoRAG? | **B4** (switch the dropdown) | **Citation graph.** The caption shows "graph added …". Then open **Citation neighbours** for the HippoRAG paper and point at the banner: arrows mean *cites*, not agreement. Mention that the graph didn't measurably improve retrieval (report §4.1), so it's off by default. |
| 5 | Which RAG method has the lowest carbon footprint per query? | B3 | **Abstention.** "Not established by the indexed papers", plus what is missing. The related claims it lists (latency, cost) are kept separate from the verdict. |

**Close with the Evaluation tab:** the B0–B5 table, QASPER (the reranker improves Hit@1 0.41 → 0.45), and answers (4/6 unanswerable questions declined, 100% valid citation IDs).

**Backup if the API is down (429/503):** show `reports/RESULTS.md` and an earlier answer from the `query_runs` table:
`sqlite3 data/researchgraph.db "select query, latency_ms from query_runs order by timestamp desc limit 10"`
