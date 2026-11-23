# Demo script (5 questions, about 6 minutes)

**Recorded version:** `reports/demo/researchgraph_demo.mp4` (narrated, 5 min 36 s, neural voice "Andrew"; regenerate with `scripts/record_demo.py`). It shows Q1, a clicked follow-up suggestion, agent mode on a comparison (tool calls and the critic's retry), a vague follow-up rewritten from session memory ("the second one" → ARES), the OpenTelemetry trace, Q5 declined, the citation neighbours and the stage-by-stage evaluation. The narration text is in `reports/demo/researchgraph_demo_narration.txt`.

**New since the mentor review:** after every answer, the **You might ask next** section shows 3 follow-up questions. Click one to ask it straight away; the question trail shows the chain.

Run the app with `.venv/bin/streamlit run app.py --server.address localhost` and open http://localhost:8501.
All five questions were re-run on 22 Nov 2026 with the final version (prompt v3, per-entity retrieval); the behaviour below is from that run. Each answer takes about 7–12 s.
**Before the demo, warn your group not to use the API key during it:** one request at a time per key.

| # | Question (paste into **Ask**) | Retrieval | What to point out |
|---|---|---|---|
| 1 | How does Self-RAG use reflection tokens to decide when to retrieve and to critique its own generations? | B3 (default) | **Single-paper lookup.** 3 claims, each cited to Self-RAG §1 with page numbers. Open an evidence card to show the exact passage, then click the arXiv link. About 5 s. |
| 2 | How do RAGAS and ARES differ in how they evaluate RAG systems? | B3 | **Cross-paper comparison.** The caption shows it searched separately for RAGAS and ARES, so both papers' abstracts are cited. It calls the ARES-vs-RAGAS ranking result directly comparable because ARES's own paper reports both; the code check allows that only when one paper reports both results. |
| 3 | Do long-context LLMs make retrieval-augmented generation unnecessary? | B3 | **Disagreement question.** It gives a nuanced "no, depends on the setting" with 7 claims from 3 papers (an LC-vs-RAG study, LongRAG, a survey). Before the fixes it wrongly called this "directly comparable"; now it's flagged **not directly comparable** because the results come from different studies. That's a good before/after moment (report §4.3). |
| 4 | How does HippoRAG 2 build on the original HippoRAG? | **B4** (switch the dropdown) | **Citation graph.** The caption shows "graph added …". Then open **Citation neighbours** for the HippoRAG paper and point at the banner: arrows mean *cites*, not agreement. Mention that the graph didn't measurably improve retrieval (report §4.1), so it's off by default. |
| 5 | Which RAG method has the lowest carbon footprint per query? | B3 | **Abstention.** "Not established by the indexed papers", plus what is missing. The related claims it lists (latency, cost) are kept separate from the verdict. |

**Close with the Evaluation tab:** the B0–B5 table, QASPER (the reranker improves Hit@1 0.41 → 0.45), and answers (4/6 unanswerable declined, gold papers cited 0.75, 0 invalid citations).

**If an answer hangs over 30 s:** it's the server (one request stalled for minutes during testing). The client retries on its own; meanwhile show `reports/RESULTS.md`.

**Backup if the API is down (429/503):** show `reports/RESULTS.md` and an earlier answer from the `query_runs` table:
`sqlite3 data/researchgraph.db "select query, latency_ms from query_runs order by timestamp desc limit 10"`
