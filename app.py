"""AI ResearchGraph — Streamlit app.

Run: .venv/bin/streamlit run app.py
"""
import json
import os

import pandas as pd
import streamlit as st

from researchgraph import ROOT, answer, db
from researchgraph.llm import ApiError
from researchgraph.retrieve import Retriever

st.set_page_config(page_title="AI ResearchGraph", layout="wide")

MODE_LABELS = {
    "rerank": "B3 · Hybrid + cross-encoder rerank (default)",
    "graph1hop": "B4 · B3 + 1-hop citation graph (multi-hop questions)",
    "graph_ppr": "B5 · B3 + PageRank citation graph (multi-hop questions)",
    "hybrid": "B2 · Hybrid (BM25 + embeddings)",
    "dense": "B1 · Embeddings only",
    "bm25": "B0 · BM25 only",
}
EXAMPLES = [
    "How does Self-RAG decide when to retrieve, and how does it critique its own output?",
    "Which methods build on or compare against IRCoT for multi-hop question answering?",
    "Does the position of relevant information in a long context affect answer accuracy?",
    "How do RAGAS and ARES differ in how they evaluate RAG systems?",
    "What is the best RAG method for medical diagnosis in hospitals?",
]


@st.cache_resource(show_spinner="Loading indexes…")
def retriever():
    return Retriever()


def edges_df():
    con = db.connect()
    return pd.read_sql_query("""SELECT e.citing_paper_id, a.title AS citing_title, e.cited_paper_id, b.title AS cited_title,
                                       GROUP_CONCAT(e.edge_source) AS sources
                                FROM citation_edges e JOIN papers a ON a.paper_id = e.citing_paper_id
                                JOIN papers b ON b.paper_id = e.cited_paper_id
                                GROUP BY 1, 3""", con)


st.title("AI ResearchGraph")
st.caption("Evidence-grounded answers about RAG and LLM-retrieval research (2022–2026). "
           "Every claim links to the passage it comes from. Answers come from Qwen3-8B and only from the indexed papers.")

tab_ask, tab_graph, tab_corpus, tab_eval = st.tabs(["Ask", "Citation neighbours", "Corpus & provenance", "Evaluation"])

def _set_example():
    if st.session_state.get("example"):
        st.session_state["q"] = st.session_state["example"]


def _ask_followup(question: str):
    st.session_state["q"] = question
    st.session_state["autorun"] = True


with tab_ask:
    c1, c2 = st.columns([3, 1])
    with c2:
        mode = st.selectbox("Retrieval", list(MODE_LABELS), format_func=MODE_LABELS.get)
        k = st.slider("Passages retrieved", 5, 15, 10)
        st.selectbox("Examples", [""] + EXAMPLES, key="example", on_change=_set_example)
        want_followups = st.toggle("Suggest follow-up questions", value=True,
                                   help="One extra Qwen3-8B call after each answer (about 2-4 s).")
    with c1:
        st.text_area("Question", key="q", height=90,
                     placeholder="e.g. Which methods decide adaptively when to retrieve?")
        q = st.session_state.get("q", "")
        go = st.button("Ask", type="primary", disabled=not q.strip())
    if st.session_state.pop("autorun", False) and q.strip():
        go = True
    if go:
        try:
            with st.spinner("Retrieving…"):
                res = retriever().search_decomposed(q.strip(), mode=mode, k=k)
            with st.spinner("Asking Qwen3-8B (one request at a time per group key)…"):
                ans = answer.ask(q.strip(), res)
            if want_followups:
                with st.spinner("Suggesting follow-up questions…"):
                    answer.suggest_followups(ans)
            st.session_state["last"] = ans
            trail = st.session_state.setdefault("trail", [])
            if not trail or trail[-1] != q.strip():
                trail.append(q.strip())
        except ApiError as e:
            msg = {429: "The group key is busy with another request. Wait a few seconds and try again.",
                   503: "The model server is busy or restarting. Try again shortly."}.get(e.status, str(e))
            st.error(msg)
    ans = st.session_state.get("last")
    if ans:
        if len(st.session_state.get("trail", [])) > 1:
            st.caption("Question trail: " + "  →  ".join(st.session_state["trail"][-4:]))
        st.markdown(answer.render_markdown(ans))
        cv = ans.get("citation_validity")
        st.caption(f"{ans['latency_s']}s · {ans['usage'].get('total_tokens', '?')} tokens · mode {ans['mode']}"
                   + (f" · graph added {', '.join(ans['graph_papers'])}" if ans["graph_used"] else "")
                   + (f" · searched separately for {', '.join(ans['entities'])}" if ans.get("entities") else "")
                   + (f" · citation IDs valid {cv:.0%}" if cv is not None else ""))
        if ans.get("followups"):
            st.subheader("You might ask next")
            for i, f in enumerate(ans["followups"]):
                st.button(f"→ {f}", key=f"fu{i}", on_click=_ask_followup, args=(f,), use_container_width=True)
            st.caption(f"Suggested by Qwen3-8B from this answer and its source papers ({ans.get('followup_latency_s', '?')}s). "
                       "Click one to ask it; new suggestions follow each answer.")
        st.subheader("Evidence given to the model")
        for e in ans["evidence"]:
            pages = f"p.{e['page_start']}" if e["page_start"] == e["page_end"] else f"pp.{e['page_start']}–{e['page_end']}"
            with st.expander(f"{e['label']} · {e['title']} ({e['year']}) · {e['section']} · {pages}"
                             + (" · via citation graph" if e["via_graph"] else "")):
                st.write(e["text"])
                st.caption(f"`{e['chunk_id']}` · [{e['url']}]({e['url']})")

with tab_graph:
    st.info("Arrows mean **cites**. A citation does not mean the citing paper agrees with, confirms or builds on the cited one.")
    df = edges_df()
    con = db.connect()
    papers = {r["paper_id"]: r["title"] for r in con.execute("SELECT paper_id, title FROM papers WHERE pdf_path IS NOT NULL ORDER BY title")}
    default = (st.session_state.get("last") or {}).get("evidence", [{}])[0].get("paper_id")
    keys = list(papers)
    pid = st.selectbox("Paper", keys, index=keys.index(default) if default in keys else 0,
                       format_func=lambda p: f"{papers[p]} ({p})")
    sub = df[(df.citing_paper_id == pid) | (df.cited_paper_id == pid)]
    short = lambda t: (t[:38] + "…") if len(t) > 40 else t
    dot = ["digraph G {", "rankdir=LR; node [shape=box, style=rounded, fontsize=10];",
           f'"{pid}" [label="{short(papers[pid])}", style="rounded,bold"];']
    for _, r in sub.iterrows():
        for p in (r.citing_paper_id, r.cited_paper_id):
            dot.append(f'"{p}" [label="{short(papers.get(p, p))}"];')
        dot.append(f'"{r.citing_paper_id}" -> "{r.cited_paper_id}" [tooltip="{r.sources}"];')
    dot.append("}")
    if len(sub):
        st.graphviz_chart("\n".join(dot))
    st.dataframe(sub.rename(columns={"sources": "found by"}), hide_index=True, use_container_width=True)
    st.caption(f"{len(df)} citation edges inside the corpus. 'openalex' = OpenAlex reference list; "
               "'pdf_reference_match' = the cited paper's title or arXiv ID appears in the citing paper's reference list.")

with tab_corpus:
    con = db.connect()
    p = pd.read_sql_query("""SELECT p.paper_id, p.title, p.year, p.scope_tag, p.rights_status, p.license, p.n_pages,
                                    (SELECT COUNT(*) FROM chunks c WHERE c.paper_id = p.paper_id) AS chunks, p.parse_status, p.url
                             FROM papers p ORDER BY p.year DESC""", con)
    a, b, c = st.columns(3)
    a.metric("Papers indexed", int((p.chunks > 0).sum()))
    b.metric("Chunks", int(p.chunks.sum()))
    c.metric("Citation edges", len(edges_df()))
    st.dataframe(p, hide_index=True, use_container_width=True,
                 column_config={"url": st.column_config.LinkColumn("arXiv")})
    d = pd.read_sql_query("SELECT * FROM decisions WHERE decision != 'accept'", con)
    if len(d):
        st.subheader("Rejected or quarantined seeds")
        st.dataframe(d, hide_index=True, use_container_width=True)
    st.caption("Full texts are stored locally only and are not redistributed. 'local_index_only' = arXiv's default "
               "licence, which lets arXiv distribute the paper but gives no reuse rights to us.")

with tab_eval:
    stages = os.path.join(ROOT, "reports", "STAGES.md")
    if os.path.exists(stages):
        with st.expander("Stage-by-stage accuracy: chunking → retrieval → rerank → evidence → answer (ROUGE) → "
                         "citations → follow-ups", expanded=True):
            st.markdown(open(stages).read().split("\n", 1)[1], unsafe_allow_html=True)  # our own report file
    path = os.path.join(ROOT, "reports", "eval_results.json")
    if not os.path.exists(path):
        st.info("No evaluation yet. Run `.venv/bin/python scripts/evaluate.py --answers`.")
    else:
        r = json.load(open(path))
        st.caption(f"Run {r['run_at']} · {r['n_papers']} papers · {r['n_chunks']} chunks. "
                   "Curated gold labels are drafts until the team reviews them; synthetic questions are easier than real ones.")
        for name, systems in r["retrieval"].items():
            st.subheader(f"Retrieval: {name}")
            rows = []
            for label, m in systems.items():
                row = {"system": label, "n": m["paper_r10"]["n"],
                       "paper R@5": m["paper_r5"]["mean"], "paper R@10": m["paper_r10"]["mean"],
                       "R@10 95% CI": str(m["paper_r10"]["ci95"]), "paper MRR": m["paper_mrr"]["mean"]}
                if "passage_hit10" in m:
                    row.update({"passage Hit@10": m["passage_hit10"]["mean"], "passage MRR": m["passage_mrr"]["mean"]})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            g = r.get("graph_ablation", {}).get(name)
            if g:
                st.caption(f"Graph forced on, {g['n']} multi-hop/comparison/disagreement questions")
                st.dataframe(pd.DataFrame([{"system": k, "paper R@10": v["paper_r10"]["mean"], "paper MRR": v["paper_mrr"]["mean"]}
                                           for k, v in g["results"].items()]), hide_index=True)
        qp = os.path.join(ROOT, "reports", "qasper_results.json")
        if os.path.exists(qp):
            q = json.load(open(qp))
            st.subheader(f"External: {q['dataset']} — {q['n_questions']} questions, {q['n_papers']} papers")
            st.dataframe(pd.DataFrame(q["summary"]), hide_index=True, use_container_width=True)
        if r.get("answers"):
            a = r["answers"]
            st.subheader(f"Answers: {a['set']}, {a['system']}, n = {a['n']}")
            st.json({k: v for k, v in a.items() if k != "per_question"})
            st.dataframe(pd.DataFrame(a["per_question"]), hide_index=True, use_container_width=True)
