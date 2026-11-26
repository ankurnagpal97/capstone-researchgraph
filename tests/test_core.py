"""Offline unit tests (no API calls). Run: .venv/bin/python -m pytest -q"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import ROOT, answer, index, parse  # noqa: E402
from researchgraph.retrieve import Hit, Result, Retriever  # noqa: E402


# -- parsing and chunking --------------------------------------------------
def test_heading_detection():
    yes = ["1 Introduction", "3.2 Retrieval Module", "Abstract", "References", "A Implementation Details",
           "5 Experiments", "Related Work"]
    no = ["1 We show that retrieval helps a lot in many settings.", "2 4 6 8 10",
          "Table 3: Results on HotpotQA with different retrievers and readers.", "This is a sentence.",
          "17 Oct 2023", "A SELF-RAG Details 17"]
    assert all(parse._is_heading(t) for t in yes), [t for t in yes if not parse._is_heading(t)]
    assert not any(parse._is_heading(t) for t in no), [t for t in no if parse._is_heading(t)]


def test_sections_split_references_out():
    B = parse.Block
    blocks = [B("Abstract", 1), B("We propose X. " * 5, 1), B("1 Introduction", 1), B("Intro text. " * 30, 1),
              B("References", 9), B("[1] Some Author. A Paper Title About Retrieval. 2023.", 9),
              B("A Appendix Details", 10), B("Appendix text. " * 20, 10)]
    sections, refs = parse.split_sections(blocks)
    names = [s.heading for s in sections]
    assert names == ["Abstract", "1 Introduction", "A Appendix Details"]
    assert "Paper Title About Retrieval" in refs
    assert all("Paper Title" not in b.text for s in sections for b in s.blocks)


def test_chunks_respect_cap_and_keep_pages():
    B = parse.Block
    long_para = ("Retrieval augmented generation improves factuality on open-domain QA benchmarks. " * 60)
    sec = parse.Section("3 Method", [B("Short intro paragraph about the method and its parts.", 3),
                                    B(long_para, 3), B("Final paragraph with results on page four. " * 10, 4)])
    chunks = parse.chunk_sections("2310.00001", [sec])
    assert len(chunks) > 3
    assert all(len(c["text"]) <= parse.MAX_CHARS for c in chunks)
    assert all(c["est_tokens"] < 512 for c in chunks)
    assert chunks[-1]["page_end"] == 4
    assert [c["chunk_id"] for c in chunks] == [f"2310.00001#{i:04d}" for i in range(len(chunks))]


def test_unbreakable_text_is_still_capped():
    sec = parse.Section("Table", [parse.Block("x1 " * 2000, 2)])
    chunks = parse.chunk_sections("p", [sec])
    assert chunks and all(len(c["text"]) <= parse.MAX_CHARS for c in chunks)


def test_reference_matching_by_title_and_id():
    corpus = {"2310.11511": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
              "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
              "2212.10509": "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"}
    refs = ("Asai et al. 2023. Self-RAG: Learning to retrieve, generate, and cri- tique through self-reflection. "
            "Trivedi et al. arXiv preprint arXiv:2212.10509.")
    found = parse.reference_matches(refs, corpus, self_id="9999.0000")
    assert set(found) == {"2310.11511", "2212.10509"}
    assert parse.reference_matches(refs, corpus, self_id="2310.11511") == ["2212.10509"]


# -- retrieval -------------------------------------------------------------
def test_bm25_prefers_matching_doc():
    docs = [index.tokenize(t) for t in ["dense passage retrieval dual encoder", "graph based retrieval communities",
                                        "evaluation of rag faithfulness metrics"]]
    s = index.BM25(docs).scores(index.tokenize("faithfulness evaluation"))
    assert int(np.argmax(s)) == 2


def test_rrf_combines_rankings():
    fused = Retriever.rrf([np.array([0, 1, 2]), np.array([2, 0, 1])])
    assert fused[0][0] == 0            # ranks 1 and 2 beat ranks 3 and 1
    assert {i for i, _ in fused} == {0, 1, 2}


# -- answers ---------------------------------------------------------------
def _hit(i, pid="2310.11511"):
    return Hit(f"{pid}#{i:04d}", pid, "Self-RAG", 2023, "3 Method", 4, 4, "text " * 50, "https://arxiv.org/abs/" + pid, 0.1)


def test_audit_flags_invalid_citations():
    pack = answer.build_pack(Result("q", "hybrid", [_hit(1), _hit(2)]))
    ans = {"claims": [{"text": "ok", "evidence": ["E1"]}, {"text": "bad", "evidence": ["E9"]},
                      {"text": "none", "evidence": []}]}
    a = answer.audit(ans, pack)
    assert [c["status"] for c in a["claims"]] == ["cited", "unsupported", "unsupported"]
    assert a["claims"][0]["chunk_ids"] == ["2310.11511#0001"]
    assert a["claims"][1]["invalid_citations"] == ["E9"]
    assert a["citation_validity"] == 0.5


def test_pack_respects_budget():
    hits = [_hit(i) for i in range(20)]
    pack = answer.build_pack(Result("q", "hybrid", hits))
    assert len(pack) <= answer.MAX_PASSAGES
    assert sum(parse.est_tokens(p["hit"].text) + 40 for p in pack) <= answer.EVIDENCE_TOKEN_BUDGET


def test_parse_json_tolerates_fences():
    assert answer._parse_json('```json\n{"answerable": true}\n```') == {"answerable": True}
    assert answer._parse_json('Sure:\n{"a": 1}') == {"a": 1}


# -- citation graph traversal ------------------------------------------------
def _toy_retriever():
    """Papers A-B-C chain plus isolated D; two chunks per paper; no index or API needed."""
    R = object.__new__(Retriever)
    R.paper_of = ["A", "A", "B", "B", "C", "C", "D", "D"]
    R.papers = ["A", "B", "C", "D"]
    R.pidx = {p: i for i, p in enumerate(R.papers)}
    R.neighbours = {"A": {"B"}, "B": {"A", "C"}, "C": {"B"}}
    A = np.zeros((4, 4))
    for a, b in (("A", "B"), ("B", "C")):
        A[R.pidx[a], R.pidx[b]] = A[R.pidx[b], R.pidx[a]] = 1
    deg = A.sum(1, keepdims=True)
    R.P = np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)
    R.title_abs = {}
    return R


def test_one_hop_adds_only_direct_neighbours():
    R = _toy_retriever()
    dense = np.arange(8)
    assert R.expand_1hop({"A": 1.0}, dense) == ["B"]
    assert set(R.expand_1hop({"B": 1.0}, dense)) == {"A", "C"}


def test_ppr_reaches_two_hops_but_not_disconnected_papers():
    R = _toy_retriever()
    got = R.expand_ppr({"A": 1.0}, np.arange(8))
    assert got[0] == "B" and "C" in got and "D" not in got


def test_ppr_skips_papers_with_no_topical_passage():
    R = _toy_retriever()
    dense_top = np.array([0, 1, 2, 3])        # only A and B chunks are topical
    assert R.expand_ppr({"A": 1.0}, dense_top) == ["B"]


def test_seed_weights_use_rank_not_raw_scores():
    R = _toy_retriever()
    seeds = R._seed_papers([(0, -9.0), (2, -9.5), (1, -10.0)], n=2)
    assert list(seeds) == ["A", "B"] and seeds["A"] > seeds["B"] > 0


def test_abstain_guard():
    g = answer.abstain_guard
    assert g({"answerable": True, "short_answer": "The evidence does not specify the cost."})["answerable"] is False
    assert g({"answerable": True, "evidence_check": "none", "short_answer": "X."})["answerable"] is False
    assert g({"answerable": True, "evidence_check": "E2", "short_answer": "Self-RAG uses reflection tokens [E2]."})["answerable"] is True
    assert g({"answerable": True, "short_answer": "It retrieves adaptively."})["answerable"] is True   # missing field: leave alone


# -- failure fixes (answer-v3) ------------------------------------------------
from researchgraph.retrieve import extract_entities  # noqa: E402


def test_extract_entities():
    assert extract_entities("How do RAGAS and ARES differ?") == ["RAGAS", "ARES"]
    assert extract_entities("How does HippoRAG 2 build on HippoRAG?") == ["HippoRAG 2", "HippoRAG"]
    assert extract_entities("Compare HyDE and Query2doc") == ["HyDE", "Query2doc"]
    assert extract_entities("Is RAG better than LLMs for QA in 2024?") == []      # generic acronyms, years
    # PII placeholders from guard.scrub_pii are not method names
    assert extract_entities("My email is [EMAIL] and my phone is [PHONE]. Which papers evaluate RAGAS?") == ["RAGAS"]


def _pack(papers):
    return answer.build_pack(Result("q", "rerank", [_hit(i, pid) for i, pid in enumerate(papers)]))


def test_comparability_guard_needs_one_paper():
    pack = _pack(["2310.11511", "2401.15884"])
    ans = {"comparisons": [{"what": "A vs B", "directly_comparable": True, "evidence": ["E1", "E2"]},
                           {"what": "same table", "directly_comparable": True, "evidence": ["E1"]},
                           {"what": "uncited", "directly_comparable": True},
                           {"what": "already no", "directly_comparable": False, "evidence": ["E1", "E2"]}]}
    out = [c["directly_comparable"] for c in answer.comparability_guard(ans, pack)["comparisons"]]
    assert out == [False, True, False, False]
    assert "2 different papers" in ans["comparisons"][0]["reason"]


def test_survey_passages_are_marked():
    h = _hit(1)
    h.scope_tag = "survey"
    pack = answer.build_pack(Result("q", "rerank", [h, _hit(2, "2401.15884")]))
    text = answer.format_pack(pack)
    assert text.count("SURVEY (secondary source") == 1
    assert "SURVEY" not in answer.format_pack(pack, mark_surveys=False)


def test_decomposition_gives_each_entity_evidence():
    R = _toy_retriever()
    hit = lambda cid, pid: Hit(cid, pid, pid, 2024, "s", 1, 1, "t", "u", 1.0)
    canned = {"main": [hit("A#1", "A"), hit("A#2", "A"), hit("A#3", "A")],
              "AlphaRAG": [hit("A#1", "A"), hit("A#4", "A")], "BetaRAG": [hit("B#1", "B"), hit("B#2", "B")]}

    def fake_search(query, mode="rerank", k=10, graph="auto"):
        key = "AlphaRAG" if query.startswith("AlphaRAG ") else "BetaRAG" if query.startswith("BetaRAG ") else "main"
        return Result(query, mode, list(canned[key])[:k])
    R.search = fake_search
    res = R.search_decomposed("Compare AlphaRAG and BetaRAG on recall", k=5)
    assert res.entities == ["AlphaRAG", "BetaRAG"]
    assert [h.chunk_id for h in res.hits][:2] == ["A#1", "B#1"]          # every entity represented up front
    assert {h.paper_id for h in res.hits} == {"A", "B"}
    assert sum(h.paper_id == "A" for h in res.hits) <= 3                 # per-paper cap still holds


def test_entity_owner_is_the_introducing_paper():
    R = _toy_retriever()
    R.title_abs = {"A": "Precise Zero-Shot Dense Retrieval \n we propose Hypothetical Document Embeddings (HyDE).",
                   "B": "A Survey of RAG \n we review many methods.", "C": "HyDE-like tricks", "D": "Other"}
    assert R.entity_owners("HyDE") == ["A"]              # "HyDE-like" is not a whole-word match
    assert R.entity_owners("RAG") == ["B"]
    R.title_abs = {p: "RAG everywhere" for p in "ABC"}
    assert R.entity_owners("RAG") == []                  # too generic: more than 2 owners


def test_entity_guard_needs_every_named_thing_in_evidence():
    pack = _pack(["2310.11511"])                      # passages: "Self-RAG" title, text "text text ..."
    a = answer.entity_guard({"answerable": True}, "What retriever does Google Gemini use with Self-RAG?", pack)
    assert a["answerable"] is False and a["entity_guard"] == ["Google", "Gemini"]
    b = answer.entity_guard({"answerable": True}, "How does Self-RAG decide when to retrieve?", pack)
    assert b["answerable"] is True and "entity_guard" not in b


def test_client_retries_dropped_connections_but_not_cert_errors():
    import io, ssl, urllib.error
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api_client import ApiError, Endpoint, RateLimiter

    class Fake(Endpoint):
        def __init__(self, script):
            self.base_url, self.api_key, self.name, self.lock = "https://fake/v1", "k", "fake", None
            self.retry_delays, self.slept, self.script = (2, 4, 8), [], list(script)
            self.sleep, self.limiter = self.slept.append, RateLimiter(sleep=lambda s: None)

        def _open(self, route, body=None, stream=False):
            x = self.script.pop(0)
            if isinstance(x, Exception):
                raise x
            return io.BytesIO(b'{"ok": true}')

    f = Fake([ConnectionResetError(54, "reset"), urllib.error.URLError(TimeoutError("timed out")), "ok"])
    assert f.request("/x") == {"ok": True} and f.slept == [2, 4]
    g = Fake([urllib.error.URLError(ssl.SSLCertVerificationError("bad cert"))])
    try:
        g.request("/x"); assert False
    except ApiError as e:
        assert e.status == 0 and g.slept == []


def test_abstain_guard_catches_adverb_phrasings():
    g = answer.abstain_guard
    for text in ("The passage does not directly answer which method is cheapest.",
                 "The evidence does not directly state the cost.", "This is not clearly reported in the passages."):
        assert g({"answerable": True, "short_answer": text})["answerable"] is False, text
    assert g({"answerable": True, "short_answer": "ARES directly compares against RAGAS on six datasets."})["answerable"] is True


def test_followups_are_cleaned_and_never_block(monkeypatch):
    class FakeClient:
        def __init__(self, content=None, error=None):
            self.content, self.error = content, error

        def chat(self, messages, **kw):
            if self.error:
                raise self.error
            return {"choices": [{"message": {"content": self.content}}]}

    ans = {"question": "How does Self-RAG retrieve?", "short_answer": "It predicts tokens.", "answerable": True,
           "evidence": [{"title": "Self-RAG"}, {"title": "Self-RAG"}, {"title": "CRAG"}]}
    reply = '```json\n{"followups": ["How does Self-RAG compare with CRAG", "how does self-rag retrieve?", ' \
            '"How does Self-RAG compare with CRAG", "What are Self-RAG\'s limits?", "Extra one?"]}\n```'
    monkeypatch.setattr(answer, "client", lambda: FakeClient(reply))
    out = answer.suggest_followups(ans)
    assert out == ["How does Self-RAG compare with CRAG?", "What are Self-RAG's limits?", "Extra one?"]
    assert ans["followups"] == out                            # repeat of the question and duplicates dropped, max 3

    monkeypatch.setattr(answer, "client", lambda: FakeClient(error=answer.ApiError(429, "busy", "/chat")))
    assert answer.suggest_followups(dict(ans)) == []          # API error -> no suggestions, no exception
    monkeypatch.setattr(answer, "client", lambda: FakeClient("not json"))
    assert answer.suggest_followups(dict(ans)) == []


def test_stage_metrics():
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_stages", os.path.join(ROOT, "scripts", "eval_stages.py"))
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    assert E.token_f1("The cat sat", "cat sat down") == 0.8
    assert E.rouge("a b c d", "a b c d")["rougeL"]["f"] == 1.0
    hits = [Hit(f"P#{i}", "P" if i == 2 else "Q", "t", 2024, "s", 1, 1, "x", "u", 1.0) for i in range(5)]
    assert E.ndcg10(hits, {"P#2"}, {"P"}) == pytest.approx(1 / math.log2(4))
    assert E.ndcg10(hits, set(), {"P"}) == pytest.approx(1 / math.log2(4))


# -- agent: planner -> executor -> generator -> critic ---------------------------------------------
def _agent_setup(monkeypatch, planner, critic):
    from researchgraph import agent
    R = _toy_retriever()
    hit = lambda cid, pid, s=1.0: Hit(cid, pid, f"Paper {pid}", 2024, "s", 1, 1, f"text about {pid} retrieval", "u", s)
    canned = {"alpha": [hit("A#1", "A"), hit("A#2", "A")], "beta": [hit("B#1", "B")], "gap": [hit("C#1", "C")]}
    searched = []

    def fake_search(q, mode="rerank", k=10, graph="auto"):
        searched.append(q)
        return Result(q, mode, list(next((v for key, v in canned.items() if key in q.lower()), canned["alpha"])))
    R.search_decomposed = fake_search
    replies = {"plan": list(planner), "critic": list(critic)}

    class FakeClient:
        llm = type("M", (), {"model": "fake"})()

        def chat(self, messages, **kw):
            sysmsg = messages[0]["content"]
            kind = "plan" if sysmsg == agent.PLANNER_SYSTEM else "critic" if sysmsg == agent.CRITIC_SYSTEM else "answer"
            content = replies[kind].pop(0) if kind != "answer" else \
                '{"answerable": true, "short_answer": "Alpha retrieves.", "claims": [{"text": "Alpha retrieves", "evidence": ["E1"]}]}'
            return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 1000}}
    monkeypatch.setattr(agent, "client", lambda: FakeClient())
    monkeypatch.setattr(answer, "client", lambda: FakeClient())
    monkeypatch.setattr(agent, "history", lambda sid, n=3: [])
    return agent, R, searched


def test_agent_happy_path(monkeypatch):
    plan = '{"standalone_question": "How does alpha compare with beta?", "plan": [' \
           '{"tool": "search_papers", "args": {"query": "alpha"}}, {"tool": "search_papers", "args": {"query": "beta"}}]}'
    agent, R, searched = _agent_setup(monkeypatch, [plan], ['{"verdict": "accept"}'])
    ans = agent.Agent(R, session_id="t", log=False).run("How does it compare with beta?")
    a = ans["agent"]
    assert a["standalone_question"] == "How does alpha compare with beta?"
    assert (a["n_tool_calls"], a["n_llm_calls"], a["rounds"], a["stop_reason"]) == (2, 3, 1, "accepted")
    assert a["completed"] and ans["mode"] == "agent" and ans["citation_validity"] == 1.0
    assert [e["chunk_id"] for e in ans["evidence"]][:2] == ["A#1", "B#1"]      # every tool call represented first


def test_agent_survives_bad_json_unknown_tools_and_loops(monkeypatch):
    agent, R, searched = _agent_setup(monkeypatch, ["not json", "still not json"], ['{"verdict": "accept"}'])
    ans = agent.Agent(R, session_id="t", log=False).run("alpha?")
    a = ans["agent"]
    assert a["json_errors"] == 2 and any("planner" in f for f in a["fallbacks"]) and a["completed"]
    assert a["tool_calls"][0]["args"] == {"query": "alpha?"}                  # fallback plan: one plain search

    plan = '{"plan": [{"tool": "delete_files", "args": {}}, {"tool": "search_papers", "args": {"query": "alpha"}},' \
           '{"tool": "search_papers", "args": {"query": "alpha"}}, {"tool": "read_paper", "args": {"paper": "no such thing"}}]}'
    agent, R, searched = _agent_setup(monkeypatch, [plan], ['{"verdict": "accept"}'])
    a = agent.Agent(R, session_id="t", log=False).run("alpha?")["agent"]
    assert [c["status"] for c in a["tool_calls"]] == ["error", "ok", "skipped", "error"]
    assert a["tool_errors"] == 2 and a["n_tool_calls"] == 3 and a["completed"]


def test_agent_critic_retry_then_stops(monkeypatch):
    plan = '{"plan": [{"tool": "search_papers", "args": {"query": "alpha"}}]}'
    critic = '{"verdict": "retry", "problems": ["no evidence on the gap"], "new_queries": ["gap"]}'
    agent, R, searched = _agent_setup(monkeypatch, [plan], [critic])
    a = agent.Agent(R, session_id="t", log=False, max_rounds=2).run("alpha and the gap?")["agent"]
    assert searched == ["alpha", "gap"] and a["rounds"] == 2 and a["stop_reason"] == "max rounds"
    assert a["n_llm_calls"] == 4                                              # plan, answer, critic, answer

    agent, R, searched = _agent_setup(monkeypatch, [plan], [critic])
    ans = agent.Agent(R, session_id="t", log=False, token_budget=1000).run("alpha?")
    assert ans["agent"]["stop_reason"] == "token budget" and not ans["agent"]["completed"]


def test_agent_pool_round_robin():
    from researchgraph import agent
    h = lambda cid, pid, s: Hit(cid, pid, pid, 2024, "s", 1, 1, "t", "u", s)
    calls = [[h("A#1", "A", 5), h("A#2", "A", 4), h("A#3", "A", 3), h("A#4", "A", 2)], [h("B#1", "B", -3), h("B#2", "B", -4)]]
    got = [x.chunk_id for x in agent.pool(calls)]
    assert got[:4] == ["A#1", "B#1", "A#2", "B#2"] and got.count("A#4") == 0   # per-paper cap 3


# -- serving guards: rate limits, cache, PII ---------------------------------------------------------
def test_rate_limiter_sliding_windows():
    from researchgraph import guard
    now = [0.0]
    lim = guard.RateLimiter(((2, 10), (3, 100)), clock=lambda: now[0])
    assert lim.check("u")[0] and lim.check("u")[0]
    ok, wait = lim.check("u")
    assert not ok and wait == 10                     # 2 per 10 s
    assert lim.check("other")[0]                     # keys are independent
    now[0] = 11
    assert lim.check("u")[0]                         # window slid
    now[0] = 25
    ok, wait = lim.check("u")
    assert not ok and wait == 75                     # 3 per 100 s


def test_answer_cache_roundtrip(tmp_path, monkeypatch):
    from researchgraph import db, guard
    path = str(tmp_path / "t.db")
    real = db.connect
    monkeypatch.setattr(guard.db, "connect", lambda *a: real(path))
    k1 = guard.cache_key("How does Self-RAG work?", mode="rerank", k=10)
    assert k1 == guard.cache_key("  how does self-rag work ", mode="rerank", k=10)   # normalised
    assert k1 != guard.cache_key("How does Self-RAG work?", mode="bm25", k=10)
    assert guard.cache_get(k1) is None
    guard.cache_put(k1, "q", {"mode": "rerank"}, {"short_answer": "x"})
    assert guard.cache_get(k1) == {"short_answer": "x"}
    assert guard.cache_get(k1, ttl=-1) is None                                       # expired
    k2 = guard.cache_key("err")
    guard.cache_put(k2, "q", {}, {"error": "API error 503"})
    assert guard.cache_get(k2) is None                                               # failures never cached


def test_pii_scrubbing_keeps_research_numbers():
    from researchgraph.guard import scrub_pii
    t, f = scrub_pii("I'm jane.doe@uni.edu, +91 98765 43210, card 4111 1111 1111 1111, PAN ABCDE1234F, host 10.0.0.12")
    assert f == ["CARD", "EMAIL", "IP", "PAN", "PHONE"] and "jane" not in t and "4111" not in t
    for safe in ["What does arXiv 2310.11511 report?", "EM 45.6 vs 38.2 at k=5 on 2,000 docs in 2024 (Table 3)",
                 "512 768 1024 2048 dimensions", "1234 5678 9012 3456"]:
        assert scrub_pii(safe) == (safe, []), safe
