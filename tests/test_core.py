"""Offline unit tests (no API calls). Run: .venv/bin/python -m pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import answer, index, parse  # noqa: E402
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
