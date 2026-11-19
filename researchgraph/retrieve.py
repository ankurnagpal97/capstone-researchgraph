"""Retrieval systems for the B0-B5 ablation.

  B0 bm25         keyword search only
  B1 dense        embedding search only (bge-small-en-v1.5)
  B2 hybrid       reciprocal rank fusion (RRF, k = 60) of BM25 + dense
  B3 rerank       B2's top 30, re-scored by a cross-encoder
  B4 graph1hop    B3 + one citation hop from the top papers
  B5 graph_ppr    B3 + personalized PageRank over the citation graph, seeded by the top papers

Graph systems (B4, B5) expand only when the question looks multi-hop/lineage (MULTIHOP rule),
unless graph="always". Expansion only nominates papers: their best passages are added to the
candidate pool and must out-score the others in the same reranking. A citation edge is never
treated as evidence or agreement.
All systems cap chunks per paper so one paper can't fill the evidence pack.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import db, index

SYSTEMS = {"bm25": "B0 BM25", "dense": "B1 Dense", "hybrid": "B2 Hybrid", "rerank": "B3 Hybrid+rerank",
           "graph1hop": "B4 +1-hop graph", "graph_ppr": "B5 +PageRank graph"}
MODES = tuple(SYSTEMS)
RRF_K = 60
POOL = 50                   # candidates from each first-stage ranker
RERANK_TOP = 30             # fused candidates sent to the cross-encoder
MAX_PER_PAPER = 3
GRAPH_SEED_PAPERS = 3
GRAPH_MAX_NEW_PAPERS = 5
GRAPH_CHUNKS_PER_PAPER = 3  # best chunks (by dense rank) contributed by each graph-nominated paper
PPR_DAMPING = 0.85
MULTIHOP = re.compile(r"\b(build[s]? (on|upon)|built on|extend[s]?|extension|follow[- ]?up|lineage|cite[sd]?|citing|"
                      r"earlier|later work|subsequent|inspired|predecessor|successor|improve[sd]? (on|over|upon)|"
                      r"compared? (to|with)|comparison|versus|vs\.?|both|relationship between|evolve[sd]?|"
                      r"original|prior work|which (papers|methods|approaches|works))\b", re.I)


@dataclass
class Hit:
    chunk_id: str
    paper_id: str
    title: str
    year: int
    section: str
    page_start: int
    page_end: int
    text: str
    url: str
    score: float
    via_graph: bool = False


@dataclass
class Result:
    query: str
    mode: str
    hits: list
    graph_used: bool = False
    graph_papers: list = field(default_factory=list)


class Retriever:
    def __init__(self):
        self.rows, self.vectors, self.bm25, self.meta = index.load()
        self.paper_of = [r["paper_id"] for r in self.rows]
        self.papers = sorted(set(self.paper_of))
        self.paper_of_arr = np.array(self.paper_of)
        self.pidx = {p: i for i, p in enumerate(self.papers)}
        con = db.connect()
        self.neighbours: dict[str, set] = {}
        n = len(self.papers)
        A = np.zeros((n, n), dtype=np.float64)
        for e in con.execute("SELECT DISTINCT citing_paper_id, cited_paper_id FROM citation_edges"):
            a, b = e[0], e[1]
            if a in self.pidx and b in self.pidx:
                self.neighbours.setdefault(a, set()).add(b)
                self.neighbours.setdefault(b, set()).add(a)
                A[self.pidx[a], self.pidx[b]] = A[self.pidx[b], self.pidx[a]] = 1.0   # undirected for traversal
        deg = A.sum(axis=1, keepdims=True)
        self.P = np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)            # row-stochastic
        self._qcache: dict[str, np.ndarray] = {}

    # -- first-stage rankers: chunk row indices, best first ------------------
    def rank_bm25(self, query: str) -> np.ndarray:
        return np.argsort(-self.bm25.scores(index.tokenize(query)), kind="stable")

    def query_vector(self, query: str) -> np.ndarray:
        if query not in self._qcache:
            from .llm import client
            v = np.asarray(client().embed([index.QUERY_PREFIX + query])[0], dtype=np.float32)
            self._qcache[query] = v / np.linalg.norm(v)
        return self._qcache[query]

    def rank_dense(self, query: str) -> np.ndarray:
        with np.errstate(all="ignore"):     # numpy 2.0 + Apple Accelerate emits spurious matmul warnings
            sims = self.vectors @ self.query_vector(query)
        return np.argsort(-sims, kind="stable")

    @staticmethod
    def rrf(rankings: list[np.ndarray], depth: int = POOL) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking[:depth]):
                scores[int(i)] = scores.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    # -- graph expansion -----------------------------------------------------
    def _seed_papers(self, scored: list[tuple[int, float]], n: int) -> dict[str, float]:
        """Top-n papers in `scored`, weighted by the reciprocal ranks of their chunks (scores may be negative logits)."""
        seeds: dict[str, float] = {}
        for pos, (i, _) in enumerate(scored):
            pid = self.paper_of[i]
            if pid not in seeds and len(seeds) >= n:
                continue
            seeds[pid] = seeds.get(pid, 0.0) + 1.0 / (pos + 1)
        return seeds

    def expand_1hop(self, seeds: dict[str, float], dense_rank: np.ndarray) -> list[str]:
        cands = {n for p in seeds for n in self.neighbours.get(p, ())} - set(seeds)
        return self._best_by_dense(cands, dense_rank)[:GRAPH_MAX_NEW_PAPERS]

    def expand_ppr(self, seeds: dict[str, float], dense_rank: np.ndarray, iters: int = 40) -> list[str]:
        """Personalized PageRank from the seed papers; returns the top non-seed papers that also
        have at least one passage in the dense top 300 (so off-topic hubs are not added)."""
        n = len(self.papers)
        r = np.zeros(n)
        for p, s in seeds.items():
            r[self.pidx[p]] = s
        r /= r.sum()
        x = r.copy()
        for _ in range(iters):
            x = (1 - PPR_DAMPING) * r + PPR_DAMPING * (self.P.T @ x)
        topical = {self.paper_of[i] for i in dense_rank[:300]}
        order = [self.papers[i] for i in np.argsort(-x) if x[i] > 0]
        return [p for p in order if p not in seeds and p in topical][:GRAPH_MAX_NEW_PAPERS]

    def _best_by_dense(self, papers: set, dense_rank: np.ndarray) -> list[str]:
        best = {}
        for pos, i in enumerate(dense_rank):
            p = self.paper_of[i]
            if p in papers and p not in best:
                best[p] = pos
        return sorted(best, key=best.get)

    def _graph_chunks(self, papers: list[str], dense_rank: np.ndarray) -> list[int]:
        out, count = [], {}
        for i in dense_rank:
            p = self.paper_of[i]
            if p in papers and count.get(p, 0) < GRAPH_CHUNKS_PER_PAPER:
                out.append(int(i)); count[p] = count.get(p, 0) + 1
        return out

    # -- assembly --------------------------------------------------------------
    def _hits(self, scored: list[tuple[int, float]], k: int, graph_papers=()) -> list[Hit]:
        per_paper, out = {}, []
        for i, s in scored:
            r = self.rows[i]
            if per_paper.get(r["paper_id"], 0) >= MAX_PER_PAPER:
                continue
            per_paper[r["paper_id"]] = per_paper.get(r["paper_id"], 0) + 1
            out.append(Hit(r["chunk_id"], r["paper_id"], r["title"], r["year"], r["section"], r["page_start"],
                           r["page_end"], r["text"], r["url"], round(float(s), 5), r["paper_id"] in graph_papers))
            if len(out) >= k:
                break
        return out

    def _rerank(self, query: str, idx: list[int]) -> list[tuple[int, float]]:
        from . import rerank
        s = rerank.scores(query, [index.doc_text(self.rows[i]) for i in idx])
        return sorted(zip(idx, s), key=lambda x: -x[1])

    def search(self, query: str, mode: str = "rerank", k: int = 10, graph: str = "auto") -> Result:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        as_scored = lambda ranking: [(int(i), 1.0 / (RRF_K + n + 1)) for n, i in enumerate(ranking[:POOL * 4])]
        if mode == "bm25":
            return Result(query, mode, self._hits(as_scored(self.rank_bm25(query)), k))
        rd = self.rank_dense(query)
        if mode == "dense":
            return Result(query, mode, self._hits(as_scored(rd), k))
        fused = self.rrf([self.rank_bm25(query), rd])
        if mode == "hybrid":
            return Result(query, mode, self._hits(fused, k))

        cand = [i for i, _ in fused[:RERANK_TOP]]
        use_graph = mode in ("graph1hop", "graph_ppr") and (graph == "always" or (graph == "auto" and MULTIHOP.search(query)))
        if not use_graph:
            return Result(query, mode, self._hits(self._rerank(query, cand), k))

        seeds = self._seed_papers(self._rerank(query, cand), GRAPH_SEED_PAPERS)
        new = self.expand_1hop(seeds, rd) if mode == "graph1hop" else self.expand_ppr(seeds, rd)
        extra = [i for i in self._graph_chunks(new, rd) if i not in set(cand)]
        scored = self._rerank(query, cand + extra)
        return Result(query, mode, self._hits(scored, k, graph_papers=set(new)), graph_used=True, graph_papers=new)
