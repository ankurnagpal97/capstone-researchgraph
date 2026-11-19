"""External benchmark: QASPER (Dasigi et al., NAACL 2021; CC BY 4.0) test set, evidence retrieval.

QASPER asks questions about one NLP paper; annotators marked the paragraphs that answer each question.
We rank that paper's paragraphs with each of our first-stage systems and the reranker, and measure
whether a gold evidence paragraph is at the top:
  Hit@1 / Hit@3 / Hit@5 (any gold paragraph in the top k), Evidence Recall@5, MRR.
Scope note: QASPER tests WITHIN-paper evidence selection on NLP papers. It does not measure
cross-paper comparison or the citation graph (B4/B5), and its papers are not our RAG corpus.

Questions are kept when at least one annotator found the answer in text paragraphs (figure/table-only
evidence and unanswerable questions are excluded). Papers are sampled with a fixed seed.

Usage: .venv/bin/python scripts/eval_qasper.py [--papers 60]
Data:  data/raw/qasper/qasper-test-v0.3.json (download: see README)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import DATA, INDEX_DIR, ROOT, index, rerank  # noqa: E402
from researchgraph.llm import client  # noqa: E402

PATH = os.path.join(DATA, "raw", "qasper", "qasper-test-v0.3.json")
RERANK_TOP = 20


def load(n_papers, seed):
    data = json.load(open(PATH))
    items = []
    for pid, p in data.items():
        paras = []
        for sec in p.get("full_text") or []:
            for para in sec.get("paragraphs") or []:
                if para and para.strip():
                    paras.append({"section": sec.get("section_name") or "", "text": para.strip()})
        if len(paras) < 5:
            continue
        texts = {x["text"] for x in paras}
        qs = []
        for qa in p.get("qas") or []:
            gold = set()
            for a in qa.get("answers") or []:
                ans = a["answer"]
                if ans.get("unanswerable"):
                    continue
                gold |= {e.strip() for e in ans.get("evidence") or [] if e.strip() in texts}
            if gold:
                qs.append({"question": qa["question"], "gold": gold})
        if qs:
            items.append({"paper_id": pid, "title": p["title"], "paras": paras, "qs": qs})
    random.Random(seed).shuffle(items)
    return items[:n_papers]


def embed_cached(texts, cache):
    todo = [t for t in texts if hashlib.sha256(t.encode()).hexdigest() not in cache]
    for i in range(0, len(todo), 16):
        batch = todo[i:i + 16]
        for t, v in zip(batch, index._embed_batch(batch)[0]):
            cache[hashlib.sha256(t.encode()).hexdigest()] = np.asarray(v, dtype=np.float32)
    m = np.stack([cache[hashlib.sha256(t.encode()).hexdigest()] for t in texts])
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def metrics(order, gold_idx):
    pos = [order.index(g) for g in gold_idx if g in order]
    first = min(pos) if pos else None
    return {"hit1": float(first is not None and first < 1), "hit3": float(first is not None and first < 3),
            "hit5": float(first is not None and first < 5),
            "recall5": sum(1 for p in pos if p < 5) / len(gold_idx),
            "mrr": 1.0 / (first + 1) if first is not None else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not os.path.exists(PATH):
        sys.exit(f"Missing {PATH}. See README for the QASPER download.")
    items = load(args.papers, args.seed)
    nq = sum(len(it["qs"]) for it in items)
    print(f"{len(items)} papers, {nq} questions with text evidence")

    cache_path = os.path.join(INDEX_DIR, "qasper_embedding_cache.npz")
    os.makedirs(INDEX_DIR, exist_ok=True)
    cache = dict(np.load(cache_path)) if os.path.exists(cache_path) else {}
    systems = ["B0 BM25", "B1 Dense", "B2 Hybrid", "B3 Hybrid+rerank"]
    res = {s: [] for s in systems}
    t0 = time.time()
    try:
        for n, it in enumerate(items):
            docs = [f"{it['title']} | {p['section']}\n{p['text']}"[:1500] for p in it["paras"]]
            P = embed_cached(docs, cache)
            bm = index.BM25([index.tokenize(d) for d in docs])
            Q = embed_cached([index.QUERY_PREFIX + q["question"] for q in it["qs"]], cache)
            for q, qv in zip(it["qs"], Q):
                gold_idx = [i for i, p in enumerate(it["paras"]) if p["text"] in q["gold"]]
                rb = list(np.argsort(-bm.scores(index.tokenize(q["question"])), kind="stable"))
                rd = list(np.argsort(-(P @ qv), kind="stable"))
                fused = {}
                for ranking in (rb, rd):
                    for r, i in enumerate(ranking):
                        fused[i] = fused.get(i, 0) + 1 / (60 + r + 1)
                rh = sorted(fused, key=lambda i: -fused[i])
                top = rh[:RERANK_TOP]
                rs = rerank.scores(q["question"], [docs[i] for i in top])
                rr = [i for i, _ in sorted(zip(top, rs), key=lambda x: -x[1])] + rh[RERANK_TOP:]
                for s, order in zip(systems, (rb, rd, rh, rr)):
                    res[s].append(metrics([int(i) for i in order], gold_idx))
            if n % 10 == 0:
                print(f"  {n + 1}/{len(items)} papers ({time.time() - t0:.0f}s)")
    finally:
        np.savez(cache_path, **cache)

    summary = []
    for s in systems:
        rows = res[s]
        summary.append({"system": s, **{k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}})
    print(f"\nQASPER test, evidence retrieval within paper (n = {nq} questions, {len(items)} papers)")
    for r in summary:
        print(f"  {r['system']:18} Hit@1 {r['hit1']:.3f}  Hit@3 {r['hit3']:.3f}  Hit@5 {r['hit5']:.3f}  "
              f"Recall@5 {r['recall5']:.3f}  MRR {r['mrr']:.3f}")
    out = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "dataset": "QASPER v0.3 test (CC BY 4.0)",
           "n_papers": len(items), "n_questions": nq, "seed": args.seed, "reranker": rerank.MODEL,
           "summary": summary}
    with open(os.path.join(ROOT, "reports", "qasper_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("Saved reports/qasper_results.json")


if __name__ == "__main__":
    main()
