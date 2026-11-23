"""Chunking experiment: small / base / large chunks, retrieval only (no LLM calls; embeddings API only).

Usage: .venv/bin/python scripts/chunking_experiment.py      (first run embeds the new chunks, ~30 min)

Every variant re-chunks the same parsed PDFs, builds its own BM25 + dense index (bge-small-en-v1.5) and is
searched with B2 (hybrid) and B3 (hybrid + cross-encoder rerank). Gold passages belong to the base
chunking, so passage hits are measured chunk-independently: a gold passage counts as found if one
retrieved chunk contains at least half of it (ROUGE-L recall >= 0.5).

Bigger chunks hold more text, so "@10" metrics favour them. The fair comparison is "@budget": the hits
that fit into the generator's 3,000-token evidence budget (answer.EVIDENCE_TOKEN_BUDGET).
Output: reports/chunking_results.json and reports/CHUNKING.md. The embeddings cache is kept in
data/index/chunking_cache.npz (gitignored with data/).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from researchgraph import DATA, INDEX_DIR, ROOT, answer, db, index, parse  # noqa: E402
from researchgraph.retrieve import Retriever  # noqa: E402
from eval_stages import load_set, rouge, summ  # noqa: E402

VARIANTS = {   # name: (TARGET_TOKENS, OVERLAP_TOKENS, MAX_CHARS)
    "small": (150, 25, 800),
    "base": (300, 50, 1500),       # the production setting
    "large": (450, 75, 1900),      # stays under the embedding service's 512-token limit for most text
}
CACHE = os.path.join(INDEX_DIR, "chunking_cache.npz")
OUT_JSON = os.path.join(ROOT, "reports", "chunking_results.json")
OUT_MD = os.path.join(ROOT, "reports", "CHUNKING.md")
BUDGET = answer.EVIDENCE_TOKEN_BUDGET


def parsed_sections():
    con = db.connect()
    papers = con.execute("SELECT paper_id, pdf_path, title, year, url, version, scope_tag FROM papers "
                         "WHERE pdf_path IS NOT NULL AND paper_id IN (SELECT DISTINCT paper_id FROM chunks)").fetchall()
    out = []
    for p in papers:
        blocks, _ = parse.extract_blocks(os.path.join(DATA, p["pdf_path"]))
        sections, _ = parse.split_sections(blocks)
        out.append((dict(p), sections))
    return out


def make_rows(parsed, target, overlap, max_chars):
    old = parse.TARGET_TOKENS, parse.OVERLAP_TOKENS, parse.MAX_CHARS
    parse.TARGET_TOKENS, parse.OVERLAP_TOKENS, parse.MAX_CHARS = target, overlap, max_chars
    try:
        rows = []
        for p, sections in parsed:
            for c in parse.chunk_sections(p["paper_id"], sections):
                rows.append({**c, "title": p["title"], "year": p["year"], "url": p["url"], "version": p["version"],
                             "scope_tag": p["scope_tag"]})
    finally:
        parse.TARGET_TOKENS, parse.OVERLAP_TOKENS, parse.MAX_CHARS = old
    return rows


def embed_rows(rows, cache):
    todo = [r for r in rows if r["text_sha256"] not in cache]
    t0 = time.time()
    try:
        for i in range(0, len(todo), 16):
            batch = todo[i:i + 16]
            vecs, _ = index._embed_batch([index.doc_text(r) for r in batch])
            for r, v in zip(batch, vecs):
                cache[r["text_sha256"]] = np.asarray(v, dtype=np.float32)
            if (i // 16) % 50 == 0:
                print(f"    embedded {i + len(batch)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
                np.savez(CACHE, **cache)
    finally:
        np.savez(CACHE, **cache)
    mat = np.stack([cache[r["text_sha256"]] for r in rows])
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def retriever_for(base: Retriever, rows, mat) -> Retriever:
    R = object.__new__(Retriever)
    R.__dict__.update(base.__dict__)
    R.rows, R.vectors = rows, mat
    R.bm25 = index.BM25([index.tokenize(index.doc_text(r)) for r in rows])
    R.paper_of = [r["paper_id"] for r in rows]
    R.paper_of_arr = np.array(R.paper_of)
    return R


def within_budget(hits):
    out, used = [], 0
    for h in hits:
        t = parse.est_tokens(h.text) + 40
        if used + t > BUDGET:
            break
        out.append(h)
        used += t
    return out


def score(hits, q, gold_texts):
    gp, ref = set(q["gold_papers"]), q.get("reference_answer")
    row = {}
    for tag, hs in (("10", hits[:10]), ("budget", within_budget(hits))):
        row[f"n_{tag}"] = len(hs)
        if gp:
            row[f"paper_recall@{tag}"] = len(gp & {h.paper_id for h in hs}) / len(gp)
        if gold_texts:
            row[f"passage_found@{tag}"] = float(any(rouge(g, h.text)["rougeL"]["r"] >= 0.5 for g in gold_texts for h in hs))
        if ref:
            row[f"ref_rougeL_recall@{tag}"] = rouge(ref, " ".join(h.text for h in hs))["rougeL"]["r"]
    return row


def main():
    t0 = time.time()
    base = Retriever()
    con = db.connect()
    chunk_text = {r[0]: r[1] for r in con.execute("SELECT chunk_id, text FROM chunks")}
    print("parsing PDFs…", flush=True)
    parsed = parsed_sections()
    cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    main_cache = os.path.join(INDEX_DIR, "embedding_cache.npz")
    if os.path.exists(main_cache):
        cache.update({k: v for k, v in np.load(main_cache).items() if k not in cache})
    sets = {name: [q for q in load_set(name) if q["gold_papers"]] for name in ("curated", "synthetic_test")}
    rep = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "budget_tokens": BUDGET, "variants": {}}
    for name, (target, overlap, max_chars) in VARIANTS.items():
        rows = make_rows(parsed, target, overlap, max_chars)
        toks = np.array([r["est_tokens"] for r in rows])
        print(f"\n== {name}: {len(rows)} chunks, tokens p50 {int(np.median(toks))}", flush=True)
        mat = embed_rows(rows, cache)
        R = retriever_for(base, rows, mat)
        v = {"target_tokens": target, "overlap_tokens": overlap, "max_chars": max_chars, "chunks": len(rows),
             "tokens_p50": int(np.median(toks)), "tokens_p95": int(np.percentile(toks, 95)), "sets": {}}
        for sname, qs in sets.items():
            res = {"B2 Hybrid": [], "B3 Hybrid+rerank": []}
            for q in qs:
                gold = [chunk_text[c] for c in (q.get("gold_chunks") or []) if c in chunk_text]
                res["B2 Hybrid"].append(score(R.search(q["question"], mode="hybrid", k=15).hits, q, gold))
                res["B3 Hybrid+rerank"].append(score(R.search(q["question"], mode="rerank", k=15).hits, q, gold))
            keys = sorted({k for rs in res.values() for r in rs for k in r})
            v["sets"][sname] = {sys_: {k: summ([r.get(k) for r in rs]) for k in keys} for sys_, rs in res.items()}
            b3 = v["sets"][sname]["B3 Hybrid+rerank"]
            print(f"  {sname}: B3 passage@budget {b3.get('passage_found@budget', {}).get('mean')} "
                  f"paper@budget {b3.get('paper_recall@budget', {}).get('mean')} "
                  f"ref ROUGE-L@budget {b3.get('ref_rougeL_recall@budget', {}).get('mean')}", flush=True)
        rep["variants"][name] = v
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(rep, open(OUT_JSON, "w"), indent=1)
    write_md(rep)
    print("Saved", os.path.relpath(OUT_JSON, ROOT), "and", os.path.relpath(OUT_MD, ROOT))


def _f(m):
    if not m or m.get("mean") is None:
        return "–"
    lo, hi = m["ci95"]
    return f"{m['mean']:.2f} [{lo:.2f}, {hi:.2f}]" if lo is not None else f"{m['mean']:.2f}"


def write_md(rep):
    V = rep["variants"]
    lines = [f"# Chunking experiment ({rep['run_at']})", "",
             "Same parsed PDFs, three chunk sizes, each with its own BM25 + bge-small index; retrieval only. "
             f"'@budget' = the hits that fit the generator's {rep['budget_tokens']:,}-token evidence budget, which is "
             "the fair comparison (bigger chunks hold more text in a top 10). A gold passage counts as found if one "
             "retrieved chunk contains at least half of it. Mean [95% bootstrap CI].", "",
             "| Variant | Target / overlap tokens | Chunks | Tokens p50 / p95 |", "|---|---|---|---|"]
    for n, v in V.items():
        lines.append(f"| {n}{' (production)' if n == 'base' else ''} | {v['target_tokens']} / {v['overlap_tokens']} | "
                     f"{v['chunks']:,} | {v['tokens_p50']} / {v['tokens_p95']} |")
    for sname in ("curated", "synthetic_test"):
        for sys_ in ("B3 Hybrid+rerank", "B2 Hybrid"):
            lines += ["", f"**{sname}, {sys_}**", "",
                      "| Variant | Passages in budget | Gold passage found @budget | Paper recall @budget | Reference ROUGE-L recall @budget | Gold passage found @10 |",
                      "|---|---|---|---|---|---|"]
            for n, v in V.items():
                m = v["sets"][sname][sys_]
                lines.append(f"| {n} | {_f(m.get('n_budget'))} | {_f(m.get('passage_found@budget'))} | "
                             f"{_f(m.get('paper_recall@budget'))} | {_f(m.get('ref_rougeL_recall@budget'))} | "
                             f"{_f(m.get('passage_found@10'))} |")
    open(OUT_MD, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
