"""Build and load the two indexes over the same chunk IDs.

Dense: bge-small-en-v1.5 vectors from the course embeddings service (384-d, L2-normalised),
       stored in data/index/vectors.npy. Vectors are cached by chunk-text hash, so rebuilding
       after a re-parse only embeds new or changed chunks.
Sparse: BM25 (k1=1.5, b=0.75), built in memory at load time (fast for a few thousand chunks).
Both index "<title> | <section>\n<text>" so a chunk carries its paper and section context.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter

import numpy as np

from . import INDEX_DIR, db
from .llm import ApiError, client

EMBED_MODEL = "bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge's query instruction
STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were "
           "with which what how does do can we our their these those than then there into over under "
           "between using use used based via".split())


def doc_text(row) -> str:
    return f"{row['title']} | {row['section']}\n{row['text']}"


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if t not in STOP and len(t) > 1]


def load_chunks(con=None):
    con = con or db.connect()
    return con.execute("""SELECT c.chunk_id, c.paper_id, c.section, c.page_start, c.page_end, c.text,
                                 c.text_sha256, p.title, p.year, p.url, p.version, p.scope_tag
                          FROM chunks c JOIN papers p USING (paper_id) ORDER BY c.chunk_id""").fetchall()


def _embed_one(text: str) -> tuple[list, bool]:
    """Embed one text; if the service says it is over 512 tokens, cut it in proportion and retry."""
    truncated = False
    for _ in range(6):
        try:
            return client().embed([text])[0], truncated
        except ApiError as e:
            m = re.search(r"Given: (\d+)", e.body)
            if e.status != 422 or not m:
                raise
            text = text[: int(len(text) * 500 / int(m.group(1)) * 0.95)]
            truncated = True
    raise RuntimeError("could not shorten text enough to embed")


def _embed_batch(texts: list[str]) -> tuple[list, list[bool]]:
    """Embed up to 16 texts; a text the service rejects as too long is embedded truncated."""
    try:
        return client().embed(texts), [False] * len(texts)
    except ApiError as e:
        if e.status != 422:
            raise
    pairs = [_embed_one(t) for t in texts]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def build(verbose: bool = True) -> dict:
    os.makedirs(INDEX_DIR, exist_ok=True)
    rows = load_chunks()
    cache_path = os.path.join(INDEX_DIR, "embedding_cache.npz")
    cache = dict(np.load(cache_path)) if os.path.exists(cache_path) else {}
    todo = [r for r in rows if r["text_sha256"] not in cache]
    if verbose:
        print(f"{len(rows)} chunks; {len(rows) - len(todo)} cached, embedding {len(todo)}…")
    t0, truncated_ids = time.time(), []
    try:
        for i in range(0, len(todo), 16):
            batch = todo[i:i + 16]
            vecs, trunc = _embed_batch([doc_text(r) for r in batch])
            truncated_ids += [r["chunk_id"] for r, t in zip(batch, trunc) if t]
            for r, v in zip(batch, vecs):
                cache[r["text_sha256"]] = np.asarray(v, dtype=np.float32)
            if verbose and (i // 16) % 20 == 0:
                print(f"  {i + len(batch)}/{len(todo)} ({time.time() - t0:.0f}s)")
    finally:
        np.savez(cache_path, **cache)        # keep progress even if a request fails
    n_trunc = len(truncated_ids)

    mat = np.stack([cache[r["text_sha256"]] for r in rows])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    np.save(os.path.join(INDEX_DIR, "vectors.npy"), mat)
    meta = {"embedding_model": EMBED_MODEL, "dims": int(mat.shape[1]), "n_chunks": len(rows),
            "chunk_ids": [r["chunk_id"] for r in rows], "truncated_for_embedding": truncated_ids,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(INDEX_DIR, "meta.json"), "w") as f:
        json.dump(meta, f)
    if verbose:
        print(f"Saved {mat.shape} vectors; {n_trunc} chunks embedded truncated.")
    return meta


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(self.tf):
            for t in d:
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def load():
    """Return (chunk rows, vector matrix, BM25 index), checking that the index matches the DB."""
    rows = load_chunks()
    with open(os.path.join(INDEX_DIR, "meta.json")) as f:
        meta = json.load(f)
    if meta["chunk_ids"] != [r["chunk_id"] for r in rows]:
        raise RuntimeError("Index is out of date with the database — run scripts/build_index.py")
    vectors = np.load(os.path.join(INDEX_DIR, "vectors.npy"))
    bm25 = BM25([tokenize(doc_text(r)) for r in rows])
    return rows, vectors, bm25, meta
