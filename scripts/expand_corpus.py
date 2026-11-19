"""Grow the corpus by citation snowballing (one hop through OpenAlex), instead of hand-picking IDs.

For every corpus paper, collect (a) the works it references and (b) the works that cite it.
A candidate's score = number of corpus papers it is connected to. Keep candidates that
  - have an arXiv version (needed for full text),
  - match the scope terms in their title,
  - are from 2022 or later, or are older but connected to >= FOUNDATION_MIN corpus papers (tagged foundation),
and take the top N by score. Writes data/seed_papers_expansion.tsv with the reason for each pick;
ingest.py then verifies titles against arXiv and records licenses as for the hand-picked seeds.

Usage: .venv/bin/python scripts/expand_corpus.py [--n 100]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import DATA, db, sources  # noqa: E402

SCOPE = re.compile(r"retriev|\brag\b|augment|multi-?hop|question answering|open-domain|dense|sparse|rerank|"
                   r"embedding|knowledge-intensive|grounded|grounding|hallucinat|attribut|citation|long-context|"
                   r"in-context|search|graph|passage|chain-of-thought|reasoning|benchmark|evaluat|faithful|fact", re.I)
OFF_TOPIC = re.compile(r"image|video|vision|visual|speech|audio|protein|molecul|medical|clinical|biomedical|"
                       r"recommend|code generation|robot|3d|diffusion", re.I)
FOUNDATION_MIN = 6
SELECT = "id,title,publication_year,ids,locations,cited_by_count"


def oa(url):
    """OpenAlex GET. Uses OPENALEX_API_KEY from .env if set (optional; register at openalex.org/settings/api).
    Without a key requests share a small free daily budget, and OpenAlex may answer 401/429 when it runs out."""
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        url += ("&" if "?" in url else "?") + "api_key=" + urllib.parse.quote(key)
    for attempt in range(4):
        time.sleep(0.2)
        try:
            return json.loads(sources._get(url, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 429) and attempt < 3:
                time.sleep(10 * (attempt + 1))
                continue
            raise


def arxiv_id_of(work) -> str | None:
    doi = (work.get("ids") or {}).get("doi") or ""
    m = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi, re.I)
    if m:
        return m.group(1)
    for loc in work.get("locations") or []:
        for key in ("landing_page_url", "pdf_url"):
            m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", loc.get(key) or "")
            if m:
                return m.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    from researchgraph.llm import client  # noqa: F401  (loads .env so OPENALEX_API_KEY is visible)
    import api_client; api_client.load_env()
    con = db.connect()
    corpus = {r["openalex_id"]: r["paper_id"] for r in con.execute(
        "SELECT paper_id, openalex_id FROM papers WHERE openalex_id IS NOT NULL")}
    have = {r[0] for r in con.execute("SELECT paper_id FROM papers")}
    print(f"{len(corpus)} corpus papers with OpenAlex IDs")

    links = defaultdict(set)          # candidate W id -> corpus papers it connects to
    info = {}
    refs_of = {}
    failed = []
    for i, (w, pid) in enumerate(corpus.items()):
        try:
            work = oa(f"https://api.openalex.org/works/{w}?select=referenced_works")
        except urllib.error.HTTPError as e:
            failed.append(pid); print(f"  skip {pid}: HTTP {e.code}"); continue
        refs_of[w] = [r.rsplit("/", 1)[-1] for r in work.get("referenced_works", [])]
        for r in refs_of[w]:
            links[r].add(pid)
        cursor = "*"
        for _ in range(1):                       # up to 200 citing works, most-cited first
            try:
                page = oa("https://api.openalex.org/works?" + urllib.parse.urlencode(
                {"filter": f"cites:{w}", "per-page": 200, "sort": "cited_by_count:desc", "select": SELECT,
                 "cursor": cursor}))
            except urllib.error.HTTPError as e:
                failed.append(pid); print(f"  citing works for {pid} skipped: HTTP {e.code}"); break
            for c in page["results"]:
                cid = c["id"].rsplit("/", 1)[-1]
                links[cid].add(pid)
                info[cid] = c
            cursor = page["meta"].get("next_cursor")
            if not cursor or len(page["results"]) < 200:
                break
        if i % 10 == 0:
            print(f"  {i + 1}/{len(corpus)} papers scanned, {len(links)} candidates")

    # details for referenced works we haven't seen
    missing = [w for w in links if w not in info and w not in corpus]
    for i in range(0, len(missing), 50):
        page = oa("https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": "openalex_id:" + "|".join(missing[i:i + 50]), "per-page": 50, "select": SELECT}))
        for c in page["results"]:
            info[c["id"].rsplit("/", 1)[-1]] = c

    picked, rejected = [], Counter()
    for w, conn in sorted(links.items(), key=lambda kv: -len(kv[1])):
        if w in corpus or w not in info:
            continue
        c = info[w]
        title, year, score = c.get("title") or "", c.get("publication_year") or 0, len(conn)
        aid = arxiv_id_of(c)
        if not aid:
            rejected["no arXiv version"] += 1; continue
        if aid in have or aid in {p[0] for p in picked}:
            continue
        if not SCOPE.search(title) or OFF_TOPIC.search(title):
            rejected["out of scope (title)"] += 1; continue
        if year < 2022 and score < FOUNDATION_MIN:
            rejected["pre-2022, weakly connected"] += 1; continue
        if score < 2:
            rejected["connected to only 1 corpus paper"] += 1; continue
        tag = "foundation" if year < 2022 else "snowball"
        words = re.sub(r"[^\w\s-]", " ", title).split()
        picked.append((aid, tag, " ".join(words[:3]), score, title, year))
        if len(picked) >= args.n:
            break

    out = os.path.join(DATA, "seed_papers_expansion.tsv")
    with open(out, "w") as f:
        f.write("# Citation-snowball expansion, generated by scripts/expand_corpus.py on "
                f"{time.strftime('%Y-%m-%d')}. Columns as seed_papers.tsv + connections + OpenAlex title.\n")
        for aid, tag, kw, score, title, year in picked:
            f.write(f"{aid}\t{tag}\t{kw}\t# {score} corpus links | {year} | {title}\n")
    print(f"\nPicked {len(picked)} papers -> {os.path.relpath(out)}")
    print("Skipped:", dict(rejected))
    if failed:
        print("OpenAlex lookups that failed (rerun later or add OPENALEX_API_KEY):", sorted(set(failed)))
    for p in picked[:15]:
        print(f"  {p[0]} {p[3]:3} links {p[5]} {p[4][:80]}")


if __name__ == "__main__":
    main()
